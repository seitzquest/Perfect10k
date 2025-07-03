"""
Interpretable Scoring System for Perfect10k
Scores candidates using discrete features with clear explanations.
"""

from dataclasses import dataclass
from enum import Enum

from feature_database import CellFeatures, FeatureType


class PreferenceCategory(Enum):
    """Categories of user preferences."""
    NATURE = "nature"
    WATER = "water"
    PARKS = "parks"
    QUIET = "quiet"
    SCENIC = "scenic"
    EXERCISE = "exercise"
    URBAN = "urban"


@dataclass
class ScoredCandidate:
    """A candidate location with its score and explanation."""
    node_id: int
    lat: float
    lon: float
    overall_score: float
    feature_scores: dict[FeatureType, float]
    explanation: str
    reasoning_details: list[str]


@dataclass
class PreferenceWeights:
    """Weights for different features based on user preferences."""
    close_to_forest: float = 0.0
    close_to_water: float = 0.0
    close_to_park: float = 0.0
    path_quality: float = 0.5  # Always somewhat important
    intersection_density: float = 0.2  # Base connectivity importance
    elevation_variety: float = 0.0


class InterpretableScorer:
    """
    Score candidates using discrete features with clear explanations.

    Translates natural language preferences into weighted feature combinations
    and provides clear reasoning for each score.
    """

    # Keywords that indicate different preference categories
    PREFERENCE_KEYWORDS = {
        PreferenceCategory.NATURE: [
            "nature", "forest", "tree", "green", "woods", "natural", "wildlife"
        ],
        PreferenceCategory.WATER: [
            "water", "river", "lake", "stream", "pond", "canal", "waterfront", "creek"
        ],
        PreferenceCategory.PARKS: [
            "park", "garden", "playground", "recreation", "open space"
        ],
        PreferenceCategory.QUIET: [
            "quiet", "peaceful", "calm", "serene", "tranquil", "residential"
        ],
        PreferenceCategory.SCENIC: [
            "scenic", "beautiful", "view", "vista", "overlook", "landscape", "picturesque"
        ],
        PreferenceCategory.EXERCISE: [
            "exercise", "fitness", "running", "jogging", "workout", "training", "hill", "steep"
        ],
        PreferenceCategory.URBAN: [
            "urban", "city", "downtown", "commercial", "shopping", "bustling"
        ]
    }

    def __init__(self):
        """Initialize the interpretable scorer."""
        # Feature importance thresholds for explanations
        self.explanation_thresholds = {
            FeatureType.CLOSE_TO_FOREST: 0.3,
            FeatureType.CLOSE_TO_WATER: 0.3,
            FeatureType.CLOSE_TO_PARK: 0.3,
            FeatureType.PATH_QUALITY: 0.6,
            FeatureType.INTERSECTION_DENSITY: 0.7,
            FeatureType.ELEVATION_VARIETY: 0.4
        }

        # Human-readable feature names
        self.feature_names = {
            FeatureType.CLOSE_TO_FOREST: "forest access",
            FeatureType.CLOSE_TO_WATER: "water features",
            FeatureType.CLOSE_TO_PARK: "park access",
            FeatureType.PATH_QUALITY: "walking paths",
            FeatureType.INTERSECTION_DENSITY: "connectivity",
            FeatureType.ELEVATION_VARIETY: "terrain variety"
        }

    def analyze_preferences(self, preference_text: str) -> PreferenceWeights:
        """
        Analyze user preference text and return feature weights.

        Args:
            preference_text: Natural language preference description

        Returns:
            PreferenceWeights with calculated weights
        """
        preference_lower = preference_text.lower()
        weights = PreferenceWeights()

        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.PREFERENCE_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in preference_lower)
            category_scores[category] = score

        # Convert category scores to feature weights
        # Nature preferences
        nature_score = category_scores.get(PreferenceCategory.NATURE, 0)
        if nature_score > 0:
            weights.close_to_forest = min(1.0, 0.3 + nature_score * 0.2)
            weights.path_quality += 0.2  # Nature lovers prefer good paths

        # Water preferences
        water_score = category_scores.get(PreferenceCategory.WATER, 0)
        if water_score > 0:
            weights.close_to_water = min(1.0, 0.3 + water_score * 0.2)

        # Park preferences
        park_score = category_scores.get(PreferenceCategory.PARKS, 0)
        if park_score > 0:
            weights.close_to_park = min(1.0, 0.3 + park_score * 0.2)

        # Quiet preferences
        quiet_score = category_scores.get(PreferenceCategory.QUIET, 0)
        if quiet_score > 0:
            weights.path_quality += 0.3  # Quiet areas prefer good pedestrian paths
            weights.intersection_density -= 0.1  # Less connectivity for quiet
            weights.close_to_park += 0.2  # Parks are often quiet

        # Scenic preferences
        scenic_score = category_scores.get(PreferenceCategory.SCENIC, 0)
        if scenic_score > 0:
            weights.elevation_variety += 0.4  # Scenic areas often have elevation
            weights.close_to_water += 0.2  # Water features are scenic
            weights.close_to_forest += 0.2  # Forests can be scenic

        # Exercise preferences
        exercise_score = category_scores.get(PreferenceCategory.EXERCISE, 0)
        if exercise_score > 0:
            weights.elevation_variety += 0.3  # Hills for exercise
            weights.path_quality += 0.2  # Good paths for running
            weights.intersection_density += 0.1  # Variety in routes

        # Urban preferences
        urban_score = category_scores.get(PreferenceCategory.URBAN, 0)
        if urban_score > 0:
            weights.intersection_density += 0.3  # Urban areas are well connected
            weights.close_to_forest -= 0.2  # Less emphasis on nature
            weights.close_to_water -= 0.1

        # Ensure all weights are in [0, 1] range
        weights.close_to_forest = max(0.0, min(1.0, weights.close_to_forest))
        weights.close_to_water = max(0.0, min(1.0, weights.close_to_water))
        weights.close_to_park = max(0.0, min(1.0, weights.close_to_park))
        weights.path_quality = max(0.1, min(1.0, weights.path_quality))  # Always some importance
        weights.intersection_density = max(0.0, min(1.0, weights.intersection_density))
        weights.elevation_variety = max(0.0, min(1.0, weights.elevation_variety))

        return weights

    def score_candidate(self, node_id: int, lat: float, lon: float,
                       features: CellFeatures, weights: PreferenceWeights) -> ScoredCandidate:
        """
        Score a candidate location using discrete features.

        Args:
            node_id: Node identifier
            lat: Latitude coordinate
            lon: Longitude coordinate
            features: Pre-computed cell features
            weights: Preference weights

        Returns:
            ScoredCandidate with score and explanation
        """
        # Extract feature scores
        feature_scores = {
            FeatureType.CLOSE_TO_FOREST: features.get_feature(FeatureType.CLOSE_TO_FOREST),
            FeatureType.CLOSE_TO_WATER: features.get_feature(FeatureType.CLOSE_TO_WATER),
            FeatureType.CLOSE_TO_PARK: features.get_feature(FeatureType.CLOSE_TO_PARK),
            FeatureType.PATH_QUALITY: features.get_feature(FeatureType.PATH_QUALITY),
            FeatureType.INTERSECTION_DENSITY: features.get_feature(FeatureType.INTERSECTION_DENSITY),
            FeatureType.ELEVATION_VARIETY: features.get_feature(FeatureType.ELEVATION_VARIETY)
        }

        # Calculate weighted score
        weighted_scores = {
            FeatureType.CLOSE_TO_FOREST: feature_scores[FeatureType.CLOSE_TO_FOREST] * weights.close_to_forest,
            FeatureType.CLOSE_TO_WATER: feature_scores[FeatureType.CLOSE_TO_WATER] * weights.close_to_water,
            FeatureType.CLOSE_TO_PARK: feature_scores[FeatureType.CLOSE_TO_PARK] * weights.close_to_park,
            FeatureType.PATH_QUALITY: feature_scores[FeatureType.PATH_QUALITY] * weights.path_quality,
            FeatureType.INTERSECTION_DENSITY: feature_scores[FeatureType.INTERSECTION_DENSITY] * weights.intersection_density,
            FeatureType.ELEVATION_VARIETY: feature_scores[FeatureType.ELEVATION_VARIETY] * weights.elevation_variety
        }

        # Calculate overall score (weighted average)
        total_weight = (weights.close_to_forest + weights.close_to_water + weights.close_to_park +
                       weights.path_quality + weights.intersection_density + weights.elevation_variety)

        if total_weight > 0:
            overall_score = sum(weighted_scores.values()) / total_weight
        else:
            overall_score = 0.5  # Neutral score if no weights

        # Generate explanation and reasoning
        explanation, reasoning_details = self._generate_explanation(
            feature_scores, weights, weighted_scores
        )

        return ScoredCandidate(
            node_id=node_id,
            lat=lat,
            lon=lon,
            overall_score=overall_score,
            feature_scores=feature_scores,
            explanation=explanation,
            reasoning_details=reasoning_details
        )

    def score_multiple_candidates(self, candidates: list[tuple[int, float, float]],
                                features_list: list[CellFeatures],
                                preference_text: str) -> list[ScoredCandidate]:
        """
        Score multiple candidates at once.

        Args:
            candidates: List of (node_id, lat, lon) tuples
            features_list: List of CellFeatures for each candidate
            preference_text: User preference text

        Returns:
            List of ScoredCandidate objects, sorted by score (highest first)
        """
        weights = self.analyze_preferences(preference_text)
        scored_candidates = []

        for (node_id, lat, lon), features in zip(candidates, features_list, strict=False):
            if features:  # Only score if features are available
                scored_candidate = self.score_candidate(node_id, lat, lon, features, weights)
                scored_candidates.append(scored_candidate)

        # Sort by overall score (highest first)
        scored_candidates.sort(key=lambda x: x.overall_score, reverse=True)

        return scored_candidates

    def _generate_explanation(self, feature_scores: dict[FeatureType, float],
                            weights: PreferenceWeights,
                            weighted_scores: dict[FeatureType, float]) -> tuple[str, list[str]]:
        """Generate human-readable explanation for the score."""

        # Find the most important contributing features
        important_features = []
        reasoning_details = []

        for feature_type, weighted_score in weighted_scores.items():
            raw_score = feature_scores[feature_type]
            weight = self._get_weight_for_feature(weights, feature_type)

            # Include feature if it contributes significantly
            if weighted_score > 0.1 and raw_score > self.explanation_thresholds[feature_type]:
                feature_name = self.feature_names[feature_type]

                # Determine quality descriptor
                if raw_score >= 0.8:
                    quality = "excellent"
                elif raw_score >= 0.6:
                    quality = "good"
                elif raw_score >= 0.4:
                    quality = "decent"
                else:
                    quality = "some"

                important_features.append(f"{quality} {feature_name}")
                reasoning_details.append(
                    f"{feature_name.title()}: {raw_score:.2f} (weight: {weight:.2f}, "
                    f"contribution: {weighted_score:.2f})"
                )

        # Generate main explanation
        if len(important_features) == 0:
            explanation = "Basic walkable area with standard amenities"
        elif len(important_features) == 1:
            explanation = f"Location with {important_features[0]}"
        elif len(important_features) == 2:
            explanation = f"Location with {important_features[0]} and {important_features[1]}"
        else:
            # Multiple features - list first two and indicate more
            explanation = f"Location with {important_features[0]}, {important_features[1]}, and more"

        return explanation, reasoning_details

    def _get_weight_for_feature(self, weights: PreferenceWeights, feature_type: FeatureType) -> float:
        """Get the weight for a specific feature type."""
        weight_map = {
            FeatureType.CLOSE_TO_FOREST: weights.close_to_forest,
            FeatureType.CLOSE_TO_WATER: weights.close_to_water,
            FeatureType.CLOSE_TO_PARK: weights.close_to_park,
            FeatureType.PATH_QUALITY: weights.path_quality,
            FeatureType.INTERSECTION_DENSITY: weights.intersection_density,
            FeatureType.ELEVATION_VARIETY: weights.elevation_variety
        }
        return weight_map.get(feature_type, 0.0)

    def explain_preferences(self, preference_text: str) -> dict:
        """
        Explain how user preferences are interpreted.

        Args:
            preference_text: User preference text

        Returns:
            Dictionary explaining the preference analysis
        """
        weights = self.analyze_preferences(preference_text)

        # Find detected preference categories
        preference_lower = preference_text.lower()
        detected_categories = []

        for category, keywords in self.PREFERENCE_KEYWORDS.items():
            matched_keywords = [kw for kw in keywords if kw in preference_lower]
            if matched_keywords:
                detected_categories.append({
                    'category': category.value,
                    'matched_keywords': matched_keywords[:3]  # Show first 3 matches
                })

        # Explain feature weights
        feature_importance = []
        for feature_type in FeatureType:
            weight = self._get_weight_for_feature(weights, feature_type)
            if weight > 0.1:
                importance_level = "high" if weight > 0.6 else "medium" if weight > 0.3 else "low"
                feature_importance.append({
                    'feature': self.feature_names[feature_type],
                    'weight': weight,
                    'importance': importance_level
                })

        return {
            'original_text': preference_text,
            'detected_categories': detected_categories,
            'feature_weights': {
                'close_to_forest': weights.close_to_forest,
                'close_to_water': weights.close_to_water,
                'close_to_park': weights.close_to_park,
                'path_quality': weights.path_quality,
                'intersection_density': weights.intersection_density,
                'elevation_variety': weights.elevation_variety
            },
            'feature_importance': sorted(feature_importance, key=lambda x: x['weight'], reverse=True)
        }
