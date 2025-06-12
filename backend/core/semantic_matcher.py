"""
Semantic matching for route preferences.
Enhanced semantic matching that translates natural language preferences into route values.
"""

from typing import Dict


class SemanticMatcher:
    """Enhanced semantic matching for route preferences."""

    PREFERENCE_KEYWORDS = {
        "nature": ["park", "forest", "green", "tree", "garden", "nature", "woods", "trail"],
        "water": ["river", "lake", "pond", "stream", "water", "creek", "canal", "waterfront"],
        "quiet": ["residential", "quiet", "peaceful", "path", "footway", "calm", "serene"],
        "scenic": ["scenic", "view", "beautiful", "hill", "overlook", "vista", "landscape"],
        "urban": ["street", "avenue", "downtown", "city", "commercial", "shopping"],
        "historic": ["historic", "old", "heritage", "monument", "church", "castle"],
        "exercise": ["fitness", "running", "jogging", "workout", "steep", "hill", "climb"],
    }

    @staticmethod
    def create_value_function(preference: str) -> Dict[str, float]:
        """Create sophisticated value function based on preferences."""
        preference_lower = preference.lower()

        # Enhanced base values for different path types
        values = {
            "path": 0.8,
            "footway": 0.9,
            "track": 0.7,
            "steps": 0.5,
            "cycleway": 0.6,
            "bridleway": 0.7,
            "pedestrian": 0.8,
            "living_street": 0.6,
            "residential": 0.4,
            "tertiary": 0.3,
            "secondary": 0.2,
            "primary": 0.1,
            "default": 0.3,
        }

        # Apply preference-based boosts
        for category, keywords in SemanticMatcher.PREFERENCE_KEYWORDS.items():
            boost_factor = sum(1 for keyword in keywords if keyword in preference_lower)
            boost_factor = min(boost_factor, 3) * 0.1  # Cap boost at 0.3

            if category == "nature" and boost_factor > 0:
                values["path"] += boost_factor * 2
                values["footway"] += boost_factor * 1.5
                values["track"] += boost_factor * 2.5
                values["bridleway"] += boost_factor * 1.5
            elif category == "water" and boost_factor > 0:
                values["path"] += boost_factor * 1.5
                values["footway"] += boost_factor * 1.2
            elif category == "quiet" and boost_factor > 0:
                values["footway"] += boost_factor * 2
                values["path"] += boost_factor * 1.8
                values["residential"] += boost_factor * 1.2
            elif category == "scenic" and boost_factor > 0:
                values["track"] += boost_factor * 2
                values["path"] += boost_factor * 1.5
                values["steps"] += boost_factor * 1.5
            elif category == "urban" and boost_factor > 0:
                values["pedestrian"] += boost_factor * 1.5
                values["cycleway"] += boost_factor * 1.2
                values["living_street"] += boost_factor * 1.3
            elif category == "exercise" and boost_factor > 0:
                values["steps"] += boost_factor * 2
                values["track"] += boost_factor * 1.2

        # Normalize to [0, 1] range
        max_val = max(values.values())
        if max_val > 1:
            values = {k: min(v / max_val, 1.0) for k, v in values.items()}

        return values