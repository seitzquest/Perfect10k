#!/usr/bin/env python3
"""
Smart Cache Manager for Perfect10k
===================================

Implements intelligent caching that preserves variety and personalization
while dramatically improving performance.

Key principles:
1. Cache computation means, not results
2. Preserve variety through probabilistic selection
3. Enable personalization through dynamic scoring
4. Achieve <1s response times through smart precomputation
"""

import json
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger

from backend.performance_profiler import profile_function


@dataclass
class CachedNodeFeatures:
    """Cached features for a graph node."""

    node_id: int
    lat: float
    lon: float

    # Basic graph features
    degree: int
    betweenness_centrality: float

    # Semantic features (precomputed)
    forest_score: float
    water_score: float
    urban_score: float
    poi_scores: dict[str, float]
    safety_score: float

    # Spatial features
    accessibility: float
    connectivity_score: float

    last_updated: float = field(default_factory=time.time)


@dataclass
class CachedScoringWeights:
    """Cached scoring weights for different preferences."""

    preference_type: str
    weights: dict[str, float]
    bias_factors: dict[str, float]
    normalization_params: dict[str, float]
    last_updated: float = field(default_factory=time.time)


@dataclass
class CandidatePool:
    """Precomputed candidate pools for fast probabilistic selection."""

    area_key: str
    high_forest_nodes: list[int]
    high_water_nodes: list[int]
    high_urban_nodes: list[int]
    high_connectivity_nodes: list[int]
    balanced_nodes: list[int]
    last_updated: float = field(default_factory=time.time)


class SmartCacheManager:
    """
    Intelligent cache manager that optimizes for speed while preserving
    variety and personalization.
    """

    def __init__(self, cache_dir: str = "cache/smart_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches for hot data
        self.node_features_cache: dict[str, dict[int, CachedNodeFeatures]] = {}
        self.scoring_weights_cache: dict[str, CachedScoringWeights] = {}
        self.candidate_pools_cache: dict[str, CandidatePool] = {}
        self.graph_cache: dict[str, nx.MultiGraph] = {}
        self.spatial_grid_cache: dict[str, Any] = {}  # For SpatialGrid objects

        # Cache statistics
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)

        # Background processing
        self._background_tasks = []
        self._shutdown_event = threading.Event()

        logger.info(f"🚀 SmartCacheManager initialized with cache dir: {cache_dir}")

    def get_area_key(self, lat: float, lon: float, precision: int = 3) -> str:
        """Generate a consistent area key for caching."""
        return f"{lat:.{precision}f}_{lon:.{precision}f}"

    @profile_function("cache_get_graph")
    def get_cached_graph(self, lat: float, lon: float) -> nx.MultiGraph | None:
        """Get cached graph for an area."""
        area_key = self.get_area_key(lat, lon)

        if area_key in self.graph_cache:
            self.cache_hits["graph"] += 1
            logger.debug(f"📥 Graph cache HIT for {area_key}")
            return self.graph_cache[area_key]

        # Try loading from disk
        graph_file = self.cache_dir / f"graph_{area_key}.pkl"
        if graph_file.exists():
            try:
                with open(graph_file, "rb") as f:
                    graph = pickle.load(f)
                self.graph_cache[area_key] = graph
                self.cache_hits["graph"] += 1
                logger.info(f"📥 Graph loaded from disk cache for {area_key}")
                return graph
            except Exception as e:
                logger.warning(f"Failed to load cached graph: {e}")
                # Remove corrupted cache file
                try:
                    graph_file.unlink()
                    logger.info(f"🧹 Removed corrupted cache file: {graph_file.name}")
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up corrupted cache file: {cleanup_error}")

        self.cache_misses["graph"] += 1
        logger.debug(f"📤 Graph cache MISS for {area_key}")
        return None

    @profile_function("cache_store_graph")
    def store_graph(self, lat: float, lon: float, graph: nx.MultiGraph):
        """Store graph in cache."""
        area_key = self.get_area_key(lat, lon)

        # Store in memory
        self.graph_cache[area_key] = graph

        # Store on disk asynchronously with thread safety
        def save_to_disk():
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            graph_file = self.cache_dir / f"graph_{area_key}.pkl"
            temp_file = self.cache_dir / f"graph_{area_key}.pkl.tmp"
            try:
                # Create a copy of the graph to avoid "dictionary changed size" errors
                graph_copy = graph.copy()

                # Write to temporary file first to avoid corruption
                with open(temp_file, "wb") as f:
                    pickle.dump(graph_copy, f)

                # Atomic rename to final location
                temp_file.rename(graph_file)
                logger.debug(f"💾 Graph saved to disk cache for {area_key}")
            except Exception as e:
                logger.error(f"Failed to save graph to disk: {e}")
                # Clean up temp file if it exists
                if temp_file.exists():
                    temp_file.unlink()

        # Run in background thread
        threading.Thread(target=save_to_disk, daemon=True).start()

    @profile_function("cache_get_node_features")
    def get_node_features(
        self, area_key: str, node_ids: list[int]
    ) -> dict[int, CachedNodeFeatures]:
        """Get cached node features for specific nodes."""
        if area_key not in self.node_features_cache:
            self._load_node_features_from_disk(area_key)

        area_features = self.node_features_cache.get(area_key, {})
        result = {}

        for node_id in node_ids:
            if node_id in area_features:
                result[node_id] = area_features[node_id]
                self.cache_hits["node_features"] += 1
            else:
                self.cache_misses["node_features"] += 1

        logger.debug(f"📥 Node features: {len(result)}/{len(node_ids)} cached for {area_key}")
        return result

    def get_all_node_features(self, area_key: str) -> dict[int, CachedNodeFeatures] | None:
        """Get all cached node features for an area (used by CleanCandidateGenerator)."""
        if area_key not in self.node_features_cache:
            self._load_node_features_from_disk(area_key)

        area_features = self.node_features_cache.get(area_key, {})
        if area_features:
            self.cache_hits["node_features"] += 1
            logger.debug(
                f"✓ All node features cache hit for {area_key} ({len(area_features)} features)"
            )
            return area_features
        else:
            self.cache_misses["node_features"] += 1
            logger.debug(f"✗ All node features cache miss for {area_key}")
            return None

    @profile_function("cache_store_node_features")
    def store_node_features(self, area_key: str, features: dict[int, CachedNodeFeatures]):
        """Store node features in cache."""
        if area_key not in self.node_features_cache:
            self.node_features_cache[area_key] = {}

        self.node_features_cache[area_key].update(features)

        # Save to disk asynchronously
        def save_to_disk():
            self._save_node_features_to_disk(area_key)

        threading.Thread(target=save_to_disk, daemon=True).start()
        logger.info(f"💾 Stored {len(features)} node features for {area_key}")

    def get_scoring_weights(self, preference: str) -> CachedScoringWeights | None:
        """Get cached scoring weights for a preference."""
        if preference in self.scoring_weights_cache:
            self.cache_hits["scoring_weights"] += 1
            return self.scoring_weights_cache[preference]

        # Try loading from disk
        weights_file = self.cache_dir / f"weights_{preference}.json"
        if weights_file.exists():
            try:
                with open(weights_file) as f:
                    data = json.load(f)
                weights = CachedScoringWeights(**data)
                self.scoring_weights_cache[preference] = weights
                self.cache_hits["scoring_weights"] += 1
                return weights
            except Exception as e:
                logger.warning(f"Failed to load cached weights: {e}")

        self.cache_misses["scoring_weights"] += 1
        return None

    def store_scoring_weights(self, preference: str, weights: CachedScoringWeights):
        """Store scoring weights in cache."""
        self.scoring_weights_cache[preference] = weights

        # Save to disk
        weights_file = self.cache_dir / f"weights_{preference}.json"
        try:
            with open(weights_file, "w") as f:
                json.dump(
                    {
                        "preference_type": weights.preference_type,
                        "weights": weights.weights,
                        "bias_factors": weights.bias_factors,
                        "normalization_params": weights.normalization_params,
                        "last_updated": weights.last_updated,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"💾 Stored scoring weights for {preference}")
        except Exception as e:
            logger.error(f"Failed to save scoring weights: {e}")

    def get_candidate_pool(self, area_key: str) -> CandidatePool | None:
        """Get precomputed candidate pool for fast selection."""
        if area_key in self.candidate_pools_cache:
            pool = self.candidate_pools_cache[area_key]

            # Check if pool is still fresh (24 hours)
            if time.time() - pool.last_updated < 86400:
                self.cache_hits["candidate_pool"] += 1
                return pool

        # Try loading from disk
        pool_file = self.cache_dir / f"pool_{area_key}.json"
        if pool_file.exists():
            try:
                with open(pool_file) as f:
                    data = json.load(f)
                pool = CandidatePool(**data)

                # Check freshness
                if time.time() - pool.last_updated < 86400:
                    self.candidate_pools_cache[area_key] = pool
                    self.cache_hits["candidate_pool"] += 1
                    return pool
            except Exception as e:
                logger.warning(f"Failed to load candidate pool: {e}")

        self.cache_misses["candidate_pool"] += 1
        return None

    def store_candidate_pool(self, area_key: str, pool: CandidatePool):
        """Store candidate pool in cache."""
        self.candidate_pools_cache[area_key] = pool

        # Save to disk
        pool_file = self.cache_dir / f"pool_{area_key}.json"
        try:
            with open(pool_file, "w") as f:
                json.dump(
                    {
                        "area_key": pool.area_key,
                        "high_forest_nodes": pool.high_forest_nodes,
                        "high_water_nodes": pool.high_water_nodes,
                        "high_urban_nodes": pool.high_urban_nodes,
                        "high_connectivity_nodes": pool.high_connectivity_nodes,
                        "balanced_nodes": pool.balanced_nodes,
                        "last_updated": pool.last_updated,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"💾 Stored candidate pool for {area_key}")
        except Exception as e:
            logger.error(f"Failed to save candidate pool: {e}")

    def probabilistic_candidate_selection(
        self,
        area_key: str,
        preference: str,
        center_lat: float,
        center_lon: float,
        max_candidates: int = 20,
        variety_factor: float = 0.3,
    ) -> list[int]:
        """
        Probabilistic candidate selection that preserves variety.

        Args:
            area_key: Area identifier
            preference: User preference type
            center_lat, center_lon: Search center
            max_candidates: Maximum candidates to return
            variety_factor: How much randomness to add (0.0 = deterministic, 1.0 = random)
        """
        pool = self.get_candidate_pool(area_key)
        if not pool:
            logger.warning(f"No candidate pool for {area_key}, falling back to basic selection")
            return []

        # Select candidate pools based on preference
        candidate_sources = []

        if "forest" in preference.lower() or "nature" in preference.lower():
            candidate_sources.extend(
                [
                    (pool.high_forest_nodes, 0.4),
                    (pool.balanced_nodes, 0.3),
                    (pool.high_water_nodes, 0.2),
                    (pool.high_connectivity_nodes, 0.1),
                ]
            )
        elif "urban" in preference.lower() or "city" in preference.lower():
            candidate_sources.extend(
                [
                    (pool.high_urban_nodes, 0.4),
                    (pool.high_connectivity_nodes, 0.3),
                    (pool.balanced_nodes, 0.2),
                    (pool.high_forest_nodes, 0.1),
                ]
            )
        elif "water" in preference.lower():
            candidate_sources.extend(
                [
                    (pool.high_water_nodes, 0.5),
                    (pool.high_forest_nodes, 0.3),
                    (pool.balanced_nodes, 0.2),
                ]
            )
        else:
            # Balanced selection for unknown preferences
            candidate_sources.extend(
                [
                    (pool.balanced_nodes, 0.4),
                    (pool.high_forest_nodes, 0.2),
                    (pool.high_urban_nodes, 0.2),
                    (pool.high_connectivity_nodes, 0.2),
                ]
            )

        # Weighted probabilistic sampling
        selected_candidates = []

        for candidates, weight in candidate_sources:
            if not candidates:
                continue

            # Calculate how many to select from this pool
            pool_size = int(max_candidates * weight * (1 + np.random.normal(0, variety_factor)))
            pool_size = max(
                1, min(pool_size, len(candidates), max_candidates - len(selected_candidates))
            )

            # Random sampling with variety
            if variety_factor > 0:
                # Add controlled randomness
                selection_probs = np.ones(len(candidates))
                selection_probs += np.random.exponential(variety_factor, len(candidates))
                selection_probs /= selection_probs.sum()

                selected_indices = np.random.choice(
                    len(candidates),
                    size=min(pool_size, len(candidates)),
                    replace=False,
                    p=selection_probs,
                )
                pool_candidates = [candidates[i] for i in selected_indices]
            else:
                # Deterministic selection (for testing)
                pool_candidates = candidates[:pool_size]

            selected_candidates.extend(pool_candidates)

            if len(selected_candidates) >= max_candidates:
                break

        # Shuffle for final variety
        if variety_factor > 0:
            np.random.shuffle(selected_candidates)

        result = selected_candidates[:max_candidates]
        logger.debug(f"🎲 Selected {len(result)} candidates with variety_factor={variety_factor}")
        return result

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_hits = sum(self.cache_hits.values())
        total_misses = sum(self.cache_misses.values())
        total_requests = total_hits + total_misses

        hit_rate = total_hits / total_requests if total_requests > 0 else 0

        return {
            "overall_hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_hits": dict(self.cache_hits),
            "cache_misses": dict(self.cache_misses),
            "cache_sizes": {
                "graphs": len(self.graph_cache),
                "node_features": sum(len(area) for area in self.node_features_cache.values()),
                "scoring_weights": len(self.scoring_weights_cache),
                "candidate_pools": len(self.candidate_pools_cache),
            },
        }

    def _load_node_features_from_disk(self, area_key: str):
        """Load node features from disk cache."""
        features_file = self.cache_dir / f"features_{area_key}.pkl"
        if features_file.exists():
            try:
                with open(features_file, "rb") as f:
                    features = pickle.load(f)
                self.node_features_cache[area_key] = features
                logger.info(f"📥 Loaded {len(features)} node features from disk for {area_key}")
            except Exception as e:
                logger.warning(f"Failed to load node features from disk: {e}")

    def _save_node_features_to_disk(self, area_key: str):
        """Save node features to disk cache."""
        if area_key not in self.node_features_cache:
            return

        features_file = self.cache_dir / f"features_{area_key}.pkl"
        try:
            with open(features_file, "wb") as f:
                pickle.dump(self.node_features_cache[area_key], f)
            logger.debug(f"💾 Saved node features to disk for {area_key}")
        except Exception as e:
            logger.error(f"Failed to save node features to disk: {e}")

    def get_spatial_grid(self, area_key: str) -> Any | None:
        """Get cached spatial grid for an area."""
        # Check memory cache first
        if area_key in self.spatial_grid_cache:
            self.cache_hits["spatial_grid"] += 1
            logger.debug(f"✓ Spatial grid cache hit for {area_key}")
            return self.spatial_grid_cache[area_key]

        # Check disk cache
        grid_file = self.cache_dir / f"grid_{area_key}.pkl"
        if grid_file.exists():
            try:
                with open(grid_file, "rb") as f:
                    spatial_grid = pickle.load(f)
                self.spatial_grid_cache[area_key] = spatial_grid
                self.cache_hits["spatial_grid"] += 1
                logger.debug(f"✓ Spatial grid loaded from disk for {area_key}")
                return spatial_grid
            except Exception as e:
                logger.error(f"Failed to load spatial grid from disk: {e}")
                # Remove corrupted cache file
                try:
                    grid_file.unlink()
                    logger.info(f"🧹 Removed corrupted spatial grid cache: {grid_file.name}")
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to clean up corrupted spatial grid cache: {cleanup_error}"
                    )

        self.cache_misses["spatial_grid"] += 1
        logger.debug(f"✗ Spatial grid cache miss for {area_key}")
        return None

    def store_spatial_grid(self, area_key: str, spatial_grid: Any):
        """Store spatial grid in cache."""
        self.spatial_grid_cache[area_key] = spatial_grid

        # Save to disk in background with thread safety
        def save_to_disk():
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            grid_file = self.cache_dir / f"grid_{area_key}.pkl"
            temp_file = self.cache_dir / f"grid_{area_key}.pkl.tmp"
            try:
                # Write to temporary file first to avoid corruption
                with open(temp_file, "wb") as f:
                    pickle.dump(spatial_grid, f)

                # Atomic rename to final location
                temp_file.rename(grid_file)
                logger.debug(f"💾 Spatial grid saved to disk for {area_key}")
            except Exception as e:
                logger.error(f"Failed to save spatial grid to disk: {e}")
                # Clean up temp file if it exists
                if temp_file.exists():
                    temp_file.unlink()

        # Use thread to avoid blocking
        threading.Thread(target=save_to_disk, daemon=True).start()

    def get_cell_features(self, area_key: str) -> dict | None:
        """Get cached cell features for an area (FeatureDatabase format)."""
        # Check memory cache first
        cell_cache_key = f"cells_{area_key}"
        if cell_cache_key in self.spatial_grid_cache:
            self.cache_hits["cell_features"] += 1
            logger.debug(f"✓ Cell features cache hit for {area_key}")
            return self.spatial_grid_cache[cell_cache_key]

        # Check disk cache
        cells_file = self.cache_dir / f"cells_{area_key}.pkl"
        if cells_file.exists():
            try:
                with open(cells_file, "rb") as f:
                    cell_features = pickle.load(f)
                self.spatial_grid_cache[cell_cache_key] = cell_features
                self.cache_hits["cell_features"] += 1
                logger.debug(f"✓ Cell features loaded from disk for {area_key}")
                return cell_features
            except Exception as e:
                logger.error(f"Failed to load cell features from disk: {e}")
                # Remove corrupted cache file
                try:
                    cells_file.unlink()
                    logger.info(f"🧹 Removed corrupted cell features cache: {cells_file.name}")
                except Exception as cleanup_error:
                    logger.error(
                        f"Failed to clean up corrupted cell features cache: {cleanup_error}"
                    )

        self.cache_misses["cell_features"] += 1
        logger.debug(f"✗ Cell features cache miss for {area_key}")
        return None

    def store_cell_features(self, area_key: str, cell_features: dict):
        """Store cell features in cache (FeatureDatabase format)."""
        cell_cache_key = f"cells_{area_key}"
        self.spatial_grid_cache[cell_cache_key] = cell_features

        # Save to disk in background with thread safety
        def save_to_disk():
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            cells_file = self.cache_dir / f"cells_{area_key}.pkl"
            temp_file = self.cache_dir / f"cells_{area_key}.pkl.tmp"
            try:
                # Write to temporary file first to avoid corruption
                with open(temp_file, "wb") as f:
                    pickle.dump(cell_features, f)

                # Atomic rename to final location
                temp_file.rename(cells_file)
                logger.debug(f"💾 Cell features saved to disk for {area_key}")
            except Exception as e:
                logger.error(f"Failed to save cell features to disk: {e}")
                # Clean up temp file if it exists
                if temp_file.exists():
                    temp_file.unlink()

        # Use thread to avoid blocking
        threading.Thread(target=save_to_disk, daemon=True).start()

    def cleanup_old_cache(self, max_age_hours: int = 168):  # 1 week
        """Clean up old cache entries."""
        cutoff_time = time.time() - (max_age_hours * 3600)

        # Clean up in-memory caches
        for area_key, features in list(self.node_features_cache.items()):
            if all(f.last_updated < cutoff_time for f in features.values()):
                del self.node_features_cache[area_key]
                logger.info(f"🧹 Cleaned up old node features for {area_key}")

        # Clean up disk files
        for file_path in self.cache_dir.glob("*.pkl"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"🧹 Cleaned up old cache file: {file_path.name}")


# Global cache manager instance
cache_manager = SmartCacheManager()
