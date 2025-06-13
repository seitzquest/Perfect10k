"""
Persistent Graph Cache System for Perfect10k
Stores OSM graphs on disk to avoid repeated downloads and processing.
"""

import os
import pickle
import json
import hashlib
import time
import math
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import networkx as nx
import osmnx as ox
from loguru import logger


class GraphCacheEntry:
    """Represents a cached graph with metadata."""
    
    def __init__(self, graph: nx.MultiGraph, center: Tuple[float, float], 
                 radius: float, cache_key: str):
        self.graph = graph
        self.center = center  # (lat, lon)
        self.radius = radius
        self.cache_key = cache_key
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0


class PersistentGraphCache:
    """
    Disk-based cache for OSM graphs with geographic indexing.
    Stores graphs by geographic regions to enable efficient reuse.
    """
    
    def __init__(self, cache_dir: str = "cache/graphs", max_cache_size_gb: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = max_cache_size_gb * 1024 * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        
        # In-memory index for fast lookups
        self.cache_index: Dict[str, Dict] = {}
        self.memory_cache: Dict[str, GraphCacheEntry] = {}
        
        # Load existing cache index
        self._load_cache_index()
        
        logger.info(f"Initialized graph cache at {self.cache_dir}")
        logger.info(f"Found {len(self.cache_index)} cached graphs")
    
    def get_graph(self, lat: float, lon: float, radius: float = 8000) -> Optional[nx.MultiGraph]:
        """
        Get a graph that covers the requested location.
        Returns None if no suitable cached graph is found.
        """
        cache_key = self._find_covering_cache_key(lat, lon, radius)
        
        if cache_key:
            logger.info(f"Found covering cached graph: {cache_key}")
            return self._load_cached_graph(cache_key)
        
        logger.info(f"No cached graph found for ({lat:.6f}, {lon:.6f})")
        return None
    
    def store_graph(self, graph: nx.MultiGraph, center: Tuple[float, float], 
                   radius: float) -> str:
        """
        Store a graph in the persistent cache.
        Returns the cache key for the stored graph.
        """
        cache_key = self._generate_cache_key(center[0], center[1], radius)
        
        # Store to disk
        self._save_graph_to_disk(graph, cache_key, center, radius)
        
        # Update in-memory cache
        entry = GraphCacheEntry(graph, center, radius, cache_key)
        self.memory_cache[cache_key] = entry
        
        # Update index
        self.cache_index[cache_key] = {
            "center_lat": center[0],
            "center_lon": center[1],
            "radius": radius,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 1,
            "node_count": len(graph.nodes()),
            "edge_count": len(graph.edges()),
            "file_size": self._get_file_size(cache_key)
        }
        
        self._save_cache_index()
        
        # Cleanup old entries if cache is too large
        self._cleanup_cache_if_needed()
        
        logger.info(f"Stored graph {cache_key} with {len(graph.nodes())} nodes")
        return cache_key
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        total_size = sum(
            entry.get("file_size", 0) for entry in self.cache_index.values()
        )
        
        return {
            "total_graphs": len(self.cache_index),
            "total_size_mb": total_size / (1024 * 1024),
            "memory_cached": len(self.memory_cache),
            "cache_dir": str(self.cache_dir),
            "max_size_gb": self.max_cache_size_bytes / (1024 * 1024 * 1024)
        }
    
    def _find_covering_cache_key(self, lat: float, lon: float, 
                                required_radius: float) -> Optional[str]:
        """
        Find a cached graph that covers the requested location and radius.
        """
        best_match = None
        best_margin = float('inf')
        
        for cache_key, entry in self.cache_index.items():
            center_lat = entry["center_lat"]
            center_lon = entry["center_lon"]
            cached_radius = entry["radius"]
            
            # Calculate distance from requested point to cache center
            distance = self._haversine_distance(lat, lon, center_lat, center_lon)
            
            # Check if this cached graph covers the required area
            coverage_radius = cached_radius * 0.9  # Safety margin
            required_coverage = distance + required_radius
            
            if required_coverage <= coverage_radius:
                # This graph covers the required area
                margin = coverage_radius - required_coverage
                if margin < best_margin:
                    best_margin = margin
                    best_match = cache_key
        
        return best_match
    
    def _load_cached_graph(self, cache_key: str) -> Optional[nx.MultiGraph]:
        """Load a graph from cache (memory or disk)."""
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._update_cache_index_access(cache_key)
            logger.info(f"Loaded graph {cache_key} from memory cache")
            return entry.graph
        
        # Load from disk
        try:
            graph_file = self.cache_dir / f"{cache_key}.graph"
            
            if not graph_file.exists():
                logger.warning(f"Cache file not found: {graph_file}")
                return None
            
            with open(graph_file, 'rb') as f:
                graph = pickle.load(f)
            
            # Update access stats
            self._update_cache_index_access(cache_key)
            
            # Store in memory cache for fast future access
            if cache_key in self.cache_index:
                entry_info = self.cache_index[cache_key]
                center = (entry_info["center_lat"], entry_info["center_lon"])
                radius = entry_info["radius"]
                
                entry = GraphCacheEntry(graph, center, radius, cache_key)
                entry.last_accessed = time.time()
                entry.access_count = entry_info.get("access_count", 1) + 1
                
                # Limit memory cache size
                if len(self.memory_cache) >= 5:  # Max 5 graphs in memory
                    self._evict_oldest_from_memory()
                
                self.memory_cache[cache_key] = entry
            
            logger.info(f"Loaded graph {cache_key} from disk cache")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to load cached graph {cache_key}: {e}")
            return None
    
    def _save_graph_to_disk(self, graph: nx.MultiGraph, cache_key: str, 
                           center: Tuple[float, float], radius: float):
        """Save a graph to disk."""
        try:
            graph_file = self.cache_dir / f"{cache_key}.graph"
            
            with open(graph_file, 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved graph {cache_key} to disk ({self._get_file_size(cache_key)} bytes)")
            
        except Exception as e:
            logger.error(f"Failed to save graph {cache_key}: {e}")
            raise
    
    def _generate_cache_key(self, lat: float, lon: float, radius: float) -> str:
        """Generate a unique cache key for a graph."""
        # Round coordinates to create cache regions
        # This allows nearby requests to share the same cache
        lat_rounded = round(lat, 3)  # ~100m precision
        lon_rounded = round(lon, 3)
        radius_rounded = round(radius, -2)  # Round to nearest 100m
        
        key_string = f"graph_{lat_rounded}_{lon_rounded}_{radius_rounded}"
        
        # Add hash for uniqueness while keeping human-readable prefix
        key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
        
        return f"{key_string}_{key_hash}"
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate haversine distance in meters."""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _load_cache_index(self):
        """Load the cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    self.cache_index = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            self.cache_index = {}
    
    def _save_cache_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _update_cache_index_access(self, cache_key: str):
        """Update access statistics for a cache entry."""
        if cache_key in self.cache_index:
            self.cache_index[cache_key]["last_accessed"] = time.time()
            self.cache_index[cache_key]["access_count"] = \
                self.cache_index[cache_key].get("access_count", 0) + 1
            self._save_cache_index()
    
    def _get_file_size(self, cache_key: str) -> int:
        """Get the file size of a cached graph."""
        try:
            graph_file = self.cache_dir / f"{cache_key}.graph"
            return graph_file.stat().st_size if graph_file.exists() else 0
        except:
            return 0
    
    def _cleanup_cache_if_needed(self):
        """Remove old cache entries if cache size exceeds limit."""
        total_size = sum(
            entry.get("file_size", 0) for entry in self.cache_index.values()
        )
        
        if total_size <= self.max_cache_size_bytes:
            return
        
        logger.info(f"Cache size ({total_size / 1024 / 1024:.1f}MB) exceeds limit, cleaning up...")
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get("last_accessed", 0)
        )
        
        # Remove oldest entries until under limit
        removed_count = 0
        for cache_key, entry in sorted_entries:
            if total_size <= self.max_cache_size_bytes * 0.8:  # Remove to 80% of limit
                break
            
            file_size = entry.get("file_size", 0)
            self._remove_cache_entry(cache_key)
            total_size -= file_size
            removed_count += 1
        
        logger.info(f"Removed {removed_count} old cache entries")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry from disk and index."""
        try:
            # Remove from disk
            graph_file = self.cache_dir / f"{cache_key}.graph"
            if graph_file.exists():
                graph_file.unlink()
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            
            # Remove from index
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
            
            logger.info(f"Removed cache entry {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key}: {e}")
    
    def _evict_oldest_from_memory(self):
        """Evict the least recently used graph from memory cache."""
        if not self.memory_cache:
            return
        
        oldest_key = min(
            self.memory_cache.keys(),
            key=lambda k: self.memory_cache[k].last_accessed
        )
        
        del self.memory_cache[oldest_key]
        logger.debug(f"Evicted {oldest_key} from memory cache")