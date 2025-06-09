import networkx as nx
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import math
from collections import defaultdict


class ResolutionStrategy(Enum):
    """Different strategies for complexity reduction."""
    RANDOM_EDGE_REMOVAL = "random_edge_removal"
    DENSITY_BASED = "density_based"
    IMPORTANCE_BASED = "importance_based"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


@dataclass
class ResolutionConfig:
    """Configuration for resolution reduction."""
    strategy: ResolutionStrategy = ResolutionStrategy.ADAPTIVE
    target_complexity: int = 1000  # Target number of nodes
    reduction_factor: float = 0.7  # Fraction of edges to keep
    preserve_connectivity: bool = True
    min_degree: int = 2  # Minimum degree for nodes to keep
    density_threshold: float = 0.01  # For density-based reduction


class ComplexityAnalyzer:
    """Analyzes graph complexity and density patterns."""
    
    @staticmethod
    def analyze_graph_complexity(G: nx.Graph) -> Dict[str, float]:
        """Analyze various complexity metrics of the graph."""
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        
        # Basic metrics
        density = nx.density(G)
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        
        # Clustering and connectivity
        try:
            avg_clustering = nx.average_clustering(G)
            connected_components = nx.number_connected_components(G)
        except:
            avg_clustering = 0
            connected_components = 1
        
        # Spatial density (if nodes have coordinates)
        spatial_density = ComplexityAnalyzer._calculate_spatial_density(G)
        
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "avg_degree": avg_degree,
            "avg_clustering": avg_clustering,
            "connected_components": connected_components,
            "spatial_density": spatial_density,
            "complexity_score": num_nodes * density * avg_clustering
        }
    
    @staticmethod
    def _calculate_spatial_density(G: nx.Graph) -> float:
        """Calculate spatial density based on node coordinates."""
        try:
            positions = []
            for node in G.nodes():
                node_data = G.nodes[node]
                if 'x' in node_data and 'y' in node_data:
                    positions.append((node_data['x'], node_data['y']))
            
            if len(positions) < 2:
                return 0.0
            
            # Calculate average distance between nodes
            total_distance = 0
            count = 0
            for i in range(len(positions)):
                for j in range(i + 1, min(i + 100, len(positions))):  # Sample to avoid O(n²)
                    dist = math.sqrt(
                        (positions[i][0] - positions[j][0]) ** 2 + 
                        (positions[i][1] - positions[j][1]) ** 2
                    )
                    total_distance += dist
                    count += 1
            
            avg_distance = total_distance / count if count > 0 else 1.0
            return len(positions) / (avg_distance ** 2)  # Density = nodes / area
            
        except:
            return 0.0


class RandomEdgeReducer:
    """Random edge removal strategy."""
    
    @staticmethod
    def reduce(G: nx.Graph, config: ResolutionConfig) -> nx.Graph:
        """Randomly remove edges while preserving connectivity."""
        G_reduced = G.copy()
        edges = list(G_reduced.edges())
        
        # Calculate number of edges to remove
        target_edges = int(len(edges) * config.reduction_factor)
        edges_to_remove = len(edges) - target_edges
        
        # Randomly shuffle edges
        random.shuffle(edges)
        
        removed_count = 0
        for u, v in edges:
            if removed_count >= edges_to_remove:
                break
                
            # Test if removal maintains connectivity
            if config.preserve_connectivity:
                G_temp = G_reduced.copy()
                G_temp.remove_edge(u, v)
                
                if nx.is_connected(G_temp):
                    G_reduced.remove_edge(u, v)
                    removed_count += 1
            else:
                G_reduced.remove_edge(u, v)
                removed_count += 1
        
        return G_reduced


class DensityBasedReducer:
    """Density-based reduction - removes edges in high-density areas."""
    
    @staticmethod
    def reduce(G: nx.Graph, config: ResolutionConfig) -> nx.Graph:
        """Remove edges based on local density."""
        G_reduced = G.copy()
        
        # Calculate local density for each edge
        edge_densities = DensityBasedReducer._calculate_edge_densities(G)
        
        # Sort edges by density (highest first)
        sorted_edges = sorted(edge_densities.items(), key=lambda x: x[1], reverse=True)
        
        # Remove edges starting from highest density areas
        target_edges = int(len(G.edges()) * config.reduction_factor)
        edges_to_remove = len(G.edges()) - target_edges
        
        removed_count = 0
        for (u, v), density in sorted_edges:
            if removed_count >= edges_to_remove:
                break
                
            if density > config.density_threshold:
                if config.preserve_connectivity:
                    G_temp = G_reduced.copy()
                    G_temp.remove_edge(u, v)
                    
                    if nx.is_connected(G_temp):
                        G_reduced.remove_edge(u, v)
                        removed_count += 1
                else:
                    if G_reduced.has_edge(u, v):
                        G_reduced.remove_edge(u, v)
                        removed_count += 1
        
        return G_reduced
    
    @staticmethod
    def _calculate_edge_densities(G: nx.Graph, radius: float = 500) -> Dict[Tuple[int, int], float]:
        """Calculate local density around each edge."""
        edge_densities = {}
        
        for u, v in G.edges():
            try:
                # Get positions of edge endpoints
                u_pos = (G.nodes[u].get('x', 0), G.nodes[u].get('y', 0))
                v_pos = (G.nodes[v].get('x', 0), G.nodes[v].get('y', 0))
                
                # Calculate midpoint
                mid_x = (u_pos[0] + v_pos[0]) / 2
                mid_y = (u_pos[1] + v_pos[1]) / 2
                
                # Count nearby nodes
                nearby_count = 0
                for node in G.nodes():
                    node_pos = (G.nodes[node].get('x', 0), G.nodes[node].get('y', 0))
                    distance = math.sqrt(
                        (node_pos[0] - mid_x) ** 2 + (node_pos[1] - mid_y) ** 2
                    )
                    if distance <= radius:
                        nearby_count += 1
                
                # Density = nodes per unit area
                area = math.pi * radius ** 2
                density = nearby_count / area
                edge_densities[(u, v)] = density
                
            except:
                edge_densities[(u, v)] = 0.0
        
        return edge_densities


class ImportanceBasedReducer:
    """Importance-based reduction using centrality measures."""
    
    @staticmethod
    def reduce(G: nx.Graph, config: ResolutionConfig) -> nx.Graph:
        """Remove edges based on importance scores."""
        G_reduced = G.copy()
        
        # Calculate edge importance scores
        edge_importance = ImportanceBasedReducer._calculate_edge_importance(G)
        
        # Sort edges by importance (lowest first for removal)
        sorted_edges = sorted(edge_importance.items(), key=lambda x: x[1])
        
        # Remove least important edges
        target_edges = int(len(G.edges()) * config.reduction_factor)
        edges_to_remove = len(G.edges()) - target_edges
        
        removed_count = 0
        for (u, v), importance in sorted_edges:
            if removed_count >= edges_to_remove:
                break
                
            if config.preserve_connectivity:
                G_temp = G_reduced.copy()
                G_temp.remove_edge(u, v)
                
                if nx.is_connected(G_temp):
                    G_reduced.remove_edge(u, v)
                    removed_count += 1
            else:
                if G_reduced.has_edge(u, v):
                    G_reduced.remove_edge(u, v)
                    removed_count += 1
        
        return G_reduced
    
    @staticmethod
    def _calculate_edge_importance(G: nx.Graph) -> Dict[Tuple[int, int], float]:
        """Calculate importance score for each edge."""
        edge_importance = {}
        
        try:
            # Calculate centrality measures
            betweenness_centrality = nx.edge_betweenness_centrality(G)
            
            # For each edge, combine different importance measures
            for u, v in G.edges():
                # Betweenness centrality
                betweenness = betweenness_centrality.get((u, v), 0)
                
                # Degree importance (sum of endpoint degrees)
                degree_importance = G.degree[u] + G.degree[v]
                
                # Length importance (shorter edges are more important)
                edge_data = G.get_edge_data(u, v, {})
                length = edge_data.get('length', 1)
                length_importance = 1.0 / (length + 1)  # Inverse of length
                
                # Combined importance score
                importance = betweenness * 0.5 + (degree_importance / 100) * 0.3 + length_importance * 0.2
                edge_importance[(u, v)] = importance
                
        except Exception as e:
            # Fallback to degree-based importance
            for u, v in G.edges():
                degree_importance = G.degree[u] + G.degree[v]
                edge_importance[(u, v)] = degree_importance / 100
        
        return edge_importance


class HierarchicalReducer:
    """Hierarchical reduction using multi-level approach."""
    
    @staticmethod
    def reduce(G: nx.Graph, config: ResolutionConfig) -> nx.Graph:
        """Apply hierarchical reduction."""
        G_reduced = G.copy()
        
        # Multi-level reduction
        levels = 3
        reduction_per_level = (1 - config.reduction_factor) / levels
        
        for level in range(levels):
            current_reduction = config.reduction_factor + reduction_per_level * level
            
            # Apply different strategy at each level
            if level == 0:
                # Level 1: Remove low-degree nodes
                G_reduced = HierarchicalReducer._remove_low_degree_nodes(G_reduced, config.min_degree)
            elif level == 1:
                # Level 2: Density-based reduction
                temp_config = ResolutionConfig(reduction_factor=current_reduction)
                G_reduced = DensityBasedReducer.reduce(G_reduced, temp_config)
            else:
                # Level 3: Random reduction
                temp_config = ResolutionConfig(reduction_factor=current_reduction)
                G_reduced = RandomEdgeReducer.reduce(G_reduced, temp_config)
        
        return G_reduced
    
    @staticmethod
    def _remove_low_degree_nodes(G: nx.Graph, min_degree: int) -> nx.Graph:
        """Remove nodes with degree below threshold."""
        G_reduced = G.copy()
        
        # Find nodes with low degree
        low_degree_nodes = [node for node, degree in G_reduced.degree() if degree < min_degree]
        
        # Remove nodes while preserving connectivity
        for node in low_degree_nodes:
            if G_reduced.has_node(node):
                G_temp = G_reduced.copy()
                G_temp.remove_node(node)
                
                if nx.is_connected(G_temp):
                    G_reduced.remove_node(node)
        
        return G_reduced


class AdaptiveReducer:
    """Adaptive reducer that chooses strategy based on graph characteristics."""
    
    @staticmethod
    def reduce(G: nx.Graph, config: ResolutionConfig) -> nx.Graph:
        """Adaptively choose reduction strategy based on graph properties."""
        # Analyze graph complexity
        complexity_metrics = ComplexityAnalyzer.analyze_graph_complexity(G)
        
        # Choose strategy based on graph characteristics
        if complexity_metrics["spatial_density"] > 0.001:
            # High spatial density - use density-based reduction
            return DensityBasedReducer.reduce(G, config)
        elif complexity_metrics["avg_clustering"] > 0.3:
            # High clustering - use importance-based reduction
            return ImportanceBasedReducer.reduce(G, config)
        elif complexity_metrics["num_nodes"] > 2000:
            # Very large graph - use hierarchical reduction
            return HierarchicalReducer.reduce(G, config)
        else:
            # Default - use random reduction
            return RandomEdgeReducer.reduce(G, config)


class ResolutionManager:
    """Main interface for complexity reduction."""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self.reducers = {
            ResolutionStrategy.RANDOM_EDGE_REMOVAL: RandomEdgeReducer,
            ResolutionStrategy.DENSITY_BASED: DensityBasedReducer,
            ResolutionStrategy.IMPORTANCE_BASED: ImportanceBasedReducer,
            ResolutionStrategy.HIERARCHICAL: HierarchicalReducer,
            ResolutionStrategy.ADAPTIVE: AdaptiveReducer,
        }
    
    def reduce_complexity(self, G: nx.Graph) -> nx.Graph:
        """Reduce graph complexity using configured strategy."""
        # Check if reduction is needed
        complexity_metrics = ComplexityAnalyzer.analyze_graph_complexity(G)
        
        if complexity_metrics["num_nodes"] <= self.config.target_complexity:
            return G  # No reduction needed
        
        # Apply reduction strategy
        reducer = self.reducers[self.config.strategy]
        reduced_graph = reducer.reduce(G, self.config)
        
        # Verify result
        final_metrics = ComplexityAnalyzer.analyze_graph_complexity(reduced_graph)
        
        print(f"Complexity reduction: {complexity_metrics['num_nodes']} → {final_metrics['num_nodes']} nodes")
        print(f"Edge reduction: {complexity_metrics['num_edges']} → {final_metrics['num_edges']} edges")
        
        return reduced_graph
    
    def get_optimal_config(self, G: nx.Graph) -> ResolutionConfig:
        """Get optimal configuration for the given graph."""
        complexity_metrics = ComplexityAnalyzer.analyze_graph_complexity(G)
        
        # Adaptive configuration based on graph characteristics
        if complexity_metrics["num_nodes"] > 5000:
            # Very large graph - aggressive reduction
            return ResolutionConfig(
                strategy=ResolutionStrategy.HIERARCHICAL,
                target_complexity=1000,
                reduction_factor=0.5
            )
        elif complexity_metrics["spatial_density"] > 0.001:
            # Dense graph - density-based reduction
            return ResolutionConfig(
                strategy=ResolutionStrategy.DENSITY_BASED,
                target_complexity=1500,
                reduction_factor=0.7
            )
        else:
            # Default adaptive approach
            return ResolutionConfig(
                strategy=ResolutionStrategy.ADAPTIVE,
                target_complexity=2000,
                reduction_factor=0.8
            )