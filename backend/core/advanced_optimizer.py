import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import heapq
import random
from collections import defaultdict
import logging

from core.value_function import SpatialValueFunction, ValueFunctionOptimizer
from core.route_algorithms import RouteConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for advanced route optimization."""
    max_iterations: int = 1000
    population_size: int = 50          # For genetic algorithm
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_ratio: float = 0.2           # Top routes to keep each generation
    value_weight: float = 0.4          # Weight for value function in fitness
    distance_weight: float = 0.4       # Weight for distance accuracy
    diversity_weight: float = 0.2      # Weight for route diversity
    temperature_schedule: str = 'exponential'  # For simulated annealing
    initial_temperature: float = 100.0
    final_temperature: float = 0.1


class AdvancedRouteOptimizer:
    """
    Advanced route optimizer using multiple optimization techniques:
    - Value function guided search
    - Genetic algorithm for global optimization
    - Simulated annealing for local optimization
    - Multi-objective optimization
    """
    
    def __init__(self, route_config: RouteConfig, opt_config: OptimizationConfig = None):
        self.route_config = route_config
        self.opt_config = opt_config or OptimizationConfig()
        self.value_function = None
        self.graph = None
        self.start_node = None
        self.target_distance = None
        
    def optimize_route(
        self,
        graph: nx.Graph,
        start_node: int,
        target_distance: float,
        value_function: SpatialValueFunction,
        method: str = 'hybrid'
    ) -> List[int]:
        """
        Optimize route using specified method.
        
        Args:
            graph: NetworkX graph
            start_node: Starting node
            target_distance: Target route distance
            value_function: Spatial value function
            method: 'genetic', 'annealing', 'hybrid', 'value_guided'
            
        Returns:
            Optimized route as list of node IDs
        """
        self.graph = graph
        self.start_node = start_node
        self.target_distance = target_distance
        self.value_function = value_function
        
        logger.info(f"Starting route optimization with method: {method}")
        
        if method == 'genetic':
            return self._genetic_algorithm_optimization()
        elif method == 'annealing':
            return self._simulated_annealing_optimization()
        elif method == 'value_guided':
            return self._value_guided_search()
        elif method == 'hybrid':
            return self._hybrid_optimization()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _genetic_algorithm_optimization(self) -> List[int]:
        """Genetic algorithm for route optimization."""
        logger.info("Starting genetic algorithm optimization")
        
        # Initialize population
        population = self._generate_initial_population()
        
        best_route = None
        best_fitness = float('-inf')
        
        for generation in range(self.opt_config.max_iterations):
            # Evaluate fitness for all routes
            fitness_scores = []
            for route in population:
                fitness = self._calculate_fitness(route)
                fitness_scores.append((fitness, route))
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_route = route.copy()
            
            # Sort by fitness (descending)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select elite routes
            elite_count = int(self.opt_config.population_size * self.opt_config.elite_ratio)
            new_population = [route for _, route in fitness_scores[:elite_count]]
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.opt_config.population_size:
                if random.random() < self.opt_config.crossover_rate:
                    # Crossover
                    parent1 = self._tournament_selection(fitness_scores)
                    parent2 = self._tournament_selection(fitness_scores)
                    offspring = self._crossover(parent1, parent2)
                else:
                    # Copy from elite
                    offspring = random.choice([route for _, route in fitness_scores[:elite_count]]).copy()
                
                # Mutation
                if random.random() < self.opt_config.mutation_rate:
                    offspring = self._mutate(offspring)
                
                new_population.append(offspring)
            
            population = new_population
            
            if generation % 50 == 0:
                logger.info(f"Generation {generation}: Best fitness = {best_fitness:.3f}")
        
        logger.info(f"Genetic algorithm completed. Final fitness: {best_fitness:.3f}")
        return best_route or population[0]
    
    def _simulated_annealing_optimization(self) -> List[int]:
        """Simulated annealing for route optimization."""
        logger.info("Starting simulated annealing optimization")
        
        # Start with a random route
        current_route = self._generate_random_route()
        current_fitness = self._calculate_fitness(current_route)
        
        best_route = current_route.copy()
        best_fitness = current_fitness
        
        for iteration in range(self.opt_config.max_iterations):
            # Calculate temperature
            temperature = self._get_temperature(iteration)
            
            # Generate neighbor solution
            neighbor_route = self._generate_neighbor(current_route)
            neighbor_fitness = self._calculate_fitness(neighbor_route)
            
            # Accept or reject neighbor
            if self._accept_neighbor(current_fitness, neighbor_fitness, temperature):
                current_route = neighbor_route
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_route = current_route.copy()
            
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: Current fitness = {current_fitness:.3f}, "
                          f"Best fitness = {best_fitness:.3f}, Temperature = {temperature:.3f}")
        
        logger.info(f"Simulated annealing completed. Final fitness: {best_fitness:.3f}")
        return best_route
    
    def _value_guided_search(self) -> List[int]:
        """Value function guided search using A* with value heuristic."""
        logger.info("Starting value-guided search")
        
        # Use A* with value function as heuristic
        def value_heuristic(node):
            node_data = self.graph.nodes[node]
            if 'y' in node_data and 'x' in node_data:
                lat, lon = node_data['y'], node_data['x']
                return self.value_function.get_value_at_coords(lat, lon)
            return 0.5
        
        # Find high-value intermediate points
        high_value_nodes = self._find_high_value_nodes(top_k=20)
        
        best_route = None
        best_score = float('-inf')
        
        # Try routes through different high-value nodes
        for intermediate_node in high_value_nodes:
            try:
                # Route: start -> intermediate -> start
                path_to = nx.shortest_path(self.graph, self.start_node, intermediate_node, weight='length')
                path_back = nx.shortest_path(self.graph, intermediate_node, self.start_node, weight='length')
                
                # Combine paths (remove duplicate intermediate node)
                full_route = path_to + path_back[1:]
                
                # Evaluate route
                score = self._calculate_fitness(full_route)
                
                if score > best_score:
                    best_score = score
                    best_route = full_route
                    
            except nx.NetworkXNoPath:
                continue
        
        logger.info(f"Value-guided search completed. Best score: {best_score:.3f}")
        return best_route or [self.start_node]
    
    def _hybrid_optimization(self) -> List[int]:
        """Hybrid optimization combining multiple methods."""
        logger.info("Starting hybrid optimization")
        
        # Phase 1: Value-guided initialization
        initial_route = self._value_guided_search()
        
        # Phase 2: Genetic algorithm refinement
        self.opt_config.max_iterations = self.opt_config.max_iterations // 2
        
        # Initialize population with variations of the value-guided route
        population = [initial_route]
        for _ in range(self.opt_config.population_size - 1):
            variant = self._mutate(initial_route.copy())
            population.append(variant)
        
        # Run genetic algorithm
        genetic_result = self._genetic_algorithm_with_population(population)
        
        # Phase 3: Simulated annealing fine-tuning
        self.opt_config.max_iterations = self.opt_config.max_iterations // 2
        final_route = self._simulated_annealing_from_start(genetic_result)
        
        logger.info("Hybrid optimization completed")
        return final_route
    
    def _generate_initial_population(self) -> List[List[int]]:
        """Generate initial population for genetic algorithm."""
        population = []
        
        # Add some value-guided routes
        high_value_nodes = self._find_high_value_nodes(top_k=self.opt_config.population_size // 2)
        
        for node in high_value_nodes:
            try:
                route = self._create_route_through_node(node)
                if route:
                    population.append(route)
            except:
                continue
        
        # Fill remaining population with random routes
        while len(population) < self.opt_config.population_size:
            route = self._generate_random_route()
            if route:
                population.append(route)
        
        return population
    
    def _generate_random_route(self) -> List[int]:
        """Generate a random circular route."""
        try:
            # Find random nodes at different distances
            distances = nx.single_source_dijkstra_path_length(
                self.graph, self.start_node, cutoff=self.target_distance * 1.5, weight='length'
            )
            
            # Filter nodes by distance range
            candidate_nodes = [
                node for node, dist in distances.items()
                if self.target_distance * 0.2 < dist < self.target_distance * 0.8
            ]
            
            if not candidate_nodes:
                return [self.start_node]
            
            # Select random intermediate nodes
            num_intermediates = random.randint(1, min(3, len(candidate_nodes)))
            intermediate_nodes = random.sample(candidate_nodes, num_intermediates)
            
            # Create route through intermediate nodes
            route = [self.start_node]
            current_node = self.start_node
            
            for intermediate in intermediate_nodes:
                try:
                    path = nx.shortest_path(self.graph, current_node, intermediate, weight='length')
                    route.extend(path[1:])  # Skip first node (already in route)
                    current_node = intermediate
                except nx.NetworkXNoPath:
                    continue
            
            # Return to start
            try:
                path_home = nx.shortest_path(self.graph, current_node, self.start_node, weight='length')
                route.extend(path_home[1:])
            except nx.NetworkXNoPath:
                pass
            
            return route
            
        except Exception as e:
            logger.warning(f"Failed to generate random route: {e}")
            return [self.start_node]
    
    def _find_high_value_nodes(self, top_k: int = 20) -> List[int]:
        """Find nodes with highest value function scores."""
        node_values = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if 'y' in node_data and 'x' in node_data:
                lat, lon = node_data['y'], node_data['x']
                value = self.value_function.get_value_at_coords(lat, lon)
                node_values.append((value, node))
        
        # Sort by value and return top k
        node_values.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in node_values[:top_k]]
    
    def _create_route_through_node(self, intermediate_node: int) -> Optional[List[int]]:
        """Create a route that goes through a specific intermediate node."""
        try:
            path_to = nx.shortest_path(self.graph, self.start_node, intermediate_node, weight='length')
            path_back = nx.shortest_path(self.graph, intermediate_node, self.start_node, weight='length')
            
            # Combine paths
            return path_to + path_back[1:]
        except nx.NetworkXNoPath:
            return None
    
    def _calculate_fitness(self, route: List[int]) -> float:
        """Calculate fitness score for a route."""
        if not route or len(route) < 2:
            return 0.0
        
        try:
            # Calculate route distance
            route_distance = sum(
                self.graph.get_edge_data(route[i], route[i + 1], {}).get('length', 0)
                for i in range(len(route) - 1)
            )
            
            # Distance fitness (closer to target is better)
            distance_error = abs(route_distance - self.target_distance)
            distance_fitness = max(0, 1 - distance_error / self.target_distance)
            
            # Value function fitness
            route_coords = []
            for node in route:
                node_data = self.graph.nodes[node]
                if 'y' in node_data and 'x' in node_data:
                    route_coords.append((node_data['y'], node_data['x']))
            
            if route_coords:
                value_fitness = self.value_function.get_path_value(route_coords)
            else:
                value_fitness = 0.5
            
            # Diversity fitness (unique nodes)
            diversity_fitness = len(set(route)) / len(route) if route else 0
            
            # Combined fitness
            fitness = (
                self.opt_config.distance_weight * distance_fitness +
                self.opt_config.value_weight * value_fitness +
                self.opt_config.diversity_weight * diversity_fitness
            )
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Error calculating fitness: {e}")
            return 0.0
    
    def _tournament_selection(self, fitness_scores: List[Tuple[float, List[int]]], tournament_size: int = 3) -> List[int]:
        """Tournament selection for genetic algorithm."""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        return max(tournament, key=lambda x: x[0])[1]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Crossover operation for genetic algorithm."""
        try:
            # Order crossover (OX) adapted for routes
            if len(parent1) <= 2 or len(parent2) <= 2:
                return parent1.copy()
            
            # Find common nodes (excluding start/end)
            common_nodes = set(parent1[1:-1]) & set(parent2[1:-1])
            
            if not common_nodes:
                return parent1.copy()
            
            # Select crossover points based on common nodes
            crossover_node = random.choice(list(common_nodes))
            
            # Split parent1 at crossover node
            split_idx = parent1.index(crossover_node)
            offspring = parent1[:split_idx + 1]
            
            # Add remaining path from parent2
            p2_start_idx = parent2.index(crossover_node)
            remaining_path = parent2[p2_start_idx + 1:]
            
            offspring.extend(remaining_path)
            
            return offspring
            
        except Exception:
            return parent1.copy()
    
    def _mutate(self, route: List[int]) -> List[int]:
        """Mutation operation for genetic algorithm."""
        if len(route) <= 2:
            return route
        
        try:
            mutated = route.copy()
            
            # Random mutation operations
            mutation_type = random.choice(['insert', 'remove', 'swap'])
            
            if mutation_type == 'insert' and len(route) < 20:
                # Insert a random nearby node
                insert_idx = random.randint(1, len(mutated) - 1)
                current_node = mutated[insert_idx]
                
                # Find neighbors of current node
                neighbors = list(self.graph.neighbors(current_node))
                if neighbors:
                    new_node = random.choice(neighbors)
                    mutated.insert(insert_idx, new_node)
            
            elif mutation_type == 'remove' and len(route) > 3:
                # Remove a random intermediate node
                remove_idx = random.randint(1, len(mutated) - 2)
                mutated.pop(remove_idx)
            
            elif mutation_type == 'swap' and len(route) > 3:
                # Swap two intermediate nodes
                idx1 = random.randint(1, len(mutated) - 2)
                idx2 = random.randint(1, len(mutated) - 2)
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
            
            return mutated
            
        except Exception:
            return route
    
    def _get_temperature(self, iteration: int) -> float:
        """Calculate temperature for simulated annealing."""
        progress = iteration / self.opt_config.max_iterations
        
        if self.opt_config.temperature_schedule == 'linear':
            return self.opt_config.initial_temperature * (1 - progress)
        elif self.opt_config.temperature_schedule == 'exponential':
            return self.opt_config.initial_temperature * (0.95 ** iteration)
        else:  # logarithmic
            return self.opt_config.initial_temperature / (1 + np.log(1 + iteration))
    
    def _generate_neighbor(self, route: List[int]) -> List[int]:
        """Generate a neighbor solution for simulated annealing."""
        return self._mutate(route)
    
    def _accept_neighbor(self, current_fitness: float, neighbor_fitness: float, temperature: float) -> bool:
        """Decide whether to accept a neighbor solution."""
        if neighbor_fitness > current_fitness:
            return True
        
        if temperature > 0:
            probability = np.exp((neighbor_fitness - current_fitness) / temperature)
            return random.random() < probability
        
        return False
    
    def _genetic_algorithm_with_population(self, initial_population: List[List[int]]) -> List[int]:
        """Run genetic algorithm with provided initial population."""
        # Simplified genetic algorithm for hybrid method
        population = initial_population
        
        for generation in range(self.opt_config.max_iterations):
            fitness_scores = [(self._calculate_fitness(route), route) for route in population]
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top half
            elite_count = len(population) // 2
            new_population = [route for _, route in fitness_scores[:elite_count]]
            
            # Generate offspring
            while len(new_population) < len(population):
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)
                offspring = self._crossover(parent1, parent2)
                
                if random.random() < self.opt_config.mutation_rate:
                    offspring = self._mutate(offspring)
                
                new_population.append(offspring)
            
            population = new_population
        
        # Return best route
        final_fitness = [(self._calculate_fitness(route), route) for route in population]
        return max(final_fitness, key=lambda x: x[0])[1]
    
    def _simulated_annealing_from_start(self, start_route: List[int]) -> List[int]:
        """Run simulated annealing starting from a given route."""
        current_route = start_route
        current_fitness = self._calculate_fitness(current_route)
        
        best_route = current_route.copy()
        best_fitness = current_fitness
        
        for iteration in range(self.opt_config.max_iterations):
            temperature = self._get_temperature(iteration)
            neighbor_route = self._generate_neighbor(current_route)
            neighbor_fitness = self._calculate_fitness(neighbor_route)
            
            if self._accept_neighbor(current_fitness, neighbor_fitness, temperature):
                current_route = neighbor_route
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_route = current_route.copy()
        
        return best_route