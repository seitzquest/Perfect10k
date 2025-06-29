#!/usr/bin/env python3
"""Test probabilistic variety in candidate selection."""

import sys
import os
import time
import json
from collections import Counter

# Add backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_candidate_generator import CleanCandidateGenerator
import networkx as nx

def create_test_graph():
    """Create a simple test graph for testing."""
    G = nx.MultiGraph()
    
    # Add nodes in a grid pattern around a center point
    center_lat, center_lon = 47.6062, -122.3321  # Seattle coordinates
    
    # Create nodes in a 5x5 grid
    node_id = 1
    for i in range(-2, 3):
        for j in range(-2, 3):
            lat = center_lat + i * 0.01  # ~1km spacing
            lon = center_lon + j * 0.01
            G.add_node(node_id, x=lon, y=lat)
            node_id += 1
    
    # Add some edges to make it a connected graph
    for node in G.nodes():
        if node < 24:  # Connect to next node
            G.add_edge(node, node + 1)
    
    return G

def test_probabilistic_variety():
    """Test that probabilistic selection provides variety."""
    print("Testing probabilistic variety in candidate selection...")
    
    # Create test graph
    graph = create_test_graph()
    generator = CleanCandidateGenerator(graph)
    
    # Initialize generator
    if not generator.initialize():
        print("Failed to initialize generator")
        return False
    
    # Test parameters
    from_lat, from_lon = 47.6062, -122.3321
    target_distance = 1000  # 1km
    preference = "parks and nature"
    num_runs = 10
    
    print(f"Running {num_runs} candidate generations...")
    
    # Run multiple times and collect results
    all_results = []
    times = []
    
    for run in range(num_runs):
        start_time = time.time()
        
        result = generator.generate_candidates(
            from_lat, from_lon, target_distance, preference
        )
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # ms
        
        # Collect candidate node IDs
        candidate_ids = [c.node_id for c in result.candidates]
        all_results.append(candidate_ids)
        
        print(f"Run {run + 1}: {len(result.candidates)} candidates in {times[-1]:.1f}ms - {candidate_ids}")
    
    # Analyze variety
    print("\nVariety Analysis:")
    
    # Count unique candidates across all runs
    all_candidate_ids = [cid for run_results in all_results for cid in run_results]
    candidate_counter = Counter(all_candidate_ids)
    unique_candidates = len(candidate_counter)
    
    print(f"Total unique candidates seen: {unique_candidates}")
    print(f"Most common candidates: {candidate_counter.most_common(5)}")
    
    # Check if we get different results
    unique_result_sets = len(set(tuple(sorted(result)) for result in all_results))
    print(f"Unique result combinations: {unique_result_sets} out of {num_runs} runs")
    
    # Performance analysis
    avg_time = sum(times) / len(times)
    max_time = max(times)
    print(f"\nPerformance:")
    print(f"Average time: {avg_time:.1f}ms")
    print(f"Maximum time: {max_time:.1f}ms")
    print(f"Target: 50ms - {'✓ PASS' if max_time <= 50 else '✗ FAIL'}")
    
    # Variety check
    variety_score = unique_result_sets / num_runs
    print(f"Variety score: {variety_score:.2f} (higher is better)")
    print(f"Variety check: {'✓ PASS' if variety_score > 0.5 else '✗ FAIL'}")
    
    return variety_score > 0.5 and max_time <= 50

def test_deterministic_comparison():
    """Compare deterministic vs probabilistic modes."""
    print("\nTesting deterministic vs probabilistic comparison...")
    
    graph = create_test_graph()
    generator = CleanCandidateGenerator(graph)
    generator.initialize()
    
    from_lat, from_lon = 47.6062, -122.3321
    target_distance = 1000
    preference = "parks and nature"
    
    # Test deterministic mode
    generator.enable_probabilistic_selection = False
    det_results = []
    for _ in range(3):
        result = generator.generate_candidates(from_lat, from_lon, target_distance, preference)
        det_results.append([c.node_id for c in result.candidates])
    
    # Test probabilistic mode
    generator.enable_probabilistic_selection = True
    prob_results = []
    for _ in range(3):
        result = generator.generate_candidates(from_lat, from_lon, target_distance, preference)
        prob_results.append([c.node_id for c in result.candidates])
    
    print("Deterministic results:")
    for i, result in enumerate(det_results):
        print(f"  Run {i+1}: {result}")
    
    print("Probabilistic results:")
    for i, result in enumerate(prob_results):
        print(f"  Run {i+1}: {result}")
    
    # Check deterministic consistency
    det_consistent = all(result == det_results[0] for result in det_results)
    print(f"Deterministic consistency: {'✓ PASS' if det_consistent else '✗ FAIL'}")
    
    # Check probabilistic variety
    unique_prob_results = len(set(tuple(sorted(result)) for result in prob_results))
    prob_variety = unique_prob_results / len(prob_results)
    print(f"Probabilistic variety: {prob_variety:.2f} - {'✓ PASS' if prob_variety > 0.3 else '✗ FAIL'}")
    
    return det_consistent and prob_variety > 0.3

if __name__ == "__main__":
    print("Starting probabilistic variety tests...\n")
    
    success = True
    
    try:
        # Test 1: Basic probabilistic variety
        success &= test_probabilistic_variety()
        
        # Test 2: Deterministic vs probabilistic comparison
        success &= test_deterministic_comparison()
        
        print(f"\nOverall result: {'✓ ALL TESTS PASSED' if success else '✗ SOME TESTS FAILED'}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)