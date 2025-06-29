#!/usr/bin/env python3
"""
Test Smart Cache Integration
============================

Quick test to validate that the smart cache manager integration
is working and providing performance improvements.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clean_router import CleanRouter
from smart_cache_manager import cache_manager
from performance_profiler import profiler, log_performance_summary

def test_cache_performance():
    """Test cache performance improvements."""
    print("🚀 Testing Smart Cache Integration")
    print("=" * 50)
    
    router = CleanRouter()
    
    # Test coordinates (Munich)
    test_lat, test_lon = 48.1351, 11.5820
    client_id = "cache_test_client"
    preference = "scenic nature"
    target_distance = 3000
    
    print(f"📍 Test Location: {test_lat:.6f}, {test_lon:.6f}")
    print(f"🎯 Target Distance: {target_distance}m")
    
    # First run - should be slow (cache miss)
    print("\n🔍 First Run (Expected Cache Miss):")
    start_time = time.perf_counter()
    
    try:
        result1 = router.start_route(
            client_id=f"{client_id}_1",
            lat=test_lat,
            lon=test_lon,
            preference=preference,
            target_distance=target_distance
        )
        
        first_run_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ First run: {first_run_time:.1f}ms")
        print(f"📊 Candidates: {len(result1['candidates'])}")
        
    except Exception as e:
        print(f"❌ First run failed: {e}")
        return False
    
    # Second run - should be much faster (cache hit)
    print("\n🚀 Second Run (Expected Cache Hit):")
    start_time = time.perf_counter()
    
    try:
        result2 = router.start_route(
            client_id=f"{client_id}_2",
            lat=test_lat,
            lon=test_lon,
            preference=preference,
            target_distance=target_distance
        )
        
        second_run_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ Second run: {second_run_time:.1f}ms")
        print(f"📊 Candidates: {len(result2['candidates'])}")
        
        # Calculate improvement
        if first_run_time > 0:
            speedup = first_run_time / second_run_time
            improvement = ((first_run_time - second_run_time) / first_run_time) * 100
            print(f"🚀 Speedup: {speedup:.1f}x faster ({improvement:.1f}% improvement)")
            
            if speedup > 2:
                print("🟢 EXCELLENT: Significant cache performance improvement!")
            elif speedup > 1.5:
                print("🟡 GOOD: Moderate cache performance improvement")
            else:
                print("🔴 WARNING: Limited cache performance improvement")
        
    except Exception as e:
        print(f"❌ Second run failed: {e}")
        return False
    
    # Cache statistics
    print("\n📊 Cache Performance Statistics:")
    print("-" * 30)
    cache_stats = cache_manager.get_cache_statistics()
    
    print(f"Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
    print(f"Total Requests: {cache_stats['total_requests']}")
    print(f"Cache Sizes: {cache_stats['cache_sizes']}")
    
    # Performance summary
    print("\n📈 Detailed Performance Analysis:")
    print("-" * 40)
    log_performance_summary()
    
    return True

def test_variety_preservation():
    """Test that caching preserves variety in results."""
    print("\n🎲 Testing Variety Preservation")
    print("=" * 30)
    
    router = CleanRouter()
    
    test_lat, test_lon = 48.1351, 11.5820
    preference = "scenic nature"
    target_distance = 3000
    
    candidates_sets = []
    
    # Generate multiple results
    for i in range(3):
        try:
            result = router.start_route(
                client_id=f"variety_test_{i}",
                lat=test_lat,
                lon=test_lon,
                preference=preference,
                target_distance=target_distance
            )
            
            candidate_ids = {c['node_id'] for c in result['candidates']}
            candidates_sets.append(candidate_ids)
            print(f"Run {i+1}: {len(candidate_ids)} candidates")
            
        except Exception as e:
            print(f"❌ Variety test run {i+1} failed: {e}")
            return False
    
    # Analyze variety
    if len(candidates_sets) >= 2:
        overlap1_2 = len(candidates_sets[0] & candidates_sets[1])
        total1 = len(candidates_sets[0])
        overlap_percent = (overlap1_2 / total1) * 100 if total1 > 0 else 0
        
        print(f"📊 Overlap between runs: {overlap_percent:.1f}%")
        
        if overlap_percent < 80:
            print("🟢 EXCELLENT: Good variety preserved!")
        elif overlap_percent < 90:
            print("🟡 GOOD: Some variety preserved")
        else:
            print("🔴 WARNING: Results may be too deterministic")
    
    return True

if __name__ == "__main__":
    print("Smart Cache Integration Test")
    print("============================\n")
    
    success = test_cache_performance()
    
    if success:
        variety_success = test_variety_preservation()
        
        if variety_success:
            print("\n🎉 Cache integration test completed successfully!")
            print("📝 Check the performance improvements above.")
        else:
            print("\n⚠️  Cache integration working but variety needs attention.")
    else:
        print("\n💥 Cache integration test failed!")
        print("🔧 Please check the errors and fix issues.")
        sys.exit(1)