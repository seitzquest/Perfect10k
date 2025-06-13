#!/usr/bin/env python3
"""
Quick performance test for the optimized routing system.
Tests cache performance and client session management.
"""

import time

from interactive_router import InteractiveRouteBuilder


def test_caching_performance():
    """Test that caching dramatically improves performance."""
    print("🚀 Testing Performance Optimizations\n")

    builder = InteractiveRouteBuilder()

    # Test location in San Francisco
    lat, lon = 37.7749, -122.4194
    preference = "scenic parks and nature"
    target_distance = 8000

    print("1. First request (will load OSM data and cache):")
    start_time = time.time()

    try:
        result1 = builder.start_route(
            client_id="test_client_1",
            lat=lat,
            lon=lon,
            preference=preference,
            target_distance=target_distance
        )
        first_duration = time.time() - start_time
        print(f"   ✅ First route started: {first_duration:.2f}s")
        print(f"   📊 Found {len(result1['candidates'])} candidates")

    except Exception as e:
        print(f"   ❌ First request failed: {e}")
        return

    # Test nearby location (should use cache)
    print("\n2. Second request (same area, should use cache):")
    start_time = time.time()

    try:
        result2 = builder.start_route(
            client_id="test_client_2",
            lat=lat + 0.01,  # Slightly different location
            lon=lon + 0.01,
            preference=preference,
            target_distance=target_distance
        )
        second_duration = time.time() - start_time
        print(f"   ✅ Second route started: {second_duration:.2f}s")
        print(f"   📊 Found {len(result2['candidates'])} candidates")

        # Calculate improvement
        improvement = (first_duration - second_duration) / first_duration * 100
        print(f"   🎯 Performance improvement: {improvement:.1f}%")

        if second_duration < 2.0:
            print("   ✨ Cache working perfectly! < 2 seconds")
        elif second_duration < 5.0:
            print("   👍 Good performance with cache")
        else:
            print("   ⚠️  Cache may not be working optimally")

    except Exception as e:
        print(f"   ❌ Second request failed: {e}")
        return

    # Test session reuse
    print("\n3. Testing session reuse:")
    print(f"   📊 Active client sessions: {len(builder.client_sessions)}")
    print(f"   💾 Cached graph areas: {len(builder.graph_cache)}")

    # Test same client, different location in same area
    start_time = time.time()
    try:
        result3 = builder.start_route(
            client_id="test_client_1",  # Same client as first request
            lat=lat + 0.005,  # Different location in same cached area
            lon=lon + 0.005,
            preference="quiet residential paths",  # Different preference
            target_distance=6000  # Different target
        )
        third_duration = time.time() - start_time
        print(f"   ✅ Session reuse: {third_duration:.2f}s")
        print(f"   📊 Found {len(result3['candidates'])} candidates")

        if third_duration < 1.0:
            print("   ⚡ Lightning fast! Session reuse working perfectly")

    except Exception as e:
        print(f"   ❌ Session reuse failed: {e}")

    print("\n🎉 Performance test complete!")
    print("Expected: First request ~30s (OSM download), subsequent < 2s (cached)")

if __name__ == "__main__":
    test_caching_performance()
