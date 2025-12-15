#!/usr/bin/env python3
"""
Performance Profiling Script for Perfect10k
===========================================

Identifies bottlenecks in the current route generation system
to guide optimization efforts.
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

# Add the backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.clean_router import CleanRouter
from backend.performance_profiler import log_performance_summary, profiler


def profile_route_generation():
    """Profile a complete route generation cycle."""
    print("🔍 PERFORMANCE PROFILING - Perfect10k Route Generation")
    print("=" * 60)

    # Test coordinates (near Munich)
    test_locations = [
        (48.1351, 11.5820, "Munich Center"),
        (48.7758, 11.4217, "Munich North"),  # Should use same cache
        (48.0500, 11.6000, "Munich South"),  # Different area
    ]

    router = CleanRouter()

    for i, (lat, lon, location_name) in enumerate(test_locations):
        print(f"\n{'=' * 20} TEST {i + 1}: {location_name} {'=' * 20}")
        print(f"📍 Location: {lat:.6f}, {lon:.6f}")

        # Test parameters
        client_id = f"profile_client_{i + 1}"
        preference = "scenic nature"
        target_distance = 3000  # 3km routes

        try:
            print("\n🚀 Testing Route Start Performance...")
            start_time = time.perf_counter()

            # Profile the route start
            result = router.start_route(
                client_id=client_id,
                lat=lat,
                lon=lon,
                preference=preference,
                target_distance=target_distance,
            )

            total_time = (time.perf_counter() - start_time) * 1000

            print(f"✅ Route started in {total_time:.1f}ms")
            print(f"📊 Candidates found: {len(result['candidates'])}")
            print(
                f"🎯 Performance: {'🟢 EXCELLENT' if total_time < 1000 else '🟡 GOOD' if total_time < 5000 else '🔴 NEEDS OPTIMIZATION'}"
            )

            # Test waypoint addition if successful
            if result["candidates"] and len(result["candidates"]) > 0:
                print("\n🚀 Testing Add Waypoint Performance...")
                waypoint_start = time.perf_counter()

                first_candidate = result["candidates"][0]
                waypoint_result = router.add_waypoint(
                    client_id=client_id, node_id=first_candidate["node_id"]
                )

                waypoint_time = (time.perf_counter() - waypoint_start) * 1000
                print(f"✅ Waypoint added in {waypoint_time:.1f}ms")
                print(f"📊 New candidates: {len(waypoint_result['candidates'])}")
                print(
                    f"🎯 Performance: {'🟢 EXCELLENT' if waypoint_time < 500 else '🟡 GOOD' if waypoint_time < 2000 else '🔴 NEEDS OPTIMIZATION'}"
                )

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Performance summary
    print("\n" + "=" * 60)
    print("📈 DETAILED PERFORMANCE ANALYSIS")
    print("=" * 60)
    log_performance_summary()

    return True


def profile_with_cprofile():
    """Run detailed cProfile analysis."""
    print("\n🔬 DETAILED PROFILING WITH cProfile")
    print("=" * 50)

    # Create a profiler
    pr = cProfile.Profile()

    # Run profiling
    pr.enable()

    router = CleanRouter()
    try:
        router.start_route(
            client_id="detailed_profile_client",
            lat=48.1351,
            lon=11.5820,
            preference="scenic nature",
            target_distance=3000,
        )
    except Exception as e:
        print(f"Profiling encountered error: {e}")

    pr.disable()

    # Analyze results
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")

    print("\n🏆 TOP 20 MOST TIME-CONSUMING FUNCTIONS:")
    stats.print_stats(20)

    print("\n🔥 BOTTLENECK ANALYSIS - Functions taking >100ms:")
    stats.print_stats(".*", 20)

    # Save detailed report
    stats.dump_stats("performance_profile.prof")
    print("\n💾 Detailed profile saved to 'performance_profile.prof'")
    print("   View with: python -m pstats performance_profile.prof")


def analyze_bottlenecks():
    """Analyze and suggest optimizations based on profiling."""
    print("\n🎯 BOTTLENECK ANALYSIS & OPTIMIZATION SUGGESTIONS")
    print("=" * 60)

    stats = profiler.get_stats()

    if not stats:
        print("⚠️  No profiling data available. Run route generation first.")
        return

    print("📊 Operation Performance Analysis:")
    print("-" * 40)

    # Define performance targets
    targets = {
        "start_route": 1000,  # 1s for complete route start
        "load_graph_for_area": 50,  # 50ms for cached graph loading
        "generate_candidates": 200,  # 200ms for candidate generation
        "initialize_generator": 100,  # 100ms for generator init
    }

    recommendations = []

    for operation, data in sorted(stats.items(), key=lambda x: x[1]["avg_ms"], reverse=True):
        avg_ms = data["avg_ms"]
        target = targets.get(operation, 100)  # Default 100ms target

        if avg_ms > target * 10:
            status = "🔴 CRITICAL"
            factor = f"{avg_ms / target:.0f}x slower than target"
            if "graph" in operation.lower():
                recommendations.append(f"🚀 Implement graph caching for {operation}")
            elif "candidate" in operation.lower():
                recommendations.append(f"🎯 Optimize candidate generation in {operation}")
            elif "semantic" in operation.lower():
                recommendations.append(f"🧠 Cache semantic features for {operation}")
        elif avg_ms > target * 2:
            status = "🟡 SLOW"
            factor = f"{avg_ms / target:.1f}x slower than target"
        elif avg_ms > target:
            status = "🟠 MODERATE"
            factor = f"{avg_ms / target:.1f}x slower than target"
        else:
            status = "🟢 GOOD"
            factor = "within target"

        print(f"{status} {operation:<25} {avg_ms:>8.1f}ms ({factor})")

    print("\n🔧 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 40)

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ All operations are performing within acceptable ranges!")

    print("\n📋 NEXT STEPS:")
    print("-" * 40)
    print("1. 🏆 Focus on CRITICAL bottlenecks first")
    print("2. 🚀 Implement caching for graph loading")
    print("3. 🎯 Add probabilistic candidate selection")
    print("4. 🔄 Create async processing pipeline")
    print("5. 📊 Re-run profiling to measure improvements")


if __name__ == "__main__":
    print("Perfect10k Performance Profiling Suite")
    print("======================================\n")

    # Run basic performance profiling
    success = profile_route_generation()

    if success:
        # Run detailed profiling
        profile_with_cprofile()

        # Analyze and provide recommendations
        analyze_bottlenecks()

        print("\n🎉 Performance profiling completed!")
        print("📝 Use the analysis above to guide optimization efforts.")
        print("🚀 Focus on the highest impact optimizations first.")
    else:
        print("\n💥 Performance profiling failed!")
        print("🔧 Please check the errors and fix issues before profiling.")
        sys.exit(1)
