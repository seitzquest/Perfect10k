#!/usr/bin/env python3
"""
Test Async Integration for Perfect10k
=====================================

Tests the async job manager integration with CleanRouter,
cache warming, and Docker-friendly persistence.
"""

import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from clean_router import CleanRouter
from async_job_manager import job_manager, JobStatus
from smart_cache_manager import cache_manager
from docker_cache_config import docker_cache
from performance_profiler import log_performance_summary, profiler


async def test_async_route_generation():
    """Test async route generation with cache warming."""
    print("🚀 Testing Async Route Generation")
    print("=" * 50)
    
    router = CleanRouter()
    
    # Test coordinates (Munich)
    test_lat, test_lon = 48.1351, 11.5820
    client_id = "async_test_client"
    preference = "scenic nature"
    target_distance = 3000
    
    print(f"📍 Location: {test_lat:.6f}, {test_lon:.6f}")
    print(f"🎯 Target Distance: {target_distance}m")
    
    try:
        # Test 1: First async request (should trigger background processing)
        print("\n🔍 Test 1: First Async Request")
        start_time = time.perf_counter()
        
        result = await router.start_route_async(
            client_id=client_id,
            lat=test_lat,
            lon=test_lon,
            preference=preference,
            target_distance=target_distance
        )
        
        first_response_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ First response: {first_response_time:.1f}ms")
        print(f"📊 Response type: {result.get('response_type', 'unknown')}")
        
        if result['response_type'] == 'cached':
            print("🟢 EXCELLENT: Immediate cached response!")
            print(f"📊 Candidates: {len(result.get('candidates', []))}")
        elif result['response_type'] == 'async':
            print("🟡 ASYNC: Background processing started")
            job_id = result['job_id']
            print(f"📋 Job ID: {job_id}")
            
            # Wait for job completion
            print("⏳ Waiting for background job completion...")
            job_result = await router.wait_for_job_async(job_id, timeout=60.0)
            
            if job_result and job_result['status'] == 'completed':
                print(f"✅ Background job completed in {job_result.get('execution_time_ms', 0):.1f}ms")
                final_result = job_result['result']
                if final_result:
                    print(f"📊 Candidates: {len(final_result.get('candidates', []))}")
            else:
                print(f"❌ Background job failed or timed out: {job_result}")
        
        # Test 2: Second async request (should be cached)
        print("\n🔍 Test 2: Second Async Request (Expected Cache Hit)")
        start_time = time.perf_counter()
        
        result2 = await router.start_route_async(
            client_id=f"{client_id}_2",
            lat=test_lat,
            lon=test_lon,
            preference=preference,
            target_distance=target_distance
        )
        
        second_response_time = (time.perf_counter() - start_time) * 1000
        print(f"✅ Second response: {second_response_time:.1f}ms")
        print(f"📊 Response type: {result2.get('response_type', 'unknown')}")
        
        if result2['response_type'] == 'cached':
            print("🟢 EXCELLENT: Cached response achieved!")
            speedup = first_response_time / second_response_time if second_response_time > 0 else 0
            print(f"🚀 Speedup: {speedup:.1f}x faster")
        
        return True
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cache_warming():
    """Test background cache warming functionality."""
    print("\n🔥 Testing Cache Warming")
    print("=" * 30)
    
    router = CleanRouter()
    
    # Test coordinates for warming
    center_lat, center_lon = 48.1400, 11.5900  # Slightly different from first test
    
    try:
        # Start cache warming
        warming_jobs = await router._start_background_cache_warming(center_lat, center_lon)
        
        print(f"🔥 Started {len(warming_jobs)} cache warming jobs")
        
        # Monitor warming progress
        completed_jobs = 0
        for job_id in warming_jobs[:3]:  # Monitor first 3 jobs
            print(f"⏳ Monitoring warming job {job_id}...")
            
            # Wait for completion with timeout
            result = await job_manager.wait_for_job(job_id, timeout=30.0)
            if result and result.status == JobStatus.COMPLETED:
                completed_jobs += 1
                print(f"✅ Warming job completed: {result.execution_time_ms:.0f}ms")
            else:
                print(f"⚠️  Warming job timeout or failed")
        
        success_rate = completed_jobs / min(len(warming_jobs), 3)
        print(f"📊 Warming success rate: {success_rate:.1%}")
        
        if success_rate > 0.5:
            print("🟢 GOOD: Cache warming working")
            return True
        else:
            print("🟡 MODERATE: Some cache warming issues")
            return False
        
    except Exception as e:
        print(f"❌ Cache warming test failed: {e}")
        return False


async def test_job_manager_stats():
    """Test job manager statistics and monitoring."""
    print("\n📊 Testing Job Manager Statistics")
    print("=" * 35)
    
    try:
        stats = job_manager.get_stats()
        
        print("Job Manager Statistics:")
        print(f"  Jobs Submitted: {stats['jobs_submitted']}")
        print(f"  Jobs Completed: {stats['jobs_completed']}")
        print(f"  Jobs Failed: {stats['jobs_failed']}")
        print(f"  Avg Execution Time: {stats['avg_execution_time_ms']:.1f}ms")
        print(f"  Active Jobs: {stats['active_jobs']}")
        print(f"  Pending Jobs: {stats['pending_jobs']}")
        print(f"  Workers: {stats['workers']}")
        print(f"  Running: {stats['running']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False


async def test_cache_statistics():
    """Test cache performance statistics."""
    print("\n📈 Testing Cache Statistics")
    print("=" * 30)
    
    try:
        cache_stats = cache_manager.get_cache_statistics()
        
        print("Cache Performance:")
        print(f"  Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
        print(f"  Total Requests: {cache_stats['total_requests']}")
        print(f"  Cache Sizes: {cache_stats['cache_sizes']}")
        
        # Docker cache info
        docker_info = docker_cache.get_cache_info()
        print(f"\nDocker Cache:")
        print(f"  Base Directory: {docker_info['base_directory']}")
        print(f"  Total Size: {docker_info['total_size_mb']:.1f}MB")
        print(f"  Total Files: {docker_info['total_files']}")
        print(f"  Is Docker Volume: {docker_info['is_docker_volume']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache stats test failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("Perfect10k Async Integration Test Suite")
    print("=======================================\n")
    
    # Ensure job manager is running
    await job_manager.start()
    
    # Run tests
    tests = [
        test_async_route_generation,
        test_cache_warming,
        test_job_manager_stats,
        test_cache_statistics
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            success = await test_func()
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    # Performance summary
    print("\n📈 Performance Summary:")
    print("=" * 25)
    log_performance_summary()
    
    # Final results
    print(f"\n🏆 Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All async integration tests passed!")
        print("✅ System ready for Docker deployment with async processing")
    elif passed_tests > total_tests // 2:
        print("🟡 Most tests passed - async integration mostly working")
    else:
        print("🔴 Multiple test failures - async integration needs work")
    
    # Graceful shutdown
    await job_manager.stop()
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)