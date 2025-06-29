#!/usr/bin/env python3
"""
Cache Management CLI for Perfect10k
===================================

Command-line interface for managing cache in Docker deployments.
Provides tools for export, import, warming, and monitoring.
"""

import asyncio
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from docker_cache_config import docker_cache
from smart_cache_manager import cache_manager  
from async_job_manager import job_manager
from clean_router import CleanRouter
from loguru import logger


async def cmd_info():
    """Show cache information."""
    print("📊 Perfect10k Cache Information")
    print("=" * 40)
    
    # Docker cache info
    docker_info = docker_cache.get_cache_info()
    print(f"Base Directory: {docker_info['base_directory']}")
    print(f"Total Size: {docker_info['total_size_mb']:.1f}MB")
    print(f"Total Files: {docker_info['total_files']}")
    print(f"Is Docker Volume: {docker_info['is_docker_volume']}")
    
    print("\nDirectory Breakdown:")
    for name, info in docker_info['directories'].items():
        print(f"  {name}: {info['size_mb']:.1f}MB ({info['file_count']} files)")
    
    # Smart cache stats
    cache_stats = cache_manager.get_cache_statistics()
    print(f"\nSmart Cache Performance:")
    print(f"  Hit Rate: {cache_stats['overall_hit_rate']:.1%}")
    print(f"  Total Requests: {cache_stats['total_requests']}")
    
    # Job manager stats
    if job_manager.running:
        job_stats = job_manager.get_stats()
        print(f"\nJob Manager:")
        print(f"  Jobs Completed: {job_stats['jobs_completed']}")
        print(f"  Avg Execution Time: {job_stats['avg_execution_time_ms']:.1f}ms")
        print(f"  Active Jobs: {job_stats['active_jobs']}")


async def cmd_export(args):
    """Export cache for backup."""
    print(f"📦 Exporting cache...")
    
    export_name = args.name if args.name else None
    export_path = docker_cache.export_cache(export_name)
    
    print(f"✅ Cache exported to: {export_path}")
    return export_path


async def cmd_import(args):
    """Import cache from backup."""
    if not Path(args.path).exists():
        print(f"❌ Import file not found: {args.path}")
        return False
    
    print(f"📥 Importing cache from {args.path}")
    success = docker_cache.import_cache(args.path, overwrite=args.overwrite)
    
    if success:
        print("✅ Cache imported successfully")
    else:
        print("❌ Cache import failed")
    
    return success


async def cmd_clear(args):
    """Clear cache."""
    cache_type = args.type if args.type else None
    
    if cache_type:
        print(f"🧹 Clearing {cache_type} cache...")
    else:
        print("🧹 Clearing all cache...")
        confirm = input("This will delete all cached data. Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled")
            return False
    
    success = docker_cache.clear_cache(cache_type)
    
    if success:
        print("✅ Cache cleared successfully")
    else:
        print("❌ Cache clear failed")
    
    return success


async def cmd_warm(args):
    """Warm cache for specific locations."""
    print("🔥 Starting cache warming...")
    
    # Parse locations
    locations = []
    if args.locations:
        for loc_str in args.locations:
            try:
                lat, lon = map(float, loc_str.split(','))
                locations.append((lat, lon))
            except ValueError:
                print(f"❌ Invalid location format: {loc_str} (expected: lat,lon)")
                return False
    else:
        # Default popular locations
        locations = [
            (48.1351, 11.5820),  # Munich
            (52.5200, 13.4050),  # Berlin  
            (53.5511, 9.9937),   # Hamburg
            (50.1109, 8.6821),   # Frankfurt
            (50.9375, 6.9603),   # Cologne
        ]
    
    # Start job manager if not running
    await job_manager.start()
    
    router = CleanRouter()
    warming_jobs = []
    
    for lat, lon in locations:
        print(f"🔥 Warming cache for ({lat:.4f}, {lon:.4f})")
        jobs = await router._start_background_cache_warming(lat, lon)
        warming_jobs.extend(jobs)
    
    print(f"⏳ Waiting for {len(warming_jobs)} warming jobs to complete...")
    
    # Monitor progress
    completed = 0
    for job_id in warming_jobs:
        result = await job_manager.wait_for_job(job_id, timeout=120.0)
        if result and result.status.value == 'completed':
            completed += 1
            print(f"✅ Job completed ({completed}/{len(warming_jobs)})")
        else:
            print(f"⚠️  Job timeout or failed ({completed}/{len(warming_jobs)})")
    
    success_rate = completed / len(warming_jobs) if warming_jobs else 0
    print(f"📊 Warming completed: {success_rate:.1%} success rate")
    
    await job_manager.stop()
    return success_rate > 0.5


async def cmd_monitor():
    """Monitor cache and job activity."""
    print("📺 Monitoring Perfect10k Cache (Ctrl+C to stop)")
    print("=" * 50)
    
    await job_manager.start()
    
    try:
        while True:
            # Clear screen
            print("\033[2J\033[H")
            
            await cmd_info()
            
            print(f"\n🔄 Live Monitoring (refreshing every 5s)")
            print("-" * 30)
            
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    finally:
        await job_manager.stop()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Perfect10k Cache Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s info                                 # Show cache information
  %(prog)s export --name backup_20231201       # Export cache with name
  %(prog)s import --path backup.tar.gz         # Import cache from file
  %(prog)s clear --type graphs                 # Clear specific cache type
  %(prog)s warm --locations "48.1,11.6"        # Warm specific location
  %(prog)s monitor                             # Monitor cache activity
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Info command
    subparsers.add_parser('info', help='Show cache information')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export cache for backup')
    export_parser.add_argument('--name', help='Export name (default: timestamp)')
    
    # Import command  
    import_parser = subparsers.add_parser('import', help='Import cache from backup')
    import_parser.add_argument('path', help='Path to cache backup file')
    import_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing cache')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--type', choices=['graphs', 'smart_cache', 'jobs'], help='Specific cache type to clear')
    
    # Warm command
    warm_parser = subparsers.add_parser('warm', help='Warm cache for locations')
    warm_parser.add_argument('--locations', nargs='+', help='Locations as "lat,lon" (default: popular cities)')
    
    # Monitor command
    subparsers.add_parser('monitor', help='Monitor cache activity')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run appropriate command
    if args.command == 'info':
        asyncio.run(cmd_info())
    elif args.command == 'export':
        asyncio.run(cmd_export(args))
    elif args.command == 'import':
        asyncio.run(cmd_import(args))
    elif args.command == 'clear':
        asyncio.run(cmd_clear(args))
    elif args.command == 'warm':
        asyncio.run(cmd_warm(args))
    elif args.command == 'monitor':
        asyncio.run(cmd_monitor())


if __name__ == "__main__":
    main()