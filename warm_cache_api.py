#!/usr/bin/env python3
"""
Simple Cache Warming Script using API
=====================================

This script calls the backend API to warm cache for specific locations.
Perfect for Docker deployments where you want to avoid timeout issues.

Usage:
    python warm_cache_api.py 48.1351 11.5820  # Munich coordinates
    python warm_cache_api.py --city munich     # Predefined city
    python warm_cache_api.py --cities          # List available cities
    python warm_cache_api.py --all-cities      # Warm all predefined cities
"""

import argparse
import sys
import time

import requests

# Predefined city coordinates
CITIES = {
    "munich": (48.1351, 11.5820),
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "frankfurt": (50.1109, 8.6821),
    "cologne": (50.9375, 6.9603),
    "stuttgart": (48.7758, 9.1829),
    "dusseldorf": (51.2277, 6.7735),
    "leipzig": (51.3397, 12.3731),
    "dortmund": (51.5136, 7.4653),
    "nuremberg": (49.4521, 11.0767),
}


def call_warm_api(locations, city_names=None, base_url="http://localhost:8000"):
    """Call the warm-cache API endpoint."""
    try:
        response = requests.post(
            f"{base_url}/api/warm-cache",
            json={"locations": locations, "city_names": city_names},
            timeout=300,  # 5 minute timeout
        )

        if response.status_code == 200:
            return True, response.json()
        return False, f"HTTP {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return False, "Request timed out after 5 minutes"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect to backend. Is the server running?"
    except Exception as e:
        return False, f"Request failed: {e!s}"


def main():
    parser = argparse.ArgumentParser(
        description="Warm Perfect10k cache using API calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s 48.1351 11.5820           # Warm Munich by coordinates
  %(prog)s --city munich             # Warm Munich by name
  %(prog)s --city munich berlin      # Warm multiple cities
  %(prog)s --all-cities              # Warm all predefined cities
  %(prog)s --cities                  # List available cities
  %(prog)s --url http://server:8000  # Use different server URL

Available cities: {", ".join(CITIES.keys())}
        """,
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Backend server URL (default: http://localhost:8000)",
    )
    parser.add_argument("coordinates", nargs="*", help="Latitude and longitude pairs")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--city", nargs="+", choices=CITIES.keys(), help="City names to warm")
    group.add_argument("--all-cities", action="store_true", help="Warm all predefined cities")
    group.add_argument("--cities", action="store_true", help="List available cities")

    args = parser.parse_args()

    if args.cities:
        print("Available cities:")
        for city, (lat, lon) in CITIES.items():
            print(f"  {city:12} - {lat:.4f}, {lon:.4f}")
        return 0

    # Prepare locations and city names
    locations = []
    city_names = []

    if args.all_cities:
        for city, (lat, lon) in CITIES.items():
            locations.append([lat, lon])
            city_names.append(city)
    elif args.city:
        for city in args.city:
            lat, lon = CITIES[city]
            locations.append([lat, lon])
            city_names.append(city)
    elif args.coordinates:
        if len(args.coordinates) % 2 != 0:
            print("❌ Error: Coordinates must be provided in lat,lon pairs")
            return 1

        for i in range(0, len(args.coordinates), 2):
            try:
                lat = float(args.coordinates[i])
                lon = float(args.coordinates[i + 1])
                locations.append([lat, lon])
                city_names.append(None)
            except ValueError:
                print(
                    f"❌ Error: Invalid coordinates: {args.coordinates[i]} {args.coordinates[i + 1]}"
                )
                return 1

    if not locations:
        print("❌ Error: No locations specified. Use coordinates, --city, or --all-cities")
        print("For help: python warm_cache_api.py --help")
        return 1

    print(f"🚀 Starting cache warming for {len(locations)} location(s) via API...")
    print(f"🌐 Backend URL: {args.url}")

    start_time = time.time()

    # Call the API
    success, result = call_warm_api(locations, [name for name in city_names if name], args.url)

    elapsed = time.time() - start_time

    if success and isinstance(result, dict):
        print(f"✅ API call completed in {elapsed:.1f}s")
        print(f"📊 {result.get('message', 'Unknown status')}")
        print(f"🔄 Total jobs started: {result.get('total_jobs_started', 0)}")
        print(
            f"📍 Successful locations: {result.get('successful_locations', 0)}/{result.get('locations_processed', 0)}"
        )

        if result.get("note"):
            print(f"Info: {result['note']}")

        # Show individual results
        print("\nDetailed Results:")
        for res in result.get("results", []):
            location = res.get("location", {})
            city_name = (
                res.get("city_name")
                or f"({location.get('lat', 0):.4f}, {location.get('lon', 0):.4f})"
            )
            status_emoji = {
                "started": "🔄",
                "already_cached": "✅",
                "loaded": "✅",
                "error": "❌",
            }.get(res.get("status", "unknown"), "?")

            print(f"  {status_emoji} {city_name:15} - {res.get('message', 'Unknown')}")

        successful = result.get("successful_locations", 0)
        if successful == len(locations):
            print("\n🎉 All cache warming operations completed successfully!")
            return 0
        print(f"\n⚠️  {len(locations) - successful} locations failed")
        return 0
    print(f"❌ API call failed after {elapsed:.1f}s: {result}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
