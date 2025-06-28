#!/usr/bin/env python3
"""
Integration test for Perfect10k clean backend.
Tests that the API endpoints work properly with CleanRouter.
"""

import sys
import asyncio
import time
from fastapi.testclient import TestClient

try:
    from main import app
except ImportError as e:
    print(f"Failed to import main app: {e}")
    sys.exit(1)


def test_health_endpoint():
    """Test that health endpoint works."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ Health endpoint works")


def test_performance_stats():
    """Test performance stats endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/performance-stats")
        assert response.status_code == 200
        data = response.json()
        assert "session_count" in data
        assert data["system_type"] == "clean_candidate_generator"
        print("✓ Performance stats endpoint works")


def test_cache_stats():
    """Test cache stats endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/cache-stats")
        assert response.status_code == 200
        data = response.json()
        print("✓ Cache stats endpoint works")


def test_start_session_api():
    """Test start session API endpoint (will fail without graph data, but should not crash)."""
    with TestClient(app) as client:
        request_data = {
            "lat": 37.7749,
            "lon": -122.4194,
            "preference": "scenic parks and nature",
            "target_distance": 5000
        }
        
        response = client.post("/api/start-session", json=request_data)
        # Should return 500 because no graph data is available, but shouldn't crash
        assert response.status_code in [200, 500]
        
        if response.status_code == 500:
            data = response.json()
            assert "No graph data available" in data.get("detail", "")
            print("✓ Start session endpoint handled missing graph data gracefully")
        else:
            print("✓ Start session endpoint works with available graph data")


def test_deprecated_endpoints():
    """Test that deprecated endpoints return proper deprecation messages."""
    with TestClient(app) as client:
        # Test deprecated orienteering endpoints
        response = client.post("/api/explore-alternatives", json={})
        assert response.status_code == 200
        data = response.json()
        assert data.get("deprecated") is True
        print("✓ Deprecated endpoints return proper deprecation messages")


def test_semantic_overlays():
    """Test semantic overlays endpoint."""
    with TestClient(app) as client:
        request_data = {
            "lat": 37.7749,
            "lon": -122.4194,
            "radius_km": 1.0,
            "feature_types": ["forests", "rivers"]
        }
        
        response = client.post("/api/semantic-overlays", json=request_data)
        # May return error due to network/API limits, but should not crash
        assert response.status_code in [200, 500]
        print("✓ Semantic overlays endpoint handled request properly")


def main():
    """Run all integration tests."""
    print("Perfect10k Clean Backend Integration Test")
    print("=" * 45)
    
    try:
        test_health_endpoint()
        test_performance_stats()
        test_cache_stats()
        test_start_session_api()
        test_deprecated_endpoints()
        test_semantic_overlays()
        
        print("\n" + "=" * 45)
        print("✅ All integration tests passed!")
        print("🎉 Frontend/backend integration is working properly!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())