#!/usr/bin/env python3
"""
Simple test for Perfect10k clean backend components.
Tests that the core components work together.
"""

import sys
import time

try:
    from clean_router import CleanRouter
    from semantic_overlays import SemanticOverlayManager
    print("✓ All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_router_creation():
    """Test that CleanRouter can be created."""
    try:
        router = CleanRouter(semantic_overlay_manager=None)
        print("✓ CleanRouter created successfully")
        
        # Test basic methods
        stats = router.get_statistics()
        assert isinstance(stats, dict)
        print("✓ Router statistics method works")
        
        # Test session cleanup
        router.cleanup_old_sessions(24.0)
        print("✓ Session cleanup method works")
        
        return True
    except Exception as e:
        print(f"❌ Router creation failed: {e}")
        return False


def test_semantic_overlay_manager():
    """Test that SemanticOverlayManager can be created."""
    try:
        overlay_manager = SemanticOverlayManager()
        print("✓ SemanticOverlayManager created successfully")
        
        # Test basic methods  
        cache_stats = overlay_manager.get_cache_stats()
        assert isinstance(cache_stats, dict)
        print("✓ Overlay manager cache stats work")
        
        return True
    except Exception as e:
        print(f"❌ SemanticOverlayManager creation failed: {e}")
        return False


def test_integration():
    """Test that CleanRouter works with SemanticOverlayManager."""
    try:
        overlay_manager = SemanticOverlayManager()
        router = CleanRouter(semantic_overlay_manager=overlay_manager)
        print("✓ CleanRouter + SemanticOverlayManager integration works")
        
        # Test that router has the overlay manager
        assert router.semantic_overlay_manager is not None
        print("✓ Router correctly stores overlay manager")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Perfect10k Clean Backend Component Test")
    print("=" * 40)
    
    tests = [
        test_router_creation,
        test_semantic_overlay_manager,
        test_integration
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Passed: {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        print("🎉 All component tests passed!")
        print("✅ Clean backend is ready for use!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())