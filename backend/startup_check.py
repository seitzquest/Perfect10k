#!/usr/bin/env python3
"""
Startup verification for Perfect10k Clean Backend
Checks that all components are working correctly.
"""

import sys
from pathlib import Path


def check_imports():
    """Verify all required modules can be imported."""
    print("🔍 Checking imports...")

    try:
        from clean_candidate_generator import CleanCandidateGenerator  # noqa: F401
        from clean_router import CleanRouter  # noqa: F401
        from core.spatial_tile_storage import SpatialTileStorage  # noqa: F401
        from semantic_overlays import SemanticOverlayManager  # noqa: F401

        print("✅ All core modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_storage():
    """Verify storage directories exist."""
    print("🗂️  Checking storage...")

    storage_dirs = [Path("storage/tiles"), Path("cache/overlays")]

    for dir_path in storage_dirs:
        if not dir_path.exists():
            print(f"⚠️  Creating missing directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)

    print("✅ Storage directories ready")
    return True


def check_components():
    """Test that components can be instantiated."""
    print("⚙️  Testing component instantiation...")

    try:
        from clean_router import CleanRouter
        from semantic_overlays import SemanticOverlayManager

        # Test semantic overlay manager
        overlay_manager = SemanticOverlayManager()

        # Test clean router
        CleanRouter(overlay_manager)

        print("✅ Components instantiate successfully")
        return True
    except Exception as e:
        print(f"❌ Component error: {e}")
        return False


def main():
    print("🚀 Perfect10k Clean Backend Startup Check")
    print("=" * 50)

    checks = [check_imports, check_storage, check_components]

    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()

    if all_passed:
        print("🎉 All startup checks passed!")
        print("✅ Clean backend is ready for use")
        return 0
    else:
        print("❌ Some startup checks failed")
        print("🔧 Please fix the issues above before starting the server")
        return 1


if __name__ == "__main__":
    sys.exit(main())
