#!/usr/bin/env python3
"""
Docker Cache Configuration for Perfect10k
==========================================

Ensures cache is properly mounted and exportable for Docker deployments.
Provides utilities for cache import/export and volume management.
"""

import os
import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
import time
from io import BytesIO


class DockerCacheConfig:
    """Manages cache configuration for Docker deployment."""
    
    def __init__(self, base_cache_dir: str = "/app/cache"):
        """
        Initialize Docker cache configuration.
        
        Args:
            base_cache_dir: Base cache directory (should be mounted as Docker volume)
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.cache_dirs = {
            'graphs': self.base_cache_dir / 'graphs',
            'smart_cache': self.base_cache_dir / 'smart_cache', 
            'jobs': self.base_cache_dir / 'jobs',
            'exports': self.base_cache_dir / 'exports'
        }
        
        # Ensure all cache directories exist
        for cache_dir in self.cache_dirs.values():
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🐳 Docker cache configured at {base_cache_dir}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        info = {
            'base_directory': str(self.base_cache_dir),
            'directories': {},
            'total_size_mb': 0,
            'total_files': 0,
            'is_docker_volume': self._is_docker_volume()
        }
        
        for name, path in self.cache_dirs.items():
            if path.exists():
                size_mb, file_count = self._get_directory_stats(path)
                info['directories'][name] = {
                    'path': str(path),
                    'size_mb': size_mb,
                    'file_count': file_count,
                    'last_modified': path.stat().st_mtime if path.exists() else None
                }
                info['total_size_mb'] += size_mb
                info['total_files'] += file_count
            else:
                info['directories'][name] = {
                    'path': str(path),
                    'size_mb': 0,
                    'file_count': 0,
                    'last_modified': None
                }
        
        return info
    
    def export_cache(self, export_name: Optional[str] = None) -> str:
        """
        Export entire cache to a tarball for backup/transfer.
        
        Args:
            export_name: Optional export name, defaults to timestamp
            
        Returns:
            Path to exported tarball
        """
        if export_name is None:
            export_name = f"perfect10k_cache_{int(time.time())}"
        
        export_path = self.cache_dirs['exports'] / f"{export_name}.tar.gz"
        
        logger.info(f"📦 Exporting cache to {export_path}")
        
        with tarfile.open(export_path, 'w:gz') as tar:
            # Export each cache directory
            for name, path in self.cache_dirs.items():
                if name == 'exports':  # Don't export the exports directory
                    continue
                    
                if path.exists():
                    tar.add(path, arcname=name)
                    logger.debug(f"Added {name} directory to export")
            
            # Add metadata
            metadata = {
                'export_name': export_name,
                'export_time': time.time(),
                'export_time_iso': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'cache_info': self.get_cache_info(),
                'version': '1.0'
            }
            
            metadata_json = json.dumps(metadata, indent=2)
            info = tarfile.TarInfo('metadata.json')
            info.size = len(metadata_json.encode())
            tar.addfile(info, fileobj=BytesIO(metadata_json.encode()))
        
        export_size_mb = export_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Cache exported successfully: {export_size_mb:.1f}MB")
        
        return str(export_path)
    
    def import_cache(self, import_path: str, overwrite: bool = False) -> bool:
        """
        Import cache from a tarball.
        
        Args:
            import_path: Path to cache tarball
            overwrite: Whether to overwrite existing cache
            
        Returns:
            True if import successful
        """
        import_path = Path(import_path)
        if not import_path.exists():
            logger.error(f"Import file not found: {import_path}")
            return False
        
        logger.info(f"📥 Importing cache from {import_path}")
        
        try:
            with tarfile.open(import_path, 'r:gz') as tar:
                # Extract metadata first
                try:
                    metadata_file = tar.extractfile('metadata.json')
                    if metadata_file:
                        metadata = json.load(metadata_file)
                        logger.info(f"Importing cache: {metadata.get('export_name', 'unknown')}")
                        logger.info(f"Export date: {metadata.get('export_time_iso', 'unknown')}")
                except:
                    logger.warning("No metadata found in import, proceeding anyway")
                
                # Extract cache directories
                for name in self.cache_dirs.keys():
                    if name == 'exports':  # Don't import exports
                        continue
                    
                    target_dir = self.cache_dirs[name]
                    
                    # Check if directory exists and handle overwrite
                    if target_dir.exists() and not overwrite:
                        if any(target_dir.iterdir()):  # Directory not empty
                            logger.warning(f"Cache directory {name} exists and not empty, skipping (use overwrite=True to replace)")
                            continue
                    
                    # Extract directory
                    try:
                        for member in tar.getmembers():
                            if member.name.startswith(f"{name}/"):
                                tar.extract(member, self.base_cache_dir)
                        logger.info(f"Imported {name} cache directory")
                    except Exception as e:
                        logger.warning(f"Failed to import {name}: {e}")
            
            logger.info("✅ Cache import completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache import failed: {e}")
            return False
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """
        Clear cache (for development/testing).
        
        Args:
            cache_type: Specific cache type to clear, or None for all
            
        Returns:
            True if successful
        """
        if cache_type:
            if cache_type not in self.cache_dirs:
                logger.error(f"Unknown cache type: {cache_type}")
                return False
            
            cache_dir = self.cache_dirs[cache_type]
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"🧹 Cleared {cache_type} cache")
            return True
        else:
            # Clear all cache (except exports)
            for name, cache_dir in self.cache_dirs.items():
                if name == 'exports':
                    continue
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("🧹 Cleared all cache directories")
            return True
    
    def setup_docker_volumes(self) -> Dict[str, str]:
        """
        Get Docker volume configuration for docker-compose.yml
        
        Returns:
            Dictionary of volume mappings
        """
        return {
            'cache_volume': f"{self.base_cache_dir}:/app/cache"
        }
    
    def _is_docker_volume(self) -> bool:
        """Check if cache directory is mounted as Docker volume."""
        try:
            # Check if mounted (simple heuristic)
            stat = self.base_cache_dir.stat()
            return stat.st_dev != Path('/').stat().st_dev
        except:
            return False
    
    def _get_directory_stats(self, directory: Path) -> tuple[float, int]:
        """Get directory size in MB and file count."""
        total_size = 0
        file_count = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return total_size / (1024 * 1024), file_count


# Global Docker cache configuration
docker_cache = DockerCacheConfig()


def setup_cache_for_docker():
    """Setup cache directories for Docker deployment."""
    return docker_cache.get_cache_info()


def export_cache_for_backup(name: Optional[str] = None) -> str:
    """Export cache for backup."""
    return docker_cache.export_cache(name)


def import_cache_from_backup(path: str, overwrite: bool = False) -> bool:
    """Import cache from backup."""
    return docker_cache.import_cache(path, overwrite)