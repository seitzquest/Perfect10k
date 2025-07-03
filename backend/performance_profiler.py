#!/usr/bin/env python3
"""
Performance Profiler for Perfect10k Route Generation
===================================================

This module provides detailed timing analysis for route generation
to identify bottlenecks and optimize for <1s response times.
"""

import functools
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


class PerformanceProfiler:
    """Context manager and decorator for performance profiling."""

    def __init__(self):
        self.timings: dict[str, list[float]] = {}
        self.current_session = {}
        self.session_start = None

    def start_session(self, operation: str):
        """Start a new profiling session."""
        self.session_start = time.perf_counter()
        self.current_session = {
            'operation': operation,
            'start_time': self.session_start,
            'steps': []
        }
        logger.info(f"🔍 PROFILING: Starting {operation}")

    def step(self, step_name: str):
        """Record a step in the current session."""
        now = time.perf_counter()
        elapsed = (now - self.session_start) * 1000  # ms

        self.current_session['steps'].append({
            'name': step_name,
            'elapsed_ms': elapsed,
            'timestamp': now
        })

        logger.info(f"⏱️  STEP: {step_name} completed in {elapsed:.1f}ms (total: {elapsed:.1f}ms)")

    def end_session(self) -> dict[str, Any]:
        """End the current session and return timing data."""
        if not self.current_session:
            return {}

        total_time = (time.perf_counter() - self.session_start) * 1000
        self.current_session['total_ms'] = total_time

        operation = self.current_session['operation']
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(total_time)

        logger.info(f"🏁 PROFILING: {operation} completed in {total_time:.1f}ms")

        # Log step breakdown
        for i, step in enumerate(self.current_session['steps']):
            if i > 0:
                step_time = step['elapsed_ms'] - self.current_session['steps'][i-1]['elapsed_ms']
                logger.info(f"   └─ {step['name']}: {step_time:.1f}ms")
            else:
                logger.info(f"   └─ {step['name']}: {step['elapsed_ms']:.1f}ms")

        return self.current_session.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {}
        for operation, times in self.timings.items():
            stats[operation] = {
                'count': len(times),
                'avg_ms': sum(times) / len(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'last_ms': times[-1] if times else 0
            }
        return stats

    def save_profile(self, filename: str = None):
        """Save profiling data to file."""
        if not filename:
            filename = f"profile_{int(time.time())}.json"

        profile_data = {
            'timestamp': time.time(),
            'current_session': self.current_session,
            'stats': self.get_stats(),
            'raw_timings': self.timings
        }

        Path(filename).write_text(json.dumps(profile_data, indent=2))
        logger.info(f"💾 Profile saved to {filename}")


# Global profiler instance
profiler = PerformanceProfiler()


def profile_function(operation_name: str = None):
    """Decorator to profile individual functions."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"⚡ FUNCTION: {operation_name} took {elapsed:.1f}ms")

                # Add to global timings
                if operation_name not in profiler.timings:
                    profiler.timings[operation_name] = []
                profiler.timings[operation_name].append(elapsed)

        return wrapper
    return decorator


class ProfilerContext:
    """Context manager for step-by-step profiling."""

    def __init__(self, operation: str):
        self.operation = operation

    def __enter__(self):
        profiler.start_session(self.operation)
        return profiler

    def __exit__(self, exc_type, exc_val, exc_tb):
        profiler.end_session()


# Convenience functions
def profile_operation(operation: str):
    """Start profiling an operation."""
    return ProfilerContext(operation)


def step(step_name: str):
    """Record a profiling step."""
    profiler.step(step_name)


def get_performance_report() -> str:
    """Get a human-readable performance report."""
    stats = profiler.get_stats()

    if not stats:
        return "No profiling data available"

    report = ["🚀 PERFORMANCE REPORT", "=" * 50]

    # Sort by average time
    sorted_ops = sorted(stats.items(), key=lambda x: x[1]['avg_ms'], reverse=True)

    for operation, data in sorted_ops:
        avg_ms = data['avg_ms']
        status = "🔴 SLOW" if avg_ms > 1000 else "🟡 MODERATE" if avg_ms > 100 else "🟢 FAST"

        report.extend([
            f"\n{status} {operation}:",
            f"  Average: {avg_ms:.1f}ms",
            f"  Range: {data['min_ms']:.1f}ms - {data['max_ms']:.1f}ms",
            f"  Count: {data['count']} calls"
        ])

    # Performance targets
    report.extend([
        "\n🎯 PERFORMANCE TARGETS:",
        "  🟢 Target: <100ms per operation",
        "  🏆 Goal: <1000ms total route generation"
    ])

    return "\n".join(report)


# Integration with existing loguru logger
def log_performance_summary():
    """Log a performance summary."""
    report = get_performance_report()
    for line in report.split('\n'):
        logger.info(line)
