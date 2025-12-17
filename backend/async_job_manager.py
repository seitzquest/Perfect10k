#!/usr/bin/env python3
"""
Async Job Manager for Perfect10k
================================

High-performance async job system that enables:
1. Immediate responses with cached data
2. Background cache warming and route processing
3. Multi-client concurrent support
4. Progressive loading with real-time updates

Designed for Docker deployment with exportable cache.
"""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from backend.performance_profiler import profile_function


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class JobResult:
    """Job execution result with comprehensive tracking."""

    job_id: str
    status: JobStatus
    result: Any = None
    error: str | None = None
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    execution_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AsyncJob:
    """Async job definition with priority and retry logic."""

    job_id: str
    job_type: str
    job_function: Callable | str  # Function or function name
    args: tuple
    kwargs: dict
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 0
    timeout_seconds: int = 300
    created_at: float = field(default_factory=time.time)
    client_id: str | None = None


class AsyncJobManager:
    """
    High-performance async job manager optimized for Perfect10k routing.

    Features:
    - Priority-based job queuing
    - Background cache warming
    - Multi-client support
    - Docker-friendly persistence
    - Real-time progress tracking
    """

    def __init__(self, max_workers: int = 1, persistence_dir: str = "cache/jobs"):
        self.max_workers = max_workers
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        # Create bounded thread pool for sync operations to prevent resource exhaustion
        import concurrent.futures

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, 2)),  # Cap at 2 threads for Pi
            thread_name_prefix="AsyncJobManager",
        )

        # Job storage
        self.jobs: dict[str, AsyncJob] = {}
        self.results: dict[str, JobResult] = {}

        # Async infrastructure
        self.job_queues = {
            JobPriority.CRITICAL: asyncio.Queue(),
            JobPriority.HIGH: asyncio.Queue(),
            JobPriority.NORMAL: asyncio.Queue(),
            JobPriority.LOW: asyncio.Queue(),
        }

        self.workers: list[asyncio.Task] = []
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Performance tracking
        self.stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "avg_execution_time_ms": 0.0,
            "cache_warming_jobs": 0,
            "route_generation_jobs": 0,
        }

        # Background tasks
        self.background_tasks: set[asyncio.Task] = set()

        logger.info(
            f"🚀 AsyncJobManager initialized (workers: {max_workers}, persistence: {persistence_dir})"
        )

    async def start(self):
        """Start the async job manager with worker pool."""
        if self.running:
            return

        self.running = True
        self._shutdown_event.clear()

        # Start priority-based workers
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._priority_worker(f"worker-{i}"))
            self.workers.append(worker)

        # Start background maintenance
        maintenance_task = asyncio.create_task(self._background_maintenance())
        self.background_tasks.add(maintenance_task)

        # Load persisted jobs
        await self._load_persisted_jobs()

        logger.info(f"✅ AsyncJobManager started with {self.max_workers} workers")

    async def stop(self):
        """Stop the job manager gracefully with persistence."""
        if not self.running:
            return

        logger.info("🛑 Stopping AsyncJobManager...")
        self.running = False
        self._shutdown_event.set()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Cancel workers
        for worker in self.workers:
            worker.cancel()

        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.workers, *self.background_tasks, return_exceptions=True),
                timeout=30.0,
            )
        except TimeoutError:
            logger.warning("Graceful shutdown timeout, forcing stop")

        # Persist pending jobs
        await self._persist_jobs()

        # Clean up thread pool
        self.thread_pool.shutdown(wait=True)

        self.workers.clear()
        self.background_tasks.clear()

        logger.info("🛑 AsyncJobManager stopped")

    def submit_job(
        self,
        job_type: str,
        job_function: Callable,
        *args,
        priority: JobPriority = JobPriority.NORMAL,
        client_id: str | None = None,
        **kwargs,
    ) -> str:
        """Submit a job for async execution."""
        job_id = str(uuid.uuid4())

        job = AsyncJob(
            job_id=job_id,
            job_type=job_type,
            job_function=job_function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            client_id=client_id,
        )

        # Create initial result
        result = JobResult(job_id=job_id, status=JobStatus.PENDING)

        self.jobs[job_id] = job
        self.results[job_id] = result

        # Add to priority queue
        try:
            self.job_queues[priority].put_nowait(job)
            self.stats["jobs_submitted"] += 1

            # Track job types
            if "cache_warm" in job_type:
                self.stats["cache_warming_jobs"] += 1
            elif "route" in job_type:
                self.stats["route_generation_jobs"] += 1

            logger.info(f"📥 Job {job_id} ({job_type}) submitted (priority: {priority.name})")

        except asyncio.QueueFull:
            logger.error(f"❌ Job queue full, rejecting job {job_id}")
            result.status = JobStatus.FAILED
            result.error = "Job queue full"

        return job_id

    def submit_background_job(self, job_type: str, job_function: Callable, *args, **kwargs) -> str:
        """Submit a low-priority background job for cache warming."""
        return self.submit_job(
            f"background_{job_type}", job_function, *args, priority=JobPriority.LOW, **kwargs
        )

    def get_job_status(self, job_id: str) -> JobResult | None:
        """Get current job status and result."""
        return self.results.get(job_id)

    def is_job_complete(self, job_id: str) -> bool:
        """Check if job is complete (success or failure)."""
        result = self.results.get(job_id)
        if not result:
            return False
        return result.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]

    async def wait_for_job(self, job_id: str, timeout: float = 30.0) -> JobResult | None:
        """Wait for job completion with adaptive polling."""
        start_time = time.time()
        poll_interval = 0.1  # Start with 100ms
        max_poll_interval = 2.0  # Cap at 2 seconds

        while time.time() - start_time < timeout and self.running:
            if self.is_job_complete(job_id):
                return self.results.get(job_id)

            await asyncio.sleep(poll_interval)

            # Gradually increase polling interval to reduce CPU usage
            poll_interval = min(poll_interval * 1.2, max_poll_interval)

        # Timeout
        result = self.results.get(job_id)
        if result and result.status == JobStatus.RUNNING:
            result.status = JobStatus.CANCELLED
            result.error = "Timeout waiting for completion"
        return result

    async def get_job_updates(self, job_id: str):
        """Async generator for real-time job updates with adaptive polling."""
        last_status = None
        start_time = time.time()
        poll_interval = 0.5  # Start with 500ms
        max_poll_interval = 3.0  # Cap at 3 seconds

        while time.time() - start_time < 300 and self.running:  # 5 minute max
            result = self.results.get(job_id)
            if not result:
                break

            # Yield updates when status changes
            if result.status != last_status:
                yield result
                last_status = result.status
                poll_interval = 0.5  # Reset to fast polling on status change

                if self.is_job_complete(job_id):
                    break

            await asyncio.sleep(poll_interval)

            # Gradually increase polling interval to reduce resource usage
            poll_interval = min(poll_interval * 1.3, max_poll_interval)

    async def _priority_worker(self, worker_name: str):
        """Worker that processes jobs by priority."""
        logger.info(f"👷 Priority worker {worker_name} started")

        while self.running:
            try:
                # Check queues by priority (highest first)
                job = None
                for priority in [
                    JobPriority.CRITICAL,
                    JobPriority.HIGH,
                    JobPriority.NORMAL,
                    JobPriority.LOW,
                ]:
                    try:
                        job = self.job_queues[priority].get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue

                if job is None:
                    # No jobs available, wait a bit
                    await asyncio.sleep(0.1)
                    continue

                # Process job
                await self._execute_job(worker_name, job)

            except asyncio.CancelledError:
                logger.info(f"👷 Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error

        logger.info(f"👷 Worker {worker_name} stopped")

    @profile_function("async_execute_job")
    async def _execute_job(self, worker_name: str, job: AsyncJob):
        """Execute a single job with comprehensive error handling."""
        job_id = job.job_id
        result = self.results[job_id]

        try:
            logger.info(f"🔄 Worker {worker_name} executing job {job_id} ({job.job_type})")

            # Mark as running
            result.status = JobStatus.RUNNING
            result.started_at = time.time()

            start_time = time.perf_counter()

            # Execute job function
            if not callable(job.job_function):
                raise ValueError(f"job_function must be callable, got {type(job.job_function)}")

            if asyncio.iscoroutinefunction(job.job_function):
                job_result = await job.job_function(*job.args, **job.kwargs)
            else:
                # Run sync function in bounded thread pool to avoid resource exhaustion
                loop = asyncio.get_event_loop()

                # Create a proper callable wrapper
                def sync_wrapper():
                    if not callable(job.job_function):
                        raise ValueError(
                            f"job_function must be callable, got {type(job.job_function)}"
                        )
                    return job.job_function(*job.args, **job.kwargs)

                job_result = await loop.run_in_executor(self.thread_pool, sync_wrapper)

            # Success
            execution_time = (time.perf_counter() - start_time) * 1000

            result.status = JobStatus.COMPLETED
            result.result = job_result
            result.completed_at = time.time()
            result.execution_time_ms = execution_time
            result.progress = 1.0

            # Update stats
            self.stats["jobs_completed"] += 1
            self.stats["avg_execution_time_ms"] = (
                self.stats["avg_execution_time_ms"] * (self.stats["jobs_completed"] - 1)
                + execution_time
            ) / self.stats["jobs_completed"]

            logger.info(f"✅ Job {job_id} completed in {execution_time:.1f}ms")

        except asyncio.CancelledError:
            result.status = JobStatus.CANCELLED
            result.error = "Job cancelled"
            result.completed_at = time.time()
            logger.warning(f"⚠️  Job {job_id} cancelled")

        except Exception as e:
            # Failure
            result.status = JobStatus.FAILED
            result.error = str(e)
            result.completed_at = time.time()

            self.stats["jobs_failed"] += 1

            logger.error(f"❌ Job {job_id} failed: {e}")

    async def _background_maintenance(self):
        """Background maintenance tasks."""
        while self.running:
            try:
                # Clean up old jobs every 5 minutes
                await asyncio.sleep(300)
                await self._cleanup_old_jobs()

                # Persist job state every 10 minutes
                await asyncio.sleep(300)
                await self._persist_jobs()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background maintenance error: {e}")

    async def _cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = time.time() - (max_age_hours * 3600)

        old_jobs = []
        for job_id, result in self.results.items():
            if (result.completed_at and result.completed_at < cutoff_time) or (
                result.created_at < cutoff_time and self.is_job_complete(job_id)
            ):
                old_jobs.append(job_id)

        for job_id in old_jobs:
            self.jobs.pop(job_id, None)
            self.results.pop(job_id, None)

        if old_jobs:
            logger.info(f"🧹 Cleaned up {len(old_jobs)} old jobs")

    async def _persist_jobs(self):
        """Persist job state for Docker container restarts."""
        try:
            state_file = self.persistence_dir / "job_state.json"
            backup_file = self.persistence_dir / "job_state.json.backup"
            temp_file = self.persistence_dir / "job_state.json.tmp"

            # Only persist pending/running jobs
            persistent_jobs = {}
            persistent_results = {}

            for job_id, job in self.jobs.items():
                result = self.results[job_id]
                if result.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                    persistent_jobs[job_id] = {
                        "job_id": job.job_id,
                        "job_type": job.job_type,
                        "args": job.args,
                        "kwargs": job.kwargs,
                        "priority": job.priority.value,
                        "created_at": job.created_at,
                        "client_id": job.client_id,
                    }
                    persistent_results[job_id] = asdict(result)

            state = {
                "jobs": persistent_jobs,
                "results": persistent_results,
                "stats": self.stats,
                "timestamp": time.time(),
            }

            # Atomic write with backup
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2)

            # Create backup of existing file
            if state_file.exists():
                state_file.rename(backup_file)

            # Move temp file to final location
            temp_file.rename(state_file)

            logger.debug(f"💾 Persisted {len(persistent_jobs)} jobs to {state_file}")

        except Exception as e:
            logger.error(f"Failed to persist job state: {e}")

    async def _load_persisted_jobs(self):
        """Load persisted job state from Docker volume."""
        state_file = self.persistence_dir / "job_state.json"
        backup_file = self.persistence_dir / "job_state.json.backup"

        try:
            if not state_file.exists():
                return

            # Try to load the main state file
            try:
                with open(state_file) as f:
                    state = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Main job state file corrupted: {e}")

                # Try backup file
                if backup_file.exists():
                    logger.info("Attempting to restore from backup...")
                    with open(backup_file) as f:
                        state = json.load(f)
                    logger.info("✅ Successfully restored from backup")
                else:
                    logger.warning("No backup file available, starting fresh")
                    return

            # Restore stats
            self.stats.update(state.get("stats", {}))

            # Restore pending jobs (skip running ones - they're stale)
            restored_count = 0
            for job_id, _job_data in state.get("jobs", {}).items():
                result_data = state.get("results", {}).get(job_id)
                if result_data and result_data["status"] == JobStatus.PENDING.value:
                    # Restore job (note: function reference is lost, will need to handle this)
                    # For now, mark as failed and log for manual retry
                    result = JobResult(**result_data)
                    result.status = JobStatus.FAILED
                    result.error = "Job lost during container restart - manual retry needed"
                    self.results[job_id] = result
                    restored_count += 1

            if restored_count > 0:
                logger.info(f"📥 Restored {restored_count} jobs from persistence")

        except Exception as e:
            logger.error(f"Failed to load persisted jobs: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive job manager statistics."""
        active_jobs = len([r for r in self.results.values() if r.status == JobStatus.RUNNING])
        pending_jobs = len([r for r in self.results.values() if r.status == JobStatus.PENDING])
        queue_sizes = {priority.name: queue.qsize() for priority, queue in self.job_queues.items()}

        return {
            **self.stats,
            "active_jobs": active_jobs,
            "pending_jobs": pending_jobs,
            "queue_sizes": queue_sizes,
            "workers": len(self.workers),
            "running": self.running,
        }


# Global job manager instance
job_manager = AsyncJobManager()


# Convenience functions for route operations
async def submit_route_job(
    operation_name: str,
    operation_func: Callable,
    *args,
    priority: JobPriority = JobPriority.NORMAL,
    client_id: str | None = None,
    **kwargs,
) -> str:
    """Submit a route operation as async job."""
    return job_manager.submit_job(
        f"route_{operation_name}",
        operation_func,
        *args,
        priority=priority,
        client_id=client_id,
        **kwargs,
    )


async def submit_cache_warming_job(
    area_lat: float, area_lon: float, warming_func: Callable, **kwargs
) -> str:
    """Submit a cache warming job for an area."""
    return job_manager.submit_background_job("cache_warm_area", warming_func, area_lat, area_lon)


async def get_job_result(job_id: str, timeout: float = 30.0) -> Any | None:
    """Get job result with timeout."""
    result = await job_manager.wait_for_job(job_id, timeout)
    if result and result.status == JobStatus.COMPLETED:
        return result.result
    return None


# Auto-start functionality
async def ensure_job_manager_running():
    """Ensure job manager is running (Docker-friendly)."""
    if not job_manager.running:
        await job_manager.start()


# Graceful shutdown for Docker
async def shutdown_job_manager():
    """Graceful shutdown for Docker container stops."""
    await job_manager.stop()
