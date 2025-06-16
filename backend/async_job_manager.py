"""
Async Job Manager for Perfect10k
Handles long-running tasks like semantic analysis in the background
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
from enum import Enum
import threading
from loguru import logger


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobInfo:
    job_id: str
    status: JobStatus
    progress: float = 0.0
    current_phase: str = ""
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    phases: list = None
    
    def to_dict(self):
        data = asdict(self)
        data['status'] = self.status.value
        return data


class AsyncJobManager:
    """Manages background jobs for long-running operations."""
    
    def __init__(self):
        self.jobs: Dict[str, JobInfo] = {}
        self.job_queue = asyncio.Queue()
        self.workers_started = False
        
    def create_job(self, job_type: str = "route_analysis") -> str:
        """Create a new background job and return job ID immediately."""
        job_id = str(uuid.uuid4())
        
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
            updated_at=time.time(),
            phases=[]
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created async job {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of a job."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        return job.to_dict()
    
    def update_job_progress(self, job_id: str, progress: float, phase: str = "", phases: list = None):
        """Update job progress and current phase."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job.progress = progress
        job.current_phase = phase
        job.updated_at = time.time()
        
        if phases:
            job.phases = phases
        
        logger.debug(f"Job {job_id} progress: {progress:.1f}% - {phase}")
    
    def complete_job(self, job_id: str, result: Dict):
        """Mark job as completed with result."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job.status = JobStatus.COMPLETED
        job.progress = 100.0
        job.result = result
        job.updated_at = time.time()
        
        logger.info(f"Job {job_id} completed successfully")
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job.status = JobStatus.FAILED
        job.error = error
        job.updated_at = time.time()
        
        logger.error(f"Job {job_id} failed: {error}")
    
    def start_job(self, job_id: str):
        """Mark job as running."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        job.status = JobStatus.RUNNING
        job.updated_at = time.time()
        
        logger.info(f"Job {job_id} started")
    
    def cleanup_old_jobs(self, max_age_hours: float = 24.0):
        """Clean up old completed/failed jobs."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                job.updated_at < cutoff_time):
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")
    
    def get_stats(self) -> Dict:
        """Get job manager statistics."""
        stats = {
            "total_jobs": len(self.jobs),
            "pending": 0,
            "running": 0, 
            "completed": 0,
            "failed": 0
        }
        
        for job in self.jobs.values():
            if job.status == JobStatus.PENDING:
                stats["pending"] += 1
            elif job.status == JobStatus.RUNNING:
                stats["running"] += 1
            elif job.status == JobStatus.COMPLETED:
                stats["completed"] += 1
            elif job.status == JobStatus.FAILED:
                stats["failed"] += 1
        
        return stats


# Global instance
job_manager = AsyncJobManager()


def run_route_analysis_job(job_id: str, route_builder, client_id: str, lat: float, lon: float, 
                          preference: str, target_distance: int):
    """Run route analysis in background thread."""
    try:
        job_manager.start_job(job_id)
        
        # Phase 1: Graph loading
        job_manager.update_job_progress(job_id, 10.0, "Loading street network")
        
        # This is the same logic as the sync version, but with progress updates
        session = route_builder.get_or_create_client_session(client_id, lat, lon, radius=6000)
        
        job_manager.update_job_progress(job_id, 30.0, "Route initialization")
        
        # Find start node
        import osmnx as ox
        start_node = ox.nearest_nodes(session.graph, lon, lat)
        
        # Create value function
        value_function = route_builder.semantic_matcher.create_value_function(preference)
        
        # Create route state
        from interactive_router import RouteState
        route_state = RouteState(
            start_node=start_node,
            start_location=(lat, lon),
            current_waypoints=[start_node],
            current_path=[start_node],
            total_distance=0.0,
            estimated_final_distance=0.0,
            target_distance=target_distance,
            preference=preference,
            value_function=value_function,
            used_edges=set()
        )
        
        session.active_route = route_state
        
        job_manager.update_job_progress(job_id, 50.0, "Analyzing natural features")
        
        # Generate candidates (this is the slow part)
        candidates = route_builder._generate_candidates_for_session(session, start_node)
        
        job_manager.update_job_progress(job_id, 90.0, "Finalizing results")
        
        # Prepare result
        result = {
            "session_id": client_id,
            "start_location": {
                "lat": session.graph.nodes[start_node]["y"],
                "lon": session.graph.nodes[start_node]["x"]
            },
            "candidates": [
                {
                    "node_id": c.node_id,
                    "lat": c.lat,
                    "lon": c.lon,
                    "value_score": c.value_score,
                    "distance": c.distance_from_current,
                    "estimated_completion": c.estimated_route_completion,
                    "explanation": c.explanation,
                    "semantic_scores": c.semantic_scores or {},
                    "semantic_details": c.semantic_details or "",
                    "score_breakdown": {
                        "forests": c.semantic_scores.get('forests', 0.0) if c.semantic_scores else 0.0,
                        "rivers": c.semantic_scores.get('rivers', 0.0) if c.semantic_scores else 0.0,
                        "lakes": c.semantic_scores.get('lakes', 0.0) if c.semantic_scores else 0.0,
                        "overall": c.value_score
                    }
                }
                for c in candidates
            ],
            "route_stats": {
                "current_distance": 0.0,
                "target_distance": target_distance,
                "progress": 0.0
            }
        }
        
        job_manager.complete_job(job_id, result)
        
    except Exception as e:
        logger.error(f"Route analysis job {job_id} failed: {e}")
        job_manager.fail_job(job_id, str(e))


def start_route_analysis_async(route_builder, client_id: str, lat: float, lon: float,
                              preference: str, target_distance: int) -> str:
    """Start route analysis job in background and return job ID immediately."""
    job_id = job_manager.create_job("route_analysis")
    
    # Start job in background thread
    thread = threading.Thread(
        target=run_route_analysis_job,
        args=(job_id, route_builder, client_id, lat, lon, preference, target_distance),
        daemon=True
    )
    thread.start()
    
    return job_id