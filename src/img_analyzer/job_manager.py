"""
Job Manager — Tracks background processing jobs for the /process endpoint.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Job:
    job_id: str
    status: str = "processing"  # "processing", "completed", "failed"
    total_properties: int = 0
    processed_properties: int = 0
    error: str | None = None

    @property
    def progress(self) -> str:
        if self.total_properties == 0:
            return ""
        return f"{self.processed_properties}/{self.total_properties} properties"


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def create_job(self, total_properties: int) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, total_properties=total_properties)
        self._jobs[job_id] = job
        logger.info(f"Job {job_id} created: {total_properties} properties")
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.processed_properties += 1

    def complete_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = "completed"
            logger.info(f"Job {job_id} completed: {job.processed_properties}/{job.total_properties}")

    def fail_job(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = "failed"
            job.error = error
            logger.error(f"Job {job_id} failed: {error}")


# Global singleton
job_manager = JobManager()
