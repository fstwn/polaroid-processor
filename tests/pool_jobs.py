"""Jobs for the pool tests: a module of its own, so that a worker process imports it without importing pytest."""
import time


def slow_job(job: dict) -> dict:
    time.sleep(job["t"])
    return {"key": str(job["i"]), "path": "", "time_s": job["t"]}
