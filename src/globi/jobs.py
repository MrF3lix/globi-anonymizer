"""Redis layout shared by the gateway and the workers. One job anonymizes one recording.

    globi:job:{id}             hash    status, language, text, transcript, worker, error, timestamps
    globi:job:{id}:target      bytes   recording to anonymize (16-bit mono wav)
    globi:job:{id}:reference   bytes   donor voice (16-bit mono wav)
    globi:job:{id}:result      bytes   anonymized wav
    globi:tasks                stream  one entry per job, consumer group "workers"
    globi:worker:{name}        string  heartbeat with TTL, one per live worker
"""

STREAM = "globi:tasks"
GROUP = "workers"


def job_key(job_id: str, part: str | None = None) -> str:
    return f"globi:job:{job_id}" + (f":{part}" if part else "")

def worker_key(name: str) -> str:
    return f"globi:worker:{name}"

WORKER_PATTERN = worker_key("*")
