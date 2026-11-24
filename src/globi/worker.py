import io
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import hydra
import redis
import rootutils
import soundfile as sf
from omegaconf import DictConfig

from globi import jobs
from globi.engine import Engine


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")
log = logging.getLogger(__name__)


class Worker:
    def __init__(self, cfg: DictConfig, engine: Engine):
        self.cfg = cfg.worker
        self.model_cfg = cfg.model
        self.model = cfg.model.id
        self.engine = engine
        self.name = f"{socket.gethostname()}-{os.getpid()}"
        self.r = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
        self.rb = redis.Redis.from_url(cfg.redis_url)  # binary values (audio)
        self._stop = threading.Event()

    def _register_model(self):
        # the gateway learns which models are loaded, and their dialects, from this entry
        info = {"id": self.model, "name": self.model_cfg.name, "languages": dict(self.model_cfg.languages)}
        self.r.hset(jobs.MODELS_KEY, self.model, json.dumps(info))

    def _heartbeat_loop(self):
        # separate thread, so a long job doesn't make the worker look dead in /health
        ttl = self.cfg.heartbeat_ttl_seconds
        while not self._stop.is_set():
            self.r.set(jobs.worker_key(self.model, self.name), time.time(), ex=ttl)
            self._stop.wait(ttl / 3)

    def _next_task(self):
        _, claimed, _ = self.r.xautoclaim(jobs.STREAM, jobs.GROUP, self.name,
                                          min_idle_time=self.cfg.claim_idle_seconds * 1000, count=1)
        if claimed:
            return claimed[0]
        resp = self.r.xreadgroup(jobs.GROUP, self.name, {jobs.STREAM: ">"}, count=1, block=2000)
        return resp[0][1][0] if resp else None

    def _process(self, job_id: str):
        key = jobs.job_key(job_id)
        if not self.r.exists(key):  # deleted or expired
            return
        if self.r.hincrby(key, "attempts", 1) > self.cfg.max_attempts:
            self.r.hset(key, mapping={"status": "failed", "error": "worker crashed repeatedly on this job"})
            return

        job = self.r.hgetall(key)
        started = time.time()
        self.r.hset(key, mapping={"status": "running", "worker": self.name, "started_at": started})
        try:
            target = self.rb.get(jobs.job_key(job_id, "target"))
            reference = self.rb.get(jobs.job_key(job_id, "reference"))
            if target is None or reference is None:
                raise RuntimeError("input recordings expired")
            wav, sr, transcript = self.engine.anonymize(target, reference, job["language"], job.get("text") or None)

            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
            ttl = self.r.ttl(key)
            if ttl < 0:  # deleted while running
                return
            self.rb.set(jobs.job_key(job_id, "result"), buf.getvalue(), ex=ttl)
            self.r.hset(key, mapping={"status": "completed", "transcript": transcript,
                                      "duration": len(wav) / sr, "finished_at": time.time()})
            log.info(f"Job {job_id}: {len(wav) / sr:.1f}s audio in {time.time() - started:.1f}s")
        except Exception as e:
            log.exception(f"Job {job_id} failed")
            if self.r.exists(key):
                self.r.hset(key, mapping={"status": "failed", "error": str(e), "finished_at": time.time()})

    def run(self):
        try:
            self.r.xgroup_create(jobs.STREAM, jobs.GROUP, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._register_model()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        log.info(f"Worker {self.name} ready")
        try:
            while True:
                try:
                    task = self._next_task()
                    if task:
                        task_id, fields = task
                        if fields:  # empty when the entry was trimmed from the stream
                            self._process(fields["job_id"])
                        self.r.xack(jobs.STREAM, jobs.GROUP, task_id)
                except (redis.ConnectionError, redis.TimeoutError):
                    # Redis restarting; reloading the model would be far slower than waiting
                    log.warning("Lost connection to Redis, retrying", exc_info=True)
                    time.sleep(2)
        finally:
            self._stop.set()
            self.r.delete(jobs.worker_key(self.model, self.name))


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="worker.yaml")
def main(cfg: DictConfig):
    # docker stop sends SIGTERM; exit normally so the heartbeat is removed
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    engine = Engine(cfg.model, cfg.device, cfg.worker.whisper_model)
    Worker(cfg, engine).run()

if __name__ == "__main__":
    main()
