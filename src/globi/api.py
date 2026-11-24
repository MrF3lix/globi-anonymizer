import io
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import hydra
import redis.asyncio as redis
import rootutils
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from omegaconf import DictConfig
from pydantic import BaseModel

from globi import jobs


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


class Language(BaseModel):
    code: str
    name: str


class ModelInfo(BaseModel):
    id: str
    name: str
    languages: list[Language]
    workers: int


def create_app(cfg: DictConfig) -> FastAPI:
    languages = list(cfg.model.languages)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.r = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
        app.state.rb = redis.Redis.from_url(cfg.redis_url)  # binary values (audio)
        try:
            await app.state.r.xgroup_create(jobs.STREAM, jobs.GROUP, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        yield
        await app.state.r.aclose()
        await app.state.rb.aclose()

    app = FastAPI(title="globi", lifespan=lifespan)

    async def read_audio(upload: UploadFile, max_seconds: float) -> bytes:
        """A clean 16-bit mono copy of an uploaded recording."""
        try:
            audio, sr = sf.read(io.BytesIO(await upload.read()), dtype="float32")
        except Exception:
            raise HTTPException(422, f"{upload.filename} could not be decoded as audio")
        if len(audio) == 0:
            raise HTTPException(422, f"{upload.filename} is empty")
        if len(audio) / sr > max_seconds:
            raise HTTPException(422, f"{upload.filename} exceeds {max_seconds:g} seconds")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    async def queued() -> int:
        """Tasks not yet handed to a worker."""
        groups = await app.state.r.xinfo_groups(jobs.STREAM)
        return next((g["lag"] or 0 for g in groups if g["name"] == jobs.GROUP), 0)

    async def load_job(job_id: str) -> dict:
        job = await app.state.r.hgetall(jobs.job_key(job_id))
        if not job:
            raise HTTPException(404, "job not found (unknown id or expired)")
        return job

    @app.post("/jobs", status_code=202)
    async def submit_job(
        target_wav: UploadFile = File(..., description="Recording to anonymize; its words are kept"),
        reference_wav: UploadFile = File(..., description="Donor voice; the output sounds like this speaker"),
        language: str = Form(..., description=f"Dialect of the output: {', '.join(languages)}"),
        text: str | None = Form(None, description="Standard German transcript of target_wav; Whisper is used if omitted"),
        seed: int | None = Form(None, ge=0, lt=2**32, description="Random seed; random if omitted. "
                                                                   "The same inputs and seed reproduce a result"),
    ):
        if language not in languages:
            raise HTTPException(422, f"language must be one of {languages}")
        target = await read_audio(target_wav, cfg.limits.max_target_seconds)
        reference = await read_audio(reference_wav, cfg.limits.max_reference_seconds)
        if await queued() >= cfg.jobs.max_queued:
            raise HTTPException(429, "queue is full, try again later", headers={"Retry-After": "30"})

        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "language": language, "text": (text or "").strip(),
               "seed": seed if seed is not None else secrets.randbelow(2**31), "created_at": time.time()}
        ttl = cfg.jobs.result_ttl_seconds
        pipe = app.state.rb.pipeline()
        pipe.hset(jobs.job_key(job_id), mapping=job)
        pipe.expire(jobs.job_key(job_id), ttl)
        pipe.set(jobs.job_key(job_id, "target"), target, ex=ttl)
        pipe.set(jobs.job_key(job_id, "reference"), reference, ex=ttl)
        # the job exists before any worker can see its task
        pipe.xadd(jobs.STREAM, {"job_id": job_id})
        await pipe.execute()
        return job

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return await load_job(job_id)

    @app.get("/jobs/{job_id}/result.wav", response_class=Response,
             responses={200: {"content": {"audio/wav": {}}}})
    async def get_result(job_id: str):
        job = await load_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, f"job is {job['status']}")
        audio = await app.state.rb.get(jobs.job_key(job_id, "result"))
        if audio is None:
            raise HTTPException(404, "result expired")
        return Response(content=audio, media_type="audio/wav")

    @app.delete("/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str):
        """Delete a job and its audio; a queued job is skipped by the workers."""
        await app.state.r.delete(*(jobs.job_key(job_id, part) for part in (None, "target", "reference", "result")))

    async def worker_count(model: str = "*") -> int:
        return len([k async for k in app.state.r.scan_iter(match=jobs.worker_pattern(model))])

    @app.get("/models", response_model=list[ModelInfo])
    async def models():
        """Models registered by the workers, with their dialects and live worker count."""
        entries = await app.state.r.hgetall(jobs.MODELS_KEY)
        infos = [json.loads(info) for info in entries.values()]
        return [ModelInfo(id=m["id"], name=m["name"],
                          languages=[Language(code=c, name=n) for c, n in m["languages"].items()],
                          workers=await worker_count(m["id"]))
                for m in infos]

    @app.get("/health")
    async def health():
        return {"status": "ok", "workers": await worker_count(), "queued": await queued()}

    return app


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="api.yaml")
def main(cfg: DictConfig):
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)

if __name__ == "__main__":
    main()
