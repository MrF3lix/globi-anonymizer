## Globi

Anonymizes Swiss German recordings: the words of a target recording are re-spoken in the
voice of a donor recording, with the XTTS v2 model fine-tuned on Swiss German. It wraps the
[anonymizer](https://github.com/rm00cr/coqui-tts) in a gateway and GPU workers connected
through Redis.

```
client ──► gateway ──► Redis stream globi:tasks ──► worker ×N
```

One job anonymizes one recording. Jobs and their audio live in Redis for 24h, or until
deleted.

| Endpoint | |
|---|---|
| `POST /jobs` | multipart: `target_wav` (≤ 27s), `reference_wav` (the donor), `language`, optional `text`. Returns `202` with the job |
| `GET /jobs/{id}` | status (`queued`, `running`, `completed`, `failed`), the transcript used, error |
| `GET /jobs/{id}/result.wav` | the anonymized recording |
| `DELETE /jobs/{id}` | delete the job and its audio; a queued job is skipped |
| `GET /health` | live workers and queued jobs |

`language` is the dialect the output is spoken in: `ch_be`, `ch_bs`, `ch_gr`, `ch_in`,
`ch_os`, `ch_vs`, `ch_zh`, or `de` for Standard German. `text` is the Standard German
transcript of the target, which is what the model was trained on. Without it, the worker
transcribes the target with Whisper, which is the weak link on dialect. The output has the
length of the target whatever the transcript, so check the stored `transcript` to catch a
bad one.

```bash
curl -F target_wav=@interview.wav -F reference_wav=@donor.wav -F language=ch_zh http://localhost:8000/jobs
curl http://localhost:8000/jobs/<id>
curl -o anonymized.wav http://localhost:8000/jobs/<id>/result.wav
```

### Configuration

Hydra configs in `configs/`: `api.yaml` (server, queue cap, result TTL, upload limits),
`worker.yaml` (device, Whisper model, crash recovery), `model/` (checkpoint directories
and dialects) and `paths/`. Any value can be overridden on the command line, e.g.
`globi-worker device=cpu worker.whisper_model=large-v3`. `REDIS_URL` is read from the
environment (default `redis://localhost:6379/0`).

### Model

The worker expects two checkpoint folders in `/models` (`paths.model_dir`):

| Folder | Files |
|---|---|
| `GPT_XTTS_v2.0_SwissGPC_epoch_6_subset_7` | `model.pth`, `config.json`, `vocab.json` |
| `XTTS_v2.0_original_model_files` | `dvae.pth`, `mel_stats.pth` |

Only the GPT was fine-tuned, so the DVAE files come from the original XTTS v2
(`anonymize download-model` in the anonymizer repo, or Hugging Face `coqui/XTTS-v2`). On
startup the worker links the files into one directory and checks that the dialect tokens
are in the vocabulary.

### Run locally

Needs a Redis, e.g. `docker run -d -p 6379:6379 redis:7.4-alpine`. Use the same extras
for both, since `uv run` syncs the shared `.venv` to exactly the extras given:

```bash
uv run --extra api --extra worker globi-gateway
```

```bash
uv run --extra api --extra worker globi-worker paths.model_dir=/path/to/models
```

### Run in Docker (DGX Spark)

Requires the NVIDIA Container Toolkit. The checkpoint folders are mounted from `MODEL_DIR`
(default `/models`), not copied into the image. Settings in `.env` (see `.env.example`).

```bash
docker compose up -d --build
```

This starts Redis, the gateway (published on `PORT`) and `WORKERS` workers sharing the
GPU. Each worker holds XTTS twice in memory (the anonymizer builds a `GPTTrainer` next to
the model), so check GPU memory before adding workers:
`docker compose up -d --scale worker=3`.

Logs: `docker compose logs -f gateway worker`.
