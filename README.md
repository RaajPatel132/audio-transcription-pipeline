# audio-transcription-pipeline

Repository: https://github.com/RaajPatel132/audio-transcription-pipeline

I built a pipeline that takes an audio file and returns a transcript with a start and end time for every segment. It runs as a command-line tool and as an HTTP service. Long files are cut at quiet moments, transcribed in parallel chunks and stitched back into one timeline. A job that is stopped or crashes resumes from the chunks it already finished. Speech-to-text is Whisper through faster-whisper, on the CPU. There is no API key, no GPU and no system ffmpeg.

Contents: [Quick start](#quick-start) · [What you get](#what-you-get) · [How it works](#how-it-works) · [Key decisions](#key-decisions) · [Production design](#production-design) · [Testing](#testing) · [Measured results](#measured-results) · [Known limitations](#known-limitations) · [What I would do next](#what-i-would-do-next) · [Layout and commands](#project-layout-and-commands)

## Quick start

You need [uv](https://docs.astral.sh/uv/) (it fetches Python 3.12 if you do not have it) or Docker. Nothing else: no ffmpeg, no GPU, no key. The only network use is a one-time download of the Whisper model (about 140 MB for the default `base.en`).

Local:

```
make setup          # create the environment from the lockfile
make test           # 269 tests, offline, about 5 seconds
make demo-mock      # the whole pipeline with a fake transcriber, no model needed
make fetch-model    # download base.en once
make demo           # transcribe samples/garden.mp3 and print it as SRT
```

Any file, any format, any output:

```
uv run transcribe path/to/recording.m4a --format vtt --output recording.vtt
uv run transcribe --help
```

HTTP service (two terminals):

```
make serve          # http://127.0.0.1:8000, API docs at /docs
make demo-api       # uploads the sample, shows progress, prints the SRT
```

The same steps by hand, with the job ID taken from the first response:

```
curl -s -F file=@samples/garden.mp3 http://127.0.0.1:8000/v1/transcriptions
curl -s http://127.0.0.1:8000/v1/transcriptions/<id>
curl -s 'http://127.0.0.1:8000/v1/transcriptions/<id>/result?format=srt'
```

Docker (the model is baked into the image, so running it needs no network):

```
make docker-build   # first build downloads the dependencies and the model
make docker-demo    # transcribe the sample with --network none
make docker-run     # the API on port 8000; job data lives in the volume transcription-data
make docker-test    # lint and the full offline suite, inside Linux
```

With `make docker-run` up, `make demo-api` works against the container.

## What you get

`transcribe samples/garden.mp3` (a 9.7 second clip of synthetic speech), with the default model:

```json
{
  "schema_version": "1.0",
  "source": {
    "filename": "garden.mp3", "size_bytes": 29600, "container": "mp3",
    "codec": "mp3float", "sample_rate": 16000, "channels": 1, "duration_s": 9.702,
    "sha256": "820ea7d2526b9d7e261b79a8bb75776951919d2e503dde71d9f97442b97ebd97"
  },
  "model": "base.en",
  "language": "en",
  "duration_s": 9.702,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.96, "text": "In early spring the garden is mostly mud with a few green shoots pushing through the soil."},
    {"id": 1, "start": 4.96, "end": 9.64, "text": "By May the beds are full of tall flowers and the bees are busy from dawn until dusk."}
  ],
  "chunks": [{"index": 0, "start": 0.0, "end": 9.702, "forced_cut": false}],
  "elapsed_s": 1.35
}
```

(Whitespace trimmed for the README; the real output is indented.) The same transcript as SRT, from `make demo`:

```
1
00:00:00,000 --> 00:00:04,960
In early spring the garden is mostly mud with a few green shoots pushing through the soil.

2
00:00:04,960 --> 00:00:09,640
By May the beds are full of tall flowers and the bees are busy from dawn until dusk.
```

## How it works

```
 audio file (CLI argument or HTTP upload)
        │
        ▼
 probe + normalise     identify by content, check limits, write a 16 kHz mono WAV      ingest.py
        │
        ▼
 plan chunks           cut at the quietest moment between 45 and 75 s                  chunker.py
        │
        ▼
 transcribe chunks     thread pool, one shared model, retries with backoff,            pipeline.py
        │              every finished chunk saved so a restart can skip it             transcribers/  retry.py
        ▼
 stitch + clean        add chunk offsets, keep order, drop repeat loops                merge.py  cleanup.py
        │
        ▼
 export                JSON (canonical), SRT, WebVTT, plain text                       export.py
```

The command line (`cli.py`) and the service (`api.py`, `jobs.py`, `store.py`, `files.py`) are two thin entry points over `pipeline.transcribe_file`, which knows nothing about HTTP or databases. The speech-to-text engine sits behind a small `Transcriber` interface with two implementations: `FasterWhisperTranscriber` and a deterministic `FakeTranscriber`. The fake is why the test suite runs offline in seconds, and why `--mock` works without a model.

### Handling different audio formats

- **Detection by content.** The file is opened as a byte stream through PyAV, which bundles FFmpeg's libraries, so FFmpeg identifies it from its bytes and never sees the name or the upload's `Content-Type`. Tests rename MP3, WAV and FLAC files with the wrong extension and check that they still decode.
- **Allow-list.** WAV, MP3, FLAC, Ogg, MP4/M4A, WebM/Matroska, AAC (ADTS) and AIFF. Each is round-tripped in the tests. A container that FFmpeg recognises but is not on the list (AVI, for example) is refused, and the message lists what is accepted.
- **One internal format.** Everything is decoded to 16 kHz mono 16-bit PCM and written once to a WAV file, frame by frame, so memory does not grow with the file. Later stages read only the slice they need. If a file has several audio streams, the one FFmpeg ranks best is used and the choice is logged; video streams are ignored.
- **Limits.** 500 MB and 4 hours by default. The duration is counted while decoding as well as read from the header, because container metadata can be missing or wrong.
- **Errors.** Every deliberate failure is a `PipelineError` with a stable `code`. The CLI maps codes to exit codes, the API maps them to statuses. Clients branch on the code, never on the message.

| Code | Meaning | CLI exit | HTTP |
|---|---|---|---|
| `empty_audio` | zero bytes, or no samples after decoding | 3 | 422 |
| `unsupported_format` | not audio, or a container outside the allow-list | 3 | 415 |
| `no_audio_stream` | a valid container with no audio in it | 3 | 422 |
| `corrupt_audio` | decoding fails part-way | 3 | 422 at upload if the header shows it, else on the failed job |
| `file_too_large` | over the size limit | 3 | 413 |
| `audio_too_long` | over the duration limit | 3 | 422 |
| `model_unavailable` | the model cannot be loaded | 4 | the service refuses to start without its model |
| `transcription_failed` | a chunk still fails after its retries | 4 | the failed job's error |

Exit code 2 is for usage errors (a missing file, a bad option). In the API, every error body is `application/problem+json` with the `code` field; unknown IDs, `409` states and `429` have their own codes.

### Handling long audio files

Whisper works on windows of about 30 seconds. faster-whisper can slide those windows over a whole file by itself, but one long call cannot run in parallel, cannot report progress, and a failure near the end starts again from zero. So the pipeline cuts the audio itself, and the chunk becomes the unit of parallel work, retry, resume and progress.

- **Where to cut.** The loudness of each 30 ms frame is computed (RMS in dBFS). The next cut goes at the quietest point between 45 and 75 seconds after the previous one, where "quiet" means the loudest frame in a 300 ms window around the point, so a short dip inside a word does not count. A file of 75 seconds or less is one chunk.
- **No overlap.** Chunks are contiguous. Because cuts land in pauses, no word crosses a join, and adding each chunk's start time to its segments is exact. I did not use overlapping windows with text de-duplication; fuzzy matching of repeated words is brittle.
- **Parallel chunks.** A thread pool of two workers shares one model (CTranslate2 releases the GIL). Total threads stay at half the logical CPUs, split between the workers, because using all of them measured several times slower on my machine.
- **Stitching.** `merge.stitch` adds the offsets, keeps each segment inside its chunk, makes sure no segment starts before the previous one ended, and drops anything left empty. A test builds a six-minute file of tones at known times and checks that each tone appears exactly once, at its true time.
- **Memory.** Only one chunk of samples (about 5 MB at 75 s) is in memory per worker.
- **Forced cuts.** If even the quietest point is louder than -40 dBFS, the cut is still made there, marked `forced_cut: true` in the output and logged. That cut can split a word; see the limitations.
- **Hallucinations.** faster-whisper's voice-activity filter skips non-speech inside each chunk, which is the main guard against text invented over silence. Previous text is not fed forward between windows, which stops one bad window from causing a repetition loop.
- **Measured.** A 12.2 minute file (12 chunks, none forced) takes 26.7 s with `base.en` and two workers, a real-time factor of 0.036; see [Measured results](#measured-results).

### Output for downstream use

JSON is the canonical output (`schema_version` lets consumers detect changes). It holds the source file's details, the model and language, the segments with `id`, `start`, `end` and `text` in seconds, and the chunk boundaries, so a join can be inspected when something looks wrong. SRT, WebVTT and plain text are rendered from it:

| Format | Notes |
|---|---|
| `json` | `application/json`; times in seconds, rounded to milliseconds |
| `srt` | numbered cues, `HH:MM:SS,mmm`; rounding happens once, in whole milliseconds, so 59.9996 s is `00:01:00,000` and never `00:00:59,1000` |
| `vtt` | `WEBVTT` header, `HH:MM:SS.mmm`, cue text HTML-escaped |
| `txt` | segments joined into paragraphs, a new paragraph after a pause of 2 s or more |

Cleanup runs before any export, in this order: collapse whitespace; drop empty segments; collapse three or more identical segments in a row into one (two are kept, since people say "No. No."); merge a fragment shorter than a second into the previous segment when it follows within half a second. I chose not to add summaries, entity extraction or other processing: that belongs to whatever consumes the transcript.

### The service

| Endpoint | Behaviour |
|---|---|
| `POST /v1/transcriptions` | multipart field `file`; `202` with the job and a `Location` header; `200` with the existing job if the same audio was already submitted; `413`, `415`, `422`, `429` |
| `GET /v1/transcriptions/{id}` | state, progress (`chunks_done` of `chunks_total`), error, timestamps, links |
| `GET /v1/transcriptions/{id}/result?format=` | the transcript in any of the four formats; `409 job_not_completed` before it is ready |
| `POST /v1/transcriptions/{id}/retry` | puts a failed job back in the queue, keeping its saved chunks; `409` for bad input, which a retry cannot fix |
| `DELETE /v1/transcriptions/{id}` | `204`; `409 job_running` while it is being processed |
| `GET /healthz` | status, model name, queued and processing counts |

```
            claim                  pipeline succeeds
 queued ───────────► processing ─────────────────────► completed
   ▲  ▲                 │    │
   │  └── stop or ──────┘    │ error, or attempt cap reached at startup
   │      restart            ▼
   └──── POST /retry ──── failed
```

- **The job table is the queue.** A worker claims the oldest `queued` row with one `UPDATE ... RETURNING`. SQLite admits one writer at a time, so two workers cannot claim the same job; a test runs eight threads against two jobs. An event wakes idle workers so they do not poll.
- **Duplicate uploads.** The key is the SHA-256 of the audio plus a fingerprint of the settings that change the words or timestamps (model, beam size, chunking settings). A partial unique index allows one live job per key, so two identical uploads racing each other are settled by the database: the loser gets the winner's job. Settings that only change speed (workers, threads) are left out of the key.
- **Bad input stops at the door.** The upload is hashed, de-duplicated, checked against the queue limit, and probed before a job exists, so junk never gets a job. The queue limit is 20 waiting jobs; beyond that `429` with `Retry-After: 30`. A repeat is recognised before the limit applies, so it is never refused.
- **Stopping.** On `SIGTERM` or Ctrl+C the runner lets chunks already in flight finish and saves them, then puts the job back in the queue and gives the attempt back. The wait is bounded at 30 seconds.
- **Crashing.** A `kill -9` leaves the job in `processing`. At the next start it goes back in the queue and the pipeline reuses every saved chunk. Each claim counts as an attempt, and a job that reaches three attempts without finishing is failed at startup instead of re-queued, which stops a file that crashes the process from doing so forever.
- **Finishing is repeatable.** If the process dies after writing the result but before marking the job complete, the rerun finds every chunk saved and only rewrites the same result.
- **Logging.** Every line written while a job runs carries its ID, including lines from the chunk threads. The linter bans `print`.

## Key decisions

The decisions that shaped the code, each with the alternative I weighed:

| Decision | Alternative | Why |
|---|---|---|
| faster-whisper, `int8` on CPU | openai-whisper, whisper.cpp, Vosk, hosted APIs | Whisper accuracy without PyTorch, a built-in voice-activity filter, and audio decoding through PyAV so no system ffmpeg. A hosted API would need a key, so the project would not run on a clean machine. |
| `base.en` by default | `tiny.en`, `small.en`, multilingual models | `tiny.en` is faster and less accurate, `small.en` is better and much slower on CPU. Multilingual is one setting away. |
| Cut the audio myself | Hand faster-whisper the whole file | Chunks give parallelism, progress, resume and a unit a worker fleet could take. The price is owning the joins. |
| Cut at the quietest moment, no overlap | Fixed lengths; overlapping windows with fuzzy de-duplication | Fixed cuts split words; fuzzy matching is brittle. |
| Energy-based cut points | A neural voice-activity detector | Plain numpy, deterministic on generated test audio. A neural detector is better in noise. |
| A thread pool sharing one model | A process pool | One copy of the model serves several chunks. |
| PyAV to decode | ffmpeg through `subprocess`, pydub, soundfile | The wheel bundles FFmpeg, so every format it reads works with nothing installed. |
| SQLite job table as the queue | FastAPI `BackgroundTasks`, Celery with Redis | `BackgroundTasks` has no limit, no states and loses work on restart. A real queue adds a service to run. |
| One process, API and workers together | Separate API and worker services | One command locally. Production splits them. |
| Idempotency by content hash and settings | A client `Idempotency-Key` header | Re-sending the same file never transcribes twice. The header is part of the production design. |
| A fake transcriber behind an interface | The real model in every test | The suite runs offline in seconds and can check timestamps exactly. |
| Model baked into the Docker image | Download on first run | The container runs offline straight away, at the cost of image size. |

## Production design

The local version is a miniature of the production one: the same stages, the same job states and the same unit of work (the chunk). Only the infrastructure changes.

**Concurrent uploads.** The API should not carry audio at all. A client asks for an upload, receives a pre-signed URL and sends the file straight to object storage; the API only creates the job. That keeps the API stateless and cheap to scale, and it fixes a limit of the local version (see limitations). Jobs go on a queue. Back-pressure comes from a bounded queue depth (`429` with `Retry-After`, as now) plus per-client rate limits at the gateway. Workers are a separate service scaled on queue depth, because transcription is CPU-bound and the number of simultaneous jobs on one machine is limited by its cores. An `Idempotency-Key` header, on top of the content hash I already use, covers clients that retry a request whose response they never saw.

**Storing audio and transcripts.** Object storage for the files: the original upload, the normalised audio (short-lived, since it can be rebuilt), and the transcript JSON, under `jobs/<id>/` with per-tenant prefixes. A relational database (Postgres) for job records, chunk results and, if consumers want to query them, the segments. Lifecycle rules delete audio after a retention period and transcripts according to policy. Encryption at rest and in transit, and access through short-lived signed URLs only. The code already isolates this: `store.py` is the only module with SQL and `files.py` is the only one that knows paths.

**Retrying and recovering failed transcriptions.** Three layers. A chunk call that fails is retried with exponential backoff and full jitter (built, 3 attempts). Every finished chunk is saved, so any retry above that resumes instead of starting over (built). A job whose worker disappears is handled by leases: workers heartbeat, and a job whose lease expires goes to another worker. This replaces my startup recovery, which assumes a single process. After a set number of attempts a job moves to a dead-letter queue and raises an alert instead of cycling. Errors caused by bad input are never retried; the failure keeps its code and message for the client.

**Exposing it as an API.** The shape is already an asynchronous job resource: `POST` returns `202` and a `Location`, `GET` returns state and progress, and the result is a separate resource in four formats. Errors are `application/problem+json` with stable codes, and FastAPI generates the OpenAPI document at `/docs`. What production adds: authentication (API keys or OAuth) with per-tenant quotas, signed webhooks on completion with retries so clients stop polling, a listing endpoint with pagination, an `Idempotency-Key` header, and versioning through the `/v1` prefix and `schema_version`.

| Built here | Production |
|---|---|
| SQLite `jobs` and `chunks` tables behind `store.py` | Postgres with the same tables; claims with `FOR UPDATE SKIP LOCKED`, or a managed queue |
| Re-queue `processing` jobs at startup | Leases with heartbeats, so another worker takes over |
| `data/service/jobs/<id>/` behind `files.py` | Object storage under `jobs/<id>/`, with lifecycle rules |
| Multipart upload through the API | Pre-signed upload straight to object storage |
| Worker threads inside the API process | A separate worker service scaled on queue depth |
| Attempt cap, then `failed` | Dead-letter queue and an alert |
| `429` on queue depth | The same, plus per-client rate limits at the gateway |
| Polling | Polling plus signed webhooks |

## Testing

`make test` runs 269 tests offline in about 5 seconds, using the fake transcriber and generated audio (tones and silences at known times, encoded to each format with PyAV). By area:

| Area | What is checked |
|---|---|
| Formats | every container on the allow-list round-trips to 16 kHz mono; detection by content; stereo 44.1 kHz input; several audio streams; empty, random, text, video-only and damaged files; limits |
| Chunking and stitching | cuts land in silences; chunks cover the file without gaps or overlap; forced cuts stay within the maximum; a six-minute file keeps every tone, once, at its time |
| Retries and failures | success after transient failures; no retry for permanent errors; delays grow and stay within bounds, using a fake clock |
| Exports and cleanup | exact SRT, VTT and text strings; timestamp rounding edges; one test per cleanup rule, including the keep-two, collapse-three boundary |
| Store and runner | eight threads claiming two jobs, recovery, the attempt cap, delete and retry rules, stop and resume with a spy that shows which chunks reached the transcriber, the job ID on every log line |
| API | every endpoint and status code, problem bodies, back-pressure, health, the race between identical uploads |
| Config, metrics, benchmark | settings validation and the settings fingerprint; word error rate; the benchmark's own helpers |

Threaded tests wait on conditions with a timeout instead of fixed sleeps. The two integration tests (`make test-integration`) use the real `tiny.en`: one checks the words of a sample, the other transcribes a three-minute assembled file and checks order, coverage and a word error rate under 15%. They are skipped when the model is missing, so they never reach the network.

CI (`.github/workflows/ci.yml`) runs lint and the suite on `ubuntu-latest` with the model hub set offline. I developed on macOS, so a green run there is the check that the suite does not depend on my machine. `make docker-test` does the same inside the container build, and lint is `ruff` plus `mypy --strict`.

## Measured results

`make benchmark` runs each configuration in its own process (so load time and peak memory are not inflated by models left over from earlier runs) against a 12.2 minute file made by joining the sample clips with short pauses. It writes `data/benchmark.json`; the table below is that output.

Machine: Apple M2 (4 performance and 4 efficiency cores, 8 logical CPUs), 8 GB, macOS, Python 3.12, faster-whisper 1.2.1, `int8`. It was not idle: the load average was 2.6 when the run started. In my earlier measurements background load moved results by tens of percent, so read these as indicative.

| model | workers x threads | load (s) | transcribe (s) | real-time factor | peak memory (MB) | word error rate |
|---|---|---|---|---|---|---|
| tiny.en | 1 x 4 | 0.2 | 21.7 | 0.030 | 411 | 2.4% |
| tiny.en | 2 x 2 | 0.3 | 15.2 | 0.021 | 581 | 2.4% |
| base.en | 1 x 4 | 0.3 | 37.6 | 0.051 | 494 | 2.3% |
| base.en | 2 x 2 | 0.3 | 26.7 | 0.036 | 685 | 2.3% |

- The audio is 734.5 s in 12 chunks. `base.en` with two workers (the default) is about 27 times faster than real time.
- Two workers with the same total thread count cut the time by 30% (`tiny.en`) and 29% (`base.en`), and raised peak memory by 170 and 191 MB.
- The word error rate is measured against the script the audio was generated from, with **synthetic speech**. It is a smoke check that the pipeline loses no words at the joins, not a claim about accuracy on human speech.
- Memory is the peak resident size of the whole process, including Python and the libraries.

Stop and crash behaviour, real model, the same file, from the service run (default settings, `make serve`):

| Test | Result |
|---|---|
| Ctrl+C after 4 of 12 chunks | The server exited 8.5 s later; the job was `queued` with its attempt returned and 7 chunk results saved. After the restart the log showed 7 chunks reused, and the job completed. |
| `kill -9` after 3 of 12 chunks | The job stayed `processing`. The restart re-queued it, ran attempt 2, reused 4 chunks and completed. |

In both cases the 148 segments equalled those of an uninterrupted run, in text and times, in order, without overlap.

Container, on Docker Desktop's VM on the same Mac (8 vCPUs, 4 GB, `linux/arm64`):

| Check | Result |
|---|---|
| Image size | 1.04 GB, of which the `base.en` model is 142 MB. `--build-arg MODEL=tiny.en` builds a 912 MB image with a 75 MB model |
| `make docker-test` | lint, `mypy` and all 269 tests pass inside Linux, in 4.9 s |
| `make docker-demo` | prints the SRT of the sample with `--network none`, so the image needs no network at run time |
| 12.2 minute file through the API | 148 segments in 27.2 s; the host took 26.7 s |
| `docker stop` during a job | the container exited after 4.5 s. The chunks in flight were saved and the job went back in the queue. A new container on the same volume reused 7 of 12 chunks and finished with 148 segments, identical in text and times to an uninterrupted run |

## Known limitations

- **Accuracy is measured only on synthetic speech.** The word error rate above says nothing certain about human speakers, accents, noise or overlapping voices.
- **Energy-based cuts in noisy audio.** With constant background noise there may be no quiet moment; the cut is then forced. A neural voice-activity detector would do better.
- **A forced cut can split a word.** It is flagged in the output (`forced_cut`), but the word's two halves may be transcribed badly. The proper fix is word timestamps with ownership of words at the join.
- **No word-level timestamps.** Only segment times.
- **One process, no leases.** Startup recovery assumes nothing else is running. Running uvicorn with several workers would make each one re-queue jobs the others are processing.
- **Uploads are written to disk before the size limit applies.** Starlette spools the multipart body to a temporary file first, so an oversized upload costs disk space before it is refused with `413`. A proxy limit or pre-signed uploads fix this.
- **The queue limit can be passed by a burst.** It counts waiting jobs when an upload arrives, so simultaneous uploads can overshoot it by a few.
- **An English model by default.** `TRANSCRIBE_MODEL=base` (or `small`) switches to multilingual and detects the language per chunk, but I have only measured the English models.
- **No authentication.** Anyone who can reach the port can upload and read.
- **No per-request options.** Model, language and chunking come from the server's settings.
- **CPU only.** The thread count comes from `os.cpu_count()`, which ignores a container's CPU quota.
- **A hung model call cannot be interrupted.** It runs in a thread; the stop waits up to 30 seconds, then exits, and the job resumes at the next start. Chunks are at most 75 seconds, which bounds the damage.
- **The image is checked on ARM only.** I built and ran it on Apple Silicon (`linux/arm64`). The Dockerfile has no architecture-specific parts and every native dependency publishes wheels for both, but I have not run the image on x86. CI runs the test suite on x86 Linux.

## What I would do next

In priority order:

1. **Test on real recorded speech** (several speakers, noise, phone-quality audio). Everything I measured is synthetic, and this is the biggest unknown.
2. **Pre-signed uploads and object storage.** It removes the disk-spooling limit and the API's role as a data path, and is the first step towards scaling out.
3. **Leases and a Postgres queue.** This allows more than one process and real horizontal scaling; it replaces the single-process recovery.
4. **Word timestamps with word ownership at forced cuts.** It removes the one case where a join can lose accuracy.
5. **A neural voice-activity detector for cut points.** It matters only for noisy audio, so it follows the real-speech test.
6. **Authentication, webhooks and an `Idempotency-Key` header.** The pieces a real client needs.
7. **Metrics and structured logs.** Queue depth, chunk latency and failure counts are what I would alert on.

## Project layout and commands

```
src/transcription/
  cli.py  api.py               the two entry points
  pipeline.py                  transcribe_file: the stages in order
  ingest.py  chunker.py  merge.py  cleanup.py  export.py
  transcribers/                Transcriber interface, faster-whisper, fake
  retry.py  metrics.py         backoff helper, word error rate
  jobs.py  store.py  files.py  runner, SQLite tables, per-job folders
  problems.py  errors.py  config.py  schema.py  log.py
tests/                         unit tests by module, integration/ for the real model
scripts/                       samples, long sample, model fetch, API demo, benchmark
samples/                       short clips of synthetic speech, with their text
Dockerfile  .github/workflows/ci.yml  Makefile
```

| Command | What it does |
|---|---|
| `make setup` / `lint` / `test` | environment, `ruff` + `mypy`, offline suite |
| `make test-integration` | the two real-model tests (downloads `tiny.en`) |
| `make demo-mock` / `fetch-model` / `demo` | CLI without a model / download the model / CLI with it |
| `make serve` / `serve-mock` / `demo-api` | the service / the service with the fake / upload through the API |
| `make long-sample` / `benchmark` | build the 12-minute file / measure speed, memory and accuracy |
| `make docker-build` / `docker-test` / `docker-run` / `docker-demo` | image, tests inside Linux, the API in a container, CLI with no network |

Settings are environment variables with the prefix `TRANSCRIBE_` (for example `TRANSCRIBE_MODEL=tiny.en`, `TRANSCRIBE_CHUNK_WORKERS=1`); the full list is in `src/transcription/config.py`. Released under the MIT licence.
