import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Self

from fastapi import APIRouter, Depends, FastAPI, Query, Request, Response, UploadFile
from pydantic import BaseModel

from transcription.config import Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.export import Format, render
from transcription.files import JobFiles
from transcription.ingest import probe
from transcription.jobs import JobRunner
from transcription.log import bind_job, configure_logging
from transcription.problems import ApiCode, ApiProblem, install_handlers
from transcription.schema import Transcript
from transcription.store import DuplicateJob, Job, JobState, JobStore
from transcription.transcribers import Transcriber, build_transcriber

logger = logging.getLogger(__name__)

PREFIX = "/v1/transcriptions"
BLOCK_BYTES = 1024 * 1024
RETRY_AFTER_S = 30

MEDIA_TYPES = {
    "json": "application/json",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
}


@dataclass(frozen=True)
class Service:
    settings: Settings
    store: JobStore
    files: JobFiles
    runner: JobRunner
    transcriber: Transcriber


class Progress(BaseModel):
    chunks_done: int
    chunks_total: int | None


class JobError(BaseModel):
    code: ErrorCode
    message: str


class JobView(BaseModel):
    """What the API says about a job. Kept apart from the stored row so the table can change
    without changing the contract."""

    id: str
    state: JobState
    filename: str
    attempts: int
    progress: Progress
    error: JobError | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    links: dict[str, str]

    @classmethod
    def of(cls, job: Job) -> Self:
        error = (
            JobError(code=job.error_code, message=job.error_message or "")
            if job.error_code
            else None
        )
        return cls(
            id=job.id,
            state=job.state,
            filename=job.filename,
            attempts=job.attempts,
            progress=Progress(chunks_done=job.chunks_done, chunks_total=job.chunks_total),
            error=error,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            links={"self": f"{PREFIX}/{job.id}", "result": f"{PREFIX}/{job.id}/result"},
        )


class Health(BaseModel):
    status: str
    model: str
    jobs: dict[str, int]


async def get_service(request: Request) -> Service:
    service: Service = request.app.state.service
    return service


ServiceDep = Annotated[Service, Depends(get_service)]

transcriptions = APIRouter(prefix=PREFIX, tags=["transcriptions"])
health = APIRouter(tags=["health"])


@transcriptions.post(
    "",
    status_code=202,
    response_model=JobView,
    responses={200: {"model": JobView, "description": "The same audio was already submitted"}},
)
def create_transcription(file: UploadFile, response: Response, service: ServiceDep) -> JobView:
    job, created = _accept(service, file)
    response.status_code = 202 if created else 200
    response.headers["Location"] = f"{PREFIX}/{job.id}"
    return JobView.of(job)


@transcriptions.get("/{job_id}", response_model=JobView)
def get_transcription(job_id: str, service: ServiceDep) -> JobView:
    return JobView.of(_get_or_404(service, job_id))


@transcriptions.get("/{job_id}/result")
def get_result(
    job_id: str,
    service: ServiceDep,
    fmt: Annotated[Format, Query(alias="format")] = "json",
) -> Response:
    job = _get_or_404(service, job_id)
    if job.state != JobState.COMPLETED:
        raise ApiProblem(
            409,
            ApiCode.JOB_NOT_COMPLETED,
            f"the job is {job.state}, not completed; check its status first",
            state=job.state,
        )
    text = service.files.result_path(job.id).read_text(encoding="utf-8")
    return Response(render(Transcript.model_validate_json(text), fmt), media_type=MEDIA_TYPES[fmt])


@health.get("/healthz", response_model=Health)
def healthz(service: ServiceDep) -> Health:
    return Health(
        status="ok",
        model=service.transcriber.name,
        jobs={
            "queued": service.store.count(JobState.QUEUED),
            "processing": service.store.count(JobState.PROCESSING),
        },
    )


def create_app(
    settings: Settings | None = None,
    *,
    transcriber: Transcriber | None = None,
    start_workers: bool = True,
) -> FastAPI:
    """Wire the service. Tests pass a fake transcriber and can leave the workers stopped to
    keep jobs in the queue."""
    settings = settings or Settings()
    files = JobFiles(settings.data_dir)
    store = JobStore(files.root / "jobs.db")
    transcriber = transcriber or build_transcriber(settings, mock=settings.mock)
    runner = JobRunner(store, files, settings, transcriber)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        store.init_schema()
        await asyncio.to_thread(transcriber.warm_up)
        runner.recover()
        if start_workers:
            runner.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runner.stop, settings.shutdown_grace_s)

    app = FastAPI(title="Audio transcription service", lifespan=lifespan)
    app.state.service = Service(settings, store, files, runner, transcriber)
    app.include_router(transcriptions)
    app.include_router(health)
    install_handlers(app)
    return app


def _accept(service: Service, upload: UploadFile) -> tuple[Job, bool]:
    """Turn an upload into a job. Returns the job and whether it is new.

    The order matters: hash first, so a repeat is recognised before any other work; then the
    queue limit, which a repeat never hits; then the cheap validity checks, so junk never gets
    a job; only then the job is created.
    """
    settings, store, files = service.settings, service.store, service.files
    path = files.incoming(upload.filename)
    try:
        size, sha256 = _receive(upload, path, settings)
        key = f"{sha256}:{settings.output_fingerprint}"
        if (existing := store.find_active(key)) is not None:
            return existing, False
        if store.count(JobState.QUEUED) >= settings.max_queued_jobs:
            raise ApiProblem(
                429,
                ApiCode.QUEUE_FULL,
                f"{settings.max_queued_jobs} jobs are already waiting; try again later",
                headers={"Retry-After": str(RETRY_AFTER_S)},
            )
        probe(path, settings)
        job = Job(dedupe_key=key, filename=path.name, size_bytes=size, sha256=sha256)
        files.adopt(path, job.id)
        try:
            store.create(job)
        except DuplicateJob as duplicate:
            files.remove_job(job.id)
            return duplicate.existing, False
        with bind_job(job.id):
            logger.info("accepted %s (%d bytes)", job.filename, size)
        service.runner.notify()
        return job, True
    finally:
        files.discard(path)


def _receive(upload: UploadFile, path: Path, settings: Settings) -> tuple[int, str]:
    """Copy the upload into the data folder, returning its size and SHA-256."""
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as out:
        while block := upload.file.read(BLOCK_BYTES):
            size += len(block)
            if size > settings.max_file_bytes:
                raise PipelineError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"the upload is larger than the {settings.max_file_mb} MB limit",
                )
            digest.update(block)
            out.write(block)
    return size, digest.hexdigest()


def _get_or_404(service: Service, job_id: str) -> Job:
    job = service.store.get(job_id)
    if job is None:
        raise ApiProblem(404, ApiCode.NOT_FOUND, f"no transcription with id {job_id}")
    return job
