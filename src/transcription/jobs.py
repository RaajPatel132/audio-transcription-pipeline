import logging
import threading
import time

from transcription.config import Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.export import to_json
from transcription.files import JobFiles
from transcription.log import bind_job
from transcription.pipeline import transcribe_file
from transcription.store import Job, JobStore, SqliteChunkStore
from transcription.transcribers import Transcriber

logger = logging.getLogger(__name__)

# How long an idle worker waits for a wake-up before it looks at the queue anyway.
POLL_S = 1.0


class JobInterrupted(Exception):
    """Raised between chunks to end a job early when the runner is stopping."""


class JobRunner:
    """Worker threads that run queued jobs. The job table is the queue; the event is only a
    doorbell that saves workers from polling."""

    def __init__(
        self,
        store: JobStore,
        files: JobFiles,
        settings: Settings,
        transcriber: Transcriber,
    ) -> None:
        self._store = store
        self._files = files
        self._settings = settings
        self._transcriber = transcriber
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def recover(self) -> None:
        """Settle what a previous process left behind. Run once, before `start`."""
        requeued, failed = self._store.recover_interrupted(self._settings.max_job_attempts)
        for job_id in requeued:
            with bind_job(job_id):
                logger.info("interrupted earlier, back in the queue")
        for job_id in failed:
            with bind_job(job_id):
                logger.warning("failed: reached %d attempts", self._settings.max_job_attempts)
            self._files.remove_work(job_id)
        self._files.clear_incoming()

    def start(self) -> None:
        for number in range(1, self._settings.job_workers + 1):
            thread = threading.Thread(target=self._work, name=f"job-worker-{number}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def notify(self) -> None:
        self._wake.set()

    def stop(self, timeout: float) -> None:
        """Ask the workers to finish. A running job stops at its next chunk boundary and goes
        back in the queue; chunks already in flight are finished and saved first."""
        self._stopping.set()
        self._wake.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        busy = [thread.name for thread in self._threads if thread.is_alive()]
        if busy:
            logger.warning(
                "%s still busy after %.0f s; their jobs resume at next start", busy, timeout
            )

    def _work(self) -> None:
        while not self._stopping.is_set():
            # Clear before looking, so a ring that arrives after the look is not lost.
            self._wake.clear()
            try:
                job = self._store.claim_next()
            except Exception:
                logger.exception("could not claim a job, trying again in %.0f s", POLL_S)
                self._stopping.wait(POLL_S)
                continue
            if job is None:
                self._wake.wait(POLL_S)
                continue
            with bind_job(job.id):
                try:
                    self._run(job)
                except Exception:
                    logger.exception("could not update the job record; it resumes at next start")

    def _run(self, job: Job) -> None:
        logger.info("started %s (attempt %d)", job.filename, job.attempts)
        try:
            transcript = transcribe_file(
                self._files.upload_path(job),
                self._settings,
                self._transcriber,
                self._files.work_dir(job.id),
                store=SqliteChunkStore(self._store, job.id),
                on_progress=lambda done, total: self._record_progress(job.id, done, total),
            )
            self._files.write_result(job.id, to_json(transcript))
        except JobInterrupted:
            self._store.requeue(job.id, refund_attempt=True)
            logger.info("stopped early, back in the queue")
        except PipelineError as exc:
            self._fail(job, exc.code, str(exc))
        except Exception:
            logger.exception("unexpected error")
            self._fail(job, ErrorCode.INTERNAL_ERROR, "unexpected error, see the server log")
        else:
            self._files.remove_work(job.id)
            self._store.complete(job.id)
            logger.info("completed with %d segments", len(transcript.segments))

    def _record_progress(self, job_id: str, done: int, total: int) -> None:
        self._store.set_progress(job_id, done, total)
        if self._stopping.is_set() and done < total:
            raise JobInterrupted

    def _fail(self, job: Job, code: ErrorCode, message: str) -> None:
        logger.warning("failed: %s [%s]", message, code)
        self._files.remove_work(job.id)
        self._store.fail(job.id, code, message)
