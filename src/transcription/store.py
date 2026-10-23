import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from transcription.errors import BAD_INPUT, ErrorCode
from transcription.schema import ChunkInfo
from transcription.transcribers import ChunkResult

BUSY_TIMEOUT_S = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id            TEXT PRIMARY KEY,
  dedupe_key    TEXT NOT NULL,
  filename      TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  sha256        TEXT NOT NULL,
  state         TEXT NOT NULL CHECK (state IN ('queued', 'processing', 'completed', 'failed')),
  attempts      INTEGER NOT NULL DEFAULT 0,
  chunks_done   INTEGER NOT NULL DEFAULT 0,
  chunks_total  INTEGER,
  error_code    TEXT,
  error_message TEXT,
  created_at    TEXT NOT NULL,
  started_at    TEXT,
  finished_at   TEXT
);
-- One live job per audio + settings; failed jobs do not block a new attempt.
CREATE UNIQUE INDEX IF NOT EXISTS jobs_dedupe ON jobs (dedupe_key) WHERE state != 'failed';
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (state, created_at);

CREATE TABLE IF NOT EXISTS chunks (
  job_id      TEXT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
  start_s     REAL NOT NULL,
  end_s       REAL NOT NULL,
  result_json TEXT NOT NULL,
  PRIMARY KEY (job_id, start_s, end_s)
);
"""

_INSERT_JOB = """
INSERT INTO jobs (id, dedupe_key, filename, size_bytes, sha256, state, attempts, chunks_done,
                  chunks_total, error_code, error_message, created_at, started_at, finished_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CLAIM_NEXT = """
UPDATE jobs SET state = 'processing', attempts = attempts + 1, started_at = ?
WHERE id = (SELECT id FROM jobs WHERE state = 'queued' ORDER BY created_at LIMIT 1)
RETURNING *
"""


class JobState(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DeleteOutcome(StrEnum):
    DELETED = "deleted"
    NOT_FOUND = "not_found"
    RUNNING = "running"


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: secrets.token_hex(6))
    dedupe_key: str
    filename: str
    size_bytes: int
    sha256: str
    state: JobState = JobState.QUEUED
    attempts: int = 0
    chunks_done: int = 0
    chunks_total: int | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None


class DuplicateJob(Exception):
    """A live job already exists for this audio and these settings."""

    def __init__(self, existing: Job) -> None:
        super().__init__(f"job {existing.id} already covers this audio")
        self.existing = existing


class JobStore:
    """Jobs and finished chunks in SQLite. Every operation opens its own connection, because
    connections must not be shared between threads and opening one is cheap."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    def create(self, job: Job) -> None:
        with self._transaction() as conn:
            try:
                conn.execute(
                    _INSERT_JOB,
                    (
                        job.id,
                        job.dedupe_key,
                        job.filename,
                        job.size_bytes,
                        job.sha256,
                        job.state.value,
                        job.attempts,
                        job.chunks_done,
                        job.chunks_total,
                        job.error_code.value if job.error_code else None,
                        job.error_message,
                        _iso(job.created_at),
                        _iso(job.started_at) if job.started_at else None,
                        _iso(job.finished_at) if job.finished_at else None,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = _live(conn, job.dedupe_key)
                if existing is None:
                    raise
                raise DuplicateJob(existing) from exc

    def get(self, job_id: str) -> Job | None:
        with self._connect() as conn:
            return _job(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def find_active(self, dedupe_key: str) -> Job | None:
        with self._connect() as conn:
            return _live(conn, dedupe_key)

    def count(self, state: JobState) -> int:
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM jobs WHERE state = ?", (state.value,))
            return int(cursor.fetchone()[0])

    def claim_next(self) -> Job | None:
        """Take the oldest queued job. This is one statement, so it is atomic: SQLite lets one
        writer in at a time, and the job is picked and marked inside that turn."""
        with self._connect() as conn:
            return _job(conn.execute(_CLAIM_NEXT, (_now(),)).fetchone())

    def set_progress(self, job_id: str, done: int, total: int) -> None:
        self._update(
            "UPDATE jobs SET chunks_done = ?, chunks_total = ? WHERE id = ?",
            (done, total, job_id),
        )

    def complete(self, job_id: str) -> None:
        self._update(
            "UPDATE jobs SET state = 'completed', finished_at = ?, error_code = NULL, "
            "error_message = NULL WHERE id = ?",
            (_now(), job_id),
        )

    def fail(self, job_id: str, code: ErrorCode, message: str) -> None:
        self._update(
            "UPDATE jobs SET state = 'failed', error_code = ?, error_message = ?, "
            "finished_at = ? WHERE id = ?",
            (code.value, message, _now(), job_id),
        )

    def requeue(self, job_id: str, *, refund_attempt: bool) -> None:
        self._update(
            "UPDATE jobs SET state = 'queued', started_at = NULL, attempts = attempts - ? "
            "WHERE id = ?",
            (int(refund_attempt), job_id),
        )

    def recover_interrupted(self, max_attempts: int) -> tuple[list[str], list[str]]:
        """Settle jobs left in `processing` by a stopped or crashed process. Those that have
        used all their attempts fail; the rest go back to the queue. Returns (requeued, failed)."""
        with self._transaction() as conn:
            failed = conn.execute(
                "UPDATE jobs SET state = 'failed', error_code = ?, error_message = ?, "
                "finished_at = ? WHERE state = 'processing' AND attempts >= ? RETURNING id",
                (
                    ErrorCode.INTERNAL_ERROR.value,
                    f"gave up after {max_attempts} attempts that did not finish",
                    _now(),
                    max_attempts,
                ),
            ).fetchall()
            requeued = conn.execute(
                "UPDATE jobs SET state = 'queued', started_at = NULL "
                "WHERE state = 'processing' RETURNING id"
            ).fetchall()
        return [row["id"] for row in requeued], [row["id"] for row in failed]

    def retry(self, job_id: str) -> Job | None:
        """Put a failed job back in the queue with a fresh set of attempts. Returns `None` when
        the job is unknown, not failed, or failed on bad input, which a retry cannot fix."""
        marks = ", ".join("?" * len(BAD_INPUT))
        with self._transaction() as conn:
            try:
                row = conn.execute(
                    "UPDATE jobs SET state = 'queued', attempts = 0, error_code = NULL, "
                    "error_message = NULL, started_at = NULL, finished_at = NULL "
                    f"WHERE id = ? AND state = 'failed' AND error_code NOT IN ({marks}) "
                    "RETURNING *",
                    (job_id, *(code.value for code in BAD_INPUT)),
                ).fetchone()
            except sqlite3.IntegrityError as exc:
                key = conn.execute("SELECT dedupe_key FROM jobs WHERE id = ?", (job_id,))
                existing = _live(conn, key.fetchone()["dedupe_key"])
                if existing is None:
                    raise
                raise DuplicateJob(existing) from exc
        return _job(row)

    def delete(self, job_id: str) -> DeleteOutcome:
        with self._transaction() as conn:
            deleted = conn.execute(
                "DELETE FROM jobs WHERE id = ? AND state != 'processing'", (job_id,)
            ).rowcount
            if deleted:
                return DeleteOutcome.DELETED
            exists = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return DeleteOutcome.RUNNING if exists else DeleteOutcome.NOT_FOUND

    def save_chunk(self, job_id: str, chunk: ChunkInfo, result: ChunkResult) -> None:
        self._update(
            "INSERT OR REPLACE INTO chunks (job_id, start_s, end_s, result_json) "
            "VALUES (?, ?, ?, ?)",
            (job_id, chunk.start, chunk.end, result.model_dump_json()),
        )

    def load_chunk(self, job_id: str, chunk: ChunkInfo) -> ChunkResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM chunks WHERE job_id = ? AND start_s = ? AND end_s = ?",
                (job_id, chunk.start, chunk.end),
            ).fetchone()
        return ChunkResult.model_validate_json(row["result_json"]) if row else None

    def _update(self, sql: str, params: tuple[object, ...]) -> None:
        with self._connect() as conn:
            conn.execute(sql, params)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Autocommit: each statement is its own transaction unless `_transaction` says otherwise.
        conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        # IMMEDIATE takes the write lock up front, so a read followed by a write cannot be
        # refused halfway because another writer got in between.
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")


class SqliteChunkStore:
    """The `ChunkStore` of one job: finished chunks survive a restart."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id

    def get(self, chunk: ChunkInfo) -> ChunkResult | None:
        return self._store.load_chunk(self._job_id, chunk)

    def put(self, chunk: ChunkInfo, result: ChunkResult) -> None:
        self._store.save_chunk(self._job_id, chunk, result)


def _live(conn: sqlite3.Connection, dedupe_key: str) -> Job | None:
    row = conn.execute(
        "SELECT * FROM jobs WHERE dedupe_key = ? AND state != 'failed'", (dedupe_key,)
    ).fetchone()
    return _job(row)


def _job(row: sqlite3.Row | None) -> Job | None:
    return Job.model_validate(dict(row)) if row else None


def _now() -> str:
    return _iso(datetime.now(UTC))


def _iso(moment: datetime) -> str:
    # One fixed layout, so the text sorts the same way the times do.
    return moment.astimezone(UTC).isoformat(timespec="microseconds")
