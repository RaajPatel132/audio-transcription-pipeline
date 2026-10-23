import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from transcription.errors import BAD_INPUT, ErrorCode
from transcription.schema import ChunkInfo, Segment
from transcription.store import (
    DeleteOutcome,
    DuplicateJob,
    Job,
    JobState,
    JobStore,
    SqliteChunkStore,
)
from transcription.transcribers import ChunkResult

EPOCH = datetime(2026, 10, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    store = JobStore(tmp_path / "jobs.db")
    store.init_schema()
    return store


def make_job(key: str = "audio:settings", age_s: int = 0, **fields: object) -> Job:
    """A queued job; a larger `age_s` means it was created later."""
    return Job(
        dedupe_key=key,
        filename="talk.wav",
        size_bytes=1000,
        sha256="0" * 64,
        created_at=EPOCH + timedelta(seconds=age_s),
        **fields,
    )


def add(store: JobStore, key: str = "audio:settings", age_s: int = 0, **fields: object) -> Job:
    job = make_job(key, age_s, **fields)
    store.create(job)
    return job


def chunk(index: int = 0, start: float = 0, end: float = 45.2) -> ChunkInfo:
    return ChunkInfo(index=index, start=start, end=end)


def result(text: str = "hello") -> ChunkResult:
    return ChunkResult(segments=[Segment(id=0, start=0.5, end=1.5, text=text)], language="en")


def state_of(store: JobStore, job: Job) -> Job:
    found = store.get(job.id)
    assert found is not None
    return found


def test_a_job_round_trips(store: JobStore) -> None:
    job = add(store)

    assert store.get(job.id) == job


def test_an_unknown_job_is_none(store: JobStore) -> None:
    assert store.get("nope") is None


def test_the_database_uses_wal_and_enforces_foreign_keys(store: JobStore, tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "jobs.db") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    with pytest.raises(sqlite3.IntegrityError):
        store.save_chunk("no-such-job", chunk(), result())


def test_init_schema_can_run_again(store: JobStore) -> None:
    job = add(store)

    store.init_schema()

    assert store.get(job.id) == job


def test_a_second_live_job_with_the_same_key_is_refused(store: JobStore) -> None:
    first = add(store)

    with pytest.raises(DuplicateJob) as excinfo:
        store.create(make_job())

    assert excinfo.value.existing == first


def test_a_completed_job_still_blocks_its_key(store: JobStore) -> None:
    first = add(store)
    store.complete(first.id)

    with pytest.raises(DuplicateJob):
        store.create(make_job())


def test_a_failed_job_does_not_block_a_new_attempt(store: JobStore) -> None:
    first = add(store)
    store.fail(first.id, ErrorCode.TRANSCRIPTION_FAILED, "gave up")

    second = add(store)

    assert store.find_active(second.dedupe_key) == second


def test_find_active_ignores_other_keys_and_failed_jobs(store: JobStore) -> None:
    failed = add(store, "a")
    store.fail(failed.id, ErrorCode.INTERNAL_ERROR, "x")
    add(store, "b")

    assert store.find_active("a") is None
    assert store.find_active("c") is None


def test_claim_takes_the_oldest_queued_job(store: JobStore) -> None:
    newer = add(store, "newer", age_s=10)
    older = add(store, "older", age_s=5)

    claimed = store.claim_next()

    assert claimed is not None
    assert claimed.id == older.id
    assert claimed.state == JobState.PROCESSING
    assert claimed.attempts == 1
    assert claimed.started_at is not None
    assert state_of(store, newer).state == JobState.QUEUED


def test_claim_on_an_empty_queue_gives_nothing(store: JobStore) -> None:
    assert store.claim_next() is None

    job = add(store)
    store.claim_next()

    assert store.claim_next() is None
    assert state_of(store, job).attempts == 1


def test_eight_threads_claiming_two_jobs_claim_each_once(store: JobStore) -> None:
    jobs = [add(store, "a"), add(store, "b", age_s=1)]
    barrier = threading.Barrier(8)

    def claim() -> str | None:
        barrier.wait()
        claimed = store.claim_next()
        return claimed.id if claimed else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = [f.result() for f in [pool.submit(claim) for _ in range(8)]]

    assert sorted(i for i in outcomes if i) == sorted(job.id for job in jobs)


def test_count_by_state(store: JobStore) -> None:
    add(store, "a")
    add(store, "b", age_s=1)
    add(store, "c", age_s=2)
    store.claim_next()

    assert store.count(JobState.QUEUED) == 2
    assert store.count(JobState.PROCESSING) == 1
    assert store.count(JobState.COMPLETED) == 0


def test_progress_completion_and_failure_are_recorded(store: JobStore) -> None:
    done = add(store, "a")
    broken = add(store, "b", age_s=1)

    store.set_progress(done.id, 3, 10)
    assert (state_of(store, done).chunks_done, state_of(store, done).chunks_total) == (3, 10)

    store.complete(done.id)
    assert state_of(store, done).state == JobState.COMPLETED
    assert state_of(store, done).finished_at is not None

    store.fail(broken.id, ErrorCode.CORRUPT_AUDIO, "bad bytes")
    failed = state_of(store, broken)
    assert (failed.state, failed.error_code, failed.error_message) == (
        JobState.FAILED,
        ErrorCode.CORRUPT_AUDIO,
        "bad bytes",
    )


def test_a_refunded_requeue_restores_the_attempt(store: JobStore) -> None:
    job = add(store)
    store.claim_next()

    store.requeue(job.id, refund_attempt=True)

    requeued = state_of(store, job)
    assert (requeued.state, requeued.attempts, requeued.started_at) == (JobState.QUEUED, 0, None)


def test_a_requeue_without_refund_keeps_the_attempt(store: JobStore) -> None:
    job = add(store)
    store.claim_next()

    store.requeue(job.id, refund_attempt=False)

    assert state_of(store, job).attempts == 1


def test_recovery_requeues_jobs_under_the_cap_and_fails_those_at_it(store: JobStore) -> None:
    under = add(store, "under", state=JobState.PROCESSING, attempts=2)
    at_cap = add(store, "at-cap", state=JobState.PROCESSING, attempts=3)
    waiting = add(store, "waiting")
    done = add(store, "done", state=JobState.COMPLETED)

    requeued, failed = store.recover_interrupted(max_attempts=3)

    assert (requeued, failed) == ([under.id], [at_cap.id])
    assert state_of(store, under).state == JobState.QUEUED
    assert state_of(store, under).attempts == 2
    assert state_of(store, at_cap).state == JobState.FAILED
    assert state_of(store, at_cap).error_code == ErrorCode.INTERNAL_ERROR
    assert state_of(store, waiting).state == JobState.QUEUED
    assert state_of(store, done).state == JobState.COMPLETED


def test_recovery_with_nothing_interrupted_changes_nothing(store: JobStore) -> None:
    add(store)

    assert store.recover_interrupted(max_attempts=3) == ([], [])


def test_chunks_round_trip_by_span(store: JobStore) -> None:
    job = add(store)
    store.save_chunk(job.id, chunk(0, 0, 45.2), result("first"))

    assert store.load_chunk(job.id, chunk(0, 0, 45.2)) == result("first")
    assert store.load_chunk(job.id, chunk(0, 0, 50.0)) is None
    assert store.load_chunk(job.id, chunk(1, 45.2, 90.0)) is None


def test_a_chunk_saved_twice_keeps_the_latest(store: JobStore) -> None:
    job = add(store)

    store.save_chunk(job.id, chunk(), result("old"))
    store.save_chunk(job.id, chunk(), result("new"))

    assert store.load_chunk(job.id, chunk()) == result("new")


def test_chunks_disappear_with_their_job(store: JobStore) -> None:
    job = add(store)
    store.save_chunk(job.id, chunk(), result())

    store.delete(job.id)

    assert store.load_chunk(job.id, chunk()) is None


def test_the_chunk_store_is_scoped_to_its_job(store: JobStore) -> None:
    one, two = add(store, "a"), add(store, "b", age_s=1)
    chunks = SqliteChunkStore(store, one.id)

    chunks.put(chunk(), result())

    assert chunks.get(chunk()) == result()
    assert SqliteChunkStore(store, two.id).get(chunk()) is None


def test_delete_removes_waiting_and_finished_jobs(store: JobStore) -> None:
    waiting = add(store, "a")
    finished = add(store, "b", age_s=1)
    store.complete(finished.id)

    assert store.delete(waiting.id) == DeleteOutcome.DELETED
    assert store.delete(finished.id) == DeleteOutcome.DELETED
    assert store.get(waiting.id) is None


def test_delete_refuses_a_running_job(store: JobStore) -> None:
    job = add(store)
    store.claim_next()

    assert store.delete(job.id) == DeleteOutcome.RUNNING
    assert state_of(store, job).state == JobState.PROCESSING


def test_delete_of_an_unknown_job(store: JobStore) -> None:
    assert store.delete("nope") == DeleteOutcome.NOT_FOUND


def test_retry_requeues_a_failed_job_with_fresh_attempts(store: JobStore) -> None:
    job = add(store)
    store.claim_next()
    store.fail(job.id, ErrorCode.TRANSCRIPTION_FAILED, "gave up")

    retried = store.retry(job.id)

    assert retried is not None
    assert (retried.state, retried.attempts) == (JobState.QUEUED, 0)
    assert (retried.error_code, retried.error_message, retried.finished_at) == (None, None, None)


def test_retry_keeps_saved_chunks(store: JobStore) -> None:
    job = add(store)
    store.save_chunk(job.id, chunk(), result())
    store.fail(job.id, ErrorCode.TRANSCRIPTION_FAILED, "gave up")

    store.retry(job.id)

    assert store.load_chunk(job.id, chunk()) == result()


@pytest.mark.parametrize("code", sorted(BAD_INPUT))
def test_retry_refuses_bad_input(store: JobStore, code: ErrorCode) -> None:
    job = add(store)
    store.fail(job.id, code, "bad")

    assert store.retry(job.id) is None
    assert state_of(store, job).state == JobState.FAILED


def test_retry_refuses_jobs_that_have_not_failed_and_unknown_ones(store: JobStore) -> None:
    job = add(store)

    assert store.retry(job.id) is None
    assert store.retry("nope") is None


def test_retry_is_refused_while_an_identical_job_is_live(store: JobStore) -> None:
    failed = add(store)
    store.fail(failed.id, ErrorCode.TRANSCRIPTION_FAILED, "gave up")
    replacement = add(store)

    with pytest.raises(DuplicateJob) as excinfo:
        store.retry(failed.id)

    assert excinfo.value.existing == replacement
    assert state_of(store, failed).state == JobState.FAILED
