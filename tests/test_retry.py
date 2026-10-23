import pytest

from transcription.errors import ErrorCode, PipelineError
from transcription.retry import call_with_retry


class Flaky:
    def __init__(self, failures: int, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or RuntimeError("boom")
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def retry(fn: Flaky, sleeps: list[float], *, attempts: int = 3, rand: float = 1.0) -> str:
    return call_with_retry(
        fn, attempts=attempts, base_s=0.5, max_s=2.0, sleep=sleeps.append, rand=lambda: rand
    )


def test_succeeds_after_transient_failures() -> None:
    fn = Flaky(failures=2)
    sleeps: list[float] = []

    assert retry(fn, sleeps) == "ok"
    assert fn.calls == 3
    assert len(sleeps) == 2


def test_permanent_errors_are_not_retried() -> None:
    fn = Flaky(failures=5, error=PipelineError(ErrorCode.MODEL_UNAVAILABLE, "no model"))
    sleeps: list[float] = []

    with pytest.raises(PipelineError) as excinfo:
        retry(fn, sleeps)

    assert excinfo.value.code == ErrorCode.MODEL_UNAVAILABLE
    assert fn.calls == 1
    assert sleeps == []


def test_retryable_pipeline_errors_are_retried() -> None:
    fn = Flaky(failures=1, error=PipelineError(ErrorCode.MODEL_UNAVAILABLE, "busy", retryable=True))

    assert retry(fn, []) == "ok"


def test_gives_up_after_the_last_attempt() -> None:
    fn = Flaky(failures=10)
    sleeps: list[float] = []

    with pytest.raises(PipelineError, match="failed after 3 attempts") as excinfo:
        retry(fn, sleeps)

    assert excinfo.value.code == ErrorCode.TRANSCRIPTION_FAILED
    assert excinfo.value.__cause__ is fn.error
    assert fn.calls == 3
    assert len(sleeps) == 2  # no wait after the final failure


def test_at_least_one_attempt_is_required() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        retry(Flaky(failures=0), [], attempts=0)


def test_delays_grow_and_are_capped() -> None:
    sleeps: list[float] = []

    with pytest.raises(PipelineError):
        retry(Flaky(failures=10), sleeps, attempts=6, rand=1.0)

    assert sleeps == [0.5, 1.0, 2.0, 2.0, 2.0]


def test_jitter_scales_the_delay() -> None:
    sleeps: list[float] = []

    with pytest.raises(PipelineError):
        retry(Flaky(failures=10), sleeps, attempts=3, rand=0.5)

    assert sleeps == [0.25, 0.5]
