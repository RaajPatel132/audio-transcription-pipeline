import threading
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from transcription.config import Settings
from transcription.log import job_id
from transcription.transcribers import ChunkResult
from transcription.transcribers.fake import FakeTranscriber

GATE_TIMEOUT_S = 15.0


class Spy:
    """Wraps the fake transcriber to count calls, and optionally to fail, to record the job ID
    or to hold every call until `gate` is set."""

    name = "spy"

    def __init__(
        self,
        fail_first: int = 0,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self._inner = FakeTranscriber(Settings())
        self._fail_first = fail_first
        self._error = error or RuntimeError("temporary glitch")
        self._gate = gate
        self._lock = threading.Lock()
        self.calls = 0
        self.warm_ups = 0
        self.job_ids: set[str] = set()
        self.started = threading.Event()

    def warm_up(self) -> None:
        self.warm_ups += 1

    def transcribe(self, samples: NDArray[np.float32], language: str | None) -> ChunkResult:
        with self._lock:
            self.calls += 1
            failing = self.calls <= self._fail_first
            self.job_ids.add(job_id.get())
        self.started.set()
        if self._gate and not self._gate.wait(GATE_TIMEOUT_S):
            raise RuntimeError("the test never opened the gate")
        if failing:
            raise self._error
        return self._inner.transcribe(samples, language)


def wait_for(condition: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)
