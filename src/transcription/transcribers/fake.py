import numpy as np
from numpy.typing import NDArray

from transcription.chunker import frame_levels_db
from transcription.config import SAMPLE_RATE, Settings
from transcription.schema import Segment
from transcription.transcribers import ChunkResult


class FakeTranscriber:
    """Deterministic stand-in: one segment per stretch of sound, with placeholder text.

    Its timestamps follow where the sound really is, which lets tests check stitching exactly.
    """

    name = "fake"

    def __init__(self, settings: Settings) -> None:
        self._frame_len = SAMPLE_RATE * settings.frame_ms // 1000
        self._frame_s = settings.frame_ms / 1000
        self._silence_db = settings.silence_db
        self._max_gap = max(1, settings.quiet_window_ms // settings.frame_ms)

    def warm_up(self) -> None:
        return None

    def transcribe(self, samples: NDArray[np.float32], language: str | None) -> ChunkResult:
        loud = frame_levels_db(samples, self._frame_len) > self._silence_db
        segments = [
            Segment(
                id=i,
                start=first * self._frame_s,
                end=last * self._frame_s,
                text=f"segment {i + 1}",
            )
            for i, (first, last) in enumerate(self._runs(loud))
        ]
        return ChunkResult(segments=segments, language=language or "en")

    def _runs(self, loud: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Frame ranges of sound, joining ranges closer than the quiet window."""
        edges = np.flatnonzero(np.diff(np.concatenate(([0], loud.astype(np.int8), [0]))))
        runs: list[tuple[int, int]] = []
        for first, last in zip(edges[::2], edges[1::2], strict=True):
            if runs and first - runs[-1][1] < self._max_gap:
                runs[-1] = (runs[-1][0], int(last))
            else:
                runs.append((int(first), int(last)))
        return runs
