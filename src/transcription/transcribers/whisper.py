import logging
import threading

import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray

from transcription.config import Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.schema import Segment
from transcription.transcribers import ChunkResult

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """Whisper on CPU through faster-whisper. One model is shared by all chunk workers."""

    def __init__(self, settings: Settings) -> None:
        self.name = settings.model
        self._settings = settings
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._load()

    def transcribe(self, samples: NDArray[np.float32], language: str | None) -> ChunkResult:
        model = self._load()
        # VAD skips non-speech, the main guard against text invented over silence. Not feeding
        # the previous text forward stops one bad window from causing repetition loops.
        segments, info = model.transcribe(
            samples,
            language=language,
            beam_size=self._settings.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        # The segments are produced lazily, so they are consumed here where errors can be retried.
        return ChunkResult(
            segments=[
                Segment(id=i, start=s.start, end=s.end, text=s.text) for i, s in enumerate(segments)
            ],
            language=info.language,
        )

    def _load(self) -> WhisperModel:
        with self._lock:
            if self._model is None:
                settings = self._settings
                logger.info("loading model %s (%s)", settings.model, settings.compute_type)
                try:
                    self._model = WhisperModel(
                        settings.model,
                        device="cpu",
                        compute_type=settings.compute_type,
                        cpu_threads=settings.threads_per_worker,
                        num_workers=settings.chunk_workers,
                        download_root=str(settings.model_dir) if settings.model_dir else None,
                    )
                except Exception as exc:
                    raise PipelineError(
                        ErrorCode.MODEL_UNAVAILABLE,
                        f"could not load model {settings.model!r}: {exc}",
                    ) from exc
            return self._model
