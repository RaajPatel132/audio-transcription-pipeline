from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from transcription.config import Settings
from transcription.schema import Segment


class ChunkResult(BaseModel):
    """What a transcriber returns for one chunk. Segment times are relative to the chunk."""

    model_config = ConfigDict(frozen=True)

    segments: list[Segment]
    language: str


class Transcriber(Protocol):
    name: str

    def warm_up(self) -> None:
        """Load the model now, so a missing model fails at startup and not on the first job."""
        ...

    def transcribe(self, samples: NDArray[np.float32], language: str | None) -> ChunkResult:
        """Transcribe 16 kHz mono samples. `language` of `None` means detect it."""
        ...


def build_transcriber(settings: Settings, *, mock: bool = False) -> Transcriber:
    # Imported here so the mock path never loads the speech libraries.
    if mock:
        from transcription.transcribers.fake import FakeTranscriber

        return FakeTranscriber(settings)

    from transcription.transcribers.whisper import FasterWhisperTranscriber

    return FasterWhisperTranscriber(settings)
