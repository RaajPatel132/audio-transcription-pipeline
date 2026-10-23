from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict

SCHEMA_VERSION = "1.0"

Seconds = Annotated[float, AfterValidator(lambda v: round(v, 3))]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Segment(_Frozen):
    """Times are relative to the chunk inside the pipeline and to the file after stitching."""

    id: int
    start: Seconds
    end: Seconds
    text: str


class AudioInfo(_Frozen):
    filename: str
    size_bytes: int
    sha256: str
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration_s: Seconds


class ChunkInfo(_Frozen):
    index: int
    start: Seconds
    end: Seconds
    forced_cut: bool = False


class Transcript(_Frozen):
    schema_version: str = SCHEMA_VERSION
    source: AudioInfo
    model: str
    language: str
    duration_s: Seconds
    segments: list[Segment]
    chunks: list[ChunkInfo]
    elapsed_s: Seconds
