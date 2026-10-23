import hashlib
import json
import os
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Internal audio format: 16 kHz mono 16-bit PCM, which is what Whisper works on anyway.
SAMPLE_RATE = 16_000

# The settings that change the words or the timestamps. The rest only change speed or limits.
_OUTPUT_FIELDS = {
    "model",
    "compute_type",
    "beam_size",
    "chunk_min_s",
    "chunk_max_s",
    "frame_ms",
    "quiet_window_ms",
    "silence_db",
    "mock",
}


class Settings(BaseSettings):
    """Pipeline settings; each field can be set with a TRANSCRIBE_<NAME> environment variable."""

    model_config = SettingsConfigDict(env_prefix="TRANSCRIBE_", frozen=True)

    model: str = "base.en"
    compute_type: str = "int8"
    beam_size: int = Field(5, ge=1)
    language: str | None = None
    chunk_workers: int = Field(2, ge=1)
    cpu_threads: int | None = Field(None, ge=1)

    chunk_min_s: float = Field(45, gt=0)
    chunk_max_s: float = Field(75, gt=0)
    frame_ms: int = Field(30, ge=10)
    quiet_window_ms: int = Field(300, ge=10)
    silence_db: float = Field(-40, le=0)

    max_file_mb: int = Field(500, gt=0)
    max_duration_s: float = Field(4 * 3600, gt=0)

    retry_attempts: int = Field(3, ge=1)
    retry_base_s: float = Field(0.5, ge=0)
    retry_max_s: float = Field(8.0, ge=0)

    model_dir: Path | None = None

    data_dir: Path = Path("data/service")
    job_workers: int = Field(1, ge=1)
    max_queued_jobs: int = Field(20, ge=1)
    max_job_attempts: int = Field(3, ge=1)
    shutdown_grace_s: float = Field(30, gt=0)
    mock: bool = False

    @model_validator(mode="after")
    def _check_chunk_bounds(self) -> Self:
        if self.chunk_min_s >= self.chunk_max_s:
            raise ValueError("chunk_min_s must be smaller than chunk_max_s")
        return self

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def threads_per_worker(self) -> int:
        """Use about half the logical CPUs in total, split between the workers.

        Logical CPUs include hyper-threads and, on Apple Silicon, the slow efficiency cores.
        Using all of them was measured to be several times slower than using half.
        """
        return self.cpu_threads or max(1, (os.cpu_count() or 2) // 2 // self.chunk_workers)

    @property
    def forced_language(self) -> str | None:
        """English-only models always run in English; for the rest `None` means detect."""
        return self.language or ("en" if self.model.endswith(".en") else None)

    @property
    def output_fingerprint(self) -> str:
        """Identifies the settings that shape a transcript, so equal audio with equal
        fingerprints always gives the same job."""
        values = self.model_dump(include=_OUTPUT_FIELDS) | {"language": self.forced_language}
        payload = json.dumps(values, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
