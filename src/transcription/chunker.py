import logging
import wave
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from transcription.config import SAMPLE_RATE, Settings
from transcription.schema import ChunkInfo

logger = logging.getLogger(__name__)

FLOOR_DB = -100.0
_TIE_DB = 1.0
_BLOCK_FRAMES = 2000

Levels = NDArray[np.float64]
Samples = NDArray[np.float32]


def frame_levels_db(samples: Samples, frame_len: int) -> Levels:
    """Loudness of each whole frame as RMS in dBFS. A trailing partial frame is ignored."""
    count = len(samples) // frame_len
    frames = samples[: count * frame_len].reshape(count, frame_len)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return np.asarray(20 * np.log10(np.maximum(rms, 10 ** (FLOOR_DB / 20))))


def plan_chunks(wav_path: Path, settings: Settings) -> list[ChunkInfo]:
    """Split the audio into contiguous chunks, cutting where it is quietest.

    The next cut goes at the quietest point between `chunk_min_s` and `chunk_max_s` after the
    previous one. "Quiet" means the loudest frame in a `quiet_window_ms` stretch around the point,
    so a short dip inside a word does not count. If even the best point is louder than
    `silence_db` the cut is made anyway and flagged.
    """
    frame_len = SAMPLE_RATE * settings.frame_ms // 1000
    with wave.open(str(wav_path), "rb") as wav:
        duration = wav.getnframes() / SAMPLE_RATE
        if duration <= settings.chunk_max_s:
            return [ChunkInfo(index=0, start=0, end=duration)]
        levels = _read_levels(wav, frame_len)

    window = max(1, round(settings.quiet_window_ms / settings.frame_ms))
    quiet = sliding_window_view(levels, window).max(axis=1)
    centres = (np.arange(len(quiet)) + window / 2) * frame_len / SAMPLE_RATE

    chunks: list[ChunkInfo] = []
    start = 0.0
    while duration - start > settings.chunk_max_s:
        cut, forced = _pick_cut(
            quiet,
            centres,
            lo=start + settings.chunk_min_s,
            hi=start + settings.chunk_max_s,
            silence_db=settings.silence_db,
        )
        chunk = ChunkInfo(index=len(chunks), start=start, end=cut, forced_cut=forced)
        if forced:
            logger.warning("no quiet point near %.1fs, cutting anyway", chunk.end)
        chunks.append(chunk)
        start = chunk.end
    chunks.append(ChunkInfo(index=len(chunks), start=start, end=duration))
    return chunks


def read_chunk(wav_path: Path, chunk: ChunkInfo) -> Samples:
    """Read one chunk as float32 samples, so only that slice is ever in memory."""
    first = round(chunk.start * SAMPLE_RATE)
    count = round(chunk.end * SAMPLE_RATE) - first
    with wave.open(str(wav_path), "rb") as wav:
        wav.setpos(first)
        return _to_float(wav.readframes(count))


def _read_levels(wav: wave.Wave_read, frame_len: int) -> Levels:
    parts = []
    while data := wav.readframes(_BLOCK_FRAMES * frame_len):
        parts.append(frame_levels_db(_to_float(data), frame_len))
    return np.concatenate(parts)


def _to_float(data: bytes) -> Samples:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) * np.float32(1 / 32768)


def _pick_cut(
    quiet: Levels, centres: Levels, *, lo: float, hi: float, silence_db: float
) -> tuple[float, bool]:
    candidates = np.flatnonzero((centres >= lo) & (centres <= hi))
    if candidates.size == 0:
        return hi, True
    best = quiet[candidates].min()
    # Among near-ties prefer the point closest to the middle, which keeps chunk lengths even.
    ties = candidates[quiet[candidates] <= best + _TIE_DB]
    pick = ties[np.argmin(np.abs(centres[ties] - (lo + hi) / 2))]
    return float(centres[pick]), bool(best > silence_db)
