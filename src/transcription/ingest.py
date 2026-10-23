import hashlib
import logging
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import av
from av.audio.stream import AudioStream
from av.container import InputContainer

from transcription.config import SAMPLE_RATE, Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.schema import AudioInfo

logger = logging.getLogger(__name__)

# FFmpeg's name for each container we accept, and the short name we report.
CONTAINERS = {
    "wav": "wav",
    "mp3": "mp3",
    "flac": "flac",
    "ogg": "ogg",
    "mov,mp4,m4a,3gp,3g2,mj2": "mp4",
    "matroska,webm": "matroska",
    "aac": "aac",
    "aiff": "aiff",
}

_MB = 1024 * 1024


def probe(path: Path, settings: Settings) -> AudioInfo:
    """Check the file against the limits and describe it, without decoding any audio."""
    size = path.stat().st_size
    if size == 0:
        raise PipelineError(ErrorCode.EMPTY_AUDIO, f"{path.name} is empty")
    if size > settings.max_file_bytes:
        raise PipelineError(
            ErrorCode.FILE_TOO_LARGE,
            f"{path.name} is {size / _MB:.0f} MB, the limit is {settings.max_file_mb} MB",
        )

    with _open_audio(path) as (container, stream):
        duration = _duration_s(container, stream)
        if duration > settings.max_duration_s:
            raise PipelineError(
                ErrorCode.AUDIO_TOO_LONG,
                f"{path.name} is {duration / 60:.0f} min, the limit is "
                f"{settings.max_duration_s / 60:.0f} min",
            )
        codec = stream.codec_context
        return AudioInfo(
            filename=path.name,
            size_bytes=size,
            sha256=_sha256(path),
            container=CONTAINERS[container.format.name],
            codec=codec.name,
            sample_rate=codec.sample_rate,
            channels=codec.layout.nb_channels,
            duration_s=duration,
        )


def normalise(path: Path, out_wav: Path, settings: Settings) -> float:
    """Decode the best audio stream to 16 kHz mono PCM in `out_wav`; return its true length.

    Frames are written as they are decoded, so memory does not grow with the file. The length is
    counted here because container metadata can be missing or wrong.
    """
    limit = int(settings.max_duration_s * SAMPLE_RATE)
    total = 0
    with _open_audio(path) as (container, stream), wave.open(str(out_wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        try:
            for frame in _resampled(container, stream):
                total += frame.samples
                if total > limit:
                    raise PipelineError(
                        ErrorCode.AUDIO_TOO_LONG,
                        f"{path.name} is longer than the {settings.max_duration_s / 60:.0f} "
                        "min limit",
                    )
                out.writeframes(frame.to_ndarray().tobytes())
        except av.error.FFmpegError as exc:
            raise PipelineError(
                ErrorCode.CORRUPT_AUDIO, f"could not decode {path.name}: {exc}"
            ) from exc

    if total == 0:
        raise PipelineError(ErrorCode.EMPTY_AUDIO, f"{path.name} contains no audio samples")
    return round(total / SAMPLE_RATE, 3)


@contextmanager
def _open_audio(path: Path) -> Iterator[tuple[InputContainer, AudioStream]]:
    """Open the file by content (the name is never shown to FFmpeg) and pick its audio stream."""
    with path.open("rb") as handle:
        try:
            container = av.open(handle, mode="r")
        except av.error.FFmpegError as exc:
            raise PipelineError(
                ErrorCode.UNSUPPORTED_FORMAT, f"{path.name} is not a recognised audio file"
            ) from exc

        with container:
            if container.format.name not in CONTAINERS:
                raise PipelineError(
                    ErrorCode.UNSUPPORTED_FORMAT,
                    f"{path.name} is a {container.format.long_name} file, which is not supported",
                )
            stream = container.streams.best("audio")
            if not isinstance(stream, AudioStream):
                raise PipelineError(ErrorCode.NO_AUDIO_STREAM, f"{path.name} has no audio stream")
            count = len(container.streams.audio)
            if count > 1:
                logger.info("%s has %d audio streams, using #%d", path.name, count, stream.index)
            yield container, stream


def _resampled(container: InputContainer, stream: AudioStream) -> Iterator[av.AudioFrame]:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    for frame in container.decode(stream):
        yield from resampler.resample(frame)
    yield from resampler.resample(None)


def _duration_s(container: InputContainer, stream: AudioStream) -> float:
    if container.duration is not None:
        return container.duration / av.time_base
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_MB):
            digest.update(block)
    return digest.hexdigest()
