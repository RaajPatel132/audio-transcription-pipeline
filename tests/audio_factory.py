"""Builds test audio in memory and encodes it to the formats the pipeline accepts."""

import wave
from pathlib import Path

import av
import numpy as np
from av.audio.stream import AudioStream
from numpy.typing import NDArray

RATE = 16_000

Samples = NDArray[np.float32]


def tone(seconds: float, freq: float = 440.0, rate: int = RATE, amplitude: float = 0.5) -> Samples:
    t = np.arange(round(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, rate: int = RATE) -> Samples:
    return np.zeros(round(seconds * rate), dtype=np.float32)


def tones_and_gaps(
    tone_s: float, gap_s: float, count: int
) -> tuple[Samples, list[tuple[float, float]]]:
    """`count` tones, each followed by a gap. Also returns the (start, end) of every tone."""
    parts: list[Samples] = []
    spans: list[tuple[float, float]] = []
    now = 0.0
    for _ in range(count):
        parts += [tone(tone_s), silence(gap_s)]
        spans.append((now, now + tone_s))
        now += tone_s + gap_s
    return np.concatenate(parts), spans


def to_int16(samples: Samples) -> NDArray[np.int16]:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16)


def write_wav(path: Path, samples: Samples, rate: int = RATE, channels: int = 1) -> Path:
    data = to_int16(samples)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(data.tobytes())
    return path


def encode(
    path: Path,
    samples: Samples,
    *,
    codec: str,
    container: str,
    rate: int = RATE,
    channels: int = 1,
) -> Path:
    """Encode `samples` with PyAV, so tests can make any format without ffmpeg installed."""
    with av.open(str(path), "w", format=container) as out:
        stream = _add_audio_stream(out, codec, rate, channels)
        _write_samples(out, stream, samples, channels)
    return path


def write_two_audio_streams(path: Path, first: Samples, second: Samples) -> Path:
    """A Matroska file with a mono and a stereo audio stream of different lengths."""
    with av.open(str(path), "w", format="matroska") as out:
        mono = _add_audio_stream(out, "pcm_s16le", RATE, channels=1)
        stereo = _add_audio_stream(out, "pcm_s16le", RATE, channels=2)
        _write_samples(out, mono, first, channels=1)
        _write_samples(out, stereo, second, channels=2)
    return path


def write_video_only_mp4(path: Path, seconds: float = 1.0) -> Path:
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(round(seconds * 10)):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 20, np.uint8), "rgb24")
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
    return path


def _layout(channels: int) -> str:
    return "mono" if channels == 1 else "stereo"


def _add_audio_stream(
    out: av.container.OutputContainer, codec: str, rate: int, channels: int
) -> AudioStream:
    stream = out.add_stream(codec, rate=rate, layout=_layout(channels))
    assert isinstance(stream, AudioStream)
    return stream


def _write_samples(
    out: av.container.OutputContainer,
    stream: av.audio.stream.AudioStream,
    samples: Samples,
    channels: int,
) -> None:
    codec = stream.codec_context
    interleaved = np.repeat(to_int16(samples)[:, None], channels, axis=1).reshape(1, -1)
    source = av.AudioFrame.from_ndarray(interleaved, format="s16", layout=_layout(channels))
    source.sample_rate = codec.sample_rate
    resampler = av.AudioResampler(
        format=codec.format.name,
        layout=_layout(channels),
        rate=codec.sample_rate,
        frame_size=codec.frame_size or None,
    )
    for frame in [*resampler.resample(source), *resampler.resample(None)]:
        out.mux(stream.encode(frame))
    out.mux(stream.encode(None))
