"""Decode and encode helpers shared by the sample scripts."""

import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np
from numpy.typing import NDArray

from transcription.config import SAMPLE_RATE, Settings
from transcription.ingest import normalise

Mono16 = NDArray[np.int16]


def read_mono(path: Path) -> Mono16:
    """Any supported audio file as 16 kHz mono samples."""
    with TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        normalise(path, wav_path, Settings())
        with wave.open(str(wav_path)) as wav:
            return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")


def write_wav(path: Path, samples: Mono16) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.tobytes())


def write_encoded(
    path: Path, samples: Mono16, *, codec: str, container: str, bit_rate: int
) -> None:
    """Encode 16 kHz mono samples (for example MP3 or AAC) without needing the ffmpeg program."""
    with av.open(str(path), "w", format=container) as out:
        stream = out.add_stream(codec, rate=SAMPLE_RATE, layout="mono")
        assert isinstance(stream, av.audio.stream.AudioStream)
        stream.bit_rate = bit_rate
        source = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        source.sample_rate = SAMPLE_RATE
        resampler = av.AudioResampler(
            format=stream.codec_context.format.name,
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=stream.codec_context.frame_size or None,
        )
        for frame in [*resampler.resample(source), *resampler.resample(None)]:
            out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
