import hashlib
import logging
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from audio_factory import (
    RATE,
    encode,
    tone,
    write_two_audio_streams,
    write_video_only_mp4,
    write_wav,
)
from transcription.config import Settings
from transcription.errors import ErrorCode, PipelineError
from transcription.ingest import normalise, probe

SETTINGS = Settings()

FORMATS = [
    pytest.param("pcm_s16le", "wav", "wav", id="wav"),
    pytest.param("libmp3lame", "mp3", "mp3", id="mp3"),
    pytest.param("flac", "flac", "flac", id="flac"),
    pytest.param("libopus", "ogg", "ogg", id="ogg"),
    pytest.param("aac", "ipod", "mp4", id="m4a"),
]


def read_wav(path: Path) -> tuple[wave._wave_params, np.ndarray]:
    with wave.open(str(path)) as wav:
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return wav.getparams(), data


@pytest.mark.parametrize(("codec", "container", "label"), FORMATS)
def test_every_format_becomes_16khz_mono(
    tmp_path: Path, codec: str, container: str, label: str
) -> None:
    source = encode(tmp_path / "in", tone(2.0), codec=codec, container=container)
    out = tmp_path / "out.wav"

    info = probe(source, SETTINGS)
    duration = normalise(source, out, SETTINGS)

    params, data = read_wav(out)
    assert info.container == label
    assert (params.framerate, params.nchannels, params.sampwidth) == (RATE, 1, 2)
    assert duration == pytest.approx(2.0, abs=0.15)
    assert params.nframes / RATE == pytest.approx(duration, abs=0.001)
    assert np.abs(data).max() > 0.3 * 32767  # the tone survived the round trip


def test_stereo_44khz_input_is_downmixed_and_resampled(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "in.wav", tone(1.0, rate=44100), rate=44100, channels=2)
    out = tmp_path / "out.wav"

    info = probe(source, SETTINGS)
    duration = normalise(source, out, SETTINGS)

    params, _ = read_wav(out)
    assert (info.sample_rate, info.channels) == (44100, 2)
    assert (params.framerate, params.nchannels) == (RATE, 1)
    assert duration == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize(
    ("codec", "container", "label", "wrong_name"),
    [
        ("libmp3lame", "mp3", "mp3", "speech.wav"),
        ("pcm_s16le", "wav", "wav", "speech.mp3"),
        ("flac", "flac", "flac", "speech.m4a"),
    ],
)
def test_format_comes_from_content_not_name(
    tmp_path: Path, codec: str, container: str, label: str, wrong_name: str
) -> None:
    real = encode(tmp_path / "real", tone(1.0), codec=codec, container=container)
    renamed = tmp_path / wrong_name
    shutil.copy(real, renamed)

    assert probe(renamed, SETTINGS).container == label
    assert normalise(renamed, tmp_path / "out.wav", SETTINGS) == pytest.approx(1.0, abs=0.15)


def test_probe_describes_the_file(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "talk.wav", tone(1.5))

    info = probe(source, SETTINGS)

    assert info.filename == "talk.wav"
    assert info.size_bytes == source.stat().st_size
    assert info.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (info.codec, info.sample_rate, info.channels) == ("pcm_s16le", RATE, 1)
    assert info.duration_s == pytest.approx(1.5, abs=0.01)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"", ErrorCode.EMPTY_AUDIO),
        (np.random.default_rng(0).bytes(4096), ErrorCode.UNSUPPORTED_FORMAT),
        (b"This is a letter, not a recording.\n" * 50, ErrorCode.UNSUPPORTED_FORMAT),
    ],
    ids=["empty", "random-bytes", "text"],
)
def test_files_that_are_not_audio_are_rejected(tmp_path: Path, content: bytes, code: str) -> None:
    fake = tmp_path / "speech.mp3"
    fake.write_bytes(content)

    with pytest.raises(PipelineError) as excinfo:
        probe(fake, SETTINGS)

    assert excinfo.value.code == code


SUPPORTED = "supported: aac, aiff, flac, matroska, mp3, mp4, ogg, wav"


def test_unrecognised_content_is_rejected_with_the_supported_formats(tmp_path: Path) -> None:
    letter = tmp_path / "speech.mp3"
    letter.write_text("This is a letter, not a recording.\n" * 50)

    with pytest.raises(PipelineError, match="not a recognised audio file") as excinfo:
        probe(letter, SETTINGS)

    assert SUPPORTED in str(excinfo.value)


def test_a_known_container_outside_the_allow_list_is_rejected(tmp_path: Path) -> None:
    source = encode(tmp_path / "clip.avi", tone(1.0), codec="pcm_s16le", container="avi")

    with pytest.raises(PipelineError, match="AVI") as excinfo:
        probe(source, SETTINGS)
    with pytest.raises(PipelineError):
        normalise(source, tmp_path / "out.wav", SETTINGS)

    assert excinfo.value.code == ErrorCode.UNSUPPORTED_FORMAT
    assert SUPPORTED in str(excinfo.value)


def test_container_without_audio_is_rejected(tmp_path: Path) -> None:
    video = write_video_only_mp4(tmp_path / "clip.mp4")

    with pytest.raises(PipelineError) as excinfo:
        probe(video, SETTINGS)

    assert excinfo.value.code == ErrorCode.NO_AUDIO_STREAM


def test_one_of_several_audio_streams_is_used_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = write_two_audio_streams(tmp_path / "two.mkv", tone(1.0), tone(2.0, freq=880))

    with caplog.at_level(logging.INFO, logger="transcription.ingest"):
        info = probe(source, SETTINGS)
        duration = normalise(source, tmp_path / "out.wav", SETTINGS)

    assert "2 audio streams, using #1" in caplog.text
    assert info.channels == 2  # FFmpeg ranks the higher-bitrate stereo stream first
    assert duration == pytest.approx(2.0, abs=0.01)


def test_damaged_audio_is_reported_as_corrupt(tmp_path: Path) -> None:
    samples = np.concatenate([tone(3, freq) for freq in (300, 500, 700, 900)])
    source = encode(tmp_path / "in.flac", samples, codec="flac", container="flac")
    data = bytearray(source.read_bytes())
    for position in np.random.default_rng(1).integers(len(data) // 4, len(data) - 100, size=200):
        data[position] ^= 0xFF  # each FLAC frame carries a checksum, so this cannot go unnoticed
    source.write_bytes(data)

    with pytest.raises(PipelineError) as excinfo:
        normalise(source, tmp_path / "out.wav", SETTINGS)

    assert excinfo.value.code == ErrorCode.CORRUPT_AUDIO


def test_audio_with_no_samples_is_rejected(tmp_path: Path) -> None:
    silent_header = write_wav(tmp_path / "none.wav", np.zeros(0, dtype=np.float32))

    with pytest.raises(PipelineError) as excinfo:
        normalise(silent_header, tmp_path / "out.wav", SETTINGS)

    assert excinfo.value.code == ErrorCode.EMPTY_AUDIO


def test_oversized_file_is_rejected_before_decoding(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "big.wav", tone(40))  # about 1.3 MB

    with pytest.raises(PipelineError) as excinfo:
        probe(source, Settings(max_file_mb=1))

    assert excinfo.value.code == ErrorCode.FILE_TOO_LARGE


def test_long_audio_is_rejected_from_metadata(tmp_path: Path) -> None:
    source = write_wav(tmp_path / "long.wav", tone(3))

    with pytest.raises(PipelineError) as excinfo:
        probe(source, Settings(max_duration_s=1))

    assert excinfo.value.code == ErrorCode.AUDIO_TOO_LONG


def test_long_audio_is_rejected_by_counting_while_decoding(tmp_path: Path) -> None:
    """Metadata can be missing or wrong, so normalise enforces the limit on what it decodes."""
    source = write_wav(tmp_path / "long.wav", tone(3))

    with pytest.raises(PipelineError) as excinfo:
        normalise(source, tmp_path / "out.wav", Settings(max_duration_s=1))

    assert excinfo.value.code == ErrorCode.AUDIO_TOO_LONG
