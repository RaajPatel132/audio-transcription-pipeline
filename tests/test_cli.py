import json
from pathlib import Path

import numpy as np
import pytest

from audio_factory import silence, tone, write_wav
from transcription.cli import main


@pytest.fixture
def speech(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "clip.wav", np.concatenate([tone(2), silence(1), tone(2)]))


def test_mock_run_prints_valid_json(speech: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(speech), "--mock"]) == 0

    out, err = capsys.readouterr()
    data = json.loads(out)
    assert data["source"]["filename"] == "clip.wav"
    assert len(data["segments"]) == 2
    assert "job=" in err  # logs went to stderr, leaving stdout clean


def test_srt_can_be_written_to_a_file(
    speech: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "out.srt"

    assert main([str(speech), "--mock", "--format", "srt", "--output", str(target)]) == 0

    assert target.read_text().startswith("1\n00:00:00,")
    assert capsys.readouterr().out == ""


def test_bad_input_exits_3_with_a_one_line_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = tmp_path / "notes.mp3"
    fake.write_text("this is not audio\n" * 50)

    assert main([str(fake), "--mock"]) == 3

    out, err = capsys.readouterr()
    messages = [line for line in err.splitlines() if line.startswith("error:")]
    assert out == ""
    assert len(messages) == 1
    assert "unsupported_format" in messages[0]


def test_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "nope.wav"), "--mock"])

    assert excinfo.value.code == 2


def test_invalid_setting_is_a_usage_error(speech: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_CHUNK_MIN_S", "500")

    with pytest.raises(SystemExit) as excinfo:
        main([str(speech), "--mock"])

    assert excinfo.value.code == 2
