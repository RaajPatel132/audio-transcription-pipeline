"""Tests with the real model. Run them with `make test-integration`.

They are skipped when the model has not been downloaded, so they never reach the network.
"""

import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
from faster_whisper.utils import download_model

from transcription.config import Settings
from transcription.pipeline import transcribe_file
from transcription.transcribers import build_transcriber

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MODEL = "tiny.en"


@pytest.fixture(scope="module", autouse=True)
def _model_is_downloaded() -> None:
    try:
        download_model(MODEL, local_files_only=True)
    except Exception:
        pytest.skip(f"{MODEL} is not downloaded; run `make test-integration`")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


def test_sample_clip_contains_the_spoken_words(tmp_path: Path) -> None:
    settings = Settings(model=MODEL)

    transcript = transcribe_file(
        ROOT / "samples" / "garden.mp3", settings, build_transcriber(settings), tmp_path
    )

    spoken = set(words(" ".join(s.text for s in transcript.segments)))
    assert {"garden", "spring", "mud", "flowers", "bees"} <= spoken
    assert transcript.language == "en"


def test_long_recording_is_stitched_in_order_and_matches_the_script(tmp_path: Path) -> None:
    audio = tmp_path / "long.mp3"
    script = str(ROOT / "scripts" / "make_long_sample.py")
    subprocess.run([sys.executable, script, "--minutes", "3", "--out", str(audio)], check=True)
    settings = Settings(model=MODEL, chunk_min_s=20, chunk_max_s=40)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    transcript = transcribe_file(audio, settings, build_transcriber(settings), work_dir)

    segments = transcript.segments
    assert len(transcript.chunks) >= 4
    assert segments[0].start < 3
    assert transcript.duration_s - segments[-1].end < 3
    assert all(a.end <= b.start for a, b in pairwise(segments))
    reference = audio.with_suffix(".txt").read_text()
    assert word_error_rate(reference, " ".join(s.text for s in segments)) < 0.15
