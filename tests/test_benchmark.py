import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark import Result, describe_machine, measure, peak_memory_mb, render_table
from fakes import Spy
from transcription.config import Settings

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def make_result(**changes: object) -> Result:
    values = {
        "model": "tiny.en",
        "workers": 1,
        "threads_per_worker": 4,
        "load_s": 0.64,
        "audio_s": 734.5,
        "transcribe_s": 14.21,
        "real_time_factor": 0.019,
        "chunks": 12,
        "segments": 148,
        "word_error_rate": 0.0231,
        "peak_memory_mb": 480,
    }
    return Result(**(values | changes))


def test_measure_reports_the_run_and_loads_the_model_first() -> None:
    spy = Spy()
    settings = Settings(model="tiny.en", chunk_workers=2, cpu_threads=3)

    result = measure(SAMPLES / "garden.mp3", "segment segment", settings, spy)

    assert spy.warm_ups == 1
    assert (result.model, result.workers, result.threads_per_worker) == ("tiny.en", 2, 3)
    assert (result.chunks, result.segments) == (1, 2)
    assert result.audio_s > 0
    assert result.real_time_factor == pytest.approx(result.transcribe_s / result.audio_s, abs=1e-3)
    assert result.peak_memory_mb > 0


def test_measure_scores_the_text_against_the_reference() -> None:
    audio = SAMPLES / "garden.mp3"

    assert measure(audio, "segment segment", Settings(), Spy()).word_error_rate == 0
    assert measure(audio, "segment", Settings(), Spy()).word_error_rate == 1


@pytest.mark.parametrize(
    ("platform_name", "reported", "expected"),
    [("darwin", 512 * 1024 * 1024, 512), ("linux", 512 * 1024, 512)],
)
def test_peak_memory_is_normalised_to_megabytes(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, reported: int, expected: float
) -> None:
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=reported))

    assert peak_memory_mb() == expected


def test_table_has_a_header_and_one_row_per_result() -> None:
    table = render_table([make_result(), make_result(model="base.en", workers=2)])

    assert table.splitlines() == [
        "| model | workers x threads | load (s) | transcribe (s) | real-time factor "
        "| peak memory (MB) | word error rate |",
        "|---|---|---|---|---|---|---|",
        "| tiny.en | 1 x 4 | 0.6 | 14.2 | 0.019 | 480 | 2.3% |",
        "| base.en | 2 x 4 | 0.6 | 14.2 | 0.019 | 480 | 2.3% |",
    ]


def test_machine_description_names_the_libraries_that_matter() -> None:
    machine = describe_machine()

    assert machine.logical_cpus >= 1
    assert machine.memory_gb > 0
    assert machine.cpu
    assert {"faster-whisper", "ctranslate2"} <= machine.libraries.keys()
