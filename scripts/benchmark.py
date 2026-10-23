"""Measure speed, memory and accuracy for each model and worker count.

Every configuration runs in its own process, so load time and peak memory are not inflated by
models left over from earlier runs. The report is written to data/benchmark.json.
"""

import argparse
import itertools
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel

from fetch_model import fetch
from transcription.config import Settings
from transcription.metrics import word_error_rate
from transcription.pipeline import transcribe_file
from transcription.transcribers import Transcriber, build_transcriber

DATA = Path(__file__).resolve().parent.parent / "data"
LIBRARIES = ["faster-whisper", "ctranslate2", "av", "onnxruntime", "numpy"]


class Result(BaseModel):
    model: str
    workers: int
    threads_per_worker: int
    load_s: float
    audio_s: float
    transcribe_s: float
    real_time_factor: float
    chunks: int
    segments: int
    word_error_rate: float
    peak_memory_mb: int


class Machine(BaseModel):
    platform: str
    cpu: str
    logical_cpus: int
    memory_gb: float
    python: str
    libraries: dict[str, str]


class Report(BaseModel):
    date: str
    audio: str
    load_average_at_start: float
    machine: Machine
    results: list[Result]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, default=DATA / "long-sample.mp3")
    parser.add_argument("--reference", type=Path, help="text spoken in the audio (default: .txt)")
    parser.add_argument("--models", nargs="+", default=["tiny.en", "base.en"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--out", type=Path, default=DATA / "benchmark.json")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    reference = args.reference or args.audio.with_suffix(".txt")

    if args.child:
        settings = Settings(model=args.models[0], chunk_workers=args.workers[0])
        result = measure(args.audio, reference.read_text(), settings, build_transcriber(settings))
        sys.stdout.write(result.model_dump_json())
        return

    for model in args.models:
        fetch(Settings(model=model))
    report = Report(
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        audio=args.audio.name,
        load_average_at_start=round(os.getloadavg()[0], 2),
        machine=describe_machine(),
        results=[
            run_in_new_process(args.audio, reference, model, workers)
            for model, workers in itertools.product(args.models, args.workers)
        ],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=2) + "\n")
    sys.stdout.write(render_table(report.results))


def measure(audio: Path, reference: str, settings: Settings, transcriber: Transcriber) -> Result:
    started = time.perf_counter()
    transcriber.warm_up()
    load_s = time.perf_counter() - started

    with tempfile.TemporaryDirectory(prefix="benchmark-") as work_dir:
        transcript = transcribe_file(audio, settings, transcriber, Path(work_dir))

    text = " ".join(segment.text for segment in transcript.segments)
    return Result(
        model=settings.model,
        workers=settings.chunk_workers,
        threads_per_worker=settings.threads_per_worker,
        load_s=round(load_s, 2),
        audio_s=transcript.duration_s,
        transcribe_s=transcript.elapsed_s,
        real_time_factor=round(transcript.elapsed_s / transcript.duration_s, 3),
        chunks=len(transcript.chunks),
        segments=len(transcript.segments),
        word_error_rate=round(word_error_rate(reference, text), 4),
        peak_memory_mb=round(peak_memory_mb()),
    )


def run_in_new_process(audio: Path, reference: Path, model: str, workers: int) -> Result:
    sys.stderr.write(f"measuring {model} with {workers} worker(s)\n")
    command = [
        sys.executable,
        __file__,
        "--child",
        *("--audio", str(audio), "--reference", str(reference)),
        *("--models", model, "--workers", str(workers)),
    ]
    # Offline, so a model that is missing fails here and never downloads in the middle of a run.
    done = subprocess.run(
        command, capture_output=True, text=True, env=os.environ | {"HF_HUB_OFFLINE": "1"}
    )
    if done.returncode:
        raise SystemExit(f"{model} with {workers} worker(s) failed:\n{done.stderr}")
    return Result.model_validate_json(done.stdout)


def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes.
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def describe_machine() -> Machine:
    return Machine(
        platform=platform.platform(),
        cpu=_cpu_name(),
        logical_cpus=os.cpu_count() or 1,
        memory_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1),
        python=platform.python_version(),
        libraries={name: version(name) for name in LIBRARIES},
    )


def _cpu_name() -> str:
    if sys.platform == "darwin":
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        return brand.stdout.strip() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def render_table(results: list[Result]) -> str:
    rows = [
        "| model | workers x threads | load (s) | transcribe (s) | real-time factor "
        "| peak memory (MB) | word error rate |",
        "|---|---|---|---|---|---|---|",
    ]
    rows += [
        f"| {r.model} | {r.workers} x {r.threads_per_worker} | {r.load_s:.1f} "
        f"| {r.transcribe_s:.1f} | {r.real_time_factor:.3f} | {r.peak_memory_mb} "
        f"| {r.word_error_rate:.1%} |"
        for r in results
    ]
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    main()
