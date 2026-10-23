"""Build a long test recording by joining the sample clips with short pauses.

Writes data/long-sample.mp3 and data/long-sample.txt, the text spoken in it. The pauses
are seeded, so the result is the same every time. Runs on any system.
"""

import argparse
import random
from pathlib import Path

import numpy as np

from audio_io import Mono16, read_mono, write_encoded
from transcription.config import SAMPLE_RATE

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
PAUSE_RANGE_S = (0.4, 1.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "long-sample.mp3")
    args = parser.parse_args()

    clips = {mp3: read_mono(mp3) for mp3 in sorted(SAMPLES.glob("*.mp3"))}
    rng = random.Random(7)
    parts: list[Mono16] = []
    spoken: list[str] = []
    seconds = 0.0
    while seconds < args.minutes * 60:
        for mp3, samples in clips.items():
            pause = rng.uniform(*PAUSE_RANGE_S)
            parts += [samples, np.zeros(round(pause * SAMPLE_RATE), dtype=np.int16)]
            spoken.append(mp3.with_suffix(".txt").read_text().strip())
            seconds += len(samples) / SAMPLE_RATE + pause

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_encoded(
        args.out, np.concatenate(parts), codec="libmp3lame", container="mp3", bit_rate=32_000
    )
    args.out.with_suffix(".txt").write_text("\n".join(spoken) + "\n")


if __name__ == "__main__":
    main()
