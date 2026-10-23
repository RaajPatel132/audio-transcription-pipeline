"""Regenerate the speech clips in samples/ from the sentences in samples/*.txt.

macOS only, because it uses the `say` command. The results are committed so other
systems do not need to run it.
"""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from audio_io import read_mono, write_encoded, write_wav

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main() -> None:
    texts = sorted(SAMPLES.glob("*.txt"))
    if not texts:
        sys.exit(f"no sentences found in {SAMPLES}")

    with TemporaryDirectory() as tmp:
        for text in texts:
            spoken = Path(tmp) / f"{text.stem}.wav"
            command = ["say", "-f", str(text), "-o", str(spoken)]
            subprocess.run(
                [*command, "--file-format=WAVE", "--data-format=LEI16@16000"], check=True
            )
            samples = read_mono(spoken)
            write_encoded(
                text.with_suffix(".mp3"),
                samples,
                codec="libmp3lame",
                container="mp3",
                bit_rate=24_000,
            )
            if text is texts[0]:  # one clip also in WAV and M4A, to show other formats
                write_wav(text.with_suffix(".wav"), samples)
                write_encoded(
                    text.with_suffix(".m4a"),
                    samples,
                    codec="aac",
                    container="ipod",
                    bit_rate=24_000,
                )


if __name__ == "__main__":
    main()
