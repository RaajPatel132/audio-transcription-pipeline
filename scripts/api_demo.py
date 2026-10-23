"""Upload an audio file to a running service, follow the job and print the transcript.

Progress goes to stderr and the transcript to stdout, so the output can be redirected to a file.
Start the service first with `make serve` (or `make serve-mock` to skip the model).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, get_args

import httpx2

from transcription.export import Format

DONE = {"completed", "failed"}
POLL_S = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--format", choices=get_args(Format), default="json")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for the job")
    args = parser.parse_args()

    with httpx2.Client(base_url=args.url, timeout=30) as client:
        try:
            return _run(client, args)
        except httpx2.ConnectError:
            sys.stderr.write(f"cannot reach {args.url}; start the service with `make serve`\n")
            return 2


def _run(client: httpx2.Client, args: argparse.Namespace) -> int:
    with args.audio.open("rb") as handle:
        response = client.post("/v1/transcriptions", files={"file": (args.audio.name, handle)})
    body: dict[str, Any] = response.json()
    if response.is_error:
        sys.stderr.write(f"rejected: {body['detail']} [{body['code']}]\n")
        return 2

    note = "accepted" if response.status_code == 202 else "already submitted"
    sys.stderr.write(f"job {body['id']} {note}\n")
    job = _follow(client, body, args.timeout)
    if job["state"] != "completed":
        error = job["error"] or {"code": job["state"], "message": "timed out waiting for the job"}
        sys.stderr.write(f"failed: {error['message']} [{error['code']}]\n")
        return 1

    result = client.get(job["links"]["result"], params={"format": args.format})
    result.raise_for_status()
    sys.stdout.write(result.text)
    return 0


def _follow(client: httpx2.Client, job: dict[str, Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    shown = ""
    while True:
        line = _describe(job)
        if line != shown:
            sys.stderr.write(line + "\n")
            shown = line
        if job["state"] in DONE or time.monotonic() > deadline:
            return job
        time.sleep(POLL_S)
        job = client.get(job["links"]["self"]).json()


def _describe(job: dict[str, Any]) -> str:
    state: str = job["state"]
    progress = job["progress"]
    if progress["chunks_total"] is None:
        return state
    return f"{state}: {progress['chunks_done']}/{progress['chunks_total']} chunks"


if __name__ == "__main__":
    sys.exit(main())
