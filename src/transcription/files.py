import os
import re
import secrets
import shutil
from pathlib import Path, PureWindowsPath

from transcription.store import Job

MAX_STEM_LENGTH = 100
MAX_SUFFIX_LENGTH = 10
DEFAULT_NAME = "upload"

_UNSAFE = re.compile(r"[^\w.\- ()]")


class JobFiles:
    """Where everything a job owns lives on disk. This class stands in for object storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._incoming = root / "incoming"
        self._jobs = root / "jobs"
        self._incoming.mkdir(parents=True, exist_ok=True)
        self._jobs.mkdir(parents=True, exist_ok=True)

    def incoming(self, filename: str | None) -> Path:
        """A fresh path for an upload that is still being received."""
        folder = self._incoming / secrets.token_hex(8)
        folder.mkdir()
        return folder / safe_name(filename)

    def adopt(self, incoming_path: Path, job_id: str) -> Path:
        """Move a received upload into its job's folder."""
        folder = self._job_dir(job_id) / "upload"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / incoming_path.name
        incoming_path.replace(target)
        incoming_path.parent.rmdir()
        return target

    def discard(self, incoming_path: Path) -> None:
        shutil.rmtree(incoming_path.parent, ignore_errors=True)

    def upload_path(self, job: Job) -> Path:
        return self._job_dir(job.id) / "upload" / safe_name(job.filename)

    def work_dir(self, job_id: str) -> Path:
        path = self._job_dir(job_id) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def result_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "transcript.json"

    def write_result(self, job_id: str, text: str) -> None:
        """Write to a temporary file and rename it, so a crash never leaves a half-written
        result in place."""
        path = self.result_path(job_id)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def remove_work(self, job_id: str) -> None:
        shutil.rmtree(self._job_dir(job_id) / "work", ignore_errors=True)

    def remove_job(self, job_id: str) -> None:
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)

    def clear_incoming(self) -> None:
        shutil.rmtree(self._incoming, ignore_errors=True)
        self._incoming.mkdir(parents=True)

    def _job_dir(self, job_id: str) -> Path:
        return self._jobs / job_id


def safe_name(filename: str | None) -> str:
    """Reduce a client-supplied name to a plain file name: no directories, no leading dots,
    no characters that need escaping, a bounded length and the extension kept."""
    # A Windows path splits on both kinds of slash, which is what a client might send.
    name = _UNSAFE.sub("_", PureWindowsPath(filename or "").name).strip(". ")
    stem, suffix = os.path.splitext(name)
    return stem[:MAX_STEM_LENGTH] + suffix[:MAX_SUFFIX_LENGTH] or DEFAULT_NAME
