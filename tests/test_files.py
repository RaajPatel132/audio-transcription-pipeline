from pathlib import Path

import pytest

from transcription.files import DEFAULT_NAME, JobFiles, safe_name
from transcription.store import Job


@pytest.fixture
def files(tmp_path: Path) -> JobFiles:
    return JobFiles(tmp_path / "service")


def received(files: JobFiles, name: str = "talk.mp3", content: bytes = b"audio") -> Path:
    path = files.incoming(name)
    path.write_bytes(content)
    return path


def make_job(job_id: str, filename: str = "talk.mp3") -> Job:
    return Job(id=job_id, dedupe_key="k", filename=filename, size_bytes=5, sha256="0" * 64)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("talk.mp3", "talk.mp3"),
        ("../../x.mp3", "x.mp3"),
        ("/etc/passwd", "passwd"),
        ("C:\\Users\\me\\clip.wav", "clip.wav"),
        ("a/b/c.flac", "c.flac"),
        (".hidden", "hidden"),
        ("..", DEFAULT_NAME),
        ("", DEFAULT_NAME),
        (None, DEFAULT_NAME),
        ("   ", DEFAULT_NAME),
        ("two words (1).mp3", "two words (1).mp3"),
        ("semi;colon$.mp3", "semi_colon_.mp3"),
        ("new\nline\x00.mp3", "new_line_.mp3"),
        ("über.mp3", "über.mp3"),
        ("archive.tar.gz", "archive.tar.gz"),
    ],
)
def test_safe_name(given: str | None, expected: str) -> None:
    assert safe_name(given) == expected


def test_very_long_names_are_cut_but_keep_their_extension() -> None:
    name = safe_name("a" * 300 + ".mp3")

    assert name == "a" * 100 + ".mp3"
    assert len(safe_name("clip." + "x" * 50)) <= 100 + 10


def test_the_layout_is_created_on_construction(tmp_path: Path) -> None:
    JobFiles(tmp_path / "new" / "service")

    assert (tmp_path / "new" / "service" / "incoming").is_dir()
    assert (tmp_path / "new" / "service" / "jobs").is_dir()


def test_an_upload_is_received_under_a_safe_name(files: JobFiles) -> None:
    path = received(files, "../../evil.mp3")

    assert path.name == "evil.mp3"
    assert path.is_relative_to(files.root / "incoming")


def test_two_uploads_with_the_same_name_do_not_collide(files: JobFiles) -> None:
    first = received(files, content=b"one")
    second = received(files, content=b"two")

    assert first != second
    assert (first.read_bytes(), second.read_bytes()) == (b"one", b"two")


def test_adopting_moves_the_upload_into_the_job_folder(files: JobFiles) -> None:
    path = received(files)

    adopted = files.adopt(path, "abc")

    assert adopted == files.upload_path(make_job("abc"))
    assert adopted.read_bytes() == b"audio"
    assert not path.exists()
    assert not path.parent.exists()


def test_discarding_removes_the_upload_and_its_folder(files: JobFiles) -> None:
    path = received(files)

    files.discard(path)
    files.discard(path)

    assert not path.parent.exists()


def test_the_work_dir_is_created_on_demand(files: JobFiles) -> None:
    assert files.work_dir("abc").is_dir()


def test_a_result_is_written_whole(files: JobFiles) -> None:
    files.adopt(received(files), "abc")

    files.write_result("abc", '{"ok": true}\n')

    path = files.result_path("abc")
    assert path.read_text() == '{"ok": true}\n'
    assert sorted(p.name for p in path.parent.iterdir()) == ["transcript.json", "upload"]


def test_a_failed_write_leaves_the_previous_result_untouched(
    files: JobFiles, monkeypatch: pytest.MonkeyPatch
) -> None:
    files.adopt(received(files), "abc")
    files.write_result("abc", "old")

    def refuse(self: Path, target: Path) -> Path:
        raise OSError("disk went away")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk went away"):
        files.write_result("abc", "new")

    assert files.result_path("abc").read_text() == "old"


def test_removing_the_work_folder_keeps_the_upload_and_result(files: JobFiles) -> None:
    job = make_job("abc")
    files.adopt(received(files), job.id)
    files.write_result(job.id, "{}")
    (files.work_dir(job.id) / "audio.wav").write_bytes(b"pcm")

    files.remove_work(job.id)

    assert not (files.root / "jobs" / "abc" / "work").exists()
    assert files.upload_path(job).exists()
    assert files.result_path(job.id).exists()


def test_removing_a_job_removes_its_whole_folder(files: JobFiles) -> None:
    files.adopt(received(files), "abc")
    files.write_result("abc", "{}")

    files.remove_job("abc")
    files.remove_job("abc")

    assert not (files.root / "jobs" / "abc").exists()


def test_clearing_incoming_drops_half_received_uploads(files: JobFiles) -> None:
    received(files)
    received(files)

    files.clear_incoming()

    assert list((files.root / "incoming").iterdir()) == []
