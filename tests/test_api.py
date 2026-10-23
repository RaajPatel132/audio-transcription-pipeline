import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audio_factory import silence, tone, write_wav
from fakes import Spy, wait_for
from transcription import api
from transcription.api import Service, create_app
from transcription.config import Settings
from transcription.errors import ErrorCode
from transcription.schema import Transcript

JOBS = "/v1/transcriptions"
PROBLEM = "application/problem+json"

MakeClient = Callable[..., TestClient]


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[MakeClient]:
    with ExitStack() as stack:

        def build(
            spy: Spy | None = None, *, start_workers: bool = True, **overrides: Any
        ) -> TestClient:
            settings = Settings(
                data_dir=tmp_path / "service",
                retry_base_s=0,
                retry_max_s=0,
                shutdown_grace_s=5,
                **overrides,
            )
            app = create_app(settings, transcriber=spy or Spy(), start_workers=start_workers)
            return stack.enter_context(TestClient(app))

        yield build


@pytest.fixture
def speech(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "clip.wav", np.concatenate([tone(2), silence(1), tone(2)]))


@pytest.fixture
def other_speech(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "other.wav", np.concatenate([tone(3, 330), silence(1), tone(1)]))


def upload(
    client: TestClient, path: Path, name: str | None = None, content_type: str = "audio/wav"
) -> httpx2.Response:
    return client.post(JOBS, files={"file": (name or path.name, path.read_bytes(), content_type)})


def service_of(client: TestClient) -> Service:
    app = client.app
    assert isinstance(app, FastAPI)
    service: Service = app.state.service
    return service


def state_of(client: TestClient, job_id: str) -> str:
    return str(client.get(f"{JOBS}/{job_id}").json()["state"])


def wait_for_state(client: TestClient, job_id: str, state: str) -> dict[str, Any]:
    wait_for(lambda: state_of(client, job_id) == state)
    body: dict[str, Any] = client.get(f"{JOBS}/{job_id}").json()
    return body


def assert_problem(response: httpx2.Response, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"].startswith(PROBLEM)
    body: dict[str, Any] = response.json()
    assert body["code"] == code
    assert body["status"] == status
    assert body["type"] == "about:blank"
    assert body["title"]
    assert body["detail"]
    return body


def test_an_upload_is_accepted_and_runs_to_completion(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client()

    response = upload(client, speech)

    body = response.json()
    assert response.status_code == 202
    assert response.headers["location"] == f"{JOBS}/{body['id']}"
    assert body["filename"] == "clip.wav"
    assert body["links"] == {
        "self": f"{JOBS}/{body['id']}",
        "result": f"{JOBS}/{body['id']}/result",
    }
    done = wait_for_state(client, body["id"], "completed")
    assert done["progress"] == {"chunks_done": 1, "chunks_total": 1}
    assert done["attempts"] == 1
    assert done["error"] is None
    assert done["finished_at"] is not None


@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("json", "application/json"),
        ("srt", "application/x-subrip; charset=utf-8"),
        ("vtt", "text/vtt; charset=utf-8"),
        ("txt", "text/plain; charset=utf-8"),
    ],
)
def test_a_result_comes_back_in_every_format(
    make_client: MakeClient, speech: Path, fmt: str, media_type: str
) -> None:
    client = make_client()
    job_id = upload(client, speech).json()["id"]
    wait_for_state(client, job_id, "completed")

    response = client.get(f"{JOBS}/{job_id}/result", params={"format": fmt})

    assert response.status_code == 200
    assert response.headers["content-type"] == media_type
    assert "segment 1" in response.text


def test_the_json_result_is_a_valid_transcript_and_the_default(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client()
    job_id = upload(client, speech).json()["id"]
    wait_for_state(client, job_id, "completed")

    transcript = Transcript.model_validate_json(client.get(f"{JOBS}/{job_id}/result").text)

    assert transcript.source.filename == "clip.wav"
    assert len(transcript.segments) == 2
    assert transcript.model == "spy"


def test_subtitle_formats_have_the_expected_shape(make_client: MakeClient, speech: Path) -> None:
    client = make_client()
    job_id = upload(client, speech).json()["id"]
    wait_for_state(client, job_id, "completed")

    srt = client.get(f"{JOBS}/{job_id}/result", params={"format": "srt"}).text
    vtt = client.get(f"{JOBS}/{job_id}/result", params={"format": "vtt"}).text

    assert srt.startswith("1\n00:00:00,")
    assert vtt.startswith("WEBVTT\n\n00:00:00.")


def test_uploading_the_same_file_again_returns_the_same_job(
    make_client: MakeClient, speech: Path
) -> None:
    spy = Spy()
    client = make_client(spy)
    first = upload(client, speech).json()
    wait_for_state(client, first["id"], "completed")
    calls = spy.calls

    again = upload(client, speech, name="renamed.wav")

    assert again.status_code == 200
    assert again.json()["id"] == first["id"]
    assert spy.calls == calls
    assert client.get("/healthz").json()["jobs"] == {"queued": 0, "processing": 0}


def test_a_repeat_while_the_first_is_still_queued_returns_the_same_job(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(start_workers=False)

    first = upload(client, speech)
    again = upload(client, speech)

    assert (first.status_code, again.status_code) == (202, 200)
    assert again.json()["id"] == first.json()["id"]


def test_the_format_is_found_by_content_not_by_name(make_client: MakeClient, speech: Path) -> None:
    client = make_client()

    response = upload(client, speech, name="voice.mp3", content_type="audio/mpeg")

    assert response.status_code == 202
    wait_for_state(client, response.json()["id"], "completed")


def test_the_client_supplied_name_cannot_escape_the_data_folder(
    make_client: MakeClient, speech: Path, tmp_path: Path
) -> None:
    client = make_client()

    response = upload(client, speech, name="../../outside.wav")

    assert response.json()["filename"] == "outside.wav"
    assert not (tmp_path / "outside.wav").exists()
    assert (
        tmp_path / "service" / "jobs" / response.json()["id"] / "upload" / "outside.wav"
    ).exists()


def test_a_text_file_renamed_mp3_is_unsupported(make_client: MakeClient, tmp_path: Path) -> None:
    client = make_client()
    text = tmp_path / "notes.mp3"
    text.write_text("this is not audio\n" * 50)

    response = upload(client, text, content_type="audio/mpeg")

    assert_problem(response, 415, "unsupported_format")
    assert list((tmp_path / "service" / "incoming").iterdir()) == []


def test_an_empty_file_is_unprocessable(make_client: MakeClient, tmp_path: Path) -> None:
    client = make_client()
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")

    assert_problem(upload(client, empty), 422, "empty_audio")


def test_a_file_over_the_limit_is_stopped_while_it_is_copied(
    make_client: MakeClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(*_: Any) -> None:
        raise AssertionError("an oversized upload must not get as far as probing")

    monkeypatch.setattr(api, "probe", unreachable)
    client = make_client(max_file_mb=1)
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (1024 * 1024 + 1))

    assert_problem(upload(client, big), 413, "file_too_large")
    assert list((tmp_path / "service" / "incoming").iterdir()) == []


def test_rejected_uploads_never_become_jobs(make_client: MakeClient, tmp_path: Path) -> None:
    client = make_client(start_workers=False)
    text = tmp_path / "notes.mp3"
    text.write_text("this is not audio\n" * 50)

    upload(client, text)

    assert client.get("/healthz").json()["jobs"] == {"queued": 0, "processing": 0}
    assert list((tmp_path / "service" / "jobs").iterdir()) == []


def test_a_full_queue_answers_429_but_never_refuses_a_repeat(
    make_client: MakeClient, speech: Path, other_speech: Path
) -> None:
    client = make_client(start_workers=False, max_queued_jobs=1)
    first = upload(client, speech)

    full = upload(client, other_speech)
    repeat = upload(client, speech)

    assert first.status_code == 202
    assert_problem(full, 429, "queue_full")
    assert full.headers["retry-after"] == "30"
    assert repeat.status_code == 200
    assert repeat.json()["id"] == first.json()["id"]


def test_an_unknown_job_is_404(make_client: MakeClient) -> None:
    client = make_client()

    assert_problem(client.get(f"{JOBS}/nope"), 404, "not_found")
    assert_problem(client.get(f"{JOBS}/nope/result"), 404, "not_found")


def test_an_unknown_route_is_404_in_the_same_shape(make_client: MakeClient) -> None:
    assert_problem(make_client().get("/v2/anything"), 404, "not_found")


def test_a_result_before_completion_is_a_conflict_that_names_the_state(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(start_workers=False)
    job_id = upload(client, speech).json()["id"]

    body = assert_problem(client.get(f"{JOBS}/{job_id}/result"), 409, "job_not_completed")

    assert body["state"] == "queued"


def test_a_request_without_a_file_is_a_validation_error(make_client: MakeClient) -> None:
    body = assert_problem(make_client().post(JOBS), 422, "validation_error")

    assert any(error["loc"][-1] == "file" for error in body["errors"])


def test_an_unknown_result_format_is_a_validation_error(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(start_workers=False)
    job_id = upload(client, speech).json()["id"]

    response = client.get(f"{JOBS}/{job_id}/result", params={"format": "docx"})

    assert_problem(response, 422, "validation_error")


def test_the_model_is_loaded_before_requests_are_served(make_client: MakeClient) -> None:
    spy = Spy()

    make_client(spy)

    assert spy.warm_ups == 1


def test_health_reports_the_model_and_the_queue(
    make_client: MakeClient, speech: Path, other_speech: Path
) -> None:
    client = make_client(start_workers=False)
    upload(client, speech)
    upload(client, other_speech)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "spy",
        "jobs": {"queued": 2, "processing": 0},
    }


def test_the_accepted_line_carries_the_job_id(
    make_client: MakeClient, speech: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = make_client()

    job_id = upload(client, speech).json()["id"]

    accepted = [line for line in capsys.readouterr().err.splitlines() if "accepted" in line]
    assert len(accepted) == 1
    assert f"job={job_id} " in accepted[0]


def test_a_completed_job_can_be_deleted_with_everything_it_owns(
    make_client: MakeClient, speech: Path, tmp_path: Path
) -> None:
    client = make_client()
    job_id = upload(client, speech).json()["id"]
    wait_for_state(client, job_id, "completed")
    assert (tmp_path / "service" / "jobs" / job_id).is_dir()

    assert client.delete(f"{JOBS}/{job_id}").status_code == 204

    assert_problem(client.get(f"{JOBS}/{job_id}"), 404, "not_found")
    assert_problem(client.get(f"{JOBS}/{job_id}/result"), 404, "not_found")
    assert not (tmp_path / "service" / "jobs" / job_id).exists()


def test_deleting_a_job_frees_the_audio_to_be_submitted_again(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(start_workers=False)
    first = upload(client, speech).json()["id"]

    assert client.delete(f"{JOBS}/{first}").status_code == 204
    again = upload(client, speech)

    assert again.status_code == 202
    assert again.json()["id"] != first


def test_deleting_an_unknown_job_is_404(make_client: MakeClient) -> None:
    assert_problem(make_client().delete(f"{JOBS}/nope"), 404, "not_found")


def test_a_running_job_cannot_be_deleted(make_client: MakeClient, speech: Path) -> None:
    gate = threading.Event()
    spy = Spy(gate=gate)
    client = make_client(spy)
    job_id = upload(client, speech).json()["id"]
    try:
        wait_for(spy.started.is_set)

        assert_problem(client.delete(f"{JOBS}/{job_id}"), 409, "job_running")
        assert state_of(client, job_id) == "processing"
    finally:
        gate.set()

    wait_for_state(client, job_id, "completed")
    assert client.delete(f"{JOBS}/{job_id}").status_code == 204


def test_a_failed_job_can_be_retried_once_the_transcriber_recovers(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(Spy(fail_first=3), retry_attempts=3)
    job_id = upload(client, speech).json()["id"]
    failed = wait_for_state(client, job_id, "failed")
    assert failed["error"]["code"] == "transcription_failed"
    assert_problem(client.get(f"{JOBS}/{job_id}/result"), 409, "job_not_completed")

    response = client.post(f"{JOBS}/{job_id}/retry")

    assert response.status_code == 202
    assert response.headers["location"] == f"{JOBS}/{job_id}"
    done = wait_for_state(client, job_id, "completed")
    assert done["error"] is None
    assert done["attempts"] == 1
    assert client.get(f"{JOBS}/{job_id}/result").status_code == 200


def test_only_a_failed_job_can_be_retried(make_client: MakeClient, speech: Path) -> None:
    client = make_client(start_workers=False)
    job_id = upload(client, speech).json()["id"]

    body = assert_problem(client.post(f"{JOBS}/{job_id}/retry"), 409, "job_not_retryable")

    assert "queued" in body["detail"]


def test_a_job_that_failed_on_bad_input_is_not_retried(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(start_workers=False)
    job_id = upload(client, speech).json()["id"]
    store = service_of(client).store
    store.claim_next()
    store.fail(job_id, ErrorCode.CORRUPT_AUDIO, "damaged part-way through")

    body = assert_problem(client.post(f"{JOBS}/{job_id}/retry"), 409, "job_not_retryable")

    assert "corrupt_audio" in body["detail"]
    assert state_of(client, job_id) == "failed"


def test_a_failed_job_is_not_retried_beside_an_identical_live_one(
    make_client: MakeClient, speech: Path
) -> None:
    client = make_client(start_workers=False)
    old = upload(client, speech).json()["id"]
    store = service_of(client).store
    store.claim_next()
    store.fail(old, ErrorCode.TRANSCRIPTION_FAILED, "gave up")
    replacement = upload(client, speech)
    assert replacement.status_code == 202

    body = assert_problem(client.post(f"{JOBS}/{old}/retry"), 409, "job_not_retryable")

    assert replacement.json()["id"] in body["detail"]


def test_retrying_an_unknown_job_is_404(make_client: MakeClient) -> None:
    assert_problem(make_client().post(f"{JOBS}/nope/retry"), 404, "not_found")
