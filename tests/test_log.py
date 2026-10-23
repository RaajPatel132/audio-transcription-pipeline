import logging

import pytest

from transcription.log import bind_job, configure_logging, job_id


def test_lines_carry_the_job_id(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    token = job_id.set("abc123")
    try:
        logging.getLogger("transcription.demo").info("hello")
    finally:
        job_id.reset(token)

    assert "job=abc123 transcription.demo: hello" in capsys.readouterr().err


def test_job_id_defaults_to_dash(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logging.getLogger("transcription.demo").info("hello")

    assert "job=- " in capsys.readouterr().err


def test_level_applies_to_our_loggers_only(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    logging.getLogger("transcription.demo").debug("ours, hidden")
    logging.getLogger("some_library").info("theirs, hidden")
    logging.getLogger("some_library").warning("theirs, shown")

    err = capsys.readouterr().err
    assert "hidden" not in err
    assert "theirs, shown" in err


def test_configuring_twice_installs_one_handler() -> None:
    configure_logging()
    configure_logging()

    ours = [h for h in logging.getLogger().handlers if h.get_name() == "transcription-stderr"]
    assert len(ours) == 1


def test_bind_job_sets_the_id_for_the_block_only() -> None:
    with bind_job("abc123"):
        assert job_id.get() == "abc123"

    assert job_id.get() == "-"


def test_bind_job_restores_the_id_after_an_error() -> None:
    with pytest.raises(RuntimeError), bind_job("abc123"):
        raise RuntimeError("boom")

    assert job_id.get() == "-"
