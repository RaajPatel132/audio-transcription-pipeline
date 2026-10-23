import pytest

from transcription.errors import ErrorCode, PipelineError


def test_error_keeps_message_and_stable_code() -> None:
    err = PipelineError(ErrorCode.EMPTY_AUDIO, "the file is empty")

    assert str(err) == "the file is empty"
    assert err.code == "empty_audio"
    assert not err.retryable


@pytest.mark.parametrize(
    ("code", "bad_input"),
    [
        (ErrorCode.EMPTY_AUDIO, True),
        (ErrorCode.UNSUPPORTED_FORMAT, True),
        (ErrorCode.NO_AUDIO_STREAM, True),
        (ErrorCode.CORRUPT_AUDIO, True),
        (ErrorCode.FILE_TOO_LARGE, True),
        (ErrorCode.AUDIO_TOO_LONG, True),
        (ErrorCode.MODEL_UNAVAILABLE, False),
        (ErrorCode.TRANSCRIPTION_FAILED, False),
        (ErrorCode.INTERNAL_ERROR, False),
    ],
)
def test_bad_input_classification(code: ErrorCode, bad_input: bool) -> None:
    assert PipelineError(code, "x").is_bad_input is bad_input
