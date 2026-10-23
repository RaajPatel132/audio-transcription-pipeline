from enum import StrEnum


class ErrorCode(StrEnum):
    EMPTY_AUDIO = "empty_audio"
    UNSUPPORTED_FORMAT = "unsupported_format"
    NO_AUDIO_STREAM = "no_audio_stream"
    CORRUPT_AUDIO = "corrupt_audio"
    FILE_TOO_LARGE = "file_too_large"
    AUDIO_TOO_LONG = "audio_too_long"
    MODEL_UNAVAILABLE = "model_unavailable"
    TRANSCRIPTION_FAILED = "transcription_failed"


# The caller sent something we cannot process. Everything else is a failure on our side.
BAD_INPUT = frozenset(
    {
        ErrorCode.EMPTY_AUDIO,
        ErrorCode.UNSUPPORTED_FORMAT,
        ErrorCode.NO_AUDIO_STREAM,
        ErrorCode.CORRUPT_AUDIO,
        ErrorCode.FILE_TOO_LARGE,
        ErrorCode.AUDIO_TOO_LONG,
    }
)


class PipelineError(Exception):
    """An error the pipeline raises on purpose. `code` is stable; callers branch on it."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    @property
    def is_bad_input(self) -> bool:
        return self.code in BAD_INPUT
