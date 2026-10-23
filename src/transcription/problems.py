from collections.abc import Mapping
from enum import StrEnum
from http import HTTPStatus
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from transcription.errors import ErrorCode, PipelineError

PROBLEM_JSON = "application/problem+json"

# Pipeline errors that can be answered straight away; anything else is a fault on our side.
_BAD_INPUT_STATUS = {
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_FORMAT: 415,
    ErrorCode.EMPTY_AUDIO: 422,
    ErrorCode.NO_AUDIO_STREAM: 422,
    ErrorCode.AUDIO_TOO_LONG: 422,
    ErrorCode.CORRUPT_AUDIO: 422,
}


class ApiCode(StrEnum):
    """Codes for errors that exist only at the HTTP layer. Clients branch on these, not on the
    message text."""

    VALIDATION_ERROR = "validation_error"
    QUEUE_FULL = "queue_full"
    NOT_FOUND = "not_found"
    JOB_NOT_COMPLETED = "job_not_completed"
    JOB_RUNNING = "job_running"
    JOB_NOT_RETRYABLE = "job_not_retryable"
    HTTP_ERROR = "http_error"


class ApiProblem(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        detail: str,
        *,
        headers: Mapping[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.headers = headers
        self.extra = extra


def status_for(code: ErrorCode) -> int:
    return _BAD_INPUT_STATUS.get(code, 500)


def problem(
    status: int,
    code: str,
    detail: str,
    *,
    headers: Mapping[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "code": str(code),
        **extra,
    }
    return JSONResponse(body, status_code=status, headers=headers, media_type=PROBLEM_JSON)


def install_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiProblem, _api_problem)
    app.add_exception_handler(PipelineError, _pipeline_error)
    app.add_exception_handler(RequestValidationError, _invalid_request)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(Exception, _unexpected_error)


async def _api_problem(_: Request, exc: Exception) -> JSONResponse:
    error = cast(ApiProblem, exc)
    return problem(error.status, error.code, error.detail, headers=error.headers, **error.extra)


async def _pipeline_error(_: Request, exc: Exception) -> JSONResponse:
    error = cast(PipelineError, exc)
    return problem(status_for(error.code), error.code, str(error))


async def _invalid_request(_: Request, exc: Exception) -> JSONResponse:
    errors = [
        {"loc": list(item["loc"]), "message": item["msg"]}
        for item in cast(RequestValidationError, exc).errors()
    ]
    return problem(422, ApiCode.VALIDATION_ERROR, "the request is not valid", errors=errors)


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    error = cast(HTTPException, exc)
    code = ApiCode.NOT_FOUND if error.status_code == 404 else ApiCode.HTTP_ERROR
    return problem(error.status_code, code, str(error.detail), headers=error.headers)


async def _unexpected_error(_: Request, __: Exception) -> JSONResponse:
    return problem(500, ErrorCode.INTERNAL_ERROR, "unexpected error")
