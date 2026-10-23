import logging
import random
import time
from collections.abc import Callable

from transcription.errors import ErrorCode, PipelineError

logger = logging.getLogger(__name__)


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    attempts: int,
    base_s: float,
    max_s: float,
    what: str = "call",
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> T:
    """Call `fn`, retrying failures with exponential backoff and full jitter.

    The wait before retry n is `rand() * min(max_s, base_s * 2**n)`, so parallel retries spread
    out instead of firing together. A `PipelineError` that is not retryable is raised at once.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except PipelineError as exc:
            if not exc.retryable:
                raise
            error: Exception = exc
        except Exception as exc:
            error = exc

        logger.warning("%s failed (attempt %d/%d): %s", what, attempt + 1, attempts, error)
        if attempt + 1 < attempts:
            sleep(rand() * min(max_s, base_s * 2**attempt))

    raise PipelineError(
        ErrorCode.TRANSCRIPTION_FAILED, f"{what} failed after {attempts} attempts: {error}"
    ) from error
