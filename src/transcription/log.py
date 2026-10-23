import logging
from contextvars import ContextVar

# Set once per job (or CLI run); every log line then carries it without passing it around.
job_id: ContextVar[str] = ContextVar("job_id", default="-")

_HANDLER_NAME = "transcription-stderr"
_FORMAT = "%(asctime)s %(levelname)s job=%(job_id)s %(name)s: %(message)s"


class _JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = job_id.get()
        return True


def configure_logging(level: str | int = "INFO") -> None:
    """Send logs to stderr. Our own loggers use `level`; libraries only report warnings."""
    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(_JobIdFilter())
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    for old in [h for h in root.handlers if h.get_name() == _HANDLER_NAME]:
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("transcription").setLevel(level)
