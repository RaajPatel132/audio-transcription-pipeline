from collections.abc import Callable
from itertools import groupby

from transcription.schema import Segment

LOOP_RUN = 3
MIN_FRAGMENT_S = 1.0
MAX_MERGE_GAP_S = 0.5


def clean(segments: list[Segment]) -> list[Segment]:
    """Tidy the transcript for consumers. The steps run in this order."""
    steps: list[Callable[[list[Segment]], list[Segment]]] = [
        normalise_whitespace,
        drop_empty,
        collapse_loops,
        merge_fragments,
        renumber,
    ]
    for step in steps:
        segments = step(segments)
    return segments


def normalise_whitespace(segments: list[Segment]) -> list[Segment]:
    return [s.model_copy(update={"text": " ".join(s.text.split())}) for s in segments]


def drop_empty(segments: list[Segment]) -> list[Segment]:
    return [s for s in segments if s.text]


def collapse_loops(segments: list[Segment]) -> list[Segment]:
    """Keep the first of three or more identical segments in a row.

    Two in a row is kept because speech does that ("No. No."); three or more is the typical
    Whisper repetition loop.
    """
    kept: list[Segment] = []
    for _, group in groupby(segments, key=lambda s: s.text.casefold()):
        run = list(group)
        kept.extend(run[:1] if len(run) >= LOOP_RUN else run)
    return kept


def merge_fragments(segments: list[Segment]) -> list[Segment]:
    """Join a fragment shorter than a second onto the segment before it, if it follows closely."""
    merged: list[Segment] = []
    for segment in segments:
        if (
            merged
            and segment.end - segment.start < MIN_FRAGMENT_S
            and segment.start - merged[-1].end < MAX_MERGE_GAP_S
        ):
            previous = merged.pop()
            segment = previous.model_copy(
                update={"end": segment.end, "text": f"{previous.text} {segment.text}"}
            )
        merged.append(segment)
    return merged


def renumber(segments: list[Segment]) -> list[Segment]:
    return [s.model_copy(update={"id": i}) for i, s in enumerate(segments)]
