from collections.abc import Sequence

from transcription.schema import ChunkInfo, Segment
from transcription.transcribers import ChunkResult


def stitch(chunks: Sequence[ChunkInfo], results: Sequence[ChunkResult]) -> list[Segment]:
    """Put the chunks' segments on one timeline.

    Chunks do not overlap and are cut in quiet moments, so adding each chunk's start time is
    exact. The checks below only guard the edges: a segment is kept inside its chunk, may not
    start before the previous one ended, and is dropped if nothing is left of it.
    """
    stitched: list[Segment] = []
    previous_end = 0.0
    for chunk, result in zip(chunks, results, strict=True):
        for segment in sorted(result.segments, key=lambda s: (s.start, s.end)):
            candidate = Segment(
                id=len(stitched),
                start=max(min(segment.start + chunk.start, chunk.end), previous_end),
                end=min(segment.end + chunk.start, chunk.end),
                text=segment.text,
            )
            if candidate.end <= candidate.start:
                continue
            stitched.append(candidate)
            previous_end = candidate.end
    return stitched
