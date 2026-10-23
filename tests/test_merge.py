from transcription.merge import stitch
from transcription.schema import ChunkInfo, Segment
from transcription.transcribers import ChunkResult


def seg(start: float, end: float, text: str = "x") -> Segment:
    return Segment(id=0, start=start, end=end, text=text)


def result(*segments: Segment) -> ChunkResult:
    return ChunkResult(segments=list(segments), language="en")


CHUNKS = [ChunkInfo(index=0, start=0, end=30), ChunkInfo(index=1, start=30, end=60)]


def test_adds_each_chunks_start_time() -> None:
    stitched = stitch(CHUNKS, [result(seg(1, 2, "a")), result(seg(1, 2, "b"))])

    assert [(s.start, s.end, s.text) for s in stitched] == [(1, 2, "a"), (31, 32, "b")]


def test_segment_running_past_its_chunk_is_clamped() -> None:
    stitched = stitch(CHUNKS, [result(seg(28, 33)), result(seg(2, 3))])

    assert [(s.start, s.end) for s in stitched] == [(28, 30), (32, 33)]


def test_overlapping_segments_are_made_to_follow_each_other() -> None:
    stitched = stitch(CHUNKS[:1], [result(seg(1, 5), seg(4, 8))])

    assert [(s.start, s.end) for s in stitched] == [(1, 5), (5, 8)]


def test_segments_with_nothing_left_are_dropped() -> None:
    stitched = stitch(CHUNKS[:1], [result(seg(1, 5), seg(2, 4), seg(6, 6))])

    assert [(s.start, s.end) for s in stitched] == [(1, 5)]


def test_output_is_sorted_with_contiguous_ids() -> None:
    stitched = stitch(CHUNKS, [result(seg(10, 12, "b"), seg(1, 2, "a")), result(seg(1, 2, "c"))])

    assert [s.text for s in stitched] == ["a", "b", "c"]
    assert [s.id for s in stitched] == [0, 1, 2]
