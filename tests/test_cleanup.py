from transcription.cleanup import (
    clean,
    collapse_loops,
    drop_empty,
    merge_fragments,
    normalise_whitespace,
    renumber,
)
from transcription.schema import Segment


def seg(start: float, end: float, text: str, id: int = 0) -> Segment:
    return Segment(id=id, start=start, end=end, text=text)


def test_whitespace_is_trimmed_and_collapsed() -> None:
    assert normalise_whitespace([seg(0, 2, "  hello \n  world ")])[0].text == "hello world"


def test_empty_segments_are_dropped() -> None:
    assert [s.text for s in drop_empty([seg(0, 1, ""), seg(1, 2, "hi")])] == ["hi"]


def test_two_repeats_are_kept() -> None:
    segments = [seg(0, 2, "No."), seg(2, 4, "No.")]

    assert collapse_loops(segments) == segments


def test_three_or_more_repeats_collapse_to_the_first() -> None:
    segments = [seg(0, 2, "Thanks."), seg(2, 4, "thanks."), seg(4, 6, "Thanks."), seg(6, 8, "Bye.")]

    assert collapse_loops(segments) == [segments[0], segments[3]]


def test_repeats_that_are_not_consecutive_are_kept() -> None:
    segments = [seg(0, 2, "a"), seg(2, 4, "b"), seg(4, 6, "a"), seg(6, 8, "b"), seg(8, 10, "a")]

    assert collapse_loops(segments) == segments


def test_short_fragment_joins_the_previous_segment() -> None:
    merged = merge_fragments([seg(0, 4, "I think so"), seg(4.2, 4.8, "maybe")])

    assert [(s.start, s.end, s.text) for s in merged] == [(0, 4.8, "I think so maybe")]


def test_short_fragment_after_a_long_gap_stays_separate() -> None:
    segments = [seg(0, 4, "first"), seg(5, 5.5, "second")]

    assert merge_fragments(segments) == segments


def test_long_segment_is_never_merged() -> None:
    segments = [seg(0, 4, "first"), seg(4.1, 6, "second")]

    assert merge_fragments(segments) == segments


def test_leading_fragment_has_nothing_to_join() -> None:
    segments = [seg(0, 0.4, "Hm."), seg(0.5, 4, "Right.")]

    assert merge_fragments(segments)[0].text == "Hm."


def test_renumber_starts_at_zero() -> None:
    assert [s.id for s in renumber([seg(0, 1, "a", id=7), seg(1, 2, "b", id=9)])] == [0, 1]


def test_clean_applies_every_step() -> None:
    segments = [
        seg(0, 4, " Hello   there "),
        seg(4, 5, "  "),
        seg(5, 9, "Okay."),
        seg(9, 13, "Okay."),
        seg(13, 17, "Okay."),
        seg(30, 34, "right"),
        seg(34.2, 34.6, "sure"),
    ]

    cleaned = clean(segments)

    assert [(s.id, s.text) for s in cleaned] == [
        (0, "Hello there"),
        (1, "Okay."),
        (2, "right sure"),
    ]
