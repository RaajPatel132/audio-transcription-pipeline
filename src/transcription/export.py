import html
from typing import Literal

from transcription.schema import Segment, Transcript

Format = Literal["json", "srt", "vtt", "txt"]

PARAGRAPH_GAP_S = 2.0


def render(transcript: Transcript, fmt: Format) -> str:
    return _RENDERERS[fmt](transcript)


def to_json(transcript: Transcript) -> str:
    return transcript.model_dump_json(indent=2) + "\n"


def to_srt(transcript: Transcript) -> str:
    cues = [
        f"{number}\n{_cue_times(s, ',')}\n{s.text}"
        for number, s in enumerate(transcript.segments, start=1)
    ]
    return _join_blocks(cues)


def to_vtt(transcript: Transcript) -> str:
    cues = [
        f"{_cue_times(s, '.')}\n{html.escape(s.text, quote=False)}" for s in transcript.segments
    ]
    return _join_blocks(["WEBVTT", *cues])


def to_txt(transcript: Transcript) -> str:
    paragraphs: list[list[str]] = []
    previous_end: float | None = None
    for segment in transcript.segments:
        if previous_end is None or segment.start - previous_end >= PARAGRAPH_GAP_S:
            paragraphs.append([])
        paragraphs[-1].append(segment.text)
        previous_end = segment.end
    return _join_blocks([" ".join(p) for p in paragraphs])


def format_timestamp(seconds: float, separator: str) -> str:
    # Round once, in whole milliseconds, so 59.9996 s is 00:01:00 and never 00:00:59,1000.
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _cue_times(segment: Segment, separator: str) -> str:
    start = format_timestamp(segment.start, separator)
    return f"{start} --> {format_timestamp(segment.end, separator)}"


def _join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(blocks) + "\n" if blocks else ""


_RENDERERS = {"json": to_json, "srt": to_srt, "vtt": to_vtt, "txt": to_txt}
