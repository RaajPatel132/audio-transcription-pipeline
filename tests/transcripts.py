from transcription.schema import AudioInfo, ChunkInfo, Segment, Transcript


def make_transcript(segments: list[Segment] | None = None) -> Transcript:
    return Transcript(
        source=AudioInfo(
            filename="talk.wav",
            size_bytes=1000,
            sha256="0" * 64,
            container="wav",
            codec="pcm_s16le",
            sample_rate=16000,
            channels=1,
            duration_s=10.0,
        ),
        model="base.en",
        language="en",
        duration_s=10.0,
        segments=segments if segments is not None else [Segment(id=0, start=0, end=1.5, text="hi")],
        chunks=[ChunkInfo(index=0, start=0, end=10)],
        elapsed_s=0.5,
    )
