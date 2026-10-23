import contextvars
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

from transcription.chunker import plan_chunks, read_chunk
from transcription.cleanup import clean
from transcription.config import Settings
from transcription.ingest import normalise, probe
from transcription.merge import stitch
from transcription.retry import call_with_retry
from transcription.schema import ChunkInfo, Transcript
from transcription.transcribers import ChunkResult, Transcriber

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


class ChunkStore(Protocol):
    """Where finished chunks are kept, so a restarted job can skip them."""

    def get(self, chunk: ChunkInfo) -> ChunkResult | None: ...

    def put(self, chunk: ChunkInfo, result: ChunkResult) -> None: ...


class MemoryChunkStore:
    """Keyed by the chunk's span rather than its index, so a change in chunking settings can
    never match a result to the wrong audio."""

    def __init__(self) -> None:
        self._results: dict[tuple[float, float], ChunkResult] = {}

    def get(self, chunk: ChunkInfo) -> ChunkResult | None:
        return self._results.get((chunk.start, chunk.end))

    def put(self, chunk: ChunkInfo, result: ChunkResult) -> None:
        self._results[(chunk.start, chunk.end)] = result


def transcribe_file(
    path: Path,
    settings: Settings,
    transcriber: Transcriber,
    work_dir: Path,
    store: ChunkStore | None = None,
    on_progress: ProgressCallback | None = None,
) -> Transcript:
    """Run the whole pipeline on one audio file. `work_dir` holds the normalised audio."""
    started = time.perf_counter()
    source = probe(path, settings)
    wav = work_dir / "audio.wav"
    duration = normalise(path, wav, settings)
    chunks = plan_chunks(wav, settings)
    logger.info("%s: %.1fs of audio in %d chunk(s)", path.name, duration, len(chunks))

    results = _transcribe_chunks(
        wav, chunks, settings, transcriber, store or MemoryChunkStore(), on_progress
    )
    return Transcript(
        source=source.model_copy(update={"duration_s": duration}),
        model=transcriber.name,
        language=_dominant_language(chunks, results),
        duration_s=duration,
        segments=clean(stitch(chunks, results)),
        chunks=chunks,
        elapsed_s=time.perf_counter() - started,
    )


def _transcribe_chunks(
    wav: Path,
    chunks: list[ChunkInfo],
    settings: Settings,
    transcriber: Transcriber,
    store: ChunkStore,
    on_progress: ProgressCallback | None,
) -> list[ChunkResult]:
    def work(chunk: ChunkInfo) -> ChunkResult:
        label = f"chunk {chunk.index + 1}/{len(chunks)}"
        if (saved := store.get(chunk)) is not None:
            logger.info("%s: using saved result", label)
            return saved
        samples = read_chunk(wav, chunk)
        result = call_with_retry(
            lambda: transcriber.transcribe(samples, settings.forced_language),
            attempts=settings.retry_attempts,
            base_s=settings.retry_base_s,
            max_s=settings.retry_max_s,
            what=label,
        )
        store.put(chunk, result)
        logger.info("%s: %d segments", label, len(result.segments))
        return result

    results: dict[int, ChunkResult] = {}
    with ThreadPoolExecutor(max_workers=settings.chunk_workers) as pool:
        # Each task gets its own copy of the context, so log lines keep the job ID.
        futures = {
            pool.submit(contextvars.copy_context().run, work, chunk): chunk for chunk in chunks
        }
        try:
            for future in as_completed(futures):
                results[futures[future].index] = future.result()
                if on_progress:
                    on_progress(len(results), len(chunks))
        except BaseException:
            for pending in futures:
                pending.cancel()
            raise
    return [results[chunk.index] for chunk in chunks]


def _dominant_language(chunks: Sequence[ChunkInfo], results: Sequence[ChunkResult]) -> str:
    """The language covering the most audio."""
    seconds: defaultdict[str, float] = defaultdict(float)
    for chunk, result in zip(chunks, results, strict=True):
        seconds[result.language] += chunk.end - chunk.start
    return max(seconds, key=lambda language: seconds[language])
