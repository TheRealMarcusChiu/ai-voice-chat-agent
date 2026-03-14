import asyncio
from kokoro import KPipeline
import sounddevice as sd
from typing import AsyncIterator


pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')


async def speak_stream(text_stream: AsyncIterator[str]) -> None:
    """
    Consume an async text stream, synthesize TTS incrementally,
    and play audio chunks as they're ready.
    """
    buffer = ""
    sentence_endings = {'.', '!', '?', '\n'}

    async def synthesize_and_play(chunk_text: str) -> None:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return
        generator = pipeline(chunk_text, voice='af_bella', speed=1.2)
        for i, (gs, ps, audio) in enumerate(generator):
            print(f"  Playing audio segment {i} for: {chunk_text[:40]!r}...")
            sd.play(audio, samplerate=24000)
            sd.wait()

    async for token in text_stream:
        buffer += token

        # Flush buffer whenever we hit a sentence boundary
        while True:
            flush_idx = -1
            for idx, ch in enumerate(buffer):
                if ch in sentence_endings:
                    flush_idx = idx
                    break
            if flush_idx == -1:
                break

            sentence = buffer[: flush_idx + 1]
            buffer = buffer[flush_idx + 1 :]
            await synthesize_and_play(sentence)

    # Flush any remaining text
    if buffer.strip():
        await synthesize_and_play(buffer)


# ---------------------------------------------------------------------------
# Example: simulate an async LLM token stream
# ---------------------------------------------------------------------------

async def fake_llm_stream() -> AsyncIterator[str]:
    """Simulates tokens arriving one-by-one, like an LLM streaming response."""
    tokens = "Using uv is a great choice. It's much faster and handles deep dependencies better than standard tools.".split()
    for token in tokens:
        await asyncio.sleep(0.05)   # simulate network latency
        yield token + " "  # add space after each token


async def main() -> None:
    await speak_stream(fake_llm_stream())


if __name__ == "__main__":
    asyncio.run(main())
