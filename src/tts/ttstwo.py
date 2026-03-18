from typing import AsyncIterator
from kokoro import KPipeline
import sounddevice as sd


pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

async def speak_stream2(text_stream: AsyncIterator[str]) -> None:
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
        generator = pipeline(chunk_text, voice='af_bella', speed=1.3)
        for i, (gs, ps, audio) in enumerate(generator):
            sd.play(audio, samplerate=24000)
            sd.wait()

    async for token in text_stream:
        # yield token

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
