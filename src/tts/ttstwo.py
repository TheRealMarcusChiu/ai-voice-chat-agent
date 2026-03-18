from typing import AsyncIterator, Callable, Awaitable
import asyncio
from kokoro import KPipeline
import numpy as np


pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

# Callback type: receives raw PCM bytes (float32, 24000 Hz) and a message id
AudioChunkCallback = Callable[[bytes, str], Awaitable[None]]


class TextToSpeechStreamer:
    """
    Consumes an async text stream, synthesizes TTS incrementally sentence-by-sentence,
    and emits audio chunks via a callback (e.g. sending over a WebSocket).
    """

    SENTENCE_ENDINGS = {'.', '!', '?', '\n'}

    def __init__(
        self,
        on_audio_chunk: AudioChunkCallback,
        voice: str = 'af_bella',
        speed: float = 1.3,
        samplerate: int = 24000,
    ):
        """
        Args:
            on_audio_chunk: async callback(audio_bytes: bytes, msg_id: str) called
                            for each synthesized audio chunk.
            voice:          Kokoro voice ID.
            speed:          TTS speed multiplier.
            samplerate:     Output sample rate in Hz.
        """
        self.on_audio_chunk = on_audio_chunk
        self.voice = voice
        self.speed = speed
        self.samplerate = samplerate

    async def _synthesize_and_emit(self, chunk_text: str, msg_id: str) -> None:
        """Synthesize a single sentence and fire the callback for every audio segment."""
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return

        loop = asyncio.get_running_loop()

        # Run the blocking Kokoro pipeline in a thread-pool executor
        def _run_pipeline():
            return list(pipeline(chunk_text, voice=self.voice, speed=self.speed))

        segments = await loop.run_in_executor(None, _run_pipeline)

        for _, _, audio in segments:
            # Kokoro may return a PyTorch Tensor or a numpy array depending on version
            if hasattr(audio, "numpy"):
                audio = audio.numpy()  # Tensor → numpy
            audio_bytes = np.asarray(audio, dtype=np.float32).tobytes()
            await self.on_audio_chunk(audio_bytes, msg_id)

    async def speak_stream(self, text_stream: AsyncIterator[str], msg_id: str) -> None:
        """
        Consume *text_stream*, buffer tokens until a sentence boundary is reached,
        then synthesize and emit audio.  Any leftover text is flushed at the end.

        Args:
            text_stream: async iterator of text tokens from the LLM.
            msg_id:      identifier forwarded to every on_audio_chunk callback call
                         so the client can correlate audio with a specific response.
        """
        buffer = ""

        async for token in text_stream:
            buffer += token

            # Flush complete sentences as they accumulate
            while True:
                flush_idx = -1
                for idx, ch in enumerate(buffer):
                    if ch in self.SENTENCE_ENDINGS:
                        flush_idx = idx
                        break
                if flush_idx == -1:
                    break

                sentence = buffer[: flush_idx + 1]
                buffer = buffer[flush_idx + 1:]
                await self._synthesize_and_emit(sentence, msg_id)

        # Flush any trailing text that didn't end with punctuation
        if buffer.strip():
            await self._synthesize_and_emit(buffer, msg_id)