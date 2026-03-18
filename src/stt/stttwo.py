import numpy as np
import queue
import threading
import asyncio
import inspect
from faster_whisper import WhisperModel
import uuid as uuid_lib


class AudioToTextRecorder2:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 256

    def __init__(
        self,
        on_realtime_transcription_update=None,
        silence_threshold=0.001,
        silence_duration=1.0,
        model_size="tiny",
        language="en",
        loop=None,
    ):
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.language = language
        self.loop = loop
        self.audio_queue = queue.Queue()
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def feed(self, chunk: np.ndarray):
        """Push audio from an external source into the queue.
        chunk should be a 1-D float32 numpy array at SAMPLE_RATE Hz."""
        self.audio_queue.put(chunk.flatten().astype(np.float32))

    async def text(self, on_final, *args):
        loop = asyncio.get_running_loop()
        audio, utterance_id = await loop.run_in_executor(None, self._record_utterance)
        result = self._transcribe(audio)
        if result.strip():
            if inspect.iscoroutinefunction(on_final):
                await on_final(result, utterance_id, *args)
            else:
                on_final(result, utterance_id, *args)

    def _record_utterance(self) -> tuple[np.ndarray, str]:
        utterance_id = str(uuid_lib.uuid4())   # <-- born here, once
        buffer = []
        chunks_per_transcription = max(1, int(0.5 * self.SAMPLE_RATE / self.CHUNK_SIZE))

        while True:
            chunk = self.audio_queue.get()
            buffer.append(chunk)
            if len(buffer) % chunks_per_transcription == 0:
                audio = np.concatenate(buffer)
                if self._transcribe(audio, realtime=True).strip():
                    break
            if len(buffer) > chunks_per_transcription * 4:
                buffer = buffer[-chunks_per_transcription:]

        silence_chunk_count_duration = int(self.silence_duration * self.SAMPLE_RATE / self.CHUNK_SIZE)
        silence_chunk_count = 0
        while True:
            chunk = self.audio_queue.get()
            buffer.append(chunk)
            silence_chunk_count += 1

            if len(buffer) % chunks_per_transcription == 0:
                recent = np.concatenate(buffer[-chunks_per_transcription:])
                has_speech = bool(self._transcribe(recent, realtime=True).strip())
                if has_speech:
                    silence_chunk_count = 0

                if self.on_realtime_transcription_update:
                    partial = np.concatenate(buffer)
                    uid = utterance_id  # capture in closure
                    def _fire(a=partial, u=uid):
                        result = self._transcribe(a, realtime=True)
                        cb = self.on_realtime_transcription_update(result, u)  # pass uuid
                        if inspect.iscoroutine(cb) and self.loop:
                            asyncio.run_coroutine_threadsafe(cb, self.loop)
                    threading.Thread(target=_fire, daemon=True).start()

            if silence_chunk_count >= silence_chunk_count_duration:
                break

        return np.concatenate(buffer), utterance_id   # <-- return both

    def _transcribe(self, audio: np.ndarray, realtime=False) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=1 if realtime else 5,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments)
