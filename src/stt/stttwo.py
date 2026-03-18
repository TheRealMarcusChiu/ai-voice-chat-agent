import sounddevice as sd
import numpy as np
import queue
import threading
import asyncio
import inspect
from faster_whisper import WhisperModel


class AudioToTextRecorder2:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 256

    def __init__(
        self,
        on_realtime_transcription_update=None,
        silence_threshold=0.001,
        silence_duration=0.5,
        model_size="tiny",
        language="en",
        source="mic",  # "mic" or "websocket"
        loop=None,
    ):
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.language = language
        self.source = source
        self.loop = loop

        self.audio_queue = queue.Queue()
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

        self._stream = None
        if source == "mic":
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.CHUNK_SIZE,
                callback=self._audio_callback,
            )
            self._stream.start()

    def feed(self, chunk: np.ndarray):
        """Push audio from an external source (e.g. websocket) into the queue.
        chunk should be a 1-D float32 numpy array at SAMPLE_RATE Hz."""
        if self.source != "websocket":
            raise RuntimeError("feed() is only available when source='websocket'")
        self.audio_queue.put(chunk.flatten().astype(np.float32))

    def _audio_callback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy().flatten())

    def _is_silent(self, chunk):
        return np.abs(chunk).mean() < self.silence_threshold

    def _transcribe(self, audio: np.ndarray, realtime=False) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=1 if realtime else 5,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments)

    def _record_utterance(self) -> np.ndarray:
        silence_chunks = int(self.silence_duration * self.SAMPLE_RATE / self.CHUNK_SIZE)
        buffer = []
        silent_count = 0

        while True:
            chunk = self.audio_queue.get()
            if not self._is_silent(chunk):
                buffer.append(chunk)
                break

        while True:
            chunk = self.audio_queue.get()
            buffer.append(chunk)

            if self._is_silent(chunk):
                silent_count += 1
            else:
                silent_count = 0

            if self.on_realtime_transcription_update:
                if len(buffer) % max(1, int(0.5 * self.SAMPLE_RATE / self.CHUNK_SIZE)) == 0:
                    partial = np.concatenate(buffer)
                    def _fire(a=partial):
                        result = self._transcribe(a, realtime=True)
                        cb = self.on_realtime_transcription_update(result)
                        if inspect.iscoroutine(cb) and self.loop:
                            asyncio.run_coroutine_threadsafe(cb, self.loop)
                    threading.Thread(target=_fire, daemon=True).start()

            if silent_count >= silence_chunks:
                break

        return np.concatenate(buffer)

    # def _record_utterance(self) -> np.ndarray:
        silence_chunks = int(self.silence_duration * self.SAMPLE_RATE / self.CHUNK_SIZE)
        buffer = []
        silent_count = 0

        while True:
            chunk = self.audio_queue.get()
            if not self._is_silent(chunk):
                buffer.append(chunk)
                break

        while True:
            chunk = self.audio_queue.get()
            buffer.append(chunk)

            if self._is_silent(chunk):
                silent_count += 1
            else:
                silent_count = 0

            if self.on_realtime_transcription_update:
                if len(buffer) % max(1, int(0.5 * self.SAMPLE_RATE / self.CHUNK_SIZE)) == 0:
                    partial = np.concatenate(buffer)
                    threading.Thread(
                        target=lambda a=partial: self.on_realtime_transcription_update(
                            self._transcribe(a, realtime=True)
                        ),
                        daemon=True,
                    ).start()

            if silent_count >= silence_chunks:
                break

        return np.concatenate(buffer)

    async def text(self, on_final, *args):
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, self._record_utterance)
        result = self._transcribe(audio)
        if result.strip():
            if inspect.iscoroutinefunction(on_final):
                await on_final(result, *args)
            else:
                on_final(result, *args)

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()