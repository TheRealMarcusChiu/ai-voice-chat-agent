import numpy as np
import queue
import threading
import asyncio
import inspect
from faster_whisper import WhisperModel
import uuid as uuid_lib


class MySTT:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 256

    def __init__(
        self,
        on_user_transcript_unfinished=None,
        silence_threshold=0.001,
        silence_duration=0.5,
        model_size="tiny",
        language="en",
        loop=None,
        device="cpu",
    ):
        self.on_user_transcript_unfinished = on_user_transcript_unfinished
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.language = language
        self.loop = loop  # will be set on first call to text()
        self.audio_queue = queue.Queue()
        if device == "cpu":
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        elif device == "cuda":
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def feed(self, chunk: np.ndarray):
        self.audio_queue.put(chunk.flatten().astype(np.float32))

    async def text(self, on_final, *args):
        # Capture the running loop so background threads can schedule coroutines
        self.loop = asyncio.get_running_loop()

        audio, utterance_id = await self.loop.run_in_executor(None, self._record_utterance)
        result = self._transcribe(audio)
        if result.strip():
            if inspect.iscoroutinefunction(on_final):
                await on_final(result, utterance_id, *args)
            else:
                on_final(result, utterance_id, *args)

    def _record_utterance(self) -> tuple[np.ndarray, str]:
        utterance_id = str(uuid_lib.uuid4())
        buffer = []
        chunks_per_transcription = max(1, int(0.5 * self.SAMPLE_RATE / self.CHUNK_SIZE))

        # Wait until user starts speaking (i.e., until we get a non-silent transcription)
        while True:
            chunk = self.audio_queue.get()
            buffer.append(chunk)
            if len(buffer) % chunks_per_transcription == 0:
                audio = np.concatenate(buffer)
                if self._transcribe(audio, realtime=True).strip():
                    break
            if len(buffer) > chunks_per_transcription * 4:
                buffer = buffer[-chunks_per_transcription:]

        # Once we detect speech, keep recording until we get sustained silence
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

                if self.on_user_transcript_unfinished and self.loop:
                    partial_audio = np.concatenate(buffer)
                    uid = utterance_id

                    def _fire(a=partial_audio, u=uid):
                        result = self._transcribe(a, realtime=True)
                        # on_user_transcript_unfinished is async — schedule it safely
                        future = asyncio.run_coroutine_threadsafe(
                            self.on_user_transcript_unfinished(result, u),
                            self.loop,
                        )
                        # Optional: log errors from the scheduled coroutine
                        try:
                            future.result(timeout=10)
                        except Exception as e:
                            print(f"[STT] realtime callback error: {e}")

                    threading.Thread(target=_fire, daemon=True).start()

            if silence_chunk_count >= silence_chunk_count_duration:
                break

        return np.concatenate(buffer), utterance_id

    def _transcribe(self, audio: np.ndarray, realtime=False) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=1 if realtime else 5,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments)