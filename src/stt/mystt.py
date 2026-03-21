from typing import AsyncGenerator, Optional

import numpy as np
import queue
import threading
import asyncio
from faster_whisper import WhisperModel
import uuid as uuid_lib
from dataclasses import dataclass


@dataclass
class Transcript:
    id: int
    transcript: Optional[str] = None


class MySTT:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 256

    def __init__(
        self,
        on_user_transcript_start=None,
        on_user_transcript_unfinished=None,
        on_user_transcript_end=None,
        silence_threshold=0.001,
        silence_duration=0.5,
        model_size="tiny",
        language="en",
        device="cpu",
    ):
        self.on_user_transcript_start = on_user_transcript_start
        self.on_user_transcript_unfinished = on_user_transcript_unfinished
        self.on_user_transcript_end = on_user_transcript_end
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.language = language
        self.audio_queue = queue.Queue()
        if device == "cpu":
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        elif device == "cuda":
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def transcribe_audio(self, audio_source: AsyncGenerator[np.ndarray, None]):
        asyncio.run_coroutine_threadsafe(self._queue_audio(audio_source), asyncio.get_event_loop())
        threading.Thread(target=self._transcribe_audio, daemon=True).start()

    async def _queue_audio(self, audio_source: AsyncGenerator[np.ndarray, None]):
        async for chunk in audio_source:
            self.audio_queue.put(chunk.flatten().astype(np.float32))

    def _transcribe_audio(self):
        while True:
            self._transcribe_a_sentence()

    def _transcribe_a_sentence(self):
        id = str(uuid_lib.uuid4())
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

        self.on_user_transcript_start(Transcript(id=id))

        # Once we detect speech, keep recording until we get sustained silence
        silence_chunk_count_duration = int(self.silence_duration * self.SAMPLE_RATE / self.CHUNK_SIZE)
        silence_chunk_count = 0
        while True:
            chunk = self.audio_queue.get()
            buffer.append(chunk)
            silence_chunk_count += 1

            transcript = None
            if len(buffer) % chunks_per_transcription == 0:
                recent = np.concatenate(buffer[-chunks_per_transcription:])
                has_speech = bool(self._transcribe(recent, realtime=True).strip())
                if has_speech:
                    silence_chunk_count = 0

                if self.on_user_transcript_unfinished:
                    partial_audio = np.concatenate(buffer)
                    transcript = self._transcribe(partial_audio, realtime=True)
                    self.on_user_transcript_unfinished(Transcript(id=id, transcript=transcript))

            if silence_chunk_count >= silence_chunk_count_duration:
                self.on_user_transcript_end(Transcript(id=id, transcript=transcript))
                break

    def _transcribe(self, audio: np.ndarray, realtime=False) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=1 if realtime else 5,
            vad_filter=True,
        )
        return " ".join(s.text.strip() for s in segments)