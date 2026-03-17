import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.001  # Adjust this threshold based on your microphone sensitivity and environment noise level
SILENCE_DURATION = 0.5     # seconds of silence = end of utterance
CHUNK_SIZE = 512           # frames per callback


model = WhisperModel("tiny", device="cpu", compute_type="int8")
audio_queue = queue.Queue()


def on_realtime(text):
    print(f"\r🔴 {text}    ", end="", flush=True)


def on_final(text):
    print(f"\n✅ {text}", flush=True)


def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy().flatten())


def is_silent(chunk):
    return np.abs(chunk).mean() < SILENCE_THRESHOLD


def transcribe(audio: np.ndarray, realtime=False) -> str:
    segments, _ = model.transcribe(
        audio.astype(np.float32),
        language="en",
        beam_size=1 if realtime else 5,
        vad_filter=True,
    )
    return " ".join(s.text.strip() for s in segments)


def record_utterance() -> np.ndarray:
    """Block until speech starts, then collect until silence."""
    silence_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
    buffer = []
    silent_count = 0

    # Wait for speech
    while True:
        chunk = audio_queue.get()
        if not is_silent(chunk):
            buffer.append(chunk)
            break

    # Collect until trailing silence
    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)

        if is_silent(chunk):
            silent_count += 1
        else:
            silent_count = 0

        # Emit realtime preview every ~0.5s
        if len(buffer) % max(1, int(0.5 * SAMPLE_RATE / CHUNK_SIZE)) == 0:
            partial = np.concatenate(buffer)
            threading.Thread(
                target=lambda a=partial: on_realtime(transcribe(a, realtime=True)),
                daemon=True,
            ).start()

        if silent_count >= silence_chunks:
            break

    return np.concatenate(buffer)


def main():
    print("🎙️ Listening... (Ctrl+C to stop)\n")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    ):
        while True:
            audio = record_utterance()
            text = transcribe(audio)
            if text.strip():
                on_final(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")