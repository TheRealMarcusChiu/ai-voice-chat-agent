import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# Config
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000        # ~250ms chunks
SILENCE_THRESHOLD = 0.001
BUFFER_SECONDS = 0.5       # transcribe every N seconds of audio

model = WhisperModel("tiny", device="cpu", compute_type="int8")

audio_queue = queue.Queue()
audio_buffer = []

def audio_callback(indata, frames, time, status):
    """Called by sounddevice for each audio chunk."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def transcribe_loop():
    global audio_buffer
    while True:
        # Drain the queue into buffer
        while not audio_queue.empty():
            chunk = audio_queue.get()
            audio_buffer.append(chunk)

        # Transcribe every BUFFER_SECONDS worth of audio
        if len(audio_buffer) >= (SAMPLE_RATE * BUFFER_SECONDS) / BLOCK_SIZE:
            audio_np = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
            audio_buffer = []

            # Skip if too quiet (silence)
            if np.abs(audio_np).mean() < SILENCE_THRESHOLD:
                continue

            segments, _ = model.transcribe(
                audio_np,
                vad_filter=True,
                language="en"
            )
            for segment in segments:
                print(segment.text.strip(), flush=True)

# Run transcription in background thread
t = threading.Thread(target=transcribe_loop, daemon=True)
t.start()

# Start mic stream
print("🎙️ Listening... (Ctrl+C to stop)\n")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
    threading.Event().wait()  # block forever
