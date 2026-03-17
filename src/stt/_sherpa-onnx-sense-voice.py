import sys
import queue
import numpy as np
import sounddevice as sd
import sherpa_onnx

# ---------------------------------------------------------------------------
# Download model:
# curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
# tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
# ---------------------------------------------------------------------------
MODEL_DIR = "./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

SAMPLE_RATE = 16000
BLOCK_SIZE = int(0.1 * SAMPLE_RATE)   # 100ms chunks
SILENCE_THRESHOLD = 0.001              # RMS below this = silence
SILENCE_CHUNKS_NEEDED = 8             # ~0.8s of silence → finalize
MIN_SPEECH_CHUNKS = 3                 # ignore very short blips

audio_queue: queue.Queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(f"[audio] {status}", file=sys.stderr)
    audio_queue.put(np.frombuffer(indata, dtype=np.float32).copy())


def create_recognizer() -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{MODEL_DIR}/model.int8.onnx",
        tokens=f"{MODEL_DIR}/tokens.txt",
        num_threads=2,
        use_itn=True,        # inverse text normalization (e.g. "3" not "three")
        debug=False,
        language="en",       # or "zh", "ja", "ko", "yue", or "auto"
    )


def transcribe():
    print("Loading model...")
    recognizer = create_recognizer()
    print("🎙️ Listening... (Ctrl+C to stop)\n")

    speech_buffer: list[np.ndarray] = []
    silence_count = 0
    is_speaking = False

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            chunk = audio_queue.get()
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_silent = rms < SILENCE_THRESHOLD

            if not is_silent:
                speech_buffer.append(chunk)
                silence_count = 0
                if not is_speaking and len(speech_buffer) >= MIN_SPEECH_CHUNKS:
                    is_speaking = True
                    print("\r🔴 Listening...    ", end="", flush=True)
            elif is_speaking:
                speech_buffer.append(chunk)  # include trailing silence
                silence_count += 1

                if silence_count >= SILENCE_CHUNKS_NEEDED:
                    # --- Finalize utterance ---
                    audio = np.concatenate(speech_buffer)
                    stream = recognizer.create_stream()
                    stream.accept_waveform(SAMPLE_RATE, audio)
                    recognizer.decode_stream(stream)
                    text = stream.result.text.strip()
                    if text:
                        print(f"\r✅ {text}          ", flush=True)
                    # reset
                    speech_buffer = []
                    silence_count = 0
                    is_speaking = False
            else:
                # silence before any speech — discard
                speech_buffer = []


if __name__ == "__main__":
    try:
        transcribe()
    except KeyboardInterrupt:
        print("\n\nStopped.")