import sys
import queue
import sherpa_onnx
import sounddevice as sd

# ---------------------------------------------------------------------------
# Model paths — download one of these streaming models and point here:
#
# English only (~70 MB int8, fast):
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile.tar.bz2
#
# English only (~300 MB, more accurate):
#   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
# ---------------------------------------------------------------------------
MODEL_DIR = "./models/sherpa-onnx-streaming-zipformer-en-2023-06-26"

SAMPLE_RATE = 16000
SAMPLES_PER_READ = int(0.1 * SAMPLE_RATE)  # 100ms chunks

audio_queue: queue.Queue[bytes] = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(f"[audio] {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))


def create_recognizer() -> sherpa_onnx.OnlineRecognizer:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{MODEL_DIR}/tokens.txt",
        encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        # encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        # decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        # joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,  # pause → final result
        rule2_min_trailing_silence=1.2,  # shorter pause mid-sentence
        rule3_min_utterance_length=300,  # force finalize long utterances
    )


def transcribe():
    print("Loading model...")
    recognizer = create_recognizer()
    stream = recognizer.create_stream()

    last_partial = ""
    print("🎙️ Listening... (Ctrl+C to stop)\n")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLES_PER_READ,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()

            import numpy as np
            samples = np.frombuffer(data, dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, samples)

            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            partial = recognizer.get_result(stream).strip()

            if recognizer.is_endpoint(stream):
                # --- Final result ---
                if partial:
                    print(f"\n✅ {partial}", flush=True)
                recognizer.reset(stream)
                last_partial = ""
            elif partial != last_partial:
                # --- Realtime partial ---
                print(f"\r🔴 {partial}    ", end="", flush=True)
                last_partial = partial


if __name__ == "__main__":
    try:
        transcribe()
    except KeyboardInterrupt:
        print("\n\nStopped.")