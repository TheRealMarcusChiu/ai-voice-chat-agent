from RealtimeSTT import AudioToTextRecorder

def on_realtime(text):
    print(f"\r🔴 {text}    ", end="", flush=True)

def on_final(text):
    print(f"\n✅ {text}", flush=True)

if __name__ == '__main__':
    # Model         | Size      | Relative Speed
    # --------------|-----------|---------------
    # tiny          | ~75 MB    | ~32x
    # base          | ~145 MB   | ~16x
    # small         | ~466 MB   | ~6x
    # medium        | ~1.5 GB   | ~2x
    # large-v3-turbo| ~1.6 GB   | ~8x
    # large-v3      | ~3 GB     | 1x (baseline)
    recorder = AudioToTextRecorder(
        model="tiny",
        silero_sensitivity=0.4,
        enable_realtime_transcription=True,
        realtime_processing_pause=0.05,
        on_realtime_transcription_update=on_realtime,
        spinner=False)

    print("🎙️ Listening... (Ctrl+C to stop)\n")
    while True:
        recorder.text(on_final)