from kokoro import KPipeline
import sounddevice as sd
import soundfile as sf
import numpy as np

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

text = "Using uv is a great choice. It's much faster and handles deep dependencies better than standard tools."

generator = pipeline(text, voice='af_bella', speed=1.1)

# Collect all chunks and play sequentially
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Playing chunk {i}...")
    sd.play(audio, samplerate=24000)
    sd.wait()  # Block until chunk finishes playing