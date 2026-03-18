import asyncio
import json
import numpy as np
import websockets
from faster_whisper import WhisperModel


model = WhisperModel("tiny", device="cpu", compute_type="int8")

SAMPLE_RATE = 16000       # must match client
CHUNK_SECONDS = 2         # transcribe every N seconds of audio
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


async def handle_client(websocket):
    print(f"[+] Connected: {websocket.remote_address}")
    audio_buffer = np.array([], dtype=np.float32)

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            # Each message is a chunk of raw 32-bit float PCM samples
            chunk = np.frombuffer(message, dtype=np.float32)
            audio_buffer = np.concatenate([audio_buffer, chunk])

            # Transcribe whenever we have enough audio
            if len(audio_buffer) >= CHUNK_SAMPLES:
                segment_audio = audio_buffer[:CHUNK_SAMPLES]
                audio_buffer = audio_buffer[CHUNK_SAMPLES:]  # keep remainder

                # Run transcription (runs in thread to avoid blocking event loop)
                loop = asyncio.get_event_loop()
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: model.transcribe(segment_audio, language="en", vad_filter=True)
                )

                # Stream each segment's text back as it arrives
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        await websocket.send(json.dumps({"type": "transcript", "text": text}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Transcribe any remaining buffered audio on disconnect
        if len(audio_buffer) > SAMPLE_RATE * 0.5:  # ignore < 0.5s of audio
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: model.transcribe(audio_buffer, language="en", vad_filter=True)
            )
            for segment in segments:
                text = segment.text.strip()
                if text:
                    try:
                        await websocket.send(json.dumps({"type": "transcript", "text": text}))
                    except Exception:
                        pass

        print(f"[-] Disconnected: {websocket.remote_address}")


async def main():
    print("Transcription Server — ws://localhost:8765")
    print("Open index.html in your browser")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())