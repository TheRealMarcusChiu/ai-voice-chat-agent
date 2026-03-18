import asyncio
import json
import numpy as np
import websockets
from stt import AudioToTextRecorder2

def on_realtime(text: str):
    print(f"realtime: {text}")

async def handle_client(websocket):
    print("Client connected")
    stt = AudioToTextRecorder2(
        on_realtime_transcription_update=on_realtime,
        source="websocket")

    async def feed_audio():
        async for message in websocket:
            if isinstance(message, bytes):
                chunk = np.frombuffer(message, dtype=np.float32)
                stt.feed(chunk)

    async def transcribe_loop():
        while True:
            await stt.text(
                lambda result, ws=websocket: asyncio.ensure_future(
                    ws.send(json.dumps({"type": "transcript", "text": result}))
                )
            )

    await asyncio.gather(feed_audio(), transcribe_loop())

async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

asyncio.run(main())