import asyncio
import json
import numpy as np
import websockets
from stt import AudioToTextRecorder2
from functools import partial


async def on_realtime(text: str, websocket):
    print(f"realtime: {text}")
    await websocket.send(json.dumps({"type": "transcript", "text": text}))

async def handle_client(websocket):
    print("Client connected")
    loop = asyncio.get_running_loop()  # called from async context, always works
    stt = AudioToTextRecorder2(
        on_realtime_transcription_update=partial(on_realtime, websocket=websocket),
        source="websocket",
        loop=loop)

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