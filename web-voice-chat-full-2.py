import asyncio
import json
import numpy as np
import websockets
from stt import AudioToTextRecorder2
from functools import partial
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from tools import get_user_location, get_weather_for_location
from config import MyContext, MyResponseFormat
from tts import TextToSpeechStreamer


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")


agent = create_agent(
    model=ChatOllama(
        model="llama3.2",
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    ),
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=MyContext,
    response_format=ToolStrategy(MyResponseFormat),
    checkpointer=InMemorySaver(),
)


async def on_user_transcript_unfinished(user_transcript: str, utterance_id: str, websocket):
    print(f"[{utterance_id[:8]}] user transcript unfinished: {user_transcript}")
    await websocket.send(json.dumps({
        "type": "on-transcript-unfinished",
        "utterance_id": utterance_id,
        "text": user_transcript,
    }))

async def on_user_transcript_finished(user_transcript: str, utterance_id: str, websocket):
    print(f"[{utterance_id[:8]}] user transcript finished:   {user_transcript}")
    await asyncio.gather(
        websocket.send(json.dumps({
            "type": "on-transcript-finished",
            "utterance_id": utterance_id,
            "text": user_transcript,
        })),
        invoke_ai(websocket, user_transcript),
    )


async def get_ai_response_stream(user_transcript: str):
    stream = agent.astream(
        {"messages": [{"role": "user", "content": user_transcript}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages",
    )
    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            yield message.text


async def _fan_out_stream(source_stream, queues: list[asyncio.Queue]):
    """
    Drain *source_stream* and push every item into every queue in *queues*.
    Sends `None` as a sentinel to signal end-of-stream.
    """
    try:
        async for item in source_stream:
            for q in queues:
                await q.put(item)
    finally:
        for q in queues:
            await q.put(None)  # sentinel


async def _queue_to_async_iter(queue: asyncio.Queue):
    """Yield items from *queue* until the `None` sentinel is received."""
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def invoke_ai(websocket, user_transcript: str):
    msg_id = str(uuid.uuid4())

    # Two queues — one per consumer of the LLM token stream
    text_queue: asyncio.Queue = asyncio.Queue()
    tts_queue: asyncio.Queue = asyncio.Queue()

    # --- Consumer 1: forward text chunks to the WebSocket client ---
    async def stream_text_to_websocket():
        await websocket.send(json.dumps({"type": "on-ai-text-response-start", "id": msg_id}))
        async for chunk in _queue_to_async_iter(text_queue):
            await websocket.send(json.dumps({"type": "on-ai-text-response-chunk", "id": msg_id, "chunk": chunk}))
        await websocket.send(json.dumps({"type": "on-ai-text-response-end", "id": msg_id}))

    # --- Consumer 2: TTS → audio chunks → WebSocket client ---
    async def on_audio_chunk(audio_bytes: bytes, _msg_id: str):
        """Called by TextToSpeechStreamer for every synthesized audio segment."""
        await websocket.send(json.dumps({
            "type": "on-ai-audio-chunk",
            "id": _msg_id,
            # Encode as a list of floats so it's JSON-serialisable.
            # Switch to a binary WebSocket frame / base64 if you prefer.
            "audio": np.frombuffer(audio_bytes, dtype=np.float32).tolist(),
            "samplerate": 24000,
        }))

    tts = TextToSpeechStreamer(on_audio_chunk=on_audio_chunk)

    async def stream_tts():
        await websocket.send(json.dumps({"type": "on-ai-audio-start", "id": msg_id}))
        await tts.speak_stream(_queue_to_async_iter(tts_queue), msg_id)
        await websocket.send(json.dumps({"type": "on-ai-audio-end", "id": msg_id}))

    # Run: fan-out producer + both consumers concurrently
    await asyncio.gather(
        _fan_out_stream(get_ai_response_stream(user_transcript), [text_queue, tts_queue]),
        stream_text_to_websocket(),
        stream_tts(),
    )


async def handle_client(websocket):
    print("Client connected")

    stt = AudioToTextRecorder2(
        on_realtime_transcription_update=partial(on_user_transcript_unfinished, websocket=websocket),
        loop=asyncio.get_running_loop(),
    )

    async def consume_user_audio():
        async for message in websocket:
            if isinstance(message, bytes):
                chunk = np.frombuffer(message, dtype=np.float32)
                stt.feed(chunk)

    async def produce_user_transcript():
        while True:
            on_final = lambda result, uid, ws=websocket: asyncio.ensure_future(
                on_user_transcript_finished(result, uid, ws)
            )
            await stt.text(on_final=on_final)

    await asyncio.gather(consume_user_audio(), produce_user_transcript())


async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())