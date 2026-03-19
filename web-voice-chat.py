import asyncio
import json
import numpy as np
import websockets
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
from tts import MyTTS
from stt import MySTT
from util import queue_to_async_iter, fan_out_stream
import urllib.parse

async def on_user_transcript_unfinished(user_transcript: str, utterance_id: str):
    print(f"[{utterance_id[:8]}] user transcript unfinished: {user_transcript}")
    await ws.send(json.dumps({
        "type": "on-transcript-unfinished",
        "utterance_id": utterance_id,
        "text": user_transcript,
    }))

async def on_user_transcript_finished(user_transcript: str, utterance_id: str):
    print(f"[{utterance_id[:8]}] user transcript finished:   {user_transcript}")
    await asyncio.gather(
        ws.send(json.dumps({
            "type": "on-transcript-finished",
            "utterance_id": utterance_id,
            "text": user_transcript,
        })),
        invoke_ai(user_transcript),
    )

async def get_ai_response_stream(user_transcript: str):
    stream = agent.astream(
        {"messages": [{"role": "user", "content": user_transcript}]},
        config={"configurable": {"thread_id": user_id}},
        context=MyContext(user_id=user_id),
        stream_mode="messages",
    )
    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            yield message.text

async def on_ai_text_response(msg_id: str, text_queue: asyncio.Queue):
    await ws.send(json.dumps({"type": "on-ai-text-response-start", "id": msg_id}))
    async for chunk in queue_to_async_iter(text_queue):
        await ws.send(json.dumps({"type": "on-ai-text-response-chunk", "id": msg_id, "chunk": chunk}))
    await ws.send(json.dumps({"type": "on-ai-text-response-end", "id": msg_id}))

async def on_ai_audio_repsonse(msg_id: str, tts_queue: asyncio.Queue):
    await ws.send(json.dumps({"type": "on-ai-audio-start", "id": msg_id}))
    await tts.speak_stream(queue_to_async_iter(tts_queue), msg_id)
    await ws.send(json.dumps({"type": "on-ai-audio-end", "id": msg_id}))

async def on_ai_audio_response_chunk(audio_bytes: bytes, _msg_id: str):
    """Called by TextToSpeechStreamer for every synthesized audio segment."""
    await ws.send(json.dumps({
        "type": "on-ai-audio-chunk",
        "id": _msg_id,
        # Encode as a list of floats so it's JSON-serialisable.
        # Switch to a binary WebSocket frame / base64 if you prefer.
        "audio": np.frombuffer(audio_bytes, dtype=np.float32).tolist(),
        "samplerate": 24000,
    }))

async def invoke_ai(user_transcript: str):
    print("Invoking AI with prompt: ", user_transcript)
    msg_id = str(uuid.uuid4())

    # Two queues — one per consumer of the LLM token stream
    text_queue: asyncio.Queue = asyncio.Queue()
    tts_queue: asyncio.Queue = asyncio.Queue()

    # Run: fan-out producer + both consumers concurrently
    await asyncio.gather(
        fan_out_stream(get_ai_response_stream(user_transcript), [text_queue, tts_queue]),
        on_ai_text_response(msg_id, text_queue),
        on_ai_audio_repsonse(msg_id, tts_queue),
    )

async def handle_client(websocket):
    global user_id
    global ws
    global stt
    global tts

    query = websocket.request.path  # e.g. "/?user_id=alice"
    params = urllib.parse.parse_qs(urllib.parse.urlparse(query).query)
    user_id = params.get("user_id", [None])[0]
    
    print(f"Client connected: user_id={user_id}")

    ws = websocket
    stt = MySTT(on_user_transcript_unfinished=on_user_transcript_unfinished, device=DEVICE)
    tts = MyTTS(on_ai_audio_response_chunk=on_ai_audio_response_chunk, device=DEVICE)

    async def consume_user_audio():
        async for message in websocket:
            if isinstance(message, bytes):
                chunk = np.frombuffer(message, dtype=np.float32)
                stt.feed(chunk)

    async def produce_user_transcript():
        while True:
            on_final = lambda result, uid: asyncio.ensure_future(
                on_user_transcript_finished(result, uid)
            )
            await stt.text(on_final=on_final)

    await asyncio.gather(consume_user_audio(), produce_user_transcript())


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
DEVICE = os.getenv("DEVICE", "cpu")

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

user_id = None
ws = None
stt = None
tts = None

async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())