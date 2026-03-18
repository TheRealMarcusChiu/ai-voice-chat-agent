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


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
ENABLE_VOICE = os.getenv("ENABLE_VOICE", "false").lower() == "true"


agent = create_agent(
    model = ChatOllama(
        model = "llama3.2",
        temperature = 0,
        base_url = OLLAMA_BASE_URL),
    system_prompt = SYSTEM_PROMPT,
    tools = [get_user_location, get_weather_for_location],
    context_schema = MyContext,
    response_format = ToolStrategy(MyResponseFormat),
    checkpointer = InMemorySaver())


async def stream_user_transcript_unfinished(text: str, utterance_id: str, websocket):
    print(f"[{utterance_id[:8]}] user transcript unfinished: {text}")
    await websocket.send(json.dumps({
        "type": "on-transcript-realtime",
        "utterance_id": utterance_id,
        "text": text,
    }))

async def on_user_transcript_finished(result, uid, websocket):
    print(f"[{uid[:8]}] user transcript finished:   {result}")
    await asyncio.gather(
        websocket.send(json.dumps({
            "type": "on-transcript-full",
            "utterance_id": uid,
            "text": result,
        })),
        stream_ai_response(websocket, result)
    )

async def stream_ai_response(websocket, prompt):
    msg_id = str(uuid.uuid4())
    await websocket.send(json.dumps({"type": "ai_stream_start", "id": msg_id}))

    stream = agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages")

    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            await websocket.send(json.dumps({"type": "ai_stream_chunk", "id": msg_id, "chunk": message.text}))
        
    await websocket.send(json.dumps({"type": "ai_stream_end", "id": msg_id}))

async def handle_client(websocket):
    print("Client connected")
    stt = AudioToTextRecorder2(
        on_realtime_transcription_update=partial(stream_user_transcript_unfinished, websocket=websocket),
        loop=asyncio.get_running_loop(),
    )

    async def feed_audio():
        async for message in websocket:
            if isinstance(message, bytes):
                chunk = np.frombuffer(message, dtype=np.float32)
                stt.feed(chunk)

    async def transcribe_loop():
        while True:
            on_final = lambda result, uid, ws=websocket: asyncio.ensure_future(
                    on_user_transcript_finished(result, uid, ws)
            )
            await stt.text(on_final=on_final)

    await asyncio.gather(feed_audio(), transcribe_loop())


async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
