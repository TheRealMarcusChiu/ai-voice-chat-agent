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
        invoke_ai(websocket, user_transcript)
    )

async def get_ai_response_stream(user_transcript: str) -> AsyncIterator[str]:
    stream = agent.astream(
        {"messages": [{"role": "user", "content": user_transcript}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages")

    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            yield message.text

async def invoke_ai(websocket, user_transcript: str):
    msg_id = str(uuid.uuid4())
    ai_response_stream = get_ai_response_stream(user_transcript)

    # simultaneously stream ai response text back to client as it comes in from the agent, and also stream it to TTS engine to stream audio back to client as well

    # 1. Stream text chunks back to client as they come in from the agent
    await websocket.send(json.dumps({"type": "on-ai-text-response-start", "id": msg_id}))
    async for chunk in ai_response_stream:
        await websocket.send(json.dumps({"type": "on-ai-text-response-chunk", "id": msg_id, "chunk": chunk}))        
    await websocket.send(json.dumps({"type": "on-ai-text-response-end", "id": msg_id}))

    # 2. TODO stream ai_response_stream to TTS engine and stream audio back to client as well

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
