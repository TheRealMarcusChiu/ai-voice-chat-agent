import asyncio
import json
import websockets
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from typing import AsyncIterator
from tools import get_user_location, get_weather_for_location
from config import MyContext, MyResponseFormat


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
ENABLE_VOICE = os.getenv("ENABLE_VOICE", "false").lower() == "true"


client = None
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


async def stream_agent(user_prompt: str) -> AsyncIterator[str]:
    stream = agent.astream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages")

    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            yield message.text

async def stream_echo(websocket, text):
    msg_id = str(uuid.uuid4())
    await websocket.send(json.dumps({"type": "stream_start", "id": msg_id}))
    async for chunk in stream_agent(text):
        await websocket.send(json.dumps({"type": "stream_chunk", "id": msg_id, "chunk": chunk}))
    await websocket.send(json.dumps({"type": "stream_end", "id": msg_id}))


async def handle_client(websocket):
    global client
    client = websocket
    print(f"[+] Connected: {websocket.remote_address}")

    try:
        async for raw in websocket:
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "stream_chunk":
                print(msg.get("chunk", ""), end="", flush=True)
            elif mtype == "stream_end":
                print()
                text = msg.get("text", "")
                await stream_echo(websocket, text)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        client = None
        print(f"[-] Disconnected: {websocket.remote_address}")


async def main():
    print("WebSocket Streaming Server — ws://localhost:8765")
    print("Open index.html in your browser")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())