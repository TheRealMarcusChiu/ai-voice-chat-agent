import asyncio
import json
import numpy as np
import websockets
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain.messages import AIMessage
from tools import get_user_location, get_weather_for_location
from config import MyContext, MyResponseFormat
from tts import MyTTS
from stt import MySTT
from util import queue_to_async_iter, fan_out_stream
import urllib.parse


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
sessions = set()


class ClientSession:
    """Holds all mutable state that belongs to a single connected client."""

    def __init__(self, websocket, user_id: str):
        self.ws = websocket
        self.user_id = user_id
        self.stt = MySTT(
            on_user_transcript_unfinished=self.on_user_transcript_unfinished,
            device=DEVICE,
        )
        self.tts = MyTTS(
            on_ai_audio_response_chunk=self.on_ai_audio_response_chunk,
            device=DEVICE,
        )

    # ------------------------------------------------------------------
    # STT callbacks
    # ------------------------------------------------------------------

    async def on_user_transcript_unfinished(self, user_transcript: str, utterance_id: str):
        print(f"[{self.user_id}][{utterance_id[:8]}] transcript unfinished: {user_transcript}")
        await self.ws.send(json.dumps({
            "type": "on-transcript-unfinished",
            "utterance_id": utterance_id,
            "text": user_transcript,
        }))

    async def on_user_transcript_finished(self, user_transcript: str, utterance_id: str):
        print(f"[{self.user_id}][{utterance_id[:8]}] transcript finished:   {user_transcript}")
        await asyncio.gather(
            self.ws.send(json.dumps({
                "type": "on-transcript-finished",
                "utterance_id": utterance_id,
                "text": user_transcript,
            })),
            self.invoke_ai(user_transcript),
        )

    # ------------------------------------------------------------------
    # AI pipeline
    # ------------------------------------------------------------------

    async def get_ai_response_stream(self, user_transcript: str):
        stream = agent.astream(
            {"messages": [{"role": "user", "content": user_transcript}]},
            config={"configurable": {"thread_id": self.user_id}},
            context=MyContext(user_id=self.user_id),
            stream_mode="messages",
        )
        async for message, _metadata in stream:
            if isinstance(message, AIMessage):
                yield message.text

    async def on_ai_text_response(self, msg_id: str, text_queue: asyncio.Queue):
        await self.ws.send(json.dumps({"type": "on-ai-text-response-start", "id": msg_id}))
        async for chunk in queue_to_async_iter(text_queue):
            await self.ws.send(json.dumps({
                "type": "on-ai-text-response-chunk",
                "id": msg_id,
                "chunk": chunk,
            }))
        await self.ws.send(json.dumps({"type": "on-ai-text-response-end", "id": msg_id}))

    async def on_ai_audio_response(self, msg_id: str, tts_queue: asyncio.Queue):
        await self.ws.send(json.dumps({"type": "on-ai-audio-start", "id": msg_id}))
        await self.tts.speak_stream(queue_to_async_iter(tts_queue), msg_id)
        await self.ws.send(json.dumps({"type": "on-ai-audio-end", "id": msg_id}))

    async def on_ai_audio_response_chunk(self, audio_bytes: bytes, msg_id: str):
        """Called by TextToSpeechStreamer for every synthesized audio segment."""
        await self.ws.send(json.dumps({
            "type": "on-ai-audio-chunk",
            "id": msg_id,
            "audio": np.frombuffer(audio_bytes, dtype=np.float32).tolist(),
            "samplerate": 24000,
        }))

    async def invoke_ai(self, user_transcript: str):
        print(f"[{self.user_id}] Invoking AI with prompt: {user_transcript}")
        msg_id = str(uuid.uuid4())

        text_queue: asyncio.Queue = asyncio.Queue()
        tts_queue: asyncio.Queue = asyncio.Queue()

        await asyncio.gather(
            fan_out_stream(self.get_ai_response_stream(user_transcript), [text_queue, tts_queue]),
            self.on_ai_text_response(msg_id, text_queue),
            self.on_ai_audio_response(msg_id, tts_queue),
        )

    # ------------------------------------------------------------------
    # Main read loop
    # ------------------------------------------------------------------

    async def run(self):
        async def consume_user_audio():
            async for message in self.ws:
                if isinstance(message, bytes):
                    chunk = np.frombuffer(message, dtype=np.float32)
                    self.stt.feed(chunk)

        async def produce_user_transcript():
            while True:
                def on_final(result, uid):
                    asyncio.ensure_future(
                        self.on_user_transcript_finished(result, uid)
                    )
                await self.stt.text(on_final=on_final)

        await asyncio.gather(consume_user_audio(), produce_user_transcript())

    async def close(self):
        await self.tts.close()   # if your TTS has cleanup
        await self.stt.close()   # if your STT has cleanup


# ------------------------------------------------------------------
# WebSocket entry point — one ClientSession per connection
# ------------------------------------------------------------------

async def handle_client(websocket):
    query = websocket.request.path
    params = urllib.parse.parse_qs(urllib.parse.urlparse(query).query)
    user_id = params.get("user_id", [None])[0] or str(uuid.uuid4())

    session = ClientSession(websocket, user_id)
    sessions.add(session)

    print(f"Client connected: user_id={user_id}")
    print(f"Active sessions: {len(sessions)}")

    try:
        await session.run()
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: user_id={user_id}")
    finally:
        await session.close()
        sessions.remove(session)
        print(f"Client disconnected: user_id={user_id}")
        print(f"Active sessions: {len(sessions)}")


async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())