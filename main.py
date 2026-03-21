import asyncio
from concurrent.futures import ThreadPoolExecutor
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
from stt.mystt import Transcript
from tools import get_user_location, get_weather_for_location
from config import MyContext, MyResponseFormat
from tts import MyTTS
from stt import MySTT
from util import queue_to_async_iter, fan_out_stream
import urllib.parse
from aiohttp import web
from typing import AsyncGenerator


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")
DEVICE = os.getenv("DEVICE", "cpu")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


agent = create_agent(
    model=ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    ),
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=MyContext,
    response_format=ToolStrategy(MyResponseFormat),
    checkpointer=InMemorySaver(),
)


class ClientSession:
    """Holds all mutable state that belongs to a single connected client."""

    def __init__(self, websocket, user_id: str):
        self.ws = websocket
        self.user_id = user_id
        self.stt = MySTT(
            on_user_transcript_start=self.on_user_transcript_start,
            on_user_transcript_unfinished=self.on_user_transcript_unfinished,
            on_user_transcript_end=self.on_user_transcript_end,
            device=DEVICE,
        )
        self.tts = MyTTS(
            on_ai_audio_response_chunk=self.on_ai_audio_response_chunk,
            device=DEVICE,
        )
        self.loop = asyncio.get_event_loop()

    def run(self):
        audio_source = self._get_audio_source()
        self.stt.transcribe_audio(audio_source)

    # ------------------------------------------------------------------
    # Other methods related to STT, TTS, and agent invocation
    # ------------------------------------------------------------------

    async def _get_audio_source(self) -> AsyncGenerator[np.ndarray, None]:
        async for message in self.ws:
            if isinstance(message, bytes):
                yield np.frombuffer(message, dtype=np.float32)

    def _async_send(self, data):
        asyncio.run_coroutine_threadsafe(
            self.ws.send(json.dumps(data)),
            self.loop
        )


    def on_user_transcript_start(self, t: Transcript):
        print(f"[{self.user_id}][{t.id[:8]}] transcript started")
        self._async_send({
                "type": "on-transcript-start",
                "utterance_id": t.id,
            })

    def on_user_transcript_unfinished(self, t: Transcript):
        print(f"[{self.user_id}][{t.id[:8]}] transcript unfinished: {t.transcript}")
        self._async_send({
            "type": "on-transcript-unfinished",
            "utterance_id": t.id,
            "text": t.transcript,
        })

    def on_user_transcript_end(self, t: Transcript):
        print(f"[{self.user_id}][{t.id[:8]}] transcript ended:      {t.transcript}")
        self._async_send({
            "type": "on-transcript-end",
            "utterance_id": t.id,
            "text": t.transcript,
        })
        self._prompt_ai(t.transcript)


    def _prompt_ai(self, user_transcript: str):
        print(f"[{self.user_id}] Invoking AI with prompt: {user_transcript}")
        msg_id = str(uuid.uuid4())

        async def _invoke():
            ai_response_stream = self.get_ai_response_stream(user_transcript)
            text_queue: asyncio.Queue = asyncio.Queue()
            tts_queue: asyncio.Queue = asyncio.Queue()
            await asyncio.gather(
                fan_out_stream(ai_response_stream, [text_queue, tts_queue]),
                self.on_ai_text_response(msg_id, text_queue),
                self.on_ai_audio_response(msg_id, tts_queue),
            )

        asyncio.run_coroutine_threadsafe(_invoke(), self.loop)


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

    async def on_ai_audio_response_chunk(self, audio_bytes: bytes, samplerate: int, msg_id: str):
        await self.ws.send(json.dumps({
            "type": "on-ai-audio-chunk",
            "id": msg_id,
            "audio": np.frombuffer(audio_bytes, dtype=np.float32).tolist(),
            "samplerate": samplerate,
        }))


# ------------------------------------------------------------------
# WebSocket entry point — one ClientSession per connection
# ------------------------------------------------------------------

async def handle_client(websocket):
    query = websocket.request.path
    params = urllib.parse.parse_qs(urllib.parse.urlparse(query).query)
    user_id = params.get("user_id", [None])[0] or str(uuid.uuid4())
    session = ClientSession(websocket, user_id)
    print(f"Client connected: user_id={user_id}")
    session.run()
    await websocket.wait_closed() # keeps handle_client alive indefinitely


# ------------------------------------------------------------------
# HTTP server — serves assets/index.html at port 8080
# ------------------------------------------------------------------

async def serve_http():
    async def index(request):
        raise web.HTTPFound("/index.html")

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_static("/", path="assets", name="static")

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    print("\n\nServing assets/index.html at http://localhost:8080")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

async def main():
    await serve_http()
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
