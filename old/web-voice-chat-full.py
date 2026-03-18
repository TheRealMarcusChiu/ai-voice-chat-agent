import asyncio
import json
import numpy as np
import websockets
import base64
from stt import AudioToTextRecorder2
from functools import partial
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
from kokoro import KPipeline

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT")

tts_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

agent = create_agent(
    model=ChatOllama(model="llama3.2", temperature=0, base_url=OLLAMA_BASE_URL),
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=MyContext,
    response_format=ToolStrategy(MyResponseFormat),
    checkpointer=InMemorySaver())


# ── Helpers ───────────────────────────────────────────────────────────────────

def float32_to_pcm16_b64(audio) -> str:
    # Kokoro returns a torch.Tensor — convert to numpy if needed
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = np.asarray(audio, dtype=np.float32)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


async def synthesize_and_send(text: str, msg_id: str, websocket) -> None:
    """Synthesize *text* with Kokoro and send each audio chunk over the socket."""
    text = text.strip()
    if not text:
        return

    loop = asyncio.get_running_loop()

    def _synth():
        return [audio for _gs, _ps, audio in tts_pipeline(text, voice='af_bella', speed=1.3)]

    audio_chunks = await loop.run_in_executor(None, _synth)

    for audio in audio_chunks:
        await websocket.send(json.dumps({
            "type": "audio_chunk",
            "id": msg_id,
            "sampleRate": 24000,
            "data": float32_to_pcm16_b64(audio),
        }))


# ── STT callbacks ─────────────────────────────────────────────────────────────

async def on_user_transcript_unfinished(user_transcript: str, utterance_id: str, websocket):
    await websocket.send(json.dumps({
        "type": "on-transcript-unfinished",
        "utterance_id": utterance_id,
        "text": user_transcript,
    }))


async def on_user_transcript_finished(user_transcript: str, utterance_id: str, websocket):
    await asyncio.gather(
        websocket.send(json.dumps({
            "type": "on-transcript-finished",
            "utterance_id": utterance_id,
            "text": user_transcript,
        })),
        stream_ai_response(websocket, user_transcript)
    )


# ── AI response streaming with sequential TTS ─────────────────────────────────

async def stream_ai_response(websocket, prompt: str):
    msg_id = str(uuid.uuid4())
    await websocket.send(json.dumps({"type": "ai_stream_start", "id": msg_id}))

    stream = agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages")

    sentence_endings = {'.', '!', '?', '\n'}
    text_buffer = ""
    tts_tasks   = []   # collect coroutines, await them all before stream_end

    async for message, metadata in stream:
        if not isinstance(message, AIMessage) or not message.text:
            continue

        chunk = message.text
        await websocket.send(json.dumps({"type": "ai_stream_chunk", "id": msg_id, "chunk": chunk}))

        text_buffer += chunk

        # Flush complete sentences to TTS
        while True:
            flush_idx = next(
                (i for i, ch in enumerate(text_buffer) if ch in sentence_endings), -1)
            if flush_idx == -1:
                break
            sentence    = text_buffer[:flush_idx + 1]
            text_buffer = text_buffer[flush_idx + 1:]
            # Schedule TTS concurrently but keep a handle so we can await all
            tts_tasks.append(asyncio.ensure_future(
                synthesize_and_send(sentence, msg_id, websocket)))

    # Flush any trailing text
    if text_buffer.strip():
        tts_tasks.append(asyncio.ensure_future(
            synthesize_and_send(text_buffer, msg_id, websocket)))

    # Wait for ALL TTS audio chunks to be sent before telling the client we're done
    if tts_tasks:
        await asyncio.gather(*tts_tasks)

    await websocket.send(json.dumps({"type": "ai_stream_end", "id": msg_id}))


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_client(websocket):
    print("Client connected")

    stt = AudioToTextRecorder2(
        on_realtime_transcription_update=partial(
            on_user_transcript_unfinished, websocket=websocket),
        loop=asyncio.get_running_loop(),
    )

    async def consume_user_audio():
        async for message in websocket:
            if isinstance(message, bytes):
                stt.feed(np.frombuffer(message, dtype=np.float32))

    async def produce_user_transcript():
        while True:
            on_final = lambda result, uid, ws=websocket: asyncio.ensure_future(
                on_user_transcript_finished(result, uid, ws))
            await stt.text(on_final=on_final)

    await asyncio.gather(consume_user_audio(), produce_user_transcript())


async def main():
    print("Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
