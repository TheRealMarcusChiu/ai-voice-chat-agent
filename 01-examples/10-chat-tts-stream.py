import asyncio
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel 
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import messages_to_dict
import json
import os
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from uuid import uuid4
from rich.console import Console
from rich.live import Live
from rich.text import Text
from typing import AsyncIterator
from kokoro import KPipeline
import sounddevice as sd


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")


@dataclass
class MyContext:
    """Custom runtime context schema."""
    user_id: str

class MyResponseFormat(BaseModel):
    agent_response: str


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[MyContext]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


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

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

async def speak_stream(text_stream: AsyncIterator[str]) -> None:
    """
    Consume an async text stream, synthesize TTS incrementally,
    and play audio chunks as they're ready.
    """
    buffer = ""
    sentence_endings = {'.', '!', '?', '\n'}

    async def synthesize_and_play(chunk_text: str) -> None:
        chunk_text = chunk_text.strip()
        if not chunk_text:
            return
        generator = pipeline(chunk_text, voice='af_bella', speed=1.3)
        for i, (gs, ps, audio) in enumerate(generator):
            sd.play(audio, samplerate=24000)
            sd.wait()

    async for token in text_stream:
        # yield token

        buffer += token

        # Flush buffer whenever we hit a sentence boundary
        while True:
            flush_idx = -1
            for idx, ch in enumerate(buffer):
                if ch in sentence_endings:
                    flush_idx = idx
                    break
            if flush_idx == -1:
                break

            sentence = buffer[: flush_idx + 1]
            buffer = buffer[flush_idx + 1 :]
            await synthesize_and_play(sentence)

    # Flush any remaining text
    if buffer.strip():
        await synthesize_and_play(buffer)


async def main():
    console = Console()

    while True:
        user_prompt = console.input("[bold cyan]You: [/]")
        
        tts_queue = asyncio.Queue()

        async def tts_task() -> None:
            async def queue_gen() -> AsyncIterator[str]:
                while True:
                    chunk = await tts_queue.get()
                    if chunk is None:
                        return
                    yield chunk
            await speak_stream(queue_gen())

        async def tee_stream() -> AsyncIterator[str]:
            async for chunk in stream_agent(user_prompt):
                await tts_queue.put(chunk)
                yield chunk
            await tts_queue.put(None)

        tts = asyncio.create_task(tts_task())
        console.print("AI: ", style="bold red", end="")
        async for chunk in tee_stream():
            console.print(chunk, end="", highlight=False)

        await tts
        console.print()

if __name__ == "__main__":
    asyncio.run(main())
