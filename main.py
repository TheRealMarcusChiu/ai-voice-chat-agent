import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import messages_to_dict
import os
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from typing import AsyncIterator

from tools import get_user_location, get_weather_for_location
from tts import speak_stream
from stt import AudioToTextRecorder
from config import MyContext, MyResponseFormat
from functools import partial


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


async def stream_agent(user_prompt: str) -> AsyncIterator[str]:
    stream = agent.astream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages")

    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            yield message.text


def on_realtime(text, console):
    console.print(f"You: ", style="bold yellow", end="")
    console.print(f"{text}", highlight=False, end="\r")

async def on_final(text, console):
    console.print(f"You: ", style="bold cyan", end="")
    console.print(f"{text}", highlight=False)

    user_prompt = text
    console.print("AI: ", style="bold red", end="")
    async for chunk in stream_agent(user_prompt):
        console.print(chunk, end="", highlight=False)
    console.print("\n")

    on_realtime("", console)


# async def on_final(text, console):
#     console.print(f"You: ", style="bold cyan", end="")
#     console.print(f"{text}", highlight=False)

#     user_prompt = text
#     tts_queue = asyncio.Queue()

#     async def tts_task() -> None:
#         async def queue_gen() -> AsyncIterator[str]:
#             while True:
#                 chunk = await tts_queue.get()
#                 if chunk is None:
#                     return
#                 yield chunk
#         await speak_stream(queue_gen())

#     async def tee_stream() -> AsyncIterator[str]:
#         async for chunk in stream_agent(user_prompt):
#             await tts_queue.put(chunk)
#             yield chunk
#         await tts_queue.put(None)

#     tts = asyncio.create_task(tts_task())
#     console.print("AI: ", style="bold red", end="")
#     async for chunk in tee_stream():
#         console.print(chunk, end="", highlight=False)

#     await tts
#     console.print()
#     on_realtime("", console)


async def main():
    console = Console()

    if ENABLE_VOICE:
        print("🎙️ Voice mode enabled. Say something! (Ctrl+C to stop)\n")
        with AudioToTextRecorder(
            on_realtime_transcription_update=partial(on_realtime, console=console),
        ) as recorder:
            on_realtime("", console)
            while True:
                await recorder.text(on_final, console)
    else:
        while True:
            user_prompt = console.input("[bold cyan]You: [/]")

            console.print("AI: ", style="bold red", end="")
            async for chunk in stream_agent(user_prompt):
                console.print(chunk, end="", highlight=False)
            console.print()


if __name__ == "__main__":
    asyncio.run(main())
