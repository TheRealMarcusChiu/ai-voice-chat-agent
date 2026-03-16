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


async def main():
    console = Console()

    while True:
        user_prompt = console.input("[bold cyan]You: [/]")
        
        accumulated = ""
        with Live(console=console, refresh_per_second=20) as live:
            async for chunk in stream_agent(user_prompt):
                accumulated += chunk
                text = Text()
                text.append("AI: ", style="bold red")
                text.append(accumulated)
                live.update(text)

asyncio.run(main())
