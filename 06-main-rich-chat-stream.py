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


async def stream_agent():
    stream = agent.astream(
        {"messages": [{"role": "user", "content": "What's the weather in Plano?"}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages")

    i = 0
    async for message, metadata in stream:
        if isinstance(message, AIMessage):
            print(f"{i} - AgentChunkEvent: message.text={message.text}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    id = tool_call.get("id", str(uuid4()))
                    name = tool_call.get("name", "unknown")
                    args = tool_call.get("args", {})
                    print(f"{i} - ToolCall: id={id}, name={name}, args={args}")

        if isinstance(message, ToolMessage):
            tool_call_id = getattr(message, "tool_call_id", "")
            name = getattr(message, "name", "unknown")
            result = str(message.content) if message.content else ""
            print(f"{i} - ToolMessage: tool_call_id={tool_call_id}, name={name}, result={result}")

        i += 1

async def main():
    task = asyncio.create_task(stream_agent())

    async def wait_for_cancel():
        await asyncio.sleep(0.75)
        task.cancel()

    results = await asyncio.gather(task, wait_for_cancel(), return_exceptions=True)
    
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            print("task was cancelled successfully")
        elif isinstance(result, Exception):
            print(f"task raised an exception: {result}")

asyncio.run(main())

# while True:

#     user_input = input("\nHUMAN: ")
#     response = agent.invoke(
#         {"messages": [{"role": "user", "content": user_input}]},
#         config = {"configurable": {"thread_id": "1"}},
#         context = MyContext(user_id = "1")
#     )
    
#     # print(response["messages"])
#     # print(json.dumps(messages_to_dict(response["messages"]), indent=2))

#     last_n_messages_to_print = 3
#     for i in range(last_n_messages_to_print, 0, -1):
#         message = response["messages"][-i]

#         if message.type == "ai": 
#             print(f"\nAI: {message.content}")
#             if len(message.tool_calls) > 0:
#                 print("AI tool-calls:")
#                 for tool_call in message.tool_calls:
#                     print("-", tool_call)
#         elif message.type == "tool":
#             print(f"\nTOOL: {message.content}")
#         elif message.type == "human":
#             print(f"\nHUMAN: {message.content}")
#         else:
#             print("UNKNOWN TYPE: ", message.type)