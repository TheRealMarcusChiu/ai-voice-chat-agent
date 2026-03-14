from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from pydantic import BaseModel 
from langchain_ollama import ChatOllama
from langchain_core.messages import messages_to_dict
import json
import os
from dotenv import load_dotenv


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.111.160:11434")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location. Otherwise use the provided location in the question.
""")


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
        base_url = "http://192.168.111.160:11434"),
    system_prompt = SYSTEM_PROMPT,
    tools = [get_user_location, get_weather_for_location],
    context_schema = MyContext,
    response_format = ToolStrategy(MyResponseFormat),
    checkpointer = InMemorySaver())

while True:

    user_input = input("\nHUMAN: ")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config = {"configurable": {"thread_id": "1"}},
        context = MyContext(user_id = "1")
    )
    
    # print(response["messages"])
    # print(json.dumps(messages_to_dict(response["messages"]), indent=2))

    last_n_messages_to_print = 3
    for i in range(last_n_messages_to_print, 0, -1):
        message = response["messages"][-i]

        if message.type == "ai": 
            print(f"\nAI: {message.content}")
            if len(message.tool_calls) > 0:
                print("AI tool-calls:")
                for tool_call in message.tool_calls:
                    print("-", tool_call)
        elif message.type == "tool":
            print(f"\nTOOL: {message.content}")
        elif message.type == "human":
            print(f"\nHUMAN: {message.content}")
        else:
            print("UNKNOWN TYPE: ", message.type)