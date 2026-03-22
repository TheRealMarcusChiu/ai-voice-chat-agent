
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from config import MyContext


@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    if "plano" in city.lower():
        return "sunny"
    elif "sf" in city.lower() or "san francisco" in city.lower():
        return "foggy"
    elif "houston" in city.lower():
        return "hot and humid"  
    elif "austin" in city.lower():
        return "partly cloudy"
    elif "new york" in city.lower():
        return "rainy"
    else:
        return f"default sunny"

@tool
def get_user_location(runtime: ToolRuntime[MyContext]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"