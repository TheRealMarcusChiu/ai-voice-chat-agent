import asyncio
import time
from rich.console import Console
from rich.live import Live
from rich.text import Text

console = Console()

def fake_stream(text: str, delay: float = 0.05):
    for char in text:
        yield char
        time.sleep(delay)

def stream_response(prompt: str):
    response = f"You asked: '{prompt}'. Here is my response! " * 3
    accumulated = ""
    with Live(console=console, refresh_per_second=20) as live:
        for chunk in fake_stream(response):
            accumulated += chunk
            live.update(Text(accumulated))

stream_response("Tell me a joke")