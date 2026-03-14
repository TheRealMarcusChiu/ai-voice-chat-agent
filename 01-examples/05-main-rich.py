import time
from rich.console import Console
from rich.live import Live
from rich.text import Text

console = Console()

def fake_stream(text: str, delay: float = 0.05):
    for char in text:
        yield char
        time.sleep(delay)

def get_response(prompt: str) -> str:
    return f"You prompted: '{prompt}'. Here is my response!"

def chat():
    while True:
        user_input = console.input("[bold cyan]You: [/]")

        accumulated = ""
        with Live(console=console, refresh_per_second=20) as live:
            for chunk in fake_stream(get_response(user_input)):
                accumulated += chunk
                text = Text()
                text.append("AI: ", style="bold red")
                text.append(accumulated)
                live.update(text)

        console.print()

chat()