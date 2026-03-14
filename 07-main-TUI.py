import asyncio
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from typing import AsyncIterator
import os
from dotenv import load_dotenv

from prompt_toolkit import PromptSession
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout.containers import FloatContainer, Float
from prompt_toolkit.layout.menus import CompletionsMenu


load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")


# ── Domain types ─────────────────────────────────────────────────────────────

@dataclass
class MyContext:
    user_id: str

class MyResponseFormat(BaseModel):
    agent_response: str


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[MyContext]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# ── Agent ────────────────────────────────────────────────────────────────────

agent = create_agent(
    model=ChatOllama(model="llama3.2", temperature=0, base_url=OLLAMA_BASE_URL),
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=MyContext,
    response_format=ToolStrategy(MyResponseFormat),
    checkpointer=InMemorySaver(),
)

async def stream_agent(user_prompt: str) -> AsyncIterator[str]:
    stream = agent.astream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"configurable": {"thread_id": "1"}},
        context=MyContext(user_id="1"),
        stream_mode="messages",
    )
    async for message, _ in stream:
        if isinstance(message, AIMessage):
            yield message.text


# ── TUI ──────────────────────────────────────────────────────────────────────

class ChatUI:
    def __init__(self):
        self._lines: list[str] = []          # rendered history lines
        self._history_control = FormattedTextControl(
            text=self._get_history_text,
            focusable=False,
        )

        # Scrollable history pane (fills all available space)
        self._history_window = Window(
            content=self._history_control,
            wrap_lines=True,
            # Always scroll to bottom
            get_line_prefix=None,
        )

        # Single-line input at the very bottom
        self._input_area = TextArea(
            height=1,
            prompt=HTML("<ansicyan><b>You: </b></ansicyan>"),
            multiline=False,
            wrap_lines=False,
            style="class:input-area",
        )

        # Thin separator line
        self._separator = Window(height=1, char="─", style="class:separator")

        self._layout = Layout(
            FloatContainer(
                content=HSplit([
                    self._history_window,
                    self._separator,
                    self._input_area,
                ]),
                floats=[],
            ),
            focused_element=self._input_area,
        )

        self._kb = KeyBindings()
        self._app: Application | None = None

        @self._kb.add("enter")
        def _on_enter(event):
            text = self._input_area.text.strip()
            if not text:
                return
            self._input_area.text = ""
            # Fire-and-forget: schedule coroutine on the running loop
            asyncio.get_event_loop().create_task(self._handle_message(text))

        @self._kb.add("c-c")
        @self._kb.add("c-q")
        def _quit(event):
            event.app.exit()

        self._style = Style.from_dict({
            "separator":  "fg:#444444",
            "input-area": "bg:#1a1a1a fg:#ffffff",
        })

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_history_text(self) -> list[tuple[str, str]]:
        """Return prompt_toolkit formatted-text for the history pane."""
        result = []
        for line in self._lines:
            result.append(("", line + "\n"))
        return result

    def _push_line(self, line: str) -> None:
        self._lines.append(line)
        if self._app:
            self._app.invalidate()

    def _append_to_last(self, chunk: str) -> None:
        """Stream tokens into the last line without adding a newline."""
        if self._lines:
            self._lines[-1] += chunk
        else:
            self._lines.append(chunk)
        if self._app:
            self._app.invalidate()

    # ── message handler ───────────────────────────────────────────────────────

    async def _handle_message(self, user_text: str) -> None:
        # Show user bubble
        self._push_line(f"[You]  {user_text}")
        # Reserve an empty line for the AI response
        self._push_line("[AI]   ")

        async for chunk in stream_agent(user_text):
            self._append_to_last(chunk)

        # Blank line between exchanges for readability
        self._push_line("")

    # ── run ───────────────────────────────────────────────────────────────────

    async def run_async(self) -> None:
        self._app = Application(
            layout=self._layout,
            key_bindings=self._kb,
            style=self._style,
            full_screen=True,
            mouse_support=True,
            color_depth=None,
        )
        await self._app.run_async()


# ── Entry point ──────────────────────────────────────────────────────────────

async def main():
    ui = ChatUI()
    await ui.run_async()

if __name__ == "__main__":
    asyncio.run(main())