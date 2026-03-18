# Setup Instructions

- install `espeak-ng` for TTS:
  - Ubuntu/Linux: `sudo apt-get install espeak-ng`
  - macOS: `brew install espeak-ng`
  - Windows: Download the `.msi` from the [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng/releases)
- install UV (https://github.com/astral-sh/uv)
- `uv sync`
- `uv pip install pip`

# How to Run (Terminal)

Way #1:

- `uv run main.py`

Way #2:

- `source .venv/bin/activate`
- `python main.py`

# How to Run (Web)

- `uv run main-web.py`
- open `index.html` in browser

# Adding Additional Dependencies

- `uv add DEPENDENCY_NAME_HERE`
- `uv sync`

# Customize Dev Env

- create new file `.env` with `.env.example` as template
