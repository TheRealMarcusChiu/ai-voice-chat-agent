# Setup Instructions

- install `espeak-ng` for TTS:
  - Ubuntu/Linux: `sudo apt-get install espeak-ng`
  - macOS: `brew install espeak-ng`
  - Windows: Download the `.msi` from the [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng/releases)
- install UV (https://github.com/astral-sh/uv)
- `uv sync`
- `uv pip install pip`
- `source .venv/bin/activate` (optional)
- `python main.py`

To add additional dependencies:

- `uv add DEPENDENCY_NAME_HERE`
- `uv sync`

# Customize Dev Env

- create new file `.env` with `.env.example` as template
