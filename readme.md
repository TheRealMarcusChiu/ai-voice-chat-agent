# Setup Instructions

- install `espeak-ng` for TTS:
  - Ubuntu/Linux: `sudo apt-get install espeak-ng`
  - macOS: `brew install espeak-ng`
  - Windows: Download the `.msi` from the [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng/releases)
- install UV (https://github.com/astral-sh/uv)
- `uv sync`
- `uv pip install pip`

# How to Run

Way #1:

- `uv run main.py`

Way #2:

- `source .venv/bin/activate`
- `python main.py`


# Adding Additional Dependencies

- `uv add DEPENDENCY_NAME_HERE`
- `uv sync`

# Customize Dev Env

- create new file `.env` with `.env.example` as template

## Gmail API Integration

### Step 1: Create a Google Cloud Project

1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Click **"Select a project"** (top left) → **"New Project"**
3. Name it `ai-voice-agent` → click **Create**

### Step 2: Enable Gmail API

1. In the search bar type **"Gmail API"**
2. Click it → click **"Enable"**

### Step 3: Configure OAuth Consent Screen

1. Go to **APIs & Services** → **OAuth consent screen**
2. Choose **External** → click **Create**
3. Fill in:
   - App name: `ai-voice-agent`
   - User support email: your Gmail
4. Click **Save and Continue**
5. Click **"Add or Remove Scopes"** → add:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
6. Click **Update** → **Save and Continue**
7. Under **"Audience"** tab → **"+ Add Users"** → add your Gmail address
8. Click **Save**

### Step 4: Create OAuth Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **"+ Create Credentials"** → **"OAuth Client ID"**
3. Select **"User data"** → **Next**
4. Application type → **Desktop App**
5. Name: `ai-voice-agent` → **Create**
6. Click **Download JSON**
7. Rename it to `credentials.json`
8. Move it to your project root:

```powershell
move "$env:USERPROFILE\Downloads\client_secret_*.json" credentials.json
```

### Step 5: Add Redirect URI

1. Go to **APIs & Services** → **Credentials** → click your OAuth client
2. Under **"Authorized redirect URIs"** → **"+ Add URI"**
3. Add: `http://localhost:8080/`
4. Click **Save** and wait 2 minutes

### Step 6: Install Google auth packages

```powershell
uv add google-auth google-auth-oauthlib google-api-python-client
```

---
### Step 1: Authenticate with Gmail (one time only)

```powershell
cd ai-voice-chat-agent
uv run src/tools/gmail-auth.py
```

- A browser window will open
- Log in with your Gmail account
- Complete the consent screen
- `token.json` will be automatically saved to your project root

> You will never need to run this again unless `token.json` is deleted.

### Step 2: Test Gmail tools

```powershell
uv run src/tools/gmail-tools.py
```

This will interactively test:
- Fetching recent emails
- Searching your inbox
- Sending a test email
- Replying to an email
- Extracting email body

### Step 3: Run the RAG Email Agent

```powershell
uv run src/tools/rag.py
```

- Enter how many emails to index (e.g. `50`)
- Wait for indexing to complete
- Ask questions about your emails:

```
Ask a question about your emails: What did John email me about?
Ask a question about your emails: Any emails about the project?
Ask a question about your emails: Summarize emails from last week
```

Type `exit` to quit.


