from google_auth_oauthlib.flow import InstalledAppFlow
import os

# Both scopes needed:
# - readonly: to read/search emails for RAG indexing
# - send: to send and reply to emails via tools
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]
CREDENTIALS_PATH = "credentials.json"
def authenticate():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=8080)

    with open("token.json", "w") as f:
        f.write(creds.to_json())

    print("✅ token.json saved — you won't need to run this again.")

if __name__ == "__main__":
    authenticate()