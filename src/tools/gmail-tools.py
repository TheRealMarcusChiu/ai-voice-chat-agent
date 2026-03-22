import base64
from email.mime.text import MIMEText
from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

def get_gmail_service():
    """Get authenticated Gmail service, auto-refreshing token if expired."""
    creds = Credentials.from_authorized_user_file(TOKEN_PATH)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def extract_body(email: dict) -> str:
    """Extract plain text body from raw Gmail message."""
    # emails with multiple parts
    parts = email.get("payload", {}).get("parts", [])
    for part in parts:
        if part["mimeType"] == "text/plain":
            data = part["body"].get("data", "")
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    # simple emails with no parts
    data = email.get("payload", {}).get("body", {}).get("data", "")
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore") if data else ""

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a new email to a recipient."""
    service = get_gmail_service()
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return f"✅ Email sent to {to}"

@tool
def reply_to_email(email_id: str, body: str) -> str:
    """Reply to an existing email using its ID."""
    service = get_gmail_service()
    original = service.users().messages().get(
        userId="me", id=email_id, format="full"
    ).execute()
    headers = {h["name"]: h["value"] for h in original["payload"]["headers"]}
    message = MIMEText(body)
    message["to"]           = headers.get("From")
    message["subject"]      = "Re: " + headers.get("Subject", "")
    message["In-Reply-To"]  = headers.get("Message-ID")
    message["References"]   = headers.get("Message-ID")
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(
        userId="me",
        body={"raw": raw, "threadId": original["threadId"]}
    ).execute()
    return f"✅ Replied to email {email_id}"

@tool
def search_inbox(query: str) -> str:
    """Search Gmail inbox for emails matching a query."""
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", q=query, maxResults=5
    ).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No emails found."
    return "\n".join(f"Email ID: {m['id']}" for m in messages)

@tool
def get_recent_emails(max_results: int = 10) -> str:
    """Get the most recent emails from inbox."""
    service = get_gmail_service()
    results = service.users().messages().list(
        userId="me", maxResults=max_results
    ).execute()
    messages = results.get("messages", [])
    if not messages:
        return "No emails found."

    output = []
    for msg in messages:
        email = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()
        headers = {h["name"]: h["value"] for h in email["payload"]["headers"]}
        output.append(
            f"ID: {msg['id']}\n"
            f"From: {headers.get('From', 'Unknown')}\n"
            f"Subject: {headers.get('Subject', 'No Subject')}\n"
            f"Date: {headers.get('Date', '')}\n"
        )
    return "\n---\n".join(output)


# Quick test 
if __name__ == "__main__":
    print("=" * 40)

    # Test 1: Get recent emails
    print("TEST 1: Get recent emails")
    emails = get_recent_emails.invoke({"max_results": 2})
    print(emails)

    # Test 2: Search inbox
    print("=" * 40)
    print("TEST 2: Search inbox")
    query = input("Enter search query: ")
    search_result = search_inbox.invoke({"query": query})
    print(search_result)

    # Test 3: Send email
    print("=" * 40)
    print("TEST 3: Send email")
    to      = input("Enter recipient gmail ID: ")
    subject = input("Enter subject: ")
    body    = input("Enter body: ")
    send_result = send_email.invoke({"to": to, "subject": subject, "body": body})
    print(send_result)

    # Test 4: Reply to email
    print("=" * 40)
    print("TEST 4: Reply to email")
    email_id = input("Enter email ID to reply to (from TEST 1 output): ")
    body     = input("Enter reply body: ")
    reply_result = reply_to_email.invoke({"email_id": email_id, "body": body})
    print(reply_result)

    # Test 5: Extract body
    print("=" * 40)
    print("TEST 5: Extract body")
    email_id = input("Enter email ID to extract body (from TEST 1 output): ")
    service  = get_gmail_service()
    raw_email = service.users().messages().get(
        userId="me", id=email_id, format="full"
    ).execute()
    body = extract_body(raw_email)
    print(f"Body preview: {body[:200]}")