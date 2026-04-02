import os
import importlib.util
os.environ["USER_AGENT"] = "rag/1.0"

from typing import Any
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Import from gmail-tools.py using importlib
spec = importlib.util.spec_from_file_location(
    "gmail_tools",
    os.path.join(os.path.dirname(__file__), "gmail-tools.py")
)
gmail_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gmail_tools)

get_gmail_service = gmail_tools.get_gmail_service
extract_body      = gmail_tools.extract_body
send_email        = gmail_tools.send_email
reply_to_email    = gmail_tools.reply_to_email
search_inbox      = gmail_tools.search_inbox
get_recent_emails = gmail_tools.get_recent_emails

# Step 1: Model + vector store
model = ChatOllama(model="llama3.2", temperature=0, base_url="http://192.168.111.160:11434")
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://192.168.111.160:11434")
vector_store = InMemoryVectorStore(embeddings)

# Step 2: Fetch emails → summarize → index into RAG
def index_emails(max_results: int = 50):
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=max_results).execute()
    docs = []
    for msg in results.get("messages", []):
        email = service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()
        headers = {h["name"]: h["value"] for h in email["payload"]["headers"]}
        subject = headers.get("Subject", "No Subject")
        sender  = headers.get("From", "Unknown")
        date    = headers.get("Date", "")
        body    = extract_body(email)
        if not body.strip():
            continue

        summary = model.invoke(
            f"Summarize this email in 3 sentences.\n"
            f"From: {sender}\nSubject: {subject}\nBody: {body[:2000]}"
        ).content

        docs.append(Document(
            page_content=summary,
            metadata={"subject": subject, "from": sender, "date": date, "email_id": email["id"]}
        ))

    vector_store.add_documents(docs)
    print(f"Indexed {len(docs)} emails into RAG.")

# Step 3: Middleware
class RetrieveDocumentsMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text, k=3)
        docs_content = "\n\n".join(
            f"From: {d.metadata['from']}\n"
            f"Subject: {d.metadata['subject']}\n"
            f"Date: {d.metadata['date']}\n"
            f"Summary: {d.page_content}"
            for d in retrieved_docs
        )
        return {
            "system": (
                "You are a personal voice email assistant. "
                "Use the retrieved email summaries to answer questions. "
                "If you don't know the answer, say you don't know. "
                "Keep answers concise — three sentences maximum."
                f"\n\n{docs_content}"
            )
        }

# Step 4: Index emails on startup
index_emails(max_results=50)

# Step 5: Agent
agent = create_agent(
    model,
    tools=[send_email, reply_to_email, search_inbox, get_recent_emails],
    middleware=[RetrieveDocumentsMiddleware()]
)

if __name__ == "__main__":
    print("=" * 40)
    print("Indexing emails into RAG...")
    max_results = int(input("How many emails to index? (e.g. 10, 50): "))
    index_emails(max_results=max_results)

    print("=" * 40)
    print("RAG Email Assistant ready!")
    print("Type 'exit' to quit.")
    print("=" * 40)

    while True:
        query = input("\nAsk a question about your emails: ")
        if query.lower() == "exit":
            break
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()