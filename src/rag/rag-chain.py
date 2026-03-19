import os
os.environ["USER_AGENT"] = "rag/1.0"

import bs4
from typing import Any
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents.middleware import AgentMiddleware, AgentState

#  Step1: need to load our data. This is done with Document Loaders.
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Step2:Text splitters break large Documents into smaller chunks.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True, )
all_splits = text_splitter.split_documents(docs)


# Step3:to store and index our splits, so that they can be searched over later.
embeddings = OllamaEmbeddings(model="nomic-embed-text")  # best local embedding model
vector_store = InMemoryVectorStore(embeddings)
# Index chunks
_ = vector_store.add_documents(documents=all_splits)


# Step 4: Retrieval middleware
class RetrieveDocumentsMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return {
            "system": (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Keep the answer concise — three sentences maximum."
                f"\n\n{docs_content}"
            )
        }

# Step 5: Model and agent
model = ChatOllama(
        model = "llama3.2",
        temperature = 0,
        base_url = "http://192.168.111.160:11434")
agent = create_agent(model, tools=[], middleware=[RetrieveDocumentsMiddleware()])

query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()