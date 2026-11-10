## that is add teh rag function import os
import os
import hashlib
import sqlite3
import tempfile
from io import BytesIO
from typing import TypedDict, Annotated

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

MODEL = "openai/gpt-oss-120b"

# Main LLM (answers, streamed to UI)
llm = ChatGroq(model=MODEL, temperature=0)

# Helper LLM (router / rewrite): "nostream" tag => UI mein token stream nahi hoga
helper_llm = ChatGroq(model=MODEL, temperature=0, max_tokens=400, tags=["nostream"])


# ======================= RAG: vector store =======================
_VS_PATH = os.path.join(tempfile.gettempdir(), "samvad_vector_store")
_chroma = chromadb.PersistentClient(path=_VS_PATH)
_collection = _chroma.get_or_create_collection(
    name="samvad_docs", metadata={"hnsw:space": "cosine"}
)

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:  # lazy load: app startup tez rahe
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def add_pdf(file_bytes: bytes, filename: str, thread_id: str) -> int:
    """PDF ko chunk + embed karke current thread ke liye store karta hai. Returns chunks added."""
    reader = PdfReader(BytesIO(file_bytes))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    texts, metas = [], []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            metas.append({
                "thread_id": str(thread_id),
                "source": filename,
                "page_label": str(page_no),
                "chunk_index": len(texts),
            })
            texts.append(piece)

    if not texts:
        return 0

    embeddings = _get_embedder().encode(texts).tolist()
    file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
    ids = [f"{thread_id}_{file_hash}_{i}" for i in range(len(texts))]

    # upsert: same PDF dobara upload karo to duplicate nahi banega
    _collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
    return len(texts)


def doc_count(thread_id: str) -> int:
    """Is thread mein kitne chunks stored hain."""
    res = _collection.get(where={"thread_id": str(thread_id)}, include=[])
    return len(res["ids"])


def _retrieve(query: str, thread_id: str, top_k: int = 5, score_threshold: float = 0.3):
    if doc_count(thread_id) == 0:
        return []
    q_emb = _get_embedder().encode([query])[0].tolist()
    res = _collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"thread_id": str(thread_id)},
    )
    hits = []
    for i in range(len(res["ids"][0])):
        score = 1 - res["distances"][0][i]
        if score >= score_threshold:
            hits.append({
                "content": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": score,
            })
    return hits


def _to_sources(hits):
    return [
        {
            "source": h["metadata"]["source"],
            "page": h["metadata"]["page_label"],
            "score": round(h["score"], 2),
            "snippet": h["content"][:300],
        }
        for h in hits
    ]


# ======================= LangGraph state =======================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    sources: list


def _thread_id(config: RunnableConfig) -> str:
    return str(config.get("configurable", {}).get("thread_id", "default"))


def _last_user_text(state: ChatState) -> str:
    for m in reversed(state["messages"]):
        if m.type == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


# ======================= Nodes =======================
def router_node(state: ChatState, config: RunnableConfig):
    """chat / rag / summary decide karta hai."""
    tid = _thread_id(config)
    if doc_count(tid) == 0:
        return {"route": "chat"}

    query = _last_user_text(state)
    prompt = (
        "Classify the user message. Reply with exactly ONE word: chat, rag or summary.\n"
        "chat = greeting, thanks, small talk, or a general question that needs no document.\n"
        "rag = a specific question that needs information from the uploaded documents.\n"
        "summary = asks for a summary, overview, main/important topics or key points of the whole document.\n\n"
        "Examples:\n"
        "hello -> chat\n"
        "thanks! -> chat\n"
        "what is the capital of France? -> chat\n"
        "write a python function to reverse a string -> chat\n"
        "what does the document say about attention? -> rag\n"
        "explain section 3 -> rag\n"
        "in big data -> rag\n"
        "summarize the document -> summary\n"
        "what are the important topics in the document? -> summary\n\n"
        f"Message: {query}\nAnswer:"
    )
    try:
        out = helper_llm.invoke(prompt).content.strip().lower()
    except Exception:
        return {"route": "rag"}  # router fail ho to document mode safe hai

    if "summary" in out:
        return {"route": "summary"}
    if "chat" in out and "rag" not in out:
        return {"route": "chat"}
    return {"route": "rag"}


def route_decider(state: ChatState) -> str:
    return state.get("route", "chat")


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response], "sources": []}


def _rewrite(state: ChatState, query: str) -> str:
    """Follow-up ("in big data") ko standalone question banata hai."""
    history = state["messages"][:-1][-4:]
    if not history:
        return query
    convo = "\n".join(f"{m.type}: {str(m.content)[:300]}" for m in history)
    prompt = (
        "Rewrite the last user message as a standalone question using the conversation for context. "
        "If it is already standalone, return it unchanged. Output only the question.\n\n"
        f"Conversation:\n{convo}\n\nLast user message: {query}\nStandalone question:"
    )
    try:
        return helper_llm.invoke(prompt).content.strip() or query
    except Exception:
        return query


def rag_node(state: ChatState, config: RunnableConfig):
    tid = _thread_id(config)
    query = _last_user_text(state)
    q = _rewrite(state, query)

    hits = _retrieve(q, tid, top_k=6, score_threshold=0.3)
    if not hits:
        hits = _retrieve(q, tid, top_k=6, score_threshold=0.0)  # threshold fallback
    if not hits:
        msg = AIMessage(content="Ye information uploaded documents mein nahi mili.")
        return {"messages": [msg], "sources": []}

    context = "\n\n".join(h["content"] for h in hits)
    system = (
        "Answer ONLY using the context below. Do not use outside knowledge.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I don't know based on the provided documents.\"\n"
        "Write formulas in plain text, not LaTeX.\n\n"
        f"Context:\n{context}"
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=q)])
    return {"messages": [response], "sources": _to_sources(hits)}


def summary_node(state: ChatState, config: RunnableConfig):
    tid = _thread_id(config)
    query = _last_user_text(state)

    data = _collection.get(
        where={"thread_id": tid}, include=["documents", "metadatas"]
    )
    items = sorted(
        zip(data["documents"], data["metadatas"]),
        key=lambda x: (x[1].get("source", ""), x[1].get("chunk_index", 0)),
    )
    if not items:
        msg = AIMessage(content="Is chat mein koi document upload nahi hua.")
        return {"messages": [msg], "sources": []}

    max_chunks = 12
    step = max(1, len(items) // max_chunks)
    picked = items[::step][:max_chunks]  # poore doc se evenly spaced chunks

    context = "\n\n".join(d for d, _ in picked)
    system = (
        "Using ONLY the document excerpts below, answer the request. "
        "Give a clear structured answer with the main topics as short bullet points. "
        "Write formulas in plain text, not LaTeX.\n\n"
        f"Excerpts:\n{context}"
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=query)])
    hits = [{"content": d, "metadata": m, "score": 1.0} for d, m in picked]
    return {"messages": [response], "sources": _to_sources(hits)}


# ======================= Checkpointer =======================
DB_PATH = os.path.join(tempfile.gettempdir(), "chatbot.db")
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ======================= Graph =======================
graph = StateGraph(ChatState)
graph.add_node("router", router_node)
graph.add_node("chat_node", chat_node)
graph.add_node("rag_node", rag_node)
graph.add_node("summary_node", summary_node)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_decider,
    {"chat": "chat_node", "rag": "rag_node", "summary": "summary_node"},
)
graph.add_edge("chat_node", END)
graph.add_edge("rag_node", END)
graph.add_edge("summary_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_thread():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_thread)

if __name__ == "__main__":
    png = chatbot.get_graph().draw_mermaid_png()
    with open("samvad_graph.png", "wb") as f:
        f.write(png)
    print("Saved: samvad_graph.png")










# import os
# import sqlite3
# import tempfile
# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph.message import add_messages
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
# import sqlite3
# import tempfile

# load_dotenv()

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0
# )

# class ChatState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# def chat_node(state: ChatState):
#     messages = state['messages']
#     response = llm.invoke(messages)
#     return {"messages": [response]}


# DB_PATH = os.path.join(tempfile.gettempdir(), "chatbot.db")
# conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
# # Checkpointer
# checkpointer = SqliteSaver(conn=conn)           

# graph = StateGraph(ChatState)
# graph.add_node("chat_node", chat_node)
# graph.add_edge(START, "chat_node")
# graph.add_edge("chat_node", END)

# chatbot = graph.compile(checkpointer=checkpointer)

# def retrive_all_thread():
#     all_thread = set()
#     for checkpoint in checkpointer.list(None):
#         all_thread.add(checkpoint.config['configurable']['thread_id'])
#     return list(all_thread)   



# CONFIG = {'configurable': {'thread_id': 'thread-1'}}



# response = chatbot.invoke(
#     {
#         'messages': [
#             HumanMessage(content='hey my name is vivek')
#         ]
#     },
#     config= CONFIG
# )

## chatbot.get_state(config =CONFIG).value['message']
