from pathlib import Path
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#from langchain_ollama import ChatOllama
#llm = ChatOllama(model="llama3.2", temperature=0)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION = "agent_rag_demo"

# --- build / load vector store ---
docs = []
for fp in DATA_DIR.glob("*.txt"):
    loaded = TextLoader(str(fp), encoding="utf-8").load()
    for d in loaded:
        d.metadata["source"] = fp.name
    docs.extend(loaded)

splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vs = Chroma.from_documents(
    documents=splits,
    embedding=emb,
    persist_directory=str(PERSIST_DIR),
    collection_name=COLLECTION,
)
retriever = vs.as_retriever(search_kwargs={"k": 4})

class State(TypedDict):
    question: str
    decision: Literal["rag", "direct"]
    retrieved_context: str
    sources: str
    answer: str

decide_prompt = ChatPromptTemplate.from_template(
    "Decide si necesitas consultar documentos (rag) o puedes responder sin documentos (direct).\n"
    "Si la pregunta menciona porcentajes, políticas, convocatoria, normativa o asistencia, usa rag.\n\n"
    "Pregunta: {q}\nDecisión:"
)

def decide(state: State) -> State:
    d = (decide_prompt | llm | StrOutputParser()).invoke({"q": state["question"]}).strip().lower()
    state["decision"] = "rag" if "rag" in d else "direct"
    print("[decide] decision:", state["decision"])
    return state

def rag_tool(state: State) -> State:
    retrieved_docs = retriever.invoke(state["question"])
    state["retrieved_context"] = "\n\n".join(d.page_content for d in retrieved_docs)

    sources = []
    for d in retrieved_docs:
        s = d.metadata.get("source", "desconocido")
        if s not in sources:
            sources.append(s)
    state["sources"] = "\n".join(f"- {s}" for s in sources)
    print("[rag_tool] sources:", sources)
    return state

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Responde SOLO con el contexto. Si el contexto no es suficiente, responde exactamente: "
     "'No puedo responder con la información disponible.' "
     "Termina con una sección 'Fuentes:' listando las fuentes."),
    ("human", "Pregunta: {q}\n\nContexto:\n{ctx}")
])

def final_answer(state: State) -> State:
    ctx = state["retrieved_context"] if state["decision"] == "rag" else ""
    out = (answer_prompt | llm | StrOutputParser()).invoke({"q": state["question"], "ctx": ctx})

    if "Fuentes:" not in out:
        out = out.rstrip() + "\n\nFuentes:\n" + (state["sources"] if state["sources"] else "- (ninguna)")
    state["answer"] = out
    return state

def route(state: State):
    return state["decision"]

g = StateGraph(State)
g.add_node("decide", decide)
g.add_node("rag_tool", rag_tool)
g.add_node("final", final_answer)

g.set_entry_point("decide")
g.add_conditional_edges("decide", route, {"rag": "rag_tool", "direct": "final"})
g.add_edge("rag_tool", "final")
g.add_edge("final", END)

app = g.compile()

if __name__ == "__main__":
    tests = [
        "¿Cuáles son los porcentajes de evaluación?",
        "¿Hay convocatoria extraordinaria?",
        "Explica brevemente qué es un LLM."
    ]
    for t in tests:
        print("\n========================")
        print("Q:", t)
        out = app.invoke({"question": t, "decision": "direct", "retrieved_context": "", "sources": "", "answer": ""})
        print("A:", out["answer"])
