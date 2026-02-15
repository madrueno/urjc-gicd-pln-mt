from typing import TypedDict
from langgraph.graph import StateGraph, END

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ===== LLM (Ollama por defecto) =====
# from langchain_ollama import ChatOllama
#llm = ChatOllama(model="llama3.2", temperature=0)

# Si prefieres OpenAI:
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class State(TypedDict):
    question: str
    normalized_question: str
    answer: str

def preprocess(state: State) -> State:
    q = state["question"].strip()
    state["normalized_question"] = " ".join(q.lower().split())
    print("[preprocess] normalized_question:", state["normalized_question"])
    return state

prompt = ChatPromptTemplate.from_template(
    "Responde con este formato exacto:\n"
    "Título: <una línea>\n"
    "- bullet 1\n"
    "- bullet 2\n"
    "- bullet 3\n\n"
    "Pregunta: {q}"
)

def llm_answer(state: State) -> State:
    out = (prompt | llm | StrOutputParser()).invoke({"q": state["normalized_question"]})
    state["answer"] = out
    print("[llm_answer] chars:", len(out))
    return state

def postprocess(state: State) -> State:
    a = state.get("answer", "").strip()
    quality = []
    quality.append("✅ hay respuesta" if a else "❌ sin respuesta")
    quality.append("✅ es clara" if (0 < len(a) < 900) else "⚠️ demasiado larga o vacía")

    state["answer"] = a + "\n\nCalidad:\n- " + "\n- ".join(quality)
    print("[postprocess] done")
    return state

g = StateGraph(State)
g.add_node("preprocess", preprocess)
g.add_node("llm_answer", llm_answer)
g.add_node("postprocess", postprocess)

g.set_entry_point("preprocess")
g.add_edge("preprocess", "llm_answer")
g.add_edge("llm_answer", "postprocess")
g.add_edge("postprocess", END)

app = g.compile()

if __name__ == "__main__":
    result = app.invoke({"question": "  ¿Qué es RAG?  ", "normalized_question": "", "answer": ""})
    print("\n=== OUTPUT ===\n", result["answer"])
