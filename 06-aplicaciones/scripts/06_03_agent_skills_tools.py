from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#from langchain_ollama import ChatOllama
#llm = ChatOllama(model="llama3.2", temperature=0)

from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- (Opcional) Tool RAG real: aquí lo dejamos como placeholder para que funcione siempre ---
# Si quieres RAG real, reutiliza el retriever de P3 y reemplaza rag_tool().

class State(TypedDict):
    question: str
    intent: Literal["explain", "procedure", "other"]
    tool_needed: Literal["none", "rag", "calc"]
    context: str
    sources: str
    answer: str
    steps: int

intent_prompt = ChatPromptTemplate.from_template(
    "Clasifica intención en UNA palabra: explain, procedure, other.\n"
    "- explain: definir y explicar un concepto.\n"
    "- procedure: pedir pasos o procedimiento.\n"
    "- other: resto.\n\n"
    "Pregunta: {q}\nIntent:"
)

tool_prompt = ChatPromptTemplate.from_template(
    "Decide herramienta en UNA palabra: rag, calc, none.\n"
    "- rag: si necesitas normativa/documentos.\n"
    "- calc: si hay operación numérica.\n"
    "- none: si no.\n\n"
    "Pregunta: {q}\nTool:"
)

def intent_router(state: State) -> State:
    state["steps"] += 1
    label = (intent_prompt | llm | StrOutputParser()).invoke({"q": state["question"]}).strip().lower()
    state["intent"] = label if label in ("explain", "procedure", "other") else "other"
    print("[intent_router]", state["intent"])
    return state

def tool_router(state: State) -> State:
    t = (tool_prompt | llm | StrOutputParser()).invoke({"q": state["question"]}).strip().lower()
    if "rag" in t:
        state["tool_needed"] = "rag"
    elif "calc" in t:
        state["tool_needed"] = "calc"
    else:
        state["tool_needed"] = "none"
    print("[tool_router]", state["tool_needed"])
    return state

def calc_tool(state: State) -> State:
    import re
    expr = re.sub(r"[^0-9\+\-\*\/\.\(\) ]", "", state["question"])
    try:
        val = eval(expr, {"__builtins__": {}})
        state["context"] = f"Resultado cálculo: {val}"
        state["sources"] = "- calc_tool"
    except:
        state["context"] = ""
        state["sources"] = "- calc_tool (fallo)"
    return state

def rag_tool(state: State) -> State:
    # Placeholder: sustituye por RAG real (retriever.invoke + fuentes) cuando quieras.
    state["context"] = "Contexto recuperado (placeholder). Sustituye por tu RAG real."
    state["sources"] = "- (placeholder)"
    return state

explainer_prompt = ChatPromptTemplate.from_template(
    "Genera una explicación pedagógica en español con este formato:\n"
    "Definición: ...\n"
    "Ejemplo: ...\n"
    "Advertencia: ...\n\n"
    "Usa el contexto si existe.\n"
    "Pregunta: {q}\nContexto: {ctx}\n\nRespuesta:"
)

checklist_prompt = ChatPromptTemplate.from_template(
    "Genera una checklist (máx. 6 pasos) para el procedimiento solicitado.\n"
    "Usa el contexto si existe.\n"
    "Pregunta: {q}\nContexto: {ctx}\n\nChecklist:"
)

other_prompt = ChatPromptTemplate.from_template(
    "Responde de forma clara y breve. Usa el contexto si existe.\n"
    "Pregunta: {q}\nContexto: {ctx}\n\nRespuesta:"
)

def skills_node(state: State) -> State:
    if state["steps"] > 8:
        state["answer"] = "max_steps alcanzado. Reformula la pregunta o reduce el alcance."
        return state

    if state["intent"] == "explain":
        out = (explainer_prompt | llm | StrOutputParser()).invoke({"q": state["question"], "ctx": state["context"]})
    elif state["intent"] == "procedure":
        out = (checklist_prompt | llm | StrOutputParser()).invoke({"q": state["question"], "ctx": state["context"]})
    else:
        out = (other_prompt | llm | StrOutputParser()).invoke({"q": state["question"], "ctx": state["context"]})

    if "Fuentes:" not in out:
        out = out.rstrip() + "\n\nFuentes:\n" + (state["sources"] if state["sources"] else "- (ninguna)")
    state["answer"] = out
    return state

def route_tool(state: State):
    return state["tool_needed"]

g = StateGraph(State)
g.add_node("intent_router", intent_router)
g.add_node("tool_router", tool_router)
g.add_node("calc", calc_tool)
g.add_node("rag", rag_tool)
g.add_node("skills", skills_node)

g.set_entry_point("intent_router")
g.add_edge("intent_router", "tool_router")
g.add_conditional_edges("tool_router", route_tool, {"calc": "calc", "rag": "rag", "none": "skills"})
g.add_edge("calc", "skills")
g.add_edge("rag", "skills")
g.add_edge("skills", END)

app = g.compile()

if __name__ == "__main__":
    tests = [
        "Explica qué es RAG y para qué sirve.",
        "¿Cómo entregar las prácticas paso a paso?",
        "¿Cuánto es (8+2)*3?"
    ]
    for t in tests:
        print("\n========================")
        print("Q:", t)
        out = app.invoke({
            "question": t,
            "intent": "other",
            "tool_needed": "none",
            "context": "",
            "sources": "",
            "answer": "",
            "steps": 0
        })
        print("A:", out["answer"])
