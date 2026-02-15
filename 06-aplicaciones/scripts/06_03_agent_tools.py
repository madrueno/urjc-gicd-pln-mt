from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
#from langchain_ollama import ChatOllama
#llm = ChatOllama(model="llama3.2", temperature=0)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

POLICY_DB = {
    "evaluación continua": "Sistema principal: prácticas 40%, proyecto 30%, cuestionarios 20%, participación 10%.",
    "convocatoria extraordinaria": "Existe convocatoria extraordinaria; se permite recuperar partes suspensas (según normativa).",
    "asistencia": "La asistencia no es obligatoria, pero la participación puede puntuar si hay evidencia."
}

class State(TypedDict):
    question: str
    route: Literal["calc", "policy", "final"]
    tool_result: str
    steps: int
    answer: str

router_prompt = ChatPromptTemplate.from_template(
    "Clasifica la intención en UNA palabra: calc, policy o final.\n"
    "- calc: si hay una operación numérica explícita.\n"
    "- policy: si pregunta sobre normativa, porcentajes, convocatorias, asistencia.\n"
    "- final: en otros casos.\n\n"
    "Pregunta: {q}\nEtiqueta:"
)

def router(state: State) -> State:
    state["steps"] += 1
    if state["steps"] > 6:
        state["route"] = "final"
        state["tool_result"] = "max_steps alcanzado; responde de forma segura."
        return state

    label = (router_prompt | llm | StrOutputParser()).invoke({"q": state["question"]}).strip().lower()
    if label not in ("calc", "policy", "final"):
        label = "final"
    state["route"] = label
    print("[router] route:", label)
    return state

def calc_tool(state: State) -> State:
    import re
    expr = re.sub(r"[^0-9\+\-\*\/\.\(\) ]", "", state["question"])
    try:
        val = eval(expr, {"__builtins__": {}})
        state["tool_result"] = f"Resultado cálculo: {val}"
    except:
        state["tool_result"] = "No pude calcular de forma segura."
    print("[calc_tool]", state["tool_result"])
    return state

def policy_tool(state: State) -> State:
    q = state["question"].lower()
    hit = None
    for k in POLICY_DB:
        if k in q:
            hit = k
            break
    state["tool_result"] = POLICY_DB.get(hit, "No encontré esa política en el diccionario.")
    print("[policy_tool] hit:", hit)
    return state

final_prompt = ChatPromptTemplate.from_template(
    "Responde al usuario en español, claro y breve.\n"
    "Si tool_result aporta información útil, úsala.\n\n"
    "Pregunta: {q}\n"
    "tool_result: {tool}\n"
    "Respuesta:"
)

def final_answer(state: State) -> State:
    out = (final_prompt | llm | StrOutputParser()).invoke({"q": state["question"], "tool": state["tool_result"]})
    state["answer"] = out
    return state

def route_to_next(state: State):
    return state["route"]

g = StateGraph(State)
g.add_node("router", router)
g.add_node("calc", calc_tool)
g.add_node("policy", policy_tool)
g.add_node("final", final_answer)

g.set_entry_point("router")
g.add_conditional_edges("router", route_to_next, {"calc": "calc", "policy": "policy", "final": "final"})
g.add_edge("calc", "final")
g.add_edge("policy", "final")
g.add_edge("final", END)

app = g.compile()

if __name__ == "__main__":
    tests = [
        "¿Cuánto es 7*(3+2)?",
        "¿Qué es la evaluación continua?",
        "¿La asistencia es obligatoria?"
    ]
    for t in tests:
        print("\n========================")
        print("Q:", t)
        result = app.invoke({"question": t, "route": "final", "tool_result": "", "steps": 0, "answer": ""})
        print("A:", result["answer"])
