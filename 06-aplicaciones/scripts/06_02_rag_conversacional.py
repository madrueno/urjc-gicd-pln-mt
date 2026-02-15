from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# 0) ENV (OpenAI opcional)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# =========================
# 1) LLM (elige uno)
# =========================
USE_OPENAI = True

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)

# =========================
# 2) Vector store (Chroma)
# =========================
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION = "pln_rag_demo"

docs = []
for fp in DATA_DIR.glob("*.txt"):
    loaded = TextLoader(str(fp), encoding="utf-8").load()
    for d in loaded:
        d.metadata["source"] = fp.name
    docs.extend(loaded)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
splits = splitter.split_documents(docs)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vs = Chroma.from_documents(
    documents=splits,
    embedding=emb,
    persist_directory=str(PERSIST_DIR),
    collection_name=COLLECTION,
)

# =========================
# 3) Prompt de respuesta (grounded + fuentes)
# =========================
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente cuidadoso. Responde SOLO usando el contexto proporcionado. "
     "Si el contexto es insuficiente, responde exactamente: "
     "'No puedo responder con la información disponible.' "
     "Siempre termina con una sección final 'Fuentes:' listando las fuentes únicas "
     "usadas (metadata 'source')."),
    ("human", "Pregunta: {question}\n\nContexto:\n{context}")
])

def format_docs_with_sources(retrieved_docs):
    context = "\n\n".join(d.page_content for d in retrieved_docs)

    sources = []
    for d in retrieved_docs:
        src = d.metadata.get("source", "desconocido")
        if src not in sources:
            sources.append(src)

    sources_text = "\n".join(f"- {s}" for s in sources)
    return context, sources_text

# =========================
# 4) Query rewriting (con historial)
# =========================
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Reescribe la pregunta del usuario como una consulta de búsqueda concisa. "
     "Usa el historial si ayuda. Devuelve SOLO la consulta, sin comillas."
     "Si el contexto contiene afirmaciones contradictorias, indícalo y explica ambas."),
    ("human", "Historial:\n{history}\n\nPregunta: {question}\n\nConsulta de búsqueda:")
])

history = []

def history_to_text(hist):
    parts = []
    for m in hist:
        role = "Usuario" if isinstance(m, HumanMessage) else "Asistente"
        parts.append(f"{role}: {m.content}")
    return "\n".join(parts)[-2000:]  # truncado simple

def rewrite_query(user_question: str) -> str:
    q = (rewrite_prompt | llm | StrOutputParser()).invoke(
        {"history": history_to_text(history), "question": user_question}
    ).strip()
    return q

# =========================
# 5) Retrievers (baseline / MMR) + filtro opcional
# =========================
def make_retriever(*, k=4, use_mmr=False):
    if use_mmr:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
        )
    return vs.as_retriever(search_kwargs={"k": k})

# =========================
# 6) RAG conversacional
# =========================
def ask(
    user_question: str,
    *,
    k: int = 4,
    use_mmr: bool = False,
    filter_dict: dict | None = None,
    show_debug: bool = True,
) -> str:
    # (1) Reescribir consulta con historial
    query = rewrite_query(user_question)

    if show_debug:
        print("🔎 Query usada:", query)

    # (2) Recuperación
    retriever = make_retriever(k=k, use_mmr=use_mmr)
    # API nueva: invoke(). Para filtros por metadata, se pasa como kwarg.
    if filter_dict:
        retrieved_docs = retriever.invoke(query, filter=filter_dict)
    else:
        retrieved_docs = retriever.invoke(query)

    # (3) Contexto + fuentes
    context, sources_text = format_docs_with_sources(retrieved_docs)

    # (4) Generación grounded
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {"question": user_question, "context": context}
    )

    # Robustez: asegurar Fuentes
    if "Fuentes:" not in answer:
        answer = answer.rstrip() + "\n\nFuentes:\n" + sources_text

    # (5) Actualizar historial
    history.append(HumanMessage(content=user_question))
    history.append(AIMessage(content=answer))

    return answer

# =========================
# 7) Demo
# =========================
print(ask("¿Cuáles son los porcentajes de evaluación?", k=4, use_mmr=False))
print()
print(ask("¿Y cómo afecta eso a la segunda convocatoria?", k=4, use_mmr=True))
print()
print(ask("¿Es obligatorio aprobar el proyecto final?", k=4, use_mmr=True))
