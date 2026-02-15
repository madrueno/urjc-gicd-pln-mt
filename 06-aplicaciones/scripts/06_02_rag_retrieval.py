from pathlib import Path
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# 0) ENV (solo si usas OpenAI)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# =========================
# 1) LLM (elige una opción)
# =========================
USE_OPENAI = True

if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    # Requiere OPENAI_API_KEY en entorno/.env
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)

# =========================
# 2) Datos + vectorstore
# =========================
DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION = "pln_rag_demo"

docs = []
for fp in DATA_DIR.glob("*.txt"):
    loaded = TextLoader(str(fp), encoding="utf-8").load()
    for d in loaded:
        d.metadata["source"] = fp.name  # <- clave para filtros y "Fuentes"
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
# 3) Prompt "grounded" (como tu ejercicio)
# =========================
prompt = ChatPromptTemplate.from_messages([
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
    return context, sources_text, sources

# =========================
# 4) "RAG runner" (API nueva: retriever.invoke)
# =========================
def rag_with_retriever(question: str, retriever, *, filter_dict=None) -> str:
    """
    - retriever: objeto devuelto por vs.as_retriever(...)
    - filter_dict: dict para filtrar por metadata (Chroma), p.ej. {"source": "normativa_evaluacion.txt"}
    """
    # En LangChain actual, el retriever se invoca con .invoke(...)
    # y algunos backends aceptan filtros como kwargs: retriever.invoke(query, filter={...})
    if filter_dict:
        retrieved_docs = retriever.invoke(question, filter=filter_dict)
    else:
        retrieved_docs = retriever.invoke(question)

    context, sources_text, sources = format_docs_with_sources(retrieved_docs)

    answer = (prompt | llm | StrOutputParser()).invoke(
        {"question": question, "context": context}
    )

    # Robustez: si el LLM no pone "Fuentes", lo añadimos nosotros
    if "Fuentes:" not in answer:
        answer = answer.rstrip() + "\n\nFuentes:\n" + sources_text

    return answer

# =========================
# 5) Retrievers para comparar
# =========================
# A) Baseline (similarity) con k configurable
def make_retriever_k(k: int):
    return vs.as_retriever(search_kwargs={"k": k})

# B) MMR (diversidad). Parámetros típicos de LangChain. :contentReference[oaicite:1]{index=1}
retriever_mmr = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
)

# C) Baseline + filtro por metadata (se pasa en invoke)
retriever_filtered_base = vs.as_retriever(search_kwargs={"k": 4})

# =========================
# 6) Demo / comparación
# =========================
question = "Resume la política de evaluación continua."

print("\n=== Baseline: comparar k=2 vs k=5 ===")
for k in [2, 5]:
    retr = make_retriever_k(k)
    out = rag_with_retriever(question, retr)
    print(f"\n--- k={k} ---")
    print(out)

print("\n=== MMR (más diversidad) ===")
out_mmr = rag_with_retriever(question, retriever_mmr)
print(out_mmr)

print("\n=== Filtro por metadata (solo normativa_evaluacion.txt) ===")
out_filter = rag_with_retriever(
    question,
    retriever_filtered_base,
    filter_dict={"source": "normativa_evaluacion.txt"}
)
print(out_filter)
