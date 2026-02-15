# Tema 6: Aplicaciones

Ejercicios de aplicaciones de modelos de lenguaje que cubren prompt engineering, Retrieval-Augmented Generation (RAG) y agentes LLM con herramientas y skills.

- [Ejercicios](#ejercicios)
- [Entorno](#entorno)
- [Datasets](#datasets)

## Ejercicios

Los enunciados de los ejercicios están en `exercises/`, las soluciones teóricas en `notebooks/` y las soluciones prácticas en `scripts/`:

- **06_01_prompt_engineering**: Ingeniería de prompts (zero-shot, one-shot, few-shot, chain-of-thought, prompt mining, control de formato y mitigación de alucinaciones).

- **06_02_rag**: RAG (pipeline y puntos de fallo, RAG vs fine-tuning vs prompting, evaluación). Soluciones prácticas: `06_02_rag_basico.py`, `06_02_rag_retrieval.py`, `06_02_rag_conversacional.py`.

- **06_03_agentes**: Agentes LLM (tool vs skill vs agent, control de loops y alucinaciones, RAG como tool). Soluciones prácticas: `06_03_flow_lineal.py`, `06_03_agent_tools.py`, `06_03_agent_rag_tool.py`, `06_03_agent_skills_tools.py`.

## Entorno

### Instalación

```bash
# Navegar al directorio
cd 06-aplicaciones

# Instalar dependencias
uv sync
```

### Configuración del LLM

Los scripts prácticos requieren un LLM. Hay dos opciones:

**Opción A: OpenAI (requiere API key)**

Crea un archivo `.env` en la raíz del proyecto con tu clave:

```
OPENAI_API_KEY=sk-...
```

**Opción B: Ollama (local, sin API key)**

Instala [Ollama](https://ollama.com/) y descarga un modelo:

```bash
ollama pull llama3.2
```

Después, en cada script, comenta la línea de OpenAI y descomenta la de Ollama.

### Ejecución

**VSCode**

Abre el directorio y selecciona el kernel del entorno virtual:

```bash
code .
```

**Jupyter Lab**
```bash
uv run jupyter lab
```

## Datasets

Corpus de normativa universitaria utilizado por los scripts de RAG y agentes (ya incluido en `data/`):

- **`guia_docente_resumen.txt`** - Resumen de la guía docente de la asignatura.
- **`normativa_asistencia.txt`** - Normativa sobre asistencia a clase.
- **`normativa_evaluacion.txt`** - Normativa de evaluación continua y calificaciones.
- **`normativa_examenes.txt`** - Normativa de exámenes y convocatorias.
