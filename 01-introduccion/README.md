# Introducción

**Proyecto de Ejemplo - YouTube Spam Detection que presenta buenas prácticas para proyectos de investigación en ciencia de datos**

Repositorio disponible en [madrueno/data-science-project-example](https://github.com/madrueno/data-science-project-example)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-gestor%20de%20paquetes-5A67D8)](https://docs.astral.sh/uv/)
[![VSCode](https://img.shields.io/badge/VSCode-IDE-007ACC?logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/)
[![Ruff](https://img.shields.io/badge/estilo-ruff-black)](https://docs.astral.sh/ruff/)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](LICENSE)

## Indice de contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
- [Licencia](#licencia)

## Descripción

Proyecto para detectar spam en comentarios de YouTube mediante un pipeline completo de experimentación reproducible.

Trabaja con el dataset *YouTube Spam Collection*, incluyendo diversos experimentos que comparan múltiples estrategias de procesamiento de lenguaje natural y minería de texto. Mantiene una organización modular, con scripts y notebooks reproducibles y tooling moderno.


La organización de este proyecto de ejemplo está inspirada en la estructura propuesta por  [cookiecutter-data-science](https://github.com/drivendataorg/cookiecutter-data-science).

```bash
.
├── data/                           # datos organizados por etapas de transformacion 
│   ├── raw/                        # datos originales en crudo sin modificar
│   ├── interim/                    # datos con transformaciones intermedias
│   └── processed/                  # datos listos para experimentacion
├── models/                         # modelos organizados por origen
│   ├── external/                   # modelos externos preentrenados
│   └── custom/                     # modelos personalizados
├── notebooks/                      # notebooks de analisis
│   ├── eda.ipynb                   # analisis exploratorio
│   └── compare_test_results.ipynb  # comparativa de resultados
├── reports/                        # documentacion elaborada
├── results/                        # resultados de experimentos
├── scripts/                        # procesamiento y experimentacion
│   ├── processing/                 # transformaciones de datos
│   └── experiments/                # pipelines de experimentacion
├── src/spam_ham_detector/          # codigo importable
│   ├── config.py                   # configuracion general
│   ├── dataset.py                  # carga de datos
│   ├── evaluation.py               # evaluacion de modelos
│   ├── classifier/                 # clasificadores de texto
│   ├── tokenizer/                  # tokenizadores de texto
│   └── vectorizer/                 # vectorizadores de texto
├── .vscode/                        # configuracion del editor
├── pyproject.toml                  # dependencias y tooling
└── uv.lock                         # versiones fijadas
```

## Instalación

### Configuración VSCode

El proyecto está optimizado para VSCode. Al abrirlo, sugiere las siguientes extensiones:

- **Python**: soporte general de Python.
- **Pylance**: autocompletado y análisis estático.
- **Jupyter**: ejecución de notebooks.
- **Google Colab**: integración con Colab.
- **Ruff**: linting y formateo rápido.

Ruff aplica las reglas de linting y formato definidas en el proyecto sincronizándose con VSCode para mantener un estilo consistente.

### Instalación de uv

**uv** es el gestor de dependencias y entornos virtuales usado en este proyecto.

La instalación de uv depende del sistema operativo:
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternativamente con pip
pip install uv
```

Con uv instalado, `uv sync` crea el entorno virtual y sincroniza las dependencias del proyecto.

Algunos de los comandos básicos de uv son los siguientes:

```bash
uv sync             # sincronizar dependencias y crear entorno virtual
uv add <paquete>    # añadir una dependencia
uv remove <paquete> # eliminar dependencias
```

Más información sobre cómo usar uv y formas alternativas de instalarlo puede encontrarse en los siguientes enlaces:
- [Documentación oficial de uv](https://docs.astral.sh/uv/)
- [Instalación de uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Guía de comandos de uv](https://docs.astral.sh/uv/reference/cli/)
- [Gestión de proyectos con uv](https://docs.astral.sh/uv/guides/projects/)

### Descarga de datos y modelos

Para la ejecución de los experimentos, es necesario descargararse el siguiente dataset y modelo preentrenado:

**1. Descarga del dataset de YouTube Spam Collection:**
```bash
wget https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip
unzip youtube+spam+collection.zip -d data/raw/youtube-spam-collection/
```

**2. Descarga de los embeddings preentrenados de FastText:**
```bash
cd models/external && \
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip && \
  unzip wiki.simple.zip
```

## Ejecución

El proyecto incluye notebooks de análisis y exploración, así como scripts para preparar los datos y ejecutar los diferentes experimentos de clasificación de comentarios spam.

### Análisis y exploración

Los notebooks se encuentran dentro de `notebooks/` y se pueden abrir directamente en VSCode o a través de Jupyter.

En particular, se dispone de los siguientes notebooks:

- `eda.ipynb`: realiza un análisis exploratorio básico de los comentarios de Youtube.
- `compare.ipynb`: compara los resultados de los diferentes clasificadores experimentados.

### Preparación de datos

Los scripts de `scripts/processing/` preparan los datos para los posteriores experimentos. Según el nivel de procesamiento, guardan los resultados de estas transformaciones en `data/interim/` y `data/processed/`.

En particular, se disponen de los siguientes scripts de preparación:

- `extract_comments.py`: extrae los comentarios y los unifica en un único archivo (datos interim).
- `prepare_train_test.py`: prepara los conjuntos de datos para la experimentación (datos processed).

> Es importante ejecutar estos scripts antes de lanzar los experimentos descritos posteriormente.

### Experimentos

Los scripts de `scripts/experiments/` comparan distintas estrategias de procesamiento de lenguaje natural y minería de texto para la clasificación de comentarios spam en YouTube.

En particular, se disponen de los siguientes scripts de experimentación:

- `lr_tfidf_simple.py`: TF‑IDF con un preprocesamiento simple y regresión logística.
- `lr_tfidf_advanced.py`: TF‑IDF con un preprocesamiento más avanzado y regresión logística.
- `lr_ft_simple.py`: embeddings de FastText con un preprocesamiento simple y regresión logística.
- `lr_ft_advanced.py`: embeddings FastText con un preprocesamiento más avanzado y regresión logística.
- `lr_sbert_minilm.py`: embeddings de SBERT con el modelo preentrenado `all-MiniLM-L6-v2` y regresión logística.
- `lr_sbert_mpnet.py`: embeddings de SBERT con el modelo preentrenado `all-mpnet-base-v2` y regresión logística.
- `bert_distilbert_finetuned.py`: fine‑tuning de DistilBERT con caché de modelo.
- `bert_albert_finetuned.py`: fine‑tuning de ALBERT base v2 con caché de modelo.

Los resultados en **test** se guardan dentro de la carpeta `results/<nombre-del-experimento>/`. Posteriormente, se pueden comparar estos resultados en el notebook `notebooks/compare_test_results.ipynb`.

> Es importante realizar la experimentación inicial y la selección de los hiperparámetros en base a **dev**. Una vez seleccionados los mejores modelos con ese split, se debe validar su rendimiento en base a **test**. Esta separación evita sobreajuste y mantiene la evaluación imparcial.
