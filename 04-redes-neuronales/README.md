# Tema 4: Redes Neuronales

Ejercicios de redes neuronales aplicadas a clasificación de texto, abarcando redes feedforward y redes recurrentes.

- [Ejercicios](#ejercicios)
- [Entorno](#entorno)
- [Datasets](#datasets)
- [Modelos](#modelos)

## Ejercicios

Los enunciados de los ejercicios están en `exercises/`, y sus soluciones en `notebooks/`:

- **04_01_redes_neuronales**: Implementación de tres arquitecturas de redes neuronales para clasificación de texto: MLP simple para análisis de sentimiento y RNN y LSTM para detección de noticias falsas. Incluye entrenamiento con embeddings desde cero y con Word2Vec pre-entrenado.


## Entorno


### Instalación

```bash
# Navegar al directorio
cd 04-redes-neuronales

# Instalar dependencias
uv sync
```

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

- **`train.xlsx`** - Corpus de Fake News en español obtenido de https://github.com/jpposadas/FakeNewsCorpusSpanish. Contiene títulos (Headline), contenido (Text) y categoría (True/Fake). Debe descargarse y colocarse en la carpeta `data/`.

## Modelos

Todos los comandos deben ejecutarse desde la carpeta `04-redes-neuronales` y guardarse dentro de la carpeta `models/`:

**Word Embeddings para Español - Kaggle**

Descarga manual
1. Descarga el dataset desde Kaggle: https://www.kaggle.com/datasets/rtatman/pretrained-word-vectors-for-spanish
2. Extrae el archivo `SBW-vectors-300-min5.txt` en la carpeta `models/`
