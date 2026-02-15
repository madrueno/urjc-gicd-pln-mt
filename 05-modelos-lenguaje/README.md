# Tema 5: Modelos de Lenguaje

Ejercicios de modelos de lenguaje basados en Transformers, abarcando desde la construcción de arquitecturas encoder y decoder desde cero con Keras hasta fine-tuning de modelos preentrenados (BERT) y post-training para alineamiento.

- [Ejercicios](#ejercicios)
- [Entorno](#entorno)
- [Datasets](#datasets)

## Ejercicios

Los enunciados de los ejercicios están en `exercises/`, y sus soluciones en `notebooks/`:

- **05_01_transformers**: Construcción de Transformers desde cero (encoder, decoder, encoder-decoder para clasificación y traducción).

- **05_02_fine_tuning**: Fine-tuning de modelos preentrenados (BERT, DistilBERT, ALBERT, XLM-RoBERTa) para clasificación de sentimiento.

- **05_03_post_training**: Post-training y alineamiento de LLMs (SFT, RLHF, DPO, reward hacking, sobrealineación).



## Entorno


### Instalación

```bash
# Navegar al directorio
cd 05-modelos-lenguaje

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

- **`train.xlsx`** - Corpus de Fake News en español obtenido de https://github.com/jpposadas/FakeNewsCorpusSpanish. Debe descargarse y colocarse en la carpeta `data/`.
