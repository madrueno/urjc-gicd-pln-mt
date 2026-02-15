# Tema 3: Representación del Texto

Ejercicios de representación del texto, abarcando desde métodos clásicos hasta embeddings estáticos y contextuales.

- [Ejercicios](#ejercicios)
- [Entorno](#entorno)
- [Datasets](#datasets)
- [Modelos](#modelos)

## Ejercicios

Los enunciados de los ejercicios están en `exercises/`, y sus soluciones en `notebooks/`, con el mismo nombre base:

- **03_01_representaciones_claicas**: Representación clásica basada en bolsa de palabras, matrices de coocurrencias y modelos de temas latentes.

- **03_02_word_embeddings**: Embeddings estáticos y contextuales preentrenados de diferentes fuentes. Analogías semánticas, visualización con PCA y similitudes entre frases.


## Entorno


### Instalación

```bash
# Navegar al directorio
cd 03-representacion-texto

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

Los datasets requeridos se encuentran incluidos en la carpeta `data/`:

- **`Noticias.json`** - Artículos de noticias en español para modelado de tópicos y coocurrencias.

## Modelos

Todos los comandos deben ejecutarse desde la carpeta `03-representacion-texto` y guardarse dentro de la carpeta `models/`:


**FastText - Wiki Simple**

Descarga por terminal
```bash
cd models && \
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip && \
  unzip wiki.simple.zip
```

Descarga manual
1. Descarga el archivo desde tu navegador: https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip
2. Extrae el archivo `wiki.simple.bin` en la carpeta `models/`
