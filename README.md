# Procesamiento de Lenguaje Natural y Minería de Texto

[![SWH](https://archive.softwareheritage.org/badge/swh:1:dir:a1452b5261ebdf832c8298c4e551166d77624c60/)](https://archive.softwareheritage.org/swh:1:dir:a1452b5261ebdf832c8298c4e551166d77624c60;origin=https://github.com/madrueno/urjc-gicd-pln-mt;visit=swh:1:snp:cfb7c55f09526c8f033a1a14e3b60b6a129e4275;anchor=swh:1:rev:c782fc0130db6cef7c4ae31e7a55c07cdd970148)

Material docente en abierto de la Universidad Rey Juan Carlos para la asignatura de Procesamiento de Lenguaje Natural y Minería de Texto del Grado en Ciencia e Ingeniería de Datos.

Este repositorio incluye ejercicios prácticos y códigos de ejemplo:
- **GitHub**: [madrueno/urjc-gicd-pln-mt](https://github.com/madrueno/urjc-gicd-pln-mt)
- **Software Heritage**: [swh:1:dir:a1452b5261ebdf832c8298c4e551166d77624c60](https://archive.softwareheritage.org/swh:1:dir:a1452b5261ebdf832c8298c4e551166d77624c60)

Actualizado el 15/02/2026. Elaborado por:
- Natalia Madrueño Sierro (natalia.madrueno@urjc.es) - URJC
- Alberto Fernández Isabel (alberto.fernandez.isabel@urjc.es) - URJC
- Soto Montalvo Herranz (soto.montalvo@urjc.es) - URJC


## Índice de Temas

1. [Introducción](01-introduccion/)
2. [Análisis Textual](02-analisis-textual/)
3. [Representación del Texto](03-representacion-texto/)
4. [Redes Neuronales](04-redes-neuronales/)
5. [Modelos de Lenguaje](05-modelos-lenguaje/)
6. [Aplicaciones](06-aplicaciones/)
7. [Aumento de Datos](07-aumento-datos/)

## Requisitos Previos

- **Python 3.12+** (recomendado)
- **uv** - Gestor de dependencias

Puedes instalar uv mediante los siguientes comandos:
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternativamente con pip
pip install uv
```

## Cómo Usar este Repositorio

### Estructura del Repositorio

Cada tema tiene su propia carpeta. El primer tema proporciona un proyecto de ejemplo de procesamiento de lenguaje natural. A partir del segundo tema la estructura base tiene la siguiente forma:

```bash
XX-nombre/
├── README.md          # Descripción y contenidos del tema
├── pyproject.toml     # Dependencias específicas
├── uv.lock            # Lockfile del entorno
├── exercises/         # Enunciados de los ejercicios
├── notebooks/         # Ejercicios resueltos con notebooks
├── data/              # Datasets empleados para resolverlos
└── models/            # Modelos descargados y entrenados
```

**Nota**: El tema 06 `06-aplicaciones` incluye las soluciones teóricas en notebooks y una carpeta `scripts/` con implementaciones prácticas en Python.

### 1. Clonar el repositorio

```bash
git clone git@github.com:madrueno/urjc-gicd-pln-mt.git
cd urjc-gicd-pln-mt
```

### 2. Trabajar con un tema específico

Cada tema tiene sus propias dependencias aisladas. Hay dos formas principales de trabajar:

#### Opción A: Usando VSCode (Recomendado)

```bash
# Navegar al tema deseado
cd 03-representacion-texto

# Instalar dependencias y crear entorno virtual
uv sync

# Abrir en VSCode
code .
```

Luego, en VSCode:
1. Abre un notebook (archivo `.ipynb`)
2. VSCode detectará automáticamente el entorno `.venv/`
3. Selecciona el kernel de Python del `.venv/` si no se selecciona automáticamente
4. Comienza a trabajar directamente en VSCode

#### Opción B: Usando Jupyter Lab/Notebook (Terminal)

```bash
# Navegar al tema deseado
cd 03-representacion-texto

# Instalar dependencias y crear entorno virtual
uv sync

# Ejecutar Jupyter Lab (interfaz moderna)
uv run jupyter lab

# O Jupyter Notebook (interfaz clásica)
uv run jupyter notebook
```

### Convenciones de Nomenclatura

#### Temas
- Formato: `XX-nombre` donde XX es el número del tema (01-07)
- Ejemplo: `03-representacion-texto`

#### Ejercicios
- Formato: `XX_MM_nombre`
  - `XX` = número del tema
  - `MM` = número del notebook dentro del tema
  - `nombre` = descripción breve
- Ejemplos: `03_01_representaciones_clasicas.pdf`, `03_01_representaciones_clasicas.ipynb`


## Licencia

Este trabajo está licenciado bajo [Creative Commons Reconocimiento-CompartirIgual 4.0 Internacional](https://creativecommons.org/licenses/by-sa/4.0/).

>©2026 Natalia Madrueño Sierro, Alberto Fernández Isabel y Soto Montalvo Herranz
>Algunos derechos reservados  
>Este documento se distribuye bajo la licencia  
>“Atribución-CompartirIgual 4.0 Internacional” de Creative Commons, disponible en  
>https://creativecommons.org/licenses/by-sa/4.0/deed.es