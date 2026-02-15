"""Project paths configuration."""

from pathlib import Path


# Find project root (where pyproject.toml is located)
PROJECT_ROOT = Path(__file__).parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Models directories
MODELS_DIR = PROJECT_ROOT / 'models'
EXTERNAL_MODELS_DIR = MODELS_DIR / 'external'
CUSTOM_MODELS_DIR = MODELS_DIR / 'custom'

# Results directory
RESULTS_DIR = PROJECT_ROOT / 'results'
