"""Módulo de configurações e constantes globais do projeto."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Diretórios base
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
LOGS_DIR: Path = BASE_DIR / "logs"

# Garante que os diretórios necessários existam
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Carrega variáveis do arquivo .env
ENV_FILE_PATH: Path = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE_PATH)

# Configurações de logging
LOG_FILE_PATH: Path = LOGS_DIR / "execucao.log"

# Configurações do Hugging Face
HUGGINGFACE_DATASET_NAME: str = "lavita/MedQuAD"
HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

# Caminhos dos arquivos de saída de fine-tuning
TRAIN_JSONL_PATH: Path = PROCESSED_DATA_DIR / "finetuning_train.jsonl"
VAL_JSONL_PATH: Path = PROCESSED_DATA_DIR / "finetuning_val.jsonl"
DISCARDED_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded.json"

# Prompt do Sistema para o fine-tuning biomédico
SYSTEM_PROMPT: str = (
    "You are a biomedical evidence assistant for healthcare professionals. "
    "Answer using the supplied scientific evidence. Distinguish findings from interpretation, "
    "state limitations, and do not invent evidence. You are not intended to diagnose, "
    "but to give assistance. The final word is always from a professional"
)

# Parâmetros de separação de dataset
DEFAULT_TRAIN_RATIO: float = 0.9
RANDOM_SEED: int = 42
