"""Módulo de configurações e constantes globais do projeto."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Diretórios base
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
LOGS_DIR: Path = BASE_DIR / "logs"

# Garante que os diretórios necessários existam
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Carrega variáveis do arquivo .env
ENV_FILE_PATH: Path = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE_PATH)

# Configurações de logging
LOG_FILE_PATH: Path = LOGS_DIR / "execucao.log"

# Configurações do Hugging Face
HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

# --- Fonte 1: MedQuAD (perguntas e respostas de saúde do NIH) ---
HUGGINGFACE_DATASET_NAME: str = "lavita/MedQuAD"
USE_MEDQUAD: bool = True

# --- Fonte 2: PubMedQA (perguntas clínicas com evidência de artigos PubMed) ---
# Sugerido no enunciado do Tech Challenge Fase 3.
PUBMEDQA_DATASET_NAME: str = "qiaojin/PubMedQA"
USE_PUBMEDQA: bool = True
# 'pqa_artificial' (~211 mil, gerado automaticamente) alimenta o treino por amostragem.
PUBMEDQA_TRAIN_CONFIG: str = "pqa_artificial"
PUBMEDQA_TRAIN_SAMPLE_SIZE: int = int(os.getenv("PUBMEDQA_TRAIN_SAMPLE_SIZE", "10000"))
# 'pqa_labeled' (~1 mil, rotulado por especialistas) fica reservado para avaliação.
PUBMEDQA_EVAL_CONFIG: str = "pqa_labeled"
# Limite de caracteres do contexto (abstract) incluído na pergunta, para controlar o tamanho do prompt.
PUBMEDQA_MAX_CONTEXT_CHARS: int = 3000

# --- Fonte 3: Base sintética do hospital (protocolos, FAQ de médicos, laudos, receitas) ---
USE_SYNTHETIC: bool = True
SYNTHETIC_CASES_PER_PROTOCOL: int = int(os.getenv("SYNTHETIC_CASES_PER_PROTOCOL", "120"))
SYNTHETIC_RAW_PATH: Path = RAW_DATA_DIR / "synthetic_hospital_raw.jsonl"

# Caminhos dos arquivos de saída de fine-tuning
TRAIN_JSONL_PATH: Path = PROCESSED_DATA_DIR / "finetuning_train.jsonl"
VAL_JSONL_PATH: Path = PROCESSED_DATA_DIR / "finetuning_val.jsonl"
EVAL_PUBMEDQA_JSONL_PATH: Path = PROCESSED_DATA_DIR / "eval_pubmedqa_labeled.jsonl"
DATASET_SUMMARY_PATH: Path = PROCESSED_DATA_DIR / "dataset_summary.json"
DATASET_MANIFEST_PATH: Path = PROCESSED_DATA_DIR / "dataset_manifest.csv"

# Arquivos de descarte (auditoria)
DISCARDED_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded.json"  # MedQuAD (nulos/vazios)
DISCARDED_PUBMEDQA_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded_pubmedqa.json"
DISCARDED_CURATION_JSON_PATH: Path = PROCESSED_DATA_DIR / "discarded_curation.json"

# Curadoria: filtros de tamanho (em caracteres)
MIN_QUESTION_CHARS: int = 10
MIN_ANSWER_CHARS: int = 20
MAX_ANSWER_CHARS: int = 6000
# Acrescenta a fonte ao final de cada resposta, para o modelo aprender a citar (explainability).
APPEND_SOURCE_CITATION: bool = True

# Prompt do Sistema para o fine-tuning biomédico
SYSTEM_PROMPT: str = (
    "You are a biomedical evidence assistant for healthcare professionals. "
    "Answer using the supplied scientific evidence and the hospital's internal protocols. "
    "Distinguish findings from interpretation, state limitations, and do not invent evidence. "
    "Always cite the source of the information used. "
    "Answer in the same language as the question. "
    "You are not intended to diagnose or prescribe: any treatment or prescription you mention "
    "is a suggestion that must be validated by the responsible physician. "
    "The final word is always from a professional."
)

# Parâmetros de separação de dataset
DEFAULT_TRAIN_RATIO: float = 0.9
RANDOM_SEED: int = 42
