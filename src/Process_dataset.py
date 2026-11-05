"""Pipeline principal para processamento do MedQuAD via Hugging Face e geração dos arquivos JSONL."""

import sys
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path para execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    DEFAULT_TRAIN_RATIO,
    DISCARDED_JSON_PATH,
    HUGGINGFACE_API_KEY,
    HUGGINGFACE_DATASET_NAME,
    RANDOM_SEED,
    SYSTEM_PROMPT,
    TRAIN_JSONL_PATH,
    VAL_JSONL_PATH,
)
from src.data_loader import MedQuadDataLoader
from src.jsonl_converter import JsonlConverter
from src.utils.logger import setup_logger

logger = setup_logger("pipeline_runner")


def run_pipeline() -> None:
    """Executa o pipeline completo: download do Hugging Face, limpeza, descarte, conversão, split e validação."""
    logger.info("=== Iniciando Pipeline de Preparação para Fine-Tuning (Hugging Face) ===")

    # 1. Carregamento, Limpeza e Gravação dos descartes a partir do Hugging Face
    loader = MedQuadDataLoader(
        dataset_name=HUGGINGFACE_DATASET_NAME,
        token=HUGGINGFACE_API_KEY,
        split="train",
        discarded_output_path=DISCARDED_JSON_PATH,
    )
    cleaned_df = loader.clean_data(save_discarded=True)

    # 2. Conversão e Estruturação ChatML
    converter = JsonlConverter(system_prompt=SYSTEM_PROMPT)
    records = converter.convert_dataframe(df=cleaned_df)

    # 3. Separação em Treino e Validação
    train_records, val_records = converter.split_train_val(
        records=records,
        train_ratio=DEFAULT_TRAIN_RATIO,
        seed=RANDOM_SEED,
    )

    # 4. Exportação dos Arquivos JSONL
    converter.export_to_jsonl(records=train_records, output_path=TRAIN_JSONL_PATH)
    converter.export_to_jsonl(records=val_records, output_path=VAL_JSONL_PATH)

    # 5. Validação de Conformidade dos Arquivos Gerados
    logger.info("Validando conformidade dos arquivos gerados...")
    converter.validate_jsonl_file(file_path=TRAIN_JSONL_PATH)
    converter.validate_jsonl_file(file_path=VAL_JSONL_PATH)

    logger.info("=== Pipeline concluído com sucesso! ===")
    logger.info(f"Treino salvo em: {TRAIN_JSONL_PATH} ({len(train_records)} amostras)")
    logger.info(f"Validação salva em: {VAL_JSONL_PATH} ({len(val_records)} amostras)")
    logger.info(f"Descartes salvos em: {DISCARDED_JSON_PATH}")


if __name__ == "__main__":
    run_pipeline()
