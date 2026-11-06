"""Pipeline principal de preparação do dataset de fine-tuning.

Etapas:
    1. Carrega e limpa as fontes habilitadas em src/config.py:
       - MedQuAD (Hugging Face) – perguntas e respostas de saúde do NIH;
       - PubMedQA (Hugging Face) – perguntas clínicas com evidência de artigos (amostra de 'pqa_artificial');
       - Base sintética do hospital – protocolos, FAQ de médicos, casos, laudos e minutas de receita.
    2. Anonimiza dados pessoais (nomes, CPF, prontuário, datas, telefones...).
    3. Faz a curadoria (normalização, filtros de tamanho, deduplicação).
    4. Converte para o formato de mensagens (system/user/assistant), separa treino/validação e exporta JSONL.
    5. Exporta o conjunto de avaliação (PubMedQA 'pqa_labeled'), o manifesto e o resumo do dataset.
    6. Valida a conformidade dos arquivos gerados.

Uso:
    python scripts/Process_dataset.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Garante que a raiz do projeto esteja no sys.path para execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.anonymizer import Anonymizer  # noqa: E402
from src.curation import DatasetCurator  # noqa: E402
from src.data_loader import MedQuadDataLoader, PubMedQADataLoader  # noqa: E402
from src.jsonl_converter import JsonlConverter  # noqa: E402
from src.synthetic_generator import SyntheticHospitalDataGenerator  # noqa: E402
from src.utils.logger import setup_logger  # noqa: E402

logger = setup_logger("pipeline_runner")

UNIFIED_COLUMNS: List[str] = ["question", "answer", "source", "source_ref", "category"]


def load_medquad() -> pd.DataFrame:
    """MedQuAD limpo, com a fonte do documento anexada à resposta (explainability)."""
    loader = MedQuadDataLoader(
        dataset_name=config.HUGGINGFACE_DATASET_NAME,
        token=config.HUGGINGFACE_API_KEY,
        split="train",
        discarded_output_path=config.DISCARDED_JSON_PATH,
        include_metadata=True,
    )
    df = loader.clean_data(save_discarded=True)

    source_col = df["document_source"] if "document_source" in df else pd.Series("", index=df.index)
    url_col = df["document_url"] if "document_url" in df else pd.Series("", index=df.index)
    refs = (source_col.fillna("").astype(str) + " " + url_col.fillna("").astype(str)).str.strip()

    if config.APPEND_SOURCE_CITATION:
        citation = refs.where(refs != "", "NIH")
        df["answer"] = df["answer"] + "\n\nSource: MedQuAD (NIH) – " + citation + "."

    df["source"] = "medquad"
    df["source_ref"] = refs
    df["category"] = df["question_type"].fillna("") if "question_type" in df else ""
    return df[UNIFIED_COLUMNS]


def load_pubmedqa_train() -> pd.DataFrame:
    """Amostra do PubMedQA 'pqa_artificial' para treino."""
    loader = PubMedQADataLoader(
        dataset_name=config.PUBMEDQA_DATASET_NAME,
        config_name=config.PUBMEDQA_TRAIN_CONFIG,
        token=config.HUGGINGFACE_API_KEY,
        sample_size=config.PUBMEDQA_TRAIN_SAMPLE_SIZE,
        seed=config.RANDOM_SEED,
        max_context_chars=config.PUBMEDQA_MAX_CONTEXT_CHARS,
        discarded_output_path=config.DISCARDED_PUBMEDQA_JSON_PATH,
    )
    df = loader.clean_data(save_discarded=True)
    df["source"] = "pubmedqa"
    df["source_ref"] = "PMID " + df["pubid"].astype(str)
    df["category"] = df["final_decision"].fillna("")
    return df[UNIFIED_COLUMNS]


def export_pubmedqa_eval(exclude_pubids: set) -> int:
    """Exporta o 'pqa_labeled' (rotulado por especialistas) como conjunto de avaliação.

    Artigos que já entraram no treino (exclude_pubids) são removidos, para evitar vazamento.

    Cada linha traz a pergunta já no formato de prompt usado no treino e a resposta esperada
    (yes/no/maybe), para medir a acurácia do modelo base e do modelo ajustado.
    """
    loader = PubMedQADataLoader(
        dataset_name=config.PUBMEDQA_DATASET_NAME,
        config_name=config.PUBMEDQA_EVAL_CONFIG,
        token=config.HUGGINGFACE_API_KEY,
        max_context_chars=config.PUBMEDQA_MAX_CONTEXT_CHARS,
    )
    df = loader.clean_data(save_discarded=False)
    df = df[~df["pubid"].isin(exclude_pubids)]
    config.EVAL_PUBMEDQA_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.EVAL_PUBMEDQA_JSONL_PATH.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "pubid": int(row["pubid"]) if pd.notna(row["pubid"]) else None,
                "system": config.SYSTEM_PROMPT,
                "user": row["question"],
                "expected_decision": row["final_decision"],
                "reference_answer": row["long_answer"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Conjunto de avaliação PubMedQA salvo em: {config.EVAL_PUBMEDQA_JSONL_PATH} ({len(df)} itens)")
    return len(df)


def build_synthetic() -> pd.DataFrame:
    """Gera a base sintética, salva a versão bruta (com PII fictícia) e anonimiza."""
    generator = SyntheticHospitalDataGenerator(seed=config.RANDOM_SEED)
    records = generator.generate(cases_per_protocol=config.SYNTHETIC_CASES_PER_PROTOCOL)

    # Versão bruta salva em data/raw/ apenas para demonstrar o antes/depois da anonimização.
    with config.SYNTHETIC_RAW_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Base sintética bruta (com dados pessoais fictícios) salva em: {config.SYNTHETIC_RAW_PATH}")

    anonymizer = Anonymizer(known_names=generator.generated_names)
    return anonymizer.anonymize_dataframe(pd.DataFrame(records)[UNIFIED_COLUMNS])


def run_pipeline() -> None:
    """Executa o pipeline completo de preparação dos dados."""
    logger.info("=== Iniciando Pipeline de Preparação para Fine-Tuning (multi-fonte) ===")

    # 1. Carregamento das fontes
    frames: List[pd.DataFrame] = []
    if config.USE_MEDQUAD:
        frames.append(load_medquad())
    if config.USE_PUBMEDQA:
        frames.append(load_pubmedqa_train())
    if config.USE_SYNTHETIC:
        frames.append(build_synthetic())
    if not frames:
        raise ValueError("Nenhuma fonte de dados habilitada em src/config.py.")

    unified = pd.concat(frames, ignore_index=True)
    logger.info(f"Registros após juntar as fontes: {len(unified)} | por fonte: {unified['source'].value_counts().to_dict()}")

    # 2. Anonimização geral (passada de segurança também nas fontes públicas)
    unified = Anonymizer().anonymize_dataframe(unified)

    # 3. Curadoria
    curator = DatasetCurator(
        min_question_chars=config.MIN_QUESTION_CHARS,
        min_answer_chars=config.MIN_ANSWER_CHARS,
        max_answer_chars=config.MAX_ANSWER_CHARS,
        discarded_output_path=config.DISCARDED_CURATION_JSON_PATH,
    )
    curated = curator.curate(unified, save_discarded=True)

    # 4. Conversão, separação e exportação
    converter = JsonlConverter(system_prompt=config.SYSTEM_PROMPT)
    records = converter.convert_dataframe(df=curated)
    for rec, (_, row) in zip(records, curated.iterrows()):
        rec["_source"] = row["source"]  # usado só para o manifesto; removido antes de exportar

    train_records, val_records = converter.split_train_val(
        records=records,
        train_ratio=config.DEFAULT_TRAIN_RATIO,
        seed=config.RANDOM_SEED,
    )

    manifest_rows: List[Dict[str, Any]] = []
    for split_name, split in (("train", train_records), ("val", val_records)):
        for idx, rec in enumerate(split):
            manifest_rows.append({"split": split_name, "line": idx + 1, "source": rec.pop("_source")})
    pd.DataFrame(manifest_rows).to_csv(config.DATASET_MANIFEST_PATH, index=False)

    converter.export_to_jsonl(records=train_records, output_path=config.TRAIN_JSONL_PATH)
    converter.export_to_jsonl(records=val_records, output_path=config.VAL_JSONL_PATH)

    # 5. Validação de Conformidade dos Arquivos Gerados
    logger.info("Validando conformidade dos arquivos gerados...")
    converter.validate_jsonl_file(file_path=config.TRAIN_JSONL_PATH)
    converter.validate_jsonl_file(file_path=config.VAL_JSONL_PATH)

    # 6. Conjunto de avaliação (uma falha aqui não invalida os arquivos de treino já gerados)
    eval_count = 0
    if config.USE_PUBMEDQA:
        train_pubids = {
            int(ref.replace("PMID ", ""))
            for ref in unified.loc[unified["source"] == "pubmedqa", "source_ref"]
            if str(ref).replace("PMID ", "").isdigit()
        }
        try:
            eval_count = export_pubmedqa_eval(exclude_pubids=train_pubids)
        except RuntimeError as exc:
            logger.warning(
                f"Conjunto de avaliação NÃO gerado ({exc}). Os arquivos de treino e validação estão OK. "
                "Para gerar só a avaliação depois: python scripts/Process_dataset.py --only-eval"
            )

    # 7. Resumo
    manifest = pd.DataFrame(manifest_rows)
    summary = {
        "total_curated": len(curated),
        "train": len(train_records),
        "val": len(val_records),
        "eval_pubmedqa_labeled": eval_count,
        "by_source": curated["source"].value_counts().to_dict(),
        "by_split_and_source": manifest.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict(orient="index"),
    }
    config.DATASET_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Resumo do dataset salvo em: {config.DATASET_SUMMARY_PATH}")

    logger.info("=== Pipeline concluído com sucesso! ===")
    logger.info(f"Treino salvo em: {config.TRAIN_JSONL_PATH} ({len(train_records)} amostras)")
    logger.info(f"Validação salva em: {config.VAL_JSONL_PATH} ({len(val_records)} amostras)")
    logger.info(f"Resumo por fonte: {summary['by_source']}")


def run_only_eval() -> None:
    """Gera apenas o conjunto de avaliação, sem refazer o treino."""
    train_pubids: set = set()
    for jsonl_path in (config.TRAIN_JSONL_PATH, config.VAL_JSONL_PATH):
        if not jsonl_path.exists():
            continue
        for line in jsonl_path.open(encoding="utf-8"):
            content = json.loads(line)["messages"][2]["content"]
            if "Source: PubMed PMID " in content:
                pmid = content.rsplit("Source: PubMed PMID ", 1)[1].rstrip(". \n")
                if pmid.isdigit():
                    train_pubids.add(int(pmid))
    export_pubmedqa_eval(exclude_pubids=train_pubids)


if __name__ == "__main__":
    if "--only-eval" in sys.argv:
        run_only_eval()
    else:
        run_pipeline()
