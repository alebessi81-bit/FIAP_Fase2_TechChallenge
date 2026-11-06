"""Módulo de curadoria do dataset unificado: normalização, deduplicação e filtros de qualidade."""

import logging
from pathlib import Path
import re
from typing import Optional

import pandas as pd

from src.data_loader import BaseQADataLoader
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("curation")

_MULTI_SPACES = re.compile(r"[ \t]+")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Colapsa espaços repetidos e linhas em branco em excesso, preservando quebras de linha."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NEWLINES.sub("\n\n", text).strip()


class DatasetCurator:
    """Aplica as regras de curadoria e guarda o motivo de cada descarte para auditoria."""

    def __init__(
        self,
        min_question_chars: int = 10,
        min_answer_chars: int = 20,
        max_answer_chars: int = 6000,
        discarded_output_path: Optional[Path] = None,
    ) -> None:
        self.min_question_chars = min_question_chars
        self.min_answer_chars = min_answer_chars
        self.max_answer_chars = max_answer_chars
        self.discarded_output_path = discarded_output_path

    def curate(self, df: pd.DataFrame, save_discarded: bool = True) -> pd.DataFrame:
        """Normaliza, filtra por tamanho e remove duplicatas exatas (pergunta + resposta).

        Duplicatas só de pergunta são mantidas: no MedQuAD a mesma pergunta genérica
        ("What is (are) X?") aparece com respostas diferentes de documentos diferentes.
        """
        work = df.copy()
        work["question"] = work["question"].astype(str).map(normalize_text)
        work["answer"] = work["answer"].astype(str).map(normalize_text)

        reason = pd.Series("", index=work.index, dtype=object)
        reason[work["question"].str.len() < self.min_question_chars] = "pergunta_curta"
        reason[(reason == "") & (work["answer"].str.len() < self.min_answer_chars)] = "resposta_curta"
        reason[(reason == "") & (work["answer"].str.len() > self.max_answer_chars)] = "resposta_longa"

        key = work["question"].str.lower() + "␟" + work["answer"].str.lower()
        is_dup = key.duplicated(keep="first")
        reason[(reason == "") & is_dup] = "duplicata"

        keep_mask = reason == ""
        curated = work[keep_mask].reset_index(drop=True)
        discarded = work[~keep_mask].assign(motivo_descarte=reason[~keep_mask])

        counts = discarded["motivo_descarte"].value_counts().to_dict()
        logger.info(
            f"Curadoria concluída: {len(curated)} mantidos, {len(discarded)} descartados. Motivos: {counts or 'nenhum'}"
        )
        if save_discarded and self.discarded_output_path is not None:
            BaseQADataLoader._save_discarded_records(discarded, self.discarded_output_path)
        return curated
