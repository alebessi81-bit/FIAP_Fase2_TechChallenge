"""Módulo responsável pelo carregamento e higienização do dataset MedQuAD via Hugging Face."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
import pandas as pd

from src.config import DISCARDED_JSON_PATH
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("data_loader")


class MedQuadDataLoader:
    """Carregador e validador do dataset MedQuAD a partir do Hugging Face.

    Esta classe encapsula a conexão com a API do Hugging Face (utilizando token de autenticação),
    a extração das colunas essenciais ('question' e 'answer'), o descarte de registros inválidos
    e o salvamento dos descartes para fins de auditoria/validação.
    """

    REQUIRED_COLUMNS: List[str] = ["question", "answer"]

    def __init__(
        self,
        dataset_name: str = "lavita/MedQuAD",
        token: Optional[str] = None,
        split: str = "train",
        discarded_output_path: Optional[Path] = None,
    ) -> None:
        """Inicializa o carregador com as configurações de acesso ao Hugging Face.

        Args:
            dataset_name: Identificador do repositório no Hugging Face (ex: 'lavita/MedQuAD').
            token: Chave de API / Token de acesso pessoal do Hugging Face.
            split: Divisão do dataset a ser carregada (padrão: 'train').
            discarded_output_path: Caminho para salvar os registros descartados (JSON).
        """
        self.dataset_name: str = dataset_name
        self.token: Optional[str] = token
        self.split: str = split
        self.discarded_output_path: Optional[Path] = (
            discarded_output_path if discarded_output_path is not None else DISCARDED_JSON_PATH
        )

    def load_raw_data(self) -> pd.DataFrame:
        """Conecta ao Hugging Face, baixa e carrega os registros do dataset.

        Returns:
            pd.DataFrame: DataFrame contendo as colunas brutas essenciais ('question', 'answer').

        Raises:
            ValueError: Se o dataset carregado estiver vazio ou não contiver as colunas esperadas.
            RuntimeError: Em caso de falha de autenticação ou rede com a API do Hugging Face.
        """
        logger.info(f"Conectando ao Hugging Face para baixar o dataset '{self.dataset_name}' (split='{self.split}')...")

        try:
            hf_dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                token=self.token,
            )
        except Exception as exc:
            error_msg = f"Falha ao carregar o dataset '{self.dataset_name}' do Hugging Face: {exc}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from exc

        if len(hf_dataset) == 0:
            error_msg = f"O dataset '{self.dataset_name}' retornado está vazio."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Colunas disponíveis no dataset remoto: {hf_dataset.column_names}")

        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in hf_dataset.column_names]
        if missing_cols:
            error_msg = f"Colunas obrigatórias ausentes no dataset do Hugging Face: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Extraindo colunas necessárias: {self.REQUIRED_COLUMNS}")
        raw_df = pd.DataFrame(
            {
                "question": hf_dataset["question"],
                "answer": hf_dataset["answer"],
            }
        )

        logger.info(f"Total de registros brutos obtidos: {len(raw_df)}")
        return raw_df

    def clean_data(
        self,
        df: Optional[pd.DataFrame] = None,
        save_discarded: bool = True,
    ) -> pd.DataFrame:
        """Executa a limpeza e sanitização dos dados.

        Remove registros nulos, vazios ou compostos exclusivamente por espaços em branco
        nas colunas 'question' e 'answer'. Registros descartados são salvos em arquivo JSON
        caso save_discarded seja True.

        Args:
            df: DataFrame a ser limpo. Se None, busca automaticamente via load_raw_data().
            save_discarded: Se True, grava os registros descartados no caminho configurado.

        Returns:
            pd.DataFrame: DataFrame contendo apenas registros válidos e sanitizados.
        """
        if df is None:
            df = self.load_raw_data()

        # Identifica linhas válidas: valores não nulos e não vazios após strip
        is_not_null = df["question"].notna() & df["answer"].notna()
        q_clean = df["question"].fillna("").astype(str).str.strip()
        a_clean = df["answer"].fillna("").astype(str).str.strip()

        valid_mask = is_not_null & (q_clean != "") & (a_clean != "")

        cleaned_df = df[valid_mask].copy()
        cleaned_df["question"] = q_clean[valid_mask]
        cleaned_df["answer"] = a_clean[valid_mask]

        discarded_df = df[~valid_mask].copy()
        dropped_count = len(discarded_df)

        logger.info(
            f"Limpeza concluída: {len(cleaned_df)} registros válidos "
            f"({dropped_count} descartados por inconsistências/nulos)."
        )

        if save_discarded and self.discarded_output_path is not None:
            self._save_discarded_records(discarded_df, self.discarded_output_path)

        return cleaned_df

    @staticmethod
    def _save_discarded_records(discarded_df: pd.DataFrame, output_path: Path) -> None:
        """Grava os registros descartados em formato JSON convencional para auditoria.

        Trata valores NaN/None convertendo-os explicitamente para null em JSON.

        Args:
            discarded_df: DataFrame contendo as linhas descartadas.
            output_path: Caminho do arquivo JSON de destino.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Gravando {len(discarded_df)} registros descartados em: {output_path}")

        records: List[Dict[str, Any]] = [
            {k: (None if pd.isna(v) else v) for k, v in row.items()}
            for row in discarded_df.to_dict(orient="records")
        ]

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info(f"Arquivo de registros descartados salvo com sucesso: {output_path}")
