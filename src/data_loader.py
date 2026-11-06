"""Módulo responsável pelo carregamento e higienização dos datasets via Hugging Face.

Fontes suportadas:
    - MedQuAD (lavita/MedQuAD): perguntas e respostas de saúde do NIH.
    - PubMedQA (qiaojin/PubMedQA): perguntas clínicas com evidência de abstracts do PubMed.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from datasets import load_dataset
import pandas as pd

from src.config import DISCARDED_JSON_PATH, DISCARDED_PUBMEDQA_JSON_PATH
from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("data_loader")


class BaseQADataLoader:
    """Base comum de limpeza e auditoria para carregadores de pares pergunta/resposta."""

    REQUIRED_COLUMNS: List[str] = ["question", "answer"]

    def __init__(self, discarded_output_path: Optional[Path] = None) -> None:
        self.discarded_output_path: Optional[Path] = discarded_output_path

    def load_raw_data(self) -> pd.DataFrame:  # pragma: no cover - implementado nas subclasses
        raise NotImplementedError

    def clean_data(
        self,
        df: Optional[pd.DataFrame] = None,
        save_discarded: bool = True,
    ) -> pd.DataFrame:
        """Executa a limpeza e sanitização dos dados.

        Remove registros nulos, vazios ou compostos exclusivamente por espaços em branco
        nas colunas 'question' e 'answer'. Registros descartados são salvos em arquivo JSON
        caso save_discarded seja True. Demais colunas (metadados) são preservadas.

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

        def _to_json_value(value: Any) -> Any:
            if isinstance(value, (list, dict)):
                return value
            try:
                return None if pd.isna(value) else value
            except (TypeError, ValueError):
                return str(value)

        records: List[Dict[str, Any]] = [
            {k: _to_json_value(v) for k, v in row.items()}
            for row in discarded_df.to_dict(orient="records")
        ]

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Arquivo de registros descartados salvo com sucesso: {output_path}")


class MedQuadDataLoader(BaseQADataLoader):
    """Carregador e validador do dataset MedQuAD a partir do Hugging Face.

    Esta classe encapsula a conexão com a API do Hugging Face (utilizando token de autenticação),
    a extração das colunas essenciais ('question' e 'answer'), o descarte de registros inválidos
    e o salvamento dos descartes para fins de auditoria/validação.

    Observação: das 12 fontes do MedQuAD, 3 (MedlinePlus de medicamentos, ervas/suplementos e
    A.D.A.M.) tiveram as respostas removidas pelos autores por direitos autorais. Por isso cerca
    de 31 mil dos ~47 mil registros vêm sem resposta e são descartados na limpeza.
    """

    # Colunas de metadados usadas para rastrear a fonte de cada resposta (explainability).
    METADATA_COLUMNS: List[str] = ["document_source", "document_url", "question_type", "question_focus"]

    def __init__(
        self,
        dataset_name: str = "lavita/MedQuAD",
        token: Optional[str] = None,
        split: str = "train",
        discarded_output_path: Optional[Path] = None,
        include_metadata: bool = False,
    ) -> None:
        """Inicializa o carregador com as configurações de acesso ao Hugging Face.

        Args:
            dataset_name: Identificador do repositório no Hugging Face (ex: 'lavita/MedQuAD').
            token: Chave de API / Token de acesso pessoal do Hugging Face.
            split: Divisão do dataset a ser carregada (padrão: 'train').
            discarded_output_path: Caminho para salvar os registros descartados (JSON).
            include_metadata: Se True, inclui também as colunas de METADATA_COLUMNS disponíveis.
        """
        super().__init__(
            discarded_output_path if discarded_output_path is not None else DISCARDED_JSON_PATH
        )
        self.dataset_name: str = dataset_name
        self.token: Optional[str] = token
        self.split: str = split
        self.include_metadata: bool = include_metadata

    def load_raw_data(self) -> pd.DataFrame:
        """Conecta ao Hugging Face, baixa e carrega os registros do dataset.

        Returns:
            pd.DataFrame: DataFrame com as colunas 'question' e 'answer' (e metadados, se pedidos).

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

        columns = list(self.REQUIRED_COLUMNS)
        if self.include_metadata:
            columns += [c for c in self.METADATA_COLUMNS if c in hf_dataset.column_names]

        logger.info(f"Extraindo colunas necessárias: {columns}")
        raw_df = pd.DataFrame({col: hf_dataset[col] for col in columns})

        logger.info(f"Total de registros brutos obtidos: {len(raw_df)}")
        return raw_df


class PubMedQADataLoader(BaseQADataLoader):
    """Carregador do PubMedQA (qiaojin/PubMedQA) no mesmo formato pergunta/resposta.

    Cada registro do PubMedQA tem uma pergunta, o abstract do artigo dividido em seções
    ('context'), uma resposta longa ('long_answer') e uma decisão final (yes/no/maybe).

    A pergunta enviada ao modelo inclui o abstract como evidência, e a resposta termina com a
    conclusão e o PMID do artigo, para que o modelo aprenda a responder com base na evidência
    fornecida e a citar a fonte.
    """

    def __init__(
        self,
        dataset_name: str = "qiaojin/PubMedQA",
        config_name: str = "pqa_artificial",
        token: Optional[str] = None,
        split: str = "train",
        sample_size: Optional[int] = None,
        seed: int = 42,
        max_context_chars: int = 3000,
        discarded_output_path: Optional[Path] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 10.0,
    ) -> None:
        """Inicializa o carregador.

        Args:
            dataset_name: Identificador do repositório no Hugging Face.
            config_name: Subconjunto ('pqa_artificial', 'pqa_labeled' ou 'pqa_unlabeled').
            token: Token de acesso do Hugging Face.
            split: Divisão do dataset (o PubMedQA só tem 'train').
            sample_size: Se informado, sorteia esse número de registros (reprodutível pela seed).
            seed: Semente para a amostragem.
            max_context_chars: Limite de caracteres do abstract incluído na pergunta.
            discarded_output_path: Caminho para salvar os registros descartados (JSON).
            max_retries: Tentativas de download antes de desistir (falhas de rede são comuns).
            retry_delay_seconds: Espera entre tentativas (dobra a cada nova tentativa).
        """
        super().__init__(
            discarded_output_path if discarded_output_path is not None else DISCARDED_PUBMEDQA_JSON_PATH
        )
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.token = token
        self.split = split
        self.sample_size = sample_size
        self.seed = seed
        self.max_context_chars = max_context_chars
        self.max_retries = max(1, max_retries)
        self.retry_delay_seconds = retry_delay_seconds

    def _load_hf(self) -> Any:
        logger.info(
            f"Conectando ao Hugging Face para baixar '{self.dataset_name}' "
            f"(config='{self.config_name}', split='{self.split}')..."
        )
        hf_dataset = None
        for attempt in range(1, self.max_retries + 1):
            try:
                hf_dataset = load_dataset(
                    self.dataset_name,
                    self.config_name,
                    split=self.split,
                    token=self.token,
                )
                break
            except Exception as exc:
                if attempt == self.max_retries:
                    error_msg = (
                        f"Falha ao carregar o dataset '{self.dataset_name}/{self.config_name}' "
                        f"após {attempt} tentativa(s): {exc}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from exc
                wait = self.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"Tentativa {attempt}/{self.max_retries} falhou ({exc}). Nova tentativa em {wait:.0f} s..."
                )
                time.sleep(wait)

        if len(hf_dataset) == 0:
            error_msg = f"O dataset '{self.dataset_name}/{self.config_name}' retornado está vazio."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.sample_size is not None and self.sample_size < len(hf_dataset):
            logger.info(f"Amostrando {self.sample_size} de {len(hf_dataset)} registros (seed={self.seed}).")
            hf_dataset = hf_dataset.shuffle(seed=self.seed).select(range(self.sample_size))

        return hf_dataset

    def format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Monta o texto da evidência a partir das seções do abstract, respeitando o limite."""
        if not context:
            return ""
        contexts: List[str] = list(context.get("contexts") or [])
        labels: List[str] = list(context.get("labels") or [])
        parts: List[str] = []
        for idx, text in enumerate(contexts):
            label = labels[idx] if idx < len(labels) and labels[idx] else ""
            parts.append(f"[{label}] {text}" if label else str(text))
        evidence = "\n".join(parts).strip()
        if len(evidence) > self.max_context_chars:
            evidence = evidence[: self.max_context_chars].rsplit(" ", 1)[0] + " [...]"
        return evidence

    def build_user_content(self, question: str, evidence: str, pubid: Any) -> str:
        """Pergunta + evidência no formato que será usado também na inferência (RAG)."""
        return (
            f"Question: {question}\n\n"
            f"Evidence (PubMed abstract, PMID {pubid}):\n{evidence}"
        )

    @staticmethod
    def build_answer(long_answer: str, final_decision: str, pubid: Any) -> str:
        """Resposta longa + conclusão + fonte."""
        decision = (final_decision or "").strip().lower()
        conclusion = f"\n\nConclusion: {decision}." if decision else ""
        return f"{long_answer}{conclusion}\nSource: PubMed PMID {pubid}."

    def load_raw_data(self) -> pd.DataFrame:
        """Baixa o PubMedQA e devolve um DataFrame no formato unificado.

        Returns:
            pd.DataFrame com 'question' (pergunta + evidência), 'answer', 'pubid',
            'raw_question', 'evidence', 'long_answer' e 'final_decision'.
        """
        hf_dataset = self._load_hf()
        rows: List[Dict[str, Any]] = []
        for item in hf_dataset:
            pubid = item.get("pubid")
            raw_q = (item.get("question") or "").strip()
            evidence = self.format_context(item.get("context"))
            long_answer = (item.get("long_answer") or "").strip()
            decision = (item.get("final_decision") or "").strip()
            valid = bool(raw_q and evidence and long_answer)
            rows.append(
                {
                    "question": self.build_user_content(raw_q, evidence, pubid) if valid else None,
                    "answer": self.build_answer(long_answer, decision, pubid) if valid else None,
                    "pubid": pubid,
                    "raw_question": raw_q,
                    "evidence": evidence,
                    "long_answer": long_answer,
                    "final_decision": decision,
                }
            )
        raw_df = pd.DataFrame(rows)
        logger.info(f"Total de registros brutos obtidos do PubMedQA ({self.config_name}): {len(raw_df)}")
        return raw_df
