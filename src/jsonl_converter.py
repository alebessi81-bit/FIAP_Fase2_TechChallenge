"""Módulo para conversão, estruturação e exportação de dados no formato JSONL para fine-tuning."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("jsonl_converter")


@dataclass(frozen=True)
class ChatMessage:
    """Representa uma única mensagem no diálogo de fine-tuning."""

    role: str
    content: str


@dataclass(frozen=True)
class ChatConversation:
    """Representa a conversa completa contendo a lista de mensagens."""

    messages: List[ChatMessage]

    def to_dict(self) -> Dict[str, Any]:
        """Converte a conversa para dicionário compatível com o schema alvo."""
        return asdict(self)


class JsonlConverter:
    """Responsável por converter registros tabulares em formato JSONL para fine-tuning."""

    def __init__(self, system_prompt: str) -> None:
        """Inicializa o conversor com o prompt de sistema configurado.

        Args:
            system_prompt: Texto com as diretrizes do papel do sistema.
        """
        self.system_prompt: str = system_prompt

    def format_row(self, question: str, answer: str) -> Dict[str, Any]:
        """Formata uma tupla de pergunta e resposta na estrutura ChatML/Messages.

        Args:
            question: Pergunta médica do usuário.
            answer: Resposta baseada em evidências científicas.

        Returns:
            Dict[str, Any]: Dicionário no formato {"messages": [...]}.
        """
        conversation = ChatConversation(
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=question),
                ChatMessage(role="assistant", content=answer),
            ]
        )
        return conversation.to_dict()

    def convert_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Converte todas as linhas de um DataFrame limpo na lista de amostras formatadas.

        Args:
            df: DataFrame contendo as colunas 'question' e 'answer'.

        Returns:
            List[Dict[str, Any]]: Lista de dicionários prontos para serialização.
        """
        logger.info(f"Convertendo {len(df)} registros para o formato de mensagens...")
        records: List[Dict[str, Any]] = [
            self.format_row(row["question"], row["answer"])
            for _, row in df.iterrows()
        ]
        return records

    @staticmethod
    def split_train_val(
        records: List[Dict[str, Any]],
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Divide uma lista de amostras em partições de treino e validação.

        Args:
            records: Lista de dicionários de conversa.
            train_ratio: Proporção para a partição de treino (ex: 0.9 = 90%).
            seed: Semente randômica para reprodutibilidade.

        Returns:
            Tuple contendo (train_records, val_records).
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("O train_ratio deve ser um valor entre 0.0 e 1.0 exclusivo.")

        shuffled = records.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]

        logger.info(
            f"Divisão realizada: {len(train_data)} treino ({train_ratio * 100:.0f}%) | "
            f"{len(val_data)} validação ({(1 - train_ratio) * 100:.0f}%)"
        )
        return train_data, val_data

    @staticmethod
    def export_to_jsonl(records: List[Dict[str, Any]], output_path: Path) -> Path:
        """Serializa a lista de registros para um arquivo JSONL no disco com encoding UTF-8.

        Args:
            records: Amostras a serem salvas.
            output_path: Caminho de destino do arquivo .jsonl.

        Returns:
            Path: Caminho do arquivo gravado.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Gravando {len(records)} linhas em: {output_path}")
        with output_path.open("w", encoding="utf-8") as f:
            for item in records:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + "\n")

        logger.info(f"Arquivo gerado com sucesso: {output_path}")
        return output_path

    @staticmethod
    def validate_jsonl_file(file_path: Path) -> bool:
        """Valida a integridade sintática e estrutural de um arquivo JSONL gerado.

        Garante que cada linha seja um JSON válido, possua a chave 'messages'
        e os três papéis obrigatórios ('system', 'user', 'assistant') com conteúdo preenchido.

        Args:
            file_path: Caminho do arquivo JSONL a ser validado.

        Returns:
            bool: True se o arquivo for 100% válido.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se houver qualquer linha malformada ou violando o schema.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        expected_roles = ["system", "user", "assistant"]
        line_count = 0

        with file_path.open("r", encoding="utf-8") as f:
            for idx, raw_line in enumerate(f, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Linha {idx}: JSON inválido ({err})") from err

                if "messages" not in data or not isinstance(data["messages"], list):
                    raise ValueError(f"Linha {idx}: Chave 'messages' ausente ou inválida.")

                messages = data["messages"]
                if len(messages) != 3:
                    raise ValueError(f"Linha {idx}: Esperado exatamente 3 mensagens, encontrado {len(messages)}.")

                for msg, expected_role in zip(messages, expected_roles):
                    if msg.get("role") != expected_role:
                        raise ValueError(f"Linha {idx}: Papel incorreto ({msg.get('role')} != {expected_role}).")
                    if not msg.get("content") or not isinstance(msg.get("content"), str):
                        raise ValueError(f"Linha {idx}: Conteúdo vazio ou inválido para o papel '{expected_role}'.")

                line_count += 1

        logger.info(f"Validação concluída: {line_count} linhas verificadas com 100% de conformidade em {file_path}")
        return True

