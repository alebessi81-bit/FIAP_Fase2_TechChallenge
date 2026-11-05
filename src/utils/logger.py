"""Módulo para configuração de logging estruturado (console e arquivo)."""

import logging
from pathlib import Path
import sys
from typing import Optional

from src.config import LOG_FILE_PATH, LOGS_DIR


def setup_logger(
    name: str = "medquad_pipeline",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configura e retorna uma instância de Logger estruturado.

    Garante que as mensagens sejam emitidas tanto no console (sys.stdout)
    quanto gravadas no arquivo de log especificado (padrão: logs/execucao.log).

    Args:
        name: Nome do logger. Padrão "medquad_pipeline".
        level: Nível de severidade dos logs (ex: logging.INFO).
        log_file: Caminho customizado para o arquivo de log. Padrão LOG_FILE_PATH.

    Returns:
        logging.Logger: Instância configurada do logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 1. Handler para console (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Handler para arquivo em disco (logs/execucao.log)
        target_file = log_file or LOG_FILE_PATH
        target_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(target_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
