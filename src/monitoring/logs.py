"""
Logging estruturado.

Decisao: logs em JSON, uma linha por evento, com os campos "severity" e
"message". Nao e capricho -- o Cloud Logging (e o Cloud Run por tabela)
faz o parsing automatico desse formato, entao os campos extras viram
campos pesquisaveis no console em vez de texto solto. Um log de texto
comum vira uma string opaca la.

Cada requisicao carrega um request_id, o que permite juntar todos os
eventos de uma mesma chamada quando varias acontecem em paralelo.
"""

import json
import logging
import sys
from datetime import datetime, timezone

# Mapeia os niveis do Python para os niveis que o Cloud Logging entende.
NIVEIS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}

CAMPOS_PADRAO = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class FormatadorJSON(logging.Formatter):
    def format(self, registro):
        saida = {
            "timestamp": datetime.fromtimestamp(
                registro.created, tz=timezone.utc
            ).isoformat(),
            "severity": NIVEIS.get(registro.levelname, "DEFAULT"),
            "message": registro.getMessage(),
            "logger": registro.name,
        }

        # Qualquer campo passado via extra={...} entra no JSON.
        for chave, valor in registro.__dict__.items():
            if chave not in CAMPOS_PADRAO and not chave.startswith("_"):
                saida[chave] = valor

        if registro.exc_info:
            saida["exception"] = self.formatException(registro.exc_info)

        return json.dumps(saida, ensure_ascii=False, default=str)


def configurar_logging(nivel="INFO"):
    """Instala o formatador JSON na saida padrao."""
    manipulador = logging.StreamHandler(sys.stdout)
    manipulador.setFormatter(FormatadorJSON())

    raiz = logging.getLogger()
    raiz.handlers.clear()
    raiz.addHandler(manipulador)
    raiz.setLevel(nivel)

    # O uvicorn instala os proprios handlers; redirecionamos para o nosso
    # para nao termos dois formatos misturados na mesma saida.
    for nome in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(nome)
        logger.handlers.clear()
        logger.propagate = True

    return raiz
