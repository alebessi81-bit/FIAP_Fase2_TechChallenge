"""Módulo de anonimização de dados sensíveis (PII) em textos clínicos.

Substitui identificadores pessoais por marcadores ([NOME], [CPF], [DATA] etc.) antes que o
texto entre no dataset de fine-tuning. Usa regras (regex) para padrões brasileiros comuns e,
opcionalmente, uma lista de nomes conhecidos (por exemplo, o cadastro de pacientes).

Limitação: regex não detecta todo nome próprio em texto livre. Em produção, combine com um
modelo de NER (por exemplo, Microsoft Presidio com spaCy pt_core_news_lg) e revisão humana.
"""

from collections import Counter
from dataclasses import dataclass, field
import logging
import re
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

import pandas as pd

from src.utils.logger import setup_logger

logger: logging.Logger = setup_logger("anonymizer")

# Ordem importa: padrões mais específicos primeiro (CNS antes de telefone, CPF antes de RG).
DEFAULT_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\bCPF:?\s*\d{11}\b", re.IGNORECASE)),
    ("CNS", re.compile(r"\b(?:CNS|cart[aã]o SUS):?\s*\d{15}\b|\b\d{3}\s\d{4}\s\d{4}\s\d{4}\b", re.IGNORECASE)),
    ("RG", re.compile(r"\bRG:?\s*[\d.]{7,12}-?[\dXx]?\b", re.IGNORECASE)),
    ("TELEFONE", re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b")),
    # Grupo 1 (rótulo) é preservado; só o número é substituído.
    ("PRONTUARIO", re.compile(r"\b((?:prontu[aá]rio|registro|MRN)\s*(?:n[ºo°.]*\s*)?:?\s*)\d{4,}\b", re.IGNORECASE)),
    ("DATA", re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
]

# Nome logo após um título/rótulo típico de prontuário: "Paciente: Maria Souza", "Sra. Ana Lima".
NAME_AFTER_LABEL: Pattern[str] = re.compile(
    r"\b(Paciente:?|Sr\.?|Sra\.?|Dr\.?|Dra\.?|Nome:?)\s+"
    r"([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+){0,4})"
)


@dataclass
class Anonymizer:
    """Anonimizador baseado em regras com contagem de substituições para auditoria."""

    known_names: Iterable[str] = field(default_factory=list)
    patterns: List[Tuple[str, Pattern[str]]] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    stats: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        names = sorted({n.strip() for n in self.known_names if n and n.strip()}, key=len, reverse=True)
        self._names_pattern: Optional[Pattern[str]] = (
            re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b") if names else None
        )

    def anonymize(self, text: Optional[str]) -> Optional[str]:
        """Devolve o texto com os identificadores trocados por marcadores."""
        if text is None or not isinstance(text, str) or not text:
            return text

        if self._names_pattern is not None:
            text, n = self._names_pattern.subn("[NOME]", text)
            self.stats["NOME"] += n

        def _label_name(match: "re.Match[str]") -> str:
            self.stats["NOME"] += 1
            return f"{match.group(1)} [NOME]"

        text = NAME_AFTER_LABEL.sub(_label_name, text)

        for label, pattern in self.patterns:
            replacement = f"\\g<1>[{label}]" if pattern.groups else f"[{label}]"
            text, n = pattern.subn(replacement, text)
            self.stats[label] += n
        return text

    def anonymize_dataframe(self, df: pd.DataFrame, columns: Iterable[str] = ("question", "answer")) -> pd.DataFrame:
        """Aplica a anonimização nas colunas indicadas e registra o total por tipo no log."""
        before = Counter(self.stats)
        out = df.copy()
        for col in columns:
            if col in out.columns:
                out[col] = out[col].map(self.anonymize)
        delta: Dict[str, int] = {k: v - before.get(k, 0) for k, v in self.stats.items() if v - before.get(k, 0)}
        logger.info(f"Anonimização concluída em {len(out)} registros. Substituições: {delta or 'nenhuma'}")
        return out
