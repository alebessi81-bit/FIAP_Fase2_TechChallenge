"""Testes unitários para o módulo jsonl_converter."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.jsonl_converter import JsonlConverter


class TestJsonlConverter(unittest.TestCase):
    """Conjunto de testes para a classe JsonlConverter."""

    def setUp(self) -> None:
        """Configura ambiente temporário e dados de teste."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.system_prompt = "Test System Prompt"
        self.converter = JsonlConverter(system_prompt=self.system_prompt)

    def tearDown(self) -> None:
        """Limpa o diretório temporário após os testes."""
        self.temp_dir.cleanup()

    def test_format_row_structure(self) -> None:
        """Garante que a estrutura de mensagens gerada obedece à especificação."""
        row_dict = self.converter.format_row(
            question="What is hypertension?",
            answer="Hypertension is high blood pressure.",
        )

        self.assertIn("messages", row_dict)
        messages = row_dict["messages"]
        self.assertEqual(len(messages), 3)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], self.system_prompt)

        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What is hypertension?")

        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Hypertension is high blood pressure.")

    def test_convert_dataframe(self) -> None:
        """Testa conversão em lote a partir de DataFrame."""
        df = pd.DataFrame(
            {
                "question": ["Q1", "Q2"],
                "answer": ["A1", "A2"],
            }
        )
        records = self.converter.convert_dataframe(df)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["messages"][1]["content"], "Q1")
        self.assertEqual(records[1]["messages"][2]["content"], "A2")

    def test_split_train_val_proportions_and_reproducibility(self) -> None:
        """Verifica se a divisão obedece ao train_ratio e é reprodutível com seed fixo."""
        records = [self.converter.format_row(f"Q{i}", f"A{i}") for i in range(100)]

        train1, val1 = self.converter.split_train_val(records, train_ratio=0.8, seed=42)
        train2, val2 = self.converter.split_train_val(records, train_ratio=0.8, seed=42)

        self.assertEqual(len(train1), 80)
        self.assertEqual(len(val1), 20)
        # Reprodutibilidade com a mesma semente
        self.assertEqual(train1, train2)
        self.assertEqual(val1, val2)

    def test_export_and_validate_jsonl(self) -> None:
        """Valida a exportação física em JSONL e a validação de integridade."""
        records = [
            self.converter.format_row("Pergunta 1?", "Resposta 1."),
            self.converter.format_row("Pergunta 2?", "Resposta 2."),
        ]
        target_file = self.temp_path / "dataset.jsonl"
        self.converter.export_to_jsonl(records, target_file)

        self.assertTrue(target_file.exists())

        # Verifica se o arquivo é válido pelo método de validação
        is_valid = self.converter.validate_jsonl_file(target_file)
        self.assertTrue(is_valid)

        # Validação manual linha a linha
        lines = target_file.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        for line in lines:
            parsed = json.loads(line)
            self.assertIn("messages", parsed)

    def test_validate_jsonl_catches_invalid_json(self) -> None:
        """Garante que a validação falhe se houver uma linha corrompida."""
        corrupted_file = self.temp_path / "corrupted.jsonl"
        corrupted_file.write_text("NOT A VALID JSON\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.converter.validate_jsonl_file(corrupted_file)

    def test_validate_jsonl_catches_missing_roles(self) -> None:
        """Garante que a validação falhe se faltar o papel do assistente."""
        invalid_sample = {
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "User"},
            ]
        }
        invalid_file = self.temp_path / "invalid_schema.jsonl"
        invalid_file.write_text(json.dumps(invalid_sample) + "\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            self.converter.validate_jsonl_file(invalid_file)


if __name__ == "__main__":
    unittest.main()

