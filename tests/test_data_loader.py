"""Testes unitários para o módulo data_loader (Hugging Face)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data_loader import MedQuadDataLoader


class TestMedQuadDataLoader(unittest.TestCase):
    """Conjunto de testes para a classe MedQuadDataLoader."""

    def setUp(self) -> None:
        """Inicializa instância padrão do loader e diretório temporário."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.discarded_path = self.temp_path / "discarded.json"

        self.loader = MedQuadDataLoader(
            dataset_name="lavita/MedQuAD",
            token="fake_token",
            split="train",
            discarded_output_path=self.discarded_path,
        )

    def tearDown(self) -> None:
        """Remove o diretório temporário."""
        self.temp_dir.cleanup()

    @patch("src.data_loader.load_dataset")
    def test_load_raw_data_success(self, mock_load_dataset: MagicMock) -> None:
        """Deve carregar e extrair com sucesso as colunas 'question' e 'answer'."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.column_names = ["question", "answer", "other_col"]
        mock_dataset.__getitem__.side_effect = lambda key: {
            "question": ["What is asthma?", "What is diabetes?"],
            "answer": ["Asthma is...", "Diabetes is..."],
        }[key]

        mock_load_dataset.return_value = mock_dataset

        df = self.loader.load_raw_data()
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["question", "answer"])
        mock_load_dataset.assert_called_once_with(
            "lavita/MedQuAD",
            split="train",
            token="fake_token",
        )

    @patch("src.data_loader.load_dataset")
    def test_load_raw_data_missing_columns_raises_value_error(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """Deve lançar ValueError se faltar alguma coluna obrigatória."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 1
        mock_dataset.column_names = ["pergunta", "resposta"]

        mock_load_dataset.return_value = mock_dataset

        with self.assertRaises(ValueError):
            self.loader.load_raw_data()

    @patch("src.data_loader.load_dataset")
    def test_load_raw_data_empty_dataset_raises_value_error(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """Deve lançar ValueError se o dataset retornado for vazio."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 0

        mock_load_dataset.return_value = mock_dataset

        with self.assertRaises(ValueError):
            self.loader.load_raw_data()

    @patch("src.data_loader.load_dataset")
    def test_load_raw_data_network_error_raises_runtime_error(
        self, mock_load_dataset: MagicMock
    ) -> None:
        """Deve capturar exceção de rede/auth e encapsular em RuntimeError."""
        mock_load_dataset.side_effect = Exception("Conexão recusada pela API")

        with self.assertRaises(RuntimeError):
            self.loader.load_raw_data()

    def test_clean_data_removes_nulls_and_empty_strings(self) -> None:
        """Deve remover linhas com NaN ou strings vazias e salvar discarded.json."""
        df_dirty = pd.DataFrame(
            {
                "question": [
                    "What is glaucoma?",
                    None,
                    "   ",
                    "What causes asthma?",
                    "Valid question",
                ],
                "answer": [
                    "Glaucoma is an eye disease.",
                    "Some answer",
                    "Valid answer",
                    "",
                    "  Valid answer with spaces  ",
                ],
            }
        )

        cleaned_df = self.loader.clean_data(df=df_dirty, save_discarded=True)

        # Apenas as linhas 0 e 4 são válidas
        self.assertEqual(len(cleaned_df), 2)

        # Verifica se os espaços foram removidos
        row_last = cleaned_df.iloc[1]
        self.assertEqual(row_last["question"], "Valid question")
        self.assertEqual(row_last["answer"], "Valid answer with spaces")

        # Verifica se o arquivo de descartes foi gerado com 3 registros
        self.assertTrue(self.discarded_path.exists())
        with self.discarded_path.open("r", encoding="utf-8") as f:
            discarded_data = json.load(f)

        self.assertEqual(len(discarded_data), 3)


if __name__ == "__main__":
    unittest.main()
