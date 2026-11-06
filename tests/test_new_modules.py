"""Testes unitários dos módulos de PubMedQA, anonimização, curadoria e base sintética."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.anonymizer import Anonymizer
from src.curation import DatasetCurator, normalize_text
from src.data_loader import PubMedQADataLoader
from src.synthetic_generator import SyntheticHospitalDataGenerator


def _fake_pubmedqa_items():
    return [
        {
            "pubid": 111,
            "question": "Does drug X reduce mortality?",
            "context": {"contexts": ["Background text.", "Results text."], "labels": ["BACKGROUND", "RESULTS"]},
            "long_answer": "Drug X reduced mortality in this cohort.",
            "final_decision": "yes",
        },
        {
            "pubid": 222,
            "question": "Is Y associated with Z?",
            "context": {"contexts": [], "labels": []},
            "long_answer": "No evidence.",
            "final_decision": "no",
        },
    ]


class TestPubMedQADataLoader(unittest.TestCase):
    @patch("src.data_loader.load_dataset")
    def test_load_and_clean(self, mock_load_dataset: MagicMock) -> None:
        items = _fake_pubmedqa_items()
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = len(items)
        mock_ds.__iter__.return_value = iter(items)
        mock_load_dataset.return_value = mock_ds

        with tempfile.TemporaryDirectory() as tmp:
            loader = PubMedQADataLoader(token="t", discarded_output_path=Path(tmp) / "d.json")
            df = loader.clean_data()

        mock_load_dataset.assert_called_once_with("qiaojin/PubMedQA", "pqa_artificial", split="train", token="t")
        # O segundo item não tem contexto e deve ser descartado
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertIn("Evidence (PubMed abstract, PMID 111)", row["question"])
        self.assertIn("[BACKGROUND] Background text.", row["question"])
        self.assertTrue(row["answer"].endswith("Source: PubMed PMID 111."))
        self.assertIn("Conclusion: yes.", row["answer"])

    @patch("src.data_loader.time.sleep")
    @patch("src.data_loader.load_dataset")
    def test_retries_then_succeeds(self, mock_load_dataset: MagicMock, mock_sleep: MagicMock) -> None:
        items = _fake_pubmedqa_items()
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = len(items)
        mock_ds.__iter__.return_value = iter(items)
        mock_load_dataset.side_effect = [ConnectionError("rede caiu"), mock_ds]

        df = PubMedQADataLoader(max_retries=3, retry_delay_seconds=1).load_raw_data()
        self.assertEqual(len(df), 2)
        self.assertEqual(mock_load_dataset.call_count, 2)
        mock_sleep.assert_called_once_with(1)

    @patch("src.data_loader.time.sleep")
    @patch("src.data_loader.load_dataset", side_effect=ConnectionError("rede caiu"))
    def test_retries_exhausted_raises(self, mock_load_dataset: MagicMock, _sleep: MagicMock) -> None:
        with self.assertRaises(RuntimeError):
            PubMedQADataLoader(max_retries=2, retry_delay_seconds=1).load_raw_data()
        self.assertEqual(mock_load_dataset.call_count, 2)

    def test_context_truncation(self) -> None:
        loader = PubMedQADataLoader(max_context_chars=50)
        text = loader.format_context({"contexts": ["palavra " * 50], "labels": ["METHODS"]})
        self.assertLessEqual(len(text), 60)
        self.assertTrue(text.endswith("[...]"))


class TestAnonymizer(unittest.TestCase):
    def test_replaces_pii(self) -> None:
        anon = Anonymizer(known_names=["Maria Souza Lima"])
        text = (
            "Paciente: Maria Souza Lima, CPF 123.456.789-01, prontuário 445566, "
            "telefone (11) 98765-4321, admitida em 03/11/2026, email maria@exemplo.com."
        )
        out = anon.anonymize(text)
        for leaked in ["Maria", "123.456.789-01", "445566", "98765-4321", "03/11/2026", "maria@exemplo.com"]:
            self.assertNotIn(leaked, out)
        for tag in ["[NOME]", "[CPF]", "[PRONTUARIO]", "[TELEFONE]", "[DATA]", "[EMAIL]"]:
            self.assertIn(tag, out)

    def test_keeps_clinical_values(self) -> None:
        out = Anonymizer().anonymize("PA 180/120 mmHg, glicemia 54 mg/dL, lactato 4.2 mmol/L.")
        self.assertEqual(out, "PA 180/120 mmHg, glicemia 54 mg/dL, lactato 4.2 mmol/L.")


class TestCurator(unittest.TestCase):
    def test_filters_and_dedup(self) -> None:
        df = pd.DataFrame(
            {
                "question": ["What is asthma?", "What is asthma?", "Hi?", "What is gout?", "What is (are) X?"],
                "answer": ["Asthma is a chronic disease of airways.", "Asthma  is a chronic disease of airways.",
                           "Answer long enough here.", "Short", "Answer from document B, long enough."],
                "source": ["medquad"] * 5,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = DatasetCurator(discarded_output_path=Path(tmp) / "c.json").curate(df)
            self.assertTrue((Path(tmp) / "c.json").exists())
        self.assertEqual(list(out["question"]), ["What is asthma?", "What is (are) X?"])

    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("  a   b \r\n\n\n\n c  "), "a b\n\nc")


class TestSyntheticGenerator(unittest.TestCase):
    def test_generate_is_reproducible_and_anonymizable(self) -> None:
        g1 = SyntheticHospitalDataGenerator(seed=1)
        r1 = g1.generate(cases_per_protocol=5)
        r2 = SyntheticHospitalDataGenerator(seed=1).generate(cases_per_protocol=5)
        self.assertEqual(r1, r2)

        categories = {r["category"] for r in r1}
        self.assertTrue({"protocolo_faq", "caso_clinico", "modelo_laudo", "minuta_receita", "seguranca"} <= categories)
        # Toda resposta cita a fonte
        self.assertTrue(all("Fonte" in r["answer"] or "protocolo" in r["answer"] for r in r1))

        df = Anonymizer(known_names=g1.generated_names).anonymize_dataframe(pd.DataFrame(r1))
        cases = df[df["category"] == "caso_clinico"]
        for name in set(g1.generated_names):
            self.assertFalse(cases["question"].str.contains(name, regex=False).any())
        self.assertTrue(cases["question"].str.contains(r"\[CPF\]").all())


if __name__ == "__main__":
    unittest.main()
