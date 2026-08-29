"""
Executa o baseline da Fase 1 e os tres experimentos com o algoritmo
genetico exigidos pelo enunciado.

Uso:
    python run_experiments.py --dados data/diabetes.csv

Saidas (na pasta resultados/):
    historico_exp_A.csv, historico_exp_B.csv, historico_exp_C.csv
    comparativo.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import (
    carregar_dados,
    construir_pipeline,
    limiar_por_quantil,
    pipeline_baseline,
    separar_treino_teste,
)
from src.genetic.chromosome import decodificar, descrever
from src.genetic.engine import AlgoritmoGenetico
from src.genetic.fitness import AvaliadorFitness

# Os tres experimentos obrigatorios. Cada um explora o espaco de busca de
# um jeito diferente, e nao apenas com numeros distintos.
EXPERIMENTOS = {
    "A": {
        "descricao": "Baseline do AG - equilibrio entre exploracao e refinamento",
        "tamanho_populacao": 30,
        "n_geracoes": 20,
        "taxa_mutacao": 0.10,
        "taxa_cruzamento": 0.80,
        "selecao": "torneio",
        "cruzamento": "uniforme",
        "n_elite": 2,
    },
    "B": {
        "descricao": "Exploracao - populacao grande e mutacao alta",
        "tamanho_populacao": 60,
        "n_geracoes": 20,
        "taxa_mutacao": 0.30,
        "taxa_cruzamento": 0.60,
        "selecao": "torneio",
        "cruzamento": "blend",
        "n_elite": 2,
    },
    "C": {
        "descricao": "Intensificacao - populacao pequena, muitas geracoes",
        "tamanho_populacao": 20,
        "n_geracoes": 40,
        "taxa_mutacao": 0.05,
        "taxa_cruzamento": 0.90,
        "selecao": "roleta",
        "cruzamento": "blend",
        "n_elite": 4,
    },
}


def avaliar_no_teste(pipeline, X_test, y_test, limiar=None):
    """
    Avaliacao final no conjunto de teste, que ficou intocado durante toda
    a evolucao. Estes sao os numeros que vao para o relatorio.
    """
    escores = pipeline.decision_function(X_test)
    if limiar is None:
        y_pred = pipeline.predict(X_test)
    else:
        y_pred = (escores >= limiar).astype(int)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, escores),
    }


def main():
    # Individuos que batem no teto de iteracoes geram ConvergenceWarning.
    # E esperado e nao e erro -- silenciamos para nao poluir a saida.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="data/diabetes.csv")
    parser.add_argument("--saida", default="resultados")
    parser.add_argument("--experimentos", default="A,B,C")
    args = parser.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    dados = carregar_dados(args.dados)
    X_train, X_test, y_train, y_test = separar_treino_teste(dados)
    print(f"Treino: {len(X_train)} pacientes | Teste: {len(X_test)} pacientes\n")

    linhas = []

    # ------------------------------------------------- baseline da Fase 1
    print("=" * 70)
    print("BASELINE - SVM da Fase 1 (sem otimizacao de hiperparametros)")
    print("=" * 70)
    baseline = pipeline_baseline()
    baseline.fit(X_train, y_train)
    metricas_baseline = avaliar_no_teste(baseline, X_test, y_test)
    for nome, valor in metricas_baseline.items():
        print(f"  {nome:10s}: {valor:.4f}")
    linhas.append({"modelo": "Baseline (Fase 1)", **metricas_baseline})

    # ------------------------------------------------------- experimentos
    melhores = {}
    for chave in args.experimentos.split(","):
        chave = chave.strip()
        config = dict(EXPERIMENTOS[chave])
        descricao = config.pop("descricao")

        print()
        print("=" * 70)
        print(f"EXPERIMENTO {chave} - {descricao}")
        print("=" * 70)
        print(f"  {config}\n")

        avaliador = AvaliadorFitness(X_train, y_train)
        ag = AlgoritmoGenetico(avaliador, **config)
        melhor = ag.executar()

        ag.historico_df().to_csv(saida / f"historico_exp_{chave}.csv", index=False)

        # Retreina o melhor individuo em todo o conjunto de treino e avalia
        # uma unica vez no teste.
        d = decodificar(melhor)
        pipeline = construir_pipeline(d["svc_params"])
        X_train_sel = X_train.loc[:, d["mascara"]]
        pipeline.fit(X_train_sel, y_train)

        # O corte absoluto e recalculado a partir do modelo retreinado.
        limiar = limiar_por_quantil(pipeline, X_train_sel, d["quantil"])

        metricas = avaliar_no_teste(
            pipeline, X_test.loc[:, d["mascara"]], y_test, limiar=limiar
        )

        print("\n  Metricas no conjunto de teste:")
        for nome, valor in metricas.items():
            print(f"    {nome:10s}: {valor:.4f}")

        melhores[chave] = {
            "genes": melhor.tolist(),
            "descricao": descrever(melhor),
            "config": ag.configuracao(),
            "fitness_cv": ag.melhor_fitness,
            "limiar_absoluto": limiar,
            "tempo_s": ag.tempo_execucao,
            "metricas_teste": metricas,
        }
        linhas.append({"modelo": f"AG - Experimento {chave}", **metricas})

    # --------------------------------------------------------- comparativo
    comparativo = pd.DataFrame(linhas)
    comparativo.to_csv(saida / "comparativo.csv", index=False)

    with open(saida / "melhores_individuos.json", "w", encoding="utf-8") as f:
        json.dump(melhores, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("COMPARATIVO FINAL (conjunto de teste)")
    print("=" * 70)
    print(comparativo.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nResultados salvos em {saida.resolve()}")


if __name__ == "__main__":
    main()
