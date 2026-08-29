"""
Carregamento e preparacao dos dados do Pima Indians Diabetes Dataset.

Mantem o mesmo tratamento da Fase 1 (zeros fisiologicamente impossiveis ->
NaN, StandardScaler, KNNImputer), mas encapsulado num Pipeline do sklearn.

Por que Pipeline: na Fase 1 o scaler e o imputer foram ajustados uma unica
vez sobre o conjunto de treino inteiro. Isso esta correto para uma avaliacao
simples treino/teste, mas dentro de uma validacao cruzada causaria vazamento
entre os folds. Com o Pipeline, o scaler e o imputer sao reajustados dentro
de cada fold automaticamente.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Colunas onde o valor zero e fisiologicamente impossivel e, portanto,
# representa um dado ausente disfarcado.
COLUNAS_COM_ZERO_INVALIDO = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET = "Outcome"


def carregar_dados(caminho_csv):
    """Le o CSV e substitui os zeros invalidos por NaN."""
    dados = pd.read_csv(caminho_csv)
    dados[COLUNAS_COM_ZERO_INVALIDO] = dados[COLUNAS_COM_ZERO_INVALIDO].replace(
        0, np.nan
    )
    return dados


def separar_treino_teste(dados, test_size=0.2, random_state=42):
    """
    Separacao estratificada identica a da Fase 1 (80/20, random_state=42),
    para que a comparacao 'original vs. otimizado' seja justa.

    O conjunto de teste NAO e usado em nenhum momento pelo algoritmo
    genetico. Ele so aparece na avaliacao final.
    """
    X = dados[FEATURE_NAMES]
    y = dados[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def construir_pipeline(svc_params, n_neighbors_imputer=5):
    """
    Monta o pipeline completo: padronizacao -> imputacao -> SVM.

    A ordem (escalar antes de imputar) e a mesma da Fase 1 e e proposital:
    o KNNImputer calcula distancias entre pacientes, entao as features
    precisam estar na mesma escala antes da imputacao. O StandardScaler
    ignora NaN ao calcular media e desvio padrao.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=n_neighbors_imputer)),
            ("svc", SVC(**svc_params)),
        ]
    )


def pipeline_baseline():
    """
    Reproduz exatamente o SVM da Fase 1: kernel rbf, C e gamma no padrao
    do sklearn, class_weight balanceado. Nenhum hiperparametro foi ajustado
    na Fase 1 -- e justamente por isso ha espaco para o algoritmo genetico.
    """
    return construir_pipeline(
        {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "random_state": 42,
        }
    )


def limiar_por_quantil(pipeline, X, quantil):
    """
    Converte um quantil (0-1) no valor absoluto de corte para este modelo
    e estes dados.

    Precisa ser recalculado toda vez que o modelo e retreinado, porque a
    escala do decision_function depende do conjunto de treino. E esse
    recalculo que garante que o ponto de operacao encontrado pelo
    algoritmo genetico se mantenha valido no modelo final.
    """
    escores = pipeline.decision_function(X)
    return float(np.quantile(escores, quantil))
