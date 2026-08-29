"""
Configuracao do modelo campeao.

Os tres experimentos produziram modelos parecidos no conjunto de teste,
separados por 3 pacientes -- diferenca dentro do ruido de uma amostra de
154 pessoas. O criterio de escolha, portanto, NAO pode ser o desempenho no
teste: usar o teste para escolher entre modelos e o mesmo erro que a Fase 1
cometeu ao selecionar o k do KNN.

O criterio correto e o fitness em validacao cruzada, medido apenas sobre o
conjunto de treino:

    Experimento A  ->  0.7909
    Experimento B  ->  0.7959   <-- escolhido
    Experimento C  ->  0.7828

Por isso o campeao e o experimento B.

Observacao sobre reprodutibilidade: os valores abaixo foram transcritos da
saida do experimento B e estao arredondados (gamma aparece com 5 casas
decimais no log). Para regenerar os genes com precisao total, rode
novamente `python run_experiments.py --experimentos B` -- o arquivo
resultados/melhores_individuos.json tera os genes exatos.
"""

CAMPEAO = {
    "origem": "Experimento B",
    "fitness_cv": 0.7959,
    "sensibilidade_cv": 0.9300,
    "especificidade_cv": 0.5825,
    "features": [
        "Pregnancies",
        "Glucose",
        "Insulin",
        "BMI",
        "Age",
    ],
    "quantil": 0.404,
    "svc_params": {
        "kernel": "rbf",
        "C": 0.0424,
        "gamma": 0.00090,
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 200_000,
        "cache_size": 500,
    },
    # Desempenho no conjunto de teste, medido uma unica vez.
    "metricas_teste": {
        "accuracy": 0.6818,
        "recall": 0.8704,
        "precision": 0.5281,
        "f1": 0.6573,
        "roc_auc": 0.7989,
    },
}
