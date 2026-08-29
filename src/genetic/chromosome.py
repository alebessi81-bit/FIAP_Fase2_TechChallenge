"""
Codificacao do cromossomo.

Decisao de projeto: todos os genes sao numeros reais no intervalo [0, 1],
independentemente do que representam. Um gene continuo (C), um gene
categorico (kernel) e um gene binario (usar ou nao uma feature) sao todos
armazenados da mesma forma e so ganham significado na decodificacao.

Isso traz duas vantagens praticas:
  1. Os operadores de cruzamento e mutacao funcionam sobre o vetor inteiro
     sem precisar de tratamento especial por tipo de gene;
  2. Qualquer vetor em [0, 1]^13 e sempre um individuo valido -- nao existe
     combinacao que gere um SVM impossivel de treinar, entao o AG nunca
     desperdica avaliacoes com individuos invalidos.

Layout do cromossomo (13 genes):

  indice | gene              | decodificacao
  -------|-------------------|--------------------------------------------
     0   | C                 | 10^(-2 + 4*g)      -> [0.01, 100]
     1   | gamma             | 10^(-4 + 5*g)      -> [1e-4, 10]
     2   | kernel            | rbf | poly | sigmoid
     3   | class_weight      | None | balanced
   4-11  | mascara features  | g >= 0.5 -> feature usada
    12   | quantil do corte  | 0.20 + 0.70*g      -> [0.20, 0.90]

Sobre o gene 12 (decisao revista depois de um bug):

A primeira versao guardava o limiar como um valor absoluto sobre
decision_function(), algo como -0.686. Isso quebrou na pratica: o limiar
era escolhido durante a validacao cruzada, onde cada modelo ve 80% do
treino, mas depois era aplicado ao modelo final, treinado com 100%. A
escala do decision_function muda entre os dois, e um limiar que separava
bem nos folds passou a ficar abaixo de TODOS os escores no modelo final --
resultado: todo paciente classificado como diabetico.

A correcao e guardar um quantil em vez de um valor absoluto. O gene diz
"corte no percentil 35 dos escores de treino", e o valor numerico do corte
e recalculado a partir dos dados sempre que o modelo e retreinado. Assim o
ponto de operacao se mantem, independentemente da escala.

A faixa comeca em 0.20 (e nao em 0.0) justamente para impedir a solucao
degenerada: no minimo 20% dos pacientes serao classificados como
negativos.

O corte age sobre decision_function() e nao sobre predict_proba(). Isso
evita a calibracao de Platt (que exigiria probability=True e treinaria 5
SVMs internos por avaliacao), deixando o AG varias vezes mais rapido.
"""

import numpy as np

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

KERNELS = ["rbf", "poly", "sigmoid"]
CLASS_WEIGHTS = [None, "balanced"]

IDX_C = 0
IDX_GAMMA = 1
IDX_KERNEL = 2
IDX_CLASS_WEIGHT = 3
IDX_MASCARA = slice(4, 4 + len(FEATURE_NAMES))
IDX_QUANTIL = 4 + len(FEATURE_NAMES)

N_GENES = IDX_QUANTIL + 1  # 13


def _categorico(valor, opcoes):
    """Mapeia um gene em [0, 1] para uma das opcoes discretas."""
    indice = int(valor * len(opcoes))
    return opcoes[min(indice, len(opcoes) - 1)]


def decodificar(genes):
    """
    Converte um vetor de 13 genes reais nos parametros efetivos do modelo.

    Retorna um dicionario com:
      svc_params      -- pronto para ser passado ao SVC
      mascara         -- array booleano de 8 posicoes
      features        -- nomes das features selecionadas
      quantil         -- percentil dos escores de treino onde fica o corte
    """
    genes = np.asarray(genes, dtype=float)

    C = 10.0 ** (-2.0 + 4.0 * genes[IDX_C])
    gamma = 10.0 ** (-4.0 + 5.0 * genes[IDX_GAMMA])
    kernel = _categorico(genes[IDX_KERNEL], KERNELS)
    class_weight = _categorico(genes[IDX_CLASS_WEIGHT], CLASS_WEIGHTS)
    mascara = genes[IDX_MASCARA] >= 0.5
    quantil = 0.20 + 0.70 * genes[IDX_QUANTIL]

    svc_params = {
        "C": C,
        "gamma": gamma,
        "kernel": kernel,
        "class_weight": class_weight,
        "random_state": 42,
        # Teto de iteracoes: sem ele, combinacoes patologicas (kernel poly
        # com C alto e gamma alto, por exemplo) podem levar minutos ou
        # simplesmente nao convergir, travando a evolucao inteira. Com o
        # teto, o SVM devolve a melhor solucao encontrada ate ali. Um
        # individuo que nao converge tende a ter fitness ruim e some
        # sozinho da populacao.
        "max_iter": 200_000,
        # Cache do kernel em MB: acelera bastante o treino do SVM.
        "cache_size": 500,
    }
    if kernel == "poly":
        svc_params["degree"] = 3  # fixo, para nao inflar o espaco de busca

    return {
        "svc_params": svc_params,
        "mascara": mascara,
        "features": [f for f, m in zip(FEATURE_NAMES, mascara) if m],
        "quantil": float(quantil),
    }


def individuo_aleatorio(rng):
    """Gera um individuo uniformemente distribuido no espaco de busca."""
    return rng.random(N_GENES)


def populacao_inicial(tamanho, rng):
    """Gera a populacao da geracao 0."""
    return np.array([individuo_aleatorio(rng) for _ in range(tamanho)])


def descrever(genes):
    """Descricao legivel de um individuo, para logs e para o relatorio."""
    d = decodificar(genes)
    p = d["svc_params"]
    return (
        f"kernel={p['kernel']}, C={p['C']:.4f}, gamma={p['gamma']:.5f}, "
        f"class_weight={p['class_weight']}, quantil={d['quantil']:.3f}, "
        f"features={len(d['features'])}/8 {d['features']}"
    )
