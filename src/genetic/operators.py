"""
Operadores geneticos.

Duas opcoes de selecao e duas de cruzamento, para que os tres experimentos
exigidos pelo enunciado possam variar tambem a estrategia, e nao apenas os
numeros (tamanho de populacao e taxas).
"""

import numpy as np


# ---------------------------------------------------------------- SELECAO


def selecao_torneio(populacao, fitness, rng, k=3):
    """
    Sorteia k individuos e devolve o melhor deles.

    E a selecao mais usada na pratica porque a pressao seletiva depende
    apenas de k, e nao da escala do fitness. Aumentar k acelera a
    convergencia; diminuir preserva diversidade.
    """
    indices = rng.integers(0, len(populacao), size=k)
    vencedor = indices[np.argmax(fitness[indices])]
    return populacao[vencedor].copy()


def selecao_roleta(populacao, fitness, rng, k=None):
    """
    Probabilidade de selecao proporcional ao fitness.

    Sensivel a escala: se todos os individuos tiverem fitness parecido, a
    roleta vira quase um sorteio uniforme e a convergencia fica lenta.
    Deslocamos o fitness para o positivo antes de normalizar.
    """
    ajustado = fitness - fitness.min() + 1e-9
    probabilidades = ajustado / ajustado.sum()
    indice = rng.choice(len(populacao), p=probabilidades)
    return populacao[indice].copy()


SELECOES = {"torneio": selecao_torneio, "roleta": selecao_roleta}


# ------------------------------------------------------------- CRUZAMENTO


def cruzamento_uniforme(pai1, pai2, rng, taxa):
    """
    Cada gene vem do pai 1 ou do pai 2, sorteado independentemente.

    Bom para os genes categoricos e para a mascara de features, porque
    preserva valores que ja funcionam em vez de criar medias sem sentido.
    """
    if rng.random() > taxa:
        return pai1.copy(), pai2.copy()
    mascara = rng.random(len(pai1)) < 0.5
    filho1 = np.where(mascara, pai1, pai2)
    filho2 = np.where(mascara, pai2, pai1)
    return filho1, filho2


def cruzamento_blend(pai1, pai2, rng, taxa, alpha=0.5):
    """
    Cruzamento BLX-alpha: cada gene do filho e sorteado num intervalo que
    se estende alem dos valores dos pais.

    Bom para os genes continuos (C e gamma), porque permite explorar
    regioes fora do segmento entre os pais em vez de so interpolar.
    """
    if rng.random() > taxa:
        return pai1.copy(), pai2.copy()
    minimo = np.minimum(pai1, pai2)
    maximo = np.maximum(pai1, pai2)
    intervalo = maximo - minimo
    inferior = minimo - alpha * intervalo
    superior = maximo + alpha * intervalo
    filho1 = rng.uniform(inferior, superior)
    filho2 = rng.uniform(inferior, superior)
    return np.clip(filho1, 0.0, 1.0), np.clip(filho2, 0.0, 1.0)


CRUZAMENTOS = {"uniforme": cruzamento_uniforme, "blend": cruzamento_blend}


# ---------------------------------------------------------------- MUTACAO


def mutacao_gaussiana(individuo, rng, taxa, sigma=0.15):
    """
    Cada gene tem probabilidade 'taxa' de receber um ruido gaussiano.

    Mutacao gaussiana (e nao substituicao por valor aleatorio) faz uma
    busca local em torno do individuo, o que preserva boas solucoes
    enquanto ainda permite escapar de otimos locais.
    """
    mutante = individuo.copy()
    sortear = rng.random(len(mutante)) < taxa
    ruido = rng.normal(0.0, sigma, size=len(mutante))
    mutante[sortear] += ruido[sortear]
    return np.clip(mutante, 0.0, 1.0)


# --------------------------------------------------------------- ELITISMO


def elite(populacao, fitness, n):
    """Devolve copias dos n melhores individuos da geracao."""
    if n <= 0:
        return []
    melhores = np.argsort(fitness)[::-1][:n]
    return [populacao[i].copy() for i in melhores]
