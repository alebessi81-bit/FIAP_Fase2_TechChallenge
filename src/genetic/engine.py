"""
Motor do algoritmo genetico.

Laco evolutivo classico:

    populacao inicial aleatoria
    repetir por N geracoes:
        avaliar o fitness de cada individuo
        preservar a elite
        enquanto a nova populacao nao estiver cheia:
            selecionar dois pais
            cruzar
            mutar os filhos
        substituir a populacao
"""

import time

import numpy as np
import pandas as pd

from .chromosome import descrever, populacao_inicial
from .operators import CRUZAMENTOS, SELECOES, elite, mutacao_gaussiana


class AlgoritmoGenetico:
    def __init__(
        self,
        avaliador,
        tamanho_populacao=30,
        n_geracoes=20,
        taxa_mutacao=0.10,
        taxa_cruzamento=0.80,
        selecao="torneio",
        k_torneio=3,
        cruzamento="uniforme",
        n_elite=2,
        sigma_mutacao=0.15,
        random_state=42,
        verbose=True,
    ):
        self.avaliador = avaliador
        self.tamanho_populacao = tamanho_populacao
        self.n_geracoes = n_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_cruzamento = taxa_cruzamento
        self.selecao = SELECOES[selecao]
        self.nome_selecao = selecao
        self.k_torneio = k_torneio
        self.cruzamento = CRUZAMENTOS[cruzamento]
        self.nome_cruzamento = cruzamento
        self.n_elite = n_elite
        self.sigma_mutacao = sigma_mutacao
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        self.melhor_individuo = None
        self.melhor_fitness = -np.inf
        self.historico = []

    # ------------------------------------------------------------------

    def _avaliar_populacao(self, populacao):
        return np.array([self.avaliador.avaliar(ind) for ind in populacao])

    def _nova_geracao(self, populacao, fitness):
        nova = elite(populacao, fitness, self.n_elite)
        while len(nova) < self.tamanho_populacao:
            pai1 = self.selecao(populacao, fitness, self.rng, self.k_torneio)
            pai2 = self.selecao(populacao, fitness, self.rng, self.k_torneio)
            filho1, filho2 = self.cruzamento(
                pai1, pai2, self.rng, self.taxa_cruzamento
            )
            nova.append(
                mutacao_gaussiana(
                    filho1, self.rng, self.taxa_mutacao, self.sigma_mutacao
                )
            )
            if len(nova) < self.tamanho_populacao:
                nova.append(
                    mutacao_gaussiana(
                        filho2, self.rng, self.taxa_mutacao, self.sigma_mutacao
                    )
                )
        return np.array(nova)

    # ------------------------------------------------------------------

    def executar(self):
        inicio = time.time()
        populacao = populacao_inicial(self.tamanho_populacao, self.rng)

        for geracao in range(self.n_geracoes):
            fitness = self._avaliar_populacao(populacao)

            indice_melhor = int(np.argmax(fitness))
            if fitness[indice_melhor] > self.melhor_fitness:
                self.melhor_fitness = float(fitness[indice_melhor])
                self.melhor_individuo = populacao[indice_melhor].copy()

            self.historico.append(
                {
                    "geracao": geracao,
                    "melhor_fitness": float(fitness.max()),
                    "fitness_medio": float(fitness.mean()),
                    "desvio_fitness": float(fitness.std()),
                    "melhor_global": self.melhor_fitness,
                }
            )

            if self.verbose:
                print(
                    f"  Geracao {geracao + 1:3d}/{self.n_geracoes} | "
                    f"melhor {fitness.max():.4f} | "
                    f"media {fitness.mean():.4f} | "
                    f"desvio {fitness.std():.4f}"
                )

            if geracao < self.n_geracoes - 1:
                populacao = self._nova_geracao(populacao, fitness)

        self.tempo_execucao = time.time() - inicio

        if self.verbose:
            detalhes = self.avaliador.detalhes(self.melhor_individuo)
            print(f"\n  Concluido em {self.tempo_execucao:.1f}s")
            print(f"  Melhor fitness: {self.melhor_fitness:.4f}")
            print(f"  Sensibilidade (CV): {detalhes['sensibilidade']:.4f} | "
                  f"Especificidade (CV): {detalhes['especificidade']:.4f}")
            print(f"  {descrever(self.melhor_individuo)}")

        return self.melhor_individuo

    # ------------------------------------------------------------------

    def historico_df(self):
        """Historico como DataFrame, pronto para plotar a convergencia."""
        return pd.DataFrame(self.historico)

    def configuracao(self):
        """Configuracao usada, para a tabela comparativa do relatorio."""
        return {
            "populacao": self.tamanho_populacao,
            "geracoes": self.n_geracoes,
            "taxa_mutacao": self.taxa_mutacao,
            "taxa_cruzamento": self.taxa_cruzamento,
            "selecao": self.nome_selecao,
            "cruzamento": self.nome_cruzamento,
            "n_elite": self.n_elite,
        }
