"""
Funcao fitness.

Ponto mais importante do projeto inteiro: o fitness e calculado por
validacao cruzada estratificada SOBRE O CONJUNTO DE TREINO. O conjunto de
teste nunca e tocado durante a evolucao.

Isso corrige uma falha metodologica da Fase 1, onde o melhor valor de k do
KNN foi escolhido medindo o erro no proprio conjunto de teste. Como um
algoritmo genetico avalia milhares de configuracoes, repetir esse padrao
aqui levaria a um overfitting muito mais severo: o AG encontraria o
individuo que melhor decora o conjunto de teste, e as metricas finais nao
significariam nada.

Fitness = w_sens * sensibilidade + w_esp * especificidade - parcimonia

Sobre a escolha da fitness (vale para o relatorio -- foi uma decisao
tomada depois de um erro):

A primeira versao usava 0.6*recall + 0.4*F1. Parecia razoavel, mas o AG
descobriu a solucao degenerada: baixar tanto o limiar que TODOS os
pacientes sao classificados como diabeticos. Isso da recall 1.0 e, numa
base com ~35% de positivos, F1 em torno de 0.52 -- o suficiente para
vencer qualquer modelo honesto. O AG nao estava errado; a funcao fitness
e que premiava um classificador inutil.

O problema do F1 e que ele ignora os verdadeiros negativos: um modelo que
diz "sim" para todo mundo nunca e punido por isso. A especificidade
(quantos saudaveis foram corretamente identificados) fecha essa brecha,
porque a solucao degenerada tem especificidade exatamente zero.

Sensibilidade e especificidade tambem sao o par padrao em medicina
diagnostica, o que torna a escolha facil de defender. Os pesos 0.65/0.35
preservam a prioridade clinica da Fase 1 (falso negativo custa mais que
falso positivo) sem permitir o colapso.

Alem dos pesos, ha uma RESTRICAO explicita: especificidade minima de 0.50.
Um exame de triagem que sinaliza mais da metade dos pacientes saudaveis
como suspeitos sobrecarrega o servico e perde utilidade clinica, por mais
alto que seja o recall. Individuos abaixo desse piso recebem uma
penalidade forte e sao eliminados pela selecao.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from ..data import construir_pipeline, limiar_por_quantil
from .chromosome import decodificar


class AvaliadorFitness:
    """
    Avalia individuos e mantem um cache dos resultados.

    O cache importa: ao longo das geracoes o AG revisita repetidamente
    individuos identicos ou quase identicos (efeito do elitismo e da
    convergencia). Sem cache, boa parte do tempo de execucao seria gasta
    retreinando os mesmos SVMs.
    """

    def __init__(
        self,
        X_train,
        y_train,
        n_splits=5,
        w_sensibilidade=0.65,
        w_especificidade=0.35,
        peso_parcimonia=0.02,
        especificidade_minima=0.50,
        penalidade_restricao=2.0,
        random_state=42,
    ):
        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.w_sensibilidade = w_sensibilidade
        self.w_especificidade = w_especificidade
        self.peso_parcimonia = peso_parcimonia
        self.especificidade_minima = especificidade_minima
        self.penalidade_restricao = penalidade_restricao
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        self._cache = {}
        self.n_avaliacoes = 0
        self.n_acertos_cache = 0

    # ------------------------------------------------------------------

    def _chave(self, genes):
        """Arredonda os genes para agrupar individuos praticamente iguais."""
        return tuple(np.round(genes, 4))

    def avaliar(self, genes):
        chave = self._chave(genes)
        if chave in self._cache:
            self.n_acertos_cache += 1
            return self._cache[chave]["fitness"]
        resultado = self._avaliar_sem_cache(genes)
        self._cache[chave] = resultado
        self.n_avaliacoes += 1
        return resultado["fitness"]

    def detalhes(self, genes):
        """Metricas completas de um individuo (usado no relatorio final)."""
        chave = self._chave(genes)
        if chave not in self._cache:
            self._cache[chave] = self._avaliar_sem_cache(genes)
        return self._cache[chave]

    # ------------------------------------------------------------------

    def _avaliar_sem_cache(self, genes):
        d = decodificar(genes)

        # Individuo sem nenhuma feature nao pode ser treinado. Em vez de
        # descartar, damos fitness zero: ele simplesmente perde todos os
        # torneios e desaparece naturalmente da populacao.
        if not d["mascara"].any():
            return {
                "fitness": 0.0,
                "sensibilidade": 0.0,
                "especificidade": 0.0,
                "f1": 0.0,
                "n_features": 0,
            }

        X = self.X_train.loc[:, d["mascara"]]
        y = self.y_train

        sensibilidades, especificidades, f1s = [], [], []
        for idx_treino, idx_val in self.cv.split(X, y):
            pipeline = construir_pipeline(d["svc_params"])
            pipeline.fit(X.iloc[idx_treino], y[idx_treino])

            # O corte e derivado dos escores do PROPRIO fold de treino e
            # so entao aplicado ao fold de validacao. E exatamente o mesmo
            # procedimento usado no retreino final, o que garante que o
            # ponto de operacao encontrado aqui continue valido depois.
            limiar = limiar_por_quantil(pipeline, X.iloc[idx_treino], d["quantil"])

            # decision_function em vez de predict_proba: sem calibracao de
            # Platt, a avaliacao fica varias vezes mais rapida.
            escores = pipeline.decision_function(X.iloc[idx_val])
            y_pred = (escores >= limiar).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y[idx_val], y_pred, labels=[0, 1]
            ).ravel()

            sensibilidades.append(recall_score(y[idx_val], y_pred, zero_division=0))
            especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            f1s.append(f1_score(y[idx_val], y_pred, zero_division=0))

        sensibilidade = float(np.mean(sensibilidades))
        especificidade = float(np.mean(especificidades))
        f1 = float(np.mean(f1s))
        n_features = int(d["mascara"].sum())

        # Penalidade de parcimonia: entre dois individuos equivalentes,
        # prefere o que usa menos exames clinicos. Isso tem valor pratico
        # real -- menos exames significa triagem mais barata e mais rapida.
        penalidade = self.peso_parcimonia * (n_features / len(d["mascara"]))

        # Restricao de especificidade minima.
        falta = max(0.0, self.especificidade_minima - especificidade)
        penalidade += self.penalidade_restricao * falta

        fitness = (
            self.w_sensibilidade * sensibilidade
            + self.w_especificidade * especificidade
            - penalidade
        )

        return {
            "fitness": float(fitness),
            "sensibilidade": sensibilidade,
            "especificidade": especificidade,
            "f1": f1,
            "n_features": n_features,
        }
