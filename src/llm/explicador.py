"""
Construcao da evidencia que alimenta o LLM.

O ponto central deste modulo: o LLM NAO ve o modelo nem os dados brutos.
Ele recebe um dicionario fechado, com numeros ja calculados, e a unica
tarefa dele e transformar isso em texto. Toda a estatistica acontece aqui,
em Python, de forma auditavel.

Essa separacao e o que torna a solucao defensavel num contexto clinico: se
o texto estiver errado, o erro esta na redacao, nao na conta.

CONTRIBUICAO POR OCLUSAO
-------------------------
Para saber o quanto cada exame pesou na decisao de um paciente especifico,
usamos oclusao: substituimos o valor daquele exame pela media da populacao
(zero, no espaco padronizado) e medimos o quanto o escore se move.

    contribuicao_j = f(paciente) - f(paciente com o exame j na media)

Contribuicao positiva significa que o valor daquele paciente empurrou o
escore na direcao de "risco elevado".

Escolhemos oclusao em vez de SHAP KernelExplainer por dois motivos
praticos: e deterministica (nao depende de amostragem, entao dois
relatorios do mesmo paciente nunca divergem) e roda em milissegundos, o
que permite explicar pacientes sob demanda numa API. A contrapartida e que
a oclusao nao distribui o credito de features correlacionadas tao bem
quanto o SHAP -- limitacao que vale registrar no relatorio.
"""

import numpy as np

from ..data import (
    FEATURE_NAMES,
    carregar_dados,
    construir_pipeline,
    limiar_por_quantil,
    separar_treino_teste,
)
from ..modelo_final import CAMPEAO

# Como cada exame e chamado no texto, e a unidade dele.
ROTULOS = {
    "Pregnancies": ("Numero de gestacoes", ""),
    "Glucose": ("Glicose", "mg/dL"),
    "BloodPressure": ("Pressao diastolica", "mm Hg"),
    "SkinThickness": ("Prega cutanea triciptal", "mm"),
    "Insulin": ("Insulina serica", "mu U/mL"),
    "BMI": ("IMC", "kg/m2"),
    "DiabetesPedigreeFunction": ("Indice de historico familiar", ""),
    "Age": ("Idade", "anos"),
}


class ExplicadorDiagnostico:
    def __init__(self, caminho_csv, config=None):
        self.config = config or CAMPEAO
        self.features = self.config["features"]

        dados = carregar_dados(caminho_csv)
        X_train, X_test, y_train, y_test = separar_treino_teste(dados)

        self.X_train = X_train[self.features]
        self.X_test = X_test[self.features]
        self.X_test_completo = X_test
        self.y_test = y_test

        self.pipeline = construir_pipeline(self.config["svc_params"])
        self.pipeline.fit(self.X_train, y_train)

        self.limiar = limiar_por_quantil(
            self.pipeline, self.X_train, self.config["quantil"]
        )

        # Expostos para a API reaproveitar sem duplicar a logica.
        self.ROTULOS_PUBLICOS = ROTULOS
        self.descartados = [
            ROTULOS[f][0] for f in FEATURE_NAMES if f not in self.features
        ]

        # Espaco padronizado e imputado, onde a oclusao acontece.
        self._pre = self.pipeline[:-1]
        self._svc = self.pipeline[-1]
        self._Z_train = self._pre.transform(self.X_train)

    # ------------------------------------------------------------------

    def _percentil(self, feature, valor):
        """Posicao do valor do paciente na populacao de treino."""
        if valor is None or np.isnan(valor):
            return None
        coluna = self.X_train[feature].dropna()
        return round(100.0 * float((coluna < valor).mean()), 1)

    def _contribuicoes(self, z):
        """Contribuicao de cada feature por oclusao."""
        escore = float(self._svc.decision_function(z.reshape(1, -1))[0])
        contribuicoes = {}
        for i, feature in enumerate(self.features):
            z_ocluso = z.copy()
            z_ocluso[i] = 0.0  # media da populacao no espaco padronizado
            escore_sem = float(
                self._svc.decision_function(z_ocluso.reshape(1, -1))[0]
            )
            contribuicoes[feature] = escore - escore_sem
        return escore, contribuicoes

    # ------------------------------------------------------------------

    def construir_evidencia(self, indice):
        """
        Monta o dicionario de evidencias de um paciente do conjunto de teste.

        'indice' e a posicao dentro de X_test (0, 1, 2, ...).
        """
        linha_bruta = self.X_test_completo.iloc[indice]
        linha = self.X_test.iloc[[indice]]
        z = self._pre.transform(linha)[0]

        escore, contribuicoes = self._contribuicoes(z)
        margem = escore - self.limiar
        risco_elevado = escore >= self.limiar

        fatores = []
        for feature, contribuicao in sorted(
            contribuicoes.items(), key=lambda kv: abs(kv[1]), reverse=True
        ):
            valor = linha_bruta[feature]
            ausente = valor is None or np.isnan(valor)
            rotulo, unidade = ROTULOS[feature]
            fatores.append(
                {
                    "exame": rotulo,
                    "valor": None if ausente else round(float(valor), 2),
                    "unidade": unidade,
                    "valor_imputado": bool(ausente),
                    "percentil_na_populacao": self._percentil(feature, valor),
                    "contribuicao": round(contribuicao, 4),
                    "direcao": "aumenta o risco"
                    if contribuicao > 0
                    else "reduz o risco",
                }
            )

        return {
            "condicao_avaliada": "diabetes tipo 2",
            "populacao_de_referencia": (
                "Pima Indians Diabetes Dataset -- mulheres de origem Pima, "
                "com 21 anos ou mais"
            ),
            "classificacao_do_modelo": "risco elevado"
            if risco_elevado
            else "risco baixo",
            "escore": round(escore, 4),
            "ponto_de_corte": round(self.limiar, 4),
            "distancia_do_corte": round(margem, 4),
            "fatores": fatores,
            "exames_nao_usados_pelo_modelo": self.descartados,
            "desempenho_do_modelo": {
                "sensibilidade_validacao_cruzada": self.config[
                    "sensibilidade_cv"
                ],
                "especificidade_validacao_cruzada": self.config[
                    "especificidade_cv"
                ],
                "observacao": (
                    "Metricas obtidas em validacao cruzada de 5 folds sobre "
                    "o conjunto de treino. O modelo foi calibrado para "
                    "priorizar sensibilidade."
                ),
            },
        }

    # ------------------------------------------------------------------

    def explicar(self, cliente, indice):
        """Gera a explicacao em linguagem natural de um paciente."""
        from .prompts import montar_prompt

        evidencia = self.construir_evidencia(indice)
        prompt = montar_prompt(evidencia)
        texto = cliente.gerar(prompt)
        return evidencia, texto
