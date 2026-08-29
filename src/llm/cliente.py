"""
Cliente do Gemini.

Duas decisoes de projeto que valem explicar no relatorio:

1. MODO SIMULADO. O cliente aceita simular=True e devolve uma resposta
   pre-fabricada sem chamar a API. Serve para testar o pipeline inteiro
   (montagem do prompt, parsing, avaliacao automatica) sem gastar cota e
   sem depender de rede. E como o codigo foi validado antes de existir
   uma chave.

2. COMPATIBILIDADE DE API. O SDK do Google mudou a interface principal:
   a versao nova usa client.interactions.create(), a anterior usava
   client.models.generate_content(). O metodo gerar() tenta a nova e cai
   para a antiga automaticamente, entao o codigo funciona nas duas.
"""

import os
import re
import time

MODELO_PADRAO = os.getenv("GEMINI_MODELO", "gemini-3.5-flash")


class ErroLLM(RuntimeError):
    pass


class ErroCota(ErroLLM):
    """Limite de requisicoes atingido (HTTP 429)."""


# Erros que nao adiantam retentar: nome de modelo errado, chave invalida,
# prompt rejeitado. Retentar so faz o usuario esperar a toa.
PADROES_DEFINITIVOS = (
    "not_found",
    "404",
    "api key not valid",
    "api_key_invalid",
    "permission_denied",
)


def _e_definitivo(erro):
    texto = str(erro).lower()
    return any(padrao in texto for padrao in PADROES_DEFINITIVOS)


def _espera_sugerida(erro):
    """
    Extrai da mensagem de erro o tempo que a API pede para esperar.

    A API responde 429 com algo como "Please retry in 3.45s". Respeitar
    esse valor e melhor que adivinhar: um backoff fixo curto demais so
    consome tentativas, e longo demais faz o usuario esperar a toa.
    """
    achado = re.search(r"retry in ([\d.]+)s", str(erro))
    if achado:
        return float(achado.group(1)) + 1.0
    return None


class ClienteGemini:
    def __init__(
        self,
        api_key=None,
        modelo=None,
        temperatura=0.2,
        simular=False,
        max_tentativas=3,
    ):
        self.modelo = modelo or MODELO_PADRAO
        # Temperatura baixa de proposito: o objetivo aqui e fidelidade aos
        # numeros do modelo, nao criatividade. Temperatura alta aumenta o
        # risco de o LLM inventar valores que nao estao na evidencia.
        self.temperatura = temperatura
        self.simular = simular
        self.max_tentativas = max_tentativas
        self.chamadas = 0
        self._cliente = None

        if not simular:
            chave = api_key or os.getenv("GEMINI_API_KEY")
            if not chave:
                raise ErroLLM(
                    "Chave nao encontrada. Defina a variavel de ambiente "
                    "GEMINI_API_KEY ou rode com --simular."
                )
            try:
                from google import genai
            except ImportError as erro:
                raise ErroLLM(
                    "SDK nao instalado. Rode: pip install google-genai"
                ) from erro
            self._cliente = genai.Client(api_key=chave)

    # ------------------------------------------------------------------

    def gerar(self, prompt):
        """Envia o prompt e devolve o texto da resposta."""
        if self.simular:
            self.chamadas += 1
            return _resposta_simulada(prompt)

        ultimo_erro = None
        for tentativa in range(self.max_tentativas):
            try:
                texto = self._chamar(prompt)
                self.chamadas += 1
                return texto
            except Exception as erro:  # noqa: BLE001
                ultimo_erro = erro
                if _e_definitivo(erro):
                    raise ErroLLM(str(erro)) from erro
                if tentativa < self.max_tentativas - 1:
                    # Preferimos o tempo que a propria API sugeriu; sem
                    # ele, backoff exponencial.
                    espera = _espera_sugerida(erro) or (2 ** tentativa * 3)
                    time.sleep(min(espera, 60))

        if "429" in str(ultimo_erro) or "quota" in str(ultimo_erro).lower():
            raise ErroCota(
                "Cota do nivel gratuito esgotada para o modelo "
                f"{self.modelo}. Opcoes: aguardar a renovacao da cota, "
                "rodar com --sem-juiz (metade das chamadas) ou trocar de "
                "modelo com $env:GEMINI_MODELO. Use listar_modelos.py "
                "para ver os modelos disponiveis na sua chave."
            )
        raise ErroLLM(f"Falha apos {self.max_tentativas} tentativas: {ultimo_erro}")

    def _chamar(self, prompt):
        # Interface nova do SDK.
        if hasattr(self._cliente, "interactions"):
            try:
                interacao = self._cliente.interactions.create(
                    model=self.modelo, input=prompt
                )
                texto = getattr(interacao, "output_text", None)
                if texto:
                    return texto
                return interacao.outputs[-1].text
            except (AttributeError, TypeError):
                pass

        # Interface anterior, mantida como alternativa.
        resposta = self._cliente.models.generate_content(
            model=self.modelo, contents=prompt
        )
        return resposta.text


# ---------------------------------------------------------------------
# Resposta simulada -- usada apenas para testar o encanamento offline.
# Ela e deliberadamente bem formatada, entao NAO serve para avaliar a
# qualidade real do LLM. Serve so para confirmar que o pipeline roda.
# ---------------------------------------------------------------------

_SIMULADA = """RESUMO
Paciente classificado pelo modelo como de risco elevado para diabetes tipo 2,
com escore acima do ponto de corte definido para triagem.

FATORES QUE MAIS PESARAM
- Glicose: valor acima da mediana da populacao de referencia, principal
  contribuinte para o escore.
- BMI: valor elevado, segundo maior contribuinte.
- Idade: contribuicao moderada no mesmo sentido.

O QUE ISSO SUGERE NA PRATICA
Recomenda-se confirmacao laboratorial por glicemia de jejum ou HbA1c antes
de qualquer conduta. O resultado isolado nao substitui avaliacao clinica.

LIMITACOES
Modelo de triagem com sensibilidade de 93% e especificidade de 58% em
validacao cruzada. A especificidade moderada significa que uma parcela
relevante dos sinalizados nao tera a doenca."""


def _resposta_simulada(prompt):
    if "AVALIE" in prompt.upper():
        return (
            '{"fidelidade": 4, "clareza": 4, "acionabilidade": 4, '
            '"seguranca": 5, "comentario": "Resposta simulada para teste."}'
        )
    return _SIMULADA
