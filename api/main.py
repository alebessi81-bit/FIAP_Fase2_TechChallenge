"""
API de triagem de diabetes.

Rotas:
    GET  /health    -- verificacao de vida, usada pelo Cloud Run
    GET  /metrics   -- metricas de servico e de modelo
    POST /predict   -- classificacao + contribuicoes (sem LLM, rapido)
    POST /explain   -- o mesmo, com interpretacao em linguagem natural

Por que /predict e /explain sao rotas separadas, e nao um parametro:
elas tem perfis de desempenho e de falha completamente diferentes.
/predict roda em milissegundos, e deterministica e nao depende de nada
externo. /explain leva segundos, custa cota e depende de um servico de
terceiros. Separadas, uma indisponibilidade do Gemini nunca derruba a
triagem -- que e a funcao critica. Fosse um parametro numa rota so, a
tentacao de tratar as duas com o mesmo timeout e o mesmo alerta seria
grande, e o resultado seria um acoplamento desnecessario.

O modelo e treinado na subida do processo, a partir do CSV embarcado na
imagem. Leva cerca de um segundo e evita carregar um .pkl, que traria
dependencia da versao exata do scikit-learn usada para serializar.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    PacienteEntrada,
    RespostaExplicacao,
    RespostaPredicao,
    RespostaSaude,
)
from src.llm.avaliacao import pontuacao_automatica, verificacoes_automaticas
from src.llm.cliente import ClienteGemini, ErroLLM
from src.llm.explicador import ExplicadorDiagnostico
from src.modelo_final import CAMPEAO
from src.monitoring.logs import configurar_logging
from src.monitoring.metricas import metricas

logger = logging.getLogger("api")

CAMINHO_DADOS = os.getenv("CAMINHO_DADOS", "data/diabetes.csv")
MODELO_VERSAO = f"ag-{CAMPEAO['origem'].lower().replace(' ', '-')}"

estado = {"explicador": None}


@asynccontextmanager
async def ciclo_de_vida(app):
    configurar_logging(os.getenv("LOG_LEVEL", "INFO"))
    inicio = time.time()
    logger.info("Treinando o modelo campeao", extra={"evento": "startup_inicio"})

    estado["explicador"] = ExplicadorDiagnostico(CAMINHO_DADOS)

    logger.info(
        "Modelo pronto",
        extra={
            "evento": "startup_fim",
            "duracao_ms": round((time.time() - inicio) * 1000, 1),
            "modelo_versao": MODELO_VERSAO,
            "features": estado["explicador"].features,
            "ponto_de_corte": round(estado["explicador"].limiar, 4),
        },
    )
    yield
    logger.info("Encerrando", extra={"evento": "shutdown"})


app = FastAPI(
    title="Triagem de Diabetes - Tech Challenge Fase 2",
    description=(
        "Modelo SVM otimizado por algoritmo genetico, com interpretacao "
        "dos resultados em linguagem natural via LLM."
    ),
    version="2.0.0",
    lifespan=ciclo_de_vida,
)


# ---------------------------------------------------------------------
# Middleware: um request_id por chamada, log de entrada e saida, metricas
# ---------------------------------------------------------------------


@app.middleware("http")
async def observar(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    inicio = time.perf_counter()

    try:
        resposta = await call_next(request)
        status = resposta.status_code
    except Exception:
        # Loga a excecao com o request_id antes de deixar subir, senao a
        # falha aparece no Cloud Logging sem vinculo com a requisicao.
        duracao = (time.perf_counter() - inicio) * 1000
        metricas.registrar_requisicao(request.url.path, duracao, 500)
        logger.exception(
            "Erro nao tratado",
            extra={"request_id": request_id, "rota": request.url.path},
        )
        raise

    duracao = (time.perf_counter() - inicio) * 1000
    metricas.registrar_requisicao(request.url.path, duracao, status)
    resposta.headers["X-Request-ID"] = request_id

    # /health e chamado o tempo todo pelo Cloud Run; logar tudo poluiria.
    if request.url.path != "/health":
        logger.info(
            "Requisicao concluida",
            extra={
                "request_id": request_id,
                "rota": request.url.path,
                "metodo": request.method,
                "status": status,
                "duracao_ms": round(duracao, 2),
            },
        )
    return resposta


# ---------------------------------------------------------------------


def _explicador():
    if estado["explicador"] is None:
        raise HTTPException(status_code=503, detail="Modelo ainda carregando")
    return estado["explicador"]


def _para_dataframe(paciente: PacienteEntrada, explicador):
    """
    Converte a entrada da API no formato que o modelo espera.

    O zero vira NaN nas colunas onde ele significa ausencia, reproduzindo
    exatamente a convencao usada no treino. Fazer isso diferente aqui
    seria treino-servico divergente: o modelo veria, em producao, um
    valor que nunca viu no treino.
    """
    dados = paciente.model_dump()
    colunas_com_zero_invalido = {"Glucose", "Insulin", "BMI"}

    linha = {}
    for feature in explicador.features:
        valor = dados.get(feature)
        if valor == 0 and feature in colunas_com_zero_invalido:
            valor = None
        linha[feature] = np.nan if valor is None else float(valor)

    return pd.DataFrame([linha], columns=explicador.features)


def _avaliar(explicador, paciente):
    """Roda o modelo e monta a evidencia, sem chamar o LLM."""
    X = _para_dataframe(paciente, explicador)
    z = explicador._pre.transform(X)[0]
    escore, contribuicoes = explicador._contribuicoes(z)

    fatores = []
    n_ausentes = 0
    for feature, contribuicao in sorted(
        contribuicoes.items(), key=lambda kv: abs(kv[1]), reverse=True
    ):
        valor = X.iloc[0][feature]
        ausente = bool(pd.isna(valor))
        n_ausentes += int(ausente)
        rotulo, unidade = explicador.ROTULOS_PUBLICOS[feature]
        fatores.append(
            {
                "exame": rotulo,
                "valor": None if ausente else round(float(valor), 2),
                "unidade": unidade,
                "valor_imputado": ausente,
                "percentil_na_populacao": explicador._percentil(feature, valor),
                "contribuicao": round(contribuicao, 4),
                "direcao": "aumenta o risco"
                if contribuicao > 0
                else "reduz o risco",
            }
        )

    risco_elevado = escore >= explicador.limiar
    metricas.registrar_predicao(escore, risco_elevado, n_ausentes)

    return {
        "classificacao": "risco elevado" if risco_elevado else "risco baixo",
        "escore": round(escore, 4),
        "ponto_de_corte": round(explicador.limiar, 4),
        "distancia_do_corte": round(escore - explicador.limiar, 4),
        "fatores": fatores,
    }, n_ausentes


# ---------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------


@app.get("/health", response_model=RespostaSaude)
def saude():
    """Verificacao de vida. O Cloud Run usa esta rota para decidir se a
    instancia pode receber trafego."""
    return {
        "status": "ok" if estado["explicador"] else "carregando",
        "modelo_carregado": estado["explicador"] is not None,
        "modelo_versao": MODELO_VERSAO,
        "uptime_segundos": round(time.time() - metricas.inicio, 1),
    }


@app.get("/metrics")
def obter_metricas():
    """Metricas de servico e de modelo desta instancia."""
    resumo = metricas.resumo()
    resumo["modelo_versao"] = MODELO_VERSAO
    return resumo


@app.post("/predict", response_model=RespostaPredicao)
def prever(paciente: PacienteEntrada, request: Request):
    """Classificacao e contribuicoes por exame. Sem LLM, sem rede."""
    explicador = _explicador()
    resultado, n_ausentes = _avaliar(explicador, paciente)

    logger.info(
        "Predicao realizada",
        extra={
            "request_id": request.state.request_id,
            "evento": "predicao",
            "classificacao": resultado["classificacao"],
            "escore": resultado["escore"],
            "campos_imputados": n_ausentes,
            "modelo_versao": MODELO_VERSAO,
        },
    )

    return {
        **resultado,
        "modelo_versao": MODELO_VERSAO,
        "request_id": request.state.request_id,
    }


@app.post("/explain", response_model=RespostaExplicacao)
def explicar(paciente: PacienteEntrada, request: Request):
    """Classificacao mais interpretacao em linguagem natural."""
    explicador = _explicador()
    resultado, _ = _avaliar(explicador, paciente)

    evidencia = {
        "condicao_avaliada": "diabetes tipo 2",
        "populacao_de_referencia": (
            "Pima Indians Diabetes Dataset -- mulheres de origem Pima, "
            "com 21 anos ou mais"
        ),
        "classificacao_do_modelo": resultado["classificacao"],
        "escore": resultado["escore"],
        "ponto_de_corte": resultado["ponto_de_corte"],
        "distancia_do_corte": resultado["distancia_do_corte"],
        "fatores": resultado["fatores"],
        "exames_nao_usados_pelo_modelo": explicador.descartados,
        "desempenho_do_modelo": {
            "sensibilidade_validacao_cruzada": CAMPEAO["sensibilidade_cv"],
            "especificidade_validacao_cruzada": CAMPEAO["especificidade_cv"],
        },
    }

    from src.llm.prompts import montar_prompt

    try:
        cliente = ClienteGemini(simular=os.getenv("LLM_SIMULAR") == "1")
        texto = cliente.gerar(montar_prompt(evidencia))
    except ErroLLM as erro:
        # 503 e nao 500: o problema e uma dependencia externa
        # indisponivel, e a chamada pode dar certo mais tarde. A
        # distincao importa para quem consome a API decidir se tenta de
        # novo, e para os alertas nao apontarem para o nosso codigo.
        logger.warning(
            "LLM indisponivel",
            extra={"request_id": request.state.request_id, "erro": str(erro)},
        )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Interpretacao indisponivel: {erro}. A classificacao "
                "continua acessivel em /predict."
            ),
        ) from erro

    verificacoes = verificacoes_automaticas(texto, evidencia)
    acertos, total = pontuacao_automatica(verificacoes)

    logger.info(
        "Explicacao gerada",
        extra={
            "request_id": request.state.request_id,
            "evento": "explicacao",
            "pontuacao_automatica": f"{acertos}/{total}",
            "verificacoes_falhas": [
                c for c in verificacoes if verificacoes.get(c) is False
            ],
        },
    )

    return {
        **resultado,
        "modelo_versao": MODELO_VERSAO,
        "request_id": request.state.request_id,
        "explicacao": texto,
        "verificacoes_automaticas": verificacoes,
        "pontuacao_automatica": f"{acertos}/{total}",
    }
