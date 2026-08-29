"""
Testes automatizados.

Rode com:
    pytest -v

Os testes da API sao pulados automaticamente se o FastAPI nao estiver
instalado, para que os demais continuem rodando.

Cada teste marcado com REGRESSAO corresponde a um bug que realmente
aconteceu durante o desenvolvimento. Sao os mais valiosos do arquivo:
todos os outros verificam o que eu imaginei que poderia quebrar, esses
verificam o que de fato quebrou.
"""

import numpy as np
import pandas as pd
import pytest

from src.genetic.chromosome import N_GENES, decodificar, populacao_inicial
from src.genetic.operators import (
    cruzamento_blend,
    cruzamento_uniforme,
    elite,
    mutacao_gaussiana,
    selecao_torneio,
)
from src.llm.avaliacao import (
    classificacao_consistente,
    pontuacao_automatica,
    verificacoes_automaticas,
)
from src.monitoring.metricas import RegistroMetricas

RNG = np.random.default_rng(42)


# =====================================================================
# Cromossomo
# =====================================================================


def test_genes_sempre_produzem_modelo_valido():
    """
    Qualquer vetor em [0,1]^13 tem que virar um SVM treinavel.

    Essa propriedade e o que permite ao AG nunca desperdicar avaliacoes
    com individuos invalidos.
    """
    for _ in range(200):
        genes = RNG.random(N_GENES)
        d = decodificar(genes)
        p = d["svc_params"]
        assert p["kernel"] in ("rbf", "poly", "sigmoid")
        assert 0.01 <= p["C"] <= 100.0
        assert 1e-4 <= p["gamma"] <= 10.0
        assert p["class_weight"] in (None, "balanced")
        assert 0.20 <= d["quantil"] <= 0.90


def test_limites_do_cromossomo():
    minimo = decodificar(np.zeros(N_GENES))
    maximo = decodificar(np.ones(N_GENES))
    assert minimo["svc_params"]["C"] == pytest.approx(0.01)
    assert maximo["svc_params"]["C"] == pytest.approx(100.0)
    assert minimo["quantil"] == pytest.approx(0.20)
    assert maximo["quantil"] == pytest.approx(0.90)


def test_gene_de_corte_e_quantil_e_nao_valor_absoluto():
    """
    REGRESSAO. A primeira versao guardava o limiar como valor absoluto do
    decision_function. Ele era calibrado na validacao cruzada e aplicado
    ao modelo retreinado com 100% do treino, onde a escala e outra --
    resultado: todos os pacientes classificados como diabeticos.

    A chave 'limiar' nao pode voltar a existir no cromossomo.
    """
    d = decodificar(RNG.random(N_GENES))
    assert "quantil" in d
    assert "limiar" not in d
    assert 0.0 < d["quantil"] < 1.0


def test_teto_de_iteracoes_presente():
    """
    REGRESSAO. Sem max_iter, combinacoes patologicas de kernel e C faziam
    o SVM rodar indefinidamente e travavam o experimento inteiro.
    """
    d = decodificar(RNG.random(N_GENES))
    assert d["svc_params"]["max_iter"] > 0


# =====================================================================
# Operadores geneticos
# =====================================================================


def test_operadores_respeitam_o_intervalo():
    """Nenhum operador pode produzir gene fora de [0,1]."""
    pop = populacao_inicial(20, RNG)
    for _ in range(50):
        f1, f2 = cruzamento_blend(pop[0], pop[1], RNG, taxa=1.0)
        for filho in (f1, f2):
            mutante = mutacao_gaussiana(filho, RNG, taxa=1.0, sigma=0.5)
            assert mutante.min() >= 0.0
            assert mutante.max() <= 1.0


def test_torneio_escolhe_individuo_da_populacao():
    pop = populacao_inicial(10, RNG)
    fitness = RNG.random(10)
    escolhido = selecao_torneio(pop, fitness, RNG, k=3)
    assert any(np.allclose(escolhido, individuo) for individuo in pop)


def test_elitismo_devolve_os_melhores():
    pop = populacao_inicial(5, RNG)
    fitness = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    melhores = elite(pop, fitness, 2)
    assert np.allclose(melhores[0], pop[1])
    assert np.allclose(melhores[1], pop[3])


def test_cruzamento_uniforme_preserva_genes_dos_pais():
    pai1 = np.zeros(N_GENES)
    pai2 = np.ones(N_GENES)
    f1, f2 = cruzamento_uniforme(pai1, pai2, RNG, taxa=1.0)
    # Cada gene tem que vir de um dos pais, sem valores intermediarios.
    assert set(np.unique(np.concatenate([f1, f2]))).issubset({0.0, 1.0})


# =====================================================================
# Avaliacao das explicacoes
# =====================================================================


@pytest.fixture
def evidencia():
    return {
        "condicao_avaliada": "diabetes tipo 2",
        "populacao_de_referencia": (
            "Pima Indians Diabetes Dataset -- mulheres de origem Pima, "
            "com 21 anos ou mais"
        ),
        "classificacao_do_modelo": "risco elevado",
        "escore": 0.8653,
        "ponto_de_corte": 0.8486,
        "distancia_do_corte": 0.0167,
        "fatores": [
            {"exame": "IMC", "valor": 39.7, "percentil_na_populacao": 86.8},
            {"exame": "Insulina serica", "valor": 105.0,
             "percentil_na_populacao": 38.9},
            {"exame": "Numero de gestacoes", "valor": 2.0,
             "percentil_na_populacao": 32.1},
        ],
        "desempenho_do_modelo": {
            "sensibilidade_validacao_cruzada": 0.93,
            "especificidade_validacao_cruzada": 0.5825,
        },
    }


TEXTO_BOM = """RESUMO
Paciente classificado como risco elevado. Escore 0,0167 acima do corte.

FATORES QUE MAIS PESARAM
- IMC 39,7 kg/m2, acima de 86,8% da referencia.
- Insulina sérica 105,0 mu U/mL, acima de 38,9%.
- Número de gestações 2,0, acima de 32,1%.

O QUE ISSO SUGERE NA PRATICA
Investigacao confirmatoria com HbA1c. A prega cutanea triciptal e o
indice de historico familiar nao foram usados por este modelo e devem ser
avaliados clinicamente pelo profissional responsavel pelo atendimento.

LIMITACOES
Sensibilidade de 93% e especificidade de 58,25% em validacao cruzada: em
100 pacientes saudaveis cerca de 42 seriam sinalizados sem ter a doenca.
A populacao de referencia sao mulheres de origem Pima com 21 anos ou
mais, e os percentis citados nao valem fora desse grupo. Trata-se de
triagem e nao de diagnostico definitivo."""


def test_texto_bom_passa_em_todas_as_verificacoes(evidencia):
    v = verificacoes_automaticas(TEXTO_BOM, evidencia)
    acertos, total = pontuacao_automatica(v)
    assert acertos == total, [c for c in v if v.get(c) is False]


def test_acento_nao_causa_falso_positivo(evidencia):
    """
    REGRESSAO. Os rotulos internos sao sem acento ("Insulina serica") e o
    LLM escreve portugues correto ("Insulina sérica"). A comparacao
    ingenua acusava ausencia de um fator que estava citado.
    """
    v = verificacoes_automaticas(TEXTO_BOM, evidencia)
    assert v["cita_top3_fatores"]


def test_percentual_derivado_e_aceito(evidencia):
    """
    REGRESSAO. 58,25% de especificidade autoriza o texto a falar em ~42%
    de falsos positivos. Arredondar cedo demais e exigir coincidencia
    exata acusava esses numeros como inventados.
    """
    v = verificacoes_automaticas(TEXTO_BOM, evidencia)
    assert v["sem_numeros_inventados"], v["numeros_suspeitos"]


def test_numero_dentro_de_string_e_aceito(evidencia):
    """
    REGRESSAO. O "21" de "com 21 anos ou mais" vive dentro de uma string
    da evidencia. A extracao so olhava valores numericos e acusava o 21
    como inventado.
    """
    v = verificacoes_automaticas(TEXTO_BOM, evidencia)
    assert 21.0 not in v["numeros_suspeitos"]


def test_numero_realmente_inventado_e_detectado(evidencia):
    texto = TEXTO_BOM.replace("86,8%", "99,4%")
    v = verificacoes_automaticas(texto, evidencia)
    assert not v["sem_numeros_inventados"]


def test_contradicao_com_o_modelo_e_detectada(evidencia):
    """
    REGRESSAO. Um paciente classificado como risco BAIXO recebeu um texto
    afirmando risco ELEVADO e as verificacoes passaram todas -- nenhuma
    delas conferia se o texto concordava com o modelo.
    """
    texto = TEXTO_BOM.replace(
        "classificado como risco elevado", "classificado como risco baixo"
    )
    assert not classificacao_consistente(texto, evidencia)


def test_afirmacao_de_diagnostico_e_bloqueada(evidencia):
    texto = TEXTO_BOM.replace(
        "Trata-se de", "O paciente tem diabetes. Trata-se de"
    )
    v = verificacoes_automaticas(texto, evidencia)
    assert not v["sem_expressoes_proibidas"]


def test_omissao_da_populacao_de_referencia_e_detectada(evidencia):
    texto = TEXTO_BOM.replace(
        "A populacao de referencia sao mulheres de origem Pima com 21 anos ou\nmais, e os percentis citados nao valem fora desse grupo. ",
        "",
    )
    v = verificacoes_automaticas(texto, evidencia)
    assert not v["declara_populacao_referencia"]


# =====================================================================
# Metricas
# =====================================================================


def test_metricas_contabilizam_erros_e_latencia():
    m = RegistroMetricas()
    for i in range(10):
        m.registrar_requisicao("/predict", 10.0 + i, 200 if i < 8 else 500)
    r = m.resumo()
    assert r["servico"]["requisicoes_total"] == 10
    assert r["servico"]["erros_total"] == 2
    assert r["servico"]["taxa_erro"] == pytest.approx(0.2)
    assert r["servico"]["latencia_ms"]["p50"] is not None


def test_metricas_de_modelo_rastreiam_taxa_de_risco():
    m = RegistroMetricas()
    for i in range(10):
        m.registrar_predicao(escore=0.9, risco_elevado=i < 4, n_ausentes=0)
    r = m.resumo()
    assert r["modelo"]["taxa_risco_elevado"] == pytest.approx(0.4)


def test_metricas_sem_dados_nao_quebram():
    r = RegistroMetricas().resumo()
    assert r["servico"]["taxa_erro"] == 0.0
    assert r["modelo"]["taxa_risco_elevado"] is None


# =====================================================================
# API
# =====================================================================

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="FastAPI nao instalado"
)


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c


PACIENTE_RISCO = {
    "Pregnancies": 2,
    "Glucose": 189,
    "Insulin": 846,
    "BMI": 30.1,
    "Age": 59,
}

PACIENTE_SAUDAVEL = {
    "Pregnancies": 0,
    "Glucose": 73,
    "Insulin": None,
    "BMI": 21.1,
    "Age": 25,
}


def test_health_responde_ok(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_carregado"] is True


def test_predict_classifica_paciente_de_risco(cliente):
    r = cliente.post("/predict", json=PACIENTE_RISCO)
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["classificacao"] in ("risco elevado", "risco baixo")
    assert len(corpo["fatores"]) == 5
    assert "X-Request-ID" in r.headers


def test_predict_marca_valor_ausente_como_imputado(cliente):
    r = cliente.post("/predict", json=PACIENTE_SAUDAVEL)
    assert r.status_code == 200
    insulina = [
        f for f in r.json()["fatores"] if "nsulina" in f["exame"]
    ][0]
    assert insulina["valor_imputado"] is True
    assert insulina["valor"] is None


def test_zero_e_tratado_como_ausente(cliente):
    """
    Zero em Glicose significa "nao medido" no dataset original. Se a API
    tratasse como valor real, o modelo veria em producao algo que nunca
    viu no treino.
    """
    entrada = dict(PACIENTE_RISCO, Glucose=0)
    r = cliente.post("/predict", json=entrada)
    assert r.status_code == 200
    glicose = [f for f in r.json()["fatores"] if "licose" in f["exame"]][0]
    assert glicose["valor_imputado"] is True


def test_valor_fora_da_faixa_e_rejeitado(cliente):
    r = cliente.post("/predict", json=dict(PACIENTE_RISCO, Glucose=5000))
    assert r.status_code == 422


def test_campo_obrigatorio_ausente_e_rejeitado(cliente):
    r = cliente.post("/predict", json={"Glucose": 120})
    assert r.status_code == 422


def test_metrics_reflete_as_predicoes(cliente):
    antes = cliente.get("/metrics").json()["modelo"]["predicoes_total"]
    cliente.post("/predict", json=PACIENTE_RISCO)
    depois = cliente.get("/metrics").json()["modelo"]["predicoes_total"]
    assert depois == antes + 1


def test_explain_funciona_em_modo_simulado(cliente, monkeypatch):
    monkeypatch.setenv("LLM_SIMULAR", "1")
    r = cliente.post("/explain", json=PACIENTE_RISCO)
    assert r.status_code == 200
    corpo = r.json()
    assert "explicacao" in corpo
    assert "/" in corpo["pontuacao_automatica"]


def test_explain_devolve_503_sem_chave(cliente, monkeypatch):
    """
    Falha de dependencia externa e 503, nao 500. A distincao diz a quem
    consome a API que vale a pena tentar de novo, e evita que o alerta
    aponte para o nosso codigo.
    """
    monkeypatch.delenv("LLM_SIMULAR", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = cliente.post("/explain", json=PACIENTE_RISCO)
    assert r.status_code == 503
    assert "/predict" in r.json()["detail"]
