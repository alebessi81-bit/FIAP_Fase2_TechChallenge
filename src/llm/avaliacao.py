"""
Avaliacao da qualidade das interpretacoes geradas.

O enunciado pede explicitamente para "avaliar a qualidade das
interpretacoes geradas". Fazemos isso em duas camadas independentes,
porque cada uma pega um tipo diferente de problema:

CAMADA 1 - VERIFICACOES AUTOMATICAS (deterministicas)
    Regras objetivas que rodam sem custo e sem LLM. Pegam falhas de
    formato e, o mais importante, numeros inventados. A verificacao de
    ancoragem numerica e a mais valiosa das duas camadas: ela extrai todo
    numero do texto e confere se ele aparece na evidencia. Num contexto
    clinico, um valor alucinado e a falha mais grave possivel.

CAMADA 2 - LLM COMO AVALIADOR
    Uma segunda chamada, com prompt de auditoria, nota o texto de 1 a 5 em
    fidelidade, clareza, acionabilidade e seguranca. Captura problemas
    subjetivos que regra nenhuma pega -- texto tecnicamente correto mas
    confuso, ou uma recomendacao vaga demais para ser util.

Limitacao que precisa constar no relatorio: usar um LLM para avaliar outro
LLM tem vies conhecido, especialmente quando os dois sao do mesmo modelo.
A camada 1 existe justamente para nao depender so disso.
"""

import json
import re
import unicodedata

SECOES_OBRIGATORIAS = [
    "RESUMO",
    "FATORES QUE MAIS PESARAM",
    "O QUE ISSO SUGERE NA PRATICA",
    "LIMITACOES",
]

# Expressoes que nao podem aparecer: afirmam diagnostico ou prescrevem.
EXPRESSOES_PROIBIDAS = [
    "tem diabetes",
    "nao tem diabetes",
    "diagnostico confirmado",
    "diagnóstico confirmado",
    "com certeza",
    "definitivamente",
    "prescrevo",
    "receito",
    "iniciar metformina",
]


def _sem_acento(texto):
    """
    Remove acentos para comparar rotulos.

    Necessario porque os rotulos internos estao sem acento
    ("Insulina serica") enquanto o LLM escreve portugues correto
    ("Insulina serica" com acento). Sem essa normalizacao, a verificacao
    acusava ausencia de um fator que estava citado corretamente.
    """
    normalizado = unicodedata.normalize("NFD", texto)
    return "".join(c for c in normalizado if unicodedata.category(c) != "Mn")


def _numeros(texto):
    """Extrai todos os numeros de um texto, como floats."""
    achados = re.findall(r"-?\d+[.,]?\d*", texto)
    valores = []
    for achado in achados:
        try:
            valores.append(float(achado.replace(",", ".")))
        except ValueError:
            continue
    return valores


# Bases retoricas: "em 100 pacientes, cerca de 42..." e a forma natural
# de comunicar uma taxa a um medico. O 100 e o 1000 ai nao sao dados
# citados, sao denominadores. Trata-los como valores inventados punia
# justamente a redacao mais clara.
BASES_RETORICAS = {100.0, 1000.0}


def _numeros_da_evidencia(evidencia):
    """Todos os numeros que o LLM tem permissao de citar."""
    permitidos = set(BASES_RETORICAS)

    def percorrer(no):
        if isinstance(no, dict):
            for valor in no.values():
                percorrer(valor)
        elif isinstance(no, list):
            for item in no:
                percorrer(item)
        elif isinstance(no, str):
            # Numeros escritos dentro de texto tambem sao permitidos. A
            # descricao da populacao de referencia diz "com 21 anos ou
            # mais"; sem esta linha, o 21 citado corretamente pelo LLM
            # era acusado de inventado.
            for numero in _numeros(no):
                permitidos.add(round(numero, 4))
        elif isinstance(no, (int, float)) and no is not None:
            # Precisao de 4 casas, e nao 2: arredondar cedo demais
            # quebrava os derivados. Uma especificidade de 0.5825 virava
            # 0.58 e gerava 58.0 em vez de 58.25, fazendo a verificacao
            # acusar como inventado um numero que estava correto.
            permitidos.add(round(float(no), 4))

    percorrer(evidencia)

    # Derivados legitimos: valores convertidos para percentual e seus
    # complementos. Uma especificidade de 0.5825 autoriza o texto a falar
    # em 58,25% de acerto entre saudaveis e em ~42% de falsos positivos.
    derivados = set()
    for valor in permitidos:
        if 0.0 <= valor <= 1.0:
            # Proporcao: autoriza a forma percentual e o complemento dela.
            derivados.add(round(valor * 100, 4))
            derivados.add(round(100.0 - valor * 100, 4))
        elif 0.0 <= valor <= 100.0:
            # Ja e percentual: autoriza so o complemento.
            derivados.add(round(100.0 - valor, 4))

    # Nota: derivar "100 - valor" tambem para proporcoes era um erro. Uma
    # especificidade de 0.5825 gerava o permitido 99.4175, e com isso
    # qualquer percentil inventado na casa dos 99 passava despercebido.
    # Um teste de regressao pegou exatamente esse caso.
    return permitidos | derivados


def _corresponde(numero, permitidos, tol_rel=0.005, tol_abs=0.5):
    """
    Confere se um numero citado bate com algum valor permitido, com
    tolerancia.

    A tolerancia existe porque o LLM arredonda de forma legitima ao
    escrever: 41,75% vira "aproximadamente 42%". Exigir coincidencia
    exata transformaria boa redacao em falha de fidelidade.

    Mas ela precisa ser apertada. Com tolerancia relativa de 2%, um
    percentil inventado de 99,4 caia dentro da margem de 98,33 (um
    derivado legitimo) e passava despercebido. Meia unidade cobre o
    arredondamento decimal sem abrir essa brecha.

    Limitacao conhecida: quanto mais numeros a evidencia tiver, maior a
    chance de um valor inventado coincidir por acaso com algum permitido.
    A verificacao reduz o risco de alucinacao, nao o elimina.
    """
    for permitido in permitidos:
        if abs(numero - permitido) <= max(tol_abs, abs(permitido) * tol_rel):
            return True
    return False


def _resumo(texto):
    """
    Recorta apenas a secao RESUMO.

    A checagem de consistencia precisa olhar so para o veredito, e nao
    para o texto inteiro: e legitimo que a secao de limitacoes mencione as
    duas classificacoes ao explicar os erros do modelo.
    """
    inicio = texto.upper().find("RESUMO")
    fim = texto.upper().find("FATORES")
    if inicio == -1:
        return texto[:400].lower()
    if fim == -1 or fim < inicio:
        fim = inicio + 400
    return texto[inicio:fim].lower()


def classificacao_consistente(texto, evidencia):
    """
    O texto concorda com o veredito do modelo?

    Esta e a verificacao mais importante do conjunto, e foi adicionada
    depois de um teste em modo simulado revelar a lacuna: um paciente
    classificado como risco BAIXO recebeu um texto afirmando risco
    ELEVADO, e as outras verificacoes passaram todas. Um laudo que
    contradiz o proprio modelo e a falha mais grave possivel aqui, e era
    a unica que nenhuma regra pegava.
    """
    resumo = _resumo(texto)
    classificacao = evidencia["classificacao_do_modelo"].lower()
    oposta = "risco baixo" if "elevado" in classificacao else "risco elevado"
    return classificacao in resumo and oposta not in resumo


def verificacoes_automaticas(texto, evidencia):
    """Roda as regras deterministicas e devolve o resultado de cada uma."""
    texto_maiusculo = texto.upper()
    texto_minusculo = texto.lower()

    secoes_presentes = [s for s in SECOES_OBRIGATORIAS if s in texto_maiusculo]
    proibidas = [e for e in EXPRESSOES_PROIBIDAS if e in texto_minusculo]

    # Ancoragem numerica.
    permitidos = _numeros_da_evidencia(evidencia)
    citados = _numeros(texto)
    # Numeros pequenos costumam ser ordinais de lista ou anos; ignoramos
    # os inteiros de 0 a 10 para nao gerar falso positivo.
    suspeitos = [
        n
        for n in citados
        if not _corresponde(n, permitidos) and not (0 <= n <= 10 and n == int(n))
    ]

    palavras = len(texto.split())

    # As tres features de maior contribuicao precisam ser citadas.
    top3 = [f["exame"] for f in evidencia["fatores"][:3]]
    texto_comparavel = _sem_acento(texto_minusculo)
    citadas = [r for r in top3 if _sem_acento(r.lower()) in texto_comparavel]

    return {
        "classificacao_consistente": classificacao_consistente(texto, evidencia),
        "classificacao_do_modelo": evidencia["classificacao_do_modelo"],
        "todas_secoes_presentes": len(secoes_presentes) == len(SECOES_OBRIGATORIAS),
        "secoes_faltando": [
            s for s in SECOES_OBRIGATORIAS if s not in secoes_presentes
        ],
        "sem_expressoes_proibidas": not proibidas,
        "expressoes_proibidas_encontradas": proibidas,
        "sem_numeros_inventados": not suspeitos,
        "numeros_suspeitos": suspeitos[:10],
        "cita_top3_fatores": len(citadas) == 3,
        "fatores_top3_nao_citados": [r for r in top3 if r not in citadas],
        "extensao_palavras": palavras,
        "extensao_adequada": 130 <= palavras <= 340,
        "menciona_limitacoes": "sensibilidade" in texto_minusculo
        or "especificidade" in texto_minusculo,
        # Verificacao adicionada depois que o LLM avaliador apontou, nos
        # tres pacientes testados, que os textos omitiam a restricao da
        # populacao de referencia. As sete regras anteriores passaram
        # todas por cima dessa falha -- e um caso em que a camada
        # subjetiva enxergou o que a deterministica nao tinha como ver.
        "declara_populacao_referencia": "pima" in _sem_acento(texto_minusculo),
    }


def pontuacao_automatica(verificacoes):
    """Resume as verificacoes num placar de 0 a 6."""
    criterios = [
        "classificacao_consistente",
        "todas_secoes_presentes",
        "sem_expressoes_proibidas",
        "sem_numeros_inventados",
        "cita_top3_fatores",
        "extensao_adequada",
        "menciona_limitacoes",
        "declara_populacao_referencia",
    ]
    return sum(1 for c in criterios if verificacoes.get(c)), len(criterios)


def avaliar_com_llm(cliente, evidencia, texto):
    """Segunda camada: pede ao LLM que audite o proprio texto."""
    from .prompts import montar_prompt_avaliacao

    bruto = cliente.gerar(montar_prompt_avaliacao(evidencia, texto))
    limpo = bruto.strip().removeprefix("```json").removeprefix("```")
    limpo = limpo.removesuffix("```").strip()

    try:
        return json.loads(limpo)
    except json.JSONDecodeError:
        # O modelo pode devolver o JSON dentro de texto. Tentamos recortar.
        casamento = re.search(r"\{.*\}", limpo, re.DOTALL)
        if casamento:
            try:
                return json.loads(casamento.group())
            except json.JSONDecodeError:
                pass
        return {"erro": "resposta nao pode ser interpretada", "bruto": bruto[:300]}
