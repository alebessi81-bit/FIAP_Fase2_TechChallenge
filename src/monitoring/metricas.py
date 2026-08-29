"""
Coleta de metricas em memoria.

Duas categorias, e a segunda e a que costuma faltar em trabalhos de ML:

METRICAS DE SERVICO
    Contagem de requisicoes, taxa de erro, latencia (p50, p95, p99).
    Respondem "a API esta de pe e rapida?".

METRICAS DE MODELO
    Proporcao de pacientes classificados como risco elevado, distribuicao
    dos escores e proporcao de campos ausentes na entrada. Respondem "o
    modelo continua vendo o mesmo tipo de paciente que viu no treino?".

A segunda existe porque um modelo pode degradar silenciosamente sem que
nada quebre: se a taxa de risco elevado sair dos ~40% observados em
validacao para 80%, ou a API esta recebendo uma populacao diferente, ou
algo mudou no pre-processamento. Nenhum monitoramento de infraestrutura
detectaria isso -- a latencia continuaria otima e o codigo de status
continuaria 200.

LIMITACAO IMPORTANTE (para o relatorio): estas metricas vivem na memoria
do processo. No Cloud Run, com varias instancias e scale-to-zero, cada
instancia tem o proprio contador e tudo se perde quando ela e desligada.
Serve para inspecao pontual e para o video de demonstracao. Em producao
de verdade, os numeros teriam que ir para o Cloud Monitoring, o que e
viavel justamente porque ja emitimos logs estruturados: cada linha de log
pode virar uma metrica baseada em log.
"""

import threading
import time
from collections import deque


class RegistroMetricas:
    def __init__(self, janela=1000):
        # Guardamos apenas as ultimas N latencias: memoria constante,
        # independente de quanto tempo o servico fique no ar.
        self._latencias = deque(maxlen=janela)
        self._escores = deque(maxlen=janela)
        self._trava = threading.Lock()
        self.inicio = time.time()

        self.requisicoes = 0
        self.erros = 0
        self.por_rota = {}
        self.predicoes = 0
        self.risco_elevado = 0
        self.campos_ausentes = 0

    # ------------------------------------------------------------------

    def registrar_requisicao(self, rota, duracao_ms, status):
        with self._trava:
            self.requisicoes += 1
            self._latencias.append(duracao_ms)
            if status >= 400:
                self.erros += 1

            atual = self.por_rota.setdefault(
                rota, {"total": 0, "erros": 0, "latencia_soma": 0.0}
            )
            atual["total"] += 1
            atual["latencia_soma"] += duracao_ms
            if status >= 400:
                atual["erros"] += 1

    def registrar_predicao(self, escore, risco_elevado, n_ausentes):
        with self._trava:
            self.predicoes += 1
            self._escores.append(escore)
            if risco_elevado:
                self.risco_elevado += 1
            self.campos_ausentes += n_ausentes

    # ------------------------------------------------------------------

    @staticmethod
    def _percentil(valores, p):
        if not valores:
            return None
        ordenados = sorted(valores)
        indice = min(int(len(ordenados) * p / 100.0), len(ordenados) - 1)
        return round(ordenados[indice], 2)

    def resumo(self):
        with self._trava:
            latencias = list(self._latencias)
            escores = list(self._escores)
            rotas = {
                nome: {
                    "total": d["total"],
                    "erros": d["erros"],
                    "latencia_media_ms": round(
                        d["latencia_soma"] / d["total"], 2
                    ),
                }
                for nome, d in self.por_rota.items()
            }
            requisicoes = self.requisicoes
            erros = self.erros
            predicoes = self.predicoes
            risco_elevado = self.risco_elevado
            campos_ausentes = self.campos_ausentes
            inicio = self.inicio

        return {
            "servico": {
                "uptime_segundos": round(time.time() - inicio, 1),
                "requisicoes_total": requisicoes,
                "erros_total": erros,
                "taxa_erro": round(erros / requisicoes, 4) if requisicoes else 0.0,
                "latencia_ms": {
                    "p50": self._percentil(latencias, 50),
                    "p95": self._percentil(latencias, 95),
                    "p99": self._percentil(latencias, 99),
                },
                "por_rota": rotas,
            },
            "modelo": {
                "predicoes_total": predicoes,
                "risco_elevado_total": risco_elevado,
                "taxa_risco_elevado": round(risco_elevado / predicoes, 4)
                if predicoes
                else None,
                "taxa_risco_elevado_esperada": 0.40,
                "escore_p50": self._percentil(escores, 50),
                "escore_p95": self._percentil(escores, 95),
                "campos_ausentes_total": campos_ausentes,
                "observacao": (
                    "Uma taxa de risco elevado muito distante da esperada "
                    "sugere que a populacao atendida difere da populacao "
                    "de treino."
                ),
            },
        }


metricas = RegistroMetricas()
