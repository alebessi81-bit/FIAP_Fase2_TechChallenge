"""
Contratos de entrada e saida da API.

Sobre a validacao de faixas: os limites abaixo nao sao decorativos. O
modelo foi treinado numa faixa especifica de valores e nao tem como saber
que uma glicose de 5000 e erro de digitacao -- ele produziria um escore
com aparencia normal. Rejeitar na porta de entrada e mais seguro do que
devolver uma predicao sem sentido.

Sobre o zero: no dataset original, zero em Glicose, Insulina ou IMC
significa "nao medido", nao o valor zero. A API reproduz essa convencao,
mas aceita tambem null, que e a forma correta de dizer isso. As duas
entram no modelo como ausentes e sao imputadas.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PacienteEntrada(BaseModel):
    """
    Dados de um paciente.

    As tres ultimas features nao sao usadas pelo modelo campeao -- o
    algoritmo genetico as descartou. Sao aceitas mesmo assim, para que o
    sistema que chama a API nao precise saber quais features a versao
    atual do modelo usa. Se um modelo futuro voltar a precisar delas, o
    contrato nao muda.
    """

    Pregnancies: int = Field(..., ge=0, le=25, description="Numero de gestacoes")
    Glucose: Optional[float] = Field(
        None, ge=0, le=400, description="Glicose plasmatica em mg/dL; 0 ou null = ausente"
    )
    Insulin: Optional[float] = Field(
        None, ge=0, le=1000, description="Insulina serica em mu U/mL; 0 ou null = ausente"
    )
    BMI: Optional[float] = Field(
        None, ge=0, le=80, description="Indice de massa corporal; 0 ou null = ausente"
    )
    Age: int = Field(..., ge=21, le=120, description="Idade em anos")

    BloodPressure: Optional[float] = Field(
        None, ge=0, le=200, description="Nao usado pelo modelo atual"
    )
    SkinThickness: Optional[float] = Field(
        None, ge=0, le=100, description="Nao usado pelo modelo atual"
    )
    DiabetesPedigreeFunction: Optional[float] = Field(
        None, ge=0, le=3, description="Nao usado pelo modelo atual"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 2,
                "Glucose": 189,
                "Insulin": 846,
                "BMI": 30.1,
                "Age": 59,
            }
        }
    }


class Fator(BaseModel):
    exame: str
    valor: Optional[float]
    unidade: str
    valor_imputado: bool
    percentil_na_populacao: Optional[float]
    contribuicao: float
    direcao: str


class RespostaPredicao(BaseModel):
    classificacao: str
    escore: float
    ponto_de_corte: float
    distancia_do_corte: float
    fatores: list[Fator]
    modelo_versao: str
    request_id: str


class RespostaExplicacao(RespostaPredicao):
    explicacao: str
    verificacoes_automaticas: dict
    pontuacao_automatica: str


class RespostaSaude(BaseModel):
    status: str
    modelo_carregado: bool
    modelo_versao: str
    uptime_segundos: float
