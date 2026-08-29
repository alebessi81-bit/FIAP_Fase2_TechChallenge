# Imagem slim: a completa passa de 1 GB e o Cloud Run cobra tempo de
# cold start. A slim fica em torno de 400 MB com scikit-learn e pandas.
FROM python:3.12-slim

# PYTHONUNBUFFERED garante que os logs saiam na hora, e nao em blocos.
# Sem isso, no Cloud Run as linhas aparecem atrasadas e fora de ordem.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# As dependencias vao numa camada separada do codigo. Assim, mudar uma
# linha de Python nao invalida o cache do pip, e o rebuild cai de
# minutos para segundos.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY data/ ./data/

# Usuario sem privilegios: se a aplicacao for comprometida, o atacante
# nao e root dentro do container.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# O Cloud Run injeta a porta pela variavel PORT e ignora o EXPOSE; o 8080
# aqui e so o padrao para execucao local.
ENV PORT=8080
EXPOSE 8080

# Uma unica worker por container e proposital: o Cloud Run escala criando
# mais instancias, nao mais processos dentro da mesma. Varios workers
# multiplicariam o modelo na memoria sem ganho de vazao.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1
