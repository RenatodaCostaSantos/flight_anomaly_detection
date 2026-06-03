FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.1.4

# Copia arquivos de dependência ANTES do código.
# Por que? Docker cacheia cada instrução como uma camada.
# Se só o código mudou (não as deps), essa camada reutiliza o cache
# e não reinstala pacotes — build muito mais rápido.
COPY pyproject.toml poetry.lock* ./

# Instala só deps de produção, sem criar virtualenv dentro do container
# (o container já é o isolamento)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY app/ ./app/

# Boa prática de segurança: nunca rodar como root em produção
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]