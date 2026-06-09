# Motor Failure Detection — Deploy

Deploy local da API de detecção de falhas de motor em UAVs.

**Modelo:** Isolation Forest · all_features · F1 = 0.86 · Latência = 0.14s

---

## Pré-requisito: esteja sempre nesta pasta

Todos os comandos abaixo devem ser executados a partir de `aeroespacial-2/deploy/`:

```bash
cd aeroespacial-2/deploy
```

---

## Desenvolvimento local (sem Docker)

```bash
# Instalar dependências
poetry install

# Subir o servidor com hot-reload
poetry run uvicorn app.main:app --reload
```

Acesse a interface web em **http://localhost:8000**

---

## Interface web

1. Abra http://localhost:8000 no browser
2. Selecione um CSV de `data/03_primary/`
3. Ajuste a velocidade de simulação (50× é um bom ponto de partida)
4. Clique **Iniciar** — o gráfico de anomaly score evolui em tempo real

---

## Simulação pelo terminal

```bash
poetry run python simulate.py ../data/03_primary/<arquivo>.csv --speed 50
```

Exemplo com voo de falha:

```bash
poetry run python simulate.py \
  ../data/03_primary/carbonZ_2018-07-18-15-53-31_1_engine_failure.csv \
  --speed 50
```

---

## Testes

```bash
# Testes rápidos (carregamento do modelo, validação de schema)
poetry run pytest tests/

# Testes completos com voos reais (mais lentos)
poetry run pytest tests/ -m slow
```

---

## Docker

O Docker está instalado via Snap — use `sudo` em todos os comandos:

```bash
# Construir a imagem
sudo docker compose build

# Subir o container em background
sudo docker compose up -d

# Acompanhar os logs
sudo docker compose logs -f

# Parar o container
sudo docker compose down
```

A API ficará disponível em **http://localhost:8000** (mesmo endereço que o modo local).

> Se aparecer `[Errno 98] Address already in use`, um container anterior ainda está rodando.
> Execute `sudo docker compose down` antes de subir novamente.

---

## Estrutura

```
deploy/
├── app/
│   ├── main.py          # FastAPI: /health, /predict, /stream (SSE), GET /
│   ├── inference.py     # Pipeline de inferência (replica o Kedro)
│   ├── schemas.py       # Contratos Pydantic de entrada/saída
│   └── static/
│       └── index.html   # Interface web (Chart.js, streaming em tempo real)
├── tests/
│   └── test_predict.py  # Testes de integração com TestClient
├── simulate.py          # Simulação em tempo real no terminal
├── Dockerfile
└── docker-compose.yml
```

---

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Interface web |
| `GET` | `/health` | Status do servidor e modelo |
| `POST` | `/predict` | Inferência em batch (JSON) |
| `POST` | `/stream` | Inferência via SSE (CSV upload) |
