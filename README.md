# UAV Motor Failure Detection

![Demo do deploy — detecção de falha em tempo real](assets/demo.gif)

---

## Visão geral

Sistema de detecção de falhas de motor em VANTs (Veículos Aéreos Não Tripulados) de asa fixa, desenvolvido com abordagem não-supervisionada. O modelo analisa dados de telemetria em tempo real e emite um alerta de anomalia assim que os padrões de voo divergem do comportamento normal — sem exigir exemplos rotulados de falha para treinar.

**Melhor resultado:** Isolation Forest · all_features · F1 = 0.86 · Latência de detecção = 0.14 s

---

## Objetivo

Detectar a falha do motor o mais cedo possível após ela ocorrer, usando exclusivamente dados de sensores disponíveis a bordo (IMU, magnetômetro, GPS, pressão, throttle, etc.). O modelo deve generalizar entre voos distintos e operar com latência compatível com sistemas de emergência de bordo.

---

## Dataset

**ALFA — Autonomous aircraft Loss-of-control Flight Analysis**

| Atributo | Detalhe |
|---|---|
| Aeronave | CarbonZ (asa fixa) |
| Período | Julho–Outubro de 2018 |
| Voos com falha | Falha de motor induzida, com e sem trajetória de emergência (EMR traj) |
| Voos normais | Voos de referência sem falha |
| Formato | CSVs por voo em `aeroespacial-2/data/03_primary/` |
| Fonte dos sinais | Tópicos ROS mergeados por `merge_asof` no tempo |

O dataset contém ~30 voos cobrindo diferentes cenários de falha. Cada linha representa um instante de tempo e cada coluna um canal de sensor ou feature derivada.

---

## Instalação do ambiente virtual

O projeto usa [Poetry](https://python-poetry.org/) para gerenciar dependências. Execute todos os comandos a partir da **raiz do repositório** (onde fica o `pyproject.toml`).

### Pré-requisito: Poetry instalado

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verifique a instalação:

```bash
poetry --version
```

### Instalar as dependências

```bash
poetry install
```

Isso cria o ambiente virtual em `.venv/` e instala o pacote `aeroespacial_2` em modo editável, tornando-o importável nos notebooks e nas pipelines.

### Ativar o ambiente

```bash
poetry shell
```

Ou prefixar qualquer comando com `poetry run <comando>`.

---

## Pipelines Kedro

O projeto possui dois tracks de pipelines, cada um com sua estratégia de features. Todos os comandos abaixo são executados na **raiz do repositório**.

### Track 1 — All Features (features físicas)

Pipeline completa com features baseadas em física de voo (energia específica, altitude, erro de controle, etc.). Documentada nos notebooks `aeroespacial-2/notebooks/all_features/`.

```
data_ingestion → data_preparation → feature_engineering → model_training
```

| Pipeline | Comando | O que faz |
|---|---|---|
| `data_ingestion` | `kedro run --pipeline=data_ingestion` | Merge dos tópicos ROS de cada voo em um único DataFrame alinhado no tempo; filtragem de colunas de ruído |
| `data_preparation` | `kedro run --pipeline=data_preparation` | Renomeia colunas ROS, remove redundâncias, descarta o primeiro segundo e cria 6 features de erro (comandado − medido) |
| `feature_engineering` | `kedro run --pipeline=feature_engineering` | Calcula features físicas: energia específica total, variação de altitude e rolling statistics sobre todos os sinais |
| `model_training` | `kedro run --pipeline=model_training` | Seleção de features, treinamento do Isolation Forest e avaliação por voo; artefatos salvos em `data/06_models/` |

Para rodar a pipeline completa de uma vez:

```bash
kedro run
```

### Track 2 — FFT Features (features espectrais)

Pipeline alternativa com features espectrais sobre os 7 sinais com vínculo físico direto à frequência de rotação do motor (IMU, magnetômetro, velocidade do ar). Documentada nos notebooks `aeroespacial-2/notebooks/fft_features/`.

```
fft_ingestion → fft_data_preparation → fft_feature_engineering → fft_model_training
```

| Pipeline | Comando | O que faz |
|---|---|---|
| `fft_ingestion` | `kedro run --pipeline=fft_ingestion` | Carrega apenas os sinais com periodicidade ligada ao motor |
| `fft_data_preparation` | `kedro run --pipeline=fft_data_preparation` | Mesmo fluxo do track 1, restrito ao subconjunto FFT |
| `fft_feature_engineering` | `kedro run --pipeline=fft_feature_engineering` | Features espectrais (FFT) e rolling statistics sobre os 7 sinais do motor |
| `fft_model_training` | `kedro run --pipeline=fft_model_training` | Treinamento e avaliação do Isolation Forest sobre features espectrais |

### Visualizar o grafo de pipelines

```bash
kedro viz
```

### Rodar os notebooks

```bash
poetry run jupyter lab
```

Os notebooks estão em `aeroespacial-2/notebooks/` e já têm acesso ao pacote `aeroespacial_2` após o `poetry install`. Cada notebook documenta uma etapa da pipeline com exemplos passo a passo e visualizações.

---

## Deploy

A API de inferência em tempo real está em `aeroespacial-2/deploy/`. Consulte o [README do deploy](aeroespacial-2/deploy/README.md) para instruções de como subir o servidor local e o container Docker.

---

## Estrutura

```
.
├── pyproject.toml                  # dependências e configuração Kedro
├── aeroespacial-2/
│   ├── conf/                       # parâmetros e catálogo do Kedro
│   ├── data/
│   │   ├── 03_primary/             # voos preparados (all features)
│   │   ├── 04_feature/             # features engineered (all features)
│   │   ├── 03_primary_fft/         # voos preparados (FFT)
│   │   ├── 04_feature_fft/         # features engineered (FFT)
│   │   ├── 06_models/              # modelos e scalers treinados
│   │   └── 07_model_output/        # métricas de avaliação
│   ├── notebooks/
│   │   ├── all_features/           # exploração e documentação do track 1
│   │   └── fft_features/           # exploração e documentação do track 2
│   ├── src/aeroespacial_2/         # código das pipelines Kedro
│   └── deploy/                     # API FastAPI + interface web
└── assets/
    └── demo.gif                    # demo da interface de detecção em tempo real
```
