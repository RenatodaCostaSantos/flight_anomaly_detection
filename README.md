# aeroespacial_2

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Instalação

Este projeto usa [Poetry](https://python-poetry.org/) para gerenciar dependências. Para instalar tudo (incluindo o pacote local, necessário para os notebooks):

```
poetry install
```

## Rodar a pipeline Kedro

```
cd aeroespacial-2
kedro run
```

## Rodar os notebooks

Os notebooks estão em `aeroespacial-2/notebooks/`. Após `poetry install`, o pacote `aeroespacial_2` já estará disponível no ambiente.

## Testes

```
pytest
```

## Estrutura

```
aeroespacial-2/
  conf/        # parâmetros e catálogo do Kedro
  data/        # dados brutos e processados (não versionados)
  notebooks/   # exploração e experimentação
  src/
    aeroespacial_2/   # código das pipelines
```
