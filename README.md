# My Chance — NLP / Matching

Serviço de ranqueamento de candidatos para vagas. Executa filtragem por competências obrigatórias (com tolerância configurável), cobertura ponderada dos requisitos, aderência por similaridade de cosseno, experiência, projetos e senioridade, e devolve um ranking explicável. Chamado pelo **backend-services** via HTTP.

## Requisitos

- Python 3.12+

## Execução local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Health check |
| POST | `/api/v1/matching/rank` | Ranqueia candidatos para uma vaga |

Contrato de request/response: `docs/api-contracts.md` no repositório **backend-services**.

## Testes

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t mychance-nlp .
docker run --rm -p 8000:8000 mychance-nlp
```

## Estrutura

```
app/
├── main.py
├── pipeline.py
├── schemas.py
└── matching/
    ├── parametros.py         # parâmetros ajustáveis (sobrescritos por `parametros` na requisição)
    ├── taxonomia.py          # sinônimos e crédito parcial entre competências relacionadas
    ├── weighted_scoring.py   # etapa 1: filtro de obrigatórias + cobertura ponderada
    ├── cosine_similarity.py  # etapa 2: aderência (cosseno no espaço da vaga)
    └── ranking.py            # combinação dos componentes, desempate e explicação
```

## Score

```
score = gate × fator_senioridade × (0.55·cobertura + 0.20·aderência + 0.15·experiência + 0.10·projetos)
```

Campos opcionais novos no contrato: `vaga.competencias.*.nivel_alvo`, `parametros` (request) e `componentes`/`explicacao` (response). `vocabulario` passou a ser opcional e é ignorado.
