# My Chance — NLP / Matching

Serviço de ranqueamento de candidatos para vagas. Executa filtragem por competências obrigatórias e ranking por similaridade de cosseno. Chamado pelo **backend-services** via HTTP.

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
    ├── weighted_scoring.py
    └── cosine_similarity.py
```
