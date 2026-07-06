from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_matching_rank_filters_mandatory_gaps() -> None:
    payload = {
        "vocabulario": ["python", "sql", "docker"],
        "vaga": {
            "competencias": {
                "python": {"peso": 5, "obrigatoria": True, "nivel_min": 3},
                "sql": {"peso": 4, "obrigatoria": True, "nivel_min": 3},
            }
        },
        "candidatos": [
            {
                "candidato_id": "usr_weak",
                "competencias": {"python": 1, "sql": 5},
                "projetos": 1,
                "anos_experiencia": 1.0,
            },
            {
                "candidato_id": "usr_strong",
                "competencias": {"python": 5, "sql": 5, "docker": 4},
                "projetos": 2,
                "anos_experiencia": 2.0,
            },
        ],
    }

    response = client.post("/api/v1/matching/rank", json=payload)
    assert response.status_code == 200

    ranking = response.json()["ranking"]
    assert len(ranking) == 2
    assert ranking[0]["candidato_id"] == "usr_strong"
    assert ranking[0]["compatibilidade_score"] > 0
    assert ranking[1]["candidato_id"] == "usr_weak"
    assert ranking[1]["compatibilidade_score"] == 0.0
    assert ranking[1]["aprovado_filtragem"] is False
