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


def _payload_basico() -> dict:
    return {
        "vaga": {
            "senioridade": 2,
            "competencias": {
                "python": {"peso": 5, "obrigatoria": True, "nivel_min": 3, "nivel_alvo": 4},
                "postgresql": {"peso": 4, "obrigatoria": True, "nivel_min": 2},
                "docker": {"peso": 3},
            },
        },
        "candidatos": [
            {"candidato_id": "usr_quase", "competencias": {"python": 2, "postgresql": 4, "docker": 4}, "senioridade": 2},
            {"candidato_id": "usr_mysql", "competencias": {"python": 4, "mysql": 4, "docker": 3}, "senioridade": 2},
        ],
    }


def test_matching_rank_sem_vocabulario_retorna_componentes_e_explicacao() -> None:
    response = client.post("/api/v1/matching/rank", json=_payload_basico())
    assert response.status_code == 200

    ranking = {item["candidato_id"]: item for item in response.json()["ranking"]}
    mysql = ranking["usr_mysql"]
    assert mysql["aprovado_filtragem"] is True
    assert {"cobertura", "aderencia", "experiencia", "projetos", "gate", "senioridade"} <= set(mysql["componentes"])
    assert any("via mysql" in motivo for motivo in mysql["explicacao"])
    assert ranking["usr_quase"]["compatibilidade"] == "Eliminado"


def test_matching_rank_tolerancia_configuravel() -> None:
    payload = _payload_basico()
    payload["parametros"] = {"tolerancia_obrigatoria": 1}

    response = client.post("/api/v1/matching/rank", json=payload)
    assert response.status_code == 200

    quase = next(item for item in response.json()["ranking"] if item["candidato_id"] == "usr_quase")
    assert quase["aprovado_filtragem"] is True
    assert quase["componentes"]["gate"] < 1.0
    assert quase["compatibilidade_score"] > 0


def test_matching_rank_rejeita_parametros_fora_dos_limites() -> None:
    payload = _payload_basico()
    payload["parametros"] = {"penalidade_senioridade_acima": 0.5}
    assert client.post("/api/v1/matching/rank", json=payload).status_code == 422


def test_matching_rank_rejeita_ids_duplicados() -> None:
    payload = _payload_basico()
    payload["candidatos"][1]["candidato_id"] = "usr_quase"
    assert client.post("/api/v1/matching/rank", json=payload).status_code == 422
