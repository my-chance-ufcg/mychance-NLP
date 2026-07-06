"""
tests/test_weighted_scoring.py
==============================
Testes unitários para o módulo weighted_scoring.py
Execute com: pytest tests/ -v
"""

import math
import pytest

from app.matching.weighted_scoring import (
    Vaga,
    Candidato,
    ConfigCompetencia,
    calcular_fator_recencia,
    calcular_score_ponderado,
    rankear_candidatos,
    NIVEL_MAXIMO,
    DECAIMENTO_RECENCIA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vaga_basica():
    return Vaga(
        id=1,
        titulo="Analista de Dados",
        competencias={
            "python": {"peso": 5, "obrigatoria": True,  "nivel_min": 3},
            "sql":    {"peso": 4, "obrigatoria": True,  "nivel_min": 2},
            "docker": {"peso": 3, "obrigatoria": False, "nivel_min": 1},
            "powerbi":{"peso": 2, "obrigatoria": False, "nivel_min": 1},
        },
    )


@pytest.fixture
def candidato_completo():
    return Candidato(
        id=1, nome_anonimo="CAND-001",
        competencias={"python": 4, "sql": 5, "docker": 3, "powerbi": 2},
        projetos=0, anos_experiencia=0.0,
    )


@pytest.fixture
def candidato_sem_obrigatorias():
    """Candidato com python abaixo do mínimo (3) e sql abaixo do mínimo (2)."""
    return Candidato(
        id=2, nome_anonimo="CAND-002",
        competencias={"python": 1, "sql": 0, "docker": 5, "powerbi": 5},
        projetos=0, anos_experiencia=0.0,
    )


# ---------------------------------------------------------------------------
# calcular_fator_recencia
# ---------------------------------------------------------------------------

class TestFatorRecencia:
    def test_zero_anos_retorna_um(self):
        assert calcular_fator_recencia(0.0) == pytest.approx(1.0)

    def test_negativo_tratado_como_zero(self):
        assert calcular_fator_recencia(-5.0) == pytest.approx(1.0)

    def test_decaimento_exponencial(self):
        esperado = math.exp(-DECAIMENTO_RECENCIA * 5)
        assert calcular_fator_recencia(5.0) == pytest.approx(esperado, rel=1e-6)

    def test_fator_sempre_positivo(self):
        for anos in [0, 1, 10, 50, 100]:
            assert calcular_fator_recencia(float(anos)) > 0


# ---------------------------------------------------------------------------
# calcular_score_ponderado
# ---------------------------------------------------------------------------

class TestCalcularScorePonderado:
    def test_retorna_chaves_esperadas(self, candidato_completo, vaga_basica):
        resultado = calcular_score_ponderado(candidato_completo, vaga_basica)
        chaves = {
            "candidato_id", "nome_anonimo", "score_base_normalizado",
            "score_final", "ausencias_obrigatorias",
            "aprovado_filtragem", "detalhes_por_competencia",
        }
        assert chaves.issubset(resultado.keys())

    def test_candidato_perfeito_score_alto(self, vaga_basica):
        """Candidato com nível máximo em tudo → score próximo de 1."""
        perfeito = Candidato(
            id=99, nome_anonimo="CAND-MAX",
            competencias={c: NIVEL_MAXIMO for c in vaga_basica.competencias},
            projetos=0, anos_experiencia=0.0,
        )
        r = calcular_score_ponderado(perfeito, vaga_basica)
        assert r["score_base_normalizado"] == pytest.approx(1.0)
        assert r["aprovado_filtragem"] is True

    def test_candidato_sem_competencias_score_zero(self, vaga_basica):
        vazio = Candidato(id=0, nome_anonimo="CAND-000", competencias={})
        r = calcular_score_ponderado(vazio, vaga_basica)
        assert r["score_base_normalizado"] == pytest.approx(0.0)
        assert r["score_final"] == pytest.approx(0.0)

    def test_ausencia_obrigatoria_detectada(self, candidato_sem_obrigatorias, vaga_basica):
        r = calcular_score_ponderado(candidato_sem_obrigatorias, vaga_basica)
        assert r["ausencias_obrigatorias"] == 2
        assert r["aprovado_filtragem"] is False

    def test_bonus_projetos_aumenta_score(self, candidato_completo, vaga_basica):
        sem_projetos = calcular_score_ponderado(candidato_completo, vaga_basica)
        candidato_completo.projetos = 5
        com_projetos = calcular_score_ponderado(candidato_completo, vaga_basica)
        assert com_projetos["score_final"] > sem_projetos["score_final"]

    def test_experiencia_alta_reduz_score_via_recencia(self, vaga_basica):
        junior = Candidato(
            id=10, nome_anonimo="CAND-JR",
            competencias={"python": 4, "sql": 4, "docker": 3, "powerbi": 2},
            projetos=0, anos_experiencia=0.0,
        )
        senior = Candidato(
            id=11, nome_anonimo="CAND-SR",
            competencias={"python": 4, "sql": 4, "docker": 3, "powerbi": 2},
            projetos=0, anos_experiencia=20.0,
        )
        r_jr = calcular_score_ponderado(junior, vaga_basica)
        r_sr = calcular_score_ponderado(senior, vaga_basica)
        assert r_jr["score_final"] > r_sr["score_final"]

    def test_score_final_nunca_negativo(self, vaga_basica):
        candidato = Candidato(
            id=5, nome_anonimo="CAND-BAD",
            competencias={"python": 0, "sql": 0},
            projetos=0, anos_experiencia=0.0,
        )
        r = calcular_score_ponderado(candidato, vaga_basica)
        assert r["score_final"] >= 0.0

    def test_detalhes_contem_todas_competencias(self, candidato_completo, vaga_basica):
        r = calcular_score_ponderado(candidato_completo, vaga_basica)
        for comp in vaga_basica.competencias:
            assert comp in r["detalhes_por_competencia"]

    def test_calculo_manual_score_base(self, vaga_basica):
        """Verifica o cálculo aritmético diretamente."""
        c = Candidato(
            id=3, nome_anonimo="CAND-003",
            competencias={"python": 4, "sql": 5, "docker": 2, "powerbi": 0},
            projetos=0, anos_experiencia=0.0,
        )
        r = calcular_score_ponderado(c, vaga_basica)
        score_base_esperado = (5*4 + 4*5 + 3*2 + 2*0) / ((5+4+3+2)*5)
        assert r["score_base_normalizado"] == pytest.approx(score_base_esperado, rel=1e-4)


# ---------------------------------------------------------------------------
# rankear_candidatos
# ---------------------------------------------------------------------------

class TestRankearCandidatos:
    def test_ordem_decrescente_por_score(self, vaga_basica):
        candidatos = [
            Candidato(1, "CAND-001", {"python": 3, "sql": 3, "docker": 2, "powerbi": 1}, 0, 0.0),
            Candidato(2, "CAND-002", {"python": 5, "sql": 5, "docker": 5, "powerbi": 5}, 0, 0.0),
            Candidato(3, "CAND-003", {"python": 4, "sql": 4, "docker": 3, "powerbi": 2}, 0, 0.0),
        ]
        ranking = rankear_candidatos(candidatos, vaga_basica)
        scores = [r["score_final"] for r in ranking]
        assert scores == sorted(scores, reverse=True)

    def test_filtro_apenas_aprovados(self, vaga_basica):
        candidatos = [
            Candidato(1, "CAND-OK",  {"python": 4, "sql": 3, "docker": 2}, 0, 0.0),
            Candidato(2, "CAND-NOK", {"python": 1, "sql": 0, "docker": 5}, 0, 0.0),
        ]
        ranking = rankear_candidatos(candidatos, vaga_basica, apenas_aprovados=True)
        assert all(r["aprovado_filtragem"] for r in ranking)
        assert len(ranking) == 1

    def test_sem_filtro_retorna_todos(self, vaga_basica):
        candidatos = [
            Candidato(1, "CAND-OK",  {"python": 4, "sql": 3}, 0, 0.0),
            Candidato(2, "CAND-NOK", {"python": 1, "sql": 0}, 0, 0.0),
        ]
        ranking = rankear_candidatos(candidatos, vaga_basica, apenas_aprovados=False)
        assert len(ranking) == 2

    def test_lista_vazia_retorna_vazia(self, vaga_basica):
        assert rankear_candidatos([], vaga_basica) == []
