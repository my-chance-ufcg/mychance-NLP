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
    calcular_fator_experiencia,
    calcular_fator_projetos,
    calcular_fator_senioridade,
    calcular_score_ponderado,
    rankear_candidatos,
)
from app.matching.parametros import NIVEL_MAXIMO, ParametrosMatching


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
# Fatores auxiliares
# ---------------------------------------------------------------------------

class TestFatorExperiencia:
    def test_zero_anos_retorna_zero(self):
        assert calcular_fator_experiencia(0.0) == pytest.approx(0.0)

    def test_negativo_tratado_como_zero(self):
        assert calcular_fator_experiencia(-5.0) == pytest.approx(0.0)

    def test_curva_saturante(self):
        assert calcular_fator_experiencia(3.0, tau=3.0) == pytest.approx(1 - math.exp(-1), rel=1e-6)

    def test_crescente_e_limitado(self):
        valores = [calcular_fator_experiencia(float(a)) for a in [0, 1, 5, 10, 50]]
        assert valores == sorted(valores)
        assert all(0.0 <= v < 1.0 for v in valores)


class TestFatorProjetos:
    def test_satura_no_teto(self):
        assert calcular_fator_projetos(10, saturacao=3) == pytest.approx(1.0)

    def test_proporcional_abaixo_do_teto(self):
        assert calcular_fator_projetos(1, saturacao=4) == pytest.approx(0.25)


class TestFatorSenioridade:
    def test_mesmo_nivel_retorna_um(self):
        assert calcular_fator_senioridade(2, 2) == pytest.approx(1.0)

    def test_abaixo_penaliza_mais_que_acima(self):
        assert calcular_fator_senioridade(1, 2) < calcular_fator_senioridade(3, 2)

    def test_respeita_piso(self):
        p = ParametrosMatching(fator_senioridade_minimo=0.5)
        assert calcular_fator_senioridade(0, 5, p) == pytest.approx(0.5)


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

    def test_cobertura_satura_no_alvo(self, vaga_basica):
        """Exceder o alvo não aumenta a cobertura."""
        no_alvo = Candidato(1, "CAND-A", {"python": 3, "sql": 3, "docker": 3, "powerbi": 3})
        acima = Candidato(2, "CAND-B", {"python": 5, "sql": 5, "docker": 5, "powerbi": 5})
        assert calcular_score_ponderado(no_alvo, vaga_basica)["score_base_normalizado"] == pytest.approx(1.0)
        assert calcular_score_ponderado(acima, vaga_basica)["score_base_normalizado"] == pytest.approx(1.0)

    def test_score_final_no_intervalo_unitario(self, candidato_completo, vaga_basica):
        r = calcular_score_ponderado(candidato_completo, vaga_basica)
        assert 0.0 <= r["score_final"] <= 1.0

    def test_tolerancia_aprova_com_penalidade(self, vaga_basica):
        c = Candidato(4, "CAND-TOL", {"python": 2, "sql": 3, "docker": 3, "powerbi": 3})
        estrito = calcular_score_ponderado(c, vaga_basica)
        tolerante = calcular_score_ponderado(
            c, vaga_basica, ParametrosMatching(tolerancia_obrigatoria=1, penalidade_por_nivel_faltante=0.2)
        )
        assert estrito["aprovado_filtragem"] is False
        assert tolerante["aprovado_filtragem"] is True
        assert tolerante["obrigatorias_toleradas"] == 1
        assert tolerante["fator_gate"] == pytest.approx(0.8)

    def test_deficit_acima_da_tolerancia_elimina(self, vaga_basica):
        c = Candidato(5, "CAND-ELIM", {"python": 1, "sql": 3})
        r = calcular_score_ponderado(c, vaga_basica, ParametrosMatching(tolerancia_obrigatoria=1))
        assert r["aprovado_filtragem"] is False
        assert r["score_final"] == pytest.approx(0.0)

    def test_credito_parcial_por_competencia_relacionada(self):
        vaga = Vaga(1, "Backend", {"postgresql": {"peso": 4, "obrigatoria": True, "nivel_min": 2}})
        c = Candidato(6, "CAND-MYSQL", {"MySQL": 4})
        r = calcular_score_ponderado(c, vaga)
        detalhe = r["detalhes_por_competencia"]["postgresql"]
        assert r["aprovado_filtragem"] is True
        assert detalhe["credito_via"] == "mysql"
        assert detalhe["nivel_candidato"] == pytest.approx(2.8)

    def test_sinonimos_colidindo_na_vaga_mantem_requisito_mais_exigente(self):
        vaga = Vaga(1, "Backend", {
            "postgres":   {"peso": 5, "obrigatoria": True, "nivel_min": 4},
            "postgresql": {"peso": 1},
        })
        r = calcular_score_ponderado(Candidato(8, "CAND-PG", {"postgresql": 1}), vaga)
        assert r["aprovado_filtragem"] is False
        assert r["detalhes_por_competencia"]["postgresql"]["peso"] == 5

    def test_nivel_alvo_nunca_abaixo_do_minimo(self):
        vaga = Vaga(1, "Backend", {"python": {"peso": 5, "obrigatoria": True, "nivel_min": 4, "nivel_alvo": 2}})
        r = calcular_score_ponderado(Candidato(9, "CAND-PY", {"python": 2}), vaga)
        assert r["detalhes_por_competencia"]["python"]["nivel_alvo"] == 4
        assert r["score_base_normalizado"] == pytest.approx(0.5)

    def test_sinonimos_na_vaga_e_no_candidato(self):
        vaga = Vaga(1, "Backend", {"Postgres": {"peso": 4, "obrigatoria": True, "nivel_min": 3}})
        c = Candidato(7, "CAND-PG", {"postgresql": 4})
        r = calcular_score_ponderado(c, vaga)
        assert r["aprovado_filtragem"] is True
        assert "postgresql" in r["detalhes_por_competencia"]

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
        # alvo = max(nivel_min, 3): python 3, sql 3, docker 3, powerbi 3
        score_base_esperado = (5*1 + 4*1 + 3*(2/3) + 2*0) / (5+4+3+2)
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
