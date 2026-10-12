"""
tests/test_ranking.py
=====================
Testes do pipeline combinado (ranking.py), incluindo propriedades de
monotonicidade que um ranking justo e explicável deve respeitar.
Execute com: pytest tests/ -v
"""

from dataclasses import replace

import pytest

from app.matching.parametros import ParametrosMatching
from app.matching.ranking import avaliar_candidato, rankear
from app.matching.weighted_scoring import Candidato, Vaga


@pytest.fixture
def vaga():
    return Vaga(
        id=1,
        titulo="Backend Python Pleno",
        competencias={
            "python":     {"peso": 5, "obrigatoria": True, "nivel_min": 3, "nivel_alvo": 4},
            "postgresql": {"peso": 4, "obrigatoria": True, "nivel_min": 2},
            "docker":     {"peso": 3},
            "aws":        {"peso": 2},
            "git":        {"peso": 2},
        },
        senioridade=2,
    )


@pytest.fixture
def candidato():
    return Candidato(
        id=1, nome_anonimo="CAND-001",
        competencias={"python": 3, "postgresql": 2, "docker": 2, "aws": 1, "git": 1},
        projetos=1, anos_experiencia=2.0, senioridade=2,
    )


class TestPropriedades:
    def test_mais_nivel_nunca_reduz_score(self, candidato, vaga):
        base = avaliar_candidato(candidato, vaga).score
        for skill in vaga.competencias:
            melhorado = replace(
                candidato,
                competencias={**candidato.competencias, skill: candidato.competencias.get(skill, 0) + 1},
            )
            assert avaliar_candidato(melhorado, vaga).score >= base

    def test_mais_experiencia_nunca_reduz_score(self, candidato, vaga):
        scores = [
            avaliar_candidato(replace(candidato, anos_experiencia=a), vaga).score
            for a in [0, 1, 3, 6, 12, 25]
        ]
        assert scores == sorted(scores)

    def test_mais_projetos_nunca_reduz_score(self, candidato, vaga):
        scores = [avaliar_candidato(replace(candidato, projetos=p), vaga).score for p in range(8)]
        assert scores == sorted(scores)

    def test_competencia_extra_nao_penaliza(self, candidato, vaga):
        com_extra = replace(candidato, competencias={**candidato.competencias, "java": 5, "react": 4})
        assert avaliar_candidato(com_extra, vaga).score == pytest.approx(avaliar_candidato(candidato, vaga).score)

    def test_score_no_intervalo_unitario(self, vaga):
        maximo = Candidato(9, "CAND-MAX", {s: 5 for s in vaga.competencias}, 50, 40.0, 2)
        r = avaliar_candidato(maximo, vaga)
        assert 0.0 <= r.score <= 1.0
        assert r.score == pytest.approx(1.0, abs=1e-3)


class TestCombinacao:
    def test_nivel_importa_e_nao_so_direcao(self, vaga):
        """Regressão: com vetor binário, nível 1 em tudo empatava com nível 5 em tudo."""
        fraco = Candidato(1, "FRACO", {s: 3 if s == "python" else 2 for s in vaga.competencias}, senioridade=2)
        forte = Candidato(2, "FORTE", {s: 5 for s in vaga.competencias}, senioridade=2)
        assert avaliar_candidato(forte, vaga).score > avaliar_candidato(fraco, vaga).score

    def test_eliminado_tem_score_zero_e_explicacao(self, vaga):
        c = Candidato(1, "CAND-ELIM", {"python": 1, "postgresql": 4})
        r = avaliar_candidato(c, vaga)
        assert r.aprovado is False
        assert r.score == 0.0
        assert any("eliminatório" in motivo for motivo in r.explicacao)

    def test_explicacao_cita_credito_parcial(self, vaga):
        c = Candidato(1, "CAND-MYSQL", {"python": 4, "mysql": 4}, senioridade=2)
        r = avaliar_candidato(c, vaga)
        assert r.aprovado is True
        assert any("via mysql" in motivo for motivo in r.explicacao)
        assert any("competências obrigatórias" in motivo for motivo in r.explicacao)

    def test_comportamentais_coincidentes_sobem_o_score_sem_eliminar(self, candidato, vaga):
        base = avaliar_candidato(candidato, vaga).score
        vaga_com = replace(
            vaga,
            competencias_comportamentais=["Comunicação", "Empatia", "Autonomia"],
        )
        parcial = replace(candidato, competencias_comportamentais=["Comunicação"])
        total = replace(
            candidato,
            competencias_comportamentais=["Comunicação", "Empatia", "Autonomia"],
        )
        eliminado = Candidato(
            1,
            "CAND-ELIM",
            {"python": 1},
            competencias_comportamentais=["Comunicação", "Empatia", "Autonomia"],
        )

        score_parcial = avaliar_candidato(parcial, vaga_com).score
        score_total = avaliar_candidato(total, vaga_com).score
        assert score_parcial > base
        assert score_total > score_parcial
        assert avaliar_candidato(eliminado, vaga_com).aprovado is False
        assert avaliar_candidato(eliminado, vaga_com).score == 0.0
        assert any(
            "3 de 3" in motivo
            for motivo in avaliar_candidato(total, vaga_com).explicacao
        )

    def test_componentes_expostos(self, candidato, vaga):
        r = avaliar_candidato(candidato, vaga)
        assert set(r.componentes) == {"cobertura", "aderencia", "experiencia", "projetos", "gate", "senioridade"}

    def test_pesos_sao_normalizados(self, candidato, vaga):
        padrao = avaliar_candidato(candidato, vaga).score
        dobrado = ParametrosMatching(
            peso_cobertura=1.1, peso_aderencia=0.4, peso_experiencia=0.3, peso_projetos=0.2
        )
        assert avaliar_candidato(candidato, vaga, dobrado).score == pytest.approx(padrao, abs=1e-4)


class TestRankear:
    def test_ordem_e_desempate_deterministico(self, vaga):
        perfil = {"python": 4, "postgresql": 3, "docker": 3, "aws": 3, "git": 3}
        candidatos = [
            Candidato(1, "CAND-B", perfil, senioridade=2),
            Candidato(2, "CAND-A", perfil, senioridade=2),
            Candidato(3, "CAND-ELIM", {"python": 1}),
        ]
        ranking = rankear(candidatos, vaga)
        assert [r.nome_anonimo for r in ranking] == ["CAND-A", "CAND-B", "CAND-ELIM"]

    def test_ids_duplicados_levantam_erro(self, candidato, vaga):
        with pytest.raises(ValueError):
            rankear([candidato, replace(candidato, id=2)], vaga)
