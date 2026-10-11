"""
tests/test_cosine_similarity.py
================================
Testes unitários para o módulo cosine_similarity.py
Execute com: pytest tests/ -v
"""

import math
import pytest
import numpy as np
import pandas as pd

from app.matching.cosine_similarity import (
    vetorizar_perfil,
    cosine_similarity,
    angulo_graus,
    calcular_aderencia,
    rotulo_compatibilidade,
    calcular_similaridade_candidato,
    construir_matriz_scores,
    vocabulario_da_vaga,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vocabulario():
    return ["python", "sql", "docker", "react", "git"]


@pytest.fixture
def vaga(vocabulario):
    return {
        "id": 1,
        "competencias": {"python": 1, "sql": 1, "docker": 1, "git": 1},
        "pesos": {"python": 5, "sql": 4, "docker": 3, "git": 2},
    }


@pytest.fixture
def candidatos():
    return [
        {"id": "CAND-001", "competencias": {"python": 4, "sql": 3, "docker": 2, "git": 5}},
        {"id": "CAND-002", "competencias": {"python": 5, "sql": 5, "docker": 5, "git": 5}},
        {"id": "CAND-003", "competencias": {"react": 5}},  # Sem sobreposição
        {"id": "CAND-004", "competencias": {"python": 2, "sql": 2, "git": 2}},
    ]


# ---------------------------------------------------------------------------
# vetorizar_perfil
# ---------------------------------------------------------------------------

class TestVetorizarPerfil:
    def test_tamanho_igual_ao_vocabulario(self, vocabulario):
        v = vetorizar_perfil({"python": 3}, vocabulario)
        assert len(v) == len(vocabulario)

    def test_competencia_presente_corretamente(self, vocabulario):
        v = vetorizar_perfil({"python": 4, "sql": 2}, vocabulario)
        assert v[vocabulario.index("python")] == pytest.approx(4.0)
        assert v[vocabulario.index("sql")] == pytest.approx(2.0)

    def test_competencia_ausente_vale_zero(self, vocabulario):
        v = vetorizar_perfil({"python": 3}, vocabulario)
        assert v[vocabulario.index("docker")] == pytest.approx(0.0)

    def test_perfil_vazio_retorna_zeros(self, vocabulario):
        v = vetorizar_perfil({}, vocabulario)
        assert np.all(v == 0.0)

    def test_retorna_ndarray(self, vocabulario):
        v = vetorizar_perfil({"python": 1}, vocabulario)
        assert isinstance(v, np.ndarray)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_vetores_identicos_retornam_um(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_vetores_ortogonais_retornam_zero(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_resultado_entre_zero_e_um(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = rng.uniform(0, 5, 10)
            b = rng.uniform(0, 5, 10)
            sim = cosine_similarity(a, b)
            assert 0.0 <= sim <= 1.0

    def test_vetor_zero_retorna_zero(self):
        a = np.array([1.0, 2.0])
        z = np.zeros(2)
        assert cosine_similarity(a, z) == pytest.approx(0.0)
        assert cosine_similarity(z, a) == pytest.approx(0.0)

    def test_versao_ponderada_difere_da_padrao(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([3.0, 1.0, 1.0])
        w = np.array([5.0, 1.0, 1.0])
        sim_padrao   = cosine_similarity(a, b)
        sim_ponderada = cosine_similarity(a, b, pesos=w)
        assert sim_padrao != sim_ponderada

    def test_pesos_zeros_comporta_como_sem_pesos(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        w = np.zeros(2)
        # Pesos todos zero → denominador zero → retorna 0.0 (sem crash)
        assert cosine_similarity(a, b, pesos=w) == pytest.approx(0.0)

    def test_tamanhos_diferentes_levantam_erro(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cosine_similarity(a, b)

    def test_invariante_a_escala(self):
        """Multiplicar um vetor por escalar não muda a similaridade."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # b = 2 * a
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_calculo_manual(self):
        """Verifica com cálculo à mão: (1,1,1,0) vs (1,1,0,1)."""
        a = np.array([1.0, 1.0, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.0, 1.0])
        esperado = 2.0 / (math.sqrt(3) * math.sqrt(3))
        assert cosine_similarity(a, b) == pytest.approx(esperado, rel=1e-6)


# ---------------------------------------------------------------------------
# calcular_aderencia
# ---------------------------------------------------------------------------

class TestCalcularAderencia:
    alvos = {"python": 4, "sql": 3, "docker": 3}
    pesos = {"python": 5, "sql": 4, "docker": 3}

    def test_perfil_no_alvo_retorna_um(self):
        assert calcular_aderencia(self.alvos, dict(self.alvos), self.pesos) == pytest.approx(1.0)

    def test_exceder_alvo_nao_altera(self):
        acima = {"python": 5, "sql": 5, "docker": 5}
        assert calcular_aderencia(self.alvos, acima, self.pesos) == pytest.approx(1.0)

    def test_competencias_fora_da_vaga_ignoradas(self):
        base = {"python": 4, "sql": 2, "docker": 1}
        com_extra = {**base, "java": 5, "react": 5}
        assert calcular_aderencia(self.alvos, com_extra, self.pesos) == pytest.approx(
            calcular_aderencia(self.alvos, base, self.pesos)
        )

    def test_perfil_desequilibrado_reduz(self):
        desequilibrado = {"python": 0, "sql": 3, "docker": 3}
        assert calcular_aderencia(self.alvos, desequilibrado, self.pesos) < 0.9


# ---------------------------------------------------------------------------
# angulo_graus e rotulo_compatibilidade
# ---------------------------------------------------------------------------

class TestAuxiliares:
    def test_similaridade_um_da_angulo_zero(self):
        assert angulo_graus(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_similaridade_zero_da_angulo_noventa(self):
        assert angulo_graus(0.0) == pytest.approx(90.0, abs=1e-6)

    def test_rotulos_corretos(self):
        assert rotulo_compatibilidade(1.00) == "Excelente"
        assert rotulo_compatibilidade(0.90) == "Alta"
        assert rotulo_compatibilidade(0.70) == "Moderada"
        assert rotulo_compatibilidade(0.40) == "Baixa"
        assert rotulo_compatibilidade(0.10) == "Muito baixa"


# ---------------------------------------------------------------------------
# calcular_similaridade_candidato
# ---------------------------------------------------------------------------

class TestCalcularSimilaridadeCandidato:
    def test_retorna_chaves_esperadas(self, vaga, vocabulario):
        c = {"id": "CAND-X", "competencias": {"python": 3}}
        r = calcular_similaridade_candidato(c, vaga, vocabulario)
        assert {"candidato_id", "similaridade", "angulo_graus",
                "compatibilidade", "contribuicoes"}.issubset(r.keys())

    def test_similaridade_entre_zero_e_um(self, vaga, vocabulario, candidatos):
        for c in candidatos:
            r = calcular_similaridade_candidato(c, vaga, vocabulario)
            assert 0.0 <= r["similaridade"] <= 1.0

    def test_candidato_sem_sobreposicao_retorna_zero(self, vaga, vocabulario):
        c = {"id": "CAND-Z", "competencias": {"react": 5}}
        r = calcular_similaridade_candidato(c, vaga, vocabulario)
        assert r["similaridade"] == pytest.approx(0.0)

    def test_sem_pesos_aceito(self, vocabulario):
        vaga_sem_pesos = {"id": 2, "competencias": {"python": 1, "sql": 1}}
        c = {"id": "CAND-Y", "competencias": {"python": 3, "sql": 2}}
        r = calcular_similaridade_candidato(c, vaga_sem_pesos, vocabulario, usar_pesos=False)
        assert 0.0 <= r["similaridade"] <= 1.0


# ---------------------------------------------------------------------------
# construir_matriz_scores
# ---------------------------------------------------------------------------

class TestConstruirMatrizScores:
    def test_retorna_dataframe(self, candidatos, vaga, vocabulario):
        df = construir_matriz_scores(candidatos, vaga, vocabulario)
        assert isinstance(df, pd.DataFrame)

    def test_colunas_presentes(self, candidatos, vaga, vocabulario):
        df = construir_matriz_scores(candidatos, vaga, vocabulario)
        for col in ["candidato_id", "similaridade", "angulo_graus", "compatibilidade"]:
            assert col in df.columns

    def test_ordenado_por_similaridade_decrescente(self, candidatos, vaga, vocabulario):
        df = construir_matriz_scores(candidatos, vaga, vocabulario)
        sims = df["similaridade"].tolist()
        assert sims == sorted(sims, reverse=True)

    def test_top_n_limita_resultados(self, candidatos, vaga, vocabulario):
        df = construir_matriz_scores(candidatos, vaga, vocabulario, top_n=2)
        assert len(df) == 2

    def test_lista_vazia_retorna_dataframe_vazio(self, vaga, vocabulario):
        df = construir_matriz_scores([], vaga, vocabulario)
        assert len(df) == 0
        assert "candidato_id" in df.columns

    def test_candidato_perfeito_fica_primeiro(self, vaga, vocabulario):
        candidatos = [
            {"id": "PARCIAL", "competencias": {"python": 3, "sql": 2}},
            {"id": "TOTAL",   "competencias": {c: 5 for c in vaga["competencias"]}},
        ]
        df = construir_matriz_scores(candidatos, vaga, vocabulario)
        assert df.iloc[0]["candidato_id"] == "TOTAL"
