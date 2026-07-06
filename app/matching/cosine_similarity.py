"""
cosine_similarity.py
====================
Algoritmo de Matching por Similaridade Vetorial (Cosine Similarity).

Segunda etapa do pipeline de matching do My Chance:
ranqueamento fino dos candidatos pré-filtrados pela Pontuação Ponderada,
medindo a similaridade angular entre vetores de competências.

Uso básico
----------
>>> from cosine_similarity import construir_matriz_scores
>>> vocabulario = ["python", "sql", "docker", "git"]
>>> vaga = {
...     "id": 1,
...     "competencias": {"python": 1, "sql": 1, "docker": 1},
...     "pesos": {"python": 5, "sql": 4, "docker": 3},
... }
>>> candidatos = [
...     {"id": "CAND-001", "competencias": {"python": 4, "sql": 3, "docker": 2}},
...     {"id": "CAND-002", "competencias": {"python": 5, "sql": 5, "docker": 5}},
... ]
>>> ranking = construir_matriz_scores(candidatos, vaga, vocabulario)
>>> print(ranking)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Limiares de interpretação
# ---------------------------------------------------------------------------

LIMIARES_COMPATIBILIDADE: list[tuple[float, str]] = [
    (0.95, "Excelente"),
    (0.85, "Alta"),
    (0.60, "Moderada"),
    (0.30, "Baixa"),
    (0.00, "Muito baixa"),
]
"""
Intervalos de compatibilidade para rótulo qualitativo do score.
Cada entrada é (limiar_inferior, rótulo).
"""


# ---------------------------------------------------------------------------
# Vetorização
# ---------------------------------------------------------------------------

def vetorizar_perfil(
    competencias_perfil: dict[str, float],
    vocabulario: list[str],
) -> np.ndarray:
    """
    Converte um dicionário de competências em vetor numpy alinhado ao vocabulário.

    Competências não presentes no perfil recebem valor 0.

    Parâmetros
    ----------
    competencias_perfil : dict[str, float]
        Ex: {"python": 4, "sql": 3, "docker": 2}
    vocabulario : list[str]
        Lista ordenada de todas as competências reconhecidas pelo sistema.

    Retorna
    -------
    np.ndarray
        Vetor de floats com tamanho len(vocabulario).

    Exemplo
    -------
    >>> vetorizar_perfil({"python": 4, "sql": 0}, ["python", "sql", "docker"])
    array([4., 0., 0.])
    """
    return np.array(
        [float(competencias_perfil.get(c, 0.0)) for c in vocabulario],
        dtype=np.float64,
    )


def vocabulario_da_vaga(vaga: dict) -> list[str]:
    """
    Retorna a lista de competências definidas na vaga como vocabulário mínimo.

    Útil quando não há vocabulário global disponível.
    """
    return list(vaga.get("competencias", {}).keys())


# ---------------------------------------------------------------------------
# Cálculo de similaridade
# ---------------------------------------------------------------------------

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    pesos: Optional[np.ndarray] = None,
) -> float:
    """
    Calcula a similaridade por cosseno entre dois vetores.

    Versão padrão
    -------------
        cos(θ) = (A · B) / (||A|| × ||B||)

    Versão ponderada (quando `pesos` é fornecido)
    ----------------------------------------------
        cos_w(θ) = Σ(w_i × A_i × B_i)
                   ─────────────────────────────────────
                   √(Σ w_i A_i²) × √(Σ w_i B_i²)

    Parâmetros
    ----------
    a : np.ndarray
        Vetor do perfil A (tipicamente a vaga).
    b : np.ndarray
        Vetor do perfil B (tipicamente o candidato).
    pesos : np.ndarray, opcional
        Vetor de pesos por dimensão. Ativa a versão ponderada.

    Retorna
    -------
    float
        Similaridade em [0, 1] (vetores não negativos).

    Raises
    ------
    ValueError
        Se os vetores tiverem tamanhos diferentes.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Vetores com tamanhos incompatíveis: {a.shape} vs {b.shape}"
        )

    if pesos is not None:
        num = float(np.sum(pesos * a * b))
        den = float(
            math.sqrt(np.sum(pesos * a ** 2)) *
            math.sqrt(np.sum(pesos * b ** 2))
        )
    else:
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b))

    if den == 0.0:
        return 0.0

    # Clamp para evitar erros de arredondamento no arccos
    return float(np.clip(num / den, 0.0, 1.0))


def angulo_graus(similaridade: float) -> float:
    """Converte similaridade cossenoidal para ângulo em graus."""
    return math.degrees(math.acos(np.clip(similaridade, 0.0, 1.0)))


def rotulo_compatibilidade(similaridade: float) -> str:
    """Retorna rótulo qualitativo para um valor de similaridade."""
    for limiar, rotulo in LIMIARES_COMPATIBILIDADE:
        if similaridade >= limiar:
            return rotulo
    return "Muito baixa"


# ---------------------------------------------------------------------------
# Ranqueamento
# ---------------------------------------------------------------------------

def calcular_similaridade_candidato(
    candidato: dict,
    vaga: dict,
    vocabulario: list[str],
    usar_pesos: bool = True,
) -> dict:
    """
    Calcula a similaridade de um único candidato com a vaga.

    Parâmetros
    ----------
    candidato : dict
        Deve conter 'id' e 'competencias' (dict[str, float]).
    vaga : dict
        Deve conter 'competencias' e opcionalmente 'pesos'.
    vocabulario : list[str]
        Lista global de competências reconhecidas.
    usar_pesos : bool
        Se True, usa a versão ponderada da similaridade.

    Retorna
    -------
    dict com as chaves:
        - candidato_id      : str | int
        - similaridade      : float  ∈ [0, 1]
        - angulo_graus      : float  (0° = idênticos, 90° = sem sobreposição)
        - compatibilidade   : str    (rótulo qualitativo)
        - contribuicoes     : dict   (por competência)
    """
    vetor_vaga = vetorizar_perfil(vaga.get("competencias", {}), vocabulario)
    vetor_candidato = vetorizar_perfil(candidato.get("competencias", {}), vocabulario)

    pesos: Optional[np.ndarray] = None
    if usar_pesos and vaga.get("pesos"):
        pesos = vetorizar_perfil(vaga["pesos"], vocabulario)
        if pesos.sum() == 0:
            pesos = None

    sim = cosine_similarity(vetor_vaga, vetor_candidato, pesos)

    # Contribuição por competência (produto elemento a elemento)
    contribuicoes = {
        comp: round(float(vetor_vaga[i] * vetor_candidato[i]), 4)
        for i, comp in enumerate(vocabulario)
        if vetor_vaga[i] > 0 or vetor_candidato[i] > 0
    }

    return {
        "candidato_id": candidato["id"],
        "similaridade": round(sim, 4),
        "angulo_graus": round(angulo_graus(sim), 2),
        "compatibilidade": rotulo_compatibilidade(sim),
        "contribuicoes": contribuicoes,
    }


def construir_matriz_scores(
    candidatos: list[dict],
    vaga: dict,
    vocabulario: list[str],
    usar_pesos: bool = True,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Gera o DataFrame de ranking de candidatos para uma vaga.

    Parâmetros
    ----------
    candidatos : list[dict]
        Lista de candidatos (cada um com 'id' e 'competencias').
    vaga : dict
        Dicionário da vaga com 'competencias' e opcionalmente 'pesos'.
    vocabulario : list[str]
        Lista global de competências.
    usar_pesos : bool
        Ativa a versão ponderada da Cosine Similarity.
    top_n : int, opcional
        Retorna apenas os top-N candidatos.

    Retorna
    -------
    pd.DataFrame
        Colunas: candidato_id, similaridade, angulo_graus, compatibilidade.
        Ordenado por similaridade decrescente.
    """
    if not candidatos:
        return pd.DataFrame(
            columns=["candidato_id", "similaridade", "angulo_graus", "compatibilidade"]
        )

    resultados = [
        calcular_similaridade_candidato(c, vaga, vocabulario, usar_pesos)
        for c in candidatos
    ]

    df = pd.DataFrame([
        {
            "candidato_id": r["candidato_id"],
            "similaridade": r["similaridade"],
            "angulo_graus": r["angulo_graus"],
            "compatibilidade": r["compatibilidade"],
        }
        for r in resultados
    ])

    df = df.sort_values("similaridade", ascending=False).reset_index(drop=True)
    df.index += 1  # Ranking começa em 1
    df.index.name = "posicao"

    if top_n is not None:
        df = df.head(top_n)

    return df
