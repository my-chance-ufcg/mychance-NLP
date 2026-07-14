"""
weighted_scoring.py
===================
Algoritmo de Pontuação Ponderada por Competências (Weighted Skill Scoring).

Primeira etapa do pipeline de matching do My Chance:
filtragem rápida e eliminação de candidatos com lacunas em competências
obrigatórias antes do ranqueamento fino pela Cosine Similarity.

Uso básico
----------
>>> from weighted_scoring import Vaga, Candidato, rankear_candidatos
>>> vaga = Vaga(id=1, titulo="Dev Backend", competencias={
...     "python": {"peso": 5, "obrigatoria": True,  "nivel_min": 3},
...     "sql":    {"peso": 3, "obrigatoria": False, "nivel_min": 1},
... })
>>> candidato = Candidato(id=1, nome_anonimo="CAND-001",
...     competencias={"python": 4, "sql": 3}, projetos=2,
...     anos_experiencia=1.5)
>>> resultado = calcular_score_ponderado(candidato, vaga)
>>> print(resultado["score_final"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# -----------------
# Constantes padrão
# Quero que o cliente possa mudar esses parâmetros. Falar com os meninos!!!!!!!
# -----------------

NIVEL_MAXIMO: int = 5
"""Valor máximo de nível em uma competência (escala 0–5)."""

BONUS_POR_PROJETO: float = 2.0
"""Pontos brutos adicionados por projeto real no portfólio."""

PENALIZACAO_OBRIGATORIA: float = 15.0
"""Pontos brutos descontados por cada competência obrigatória ausente."""

DECAIMENTO_RECENCIA: float = 0.1
"""Taxa de decaimento exponencial para o fator de recência de experiência.
   Fórmula: exp(-DECAIMENTO_RECENCIA * anos_experiencia)
   - 0 anos  → fator ≈ 1.00
   - 5 anos  → fator ≈ 0.61
   - 10 anos → fator ≈ 0.37
   Obs.: anos_experiencia já deve chegar ponderado por recência no payload.
"""

PENALIDADE_POR_DEGRAU_SENIORIDADE: float = 0.15
"""Redução no fator de senioridade por cada degrau de distância (vaga ↔ candidato)."""

FATOR_SENIORIDADE_MINIMO: float = 0.40
"""Piso do fator de distância de senioridade."""


# ----------------
# Modelos de dados
# ----------------

@dataclass
class ConfigCompetencia:
    """Configuração de uma competência dentro de uma vaga."""
    peso: float
    """Importância relativa da competência (quanto maior, mais relevante)."""
    obrigatoria: bool = False
    """Se True, ausência elimina o candidato da filtragem."""
    nivel_min: int = 1
    """Nível mínimo aceitável quando a competência é obrigatória."""


@dataclass
class Vaga:
    """Representa uma vaga de emprego com seus requisitos técnicos."""
    id: int
    titulo: str
    competencias: dict[str, dict]
    """
    Dicionário no formato:
        {
            "python": {"peso": 5, "obrigatoria": True, "nivel_min": 3},
            "docker": {"peso": 3, "obrigatoria": False},
        }
    """
    senioridade: int = 0
    """Nível alvo da vaga (0=bolsa/iniciação … 4=especialista)."""
    idiomas: dict[str, int] = field(default_factory=dict)
    """Mapa idioma -> nível mínimo (rank). Filtragem principal ocorre no backend."""

    def get_config(self, competencia: str) -> ConfigCompetencia:
        """Retorna ConfigCompetencia para uma competência da vaga."""
        dados = self.competencias.get(competencia, {})
        return ConfigCompetencia(
            peso=float(dados.get("peso", 1)),
            obrigatoria=bool(dados.get("obrigatoria", False)),
            nivel_min=int(dados.get("nivel_min", 1)),
        )


@dataclass
class Candidato:
    """Representa o perfil anonimizado de um candidato."""
    id: int
    nome_anonimo: str
    """Identificador público do candidato no My Chance (ex: 'CAND-042')."""
    competencias: dict[str, float]
    """
    Dicionário de competências com nível declarado:
        {"python": 4, "sql": 5, "docker": 2}
    Competências ausentes são tratadas como nível 0.
    """
    projetos: int = 0
    """Número de projetos reais vinculados ao perfil."""
    anos_experiencia: float = 0.0
    """Anos de experiência ponderados por recência."""
    senioridade: int = 0
    """Nível atual do candidato (0–4)."""
    idiomas: dict[str, int] = field(default_factory=dict)
    """Mapa idioma -> nível (rank)."""


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def calcular_fator_recencia(anos: float) -> float:
    """
    Calcula o fator de recência via decaimento exponencial.

    Experiências muito antigas recebem peso menor, incentivando candidatos
    com histórico recente e atualizado.

    Parâmetros
    ----------
    anos : float
        Anos de experiência total do candidato.

    Retorna
    -------
    float
        Fator multiplicativo em (0, 1].
    """
    return math.exp(-DECAIMENTO_RECENCIA * max(0.0, anos))


def calcular_fator_senioridade(
    senioridade_candidato: int,
    senioridade_vaga: int,
    penalidade_por_degrau: float = PENALIDADE_POR_DEGRAU_SENIORIDADE,
    fator_minimo: float = FATOR_SENIORIDADE_MINIMO,
) -> float:
    """
    Distância de senioridade: mesmo nível = 1.0; cada degrau reduz o fator.
    Overqualified e underqualified são tratados simetricamente.
    """
    distancia = abs(int(senioridade_candidato) - int(senioridade_vaga))
    return max(fator_minimo, 1.0 - penalidade_por_degrau * distancia)


def _score_maximo_teorico(vaga: Vaga, nivel_maximo: int = NIVEL_MAXIMO) -> float:
    """Soma ponderada máxima possível para uma vaga."""
    return sum(
        float(cfg.get("peso", 1)) * nivel_maximo
        for cfg in vaga.competencias.values()
    )


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------

def calcular_score_ponderado(
    candidato: Candidato,
    vaga: Vaga,
    nivel_maximo: int = NIVEL_MAXIMO,
    bonus_por_projeto: float = BONUS_POR_PROJETO,
    penalizacao_obrigatoria: float = PENALIZACAO_OBRIGATORIA,
) -> dict:
    """
    Calcula o score ponderado de um candidato para uma vaga.

    Fórmula
    -------
    Score_base  = (somatório)Σ (peso_i × nivel_i)
    Score_norm  = Score_base / Score_máximo           ∈ [0, 1]
    Score_final = (Score_norm + bônus_projetos) × fator_recência
                  − penalização_obrigatórias

    Parâmetros
    ----------
    candidato : Candidato
    vaga : Vaga
    nivel_maximo : int
        Valor máximo de nível possível (padrão: 5).
    bonus_por_projeto : float
        Pontos brutos por projeto real declarado.
    penalizacao_obrigatoria : float
        Pontos brutos descontados por competência obrigatória ausente.

    Retorna
    -------
    dict com as chaves:
        - candidato_id          : int
        - score_base_normalizado: float  ∈ [0, 1] (para melhor leitura e comparação)
        - score_final           : float  ∈ [0, 1] (após bônus/penalizações)
        - ausencias_obrigatorias: int
        - aprovado_filtragem    : bool   (False se há ausências obrigatórias)
        - detalhes_por_competencia: dict
    """
    score_base = 0.0
    score_maximo = _score_maximo_teorico(vaga, nivel_maximo)
    ausencias_obrigatorias = 0
    detalhes: dict[str, dict] = {}

    for competencia, cfg_raw in vaga.competencias.items():
        cfg = ConfigCompetencia(
            peso=float(cfg_raw.get("peso", 1)),
            obrigatoria=bool(cfg_raw.get("obrigatoria", False)),
            nivel_min=int(cfg_raw.get("nivel_min", 1)),
        )
        nivel_candidato = float(
            candidato.competencias.get(competencia, 0)
        )
        contribuicao = cfg.peso * nivel_candidato
        score_base += contribuicao

        ausente = cfg.obrigatoria and nivel_candidato < cfg.nivel_min
        if ausente:
            ausencias_obrigatorias += 1

        detalhes[competencia] = {
            "peso": cfg.peso,
            "obrigatoria": cfg.obrigatoria,
            "nivel_min": cfg.nivel_min,
            "nivel_candidato": nivel_candidato,
            "contribuicao": round(contribuicao, 4),
            "lacuna_critica": ausente,
        }

    # Normalização
    score_normalizado = score_base / score_maximo if score_maximo > 0 else 0.0

    # Bônus por projetos (normalizado pelo score máximo)
    bonus_projetos = (candidato.projetos * bonus_por_projeto) / score_maximo if score_maximo > 0 else 0.0

    # Fator de recência
    fator_recencia = calcular_fator_recencia(candidato.anos_experiencia)

    # Distância de senioridade (novo parâmetro de score)
    fator_senioridade = calcular_fator_senioridade(
        candidato.senioridade,
        vaga.senioridade,
    )

    # Penalização por ausências obrigatórias
    penalizacao = (
        ausencias_obrigatorias * penalizacao_obrigatoria / score_maximo
        if score_maximo > 0
        else 0.0
    )

    score_final = max(
        0.0,
        (score_normalizado + bonus_projetos) * fator_recencia * fator_senioridade - penalizacao,
    )

    return {
        "candidato_id": candidato.id,
        "nome_anonimo": candidato.nome_anonimo,
        "score_base_normalizado": round(score_normalizado, 4),
        "score_final": round(score_final, 4),
        "fator_senioridade": round(fator_senioridade, 4),
        "ausencias_obrigatorias": ausencias_obrigatorias,
        "aprovado_filtragem": ausencias_obrigatorias == 0,
        "detalhes_por_competencia": detalhes,
    }


def rankear_candidatos(
    candidatos: list[Candidato],
    vaga: Vaga,
    apenas_aprovados: bool = True,
    **kwargs,
) -> list[dict]:
    """
    Gera o ranking ponderado de candidatos para uma vaga.

    Parâmetros
    ----------
    candidatos : list[Candidato]
    vaga : Vaga
    apenas_aprovados : bool
        Se True (padrão), filtra candidatos com ausências obrigatórias.
    **kwargs
        Repassados para `calcular_score_ponderado`.

    Retorna
    -------
    list[dict]
        Lista ordenada por score_final decrescente.
    """
    resultados = [
        calcular_score_ponderado(c, vaga, **kwargs) for c in candidatos
    ]
    if apenas_aprovados:
        resultados = [r for r in resultados if r["aprovado_filtragem"]]
    return sorted(resultados, key=lambda x: x["score_final"], reverse=True)
