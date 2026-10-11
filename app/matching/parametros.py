"""
parametros.py
=============
Parâmetros ajustáveis do pipeline de matching.

Os valores padrão preservam o comportamento conservador (sem tolerância em
competências obrigatórias). O cliente pode sobrescrevê-los por requisição,
dentro dos limites validados em `app.schemas.ParametrosSchema`.
"""

from __future__ import annotations

from dataclasses import dataclass


NIVEL_MAXIMO: int = 5
"""Valor máximo de nível em uma competência (escala 0–5)."""


@dataclass(frozen=True)
class ParametrosMatching:
    tolerancia_obrigatoria: int = 0
    """Níveis abaixo de `nivel_min` aceitos com penalidade (0 = filtro estrito)."""

    penalidade_por_nivel_faltante: float = 0.20
    """Redução multiplicativa por nível faltante dentro da tolerância."""

    nivel_alvo_padrao: int = 3
    """Alvo de competências sem `nivel_alvo` explícito (nunca abaixo de `nivel_min`)."""

    tau_experiencia: float = 3.0
    """Anos para o fator de experiência atingir ~63%: 1 − exp(−anos/τ)."""

    projetos_saturacao: int = 3
    """Número de projetos a partir do qual o componente de projetos satura."""

    peso_cobertura: float = 0.55
    peso_aderencia: float = 0.20
    peso_experiencia: float = 0.15
    peso_projetos: float = 0.10
    """Pesos da combinação final (normalizados para somar 1)."""

    penalidade_senioridade_abaixo: float = 0.15
    """Redução por degrau quando o candidato está abaixo da senioridade da vaga."""

    penalidade_senioridade_acima: float = 0.05
    """Redução por degrau quando o candidato está acima. Mantida baixa: é proxy de idade."""

    fator_senioridade_minimo: float = 0.40
    """Piso do fator de senioridade."""


PARAMETROS_PADRAO = ParametrosMatching()
