"""
taxonomia.py
============
Normalização de nomes de competências e relações entre competências.

- SINONIMOS: grafias alternativas mapeadas para o nome canônico.
- RELACOES: crédito parcial quando o candidato possui uma competência
  transferível para a pedida (ex.: MySQL → PostgreSQL).

Mantida curada manualmente para que todo crédito seja explicável.
"""

from __future__ import annotations

from typing import Optional


SINONIMOS: dict[str, str] = {
    "postgres": "postgresql",
    "pg": "postgresql",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node.js": "nodejs",
    "react.js": "react",
    "reactjs": "react",
    "golang": "go",
    "amazon web services": "aws",
    "google cloud": "gcp",
}

RELACOES: dict[tuple[str, str], float] = {
    # (competência possuída, competência pedida): fração do nível transferida
    ("mysql", "postgresql"): 0.7,
    ("postgresql", "mysql"): 0.7,
    ("mariadb", "mysql"): 0.9,
    ("mysql", "mariadb"): 0.9,
    ("gcp", "aws"): 0.6,
    ("azure", "aws"): 0.6,
    ("aws", "gcp"): 0.6,
    ("aws", "azure"): 0.6,
    ("podman", "docker"): 0.8,
    ("typescript", "javascript"): 0.9,
    ("javascript", "typescript"): 0.6,
    ("vue", "react"): 0.5,
    ("angular", "react"): 0.5,
    ("gitlab ci", "github actions"): 0.7,
    ("github actions", "gitlab ci"): 0.7,
}


def normalizar_skill(skill: str) -> str:
    """Minúsculas, sem espaços nas pontas e com sinônimos resolvidos."""
    nome = skill.strip().lower()
    return SINONIMOS.get(nome, nome)


def normalizar_competencias(competencias: dict[str, float]) -> dict[str, float]:
    """Normaliza as chaves; sinônimos que colidem mantêm o maior nível."""
    resultado: dict[str, float] = {}
    for skill, nivel in competencias.items():
        nome = normalizar_skill(skill)
        resultado[nome] = max(resultado.get(nome, 0.0), float(nivel))
    return resultado


def nivel_efetivo(
    competencias: dict[str, float],
    pedida: str,
) -> tuple[float, Optional[str]]:
    """
    Nível do candidato na competência pedida, considerando relações.

    Retorna (nível, origem): `origem` é a competência relacionada que gerou
    o crédito parcial, ou None quando o nível vem da própria competência.
    """
    melhor, origem = float(competencias.get(pedida, 0.0)), None
    for possuida, nivel in competencias.items():
        fracao = RELACOES.get((possuida, pedida))
        if fracao and fracao * nivel > melhor:
            melhor, origem = fracao * nivel, possuida
    return melhor, origem
