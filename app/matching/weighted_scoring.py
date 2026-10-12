"""
weighted_scoring.py
===================
Algoritmo de Pontuação Ponderada por Competências (Weighted Skill Scoring).

Primeira etapa do pipeline de matching do My Chance:
filtragem de candidatos com lacunas em competências obrigatórias e cálculo
da cobertura ponderada dos requisitos da vaga.

Uso básico
----------
>>> from weighted_scoring import Vaga, Candidato, calcular_score_ponderado
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

from app.matching.parametros import NIVEL_MAXIMO, PARAMETROS_PADRAO, ParametrosMatching
from app.matching.taxonomia import nivel_efetivo, normalizar_competencias, normalizar_skill


# ----------------
# Modelos de dados
# ----------------

@dataclass
class ConfigCompetencia:
    """Configuração de uma competência dentro de uma vaga."""
    peso: float
    """Importância relativa da competência (quanto maior, mais relevante)."""
    obrigatoria: bool = False
    """Se True, nível abaixo de `nivel_min` (além da tolerância) elimina o candidato."""
    nivel_min: int = 1
    """Nível mínimo aceitável quando a competência é obrigatória."""
    nivel_alvo: Optional[int] = None
    """Nível a partir do qual a competência está plenamente atendida."""

    def alvo(self, nivel_alvo_padrao: int) -> int:
        """Nível alvo efetivo (explícito ou padrão), nunca abaixo de nivel_min, limitado ao máximo."""
        alvo = max(self.nivel_alvo or nivel_alvo_padrao, self.nivel_min)
        return min(max(alvo, 1), NIVEL_MAXIMO)


@dataclass
class Vaga:
    """Representa uma vaga de emprego com seus requisitos técnicos."""
    id: int
    titulo: str
    competencias: dict[str, dict]
    """
    Dicionário no formato:
        {
            "python": {"peso": 5, "obrigatoria": True, "nivel_min": 3, "nivel_alvo": 4},
            "docker": {"peso": 3, "obrigatoria": False},
        }
    """
    senioridade: int = 0
    """Nível alvo da vaga (0=bolsa/iniciação … 4=especialista)."""
    idiomas: dict[str, int] = field(default_factory=dict)
    """Mapa idioma -> nível mínimo (rank). Filtragem principal ocorre no backend."""
    competencias_comportamentais: list[str] = field(default_factory=list)
    """Até 3 habilidades pedidas pela vaga. Coincidência dá bônus e nunca elimina."""

    def get_config(self, competencia: str) -> ConfigCompetencia:
        """Retorna ConfigCompetencia para uma competência da vaga."""
        dados = self.competencias.get(competencia, {})
        nivel_alvo = dados.get("nivel_alvo")
        return ConfigCompetencia(
            peso=float(dados.get("peso", 1)),
            obrigatoria=bool(dados.get("obrigatoria", False)),
            nivel_min=int(dados.get("nivel_min", 1)),
            nivel_alvo=int(nivel_alvo) if nivel_alvo is not None else None,
        )

    def configs(self) -> dict[str, ConfigCompetencia]:
        """
        Configurações por competência, com nomes normalizados.

        Sinônimos que colidem (ex.: "postgres" e "postgresql") são fundidos
        mantendo o requisito mais exigente, para nenhuma obrigatória se perder.
        """
        configs: dict[str, ConfigCompetencia] = {}
        for competencia in self.competencias:
            nome = normalizar_skill(competencia)
            cfg = self.get_config(competencia)
            existente = configs.get(nome)
            if existente is not None:
                alvos = [a for a in (existente.nivel_alvo, cfg.nivel_alvo) if a is not None]
                cfg = ConfigCompetencia(
                    peso=max(existente.peso, cfg.peso),
                    obrigatoria=existente.obrigatoria or cfg.obrigatoria,
                    nivel_min=max(existente.nivel_min, cfg.nivel_min),
                    nivel_alvo=max(alvos) if alvos else None,
                )
            configs[nome] = cfg
        return configs


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
    """Anos de experiência já ponderados por recência (feito no backend)."""
    senioridade: int = 0
    """Nível atual do candidato (0–4)."""
    idiomas: dict[str, int] = field(default_factory=dict)
    """Mapa idioma -> nível (rank)."""
    competencias_comportamentais: list[str] = field(default_factory=list)
    """Até 3 habilidades declaradas no currículo."""


# ---------------------------------------------------------------------------
# Fatores auxiliares
# ---------------------------------------------------------------------------

def calcular_fator_experiencia(
    anos: float,
    tau: float = PARAMETROS_PADRAO.tau_experiencia,
) -> float:
    """
    Fator de experiência crescente e saturante: 1 − exp(−anos/τ).

    Mais experiência nunca reduz o fator. A recência já vem aplicada em
    `anos` pelo backend e não é aplicada de novo aqui.
    Com τ=3: 1 ano → 0.28, 3 anos → 0.63, 10 anos → 0.96.
    """
    return 1.0 - math.exp(-max(0.0, anos) / tau)


def calcular_fator_projetos(
    projetos: int,
    saturacao: int = PARAMETROS_PADRAO.projetos_saturacao,
) -> float:
    """Fração de projetos até o teto de saturação, em [0, 1]."""
    return min(max(0, projetos), saturacao) / saturacao


def calcular_fator_senioridade(
    senioridade_candidato: int,
    senioridade_vaga: int,
    parametros: ParametrosMatching = PARAMETROS_PADRAO,
) -> float:
    """
    Distância de senioridade: mesmo nível = 1.0; cada degrau reduz o fator.

    Assimétrica: estar abaixo da vaga pesa mais do que estar acima, pois a
    penalidade por "superqualificação" funciona como proxy de idade.
    """
    distancia = int(senioridade_candidato) - int(senioridade_vaga)
    penalidade = (
        parametros.penalidade_senioridade_abaixo * max(0, -distancia)
        + parametros.penalidade_senioridade_acima * max(0, distancia)
    )
    return max(parametros.fator_senioridade_minimo, 1.0 - penalidade)


# ---------------------------------------------------------------------------
# Algoritmo principal
# ---------------------------------------------------------------------------

def calcular_score_ponderado(
    candidato: Candidato,
    vaga: Vaga,
    parametros: ParametrosMatching = PARAMETROS_PADRAO,
) -> dict:
    """
    Calcula a cobertura ponderada e o filtro de obrigatórias de um candidato.

    Fórmula
    -------
    nivel_i      = nível do candidato, ou crédito parcial via competência relacionada
    cobertura_i  = min(nivel_i, alvo_i) / alvo_i                 (saturada no alvo)
    cobertura    = Σ peso_i × cobertura_i / Σ peso_i              ∈ [0, 1]
    fator_gate   = Π (1 − penalidade × déficit_i), para obrigatórias com déficit
                   dentro da tolerância; déficit acima da tolerância elimina
    score_final  = cobertura × fator_gate                          ∈ [0, 1]

    Retorna
    -------
    dict com as chaves:
        - candidato_id, nome_anonimo
        - score_base_normalizado : float ∈ [0, 1] (cobertura)
        - fator_gate             : float ∈ [0, 1]
        - score_final            : float ∈ [0, 1] (0 se eliminado)
        - ausencias_obrigatorias : int (lacunas eliminatórias)
        - obrigatorias_toleradas : int (lacunas dentro da tolerância)
        - aprovado_filtragem     : bool
        - detalhes_por_competencia : dict
    """
    competencias_candidato = normalizar_competencias(candidato.competencias)
    configs = vaga.configs()

    soma_pesos = 0.0
    soma_cobertura = 0.0
    fator_gate = 1.0
    ausencias_obrigatorias = 0
    obrigatorias_toleradas = 0
    detalhes: dict[str, dict] = {}

    for competencia, cfg in configs.items():
        nivel, origem = nivel_efetivo(competencias_candidato, competencia)
        alvo = cfg.alvo(parametros.nivel_alvo_padrao)
        cobertura_i = min(nivel, alvo) / alvo
        soma_pesos += cfg.peso
        soma_cobertura += cfg.peso * cobertura_i

        deficit = cfg.nivel_min - nivel if cfg.obrigatoria else 0.0
        lacuna_critica = deficit > parametros.tolerancia_obrigatoria
        tolerada = 0 < deficit <= parametros.tolerancia_obrigatoria
        if lacuna_critica:
            ausencias_obrigatorias += 1
        elif tolerada:
            obrigatorias_toleradas += 1
            fator_gate *= max(0.0, 1.0 - parametros.penalidade_por_nivel_faltante * deficit)

        detalhes[competencia] = {
            "peso": cfg.peso,
            "obrigatoria": cfg.obrigatoria,
            "nivel_min": cfg.nivel_min,
            "nivel_alvo": alvo,
            "nivel_candidato": round(nivel, 4),
            "credito_via": origem,
            "cobertura": round(cobertura_i, 4),
            "lacuna_critica": lacuna_critica,
            "abaixo_minimo_tolerado": tolerada,
        }

    cobertura = soma_cobertura / soma_pesos if soma_pesos > 0 else 0.0
    aprovado = ausencias_obrigatorias == 0
    if not aprovado:
        fator_gate = 0.0

    return {
        "candidato_id": candidato.id,
        "nome_anonimo": candidato.nome_anonimo,
        "score_base_normalizado": round(cobertura, 4),
        "fator_gate": round(fator_gate, 4),
        "score_final": round(cobertura * fator_gate, 4),
        "ausencias_obrigatorias": ausencias_obrigatorias,
        "obrigatorias_toleradas": obrigatorias_toleradas,
        "aprovado_filtragem": aprovado,
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
        Se True (padrão), filtra candidatos com lacunas obrigatórias eliminatórias.
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
