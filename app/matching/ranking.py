"""
ranking.py
==========
Combinação das etapas do pipeline de matching em um ranking explicável.

    score = fator_gate × fator_senioridade ×
            (w_cob·cobertura + w_ade·aderência + w_exp·experiência + w_proj·projetos)

Fatores multiplicativos representam requisitos (não se compensam); a soma
ponderada reúne qualidades que se compensam. Os pesos são normalizados para
somar 1, de modo que o score fica em [0, 1] por construção.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.matching.cosine_similarity import calcular_aderencia
from app.matching.parametros import PARAMETROS_PADRAO, ParametrosMatching
from app.matching.weighted_scoring import (
    Candidato,
    Vaga,
    calcular_fator_experiencia,
    calcular_fator_projetos,
    calcular_fator_senioridade,
    calcular_score_ponderado,
)


@dataclass
class ResultadoMatching:
    nome_anonimo: str
    score: float
    aprovado: bool
    score_filtragem: float
    componentes: dict[str, float] = field(default_factory=dict)
    explicacao: list[str] = field(default_factory=list)


def _percentual_atendido(grupo: list[dict], campo: str = "nivel_min") -> int | None:
    if not grupo:
        return None
    atendidas = sum(1 for item in grupo if item["nivel_candidato"] >= item[campo])
    return round(100 * atendidas / len(grupo))


def _resumo_cobertura(detalhes: dict[str, dict]) -> str | None:
    obrigatorias = [item for item in detalhes.values() if item["obrigatoria"]]
    desejaveis = [item for item in detalhes.values() if not item["obrigatoria"]]
    partes: list[str] = []
    cobertura_obrigatorias = _percentual_atendido(obrigatorias)
    if cobertura_obrigatorias is not None:
        partes.append(f"{cobertura_obrigatorias}% das competências obrigatórias")
    cobertura_desejaveis = _percentual_atendido(desejaveis)
    if cobertura_desejaveis is not None:
        partes.append(f"{cobertura_desejaveis}% das desejáveis")
    if not partes:
        return None
    return "Candidato possui " + " e ".join(partes) + "."


BONUS_COMPORTAMENTAL = 0.06


def _bonus_comportamental(candidato: Candidato, vaga: Vaga) -> tuple[float, str | None]:
    pedidas = {skill.strip().casefold() for skill in vaga.competencias_comportamentais if skill.strip()}
    if not pedidas:
        return 0.0, None
    declaradas = {skill.strip().casefold() for skill in candidato.competencias_comportamentais if skill.strip()}
    coincidencias = len(pedidas & declaradas)
    bonus = BONUS_COMPORTAMENTAL * coincidencias / len(pedidas)
    texto = (
        f"Competências comportamentais: {coincidencias} de {len(pedidas)} "
        "coincidem com a vaga."
    )
    return bonus, texto


def _explicar(detalhes: dict[str, dict]) -> list[str]:
    motivos: list[str] = []
    resumo = _resumo_cobertura(detalhes)
    if resumo:
        motivos.append(resumo)
    for skill, d in detalhes.items():
        nivel = d["nivel_candidato"]
        if d["lacuna_critica"]:
            motivos.append(f"{skill}: nível {nivel:g} abaixo do mínimo {d['nivel_min']} (eliminatório)")
        elif d["abaixo_minimo_tolerado"]:
            motivos.append(f"{skill}: nível {nivel:g} abaixo do mínimo {d['nivel_min']} (penalizado)")
        elif nivel < d["nivel_alvo"]:
            motivos.append(f"{skill}: nível {nivel:g} abaixo do alvo {d['nivel_alvo']}")
        if d["credito_via"]:
            motivos.append(f"{skill}: crédito parcial via {d['credito_via']}")
    return motivos


def avaliar_candidato(
    candidato: Candidato,
    vaga: Vaga,
    parametros: ParametrosMatching = PARAMETROS_PADRAO,
) -> ResultadoMatching:
    filtragem = calcular_score_ponderado(candidato, vaga, parametros)
    detalhes = filtragem["detalhes_por_competencia"]
    explicacao = _explicar(detalhes)

    if not filtragem["aprovado_filtragem"]:
        return ResultadoMatching(
            nome_anonimo=candidato.nome_anonimo,
            score=0.0,
            aprovado=False,
            score_filtragem=filtragem["score_final"],
            componentes={"cobertura": filtragem["score_base_normalizado"], "gate": 0.0},
            explicacao=explicacao,
        )

    componentes = {
        "cobertura": filtragem["score_base_normalizado"],
        "aderencia": calcular_aderencia(
            alvos={s: d["nivel_alvo"] for s, d in detalhes.items()},
            niveis={s: d["nivel_candidato"] for s, d in detalhes.items()},
            pesos={s: d["peso"] for s, d in detalhes.items()},
        ),
        "experiencia": calcular_fator_experiencia(candidato.anos_experiencia, parametros.tau_experiencia),
        "projetos": calcular_fator_projetos(candidato.projetos, parametros.projetos_saturacao),
    }
    pesos = {
        "cobertura": parametros.peso_cobertura,
        "aderencia": parametros.peso_aderencia,
        "experiencia": parametros.peso_experiencia,
        "projetos": parametros.peso_projetos,
    }
    soma_pesos = sum(pesos.values())
    base = sum(pesos[k] * componentes[k] for k in pesos) / soma_pesos if soma_pesos > 0 else 0.0

    componentes["gate"] = filtragem["fator_gate"]
    componentes["senioridade"] = calcular_fator_senioridade(candidato.senioridade, vaga.senioridade, parametros)
    score = base * componentes["gate"] * componentes["senioridade"]
    bonus, texto_comportamental = _bonus_comportamental(candidato, vaga)
    if texto_comportamental:
        componentes["comportamentais"] = round(bonus / BONUS_COMPORTAMENTAL, 4)
        explicacao = [*explicacao, texto_comportamental]
        score += bonus

    return ResultadoMatching(
        nome_anonimo=candidato.nome_anonimo,
        score=round(min(1.0, max(0.0, score)), 4),
        aprovado=True,
        score_filtragem=filtragem["score_final"],
        componentes={k: round(v, 4) for k, v in componentes.items()},
        explicacao=explicacao,
    )


def rankear(
    candidatos: list[Candidato],
    vaga: Vaga,
    parametros: ParametrosMatching = PARAMETROS_PADRAO,
) -> list[ResultadoMatching]:
    """Ranking completo (eliminados ao final com score 0), com desempate determinístico."""
    ids = [c.nome_anonimo for c in candidatos]
    if len(ids) != len(set(ids)):
        raise ValueError("Identificadores de candidato duplicados")

    resultados = [avaliar_candidato(c, vaga, parametros) for c in candidatos]
    return sorted(
        resultados,
        key=lambda r: (-r.score, -r.componentes.get("cobertura", 0.0), r.nome_anonimo),
    )
