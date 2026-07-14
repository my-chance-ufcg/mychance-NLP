from app.matching.cosine_similarity import construir_matriz_scores, rotulo_compatibilidade
from app.matching.weighted_scoring import Candidato, Vaga, calcular_fator_senioridade, rankear_candidatos
from app.schemas import (
    CandidateMatchingSchema,
    JobMatchingSchema,
    MatchingRankRequest,
    MatchingRankResponse,
    RankedCandidateResponse,
)


def _to_vaga(job: JobMatchingSchema) -> Vaga:
    competencias = {
        skill: {
            "peso": cfg.peso,
            "obrigatoria": cfg.obrigatoria,
            "nivel_min": cfg.nivel_min,
        }
        for skill, cfg in job.competencias.items()
    }
    return Vaga(
        id=0,
        titulo="vaga",
        competencias=competencias,
        senioridade=job.senioridade,
        idiomas=dict(job.idiomas),
    )


def _to_candidato(candidate: CandidateMatchingSchema, index: int) -> Candidato:
    return Candidato(
        id=index,
        nome_anonimo=candidate.candidato_id,
        competencias={skill: float(level) for skill, level in candidate.competencias.items()},
        projetos=candidate.projetos,
        anos_experiencia=candidate.anos_experiencia,
        senioridade=candidate.senioridade,
        idiomas=dict(candidate.idiomas),
    )


def rank_candidates(request: MatchingRankRequest) -> MatchingRankResponse:
    if not request.candidatos:
        return MatchingRankResponse(ranking=[])

    vaga = _to_vaga(request.vaga)
    candidatos = [_to_candidato(candidate, index) for index, candidate in enumerate(request.candidatos)]

    filtering_results = {
        result["nome_anonimo"]: result
        for result in rankear_candidatos(candidatos, vaga, apenas_aprovados=False)
    }

    approved_ids = {
        candidate_id
        for candidate_id, result in filtering_results.items()
        if result["aprovado_filtragem"]
    }

    vaga_dict = {
        "competencias": {skill: 1 for skill in request.vaga.competencias},
        "pesos": {skill: cfg.peso for skill, cfg in request.vaga.competencias.items()},
    }

    cosine_inputs = [
        {
            "id": candidate.candidato_id,
            "competencias": {skill: float(level) for skill, level in candidate.competencias.items()},
        }
        for candidate in request.candidatos
        if candidate.candidato_id in approved_ids
    ]

    cosine_scores: dict[str, float] = {}
    if cosine_inputs:
        ranking_df = construir_matriz_scores(
            cosine_inputs,
            vaga_dict,
            request.vocabulario,
            usar_pesos=True,
        )
        for _, row in ranking_df.reset_index().iterrows():
            cosine_scores[str(row["candidato_id"])] = float(row["similaridade"])

    ranking: list[RankedCandidateResponse] = []
    for candidate in request.candidatos:
        filtering = filtering_results[candidate.candidato_id]
        approved = filtering["aprovado_filtragem"]
        base_score = cosine_scores.get(candidate.candidato_id, 0.0) if approved else 0.0
        fator_senioridade = calcular_fator_senioridade(
            candidate.senioridade,
            request.vaga.senioridade,
        )
        score = base_score * fator_senioridade if approved else 0.0
        ranking.append(
            RankedCandidateResponse(
                candidato_id=candidate.candidato_id,
                compatibilidade_score=round(score, 4),
                compatibilidade=rotulo_compatibilidade(score) if approved else "Eliminado",
                aprovado_filtragem=approved,
                score_filtragem=filtering["score_final"],
            )
        )

    ranking.sort(key=lambda item: item.compatibilidade_score, reverse=True)
    return MatchingRankResponse(ranking=ranking)
