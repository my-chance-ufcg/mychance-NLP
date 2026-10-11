from app.matching.cosine_similarity import rotulo_compatibilidade
from app.matching.parametros import PARAMETROS_PADRAO
from app.matching.ranking import rankear
from app.matching.weighted_scoring import Candidato, Vaga
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
            "nivel_alvo": cfg.nivel_alvo,
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

    parametros = request.parametros.to_domain() if request.parametros else PARAMETROS_PADRAO
    vaga = _to_vaga(request.vaga)
    candidatos = [_to_candidato(candidate, index) for index, candidate in enumerate(request.candidatos)]

    ranking = [
        RankedCandidateResponse(
            candidato_id=resultado.nome_anonimo,
            compatibilidade_score=resultado.score,
            compatibilidade=rotulo_compatibilidade(resultado.score) if resultado.aprovado else "Eliminado",
            aprovado_filtragem=resultado.aprovado,
            score_filtragem=resultado.score_filtragem,
            componentes=resultado.componentes,
            explicacao=resultado.explicacao,
        )
        for resultado in rankear(candidatos, vaga, parametros)
    ]
    return MatchingRankResponse(ranking=ranking)
