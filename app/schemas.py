from typing import Optional

from pydantic import BaseModel, Field


class JobRequirementSchema(BaseModel):
    peso: int = Field(ge=1, le=5)
    obrigatoria: bool = False
    nivel_min: int = Field(default=1, ge=1, le=5, alias="nivel_min")

    model_config = {"populate_by_name": True}


class JobMatchingSchema(BaseModel):
    competencias: dict[str, JobRequirementSchema]


class CandidateMatchingSchema(BaseModel):
    candidato_id: str
    competencias: dict[str, int] = Field(default_factory=dict)
    projetos: int = Field(default=0, ge=0)
    anos_experiencia: float = Field(default=0.0, ge=0.0, alias="anos_experiencia")

    model_config = {"populate_by_name": True}


class MatchingRankRequest(BaseModel):
    vocabulario: list[str]
    vaga: JobMatchingSchema
    candidatos: list[CandidateMatchingSchema]


class RankedCandidateResponse(BaseModel):
    candidato_id: str
    compatibilidade_score: float
    compatibilidade: str
    aprovado_filtragem: bool
    score_filtragem: Optional[float] = None


class MatchingRankResponse(BaseModel):
    ranking: list[RankedCandidateResponse]
