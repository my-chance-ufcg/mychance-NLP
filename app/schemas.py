from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.matching.parametros import PARAMETROS_PADRAO as _P
from app.matching.parametros import ParametrosMatching


class JobRequirementSchema(BaseModel):
    peso: int = Field(ge=1, le=5)
    obrigatoria: bool = False
    nivel_min: int = Field(default=1, ge=1, le=5, alias="nivel_min")
    nivel_alvo: Optional[int] = Field(default=None, ge=1, le=5)

    model_config = {"populate_by_name": True}


class JobMatchingSchema(BaseModel):
    competencias: dict[str, JobRequirementSchema]
    senioridade: int = Field(default=0, ge=0, le=5)
    idiomas: dict[str, int] = Field(default_factory=dict)
    competencias_comportamentais: list[str] = Field(default_factory=list, max_length=3)


class CandidateMatchingSchema(BaseModel):
    candidato_id: str
    competencias: dict[str, int] = Field(default_factory=dict)
    projetos: int = Field(default=0, ge=0)
    anos_experiencia: float = Field(default=0.0, ge=0.0, alias="anos_experiencia")
    senioridade: int = Field(default=0, ge=0, le=5)
    idiomas: dict[str, int] = Field(default_factory=dict)
    competencias_comportamentais: list[str] = Field(default_factory=list, max_length=3)

    model_config = {"populate_by_name": True}


class ParametrosSchema(BaseModel):
    """Ajustes do algoritmo por requisição. Limites impedem configurações enviesadas."""

    tolerancia_obrigatoria: int = Field(default=_P.tolerancia_obrigatoria, ge=0, le=2)
    penalidade_por_nivel_faltante: float = Field(default=_P.penalidade_por_nivel_faltante, ge=0.0, le=0.5)
    nivel_alvo_padrao: int = Field(default=_P.nivel_alvo_padrao, ge=1, le=5)
    tau_experiencia: float = Field(default=_P.tau_experiencia, ge=0.5, le=10.0)
    projetos_saturacao: int = Field(default=_P.projetos_saturacao, ge=1, le=10)
    peso_cobertura: float = Field(default=_P.peso_cobertura, ge=0.0, le=1.0)
    peso_aderencia: float = Field(default=_P.peso_aderencia, ge=0.0, le=1.0)
    peso_experiencia: float = Field(default=_P.peso_experiencia, ge=0.0, le=1.0)
    peso_projetos: float = Field(default=_P.peso_projetos, ge=0.0, le=1.0)
    penalidade_senioridade_abaixo: float = Field(default=_P.penalidade_senioridade_abaixo, ge=0.0, le=0.3)
    penalidade_senioridade_acima: float = Field(default=_P.penalidade_senioridade_acima, ge=0.0, le=0.1)
    fator_senioridade_minimo: float = Field(default=_P.fator_senioridade_minimo, ge=0.2, le=1.0)

    @model_validator(mode="after")
    def _pesos_nao_nulos(self) -> "ParametrosSchema":
        if self.peso_cobertura + self.peso_aderencia + self.peso_experiencia + self.peso_projetos <= 0:
            raise ValueError("Ao menos um peso de combinação deve ser positivo")
        return self

    def to_domain(self) -> ParametrosMatching:
        return ParametrosMatching(**self.model_dump())


class MatchingRankRequest(BaseModel):
    vocabulario: list[str] = Field(
        default_factory=list,
        description="Obsoleto: a comparação usa apenas as competências da vaga. Aceito por compatibilidade.",
    )
    vaga: JobMatchingSchema
    candidatos: list[CandidateMatchingSchema]
    parametros: Optional[ParametrosSchema] = None

    @model_validator(mode="after")
    def _ids_unicos(self) -> "MatchingRankRequest":
        ids = [c.candidato_id for c in self.candidatos]
        if len(ids) != len(set(ids)):
            raise ValueError("candidato_id duplicado na lista de candidatos")
        return self


class RankedCandidateResponse(BaseModel):
    candidato_id: str
    compatibilidade_score: float
    compatibilidade: str
    aprovado_filtragem: bool
    score_filtragem: Optional[float] = None
    componentes: dict[str, float] = Field(default_factory=dict)
    explicacao: list[str] = Field(default_factory=list)


class MatchingRankResponse(BaseModel):
    ranking: list[RankedCandidateResponse]
