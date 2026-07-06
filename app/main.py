from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import rank_candidates
from app.schemas import MatchingRankRequest, MatchingRankResponse

app = FastAPI(
    title="My Chance NLP",
    description="Motor matemático de matching anônimo por competências técnicas.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mychance-nlp"}


@app.post("/api/v1/matching/rank", response_model=MatchingRankResponse)
def matching_rank(request: MatchingRankRequest) -> MatchingRankResponse:
    return rank_candidates(request)
