from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from claimlens.models import ClaimResult
from claimlens.pipeline import run

app = FastAPI(title="ClaimLens", version="0.5.0")


class VerifyRequest(BaseModel):
    claim: str


@app.post("/verify", response_model=ClaimResult)
def verify(req: VerifyRequest) -> ClaimResult:
    try:
        return run(req.claim)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
