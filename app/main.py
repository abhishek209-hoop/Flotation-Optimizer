"""FastAPI entry point. Run: uvicorn app.main:app --host 127.0.0.1 --port 8000"""
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from .service import Service

app = FastAPI(title="Flotation Optimizer")
svc = Service()
INDEX = Path(__file__).parent / "static" / "index.html"


class Req(BaseModel):
    algo: str = "xgboost"                      # "xgboost" | "random_forest"
    inputs: dict[str, Any]
    target: Optional[Any] = None               # desired recovery %, optional
    max_change_pct: Any = 25                   # largest move allowed per variable


@app.get("/")
def index():
    return FileResponse(INDEX)


@app.get("/api/meta")
def meta():
    return svc.public_meta()


@app.post("/api/analyze")
def analyze(req: Req):
    out = svc.analyze(req.model_dump())
    return JSONResponse(out, status_code=422 if "errors" in out else 200)
