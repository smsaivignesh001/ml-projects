"""FastAPI recommendation service (step 6 of the PDF).

A small, production-style API over the trained hybrid recommender:

    GET /                      service info
    GET /health                liveness
    GET /recommend/{user_id}   personalized top-K (with reasons + strategy used)
    GET /similar/{item_id}     content-based item-to-item (works for cold-start items)
    GET /popular               popularity baseline top-K
    GET /metrics               offline Precision@K / Recall@K / NDCG@K per model
    GET /segments              recommendation analysis across user segments

Run:
    uvicorn api:app --reload --port 8000
Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src import config
from src.recommender import UnifiedRecommender, load_recommender

_state: dict = {"rec": None, "metrics": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.MODEL_PATH.exists():
        _state["rec"] = load_recommender(config.MODEL_PATH)
    if config.METRICS_JSON.exists():
        with open(config.METRICS_JSON) as f:
            _state["metrics"] = json.load(f)
    yield
    _state.clear()


app = FastAPI(
    title="Personalized Product Recommendation API",
    version="1.0.0",
    description="Hybrid recommender (ALS collaborative filtering + content-based + popularity).",
    lifespan=lifespan,
)


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    reason: str
    category: Optional[str] = None
    price: Optional[float] = None


class RecommendResponse(BaseModel):
    user_id: str
    strategy: str
    recommendations: list[RecommendedItem]


def _rec() -> UnifiedRecommender:
    rec = _state.get("rec")
    if rec is None:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train` first.")
    return rec


@app.get("/")
def root():
    return {
        "service": "Personalized Product Recommendation API",
        "model_loaded": _state.get("rec") is not None,
        "endpoints": ["/health", "/recommend/{user_id}", "/similar/{item_id}",
                      "/popular", "/metrics", "/segments"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state.get("rec") is not None}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: str, k: int = Query(10, ge=1, le=50)):
    rec = _rec()
    result = rec.recommend(user_id, k)
    items = [
        RecommendedItem(
            item_id=r["item_id"], score=r["score"], reason=r["reason"],
            category=r.get("category"), price=r.get("price"),
        )
        for r in result["recommendations"]
    ]
    return RecommendResponse(user_id=user_id, strategy=result["strategy"], recommendations=items)


@app.get("/similar/{item_id}")
def similar(item_id: str, k: int = Query(10, ge=1, le=50)):
    rec = _rec()
    sims = rec.similar_items(item_id, k)
    if not sims:
        raise HTTPException(404, f"Unknown item_id: {item_id}")
    return {"item_id": item_id, "similar_items": sims}


@app.get("/popular")
def popular(k: int = Query(10, ge=1, le=50)):
    rec = _rec()
    return {"strategy": "popularity", "items": rec.popularity.top_items(k)}


@app.get("/metrics")
def metrics():
    if _state.get("metrics") is None:
        raise HTTPException(503, "Metrics not available. Run `python -m src.train`.")
    return _state["metrics"]


@app.get("/segments")
def segments():
    if _state.get("metrics") is None:
        raise HTTPException(503, "Metrics not available. Run `python -m src.train`.")
    return _state["metrics"].get("segment_analysis", [])
