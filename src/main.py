from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.recommender.baseline import recommend_baseline_for_user
from src.recommender.pipeline import TrainArtifacts, train_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datafromdoctor"
TEMPLATES_DIR = BASE_DIR / "web"

app = FastAPI(title="BIA601 GA Recommender")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

state: dict[str, TrainArtifacts | None] = {"artifacts": None}


def ensure_trained(top_k: int = 10) -> TrainArtifacts:
    artifacts = state["artifacts"]
    if artifacts is not None:
        return artifacts
    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Data directory not found: {DATA_DIR}")
    artifacts = train_pipeline(str(DATA_DIR), top_k=top_k)
    state["artifacts"] = artifacts
    return artifacts


@app.on_event("startup")
def startup_train():
    try:
        ensure_trained(top_k=10)
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def home(request: Request, message: str | None = None, error: str | None = None):
    artifacts = state["artifacts"]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "trained": artifacts is not None,
            "metrics": artifacts.metrics if artifacts else None,
            "message": message,
            "error": error,
            "recommendation": None,
        },
    )


@app.post("/ui/recommend", response_class=HTMLResponse)
def recommend_from_ui(request: Request, user_id: int = Form(...), top_k: int = Form(default=10)):
    try:
        artifacts = ensure_trained(top_k=top_k)
        result = recommend(user_id=user_id, top_k=top_k)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "trained": True,
                "metrics": artifacts.metrics,
                "message": "تم توليد التوصيات بنجاح",
                "error": None,
                "recommendation": result,
            },
        )
    except HTTPException as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "trained": True,
                "metrics": artifacts.metrics,
                "message": None,
                "error": str(exc.detail),
                "recommendation": None,
            },
        )


@app.post("/train")
def train(top_k: int = 10):
    artifacts = ensure_trained(top_k=top_k)
    return {
        "message": "Training completed",
        "metrics": artifacts.metrics,
        "users_count": len(artifacts.data.users),
        "products_count": len(artifacts.data.products),
    }


@app.get("/metrics")
def metrics():
    artifacts = ensure_trained(top_k=10)
    return artifacts.metrics


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 10):
    artifacts = ensure_trained(top_k=top_k)

    user_exists = user_id in set(artifacts.data.users["user_id"].tolist())
    if not user_exists:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_id}")

    baseline_recs = recommend_baseline_for_user(
        user_id=user_id,
        model=artifacts.baseline,
        behavior=artifacts.data.behavior,
        top_k=top_k,
    )
    seen_products = set(
        artifacts.data.behavior.loc[
            artifacts.data.behavior["user_id"] == user_id, "product_id"
        ].tolist()
    )
    ga_recs = artifacts.ga.optimize_for_user(
        user_id=user_id, top_k=top_k, seen_products=seen_products
    )

    return {
        "user_id": user_id,
        "baseline_recommendations": baseline_recs,
        "ga_optimized_recommendations": ga_recs,
    }
