from __future__ import annotations

from pathlib import Path
import random
import os

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.recommender.data_loader import DatasetBundle, load_data

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "web"

app = FastAPI(title="BIA601 Recommender")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

state: dict[str, object | None] = {
    "data": None,
    "product_scores": None,
}


def resolve_data_dir() -> Path:
    env_data_dir = os.getenv("BIA_DATA_DIR")
    if env_data_dir:
        p = Path(env_data_dir)
        if p.exists():
            return p

    candidates = [
        BASE_DIR / "datafromdoctor",
        BASE_DIR / "HW__Data_S25",
        BASE_DIR / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return BASE_DIR / "datafromdoctor"


def ensure_data_ready() -> tuple[DatasetBundle, pd.DataFrame]:
    data = state["data"]
    product_scores = state["product_scores"]

    if data is not None and product_scores is not None:
        return data, product_scores

    data_dir = resolve_data_dir()
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Data directory not found: {data_dir}")

    data = load_data(data_dir)

    rating_score = data.ratings.groupby("product_id", as_index=False)["rating"].mean().rename(
        columns={"rating": "avg_rating"}
    )
    behavior = data.behavior.copy()
    behavior["engagement"] = (
        0.2 * behavior.get("viewed", 0)
        + 0.3 * behavior.get("clicked", 0)
        + 0.5 * behavior.get("purchased", 0)
    )
    engagement = behavior.groupby("product_id", as_index=False)["engagement"].mean()

    product_scores = rating_score.merge(engagement, on="product_id", how="outer").fillna(0.0)
    product_scores = product_scores.merge(
        data.products[["product_id", "category"]], on="product_id", how="left"
    )
    product_scores["base_score"] = 0.65 * product_scores["avg_rating"] + 0.35 * (
        product_scores["engagement"] * 5
    )

    state["data"] = data
    state["product_scores"] = product_scores
    return data, product_scores


def recommend_heuristic(user_id: int, top_k: int = 10) -> dict:
    data, product_scores = ensure_data_ready()

    user_exists = user_id in set(data.users["user_id"].tolist())
    if not user_exists:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_id}")

    user_behavior = data.behavior[data.behavior["user_id"] == user_id].copy()
    seen_products = set(user_behavior["product_id"].tolist())

    merged = user_behavior.merge(
        data.products[["product_id", "category"]], on="product_id", how="left"
    )
    merged["signal"] = (
        0.2 * merged.get("viewed", 0) + 0.3 * merged.get("clicked", 0) + 0.5 * merged.get("purchased", 0)
    )
    pref = merged.groupby("category", as_index=False)["signal"].mean()
    pref_map = {row["category"]: float(row["signal"]) for _, row in pref.iterrows()}

    candidates = product_scores[~product_scores["product_id"].isin(seen_products)].copy()
    candidates["pref_bonus"] = candidates["category"].map(lambda c: pref_map.get(c, 0.0))
    candidates["final_score"] = candidates["base_score"] + 1.2 * candidates["pref_bonus"]
    top = candidates.sort_values("final_score", ascending=False).head(top_k)
    items = []
    for _, row in top.iterrows():
        items.append(
            {
                "product_id": int(row["product_id"]),
                "category": str(row["category"]) if pd.notna(row["category"]) else "غير محدد",
                "price": float(row["price"]) if "price" in row and pd.notna(row["price"]) else None,
            }
        )

    return {
        "user_id": user_id,
        "recommendations": items,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request, message: str | None = None, error: str | None = None):
    sample_user_ids: list[int] = []
    try:
        data, _ = ensure_data_ready()
        sample_user_ids = sorted(data.users["user_id"].astype(int).tolist())[:50]
    except Exception:
        sample_user_ids = []

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": message,
            "error": error,
            "recommendation": None,
            "sample_user_ids": sample_user_ids,
        },
    )


@app.post("/ui/recommend", response_class=HTMLResponse)
def recommend_from_ui(request: Request, user_id: int | None = Form(default=None), top_k: int = Form(default=10)):
    try:
        if user_id is None:
            data, _ = ensure_data_ready()
            user_id = int(random.choice(data.users["user_id"].tolist()))
        result = recommend_heuristic(user_id=user_id, top_k=top_k)
        data, _ = ensure_data_ready()
        sample_user_ids = sorted(data.users["user_id"].astype(int).tolist())[:50]
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": "تم توليد التوصيات بنجاح",
                "error": None,
                "recommendation": result,
                "sample_user_ids": sample_user_ids,
            },
        )
    except HTTPException as exc:
        sample_user_ids: list[int] = []
        try:
            data, _ = ensure_data_ready()
            sample_user_ids = sorted(data.users["user_id"].astype(int).tolist())[:50]
        except Exception:
            pass
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": None,
                "error": str(exc.detail),
                "recommendation": None,
                "sample_user_ids": sample_user_ids,
            },
        )


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 10):
    return recommend_heuristic(user_id=user_id, top_k=top_k)
