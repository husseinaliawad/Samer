from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselineModel:
    score_table: pd.DataFrame


def build_baseline_scores(ratings: pd.DataFrame, behavior: pd.DataFrame) -> BaselineModel:
    rating_score = ratings.groupby("product_id", as_index=False)["rating"].mean().rename(columns={"rating": "avg_rating"})

    behavior = behavior.copy()
    behavior["engagement"] = (
        0.2 * behavior.get("viewed", 0)
        + 0.3 * behavior.get("clicked", 0)
        + 0.5 * behavior.get("purchased", 0)
    )
    engagement = behavior.groupby("product_id", as_index=False)["engagement"].mean()

    merged = rating_score.merge(engagement, on="product_id", how="outer").fillna(0.0)
    merged["score"] = 0.6 * merged["avg_rating"] + 0.4 * (merged["engagement"] * 5)
    merged = merged.sort_values("score", ascending=False).reset_index(drop=True)

    return BaselineModel(score_table=merged)


def recommend_baseline_for_user(
    user_id: int,
    model: BaselineModel,
    behavior: pd.DataFrame,
    top_k: int = 10,
) -> list[int]:
    seen = set(behavior.loc[behavior["user_id"] == user_id, "product_id"].tolist())
    recs = [pid for pid in model.score_table["product_id"].tolist() if pid not in seen]
    return recs[:top_k]


def precision_recall_at_k(pred: list[int], truth: set[int], k: int = 10) -> tuple[float, float]:
    if not pred:
        return 0.0, 0.0
    pred_k = pred[:k]
    hits = sum(1 for p in pred_k if p in truth)
    precision = hits / max(len(pred_k), 1)
    recall = hits / max(len(truth), 1) if truth else 0.0
    return precision, recall


def ndcg_at_k(pred: list[int], truth: set[int], k: int = 10) -> float:
    pred_k = pred[:k]
    dcg = 0.0
    for i, p in enumerate(pred_k, start=1):
        rel = 1.0 if p in truth else 0.0
        dcg += rel / np.log2(i + 1)

    ideal_hits = min(len(truth), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return float(dcg / idcg)
