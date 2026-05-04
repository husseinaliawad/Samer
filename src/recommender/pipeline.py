from __future__ import annotations

from dataclasses import dataclass
import random

from .baseline import (
    BaselineModel,
    build_baseline_scores,
    ndcg_at_k,
    precision_recall_at_k,
    recommend_baseline_for_user,
)
from .data_loader import DatasetBundle, load_data
from .genetic_optimizer import GARecommender


@dataclass
class TrainArtifacts:
    data: DatasetBundle
    baseline: BaselineModel
    ga: GARecommender
    metrics: dict


def train_pipeline(data_dir: str, top_k: int = 10) -> TrainArtifacts:
    data = load_data(data_dir)
    baseline = build_baseline_scores(data.ratings, data.behavior)
    candidates = baseline.score_table["product_id"].tolist()
    ga = GARecommender(candidates=candidates, behavior=data.behavior)

    purchased_rows = data.behavior[data.behavior["purchased"] == 1].copy()
    user_purchase_groups = purchased_rows.groupby("user_id")
    eligible_users = [uid for uid, grp in user_purchase_groups if len(grp) >= 2]
    if len(eligible_users) > 100:
        eligible_users = eligible_users[:100]

    rng = random.Random(42)
    holdout_map: dict[int, int] = {}
    for uid in eligible_users:
        user_products = user_purchase_groups.get_group(uid)["product_id"].tolist()
        holdout_map[uid] = rng.choice(user_products)

    train_behavior = data.behavior.copy()
    for uid, held_pid in holdout_map.items():
        mask = (train_behavior["user_id"] == uid) & (train_behavior["product_id"] == held_pid)
        train_behavior = train_behavior.loc[~mask]

    eval_baseline = build_baseline_scores(data.ratings, train_behavior)
    eval_candidates = eval_baseline.score_table["product_id"].tolist()
    eval_ga = GARecommender(candidates=eval_candidates, behavior=train_behavior)

    p_base, r_base, n_base = [], [], []
    p_ga, r_ga, n_ga = [], [], []

    for uid in eligible_users:
        truth = {holdout_map[uid]}
        seen_train = set(
            train_behavior.loc[train_behavior["user_id"] == uid, "product_id"].tolist()
        )

        base_pred = recommend_baseline_for_user(uid, eval_baseline, train_behavior, top_k=top_k)
        ga_pred = eval_ga.optimize_for_user(uid, top_k=top_k, seen_products=seen_train)

        pb, rb = precision_recall_at_k(base_pred, truth, top_k)
        pg, rg = precision_recall_at_k(ga_pred, truth, top_k)

        p_base.append(pb)
        r_base.append(rb)
        n_base.append(ndcg_at_k(base_pred, truth, top_k))

        p_ga.append(pg)
        r_ga.append(rg)
        n_ga.append(ndcg_at_k(ga_pred, truth, top_k))

    metrics = {
        "baseline": {
            "precision_at_k": round(sum(p_base) / max(len(p_base), 1), 4),
            "recall_at_k": round(sum(r_base) / max(len(r_base), 1), 4),
            "ndcg_at_k": round(sum(n_base) / max(len(n_base), 1), 4),
        },
        "ga_optimized": {
            "precision_at_k": round(sum(p_ga) / max(len(p_ga), 1), 4),
            "recall_at_k": round(sum(r_ga) / max(len(r_ga), 1), 4),
            "ndcg_at_k": round(sum(n_ga) / max(len(n_ga), 1), 4),
        },
    }

    return TrainArtifacts(data=data, baseline=baseline, ga=ga, metrics=metrics)
