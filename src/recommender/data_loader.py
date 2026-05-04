from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetBundle:
    users: pd.DataFrame
    products: pd.DataFrame
    ratings: pd.DataFrame
    behavior: pd.DataFrame


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_dataset_file(data_path: Path, logical_name: str) -> Path:
    files = [
        p
        for p in data_path.iterdir()
        if p.is_file() and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
    ]
    if not files:
        raise FileNotFoundError(f"No dataset files found in: {data_path}")

    exact_candidates = [p for p in files if p.stem.strip().lower() == logical_name]
    if exact_candidates:
        return exact_candidates[0]

    prefix_candidates = [p for p in files if p.stem.strip().lower().startswith(logical_name)]
    if prefix_candidates:
        return sorted(prefix_candidates, key=lambda x: len(x.name))[0]

    contains_candidates = [p for p in files if logical_name in p.stem.strip().lower()]
    if contains_candidates:
        return sorted(contains_candidates, key=lambda x: len(x.name))[0]

    raise FileNotFoundError(f"Could not find a file for '{logical_name}' in {data_path}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _require_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def load_data(data_dir: str | Path) -> DatasetBundle:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    users = _normalize_columns(_read_table(_find_dataset_file(data_path, "users")))
    products = _normalize_columns(_read_table(_find_dataset_file(data_path, "products")))
    ratings = _normalize_columns(_read_table(_find_dataset_file(data_path, "ratings")))
    behavior = _normalize_columns(_read_table(_find_dataset_file(data_path, "behavior")))

    _require_columns(users, {"user_id"}, "users")
    _require_columns(products, {"product_id"}, "products")
    _require_columns(ratings, {"user_id", "product_id", "rating"}, "ratings")
    _require_columns(behavior, {"user_id", "product_id"}, "behavior")

    users["user_id"] = pd.to_numeric(users["user_id"], errors="coerce")
    products["product_id"] = pd.to_numeric(products["product_id"], errors="coerce")

    for col in ["user_id", "product_id", "rating"]:
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")

    for col in ["user_id", "product_id"]:
        behavior[col] = pd.to_numeric(behavior[col], errors="coerce")

    users = users.dropna(subset=["user_id"]).copy()
    products = products.dropna(subset=["product_id"]).copy()
    ratings = ratings.dropna(subset=["user_id", "product_id", "rating"]).copy()
    behavior = behavior.dropna(subset=["user_id", "product_id"]).copy()

    for col in ["viewed", "clicked", "purchased"]:
        if col in behavior.columns:
            behavior[col] = pd.to_numeric(behavior[col], errors="coerce").fillna(0).astype(int)
        else:
            behavior[col] = 0

    users["user_id"] = users["user_id"].astype(int)
    products["product_id"] = products["product_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["product_id"] = ratings["product_id"].astype(int)
    behavior["user_id"] = behavior["user_id"].astype(int)
    behavior["product_id"] = behavior["product_id"].astype(int)

    return DatasetBundle(users=users, products=products, ratings=ratings, behavior=behavior)
