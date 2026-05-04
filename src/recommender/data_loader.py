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


def load_data(data_dir: str | Path) -> DatasetBundle:
    data_path = Path(data_dir)
    users = pd.read_excel(data_path / "users.xlsx")
    products = pd.read_excel(data_path / "products.xlsx")
    ratings = pd.read_excel(data_path / "ratings.xlsx")
    behavior = pd.read_excel(data_path / "behavior_15500.xlsx")

    users.columns = [c.strip().lower() for c in users.columns]
    products.columns = [c.strip().lower() for c in products.columns]
    ratings.columns = [c.strip().lower() for c in ratings.columns]
    behavior.columns = [c.strip().lower() for c in behavior.columns]

    for col in ["viewed", "clicked", "purchased"]:
        if col in behavior.columns:
            behavior[col] = behavior[col].fillna(0).astype(int)

    return DatasetBundle(users=users, products=products, ratings=ratings, behavior=behavior)
