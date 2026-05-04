from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass
class GAConfig:
    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8


class GARecommender:
    def __init__(self, candidates: list[int], behavior: pd.DataFrame, config: GAConfig | None = None):
        self.candidates = candidates
        self.behavior = behavior
        self.config = config or GAConfig()
        behavior_copy = behavior.copy()
        behavior_copy["item_signal"] = (
            0.2 * behavior_copy.get("viewed", 0)
            + 0.3 * behavior_copy.get("clicked", 0)
            + 0.5 * behavior_copy.get("purchased", 0)
        )
        self.product_prior = (
            behavior_copy.groupby("product_id")["item_signal"].mean().to_dict()
        )

    def _fitness(self, chromosome: list[int], user_id: int) -> float:
        if not chromosome:
            return 0.0
        scores = []
        user_subset = self.behavior[self.behavior["user_id"] == user_id]
        for pid in chromosome:
            row = user_subset[user_subset["product_id"] == pid]
            if not row.empty:
                viewed = row["viewed"].mean() if "viewed" in row else 0.0
                clicked = row["clicked"].mean() if "clicked" in row else 0.0
                purchased = row["purchased"].mean() if "purchased" in row else 0.0
                scores.append(0.2 * viewed + 0.3 * clicked + 0.5 * purchased)
            else:
                scores.append(float(self.product_prior.get(pid, 0.0)))
        return sum(scores) / len(scores)

    def _init_population(self, top_k: int) -> list[list[int]]:
        pop = []
        for _ in range(self.config.population_size):
            pop.append(random.sample(self.candidates, k=min(top_k, len(self.candidates))))
        return pop

    def _select(self, population: list[list[int]], scores: list[float]) -> list[int]:
        total = sum(scores)
        if total == 0:
            return random.choice(population)
        pick = random.uniform(0, total)
        current = 0.0
        for c, s in zip(population, scores):
            current += s
            if current >= pick:
                return c
        return population[-1]

    def _crossover(self, p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
        if len(p1) < 2 or random.random() > self.config.crossover_rate:
            return p1[:], p2[:]
        point = random.randint(1, len(p1) - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return self._repair(c1), self._repair(c2)

    def _mutate(self, chrom: list[int]) -> list[int]:
        c = chrom[:]
        for i in range(len(c)):
            if random.random() < self.config.mutation_rate:
                c[i] = random.choice(self.candidates)
        return self._repair(c)

    def _repair(self, chrom: list[int]) -> list[int]:
        seen = set()
        repaired = []
        for pid in chrom:
            if pid not in seen:
                repaired.append(pid)
                seen.add(pid)
        while len(repaired) < len(chrom):
            candidate = random.choice(self.candidates)
            if candidate not in seen:
                repaired.append(candidate)
                seen.add(candidate)
        return repaired

    def optimize_for_user(self, user_id: int, top_k: int = 10, seen_products: set[int] | None = None) -> list[int]:
        seen_products = seen_products or set()
        candidate_pool = [pid for pid in self.candidates if pid not in seen_products]
        if not candidate_pool:
            return []

        original_candidates = self.candidates
        self.candidates = candidate_pool
        population = self._init_population(top_k)

        for _ in range(self.config.generations):
            scores = [self._fitness(ch, user_id) for ch in population]
            new_population = []

            while len(new_population) < self.config.population_size:
                p1 = self._select(population, scores)
                p2 = self._select(population, scores)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[: self.config.population_size]

        final_scores = [self._fitness(ch, user_id) for ch in population]
        best_idx = max(range(len(population)), key=lambda i: final_scores[i])
        best = population[best_idx][:top_k]
        self.candidates = original_candidates
        return best
