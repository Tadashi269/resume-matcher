"""Evaluate the matching methods against hand-labelled relevance scores.

Usage:
    python evaluate.py                 # auto: SBERT if available, else LSA
    python evaluate.py --backend lsa   # force the offline LSA model

Each (job, resume) pair in data/labels.csv has a relevance grade:
    3 = ideal candidate, 2 = good, 1 = partially relevant, 0 = not relevant

Metrics (averaged over jobs):
    NDCG@5     - quality of the top-5 ranking, graded relevance
    MRR        - 1 / rank of the best (grade-3) candidate
    Top-1 Acc  - share of jobs where the best candidate is ranked first
    Spearman   - rank correlation between scores and relevance grades
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from matcher import ResumeMatcher, extract_text

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "results"


def dcg(rels: list[float], k: int) -> float:
    rels = np.asarray(rels[:k], dtype=float)
    return float(np.sum((2 ** rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(ranked_rels: list[float], k: int = 5) -> float:
    ideal = dcg(sorted(ranked_rels, reverse=True), k)
    return dcg(ranked_rels, k) / ideal if ideal > 0 else 0.0


def evaluate_method(scores: dict[str, float], labels: dict[str, int]) -> dict[str, float]:
    ranked = sorted(scores, key=scores.get, reverse=True)
    rels = [labels.get(name, 0) for name in ranked]
    best = max(labels.values())
    first_best = next(i for i, r in enumerate(rels, 1) if r == best)
    rho = spearmanr([scores[n] for n in ranked], rels).statistic
    return {
        "NDCG@5": ndcg_at_k(rels, 5),
        "MRR": 1.0 / first_best,
        "Top-1 Acc": float(rels[0] == best),
        "Spearman": float(np.nan_to_num(rho)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "sbert", "lsa"])
    args = ap.parse_args()

    resumes = {p.stem: extract_text(p) for p in sorted((DATA / "resumes").iterdir())}
    jobs = {p.stem: extract_text(p) for p in sorted((DATA / "jobs").iterdir())}
    labels_df = pd.read_csv(DATA / "labels.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matcher = ResumeMatcher(backend=args.backend)
    print(f"Semantic backend: {matcher.backend.upper()}\n")

    rows, rankings = [], []
    for job, jd_text in jobs.items():
        labels = dict(labels_df[labels_df.job == job][["resume", "relevance"]].values)
        comps = matcher.component_scores(jd_text, resumes)
        for method, scores in comps.items():
            rows.append({"job": job, "method": method, **evaluate_method(scores, labels)})
        hybrid = comps["Hybrid"]
        for rank, name in enumerate(sorted(hybrid, key=hybrid.get, reverse=True), 1):
            rankings.append({"job": job, "rank": rank, "resume": name,
                             "hybrid_score": hybrid[name], "true_relevance": labels.get(name, 0)})

    per_job = pd.DataFrame(rows)
    summary = per_job.groupby("method", sort=False)[["NDCG@5", "MRR", "Top-1 Acc", "Spearman"]].mean()

    OUT.mkdir(exist_ok=True)
    per_job.round(3).to_csv(OUT / "metrics_per_job.csv", index=False)
    summary.round(3).to_csv(OUT / "metrics_summary.csv")
    pd.DataFrame(rankings).to_csv(OUT / "hybrid_rankings.csv", index=False)

    pd.set_option("display.width", 120)
    print("=== Average over all jobs ===")
    print(summary.round(3).to_string())
    print("\n=== Hybrid model: top 3 per job (true relevance in brackets) ===")
    rk = pd.DataFrame(rankings)
    for job, grp in rk.groupby("job", sort=False):
        top = ", ".join(f"{r.resume} ({r.true_relevance})" for r in grp.head(3).itertuples())
        print(f"{job:20s} -> {top}")
    print(f"\nSaved CSVs to {OUT}/")


if __name__ == "__main__":
    main()
