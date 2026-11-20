"""Scoring models for resume - job description matching.

Three signals are computed for every (JD, resume) pair:

1. TF-IDF cosine similarity  - lexical baseline (bag of words + bigrams)
2. Skill coverage            - weighted share of the JD's skills found in
                               the resume (dictionary-based extraction)
3. Semantic similarity       - Sentence-BERT embeddings, compared at the
                               chunk level: every JD requirement is matched
                               to its most similar resume line, and the
                               similarities are averaged. This catches
                               paraphrases such as "built backend services"
                               vs "develop server-side APIs".

If sentence-transformers or the model cannot be loaded (e.g. offline), the
semantic signal falls back to LSA (TF-IDF + truncated SVD), a classic
count-based semantic model, so the project still runs end to end.

The final score is a weighted combination of the three signals.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .skills import alternatives, extract_jd_skills, extract_skills
from .text_utils import clean_text, estimate_years_experience, required_years, split_into_chunks

DEFAULT_WEIGHTS = {"semantic": 0.5, "skill": 0.35, "tfidf": 0.15}

# Raw similarity ranges used to rescale each signal to roughly 0..1 so they
# can be combined. (lo -> 0, hi -> 1, clipped). Tune these on your own data.
CALIBRATION = {
    "sbert": (0.15, 0.65),
    "lsa": (0.0, 0.60),
    "tfidf": (0.0, 0.40),
}


def _rescale(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class MatchResult:
    name: str
    score: float                       # final hybrid score, 0..100
    semantic: float                    # rescaled 0..1
    skill: float                       # 0..1
    tfidf: float                       # rescaled 0..1
    semantic_raw: float
    tfidf_raw: float
    matched_skills: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    years_experience: float = 0.0
    years_required: int | None = None
    evidence: list[tuple[str, str, float]] = field(default_factory=list)   # best-supported JD lines
    gaps: list[tuple[str, str, float]] = field(default_factory=list)       # least-supported JD lines

    @property
    def verdict(self) -> str:
        if self.score >= 70:
            return "Strong match"
        if self.score >= 50:
            return "Good match"
        if self.score >= 30:
            return "Partial match"
        return "Weak match"


class SBERTEncoder:
    """Thin wrapper around sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # heavy import
        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True,
                                 batch_size=64, show_progress_bar=False)


class ResumeMatcher:
    """Rank resumes against a job description.

    Parameters
    ----------
    backend : "auto" | "sbert" | "lsa"
        Which semantic model to use. "auto" tries Sentence-BERT and falls
        back to LSA if it is unavailable.
    weights : dict with keys semantic / skill / tfidf
    encoder : optional callable(list[str]) -> np.ndarray of L2-normalised
        embeddings. Lets you plug in any embedding model.
    """

    def __init__(self, backend: str = "auto", model_name: str = "all-MiniLM-L6-v2",
                 weights: dict[str, float] | None = None,
                 encoder: Callable[[Sequence[str]], np.ndarray] | None = None):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.encoder = encoder
        self.backend = "sbert" if encoder is not None else backend
        if self.encoder is None and backend in ("auto", "sbert"):
            try:
                self.encoder = SBERTEncoder(model_name)
                self.backend = "sbert"
            except Exception as exc:  # ImportError, network errors, ...
                if backend == "sbert":
                    raise
                warnings.warn(f"Sentence-BERT unavailable ({type(exc).__name__}: {exc}); "
                              "falling back to LSA semantic model.")
                self.backend = "lsa"
        elif self.encoder is None:
            self.backend = "lsa"

    # ------------------------------------------------------------------
    # Individual signals
    # ------------------------------------------------------------------
    @staticmethod
    def tfidf_scores(jd_text: str, resume_texts: Sequence[str]) -> np.ndarray:
        docs = [clean_text(jd_text)] + [clean_text(t) for t in resume_texts]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                              token_pattern=r"(?u)\b[\w+#./-]*\w[\w+#]*\b")
        X = vec.fit_transform(docs)
        return cosine_similarity(X[0], X[1:]).ravel()

    @staticmethod
    def skill_score(jd_weights: dict[str, float], resume_skills: dict[str, int]):
        if not jd_weights:
            return None, [], [], []
        def has(req: str) -> bool:  # any alternative present counts
            return any(alt in resume_skills for alt in alternatives(req))
        matched = [s for s in jd_weights if has(s)]
        missing_req = [s for s, w in jd_weights.items() if not has(s) and w >= 1.0]
        missing_opt = [s for s, w in jd_weights.items() if not has(s) and w < 1.0]
        total = sum(jd_weights.values())
        score = sum(jd_weights[s] for s in matched) / total
        return score, matched, missing_req, missing_opt

    def _embed_chunks(self, jd_chunks: list[str], resume_chunks: list[list[str]]):
        """Return (jd_emb, [resume_emb, ...]) with L2-normalised rows."""
        all_chunks = jd_chunks + [c for chunks in resume_chunks for c in chunks]
        if self.backend == "sbert":
            emb = np.asarray(self.encoder(all_chunks))
        else:  # LSA fitted on the chunks of this ranking job
            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            X = vec.fit_transform(all_chunks)
            k = min(100, X.shape[0] - 1, X.shape[1] - 1)
            if k >= 2:
                emb = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
            else:
                emb = X.toarray()
            emb = normalize(emb)
        jd_emb = emb[: len(jd_chunks)]
        out, i = [], len(jd_chunks)
        for chunks in resume_chunks:
            out.append(emb[i: i + len(chunks)])
            i += len(chunks)
        return jd_emb, out

    def semantic_details(self, jd_text: str, resume_texts: Sequence[str]):
        """For each resume: (mean of best-match similarity per JD chunk,
        list of (jd_chunk, best_resume_chunk, sim))."""
        jd_chunks = split_into_chunks(jd_text) or [clean_text(jd_text)]
        res_chunks = [split_into_chunks(t) or [clean_text(t) or "empty"] for t in resume_texts]
        jd_emb, res_embs = self._embed_chunks(jd_chunks, res_chunks)
        results = []
        for chunks, emb in zip(res_chunks, res_embs):
            sims = jd_emb @ emb.T                      # (n_jd, n_resume)
            best_idx = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            pairs = [(jd_chunks[j], chunks[best_idx[j]], float(best_sim[j])) for j in range(len(jd_chunks))]
            results.append((float(best_sim.mean()), pairs))
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def component_scores(self, jd_text: str, resumes: dict[str, str]) -> dict[str, dict[str, float]]:
        """Raw per-method scores, used by the evaluation script.
        Returns {method: {resume_name: score}}."""
        names = list(resumes)
        texts = [resumes[n] for n in names]
        tfidf = self.tfidf_scores(jd_text, texts)
        jd_w = extract_jd_skills(jd_text)
        sem = self.semantic_details(jd_text, texts)
        results = self.rank(jd_text, resumes, _precomputed=(tfidf, jd_w, sem))
        by_name = {r.name: r for r in results}
        return {
            "TF-IDF": dict(zip(names, map(float, tfidf))),
            "Skill match": {n: by_name[n].skill for n in names},
            f"Semantic ({self.backend.upper()})": {n: by_name[n].semantic_raw for n in names},
            "Hybrid": {n: by_name[n].score for n in names},
        }

    def rank(self, jd_text: str, resumes: dict[str, str], _precomputed=None) -> list[MatchResult]:
        """Score every resume against the JD and return results sorted best-first."""
        names = list(resumes)
        texts = [resumes[n] for n in names]
        if not names:
            return []
        if _precomputed:
            tfidf, jd_w, sem = _precomputed
        else:
            tfidf = self.tfidf_scores(jd_text, texts)
            jd_w = extract_jd_skills(jd_text)
            sem = self.semantic_details(jd_text, texts)
        yrs_req = required_years(jd_text)

        lo, hi = CALIBRATION[self.backend]
        tlo, thi = CALIBRATION["tfidf"]
        results = []
        for name, text, t_raw, (s_raw, pairs) in zip(names, texts, tfidf, sem):
            res_skills = extract_skills(text)
            sk, matched, miss_req, miss_opt = self.skill_score(jd_w, res_skills)
            sem_s = _rescale(s_raw, lo, hi)
            tf_s = _rescale(float(t_raw), tlo, thi)

            w = dict(self.weights)
            if sk is None:  # JD had no recognisable skills -> drop that signal
                w.pop("skill")
                sk = 0.0
            total_w = sum(w.values())
            final = (w.get("semantic", 0) * sem_s + w.get("skill", 0) * sk + w.get("tfidf", 0) * tf_s) / total_w

            sorted_pairs = sorted(pairs, key=lambda p: p[2], reverse=True)
            jd_all = {a for req in jd_w for a in alternatives(req)}
            extra = sorted(s for s, c in res_skills.items() if c > 0 and s not in jd_all)
            results.append(MatchResult(
                name=name, score=round(100 * final, 1), semantic=sem_s, skill=sk, tfidf=tf_s,
                semantic_raw=s_raw, tfidf_raw=float(t_raw),
                matched_skills=matched, missing_required=miss_req, missing_optional=miss_opt,
                extra_skills=extra,
                years_experience=estimate_years_experience(text), years_required=yrs_req,
                evidence=sorted_pairs[:3], gaps=sorted_pairs[::-1][:3],
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
