"""Streamlit demo: rank resumes against a job description.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from matcher import ResumeMatcher, extract_text

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Resume-JD Matcher", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading language model...")
def get_matcher(backend: str) -> ResumeMatcher:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ResumeMatcher(backend=backend)


def pretty_name(stem: str) -> str:
    parts = stem.split("_")
    if parts[0][:1] == "r" and parts[0][1:].isdigit():
        parts = parts[1:]
    return " ".join(p.capitalize() for p in parts)


# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    backend_label = st.radio("Semantic model", ["Sentence-BERT (best)", "LSA (offline, fast)"])
    backend = "auto" if backend_label.startswith("Sentence") else "lsa"
    st.subheader("Score weights")
    w_sem = st.slider("Semantic similarity", 0.0, 1.0, 0.50, 0.05)
    w_skill = st.slider("Skill coverage", 0.0, 1.0, 0.35, 0.05)
    w_tfidf = st.slider("Keyword overlap (TF-IDF)", 0.0, 1.0, 0.15, 0.05)
    if w_sem + w_skill + w_tfidf == 0:
        st.error("At least one weight must be above 0.")
        st.stop()
    st.caption("Weights are normalised to sum to 1.")

matcher = get_matcher(backend)
matcher.weights = {"semantic": w_sem, "skill": w_skill, "tfidf": w_tfidf}
if backend == "auto" and matcher.backend != "sbert":
    st.sidebar.warning("Sentence-BERT could not be loaded, so the LSA model is being used. "
                       "Install `sentence-transformers` and make sure you're online the first time you run it.")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.title("📄 Resume - Job Description Matcher")
st.write("Rank candidates for a job using NLP: skill extraction, TF-IDF and Sentence-BERT semantic similarity.")

col_jd, col_res = st.columns(2)

with col_jd:
    st.subheader("1. Job description")
    sample_jobs = {pretty_name(p.stem): p for p in sorted((DATA / "jobs").glob("*.txt"))}
    jd_source = st.selectbox("Start from a sample or paste your own",
                             ["Paste my own"] + list(sample_jobs))
    default_jd = sample_jobs[jd_source].read_text() if jd_source in sample_jobs else ""
    jd_text = st.text_area("Job description text", value=default_jd, height=320,
                           key=f"jd_{jd_source}")

with col_res:
    st.subheader("2. Resumes")
    uploads = st.file_uploader("Upload resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                               accept_multiple_files=True)
    use_samples = st.checkbox("Also include the 12 sample resumes", value=not uploads)
    resumes: dict[str, str] = {}
    for up in uploads or []:
        try:
            text = extract_text(up, up.name)
            if text.strip():
                resumes[Path(up.name).stem] = text
            else:
                st.warning(f"No text found in {up.name} (scanned PDF?). Skipped.")
        except Exception as exc:
            st.warning(f"Could not read {up.name}: {exc}")
    if use_samples:
        for p in sorted((DATA / "resumes").glob("*.txt")):
            resumes[pretty_name(p.stem)] = p.read_text()
    st.caption(f"{len(resumes)} resume(s) ready")

run = st.button("Rank candidates", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if run:
    if not jd_text.strip():
        st.error("Please enter a job description.")
        st.stop()
    if not resumes:
        st.error("Please upload at least one resume or include the samples.")
        st.stop()

    with st.spinner("Scoring resumes..."):
        results = matcher.rank(jd_text, resumes)
    st.session_state["results"] = results
    st.session_state["backend_used"] = matcher.backend

results = st.session_state.get("results")
if results:
    st.divider()
    st.subheader(f"3. Ranking ({st.session_state['backend_used'].upper()} semantic model)")

    table = pd.DataFrame([{
        "Rank": i, "Candidate": r.name, "Score": r.score, "Verdict": r.verdict,
        "Semantic": round(100 * r.semantic), "Skills": round(100 * r.skill),
        "Keywords": round(100 * r.tfidf), "Exp. (yrs)": r.years_experience,
        "Missing required skills": ", ".join(r.missing_required) or "-",
    } for i, r in enumerate(results, 1)])

    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Score": st.column_config.ProgressColumn(
                     "Score", min_value=0, max_value=100, format="%.1f")})
    st.download_button("Download results as CSV", table.to_csv(index=False),
                       "ranking.csv", "text/csv")

    st.subheader("4. Candidate details")
    for i, r in enumerate(results, 1):
        with st.expander(f"#{i}  {r.name} - {r.score:.1f}  ({r.verdict})", expanded=(i == 1)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall", f"{r.score:.0f}/100")
            c2.metric("Semantic", f"{100 * r.semantic:.0f}")
            c3.metric("Skill coverage", f"{100 * r.skill:.0f}%")
            if r.years_required:
                delta = round(r.years_experience - r.years_required, 1)
                c4.metric("Experience", f"{r.years_experience} yrs", f"{delta:+} vs {r.years_required} required")
            else:
                c4.metric("Experience", f"{r.years_experience} yrs")

            s1, s2, s3 = st.columns(3)
            s1.markdown("**✅ Matched skills**")
            s1.write(", ".join(r.matched_skills) or "-")
            s2.markdown("**❌ Missing (required)**")
            s2.write(", ".join(r.missing_required) or "None")
            s2.markdown("**➖ Missing (nice to have)**")
            s2.write(", ".join(r.missing_optional) or "None")
            s3.markdown("**➕ Other skills on resume**")
            s3.write(", ".join(r.extra_skills) or "-")

            st.markdown("**Best evidence** (JD requirement → most similar resume line)")
            for jd_line, res_line, sim in r.evidence:
                st.markdown(f"- *{jd_line}*  \n  → {res_line}  `sim={sim:.2f}`")
            st.markdown("**Weakest coverage** (requirements with little support in the resume)")
            for jd_line, res_line, sim in r.gaps:
                st.markdown(f"- *{jd_line}*  `best sim={sim:.2f}`")
