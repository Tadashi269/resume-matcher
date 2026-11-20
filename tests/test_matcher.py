"""Unit tests. Run with:  python -m pytest -q"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matcher import ResumeMatcher, estimate_years_experience, extract_jd_skills, extract_skills, required_years


def test_skill_boundaries():
    found = extract_skills("Worked with JavaScript, C++ and Node.js on HTML/CSS pages")
    assert {"JavaScript", "C++", "Node.js", "HTML", "CSS"} <= set(found)
    assert "Java" not in found           # must not match inside "JavaScript"


def test_false_positives_filtered():
    found = extract_skills("Spring 2022 internship. I excel at teamwork.")
    assert "Spring Boot" not in found and "Excel" not in found


def test_implied_skills():
    found = extract_skills("Built APIs in Django with PostgreSQL")
    assert found["Python"] == 0 and found["SQL"] == 0     # implied, not written


def test_jd_required_vs_optional_and_alternatives():
    jd = "Requirements:\n- Java and Spring Boot\n- MySQL or PostgreSQL\nNice to have:\n- Kubernetes"
    w = extract_jd_skills(jd)
    assert w["Java"] == 1.0 and w["Kubernetes"] == 0.5
    assert "MySQL or PostgreSQL" in w


def test_alternative_satisfied_by_one():
    score, matched, missing, _ = ResumeMatcher.skill_score({"MySQL or PostgreSQL": 1.0}, {"MySQL": 1})
    assert score == 1.0 and not missing


def test_years_experience():
    text = "EXPERIENCE\nDev, A (Jan 2020 - Dec 2021)\nDev, B (Jan 2021 - Dec 2022)\nEDUCATION\nB.Tech 2014 - 2018\n"
    assert estimate_years_experience(text, today=date(2026, 1, 1)) == 2.9   # overlap merged, education ignored
    assert required_years("We need 3+ years of experience") == 3


def test_ranking_orders_relevant_first():
    m = ResumeMatcher(backend="lsa")
    jd = "Requirements:\n- Build REST APIs in Java with Spring Boot\n- SQL and Docker experience"
    resumes = {
        "backend": "Built REST APIs and microservices in Java using Spring Boot, MySQL and Docker",
        "designer": "Designed mobile app screens in Figma and ran user research interviews",
    }
    ranked = m.rank(jd, resumes)
    assert ranked[0].name == "backend" and ranked[0].score > ranked[1].score
