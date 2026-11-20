"""Skill extraction using a curated skill taxonomy (dictionary-based NER).

Each canonical skill has a list of surface forms (aliases). Matching uses
custom word boundaries so that tokens like "C++", "C#", ".NET" and
"Node.js" work, and so that "Java" does not match inside "JavaScript".

Some skills imply others (e.g. "Django" implies "Python", "PostgreSQL"
implies "SQL"); these are added as implied skills.

The JD parser also detects "nice to have / preferred" sections so that
optional skills get a lower weight than required ones.
"""

from __future__ import annotations

import re
from functools import lru_cache

# canonical skill -> (category, [aliases])
SKILLS: dict[str, tuple[str, list[str]]] = {
    # --- Programming languages
    "Python": ("Language", ["python", "python3"]),
    "Java": ("Language", ["java", "core java", "java 8", "java 11", "java 17"]),
    "JavaScript": ("Language", ["javascript", "js", "es6", "ecmascript"]),
    "TypeScript": ("Language", ["typescript", "ts"]),
    "C++": ("Language", ["c++", "cpp"]),
    "C#": ("Language", ["c#", "csharp"]),
    "Go": ("Language", ["golang"]),
    "Kotlin": ("Language", ["kotlin"]),
    "PHP": ("Language", ["php"]),
    "Bash": ("Language", ["bash", "shell scripting", "shell script", "powershell"]),
    "SQL": ("Database", ["sql", "t-sql", "pl/sql", "sql queries"]),
    # --- Web / frontend
    "HTML": ("Frontend", ["html", "html5"]),
    "CSS": ("Frontend", ["css", "css3", "sass", "scss", "tailwind", "tailwind css", "bootstrap"]),
    "React": ("Frontend", ["react", "react.js", "reactjs", "react js"]),
    "Redux": ("Frontend", ["redux", "redux toolkit"]),
    "Angular": ("Frontend", ["angular", "angularjs"]),
    "Vue": ("Frontend", ["vue", "vue.js", "vuejs"]),
    "Next.js": ("Frontend", ["next.js", "nextjs"]),
    "Responsive Design": ("Frontend", ["responsive design", "responsive web design", "mobile-first"]),
    "Figma": ("Design", ["figma"]),
    "UI/UX Design": ("Design", ["ui/ux", "ux design", "ui design", "user experience", "wireframing", "prototyping", "user research"]),
    "Jest": ("Testing", ["jest", "react testing library"]),
    # --- Backend
    "Node.js": ("Backend", ["node.js", "nodejs", "node js", "express", "express.js"]),
    "Django": ("Backend", ["django", "django rest framework", "drf"]),
    "Flask": ("Backend", ["flask"]),
    "FastAPI": ("Backend", ["fastapi"]),
    "Spring Boot": ("Backend", ["spring boot", "spring", "spring mvc", "spring framework"]),
    "Hibernate": ("Backend", ["hibernate", "jpa"]),
    "REST APIs": ("Backend", ["rest api", "rest apis", "restful", "restful apis", "restful services", "rest services", "web services"]),
    "GraphQL": ("Backend", ["graphql"]),
    "Microservices": ("Backend", ["microservices", "microservice architecture", "micro-services"]),
    "Kafka": ("Backend", ["kafka", "apache kafka"]),
    "Redis": ("Database", ["redis"]),
    "MySQL": ("Database", ["mysql"]),
    "PostgreSQL": ("Database", ["postgresql", "postgres"]),
    "MongoDB": ("Database", ["mongodb", "mongo"]),
    "Oracle": ("Database", ["oracle db", "oracle database"]),
    # --- Cloud / DevOps
    "AWS": ("Cloud", ["aws", "amazon web services", "ec2", "s3", "aws lambda"]),
    "Azure": ("Cloud", ["azure", "microsoft azure"]),
    "GCP": ("Cloud", ["gcp", "google cloud", "google cloud platform"]),
    "Docker": ("DevOps", ["docker", "containerization", "containers"]),
    "Kubernetes": ("DevOps", ["kubernetes", "k8s", "eks", "aks"]),
    "CI/CD": ("DevOps", ["ci/cd", "continuous integration", "jenkins", "github actions", "gitlab ci"]),
    "Terraform": ("DevOps", ["terraform", "infrastructure as code"]),
    "Linux": ("DevOps", ["linux", "unix", "ubuntu", "centos"]),
    "Git": ("Tools", ["git", "github", "gitlab", "version control"]),
    "Ansible": ("DevOps", ["ansible"]),
    "Prometheus/Grafana": ("DevOps", ["prometheus", "grafana"]),
    # --- Data / ML
    "Machine Learning": ("Data/ML", ["machine learning", "ml models", "ml model", "predictive modeling", "predictive modelling"]),
    "Deep Learning": ("Data/ML", ["deep learning", "neural networks", "neural network", "cnn", "rnn", "lstm"]),
    "NLP": ("Data/ML", ["nlp", "natural language processing", "text classification", "text mining"]),
    "Computer Vision": ("Data/ML", ["computer vision", "image classification", "object detection", "opencv"]),
    "Transformers": ("Data/ML", ["transformers", "bert", "hugging face", "huggingface", "llm", "llms", "gpt"]),
    "scikit-learn": ("Data/ML", ["scikit-learn", "sklearn", "scikit learn"]),
    "TensorFlow": ("Data/ML", ["tensorflow", "keras"]),
    "PyTorch": ("Data/ML", ["pytorch", "torch"]),
    "Pandas": ("Data/ML", ["pandas"]),
    "NumPy": ("Data/ML", ["numpy"]),
    "Statistics": ("Data/ML", ["statistics", "statistical analysis", "hypothesis testing", "a/b testing", "regression analysis", "probability"]),
    "Data Visualization": ("Data/ML", ["data visualization", "matplotlib", "seaborn", "plotly", "dashboards", "dashboard"]),
    "Tableau": ("Data/ML", ["tableau"]),
    "Power BI": ("Data/ML", ["power bi", "powerbi"]),
    "Excel": ("Data/ML", ["excel", "ms excel", "advanced excel", "vlookup", "pivot tables"]),
    "Spark": ("Data/ML", ["spark", "pyspark", "apache spark"]),
    "Feature Engineering": ("Data/ML", ["feature engineering", "feature selection"]),
    "MLOps": ("Data/ML", ["mlops", "mlflow", "model deployment"]),
    "ETL": ("Data/ML", ["etl", "data pipelines", "data pipeline", "airflow"]),
    # --- Security / networking
    "Network Security": ("Security", ["network security", "firewall", "firewalls", "ids/ips", "ids", "ips", "vpn"]),
    "SIEM": ("Security", ["siem", "splunk", "qradar", "elk", "elastic siem", "sentinel"]),
    "Incident Response": ("Security", ["incident response", "incident handling", "triage", "soc", "security operations"]),
    "Vulnerability Assessment": ("Security", ["vulnerability assessment", "vulnerability scanning", "nessus", "openvas", "vapt"]),
    "Penetration Testing": ("Security", ["penetration testing", "pentesting", "ethical hacking", "burp suite", "metasploit", "kali linux"]),
    "Wireshark": ("Security", ["wireshark", "packet analysis", "tcpdump"]),
    "Threat Intelligence": ("Security", ["threat intelligence", "threat hunting", "mitre att&ck", "mitre attack", "ioc", "iocs"]),
    "Malware Analysis": ("Security", ["malware analysis", "reverse engineering"]),
    "TCP/IP": ("Networking", ["tcp/ip", "networking", "dns", "dhcp", "routing", "switching", "osi model", "subnetting"]),
    "Cisco": ("Networking", ["cisco", "ccna", "ccnp"]),
    "Security Compliance": ("Security", ["iso 27001", "nist", "pci dss", "gdpr", "compliance"]),
    "Security+": ("Security", ["security+", "comptia security+", "ceh", "cissp"]),
    # --- General / soft
    "Agile": ("Process", ["agile", "scrum", "jira", "kanban"]),
    "Unit Testing": ("Testing", ["unit testing", "unit tests", "junit", "pytest", "test-driven development", "tdd"]),
    "Communication": ("Soft skill", ["communication skills", "communication", "stakeholder management"]),
    "Leadership": ("Soft skill", ["leadership", "team lead", "mentoring", "mentored"]),
    "Problem Solving": ("Soft skill", ["problem solving", "problem-solving", "analytical skills"]),
    # --- Mechanical (used by an off-domain sample resume)
    "AutoCAD": ("Mechanical", ["autocad", "solidworks", "catia"]),
}

# Skill A implies skill B
IMPLIES: dict[str, list[str]] = {
    "Django": ["Python"], "Flask": ["Python"], "FastAPI": ["Python"],
    "Pandas": ["Python"], "NumPy": ["Python"], "scikit-learn": ["Python", "Machine Learning"],
    "PyTorch": ["Deep Learning"], "TensorFlow": ["Deep Learning"],
    "Spring Boot": ["Java"], "Hibernate": ["Java"],
    "React": ["JavaScript"], "Angular": ["TypeScript"], "Vue": ["JavaScript"],
    "Next.js": ["React"], "Redux": ["React"], "Node.js": ["JavaScript"],
    "MySQL": ["SQL"], "PostgreSQL": ["SQL"], "Oracle": ["SQL"],
    "Kubernetes": ["Docker"], "Transformers": ["NLP", "Deep Learning"],
}

# Headers that start an "optional" part of a JD
_OPTIONAL_HEADER = re.compile(
    r"(nice[\s-]to[\s-]have|good[\s-]to[\s-]have|preferred|bonus|desirable|plus points?|optional|is a plus)",
    re.IGNORECASE,
)
# Headers that start a "required" part of a JD
_REQUIRED_HEADER = re.compile(
    r"(requirements|required|must[\s-]have|qualifications|what you.ll need|responsibilities|skills)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _compiled_patterns() -> list[tuple[str, re.Pattern]]:
    pats = []
    for canonical, (_, aliases) in SKILLS.items():
        alts = sorted({a.lower() for a in aliases} | {canonical.lower()}, key=len, reverse=True)
        body = "|".join(re.escape(a) for a in alts)
        # Custom boundaries: no letter/digit/+/# directly before or after.
        # A trailing '.' is allowed after (end of sentence) but not '.js' etc.
        pattern = re.compile(rf"(?<![a-z0-9+#])(?:{body})(?![a-z0-9+#]|\.[a-z])", re.IGNORECASE)
        pats.append((canonical, pattern))
    return pats


def _ambiguous_hit(canonical: str, text: str, m: re.Match) -> bool:
    """Filter a few known false positives for short aliases."""
    token = m.group(0).lower()
    if token in {"ts", "js", "ips", "ids", "ioc", "s3", "elk", "soc"}:
        # require them to be written in upper case in the source to count
        return m.group(0) != m.group(0).upper()
    if token == "spring" and canonical == "Spring Boot":
        # "spring 2022" / "spring semester" are dates, not the framework
        after = text[m.end(): m.end() + 12].lower()
        return bool(re.match(r"\s*(?:19|20)\d{2}|\s*semester|\s*term|\s*break", after))
    if token == "excel":
        after = text[m.end(): m.end() + 4].lower()
        return after.startswith((" at", " in"))  # the verb "excel at/in"
    if token == "containers" and canonical == "Docker":
        before = text[max(0, m.start() - 20): m.start()].lower()
        return "shipping" in before or "cargo" in before
    return False


def find_skill_spans(text: str) -> list[tuple[str, int, int]]:
    """Return [(canonical_skill, start, end)] for every mention, sorted by position."""
    spans = []
    for canonical, pattern in _compiled_patterns():
        for m in pattern.finditer(text):
            if not _ambiguous_hit(canonical, text, m):
                spans.append((canonical, m.start(), m.end()))
    return sorted(spans, key=lambda s: (s[1], -s[2]))


def extract_skills(text: str, add_implied: bool = True) -> dict[str, int]:
    """Return {canonical_skill: mention_count} found in `text`."""
    found: dict[str, int] = {}
    for canonical, _, _ in find_skill_spans(text):
        found[canonical] = found.get(canonical, 0) + 1
    if add_implied:
        for skill in list(found):
            for implied in IMPLIES.get(skill, []):
                found.setdefault(implied, 0)  # count 0 = implied, not written
    return found


ALT_SEP = " or "
_LIST_GAP = re.compile(r"^\s*(?:,|/|or|,\s*or|and|,\s*and|&)?\s*$", re.IGNORECASE)


def _line_requirements(line: str) -> list[str]:
    """Group skills in one JD line into requirements.

    Skills listed with "or" / "/" ("MySQL or PostgreSQL", "PyTorch or
    TensorFlow") become one requirement with alternatives, labelled
    "MySQL or PostgreSQL". Other skills are separate requirements.
    """
    # 1. Collapse adjacent mentions of the same skill ("Hibernate/JPA").
    items: list[list] = []           # [canonical, start, end, gap_before]
    for canonical, start, end in find_skill_spans(line):
        if items and start < items[-1][2]:
            continue                 # overlapping match, keep the first
        gap = line[items[-1][2]:start] if items else None
        if items and items[-1][0] == canonical and _LIST_GAP.match(gap):
            items[-1][2] = end
            continue
        items.append([canonical, start, end, gap])

    # 2. Split into runs: a new run starts when the gap is not a list
    #    separator, or is "and"/"&" (which means both are needed).
    runs: list[tuple[list[str], bool]] = []   # (skills, is_or_group)
    for canonical, _, _, gap in items:
        g = (gap or "").strip().lower()
        joins = gap is not None and _LIST_GAP.match(gap) and not g.endswith(("and", "&"))
        if runs and joins:
            skills, is_or = runs[-1]
            runs[-1] = (skills + [canonical], is_or or g.endswith("or") or g == "/")
        else:
            runs.append(([canonical], False))

    labels = []
    for skills, is_or in runs:
        unique = list(dict.fromkeys(skills))
        if is_or and len(unique) > 1:
            labels.append(ALT_SEP.join(unique))
        else:
            labels.extend(unique)
    return labels


def alternatives(requirement: str) -> list[str]:
    """'MySQL or PostgreSQL' -> ['MySQL', 'PostgreSQL']."""
    return requirement.split(ALT_SEP)


def extract_jd_skills(jd_text: str) -> dict[str, float]:
    """Return {requirement: weight} for a JD.

    A requirement is a skill ("Docker") or a group of alternatives
    ("MySQL or PostgreSQL") that is satisfied by any one of them.
    Required skills weigh 1.0 and optional ("nice to have") ones 0.5.
    A requirement mentioned in both places keeps the higher weight.
    """
    weights: dict[str, float] = {}
    mode = 1.0
    for line in jd_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        is_header = len(stripped.split()) <= 6 and (stripped.endswith(":") or stripped.isupper())
        if _OPTIONAL_HEADER.search(stripped) and (is_header or stripped.lower().startswith(("nice", "good to", "preferred", "bonus"))):
            mode = 0.5
        elif is_header and _REQUIRED_HEADER.search(stripped):
            mode = 1.0
        line_weight = 0.5 if _OPTIONAL_HEADER.search(stripped) else mode
        for req in _line_requirements(stripped):
            weights[req] = max(weights.get(req, 0.0), line_weight)
    # A single skill that is also required on its own makes the
    # alternative group redundant ("Java" + "Java or Kotlin" -> keep "Java").
    for req in [r for r in weights if ALT_SEP in r]:
        if any(alt in weights and weights[alt] >= weights[req] for alt in alternatives(req)):
            del weights[req]
    # Soft skills are rarely discriminative between candidates -> down-weight
    for req in list(weights):
        if all(SKILLS[s][0] == "Soft skill" for s in alternatives(req)):
            weights[req] = min(weights[req], 0.25)
    return weights


def skill_category(skill: str) -> str:
    return SKILLS.get(skill, ("Other", []))[0]
