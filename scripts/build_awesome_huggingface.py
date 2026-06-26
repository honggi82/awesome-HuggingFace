import csv
import html
import json
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

import markdown
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
PAPER_DIR = ROOT / "paper"
CACHE_DIR = DATA_DIR / "cache"

OWNER = "honggi82"
REPO_NAME = "awesome-HuggingFace"
START_MONTH = "2023-05"
END_MONTH = "2026-06"
PERIOD_TEXT = f"{START_MONTH} through {END_MONTH}"
PERIOD_STEM = "2023_05_2026_06"

API_URL = "https://huggingface.co/api/daily_papers"
USER_AGENT = "awesome-HuggingFace-builder/1.0"
REQUEST_DELAY = 0.15

PAPERS_JSON = f"papers_{PERIOD_STEM}.json"
PAPERS_CSV = f"papers_{PERIOD_STEM}.csv"
TAXONOMY_CSV = f"papers_taxonomy_{PERIOD_STEM}.csv"
MONTHLY_INDEX_JSON = f"hf_daily_papers_monthly_index_{PERIOD_STEM}.json"
PERIOD_ANALYSIS_JSON = f"period_analysis_{PERIOD_STEM}.json"
LINK_AUDIT_JSON = f"link_audit_{PERIOD_STEM}.json"
PROVENANCE_JSON = "paper_curation_skill2_provenance.json"


CATEGORY_RULES = [
    {
        "name": "Foundation Models and Large Language Models",
        "color": "#2563eb",
        "keywords": [
            "large language model",
            "language model",
            "llm",
            "foundation model",
            "gpt",
            "llama",
            "qwen",
            "bert",
            "mistral",
            "pretrain",
            "instruction tuning",
            "alignment",
            "prompt",
            "rag",
            "retrieval-augmented",
            "token",
        ],
        "trends": [
            "HF Daily Papers in this area concentrate on open model families, instruction tuning, retrieval, alignment, long context, and adaptation.",
            "The strongest signals often combine public weights or code with practical training, evaluation, and deployment recipes.",
            "Recent months show increasingly system-oriented LLM work: adapters, agents, benchmarks, tool use, and efficient serving.",
        ],
        "limitations": [
            "Community attention does not prove model quality or safety.",
            "Model capability claims depend heavily on benchmark design, data leakage controls, and deployment context.",
            "Metadata cannot replace reading the full paper and release artifacts.",
        ],
    },
    {
        "name": "Agents, Tool Use, and Autonomous Workflows",
        "color": "#dc2626",
        "keywords": [
            "agent",
            "agents",
            "tool use",
            "tool-use",
            "autonomous",
            "workflow",
            "planning",
            "reasoning",
            "web automation",
            "computer use",
            "multi-agent",
            "benchmark for agents",
        ],
        "trends": [
            "Agent papers increasingly evaluate long-horizon workflows rather than single-turn benchmark answers.",
            "Tool-use, memory, planning, and verifiable task completion are common design axes.",
            "HF engagement is strongest when papers pair agent methods with public harnesses, benchmark suites, or usable repositories.",
        ],
        "limitations": [
            "Agent benchmarks can overstate reliability when task distributions are narrow.",
            "Long-horizon systems need stronger evidence on recovery, safety, and monitoring.",
            "Repository availability does not guarantee reproducibility.",
        ],
    },
    {
        "name": "Vision, Multimodal, and Video Understanding",
        "color": "#0891b2",
        "keywords": [
            "vision",
            "image",
            "video",
            "visual",
            "multimodal",
            "vision-language",
            "vlm",
            "clip",
            "segmentation",
            "object detection",
            "ocr",
            "document understanding",
            "point cloud",
        ],
        "trends": [
            "Vision-language models, video understanding, document intelligence, and segmentation-oriented foundation models dominate the visible HF stream.",
            "The field is moving from isolated perception tasks toward integrated multimodal assistants and retrieval workflows.",
            "Public demos, thumbnails, and repositories make visual papers especially discoverable on HF.",
        ],
        "limitations": [
            "Visual benchmark gains may not transfer to rare classes, low-resource domains, or messy deployment inputs.",
            "Multimodal models can inherit web-scale biases and hallucinate grounded details.",
            "Metadata summaries rarely capture dataset composition or annotation quality.",
        ],
    },
    {
        "name": "Generative Media, Diffusion, and World Models",
        "color": "#a855f7",
        "keywords": [
            "diffusion",
            "generative",
            "text-to-image",
            "image generation",
            "video generation",
            "world model",
            "3d",
            "gaussian splatting",
            "gan",
            "vae",
            "synthesis",
            "editing",
            "avatar",
            "music generation",
        ],
        "trends": [
            "Generative work spans diffusion, controllable editing, 3D scenes, video, audio, and world-model simulation.",
            "HF attention often follows papers that ship demos, code, model weights, or striking visual examples.",
            "The category increasingly connects generation quality with controllability, evaluation, and downstream embodied use.",
        ],
        "limitations": [
            "Aesthetic or demo quality is not the same as factuality, controllability, or safe deployment.",
            "Copyright, data provenance, and misuse risks are frequently under-specified in metadata.",
            "Evaluation remains difficult across cultures, modalities, and creative tasks.",
        ],
    },
    {
        "name": "Efficient Training, Inference, and AI Systems",
        "color": "#0f766e",
        "keywords": [
            "efficient",
            "inference",
            "serving",
            "quantization",
            "lora",
            "peft",
            "adapter",
            "training",
            "fine-tuning",
            "finetuning",
            "distillation",
            "compression",
            "memory",
            "gpu",
            "accelerat",
            "kernel",
            "throughput",
        ],
        "trends": [
            "Systems papers focus on making foundation models cheaper to train, adapt, serve, compress, and personalize.",
            "Quantization, adapters, memory management, speculative decoding, and inference kernels recur across months.",
            "GitHub-linked releases are especially important because implementation details often carry the contribution.",
        ],
        "limitations": [
            "Reported speedups can depend on hardware, batch size, compiler stack, and hidden engineering assumptions.",
            "Efficiency can trade off with robustness, calibration, or multilingual coverage.",
            "Metadata does not expose enough detail to audit all benchmark settings.",
        ],
    },
    {
        "name": "Data, Evaluation, and Benchmarks",
        "color": "#f59e0b",
        "keywords": [
            "dataset",
            "data",
            "benchmark",
            "evaluation",
            "eval",
            "leaderboard",
            "test set",
            "annotation",
            "synthetic data",
            "corpus",
            "survey",
            "taxonomy",
            "metric",
        ],
        "trends": [
            "Benchmark and dataset papers anchor the HF stream by defining what new models are asked to do.",
            "Recent work emphasizes long-horizon tasks, multimodal evaluation, domain-specific datasets, and benchmark contamination controls.",
            "Community traction is strongest when data and evaluation code are public and reusable.",
        ],
        "limitations": [
            "Benchmarks can saturate, leak into training data, or miss real-world failure modes.",
            "Dataset representativeness and annotation reliability require full-document inspection.",
            "A high upvote count may reflect usefulness rather than scientific completeness.",
        ],
    },
    {
        "name": "Responsible, Safe, and Interpretable AI",
        "color": "#be123c",
        "keywords": [
            "safety",
            "safe",
            "alignment",
            "robust",
            "robustness",
            "fairness",
            "bias",
            "privacy",
            "security",
            "jailbreak",
            "red team",
            "interpretability",
            "interpretable",
            "explainable",
            "uncertainty",
            "watermark",
            "toxicity",
        ],
        "trends": [
            "Responsible AI papers move between interpretability, jailbreak resistance, data governance, privacy, and evaluation for deployed models.",
            "The HF corpus makes safety work visible when it includes reproducible attacks, datasets, or inspection tools.",
            "Interpretability and alignment topics increasingly overlap with model scaling and agentic behavior.",
        ],
        "limitations": [
            "Safety claims need adversarial and real-world validation beyond benchmark results.",
            "Explanations can be plausible without being faithful to internal mechanisms.",
            "Metadata cannot capture all threat models or deployment constraints.",
        ],
    },
    {
        "name": "Robotics, Embodied AI, and Control",
        "color": "#7c3aed",
        "keywords": [
            "robot",
            "robotics",
            "embodied",
            "control",
            "navigation",
            "manipulation",
            "policy",
            "reinforcement learning",
            "rl",
            "imitation",
            "sim-to-real",
            "autonomous driving",
            "uav",
        ],
        "trends": [
            "Embodied AI work connects foundation models with simulation, navigation, manipulation, and policy learning.",
            "World models, video-language grounding, and dataset-driven robotics benchmarks are increasingly interdependent.",
            "Public code and simulators help readers move from paper claims to reproducible experiments.",
        ],
        "limitations": [
            "Simulated results can fail under physical dynamics, hardware limits, and safety constraints.",
            "Real-world evaluation is expensive and often narrower than benchmark framing.",
            "HF metadata rarely captures all robot platforms or environment assumptions.",
        ],
    },
    {
        "name": "AI for Science, Medicine, and Engineering",
        "color": "#16a34a",
        "keywords": [
            "protein",
            "molecule",
            "drug",
            "biology",
            "biomedical",
            "medical",
            "clinical",
            "health",
            "science",
            "chemistry",
            "physics",
            "materials",
            "climate",
            "earth",
            "engineering",
            "genomic",
        ],
        "trends": [
            "AI-for-science papers apply foundation and generative methods to biology, chemistry, medicine, climate, materials, and engineering design.",
            "Visible HF papers often combine domain data with open models, code, or benchmark resources.",
            "Recent work is moving toward specialized scientific agents and editable scientific artifacts.",
        ],
        "limitations": [
            "Domain claims require validation under expert protocols and external datasets.",
            "Biomedical and engineering deployment introduces safety, regulation, and reproducibility constraints.",
            "Metadata summaries cannot substitute for domain expert review.",
        ],
    },
    {
        "name": "Speech, Audio, NLP, and Code Applications",
        "color": "#4f46e5",
        "keywords": [
            "speech",
            "audio",
            "voice",
            "music",
            "translation",
            "summarization",
            "dialogue",
            "question answering",
            "retrieval",
            "information retrieval",
            "code",
            "programming",
            "software",
            "nlp",
            "text",
        ],
        "trends": [
            "Application papers translate core model advances into speech, translation, retrieval, dialogue, code, and document workflows.",
            "HF traction often follows practical demos, evaluation suites, or repositories that let users reproduce a workflow.",
            "Code and retrieval papers increasingly overlap with agentic and LLM-system categories.",
        ],
        "limitations": [
            "Task-specific benchmarks can hide brittle behavior outside their domain.",
            "Language and locale coverage may be uneven.",
            "Repository links are helpful but do not guarantee maintained implementations.",
        ],
    },
    {
        "name": "Graph, Recommendation, and Structured Learning",
        "color": "#64748b",
        "keywords": [
            "graph",
            "knowledge graph",
            "recommendation",
            "recommender",
            "tabular",
            "structured",
            "time series",
            "forecast",
            "optimization",
            "bayesian",
            "causal",
        ],
        "trends": [
            "Structured learning papers cover graph reasoning, recommenders, tabular models, time series, causality, and optimization.",
            "These papers are less visually dominant but often provide reusable methods for domain systems.",
            "The category bridges core ML with applied workflows that do not fit pure language or vision buckets.",
        ],
        "limitations": [
            "Benchmark datasets can be narrow and sensitive to preprocessing.",
            "Graph and recommender evaluation may not reflect real deployment feedback loops.",
            "Metadata can underspecify assumptions about structure and leakage.",
        ],
    },
    {
        "name": "General Machine Learning and Optimization",
        "color": "#334155",
        "keywords": [
            "machine learning",
            "deep learning",
            "neural network",
            "optimization",
            "classification",
            "regression",
            "clustering",
            "bayesian",
            "gradient",
            "loss",
            "architecture",
            "learning",
        ],
        "trends": [
            "General ML papers collect methods, surveys, and cross-cutting improvements that do not sit cleanly in one application area.",
            "This category is useful as a catch-all map of methods that later diffuse into LLM, vision, robotics, and science workflows.",
            "Citation and HF engagement signals should be read as visibility signals, not final judgments.",
        ],
        "limitations": [
            "Broad method categories can obscure task-specific constraints.",
            "Metadata-driven taxonomy may under-classify specialized contributions.",
            "Full-paper reading is required for methodological rigor.",
        ],
    },
]

CATEGORY_BY_NAME = {item["name"]: item for item in CATEGORY_RULES}

KEYWORD_CONVENTION = [
    ("foundation-models", "Foundation models, LLMs, scaling, prompting, alignment, or retrieval-augmented generation.", "#2563eb", ["foundation model", "large language model", "llm", "gpt", "llama", "qwen", "rag", "prompt"]),
    ("agents", "Agentic systems, tool use, planning, memory, autonomous workflows, or long-horizon task execution.", "#dc2626", ["agent", "tool use", "planning", "workflow", "autonomous"]),
    ("vision", "Image, video, segmentation, OCR, visual recognition, and vision-language understanding.", "#0891b2", ["vision", "image", "video", "visual", "segmentation", "ocr"]),
    ("multimodal", "Cross-modal representation learning and models that connect text, image, audio, video, or 3D data.", "#0e7490", ["multimodal", "vision-language", "vlm", "audio-language", "cross-modal"]),
    ("generative-media", "Diffusion, image/video/audio generation, editing, 3D generation, GANs, and world models.", "#a855f7", ["diffusion", "generative", "text-to-image", "video generation", "world model", "3d", "gan"]),
    ("efficient-ai", "Quantization, LoRA/PEFT, distillation, serving, kernels, compression, memory, and training efficiency.", "#0f766e", ["efficient", "quantization", "lora", "peft", "inference", "serving", "distillation", "kernel"]),
    ("datasets-benchmarks", "Datasets, evaluations, benchmarks, metrics, surveys, leaderboards, and annotation workflows.", "#f59e0b", ["dataset", "benchmark", "evaluation", "eval", "metric", "survey", "leaderboard"]),
    ("trustworthy-ai", "Safety, interpretability, privacy, robustness, fairness, security, jailbreaks, and responsible AI.", "#be123c", ["safety", "interpretability", "privacy", "robust", "fairness", "jailbreak", "security", "bias"]),
    ("robotics", "Embodied AI, robotics, control, manipulation, navigation, policies, autonomous driving, and UAVs.", "#7c3aed", ["robot", "robotics", "embodied", "control", "navigation", "manipulation", "uav"]),
    ("ai4science", "AI for science, healthcare, biology, chemistry, medicine, climate, materials, and engineering.", "#16a34a", ["protein", "molecule", "biology", "medical", "health", "chemistry", "climate", "earth", "science"]),
    ("audio-speech", "Speech, audio, voice, music, translation, and spoken-language interfaces.", "#db2777", ["speech", "audio", "voice", "music", "translation"]),
    ("code-ai", "Code models, program synthesis, software engineering, repository understanding, and developer workflows.", "#475569", ["code", "programming", "software", "repository", "debug"]),
]
KEYWORD_COLORS = {name: color for name, _, color, _ in KEYWORD_CONVENTION}


def month_range(start, end):
    year, month = [int(part) for part in start.split("-")]
    end_year, end_month = [int(part) for part in end.split("-")]
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            month = 1
            year += 1


MONTHS = list(month_range(START_MONTH, END_MONTH))


def ensure_dirs():
    for path in (
        DATA_DIR,
        DATA_DIR / "monthly",
        DOCS_DIR,
        DOCS_DIR / "data",
        DOCS_DIR / "data" / "monthly",
        DOCS_DIR / "assets",
        DOCS_DIR / "paper",
        PAPER_DIR,
        CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def as_int(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def first_sentence(text, limit=260):
    text = clean_text(text)
    if not text:
        return ""
    match = re.search(r"(.{40,}?[.!?])\s+", text)
    out = match.group(1) if match else text
    if len(out) > limit:
        out = out[: limit - 1].rstrip() + "..."
    return out


def author_names(authors):
    names = []
    for author in authors or []:
        if isinstance(author, dict):
            name = clean_text(author.get("name") or author.get("fullname"))
        else:
            name = clean_text(author)
        if name and name not in names:
            names.append(name)
    return names


def normalize_paper(item, source_month):
    paper = item.get("paper") or {}
    paper_id = clean_text(paper.get("id") or item.get("paperId") or item.get("id"))
    title = clean_text(item.get("title") or paper.get("title"))
    summary = clean_text(paper.get("summary") or item.get("summary"))
    ai_summary = clean_text(paper.get("ai_summary") or item.get("ai_summary"))
    ai_keywords = paper.get("ai_keywords") or item.get("ai_keywords") or []
    if not isinstance(ai_keywords, list):
        ai_keywords = []

    authors = author_names(paper.get("authors") or item.get("authors") or [])
    submitted_by = item.get("submittedBy") or paper.get("submittedOnDailyBy") or {}
    organization = item.get("organization") or paper.get("organization") or {}
    published_at = clean_text(item.get("publishedAt") or paper.get("publishedAt"))
    submitted_at = clean_text(paper.get("submittedOnDailyAt") or item.get("submittedOnDailyAt"))

    github_repo = clean_text(paper.get("githubRepo") or item.get("githubRepo"))
    project_page = clean_text(paper.get("projectPage") or item.get("projectPage"))
    upvotes = as_int(paper.get("upvotes") if paper.get("upvotes") is not None else item.get("upvotes"))
    comments = as_int(item.get("numComments") if item.get("numComments") is not None else paper.get("numComments"))
    github_stars = as_int(paper.get("githubStars") if paper.get("githubStars") is not None else item.get("githubStars"))
    thumbnail = clean_text(item.get("thumbnail") or paper.get("thumbnail") or "")
    media_urls = item.get("mediaUrls") or paper.get("mediaUrls") or []
    if not isinstance(media_urls, list):
        media_urls = []

    hf_url = f"https://huggingface.co/papers/{paper_id}" if paper_id else ""
    arxiv_abs = f"https://arxiv.org/abs/{paper_id}" if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", paper_id) else ""
    arxiv_pdf = f"https://arxiv.org/pdf/{paper_id}.pdf" if arxiv_abs else ""

    record = {
        "id": paper_id,
        "title": title,
        "authors": authors,
        "authors_text": ", ".join(authors[:8]) + (" et al." if len(authors) > 8 else ""),
        "published_at": published_at,
        "published_date": published_at[:10],
        "submitted_at": submitted_at,
        "submitted_date": submitted_at[:10],
        "source_month": source_month,
        "source_months": [source_month],
        "year": source_month[:4],
        "summary": summary,
        "ai_summary": ai_summary,
        "ai_keywords": [clean_text(k) for k in ai_keywords if clean_text(k)],
        "upvotes": upvotes,
        "num_comments": comments,
        "github_stars": github_stars,
        "github_repo": github_repo,
        "project_page": project_page,
        "hf_url": hf_url,
        "arxiv_url": arxiv_abs,
        "pdf_url": arxiv_pdf,
        "thumbnail": thumbnail,
        "media_urls": media_urls,
        "submitted_by": clean_text(submitted_by.get("name") or submitted_by.get("fullname") or submitted_by.get("user")),
        "organization": clean_text(organization.get("fullname") or organization.get("name")),
    }
    record["category"] = classify(record)
    record["keyword_tags"] = assign_keywords(record)
    record["key_idea"] = ai_summary or first_sentence(summary) or f"HF Daily Paper entry for {title}."
    record["strengths"] = strengths(record)
    record["limitations"] = limitations(record)
    record["rank_score"] = engagement_score(record)
    return record


def searchable_text(record):
    pieces = [
        record.get("title", ""),
        record.get("summary", ""),
        record.get("ai_summary", ""),
        " ".join(record.get("ai_keywords") or []),
    ]
    return " ".join(pieces).lower()


def classify(record):
    text = searchable_text(record)
    best_name = "General Machine Learning and Optimization"
    best_score = 0
    for category in CATEGORY_RULES:
        score = 0
        for keyword in category["keywords"]:
            keyword_l = keyword.lower()
            if keyword_l in text:
                score += 3 if " " in keyword_l else 1
        if score > best_score:
            best_name = category["name"]
            best_score = score
    return best_name


def assign_keywords(record):
    text = searchable_text(record)
    tags = []
    for tag, _description, _color, needles in KEYWORD_CONVENTION:
        if any(needle.lower() in text for needle in needles):
            tags.append(tag)
    if not tags:
        tags = ["ai-research"]
    return tags[:6]


def engagement_score(record):
    return (
        record["upvotes"] * 10
        + record["num_comments"] * 3
        + int(math.log1p(record["github_stars"]) * 8)
        + (12 if record["github_repo"] else 0)
        + (8 if record["project_page"] else 0)
    )


def strengths(record):
    out = []
    if record["upvotes"]:
        out.append(f"HF community signal: {record['upvotes']} upvotes")
    if record["num_comments"]:
        out.append(f"discussion activity: {record['num_comments']} comments")
    if record["github_repo"]:
        repo_note = "official GitHub repository linked"
        if record["github_stars"]:
            repo_note += f" ({record['github_stars']:,} stars recorded by HF)"
        out.append(repo_note)
    if record["project_page"]:
        out.append("project page linked")
    if record["ai_keywords"]:
        out.append("HF AI keywords available")
    if not out:
        out.append("included in HF Daily Papers monthly archive")
    return "; ".join(out[:4])


def limitations(record):
    notes = [
        "metadata-driven entry; full PDF was not reviewed",
        "HF engagement reflects visibility, not methodological quality",
    ]
    if record["source_month"] == END_MONTH:
        notes.append("latest month is still time-sensitive and engagement may change")
    if not record["github_repo"]:
        notes.append("no GitHub repository was linked in HF metadata")
    return "; ".join(notes[:4])


def fetch_month(month, session):
    cache_path = CACHE_DIR / f"daily_papers_{month}.json"
    if "--reuse-cache" in sys.argv and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    all_items = []
    page = 0
    while True:
        params = {"month": month, "sort": "publishedAt", "limit": 100, "p": page}
        for attempt in range(5):
            try:
                response = session.get(API_URL, params=params, timeout=60)
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 + attempt * 2)
                    continue
                response.raise_for_status()
                payload = response.json()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 + attempt * 2)
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected response for {month} page {page}: {type(payload).__name__}")
        print(f"[fetch] {month} page {page}: {len(payload)}", flush=True)
        if not payload:
            break
        all_items.extend(payload)
        page += 1
        time.sleep(REQUEST_DELAY)

    cache_path.write_text(json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_items


def collect_papers():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    monthly_files = []
    records_by_id = {}
    for month in MONTHS:
        items = fetch_month(month, session)
        month_file = DATA_DIR / "monthly" / f"daily_papers_{month}.json"
        month_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        monthly_files.append({"month": month, "count": len(items), "file": f"data/monthly/{month_file.name}"})
        for item in items:
            record = normalize_paper(item, month)
            if not record["id"] or not record["title"]:
                continue
            existing = records_by_id.get(record["id"])
            if existing:
                if month not in existing["source_months"]:
                    existing["source_months"].append(month)
                if record["rank_score"] > existing["rank_score"]:
                    record["source_months"] = existing["source_months"]
                    records_by_id[record["id"]] = record
            else:
                records_by_id[record["id"]] = record

    papers = list(records_by_id.values())
    papers.sort(key=lambda p: (-p["rank_score"], -p["upvotes"], -p["github_stars"], p["source_month"], p["title"].lower()))
    for idx, paper in enumerate(papers, 1):
        paper["rank"] = idx
        paper["source_months_text"] = ", ".join(paper["source_months"])

    raw_payload = {
        "source": "https://huggingface.co/api/daily_papers",
        "query_months": MONTHS,
        "fetched_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "monthly_files": monthly_files,
        "note": "Raw monthly API payloads are split into one JSON file per month to stay below GitHub's single-file size limit.",
    }
    (DATA_DIR / MONTHLY_INDEX_JSON).write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return papers


def csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_data(papers):
    fields = [
        "rank",
        "id",
        "title",
        "authors_text",
        "published_date",
        "submitted_date",
        "source_month",
        "source_months_text",
        "year",
        "category",
        "keyword_tags",
        "upvotes",
        "num_comments",
        "github_stars",
        "github_repo",
        "project_page",
        "hf_url",
        "arxiv_url",
        "pdf_url",
        "thumbnail",
        "submitted_by",
        "organization",
        "key_idea",
        "strengths",
        "limitations",
        "ai_keywords",
        "ai_summary",
        "summary",
    ]
    (DATA_DIR / PAPERS_JSON).write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")
    with (DATA_DIR / PAPERS_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for paper in papers:
            writer.writerow({field: csv_value(paper.get(field, "")) for field in fields})

    taxonomy_fields = [
        "rank",
        "id",
        "title",
        "category",
        "keyword_tags",
        "key_idea",
        "strengths",
        "limitations",
        "upvotes",
        "num_comments",
        "github_stars",
        "hf_url",
        "github_repo",
        "project_page",
    ]
    with (DATA_DIR / TAXONOMY_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=taxonomy_fields, extrasaction="ignore")
        writer.writeheader()
        for paper in papers:
            writer.writerow({field: csv_value(paper.get(field, "")) for field in taxonomy_fields})


def category_stats(papers):
    grouped = defaultdict(list)
    for paper in papers:
        grouped[paper["category"]].append(paper)
    return {
        category: sorted(rows, key=lambda p: (-p["rank_score"], p["rank"]))
        for category, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def period_analysis(papers):
    by_month = Counter(p["source_month"] for p in papers)
    by_year = Counter(p["year"] for p in papers)
    by_category = Counter(p["category"] for p in papers)
    by_keyword = Counter(tag for p in papers for tag in p["keyword_tags"])
    analysis = {
        "generated_at": date.today().isoformat(),
        "source": "Hugging Face Daily Papers monthly API",
        "source_months": MONTHS,
        "period": PERIOD_TEXT,
        "total_papers": len(papers),
        "year_counts": dict(sorted(by_year.items())),
        "month_counts": {month: by_month.get(month, 0) for month in MONTHS},
        "category_counts": {"All Taxonomies": len(papers), **dict(by_category.most_common())},
        "keyword_counts": dict(by_keyword.most_common()),
        "github_repo_count": sum(1 for p in papers if p["github_repo"]),
        "project_page_count": sum(1 for p in papers if p["project_page"]),
        "arxiv_url_count": sum(1 for p in papers if p["arxiv_url"]),
        "top_upvoted": [
            {"rank": p["rank"], "id": p["id"], "title": p["title"], "upvotes": p["upvotes"], "hf_url": p["hf_url"]}
            for p in sorted(papers, key=lambda p: (-p["upvotes"], p["rank"]))[:25]
        ],
        "top_github_stars": [
            {"rank": p["rank"], "id": p["id"], "title": p["title"], "github_stars": p["github_stars"], "github_repo": p["github_repo"]}
            for p in sorted(papers, key=lambda p: (-p["github_stars"], p["rank"]))[:25]
            if p["github_stars"]
        ],
        "notes": [
            "This is a metadata-driven curation of Hugging Face Daily Papers monthly pages, not a full-PDF systematic review.",
            f"{END_MONTH} is interpreted as the state available when generated on {date.today().isoformat()}.",
        ],
    }
    (DATA_DIR / PERIOD_ANALYSIS_JSON).write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    return analysis


def is_valid_url(url):
    if not url:
        return True
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def audit_links(papers, sample_size=120):
    url_fields = ["hf_url", "arxiv_url", "pdf_url", "github_repo", "project_page", "thumbnail"]
    invalid = []
    counts = Counter()
    for paper in papers:
        for field in url_fields:
            url = paper.get(field, "")
            if url:
                counts[field] += 1
            if url and not is_valid_url(url):
                invalid.append({"id": paper["id"], "title": paper["title"], "field": field, "url": url})

    http_checked = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    candidates = []
    for paper in sorted(papers, key=lambda p: (-p["rank_score"], p["rank"])):
        for field in ("hf_url", "github_repo", "project_page", "arxiv_url"):
            url = paper.get(field)
            if url:
                candidates.append((paper, field, url))
        if len(candidates) >= sample_size:
            break
    for paper, field, url in candidates[:sample_size]:
        status = None
        ok = False
        error = ""
        try:
            response = session.head(url, allow_redirects=True, timeout=12)
            status = response.status_code
            if status in (403, 405) or status >= 500:
                response = session.get(url, stream=True, allow_redirects=True, timeout=12)
                status = response.status_code
            ok = status < 500
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        http_checked.append({"id": paper["id"], "field": field, "url": url, "status": status, "ok": ok, "error": error})
        time.sleep(0.04)

    payload = {
        "generated_at": date.today().isoformat(),
        "format_audit": {
            "url_counts": dict(counts),
            "invalid_count": len(invalid),
            "invalid": invalid,
        },
        "http_sample_audit": {
            "sample_size": len(http_checked),
            "checked_top_ranked_links_only": True,
            "ok_count": sum(1 for item in http_checked if item["ok"]),
            "failures": [item for item in http_checked if not item["ok"]],
            "checked": http_checked,
        },
        "scope_note": "All URLs were format-audited. HTTP checks were sampled from top-ranked HF, GitHub, project, and arXiv links to avoid hammering public services.",
    }
    (DATA_DIR / LINK_AUDIT_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def badge(tag):
    color = KEYWORD_COLORS.get(tag, "64748b")
    label = tag.replace("-", "--")
    return f"![{tag}](https://img.shields.io/badge/keyword-{label}-{color.lstrip('#')})"


def shields_static_message(value):
    return str(value).replace("-", "--").replace("_", "__").replace(" ", "_").replace("/", "%2F")


def link_for_paper(paper):
    return paper["hf_url"] or paper["arxiv_url"] or paper["project_page"] or paper["github_repo"] or "#"


def md_table_row(cells):
    return "| " + " | ".join(str(cell).replace("\n", " ").replace("|", "\\|") for cell in cells) + " |"


def write_readme(papers, analysis):
    grouped = category_stats(papers)
    category_lines = []
    for category, rows in grouped.items():
        meta = CATEGORY_BY_NAME.get(category, {})
        upvotes = sum(p["upvotes"] for p in rows)
        github_count = sum(1 for p in rows if p["github_repo"])
        top_keywords = Counter(tag for p in rows for tag in p["keyword_tags"]).most_common(5)
        category_lines.append(f"### {category}\n")
        category_lines.append(f"- Papers covered: **{len(rows):,}**")
        category_lines.append(f"- HF upvotes in category: **{upvotes:,}**")
        category_lines.append(f"- GitHub repos linked: **{github_count:,}**")
        category_lines.append(f"- Top keyword tags: {', '.join(f'`{tag}` ({count})' for tag, count in top_keywords) or 'none'}")
        category_lines.append("- Category overview:")
        for trend in meta.get("trends", [])[:3]:
            category_lines.append(f"  - {trend}")
        category_lines.append("- Limitations:")
        for note in meta.get("limitations", [])[:3]:
            category_lines.append(f"  - {note}")
        category_lines.append("\n<details>")
        category_lines.append(f"<summary><strong>Show representative papers for {html.escape(category)}</strong></summary>\n")
        category_lines.append(md_table_row(["Rank", "Paper", "Month", "Signals", "Tags", "Key idea"]))
        category_lines.append(md_table_row(["---:", "---", "---", "---", "---", "---"]))
        for paper in rows[:20]:
            signals = f"{paper['upvotes']} upvotes"
            if paper["github_stars"]:
                signals += f"; {paper['github_stars']:,} GitHub stars"
            links = f"[HF]({paper['hf_url']})"
            if paper["github_repo"]:
                links += f" · [Code]({paper['github_repo']})"
            if paper["project_page"]:
                links += f" · [Project]({paper['project_page']})"
            title = f"[{paper['title']}]({link_for_paper(paper)})<br><sub>{paper['authors_text'] or 'Unknown authors'}</sub><br><sub>{links}</sub>"
            category_lines.append(md_table_row([paper["rank"], title, paper["source_month"], signals, " ".join(badge(tag) for tag in paper["keyword_tags"][:3]), paper["key_idea"]]))
        if len(rows) > 20:
            category_lines.append(f"\n_{len(rows) - 20:,} additional papers in this category are available in `data/{PAPERS_CSV}` and the interactive website._")
        category_lines.append("\n</details>\n")

    month_rows = [md_table_row(["Month", "Papers"])]
    month_rows.append(md_table_row(["---:", "---:"]))
    for month in MONTHS:
        month_rows.append(md_table_row([month, f"{analysis['month_counts'].get(month, 0):,}"]))

    keyword_cards = []
    for tag, description, _color, _needles in KEYWORD_CONVENTION:
        keyword_cards.append(f"- {badge(tag)} **{tag}**: {description}")
    keyword_cards.append("- ![ai-research](https://img.shields.io/badge/keyword-ai--research-64748b) **ai-research**: General AI research when no narrower deterministic tag is triggered.")
    website_badge_message = shields_static_message(f"{OWNER}.github.io/{REPO_NAME}")
    top_month = max(analysis["month_counts"].items(), key=lambda item: item[1])
    top_keyword = max(analysis["keyword_counts"].items(), key=lambda item: item[1])

    readme = f"""# Awesome Hugging Face Papers

[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

A taxonomy-first archive of Hugging Face Daily Papers from {PERIOD_TEXT}.

<p align="center">
  <a href="https://{OWNER}.github.io/{REPO_NAME}/">
    <img src="https://img.shields.io/badge/Open_Interactive_Website-{website_badge_message}-0f766e?style=for-the-badge" alt="Open Interactive Website">
  </a>
</p>

Generated on {date.today().isoformat()} from the public Hugging Face Daily Papers API. This edition covers **{len(MONTHS)} monthly pages** from `{START_MONTH}` through `{END_MONTH}` and includes **{len(papers):,} unique papers** submitted to HF Daily Papers during that period.

## Project Links

- Open Interactive Website: https://{OWNER}.github.io/{REPO_NAME}/
- Complete paper dataset: `data/{PAPERS_CSV}`
- Taxonomy dataset with key ideas, strengths, and limitations: `data/{TAXONOMY_CSV}`
- Raw monthly API archive index: `data/{MONTHLY_INDEX_JSON}`
- Per-month raw API payloads: `data/monthly/`
- Period analysis: `data/{PERIOD_ANALYSIS_JSON}`
- Link audit: `data/{LINK_AUDIT_JSON}`
- English review draft: `paper/review_en.html`
- Korean review draft: `paper/review_ko.html`

## Keywords Convention

These badges define the keyword tags used to scan and extend this collection.

{chr(10).join(keyword_cards)}

## Taxonomy Overview

- **All Taxonomies**: {len(papers):,} papers
- **Years covered**: 2023, 2024, 2025, 2026
- **Months covered**: {START_MONTH} through {END_MONTH}
- **GitHub repositories linked in HF metadata**: {analysis['github_repo_count']:,}
- **Project pages linked in HF metadata**: {analysis['project_page_count']:,}
- **arXiv links generated from HF paper IDs**: {analysis['arxiv_url_count']:,}
- **HF upvotes captured**: {sum(p['upvotes'] for p in papers):,}
- **HF comments captured**: {sum(p['num_comments'] for p in papers):,}

## Research Insights

- **Most active month**: `{top_month[0]}` with **{top_month[1]:,} papers**
- **Most common keyword convention**: `{top_keyword[0]}` across **{top_keyword[1]:,} tagged papers**
- **Top taxonomy by paper count**: `{next(iter(grouped.keys()))}` with **{len(next(iter(grouped.values()))):,} papers**
- The interactive website supports period range, taxonomy, keyword convention, repository-link, and text-search filtering.

## Taxonomy Collections

{chr(10).join(category_lines)}

## Research Timeline

{chr(10).join(month_rows)}

## Methodology

The collection uses the Hugging Face Daily Papers monthly API endpoint, equivalent to the public monthly pages at `https://huggingface.co/papers/month/YYYY-MM`. Each month from `{START_MONTH}` through `{END_MONTH}` is paginated until the API returns no more results. Records are deduplicated by HF paper/arXiv id, then enriched with deterministic taxonomy, keyword tags, key ideas, strengths, and limitations using only public metadata fields exposed by Hugging Face.

Ranking is not a quality score. It is a deterministic browsing order based on HF upvotes, discussion comments, linked GitHub stars, and the presence of repository or project-page metadata. The full archive keeps every collected monthly paper rather than selecting only top papers.

This repository follows `github-awesome-skill2` in metadata-adapter mode. The local `jehyunlee/paper-curation` checkout was inspected, but full PDF review stages were not run because they require separate explicit approval for paid or metered APIs and are impractical for this full monthly HF archive.

## Caveats

- This is a metadata-driven archive, not a full systematic review of every PDF.
- HF upvotes, comments, and GitHub stars measure visibility and community attention, not scientific validity.
- `{END_MONTH}` is time-sensitive; counts may change if Hugging Face updates historical metadata.
- Some HF entries have missing repository, project page, author, thumbnail, or keyword metadata.
- Link audit combines full URL format checks with sampled HTTP checks to avoid excessive requests to public services.

## Acknowledgements

This repository and interactive site were created with appreciation for [jehyunlee/paper-curation](https://github.com/jehyunlee/paper-curation). Its workflow informed the taxonomy-first organization, provenance tracking, and honest metadata-driven limitations used here.

## License

CC-BY-4.0 for text and metadata curation; upstream paper metadata belongs to the original sources.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    write_markdown_html(ROOT / "README.md", ROOT / "README.html", "Awesome Hugging Face Papers")


def markdown_shell(title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, 'Noto Sans KR', sans-serif; max-width: 1000px; margin: 36px auto; padding: 0 22px; line-height: 1.65; color: #172033; background: #fbfcfe; }}
    h1, h2, h3 {{ line-height: 1.2; color: #111827; }}
    h1 {{ border-bottom: 2px solid #d8e0ea; padding-bottom: 12px; }}
    a {{ color: #0f766e; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d8e0ea; padding: 9px; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    code {{ background: #edf2f7; padding: 2px 5px; border-radius: 4px; }}
    img {{ max-width: 100%; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def write_markdown_html(md_path, html_path, title):
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "toc"])
    html_path.write_text(markdown_shell(title, body), encoding="utf-8")


def escape_attr(value):
    return html.escape(str(value or ""), quote=True)


def write_charts(papers):
    grouped = category_stats(papers)
    width = 1180
    bar_h = 24
    gap = 12
    left = 360
    height = 70 + len(grouped) * (bar_h + gap)
    max_count = max((len(rows) for rows in grouped.values()), default=1)
    rows = [
        f'<text x="20" y="34" font-size="22" font-family="Arial" fill="#111827">Papers by Taxonomy</text>'
    ]
    y = 62
    for category, rows_for_category in grouped.items():
        count = len(rows_for_category)
        color = CATEGORY_BY_NAME.get(category, {}).get("color", "#64748b")
        bar_w = int((width - left - 60) * count / max_count)
        rows.append(f'<text x="20" y="{y + 17}" font-size="14" font-family="Arial" fill="#334155">{html.escape(category)}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{color}"></rect>')
        rows.append(f'<text x="{left + bar_w + 8}" y="{y + 17}" font-size="14" font-family="Arial" fill="#111827">{count:,}</text>')
        y += bar_h + gap
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Category distribution">{"".join(rows)}</svg>'
    (DOCS_DIR / "assets" / "category_distribution.svg").write_text(svg, encoding="utf-8")

    month_counts = Counter(p["source_month"] for p in papers)
    w = 1180
    h = 340
    margin_l = 58
    margin_b = 54
    max_month = max(month_counts.values(), default=1)
    bar_w = max(10, int((w - margin_l - 40) / len(MONTHS)) - 4)
    parts = [f'<text x="20" y="34" font-size="22" font-family="Arial" fill="#111827">Monthly Coverage</text>']
    baseline = h - margin_b
    for i, month in enumerate(MONTHS):
        count = month_counts.get(month, 0)
        x = margin_l + i * ((w - margin_l - 40) / len(MONTHS))
        bh = int((h - 100) * count / max_month)
        color = "#2563eb" if month[:4] in ("2023", "2025") else "#16a34a" if month[:4] == "2024" else "#dc2626"
        parts.append(f'<rect x="{x:.1f}" y="{baseline - bh}" width="{bar_w}" height="{bh}" rx="3" fill="{color}"></rect>')
        if i % 3 == 0:
            parts.append(f'<text x="{x:.1f}" y="{baseline + 20}" font-size="11" font-family="Arial" fill="#475569" transform="rotate(45 {x:.1f},{baseline + 20})">{month}</text>')
    parts.append(f'<line x1="{margin_l}" y1="{baseline}" x2="{w - 30}" y2="{baseline}" stroke="#94a3b8"></line>')
    parts.append(f'<text x="18" y="62" font-size="13" font-family="Arial" fill="#475569">Max monthly count: {max_month:,}</text>')
    month_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="Monthly paper counts">{"".join(parts)}</svg>'
    (DOCS_DIR / "assets" / "monthly_coverage.svg").write_text(month_svg, encoding="utf-8")


def site_paper_payload(papers):
    keep = []
    for paper in papers:
        keep.append({
            "rank": paper["rank"],
            "id": paper["id"],
            "title": paper["title"],
            "authors": paper["authors_text"],
            "month": paper["source_month"],
            "year": paper["year"],
            "category": paper["category"],
            "tags": paper["keyword_tags"],
            "upvotes": paper["upvotes"],
            "comments": paper["num_comments"],
            "stars": paper["github_stars"],
            "summary": paper["key_idea"],
            "strengths": paper["strengths"],
            "limitations": paper["limitations"],
            "hf": paper["hf_url"],
            "arxiv": paper["arxiv_url"],
            "repo": paper["github_repo"],
            "project": paper["project_page"],
            "thumb": paper["thumbnail"],
        })
    return keep


def write_site(papers, analysis):
    payload = json.dumps(site_paper_payload(papers), ensure_ascii=False, separators=(",", ":"))
    safe_payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    categories = ["All Taxonomies"] + list(category_stats(papers).keys())
    category_options = "\n".join(f'<option value="{escape_attr(cat)}">{html.escape(cat)}</option>' for cat in categories)
    month_options = "\n".join(f'<option value="{month}">{month}</option>' for month in MONTHS)
    period_presets = [("all", START_MONTH, END_MONTH, f"All months ({START_MONTH} to {END_MONTH})")]
    for year in sorted({month[:4] for month in MONTHS}):
        year_months = [month for month in MONTHS if month.startswith(year)]
        period_presets.append((year, year_months[0], year_months[-1], year))
    for value, start, label in [
        ("2024-2026", "2024-01", "2024 to 2026"),
        ("2025-2026", "2025-01", "2025 to 2026"),
    ]:
        if start in MONTHS:
            period_presets.append((value, start, END_MONTH, label))
    period_options = ['<option value="custom">Custom range</option>']
    for value, start, end, label in period_presets:
        selected = " selected" if value == "all" else ""
        period_options.append(f'<option value="{escape_attr(value)}" data-from="{start}" data-to="{end}"{selected}>{html.escape(label)}</option>')
    period_options = "\n".join(period_options)
    keyword_items = list(KEYWORD_CONVENTION) + [("ai-research", "General AI research when no narrower deterministic tag is triggered.", "#64748b", [])]
    keyword_options = '<option value="All Keywords">All Keywords</option>\n' + "\n".join(
        f'<option value="{escape_attr(tag)}">{html.escape(tag)}</option>'
        for tag, _description, _color, _needles in keyword_items
    )
    keyword_cards = "\n".join(
        f'<button class="keyword-card" type="button" data-keyword="{escape_attr(tag)}" aria-pressed="false"><span class="tag" style="--tag-color:{color}">{html.escape(tag)}</span><span>{html.escape(description)}</span></button>'
        for tag, description, color, _needles in keyword_items
    )
    color_map = {tag: color for tag, _desc, color, _needles in keyword_items}
    keyword_descriptions = {tag: desc for tag, desc, _color, _needles in keyword_items}
    stat_cards = "\n".join(
        [
            f'<div><strong>{len(papers):,}</strong><span>Papers</span></div>',
            f'<div><strong>{len(MONTHS)}</strong><span>Months</span></div>',
            f'<div><strong>{analysis["github_repo_count"]:,}</strong><span>GitHub Links</span></div>',
            f'<div><strong>{sum(p["upvotes"] for p in papers):,}</strong><span>HF Upvotes</span></div>',
        ]
    )
    months_json = json.dumps(MONTHS)
    colors_json = json.dumps(color_map, ensure_ascii=False)
    keyword_descriptions_json = json.dumps(keyword_descriptions, ensure_ascii=False)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Awesome Hugging Face Papers</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #111827;
      --muted: #5b6472;
      --line: #d7dee8;
      --panel: #ffffff;
      --wash: #f5f7fa;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --accent-3: #be123c;
      --soft: #eef5f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", sans-serif;
      color: var(--ink);
      background: #fbfcfe;
      line-height: 1.55;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
    .topbar {{ display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }}
    h1 {{ margin: 0; font-size: clamp(26px, 4vw, 42px); line-height: 1.05; letter-spacing: 0; }}
    .subtitle {{ margin: 8px 0 0; color: var(--muted); max-width: 780px; }}
    .links {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .links a {{ color: #fff; background: var(--accent); text-decoration: none; padding: 8px 11px; border-radius: 6px; font-size: 14px; }}
    .links a:nth-child(2) {{ background: var(--accent-2); }}
    .links a:nth-child(3) {{ background: var(--accent-3); }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-top: 18px; }}
    .stats div {{ border: 1px solid var(--line); background: var(--wash); padding: 12px; border-radius: 6px; }}
    .stats strong {{ display: block; font-size: 24px; }}
    .stats span {{ color: var(--muted); font-size: 13px; }}
    .controls {{
      position: sticky;
      top: 0;
      z-index: 3;
      border-bottom: 1px solid var(--line);
      background: rgba(251, 252, 254, 0.96);
      backdrop-filter: blur(10px);
    }}
    .control-grid {{ display: grid; grid-template-columns: minmax(220px, 1.5fr) repeat(5, minmax(132px, 1fr)) minmax(140px, .8fr); gap: 10px; }}
    input, select {{
      width: 100%;
      min-height: 40px;
      border: 1px solid #c8d2df;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
    }}
    label.check {{ display: flex; align-items: center; gap: 8px; border: 1px solid #c8d2df; border-radius: 6px; padding: 8px 10px; background: #fff; color: var(--muted); }}
    label.check input {{ width: auto; min-height: 0; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
    .figures {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 14px 0 24px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 12px; }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
    .analysis-section {{ margin: 26px 0; }}
    .section-head {{ display: flex; gap: 16px; justify-content: space-between; align-items: baseline; flex-wrap: wrap; margin-bottom: 12px; }}
    .section-head h2 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .section-head p {{ margin: 0; color: var(--muted); max-width: 720px; }}
    .timeline-chart {{ display: grid; gap: 8px; border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 12px; }}
    .timeline-row {{ display: grid; grid-template-columns: 74px minmax(120px, 1fr) 68px; gap: 10px; align-items: center; font-size: 13px; }}
    .timeline-track {{ height: 12px; background: #e7edf4; border-radius: 999px; overflow: hidden; }}
    .timeline-bar {{ height: 100%; min-width: 2px; background: var(--accent); border-radius: 999px; }}
    .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .insight-card {{ border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 12px; }}
    .insight-card strong {{ display: block; font-size: 24px; line-height: 1.15; }}
    .insight-card span {{ display: block; color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .insight-list {{ border: 1px solid var(--line); border-radius: 6px; background: var(--soft); padding: 12px 16px; margin: 10px 0 0; }}
    .insight-list h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .insight-list ol {{ margin: 0; padding-left: 22px; }}
    .insight-list li {{ margin: 5px 0; }}
    .keyword-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-top: 12px; }}
    .keyword-card {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fff; display: grid; gap: 8px; align-items: start; text-align: left; font: inherit; cursor: pointer; color: var(--muted); }}
    .keyword-card[aria-pressed="true"], .keyword-card.is-selected {{ border-color: var(--accent); box-shadow: 0 0 0 2px rgba(15,118,110,.14); color: var(--ink); }}
    .keyword-card .tag {{ display: inline-block; color: #fff; background: var(--tag-color); border-radius: 4px; padding: 3px 7px; font-size: 12px; font-weight: 700; }}
    .keyword-card span:not(.tag) {{ display: block; width: 100%; font-size: 13px; line-height: 1.45; }}
    .keyword-filter-status {{ color: var(--accent); font-weight: 700; margin: 10px 0 0; }}
    .result-line {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin: 8px 0 14px; color: var(--muted); }}
    .paper-list {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 12px; }}
    .paper-card {{ border: 1px solid var(--line); border-radius: 6px; background: #fff; overflow: hidden; display: grid; grid-template-columns: 116px 1fr; min-height: 160px; }}
    .thumb {{ background: #eef2f7; min-height: 100%; }}
    .thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .thumb .fallback {{ height: 100%; min-height: 160px; display: grid; place-items: center; color: #64748b; font-weight: 700; padding: 10px; text-align: center; }}
    .paper-body {{ padding: 12px; min-width: 0; }}
    .paper-meta {{ color: var(--muted); font-size: 12px; display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }}
    .paper-card h3 {{ margin: 0 0 5px; font-size: 16px; line-height: 1.25; letter-spacing: 0; }}
    .paper-card h3 a {{ color: var(--ink); text-decoration: none; }}
    .paper-card h3 a:hover {{ color: var(--accent); }}
    .authors {{ color: var(--muted); font-size: 13px; margin: 0 0 8px; }}
    .summary {{ margin: 0 0 8px; font-size: 13px; }}
    .chips {{ display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px; }}
    .chip {{ color: #fff; border-radius: 4px; padding: 2px 6px; font-size: 11px; font-weight: 700; }}
    .signals {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .card-links {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .card-links a {{ color: var(--accent-2); font-size: 12px; text-decoration: none; font-weight: 700; }}
    .empty {{ padding: 36px; border: 1px dashed #a7b2c1; border-radius: 6px; text-align: center; color: var(--muted); }}
    @media (max-width: 860px) {{
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      .control-grid {{ grid-template-columns: 1fr 1fr; }}
      .figures {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      .wrap, main {{ padding: 16px; }}
      .stats, .control-grid, .paper-list {{ grid-template-columns: 1fr; }}
      .paper-card {{ grid-template-columns: 1fr; }}
      .thumb .fallback {{ min-height: 90px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="topbar">
        <div>
          <h1>Awesome Hugging Face Papers</h1>
          <p class="subtitle">A taxonomy-first archive of Hugging Face Daily Papers from {PERIOD_TEXT}. Every paper in the monthly pages is kept in the dataset and can be filtered below.</p>
        </div>
        <nav class="links" aria-label="Project links">
          <a href="../README.md">README</a>
          <a href="data/{PAPERS_CSV}">CSV</a>
          <a href="data/{PERIOD_ANALYSIS_JSON}">Analysis JSON</a>
        </nav>
      </div>
      <section class="stats" aria-label="Corpus statistics">
        {stat_cards}
      </section>
    </div>
  </header>
  <section class="controls">
    <div class="wrap control-grid">
      <input id="q" type="search" placeholder="Search title, authors, summaries, tags">
      <select id="category">{category_options}</select>
      <select id="periodPreset" aria-label="Period preset">{period_options}</select>
      <select id="fromMonth" aria-label="Start month">{month_options}</select>
      <select id="toMonth" aria-label="End month">{month_options}</select>
      <select id="keyword" aria-label="Keyword convention">{keyword_options}</select>
      <label class="check"><input id="hasRepo" type="checkbox"> GitHub only</label>
    </div>
  </section>
  <main>
    <section class="figures" aria-label="Charts">
      <figure>
        <img src="assets/category_distribution.svg" alt="Category distribution">
        <figcaption>Taxonomy-first distribution over the full HF Daily Papers archive.</figcaption>
      </figure>
      <figure>
        <img src="assets/monthly_coverage.svg" alt="Monthly paper counts">
        <figcaption>Monthly coverage from {START_MONTH} through {END_MONTH}.</figcaption>
      </figure>
    </section>
    <section class="analysis-section" aria-labelledby="timelineTitle">
      <div class="section-head">
        <h2 id="timelineTitle">Research Timeline</h2>
        <p id="timelineSummary"></p>
      </div>
      <div id="timelineChart" class="timeline-chart"></div>
    </section>
    <section class="analysis-section" aria-labelledby="insightsTitle">
      <div class="section-head">
        <h2 id="insightsTitle">Research Insights</h2>
        <p>Insights update from the currently selected period, taxonomy, keyword convention, search, and repository-link filters.</p>
      </div>
      <div id="insights" class="insight-grid"></div>
      <div id="topPapers" class="insight-list"></div>
    </section>
    <section class="analysis-section" id="keywords-convention" aria-labelledby="keywordsTitle">
      <div class="section-head">
        <h2 id="keywordsTitle">Keywords Convention</h2>
        <p>Click a keyword card or use the keyword selector to inspect papers grouped by the collection's keyword convention.</p>
      </div>
      <div class="keyword-grid">{keyword_cards}</div>
      <p class="keyword-filter-status" id="keywordFilterStatus"></p>
    </section>
    <div class="result-line">
      <strong id="resultCount"></strong>
      <span id="activeFilters"></span>
    </div>
    <section id="papers" class="paper-list" aria-live="polite"></section>
  </main>
  <script id="paper-data" type="application/json">{safe_payload}</script>
  <script>
    const papers = JSON.parse(document.getElementById('paper-data').textContent);
    const colors = {colors_json};
    const months = {months_json};
    const keywordDescriptions = {keyword_descriptions_json};
    const els = {{
      q: document.getElementById('q'),
      category: document.getElementById('category'),
      periodPreset: document.getElementById('periodPreset'),
      fromMonth: document.getElementById('fromMonth'),
      toMonth: document.getElementById('toMonth'),
      keyword: document.getElementById('keyword'),
      hasRepo: document.getElementById('hasRepo'),
      list: document.getElementById('papers'),
      count: document.getElementById('resultCount'),
      filters: document.getElementById('activeFilters'),
      timeline: document.getElementById('timelineChart'),
      timelineSummary: document.getElementById('timelineSummary'),
      insights: document.getElementById('insights'),
      topPapers: document.getElementById('topPapers'),
      keywordStatus: document.getElementById('keywordFilterStatus')
    }};
    els.fromMonth.value = months[0];
    els.toMonth.value = months[months.length - 1];
    function textMatch(p, q) {{
      if (!q) return true;
      const haystack = [p.title, p.authors, p.summary, p.category, p.tags.join(' ')].join(' ').toLowerCase();
      return haystack.includes(q);
    }}
    function link(label, href) {{
      return href ? `<a href="${{href}}" target="_blank" rel="noopener">${{label}}</a>` : '';
    }}
    function card(p) {{
      const chips = p.tags.map(tag => `<span class="chip" style="background:${{colors[tag] || '#64748b'}}">${{tag}}</span>`).join('');
      const thumb = p.thumb ? `<img src="${{p.thumb}}" alt="">` : `<div class="fallback">${{p.category.split(',')[0]}}</div>`;
      const signals = `${{p.upvotes}} upvotes | ${{p.comments}} comments${{p.stars ? ' | ' + p.stars.toLocaleString() + ' stars' : ''}}`;
      const links = [link('HF', p.hf), link('arXiv', p.arxiv), link('Code', p.repo), link('Project', p.project)].filter(Boolean).join('');
      return `<article class="paper-card">
        <div class="thumb">${{thumb}}</div>
        <div class="paper-body">
          <div class="paper-meta"><span>#${{p.rank}}</span><span>${{p.month}}</span><span>${{p.category}}</span></div>
          <h3><a href="${{p.hf}}" target="_blank" rel="noopener">${{escapeHtml(p.title)}}</a></h3>
          <p class="authors">${{escapeHtml(p.authors || 'Unknown authors')}}</p>
          <div class="chips">${{chips}}</div>
          <p class="summary">${{escapeHtml(p.summary)}}</p>
          <div class="signals">${{signals}}</div>
          <div class="card-links">${{links}}</div>
        </div>
      </article>`;
    }}
    function escapeHtml(value) {{
      return String(value || '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[ch]));
    }}
    function periodBounds() {{
      const from = els.fromMonth.value;
      const to = els.toMonth.value;
      return from <= to ? {{from, to}} : {{from: to, to: from}};
    }}
    function rangeMonths(from, to) {{
      return months.filter(month => month >= from && month <= to);
    }}
    function topEntries(map, limit) {{
      return Array.from(map.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, limit);
    }}
    function filterPapers() {{
      const q = els.q.value.trim().toLowerCase();
      const category = els.category.value;
      const bounds = periodBounds();
      const keyword = els.keyword.value;
      const hasRepo = els.hasRepo.checked;
      return papers.filter(p =>
        textMatch(p, q) &&
        (category === 'All Taxonomies' || p.category === category) &&
        (p.month >= bounds.from && p.month <= bounds.to) &&
        (keyword === 'All Keywords' || p.tags.includes(keyword)) &&
        (!hasRepo || !!p.repo)
      );
    }}
    function renderTimeline(filtered) {{
      const bounds = periodBounds();
      const visibleMonths = rangeMonths(bounds.from, bounds.to);
      const counts = new Map(visibleMonths.map(month => [month, 0]));
      filtered.forEach(p => counts.set(p.month, (counts.get(p.month) || 0) + 1));
      const maxCount = Math.max(1, ...Array.from(counts.values()));
      els.timelineSummary.textContent = `${{bounds.from}} to ${{bounds.to}} | ${{filtered.length.toLocaleString()}} matching papers across ${{visibleMonths.length}} months`;
      els.timeline.innerHTML = visibleMonths.map(month => {{
        const count = counts.get(month) || 0;
        const width = count ? Math.max(2, Math.round((count / maxCount) * 100)) : 0;
        return `<div class="timeline-row">
          <strong>${{month}}</strong>
          <div class="timeline-track" aria-label="${{month}} ${{count}} papers"><div class="timeline-bar" style="width:${{width}}%"></div></div>
          <span>${{count.toLocaleString()}}</span>
        </div>`;
      }}).join('');
    }}
    function renderInsights(filtered) {{
      const categoryCounts = new Map();
      const keywordCounts = new Map();
      let upvotes = 0;
      let comments = 0;
      let repoCount = 0;
      let projectCount = 0;
      filtered.forEach(p => {{
        categoryCounts.set(p.category, (categoryCounts.get(p.category) || 0) + 1);
        p.tags.forEach(tag => keywordCounts.set(tag, (keywordCounts.get(tag) || 0) + 1));
        upvotes += Number(p.upvotes || 0);
        comments += Number(p.comments || 0);
        if (p.repo) repoCount += 1;
        if (p.project) projectCount += 1;
      }});
      const topCategory = topEntries(categoryCounts, 1)[0] || ['none', 0];
      const topKeyword = topEntries(keywordCounts, 1)[0] || ['none', 0];
      const avgUpvotes = filtered.length ? Math.round(upvotes / filtered.length) : 0;
      const cards = [
        {{ label: 'Matching papers', value: filtered.length.toLocaleString(), detail: 'Current filtered corpus' }},
        {{ label: 'Top taxonomy', value: topCategory[1].toLocaleString(), detail: topCategory[0] }},
        {{ label: 'Top keyword', value: topKeyword[1].toLocaleString(), detail: topKeyword[0] }},
        {{ label: 'GitHub linked', value: repoCount.toLocaleString(), detail: `${{projectCount.toLocaleString()}} project pages` }},
        {{ label: 'HF upvotes', value: upvotes.toLocaleString(), detail: `${{avgUpvotes.toLocaleString()}} average per paper` }},
        {{ label: 'HF comments', value: comments.toLocaleString(), detail: 'Discussion signals captured from HF' }}
      ];
      els.insights.innerHTML = cards.map(item => `<article class="insight-card"><strong>${{escapeHtml(item.value)}}</strong><span>${{escapeHtml(item.label)}} - ${{escapeHtml(item.detail)}}</span></article>`).join('');
      const topPapers = [...filtered].sort((a, b) => (b.upvotes - a.upvotes) || (b.stars - a.stars) || (a.rank - b.rank)).slice(0, 5);
      els.topPapers.innerHTML = topPapers.length
        ? `<h3>Most visible papers in this view</h3><ol>${{topPapers.map(p => `<li><a href="${{p.hf}}" target="_blank" rel="noopener">${{escapeHtml(p.title)}}</a> <span>(${{p.month}}, ${{p.upvotes.toLocaleString()}} upvotes${{p.stars ? ', ' + p.stars.toLocaleString() + ' stars' : ''}})</span></li>`).join('')}}</ol>`
        : '<h3>Most visible papers in this view</h3><p>No papers match the current filters.</p>';
    }}
    function syncKeywordCards(filtered) {{
      const keyword = els.keyword.value;
      document.querySelectorAll('[data-keyword]').forEach(button => {{
        const selected = button.dataset.keyword === keyword;
        button.setAttribute('aria-pressed', String(selected));
        button.classList.toggle('is-selected', selected);
      }});
      const description = keyword === 'All Keywords' ? 'all keyword conventions' : `${{keyword}} - ${{keywordDescriptions[keyword] || ''}}`;
      els.keywordStatus.textContent = `Selected keyword: ${{description}} | Matching papers: ${{filtered.length.toLocaleString()}}`;
    }}
    function syncPreset() {{
      const bounds = periodBounds();
      const matched = Array.from(els.periodPreset.options).find(option => option.dataset.from === bounds.from && option.dataset.to === bounds.to);
      els.periodPreset.value = matched ? matched.value : 'custom';
    }}
    function normalizeRangeInputs() {{
      if (els.fromMonth.value > els.toMonth.value) {{
        const oldFrom = els.fromMonth.value;
        els.fromMonth.value = els.toMonth.value;
        els.toMonth.value = oldFrom;
      }}
    }}
    function render() {{
      const bounds = periodBounds();
      const category = els.category.value;
      const keyword = els.keyword.value;
      const hasRepo = els.hasRepo.checked;
      const q = els.q.value.trim();
      const filtered = filterPapers();
      els.count.textContent = `${{filtered.length.toLocaleString()}} papers`;
      els.filters.textContent = [category, `${{bounds.from}} to ${{bounds.to}}`, keyword === 'All Keywords' ? 'all keywords' : keyword, hasRepo ? 'GitHub linked' : 'all links', q ? 'search active' : ''].filter(Boolean).join(' | ');
      renderTimeline(filtered);
      renderInsights(filtered);
      syncKeywordCards(filtered);
      const shown = filtered.slice(0, 500);
      els.list.innerHTML = shown.length ? shown.map(card).join('') : '<div class="empty">No papers match the current filters.</div>';
      if (filtered.length > shown.length) {{
        els.list.insertAdjacentHTML('beforeend', `<div class="empty">Showing first ${{shown.length.toLocaleString()}} of ${{filtered.length.toLocaleString()}} matching papers. Refine search or use the CSV for the complete filtered set.</div>`);
      }}
    }}
    function applyPeriodPreset() {{
      const selected = els.periodPreset.selectedOptions[0];
      if (!selected || selected.value === 'custom') {{
        render();
        return;
      }}
      els.fromMonth.value = selected.dataset.from;
      els.toMonth.value = selected.dataset.to;
      render();
    }}
    function handleRangeChange() {{
      normalizeRangeInputs();
      syncPreset();
      render();
    }}
    els.periodPreset.addEventListener('input', applyPeriodPreset);
    [els.fromMonth, els.toMonth].forEach(el => el.addEventListener('input', handleRangeChange));
    [els.q, els.category, els.keyword, els.hasRepo].forEach(el => el.addEventListener('input', render));
    document.querySelectorAll('[data-keyword]').forEach(button => button.addEventListener('click', () => {{
      els.keyword.value = button.dataset.keyword;
      render();
      document.getElementById('papers').scrollIntoView({{behavior: 'smooth', block: 'start'}});
    }}));
    render();
  </script>
</body>
</html>
"""
    (DOCS_DIR / "index.html").write_text(html_doc, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")


def write_reviews(papers, analysis):
    cats = category_stats(papers)
    top = sorted(papers, key=lambda p: (-p["upvotes"], p["rank"]))[:12]
    top_lines = "\n".join(
        f"- [{paper['title']}]({paper['hf_url']}) ({paper['source_month']}, {paper['upvotes']} upvotes): {paper['key_idea']}"
        for paper in top
    )
    cat_lines = "\n".join(f"- {category}: {len(rows):,} papers" for category, rows in cats.items())
    en_md = f"""# Hugging Face Daily Papers Review ({PERIOD_TEXT})

## Abstract

This review draft summarizes a metadata-driven archive of {len(papers):,} unique Hugging Face Daily Papers submitted from {START_MONTH} through {END_MONTH}. The archive keeps every monthly paper surfaced through the public HF Daily Papers API, then organizes the corpus into taxonomy-first collections with deterministic keyword tags, key ideas, strengths, and limitations.

## Method

Each monthly page was fetched through `https://huggingface.co/api/daily_papers?month=YYYY-MM`, paginated until empty, deduplicated by HF paper/arXiv id, and enriched only from public metadata. No paid API, paid LLM, paid translation, or paid compute was used.

## Taxonomy Counts

{cat_lines}

## Highly Visible Papers

{top_lines}

## Interpretation

The corpus shows how HF Daily Papers became a practical signal layer for open AI research: papers with code, project pages, demos, model releases, and benchmark artifacts are easier to discover and reuse. The strongest metadata signals cluster around foundation models, agents, multimodal models, generative media, efficient training/inference, and evaluation resources.

## Limitations

This is not a full-PDF systematic review. HF upvotes, comments, and GitHub stars measure community visibility, not scientific validity. Full methodological claims require reading the papers, code, datasets, and evaluation details directly.
"""
    ko_md = f"""# Hugging Face Daily Papers 리뷰 ({PERIOD_TEXT})

## 초록

이 문서는 {START_MONTH}부터 {END_MONTH}까지 Hugging Face Daily Papers 월별 페이지에 올라온 고유 논문 {len(papers):,}편을 metadata 기반으로 정리한 리뷰 초안입니다. 모든 월별 논문을 보존하고, 제목/초록/HF AI summary/키워드/업보트/댓글/GitHub 링크 같은 공개 메타데이터를 이용해 taxonomy-first 구조로 재분류했습니다.

## 방법

각 월은 `https://huggingface.co/api/daily_papers?month=YYYY-MM` 공개 API로 페이지가 빌 때까지 수집했습니다. 논문은 HF paper/arXiv id로 중복 제거했고, key idea, strengths, limitations, keyword tags는 공개 메타데이터에서 결정론적으로 생성했습니다. 유료 API, 유료 LLM, 유료 번역, 유료 compute는 사용하지 않았습니다.

## 분류별 규모

{cat_lines}

## 주목도가 높은 논문

{top_lines}

## 해석

이 아카이브는 HF Daily Papers가 공개 AI 연구의 실용적 발견 계층으로 작동한다는 점을 보여줍니다. 코드, 프로젝트 페이지, 데모, 모델 릴리스, 벤치마크를 함께 제공하는 논문일수록 재사용 가능성과 커뮤니티 가시성이 커지는 경향이 있습니다.

## 한계

이 결과물은 PDF 전문을 읽고 작성한 systematic review가 아닙니다. HF upvotes, comments, GitHub stars는 과학적 타당성이 아니라 커뮤니티 가시성의 신호입니다. 방법론적 품질과 재현성은 각 논문, 코드, 데이터셋, 평가 세부사항을 직접 확인해야 합니다.
"""
    (PAPER_DIR / "review_en.md").write_text(en_md, encoding="utf-8")
    (PAPER_DIR / "review_ko.md").write_text(ko_md, encoding="utf-8")
    write_markdown_html(PAPER_DIR / "review_en.md", PAPER_DIR / "review_en.html", "Hugging Face Daily Papers Review")
    write_markdown_html(PAPER_DIR / "review_ko.md", PAPER_DIR / "review_ko.html", "Hugging Face Daily Papers Review Korean")


def write_method_and_project_files(papers, analysis):
    method = f"""# Curation Method

## Scope

- Repository: `{OWNER}/{REPO_NAME}`
- Source: Hugging Face Daily Papers monthly pages and public API.
- Period: `{START_MONTH}` through `{END_MONTH}`.
- Included papers: every unique paper returned by the monthly API pages in that period.

## Data Source

The source endpoint is `https://huggingface.co/api/daily_papers`, using `month=YYYY-MM`, `sort=publishedAt`, `limit=100`, and incrementing `p` until an empty page is returned. This corresponds to the public monthly pages such as `https://huggingface.co/papers/month/2023-05`.

## Ranking And Taxonomy

No paper is excluded after successful monthly collection and deduplication. Ranking is a browsing order derived from HF upvotes, comments, GitHub stars recorded by HF, and availability of repository or project-page metadata. Taxonomy, keyword tags, key ideas, strengths, and limitations are deterministic metadata adapter outputs.

## GitHub-Awesome Skill2 And Paper-Curation Provenance

This repository follows `github-awesome-skill2` in metadata-adapter mode. The local `jehyunlee/paper-curation` checkout at `E:\\조선대\\연구\\paper-curation` was inspected. Full PDF review via direct `paper-curation` was not run because the requested full monthly HF archive is large and upstream full review stages require separate explicit approval for paid or metered APIs.

## Validation

Generated outputs include CSV/JSON datasets, README, README HTML companion, review Markdown files with HTML companions, a static GitHub Pages site, period analysis JSON, link audit JSON, and provenance JSON.
"""
    (PAPER_DIR / "curation_method.md").write_text(method, encoding="utf-8")
    write_markdown_html(PAPER_DIR / "curation_method.md", PAPER_DIR / "curation_method.html", "Curation Method")

    citation = f"""cff-version: 1.2.0
title: "Awesome Hugging Face Papers: HF Daily Papers Archive, {PERIOD_TEXT}"
message: "If you use this curated dataset or review draft, please cite this repository."
type: dataset
authors:
  - name: "Honggi"
repository-code: "https://github.com/{OWNER}/{REPO_NAME}"
date-released: "{date.today().isoformat()}"
license: "CC-BY-4.0"
keywords:
  - "Hugging Face"
  - "Daily Papers"
  - "artificial intelligence"
  - "machine learning"
  - "bibliometrics"
"""
    (ROOT / "CITATION.cff").write_text(citation, encoding="utf-8")
    (ROOT / "LICENSE").write_text("CC-BY-4.0 for text and metadata curation; upstream paper metadata belongs to original sources.\n", encoding="utf-8")
    (ROOT / ".gitignore").write_text("__pycache__/\n*.pyc\n.playwright-cli/\noutput/playwright/\ndata/cache/\n", encoding="utf-8")
    publish = f"""@echo off
setlocal
cd /d "%~dp0"

gh auth status
if errorlevel 1 (
  echo.
  echo GitHub login is required. Run:
  echo   gh auth login --hostname github.com --web --scopes repo
  exit /b 1
)

gh repo view {OWNER}/{REPO_NAME} >nul 2>nul
if errorlevel 1 (
  gh repo create {OWNER}/{REPO_NAME} --public --description "Awesome Hugging Face Papers: HF Daily Papers archive, {START_MONTH} to {END_MONTH}" --source . --remote origin --push
) else (
  git remote remove origin >nul 2>nul
  git remote add origin https://github.com/{OWNER}/{REPO_NAME}.git
  git push -u origin main
)
if errorlevel 1 exit /b %errorlevel%

gh api repos/{OWNER}/{REPO_NAME}/pages -X POST -f "source[branch]=main" -f "source[path]=/docs" >nul 2>nul
if errorlevel 1 (
  gh api repos/{OWNER}/{REPO_NAME}/pages -X PUT -f "source[branch]=main" -f "source[path]=/docs" >nul 2>nul
)

echo.
echo Done: https://github.com/{OWNER}/{REPO_NAME}
echo Pages: https://{OWNER}.github.io/{REPO_NAME}/
"""
    (ROOT / "publish_to_github.bat").write_text(publish, encoding="utf-8")

    provenance = {
        "skill": "github-awesome-skill2",
        "mode": "metadata-adapter",
        "repo": f"{OWNER}/{REPO_NAME}",
        "paper_curation_source": r"E:\조선대\연구\paper-curation",
        "zotero_used": False,
        "paid_or_metered_api_used": False,
        "full_pdf_review_run": False,
        "source": "Hugging Face Daily Papers public monthly API",
        "source_months": MONTHS,
        "period": PERIOD_TEXT,
        "papers": len(papers),
        "selected_dataset": f"data/{PAPERS_CSV}",
        "taxonomy_dataset": f"data/{TAXONOMY_CSV}",
        "raw_monthly_archive_index": f"data/{MONTHLY_INDEX_JSON}",
        "raw_monthly_archive_dir": "data/monthly/",
        "ranking": "HF upvotes, comments, GitHub stars, and link metadata for browsing order; no papers excluded after monthly collection.",
        "limitations": "Metadata-driven curation; full PDF LLM reviews require separate explicit approval.",
    }
    (DATA_DIR / PROVENANCE_JSON).write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_public_assets():
    for filename in (PAPERS_JSON, PAPERS_CSV, TAXONOMY_CSV, MONTHLY_INDEX_JSON, PERIOD_ANALYSIS_JSON, LINK_AUDIT_JSON, PROVENANCE_JSON):
        shutil.copyfile(DATA_DIR / filename, DOCS_DIR / "data" / filename)
    for file in sorted((DATA_DIR / "monthly").glob("*.json")):
        shutil.copyfile(file, DOCS_DIR / "data" / "monthly" / file.name)
    for filename in ("review_en.html", "review_ko.html", "curation_method.html"):
        shutil.copyfile(PAPER_DIR / filename, DOCS_DIR / "paper" / filename)


def main():
    ensure_dirs()
    papers = collect_papers()
    write_data(papers)
    analysis = period_analysis(papers)
    audit_links(papers)
    write_readme(papers, analysis)
    write_charts(papers)
    write_site(papers, analysis)
    write_reviews(papers, analysis)
    write_method_and_project_files(papers, analysis)
    copy_public_assets()
    print(f"[done] generated {len(papers):,} unique HF Daily Papers from {len(MONTHS)} monthly pages", flush=True)


if __name__ == "__main__":
    main()
