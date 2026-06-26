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

LANGUAGE_OPTIONS = [
    ("en", "English"),
    ("ko", "한국어"),
    ("zh", "中文"),
    ("ja", "日本語"),
]
LANGUAGE_CODES = [code for code, _label in LANGUAGE_OPTIONS]


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


def top_counter_items(counter, limit=3):
    return [{"name": name, "count": count} for name, count in counter.most_common(limit)]


def names_with_counts(items):
    if not items:
        return "no dominant signal"
    return ", ".join(f"{item['name']} ({item['count']:,})" for item in items)


def localized_range_label(start_month, end_month, language):
    if language == "ko":
        return f"{start_month}부터 {end_month}까지"
    if language == "zh":
        return f"{start_month} 至 {end_month}"
    if language == "ja":
        return f"{start_month}から{end_month}まで"
    return f"{start_month} to {end_month}"


def localized_movement(movement_key, language):
    phrases = {
        "expanded": {
            "en": "expanded toward the end of the selected period",
            "ko": "선택 기간 후반으로 갈수록 확대되었습니다",
            "zh": "在所选时期后段扩大",
            "ja": "選択期間の後半に向けて拡大しました",
        },
        "early": {
            "en": "was more concentrated near the beginning of the selected period",
            "ko": "선택 기간 초반에 더 집중되었습니다",
            "zh": "更集中在所选时期前段",
            "ja": "選択期間の前半により集中しました",
        },
        "steady": {
            "en": "stayed relatively steady across the selected period",
            "ko": "선택 기간 전반에 비교적 안정적으로 유지되었습니다",
            "zh": "在所选时期内保持相对稳定",
            "ja": "選択期間を通じて比較的安定していました",
        },
    }
    return phrases.get(movement_key, phrases["steady"]).get(language, phrases["steady"]["en"])


def period_research_copy(language, stats):
    range_label = localized_range_label(stats["start"], stats["end"], language)
    movement = localized_movement(stats["movement_key"], language)
    if language == "ko":
        return {
            "timelineTitle": "연구 타임라인",
            "timelineSummary": [
                f"{range_label}의 Hugging Face Daily Papers 코퍼스에는 활성 월 {stats['active_months']:,}개월에 걸친 논문 {stats['papers']:,}편이 포함됩니다. 활동은 {stats['peak_month']}에 {stats['peak_month_count']:,}편으로 정점에 도달했고, 월별 패턴은 {movement}.",
                f"가장 강한 taxonomy 신호는 {stats['top_category_text']}입니다. 선택 기간 초반은 {stats['early_category_text']} 중심이고 후반은 {stats['late_category_text']} 중심으로, HF에서 보이는 연구 관심이 기간 안에서 어떻게 이동했는지 보여줍니다.",
                f"이 기간의 HF 가시성 최상위 논문은 {stats['top_paper_category']} 범주의 \"{stats['top_paper_title']}\"({stats['top_paper_month']}, upvote {stats['top_paper_upvotes']:,}개)입니다. {stats['top_keyword_text']} 같은 빈번한 키워드는 foundation model, multimodal system, generative media, agent, evaluation, efficient AI와의 연결을 보여줍니다.",
            ],
            "insightsTitle": "연구 인사이트",
            "insights": [
                {
                    "label": "기반 계층",
                    "title": "파운데이션 모델이 공유 기반으로 남아 있습니다",
                    "body": f"{stats['top_category_text']}가 {range_label}을 주도하며, HF 연구 흐름의 어떤 부분이 downstream 작업을 위한 재사용 가능한 층이 되었는지 보여줍니다.",
                    "implication": "시사점: 기간을 비교할 때 모델 이름만 보지 말고 데이터, 모델 인터페이스, 평가, 배포 산출물을 함께 봐야 합니다.",
                },
                {
                    "label": "공개 산출물",
                    "title": "GitHub와 프로젝트 페이지가 재사용성을 좌우합니다",
                    "body": f"{stats['repo_count']:,}편은 GitHub 저장소를, {stats['project_count']:,}편은 프로젝트 페이지를 연결합니다. 구현 세부사항과 데모가 이 기간을 읽는 핵심 단서입니다.",
                    "implication": "시사점: 논문 주장과 함께 저장소 품질, 라이선스, 재현성 노트, 유지되는 데모를 확인해야 합니다.",
                },
                {
                    "label": "커뮤니티 신호",
                    "title": "HF 관심도는 눈에 띄는 연구 순간을 드러냅니다",
                    "body": f"선택 기간에는 HF upvote {stats['upvotes']:,}개와 댓글 {stats['comments']:,}개가 있으며, 월별 활동은 {stats['peak_month']}에 가장 강합니다.",
                    "implication": "시사점: HF engagement는 발견 신호이지 과학적 타당성 점수가 아니므로, 높은 신호의 논문도 전체 방법론 검토가 필요합니다.",
                },
                {
                    "label": "키워드 관례",
                    "title": "키워드 조합이 기간의 실용적 초점을 보여줍니다",
                    "body": f"{stats['top_keyword_text']} 같은 빈번한 태그는 이 기간이 모델, benchmark, multimodal, generation, agent, code, system 중 어디에 초점을 두는지 보여줍니다.",
                    "implication": "시사점: keyword filter를 사용해 넓은 커뮤니티 가시성과 좁은 연구 질문을 분리해 볼 수 있습니다.",
                },
                {
                    "label": "메타데이터 한계",
                    "title": "최신 HF 지도에는 전문가 보정이 필요합니다",
                    "body": f"이 기간 뷰는 메타데이터 기반이며 {range_label}에 수집된 모든 HF Daily Paper를 보존합니다. PDF, 코드, 데이터셋, 평가 세부사항을 읽는 일을 대체하지 않습니다.",
                    "implication": "시사점: 이 사이트는 탐색 가능한 지도층으로 사용하고, 방법론적 결론에는 도메인 전문성을 더해야 합니다.",
                },
            ],
        }
    if language == "zh":
        return {
            "timelineTitle": "研究时间线",
            "timelineSummary": [
                f"在 {range_label} 期间，Hugging Face Daily Papers 语料包含 {stats['papers']:,} 篇论文，覆盖 {stats['active_months']:,} 个活跃月份。活动在 {stats['peak_month']} 达到峰值，共 {stats['peak_month_count']:,} 篇，月度模式{movement}。",
                f"最强的 taxonomy 信号是 {stats['top_category_text']}。所选时期前段偏向 {stats['early_category_text']}，后段偏向 {stats['late_category_text']}，显示 HF 可见研究注意力如何在时间段内移动。",
                f"该时期 HF 可见度最高的论文是 {stats['top_paper_category']} 中的《{stats['top_paper_title']}》（{stats['top_paper_month']}，{stats['top_paper_upvotes']:,} upvotes）。{stats['top_keyword_text']} 等高频关键词把这一时期连接到 foundation model、multimodal system、generative media、agent、evaluation 和 efficient AI。",
            ],
            "insightsTitle": "研究洞察",
            "insights": [
                {
                    "label": "基础层",
                    "title": "基础模型仍是共享底座",
                    "body": f"{stats['top_category_text']} 主导 {range_label}，显示 HF 研究流的哪些部分成为下游工作的可复用层。",
                    "implication": "启示：比较时期时应同时看数据、模型接口、评估和部署产物，而不只是模型名称。",
                },
                {
                    "label": "开放产物",
                    "title": "GitHub 与项目页塑造复用路径",
                    "body": f"{stats['repo_count']:,} 篇论文链接 GitHub 仓库，{stats['project_count']:,} 篇链接项目页，说明实现细节和演示是理解该时期的核心线索。",
                    "implication": "启示：论文主张之外，还要检查仓库质量、许可证、复现说明和持续维护的演示。",
                },
                {
                    "label": "社区信号",
                    "title": "HF 关注度突出可见研究时刻",
                    "body": f"所选时期共有 {stats['upvotes']:,} 个 HF upvotes 和 {stats['comments']:,} 条评论，月度活动在 {stats['peak_month']} 最强。",
                    "implication": "启示：HF engagement 是发现信号，不是科学有效性评分；高信号论文仍需要完整方法审阅。",
                },
                {
                    "label": "关键词约定",
                    "title": "关键词组合揭示时期的实践重点",
                    "body": f"{stats['top_keyword_text']} 等高频标签显示该时期是围绕模型、benchmark、多模态、生成、智能体、代码还是系统展开。",
                    "implication": "启示：使用 keyword filter 可把广泛社区可见度与更窄的研究问题分开观察。",
                },
                {
                    "label": "元数据限制",
                    "title": "近期 HF 地图需要专家校正",
                    "body": f"该时期视图基于元数据，并保留 {range_label} 收集到的所有 HF Daily Paper；它不能替代阅读 PDF、代码、数据集和评估细节。",
                    "implication": "启示：把本站作为可导航地图，再为方法论结论加入领域专家判断。",
                },
            ],
        }
    if language == "ja":
        return {
            "timelineTitle": "研究タイムライン",
            "timelineSummary": [
                f"{range_label} の Hugging Face Daily Papers コーパスには、{stats['active_months']:,} のアクティブ月にわたる {stats['papers']:,} 本の論文が含まれます。活動は {stats['peak_month']} に {stats['peak_month_count']:,} 本でピークとなり、月次パターンは{movement}。",
                f"最も強い taxonomy 信号は {stats['top_category_text']} です。選択期間の前半は {stats['early_category_text']}、後半は {stats['late_category_text']} に寄っており、HF 上で見える研究関心が期間内でどう移ったかを示します。",
                f"この期間で HF 可視性が最も高い論文は {stats['top_paper_category']} の「{stats['top_paper_title']}」（{stats['top_paper_month']}、upvote {stats['top_paper_upvotes']:,} 件）です。{stats['top_keyword_text']} などの頻出キーワードは、この期間を foundation model、multimodal system、generative media、agent、evaluation、efficient AI と結びつけます。",
            ],
            "insightsTitle": "研究インサイト",
            "insights": [
                {
                    "label": "基盤レイヤー",
                    "title": "Foundation model は共有基盤であり続けます",
                    "body": f"{stats['top_category_text']} が {range_label} を主導し、HF 研究ストリームのどの部分が下流作業の再利用可能な層になったかを示します。",
                    "implication": "示唆：期間比較ではモデル名だけでなく、データ、モデルインターフェース、評価、デプロイ成果物を併せて見る必要があります。",
                },
                {
                    "label": "公開成果物",
                    "title": "GitHub とプロジェクトページが再利用を形作ります",
                    "body": f"{stats['repo_count']:,} 本は GitHub リポジトリに、{stats['project_count']:,} 本はプロジェクトページにリンクしており、実装詳細とデモがこの期間を読む中心的な手がかりです。",
                    "implication": "示唆：論文の主張と並べて、リポジトリ品質、ライセンス、再現性ノート、維持されているデモを確認すべきです。",
                },
                {
                    "label": "コミュニティ信号",
                    "title": "HF の注目は可視化された研究の節目を示します",
                    "body": f"選択期間には HF upvote {stats['upvotes']:,} 件とコメント {stats['comments']:,} 件があり、月次活動は {stats['peak_month']} が最も強くなっています。",
                    "implication": "示唆：HF engagement は発見シグナルであり科学的妥当性スコアではないため、高シグナル論文にも完全な方法レビューが必要です。",
                },
                {
                    "label": "キーワード規約",
                    "title": "キーワード構成が期間の実践的焦点を示します",
                    "body": f"{stats['top_keyword_text']} などの頻出タグは、この期間が model、benchmark、multimodal、generation、agent、code、system のどこに焦点を置くかを示します。",
                    "implication": "示唆：keyword filter を使うと、広いコミュニティ可視性と狭い研究質問を分けて確認できます。",
                },
                {
                    "label": "メタデータ限界",
                    "title": "最近の HF マップには専門家の補正が必要です",
                    "body": f"この期間ビューはメタデータベースで、{range_label} に収集されたすべての HF Daily Paper を保持します。PDF、コード、データセット、評価詳細を読むことの代替にはなりません。",
                    "implication": "示唆：このサイトをナビゲーション可能な地図として使い、方法論的な結論には分野専門性を重ねてください。",
                },
            ],
        }
    return {
        "timelineTitle": "Research Timeline",
        "timelineSummary": [
            f"For {range_label}, the Hugging Face Daily Papers corpus contains {stats['papers']:,} papers across {stats['active_months']:,} active months. Activity peaks in {stats['peak_month']} with {stats['peak_month_count']:,} papers, and the monthly pattern {movement}.",
            f"The strongest taxonomy signals are {stats['top_category_text']}. Earlier selected months lean toward {stats['early_category_text']}, while later selected months lean toward {stats['late_category_text']}, showing how HF-visible research attention shifts across the range.",
            f"The leading HF-visible paper in this period is \"{stats['top_paper_title']}\" ({stats['top_paper_month']}, {stats['top_paper_upvotes']:,} upvotes) in {stats['top_paper_category']}. Frequent keywords such as {stats['top_keyword_text']} connect the period to foundation models, multimodal systems, generative media, agents, evaluation, and efficient AI.",
        ],
        "insightsTitle": "Research Insights",
        "insights": [
            {
                "label": "Foundation Layer",
                "title": "Foundation models remain the shared substrate",
                "body": f"{stats['top_category_text']} dominate {range_label}, showing which parts of the HF research stream became reusable layers for downstream work.",
                "implication": "Implication: compare periods by data, model interface, evaluation, and deployment artifacts rather than by model names alone.",
            },
            {
                "label": "Open Artifacts",
                "title": "GitHub and project pages shape reuse",
                "body": f"{stats['repo_count']:,} papers link GitHub repositories and {stats['project_count']:,} link project pages, making implementation details and demos central to how the period is read.",
                "implication": "Implication: repository quality, licenses, reproducibility notes, and maintained demos matter alongside paper claims.",
            },
            {
                "label": "Community Signal",
                "title": "HF attention highlights visible research moments",
                "body": f"The selected range carries {stats['upvotes']:,} HF upvotes and {stats['comments']:,} comments, with the strongest monthly activity in {stats['peak_month']}.",
                "implication": "Implication: HF engagement is a discovery signal, not a scientific validity score; high-signal papers still need full-method review.",
            },
            {
                "label": "Keyword Convention",
                "title": "Keyword mix reveals the period's practical focus",
                "body": f"Frequent tags such as {stats['top_keyword_text']} show whether the period centers on models, benchmarks, multimodal work, generation, agents, code, or systems.",
                "implication": "Implication: use keyword filters to separate broad community visibility from narrower research questions.",
            },
            {
                "label": "Metadata Limits",
                "title": "Recent HF maps need expert correction",
                "body": f"The period view is metadata-driven and preserves every collected HF Daily Paper from {range_label}; it does not replace reading PDFs, code, datasets, and evaluation details.",
                "implication": "Implication: treat this site as a navigable map, then layer in domain expertise for methodological conclusions.",
            },
        ],
    }


def paper_language_copy(paper, language):
    title = paper.get("title") or "Untitled paper"
    category = paper.get("category") or "General Machine Learning and Optimization"
    month = paper.get("source_month") or ""
    tags = ", ".join((paper.get("keyword_tags") or [])[:3]) or "ai-research"
    upvotes = paper.get("upvotes", 0)
    if language == "ko":
        return f"\"{title}\"은(는) {category} 범주의 HF Daily Paper입니다. {month}에 수집되었고, {tags} 키워드와 HF upvote {upvotes:,}개 신호를 통해 해당 기간의 연구 관심을 보여줍니다."
    if language == "zh":
        return f"《{title}》是 {category} 类别的 HF Daily Paper。它在 {month} 被收集，并通过 {tags} 关键词与 {upvotes:,} 个 HF upvotes 展示该时期的研究关注。"
    if language == "ja":
        return f"「{title}」は {category} カテゴリの HF Daily Paper です。{month} に収集され、{tags} キーワードと HF upvote {upvotes:,} 件の信号から、この期間の研究関心を示します。"
    return paper.get("key_idea") or f"HF Daily Paper entry for {title}."


def period_insight_for_range(papers, start_month, end_month):
    rows = [paper for paper in papers if start_month <= paper["source_month"] <= end_month]
    months = [month for month in MONTHS if start_month <= month <= end_month]
    month_counts = Counter(p["source_month"] for p in rows)
    category_counts = Counter(p["category"] for p in rows)
    keyword_counts = Counter(tag for p in rows for tag in p["keyword_tags"])
    peak_month, peak_month_count = max(((month, month_counts.get(month, 0)) for month in months), key=lambda item: (item[1], item[0]), default=(start_month, 0))
    active_months = sum(1 for month in months if month_counts.get(month, 0))
    midpoint = max(1, len(months) // 2)
    early_months = set(months[:midpoint])
    late_months = set(months[midpoint:])
    early_category = Counter(p["category"] for p in rows if p["source_month"] in early_months).most_common(1)
    late_category = Counter(p["category"] for p in rows if p["source_month"] in late_months).most_common(1)
    first_count = month_counts.get(months[0], 0) if months else 0
    last_count = month_counts.get(months[-1], 0) if months else 0
    if last_count > first_count * 1.25:
        movement_key = "expanded"
    elif first_count > last_count * 1.25:
        movement_key = "early"
    else:
        movement_key = "steady"
    movement = localized_movement(movement_key, "en")
    top_categories = top_counter_items(category_counts, 3)
    top_keywords = top_counter_items(keyword_counts, 5)
    top_papers = sorted(rows, key=lambda p: (-p["upvotes"], -p["github_stars"], p["rank"]))[:5]
    top_paper = top_papers[0] if top_papers else {}
    repo_count = sum(1 for p in rows if p["github_repo"])
    project_count = sum(1 for p in rows if p["project_page"])
    upvotes = sum(p["upvotes"] for p in rows)
    comments = sum(p["num_comments"] for p in rows)
    range_label = f"{start_month} to {end_month}"
    top_category_text = names_with_counts(top_categories)
    top_keyword_text = names_with_counts(top_keywords[:3])
    early_category_text = early_category[0][0] if early_category else "no dominant taxonomy"
    late_category_text = late_category[0][0] if late_category else early_category_text
    top_paper_title = top_paper.get("title", "No paper")
    top_paper_month = top_paper.get("source_month", start_month)
    top_paper_category = top_paper.get("category", "no taxonomy")
    top_paper_upvotes = top_paper.get("upvotes", 0)
    translation_stats = {
        "start": start_month,
        "end": end_month,
        "papers": len(rows),
        "active_months": active_months,
        "peak_month": peak_month,
        "peak_month_count": peak_month_count,
        "movement_key": movement_key,
        "top_category_text": top_category_text,
        "top_keyword_text": top_keyword_text,
        "early_category_text": early_category_text,
        "late_category_text": late_category_text,
        "top_paper_title": top_paper_title,
        "top_paper_month": top_paper_month,
        "top_paper_category": top_paper_category,
        "top_paper_upvotes": top_paper_upvotes,
        "repo_count": repo_count,
        "project_count": project_count,
        "upvotes": upvotes,
        "comments": comments,
    }
    translations = {language: period_research_copy(language, translation_stats) for language in LANGUAGE_CODES}
    return {
        "key": f"{start_month}..{end_month}",
        "start": start_month,
        "end": end_month,
        "range": range_label,
        "papers": len(rows),
        "activeMonths": active_months,
        "peakMonth": peak_month,
        "peakMonthCount": peak_month_count,
        "firstMonthCount": first_count,
        "lastMonthCount": last_count,
        "movement": movement,
        "topCategories": top_categories,
        "topKeywords": top_keywords,
        "repoCount": repo_count,
        "projectCount": project_count,
        "upvotes": upvotes,
        "comments": comments,
        "topPapers": [
            {
                "rank": p["rank"],
                "id": p["id"],
                "title": p["title"],
                "month": p["source_month"],
                "category": p["category"],
                "upvotes": p["upvotes"],
                "githubStars": p["github_stars"],
                "hfUrl": p["hf_url"],
            }
            for p in top_papers
        ],
        "timeline": {
            "title": translations["en"]["timelineTitle"],
            "summary": translations["en"]["timelineSummary"],
        },
        "insights": translations["en"]["insights"],
        "translations": translations,
    }


def build_period_insights(papers):
    insights = {}
    for start_index, start_month in enumerate(MONTHS):
        for end_month in MONTHS[start_index:]:
            item = period_insight_for_range(papers, start_month, end_month)
            insights[item["key"]] = item
    return insights


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
        "periodInsights": build_period_insights(papers),
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
            "copy": {language: paper_language_copy(paper, language) for language in LANGUAGE_CODES},
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
    language_options = "\n".join(f'<option value="{code}">{html.escape(label)}</option>' for code, label in LANGUAGE_OPTIONS)
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
    category_meta = {
        item["name"]: {
            "color": item["color"],
            "trends": item.get("trends", []),
            "limitations": item.get("limitations", []),
        }
        for item in CATEGORY_RULES
    }
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
    category_meta_json = json.dumps(category_meta, ensure_ascii=False)
    period_insights_payload = json.dumps(analysis["periodInsights"], ensure_ascii=False, separators=(",", ":"))
    safe_period_insights = period_insights_payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")

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
    .control-grid {{ display: grid; grid-template-columns: minmax(220px, 1.5fr) repeat(6, minmax(124px, 1fr)) minmax(140px, .8fr); gap: 10px; }}
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
    .timeline-narrative {{ border: 1px solid var(--line); border-radius: 6px; background: var(--soft); padding: 12px 16px; margin-top: 10px; color: var(--muted); }}
    .timeline-narrative h3 {{ margin: 0 0 8px; font-size: 16px; color: var(--ink); }}
    .timeline-narrative ul {{ margin: 0; padding-left: 20px; }}
    .timeline-narrative li {{ margin: 5px 0; }}
    .timeline-copy p {{ margin: 8px 0; }}
    .insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .insight-card {{ border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 12px; }}
    .insight-card strong {{ display: block; font-size: 24px; line-height: 1.15; }}
    .insight-card span {{ display: block; color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .period-insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-bottom: 10px; }}
    .insight-box {{ border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 12px; }}
    .insight-label {{ color: var(--accent); font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .03em; }}
    .insight-box h3 {{ margin: 6px 0 8px; font-size: 16px; line-height: 1.25; }}
    .insight-box p {{ margin: 6px 0; color: var(--muted); font-size: 13px; }}
    .insight-implication {{ color: var(--ink); font-weight: 700; }}
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
    .taxonomy-list {{ display: grid; gap: 16px; }}
    .taxonomy-total-summary {{ color: var(--muted); margin: -2px 0 12px; }}
    .taxonomy-total-summary strong {{ color: var(--ink); }}
    .taxonomy-section[hidden], .paper-card[hidden] {{ display: none !important; }}
    .taxonomy-section {{ margin-top: 0; }}
    .taxonomy-section > details {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .taxonomy-section summary {{ cursor: pointer; list-style: none; padding: 14px 18px; display: grid; grid-template-columns: 64px minmax(260px, 1fr) repeat(3, minmax(112px, auto)); gap: 12px; align-items: center; font-weight: 700; }}
    .taxonomy-section summary::-webkit-details-marker {{ display: none; }}
    .summary-thumb, .all-taxonomy-thumb {{ width: 56px; height: 40px; border: 1px solid var(--line); border-radius: 6px; background: var(--thumb-color, #eef5f3); color: #fff; display: inline-flex; align-items: center; justify-content: center; font-weight: 800; font-size: 13px; }}
    .all-taxonomy-thumb {{ background: var(--accent); }}
    .summary-title {{ color: var(--accent); min-width: 0; }}
    .category-count, .category-years, .category-citations {{ color: var(--muted); font-size: 13px; white-space: nowrap; }}
    .category-count {{ color: var(--accent); font-weight: 800; }}
    .section-intro {{ padding: 0 18px 14px; border-top: 1px solid var(--line); }}
    .section-intro p {{ margin: 10px 0; color: var(--muted); }}
    .section-intro strong {{ color: var(--ink); }}
    .category-insight-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; }}
    .category-insight-box {{ padding: 12px 14px; background: #f4faf8; border: 1px solid #cfe7df; border-radius: 8px; }}
    .category-insight-box.limitation-box {{ background: #fff8f1; border-color: #ead7c1; }}
    .category-insight-box strong {{ display: block; margin-bottom: 6px; }}
    .category-insight-box ul {{ margin: 0; padding-left: 20px; color: var(--muted); line-height: 1.55; }}
    .taxonomy-body {{ border-top: 1px solid var(--line); background: #f9fbfd; }}
    .paper-list {{ display: grid; gap: 12px; padding: 16px; background: #f9fbfd; }}
    .paper-card {{ border: 1px solid var(--line); border-radius: 8px; background: #fff; overflow: hidden; display: grid; grid-template-columns: 58px 1fr; gap: 14px; padding: 14px; }}
    .paper-rank {{ width: 44px; height: 44px; display: inline-grid; place-items: center; border-radius: 8px; background: var(--soft); color: var(--accent); font-weight: 900; }}
    .paper-body {{ min-width: 0; }}
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
    .load-more {{ margin-top: 12px; border: 1px solid var(--accent); border-radius: 6px; background: #fff; color: var(--accent); font: inherit; font-weight: 800; padding: 8px 10px; cursor: pointer; }}
    @media (max-width: 860px) {{
      .stats {{ grid-template-columns: repeat(2, 1fr); }}
      .control-grid {{ grid-template-columns: 1fr 1fr; }}
      .figures {{ grid-template-columns: 1fr; }}
      .taxonomy-section summary {{ grid-template-columns: 56px 1fr; }}
      .category-count, .category-years, .category-citations {{ white-space: normal; }}
    }}
    @media (max-width: 560px) {{
      .wrap, main {{ padding: 16px; }}
      .stats, .control-grid, .paper-list {{ grid-template-columns: 1fr; }}
      .paper-card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="topbar">
        <div>
          <h1>Awesome Hugging Face Papers</h1>
          <p class="subtitle" id="siteSubtitle">A taxonomy-first archive of Hugging Face Daily Papers from {PERIOD_TEXT}. Every paper in the monthly pages is kept in the dataset and can be filtered below.</p>
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
      <select id="language" aria-label="Language">{language_options}</select>
      <label class="check"><input id="hasRepo" type="checkbox"> <span id="githubOnlyText">GitHub only</span></label>
    </div>
  </section>
  <main>
    <section class="figures" aria-label="Charts">
      <figure>
        <img src="assets/category_distribution.svg" alt="Category distribution">
        <figcaption id="categoryFigureCaption">Taxonomy-first distribution over the full HF Daily Papers archive.</figcaption>
      </figure>
      <figure>
        <img src="assets/monthly_coverage.svg" alt="Monthly paper counts">
        <figcaption id="monthlyFigureCaption">Monthly coverage from {START_MONTH} through {END_MONTH}.</figcaption>
      </figure>
    </section>
    <section class="analysis-section" aria-labelledby="timelineTitle">
      <div class="section-head">
        <h2 id="timelineTitle">Research Timeline</h2>
        <p id="timelineSummary"></p>
      </div>
      <div id="timelineChart" class="timeline-chart"></div>
      <div id="timelineNarrative" class="timeline-narrative"></div>
    </section>
    <section class="analysis-section" aria-labelledby="insightsTitle">
      <div class="section-head">
        <h2 id="insightsTitle">Research Insights</h2>
        <p id="insightsLead">Period-specific insight cards are generated from the selected month range; the numeric cards below still update with taxonomy, keyword, search, and repository-link filters.</p>
      </div>
      <div id="periodInsights" class="period-insights"></div>
      <div id="insights" class="insight-grid"></div>
      <div id="topPapers" class="insight-list"></div>
    </section>
    <section class="analysis-section" id="keywords-convention" aria-labelledby="keywordsTitle">
      <div class="section-head">
        <h2 id="keywordsTitle">Keywords Convention</h2>
        <p id="keywordsLead">Click a keyword card or use the keyword selector to inspect papers grouped by the collection's keyword convention.</p>
      </div>
      <div class="keyword-grid">{keyword_cards}</div>
      <p class="keyword-filter-status" id="keywordFilterStatus"></p>
    </section>
    <div class="result-line">
      <strong id="resultCount"></strong>
      <span id="activeFilters"></span>
    </div>
    <section class="analysis-section" aria-labelledby="taxonomyBrowserTitle">
      <div class="section-head">
        <h2 id="taxonomyBrowserTitle">Taxonomy Papers</h2>
        <p id="taxonomyLead">Open or close each taxonomy to inspect the matching papers for the current period and filters.</p>
      </div>
      <p id="taxonomyTotalSummary" class="taxonomy-total-summary"></p>
      <section id="papers" class="taxonomy-list" aria-live="polite"></section>
    </section>
  </main>
  <script id="paper-data" type="application/json">{safe_payload}</script>
  <script id="period-insights-data" type="application/json">{safe_period_insights}</script>
  <script>
    const papers = JSON.parse(document.getElementById('paper-data').textContent);
    const periodInsights = JSON.parse(document.getElementById('period-insights-data').textContent);
    const colors = {colors_json};
    const months = {months_json};
    const keywordDescriptions = {keyword_descriptions_json};
    const categoryMeta = {category_meta_json};
    const els = {{
      q: document.getElementById('q'),
      category: document.getElementById('category'),
      periodPreset: document.getElementById('periodPreset'),
      fromMonth: document.getElementById('fromMonth'),
      toMonth: document.getElementById('toMonth'),
      keyword: document.getElementById('keyword'),
      language: document.getElementById('language'),
      hasRepo: document.getElementById('hasRepo'),
      siteSubtitle: document.getElementById('siteSubtitle'),
      githubOnlyText: document.getElementById('githubOnlyText'),
      categoryFigureCaption: document.getElementById('categoryFigureCaption'),
      monthlyFigureCaption: document.getElementById('monthlyFigureCaption'),
      list: document.getElementById('papers'),
      count: document.getElementById('resultCount'),
      filters: document.getElementById('activeFilters'),
      timeline: document.getElementById('timelineChart'),
      timelineSummary: document.getElementById('timelineSummary'),
      timelineNarrative: document.getElementById('timelineNarrative'),
      periodInsights: document.getElementById('periodInsights'),
      insights: document.getElementById('insights'),
      topPapers: document.getElementById('topPapers'),
      timelineTitle: document.getElementById('timelineTitle'),
      insightsTitle: document.getElementById('insightsTitle'),
      insightsLead: document.getElementById('insightsLead'),
      keywordsTitle: document.getElementById('keywordsTitle'),
      keywordsLead: document.getElementById('keywordsLead'),
      taxonomyBrowserTitle: document.getElementById('taxonomyBrowserTitle'),
      taxonomyLead: document.getElementById('taxonomyLead'),
      taxonomyTotalSummary: document.getElementById('taxonomyTotalSummary'),
      keywordStatus: document.getElementById('keywordFilterStatus')
    }};
    const uiCopy = {{
      en: {{
        siteSubtitle: "A taxonomy-first archive of Hugging Face Daily Papers from {{period}}. Every paper in the monthly pages is kept in the dataset and can be filtered below.",
        searchPlaceholder: "Search title, authors, summaries, tags",
        githubOnly: "GitHub only",
        categoryFigureCaption: "Taxonomy-first distribution over the full HF Daily Papers archive.",
        monthlyFigureCaption: "Monthly coverage from {{start}} through {{end}}.",
        timelineTitle: "Research Timeline",
        timelineSummary: "{{from}} to {{to}} | {{count}} matching papers across {{months}} months",
        insightsTitle: "Research Insights",
        insightsLead: "Period-specific insight cards are generated from the selected month range; the numeric cards below still update with taxonomy, keyword, search, and repository-link filters.",
        keywordsTitle: "Keywords Convention",
        keywordsLead: "Click a keyword card or use the keyword selector to inspect papers grouped by the collection's keyword convention.",
        taxonomyBrowserTitle: "Taxonomy Papers",
        taxonomyLead: "Open or close each taxonomy to inspect the matching papers for the current period and filters.",
        totalSelected: "Total selected papers",
        categoryCount: "Categories",
        categories: "categories",
        representativeEmphasis: "Representative emphasis",
        topRankedPaper: "Top-ranked paper",
        categoryOverview: "Category Overview",
        limitationsTitle: "Limitations",
        hfSignal: "HF upvotes",
        filteredMovementTitle: "Filtered stream movement",
        activityNote: "Activity starts at {{firstActive}} with {{firstCount}} matching papers and reaches {{lastActive}} with {{lastCount}} papers, so this filtered stream {{trend}}.",
        busiestNote: "The busiest month in this view is {{peakMonth}} with {{peakCount}} papers, marking the strongest HF Daily Papers visibility signal for the selected filters.",
        taxonomyShiftNote: "Earlier selected months lean toward {{earlyCategory}}; later selected months lean toward {{recentCategory}}. This describes metadata-visible community attention, not a full-PDF scientific judgement.",
        trendNoActivity: "has no visible activity in the selected range",
        trendExpanded: "expands toward the end of the selected range",
        trendEarly: "is heavier near the beginning of the selected range",
        trendSteady: "stays relatively steady across the selected range",
        matchingPapers: "Matching papers",
        currentFilteredCorpus: "Current filtered corpus",
        topTaxonomy: "Top taxonomy",
        topKeyword: "Top keyword",
        githubLinked: "GitHub linked",
        projectPages: "{{count}} project pages",
        hfUpvotes: "HF upvotes",
        averagePerPaper: "{{count}} average per paper",
        hfComments: "HF comments",
        discussionSignals: "Discussion signals captured from HF",
        mostVisibleTitle: "Most visible papers in this view",
        noPapers: "No papers match the current filters.",
        selectedKeyword: "Selected keyword: {{description}} | Matching papers: {{count}}",
        allKeywordConventions: "all keyword conventions",
        showNext: "Show next {{count}} papers",
        emptyTaxonomy: "No papers match this taxonomy under the current filters.",
        emptyTaxonomyList: "No taxonomy contains papers matching the current filters.",
        topKeywords: "Top keywords",
        githubLinkedPapers: "GitHub-linked papers",
        noKeywords: "no keywords",
        paperCount: "{{count}} papers",
        allKeywords: "all keywords",
        allLinks: "all links",
        searchActive: "search active",
        allTaxonomies: "All Taxonomies",
        allKeywordsOption: "All Keywords",
        customRange: "Custom range",
        allMonths: "All months ({{start}} to {{end}})",
        unknownAuthors: "Unknown authors"
      }},
      ko: {{
        siteSubtitle: "{{period}} 기간의 Hugging Face Daily Papers를 taxonomy-first 방식으로 정리한 아카이브입니다. 월별 페이지의 모든 논문을 보존하고 아래에서 필터링할 수 있습니다.",
        searchPlaceholder: "제목, 저자, 요약, 태그 검색",
        githubOnly: "GitHub 연결만",
        categoryFigureCaption: "전체 HF Daily Papers 아카이브의 taxonomy-first 분포입니다.",
        monthlyFigureCaption: "{{start}}부터 {{end}}까지의 월별 수집 범위입니다.",
        timelineTitle: "연구 타임라인",
        timelineSummary: "{{from}}부터 {{to}}까지 | 일치 논문 {{count}}편 | {{months}}개월",
        insightsTitle: "연구 인사이트",
        insightsLead: "선택한 월 범위에 맞춘 인사이트 카드가 표시됩니다. 아래 숫자 카드는 taxonomy, keyword, search, repository-link 필터에 따라 계속 갱신됩니다.",
        keywordsTitle: "키워드 관례",
        keywordsLead: "키워드 카드를 클릭하거나 keyword selector를 사용해 이 컬렉션의 keyword convention별 논문을 확인할 수 있습니다.",
        taxonomyBrowserTitle: "Taxonomy별 논문",
        taxonomyLead: "각 taxonomy를 열거나 닫아 현재 기간과 필터에 맞는 논문을 확인하세요.",
        totalSelected: "선택 논문 합계",
        categoryCount: "카테고리",
        categories: "개 카테고리",
        representativeEmphasis: "대표 초점",
        topRankedPaper: "상위 논문",
        categoryOverview: "카테고리 개요",
        limitationsTitle: "한계",
        hfSignal: "HF upvote",
        filteredMovementTitle: "필터된 흐름 변화",
        activityNote: "활동은 {{firstActive}}에 일치 논문 {{firstCount}}편으로 시작해 {{lastActive}}에 {{lastCount}}편에 도달하므로, 이 필터된 흐름은 {{trend}}.",
        busiestNote: "이 보기에서 가장 바쁜 달은 {{peakMonth}}이며 논문 {{peakCount}}편으로, 선택 필터의 가장 강한 HF Daily Papers 가시성 신호입니다.",
        taxonomyShiftNote: "선택 기간 초반은 {{earlyCategory}} 쪽으로, 후반은 {{recentCategory}} 쪽으로 기울어집니다. 이는 전체 PDF 과학 평가가 아니라 메타데이터로 보이는 커뮤니티 관심입니다.",
        trendNoActivity: "선택 범위에서 보이는 활동이 없습니다",
        trendExpanded: "선택 범위 후반으로 갈수록 확대됩니다",
        trendEarly: "선택 범위 초반에 더 집중됩니다",
        trendSteady: "선택 범위 전반에 비교적 안정적입니다",
        matchingPapers: "일치 논문",
        currentFilteredCorpus: "현재 필터된 코퍼스",
        topTaxonomy: "상위 taxonomy",
        topKeyword: "상위 keyword",
        githubLinked: "GitHub 연결",
        projectPages: "프로젝트 페이지 {{count}}개",
        hfUpvotes: "HF upvote",
        averagePerPaper: "논문당 평균 {{count}}",
        hfComments: "HF 댓글",
        discussionSignals: "HF에서 수집된 토론 신호",
        mostVisibleTitle: "이 보기에서 가장 눈에 띄는 논문",
        noPapers: "현재 필터와 일치하는 논문이 없습니다.",
        selectedKeyword: "선택 keyword: {{description}} | 일치 논문: {{count}}편",
        allKeywordConventions: "모든 keyword convention",
        showNext: "다음 논문 {{count}}편 보기",
        emptyTaxonomy: "현재 필터에서 이 taxonomy와 일치하는 논문이 없습니다.",
        emptyTaxonomyList: "현재 필터와 일치하는 논문을 가진 taxonomy가 없습니다.",
        topKeywords: "상위 keyword",
        githubLinkedPapers: "GitHub 연결 논문",
        noKeywords: "keyword 없음",
        paperCount: "논문 {{count}}편",
        allKeywords: "모든 keyword",
        allLinks: "모든 링크",
        searchActive: "검색 적용",
        allTaxonomies: "모든 taxonomy",
        allKeywordsOption: "모든 keyword",
        customRange: "직접 기간 설정",
        allMonths: "전체 기간 ({{start}}부터 {{end}}까지)",
        unknownAuthors: "저자 정보 없음"
      }},
      zh: {{
        siteSubtitle: "{{period}} 的 Hugging Face Daily Papers taxonomy-first 档案。月度页面中的每篇论文都保留在数据集中，并可在下方筛选。",
        searchPlaceholder: "搜索标题、作者、摘要、标签",
        githubOnly: "仅 GitHub 链接",
        categoryFigureCaption: "完整 HF Daily Papers 档案的 taxonomy-first 分布。",
        monthlyFigureCaption: "{{start}} 至 {{end}} 的月度覆盖。",
        timelineTitle: "研究时间线",
        timelineSummary: "{{from}} 至 {{to}} | {{count}} 篇匹配论文 | {{months}} 个月",
        insightsTitle: "研究洞察",
        insightsLead: "洞察卡片按所选月份范围生成；下方数值卡片仍会随 taxonomy、keyword、search 与 repository-link 筛选更新。",
        keywordsTitle: "关键词约定",
        keywordsLead: "点击关键词卡片或使用关键词选择器，按本集合的 keyword convention 查看论文。",
        taxonomyBrowserTitle: "Taxonomy 论文",
        taxonomyLead: "展开或收起每个 taxonomy，查看当前时期和筛选条件下的匹配论文。",
        totalSelected: "入选论文总数",
        categoryCount: "类别",
        categories: "个类别",
        representativeEmphasis: "代表性重点",
        topRankedPaper: "排名最高论文",
        categoryOverview: "类别概览",
        limitationsTitle: "局限性",
        hfSignal: "HF upvotes",
        filteredMovementTitle: "筛选流变化",
        activityNote: "活动从 {{firstActive}} 的 {{firstCount}} 篇匹配论文开始，到 {{lastActive}} 达到 {{lastCount}} 篇，因此此筛选流{{trend}}。",
        busiestNote: "此视图中最活跃的月份是 {{peakMonth}}，共有 {{peakCount}} 篇论文，是所选筛选条件下最强的 HF Daily Papers 可见度信号。",
        taxonomyShiftNote: "所选月份前段偏向 {{earlyCategory}}；后段偏向 {{recentCategory}}。这描述的是元数据可见的社区关注，而非完整 PDF 科学判断。",
        trendNoActivity: "在所选范围内没有可见活动",
        trendExpanded: "在所选范围后段扩大",
        trendEarly: "更集中在所选范围前段",
        trendSteady: "在所选范围内相对稳定",
        matchingPapers: "匹配论文",
        currentFilteredCorpus: "当前筛选语料",
        topTaxonomy: "最高 taxonomy",
        topKeyword: "最高 keyword",
        githubLinked: "GitHub 链接",
        projectPages: "{{count}} 个项目页",
        hfUpvotes: "HF upvotes",
        averagePerPaper: "每篇平均 {{count}}",
        hfComments: "HF 评论",
        discussionSignals: "从 HF 捕获的讨论信号",
        mostVisibleTitle: "此视图中最可见的论文",
        noPapers: "没有论文匹配当前筛选条件。",
        selectedKeyword: "已选 keyword：{{description}} | 匹配论文：{{count}} 篇",
        allKeywordConventions: "全部 keyword convention",
        showNext: "显示接下来的 {{count}} 篇论文",
        emptyTaxonomy: "当前筛选下此 taxonomy 没有匹配论文。",
        emptyTaxonomyList: "没有 taxonomy 包含匹配当前筛选的论文。",
        topKeywords: "高频 keyword",
        githubLinkedPapers: "GitHub 链接论文",
        noKeywords: "无 keyword",
        paperCount: "{{count}} 篇论文",
        allKeywords: "全部 keyword",
        allLinks: "全部链接",
        searchActive: "搜索已启用",
        allTaxonomies: "全部 taxonomy",
        allKeywordsOption: "全部 keyword",
        customRange: "自定义范围",
        allMonths: "全部月份（{{start}} 至 {{end}}）",
        unknownAuthors: "作者未知"
      }},
      ja: {{
        siteSubtitle: "{{period}} の Hugging Face Daily Papers を taxonomy-first で整理したアーカイブです。月別ページのすべての論文をデータセットに保持し、下でフィルタできます。",
        searchPlaceholder: "タイトル、著者、要約、タグを検索",
        githubOnly: "GitHub ありのみ",
        categoryFigureCaption: "HF Daily Papers 全体アーカイブの taxonomy-first 分布です。",
        monthlyFigureCaption: "{{start}} から {{end}} までの月次カバレッジです。",
        timelineTitle: "研究タイムライン",
        timelineSummary: "{{from}}から{{to}}まで | 一致論文 {{count}} 本 | {{months}} か月",
        insightsTitle: "研究インサイト",
        insightsLead: "選択した月範囲に合わせたインサイトカードを表示します。下の数値カードは taxonomy、keyword、search、repository-link フィルタに応じて更新されます。",
        keywordsTitle: "キーワード規約",
        keywordsLead: "キーワードカードをクリックするか keyword selector を使って、このコレクションの keyword convention 別に論文を確認できます。",
        taxonomyBrowserTitle: "Taxonomy 別論文",
        taxonomyLead: "各 taxonomy を開閉して、現在の期間とフィルタに一致する論文を確認してください。",
        totalSelected: "選択論文合計",
        categoryCount: "カテゴリ",
        categories: "カテゴリ",
        representativeEmphasis: "代表的な焦点",
        topRankedPaper: "上位論文",
        categoryOverview: "カテゴリ概要",
        limitationsTitle: "限界",
        hfSignal: "HF upvotes",
        filteredMovementTitle: "フィルタ後の流れ",
        activityNote: "活動は {{firstActive}} の一致論文 {{firstCount}} 本から始まり、{{lastActive}} で {{lastCount}} 本に達するため、このフィルタ後の流れは{{trend}}。",
        busiestNote: "このビューで最も活発な月は {{peakMonth}} で {{peakCount}} 本の論文があり、選択フィルタで最も強い HF Daily Papers 可視性シグナルです。",
        taxonomyShiftNote: "選択期間の前半は {{earlyCategory}}、後半は {{recentCategory}} に寄っています。これは完全な PDF 科学評価ではなく、メタデータで見えるコミュニティ関心を表します。",
        trendNoActivity: "選択範囲で見える活動がありません",
        trendExpanded: "選択範囲の後半に向けて拡大します",
        trendEarly: "選択範囲の前半により集中します",
        trendSteady: "選択範囲全体で比較的安定しています",
        matchingPapers: "一致論文",
        currentFilteredCorpus: "現在のフィルタ済みコーパス",
        topTaxonomy: "上位 taxonomy",
        topKeyword: "上位 keyword",
        githubLinked: "GitHub 連携",
        projectPages: "プロジェクトページ {{count}} 件",
        hfUpvotes: "HF upvotes",
        averagePerPaper: "論文あたり平均 {{count}}",
        hfComments: "HF コメント",
        discussionSignals: "HF から取得した議論シグナル",
        mostVisibleTitle: "このビューで最も目立つ論文",
        noPapers: "現在のフィルタに一致する論文はありません。",
        selectedKeyword: "選択 keyword: {{description}} | 一致論文: {{count}} 本",
        allKeywordConventions: "すべての keyword convention",
        showNext: "次の {{count}} 本の論文を表示",
        emptyTaxonomy: "現在のフィルタでは、この taxonomy に一致する論文がありません。",
        emptyTaxonomyList: "現在のフィルタに一致する論文を含む taxonomy はありません。",
        topKeywords: "上位 keyword",
        githubLinkedPapers: "GitHub 連携論文",
        noKeywords: "keyword なし",
        paperCount: "論文 {{count}} 本",
        allKeywords: "すべての keyword",
        allLinks: "すべてのリンク",
        searchActive: "検索適用中",
        allTaxonomies: "すべての taxonomy",
        allKeywordsOption: "すべての keyword",
        customRange: "カスタム範囲",
        allMonths: "全期間（{{start}}から{{end}}まで）",
        unknownAuthors: "著者不明"
      }}
    }};
    function currentLanguage() {{
      return uiCopy[els.language.value] ? els.language.value : 'en';
    }}
    function t(key) {{
      const language = currentLanguage();
      return (uiCopy[language] && uiCopy[language][key]) || uiCopy.en[key] || key;
    }}
    function formatText(template, values) {{
      return String(template || '').replace(/\{{(\w+)\}}/g, (_match, key) => Object.prototype.hasOwnProperty.call(values, key) ? values[key] : '');
    }}
    function updateOptionText(select, value, label) {{
      const option = Array.from(select.options).find(item => item.value === value);
      if (option) option.textContent = label;
    }}
    function updateLanguageText() {{
      const values = {{period: '{PERIOD_TEXT}', start: '{START_MONTH}', end: '{END_MONTH}'}};
      document.documentElement.lang = currentLanguage();
      els.siteSubtitle.textContent = formatText(t('siteSubtitle'), values);
      els.q.placeholder = t('searchPlaceholder');
      els.githubOnlyText.textContent = t('githubOnly');
      els.categoryFigureCaption.textContent = formatText(t('categoryFigureCaption'), values);
      els.monthlyFigureCaption.textContent = formatText(t('monthlyFigureCaption'), values);
      els.timelineTitle.textContent = t('timelineTitle');
      els.insightsTitle.textContent = t('insightsTitle');
      els.insightsLead.textContent = t('insightsLead');
      els.keywordsTitle.textContent = t('keywordsTitle');
      els.keywordsLead.textContent = t('keywordsLead');
      els.taxonomyBrowserTitle.textContent = t('taxonomyBrowserTitle');
      els.taxonomyLead.textContent = t('taxonomyLead');
      updateOptionText(els.category, 'All Taxonomies', t('allTaxonomies'));
      updateOptionText(els.keyword, 'All Keywords', t('allKeywordsOption'));
      updateOptionText(els.periodPreset, 'custom', t('customRange'));
      updateOptionText(els.periodPreset, 'all', formatText(t('allMonths'), values));
    }}
    els.fromMonth.value = months[0];
    els.toMonth.value = months[months.length - 1];
    const taxonomyOrder = Array.from(els.category.options).map(option => option.value).filter(value => value !== 'All Taxonomies');
    const taxonomyBatchSize = 120;
    let taxonomyRowsByCategory = new Map();
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
      const signals = `${{p.upvotes}} ${{t('hfUpvotes')}} | ${{p.comments}} ${{t('hfComments')}}${{p.stars ? ' | ' + p.stars.toLocaleString() + ' stars' : ''}}`;
      const links = [link('HF', p.hf), link('arXiv', p.arxiv), link('Code', p.repo), link('Project', p.project)].filter(Boolean).join('');
      return `<article class="paper-card">
        <div class="paper-rank">#${{p.rank}}</div>
        <div class="paper-body">
          <div class="paper-meta"><span>${{p.month}}</span><span>${{p.category}}</span></div>
          <h3><a href="${{p.hf}}" target="_blank" rel="noopener">${{escapeHtml(p.title)}}</a></h3>
          <p class="authors">${{escapeHtml(p.authors || t('unknownAuthors'))}}</p>
          <div class="chips">${{chips}}</div>
          <p class="summary">${{escapeHtml(localizedPaperSummary(p))}}</p>
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
    function periodInsightForBounds(bounds) {{
      return periodInsights[`${{bounds.from}}..${{bounds.to}}`] || null;
    }}
    function translatedPeriodBrief(periodBrief) {{
      if (!periodBrief) return null;
      const language = currentLanguage();
      return (periodBrief.translations && (periodBrief.translations[language] || periodBrief.translations.en)) || {{
        timelineTitle: periodBrief.timeline?.title || t('timelineTitle'),
        timelineSummary: periodBrief.timeline?.summary || [],
        insightsTitle: t('insightsTitle'),
        insights: periodBrief.insights || []
      }};
    }}
    function localizedPaperSummary(p) {{
      const language = currentLanguage();
      return (p.copy && (p.copy[language] || p.copy.en)) || p.summary || '';
    }}
    function rangeMonths(from, to) {{
      return months.filter(month => month >= from && month <= to);
    }}
    function topEntries(map, limit) {{
      return Array.from(map.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, limit);
    }}
    function topCategoryFor(rows) {{
      const counts = new Map();
      rows.forEach(p => counts.set(p.category, (counts.get(p.category) || 0) + 1));
      return topEntries(counts, 1)[0] || ['none', 0];
    }}
    function trendLabel(firstCount, lastCount) {{
      if (!firstCount && !lastCount) return t('trendNoActivity');
      if (lastCount > firstCount * 1.25) return t('trendExpanded');
      if (firstCount > lastCount * 1.25) return t('trendEarly');
      return t('trendSteady');
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
      const periodBrief = periodInsightForBounds(bounds);
      const translatedBrief = translatedPeriodBrief(periodBrief);
      const visibleMonths = rangeMonths(bounds.from, bounds.to);
      const counts = new Map(visibleMonths.map(month => [month, 0]));
      filtered.forEach(p => counts.set(p.month, (counts.get(p.month) || 0) + 1));
      const maxCount = Math.max(1, ...Array.from(counts.values()));
      els.timelineTitle.textContent = translatedBrief?.timelineTitle || t('timelineTitle');
      els.timelineSummary.textContent = formatText(t('timelineSummary'), {{
        from: bounds.from,
        to: bounds.to,
        count: filtered.length.toLocaleString(),
        months: visibleMonths.length.toLocaleString()
      }});
      els.timeline.innerHTML = visibleMonths.map(month => {{
        const count = counts.get(month) || 0;
        const width = count ? Math.max(2, Math.round((count / maxCount) * 100)) : 0;
        return `<div class="timeline-row">
          <strong>${{month}}</strong>
          <div class="timeline-track" aria-label="${{month}} ${{count}} papers"><div class="timeline-bar" style="width:${{width}}%"></div></div>
          <span>${{count.toLocaleString()}}</span>
        </div>`;
      }}).join('');
      const activeMonths = visibleMonths.filter(month => (counts.get(month) || 0) > 0);
      const firstActive = activeMonths[0] || bounds.from;
      const lastActive = activeMonths[activeMonths.length - 1] || bounds.to;
      const peak = visibleMonths.map(month => [month, counts.get(month) || 0]).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0] || [bounds.from, 0];
      const midpoint = Math.floor(visibleMonths.length / 2);
      const earlyMonths = new Set(visibleMonths.slice(0, Math.max(1, midpoint)));
      const recentMonths = new Set(visibleMonths.slice(Math.max(1, midpoint)));
      const earlyRows = filtered.filter(p => earlyMonths.has(p.month));
      const recentRows = filtered.filter(p => recentMonths.has(p.month));
      const earlyCategory = topCategoryFor(earlyRows);
      const recentCategory = topCategoryFor(recentRows);
      const firstCount = counts.get(firstActive) || 0;
      const lastCount = counts.get(lastActive) || 0;
      const periodSummary = translatedBrief && translatedBrief.timelineSummary
        ? `<div class="timeline-copy">${{translatedBrief.timelineSummary.map(item => `<p>${{escapeHtml(item)}}</p>`).join('')}}</div>`
        : '';
      els.timelineNarrative.innerHTML = `${{periodSummary}}<h3>${{escapeHtml(t('filteredMovementTitle'))}}</h3><ul>
        <li>${{formatText(t('activityNote'), {{firstActive: `<strong>${{firstActive}}</strong>`, firstCount: firstCount.toLocaleString(), lastActive: `<strong>${{lastActive}}</strong>`, lastCount: lastCount.toLocaleString(), trend: trendLabel(firstCount, lastCount)}})}}</li>
        <li>${{formatText(t('busiestNote'), {{peakMonth: `<strong>${{peak[0]}}</strong>`, peakCount: peak[1].toLocaleString()}})}}</li>
        <li>${{formatText(t('taxonomyShiftNote'), {{earlyCategory: `<strong>${{escapeHtml(earlyCategory[0])}}</strong>`, recentCategory: `<strong>${{escapeHtml(recentCategory[0])}}</strong>`}})}}</li>
      </ul>`;
    }}
    function renderPeriodInsights() {{
      const bounds = periodBounds();
      const periodBrief = periodInsightForBounds(bounds);
      const translatedBrief = translatedPeriodBrief(periodBrief);
      els.insightsTitle.textContent = translatedBrief?.insightsTitle || t('insightsTitle');
      if (!translatedBrief || !translatedBrief.insights) {{
        els.periodInsights.innerHTML = '';
        return;
      }}
      els.periodInsights.innerHTML = translatedBrief.insights.map(item => `<article class="insight-box">
        <div class="insight-label">${{escapeHtml(item.label)}}</div>
        <h3>${{escapeHtml(item.title)}}</h3>
        <p>${{escapeHtml(item.body)}}</p>
        <p class="insight-implication">${{escapeHtml(item.implication)}}</p>
      </article>`).join('');
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
        {{ label: t('matchingPapers'), value: filtered.length.toLocaleString(), detail: t('currentFilteredCorpus') }},
        {{ label: t('topTaxonomy'), value: topCategory[1].toLocaleString(), detail: topCategory[0] }},
        {{ label: t('topKeyword'), value: topKeyword[1].toLocaleString(), detail: topKeyword[0] }},
        {{ label: t('githubLinked'), value: repoCount.toLocaleString(), detail: formatText(t('projectPages'), {{count: projectCount.toLocaleString()}}) }},
        {{ label: t('hfUpvotes'), value: upvotes.toLocaleString(), detail: formatText(t('averagePerPaper'), {{count: avgUpvotes.toLocaleString()}}) }},
        {{ label: t('hfComments'), value: comments.toLocaleString(), detail: t('discussionSignals') }}
      ];
      els.insights.innerHTML = cards.map(item => `<article class="insight-card"><strong>${{escapeHtml(item.value)}}</strong><span>${{escapeHtml(item.label)}} - ${{escapeHtml(item.detail)}}</span></article>`).join('');
      const topPapers = [...filtered].sort((a, b) => (b.upvotes - a.upvotes) || (b.stars - a.stars) || (a.rank - b.rank)).slice(0, 5);
      els.topPapers.innerHTML = topPapers.length
        ? `<h3>${{escapeHtml(t('mostVisibleTitle'))}}</h3><ol>${{topPapers.map(p => `<li><a href="${{p.hf}}" target="_blank" rel="noopener">${{escapeHtml(p.title)}}</a> <span>(${{p.month}}, ${{p.upvotes.toLocaleString()}} ${{escapeHtml(t('hfUpvotes'))}}${{p.stars ? ', ' + p.stars.toLocaleString() + ' stars' : ''}})</span></li>`).join('')}}</ol>`
        : `<h3>${{escapeHtml(t('mostVisibleTitle'))}}</h3><p>${{escapeHtml(t('noPapers'))}}</p>`;
    }}
    function syncKeywordCards(filtered) {{
      const keyword = els.keyword.value;
      document.querySelectorAll('[data-keyword]').forEach(button => {{
        const selected = button.dataset.keyword === keyword;
        button.setAttribute('aria-pressed', String(selected));
        button.classList.toggle('is-selected', selected);
      }});
      const description = keyword === 'All Keywords' ? t('allKeywordConventions') : `${{keyword}} - ${{keywordDescriptions[keyword] || ''}}`;
      els.keywordStatus.textContent = formatText(t('selectedKeyword'), {{description, count: filtered.length.toLocaleString()}});
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
    function renderTaxonomyBody(section) {{
      const category = section.dataset.category;
      const rows = taxonomyRowsByCategory.get(category) || [];
      const list = section.querySelector('.paper-list');
      if (!list) return;
      section.querySelector('.load-more')?.remove();
      const visible = Math.min(rows.length, Number(section.dataset.visible || taxonomyBatchSize));
      const cards = rows.slice(0, visible).map(card).join('');
      const more = rows.length > visible
        ? `<button class="load-more" type="button">${{escapeHtml(formatText(t('showNext'), {{count: Math.min(taxonomyBatchSize, rows.length - visible).toLocaleString()}}))}}</button>`
        : '';
      list.innerHTML = cards || `<div class="empty">${{escapeHtml(t('emptyTaxonomy'))}}</div>`;
      if (more) {{
        list.insertAdjacentHTML('afterend', more);
        section.querySelector('.load-more').addEventListener('click', event => {{
          event.preventDefault();
          section.dataset.visible = String(visible + taxonomyBatchSize);
          section.querySelector('.load-more')?.remove();
          renderTaxonomyBody(section);
        }});
      }}
    }}
    function monthSpan(rows) {{
      const rowMonths = rows.map(p => p.month).filter(Boolean).sort();
      if (!rowMonths.length) return '-';
      const first = rowMonths[0];
      const last = rowMonths[rowMonths.length - 1];
      return first === last ? first : `${{first}}-${{last}}`;
    }}
    function totalUpvotes(rows) {{
      return rows.reduce((sum, p) => sum + Number(p.upvotes || 0), 0);
    }}
    function topPaper(rows) {{
      return [...rows].sort((a, b) => (b.upvotes - a.upvotes) || (b.stars - a.stars) || (a.rank - b.rank))[0] || null;
    }}
    function categoryInitials(category) {{
      if (category === 'All Taxonomies') return 'All';
      return category.split(/[\s,]+/).filter(Boolean).slice(0, 2).map(word => word[0]).join('').toUpperCase();
    }}
    function categoryKeywordText(rows, limit = 4) {{
      const keywordCounts = new Map();
      rows.forEach(p => p.tags.forEach(tag => keywordCounts.set(tag, (keywordCounts.get(tag) || 0) + 1)));
      return topEntries(keywordCounts, limit).map(([tag]) => tag).join(', ') || t('noKeywords');
    }}
    function listItems(items) {{
      return items && items.length
        ? `<ul>${{items.slice(0, 3).map(item => `<li>${{escapeHtml(item)}}</li>`).join('')}}</ul>`
        : '';
    }}
    function taxonomyIntro(category, rows, isAll) {{
      const top = topPaper(rows);
      const topTitle = top ? top.title : t('noPapers');
      const emphasis = isAll ? `${{t('allTaxonomies')}} - ${{categoryKeywordText(rows)}}` : categoryKeywordText(rows);
      const meta = categoryMeta[category] || {{}};
      const overview = isAll ? '' : `<div class="category-insight-grid">
        <div class="category-insight-box"><strong>${{escapeHtml(t('categoryOverview'))}}</strong>${{listItems(meta.trends || [])}}</div>
        <div class="category-insight-box limitation-box"><strong>${{escapeHtml(t('limitationsTitle'))}}</strong>${{listItems(meta.limitations || [])}}</div>
      </div>`;
      return `<div class="section-intro">
        <p><strong>${{escapeHtml(t('representativeEmphasis'))}}:</strong> ${{escapeHtml(emphasis)}}</p>
        <p><strong>${{escapeHtml(t('topRankedPaper'))}}:</strong> <span class="top-paper">${{escapeHtml(topTitle)}}</span></p>
        ${{overview}}
      </div>`;
    }}
    function taxonomySection(category, rows, isAll, shouldOpen) {{
      const meta = categoryMeta[category] || {{}};
      const color = meta.color || '#0f766e';
      const thumb = isAll
        ? '<span class="all-taxonomy-thumb" aria-hidden="true">All</span>'
        : `<span class="summary-thumb" style="--thumb-color:${{escapeHtml(color)}}">${{escapeHtml(categoryInitials(category))}}</span>`;
      const upvoteText = `${{totalUpvotes(rows).toLocaleString()}} ${{t('hfSignal')}}`;
      return `<section class="taxonomy-section${{isAll ? ' all-taxonomy-section' : ''}}" data-category="${{escapeHtml(category)}}">
        <details${{shouldOpen ? ' open' : ''}}>
          <summary>
            ${{thumb}}
            <span class="summary-title">${{escapeHtml(isAll ? t('allTaxonomies') : category)}}</span>
            <span class="category-count">${{escapeHtml(formatText(t('paperCount'), {{count: rows.length.toLocaleString()}}))}}</span>
            <span class="category-years">${{escapeHtml(monthSpan(rows))}}</span>
            <span class="category-citations">${{escapeHtml(upvoteText)}}</span>
          </summary>
          ${{taxonomyIntro(category, rows, isAll)}}
          <div class="taxonomy-body"><div class="paper-list"></div></div>
        </details>
      </section>`;
    }}
    function renderTaxonomies(filtered) {{
      taxonomyRowsByCategory = new Map();
      filtered.forEach(p => {{
        if (!taxonomyRowsByCategory.has(p.category)) taxonomyRowsByCategory.set(p.category, []);
        taxonomyRowsByCategory.get(p.category).push(p);
      }});
      const activeCategoryCount = taxonomyRowsByCategory.size;
      if (els.taxonomyTotalSummary) {{
        els.taxonomyTotalSummary.innerHTML = `<strong>${{escapeHtml(t('totalSelected'))}}:</strong> ${{escapeHtml(formatText(t('paperCount'), {{count: filtered.length.toLocaleString()}}))}}; <strong>${{escapeHtml(t('categoryCount'))}}:</strong> ${{activeCategoryCount.toLocaleString()}} ${{escapeHtml(t('categories'))}}.`;
      }}
      const orderedGroups = taxonomyOrder
        .filter(category => taxonomyRowsByCategory.has(category))
        .map(category => [category, taxonomyRowsByCategory.get(category)]);
      if (!orderedGroups.length || !filtered.length) {{
        els.list.innerHTML = `<div class="empty">${{escapeHtml(t('emptyTaxonomyList'))}}</div>`;
        return;
      }}
      const selectedCategory = els.category.value;
      taxonomyRowsByCategory.set('All Taxonomies', filtered);
      const allSection = taxonomySection('All Taxonomies', filtered, true, false);
      const categorySections = orderedGroups.map(([category, rows]) => taxonomySection(category, rows, false, selectedCategory !== 'All Taxonomies' || orderedGroups.length === 1)).join('');
      els.list.innerHTML = allSection + categorySections;
      els.list.querySelectorAll('.taxonomy-section').forEach(details => {{
        details.dataset.visible = String(taxonomyBatchSize);
        const panel = details.querySelector('details');
        if (panel?.open) renderTaxonomyBody(details);
        panel?.addEventListener('toggle', () => {{
          if (panel.open && !details.querySelector('.paper-card')) renderTaxonomyBody(details);
        }});
      }});
    }}
    function render() {{
      updateLanguageText();
      const bounds = periodBounds();
      const category = els.category.value;
      const keyword = els.keyword.value;
      const hasRepo = els.hasRepo.checked;
      const q = els.q.value.trim();
      const filtered = filterPapers();
      const categoryLabel = category === 'All Taxonomies' ? t('allTaxonomies') : category;
      const keywordLabel = keyword === 'All Keywords' ? t('allKeywords') : keyword;
      els.count.textContent = formatText(t('paperCount'), {{count: filtered.length.toLocaleString()}});
      els.filters.textContent = [categoryLabel, `${{bounds.from}} to ${{bounds.to}}`, keywordLabel, hasRepo ? t('githubLinked') : t('allLinks'), q ? t('searchActive') : ''].filter(Boolean).join(' | ');
      renderTimeline(filtered);
      renderPeriodInsights();
      renderInsights(filtered);
      syncKeywordCards(filtered);
      renderTaxonomies(filtered);
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
    [els.q, els.category, els.keyword, els.language, els.hasRepo].forEach(el => el.addEventListener('input', render));
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
