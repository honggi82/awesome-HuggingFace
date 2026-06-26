# Curation Method

## Scope

- Repository: `honggi82/awesome-HuggingFace`
- Source: Hugging Face Daily Papers monthly pages and public API.
- Period: `2023-05` through `2026-06`.
- Included papers: every unique paper returned by the monthly API pages in that period.

## Data Source

The source endpoint is `https://huggingface.co/api/daily_papers`, using `month=YYYY-MM`, `sort=publishedAt`, `limit=100`, and incrementing `p` until an empty page is returned. This corresponds to the public monthly pages such as `https://huggingface.co/papers/month/2023-05`.

## Ranking And Taxonomy

No paper is excluded after successful monthly collection and deduplication. Ranking is a browsing order derived from HF upvotes, comments, GitHub stars recorded by HF, and availability of repository or project-page metadata. Taxonomy, keyword tags, key ideas, strengths, and limitations are deterministic metadata adapter outputs.

## GitHub-Awesome Skill2 And Paper-Curation Provenance

This repository follows `github-awesome-skill2` in metadata-adapter mode. The local `jehyunlee/paper-curation` checkout at `E:\조선대\연구\paper-curation` was inspected. Full PDF review via direct `paper-curation` was not run because the requested full monthly HF archive is large and upstream full review stages require separate explicit approval for paid or metered APIs.

## Validation

Generated outputs include CSV/JSON datasets, README, README HTML companion, review Markdown files with HTML companions, a static GitHub Pages site, period analysis JSON, link audit JSON, and provenance JSON.
