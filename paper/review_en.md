# Hugging Face Daily Papers Review (2023-05 through 2026-06)

## Abstract

This review draft summarizes a metadata-driven archive of 16,240 unique Hugging Face Daily Papers submitted from 2023-05 through 2026-06. The archive keeps every monthly paper surfaced through the public HF Daily Papers API, then organizes the corpus into taxonomy-first collections with deterministic keyword tags, key ideas, strengths, and limitations.

## Method

Each monthly page was fetched through `https://huggingface.co/api/daily_papers?month=YYYY-MM`, paginated until empty, deduplicated by HF paper/arXiv id, and enriched only from public metadata. No paid API, paid LLM, paid translation, or paid compute was used.

## Taxonomy Counts

- Foundation Models and Large Language Models: 8,096 papers
- Generative Media, Diffusion, and World Models: 2,039 papers
- Data, Evaluation, and Benchmarks: 1,947 papers
- Vision, Multimodal, and Video Understanding: 1,235 papers
- Robotics, Embodied AI, and Control: 831 papers
- Efficient Training, Inference, and AI Systems: 747 papers
- Agents, Tool Use, and Autonomous Workflows: 520 papers
- General Machine Learning and Optimization: 394 papers
- Speech, Audio, NLP, and Code Applications: 260 papers
- Responsible, Safe, and Interpretable AI: 126 papers
- Graph, Recommendation, and Structured Learning: 34 papers
- AI for Science, Medicine, and Engineering: 11 papers

## Highly Visible Papers

- [Sharing is Caring: Efficient LM Post-Training with Collective RL Experience Sharing](https://huggingface.co/papers/2509.08721) (2025-09, 665 upvotes): Swarm sAmpling Policy Optimization (SAPO) is a decentralized and asynchronous RL algorithm that enhances post-training language models without supervised fine-tuning, achieving significant reward gains and scalability across diverse hardware.
- [GrandCode: Achieving Grandmaster Level in Competitive Programming via Agentic Reinforcement Learning](https://huggingface.co/papers/2604.02721) (2026-04, 637 upvotes): GrandCode is a multi-agent reinforcement learning system that outperforms human competitors in competitive programming challenges by orchestrating specialized agent modules and employing novel reward policy optimization techniques.
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://huggingface.co/papers/2402.17764) (2024-02, 630 upvotes): A 1-bit LLM variant, BitNet b1.58, achieves comparable performance to full-precision models with reduced computational costs and introduces new scaling laws and hardware design opportunities.
- [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://huggingface.co/papers/2509.26507) (2025-10, 551 upvotes): BDH, a biologically inspired Large Language Model, combines scale-free network architecture with Hebbian learning to achieve Transformer-like performance while maintaining interpretability.
- [A Very Big Video Reasoning Suite](https://huggingface.co/papers/2602.20159) (2026-02, 526 upvotes): A large-scale video reasoning dataset and benchmark are introduced to study video intelligence capabilities beyond visual quality, enabling systematic analysis of spatiotemporal reasoning and generalization across diverse tasks.
- [Less is More: Recursive Reasoning with Tiny Networks](https://huggingface.co/papers/2510.04871) (2025-10, 517 upvotes): Tiny Recursive Model (TRM) achieves high generalization on complex puzzle tasks using a small, two-layer network with minimal parameters, outperforming larger language models.
- [Adam's Law: Textual Frequency Law on Large Language Models](https://huggingface.co/papers/2604.02176) (2026-04, 509 upvotes): A novel framework for improving large language model performance through textual frequency analysis, including laws, distillation, and curriculum training approaches.
- [ABot-Earth 0.5: Generative 3D Earth Model](https://huggingface.co/papers/2606.09967) (2026-06, 482 upvotes): ABot-Earth 0.5 generates realistic 3D environments from satellite imagery using 3D Gaussian Splatting representation, enabling fast synthesis and real-time visualization for Embodied AI applications.
- [Looped World Models](https://huggingface.co/papers/2606.18208) (2026-06, 463 upvotes): Looped World Models introduce iterative latent state refinement through shared transformer blocks, achieving 100x parameter efficiency while adapting computational depth to prediction complexity.
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948) (2025-01, 454 upvotes): DeepSeek-R1-Zero and DeepSeek-R1 utilize reinforcement learning and multi-stage training to enhance reasoning capabilities, with DeepSeek-R1 achieving performance comparable to OpenAI-o1-1217.
- [AI Can Learn Scientific Taste](https://huggingface.co/papers/2603.14473) (2026-03, 431 upvotes): Great scientists have strong judgement and foresight, closely tied to what we call scientific taste.
- [Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players](https://huggingface.co/papers/2605.28816) (2026-05, 431 upvotes): A generative multi-agent world model is presented that uses simplex rotary agent encoding and sparse hub attention to enable scalable, permutation-symmetric interaction between multiple agents in interactive video generation.

## Interpretation

The corpus shows how HF Daily Papers became a practical signal layer for open AI research: papers with code, project pages, demos, model releases, and benchmark artifacts are easier to discover and reuse. The strongest metadata signals cluster around foundation models, agents, multimodal models, generative media, efficient training/inference, and evaluation resources.

## Limitations

This is not a full-PDF systematic review. HF upvotes, comments, and GitHub stars measure community visibility, not scientific validity. Full methodological claims require reading the papers, code, datasets, and evaluation details directly.
