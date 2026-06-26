# Hugging Face Daily Papers 리뷰 (2023-05 through 2026-06)

## 초록

이 문서는 2023-05부터 2026-06까지 Hugging Face Daily Papers 월별 페이지에 올라온 고유 논문 16,240편을 metadata 기반으로 정리한 리뷰 초안입니다. 모든 월별 논문을 보존하고, 제목/초록/HF AI summary/키워드/업보트/댓글/GitHub 링크 같은 공개 메타데이터를 이용해 taxonomy-first 구조로 재분류했습니다.

## 방법

각 월은 `https://huggingface.co/api/daily_papers?month=YYYY-MM` 공개 API로 페이지가 빌 때까지 수집했습니다. 논문은 HF paper/arXiv id로 중복 제거했고, key idea, strengths, limitations, keyword tags는 공개 메타데이터에서 결정론적으로 생성했습니다. 유료 API, 유료 LLM, 유료 번역, 유료 compute는 사용하지 않았습니다.

## 분류별 규모

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

## 주목도가 높은 논문

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

## 해석

이 아카이브는 HF Daily Papers가 공개 AI 연구의 실용적 발견 계층으로 작동한다는 점을 보여줍니다. 코드, 프로젝트 페이지, 데모, 모델 릴리스, 벤치마크를 함께 제공하는 논문일수록 재사용 가능성과 커뮤니티 가시성이 커지는 경향이 있습니다.

## 한계

이 결과물은 PDF 전문을 읽고 작성한 systematic review가 아닙니다. HF upvotes, comments, GitHub stars는 과학적 타당성이 아니라 커뮤니티 가시성의 신호입니다. 방법론적 품질과 재현성은 각 논문, 코드, 데이터셋, 평가 세부사항을 직접 확인해야 합니다.
