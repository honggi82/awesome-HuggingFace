# arXiv:2601.09668v2[cs.CV]15Jan2026

[Figure 1]

## STEP3-VL-10B Technical Report

Multimodal Intelligence Team, StepFun

Homepage: https://stepfun-ai.github.io/Step3-VL-10B ModelScope: https://modelscope.cn/collections/stepfun-ai/Step3-VL-10B Huggingface: https://huggingface.co/collections/stepfun-ai/step3-vl-10b

[Figure 2]

### Abstract

We present STEP3-VL-10B, a lightweight open-source foundation model designed to redefine the trade-off between compact efficiency and frontier-level multimodal intelligence. STEP3VL-10B is realized through two strategic shifts: first, a unified, fully unfrozen pre-training strategy on 1.2T multimodal tokens that integrates a language-aligned Perception Encoder with a Qwen3-8B decoder to establish intrinsic vision-language synergy; and second, a scaled post-training pipeline featuring over 1k iterations of reinforcement learning. Crucially, we implement Parallel Coordinated Reasoning (PaCoRe) to scale test-time compute, allocating resources to scalable perceptual reasoning that explores and synthesizes diverse visual hypotheses. Consequently, despite its compact 10B footprint, STEP3-VL-10B rivals or surpasses models 10×–20× larger (e.g., GLM-4.6V-106B, Qwen3-VL-235B) and top-tier proprietary flagships like Gemini 2.5 Pro and Seed-1.5-VL. Delivering best-in-class performance, it records 92.2% on MMBench and 80.11% on MMMU, while excelling in complex reasoning with 94.43% on AIME2025 and 75.95% on MathVision. We release the full model suite to provide the community with a powerful, efficient baseline.

Step3-VL-10B Step3-VL-10B (PaCoRe)

GLM-4.6V-106B-A12B Qwen3VL-235B-A22B-Thinking

Seed-1.5-VL-Thinking Gemini-2.5-Pro

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

100

92.2 94.4

91.8

85.5

87.7

84.0

80.1

78.1

80

76.0

70.8

Accuracy(%)

62.6

60

40

20

0

MMMU MathVista MathVision MMBench AIME2025 MultiChallenge

Figure 1 | Performance comparison of STEP3-VL-10B against state-of-the-art multimodal foundation models. With PaCoRe (Parallel Coordinated Reasoning (Hu et al., 2026), STEP3-VL-10B scales test-time compute to bridge the perception and reasoning performance gap with 100B+ parameter models.

##### Contents

- 1 Introduction 3
- 2 Pre-train 4

- 2.1 Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Data Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.3 Training Recipe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 3 Post-Train 7

- 3.1 Supervised Finetuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2 Reinforcement Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 3.2.1 Optimization Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 3.2.2 Reward System . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3.2.3 Scaling Sequential Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . 9
- 3.2.4 Further Scaling Parallel Coordinated Reasoning . . . . . . . . . . . . . . . 10

- 4 Evaluations 10

- 4.1 Evaluation Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 4.2 Multimodal Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4.3 Text-Centric Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 4.4 Comparison with Larger Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 5 Discussion 16

- 5.1 Ablations and Design Insights . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 5.2 RL Dynamics, Performance, and Emergence . . . . . . . . . . . . . . . . . . . . . 17

- 6 Conclusion and Future Work 19
- 7 Author List 33

- A More Results 34
- B Serialization Details for Synthesis in PaCoRe 38
- C Evaluation Details 38

- C.1 Evaluation Details for Multimodal Benchmarks . . . . . . . . . . . . . . . . . . . . 38
- C.2 Evaluation Details for Text-Centric Benchmarks . . . . . . . . . . . . . . . . . . . 50
- C.3 Evaluation Details for Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

##### 1. Introduction

The development of Multimodal Large Language Models (MLLMs) has largely been driven by a relentless pursuit of scale. While proprietary frontier models like Gemini-3-Pro (Team, 2025b) and GPT-5.2 (OpenAI, 2025a) have pushed the boundaries of multimodal intelligence through massive scaling, their heavy computational demands pose barriers to practical deployment in the real world. Conversely, lightweight models (under 10B parameters) have traditionally been characterized as “efficient but limited”, which struggle to advance sophisticated reasoning and perceptual capabilities within restricted parameter budgets.

In this work, we introduce STEP3-VL-10B, a foundation model that redefine the trade-off between compact efficiency and frontier-level multimodal intelligence. Despite its modest 10B parameter footprint, STEP3-VL-10B excels in visual perception, complex reasoning, and human-centric alignment. It consistently outperforms models under the 10B scale and rivals or even surpasses significantly larger open-weights models (10×–20× its size), such as GLM-4.6V (106B-A12B)(Team et al., 2025d) and Qwen3-VL-Thinking (235B-A22B)(Bai et al., 2025), as well as established proprietary flagships like Gemini-2.5-Pro(Team, 2025a) and Seed-1.5-VL (Guo et al., 2025a). Across representative benchmarks, STEP3-VL-10B achieves 75.95% on MathVision, 80.11% on MMMU, and a staggering 94.43% on AIME2025 (Fig. 1).

The success of STEP3-VL-10B is driven by two key strategic design in how we build efficient and powerful multimodal models:

- • Unified Pre-training on High-Quality Multimodal Corpus: We implement a singlestage, fully unfrozen training strategy on a 1.2T token multimodal corpus, focusing on two foundational capabilities: reasoning (e.g., general knowledge and education-centric tasks) and perception (e.g., grounding, counting, Optical Character Recognition, and Graphical User Interface interactions). By jointly optimizing the Perception Encoder (Bolya et al., 2025) and the Qwen3-8B (Yang et al., 2025a) decoder, STEP3-VL-10B establishes a intrinsic vision-language synergy.
- • Scaled Multimodal Reinforcement Learning (RL) and Parallel Reasoning: We unlock frontier capabilities through a rigorous post-training pipeline, comprising two-stage supervised finetuning (SFT) and over 1k iterations of RL with both verifiable rewards (RLVR) and human feedback (RLHF). Beyond sequential reasoning, we adopt Parallel Coordinated Reasoning (PaCoRe) (Hu et al., 2026), which allocates test-time compute to aggregate evidence from parallel visual exploration. These designs enable the 10B model to solve complex perceptual and reasoning tasks that typically require substantially larger systems.

To understand the drivers of this efficiency, we provide a in-depth analysis of the model’s internal mechanisms in Sec.5, with an emphasis on the learning dynamics unlocked by RL scaling. In particular, to counteract the length diminishment characteristic of perception tasks, we leverage PaCoRe to facilitate a form of multi-agent synthesis: parallel proposers generate diverse hypotheses, which are subsequently distilled through sequential cross-checking. This emergent synthesis effectively externalizes implicit visual processes, offering a promising direction for scaling perceptual reasoning.

This trajectory not only sheds light on how models can progressively bridge intelligence and interaction with the physical world (Sec.6), but also motivates our commitment to closing critical technical gaps in the open ecosystem. By releasing the final model weights and detailed training documentation, STEP3-VL-10B demonstrates that with a right perception- and reasoningcentric design, the gap between “compact” and “frontier” is no longer intractable.

##### 2. Pre-train

Our pre-training framework is designed to construct a capable vision foundation model that sets a high upper bound for subsequent post-training stages, prioritizing data quality and architectural synergy over unnecessary complexity.

###### 2.1. Architecture

STEP3-VL-10B integrates the 1.8B language-optimized Perception Encoder (Bolya et al., 2025), selected over the spatial-optimized variant for its pre-aligned linguistic features that ensure superior convergence. This visual backbone is coupled with Qwen3-8B (Yang et al., 2025a), utilized as the decoder for its robust text generation foundation and proven plasticity for multimodal adaptation. Similar to Step-3 (Team, 2025d) and DeepSeek-OCR (Wei et al., 2025a), these components are bridged by a projector performing 16× spatial downsampling via two consecutive stride-2 layers, effectively compressing visual tokens while retaining essential information. To capture fine-grained details efficiently, we adopt a multi-crop strategy (Caron et al., 2021) that decomposes images into a 728 × 728 global view and multiple 504 × 504 local crops. This design leverages batch-dimension parallelism to sidestep the complexity of variablelength packing (Shah et al., 2024). Finally, we encode spatial structure by appending newline tokens to patch rows and utilize standard 1D RoPE (Su et al., 2024) for positional modeling, as advanced variants yielded no significant gains in our setup.

###### 2.2. Data Construction

To equip STEP3-VL-10B with both fine-grained perception and complex reasoning capabilities, we incorporate a large-scale text corpus (Bakouch et al., 2025) and construct a comprehensive multimodal pre-training dataset spanning the following key domains.

Knowledge. We curate visual data with high knowledge density from multiple complementary channels, covering both structured interleaved data and diverse image–text pairs.

- • Interleaved Data. We collect interleaved image–text data from Common Crawl (CommonCrawl) and our in-house crawler StepCrawl, which targets the domestic internet, and further augment this corpus with keyword-based search results. To suppress noise inherent in web-scale data, we discard webpages with excessive image download failures (> 90%), QR-code content, and images with extreme aspect ratios.
- • Image–Text Pairs. We organize image–text pairs into four complementary categories:

(1) Open-source datasets, including LAION (Schuhmann et al., 2022), COYO (Byeon et al., 2022), BLIP-CCS (Li et al., 2022), and Zero (Xie et al., 2023). To mitigate long-tail concept imbalance, we apply concept-balanced resampling via CLIP (Radford et al., 2021)based clustering. (2) Keyword-based retrieval, where keywords mined from high-quality knowledge websites are used to query commercial search engines (e.g., Baidu and Bing), to gather targeted domain-specific data. (3) Pairs extracted from interleaved data, where for each image we extract candidate descriptions from surrounding text (above, below, and alt-text). Then we select the most suitable one using CLIP-based similarity to assess alignment and aesthetic scores to evaluate image quality. (4) Mosaic augmentation, in which four images are concatenated into a single input. This effectively extends the input resolution, increases visual content density within each sample, and encourages the model to learn spatial and positional reasoning across multiple regions.

Education. We curate a dataset of approximately 15M samples spanning K-12 education, higher education, and adult learning. The K-12 subset covers mathematics, physics, chemistry, and humanities, and includes specialized data such as chemical formulas and structure diagrams sourced from open datasets and synthetically generated using CoSyn (Yang et al., 2025c), as well as geometry (including analytic geometry) problems constructed from a mixture of synthetic data and real exam images with annotated captions. Beyond this, the dataset extends to universitylevel domains including STEM, medicine, arts, and finance, as well as adult education scenarios such as driving license exams, CPA, and legal examinations. Exam questions are collected from a combination of licensed exam materials and open-source problem sets (Ben Abacha et al., 2019; He et al., 2020; Sujet AI, 2024), while supporting knowledge content is sourced from textbooks, workbooks, and high-quality educational websites.

Optical Character Recognition (OCR). We curate a comprehensive OCR corpus spanning image-level and document-level text recognition, and visual-to-code reconstruction.

- • Image to Text. We curate a dataset comprising 10M real-world images and 30M synthetic samples covering diverse fonts, layouts, and text orientations. Real-world data are collected from open-source datasets (Shi et al., 2017; Yao et al., 2012) and annotated using PaddleOCR (Cui et al., 2025a), while synthetic samples are generated using SynthDog (Kim et al., 2022).
- • Image to Code. We organize this dataset by target code form, spanning markup-based and

programmatic graphics. (1) Markup-based Code. For Markdown, LATEX, and Matplotlib, we combine over 10M samples from open-source datasets (Chen et al., 2024a; Masry et al., 2023; Xia et al., 2023) with an automated data generation pipeline that produces more than 15M synthetic infographics. Instead of fully delegating generation to LLMs (Yang et al., 2025c), we enforce fine-grained rendering rules across multiple render tools. (2) Programmatic Graphics Code. For languages such as TikZ and Graphviz, we curate approximately 5M reconstruction tasks that require translating visual inputs into executable code, spanning diverse visual inputs including natural images, human-created tables, and geometries.

- • Document to Text. This dataset comprises approximately 80M full-page documents. Concretely, we apply PaddleOCR or MinerU 2.0 (Wang et al., 2024a) to annotate collected pages from books and academic papers.
- • Document to Code. We curate data spanning three primary markup languages including HTML, Markdown and latex. HTML data focus on table-centric content and are sourced from rendered web page code, Markdown conversions. Markdown data cover tables and lightweight documents, collected from rendered GitHub README files, HTML conversions. Latex data are extracted at scale from arXiv corpora, comprising approximately 4M tables and 100M formulas. During rendering, we explicitly handle elements such as references and hyperlinks to prevent mismatches between visual and textual content. In addition, we incorporate open-source datasets (Chai et al., 2025; Laurençon et al., 2024; Liu et al., 2024a; Yuan et al., 2022) to further enrich this subset.

Grounding & Counting. We collect approximately 400M samples to support fine-grained perceptual understanding. Grounding data include both bounding-box-based and point-based annotations sourced from open detection datasets such as OpenImages (Kuznetsova et al., 2020), COCO (Lin et al., 2015), Merlin (Yu et al., 2024, 2025a), and PixMo (Deitke et al., 2024), as well as in-house text paragraph detection tasks. Counting data are drawn from open-source counting (Kaggle; SakiRinn) and are further constructed by converting high-quality object detection annotations into counting formulations.

Visual Question Answering (VQA). This subset comprises approximately 10M samples targeting holistic image content understanding. It includes curated open-source VQA datasets (Liu et al., 2025b; Zellers et al., 2019), as well as high-quality question–answer pairs automatically generated from image caption data. In addition, we construct a OCR VQA subset with around 20M samples, combining open-source data (aallail; Poznanski et al., 2025; Wei et al., 2024) with QA pairs generated from other OCR-related task.

Graphical User Interface (GUI). We construct a large-scale GUI dataset as in Step-GUI (Yan et al., 2025) comprising approximately 23M samples to endow the model with practical and executable UI understanding and interaction capabilities. The dataset covers both mobile platforms, including Android and iOS, and desktop environments spanning Windows, Linux, and macOS, with data collected from over 200 applications. Notably, accurate grounding annotations are generated jointly with trajectory data, ensuring consistent supervision between action execution and perception.

- • Caption. This subset provides 700K detailed captions for UI interfaces, conveying explicit knowledge about page layouts and functional regions to support structural understanding of interfaces.
- • Knowledge VQA. We include over 1M question–answer pairs that reinforce precise localization and functional understanding of UI elements.
- • Trajectory Modeling. To model realistic sequential human–computer interactions, we curate more than 2M trajectory samples defined over a granular action space comprising 12 atomic actions, such as CLICK, SLIDE, and TYPE. These trajectories strengthen action output formatting, state summarization, and decision-making capabilities, and cover a wide range of tasks including operation execution, information retrieval, and information summarization.
- • Grounding. The dataset further includes over 19M grounding samples with both pointbased and bounding-box-based grounding across diverse interface layouts and resolutions.
- • OCR. For web-based interfaces targeting Browser-use-GUI, we crawl approximately 30M web pages and extract textual content together with precise element coordinates, supporting fine-grained layout-aware understanding.

###### 2.3. Training Recipe

We adopt a single-stage, fully unfrozen training strategy optimized with AdamW (Loshchilov and Hutter, 2017) (𝛽1 = 0.9, 𝛽2 = 0.95, 𝜀 = 10−8, and weight decay = 0.01), training the model on a total of 1.2T tokens over 370K iterations with a global batch size of 8,192 and a sequence length of 4,096.

To balance training scale and data quality, we employ a two-phase learning rate schedule.

During the first phase, covering the initial 900B tokens, the learning rate is decayed from 5×10−5 to 1 × 10−5 to emphasize broad representation learning. In the second phase, spanning the remaining 300B tokens, we transition to a higher-quality data mixture and further anneal the learning rate from 1 × 10−5 to 6 × 10−6, acting as a cool-down phase to consolidate fine-grained perceptual (e.g., OCR and grounding) and reasoning capabilities .

##### 3. Post-Train

In the post-training stage, we adopt a two-stage Supervised Finetuning (SFT) strategy followed by Reinforcement Learning (RL). For the RL phase, we employ Proximal Policy Optimization (PPO) (Schulman et al., 2017) with Generalized Advantage Estimation (GAE) (Schulman et al., 2015) as the core optimization algorithm, coupled with a meticulously designed reward system. Crucially, we scale inference compute by progressing from sequential reasoning to parallel coordinated reasoning, aiming to fully unlock STEP3-VL-10B’s perception and reasoning capabilities.

- 3.1. Supervised Finetuning

Data Construction. Our SFT strategy focuses on multi-modal, high-quality, reasoning-oriented data. We initially collected millions of prompts from the open-source community (Guha et al., 2025; LI et al., 2024), spanning diverse domains such as mathematics, coding, science, and logical reasoning. We also incorporated open-source datasets (An et al., 2025; Tong et al.,

- 2024a; Wiedmann et al., 2025) for visual perception and recognition, including grounding, OCR, and complex document/chart understanding, to ensure the model can precisely perceive and reason over visual elements. Leveraging these prompts, we distilled high-quality responses from internal frontier model. This foundational dataset underwent a rigorous “two-pipe” filtration process: first, applying predefined rules to eliminate degenerate patterns (e.g., infinite repetitions); and second, performing comprehensive benchmark decontamination via exact matching and 𝑁-gram matching (𝑁 = 64).

Two-Stage SFT Strategy. We implemented a phased training approach to progressively align the model’s reasoning capabilities across modalities. The training was conducted with a global batch size of 32 and an extended sequence length of 128k to support long-context understanding.

- • Stage 1: Text-Dominant Reasoning. The data mixture was set at a 9 : 1 ratio of text to multimodal samples, establishing a strong logical and linguistic foundation.
- • Stage 2: Multimodal Integration. We shifted the composition to a 1 : 1 ratio, effectively balancing textual reasoning with visual intelligence to enhance the model’s performance on interleaved multimodal tasks.

Training Recipe. We employed a cosine learning rate scheduler with a 200-step warmup phase, where the learning rate peaks at 1 × 10−4 and anneals to a final value of 1 × 10−5. To optimize the learning process across diverse data sources, we implemented domain-specific sampling weights in the dataloader, which translate to varying epoch counts for different domains. Throughout the entire two-stage process, the model was trained on a total of approximately 190B tokens during stage 1 and 36B during stage 2.

- 3.2. Reinforcement Learning

- 3.2.1. Optimization Algorithm

We adopt PPO combined with GAE as our optimization algorithm for reinforcement learning, following Open-Reasoner-Zero (Hu et al., 2025) and Open-Vision-Reasoner (Wei et al., 2025c).

Formally, given a multimodal input tuple consisting of an image 𝐼 and a textual prompt 𝑞, the policy network 𝜋𝜃 generates a response trajectory 𝜏 = (𝑠0, 𝑎0, . . . , 𝑠𝑇−1, 𝑎𝑇−1) of length 𝑇. The state 𝑠𝑡 encapsulates the input context (𝐼,𝑞) and the sequence of tokens generated prior to step 𝑡,

while 𝑎𝑡 denotes the action (token) sampled at step 𝑡.

To effectively balance the bias-variance trade-off in policy gradient estimation, we utilize GAE for advantage computation. The advantage estimator 𝐴ˆ𝑡 for a state-action pair (𝑠𝑡, 𝑎𝑡) is defined as:

𝑇∑︁−𝑡−1

𝐴ˆ𝑡 =

(𝛾𝜆)𝑙𝛿𝑡+𝑙, with 𝛿𝑡′ = 𝑟𝑡′ + 𝛾𝑉𝜑(𝑠𝑡′+1) −𝑉𝜑(𝑠𝑡′), (1)

𝑙=0

where 𝑟𝑡′ is the reward at step 𝑡′, 𝑉𝜑 represents the value function parameterized by 𝜑, and 𝛾, 𝜆 ∈ [0,1] are the discount factor and GAE smoothing parameter, respectively.

The policy parameters 𝜃 are updated by maximizing the clipped surrogate objective, which penalizes large policy deviations to maintain training stability:

JPPO(𝜃) = Eˆ 𝑡 min 𝜌𝑡(𝜃)𝐴ˆ𝑡,clip (𝜌𝑡(𝜃),1 − 𝜀,1 + 𝜀) 𝐴ˆ𝑡 , (2)

where 𝜌𝑡(𝜃) = 𝜋𝜋𝜃(𝑎𝑡|𝑠𝑡)

old(𝑎𝑡|𝑠𝑡) is the probability ratio, and 𝜀 is the clipping hyperparameter.

Concurrently, the value function is updated to minimize the mean squared error between the estimated value and a target value 𝑉𝑡target, typically the estimated discounted return 𝐺𝑡 = 𝐴ˆGAE𝑡 (𝛾,𝜆) +𝑉𝜑(𝑠𝑡):

###### 𝑇∑︁−1

- 1

- 2

2

𝑉𝜑(𝑠𝑡) −𝑉𝑡target

, (3)

Jvalue(𝜑) =

E𝜏∼𝜋

𝜃old

𝑡=0

Concretely, we adopt a variant of PPO algorithm with GAE (𝛾 = 1, 𝜆 = 1) for off-policy setting, omitting standard importance sampling. Each iteration splits samples into four minibatches. The actor and critic learning rates are set to 2 × 10−6 and 5 × 10−6, respectively. To mitigate training–inference inconsistency, we apply the truncated importance sampling ratio with threshold 𝐶 = 8 following (Yao et al., 2025). The entire RL phase runs for 1,400 training iterations, updating only the decoder while keeping the encoder frozen.

###### 3.2.2. Reward System

To support scalable training across heterogeneous modalities and task types, we design a bifurcated reward framework that explicitly distinguishes between verifiable tasks, where objective correctness can be reliably assessed, and non-verifiable tasks, where alignment must be guided by preference modeling and constraints.

Verifiable Rewards: Precision and Consistency. For tasks with accessible ground truth, our reward design prioritizes strict correctness and reasoning consistency. We implement a multifaceted verification pipeline that combines perception-based, and model-assisted signals.

- • Perception Rewards. For perception tasks such as pointing and grounding, following Perception-R1 (Yu et al., 2025b), we align the model’s geometric outputs with deterministic ground truths using metrics like Intersection over Union (IoU) or Euclidean distance. Crucially, we implement strict, distance-decay reward shaping to guarantee a distinct and unambiguous optimization landscape and robust RL convergence.
- • Model-Based Verification. For general visual tasks, we deploy GPT-OSS-120B (OpenAI, 2025b) as the answer verifier. Compared to simple string matching or mathverify-style heuristics, this model-based verification mechanism is substantially more robust to noisy or imperfect ground truth and significantly improves training stability. In particular, it provides parse-invariant evaluation that is resilient to formatting variations (e.g., idiosyncratic

LATEX), recognizes semantic equivalence among mathematically identical expressions or reordered derivation steps, and enforces process consistency by penalizing false positives, cases where the model arrives at a correct final answer through flawed or unsupported reasoning. Together, these properties yield more reliable reward signals for supervising complex reasoning behaviors.

Non-Verifiable Rewards: Preference and Constraints. For open-ended generation where ground truth is absent, we rely on learned preference models and heuristic constraints to guide human-centric alignment.

- • Generative Reward Modeling (GenRM). We adopt a pairwise preference framework where the GenRM evaluates rollouts against responses generated by a more capable teacher model. Moving beyond direct outcome continuous rewarding, our GenRM incorporates an explicit reasoning judgment before deriving a fine-grained scalar score to discern subtle quality differences between plausible responses.
- • Behavioral Regularization. To mitigate “reward hacking” and enforce safety and reliability constraints during optimization, we incorporate a set of model-based penalty terms as behavioral regularization. Specifically, we impose language consistency penalties to discourage code-switching and question–answer language mismatch; apply strict citation verification that zeros the reward when fabricated references or links are detected, directly targeting hallucinations at the source; and introduce epistemic calibration penalties to suppress unjustified certainty or overconfident claims, encouraging the model to appropriately express uncertainty in ambiguous or underspecified settings. Together, these constraints act as guardrails that stabilize preference optimization and align model behavior with safety and trustworthiness objectives.

###### 3.2.3. Scaling Sequential Reasoning

We aim to scale the model’s reasoning capability by incentivizing extended sequential thinking processes, effectively converting test-time compute into performance gains. Concretely, we structure our sequential reasoning training to first establish robust logical foundations on verifiable tasks before aligning with subjective human preferences.

Reinforcement Learning with Verifiable Rewards (RLVR). We conduct training on a diverse set of verifiable multimodal tasks, drawing from large-scale open-source datasets such as Open-Vision-Reasoner (Wei et al., 2025c), which cover mathematics, geometry, physics, scientific reasoning, perception, recognition, chart-based reasoning, and puzzles, together with visual grounding tasks from Perception-R1 (Yu et al., 2025b) and internal K–12 educational resources.

To ensure high-quality supervision, we design a multi-dimensional filtration pipeline along three axes: (1) Checkability is enforced by employing GPT-OSS-120B to perform four independent verification passes per prompt, retaining only all-agree samples. (2) Visual relevance is ensured by using an early version of STEP3-VL-10B to evaluate the semantic correlation and mutual contribution between images and questions, filtering out redundant or misaligned multimodal pairs. (3) Finally, to control difficulty, we perform 24 rollouts per prompt to identify some-accept samples, namely cases that are neither trivially solvable nor consistently failed. Each filtration stage plays a non-trivial role in ensuring long-term training stability and enabling sustained performance gains. The RLVR stage is executed for 600 iterations with a maximum sequence length of 24k. For each iteration, we sample 512 prompts with 16 rollouts per prompt, optimizing via the aforementioned verifiable reward system.

Reinforcement Learning from Human Feedback (RLHF). Building on the reasoning-focused checkpoint from RLVR, we further align the model with human preferences using open-ended tasks. We curate prompts from opensourced arena datasets (Chiang et al., 2024; Chou et al., 2024; Tang et al., 2025) and internal instruction pools, explicitly filtering for uncheckable queries that lack deterministic ground-truth. For these prompts, we leverage our strongest internal models to generate high-quality reference responses as anchors for preference learning. This stage proceeds for 300 iterations using a maximum sequence length of 32k. We maintain a throughput of 512 prompts per iteration with 8 rollouts per prompt, optimizing the model against an unverifiable reward system to refine its conversational and alignment capabilities while preserving its underlying reasoning strength.

###### 3.2.4. Further Scaling Parallel Coordinated Reasoning

To further scale test-time compute beyond the limits of sequential generation, we adopt a parallel coordinated reasoning paradigm following PaCoRe (Hu et al., 2026). This approach allocates compute to explore diverse perceptual hypotheses in parallel and synthesizes them into a unified conclusion.

To construct the training data for parallel reasoning, we extend the difficulty filtration (Axis 3) from the RLVR stage. We repurpose the 24 rollouts from the difficulty tagging phase as a message cache pool. Starting with the identified some-accept prompts, we apply a secondary Synthesis Filtration to ensure coordinated solvability: (1) We simulate the parallel reasoning process by sampling 16–24 messages from the pool and feeding them back into the model as a "synthesis context" to regenerate responses. (2) We strictly retain instances that remain categorized as some-accept under this coordinated setting. Crucially, this prevents task trivialization to maintain effective reward signals, while compelling the model to perform multi-perspective self-verification and cross-checking.

The model is optimized using PPO in a strict on-policy setting for 500 iterations. We utilize a maximum sequence length of 64k to accommodate the aggregated context. Each iteration samples 64 prompts with 16 rollouts per instance, stabilizing the optimization of coordinated behaviors against the verifiable reward system.

##### 4. Evaluations

To rigorously validate the capabilities of STEP3-VL-10B, we conduct extensive evaluations across a broad spectrum of multimodal and text-centric benchmarks. Our results position STEP3VL-10B as the most powerful open-source model in the 10B parameter class, demonstrating performance that not only significantly outperforms 7–10B open-source models but also rivals frontier open-source systems (10×–20× larger) and proprietary flagships in reasoning and perception domains.

###### 4.1. Evaluation Setup

Benchmark Protocols. We evaluate STEP3-VL-10B on a comprehensive suite of over 60 benchmarks. To ensure a holistic assessment consistent with our reported results, we categorize these benchmarks into specific capability domains across multimodal and text-centric modalities.

- I. Multimodal Benchmarks. We assess vision-language capabilities across nine distinct

domains:

- • STEM / Multimodal Reasoning: We employ MMMU (Standard/Pro) (Yue et al., 2024), MathVista (Lu et al., 2023), MathVision (Wang et al., 2024b), MathVerse (Zhang et al., 2024), DynaMath (Zou et al., 2024), We-Math (Qiao et al., 2024), and PhyX (Shen et al., 2025) for scientific and mathematical reasoning. Logical and puzzle-solving abilities are tested via LogicVista (Xiao et al., 2024), ZeroBench (Roberts et al., 2025), VisuLogic (Xu et al., 2025), and HLE (Team, 2025c) .
- • Recognition / General VQA: General perception is evaluated using MMBench (EN/CN) (Liu et al., 2024b), SimpleVQA (Cheng et al., 2025), and MMStar (Chen et al., 2024b). Robustness and fine-grained recognition are assessed via HallusionBench (Guan et al., 2024), MMVP (Tong et al., 2024c), ReMI (Kazemi et al., 2024), M3GIA (Song et al., 2024), and DoYouSeeMe (Kanade and Ganu, 2025).
- • Counting: We utilize CountBench (Paiss et al., 2023), CountQA (Tamarapalli et al., 2025), and PixMo-Count (Deitke et al., 2024) to evaluate precise object enumeration.
- • Instruction Following: Multimodal compliance is tested on MM-MT-Bench (Ying et al., 2024), MIA-Bench (Qian et al., 2025), and MM-IFEval (Ding et al., 2025).
- • Multimodal Code: Visual coding capabilities are evaluated using HumanEval-V (Zhang et al., 2025), and Design2Code (including Design2Code-Hard) (Si et al., 2025).
- • OCR: Text-rich image understanding is assessed via OCRBench (Liu et al., 2024c), OmniOCR (OmniAI), and CC-OCR (Yang et al., 2024).
- • 2D / 3D Spatial Understanding: We conduct extensive spatial reasoning tests using BLINK (Fu et al., 2024), CVBench (Tong et al., 2024b), MMSI-Bench (Yang et al., 2025b), ERQA (Team et al., 2025b), OmniSpatial (Jia et al., 2025), All-Angles-Bench (Yeh et al., 2025), MindCube-tiny (Yin et al., 2025), RealWorldQA (X.AI, 2024), SpatialViz-Bench (Wang et al., 2025a), STARE (Li et al., 2025c), CoreCognition (Li et al., 2025d), V* (Wu and Xie, 2023), and ViewSpatial (Li et al., 2025a).
- • Document & Chart Understanding: Complex parsing is tested on CharXiv (RQ) (Wang et al., 2024d), AI2D (Kembhavi et al., 2016), OmniDocBench (Ouyang et al., 2024), and CSVQA (Jian et al., 2025).
- • GUI Grounding: To evaluate actionable intelligence, we employ ScreenSpot-Pro (Li et al., 2025b), ScreenSpot-V2 (Wu et al., 2024), OSWorld-G (Xie et al., 2025), and MMBenchGUI (Wang et al., 2025c).

- II. Text-Centric Benchmarks. We verify LLM foundations across six categories:

- • Exam: General knowledge is evaluated on MMLU-Pro (Wang et al., 2024c), GPQADiamond (Rein et al., 2023), SuperGPQA (Team et al., 2025c), and LiveBench (White et al., 2025).
- • Mathematics: Mathematical reasoning is rigorously tested on AIME (2024/2025) (MAA, a,b), Beyond-AIME (ByteDance-Seed, 2025), HMMT25 (HMMT, 2025), CNMO 2024 (CNMO Committee, 2024), and IMO-AnswerBench (Luong et al., 2025).
- • Code: Pure text coding is evaluated via LiveCodeBench (2408-2505) (Jain et al., 2024).
- • Instruction Following: We use IFEval (Zhou et al., 2023), IFBench (Pyatkin et al., 2025), and MultiChallenge (Sirdeshmukh et al., 2025).
- • Subjective: Open-ended generation quality is assessed on Arena-Hard-V2 (Li et al., 2024) and WildBench (Lin et al., 2024).
- • Medical: Domain-specific knowledge is tested on HealthBench (Arora et al., 2025).

Inference Settings. We evaluate STEP3-VL-10B using a fixed configuration (temperature=1.0, top-p=1.0, top-k=0). By default, the model uses Sequential Reasoning (SeRe), generating thoughts wrapped in <think> and </think> tags before the answer, with a maximum length of 65,536 tokens. For complex perception and advanced reasoning tasks, we employ Parallel Coordinated Reasoning (PaCoRe) (Hu et al., 2026), which synthesizes 16 SeRe rollouts into a context for final inference (more details refer to Sec. B). In PaCoRe mode, the maximum length is extended to 131,072 tokens to support the expanded context, while other hyperparameters remain consistent.

Comparison Models. We benchmark STEP3-VL-10B against representative open-source models (7B–10B) including GLM-4.6V-Flash (9B) (Team et al., 2025d), Qwen3-VL-Thinking (8B) (Bai et al., 2025), InternVL-3.5 (8B) (Wang et al., 2025b), and MiMo-VL-RL-2508 (7B) (Team et al., 2025a). For scalability analysis, we compare against larger systems: GLM-4.6V (106BA12B) (Team et al., 2025d), Qwen3-VL (235B-A22B) (Bai et al., 2025), Gemini-2.5-Pro (Team,

- 2025a), and Seed-1.5-VL (Guo et al., 2025a).

###### 4.2. Multimodal Evaluation Results

In Table 1, we benchmark STEP3-VL-10B against strong open-source models within the 7B– 10B parameter range. The results indicate that STEP3-VL-10B establishes a new performance standard for compact models, securing the top position in almost all capability domains. We provide a detailed breakdown below.

STEM and Multimodal Reasoning. STEP3-VL-10B consistently outperforms competitive open-source models in the 7B–10B regime across all benchmarks targeting mathematical and scientific reasoning. STEP3-VL-10B achieves 78.11%/64.08% on MMMU (Standard/Pro) and notably, on MathVision, it surpasses strong baselines such as MiMo-VL-RL-2508 and Qwen3-VL by more than 10 points. These gains can be primarily attributed to sufficient pre-training and scaled RL compute.

Recognition and General VQA. STEP3-VL-10B consistently exhibits superior performance in visual recognition and VQA tasks, outperforming existing baselines across all evaluated benchmarks. Notably, STEP3-VL-10B achieves 92.05%/91.55% on MMBench (EN/CN), establishing the strongest performance among models within the 10B parameter scale. We attribute this exceptional performance to large-scale, high-quality multimodal pre-training, particularly the scaling of the 1.8B Perception Encoder.

- 2D / 3D Spatial Understanding. Despite the absence of specific data curation, STEP3-VL10B demonstrates remarkable spatial awareness and reasoning capabilities across both 2D and
- 3D environments. This emergent proficiency underscores its immense potential as a robust baseline for downstream actionable scenarios, such as embodied intelligence and robotic control.

OCR and Document Understanding. STEP3-VL-10B exhibits frontier-class document intelligence, achieving 86.75% on OCRBench and 89.35% on AI2D. This capability stems from our systematic OCR data construction, which combines extensive real-world collection with high-fidelity synthetic generation.

GUI Grounding and Interaction. In actionable intelligence, STEP3-VL-10B dominates the leaderboard with 92.61% on ScreenSpot-V2 and 59.02% on OSWorld-G. These margins validate our Trajectory Modeling approach, where training on granular action trajectories (e.g., CLICK, SCROLL) effectively grounds visual elements into executable actions, surpassing methods

Table 1 | Comparison with state-of-the-art open-source models (7B–10B) on multimodal benchmarks. The bold and underlined numbers indicate the best and second-best results, respectively. ∗ indicates results reported in the original papers.

Model STEP3-VL-10B

Benchmark

GLM-4.6V Qwen3-VL InternVL MiMo-VL

Flash Thinking 3.5 RL-2508 10B 9B 8B 8B 7B

MMMU 78.11 71.17 73.53 71.69 71.14 MMMU-Pro 64.08 59.93 60.94 59.11 60.29 MathVision 70.81 54.05 59.60 52.05 59.65 MathVista 83.97 82.85 78.50 76.78 79.86 LogicVista 66.89 60.29 64.37 54.03 63.37 DynaMath 56.39 48.40 45.11 39.47 51.65 ZeroBench (main) 1.00 0.50 0.50 0.75 0.50 ZeroBench (sub) 27.54 24.03 20.58 18.56 21.18 MathVerse (vision) 75.73 71.41 73.19 65.18 73.19 We-Math 73.03 61.86 67.31 51.26 63.24 VisuLogic 29.68 26.50 27.82 27.20 24.52 PhyX 59.45 52.28 57.67 50.51 56.00 HLE 10.73 3.82 5.98 4.51 5.90

STEM / Multimodal Reasoning

MMBench EN 92.05 91.04 90.55 88.20 89.91 MMBench CN 91.55 89.56 89.75 86.24 88.79 SimpleVQA 53.08 52.09 48.69 41.43 49.65 MMStar 77.48 74.26 73.58 69.83 72.93 HallusionBench 64.91 59.92 62.23 58.79 63.53 MMVP 68.16 63.33 57.17 51.33 63.50 ReMI 67.29 60.75 57.17 52.65 63.13 M3GIA 78.36 76.66 76.86 62.38 65.84 DoYouSeeMe 67.48 65.52 56.90 38.22 60.92

Recognition / General VQA

CountBench 88.75 90.22 88.85 82.18 83.60 CountQA 33.69 33.79 23.35 22.32 27.41 PixMo-Count 70.85 76.42 69.51 63.24 69.98

Counting

MM-MT-Bench 8.14 6.94 8.16 7.46 8.08 MIA-Bench 92.00 89.99 92.35 90.89 90.91 MM-IFEval 61.87 60.78 60.93 60.80 61.28

Instruction Following

HumanEval-V 66.05 29.26 26.94 24.31 31.96 Design2Code 79.55 25.93 72.21 64.15 82.54 Design2Code (hard) 54.69 19.37 48.75 46.25 59.38

Code

OCRBench 86.75 85.97 82.85 83.70 85.40 OmniOCR 76.98 80.24 75.53 70.97 74.38 CC-OCR (Multi-Lang-OCR) 76.59 70.85 69.25 66.42 72.06

OCR

BLINK 66.79 64.90 62.78 55.40 62.57 CVBench 83.49 86.01 84.81 77.52 82.04 MMSI-Bench 32.18 31.13 29.05 28.12 29.60 ERQA 48.87 45.13 44.31 38.88 41.94 OmniSpatial 51.58 49.41 46.56 47.49 46.74 All-Angles-Bench 57.21 53.24 45.88 45.29 51.62 MindCube-tiny 62.81 45.00 41.06 34.65 39.06 RealWorldQA 74.44 76.93 71.93 66.93 72.78 SpatialViz-Bench 45.51 33.79 35.15 32.42 28.94 STARE 61.75 56.45 54.95 48.20 55.06 CoreCognition 66.69 65.11 64.04 61.70 65.30 V* 82.85 83.51 81.02 66.89 83.38 ViewSpatial 46.14 43.26 45.20 35.96 40.19

2D / 3D Spatial Un-

derstanding

CharXiv (RQ) 59.52 59.70 53.48 47.15 59.32 AI2D 89.35 88.93 83.32 82.34 84.96 CSVQA 63.33 54.99 61.32 46.62 60.23 OmniDocBenchNED↓ 21.51 24.88 23.38 29.47 23.30

Document & Chart Understanding

ScreenSpot-Pro 51.55 45.68 46.60* 15.39 34.84 OSWorld-G 59.02 54.71 56.70* 31.91 50.54 ScreenSpot-V2 92.61 92.14 93.60* 84.02 90.82 MMBench-GUI (L2) 81.50 78.46 76.60 63.95 75.69

GUI Grounding

relying solely on static captioning. Notably, these gains are further amplified by RL integrated with a perception reward system, which significantly enhances the model’s ability to generalize in complex GUI environments.

###### 4.3. Text-Centric Evaluation Results

Table 2 illustrates that STEP3-VL-10B preserves high-fidelity linguistic intelligence while scaling multimodal training. Unlike prior VL models, STEP3-VL-10B effectively avoids the performance trade-off between text and vision modalities.

Table 2 | Comparison with SOTA open-source models (7B–10B) on text-centric benchmarks.

Model STEP3-VL-10B

Benchmark

GLM-4.6V Qwen3-VL InternVL MiMo-VL

Flash Thinking 3.5 RL-2508 10B 9B 8B 8B 7B

MMLU-Pro 76.02 72.30 77.09 76.03 73.81 GPQA-Diamond 70.83 49.37 67.55 65.12 59.97 SuperGPQA 50.38 42.95 49.52 42.35 45.69 LiveBench(2024-11-25) 69.71 44.11 70.79 55.62 54.35

Exam

- AIME2024 90.94 37.92 74.06 78.18 75.36

- AIME2025 87.66 33.02 62.92 62.50 66.51 HMMT25 78.18 19.17 45.21 35.78 47.34 CNMO2024 78.20 61.72 79.22 66.56 76.17 BeyondAIME 63.23 11.80 30.59 66.56 43.94 IMO-AnswerBench 62.12 22.62 38.69 35.00 48.44

Mathematics

Code LiveCodeBench (2408-2505) 75.77 22.17 51.05 45.90 39.65

IFEval 82.16 74.86 83.23 79.72 68.62 IFBench 43.28 27.47 36.65 28.14 23.47 MultiChallenge 62.64 42.49 49.82 37.73 44.69

Instruction Following

Arena-Hard-V2 58.57 9.26 47.34 15.57 28.59 WildBench 86.04 34.04 72.36 56.45 63.09

Subjective

Medical HealthBench 44.67 31.80 47.45 35.54 43.58

Mathematics and Code. STEP3-VL-10B significantly outpaces its counterparts in complex reasoning tasks. Its exceptional performance on challenging benchmarks like IMO-AnswerBench (62.12%) and LiveCodeBench (2408-2505) (75.77 %) serves as a strong testament to its robust logical inference capabilities, positioning it as a leading model for tasks requiring high-level problem-solving skills.

Human Alignment. The exceptional instruction-following capabilities and subjective performance of STEP3-VL-10B further reveal its superior alignment with human preferences, effectively bridging the usability gap traditionally associated with models of 10B size. Our internal Elo-based evaluation confirms that, STEP3-VL-10B achieves a preference score that matches significantly larger open-source models, demonstrating its potential for high-quality, real-world deployment.

###### 4.4. Comparison with Larger Models

To evaluate the performance ceiling of STEP3-VL-10B, we benchmark it against strong opensource (10×–20× larger) and closed-source flagships. As shown in Table 3, STEP3-VL-10B effectively bridges the gap between limited parameter scales (10B) and frontier intelligence. On standard benchmarks, STEP3-VL-10B outperforms GLM-4.6V (106B-A12B) across perception, recognition, and complex reasoning tasks, while remaining competitive with Qwen3-VL-

- Table 3 | Comparisons against models that are 10×–20× larger, as well as leading proprietary systems, on multimodal and text-centric benchmarks. SeRe and PaCoRe refer to Sequential

###### Reasoning and Parallel Coordinated Reasoning (Hu et al., 2026), respectively.

Model STEP3-VL-10B

Qwen3-VL Gemini-2.5 Seed-1.5-VL

Benchmark

GLM-4.6V

SeRe PaCoRe Thinking Pro Thinking 10B 10B 106B-A12B 235B-A22B — Multimodal Benchmarks

MMMU 78.11 80.11 75.20 78.70 83.89 79.11 MMMU-Pro 64.08 67.18 65.84 72.37 76.96 70.60 MathVision 70.81 75.95 63.50* 72.10 73.30* 68.70* MathVista 83.97 85.50 83.51 85.10 83.88 85.60 LogicVista 66.89 71.36 64.88 73.15 69.80 72.93 DynaMath 56.39 61.48 56.29 60.30 52.30 58.88 ZeroBench (main) 1.00 5.00 1.00 3.00 4.00 1.00 ZeroBench (sub) 27.54 29.94 29.04 28.40 33.53 31.74 MathVerse (vision) 75.73 78.30 72.84 76.65 78.30 77.79 We-Math 73.03 73.90 71.14 74.70 80.10 79.05 VisuLogic 29.68 32.70 28.30 31.80 31.40 34.30 PhyX 59.45 66.01 59.70 66.30 67.56 62.53

STEM / Multimodal Reasoning

MMBench EN 92.05 92.38 92.75 92.70 93.19 92.11 MMBench CN 91.55 91.96 91.88 91.80 93.13 91.76 SimpleVQA 53.08 54.64 57.95 59.30 66.85 64.72 MMStar 77.48 77.64 75.30 76.80 79.18 77.91 HallusionBench 64.91 64.54 60.63 65.58 65.63 64.13 MMVP 68.16 68.00 71.33 71.30 70.67 74.00 ReMI 67.29 69.12 64.42 74.70 71.69 72.19 M3GIA 78.33 73.50 78.72 81.00 83.11 83.22 DoYouSeeMe 67.48 68.54 67.50 72.89 71.19 71.94

Recognition / General VQA

CountBench 88.75 88.80 92.06 92.46 87.78 91.85 CountQA 33.69 38.29 36.32 45.62 38.02 48.89 PixMo-Count 70.85 71.61 76.47 79.80 75.54 83.38

Counting

OCRBench 86.75 89.00 86.20 87.30 85.90 85.20 OmniOCR 76.98 78.14 84.53 87.20 66.05 87.80 CC-OCR (Multi-Lang-OCR) 76.59 77.51 74.08 80.80 81.10 78.82

OCR

BLINK 66.79 67.39 68.17 67.12 72.01 71.54 CVBench 83.49 85.92 83.72 87.86 84.36 86.27 MMSI-Bench 32.18 36.40 30.80 32.50 40.40 30.60 ERQA 48.87 51.75 47.75 53.50 62.25 48.50 OmniSpatial 51.58 52.58 50.49 53.10 55.64 51.99 All-Angles-Bench 57.21 64.71 62.94 60.59 65.88 57.65 MindCube-tiny 62.81 68.58 52.83 47.58 58.92 39.83 RealWorldQA 74.44 75.56 77.78 78.80 77.78 79.61 SpatialViz-Bench 45.51 52.03 37.46 46.36 45.34 35.25 STARE 61.75 64.57 60.38 70.89 62.36 62.99 CoreCognition 66.69 71.54 69.50 72.66 78.78 72.38 V* 82.85 84.29 85.86 89.53 80.63 90.58 ViewSpatial 46.14 48.41 43.87 48.58 44.15 44.14

2D / 3D Spatial Understanding

Text-Centric Benchmarks

MMLU-Pro 76.02 77.09 79.96 83.75 86.45 83.39 GPQA-Diamond 70.83 73.99 69.19 77.68 84.06 71.91 SuperGPQA 50.38 53.15 53.28 64.20 65.00 60.50 LiveBench(2024-11-25) 69.71 71.69 62.75 80.14 76.34 65.62

Exam

- AIME2024 90.94 93.33 80.63 91.93 79.53 79.48

- AIME2025 87.66 94.43 71.88 83.59 83.96 64.06 HMMT25 78.18 92.14 57.29 67.71 65.68 51.30 CNMO2024 78.20 81.17 72.11 88.36 74.53 83.67 BeyondAIME 63.23 74.00 39.83 57.42 54.45 42.83 IMO-AnswerBench 62.12 76.66 51.25 69.25 72.00 44.75

Mathematics

Code LiveCodeBench (2408-2505) 75.77 76.43 48.71 69.45 72.01 57.10

Thinking (235B-A22B). Notably, it achieves 70.81% on MathVision, 87.66% on AIME 2025, and 77.48% on MMStar, demonstrating exceptional multimodal intelligence within a compact 10B budget.

We further explore the model’s limits by scaling test-time compute via the parallel coordinated reasoning setting. As shown in Table 3, the PaCoRe mode of STEP3-VL-10B consistently surpasses its standard SeRe mode and achieves frontier-level performance on several reasoningheavy and perception-centric benchmarks, even outperforming Gemini-2.5-Pro and Seed-1.5-VL. Specifically, STEP3-VL-10B achieves 80.11% on the multimodal understanding and reasoning benchmark MMMU. On the challenging multimodal mathematical reasoning benchmarks, MathVision and MathVista, it scores 75.95% and 85.50%, respectively. Furthermore, on representative visual recognition tasks such as MMBench and MMStar, it attains 92.17% (average on CN & EN) and 77.64%, respectively. These results demonstrate that STEP3-VL-10B has reached a leading level in multimodal perception and reasoning. Even more notably, on challenging high-level textual mathematics tasks like AIME2025 and HMMT25, it achieves remarkable scores of 94.43% and 92.14%, respectively. These results significantly outperform competing models, underscoring that intelligence is not strictly constrained by model size.

##### 5. Discussion

This section presents a two-fold analysis of the empirical findings that shaped STEP3-VL-10B. First, we distill key design insights regarding model architecture and optimization strategies that informed our final configurations. Second, we characterize the learning dynamics during RLVR and the emergent capabilities arising from subsequent RL scaling.

###### 5.1. Ablations and Design Insights

Vision Encoder Selection: PE-lang vs. DINOv3. We compare the Perception Encoder (PElang, 300M parameters specifically selected for ablation) with DINOv3 (ViT-large-16, 300M parameters) (Siméoni et al., 2025) as the vision backbone. While DINOv3 excels in pure vision tasks, it suffers from slow convergence in our multimodal setting due to the modality gap. Conversely, PE-lang explicitly pre-aligned with LLMs achieves superior data efficiency and benchmark performance (Tab. 4). This underscores that language alignment in the vision encoder remains a prerequisite for efficient VL modeling, irrespective of subsequent trillionscale generative training.

- Table 4 | Comparison of Vision Encoders: DINOv3 vs. PE-lang. Here, Omni. and SVQA denote OmniSpatial and SimpleVQA, respectively.

Vision Encoder Perception General

BLINK Omni. MMVP OCRBench MMStar SVQA CCBench V* MMMU ReMI

DINOv3 42.35 43.31 28.00 57.60 41.43 22.18 56.32 34.55 46.56 24.50 PE-lang (Ours) 41.19 43.57 32.00 70.10 42.10 21.15 59.39 37.17 47.67 26.08 Δ -1.16 +0.26 +4.00 +12.50 +0.67 -1.03 +3.07 +2.62 +1.11 +1.58

Optimizer Choice: Muon vs. AdamW. We investigate Muon (Keller, 2024), a matrix-wise optimizer utilizing Newton-Schulz iteration (Bernstein and Newhouse, 2024) to regularize weight topology. Muon effectively addresses the noise and imbalance inherent in large-scale multimodal data, yielding notable improvements in Tab. 5 for tail-knowledge tasks (+6.48% SimpleVQA). These results suggest Muon effectively reduces sensitivity to data scarcity. Despite

these capabilities, we exclude Muon from the final architecture due to initialization mismatch. Recent literature (Liu et al., 2025a) indicates that Muon is sensitive to weights initially optimized by element-wise methods like AdamW. In our setting, this necessitates a prolonged warmup period to stabilize the transition, which paradoxically limits overall training efficiency compared to a well-tuned AdamW baseline. We therefore leave a more thorough exploration of Muon for future work.

Table 5 | AdamW vs. Muon optimizers across selected benchmarks.

Optimizer Perception General

BLINK Omni. MMVP OCRBench MMStar SVQA CCB V* MMMU ReMI

Muon 41.14 42.73 32.00 67.70 44.58 27.08 60.72 36.65 47.56 22.23 Adam (Ours) 40.72 44.94 29.33 71.10 41.77 20.60 60.13 39.27 46.11 25.00 Δ -0.42 +2.21 -2.67 +3.40 -2.81 -6.48 -0.59 +2.62 -1.45 +2.77

Ablation for Deepstack. We investigate the utility of Deepstack (Meng et al., 2024), a depthextension technique successfully utilized in Qwen3-VL (Bai et al., 2025). While enabling Deepstack effectively accelerates training convergence, this optimization-level improvement does not translate into meaningful gains on downstream evaluation benchmarks, as shown in Tab. 6. Given the computational overhead versus the marginal utility, we exclude it from the final model configuration.

Table 6 | Ablation study of Deepstack architecture scaling.

Technique Perception General

BLINK Omni. MMVP OCRBench MMStar SVQA CCB V* MMMU ReMI

w/ DeepStack 40.72 42.92 26.00 71.20 43.31 28.66 63.94 36.65 47.44 26.96 w/o DeepStack (Ours) 40.61 43.57 31.33 69.30 42.44 25.20 62.80 38.22 47.78 26.96 Δ -0.11 +0.65 +5.33 -1.90 -0.87 -3.46 -1.14 +1.57 +0.34 +0.00

###### 5.2. RL Dynamics, Performance, and Emergence

Training Dynamics and Continuous Improvement. We track the evolution of RLVR over 600 training iterations, monitoring reward progression, average rollout length, and downstream performance across multimodal reasoning, recognition, OCR, and grounding tasks (assessed every 100 iterations). As illustrated in Fig. 2 (right) and Fig. 3, the model exhibits a robust two-phase growth trajectory: an initial rapid ascent in both rewards and metrics during the first 200 iterations, followed by a steady, linear increase. Remarkably, the reward consistently approaches 0.8 without observing saturation, mirrored by continuous gains in downstream metrics.

Distinct Length Dynamics. In contrast to the “sequential scaling” (i.e., the progressive lengthening of reasoning paths) typically observed in text-only RL (Guo et al., 2025b; Hu et al., 2025), the average rollout length in STEP3-VL-10B does not increase monotonically. Instead, it rises initially but eventually returns to its starting level (see Fig. 2, left). We identify this as a cancellation effect between two opposing scaling properties:

- 1. Reasoning Tasks (e.g., STEM, Puzzles): Exhibit standard sequential scaling, where model performance is positively correlated with the extension of inference-time compute (i.e., chain-of-thought length).
- 2. Deterministic Perception Tasks (e.g., Grounding, OCR): Characterized by length diminishment via policy refinement. Unlike the expansive “thinking” chains required for

4000

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.80

3500

0.75

RolloutTokens

0.70

3000

Reward

0.65

2500

0.60

2000

0.55

0.50

0 100 200 300 400 500 600

0 100 200 300 400 500 600

Training Steps

Training Steps

Figure 2 | RLVR dynamics. While the reward continuously increases without saturating (right), the average rollout tokens decrease towards the starting level after an initial rise (left).

MathVision

We-Math

###### MMMU-Pro

ReMI

72

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

75

65.0

69

62.5

68

72

66

60.0

69

64

63

57.5

66

60

100 300 500

100 300 500

100 300 500

100 300 500

Figure 3 | Trends of representative multimodal reasoning and perception metrics during RLVR. Evaluated every 100 iterations, performance mirrors the reward dynamics: rapid initial growth followed by steady improvement.

reasoning, RL gains in perception stem from entropy reduction (Cui et al., 2025b). To be specific, RL optimization induces a systematic collapse of the search space by pruning redundant exploratory tokens. This process concentrates the probability mass onto the singular deterministic mode, effectively converting high-temperature Pass@N exploration into robust Pass@1 accuracy (Yue et al., 2025). In this regime, shorter rollout lengths serve as a direct proxy for higher model confidence and sharpened perceptual focus.

The "Missing Trace" Hypothesis. Given that RL scaling strictly depends on the initial policy’s behavioral coverage (Fan et al., 2025; Wang et al., 2025d), we hypothesize that the perceptual "unscaling" stems from a critical data deficiency. While human visual cognition relies on iterative, coarse-to-fine processes (e.g., "glance-and-focus" or "try-error-correct" mechanisms akin to o3 (OpenAI, 2025c) or RePer (Wei et al., 2025b)), these internal mental states are rarely explicitly verbalized in training corpora. Consequently, the RL optimization landscape lacks the "cognitive traces" necessary to spontaneously incentivize sequential perceptual reasoning.

Emergence via PaCoRe. To address this limitation, we employ PaCoRe (Hu et al., 2026), a test-time scaling strategy that effectively explicates these implicit visual processes. Conceptually, this approach mirrors the proposal-then-refinement philosophy of foundational computer vision architectures, such as the Region Proposal Networks (RPN) in Faster R-CNN (Ren et al., 2016). In this framework, the model first samples multiple perceptual hypotheses (proposals); conditioning on these, the model then performs a multi-faceted synthesis to derive the final

Question: Locate the mish and Mennonite section in the image. <image> . . . Now, based on the original problem and reference responses above, please provide your own comprehensive solution.

Response: <think> Got it, let’s look at the references. The references have different x-coordinates (like 220, 228, 227, etc.) but the common y is 185 or around there. Let’s check the problem’s context. The references’ labels mention . . . Let’s see the reference 5 has 228, reference 6 227, reference 9 228, etc. To create a new point, we need to follow the logic. Let’s check the references’ labels and their coordinates’ variations.

. . . Maybe the key is to find a value that’s in the middle of the range. Alternatively, check the problem’s structure. . . .

Table 7 | A perception case of parallel coordinated reasoning: The model first aggregates reference answers, then cross-validates their validity, and ultimately identifies the underlying patterns of the final answer.

output. As illustrated in Tab. 7, this process spontaneously gives rise to complex, human-like verification behaviors during inference.

Crucially, this paradigm exhibits two distinctive scaling properties: (1) A steady, deliberate growth in response length, indicating the model’s ability to effectively allocate additional compute for hypothesis verification. (2) Significant performance gains in PaCoRe mode over the vanilla SeRe mode, as shown in Tab. 3. These gains are evident across benchmarks demanding intensive reasoning, such as MathVision (+5.14%) and DynaMath (+5.09%), as well as those requiring exhaustive perception (especially depends on high recall rate), including visual counting (CountQA, +4.6%), OCR (OCRBench, +2.25%), and especially, spatial understanding (All-Angles-Bench, +7.50%, SpatialViz-Bench, +6.52%).

Compress System 2 to System 1. PaCoRe, functioning as a primitive multi-agent framework, indeed enables perceptual scaling, where the proposer generates massive visual proposals in parallel, and the controller subsequently performs sequential cross-checking and self-verification. Looking forward, we aim to employ self-distillation to internalize these materialized, parallel coordinated reasoning traces. By injecting the logic of parallel deliberation directly into the model’s parameters, we seek to transform expensive “slow-thinking” (Kahneman, 2011) traces into high-fidelity, intrinsic intuition, ultimately fostering a more efficient and accurate perceptual foundation.

##### 6. Conclusion and Future Work

Anchored by a rigorously curated corpus of 1.2T multimodal tokens and sharpened via over 1k iterations of sequential and parallel coordinated RL, STEP3-VL-10B has achieved capabilities in perception, reasoning, and alignment that rival the strongest proprietary and open-source frontiers. Yet, raw capability is not synonymous with systemic maturity. On the trajectory toward comprehensive multimodal intelligence, we identify critical bottlenecks in computational density

and physical grounding. Our strategic roadmap aims to transform these limitations into the next engines of growth:

Maximizing Token Efficiency via Universal RL Scaling. We prioritize the principle that every unit of compute, during both training and inference, must contribute directly to intelligence density.

- • Shifting Compute from Pre-training to RL. RL scaling demonstrates continuous, saturationfree performance leaps that pre-training alone cannot sustain. We intend to aggressively pivot computational resources toward RL. By scaling universally in both depth (sequential reasoning) and width (parallel exploration), we aim to uncover high-value perception and reasoning traces, pushing the upper bounds of multimodal intelligence for models of all scales.
- • Optimizing Reasoning Density. We aim to bridge the gap between the high performance of extensive search and the low latency of standard inference. Our goal is to internalize the benefits of parallel exploration and eliminate redundant “over-thinking.” We envision a regime that continuously compresses reasoning paths, transforming explicit, coordinated search into efficient sequentiality, and ultimately distilling these capabilities into instinctive, “System 1”-like responses.

Bridging the Reality Gap. While the model excels in digital tasks, the "reality gap" remains the critical frontier. We posit that bridging this gap necessitates a paradigm shift: moving beyond passive data consumption to active physical grounding.

- • From Semantic to Physical World Models. We regard current text-based multi-agent synthesis as a foundational step—constructing a semantic world model. To achieve true embodiment, we must scale this synthesis to encompass massive video trajectories and sensorimotor action sequences. This unifies distinct modalities into a holistic world model that transcends linguistic logic to internalize physical causality and spatiotemporal dynamics.
- • Physics as the Ultimate Verifier. Current multimodal RL often relies on static or noisy proxy labels. We intend to integrate high-fidelity simulation environments where rewards are strictly governed by immutable physical laws. This shifts the learning paradigm from surface-level imitation to interaction-driven mastery, grounding the model’s reasoning in verifiable causality rather than statistical correlation.
- • Embodied Chain-of-Thought (E-CoT). We envision extending the reasoning context to explicitly model temporal dynamics and physical state transitions. By training the model to articulate “physical intuition” via predicting dynamics prior to action, we aim to develop agents capable of robust long-horizon planning in dynamic, open-world environments.

##### References

aallail. Nyu-book-eval-eg2 dataset. URL https://huggingface.co/datasets/aallail/ nyu_book_eval_eg2.

X. An, Y. Xie, K. Yang, W. Zhang, X. Zhao, Z. Cheng, Y. Wang, S. Xu, C. Chen, D. Zhu, et al. Llava-onevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025.

- R. K. Arora, J. Wei, R. S. Hicks, P. Bowman, J. Quiñonero-Candela, F. Tsimpourlas, M. Sharman, M. Shah, A. Vallone, A. Beutel, J. Heidecke, and K. Singhal. Healthbench: Evaluating large language models towards improved human health, 2025. URL https://arxiv.org/abs/ 2505.08775.
- S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren,

- X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie,
- Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang,

- B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu. Qwen3-vl technical report,

2025. URL https://arxiv.org/abs/2511.21631.

E. Bakouch, L. Ben Allal, A. Lozhkov, N. Tazi, L. Tunstall, C. M. Patiño, E. Beeching, A. Roucher, A. J. Reedi, Q. Gallouédec, K. Rasul, N. Habib, C. Fourrier, H. Kydlicek, G. Penedo, H. Larcher, M. Morlon, V. Srivastav, J. Lochner, X.-S. Nguyen, C. Raffel, L. von Werra, and T. Wolf. SmolLM3: smol, multilingual, long-context reasoner. https://huggingface.co/blog/sm ollm3, 2025.

A. Ben Abacha, S. A. Hasan, V. V. Datla, J. Liu, D. Demner-Fushman, and H. Müller. Vqa-med: Overview of the medical visual question answering task at imageclef 2019. In Working Notes of CLEF 2019, volume 2380 of CEUR Workshop Proceedings, Lugano, Switzerland, September 9-12 2019. CEUR-WS.org. URL https://ceur-ws.org/Vol-2380/paper_272.pdf.

J. Bernstein and L. Newhouse. Old optimizer, new norm: An anthology, 2024. URL https: //arxiv.org/abs/2409.20325.

- D. Bolya, P.-Y. Huang, P. Sun, J. H. Cho, A. Madotto, C. Wei, T. Ma, J. Zhi, J. Rajasegaran, H. Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181, 2025.

M. Byeon, B. Park, H. Kim, S. Lee, W. Baek, and S. Kim. Coyo-700m: Image-text pair dataset.

https://github.com/kakaobrain/coyo-dataset, 2022.

ByteDance-Seed. Beyondaime: Advancing math reasoning evaluation beyond high school olympiads. [https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME ](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME), 2025.

M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, and A. Joulin. Unsupervised learning of visual features by contrasting cluster assignments, 2021. URL https://arxiv.org/abs/ 2006.09882.

M. Chai, Z. Shen, C. Zhang, Y. Zhang, X. Wang, S. Dou, J. Kang, J. Zhang, and Q. Zhang. Docfusion: A unified framework for document parsing tasks, 2025. URL https://arxiv. org/abs/2412.12505.

J. Chen, L. Kong, H. Wei, C. Liu, Z. Ge, L. Zhao, J. Sun, C. Han, and X. Zhang. Onechart: Purify the chart structural extraction via one auxiliary token. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 147–155, 2024a.

L. Chen, J. Li, X. Dong, P. Zhang, Y. Zang, Z. Chen, H. Duan, J. Wang, Y. Qiao, D. Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024b.

- X. Cheng, W. Zhang, S. Zhang, J. Yang, X. Guan, X. Wu, X. Li, G. Zhang, J. Liu, Y. Mai, Y. Zeng, Z. Wen, K. Jin, B. Wang, W. Zhou, Y. Lu, T. Li, W. Huang, and Z. Li. Simplevqa: Multimodal factuality evaluation for multimodal large language models, 2025. URL https://arxiv.or g/abs/2502.13059.

- W.-L. Chiang, L. Zheng, Y. Sheng, A. N. Angelopoulos, T. Li, D. Li, H. Zhang, B. Zhu, M. Jordan, J. E. Gonzalez, and I. Stoica. Chatbot arena: An open platform for evaluating llms by human preference, 2024.

- C. Chou, L. Dunlap, K. Mashita, K. Mandal, T. Darrell, I. Stoica, J. E. Gonzalez, and W.-L. Chiang. Visionarena: 230k real world user-vlm conversations with preference labels. 2024. URL https://arxiv.org/abs/2412.08687.

CNMO Committee. Chinese national mathematical olympiad (cnmo), 2024. Accessed: 2025. CommonCrawl. Common crawl. URL https://commoncrawl.org/. C. Cui, T. Sun, M. Lin, T. Gao, Y. Zhang, J. Liu, X. Wang, Z. Zhang, C. Zhou, H. Liu, Y. Zhang,

- W. Lv, K. Huang, Y. Zhang, J. Zhang, J. Zhang, Y. Liu, D. Yu, and Y. Ma. Paddleocr 3.0 technical report, 2025a. URL https://arxiv.org/abs/2507.05595.

- G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, et al. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025b.

M. Deitke, C. Clark, S. Lee, R. Tripathi, Y. Yang, J. S. Park, M. Salehi, N. Muennighoff, K. Lo, L. Soldaini, J. Lu, T. Anderson, E. Bransom, K. Ehsani, H. Ngo, Y. Chen, A. Patel, M. Yatskar, C. Callison-Burch, A. Head, R. Hendrix, F. Bastani, E. VanderBilt, N. Lambert, Y. Chou, A. Chheda, J. Sparks, S. Skjonsberg, M. Schmitz, A. Sarnat, B. Bischoff, P. Walsh, C. Newell, P. Wolters, T. Gupta, K.-H. Zeng, J. Borchardt, D. Groeneveld, C. Nam, S. Lebrecht, C. Wittlif, C. Schoenick, O. Michel, R. Krishna, L. Weihs, N. A. Smith, H. Hajishirzi, R. Girshick, A. Farhadi, and A. Kembhavi. Molmo and pixmo: Open weights and open data for state-ofthe-art vision-language models, 2024. URL https://arxiv.org/abs/2409.17146.

- S. Ding, S. Wu, X. Zhao, Y. Zang, H. Duan, X. Dong, P. Zhang, Y. Cao, D. Lin, and J. Wang. Mm-ifengine: Towards multimodal instruction following. arXiv preprint arXiv:2504.07957, 2025.

- R.-Z. Fan, Z. Wang, and P. Liu. Megascience: Pushing the frontiers of post-training datasets for science reasoning. arXiv preprint arXiv:2507.16812, 2025.

- X. Fu, Y. Hu, B. Li, Y. Feng, H. Wang, X. Lin, D. Roth, N. A. Smith, W.-C. Ma, and R. Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.

- T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen, F. Huang, Y. Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14375–14385, 2024.

- E. Guha, R. Marten, S. Keh, N. Raoof, G. Smyrnis, H. Bansal, M. Nezhurina, J. Mercat, T. Vu, Z. Sprague, A. Suvarna, B. Feuer, L. Chen, Z. Khan, E. Frankel, S. Grover, C. Choi, N. Muennighoff, S. Su, W. Zhao, J. Yang, S. Pimpalgaonkar, K. Sharma, C. C.-J. Ji, Y. Deng, S. Pratt,

- V. Ramanujan, J. Saad-Falcon, J. Li, A. Dave, A. Albalak, K. Arora, B. Wulfe, C. Hegde,

- G. Durrett, S. Oh, M. Bansal, S. Gabriel, A. Grover, K.-W. Chang, V. Shankar, A. Gokaslan,

- M. A. Merrill, T. Hashimoto, Y. Choi, J. Jitsev, R. Heckel, M. Sathiamoorthy, A. G. Dimakis, and L. Schmidt. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.

D. Guo, F. Wu, F. Zhu, F. Leng, G. Shi, H. Chen, H. Fan, J. Wang, J. Jiang, J. Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025a.

D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025b.

X. He, Y. Zhang, L. Mou, E. Xing, and P. Xie. Pathvqa: 30000+ questions for medical visual

question answering. arXiv preprint arXiv:2003.10286, 2020. HMMT. Hmmt 2025, 2025. URL https://www.hmmt.org/. Accessed: 2025. J. Hu, Y. Zhang, Q. Han, D. Jiang, X. Zhang, and H.-Y. Shum. Open-reasoner-zero: An open

source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.

J. Hu, Y. Zhang, S. Shang, X. Yang, Y. Peng, Z. Huang, H. Zhou, X. Wu, J. Cheng, F. Wan, X. Kong, C. Yao, K. Yan, A. Huang, H. Zhou, Q. Han, Z. Ge, D. Jiang, X. Zhang, and H.-Y. Shum. Pacore: Learning to scale test-time compute with parallel coordinated reasoning, 2026. URL https://arxiv.org/abs/2601.05593.

- N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. Wang, A. Solar-Lezama, K. Sen, and I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

M. Jia, Z. Qi, S. Zhang, W. Zhang, X. Yu, J. He, H. Wang, and L. Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models, 2025. URL https: //arxiv.org/abs/2506.03135.

A. Jian, W. Qiu, X. Wang, P. Wang, Y. Hao, J. Pei, Y. Wei, Y. Peng, and X. Song. Csvqa: A chinese multimodal benchmark for evaluating stem reasoning capabilities of vlms, 2025. URL https://arxiv.org/abs/2505.24120.

Kaggle. Fcs dataset. URL https://www.kaggle.com/datasets/xuncngng/fsc147-0. D. Kahneman. Thinking, fast and slow. Farrar, Straus and Giroux, 2011. A. Kanade and T. Ganu. Do you see me : A multidimensional benchmark for evaluating visual

###### perception in multimodal llms, 2025. URL https://arxiv.org/abs/2506.02022.

M. Kazemi, N. Dikkala, A. Anand, P. Devic, I. Dasgupta, F. Liu, B. Fatemi, P. Awasthi, D. Guo,

- S. Gollapudi, and A. Qureshi. Remi: A dataset for reasoning with multiple images, 2024. URL https://arxiv.org/abs/2406.09175.

J. Keller. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/.

A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi. A diagram is worth a

dozen images, 2016. URL https://arxiv.org/abs/1603.07396.

- G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang, S. Yun, D. Han, and S. Park. Ocr-free document understanding transformer. In European Conference on Computer Vision (ECCV), 2022.

- A. Kuznetsova, H. Rom, N. Alldrin, J. Uijlings, I. Krasin, J. Pont-Tuset, S. Kamali, S. Popov, M. Malloci, A. Kolesnikov, T. Duerig, and V. Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, Mar. 2020. ISSN 1573-1405. doi: 10.1007/s112 63-020-01316-z. URL http://dx.doi.org/10.1007/s11263-020-01316-z.

- H. Laurençon, L. Tronchon, and V. Sanh. Unlocking the conversion of web screenshots into html code with the websight dataset, 2024.

D. Li, H. Li, Z. Wang, Y. Yan, H. Zhang, S. Chen, G. Hou, S. Jiang, W. Zhang, Y. Shen, W. Lu, and Y. Zhuang. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models, 2025a. URL https://arxiv.org/abs/2505.21500.

J. Li, D. Li, C. Xiong, and S. Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation, 2022. URL https://arxiv.org/abs/22 01.12086.

- J. LI, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. C. Huang, K. Rasul, L. Yu, A. Jiang, Z. Shen, Z. Qin, B. Dong, L. Zhou, Y. Fleureau, G. Lample, and S. Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-1.5](https://github.com/pro ject-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf), 2024.
- K. Li, Z. Meng, H. Lin, Z. Luo, Y. Tian, J. Ma, Z. Huang, and T.-S. Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use, 2025b. URL https://arxiv.or g/abs/2504.07981.
- L. Li, M. Bigverdi, J. Gu, Z. Ma, Y. Yang, Z. Li, Y. Choi, and R. Krishna. Unfolding spatial cognition: Evaluating multimodal models on visual simulations, 2025c. URL https://arxi v.org/abs/2506.04633.

- T. Li, W.-L. Chiang, E. Frick, L. Dunlap, T. Wu, B. Zhu, J. E. Gonzalez, and I. Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline, 2024. URL https://arxiv.org/abs/2406.11939.

- Y. Li, Q. Gao, T. Zhao, B. Wang, H. Sun, H. Lyu, R. D. Hawkins, N. Vasconcelos, T. Golan, D. Luo, and H. Deng. Core knowledge deficits in multi-modal language models, 2025d. URL https://arxiv.org/abs/2410.10855.

- B. Y. Lin, Y. Deng, K. Chandu, F. Brahman, A. Ravichander, V. Pyatkin, N. Dziri, R. L. Bras, and Y. Choi. Wildbench: Benchmarking llms with challenging tasks from real users in the wild,

2024. URL https://arxiv.org/abs/2406.04770.

T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollár. Microsoft coco: Common objects in context, 2015. URL https: //arxiv.org/abs/1405.0312.

- C. Liu, H. Wei, J. Chen, L. Kong, Z. Ge, Z. Zhu, L. Zhao, J. Sun, C. Han, and X. Zhang. Focus anywhere for fine-grained multi-page document understanding. arXiv preprint arXiv:2405.14295, 2024a.

J. Liu, J. Su, X. Yao, Z. Jiang, G. Lai, Y. Du, Y. Qin, W. Xu, E. Lu, J. Yan, Y. Chen, H. Zheng, Y. Liu, S. Liu, B. Yin, W. He, H. Zhu, Y. Wang, J. Wang, M. Dong, Z. Zhang, Y. Kang, H. Zhang,

- X. Xu, Y. Zhang, Y. Wu, X. Zhou, and Z. Yang. Muon is scalable for llm training, 2025a. URL https://arxiv.org/abs/2502.16982.

- X. Liu, W. Wang, Y. Yuan, J. tse Huang, Q. Liu, P. He, and Z. Tu. Insight over sight: Exploring the vision-knowledge conflicts in multimodal llms, 2025b. URL https://arxiv.org/abs/ 2410.08145.
- Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024b.

- Y. Liu, Z. Li, M. Huang, B. Yang, W. Yu, C. Li, X.-C. Yin, C.-L. Liu, L. Jin, and X. Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), Dec. 2024c. ISSN 1869-1919. doi: 10.1007/s11432-024-4235-6. URL http://dx.doi

###### .org/10.1007/s11432-024-4235-6.

- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

T. Luong, D. Hwang, H. H. Nguyen, G. Ghiasi, Y. Chervonyi, I. Seo, J. Kim, G. Bingham, J. Lee, S. Mishra, A. Zhai, C. H. Hu, H. Michalewski, J. Kim, J. Ahn, J. Bae, X. Song, T. H. Trinh, Q. V. Le, and J. Jung. Towards robust mathematical reasoning, 2025. URL https: //arxiv.org/abs/2511.01846.

- MAA. American invitational mathematics examination 2024, a.
- MAA. American invitational mathematics examination 2025, b.

A. Masry, P. Kavehzadeh, X. L. Do, E. Hoque, and S. Joty. Unichart: A universal vision-language pretrained model for chart comprehension and reasoning, 2023.

L. Meng, J. Yang, R. Tian, X. Dai, Z. Wu, J. Gao, and Y.-G. Jiang. Deepstack: Deeply stacking visual tokens is surprisingly simple and effective for lmms. Advances in Neural Information Processing Systems, 37:23464–23487, 2024.

###### OmniAI. Omni ocr benchmark. URL https://getomni.ai/blog/ocr-benchmark.

OpenAI. Introducing gpt-5.2, 2025a. URL https://openai.com/index/introducing-gpt

-5-2/.

OpenAI. Gpt-oss-120b and gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025b.

URL https://arxiv.org/abs/2508.10925. OpenAI. Introducing openai o3 and o4-mini. https://openai.com/index/introducing

-o3-and-o4-mini/, 2025c.

L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao, J. Shi, F. Wu, P. Chu, M. Liu, Z. Li, C. Xu, B. Zhang, B. Shi, Z. Tu, and C. He. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations, 2024. URL https://arxiv.org/abs/2412.07626.

- R. Paiss, A. Ephrat, O. Tov, S. Zada, I. Mosseri, M. Irani, and T. Dekel. Teaching clip to count to ten, 2023. URL https://arxiv.org/abs/2302.12066.

- J. Poznanski, A. Rangapur, J. Borchardt, J. Dunkelberger, R. Huff, D. Lin, A. Rangapur, C. Wilhelm, K. Lo, and L. Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models, 2025. URL https://arxiv.org/abs/2502.18443.

- V. Pyatkin, S. Malik, V. Graf, H. Ivison, S. Huang, P. Dasigi, N. Lambert, and H. Hajishirzi. Generalizing verifiable instruction following, 2025.

- Y. Qian, H. Ye, J.-P. Fauconnier, P. Grasch, Y. Yang, and Z. Gan. Mia-bench: Towards better instruction following evaluation of multimodal llms, 2025. URL https://arxiv.org/abs/ 2407.01509.

- R. Qiao, Q. Tan, G. Dong, M. Wu, C. Sun, X. Song, Z. GongQue, S. Lei, Z. Wei, M. Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.

D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

- S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks, 2016. URL https://arxiv.org/abs/1506.01497.

J. Roberts, M. R. Taesiri, A. Sharma, A. Gupta, S. Roberts, I. Croitoru, S.-V. Bogolin, J. Tang,

- F. Langer, V. Raina, V. Raina, H. Xiong, V. Udandarao, J. Lu, S. Chen, S. Purkis, T. Yan, W. Lin,
- G. Shin, Q. Yang, A. T. Nguyen, D. I. Atkinson, A. Baranwal, A. Coca, M. Dang, S. Dziadzio, J. D. Kunz, K. Liang, A. Lo, B. Pulfer, S. Walton, C. Yang, K. Han, and S. Albanie. Zerobench: An impossible visual benchmark for contemporary large multimodal models, 2025. URL https://arxiv.org/abs/2502.09696.

SakiRinn. Locount dataset. URL https://github.com/SakiRinn/mmdetection-locount. C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes,

- A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt,

- R. Kaczmarczyk, and J. Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. URL https://arxiv.org/abs/2210.08402.

J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.

- H. Shen, T. Wu, Q. Han, Y. Hsieh, J. Wang, Y. Zhang, Y. Cheng, Z. Hao, Y. Ni, X. Wang, et al. Phyx: Does your model have the" wits" for physical reasoning? arXiv preprint arXiv:2505.15929, 2025.

- B. Shi, C. Yao, M. Liao, M. Yang, P. Xu, L. Cui, S. Belongie, S. Lu, and X. Bai. Icdar2017 competition on reading chinese text in the wild (rctw-17). In 2017 14th iapr international conference on document analysis and recognition (ICDAR), volume 1, pages 1429–1434. IEEE, 2017.

- C. Si, Y. Zhang, R. Li, Z. Yang, R. Liu, and D. Yang. Design2code: Benchmarking multimodal code generation for automated front-end engineering, 2025. URL https://arxiv.org/ab s/2403.03163.

- O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, and P. Bojanowski. Dinov3, 2025. URL https://arxiv.org/abs/2508.10104.

- V. Sirdeshmukh, K. Deshpande, J. Mols, L. Jin, E.-Y. Cardona, D. Lee, J. Kritz, W. Primack, S. Yue, and C. Xing. Multichallenge: A realistic multi-turn conversation evaluation benchmark challenging to frontier llms, 2025. URL https://arxiv.org/abs/2501.17399.
- W. Song, Y. Li, J. Xu, G. Wu, L. Ming, K. Yi, W. Luo, H. Li, Y. Du, F. Guo, and K. Yu. M3gia: A cognition inspired multilingual and multimodal general intelligence ability benchmark, 2024. URL https://arxiv.org/abs/2406.05343.

J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- H. R. Sujet AI, Allaa Boutaleb. Sujet-finance-qa-vision-100k: A large-scale dataset for financial document vqa, 2024. URL https://huggingface.co/datasets/sujet-ai/Sujet-Fin ance-QA-Vision-100k.

- J. S. Tamarapalli, R. Grover, N. Pande, and S. Yerramilli. Countqa: How well do mllms count in the wild?, 2025. URL https://arxiv.org/abs/2508.06585.
- K. Tang, W.-L. Chiang, and A. N. Angelopoulos. Arena explorer: A topic modeling pipeline for llm evals & analytics, 2025.

C. Team, Z. Yue, Z. Lin, Y. Song, W. Wang, S. Ren, S. Gu, S. Li, P. Li, L. Zhao, L. Li, K. Bao, H. Tian, H. Zhang, G. Wang, D. Zhu, Cici, C. He, B. Ye, B. Shen, Z. Zhang, Z. Jiang, Z. Zheng, Z. Song, Z. Luo, Y. Yu, Y. Wang, Y. Tian, Y. Tu, Y. Yan, Y. Huang, X. Wang, X. Xu, X. Song,

- X. Zhang, X. Yong, X. Zhang, X. Deng, W. Yang, W. Ma, W. Lv, W. Zhuang, W. Liu, S. Deng, S. Liu, S. Chen, S. Yu, S. Liu, S. Wang, R. Ma, Q. Wang, P. Wang, N. Chen, M. Zhu, K. Zhou, K. Zhou, K. Fang, J. Shi, J. Dong, J. Xiao, J. Xu, H. Liu, H. Xu, H. Qu, H. Zhao, H. Lv, G. Wang,

- D. Zhang, D. Zhang, D. Zhang, C. Ma, C. Liu, C. Cai, and B. Xia. Mimo-vl technical report, 2025a. URL https://arxiv.org/abs/2506.03569.

- G. Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025a. URL https://arxiv.org/abs/2507.062 61.

###### G. Team. Gemini 3 pro: the frontier of vision ai, 2025b. URL https://blog.google/techno logy/developers/gemini-3-pro-vision/.

- G. R. Team, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, T. Armstrong, A. Balakrishna, R. Baruch, M. Bauza, M. Blokzijl, S. Bohez, K. Bousmalis, A. Brohan, T. Buschmann, A. Byravan, S. Cabi, K. Caluwaerts, F. Casarini, O. Chang, J. E. Chen, X. Chen, H.-T. L. Chiang, K. Choromanski, D. D’Ambrosio, S. Dasari, T. Davchev, C. Devin, N. D. Palo, T. Ding, A. Dostmohamed, D. Driess, Y. Du, D. Dwibedi, M. Elabd, C. Fantacci, C. Fong, E. Frey, C. Fu, M. Giustina, K. Gopalakrishnan, L. Graesser, L. Hasenclever, N. Heess, B. Hernaez, A. Herzog,

- R. A. Hofer, J. Humplik, A. Iscen, M. G. Jacob, D. Jain, R. Julian, D. Kalashnikov, M. E. Karagozler, S. Karp, C. Kew, J. Kirkland, S. Kirmani, Y. Kuang, T. Lampe, A. Laurens, I. Leal, A. X. Lee, T.-W. E. Lee, J. Liang, Y. Lin, S. Maddineni, A. Majumdar, A. H. Michaely, R. Moreno, M. Neunert, F. Nori, C. Parada, E. Parisotto, P. Pastor, A. Pooley, K. Rao, K. Reymann, D. Sadigh,
- S. Saliceti, P. Sanketi, P. Sermanet, D. Shah, M. Sharma, K. Shea, C. Shu, V. Sindhwani, S. Singh,

- R. Soricut, J. T. Springenberg, R. Sterneck, R. Surdulescu, J. Tan, J. Tompson, V. Vanhoucke, J. Varley, G. Vesom, G. Vezzani, O. Vinyals, A. Wahid, S. Welker, P. Wohlhart, F. Xia, T. Xiao,

- A. Xie, J. Xie, P. Xu, S. Xu, Y. Xu, Z. Xu, Y. Yang, R. Yao, S. Yaroshenko, W. Yu, W. Yuan, J. Zhang, T. Zhang, A. Zhou, and Y. Zhou. Gemini robotics: Bringing ai into the physical world, 2025b. URL https://arxiv.org/abs/2503.20020.

H. Team. Humanity’s last exam, 2025c. URL https://arxiv.org/abs/2501.14249.

M.-A.-P. Team, X. Du, Y. Yao, K. Ma, B. Wang, T. Zheng, K. Zhu, M. Liu, Y. Liang, X. Jin, Z. Wei, C. Zheng, K. Deng, S. Guo, S. Jia, S. Jiang, Y. Liao, R. Li, Q. Li, S. Li, Y. Li, Y. Li, D. Ma, Y. Ni, H. Que, Q. Wang, Z. Wen, S. Wu, T. Xing, M. Xu, Z. Yang, Z. M. Wang, J. Zhou, Y. Bai, X. Bu, C. Cai, L. Chen, Y. Chen, C. Cheng, T. Cheng, K. Ding, S. Huang, Y. Huang, Y. Li, Y. Li, Z. Li, T. Liang, C. Lin, H. Lin, Y. Ma, Z. Peng, Z. Peng, Q. Qi, S. Qiu, X. Qu, Y. Tan, Z. Wang, C. Wang, H. Wang, Y. Wang, Y. Wang, J. Xu, K. Yang, R. Yuan, Y. Yue, T. Zhan, C. Zhang, J. Zhang,

- X. Zhang, X. Zhang, Y. Zhang, Y. Zhao, X. Zheng, C. Zhong, Y. Gao, Z. Li, D. Liu, Q. Liu, T. Liu, S. Ni, J. Peng, Y. Qin, W. Su, G. Wang, S. Wang, J. Yang, M. Yang, M. Cao, X. Yue, Z. Zhang, W. Zhou, J. Liu, Q. Lin, W. Huang, and G. Zhang. Supergpqa: Scaling llm evaluation across 285 graduate disciplines, 2025c. URL https://arxiv.org/abs/2502.14739.

- S. Team. Step-3 is large yet affordable: Model-system co-design for cost-effective decoding, 2025d. URL https://arxiv.org/abs/2507.19427.

- V. Team, W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, L. Pan, S. Duan,
- W. Wang, Y. Wang, Y. Cheng, Z. He, Z. Su, Z. Yang, Z. Pan, A. Zeng, B. Wang, B. Chen, B. Shi, C. Pang, C. Zhang, D. Yin, F. Yang, G. Chen, J. Xu, J. Zhu, J. Chen, J. Chen, J. Chen, J. Lin,

- J. Wang, J. Chen, L. Lei, L. Gong, L. Pan, M. Liu, M. Xu, M. Zhang, Q. Zheng, S. Yang, S. Zhong, S. Huang, S. Zhao, S. Xue, S. Tu, S. Meng, T. Zhang, T. Luo, T. Hao, T. Tong, W. Li, W. Jia, X. Liu,

- X. Zhang, X. Lyu, X. Fan, X. Huang, Y. Wang, Y. Xue, Y. Wang, Y. Wang, Y. An, Y. Du, Y. Shi,
- Y. Huang, Y. Niu, Y. Wang, Y. Yue, Y. Li, Y. Zhang, Y. Wang, Y. Wang, Y. Zhang, Z. Xue, Z. Hou,
- Z. Du, Z. Wang, P. Zhang, D. Liu, B. Xu, J. Li, M. Huang, Y. Dong, and J. Tang. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025d. URL https://arxiv.org/abs/2507.01006.

- P. Tong, E. Brown, P. Wu, S. Woo, A. J. V. IYER, S. C. Akula, S. Yang, J. Yang, M. Middepogu, Z. Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024a.

- S. Tong, E. Brown, P. Wu, S. Woo, M. Middepogu, S. C. Akula, J. Yang, S. Yang, A. Iyer, X. Pan, Z. Wang, R. Fergus, Y. LeCun, and S. Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024b. URL https://arxiv.org/abs/2406.16860.

- S. Tong, Z. Liu, Y. Zhai, Y. Ma, Y. LeCun, and S. Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms, 2024c. URL https://arxiv.org/abs/2401.06209.

- B. Wang, C. Xu, X. Zhao, L. Ouyang, F. Wu, Z. Zhao, R. Xu, K. Liu, Y. Qu, F. Shang, B. Zhang, L. Wei, Z. Sui, W. Li, B. Shi, Y. Qiao, D. Lin, and C. He. Mineru: An open-source solution for precise document content extraction, 2024a. URL https://arxiv.org/abs/2409.18839.

- K. Wang, J. Pan, W. Shi, Z. Lu, M. Zhan, and H. Li. Measuring multimodal mathematical reasoning with math-vision dataset, 2024b. URL https://arxiv.org/abs/2402.14804.

- S. Wang, L. Sun, C. Deng, K. Shao, M. Pei, Z. Tian, H. Zhang, and J. Wang. Spatialviz-bench: Automatically generated spatial visualization reasoning tasks for mllms. arXiv e-prints, pages arXiv–2507, 2025a.

- W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, Z. Wang, Z. Chen, H. Zhang, G. Yang, H. Wang, Q. Wei, J. Yin, W. Li, E. Cui, G. Chen, Z. Ding, C. Tian, Z. Wu, J. Xie, Z. Li, B. Yang, Y. Duan, X. Wang, Z. Hou, H. Hao, T. Zhang, S. Li, X. Zhao, H. Duan, N. Deng, B. Fu, Y. He, Y. Wang, C. He, B. Shi, J. He, Y. Xiong, H. Lv, L. Wu, W. Shao, K. Zhang, H. Deng, B. Qi, J. Ge, Q. Guo, W. Zhang, S. Zhang, M. Cao, J. Lin, K. Tang, J. h. Gao, H. Huang,

- Y. Gu, C. Lyu, H. Tang, R. Wang, H. Lv, W. Ouyang, L. Wang, M. Dou, X. Zhu, T. Lu, D. Lin, J. Dai, W. Su, B. Zhou, K. Chen, Y. Qiao, W. Wang, and G. Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b.

- X. Wang, Z. Wu, J. Xie, Z. Ding, B. Yang, Z. Li, Z. Liu, Q. Li, X. Dong, Z. Chen, W. Wang, X. Zhao, J. Chen, H. Duan, T. Xie, C. Yang, S. Su, Y. Yu, Y. Huang, Y. Liu, X. Zhang, Y. Zhang, X. Yue, W. Su, X. Zhu, W. Shen, J. Dai, and W. Wang. Mmbench-gui: Hierarchical multi-platform evaluation framework for gui agents, 2025c. URL https://arxiv.org/abs/2507.19478.
- Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, NeurIPS 2024, 2024c.

- Z. Wang, M. Xia, L. He, H. Chen, Y. Liu, R. Zhu, K. Liang, X. Wu, H. Liu, S. Malladi, A. Chevalier,

- S. Arora, and D. Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms, 2024d. URL https://arxiv.org/abs/2406.18521.

- Z. Wang, F. Zhou, X. Li, and P. Liu. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512, 2025d.

- H. Wei, C. Liu, J. Chen, J. Wang, L. Kong, Y. Xu, Z. Ge, L. Zhao, J. Sun, Y. Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.

- H. Wei, Y. Sun, and Y. Li. Deepseek-ocr: Contexts optical compression, 2025a. URL https: //arxiv.org/abs/2510.18234.

- Y. Wei, L. Zhao, K. Lin, E. Yu, Y. Peng, R. Dong, J. Sun, H. Wei, Z. Ge, X. Zhang, et al. Perception in reflection. arXiv preprint arXiv:2504.07165, 2025b.

- Y. Wei, L. Zhao, J. Sun, K. Lin, J. Yin, J. Hu, Y. Zhang, E. Yu, H. Lv, Z. Weng, et al. Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning. arXiv preprint arXiv:2507.05255, 2025c.

- C. White, S. Dooley, M. Roberts, A. Pal, B. Feuer, S. Jain, R. Shwartz-Ziv, N. Jain, K. Saifullah, S. Dey, Shubh-Agrawal, S. S. Sandha, S. V. Naidu, C. Hegde, Y. LeCun, T. Goldstein,

- W. Neiswanger, and M. Goldblum. Livebench: A challenging, contamination-free LLM benchmark. In The Thirteenth International Conference on Learning Representations, 2025.

L. Wiedmann, O. Zohar, A. Mahla, X. Wang, R. Li, T. Frere, L. von Werra, A. R. Gosthipaty, and A. Marafioti. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025.

P. Wu and S. Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Z. Wu, Z. Wu, F. Xu, Y. Wang, Q. Sun, C. Jia, K. Cheng, Z. Ding, L. Chen, P. P. Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024.

- X.AI. Grok-2 beta release. https://x.ai/blog/grok-2, 2024. Accessed on: 2024-07-02.

- R. Xia, B. Zhang, H. Peng, H. Ye, X. Yan, P. Ye, B. Shi, J. Yan, and Y. Qiao. Structchart: Perception, structuring, reasoning for visual chart understanding. arXiv preprint arXiv:2309.11268, 2023.

- Y. Xiao, E. Sun, T. Liu, and W. Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024. URL https://arxiv.org/abs/2407.04973.

C. Xie, H. Cai, J. Li, F. Kong, X. Wu, J. Song, H. Morimitsu, L. Yao, D. Wang, X. Zhang, D. Leng, B. Zhang, X. Ji, and Y. Deng. Ccmb: A large-scale chinese cross-modal benchmark. In Proceedings of the 31st ACM International Conference on Multimedia, page 4219–4227. ACM, Oct. 2023. doi: 10.1145/3581783.3611877. URL http://dx.doi.org/10.1145/3581783.3 611877.

- T. Xie, J. Deng, X. Li, J. Yang, H. Wu, J. Chen, W. Hu, X. Wang, Y. Xu, Z. Wang, Y. Xu, J. Wang,

- D. Sahoo, T. Yu, and C. Xiong. Scaling computer-use grounding via user interface decomposition and synthesis, 2025. URL https://arxiv.org/abs/2505.13227.

- W. Xu, J. Wang, W. Wang, Z. Chen, W. Zhou, A. Yang, L. Lu, H. Li, X. Wang, X. Zhu, W. Wang, J. Dai, and J. Zhu. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models, 2025. URL https://arxiv.org/abs/2504.15279.

- H. Yan, J. Wang, X. Huang, Y. Shen, Z. Meng, Z. Fan, K. Tan, J. Gao, L. Shi, M. Yang, et al. Step-gui technical report. arXiv preprint arXiv:2512.15431, 2025.

- A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng,

- D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang,

- J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang,

- W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report, 2025a. URL https://arxiv.org/abs/2505.09388.

- S. Yang, R. Xu, Y. Xie, S. Yang, M. Li, J. Lin, C. Zhu, X. Chen, H. Duan, X. Yue, D. Lin, T. Wang, and J. Pang. Mmsi-bench: A benchmark for multi-image spatial intelligence, 2025b. URL https://arxiv.org/abs/2505.23764.

- Y. Yang, A. Patel, M. Deitke, T. Gupta, L. Weihs, A. Head, M. Yatskar, C. Callison-Burch, R. Krishna, A. Kembhavi, and C. Clark. Scaling text-rich image understanding via code-guided synthetic multimodal data generation, 2025c. URL https://arxiv.org/abs/2502.14846.
- Z. Yang, J. Tang, Z. Li, P. Wang, J. Wan, H. Zhong, X. Liu, M. Yang, P. Wang, S. Bai, L. Jin, and J. Lin. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy, 2024. URL https://arxiv.org/abs/2412.02210.

C. Yao, X. Bai, W. Liu, Y. Ma, and Z. Tu. Detecting texts of arbitrary orientations in natural images. 2012. URL https://pages.ucsd.edu/~ztu/publication/cvpr12_textdete ction.pdf.

- F. Yao, L. Liu, D. Zhang, C. Dong, J. Shang, and J. Gao. Your efficient rl framework secretly brings you off-policy rl training, Aug. 2025. URL https://fengyao.notion.site/off-p olicy-rl.

C.-H. Yeh, C. Wang, S. Tong, T.-Y. Cheng, R. Wang, T. Chu, Y. Zhai, Y. Chen, S. Gao, and Y. Ma. Seeing from another perspective: Evaluating multi-view understanding in mllms, 2025. URL https://arxiv.org/abs/2504.15280.

- B. Yin, Q. Wang, P. Zhang, J. Zhang, K. Wang, Z. Wang, J. Zhang, K. Chandrasegaran, H. Liu, R. Krishna, S. Xie, M. Li, J. Wu, and L. Fei-Fei. Spatial mental modeling from limited views,

###### 2025. URL https://arxiv.org/abs/2506.21458.

- K. Ying, F. Meng, J. Wang, Z. Li, H. Lin, Y. Yang, H. Zhang, W. Zhang, Y. Lin, S. Liu, J. Lei, Q. Lu, R. Chen, P. Xu, R. Zhang, H. Zhang, P. Gao, Y. Wang, Y. Qiao, P. Luo, K. Zhang, and W. Shao. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi, 2024. URL https://arxiv.org/abs/2404.16006.

- E. Yu, L. Zhao, Y. Wei, J. Yang, D. Wu, L. Kong, H. Wei, T. Wang, Z. Ge, X. Zhang, et al. Merlin: Empowering multimodal llms with foresight minds. In European Conference on Computer Vision, pages 425–443. Springer, 2024.

- E. Yu, K. Lin, L. Zhao, Y. Wei, Z. Zhu, H. Wei, J. Sun, Z. Ge, X. Zhang, J. Wang, et al. Unhackable temporal rewarding for scalable video mllms. arXiv preprint arXiv:2502.12081, 2025a.

- E. Yu, K. Lin, L. Zhao, J. Yin, Y. Wei, Y. Peng, H. Wei, J. Sun, C. Han, Z. Ge, et al. Perception-r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025b.

- Y. Yuan, X. Liu, W. Dikubab, H. Liu, Z. Ji, Z. Wu, and X. Bai. Syntax-aware network for handwritten mathematical expression recognition. arXiv preprint arXiv:2203.01601, 2022.

- X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.

- Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

R. Zellers, Y. Bisk, A. Farhadi, and Y. Choi. From recognition to cognition: Visual commonsense

###### reasoning, 2019. URL https://arxiv.org/abs/1811.10830.

- F. Zhang, L. Wu, H. Bai, G. Lin, X. Li, X. Yu, Y. Wang, B. Chen, and J. Keung. Humaneval-v: Benchmarking high-level visual reasoning with complex diagrams in coding tasks, 2025. URL https://arxiv.org/abs/2410.12381.

R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K.-W. Chang, P. Gao, and H. Li. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems?,

###### 2024. URL https://arxiv.org/abs/2403.14624.

J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, and L. Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv.org/abs/2311.07911.

- C. Zou, X. Guo, R. Yang, J. Zhang, B. Hu, and H. Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models, 2024.

##### 7. Author List

All authors are listed in alphabetical order by their first names. † indicates project leaders.

Core Contributors Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun†, Jingcheng Hu, Kangheng Lin, Liang Zhao†, Mitt Huang, Song Yuan, Wenwen Qu, Xiangfeng Wang, Yanlin Lai, Yingxiu Zhao, Yinmin Zhang, Yukang Shi, Yuyang Chen, Zejia Weng, Ziyang Meng

Contributors Ang Li, Aobo Kong, Bo Dong, Changyi Wan, David Wang, Di Qi, Dingming Li, En Yu, Guopeng Li, Haiquan Yin, Han Zhou, Hanshan Zhang, Haolong Yan, Hebin Zhou, Hongbo Peng, Jiaran Zhang, Jiashu Lv, Jiayi Fu, Jie Cheng, Jie Zhou, Jisheng Yin, Jingjing Xie, Jingwei Wu, Jun Zhang, Junfeng Liu, Kaijun Tan, Kaiwen Yan, Liangyu Chen, Lina Chen, Mingliang Li, Qian Zhao, Quan Sun, Shaoliang Pang, Shengjie Fan, Shijie Shang, Siyuan Zhang, Tianhao You, Wei Ji, Wuxun Xie, Xiaobo Yang, Xiaojie Hou, Xiaoran Jiao, Xiaoxiao Ren, Xiangwen Kong, Xin Huang, Xin Wu, Xing Chen, Xinran Wang, Xuelin Zhang, Yana Wei, Yang Li, Yanming Xu, Yeqing Shen, Yuang Peng, Yue Peng, Yu Zhou, Yusheng Li, Yuxiang Yang, Yuyang Zhang, Zhe Xie, Zhewei Huang, Zhenyi Lu, Zhimin Fan, Zihui Cheng

Sponsors Daxin Jiang, Qi Han, Xiangyu Zhang, Yibo Zhu, Zheng Ge

##### Appendix

##### A. More Results

Qualitative Results. This section presents a selection of qualitative examples illustrating the capabilities of STEP3-VL-10B through responses generated from various prompts. The provided cases underscore the model’s superior ability of fine-grained perception and structured logical reasoning. More importantly, it demonstrates a high degree of cognitive systematicity, effectively integrating domain knowledge to execute multi-step tasks.

###### Case 1: Visual Pattern Matching and Sequential Decoding

[Figure 3]

Figure 4 | Morse Code Reference

User Prompt 根据该表生成"STEPFUN"的编码

###### Model Thinking Process

用户现在需要把每个字母转换成莫尔斯电码，先看每个字母在表中的编码。首先分解"STEPFUN"每 个字母：S、T、E、P、F、U、N。先逐个找：

. . .

- • S：看右边第三列第一个，S是• • •（三个点）
- • T：第三列第二个，T是−（一个短横）
- • E：左边第一列第五个，E是•（一个点）
- • P：中间列（J到R）中P对应的符号是• − −•（点、短横、短横、点）
- • F：最左侧列第六行，F的符号是• • −•（两个点、一个短横、一个点）
- • U：最右侧列第三行，U的符号是• • −（两个点、一个短横）
- • N：中间列（J到R）第五行，N的符号是−•（一个短横、一个点）

. . .

再将每个字母的编码按顺序拼接（字母间用空格分隔更清晰），"STEPFUN"的莫尔斯电码为：• •

- • − • • − − • • • − • • • − −• Final Answer

- • • • − • • − − • • • − • • • − −•

###### Case 2: Fine-grained Visual Counting and Semantic Filtering

[Figure 4]

Figure 5 | Screenshot of the Compiler

###### User Prompt

The screenshot shows a snapshot of a LaTeX text editor a student uses for writing their PhD dissertation. What is the minimum number of chapter .tex files the student has opened?

Model Thinking Process So, let’s look at the tabs at the top of the editor.

. . .

The visible tabs are:

- • chapte. . . (leftmost, maybe chapter1.tex as the active file in the center)
- • chapte. . . (next)
- • thesis. . . (this is thesis.tex, not a chapter)
- • chapte. . . (next)
- • chapte. . . (next)
- • chapte. . . (last one before the dark theme part)

. . .

To determine the minimum number of chapter .tex files opened, we analyze the tabs at the top of the editor.

- 1. Identify tabs: The image shows multiple tabs labeled chapte. . . (truncated for "chapter. . . ") and one tab labeled thesis. . . .
- 2. Exclude non-chapter tabs: The thesis. . . tab is not a chapter file, so we count only the chapte. . . tabs.
- 3. Count chapter tabs: There are 5 visible chapte. . . tabs (excluding the thesis. . . tab).

###### Final Answer 5

###### Case 3: Low-Resolution Visual Detail Recognition

[Figure 5]

Figure 6 | Out-of-focus Target

###### User Prompt

What is the brand name of the monitor in the image? What is its encoding with a Caesar cipher with a left shift of 5. Give your answer as a combined string.

Model Thinking Process Got it, let’s break this down. First, find the monitor’s brand.

. . .

The logo on the monitor’s base is HP. Then, we need to do a Caesar cipher with a left shift of 5.

. . .

- • First, H is the 8th letter (A=1, B=2. . . H=8). Left shift 5: 8-5=3 → C.
- • Then, P is the 16th letter. 16-5=11 → K. So "HP" shifted left 5 is "CK".

###### Final Answer HPCK

###### Case 4: Topological Graph Analysis and Minimum Weight Search

[Figure 6]

Figure 7 | Weighted Graph

User Prompt How many edges have the shortest length?

###### Model Thinking Process

To determine the number of edges with the shortest length, we first identify the smallest edge weight in the diagram.

. . .

The smallest weight is 1. Next, we count all edges labeled with 1:

. . .

- • 𝐴 → 𝑇: labeled 1
- • 𝑇 → 𝑋: labeled 1
- • 𝑃 → 𝑂: labeled 1
- • 𝑊 → 𝑋: labeled 1

. . .

No other edges have a weight of 1. Thus, there are 4 edges with the shortest length.

###### Final Answer

- 4

##### B. Serialization Details for Synthesis in PaCoRe

You are given a problem and a list of reference responses. Your job is to analyze these references and provide your own response. Original Problem: {{ original_prompt }}

Reference Responses: {% for response in ref_responses %} Reference {{ loop.index }}: {{ response }} {% endfor %}

Now, based on the original problem and reference responses above, please provide your own comprehensive solution.

Table 8 | Input serialization template for PaCoRe synthesis. We use this template to embed the current problem 𝑥 (denoted as original_prompt) and the compact message set 𝑀 (denoted as ref_responses) into the model’s context. In the degenerate case where the message set is empty (𝑀 = ∅), this template is bypassed, and the original problem input is passed to the model unmodified.

As detailed in Table 8, we frame compact messages as “Reference Responses” to encourage the model to synthesize diverse perspectives. By populating the “Original Problem” slot with the latest observation while maintaining the interaction history in context, PaCoRe ensures seamless compatibility with existing reasoning ecosystems. Further implementation details regarding the synthesis process can be found in Hu et al. (2026), Section C.

###### C. Evaluation Details This section outlines the evaluation setup and the corresponding evaluation prompts.

###### C.1. Evaluation Details for Multimodal Benchmarks

We detail the prompt formats used for evaluation across different benchmarks. For each benchmark, we present the corresponding prompt template, where {question} denotes the textual problem description, potentially including answer options, and <image> represents the visual input. When images are embedded in the question with explicit positional semantics, their original positions are preserved; otherwise, images are placed before the question text.

MMMU. We adopt the evaluation metric suggested by OpenCompass.1. The placement of image placeholders follows the original MMMU samples, allowing for interleaved visual inputs.

1https://github.com/open-compass/VLMEvalKit

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

MMMU-Pro. We use the official metric of MMMU-Pro.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

MathVision. We use the official metric of MathVision.

|Please solve the problem and put your answer in one "\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\boxed{}". <image> {question}|
|---|

MathVista. For MathVista, we follow the official evaluation protocol and use distinct prompt templates corresponding to different answer formats.2 For questions requiring floating-point answers with one or two decimal places, we use the following prompts, respectively:

|<image> Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end. Question: {question}|
|---|

|<image> Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value (e.g., 1.2, 1.3, 1.4) at the end. Question: {question}|
|---|

For multiple-choice questions, we use the following prompt:

|<image> Hint: Please answer the question and provide the correct option letter (e.g., A, B, C, D) at the end. Question: {question}|
|---|

For questions requiring an integer answer, we use the following prompt:

2https://github.com/lupantech/MathVista

|<image> Hint: Please answer the question requiring an integer answer and provide the final value (e.g., 1, 2, 3) at the end. Question: {question}|
|---|

For questions requiring a Python list as the answer, we use the following prompt:

|<image> Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end. Question: {question}|
|---|

Additional details are available on the official MathVista website. LogicVista. We use the official metric of LogicVista.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

DynaMath. For DynaMath, we adopt the official evaluation protocol and use the worstcase accuracy metric, defined as the percentage of correctly answered seed questions across all generated variations, to assess model robustness on mathematical reasoning tasks.3

For multiple-choice questions, we use the following prompt:

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" (option letter only).|
|---|

For questions requiring a floating-point answer, we use the following prompt:

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: \boxed{{answer}}." Round the answer to three decimal places.|
|---|

For all other questions, we use the following prompt:

3https://github.com/DynaMath/DynaMath

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: \boxed{{answer}}."|
|---|

Additional details are available on the official DynaMath repository. ZeroBench. We use the official metric of ZeroBench.

|<image> {question}<br><br>Give the final answer in curly braces, like: \boxed{final_answer}.|
|---|

MathVerse. We use the official metric of MathVerse and focus on the Vision-only subset. Details of the answer extraction and judgement can be seen in the official MathVerse repository. 4

For multiple-choice questions, we use the following prompt:

|Answer the question in the image. Provide the correct option letter, e.g., A, B, C, D, within \boxed{}. <image>|
|---|

For other questions, we use the following prompt:

|Answer the question in the image. Put your answer within \boxed{}. <image>|
|---|

We-Math. We use the official metric of We-Math.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

VisuLogic. We use the official metric of VisuLogic.

|<image> {question} Your response can be freely expressed in any format, but the final answer should follow this format: Answer: \boxed{$LETTER}.|
|---|

PhyX. We use the official metric of PhyX.

4https://github.com/ZrrSkywalker/MathVerse

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

HLE. We follow the official evaluation metrics and LLM-based judgement protocols of HLE.

|<image> {question}|
|---|

MMBench. We report accuracy on the MMBench v1.1 Dev set. We use the following prompt for MMBench-EN, and apply its Chinese translation for MMBench-CN.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

SimpleVQA. We follow the official evaluation metrics and LLM-based judgement protocols of SimpleVQA.

|<image> {question}|
|---|

MMStar. We use official metric of MMStar.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

HallusionBench. We use official metric of HallusionBench.

|<image> {question}<br><br>Please answer yes or no.|
|---|

MMVP. We use the official metric of MMVP. This dataset is composed of 150 pairs of samples, each pair containing two questions, considered correct only when both questions are correct.

|<image> {question}<br><br>Please select the correct answer from the options above. Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

ReMI. We use the official metric of ReMI. The placement of image placeholders follows the original ReMI samples, allowing for interleaved visual inputs.

|<image> {question}<br><br>Please select the correct answer from the options above. Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

M3GIA. We use the official metric and adopt the following system and user prompts. We use the system prompt format below:

|Answer following questions with the option’s letter from the given choices directly.|
|---|

DoYouSeeMe. We use the official metric and adopt the following system and user prompts. We adopt the system prompt specified by the question domain.

|# [’Shape Discrimination’, ’Joint Shape-Color’, ’Spatial Grids’] You are an AI assistant specialized in visual perception tasks. Please analyze the image carefully and provide your answer as an integer number. Format your response by putting your final answer after ’Answer:’, for example: Answer: 5<br><br># [’Letter Discrimination’] You are an AI assistant specialized in visual perception tasks. Please analyze the image carefully and identify the letter or text. Format your response by putting your final answer after ’Answer:’, for example: Answer: A<br><br># [’Form Constancy’, ’Visual Closure’, ’Visual Figure-Ground’] You are an AI assistant specialized in visual perception tasks. Please analyze the image carefully and select the correct option. Format your response by putting your final answer after ’Answer:’, using only the option number (1-4), for example: Answer: 2<br><br># Otherwise: You are an AI assistant specialized in visual perception tasks. Please analyze the image carefully and provide your answer. Format your response by putting your final answer after ’Answer:’|
|---|

And we use the user prompt format below:

|<image> {question}|
|---|

CountBench. We use the official metric of CountBench.

|<image> {question}<br><br>Please select the correct answer from the options above. Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct count number]" with the number only.|
|---|

CountQA. We use the official metric of CountQA.

|<image> {question}<br><br>Please select the correct answer from the options above. Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct answer]".|
|---|

PixMo-Count. We use the official metric of PixMo-Count.

|<image> {question}<br><br>Please select the correct answer from the options above. Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct count number]" with the number only.|
|---|

MM-MT-Bench. We use the official metric of MM-MT-Bench.5

|<image> {question}|
|---|

MIA-Bench. We follow the official evaluation metrics and LLM-based judgement protocols of MIA-Bench.

|<image> {question}|
|---|

MM-IFEval. We follow the official metric and evaluation protocols. We adopt the system prompts specified by the question types. For the P-Level questions, we use the following system prompt:

|You are an AI assistant. Please answer the question based on the image. Provide a clear and concise answer.|
|---|

For the C-Level questions, we use the following system prompt:

|# Have constraints You are an AI assistant. Please answer the question based on the image while strictly following these constraints:<br><br>{constrains} Make sure your response adheres to ALL the constraints above. # Others You are an AI assistant. Please answer the question based on the image while following any instructions or constraints mentioned in the question.|
|---|

And we use the user prompt format below:

|<image> {question}|
|---|

HumanEval-V. We use the official metric of HumanEval-V.

|<image> {question}|
|---|

Design2Code. We use the official metric of Design2Code.

5https://github.com/mistralai/mistral-evals/tree/main

|You are an expert web developer who specializes in HTML and CSS. A user will provide you with a screenshot of a webpage. You need to return a single html file that uses HTML and CSS to reproduce the given website. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder. Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well. Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+CSS file: <image>|
|---|

OCRBench . We use the official metric and follow the prompt.

|<image> {question}|
|---|

Omni-OCR. We use the official metric of Omni-OCR.

|<image> {question}|
|---|

CC-OCR (Multi-Lang-OCR subset). We use the official metric of CC-OCR.

|<image> Please output only the text content from the image without any additional descriptions or formatting.|
|---|

BLINK. We use the official metric of BLINK.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

CVBench. We use the official metric of CVBench.

|<image> {question}|
|---|

MMSI-Bench. We use the official metric of MMSI-Bench.

|<image> {question} Answer with the option’s letter from the given choices directly. Enclose the option’s letter within ‘ ’.|
|---|

ERQA. We use the official metric of ERQA.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

OmniSpatial. We use the official metric of OmniSpatial.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

All-Angles-Bench. We use the official metric of All-Angles-Bench.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

MindCube-tiny. MindCube-tiny is a condensed subset of the MindCube benchmark designed for efficient evaluation of Vision-Language Models in reconstructing 3D spatial structures and performing mental simulations from limited perspectives. We use its official metric.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

RealWorldQA. We use the official metric of RealWorldQA.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

SpatialViz-Bench. We use the official metric of SpatialViz-Bench.

|<image> {question} You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.|
|---|

STARE. We use the official metric of STARE.

|<image> {question} Answer with the option’s letter from the given choices and put the letter in one \boxed{}. Please solve the problem step by step.|
|---|

CoreCognition. We use the official metric of CoreCognition. For multiple-choice questions, we use the following prompt:

|<image> {question} Answer with the option’s letter from the given choices and put the letter in one \boxed{}.|
|---|

For the True/False (Yes/No) questions, we use the following prompt:

|<image> {question} Answer with YES or NO and put the answer in one \boxed{}.|
|---|

V*. We use the official metric of V*.

|<image> {question} Your response can be freely expressed in any format, but the final answer must be presented in this format: "Final answer: [the correct option]" with the option letter only.|
|---|

ViewSpatial. We use the official metric of ViewSpatial.

|<image> {question}<br><br>Reply only to the corresponding option. Answer:|
|---|

CharXiv (RQ). We use the official metric of CharXiv and adopt the following system and user prompts. We adopt the system prompt specified by the question domain.

|You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here.|
|---|

And we use the user prompt format below:

|<image> {question}|
|---|

AI2D. We use the official metric of AI2D.

|Answer following questions with the option’s letter from the given choices directly. <image> {question}|
|---|

CSVQA. We use the official metric of CSVQA. For multiple-choice questions, we use the following prompt:

|<image> 请回答图片中的问题。将正确选项字母（如A、B、C、D）放在\boxed{} 中。|
|---|

For other questions, we use the following prompt:

|<image> 请回答图片中的问题。将最终答案放在\boxed{} 中。|
|---|

EncQA. We use the official metric and adopt the following user prompts specified by the question type.

|# For multiple choices problem Answer using only a single word or letter from the options provided. {question} Options: {options} # For set problems Answer choosing only from the options provided, your answer should be just a simple comma separated list.<br><br>{question} Options: {options} # For numeric problem Answer using only a single number. {question}|
|---|

OmniDocBench. We use the NED (Normalized Edit Distance) metric and adopt the official system prompt and user prompt. 6

6https://github.com/opendatalab/OmniDocBench

ScreenSpot-Pro & ScreenSpot-V2 & OSWorld-G & MMBench-GUI-L2. For the GUI grounding tasks, we use a unified prompt format designed to localize visual elements and output their coordinates in a structured form.

|<image> Based on the instruction ‘{question}’, locate the target element and output its coordinate point in JSON format.|
|---|

###### C.2. Evaluation Details for Text-Centric Benchmarks

To reduce metric variance and improve result reliability on text-centric benchmarks, we perform repeated evaluation for selected benchmarks. For a benchmark with Repeat = 𝑁, each sample is evaluated independently 𝑁 times, and the final score is reported as the average over all runs.

The repetition settings for each text-centric benchmark are listed below:

- • MMLU-Pro: Repeat = 1
- • GPQA-Diamond: Repeat = 16
- • SuperGPQA: Repeat = 1
- • LiveBench(2024-11-25): Repeat = 1
- • AIME 2024: Repeat = 64
- • AIME 2025: Repeat = 64
- • HMMT25: Repeat = 64
- • CNMO2024: Repeat = 64
- • BeyondAIME: Repeat = 64
- • IMO-AnswerBench: Repeat = 1
- • LiveCodeBench (2408-2505): Repeat = 16
- • IFEval: Repeat = 4
- • IFBench: Repeat = 4
- • MultiChallenge: Repeat = 1
- • Arena-Hard-V2: Repeat = 1
- • WildBench: Repeat = 1
- • HealthBench: Repeat = 1

###### C.3. Evaluation Details for Ablations

Each ablation study in Sec. 5.1 is conducted on checkpoints pre-trained with the same number of billions of tokens, ensuring fair and controlled comparisons, but without extending to the final checkpoint due to computational cost. In terms of evaluation setups, these results are attained from a few-shot evaluation manner on the pre-trained checkpoints.

