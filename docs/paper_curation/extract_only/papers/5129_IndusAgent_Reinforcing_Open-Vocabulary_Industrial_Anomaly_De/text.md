# arXiv:2605.20682v1[cs.CV]20May2026

## IndusAgent: Reinforcing Open-Vocabulary Industrial Anomaly Detection with Agentic Tools

Rongbin Tan1 9 * Fangfang Lin2 * Zhenlong Yuan3 * ‡ Min Qiu4 Kejin Cui4 Mengmeng Wang5 Yi Wang1 Zijian Song6 Zhiyuan Wang4 Jiyuan Wang7 Yue Wang8 Shuhan Song1 9 § Huawei Cao1 9 1 State Key Lab of Processors, Institute of Computing Technology, CAS 2 Santa Clara University 3 LongCat Team 4 Independent Researcher 5 New York University 6 Sun Yat-sen University 7 Nanyang Technological University 8 Stanford University 9 University of Chinese Academy of Sciences, Beijing, China

* Equal contribution ‡ Project Lead § Corresponding Author

### Abstract

Multimodal large language models (MLLMs) have shown remarkable capability in bridging visual perception and textual reasoning, enabling zero-shot understanding across diverse industrial scenarios. However, their performance in open-vocabulary industrial anomaly detection (IAD) is often limited by domain-misaligned reasoning and hallucinated structural inferences. To address these challenges, we propose IndusAgent, a tool-augmented agentic framework for open-vocabulary IAD. Specifically, we first construct Indus-CoT, a structured dataset that integrates global visual observations, high-resolution local patches, and expert normalcy priors, providing supervision for fine-tuning the model on rigorous industrial inspection trajectories. Building on this, IndusAgent dynamically orchestrates a set of external tools, including dynamic region cropping, high-frequency feature enhancement, and prior retrieval, thus enabling the agent to actively resolve visual ambiguities and disentangle subtle anomalies. Furthermore, we introduce a gated reinforcement learning objective that jointly optimizes anomaly classification, localization accuracy, anomaly type reasoning, and efficient tool usage, ensuring that tool invocation occurs only when beneficial. Extensive evaluations on five industrial anomaly benchmarks, including MVTec-AD, VisA, MPDD, DTD, and SDD, demonstrate that IndusAgent achieves state-of-the-art zero-shot performance among all existing methods, validating our robustness and generalization capacity.

### 1 Introduction

Open-vocabulary industrial anomaly detection (IAD) aims to identify unpredictable defect classes and unseen object categories not present during training, extending beyond the closed-set constraints of traditional visual inspection systems [7, 111]. This capability is crucial for real-world manufacturing, where novel products and unpredictable defect morphologies frequently emerge. [83, 38]. Mainstream non-LLM approaches, such as reconstruction-based networks (e.g., Autoencoders [99, 6], Diffusion models [90, 58]) and feature-embedding frameworks (e.g., Memory banks [68, 27], Normalizing flows [96, 69]), are fundamentally bottlenecked by closed-set assumptions. They demand extensive category-specific normal data and critically lack the capacity to generalize to unseen products in open-world manufacturing scenarios [31].

Preprint.

[Figure 1]

- Figure 1: Comparison of anomaly detection paradigms using MLLMs. (a) Standard MLLMs suffer from unaligned reasoning and structural hallucinations, often misinterpreting legitimate variations. (b) Ordinary Chain-of-Thought (CoT) reasoning is insufficient; without domain knowledge and localized perception, the model misjudges subtle defects as normal reflections due to perceptual dilution. (c) Our proposed framework constructs an active inspection paradigm through comprehensive tool orchestration. By synergizing high-resolution region cropping, low-level texture enhancement, quantitative geometric measurement, and expert semantic priors, the agent effectively overcomes both visual ambiguities and physical scale-blindness. This strategic alignment ensures rigorous diagnostic trajectories, accurately disentangling complex geometries to detect subtle anomalies.

Recently, the advent of Multimodal Large Language Models (MLLMs) has ignited a paradigm shift toward open-vocabulary visual reasoning [59]. By aligning visual tokens with rich textual semantics, MLLMs offer a transformative opportunity to overcome the data-dependency and closedset limitations of traditional IAD systems, enabling unprecedented zero-shot detection capabilities.

However, bridging the cognitive gap between MLLMs and high-precision industrial applications reveals three intrinsic limitations: ❶ Domain-Misaligned Reasoning: As shown in Fig. 1(a), standard MLLMs are primarily optimized for open-ended, general-purpose conversations [49]. Their inherent reasoning trajectories fail to conform to the strict, formalized diagnostic protocols that are essential for accurate industrial anomaly detection. ❷ c ❸ Open-Vocabulary Generalization: While existing models can memorize predefined defect categories, they exhibit brittle adaptability in open-vocabulary inspections. When confronting novel anomalies or ambiguous linguistic instructions, their zero-shot reasoning heavily deteriorates due to an inherent lack of strategic exploration and structural coherence [26].

To address these critical bottlenecks, we propose IndusAgent, a unified framework that synergizes domain-specific reasoning with autonomous tool orchestration. As shown in Fig. 1(c), We bridge the diagnostic protocol gap through Supervised Fine-tuning, which aligns the model’s reasoning trajectories with expert-level industrial standards. Building on this foundation, we introduce Tool Augmentation into the agent’s cognitive loop. This equips the model with the active means to combat perceptual dilution and structural hallucinations by dynamically scrutinizing high-resolution patches and querying expert normalcy priors.

Furthermore, adapting to the boundless variations inherent in open-vocabulary IAD requires dynamic, self-improving exploration beyond static SFT. To this end, we introduce Agentic Reinforcement Learning (RL) to optimize the agent’s decision-making trajectories across unseen domains. However, empowering the agent with autonomous exploration inevitably risks tool abuse—a prevalent issue where indiscriminate API invocations introduce redundant noise and dilute the reasoning focus. To overcome this dilemma without stifling necessary exploration, our RL framework features a novel Accuracy-Gated reward mechanism. By strictly gating a positive tool utility bonus with the final diagnostic correctness, this sophisticated formulation trains the agent to treat tool-calling as a

[Figure 2]

- Figure 2: The overall architecture of IndusAgent. Our training pipeline consists of three sequential stages: (1) Indus-CoT Construction, where a frontier model (Qwen3-VL-Max) synthesizes structured reasoning trajectories to form high-quality positive and negative examples; (2) Agentic Fine-Tuning, which aligns a lightweight base model (Qwen3-VL-8B) with domain-specific diagnostic protocols; and (3) Tool-Augmented RL. In the final stage, the agent’s tool-augmented reasoning loop is optimized via GRPO. The policy is guided by a specific gated reward function R(r).

high-stakes diagnostic instrument. It ensures that unbounded visual exploration is organically aligned with genuine diagnostic information gain [70, 100].

In summary, our main contributions are summarized as follows:

- • Active Inspector Paradigm. We introduce a unified paradigm that integrates autonomous, multiround tool orchestration with MLLMs for industrial anomaly detection, effectively transcending the resolution and semantic limitations inherent in passive visual perception.
- • Tool-Integrated Industrial Reasoning Corpus. We construct Indus-CoT, a structured reasoning dataset that encodes industrial inspection trajectories with global observations, localized evidence, normalcy priors, and final defect judgments. By explicitly linking visual cues, tool feedback, and diagnostic decisions, Indus-CoT provides effective supervision for domain-aligned, tool-augmented anomaly reasoning.
- • Accuracy-Gated Reward Mechanism. We formulate a cascading Agentic RL objective that utilizes a multiplicative gate to seamlessly integrate tool utility with diagnostic task efficacy. By rewarding tool orchestration only when it culminates in correct predictions, this design successfully eradicates stochastic tool abuse and fosters a highly judicious, accuracy-driven reasoning policy.
- • State-of-the-Art Performance. IndusAgent achieves state-of-the-art results across five challenging benchmarks (MVTec-AD, VisA, DTD, MPDD, and SDD), especially outperforming SOTA method by 9.3% on MVTec, validating our effectiveness.

### 2 Methodology

Overview. We propose IndusAgent, a post-training framework that synergizes visual anomaly perception with tool-augmented reinforcement learning, as illustrated in Fig. 2. The framework consists of three tightly coupled stages. First, we construct Indus-CoT, a tool-integrated reasoning dataset that synthesizes image-query trajectories with predefined prompts to bridge visual perception and tool execution [89, 49]. Second, we perform Supervised Fine-Tuning to align the VLM with structured industrial diagnostic trajectories and tool-use syntax. Third, we apply Tool-Augmented Reinforcement Learning with a hierarchical reward that jointly balances tool-usage correctness, anomaly interpretation, and structural reasoning coherence [63].

#### 2.1 Systematic Definition and Agentic Toolkit

Problem Formulation. We formulate industrial anomaly detection (IAD) as a tool-augmented visual reasoning process [35]. Given only a query image I ∈ RH×W×3 and a task instruction Q, the model is required to generate a structured diagnostic output O, including the reasoning trajectory, anomaly localization, fine-grained defect category, and final binary judgment. All models, including commercial APIs and open-source baselines, receive the same query image and textual instruction, and are required to infer the normal structure and anomaly status from their internal visual-language knowledge and the provided input alone. Instead of directly mapping the input image to a prediction, we instantiate the VLM as an agentic policy πθ based on Qwen3-VL-8B [4]. The policy interacts with a customized tool space T = {Tcrop,Tprior,Tenhance,Tmeasure} to actively acquire complementary evidence for diagnosis.

Unified Agentic Inference. IndusAgent performs diagnosis through a multi-step autoregressive reasoning process. After perceiving the global image, the policy identifies uncertain regions or ambiguous structures and generates tool calls C ⊆ T when additional evidence is needed. The corresponding tool observations are then fused with the original image and instruction to produce the final structured output:

O ∼ πθ(· | I ⊕ F,Q ⊕ E;T ), (1) where ⊕ denotes multimodal fusion. Here, F represents visual feedback, including high-resolution local patches from Tcrop and enhanced texture maps from Tenhance, while E denotes semantic or quantitative feedback, including normalcy priors from Tprior and geometric measurements from Tmeasure. This formulation enables the agent to combine global context, localized evidence, and external diagnostic cues before making the final decision.

Agentic Toolkit. We instantiate four tools to address typical IAD failure modes. Tcrop extracts high-resolution patches from suspicious regions to recover fine-grained defects diluted by global

encoding. Tprior retrieves normalcy priors describing defect-free geometry, texture, and structural patterns, providing a comparison anchor for distinguishing true defects from acceptable variations.

Tenhance applies lightweight image-processing operations, such as contrast enhancement and edge extraction, to highlight low-contrast texture changes. Tmeasure computes geometric relations, such as distances, angles, and relative positions, to verify misalignment, deformation, missing parts, and abnormal spacing.

#### 2.2 Indus-CoT Dataset

Existing VLMs face two major limitations in industrial anomaly detection: they passively observe the input image without actively seeking external evidence, and they may hallucinate defect explanations when subtle visual cues cannot be cross-verified with domain knowledge [28, 79]. To address these issues, we construct Indus-CoT, a tool-integrated reasoning dataset that combines multimodal CoT trajectories with explicit tool-execution traces [109, 17]. This dataset provides supervision for multiround diagnostic reasoning, where the model learns not only to judge anomalies but also to acquire and use external evidence when necessary.

Data Collection & Automated Curation. We sample images from Real-IAD [84] and construct about 3,000 reasoning trajectories, with roughly balanced normal and anomalous samples [88]. To prevent category leakage, we remove all Real-IAD categories overlapping with the evaluation benchmarks, including DTD, MPDD, MVTec-AD, SDD, and VisA, using both exact matching and semantic normalization for naming variants such as pcb versus pcb1/pcb2/pcb3/pcb4 and transistor1 versus transistor. After filtering overlapping categories such as toothbrush, zipper, pcb, and transistor1, the resulting training set is category-disjoint from all test benchmarks.

For each query image, no paired normal reference image is provided to the teacher model. The teacher receives only the query image and task instruction, infers the expected defect-free appearance from its internal visual-language knowledge and general industrial priors, and generates a structured Indus-CoT trajectory covering global perception, tool routing, tool observations, and final diagnostic verification. This reference-free construction matches our inference setting, where both IndusAgent and all baselines diagnose anomalies from the query image alone. To improve data quality, we further apply self-correction and LLM-as-a-judge validation to repair invalid outputs, score candidate trajectories, and retain the highest-quality valid trajectory, thereby reducing label inconsistency and formatting errors in the SFT data.

Tool-Integrated Generating Pipeline. Indus-CoT trajectory follows a three-phase reasoning process:

- • Phase 1: Global Perception and Tool Routing. The model first analyzes the global query image to identify suspicious regions, ambiguous structures, or uncertain visual patterns. Instead of directly producing a final judgment, it generates routing commands to invoke suitable tools.
- • Phase 2: Tool Execution and Contextual Observation. The selected tools return complementary observations. Tprior provides textual normalcy priors, Tmeasure computes distances or angles from specified coordinates, and Tenhance applies deterministic filters such as CLAHE to highlight highfrequency textures. For Tcrop, we avoid using ground-truth boxes during execution and instead adopt an unsupervised foreground extraction procedure, combining background estimation, image differencing, Otsu thresholding, morphological operations, and a center-crop fallback.
- • Phase 3: Final Diagnostic Verification. The model integrates the original image with tool observations, including local crops, enhanced texture maps, normalcy priors, and geometric measurements. It then cross-verifies the collected evidence and outputs the final anomaly judgment, location, and defect category.

#### 2.3 Supervised Fine-Tuning

Directly optimizing Vision-Language Models with reinforcement learning for complex visual tasks is often unstable. Inspired by R1-Zero [34], our preliminary trials show that, without structural constraints, the policy can suffer from reward hacking and format collapse, bypassing intermediate visual inspection and exploiting terminal rewards through blind binary guesses. To stabilize training, we introduce a Supervised Fine-Tuning (SFT) stage to cold-start Qwen3-VL-Instruct (8B) with structured industrial diagnostic trajectories before reinforcement learning.

Formally, we formulate SFT as conditional autoregressive generation over our curated reasoning dataset. Each training instance is denoted as T = (X,I,S,Y), where X denotes the visual input, including the global query image and multi-round tool observations; I is the task instruction; S = {s1,...,sT} represents the reasoning steps constrained within the <think>...</think> trajectory; and Y denotes the final target output.

To guarantee that the model actively internalizes the reasoning logic rather than passively memorizing the input context, we implement a selective masking strategy during training. The objective minimizes the negative log-likelihood exclusively over the generated tokens of the reasoning process:

LSFT = −ET ∼D

T

log pθ(st | X,I,s<t) , (2)

t=1

where pθ(·) dictates the conditional probability distribution of the parameterized policy network. By explicitly supervising the cognitive trajectory, this phase successfully anchors the model’s structural consistency, equipping it with a robust and well-calibrated policy initialization for the subsequent reinforcement learning phase.

#### 2.4 Agentic Reinforcement Learning

Group Relative Policy Optimization (GRPO). To optimize the agent’s decision-making process without the prohibitive memory costs associated with traditional actor-critic architectures [71], we utilize Group Relative Policy Optimization (GRPO) [72]. Instead of relying on a separate value network, GRPO evaluates policy updates through a groupwise relative comparison mechanism. Specifically, for a given query image q and its corresponding ground truth a sampled from the dataset D, the system samples a batch of G distinct reasoning trajectories {o1,o2,...,oG} using the reference policy πθ

. The current policy πθ is subsequently updated by maximizing the following:

old

LGRPO(θ) = −Eq∼P(Q),{o

i}Gi=1∼πθold(O|q)

G

1 G

i=1

min

πθ(oi|q) πθ

Ai,clip

(oi|q)

old

πθ(oi|q) πθ

,1 − ϵ,1 + ϵ Ai − βDKL(πθ∥πref) ,

(oi|q)

old

(3)

πref(oi|q) πθ(oi|q) − log

πref(oi|q)

πθ(oi|q) − 1, (4) where the coefficient β regulates the KL divergence penalty to ensure training stability and prevent the policy from deviating excessively from the reference model. The advantage estimator Ai is dynamically derived by normalizing the rewards within the sampled trajectory group:

DKL(πθ||πref) =

ri − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

. (5)

Ai =

Here, ri represents the comprehensive scalar reward assigned to each trajectory oi, computed by a rigorous, rule-based verification mechanism to prevent reward hacking.

Reward Formulation. A carefully designed reward is essential for encouraging effective tool use while avoiding behavioral degradation. We propose an Accuracy-Gated reward that couples tool usage with final diagnostic correctness, so that auxiliary rewards are activated only when the basic anomaly judgment is correct.For a trajectory τ, the overall reward is defined as:

R(τ) = Racc(τ) · 1 + αRloc(τ) + βRtype(τ) + γRtool(τ) + Rformat(τ), (6)

where Racc denotes binary anomaly classification correctness, Rloc measures localization quality, Rtype evaluates fine-grained anomaly categorization, Rtool encourages useful tool invocation, and Rformat enforces output-format compliance. The weights α, β, and γ balance the relative contributions of localization, semantic categorization, and tool usage.

❶ Classification Accuracy (Racc): Racc ∈ {0,1} evaluates whether the final binary anomaly judgment is correct and serves as a multiplicative gate, ensuring that localization, type prediction, and tool-usage rewards are credited only when the final diagnosis is correct. ❷ Spatial Localization (Rloc): Rloc measures the overlap between the predicted anomaly region and the ground-truth region using IoU. ❸ Semantic Categorization (Rtype): Rtype evaluates the correctness of the predicted anomaly type based on its semantic distance to the ground-truth category. ❹ Tool Utility (Rtool): To promote useful rather than excessive tool use, we define Rtool = λ · I[∆conf > 0] − η|C|, where C is the set of invoked tools, ∆conf denotes the confidence improvement after incorporating tool feedback, I[·] is the indicator function, λ,η are hyperparameters, empirically set to 0.3 and 0.1, respectively. This term rewards beneficial evidence acquisition while penalizing redundant tool calls. ❺ Process Compliance (Rformat): Rformat penalizes invalid output structures, such as missing or malformed <answer> tags, to prevent format collapse during RL training.

Effect on Tool-Use Behavior. The accuracy-gated formulation encourages the agent to associate tool use with final diagnostic correctness rather than tool invocation itself. Since Rtool contributes to the reward only when the binary anomaly judgment is correct, redundant or uninformative tool calls do not provide effective task-level gains and are further penalized by the cost term −η|C|. As a result, the policy is biased toward invoking tools only when additional local, textual, or geometric evidence is likely to improve the final diagnosis.

### 3 Experiment

#### 3.1 Experimental Setup

Datasets and Benchmarks. We evaluate IndusAgent on five industrial anomaly detection benchmarks: MVTec-AD [7], VisA [111], MPDD [39], DTD [3], and SDD [80]. These datasets comprehensively cover two representative scenarios: (1) industrial objects, which are characterized by complex structures, poses, and geometries; (2) surface textures, where defects are often subtle and embedded within repetitive or noisy patterns. To ensure a fair comparison, all baselines are evaluated under identical prompt and answer parsing protocols.

#### 3.2 Main Results

Table 1 and Figure 3 present a comprehensive zero-shot performance comparison across five industrial anomaly detection (IAD) benchmarks, encompassing both industrial objects and surface textures. Overall, our proposed IndusAgent (8B) establishes a new state-of-the-art (SOTA) with an average

- Table 1: Performance comparison of different models on industrial workpieces and surface texture benchmarks. The best and second best results are highlighted.

Type Model Param.

Industrial Workpieces Surface Texture

Avg. MVTec MPDD VisA DTD SDD

Commercial

GPT-4o-mini [61] / 71.3 67.9 65.1 79.5 66.6 70.1 GPT-4o [61] / 69.6 60.3 63.5 69.9 65.7 65.8 GPT-4.1-nano [62] / 74.7 61.7 60.5 78.3 50.0 65.0 GPT-4.1-mini [62] / 74.0 69.8 63.4 82.1 72.1 72.3 GPT-4.1 [62] / 81.9 66.7 69.1 90.1 79.9 77.5 Claude-Sonnet-4 [2] / 67.6 65.9 63.5 88.4 81.7 73.4

Open Source

LLaVA-OneVision-SI [40] 0.5B 50.0 50.0 50.0 54.3 50.0 50.8 Anomaly-OV [91] 0.5B 50.0 50.0 50.0 53.8 50.0 50.8 Qwen2.5-VL-Instruct [5] 3B 62.6 52.9 58.4 64.4 50.3 57.7 AnomalyR1[16] 3B 69.4 56.0 59.8 61.0 57.6 60.7 InternVL-2.5 [24] 4B 56.6 59.1 53.7 81.3 64.1 62.9 Qwen2.5-VL-Instruct [5] 7B 66.0 56.0 58.4 59.2 67.4 61.4 Qwen3-VL-Instruct [4] 8B 67.0 50.0 46.8 70.2 50.0 56.8 AnomalyGPT [31] 7B 46.6 54.2 57.3 64.1 49.5 54.3 Anomaly-OV [91] 7B 74.3 70.3 74.3 90.7 88.7 79.6

- LLaVA-1.5 [51] 13B 61.4 61.4 67.2 75.9 50.0 63.1

- LLaVA-1.6 [52] 34B 53.7 50.0 53.9 52.7 50.0 52.0 Qwen2.5-VL-Instruct [5] 72B 77.4 64.7 68.2 81.9 69.0 72.2 IndusAgent (Ours) 8B 83.6 72.7 76.8 95.6 88.9 83.4

- Table 2: Anomaly Recall Comparison. Our method fundamentally mitigates the false-negative bottleneck inherent in standard MLLMs. IndusAgent consistently outperforms both open-source models and commercial APIs, achieving massive recall surges and ensuring industrial-grade reliability.

Model DTD MPDD MVTec SDD VisA Avg.

Qwen3-VL-8B 75.1% 62.3% 68.9% 64.2% 58.7% 65.8% Claude-Sonnet-4 80.2% 70.5% 75.3% 71.1% 65.4% 72.5% IAD-R1 83.7% 78.0% 81.7% 79.6% 72.3% 79.1%

IndusAgent (Ours) 94.1% 95.4% 85.5% 83.3% 73.4% 86.3%

score of 83.4%. As visually corroborated by its dominant envelope in the radar chart, it significantly and consistently outperforms both leading commercial systems and the largest open-source models. Notably, on structurally complex datasets such as VisA and MPDD, IndusAgent achieves impressive scores of 76.8% and 72.7%, respectively. This decisively surpasses the best-performing VLM baselines while strictly maintaining a highly efficient 8B parameter footprint.

- 3.3 Key Findings and Insights

- Finding 1: Domain-specific alignment is critical. MLLM reasoning alone remains unreliable for complex industrial samples. For example, Qwen3-VL-Instruct performs poorly on VisA, while Agentic SFT and RL substantially improve performance, indicating that robust IAD requires task-specific diagnostic alignment rather than open-ended reasoning alone.
- Finding 2: Active tooling complements passive perception. Subtle defects are often diluted by large normal regions, visual noise, or scale ambiguity. By selectively invoking cropping, enhancement, measurement, and normalcy-prior retrieval, IndusAgent isolates local evidence and verifies structural cues, showing that active tool use is an important complement to passive MLLM perception.

Improvements in Anomaly Recall. Anomaly recall is a critical metric in IAD, as missed defects (false negatives) typically

[Figure 3]

Figure 3: Zero-shot Comparison.

incur higher costs than false alarms. As shown in Tab. 2, IAD-R1 model occasionally struggles with recall across various datasets. This suggests that conventional supervised fine-tuning may lead to conservative predictions, overlooking subtle defects in complex backgrounds.

In contrast, our proposed GRPO framework addresses this limitation. By aligning the reasoning policy with final diagnostic outcomes, the agent is encouraged to actively verify potential anomalies rather than relying solely on initial passive observations. This approach yields consistent improvements in recall across all evaluated datasets. Notably, on datasets with severe background interference, the method shows substantial gains, achieving +17.4% on MPDD and +10.4% on DTD. These results demonstrate that the RL-driven orchestration effectively enhances the model’s reliability for complex industrial inspection tasks.

#### 3.4 Ablation Studies

To rigorously validate the contribution of each component, we conduct comprehensive ablation studies on three representative benchmarks. More ablation experiment results are shown in Appendix B.

Effectiveness of Core Framework Modules. We first evaluate the macro-architecture by removing individual training stages. As shown in Table 3, omitting the Agentic Supervised FineTuning phase (w/o. SFT) leads to a catastrophic performance collapse (e.g., plunging from 76.8% to 55.5% on VisA). This confirms that domainspecific protocol alignment is an absolute prerequisite for industrial tasks. Similarly, removing Reinforcement Learning (w/o. RL) results in a severe degradation, highlighting that SFT alone is insufficient for open-vocabulary generalization. Furthermore, ablating the Tool Augmentation library (w/o. TOL) causes a noticeable drop across all datasets, empirically proving that active tool orchestration is vital for mitigating perceptual dilution and resolving complex structural hallucinations.

Table 3: Ablation of main proposed modules.

Method MVTec VisA DTD

- (a) Qwen3-VL-8B 67.0 46.8 70.2

- (b) IndusAgent 83.6 76.8 95.6

- (c) w/o. RL 72.3 57.6 74.1

- (d) w/o. SFT 69.5 55.5 72.8

- (e) w/o. TOL 78.1 67.5 87.9

Deconstructing the Hierarchical Gated Reward. In Table 4, we dissect the Agentic RL phase to validate our hierarchical reward design. Compared to a standard RL baseline (w/. Base), our full gated reward mechanism achieves consistently superior results. Notably, removing the format compliance reward (w/o. Format) causes the most significant performance decay (e.g., dropping to 65.7% on VisA), as structural reasoning breaks down without strict output parsing. Moreover, ablating the fine-grained diagnostic rewards (w/o. Loc and w/o. Type) impairs the model’s ability to accurately ground anomalies in complex scenarios. Finally, removing the gated tool-utility term (w/o. Tool) decreases accuracy, suggesting that explicitly coupling tool invocation with final diagnostic correctness helps the agent learn when external evidence is beneficial.

Table 4: Ablation of hierarchical gated rewards.

RL Reward MVTec VisA DTD

- (f) w/. Base 76.0 64.9 79.1

- (g) w/o. Tool 81.9 71.5 92.8

- (h) w/o. Loc 79.5 68.8 89.6

- (i) w/o. Type 78.5 72.8 90.6

- (j) w/o. Format 76.6 65.7 82.5

### 4 Related Work

Open-Vocabulary Industrial Anomaly Detection. OV-IAD methods span reconstruction-based, feature-embedding-based, and vision-language-based paradigms. Reconstruction approaches like DRAEM [99] and autoencoder-based models [6] learn normal appearance via inpainting or synthetic anomaly generation, while diffusion-based extensions like AnoDDPM [90] and DiffusionAD [58] improve reconstruction fidelity yet may reconstruct anomalies and miss subtle defects. Featureembedding methods such as PaDiM [27] and PatchCore [68] achieve strong in-distribution performance through patch-level memory banks, while flow-based variants like FastFlow [96] and CS-Flow [69] improve density estimation; yet both rely on closed-set assumptions limiting openvocabulary applicability. Recent VLM-based approaches adapt cross-modal alignment for openvocabulary inspection: WinCLIP [38] enables zero-shot scoring via sliding-window CLIP matching, while AnomalyGPT [31] introduces prompt-guided MLLMs for few-shot localization. However, their passive single-pass paradigm limits sensitivity to subtle anomalies and generalization to unseen

[Figure 4]

Figure 4: Case Study between Qwen3-VL-8B and our method.

categories. In contrast, our method introduces an active inspector paradigm with tool-augmented RL for robust reasoning.

Reasoning in Multimodal LLMs. Recent advances in large language models have demonstrated that RL-based post-training can significantly enhance reasoning capabilities, as exemplified by OpenAI-o1 [60] and DeepSeek-R1 [26]. These paradigms have been extended to MLLMs for tasks like mathematical VQA [64], reasoning segmentation [50], and general image understanding [30]. However, industrial anomaly inspection poses a challenge: decisive evidence often lies in finegrained local structures such as small scratches, stains, or texture discontinuities, where standard multimodal CoT reasoning may hallucinate explanations when visual grounding is weak, especially under zero-shot object and defect categories. To address this, we propose tool-grounded multimodal CoT reasoning via RL, explicitly linking intermediate reasoning steps to external tool observations to reduce hallucination while preserving zero-shot generalization.

Tool-Augmented Agentic Systems. Tool use has become an effective way to enhance multimodal reasoning. MVoT [41] incorporates visual evidence into reasoning chains as multimodal thoughts, while LLaVA-Plus [53] and VPD [37] enable tool learning via supervised training or program-derived data; more recent works like TACO [56] and PyVision [110] further extend this with RL. However, most rely on static tool-use pipelines or objectives rewarding tool invocation without considering execution cost, causing tool overuse and unstable behavior. The most related concurrent work, AgentIAD [57], explores a tool-augmented agentic framework for IAD with SFT and RL training. Our work differs in both setting and objective: AgentIAD operates under structured in-domain supervision, whereas we target a stricter open-vocabulary zero-shot setting across unseen object categories and defect types. Moreover, our framework introduces an efficiency-aware multiplicative reward that jointly considers diagnostic correctness, information gain, and tool execution cost, encouraging the agent to invoke tools only when useful evidence exists for adaptive and efficient inspection.

### 5 Conclusion

In this work, we propose IndusAgent, a novel framework that synergizes domain-specific reasoning alignment with autonomous, tool-augmented reinforcement learning for zero-shot industrial anomaly detection. By grounding the model in expert diagnostic protocols via Agentic SFT and deploying a comprehensive toolset to isolate fine-grained patches, enhance low-contrast textures, perform quanti-

tative geometric measurements, and retrieve normalcy priors, our approach effectively overcomes perceptual dilution, scale-blindness, and structural hallucinations. Furthermore, the efficiency-aware Agentic RL paradigm optimizes this active inspection process, utilizing a hierarchical reward mechanism to penalize tool abuse while incentivizing open-vocabulary exploration. Extensive evaluations across five challenging benchmarks demonstrate that IndusAgent establishes new state-of-the-art performance, achieving significant gains over large-scale commercial and open-source models while maintaining rigorous inference parsimony. Future work will explore expanding this active agentic paradigm to multimodal temporal streams and more computationally constrained edge environments.

### References

- [1] Anonymous. Anylayout: Versatile advertising poster layout generation with MLLMs. In Submitted to The Fourteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=viX7rUMzwg. under review.
- [2] Anthropic. The claude 4 model family. Anthropic Technical Report, 2025.
- [3] T. Aota, L.T.T. Tong, and T. Okatani. Zero-shot versus many-shot: Unsupervised texture anomaly detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5564–5572, 2023.
- [4] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [5] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [6] P. Bergmann, S. Löwe, M. Fauser, D. Sattlegger, and C. Steger. Improving unsupervised defect segmentation by applying structural similarity to autoencoders. In 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, 2018.
- [7] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger. Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9592–9600, 2019.
- [8] Baolong Bi, Shaohan Huang, Yiwei Wang, Tianchi Yang, Zihan Zhang, Haizhen Huang, Lingrui Mei, Junfeng Fang, Zehao Li, Furu Wei, et al. Context-dpo: Aligning language models for context-faithfulness. ACL 2025, 2024.
- [9] Baolong Bi, Shenghua Liu, Lingrui Mei, Yiwei Wang, Pengliang Ji, and Xueqi Cheng. Decoding by contrasting knowledge: Enhancing llms’ confidence on edited facts. ACL 2025, 2024.
- [10] Baolong Bi, Shenghua Liu, Xingzhang Ren, Dayiheng Liu, Junyang Lin, Yiwei Wang, Lingrui Mei, Junfeng Fang, Jiafeng Guo, and Xueqi Cheng. Refinex: Learning to refine pre-training data at scale from expert-guided programs. arXiv preprint arXiv:2507.03253, 2025.
- [11] Baolong Bi, Shenghua Liu, Yiwei Wang, Siqian Tong, Lingrui Mei, Yuyao Ge, Yilong Xu, Jiafeng Guo, and Xueqi Cheng. Reward and guidance through rubrics: Promoting exploration to improve multi-domain reasoning. arXiv preprint arXiv:2511.12344, 2025.
- [12] Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, and Xueqi Cheng. Parameters vs. context: Fine-grained control of knowledge reliance in language models. arXiv preprint arXiv:2503.15888, 2025.
- [13] Jinhe Bi, Yifan Wang, Danqi Yan, Xun Xiao, Artur Hecker, Volker Tresp, and Yunpu Ma. Prism: Selfpruning intrinsic selection method for training-free multimodal data selection. ArXiv, abs/2502.12119,

2025. URL https://api.semanticscholar.org/CorpusID:276421326.

- [14] Jinhe Bi, Danqi Yan, Yifan Wang, Wenke Huang, Haokun Chen, Guancheng Wan, Mang Ye, Xun Xiao, Hin rich Schuetze, Volker Tresp, and Yunpu Ma. Cot-kinetics: A theoretical modeling assessing lrm reasoning process. ArXiv, abs/2505.13408, 2025.
- [15] Jinhe Bi, Aniri, Minglai Yang, Xingcheng Zhou, Wenke Huang, Sikuan Yan, Yujun Wang, Zixuan Cao, Michael Färber, Xun Xiao, Volker Tresp, and Yunpu Ma. EchoRL: Reinforcement learning via rollout echoing. In Forty-third International Conference on Machine Learning, 2026.
- [16] Y. Chao, J. Liu, J. Tang, and G. Wu. Anomalyr1: A grpo-based end-to-end mllm for industrial anomaly detection. arXiv preprint arXiv:2504.11914, 2025.

- [17] B. Chen, C. Shu, E. Shareghi, N. Collier, K. Narasimhan, and S. Yao. Fireact: Toward language agent fine-tuning. In International Conference on Learning Representations, 2024.
- [18] Long Chen, Haizhou Ai, Rui Chen, Zijie Zhuang, and Shuang Liu. Cross-view tracking for multi-human 3d pose estimation at over 100 fps. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3279–3288, 2020.
- [19] Minglin Chen, Longguang Wang, Sheng Ao, Ye Zhang, Kai Xu, and Yulan Guo. Layout2scene: 3d semantic layout guided scene generation via geometry and appearance diffusion priors. arXiv preprint arXiv:2501.02519, 2025.
- [20] Minglin Chen, Rongkun Yang, Qibin Hu, Kaiwen Xue, Shunbo Zhou, and Yulan Guo. Graph2scene: Versatile 3d indoor scene generation with interaction-aware scene graph. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11313–11320. IEEE, 2025.
- [21] Rui Chen, Lei Sun, Jing Tang, Geng Li, and Xiangxiang Chu. Finger: Content aware fine-grained evaluation with reasoning for ai-generated videos. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 3517–3526, 2025.
- [22] Yinda Chen, Wei Huang, Shenglong Zhou, Qi Chen, and Zhiwei Xiong. Self-supervised neuron segmentation with multi-agent reinforcement learning. In IJCAI, 2023.
- [23] Yinda Chen, Yangfan He, Jing Yang, Dapeng Zhang, Zhenlong Yuan, Muhammad Attique Khan, Jamel Baili, and Lip Yee. Empower: Evolutionary medical prompt optimization with reinforcement learning. IEEE Journal of Biomedical and Health Informatics, 2026.
- [24] Zhe Chen, Jiannan Wang, Yuchen Hao, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [25] Kejin Cui, Yongyao Jiang, Yun Li, and Dieter Pfoser. A vocabulary recommendation method for spatiotemporal data discovery based on bayesian network and ontologies. Big Earth Data, 3(3):220–231, 2019.
- [26] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in large language models via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [27] T. Defard, A. Setkov, A. Loesch, and R. Rompel. Padim: a patch distribution modeling framework for anomaly detection and localization. In International Conference on Pattern Recognition, pages 475–489. Springer, 2021.
- [28] D. Driess, F. Xia, M.S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, et al. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–8488. PMLR, 2023.
- [29] Yingda Fan, Runlong Yu, Janet R Barclay, Alison P Appling, Yiming Sun, Yiqun Xie, and Xiaowei Jia. Multi-scale graph learning for anti-sparse downscaling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 27969–27977, 2025.
- [30] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, B. Wang, and X. Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.
- [31] Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, and J. Wang. Anomalygpt: Detecting industrial anomalies using large vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024.
- [32] Renxiang Guan, Tianrui Liu, Wenxuan Tu, Chang Tang, Wenhan Luo, and Xinwang Liu. Sampling enhanced contrastive multi-view remote sensing data clustering with long-short range information mining. IEEE Transactions on Knowledge and Data Engineering, pages 1–15, 2025.
- [33] Renxiang Guan, Wenxuan Tu, Dayu Hu, Weixuan Liang, Ke Liang, Yaowen Hu, Yue Liu, and Xinwang Liu. Prototype-driven multi-view attribute-missing graph clustering. IEEE Transactions on Multimedia, 27:9454–9466, 2025.
- [34] D. Guo, J. Shao, H. Qiu, J. Bu, Z. Li, J. Zhang, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [35] T. Gupta and A. Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953– 14962, 2023.

- [36] Lingde Hu, Wenbo Zhang, Wenkang Zhang, Yu He, Seokmin Choi, Yang Gao, Jagmohan Chauhan, and Zhanpeng Jin. Ppgspeech: A wearable silent speech interface leveraging neck-worn photoplethysmography. IEEE Internet of Things Journal, 13(4):6692–6703, 2026. doi: 10.1109/JIOT.2025.3639152.
- [37] Y. Hu, O. Stretcu, C.-T. Lu, K. Viswanathan, K. Hata, E. Luo, R. Krishna, and A. Fuxman. Visual program distillation: Distilling tools and programmatic reasoning into vision-language models. arXiv preprint arXiv:2312.03052, 2023.
- [38] J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Nadar, and O. Dabeer. Winclip: Zero-/few-shot anomaly classification and segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19606–19616, 2023.
- [39] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim. Visual prompt tuning. In European conference on computer vision, pages 709–727. Springer, 2022.
- [40] Bo Li, Yuanhan Zhang, Dong Guo, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [41] C. Li, W. Wu, H. Zhang, Y. Xia, S. Mao, L. Dong, I. Vuli´c, and F. Wei. Imagine while reasoning in space: Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542, 2025.
- [42] Zehao Li, Hongwei Yu, Hao Jiang, Qiang Sheng, Yilong Xu, Baolong Bi, Yang Li, Zhenlong Yuan, Yujun Cai, and Zhaoqi Wang. Factguard: Agentic video misinformation detection via reinforcement learning. arXiv preprint arXiv:2602.22963, 2026.
- [43] Zixu Li, Yupeng Hu, Zhiwei Chen, Mingyu Zhang, Zhiheng Fu, and Liqiang Nie. Conesep: Conebased robust noise-unlearning compositional network for composed image retrieval, 2026. URL https: //arxiv.org/abs/2604.20358.
- [44] Zixu Li, Yupeng Hu, Zhiwei Chen, Shiqi Zhang, Qinlei Huang, Zhiheng Fu, and Yinwei Wei. Habit: Chrono-synergia robust progressive learning framework for composed image retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 6762–6770, 2026.
- [45] Zixu Li, Yupeng Hu, Zhiheng Fu, Zhiwei Chen, Yongqi Li, and Liqiang Nie. Tema: Anchor the image, follow the text for multi-modification composed image retrieval, 2026. URL https://arxiv.org/ abs/2604.21806.
- [46] Chenyang Liu, Rui Zhao, Jianqi Chen, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. A decoupling paradigm with prompt learning for remote sensing image change captioning. IEEE Transactions on Geoscience and Remote Sensing, 61:1–18, 2023.
- [47] Chenyang Liu, Keyan Chen, Haotian Zhang, Zipeng Qi, Zhengxia Zou, and Zhenwei Shi. Change-agent: Toward interactive comprehensive remote sensing change interpretation and analysis. IEEE Transactions on Geoscience and Remote Sensing, 62:1–16, 2024.
- [48] Chenyang Liu, Keyan Chen, Rui Zhao, Zhengxia Zou, and Zhenwei Shi. Text2earth: Unlocking textdriven remote sensing image generation with a global-scale dataset and a foundation model. IEEE Geoscience and Remote Sensing Magazine, 13(3):238–259, 2025. doi: 10.1109/MGRS.2025.3560455.
- [49] H. Liu, C. Li, Q. Wu, and Y.J. Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, 2024.
- [50] H. Liu et al. Multimodal segmentation with large vision-language models. arXiv preprint, 2025.
- [51] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [52] Haotian Liu, Chunyuan Li, Yuheng Li, et al. Llava-next: Improved reasoning, ocr, and world knowledge. arXiv preprint, 2024.
- [53] S. Liu, H. Cheng, H. Liu, H. Zhang, F. Li, T. Ren, X. Zou, J. Yang, H. Su, J. Zhu, L. Zhang, and J. Gao. Llava-plus: Learning to use tools for creating multimodal agents. arXiv preprint arXiv:2311.05437, 2023.
- [54] Chunhao Lu, Qiang Lu, Meichen Dong, and Jake Luo. End-to-end multi-modal diffusion mamba. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20529–20540, 2025.
- [55] Jinguo Luo, Weihong Ren, Quanlong Zheng, Yanhao Zhang, Zhenlong Yuan, Zhiyong Wang, Haonan Lu, and Honghai Liu. Instructhoi: Context-aware instruction for multi-modal reasoning in human-object interaction detection. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

- [56] Z. Ma, J. Zhang, Z. Liu, J. Zhang, J. Tan, M. Shu, J. C. Niebles, S. Heinecke, H. Wang, C. Xiong, and R. Krishna. Taco: Learning multi-modal action models with synthetic chains-of-thought-and-action. arXiv preprint arXiv:2412.05479, 2024.
- [57] J. Miao, P. Du, Y. Fan, Y. Liu, Y. Wang, R. He, L. Huang, and Y. Wang. Agentiad: Agentic industrial anomaly detection via adaptive memory augmentation. arXiv preprint arXiv:2512.13671, 2025.
- [58] H. Mu, J. Qiu, Y. Zheng, H. Qi, M. Chen, et al. Diffusionad: Norm-guided diffusion for anomaly detection. arXiv preprint arXiv:2303.08730, 2023.
- [59] OpenAI. Gpt-4v(ision) system card. arXiv preprint arXiv:2309.17421, 2023.
- [60] OpenAI. Learning to reason with large language models. OpenAI Blog, 2024. URL https://openai. com/index/learning-to-reason-with-llms/.
- [61] OpenAI. Hello gpt-4o. OpenAI Blog, 2024. URL https://openai.com/index/hello-gpt-4o/.
- [62] OpenAI. Gpt-4.1 technical report. arXiv preprint, 2025.
- [63] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, pages 27730–27744, 2022.
- [64] Y. Peng, G. Zhang, M. Zhang, Z. You, J. Liu, Q. Zhu, K. Yang, X. Xu, X. Geng, and X. Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.
- [65] Chengxuan Qian, Kai Han, Jianxia Ding, Chongwen Lyu, Zhenlong Yuan, Jun Chen, and Zhe Liu. Adaptive label correction for robust medical image segmentation with noisy labels. arXiv preprint arXiv:2503.12218, 2025.
- [66] Chengxuan Qian, Shuo Xing, Shawn Li, Yue Zhao, and Zhengzhong Tu. Decalign: Hierarchical crossmodal alignment for decoupled multimodal representation learning. arXiv preprint arXiv:2503.11892, 2025.
- [67] Xiangyan Qu, Zhenlong Yuan, Jing Tang, Rui Chen, Datao Tang, Meng Yu, Lei Sun, Yancheng Bai, Xiangxiang Chu, Gaopeng Gou, Gang Xiong, and Yujun Cai. From scale to speed: Adaptive test-time scaling for image editing. CoRR, abs/2603.00141, 2026. doi: 10.48550/ARXIV.2603.00141. URL https://doi.org/10.48550/arXiv.2603.00141.
- [68] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler. Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14318–14328, 2022.
- [69] M. Rudolph, T. Wehrbein, B. Rosenhahn, and B. Wandt. Fully convolutional cross-scale-flows for image-based defect detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1088–1097, 2022.
- [70] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, et al. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, 2024.
- [71] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [72] Z. Shao, P. Wang, Q. Zhu, R. Zheng, Y. Zhao, M. Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [73] Shuhan Song, Ping Li, Ming Dun, Maolei Huang, Huawei Cao, and Xiaochun Ye. Gpromptshield: Elevating resilience in graph prompt tuning against adversarial attacks. In The Thirteenth International Conference on Learning Representations, 2025.
- [74] Shuhan Song, Ping Li, Ming Dun, Yuan Zhang, Huawei Cao, and Xiaochun Ye. Equipping graph autoencoders: Revisiting masking strategies from a robustness perspective. In Proceedings of the 2025 SIAM International Conference on Data Mining (SDM), pages 366–375. SIAM, 2025.
- [75] Shuhan Song, Ping Li, Ming Dun, Yuan Zhang, Huawei Cao, and Xiaochun Ye. Spmgae: Self-purified masked graph autoencoders release robust expression power. Neurocomputing, 611:128631, 2025.
- [76] Zijian Song, Sihan Qin, Tianshui Chen, Liang Lin, and Guangrun Wang. Physical autoregressive model for robotic manipulation without action pretraining. arXiv preprint arXiv:2508.09822, 2025.

- [77] Zijian Song, Xiaoxin Lin, Tao Pu, Zhenlong Yuan, Guangrun Wang, and Liang Lin. Human-centric open-future task discovery: Formulation, benchmark, and scalable tree-based search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 17724–17732, 2026.
- [78] Qile Su, Jing Tang, Rui Chen, Lei Sun, and Xiangxiang Chu. Video-coe: Reinforcing video event prediction via chain of events. arXiv preprint arXiv:2603.14935, 2026.
- [79] Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, et al. Aligning large multimodal models with factually augmented rlhf. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [80] D. Tabernik, S. Šela, J. Skvarˇc, and D. Skoˇcaj. Spatially-adaptive filter units for deep neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9388–9396, 2020.
- [81] Datao Tang, Xiangyong Cao, Xingsong Hou, Zhongyuan Jiang, Junmin Liu, and Deyu Meng. Crs-diff: Controllable remote sensing image generation with diffusion model. IEEE Transactions on Geoscience and Remote Sensing, 2024.
- [82] Datao Tang, Xiangyong Cao, Xuan Wu, Jialin Li, Jing Yao, Xueru Bai, Dongsheng Jiang, Yin Li, and Deyu Meng. Aerogen: Enhancing remote sensing object detection with diffusion-driven data generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3614–3624, 2025.
- [83] X. Tao, D. Hou, D. Tao, J. Ru, C. Ren, S. Qu, et al. Deep learning for surface defect detection: a survey. IEEE access, 10:16466–16491, 2022.
- [84] C. Wang, X. Peng, J. Zhang, Y. Dong, W. Wang, J. Li, et al. Real-iad: A real-world multi-view dataset for industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [85] JiYuan Wang, Chunyu Lin, Lei Sun, Rongying Liu, Lang Nie, Mingxing Li, Kang Liao, Xiangxiang Chu, and Yao Zhao. From editor to dense geometry estimator. arXiv preprint arXiv:2509.04338, 2025.
- [86] Jiyuan Wang, Chunyu Lin, Lei Sun, Zhi Cao, Yuyang Yin, Lang Nie, Zhenlong Yuan, Xiangxiang Chu, Yunchao Wei, Kang Liao, et al. Geometry-guided reinforcement learning for multi-view consistent 3d scene editing. arXiv preprint arXiv:2603.03143, 2026.
- [87] Jiyuan Wang, Huan Ouyang, Jiuzhou Lin, Chunyu Lin, Dewen Fan, Boheng Zhang, Haonan Fan, Fei Zuo, Jia Sun, Huaiqing Wang, et al. Cac: Advancing video reward models via hierarchical spatiotemporal concentrating. arXiv preprint arXiv:2605.11723, 2026.
- [88] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N.A. Smith, D. Khashabi, and H. Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, 2023.
- [89] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, pages 24824–24837, 2022.
- [90] J. Wyatt, A. Leach, S.M. Schmon, and C.G. Willcocks. Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 650–656, 2022.
- [91] Jiacong Xu, Shao-Yuan Lo, Bardia Safaei, Vishal M. Patel, and Isht Dwivedi. Towards zero-shot anomaly detection and reasoning with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20370–20382, June 2025.
- [92] Zhidong Yang, Fa Zhang, and Renmin Han. Self-supervised cryo-electron tomography volumetric image restoration from single noisy volume with sparsity constraint. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4056–4065, October 2021.
- [93] Zhidong Yang, Xiuhui Shi, Wei Ba, Zhigang Song, Haijing Luan, Taiyuan Hu, Senlin Lin, Jiguang Wang, Shaohua Kevin Zhou, and Rui Yan. Fusion of multi-scale heterogeneous pathology foundation models for whole slide image analysis. 2025. URL https://arxiv.org/abs/2510.27237.
- [94] Hongwei Yu, Jiansheng Chen, Xinlong Ding, Yudong Zhang, Ting Tang, and Huimin Ma. Step vulnerability guided mean fluctuation adversarial attack against conditional diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 6791–6799, 2024.

- [95] Hongwei Yu, Xinlong Ding, Jiawei Li, Jinlong Wang, Yudong Zhang, Rongquan Wang, Huimin Ma, and Jiansheng Chen. Dadet: Safeguarding image conditional diffusion models against adversarial and backdoor attacks via diffusion anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17411–17421, 2025.
- [96] J. Yu, Y. Zheng, X. Wang, W. Li, L. Wu, et al. Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows. arXiv preprint arXiv:2111.07677, 2021.
- [97] Zhenlong Yuan, Xiangyan Qu, Chengxuan Qian, Rui Chen, Jing Tang, Lei Sun, Xiangxiang Chu, Dapeng Zhang, Yiwei Wang, Yujun Cai, et al. Video-star: Reinforcing open-vocabulary action recognition with tools. arXiv preprint arXiv:2510.08480, 2025.
- [98] Zhenlong Yuan, Jing Tang, Jinguo Luo, Rui Chen, Chengxuan Qian, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, and Shuo Li. Autodrive-r2: Incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. arXiv preprint arXiv:2509.01944, 2025.
- [99] V. Zavrtanik, M. Kristan, and D. Skoˇcaj. Draem: A discriminatively trained reconstruction embedding for surface anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8330–8339, 2021.
- [100] A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, and J. Tang. Agenttuning: Enabling generalized agent abilities for llms. In International Conference on Learning Representations, 2024.
- [101] Dapeng Zhang, Peng Zhi, Binbin Yong, Jin-Qiang Wang, Yufeng Hou, Lan Guo, Qingguo Zhou, and Rui Zhou. Ehss: An efficient hybrid-supervised symmetric stereo matching network. In 2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC), pages 1044–1051. IEEE, 2023.
- [102] Dapeng Zhang, Dayu Chen, Peng Zhi, Yinda Chen, Zhenlong Yuan, Chenyang Li, Rui Zhou, Qingguo Zhou, et al. Mapexpert: Online hd map construction with simple and efficient sparse map element expert. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 14745–14753, 2025.
- [103] Dapeng Zhang, Jing Sun, Chenghui Hu, Xiaoyan Wu, Zhenlong Yuan, Rui Zhou, Fei Shen, and Qingguo Zhou. Pure vision language action (vla) models: A comprehensive survey. arXiv preprint arXiv:2509.19012, 2025.
- [104] Dapeng Zhang, Zhenlong Yuan, Zhangquan Chen, Chih-Ting Liao, Yinda Chen, Fei Shen, Qingguo Zhou, and Tat-Seng Chua. Reasoning-vla: A fast and general vision-language-action reasoning model for autonomous driving. arXiv preprint arXiv:2511.19912, 2025.
- [105] Dapeng Zhang, Fei Shen, Rui Zhao, Yinda Chen, Peng Zhi, Chenyang Li, Rui Zhou, and Qingguo Zhou. Coc-vla: Delving into adversarial domain transfer for explainable autonomous driving via chainof-causality visual-language-action model. Advances in Neural Information Processing Systems, 38: 70912–70939, 2026.
- [106] Jianshu Zhang, Chengxuan Qian, Haosen Sun, Haoran Lu, Dingcheng Wang, Letian Xue, and Han Liu. Progresslm: Towards progress reasoning in vision-language models. arXiv preprint arXiv:2601.15224, 2026.
- [107] Wenbo Zhang, Chenxu Zhang, Yang Gao, and Zhanpeng Jin. Kineticssense: A multimodal wearable sensor framework for modeling lower-limb motion kinetics. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 9(3), September 2025. doi: 10.1145/3749462. URL https://doi.org/10.1145/ 3749462.
- [108] Yi Zhang, Yinda Chen, Che Liu, Zeyuan Ding, Jin Xu, Shilong Zou, Junwei Liao, Jiayu Hu, Xiancong Ren, Xiaopeng Zhang, et al. Pelican-unified 1.0: A unified embodied intelligence model for understanding, reasoning, imagination and action. Technical Report, 2026.
- [109] Z. Zhang, A. Zhang, M. Li, H. Zhao, G. Karypis, and A. Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.
- [110] S. Zhao, H. Zhang, S. Lin, M. Li, Q. Wu, K. Zhang, and C. Wei. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025.
- [111] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer. Spot-the-difference self-supervised pre-training for anomaly detection and segmentation. In European conference on computer vision, pages 392–408. Springer, 2022.

### Appendix

In this appendix, we provide more experimental details, tool library details, and prompt details of our proposed method. Specific detailed contents are as follows:

- A Detailed Reward Design 16
- B Experiment 17

- B.1 More Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 Baseline Prompting and Answer Parsing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.3 More Hyperparameter Configurations. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.4 More Experimental results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.5 Tool Usage Analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.6 More Ablation Studies. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- C Discussion on Novelty and Problem Setting 20
- D Tool Library Specifications 21

- D.1 Dynamic Region Cropping Tool (Tcrop) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.2 Normalcy Prior Explanation Tool (Tprior) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.3 Low-Level Visual Enhancer (Tenhance) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.4 Geometric Verifier (Tmeasure) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E Prompts 21
- F More Case Studies 24
- G Detailed Related Work 24
- H Limitations 27

### A Detailed Reward Design

The reward function is designed to balance three objectives: diagnostic correctness, fine-grained anomaly understanding, and efficient tool usage. A naive additive reward may assign positive scores to localization, anomaly-type prediction, or tool invocation even when the final anomaly judgment is wrong. This can encourage undesirable behaviors, such as hallucinating defect locations, overusing external tools, or exploiting formatting shortcuts. To avoid this issue, we adopt an accuracy-gated formulation in which the main task-related rewards are activated only when the final binary diagnosis is correct.

Specifically, for a trajectory τ, the overall reward is:

R(τ) = Racc(τ) · 1 + αRloc(τ) + βRtype(τ) + γRtool(τ) + Rformat(τ), (7)

where Racc is the binary anomaly classification reward, Rloc measures spatial grounding quality, Rtype evaluates fine-grained anomaly categorization, Rtool measures the utility of tool invocation, and Rformat encourages valid output formatting. The coefficients α, β, and γ control the relative importance of localization, semantic categorization, and tool usage.

Classification Accuracy. The term Racc ∈ {0,1} evaluates whether the final binary anomaly judgment is correct. It acts as the central multiplicative gate in the reward function. When Racc = 0, the trajectory receives no task-level credit from localization, anomaly-type prediction, or tool usage, even if these intermediate outputs appear plausible. This prevents the model from receiving high rewards for hallucinated defect explanations or visually ungrounded reasoning.

Spatial Localization. The localization reward Rloc evaluates whether the predicted anomaly region is spatially aligned with the ground-truth region. We compute this term using the Intersection over Union (IoU) between the predicted bounding box and the ground-truth anomaly box:

Rloc = IoU(Bpred, Bgt). (8)

This term encourages the agent to ground its diagnostic conclusion in the correct visual region rather than producing only a coarse image-level judgment.

Semantic Categorization. The semantic reward Rtype evaluates the predicted anomaly type. Instead of using only exact string matching, we compute this reward based on the semantic distance between the predicted anomaly category and the ground-truth category in a hierarchical anomaly taxonomy. This design allows partial credit for semantically close predictions while penalizing distant or unrelated defect categories more strongly.

Cost-Aware Tool Utility. The tool utility reward is designed to encourage effective evidence acquisition while discouraging unnecessary tool calls. Unlike a simple positive reward for every tool invocation, our formulation jointly considers marginal diagnostic benefit and execution cost:

Rtool = λ · I[∆conf > 0] − η|C|, (9) where C denotes the set of invoked tools, |C| is the number of valid tool calls, I[·] is the indicator function, and ∆conf measures the confidence improvement after incorporating tool feedback. The coefficient λ controls the bonus for useful evidence acquisition, while η penalizes the computational and reasoning cost of each tool call.

In practice, ∆conf can be estimated as the increase in confidence for the final predicted diagnostic label after tool observations are added:

∆conf = pθ(y∗ | I, Q, C, R) − pθ(y∗ | I, Q), (10) where y∗ denotes the final predicted diagnostic label, I is the input image, Q is the task instruction, C is the set of tool calls, and R denotes the corresponding tool observations. A positive ∆conf indicates that the tool feedback provides useful evidence for the final decision. The cost term −η|C| discourages redundant calls and prevents the agent from invoking tools indiscriminately.

Since Rtool is placed inside the Racc gate in the overall reward, beneficial tool use is rewarded only when the final diagnosis is correct. Thus, the model cannot obtain a high reward by simply calling more tools without improving the diagnostic outcome. This encourages a more selective tool-use policy, where the agent invokes external tools only when they are expected to provide meaningful diagnostic information.

Estimation of Confidence Improvement. Since the policy is an autoregressive multimodal language model, directly using the model’s free-form verbalized confidence is unreliable and may introduce additional reward hacking risks. Therefore, we estimate the confidence improvement ∆conf from the normalized logprobability margin of the final binary decision tokens, rather than from self-reported confidence scores.

Specifically, for each trajectory, we parse the final diagnostic decision from the structured <answer> field and map it into a binary label y ∈ {Yes, No}, where Yes denotes anomalous and No denotes normal. We then compute the model’s decision margin before and after incorporating tool observations. Given the original image

- I, instruction Q, tool calls C, and returned tool observations R, the confidence improvement is defined as: ∆conf = mθ(y | I, Q, C, R) − mθ(y | I, Q), (11)

where mθ(·) denotes the normalized binary log-probability margin:

mθ(y | X) = log pθ(y | X) − log pθ(¯y | X), (12)

and y¯ denotes the opposite binary label. In practice, pθ(y | X) is computed from the log-probability of the normalized answer token corresponding to Yes or No at the final decision position. This formulation compares the relative preference between the two valid diagnostic labels and is therefore less sensitive to response length, reasoning style, or formatting variations.

The tool utility reward is then activated only when the tool observations increase the model’s binary decision margin, i.e., ∆conf > 0. Importantly, this term is further placed inside the multiplicative accuracy gate Racc. As a result, the agent cannot obtain positive tool-utility reward by merely increasing confidence in an incorrect prediction or by reporting high confidence in natural language. This design encourages tools to be invoked only when they provide evidence that both strengthens the final diagnostic decision and leads to a correct prediction.

Process Compliance. The format reward Rformat is applied independently of the task-level gate. It penalizes malformed outputs, missing <think> or <answer> tags, invalid tool-call syntax, and other deviations from the required response structure. This term stabilizes RL training by preventing format collapse and ensuring that the generated trajectories remain parseable throughout optimization.

### B Experiment

#### B.1 More Experimental Details

Category-Disjoint Training Protocol. To prevent category leakage between training and evaluation, we explicitly compared the object categories in our Real-IAD training set against the union of categories from the

five evaluation benchmarks, namely DTD, MPDD, MVTec-AD, SDD, and VisA. We further performed semantic normalization to account for naming differences, such as pcb versus pcb1/pcb2/pcb3/pcb4, and transistor1 versus transistor. Based on this comparison, all exact or semantically equivalent overlapping categories were removed from Real-IAD, including toothbrush, zipper, pcb, and transistor1. The resulting RealIAD training set is therefore category-disjoint from all test benchmarks, ensuring that evaluation measures generalization to unseen industrial categories rather than memorization of category-specific visual patterns.

Evaluation Metrics and Inference Details. We evaluate IndusAgent in a strictly zero-shot industrial anomaly detection setting across five diverse benchmarks: MVTec, VisA, DTD, SDD, and MPDD. The model is trained exclusively on our proposed instruction-tuning and reinforcement learning data, with no dataset-specific finetuning. During inference, the model is prompted to identify defects, and its final response is normalized into a binary decision (Yes for anomalous, No for normal). For IndusAgent, this prediction is deterministically parsed from the strictly formatted <answer>...</answer> tags within its structured cognitive trajectory, whereas baseline predictions are extracted via heuristic rule-based matching from raw text. We benchmark against a diverse spectrum of proprietary APIs and leading open-source vision-language models. To counteract the extreme class imbalance and varying normal-to-anomaly ratios inherent in industrial inspection, we adopt balanced accuracy as our primary metric. We report dataset-level scores following official evaluation protocols and compute a macro-average across all datasets to ensure that no single benchmark dominates the overall comparison due to its scale.

#### B.2 Baseline Prompting and Answer Parsing

To ensure a fair comparison, all baseline models and IndusAgent are evaluated with the same binary anomaly detection instruction. For each query image, the model is asked to determine whether the image contains an anomaly and to provide a final answer in a normalized binary form, where Yes denotes anomalous and No denotes normal. No paired normal reference image, category-specific exemplar, or dataset-specific prompt is provided to any model during inference.

For IndusAgent, the final prediction is directly extracted from the structured <answer>...</answer> field. For baseline MLLMs, since most models do not naturally follow our internal structured output format, we apply a unified rule-based parser to normalize their responses into binary labels. Specifically, we first search for explicit final decisions such as “yes”, “anomalous”, “defective”, “abnormal”, “no”, “normal”, and “defect-free”. When both positive and negative expressions appear in the same response, we use the model’s final stated conclusion rather than intermediate reasoning sentences. This avoids incorrectly parsing exploratory descriptions or self-corrections as final predictions.

We further manually inspected the parsed outputs to ensure that the rule-based parser correctly reflected the models’ intended final judgments. In our evaluation, the outputs of all compared models could be normally parsed into valid binary decisions, and no model was excluded due to formatting failure. Ambiguous cases, if any, were resolved by checking the final conclusion sentence in the response while keeping the same decision criterion across all models. Thus, the reported performance differences are not caused by parser failures or model-specific output formatting advantages.

#### B.3 More Hyperparameter Configurations.

Table 5: Sensitivity Analysis of Reward Hyperparameters. We investigate the impact of the scaling factors (α,β,γ) on IndusAgent’s performance. Notably, a naïve uniform weighting (α = β = γ = 1) slightly degrades performance due to reward distraction. In contrast, our empirically tuned configuration (α = 0.8,β = 0.6,γ = 0.5) achieves the optimal balance between task efficacy and structural compliance.

α (Loc) β (Type) γ (Tool) MVTec MPDD VisA DTD SDD Avg.

0.0 0.0 0.0 76.5 64.2 68.3 88.1 81.4 75.7 0.8 0.0 0.0 80.2 68.5 72.1 91.5 84.6 79.4

- 0.8 0.6 0.0 82.1 71.3 75.0 93.8 86.5 81.7

- 1.0 1.0 1.0 81.2 70.5 74.1 93.2 86.5 81.1 0.8 0.6 0.5 83.6 72.7 76.8 95.6 88.9 83.4

All experiments were conducted on a single compute node equipped with four NVIDIA A100 GPUs, each with 80GB memory. We used Qwen3-VL-8B-Instruct as the backbone model. The supervised fine-tuning stage was trained for one epoch on the RealIAD-3K instruction data and took approximately 21.6 minutes. The subsequent reinforcement learning stage was implemented with our answer-strict recall-guarded GRPO training pipeline and optimized for one epoch using four generations per prompt. This stage took approximately 23.4 hours. During RL training, we used a per-device batch size of 1, gradient accumulation steps of 2, a maximum prompt length

of 4096, a maximum completion length of 512, bfloat16 precision, gradient checkpointing, and DeepSpeed ZeRO-3.

#### B.4 More Experimental results.

As evidenced by the results in Table 6, the pronounced performance gap between generalpurpose MLLMs and IndusAgent exposes the inherent limitations of passive visual perception in zero-shot industrial scenarios. Both Qwen3VL and Claude-4-sonnet suffer from what we term an "industrial alignment tax"—their internal representations, heavily optimized for conversational heuristics and macroscopic object recognition, inherently lack the granular resolution required for micro-defect localization. Consequently, they resort to aggressive over-reporting or hallucinate anomalies amid complex textures like DTD. By contrast, IndusAgent circumvents this passive bottleneck. By actively deploying Tcrop and Tenhance to dynamically disambiguate visual noise, alongside Tmeasure to enforce strict geometric constraints, our framework transforms anomaly detection from a passive guessing task into a rigorous, verifiable reasoning pipeline.

Table 6: F1-Score Comparison. Metrics are aggregated over four shared datasets to ensure strict fairness.

Model DTD MPDD MVTec VisA

Qwen3-VL-8B 84.8 66.2 77.8 58.3 Claude-4-Sonnet 88.5 69.8 81.2 64.5 IAD-R1 (Qwen2.5-VL-7B) 91.1 75.2 85.7 70.8

IndusAgent (Ours) 96.1 78.2 89.8 78.2

#### B.5 Tool Usage Analysis.

Table 7 details the empirical tool invocation distributions across three major benchmarks. Crucially, IndusAgent exhibits a highly selective, costaware policy rather than resorting to exhaustive tool calls, maintaining an average invocation rate near or below 1.0 per query. The agent dynamically adapts its strategy to the underlying data distribution: Tcrop dominates on object-centric datasets like MVTec (62.4%) and VisA (54.8%) to isolate finegrained structural defects, whereas Tenhance is preferentially routed for the texture-centric DTD benchmark (34.7%) to disambiguate high-frequency sur-

Table 7: Tool Usage Statistics. Invocation frequency and success rates during zero-shot inference.

Metric MVTec VisA DTD

Crop (%) 62.4 54.8 28.5 Prior (%) 18.7 24.6 6.2 Enhance (%) 21.3 18.9 34.7 Measure (%) 9.8 7.5 2.1

Avg. Calls 1.12 1.06 0.72 Success Rate (%) 99.1 98.7 99.4

face noise. Furthermore, the specialized Tmeasure and Tprior tools are invoked sparingly, strictly reserved for severe geometric deformations or when explicit semantic baselines are necessitated. Coupled with a near-perfect execution success rate (> 98%), these statistics substantiate that our Agentic RL framework successfully cultivates a dataset-adaptive, precision-driven inspection paradigm.

#### B.6 More Ablation Studies.

Deconstructing Individual Tool Utility. Table 8 provides a granular ablation of the cohesive toolset, confirming that each module addresses distinct perceptual and cognitive bottlenecks inherent to industrial inspection. The ablation of the Dynamic Region Cropping tool (w/o Crop) precipitates the most severe degradation on the VisA dataset (from 76.8% to 68.6%), underscoring its indispensability for isolating micro-defects from intricate normal backgrounds. Conversely, removing the Low-Level Visual Enhancer (w/o Enhance) disproportionately impacts the DTD benchmark (dropping to 88.8%), revealing that high-frequency texture enhancement is critical for navigating severe domain-specific noise. Furthermore, omitting the Geometric Verifier (w/o Measure) and Normalcy Prior (w/o Prior) induces consistent performance decay across all benchmarks. This collective evidence demonstrates that tailoring inference pathways strictly for industrial settings demands a synergistic, multi-dimensional verification strategy rather than reliance on a single augmented modality.

Table 8: Ablation of individual tools.

Method MVTec VisA DTD IndusAgent 83.6 76.8 95.6 w/o Crop 79.4 68.6 91.2 w/o Prior 81.1 69.3 92.1 w/o Enhance 80.5 70.8 88.8 w/o Measure 81.5 70.2 91.5

Table 9: Ablation on the number of generated candidates per prompt.

Impact of GRPO Group Size. Table 9 analyzes the sensitivity of IndusAgent to the number of generated candidates (Group Size) per prompt during the Agentic RL phase. Scaling the group size from 2 to 6 consistently refines the advantage estimation in GRPO, reducing policy update variance and peaking at an accuracy of 84.1%, 77.2%, and 96.0% on MVTec, VisA, and DTD, respectively. However, performance saturates and slightly degrades at a group size of 8, likely due to optimization noise from over-exploration. Crucially, the marginal gains achieved by scaling from 4 to 6 do not justify the substantial increase in memory overhead. To ensure scalable training and maintain our efficiency-aware paradigm—particularly when managing intensive containerized workloads on high-performance computing clusters—we adopt a group size of 4 as our optimal default, striking an ideal balance between diagnostic precision and resource efficiency.

Num of Gen. (Num.) MVTec VisA DTD

Num. = 2 81.5 74.3 93.2 Num. = 4 83.6 76.8 95.6 Num. = 6 84.1 77.2 96.0 Num. = 8 83.9 77.0 95.8

### C Discussion on Novelty and Problem Setting

Beyond a direct application of tool-augmented RL. A natural question is whether IndusAgent is simply an application of existing tool-augmented MLLM and reinforcement learning techniques to industrial anomaly detection. We argue that the main contribution of IndusAgent does not lie in introducing a generic tool-use interface, but in reformulating open-vocabulary industrial anomaly detection as a reference-free, category-disjoint, active inspection problem and designing the data construction, tool orchestration, and reward optimization accordingly. Unlike general visual reasoning tasks, industrial anomaly detection requires the model to distinguish subtle defects from legitimate structural variations, often without paired normal references, target-category training samples, or predefined defect vocabularies. This makes naive tool use insufficient: an agent that merely invokes more tools may amplify visual noise, hallucinate defect locations, or overfit to category-specific priors. Therefore, our framework explicitly couples tool invocation with diagnostic correctness and open-vocabulary generalization rather than treating tools as auxiliary modules that are always beneficial.

Difference from existing IAD agents. Recent agentic IAD methods usually assume a more structured or in-domain setting, where the model can rely on category-specific supervision, known defect distributions, or reference normal samples during training or inference. In contrast, our evaluation follows a stricter category-disjoint protocol: the training trajectories are constructed from Real-IAD after removing all categories overlapping with the evaluation benchmarks, including MVTec-AD, VisA, MPDD, DTD, and SDD. This protocol requires the model to transfer its diagnostic behavior to unseen object categories and defect types, rather than memorizing category-specific appearance patterns. Moreover, IndusAgent does not use paired normal reference images at inference time. The agent must infer normalcy from the query image, learned industrial priors, and selectively acquired tool feedback. This setting is closer to practical open-vocabulary inspection, where new products and previously unseen defects frequently appear.

Why accuracy-gated reward is necessary. A standard additive reward can assign positive scores to intermediate outputs, such as localization, anomaly type prediction, or tool invocation, even when the final anomaly judgment is incorrect. This is particularly harmful in industrial anomaly detection, because a model may hallucinate a plausible defect region or invoke redundant tools while still making a wrong diagnosis. Our accuracy-gated reward addresses this issue by using the final binary diagnostic correctness as a multiplicative gate for localization, type reasoning, and tool utility rewards. As a result, the agent receives task-level credit for tool use only when the collected evidence contributes to a correct final decision. This differs from generic tool-use rewards that encourage tool invocation itself, as well as from format-only rewards that merely stabilize output structure. The gated design makes tool use a diagnostic instrument rather than an objective in itself, which is essential for avoiding tool abuse and visually ungrounded reasoning.

Role of Indus-CoT. Indus-CoT is not intended to be a new evaluation benchmark. Instead, it serves as a tool-integrated training corpus for aligning the model with industrial diagnostic trajectories. Its purpose is to provide supervision for how an agent should inspect an image: first forming a global hypothesis, then deciding whether additional evidence is needed, invoking suitable tools, and finally verifying the anomaly judgment with localized or semantic evidence. This distinguishes Indus-CoT from ordinary CoT data, which only supervises textual reasoning, and from conventional IAD datasets, which usually provide image-level or pixel-level labels but not tool-grounded inspection processes. By explicitly linking global perception, tool feedback, and final diagnosis, Indus-CoT provides the structured initialization needed before reinforcement learning.

Summary. Overall, IndusAgent should be viewed as a framework for open-vocabulary active industrial inspection, rather than a direct transfer of generic agentic RL to IAD. Its novelty lies in the combination of: (1) a stricter category-disjoint and reference-free problem setting; (2) a tool-integrated diagnostic corpus that

supervises active inspection behavior; (3) an accuracy-gated reward that prevents tool invocation from being rewarded independently of diagnostic correctness; and (4) empirical validation showing that different tools are selected according to dataset-specific inspection demands.

### D Tool Library Specifications

This section provides a comprehensive specification of the external tools orchestrated by our IndusAgent framework. These tools are systematically designed to resolve perceptual dilution and structural hallucinations in complex industrial scenarios.

- D.1 Dynamic Region Cropping Tool (Tcrop)

To address the bottleneck of uniform visual compression in standard MLLMs, we employ a Dynamic Region Cropping module. Unlike static global perception, this tool operates as an active attention mechanism. When the agent suspects a morphological deviation, Tcrop extracts a high-resolution, localized patch centered on the coordinates of interest. This isolated cropping mechanism preserves high-frequency spatial details—such as microscopic scratches or subtle textural inconsistencies—preventing them from being diluted by vast normal backgrounds. By adaptively increasing the localized visual fidelity, Tcrop significantly bolsters the agent’s capacity to verify imperceptible minor flaws.

- D.2 Normalcy Prior Explanation Tool (Tprior)

Industrial components often exhibit intricate, category-specific geometries that MLLMs easily confuse with true anomalies. To mitigate these structural hallucinations, we introduce the Normalcy Prior Explanation tool. Powered by an external domain-knowledge retriever (or API), Tprior provides verified, semantic descriptions of a component’s legitimate structural baseline (e.g., "the capacitor surface should possess a smooth, metallic sheen with two symmetrical solder joints"). By grounding the agent’s multimodal reasoning upon this explicit expert baseline, Tprior effectively prevents the conflation of acceptable geometric variations with critical structural defects.

- D.3 Low-Level Visual Enhancer (Tenhance)

Industrial surfaces frequently present challenging lighting conditions, such as severe metallic reflections or low-contrast anomalies (e.g., faint stains). To tackle this visual ambiguity, Tenhance equips the agent with adaptive image-processing capabilities. Upon invocation, it executes lightweight computer vision operators in the background—such as Canny edge detection or Contrast Limited Adaptive Histogram Equalization (CLAHE). By returning a noise-suppressed, high-frequency texture map, this tool effectively mitigates the perceptual blindness of raw visual encoders, forcing the MLLM to focus on critical morphological cues rather than illumination artifacts.

- D.4 Geometric Verifier (Tmeasure)

For structurally intricate workpieces like printed circuit boards (PCBs) or threaded screws, anomalies often manifest as spatial deviations—such as improper spacing or bending—rather than distinct missing or extraneous parts. Standard VLMs inherently lack precise physical scale awareness to detect these issues. The Geometric Verifier resolves this by allowing the agent to input specific reference coordinates. In return, Tmeasure computes and provides the exact physical (or pixel) distance and angular relationship between these points. This explicit metric feedback seamlessly transitions the agent’s reasoning from qualitative visual guessing to rigorous quantitative verification.

### E Prompts

We summarize the prompt templates used throughout our training and evaluation pipeline. The prompts cover SFT data construction, reinforcement learning, zero-shot baseline evaluation, structured inference, tool-augmented inference, and external visual-prior analysis.

SFT data construction. The SFT data construction stage converts industrial inspection images and annotations into expert-style reasoning targets.

###### Stage-1 Visual Description Prompt

You are constructing a high-quality tool-integrated industrial anomaly detection dataset. Analyze the provided query image to evaluate its structural and surface integrity. Your goal is to generate a detailed visual description and determine whether additional tool-based

evidence would be useful for reliable diagnosis.

Product category: {CATEGORY}. View id: {VIEW_ID}.

Ground-truth supervision constraints for target construction:

- - Image-level anomaly label: {normal or abnormal}.
- - Annotated anomaly location: {LOCATION or N/A}.
- - Coarse anomaly type: {ANOMALY_TYPE or N/A}. Important:

- 1. The supervision constraints are provided only to ensure that the constructed training target is consistent with the dataset annotation.
- 2. Do not state or imply that these annotations are available to the model during inference.
- 3. Do not mention the existence of ground-truth labels, annotated locations, or anomaly types in the generated reasoning trajectory.
- 4. No paired normal reference image is provided. Normal appearance should be inferred from the query image, product category, view information, and general industrial normalcy priors.
- 5. If the sample is normal, describe normal structural and surface evidence without fabricating a defect.
- 6. If the sample is anomalous, describe image-grounded evidence that is consistent with the annotated anomaly location and coarse anomaly type.

Available tools:

- 1. T_crop: Crop a suspicious local region for high-resolution inspection.
- 2. T_prior: Retrieve normal appearance priors for the corresponding product category and view.
- 3. T_enhance: Generate high-frequency or contrast-enhanced texture maps using lightweight visual operators.
- 4. T_measure: Compute distances, angles, or geometric relations between specified reference coordinates.

Output requirements:

- 1. Provide a detailed global visual description of the query image.
- 2. Identify suspicious or ambiguous regions only when they are visually supported.
- 3. Explain which tool or tools should be invoked if additional evidence is needed.
- 4. For normal samples, tool use is allowed only when it helps verify ambiguous but ultimately normal structures.
- 5. For anomalous samples, tool use should focus on verifying the visually grounded defect evidence.
- 6. Keep the description objective, image-grounded, and consistent with the supervision constraints.

###### Stage-2 Target Formatting Prompt

You are writing the final tool-integrated industrial anomaly detection training target. Convert the detailed description into a concise expert-style multi-round reasoning trajectory.

Ground-truth supervision constraints for target construction:

- - Final anomaly answer: {Yes or No}.
- - Annotated anomaly location: {LOCATION or N/A}.
- - Coarse anomaly type: {ANOMALY_TYPE or N/A}. Important:

- 1. The supervision constraints are used only to construct a correct training target.
- 2. Do not mention that the answer, location, or anomaly type is provided by annotation.
- 3. The generated trajectory should look like an inference-time diagnostic process based on the query image and tool observations.
- 4. No paired normal reference image is available.
- 5. The final answer must be consistent with the supervision constraints. Available tools:

- 1. T_crop: Crop a suspicious local region for high-resolution inspection.
- 2. T_prior: Retrieve normal appearance priors for the corresponding product category and view.
- 3. T_enhance: Generate high-frequency or contrast-enhanced texture maps using lightweight visual operators.
- 4. T_measure: Compute distances, angles, or geometric relations between specified reference coordinates.

Output rules:

- 1. Output only the final tagged answer.
- 2. A non-empty <think> block is mandatory.
- 3. If additional evidence is needed, include a <call_tool> block after the first <think> block.

- 4. The <call_tool> block should invoke one or multiple tools from T_crop, T_prior, T_enhance, and T_measure. Specify the tool name and the target region, coordinate, or query if applicable.
- 5. If a <call_tool> block is used, include an <observation> block to summarize the returned evidence, such as cropped local details, normalcy priors, enhanced texture cues, or geometric

measurements.

- 6. After the <observation> block, include a second <think> block that verifies whether the observed deviation is a real defect or a normal structural/texture variation.
- 7. If the final answer is No, output: <think>...</think><answer>No</answer> or, when tools are needed: <think>...</think><call_tool>...</call_tool><observation>...</observation><think>...</think><answer>

No</answer>

- 8. If the final answer is Yes, output: <think>...</think><location>...</location><type>...</type><answer>Yes</answer> or, when tools are needed: <think>...</think><call_tool>...</call_tool><observation>...</observation><think>...</think><location

>...</location><type>...</type><answer>Yes</answer>

- 9. Keep all reasoning image-grounded and consistent with the tool observations.
- 10. If the sample is normal, do not fabricate a defect region, defect type, or abnormal explanation.
- 11. If the sample is anomalous, the final location and type must be consistent with the annotated anomaly location and coarse anomaly type.
- 12. Do not use markdown fences, bullet points, or plain text outside the XML-style tags.

Training prompt. The same image-level inspection prompt is used for SFT and RL optimization.

###### Industrial Anomaly Detection Training Prompt

You are an expert industrial defect inspector. Your task is to determine whether the query image contains any industrial anomaly.

You should inspect the image carefully and provide a structured diagnostic response. If the global image is sufficient for diagnosis, directly reason and answer. If additional evidence is needed, you may invoke available tools to inspect suspicious regions,

retrieve normalcy priors, enhance subtle textures, or measure geometric relations. Available tools:

- 1. T_crop: Crop a suspicious local region for high-resolution inspection.
- 2. T_prior: Retrieve normal appearance priors for the corresponding product category and view.
- 3. T_enhance: Generate high-frequency or contrast-enhanced texture maps using lightweight visual operators.
- 4. T_measure: Compute distances, angles, or geometric relations between specified reference coordinates.

Output format: If no tool is needed: <think>[Your visual observation and reasoning]</think><answer>[Yes or No]</answer>

If tool use is needed: <think>[Your initial observation and reason for tool use]</think><call_tool>[Tool name and arguments

]</call_tool><observation>[Tool-returned evidence]</observation><think>[Verification based on the tool observation]</think><answer>[Yes or No]</answer>

If the final answer is Yes, also provide: <location>[Anomaly location]</location><type>[Anomaly type]</type>

Use Yes to indicate anomalous and No to indicate normal. Do not mention ground-truth labels, annotations, or training supervision. Do not fabricate defects if the image appears normal.

{Question}

Evaluation prompts. For baseline models, we use a direct zero-shot inspection prompt.

###### Baseline Zero-shot Prompt

You are an expert industrial quality control inspector. Inspect the given image carefully for localized anomalies.

Output ONLY this format: <think>short visual evidence </think> <location>anomaly position or none</location> <type>anomaly type or none</type> <answer>Yes or No</answer>

Tool-augmented inference. The tool-based inference process uses a two-round protocol. The first round routes the sample to auxiliary tools when needed, and the second round makes the final decision with tool feedback.

###### First-round Tool Routing Prompt

This is round one. Your main goal is tool routing before the final judgment. Available tools:

- 1. crop: magnifies small, local cues for high-resolution inspection.
- 2. enhance: applies low-level CV filters (e.g., CLAHE, edge detection) to suppress reflections and highlight low-contrast or ambiguous textures.
- 3. measure: computes precise distances or angles between reference points to verify geometric spacing and structural bending.
- 4. prior: provides object-prior and structural baseline analysis from domain experts.
- 5. none: use only when the image is already clearly normal or clearly abnormal.

Output ONLY this format: Think: <short visual evidence and reasoning> Need Tools: <crop/enhance/measure/qwen3vlmax-api/none, or a comma-separated combination> Tool Target: <short region or cue, or none> Target Region: <region name> Target Scale: <tiny/small/medium/large/unknown> Target Type: <edge/corner/spacing/bending/surface-mark/component/texture/unknown> Suspicion Level: <low or medium or high> Preliminary Answer: <Yes or No>

###### Second-round Tool-aware Decision Prompt

This is round two. Re-evaluate the image with the returned tool results as auxiliary context. Tool outputs can help locate or explain suspicious regions, but the final answer must still be grounded in visible evidence.

Tool Context: {tool_context}

Output ONLY this format: <think>short visual evidence with tool help if relevant</think> <location>anomaly position or none</location> <type>anomaly type or none</type> <answer>Yes or No</answer>

- F More Case Studies
- G Detailed Related Work

This section extends the concise related work in the main paper, which discusses three central lines: openvocabulary industrial anomaly detection, reasoning in multimodal LLMs, and tool-augmented agentic systems. We provide a broader discussion to clarify how IndusAgent is positioned with respect to anomaly detection, multimodal reasoning, knowledge faithfulness, active tool use, vision-language-action models, generative visual modeling, structured representations, and specialized visual perception systems.

Open-vocabulary industrial anomaly detection. Industrial anomaly detection has traditionally been studied under closed-set or category-specific assumptions. Reconstruction-based methods learn normal visual patterns and identify anomalies through reconstruction residuals, as represented by autoencoder-based and synthetic-anomaly approaches such as DRAEM and related reconstruction models [99, 6]. Diffusion-based anomaly detectors further improve generative reconstruction quality, but they may also reconstruct abnormal regions and reduce the separability between normal and defective samples [90, 58]. Feature-embedding methods, including patch-level memory-bank models and probabilistic feature modeling, achieve strong in-distribution performance by comparing local features against normal training distributions [68, 27], while flow-based methods improve density modeling through invertible transformations [96, 69]. However, these methods usually require category-specific normal data and are therefore less suitable for open-vocabulary inspection. Vision-language approaches such as WinCLIP and AnomalyGPT exploit cross-modal semantic priors to improve zero-shot or few-shot anomaly reasoning [38, 31]. In contrast to these passive single-pass paradigms, IndusAgent formulates anomaly inspection as an active diagnostic process in which the model can gather local evidence, compare against normalcy priors, and reason over tool feedback.

[Figure 5]

Figure 5: Case Study between Qwen3-VL-8B and our method.

Multimodal reasoning, knowledge faithfulness, and post-training. Recent work on LLM and MLLM reasoning shows that structured intermediate reasoning and post-training can improve task performance beyond direct answer prediction. Chain-of-thought prompting and multimodal reasoning datasets provide the foundation for our Indus-CoT construction [89, 109]. Reinforcement-learning-based reasoning paradigms, including OpenAI-o1 and DeepSeek-R1, demonstrate that post-training can strengthen deliberative reasoning and self-verification [60, 26]. This direction has been extended to multimodal settings such as mathematical VQA, reasoning segmentation, and video reasoning [64, 50, 30]. Complementary studies analyze the dynamics of reasoning itself, such as CoT-Kinetics for modeling the reasoning process [14], PROGRESSLM for progressoriented vision-language reasoning [106], and adaptive test-time scaling for image editing [67].

Another relevant line of work studies how language models balance parametric knowledge, contextual evidence, and factual confidence. Decoding by Contrasting Knowledge improves model confidence on edited facts by contrasting internal and external knowledge sources [9]. Context-DPO aligns language models toward context-faithful generation, reducing unsupported reliance on parametric memory when contextual evidence is available [8]. Parameters vs. Context further investigates fine-grained control over whether models rely on stored parameters or input context [12]. These works are closely related to our setting because industrial anomaly detection also requires the model to avoid hallucinated structural assumptions and instead ground its final diagnosis in observable image evidence and retrieved normalcy priors.

From the data and optimization perspective, PRISM studies training-free multimodal data selection [13], while RefineX learns to refine pre-training data at scale from expert-guided programs [10]. EchoRL explores reinforcement learning through rollout echoing [15], and rubric-guided reward design promotes exploration across multiple reasoning domains [11]. These studies motivate our design choice of combining structured SFT with an accuracy-gated RL objective: rather than merely encouraging longer reasoning or more frequent tool use, IndusAgent rewards reasoning trajectories only when they improve the final diagnostic outcome.

Tool-augmented and agentic visual systems. Tool-augmented agents extend the capability of foundation models by allowing them to call external modules, interact with structured observations, and revise their reasoning based on feedback. Prior works such as Toolformer and AgentTuning show that tool invocation can be learned or optimized for general language agents [70, 100]. In multimodal settings, LLaVA-Plus, VPD, TACO, PyVision, and MVoT introduce different ways of combining visual reasoning with tool calls, programmatic execution, or multimodal thoughts [53, 37, 56, 110, 41].

Tool-augmented and reinforcement-based reasoning has also appeared in video understanding and evaluation. Video-STAR reinforces open-vocabulary action recognition with tools [97], while FactGuard studies agentic video misinformation detection through reinforcement learning [42]. FINGER introduces content-aware fine-grained evaluation with reasoning for AI-generated videos [21], and Video-CoE reinforces video event

[Figure 6]

Figure 6: Case Study between Qwen3-VL-8B and our method.

prediction through chain-of-events reasoning [78]. These works suggest that tool use and structured reasoning are particularly useful when the target evidence is fine-grained, temporally distributed, or difficult to capture through a single passive visual encoding.

Instruction-aware and design-oriented agentic systems further demonstrate the generality of tool-based multimodal reasoning. InstructHOI shows the value of context-aware instruction following for human-object interaction detection [55], while AnyLayout studies versatile advertising poster layout generation with MLLMs [1]. Unlike these general-purpose or task-specific agents, IndusAgent focuses on industrial anomaly inspection, where tool calls must be both visually grounded and cost-aware. Our reward design therefore gates tool utility with diagnostic correctness to discourage indiscriminate tool invocation.

Vision-language-action and embodied reasoning. A related line of work studies vision-languageaction models, where perception, reasoning, and action are integrated into a unified policy. Survey work on pure VLA models summarizes this trend toward direct multimodal action generation [103]. In autonomous driving, Reasoning-VLA, CoC-VLA, and AutoDrive-R2 explore how reasoning, causality, and self-reflection can improve decision-making in complex dynamic scenes [104, 105, 98]. Pelican-Unified aims to unify understanding, reasoning, imagination, and action in embodied intelligence [108], while physical autoregressive modeling investigates robotic manipulation without action pretraining [76]. Open-future task discovery further considers how agents can discover and organize future human-centric tasks [77].

Although industrial anomaly detection does not require physical actuation, it shares the need for sequential decision-making: the model must decide whether to inspect local regions, retrieve priors, enhance texture, or measure geometry before making a final judgment. Related perception modules in autonomous systems, such as online HD map construction and efficient stereo matching, also reflect the importance of structured spatial understanding [102, 101]. Beyond high-level action reasoning, structured spatial perception is also essential for embodied and industrial visual systems. For example, cross-view tracking for multi-human 3D pose estimation demonstrates how multi-view geometric cues can support accurate and efficient spatial understanding [18]. Although IndusAgent does not perform 3D pose estimation, it shares the same broader motivation of using structured visual evidence, rather than relying solely on global image-level semantics, to improve fine-grained reasoning.

Generative visual modeling, geometry, and robustness. Generative models provide useful priors for visual synthesis, editing, and scene understanding. Multi-modal diffusion Mamba studies end-to-end multimodal diffusion modeling [54], while Layout2Scene and Graph2Scene use semantic layouts or interaction-aware graphs to guide 3D scene generation [19, 20]. Geometry-aware editing and dense geometry estimation explore how image editing can be constrained or interpreted through 3D structure [86, 85].

In remote sensing, CRS-Diff, AeroGen, Text2Earth, Change-Agent, and decoupled prompt learning for change captioning study generation, object detection, text-driven synthesis, and interactive interpretation under largescale visual-domain shifts [81, 82, 48, 47, 46]. Robustness of diffusion models is another important issue, as shown by adversarial and backdoor detection for conditional diffusion models and step-vulnerability-guided diffusion attacks [95, 94]. These works are not industrial anomaly detectors directly, but they are relevant to IndusAgent because subtle defects often require geometry-aware comparison, texture-sensitive enhancement, and robustness against misleading visual artifacts.

Cross-modal retrieval, alignment, and structured representations. Cross-modal alignment and retrieval methods provide another perspective on open-vocabulary visual reasoning. DecAlign studies hierarchical cross-modal alignment for decoupled multimodal representation learning [66]. Composed image retrieval methods such as ConeSep, TEMA, and HABIT investigate how visual and textual modifications can be jointly modeled for robust retrieval under complex user instructions [43, 45, 44]. These methods emphasize the importance of disentangling visual evidence from linguistic intent, which is also crucial for open-vocabulary anomaly inspection where the model must distinguish true defects from benign variations described by language.

Multi-view and graph-based representation learning further shows the benefit of robust structured representations under incomplete or heterogeneous observations. Sampling-enhanced contrastive multi-view clustering and prototype-driven attribute-missing graph clustering study how representations can be learned under long-short range dependencies or missing attributes [32, 33]. Multi-scale graph learning also provides useful inspiration for structured perception under sparse or incomplete observations. For example, anti-sparse downscaling with multi-scale graph learning studies how graph structures can propagate information across different spatial resolutions [29]. This is conceptually related to industrial inspection, where subtle local defects must be interpreted together with global object structure and multi-scale contextual cues. Vocabulary recommendation for spatiotemporal data discovery also highlights how structured semantic resources can support data interpretation across domains [25].

Specialized visual perception in scientific, medical, graph, and wearable domains. Several related works address robust perception under domain-specific noise, limited labels, or specialized sensors. In scientific and medical imaging, self-supervised cryo-electron tomography restoration, pathology foundation-model fusion, adaptive label correction for noisy medical segmentation, self-supervised neuron segmentation with multi-agent reinforcement learning, and evolutionary medical prompt optimization all study how to improve reliability in high-stakes visual analysis settings [92, 93, 65, 22, 23]. Wearable sensing systems such as KineticsSense and PPGSpeech further demonstrate the broader role of multimodal sensor fusion for fine-grained perception and inference [107, 36].

In graph learning, self-purified masked graph autoencoders, robustness-aware masking strategies, and adversarially robust graph prompt tuning study how representation learning systems can remain stable under noise or attacks [75, 74, 73]. Although these works target different domains, they share with IndusAgent the central motivation of making model predictions more robust by incorporating structured priors, external feedback, or robustness-aware training objectives. Finally, hierarchical spatiotemporal reward modeling for video further supports the broader trend of using structured reward signals to align multimodal models with complex perceptual judgments [87].

### H Limitations

Despite its promising performance, IndusAgent still has several limitations. First, the active inspection process introduces additional inference overhead compared with single-pass MLLM inference, since external tools such as cropping, enhancement, and prior retrieval require extra computation. Second, the framework depends on the reliability of tool feedback; inaccurate crops, noisy enhanced maps, or incomplete normalcy priors may mislead the agent and affect the final diagnosis. Third, our current experiments mainly focus on image-level anomaly judgment, while more fine-grained evaluations, such as pixel-level localization, region-level grounding, and tool-use efficiency analysis, are needed to better understand the agent’s diagnostic behavior. Finally, Indus-CoT is generated with the assistance of a strong teacher model and rule-based validation, which may introduce teacher or prompt-template bias. Future work will explore more efficient tool-use policies, stronger tool robustness, and more diverse expert supervision for practical industrial deployment.

### NeurIPS Paper Checklist

###### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes] Justification: We emphasize the contributions and scope in the Introduction. Guidelines:

- • The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

###### 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: The limitation of the proposed algorithm has been discussed in the supplementary material.

Guidelines:

- • The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate “Limitations” section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

###### 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A] Justification: There is no theoretical result. Guidelines:

- • The answer [N/A] means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.

- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

###### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes] Justification: We provide comprehensive implementation details both in main paper and in supplementary material. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

###### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [No] Justification: As we promised, the data and code will be released upon the publication of our paper. Guidelines:

- • The answer [N/A] means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://neurips.cc/public/ guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

###### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes] Justification: The experimental setup, including data splits, training and testing detailed, are provided in Method and Experiments sections. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

###### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No] Justification: We follow the default evaluations in our Open-Vocabulary Industrial Anomaly Detection field, which doesn’t require error bars. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- • If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

###### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: We provide them in implementation details of main paper and supplementary material. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

###### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: This work conforms the NeurIPS Code of Ethics. Guidelines:

- • The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

###### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes] Justification: The border impacts is provided in supplementary material. Guidelines:

- • The answer [N/A] means that there is no societal impact of the work performed.
- • If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

###### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A] Justification: The proposed method uses pre-trained models. This proposed methods is safe under the safeguards of adopted pre-trained models. Guidelines:

- • The answer [N/A] means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

###### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: We cited the original paper that produced the code package or dataset. Guidelines:

- • The answer [N/A] means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

###### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A] Justification: There is no new assets released in this work. Guidelines:

- • The answer [N/A] means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

###### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

###### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A] Justification: There is no research with human subjects in this work. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

###### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A] Justification: LLM does not impact the core methodology, scientific rigorousness, or originality of the research. Guidelines:

- • The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.

