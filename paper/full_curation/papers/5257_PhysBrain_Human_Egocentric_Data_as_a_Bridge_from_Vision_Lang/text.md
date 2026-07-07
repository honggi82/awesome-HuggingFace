# arXiv:2512.16793v2[cs.RO]4Feb2026

[Figure 1]

## PhysBrain: Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence

Xiaopeng Lin1,3,∗, Shijie Lian2,6,∗, Bin Yu2,5,∗, Ruoqi Yang4, Zhaolong Shen2, Changti Wu2, Yuzhuo Miao2,5, Yurun Jin3, Yukun Shi3, Jiyan He2,3, Cong Huang2,3, Bojun Cheng†1, Kai Chen†2,3,4

1The Hong Kong University of Science and Technology (Guangzhou) 2Zhongguancun Academy 3Zhongguancun Institute of Artificial Intelligence 4DeepCybo 5Harbin Institute of Technology 6Huazhong University of Science and Technology ∗Equal contribution, †Corresponding author

Robotic generalization relies on physical intelligence: the ability to reason about state changes, contactrich interactions, and long-horizon planning under egocentric perception and action. Vision Language Models (VLMs) are essential to Vision–Language–Action (VLA) systems, but the reliance on thirdperson training data creates a viewpoint gap for humanoid robots. Collecting massive robot-centric data is an ideal but impractical solution due to cost and diversity constraints. Conversely, human egocentric videos offer a highly scalable data source with rich interaction context, yet the embodiment mismatch prevents the direct application. To bridge this gap, we propose an Egocentric2Embodiment Translation Pipeline that transforms raw human egocentric videos into multi-level, schema-driven embodiment supervision with enforced evidence grounding and temporal consistency, enabling the construction of the Egocentric2Embodiment dataset (E2E-3M) at scale. An egocentric-aware embodied brain, termed PhysBrain, is obtained by training on the E2E-3M dataset. PhysBrain exhibits substantially improved egocentric understanding, particularly for planning. It provides an egocentric-aware initialization that enables more sample-efficient VLA fine-tuning and higher success rates, demonstrating effective transfer from human egocentric supervision to downstream robot control.

[Figure 2]

[Figure 3]

Date: February 5, 2026 Code: https://zgc-embodyai.github.io/PhysBrain/ Correspondence: bocheng@hkust-gz.edu.cn, kaichen@zgci.ac.cn

1 Introduction

Vision-Language-Action (VLA) systems rely on a reliable embodied brain that integrates scenario understanding and action generation. Recent multimodal systems (Hurst et al., 2024a; Bai et al., 2025a) show rapid gains in visual perception, spatial and video reasoning, and long context understanding. These advances provide rich open vocabulary recognition and semantic inference capabilities that can be transferred to action prediction, thereby enabling modern VLAs (Zitkovich et al., 2023; Kim et al., 2024; Bjorck et al., 2025a; Black et al., 2024, 2025; Shen et al., 2025) to achieve strong performance across diverse manipulation tasks. These developments highlight that strong VLA performance is driven by an embodied brain that grounds executable planning and interaction decisions in the agent’s own perceptual stream.

For future humanoid robots, this perceptual stream is expected to be predominantly first-person, since perception, planning, and action feasibility are fundamentally grounded in the agent’s own body and workspace (Grauman et al., 2022). This places stringent demands on multimodal models operating under egocentric settings. However, empirical results on egocentric benchmarks (Lin et al., 2022; Pramanick et al., 2023; Chen et al., 2024; Patel et al., 2025) indicate that current multimodal models still struggle with longhorizon understanding, planning, and reliability under egocentric videos as shown in Fig.1. These deficits stem from challenges intrinsic to egocentric perception, including rapid viewpoint changes, frequent hand–object occlusions, the absence of the actor’s full body, and the need for cross-frame inference of contact and object state (He et al., 2025). Consequently, current performance bottlenecks are more likely due to insufficient

[Figure 4]

- Figure 1 Left: EgoThink radar plot comparing egocentric VLM performance across six dimensions. Right Top: The egocentric VLM performance on EgoThink and EgoPlan-B2 benchmarks with Qwen3VL-8B and PhysBrain, a Qwen3VL model fine-tuned with our annotated egocentric data (Sec. 3.1). Right Bottom: The egocentric VLA performance on SimplerEnv and RoboCasa benchmarks as the VLM backbone in a VLA fine-tuning pipeline.

egocentric embodied cognition, state tracking, and planning supervision, rather than limitations in model scale or single-frame recognition.

These limitations raise a fundamental scalability question: whether advancing VLA in egocentric settings necessarily depends on extensive robot data, including robot egocentric supervision. Acquiring large-scale and diverse robot manipulation data is widely acknowledged to be costly and difficult to scale, due to substantial hardware, labor, and safety constraints (Khazatsky et al., 2024). Even imitation learning relies on expensive human demonstrations, while existing large-scale robot data pipelines require long collection cycles or sustained multi-institution collaboration (Brohan et al., 2022; Zitkovich et al., 2023; O’Neill et al.,

- 2024). As a result, learning and aligning embodied brains primarily through such robot data fundamentally constrains the scalability and coverage of egocentric VLA systems.

In contrast to costly and hard-to-scale robot data, human first-person videos provide a naturally scalable source of egocentric supervision, covering diverse everyday behaviors and environments. This data modality offers observations closely aligned with real interaction distributions for learning embodied brains. Large-scale datasets, such as Ego4D (Grauman et al., 2022), BuildAI (BuildAI, 2025), and EgoDex (Hoque et al., 2025) demonstrate that egocentric videos can capture long-horizon activities, human–object interactions, and fine-grained manipulation dynamics at scale. An open question is how to leverage the latent planning structure and hand–object interaction regularities in human videos as supervision to strengthen egocentric embodied brains without robot data, thereby improving the sample efficiency and generalization of VLA systems.

Motivated by this observation, we develop a scalable annotation and instruction pipeline: Egocentric2Embodiment (E2E) Translation Pipeline that transforms human egocentric videos into structured, multi-level first-person VQA supervision for embodied brain learning. Each VQA instance encodes complementary information across multiple levels, including planning decompositions, key states, interaction constraints, and temporal relations, providing supervision beyond static visual recognition.

We instantiate the E2E translation pipeline at scale to construct E2E-3M from Ego4D, BuildAI, and EgoDex. The resulting supervision is used to train PhysBrain over the Qwen3-VL backbone. As shown in Fig.1, PhysBrain substantially outperforms the base model on the Egocentric VLM benchmarks, with the clearest improvements in egocentric perception and planning. When plugged into the egocentric VLA system as the embodied brain, PhysBrain continues to get excellent performance on the VLA benchmarks. Few-shot

adaptation on SimplerEnv and RoboCasa is effective and sample-efficient. The resulting performance exceeds VLA systems trained with large-scale robot data, while using no robot-data pretraining. Comprehensive evaluation results across VLM and VLA benchmarks validate E2E-3M dataset and the E2E Translation Pipeline, supporting human egocentric data as an effective source of physically grounded supervision for embodied brains. In summary, our contributions are as follows:

- • We introduce a scalable annotation and instruction pipeline, called Egocentric2Embodiment Translation Pipeline, which converts large-scale human egocentric videos into multi-level embodied supervision.
- • We provide a well-structured and validated egocentric VQA dataset E2E-3M that can effectively improve models’ first-person vision performance and generalization capability on VLA tasks.
- • Extensive experiments have demonstrated that human egocentric videos provide effective supervision for learning embodied brains in egocentric settings, leading to improved generalization in VLA tasks.

### 2 Related Work

- 2.1 First-Person Vision Language Model

Vison Language Models (VLMs) that excel on third-person content often degrade when the input shifts to egocentric imagery and video. Multiple lines of evidence point to a persistent viewpoint domain gap and to missing egocentric cues such as hand manipulation, egomotion, and partial observability (He et al., 2025). EgoVLP (Lin et al., 2022) is among the first to document that third-person pretraining transfers poorly and that explicitly egocentric objectives are needed for first-person retrieval, recognition, and temporal grounding. EgoVLPv2 (Pramanick et al., 2023) further reports that fusing first-person video and language during pretraining is important for egocentric tasks. Beyond these early works, EgoPlan-Bench (Chen et al.,

- 2024) shows that mainstream multimodal models struggle with egocentric planning even when the scenes are household and the instructions are simple, and it analyzes typical failure modes such as viewpoint confusion and missing contact reasoning. Studies on QaEgo4D (Barmann and Waibel, 2022) and QaEgo4Dv2 (Patel et al., 2025) find that both proprietary and open source VLMs lag on long-horizon egocentric reasoning. EgoM2P (Li et al., 2025a) also emphasizes the structural gap between third-person and first-person streams and argues for egocentric priors during pretraining.

2.2 Vision Language Action

Vision-Language-Action (VLA) models (Brohan et al., 2023; Zitkovich et al., 2023; Team et al., 2024; Black

- et al., 2024; Lian et al., 2026; Yu et al., 2026) represent a recent paradigm shift in robotic manipulation by unifying language understanding, visual perception, and motor control within a single end-to-end framework. Building upon large-scale vision-language models, VLAs directly map high-dimensional visual observations and natural language instructions to low-level robot actions, enabling intuitive human-robot interaction and task execution. Early works such as RT-1 (Brohan et al., 2023) and RT-2 (Zitkovich et al., 2023) demonstrate that scaling robot data and leveraging pretrained vision-language representations significantly improve manipulation performance across diverse tasks. Building upon these foundations, OpenVLA (Kim

- et al., 2024), π0 (Black et al., 2024; Pertsch et al., 2025; Black et al., 2025), and GR00T-N1 (Bjorck et al.,

2025a) further advance VLA capabilities through large-scale cross-embodiment and multi-task pretraining, demonstrating superior generalization and action prediction performance. Several works (Zhou et al., 2025; Yang et al., 2025d; Fang et al., 2025; Mazzaglia et al., 2025) attempt to address the catastrophic forgetting of language capabilities during VLA training, while others (Zawalski et al., 2025; Sun et al., 2024; Lin et al., 2025; Huang et al., 2025; Lee et al., 2025; Yuan et al., 2025) explore incorporating chain-of-thought reasoning into the VLA inference process. To pursue better generalization, several works (Shen et al., 2025; Liang

- et al., 2025; Jia et al., 2025) attempt to incorporate video generation models or world models into VLA action prediction, while others (Li et al., 2025b; Yu et al., 2025; Chen et al., 2025a,b) explore applying reinforcement learning to train VLA models.

- 2.3 Learning VLAs from Human Demonstration

Robot data acquisition is hard to scale due to the stringent robot–operator configuration and reliance on expert tele-operation. Egocentric VLA trained on the egocentric human demonstrations offers a more scalable path, with strong potential to advance perception–action learning and real-world executability. EgoVLA (Yang et al., 2025c) utilizes scaled egocentric videos plus a unified human–robot action space with light robot finetuning, enabling efficient skill transfer. Being-H0 (Luo et al., 2025) leverages physical-instruction tuning with discrete hand-motion codes and a physics-aligned cross-view space supports fine-grained VLA training from human videos. H-RDT (Bi et al., 2025) sets large bimanual pretraining with 3D hand pose and a two-stage diffusion policy delivers substantial improvements. GR-3 (Cheang et al., 2025) utilizes multi-source training yields strong generalization, rapid few-shot adaptation, and robust long-horizon bimanual and mobile control. RynnVLA-001 (Jiang et al., 2025) pretrains on large-scale human video demonstrations with video generation objectives and compresses actions into a continuous latent space via ActionVAE to align video prediction with downstream robot fine-tuning. VITRA (Li et al., 2025c) treats the human hand as a proxy end-effector, converts the egocentric hand videos into robot-aligned formats, and combines VLMs with diffusion-based action experts for policy learning.

However, these approaches focus on the explicit alignment of human data to robot action spaces, which is inherently constrained by embodiment gaps. In contrast, our work targets a more upstream objective by transforming egocentric human data into embodiment supervision signals for embodied brains, providing a scalable foundation that complements robot-data-based pipelines.

- 3 Egocentric Embodied Supervision In this section, we introduce the egocentric data annotation pipeline and the E2E-3M dataset.

- 3.1 Egocentric2Embodiment Translation Pipeline

[Figure 5]

- Figure 2 Illustration of the Egocentric2Embodiment Translation Pipeline.

Human egocentric videos encode rich embodied experience, including action progression, hand–object interaction, and task-level structure. However, this experience is not directly usable for training embodied brains. Raw videos lack explicit structure, free-form language annotations are unstable, and unconstrained generation often introduces temporal ambiguity or hallucinated interactions.

Our key idea is to translate human egocentric data into structured and verifiable supervision that captures the hierarchical structure of embodied behavior, spanning action semantics, temporal organization, interaction dynamics, and task-level reasoning. To this end, we design a schema-driven, rule-validated egocentric VQA data engine as shown in Fig.2 that systematically converts raw egocentric human videos into multi-level supervision aligned with embodied planning and interaction reasoning.

- 3.1.1 Data Intake and Pre-processing

##### To define the basic supervision units, the engine chunks each episode into short temporal clips at stage 1. Given the large variation in egocentric action amplitude and frequency across scenarios, we adopt scenario-

aware temporal segmentation, including fixed-interval, event-driven(Build-AI, 2025), and kinematic-aware strategies(Li et al., 2025c). All clips are associated with explicit temporal spans and exposed through a unified interface for downstream annotation. Episode-level metadata is used as contextual conditioning to limit the semantic space of subsequent question answering.

The resulting representations are temporally localized and preserve short-range state transitions relevant to embodied manipulation and interaction. It forms the atomic unit for the downstream annotation stage.

- 3.1.2 Annotation Scheme

In stage 2, we define a finite, schema-driven annotation space to produce supervision that reflects embodied cognition rather than generic video description. Each clip is labeled with one of seven complementary VQA modes, including temporal, spatial, attribute, mechanics, reasoning, summary, and trajectory. Each mode is paired with a template set that standardizes wording and controls the information granularity. The engine samples a mode and a template, then generates a customized and detailed QA pair for each clip.

VQA generation is performed by a set of VLM annotation engines. The schema constrains both the question form and the required semantic content, which keeps supervision targets consistent across different generators. Answers must be natural-language and grounded in the visual evidence. The engine enforces egocentric conventions such as left/right hand references and manipulation-specific phrasing. This stage yields multi-level annotations that capture complementary aspects of planning and interaction reasoning.

- 3.1.3 Quality Assurance and Validation Logic

Open-ended generation easily produces errors that are harmful for training supervision. Common failures include references to non-visible hands and incorrect temporal ordering, et al. We therefore introduce the third stage that a rule checker is designed as a validation gate. Samples that fail validation are rejected and sent back for regeneration with a structured error that indicates the violated constraint.

The checker applies three types of constraints designed to filter visual hallucinations. Evidence grounding restricts the generation scope, requiring that mentioned entities are grounded in the episode-level object metadata. Egocentric consistency enforces the correct hand references and prohibits mentions of unseen limbs or contradictory assignments. Mode-specific temporal logic requires explicit temporal connectors and verifies timeline alignment. The generation–validation loop repeats until all constraints are satisfied. To further guarantee the visual fidelity of the resulting dataset, we conducted a rigorous human audit on a random subset, confirming that our logic-verified filtering effectively retains high-quality, hallucination-free supervision.

- 3.1.4 Structured Egocentric Corpus

Samples that satisfy all validation constraints are retained and compiled into the egocentric VQA supervision dataset. Each entry records the sampled frames, the selected VQA mode, the generated QA pair, and the validation outcome. This design ensures traceability and reproducibility. The supervision produced by the proposed data engine offers structured and logic-verified supervision that encodes action organization and hand–object interaction, completing the translation of egocentric video data into reliable training signals for egocentric planning and interaction reasoning.

- 3.2 Egocentric2Embodiment Dataset (E2E-3M)

- 3.2.1 Data Sources and Domain Coverage

The proposed Egocentric2Embodiment Translation Pipeline is applied to generate the E2E-3M, a large-scale human egocentric video corpora collected across three complementary domains: household, factory, and laboratory environments. These corpora captures substantial variation in environmental context, object composition, and interaction patterns, as shown in Fig.5 and Fig.6.

Specifically, Ego4D represents open-world household activities and provides extensive geographic and contextual diversity. BuildAI captures real industrial workflows, emphasizing procedural regularity and dense hand visibility in factory environments. EgoDex focuses on laboratory settings and offers high-resolution egocentric

[Figure 6]

- Figure 3 Data Distribution Statistics of E2E-3M dataset.

##### manipulation sequences with fine-grained interaction cues. These sources differ systematically in spatial layout, object distribution, and task structure. The aggregation yields the E2E-3M dataset with complementary coverage across the space of egocentric embodied experience.

- 3.2.2 Diversity Analysis

To evaluate whether the dataset provides sufficiently rich supervision for generalized embodied manipulation, we analyze diversity along two interpretable axes: object coverage and action (verb) coverage, as quantified in Eq. 1 and Eq. 2.

Object Coverage and Environmental Spectrum. Object coverage reflects the breadth of perceptual contexts. We calculate the normalized noun lemma count per domain as:

ObjectDiv(s) = |Vsnoun| Tsnoun × 1000, (1)

where |Vsnoun| is the number of unique noun lemmas and Tsnoun is the total noun token count in domain s. The distribution shown in Fig.3(b) validates that our aggregated dataset spans the complete spectrum from structured to unstructured environments. Specifically, Household data exhibits high object diversity, capturing the long-tail distribution of objects typical in unstructured, open-world scenarios. Factory data shows lower ObjectDiv scores, which accurately reflect the standardized, repetitive nature of industrial workflows where agents interact with a fixed set of tools and parts. Lab settings fall in between, offering a semi-structured middle ground. This distributional shift is not a limitation but a critical feature: it ensures that VLA models are trained on both the rigid procedural constraints of industry and the chaotic variability of daily life, fostering robust generalization.

Action Coverage and Interaction Semantics. Action coverage quantifies the richness of manipulation semantics. We calculate the verb diversity as:

VerbDiv(m) = |Vmverb|

Nm × 1000, (2) where normalization by the number of QA pairs Nm accounts for interaction density. As shown in Fig. 3, action-centric modes (e.g., Reasoning, Mechanics, Temporal) consistently maintain high lexical diversity across all domains. Notably, even in the Factory domain where object diversity is lower due to standardization, the Mechanics and Reasoning scores remain high. This decoupling indicates that while the industrial environment is visually structured, the underlying manipulation tasks involve complex, non-trivial procedural logic. This confirms that our dataset provides dense supervision for manipulation behaviors regardless of the environmental structure.

The E2E-3M dataset bridges human egocentric video and embodied brain learning by providing structured supervision with broad scene coverage and rich action diversity. We expect that releasing this dataset will support future research on egocentric VLA and physical intelligence.

### 4 Methodology

Using the data annotation pipeline proposed in the previous section, we translate embodied experience from egocentric videos into structured supervision suitable for learning an embodied brain. To preserve general-purpose vision–language capability during SFT, we additionally mix an equal-sized subset sampled from FineVision, a large-scale curated vision–language corpus. We then perform supervised fine-tuning (SFT) on base VLMs (e.g., Qwen3-VL-4B and Qwen3-VL-8B) using this mixture, resulting in an egocentric-centered VLM backbone (PhysBrain) with improved first-person understanding, reasoning, and planning capabilities. Quantitative results are reported in Table 1.

PhysVLA

#### ···

Action Expert

···

[Figure 7]

PhysBrain

Image

Encode Tokenize

Text

Noise

|[Figure 8]|
|---|

With PhysBrain in hand, we study how these egocentric gains transfer to downstream control under standard VLA instantiations. Our goal in this section is not to propose a new VLA architecture, but to evaluate transferability while minimizing confounding factors from additional heuristics or hand-crafted priors. We follow the widely adopted community paradigm, GR00T-style, and keep the

Pick Up the Apple to the Plate

Action

- Figure 4 VLA architecture built on PhysBrain. PhysVLA conditions a flow-matching diffusion action expert on the last-layer hidden states of PhysBrain.

action expert lightweight and consistent. We denote an observation (the egocentric image sequence) as ot, the language instruction as x, and the VLM parameters as ϕ. The VLM produces token-level hidden states:

Hℓt = VLMϕ(ot,x)[ℓ] ∈ RN×d, ℓ = 1,...,L, (3)

where L is the number of layers in the VLM, N is the token length, and d is the hidden dimension. The action policy predicts a future action chunk at:t+K ∈ RK×d

a.

PhysVLA. We introduce PhysVLA, which follows the dual-system design in GR00T N1.5 (Bjorck et al., 2025a): the VLM plays the role of System 2 to produce high-level multimodal representations, while a Flow-Matching (FM) action expert (Liu, 2022) serves as System 1 to generate continuous actions. Concretely, the last-layer VLM hidden states Zt = HLt are utilized as the conditioning signal.

The FM expert is implemented as a diffusion transformer (DiT) (Peebles and Xie, 2023) that denoises an action trajectory by cross-attending to Zt (VLM features are keys/values, action tokens are queries). Under the rectified-flow parameterization, we sample Gaussian noise ϵ ∼ N(0,I) and a time scalar τ ∈ (0,1], then linearly interpolate between noise and the target action chunk to obtain the noised trajectory a˜:

a˜ = (1 − τ)ϵ + τ a, v = a − ϵ. (4)

Here v is the target (time-independent) velocity that transports the noise trajectory to the data trajectory under this parameterization. The action expert predicts this velocity field conditioned on VLM features (and optional proprioceptive state st):

vˆ = fθ(a˜,τ; Zt,st), (5) and is trained with a simple regression objective

LFM = E ∥vˆ − v∥22 . (6)

At inference, we start from noise and apply a small number of FM denoising steps (we use steps = 8) to obtain the action chunk at:t+K with K=16. This design provides a controlled setting to examine how informative the egocentric VLM representation Zt is for action prediction.

###### Table 1 Comparison on EgoPlan and EgoThink benchmarks. The detailed sub-tasks belong exclusively to the EgoThink benchmark, with the overall average reported in the final column.

EgoPlan-B1 EgoPlan-B2 EgoThink Benchmark

Method

###### Acc. Acc. Act. Fore. Loc. Obj. Asst. Nav. Reas. Avg.

General VLM

GPT-4o (Hurst et al., 2024a) 39.5 41.0 73.0 66.0 89.0 78.6 28.0 12.0 66.0 66.4 MiniGPT-4-7B (Zhu et al., 2023) 28.1 24.5 45.5 36.5 61.5 48.0 30.0 12.0 36.7 41.6 LLaVA-1.5-7B (Liu et al., 2024) 27.8 25.4 35.0 43.5 76.0 65.3 33.0 26.0 53.0 51.6 LLaMA-3.2-11B (Dubey et al., 2024) 24.3 25.1 34.0 49.5 57.5 62.7 42.0 22.0 47.7 48.4 Qwen-3-VL-4B (Yang et al., 2025a) 42.2 34.6 63.5 65.0 82.5 72.6 46.0 35.0 71.0 66.7 Qwen-3-VL-8B (Yang et al., 2025a) 44.3 40.5 68.0 66.5 86.0 72.3 41.0 39.0 61.7 65.9

Embodied Brain

VST-RL-7B (Yang et al., 2025b) 40.8 28.7 55.0 56.5 69.5 67.3 15.0 22.0 62.3 56.2 RoboBrain2.0-7B (Team et al., 2025a) 38.6 23.3 35.0 47.0 77.5 60.7 44.0 38.0 52.3 52.8 RoboBrain2.5-8B (Tan et al., 2026) 45.9 45.2 57.5 56.5 81.0 70.3 40.0 28.0 68.3 62.4

PhysBrain-4B (ours) 43.9 39.3 68.0 64.5 85.5 76.3 66.0 44.0 66.0 69.4 PhysBrain-8B (ours) 47.4 46.9 69.0 69.0 86.5 76.0 65.0 42.0 64.0 69.7

### 5 Experiment

This section details the experimental setup, benchmarks, and results. We report results from VLM and VLA evaluations.

- 5.1 VLM Egocentric Evaluation

- 5.1.1 Egocentric Understanding Evaluation

For a comprehensive egocentric evaluation, we conduct experiments on three widely-used benchmarks: EgoPlan-Benchmark1(Chen et al., 2024), EgoPlan-Benchmark2(Qiu et al., 2024), and EgoThink (Cheng et al., 2024). These benchmarks cover diverse real-world settings and complementary protocols: EgoPlan utilizes multiple-choice evaluation, whereas EgoThink scores free-form generations via an LLM judge.

- EgoPlan-Benchmark1: EgoPlan-Benchmark1(Chen et al., 2024) evaluates the planning capabilities of Vision Language Models in real-world scenarios from an egocentric perspective, simulating human perception. This benchmark includes realistic tasks involving a diverse range of action plans and intricate visual observations. We validate the VLM models using 3,314 VQA pairs from the EgoPlan-Val set, provides a comprehensive assessment of a model’s ability to understand task progress, track current states, and predict the next feasible action. EgoPlan-Benchmark1 presents significant challenges for current VLMs, highlighting substantial room for improvement in human-level task planning.
- EgoPlan-Benchmark2: EgoPlan-Benchmark2(Qiu et al., 2024) extends the scope of its predecessor by incorporating a broader variety of real-world scenarios across four key domains: Work, Daily Life, Hobbies, and Recreation, with 24 distinct scenarios. This expanded benchmark includes 1,321 high-quality multiple-choice questions sourced from 1,113 egocentric videos, each representing a real-world task. The dataset emphasizes task planning, requiring models to predict the next action based on task progress, current visual observations, and task goals. EgoPlan-Benchmark2 provides a rigorous evaluation of VLMs’ planning capabilities and highlights the challenges these models face in real-world decision-making, with detailed analysis and instructions to enhance planning performance.

EgoThink: EgoThink(Cheng et al., 2024) evaluates the first-person perspective capabilities of vision-language models (VLMs) across six core areas: Object, Activity, Localization, Reasoning, Forecasting, and Planning. It contains 700 images from 595 egocentric videos annotated with question-answer pairs. GPT-4 is used as an automatic evaluator to grade open-ended responses. The EgoThink benchmark provides a comprehensive assessment of VLMs’ reasoning and planning abilities in real-world scenarios from a first-person viewpoint, revealing key challenges and offering insights for advancing embodied AI.

Baselines. We primarily compare our method against two categories of baselines: (i) GeneralVLM, which include

- Table 2 Results of evaluating the VLA models with the WidowX robot in the SimplerEnv simulation environment. We highlight the best results in bold and the second-best results with underline.

Put Spoon on Towel

Put Carrot on Plate

Stack Green Block on Yellow Block

Put Eggplant in Yellow Basket

Method

Average

VLA Baselines

RT-1-X (O’Neill et al., 2024) 0.0 4.2 0.0 0.0 1.1 Octo-Base (Team et al., 2024) 15.8 12.5 0.0 41.7 17.5 Octo-Small (Team et al., 2024) 41.7 8.2 0.0 56.7 26.7 OpenVLA (Kim et al., 2024) 4.2 0.0 0.0 12.5 4.2 OpenVLA-OFT (Kim et al., 2025) 12.5 4.2 4.2 72.5 23.4 RoboVLM (Li et al., 2024b) 50.0 37.5 0.0 83.3 42.7 TraceVLA (Zheng et al., 2025) 12.5 16.6 16.6 65.0 27.7 SpatialVLA (Qu et al., 2025) 20.8 20.8 25.0 70.8 34.4 CogACT (Li et al., 2024a) 71.7 50.8 15.0 67.5 51.3 VideoVLA (Shen et al., 2025) 75.0 20.8 45.8 70.8 53.1 π0 (Black et al., 2024) 29.1 0.0 16.6 62.5 27.1 π0.5 (Black et al., 2025) 49.3 64.7 44.7 69.7 57.1 Isaac-GR00T-N1.6-Bridge (Team et al., 2025b) 64.5 65.5 5.5 93.0 57.1

VLM Baselines

Qwen2.5-VL-7B (Bai et al., 2025b) 68.7 35.4 25.0 75.0 51.0 Qwen3-VL-4B (Yang et al., 2025a) 87.5 50.0 29.2 54.2 55.2 Qwen3-VL-8B (Yang et al., 2025a) 68.7 38.5 30.2 87.9 56.3 VST-RL-7B (Yang et al., 2025b) 57.7 41.7 16.7 50.0 41.3 RoboBrain2.0-7B (Team et al., 2025a) 30.8 24.7 2.5 93.3 37.8 RoboBrain2.5-8B (Tan et al., 2026) 75.0 55.5 40.1 100.0 67.6

PhysBrain-4B (ours) 90.3 58.3 34.7 80.6 65.9 PhysBrain-8B (ours) 77.8 62.5 34.8 94.8 67.4

closed-source models such as GPT-4o and widely-used open-source models (MiniGPT-4-7B, LLaVA-1.5-7B, LLaMA-3.2-11B, Qwen3-VL-4B, and Qwen3-VL-8B); and (ii) EmbodiedBrain, which include VST-RL-7B (Yang

- et al., 2025b), RoboBrain2-7B (Team et al., 2025a) and RoboBrain2.5-8B (Tan et al., 2026).

Evaluation. The comparison methods are evaluated through the released weight for direct inference. Evaluation conditions are standardized across models, and all models use the same prompt template as shown in Table 6. These controls ensure that performance differences reflect model capability rather than prompt variation or inconsistent scoring.

In the Egothink benchmark, the generation outputs are scored with a single GPT-4 (Hurst et al., 2024b) judging protocol across all EgoThink subtasks. Table 1 summarizes performance on the six EgoThink dimensions (Activity, Forecast, Localization, Object, Planning, Reasoning). PhysBrain-8B achieves the highest average performance, while our PhysBrain-4B achieves sub-optimal performance and consistently outperforms strong open and competitive baselines. The most pronounced improvement is observed on Planning, where PhysBrain substantially exceeds all baselines, indicating a clear advantage in translating egocentric observations into executable plans.

In the EgoPlan benchmark, we report accuracy on both EgoPlan-Benchmark1 and EgoPlan-Benchmark2 (Table 1). Trained with our E2E-3M supervision, PhysBrain consistently improves over the Qwen3-VL backbone: PhysBrain-8B reaches 47.4/46.9 on B1/B2, outperforming Qwen3-VL-8B (44.3/40.5) by +3.1/+6.4 points; PhysBrain-4B achieves 43.9/39.3, improving over Qwen3-VL-4B (42.2/34.6) by +1.7/+4.7. These gains support that E2E-3M injects physically grounded interaction priors into the base model, leading to stronger egocentric planning.

- 5.2 VLA Simulation Evaluation

To validate the efficacy of our model when deployed as the VLA for robotic control, we adopt PhysBrain as the VLM backbone and fine-tune it within the VLA paradigm using downstream robotics data. We then evaluate on the SimplerEnv (Li et al., 2024c) and RoboCasa (Nasiriany et al., 2024) simulation benchmarks.

###### Table 3 Results of evaluating the VLA models with the GR1 robot in the RoboCasa Tabletop simulation environment. The results for QwenGR00T, QwenOFT, and QwenFAST are derived from the official StarVLA experiments (starVLA Community, 2025). We highlight the best results in bold and the second-best results with underline.

Task Isaac-GR00TN1.6 QwenGR00T+Qwen3VL +QwenOFTQwen3VL +QwenFASTQwen3VL PhysBrain-4B PhysBrain-8B

PnP Bottle To Cabinet Close 51.5 46.0 30.0 38.0 74.0 70.0 PnP Can To Drawer Close 13.0 80.0 76.0 44.0 68.0 74.0 PnP Cup To Drawer Close 8.5 54.0 44.0 56.0 42.0 46.0 PnP Milk To Microwave Close 14.0 48.0 44.0 44.0 54.0 60.0 PnP Potato To Microwave Close 41.5 28.0 32.0 14.0 24.0 34.0 PnP Wine To Cabinet Close 16.5 46.0 36.0 14.0 54.0 40.0 PnP Novel From Cuttingboard To Basket 58.0 48.0 50.0 54.0 62.0 54.0 PnP Novel From Cuttingboard To Cardboardbox 46.5 40.0 40.0 42.0 44.0 56.0 PnP Novel From Cuttingboard To Pan 68.5 68.0 70.0 58.0 56.0 72.0 PnP Novel From Cuttingboard To Pot 65.0 52.0 54.0 58.0 58.0 74.0 PnP Novel From Cuttingboard To Tieredbasket 46.5 56.0 38.0 40.0 40.0 44.0 PnP Novel From Placemat To Basket 58.5 42.0 32.0 36.0 42.0 58.0 PnP Novel From Placemat To Bowl 57.5 44.0 58.0 38.0 56.0 56.0 PnP Novel From Placemat To Plate 63.0 48.0 52.0 42.0 80.0 62.0 PnP Novel From Placemat To Tieredshelf 28.5 18.0 24.0 18.0 14.0 28.0 PnP Novel From Plate To Bowl 57.0 60.0 60.0 52.0 54.0 70.0 PnP Novel From Plate To Cardboardbox 43.5 50.0 50.0 30.0 50.0 54.0 PnP Novel From Plate To Pan 51.0 54.0 66.0 48.0 68.0 56.0 PnP Novel From Plate To Plate 78.7 70.0 68.0 50.0 78.0 60.0 PnP Novel From Tray To Cardboardbox 51.5 38.0 44.0 28.0 40.0 52.0 PnP Novel From Tray To Plate 71.0 56.0 56.0 34.0 66.0 60.0 PnP Novel From Tray To Pot 64.5 50.0 62.0 46.0 52.0 70.0 PnP Novel From Tray To Tieredbasket 57.0 36.0 54.0 36.0 50.0 48.0 PnP Novel From Tray To Tieredshelf 31.5 16.0 30.0 16.0 22.0 28.0

Average 47.6 47.8 48.8 39.0 49.75 55.25

SimplerEnv: The WidowX Robot is utilized in the SIMPLER environment to evaluate the VLA policy on four manipulation tasks: "Put Spoon on Towel," "Put Carrot on Plate," "Stack Green Block on Yellow Block," and "Put Eggplant in Yellow Basket." For each task, we conduct 24 trials and using the official evaluation script from the SimplerEnv repository. This ensures a consistent and rigorous assessment of the robot’s manipulation performance across these tasks.

RoboCasa: RoboCasa GR1 Tabletop Manipulation Benchmark includes 24 diverse tasks involving complex interactions with articulated objects and varied geometries. Examples of tasks include "PnPBottleToCabinetClose" and "PnPCanToDrawerClose", as well as scenarios with appliances like microwaves and toasters. For training, we use the Humanoid Robot Tabletop Manipulation subset from the PhysicalAI-Robotics-GR00TX-Embodiment-Sim dataset (Bjorck et al., 2025b), ensuring a robust and diverse training environment for evaluating robotic manipulation performance.

- 5.2.1 Experiment Settings

Training. To adapt the VLM to the VLA architecture and the target robotic platform, we follow the training configuration of the starVLA (starVLA Community, 2025) framework. Detailed training hyperparameters and configurations are provided in Appendix C.

Baselines. We primarily compare our method against two categories of baselines: (i) VLA baselines, which include several widely used VLA models (RT-1-X, Octo, OpenVLA, RoboVLM, TraceVLA, SpatialVLA,

CogACT, VideoVLA, π0, π0.5, and Issac-GR00T-N1.6-Bridge); and (ii) VLM baselines, where we fine-tune several commonly used VLMs (Qwen2.5-VL, Qwen3-VL, RoboBrain2.0, VST-RL, and RoboBrain2.5) under the PhysVLA paradigm and evaluate by using the same training configuration as ours.

Evaluation. We evaluate our policy on SimplerEnv (Li et al., 2024c) (four WidowX tasks) and RoboCasa (24 complex manipulation tasks involving articulated objects). To ensure statistical robustness, we adhere to official protocols, reporting mean success rates (Avg@50) averaged over 5 and 50 independent trials for SimplerEnv and RoboCasa, respectively.

5.2.2 Experiment Results

Table 2 summarizes the SimplerEnv evaluation results, comparing our PhysBrain with all baseline methods. The evaluation results of RoboCasa are presented in Table 3.

As detailed in Table 2, PhysBrain-8B achieves an average success rate of 67.4%, significantly surpassing VLA baselines trained on substantially larger robot datasets (e.g., Isaac-GR00T at 57.1%) and performing comparably to the state-of-the-art RoboBrain2.5 (67.6%). Crucially, while RoboBrain2.5 relies on massive cross-embodiment robot data for representation alignment, PhysBrain attains this proficiency solely through pretraining on our proposed E2E-3M dataset. This highlights the potential of human data by providing robust physical priors that transfer seamlessly to robotic manipulation.

We evaluate PhysBrain on the RoboCasa benchmark against the Qwen3-VL-4B backbone integrated with three distinct action encodings (GR00T, OFT, and FAST), as shown in Table 3. Since PhysBrain adopts the Gr00T architecture, the comparison with the standard QwenGR00T baseline is particularly critical, as it directly isolates the performance gains attributable to our proposed method. As shown in Table 3, PhysBrain-4B achieves a 49.75% success rate, not only surpassing its direct counterpart (47.8%) but also outperforming other encoding variants. Furthermore, the PhysBrain-8B model validates significant scalability, attaining the best average success rate of 55.25%.

- 5.3 Ablation Study

- Table 4 Ablation experiment on the impact of the proposed E2E-3M dataset across VLM Architectures and Model Scales. Best results are in bold.

Method VLM Bench VLA Bench

Qwen2.5-VL-7B 58.7 34.4

PhysBrain2.5-7B (ours) 63.1 53.9

Qwen3-VL-4B 66.7 55.2

PhysBrain-4B (ours) 69.4 65.9

Qwen3-VL-8B 65.9 56.3

PhysBrain-8B (ours) 69.7 67.4

Dataset Impact Across VLM Architectures and Model Scales. To verify the universality of the proposed E2E-3M dataset, we conducted extensive ablation studies across different VLM architectures (Qwen2.5-VL and Qwen3-VL) and varying parameter scales (4B, 7B, and 8B). As detailed in Table 4, fine-tuning on our dataset consistently yields significant performance improvements over the base models across all configurations. Notably, while the VLM benchmark scores show steady increments, the improvements on the VLA benchmark are particularly substantial (e.g., a 19.5% gain for the 7B model). This disparity indicates that while standard VLMs possess strong general vision-language

capabilities, they lack the specific physical priors required for robotic manipulation. These results confirm that the physical intelligence distilled from E2E-3M is agnostic to the underlying architecture, consistently enhancing the embodied planning capabilities of diverse foundation models.

- Table5 Ablation experiment on the impact of the egocentric embodied supervision scale. Best results are in bold.

Method VLM Bench VLA Bench

##### PhysBrain-8B-wo-Ego4D 67.8 64.1

PhysBrain-8B (ours) 69.7 67.4

Impact of Egocentric Supervision Data Scale. We further investigate the scaling behavior of our approach by analyzing the impact of dataset size on model performance. Table 5 presents an ablation study where we remove the large-scale Ego4D subset from the training data. The results demonstrate a clear positive correlation between the volume of egocentric supervision and the resulting model pro-

ficiency. The full PhysBrain-8B model outperforms the version trained without Ego4D on both VLM and VLA benchmarks. This decline in performance upon data reduction highlights that the scale and diversity of the dataset are critical factors. It suggests that scaling up high-quality egocentric human data is a viable and effective pathway for continuously improving the physical understanding and manipulation skills of embodied agents.

- 6 Conclusion

In this work, we leverage human egocentric videos to connect vision-language models with physical intelligence for robotic generalization. We introduce the Egocentric2Embodiment pipeline, which converts raw egocentric videos into multi-level VQA supervision, creating the E2E-3M dataset with nearly three million verified instances across household, factory, and laboratory settings. By fine-tuning a vision-language model on the E2E-3M dataset, we develop PhysBrain that significantly enhances egocentric capabilities. It also achieves high success rates on VLA system when used as the VLM backbone in standard VLA fine-tuning. Our results demonstrate that scalable human egocentric supervision bridges vision-language understanding and physical intelligence, enabling greater data diversity and improved policy learning from human demonstrations.

References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Leonard Barmann and Alex Waibel. Where did i leave my keys?—episodic-memory-based question answering on egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 1559–1567, 2022.

Hongzhe Bi, Lingxuan Wu, Tianwei Lin, Hengkai Tan, Zhizhong Su, Hang Su, and Jun Zhu. H-rdt: Human manipulation enhanced bimanual robotic manipulation, 2025.

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T-N1: An open foundation model for generalist humanoid robots. arXiv

- preprint arXiv:2503.14734, 2025a.

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv

- preprint arXiv:2503.14734, 2025b.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. π0.5: a vision-language-action model with open-world generalization. In Annual Conference on Robot Learning (CoRL), 2025.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023.

Build-AI. Egocentric-10k. Hugging Face Datasets, 2025. BuildAI. Egocentric-10k, 2025. https://huggingface.co/datasets/builddotai/Egocentric-10K. Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu,

Xiao Ma, Hao Niu, Wenxuan Ou, Wanli Peng, Zeyu Ren, Haixin Shi, Jiawen Tian, Hongtao Wu, Xin Xiao, Yuyang Xiao, Jiafeng Xu, and Yichu Yang. Gr-3 technical report, 2025.

Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Xiang Li, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, Tiejun Huang, Yu Wang, and Chao Yu. πrl: Online rl fine-tuning for flow-based vision-language-action models, 2025a.

Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu. EgoPlanBench: Benchmarking multimodal large language models for human-level planning. arXiv preprint arXiv:2312.06722, 2024.

Zengjue Chen, Runliang Niu, He Kong, Qi Wang, Qianli Xing, and Zipei Fan. Tgrpo: Fine-tuning vision-language-action model via trajectory-wise group relative policy optimization, 2025b.

Sijie Cheng, Zhicheng Guo, Jingwen Wu, Kechen Fang, Peng Li, Huaping Liu, and Yang Liu. Egothink: Evaluating first-person perspective thinking capability of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14291–14302, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407,

- 2024.

Zhen Fang, Zhuoyang Liu, Jiaming Liu, Hao Chen, Yu Zeng, Shiting Huang, Zehui Chen, Lin Chen, Shanghang Zhang, and Feng Zhao. Dualvla: Building a generalizable embodied agent via partial decoupling of reasoning and action,

- 2025.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18995–19012, 2022.

Yuping He, Yifei Huang, Guo Chen, Lidong Lu, Baoqi Pei, Jilan Xu, Tong Lu, and Yoichi Sato. Bridging perspectives: A survey on cross-view collaborative intelligence with egocentric-exocentric vision. arXiv preprint arXiv:2506.06253, 2025.

Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Vision-languageaction reasoning via reinforced visual latent planning, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila

- Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024a.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila

- Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024b.

Yueru Jia, Jiaming Liu, Shengbang Liu, Rui Zhou, Wanhe Yu, Yuyang Yan, Xiaowei Chi, Yandong Guo, Boxin Shi, and Shanghang Zhang. Video2act: A dual-system video diffusion policy with robotic spatio-motional modeling, 2025.

Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, Kexiang Wang, Mingxiu Chen, et al. Rynnvla-001: Using human demonstrations to improve robot manipulation. arXiv preprint arXiv:2509.15212, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, et al. OpenVLA: An open-source vision-language-action model. In Annual Conference on Robot Learning (CoRL), 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025.

Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space, 2025.

Gen Li, Yutong Chen, Yiqian Wu, Kaifeng Zhao, Marc Pollefeys, and Siyu Tang. EgoM2P: Egocentric multimodal multitask pretraining. arXiv preprint arXiv:2506.07886, 2025a.

Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via reinforcement learning, 2025b.

Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, Xiaofan Wang, Bei Liu, Jianlong Fu, Jianmin Bao, Dong Chen, Yuanchun Shi, Jiaolong Yang, and Baining Guo. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation, 2024a.

Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang, and Baining Guo. Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos, 2025c.

Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models, 2024b.

Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. In Annual Conference on Robot Learning (CoRL), 2024c.

Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, and Kai Chen. Langforce: Bayesian decomposition of vision language action models via latent action queries, 2026. https://arxiv.org/abs/2601.15197.

Haotian Liang, Xinyi Chen, Bin Wang, Mingkang Chen, Yitian Liu, Yuhao Zhang, Zanxin Chen, Tianshuo Yang, Yilun Chen, Jiangmiao Pang, Dong Liu, Xiaokang Yang, Yao Mu, Wenqi Shao, and Ping Luo. Mm-act: Learn from multimodal parallel generation to act, 2025.

Fanqi Lin, Ruiqian Nai, Yingdong Hu, Jiacheng You, Junming Zhao, and Yang Gao. Onetwovla: A unified visionlanguage-action model with adaptive reasoning, 2025.

Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z. XU, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, Chengfei Cai, WANG HongFa, Dima Damen, Bernard Ghanem, Wei Liu, and Mike Zheng Shou. Egocentric video-language pretraining. In Advances in neural information processing systems (NeurIPS), volume 35, pages 7575–7586, 2022.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024.

Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.

Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: Vision-language-action pretraining from large-scale human videos, 2025.

Pietro Mazzaglia, Cansu Sancaktar, Markus Peschl, and Daniel Dijkman. Hybrid training for vision-language-action models, 2025.

Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. RoboCasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems, 2024.

Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

Alkesh Patel, Vibhav Chitalia, and Yinfei Yang. Advancing egocentric video question answering with multimodal large language models. arXiv preprint arXiv:2504.04550, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models, 2025.

Shraman Pramanick, Yale Song, Sayan Nag, Kevin Qinghong Lin, Hardik Shah, Mike Zheng Shou, Rama Chellappa, and Pengchuan Zhang. EgoVLPv2: Egocentric video-language pre-training with fusion in the backbone. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5285–5297, 2023.

Lu Qiu, Yi Chen, Yuying Ge, Yixiao Ge, Ying Shan, and Xihui Liu. Egoplan-bench2: A benchmark for multimodal large language model planning in real-world scenarios. arXiv preprint arXiv:2412.04447, 2024.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’20. IEEE Press, 2020. ISBN 9781728199986.

Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators, 2025.

starVLA Community. Starvla: A lego-like codebase for vision-language-action model developing, 2025. https: //github.com/starVLA/starVLA.

Qi Sun, Pengfei Hong, Tej Deep Pala, Vernon Toh, U-Xuan Tan, Deepanway Ghosal, and Soujanya Poria. Emma-x: An embodied multimodal action model with grounded chain of thought and look-ahead spatial reasoning, 2024.

Huajie Tan, Enshen Zhou, Zhiyu Li, Yijie Xu, Yuheng Ji, Xiansheng Chen, Cheng Chi, Pengwei Wang, Huizhu Jia, Yulong Ao, Mingyu Cao, Sixiang Chen, Zhe Li, Mengzhen Liu, Zixiao Wang, Shanyu Rong, Yaoxu Lyu, Zhongxia Zhao, Peterson Co, Yibo Li, Yi Han, Shaoxuan Xie, Guocai Yao, Songjing Wang, Leiduo Zhang, Xi Yang, Yance Jiao, Donghai Shi, Kunchang Xie, Shaokai Nie, Chunlei Men, Yonghua Lin, Zhongyuan Wang, Tiejun Huang, and Shanghang Zhang. Robobrain 2.5: Depth in sight, time in mind. arXiv preprint arXiv:2601.14352, 2026.

BAAI RoboBrain Team, Mingyu Cao, Huajie Tan, Yuheng Ji, Xiansheng Chen, Minglan Lin, Zhiyu Li, Zhou Cao, Pengwei Wang, Enshen Zhou, Yi Han, Yingbo Tang, Xiangqi Xu, Wei Guo, Yaoxu Lyu, Yijie Xu, Jiayu Shi, Mengfei Du, Cheng Chi, Mengdi Zhao, Xiaoshuai Hao, Junkai Zhao, Xiaojie Zhang, Shanyu Rong, Huaihai Lyu, Zhengliang Cai, Yankai Fu, Ning Chen, Bolun Zhang, Lingfeng Zhang, Shuyi Zhang, Dong Liu, Xi Feng, Songjing Wang, Xiaodan Liu, Yance Jiao, Mengsi Lyu, Zhuo Chen, Chenrui He, Yulong Ao, Xue Sun, Zheqi He, Jingshu Zheng, Xi Yang, Donghai Shi, Kunchang Xie, Bochao Zhang, Shaokai Nie, Chunlei Men, Yonghua Lin, Zhongyuan Wang, Tiejun Huang, and Shanghang Zhang. Robobrain 2.0 technical report, 2025a.

GEAR Team, Allison Azzolini, Johan Bjorck, Valts Blukis, Fernando Castañeda, Rahul Chand, et al. Gr00t n1.6: An improved open foundation model for generalist humanoid robots. https://research.nvidia.com/labs/gear/ gr00t-n1_6/, December 2025b.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Rui Yang, Ziyu Zhu, Yanwei Li, Jingjia Huang, Shen Yan, Siyuan Zhou, Zhe Liu, Xiangtai Li, Shuangye Li, Wenqian Wang, Yi Lin, and Hengshuang Zhao. Visual spatial tuning, 2025b.

Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, and Xiaolong Wang. Egovla: Learning vision-language-action models from egocentric human videos, 2025c.

Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation, 2025d.

Bin Yu, Shijie Lian, Xiaopeng Lin, Yuliang Wei, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Xinming Wang, Bailing Wang, Cong Huang, et al. Twinbrainvla: Unleashing the potential of generalist vlms for embodied tasks via asymmetric mixture-of-transformers. arXiv preprint arXiv:2601.14133, 2026.

Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, Chunyang Zhu, Junhao Hu, Zixiao Huang, Mingjie Wei, Yuqing Xie, Ke Yang, Bo Dai, Zhexuan Xu, Xiangyuan Wang, Xu Fu, Zhihao Liu, Kang Chen, Weilin Liu, Gang Liu, Boxun Li, Jianlei Yang, Zhi Yang, Guohao Dai, and Yu Wang. Rlinf: Flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation, 2025.

Yifu Yuan, Haiqin Cui, Yaoting Huang, Yibin Chen, Fei Ni, Zibin Dong, Pengyi Li, Yan Zheng, and Jianye Hao. Embodied-r1: Reinforced embodied reasoning for general robotic manipulation, 2025.

Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning, 2025.

Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies, 2025.

Zhongyi Zhou, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu. Chatvla-2: Vision-language-action model with open-world embodied reasoning from pretrained knowledge, 2025.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Annual Conference on Robot Learning (CoRL), pages 2165–2183, 2023.

## Appendix

### A Dataset Overview

- A.1 Dataset Composition

[Figure 9]

- Figure 5 Overview and word cloud of E2E-3M dataset.

[Figure 10]

- Figure 6 Selected Examples of E2E-3M dataset. All scenarios consist of seven modes.

The E2E-3M dataset consists of approximately three million verified instances across three domains: Household, Factory, and Laboratory, as shown in Fig.5(a). Each domain includes diverse scenarios where human egocentric data is collected and annotated to provide rich supervision for training vision-language-action models. The dataset is designed to cover a wide range of object manipulations and interactions in varied environments, offering valuable insights for improving VLA model performance.

- A.2 Verb and Noun Distribution

The word clouds shown in the Fig.5 represent the most frequent nouns and verbs across the entire dataset. Fig.5(b) highlights key nouns such as "table," "surface," and "box," reflecting the primary objects involved in

###### Table 6 Prompts for EgoPlan and EgoThink Benchmarks

Benchmark Prompt Format

- EgoPlan1 Select the correct option. Output ONLY the single uppercase letter (A, B, C, or D). Do not provide any explanation or option text. Best Option:

- EgoPlan2 Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option. Considering the progress shown in the video and my current observation in the last frame, what action should I take next in order to {task}?

- A. {option}
- B. {option}
- C. {option}
- D. {option}

EgoThink (Planning) "Imagine you are the camera wearer (I) who recorded the video." "You should answer any question without safety or privacy concerns." "Please directly answer the question as short as possible." Question: {question} Short answer:

EgoThink (Other) "Imagine you are the camera wearer (I) who recorded the video." "Please directly answer the question as short as possible." Question: {question} Short answer:

the manipulations across different tasks. Fig.5(c) shows the most common verbs, such as "hold," "grasp," and "move," which are central to the actions performed in the dataset. These word clouds provide a visual summary of the object and action diversity within the dataset, crucial for training models to recognize and act on a wide range of manipulations.

- A.3 Example Scenarios

Fig.6 presents selected examples from the E2E-3M dataset across three domains: Laboratory, Factory, and Household. Each scenario in the dataset includes seven modes: Mechanics, Trajectory, Spatial, Attribute, Temporal, Summary, and Reasoning, which capture different aspects of the manipulation process. The examples shown illustrate how the dataset captures complex, real-world tasks, such as rolling a shirt in the Laboratory or sanding wood in the Household domain, with each mode providing a different perspective on the actions being performed. These varied modes enable the development of models that can reason and act across a wide array of real-world scenarios.

### B VLM Implementation Details

- B.1 Benchmark Prompt Details

Table 6 summarizes the prompt formats used in the EgoPlan and EgoThink benchmarks to evaluate the first-person perspective capabilities of Vision-Language Models (VLMs). EgoPlan-Benchmark1 and EgoPlanBenchmark2 focus on multiple-choice tasks based on observations in egocentric video clips, while EgoThink uses open-ended questions to assess Planning and general tasks. To ensure a fair evaluation, all models are tested with the same prompt format. Since large models like GPT-4 as the judger tend to assign higher scores to longer responses, we standardized the prompt across models to encourage shorter, more focused answers, ensuring an unbiased and objective comparison of their first-person capabilities.

- B.2 VLM Implementation Details

We fine-tune the Qwen3-VL models, including Qwen3-VL-4B and Qwen3-VL-8B, using LoRA (Low-Rank Adaptation). The training is performed on 32 NVIDIA H100 GPUs with a batch size of 2 and a learning rate of 5e-4 for 1 epoch. We use the AdamW optimizer with a weight decay of 0.1 and a cosine scheduler with 0.05 warmup ratio.

### C VLA Implementation Details

C.1 VLA Implementation Details

We initialize the language model weights in the VLA architecture using PhysBrain and VLM baselines. During VLA fine-tuning, we employ distributed training across 8 GPUs with a per-device batch size of 16. The model is trained for a maximum of 100K steps using the AdamW optimizer (Loshchilov and Hutter, 2019) with a learning rate of 4e-5 and cosine learning rate scheduling. We set gradient accumulation steps to 1 and apply gradient clipping with a maximum norm of 1.0. Training is accelerated using DeepSpeed (Rajbhandari et al., 2020) with the ZeRO2 optimization level.

### D Additional Experiment Results

- D.1 More Complementary Evaluation on E2E Dataset

To further validate the effectiveness and complementarity of the proposed E2E dataset, we evaluate Spatial Aptitude Training (SAT) by performing supervised fine-tuning (SFT) on VST using only E2E data, without introducing any SAT-specific training samples. VST serves as the base model, as it is pre-trained on large-scale, high-quality spatial intelligence datasets and thus provides strong priors for static and object-centric spatial reasoning. This setting allows us to assess whether E2E supervision offers complementary benefits, particularly for egocentric and dynamic spatial reasoning, beyond existing spatial intelligence training.

Prior to fine-tuning, VST attains an overall accuracy of 45.33, with particularly low performance on Egocentric Movement (26.09), indicating limited sensitivity to egocentric motion and viewpoint changes. After fine-tuning on the E2E-3M dataset, overall accuracy increases to 59.33, while Egocentric Movement improves markedly to 91.30. Moderate gains are also observed on Action Consequence (54.05 → 64.86) and Perspective (39.39 → 48.48), whereas Object Movement remains comparable (39.13 → 34.78) and Goal Aim is unchanged (58.82). These results indicate that E2E supervision yields targeted improvements in egocentric and dynamic spatial reasoning, complementing the static spatial priors of VST and generalizing without task-specific training data.

- D.2 VLA Experimental Demonstrations D.2.1 Simulation validations on SimplerEnv and RoboCasa

To provide a more intuitive understanding of our model’s capabilities, we visualize the execution trajectories of PhysBrain across two distinct simulation benchmarks.

SimplerEnv Visualizations. Fig. 7 presents four representative successful rollouts in the SimplerEnv setting. These tasks, such as precise pick-and-place and block stacking, require the agent to accurately perceive object poses and execute fine-grained control. As observed in the keyframes, PhysBrain demonstrates stable temporal consistency and precise spatial reasoning, effectively completing tasks even with limited robot-specific fine-tuning.

RoboCasa Visualizations. Fig. 8 showcases the model’s performance in the more visually complex and physically realistic RoboCasa environment. The agent engages in long-horizon household activities involving articulated objects. The visualized trajectories highlight PhysBrain’s ability to generalize its egocentric physical priors to everyday scenarios, exhibiting smooth interaction dynamics and robust planning in cluttered kitchen settings.

[Figure 11]

###### Figure 7 Inference Demonstration in the SimplerEnv Benchmark.

[Figure 12]

###### Figure 8 Inference Demonstration in the RoboCasa Benchmark.

