# arXiv:2512.14691v2[cs.CL]17Dec2025

## MMGR: Multi-Modal Generative Reasoning

Zefan Cai∗,1, Haoyi Qiu∗,2, Tianyi Ma∗,3, Haozhe Zhao∗,4, Gengze Zhou5, Kung-Hsiang Huang6, Parisa Kordjamshidi3, Minjia Zhang4, Wen Xiao7, Jiuxiang Gu8, Nanyun Peng2, Junjie Hu1

1University of Wisconsin–Madison, 2University of California, Los Angeles, 3Michigan State University, 4University of Illinois Urbana–Champaign, 5University of Adelaide, 6Salesforce AI Research, 7Microsoft, 8Adobe Research

∗Equal Contribution

Video foundation models have made striking progress in synthesizing visually compelling and temporally coherent content, yet their viability as world simulators hinges on whether they internalize the physical, logical, and spatial constraints that govern reality. Existing evaluation metrics—such as Fréchet Video Distance (FVD)—largely emphasize perceptual fidelity, leaving critical reasoning failures undetected, including hallucinations that violate causal structure, physical laws, and global consistency. To address this gap, we propose a principled evaluation framework grounded in five core reasoning abilities: Physical, Logical, 3D Spatial, 2D Spatial, and Temporal reasoning. Building on this framework, we introduce MMGR (Multi-Modal Generative Reasoning Evaluation and Benchmark), a comprehensive benchmark suite designed to assess generative reasoning across three complementary domains: Abstract Reasoning (e.g., ARC-AGI, Sudoku), Embodied Navigation (e.g., real-world 3D navigation and localization), and Physical Commonsense (e.g., sports and compositional physical interactions). MMGR evaluates both video and image generative models using fine-grained, domainspecific metrics that require holistic correctness rather than partial success. We benchmark state-ofthe-art video generation models—including Veo-3, Sora-2, and Wan-2.2—alongside leading image generation models such as Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image, revealing a pronounced performance asymmetry across modalities. While current models achieve moderate success on Physical Commonsense tasks, they fail catastrophically on Abstract Reasoning (achieving < 10% accuracy on ARC-AGI) and struggle with long-horizon spatial planning in embodied settings. Through detailed quantitative analysis and human evaluation, we identify key limitations in existing training paradigms: a severe imbalance favoring perceptual data over symbolic reasoning, architectural weaknesses in maintaining global state consistency, and optimization objectives that reward visual plausibility over causal correctness. By unifying abstract logic, embodied interaction, and intuitive physics under a single evaluation framework, MMGR provides a diagnostic lens into the reasoning deficits of modern generative models and outlines a concrete roadmap toward physically grounded, logically consistent, and reasoning-aware world models.

Website: https://zefan-cai.github.io/MMGR.github.io/ Repo: https://github.com/Zefan-Cai/MMGR Contact: zefncai@gmail.com, haoyiqiu@g.ucla.edu, matiany3@msu.edu, haozhez6@illinois.edu

1 Introduction

The field of generative artificial intelligence has achieved a paradigm shift with the advent of large-scale text-tovideo models (OpenAI, 2024b; Ho et al., 2022a; Singer et al., 2022; Blattmann et al., 2023). These systems can now synthesize photorealistic, diverse, and temporally rich scenes from simple natural-language prompts. This capacity to generate dynamic visual narratives promises to revolutionize filmmaking, scientific visualization, embodied simulation, and robotics. However, as generative models scale, evaluation remains a critical bottleneck. Conventional metrics—such as Fréchet Video Distance (FVD) (Unterthiner et al., 2018a), Inception Score (IS) (Salimans et al., 2016), and CLIP-based similarity (Radford et al., 2021)—prioritize perceptual fidelity: assessing whether a video looks realistic or aligns semantically with a caption. Yet, these metrics remain blind to world consistency and physical plausibility. Consequently, a model might render a visually stunning billiards shot where balls pass through one another, or a navigation sequence where an agent teleports through walls—hallucinations that satisfy texture-based metrics while violating fundamental laws of reality.

###### Benchmark Generation Evaluation

Input

[Granularity and Hard-level Control]

Input: solution image Video: generated video/image Prompt: Justify whether the video/image presents a solution based on the

[Figure 1]

[Figure 2]

###### Domain 1: Abstract Reasoning

Maze: 7x7, Medium

start

Prompt: Create a 2D animation

Maze: Grid Size, Maze Generator …

Sudoku: Grid Size, Puzzle Difficulty…

based on the provided

end

Math: GSM8K, Math500, AIME, Omni-Math ARC-AGI: v1, v2, match, mismatch, …

provided solution ….

image of a maze.

[Figure 3]

Video Generative Models

[Figure 4]

[Figure 5]

[Figure 6]

Domain 2: Embodied Navigation

Human

VLM as Judge

[Figure 7]

[Figure 8]

[Figure 9]

###### Veo-3 Sora-2

Wan-2.2

Tasks :

Evaluation

- - 3D Real World Navigation
- - Last-Mile Navigation
- - Top-Down View Navigation
- - Simultaneous Localization Hard-Level:
- - Environmental Complexity (# Floors)
- - Destination Specification (Color, Text)
- - View Fidelity
- - Trajectory Distance Domain 3: Physical Commonsense

[Figure 10]

Image Generative Models

NanoBanana (Pro)

Action Reflection: 0

[Figure 11]

Qwen-

[Figure 12]

[Figure 13]

GPT-4o-

Target Achievement: 1

Image

Image

Failure Mode: Maze Change: 0

Model: Veo-3

Cross Wall: 0

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Fine-grained Metrics

[Figure 18]

Overall: 1

[Figure 19]

Task: Physical Generation

Primary Metrics

Frame 1 Frame 3 Frame 5

- Figure 1 Overview of our proposed Multi-Modal Generative Reasoning (MMGR) benchmark. MMGR assesses whether generative models—both video and image—can perform coherent reasoning across three domains: Abstract Reasoning, Embodied Navigation, and Physical Commonsense. Given an input image and a generation prompt, video models (Veo-3, Sora-2, Wan-2.2) produce multi-frame trajectories, while image models (Nano-Banana/Pro, GPT-4o-image, Qwen-image) generate single-frame solutions. A VLM-based evaluator (Gemini-2.5-Pro) then scores each output using structured criteria, including an overall primary metric. For a curated subset of samples, we additionally conduct human evaluations. The full pipeline enables fine-grained, domain-sensitive analysis of generative reasoning capabilities.

We argue that for video generation to evolve from mere image animation to genuine world modeling (Ha & Schmidhuber, 2018; LeCun, 2022), models must acquire foundational reasoning capabilities akin to human intuitive physics and cognition. Moving beyond superficial fidelity (Huang et al., 2024; Liu et al., 2024b), we propose a formal evaluation framework asking: Can a video model reason about the physical and logical constraints of the content it generates? Drawing on theories of core knowledge and cognitive development (Spelke & Kinzler, 2007; Lake et al., 2017), we posit that robust world simulation rests on five complementary pillars of reasoning:

- 1. Physical Reasoning: Understanding intuitive physics, such as object permanence, gravity, collisions, and material properties. This capability aligns with theories of “core knowledge” in human cognition (Spelke & Kinzler, 2007; Baillargeon, 1987; Ullman et al., 2017; Piloto et al., 2022) and is a prerequisite for robust simulation and interaction (Battaglia et al., 2013; Yi et al., 2020; Wu et al., 2015; Bear et al., 2021; Riochet et al., 2021; Bakhtin et al., 2019; Allen et al., 2020).
- 2. Logical Reasoning: Manipulating abstract concepts, following rules, and performing logical operations (e.g., “if A happens, then B follows”). This mirrors the symbolic processing required for System 2 reasoning (Kahneman, 2011; Marcus, 2001; Lake et al., 2017), enabling generalization beyond simple pattern matching (Chollet, 2019; Johnson et al., 2017; Xu et al., 2024; Barrett et al., 2018; Zhang et al., 2021; Webb et al., 2023).
- 3. 3D Spatial Reasoning: Understanding 3D spatial relationships, navigating environments, and grasping topology. This involves building an internal “cognitive map” of the world (Tolman, 1948; Gibson, 1979; Epstein et al., 1999) to ensure geometric consistency across camera viewpoints (Hudson & Manning, 2019; Zhong et al., 2020; Wu et al., 2022).
- 4. 2D Spatial Reasoning: The accurate interpretation of visual layouts, shapes, and relative positions in the projected image plane. This relies on compositional image understanding (Biederman, 1987; Kosslyn,

1980) to correctly ground spatial prepositions in complex prompts (Johnson et al., 2017; Chollet, 2019; Hudson & Manning, 2019).

- 5. Temporal Reasoning: Modeling causality, the order of events, and long-range dependencies. This captures the human perceptual ability to segment continuous streams into discrete causal events (Michotte, 1963; Zacks & Tversky, 2001), which is essential for maintaining narrative coherence (Xiao et al., 2020; Piergiovanni et al., 2020; Zhou et al., 2022; Yi et al., 2020).

[Figure 20]

Domain 1: Abstract Reasoning

Maze Solving Sudoku Solving

ARC-AGI

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Hard, 13×13, (r->c) Medium, 9×9, (c->c) 4x4, Easy 9x9, Medium

Logical, 2D Spatial, Temporal Reasoning Logical, 2D Spatial Reasoning

Math Challenge

Question: Ken is the best sugar cube retailer in the nation. Trevor, who loves sugar, is

coming over to make an order. Ken knows Trevor cannot afford more than 127 sugar cubes, but might ask for any number of cubes less than or equal to that. Ken prepares

seven cups of cubes, with which he can satisfy any order Trevor might make. How

many cubes are in the cup with the most sugar? Category: Omni-Math, T3, Applied

V1, Easy, Match

Logical Reasoning Logical, 2D Spatial, Temporal Reasoning

[Figure 27]

Domain 2: Embodied Navigation

[Figure 28]

[Figure 29]

Physical, 3D Spatial, Temporal Reasoning Physical, 2D Spatial, Temporal Reasoning

[Figure 30]

[Figure 31]

Physical, 3D Spatial, Temporal Reasoning

Physical, 2D Spatial, 3D Spatial, Temporal Reasoning

[Figure 32]

Domain 3: Physical Commonsense

Physical Concept Sports

Prompt Text: A ballet dancer performs a fouetté turn, whipping one leg around while spinning on the other.

Prompt Text: A parasail is being inflated and prepared by crew,

showing the process of inflation and the parachute's texture.

Physics Focus: Conservation of angular momentum, centripetal force

Physics Focus: Action: parasailing Expected Motion: Action should be performed correctly

Expected Motion: Rapid leg whip, continuous rotation, stable center

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Physical, 3D Spatial, Temporal Reasoning Physical, 3D Spatial, Temporal Reasoning

- Figure 2 Overview of the three domains in the MMGR benchmark. MMGR evaluates multi-modal generative reasoning across Domain 1: Abstract Reasoning, Domain 2: Embodied Navigation, and Domain 3: Physical Commonsense. (1) Abstract Reasoning includes Maze Solving, Sudoku Solving, ARC-AGI, and Math Challenge tasks, which test logical, 2D spatial, and temporal reasoning. (2) Embodied Navigation spans four environment-conditioned tasks: Panoramic View Last-Mile Navigation, Top-down View Real-World Navigation, 3D Real-World Navigation, and Simultaneous Localization and Generation (SLAG). The four tasks probe 2D/3D spatial reasoning, physical scene understanding, and coherent temporal planning. (3) Physical Commonsense covers Physical Concept scenarios and Sports activities, evaluating whether models produce videos that follow intuitive physics such as force, momentum, rotation, material behavior, and continuous motion. Together, these domains provide a comprehensive testbed for assessing a model’s ability to generate physically plausible, spatially grounded, and logically coherent solutions.

##### Table 1 Overview of MMGR’s three task domains and their alignment with the five core reasoning abilities. Each task is annotated with the specific reasoning skills it evaluates along with the total number of samples per task. Together, these domains provide a comprehensive and systematic assessment of foundational reasoning competencies across abstract reasoning, embodied navigation, and physical commonsense.

Benchmark Task Domain Physical Logical 3D Spatial 2D Spatial Temporal # Samples

- Domain 1: Abstract Reasoning Maze (Section 5) ✓ ✓ ✓ 240 Sudoku (Section 6) ✓ ✓ 300 ARC-AGI (Section 7) ✓ ✓ ✓ 456 Math (Section 8) ✓ 327

- Domain 2: Embodied Navigation 3D Real-World Navigation (Section 12) ✓ ✓ ✓ 120 Last-Mile Navigation (Ego-centric) (Section 10) ✓ ✓ ✓ 120 Top-down View Navigation (Section 11) ✓ ✓ ✓ 120 Simultaneous Localization and Generation (Section 13) ✓ ✓ ✓ ✓ 120

- Domain 3: Physical Commonsense Physical Concept (Section 14) ✓ ✓ ✓ 25 Sports (Section 14) ✓ ✓ ✓ 25

Total Evaluation Samples 1,853

We explicitly distinguish 2D from 3D spatial reasoning because they rely on fundamentally different perceptual and computational mechanisms. While 2D reasoning operates on planar relationships—such as adjacency and relative positioning—3D reasoning necessitates depth estimation, viewpoint transformation, and occlusion handling. This separation mirrors human cognition, which processes flat representations (e.g., maps) differently than volumetric environments1.

Building upon this five-ability framework, we introduce MMGR (Multi-Modal Generative Reasoning), a benchmark suite designed to systematically assess generative reasoning across diverse settings. MMGR encompasses three complementary domains—ranging from abstract logic to embodied interaction—that each necessitate the coordination of multiple reasoning abilities (see Table 1, Figure 1, Figure 2):

- 1. Abstract Reasoning: Evaluates Logical, 2D Spatial, and Temporal reasoning in non-photorealistic environments. Key tasks include synthetic Maze environments (Ivanitskiy et al., 2023), Sudoku (Seely et al., 2025) (applying rule-based 2D spatial logic), Math (visualizing symbolic solution paths), and ARC-AGI (Chollet, 2019; Xu et al., 2024) (performing spatial–logical transformations).
- 2. Embodied Navigation: Assesses the synthesis of Physical, 2D/3D Spatial, and Temporal reasoning from an agent-centric perspective. Models are tasked with generating successful trajectories within diverse settings, specifically complex real-world navigation scenes utilizing both egocentric and top-down views (Chang et al., 2017; Ramakrishnan et al., 2021b; Savva et al., 2019; Anderson et al., 2018b; Zhu et al., 2017; Ramakrishnan et al., 2021a; Chaplot et al., 2020a; Deitke et al., 2020).
- 3. Physical Commonsense: Probes the understanding of intuitive physics (Battaglia et al., 2013; Yi et al., 2020; Bear et al., 2021; Bakhtin et al., 2019) and object dynamics. The scope extends from fundamental concepts (leveraging the VideoPhy (Bansal et al., 2024) ontology) to compositional sports scenarios that require modeling physically plausible interactions consistent with real-world constraints.

By evaluating state-of-the-art image generative models (i.e., Nano-banana, Nano-banana Pro, GPT-4o-image, Qwen-image) and video generative models (i.e., Veo-3, Sora-2, Wan-2.2) (DeepMind, 2025b; OpenAI, 2024a; Qwen, 2024; DeepMind, 2025a; OpenAI, 2025; Wan, 2025) on MMGR, we deliver the first comprehensive characterization of their generative reasoning capabilities. Our results reveal a consistent trend: while models demonstrate encouraging performance on Physical Commonsense tasks (e.g., Sports: 60%), they struggle markedly with Abstract Reasoning challenges such as ARC-AGI (<10%) (Chollet, 2019; Barrett et al., 2018) and with long-horizon, multi-step planning in the Embodied Navigation domain (e.g., S.L.A.G.: 3.64% holistic success (Ivanitskiy et al., 2023; Savva et al., 2019)).

1Our benchmark targets this dichotomy to enable fine-grained diagnosis of model capabilities: abstract tasks like Sudoku and ARC-AGI probe 2D grid-based logic, whereas embodied navigation tasks demand coherent 3D spatial understanding.

These performance patterns illuminate several critical deficiencies in current training recipes, offering a guide for future model development:

- • Training Data Imbalance: While current video corpora abound in naturalistic physical interactions (e.g., sports, everyday dynamics)—explaining strong Physical Commonsense performance—they severely lack structured, symbolic reasoning data. This deficit leads to near-random performance on logic-heavy tasks like Sudoku (< 7%) and ARC-AGI. Furthermore, the stark disparity between Final Correctness (74%) and Intermediate Correctness (12%) on Math tasks (e.g., GSM8K) suggests that models are merely memorizing answer patterns rather than learning genuine multi-step reasoning.
- • Architectural Limitations: The pronounced divergence between primary success metrics and holistic Overall scores (e.g., 80.56% vs. 20.83% on 3D Real-World Navigation) indicates that current architectures sacrifice global consistency for local plausibility. With Scene Consistency dropping to 40.28% and Destination Integrity to 25.45%, models struggle to enforce long-range spatial and temporal coherence. This highlights an urgent need for mechanisms—such as external memory, world-state representations, or structured latent spaces—to sustain context across extended generation horizons.
- • Optimization Objective Gaps: Current objectives prioritize perceptual fidelity (via reconstruction loss or adversarial objectives) over reasoning correctness. Consequently, models optimize for appearance rather than logical validity—rendering visually convincing mazes or equations without actually solving them. Future work must integrate auxiliary objectives that reward rule adherence and causal consistency, potentially leveraging reinforcement learning from structured feedback or neuro-symbolic supervision.

Ultimately, MMGR provides a unified framework for diagnosing these limitations, charting a path toward video generation systems that are physically grounded, logically consistent, and truly reasoning-aware.

- 2 Related Work

Video Generation Models. The field of video generation has witnessed a paradigm shift, evolving from early GAN-based approaches (Vondrick et al., 2016; Tulyakov et al., 2018) to diffusion-based systems (Ho et al., 2022b; Singer et al., 2022) and large-scale transformer architectures (Yan et al., 2021; Hong et al., 2022). Contemporary state-of-the-art models, including Sora (OpenAI, 2024b), Veo (DeepMind, 2024), and Kling (Kuaishou, 2024), demonstrate exceptional capacity for synthesizing high-fidelity, photorealistic video with complex temporal dynamics. However, while these models excel at surface-level perceptual quality, the extent to which they internalize the underlying physical laws and logical constraints of the world remains an active area of inquiry.

Evaluation of Generative Models. Traditional evaluation metrics have largely prioritized appearance quality over semantic consistency. Metrics such as FVD (Unterthiner et al., 2018b) and Inception Score (IS) (Salimans et al., 2016) capture perceptual fidelity, while more recent benchmarks (Huang et al., 2024; Liu et al., 2024a) focus on text–video alignment and basic temporal consistency. These tools, however, are insufficient for probing world modeling capabilities. They fall short of evaluating whether a model possesses the reasoning skills necessary to generate content that is not only visually plausible but also logically coherent and physically robust over long horizons.

From Visual Understanding to Generative Reasoning. Prior benchmarks in video understanding (Girdhar & Ramanan, 2020; Goyal et al., 2017; Chollet, 2019) primarily assess discriminative models—testing their ability to recognize interactions or perform symbolic reasoning on existing inputs. Similarly, embodied AI benchmarks (Savva et al., 2019) rely on rigid simulators to test perception. Our work shifts this paradigm from understanding to generation: requiring models to not merely interpret a video, but to manifest reasoning processes through synthesis. Recent studies have begun to explore this frontier. Wiedemer et al. (2025) identify emergent “Chain-of-Frames” (CoF) reasoning in models like Veo-3, while Guo et al. (2025) utilize MME-CoF to expose failures in geometric consistency. Tong et al. (2025) further demonstrate competitive performance by Sora-2 across vision tasks. We build upon these insights by formalizing a five-ability reasoning framework. Unlike prior works that focus on specific failure modes or emergent properties, we provide a

- Table 2 Summary of MMGR benchmark statistics across three domains.

Domain / Task # Samples Primary Metric

- Domain 1: Abstract Reasoning Maze 240 Valid Solution Sudoku 300 Valid Solution ARC-AGI 456 Valid Solution Math 327 Valid Solution

- Domain 2: Embodied Navigation 3D Real-World Navigation 120 Overall Success Last-Mile Navigation 120 Overall Success Top-down View Navigation 120 Overall Success SLAG 120 Overall Success

- Domain 3: Physical Commonsense Physical Concept 25 Physical Plausibility Sports 25 Physical Plausibility

Total 1,853

holistic assessment spanning Abstract Reasoning, Embodied Navigation, and Physical Commonsense, creating generative adaptations of rigorous tasks such as ARC-AGI to test the limits of current world models.

- 3 Benchmark Overview

Our three evaluation domains are grounded in the principle that world modeling necessitates both internal and external simulation capabilities (Ha & Schmidhuber, 2018; LeCun, 2022). Abstract Reasoning targets internal simulation—the capacity to manipulate symbolic representations, adhere to logical rules, and execute mental transformations independent of physical reality. Conversely, Embodied Navigation and Physical Commonsense evaluate external simulation—the ability to model interactions within the physical world. Specifically, Embodied Navigation tests a model’s capacity to simulate agent-environment dynamics for spatial planning, whereas Physical Commonsense assesses an understanding of the intuitive physics governing real-world objects.

These domains are strategically complementary; together, they exercise the five core reasoning abilities outlined in Section 1. Abstract Reasoning prioritizes Logical and 2D Spatial reasoning, while the external domains integrate 3D Spatial, Temporal, and Physical reasoning through dynamic scenarios (see Table 1 for the complete mapping). This design ensures a comprehensive evaluation of the reasoning competencies essential for robust world modeling. Figure 2 shows several examples from MMGR.

- 3.1 Abstract Reasoning

Maze. Designed to assess 2D spatial, logical, and temporal reasoning, this task requires models to navigate a valid path from a start cell (green) to a goal cell (red) while avoiding obstacles. We employ DFS and Wilson’s algorithms to generate 240 mazes across three difficulty levels—Easy (3 × 3–5 × 5), Medium (6 × 6– 9 × 9), and Hard (10 × 10–13 × 13)—using four distinct start-goal configurations (e.g., corner-to-corner, random-to-random) (Ivanitskiy et al., 2023).

Sudoku. This task evaluates constraint satisfaction and logical deduction (Seely et al., 2025). Models must complete grids such that every row, column, and subgrid contains unique digits. The dataset comprises 300 puzzles across two grid sizes (4 × 4 and 9 × 9) and three difficulty levels (Easy, Medium, and Hard), where complexity is modulated by the sparsity of initial clues.

ARC-AGI. To evaluate abstract reasoning and few-shot rule induction, we utilize the ARC-AGI benchmark (Chollet, 2019). Models must infer latent transformation rules from input-output demonstration examples and apply them to unseen test cases. Our benchmark comprises 456 tasks from v1 (381 tasks) and v2 (75 tasks), classified by shape consistency (Match and Mismatch) and quantitative difficulty (Easy, Medium, and Hard).

Visual Math. We assess mathematical reasoning across diverse domains using five benchmarks: GSM8K (Cobbe et al., 2021) (grade school), MATH500 (Hendrycks et al., 2021) (high school), AIME 2024/2025 (Mathematical Association of America, 2024, 2025) (invitational competitions), and Omni-MATH (Gao et al., 2024) (Olympiadlevel). The resulting dataset contains 327 problems requiring logical deduction and spatial understanding.

- 3.2 Embodied Navigation
- 3D Real-World Navigation. Utilizing cutaway “dollhouse” renderings from Matterport3D (Chang et al., 2017) and HM3D (Ramakrishnan et al., 2021b), this task assesses multi-room and multi-level spatial reasoning. Models operate from a fixed third-person perspective to generate navigation trajectories, requiring them to interpret full 3D scene structures, including verticality and complex room connectivity.

Last-Mile Navigation (Ego-centric). This setting presents a 360◦ panoramic environment via a proximal “over-the-shoulder” view. Models must synthesize wide-field visual context to execute short-range navigation, necessitating the interpretation of agent-centric layouts to generate goal-directed trajectories.

Top-down View Navigation. Adopting a fixed bird’s-eye perspective, this task targets global spatial planning and long-horizon prediction. Models generate trajectories on 2D overhead maps, emphasizing the ability to reason about global geometry and multi-step pathfinding.

Simultaneous Localization and Generation (SLAG). SLAG integrates both 3D and top-down views, challenging models to jointly localize the agent while generating the surrounding scene layout. This requires maintaining geometric coherence and performing cross-view spatial alignment across distinct observation modalities.

Dataset Configuration. We evaluate each of the four tasks on 120 samples. The dataset spans 24 configurations stratified by environmental complexity (single vs. multi-floor), view fidelity (quality 3–5), trajectory distance (short vs. long), and goal specification (visual marker vs. linguistic description).

- 3.3 Physical Commonsense

Physical Concept. Leveraging the VideoPhy ontology (Bansal et al., 2024), this task assesses intuitive understanding of fundamental physical interactions. We evaluate three core categories: Solid-Solid (143 captions), Solid-Fluid (146 captions), and Fluid-Fluid (55 captions). The dataset spans broad physical domains including statics, dynamics, kinematics, and hydrodynamics. Additionally, we incorporate VideoPhy v2 to expand the evaluation scope with 600 supplementary captions covering 197 unique physical actions. From the larger source corpus, we randomly sample 25 examples and ensure diversity across these interaction categories and physical domains to create a balanced evaluation set.

Sports. This task evaluates compositional physical reasoning within complex scenarios characterized by the intersection of multiple physical laws. The source corpus encompasses diverse activities—specifically Ballet (12), Skiing (13), Diving (12), and Swimming (13)—challenging models to analyze phenomena such as momentum conservation, balance control, projectile motion, and fluid dynamics in goal-oriented contexts. To construct a balanced evaluation set, we randomly sampled 25 diverse examples from this larger collection.

4 Experimental Setup

To systematically evaluate zero-shot reasoning capabilities (Wiedemer et al., 2025; Guo et al., 2025; Tong et al., 2025), we benchmark state-of-the-art generative models across the ten tasks outlined in Section 3. Our analysis aims to quantify model performance and disentangle granular strengths and limitations within the five core reasoning dimensions.

- 4.1 Data

- Table 2 provides a comprehensive statistical overview of the benchmark, which aggregates 1,853 testing samples across three domains and ten tasks. To facilitate fine-grained capability analysis, we employ rigorous

difficulty stratifications and human verification. For Abstract Reasoning, complexity is modulated by grid dimensions (Maze, Sudoku), shape consistency (ARC-AGI), and mathematical scope (Math). Similarly, Embodied Navigation tasks are organized into 24 distinct configurations defined by environmental complexity, visual fidelity, trajectory distance, and goal specification.

- 4.2 Generation Settings

To ensure robust performance estimation and account for stochastic variability, we generate 5 samples per prompt for every model. We strictly adhere to the default API parameters (for closed-source models such as Sora-2, Veo-3, GPT-4o-image, Nano-banana, and Nano-banana Pro) and recommended configurations (for open-weights models such as Wan-2.2 and Qwen-image) to guarantee a fair, zero-shot comparison without task-specific fine-tuning.

- 4.3 Evaluation Protocol

VLM-based Evaluation. Following established video benchmarking protocols (Huang et al., 2024; Liu et al., 2024b; Wiedemer et al., 2025), we employ Gemini 2.5-Pro (Comanici et al., 2025) as a unified automated evaluator. The model assesses generation quality using task-specific rubrics that evaluate both the plausibility of the reasoning process and the correctness of the final result.

Metrics Aggregation. We begin by reporting diverse task-specific fine-grained metrics to dissect model performance across individual reasoning dimensions. Building on these components, we define a strict primary metric for each task that necessitates the simultaneous satisfaction of all sub-metrics (detailed in Table 2). We prioritize this holistic measure to address the disparity between partial success and complete correctness—a gap that typically inflates performance estimates by 1.2–4× when ignored.

- 4.4 Models for Evaluation

Our study evaluates a diverse selection of state-of-the-art multimodal generative models, spanning both video and image modalities. We include representative closed-source and open-weights models from major research laboratories, as detailed in Table 3.

Table 3 Generative models evaluated in MMGR benchmark, categorized by modality.

Model Source

Video Generation Models Sora-2 (OpenAI, 2024b) Closed Veo-3 (DeepMind, 2024) Closed Wan-2.2 (Wan, 2025) Open

Image Generation Models Nano-banana (Comanici et al., 2025) Closed Nano-banana Pro (DeepMind, 2025b) Closed GPT-4o-image (OpenAI, 2024a) Closed Qwen-image (Qwen, 2024) Open

- 4.5 Human Evaluation

To establish ground-truth performance and validate the reliability of VLM-based automatic evaluation, we conducted systematic human annotation on generated outputs. This human evaluation serves as a critical complement to AutoEval, particularly for tasks requiring nuanced judgment of temporal consistency, spatial reasoning, and physical plausibility.

Annotation Interface. We developed a web-based annotation platform (Figure 3) featuring full video playback controls including frame-by-frame navigation and adjustable playback speed. The interface displays the original task prompt alongside the generated video and provides structured evaluation forms tailored to each

[Figure 39]

(a) Maze Navigation interface with failure mode detection (Maze Changed, Cross Wall) and navigation behavior metrics.

[Figure 40]

(b) Math Reasoning interface with process correctness and outcome accuracy assessment.

- Figure 3 Human annotation interface for evaluating generated videos. The interface provides video playback controls (frame-by-frame navigation, speed adjustment), displays the original problem/condition, and presents structured evaluation forms with task-specific metrics.

task type. Annotators assess multiple dimensions including task completion, process correctness, and failure modes with associated confidence ratings.

Evaluation Protocol. We recruited 6 annotators with bachelor education background. The training process included a 4-hour instruction session, a 50-video practice phase, and calibration meetings.

### 5 Maze

- 5.1 Task Description

We introduce the 2D Maze task to evaluate a model’s foundational reasoning capabilities. This task is a direct probe for 2D Spatial Reasoning, as the model must understand the topology of the maze (i.e., the white path vs. the black walls). Furthermore, it challenges Logical Reasoning by requiring the model to generate a valid plan (the solution path) from a start state (green square) to a goal state (red square). Finally, it tests Temporal Reasoning by requiring the model to execute this plan sequentially over time, moving the agent along the path without deviation.

- 5.2 Hard-Level Control

To ensure a diverse and controllable set of evaluation cases, we leverage the open-source Python library maze-dataset (Ivanitskiy et al., 2023) to programmatically generate mazes of varying structure and difficulty. We vary task parameters along three axes:

- • Generators (2 types): We employ two maze-generation algorithms—Depth-First Search (DFS) and Wilson’s Algorithm—to produce topologically diverse maze layouts.
- • Grid Sizes (10 levels): Maze difficulty is scaled across ten grid sizes, ranging from 3×3 to 13×13.
- • Start–Goal Placement (4 schemes): Each maze is instantiated under four placement schemes—corner-tocorner, corner-to-random, random-to-corner, and random-to-random—to prevent models from overfitting to a single trajectory pattern. A minimum start–goal distance is enforced to rule out trivial solutions.

For each generator, we produce 120 mazes, comprising 40 Easy (3×3-5×5), 40 Medium (6×6-9×9), and 40 Hard (10×10-13×13) instances. Overall, this yields 240 mazes across the two generators. With our generation algorithm, each maze only has one solution path. Figure 4 presents representative examples across difficulty levels and start–end configurations, along with their corresponding solutions.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

(a) Easy 5×5 (c→c) (b) Easy 3×3 (r→c) (c) Easy 5×5 (c→r) (d) Easy 4×15 (r→r)

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

(e) Medium 9×9 (c→c) (f) Medium 9×9 (r→c) (g) Medium 9×9 (c→r) (h) Medium 8×8 (r→r)

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

(i) Hard 11×11 (c→c) (j) Hard 13×13 (r→c) (k) Hard 13×13 (c→r) (l) Hard 13×13 (r→r)

- Figure 4 Maze-solving results across three difficulty levels (Easy, Medium, Hard) and four start-end configurations: corner→corner, random→corner, corner→random, and random→random. Each subfigure shows a unique DFS solution illustrating how maze size and start/end randomness affect traversal patterns. Solution paths are highlighted blue.

- 5.3 Evaluation and Metrics

We evaluate generated videos and images using a Vision–Language Model (VLM)–based evaluator, Gemini2.5-Pro (Comanici et al., 2025). The evaluator receives three inputs: (i) the model-generated video or image, (ii) the ground-truth maze solution image, and (iii) a structured evaluation prompt (with modality-specific variants for video vs. image). Given these inputs, the VLM judges whether the model solved the task and provides fine-grained feedback across multiple failure modes. The evaluation prompt asks:

- • Does the green square (start) reach and stop on the red square (end)?
- • Does the green square ever touch or cross a black wall?
- • Does the layout of the black walls or the position of the red square change at any time?

Using the VLM’s responses, we compute one primary metric and four fine-grained metrics:

- • Maze Changed (Failure Mode): (i) Video: 1 if the maze layout changes in any frame, 0 if unchanged throughout. (ii) Image: 1 if the maze structure differs from the solution reference, 0 otherwise.
- • Cross Wall (Failure Mode): (i) Video: 1 if the green square crosses a black wall in any frame, 0 only if it stays on white paths at all times. (ii) Image: 1 if the blue path touches or crosses black walls, 0 if fully contained within the white corridors.
- • Action Reflection: (i) Video: 1 if the video shows exploratory behavior (e.g., backtracking or trying multiple paths), 0 for a single direct route. (ii) Image: 1 if the rendered blue trajectory depicts multiple

- attempted paths, 0 for a single direct path.
- • Target Achievement: (i) Video: 1 if the green square reaches and stops on the red square in any frame, 0 otherwise. (ii) Image: 1 if a continuous, valid blue path connects start and end.
- • Overall Score: 1 only if Maze Changed=0 AND Cross Wall=1 AND Task Completion=1; 0 otherwise.

- 5.4 Case Study

Image Generation. Figure 5a illustrates Nano-Banana’s behaviors, showing not only perfectly solved outputs but also common failure modes unique to image-based generation. These include wall-crossing artifacts in the final predicted trajectory, slight distortions of maze geometry, and “action-reflection” artifacts—cases where the rendered trajectory contains redundant loops or implausible detours even though no temporal dynamics are involved. These artifacts reflect the model’s uncertainty when inferring long-range paths from a single static instruction, resulting in inconsistent or physically implausible solution traces.

Video Generation. Figure 5b shows Veo-3’s successful generations. Frame-by-frame annotations reveal that Veo3 can preserve the maze’s topology, keeps the green square strictly on valid white paths, and maintains consistent wall boundaries from start to finish. The model occasionally performs mild exploratory behaviors—such as brief backtracking or short directional adjustments between early frames—before ultimately converging on the correct route. Notably, these explorations remain structurally valid: the agent never crosses walls or distorts the environment, and the maze remains unchanged throughout the entire sequence. In contrast, Figure 5c highlights Veo-3’s failure cases. Here, the model sometimes introduces structural inconsistencies—removing or altering wall segments, inserting open passages, or shifting the geometry of the target region. In other cases, the green square traverses invalid regions (e.g., sliding across black walls during transitions) or produces contradictory intermediate frames despite ending at the correct goal cell. The examples also show how subtle frame-to-frame drifts, such as disappearing wall pixels or morphing corridors, can accumulate into integrity violations not captured by coarse success metrics. Collectively, these case studies show that both models may successfully reach the red goal but still differ dramatically in path fidelity, wall adherence, and temporal consistency. The frame-level evidence—ranging from clean, stable trajectories to structurally inconsistent or wall-violating behaviors—underscores the necessity of a fine-grained maze-evaluation framework capable of capturing these nuanced, multimodal failure modes that simple goal-achievement metrics overlook.

- 5.5 Evaluation Results

Key Finding: The Illusion of Competence and Physical Grounding

Our analysis uncovers two critical disconnects in video generation reasoning. First, a dichotomy of simulation: Veo-3 mimics the result, generating direct solution paths via pattern matching but failing to respect impermeable boundaries. Conversely, Sora-2 mimics the process, performing visible “reasoning” (backtracking, hesitation) but losing logical coherence (hallucinating maze structures). Second, human verification reveals an evaluation gap: Automated VLM metrics systematically overestimate model competence by missing transient “physics violations.” While Auto-Eval reports moderate success for Veo-3, human review reveals that the model effectively “cheats” by clipping through walls in fast motion—a failure mode invisible to current VLMs. Consequently, true adherence to physical constraints remains near zero, suggesting current models prioritize visual plausibility over logical validity.

- 5.5.1 VLM-Based Evaluation

- Table 4 isolates the reasoning capabilities of video and image generative models by correlating path planning strategies with environmental consistency. Veo-3 displays a Direct Execution” reasoning style: its near-zero Action Reflection (0.00%–3.00%) confirms it generates a single, non-exploratory route without backtracking. While this allows for high Target Achievement (up to 62.00%), the model’s reliance on one-shot generation comes at the cost of physical precision, evidenced by significant Cross Wall rates (18.00%–25.00%). This suggests Veo-3 solves mazes via pattern-matching (predicting the solution trajectory directly) rather than

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Failure Case 2: Maze puzzle is solved

Failure Case 3: Maze puzzle is solved

Success Case: Maze puzzle solved perfectly by Nano-Banana, with

Failure Case 1: Maze puzzle is solved incorrectly by Nano-Banana,

incorrectly by Nano-Banana, crossing walls even though it

incorrectly by Nano-Banana, crossing walls even though it

target completion and no constraint violations.

crossing walls even though it reaches the target, with additional

reaches the target, with additional

reaches the target, with additional

action-reflection behavior.

action-reflection behavior.

action-reflection behavior.

Maze Changed = 0

Maze Changed = 0 Cross Wall = 1

Maze Changed = 0 Cross Wall = 1

Cross Wall = 0 Target Completion = 1

Maze Changed = 0 Cross Wall = 1

Target Completion = 1 Action Reflection = 1

Target Completion = 1 Action Reflection = 1

Action Reflection = 0

Target Completion = 1 Action Reflection = 1

##### (a) Success and failure cases generated by Nano-Banana. Solution paths are highlighted in blue.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Start

End

[Figure 67]

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Maze Changed = 0

Target Achievement = 1 The green square successfully reaches

Action Reflection = 0 In the video, the green square

Cross Wall = 0

The maze structure,

Throughout all frames of the video, the green

the red square goal, which is located

does not exhibit exploratory

including all black walls and white paths, remains

square stays exclusively on white path areas. There is absolutely no frame where the green

in the same position as the goal in the reference solution image. The

behavior (e.g., backtracking or attempting multiple paths) across

completely unchanged

square touches, overlaps, or crosses into any

throughout the entire animation.

black wall regions during its navigation. The square maintains proper boundaries at all times.

video ends with the green square stopped on the goal.

frames and it shows only a direct route.

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Backtracking

Start

End

[Figure 75]

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Frame 6 Maze Changed = 0

Target Achievement = 1

Action Reflection = 1

Cross Wall = 0

The green square reaches the red goal

The green square shows

The maze layout—with

Across every frame of the video, the green square

square, matching the goal position shown in the reference solution. The

exploratory actions—such as backtracking or probing

all black walls and white pathways—stays entirely the same for the duration

remains strictly on the white path. At no point does it touch, overlap, or enter any black wall region. It consistently preserves correct

video concludes with the green square resting on the goal.

alternative routes—between frames 2 and 3 as it looks for a

of the animation.

boundaries throughout its entire movement.

new path forward.

##### (b) Success cases generated by Veo-3. Solution paths are highlighted in blue.

Start

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

End

crossing wall !

maze changed !

[Figure 81]

Frame 1 Frame 2 Frame 3 Frame 4 Frame 5 Maze Changed = 1 The maze structure in the last several frames

Cross Wall = 1 The green square crosses walls

Target Achievement = 1 The green square successfully

Action Reflection = 0 In the video, the green

during its movement. For example,

reaches the red square goal,

square does not exhibit

differ from the reference, constituting a failure of the maze integrity criterion.

between frames 3 to 5, the green square moves from one white cell

which is located in the same position as the goal in the

exploratory behavior (e.g., backtracking or

Specifically, a segment of the central black

to another non-adjacent white cell, passing directly over a black

reference solution image. The video ends with the green square

attempting multiple paths) across frames and it shows

wall immediately to the bottom of the ending position is missing, replacing a solid obstacle

wall cell in the process.

stopped on the goal.

only a direct route.

with an open path.

(c) Failure cases generated by Veo-3. Solution paths are highlighted in blue.

- Figure 5 Case studies comparing success and failure behaviors across Nano-Banana and Veo-3.

##### Table 4 Quantitative results for the 2D Maze task. We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image) across two maze generation algorithms (DFS and Wilson’s) and three difficulty levels (Easy, Medium, and Hard). The highest overall scores in each setting are highlighted in bold.

Fine-grained Metrics Primary Metric Model Maze Changed ↓ Cross Wall ↓ Action Reflection ↑ Target Achievement ↑ Overall ↑ Generator: Depth-First Search Level: Easy (3×3–5×5)

Video Models Veo-3 15.50% 25.50% 1.00% 60.50% 42.00% Sora-2 67.50% 7.50% 77.50% 12.50% 2.50% Wan-2.2 35.00% 79.17% 7.50% 10.00% 1.67%

Image Models Nano-banana 5.00% 30.00% 15.50% 85.00% 15.50% Nano-banana Pro 5.00% 42.50% 10.00% 90.00% 17.50% GPT-4o-image 95.00% 5.00% 7.50% 72.50% 0.00% Qwen-image 5.00% 23.33% 15.00% 65.00% 11.67%

Level: Medium (6×6–9×9)

Video Models Veo-3 0.50% 25.63% 0.00% 50.75% 38.69% Sora-2 47.50% 12.50% 60.00% 10.00% 7.50% Wan-2.2 10.83% 90.83% 28.33% 23.33% 1.67%

Image Models

Nano-banana 0.63% 30.63% 11.88% 71.25% 4.38% Nano-banana Pro 0.00% 25.00% 30.00% 82.50% 2.50% GPT-4o-image 72.50% 10.00% 0.00% 82.50% 5.00% Qwen-image 2.50% 28.33% 12.50% 44.17% 0.00%

Level: Hard (10×10–13×13)

Video Models

###### Veo-3 0.00% 18.50% 1.50% 60.00% 51.50% Sora-2 57.50% 7.50% 60.00% 25.00% 10.00% Wan-2.2 6.67% 80.83% 35.00% 20.00% 5.00%

Image Models

###### Nano-banana 0.00% 24.17% 18.33% 60.00% 0.83% Nano-banana Pro 0.00% 12.50% 20.00% 80.00% 5.00% GPT-4o-image 62.50% 5.00% 5.00% 77.50% 0.00% Qwen-image 7.50% 11.67% 11.67% 42.50% 1.67%

Generator: Wilson’s Algorithm

Level: Easy (3×3–5×5)

Video Models Veo-3 3.50% 21.50% 2.50% 61.50% 46.50% Sora-2 67.50% 10.00% 40.00% 15.00% 5.00% Wan-2.2 32.50% 84.17% 8.33% 15.00% 1.67%

Image Models Nano-banana 10.50% 47.50% 16.50% 81.00% 6.50% Nano-banana Pro 10.00% 32.50% 30.00% 85.00% 12.50% GPT-4o-image 82.50% 15.00% 0.00% 90.00% 2.50% Qwen-image 5.83% 12.50% 9.17% 67.50% 20.00%

Level: Medium (6×6–9×9)

Video Models

###### Veo-3 1.25% 15.63% 1.25% 55.63% 47.50% Sora-2 62.50% 12.50% 47.50% 20.00% 10.00% Wan-2.2 14.17% 85.83% 15.83% 14.17% 1.67%

Image Models

Nano-banana 0.00% 36.88% 13.75% 70.00% 1.25% Nano-banana Pro 2.50% 17.50% 15.00% 80.00% 2.50% GPT-4o-image 75.00% 5.00% 5.00% 80.00% 5.00% Qwen-image 4.17% 25.00% 14.17% 47.50% 0.00%

Level: Hard (10×10–13×13)

Video Models Veo-3 1.25% 18.75% 0.63% 58.75% 45.63% Sora-2 45.00% 10.00% 52.50% 10.00% 2.50% Wan-2.2 10.00% 89.17% 28.33% 13.33% 0.83%

Image Models

###### Nano-banana 1.25% 27.50% 23.13% 60.63% 0.00% Nano-banana Pro 0.00% 22.50% 17.50% 75.00% 5.00% GPT-4o-image 60.00% 7.50% 5.00% 77.50% 7.50% Qwen-image 2.50% 27.50% 20.00% 38.33% 0.00%

active search, often fudging” collisions to maintain momentum. In contrast, Sora-2 exhibits “Performative Reasoning”: it achieves high Action Reflection scores (40.00%–78.00%) by generating visible exploratory behaviors like backtracking and trying multiple paths. However, this exploration is fundamentally disconnected from valid state maintenance—the model frequently hallucinates new maze structures (Maze Changed 45.00%– 68.00%) while searching, and rarely achieves the target (10.00%–25.00%). Wan-2.2 and the Nano-banana family demonstrate a failure of physical constraint satisfaction: their high Cross Wall rates (up to 91.00%) indicate they treat walls as visual suggestions rather than impermeable boundaries, allowing them to traverse the maze without solving the topological puzzle. Finally, GPT-4o-image bypasses the reasoning task entirely, altering the problem definition (Maze Changed up to 95.00%) to fabricate a successful outcome.

- 5.5.2 Human Evaluation

To establish ground-truth performance estimates and validate the reliability of our VLM-based evaluator, we conducted a human evaluation on a subset of Veo-3’s generated videos. Table 5 presents a side-byside comparison of Auto-Eval versus Human-Eval across all maze generators and difficulty levels. The results highlight a substantial divergence between automated and human assessments, with human evaluators consistently uncovering failure modes that the VLM overlooks.

- Table 5 Comparison of Auto-Eval (VLM-based) and Human-Eval results for Veo-3 on the 2D Maze task across two maze generation algorithms (DFS and Wilson’s) and three difficulty levels (Easy, Medium, and Hard).

Auto-Eval Human-Eval Setting MC ↓ CW ↓ AR ↑ TA ↑ Overall ↑ MC ↓ CW ↓ AR ↑ TA ↑ Overall ↑ Generator: Depth-First Search

Easy (3×3–5×5) 15.50% 25.50% 1.00% 60.50% 42.00% 10.00% 80.00% 40.00% 70.00% 10.00% Medium (6×6–9×9) 0.50% 25.63% 0.00% 50.75% 38.69% 20.00% 90.00% 90.00% 20.00% 0.00%

- Hard (10×10–13×13) 0.00% 18.50% 1.50% 60.00% 51.50% 20.00% 100.00% 40.00% 70.00% 0.00%

Generator: Wilson’s Algorithm

Easy (3×3–5×5) 3.50% 21.50% 2.50% 61.50% 46.50% 40.00% 70.00% 60.00% 40.00% 20.00% Medium (6×6–9×9) 1.25% 15.63% 1.25% 55.63% 47.50% 20.00% 100.00% 60.00% 40.00% 0.00%

- Hard (10×10–13×13) 1.25% 18.75% 0.63% 58.75% 45.63% 40.00% 100.00% 70.00% 40.00% 0.00% Note: MC = Maze Changed, CW = Cross Wall, AR = Action Reflection, TA = Target Achievement.

The most critical discrepancy appears in the Cross Wall (CW) metric. While Auto-Eval reports moderate wallcrossing rates (16.00%–26.00%), Human-Eval detects significantly higher violation rates (70.00%–100.00%)—a 3–5× increase. Notably, human evaluators identified wall-crossings in 100.00% of samples for DFS Hard, Wilson’s Medium, and Wilson’s Hard mazes, whereas Auto-Eval reported rates of only 16.00%–19.00%. This confirms that the VLM evaluator systematically fails to capture transient wall-crossing events, likely due to insufficient temporal resolution or frame-dropping when processing fast-moving agents. These Cross Wall failures propagate directly to the Overall Score, which mandates zero wall-crossings for success. Consequently, while Auto-Eval suggests moderate competence (39.00%–52.00%), Human-Eval reveals that true performance is near zero (0.00%–20.00%). In four of the six configurations, Veo-3 achieved zero successful completions according to human review. This indicates that the automated metrics overestimate the model’s maze-solving success by a factor of 2–5×.

Conversely, the Action Reflection (AR) metric exhibits an inverse pattern. Human evaluators assigned substantially higher scores (40.00%–90.00%) than the VLM (0.00%–3.00%). This suggests that humans are sensitive to subtle exploratory behaviors—such as hesitations, micro-adjustments, or partial backtracking—that the VLM fails to classify as meaningful reflection. While these behaviors do not constitute full multi-path exploration, they indicate that the model engages in implicit trajectory reasoning that automated metrics miss.

Finally, the Maze Changed (MC) metric shows a narrower gap, with Human-Eval reporting slightly higher rates (10.00%–40.00%) than Auto-Eval (0.00%–16.00%). This suggests that structural changes to maze geometry are more visually salient to VLMs than the fleeting motion artifacts involved in wall-crossings.

In summary, these findings expose a critical limitation in VLM-based evaluation for temporally dense tasks: automated evaluators systematically overestimate success by missing transient but fatal errors. Human

calibration reveals that while Veo-3 demonstrates emerging spatial understanding, its operational reliability is substantially lower than automated metrics imply.

- 5.5.3 Limitations and Insights from VLM-Based Evaluation Our evaluation pipeline relies on a VLM-based automated evaluator (AutoEval), instantiated as Gemini-

- 2.5-Pro. While AutoEval delivers consistent ratings for Easy and Medium tasks, its reliability degrades significantly at the Hard difficulty level. In these complex scenarios, high-velocity agent movements challenge the VLM’s temporal resolution. The evaluator frequently “drops frames,” missing transient but critical violations—specifically Cross Wall and Maze Changed errors—that occur within single frames. Consequently, AutoEval systematically overestimates performance on harder tasks. To calibrate this blind spot, Human Evaluation (HumanEval) serves as an essential baseline; we argue that presenting HumanEval scores alongside AutoEval is strictly necessary for Hard settings to provide a truthful performance picture.

Beyond validation reliability, our analysis highlights the critical concept of evaluability. Models that explicitly visualize reasoning—such as Nano-Banana, which renders its intended trajectory as a static blue path—fundamentally transform the evaluation task. This “plan visualization” converts a challenging temporal verification problem (tracking frame-by-frame collisions) into a straightforward spatial comparison (checking static path validity), thereby enhancing transparency and trustworthiness.

However, this benefit relies on structural faithfulness. When prompted to generate similar trajectory visualizations, Veo-3 frequently produced erratic, hallucinated blue curves covering area the agent never visited. Far from aiding interpretation, these hallucinations obscured the model’s actual reasoning and actively confused the Gemini-2.5-Pro evaluator, further degrading AutoEval accuracy. This underscores a key insight: while explicit visual reasoning can improve evaluability, it is only beneficial when the visualized artifacts remain grounded in the physical reality of the environment.

### 6 Sudoku

We introduce the Sudoku task to evaluate a model’s core abilities in Constraint Satisfaction and Logical Reasoning. This task directly probes a model’s capacity for logical inference-its ability to derive valid conclusions under a structured set of logical rules. In Sudoku, the model must internalize the underlying constraints that govern valid solutions, ensuring that each symbol appears exactly once in every row, column, and subgrid. Moreover, the task explicitly tests deductive reasoning: the step-by-step application of these constraints to infer the only logically consistent values for each empty cell. Successful completion thus requires the model to integrate global structural understanding with local deductive consistency, producing a fully valid and complete grid (Seely et al., 2025).

- 6.1 Hard-Level Control

To build a diverse and systematically controlled evaluation set, we use an open-source Sudoku generation library (Seely et al., 2025) to create puzzles spanning a range of structural and reasoning complexities. We vary task parameters along two main axes:

- • Grid Size (2 levels): We generate puzzles in two standard configurations—4×4 (digits 1–4) and 9×9 (digits 1–9). The larger grid substantially increases combinatorial difficulty and requires deeper multi-step logical inference.
- • Puzzle Difficulty (3 levels): For each grid size, we control difficulty by varying the number of initial clues (pre-filled digits). The three levels—Easy (many clues), Medium, and Hard (few clues while ensuring a unique solution)—modulate the search space and constraint-satisfaction complexity.

For every difficulty level (Easy, Medium, Hard), we generate 100 puzzles: 50 for the 4×4 grid and 50 for the 9×9 grid. This results in a balanced evaluation set of 300 Sudoku puzzles spanning a controlled spectrum of structural sizes and reasoning challenges. Figure 6 provides representative puzzles and solutions for each difficulty level and grid size, with each puzzle displayed directly above its corresponding solution.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

(a) 4×4 (Easy) (b) 4×4 (Easy) Solution (c) 9×9 (Easy) (d) 9×9 (Easy) Solution

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

(e) 4×4 (Medium) (f) 4×4 (Medium) Solu-

tion

[Figure 90]

[Figure 91]

(g) 9×9 (Medium) (h) 9×9 (Medium) Solu-

tion

[Figure 92]

[Figure 93]

(i) 4×4 (Hard) (j) 4×4 (Hard) Solution (k) 9×9 (Hard) (l) 9×9 (Hard) Solution

- Figure 6 A 3×4 grid showing Sudoku puzzles and their solutions across Easy, Medium, and Hard difficulty levels for both 4×4 and 9×9 grid sizes, arranged so each puzzle is immediately followed by its solution.

- 6.2 Evaluation and Metrics

We evaluate both video and image outputs using a Vision-Language Model (VLM)–based evaluator, Gemini2.5-Pro (Comanici et al., 2025). The evaluator receives three inputs: (i) the generated video or image, (ii) the ground-truth solved Sudoku grid, and (iii) a structured evaluation prompt (with modality-specific variants). Using these inputs, the VLM assesses whether the model produces a valid Sudoku solution and identifies failure modes related to rule consistency, clue preservation, and reasoning behavior.

From the evaluator’s structured outputs, we derive one primary success metric and four fine-grained metrics:

- • Clues Changed (Failure Mode): (i) Video: 1 if any original digits (“clues”) are modified, removed, or displaced in any frame; 0 if all clues remain intact across the entire sequence. (ii) Image: 1 if any given clue differs from the original puzzle image; 0 otherwise.
- • Constraints Violation (Failure Mode): The fraction of Sudoku constraints correctly satisfied in the final output (rows, columns, and subgrids). A value of 1 indicates full rule compliance; 0 indicates violation.
- • Completion Accuracy: The fraction of correctly filled originally empty cells, computed by comparing the model’s final output against the ground-truth solution.
- • Action Reflection: (i) Video: 1 if the sequence shows interpretable step-by-step reasoning (e.g., gradual cell updates without overwriting earlier entries); 0 if digits appear simultaneously or in an erratic order. (ii) Image: Not applicable.
- • Overall Score: 1 only if Clues Changed=0 AND Constraints Violation=0 AND Completion Accuracy=1; 0

#### otherwise.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Easy (4x4) Example Success Case: Sudoku puzzles solved perfectly by NanoBanana, with all clues preserved

Failure Case 2: The Sudoku solution generated by Nano-Banana contains repeated digits in the

Failure Case 1: The Sudoku puzzle

is solved incorrectly by NanoBanana, exhibiting repeated digits

| |
|---|

and no constraint violations.

lower-left box and a missing digit

within the lower-right box.

in the upper-right box, resulting in an invalid solution.

Clues Changed = 0

Clues Changed = 0 Constraints Violation = 1

Constraints Violation = 0 Completion Accuracy = 1

Clues Changed = 0

Completion Accuracy = 0

Constraints Violation = 0 Completion Accuracy = 0

- Figure 7 Case Study: Success and failure cases generated by Nano-Banana.

- 6.3 Case Study

- Figure 7 and Figure 8 illustrate several solution trajectories generated by Nano-Banana (image generative model) and Veo-3 (video generative model) for the same 4×4 easy Sudoku puzzle. In the successful case, Nano-Banana solves the puzzle perfectly—preserving all initial clues, following valid reasoning steps, and reaching the correct final configuration without any constraint violations. However, when the model fails, the error patterns align closely with our diagnostic evaluation metrics: missing digits, repeated numbers within a row or box, constraint violations across frames, and changes to the original clues. These failures often arise despite our detailed, instruction-focused generation prompts, suggesting that Nano-Banana still lacks the deeper abstract reasoning capacity required for consistent multi-step logical problem solving. Although Veo-3 performs worse overall in producing fully correct solutions, its video outputs reveal a set of systematic and interpretable reasoning behaviors: (1) Veo-3 demonstrates a strong positional bias: it almost always initiates its reasoning from the top-left box of the grid. Humans, in contrast, typically adapt their solving order based on puzzle structure, difficulty, or available constraints—suggesting that Veo-3 relies more on a fixed procedural heuristic than on dynamic reasoning. (2) Veo-3 frequently engages in self-reflective edits during the generation process. These edits occasionally help the model avoid violations (e.g., at second 3, where it changes the (1,1) cell from “4” to “2” to resolve a conflict), but they can also be detrimental. At second 2, for example, Veo-3 overwrites a correct digit, changing the (0,0) entry from “2” to “3,” instantly invalidating the entire solution. This behavior illustrates a broader challenge in generative reasoning models: self-reflection without grounded logical consistency can introduce more instability than benefit. (3) Veo-3 exhibits a characteristic temporal drift pattern. Its early frames often remain stable and respect the given clues, but as generation progresses, the model increasingly modifies clues, introduces inconsistencies, or oscillates between alternative partial solutions. This suggests that the model lacks a persistent internal representation of constraints, leading to reasoning degradation over time, even when the prompt explicitly prohibits altering clues.

Overall, these case studies highlight the gapbetweencurrentgenerativemodelsandtrueabstractreasoningability. Both Nano-Banana and Veo-3 demonstrate surface-level competence—filling cells, correcting mistakes, or imitating stepwise processes—but neither maintains a robust, constraint-aware reasoning trajectory throughout the entire solution. These observations reinforce the importance of MMGR’s diagnostic metrics: beyond assessing whether the final answer is correct, evaluating how the model reasons is essential for understanding the strengths and limitations of generative reasoning models.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

- 0
- 1
- 2
- 3

constraint self-reflection violation

self-reflection

constraint violation

constraint

violation

self-reflection

constraint

violation

clues changed

constraint violation

constraint

violation

[Figure 105]

Last Frame: As a result, the model

fails to solve the puzzle, producing a

final output with missing digits, repeated violations, and altered clues

across frames.

[Figure 106]

Figure 8 Case Study: Failure cases generated by Veo-3 with frame-wise analysis highlighting three key behaviors: positional bias, self-reflective edits, and temporal drift.

6.4 Evaluation Results

Key Finding: Image Models Outperform Video Models in Symbolic Reasoning

Image generation models significantly outperform video models on Sudoku tasks, as video models suffer from severe temporal instability and a lack of global symbolic reasoning. While video models exhibit high action reflection (making step-by-step edits), they fail to maintain logical consistency, leading to frequent constraint violations and clue changes. Crucially, human evaluation reveals 0% overall success for Veo-3 across all conditions, highlighting that current video models strictly fail at complex symbolic reasoning and that VLM-based auto-evaluation tends to overestimate model performance.

- 6.4.1 VLM-Based Evaluation

- Table 6 reveals a clear and persistent performance divide between image and video generative models across grid sizes and difficulty levels. Image models (Nano-Banana, Nano-Banana Pro, GPT-4o-image, Qwen-image) consistently achieve higher completion accuracy and overall scores, reflecting stronger symbolic consistency when directly producing completed Sudoku grids. In contrast, video models (Veo-3, Sora-2, Wan-2.2) frequently generate visually coherent step-by-step edits—high action reflection—yet still suffer from widespread clue changes, constraint violations, and low end-to-end correctness.

- 4×4 Performance. Even in the simplest setting, video models struggle. On Easy puzzles, the best video performer, Veo-3, reaches an overall accuracy of only 11.38%, while Sora-2 and Wan-2.2 effectively fail (0% and 2%). Image models perform dramatically better: Nano-Banana achieves 66.25%, GPT-4o-image 61.22%, and Nano-Banana Pro 56.12%. This gap widens at Medium and Hard levels, where video models rarely produce valid solutions (overall ≤ 9.70%). Although Veo-3 maintains high action reflection (92%–96%), its completion accuracy remains low (30%–38%), and clue-change rates (24%–30%) indicate persistent temporal drift.

Second 3: Veo-3 demonstrates selfreflection by revising the filled digit from “4” to “2” to prevent a box-level violation;

Second 1: Veo-3 consistently begins its solution by filling

Second 2: Veo-3 exhibits self-reflection

0 1 2 3

by changing its initial entry from “2” to “3,” but then incorrectly fills the (1,1)

Easy (4x4) Example

the top-left cell (0,0) with

however, this adjustment still violates

the digit “2.”

cell in the box with “4,” violating constraints because another “4” already

constraints, as a “2” is already present in the same row.

exists in the same box.

[Figure 107]

[Figure 108]

[Figure 109]

Seconds 7: Veo-3 fills the

Second 6: Veo-3 exhibits selfreflection by changing the filled digit at (2,1) from “2” to “1” to avoid a

Seconds 4-5: Veo-3 overlooks

(3,1) and (3,2) cells with two

the violations in the second column and fills cell (2,1)

“2” digits – this step creates a lot of constraints

column conflict, but this introduces a

with the digit “2,” which breaks the constraints

violation.

new violation because another “1” already exists in the same row. It also

because another “2” already exists in the same column.

replaces the clue at (3,0) with “1,” creating an additional box-level

conflict and altering a provided clue.

#### 9×9 Performance. On full-scale Sudoku, where global structural consistency is essential, video models break down almost entirely. Overall scores remain below 5% for most models. Sora-2’s occasional peak at 7.14% on

##### Table6 Quantitative results for the Sudoku task. We evaluate video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-Banana, Nano-Banana Pro, GPT-4o-image, and Qwen-image) on two grid sizes (4×4, 9×9) and three difficulty levels (Easy, Medium, and Hard). Because image outputs do not support frame-by-frame reasoning, action-reflection–based metrics are omitted for image generative models (marked with “N/A”). The highest overall scores in each setting are highlighted in bold.

Fine-grained Metrics Primary Metric Model Clues Changed ↓ Constraints Violation ↓ Completion Accuracy ↑ Action Reflection ↑ Overall ↑ Grid Size: 4×4

###### Level: Easy

Video Models

###### Veo-3 27.60% 42.00% 37.21% 92.00% 11.38% Sora-2 100.00% 44.22% 25.78% 32.65% 0.00% Wan2.2 100.00% 35.67% 17.22% 4.67% 2.00%

Image Models

Nano-banana 18.40% 17.20% 67.48% N/A 66.25% Nano-banana Pro 35.25% 1.33% 93.38% N/A 56.12% GPT-4o-image 23.83% 19.65% 73.14% N/A 61.22% Qwen-image 83.50% 61.50% 46.27% N/A 6.67%

Level: Medium

Video Models

###### Veo-3 24.40% 43.53% 37.61% 95.60% 9.70% Sora-2 100.00% 45.24% 21.71% 61.90% 0.00% Wan2.2 100.00% 36.67% 14.36% 6.00% 1.33%

Image Models

Nano-banana 0.63% 21.30% 52.14% N/A 51.78% Nano-banana Pro 34.81% 0.50% 91.86% N/A 56.75% GPT-4o-image 31.68% 24.88% 56.39% N/A 46.28% Qwen-image 83.50% 61.50% 46.27% N/A 6.67%

Level: Hard

Video Models

###### Veo-3 30.12% 44.88% 30.50% 95.18% 8.71% Sora-2 100.00% 41.85% 20.03% 60.87% 0.00% Wan2.2 99.33% 38.17% 14.79% 6.00% 0.00%

Image Models

Nano-banana 24.40% 25.80% 46.21% N/A 42.45% Nano-banana Pro 40.53% 0.83% 89.50% N/A 57.38% GPT-4o-image 36.40% 24.37% 49.00% N/A 39.08% Qwen-image 83.50% 61.50% 46.27% N/A 6.67%

Grid Size: 9×9 Level: Easy

Video Models

###### Veo-3 31.60% 39.60% 15.47% 70.00% 3.18% Sora-2 95.74% 54.85% 8.47% 34.04% 4.26% Wan2.2 87.00% 44.59% 18.03% 8.00% 0.00%

Image Models

Nano-banana 10.50% 31.66% 19.87% N/A 28.80% Nano-banana Pro 19.09% 33.67% 31.52% N/A 39.28% GPT-4o-image 70.72% 59.67% 19.83% N/A 12.44% Qwen-image 28.33% 7.88% 73.12% N/A 18.08%

Level: Medium

Video Models

###### Veo-3 34.00% 40.59% 13.66% 72.00% 2.77% Sora-2 100.00% 54.96% 9.43% 30.00% 0.00% Wan2.2 85.33% 46.44% 16.13% 2.00% 0.00%

Image Models

Nano-banana 2.00% 35.54% 36.88% N/A 24.15% Nano-banana Pro 14.91% 28.40% 26.19% N/A 33.99% GPT-4o-image 71.37% 58.53% 17.92% N/A 11.43% Qwen-image 28.33% 7.88% 73.12% N/A 18.08%

Level: Hard

Video Models

###### Veo-3 31.20% 43.01% 13.45% 70.40% 2.57% Sora-2 92.86% 59.79% 8.65% 35.71% 7.14% Wan2.2 91.00% 48.07% 16.86% 3.00% 1.00%

Image Models

###### Nano-banana 2.40% 38.16% 17.03% N/A 19.94% Nano-banana Pro 14.75% 41.36% 22.52% N/A 30.86% GPT-4o-image 71.58% 57.68% 15.74% N/A 10.00% Qwen-image 28.33% 7.88% 73.12% N/A 18.08%

Hard puzzles stems from anomalously low clue-change instances rather than genuine logical success. High constraint-violation rates (39%–60%) and low completion accuracy (8%–18%) further highlight compounding temporal errors that worsen with puzzle size.

Image models, by contrast, remain substantially more stable. Nano-Banana Pro delivers the strongest overall results across difficulties—39.28%, 33.99%, and 30.86%—while standard Nano-Banana remains competitive (19.94%–28.80%). Qwen-image is a mixed case: it frequently alters clues on 4×4 puzzles but achieves surprisingly strong completion accuracy on 9×9 (73.12%). However, its solutions often contain subtle structural inconsistencies, limiting overall correctness under our strict validator.

Across all conditions, video models exhibit a fundamental weakness: despite producing locally plausible temporal edits, they lack the global, long-horizon reasoning required to satisfy Sudoku’s symbolic constraints. Their sequential generation process introduces temporal instability—digit drift, unintended clue overwrites, and cumulative constraint violations—that image models avoid by generating a single, static grid. The persistent performance gap across both grid sizes and all difficulty levels underscores a deeper architectural mismatch between current video-generation paradigms and tasks requiring coherent, symbolic logical reasoning.

- 6.4.2 Human Evaluation

To establish ground-truth performance estimates and validate the reliability of our VLM-based evaluator, we conducted a human evaluation on a subset of Veo-3’s generated videos. Table 7 presents a side-by-side comparison of Auto-Eval versus Human-Eval across all grid sizes and difficulty levels.

- Table 7 Comparison of Auto-Eval (VLM-based) and Human-Eval results for Veo-3 on the Sudoku task across two grid sizes (4×4, 9×9) across three difficulty levels (Easy, Medium, and Hard).

Fine-grained Metrics Primary Metric Grid Eval Type Clues Changed ↓ Constraints Violation ↓ Completion Accuracy ↑ Action Reflection ↑ Overall ↑ Level: Easy 4×4 Auto-Eval 27.60% 42.00% 37.21% 92.00% 11.38%

Human-Eval 10.00% 10.00% 17.50% 90.00% 0.00% 9×9 Auto-Eval 31.60% 39.60% 15.47% 70.00% 3.18% Human-Eval 30.00% 7.50% 0.00% 100.00% 0.00%

Level: Medium 4×4 Auto-Eval 24.40% 43.53% 37.61% 95.60% 9.70%

Human-Eval 30.00% 5.00% 12.50% 100.00% 0.00% 9×9 Auto-Eval 34.00% 40.59% 13.66% 72.00% 2.77% Human-Eval 20.00% 0.00% 2.50% 80.00% 0.00%

Level: Hard 4×4 Auto-Eval 30.12% 44.88% 30.50% 95.18% 8.71%

Human-Eval 50.00% 10.00% 17.50% 100.00% 0.00% 9×9 Auto-Eval 31.20% 43.01% 13.45% 70.40% 2.57% Human-Eval 10.00% 20.00% 0.00% 100.00% 0.00%

Clues Changed. The clues-changed metric exhibits a mixed pattern across conditions. For 4×4 Easy puzzles, Auto-Eval reports a higher rate (28%) than Human-Eval (10%), yet this reverses on Hard puzzles where Human-Eval detects more clue modifications (50% vs. 30%). On 9×9 grids, the discrepancy is smaller but inconsistent: Human-Eval reports lower clue-change rates on Medium (20% vs. 34%) and Hard (10% vs. 31%) levels, but comparable rates on Easy (30% vs. 32%). These divergent patterns suggest that VLM and human annotators apply different detection criteria—the VLM may flag subtle visual perturbations as clue changes, while humans focus on semantically meaningful digit alterations.

Constraints Violation. The most striking discrepancy emerges in constraint violation detection. Human evaluators consistently assign substantially lower constraint violation scores across all conditions: 10% vs. 42% on 4×4 Easy, 5% vs. 44% on 4×4 Medium, 10% vs. 45% on 4×4 Hard. For 9×9 puzzles, the gap widens further—Human-Eval yields 8% (Easy), 0% (Medium), and 20% (Hard), compared to Auto-Eval’s 40%, 41%, and 43% respectively. Notably, human annotators report zero constraint violations on 9×9 Medium puzzles,

whereas Auto-Eval detects violations in over 40% of outputs. This suggests the VLM-based evaluator may over-detect violations by misinterpreting visual artifacts, blurry digits, or frame inconsistencies as rule breaches.

Completion Accuracy. Completion accuracy is substantially lower under human evaluation across all conditions. On 4×4 grids, Auto-Eval reports accuracy of 37% (Easy), 38% (Medium), and 31% (Hard), while Human-Eval yields only 18%, 13%, and 18% respectively—roughly half the VLM’s estimates. The disparity is even more pronounced on 9×9 puzzles: Auto-Eval reports 15% (Easy), 14% (Medium), and 13% (Hard), but Human-Eval drops to 0% on both Easy and Hard levels, with only 3% on Medium. This near-zero completion accuracy under human judgment indicates that human annotators apply stricter criteria for recognizing correctly filled digits, likely requiring clearer visual rendering and unambiguous digit shapes that Veo-3 fails to consistently produce at the 9×9 scale.

Action Reflection. In contrast to other metrics, action reflection scores show reasonable agreement between evaluation methods. Both evaluators recognize that Veo-3 exhibits step-by-step reasoning behavior, with Auto-Eval reporting 92%–96% on 4×4 and 70%–72% on 9×9, while Human-Eval assigns 90%–100% on 4×4 and 80%–100% on 9×9. Human annotators even rate action reflection slightly higher on several conditions, reaching perfect scores (100%) on 4×4 Medium, 4×4 Hard, 9×9 Easy, and 9×9 Hard. This consensus confirms that Veo-3 does produce visually interpretable sequential edits—the model’s failure lies not in lacking a reasoning process, but in lacking reasoning correctness.

Overall Score. Most strikingly, no Veo-3 output achieved an overall success score under human evaluation—all conditions yield 0%, compared to Auto-Eval’s modest but non-zero scores ranging from 3% (9×9 Easy) to 11% (4×4 Easy). This complete failure under human judgment, despite the model producing visually coherent step-by-step animations, underscores that even outputs deemed partially successful by the VLM fail to meet the stringent correctness standards required for valid Sudoku solutions.

Key Insights. These findings highlight two key insights: (1) VLM-based evaluation tends to be more lenient, potentially overestimating model performance on structured reasoning tasks—particularly by over-detecting constraint violations while being more generous on completion accuracy; and (2) human evaluation remains essential for validating generative model outputs in tasks requiring strict logical correctness, as the metrics that matter most (completion accuracy and overall success) show the largest divergence between evaluation methods. The consistent zero overall scores under human evaluation reinforce our main conclusion: current video generative models lack the robust constraint-satisfaction capabilities needed for reliable Sudoku solving.

### 7 ARC-AGI

We evaluate models on the ARC-AGItask (Abstraction and Reasoning Corpus for Artificial General Intelligence), a benchmark designed to measure Abstract Reasoning, Pattern Recognition, and Rule Induction (Chollet, 2019). ARC-AGI probes a model’s ability to infer latent transformation rules from a small set of input–output examples and to apply those rules to unseen test cases—an ability central to fluid intelligence. Unlike tasks with explicit instructions or predefined rule sets, ARC-AGI demands that the model autonomously identify and generalize the underlying visual and structural principles governing each puzzle. As a result, the task jointly tests 2D spatial reasoning (interpreting grid-based patterns) and logical reasoning (deducing and executing abstract transformations).

- 7.1 Hard-Level Control

To construct a diverse and controllable evaluation suite, we build on the open-source ARC-AGI benchmark (Chollet, 2019). Our final dataset comprises 456 tasks drawn from two benchmark versions:

- • ARC-AGI v1 (381 tasks): The publicly released training set from the original benchmark, encompassing a wide variety of pattern-transformation and abstract reasoning problems.
- • ARC-AGI v2 (75 tasks): Newly added tasks that introduce novel pattern families and higher structural complexity, expanding the benchmark’s coverage.

This combined set provides broad coverage of transformation types while introducing sufficient novelty and difficulty for rigorous model evaluation. All tasks undergo manual curation to ensure clean pattern design, consistent formatting, and unambiguous transformation rules. Collectively, the benchmark spans a wide spectrum of transformation categories—including symmetry, rotation, scaling, color manipulation, and object-level reasoning—capturing the core abstractions and reasoning skills fundamental to ARC-style tasks.2

- 7.1.1 Two-Level Classification System To enable fine-grained analysis of model capabilities, we classify all cases along two dimensions:

- Level 1: Shape Consistency Classification. We categorize each case based on whether the input and output grids maintain the same spatial structure:

- • Match (316 cases): The output grid has the same dimensions as the input grid. Transformations occur “in-place” through color changes, pattern filling, or local modifications. Examples include color replacement, symmetry completion, and pattern extension within fixed boundaries.
- • Mismatch (140 cases): The output grid dimensions differ from the input. These cases require spatial restructuring, such as cropping, extraction of sub-patterns, grid concatenation, or shape reconstruction. These are generally more challenging as they demand explicit understanding of spatial relationships beyond simple transformations.

- Level 2: Quantitative Difficulty Classification. Within each shape-consistency category, we further assign each case to one of three difficulty levels—Easy, Medium, or Hard—based on five quantitative grid-level features:

- 1. Grid Size (minimum side length): < 8 cells → score 0; 8-15 cells → score 1; ≥ 16 cells → score 2.
- 2. Color Count (distinct colors in input): ≤ 3 → score 0; 4-6 → score 1; ≥ 7 → score 2.
- 3. Object Count (distinct connected components): ≤ 4 → score 0; 5-10 → score 1; > 10 → score 2.
- 4. Occupancy Ratio (non-background cells / total cells): ≤ 0.25 → score 0; (0.25,0.55] → score 1; > 0.55

→ score 2.

- 5. ∆IO (grid change ratio, for Match cases only): ≤ 0.2 → score 0; (0.2,0.5] → score 1; > 0.5 → score 2. For Mismatch cases, this feature is not applicable.

A case’s overall difficulty is determined by summing the applicable feature scores.For Match cases (5 features): Easy ≤ 3, Medium 4–6, Hard ≥ 7. For Mismatch cases (4 features): Easy ≤ 2, Medium 3–4, Hard ≥ 5.

- Table 8 reports the full distribution across categories. Figure 9 and Figure 10 present some selected examples.

Table 8 Distribution of 456 ARC-AGI cases across shape consistency and difficulty levels, separated by v1 and v2. Percentages indicate the proportion within each shape consistency group for the corresponding benchmark version.

Version Shape Consistency Easy Medium Hard Total

###### Match 102 (38.8%) 124 (47.1%) 37 (14.1%) 263 Mismatch 34 (28.8%) 57 (48.3%) 27 (22.9%) 118

V1

Total 136 181 64 381

###### Match 1 (1.9%) 27 (50.9%) 25 (47.2%) 53 Mismatch / 7 (31.8%) 15 (68.2%) 22

V2

Total 1 34 40 75

###### Overall Match 103 (31.0%) 151 (45.4%) 62 (23.6%) 316 Mismatch 34 (25.0%) 64 (45.7%) 42 (29.3%) 140

Total 133 (29.2%) 205 (45.0%) 118 (25.9%) 456

- 7.2 Evaluation and Metrics

We evaluate both video and image outputs using a Vision-Language Model (VLM)–based evaluator, Gemini2.5 Pro (Comanici et al., 2025). The evaluator takes four inputs: (i) the demonstration examples, (ii) the

2Additional examples and visualizations are available in the official repositories: https://github.com/fchollet/ARC-AGI , https://github.com/michaelhodel/re-arc , and the automated generation toolkit https://github.com/google/ARC-GEN.

###### ARC-AGI V1

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Match, Easy Match, Medium Match, Hard Mismatch, Easy Mismatch, Medium Mismatch, Hard

- Figure 9 Selected examples from ARC-AGI v1, illustrating both Match and Mismatch tasks across three difficulty levels: Easy, Medium, and Hard.

ARC-AGI V2

Match, Easy Match, Medium Match, Hard Mismatch, Medium Mismatch, Hard

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

- Figure 10 Selected examples from ARC-AGI v2, illustrating both Match and Mismatch tasks across three difficulty levels: Easy, Medium, and Hard.

test input, (iii) the ground-truth output, and (iv) the generated video or image. Using these inputs, the VLM determines whether the predicted transformation is correct and identifies errors in pattern recognition, structural consistency, and rule application.

From the evaluator’s structured responses, we compute one primary correctness metric and three fine-grained diagnostic metrics:

- • Pattern Recognition: 1 if the evaluator confirms that the model successfully identifies the transformation pattern from the demonstrations; 0 otherwise.
- • Grid Integrity: 1 if the generated output preserves the correct grid dimensions and structural layout; 0 if the grid is distorted or misaligned.
- • Color Accuracy: 1 if all colors are applied correctly according to the transformation rule; 0 otherwise.
- • Valid Solution (Primary Metric): 1 only if the generated output exactly matches the ground-truth solution; 0 otherwise.

##### Table9 Overall quantitative results for the ARC-AGIv1 task (381 cases). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image).

Fine-grained Metrics Primary Metric Model Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Video Models

Veo-3 17.32% 32.98% 8.22% 5.16% Sora-2 71.99% 94.58% 36.75% 20.18% Wan-2.2 0.61% 13.04% 0.17% 0.17%

Image Models Nano-banana 28.42% 55.79% 12.63% 9.21% Nano-banana Pro 61.98% 84.73% 40.42% 30.54% GPT-4o-image 1.05% 10.24% 0.52% 0.00% Qwen-image 1.31% 4.46% 0.52% 0.52%

- 7.3 Case Study

The case studies presented in Figure 11a and Figure 11b highlight a significant failure mode in video generation models applied to abstract reasoning tasks: the inability to maintain temporal consistency for static information. In both instances, the model fails to distinguish between the invariant problem context (the “EXAMPLES” E1-E4) and the dynamic solution generation (the “TEST” T1). As the video progresses from Frame 1 to Frame

- 4, the demonstration examples—which should remain strictly fixed to define the logic of the task—suffer from severe hallucinations, including unintended color shifts, pattern deformations, and progressive structural degradation (such as the complete erasure of grid contents in Figure 11). This “drift” suggests that the model treats the entire visual field as a mutable video sequence rather than respecting the logical constraints of the prompt, ultimately undermining the reasoning process by destabilizing the very ground truth required to solve the puzzle.

- 7.4 Evaluation Results

Key Finding: The Abstract Reasoning Divide and Temporal Instability

Our evaluation reveals two critical insights about abstract visual reasoning. First, a fundamental modality gap: Nano-banana Pro achieves 30.54% accuracy on ARC-AGI v1, establishing clear stateof-the-art performance and outperforming the best video model (Sora-2: 20.18%) by a significant margin. Second, a temporal consistency failure: video models struggle to maintain static demonstration examples, leading to “context drift” where the problem definition itself becomes corrupted during generation. Most critically, human evaluation exposes an evaluation reliability gap—while VLM-based metrics report 4–5% valid solutions for Veo-3, human annotators find 0.00% across all 98 evaluated cases, revealing that current video models cannot reliably execute abstract transformations despite occasionally producing visually plausible outputs.

- 7.4.1 VLM-Based Evaluation

Overall Performance Patterns. The quantitative results (Table 9 and Table 10) establish a clear performance hierarchy across the 456 ARC-AGI cases. Nano-banana Pro dominates with 30.54% overall accuracy on v1 and 30.36% on v2, demonstrating both superior reasoning capability and remarkable robustness to distribution shift. Among video models, Sora-2 emerges as the clear leader with 20.18% on v1—notably surpassing the base Nano-banana image model (9.21%)—yet collapses to 1.33% on the harder v2 dataset, exposing critical brittleness. Veo-3 maintains modest but stable performance (5.16% v1, 4.00% v2), while Wan-2.2 effectively fails (0.17% v1, 0.00% v2). At the bottom tier, GPT-4o-image achieves 0.00% across both versions despite non-zero Grid Integrity (10.24% on v1), and Qwen-image barely registers (0.52% v1, 1.33% v2).

The Modality Gap. The results reveal a fundamental divide between image and video generation paradigms

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Frame 1 Frame 2 Frame 3 Frame 4 Failure: Change Examples

- (a) Inconsistent demonstration examples across video frames. The model fails to maintain the static demonstration inputs (E1–E4), with colors and patterns changing between frames. This behavior reflects a critical failure to preserve the given problem context.

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Frame 1 Frame 2 Frame 3 Frame 4 Failure: Change Examples

- (b) Progressive transformation of example demonstrations across frames. The examples (E1–E4) undergo unintended color and pattern evolution from Frame 1 to Frame 4, indicating a lack of temporal consistency in preserving the demonstration context during solution generation.

- Figure 11 Failure cases generated by Veo-3 on ARC-AGI showing violation of static demonstration preservation. In both examples, the model fails to keep the input demonstrations unchanged through time, illustrating a key breakdown in temporal consistency for reasoning-based video generation.

##### Table 10 Overall quantitative results for the ARC-AGI v2 task (75 cases). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image).

Fine-grained Metrics Primary Metric Model Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Video Models

Veo-3 17.78% 31.11% 6.22% 4.00% Sora-2 4.00% 16.00% 1.33% 1.33% Wan-2.2 0.00% 5.78% 0.00% 0.00%

Image Models Nano-banana 18.67% 42.67% 8.00% 2.67% Nano-banana Pro 62.50% 83.93% 44.64% 30.36% GPT-4o-image 1.33% 2.67% 1.33% 0.00% Qwen-image 1.33% 5.33% 1.33% 1.33%

##### Table 11 Quantitative results for the ARC-AGI task across different difficulty levels (Easy, Medium, and Hard). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image).

Fine-grained Metrics Primary Metric Model Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Version: v1 & v2 Combined

###### Level: Easy

Video Models

Veo-3 18.98% 39.66% 10.46% 5.60% Sora-2 76.07% 89.74% 41.88% 22.22% Wan-2.2 0.00% 16.79% 0.00% 0.00%

Image Models Nano-banana 32.85% 64.23% 13.87% 10.22% Nano-banana Pro 62.60% 86.18% 43.90% 30.89% GPT-4o-image 0.73% 10.95% 1.46% 0.00% Qwen-image 1.46% 5.11% 0.73% 0.73%

Level: Medium

Video Models

Veo-3 17.36% 32.40% 7.29% 5.12% Sora-2 58.67% 84.18% 29.08% 16.33% Wan-2.2 0.78% 10.54% 0.31% 0.31%

Image Models Nano-banana 24.77% 51.87% 10.75% 7.01% Nano-banana Pro 62.98% 86.74% 40.33% 30.39% GPT-4o-image 0.47% 9.77% 0.00% 0.00% Qwen-image 1.40% 4.65% 0.47% 0.47%

Level: Hard

Video Models

Veo-3 15.38% 24.04% 5.77% 3.85% Sora-2 40.43% 59.57% 18.09% 10.64% Wan-2.2 0.64% 8.01% 0.00% 0.00%

Image Models Nano-banana 23.08% 43.27% 11.54% 7.69% Nano-banana Pro 59.30% 77.91% 38.37% 30.23% GPT-4o-image 2.88% 4.81% 0.96% 0.00% Qwen-image 0.96% 3.85% 0.96% 0.96%

(Table 9 and Table 10). Image models demonstrate superior abstract reasoning: Nano-banana Pro achieves 61.98% Pattern Recognition and 84.73% Grid Integrity, substantially outperforming the best video model Sora-2 (71.99% Pattern Recognition, 94.58% Grid Integrity). However, a striking anomaly emerges: Sora-2 achieves higher fine-grained scores but lower overall accuracy than Nano-banana Pro. This suggests that video models can perceive patterns and maintain structure but fail at the final integration step—translating partial

##### Table 12 Quantitative breakdown results (Match and Mismatch) for the ARC-AGI v1 task (381 cases). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image).

Fine-grained Metrics Primary Metric Model / Category Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Video Models

Veo-3

Match 17.74% 35.74% 9.51% 5.70% Mismatch 16.38% 26.84% 5.37% 3.95%

Sora-2

Match 67.29% 93.46% 35.05% 17.76% Mismatch 80.51% 96.61% 39.83% 24.58%

Wan-2.2

Match 0.51% 13.31% 0.00% 0.00% Mismatch 0.85% 12.43% 0.56% 0.56%

Image Models

Nano-banana Match 24.05% 53.82% 11.07% 8.40% Mismatch 38.14% 60.17% 16.10% 11.02%

Nano-banana Pro

Match 62.24% 89.63% 42.74% 31.54% Mismatch 61.29% 72.04% 34.41% 27.96%

GPT-4o-image

Match 0.38% 9.51% 0.76% 0.00% Mismatch 2.54% 11.86% 0.00% 0.00%

Qwen-image

Match 0.76% 4.56% 0.76% 0.76% Mismatch 2.54% 4.24% 0.00% 0.00%

understanding into correct solutions. The gap amplifies on v2: Nano-banana Pro maintains stable performance (30.36%), while Sora-2 collapses from 20.18% to 1.33%—a 93% relative decline—revealing that video models rely heavily on memorized patterns rather than generalizable reasoning.

Dataset Difficulty Progression. Performance stratification by difficulty level (Table 11) reveals distinct scaling behaviors across model types:

- • Nano-banana Pro (Robust Stability): Maintains remarkable consistency across Easy (30.89%), Medium (30.39%), and Hard (30.23%) cases. Grid Integrity shows only modest degradation (86.18% → 86.74% → 77.91%), confirming that this model has internalized abstract transformation rules rather than relying on surface-level pattern matching.
- • Sora-2 (Difficulty-Sensitive Collapse): Exhibits pronounced degradation from Easy (22.22%) to Medium (16.33%) to Hard (10.64%). Notably, Pattern Recognition drops from 76.07% to 40.43%, indicating that video models lose the ability to identify transformation rules as complexity increases.
- • Veo-3 (Consistent Underperformance): Remains stagnant around 5% across all difficulty levels, with Grid Integrity showing steady decline (39.66% → 32.40% → 24.04%). This suggests that Veo-3’s failures are systematic rather than difficulty-dependent.

Shape Consistency Analysis. Comparing Match cases (same input/output dimensions) against Mismatch cases (different dimensions) across ARC-AGI v1 and v2 reveals distinct model behaviors (Table 12 and Table 14):

- • The Inverse Pattern (v1 vs. v2): In v1, both Sora-2 and the base Nano-banana model perform better on Mismatch cases than Match cases (Sora-2: 24.58% vs 17.76%; Nano-banana: 11.02% vs 8.40%). This suggests these models may leverage visual restructuring effectively when dimensions shift. However, this advantage disappears in the more complex v2, where Sora-2 collapses to 0.00% on Mismatch cases.
- • Nano-banana Pro Stability: Demonstrates demonstrates the most robust handling of transformation types. In v1, it maintains high stability (31.54% Match vs 27.96% Mismatch). In v2, while it remains the top

- performer, it exhibits a clearer preference for Match cases (33.33%) over Mismatch (21.43%), indicating that increased task complexity reintroduces the expected difficulty penalty for dimension changes.
- • Model-Specific Difficulty Curves: Other models show varied sensitivity to grid consistency. Veo-3 displays mixed behavior—dropping slightly in v1 (5.70% to 3.95%) but surprisingly improving on Mismatch in v2 (3.77% to 4.55%), potentially due to low sample variance. Conversely, image models like Qwen-image consistently struggle with Mismatch tasks, collapsing to near-zero performance across both datasets.

Dimensionality-Difficulty Interaction. A granular intersectional analysis of difficulty and grid consistency (Table 13 and Table 15) exposes a paradox in video model behavior distinct from standard scaling laws:

- • The Sora-2 Anomaly (Hard Case Inversion): On v1 Hard cases, Sora-2 displays a highly anomalous inverted profile, achieving nearly 4× higher accuracy on Mismatch tasks (29.63%) compared to Match tasks (7.41%). Fine-grained metrics reveal this is not a structural failure—Grid Integrity remains identical (96.30%) across both categories. Instead, the collapse is driven by Color Accuracy, which plummets to 18.52% on Match compared to 44.44% on Mismatch. This implies video models struggle to execute complex pixel-level color transformations when constrained to a fixed canvas, but recover capability when generating new spatial structures.
- • Nano-banana Pro (Logical Scaling): In contrast, the leading image model exhibits expected difficulty scaling. It consistently performs better on Match (fixed-dimension) tasks than Mismatch (dynamicdimension) tasks across all difficulty levels (e.g., v1 Hard Match 37.14% vs. Hard Mismatch 27.27%), confirming a stable, dimension-agnostic reasoning core.
- • Brittleness of the Generative Advantage: Sora-2’s “generative advantage” on Mismatch tasks proves brittle. On the v2 benchmark, the anomaly vanishes entirely: Sora-2 collapses to 0.00% on Hard Mismatch cases, while Nano-banana Pro maintains robust capability (22.22%). This reinforces that while video models may occasionally exploit spatial restructuring, they lack the robust generalization of their image-based counterparts.

Metric Cascade and Bottlenecks. The four evaluation metrics form a consistent funnel where success becomes progressively more restrictive. For Nano-banana on v1: Grid Integrity (55.79%) → Pattern Recognition (28.42%) → Color Accuracy (12.63%) → Overall (9.21%). This cascade reveals that approximately 51% of structurally correct outputs achieve pattern recognition, 44% of those achieve color accuracy, and 73% of color-accurate outputs reach full correctness. Color Accuracy emerges as the critical bottleneck, representing only 20–25% of Grid Integrity performance across all models. This bottleneck is most severe for GPT-4o-image: despite achieving 10.24% Grid Integrity, it manages only 0.52% Color Accuracy, resulting in 0.00% overall success. The cascade pattern suggests that models can learn structural rules but struggle with precise color-based execution.

Version Comparison. The transition from v1 to v2 exposes model robustness to novel pattern families:

- • Dramatic Video Model Collapse: Sora-2 drops from 20.18% to 1.33% (93% decline), while Nano-banana drops from 9.21% to 2.67% (71% decline). Nano-banana Pro remains stable (30.54% → 30.36%).
- • Dataset Composition Effects: v2 skews harder, containing only 1 Easy Match case versus 102 in v1. The difficulty distribution shift disproportionately penalizes models that rely on memorized patterns.
- • Grid Integrity Preservation: Interestingly, Veo-3 maintains similar Grid Integrity across versions (32.98% v1, 31.11% v2) despite failing on the primary metric, suggesting it can preserve structure without understanding transformation semantics.

- 7.4.2 Human Evaluation

#### To establish ground-truth performance estimates and validate the reliability of our VLM-based evaluator, we conducted a human evaluation on a subset of 98 Veo-3 generated videos (60 from v1, 38 from v2). Human annotators assess the same four metrics: Pattern Recognition, Grid Integrity, Color Accuracy, and Valid Solution. Table 16 and Table 17 present side-by-side comparisons of Auto-Eval versus Human-Eval across Match and Mismatch subsets.

Pattern Recognition. Human and VLM evaluators show reasonable agreement on pattern recognition ability. On v1, humans report 18.33% versus Auto-Eval’s 16.80%—a modest 1.5 percentage point difference. On v2, the gap is slightly larger: 23.68% (Human) versus 22.67% (Auto). This consensus suggests that both evaluation methods can reliably detect whether a model has identified the underlying transformation rule, even when execution fails.

Grid Integrity. A substantial divergence emerges in structural assessment. Humans consistently rate Grid Integrity lower than the VLM evaluator: 23.33% versus 35.70% on v1 overall (12.4 pp gap), and 31.58% versus 41.33% on v2 overall (9.8 pp gap). This systematic discrepancy suggests that the VLM evaluator may be more tolerant of minor structural deformations—such as slight grid misalignments or cell boundary artifacts—that human annotators penalize. The gap is particularly pronounced for Match cases on v1 (26.67% Human vs 40.30% Auto), indicating that humans apply stricter standards when evaluating in-place transformations where structural preservation is expected.

Color Accuracy. Interestingly, the pattern reverses for color assessment. Humans rate Color Accuracy substantially higher than the VLM: 23.33% versus 7.61% on v1 (3.1× higher), and 26.32% versus 6.67% on v2 (3.9× higher). This suggests the VLM evaluator may be overly sensitive to minor color deviations—perhaps detecting subtle RGB differences that humans perceive as acceptable matches. The implication is significant: VLM-based Color Accuracy scores may systematically underestimate model performance on this metric.

Valid Solution Rate. The most critical finding is the complete failure under human evaluation: 0.00% Valid Solution rate across all 98 cases, compared to 4.72% (v1) and 2.67% (v2) from Auto-Eval. Despite achieving non-trivial scores on fine-grained metrics—with Pattern Recognition reaching 38.10% and Grid Integrity reaching 47.62% on v2 Match cases—no single generated video produces an exact match to the ground-truth solution. This reveals a fundamental gap between partial understanding and precise execution: video models can demonstrate awareness of transformation patterns and maintain structural consistency in nearly half of cases, yet consistently fail to translate this partial competence into pixel-perfect outputs.

Match versus Mismatch Performance. Human evaluation confirms and amplifies the performance gap observed in automatic evaluation. On v2, Match cases achieve 38.10% Pattern Recognition compared to only 5.88% for Mismatch cases—a 6.5× difference. Grid Integrity shows an even more pronounced gap: 47.62% for Match versus 11.76% for Mismatch (4.0×). On v1, the gaps are smaller but consistent: Match cases achieve 23.33% Pattern Recognition versus 13.33% for Mismatch (1.75×). These results confirm that video generation models fundamentally struggle with shape-changing transformations, where producing outputs with different dimensions from inputs poses a substantially greater challenge than in-place modifications.

KeyInsights. The human evaluation reveals two critical findings: (1) VLM-basedevaluationtendstooverestimate overall success while being overly strict on Color Accuracy and lenient on Grid Integrity—a pattern that may mask true model capabilities and limitations; and (2) Human evaluation remains essential for validating generative model outputs in tasks requiring strict logical correctness, as the complete 0.00% Valid Solution rate under human judgment starkly contrasts with the non-zero Auto-Eval scores. These findings underscore that current video generation models lack the robust constraint-satisfaction capabilities needed for reliable abstract reasoning.

- 7.4.3 Key Findings and Implications

Three critical insights emerge from the ARC-AGI evaluation, revealing fundamental limitations in how current generative models approach abstract visual reasoning.

- 1. The Temporal Consistency Barrier. A defining failure mode of video generation on ARC-AGI is the inability to maintain static context. As illustrated in Figure 11a and Figure 11b, video models fail to distinguish between invariant problem context (the demonstration examples E1–E4) and the dynamic solution generation (test output T1). Demonstration examples—which should remain strictly fixed to define the transformation logic—undergo progressive hallucination: color shifts, pattern deformations, and structural degradation across frames. This “context drift” effectively corrupts the problem definition itself, undermining any chance of correct solution generation. The implication is clear: current video architectures treat the entire visual field

- as a mutable sequence rather than respecting logical invariants, making them fundamentally unsuited for reasoning tasks that require stable reference points.
- 2. The Perception-Execution Gap. The metric cascade analysis reveals a systematic disconnect between pattern perception and precise execution. Models frequently achieve reasonable Pattern Recognition (up to 71.99% for Sora-2) and Grid Integrity (up to 94.58%) scores, yet fail catastrophically on Overall accuracy. This suggests that current generative models can see the transformation pattern but cannot execute it correctly. The bottleneck consistently occurs at Color Accuracy, where performance drops to 20–25% of Grid Integrity levels. This perception-execution gap implies that models may learn to recognize visual patterns without internalizing the precise symbolic rules governing color assignment—a fundamental limitation for tasks requiring exact constraint satisfaction.
- 3. The Robustness-Memorization Trade-off. The dramatic performance collapse from v1 to v2—particularly Sora-2’s 93% relative decline—exposes a critical reliance on memorized patterns rather than generalizable reasoning. In contrast, Nano-banana Pro’s stability across versions (30.54% → 30.36%) demonstrates that image-based generation can achieve robust abstract reasoning. This dichotomy suggests that video models may be overfitting to temporal motion patterns in training data rather than learning the underlying transformation logic. For ARC-AGI specifically, where novel pattern families are explicitly designed to defeat memorization, this limitation is fatal. The finding has broader implications: video generation architectures may require fundamentally different training objectives or inductive biases to support genuine rule-based reasoning rather than sophisticated pattern matching.

##### Table 13 Quantitative breakdown results for the ARC-AGI v1 task (381 cases) across different difficulty level (Easy, Medium, and Hard). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image).

Fine-grained Metrics Primary Metric Model / Difficulty Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Veo-3: Match

###### Easy 18.95% 40.52% 11.76% 5.88% Medium 17.74% 35.48% 8.06% 5.65% Hard 14.41% 23.42% 8.11% 5.41%

Veo-3: Mismatch

###### Easy 19.61% 37.25% 6.86% 4.90% Medium 14.04% 23.39% 5.26% 3.51% Hard 17.28% 20.99% 3.70% 3.70%

Sora-2: Match

###### Easy 73.17% 90.24% 42.68% 19.51% Medium 63.81% 95.24% 33.33% 19.05% Hard 62.96% 96.30% 18.52% 7.41%

Sora-2: Mismatch

###### Easy 85.29% 91.18% 41.18% 29.41% Medium 80.70% 100.00% 36.84% 19.30% Hard 74.07% 96.30% 44.44% 29.63%

Wan-2.2: Match

###### Easy 0.00% 16.34% 0.00% 0.00% Medium 0.54% 12.37% 0.00% 0.00% Hard 1.80% 8.11% 0.00% 0.00%

Wan-2.2: Mismatch

###### Easy 0.00% 17.65% 0.00% 0.00% Medium 1.75% 9.36% 1.17% 1.17% Hard 0.00% 12.35% 0.00% 0.00%

Nano-banana: Match

###### Easy 27.45% 56.86% 13.73% 8.82% Medium 20.33% 50.41% 8.13% 7.32% Hard 27.03% 56.76% 13.51% 10.81%

Nano-banana: Mismatch

###### Easy 47.06% 85.29% 11.76% 11.76% Medium 35.09% 52.63% 19.30% 10.53% Hard 33.33% 44.44% 14.81% 11.11%

Nano-banana Pro: Match

###### Easy 62.77% 90.43% 45.74% 31.91% Medium 60.71% 89.29% 37.50% 29.46% Hard 65.71% 88.57% 51.43% 37.14%

Nano-banana Pro: Mismatch

###### Easy 60.71% 71.43% 35.71% 25.00% Medium 65.12% 74.42% 37.21% 30.23% Hard 54.55% 68.18% 27.27% 27.27%

GPT-4o-image: Match

###### Easy 0.98% 10.78% 1.96% 0.00% Medium 0.00% 11.29% 0.00% 0.00% Hard 0.00% 0.00% 0.00% 0.00%

GPT-4o-image: Mismatch

###### Easy 0.00% 11.76% 0.00% 0.00% Medium 1.75% 10.53% 0.00% 0.00% Hard 7.41% 14.81% 0.00% 0.00%

Qwen-image: Match

###### Easy 0.98% 5.88% 0.98% 0.98% Medium 0.00% 3.23% 0.00% 0.00% Hard 2.70% 5.41% 2.70% 2.70%

Qwen-image: Mismatch

###### Easy 2.94% 2.94% 0.00% 0.00% Medium 3.51% 7.02% 0.00% 0.00% Hard 0.00% 0.00% 0.00% 0.00%

##### Table 14 Quantitative breakdown results (Match and Mismatch) for the ARC-AGI v2 task (75 cases). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image).

Fine-grained Metrics Primary Metric Model / Category Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Video Models

Veo-3

Match 17.61% 32.08% 6.92% 3.77% Mismatch 18.18% 28.79% 4.55% 4.55%

Sora-2

Match 5.66% 18.87% 1.89% 1.89% Mismatch 0.00% 9.09% 0.00% 0.00%

Wan-2.2

Match 0.00% 4.40% 0.00% 0.00% Mismatch 0.00% 9.09% 0.00% 0.00%

Image Models

Nano-banana

Match 18.87% 45.28% 11.32% 3.77% Mismatch 18.18% 36.36% 0.00% 0.00%

Nano-banana Pro

Match 64.29% 88.10% 50.00% 33.33% Mismatch 57.14% 71.43% 28.57% 21.43%

GPT-4o-image

Match 0.00% 1.89% 0.00% 0.00% Mismatch 4.55% 4.55% 4.55% 0.00%

Qwen-image

Match 1.89% 5.66% 1.89% 1.89% Mismatch 0.00% 4.55% 0.00% 0.00%

##### Table 15 Quantitative breakdown results for the ARC-AGI v2 task (75 cases) across different difficulty levels (Easy, Medium, and Hard). We compare the performance of video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-banana, Nano-banana Pro, GPT-4o-image, and Qwen-image). * The Easy level of ARC-AGI v2 contains only one evaluation case; therefore, a model solving this case correctly achieves a 100% score.

Fine-grained Metrics Primary Metric Model / Difficulty Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Veo-3: Match

Easy 0.00% 33.33% 0.00% 0.00% Medium 20.99% 34.57% 8.64% 6.17%

- Hard 14.67% 29.33% 5.33% 1.33%

Veo-3: Mismatch

Medium 23.81% 42.86% 4.76% 4.76%

- Hard 15.56% 22.22% 4.44% 4.44%

Sora-2: Match

###### Easy 0.00% 0.00% 0.00% 0.00% Medium 7.41% 22.22% 3.70% 3.70% Hard 4.00% 16.00% 0.00% 0.00%

Sora-2: Mismatch

###### Medium 0.00% 28.57% 0.00% 0.00% Hard 0.00% 0.00% 0.00% 0.00%

Wan-2.2: Match

###### Easy 0.00% 33.33% 0.00% 0.00% Medium 0.00% 4.94% 0.00% 0.00% Hard 0.00% 2.67% 0.00% 0.00%

Wan-2.2: Mismatch

Medium 0.00% 9.52% 0.00% 0.00% Hard 0.00% 8.89% 0.00% 0.00%

Nano-banana: Match Easy 100.00% 100.00% 100.00% 100.00%* Medium 18.52% 55.56% 7.41% 0.00% Hard 16.00% 32.00% 12.00% 4.00%

Nano-banana: Mismatch

Medium 42.86% 57.14% 0.00% 0.00% Hard 6.67% 26.67% 0.00% 0.00%

Nano-banana Pro: Match Easy 100.00% 100.00% 100.00% 100.00%* Medium 66.67% 95.24% 61.90% 38.10% Hard 60.00% 80.00% 35.00% 25.00%

Nano-banana Pro: Mismatch

###### Medium 80.00% 100.00% 40.00% 20.00% Hard 44.44% 55.56% 22.22% 22.22%

GPT-4o-image: Match

###### Easy 0.00% 0.00% 0.00% 0.00% Medium 0.00% 3.70% 0.00% 0.00% Hard 0.00% 0.00% 0.00% 0.00%

GPT-4o-image: Mismatch

###### Medium 0.00% 0.00% 0.00% 0.00% Hard 6.67% 6.67% 6.67% 0.00%

Qwen-image: Match

###### Easy 0.00% 0.00% 0.00% 0.00% Medium 3.70% 7.41% 3.70% 3.70% Hard 0.00% 4.00% 0.00% 0.00%

Qwen-image: Mismatch

###### Medium 0.00% 0.00% 0.00% 0.00% Hard 0.00% 6.67% 0.00% 0.00%

##### Table 16 Comparison of Auto-Eval (VLM-based) and Human-Eval results for Veo-3 on the ARC-AGI v1 task across two subsets (Match and Mismatch).

Fine-grained Metrics Primary Metric Model / Category Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Evaluation Methods

Human-Eval

Match 23.33% 26.67% 30.00% 0.00% Mismatch 13.33% 20.00% 16.67% 0.00% Total 18.33% 23.33% 23.33% 0.00%

Auto-Eval

Match 19.01% 40.30% 9.51% 5.70% Mismatch 11.86% 25.42% 3.39% 2.54% Total 16.80% 35.70% 7.61% 4.72%

##### Table 17 Comparison of Auto-Eval (VLM-based) and Human-Eval results for Veo-3 on the ARC-AGI v2 task across two subsets (Match and Mismatch).

Fine-grained Metrics Primary Metric Model / Category Pattern Recog. ↑ Grid Integrity ↑ Color Accuracy ↑ Overall ↑ Evaluation methods

Human-Eval

Match 38.10% 47.62% 38.10% 0.00% Mismatch 5.88% 11.76% 11.76% 0.00% Total 23.68% 31.58% 26.32% 0.00%

Auto-Eval

Match 20.75% 41.51% 7.55% 1.89% Mismatch 27.27% 40.91% 4.55% 4.55% Total 22.67% 41.33% 6.67% 2.67%

### 8 Math

We construct the Visual Math task to evaluate Logical Reasoning, Logical Deduction, and 2D Spatial Reasoning. This task probes a model’s ability to understand complex mathematical problems presented visually (e.g., geometry, diagrams) and challenges its Temporal Reasoning by requiring it to generate a video animating the step-by-step deduction from premises to solution (Huang et al., 2025).

- 8.1 Hard-Level Control

We evaluate models across five benchmarks that span a wide spectrum of difficulty: GSM8K (Cobbe et al., 2021) (grade school), MATH (Hendrycks et al., 2021) (high school competition), the AIME 2024 (Mathematical Association of America, 2024) and AIME 2025 (Mathematical Association of America, 2025) invitational examinations, and Omni-MATH (Gao et al., 2024) (Olympiad-level). For Omni-MATH, we further classify problems by difficulty (five levels: T0–T4) and category (eight types: Algebra, Applied Math, Calculus, Discrete Math, Geometry, Precalculus, Number Theory, and Other). Table 18 summarizes the evaluation benchmarks and sample counts, while Table 19 details the Omni-MATH sample distribution across difficulty levels and categories. Figure 12 shows some selected examples.

- Table 18 Summary of mathematical reasoning benchmarks used for evaluation. GSM8K contains grade school math word problems. MATH covers high school competition mathematics across multiple subjects. AIME (American Invitational Mathematics Examination) represents advanced competition-level problems. Omni-MATH provides fine-grained categorization across difficulty levels and subject areas.

Dataset Level Sample Count

GSM8K Grade School 50 MATH500 High School Competition 50

- AIME 2024 Invitational Competition 30
- AIME 2025 Invitational Competition 30 Omni-MATH Multi-level (T0-T4) 167 Total – 327

- Table 19 Sample distribution of Omni-MATH dataset across difficulty levels and mathematical categories. Note: Omni-MATH provides a fine-grained ontology for mathematical problem classification. Difficulty levels range from T0 (easiest, suitable for middle school) to T4 (hardest, olympiad-level problems). Categories cover the major branches of mathematics commonly tested in competitions and standardized assessments. These results represent video generation evaluations only; image generation was not performed for this dataset.

Difficulty Algebra Applied Math Calculus Discrete Math Geometry Precalculus Number Other

- T0 (Easiest) 4 5 5 5 5 5 4 1
- T1 5 5 4 5 4 5 5 0
- T2 5 5 5 5 5 5 5 1
- T3 5 5 5 5 5 4 5 0
- T4 (Hardest) 5 5 1 5 5 4 5 0 Total 24 25 20 25 24 23 24 2

Note: Omni-MATH provides a fine-grained ontology for mathematical problem classification. Difficulty levels range from T0 (easiest, suitable for middle school) to T4 (hardest, olympiad-level problems). Categories cover the major branches of mathematics commonly tested in competitions and standardized assessments. These results represent video generation evaluations only; image generation was not performed for this dataset.

- 8.2 Evaluation and Metrics

We use Gemini-2.5-Pro (Comanici et al., 2025) as our evaluator to assess the correctness and quality of generated solutions. The evaluator analyzes intermediate reasoning steps, final answers, and reflective behavior (for video generations). We define the following metrics:

The expression 2 ∙ 3 ∙ 4 ∙ 5 + 1 is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression

Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the

if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing 2 ∙ 3 ∙ 4 ∙ 5 + 1 = 144. In total, how many values can

remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she

be obtained from the expression 2 ∙ 3 ∙ 4 ∙ 5 + 1 by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).

make every day at the farmers' market?

GSM8K

###### MATH500

Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of 𝑠 kilometers per hour, the walk takes her 4 hours, including 𝑡 minutes spent in the coffee shop. When she walks at 𝑠 +

Find the sum of all integer bases 𝑏 > 9 for

2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at 𝑠 + 12 kilometers per hour. Find the number of minutes the walk takes her, including the𝑡 minutes spent in the coffee shop.

which 17𝑏 is a divisor of 97𝑏.

###### AIME2025

AIME2024

Consider a variable point 𝑃 inside a given triangle 𝐴𝐵𝐶. Let 𝐷, 𝐸, 𝐹 be the feet of the perpendiculars from the point 𝑃 to the lines 𝐵𝐶, 𝐶𝐴, 𝐴𝐵 respectively.

For how many pairs (𝑚,𝑛) with 𝑚 and 𝑛 integers

satisfying 1 ≤ 𝑚 ≤ 100 and 101 ≤ 𝑛 ≤ 205 is

Find all points 𝑃 which minimize the sum 𝑃𝐷𝐵𝐶 + 𝑃𝐸𝐶𝐴 + 𝐴𝐵𝑃𝐹 .

3𝑚 + 7𝑛 divisible by 10?

Omni-MATH (T0, Geometry)

Omni-MATH (T4, Number)

- Figure 12 Examples from our selected five mathematical reasoning benchmarks.

- • Final Correctness: 1 if the final solution matches the ground truth answer, 0 otherwise. This metric verifies whether the model arrives at the correct conclusion.
- • Intermediate Correctness: 1 if all reasoning steps leading to the solution are logically valid and mathematically sound, 0 otherwise. This evaluates the quality of the step-by-step deduction process.
- • Action Reflection (videos only): 1 if the generated video exhibits self-correction behavior (e.g., revising incorrect steps, reconsidering approaches), 0 if the solution proceeds without reflection. This metric is not applicable to static image generations.
- • Overall Score: 1 if and only if both Final Correctness=1 AND Intermediate Correctness=1, 0 otherwise. This is the primary metric representing complete solution correctness.

- 8.3 Evaluation Results

Key Finding: The Reasoning-outcome Disconnect

#### While image generation models maintain high consistency between reasoning process and final answers, video models exhibit a severe “reasoning disconnect.” Video models frequently generate correct final answers (Outcome Success) despite flawed intermediate logic (Process Success), suggesting they rely on visual pattern matching rather than causal deduction.

- 8.3.1 Overall Performance Patterns

The Modality Gap. The quantitative results (Table 20) establish a stark performance hierarchy where image generative models consistently outperform video generative models. The gap is most visible when comparing the state-of-the-art in both modalities: on the entry-level GSM8K benchmark, Nano-banana Pro achieves a near-perfect Overall Success Rate of 97.83%, whereas the best-performing video model, Sora-2, reaches only 30.00%—a greater than 3× performance disparity. This gap widens on competition-level mathematics; on AIME 25, Nano-banana Pro maintains a robust 66.67%, while Sora-2 collapses to 0.00%.

The Illusion of Reasoning in Video. A critical insight from the fine-grained metrics is the divergence between Outcome Success Rate and Process Success Rate in video models. On GSM8K, Veo-3 achieves a relatively high Outcome Success Rate of 74.00% but a low Process Success Rate of 12.00%. This substantial ∼62% delta

[Figure 129]

[Figure 130]

(a) t=0s (b) t=2s

[Figure 131]

[Figure 132]

(c) t=4s (d) t=6s

- Figure 13 Temporal evolution of math problem solving at t=0s, 2s, 4s, 6s. Generated by Veo-3.

indicates that video models frequently arrive at the correct final solution through “hallucinated” or logically invalid visual transitions, effectively guessing the answer without successfully modeling the mathematical steps. In contrast, image models like Nano-banana Pro show near-perfect alignment (97.83% for both metrics), demonstrating that their outputs rely on consistent step-by-step deduction.

Action Reflection and Error Correction. Video-specific Action Reflection scores remain critically low across the board (0–16%), implying a limited capacity for self-correction during temporal generation.

- • Inverse Scaling: Interestingly, different video models struggle in different regimes. Sora-2 leads video models on easier tasks (30.00% on GSM8K) but degrades on hard tasks.
- • The Wan-2.2 Anomaly: Conversely, Wan-2.2 performs poorly on simple tasks (2.00% on GSM8K) but unexpectedly outperforms other video models on the hardest benchmarks (15.56% on AIME 24), suggesting it may possess latent reasoning capabilities that are not triggered by simpler prompts, or that its training distribution favors complex visual proofs over simple arithmetic.

- 8.3.2 Dataset Difficulty Progression

Performance degradation across datasets (Table 20) reveals three distinct scaling behaviors: robust retention (Nano-banana Pro), rapid collapse (GPT-4o-image), and the “middle-peak” stability of video models (Veo-3).

GSM8K (Grade School): The Hallucination Gap. On this baseline benchmark, image models demonstrate mastery, with Nano-banana Pro hitting a ceiling of 97.8%. In contrast, Veo-3 exhibits a defining characteristic of current video generation: the “illusion of competence.” While it achieves a high Outcome Success Rate of 74.00%, its Process Success Rate is only 12.00% (and Overall Success matches at 12.00%). This massive disparity suggests that on simple word problems, Veo-3 retrieves correct answers from its training data without generating the corresponding visual reasoning to support them. Sora-2 performs better here (30.00% Overall), but still trails image models by a wide margin.

MATH500 (High School Competition): The Video Stability Phenomenon. As problem complexity increases, a

##### Table 20 Quantitative results for the Math task. We evaluate video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-Banana, Nano-Banana Pro, GPT-4o-image, and Qwen-image) on selected samples from different mathematical reasoning benchmarks. Because image outputs do not support frame-by-frame reasoning, action-reflection–based metrics are omitted for image generative models (marked with “N/A”). The highest overall scores in each setting are highlighted in bold.

Fine-grained Metrics Primary Metric Model Process Success Rate ↑ Outcome Success Rate ↑ Action Reflection ↑ Overall Success Rate ↑ Dataset: GSM8K

Video Models

###### Veo-3 12.00% 74.00% 12.00% 12.00% Sora-2 38.00% 64.00% 16.00% 30.00% Wan-2.2 2.00% 2.00% 0.00% 2.00%

Image Models

###### Nano-banana 44.00% 88.00% N/A 42.00% Nano-banana Pro 97.83% 97.83% N/A 97.83% GPT-4o-image 80.00% 83.48% N/A 75.65% Qwen-image 44.00% 44.00% N/A 44.00%

Dataset: MATH500 Video Models

###### Veo-3 20.00% 52.00% 14.00% 18.00% Sora-2 34.04% 59.57% 10.64% 31.91% Wan-2.2 3.33% 6.00% 0.00% 3.33%

Image Models

###### Nano-banana 16.00% 74.00% N/A 16.00% Nano-banana Pro 91.84% 91.84% N/A 91.84% GPT-4o-image 29.14% 42.45% N/A 27.34% Qwen-image 0.00% 0.00% N/A 0.00%

- Dataset: AIME24 Video Models

Veo-3 8.33% 8.33% 5.00% 1.67% Sora-2 8.70% 13.04% 4.35% 4.35% Wan-2.2 15.56% 22.22% 11.11% 15.56%

Image Models Nano-banana 5.00% 15.00% N/A 0.00% Nano-banana Pro 63.64% 36.36% N/A 31.82% GPT-4o-image 3.33% 10.00% N/A 0.00% Qwen-image 1.12% 1.12% N/A 1.12%

- Dataset: AIME25 Video Models

Veo-3 3.33% 11.67% 1.67% 3.33% Sora-2 8.70% 21.74% 4.35% 0.00% Wan-2.2 5.56% 10.00% 3.33% 5.56%

Image Models Nano-banana 1.75% 33.33% N/A 1.75% Nano-banana Pro 71.43% 90.48% N/A 66.67% GPT-4o-image 0.00% 3.33% N/A 0.00% Qwen-image 4.00% 4.00% N/A 4.00%

Dataset: Omni-MATH Video Models

Veo-3 4.79% 15.57% 5.09% 3.89% Sora-2 0.62% 1.88% 6.88% 0.62% Wan-2.2 0.41% 3.46% 0.61% 0.41%

Image Models Nano-banana 3.90% 39.94% N/A 3.90% Nano-banana Pro 65.77% 85.59% N/A 63.06% GPT-4o-image 0.58% 2.33% N/A 0.58% Qwen-image 15.65% 15.65% N/A 14.35%

counter-intuitive trend emerges for video models. While image models like GPT-4o-image suffer a 64% relative decline (75.65% → 27.34%), Veo-3 actually improves, rising from 12.00% on GSM8K to 18.00% on MATH500. Similarly, Sora-2 remains stable (31.91%). This suggests that the structured, formal nature of competition mathematics—which often involves distinct, sequential steps—may align better with video generation temporal priors than the varying linguistic structures of grade-school word problems. Meanwhile, Nano-banana Pro continues to dominate, maintaining a 91.84% success rate.

AIME & Omni-MATH: The Hard-Reasoning Frontier. On the most challenging benchmarks, model behaviors diverge sharply:

- • The Image SOTA: Nano-banana Pro defies the difficulty curve, achieving 66.67% on AIME25 and 63.06% on Omni-MATH, proving that visual generation models can sustain complex reasoning chains.
- • The Video Shuffle: The hierarchy among video models inverts at this level. Wan-2.2, which failed on easy tasks, spikes to 15.56% on AIME24. Furthermore, on Omni-MATH, Veo-3 emerges as the most robust video model (3.89%), significantly outperforming both Sora-2 (0.62%) and Wan-2.2 (0.41%). This indicates that while Veo-3 struggles with the “exactness” of simple arithmetic (GSM8K), it possesses a superior capacity for generalized, albeit imperfect, reasoning on novel, high-complexity tasks compared to its peers.

- 8.3.3 Omni-MATH Deep Dive

The fine-grained analysis of Omni-MATH reveals that difficulty and domain do not impact video and image models uniformly (Table 21 and Table 22).

Difficulty Analysis: The “Inverse Scaling” Anomaly. Standard benchmarks usually show linear degradation as difficulty increases (T0 → T4). However, both model types exhibit counter-intuitive behavior here:

- • Video Non-Monotonicity: Veo-3 displays a “U-shaped” performance curve. It starts at 6.06% (T0), dips to a near-zero 1.47% on T3, but surprisingly recovers to 5.00% on T4 (the hardest tier). This suggests that T4 problems, while mathematically harder, may possess canonical structures (standard Olympiad templates) that video models can memorize and reproduce more effectively than the ambiguous, semi-structured problems found in intermediate tiers (T1–T3).
- • Image Inverse Scaling: Nano-banana Pro exhibits arguably the most shocking result in the dataset: it performs better on the hardest problems than the easiest ones. Its Overall Success Rate skyrockets from 26.67% on T1 to a commanding 82.76% on T4. This implies the model is highly optimized for formal, high-complexity mathematical proofs rather than simpler, variable-heavy word problems.

Category Analysis: The Visual vs. Abstract Divide. Domain breakdowns expose the specific cognitive limitations of video generation.

- • Geometry: Video models perform best in Geometry (Veo-3: 8.33%), outperforming their own averages in other domains. This confirms that video generation logic aligns well with geometric construction, where reasoning steps (e.g., drawing auxiliary lines) correspond directly to visual frame transitions.
- • The Abstract Collapse: Conversely, video models face catastrophic failure in domains requiring abstract symbolic manipulation. In Calculus, Discrete Math, Applied Math, and Precalculus, video models collapse to an Overall Success Rate of 0.00%. Notably, in Precalculus, Veo-3 achieves a 21.74% Outcome Success Rate but 0.00% overall, illustrating extreme “hallucination”—guessing the right number through invalid visual morphing.
- • Image Robustness: Unlike video models, Nano-banana Pro maintains high process correctness across all domains, including those where video fails (e.g., 53.33% in Calculus; 68.75% in Applied Math). This confirms that SOTA image models have successfully internalized the symbolic logic required for abstract math, whereas video models remain tethered to visual pattern matching.

- 8.3.4 Key Findings and Implications

Three critical insights emerge from the evaluation, pointing to fundamental divergences in how image and video architectures process mathematical logic.

- 1. The Temporal Penalties of Reasoning. The results establish that video generation currently penalizes mathematical reasoning rather than enhancing it. While image-based models like Nano-banana Pro have achieved near-mastery of complex logic (66%+ on AIME), video models face a “temporal tax.” The requirement to maintain frame-to-frame coherence appears to compete with logical consistency, leading to a massive performance gap—often exceeding 4–6× on hard benchmarks. This suggests that current video architectures treat mathematical derivation as a visual texture to be morphed, rather than a semantic chain to be constructed, necessitating new architectures that decouple reasoning states from visual rendering.
- 2. The“Hallucination”ofCompetenceinVideo. A defining failure mode of current video models is solution-answer dissociation. Models like Veo-3 frequently achieve high Outcome Success Rates (e.g., 74% on GSM8K) while failing Process Success (12%), indicating they reach correct answers through invalid or hallucinated visual transitions. This contrasts sharply with SOTA image models, where process and outcome metrics are tightly coupled (> 99% correlation for Nano-banana Pro). This dissociation undermines the utility of video for educational or explanatory tasks, as the visual "proof" provided by the video is often mathematically illusory despite the final answer being correct.
- 3. Domain-Specific “Islands of Aptitude.” Visual mathematical reasoning is not a monolithic capability but highly domain-dependent.

- • The Geometry Bias: Video models show a clear inductive bias for Geometry (8.33% success), where the reasoning process (construction, transformation) is inherently spatial-temporal.
- • The Abstract Barrier: Conversely, video models collapse to near-zero performance on abstract domains like Calculus and Discrete Math.
- • The “Inverse Scaling” Paradox: Unexpectedly, models like Wan-2.2 (Video) and Nano-banana Pro (Image) perform significantly better on the hardest benchmarks (AIME, Omni-MATH T4) than on intermediate ones. This implies that high-difficulty problems often follow rigid, canonical templates that models can memorize, whereas “simpler” problems involving variable linguistic structures or non-standard arithmetic expose the fragility of their generalized reasoning.

- 8.3.5 Case Study Analysis: Veo-3 Failure Modes

Visual case studies of Veo-3’s generation traces reveal three structural pathologies that explain the “ReasoningOutcome Disconnect” observed in the quantitative results.

- 1. Temporal Drift and Visual Hallucination. The most pervasive failure mode is Temporal Drift, where the logical integrity of the solution degrades as the video progresses, even if the initial setup is correct. This is the primary driver of the massive gap between Outcome Success (74.00%) and Process Success (12.00%) on GSM8K. In 62% of cases, Veo-3 “teleports” to the correct final answer via visually fluid but mathematically invalid transitions—morphing numbers arbitrarily or skipping essential derivation steps. This suggests the model minimizes visual prediction error (pixel consistency) at the expense of logical semantic error, prioritizing a smooth-looking video over a mathematically valid proof.
- 2. The "Correction Paradox" (Toxic Reflection). Counter-intuitively, the model’s self-correction mechanisms often act as noise injection rather than error mitigation. While Veo-3 exhibits an Action Reflection rate of 14.00% on MATH500, these edits rarely salvage the solution. Qualitative analysis shows that when the model attempts to “backtrack” and rewrite a frame, it frequently introduces continuity errors or hallucinates new, irrelevant constraints. Rather than exhibiting genuine metacognition (detecting logical flaws), the reflection appears to be a stochastic process—randomly modifying parts of the visual field—which disrupts the linear chain of reasoning required for multi-step problems.
- 3. Spatial Attention Collapse (Positional Bias). Veo-3 displays a distinct “foveal bias,” disproportionately

attending to the center of the visual field while neglecting constraints or auxiliary figures located at the periphery. This bias is particularly detrimental in Geometry tasks (where Veo-3 otherwise performs well, relative to other domains). In failed instances, the model successfully renders the central geometric construction but fails to incorporate numerical values or variable definitions positioned in the upper or lower margins. This spatial neglect results in “correctly solved” wrong problems—logic that is internally consistent but divorced from the specific boundary conditions of the prompt.

- Table 21 Quantitative breakdown results for the Omni-MATH task. We evaluate video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-Banana, Nano-Banana Pro, GPT-4o-image, and Qwen-image) on five difficulty levels (T0–T4).

Fine-grained Metrics Primary Metric

Model Process Success Rate ↑ Outcome Success Rate ↑ Action Reflection ↑ Overall Success Rate ↑

- T0 Video Models

Veo-3 6.06% 12.12% 3.03% 6.06% Sora-2 3.03% 6.06% 15.15% 3.03% Wan-2.2 0.98% 4.90% 2.94% 0.98%

Image Models Nano-Banana 0.00% 33.82% N/A 0.00% Nano-Banana Pro 53.85% 84.62% N/A 53.85% GPT-4o-image 0.00% 0.00% N/A 0.00% Qwen-image 22.00% 22.00% N/A 22.00%

- T1 Video Models

Veo-3 4.55% 4.55% 1.52% 3.03% Sora-2 0.00% 3.12% 3.12% 0.00% Wan-2.2 0.00% 6.06% 0.00% 0.00%

Image Models Nano-Banana 0.00% 19.70% N/A 0.00% Nano-Banana Pro 26.67% 60.00% N/A 26.67% GPT-4o-image 0.00% 0.00% N/A 0.00% Qwen-image 8.42% 13.68% N/A 8.42%

- T2 Video Models

Veo-3 5.56% 16.67% 2.78% 5.56% Sora-2 0.00% 0.00% 2.86% 0.00% Wan-2.2 0.93% 3.70% 0.00% 0.93%

Image Models Nano-Banana 4.17% 47.22% N/A 4.17% Nano-Banana Pro 70.00% 96.67% N/A 66.67% GPT-4o-image 2.78% 2.78% N/A 2.78% Qwen-image 7.92% 8.91% N/A 7.92%

- T3 Video Models

Veo-3 1.96% 13.24% 1.47% 1.47% Sora-2 0.00% 0.00% 9.68% 0.00% Wan-2.2 0.00% 1.96% 0.00% 0.00%

Image Models Nano-Banana 5.88% 45.59% N/A 5.88% Nano-Banana Pro 70.83% 79.17% N/A 62.50% GPT-4o-image 0.00% 2.86% N/A 0.00% Qwen-image 16.67% 12.22% N/A 12.22%

- T4 Video Models

Veo-3 5.17% 36.67% 15.00% 5.00% Sora-2 0.00% 0.00% 3.45% 0.00% Wan-2.2 0.00% 0.00% 0.00% 0.00%

Image Models Nano-Banana 9.52% 53.97% N/A 9.52% Nano-Banana Pro 82.76% 93.10% N/A 82.76% GPT-4o-image 0.00% 6.67% N/A 0.00% Qwen-image 25.68% 22.97% N/A 22.97%

##### Table 22 Quantitative breakdown results for the Omni-MATH task. We evaluate video generative models (Veo-3, Sora-2, and Wan-2.2) and image generative models (Nano-Banana, Nano-Banana Pro, GPT-4o-image, and Qwen-image) on eight categories. Because image outputs do not support frame-by-frame reasoning, action-reflection–based metrics are omitted for image generative models (marked with “N/A”).

Fine-grained Metrics Primary Metric Model Process Success Rate ↑ Outcome Success Rate ↑ Action Reflection ↑ Overall Success Rate ↑ Algebra

Video Models

Veo-3 4.17% 12.50% 4.17% 4.17% Sora-2 4.35% 8.70% 13.04% 4.35% Wan-2.2 0.00% 5.56% 0.00% 0.00%

Image Models Nano-banana 4.17% 45.83% N/A 4.17% Nano-banana Pro 61.11% 83.33% N/A 55.56% GPT-4o-image 0.00% 4.00% N/A 0.00% Qwen-image 19.70% 24.24% N/A 19.70%

Applied Math Video Models

Veo-3 0.00% 20.00% 8.00% 0.00% Sora-2 0.00% 0.00% 8.00% 0.00% Wan-2.2 0.00% 5.33% 0.00% 0.00%

Image Models Nano-banana 16.00% 32.00% N/A 16.00% Nano-banana Pro 68.75% 87.50% N/A 68.75% GPT-4o-image 4.00% 4.00% N/A 4.00% Qwen-image 14.49% 14.49% N/A 14.49%

Calculus Video Models

Veo-3 5.00% 10.00% 5.00% 0.00% Sora-2 0.00% 0.00% 10.00% 0.00% Wan-2.2 0.00% 1.67% 0.00% 0.00%

Image Models Nano-banana 0.00% 35.00% N/A 0.00% Nano-banana Pro 66.67% 80.00% N/A 53.33% GPT-4o-image 0.00% 0.00% N/A 0.00% Qwen-image 15.79% 15.79% N/A 15.79%

Discrete Math Video Models

Veo-3 0.00% 12.00% 4.00% 0.00% Sora-2 0.00% 0.00% 4.17% 0.00% Wan-2.2 0.00% 5.33% 0.00% 0.00%

Image Models Nano-banana 0.00% 24.00% N/A 0.00% Nano-banana Pro 54.55% 72.73% N/A 54.55% GPT-4o-image 0.00% 0.00% N/A 0.00% Qwen-image 15.94% 17.39% N/A 15.94%

Geometry Video Models

Veo-3 8.33% 25.00% 4.17% 8.33% Sora-2 0.00% 0.00% 0.00% 0.00% Wan-2.2 0.00% 1.39% 4.17% 0.00%

Image Models Nano-banana 0.00% 45.83% N/A 0.00% Nano-banana Pro 78.57% 92.86% N/A 78.57% GPT-4o-image 0.00% 0.00% N/A 0.00% Qwen-image 23.64% 20.00% N/A 18.18%

Precalculus Video Models

Veo-3 4.35% 21.74% 8.70% 0.00% Sora-2 0.00% 0.00% 0.00% 0.00% Wan-2.2 0.00% 1.45% 0.00% 0.00%

Image Models Nano-banana 13.04% 65.22% N/A 13.04% Nano-banana Pro 68.18% 86.36% N/A 68.18% GPT-4o-image 0.00% 4.17% N/A 0.00% Qwen-image 20.90% 16.42% N/A 16.42%

Number Video Models

Veo-3 4.17% 8.33% 12.50% 4.17% Sora-2 0.00% 0.00% 0.00% 0.00% Wan-2.2 1.39% 1.39% 0.00% 1.39%

Image Models Nano-banana 0.00% 50.00% N/A 0.00% Nano-banana Pro 64.29% 92.86% N/A 64.29% GPT-4o-image 0.00% 4.00% N/A 0.00% Qwen-image 2.82% 4.23% N/A 2.82%

### 9 Embodied Navigation

We develop the Embodied Navigation task to evaluate 3D Spatial Understanding, 2D Spatial Grounding, Temporal Reasoning, and Physical Commonsense. This task probes a model’s ability to reason about ego-centric environments, interpreting scene geometry, anticipating future states, and respecting physical constraints, and challenges its multi-step planning ability by requiring it to generate a video or image that depicts the agent’s trajectory as it navigates toward the goal.

- 9.1 Task Definition

Panoramic View Last-Mile Navigation

Top-Down View Real-World Navigation

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

Environment View: Panoramic Camera View: Third-person "over-the-shoulder”

Environment View: Top-down Camera View: Fixed Third-person Bird’s Eye

3D Real-World Navigation

Simultaneous Localization and Generation

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Environment View: 3D (Cutway / Dollhouse) Camera View: Fixed Third-person

Environment View: 3D + Top-down Camera View: Fixed Third-person

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Veo-3 Sora-2 Wan2.2 … Nano-banana GPT-4o-Image …

- Figure 14 Visual illustration of the four Embodied Navigation task settings: Panoramic View Last-Mile Navigation, Top-down View Real-World Navigation, 3D Real-World Navigation, and Simultaneous Localization and Generation (SLAG).

- Figure 14 illustrates the four Embodied Navigation task settings evaluated in our benchmark:

- • Panoramic View Last-Mile Navigation (L.M.Nav.) (Section 10) presents a 360◦ panoramic environment from a third-person “over-the-shoulder” perspective, requiring models to reason over wide-field visual context for short-range navigation.
- • Top-down View Real-World Navigation (T.V.R.-W.Nav.) (Section 11) uses a fixed bird’s-eye camera, emphasizing global spatial planning and long-horizon path prediction.
- • 3D Real-World Navigation (3D R.-W.Nav.) (Section 12) adopts cutaway/dollhouse-style renderings to expose full 3D structure, challenging models to ground navigation decisions in multi-room, multi-level layouts from a fixed third-person view.
- • Simultaneous Localization and Generation (SLAG) (Section 13) combines both 3D and top-down environment views, requiring models to jointly localize the agent and generate the surrounding scene layout. Together, these settings form a comprehensive testbed for evaluating spatial reasoning, geometric understanding, and scene-generation capabilities of modern video and image generative models.

Main Metrics -> Fine Metrics -> Tasks

- 9.2 Hard-Level Control

To build a consistent and controllable evaluation suite across the four embodied navigation tasks—Panoramic View Last-Mile Navigation, Top-down View Real-World Navigation, 3D Real-World Navigation, and Simultaneous Localization and Generation (SLAG)—we structure the dataset along four hard-level axes: Environmental Complexity, View Fidelity, Trajectory Distance, and Destination Specification. Each axis is defined through a unified set of principles shared across tasks, while accommodating modality-specific differences in sensing,

##### Table 23 Distribution of evaluation samples across the 24 hard-level configurations defined by environmental complexity, view fidelity, trajectory distance, and destination specification. Counts are reported for the four embodied navigation tasks: Panoramic View Last-Mile Navigation (L.-M.Nav.), Top-down View Real-World Navigation (T.D.V.Nav.), 3D Real-World Navigation (3D R.-W.Nav.), and Simultaneous Localization and Generation (S.L.A.G.).

Env. Complexity View Fidelity Distance Level Destination Type L.-M.Nav. T.D.V.Nav. 3D R.-W.Nav S.L.A.G.

- 1 Floor

- quality03

short

color mark 5 5 5 5 location description 5 5 5 5

long

color mark 5 5 5 5 location description 5 5 5 5

- quality04

short

color mark 5 5 5 5 location description 5 5 5 5

long

color mark 5 5 5 5 location description 5 5 5 5

- quality05

short

color mark 5 5 5 5 location description 5 5 5 5

long

color mark 5 5 5 5 location description 5 5 5 5

- 2 Plus Floors

- quality03

short

color mark 5 5 5 5 location description 5 5 5 5

long

color mark 5 5 5 5 location description 5 5 5 5

- quality04

short

color mark 5 5 5 5 location description 5 5 5 5

long

color mark 5 5 5 5 location description 5 5 5 5

- quality05

color mark 5 5 5 5 location description 5 5 5 5

short

color mark 5 5 5 5 location description 5 5 5 5

long

Total All 24 Configurations 120 120 120 120

#### spatial representation, and scene geometry. This organization ensures cross-task comparability while preserving the unique challenges intrinsic to each navigation setting.

- 9.2.1 Environmental Complexity

We source scenes from photo-realistic indoor scans in Matterport3D (Chang et al., 2017) and HM3D (Ramakrishnan et al., 2021b), as well as rendered environments in Habitat (Savva et al., 2019). Environmental complexity varies by the structural layout visible to the agent. In Panoramic View Last-Mile Navigation, complexity is determined by the spatial arrangement captured within a single panorama: floor01 scenes correspond to single-floor homes with no vertical transitions, whereas floor02plus scenes include multi-level structures with either implicit stairs not fully visible in the panorama or explicit staircases enabling vertical navigation. For the remaining three tasks, which rely on rendered 3D views or top-down maps, multi-level environments present fully connected floors and additional branching regions. These multi-floor layouts may include basements, attics, outdoor pools, and gardens. By varying structural scale while keeping other factors fixed, the benchmark ensures controlled yet diverse navigation challenges.

- 9.2.2 View Fidelity

Because the four navigation tasks use distinct rendering modalities, we define view fidelity in a task-specific yet cross-comparable manner. For Panoramic View Last-Mile Navigation, fidelity captures how much of the environment is visually accessible. Human raters evaluate the extent and spatial distribution of occlusions from foreground objects—i.e., how much of the room layout is obstructed and how many landmarks remain visible. Scores range from 3 to 5, corresponding to quality03 through quality05. For Top-down View Real-World Navigation, 3D Real-World Navigation, and SLAG, fidelity reflects a more holistic assessment of scene realism

[Figure 146]

(a) Sample Vocabulary Distribution of Panoramic View Last-Mile Navigation.

[Figure 147]

(b) Sample Vocabulary Distribution of 3D Real-World Navigation.

- Figure 15 Word Clouds of Semantic Target Descriptions for Destination Specification (Hard Level). It visualizes the frequency of terms used by annotators to describe target locations using natural language. The dominance of spatial prepositions (e.g., “front,” “top,” “next,” “right”) and structural landmarks (e.g., “staircase,” “window,” “entrance,” “living [room]”) indicates that annotators heavily rely on relative positioning and salient architectural features. This validates the hard-level protocol, where ambiguity in complex environments is resolved by adding spatially anchored landmarks and floor identifiers to uniquely isolate the intended area.

and navigability. Raters consider factors such as the presence of holes or cracks, furnishing quality, door openness, and the plausibility of interaction with the environment. These scenes are likewise scored on the 3–5 scale aligned with quality03 to quality05.

- 9.2.3 Trajectory Distance

Trajectory distance is defined as the geodesic separation between the agent’s starting point and the target location. We categorize trajectories into two types. Short trajectories involve relatively direct motion: they require no major turns and may include vertical movement (e.g., reaching an upper floor) without substantial directional changes. Long, In contrast,trajectories include at least one significant turn. To maintain comparability across the four tasks, long trajectories are selected so that they share partial path structure with the corresponding short cases at the same hard level.

- 9.2.4 Destination Specification

Targets are specified either through direct visual annotation or through a natural-language description of the same location. For color-based targets (color mark), annotators highlight the target region in the input image using a pure red overlay (#ff0000), ensuring that it is visually distinguishable from its surroundings. For semantic targets (location description), annotators describe the corresponding region using natural language. Although both specifications are intended to reference the same location, ambiguity may arise when multiple similar regions exist. In such cases, annotators include additional disambiguating details—such as floor identifiers and spatially anchored landmarks—to ensure the description uniquely identifies the intended target area.

These four axes allow us to systematically control difficulty across all 24 configuration slots summarized in the table, while keeping the evaluation consistent across tasks that vary in sensing modality, scene rendering, and navigation objective. Table 23 provides the detailed distribution of the Embodied Navigation samples.

- 9.3 Evaluation and Metrics

Evaluating navigation-conditioned video or image generation requires metrics that jointly capture geometric fidelity and visual-semantic reasoning. Our evaluation focuses on whether the agent moves plausibly through the environment, correctly interprets the navigation instruction, adheres to the physical layout of the scene, and preserves the identity and spatial integrity of the target destination. All metrics are binary and are computed directly from the agent’s execution trace and the generated video frames. The definitions below consolidate the evaluation protocol used across all navigation tasks, drawing from the criteria specified in the evaluation prompt templates and supplementary rules. As illustrated in Figure 16, this framework is adaptive:

#### while several fine-grained metrics are shared across multiple tasks, others are task-specific to address unique navigation modalities.

[Figure 148]

##### Figure 16 Evaluation metrics flow. Decomposition of the three main metrics Task Completeness, Video Quality in Physical Understanding, and Video Quality in Instruction Following into fine-grained components (e.g., Consistency, Physical Plausibility, Instruction Alignment) and their mapping to the corresponding navigation tasks.

- 9.3.1 Task Completeness Metrics

These metrics assess whether the agent reaches or meaningfully approaches the correct destination based solely on geometric information. They intentionally ignore visual fidelity, semantic correctness, and physical plausibility, which are evaluated by separate metrics.

- • SuccessScore(S.S.2D). Measures whether the agent’s final position lies within the highlighted or textually specified goal region in the 2D overhead map. The score is 1 if the final coordinates fall entirely inside the goal footprint; otherwise 0.
- • Oracle Success Score (O.S. 2D). Provides partial credit when the agent comes sufficiently close to the 2D goal during navigation. The score is 1 if the agent’s path ever intersects or touches the goal region, even if it does not stop there; otherwise 0.
- • Success Score (S.S. 3D). Checks whether the agent ends inside the correct destination volume in the 3D navigation sequence. This metric is purely geometric and independent of any visual discrepancies at the destination. The score is 1 if the final 3D position is within the target volume; otherwise 0.
- • Oracle Success Score (O.S. 3D). Grants credit when the agent enters the vicinity of the correct 3D destination at any point during its rollout. The score is 1 if the trajectory ever crosses the predefined proximity threshold around the target; otherwise 0.
- • Trajectory Alignment Score. Evaluates whether the agent’s 2D projected route is consistent with its 3D motion path, focusing on major turns and spatial transitions. A score of 1 indicates strong correspondence between the two trajectories; otherwise 0.

- 9.3.2 Video Quality in Physical Understanding

#### These metrics assess whether the agent’s motion obeys basic physical principles and remains consistent with the underlying scene geometry. They focus on physical plausibility, continuity, and spatial coherence rather than destination correctness.

- • Object Semantic Score (Obj. Sem.). Evaluates whether the agent interacts with the environment in a physically valid way. The agent must not collide with, pass through, or visually intersect solid structures such as walls, furniture, or appliances. Score is 1 if no collision or penetration is observed; otherwise 0.
- • Agent Consistency Score (Agent Con.). Measures temporal continuity and identity preservation of the navigating agent. For image generation tasks: (1) The agent’s trajectory must remain continuous across frames, and (2) exactly one agent should appear throughout the navigation sequence. Score is 1 if the same agent moves smoothly and consistently across all frames; otherwise 0.
- • Spatial Alignment Score (Spa. Ali.). Checks whether the agent’s heading, motion direction, and elevation changes remain coherent with the expected physical layout. For image generation tasks: (1) The initial position must be visually identifiable when provided, and (2) the agent’s initial facing direction must align with its first movement. Score is 1 if heading, transitions, and movement direction are physically and visually consistent; otherwise 0.

- 9.3.3 Video Quality in Instruction Following

Because a video generation model can “cheat” in embodied navigation—by fabricating a visually plausible destination, altering the environment, or painting a new target beneath the agent—we impose strict constraints to ensure faithful instruction following. These metrics evaluate whether the generated video preserves the intended destination and maintains a static, coherent scene.

- • Destination Integrity Score (Des. Inte.). Assesses whether the destination region is preserved and correctly interpreted by the model. According to the supplementary rules: (1) The red-marked target region must remain unchanged in size, position, texture, and overall appearance; (2) The agent must not rely on hallucinated alternatives (e.g., newly created look-alike objects or fabricated goal markers). The score is 1 if the original destination remains intact and the agent stops within that region; otherwise 0.
- • Scene Consistency Score (Scene Con.). Evaluates whether the environment remains static throughout the video. No objects, lighting, geometry, or layout elements may appear, disappear, deform, or shift in a way that violates the static-scene assumption. The score is 1 if the scene stays unchanged across all frames; otherwise 0.

Across all four navigation tasks, these video-quality metrics employ a unified binary (pass/fail) definition to ensure consistent evaluation.

- 9.3.4 Holistic Performance

Across all embodied navigation tasks, we define an Overall metric that measures end-to-end success under a strict, holistic criterion: a sample is considered correct only if all fine-grained evaluation metrics simultaneously achieve a score of 1. This requirement highlights a key observation about current generative models: strong performance on isolated metrics does not necessarily translate into coherent, successful task execution. The Overall score thus captures the true difficulty of producing videos that are simultaneously geometrically correct, physically plausible, visually consistent, and instruction-faithful.

We run both automatic VLM-based evaluation and human evaluation for Embodied Navigation task. The automatic setup uses Gemini-2.5-Pro (Comanici et al., 2025) with a structured evaluation prompt; it receives (i) the model-generated video or image and (ii) task-specific context, and returns binary metric scores and short thinking of the justifications. Human raters are given the exact same media and prompts to ensure their labels align with the same judgment criteria.

- 9.4 Overall Evaluation Results

We evaluate the embodied navigation capabilities of state-of-the-art generators by consolidating per-task performance into four metric families: Task Completeness, Physical Understanding, Instruction Following, and the strict Holistic Overall metric. The comparative results across models are detailed in Table 24, while Table 25 provides a fine-grained human evaluation of Veo-3 to diagnose specific failure modes.

- 9.4.1 Comparative Analysis of Generators

As detailed in Table 24, Nano-banana establishes a distinct lead across most axes, demonstrating superior instruction adherence and holistic accuracy. It notably surpasses 74% holistic accuracy on both Panoramic and 3D navigation tasks. Among video models, performance is highly variable. Veo-3 generally outperforms other video generators like Sora-2 and Wan-2.2, particularly in maintaining physical plausibility (reaching 93.3% in Panoramic views). However, video models often exhibit a "completeness vs. coherence" trade-off: while they frequently achieve decent physical understanding scores, they struggle to combine this with instruction following, resulting in significantly lower holistic scores compared to the image-based Nano-banana. GPT-4o-image generally lags behind, struggling to maintain temporal coherence across sequences.

Performance varies significantly by task complexity:

- • Panoramic View (L.M.Nav.): Nano-banana achieves its second-highest holistic score here (74.2%). Among video models, Veo-3 performs best (60.0% holistic) and actually achieves the highest physical understanding score of any model (93.3%). In contrast, Sora-2 and Wan-2.2 fail to produce holistically valid trajectories (0.0%), despite moderate physical understanding.
- • Top-down View (T.V.R.-W.Nav.): This task proves universally difficult, yet it is the only category where a video model outperforms Nano-banana. Veo-3 achieves the highest holistic accuracy (19.5%), surpassing Nano-banana (11.1%) and Sora-2 (3.4%). This suggests that Veo-3 possesses superior overhead spatial reasoning capabilities compared to its peers.
- • 3D Real-World Navigation (3D R.-W.Nav.): This task highlights the widest gap between architecture types. Nano-banana achieves a dominant holistic score of 79.2%. While Wan-2.2 (24.2%) and Veo-3 (22.5%) manage to complete some routes, Sora-2 fails completely (0.0%), indicating that current video generators struggle to maintain long-horizon coherence in 3D environments.
- • Simultaneous Localization and Generation (SLAG): Nano-banana leads with 28.8% holistic accuracy. Contrary to simpler tasks, Sora-2 (12.9%) slightly outperforms Veo-3 (11.2%) here, while Wan-2.2 collapses almost entirely (0.8%). While video models do not fail completely (scoring > 0 on fine-grained metrics), their low holistic scores indicate a critical inability to maintain the precise cross-view alignment required for SLAG.

- 9.4.2 Fine-Grained Analysis of Veo-3

To better understand why video models struggle despite strong visual quality, we report a detailed human evaluation of Veo-3 in Table 25. The results uncover a sharp disconnect between component-level physical understanding and holistic task success.

- • Local vs. Global Success: In the Panoramic setting, Veo-3 achieves strong component scores for Object Semantic Score (87.50%) and Agent Consistency Score(92.50%). However, the Overall success rate collapses to 26.67% because the model rarely satisfies all validity checks simultaneously.
- • The “Plausible but Wrong” Problem: In 3D Real-World Navigation, Veo-3 retains high Physical Quality scores (77.50% Agent Consistency) but fails almost entirely on Instruction Following (1.67% Overall). The video generation looks physically realistic—objects are stable and lighting is consistent—but the agent fails to navigate to the correct destination.
- • GeometricFailures: The SLAG task results are particularly telling. While Scene Consistency is remarkably high (93.90%), Trajectory Alignment is nearly non-existent (6.10%). This confirms that while the model can generate temporally consistent frames, it lacks the geometric grounding necessary to align those frames with a specified trajectory map.

- 9.5 Key Observations

In this section, we distill the key insights that emerge from evaluating state-of-the-art video generation models across the four embodied navigation tasks. These findings offer a high-level view of current strengths and limitations, setting the stage for deeper investigation. In the following sections—Sections 10.3, 11.3, 12.3 and 13.3—we introduce each subtask in detail and provide comprehensive quantitative and qualitative analyses.

##### Table 24 Per-task performance across all metric families for three video generative models (Veo-3, Sora-2, and Wan-2.2) and two image generative models (Nano-banana, GPT-4o-image).

Task / Model Task Completeness ↑ Physical Understanding ↑ Instruction Following ↑ Holistic Overall ↑ Panoramic View Last-Mile Navigation (L.M.Nav.)

Veo-3 76.2 93.3 83.8 60.0 Sora-2 0.4 84.4 0.4 0.0 Wan2.2 25.0 71.4 39.6 0.0 Nano-banana 79.2 91.1 85.4 74.2 GPT-4o-Image 0.0 32.2 0.0 0.0

Top-down View Real-World Navigation (T.V.R.-W.Nav.) Veo-3 66.1 69.8 44.1 19.5 Sora-2 38.1 75.4 13.6 3.4 Wan2.2 32.2 66.7 22.5 5.1 Nano-banana 46.5 74.1 43.8 11.1 GPT-4o-Image 16.5 37.3 4.7 3.4

3D Real-World Navigation (3D R.-W.Nav.) Veo-3 76.7 79.4 47.1 22.5 Sora-2 15.4 76.9 11.2 0.0 Wan2.2 60.4 68.1 69.2 24.2 Nano-banana 79.9 96.8 86.8 79.2 GPT-4o-Image 14.6 63.1 16.2 13.3

Simultaneous Localization and Generation (SLAG) Veo-3 41.2 56.6 38.8 11.2 Sora-2 36.4 76.7 61.6 12.9 Wan2.2 12.9 24.0 25.8 0.8 Nano-banana 56.7 81.3 73.5 28.8 GPT-4o-Image 30.2 61.7 58.3 16.1

##### Table 25 Fine-grained human evaluation of Veo-3 across the four embodied navigation tasks in MMGR: Panoramic View Last-Mile Navigation, Top-down View Real-World Navigation, 3D Real-World Navigation, and Simultaneous Localization and Generation (SLAG). Each column reports per-task pass rates for the corresponding metric, and the Overall row reflects the strict holistic success rate where all applicable checks must succeed simultaneously.

Top-down View Real-World Navigation

Simultaneous Localization and Generation Task Completeness ↑

Panoramic View Last-Mile Navigation

3D Real-World Navigation

Metric

S. S. 3D 70.00 N/A 76.67 42.68 O. S. 3D 71.67 N/A 85.00 53.66 S. S. 2D) N/A 33.05 N/A 23.17 O. S. 2D) N/A 65.25 N/A 28.04 Traj. Ali. N/A N/A N/A 6.10

Physical Understanding Quality ↑

###### Obj. Sem. 87.50 38.14 78.33 34.15 Agent Con. 92.50 70.34 77.50 65.85 Spa. Ali. 96.67 58.47 68.33 65.85

Instruction Following Quality ↑

###### Des. Inte. 60.00 65.25 45.83 54.88 Scene Con. 64.17 81.36 16.67 93.90

Holistic Performance ↑ Overall 26.67 5.93 1.67 0.00

##### Note: S.S. = Success Score; O.S. = Oracle Success Score; Traj. Ali. = Trajectory Alignment Score; Obj. Sem. = Object Semantic Score; Agent Con. = Agent Consistency Score; Spa. Ali. = Spatial Alignment Score; Des. Inte. = Destination Integrity Score; Scene Con. = Scene Consistency Score. “N/A” indicates the metric is not applicable to the specific task.

- Findings 1: Strong Capability in Ego-Centric Navigation Video Generation. Our results indicate that video generation models possess a surprisingly robust capability for third-person, off-the-shoulder perspectives navigation generation. Beyond simple frame coherence, these models demonstrate abilities that have been a long-pursuing goal in embodied navigation (Batra et al., 2020; Anderson et al., 2018a), such as the understanding of scene layouts (Li et al., 2023; Fuentes-Pacheco et al., 2015; Henriques & Vedaldi, 2018) and semantic meanings (Cartillier et al., 2021; Chaplot et al., 2020b; Zhou et al., 2025; Irshad et al., 2021; Wani et al., 2020; Blanco et al., 2008; Konolige et al., 2011; Gomez et al., 2020; An et al., 2022), effectively recognizing goal objects and maintaining contextual consistency even when the agent’s body is visible within the frame. Notably, they exhibit strong capabilities in imagining spatial relationships (Koh et al., 2021; Qin et al., 2025; Sridhar et al., 2024; Bar et al., 2025; Shah et al., 2025), allowing them to consistently model the geometric interaction between the agent and the environment. This confirms that video diffusion models implicitly learn powerful geometric priors and environment continuity, which are critical for perceiving the immediate surroundings from an embodied perspective. However, such spatial imagination does not translate into functional navigation except in the Panoramic View Last-Mile Navigation task. Only when the goal is already visible, and the required trajectory is short and locally constrained, do models like Veo-3 achieve meaningful success rates. For longer-horizon or multi-view settings, their ability to reason over distance, occlusion, or multi-step spatial transformations rapidly deteriorates, exposing the gap between ego-centric perception and actionable planning.
- Findings 2: Free-form Generation is Insufficient for Navigation. Tasks requiring controlled movement, like Topdown View Real-World Navigation, SLAG, and Instruction-based Target Search, reveal a consistent failure pattern: video models excel at producing aesthetically coherent frames but fail to follow navigation constraints. Models such as Sora-2 often generate plausible but irrelevant motion, drift off the intended path, or remain static despite instructions. Even with perfect semantic grounding (e.g., Sora-2 achieving 87.35% Object Semantic Score), Success Scores remain near zero, underscoring that generative quality does not equal navigational utility. The inability to reliably execute directed motion suggests that current video models lack mechanisms for consistent temporal control, trajectory commitment, or adherence to spatial rules imposed by the prompt.
- Findings 3: Cross-View Spatial Alignment Exists, But Must Be Explicitly Activated. Our SLAG experiments reveal an unexpected finding: while trajectory alignment remains highly challenging, scene consistency improves markedly when models are evaluated in a cross-view alignment framework. In SLAG, aligning generated motion with a 2D top-down map forces the model to maintain geometric coherence across viewpoints, and this pressure appears to activate latent spatial priors that remain unused in other tasks. Veo-3, in particular, maintains high Scene Consistency even when its navigation fails, suggesting robust internal world modeling. However, this ability does not emerge in free-form video generation or text-only prompting. It requires structured multimodal conditioning, indicating that cross-view alignment is a learnable but currently underutilized capability in video models.

Additional Profile of Strengths and Weaknesses. Our results suggest a preliminary “navigation capability profile” for video generation models:

- • Strengths: strong semantic grounding, accurate object identity preservation, coherent ego-centric spatial reasoning, and latent ability for cross-view geometric alignment.
- • Weaknesses: poor rule-following under long-horizon instructions, inability to maintain goal-directed motion, hallucination under abstract destination descriptions (e.g., Veo-3 generating an entirely new scene matching the semantic description), and brittleness to environmental complexity.

Summary. Taken together, these findings suggest that currentvideogenerationmodelspossessstrongego-centric spatial perception, semantic grounding, and latent cross-view alignment capability, but they fail to reliably execute goal-driven navigation under free-form generation. Their cross-view understanding must be explicitly elicited through structured multimodal constraints such as SLAG, while long-horizon trajectory control and rule following remain largely unsolved. In the following sections, we provide detailed analyses for each navigation task, including fine-grained quantitative and qualitative breakdowns.

### 10 Panoramic View Last-Mile Navigation (L.-M. Nav.)

- 10.1 Task Definition

The Panoramic View Last-Mile Navigation task evaluates fine-grained, embodied decision-making when a target is already within the agent’s visual field. Distinct from large-scale route planning, last-mile navigation isolates the critical final phase of movement: precisely localizing a visible goal and generating an optimal shorthorizon trajectory toward it. In this setting, the model receives a single panoramic RGB observation and must infer a feasible motion plan leading directly to the destination. Targets are either explicitly defined by a red marker or implicitly specified via object-class descriptions, necessitating a synthesis of geometric reasoning and semantic recognition. A successful execution by Veo-3 is illustrated in Figure 17. Ultimately, this task probes a model’s capacity to interpret egocentric 360◦ spatial layouts, estimate relative pose and depth, and execute precise actions to bridge perception and control in the final meters of navigation.

Start Position: The robot’s position Goal: The red area

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

Frame 5 Frame 6 Frame 7 Frame 8

- Figure 17 A successful case completed by Veo-3 on the Panoramic View Last-Mile Navigation task. The model navigates the final segment of the route from a first-person panoramic viewpoint, demonstrating coherent mental mapping, spatial awareness, and accurate localization needed to reach the precise target destination.

- 10.2 Evaluation and Metrics

Evaluating navigation-conditioned video generation presents a dual challenge: quantifying geometric precision while ensuring visual-semantic fidelity. Our framework explicitly assesses whether the agent executes plausible motion, maintains structural consistency, and preserves the semantic integrity of the destination. All metrics are binary (pass/fail) and come from the rollout traces and generated frames.

We adopt the shared embodied evaluation protocol in Section 9.3: all metrics are binary and labeled via the automatic VLM/human rules on the rollout traces and generated frames. The Physical Understanding and Instruction Following checks (Object Semantic, Agent Consistency, Spatial Alignment, Destination Integrity, Scene Consistency) are unchanged; here we only note how task completeness metrics are used for Panoramic Last-Mile Navigation and how the gated scores are formed.

- 10.2.1 Task Completeness Metrics (Geometry Only)

These metrics isolate navigational accuracy, evaluating whether the agent reaches the destination based purely on geometric coordinates. Visual fidelity and physical consistency are excluded here and assessed by subsequent metrics.

- • Success Score 3D (S.S. 3D). Measures whether the final state of the agent coincides with the target location. This metric is strictly geometric; it is satisfied if the agent terminates its trajectory within the defined volume of the destination, regardless of visual preservation.

#### • Oracle Success Score 3D (O.S. 3D). A relaxed variation of Success Score 3D that credits the agent for reaching the target at any point during the trajectory. The metric passes if the agent enters the destination’s proximity threshold at any timestep, irrespective of where it ultimately stops.

- 10.2.2 Gate Metrics and Holistic Performance

To ensure a rigorous assessment, we introduce composite “Gate Metrics” that intersect geometric success with semantic and physical validity. These metrics serve as strict filters, disqualifying trajectories that achieve goal conditions through generative artifacts such as hallucination or geometry warping.

- • Success (3D) with Original Destination. This metric imposes a strict visual-geometric consistency check. It counters the tendency of generative models to exploit scene manipulation—such as fabricating a target at the agent’s feet or warping the room layout—to satisfy stopping conditions artificially. A trajectory is successful under this metric only if it satisfies the Success Score 3D (S.S. 3D) while simultaneously passing the Destination Integrity (Des. Inte.) and Scene Consistency (Scene Con.) checks.
- • PhysicsValidity. A holistic measure of physical plausibility, ensuring that movement is grounded in reality rather than dream-like transitions. Geometric success is disregarded if the agent violates fundamental physical laws. This metric requires the simultaneous satisfaction of three conditions: (1) Object Semantic Score (Obj. Sem.) (no collisions); (2) Agent Consistency Score Agent Con.) (identity persistence); and (3) Spatial Alignment Score (Spa. Ali.) (pose coherence).
- • Overall Success = Success Score 3D ∧ Oracle Success Score 3D ∧ Object Semantic ∧ Agent Consistency ∧ Spatial Alignment ∧ Destination Integrity ∧ Scene Consistency; a sample passes only when all seven binary checks are 1. This strict criterion highlights that strong performance on individual metrics does not guarantee successful, plausible end-to-end task completion.

- 10.3 Evaluation Results

Key Finding: Model Capabilities & The Evaluation Gap

- • Performance Stratification: Results are strictly binary; only Veo-3 and Nano-banana function, while Sora-2 and GPT-4o fail completely. Surprisingly, the image-based Nano-banana often outperforms Veo-3 in complex environments and instruction following.
- • Auto-Metric Failure: A massive reliability gap exists between evaluation methods. Auto-metrics rate Veo-3 at 73.33% success, while humans rate it at only 25.00%. Auto-evaluators successfully track camera motion but fail to penalize physical hallucinations and logic violations.

- 10.3.1 VLM-Based Evaluation

The quantitative results in Table 26 reveal a distinct stratification in model capabilities for the Panoramic View Last-Mile Navigation task. The performance landscape is strictly binary: Veo-3 and Nano-banana demonstrate functional navigation capabilities, while Sora-2 and GPT-4o-image fail to complete the task entirely (0.00% Overall Success).

Top-TierPerformance(Veo-3vs. Nano-banana). While Veo-3 is the primary video model of interest, Nano-banana serves as a surprisingly robust baseline, frequently outperforming the video model in complex scenarios.

- • Environmental Robustness: In simple environments (floor01), both models achieve parity with a 73.33% Overall Success rate. However, as environmental complexity increases (floor02plus), Veo-3’s performance degrades sharply to 46.67%, whereas Nano-banana maintains a high success rate of 75.00%.
- • VisualFidelityScaling: A notable divergence occurs in how the models handle visual quality. Nano-banana exhibits positive scaling with fidelity, improving from 60.00% success at quality03 to 82.50% at quality05. In contrast, Veo-3 plateaus, peaking at only 62.50% regardless of increased visual fidelity.

Instruction Following Capabilities. A critical differentiator found in the results is the response to destination specifications.

- • Visual vs. Textual Guidance: When the destination is specified by a visual marker (color mark), both models perform comparably (≈70%).
- • Descriptive Navigation: When the destination is defined by text instructions (location description), Veo-3 struggles, dropping to 50.00% success. Conversely, Nano-banana excels, achieving its highest success rate of 78.33%. This suggests that while Veo-3 offers superior temporal consistency (Agent Consistency >93% across most settings), Nano-banana possesses superior multimodal understanding, allowing it to map textual descriptions to visual goals more effectively.

Navigation Failures (Sora-2 and GPT-4o-image). The failing models illustrate two distinct modes of error.

- • Static vs. Dynamic Failure: GPT-4o-image achieves high Object Semantic Scores (up to 100%) but fails to generate the temporal trajectory required for navigation, resulting in a 0.00% success rate.
- • Steerability Failure: Sora-2 presents a “steerability” problem. While it maintains decent Physical Understanding (e.g., 88.33% Object Semantic Score in floor01) and generates coherent video, it lacks Destination Integrity (0.00%). The model hallucinates plausible scenes but cannot be constrained to a specific coordinate or target path.

Impact of Trajectory Distance. As anticipated, performance for the leading video model, Veo-3, degrades with trajectory length, dropping from 66.67% (Short) to 53.33% (Long). Nano-banana remains more resilient to distance, maintaining a 66.67% success rate even on long trajectories, further validating the robustness of image-based stepwise generation over holistic video generation for this specific navigation benchmark.

- 10.3.2 Human vs. Automated Evaluation Discrepancy

- Table 27 presents a critical finding: Automatic evaluation metrics are significantly over-optimistic compared to Human Expert evaluation. This discrepancy exposes the limitations of current auto-evaluators in detecting physical and logical inconsistencies in video generation.

The “Overall Success” Collapse. The most striking divergence appears in the Holistic Metric. While Auto Evaluation suggests Veo-3 is a competent navigator, human judges strongly disagree.

- • Baseline Discrepancy: In the simplest setting (floor01), Auto Evaluation reports a 73.33% success rate, whereas Human Evaluation rates it at only 25.00%.
- • Complexity Penalty: This gap widens in complex scenarios. For Long Trajectories, Auto claims a 53.33% success rate, while humans find that only 11.67% of the trajectories are actually successful.

The “Hallucination” Blind Spot (Physics & Consistency). The primary driver of the reliability gap is the auto-evaluator’s inability to penalize “dream-like” logic violations.

- • Physics Validness: There is a massive disconnection in physical grounding. For floor01, Auto reports 81.67% validity, while humans report 25.00%. In Long Trajectories, human-rated physics validity plummets to 16.67%, indicating that over longer durations, the model frequently clips through objects, floats, or violates lighting consistencies—errors that auto-metrics (likely based on frame-to-frame pixel similarity) fail to capture.
- • Scene Stability: Auto-evaluators consistently overrate Scene Consistency (e.g., 98.33% for floor01). In contrast, humans rate it at 66.67%. This confirms that auto-metrics struggle to detect subtle temporal morphing of walls, textures, or landmarks that humans immediately recognize as inconsistent. Notably, for Location Description tasks, Scene Consistency drops to 48.33% in human eyes, suggesting that text-conditioning may induce greater visual instability than visual prompting.

Task Completion vs. Reality. Even when the agent appears to move correctly, it often fails to satisfy the exact conditions of the destination. Auto-metrics (Destination Integrity) suggest the agent reaches the target 90.00% of the time in simple settings. Humans, however, judge this at only 55.00%, indicating that the agent often stops short, overshoots, or faces the wrong direction upon arrival—nuances the auto-evaluator misses.

##### Table 26 Quantitative results for the Panoramic View Last-Mile Navigation benchmark. We compare performance across two video generative models (Veo-3 and Sora-2) and two image generative models (Nano-banana, and GPT-4o-image).

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Success (3D) Original Destination

Destination Integrity Score

Scene Consistency Score

Object Semantic Score

Agent Consistency Score

Spatial Alignment Score

Success Score (3D)

Oracle Success Score (3D)

Physics Validness

Overall Success

Model

Environmental Complexity

- Level: floor01 Video Models

Veo-3 90.00% 90.00% 93.33% 93.33% 93.33% 90.00% 98.33% 90.00% 81.67% 73.33% Sora-2 0.00% 1.67% 93.33% 88.33% 93.33% 0.00% 0.00% 0.00% 81.67% 0.00%

Image Models Nano-banana 76.67% 80.00% 96.67% 88.33% 88.33% 80.00% 91.67% 75.00% 88.33% 73.33% GPT-4o-image 0.00% 0.00% 100.00% 1.67% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

- Level: floor02plus Video Models

Veo-3 58.33% 66.67% 95.00% 93.33% 91.67% 55.00% 91.67% 55.00% 83.33% 46.67% Sora-2 0.00% 0.00% 81.36% 77.97% 76.27% 0.00% 1.69% 0.00% 64.41% 0.00%

Image Models Nano-banana 80.00% 80.00% 96.67% 90.00% 86.67% 80.00% 90.00% 78.33% 86.67% 75.00% GPT-4o-image 0.00% 0.00% 91.67% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

View Fidelity

- Level: quality03 Video Models

Veo-3 72.50% 80.00% 92.50% 92.50% 90.00% 70.00% 92.50% 70.00% 77.50% 55.00% Sora-2 0.00% 2.56% 79.49% 82.05% 76.92% 0.00% 2.56% 0.00% 69.23% 0.00%

Image Models Nano-banana 62.50% 62.50% 92.50% 87.50% 82.50% 62.50% 80.00% 62.50% 82.50% 60.00% GPT-4o-image 0.00% 0.00% 90.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

- Level: quality04 Video Models

Veo-3 75.00% 75.00% 95.00% 95.00% 90.00% 75.00% 97.50% 75.00% 82.50% 62.50% Sora-2 0.00% 0.00% 85.00% 80.00% 87.50% 0.00% 0.00% 0.00% 67.50% 0.00%

Image Models Nano-banana 85.00% 90.00% 100.00% 87.50% 87.50% 90.00% 92.50% 80.00% 87.50% 80.00% GPT-4o-image 0.00% 0.00% 100.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

- Level: quality05 Video Models

Veo-3 75.00% 80.00% 95.00% 92.50% 97.50% 72.50% 95.00% 72.50% 87.50% 62.50% Sora-2 0.00% 0.00% 97.50% 87.50% 90.00% 0.00% 0.00% 0.00% 82.50% 0.00%

Image Models Nano-banana 87.50% 87.50% 97.50% 92.50% 92.50% 87.50% 100.00% 87.50% 92.50% 82.50% GPT-4o-image 0.00% 0.00% 97.50% 2.50% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

Trajectory Distance

Level: short

Video Models Veo-3 81.67% 83.33% 96.67% 93.33% 90.00% 80.00% 98.33% 80.00% 83.33% 66.67% Sora-2 0.00% 0.00% 88.33% 83.33% 85.00% 0.00% 1.67% 0.00% 76.67% 0.00%

Image Models Nano-banana 86.67% 86.67% 98.33% 93.33% 91.67% 86.67% 95.00% 85.00% 91.67% 81.67% GPT-4o-image 0.00% 0.00% 95.00% 1.67% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

Level: long

Video Models Veo-3 66.67% 73.33% 91.67% 93.33% 95.00% 65.00% 91.67% 65.00% 81.67% 53.33% Sora-2 0.00% 1.69% 86.44% 83.05% 84.75% 0.00% 0.00% 0.00% 69.49% 0.00%

Image Models Nano-banana 70.00% 73.33% 95.00% 85.00% 83.33% 73.33% 86.67% 68.33% 83.33% 66.67% GPT-4o-image 0.00% 0.00% 96.67% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

Destination Specification

Level: color mark

Video Models Veo-3 83.33% 88.33% 96.67% 93.33% 93.33% 80.00% 93.33% 80.00% 85.00% 70.00% Sora-2 0.00% 1.69% 86.44% 77.97% 81.36% 0.00% 0.00% 0.00% 67.80% 0.00%

Image Models Nano-banana 76.67% 80.00% 95.00% 85.00% 81.67% 80.00% 86.67% 73.33% 81.67% 70.00% GPT-4o-image 0.00% 0.00% 93.33% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

Level: location description

Video Models Veo-3 65.00% 68.33% 91.67% 93.33% 91.67% 65.00% 96.67% 65.00% 80.00% 50.00% Sora-2 0.00% 0.00% 88.33% 88.33% 88.33% 0.00% 1.67% 0.00% 78.33% 0.00%

Image Models Nano-banana 80.00% 80.00% 98.33% 93.33% 93.33% 80.00% 95.00% 80.00% 93.33% 78.33% GPT-4o-image 0.00% 0.00% 98.33% 1.67% 0.00% 0.00% 0.00% 0.00% 0.00% 0.00%

Areas of Agreement: The Camera Motion Paradox. Interestingly, Spatial Alignment is the only metric where humans frequently rate the model higher than the automatic system. For Long Trajectories, humans rated Spatial Alignment at 98.33%, exceeding the Auto rating of 95.00%. This suggests that the camera movement itself (the “ego-motion”) is highly convincing to human observers. The failure of Veo-3 is not in moving like an agent, but in maintaining the world around the agent. The motion is realistic; the environment is not.

Conclusion. While Veo-3 demonstrates state-of-the-art potential in generating semantically correct paths, the stark contrast between the 73.33% (Auto) and 25.00% (Human) success rates serves as a warning. Current automatic metrics for video generation prioritize visual similarity over physical logic, making them insufficient proxies for measuring true “World Modeling” capabilities.

- Table27 Quantitative results for the PanoramicViewLast-MileNavigation benchmark. We compare automatic evaluations against human judgments for Veo-3 across four hard-level dimensions.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Success (3D) Original Destination

Destination Integrity Score

Scene Consistency Score

Spatial Alignment Score

Object Semantic Score

Agent Consistency Score

Oracle Success Score (3D)

Success Score (3D)

Physics Validness

Overall Success

Evaluation

Environmental Complexity

- Level: floor01 Auto Evaluation 90.00% 90.00% 93.33% 93.33% 93.33% 90.00% 98.33% 90.00% 81.67% 73.33% Human Evaluation 73.33% 75.00% 88.33% 95.00% 96.67% 55.00% 66.67% 31.67% 25.00% 25.00%
- Level: floor02plus Auto Evaluation 58.33% 66.67% 95.00% 93.33% 91.67% 55.00% 91.67% 55.00% 83.33% 46.67% Human Evaluation 66.67% 68.33% 86.67% 90.00% 96.67% 65.00% 61.67% 33.33% 36.67% 28.33%

View Fidelity

- Level: quality03 Auto Evaluation 72.50% 80.00% 92.50% 92.50% 90.00% 70.00% 92.50% 70.00% 77.50% 55.00% Human Evaluation 77.50% 77.50% 87.50% 92.50% 95.00% 65.00% 67.50% 42.50% 35.00% 32.50%
- Level: quality04 Auto Evaluation 75.00% 75.00% 95.00% 95.00% 90.00% 75.00% 97.50% 75.00% 82.50% 62.50% Human Evaluation 65.00% 67.50% 92.50% 92.50% 100.00% 57.50% 67.50% 27.50% 32.50% 27.50%
- Level: quality05 Auto Evaluation 75.00% 80.00% 95.00% 92.50% 97.50% 72.50% 95.00% 72.50% 87.50% 62.50% Human Evaluation 67.50% 70.00% 82.50% 92.50% 95.00% 57.50% 57.50% 27.50% 25.00% 20.00%

Trajectory Distance

Level: short

Auto Evaluation 81.67% 83.33% 96.67% 93.33% 90.00% 80.00% 98.33% 80.00% 83.33% 66.67% Human Evaluation 83.33% 85.00% 91.67% 98.33% 95.00% 76.67% 65.00% 48.33% 45.00% 41.67%

Level: long

Auto Evaluation 66.67% 73.33% 91.67% 93.33% 95.00% 65.00% 91.67% 65.00% 81.67% 53.33% Human Evaluation 56.67% 58.33% 83.33% 86.67% 98.33% 43.33% 63.33% 16.67% 16.67% 11.67%

Destination Specification

Level: color mark

Auto Evaluation 83.33% 88.33% 96.67% 93.33% 93.33% 80.00% 93.33% 80.00% 85.00% 70.00% Human Evaluation 86.67% 86.67% 83.33% 90.00% 96.67% 60.00% 80.00% 41.67% 40.00% 33.33%

Level: location description

Auto Evaluation 65.00% 68.33% 91.67% 93.33% 91.67% 65.00% 96.67% 65.00% 80.00% 50.00% Human Evaluation 53.33% 56.67% 91.67% 95.00% 96.67% 60.00% 48.33% 23.33% 21.67% 20.00%

- 10.4 Case Study: Qualitative Failure Modes

This case study, visualized in Figure 18, evaluates the “world modeling” capabilities of video generation models during a last-mile navigation task. In this scenario, a humanoid robot must traverse a 3D environment to reach a specific target marked on the floor. The analysis reveals that while models can generate plausible motion, they suffer from distinct breakdowns in object permanence, physical laws, and instruction grounding.

Veo-3(TemporalIdentityCollapse). Initially, Veo-3 demonstrates strong spatial reasoning. The agent successfully perceives a pillar as a solid obstacle and navigates around it, exhibiting valid collision avoidance. However, the generation fails critically in Agent Consistency at the 5-second mark. The model suffers from a “doppelgänger” hallucination, where a second agent spontaneously manifests and approaches the camera while the initial agent retreats. This loss of agent identity results in an Oracle Success Score of 0, as the intended actor never completes the trajectory.

Sora-2 (Semantic and Physical Drift). Sora-2 struggles to maintain the fundamental constraints of the scene. By Frame 2s, it exhibits severe semantic drift, abruptly shifting the visual style and ungrounding the instruction:

Input Image

Prompt Text:

Create a photorealistic video depicting a humanoid robot performing "last mile" navigation. The robot must traverse a static, 3D panoramic environment populated with obstacles (furniture/walls) to navigates directly toward a bright red (#ff0000) target area marked on the floor. The camera must move smoothly to follow the robot, in a Third-person, "over-the-shoulder" view, adjusting pan and tilt to keep both the agent (foreground) and the red target (background) continuously in frame.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Veo-3

Frame 4s

Frame 1s Frame 2s

Object Semantic Score = 1 The agent recognized the pillar is an impassable obstacle.

Object Semantic Score = 1 The agent completely walked past the pillar. No touch, no collision.

Scene Consistency Score = 1 It maintained the input scene detail and the scene style.

[Figure 162]

[Figure 163]

[Figure 164]

Frame 5s Frame 6s Frame 8s

Agent Consistency Score = 0 One new agent suddenly occurs.

Spatial Alignment Score = 0 The new agent approaches frontally while the initial agent retreats; both maintain precise heading-to-movement synchronization relative to the camera's viewpoint.

(Oracle) Success Score (3D) = 0 The initial agent did not arrive at the destination and didn’t ever arrive there, though the created agent did.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Sora-2

Frame 2s Frame 4s

Frame 6s

Scene Consistency Score = 0 It created a new scene in a new video style.

Destination Integrity Score = 0 It revised the red marked area from on the floor to on the wall.

Object Semantic Score = 0 The agent stood on a ‘air’ platform near the stairs.

[Figure 169]

[Figure 170]

[Figure 171]

Frame 8s Frame 10s Frame 12s

(Oracle) Success Score (3D) = 0 The agent did not arrive at the destination, and didn’t ever arrive there.

Object Semantic Score = 0 The agent walked through the railing.

Agent Consistency Score = 1 The agent walked with smooth motion, no teleportation, no frame jumps, and consistent scale.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

###### Wan2.2

Frame 0s Frame 1s

Frame 2s

Scene Consistency Score (tmp) = 1 It maintained the input scene detail and the scene style.

Object Semantic Score = 1 The agent recognized the pillar is an impassable obstacle.

Scene Consistency Score = 0 As the camera unreasonably rose in height, it removed the glass door and created some pillars inside.

[Figure 176]

[Figure 177]

[Figure 178]

Frame 3s Frame 4s Frame 5s

Destination Integrity Score = 1 Although the indoor scene has changed, the location of the marked area remains the same.

Agent Consistency Score = 1 The agent walked with smooth motion, no teleportation, no frame jumps, and consistent scale.

(Oracle) Success Score (3D) = 0 The agent did not arrive at the destination although it is very close.

- Figure 18 Qualitative Comparison on Panoramic View Last-Mile Navigation. We analyze three models (Veo-3, Sora-2, and Wan-2.2) navigating toward a red target. The figure highlights distinct failure modes: Veo-3 suffers from agent inconsistency (hallucinating a second agent), Sora-2 exhibits severe geometric and physical violations (style shift, clipping through railings), and Wan-2.2 struggles with scene stability (altering structural elements like doors and pillars) despite smooth agent motion.

Start Position: The blue triangle Goal: The red area

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Frame 5 Frame 6 Frame 7 Frame 8

- Figure 19 Top-down View Real-World Navigation task showing bird’s-eye view navigation. The model must interpret the abstract 2D map representation and plan a path from start to goal, demonstrating 2D spatial reasoning and temporal reasoning from a global perspective. A successful case completed by Veo-3 on the Top-down View Real-World Navigation task.

the model incorrectly remaps the red target marker from the floor to a vertical wall. Furthermore, the physics engine collapses; the agent is observed floating on a non-existent “air platform” and subsequently clipping through a solid railing. This confirms that Sora-2 prioritizes visual fluidity over collision geometry or logical consistency.

Wan-2.2 (Environmental Instability). In contrast to the others, Wan-2.2 maintains high Agent Consistency, producing smooth, continuous motion without teleportation or scaling artifacts. However, it fails to maintain the “stage” around the actor. As the camera angle shifts (Frame 2s), the model suffers from background object permanence failures, inexplicably erasing a glass door and hallucinating new pillars within the room. While it preserves Destination Integrity better than Sora-2 (keeping the target relative to the agent), the volatility of the environment ultimately prevents successful navigation.

### 11 Top-down View Real-World Navigation (T.V.R.-W.Nav.)

- 11.1 Task Definition

The Top-down Real-World Navigation task benchmarks a model’s ability to synthesize advanced spatial reasoning with semantic understanding in complex, human-centric environments. This task requires models to interpret diverse and partially occluded top-down layouts—such as floor plans—to generate optimal trajectories. Simultaneously, it tests robust semantic grounding by demanding goal identification based on textual descriptions of varying abstraction.

- 11.2 Evaluation and Metrics

We adopt the shared embodied evaluation protocol in Section 9.3: Physical Understanding and Instruction Following checks (Object Semantic, Agent Consistency, Spatial Alignment, Destination Integrity, Scene Consistency) are unchanged. Here we only note how task completeness metrics are used for Top-down View Real-World Navigation and how the gated scores are formed.

- 11.2.1 Task Completeness Metrics (Geometry Only)

#### These metrics isolate navigational accuracy, evaluating whether the agent reaches the destination based purely on geometric coordinates. Visual fidelity and physical consistency are excluded here and assessed by subsequent metrics.

- • Success Score 2D (S.S. 2D). This metric evaluates whether the agent terminates its trajectory strictly within the designated goal region in the top-down 2D view. Success is achieved if and only if the agent’s final position is contained entirely within the goal footprint.
- • Oracle Success Score 2D (O.S. 2D). This metric determines if the agent encounters the goal region at any point during the rollout, regardless of the stopping location. Credit is awarded if the agent’s trajectory intersects with or passes adjacent to the 2D goal boundary at any timestep.

- 11.2.2 Gate Metrics and Holistic Performance

As in Panoramic Last-Mile Navigation, gates guard against generative shortcuts, but here the challenge comes from the 2D top-down view: success must hold on a static map, and Physical Validness is stricter because object recognition and agent orientation are blurrier from overhead.

- • Success (2D) Original Destination. This composite metric mitigates the tendency of video generation models to hallucinate solution shortcuts. It is satisfied if and only if the Success Score 2D, Destination Integrity Score, and Scene Consistency Score are met simultaneously. This ensures that successful arrival is valid and not the result of the model altering the scene layout or destination coordinates to simplify the task.
- • Physical Validity. To enforce physical realism, this composite metric acts as a strict consistency check. It is satisfied only when the Object Semantic Score, Agent Consistency Score, and Spatial Alignment Score are simultaneously met. This precludes physical anomalies such as object clipping, teleportation, or unnatural drift.
- • Overall Success = Success Score 2D ∧ Oracle Success Score 2D ∧ Object Semantic ∧ Agent Consistency ∧ Spatial Alignment ∧ Destination Integrity ∧ Scene Consistency; a sample passes only when all seven binary checks are 1.

Finally, across all embodied navigation tasks, we introduce the Overall metric to evaluate holistic performance. A sample is considered successful only if all fine-grained evaluation metrics are satisfied. This strict criterion demonstrates that strong performance on individual sub-metrics does not guarantee successful end-to-end task completion.

- 11.3 Evaluation Results

Key Finding: The Reality & Modality Gap

- • Physics Hallucination: Video models like Veo-3 excel at path coherence but fail at physical realism. Automatic metrics miss this, inflating success rates (37.14%) compared to human judgment (10.34%) by ignoring violations like wall-clipping.
- • The Semantic Cliff: Performance drops by >3× when instructions shift from visual markers to textual descriptions. Models struggle significantly to ground abstract linguistic commands into spatial actions.

The Top-down Real-World Navigation task evaluates a model’s ability to synthesize spatial planning with semantic understanding. The following analysis dissects model performance across environmental complexities, instruction modalities, and visual fidelities, followed by a critical examination of automatic metric reliability against human judgment.

- 11.3.1 VLM-Based Evaluation

Video vs. Image-based Navigation. Table 28 establishes Veo-3 as the state-of-the-art model for this benchmark, consistently outperforming peers in navigation success and trajectory adherence.

- • The Temporal Advantage: In the baseline setting (floor01), Veo-3 achieves a Success Score (2D) of 65.71%, significantly surpassing the image-based Nano-banana (41.67%) and the video-based Sora-2 (22.81%). This advantage stems from Veo-3’s superior Spatial Alignment Score (91.43%), indicating that

- video generation models with strong temporal attention can better maintain trajectory coherence than frame-by-frame image generators.
- • TheSemanticTrade-off: Interestingly, while Nano-banana lags in navigation, it remains highly competitive in static recognition. It achieves an Object Semantic Score of 88.89% in simple environments—surpassing Veo-3 (77.14%) and Sora-2 (82.46%). This suggests that current image models excel at “what” is in the scene (grounding), while video models excel at “how” to move through it (dynamics).

Resilience to Complexity and Noise.

- • Environmental Complexity: Performance degrades universally as environments become more intricate. Transitioning from floor01 to floor02plus, Veo-3’s Overall Success drops sharply from 37.14% to 13.89%. However, its Oracle Success Score remains robust (83.33%), implying that the model often identifies the correct path but fails to execute it without violating physical constraints (e.g., collisions), a hypothesis supported by the low Physics Validness (33.33%).
- • View Fidelity: The results indicate a positive correlation between visual quality and navigation success for video models. Veo-3’s Overall Success improves from 20.83% in lower fidelity settings (quality03) to 29.17% in higher fidelity (quality04), suggesting that clearer visual cues are critical for maintaining the “driver” capability in video generation.

InstructionFollowing: TheModalityGap. A distinct performance gap exists between visual and textual grounding (see Destination Specification).

- • Visual Markers: When targets are specified via a simple “color mark,” Veo-3 achieves a robust Overall Success of 38.89%.
- • Textual Descriptions: When targets require semantic parsing (“location description”), success rates collapse. Veo-3 falls to 11.43%, and Nano-banana drops to 8.33%. This confirms that mapping abstract linguistic descriptions to spatial layouts remains a significant hurdle compared to direct visual matching.

- 11.3.2 Human vs. Automatic Evaluation Discrepancy

To validate the automatic benchmarking protocols, we conducted a side-by-side comparison with human evaluators for the top-performing model, Veo-3 (Table 29). The data reveals critical divergences in how algorithms versus humans perceive “success.”

The Physics Hallucination Problem. Automatic metrics consistently overestimate physical realism. In floor01, the Auto Metric reports a Physics Validness of 54.29%, while humans rate it at only 22.41%. This discrepancy leads to a massive inflation of the Overall Success metric in automatic evaluations (37.14%) compared to the human ground truth (10.34%). This indicates that while models satisfy geometric path constraints (high Spatial Alignment), they frequently hallucinate physically impossible traversals (e.g., clipping through walls) that simple 2D projection metrics fail to penalize.

The Consistency Paradox. Conversely, automatic metrics appear overly punitive regarding visual consistency.

- • Agent Consistency: Humans rated Veo-3’s agent consistency in floor01 at a near-perfect 96.55%, whereas the automatic scorer only awarded 62.86%.
- • Scene Consistency: Similarly, humans perceived the environment as stable (79.31%), significantly higher than the algorithmic assessment (45.71%).

This suggests that current computer vision metrics for temporal consistency (likely based on pixel-wise or feature-wise similarity) are sensitive to minor generative artifacts that human observers naturally filter out as temporally coherent motion.

Grounding Reliability. The evaluation gap widens with task abstractness. For simple “color mark” destinations, the Auto and Human Success Scores (2D) are relatively close (52.78% vs. 48.28%). However, for “location descriptions,” the auto metric claims a 37.14% success rate while humans find only 18.33% of trajectories successful. This implies that automatic evaluators are prone to false positives when validating complex semantic

##### Table 28 Quantitative results for the 2D Top-down Navigation benchmark. We compare Sora-2, Veo-3, Nano-banana, and GPT-4o-image across various dimensions.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Scene Consistency Score

Destination Integrity Score

Spatial Alignment Score

Object Semantic Score

Agent Consistency Score

Oracle Success Score (2D)

Success Score (2D)

Physics Validness

Overall Success

Model

Environmental Complexity

- Level: floor01 Video Models

Veo-3 65.71% 85.71% 77.14% 62.86% 91.43% 65.71% 45.71% 54.29% 37.14% Sora-2 22.81% 47.37% 82.46% 68.42% 78.95% 7.02% 8.77% 52.63% 1.75%

Image Models

Nano-banana 41.67% 52.78% 88.89% 50.00% 77.78% 33.33% 38.89% 50.00% 5.56% GPT-4o-image 10.34% 15.52% 55.17% 6.90% 56.90% 1.72% 1.72% 6.90% 1.72%

- Level: floor02plus Video Models

Veo-3 25.00% 83.33% 61.11% 44.44% 80.56% 33.33% 45.71% 33.33% 13.89% Sora-2 28.81% 55.93% 89.83% 54.24% 86.44% 15.25% 23.73% 49.15% 5.08%

Image Models Nano-banana 38.89% 52.78% 86.11% 55.56% 86.11% 36.11% 66.67% 50.00% 16.67% GPT-4o-image 16.67% 23.33% 51.67% 11.67% 41.67% 5.00% 10.00% 10.00% 5.00%

View Fidelity

- Level: quality03 Video Models

Veo-3 45.83% 79.17% 66.67% 33.33% 75.00% 50.00% 43.48% 29.17% 20.83% Sora-2 34.21% 52.63% 92.11% 63.16% 81.58% 18.42% 18.42% 57.89% 7.89%

Image Models Nano-banana 41.67% 58.33% 83.33% 45.83% 75.00% 41.67% 58.33% 37.50% 12.50% GPT-4o-image 21.05% 23.68% 65.79% 13.16% 57.89% 5.26% 5.26% 10.53% 5.26%

- Level: quality04 Video Models

Veo-3 37.50% 83.33% 70.83% 58.33% 95.83% 41.67% 45.83% 54.17% 29.17% Sora-2 23.08% 46.15% 87.18% 66.67% 89.74% 7.69% 17.95% 51.28% 2.56%

Image Models

Nano-banana 37.50% 50.00% 87.50% 50.00% 83.33% 33.33% 37.50% 50.00% 4.17% GPT-4o-image 12.50% 17.50% 52.50% 12.50% 50.00% 5.00% 10.00% 12.50% 5.00%

- Level: quality05 Video Models

Veo-3 52.17% 91.30% 69.57% 69.57% 86.96% 56.52% 47.83% 47.83% 26.09% Sora-2 20.51% 56.41% 79.49% 53.85% 76.92% 7.69% 12.82% 43.59% 0.00%

Image Models Nano-banana 41.67% 50.00% 91.67% 62.50% 87.50% 29.17% 62.50% 62.50% 16.67% GPT-4o-image 7.50% 17.50% 42.50% 2.50% 40.00% 0.00% 2.50% 2.50% 0.00%

Trajectory Distance

Level: noturn

Video Models Veo-3 36.11% 83.33% 61.11% 52.78% 83.33% 38.89% 54.29% 38.89% 25.00% Sora-2 29.31% 60.34% 89.66% 62.07% 84.48% 15.52% 18.97% 48.28% 1.72%

Image Models

Nano-banana 41.67% 52.78% 88.89% 50.00% 83.33% 36.11% 52.78% 44.44% 11.11% GPT-4o-image 20.34% 23.73% 57.63% 10.17% 57.63% 5.08% 8.47% 10.17% 5.08%

Level: oneturn

Video Models Veo-3 54.29% 85.71% 77.14% 54.29% 88.57% 60.00% 37.14% 48.57% 25.71% Sora-2 22.41% 43.10% 82.76% 60.34% 81.03% 6.90% 13.79% 53.45% 5.17%

Image Models

###### Nano-banana 38.89% 52.78% 86.11% 55.56% 80.56% 33.33% 52.78% 55.56% 11.11% GPT-4o-image 6.78% 15.25% 49.15% 8.47% 40.68% 1.69% 3.39% 6.78% 1.69%

Destination Specification

Level: color mark

Video Models Veo-3 52.78% 91.67% 80.56% 69.44% 91.67% 58.33% 63.89% 61.11% 38.89% Sora-2 17.24% 53.45% 77.59% 56.90% 77.59% 12.07% 24.14% 44.83% 1.72%

Image Models Nano-banana 58.33% 77.78% 91.67% 44.44% 86.11% 55.56% 55.56% 41.67% 13.89% GPT-4o-image 10.34% 13.79% 39.66% 3.45% 34.48% 1.72% 3.45% 3.45% 1.72%

Level: location description

Video Models

###### Veo-3 37.14% 77.14% 57.14% 37.14% 80.00% 40.00% 26.47% 25.71% 11.43% Sora-2 34.48% 50.00% 94.83% 65.52% 87.93% 10.34% 8.62% 56.90% 5.17%

Image Models

###### Nano-banana 22.22% 27.78% 83.33% 61.11% 77.78% 13.89% 50.00% 58.33% 8.33% GPT-4o-image 16.67% 25.00% 66.67% 15.00% 63.33% 5.00% 8.33% 13.33% 5.00%

##### Table29 Quantitative results for the Top-downViewReal-WorldNavigation benchmark. We compare automatic evaluations against human judgments for Veo-3 across the same hard-level dimensions.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Scene Consistency Score

Success (2D) Original Destination

Agent Consistency Score

Spatial Alignment Score

Destination Integrity Score

Success Score (2D)

Oracle Success Score (2D)

Object Semantic Score

Physics Validness

Overall Success

Evaluation

Environmental Complexity

- Level: floor01 Auto Evaluation 65.71% 85.71% 77.14% 62.86% 91.43% 65.71% 45.71% 42.86% 54.29% 37.14% Human Evaluation 41.38% 72.41% 51.72% 96.55% 55.17% 72.41% 79.31% 34.48% 22.41% 10.34%
- Level: floor02plus Auto Evaluation 25.00% 83.33% 61.11% 44.44% 80.56% 33.33% 45.71% 13.89% 33.33% 13.89% Human Evaluation 25.00% 58.33% 25.00% 45.00% 61.67% 58.33% 83.33% 16.67% 11.67% 1.67%

View Fidelity

- Level: quality03 Auto Evaluation 45.83% 79.17% 66.67% 33.33% 75.00% 50.00% 43.48% 25.00% 29.17% 20.83% Human Evaluation 34.21% 57.89% 47.37% 63.16% 63.16% 65.79% 84.21% 31.58% 18.42% 5.26%
- Level: quality04 Auto Evaluation 37.50% 83.33% 70.83% 58.33% 95.83% 41.67% 45.83% 29.17% 54.17% 29.17% Human Evaluation 35.00% 70.00% 20.00% 72.50% 57.50% 65.00% 77.50% 22.50% 10.00% 2.50%
- Level: quality05 Auto Evaluation 52.17% 91.30% 69.57% 69.57% 86.96% 56.52% 47.83% 30.43% 47.83% 26.09% Human Evaluation 30.00% 67.50% 47.50% 75.00% 55.00% 65.00% 82.50% 22.50% 22.50% 10.00%

Trajectory Distance

Level: short Auto Evaluation 36.11% 83.33% 61.11% 52.78% 83.33% 38.89% 54.29% 22.00% 33.90% 15.30% Human Evaluation 28.81% 61.02% 45.76% 69.49% 50.85% 66.10% 83.05% 20.34% 13.56% 8.47%

Level: long Auto Evaluation 54.29% 85.71% 77.14% 54.29% 88.57% 60.00% 37.14% 32.20% 39.00% 23.70% Human Evaluation 37.29% 69.49% 30.51% 71.19% 66.10% 64.41% 79.66% 30.51% 20.34% 3.39%

Destination Specification

Level: color mark Auto Evaluation 52.78% 91.67% 80.56% 69.44% 91.67% 58.33% 63.89% 41.67% 61.11% 38.89% Human Evaluation 48.28% 81.03% 34.48% 74.14% 65.52% 87.93% 94.83% 39.66% 16.67% 6.90%

Level: location description Auto Evaluation 37.14% 77.14% 57.14% 37.14% 80.00% 40.00% 26.47% 14.29% 25.71% 11.43% Human Evaluation 18.33% 50.00% 41.67% 66.67% 51.67% 43.33% 68.33% 11.67% 17.22% 5.00%

grounding, likely crediting trajectories that end near the target by chance rather than by understanding the textual clue.

- 11.4 Case Study: Qualitative Failure Modes

This analysis, presented in Figure 20, examines the performance of three video generation models—Veo-3, Sora-2, and Wan-2.2—on a top-down embodied navigation task. Each model is instructed to generate a video depicting an agent navigating from a blue triangular start position to a red target within a static environment, enabling a direct comparison of their spatial reasoning, trajectory consistency, and goal-directed behavior.

Veo-3: Strong Spatial Grounding with Control and Termination Failures. Veo-3 exhibits the strongest initial spatial grounding among the three models, correctly identifying the agent’s starting position at the blue triangle and demonstrating partial semantic awareness of obstacles. In particular, the agent recognizes the sofa as a navigable obstruction and initially routes around it. However, the generation deteriorates due to failures in movement physics and action termination. The agent displays a critical orientation error, moving laterally to the left while facing forward, which results in a Spatial Alignment Score of 0 at Frame 5s. Although the agent briefly enters the red-marked target region (Oracle Success Score = 1), it fails to terminate upon arrival. Instead, the trajectory continues beyond the goal, deviates unpredictably, and ultimately collides with the sofa. This post-arrival drift causes both the Success Score and Object Semantic Score to drop to 0, revealing a failure to couple goal completion with action cessation.

Sora-2: Severe Scene Hallucination and Physical Constraint Violations. Sora-2 fails to preserve both the spatial ground truth and the physical constraints of the input environment. From the outset, the model reconstructs an entirely new scene in which the agent no longer originates from the blue start location. Temporal consistency further collapses as the generation introduces severe hallucinations, including the spontaneous appearance of a second agent at Frame 4s. The agent also violates fundamental physical boundaries by passing through solid obstacles (the black block), yielding an Object Semantic Score of 0. While an agent eventually reaches a red-

Input Image

Prompt Text:

[Figure 187]

Create a video showing an agent navigating a 2D top-down environment. The robot begins positioned on a bright blue (#0000ff) triangular marker, with its front aligned to the triangle‘s pointing vertex. It travels smoothly and efficiently toward a static red (#ff0000) target area. As the robot agent moves, it leaves a permanent, bright green (#00ff00) trajectory line tracing its exact path from the start point. The camera is single, strictly static and fixed top-down view that captures the full environment.

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Veo-3

Frame 1s Frame 2s

Frame 4s

Spatial Alignment Score (tmp) = 1 It knows where the starting point is， although it regenerates the agent‘s shape and this is not the shortest route.

Object Semantic Score (tmp) = 1 The agent found the sofa an obstacle, and bypassed it.

Agent Consistency Score = 1 The agent moved continuously, despite the disconnected trajectory.

[Figure 192]

[Figure 193]

[Figure 194]

Frame 5s Frame 6s Frame 7s

Spatial Alignment Score = 0 The agent moved horizontally to the left, which is inconsistent with its head facing forward.

Oracle Success Score (2D) = 1 The agent arrived the red marked area. But it created an unexpected green trajectory.

Success Score (2D) = 0, Object Semantic Score = 0 The agent did not stop there, leaved the destination, and hit the sofa.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Sora-2

Frame 0s Frame 2s

Frame 4s

Scene Consistency Score = 0, Spatial Alignment Score = 0 It created a new scene, and the agent didn’t start from blue area.

Agent Consistency Score = 1 The agent recorded its turning around motion.

Agent Consistency Score = 0 One new agent suddenly occurs.

[Figure 199]

[Figure 200]

[Figure 201]

Frame 6s Frame 8s Frame 10s

Destination Integrity Score = 1 The location of the marked area remains the same.

Object Semantic Score = 0 The agent walked through the wall (the black block).

(Oracle) Success Score (2D) = 1 The initial agent arrived the red marked area.

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

Wan2.2

Frame 0s Frame 1s

Frame 2s

Scene Consistency Score = 1 It maintained the input scene detail and the scene style.

Spatial Alignment Score = 0 It created a new agent from the dinning table and misunderstood the start point.

Agent Consistency Score = 0 The agent moved in a drifting, disjointed manner with legs.

[Figure 206]

[Figure 207]

[Figure 208]

Frame 3s Frame 4s Frame 5s

Destination Integrity Score = 1 Although the indoor scene has changed, the location of the marked area remains the same.

(Oracle) Success Score (2D) = 0 The agent did not arrive at the destination although it is very close.

Destination Integrity Score = 0 It revised the red marked area from on the floor to a corner.

##### Figure 20 Qualitative Comparison on Top-down View Real-World Navigation. We analyze three models (Veo-3, Sora-2, and Wan-2.2) navigating toward a red target.

marked region, the complete loss of scene integrity and environmental correspondence renders the navigation invalid with respect to the original prompt, highlighting a failure in maintaining world-state continuity.

Wan-2.2: Surface-Level Scene Consistency with Semantic Role Confusion. Wan-2.2 preserves the visual style and low-level layout of the scene more effectively than Sora-2, achieving an initial Scene Consistency Score of 1. However, this apparent stability masks a deeper semantic failure. The model misidentifies the controllable agent, erroneously animating a dining table as the acting entity, which immediately induces spatial misalignment. The resulting motion is disjointed and drifting, lacking coherent locomotion or intentional control. Compounding this issue, the model fails to maintain destination integrity: the red target region shifts from its original floor location to a corner of the scene, causing the Destination Integrity Score to drop to 0. As a result, the agent never reaches the correct target, demonstrating that visual fidelity alone is insufficient for semantically grounded navigation.

### 12 3D Real-World Navigation (3D R.-W.Nav.)

- 12.1 Task Definition

This task evaluates an agent’s foundational capabilities in 3D Spatial Reasoning and Visual-Semantic Grounding within real-world scanned environments (Chang et al., 2017; Ramakrishnan et al., 2021b; Savva et al., 2019; Anderson et al., 2018b; Zhu et al., 2017). It probes the agent’s ability to interpret egocentric visual streams and parse complex 3D environmental geometry using datasets such as Matterport3D, HM3D, and Habitat (Chang et al., 2017; Ramakrishnan et al., 2021b,a; Chaplot et al., 2020a). Furthermore, the task challenges the model’s sequential decision-making by requiring the generation of valid action trajectories (e.g., move forward, turn left) and tests its semantic reasoning by demanding the identification of goals based on varied abstract or textual descriptions.

Start Position: The robot’s position Goal: The red area

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Frame 5 Frame 6 Frame 7 Frame 8

- Figure 21 3D Real-World Navigation task showing navigation through a realistic 3D environment. The model must navigate from a starting position to a goal location using visual cues from the third-person perspective. A successful case completed by Veo-3 on the 3D Real-World Navigation task.

- 12.2 Evaluation and Metrics

We adopt the shared embodied evaluation protocol in Section 9.3: all metrics are binary and labeled via the automatic VLM/human rules on rollout traces and generated frames. Physical Understanding and Instruction Following checks (Object Semantic, Agent Consistency, Spatial Alignment, Destination Integrity, Scene Consistency) are unchanged; here we outline how they are applied in 3D Real-World Navigation and how the gates are formed.

- 12.2.1 Task Completeness Metrics (Geometry Only)

We report Success Score 3D and Oracle Success Score 3D as in Section 9.3, testing whether the agent stops inside or ever enters the destination volume. These ignore visual fidelity and physical plausibility, which the gate metrics enforce.

- • Success Score 3D (S.S. 3D). This checks whether the agent stops at the correct location in the 3D navigation sequence. The evaluation is geometric only and does not depend on whether the destination appearance is preserved. It is satisfied if the agent stops inside the intended destination volume.
- • Oracle Success Score 3D (O.S. 3D). This credits the agent if it ever comes within a proximity threshold of the correct 3D destination, regardless of its final stop. It is satisfied if the agent reaches the destination vicinity at any time during the video.

- 12.2.2 Gate Metrics and Holistic Performance

In 3D R.-W.Nav., the gates surface common failure modes: agents may “teleport” between floors, cut through occluded rooms, or alter the dollhouse layout to satisfy geometric hits. Requiring destination integrity and scene consistency prevents shortcutting via hallucinated staircases or duplicated goals, while the physics gate catches identity drift and implausible motion in the third-person view.

As with the Panoramic and Top-down tasks, strict conjunctions block generative shortcuts (e.g., hallucinating nearer goals or warping 3D structure), but the 3D setting stresses multi-floor occlusions and third-person viewpoint drift:

- • Success (3D) Original Destination. This composite metric addresses the tendency of video generation models to hallucinate solution shortcuts. It is satisfied only if the Success Score 3D, Destination Integrity Score, and Scene Consistency Score are all met simultaneously. This metric ensures that a successful arrival is valid and not the result of the model altering the scene layout or destination location to simplify the task.
- • Physics Validness. To provide a holistic view of physical understanding, this composite metric serves as a strict gate. It is satisfied only if the Object Semantic Score, Agent Consistency Score, and Spatial Alignment Score are all met simultaneously. This ensures the agent does not clip through objects, teleport, or drift unnaturally.
- • Overall Success = Success Score 3D ∧ Oracle Success Score 3D ∧ Object Semantic ∧ Agent Consistency ∧ Spatial Alignment ∧ Destination Integrity ∧ Scene Consistency; a sample passes only when all seven binary checks are 1.

- 12.3 Evaluation Results

Key Finding: The “Simulation Gap” in Video Generation

#### Current video generation models (Veo-3, Sora-2) exhibit a fundamental disconnect between visual plausibility and physical causality. While they often achieve high raw success rates in simple navigation tasks (85–89%), they fail catastrophically on holistic metrics (0–3%) due to severe scene hallucination and physics violations. In contrast, image-based agents (Nano-banana) demonstrate robust “simulation” capabilities, maintaining near-perfect scene consistency (>95%) and physical validity.

- 12.3.1 VLM-Based Evaluation

- Table 30 benchmarks Veo-3 against Sora-2, Nano-banana, and GPT-4o-image. The results highlight a distinct hierarchy in embodied navigation capabilities, separated by model modality and architectural stability.

Image Models vs. Video Models (Nano-banana Dominance). Nano-banana emerges as the distinct leader, showcasing that image generation is currently far more reliable for embodied tasks than video generation.

##### Table 30 Quantitative results for the 3D Real-world Navigation benchmark. We compare Sora-2, Veo-3, Nano-banana, and GPT-4o-image across the same hard-level dimensions.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Success (3D) Original Destination

Destination Integrity Score

Scene Consistency Score

Spatial Alignment Score

Object Semantic Score

Agent Consistency Score

Oracle Success Score (3D)

Success Score (3D)

Physics Validness

Overall Success

Evaluation

Environmental Complexity

###### Level: floor01

Video Models

###### Veo-3 85.00% 91.67% 81.67% 86.67% 63.33% 53.33% 15.00% 11.67% 40.00% 3.33% Sora-2 88.89% 97.22% 72.22% 83.33% 97.22% 77.78% 44.44% 0.00% 55.00% 0.00%

Image Models

Nano-banana 75.00% 75.00% 100.00% 97.22% 97.22% 75.00% 86.11% 75.00% 97.22% 75.00% GPT-4o-image 13.33% 13.33% 88.33% 48.33% 58.33% 13.33% 21.67% 13.33% 35.00% 13.33%

Level: floor02plus

Video Models

###### Veo-3 68.33% 78.33% 75.00% 68.33% 73.33% 38.33% 18.33% 5.00% 33.33% 0.00% Sora-2 72.22% 77.78% 69.44% 75.00% 83.33% 58.33% 36.11% 0.00% 60.00% 0.00%

Image Models

###### Nano-banana 83.33% 86.11% 94.44% 97.22% 94.44% 86.11% 100.00% 83.33% 94.44% 83.33% GPT-4o-image 15.00% 16.67% 75.00% 56.67% 51.67% 15.00% 15.00% 13.33% 40.00% 13.33%

View Fidelity

###### Level: quality03

Video Models

Veo-3 65.00% 77.50% 72.50% 75.00% 70.00% 40.00% 15.00% 5.00% 30.00% 0.00% Sora-2 75.00% 83.33% 79.17% 83.33% 91.67% 66.67% 41.67% 0.00% 67.50% 0.00%

Image Models Nano-banana 75.00% 79.17% 95.83% 95.83% 91.67% 79.17% 91.67% 75.00% 91.67% 75.00% GPT-4o-image 7.50% 7.50% 80.00% 45.00% 52.50% 5.00% 7.50% 5.00% 30.00% 5.00%

Level: quality04

Video Models

###### Veo-3 82.50% 85.00% 75.00% 85.00% 65.00% 50.00% 20.00% 10.00% 42.50% 2.50% Sora-2 91.67% 95.83% 54.17% 75.00% 83.33% 70.83% 41.67% 0.00% 45.00% 0.00%

Image Models

Nano-banana 87.50% 87.50% 100.00% 100.00% 100.00% 87.50% 91.67% 87.50% 100.00% 87.50% GPT-4o-image 17.50% 20.00% 80.00% 57.50% 52.50% 20.00% 27.50% 17.50% 40.00% 17.50%

Level: quality05

Video Models

###### Veo-3 82.50% 92.50% 87.50% 72.50% 70.00% 47.50% 15.00% 10.00% 37.50% 2.50% Sora-2 75.00% 83.33% 79.17% 79.17% 95.83% 66.67% 37.50% 0.00% 60.00% 0.00%

Image Models

###### Nano-banana 75.00% 75.00% 95.83% 95.83% 95.83% 75.00% 95.83% 75.00% 95.83% 75.00% GPT-4o-image 17.50% 17.50% 85.00% 55.00% 60.00% 17.50% 20.00% 17.50% 42.50% 17.50%

Trajectory Distance

###### Level: short

Video Models

###### Veo-3 73.33% 85.00% 81.67% 78.33% 68.33% 46.67% 16.67% 8.33% 38.33% 3.33% Sora-2 72.22% 83.33% 80.56% 77.78% 86.11% 63.89% 41.67% 0.00% 55.00% 0.00%

Image Models

Nano-banana 77.78% 80.56% 97.22% 97.22% 94.44% 80.56% 88.89% 77.78% 94.44% 77.78% GPT-4o-image 13.33% 13.33% 85.00% 56.67% 58.33% 11.67% 16.67% 11.67% 45.00% 11.67%

Level: long

Video Models

###### Veo-3 80.00% 85.00% 75.00% 76.67% 68.33% 45.00% 16.67% 8.33% 35.00% 0.00% Sora-2 88.89% 91.67% 61.11% 80.56% 94.44% 72.22% 38.89% 0.00% 60.00% 0.00%

Image Models

###### Nano-banana 80.56% 80.56% 97.22% 97.22% 97.22% 80.56% 97.22% 80.56% 97.22% 80.56% GPT-4o-image 15.00% 16.67% 78.33% 48.33% 51.67% 16.67% 20.00% 15.00% 30.00% 15.00%

Destination Specification

###### Level: color mark

Video Models

###### Veo-3 93.33% 96.67% 66.67% 73.33% 70.00% 31.67% 28.33% 15.00% 28.33% 3.33% Sora-2 88.89% 91.67% 58.33% 75.00% 88.89% 69.44% 36.11% 0.00% 51.67% 0.00%

Image Models

Nano-banana 80.56% 80.56% 100.00% 100.00% 100.00% 80.56% 91.67% 80.56% 100.00% 80.56% GPT-4o-image 16.67% 18.33% 71.67% 46.67% 45.00% 18.33% 20.00% 16.67% 30.00% 16.67%

Level: location description

Video Models

###### Veo-3 60.00% 73.33% 90.00% 81.67% 66.67% 60.00% 5.00% 1.67% 45.00% 0.00% Sora-2 72.22% 83.33% 83.33% 83.33% 91.67% 66.67% 44.44% 0.00% 63.33% 0.00%

Image Models

###### Nano-banana 77.78% 80.56% 94.44% 94.44% 91.67% 80.56% 94.44% 77.78% 91.67% 77.78% GPT-4o-image 11.67% 11.67% 91.67% 58.33% 65.00% 10.00% 16.67% 10.00% 45.00% 10.00%

- • Consistency as a Foundation: Nano-banana maintains near-perfect scores in Scene Consistency (86–100%) and Physics Validness (94–100%) across all complexity levels. This stability allows it to achieve high Overall Success rates (75–87.5%) that video models cannot approach.
- • Resilience to Complexity: As environment complexity increases (from floor01 to floor02plus), Nanobanana actually improves its overall success from 75% to 83.33%. In sharp contrast, video models collapse; Veo-3’s holistic success drops to 0% in complex multi-floor environments, unable to reconcile geometric consistency with longer navigation horizons.

Video Model Trade-offs (Veo-3 vs. Sora-2). While both video models struggle with holistic success (mostly 0%–3%), they exhibit different failure modes:

- • Sora-2 (Better Geometry, Worse Adherence): Sora-2 generally outperforms Veo-3 in Spatial Alignment (e.g., 97.22% vs 63.33% in floor01) and Scene Consistency (44.44% vs 15.00%). However, it suffers from a critical flaw in Gate Metric: Success (Original Destination), scoring 0.00% across almost all categories. This indicates that while Sora-2 generates smooth, consistent video, it hallucinates the destination or drifts significantly from the prompt’s specific target.
- • Veo-3 (Better Adherence, Worse Physics): Veo-3 is more “compliant” with the task, scoring higher on reaching the original destination (11.67% in floor01). However, it achieves this by sacrificing physical laws, as evidenced by its abysmal Scene Consistency scores (dropping to 5.00% in location tasks) and heavy penalization in human evaluations for physics violations.

Baseline Comparison. GPT-4o-image acts as a semantic baseline. While it demonstrates strong object recognition (Object Semantic Score ≈ 80–90%), it lacks the spatial reasoning to navigate, resulting in low success scores (≈ 13–17%). This confirms that embodied navigation requires more than just “seeing” the scene; it requires a consistent internal world model that GPT-4o-image lacks.

- Table 31 Quantitative results for the 3D Real-World Navigation benchmark. We compare automatic evaluations against human judgments for Veo-3 across the same hard-level dimensions.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Success Score (3D)

Oracle Success Score (3D)

Object Semantic Score

Agent Consistency Score

Spatial Alignment Score

Destination Integrity Score

Scene Consistency Score

Success (3D) Original Destination

Physics Validness

Overall Success

Evaluation

Environmental Complexity

- Level: floor01 Auto Evaluation 88.89% 97.22% 72.22% 83.33% 97.22% 77.78% 44.44% 38.89% 61.11% 25.00% Human Evaluation 85.00% 91.67% 81.67% 86.67% 63.33% 53.33% 15.00% 11.67% 40.00% 3.33%
- Level: floor02plus Auto Evaluation 72.22% 77.78% 69.44% 75.00% 83.33% 58.33% 36.11% 33.33% 50.00% 16.67% Human Evaluation 68.33% 78.33% 75.00% 68.33% 73.33% 38.33% 18.33% 5.00% 33.33% 0.00%

View Fidelity

- Level: quality03 Auto Evaluation 75.00% 83.33% 79.17% 83.33% 91.67% 66.67% 41.67% 37.50% 62.50% 29.17% Human Evaluation 65.00% 77.50% 72.50% 75.00% 70.00% 40.00% 15.00% 5.00% 30.00% 0.00%
- Level: quality04 Auto Evaluation 91.67% 95.83% 54.17% 75.00% 83.33% 70.83% 41.67% 41.67% 37.50% 12.50% Human Evaluation 82.50% 85.00% 75.00% 85.00% 65.00% 50.00% 20.00% 10.00% 42.50% 2.50%
- Level: quality05 Auto Evaluation 75.00% 83.33% 79.17% 79.17% 95.83% 66.67% 37.50% 29.17% 66.67% 20.83% Human Evaluation 82.50% 92.50% 87.50% 72.50% 70.00% 47.50% 15.00% 10.00% 37.50% 2.50%

Trajectory Distance

Level: short Auto Evaluation 72.22% 83.33% 80.56% 77.78% 86.11% 63.89% 41.67% 36.11% 63.89% 19.44% Human Evaluation 73.33% 85.00% 81.67% 78.33% 68.33% 46.67% 16.67% 8.33% 38.33% 3.33%

Level: long Auto Evaluation 88.89% 91.67% 61.11% 80.56% 94.44% 72.22% 38.89% 36.11% 47.22% 22.22% Human Evaluation 80.00% 85.00% 75.00% 76.67% 68.33% 45.00% 16.67% 8.33% 35.00% 0.00%

Destination Specification

Level: color mark Auto Evaluation 88.89% 91.67% 58.33% 75.00% 88.89% 69.44% 36.11% 36.11% 44.44% 16.67% Human Evaluation 93.33% 96.67% 66.67% 73.33% 70.00% 31.67% 28.33% 15.00% 28.33% 3.33%

Level: location description Auto Evaluation 72.22% 83.33% 83.33% 83.33% 91.67% 66.67% 44.44% 36.11% 66.67% 25.00% Human Evaluation 60.00% 73.33% 90.00% 81.67% 66.67% 60.00% 5.00% 1.67% 45.00% 0.00%

- 12.3.2 Human vs. Automated Evaluation Discrepancy

- Table 31 exposes a systemic bias in current automated benchmarking. While automated metrics suggest that Veo-3 possesses moderate embodied competency, human evaluation reveals a “competence illusion,” where high task completion rates mask fundamental failures in physical realism. The “Plausibility Gap” in Auto-Eval. Automated metrics exhibit severe inflation regarding agent reliability.

- • Metric Collapse: In the floor01 environment, the Auto Evaluation reports a respectable Holistic Metric (Overall Success) of 25.00%. Human evaluators, however, penalize the model rigorously, causing this metric to collapse to just 3.33%.
- • Superficial Success: Crucially, the raw Task Completeness (Success) scores are nearly identical between machines (88.89%) and humans (85.00%). This discrepancy proves that auto-evaluators correctly identify destination arrival but completely fail to penalize the impossible trajectories used to get there.

Blindness to Physics and Scene Integrity. The divergence stems from the auto-evaluator’s inability to detect violations of physical laws, creating a “physics-blind” assessment loop.

- • Scene Consistency: Human evaluators rate Scene Consistency drastically lower than the auto-evaluator (e.g., 15.00% vs. 44.44% in floor01). This indicates that the model frequently warps the environment—shifting walls, deleting obstacles, or altering lighting—to facilitate navigation, a behavior the auto-evaluator ignores.
- • The Validity Gate: While the auto-evaluator estimates Physics Validness at 50–60%, humans rate it between 30–40%. Because Overall Success is a gated metric (requiring both arrival and valid physics), this 20-point drop in validity effectively zeroes out the holistic success rate in human trials.

Overall Metric Analysis: The Consistency Bottleneck. The holistic data reveals a clear “Fragile Success” paradox. While the model achieves a high peak performance in isolation—scoring 80.56% in Success Score (3D)—it fails to integrate these capabilities into a coherent simulation.

- • The 69% Drop: Only 20.83% of samples satisfy the strict holistic criterion, representing a massive 69.45% degradation relative to the model’s best individual metric (Spatial Alignment, 90.28%).
- • Root Cause: The dominant limiting factor is Scene Consistency, which remains dangerously low at 40.28%. This confirms that the model is better at moving the agent than it is at maintaining the world.
- • Implication: Successful trajectories are frequently invalidated by environmental drift (e.g., vanishing objects, morphing geometry). This underscores that for high-quality motion generation is futile without the temporal stability required to keep the environment static.

- 12.4 Case Study: Qualitative Failure Modes

This case study in Figure 22 evaluates the ability of three video generation models—Veo-3, Sora-2, and Wan-2.2—to simulate a humanoid robot navigating a multi-story indoor space to a specific red target zone. The prompt requires strict adherence to physical constraints, step-by-step stair climbing, and a consistent “dollhouse” isometric view.

Veo-3 Analysis: Task Success with Physical Compromises. Veo-3 was the only model to achieve a successful arrival at the destination (Oracle Success Score = 1). However, this success came with significant hallucinations and physics violations:

- • Scene Consistency: The model failed to maintain the structural integrity of the input scene, notably hallucinating a straight staircase in place of the original curved one.
- • PhysicalFidelity: While the agent initially demonstrated understanding of vertical movement, it ultimately failed the physics constraint by jumping directly from the second floor to the ground floor, resulting in the body clipping into the floor. Despite these errors, the agent maintained consistency without teleporting earlier in the sequence.

[Figure 217]

Input Image

Prompt Text:

Create a video showing a humanoid robot successfully navigating from a starting point through a multistory indoor space. The robot obeys physical constraints and is the only object that moves in the entire video, beginning standing at the designated starting location on the floor toward a fixed bright red (#ff0000) region marked on the floor. When moving between floors, the robot climbs or descends the

staircase step-by-step, keeping its feet planted on individual treads. The camera is third-person isometric view (a "dollhouse" view).

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Veo-3

Frame 4s

Frame 1s Frame 2s

Scene Consistency Score = 0 It recreated a new straight staircase instead of the curved one.

Object Semantic Score (tmp) = 1 The agent understood the vertical movement on a stair.

Scene Consistency Score = 1 It maintained the input scene detail and the scene style.

[Figure 222]

[Figure 223]

[Figure 224]

Frame 5s Frame 6s Frame 8s

Agent Consistency Score = 1 The agent moved continuously without teleporting.

Object Semantic Score = 0 The agent jumped to the ground floor from the second floor, and its body was stuck on the floor.

(Oracle) Success Score (3D) = 1 The agent arrived the red marked area.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Sora-2

Frame 0s Frame 2s

Frame 6s

Scene Consistency Score = 0 It created a new scene in a new video style.

- Destination Integrity Score = 0 It revised the red marked area from on the floor to on the wall.

Object Semantic Score = 0 The agent stood on a ‘air’ platform near the stairs.

- Destination Integrity Score = 1 The red marked area did not change.

[Figure 229]

[Figure 230]

[Figure 231]

Frame 8s Frame 10s Frame 12s

(Oracle) Success Score (3D) = 0 The agent did not arrive at the destination, and didn’t ever arrive there.

Object Semantic Score = 0 The agent walked through the railing.

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

###### Wan2.2

Frame 0s Frame 1s

Frame 2s

Scene Consistency Score (tmp) = 1 It maintained the input scene detail and the scene style.

Scene Consistency Score = 0 The table on the right moved to the left.

Destination Integrity Score = 1 Although the indoor scene has changed, the location of the marked area remains the same.

[Figure 236]

[Figure 237]

[Figure 238]

Frame 3s Frame 4s Frame 5s

Agent Consistency Score = 0, Object Semantic Score = 0 The agent moved in a drifting, disjointed manner, and stood on a table.

Spatial Alignment Score = 0 The agent walked backwards down the stairs.

(Oracle) Success Score (3D) = 0 The agent did not arrive at the destination.

- Figure22 Case Study for 3D Real-World Navigation comparing Veo-3, Sora-2, and Wan-2.2. The visualization highlights model-specific failure modes: Veo-3 achieves the destination despite physics violations (floor jumping) and scene alterations (staircase geometry); Sora-2 suffers from severe scene hallucination and style drift; and Wan-2.2 exhibits temporal inconsistency with drifting agents and moving furniture.

Sora-2 Analysis: Severe Hallucination and Physics Failures. Sora-2 struggled significantly with both scene adherence and physical laws:

- • Style and Destination Drift: The model completely disregarded the input image, creating a new scene with a different video style. Crucially, it altered the task parameters by moving the red destination marker from the floor to the wall.
- • Physics Violations: The agent displayed zero understanding of solid geometry (Object Semantic Score = 0), standing on “air” platforms near the stairs and walking directly through the railing. Consequently, the agent never arrived at the valid destination.

Wan-2.2 Analysis: Temporal Instability and Agent Drifting. Wan-2.2 initially maintained the scene details well but quickly devolved into temporal incoherence:

- • Environmental Instability: The static environment proved unstable; by Frame 1s, furniture (a table) shifted position from right to left.
- • Agent Control: The navigation was disjointed. The agent drifted, stood on top of tables, and walked backwards down the stairs. This resulted in a failure to reach the destination.

While Veo-3 was the only model to “complete” the navigation task, all three models struggled with strict physical plausibility in a 3D space. Veo-3 prioritized path completion over geometry; Sora-2 hallucinated an entirely new reality; and Wan-2.2 failed to keep the static environment stationary.

### 13 Simultaneous Localization and Generation (SLAG)

- 13.1 Task Definition

The Simultaneous Localization and Generation (SLAG) task extends the paradigm of Simultaneous Localization and Mapping (SLAM) Durrant-Whyte & Bailey (2006) by coupling 3D spatial navigation with real-time generative mapping. Unlike traditional methods that estimate position within a static or progressively built map, SLAG actively synthesizes a synchronized 2D top-down trajectory that corresponds to the robot’s movement through a 3D environment. As a humanoid robot navigates complex indoor scenes—such as photorealistic apartments rendered in a “dollhouse” perspective—the system dynamically generates a 2D representation of its progress from start to goal. The core objective of SLAG is to achieve precise spatiotemporal alignment between physical 3D navigation and generative 2D plotting, demonstrating the synergy between real-time perception and generative modeling in spatial understanding.

Start Position: The robot’s position / blue triangle Goal: (Left) The red area, (Right) The same location aligned with Left

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Frame 5 Frame 6 Frame 7 Frame 8

- Figure 23 Simultaneous Localization and Goal-reaching (SLAG) task requiring the model to simultaneously maintain spatial awareness of its position while navigating toward a goal in an unknown environment.

- 13.2 Evaluation and Metrics

SLAG differs from the previous tasks by requiring simultaneous 3D navigation and 2D generative mapping, making trajectory alignment the primary bottleneck. Similarly, we adopt the shared embodied evaluation protocol in Section 9.3. Physical Understanding and Instruction Following checks (Object Semantic, Agent Consistency, Spatial Alignment, Destination Integrity, Scene Consistency) are unchanged. Here we focus on the SLAG-specific task completeness set and gates.

- 13.2.1 Task Completeness Metrics (Geometry Only)

Five task completeness metrics are evaluated jointly: Success Score 2D, Oracle Success Score 2D, Success Score 3D, Oracle Success Score 3D (all from Section 9.3), and Trajectory Alignment to ensure the projected 2D path matches the 3D motion. The 2D map is provided, but the destination is not; alignment hinges on correctly projecting the 3D navigation onto the given map, with 2D success conditioned on the same 3D destination.

- • SuccessScore(S.S.2D). Measures whether the agent’s final position lies within the highlighted or textually specified goal region in the 2D overhead map. The score is 1 if the final coordinates fall entirely inside the goal footprint; otherwise 0.
- • Oracle Success Score (O.S. 2D). Provides partial credit when the agent comes sufficiently close to the 2D goal during navigation. The score is 1 if the agent’s path ever intersects or touches the goal region, even if it does not stop there; otherwise 0.
- • Success Score (S.S. 3D). Checks whether the agent ends inside the correct destination volume in the 3D navigation sequence. This metric is purely geometric and independent of any visual discrepancies at the destination. The score is 1 if the final 3D position is within the target volume; otherwise 0.
- • Oracle Success Score (O.S. 3D). Grants credit when the agent enters the vicinity of the correct 3D destination at any point during its rollout. The score is 1 if the trajectory ever crosses the predefined proximity threshold around the target; otherwise 0.
- • Trajectory Alignment Score. Evaluates whether the agent’s 2D projected route is consistent with its 3D motion path, focusing on major turns and spatial transitions. A score of 1 indicates strong correspondence between the two trajectories; otherwise 0.

- 13.2.2 Gate Metrics and Holistic Performance

As in the prior tasks, gates block generative shortcuts, but only a 3D destination gate is possible because the 2D map has no fixed ground-truth target (it is generated, not given):

In SLAG, the holistic gate exposes compounded failures: models often align one panel but not the other, hallucinate extra corridors in the 2D drawing, or “teleport” in 3D while the 2D trace stays smooth. Because the 2D goal is synthesized, the 3D destination gate plus trajectory alignment are key to preventing such mismatches.

- • Success (3D) with Original Destination = Success Score 3D ∧ Destination Integrity ∧ Scene Consistency.
- • Physics Validness = Object Semantic ∧ Agent Consistency ∧ Spatial Alignment.
- • Overall Success = Success Score 2D ∧ Oracle Success Score 2D ∧ Success Score 3D ∧ Oracle Success Score 3D ∧ Trajectory Alignment ∧ Object Semantic ∧ Agent Consistency ∧ Spatial Alignment ∧ Destination Integrity ∧ Scene Consistency; a sample passes only when all ten binary checks are 1.

- 13.3 Evaluation Results

#### The Simultaneous Localization and Generation (SLAG) task represents a significant leap in embodied AI, requiring models to maintain strict temporal and spatial alignment between 3D physical navigation and 2D generative mapping. The following analysis dissects the performance of state-of-the-art models, revealing a critical trade-off between the visual fluidity of video models and the logical adherence of image-based models.

Key Findings: Reality & Modality Gap

- • Alignment bottleneck. Trajectory Alignment remains the weakest link for video baselines (44.07%– 55.17% vs. Nano-banana’s 88.89%/76.67%), and Destination Integrity stays low for them (20.34%– 44.83%). Even with strong Scene Consistency (45.76%–89.66%), cross-view synchronization caps holistic success.
- • Semantic cliff. Switching from color marks to textual location descriptions collapses Nano-banana’s 3D Success (81.25% → 17.65%) and Holistic score (50.00% → 8.82%); GPT-4o-image drops similarly (31.03% → 1.67%). Video models show the same gap (Veo-3: 17.24% → 5.17%, Sora-2: 19.64% → 6.67%), confirming spatial language grounding is the dominant failure mode.
- • Cross-viewpriorsactivateunderpressure. Forced 3D-to-2D alignment sustains Scene Consistency (Veo3: 45.76% → 57.89%, Nano-banana: 100.00% → 96.67%, GPT-4o-image: 98.33% → 94.83%) even when trajectories misalign, suggesting latent spatial priors surface only with structured multimodal conditioning.

- 13.3.1 VLM-Based Evaluation

- Table 32 highlights the performance disparities across Veo-3, Sora-2, Nano-banana, and GPT-4o-image. While video generation models often excel at visual fidelity, the requirement for synchronized spatial logic proves challenging.

The Logic vs. Motion Trade-off. Quantitative evaluation again ranks Nano-banana as the strongest SLAG performer: it posts the top Holistic Metric on floor01 (27.78%), edging out GPT-4o-image (25.00%), while Veo-3 and Sora-2 remain far lower (11.86% and 10.34%).

- • Trajectory Alignment Dominance: Nano-banana achieves a dominant Trajectory Alignment Score of 88.89% on floor01. Video baselines trail sharply (Veo-3: 44.07%, Sora-2: 55.17%), underscoring that smooth motion does not guarantee coordinate-level alignment with the map.
- • Gate-Induced Failures for Video Models: Despite respectable physical validity (Physics Validness: Veo-3 at 45.76%, Sora-2 at 65.52%), the Success (3D) Original Destination gate collapses to 18.64% and 22.41%, respectively, versus Nano-banana’s 38.89%. Low destination integrity (20.34%–32.76% for video models) is the primary culprit behind their depressed holistic scores.

The Grounding Gap: Visual vs. Linguistic Complexity. Performance across instruction types shows that linguistic grounding is a steeper barrier than visual grounding.

- • Drastic Drop on Text Prompts: When the destination is specified by a Color Mark, Nano-banana excels (81.25% 3D Success, 50.00% Holistic). Switching to a Location Description collapses 3D Success to 17.65% and Holistic to 8.82%.
- • Model Consistency: GPT-4o-image mirrors this trend, sliding from 44.83% to 3.33% (3D Success) and from 31.03% to 1.67% (Holistic). Video models show the same gap (Veo-3: 17.24% → 5.17% Holistic; Sora-2: 19.64% → 6.67%), confirming that textual spatial instructions remain the weakest link.

Environmental Scaling. As complexity increases from floor01 to floor02plus:

- • Nano-banana loses geometric reliability (3D Success 55.56% → 40.00%) yet nudges its Holistic Metric upward (27.78% → 30.00%) thanks to stable Scene Consistency (100.00% → 96.67%) and stronger 2D alignment.
- • Sora-2 remains limited (Holistic 10.34% → 15.52%) despite high physical priors (Physics Validness 65.52% → 56.90%), indicating that added floors do not resolve its instruction-following issues.
- • GPT-4o-image degrades sharply (Holistic 25.00% → 6.90%), revealing brittleness of single-image rollouts once multi-floor reasoning is required.

##### Table 32 Quantitative results for the Simultaneous Localization and Generation benchmark. We compare automatic evaluations across different models: Sora 2, Veo 3, Nano-banana, and GPT-4o-image.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Success (3D) Original Destination

Scene Consistency Score

Agent Consistency Score

Spatial Alignment Score

Destination Integrity Score

Object Semantic Score

Oracle Success Score (2D)

Trajectory Alignment Score

Oracle Success Score (3D)

Success Score (3D)

Success Score (2D)

Physics Validness

Overall Success

Evaluation

Environmental Complexity

- Level: floor01 Video Models

Veo 3 52.54% 44.07% 59.32% 49.15% 44.07% 72.88% 52.54% 67.80% 20.34% 45.76% 18.64% 45.76% 11.86% Sora 2 29.31% 27.59% 34.48% 37.93% 55.17% 84.48% 84.48% 72.41% 32.76% 79.31% 22.41% 65.52% 10.34%

Image Models

Nano-banana 55.56% 41.67% 55.56% 47.22% 88.89% 83.33% 83.33% 77.78% 50.00% 100.00% 38.89% 69.44% 27.78% GPT-4o-image 35.00% 28.33% 36.67% 31.67% 66.67% 81.67% 76.67% 61.67% 26.67% 98.33% 25.00% 58.33% 25.00%

- Level: floor02plus Video Models

Veo 3 21.05% 38.60% 26.32% 42.11% 33.33% 50.88% 42.11% 52.63% 31.58% 57.89% 15.79% 31.58% 10.53% Sora 2 29.31% 32.76% 31.03% 37.93% 48.28% 82.76% 67.24% 68.97% 44.83% 89.66% 20.69% 56.90% 15.52%

Image Models Nano-banana 40.00% 56.67% 40.00% 63.33% 76.67% 86.67% 86.67% 70.00% 50.00% 96.67% 36.67% 60.00% 30.00% GPT-4o-image 12.07% 17.24% 13.79% 20.69% 43.10% 68.97% 48.28% 37.93% 17.24% 94.83% 6.90% 34.48% 6.90%

View Fidelity

- Level: quality03 Video Models

Veo 3 35.00% 40.00% 35.00% 40.00% 35.00% 60.00% 45.00% 52.50% 22.50% 45.00% 15.00% 37.50% 10.00% Sora 2 27.50% 27.50% 27.50% 32.50% 47.50% 85.00% 77.50% 75.00% 37.50% 77.50% 22.50% 62.50% 17.50%

Image Models

Nano-banana 50.00% 62.50% 50.00% 62.50% 91.67% 91.67% 95.83% 91.67% 58.33% 100.00% 41.67% 83.33% 41.67% GPT-4o-image 30.00% 25.00% 32.50% 32.50% 65.00% 75.00% 65.00% 52.50% 20.00% 97.50% 17.50% 52.50% 17.50%

- Level: quality04 Video Models

Veo 3 46.15% 51.28% 51.28% 58.97% 43.59% 66.67% 46.15% 69.23% 33.33% 48.72% 25.64% 35.90% 15.38% Sora 2 26.32% 34.21% 26.32% 42.11% 55.26% 84.21% 71.05% 71.05% 34.21% 84.21% 21.05% 60.53% 10.53%

Image Models

Nano-banana 45.83% 45.83% 45.83% 54.17% 70.83% 87.50% 79.17% 58.33% 50.00% 95.83% 37.50% 54.17% 20.83% GPT-4o-image 25.00% 30.00% 25.00% 30.00% 57.50% 72.50% 62.50% 50.00% 32.50% 97.50% 22.50% 47.50% 22.50%

- Level: quality05 Video Models

Veo 3 29.73% 32.43% 43.24% 37.84% 37.84% 59.46% 51.35% 59.46% 21.62% 62.16% 10.81% 43.24% 8.11% Sora 2 34.21% 28.95% 44.74% 39.47% 52.63% 81.58% 78.95% 65.79% 44.74% 92.11% 21.05% 60.53% 10.53%

Image Models Nano-banana 50.00% 33.33% 50.00% 44.44% 88.89% 72.22% 77.78% 72.22% 38.89% 100.00% 33.33% 55.56% 22.22% GPT-4o-image 15.79% 13.16% 18.42% 15.79% 42.11% 78.95% 60.53% 47.37% 13.16% 94.74% 7.89% 39.47% 7.89%

Trajectory Distance

###### Level: short

Video Models

###### Veo 3 38.98% 40.68% 45.76% 45.76% 44.07% 61.02% 54.24% 64.41% 28.81% 62.71% 20.34% 42.37% 15.25% Sora 2 40.35% 31.58% 43.86% 40.35% 52.63% 85.96% 82.46% 75.44% 47.37% 89.47% 29.82% 66.67% 15.79%

Image Models

Nano-banana 54.55% 51.52% 54.55% 54.55% 87.88% 90.91% 90.91% 90.91% 57.58% 100.00% 42.42% 78.79% 36.36% GPT-4o-image 27.12% 20.34% 28.81% 25.42% 55.93% 77.97% 62.71% 50.85% 28.81% 96.61% 18.64% 45.76% 18.64%

Level: long

Video Models

###### Veo 3 35.09% 42.11% 40.35% 45.61% 33.33% 63.16% 40.35% 56.14% 22.81% 40.35% 14.04% 35.09% 7.02% Sora 2 18.64% 28.81% 22.03% 35.59% 50.85% 81.36% 69.49% 66.10% 30.51% 79.66% 13.56% 55.93% 10.17%

Image Models

###### Nano-banana 42.42% 45.45% 42.42% 54.55% 78.79% 78.79% 78.79% 57.58% 42.42% 96.97% 33.33% 51.52% 21.21% GPT-4o-image 20.34% 25.42% 22.03% 27.12% 54.24% 72.88% 62.71% 49.15% 15.25% 96.61% 13.56% 47.46% 13.56%

Destination Specification

###### Level: color mark

Video Models

###### Veo 3 53.45% 51.72% 60.34% 56.90% 43.10% 56.90% 46.55% 65.52% 32.76% 67.24% 25.86% 34.48% 17.24% Sora 2 44.64% 46.43% 51.79% 57.14% 48.21% 82.14% 73.21% 67.86% 62.50% 92.86% 33.93% 58.93% 19.64%

Image Models

Nano-banana 81.25% 59.38% 81.25% 68.75% 87.50% 84.38% 87.50% 78.12% 71.88% 100.00% 62.50% 71.88% 50.00% GPT-4o-image 44.83% 36.21% 44.83% 41.38% 55.17% 70.69% 62.07% 51.72% 37.93% 96.55% 31.03% 46.55% 31.03%

Level: location description

Video Models

###### Veo 3 20.69% 31.03% 25.86% 34.48% 34.48% 67.24% 48.28% 55.17% 18.97% 36.21% 8.62% 43.10% 5.17% Sora 2 15.00% 15.00% 15.00% 20.00% 55.00% 85.00% 78.33% 73.33% 16.67% 76.67% 10.00% 63.33% 6.67%

Image Models

###### Nano-banana 17.65% 38.24% 17.65% 41.18% 79.41% 85.29% 82.35% 70.59% 29.41% 97.06% 14.71% 58.82% 8.82% GPT-4o-image 3.33% 10.00% 6.67% 11.67% 55.00% 80.00% 63.33% 48.33% 6.67% 96.67% 1.67% 46.67% 1.67%

##### Table 33 Quantitative results for the Simultaneous Localization and Generation (SLAG) benchmark. We compare automatic evaluations against human judgments for Veo-3 across the same hard-level dimensions.

Task Completeness Physical Understanding Instruction Following Gate Metric Holistic Metric

Scene Consistency Score

Agent Consistency Score

Spatial Alignment Score

Destination Integrity Score

Object Semantic Score

Oracle Success Score (2D)

Trajectory Alignment Score

Oracle Success Score (3D)

Success Score (3D)

Success Score (2D)

Physics Validness

Overall Success

Evaluation

Environmental Complexity

- Level: floor01 Auto Evaluation 52.54% 44.07% 59.32% 49.15% 44.07% 72.88% 52.54% 67.80% 20.34% 45.76% 45.76% 11.86% Human Evaluation 50.00% 31.48% 64.81% 37.04% 9.26% 38.89% 68.52% 61.11% 44.44% 98.15% 5.56% 0.00%
- Level: floor02plus Auto Evaluation 21.05% 38.60% 26.32% 42.11% 33.33% 50.88% 42.11% 52.63% 31.58% 57.89% 31.58% 10.53% Human Evaluation 28.57% 7.14% 32.14% 10.71% 0.00% 25.00% 60.71% 75.00% 75.00% 85.71% 2.22% 0.00%

View Fidelity

- Level: quality03 Auto Evaluation 35.00% 40.00% 35.00% 40.00% 35.00% 60.00% 45.00% 52.50% 22.50% 45.00% 37.50% 10.00% Human Evaluation 50.00% 21.88% 59.38% 21.88% 6.25% 40.62% 68.75% 43.75% 40.62% 87.50% 3.33% 0.00%
- Level: quality04 Auto Evaluation 46.15% 51.28% 51.28% 58.97% 43.59% 66.67% 46.15% 69.23% 33.33% 48.72% 35.90% 15.38% Human Evaluation 36.67% 40.00% 53.33% 50.00% 10.00% 33.33% 53.33% 76.67% 63.33% 100.00% 5.83% 0.00%
- Level: quality05 Auto Evaluation 29.73% 32.43% 43.24% 37.84% 37.84% 59.46% 51.35% 59.46% 21.62% 62.16% 43.24% 8.11% Human Evaluation 40.00% 0.00% 45.00% 5.00% 0.00% 25.00% 80.00% 85.00% 65.00% 95.00% 2.50% 0.00%

Trajectory Distance

Level: short Auto Evaluation 38.98% 40.68% 45.76% 45.76% 44.07% 61.02% 54.24% 64.41% 28.81% 62.71% 42.37% 15.25% Human Evaluation 47.62% 21.43% 61.90% 30.95% 7.14% 35.71% 76.19% 66.67% 54.76% 92.86% 6.11% 0.00%

Level: long

Auto Evaluation 35.09% 42.11% 40.35% 45.61% 33.33% 63.16% 40.35% 56.14% 22.81% 40.35% 35.09% 7.02% Human Evaluation 37.50% 25.00% 45.00% 25.00% 5.00% 32.50% 55.00% 65.00% 55.00% 95.00% 1.67% 0.00%

Destination Specification

Level: color mark Auto Evaluation 53.45% 51.72% 60.34% 56.90% 43.10% 56.90% 46.55% 65.52% 32.76% 67.24% 34.48% 17.24% Human Evaluation 55.56% 31.48% 70.37% 33.33% 5.56% 35.19% 66.67% 61.11% 33.33% 96.30% 5.00% 0.00%

Level: location description

Auto Evaluation 20.69% 31.03% 25.86% 34.48% 34.48% 67.24% 48.28% 55.17% 18.97% 36.21% 43.10% 5.17% Human Evaluation 17.86% 7.14% 21.43% 17.86% 7.14% 32.14% 64.29% 75.00% 96.43% 89.29% 2.78% 0.00%

- 13.3.2 Human vs. Automated Evaluation Discrepancy

- Table 33 presents a critical divergence between automatic metrics and human judgment regarding the Veo-3 model. This comparison exposes the “Uncanny Valley” of physics simulation in current video models.

The “Hallucination” of Quality: Scene Consistency. Humans consider the videos visually stable (98.15% on floor01, 100% on quality04), yet auto-metrics flag instability (Scene Consistency 45.76% on floor01). Veo-3 produces temporally smooth, coherent frames to the eye, but algorithms detect subtle pixel/geometry shifts that humans overlook.

Physics Validness: Humans Are Harsher. On floor01, the auto-evaluator assigns Physics Validness of 45.76%, while humans give just 5.56%. Simple collision or alignment checks let runs pass automatically, but humans catch “soft” violations (teleportation, clipping, floating), driving the human Holistic Metric to 0.00%.

Trajectory Alignment Reality Check. Alignment between the 2D map and 3D view is sharply contested: Auto scores 44.07% on floor01, whereas humans rate only 9.26%. The map may look structurally plausible to an algorithm yet fails to reflect the exact turns and timing seen in the 3D video.

Overall Metric Analysis. SLAG shows a severe “wood barrel” effect: overall success collapses to 3.64% (2/55 samples) because weak links—Destination Integrity (25.45%), Trajectory Alignment (29.09%), Physics Validness—gate everything else. Image models (Nano-banana) occasionally assemble logically consistent frames, but video models (Veo-3, Sora-2) remain “dreamers,” generating visually compelling yet spatially and physically unreliable navigations.

- 13.4 Case Study: Qualitative Failure Modes

The Simultaneous Localization and Generation task highlights a common failure mode across all three models in Figure 24: the inability to maintain semantic linkage between split-screen representations over time.

Veo-3: High Fidelity, Low Logic. Veo-3 offers the most stable visual experience, maintaining high Scene Consistency as confirmed by human evaluation (98.15%). In the case study, the agent correctly identifies the

Input Image

Prompt Text:

Create a split-screen video for the task "Simultaneously Localization and Generation." The left panel must show a humanoid robot navigating a photorealistic 3D indoor scene to a bright red (#ff0000) target area painted on the floor, while the right panel simultaneously shows a generated bright green (#00ff00) trajectory line that grows as the robot advances in a fixed 2D top-down map in real time. Keep the camera static, isometric, and wide enough to include the entire navigable layout.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Veo-3

Frame 3s

Frame 1s Frame 2s

Spatial Alignment Score = 1 The agent in 3D panel recognized the start and the moving direction matches the agent’s orientation.

Trajectory Alignment Score (tmp) = 1 The movement in 2D top-down panel aligned with the one in 3D panel.

Scene Consistency Score = 1 It maintained the input scene detail and the scene style.

[Figure 252]

[Figure 253]

[Figure 254]

Frame 4s Frame 5s Frame 8s

(Oracle) Success Score (3D) = 1 Trajectory Alignment Score = 0 The agent in 3D panel arrived the red marked area, while the one in 2D panel misaligned with it.

Agent Consistency Score = 1, Scene Consistency Score = 1 The agent moved continuously without teleporting. And it maintained the input scene detail and the scene style.

###### (Oracle) Success Score (2D) = 0

Simply reaching the red-marked area does not constitute a 2D success; the SLAG task requires reaching the same location as the 3D area.

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

Sora-2

Frame 0s Frame 2s

Frame 5s

Scene Consistency Score = 0 It created both new scene in 3D and new map in 2D panel. The hallucinated marked area in 2D also not matched.

Object Semantic Score = 0 The agent walked through a wall and a cabinet.

Trajectory Alignment Score = 0 In the 2D panel, the agent moves along the green line and makes a turn that is not present in the 3D panel.

[Figure 259]

[Figure 260]

[Figure 261]

Frame 8s Frame 10s Frame 12s

(Oracle) Success Score (2D) = 0 The agent did not arrive at the target aligned with the one in 3D panel, even if it arrived in the red marked area in 2D.

(Oracle) Success Score (3D) = 1 The agent in 3D panel arrived the red marked area.

Destination Integrity Score = 1 It maintained the red marked area when the agent doing navigation.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

###### Wan2.2

Frame 0s Frame 1s

Frame 2s

Spatial Alignment Score= 1 The agent in 3D panel recognized the start and the moving direction matches the agent’s orientation.

Object Semantic Score = 0, Agent Consistency Score = 0

Destination Integrity Score = 1 The location of the red marked area in 3D panel remains the same.

The agent in 3D panel teleported / flied to the indigo wall and walked on that.

[Figure 266]

[Figure 267]

[Figure 268]

Frame 3s Frame 4s Frame 5s

Trajectory Alignment Score = 0 In the 2D panel, the agent moves along the green line and makes a turn that is not present in the 3D panel.

Scene Consistency Score = 1 It maintained the input scene detail and the scene style both in 3D and 2D panel.

(Oracle) Success Score (3D) = 0, (Oracle) Success Score (2D) = 0 The agent in 2D/3D panel did not arrive at the target although it is very close.

- Figure 24 Case Study for Simultaneous Localization and Generation. The figure compares Veo-3, Sora-2, and Wan-2.2 on a split-screen navigation task requiring synchronization between a 3D view and a 2D map. Veo-3 achieves 3D target success but suffers from 2D trajectory misalignment. Sora-2 exhibits severe scene hallucination and physics violations (clipping through walls). Wan-2.2 demonstrates agent inconsistency (teleporting) and fails to reach the final target.

start position and successfully reaches the red-marked goal in the 3D view. However, the Trajectory Alignment fails. As the 3D agent moves forward, the 2D dot on the map drifts independently, failing to mirror the agent’s path. This exemplifies the “disembodied” nature of the generation—the model understands it needs to generate a map and a video, but treats them as separate artistic tasks rather than coupled data streams.

Sora-2: Severe Hallucination. Sora-2 struggles to maintain the input reality. (1) Scene Hallucination: It immediately fails Scene Consistency (Score = 0) by replacing the input floor plan with a hallucinated layout. (2) Physics Violations: Consistent with the low Physics Validness scores, the agent is observed clipping through a cabinet.(3) PhantomTurns: The 2D agent executes a turn that the 3D agent never makes, further emphasizing the lack of cross-modal attention in the model architecture.

Wan-2.2: The Teleportation Problem. Wan-2.2 exhibits the most erratic behavior, leading to complete task failure. (1) Agent Instability: The model suffers from severe Agent Consistency failures (Score = 0). At Frame 1s, the agent teleports—effectively “flying”—onto an indigo wall, treating a vertical surface as a walkable floor. (2) Navigation Failure: Unlike Veo-3, Wan-2.2 fails to reach the destination in either view. The agent stops short, resulting in zero scores for both Success (2D) and Success (3D). This model demonstrates that without strong physical priors, video generation models devolve into surrealism rather than embodied simulation.

### 14 Physical Commonsense

We introduce the Physical Commonsense task to assess a model’s foundational understanding of “intuitive physics” (Battaglia et al., 2013; Yi et al., 2020; Wu et al., 2015), a critical prerequisite for robust world modeling. This task probes Physical Reasoning by requiring models to generate videos that combine photorealism with physical plausibility (Bear et al., 2021; Riochet et al., 2021; Piloto et al., 2022). Beyond static visual fidelity, the task evaluates whether the model captures essential principles—such as gravity, momentum, collisions, and material properties (rigidity, fluid dynamics)—and correctly models causal dynamics (Bakhtin et al., 2019; Allen et al., 2020). The evaluation spans both 3D Spatial Reasoning (spatial object interactions) and Temporal Reasoning (sequential cause-and-effect), organized along two complementary axes: (1) Physical Concepts, which tests fundamental principles using the VideoPhy ontology (Bansal et al., 2024); and (2) Sports Scenarios, which probe compositional reasoning through dynamic, high-velocity human movements.

- 14.1 Data Sources and Task Structure

Physical Concepts (Fundamental Interactions). To systematically evaluate atomic physical principles, we adopt the structured ontology from VideoPhy (Bansal et al., 2024) and VideoPhy-2 (Bansal et al., 2025). We draw from a diverse pool of captioned interactions, including: Solid–Solid (e.g., rigid collisions, stacking), Solid–Fluid (e.g., splashing, buoyancy), and Fluid–Fluid (e.g., diffusion, mixing). These prompts isolate specific physical laws, allowing us to test statics, dynamics, and kinematics in controlled environments.

Sports Scenarios (Compositional Reasoning). To evaluate physical reasoning in complex, real-world contexts, we synthesize a complementary dataset of sports-oriented prompts. These scenarios naturally require the integration of multiple physical laws simultaneously. The dataset spans: Precision&Arts (e.g., ballet pirouettes requiring angular momentum), Winter Sports (e.g., skiing moguls involving friction and gravity), Aquatics (e.g., diving and swimming involving fluid resistance), and Athletics. These prompts test the model’s ability to maintain physical consistency during complex human-object interactions.

- 14.2 Hard-Level Control and Evaluation Taxonomy

To ensure a rigorous assessment, we curate a balanced evaluation set of 100 samples (see Table 34), stratified along three dimensions of difficulty:

- • Interaction Type (The “What”): We categorize samples by the material properties involved:

- – Solid-Solid: Interactions between rigid bodies (testing impulse, friction, and collision response).
- – Solid-Fluid: Interactions between solids and liquids (testing displacement, splashing, and floating).

- – Fluid-Fluid: Dynamics of miscible and immiscible fluids (testing viscosity, mixing, and turbulence).

- • Scenario Context (The “Where”): We distinguish between controlled physics experiments (Physical Concepts) and unconstrained, dynamic environments (Sports Scenarios), aiming to test generalization from atomic laws to complex behaviors.
- • Interaction Complexity (The “How”): Across all categories, we vary the complexity level:

- – Simple: Single-object motion or static equilibrium.
- – Complex: Multi-object interactions with simultaneous forces.
- – Chain-Reaction: Causal sequences where an initial action triggers cascading effects.

- Table 34 Distribution of Physical Commonsense evaluation samples. The set is balanced to weigh fundamental physical understanding equally against compositional real-world application.

Task Axis Total Samples

##### Physical Concepts (Atomic Interactions) 25 Sports Scenarios (Compositional Contexts) 25

Total 50

- 14.3 Evaluation and Metrics

Modeling physical commonsense inherently requires capturing temporal dynamics such as force propagation, momentum transfer, and continuous motion. Because static image generators lack temporal modeling capabilities and cannot represent causal interactions unfolding over time, they are fundamentally limited in assessing physical plausibility. Therefore, our evaluation focuses exclusively on video generative models, capable of producing temporally coherent sequences (Cai et al., 2025).

To ensure consistent, structured judgments, we employ a Vision-Language Model (VLM) evaluator—Gemini2.5-Pro (Comanici et al., 2025)—prompted to act as a physics and video expert. For each generated video, the VLM assesses four specific dimensions of quality. To standardize the evaluation, each dimension is treated

- as a binary metric (0/1), where a score of 1 indicates the criteria are fully satisfied and 0 indicates a failure or significant violation. We define the four fine-grained metrics as follows:

- • Physics Accuracy (0/1): Evaluates whether the generated video strictly obeys fundamental physical laws. The model checks if motion adheres to gravity, momentum, and friction, and verifies that object interactions are plausible. A score of 0 is assigned if there are violations such as objects floating, moving at unrealistic speeds for the context, displaying incorrect trajectories, or deviating from the scenario’s “Physics Focus.”
- • Motion Quality (0/1): Assesses the temporal coherence and naturalness of the movement. This metric verifies that the motion follows the expected pattern described in the scenario and remains smooth and continuous. A score of 0 is assigned if the motion is jerky, exhibits unnatural accelerations, or suffers from temporal discontinuities and inconsistency.
- • Visual Realism (0/1): Measures the visual fidelity and believability of the scene. The model checks if objects and materials appear realistic, whether lighting and shadows are consistent, and if the scene composition is plausible. A score of 0 is assigned if there are significant visual artifacts, glitches, or if the scene lacks photorealism.
- • Prompt Adherence (0/1): Determines whether the video semantically matches the user input. This metric verifies that all key elements (objects, setting) are present and that the specific action described actually occurs. A score of 0 is assigned if there are significant mismatches between the generated content and the text prompt.

- • Overall Success (Aggregated Metric): To provide a holistic measure of generation capability, we compute a strict Overall Success score. A generated video is marked as successful (1) if and only if it satisfies all four fine-grained metrics simultaneously.

This rigorous aggregation ensures that high performance requires a model to generate videos that are not only physically accurate but also visually coherent, smooth, and semantically correct.

Prompt Text: A parasail is being inflated and prepared by crew, showing the process of inflation and the parachute's texture. Physics Focus: Action: parasailing

Expected Motion: Action should be performed correctly

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Frame 5 Frame 6 Frame 7 Frame 8

Physics Accuracy = 1

Prompt Adherence = 1

Motion Quality = 1

Visual Realism = 1

The physics shown in the video are accurate. The parasail inflates as air

The video faithfully follows

The motion is smooth and natural,

The video looks highly realistic,

the prompt, clearly depicting a parasail being inflated by two crew members. It focuses on

following the expected pattern for this action. The parasail inflates gradually and rises steadily, without

with natural textures in the parasail’s nylon fabric, sand, ocean, and people. The bright

fills the canopy, increasing internal

pressure and forming an airfoil that generates lift. As it rises, the parachute transitions from wrinkled

the full inflation process from

jerky movements or unrealistic

lighting and clear, accurate

start to finish, with close-up views that highlight the

speeds. The slight sway of the parachute as it fills looks realistic,

shadows match a sunny outdoor scene and shift correctly as the

fabric to a taut, stable structure,

which is physically plausible. The lines tighten correctly as they begin to

parachute’s texture and how the material responds to

and the crew’s actions—holding position and managing the rigging—

parachute rises. The recording shows no visual artifacts or

bear the load.

airflow.

are appropriate for the process.

glitches.

- Figure 25 Case Study: Success case generated by Veo-3. Physically Plausible Parachute Inflation.

- 14.4 Case Study

To better illustrate the strengths and limitations revealed by our Physical Commonsense evaluation, we present three representative case studies generated by Veo-3 (Figure 25, 26, and 27). Each example highlights how our metrics disentangle distinct dimensions of physical reasoning: physics accuracy, prompt adherence, motion quality, and visual realism. These cases further demonstrate that high visual fidelity alone is not indicative of correct physical behavior—underscoring why physics-focused evaluation is necessary for robust real-world video generation.

Success Case: Physically Plausible Parachute Inflation. Figure 25 shows a prompt involving the inflation of a parasail by a two-person crew. Veo-3 performs strongly across all evaluation dimensions, achieving full scores on physics accuracy, prompt adherence, motion quality, and visual realism. The model captures the physical mechanics of inflation: air fills the canopy, internal pressure increases, the fabric transitions from wrinkled to taut, and the lines tighten appropriately as load is applied. The motion unfolds smoothly and continuously, without discontinuities or unnatural accelerations. Detailed textures, realistic lighting, and consistent shading further contribute to the clip’s visual plausibility. This example illustrates Veo-3’s ability to correctly represent gradual force buildup, material deformation, and multi-agent coordination.

- Failure Case I: Missing Solid–Solid Interaction in Coffee Grinding. Figure 26 highlights a failure case involving a metal grinder crushing coffee beans (Solid–Solid interaction). While the static frames resemble a real grinder setup, the temporal dynamics violate fundamental physical laws. Instead of showing a realistic grinding process—where rigid beans fracture into progressively smaller particles—the video abruptly transitions from whole beans to uniform fine grounds, without any mechanical cause. This discontinuous “state-change” effect

Prompt Text: Metal grinder crushing coffee beans. Physics Focus: States of matter interaction: solid_solid

Expected Motion: Natural interaction between materials

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

whole or partially

broken beans

uniform grounds

[Figure 286]

Frame 5 Frame 6 Frame 7 Frame 8

Physics Accuracy = 0

Prompt Adherence = 0

Motion Quality = 0

Visual Realism = 0

The video fails to show actual solid-solid

The prompt requires

The motion is unnatural and

While the static elements (the

crushing mechanics. Instead of grinder burrs fracturing coffee beans, it depicts a

showing a metal grinder crushing coffee beans, but

discontinuous. The initial output of whole and broken beans is chaotic,

grinder’s metal finish, whole

beans, and final grounds) look realistic, the transition from

magical transformation: whole or partially

the video omits the crushing action. Although

lacking the controlled feed typical of a grinder. The most significant

broken beans fall from the chute, but around the 6-second mark they suddenly become uniform grounds with no physical

beans to grounds is visually implausible. The material change

the grinder and beans

issue is the abrupt transition where

appear, the scene shows a transformation rather than the required physical

the output instantly shifts from large particles to fine grounds—a state-change effect rather than a

appears more like a digital morph than a physical grinding process,

cause. This breaks conservation of mass

and ignores material fracture mechanics, since real grinders do not spontaneously

making the video feel unrealistic and staged.

interaction, failing to follow

smooth, physical grinding process.

convert whole beans mid-air.

the core process.

- Figure 26 Case Study: Failure case generated by Veo-3. Missing Solid–Solid Interaction in Coffee Grinding.

resembles a visual morph rather than a physical transformation, resulting in violations of mass conservation and material fracture mechanics. Although the model adheres partially to the prompt in terms of objects present, it fails to depict the required interaction, leading to zeros in physics accuracy, prompt adherence, motion quality, and visual realism.

- Failure Case II: Incorrect Angular Momentum Dynamics in Ballet Rotation. Figure 27 presents a ballet scenario requiring a fouetté turn, a physically demanding movement involving rapid leg whipping to generate angular momentum and sustain rotation. Veo-3 fails across all physics-focused dimensions. The dancer’s extended leg and upper body become unnaturally distorted mid-rotation, deviating from realistic human biomechanics. The model also misinterprets the action: instead of producing a sharp, continuous whipping motion that drives a true fouetté, the dancer performs a slow, controlled rotation with no momentum-generating movement. This breaks both prompt adherence and key physical principles such as angular momentum conservation and joint kinematic constraints. Although the visual appearance remains high-quality at a glance, closer inspection reveals anatomy inconsistencies and motion artifacts that undermine physical realism.

Takeaways. These case studies highlight the need to evaluate video models beyond photorealism. Veo-3 can produce visually convincing clips, but frequently fails in scenarios requiring nuanced physical reasoning—especially where material properties, continuous force propagation, or human biomechanics play a central role. The structured metrics in our Physical Commonsense task make these failure modes explicit, providing a robust diagnostic tool for guiding future model improvements.

Prompt Text: A ballet dancer performs a fouetté turn, whipping one leg around while spinning on the other.

Physics Focus: Conservation of angular momentum, centripetal force

Expected Motion: Rapid leg whip, continuous rotation, stable center

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Frame 1 Frame 2 Frame 3 Frame 4

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Frame 5 Frame 6 Frame 7 Frame 8

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Frame 9 Frame 10 Frame 11 Frame 12

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Frame 13 Frame 14 Frame 15 Frame 16

Physics Accuracy = 0

Prompt Adherence = 0 The video does not match

Visual Realism = 0

Motion Quality = 0

The video contains noticeable physics and anatomy violations, particularly in Frames

The video appears realistic at

The motion does not match a true fouetté turn, which

the prompt. Instead of

first glance, but it contains

7 and 12, where the dancer’s extended leg and upper-body become unnaturally warped

performing a fouetté turn,

noticeable physics and anatomy

requires a rapid, whipping

the dancer completes a

violations. This deformation

leg action to generate

in mid-rotation. This deformation breaks

single, slow rotation with

breaks realistic limb geometry

momentum. Instead, the

realistic limb geometry and disrupts the

her leg extended. This

and expected biomechanics,

video shows slow, controlled

expected biomechanics of a controlled turn.

movement aligns more with a pirouette à la seconde or tour à la seconde. The essential “whipping” action

showing clear failures in physical

rotations with the leg held

Such distortion is inconsistent with how a

and anatomical consistency despite the otherwise smooth motion.

steadily in à la seconde. The

leg would move under normal joint

distinct sharp “whip” that

constraints, balance, and angular

defines a fouetté is absent, so the movement lacks the expected speed and percussive quality.

momentum, revealing clear failures in

of the leg—required for a

physical and anatomical consistency despite the otherwise smooth rotation.

true fouetté—is completely

missing.

- Figure 27 Case Study: Failure case generated by Veo-3. Incorrect Angular Momentum Dynamics in Ballet Rotation.

- 14.5 Evaluation Results

Key Finding: Physical Plausibility Is Decoupled from Visual Realism

Sora-2 achieves the strongest physical commonsense performance with a 70.00% overall success rate, substantially outperforming Veo-3 (51.02%) and Wan-2.2 (24.00%). Crucially, the results reveal a consistent pattern: high visual realism does not imply correct physical reasoning. Wan-2.2 produces highly photorealistic videos (84–96% Visual Realism) yet fails to follow prompts and physical constraints (24% Overall), indicating a fundamental gap between appearance-level fidelity and physically grounded world modeling. Fine-grained analysis reveals Sports Scenarios are systematically easier than Physical Concepts across all models, while Solid-Solid interactions prove most challenging—Veo-3 achieves 0% success on rigid body collisions compared to 75% on Solid-Fluid interactions.

##### Table 35 Quantitative results for the Physical Commonsense task. We evaluate three video generative models (Veo-3, Sora-2, and Wan-2.2) on Physical Concepts and Sports Scenarios.

Fine-grained Metrics Primary Metric Model Physics Accuracy ↑ Motion Quality ↑ Visual Realism ↑ Prompt Adherence ↑ Overall ↑ Scenario Type: Physical Concepts

###### Veo-3 62.50% 54.17% 83.33% 50.00% 41.67% Sora-2 84.00% 80.00% 96.00% 76.00% 76.00% Wan-2.2 58.67% 53.33% 72.00% 38.67% 26.67%

Scenario Type: Sports Scenarios

###### Veo-3 80.00% 68.00% 92.00% 68.00% 60.00% Sora-2 88.00% 72.00% 88.00% 68.00% 64.00% Wan-2.2 42.67% 33.33% 96.00% 21.33% 21.33%

Average

Veo-3 71.43% 61.22% 87.76% 59.18% 51.02% Sora-2 86.00% 76.00% 92.00% 72.00% 70.00% Wan-2.2 50.67% 43.33% 84.00% 30.00% 24.00%

- 14.5.1 VLM-Based Evaluation

#### Table 35 reports quantitative results on the Physical Commonsense task, evaluated using Gemini-2.5-Pro (Comanici et al., 2025) as a VLM-based evaluator. We compare three state-of-the-art video generation models—Veo-

3, Sora-2, and Wan-2.2—across two complementary scenario types: Physical Concepts and Sports Scenarios. Overall Performance Trends. Sora-2 consistently outperforms competing models, achieving the highest overall success rate (70.00%) and leading across all fine-grained metrics. In contrast, Veo-3 exhibits moderate but uneven performance (51.02%), while Wan-2.2 substantially underperforms (24.00%) despite strong visual realism. Notably, all models score highly on Visual Realism (84–92%), yet exhibit large variance in Physics Accuracy (50.67–86.00%) and Prompt Adherence (30.00–72.00%), reinforcing that perceptual quality alone is an unreliable indicator of physical correctness.

Scenario-Specific Difficulty. Across all models, Sports Scenarios are consistently easier than Physical Concepts. Veo-3 improves from 41.67% overall success on Physical Concepts to 60.00% on Sports Scenarios, while Sora-2 maintains strong performance across both (76.00% vs. 64.00%). In contrast, Wan-2.2 fails to generalize even in Sports Scenarios, achieving only 21.33% overall success. These results suggest that contemporary video models benefit from learned biomechanical motion patterns in human-centric activities, while struggling with fine-grained material interactions such as collisions, splashing, and mixing.

##### Table 36 Fine-grained VLM-based evaluation results by Sports Scenarios attributes. Overall success rate (%) is reported across sport types (Ballet, Diving, Skiing, Swimming) and difficulty levels (Easy, Medium, Hard).

By Sport Type By Difficulty

Model Ballet Diving Skiing Swimming Easy Medium Hard

Veo-3 33.3% 50.0% 71.4% 83.3% 60.0% 62.5% 57.1% Sora-2 33.3% 50.0% 85.7% 83.3% 60.0% 75.0% 57.1% Wan-2.2 44.4% 0.0% 28.6% 11.1% 16.7% 33.3% 14.3%

##### Table 37 Fine-grained VLM-based evaluation results by Physical Concepts attributes. Overall success rate (%) is reported across states-of-matter interaction types and difficulty levels.

By States of Matter By Difficulty

Model Solid-Solid Solid-Fluid Fluid-Fluid Action/Other Easy Hard

Veo-3 0.0% 75.0% 50.0% 40.0% 53.3% 22.2% Sora-2 100.0% 75.0% 100.0% 66.7% 75.0% 77.8% Wan-2.2 33.3% 25.0% 83.3% 17.8% 35.4% 11.1%

Sport-Specific Patterns. Fine-grained analysis (Table 36) reveals sport-specific challenges. Ballet proves most difficult for all models (33–44% success), while Swimming achieves the highest scores (83% for Veo-3 and Sora-2). Wan-2.2 fails completely on Diving (0%) yet achieves its best performance on Ballet (44.4%), suggesting model-specific biases in motion priors. By difficulty level, Sora-2 shows the most consistent performance across Easy (60%), Medium (75%), and Hard (57%) prompts, while Wan-2.2 collapses uniformly across all difficulty levels (14–33%).

States-of-Matter Analysis. Table 37 reveals interaction-specific challenges. Solid-Solid interactions prove most difficult: Veo-3 achieves 0% success on rigid body collisions, while Sora-2 achieves perfect 100%. Fluid-Fluid interactions show more variance, with Wan-2.2 achieving 83.3%—its highest category score—while Veo-3 scores only 50%. By difficulty level, all models show degradation on hard cases, with Wan-2.2 exhibiting the most severe collapse (35.4% Easy → 11.1% Hard).

The Visual–Physical Disconnect. Wan-2.2 demonstrates the clearest dissociation between visual quality and physical reasoning. Despite achieving the highest Visual Realism score on Sports Scenarios (96.00%), it records the lowest Physics Accuracy (42.67%) and Prompt Adherence (21.33%) in the same setting. This failure mode suggests that the model prioritizes surface-level appearance over causal and physical consistency, producing videos that “look right” but violate core physical principles.

- 14.5.2 Human Evaluation

To validate VLM-based assessments, we conducted human evaluation on Veo-3 generated videos (n = 45). Human annotators assessed multiple dimensions including Physics Accuracy, Motion Quality, Visual Realism and Prompt Adherence, which we map to the same metrics used in AutoEval (Table 35).

- Table 38 Veo-3 evaluation comparison: VLM-based evaluation (AutoEval) vs. human evaluation (HumanEval) across Physical Concepts and Sports Scenarios.

Scenario Eval Mode Physics Accuracy ↑ Motion Quality ↑ Visual Realism ↑ Prompt Adherence ↑ Overall ↑ Physical Concepts AutoEval 62.50% 54.17% 83.33% 50.00% 41.67%

Human Eval 77.27% 86.36% 83.64% 72.73% 77.27% Sports Scenarios AutoEval 80.00% 68.00% 92.00% 68.00% 60.00% Human Eval 91.30% 78.26% 84.35% 65.22% 82.61% Average AutoEval 71.43% 61.22% 87.76% 59.18% 51.02% Human Eval 84.44% 82.22% 84.00% 68.89% 80.00%

AutoEval vs Human Eval Gap. Table 38 reveals a striking discrepancy between automated and human evaluation. Human evaluation consistently rates Veo-3 higher across all metrics, with the overall success rate increasing from 51.02% (AutoEval) to 80.00% (Human Eval)—a 29-point improvement. Motion Quality shows the largest gap: 82.22% (Human) vs. 61.22% (AutoEval), suggesting that temporal artifacts flagged by the VLM evaluator are often imperceptible or acceptable to human observers. Physics Accuracy similarly improves from 71.43% to 84.44%, indicating that VLM-based evaluation applies overly strict criteria that may not align with human perception of physical plausibility.

- Table 39 Veo-3 fine-grained comparison by Sports Scenarios attributes: AutoEval vs. HumanEval. Overall success rate (%) is reported across sport types (Ballet, Diving, Skiing, Swimming) and difficulty levels (Easy, Medium, Hard).

By Sport Type By Difficulty

Eval Mode Ballet Diving Skiing Swimming Easy Medium Hard

AutoEval 33.3% 50.0% 71.4% 83.3% 60.0% 62.5% 57.1% Human Eval 50.0% 100.0% 100.0% 83.3% 90.0% 57.1% 100.0%

#### Sport-Specific Insights. Fine-grained analysis (Table 39) reveals that human evaluators rate Veo-3’s Diving and Skiing scenarios at 100% Visual Realism, compared to AutoEval scores of 50.0% and 71.4% respectively. Ballet remains consistently challenging across both evaluation modes (50.0% Human, 33.3% AutoEval), confirming

##### Table 40 Veo-3 fine-grained comparison by Physical Concepts attributes: AutoEval vs. HumanEval. Overall success rate (%) is reported across states-of-matter interaction types and difficulty levels.

By States of Matter By Difficulty

Eval Mode Solid-Solid Solid-Fluid Fluid-Fluid Action/Other Easy Hard

AutoEval 0.0% 75.0% 50.0% 40.0% 53.3% 22.2% Human Eval 66.7% 100.0% 100.0% 73.3% 78.6% 77.8%

that sustained rotational dynamics in pirouettes and fouettés pose genuine difficulties for current video generators. By difficulty level, human evaluators show an unexpected pattern: hard scenarios achieve 100% success while medium scenarios score only 57.1%, suggesting that brief, dramatic actions (ski jumps, cliff dives) are easier to generate plausibly than sustained complex motions.

Physical Concept Insights. Table 40 shows that for Physical Concepts, Solid-Fluid and Fluid-Fluid interactions achieve 100% human-rated Visual Realism, while Solid-Solid interactions remain challenging at 66.7%. The AutoEval-Human gap is most pronounced for Solid-Solid: AutoEval rates Veo-3 at 0%, while humans rate it

- at 66.7%—a complete reversal that highlights the strictness of VLM-based collision detection. Across difficulty levels, human evaluation maintains consistent performance (78.6% Easy, 77.8% Hard), whereas AutoEval shows severe degradation on hard cases (53.3% Easy, 22.2% Hard).

- 15 Conclusion

We introduce a comprehensive benchmark framework MMGR to evaluate the reasoning capabilities of generative models across five complementary abilities. Our evaluation reveals a critical gap between perceptual quality and reasoning capability, with stark performance hierarchies suggesting models primarily learn pattern matching rather than true symbolic reasoning. The gap between closed-source and open-source models indicates progress relies heavily on scale rather than architectural innovations.

Furthermore, our analysis identifies a unique “temporal tax” on reasoning in video generation, where the requirement to maintain frame-to-frame coherence actively competes with logical consistency. This is evidenced by video models consistently underperforming their image-based counterparts on complex logic tasks, treating mathematical derivation as a visual texture to be morphed rather than a semantic chain to be constructed. We also observe a prevalent “hallucination of competence,” where models frequently generate correct final outcomes despite invalid intermediate reasoning steps, confirming that they are often memorizing answer patterns rather than executing genuine multi-step deduction.

These limitations stem from three fundamental bottlenecks in current training recipes: a severe scarcity of structured symbolic reasoning data compared to naturalistic physical data; architectural constraints that prioritize local visual plausibility over global, long-horizon consistency; and optimization objectives that reward perceptual fidelity rather than logical validity. To bridge the gap from image animation to true world simulation, future work must look beyond mere scaling to develop architectures that decouple reasoning states from visual rendering and integrate auxiliary objectives for causal consistency.

References

Kelsey R Allen, Kevin A Smith, and Joshua B Tenenbaum. Rapid trial-and-error learning with simulation supports flexible tool use and physical reasoning. Proceedings of the National Academy of Sciences, 117(47):29302–29310, 2020.

Dong An, Yuankai Qi, Yangguang Li, Yan Huang, Liang Wang, Tieniu Tan, and Jing Shao. Bevbert: Topo-metric map pre-training for language-guided navigation. arXiv preprint arXiv:2212.04385, 2022.

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3674–3683, 2018a.

Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sünderhauf, Ian Reid, Stephen Gould, and Anton Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In CVPR, 2018b.

Renée Baillargeon. Object permanence in 3½- and 4½-month-old infants. Developmental Psychology, 23(5):655–664, 1987.

Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. Phyre: A new benchmark for physical reasoning. 2019.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024.

Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.

Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15791–15801, 2025.

David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap. Measuring abstract reasoning in neural networks. 2018.

Dhruv Batra, Aaron Gokaslan, Aniruddha Kembhavi, Oleksandr Maksymets, Roozbeh Mottaghi, Manolis Savva, Alexander Toshev, and Erik Wijmans. ObjectNav Revisited: On Evaluation of Embodied Agents Navigating to Objects. In arXiv:2006.13171, 2020.

Peter W Battaglia, Jessica B Hamrick, and Joshua B Tenenbaum. Simulation as an engine of physical scene understanding. Proceedings of the National Academy of Sciences, 110(45):18327–18332, 2013.

Daniel M Bear, Elias Fan, Damian Mrowca, Yunzhu Li, Seth Alter, Aran Nayebi, Jeremy Schwartz, Li Fei-Fei Cao, Surya Ganguli, Daniel LK Yamins, et al. Physion: Evaluating physical prediction from vision in humans and machines. 2021.

Irving Biederman. Recognition-by-components: a theory of human image understanding. Psychological Review, 94(2): 115–147, 1987.

Jose-Luis Blanco, Juan-Antonio Fernández-Madrigal, and Javier Gonzalez. Toward a unified bayesian approach to hybrid metric–topological slam. IEEE Transactions on Robotics, 24(2):259–270, 2008.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. 2023.

Zefan Cai, Haoyi Qiu, Haozhe Zhao, Ke Wan, Jiachen Li, Jiuxiang Gu, Wen Xiao, Nanyun Peng, and Junjie Hu. From preferences to prejudice: The role of alignment tuning in shaping social bias in video diffusion models. arXiv preprint arXiv:2510.17247, 2025.

Vincent Cartillier, Zhile Ren, Neha Jain, Stefan Lee, Irfan Essa, and Dhruv Batra. Semantic mapnet: Building allocentric semantic maps and representations from egocentric views. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 964–972, 2021.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. International Conference on 3D Vision (3DV), 2017.

Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, and Ruslan Salakhutdinov. Learning to explore using active neural slam. In ICLR, 2020a.

Devendra Singh Chaplot, Dhiraj Prakashchand Gandhi, Abhinav Gupta, and Russ R Salakhutdinov. Object goal navigation using goal-oriented semantic exploration. Advances in Neural Information Processing Systems, 33: 4247–4258, 2020b.

François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert,

Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Google DeepMind. Veo: Google’s most capable video generation model. https://deepmind.google/technologies/veo/,

2024. Technical report. Google DeepMind. Veo 3: Generative video with native audio and cinematic control. Technical report, Google DeepMind, May 2025a. URL https://deepmind.google/models/veo/.

Google DeepMind. Gemini 3 pro image (nano banana pro): High-fidelity image generation with reasoning. Technical report, Google, 2025b. URL https://deepmind.google/models/gemini-image/pro/. Also covers Nano-banana model variants.

Matt Deitke, Winson Han, Alvaro Herrasti, Aniruddha Kembhavi, Eric Kolve, Roozbeh Mottaghi, Jordi Salvador, Dustin Schwenk, Eli VanderBilt, Matthew Wallingford, et al. Robothor: An open simulation-to-real embodied ai platform. In CVPR, 2020.

Hugh Durrant-Whyte and Tim Bailey. Simultaneous localization and mapping: part I. IEEE Robotics & Automation Magazine, 13(2):99–110, 2006. doi: 10.1109/MRA.2006.1638022.

Russell A Epstein, Alison Harris, Damian Stanley, and Nancy Kanwisher. The parahippocampal place area: Recognition, navigation, or encoding? Neuron, 23(1):115–125, 1999.

Jorge Fuentes-Pacheco, José Ruiz-Ascencio, and Juan Manuel Rendón-Mancha. Visual simultaneous localization and mapping: a survey. Artificial intelligence review, 43(1):55–81, 2015.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-math: A universal olympiad level mathematic benchmark for large language models. arXiv preprint arXiv:2410.07985, 2024.

James J Gibson. The Ecological Approach to Visual Perception. Houghton Mifflin, 1979. Rohit Girdhar and Deva Ramanan. Cater: A diagnostic dataset for compositional actions and temporal reasoning. In

ICLR, 2020.

Clara Gomez, Marius Fehr, Alex Millane, Alejandra C Hernandez, Juan Nieto, Ramon Barber, and Roland Siegwart. Hybrid topological and 3d dense mapping through autonomous exploration for large indoor environments. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pp. 9673–9679. IEEE, 2020.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The "something something" video database for learning and evaluating visual common sense. In ICCV, 2017.

Ziyu Guo, Xinyan Chen, Renrui Zhang, Ruichuan An, Yu Qi, Dongzhi Jiang, Xiangtai Li, Manyuan Zhang, Hongsheng Li, and Pheng-Ann Heng. Are video models ready as zero-shot reasoners? an empirical study with the mme-cof benchmark. arXiv preprint arXiv:2510.26802, 2025.

David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob

Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. Joao F Henriques and Andrea Vedaldi. Mapnet: An allocentric spatial memory for mapping environments. In

proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 8476–8484, 2018.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022a.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022b.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Kung-Hsiang Huang, Can Qin, Haoyi Qiu, Philippe Laban, Shafiq Joty, Caiming Xiong, and Chien-Sheng Wu. Why vision language models struggle with visual arithmetic? towards enhanced chart and geometry understanding. arXiv preprint arXiv:2502.11492, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. 2024.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.

Muhammad Zubair Irshad, Niluthpol Chowdhury Mithun, Zachary Seymour, Han-Pang Chiu, Supun Samarasekera, and Rakesh Kumar. Sasra: Semantically-aware spatio-temporal reasoning agent for vision-and-language navigation in continuous environments. arXiv preprint arXiv:2108.11945, 2021.

Michael Igorevich Ivanitskiy, Rusheb Shah, Alex F. Spies, Tilman Räuker, Dan Valentine, Can Rager, Lucia Quirke, Chris Mathwin, Guillaume Corlouer, Cecilia Diniz Behn, and Samy Wu Fung. A configurable library for generating and manipulating maze datasets, 2023. URL http://arxiv.org/abs/2309.10498.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick.

Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. 2017. Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011. Jing Yu Koh, Honglak Lee, Yinfei Yang, Jason Baldridge, and Peter Anderson. Pathdreamer: A world model for indoor

navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14738–14748, 2021. Kurt Konolige, Eitan Marder-Eppstein, and Bhaskara Marthi. Navigation in hybrid metric-topological maps. In 2011

IEEE International Conference on Robotics and Automation, pp. 3041–3047. IEEE, 2011. Stephen M Kosslyn. Image and Mind. Harvard University Press, 1980. Kuaishou. Kling: Large-scale video generation model. https://kling.kuaishou.com/, 2024. Technical report. Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and

think like people. Behavioral and Brain Sciences, 40:e253, 2017. Yann LeCun. A path towards autonomous machine intelligence. Open Review, 2022. Mingxiao Li, Zehao Wang, Tinne Tuytelaars, and Marie-Francine Moens. Layout-aware dreamer for embodied visual

referring expression grounding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 1386–1395, 2023.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond

- Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. 2024a.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond

- Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. 2024b.

Gary F Marcus. The Algebraic Mind: Integrating Connectionism and Cognitive Science. MIT Press, 2001.

- Mathematical Association of America. American invitational mathematics examination (aime) 2024. https://maa.

- org/maa-invitational-competitions/, 2024. American Invitational Mathematics Examination problems from 2024.

- Mathematical Association of America. American invitational mathematics examination (aime) 2025. https://maa.

- org/maa-invitational-competitions/, 2025. American Invitational Mathematics Examination problems from 2025.

Albert Michotte. The Perception of Causality. Methuen, 1963. Original work published 1946. OpenAI. Gpt-4o system card. Technical report, OpenAI, 2024a. URL https://openai.com/index/gpt-4o-system-card/. OpenAI. Video generation models as world simulators. OpenAI Technical Report, 2024b. URL https://openai.com/

research/video-generation-models-as-world-simulators. OpenAI. Sora 2 system card: Advanced video generation with physics simulation. Technical report, OpenAI, September

2025. URL https://openai.com/index/sora-2-system-card/. AJ Piergiovanni, Vincent Casser, Michael S Ryoo, and Anelia Angelova. Evolving space-time neural architectures for videos. In ICCV, 2020. Luis Piloto, Ari Weinstein, Peter Battaglia, and Matthew Botvinick. Intuitive physics grounded scene generation and understanding. In ICLR, 2022. Yiran Qin, Ao Sun, Yuze Hong, Benyou Wang, and Ruimao Zhang. Navigatediff: Visual predictors are zero-shot navigation assistants. arXiv preprint arXiv:2502.13894, 2025.

Qwen. Qwen-image: A 20b parameter mmdit-based visual generation model. arXiv preprint arXiv:2508.02324, 2024. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda

Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. 2021.

Santhosh K Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, et al. Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. 2021a.

Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. Habitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021b. URL https://arxiv.org/abs/2109.08238.

Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. Intphys 2019: A benchmark for visual intuitive physics understanding. 2021.

Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NeurIPS, 2016.

Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In ICCV, 2019.

Jeffrey Seely, Yuki Imajuku, Tianyu Zhao, Edoardo Cetin, and Llion Jones. Sudoku-bench: Evaluating creative reasoning with sudoku variants. arXiv preprint arXiv:2505.16135, 2025.

Hardik Shah, Jiaxu Xing, Nico Messikommer, Boyang Sun, Marc Pollefeys, and Davide Scaramuzza. Foresightnav: Learning scene imagination for efficient exploration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5236–5245, 2025.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

Elizabeth S Spelke and Katherine D Kinzler. Core knowledge. Developmental Science, 10(1):89–96, 2007. Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey Levine. Nomad: Goal masked diffusion policies for navigation

and exploration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 63–70. IEEE, 2024.

Edward C Tolman. Cognitive maps in rats and men. Psychological Review, 55(4):189–208, 1948. Jingqi Tong, Yurong Mou, Hangcheng Li, Mingzhe Li, Yongzhuo Yang, Ming Zhang, Qiguang Chen, Tianyi Liang,

Xiaomeng Hu, Yining Zheng, Xinchi Chen, Jun Zhao, Xuanjing Huang, and Xipeng Qiu. Thinking with video: Video generation as a promising multimodal reasoning paradigm. arXiv preprint arXiv:2511.04570, 2025.

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In CVPR, 2018.

Tomer D Ullman, Elizabeth S Spelke, Peter Battaglia, and Joshua B Tenenbaum. Mind games: Game engines as an architecture for intuitive physics. Trends in Cognitive Sciences, 21(8):586–599, 2017.

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. 2018a.

Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly.

Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018b. Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. 2016. Wan. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. Saim Wani, Shivansh Patel, Unnat Jain, Angel Chang, and Manolis Savva. Multion: Benchmarking semantic map

memory using multi-object navigation. Advances in Neural Information Processing Systems, 33:9700–9712, 2020. Taylor Webb, Keith J Holyoak, and Hongjing Lu. Emergent symbols through binding in external memory. 2023. Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank

Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025.

Jiajun Wu, Ilker Yildirim, Joseph J Lim, Bill Freeman, and Josh Tenenbaum. Physics 101: Learning physical object properties from unlabeled videos. In BMVC, 2015.

Xin Wu, Zheng Qi, Zexiang Liu, and He Wang. 3d shape reconstruction from 2d images with disentangled attribute flow. In CVPR, 2022.

Fanyi Xiao, Yong Jae Lee, Kristen Grauman, Jitendra Malik, and Christoph Feichtenhofer. Audiovisual slowfast

networks for video recognition. 2020. Dawei Xu, Yifan Zhang, and Bo Han. Can llms solve arc-agi tasks? arXiv preprint arXiv:2411.00993, 2024. Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and

transformers. arXiv preprint arXiv:2104.10157, 2021. Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. In ICLR, 2020. Jeffrey M Zacks and Barbara Tversky. Event structure in perception and conception. Psychological Bulletin, 127(1): 3–21, 2001. Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu. Abstract spatial-temporal reasoning via probabilistic abduction and execution. 2021. Yichen Zhong, Yunzhu Liu, Junyu Liang, Jessica Hodgins, and Antonio Torralba. Learning 3d part assembly from a single image. ECCV, 2020.

Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. Navgpt-2: Unleashing navigational reasoning capability for large vision-language models. In European Conference on Computer Vision, pp. 260–278. Springer, 2025.

Meng-Jiun Zhou, Zheng Shou, and Bernard Ghanem. Video timeline modeling for news story understanding. 2022. Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven

visual navigation in indoor scenes using deep reinforcement learning. 2017.

