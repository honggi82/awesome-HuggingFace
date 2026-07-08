# arXiv:2602.13294v3[cs.CV]21May2026

## VisPhyWorld: Probing Physical Reasoning via Code-Driven Video Reconstruction

##### Jiarong Liang1,* Max Ku1,* Ka-Hei Hui2 Ping Nie3 Wenhu Chen1

1University of Waterloo 2Autodesk AI Lab 3Independent Researcher

#### Abstract

Evaluating whether Multimodal Large Language Models (MLLMs) genuinely reason about physical dynamics remains challenging. Most existing benchmarks rely on recognition-style protocols such as Visual Question Answering (VQA) and Violation of Expectation (VoE), which can often be answered without committing to an explicit, testable physical hypothesis. We propose VisPhyWorld, an execution-based framework that evaluates physical reasoning by requiring models to generate executable simulator code from visual observations. By producing runnable code, the inferred world representation is directly inspectable, editable, and falsifiable. This separates physical reasoning from rendering. Building on this framework, we introduce VisPhyBench, comprising 209 evaluation scenes derived from 108 physical templates and a systematic protocol that evaluates how well models reconstruct appearance and reproduce physically plausible motion. Our pipeline produces valid reconstructed videos in 97.7% of benchmark runs before fallback. Experiments show that while state-of-the-art MLLMs achieve strong semantic scene understanding, they struggle to accurately infer physical parameters and to simulate consistent physical dynamics. Our code is available https://github.com/TIGER-AI-Lab/VisPhyWorld

GT Three.js P5.js SVG Manim

|[Figure 1]|
|---|

|[Figure 2]|
|---|

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

0.1s

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

|[Figure 10]|
|---|

- 0.6s
- 1.2s

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

Figure 1: MLLMs struggle to simulate physical dynamics. Under the same inputs, rigid-body backends (Three.js/P5.js) produce more physically consistent rollouts, whereas non-physics backends (SVG/Manim) often exhibit implausible motion or contact artifacts such as interpenetration.

#### 1 Introduction

Recent advances in Multimodal Large Language Models (MLLMs) have led to impressive performance on a wide range of visual and language tasks [16]. However, assessing whether such models exhibit principled physical reasoning remains challenging. Existing evaluation protocols often rely on recognition-based queries or surface-level judgments, which can obscure whether correct outputs arise from coherent physical reasoning or from learned visual correlations [13, 54]. Most benchmarks

Preprint. *Equal contribution. Correspondence to jiarongliangcs@gmail.com, m3ku@uwaterloo.ca and wenhu.chen@uwaterloo.ca.

|Traditional VQA Paradigm<br><br>VisPhyWorld Paradigm ( Executable Hypothesis)<br><br>[Figure 16]<br><br>[Figure 17]<br><br>[Figure 18]<br><br>Multimodal LLM<br><br>Text Answer<br><br>Text Answer(e.g., “Option A, the red ball is moving. “）<br><br>Hidden Physical Hypothesis<br><br>[Figure 19]<br><br>Evaluates implicit, unverifiable reasoning.<br><br>[Figure 20]<br><br>Input Video (Initial Frames)<br><br>[Figure 21]<br><br>Object Detection<br><br>[Figure 22]<br><br>Multimodal LLM<br><br>function animate() { var block = net(); var frames = 1000; var ... for ( i = n < 0 ; i++) {<br><br>... }<br><br>}<br><br>Executable Simulation Code (e.g., JavaScript / Python Physics)<br><br>Physics Engine Simulation (Rendering)<br><br>[Figure 23]<br><br>[Figure 24]<br><br>[Figure 25]<br><br>Evaluates explicit, testable physical hypothesis.<br><br>Ground Truth Video Generated Video<br><br>Evaluation Metrics (Physical, Semantic, Perceptual.....)<br><br>[Figure 26]|
|---|

- Figure 2: Unlike traditional VQA paradigms, VisPhyWorld accesses physical understanding evaluation by requiring MLLMs to actively reconstruct scenes via executable code, offering superior reasoning explainability compared to traditional paradigms.

probe physical understanding through passive recognition tasks such as Visual Question Answering (VQA)-style or Violation of Expectation (VoE)-inspired recognition tasks (e.g. CLEVRER [62], GRASP [27], MVPBench [30])). These settings can reward dataset-driven guessing, encouraging memorized priors and surface-level pattern matching rather than genuine causal physical understanding [49, 29]. This challenge is particularly acute for MLLMs, which typically output only text and therefore do not provide predictive likelihoods or measures of surprise commonly used to evaluate generative world models [19]. We therefore argue that in this context, evaluation should require reconstruction and re-simulation, forcing models to commit to an explicit physical hypothesis rather than merely select an answer or text reasoning. We propose VisPhyWorld, a paradigm shift: using executable code as a test of physical understanding, as illustrated in Figure 2. VisPhyWorld probes the physical reasoning capabilities of MLLMs through visual-to-code reconstruction. Given two key frames (and optionally object detections), the model produces executable simulation code that recreates the scene and rolls it forward to synthesize future frames as shown in Figure 3. This process not only produces the video but does so in a fully interpretable and editable manner. Beyond the rendered video, VisPhyWorld exposes the generated code itself as a reasoning artifact, making the model’s physical logic directly inspectable.

We also introduce VisPhyBench, a standardized evaluation suite with a systematic protocol that assesses how well models reconstruct appearance and reproduce physically plausible motion across complementary perspectives. Our investigation reveals a critical insight: while current state-of-the-art LLMs excel at semantic recognition, they exhibit significant limitations in fine-grained physical comprehension, often failing to parameterize simple Newtonian dynamics correctly even in a simple

- 2D setting, let alone in 3D environments. In summary, our contributions are threefold:

- (1) We propose VisPhyWorld, a framework that uses LLMs to interpret raw video frames and generate executable simulation code for predicting future motion. To our knowledge, this is the first paradigm that evaluates physical reasoning in MLLMs through code reconstruction and re-simulation. By making object states and dynamics explicit, VisPhyWorld provides a direct and interpretable view of a model’s physical understanding.
- (2) We introduce VisPhyBench, a unified evaluation protocol comprising 209 scenes derived from 108 physical templates that assesses physical understanding through the lens of codedriven resimulation in both 2D and 3D scenes, integrating metrics from different aspects.
- (3) We provide an in-depth analysis of current MLLMs, demonstrating that despite their linguistic prowess, they fail to grasp the fundamental dynamics of real-world motion. Our results reveal a critical gap: while models can accurately describe scene contents, they struggled to reconstruct the scene in a way that conformed to the laws of physics, indicating

[Figure 27]

##### Figure 3: VisPhyWorld Framework. (1) System & Data Construction: We process raw video

sequences to extract key frames (Istart,Ilater) and detection contexts using multimodal agents. (2) Pipeline & Simulation Flow: An LLM-based agent performs motion analysis and generates raw executable code, which is then sanitized and rendered. (3) Evaluation Benchmark: We propose a multi-metric benchmark integrating semantic and physical fidelity to compare generated videos Xˆ with ground truth X. (4) A Detailed Case: An example illustrating how VisPhyWorld translates a collision scene (red ball hits block stack) into executable simulation logic.

that they rely on superficial visual pattern matching rather than a grounded understanding of physical causality.

#### 2 Related Work

Intuitive physics. Understanding the world is commonly studied through physical reasoning tasks that probe models’ ability to infer object dynamics, interactions, and causal relationships from visual input [39, 17]. Inspired by findings from developmental psychology showing that infants exhibit sensitivity to physical violations [4], prior work on intuitive physics investigates whether models can anticipate physically plausible outcomes from visual observations. This has been studied through video prediction benchmarks that evaluate the consistency of predicted future dynamics, as well as Violation-of-Expectation (VoE) paradigms [52, 38, 27], which assess whether physically implausible events elicit higher predictive surprise. These approaches are well suited to generative world models with explicit prediction objectives. However, they do not naturally extend to MLLMs, which primarily produce textual outputs rather than predictive distributions and therefore cannot be evaluated using likelihood-based or generative video protocols [19].

Efforts on several datasets and benchmarks have been made [51, 62, 8], including Phyre [5, 33], Physion [9, 58], and IntPhys [52, 11], have been proposed to evaluate intuitive physics using videos generated from physics engines. More recent benchmarks such as PhysicsIQ [42], PhyGenBench [40], and WorldModelBench [32] extend this setting to generative video models, focusing on whether predicted videos exhibit physically plausible and temporally consistent dynamics. In parallel, researchers have developed MLLM–based evaluators [41], such as VideoPhy [6, 7] and VideoScore [24, 25], to assess physical understanding in multimodal models. These approaches typically formulate evaluation

- as recognition-based tasks like VQA. While effective for probing high-level physical knowledge, such protocols make it difficult to determine whether model performance reflects genuine physical reasoning or reliance on appearance-based heuristics and dataset-specific biases. Our framework complements previous works by requiring explicit and executable physical hypotheses evaluated through simulation. Table 1 compares our work with prior works.

LPIPS ↓

LPIPS ↓

CLIP Img↑

DINO ↑

CLIP Cap↑

BERT Score↑

RAFT EPE↓

Engine

CLIP-Img ↑

Gemini ↑

GPT-5

GPT-4.1

Three.js 0.16 0.89 0.86 0.27 0.84 31.08 SVG 0.16 0.92 0.87 0.26 0.85 30.06 Blender 0.25 0.74 0.71 0.31 0.83 30.18 P5.js 0.24 0.84 0.79 0.23 0.81 36.71 Manim 0.28 0.85 0.78 0.25 0.84 31.73

Gemini Claude Qwen3 SVD

RAFT-EPE ↓

DINO ↑

Veo

BERTScore-F1 ↑ CLIP-Cap ↑

- Figure 4: Left: cross-engine GPT-5 stress-test scores. Right: radar summary of key VisPhyBench metrics comparing code-driven reconstruction against pixel-space baselines. Engine choice changes absolute visual scores, but the scene-level motion difficulty ranking is stable across engines. Therefore, our conclusions are not merely driven by backend-specific coding artifacts.

Executable World Representations for Visual and Motion Generation. Representing visual scenes as executable programs is a foundational paradigm in computer graphics and simulation, where structured code specifies objects, motion, and physical interactions to enable interpretable and controllable world representations [15]. Recent advances in multimodal large language models have begun to enable the generation of executable code for visual content. Early efforts primarily focus on static visualizations, such as data plots and vector graphics, translating high-level semantic intent into low-level graphical instructions [18, 61, 22, 44, 43, 53, 60, 35]. These methods demonstrate the feasibility of using code as a structured intermediate between language and visual output. Subsequent work extends code-based generation to animations and motion, enabling programmatic specification of object trajectories and temporal behaviors [66, 23, 36, 37, 31]. While these approaches show that MLLM can generate executable programs that produce coherent motion, they are primarily designed for content creation or presentation, and rarely assess whether the generated programs correspond to physically consistent dynamics or reflect an underlying understanding of physical laws. In contrast to prior work that treats executable visual generation as an end goal, our work uses executable world representations as a diagnostic interface for physical reasoning. Rather than evaluating visual realism or animation quality, we assess whether models can reconstruct and resimulate physically consistent dynamics from visual observations to enable direct inspection.

- Table 1: VisPhyWorld uniquely turns physical reasoning into an executable hypothesis and enables multimetric, diagnostic evaluation beyond relative scoring. Gen.: requires future visual generation; MLLM: evaluates MLLM outputs; Exec.: requires an executable physical hypothesis.

Benchmark Gen. MLLM Exec. Scoring Output PHYRE [5] × × × Relative Action CLEVRER [62] ✓ × × Relative QA / VoE IntPhys [52] ✓ × × Relative VoE PhyGenBench [40] ✓ ✓ × Relative QA MVP [30] × ✓ × Relative QA PhysicsIQ [42] ✓ ✓ × Relative QA WorldModelBench [32] ✓ ✓ × VLM judge Video IntPhys2 [11] × ✓ × Relative VoE PhyWorld [28] ✓ × × Reconstruction Video VisPhyWorld (Ours) ✓ ✓ ✓ Reconstruction Code

#### 3 VisPhyWorld

We introduce VisPhyWorld, a framework that uses MLLM to interpret visual observations and reconstruct the underlying physical scene as executable code. We evaluate the rendered outputs under a unified protocol using a multi-metric suite.

##### 3.1 Problem Definition

We focus on 2D and 3D physical scenes involving common interactions, e.g., ball collisions and box sliding. We represent each scene as a sequence of image frames I with three color channels as in Equation 1, where H, W, and T denote frame height, frame width, and number of frames.

X = (It)Tt=1, It ∈ R3×H×W, (1)

Input. Given a scene, the MLLM backbone receives {Istart,Ilater,D}. We select two key frames from X, where Istart = It

and Ilater = It

, typically corresponding to an early frame and a later frame

s

l

(e.g., ts = 1, tl = 10). Optionally, we provide a detection context D for Istart listing objects with categories, bounding boxes, and coarse attributes (see Appendix C.2). We obtain D with GPT-5.2 [47] on the first frame and parsing its output into a structured object list; if unavailable, we set D = ∅.

Outputs. VisPhyWorld produces four interpretable artifacts: (i) a textual motion analysis A ∈ Ytext; (ii) a machine-readable first-frame JSON specification S encoding object layout and inferred

parameters; (iii) an executable program C ∈ Ycode; and (iv) a rendered video Xˆ = (Iˆt)Ttˆ=1 obtained by executing the executable program C.

##### 3.2 VisPhyWorld Architecture

(Istart,Ilater,D) −−−→fLLM (A,S,C) −−−→Rphys X.ˆ (2)

VisPhyWorld implements a composite mapping as stated in Equation 2. We include A as a lightweight, text-only diagnostic of the model’s basic scene understanding: whether it can correctly describe salient motions and interactions between the key frames, separately from code generation. We treat C as an explicit, falsifiable physical hypothesis: executing it with a renderer Rphys under a fixed configuration yields Xˆ, separating hypothesis construction from execution and enabling controlled comparisons across LLM backbones. To ensure a well-defined evaluation, we apply lightweight validation prior to execution and allow a single automatic repair attempt upon failure; if both attempts fail, we fall back to a minimal valid scene. Further implementation details, including prompt templates and renderer settings, are deferred to Appendix C, with robustness analyses in Appendix C.5.

##### 3.3 Benchmark, Metrics, and Baselines

Dataset Construction. We build on and enrich the 2D data from the PhyWorld dataset [28], using the PHYRE engine [5] for rendering to form the 2D subset of VisPhyBench. We additionally curate a

- 3D subset rendered with Three.js and simulated using Cannon.js for rigid-body dynamics. Overall, VisPhyBench comprises 108 templates and 209 videos, each paired with first-frame JSON annotations. VisPhyBench scenes are annotated with coarse difficulty levels. We construct a small test split by subsampling from the full dataset to enable rapid sanity checks and lightweight evaluation. Eight STEM-trained annotators rate each raw clip on a 1–5 scale (higher indicates greater difficulty), and we use the mean rating as the final difficulty score. The mean score is then mapped to easy, medium, or hard using fixed, interpretable cutoffs aligned with the rating scale (easy = 1–2, medium = 3, hard

= 4–5). The resulting distribution is naturally skewed, reflecting the relative rarity of challenging interactions in our template set: the sub split contains 114 easy, 67 medium, and 28 hard scenes, while the test split contains 29 easy, 17 medium, and 3 hard scenes. Scenes cover diverse object configurations (stacks, ramps, collisions) and motion patterns (slides, bounces, topples). For the 2D subset, the camera is fixed and orthographic; for the 3D subset, we use a fixed perspective camera. In both settings, the background is set to white to focus on physical dynamics. Since templates are executable programs instantiated by sampling seeds, we summarize template composition and object statistics in Appendix C.3. Inputs (Istart,Ilater,D) follow Section 3.1.

Evaluation Metrics. We report per-metric means over all scenes and group metrics into five families. (1) Reconstruction and perceptual quality. We report PSNR [26] and SSIM [59] for frame-wise reconstruction, together with LPIPS [65], FSIM [63], VSI [64], and DISTS [14] to compute on aligned frame pairs. (2) Visual semantic consistency. We compute CLIP-based image similarity (CLIP-Img) [50] and DINO feature similarity [12], which emphasize object identity and scene layout beyond exact pixels. (3) Text–video and analysis-text consistency. We compute CLIP text–image similarity (CLIP-Cap) [50] between the analysis and sampled video frames, and use ROUGE-L [34]

and BERTScore-F1 [67] to compare the analysis with a GPT-generated reference description derived from the ground-truth video. (4) Motion and physical plausibility. We use RAFT-based optical-flow diagnostics [55] with automatic temporal alignment, reporting RAFT end-point error (EPE) and the estimated temporal offset (and, when relevant, flow magnitude and angular statistics) to quantify motion consistency. Because flow discrepancy alone can be misleading as a proxy for physical plausibility, we interpret RAFT metrics jointly with holistic perceptual/physics judgments rather than using RAFT-EPE in isolation; as discussed in Section 4.1. (5) Subjective overall quality. We use a Gemini-2.5-Pro video–video judge (1–10) with a textual justification, and report a model-blind human evaluation score (1–5); details and calibration against other judges are provided in Appendix D.6. We separately report pipeline success rate based on whether a valid video is produced.

Video Model Baselines. We include Stable Video Diffusion (SVD) img2vid [10], conditioned only on Istart, Sora-2, Veo-3.1, and Cosmos-Predict2.5-2B [1], a recent world-foundation model targeted

- at physical understanding, with the direct video models conditioned on the same visual frames and prompts allowed by their APIs.

##### 3.4 Engine Evaluation and Selection

We evaluate four rendering backends, i.e., Three.js [57], P5.js [48], SVG (Scalable Vector Graphics), and Manim [56], to understand how the choice of visualization engine affects multimodal LLM-based reconstruction. As shown in Figure 1, a consistent pattern emerges: Three.js and P5.js achieve markedly better reconstruction and motion fidelity because they support native integration with rigid-body physics solvers, allowing the generated programs to offload gravity, contact constraints, friction, and collision response to a physically grounded engine. In contrast, SVG and Manim are primarily non-physics-based rendering systems: they excel at deterministic drawing and scripted animation, but lack intrinsic rigid-body dynamics. In our experimental setting, SVG and Manim serve as non-interactive, script-based backends and do not expose a comparable physics API or closed-loop simulation stepping; consequently, as illustrated in Figure 1, they often yield physically implausible behaviors, such as objects remaining static or interpenetrating. Importantly, this gap suggests a limitation of current MLLM: without access to a true physics solver, they fail to consistently infer and apply Newtonian dynamics from visual evidence, and instead revert to heuristic motion scripting. For this work, we therefore prioritize Three.js and P5.js so that our evaluation emphasizes physically grounded re-simulation rather than non-physical animation artifacts. To check that the conclusions are not merely backend-specific coding artifacts, we run a cross-engine stress test with GPT-5 across Three.js, P5.js, SVG, Manim, and Blender. The results are shown in the left panel of Figure 4, with correlation matrices reported in Appendix E.4. Scene-level RAFT-EPE rankings are highly correlated across engines, with a mean Spearman correlation of ρ = 0.84 and all pairwise p < 0.05, while appearance metrics are only moderately correlated. These results suggest that our conclusions are not solely driven by backend-specific coding artifacts.

#### 4 Experiments

Evaluation setup. We evaluate VisPhyWorld and all baselines on VisPhyBench. Local rendering, post-processing, and metric evaluation were run on a workstation with 4 NVIDIA RTX A6000 GPUs. API-based models were queried through their provider-hosted services. For each configuration, we generate one video per scene, compute all metrics, and report per-metric means over the evaluation split; unless otherwise stated, higher is better. We consider five multimodal LLM backbones: GPT-

###### 5 [46], GPT-4.1 [45], Gemini-3-Pro [21], Claude Sonnet 4.5 [3], and Qwen3-VL-Plus [2]. We evaluate two code backends, Three.js [57] and P5.js [48]. All LLM runs use the same prompt and two key frames; only the model and engine identifiers change. For each run, we aggregate metrics into five families: reconstruction & perceptual quality, visual semantic consistency, text–video & analysis-text consistency, motion (automatic RAFT-based metrics [55]), and subjective overall quality (Gemini2.5-Pro [20] judge). We observe consistent trends across scenes; per-scene metric distributions and significance analyses are reported in Appendix E.2, Fig. 21.

##### 4.1 Overall leaderboard.

- Table 2 summarizes performance across five metric families on VisPhyBench. Overall, most models achieve strong reconstruction and perceptual scores and maintain reasonable visual-semantic con-

- Table 2: Overall leaderboard on VisPhyBench. T denotes Three.js and P denotes p5.js. Higher is better (↑), lower is better (↓). “–” denotes unavailable or inapplicable metrics.

Reconst. & Perceptual

Human Preference LPIPS↓ CLIP-Img↑ DINO↑ CLIP-Cap↑ BERT-F1↑ RAFT-EPE↓ Gemini↑ Human↑

Visual Semantic Consistency

Text–Video & Analysis-Text

Motion / Physical Plausibility

Holistic Quality

Model

GPT-5T 0.17 0.89 0.86 0.26 0.84 33.65 3.50 4.33 GPT-5P 0.29 0.81 0.76 0.23 0.84 34.34 3.52 3.67 GPT-4.1T 0.18 0.89 0.83 0.26 0.85 33.71 3.06 4.17

- GPT-4.1P 0.35 0.75 0.68 0.22 0.83 37.70 2.15 2.33 Gemini-3-ProT 0.14 0.90 0.84 0.26 0.85 36.20 3.80 4.67 Gemini-3-ProP 0.33 0.75 0.67 0.22 0.84 33.10 2.35 3.50 Claude-S4.5T 0.16 0.90 0.83 0.26 0.85 36.20 2.39 3.83 Claude-S4.5P 0.33 0.76 0.71 0.22 0.82 34.14 2.56 2.67 Qwen3-VL-PlusT 0.22 0.87 0.78 0.27 0.85 35.05 2.12 2.50 Qwen3-VL-PlusP 0.55 0.64 0.55 0.20 0.84 20.82 1.46 1.83

SVD (img2vid) 0.34 0.67 0.65 0.25 – 45.46 1.43 1.83 Cosmos-2.5-2B 0.27 0.73 0.65 – – 33.12 1.25 1.16 Sora-2 0.20 0.87 0.87 0.26 – 34.91 2.35 2.83 Veo-3.1 0.21 0.86 0.88 0.27 – 32.71 2.62 4.50

sistency; these results support our central claim that, once the task is cast as executable hypotheses under a fixed physics engine, most modern MLLM can reconstruct synthetic physical events with high fidelity, and the remaining gaps become diagnosable rather than opaque.

Visual and language-based reasoning can diverge. Our benchmark jointly evaluates visual reconstruction, visual semantics, and language-mediated reasoning, and we find that these dimensions do not always move together. Some model–backend pairs achieve strong perceptual and semantic alignment with the reference frames, as shown by low LPIPS and high CLIP-Img and DINO scores. These results suggest that they can recover object identities and global layouts well from the visual input. For example, Gemini-3-Pro with Three.js achieves the lowest LPIPS and the highest CLIP-Img score, and it also obtains the strongest pixel-level reconstruction in Appendix Table 15. However, strong visual reconstruction does not necessarily imply equally strong language-based reasoning. For instance, GPT-4.1 with Three.js achieves the highest BERT-F1 score, even though its LPIPS is higher than that of Gemini-3-Pro with Three.js. This gap shows that VisPhyBench is not only measuring overall model quality. Instead, it separates a model’s ability to see the scene from its ability to explain the scene. In contrast, video-generation baselines such as Veo-3.1 do not produce executable simulators, which makes their intermediate states less transparent and harder to diagnose.

The code backend affects reconstruction quality. Across LLMs, Three.js variants generally reconstruct the scenes better than their p5.js counterparts, as shown by lower LPIPS scores in most model pairs, even though they use the same inputs and prompts. For example, for GPT-5, using Three.js reduces LPIPS by nearly 40% and improves SSIM from 0.74 to 0.94. Qualitatively, this leads to more stable object identities and cleaner scene layouts, as shown in Figure 1. Since the physical setup is kept the same, this gap suggests that the backend interface and program structure affect how well a model can translate visual evidence into an executable simulation. In other words, the benchmark captures not only physical reasoning, but also whether the chosen simulator provides a clear and controllable way for the model to express that reasoning.

Pixel-space baselines show a complementary but less interpretable profile. They can score competitively on some feature semantics, but their failures are harder to attribute to specific physical causes, such as friction, restitution, or contact timing, since the generation process does not expose interpretable latent variables. Veo-3.1 attains reasonable semantic similarity, for example reaching DINO ∼ 0.88, yet it does not expose an explicit simulator state for diagnosis or controlled interventions and often exhibits deficiencies in physical understanding by producing trajectories with implausible motion or contact events (see Sec. 4.3). This pattern also holds for world-foundation models designed for physical understanding. Cosmos-Predict2.5-2B has the lowest visual scores and the worst Gemini and human ratings among all baselines. Qualitatively, it often fails to keep object identities consistent across frames. This shows that even physics-aware video generators can still struggle with stable physical rollout, while code-based simulators remain more transparent and controllable. Conversely, our code-driven approach maintains competitive semantic and motion

scores while exposing executable states; e.g., GPT-5 (threejs) achieves DINO 0.8556 with RAFT-EPE 33.6473. This enables controlled interventions (e.g., varying friction/mass while holding layout fixed) that can isolate whether an error originates from object discovery, state initialization, or contact modeling, aligning with our goal of turning “physics understanding” into a testable, executable object.

Motion and holistic physical plausibility. Assessing physical plausibility requires combining motion-level metrics with holistic perceptual judgment. We report RAFT-EPE to measure opticalflow discrepancy, as detailed in Appendix D, and use a Gemini-2.5-Pro judge to aggregate visual and physical cues into a single holistic score, as detailed in Appendix D.6. RAFT-EPE is useful for measuring trajectory agreement, but relying on it alone can be misleading. For example, Qwen3VL-Plus with p5.js obtains the lowest RAFT-EPE, 20.82, despite weak reconstruction fidelity in Appendix Table 15 and a low Gemini score of 1.46. In contrast, Gemini-3-Pro with Three.js achieves the highest Gemini score, 3.80, consistent with its strong visual alignment, although scores around 3–4 still indicate partial scene recovery with remaining motion or contact errors. Therefore, we treat strong physical understanding as requiring both favorable motion agreement and perceptually coherent outcomes. The Gemini judge further provides textual justifications that comment on collisions, contact consistency, missing motion, and implausible dynamics, offering a qualitative sanity check alongside the quantitative flow metric. Together, these metrics provide a multi-view diagnostic evaluation of visual reconstruction, language-based reasoning, and physical plausibility under executable simulation.

##### 4.2 Robustness and Ablation Studies

Detection context ablation. We remove the structured object list D while keeping the same two input frames and prompt style. As reported in Appendix E.3, removing the detection context causes a clear drop in LPIPS and Gemini score across all models, while the other metrics change much less. This suggests that detection context mainly helps object discovery and state initialization, rather than directly solving physical reasoning. We therefore include detection context to reduce the confound from object localization.

Judge replacement. We also test whether our conclusions depend on the particular evaluator. VideoScore2 [25] provides an additional non-LLM evaluation protocol, while Appendix E.3 reports

- GPT-5.4 and Qwen3-VL-Plus as alternative holistic judges. Although the absolute score scales differ across judges, the main ordering remains stable and weak video baselines remain weak. This supports the robustness of our conclusions to judge replacement.

Forecast-only ablation. The forecast-only ablation in Appendix E.3 removes Ilater and uses only Istart. When Ilater is removed, visual metrics change little and RAFT-EPE even slightly improves, but the Gemini physical-plausibility score drops more clearly. This indicates that Ilater mainly provides useful physical context. Without it, models can still generate visually reasonable motion, but are less likely to recover the correct physical outcome.

- Table 3: MLLMs often produce visually plausible reconstructions, but fail to recover the physical parameters needed for faithful simulation. Model rows report MAE, with values after ± denoting standard deviation across scenes.

Dynamic mass

Initial speed

Model Gravity Restitution Friction

GT mean 9.82 0.15 0.40 1.25 2.59 GPT-5 0 0.17±0.14 0.10±0.07 0.15±0.17 2.07±2.69 Gemini 3 Pro 0 0.18±0.13 0.11±0.07 0.59±0.88 1.69±2.42 Claude Sonnet 4.5 0 0.16±0.05 0.10±0.03 0.26±0.42 2.39±3.23

Direct parameter recovery. Video matching is still an indirect test of physical understanding. A model may produce a plausible-looking rollout without recovering the latent physical parameters that determine the motion, such as mass, friction, restitution, gravity, and initial velocity. To measure this more directly, we extract these quantities from the generated simulator code and compare them with the ground-truth parameters using mean absolute error. Table 3 shows that models recover gravity because it is a stable default, but they make non-trivial errors on contact parameters and much larger errors on initial speed. Gemini-3-Pro also has a substantial dynamic-mass error despite strong visual

Qwen3 VL-Plus

SVD (img2vid)

GT GPT-5

Veo-3.1

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

- 0.1s
- 1.2s
- 2.3s
- 3.2s

|[Figure 33]|
|---|

|[Figure 34]|
|---|

|[Figure 35]|
|---|

|[Figure 36]|
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

|[Figure 42]|
|---|

|[Figure 43]|
|---|

|[Figure 44]|
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

|[Figure 47]|
|---|

- Figure 5: GPT-5 reconstructs object identities and collision dynamics most faithfully over time. Pixel-space baselines (Veo-3.1 and SVD/img2vid) generate trajectories with implausible motion/contact events.

GT GPT-5

Gemini 3-Pro

Claude 4.5

Qwen3 VL-Plus

- 0.1s

- 0.6s
- 1.1s

- 1.6s

|[Figure 48]|
|---|

|[Figure 49]|
|---|

|[Figure 50]|
|---|

|[Figure 51]|
|---|

|[Figure 52]|
|---|

|[Figure 53]|
|---|

|[Figure 54]|
|---|

|[Figure 55]|
|---|

|[Figure 56]|
|---|

|[Figure 57]|
|---|

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

Figure 6: This example highlights the dissociation between semantic alignment and correct physical dynamics: although Claude shows clear reconstruction deficits, its visual-semantic scores remain relatively high.

reconstruction. These results show that current MLLMs can often reconstruct the appearance of a physical event, but still fail to infer the quantitative parameters needed for faithful simulation.

- 4.3 Case Study

We present a case study, shown in Fig. 5, featuring gravity-driven multi-body interactions that require physical reasoning; the diagnostic panel is provided in Appendix Fig. 7. GPT-5 in Three.js shows strong physical grounding by correctly simulating the collision dynamics, achieving Gemini 10.0 with DINO 0.926. In contrast, pixel-space baselines such as Veo-3.1 achieve high semantic similarity, reaching DINO 0.835, but fail on event logic with Gemini 2.0, indicating plausible appearance with hallucinated dynamics. The case also motivates joint evaluation: Qwen3-VL-Plus attains low RAFT-EPE, 121.30 versus 118.66 for GPT-5, by producing static or empty outputs, yet is penalized by Gemini with a score of 4.0. These results show that optical-flow errors alone are insufficient; credible physical understanding requires both correct motion and holistic visual coherence.

- Figure 6 extends our diagnostic analysis beyond 2D templates to a perspective-rendered 3D scene with depth-dependent contacts and occlusions. Consistent with our 2D findings, we observe the same conclusion in 3D: strong appearance matching alone does not guarantee physically faithful dynamics. Valid physical understanding is only evidenced when both motion dynamics and holistic visual coherence are satisfied. For example, Claude-4.5 and Qwen3-VL-Plus exhibit clear reconstruction deviations in this sample, yet their visual-semantic scores do not separate substantially from other models, highlighting a dissociation between semantic alignment and correct physical dynamics. More broadly, the 3D setting is noticeably more challenging for current MLLMs, underscoring the necessity of incorporating 3D scenes when evaluating reconstruction-based physical reasoning.

#### 5 Conclusions

In this work, we introduced VisPhyWorld, a framework that advances the evaluation of physical understanding by requiring MLLMs to reconstruct scenes as executable code, thereby decoupling visual mimicry from physically grounded reasoning. By benchmarking state-of-the-art models on our proposed VisPhyBench, we exposed a consistent dichotomy: while current models excel at semantic scene parsing, they struggle with precise physical parameterization; when forced to commit to an executable hypothesis, models that rely on pixel-space generation often fail to reproduce even basic Newtonian dynamics. Our findings suggest that progress toward robust world modeling may benefit from moving beyond purely statistical pattern matching in pixel space toward hybrid representations that ground visual perception in verifiable, executable physical laws. We believe this direction offers a path toward more transparent and verifiable evaluations of physical understanding.

#### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Alibaba Cloud. Alibaba cloud model studio: Visual understanding (qwen-vl). https://www. alibabacloud.com/help/en/model-studio/vision, 2026. Accessed: 2026-01-15.
- [3] Anthropic. Claude sonnet 4.5. https://www.anthropic.com/news/claude-sonnet-4-5,

2025. Accessed: 2026-01-15.

- [4] Renée Baillargeon, Elizabeth S. Spelke, and Stanley Wasserman. Object permanence in five-month-old infants. Cognition, 20(3):191–208, 1985. ISSN 0010-0277. doi: https://doi. org/10.1016/0010-0277(85)90008-3. URL https://www.sciencedirect.com/science/ article/pii/0010027785900083.
- [5] Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. Phyre: A new benchmark for physical reasoning. Advances in Neural Information Processing Systems, 32, 2019.
- [6] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In International Conference on Learning Representations, volume 2025, pages 102075–102121, 2025.
- [7] Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. arXiv preprint arXiv:2503.06800, 2025.
- [8] Fabien Baradel, Natalia Neverova, Julien Mille, Greg Mori, and Christian Wolf. Cophy: Counterfactual learning of physical dynamics. arXiv preprint arXiv:1909.12000, 2019.
- [9] Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. arXiv preprint arXiv:2106.08261, 2021.
- [10] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [11] Florian Bordes, Quentin Garrido, Justine T Kao, Adina Williams, Michael Rabbat, and Emmanuel Dupoux. Intphys 2: Benchmarking intuitive physics understanding in complex synthetic environments. arXiv preprint arXiv:2506.09849, 2025.
- [12] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.
- [13] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.
- [14] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, page 1–1, 2020. ISSN 1939-3539. doi: 10.1109/tpami.2020.3045810. URL http://dx.doi.org/10.1109/TPAMI.2020.3045810.
- [15] James D. Foley, Andries van Dam, Steven K. Feiner, and John F. Hughes. Computer Graphics: Principles and Practice. Addison-Wesley, second edition, 1996. ISBN 0201848406.
- [16] Chaoyou Fu, Yi-Fan Zhang, Shukang Yin, Bo Li, Xinyu Fang, Sirui Zhao, Haodong Duan, Xing Sun, Ziwei Liu, Liang Wang, et al. Mme-survey: A comprehensive survey on evaluation of multimodal llms. arXiv preprint arXiv:2411.15296, 2024.

- [17] Pascale Fung, Yoram Bachrach, Asli Celikyilmaz, Kamalika Chaudhuri, Delong Chen, Willy Chung, Emmanuel Dupoux, Hongyu Gong, Hervé Jégou, Alessandro Lazaric, et al. Embodied ai agents: Modeling the world. arXiv preprint arXiv:2506.22355, 2025.
- [18] Timur Galimzyanov, Sergey Titov, Yaroslav Golubev, and Egor Bogomolov. Drawing pandas: A benchmark for llms in generating plotting code. In 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR), pages 503–507. IEEE, 2025.
- [19] Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun. Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831, 2025.
- [20] Google. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025. URL https://arxiv.org/abs/2507.06261.
- [21] Google AI for Developers. Gemini 3 developer guide. https://ai.google.dev/ gemini-api/docs/gemini-3, 2026. Accessed: 2026-01-15.
- [22] Kanika Goswami, Puneet Mathur, Ryan Rossi, and Franck Dernoncourt. Plotgen: multiagent llm-based scientific data visualization via multimodal retrieval feedback. In Companion Proceedings of the ACM on Web Conference 2025, pages 1672–1676, 2025.
- [23] Liu He, Yizhi Song, Hejun Huang, Pinxin Liu, Yunlong Tang, Daniel Aliaga, and Xin Zhou. Kubrick: Multimodal agent collaborations for synthetic video generation. arXiv preprint arXiv:2408.10453, 2024.
- [24] Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, et al. Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 2105–2123, 2024.
- [25] Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, et al. Videoscore2: Think before you score in generative video evaluation. arXiv preprint arXiv:2509.22799, 2025.
- [26] Q. Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics Letters, 44:800 – 801, 02 2008. doi: 10.1049/el:20080522.
- [27] Serwan Jassim, Mario Holubar, Annika Richter, Cornelius Wolff, Xenia Ohmer, and Elia Bruni. Grasp: A novel benchmark for evaluating language grounding and situated physics understanding in multimodal language models. arXiv preprint arXiv:2311.09048, 2023.
- [28] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024.
- [29] Aryan Keluskar, Amrita Bhattacharjee, and Huan Liu. Do llms understand ambiguity in text? a case study in open-world question answering. In 2024 IEEE International Conference on Big Data (BigData), pages 7485–7490. IEEE, 2024.
- [30] Benno Krojer, Mojtaba Komeili, Candace Ross, Quentin Garrido, Koustuv Sinha, Nicolas Ballas, and Mahmoud Assran. A shortcut-aware video-qa benchmark for physical understanding via minimal video pairs. arXiv preprint arXiv:2506.09987, 2025.
- [31] Max Ku, Thomas Chong, Jonathan Leung, Krish Shah, Alvin Yu, and Wenhu Chen. Theoremexplainagent: Towards multimodal explanations for llm theorem understanding. arXiv e-prints, pages arXiv–2502, 2025.
- [32] Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph Gonzalez, et al. Worldmodelbench: Judging video generation models as world models. Advances in Neural Information Processing Systems, 38, 2026.

- [33] Shiqian Li, Kewen Wu, Chi Zhang, and Yixin Zhu. I-phyre: Interactive physical reasoning. In International Conference on Learning Representations, volume 2024, pages 28195–28215, 2024.
- [34] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/.
- [35] Kevin Qinghong Lin, Yuhao Zheng, Hangyu Ran, Dantong Zhu, Dongxing Mao, Linjie Li, Philip Torr, and Alex Jinpeng Wang. Vcode: a multimodal coding benchmark with svg as symbolic visual representation. arXiv preprint arXiv:2511.02778, 2025.
- [36] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded image-to-video generation. In European Conference on Computer Vision, pages 360–378. Springer, 2024.
- [37] Jiaxi Lv, Yi Huang, Mingfu Yan, Jiancheng Huang, Jianzhuang Liu, Yifan Liu, Yafei Wen, Xiaoxin Chen, and Shifeng Chen. Gpt4motion: Scripting physical motions in text-to-video generation via blender-oriented gpt planning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1430–1440, 2024.
- [38] Francesco Margoni, Luca Surian, and Renée Baillargeon. The violation-of-expectation paradigm: A conceptual overview. Psychological Review, 131(3):716–748, 2024. doi: 10.1037/rev0000450.
- [39] Andrew Melnik, Robin Schiewer, Moritz Lange, Andrei Muresanu, Mozhgan Saeidi, Animesh Garg, and Helge Ritter. Benchmarks for physical reasoning ai. arXiv preprint arXiv:2312.10728, 2023.
- [40] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.
- [41] Saman Motamed, Minghao Chen, Luc Van Gool, and Iro Laina. Travl: A recipe for making video-language models better judges of physics implausibility. arXiv preprint arXiv:2510.07550, 2025.
- [42] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles? In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 948–958, 2026.
- [43] Yuansheng Ni, Songcheng Cai, Xiangchao Chen, Jiarong Liang, Zhiheng Lyu, Jiaqi Deng, Kai Zou, Ping Nie, Fei Yuan, Xiang Yue, et al. Viscoder2: Building multi-language visualization coding agents. arXiv preprint arXiv:2510.23642, 2025.
- [44] Yuansheng Ni, Ping Nie, Kai Zou, Xiang Yue, and Wenhu Chen. Viscoder: Fine-tuning llms for executable python visualization code generation. arXiv preprint arXiv:2506.03930, 2025.
- [45] OpenAI. Introducing gpt-4.1 in the api. https://openai.com/index/gpt-4-1/, 2025. Accessed: 2026-01-15.
- [46] OpenAI. Introducing gpt-5. https://openai.com/index/introducing-gpt-5/, 2025. Accessed: 2026-01-15.
- [47] OpenAI. GPT-5.2 Model (openai api documentation). https://platform.openai.com/ docs/models/gpt-5.2, 2025. Accessed 2026-01-06.
- [48] p5.js contributors. p5.js. https://p5js.org/, 2026. Accessed: 2026-01-15.
- [49] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 2006–2017, 2024.

- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [51] Nazneen Fatema Rajani, Rui Zhang, Yi Chern Tan, Stephan Zheng, Jeremy Weiss, Aadit Vyas, Abhijit Gupta, Caiming Xiong, Richard Socher, and Dragomir Radev. Esprit: Explaining solutions to physical reasoning tasks. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7906–7917, 2020.
- [52] Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. Intphys: A framework and benchmark for visual intuitive physics reasoning. arXiv preprint arXiv:1803.07616, 2018.
- [53] Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. Starvector: Generating scalable vector graphics code from images and text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16175–16186, 2025.
- [54] Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, et al. Phyx: Does your model have the" wits" for physical reasoning? arXiv preprint arXiv:2505.15929, 2025.
- [55] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020.
- [56] The Manim Community Developers. Manim – Mathematical Animation Framework, April

2024. URL https://www.manim.community/.

- [57] three.js contributors. Three.js – javascript 3d library. https://threejs.org/, 2026. Accessed: 2026-01-15.
- [58] Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel Bear, Chuang Gan, Josh Tenenbaum, Dan Yamins, Judith Fan, and Kevin Smith. Physion++: Evaluating physical scene understanding that requires online inference of different physical properties. Advances in Neural Information Processing Systems, 36:67048–67068, 2023.
- [59] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612,

2004. doi: 10.1109/TIP.2003.819861.

- [60] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Jiaxu Zhang, Liao Wang, Gang Yu, Xinjun Ma, and Yu-Gang Jiang. Omnisvg: A unified scalable vector graphics generation model. arXiv preprint arxiv:2504.06263, 2025.
- [61] Zhiyu Yang, Zihan Zhou, Shuo Wang, Xin Cong, Xu Han, Yukun Yan, Zhenghao Liu, Zhixing Tan, Pengyuan Liu, Dong Yu, et al. Matplotagent: Method and evaluation for llm-based agentic scientific data visualization. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11789–11804, 2024.
- [62] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019.
- [63] Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang. Fsim: A feature similarity index for image quality assessment. IEEE Transactions on Image Processing, 20(8):2378–2386, 2011. doi: 10.1109/TIP.2011.2109730.
- [64] Lin Zhang, Ying Shen, and Hongyu Li. Vsi: A visual saliency-induced index for perceptual image quality assessment. IEEE Transactions on Image Processing, 23(10):4270–4281, 2014. doi: 10.1109/TIP.2014.2346028.

- [65] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [66] Sharon Zhang, Jiaju Ma, Jiajun Wu, Daniel Ritchie, and Maneesh Agrawala. Editing motion graphics video via motion vectorization and transformation. ACM Trans. Graph., dec 2023. doi: 10.1145/3618316.
- [67] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675, 2019.

### Appendix

#### A Limitations and Discussion

While VisPhyWorld shows promising results on physics-aware video generation and evaluation, it has limitations. First, our experiments are conducted on synthetic, simulator-driven scenes with controlled object layouts and camera motion, so generalization to high-resolution, in-the-wild videos remains untested. Fundamentally limited by the capabilities of current MLLMs and the complexity of modern engines, VisPhyWorld can reliably generate code only for relatively simple rigid-body scenes: although we experimented with large engines such as Unreal Engine, we found that, without human intervention, existing MLLMs cannot, within a small fixed number of calls, autonomously produce and repair simulation code to render a stable, visually plausible video in these more complex environments. Finally, we currently target relatively short clips with moderate motion complexity, and do not address long-horizon interactions, complex 3D reasoning, or stylized or heavily cluttered scenes, which we leave for future work. Future work could integrate stronger 3D perception for scene initialization, and agentic workflows with domain-specific fine-tuning.

#### B Case Study

###### Case Study 1

Task: You are a coding-focused assistant whose primary job is to

OUTPUT RUNNABLE CODE. Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 68]|
|---|

|[Figure 69]|
|---|

|[Figure 70]|
|---|

|[Figure 71]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

First-frame JSON: ..... { "id": "lightgray_ball", "category": "circle", "color_rgb": [192, 206,

214], "position": { "center_x": 275.16, "center_y": 58.34 }, "bbox": {

"x_min": 254, "y_min": 36, "x_max": .... { "id": "red_ball", "category": "circle", "color_rgb": [240, 78, 67].....

###### Model Analysis: The scene has a

GT Analysis: The scene is set in a mostly empty vertical space

white background with a static

slanted black bar on the left, a tall slim gray vertical bar near

with no visible ground plane,

but there is a U-shaped container near the bottom made of two light-gray...... there is a single

the middle, and a gray cup at the bottom made of two slightly

tilted side walls and a flat base..... The black bar is....

black diagonal bar, slanted

upward from left to right......

[Figure 72]

Generated Video

- Figure 7: Full diagnostic case-study panel. VisPhyWorld exhibits strong physical grounding by reconstructing the scene, intermediate reasoning, generated code, and rendered rollout.

###### Case Study 2

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

###### First-frame JSON: .....

{ "id": "red_ball", "category": "circle", "color_rgb":[240, 78, 67], "position": { "center_x": 280.04,

"center_y": 147.27 }, "bbox": { "x_min": 240, "y_min": 102, "x_max": ....

{ "id": "blue_ball", "category": "circle", "color_rgb": [22, 118, 241], "position": { "center_x": 383.73....

Model Analysis: The scene is an open white

GT Analysis:The scene consists of a blank,

area with an implied flat ground along the

white background with no visible ground line,

very bottom of the frame; there are no black

platforms, walls, or other static supports.

fixed bars or obstacles. For physics

There are no ramps, pegs, or obstacles; the

containment we treat the canvas edges as

space appears open and unobstructed in

invisible boundaries, with a static floor and

both horizontal and vertical directions. The

side walls so bodies cannot leave the image;

objects seem to float in this empty 2D field,

the red_ball is the largest circle, centered a

implying that any motion will be governed

bit right of the vertical midline near the upper

mainly by gravity in the vertical direction and

third; the blue_ball is medium sized near the

free translation horizontally.\n\nThere are

lower right;.....

three dynamic circular objects ......

[Figure 77]

Generated

Video

Figure 8: A detailed case study (ID 2).

###### Case Study 3

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

###### First-frame JSON: .....

{ "id": "gray_ball", "category": "circle", "color_rgb": [0, 0, 0], "position": { "center_x": 260.62,

"center_y": 121.6 }, "bbox": { "x_min": 212, "y_min": 196, "x_max": ....

{ "id": "black_ball_1", "category": "circle", "color_rgb": [0, 0, 0], "position": { "center_x": 203.84,....

Model Analysis: The scene is set within a

GT Analysis: The scene is set in empty

2D vertical plane with a gravitational field

space with a plain white background and no

pulling downwards. Static structures include

visible ground or enclosing walls. There are

a ground plane (implied at the bottom), a

a few straight line segments that act as static

slanted black rectangular bar in the lower-left

structural elements: on the left, a thick black

corner acting as a ramp, and a vertical pillar

line is placed diagonally rising from lower left

standing on the ground near the bottom-

to upper right; near the bottom center, a thin

center. Three small static black circles are

vertical light-gray line extends upward.

arranged in a diagonal line above the ramp,

......There are four circular dynamic objects.

though they appear to be clear of the main

Near the top center is a medium-sized light-

action zone. ......

gray disk......

[Figure 82]

Generated

Video

Figure 9: A detailed case study (ID 3).

###### Case Study 4

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

###### First-frame JSON: .....

- { "id": "red_ball", "category": "circle", "color_rgb": [240, 78, 67], "position": { "center_x": 261.02, "center_y": 116.06 }, "bbox": { "x_min": 288, "y_min": 256, "x_max": ....

{ "id": "line_segment", "category": "circle", "color_rgb": [184, 200, 209], "position": { "center_x":....

GT Analysis: The scene is set against a

plain white background with no visible floor

Model Analysis: The scene is set up as a

or ground platform. Near the bottom center

2D physics puzzle on a vertical plane. Static

there is a U-shaped static gray structure,

structures include a U-shaped gray container

resembling a container or goal, formed by

resting at the bottom center and a decorative

two vertical gray bars with an open top. Near

arc of five small black circles on the left side,

the upper central region is a large red

which act as fixed obstacles or markers. The ground plane is effectively the bottom edge

circular object connected to a short vertical

gray bar or rod that appears fixed or hinged;

of the image. ......

this rod suggests a possible obstacle or

paddle-like structure attached to ......

[Figure 87]

Generated

Video

Figure 10: A detailed case study (ID 4).

###### Case Study 5

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

###### First-frame JSON: .....

- { "id": "red_ball", "category": "circle", "color_rgb": [240, 78, 68], "position": { "center_x": 48.52, "center_y": 123.48}, "bbox": { "x_min": 262, "y_min": 222, "x_max": ....

{ "id": "gray_ball", "category": "circle", "color_rgb": [0, 0, 0], "position": { "center_x": 241.7,....

GT Analysis: The scene is set in a mostly

Model Analysis: The scene presents a 2D

empty vertical space with a white

physics setup with a mix of static structures

background and no visible ground plane. At

and dynamic spheres. The static

the lower right region there are several tall,

environment at the bottom consists of four

thin, vertical light-gray bars acting as fixed

gray vertical pillars of increasing height from

obstacles or posts. One bar is on the far

left to right, and a tall, narrow X-shaped

right, another slightly left of it, and two more

structure to their left. Suspended in the

on the left side that cross each other in an

center are three small, fixed black circles

“X” pattern, forming a loose barrier or gate.

arranged in a descending diagonal, acting as

There are no ramps or horizontal platforms;

rigid pivots or obstacles ......

these vertical elements are the primary......

[Figure 92]

Generated

Video

Figure 11: A detailed case study (ID 5).

###### Case Study 6

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

###### First-frame JSON: .....

{ "id": "red_ball", "category": "circle", "color_rgb": [240, 78, 68], "position": { "center_x": 279, "center_y": 146 }, "bbox": { "x_min": 312, "y_min": 202, "x_max": ....

{ "id": line_segment", "category": "circle", "color_rgb": [192, 202, 208], "position": { "center_x": 257.14,"center_y": 285.48....

GT Analysis: The scene appears to be a

sparse environment with a white

Model Analysis: The scene is a 2D physics

puzzle contained within a 512x512 area.

background. On the left side, there is a single

thick black straight segment that is tilted

Static structures include a thick black

upward from lower left toward upper right,

diagonal barrier on the left acting as a ramp,

a curved cluster of six small black circular

acting as an inclined rigid bar or ramp. No

clear ground line or enclosing boundaries are

pegs in the center-left, and a U-shaped

visible; the rest of the space is open. There

container (cup) composed of three light-blue

bars at the bottom center. A ground plane is implied at the bottom edge......

are no obvious fixed supports other than this

black bar.Dynamic objects include: (1) a

large red circle located in the upper-right......

[Figure 97]

Generated

Video

Figure 12: A detailed case study (ID 6).

###### Case Study 7

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

[Figure 102]

GT Video

[Figure 103]

Generated Video

GT Analysis: The scene is set on a flat, light

gray rectangular platform that serves as the

Model Analysis: The scene is set on a

large, flat, grey ground plane. In the center, a

ground plane. Near the right side of the

platform there is a U-shaped......Closer to the

black rectangular ramp is inclined upwards

center of the platform lies a long, thin, black

from left to right, acting as a launchpad. To the right of the ramp, there is a static grey

rectangular plank or wedge resting on the

surface and oriented diagonally, with its left

structure shaped like a U-channel or open

end closer to the viewer and its right end

container, consisting of a bottom slab and

two vertical walls (one closer to the ramp,

pointing slightly away......There is one clearly

dynamic object: a small green sphere

one further away)......

positioned ......

Figure 13: A detailed case study (ID 7).

###### Case Study 8

Task: You are a coding-focused assistant whose primary job is to OUTPUT RUNNABLE CODE.

Your response MUST ALWAYS include executable code. Natural language is only allowed in the

ANALYSIS section; the rest must be code.....

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

GT Input frame 1 &10

Generated Output frame 1 &10

[Figure 108]

GT Video

[Figure 109]

Generated Video

GT Analysis: The scene consists of a large,

Model Analysis: The scene consists of a

flat, gray rectangular ground plane occupying

static grey environment featuring a large

most of the view. Near the center of this

ground plane and a U-shaped container

plane there is a U-shaped static structure

(trough) resting on it. The container is

made of the same gray materia......There are

composed of a flat rectangular base and two

two dynamic objects, both spheres. One

vertical side walls, oriented to form an open

sphere is red and the other is blue...... The

channel.Two dynamic spheres are

red sphere is on the left and a bit lower, while

suspended in mid-air above the container. The first is a reddish-pink sphere positioned

the blue sphere is on the right and slightly

higher. By the later frame, both spheres have

lower......

moved slightly downward under gravity......

Figure 14: A detailed case study (ID 8).

#### C Reproducibility Details

This appendix documents the reproducibility-critical components of VisPhyWorld: (i) the prompting protocol used to elicit an executable scene hypothesis, (ii) the optional detection context format, (iii) deterministic execution constraints for rendering, and (iv) robustness protocols that ensure a well-defined evaluation.

##### C.1 Prompting Protocol for Scene Hypotheses

VisPhyWorld uses a single-call prompting protocol that asks the model to (1) summarize the observed motion between two keyframes and (2) propose an executable scene hypothesis that reproduces the event. To ensure comparability across models, we enforce a fixed output format and a small set of execution constraints (e.g., a single canvas and bounded duration), which are handled by the renderer (Appendix C.4). The full prompt template is shown in Figure 15.

Scene Analysis & Code Generation Prompt

You are an expert in 2D physics, rigid-body simulation, and JavaScript. Given two key frames from a short video and an optional list of detected objects, your goals are:

- (1) Scene and motion analysis (3–8 sentences):

- • Describe the main objects (shapes, colors, approximate sizes) visible in the first frame.
- • Describe how these objects move between the first and the second frame (who moves, who stays still, collisions, stacks that topple, etc.).
- • Explain the likely physical causes of the motion (gravity, contact forces, friction, impulses).

- (2) Simulation code generation: Produce ONE complete HTML document that:

- • Imports the required rendering/physics libraries (or uses provided local copies if available).
- • Creates a 2D-like scene with an orthographic camera.
- • Adds rigid bodies (balls, boxes, planks) matching the layout of the first frame.
- • Initializes positions and orientations so that the first rendered frame closely matches the first image.
- • Assigns velocities or impulses consistent with the observed motion between the two images.
- • Runs a physics simulation and renders frames to a single canvas element.
- • Uses the provided recording helper to export a finite-duration clip.

Return your answer in the following format:

- (A) Analysis section: plain English paragraphs.
- (B) Code section: a single fenced block “‘html

<!DOCTYPE html> ...</html> “‘ Do not include any other Markdown fences or extra HTML documents.

- Figure 15: Full multimodal LLM prompt template used by VisPhyWorld for both motion analysis and code generation.

##### C.2 Detection Context D

To reduce ambiguity in object discovery and initialization, VisPhyWorld can optionally provide a structured detection context D for the first keyframe Istart. D is a per-sample JSON annotation containing a list of objects with coarse geometry and appearance attributes. All coordinates are in pixel space with origin at the image top-left (x increases rightward, y increases downward).

##### 3D Scene Analysis & Code Generation Prompt (dataset_3D)

You are an expert in 3D rigid-body physics, Three.js, and JavaScript. Given two key frames from a short video (rendered from a fixed camera) and an optional list of detected objects, your goals are:

##### (1) Scene and motion analysis (3–8 sentences):

- • Describe the major objects and supports visible in the first frame (shapes, colors, approximate sizes, and relative depth if evident).
- • Describe how these objects move between the first and the second frame, focusing on contacts, impacts, and constraint satisfaction.
- • Explain the likely physical causes of the motion (gravity, contact forces, friction, impulses).

###### (2) Simulation code generation: Produce ONE complete HTML document that:

- • Uses Three.js for rendering and Cannon.js for rigid-body simulation.
- • Creates a 3D scene with a fixed perspective camera (no camera motion).
- • Initializes objects and supports so the first rendered frame closely matches the first image.
- • Assigns velocities or impulses consistent with the observed motion between the two images.
- • Runs the physics simulation deterministically and renders frames to a single canvas element.
- • Uses the provided recording helper to export a finite-duration clip.

Return your answer in the following format:

- (A) Analysis section: plain English paragraphs.
- (B) Code section: a single fenced block “‘html

<!DOCTYPE html> ...</html> “‘ Do not include any other Markdown fences or extra HTML documents.

- Figure 16: 3D prompt variant used for dataset_3D. It mirrors Figure 15 but switches the execution target from 2D (orthographic) to 3D (fixed perspective) while keeping the output format identical.

Schema. Each detected object provides: (i) a unique identifier id; (ii) a coarse category (e.g., circle, rectangle, line, u_shape); (iii) an RGB color triplet color_rgb; (iv) a tight bounding box bbox as {x_min, y_min, x_max, y_max, width, height}; (v) a centroid position.center_x/center_y; (vi) a coarse size descriptor (e.g., radius_pixels for circles, length_pixels/thickness_pixels for bars); and (vii) an optional orientation.angle_deg for elongated primitives.

##### Example.

{

"image_size": {"width": 512, "height": 512}, "coordinate_system": {"origin": "top_left", "x_axis": "to_right"...}, "objects": [

{"id":"red_ball","category":"circle","color_rgb":[240,78,70], "position":{"center_x":363.6,"center_y":155.2}, "bbox":{"x_min":348,"y_min":140,"x_max":378,"y_max":172,"width":32...}, "size":{"radius_pixels":16.5}}

] }

##### C.3 VisPhyBench Templates and Stochasticity

VisPhyBench templates are defined as executable PHYRE-style task scripts. Unlike static assets, each template is instantiated by sampling seeds (e.g., object placements and sizes), so a single rendered snapshot does not capture the full diversity. We therefore summarize object composition over the full sub split using the detection context D on Istart (see Table 4).

Table 4: Object category statistics on VisPhyBench.

Category Scenes (%) Scenes (count) Objects (count) circle 100.0 191 779 line 83.2 159 344 rectangle 62.8 120 321 u_shape 24.6 47 47 triangle 6.3 12 16 composite_shape 7.3 14 16

3D templates. In addition to PHYRE-style 2D scripts, we include a set of programmatic 3D templates implemented in Three.js + Cannon.js. These 3D templates use simple rigid-body primitives (e.g., spheres, boxes, ramps, barriers) under a fixed perspective camera and white background, and are designed to probe depth-aware contacts and occlusions not present in purely 2D scenes. Because D is defined in 2D pixel space from a first-frame detector, the category statistics above are reported for the 2D portion of the split; for the 3D subset we instead rely on the executable template specification and deterministic rendering protocol (Appendix C.4).

##### C.4 Deterministic Execution and 2D Constraint

VisPhyWorld executes each generated scene hypothesis under a fixed, deterministic configuration to ensure comparability across models.

Canonicalization and validation. Raw model outputs may contain extraneous text or malformed markup. Before execution, we extract the HTML payload (from a fenced “‘html block when present, otherwise the outermost <html>...</html> segment), and canonicalize it into a standard executable template that injects the required libraries and a trusted recording helper. We additionally validate basic requirements (e.g., existence of a drawable canvas and finite numeric states). Retry and fallback behaviors are described in Appendix C.5.

Execution contract. For each sample, the renderer produces a fixed-length clip Xˆ at the reference frame rate and duration associated with that sample. All runs use a fixed physics time step and a fixed camera configuration; as a result, variability in Xˆ is attributable to the generated hypothesis rather than nondeterministic execution.

- 2D constraint. Although the underlying physics engine supports full 3D dynamics, we restrict motion to a 2D plane by (i) initializing all bodies with z = 0 and (ii) projecting the state back to the plane at each simulation step (clamping out-of-plane position and angular components to zero). This avoids uncontrolled 3D degrees of freedom while preserving rigid-body contact dynamics.
- 3D execution. For our 3D subset, we disable the 2D clamping rule and execute full 3D rigid-body dynamics with the same deterministic protocol (fixed physics time step, fixed recording duration, and fixed camera parameters). To preserve comparability across models, we keep the camera static and normalize all rendered videos to match the reference FPS, duration, and resolution of the corresponding ground-truth clip.

##### C.5 Robustness: Automatic Retry and Fallback

To handle syntax errors or runtime exceptions in model-generated programs, we implement a lightweight robustness protocol that ensures evaluation is well-defined for all samples.

Error-conditioned single-step repair. If the initial program fails to execute (e.g., syntax error, missing canvas, or runtime exception), we capture execution diagnostics (e.g., JavaScript console logs and error traces), summarize them, and provide the summary to the model for a single repair attempt.

##### Deterministic Rendering Protocol (High-Level)

Given a model-generated scene hypothesis C, the renderer produces the output clip Xˆ under a fixed protocol:

- • Parse & canonicalize: Extract the HTML payload and wrap it into a standard executable template with fixed library versions and a trusted recorder.
- • Validate: Check minimal execution requirements (e.g., a drawable canvas and finite numeric states).
- • Execute deterministically: Run physics with a fixed time step and a fixed orthographic camera, producing frames at the sample’s reference FPS.
- • Enforce 2D: Initialize with z = 0 and clamp out-of-plane components each step.
- • Export: Record a fixed-duration clip and convert it to a standard format for downstream evaluation.

- Figure 17: High-level deterministic rendering protocol used in VisPhyWorld. Low-level implementation details are included in the released codebase.

Fallback and well-defined evaluation. If the repair attempt also fails, we execute a minimal hand-crafted fallback template (Figure 18) that guarantees a valid canvas and finite motion. This prevents missing outputs and ensures the evaluation pipeline does not crash; such samples receive correspondingly poor scores on the metrics.

Success criteria. We distinguish two notions of success. Model-success counts a sample as successful only if the model-generated hypothesis executes and produces a non-empty clip without invoking the fallback. System-success additionally counts fallback clips as successful, and is used only to guarantee that the evaluation pipeline is well-defined. Unless otherwise stated, success rates reported in the main paper use Model-success. The 97.7% success rate in the abstract refers to this model-success criterion before fallback, aggregated over benchmark runs; fallback clips are used only to keep metric computation well-defined when generation fails.

##### Fallback Template (Simplified Sketch)

The fallback template guarantees a valid canvas and finite motion when model generation fails:

<!DOCTYPE html> <html lang="en"> <head> <meta charset="UTF-8" /> <title>VisPhyWorld Fallback Scene</title> <script src="three.min.js"></script> <script src="cannon.min.js"></script> <script src="recording.js"></script> </head> <body style="margin:0;overflow:hidden;"> <canvas id="visphyworld-canvas"></canvas> <script> // Setup renderer, camera, scene, lights // Create a flat ground plane and one spherical body // Run simulation loop and export a finite-duration clip </script> </body> </html>

Figure 18: High-level structure of the fallback template used when both model attempts fail.

#### D Evaluation Metrics: Definitions & Protocols

This appendix defines the metric families used in the main paper. All metrics are computed per scene and then averaged over the evaluated split. Unless otherwise noted, frame-wise metrics are computed after temporal alignment (Appendix D.1).

##### D.1 Default Evaluation Hyperparameters

We report our default evaluation hyperparameters for reproducibility. Unless otherwise stated, we uniformly sample frames every sample_every=3 frames for all frame-wise metrics. For temporal alignment, we use a coarse-to-fine strategy with coarse offset search up to max_offset=30 (in sampled frames), a stack window window=3, and offset penalty offset_penalty=0.05. The coarse search uses downsample=64, top_k=5, and max_samples=16. When DTW is enabled, we compute frame features using a 48 × 48 grayscale thumbnail and a step penalty of 0.005.

Table 5: Default evaluation hyperparameters used throughout the paper.

Setting Value Frame sampling sample_every=3 Coarse offset search max_offset=30 (sampled frames) Stack refinement window window=3 Offset penalty offset_penalty=0.05 Coarse downsample downsample=64 Top-k candidates top_k=5 Max coarse samples max_samples=16 DTW feature size 48 × 48 grayscale DTW step penalty 0.005

##### D.2 Reconstruction & Perceptual Quality

- • PSNR and SSIM: Computed frame-wise between aligned reference and generated videos. SSIM is averaged across the Y channel and RGB channels.
- • LPIPS, FSIM, VSI, DISTS: Deep and structural perceptual metrics computed on aligned frames. We use the piq library implementation.

##### D.3 Visual Semantic Consistency

- • CLIP-Img: Cosine similarity between CLIP (ViT-B/32) embeddings of reference and generated frames, measuring high-level semantic/layout consistency.
- • DINO Similarity: Cosine similarity of DINO ViT features, which is more sensitive to object structure and less biased by text supervision than CLIP.

##### D.4 Text–Video & Analysis Consistency

- • CLIP-Cap: Similarity between the generated motion-analysis text and the generated video frames.
- • Text Metrics (ROUGE, BERTScore): We compare the generated analysis against an automatically produced reference description of the original video (generated by a strong LLM). This validates whether the model correctly perceives and verbalizes the events in the input video.

##### D.5 Motion & Physical Plausibility

- • RAFT Optical Flow: We compute End-Point Error (EPE), flow magnitude difference, and angular error between the optical flow fields of the reference and generated videos.
- • Temporal Alignment: We use a coarse-to-fine alignment strategy (Figure 19) combining offset search and Dynamic Time Warping (DTW) to handle temporal shifts before metric computation.

##### D.6 Subjective Quality (Gemini Judge)

We employ Gemini-2.5-Pro as a holistic judge. The prompt (Figure 20) asks the model to compare the reference and generated videos and assign a score (1–10) with a justification. Scores are intentionally conservative: a score near 3–4 indicates that a video captures some objects or layout but still contains clear motion, contact, or temporal-consistency errors. We do not rescale these scores.

##### Temporal Alignment Procedure

- (1) Coarse Search: Downsample frames to grayscale vectors. Compute correlation for offsets ±30 frames. Keep top-k candidates.
- (2) Stack Refinement: Build frame stacks (w = 3). Minimize cost = MSE + 0.5 MAE + 0.1 Angular.
- (3) DTW: Run Dynamic Time Warping on low-res features to align variable-speed sequences.

Figure 19: Temporal alignment procedure used before computing frame-wise metrics.

##### Gemini-based Physics & Video Consistency Prompt

You are an expert evaluator of physical simulations and video quality. Compare the provided reference video (Ground Truth) with the generated video. Your goal is to determine if the generated video accurately reconstructs the physical event shown in the reference. Focus on the following dimensions:

- • Physical Plausibility (Crucial): Do the objects obey rigid-body physics laws (gravity, collisions, friction) in the given setting (2D or 3D)? Are there any “hallucinations” such as objects passing through each other (ghosting), floating unnaturally, or failing to move when hit?
- • Motion Consistency: Does the trajectory, speed, and timing of the movement align with the reference?
- • Scene Semantics: Are the correct objects (color, shape, count) present in the correct layout?
- • Visual Fidelity: Overall clarity, ignoring minor rendering style differences if the physics is correct.

Return a JSON object with the keys:

- • score: integer between 1 and 10.

- – 10: Perfect physical and visual match.
- – 1: Physical laws are violated (e.g., phantom collision, static scene when motion is expected), even if the image looks realistic.

- • justification: Brief explanation, specifically pointing out any physical violations if present.

- Figure 20: Prompt template used for Gemini-based evaluation. Note that the prompt is explicitly designed to penalize physical violations (e.g., incorrect collision logic), ensuring the score reflects physical understanding rather than just perceptual similarity.

Human evaluation. To calibrate the automatic judge, we additionally conduct a model-blind human evaluation with six STEM graduate students. Raters compare generated videos against ground-truth videos in randomized order and assign a 1–5 score. The human ranking shows the same broad trend as the automatic metrics: strong code-driven models and Veo-3.1 remain top-tier, while weaker baselines such as SVD and Cosmos are rated low.

VideoScore2 calibration. We also checked VideoScore2 [25] for the three code-driven models with available outputs.

Table 6: VideoScore2 calibration is reported on a 1–5 scale; higher is better.

Model Visual Temporal Dynamic Alignment Physical

Claude Sonnet 4.5 2.44 2.20 2.75 2.04 2.12 GPT-5 2.51 2.23 2.76 2.09 2.17 Gemini-3-Pro 2.57 2.27 2.66 2.24 2.14

#### E Detailed Experimental Results

##### E.1 VisPhyBench Difficulty Stratification

The main paper reports the difficulty distribution; here we provide additional details on the stratification and split construction. All annotators are graduate students with STEM backgrounds.

##### E.2 Per-Scene Distributions and Significance (Sub Split)

Mean scores can obscure whether improvements are driven by a small subset of scenes. To address this, we report (i) per-scene metric distributions via boxplots (Figure 21) and (ii) paired bootstrap confidence intervals over per-scene differences (Table 7). We use paired resampling because all methods are evaluated on the same set of scenes (N = 209), and define “mean improvement” so that positive values indicate better performance by VisPhyWorld (GPT-5, threejs), taking metric direction into account (↑ / ↓).

LPIPS

CLIP-Img

DINO

CLIP-Cap

1.0

1.0

0.1

0.9

0.9

0.30

0.8

0.2

0.8

0.25

0.7

0.3

0.7

0.6

0.20

0.4

0.6

0.5

0.4

0.5

0.5

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

RAFT-EPE

BERTScore-F1

Gemini

0

- 1

- 2

- 3

- 4

- 5

- 6

- 7

0.86

25

0.85

50

0.84

75

0.83

100

0.82

125

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

GPT 5 threejsGPT 5 p5jsVeo 3.1 SVD

Figure 21: Per-scene boxplot distributions on VisPhyBench for representative metric families (higher is better unless marked ↓).

##### E.3 Additional Ablation Studies

We report supporting ablation details that isolate whether VisPhyWorld’s conclusions depend on renderer self-repair, auxiliary detection context, a single judge model, or access to the later conditioning frame.

Iterative self-repair. If the first generation+render attempt fails, VisPhyWorld appends a concise renderer error log tail and the previous attempt to the next LLM call, then retries once. Table 8 reports success rates with and without this retry mechanism. This suggests that some failures are correctable surface-level issues such as missing canvas hooks, minor API misuse, or initialization errors rather than irrecoverable physical-scene understanding failures.

Detection context. Table 9 compares matched runs with and without the structured detection context

- D. Removing D consistently worsens LPIPS and lowers Gemini physical-plausibility scores, while CLIP/DINO/BERTScore move much less. This indicates that detection context primarily reduces the object discovery and initialization burden; it does not by itself solve the physical dynamics.

Alternative judge models. Tables 11 and 10 replace the Gemini-2.5-Pro holistic judge with GPT5.4 and Qwen3-VL-Plus judges. We selected Qwen3-VL-Plus because it natively supports video input. In contrast, neither the GPT-5 version used in our paper nor the latest GPT-5.4 API currently

- Table 7: Paired bootstrap confidence intervals (VisPhyBench sub, N = 209). “Mean improvement” is defined so that positive values indicate VisPhyWorld (GPT-5, threejs) performs better (for ↓ metrics we compute baseline−ours; for ↑ metrics ours−baseline).

Metric Comparison Mean improvement 95% bootstrap CI LPIPS↓ VisPhyWorld (GPT-5, p5js) 0.1143 [0.0743, 0.1567] LPIPS↓ Veo-3.1 0.0365 [0.0310, 0.0420] LPIPS↓ SVD (img2vid) 0.1674 [0.1581, 0.1764] CLIP-Img↑ VisPhyWorld (GPT-5, p5js) 0.0754 [0.0477, 0.1044] CLIP-Img↑ Veo-3.1 0.0365 [0.0285, 0.0446] CLIP-Img↑ SVD (img2vid) 0.2258 [0.2159, 0.2355] DINO↑ VisPhyWorld (GPT-5, p5js) 0.0957 [0.0643, 0.1290] DINO↑ Veo-3.1 -0.0276 [-0.0340, -0.0217] DINO↑ SVD (img2vid) 0.2036 [0.1917, 0.2155] CLIP-Cap↑ VisPhyWorld (GPT-5, p5js) 0.0299 [0.0243, 0.0356] CLIP-Cap↑ Veo-3.1 -0.0050 [-0.0098, -0.0002] CLIP-Cap↑ SVD (img2vid) 0.0101 [0.0050, 0.0151] BERTScore-F1↑ VisPhyWorld (GPT-5, p5js) 0.0077 [0.0059, 0.0094] BERTScore-F1↑ Veo-3.1 N/A [N/A, N/A] BERTScore-F1↑ SVD (img2vid) N/A [N/A, N/A] RAFT-EPE↓ VisPhyWorld (GPT-5, p5js) 0.6294 [-0.7743, 2.0695] RAFT-EPE↓ Veo-3.1 -0.7078 [-1.8371, 0.4628] RAFT-EPE↓ SVD (img2vid) 11.9706 [8.8544, 15.1865] Gemini↑ VisPhyWorld (GPT-5, p5js) -0.0108 [-0.5081, 0.5027] Gemini↑ Veo-3.1 0.9153 [0.4233, 1.3968] Gemini↑ SVD (img2vid) 2.0635 [1.7090, 2.4444]

- Table 8: Ablation on iterative self-repair (retry) on the VisPhyBench. “No retry” counts only samples that succeed on the first generation+render attempt; “1 retry” allows one additional attempt with renderer error feedback appended to the prompt.

Engine Success (no retry)↑ Success (1 retry)↑ Fixed by retry↑

Three.js 0.979 0.990 0.010 P5.js 0.853 0.979 0.126

- Table 9: Ablation on structured detection context. Lower is better for LPIPS and RAFT-EPE; higher is better for other metrics.

Model Mode LPIPS↓ CLIP-Img↑ DINO↑ CLIP-Cap↑ BERTScore↑ RAFT-EPE↓ Gemini↑

GPT-5 with detection 0.17 0.89 0.86 0.26 0.84 33.65 3.50 GPT-5 no detection 0.29 0.88 0.81 0.28 0.85 30.55 1.98 Gemini-3-Pro with detection 0.14 0.90 0.84 0.26 0.85 36.20 3.80 Gemini-3-Pro no detection 0.21 0.90 0.84 0.25 0.85 31.77 1.84 Claude Sonnet 4.5 with detection 0.16 0.90 0.83 0.26 0.85 36.20 2.39 Claude Sonnet 4.5 no detection 0.27 0.89 0.82 0.25 0.86 31.80 1.65

supports direct video input. Therefore, for GPT-5.4, we uniformly sampled JPEG frames from each video and provided those frames to the model for video-quality evaluation. Absolute score scales differ across judges, but the weak video baseline SVD-img2vid remains at the bottom, and sample-level correlations are significant across judge pairs. This reduces the risk that our conclusions are an artifact of a single proprietary evaluator.

- Table 10: Sample-level agreement among judge models. All reported correlations are significant with p < 0.001.

Judge pair Pearson r Spearman ρ

Gemini vs GPT-5.4 0.41 0.37 Gemini vs Qwen3-VL 0.40 0.43 GPT-5.4 vs Qwen3-VL 0.65 0.70

Table 11: Holistic physical-plausibility scores under different judge models.

Model Gemini score GPT-5.4 score Qwen3-VL score

Veo-3.1 2.56 5.09 7.82 Gemini-3-Pro 3.81 3.99 6.74 GPT-5 3.53 3.33 7.12 GPT-4.1 3.04 3.33 6.84 Claude-Sonnet-4.5 2.50 3.60 6.52 Qwen3-VL-Plus 2.13 2.25 4.94 SVD-img2vid 1.44 1.42 2.92

Forecast-only setting. Table 12 removes the later frame and conditions only on Istart. Visual metrics and RAFT-EPE change only modestly, but Gemini physical-plausibility scores drop for all tested models. This suggests that Ilater provides useful motion context; without it, models can still produce visually plausible clips but are less likely to recover the correct physical outcome.

- Table 12: Forecast-only ablation. The default setting uses frame1+frame10, while forecast-only uses frame1 only.

Model Mode LPIPS↓ CLIP-Img↑ DINO↑ CLIP-Cap↑ BERTScore↑ RAFT-EPE↓ Gemini↑

GPT-5 frame1+frame10 0.30 0.87 0.81 0.27 0.85 28.23 2.70 GPT-5 forecast-only 0.31 0.87 0.81 0.28 0.85 26.94 1.60 Gemini-3-Pro frame1+frame10 0.22 0.90 0.84 0.25 0.85 30.28 2.90 Gemini-3-Pro forecast-only 0.21 0.89 0.85 0.25 0.85 27.40 2.45 Claude-S4.5 frame1+frame10 0.27 0.89 0.81 0.26 0.86 28.46 1.95 Claude-S4.5 forecast-only 0.27 0.88 0.81 0.26 0.87 28.31 1.65

##### E.4 Cross-Engine Robustness

To separate physical reasoning from backend-specific coding familiarity, we run GPT-5 using five execution backends: Three.js, P5.js, SVG, Manim, and Blender. The backends differ in language, rendering API, and physics support, so large rank changes would indicate a strong backend confound. Absolute scores are summarized in Figure 4; here we provide the corresponding rank-correlation matrices. The scene-level ranking for the primary motion metric is stable: RAFT-EPE has an average pairwise Spearman correlation of 0.84 across engines, with all pairwise correlations significant at p < 0.05. Appearance and semantic rankings are less stable across engines (average Spearman correlations 0.47 for LPIPS and 0.60 for DINO), which is expected because rendering style varies more than motion difficulty. Engine choice affects absolute reconstruction quality, but it does not dominate the scene-level physical-difficulty ranking; the main motion-based conclusions remain robust across backends.

##### E.5 PSNR, SSIM, and Execution Success

- Table 15 reports frame-wise pixel fidelity and whether each configuration successfully produced valid videos. This table is separated from the main leaderboard because it focuses on low-level reconstruction and executability rather than the full multi-metric physical-understanding assessment.

- Table 13: Cross-engine Spearman correlation matrix on RAFT-EPE. Asterisks indicate p < 0.05; average pairwise ρ = 0.84.

RAFT-EPE Three.js P5.js SVG Manim Blender

Three.js 1.00 – – – – P5.js 0.70* 1.00 – – – SVG 0.84* 0.84* 1.00 – – Manim 0.78* 0.88* 0.92* 1.00 – Blender 0.75* 0.89* 0.88* 0.88* 1.00

- Table 14: Cross-engine Spearman correlations for visual metrics. Average pairwise correlations are moderate, unlike RAFT-EPE. Asterisks indicate p < 0.05.

Metric Engine Three.js P5.js SVG Manim Blender

LPIPS Three.js 1.00 – – – – LPIPS P5.js 0.43* 1.00 – – – LPIPS SVG 0.57* 0.50* 1.00 – – LPIPS Manim 0.34* 0.35* 0.41* 1.00 – LPIPS Blender 0.52* 0.57* 0.55* 0.46* 1.00

DINO Three.js 1.00 – – – – DINO P5.js 0.46* 1.00 – – – DINO SVG 0.70* 0.50* 1.00 – – DINO Manim 0.70* 0.60* 0.53* 1.00 – DINO Blender 0.67* 0.55* 0.60* 0.69* 1.00

##### E.6 Reconstruction & Perceptual Metrics

- Table 16 details the pixel-level and perceptual metrics. Gemini-3-Pro consistently achieves the best perceptual scores (LPIPS, FSIM), while Three.js backends generally outperform P5.js.

E.7 Visual Semantic Consistency

- Table 17 compares semantic understanding. GPT-5 and Gemini-3-Pro show strong alignment with the ground truth in terms of CLIP and DINO scores.

E.8 Text & Physical Consistency

- Table 18 and Table 19 (below) provide the remaining metrics on text analysis quality and physical motion fidelity.

- Table 15: Average PSNR/SSIM and generation success rate on VisPhyBench.

Model Engine PSNR ↑ SSIM ↑ Success ↑

GPT-5 Three.js 20.54 0.9370 0.990 GPT-5 p5.js 16.36 0.7440 0.979 GPT-4.1 Three.js 19.74 0.9337 0.948 GPT-4.1 p5.js 14.83 0.6830 1.000 Gemini-3-Pro Three.js 21.26 0.9445 0.957 Gemini-3-Pro p5.js 15.57 0.6943 0.963 Claude Sonnet 4.5 Three.js 20.75 0.9406 0.995 Claude Sonnet 4.5 p5.js 15.36 0.7160 1.000 Qwen3-VL-Plus Three.js 18.66 0.9306 0.936 Qwen3-VL-Plus p5.js 9.14 0.4296 1.000

SVD (img2vid) – 14.44 0.8802 1.000 Cosmos-Predict2.5-2B – 18.07 0.9303 1.000 Veo-3.1 – 20.04 0.9354 1.000

- Table 16: Detailed breakdown of Reconstruction and Perceptual Metrics.

Model PSNR↑ SSIM↑ LPIPS↓ FSIM↑ VSI↑ DISTS↓

VisPhyWorld (GPT-5, threejs) 20.54 0.9370 0.1736 0.9014 0.8432 0.1883 VisPhyWorld (GPT-5, p5js) 16.36 0.7440 0.2926 0.9105 0.8193 0.2724 VisPhyWorld (GPT-4.1, threejs) 19.74 0.9337 0.1818 0.9064 0.8309 0.2040 VisPhyWorld (GPT-4.1, p5js) 14.83 0.6830 0.3520 0.8977 0.8112 0.3348 VisPhyWorld (Gemini-3-Pro, threejs) 21.26 0.9445 0.1399 0.9225 0.8539 0.1859 VisPhyWorld (Gemini-3-Pro, p5js) 15.57 0.6943 0.3302 0.9055 0.8220 0.3384 VisPhyWorld (Claude Sonnet 4.5, threejs) 20.75 0.9406 0.1602 0.9118 0.8374 0.2001 VisPhyWorld (Claude Sonnet 4.5, p5js) 15.36 0.7160 0.3250 0.9030 0.8162 0.3109 VisPhyWorld (Qwen3-VL-Plus, threejs) 18.66 0.9306 0.2207 0.8972 0.8099 0.2373 VisPhyWorld (Qwen3-VL-Plus, p5js) 9.14 0.4296 0.5478 0.8797 0.7886 0.4396

SVD (img2vid) 14.44 0.8802 0.3408 0.8239 0.7585 0.3459 Cosmos-Predict2.5-2B 18.07 0.9303 0.2699 0.8198 0.8194 0.3447 Veo-3.1 20.04 0.9354 0.2102 0.8561 0.8586 0.1755

Table 17: Visual Semantic Consistency Metrics.

Model CLIP-Img↑ DINO↑ VisPhyWorld (GPT-5, threejs) 0.8930 0.8556 VisPhyWorld (GPT-5, p5js) 0.8134 0.7580 VisPhyWorld (GPT-4.1, threejs) 0.8933 0.8304 VisPhyWorld (GPT-4.1, p5js) 0.7545 0.6786 VisPhyWorld (Gemini-3-Pro, threejs) 0.8973 0.8405 VisPhyWorld (Gemini-3-Pro, p5js) 0.7460 0.6721 VisPhyWorld (Claude Sonnet 4.5, threejs) 0.8957 0.8305 VisPhyWorld (Claude Sonnet 4.5, p5js) 0.7612 0.7098 VisPhyWorld (Qwen3-VL-Plus, threejs) 0.8717 0.7837 VisPhyWorld (Qwen3-VL-Plus, p5js) 0.6446 0.5478 SVD (img2vid) 0.6677 0.6528 Cosmos-Predict2.5-2B 0.7293 0.6516 Veo-3.1 0.8564 0.8839

Table 18: Text–Video and Analysis-Text Consistency Metrics.

Model CLIP-Cap↑ ROUGE-L F1↑ BERTScore-F1↑

VisPhyWorld (GPT-5, threejs) 0.2632 0.2186 0.8436 VisPhyWorld (GPT-5, p5js) 0.2331 0.2057 0.8360 VisPhyWorld (GPT-4.1, threejs) 0.2610 0.2383 0.8522 VisPhyWorld (GPT-4.1, p5js) 0.2192 0.1689 0.8253 VisPhyWorld (Gemini-3-Pro, threejs) 0.2567 0.2141 0.8460 VisPhyWorld (Gemini-3-Pro, p5js) 0.2184 0.1886 0.8396 VisPhyWorld (Claude Sonnet 4.5, threejs) 0.2588 0.2168 0.8468 VisPhyWorld (Claude Sonnet 4.5, p5js) 0.2177 0.1599 0.8224 VisPhyWorld (Qwen3-VL-Plus, threejs) 0.2650 0.2022 0.8466 VisPhyWorld (Qwen3-VL-Plus, p5js) 0.2032 0.1733 0.8358

SVD (img2vid) 0.2533 – – Cosmos-Predict2.5-2B – – – Veo-3.1 0.2681 – –

Table 19: Motion and Physical Plausibility Metrics (Selected columns).

Model RAFT-EPE↓ RAFT-Angle↓ Align-Err↓

VisPhyWorld (GPT-5, threejs) 33.6473 68.5500 0.0210 VisPhyWorld (GPT-5, p5js) 34.3433 75.8555 0.0279 VisPhyWorld (GPT-4.1, threejs) 33.7110 67.7974 0.0249 VisPhyWorld (GPT-4.1, p5js) 37.6993 82.9492 0.0397 VisPhyWorld (Gemini-3-Pro, threejs) 36.2030 62.4494 0.0192 VisPhyWorld (Gemini-3-Pro, p5js) 33.1013 81.5723 0.0184 VisPhyWorld (Claude Sonnet 4.5, threejs) 36.1985 71.7979 0.0210 VisPhyWorld (Claude Sonnet 4.5, p5js) 34.1425 78.2841 0.0277 VisPhyWorld (Qwen3-VL-Plus, threejs) 35.0493 75.6650 0.0350 VisPhyWorld (Qwen3-VL-Plus, p5js) 20.8187 80.7413 0.8567

SVD (img2vid) 45.4606 84.7314 0.0746 Cosmos-Predict2.5-2B 33.1221 80.1338 0.0455 Veo-3.1 32.7145 77.0550 0.0193

