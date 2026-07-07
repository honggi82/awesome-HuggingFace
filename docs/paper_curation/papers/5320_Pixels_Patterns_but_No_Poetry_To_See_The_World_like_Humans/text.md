## Position: Reasoning After Perception Means Reasoning Without Vision

Hongcheng Gao*1 Zihao Huang*2 Jingyi Tang*3 Lin Xu1 Xinhao Li4 Haoyang Li5 Yue Liu6 Minhua Lin Xinlong Yang3 Taihang Hu7 Ge Wu7 Baolong Bi8 Hongyu Chen8 Olive Huang3 Wentao Zhang39

# arXiv:2507.16863v2[cs.CV]24Jun2026

### Abstract

A common belief in multimodal research is that the perceptual weaknesses of vision–language models can be compensated by stronger language reasoning (e.g., chain-of-thought, in-context learning, or external tools). We challenge this assumption. We argue that for a broad class of visual tasks hard to specify in language, failures stem from a structural fatality where the temporal decision of when to reason strictly dictates the spatial constraint of where reasoning takes place. When visual reasoning is deferred to language generation, current architectures do not merely delay computation; they displace it from the continuous visual representation to a discrete textual space. Consequently, the sequential “Perception-thenReasoning” paradigm degenerates perception into a passive, one-off feature encoding process, rendering it functionally equivalent to “Reasoningin-Text-Space”, where task-critical spatial signals are collapsed before reasoning begins. We substantiate this claim with the Turing Eye Test (TET): tasks that must be resolved in visual space and are hard to verbalize; results show text-only reasoning cannot remedy these perceptual failures. Our findings suggest rethinking the architectural divide: shifting from reasoning about perception to reasoning within perception. This facilitates actively reasoning-driven perception that operates directly on pixel-level visual representations, rather than within a collapsed textual space.

### 1. Introduction

Multimodal Large Language Models (MLLMs) exhibit striking linguistic fluency and often give the impression of com-

*Equal contribution 1Tsinghua University 2BIT 3Peking University 4Nanjing University 5BUPT 6National University of Singapore 7Nankai University 8UCAS 9Zhongguancun Academy. Correspondence to: Hongcheng Gao <gaohongcheng2000@gmail.com>, Wentao Zhang <wentao.zhang@pku.edu.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

prehensive visual understanding (Chen et al., 2024b; Bai et al., 2025a; Team, 2025e; Guo et al., 2025; Team, 2025c; An et al., 2025b). Yet they still fail on tasks that are conceptually simple but hard to specify precisely in text: deciding whether two segments intersect, whether two shapes are congruent, identifying which regions are connected, as well

- as spotting differences between similar images (Tong et al., 2024; Rudman et al., 2025; Chen et al., 2024a; Bai et al., 2025c). These questions are visually straightforward given complete images, but brittle under language-only specifications. This gap suggests a structural mismatch: current systems are excellent at talking about images, but unreliable
- at preserving and querying the fine-grained spatial evidence such decisions require (Wang et al., 2025e; Ma et al., 2026; Wu et al., 2024; Cao et al., 2025).

A common response attributes these failures to insufficient downstream reasoning, assuming that scaling languageside deliberation (e.g., CoT (Wei et al., 2022; Zhang et al., 2024b)) can resolve ambiguities. We challenge this assumption. For a broad class of geometry-, topology-, and counting-driven tasks, we argue that the limiting factor is not the reasoning depth, but the accessibility of the right visual evidence (Tong et al., 2024). This stems from a structural fatality where the temporal decision of when to reason strictly dictates the spatial constraint of where reasoning takes place. By deferring visual reasoning to language generation, current architectures displace computation from the continuous visual manifold to a discrete textual space, degenerating perception into a passive, one-off feature encoding process that collapses task-critical signals before reasoning begins.

### Position

The prevailing “Perception-then-Reasoning” paradigm creates a structural bottleneck: deferring reasoning to the decoder constrains it to a discrete, text-aligned space where task-critical signals have already collapsed. We advocate for reasoning within perception—operating on pixel-level representations rather than collapsed textual proxies.

Our central claim is that the temporal decision of when to reason dictates where reasoning can occur. In most MLLMs, decisive computation is deferred until language de-

[Figure 1]

[Figure 2]

[Figure 3]

###### Perception-then-Reasoning

###### Reasoning within Perception

munication but also the preservation of pixel-level fidelity, allowing high-level semantics to query and steer the visual encoding process to resolve ambiguities and lost information that are invisible in the discrete text space.

Visual Input Query

Visual Input Query

[Figure 4]

Please identify what word appears in this image.

[Figure 5]

Please identify what word appears in this image.

[Figure 6]

[Figure 7]

active exploration

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Vision tower

In summary, we argue that persistent visual failures stem from representational access—task-critical spatial information is discarded before reasoning begins—not insufficient reasoning capacity. We develop this claim through a structural critique of the mismatch between visual tasks and text-space reasoning (Sec. 3), a theoretical analysis formalizing information collapse in vision-to-language projection (Sec. 4), empirical validation via TET (Sec. 5), and a proposed framework for reasoning within perception (Sec. 6).

[Figure 12]

text

discrete tokens

[Figure 13]

Vision tower

+

text

+

Language Model

[Figure 14]

（reasoning in text sapce）

Language Model

[Figure 15]

I cannot visually perceive the word.

[Figure 16]

(reasoned during perception)

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

love

No Word

[Figure 21]

- Figure 1. (Left) The mainstream Perception-then-Reasoning paradigm: visual evidence is passively encoded into semantic signals, leading to Information Collapse. (Right) The proposed Reasoning within Perception paradigm: reasoning is executed actively on the original image before semantic compression.

### 2. Background: Architectural Paradigms

Before presenting our critique, we establish a taxonomy of existing approaches based on when and where visual reasoning occurs. Table 1 summarizes key architectural paradigms and their implicit assumptions.

coding, after the visual stream has been compressed into a text-facing interface (e.g., a limited set of visual tokens, pooled embeddings, or cross-attended summaries) (Bai et al., 2025a; An et al., 2025b). Consequently, the sequential “Perception-then-Reasoning” paradigm degenerates perception into a passive, one-off feature encoding process, rendering it functionally equivalent to “Reasoning-in-TextSpace.” This design is particularly detrimental for tasks that are hard to specify in text yet easy to decide from pixels: the distinctions required for correctness are not reliably available in the text-aligned representation at reasoning time. Thus, increasing textual deliberation can improve coherence and self-consistency while still failing to recover signals that were not preserved through the bottleneck (Tang et al., 2025; Qu et al., 2025).

Perception-then-Reasoning. The dominant paradigm treats vision as preprocessing, compressing images into discrete tokens consumed by LLMs (Liu et al., 2023; Dai et al., 2023; Alayrac et al., 2022). Reasoning occurs entirely post-compression in text space. Variations include static projection (direct mapping from visual features to text embeddings) (Liu et al., 2024), enhanced inference (CoT, ICL, or sampling strategies applied during text generation) (Wang et al., 2025c; Chen et al., 2025c; Aissi et al., 2025; Huang et al., 2025), and tool augmentation (external modules invoked during generation) (An et al., 2025a; Sur´ıs et al., 2023; Wang et al., 2025a).

We provide diagnostic evidence using the Turing Eye Test (TET), a probe designed to reduce the usefulness of linguistic shortcuts and isolate perceptual failures from high-level semantics. Across contemporary MLLMs, we observe that common inference-time interventions—including Chainof-Thought and in-context learning—often yield limited improvements on hard-to-verbalize visual cases, aligning with the hypothesis that scaling textual deliberation cannot recover information lost during passive encoding.

Reasoning Within Perception. Recent works explore performing computation before or during visual encoding, including visual sketching approaches that enable iterative refinement in latent visual space (Zhang et al., 2025a; Chung et al., 2025), and unified tokenization that treats vision and language symmetrically (Xie et al., 2025; Chen et al., 2025b). Our position advocates for this direction but identifies persistent limitations even in these approaches, particularly semantic bias inherited from text-aligned pretraining.

Consequently, we argue for a shift from reasoning about perception to reasoning within perception. This position implies two architectural realignments:

Table 2 situates TET relative to existing multimodal benchmarks. Many general MLLM benchmarks emphasize semantic correctness or coarse QA accuracy, so strong performance can still be achieved through language priors or distributional shortcuts. TET instead suppresses linguistic shortcuts and probes whether fine-grained visual evidence remains accessible after vision-language projection.

- 1. When and Where to Think: Visual Reasoning During Perception. Reasoning that depends on geometry and spatial relations should be executed before visual evidence is funneled into a text-aligned bottleneck, so that deliberation occurs in a visual space that supports perceptual operations.
- 2. What to Preserve: Pixel–Semantic Dual Requirements. Visual representations should facilitate active perception. This requires not only semantic alignment for com-

### 3. Structural Critique: Description ̸= Seeing

Current MLLM paradigms have achieved remarkable success in high-level semantic understanding tasks (Yue et al.,

- Table 1. Taxonomy of Visual Reasoning Paradigms. We categorize approaches by when reasoning occurs, where computation takes place, and the underlying perception mode. Our proposal advocates for iterative refinement within the naive visual space.

Paradigm When Where Perception Mode Representative Work

Current Mainstream: Perception-then-Reasoning

CoT Enhancement Post-LLM Decode Text Space Static Semantic Perception

Gemini (Team, 2025b)

QVQ (Team, 2024)

Kimi k1.5 (Team, 2025d)

Qwen3-VL (Bai et al., 2025a)

Tool Augmentation Post-LLM Decode Text Space Multi-turn Static Semantic Perception

ViperGPT (Sur´ıs et al., 2023)

Mini-o3 (Lai et al., 2025)

DeepEyes (Zheng et al., 2025)

VTool-R1 (Wu et al., 2025c)

OpenThinkIMG (Su et al., 2025) Emerging: Toward Reasoning Within Perception

Visual Sketching Post-VT Latent Space Static Semantic Perception

LatentSketchpad (Zhang et al., 2025a)

MVoT (Li et al., 2025a)

Unified Model Post-LLM Decode Visual Space Hybrid Semantic-Texture Perception

Show-o2 (Xie et al., 2025)

Janus-Pro (Chen et al., 2025b)

Bagel (Deng et al., 2025) Our Proposal Pre-VT Naive Visual Space Active Reasoning within Perception Active Visual Querying

- Table 2. Landscape of Visual Reasoning Benchmarks. Type I & II benchmarks possess high verbalizability, allowing models to reduce visual tasks to text-based deduction. Type III benchmarks expose perceptual bottlenecks where linguistic shortcuts fail. The last column highlights relevance to our central thesis.

Benchmark Core Task Focus Verbalizability Reasoning Substrate Primary Bottleneck Relevance to Thesis

Type I: Semantic & Knowledge-Intensive MMMU (Yue et al., 2024) College-level exams High World Knowledge Knowledge retrieval

Success stems from text reasoning, not visual understanding.

MathVista (Lu et al., 2024) Math problem solving Medium Text/Formulas Symbolic manipulation ScienceQA (Lu et al., 2022) Multimodal science Q&A High World Knowledge Domain knowledge OmniDocBench (Ouyang et al., 2025) Document parsing High Text & Layout Structure recognition

- Type II: Abstract & Symbolic Logic

LogicVista (Xiao et al., 2024) Logical reasoning Medium Discrete Symbols Rule induction Tests discrete rule-following, ARC-AGI-2 (Chollet et al., 2026) Grid pattern abstraction High Discrete Grid State Program synthesis bypassing visual bottleneck.

- Type III: Low-Level Perception

MMVP (Tong et al., 2024) Visual fundamentals Low Visual Features Vision encoder

Exposes perceptual bottleneck, but reasoning ability remains untested.

Blink (Fu et al., 2024) Core perception Low Visual Features Rapid perception BabyVision (Chen et al., 2026) Pre-linguistic vision Low Visual Primitives Visual primitives HiddenInPlainSight (Fu et al., 2025) Visual grounding Low Visual Features LLM integration

TET (Ours) Fine-grained reasoning Low Continuous Manifold Visual reasoning Proves text reasoning failed.

2024; Lu et al., 2024), yet they suffer from a fundamental architectural misalignment. We argue that the community has been optimizing for the wrong objective: we are teaching models to “describe” the world, not to “see” it.

- 3.1. The “Where to Think” Fallacy

must occur in the native image space (or a continuous visual manifold), not in the text space. The question of “where to think” must be answered by shifting computation back to the visual domain.

#### 3.2. The “What to See” Deficit

The dominant architecture treats vision as a preprocessing step, compressing images into discrete tokens to be processed by an LLM. This forces all reasoning to occur in the textual latent space. However, visual reasoning is inherently different from linguistic reasoning. Tasks such as mental rotation, trajectory tracking, and spatial geometry rely on continuous transformations, not discrete symbol manipulation. Forcing an LLM to simulate continuous geometric changes using discrete text tokens is computationally inefficient and fundamentally flawed. Table 3 presents representative examples illustrating this fundamental mismatch between task requirements and architectural support.

Current visual encoders (e.g., CLIP) are trained to align with text (Radford et al., 2021), which biases them towards semantic features while discarding task-critical spatial signals (Zhang et al., 2025b). They can identify “a cat on a table” but struggle to determine whether two line segments intersect or count disconnected regions—tasks that are visually straightforward yet hard to specify in text.

This creates a representational disconnect: the model knows what objects are present but cannot resolve how they relate spatially. We hypothesize that the projection from visual to textual space acts as a filter governed by verbalizability: information that can be easily described in language is preserved, while information that resists linguistic specification is systematically collapsed. Table 4 categorizes information

The bottleneck is not the LLM’s reasoning capability, but the representation itself. We argue that visual reasoning

- Table 3. Task Requirements vs. Architectural Support. Examples illustrating disparity between the continuous nature of visual tasks and the discrete representations in current architectures.

Task Category Required Operation Current Support

Mental Rotation (Tong et al., 2024; Huang et al., 2026) Continuous transform Discrete tokens Occlusion Reasoning (Wang et al., 2025b; Liu et al., 2025b) Depth ordering Flattened features Spatial Counting (Daxberger et al., 2025; Wu et al., 2025a) Pixel-level grouping Semantic categories Trajectory Prediction (Liu et al., 2025a; Kim et al., 2025) Temporal dynamics Static snapshots Geometric Measurement (Li et al., 2024; Xu et al., 2025b) Metric distances Categorical labels Fine-grainedw Recognition (He et al., 2025; Kuchibhotla et al., 2025) Stroke-level detail Patch-level semantics

types along this axis; our experimental results in Sec. 5 provide empirical validation.

3.3. The Sequential Trap vs. Perception as Reasoning

The prevailing pipeline of MLLMs follows a rigid sequence: Perception → Alignment → Reasoning. This assumes that perception is a one-off extraction process that finishes before reasoning begins. We contend that this separation is artificial. In biological systems, perception and reasoning are intertwined; we reason while we perceive (Marr, 1982). Crucially, the standard practice of “freezing” the perception

- Table 4. Hypothesized Semantic Bias of Vision-Language Projection. Verbalizability determines information survival.

model can direct its “hands” (external tools) to execute complex visual operations, the critical reasoning steps are still strictly performed within the “brain” (the LLM) in a purely textual manner. The reasoning engine merely consumes the discrete outputs of these tools without direct access to the continuous visual process. This is akin to handing a blind poet a more capable pen; he remains without eyes.

### 4. Theoretical Analysis

We now formalize the limitations of current MLLM paradigms and derive a theoretical characterization of the error introduced by text-space reasoning.

#### 4.1. Problem Formalization and Notation

Information Type Easy to Verbalize? Prediction

Semantic categories Yes Preserved Coarse spatial layout Yes Preserved

Let X ⊆ RH×W×3 denote the continuous image manifold, and let y ∈ Y represent the task label. The architecture comprises three primary functional components:

Pixel-level fidelity No Collapsed Geometric relations No Collapsed Topological structure No Collapsed Stroke-level details No Collapsed

- 1. Vision Encoder: fθ : X → Z ⊆ Rd, which maps the image manifold to a continuous visual feature space Z.
- 2. Modality Projection: gϕ : Z → E ⊆ RD, a mapping from visual features to the text-aligned latent space E of the Large Language Model. Note that d and D may differ; this projection is optimized for semantic alignment rather than geometric fidelity.
- 3. Task Head: hψ : E → Yˆ, which generates the final prediction. The training objective is to minimize the empirical risk:

Rˆ(θ,ϕ,ψ) =

1 N

N

i=1

ℓ(hψ ◦ gϕ ◦ fθ(xi),yi). (1)

In the following analysis, we focus on per-sample error to isolate the structural bottleneck.

- 4.2. The Information Collapse Bound

module reduces vision to a passive, one-off encoding step. This creates a structural barrier: the reasoning engine is unable to leverage linguistic inputs to actively re-examine the visual data for task-relevant details. Since this static representation inevitably captures less information than the raw visual input, critical fine-grained signals are often discarded before reasoning even begins. To move forward, we must abandon the “Perception then Reasoning” paradigm in favor of “Perception as Reasoning” where the model actively reasons within the visual signal, iteratively refining its understanding at both the semantic and pixel levels.

#### 3.4. External Tool Invocation Is Not Native Seeing

To bridge the gap between human thought processes and pure text-based reasoning, the community has attempted to integrate visual tools into the reasoning pipeline. Exemplified by models like o3 (OpenAI, 2025), tool-augmented methods (Lai et al., 2025; Zheng et al., 2025) enable models to engage in multi-round visual evidence by iteratively invoking external utilities—ranging from code interpreters to plotting libraries—to process or reconstruct visual data.

To characterize the representational bottleneck, we introduce two key concepts:

Task-Relevant Representation. For a task with label y, let z∗ ∈ Z denote the ideal visual representation—one that preserves exactly the geometric and spatial information required to predict y.

However, these methods fundamentally remain limited by a loose coupling between cognition and perception. While the

Reconstruction Mapping. To rigorously measure information loss across heterogeneous spaces (Z and E with potentially different dimensions), let g† : E → Z denote the optimal reconstruction mapping that attempts to recover visual features from text-aligned embeddings. The effective visual information available to the reasoning module is then z˜ = g†(gϕ(fθ(x))). Applying the triangle inequality in the visual feature space Z:

##### +∥fθ(x) − g†(gϕ(fθ(x)))∥Z

∥z∗−z˜∥Z ≤ ∥z∗ − fθ(x)∥Z

εenc

εproj

(2)

where ∥ · ∥Z denotes a task-appropriate norm on the visual feature space. Here: (1) εenc quantifies the encoding gap: the discrepancy between the vision encoder’s output and the task-relevant representation. (2) εproj quantifies the projection gap (reconstruction error): the information irreversibly lost when visual features are projected into the text-aligned space. This measures the non-invertibility of gϕ.

Under the assumption that the task function is Lipschitz continuous with respect to the visual representation (i.e., small changes in visual state should yield proportionally small changes in the output), the prediction error is bounded by the representation error:

∥yˆ − y∥ ≤ L · ∥z∗ − z˜∥Z ≤ L · (εenc + εproj) (3)

for some task-dependent constant L > 0. While Eq. 3 establishes an upper bound, the existence of a non-zero lower bound is guaranteed by the failure of injectivity described below: if distinct visual states collapse to the same z˜, the model cannot simultaneously predict distinct labels correctly.

The Projection Gap is Non-Vanishing. A critical observation is that εproj > 0 for tasks requiring fine-grained spatial reasoning. The projection gϕ is semantically contractive: it maps spatially distinct but semantically similar visual states to nearby embeddings. Formally, for two images x1,x2 with identical semantic content but different geometric configurations, the distance in E is insensitive to geometric variations compared to distance in Z:

∥gϕ(fθ(x1))−gϕ(fθ(x2))∥E ≪ ∥fθ(x1)−fθ(x2)∥Z (4)

This contraction implies that perfect reconstruction is impossible: the mapping gϕ ◦ g† cannot be the identity on Z for the subspace encoding geometric details.

Proposition 4.1 (Information Collapse). Let z∗ encode geometric or topological properties, and let gϕ be semantically contractive (collapsing geometrically distinct but semantically equivalent inputs). Then: (1) the projection gap εproj = ∥fθ(x) − g†(gϕ(fθ(x)))∥Z > 0 is strictly positive, and (2) the resulting prediction error admits a non-zero lower bound irreducible by text-space reasoning hψ alone.

Interpretation. Proposition 4.1 formalizes the “Blind Poet” paradox: when reasoning is confined to the text-aligned space E, the model can only access information that survives the lossy projection gϕ. Scaling text-side computation cannot recover geometric details that were collapsed during projection. No amount of eloquence can compensate for blindness.

### 5. Diagnostic Evidence: The Turing Eye Test

To empirically validate our theoretical claims, we introduce the Turing Eye Test (TET)—a diagnostic benchmark designed to isolate perceptual failures from high-level semantics. Unlike benchmarks that conflate visual perception with world knowledge or linguistic reasoning, TET specifically targets tasks that are easy to verify from pixels but hard to specify in text.

#### 5.1. Task Design Principles

TET comprises four task families targeting distinct aspects of fine-grained visual perception: HiddenText (150 images)—scale-variant images where text resolves into detailed scenes when magnified; 3DCaptcha (150 images)—characters with 3D perspective distortion and curved surfaces; ColorBlind (150 images)—Ishihara-style tests (Ishihara, 1951) with chromatically confounding elements; and ChineseLigatures (40 images)—fused glyphs synthesized from multiple Chinese characters. Representative examples are shown in Fig. 2 and generation details are provided in Appendix A.

#### 5.2. Main Results: Collective Blindness

We provide the specific details of our evaluation in Appendix.A.2 and report our evaluation results in Table 5. Despite impressive capabilities on standard multimodal benchmarks (Yue et al., 2025; Lu et al., 2024), all models exhibit near-zero performance on TET tasks. This may be because the information passively received by the model is irrelevant to the query. We present attention-map analyses of the Qwen2.5-VL series on HiddenText in Fig. 4, with additional analyses in Appendix C and post-fine-tuning analyses in Appendix D. This failure pattern is consistent with our theoretical analysis: continuous spatial-visual information is forcibly collapsed into discrete textual space, and this dimensional collapse results in catastrophic failure on tasks requiring fine-grained spatial reasoning. Representative response examples are provided in Fig. 15.

#### 5.3. Can Enhanced Inference Overcome the Bottleneck?

We investigate whether providing domain-specific exemplars enables models to learn relevant perceptual patterns. Table 6 shows results with 3-shot in-context learning. The

- Table 5. TET Main Results. Pass@1 and pass@32 accuracy (%) across 15 state-of-the-art MLLMs. Near-zero performance across all models validates our claim that current architectures fundamentally lack pixel-level reasoning capability.

Category Model

HiddenText 3DCaptcha ColorBlind ChineseLigatures Average Pass@1 Pass@32 Pass@1 Pass@32 Pass@1 Pass@32 Pass@1 Pass@32 Pass@1 Pass@32

Unified Multimodal

Show-o2 (Xie et al., 2025) 0 0 0 0 0 0 0 0 0 0 Bagel (Deng et al., 2025) 0 0 0 0 0 0 0 0 0 0 Janus-Pro (Chen et al., 2025b) 0 0 0 0 0 0 0 0 0 0

Closed-Source APIs

Claude 4-Sonnet (Anthropic, 2025) 0 0 0 0 0 0 0 0 0 0

Gemini 2.5 Pro (Team, 2025b) 0 0 0 0 0 0 2.5 5 0.2 0.4 OpenAI o1 (Jaech et al., 2024) 0 0 0 0 0 1.33 0 0 0 0.4 Seed-1.6 (Team, 2025a) 0 0 0 0 0 0 2.5 2.5 0.2 0.2

Open-Source Models

Qwen2.5VL-72B (Bai et al., 2025b) 0 0 0 0 0 0 0 0 0 0

Qwen2.5VL-7B (Bai et al., 2025b) 0 0.67 0 0 0 0 0 2.5 0 0.4 Qwen2.5-Omni-7B (Xu et al., 2025a) 0 0 0 0 0 0 0 2.5 0 0.2

QVQ-72B (Team, 2024) 0 0 0 0 0 0 0 0 0 0 InternVL3-78B (Zhu et al., 2025) 0 0 0 0 0 0 0 0 0 0

MiniCPM-o-2.6 (Team, 2025f) 0 0 0 0 0 0 0 2.5 0 0.2 kimi-vl-a3b (Team, 2025e) 0 0 0 0 0 0 0 5 0 0.4 kimi-vl-a3b-thinking (Team, 2025e) 0 0 0 0 0 0 0 0 0 0

introduction of in-domain exemplars yields negligible improvement. This demonstrates that the key to resolving perceptual limitations does not lie in providing additional knowledge through contextual demonstrations; rather, it points to fundamental architectural deficiencies that cannot be addressed through in-context interventions.

- Table 6. In-Context Learning Analysis. Pass@1 / Pass@32 accuracy (%) with 3-shot ICL. Minimal gains confirm that the bottleneck is representational access, not knowledge acquisition.

fidelity in visual representation:

Accuracy ↑ ⇐⇒ E |y − fθ(x)|2 ↓ (6)

Generalization Caveat: Architectural Limitations Beyond Data. Although fine-tuning the vision encoder greatly improves TET performance, the gains fail to generalize to held-out tasks with different visual styles. This fragility suggests that fine-tuned encoders memorize task-specific patterns rather than learning transferable geometric primitives. Encoder fine-tuning thus locates the bottleneck but does not resolve it—a real solution requires architectures that embed reasoning into perception, maintaining continuous spatial structure rather than collapsing it before reasoning begins.

Model Setting HiddenText 3DCaptcha ColorBlind ChineseLig. Qwen2.5VL-72B

Base 0 / 0 0 / 0 0 / 0 0 / 0 +3-shot 0 / 2.0 0 / 0 0 / 1.3 0 / 0

Base 0 / 0 0 / 0 0 / 0 2.5 / 5.0

Gemini 2.5 Pro

+3-shot 0 / 0 0 / 0 0 / 4.0 7.5 / 20.0 Seed-1.6

5.5. Whether the Model’s Upper Bound Can Overcome the Bottleneck? Reinforcement Learning Analysis

Base 0 / 0 0 / 0 0 / 0 2.5 / 2.5

+3-shot 0 / 0 0 / 0 0.7 / 0.7 0 / 5.0

#### 5.4. Where Is the Bottleneck? Fine-Tuning Analysis

To isolate the bottleneck location, we conduct SFT on Qwen2.5-7B-VL with five configurations targeting different architectural components. Table 7 reports results.

Key Finding. Only configurations updating the vision encoder fθ yield substantial gains; updating only the language backbone or adapter produces negligible improvement. This validates Theorem 4.1: error in text space is proportional to information loss during vision encoding, which cannot be eliminated by enhanced text-space reasoning. Further details are provided in Appendix E.

Theoretical Interpretation. When updating only θ (freezing gϕ,ψ), the error change approximates:

- 1

- 2

∆εtotal ≈ −

∆∥y − fθ(x)∥2Ω−1 (5)

The encoder fθ re-parameterizes X to minimize distance to ground truth y, reducing the first term of Eq. 6 while the quantization penalty (second term) remains constant. Improvement correlates directly with enhanced structural

Table 7. Fine-Tuning Component Analysis. Pass@1 / Pass@32 accuracy (%). Highlighted rows : only updating vision part.

Parameters Updated HiddenText 3DCaptcha ColorBlind No Training (Baseline) 0 / 0 0 / 0 0 / 1.3

Full Parameters 90.0 / 94.7 95.3 / 98.0 77.3 / 79.3 Vision Encoder Only 86.7 / 94.7 94.0 / 98.0 87.3 / 98.7 Vision Encoder + Adapter 82.0 / 94.7 95.3 / 97.3 99.3 / 99.3

Language Backbone Only 0 / 2.7 0 / 0 0.7 / 14.0 Adapter Only 0 / 5.3 0 / 0 1.3 / 6.7

We use reinforcement learning (RL) as a diagnostic tool to test whether Multimodal Large Language Models (MLLMs) can surpass the previously identified paradigm bottleneck. Experiments are conducted on the OCRVQA dataset (Mishra et al., 2019), using cold-start and GRPO training, where the cold-start data is distilled from Claude-4-Sonnet. We conduct architectural ablations over different optimization scopes—full model, vision tower, and LLM—with and without cold-start alignment, to identify where perceptual reasoning is effectively learned.

The paradigm bottleneck persists at the performance upper bound. Consistent with our theoretical analysis (Eq. 6) and prior experimental results (Table. 7), models

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Query: You are given an image. Please identify what word appears in this image. Your final answer should be enclosed in \box{}.

Hidden

Text

Star Dream Car Brick Wood

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Captcha

Query: You are given an image. Please identify the characters in this image. Your final answer should be enclosed in \box{}.

Ligatures 3D

2XDFF GHDTH 2PUGV DUFDN LVSYZ

Query: You are given an image. Please identify the Chinese phrase, sentence, idiom, proverb, or solar term displayed in this image. Your final answer should be enclosed in a \box{}.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Chinese

琴瑟和鸣 无中生有 喜结良缘 心想事成 十全十美

ColorBlind

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Query: You are given an image. Please identify the characters in this image. Your final answer should be enclosed in \box{}.

6G P4 EU AC 7F

- Figure 2. Evaluation cases for each TET category: HiddenText, 3DCaptcha, ColorBlind, and ChineseLigatures. The text beneath each image represents the corresponding ground truth. The third line of Chinese characters, read from left to right, symbolizes marital bliss, a serendipitous union, a perfect match, dreams fulfilled, and flawless.

0 50 100 150 200 250 300 350

Step

0.4

0.5

0.6

0.7

0.8

Accuracy

| |
|---|

| |
|---|

| |
|---|

| |
|---|
| |

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

SFT on Full

SFT on LLM

SFT on None

SFT on VIT

(a) RL on Full Parameters

0 50 100 150 200 250 300 350

Step

0.4

0.5

0.6

0.7

0.8

Accuracy

| |
|---|
| |

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

| |
|---|

| |
|---|

| |
|---|

SFT on Full

SFT on LLM

SFT on None

SFT on VIT

(b) RL on LM Backbone

0 200 400 600 800 1000 1200

Step

0.4

0.5

0.6

0.7

0.8

Accuracy

| | |
|---|---|
| | |

SFT on Full

SFT on LLM

SFT on None

SFT on VIT

(c) RL on ViT

- Figure 3. Architecture Ablation on Perception Space. We conduct architecture ablation experiments during both the cold start and RL stages, where Full denotes training the entire architecture, LLM/ViT indicates training only the corresponding module, and None represents that no cold-start training was performed.

with reinforcement learning applied solely to the Vision Tower achieve performance on par with full-parameter optimization, while additional optimization of the language module yields extremely limited gains. This indicates that the performance-limiting factor of the model stems from paradigm design rather than the algorithmic level: optimization on the language side cannot recover the perceptual information lost during the stage of passive visual encoding.

Upper-bound convergence reveals alignment constraints. Without cold-start alignment, reinforcement learning rapidly converges to the model’s perceptual upper bound; cold-start alignment significantly slows this convergence. This suggests that early alignment biases perceptual tokens toward discrete text space, causing spatial collapse that subsequent reasoning cannot recover.

#### 5.6. Resolution Sensitivity: Exposing Semantic Bias

We probe whether failures stem from impaired structural perception or semantic over-reliance by evaluating HiddenText under resolution perturbations (Fig. 5): (1) direct downsampling, which removes fine-grained detail while preserving coarse semantics, and (2) downsampling then upsampling, which preserves global layout but disrupts local geometry.

Under direct downsampling, accuracy rises as resolution drops—revealing reliance on coarse semantics over fine structure. Yet structural blurring hurts performance despite intact global layout, exposing dependence on superficial texture cues rather than true geometry. Both patterns confirm semantic bias: spatial details critical to the task are collapsed in text-aligned representations and irrecoverable by later reasoning.

Qwen2.5-VL-7B Answer: No Text GT: Love

|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>ViT<br><br>LLM<br><br>[Figure 57]|
|---|

Qwen2.5-VL-72B Answer: No Text GT: Love

|[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]<br><br>LLM<br><br>ViT|
|---|

Figure 4. Grad-CAM of Qwen2.5-VL Series Models on HiddenText .

### 6. Toward Reasoning Within Perception

Our diagnosis points to a clear prescription: visual reasoning must occur in visual space, not text space. Rather than advocating for a single architecture, we outline the design principles that any solution must satisfy to escape the information collapse bound.

#### 6.1. Design Principles for Native Visual Reasoning

We identify 3 core principles distinguish reasoning within perception from prevailing reasoning about perception:

- Principle 1: Reason Before Collapse. Reasoning must occur before visual information is projected into text-aligned embeddings. Once compressed into text space, pixellevel details and geometric relationships are irreversibly lost—reasoning on post-compression representations is already too late.
- Principle 2: Persistent Visual Access. The original visual representation must remain accessible throughout the reasoning process, allowing repeated queries to the visual substrate rather than a one-shot encoding that is subsequently discarded.
- Principle 3: Reasoning-Driven Perception. High-level semantic hypotheses should actively guide which visual regions or relationships to examine. Perception should be steered by reasoning demands, not merely consumed as a fixed, pre-computed input.

#### 6.2. A Formal Framework: Active Visual Querying

To formalize how these principles escape the information collapse bound, we propose Active Visual Querying (AVQ)—a mechanism where reasoning can issue queries back to the visual representation, and where perception itself is steered by reasoning demands.

Qwen2.5-VL-72B-Instruct

Qwen2.5-VL-72B-Instruct

70

Bagel

Bagel

20

Claude-4-sonnet

Claude-4-sonnet

60

Average

Average

50

15

ACC(%)

ACC(%)

40

10

30

20

5

10

0

0

0 5 10 15 20

0 5 10 15 20

Downsample Factor

Blur Factor

(a) Downsampling

(b) Blurring

Figure 5. Resolution Sensitivity. (a) Downsampling paradoxically improves accuracy, suggesting reliance on coarse semantics over fine structure. (b) Blurring degrades performance despite preserved global layout, confirming dependence on local texture cues.

Mechanism Formulation. Let z = fθ(x) denote the visual representation and st the reasoning state at step t. AVQ operates as:

at = A(z,Q(st);x), st+1 = U(st,at) (7)

Here Q generates a visual query from the current reasoning state at each step, and A retrieves task-relevant information from z (e.g., re-attending to specific regions, enhancing taskrelevant features). U then updates the reasoning state. The final output is yˆ = hψ(sT).

The key requirements are: (1) A operates in visual space Z rather than text space E, and (2) retrieval is conditioned on the query Q(st), making perception an active, reasoningdriven process rather than fixed encoding. This implements all three principles: reasoning occurs before collapse (Principle 1), visual access persists throughout (Principle 2), and perception is steered by reasoning demands (Principle 3).

Architectural Gap. Current MLLMs partially approximate AVQ: cross-attention to image tokens provides a form of retrieval A, but both query generation Q and perception refinement P are implicit and fixed per layer—not explicitly driven by the reasoning process. Tool-augmented approaches implement explicit Q but through discrete API calls that sacrifice differentiability. Crucially, no current architecture allows reasoning to steer perception itself—the visual encoding z remains static regardless of downstream demands. Closing this gap is a concrete research direction.

#### 6.3. Architectural Instantiations

The AVQ framework admits multiple architectural realizations—cross-attention to pixels, recurrent visual transformers (Zhang et al., 2025a), and generative refinement (Chung et al., 2025)—each implementing partial aspects of the query-retrieval-update cycle. However, no existing architecture fully instantiates AVQ with explicit query generation, reasoning-driven retrieval, and grounded state update. We detail these instantiations in Appendix H.

#### 6.4. What This Position Does Not Claim

We emphasize that our position does not require abandoning language as an output modality or reasoning medium. Many visual tasks are well-served by text-space reasoning: captioning, VQA on semantic content, or problems where linguistic structure provides useful inductive bias. For such tasks, the standard pipeline—encoding once, then reasoning in text space—remains appropriate.

Our claim is narrower: for tasks requiring fine-grained spatial grounding—geometry, topology, precise counting, occlusion reasoning—the architecture must support reasoning in visual space when needed. The goal is not to replace vision-language integration, but to provide flexibility when task-critical spatial information would otherwise be lost.

### 7. Alternative Views

We acknowledge and address several perspectives that challenge our position:

“Scale Will Solve It.” Our benchmark results (Table 5) provide direct evidence against this view: performance remains near-zero across model scales from Qwen2.5VL-7B to 72B. Even with RLVR-scaled reasoning, SOTA models fail on fundamental visual primitives (Chen et al., 2026; Wang et al.,

2026b; Chen et al., 2025a; Li et al., 2025c)—the bottleneck is information-theoretic, not computational (Liu et al., 2026; Weng et al., 2025).

“Better Vision Encoders Suffice.” Our evidence partially supports this—fine-tuning the encoder helps substantially. However, these gains often fail to generalize beyond the training distribution (Zhai et al., 2023; Wu et al., 2025b; Wang et al., 2025d), addressing only “what to see,” not “when to reason.” A better encoder still faces the sequential bottleneck if reasoning remains deferred to text space (Wang et al., 2026a).

“Tool Use Is the Answer.” Tool-augmented approaches (Sur´ıs et al., 2023; Lai et al., 2025) extend capabilities, but perception is accessed only via discrete API queries and structured outputs, not a continuously manipulable visual state (Ke et al., 2025; Li et al., 2025b). This is augmentation, not native visual reasoning.

More alternative views can be found in Appendix F.

### 8. Implications and Future Directions

Our position suggests several directions for future research, including architectural designs that prioritize reasoning within visual representations, training objectives that preserve spatial fidelity beyond semantic alignment, benchmarks that isolate perceptual from reasoning failures, and adaptive routing between visual and text pathways based on task demands. These directions share a common principle: enabling computation to return to visual space when task demands require it. We elaborate on these in Appendix G.

### 9. Conclusion

In this paper, we have argued that persistent visual failures in MLLMs stem from a structural misalignment: deferring reasoning to language generation displaces computation from the visual manifold to textual space, collapsing task-critical spatial signals before deliberation begins. Our theoretical analysis bounds this information loss, and the Turing Eye Test confirms that vision encoder modifications produce dramatic improvements while inference-time interventions yield minimal gains—implicating the perceptual bottleneck rather than reasoning capacity. These findings motivate a paradigm shift from reasoning about perception to reasoning within perception: architectures must enable deliberation on pixel-level representations before semantic compression, not after. Only then can we bridge the gap between fluent visual description and faithful visual understanding.

### Acknowledgements

This work is supported by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM113), Na-

tional Natural Science Foundation of China (92470121, 62402016), National Key R&D Program of China (2024YFA1014003), Zhongguancun Academy (C20250204, C20250602), and Beijing Major Science and Technology Project (Z251100008125043, Z251100008425023).

### References

Aissi, M. S., Grislain, C., Chetouani, M., Sigaud, O., Soulier, L., and Thome, N. Viper: Visual perception and explainable reasoning for sequential decision-making, 2025. URL https://arxiv.org/abs/2503.15108.

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., Ring, R., Rutherford, E., Cabi, S., Han, T., Gong, Z., Samangooei, S., Monteiro, M., Menick, J., Borgeaud, S., Brock, A., Nematzadeh, A., Sharifzadeh, S., Binkowski, M., Barreira, R., Vinyals, O., Zisserman, A., and Simonyan, K. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2204.14198.

- An, W., Nie, J., Wu, Y., Tian, F., Lu, S., and Zheng, Q. Empowering multimodal llms with external tools: A comprehensive survey, 2025a. URL https://arxiv.org/ abs/2508.10955.
- An, X., Xie, Y., Yang, K., Zhang, W., Zhao, X., Cheng, Z., Wang, Y., Xu, S., Chen, C., Zhu, D., et al. Llavaonevision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661, 2025b.

Anthropic. Introducing Claude 4. Anthropic News, May 2025. URL https://www.anthropic.com/ news/claude-4. Accessed: [Insert Access Date].

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.

- Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu,

- Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang,
- Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

- Bai, T., Fan, Y., Qiu, J., Sun, F., Song, J., Han, J., Liu, Z., He, C., Zhang, W., and Yuan, B. Hallucination at a glance: Controlled visual edits and fine-grained multimodal learning. arXiv preprint arXiv:2506.07227, 2025c.

Cao, S., Li, J., Li, X., Pu, Y., Zhu, K., Gao, Y., Luo, S., Xin, Y., Qin, Q., Zhou, Y., et al. Unipercept: Towards unified perceptual-level image understanding across aesthetics, quality, structure, and texture. arXiv preprint arXiv:2512.21675, 2025.

Cartuyvels, R., Spinks, G., and Moens, M.-F. Discrete and continuous representations and processing in deep learning: Looking forward. AI Open, 2:143–159, 2021. ISSN 2666-6510. doi: 10.1016/j.aiopen.2021. 07.002. URL http://dx.doi.org/10.1016/j.

aiopen.2021.07.002.

Chen, G., Xu, W., Zhang, H., Chan, H. P., Zhao, D., Luu, A. T., and Rong, Y. Geopqa: Bridging the visual perception gap in mllms for geometric reasoning. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pp. 25680–25688, 2025a. URL https://arxiv.org/abs/2509.17437.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? In Advances in Neural Information Processing Systems, 2024a. URL https://arxiv.org/abs/ 2403.20330.

Chen, L., Xie, W., Liang, Y., He, H., Zhao, H., Yang, Z., Huang, Z., Wu, H., Lu, H., charles, Y., Bao, Y., Fan, Y., Li, G., Shen, H., Chen, X., Xu, W., Si, S., Cai, Z., Chai, W., Huang, Z., Liu, F., Liu, T., Chang, B., Hu, X., Chen, K., Ren, Y., Liu, Y., Gong, Y., and Li, K. Babyvision: Visual reasoning beyond language, 2026. URL https://arxiv.org/abs/2601.06521.

- Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.
- Chen, Y., Li, L., Xi, T., Zeng, L., and Wang, J. Perception before reasoning: Two-stage reinforcement learning for visual reasoning in vision-language models, 2025c. URL https://arxiv.org/abs/2509.13031.
- Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024b.

Chollet, F., Knoop, M., Kamradt, G., Landers, B., and Pinkard, H. Arc-agi-2: A new challenge for frontier ai reasoning systems, 2026. URL https://arxiv. org/abs/2505.11831.

Chung, J., Kim, J., Kim, S., Lee, J., Kim, M. S., and Yu, Y. v1: Learning to point visual tokens for multimodal grounded reasoning, 2025. URL https://arxiv. org/abs/2505.18842.

Dai, W., Li, J., Li, D., Tiong, A. M. H., Zhao, J., Wang, W., Li, B., Fung, P., and Hoi, S. Instructblip: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/ 2305.06500.

Daxberger, E., Wenzel, N., Griffiths, D., Gang, H., Lazarow, J., Kohavi, G., Kang, K., Eichner, M., Yang, Y., Dehghan, A., et al. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7395– 7408, 2025.

Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Fu, S., Bonnen, T., Guillory, D., and Darrell, T. Hidden in plain sight: Vlms overlook their visual representations, 2025. URL https://arxiv.org/abs/ 2506.08008.

Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N. A., Ma, W.-C., and Krishna, R. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, 2024. URL https://arxiv.org/abs/2404.12390.

Guo, D., Wu, F., Zhu, F., Leng, F., Shi, G., Chen, H., Fan, H., Wang, J., Jiang, J., Wang, J., Chen, J., Huang, J., Lei, K., Yuan, L., Luo, L., Liu, P., Ye, Q., Qian, R., Yan, S., Zhao, S., Peng, S., Li, S., Yuan, S., Wu, S., Cheng, T., Liu, W., Wang, W., Zeng, X., Liu, X., Qin, X., Ding,

- X., Xiao, X., Zhang, X., Zhang, X., Xiong, X., Peng, Y., Chen, Y., Li, Y., Hu, Y., Lin, Y., Hu, Y., Zhang, Y., Wu,
- Y., Li, Y., Liu, Y., Ling, Y., Qin, Y., Wang, Z., He, Z., Zhang, A., Yi, B., Liao, B., Huang, C., Zhang, C., Deng, C., Deng, C., Lin, C., Yuan, C., Li, C., Gou, C., Lou, C., Wei, C., Liu, C., Li, C., Zhu, D., Zhong, D., Li, F., Zhang, F., Wu, G., Li, G., Xiao, G., Lin, H., Yang, H., Wang, H., Ji, H., Hao, H., Shen, H., Li, H., Li, J., Wu, J., Zhu, J., Jiao, J., Feng, J., Chen, J., Duan, J., Liu, J., Zeng, J., Tang, J., Sun, J., Chen, J., Long, J., Feng, J., Zhan, J., Fang, J., Lu, J., Hua, K., Liu, K., Shen, K., Zhang, K., Shen, K., Wang, K., Pan, K., Zhang, K., Li, K., Li, L., Li, L., Shi, L., Han, L., Xiang, L., Chen, L., Chen, L., Li, L., Yan, L., Chi, L., Liu, L., Du, M., Wang, M., Pan, N., Chen, P., Chen, P., Wu, P., Yuan, Q., Shuai, Q., Tao, Q., Zheng, R., Zhang, R., Zhang, R., Wang, R., Yang,

- R., Zhao, R., Xu, S., Liang, S., Yan, S., Zhong, S., Cao,
- S., Wu, S., Liu, S., Chang, S., Cai, S., Ao, T., Yang, T., Zhang, T., Zhong, W., Jia, W., Weng, W., Yu, W., Huang,

- W., Zhu, W., Yang, W., Wang, W., Long, X., Yin, X., Li,
- X., Zhu, X., Jia, X., Zhang, X., Liu, X., Zhang, X., Yang,

- X., Luo, X., Chen, X., Zhong, X., Xiao, X., Li, X., Wu,
- Y., Wen, Y., Du, Y., Zhang, Y., Ye, Y., Wu, Y., Liu, Y., Yue, Y., Zhou, Y., Yuan, Y., Xu, Y., Yang, Y., Zhang, Y., Fang, Y., Li, Y., Ren, Y., Xiong, Y., Hong, Z., Wang, Z., Sun, Z., Wang, Z., Cai, Z., Zha, Z., An, Z., Zhao, Z., Xu,
- Z., Chen, Z., Wu, Z., Zheng, Z., Wang, Z., Huang, Z., Zhu, Z., and Song, Z. Seed1.5-vl technical report, 2025. URL https://arxiv.org/abs/2505.07062.

He, H., Li, G., Geng, Z., Xu, J., and Peng, Y. Analyzing and boosting the power of fine-grained visual recognition for multi-modal large language models. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2501.15140.

Huang, J.-T., Dai, D., Huang, J.-Y., Yuan, Y., Liu, X., Wang, W., Jiao, W., He, P., Tu, Z., and Duan, H. Human cognitive benchmarks reveal foundational visual gaps in mllms, 2026. URL https://arxiv.org/abs/ 2502.16435.

Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., and Lin, S. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. URL https://arxiv.org/abs/2503.06749.

Ishihara, S. Tests for colour-blindness, 1951.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Ke, F., Hsu, J., Cai, Z., Ma, Z., Zheng, X., Wu, X., Huang, S., Wang, W., Haghighi, P. D., Haffari, G., Krishna, R., Wu, J., and Rezatofighi, H. Explain before you answer: A survey on compositional visual reasoning, 2025. URL https://arxiv.org/abs/2508.17298.

Kim, S., Jin, J., Hong, S., Ka, D., Kim, H., and Noh, B. Dsc-llm: Driving scene context representation-based trajectory prediction framework with risk factor reasoning using llms. Sensors, 25(23):7112, 2025.

Kuchibhotla, H. C., Kancheti, S. S., Reddy, A. G., and Balasubramanian, V. N. Efficient vocabulary-free finegrained visual recognition in the age of multimodal llms. Transactions on Machine Learning Research, 2025. URL https://arxiv.org/abs/2505.01064.

Lai, X., Li, J., Li, W., Liu, T., Li, T., and Zhao, H. Minio3: Scaling up reasoning patterns and interaction turns

for visual search, 2025. URL https://arxiv.org/ abs/2509.07969.

Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vuli´c, I., and Wei, F. Imagine while reasoning in space: Multimodal visualization-of-thought. In Proceedings of the International Conference on Machine Learning, 2025a. URL https://arxiv.org/abs/2501.07542.

- Li, K., Shang, C., Karlinsky, L., Feris, R., Darrell, T., and Herzig, R. Latent implicit visual reasoning, 2025b. URL https://arxiv.org/abs/2512.21218.
- Li, L., Wei, Y., Xie, Z., Yang, X., Song, Y., Wang, P., An, C., Liu, T., Li, S., Lin, B. Y., Kong, L., and Liu, Q. Vl-rewardbench: A challenging benchmark for vision-language generative reward models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24657–24668, 2025c. doi: 10.1109/CVPR52734.2025.02296. URL https: //arxiv.org/abs/2411.17451.

Li, Z., Du, Y., Liu, Y., Zhang, Y., Liu, Y., Zhang, M., and Cai, X. Eagle: Elevating geometric reasoning through llm-empowered visual instruction tuning. arXiv preprint arXiv:2408.11397, 2024.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/ 2304.08485.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024.

Liu, L., Chen, L., Zhang, Z., Tian, M., Cui, H., Li, R., Liu, Z., Ju, Q., Li, Q., and Zhou, H.-Y. Skinflow: Efficient information transmission for open dermatological diagnosis via dynamic visual encoding and staged rl, 2026. URL https://arxiv.org/abs/2601.09136.

Liu, S., Yao, D., Lin, Y., Cong, G., and Bi, J. Traj-mllm: Can multimodal large language models reform trajectory data mining? arXiv preprint arXiv:2509.00053, 2025a.

Liu, Z., Gao, K., Liang, S., Xiao, B., Qiao, L., Ma, L., and Jiang, T. Beyond the visible: Benchmarking occlusion perception in multimodal large language models. arXiv preprint arXiv:2508.04059, 2025b.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pp. 2507–2521, 2022.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.02255.

Ma, W., Wang, C., Yuan, R., Chen, H., Dai, N., Zhou, S. K., Yang, Y., Yuille, A., and Chen, J. Causalspatial: A benchmark for object-centric causal spatial reasoning, 2026. URL https://arxiv.org/abs/2601.13304.

Marr, D. Vision. 1982.

Mathew, M., Karatzas, D., and Jawahar, C. V. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 2199–2208, 2021. doi: 10.1109/WACV48630.2021.00225. URL https:// arxiv.org/abs/2007.00398.

Mishra, A., Shekhar, S., Singh, A. K., and Chakraborty, A. Ocr-vqa: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pp. 947–952. IEEE, 2019.

Nagrani, A., Yang, S., Arnab, A., Jansen, A., Schmid, C., and Sun, C. Attention bottlenecks for multimodal fusion. In Advances in Neural Information Processing Systems, 2021. URL https://arxiv.org/abs/ 2107.00135.

OpenAI. Openai o3 model card, 2025. URL https://cdn.openai.com/pdf/ 2221c875-02dc-4789-800b-e7758f3722c1/ o3-and-o4-mini-system-card.pdf.

Ouyang, L., Qu, Y., Zhou, H., Zhu, J., Zhang, R., Lin, Q., Wang, B., Zhao, Z., Jiang, M., Zhao, X., et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24838–24848, 2025.

Qu, L., Zhang, H., Liu, Y., Wang, X., Jiang, Y., Gao, Y., Ye, H., Du, D. K., Yuan, Z., and Wu, X. Tokenflow: Unified image tokenizer for multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2545–2555, 2025.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, 2021. URL https://arxiv.org/abs/ 2103.00020.

Rudman, W., Golovanevsky, M., Bar, A., Palit, V., LeCun, Y., Eickhoff, C., and Singh, R. Forgotten polygons: Multimodal large language models are shapeblind. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2025. URL https://arxiv.org/abs/2502.15969.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pp. 618–626, 2017.

Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., and Cheng, Y. Openthinkimg: Learning to think with images via visual tool reinforcement learning, 2025. URL https://arxiv.org/ abs/2505.08617.

Sur´ıs, D., Menon, S., and Vondrick, C. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11888–11898, 2023.

Tang, H., Xie, C., Bao, X., Weng, T., Li, P., Zheng, Y., and Wang, L. Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278, 2025.

Team, B. S. Introduction to techniques used in seed1.6, June 2025a. URL https://seed.bytedance.com/ en/seed1_6. Technical Report.

Team, G. . Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, 2025b. URL https: //arxiv.org/abs/2507.06261.

Team, G. Gemini: A family of highly capable multimodal models, 2025c. URL https://arxiv.org/abs/ 2312.11805.

Team, K. Kimi k1.5: Scaling reinforcement learning with llms, 2025d. URL https://arxiv.org/abs/ 2501.12599.

Team, K. Kimi-vl technical report, 2025e. URL https: //arxiv.org/abs/2504.07491.

Team, O. M.-o. Minicpm-o 2.6: A gpt-4o level mllm for vision, speech, and multimodal live streaming on your phone, 2025f.

Team, Q. Qvq: To see the world with wisdom, December

2024. URL https://qwenlm.github.io/blog/ qvq-72b-preview/.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.

Wang, C., Luo, W., Dong, S., Xuan, X., Li, Z., Ma, L., and Gao, S. Mllm-tool: A multimodal large language model for tool agent learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 6678–6687, 2025a. doi: 10.1109/WACV61041.2025.00650. URL https:// arxiv.org/abs/2401.10727.

Wang, C., Meng, F., Pei, J., Xia, L., Liu, J., Yuan, X., and Di, X. Occ-mllm-cot: Self–correction enhanced occlusion recognition with large language models via 3d-aware supervision, chain–of–thoughts guidance. Image and Vision Computing, pp. 105881, 2025b.

Wang, X., Huang, J., Zhang, X., Wang, T., and Ma, J. W. Your reasoning benchmark may not test reasoning: Revealing perception bottleneck in abstract reasoning benchmarks, 2026a. URL https://arxiv.org/abs/ 2512.21329.

- Wang, X., Zhang, K., Jia, Q., Chen, Z., Zhai, G., and Min, X. Kidvis: Do multimodal large language models possess the visual perceptual capabilities of a 6-year-old?, 2026b. URL https://arxiv.org/abs/2601.08292.
- Wang, Y., Wu, S., Zhang, Y., Yan, S., Liu, Z., Luo, J., and Fei, H. Multimodal chain-of-thought reasoning: A comprehensive survey, 2025c. URL https://arxiv. org/abs/2503.12605.
- Wang, Z., Che, C., Wang, Q., Li, Y., Shi, Z., and Wang, M. Smolora: Exploring and defying dual catastrophic forgetting in continual visual instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 177–186, 2025d. doi: 10.1109/ICCV51701.2025.00024. URL https: //arxiv.org/abs/2411.13949.

Wang, Z., Zhang, T., Guo, M., Wang, Y., and Ye, Q. Videoanchor: Reinforcing subspace-structured visual cues for coherent visual-spatial reasoning, 2025e. URL https://arxiv.org/abs/2509.25151.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/ 2201.11903.

Weng, Z., Gomez, L., Webb, T. W., and Bashivan, P. Caption this, reason that: Vlms caught in the middle, 2025. URL https://arxiv.org/abs/2505.21538.

Wu, D., Liu, F., Hung, Y.-H., and Duan, Y. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025a.

Wu, H., Zhang, Z., Zhang, W., Chen, C., Liao, L., Li, C., Gao, Y., Wang, A., Zhang, E., Sun, W., et al. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. In Proceedings of the International Conference on Machine Learning, 2024. URL https://arxiv.

org/abs/2312.17090.

Wu, J., Xiong, Y., Li, X., Xia, Y., Wang, R., Wang, Y., Yu, T., Kim, S., Rossi, R. A., Yao, L., Shang, J., and McAuley, J. Mitigating visual knowledge forgetting in mllm instruction-tuning via modality-decoupled gradient descent. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2025b. URL https://arxiv.org/abs/2502.11740.

Wu, M., Yang, J., Jiang, J., Li, M., Yan, K., Yu, H., Zhang, M., Zhai, C., and Nahrstedt, K. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use, 2025c. URL https://arxiv.org/ abs/2505.19255.

Xiao, Y., Sun, E., Liu, T., and Wang, W. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024. URL https://arxiv.org/abs/ 2407.04973.

Xie, J., Yang, Z., and Shou, M. Z. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., Zhang, B., Wang, X., Chu, Y., and Lin, J. Qwen2.5-omni technical report. arXiv preprint arXiv:2503.20215, 2025a.

Xu, L., Zhao, Y., Wang, J., Wang, Y., Pi, B., Wang, C., Zhang, M., Gu, J., Li, X., Zhu, X., et al. Geosense: Evaluating identification and application of geometric principles in multimodal reasoning. arXiv preprint arXiv:2504.12597, 2025b.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., Wei, C., Yu, B., Yuan, R., Sun, R., Yin, M., Zheng, B., Yang, Z., Liu, Y., Huang, W., Sun, H., Su, Y., and Chen, W. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567,

2024. doi: 10.1109/CVPR52733.2024.00913. URL https://arxiv.org/abs/2311.16502.

Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yu, B., Zhang, G., Sun, H., et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2025. URL https://arxiv.org/abs/2409.02813.

Zhai, Y., Tong, S., Li, X., Cai, M., Qu, Q., Lee, Y. J., and Ma, Y. Investigating the catastrophic forgetting in multimodal large language models, 2023. URL https: //arxiv.org/abs/2309.10313.

Zhang, H., Wu, W., Li, C., Shang, N., Xia, Y., Huang, Y.,

- Zhang, Y., Dong, L., Zhang, Z., Wang, L., Tan, T., and Wei, F. Latent sketchpad: Sketching visual thoughts to elicit multimodal reasoning in mllms, 2025a. URL https://arxiv.org/abs/2510.24514.

Zhang, J., Khayatkhoei, M., Chhikara, P., and Ilievski, F. Mllms know where to look: Training-free perception of small visual details with multimodal llms. In International Conference on Learning Representations, 2025b. URL https://arxiv.org/abs/2502.17422.

Zhang, X., Quan, Y., Shen, C., Yuan, X., Yan, S., Xie, L., Wang, W., Gu, C., Tang, H., and Ye, J. From redundancy to relevance: Information flow in lvlms across reasoning tasks, 2024a. URL https://arxiv.org/abs/ 2406.06579.

- Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research, 2024b. URL https://arxiv.org/abs/ 2302.00923.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing ”thinking with images” via reinforcement learning, 2025. URL https://arxiv.org/abs/2505.14362.

Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

### A. TET Dataset Generation and Evaluation Details

- A.1. Generation Details

HiddenText. We render text characters using composite image elements that appear as recognizable letters/words when viewed at reduced scale but resolve into detailed scenes when magnified. The generation pipeline involves: (1) selecting target text strings, (2) decomposing each character into constituent visual elements, (3) replacing elements with thematically consistent image patches, and (4) compositing at multiple scales to verify the dual-interpretation property.

3DCaptcha. Characters are rendered with 3D perspective distortion and curved surfaces. We apply: (1) random rotation around all three axes, (2) perspective projection with varying focal lengths, (3) surface curvature following Bezier curves, and (4) realistic shading and shadow effects.

ColorBlind. Building on Ishihara plate design (Ishihara, 1951), we: (1) define target characters, (2) generate foreground dots with target-distinguishing chromaticity, (3) add background dots with confounding colors at similar luminance, and (4) introduce additional noise dots that share partial chromatic similarity with the target.

ChineseLigatures. We synthesize fused glyphs by: (1) decomposing source characters into radicals and strokes, (2) applying morphological transformations (scaling, rotation, skewing), (3) spatially compositing multiple characters with overlap, and (4) applying calligraphic style transfer for visual coherence.

- A.2. Evaluation Details

We evaluate 15 state-of-the-art models spanning three categories: (1) Unified multimodal models: Show-o2 (Xie et al., 2025), Bagel (Deng et al., 2025), Janus-Pro (Chen et al., 2025b); (2) Closed-source APIs: Claude 4-Sonnet (Anthropic, 2025), Gemini 2.5 Pro (Team, 2025b), OpenAI o1 (Jaech et al., 2024), Seed-1.6 (Team, 2025a); (3) Open-source models: Qwen2.5VL-72B (Bai et al., 2025b), QVQ-72B (Team, 2024), InternVL3-78B (Zhu et al., 2025), MiniCPM-o-2.6 (Team, 2025f), kimi-vl (Team, 2025e). We maintain original inference settings for unified models and configure all others with temperature 0.3 and 16384 max tokens.

### B. Detailed Pass@K Metrics for TET and More Response Case

Fig. 6 reports the pass@K curves across tasks, together with their standard deviations. As K increases, existing MLLMs show consistently flat performance trajectories: average accuracy exhibits little variation across tasks, and even the best-performing cases achieve gains of less than 4%.

This behavior indicates that these tasks are not amenable to improvement through expanded exploration in the reasoning space, in contrast to prior benchmarks where higher K often yields substantial gains. The uniformly flat curves suggest that the dominant bottleneck lies not in insufficient reasoning diversity or search depth, but in the models’ inability to access and preserve task-critical visual evidence. The failure cases in Fig. 7 further substantiate this diagnosis, with additional examples provided in Fig. 15.

- 0

- 1

- 2

- 3

- 4

3DCaptcha

ChineseLigatures

HiddenText

ColorBlind

Accuracy(%)

### C. Interpretation of MLLMs with Grad-CAM

1 2 4 8 16 32

To investigate why models, despite possessing sophisticated linguistic reasoning capabilities, still fail to perceive images accurately, we conducted Grad-CAM analyses on all datasets following the protocols of prior work (Selvaraju et al., 2017; Zhang et al., 2024a). We systematically examined two repre-

K of pass@K

Figure 6. Pass@k Results. Mean and variance on four TET tasks.

sentative models from the Qwen2.5-VL series (with 7B and 72B parameters), analyzing the attention maps of both the visual backbone network and the language decoder components. Attention patterns were uniformly sampled across multiple layers to capture the evolutionary process of attention throughout the architectural depth of each component. This analysis revealed a key insight: as visual reasoning is deferred to the language decoder downstream of the visual encoding bottleneck,

the spatial focus of attention becomes misaligned with the task-critical visual evidence. Consequently, no matter how the parameter count of the language decoder is scaled up, the model still cannot perceive images accurately.

Information flow in the visual encoder. As illustrated in Figures 4, 8, 9, and 10, the visual encoder operates without access to downstream reasoning requirements, functioning solely to extract generic image representations for the LLM decoder. This embodies the passive, one-off feature encoding process critiqued in our framework: the encoder must compress visual information into a fixed representation before knowing what spatial distinctions the reasoning stage will require. This temporal-spatial mismatch manifests in systematic attention failures. While the ViT allocates attention across various image regions, this attention frequently falls outside target character regions or captures only partial segments, prioritizing salient object-level features over task-critical spatial details. Because the decision of when to reason (deferred to language generation) is made before knowing where visual evidence resides, the encoder cannot preserve the fine-grained geometric and topological distinctions that these tasks require. In the 3DCaptcha task, the ViT exhibits region-specific rather than global recognition patterns, demonstrating how this passive encoding paradigm degenerates into a lossy compression that discards task-critical spatial signals before reasoning begins—signals that cannot be recovered through subsequent language-side deliberation.

Information flow in the LLM decoder. As shown in Figures 4, 8, 9, and 10, the LLM decoder—where actual reasoning occurs in current architectures—exhibits consistent attention failures across both model scales (7B and 72B). Except for the ChineseLigature task, the decoder systematically fails to attend to regions containing task-critical visual information, instead scattering attention over irrelevant areas or entirely missing essential visual elements. This pattern reveals the core structural problem: by the time reasoning begins in the decoder, the visual representation has already been collapsed into a text-aligned interface that no longer preserves the pixel-level distinctions necessary for correct decisions. The decoder cannot “look back” to recover lost spatial information; it can only reason within the impoverished visual representation it receives. What is needed is not passive encoding followed by language reasoning, but rather active, reasoning-driven perception where the encoding process itself is guided by task-specific requirements—allowing high-level reasoning to dynamically query and steer visual feature extraction to preserve the precise spatial evidence needed for decision-making.

Speculations on the causes of failure across different tasks. These failures demonstrate that task-critical spatial signals are collapsed before reasoning begins and cannot be recovered through language-side deliberation.

As shown in Fig. 4, for HiddenText, MLLMs fail to recognize symbols formed by spatially distributed objects, indicating that geometric relationships and global compositional structure are not preserved through the text-aligned encoding bottleneck. The passive encoding captures local features but discards the relational topology needed for integration.

- As shown in Fig. 8, for 3DCaptcha, models cannot disambiguate overlapping characters, demonstrating that depth cues, occlusion boundaries, and figure-ground segmentation—measurement-like operations requiring continuous spatial reasoning—cannot be executed in discrete textual space.
- As shown in Fig. 9, for ColorBlind, MLLMs struggle to construct characters from same-colored dots scattered among

[Figure 74]

[Figure 75]

###### Response

The user wants me to identify the word represented in the image.\n 1. **Analyze the image:** The image shows a large, ornate building with towers, turrets, and high walls, situated on a cliff by a body of water.\n\n 2. **Identify the object:** This type of structure is universally recognized as a castle.\n\n 3. **Look for literal text:** I have scanned the image and its cropped versions for any written words on flags, banners, or walls. There is no discernible text present.\n\n 4. **Interpret the request:** Since there is no literal text, the request is asking for the word that describes the main subject of the image.\n\n 5. **Conclusion:** The central and most prominent subject in the image is a castle.\n\nTherefore, the word that appears in the image is "castle". \n\n \box{castle}"

Instruction You are given an image. Please identify what word appears in this image. Your final answer should be enclosed in \\box{} (e.g., \\box{word}).

[Figure 76]

Figure 7. Model response on question of HiddenText. The goal is to identify the hidden word in an image. Gemini-2.5-Pro-0506 answers the hidden word as “castle”.

Answer: ROAROMA GT: DQQ5N

|ViT<br><br>LLM<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]<br><br>[Figure 92]|
|---|

Qwen2.5-VL-72B Answer: EUAZOMA GT: DQQ5N

|LLM<br><br>ViT<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]<br><br>[Figure 100]<br><br>[Figure 101]<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]<br><br>[Figure 106]<br><br>[Figure 107]<br><br>[Figure 108]|
|---|

Figure 8. Grad-CAM of Qwen2.5-VL Series Models on 3DCaptcha.

noise, illustrating that perceptual grouping—conceptually simple but hard to verbalize—requires active visual reasoning on pixel-level representations.

- As shown in Fig. 10, for ChineseLigature, MLLMs perceive individual components but cannot compositionally extend them into plausible phrases, revealing that fine-grained spatial arrangements—distinctions hard to specify in text—are lost during one-off encoding.

### D. Grad-CAM Analysis of Fine-Tuned MLLMs

Fig. 11, 12, 13 presents Grad-CAM visualizations of Qwen2.5-VL-7B before and after vision module fine-tuning across different datasets. Following vision module fine-tuning, the model demonstrates enhanced perceptual capabilities, as attention coverage over effective character regions across inter-module interactions increases. This phenomenon validates that targeted optimization of the vision module effectively improves the generalization of the model’s perceptual patterns.

### E. Further details of supervised fine-tuning

To validate that visual perception, rather than language reasoning, is the primary bottleneck, we incorporate traditional benchmarks—OCR-VQA, GeoQA, and CLEVR—for comparison. As shown in Fig. 14, on TET tasks, configurations excluding visual fine-tuning plateau early, while those including it converge efficiently. In contrast, all configurations converge similarly on traditional benchmarks. This divergence reveals that traditional tasks fall within current MLLM pretraining coverage, requiring only language-side improvements, while TET demands enhanced visual perception that cannot be compensated by scaling language reasoning alone. This supports our claim that the limiting factor is representational access—visual fine-tuning preserves task-critical spatial distinctions that passive encoding collapses before reasoning begins.

### F. More Alternative Views and Counterarguments

“Perception and Language Should Remain Separate.” From a modularity perspective, clean separation aids interpretability and compositionality. We acknowledge this concern but note that biological vision does not maintain such separation (Marr, 1982). The question is whether the interface between modules preserves sufficient information—our evidence suggests

Answer: 74 GT: M3

|ViT<br><br>LLM<br><br>[Figure 109]<br><br>[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]<br><br>[Figure 119]<br><br>[Figure 120]<br><br>[Figure 121]<br><br>[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]|
|---|

Qwen2.5-VL-72B Answer: 74 GT: M3

|LLM<br><br>ViT<br><br>[Figure 125]<br><br>[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]<br><br>[Figure 138]<br><br>[Figure 139]<br><br>[Figure 140]|
|---|

Figure 9. Grad-CAM of Qwen2.5-VL Series Models on ColorBlind subset.

current interfaces do not for fine-grained visual tasks. (Cartuyvels et al., 2021; Nagrani et al., 2021)

“These Tasks Are Artificial.” One might dismiss TET tasks as contrived edge cases. We counter that these are diagnostic: they isolate specific failure modes that manifest subtly in realistic settings. Failures on 3DCaptcha predict failures on real-world 3D reasoning; failures on ChineseLigatures predict failures on fine-grained document understanding (Mathew et al., 2021). TET is a stress test, not an end goal.

### G. Details of Implications and Future Direction Our position suggests several directions for future research:

Architectural Design. The design space should prioritize mechanisms for reasoning within visual representations—recurrent refinement, cross-attention back to pixels, or hybrid continuous-discrete representations—over scaling text-side computation.

Training Objectives. Contrastive objectives (e.g., CLIP) optimize for semantic alignment at the expense of structural fidelity. Future objectives should explicitly preserve fine-grained spatial information required for downstream reasoning, potentially through multi-scale reconstruction losses or geometric consistency constraints.

Benchmark Development. Current benchmarks often conflate perceptual and reasoning capabilities. We advocate for diagnostic probes like TET that isolate perceptual failures, enabling precise identification of architectural bottlenecks.

Adaptive Routing. Not all tasks require pixel-level reasoning. Future architectures might learn to route computation through visual vs. text pathways based on task demands—reasoning in visual space for spatial tasks, in text space for semantic tasks.

### H. Architectural Instantiations of AVQ

The AVQ framework admits multiple concrete realizations, each implementing the query-retrieval-update cycle (Q, A, U) with distinct trade-offs:

Cross-Attention to Pixels. The reasoning module repeatedly attends back to the original image (or early feature maps) when resolving ambiguous predicates. Here Q is implicit in the query vectors of attention layers, A is cross-attention over

Answer: 想班肚肰 GT: 不想上班那就别上

Qwen2.5-VL-7B

|ViT<br><br>LLM<br><br>[Figure 141]<br><br>[Figure 142]<br><br>[Figure 143]<br><br>[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]|
|---|

Qwen2.5-VL-72B Answer: 憩班肧肚 GT: 不想上班那就别上

|LLM<br><br>ViT<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]<br><br>[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>[Figure 167]<br><br>[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]|
|---|

Figure 10. Grad-CAM of Qwen2.5-VL Series Models on ChineseLigature subset.

visual tokens, and U is the residual update. This maintains pixel-level access but lacks explicit query generation (Q) and perception refinement (P)—the visual representation remains static.

Recurrent Visual Transformers. Stacking transformer blocks with shared weights enables iterative refinement within visual latent space (Zhang et al., 2025a). The recurrence implements P and U through repeated processing, but Q remains implicit—refinement is not explicitly conditioned on semantic hypotheses from the reasoning process.

Generative Refinement. Diffusion-based models that “re-render” interpretations in visual space (Chung et al., 2025) implement P and U as iterative denoising, maintaining pixel-level fidelity throughout. However, the conditioning signal acts as a fixed prompt rather than dynamically generated queries—Q is absent.

No existing architecture fully instantiates AVQ with explicit query generation (Q), reasoning-driven retrieval with perception refinement (A), and grounded state update (U). This gap defines a concrete research agenda.

###### Qwen2.5-VL-7B

Answer: No Text GT: Love

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

ViT

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

LLM

Qwen2.5-VL-7B-FT Answer: Love GT: Love

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

ViT

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

LLM

Figure 11. Grad-CAM of Qwen2.5-VL-7B before and after visual fine-tuning on HiddenText.

###### Qwen2.5-VL-7B

Answer: 74 GT: M3

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

ViT

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

LLM

###### Qwen2.5-VL-7B-FT Answer: M3 GT: M3

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

ViT

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

LLM

Figure 12. Grad-CAM of Qwen2.5-VL-7B before and after visual fine-tuning on ColorBlind.

###### Qwen2.5-VL-7B

Answer: ROAROMA GT: DQQ5N

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

ViT

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

LLM

Qwen2.5-VL-7B- Answer: QQQ5N GT: DQQ5N FT

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

ViT

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

LLM

Figure 13. Grad-CAM of Qwen2.5-VL-7B before and after visual fine-tuning on 3DCaptcha.

Full parameters

2.5

Vision encoder only

Vision encoder with adapter

2.0

Language backbone only

Vision-language adapter only

1.5

Loss

1.0

0.5

0.0

0 25 50 75 100 125 150 175

Training Steps

(a) ColorBlind

3.5

Full parameters

3.0

Vision encoder only

Vision encoder with adapter

2.5

Language backbone only

Vision-language adapter only

2.0

Loss

1.5

1.0

0.5

0.0

0 25 50 75 100 125 150 175

Training Steps

(b) 3DCaptcha

3.0

Full parameters

Vision encoder only

2.5

Vision encoder with adapter

Language backbone only

2.0

Vision-language adapter only

Loss

1.5

1.0

0.5

0.0

0 25 50 75 100 125 150 175

Training Steps

(c) HiddenText

Full parameters

Vision encoder only

8

Vision encoder with adapter

Language backbone only

6

Vision-language adapter only

Loss

4

2

0

0 50 100 150 200 250 300 350

Training Steps

(d) OCRVQA

Full parameters

Full parameters

2.5

- 0

- 1

- 2

- 3

- 4

Vision encoder only

Vision encoder only

Vision encoder with adapter

Vision encoder with adapter

2.0

Language backbone only

Language backbone only

Vision-language adapter only

Vision-language adapter only

1.5

Loss

Loss

1.0

0.5

0.0

0 50 100 150 200 250 300 350

0 50 100 150 200 250 300 350

Training Steps

Training Steps

(e) GEOQA

(f) CLEVR

- Figure 14. Training Curves. Training loss curve for different settings on finetuning parameters for both our tasks and traditional tasks.

[Figure 269]

You are given an image. Please identify what word appears in this image. Your final answer should be enclosed in \box{}.

[Figure 270]

Based on a close examination of the image and its cropped sections, the most plausible word that appears in the image is formed by a distinct shape on the building.\n\nOn the white, upper section of the building, below the three vertical black slats, there is a circular objectthat resembles a security camera or an outdoor light. This circular shape can be

interpreted as the letter \"O\". Since \"O\" is also a word in the English language, this is the most likely answer.\n\nOther text-like features, such as the sign on the right-hand side of the building, appear to be garbled and do not form a recognizable word, which is a common artifact in AI-generated images like this one. The circular object provides the clearest and most interpretable representation of a letter that is also a word.\n\n\box{O}.

[Figure 271]

HiddenText Gemini-2.5pro-0506

[Figure 272]

You are given an image. Please identify the characters in this image. Your final answer should be enclosed in \box{}.

[Figure 273]

###### This image does not contain any characters. It appears to be a graphical representation, such as a waveform or a 3D surface plot. Therefore, there are no characters to identify.\n\n\box{This image does not contain any characters}.

[Figure 274]

3DCaptcha Gemini-2.5pro-0506

[Figure 275]

You are given an image. Please identify the Chinese phrase, sentence, idiom, proverb, or solar term displayed in this image. Your final answer should be enclosed in a \box{}.

[Figure 276]

This is a highly stylized and auspicious composite Chinese character. It is an artistic representation of the character

**鸞 (luán)**.\n\nThe standard character 鸞 refers to a mythical bird, similar to a phoenix, that symbolizes good fortune, peace, and marital harmony.\n\nThis specific artistic version is created by combining several characters with positive meanings, all packed into a single glyph:\n **王王 (wáng wáng) **: Two \"king\" characters, symbolizing supreme status or great prosperity.\n**令 (lìng) **: Meaning \"good\" or \"order\".\n* **懿 (yì) **: Meaning \"virtuous\" or \"beautiful\".\n **禾 (hé) **: Meaning \"grain,\" symbolizing a bountiful harvest.\n**鳥 (niǎo) **: The \"bird\" radical, indicating the nature of the character.\n\nAll these components are combined to create a superauspicious version of the already fortunate character 鸞.\n\n\box{鸞}.

[Figure 277]

ChineseLigatures

Gemini-2.5pro-0506

[Figure 278]

You are given an image. Please identify the charactersin this image. Your final answer should be enclosed in \box{}.

[Figure 279]

The image contains characters formed by a pattern of dots. By focusing on the cluster of brown dots against the background of green and cyan dots, one can discern two letters. The letter on the left is a 'U', and the letter on the right is a 'C'.\n\n\box{UC}.

[Figure 280]

ColorBlind Gemini-2.5pro-0506

- Figure 15. Example responses from the Gemini model on the four tasks of TET. For each task, the model fails to provide correct analysis and conclusion based on its flawed initial perception.

