# arXiv:2601.17027v1[cs.CV]17Jan2026

[Figure 1]

## Scientific Image Synthesis: Benchmarking, Methodologies, and Downstream Utility

Honglin Lin1,2†, Chonghan Qin3,2†, Zheng Liu4,2†, Qizhi Pei2, Yu Li2, Zhanping Zhong1,2, Xin Gao1,2, Yanfeng Wang1, Conghui He2, Lijun Wu2∗

1Shanghai Jiao Tong University, 2OpenDataLab, Shanghai Artificial Intelligence Laboratory, 3The University of Hong Kong, 4Peking University

While synthetic data has proven effective for improving scientific reasoning in the text domain, multimodal reasoning remains constrained by the difficulty of synthesizing scientifically rigorous images. Existing Text-to-Image (T2I) models often produce outputs that are visually plausible yet scientifically incorrect, resulting in a persistent visual–logic divergence that limits their value for downstream reasoning. Motivated by recent advances in next-generation T2I models, we conduct a systematic study of scientific image synthesis across generation paradigms, evaluation, and downstream use. We analyze both direct pixel-based generation and programmatic synthesis, and propose ImgCoder, a logic-driven framework that follows an explicit “understand → plan → code” workflow to improve structural precision. To rigorously assess scientific correctness, we introduce SciGenBench, which evaluates generated images based on information utility and logical validity. Our evaluation reveals systematic failure modes in pixel-based models and highlights a fundamental expressiveness–precision trade-off. Finally, we show that fine-tuning Large Multimodal Models (LMMs) on rigorously verified synthetic scientific images yields consistent reasoning gains, with potential scaling trends analogous to the text domain, validating high-fidelity scientific synthesis as a viable path to unlocking massive multimodal reasoning capabilities.

Date: January 27, 2026 Equal contribution: Honglin Lin, Chonghan Qin, Zheng Liu Correspondence: Lijun Wu, wulijun@pjlab.org.cn Project Page: https://SciGenbench.github.io

## 1 Introduction

With the advancement of Large Multimodal Models (LMMs) [12, 24, 1], enabling robust reasoning in scientific domains such as mathematics, physics, and engineering has become an increasingly important goal [39, 34, 31]. In the text-only setting, large-scale synthetic data has proven effective in alleviating data bottlenecks and improving scientific reasoning [14, 17, 18]. However, multimodal scientific reasoning remains significantly underdeveloped. Beyond data scarcity, a fundamental challenge lies in the difficulty of synthesizing scientifically rigorous images. Unlike natural images, scientific visuals must satisfy strict geometric, physical, and relational constraints, which existing Text-to-Image (T2I) models often fail to enforce, producing visually plausible yet scientifically incorrect results.

The recent emergence of next-generation T2I models [33, 25, 28], such as Nanobanana-Pro [13], has renewed interest in this challenge. These models promise improved semantic understanding and visual control, raising a natural question: can modern T2I systems overcome the long-standing limitations of scientific image synthesis and unlock scalable multimodal data generation for reasoning? To answer this question, we propose a systematic investigation with methodological innovation, benchmarking, and

downstream reasoning.

From a methodological perspective, scientific image synthesis can be broadly categorized into two paradigms. Pixel-based T2I models generate images end-to-end [2, 33, 32, 6, 4] and offer strong visual expressiveness, but often struggle to reliably satisfy strict structural constraints. In contrast, programmatic approaches [23, 36] generate images through explicit, executable codes, providing stronger structural control. Building on this paradigm, we propose ImgCoder, a logic-driven framework that follows an “Understand→Plan→Code” workflow to decouple reasoning from rendering. Our analysis reveals a fundamental precision–expressiveness trade-off between these two paradigms.

To rigorously evaluate scientific image generation, we observe that existing benchmarks and metrics—largely focused on pixel similarity or coarse semantic alignment—are insufficient for assessing scientific correctness, where correctness hinges on precise structural and relational validity. We therefore introduce SciGenBench, a specialized benchmark comprising 1.4K problems that are designed to evaluate the information utility and logical validity of generated scientific images across 5 domains and 25 image types. SciGenBench adopts a hybrid evaluation protocol that combines fine-grained LLM-as-Judge scoring with an automated inverse validation pipeline, shifting the focus from visual aesthetics to information utility and logical rigor. Our evaluation uncovers a pervasive visual–logic divergence in current pixel-based models, while code-driven approaches achieve higher structural precision at the cost of reduced expressiveness. We further categorize these failure modes, providing a systematic understanding of where and why different paradigms break down.

Finally, we examine the downstream utility of synthetic scientific images for multimodal reasoning. By grounding image synthesis in scientifically verified textual sources, the resulting visual–text pairs provide reliable supervision for training. Experiments demonstrate that fine-tuning LMMs on such rigorously verified synthetic data yields consistent improvements in scientific reasoning, with evidence of potential scaling trends similar to those observed in text-only settings. These findings suggest that high-fidelity scientific image synthesis offers a viable and scalable pathway toward advancing multimodal scientific reasoning.

## 2 Related Work

### 2.1 Scientific Diagram Generation

Text-to-Image Models. Following traditional UNet-based Latent Diffusion Models (LDMs) [27], Diffusion Transformers (DiT) [26, 9, 2] and Autoregressive Transformers [32, 6, 8] have recently emerged to dominate modern text-to-image generation. While proprietary systems [13, 25] and opensource models [33, 4] achieve high visual fidelity, generating structured diagrams directly in pixel space remains challenging [41], revealing a gap between visual realism and structural correctness.

Code-based Methods. Early approaches generate visualization code from captions [3], while later works scale instruction-based plotting via curated code corpora [23]. Some methods rely on manually defined templates [36, 37], at the cost of limited output diversity.

### 2.2 Benchmarking Text-to-Image Generation

From Fidelity to Reasoning. Early benchmarks [38, 15, 16, 7, 11] focus on visual fidelity and semantic alignment via automated metrics (e.g., FID, CLIP), while recent works shift toward Reasoning-Informed evaluation [5], assessing physical plausibility or contextual consistency in natural scenes (e.g., T2IReasonBench [30], RISEBench [40]).

The Gap: Reasoning-Oriented Synthesis. In contrast, scientific image synthesis requires explicitly materializing abstract axioms into precise visual structures (e.g., circuit topologies). Unlike natural scenes with implied logic, these tasks demand active computation under strict domain laws.

###### Scientific Image Generation SciGenBench Construction

###### Data Selection Hierarchical Taxonomy Visual-Dependent Quiz Generation

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Sparse Reasoning Data

[Figure 9]

5 Subject

Previous Model

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Text Problem

Rigorous, Precise Revisit Data Synthesis

[Figure 22]

[Figure 23]

Fact Extration

Hindered Performance

Text Problems

Highquality Problems

Vis. Filtration

[Figure 24]

Enhanced Model

[Figure 25]

Math Phys Chem Bio Universal

[Figure 26]

Atomic Quizes

Closed-source Models Open-source Models

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Qwen-Image, HunyuanImage-3.0

[Figure 31]

[Figure 32]

Nano Banana Pro, GPT-Image-1.5

[Figure 33]

[Figure 34]

Blind Fitration

25 Image Types

[Figure 35]

Multimodal Problems

Reference Real Image

(e.g., Molecular Structure, Circuit Diagram)

[Figure 36]

Final Quizes

Evaluation Framework

[Figure 37]

[Figure 38]

[Figure 39]

5 dimentional LLM-as-Judge

###### Factual Quiz Test

Struggling with Text Blur & Semantic Misalignment

Powerful but high cost

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Correctness

[Figure 44]

Scientific Plausibility

[Figure 45]

[Figure 46]

ImgCoder: Logic-Driven Programmic Framework

Gen Image

[Figure 47]

[Figure 48]

Layout

[Figure 49]

[Figure 50]

[Figure 51]

VQA Model

[Figure 52]

[Figure 53]

[Figure 54]

Accuracy

[Figure 55]

Expressiveness

[Figure 56]

Quiz

Readability

2. Plan 3. Code

1. Understand

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Content: [Incline]: Triangle with

Dist. Consistency Downstream Task

Original Question: A 5kg block slides down a frictionless inclined plane at 30°. It is connected by a massless string over a pulley to a hanging 3kg mass. Calculate the tension in the string.

$30^\circ$ slope. [Block 1]: Rotated $30^\circ$ to sit flush on the slope. [Pulley]: Circle at the top vertex...

Layout: [Incline]: Base length approx. 10 units. Height determined by $\tan(30^\circ)$.... Label: [Masses](5kg, 3kg), [angle] (30°)

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

FID, SSIM, PSNR

[Figure 68]

Data Synthesis Finetuning

Considerations: Use ax.set_aspect('equal') to ensure the 30° angle is visually accurate. [Block 1] must be rotated by 30° to align flush with the slope...

Reference Image

Generated Image

Embedding

Balanced Cost & High Fidelity

[Figure 69]

Performance Gain (Acc ++)

- Figure 1: Methodological Overview. The framework consists of three core components: (1) Scientific Image Generation (Left), where we propose ImgCoder, a programmatic approach decoupling planning from implementation to outperform pixel-based baselines; (2) SciGenBench Construction (Top Right), a rigorously curated benchmark with a fine-grained taxonomy and atomic quizzes; and (3) Evaluation Framework (Bottom Right), a multi-faceted assessment system combining LMM judges, inverse validation, standard metrics, and downstream performance.

## 3 Scientific Image Generation

We investigate the scientific image synthesis by comparing two different paradigms. As shown in Figure 1 (Left), pixel-based models synthesize images end-to-end from text, while code-driven approaches generate executable specifications that are deterministically rendered. Our method, ImgCoder, adopts the code-driven paradigm and implements a logic-first “Understand → Plan → Code” workflow, enabling explicit planning and structured execution for science-intensive visuals.

### 3.1 Task Definition

Let T denote the space of scientific descriptions and I the image space. We formulate scientific image generation as a conditional generation problem with a parameterized model Mθ : T → I. Unlike open-domain synthesis, valid images are constrained by a set of latent scientific axioms A. Given a query T ∈ T , the goal is to generate an image I∗ that maximizes the likelihood of the reasoning outcome y under a downstream solver S:

I∗ = argmax

PS(y | I, T, A), (1)

I∼Mθ(·|T)

where y is determined by A and PS denotes the probability assigned by the solver to y. This objective encourages images that are not only visually coherent but also structurally faithful for downstream reasoning.

### 3.2 Pixel-based Synthesis Framework

This framework represents the direct synthesis paradigm, treating scientific image generation as an end-to-end translation task from textual descriptions to visual pixel space. We leverage SOTA T2I models as generators, including both proprietary systems (e.g., Nanobanana-Pro, GPT-Image-1.5, Seedream-4.0, Flux2) known for their robust instruction following, and leading open-source models (e.g., Qwen-Image, HunyuanImage-3.0) representing accessible multimodal capabilities.

To bridge the gap between general T2I capabilities and STEM requirements, we employ a constraintinjection strategy (Prompt 14) for image generation. This pipeline regulates generation by enforcing information fidelity (explicit visualization of all entities), applying strict negative constraints (preventing solution leakage like calculation steps), and ensuring stylistic standardization (prioritizing a clean, textbook-style aesthetic).

### 3.3 ImgCoder: A Logic-Driven Programmatic Framework

While models like Nanobanana-Pro show initial promise for scientific image synthesis, pixel-based approaches remain limited in enforcing strict logical constraints and supporting open research. We therefore explore a programmatic paradigm and propose ImgCoder, which follows an explicit “Understand → Plan → Code” workflow to generate executable code (e.g., Python) for deterministic image rendering.

To mitigate the chaotic layouts often observed when visualization code is generated directly from text, ImgCoder introduces an explicit reasoning and planning stage prior to code synthesis. Following a “Think-before-Act” strategy (Prompt 15), the model is required to construct a structured chain-ofthought that captures the complete visualization intent before emitting code. Concretely, the planning phase explicitly defines four aspects of the target figure: (1) Image Content, by exhaustively identifying all geometric entities, physical components, and their logical relationships; (2) Layout, by pre-planning coordinate systems and topological arrangements to prevent visual clutter and unintended overlaps; (3) Labels, by determining both the semantic content and precise anchor points of textual annotations; and (4) Drawing Constraints, by validating the plan against domain-specific axioms (e.g., geometric rules or physical laws) while strictly avoiding answer leakage.

By decoupling logical planning from syntactic implementation, this plan-then-code paradigm substantially improves both the compilation success rate and the logical fidelity of complex scientific illustrations. To examine the scalability of ImgCoder across model backbones, we implement two variants: Qwen3-ImgCoder and Gemini3-ImgCoder, built upon Qwen3-235B-Instruct [35] and Gemini3, respectively.

## 4 SciGenBench: Benchmarking Scientific Image Synthesis

We next turn to evaluation. Standard image-generation metrics are not designed for scientific imagery, where even minor structural errors can invalidate the underlying semantics. We therefore introduce SciGenBench to benchmark the paradigms in Section 3 by measuring the information utility and logical correctness of generated images. Figure 1 (Right) overviews our pipeline, which integrates curated data construction, a hierarchical taxonomy, and a hybrid evaluation framework.

### 4.1 Data Acquisition & Selection

SciGenBench is designed with two complementary data sources that serve distinct evaluation purposes: a primary instruction-driven benchmark for assessing synthesized scientific images, and a real-world visual reference set for distributional comparison.

- Table 1: Evaluation Dimensions for LMM-as-Judge. Gemini-3-Flash scores images (0-2) across five criteria.

Dimension Evaluation Criteria & Focus Correctness & Fidelity Strict prompt adherence; detects compositional errors (quantity/attribute mismatch). Layout & Precision Precision of geometric construction, topological accuracy, and coordinate system alignment. Readability & Occlusion Clarity of textual labels, checking for occlusion, garbled text, or background interference. Scientific Plausibility Conformity to domain-specific axioms (e.g., valency balance, Newton’s laws). Expressiveness & Richness Ensures holistic context and scenario completeness, avoiding isolated elements.

- • Verified Scientific Instructions. The core of SciGenBench is constructed from high-quality scientific text corpora, including MegaScience [10] and WebInstruct-verified [22], which are selected for their logical correctness and factual rigor. To ensure suitability for visual generation, we apply a visualizability filtration (Prompt 16) to remove non-visual content (e.g., abstract derivations), retaining only descriptions with concrete, imageable structures.
- • Real-World Visual Reference. To contextualize synthetic scientific images against authentic visual data, we incorporate a vision-optional subset of SeePhys [34] as a human-authored reference set, denoted SciGenBench-SeePhys. This subset participates in all evaluation dimensions and serves as the exclusive ground truth for reference-based standard image metrics used in real–synthetic comparison.

### 4.2 Hierarchical Taxonomy

To establish a structured data distribution, we organize SciGenBench using a rigorous two-level

“Subject–Image Type” taxonomy. We employ Gemini-3-Flash as a unified data curator (Prompt 16) to perform visualizability filtration and fine-grained classification jointly within a single inference pass, enabling scalable and consistent annotation. The resulting taxonomy is defined along two hierarchical dimensions. (1) 5 Subjects comprise Mathematics, Physics, Chemistry, Biology categories, and a Universal category that captures cross-domain visual structures. (2) 25 Image Types include fine-grained categories reflecting domain-specific visual conventions, such as molecular structure in Chemistry or circuit diagram in Physics, as well as generic types under the Universal subject, including chart & graph and experimental setup. Detailed definitions of all image types are provided in Appendix C.

### 4.3 Visual-Dependent Quiz Generation

To ensure that generated images provide high information utility and are strictly indispensable for problem solving, we design an automated “Generate–Filter–Select” pipeline to construct atomic, visually grounded quizzes.

Fact-based Question Formulation. Using Gemini-3-Flash with Prompt 17, we first analyze each source instruction to extract verifiable factual elements, such as numerical values, geometric relations, or domain-specific properties (e.g., electron configurations). These elements are then converted into atomic questions, each targeting a single, concrete piece of visual information.

Blind Filtration for Visual Necessity. To eliminate pseudo-multimodal questions that can be solved without visual input, we introduce a blind filtration mechanism. Specifically, GPT-5-nano is employed as a blind solver that receives only the above question text, without access to image. Each question is evaluated across 4 independent trials; those answered correctly in all trials are discarded, as they indicate text leakage or reliance on commonsense rather than visual evidence.

Density-based Selection and Human Review. To prioritize information-rich samples, we retain images associated with the highest number of valid atomic quizzes within each image type. The

resulting quiz set is further reviewed by expert annotators to ensure logical consistency, scientific rigor, and the absence of hallucinations.

### 4.4 Evaluation Framework

Evaluating scientific image generation poses unique challenges, as traditional pixel-level metrics fail to capture logical correctness and scientific factuality. To establish a rigorous standard, we adopt a hybrid evaluation strategy comprising automated judge, inverse validation, traditional metrics and downstream utility assessment, as shown in Figure 1 (Bottom Right).

Multi-dimensional LMM-as-Judge. Following the structured rubric defined in Prompt 18, we employ Gemini-3-Flash as an automated evaluator. For each generated image, the judge produces a reasoning critique along with a score s ∈ {0,1,2} for each evaluation dimension specified in Table 1, enabling fine-grained and holistic assessment of visual logical correctness.

Inverse Quiz Validation. To quantify whether a generated image faithfully encodes its intended information, we introduce an inverse validation metric. Let QI denote the set of atomic quizzes associated with image I, and V(I, q) ∈ {0,1} indicate the correctness of a strong VQA model’s response to question q ∈ QI. We define the inverse validation rate (Rinv) as the proportion of images for which all associated quizzes are answered correctly:

1

|D| ∑

I ∑

V(I, q) = |QI| , (2)

Rinv =

I∈D

q∈QI

where I(·) is the indicator function and D denotes the evaluation set. To ensure that validation failures primarily reflect informational deficiencies in the image rather than limitations of the solver, we use Gemini-3-Flash as the VQA engine.

Reference-based Standard Metrics. For comparability with prior T2I research, we also report conventional automated metrics (FID, PSNR, SSIM) computed against ground-truth images. These metrics quantify the visual domain gap between synthetic and authentic scientific images, but serve as auxiliary references due to the sparse pixel distribution of scientific diagrams. These metrics are applied only to images from SciGenBench-SeePhys.

Data Utility for Training. As the ultimate measure of quality, we evaluate the performance gains of LMMs fine-tuned on synthetic images produced by different generation paradigms. This directly assesses the functional utility and logical consistency of synthesized data from a downstream reasoning perspective (see Section 6). To prevent textual shortcuts during training, we apply a multimodal adaptation strategy (Prompt 19), which masks textual cues and enforces reliance on visual evidence.

## 5 Benchmarking Generative Capabilities

In this section, we aim to address pivotal research questions (RQs) regarding scientific image synthesis through comprehensive benchmarking. RQ1- Generative Capability: How do current SOTA models perform in scientific image generation task? RQ2 - Paradigm Comparison: What are the respective trade-offs between generative models and programmatic frameworks? Detailed experimental setup are provided in Appendix A.2.

### 5.1 Quantitative Results

- Table 2 presents the comprehensive benchmarking results across our proposed taxonomy. The quantitative analysis reveals several key insights:

- Table 2: Overall results on SciGenBench. We report the inverse validation rate (Rinv) and the LMMas-Judge scores. Standard-metrics are computed on the real-image SeePhys subset.

LMM-as-Judge (0–2) ↑ Standard-metrics

Model Rinv (%) ↑

C&F L&P R&O SP E&R PSNR ↑ SSIM ↑ CLIP ↑ FID ↓ Open-source T2I Models

HunyuanImage-3.0 30.79 0.39 0.78 1.44 0.56 0.81 12.21 0.82 25.01 93.27 Qwen-Image 38.86 0.24 0.70 1.48 0.30 0.76 9.63 0.78 25.02 120.42

###### Closed-source T2I Models

GPT-Image-1 42.97 0.57 1.37 1.90 0.84 1.19 13.07 0.84 25.14 77.31 Seedream-4.0 52.67 0.44 0.94 1.67 0.55 0.95 10.65 0.74 25.02 98.22 Nanobanana 57.75 0.43 0.92 1.60 0.60 1.15 14.12 0.85 25.13 104.70 Flux2-flex 58.83 0.48 1.06 1.70 0.67 1.20 14.11 0.85 25.10 96.74 GPT-Image-1.5 63.52 0.98 1.70 1.97 1.17 1.62 14.79 0.88 25.16 112.52 Nanobanana-Pro 73.41 1.59 1.87 1.98 1.72 1.93 12.02 0.81 25.01 87.72

###### ImgCoder

Qwen3-ImgCoder 56.38 1.21 1.30 1.62 1.39 1.29 14.71 0.86 25.21 121.55 Gemini-3-Flash-ImgCoder 76.93 1.80 1.88 1.88 1.92 1.91 14.63 0.85 25.18 117.83 Gemini-3-Pro-ImgCoder 77.87 1.82 1.93 1.91 1.93 1.90 14.59 0.86 25.16 107.67

Closed- vs. Open-source Performance Gap. Closed pixel-based models consistently outperform open-source T2I systems across both inverse validation rate and LMM-as-Judge evaluations, benefiting from large-scale proprietary data and optimized training pipelines. For instance, Nanobanana-Pro achieves a Rinv of 73.41% with strong Judge scores on C&F (1.59) and L&P (1.87), whereas open-source models such as HunyuanImage-3.0 and Qwen-Image remain below 40% Rinv and score under 0.8 on these structure-sensitive dimensions. This persistent gap indicates that scaling model size alone is insufficient for scientific diagram generation without stronger inductive biases or explicit constraint mechanisms.

###### Precision Expressiveness

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Nanobanana-Pro Gemini-Flash-ImgCoder Nanobanana-Pro Gemini-Flash-ImgCoder

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Webinstruct-84257 Webinstruct-1478564

- Figure 2: Precision vs. Expressiveness Trade-off. Left (a): When plotting the function y = x ln x, pixel-based models produce visually smooth but mathematically inaccurate plots, while code-based methods ensure exactness via execution. Right (b): Conversely, for physical scenarios like a spring system, pixel-based models offer richer visual expressiveness, whereas code-based outputs remain schematic.

Advantages of ImgCoder. Interestingly, ImgCoder models consistently achieve the strong inverse validation rates and LMM-as-Judge performance, particularly on structure- and reasoning-sensitive dimensions such as C&F, L&P, and SP. Notably, Gemini-3-Pro-ImgCoder attains the top Rinv (77.87%), surpassing all pixel-based models. Besides, Qwen3-ImgCoder achieves a large improvement over Qwen-Image (56.38% vs. 38.86% Rinv) despite being fully open-source, confirming that code-based execution—rather than closed data—drives the observed gains.

Disparity between Evaluation Metrics. We observe a clear divergence between perceptual standard

|Compositional error|Rendering error|Domain knowledge error|Structural error|Dense data error|
|---|---|---|---|---|
|[Figure 78]<br><br>WebInstruct-154761, Nanobanana-Pro|WebInstruct-1985970, HunyuanImage-3.0<br><br>[Figure 79]<br><br>|[Figure 80]<br><br>WebInstruct-182109, GPT-Image-1.5|[Figure 81]<br><br>WebInstruct-531926, Seedream-4.0|[Figure 82]<br><br>WebInstruct-1680377, Nanobanana|
|Question:A capacitor of 3 Farads, 6 Farads, 11 Farads, and 17 Farads are connected in series to a connected of 33 volt battery… Determine the difference in stored energy between them: 𝑈most − 𝑈least in Joules.|Question:Suppose you burn 0.300 g of C(graphite) in excess of 𝑂2 𝑔 in a constant volume calorimeter to give 𝐶𝑂2 𝑔 …The temperature of the calorimeter, which contains 775 g of water, …. Calculate ∆𝑈 per mole of carbon.<br><br>|Question:What is the molecular structure for BrFI2?|Question:A side-angle-side triangle has<br><br>sides 9 and 1 and an angle 11𝜋90 . Determine the centroid, circumcenter, incenter, orthocenter, nine-point center, and symmedian point of the triangle.<br><br>|Question:Let A be an 8×8 square matrix<br><br>with 𝑎12 = 𝑎24 = 𝑎33 = 𝑎41 = 𝑎58 = 𝑎65 = 𝑎77 = 𝑎86 = 1 and all other entries 0. Then 𝐴𝑛 = 𝐼 for some n; find n.|
| | | | | |
| | |Error:Question asks for the molecular<br><br>structure of BrFI𝟐 (1 Bromine, 1 Fluorine, 2 Iodines). The model hallucinated a structure<br><br>with an extraFluorine(Br𝐅𝟐I𝟐).| | |
| | | | | |
| | | | |Error:Model failed to render the specific sparse matrix where only selectedentries (like 𝑎12,𝑎24) are 1 and the rest are 0. Instead, it generated a dense matrix with random integers.<br><br>|
|Error:Question clearly specified 4 series capacitors while image incorrectly depicts 5.<br><br>Severity: HunyuanImage-3.0 (Medium), Others (Minor).| | | | |
| | | |Error:Question described a flat obtuse<br><br>triangle while model drew an equilateral<br><br>triangle with wrong point labels. Severity: ImgCoders (Minor), NanobananaPro (Medium), Others (Severe)| |
| |Error:Scientific labels and technical descriptions are rendered as unreadable glitched text and indecipherablecharacters.<br><br>Severity: Qwen-Image (Medium), HunyuanImage-3.0 (Severe), Others (Minor)| | | |
| | |Severity: ImgCoders, GPT-Image-1.5, Nanobanana-Pro (Minor), Others (Severe)| | |
| | | | |Severity: ImgCoders (Minor), NanobananaPro (Medium), Others (Severe)|

- Figure 3: Qualitative Error Taxonomy. We categorize failures into five modes ranging from low-level visual artifacts to high-level semantic hallucinations. The specific errors are annotated in red.

metrics and reasoning-oriented evaluations. Models with competitive PSNR or FID scores do not necessarily achieve high inverse validation rates or judge scores. A closer inspection of judge dimensions reveals that perceptual fidelity primarily correlates with E&R, while exhibiting weak alignment with C&F and L&P, which require strict logical consistency and structural correctness. This discrepancy underscores the limitation of previous pixel-level metrics in evaluating scientific diagrams, where visual similarity can mask subtle yet critical factual/relational errors.

Spiral Co-evolution Hypothesis. We hypothesize a complementary co-evolution between codebased and pixel-based paradigms. On one hand, code-based methods enforce structural and logical validity, and as LMMs improve, such structured reasoning can transfer to pixel-based T2I models through shared backbones or by using code-rendered synthetic images as training data. Empirically, Nanobanana-Pro (Gemini-3-Pro-Image) and Gemini-3-ImgCoder, which share the Gemini-3 backbone, exhibit highly similar diagram construction strategies across many samples (Figure 2, Appendix D), supporting this transfer. On the other hand, pixel-based models contribute visually diverse and expressive imagery that enriches multimodal training data, which in turn benefits code-based reasoning and LMMs learning, forming a mutually reinforcing spiral between reasoning and synthesis.

### 5.2 Qualitative Analysis

Error Taxonomy. As illustrated in Figure 3, we categorize observed failures into five primary classes based on a systematic audit. (1) Compositional error: Misalignment among multiple visual elements, including attribute leakage (e.g., incorrect label binding), spatial relation confusion, or incorrect object counts. (2) Rendering error: Low-level fidelity issues such as illegible text glyphs, blurred lines, or visually corrupted labels. (3) Structural error: Violations of geometric logic or topological integrity, including distorted shapes, non-closed curves, or logically inconsistent intersections (e.g., broken parallelism). (4) Dense data error: Failures in high-information-density scenarios (e.g., tables or matrices), manifested as coordinate drift, axis misalignment, or collapsed rows and columns. (5) Domain knowledge error: Scientific hallucinations where images appear visually plausible but violate domain-specific laws (e.g., incorrect electron configurations or optical paths).

We observe a hierarchy in failure modes. Traditional text-to-image errors (compositional, rendering) are largely resolved in SOTA models, persisting only in weaker baselines. Domain knowledge error serves as a watershed: while open-source models struggle, top-tier closed-source models have significantly reduced scientific hallucinations. However, structural and dense data errors remain the most persistent bottlenecks. These tasks demand high-precision rendering and information density that challenge the probabilistic nature of diffusion models, causing even Nanobanana-Pro to falter. Notably, ImgCoder outperforms pixel-based SOTAs in these high-precision regimes due to its rigorous code-driven

grounding.

Precision-Expressiveness Trade-off. In Figure 2, the two paradigms exhibit distinct trade-offs. While code-based methods may fall slightly short in visual expressiveness—often producing schematic or “flat" diagrams compared to the rich renderings of Nanobanana-Pro (e.g., the textured spring and hand illustration)—they offer an intrinsic advantage in precision. Pixel-based models, constrained by their probabilistic generation nature, fundamentally struggle to guarantee strict mathematical accuracy. This is evident in the function plotting task (y = x ln x), where Nanobanana-Pro produces a plausiblelooking curve but fails to capture the correct intercepts and extrema. In contrast, code-based methods leverage deterministic execution engines to ensure the rigorous alignment of geometric shapes and data coordinates, effectively solving the "hallucination" problem in quantitative visualization.

### 5.3 Distributional Gap with Real Images

FID Discrepancy. As shown in Table 2, even top-tier models that avoid overt factual errors still differ subtly from real-image distributions. Models such as GPT-Image-1 and Nanobanana-Pro achieve comparatively low FID scores (70.73 and 79.89), due to a concise “textbook-style” visual appearance (see Appendix D).

Representation Gap. We further visualize CLIP embeddings using t-SNE. As shown in Figure 4a, images generated from identical prompts form a distinct, linearly separable cluster (red) from real images (black), indicating a persistent domain gap in visual style despite semantic alignment.

Spectral Bias. Spectral analysis (Figure 4b) shows that while low-frequency components align well, Nanobanana-Pro exhibits consistently higher high-frequency energy than real images, with the magnitude difference ∆ peaking in the high-frequency range. This indicates artificial “digital sharpness” devoid of natural spectral decay, distinguishing synthetic outputs from real scans. More discussion are provided in Appendix B.1.

| |Real (S|eePhys GT)| | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| |NanoBa<br><br>Magn<br><br>|nana-Pro itude| | | | | |
| | | | | | | | |
| | | |High-Freq (Digital S|Artifacts harpness)| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

30

Real (SeePhys GT)

350

MagnitudeDifference(dB)

NanoBanana-Pro

25

LogMagnitude(dB)

300

20

250

15

10

200

5

150

0

0 50 100 150 200 250 Frequency (Cycles/Image)

Figure 4: Analysis of Distributional Gaps. Left: CLIP t-SNE shows clear separation between NanoBanana-Pro and real images. Right: Spectral analysis attributes this gap to excessive high-frequency energy in generated images.

### 5.4 Domain-specific Breakdown

- As shown in Table 3, We analyze performance across domains to understand how generation paradigms interact with domain-specific requirements. Mathematics, physics, and universal diagram types demand strict geometric, symbolic, and layout accuracy, where code-based ImgCoder models consistently outperform pixel-based approaches. In contrast, biology and visually rich chemistry subdomains favor pixel-based models, benefiting from organic shapes and dense textures. Chemistry represents a boundary case: molecular structures favor code-based reasoning, while crystal and reaction diagrams are better handled by pixel-based models. Overall, domain performance depends more on struc-

- Table 3: Breakdown of Performance by Subject. We report the inverse validation rate (Rinv) and the LMM-as-Judge mean score (Judge) for each discipline. The last row reports the average performance across all baselines.

Model Math Physics Chemistry Biology Universal

Rinv (%) Judge Rinv (%) Judge Rinv (%) Judge Rinv (%) Judge Rinv (%) Judge Open-source T2I Models

HunyuanImage-3.0 13.70 0.56 25.39 0.94 22.92 0.86 33.33 1.06 12.58 0.68 Qwen-Image 30.14 0.57 28.12 0.85 40.28 0.66 43.14 0.95 15.23 0.62

###### Closed-source T2I Models

GPT-Image-1 31.51 1.07 32.42 1.26 38.19 1.20 49.02 1.38 29.14 1.09 Flux2-flex 41.78 0.83 53.52 1.18 60.42 1.06 58.82 1.17 36.42 0.86 Seedream-4.0 42.47 0.78 51.56 0.98 46.53 1.11 41.18 1.12 37.09 0.82 Nanobanana 43.84 0.78 51.17 1.09 51.39 1.00 66.67 1.20 31.13 0.77 GPT-Image-1.5 52.05 1.43 60.16 1.61 56.94 1.48 72.55 1.58 47.68 1.41 Nanobanana-Pro 64.38 1.80 62.50 1.84 54.86 1.80 74.51 1.89 60.93 1.82

###### ImgCoder

Qwen3-ImgCoder 44.52 1.54 43.36 1.39 52.08 1.14 41.18 0.96 43.05 1.34 Gemini-3-Flash-ImgCoder 66.44 1.89 70.31 1.91 63.19 1.76 80.39 1.82 71.52 1.89 Gemini-3-Pro-ImgCoder 69.86 1.94 75.39 1.93 59.03 1.76 76.47 1.84 72.85 1.91

Average 45.52 1.20 50.36 1.36 49.62 1.26 57.93 1.36 41.60 1.20

tural constraints and information density than on model scale, highlighting the potential of hybrid approaches for scientific image synthesis. Further analyses are provided in Appendix B.2.

###### Training Reward Curves

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | |a|n|o|B|an|a|n|a-|P|ro| |fi|lte|r|ed|)| | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |w|e|n|-I|m|a|g|e|(f|ilt|e|re|d|)| | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |e|m|i|ni|-I|m|g|C|od|e|r| | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |a|n|o|B|an|a|n|a-|P|ro| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |w|e|n|-I|m|g|C|od|e|r| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | |w|e|n|-I|m|a|g|e| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

0.8

0.7

0.6

Reward

0.5

0.4

0.3

0.2

0 25 50 75 100 125 150 175 200

Training Steps

###### Validation Accuracy

48

46

Accuracy(Avg@4)

44

42

40

NanoBanana-Pro

NanoBanana-Pro (Filtered)

0 25 50 75 100 125 150 175 200

Global Step

###### Scaling Law on MV

46.5

46.1

46.0

###### Accuracy(Avg@4)

45.4

45.5

45.0

44.6

44.5

43.9

44.0

###### NanoBanana-Pro

43.5

50 200 500 1400

Training Samples

- Figure 5: Downstream Data Utility. Left (a): Stronger teachers yield higher training rewards. Middle (b): Filtered data consistently outperforms unfiltered data. Right (c): Performance scales predictably with data size.
- 6 Data Utility for Downstream Reasoning

While Section 5 assessed intrinsic visual quality, this section evaluate the extrinsic utility of generated images as training data for LMMs. Specifically, we address RQ3 - Data Utility: Whether synthetic scientific images can improve downstream multimodal reasoning via fine-tuning? Implementation details are in Appendix A.3.

### 6.1 Effectiveness of Synthetic Image Training

- As shown in Table 4, training with synthetic images yields substantial performance improvements over the baseline across all model variants on both GEO3K and MathVision. For instance, the best-

#### Table 4: Downstream Performance Comparison.

###### RL Data GEO3K MV AVG

Base 61.9 39.0 54.5 Qwen-Image 68.2 45.9 57.1 Qwen-Imgcoder 67.9 46.9 57.4 Qwen-Image (Filt) 68.6 47.0 57.8 Nanobanana-Pro 70.1 46.1 58.1 Gemini-ImgCoder 69.1 46.9 58.0 Nanobanana-Pro (Filt) 68.7 47.7 58.2

performing Nanobanana-Pro (Filt) achieves an average score of 58.2, corresponding to a 3.7-point absolute gain over the baseline (54.5). Consistently, Figure 5a shows a steadily increasing reward trend during training, indicating stable and effective optimization dynamics. Notably, ImgCoder variants show a clear advantage on the more challenging and discriminative MV benchmark compared to pixel-based counterparts, indicating that ImgCoder can serve as a low-cost and scalable data synthesis approach with strong structural rigor.

### 6.2 Impact of Image Quality

We further examine the effect of synthetic image quality on downstream performance. Training on higher-quality images leads to better results, with Nanobanana-Pro–based data outperforming Qwen-Image (58.1 vs. 57.1). In addition, applying data filtration consistently improves both training rewards (Figure 5a) and test accuracy (Figure 5b), indicating that downstream performance is sensitive to the quality of synthetic images rather than data quantity alone.

### 6.3 Scaling Law of Synthetic Data

To further investigate scalability, we conduct a controlled data-scaling study using synthetic images generated by NanoBanana-Pro. Training sets ranging from 50 to 1.4K samples are constructed. As shown in Figure 5c, downstream accuracy exhibits a clear and stable increase with data scale, rising from 43.9% to 46.1% (+2.2%). Notably, the performance curve follows a distinct log-linear growth trend without observable saturation, suggesting that high-fidelity synthetic images can continuously serve as effective training signals. These results suggest that high-fidelity synthetic images provide consistently effective supervision signals rather than diminishing returns, and that the well-known “Data Engine” scaling behavior observed in text-domain training extends naturally to multimodal reasoning scenarios.

## 7 Conclusion

In this paper, we investigate scientific image synthesis for multimodal reasoning, revealing a precision–expressiveness trade-off between pixel- and code-based generation. We introduce SciGenBench to systematically evaluate structural correctness and information utility. Experiments show code-based methods like ImgCoder excel in structural precision and reduce failure modes, while pixel-based models offer richer visual expressiveness. Moreover, high-fidelity synthetic images consistently improve downstream multimodal reasoning performance, underscoring the value of scalable, logic-grounded data synthesis for advancing intelligence in LMMs.

## References

- [1] Shuai Bai et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/ abs/2511.21631.
- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pp. arXiv–2506, 2025.
- [3] Jonas Belouadi, Anne Lauscher, and Steffen Eger. Automatikz: Text-guided synthesis of scientific vector graphics with tikz. arXiv preprint arXiv:2310.00367, 2023.
- [4] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weijie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Linus, Lucaz Liu, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyang Peng, Yuanbo Peng, Xiangwei Shen, Yixuan Shi, Jiale Tao, Yangyu Tao, Qi Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Lucas Wang, Qixun Wang, Weiyan Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fang Yang, Miles Yang, Xiaofeng Yang, Xuan Yang, Zhantao Yang, Jingmiao Yu, Zheng Yuan, Chao Zhang, Jian-Wei Zhang, Peizhen Zhang, Shi-Xue Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zijian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jianchen Zhu, and Zhao Zhong. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.
- [5] Kaijie Chen, Zihao Lin, Zhiyang Xu, Ying Shen, Yuguang Yao, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. R2i-bench: Benchmarking reasoning-driven text-to-image generation. arXiv preprint arXiv:2505.23493, 2025.
- [6] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [7] Zhaorun Chen, Yichao Du, Zichen Wen, Yiyang Zhou, Chenhang Cui, Zhenzhen Weng, Haoqin Tu, Chaoqi Wang, Zhengwei Tong, Qinglan Huang, et al. Mj-bench: Is your multimodal reward model really a good judge for text-to-image generation? arXiv preprint arXiv:2407.04842, 2024.
- [8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, pp. 12606–12633. PMLR, 2024.
- [10] Run-Ze Fan, Zengzhi Wang, and Pengfei Liu. Megascience: Pushing the frontiers of post-training datasets for science reasoning. arXiv preprint arXiv:2507.16812, 2025.
- [11] Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025.
- [12] Google. Gemini 3 Model Card. https://deepmind.google/models/gemini, 2025. Accessed: 2025-12-28.
- [13] Google. Introducing nano banana pro: Turn your visions into studio-quality designs with unprecedented control, improved text rendering and enhanced world knowledge., 2025. URL https://blog.google/ technology/ai/nano-banana-pro/.
- [14] Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel, Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar, Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill, Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran

- Sathiamoorthy, Alexandros G. Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.
- [15] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20406–20417, 2023.
- [16] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.
- [17] Yiming Huang, Xiao Liu, Yeyun Gong, Zhibin Gou, Yelong Shen, Nan Duan, and Weizhu Chen. Key-pointdriven data synthesis with its enhancement on mathematical reasoning. arXiv preprint arXiv:2403.02333, 2024.
- [18] Hugging Face. Open r1: A fully open reproduction of deepseek-r1. https://github.com/huggingface/ open-r1, 2025. GitHub repository.
- [19] Black Forest Labs. FLUX.2: Frontier Visual Intelligence, 2025.
- [20] Shudong Liu, Hongwei Liu, Junnan Liu, Linchen Xiao, Songyang Gao, Chengqi Lyu, Yuzhe Gu, Wenwei Zhang, Derek F. Wong, Songyang Zhang, and Kai Chen. CompassVerifier: A Unified and Robust Verifier for Large Language Models. arXiv preprint arXiv:2508.03686, 2025.
- [21] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Intergps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021), 2021.
- [22] Xueguang Ma, Qian Liu, Dongfu Jiang, Ge Zhang, Zejun MA, and Wenhu Chen. General-Reasoner: Advancing LLM reasoning across all domains. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=pBFVoll8Xa.
- [23] Yuansheng Ni, Ping Nie, Kai Zou, Xiang Yue, and Wenhu Chen. Viscoder: Fine-tuning llms for executable python visualization code generation. arXiv preprint arXiv:2506.03930, 2025.
- [24] OpenAI. Gpt-5 system card. Technical report, OpenAI, 2025. URL https://cdn.openai.com/ gpt-5-system-card.pdf.
- [25] OpenAI. The new chatgpt images is here, 2025. URL https://openai.com/index/ new-chatgpt-images-is-here/.
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.
- [27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024.
- [28] Team Seedream, :, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, Xiaowen Jian, Huafeng Kuang, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, Wei Liu, Yanzuo Lu, Zhengxiong Luo, Tongtong Ou, Guang Shi, Yichun Shi, Shiqi Sun, Yu Tian, Zhi Tian, Peng Wang, Rui Wang, Xun Wang, Ye Wang, Guofeng Wu, Jie Wu, Wenxu Wu, Yonghui Wu, Xin Xia, Xuefeng Xiao, Shuang Xu, Xin Yan, Ceyuan Yang, Jianchao Yang, Zhonghua Zhai, Chenlin Zhang, Heng Zhang, Qi Zhang, Xinyu Zhang, Yuwei Zhang, Shijia Zhao, Wenliang Zhao, and Wenjia Zhu. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.
- [29] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [30] Kaiyue Sun, Rongyao Fang, Chengqi Duan, Xian Liu, and Xihui Liu. T2i-reasonbench: Benchmarking reasoning-informed text-to-image generation. arXiv preprint arXiv:2508.17472, 2025.

- [31] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.
- [32] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.
- [33] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [34] Kun Xiang, Heng Li, Terry Jingchen Zhang, Yinya Huang, Zirong Liu, Peixin Qu, Jixi He, Jiaqi Chen, Yu-Jie Yuan, Jianhua Han, et al. Seephys: Does seeing help thinking?–benchmarking vision-based physics reasoning. arXiv preprint arXiv:2505.19099, 2025.
- [35] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [36] Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris CallisonBurch, Ranjay Krishna, Aniruddha Kembhavi, et al. Scaling text-rich image understanding via code-guided synthetic multimodal data generation. arXiv preprint arXiv:2502.14846, 2025.
- [37] Yuwei Yang, Zeyu Zhang, Yunzhong Hou, Zhuowan Li, Gaowen Liu, Ali Payani, Yuan-Sen Ting, and Liang Zheng. Effective training data synthesis for improving mllm chart understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2653–2663, 2025.
- [38] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.
- [39] Yanzhe Zhang et al. Multimodal reasoning with large language models: A survey. arXiv preprint arXiv:2505.04921, 2025. URL https://arxiv.org/abs/2505.04921.
- [40] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.
- [41] Le Zhuo, Songhao Han, Yuandong Pu, Boxiang Qiu, Sayak Paul, Yue Liao, Yihao Liu, Jie Shao, Xi Chen, Si Liu, et al. Factuality matters: When image generation and editing meet structured visuals. arXiv preprint arXiv:2510.05091, 2025.

## A Experimental Details

- A.1 Baseline Models Details We provide the detailed citations for the baseline models evaluated in Table 2 below.

Open-source T2I Models. We evaluate representative open-source models including HunyuanImage-3.0 [4] and Qwen-Image [33].

Closed-source T2I Models. For proprietary models, we include GPT-Image-1 and its upgraded version GPT-Image-1.5 [25]. We also evaluate Seedream-4.0 [28], Flux2-flex [19], as well as Nanobanana and Nanobanana-Pro [13].

ImgCoder Backbones. Our ImgCoder framework is evaluated using different LLM backbones, specifically Qwen3 [35], Gemini-3-Flash, and Gemini-3-Pro [12].

- A.2 Generative Model Settings

All model inferences are conducted using default settings. For closed-source T2I models, we utilize their official APIs. Open-source T2I models are deployed locally on NVIDIA A100 GPUs and implement the official prompt rewriting strategy to ensure optimal performance.

For the ImgCoder framework, we set the sampling temperature to 0.6. To ensure robustness against syntactic instability, we implement an error-recovery mechanism that allows up to three retries in cases of code extraction failure or compilation errors. If execution remains unsuccessful after these attempts, a blank image is returned as a fallback to maintain pipeline continuity.

- A.3 Downstream Training Configuration

We utilize four synthetic datasets in Section 5(Qwen-Image, Nanobanana-Pro, Qwen-Imgcoder, Gemini-Flash-ImgCoder) to perform reinforcement learning fine-tuning on the Qwen3-VL-8B-Instruct model. Additionally, we filter out incorrect images from the raw datasets to construct high-quality subsets, specifically Nanobanana-Pro (Filt) and Qwen-Image (Filt).

For the training hyperparameters, the model is trained for 200 global steps with a batch size of 128. We perform 8 rollouts per prompt and set the maximum response length to 8192 tokens. All training and evaluation processes are implemented based on the VeRL [29] library.

For evaluation, we select the MathVisionmini [31] and Geometry3Ktest [21] benchmarks. Following the official recommendation for Qwen models, the sampling temperature is set to 0.6. Given the limited

sample size of MathVisionmini, we report the average score over 4 samples (AVG@4) to ensure result stability. To address the challenge of rule-based answer matching in scientific domains, we employ the Compass-Verifier-8B [20] as the judge model for both reward computation during training and accuracy assessment during evaluation.

## B Additional Results

### B.1 Distributional Gap with Real Images

To validate whether the distributional discrepancy observed in Section 5.3 is specific to Nanobanana Pro or a broader issue in scientific image generation, we extended our spectral and representation analysis to include all evaluated models.

Universal Representation Gap. As shown in Figure 6a, we visualize the CLIP embedding space for all 11 evaluated models against the real SeePhys dataset. The t-SNE projection reveals a striking pattern:

###### Multi-Model Feature Distribution Analysis via CLIP Embeddings

60

40

20

0

20

40

Real (SeePhys GT)

GPT-Image-1.5

NanoBanana-Pro

Seedream-4.0

NanoBanana

Qwen-Image

60

HunyuanImage-3.0

Gemini-3-Pro-ImgCoder

Flux2-flex

Gemini-3-Flash-ImgCoder

GPT-Image-1

Qwen3-ImgCoder

60 40 20 0 20 40 60

(a) Feature distribution analysis (t-SNE).

###### Comparative Spectral Analysis Across Different Models

Real (SeePhys GT)

GPT-Image-1.5

350

NanoBanana-Pro

Seedream-4.0

NanoBanana

Qwen-Image

HunyuanImage-3.0

Gemini-3-Pro-ImgCoder

300

LogMagnitude(Energy)

Flux2-flex

Gemini-3-Flash-ImgCoder

GPT-Image-1

Qwen3-ImgCoder

250

200

150

0 50 100 150 200 250

Frequency (Low -> High)

(b) Spectral frequency analysis.

- Figure 6: Systemic domain gap analysis. (a) Visualization of CLIP embeddings shows that real scientific images (black cluster) occupy a distinct region from synthetic ones. (b) Spectral analysis reveals that real images (black curve) have significantly lower high-frequency energy compared to the "digital sharpness" of synthetic models.

the real scientific images (black points) form a tightly clustered manifold that is distinct from almost all synthetic distributions. The majority of high-fidelity models cluster away from the ground truth. This suggests that current generative paradigms have converged on a "digital scientific style" that is internally consistent but visually distinct from the "natural" distribution of real-world scientific literature.

Consistent Spectral Bias. Figure 6b presents the frequency domain analysis across all models. The results confirm that the high-frequency energy divergence is a universal phenomenon. Every single generated model (colored curves) exhibits higher log-magnitude energy in the high-frequency spectrum compared to the real images (black curve). This consistent "spectral gap" indicates that synthetic images lack the natural degradation processes—such as printing imperfections, scanning noise, and paper texture—that characterize real-world scientific data.

### B.2 Domain-specific Breakdown

- Table 3, Figure 7 and Figure 8, reveal clear domain-dependent performance patterns across generation paradigms. Overall, code-based methods consistently dominate in structure-intensive domains, while pixel-based models show relative advantages in visually expressive or loosely constrained scenarios.

Mathematics and Physics. In highly structured domains such as mathematics and physics, codebased ImgCoder models significantly outperform pixel-based counterparts. As shown in Table 3, Gemini-3-Pro-ImgCoder achieves the highest inverse validation rates in Math (69.86%) and Physics (75.39%), alongside top LMM-as-Judge scores (1.94 and 1.93, respectively). Fine-grained results in Figures 6 and 7 further indicate consistent gains across subcategories involving geometric constructions, field diagrams, optical rays, and analytical plots. These tasks demand strict adherence to geometric constraints, coordinate consistency, and topological correctness—properties naturally enforced by executable code.

Chemistry. Chemical domains exhibit a mixed pattern. For abstract and symbol-heavy tasks such as molecular structures and electron configurations, code-based models remain competitive and often superior in Judge scores (Figure 7). However, pixel-based models (e.g., Flux2-flex, Nanobanana-Pro)

|55.0| |60.0| |26.1| |42.9| |23.1| |25.6| |33.3| |22.7| |66.7| |0.0| |16.3| |32.4| |23.2| |28.8| |61.0| |56.4| |46.2| |40.6| |45.2| |33.3| |35.7| |64.3| |17.3| |34.8| |29.0| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|70.0| |60.0| |34.8| |50.0| |53.8| |65.1| |44.4| |40.9| |59.5| |66.7| |58.1| |52.9| |47.8| |54.5| |48.8| |59.0| |53.8| |51.6| |42.6| |55.6| |57.1| |64.3| |30.8| |47.8| |26.1| |
|75.0| |50.0| |47.8| |50.0| |61.5| |41.9| |44.4| |45.5| |66.7| |66.7| |41.9| |45.6| |55.1| |51.5| |48.8| |38.5| |61.5| |60.9| |56.5| |59.3| |48.2| |71.4| |25.0| |47.8| |44.9| |
|40.0| |70.0| |52.2| |57.1| |61.5| |65.1| |69.8| |68.2| |54.8| |0.0| |48.8| |64.7| |59.4| |62.1| |78.0| |64.1| |80.0| |84.4| |75.8| |66.7| |73.2| |64.3| |46.2| |72.5| |60.9| |
|70.0| |80.0| |56.5| |57.1| |92.3| |53.5| |30.2| |63.6| |81.0| |100.0| |60.5| |58.8| |52.2| |54.5| |73.2| |61.5| |70.8| |56.2| |74.2| |81.5| |69.6| |100.0| |32.7| |68.1| |46.4| |
|75.0| |50.0| |73.9| |71.4| |69.2| |67.4| |49.2| |72.7| |92.9| |33.3| |58.1| |69.1| |69.6<br><br>63.8| |63.6| |73.2| |66.7| |70.8| |73.4| |79.0| |92.6| |69.6| |78.6| |32.7| |75.4| |52.2| |
|85.0| |50.0| |73.9| |78.6| |84.6| |72.1| |46.0| |81.8| |92.9| |100.0| |60.5| |69.1| | | |59.1| |73.2| |76.9| |67.7| |75.0| |82.3| |85.2| |62.5| |85.7| |28.8| |72.5| |52.2| |
|90.0| |70.0| |65.2| |78.6| |84.6| |60.5| |54.0| |63.6| |88.1| |100.0| |74.4| |64.7| |66.7| |66.7| |78.0| |76.9| |81.5| |73.4| |83.9| |85.2| |73.2| |85.7| |44.2| |71.0| |60.9| |
|90.0| |90.0| |82.6| |71.4| |92.3| |74.4| |66.7| |72.7| |88.1| |66.7| |79.1| |86.8| |78.3| |78.8| |90.2| |87.2| |96.9| |93.8| |91.9| |85.2| |89.3| |100.0| |65.4| |87.0| |84.1| |
|85.0| |100.0| |95.7| |71.4| |84.6| |69.8| |69.8| |72.7| |92.9| |66.7| |83.7| |86.8| |78.3| |84.8| |92.7| |84.6| |90.8| |96.9| |88.7| |81.5| |85.7| |100.0| |65.4| |87.0| |81.2| |
|90.0| |90.0| |91.3| |78.6| |100.0| |79.1| |68.3| |77.3| |90.5| |66.7| |83.7| |85.3| |84.1| |75.8| |95.1| |79.5| |87.7| |87.5| |90.3| |88.9| |91.1| |92.9| |61.5| |89.9| |82.6| |
|75.0| |70.0| |63.6| |64.3| |73.4| |61.3| |52.4| |62.0| |79.4| |60.6| |60.5| |65.1| |61.7| |61.8| |73.8| |68.3| |73.4| |72.2| |73.7| |74.1| |68.7| |82.5| |40.9| |68.5| |56.4| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

HunyuanImage-3.0

Qwen-Image

GPT-Image-1

Qwen3-ImgCoder

Seedream-4.0

Nanobanana

Flux2-Flex

GPT-Image-1.5

Gemini3-P-ImgCoder

Gemini3-F-ImgCoder

Nanobanana-Pro

Average

Biology-CellDiagramBiology-EcologicalBiology-GeneticsBiology-MolecularProcessChemistry-CrystalStructureChemistry-ElectronConfigChemistry-MolecularStructureChemistry-Orbital/QuantumChemistry-ReactionSchemeChemistry-SpectraMath-AnalyticGeometryMath-PlaneGeometricMath-Set&ProbabilityMath-SolidGeometricPhysics-AstronomicalPhysics-CircuitPhysics-FieldDiagramPhysics-MechanicalPhysics-OpticalRayPhysics-ThermodynamicPhysics-WaveformUniversal-Exp.SetupUniversal-Graph&FlowUniversal-Plot&ChartUniversal-Table/Grid

100

[Figure 83]

80

60

40

20

0

- Figure 7: Perfect Image Rate by Domain. The heatmap shows the percentage of strictly correct images (passing all visual validation quizzes) generated by each model across fine-grained scientific sub-categories in the SciGen dataset.

|1.16| |1.02| |0.64| |1.06| |0.88| |0.53| |0.49| |0.74| |0.73| |0.33| |0.47| |0.53| |0.54| |0.63| |0.91| |0.92| |0.66| |0.93| |0.73| |0.92| |0.66| |0.94| |0.61| |0.48| |0.56| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1.25| |1.05| |0.65| |1.24| |0.92| |0.54| |0.71| |0.72| |1.13| |0.67| |0.40| |0.66| |0.54| |0.46| |1.19| |0.94| |0.66| |1.07| |1.00| |1.01| |0.69| |1.29| |0.68| |0.44| |0.61| |
|1.36| |1.12| |0.71| |1.11| |1.51| |0.84| |0.73| |1.15| |1.22| |1.07| |0.54| |0.66| |0.84| |0.85| |1.18| |1.02| |0.65| |0.90| |0.78| |1.18| |0.96| |1.56| |0.73| |0.59| |0.78| |
|1.38| |1.28| |0.70| |1.37| |1.32| |0.65| |0.73| |1.04| |1.18| |0.67| |0.58| |0.86| |0.87| |0.70| |1.18| |1.07| |0.84| |1.22| |1.10| |1.38| |0.85| |1.53| |0.67| |0.60| |0.68| |
|1.44| |1.00| |0.89| |1.21| |1.17| |0.83| |0.71| |1.01| |1.47| |0.53| |0.65| |0.94| |0.81| |0.78| |1.23| |1.16| |0.85| |1.28| |1.05| |1.38| |1.01| |1.47| |0.85| |0.62| |0.77| |
|1.71| |1.36| |1.00| |1.40| |1.42| |0.92| |0.96| |1.43| |1.41| |1.07| |0.86| |1.04| |1.09<br><br>1.39| |1.10| |1.24| |1.22| |1.06| |1.33| |1.18| |1.46| |1.09| |1.40| |1.02| |0.95| |1.03| |
|0.87| |0.64| |1.18| |0.87| |1.46| |1.20| |0.90| |1.23| |1.20| |0.60| |1.64| |1.61| | | |1.50| |1.60| |0.93| |1.46| |1.48| |1.44| |1.02| |1.51| |0.77| |1.08| |1.49| |1.58| |
|1.87| |1.62| |1.23| |1.70| |1.71| |1.11| |1.08| |1.70| |1.84| |1.40| |1.38| |1.39| |1.41| |1.48| |1.70| |1.57| |1.42| |1.65| |1.58| |1.67| |1.41| |1.99| |1.27| |1.26| |1.40| |
|1.96| |1.86| |1.88| |1.97| |1.83| |1.74| |1.68| |1.85| |1.88| |1.93| |1.82| |1.75| |1.85| |1.79| |1.83| |1.76| |1.68| |1.84| |1.86| |1.91| |1.84| |1.91| |1.77| |1.77| |1.82| |
|1.97| |1.78| |1.75| |1.89| |1.83| |1.74| |1.61| |1.91| |1.82| |1.87| |1.90| |1.93| |1.83| |1.88| |1.98| |1.91| |1.88| |1.87| |1.86| |1.87| |1.94| |1.84| |1.83| |1.93| |1.89| |
|1.97| |1.74| |1.80| |1.97| |1.89| |1.56| |1.54| |1.95| |1.91| |2.00| |1.99| |1.91| |1.91| |1.94| |1.93| |1.87| |1.90| |1.88| |1.94| |1.98| |1.94| |1.96| |1.86| |1.92| |1.91| |
|1.54| |1.32| |1.13| |1.44| |1.45| |1.06| |1.01| |1.34| |1.44| |1.10| |1.11| |1.21| |1.19| |1.19| |1.45| |1.31| |1.19| |1.41| |1.32| |1.44| |1.26| |1.51| |1.13| |1.10| |1.18| |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

Biology-CellDiagramBiology-EcologicalBiology-GeneticsBiology-MolecularProcessChemistry-CrystalStructureChemistry-ElectronConfigChemistry-MolecularStructureChemistry-Orbital/QuantumChemistry-ReactionSchemeChemistry-SpectraMath-AnalyticGeometryMath-PlaneGeometricMath-Set&ProbabilityMath-SolidGeometricPhysics-AstronomicalPhysics-CircuitPhysics-FieldDiagramPhysics-MechanicalPhysics-OpticalRayPhysics-ThermodynamicPhysics-WaveformUniversal-Exp.SetupUniversal-Graph&FlowUniversal-Plot&ChartUniversal-Table/Grid

Qwen-Image

HunyuanImage-3.0

Seedream-4.0

Nanobanana

Flux2-Flex

GPT-Image-1

Qwen3-ImgCoder

GPT-Image-1.5

Nanobanana-Pro

Gemini3-F-ImgCoder

Gemini3-P-ImgCoder

Average

0.00

0.25

0.50

0.75

1.00

1.25

1.50

1.75

2.00

[Figure 84]

- Figure 8: Average LMM-Judge Score by Domain. The heatmap visualizes the mean comprehensive quality score (0–2 scale) awarded by the LMM judge for each model across fine-grained scientific sub-categories.

achieve comparable or slightly higher inverse validation rates in visually rich subdomains such as crystal structures and reaction schematics (Figure 6, Table 3). This suggests that chemistry lies at the boundary between rigid structural reasoning and perceptual pattern recognition.

Biology. In biology, pixel-based models demonstrate stronger performance overall. As shown in Table 3, Nanobanana-Pro attains the highest Biology Rinv (74.51%) and Judge score (1.89), outperforming most code-based variants. Fine-grained heatmaps (Figure 6) indicate that tasks such as cell diagrams and molecular processes benefit from rich textures, organic shapes, and visual priors learned from natural imagery, where strict geometric constraints are less dominant.

Universal Diagram Types. For cross-domain universal diagram categories (e.g., plots, tables, flow charts), ImgCoder regains a clear advantage. Table 3 shows that Gemini-3-Pro-ImgCoder achieves the best Universal performance (72.85%, Judge 1.91). Figures 6 and 7 further highlight ImgCoder’s robustness in tables, grids, and charts, where precise alignment, spacing, and symbolic accuracy are critical and pixel-based models frequently suffer from dense data collapse.

Summary. Taken together, these results indicate that domain performance is primarily governed by the degree of structural constraint and information density. Code-based generation excels in domains requiring explicit computation and strict layout control (Math, Physics, Universal), whereas pixel-based models remain competitive in visually organic domains (Biology) and certain chemistry subfields. This domain-specific complementarity further motivates hybrid and co-evolutionary approaches to scientific image synthesis.

## C Detailed Taxonomy Definitions

To construct a structured and comprehensive benchmark, we devised a hierarchical taxonomy comprising 5 primary subject categories and 25 secondary image types. This taxonomy is designed to decouple disciplinary semantics from visual structures. Below, we provide detailed definitions and examples for each category.

### C.1 Universal

This category encompasses visual structures that are not bound to a specific scientific discipline but are widely used across STEM fields for data representation and logical illustration.

Plot & Chart This category unifies mathematical function graphs and statistical data charts. It includes visualizations defined on 2D/3D coordinate systems, such as continuous function curves (e.g., y = f(x), parabolas, sinusoids) and discrete data visualizations (e.g., scatter plots, histograms, bar charts). We merged these categories due to their shared reliance on coordinate mapping and similar visual features in scientific contexts.

Graph & Flow Visual representations of relationships, hierarchies, or processes consisting of nodes and connecting edges. This includes flowcharts, neural network architectures, decision trees, state transition diagrams, and concept maps.

Table & Grid Structured textual or numerical data organized in rows and columns (tables) or regular grid-based layouts (matrices, game grids).

Exp. Setup Diagrams illustrating generic experimental apparatus or laboratory equipment assemblies that are not strictly limited to a single physics or chemistry domain (e.g., a balance scale, a beaker setup, or a generic measurement device).

### C.2 Mathematics

Plane Geometric 2D Euclidean geometry problems involving shapes like triangles, circles, polygons, and calculations of area, perimeter, or angles.

Solid Geometric 3D geometry problems involving spatial figures such as cubes, spheres, prisms, and cylinders, often focusing on volume or surface area. Analytic Geometry Problems involving geometric shapes defined within a coordinate system, including lines, conic sections (ellipses, hyperbolas), and vector coordinates. Set & Probability Visualizations related to set theory (Venn diagrams) and probability theory (probability trees, distribution areas).

### C.3 Physics

Mechanical Diagrams depicting forces, motion, and equilibrium. Common elements include blocks on inclined planes, pulley systems, levers, and free-body diagrams. Field Diagram Visualizations of invisible physical fields, including electric field lines, magnetic field lines, and vector fields representing forces. Waveform Representations of wave propagation, including simple harmonic motion, sound waves, and light wave interference patterns. Optical Ray Geometric optics diagrams illustrating light paths, reflection (mirrors), refraction (lenses), and image formation. Astronomical Diagrams related to celestial mechanics, orbital paths, solar system models, and astrophysical phenomena. Circuit Electrical schematics showing components like resistors, capacitors, inductors, batteries, and switches arranged in series or parallel. Thermodynamic Visualizations of thermodynamic processes, including P-V diagrams, heat engine cycles (e.g., Carnot cycle), and phase change diagrams.

### C.4 Chemistry

Molecular Structure 2D or 3D representations of chemical molecules, including Lewis structures, ball-and-stick models, and skeletal formulas. Reaction Scheme Diagrams illustrating chemical transformations, including reactants, products, transition states, and reaction mechanisms with electron-pushing arrows.

Electron Config Visualizations of electron distribution, such as orbital box diagrams and shell models. Crystal Structure Illustrations of solid-state lattice structures, unit cells, and atomic packing arrangements.

Spectra Graphical representations of spectral data, such as NMR, IR, or Mass Spectrometry outputs used for chemical analysis. Orbital / Quantum Visualizations of atomic or molecular orbitals (s, p, d, f orbitals) and quantum mechanical probability clouds.

### C.5 Biology

Cell Diagram Illustrations of cellular structures, including organelles (mitochondria, nucleus), cell membranes, and microscopic views of tissues. Ecological Diagrams representing ecosystem relationships, such as food webs, food chains, energy pyramids, and biogeochemical cycles. Genetics Visualizations related to heredity, including Punnett squares, pedigree charts, and DNA replication/transcription illustrations. Molecular Process Diagrams depicting biological mechanisms at the molecular level, such as protein synthesis, enzyme catalysis, and signaling pathways.

## D Qualitative Examples

[Figure 85]

- Figure 9: Qualitative comparison in Math problems.

[Figure 86]

21

- Figure 10: Qualitative comparison in Physics problems.

[Figure 87]

##### Figure 11: Qualitative comparison in Chemistry problems.

[Figure 88]

##### Figure 12: Qualitative comparison in Biology problems.

[Figure 89]

##### Figure 13: Qualitative comparison in Universal problems.

## E Prompts

#### Prompt: STEM Text-to-image Generation

You are a professional STEM problem illustrator. Your task is to generate a clear, precise, and information-complete diagram that visually represents the scientific scenario described in the problem below. The illustration must:

- - Faithfully reflect the problem context and scenario.
- - Explicitly depict all entities, objects, variables, and conditions mentioned in the problem.
- - Encode quantitative or relational information visually when possible (e.g., distances, angles, forces, directions, labels, axes).
- - Help a student understand the setup of the problem, not the solution. The illustration must NOT:
- - Introduce any assumptions, values, or objects not stated or directly implied by the problem.
- - Include solution steps, calculations, or conclusions.
- - Add decorative or artistic elements unrelated to the problem. Use a clean, textbook-style schematic:
- - Neutral colors
- - Clear labels and annotations
- - Simple geometric shapes or standard scientific symbols

**Problem Text:** {question}

Figure 14: The system prompt used for STEM text-to-image generation.

###### Prompt: ImgCoder

You are a multimodal reasoning assistant skilled in scientific visualization using Python. You will be given:

- - **Original Question** (‘question‘): The complete problem text. Your task: First, carefully **PLAN** the diagram (reasoning stage). Then, produce a standalone, runnable Python script using Matplotlib (coding stage).

**Key Objective**: Create a diagram that fully represents the **Initial Setting** of the problem.

- - **Completeness**: Visualize all physical objects, geometric shapes, and **given values** (e.g., lengths, angles, forces, labels) mentioned in the text.
- - **Confidentiality**: Do NOT reveal the final answer, result, or derivation steps. Only show what is *given* before the problem is solved. Always use a clear **textbook-style illustration style**. ### Output Format (two sections, clearly separated)

- #### **Section 1: Plan** Provide a structured reasoning plan containing the following **four parts**:

- 1. **Image Content** — Describe the elements to be drawn (shapes, objects, points, lines). Ensure all entities mentioned in the text are accounted for.
- 2. **Layout** — Explain the approximate spatial arrangement: relative positioning, coordinates estimation, scale.
- 3. **Labels** — Specify labels and annotations. **Crucial**: List all **given values** (numbers, variables) from the text that must be labeled on the diagram to show the initial state.
- 4. **Drawing Considerations** — Mention stylistic or logical constraints:

- - What must **NOT** be shown (to avoid solution leakage).
- - Matplotlibspecific details (e.g., ‘set_aspect(’equal’)‘, patches). Each section should be 1–4 bullet points.

—

- #### **Section 2: Python Code** Provide a **complete and runnable Python script**, formatted like this:

```python import matplotlib.pyplot as plt import matplotlib.patches as patches import numpy as np

def draw_diagram():

- # 1. Setup Figure fig , ax = plt.subplots(figsize=(6, 6)) ax.set_aspect('equal') # Crucial for geometry/physics ax.axis('off') # Hide axes unless strictly necessary for graphs
- # 2. Define Coordinates # ...
- # 3. Draw Elements (Shapes , Lines , etc.) # ...
- # 4. Add Labels and Annotations # (Ensure all given values from the question are visibly labeled) # ...
- # 5. Finalize and Show plt.tight_layout() plt.show()

if __name__ == "__main__":

draw_diagram() ````

**Rules:**

- * **Initial Setting Only**: The diagram must represent the problem *before* any solution steps are taken. Visualize the "Given" but hide the "Solution".
- * **Completeness**: If the text says "radius is 5", the diagram must show the circle and label the radius "r=5".
- * **Geometric Accuracy**: Ensure ‘ax.set_aspect(’equal’)‘ is used to prevent distortion for geometric/physical diagrams.
- * **Style**: Textbook aesthetic—consistent line thickness, clean alignment, clear font sizes.

—– ### **Input Fields** Original Question: {question} Now, follow the format strictly and generate the output.

Figure 15: The system prompt used for Imgcoder method.

###### Prompt: STEM Data Curation (Filtering & Taxonomy)

You are an expert STEM data curator and multimodal content analyzer. Your task is to process a raw text-based STEM problem to determine its suitability for "Text-to-Image" synthesis. If the problem is valid, you must classify it into a visual taxonomy and assess its quality based on scene clarity and visual complexity. Input Data:

- - Subject: {subject}
- - Question: {question}

—

- ### Step 1: Data Filtering & Visualizability Check Analyze the question text. Set ‘is_valid‘ to ‘false‘ if ANY of the following "Dirty Data" criteria are met:

- 1. **Dependency on Missing Context:** - The question references a missing image (e.g., "As shown in the figure...", "Which label in the diagram...").
- 2. **Missing Essential Data:** - The question asks for a calculation but lacks specific numbers/equations (e.g., "Calculate the probability given the provided values..." but values are missing).

- General theoretical discussion without a specific scenario.

- 3. **Wrong Task Type:** - Pure proof, code writing, or instructions to "draw a graph" (we need descriptions, not commands).
- 4. **Non-Visual / Abstract:**

- Concepts too abstract to be represented by a scientific diagram.

—

- ### Step 2: Taxonomy Classification (Only if ‘is_valid‘ is true) Classify the image type based on the ‘subject‘ and visual nature.

- *Use ’Universal’ if the visual structure is generic (e.g., charts, graphs) and not unique to the subject.*
- **Taxonomy:** 1. **Math:** Plane Geometric, Solid Geometric, Analytic Geometry, Set & Probability

- 2. **Physics:** Mechanical, Field Diagram, Waveform, Optical Ray, Astronomical, Circuit, Thermodynamic
- 3. **Chemistry:** Molecular Structure, Electron Config, Reaction Scheme, Crystal Structure, Spectra, Orbital / Quantum
- 4. **Biology:** Cell Diagram, Ecological, Genetics, Molecular Process
- 5. **Universal:** Function Graph, Table / Grid, Data Chart, Node-Link, Exp. Setup

—

- ### Step 3: Quality Assessment (Only if ‘is_valid‘ is true) Rate the following two metrics on a scale of 1 to 5.

**1. Scene Clarity Score (1-5): How explicit is the visual description?**

- * **1 (Vague):** Ambiguous scenario. Hard to draw without hallucinating details (e.g., "A car moves on a road" - no speed, angle, or surroundings specified).
- * **3 (Moderate):** Key elements are present, but some layout details need inference (e.g., "A triangle with sides 3 and 4" - angle or orientation implied but not stated).
- * **5 (Crystal Clear):** Fully deterministic. All coordinates, values, labels, and spatial relationships are explicitly stated (e.g., "A circle centered at (0,0) with radius 5, intersecting a line y=2").
- **2. Visual Complexity Score (1-5): How dense/complex is the resulting image?** * **1 (Simple):** Single object or very simple relationship (e.g., one standalone chemical bond, a single rectangle).
- * **3 (Medium):** Multiple interacting components (e.g., a pulley system with 2 blocks, a benzene ring).
- * **5 (Complex):** Highly dense information, intricate topology, or many distinct entities (e.g., a complex food web, a circuit with mixed parallel/series resistors and bridges, a detailed eukaryotic cell structure).

### Output Format Provide your response in JSON format only.

```json {

"reasoning": "Briefly explain validity decision , taxonomy choice , and score justification.", "is_valid": true/false , "primary_category": "Math/Physics/Chemistry/Biology/Universal or null", "secondary_type": "Specific Type or null", "scene_clarity_score": 1-5 (Integer) or null , "visual_complexity_score": 1-5 (Integer) or null

}

Figure 16: The system prompt used for visualizability filtering and taxonomy classification.

###### Prompt: Visual Quiz Generator

You are a precision-focused Dataset Generator for evaluating Large Multimodal Models. Your goal is to convert scientific text descriptions into a structured JSON object containing a visual checklist and a validation quiz.

**Task:** Analyze the provided ‘[Input Text]‘ to:

- 1. Extract a comprehensive **Visual Checklist** of atomic facts that *must* be visualized.
- 2. Generate a **Validation Quiz** to verify these visual details without relying on external knowledge.

**Input Text:** {question}

**Phase 1: Visual Checklist Generation (‘elements‘)**

- * **Goal:** Create a list of "Atomic Visual Constraints" that a validator would check off one by one.
- * **Requirements:**
- * Each element must be a specific, standalone visual fact derived from the text.
- * Include **Values** (e.g., "Label ’50kg’"), **Relationships** (e.g., "Box A is on top of Box B"),
- **Attributes** (e.g., "Dashed line for the normal force"), and **Directions** (e.g., "Arrow points to the right").
- * **Format:** Short, descriptive strings (e.g., "Resistor R1 = 10kω", "Current I flows clockwise").
- **Phase 2: Quiz Generation (‘quiz‘)**
- * **Goal:** Create Multiple-Choice Questions (MCQs) that force a model to *look* at the image to verify the checklist items. * **Anti-Cheating Rules:**

- 1. **Instance-Specific ONLY:** Do not ask general knowledge questions (e.g., "What is gravity?"). Ask only about the specific values/labels described (e.g., "What value is labeled for gravity in this diagram?").
- 2. **Visual Verifiability:** Phrasing should imply visual inspection (e.g., "What is the direction of the arrow labeled ’F’?").
- 3. **Hard Negatives:** Use *other* numbers/entities from the text as distractors to test precise visual grounding.

- **Output Format (Strict JSON):** Return a single JSON object. Ensure the JSON is valid and parsable.
- **JSON Schema:**

```json {{

"elements": [

- "String describing visual fact 1 (e.g., 'Resistor R1 is labeled 100 ohm ')",
- "String describing visual fact 2 (e.g., 'Vector F points at 45 degrees ')",
- "String describing visual fact 3 (e.g., 'Point A is connected to Point B')"

], "quiz": [

{{

"question": "The question string (visually grounded)", "options": {{

- "A": "Option text",
- "B": "Option text",
- "C": "Option text",
- "D": "Option text"

}}, "correct_option": "A", "evidence_snippet": "Substring from text proving this fact"

}} ]

}}

Figure 17: The system prompt used for constructing visual validation quizzes.

###### Prompt: LMM-as-Judge Evaluator

You are an expert evaluator of scientific and technical diagrams (e.g., geometry, physics, chemistry). Evaluate the image against the caption on these 5 dimensions:

- ### 1. Correctness & Fidelity (0–2) Core Question: Does the image completely and accurately represent all elements, labels, and spatial/logical relationships from the caption, with no omissions OR hallucinations?

- * **2 (High):** Perfect match. All elements (points, lines, shapes, labels) from the caption are present and correct. All specified spatial (e.g., ’left of’, ’inside’) and logical (e.g., ’perpendicular’, ’tangent’, ’connected to’) relationships are perfectly accurate. Crucially, there are NO spurious or "hallucinated" elements (e.g., random lines, meaningless intersections) not implied by the caption.
- * **1 (Medium):** Mostly correct. Most key elements are present, but with minor omissions, misplacements, or simplifications. Spatial/logical relationships are mostly right but have slight inaccuracies. May have minor spurious elements that don’t confuse the main subject.
- * **0 (Low):** Major mismatch. Key elements are missing, incorrect, or relationships are wrong. Or, the image contains significant spurious content (visual noise, random intersections) that contradicts or confuses the caption.

—

- ### 2. Layout & Precision (0–2) Core Question: Is the layout clear and technically precise? Does the visual arrangement correctly reflect the logical coordinates and relative spatial positions described?

- * **2 (High):** Professional and spatially accurate. Layout is clear, balanced, and precise (straight lines, exact connections). Visual positions perfectly match the logical labels/coordinates (e.g., (10, 0) is distinctively right of (2, 0)) and relative positions are strictly maintained.
- * **1 (Medium):** Generally readable but with minor distortions. Layout is understandable but may have slight alignment issues or imprecision. Relative positions are mostly correct (topology preserved), but scale or visual distances may be inaccurate (e.g., order is right, but proportions are off).
- * **0 (Low):** Sloppy or spatially contradictory. Layout is cluttered, chaotic, or elements are poorly proportioned. Lines are visibly imprecise/disconnected, OR elements are placed in positions that contradict their coordinates (e.g., positive coordinates drawn on the negative axis, or inverted positions).

—

- ### 3. Readability & Occlusion (0–2) Core Question: Do visual elements or labels overlap or occlude each other in a way that obscures meaning or reduces readability?

- * **2 (High):** No occlusion. Every element (shapes, arrows, text labels) is fully distinct and clearly separated, with no confusing overlap.
- * **1 (Medium):** Minor overlap. Some elements or labels slightly touch or overlap, but it only marginally affects readability (e.g., an arrowhead just touches a label). The core content remains understandable.
- * **0 (Low):** Significant occlusion. Key elements or labels overlap heavily, making parts of the diagram unreadable, ambiguous, or indistinguishable.

—

- ### 4. Scientific Plausibility (0–2) Core Question: Does the image visually conform to the basic principles and conventions of its scientific domain (e.g., physics, geometry), even if not explicitly stated in the caption?

- * **2 (High):** Visually plausible. The image "looks right" for its domain. E.g., geometric angles/proportions look reasonable; physics vectors (if representing equilibrium) look balanced; chemical bond angles appear conventional (e.g., VSEPR).
- * **1 (Medium):** Minor implausibility. The image is scientifically/logically functional but has minor visual flaws (e.g., a 90° angle looks like 80°; a molecule’s bond angle is visibly awkward but still conveys the connection).
- * **0 (Low):** Visually implausible. The image clearly violates basic scientific/logical principles in its visual representation (e.g., a force diagram that is obviously unbalanced; a geometric proof figure that is impossibly skewed).

—

- ### 5. Expressiveness & Richness (0–2) Core Question: Does the image completely and vividly reproduce the scenario described in the problem?

- * **2 (High):** Comprehensive reproduction. The image not only contains the correct elements but also effectively conveys the full *context* or *situation* of the problem. It is visually rich and fully illustrates the prompt’s intent.
- * **1 (Medium):** Basic representation. The image depicts the necessary elements for the problem but lacks contextual richness or detail. It is functional but minimal.
- * **0 (Low):** Incomplete scenario. The image fails to convey the setting or context of the problem, making it difficult to understand the "story" or situation behind the diagram.

### **Output Format** Provide short reasoning for each dimension, then output a JSON object with integer scores.

- **Example Output:** **Reasoning:**
- * **Correctness & Fidelity:** The image correctly shows all 5 points and the 3 lines connecting them as described. All labels are present. No extra lines appear.
- * **Layout & Precision:** Lines are straight and connect perfectly at the nodes. The layout is balanced.
- * **Readability & Occlusion:** Label ’A’ and ’B’ are slightly too close, but do not overlap. All elements are readable.
- * **Scientific Plausibility:** The diagram (a geometric proof) shows angles that appear consistent with the "given" perpendicular lines.
- * **Expressiveness & Richness:** The diagram fully captures the geometry problem’s scenario, clearly visualizing the intersecting planes described in the text.

JSON {

"Correctness_Fidelity": 2,

"Layout_Precision": 2, "Readability_Occlusion": 2, "Scientific_Plausibility": 2, "Expressiveness_Richness": 2

29

} Question: {question} Reason & JSON output:

Figure 18: The system prompt used for LMM-as-Judge evaluation on 5 dimensions.

###### Prompt: Multimodal Adaptation

**Role:** You are an expert Educational Content Editor specializing in **Multimodal Adaptation**. Your goal is to transform text-only STEM problems into "Visual-Dependency" problems.

**Task:** You will receive a raw text question. You must rewrite it so that the specific **numerical values, geometric properties, or component parameters** are removed from the text and implied to be present in an accompanying image.

**Core Logic:** The user MUST look at the image to solve the problem. If the text contains all the data, the image becomes redundant. You must fix this redundancy.

### Step-by-Step Execution Rules

**1. Identification (What to Hide):** Identify the **"Given Data"** (Explicit Parameters) that can be visualized as labels.

- * *Target:* Numbers (e.g., "5 kg", "30°", "10V", "radius 4"), Coordinates (e.g., "(0,2)"), Component types if labeled (e.g., "a 500nm wavelength").
- * *Exception:* Do NOT hide the **"Unknown/Target"** (what the user needs to find) or generic constants (e.g., "g = 9.8 m/s²") unless they are specific to the diagram setup.
- **2. Removal & Replacement (How to Hide):**
- * **REMOVE** the identified specific values/properties from the text.
- * **REPLACE** them with visual pointers such as: * *"as shown in the figure"* * *"the labeled angle"* *
- *"the indicated dimensions"* * *"the circuit diagram below"* * *"shown/depicted"*
- **3. Grammatical Repair:**
- * Ensure the rewritten sentence flows naturally. * *Bad:* "A block of [removed] is on a [removed] plane." *
- *Good:* "A block of **labeled mass** is on an inclined plane **as shown**."

### Examples of Adaptation

- **Example 1: Geometry**
- * **Input:** "In triangle ABC, angle A is 60 degrees and angle B is 45 degrees. Find angle C." * **Output:** "In the triangle ABC **shown in the figure**, find angle C given **the labeled angles**." * *(Note: 60 and 45 are hidden).*
- **Example 2: Physics (Circuit)**
- * **Input:** "A 10V battery is connected to a 5-ohm resistor." * **Output:** "A battery is connected to a resistor **as shown in the circuit diagram**. Calculate the current." * *(Note: 10V and 5-ohm are hidden).*
- **Example 3: Physics (Mechanics)**
- * **Input:** "A ball is thrown with an initial velocity of 20 m/s at an angle of 30 degrees." * **Output:** "A ball is thrown with the initial velocity and angle **indicated in the diagram**." * *(Note: 20 m/s and 30 degrees are hidden).*
- **Example 4: Math (Simple)**

* **Input:** "Find the area of a circle with radius 4." * **Output:** "Find the area of the circle **shown below**." * *(Note: Radius 4 is hidden).*

### Output Format Return the result in JSON format:

{

"original_question": "[Input Question]", "hidden_parameters": ["List specific values you removed , e.g., '5kg', '30 degrees '"], "multimodal_question": "The rewritten text with visual pointers"

}

### User Input

**Question:** {question}

Figure 19: The system prompt used for adapting text-only problems into multimodal format.

