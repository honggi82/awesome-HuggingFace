arXiv:2506.03139v1[cs.CV]3Jun2025

# SVGenius: Benchmarking LLMs in SVG Understanding, Editing and Generation

Siqi Chen∗, Xinyu Dong∗, Haolei Xu, Xingyu Wu, Fei Tang, Hang Zhang, Yuchen Yan, Linjuan Wu, Wenqi Zhang, Guiyang Hou, Yongliang Shen, Weiming Lu, Yueting Zhuang

{siqichen,syl}@zju.edu.cn Zhejiang University Hangzhou, China

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

##### Understanding

##### Editing

##### Generation

##### SVG Processing Capability

###### Text-to-SVG

Image-to-SVG

Style Editing

Perceptual QA

Bug Fixing

[Figure 5]

Transform element

Transform SVG

Image-to-SVG Generation

[Figure 6]

Set Contour

[Figure 7]

Generate an SVG icon about {caption} and reference image

[Figure 8]

Shape

###### Quantity

[Figure 9]

[Figure 10]

[Figure 11]

Semantic QA

Generate an SVG icon about {caption}

[Figure 12]

[Figure 13]

What is the primary shape that forms the body of the icon?

How many chimneys are emitting smoke in the factory icon?

Original

[Figure 14]

<svg>

<svg>

[Figure 15]

[Figure 16]

Change color

Set blur

Add gradient

[Figure 17]

[Figure 18]

- <path {path1}/>
- <path {path2}/>

[Figure 19]

- <path {path1}/>
- <path {path2}/>

Perceptual QA

[Figure 20]

Style Editing

###### Color

... </svg>

... </svg>

What is the primary color observed in the icon?

[Figure 21]

[Figure 22]

[Figure 23]

Semantic QA

###### Bug fixing

###### Code Optimization

###### Style transfer

[Figure 24]

Text-to-SVG Generation

Meaning

###### Function

[Figure 25]

Style Transfer Code Optimization

[Figure 26]

[Figure 27]

Please transform the {SVG} to {3D/Line/Pixel} style

[Figure 28]

Original: 11.2KB

<path fll=”...>.....</path>

What does this icon typically represent in a physics context?

In which context would this icon most likely be used?

[Figure 29]

Original

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

###### Usage

[Figure 40]

[Figure 41]

Based on the design and style of the jacket, which setting is it most likely associated with?

[Figure 42]

Line style Pixel style 3D style

Compressed: 7.5KB

<path fill=”...”>...</path>

- Figure 1: Overview of SVGenius. SVGenius evaluates (M)LLMs capabilities across three progressive dimensions: Understanding (perceptual and semantic QA), Editing (bug fixing, code optimization, style editing), and Generation (text-to-SVG, image-to-SVG, style transfer). Built on real-world data from 24 domains with systematic complexity stratification, our benchmark enables comprehensive assessment of SVG processing capabilities. The radar chart shows representative model performance patterns, revealing distinct capability boundaries.

## Abstract

though style transfer remains the most challenging capability across all model types. SVGenius establishes the first systematic evaluation framework for SVG processing, providing crucial insights for developing more capable vector graphics models and advancing automated graphic design applications. Appendix and supplementary materials (including all data and code) are available at https://zju-real.github.io/SVGenius.

Large Language Models (LLMs) and Multimodal LLMs have shown promising capabilities for SVG processing, yet existing benchmarks suffer from limited real-world coverage, lack of complexity stratification, and fragmented evaluation paradigms. We introduce SVGenius, a comprehensive benchmark comprising 2,377 queries across three progressive dimensions: understanding, editing, and generation. Built on real-world data from 24 application domains with systematic complexity stratification, SVGenius evaluates models through 8 task categories and 18 metrics. We assess 22 mainstream models spanning different scales, architectures, training paradigms, and accessibility levels. Our analysis reveals that while proprietary models significantly outperform open-source counterparts, all models exhibit systematic performance degradation with increasing complexity, indicating fundamental limitations in current approaches; however, reasoning-enhanced training proves more effective than pure scaling for overcoming these limitations,

## CCS Concepts

• Do Not Use This Code → Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper; Generate the Correct Terms for Your Paper.

## Keywords

SVG Processing, SVG Benchmark, Large Language Model

ACM Reference Format:

Siqi Chen∗, Xinyu Dong∗, Haolei Xu, Xingyu Wu, Fei Tang, Hang Zhang, Yuchen Yan, Linjuan Wu, Wenqi Zhang, Guiyang Hou, Yongliang Shen, Weiming Lu, Yueting Zhuang. 2025. SVGenius: Benchmarking LLMs in SVG Understanding, Editing and Generation. In . ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

∗Both authors contributed equally to this research.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

## 1 Introduction

Scalable Vector Graphics (SVG) has become essential for modern applications ranging from UI design to data visualization, offering lossless scalability and high editability compared to raster formats.

Conference’17, Washington, DC, USA © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

- Table 1: Comparison of SVGenius with existing SVG processing benchmarks. We compare construction methods, domain diversity, complexity metrics (paths and control points), and task coverage. SVGenius provides the first comprehensive evaluation across Understanding, Editing, and Generation with systematic complexity stratification. Task abbreviations: PQA (Perceptual QA), SQA (Semantic QA), BF (Bug Fixing), CO (Code Optimization), SE (Style Editing), TTS (Text-to-SVG), ITS (Image-to-SVG), ST (Style Transfer).

Complex Understanding Editing Generation Paths Points PQA SQA BF CO SE TTS ITS ST

Benchmark Construct-Method Domain-Diversity

SVGEditBench [30] Automated Single 6.12 337.22 ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ SVGEditBench V2 [31] Hybrid Single 9.96 205.66 ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ MMSVG-2M [60] Automated Multi - - ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ VGBench [64] Hybrid Multi 5.64 414.76 ✓ ✓ ✗ ✗ ✗ ✓ ✗ ✗ SVGBench [37] Automated Single 6.08 512.54 ✗ ✗ ✗ ✗ ✗ ✗ ✓ ✗ Image-Text Bridging [4] Hybrid Single - - ✓ ✓ ✗ ✗ ✓ ✗ ✗ ✗ ColorSVG-100K [7] Automated Multi 13.22 952.14 ✗ ✗ ✗ ✗ ✗ ✓ ✓ ✗ SVG-Taxonomy [58] Automated Single 3.98 55.58 ✗ ✗ ✗ ✗ ✓ ✗ ✗ ✗ SGP-Bench [33] Hybrid Multi 5.50 334.38 ✓ ✓ ✗ ✗ ✗ ✗ ✗ ✗

SVGenius(Ours) Hybrid Multi 10.15 673.64 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓

However, its XML-based syntax and geometric complexity create significant barriers for non-expert users, motivating automated approaches through differentiable optimization [18, 19, 29, 57] and deep learning methods [5, 13, 36, 42].

Recent advances in Large Language Models (LLMs)[3, 8, 47] and Multimodal LLMs[2, 26, 43] have opened new possibilities for SVG processing. These models demonstrate strong capabilities in structured content understanding and generation [6, 9, 23], making them well-suited for SVG manipulation. The textual nature of SVG code naturally aligns with these models’ language-centric architecture, enabling direct interpretation and processing of vector graphics. This has led to promising applications across diverse tasks: Iconshop [53] and StarVector [37] focus on icon design and generation, while OmniSVG [60] tackles complex illustration synthesis. However, a critical challenge emerges: how do we systematically evaluate and compare the SVG processing capabilities of different models?

As shown in Table 1, existing benchmarks suffer from three critical limitations: (1) limited scope: reliance on synthetic or overly simplified samples that fail to reflect the structural and semantic diversity of real-world graphics; (2) lack of complexity stratification: uniform treatment of all samples without considering structural complexity that are crucial for understanding model capability boundaries; (3) fragmented evaluation: focus on isolated capabilities rather than comprehensive SVG processing capabilities required for practical applications. Most benchmarks [4, 30, 31, 37, 60] evaluate on narrow task subsets, ignoring the multi-stage nature of SVG processing tasks. These gaps hinder fair model comparison and effective guidance for future improvements.

To address these challenges, we introduce SVGenius, a comprehensive benchmark for evaluating SVG processing across multiple dimensions and complexity levels. Built on real-world data from IconFont [17] spanning 24 application domains, our benchmark features a novel complexity stratification framework based on quantitative metrics that organizes samples into three hierarchical levels mirroring real-world design challenges. We evaluate models across three progressive dimensions that capture the full spectrum of SVG

processing capabilities: understanding (perceptual and semantic comprehension), editing (code manipulation and optimization), and generation (SVG synthesis and style transfer).

Our benchmark comprises 2,377 queries across 8 task categories and 18 evaluation metrics, enabling systematic assessment of model capabilities. We conduct extensive experiments on 22 mainstream models spanning different scales, architectures, training paradigms, and accessibility (open-source vs. proprietary). Our analysis reveals four key findings: (1) substantial performance gaps exist between open-source and proprietary models, with the gap widening on complex tasks; (2) all model families exhibit systematic performance degradation as SVG complexity increases, indicating fundamental limitations in current approaches; (3) reasoning-enhanced training strategies significantly improve performance, particularly for generation tasks requiring multi-step planning; and (4) style transfer emerges as the most challenging capability, with even state-of-theart models achieving modest performance.

Out contributions can be summaried as: (1) We identify key limitations in existing SVG evaluation approaches and propose a comprehensive solution; (2) We introduce SVGenius, the first large-scale, complexity-stratified benchmark for SVG processing with real-world data; (3) We provide extensive evaluation of 22 models, establishing performance baselines and identifying key factors influencing SVG processing capabilities.

## 2 Related Works 2.1 SVG Processing

EarlyAI-drivenSVGprocessingfocused on generation using RNNs [13,

14, 36, 39, 40], VAEs [5, 27, 41, 42, 46], and Transformers [5, 53] to model SVG command sequences. However, these methods faced limitations in representational capacity and geometric consistency for complex graphics. Subsequent work introduced differentiable rasterizers [24] to bridge vector-raster representation gaps, followed by diffusion-based approaches [19, 44, 56, 57, 61] that improved visual fidelity through iterative refinement.

The emergence of (M)LLMs [6, 9, 23, 48–50] has opened new possibilities for SVG processing. Recent works [37, 52, 53, 55, 60]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Dataset Construction

- (1a) Data Processing

<svg version="1.1" id="Layer_1" xmlns="http://www. w3.org/2000/svg"

xmlns:xlink="http:// www.w3.org/1999/x link" viewBox="0 0

1000 1000" > <path d="" /> </svg>

Structural Validity Semantic Validity

<?xml version="1.0" encoding="utf8"? ?<!-- Generator: Adobe Illustrator 23.1.1, SVG Export Plug-In . SVG Version: 6.00 Build 0) --> <svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/199 9/xlink" viewBox="0 0 1000 1000" > <path d="" /> </svg>

[Figure 47]

[Figure 48]

<svg>

- <path>{path1}</path>
- <path>{path2}</path>
- <path>{path3}</path>

... </svg>

SVG Code

[Figure 49]

- (2) Complex Stratification Path Count

UI

Services

Web &

Urban

Data Visualization

&

Meteorology

Design

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Weather

[Figure 54]

[Figure 55]

Logo

Transportation &

Navigation

& Icons E-commerce

Environmental

[Figure 56]

Data Collection

- (1a) Data Processing

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

- (1b) Human Filter

BFiunsainnecses &

Smart Home

Legal & Compliance

Time Management

# Instance-Level feature distribution

Social & Communication

Security & Privacy

[Figure 63]

[Figure 64]

[Figure 65]

Documentation & Ofice

[Figure 66]

Industrial & Manufacturing

Complex Score

Tourism

###### (2) Complex Stratification

Healthcare

& Hospitality

&

Food

Science

Research

&

Entertain ment

& Dining

Ga ming

Sports &

Education

Training

Fitness

[Figure 67]

&

SVGenius

Complex Demand Point Count

Fashion

Beauty

&

[Figure 68]

[Figure 69]

Hard Level

Easy Level

Medium Level

Overview of SVGenius dimensions

Manual Assistance

# Complexity Score of curvature

- Figure 2: SVGenius dataset construction and complexity validation. Left: systematic pipeline from data collection, processing, human filtering to complexity stratification. Center: 24-domain coverage across diverse applications. Right: validation of complexity stratification showing clear hierarchical separation across Easy, Medium, and Hard levels through feature distributions and complexity scores.

have demonstrated the feasibility of leveraging these models’ text understanding capabilities for SVG manipulation, capitalizing on the natural compatibility between XML-based SVG format and language modeling architectures.

2.2 SVG Benchmarks

Existing benchmarks evaluate specific aspects of SVG processing capabilities. Image-Text Bridging [4] assesses cross-modal perception combining SVG understanding with basic editing. SGP-Bench [33] focuses on semantic consistency in symbolic graphic programs. SVGEditBench [30] and V2 [31] emphasize editing performance using syntactic metrics like MSE and compression ratios. Generationfocused methods include OmniSVG [60], StarVector [37], and SVGBuilder [7] for text/visual-conditioned synthesis. VGBench [64] introduces dual-task evaluation with understanding (VGQA) and generation (VGen) components.

However, existing benchmarks suffer from fragmented evaluation focusing on isolated capabilities, lack of complexity stratification, and reliance on structurally simple samples. SVGenius addresses these limitations by providing a comprehensive framework spanning understanding, editing, and generation with systematic complexity modeling and multi-dimensional evaluation metrics.

- 3 Dataset Construction

to ensure semantic clarity. Following standardized preprocessing (geometric normalization, center alignment, attribute standardization), we obtain 927 high-quality SVGs.

Complexity Stratification. We define complexity through three quantitative indicators: path count (structural complexity), control points (geometric intricacy), and complex commands (advanced operations). These metrics are normalized and combined using empirically determined weights. Based on score distributions, we stratify samples into Easy, Medium, and Hard levels (33%/34%/33% split). From each level, we sample 200 candidates, conduct manual visual inspection, and retain 100 high-quality SVGs per level, yielding a balanced 300-sample spanning 24 practical domains dataset with clear quality separation validated in Appendix B.

## 4 Task Framework

SVGenius introduces a comprehensive evaluation framework spanning three progressive competency dimensions: Understanding, Editing, and Generation. This design reflects the natural progression of SVG processing skills, from basic comprehension to sophisticated content creation, enabling systematic assessment of model capabilities across the full spectrum of practical applications.

## 4.1 Understanding

Understanding capabilities form the cornerstone of effective SVG processing, encompassing both low-level instruction parsing and high-level semantic construction. Existing benchmarks [4, 58, 64] oversimplify this dimension through basic question-answer pairs, failing to systematically evaluate multi-level comprehension abilities. We introduce two complementary understanding tasks that progressively assess model capabilities from perceptual recognition to semantic interpretation:

Current SVG benchmarks face critical limitations in data quality and diversity. Early efforts like SVGEditBench [30] and Image-Text Bridging [4] rely on structurally simple icons or synthetic data, while recent works such as OmniSVG [60] and VGBench [64] attempt stylistic variation but lack systematic complexity modeling.

To address these limitations, we construct a comprehensive dataset with principled complexity stratification in Figure 2. Starting with over 100K SVG samples from Iconfont [17]—a platform of user—created vector icons- we apply systematic processing based on structural validity, semantic effectiveness, and representational compactness. Ten volunteers manually review rasterized versions

• Perceptual QA (PQA) evaluates fundamental visual recognition capabilities essential for SVG interpretation. Models must extract visual cues from SVG code to recognize basic attributes including colors, shapes, spatial relationships, and quantities. Implemented

as four-option multiple-choice questions requiring direct code interpretation, this task establishes baseline perceptual processing capabilities using accuracy as the evaluation metric. (cf. Appendix F for more details)

• Semantic QA (SQA) advances to higher-level comprehension and abstract reasoning. The task encompasses semantic understanding through three categories: function identification, meaning summarization, and usage prediction. This evaluation requires sophisticated visual-language understanding beyond literal attributes, with accuracy serving as the primary metric for semantic inference capabilities. (cf. Appendix F for more details)

## 4.2 Editing

Building upon understanding foundations, editing tasks assess models’ ability to perform precise, structured code manipulations—a critical requirement for practical applications. Current efforts [4, 30, 31, 33] focus narrowly on basic attribute modifications with limited scope diversity. We design three comprehensive editing scenarios that systematically evaluate code manipulation competency:

- • Bug Fixing (BF) addresses SVG-specific error correction across three primary categories: tag errors (malformed XML structure), attribute errors (incorrect formats), and path command errors (malformed data or sequences). Unlike general program repair benchmarks [20, 22, 25, 28], this task targets unique SVG characteristics requiring both syntactic understanding and semantic preservation. We evaluate the performance of models using repair accuracy to measure the correctness of the fixed output. (cf. Appendix G.1 for more details)
- • Code Optimization (CO) evaluates quality improvement beyond visual correctness. Real-world SVG generation often produces suboptimal code with inefficient structures. The task involves optimizing code following SVGO [10]-inspired principles while preserving rendering output. We evaluate performance using two metrics: mean squared error (MSE) to ensure rendering consistency, and code compression ratio to quantify optimization effectiveness. (cf. Appendix G.2 for more details)
- • Style Editing (SE) assesses interactive modification capabilities through six representative operations: global position adjustment, local element movement, set contour, color modification, gradient filling, and blur effects. Since localized modifications yield minimal absolute differences, we introduce relative MSE (rMSE) normalized by reference variance for sensitive quality detection, integrated with existing measures in a four-indicator framework. (cf. Appendix G.3 for more details)

## 4.3 Generation

Generation represents the most sophisticated capability dimension, requiring models to synthesize complete SVGs from scratch conditioned on natural language or multimodal inputs. Existing benchmarks [4, 60, 64] focus primarily on unimodal scenarios, missing broader challenges of structured construction. We introduce three progressively challenging generation tasks:

• Text-to-SVG (TTS) evaluates fundamental natural language to vector graphics translation. To comprehensively assess semantic alignment and code-level differences, we introduce two novel

metrics: rCLIP and PSS. These integrate with existing measures to form a three-dimensional framework: (1) Perceptual Quality (HPS [54], Aesthetic [38]) measuring subjective visual appeal, (2) Visual Reproducibility (PSS) assessing code structure consistency, and (3) Semantic Consistency (CLIP [35], rCLIP) evaluating semantic preservation. (cf. Appendix E.1 for more details)

- • Image-to-SVG (ITS) addresses natural language ambiguity by requiring generation from both images and text. This paradigm mitigates discrepancies between textual descriptions and user expectations by providing visual guidance. Evaluation employs two categories: (1) Perceptual Similarity using LPIPS [62], SSIM [51], and DINO [32] for alignment as assessment, and (2) Visual Reproducibility through PSS and MSE for consistency evaluation. (cf. Appendix E.2 for more details)
- • Style Transfer (ST) presents the ultimate challenge, requiring simultaneous content preservation and style adaptation. Despite its importance, standardized SVG style transfer benchmarks remain absent. We introduce a task demanding generation of SVGs retaining structural content while conforming to four predefined stylistic categories. We develop a two-tier automated assessment framework leveraging LLMs to quantify transfer quality from global and local perspectives. (cf. Appendix E.3 for more details)

5 Experiment

- 5.1 Experimental Setup

We evaluate a diverse set of models on SVGenius to assess SVG processing capabilities across different architectures, scales, and training paradigms. Our evaluation encompasses proprietary models (GPT-4o [16], Gemini-2.0-Flash [43], Claude-3.7-Sonnet [1]), representative open-source models (DeepSeek-R1 [12], Qwen2.5/3 [34, 59], Llama-3.2 [11], Mistral-Small [21]), and SVG-specialized systems (Iconshop [53], StarVector [37], LLM4SVG [55]). All models are evaluated under zero-shot settings using default configurations across three complexity levels (Easy, Medium, Hard) with three independent runs per setting for statistical robustness. Due to space constraints, we present results for representative models in the main text, with the complete leaderboard provided in Appendix C. In all tables, models are marked as reasoning (★), code (★), open-source (★), proprietary (★), or specialized (★) variants. These metrics (MSE, HPS, PSS, SSIM, LPIPS, DINO) are reported in scientific notation with units of 10−2 for consistency and readability.

- 5.2 Main Results

We evaluate 22 models across SVGenius spanning understanding, editing, and generation tasks. Tables 2, 3, 4 and 5 present results for representative models, with complete leaderboard in Appendix C. Proprietary models achieve superior performance but face complexity barriers. Claude-3.7-Sonnet [1] leads in understanding tasks (80.25% Easy PQA, 77.78% Easy SQA) and editing (76% bug fixing accuracy), while GPT-4o [16] excels in generation (20.35 HPS, 19.72 PSS in text-to-SVG, 23.43 PSS in multimodal generation). However, all proprietary models exhibit substantial performance degradation with increasing complexity. GPT-4o [16] drops from 82.72% to 42.22% in Perceptual QA across difficulty levels, while Gemini-2.0-Flash [43] shows similar degradation from 77.78% to

- Table 2: Performance on SVG understanding dimension across different model types and difficulty levels for selected models. Accuracy scores are shown for Perceptual QA and Semantic QA tasks.

Method

Perceptual QA(ACC↑) Semantic QA(ACC↑) Easy Med. Hard Easy Med. Hard

DS-R1-Qwen-32B [12]★ 64.20 43.42 42.22 51.85 52.63 37.78 DeepSeek-R1 [12]★ 74.19 55.13 44.44 74.19 71.79 55.56 Qwen2.5-72B-Ins [34]★ 67.74 38.46 24.44 50.54 47.43 51.11 QwQ-32B [59]★ 62.37 34.62 33.33 53.76 57.69 37.78 Qwen3-1.7B [59]★ 46.91 22.37 28.89 41.98 44.74 22.22 Qwen3-8B [59]★ 70.96 43.59 22.22 44.09 46.15 42.22 Qwen3-32B [59]★ 71.60 42.11 24.44 60.49 55.26 42.22 Gemini-2.0-Flash [43]★ 77.78 40.79 31.11 62.96 55.26 51.11 GPT-4o [16]★ 82.72 35.53 42.22 67.90 56.58 64.44 Claude-3.7-Sonnet [1]★ 80.25 47.37 33.33 77.78 65.79 71.11

- Table 3: Performance on SVG editing dimension across different model types and difficulty levels for selected models. Results are reported using task-specific metrics (ACC, rMSE, RLD, MSE, CCR) for Bug Fixing, Style Editing, and Code Optimization tasks.

Bug F. Style Editing Code Optim. ACC↑ ACC↑ rMSE↑ RLD↓ MSE↓ CCR↑

Method

Easy DS-R1-Qwen-32B [12]★ 62.63 84.81 75.16 1.45 5.63 23.16 DeepSeek-R1 [12]★ 71.00 84.62 73.66 18.21 1.01 23.67 Qwen2.5-72B-Ins [34]★ 71.00 75.95 64.25 2.08 2.98 20.57 QwQ-32B [59]★ 71.43 91.14 81.70 2.61 3.20 20.20 Qwen3-1.7B [59]★ 22.34 74.36 67.22 33.40 11.43 35.79 Qwen3-8B [59]★ 53.00 87.34 80.40 221.88 2.84 18.61 Qwen3-32B [59]★ 56.12 88.46 82.63 1.17 2.55 17.96 Gemini-2.0-Flash [43]★ 69.00 86.07 79.70 4.78 0.72 20.30 GPT-4o [16]★ 74.00 78.48 65.81 46.53 1.30 10.57 Claude-3.7-Sonnet [1]★ 76.00 79.75 67.17 20.89 0.31 16.81

Medium DS-R1-Qwen-32B [12]★ 46.00 56.41 53.11 140.68 8.14 33.10 DeepSeek-R1 [12]★ 63.83 62.16 53.93 1227.702.02 36.70 Qwen2.5-72B-Ins [34]★ 50.56 59.49 51.90 136.00 3.41 30.49 QwQ-32B [59]★ 46.00 64.56 61.27 41.67 5.03 29.28 Qwen3-1.7B [59]★ 1.22 33.90 25.55 1656.5010.20 34.87 Qwen3-8B [59]★ 35.42 62.34 52.10 826.21 2.63 21.69 Qwen3-32B [59]★ 46.47 66.67 62.54 17.13 2.78 26.61 Gemini-2.0-Flash [43]★ 60.00 65.82 60.43 113.44 1.59 27.49 GPT-4o [16]★ 49.00 50.63 42.64 840.73 4.74 43.01 Claude-3.7-Sonnet [1]★ 75.00 59.74 53.61 134.52 0.86 26.41

Hard DS-R1-Qwen-32B [12]★ 34.02 55.26 41.68 226.50 5.89 36.41 Deepseek-R1 [12]★ 61.80 51.56 41.83 2610.482.41 39.95 Qwen2.5-72B-Ins [34]★ 40.00 54.67 43.80 722.00 3.24 36.40 QwQ-32B [59]★ 10.42 50.00 44.04 533.85 5.82 12.98 Qwen3-1.7B [59]★ 0.00 26.42 21.94 3650.715.60 32.42 Qwen3-8B [59]★ 25.30 60.81 52.21 935.13 2.15 23.68 Qwen3-32B [59]★ 40.86 55.26 48.06 58.79 1.50 26.75 Gemini-2.0-Flash [43]★ 53.95 57.38 42.34 628.14 2.62 29.35 GPT-4o [16]★ 51.02 42.67 35.43 843.81 5.14 46.91 Claude-3.7-Sonnet [1]★ 69.00 52.56 40.75 27.00 0.88 28.27

31.11%, indicating fundamental limitations in processing complex SVG structures.

Open-source models reveal significant gaps but show specialized strengths. Conventional open-source models lag substantially, with Qwen2.5-72B [34] achieving 67.74% Easy PQA compared to Claude-3.7-Sonnet’s [1] 80.25%, representing a 12+% gap. However, reasoning-enhanced models bridge this gap: DeepSeek-R1 [12] achieves 74.19% in both Easy PQA and SQA, closely matching proprietary performance, while QwQ-32B [59] excels in editing with 91.14% style editing accuracy, surpassing most proprietary models including GPT-4o [16] (78.48%). Smaller open-source models face severe limitations, with Qwen3-1.7B [59] achieving only 22.34% bug fixing accuracy and many models completely failing on complex tasks.

Specialized models excel narrowly but lack robustness. Iconshop [53] outperforms most open-source models in text-to-SVG but deteriorates severely on complex tasks (12.95 HPS on Hard) and fails completely at style transfer. StarVector [37] shows reasonable multimodal performance with competitive SSIM scores (37.60-56.53) but similarly fails other tasks, indicating fundamental limitations in SVG processing. These results show that specialized approaches achieve domain expertise at the cost of general robustness.

## 5.3 Analysis

Model Scale and Architecture Effects Scaling yields substantial improvements within model families: Qwen3 variants improve from 22.34% to 56.12% easy bug fixing accuracy when scaling from 1.7B to 32B parameters. Multimodal architectures consistently outperform text-only variants in generation, with GPT-4o [16] achieving 23.43 vs 19.72 PSS scores, indicating that visual modalities enhance spatial reasoning capabilities essential for SVG generation.

Reasoning-Enhanced Training Improves SVG Processing Reasoning augmentation enables models to transcend scale limitations throughsystematicproblem-solving approaches. DS-R1-Qwen-32B [12] achieves 51.85% in Easy SQA despite having fewer parameters than Qwen2.5-72B [2] (50.54%). QwQ-32B similarly outperforms conventional models of similar scale, achieving 91.14% vs 88.46% easy style editing accuracy compared to Qwen3-32B [59]. These results suggest reasoning training develops better systematic approaches to SVG’s hierarchical structure and semantic relationships.

Complexity-Performance Degradation Patterns Performance degradation with increasing complexity is universal but varies by task type. Understanding tasks show steep degradation (Claude-3.7Sonnet [1]: 80.25% to 33.33% PQA, GPT-4o [16]: 82.72% to 42.22%), while editing exhibits moderate drops (10-30% across difficulty levels for most models). Generation tasks prove most resilient with 5-10% PSS score declines. Critically, degradation patterns remain consistent across model families—all Qwen variants show similar proportional decreases, and reasoning-enhanced models follow identical trends despite higher baselines, indicating fundamental limitations in current approaches rather than model-specific weaknesses.

Task-Specific Capability Boundaries Distinct capability boundaries reveal fundamental SVG processing challenges. Understanding consistently exceeds generation performance: top models achieve

- Table 4: Performance on SVG generation dimension across different model types and difficulty levels for selected models. Results are reported using task-specific metrics (HPS, rCLIP, FSS, Cart., Line, 3D) for Text-based Generation and Style Transfer tasks.

Text-to-SVG Style Transfer HPS↑ rCLIP↑PSS↑ Cart.↑ Line↑ 3D↑

Method

Easy Iconshop [53]★ 17.99 86.22 5.26 - - LLM4SVG [55]★ 16.76 75.78 3.01 - - DS-R1-Qwen-32B [12]★ 17.42 84.09 10.47 3.14 3.04 2.73 DeepSeek-R1 [12]★ 20.37 91.39 18.07 3.13 2.68 2.87 Qwen2.5-72B-Ins [34]★ 17.38 91.18 15.77 2.83 2.72 2.90 QwQ-32B [59]★ 19.19 89.30 16.78 3.56 3.30 3.12 Qwen3-1.7B [59]★ 16.92 79.80 10.22 1.64 3.00 1.97 Qwen3-8B [59]★ 18.15 85.53 12.73 - - Qwen3-32B [59]★ 19.39 89.13 12.62 2.93 2.92 2.80 Gemini-2.0-Flash [43]★ 19.82 90.96 15.94 3.17 3.16 3.25 GPT-4o [16]★ 20.35 90.95 19.72 3.27 2.88 2.53 Claude-3.7-Sonnet [1]★ 21.35 92.90 16.69 3.68 2.08 2.93

Medium Iconshop [53]★ 14.68 77.35 3.24 - - LLM4SVG [55]★ 14.88 68.84 2.70 - - DS-R1-Qwen-32B [12]★ 15.76 75.52 7.55 1.27 2.38 3.16 DeepSeek-R1 [12]★ 17.46 82.99 14.78 2.40 2.12 2.44 Qwen2.5-72B-Ins [34]★ 16.35 79.91 13.11 3.07 2.12 3.52 QwQ-32B [59]★ 16.73 77.87 17.56 3.00 2.31 3.48 Qwen3-1.7B [59]★ 16.00 70.66 11.97 1.87 1.89 1.84 Qwen3-8B [59]★ 16.70 80.37 11.26 - - Qwen3-32B [59]★ 17.17 80.88 14.38 3.33 1.56 2.76 Gemini-2.0-Flash [43]★ 17.00 80.36 13.70 3.13 2.13 3.00 GPT-4o [16]★ 17.51 84.72 11.97 1.67 2.00 1.56 Claude-3.7-Sonnet [1]★ 85.71 87.62 16.60 3.67 1.94 3.28

Hard Iconshop [53]★ 12.95 72.55 2.12 - - LLM4SVG [55]★ 14.62 62.98 0.02 - - DS-R1-Qwen-32B [12]★ 14.56 72.71 6.56 2.20 2.08 2.40 DeepSeek-R1 [12]★ 16.86 83.06 10.07 2.16 2.00 2.13 Qwen2.5-72B-Ins [34]★ 15.51 75.22 9.57 2.64 1.96 3.20 QwQ-32B [59]★ 16.36 82.24 8.45 1.92 1.92 2.40 Qwen3-1.7B [59]★ 14.52 68.91 4.46 1.32 2.24 1.20 Qwen3-8B [59]★ 15.34 77.47 8.64 - - Qwen3-32B [59]★ 16.02 81.84 9.88 2.96 1.60 3.33 Gemini-2.0-Flash [43]★ 16.01 78.61 9.89 1.32 1.08 1.26 GPT-4o [16]★ 16.69 82.81 10.39 1.84 1.96 2.73 Claude-3.7-Sonnet [1]★ 18.74 87.97 10.19 2.64 1.80 3.33

70-80% semantic understanding but struggle with structural synthesis (PSS scores rarely exceed 20), indicating a comprehensioncreation gap. Multimodal inputs enhance PSS scores by 5-10%, though benefits decrease on complex tasks, suggesting visual guidance supports basic structural comprehension more than intricate geometric synthesis. Style transfer emerges as the most challenging task, with only reasoning-enhanced and proprietary models achieving meaningful adaptation (scores >3.0) while others perform superficial modifications (<2.5).

Failure Mode Analysis Analysis of failure patterns reveals scaledependent weaknesses that explain performance boundaries. Small

Table 5: Performance on SVG generation dimension across different model types and difficulty levels. Results are reported using taskspecific metrics (SSIM, LPIPS, MSE, DINO, PSS) for Image-to-SVG.

Image-to-SVG SSIM↑ LPIPS↓ MSE↓ DINO↑ PSS↑

Method

Easy StarVector(8B) [37]★ 37.60 36.80 43.71 73.98 InternVL3-8B [63]★ 46.16 44.10 27.26 76.35 2.72 Qwen2.5-VL-3B-Ins [2]★ 50.66 38.91 28.32 73.92 8.15 Qwen2.5-VL-72B-Ins [2]★ 46.75 40.85 24.52 79.83 17.60 Gemini-2.0-Flash [43]★ 50.07 37.01 21.10 85.21 19.80 GPT-4o [16]★ 52.41 33.81 19.21 87.48 23.43 Claude-3.7-Sonnet [1]★ 54.02 32.12 17.13 89.91 23.70

#### Medium

StarVector(8B) [37]★ 52.95 41.20 21.63 67.28 -

- InternVL3-8B [63]★ 45.26 48.47 19.51 73.90 1.72 Qwen2.5-VL-3B-Ins [2]★ 47.50 44.60 22.85 69.33 1.93 Qwen2.5-VL-72B-Ins [2]★ 45.29 47.93 20.04 76.41 15.82 Gemini-2.0-Flash [43]★ 46.98 45.48 15.65 79.74 10.16

- GPT-4o [16]★ 49.41 42.29 15.44 81.29 12.64 Claude-3.7-Sonnet [1]★ 51.36 37.67 12.33 86.17 16.58

Hard StarVector(8B) [37]★ 56.53 42.12 16.47 65.81 InternVL3-8B [63]★ 46.83 51.39 18.48 75.05 1.77 Qwen2.5-VL-3B-Ins [2]★ 52.84 47.35 19.41 70.88 1.52 Qwen2.5-VL-72B-Ins [2]★ 42.74 49.86 17.72 75.88 11.60 Gemini-2.0-Flash [43]★ 48.29 46.38 12.54 80.21 7.30

- GPT-4o [16]★ 50.66 43.40 12.77 81.36 11.67 Claude-3.7-Sonnet [1]★ 51.15 39.59 11.23 85.51 15.47

models (<7B) exhibit fundamental syntactic failures, with Qwen31.7B [59] achieving only 22.34% bug fixing accuracy and 0% on hard tasks. Medium models (7-30B) demonstrate semantic limitations, excelling at local edits but struggling with global manipulations as evidenced by Qwen3-8B’s [59] inconsistent performance (Style Editing: 87.34% vs Bug Fixing: 53.00%). Large models show improved global structural understanding but suffer from style abstraction failures, with even top performers like Claude [1] achieving only 2-4 range scores in style transfer. This progression from syntactic to semantic to abstraction failures indicates that SVG processing requires hierarchical skill development that current training approaches only partially address.

## 6 Conclusion

We introduce SVGenius, the first comprehensive benchmark for evaluating LLMs capabilities in SVG processing, comprising 2,377 queries across understanding, editing, and generation with systematic complexity stratification. Evaluation of 22 models reveals that while proprietary models outperform open-source counterparts, all models degrade with increasing complexity, and reasoningenhanced training proves more effective than pure scaling. These findings indicate fundamental limitations in current approaches and suggest specialized training methods that better capture vector graphics’ structured nature. SVGenius establishes a foundation for advancing SVG processing research and automated graphic design.

## References

- [1] Anthropic. 2023. Claude 2. https://www.anthropic.com/index/claude-2. Accessed: 2025-05-29.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025).
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- [4] Mu Cai, Zeyi Huang, Yuheng Li, Utkarsh Ojha, Haohan Wang, and Yong Jae Lee. 2023. Leveraging large language models for scalable vector graphics-driven image understanding. arXiv preprint arXiv:2306.06094 (2023).
- [5] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. 2020. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems 33 (2020), 16351–16361.
- [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374 (2021).
- [7] Zehao Chen and Rong Pan. 2025. SVGBuilder: Component-Based Colored SVG Generation with Text-Guided Autoregressive Transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 2358–2366.
- [8] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research 24, 240 (2023), 1–113.
- [9] Louis Clouâtre and Marc Demers. 2019. Figr: Few-shot image generation with reptile. arXiv preprint arXiv:1901.02199 (2019).
- [10] SVG Contributors. 2024. SVGO: Node.js tool for optimizing SVG files. https: //github.com/svg/svgo Accessed: 2024-12-XX.
- [11] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783

(2024).

- [12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).
- [13] David Ha and Douglas Eck. 2017. A neural representation of sketch drawings. arXiv preprint arXiv:1704.03477 (2017).
- [14] Teng Hu, Ran Yi, Baihong Qian, Jiangning Zhang, Paul L Rosin, and Yu-Kun Lai. 2024. Supersvg: Superpixel-based scalable vector graphics synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 24892–24901.
- [15] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186 (2024).
- [16] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [17] IconFont. 2024. IconFont - Alibaba Vector Icon Library. https://www.iconfont.cn/. Accessed: 2025-05-29.
- [18] Ghfran Jabour, Sergey Muravyov, and Valeria Efimova. 2025. Layerwise Image Vectorization via Bayesain-Optimized Contour. Proceedings Copyright 831 (2025), 838.
- [19] Ajay Jain, Amber Xie, and Pieter Abbeel. 2023. Vectorfusion: Text-to-svg by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1911–1920.
- [20] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770 (2023).
- [21] Gueyoung Jung, Matti A Hiltunen, Kaustubh R Joshi, Richard D Schlichting, and Calton Pu. 2010. Mistral: Dynamically managing power, performance, and adaptation cost in cloud infrastructures. In 2010 IEEE 30th International Conference on Distributed Computing Systems. IEEE, 62–73.
- [22] Claire Le Goues, Neal Holtschulte, Edward K Smith, Yuriy Brun, Premkumar Devanbu, Stephanie Forrest, and Westley Weimer. 2015. The ManyBugs and IntroClass benchmarks for automated repair of C programs. IEEE Transactions on Software Engineering 41, 12 (2015), 1236–1256.
- [23] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning. PMLR, 12888–12900.
- [24] Tzu-Mao Li, Michal Lukáč, Michaël Gharbi, and Jonathan Ragan-Kelley. 2020. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG) 39, 6 (2020), 1–15.

- [25] Derrick Lin, James Koppel, Angela Chen, and Armando Solar-Lezama. 2017. QuixBugs: A multi-lingual program repair benchmark set based on the Quixey Challenge. In Proceedings Companion of the 2017 ACM SIGPLAN international conference on systems, programming, languages, and applications: software for humanity. 55–56.
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.
- [27] Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. 2019. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7930–7939.
- [28] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664 (2021).
- [29] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. 2022. Towards layer-wise image vectorization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16314– 16323.
- [30] Kunato Nishina and Yusuke Matsui. 2024. SVGEditBench: A Benchmark Dataset for Quantitative Assessment of LLM’s SVG Editing Capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8142–8147.
- [31] Kunato Nishina and Yusuke Matsui. 2025. SVGEditBench V2: A Benchmark for Instruction-based SVG Editing. arXiv preprint arXiv:2502.19453 (2025).
- [32] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin ElNouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).
- [33] Zeju Qiu, Weiyang Liu, Haiwen Feng, Zhen Liu, Tim Z Xiao, Katherine M Collins, Joshua B Tenenbaum, Adrian Weller, Michael J Black, and Bernhard Schölkopf.

2024. Can Large Language Models Understand Symbolic Graphics Programs? arXiv preprint arXiv:2408.08313 (2024).

- [34] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. (2025). arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.
- [36] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. 2021. Im2vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7342–7351.
- [37] Juan A Rodriguez, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, David Vazquez, Christopher Pal, and Marco Pedersoli. 2023. Starvector: Generating scalable vector graphics code from images. arXiv preprint arXiv:2312.11556 (2023).
- [38] Christoph Schuhmann. 2022. Improved Aesthetic Predictor. https://github.com/ christophschuhmann/improved-aesthetic-predictor. Accessed: 2025-05-29.
- [39] I-Chao Shen and Bing-Yu Chen. 2021. Clipgen: A deep generative model for clipart vectorization and synthesis. IEEE Transactions on Visualization and Computer Graphics 28, 12 (2021), 4211–4224.
- [40] Yiren Song, Xuning Shao, Kang Chen, Weidong Zhang, Zhongliang Jing, and Minzhe Li. 2023. Clipvg: Text-guided image manipulation using differentiable vector graphics. In Proceedings of the AAAI conference on artificial intelligence, Vol. 37. 2312–2320.
- [41] Hao Su, Xuefeng Liu, Jianwei Niu, Jiahe Cui, Ji Wan, Xinghao Wu, and Nana Wang. 2023. Marvel: Raster gray-level manga vectorization via primitive-wise deep reinforcement learning. IEEE Transactions on Circuits and Systems for Video Technology 34, 4 (2023), 2677–2693.
- [42] Zecheng Tang, Chenfei Wu, Zekai Zhang, Mingheng Ni, Shengming Yin, Yu Liu, Zhengyuan Yang, Lijuan Wang, Zicheng Liu, Juntao Li, et al. 2024. Strokenuwa: Tokenizing strokes for vector graphic synthesis. arXiv preprint arXiv:2401.17093

(2024).

- [43] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
- [44] Vikas Thamizharasan, Difan Liu, Matthew Fisher, Nanxuan Zhao, Evangelos Kalogerakis, and Michal Lukac. 2024. Nivel: Neural implicit vector layers for text-to-vector generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4589–4597.
- [45] Lucas Theis, Aäron van den Oord, and Matthias Bethge. 2015. A note on the evaluation of generative models. arXiv preprint arXiv:1511.01844 (2015).

- [46] Yingtao Tian and David Ha. 2022. Modern evolution strategies for creativity: Fitting concrete images and abstract concepts. In International conference on computational intelligence in music, sound, art and design (part of evostar). Springer, 275–291.
- [47] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
- [48] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of machine learning research 9, 11 (2008).
- [49] Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir. 2022. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–11.
- [50] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. 2023. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. Advances in Neural Information Processing Systems 36 (2023), 61501–61513.
- [51] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13, 4 (2004), 600–612.
- [52] Ronghuan Wu, Wanchao Su, and Jing Liao. 2024. Chat2SVG: Vector Graphics Generation with Large Language Models and Image Diffusion Models. arXiv preprint arXiv:2411.16602 (2024).
- [53] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. 2023. Iconshop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics (TOG) 42, 6 (2023), 1–14.
- [54] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. 2023. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2096– 2105.
- [55] Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang, Dong Xu, and Qian Yu.

2024. Empowering LLMs to Understand and Generate Complex Vector Graphics. arXiv preprint arXiv:2412.11102 (2024).

- [56] Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu.

2023. Diffsketcher: Text guided vector sketch synthesis through latent diffusion models. Advances in Neural Information Processing Systems 36 (2023), 15869– 15889.

- [57] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. 2024. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4546–4555.
- [58] Zhongzheng Xu and Emily Wall. 2024. Exploring the capability of llms in performing low-level visual analytic tasks on svg data visualizations. In 2024 IEEE Visualization and Visual Analytics (VIS). IEEE, 126–130.
- [59] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).
- [60] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Jiaxu Zhang, Liao Wang, Gang Yu, Xingjun Ma, and Yu-Gang Jiang. 2025. OmniSVG: A Unified Scalable Vector Graphics Generation Model. arXiv preprint arXiv:2504.06263 (2025).
- [61] Peiying Zhang, Nanxuan Zhao, and Jing Liao. 2024. Text-to-vector generation with neural path representation. ACM Transactions on Graphics (TOG) 43, 4

(2024), 1–13.

- [62] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang.

2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

- [63] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. 2025. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025).
- [64] Bocheng Zou, Mu Cai, Jianrui Zhang, and Yong Jae Lee. 2024. Vgbench: Evaluating large language models on vector graphics understanding and generation. arXiv preprint arXiv:2407.10972.

## A Overview of SVGenius

SVGenius incorporates samples from IconFont [17], a real-world vector icon repository, to ensure comprehensive coverage of SVG usage patterns and semantic contexts across diverse domains. Our dataset spans six major thematic clusters including Digital Products (encompassing UI/UX design, e-commerce, and web applications), Professional services (including business, finance, legal, and healthcare sectors), Lifestyle categories (covering food, sports, entertainment, and personal items), Work & Systems (featuring tools, time management, and security applications), Environmental domains (incorporating nature, science, and sustainability themes), and Smart Home technologies (including IoT devices and home automation systems). This comprehensive domain architecture is visualized through a word cloud representation of vector icon captions in Figure 3.

[Figure 70]

Figure 3: Word cloud visualization of the 24 diverse application domains covered in SVGenius dataset.

## B Dataset Quality Assessment B.1 Instance-level Complexity Metrics

Our instance-level complexity quantification employs automated SVG parsing to extract fundamental structural metrics. We compute path count through XML parsing, numerical parameter count via regular expression matching, and complex command frequency by identifying curve-type operations (C, Q, A, S, T variants). Additional metrics include total command count, average commands per path, and file size in bytes to capture geometric density and syntactic complexity. Dataset complexity scores are presented in Table 6.

Table 6: Instance-level complexity metrics

Level Paths Commands Points Complex Commands Cmd/Path File Size (KB)

Easy 2.14 35.84 132.7 12.03 21.39 0.96 Medium 9.87 150.93 739.56 56.01 27.68 4.83 Hard 16.02 228.87 1148.67 79.94 32.57 7.63

## B.2 Structure-Aware Complexity Modeling

To systematically stratify SVG samples across hierarchical complexity levels, we implement a multi-dimensional scoring framework based on weighted command analysis and advanced structural metrics. Each SVG path command receives complexity weights reflecting computational demands: basic operations (M, L, Z) weighted at 1, quadratic curves (Q, T) at 2-3, cubic curves (C, S) at 3-4, and elliptical arcs (A) at 5. We further compute command type entropy, coordinate precision entropy, and curvature complexity. Final stratification integrates these orthogonal measures through principal component analysis, enabling robust classification into three hierarchical levels that demonstrate statistically significant differences (ANOVA F-tests, p < 0.001) across multiple complexity dimensions.Dataset complexity scores are presented in Table 7.

Table 7: Complexity metrics across hierarchical levels

Level Curvature Structure Type Entropy

Easy 0.70 3.58 1.48 Medium 3.56 15.09 1.67 Hard 5.46 22.89 1.68

## B.3 Manual Assessment

To systematically evaluate the quality of our dataset, we conducted a user study involving 10 volunteer participants. As shown in Figure 4, the assessment focused on the following three key dimensions:

- • Visual Quality: The aesthetic appeal and clarity of the SVG graphics, including factors such as compositional balance, rendering fidelity, and visual coherence.
- • SemanticAlignment: Theconsistency betweeneachSVG graphic and its associated image description, evaluating whether the semantic content of the text accurately reflects the visual content.
- • Difficulty Stratification: The appropriateness and discriminative power of the assigned difficulty levels, assessing whether the difficulty annotations reflect meaningful variation in task complexity.

This study was designed to provide a holistic evaluation of the dataset’s validity across visual presentation, semantic alignment, and difficulty calibration, ensuring that it meets the standards required for reliable model evaluation.

## C Full Leaderboard

For completeness and reproducibility, we provide the full versions of the tables referenced in the main text. Due to space constraints, only abridged versions of these tables were included in the body of the paper. Table 8, Table 9, Table 10 and Table 11 presented more detailed statistics and results, which may be of interest to researchers seeking to replicate or analyze our findings further.

## D Path-Structure Similarity Score

We propose a novel evaluation metric to evaluate the quality of SVG image generation, termed the Path-Structure Similarity Score (PSS). Unlike conventional image quality metrics such as MSE,

- Table 8: Performance on SVG understanding dimension across different model types and difficulty levels. Accuracy scores are shown for Perceptual QA and Semantic QA tasks, with models marked as reasoning (★), code (★), open-source (★), or proprietary (★) variants.

Perceptual QA(ACC↑)(%) Semantic QA(ACC↑)(%) Easy Medium Hard Easy Medium Hard

Method

DeepSeek-R1-Distill-Qwen-1.5B [12]★ 43.01 26.92 17.78 31.18 20.51 26.67 DeepSeek-R1-Distill-Qwen-7B [12]★ 53.09 32.89 35.56 34.57 39.47 28.89 DS-R1-Qwen-32B [12]★ 64.20 43.42 42.22 51.85 52.63 37.78 DeepSeek-R1 [12]★ 74.19 55.13 44.44 74.19 71.79 55.56 Llama3.2-3B-Instruct [11]★ 47.31 26.92 26.67 44.09 43.59 46.67 Mistral-Small-3.1-24B-Instruct [21]★ 62.37 34.62 28.89 54.84 62.82 44.44 Qwen2.5-Coder-3B [15]★ 24.73 29.49 20.00 26.88 20.51 15.56 Qwen2.5-3B-Instruct [34]★ 43.01 37.18 26.67 46.24 46.15 46.67 Qwen2.5-72B-Ins [34]★ 67.74 38.46 24.44 50.54 47.43 51.11 QwQ-32B [59]★ 62.37 34.62 33.33 53.76 57.69 37.78 Qwen3-1.7B [59]★ 46.91 22.37 28.89 41.98 44.74 22.22 Qwen3-8B [59]★ 70.96 43.59 22.22 44.09 46.15 42.22 Qwen3-32B [59]★ 71.60 42.11 24.44 60.49 55.26 42.22 Gemini-2.0-Flash [43]★ 77.78 40.79 31.11 62.96 55.26 51.11 GPT-4o [16]★ 82.72 35.53 42.22 67.90 56.58 64.44 Claude-3.7-Sonnet [1]★ 80.25 47.37 33.33 77.78 65.79 71.11

SSIM [51], or FID [45], which rely primarily on pixel-level comparisons, our approach leverages the structural properties of SVG, an XML-based vector graphics format, to capture both visual fidelity and underlying XML code consistency. SVG representations are not merely visual outputs; they are composed of structured commands, attribute sets, and rendering logic that encode information often missed by purely perceptual metrics.

Our metric incorporates both macroscopic visual alignment and microscopic structural correspondence. At the macroscopic level, we compute a global Intersection-over-Union (IoU) score to quantify the overlap of visual shapes. At the microscopic level, we perform a comprehensive analysis of SVG code. This begins with preprocessing steps such as center alignment and scale normalization, followed by path extraction and the construction of a multi-dimensional similarity matrix. The matrix integrates several components, including path-wise IoU, similarity to color attribute, and spatial relationship metrics. We employ the Hungarian algorithm to perform optimal path-level matching, ensuring robust structural alignment.Beyond direct path matching, we further introduce two structural consistency modules: path count consistency and path order scoring. The latter evaluates the semantic fidelity of rendering by assessing the alignment of drawing order, which in SVG inherently determines visual stacking and rendering priority.

The final PSS score is computed through weighted summation, where the macroscopic visual alignment score receives a weight of 0.6 and the microscopic structural correspondence score receives a weight of 0.4, balancing the importance of overall visual quality and structural precision.

## E Generation E.1 Text-to-SVG Generation

Construction Details. In the Text-to-SVG Generation task, each evaluation sample is constructed as a paired instance of a natural language caption and its corresponding SVG. To ensure that the

textual descriptions exhibit high semantic coverage and expressive fidelity, we employ a semi-automated pipeline as follows.

First, we render the original SVG into a high-resolution raster image using the Cairosvg library. The rendered image is then passed to the Claude-3.7-sonnet [1] model along with a minimal prompt, for example, "Describe this image in brief". The model generates an initial image caption that is then manually reviewed and refined. This human-in-the-loop step ensures that each caption accurately captures the core visual elements of the graphic, including structural layout, component composition, color attributes, and stylistic traits. To maintain quality, outputs exhibiting semantic ambiguity, structural inaccuracies, or excessive token length are systematically filtered.

This hybrid pipeline combines the scalability of automatic generation with the precision of manual validation, resulting in a highquality dataset of paired captions and SVGs suitable for model evaluation.

Evaluation Metrics. To systematically evaluate the performance of models on the Text-to-SVG task, we propose a three-dimensional evaluation framework encompassing perceptual quality, visual reconstructability, and semantic alignment:

Perceptual Quality. This dimension assesses subjective aesthetic appeal and user-level acceptability of generated SVG. We adopt two metrics:

- • Aesthetic Score [38]: Computed using a pre-trained image aesthetics model, this score evaluates the visual pleasantness of the rendered SVG image.
- • Human Preference Score (HPS) [54]: Derived from a series of A/B testing rounds, this metric captures human preferences by comparing model-generated SVGs against reference graphics. The final score reflects the frequency with which participants favor the generated output.

- Table 9: Performance on SVG editing dimension across different model types and difficulty levels. Results are reported using task-specific metrics (ACC, rMSE, MSE, RLD, CCR) for Bug Fixing, Style Editing and Code Optimization tasks, with models marked as specialized (★), reasoning (★), code (★), open-source (★), or proprietary (★) variants.

Bug Fixing Style Editing Code Optimization ACC↑ ACC↑ rMSE↑ RLD↓ MSE↓ CCR↑

Method

#### Easy

DeepSeek-R1-Distill-Qwen-1.5B [12]★ 3.85 56.06 28.29 234.41 11.50 13.04 DeepSeek-R1-Distill-Qwen-7B [12]★ 3.80 69.69 54.51 136.09 15.56 18.35 DS-R1-Dis-Qwen-32B [12]★ 62.63 84.81 75.16 1.45 5.63 23.16 DeepSeek-R1 [12]★ 71.00 84.62 73.66 18.21 1.01 23.67 Llama3.2-3B-Instruct [11]★ 33.70 83.54 57.04 82.61 7.96 19.83 Mistral-Small-3.1-24B-Instruct [21]★ 54.00 74.36 62.39 25.84 12.05 23.51 Qwen2.5-Coder-3B [15]★ 9.43 56.82 39.55 136.92 17.29 20.90 Qwen2.5-3B-Instruct [34]★ 36.46 70.51 60.99 126.06 13.13 13.41 Qwen2.5-72B-Ins [34]★ 71.00 75.95 64.25 2.08 2.98 20.57 QwQ-32B [59]★ 71.43 91.14 81.70 2.61 3.20 20.20 Qwen3-1.7B [59]★ 22.34 74.36 67.22 33.40 11.43 35.79 Qwen3-8B [59]★ 53.00 87.34 80.40 221.88 2.84 18.61 Qwen3-32B [59]★ 56.12 88.46 82.63 1.17 2.55 17.96 Gemini-2.0-Flash [43]★ 69.00 86.07 79.70 4.78 0.72 20.30 GPT-4o [16]★ 74.00 78.48 65.81 46.53 1.30 10.57 Claude-3.7-Sonnet [1]★ 76.00 79.75 67.17 20.89 0.31 16.81

#### Medium

DeepSeek-R1-Distill-Qwen-1.5B [12]★ 0.00 12.00 9.56 4757.67 14.73 47.38 DeepSeek-R1-Distill-Qwen-7B [12]★ 0.00 36.36 36.33 5811.5 19.05 44.55 DS-R1-Dis-Qwen-32B [12]★ 46.00 56.41 53.11 140.68 8.14 33.10 DeepSeek-R1 [12]★ 63.83 62.16 53.93 1227.70 2.02 36.70 Llama3.2-3B-Instruct [11]★ 4.60 50.67 37.93 442.66 5.33 15.53 Mistral-Small-3.1-24B-Instruct [21]★ 16.85 37.88 32.46 742.92 14.50 46.48 Qwen2.5-Coder-3B [15]★ 0.00 37.00 29.96 3992.66 12.85 79.18 Qwen2.5-3B-Instruct [34]★ 4.30 34.33 27.11 1616.04 8.42 19.81 Qwen2.5-72B-Ins [34]★ 50.56 59.49 51.90 136.00 3.41 30.49 QwQ-32B [59]★ 46.00 64.56 61.27 41.67 5.03 29.28 Qwen3-1.7B [59]★ 1.22 33.90 25.55 1656.50 10.20 34.87 Qwen3-8B [59]★ 35.42 62.34 52.10 826.21 2.63 21.69 Qwen3-32B [59]★ 46.47 66.67 62.54 17.13 2.78 26.61 Gemini-2.0-Flash [43]★ 60.00 65.82 60.43 113.44 1.59 27.49 GPT-4o [16]★ 49.00 50.63 42.64 840.73 4.74 43.01 Claude-3.7-Sonnet [1]★ 75.00 59.74 53.61 134.52 0.86 26.41

Hard DeepSeek-R1-Distill-Qwen-1.5B [12]★ 0.00 28.57 6933.75 352.85 12.38 57.41 DeepSeek-R1-Distill-Qwen-7B [12]★ 0.00 42.86 42.85 11559.67 8.80 61.35 DS-R1-Dis-Qwen-32B [12]★ 34.02 55.26 41.68 226.50 5.89 36.41 Deepseek-R1 [12]★ 61.80 51.56 41.83 2610.48 2.41 39.95 Llama3.2-3B-Instruct [11]★ 1.37 40.35 24.03 3631.91 5.34 9.52 mistral-small-3.1-24b-instruct [21]★ 3.95 25.86 20.25 2247.67 11.99 55.31 Qwen2.5-coder-3B [15]★ 0.00 36.36 32.81 7347.00 12.39 82.32 Qwen2.5-3B-Instruct [34]★ 0.00 24.53 15.90 4122.69 4.94 20.76 Qwen2.5-72B-Ins [34]★ 40.00 54.67 43.80 722.00 3.24 36.40 QwQ-32B [59]★ 10.42 50.00 44.04 533.85 5.82 12.98 Qwen3-1.7B [59]★ 0.00 26.42 21.94 3650.71 5.60 32.42 Qwen3-8B [59]★ 25.30 60.81 52.21 935.13 2.15 23.68 Qwen3-32B [59]★ 40.86 55.26 48.06 58.79 1.50 26.75 Gemini-2.0-Flash [43]★ 53.95 57.38 42.34 628.14 2.62 29.35 GPT-4o [16]★ 51.02 42.67 35.43 843.81 5.14 46.91 Claude-3.7-Sonnet [1]★ 69.00 52.56 40.75 27.00 0.88 28.27

metrics (CLIP, AES, HPS, rCLIP, PSS, Cart., Pixel, Line, 3D) for Text-based Generation and Style Transfer tasks, with models marked as specialized (★), reasoning (★), code (★), open-source (★), or proprietary (★) variants.

Text-based Generation Style Transfer CLIP↑ AES↑ HPS↑ rCLIP↑ PSS↑ Cart. Pixel Line 3D

Method

Easy Iconshop [53]★ 22.74 3.56 17.99 86.22 5.26 - - - LLM4SVG(GPT-2 XL) [55]★ 18.09 3.63 16.76 75.78 3.01 - - - DeepSeek-R1-Distill-Qwen-1.5B [12]★ 18.43 3.55 16.51 73.51 0.79 1.27 1.25 0.36 1.30 DeepSeek-R1-Distill-Qwen-7B [12]★ 19.27 3.50 16.71 76.76 0.93 0.77 0.65 1.60 0.90 DS-R1-Qwen-32B [12]★ 22.04 3.47 17.42 84.09 10.47 3.14 1.55 3.04 2.73 DeepSeek-R1 [12]★ 24.34 3.61 20.37 91.39 18.07 3.13 1.70 2.68 2.87 Mistral-Small-3.1-24B-Instruct [21]★ 21.15 3.25 16.64 81.81 8.31 2.07 1.65 2.28 1.30 Qwen2.5-Coder-3B [15]★ 19.23 3.22 14.97 74.08 1.32 - - - Qwen2.5-3B-Instruct [34]★ 20.03 3.19 15.81 78.25 2.33 1.70 1.87 1.68 2.17 Qwen2.5-72B-Ins [34]★ 20.86 3.59 17.38 91.18 15.77 2.83 1.85 2.72 2.90 QwQ-32B [59]★ 23.01 3.51 19.19 89.30 16.78 3.56 2.24 3.30 3.12 Qwen3-1.7B [59]★ 20.37 3.50 16.92 79.80 10.22 1.64 2.35 3.00 1.97 Qwen3-8B [59]★ 22.39 3.48 18.15 85.53 12.73 - - - Qwen3-32B [59]★ 23.81 3.48 19.39 89.13 12.62 2.93 2.10 2.92 2.80 Gemini-2.0-Flash [43]★ 24.20 3.70 19.82 90.96 15.94 3.17 2.32 3.16 3.25 GPT-4o [16]★ 24.97 3.63 20.35 90.95 19.72 3.27 1.95 2.88 2.53 Claude-3.7-Sonnet [1]★ 25.11 3.96 21.35 92.90 16.69 3.68 2.00 2.08 2.93

Medium Iconshop [53]★ 20.76 3.46 14.68 77.35 3.24 - - - LLM4SVG(GPT-2 XL) [55]★ 17.98 3.55 14.88 68.84 2.70 - - - DeepSeek-R1-Distill-Qwen-1.5B [12]★ 17.23 3.49 14.99 66.43 0.23 1.13 0.53 0.00 0.00 DeepSeek-R1-Distill-Qwen-7B [12]★ 18.21 3.53 15.00 69.20 0.41 0.00 0.65 0.62 0.52 DS-R1-Qwen-32B [12]★ 20.04 3.52 15.76 75.52 7.55 1.27 1.43 2.38 3.16 DeepSeek-R1 [12]★ 22.54 3.55 17.46 82.99 14.78 2.40 1.82 2.12 2.44 Mistral-Small-3.1-24B-Instruct [21]★ 20.09 3.32 15.34 75.40 5.39 1.80 0.68 1.38 0.00 Qwen2.5-Coder-3B [15]★ 16.75 3.38 13.48 63.74 1.89 - - - Qwen2.5-3B-Instruct [34]★ 18.08 3.32 13.40 69.46 1.47 1.00 1.18 1.48 1.12 Qwen2.5-72B-Ins [34]★ 20.40 3.50 16.35 79.91 13.11 3.07 2.13 2.12 3.52

- QwQ-32B [59]★ 21.39 3.57 16.73 77.87 17.56 3.00 1.47 2.31 3.48 Qwen3-1.7B [59]★ 19.26 3.59 16.00 70.66 11.97 1.87 2.14 1.89 1.84 Qwen3-8B [59]★ 21.65 3.45 16.70 80.37 11.26 - - - -

- Qwen3-32B [59]★ 21.80 3.51 17.17 80.88 14.38 3.33 1.67 1.56 2.76 Gemini-2.0-Flash [43]★ 21.77 3.65 17.00 80.36 13.70 3.13 1.87 2.13 3.00 GPT-4o [16]★ 23.03 3.49 17.51 84.72 11.97 1.67 1.69 2.00 1.56 Claude-3.7-Sonnet [1]★ 24.18 3.78 85.71 87.62 16.60 3.67 2.61 1.94 3.28

Hard Iconshop [53]★ 19.34 3.48 12.95 72.55 2.12 - - - LLM4SVG(GPT-2 XL) [55]★ 16.19 3.61 14.62 62.98 0.02 - - - DeepSeek-R1-Distill-Qwen-1.5B [12]★ 16.19 3.46 14.01 61.25 0.12 0.00 0.08 0.32 0.00 DeepSeek-R1-Distill-Qwen-7B [12]★ 17.51 3.48 14.10 65.33 0.25 0.24 0.35 1.04 0.00 DS-R1-Qwen-32B [12]★ 19.22 3.63 14.56 72.71 6.56 2.20 1.50 2.08 2.40 DeepSeek-R1 [12]★ 22.73 3.63 16.86 83.06 10.07 2.16 1.95 2.00 2.13 Mistral-Small-3.1-24B-Instruct [21]★ 18.61 3.33 14.35 70.60 4.11 2.07 1.65 2.28 1.30 Qwen2.5-Coder-3B [15]★ 16.07 3.35 12.50 59.91 1.31 - - - Qwen2.5-3B-Instruct [34]★ 17.38 3.30 13.31 65.59 2.08 0.96 1.15 1.24 1.20 Qwen2.5-72B-Ins [34]★ 20.08 3.65 15.51 75.22 9.57 2.64 2.00 1.96 3.20 QwQ-32B [59]★ 22.03 3.51 16.36 82.24 8.45 1.92 1.40 1.92 2.40 Qwen3-1.7B [59]★ 18.37 3.50 14.52 68.91 4.46 1.32 1.30 2.24 1.20 Qwen3-8B [59]★ 20.60 3.43 15.34 77.47 8.64 - - - -

- Qwen3-32B [59]★ 22.06 3.52 16.02 81.84 9.88 2.96 1.27 1.60 3.33 Gemini-2.0-Flash [43]★ 20.95 3.73 16.01 78.61 9.89 1.32 1.39 1.08 1.26 GPT-4o [16]★ 22.57 3.64 16.69 82.81 10.39 1.84 1.92 1.96 2.73 Claude-3.7-Sonnet [1]★ 24.54 3.96 18.74 87.97 10.19 2.64 2.36 1.80 3.33

metrics (CP, DF, SC, CH, CB) for Style Transfer tasks, with models marked as reasoning (★), open-source (★), or proprietary (★) variants.

Cartoon Style Pixel art Line art 3D style CP DF SC CH CB CP DF SC CH CB CP DF SC CH CB CP DF SC CH CB

Method

#### Easy

DeepSeek-R1-Distill-Qwen-1.5B [12]★ 1.17 0.67 0.67 1.17 2.67 0.75 0.75 0.75 1.25 2.75 0.40 0.20 0.40 0.40 0.40 1.17 1.00 0.67 1.83 1.83 DeepSeek-R1-Distill-Qwen-7B [12]★ 0.33 0.33 0.83 0.67 1.67 0.50 0.50 0.50 0.75 1.00 1.20 1.20 1.60 1.60 2.40 0.67 0.67 0.67 1.17 1.33 DS-R1-Qwen-32B [12]★ 3.17 2.67 2.67 2.17 5.00 1.25 1.25 1.25 1.75 2.25 2.80 2.60 3.00 2.40 4.40 2.83 2.00 2.33 3.00 3.50 DeepSeek-R1 [12]★ 3.67 2.00 2.83 2.67 4.50 1.50 1.25 1.25 2.00 2.50 2.20 2.20 3.00 1.80 4.20 3.17 2.33 2.33 3.00 3.50 Llama3.2-3B-Instruct [11]★ 0.83 1.00 0.83 1.33 1.83 1.75 1.75 1.50 2.25 2.75 1.00 0.80 0.40 0.60 1.40 0.83 0.67 0.67 1.17 1.33 Mistral-Small-3.1-24B-Instruct [21]★ 2.00 1.67 1.83 1.50 3.33 1.25 1.25 1.00 2.00 2.75 1.60 2.40 2.00 1.40 4.00 1.50 1.17 0.83 1.00 2.00 Qwen2.5-3B-Instruct [34]★ 1.50 1.33 1.17 1.50 3.00 1.50 1.00 1.17 2.17 3.50 1.60 1.00 1.20 2.20 2.40 2.33 1.67 1.00 2.83 3.00 Qwen2.5-72B-Ins [34]★ 3.17 2.00 1.83 2.67 4.50 1.25 1.00 1.50 2.50 3.00 2.20 2.60 2.80 1.40 4.60 3.50 2.33 2.17 2.83 3.67 QwQ-32B [59]★ 3.80 2.80 3.80 2.80 4.60 1.60 1.80 1.40 2.80 3.60 3.00 2.50 3.75 2.75 4.50 4.00 2.00 2.60 3.00 4.00

- Qwen3-1.7B [59]★ 1.17 1.00 1.17 1.67 3.17 2.00 2.00 1.75 2.50 3.50 3.00 3.00 2.80 1.80 4.40 2.67 1.50 1.00 2.17 2.50 Qwen3-4B [59]★ 3.17 2.00 2.67 2.50 4.33 2.75 3.00 2.75 3.25 3.50 1.80 2.20 2.40 0.80 3.60 3.50 2.67 2.33 3.00 3.83 Qwen3-32B [59]★ 3.17 2.33 2.67 2.50 4.00 1.50 1.50 1.75 2.75 3.00 3.60 2.80 1.80 1.80 4.60 3.00 2.33 2.67 2.83 3.17 Gemini-2.0-Flash [43]★ 3.17 2.67 3.17 2.67 4.17 1.75 1.50 1.75 2.25 4.33 3.60 3.00 2.20 2.20 4.80 3.67 2.67 3.40 3.00 3.50 GPT-4o [16]★ 3.67 2.67 2.67 2.83 4.50 1.50 1.50 1.50 2.25 3.00 3.20 2.00 3.40 1.80 4.00 3.00 2.00 1.83 2.50 3.33 Claude-3.7-Sonnet [1]★ 4.00 2.75 3.83 3.00 4.83 1.50 1.50 1.75 2.25 3.00 2.00 1.40 1.40 1.20 4.40 3.33 2.50 2.33 2.67 3.83

Medium

DeepSeek-R1-Distill-Qwen-1.5B [12]★ 0.67 0.67 0.67 1.33 2.33 0.42 0.58 0.33 0.50 0.83 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 DeepSeek-R1-Distill-Qwen-7B [12]★ 0.00 0.00 0.00 0.00 0.00 0.58 0.50 0.42 0.75 1.00 0.50 0.50 0.60 0.60 0.90 0.40 0.40 0.40 0.60 0.80

- DS-R1-Qwen-32B [12]★ 1.00 1.33 1.00 1.33 1.67 1.25 1.17 1.17 1.58 2.00 2.50 1.80 1.60 1.70 4.30 3.20 3.00 3.20 2.80 3.60 DeepSeek-R1 [12]★ 2.00 1.67 2.00 2.67 3.67 1.17 1.42 1.83 2.00 2.67 1.90 1.90 1.80 1.60 3.40 2.40 2.80 2.00 2.60 2.40 Llama3.2-3B-Instruct [11]★ 0.67 0.67 0.67 1.00 0.67 0.75 0.75 0.64 1.17 1.25 1.00 0.90 0.17 0.70 1.50 1.40 1.20 1.40 1.20 1.60

- Mistral-Small-3.1-24B-Instruct [21]★ 1.67 1.33 1.67 1.67 2.67 0.42 0.42 0.50 1.00 1.08 1.10 1.10 1.50 0.90 2.30 0.00 0.00 0.00 0.00 0.00 Qwen2.5-3B-Instruct [34]★ 0.75 0.75 1.00 1.25 1.25 1.00 0.92 0.83 1.33 1.82 1.30 1.20 1.10 1.80 2.00 0.80 0.60 0.80 1.80 1.60 Qwen2.5-72B-Ins [34]★ 3.00 2.67 2.67 3.00 4.00 2.00 1.75 1.50 2.25 3.17 1.90 1.60 2.30 1.20 3.60 3.60 2.80 3.60 3.40 4.20 QwQ-32B [59]★ 2.67 3.00 3.33 2.67 3.33 1.36 1.21 1.21 1.43 2.14 2.09 1.91 2.45 1.64 3.45 3.80 3.00 3.60 3.40 3.60

Qwen3-1.7B [59]★ 2.00 2.00 0.67 2.00 2.67 1.91 2.20 1.83 2.25 2.50 2.20 1.44 1.30 1.40 3.10 2.20 1.80 1.20 1.80 2.20 Qwen3-4B [59]★ 3.33 3.00 2.33 2.33 3.33 3.00 2.67 2.92 2.58 3.17 2.20 2.00 2.40 1.20 3.70 4.00 2.60 2.60 3.60 3.60 Qwen3-32B [59]★ 3.33 3.00 2.67 3.00 4.67 1.00 1.25 1.50 2.00 2.58 1.60 1.30 1.10 1.20 2.58 2.60 2.40 2.80 2.60 3.40 Gemini-2.0-Flash [43]★ 3.00 3.00 3.33 3.00 3.33 1.64 1.42 1.58 2.25 2.45 1.67 1.90 2.10 1.30 3.70 3.00 2.80 3.20 2.80 3.20 GPT-4o [16]★ 1.67 2.00 1.00 1.33 2.33 1.45 1.55 1.45 1.91 2.09 2.00 1.30 1.30 1.40 4.00 1.60 1.60 1.40 1.60 1.60 Claude-3.7-Sonnet [1]★ 3.00 3.00 4.00 3.33 5.00 2.50 2.33 2.58 2.83 2.83 1.60 1.40 1.30 1.50 3.90 3.60 3.00 3.00 2.80 4.00

Hard

DeepSeek-R1-Distill-Qwen-1.5B [12]★ 0.00 0.00 0.00 0.00 0.00 0.06 0.06 0.06 0.12 0.12 0.20 0.20 0.20 0.20 0.80 0.00 0.00 0.00 0.00 0.00 DeepSeek-R1-Distill-Qwen-7B [12]★ 0.20 0.20 0.20 0.20 0.40 0.19 0.19 0.25 0.50 0.62 0.60 1.00 1.40 0.60 1.60 0.00 0.00 0.00 0.00 0.00 DS-R1-Qwen-32B [12]★ 2.20 1.80 1.60 2.20 3.20 1.25 1.19 1.44 1.69 1.94 1.00 1.40 2.60 1.20 4.20 2.67 2.00 1.67 2.67 3.00 DeepSeek-R1 [12]★ 1.40 1.60 1.80 2.60 3.40 1.60 1.25 1.88 2.19 2.81 1.00 1.40 3.20 1.20 3.20 2.00 1.67 2.00 2.67 2.33 Llama3.2-3B-Instruct [11]★ 0.83 1.00 0.83 1.33 1.83 1.75 1.75 1.50 2.25 2.75 1.00 0.80 0.40 0.60 1.40 0.83 0.67 0.67 1.17 1.33

- Mistral-Small-3.1-24B-Instruct [21]★ 2.0 1.67 1.83 1.5 3.33 1.25 1.25 1.0 2.0 2.75 1.6 2.4 2.0 1.4 4.0 1.5 1.17 0.83 1.0 2.0 Qwen2.5-3B-Instruct [34]★ 0.80 0.80 0.80 0.80 1.60 0.79 0.79 0.95 1.32 1.89 1.00 1.00 1.00 1.60 1.60 1.00 1.00 1.00 1.33 1.67 Qwen2.5-72B-Ins [34]★ 3.40 2.60 1.60 2.00 3.60 1.94 1.69 1.62 2.12 2.62 1.20 1.40 2.40 1.00 3.80 3.33 3.33 2.00 3.33 4.00 QwQ-32B [59]★ 1.80 1.60 1.60 1.80 2.80 0.94 1.00 1.31 1.62 2.12 1.40 1.60 2.20 1.80 2.60 2.33 2.00 2.67 2.33 2.67 Qwen3-1.7B [59]★ 1.00 1.00 1.00 1.40 2.20 1.27 1.12 1.12 1.19 1.81 2.20 1.80 2.80 1.60 2.80 0.67 0.67 1.00 1.67 2.00 Qwen3-4B [59]★ 2.60 2.20 2.60 2.20 4.20 3.06 2.44 2.81 3.00 3.69 1.80 1.80 3.40 1.60 4.20 2.67 2.33 2.67 2.00 3.33 Qwen3-32B [59]★ 3.20 2.60 2.00 2.80 4.20 1.12 1.00 1.06 1.25 1.94 1.00 1.00 1.60 1.00 3.40 4.00 3.00 3.33 2.67 3.67 Gemini-2.0-Flash [43]★ 1.20 1.00 1.20 1.40 1.80 1.00 0.94 1.25 1.56 2.19 0.60 1.00 1.20 0.60 2.00 1.33 1.00 1.33 1.33 1.33 GPT-4o [16]★ 1.40 1.20 1.40 1.60 3.60 1.69 1.44 1.81 2.12 2.56 1.00 1.40 2.20 1.40 3.80 3.00 3.00 2.00 2.33 3.33 Claude-3.7-Sonnet [1]★ 2.60 2.40 2.20 2.20 3.80 2.25 2.06 2.06 2.44 3.00 1.00 1.40 1.60 1.20 3.80 3.67 3.00 3.00 3.00 4.00

Visual Reconstructability. This dimension evaluates how well the generated SVG replicates the structural code characteristics of the reference SVG. We design a suite of custom metrics PSS(cf. Appendix D for more details) that compare both SVG code and

rendered output across subdimensions including path topology, geometric layout, and attribute encoding.

[Figure 71]

Figure 4: Screenshot of Our Questionnaires.

Semantic Alignment. This dimension measures the degree to which the generated SVG semantically aligns with the input caption. We employ two complementary metrics:

- • CLIP Score: The generated SVG is rendered and embedded alongside the input caption using a CLIP model. The cosine similarity between their embeddings quantifies semantic alignment.
- • Relative CLIP Consistency (rCLIP): We propose this novel metric to evaluate the semantic degradation of the generated image relative to the ground truth. It is defined as:

𝑟𝐶𝐿𝐼𝑃 = 1 − max 0, 𝐶𝐿𝐼𝑃(𝑐𝑎𝑝𝑡𝑖𝑜𝑛,𝐺𝑇𝐶𝐿𝐼𝑃(𝑐𝑎𝑝𝑡𝑖𝑜𝑛,𝐺𝑇)−𝐶𝐿𝐼𝑃(𝑐𝑎𝑝𝑡𝑖𝑜𝑛,𝐺𝐸𝑁) ) (1)

Here, 𝐶𝐿𝐼𝑃(𝑐𝑎𝑝𝑡𝑖𝑜𝑛,𝐺𝑇) denotes the CLIP similarity between the inputcaptionandtheground-truth rendering, while𝐶𝐿𝐼𝑃(𝑐𝑎− 𝑝𝑡𝑖𝑜𝑛,𝐺𝐸𝑁) measures similarity with the model-generated output. Higher rCLIP values indicate stronger semantic preservation.

### Prompt for Text-based Generation

You are a professional SVG designer with extensive experience in vector graphics creation. Please generate the corresponding SVG code based on the user’s description. Provide your answer strictly in the following format: Answer: {SVG code}, providing the complete fixed code only in the SVG code position, without adding any explanations, comments, or other content. Important Notes: Your SVG must ONLY include <path> elements with "fill" and "d" attributes. Do not use any other SVG elements or attributes,and must use exactly this opening tag: <svg class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg">

Please generate an SVG code of: {prompt}

## E.2 Image-to-SVG Generation

Construction Details. In the Image-to-SVG Generation task, we aim to evaluate a model’s capability to generate structured SVG conditioned on both image and textual inputs. Each evaluation sample comprises three components: (1) a ground-truth SVG file, (2) its corresponding rendered raster image (generated via the CairoSVG library), and (3) a paired natural language description. The imagecaption pairs are constructed following the same pipeline as in the Text-to-SVG task.

By leveraging both visual and textual modalities, this task simulates realistic usage scenarios in which users express design intent through a combination of imagery and language. Consequently, it imposes a more challenging multimodal grounding requirement on generative models.

Evaluation Metrics. To comprehensively assess the performance of multimodal large language models (MLLMs) on this task, we propose two complementary categories of evaluation metrics, each targeting a distinct aspect of generation quality: perceptual similarity and visual reconstructability.

Perceptual Similarity These metrics quantify the visual fidelity of the generated SVGs relative to the reference graphics, both globally and locally:

- • Learned Perceptual Image Patch Similarity (LPIPS) [62]: Measures perceptual differences in deep feature space, capturing consistency in local texture and edge structures.
- • Structural Similarity Index Measure (SSIM) [51]: Evaluates similarity in terms of luminance, contrast, and structural alignment, with an emphasis on holistic visual reconstruction.

- • DINO Score [32]: Leverages semantic embeddings from the selfsupervised DINO model to assess structural alignment between generated and reference images, particularly suited to evaluating contour preservation and compositional fidelity.

Visual Reconstructability Given the structural and syntactic nature of SVG graphics, we further introduce metrics that analyze both image-level and code-level alignment:

- • MSE: Computes pixel-wise differences between the rendered outputs of generated and reference SVGs within a spatially aligned viewport, incorporating boundary-aware weighting to emphasize contour fidelity.
- • Path-Structure Similarity Score (PSS): Extracts and compares SVG path commands, attributes, and hierarchical organization to assess fine-grained geometric and syntactic alignment. (cf. Appendix D for more details)

Together, these perceptual and structural metrics provide a robust and reproducible evaluation framework for multimodal generative tasks, capturing both the visual plausibility and syntactic correctness of generated SVGs.

### Prompt for Image-based Generation

You are a professional SVG designer with extensive experience in vector graphics creation. Please generate the corresponding SVG code based on the user’s description and the provided image reference. Provide your answer strictly in the following format: Answer: {SVG code}, providing the complete fixed code only in the SVG code position, without adding any explanations, comments, or other content. Important Notes: Your SVG must ONLY include <path> elements with "fill" and "d" attributes. Do not use any other SVG elements or attributes,and must use exactly this opening tag: svg class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg">

Please generate an SVG code based on this description: {prompt} and the reference image I’ve provided.

## E.3 Style Transfer

Construction Details. In the Style Transfer Generation task, we aim to evaluate the model’s ability to perform cross-style transformations of SVG graphics while preserving both the underlying structure and semantic content. Given the context-length limitations of large multimodal models when processing long SVG sequences, we prioritize the selection of compact and semantically clear SVG samples during dataset construction. This ensures that the transformation process focuses on stylistic expression rather than structural reconstruction.

We define four representative target styles that span key axes of variation across mainstream visual aesthetics:

- • 3D Style: Enhances depth and lighting effects, assessing the model’s ability to abstract and reconstruct complex rendering semantics.
- • Line art: Emphasizes outlines and linear representations, requiring the model to simplify visual elements while retaining structural integrity.
- • Pixel art: Mimics low-resolution rasterized appearances, testing the model’s capacity for detail compression under strict structural constraints.
- • Cartoon Style: Introduces dynamic and emotive visual semantics, such as cartoonish or anthropomorphic traits, highlighting the model’s ability in affective style transfer.

To increase both the discriminability and difficulty of the Style transfer task, each sample is manually assigned a target style that is maximally divergent from the original SVG style. For instance, icon-like SVGs with well-defined geometric outlines are more often mapped to Pixel or Cartoon styles, while graphics involving shading or gradients are preferentially assigned to the Line art style. This strategy maximizes the stylistic shift in each task instance, thus enhancing the challenge for models and the effectiveness of evaluation.

Each task instance is composed of three elements: (1) an original SVG graphic, (2) a target style label (chosen from the four defined categories), and (3) a natural language instruction (e.g., please convert this graphic to pixel art style). This formulation not only provides a consistent prompt format, but also enforces the requirement that the output preserves semantic structure while completing the stylistic transformation, facilitating clearly aligned downstream evaluation.

Evaluation Metrics. We develop a two-tier automated assessment framework leveraging LLMs to quantify transfer quality from global and local perspectives.

Global Ranking Evaluation. Based on the results of the multidimensional evaluation, as shown in Table 12 we identify six highperforming models for subsequent holistic assessment: the proprietary models Claude-3.7-Sonnet [1], Gemini-2.0-Flash [43], and GPT-4o [16], along with the open-source models DeepSeek-R1 [12], QwQ-32B [59], and Qwen3-32B [59].

Table 12: Comparison of Overall Rank Evaluation

Models Winrate(%) Claude-3.7-Sonnet [1] 61.54

DeepSeek-R1(Reference Model) [12] 50.55 Gemini-2.0-Flash [43] 49.28 QwQ-32B [59] 47.06

GPT-4o [16] 45.45 Qwen3-32B [59] 40.26

The final evaluation focuses on three criteria: semantic preservation, stylistic fidelity, and visual quality. We employ the AlpacaEval framework using DeepSeek-R1 [12] as the reference model. The outputs of each competing model are compared against the reference model’s results across all samples. As shown in Table 12, the final rankings are derived from aggregated win–loss statistics, sorted by win rate relative to the baseline.

Local Automated Multi-dimensional Evaluation. The stylized SVG outputs are first rendered into image format and subsequently scored on a scale of 1–5 by an evaluation model (with a score of 0 assigned to invalid or erroneous generations). We design five metric including Content Preservation (CP), Detail Fidelity (DF), Style Consistency (SC), Color Harmony (CH) and Composition Balance (CB). More details are shown in Table 13. For each dimension of evaluation, we define qualitative descriptors that correspond to each level of score. We adopt GPT-4o-mini [16] as the evaluation model, which references the original image, the stylized image, and the descriptor texts to first generate detailed feedback and then assign a final score. To ensure consistency of evaluation, the model operates with a deterministic temperature setting of 0.05.

### Prompt for Automatic Evaluation of Style Transfer

You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance.

Task Description: An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.

- 1. Make a brief description of the style transfer that how it modifies the image 1 to image 2.
- 2. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
- 3. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
- 4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
- 5. Please do not generate any other opening, closing, and explanations.

The instruction to evaluate: transfer the provided Image 1 to {style}

Response to evaluate: the given Image 2

Score Rubrics: {rubric}

Feedback:

### Prompt for Style Transfer

You are a professional SVG designer with extensive experience in vector graphics creation. Your task is to perform a style transfer - recreate the provided reference SVG. Maintain the basic structure and given semantic description of the SVG, but adjust it to the desired style. Provide your answer strictly in the following format: Answer:{SVG Code}, without adding any explanations, comments, or other content.

Reference SVG to transform: {reference_svg}

Description of the reference SVG: {description}

the style to transfer: {style}

## F Understanding

Construction Details. To systematically evaluate the cognitive capabilities of large vision-language models (VLMs) on SVG-based graphics, we propose an understanding task comprising two subtasks:PerceptualUnderstandingand Semantic Understanding. Given the current limitations of multimodal models in directly parsing SVG code, we adopt an indirect approach. Specifically, raw SVGs are rendered into standard bitmap images using CairoSVG, which are then fed into the Qwen2.5-VL-72B-Instruct model for automatic generation of multi-choice question-answer (QA) samples.

Perceptual QA focuses on low-level visual features of the graphic, including shape recognition, color identification, and relative spatial positioning. Representative questions include: "Which of the following basic shapes is present in the image?", "What is the dominant color of the figure?", or "In which direction is the ellipse located relative to the center?"

Semantic QA targets higher-level semantic interpretation, such as the meaning, functionality, or plausible real-world use of the graphic. Example questions include: "Which of the following scenarios is the image most likely to represent?" or "Which software interface might use this icon?"

To ensure the quality of the QA samples, all generated questions and answer options undergo manual validation. This includes filtering out ambiguous phrasing and reconstructing distractors to guarantee that each question has a single correct answer and well-separated alternatives.

Evaluation Metric. We adopt accuracy as the sole evaluation metric for the Understanding Task. For each multiple-choice question, the model is required to select exactly one correct answer. Accuracy is computed as the ratio of correctly answered questions to the total number of questions within each subtask. This metric offers a straightforward and interpretable means of assessing model performance on both perceptual and semantic dimensions of visual understanding.

Table 13: Details of Score Rubrics used in Automatic Evaluation

criteria description Content Preservation: To what extent does the SVG conversion preserve the basic content and main elements of the original image?

- score1 description Essential elements are missing or severely distorted; core content is unrecognizable.
- score2 description Key elements are present but with noticeable omissions or alterations.
- score3 description Most main elements are preserved, though some secondary features may be simplified.
- score4 description All critical content is accurately retained with minor detail loss.
- score5 description Perfect preservation of all primary and secondary elements without compromise. criteria description Detail Fidelity: To what extent does the SVG conversion preserve the details of the original image?

- score1 description Fine details are completely lost; output appears overly simplified or blurred.
- score2 description Basic details are visible but intricate features (e.g., textures, small text) are poorly rendered.
- score3 description Moderate detail retention; some high-frequency elements may lack precision.
- score4 description High-fidelity details with only subtle imperfections in complex areas.
- score5 description Pixel-perfect detail reproduction matching the original’s complexity. criteria description Style Consistency: Compared to the reference style, how well does the conversion match the target style?

- score1 description Style is inconsistent or contradictory to reference (e.g., line art vs. painterly).
- score2 description Partial style adherence with noticeable deviations in techniques/effects.
- score3 description Broadly matches reference style but lacks nuanced execution.
- score4 description Close stylistic alignment with minor variations in rendering methods.
- score5 description Seamless style replication that could pass as original artwork. criteria description Color Harmony: How harmonious is the color combination in the conversion result?

- score1 description Clashing or jarring colors; poor contrast/balance distracts from content.
- score2 description Some discordant color pairs but maintains basic readability.
- score3 description Generally pleasing palette though certain hues may feel slightly off.
- score4 description Well-balanced colors with intentional aesthetic cohesion.
- score5 description Masterful color theory application enhancing visual appeal. criteria description Composition Balance: How balanced is the visual composition of the conversion result?

- score1 description Unbalanced weighting creates visual tension or emptiness.
- score2 description Some elements feel misaligned or disproportionately emphasized.
- score3 description Adequate balance though certain areas could benefit from adjustment.
- score4 description Strong compositional flow with deliberate focal points.
- score5 description Expert-level layout adhering to design principles (e.g., rule of thirds, negative space).

### Prompt for Perceptual and Semantic QA

### Prompt for Generating Questions in Perceptual and Semantic QA

You are a svg analysis expert. Follow these steps carefully to answer the given multiple choice question. Task instruction:

Please analyze the provided icon image and its caption, then generate 2 multiple choice questions (one for each category below). Each question should have four options (A, B, C, D) with exactly one correct answer. Generate these two types of questions:

- 1. Answer the given multiple choice question below according to the svg code.
- 2.Output the answer in the format ’Answer: X’ in the last line, where X is one of A, B, C, or D.

- 1. Perceptual Question: Focus on the visual features of the icon such as shapes, number of elements, or spatial arrangement.
- 2. Semantic Question: Explore the meaning, function, or usecase of the icon. Focus on what the icon represents or where it might typically appear. Format requirements: - Question: [question text] Options: A) [option A]; B) [option B]; C) [option C]; D) [option D] Answer: [correct option letter] Guidelines:

SVG Code: {svg_image} Question: {question} Options: {options_str}

- - All questions must be directly grounded in the icon image and its caption
- - All alternative options should be plausible but clearly distinguishable from the correct answer
- - Ensure questions have varying difficulty levels appropriate to their category
- - The correct answer should be objectively verifiable based on the image and caption

Important Notes:

- - You should answer exactly the given multiple-choice question, DO NOT propose a new question.
- - Your output in the last line must be strictly in the format ’Answer: X’, where X is one of A, B, C, or D. Now, answer the given question:

## G Editing G.1 Bug Fixing

Construction Details. The bug fixing task is designed to systematically evaluate a model’s ability to identify and repair syntactic anomalies in SVG. This task simulates common error scenarios that arise in real-world design workflows, focusing on three major categories of typical faults:

- • Tag Errors: Structural issues in the XML hierarchy due to incorrect or malformed element tags.
- • Path Command Errors: Logical inconsistencies in shape rendering caused by invalid or misused path commands.
- • Attribute Errors: Disruptions in visual appearance resulting from incorrect attribute names or values. Each task instance consists of the following components:
- • Corrupted SVG: Automatically generated using our custom tool SVGErrorGenerator, which injects faults with clearly defined types and diverse manifestations, ensuring that the rendered output is visibly degraded.
- • Ground-Truth SVG: The original, error-free SVG file, used to evaluate the correctness of the model’s output.

The error generation process supports fine-grained control over fault injection parameters, including error type, frequency, and random seed, allowing for balanced sample diversity and task difficulty. Supported error types include tag misspellings or omissions, illegal path command substitutions, and incorrect attribute keys or values. Evaluation Metrics. To comprehensively assess the performance of models in this task, we adopt the accuracy of bug fixing as the core evaluation metric:

• Accuracy (ACC): Measures the proportion of SVG outputs that are fully restored to a structurally and semantically correct state. Evaluation is based on a strict equivalence check against the ground-truth SVG to determine successful repairs.

This metric characterizes a model’s end-to-end capability in the detect-locate-repair loop. Importantly, we emphasize that successful repairs must not only restore structural validity, but also recover the intended design semantics and rendering fidelity.

### Prompt for Bug Fixing Task

You are a professional SVG repair engineer with expertise in SVG standards and common error types. Your task is to analyze submitted SVG code, precisely locate errors, and fix problems using the principle of minimal modification. Please only modify the parts causing errors while keeping the rest of the code unchanged. After fixing, return the complete corrected code in the following strict format: Answer:{SVG code}, providing the complete fixed code only in the SVG code position, without adding any explanations, comments, or other content.

BUG SVG: {bug_svg}

## G.2 Code Optimization

Construction Details. The code optimization task is designed to evaluate the model’s ability to perform structural compression and syntactic refactoring of SVG source code while preserving its rendered appearance. Inspired by the optimization strategies employed by the widely adopted open-source tool SVGO [10], we construct an automated dataset tailored for structure-aware SVG code optimization.

Each sample in the dataset comprises the following three components:

- • Optimization Instruction: Expressed in natural language, explicitly prompting referencing the SVGO [10] principle the model to compress, simplify, and remove redundancies in the SVG code according to professional standards.
- • Original SVG: Our dataset, featuring diverse structures and frequently containing optimizable redundancies.
- • Groud-TruthSVG: Generatedautomaticallyviaa custom Python interface to the SVGO [10] library, ensuring structural validity and visual fidelity with respect to the original SVG.

The dataset spans a wide range of graphical complexities, providing the model with realistic optimization scenarios that reflect practical code refinement tasks.

Evaluation Metrics. To comprehensively assess the performance of the model in two key dimensions: compression effectiveness and visual fidelity. We propose the following dual-metric evaluation scheme:

- • Compression Code Ratio (CCR): This metric quantifies the reduction in byte size of the optimized SVG relative to the original SVG. It serves as a proxy for the model’s ability to compactly restructure the code without altering its visual rendering. The compression ratio is defined as:

𝐶𝐶𝑅 = (1 −

Optimized Size Original Size ) × 100% (2)

- • MSE: To ensure visual fidelity, we compute the pixel-wise Mean Squared Error (MSE) between rasterized images of the original SVG and the model-optimized SVG. This metric captures rendering discrepancies and evaluates whether the visual output remains perceptually consistent after optimization.

To enhance sensitivity, we additionally report the MSE trend under varying compression levels, allowing for finer-grained analysis of whether aggressive compression leads to visual degradation. This enables detection of trade-offs where structural compactness may come at the cost of fidelity.

The core objective of this task is to encourage models to achieve structural compression without compromising visual consistency. Hence, optimized outputs with higher compression ratios and lower MSE scores are considered indicative of stronger optimization capabilities.

### Prompt for Code Optimization Task

You are an advanced SVG optimization expert, skilled at maximizing compression and optimization of SVG code while maintaining visual consistency and ensuring code correctness. Please optimize the user-provided SVG code according to the following core principles:

- 1. Remove metadata and editor information

- - Clear metadata, comments, and unnecessary attributes generated by design software
- - Remove hidden elements and empty tags

- 2. Path optimization

- - Simplify path data, reduce control points
- - Lower decimal precision (1-2 places is usually sufficient)
- - Merge similar paths

- 3. Attribute and style processing

- - Remove redundant or default attribute values
- - Merge duplicate styles
- - Optimize color representation (e.g., #000 instead of #000000)

- 4. Structure optimization

- - Remove unnecessary grouping and nesting
- - Optimize IDs and class names
- - Ensure viewBox is set correctly

- 5. Compression and fine-tuning

- - Remove unnecessary whitespace and units
- - Use short commands instead of long format commands

After optimization, please strictly return the complete optimized code in the following format:Answer: {SVG code},Provide the complete optimized code only in the SVG code position, without adding any explanations, comments, or other content.

Here is the original SVG to optimize: {origin_svg}

## G.3 Style Edting

Construction Details. The SVG Editing Task is designed to evaluate the capability of large language models (LLMs) to perform localized and controllable modifications on structured graphics. We construct a synthetic dataset containing diverse types of editing operations. Each sample consists of three components: a natural language editing instruction, the original SVG graphic, and the corresponding ground-truth SVG after modification.

The editing instructions span a variety of common transformation types, including but not limited to:

- • Element-level color modifications (e.g., fill color, stroke color)
- • Geometric transformations (e.g., rotation, translation, scaling)
- • Stylistic enhancements and adjustments (e.g., blur filters, gradients, stroke width)

All samplesareautomaticallygenerated via programmatic scripts to ensure semantic clarity, structural validity, and compatibility with automated evaluation pipelines. To support fine-grained manipulation of SVG elements, we develop a custom SVG editing

toolkit based on Python and lxml, enabling precise control over element-level modifications. Each task instance pairs a well-defined natural language instruction with a reproducible target SVG, ensuring consistency and objectivity across the evaluation dataset.

Evaluation Metrics. Traditional image-level metrics (e.g., Mean Squared Error, MSE) are often insufficiently sensitive for SVG editing tasks, as they may not capture fine-grained structural changes and can be misleading due to global rendering differences. To address this, we introduce three metric evaluation schemes that emphasize both structural fidelity and perceptual relevance:

- • Relative Levenshtein Distance (RLD): Measures the minimal structural modification cost between the model-generated SVG and the ground-truth SVG, computed over the raw SVG markup. This captures the syntactic efficiency of the model’s edits.
- • Relative Mean Squared Error (rMSE): A structure-aware ratio metric that quantifies the degree of visual correction (or degradation) relative to the original input. It is defined as:

𝑟𝑀𝑆𝐸 =

√︄

1 − min 1,

𝑀𝑆𝐸(𝑔𝑒𝑛,𝑔𝑡) 𝑀𝑆𝐸(𝑔𝑡,𝑜𝑟𝑖)

(3)

where 𝑔𝑒𝑛 is the model’s rendered output, 𝑔𝑡 is the rendered ground-truth SVG, and 𝑜𝑟𝑖 is the rendered original SVG. A larger 𝑟𝑀𝑆𝐸 indicates that the model output more closely aligns with the intended target and represents a significant structural improvement over the original.

- • Accuracy: We employ accuracy as the primary evaluation metric to assess models’ proficiency in executing specific editing operations correctly, where accuracy is computed as the proportion of successful task completions over the total number of editing attempts across all test samples.

This evaluation framework jointly captures the syntactic and semantic accuracy of model edits, offering fine-grained insight into model performance on real-world SVG editing tasks and enhancing the reliability of comparative analysis.

### Prompt for Style Editing Task

You are a professional SVG editing engineer with extensive experience in SVG editing. Your task is to receive SVG code and modification requests from users, and make precise modifications according to their instructions. Please only modify the parts specified by the user, keeping the rest of the code unchanged. After fixing, return the complete corrected code in the following strict format: Answer:{SVG code}, providing the complete fixed code only in the {SVG code} position, without adding any explanations, comments, or other content.

Here is the original SVG: {original_svg}

Edit command: {edit_command}

