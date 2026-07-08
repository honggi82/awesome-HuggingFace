## Open Multimodal Retrieval-Augmented Factual Image Generation

Yang Tian‡*, Fan Liu†, Jingyuan Zhang||, Wei Bi¶, Yupeng Hu‡, Liqiang Nie§ ‡ Shandong University, † National University of Singapore, ||Kuaishou Technology, ¶ Independent Researcher § Harbin Institute of Technology, Shenzhen

{tianyangchn,liufancs,nieliqiang}@gmail.com

### Abstract

(a) Parametric (b) Closed-Domain Retrieval-Augmented Generation Generation

[Figure 1]

| |[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Unitree|Image|
|---|---|---|
| | | |
| |G1|With<br><br>Reference|

[Figure 5]

WithImage

Generate an image of a Tesla Optimus and a Unitree-G1 robot shaking hands Local Dataset Outdated Missing

Reference

# arXiv:2510.22521v1[cs.CV]26Oct2025

Large Multimodal Models (LMMs) have achieved remarkable progress in generating photorealistic and prompt-aligned images, but they often produce outputs that contradict verifiable knowledge, especially when prompts involve fine-grained attributes or time-sensitive events. Conventional retrieval-augmented approaches attempt to address this issue by introducing external information, yet they are fundamentally incapable of grounding generation in accurate and evolving knowledge due to their reliance on static sources and shallow evidence integration. To bridge this gap, we introduce ORIG, an agentic open multimodal retrievalaugmented framework for Factual Image Generation (FIG), a new task that requires both visual realism and factual grounding. ORIG iteratively retrieves and filters multimodal evidence from the web and incrementally integrates the refined knowledge into enriched prompts to guide generation. To support systematic evaluation, we build FIG-Eval, a benchmark spanning ten categories across perceptual, compositional, and temporal dimensions. Experiments demonstrate that ORIG substantially improves factual consistency and overall image quality over strong baselines, highlighting the potential of open multimodal retrieval for factual image generation.

| | |
|---|---|
| | |

[Figure 6]

(c) Open Multimodal Retrieval-Augmented Generation

[Figure 7]

[Figure 8]

[Figure 9]

Optimus and Unitree G-1 both have humanoid structures

Retrieved Images

Retrieved Text

Human-like form including two arms, two legs, and a torso.

###### Tesla Optimus is about 173 cm tall

Tesla (Optimus) is weight 73 kg Tesla robot is constructed using metal and plastic materials

###### Unitree G1 robot is about 127 cm

Unitree G1 is made from aluminum alloy and high-strength engineering plastic

With Text Reference With Multimodal Reference With Image Reference

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Figure 1: Motivation of factual image generation (FIG) with open multimodal retrieval. (a) Reliance on internal knowledge alone often leads to outdated or hallucinated content. (b) Incorporating external information improves grounding but remains constrained by static and unimodal sources. (c) Leveraging open retrieval of multimodal evidence integrates evolving knowledge and complementary cues to achieve FIG.

compose multiple entities, and integrate contextual signals to create visually compelling results.

Despite these advances, existing generation paradigms, including both standalone LMMs (Bai et al., 2024; Huang et al., 2025b) and retrievalaugmented approaches (Hu et al., 2024b; Yuan et al., 2025), still struggle with factual consistency. Beyond creative applications, image generation plays an increasing role in education and scientific communication, where generated visuals are expected to convey accurate knowledge rather than merely appealing visuals. This issue becomes especially pronounced when prompts involve finegrained attributes, such as relative scale or material composition, or require up-to-date knowledge about real-world entities and events.

### 1 Introduction

Recent advances in image generation have enabled models to produce photorealistic and compositionally rich visual content (Karras et al., 2019; Rombach et al., 2022; Zhao et al., 2024; Wu et al., 2025). Large multimodal models (LMMs) further extended these capabilities by jointly modeling language and vision under a unified framework (Fu et al., 2024; Wu et al., 2025; Wang et al., 2025). They can flexibly interpret open-domain prompts,

As illustrated in Figure 1, generating “Tesla Optimus shaking hands with Unitree-G1” demands more than basic object recognition. Textual knowledge is needed to capture factual details such as relative heights and materials, while visual references

*Work done during an internship at Kuaishou Technology.

provide essential cues about appearance, proportions, and spatial configuration. Moreover, prompts that reference evolving real-world events, such as new product releases or sports outcomes, require timely knowledge updates that are beyond the capabilities of both static pre-trained models and conventional retrieval-augmented approaches reliant on fixed corpora.

The limitation arises from two fundamental factors. First, current generation paradigms rely on static parametric memory or closed-domain corpora, making them unable to capture new facts, evolving entity states, or time-sensitive events. Second, factual grounding requires complementary multimodal evidence. Text provides attributes and relations, while images provide perceptual cues such as appearance, proportion, and spatial configuration. However, most existing approaches process these modalities separately and use retrieved knowledge only as auxiliary information, limiting their ability to generate factually grounded content.

Motivated by these challenges, we formalize Factual Image Generation (FIG) as a new task setting that requires outputs to be both visually realistic and factually grounded. To address this task, we propose ORIG, an agentic Open multimodal Retrieval-augmented framework for Image Generation. The framework consists of three key components: an open multimodal retrieval module that iteratively gathers external evidence and performs coarse cross-modal filtering to discard irrelevant content; a prompt construction module that extracts fine-grained factual elements from the filtered evidence and assembles them into a generation-ready prompt; and an image generation module that produces the final output guided by this enriched prompt.

To evaluate the effectiveness of multimodal knowledge integration, we further introduce FIGEval (Factual Image Generation Evaluation). FIGEval spans ten categories annotated across three key conceptual dimensions: perceptual elements, object and scene composition, and temporal sequences. It provides a comprehensive benchmark for evaluating models’ ability to leverage retrieved text and images for factual image generation. To facilitate reproducibility, code and data are released at https://tyangjn.github.io/orig.

github.io/. In summary, our main contributions are as follows:

• We formalize Factual Image Generation as a new task setting that emphasizes factual grounding in

addition to realism.

- • We propose ORIG, an agentic open multimodal retrieval-augmented framework that iteratively retrieves evidence from the web and distills it into task-relevant knowledge for generation.
- • We construct FIG-Eval, a benchmark spanning ten entity classes and three concept dimensions, enabling systematic evaluation of factual consistency in image generation.
- • Extensive experiments demonstrate that ORIG consistently improves factual grounding and overall generation quality across diverse settings.

### 2 Related Work

Advances in Image Generation. Image generation has progressed from GAN-based models (Zhang et al., 2017) to diffusion models (Rombach et al., 2022; Ramesh et al., 2022; Saharia et al., 2022), which significantly improved fidelity, diversity, and controllability. Recent developments have shifted toward LLM-based architectures, which can be broadly categorized into two primary paradigms: auto-regressive models that generate images from unified visual-text tokens (Wang et al., 2024; Batifol et al., 2025), and hybrid models that use an LLM backbone to steer a diffusion decoder (Zhou et al., 2024; Wu et al., 2025; Geng et al., 2025). However, despite their impressive realism, these models often suffer from factual inconsistency. This issue primarily stems from a fundamental gap between the models’ static, internal knowledge and the dynamic, real-world facts (Kalai and Vempala, 2024; Xu et al., 2025). This limitation undermines their reliability in critical domains such as education, media, and science, where accuracy is crucial (Robertson, 2024; Liu et al., 2023).

Retrieval-Augmented Generation. To improve the factual grounding, recent works retrieve reference images from local datasets to guide generation (Shalev-Arkushin et al., 2025). While this improves visual fidelity, methods such as FineRAG (Yuan et al., 2025) and Tiger (Qu et al., 2025) are fundamentally limited by their reliance on closed-domain corpora with static image references, which fail to capture evolving real-world knowledge (Hu et al., 2024a). In contrast, existing multimodal web-retrieval frameworks (e.g., OmniSearch (Li et al., 2024c), OpenManus (Liang et al., 2025)) are primarily designed for text generation. They often treat images as auxiliary signals, which limits their ability to filter noisy data and

selectively exploit visual evidence, both of which are essential for generating complex and factually consistent images.

Evaluation Benchmarks for Image Generation. Evaluation benchmarks for image generation have traditionally centered on metrics like prompt alignment, image quality, and compositional correctness (Hu et al., 2024a; Li et al., 2024a; Xu et al., 2023; Li et al., 2024b; Niu et al., 2025). While recent efforts have expanded this scope to include more complex reasoning tasks, such as commonsense and physical understanding (Peng et al., 2025; Huang et al., 2024; Fu et al., 2025; Yan et al., 2025). However, a critical dimension remains largely underexplored. The factual accuracy of generated content, particularly in scenarios requiring the integration of multimodal external knowledge, is not adequately addressed by current evaluation protocols.

### 3 Methodology

#### 3.1 Task Definition

We formalize FIG as a new task setting where the defining goal is to ensure factual consistency in generated images. Formally, given a query prompt P, the task requires producing an image that is semantically aligned with P, and grounded in verifiable knowledge about entities, attributes, relations, and temporal events. Specifically, factual consistency in FIG spans three dimensions: Perceptual Fidelity, ensuring faithful perception and accurate rendering of objects’ visual appearance; Compositional Consistency, which enforces accurate object properties and spatial relations, and Temporal Consistency, which ensures proper depiction of event timing and entity states. Unlike conventional image generation, FIG requires grounding in external evidence beyond the limited and static parametric memory of LMMs. In practice, this necessitates open retrieval from the web, where textual and visual evidence contribute complementary knowledge to support Perceptual Fidelity and Compositional Consistency, while the real-time nature of retrieval supplies updated information essential for Temporal Consistency.

#### 3.2 The ORIG Framework

We propose ORIG, an agentic open multimodal retrieval-augmented framework that grounds image generation in verifiable knowledge. As illustrated in Figure 2, ORIG adopts an iterative pipeline that

plans sub-queries, retrieves modality-specific evidence, filters noise through coarse-to-fine filtering, and integrates refined knowledge into enriched prompts that guide factual image generation. The framework comprises three modules: Open Multimodal Retrieval Module that collects and filters web-scale evidence through adaptive query planning and sufficiency evaluation; Prompt Construction Module that integrates the input prompt with extracted features from filtered evidence to create generation-ready prompts, and Image Generation Module that produces factually grounded images based on the enriched prompts.

#### 3.2.1 Open Multimodal Retrieval

This module is implemented as an agentic openretrieval loop that incrementally builds a reliable knowledge base from the web. ORIG operates as a closed feedback cycle guided by both the input prompt and the accumulated external knowledge. The loop consists of five stages:

Bootstrapping Retrieval. To ensure effective query planning, the module begins with a lightweight bootstrapping retrieval before entering the retrieval loop, which provides basic knowledge of the entities or concepts contained in the prompt P, preventing the model from generating misaligned sub-queries due to insufficient understanding of rare or novel terms. The retrieval results are stored in the local knowledge base K, serving as the initial context for subsequent query-planning and enabling early error correction.

Query Planning. The stage begins by analyzing the input P against its current K to identify underspecified or missing information. These identified knowledge gaps are then decomposed into a set of sub-questions Q, which are mapped to their optimal modality. This complementary mapping generates textual queries St to retrieve contextual knowledge (e.g., attributes, relations), while visual queries Sv are designed to capture perceptual information (e.g., appearance, spatial configurations). Instructed by IQ, the output of retrieval backbone model M 1 for planning stage is:

⟨Q, St, Sv⟩ = M(IQ, P, K). (1)

Modality-Specific Retrieval. The module performs retrieval using publicly available web re-

1We denote the large multimodal model as M, which serves as the core reasoning and generation component. In our implementation, M is instantiated with models such as Qwen-VL and GPT-5, while the framework itself remains model-agnostic.

Prompt: Generate an image of a Tesla Optimus and a Unitree-G1 robot shaking hands. Prompt

Query Planning

Modality-Specific Retrieval

Multimodal Knowledge Accumulation

Bootstrapping Retrieval

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Filter

Prompt

[Figure 21]

What is Unitree-G1

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Text

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Update

[Figure 32]

What is Tesla Optimus

###### Open Multimodal Retrieval

SubQuestions

SubQuestions

[Figure 33]

[Figure 34]

[Figure 35]

Filter

Prompt

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Image

Saving

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Text Image

[Figure 51]

Update

Insufficient: Next Round Query Planning

Sufficient: Prompt Refinements

Retrieved Multimodal Knowledge

[Figure 52]

Sufficiency Evaluation

| | | |
|---|---|---|
| | | |
| | | |

Image Generation New Module

Tesla Optimus is taller than Unitree G1, Tesla Optimus is made from metal … Unitree is covered by engineering plastic …. Tesla Optimus is made from metal …

Prompt

[Figure 53]

Fine-Grained

[Figure 54]

[Figure 55]

Prompt Construction Module

Extraction

Features

[Figure 56]

<Features> from Text and Image

[Figure 57]

[Figure 58]

- <Img#0> Preserved Features: Rectangular screen…
- <Img#1> Preserved Features: Circular LED ring …

Prompt

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Reference Images

Fine-grained Multimodal Refinement

Prompt Extension

Figure 2: The overall pipeline of the ORIG framework. ORIG adaptively controls multimodal retrieval and prompt construction, dynamically deciding whether to continue retrieval or proceed based on the current state of accumulated knowledge.

trieval APIs, one for each modality. The outputs are denoted as Rt and Rv, which correspond to the raw retrieved text and image sets, respectively. For each query S in the query sets St and Sv, the appropriate API is invoked:

where ISE is the instruction guiding sufficiency evaluation. If the knowledge base is incomplete or lacks essential details, the module is advised to initiate another retrieval round. Otherwise, it proceeds to the next stage of prompt construction. This feedback-driven control mechanism enables ORIG to adaptively determine the optimal number of retrieval rounds required to support grounded and informed image generation.

Rm = {APIm(S) | S ∈ Sm}, m ∈ {t, v}. (2)

Multimodal Knowledge Accumulation. The module incrementally builds a relevant knowledge base by performing coarse-grained filtering on all retrieved content. The filtering operates in two steps: First, the model evaluates retrieved text snippets Rt against the existing multimodal context. A text is retained only if it satisfies the joint conditions of semantic alignment with the prompt and factual consistency with the multimodal knowledge

3.2.2 Prompt Construction and Image Generation

Once the retrieval module determines that sufficient evidence has been gathered, ORIG transitions to the prompt construction module. This module transforms the multimodal knowledge base K into a generation-ready prompt through fine-grained knowledge refinement and prompt extension. The enhanced prompt is then fed into the image generation module for factually grounded synthesis.

base K: ˆ

Rt = M(ITF, P, K, Rt), (3) where ITF guides the textual filtering process. Next, the module applies similar multimodal consistencybased filtering to retrieved images Rv, retaining those that maintain coherence with both textual evidence and existing visual evidence:

Fine-grained Multimodal Refinement. While coarse-grained filtering during retrieval eliminated obviously irrelevant content, fine-grained refinement focuses on precision and semantic quality. This stage applies stricter criteria to extract the most discriminative features from the remaining multimodal evidence, moving beyond simple relevance to identify content that directly supports generation objectives. The refinement process operates in two stages. First, the module identifies visual descriptors and generation-relevant attributes within the textual knowledge to yield a set of core textual features Ft, while simultaneously deduplicating the image set Kv to refined image set KvR:

Rˆ v = M(IVF, P, K, Rv), (4)

where IIF is the instruction for image filtering. The filtered texts Rˆ t and images Rˆ v are incorporated into the knowledge base, which is updated for subsequent iterations:

K ← K ∪ {Rˆ t, Rˆ v}. (5)

Sufficiency Evaluation. Once the knowledge base is updated, the module determines whether the retrieved knowledge sufficiently addresses the identified sub-queries Q. This is formalized as:

⟨Ft, KvR⟩ = M(ICR, P, K), (7)

D = M(ISE, P, K, Q), D ∈ {Retrieval, Refine} (6)

Table 1: Prompt and Question Distribution Across 10 Entity Classes and three Concept Categories. The entity classes include Animal (An.), Sports (Sp.), Transportation (Tr.), Landmarks (La.), Food (Fo.), People (Pe.), Plants (Pl.), Products (Pr.), Culture (Cu.), and Events (Ev.).

Categories An. Sp. Tr. La. Fo. Pe. Pl. Pr. Cu. Ev. Total Prompt Number 55 52 50 50 49 52 51 56 50 49 514 Perceptual Fidelity (PF) 233 197 207 138 116 194 219 287 171 105 1,867 Compositional Consistency (CC) 120 148 143 148 189 185 117 106 191 243 1,590 Temporal Consistency (TC) 73 46 44 95 88 40 89 31 65 65 636 All Concepts Categories 426 391 394 381 388 419 425 418 427 413 4,093

where ICR guides the extraction of visuallydescriptive textual features and image deduplication. Subsequently, visual control features are extracted from KvR using cross-modal guidance:

Fv = M(IVR, P, KvR, Ft), (8)

where IVR leverages textual features Ft to identify critical visual elements that should be preserved for generation control, while filtering out irrelevant background elements that could compromise generation fidelity. This process produces attention guidance that directs the generator to focus on essential visual cues within the reference images.

Prompt Extension. The extracted multimodal features are integrated to synthesize the final generation prompt:

Pˆ = M(IPE, P, Fv, KvR, Ft), (9)

where IPE guides the extension process. This enriched textual prompt Pˆ not only incorporates the retrieved factual knowledge but also instructs the model to focus on critical visual elements within the reference images. The prompt Pˆ and filtered reference images KvR are jointly fed to an image generation module for multimodal factually grounded synthesis.

#### 3.3 Discussion

ORIG can be viewed as a self-adaptive retrieval framework that rethinks how external knowledge is integrated into image generation. Traditional unimodal retrieval approaches often rely on static visual references, which limit their ability to incorporate abstract concepts, contextual cues, and timesensitive information. In contrast, ORIG employs an iterative agentic retrieval process that plans subqueries, filters relevant evidence, and incrementally refines a multimodal knowledge base conditioned on the evolving prompt state. This dynamic retrieval–integration cycle enables the model to go beyond static grounding, mitigating factual inconsistencies and aligning generated images with evolving real-world knowledge.

### 4 Dataset: FIG-Eval

We introduce FIG-Eval, a curated dataset designed to evaluate whether image-generation models can effectively leverage web-retrieved multimodal evidence to achieve FIG. It covers 10 entity classes and three concept groups, featuring knowledge-intensive prompts that encode implicit, domain-specific facts requiring external evidence beyond parametric knowledge. For reliable assessment, each prompt is paired with humanannotated ground-truth references. From these evidence, we manually derive multimodal QA questions that rigorously operationalize factual correctness, enabling automated scoring via state-of-theart Vision-Language Models.

Prompt Construction. Building on prior typologies (Lim et al., 2025; Niu et al., 2025; Huang et al., 2024), we established a taxonomy of ten entity classes (Table 1) and constructed a retrievaloriented dataset guided by two key principles: multi-hop sequential retrieval, which enables stepwise text-to-image grounding, and parallel coretrieval, which jointly integrates textual and visual evidence. Subsequently, five expert annotators (each with over one year of evaluation experience) curated 1,132 prompts, each paired with ∼8 annotated multimodal references. To ensure retrieval dependence, we applied a two-stage filtering process. First, ambiguous, underspecified, or trivial items were removed. Second, the remaining prompts were stress-tested with GPT, Gemini, and Qwen2.5VL under no-retrieval settings, instructing models direct generation with parametric knowledge. If the generated images faithfully reflected all intended reference facts, the prompts were deemed solvable without retrieval and discarded. This process yielded 514 validated prompts, balanced across 10 entity classes.

Evaluation. Following prior dataset designs (Niu et al., 2025; Huang et al., 2025a), we group factual information into three visual concept categories (Table 2): Perceptual Fidelity, Compositional Consistency, and Temporal Consistency.

Table 2: Visual concepts by categories.

Concepts Categories Visual Concepts Perceptual Fidelity (PF) Color, Appearance, Size, Texture Compositional Object Number, Position, Interaction, Consistency (CC) Fore-Background Temporal Consistency (TC) Time, Sequence Order, Process Steps

Each prompt Pi is paired with expert-annotated references from which we extract a ground-truth fea-

ture set FGTi = {FijGT}nj=1i (ni = |FGTi |). Annotators then construct multimodal true–false questions

Xi = {Xik}mk=1i (mi = |Xi|), each mapped to a concept group, with all answers fixed as True. This

yields 4,093 questions in total, all dual-annotated with adjudication. Generated images Ii are then evaluated with model Mˆ (GPT-5) under a fixed QA template. For each question Xik, the model outputs score Yik ∈{True,False}:

M ˆ (Xik, Ii, IGT) required reference image Mˆ (Xik, Ii), else

(10)

Yik =

The per-prompt accuracy is the mean of {Yik}mk=1i , which is then macro-averaged for concept- and

class-level reporting:

1 mi

Si =

mi

1[Yik = True]. (11)

k=1

Given retrieved text Rti and images Rvi , we also evaluate retrieval quality by measuring the align-

ment between {Rti,Rvi } and FGTi with Mˆ :

ni

1 ni

Areti =

Align(FijGT | Rti, Rvi ), (12)

j=1

where Align ∈ {0,1}. This yields a feature-level retrieval accuracy in [0,1].

Correlation with Human Judgments. To validate our automated evaluator Mˆ , we compared it against a human baseline. We randomly sampled 200 prompts and generated images using three image generation models (GPT-Image, GeminiImage, and Qwen-Image). Five expert annotators independently answered the QA pairs for all model outputs, with majority voting used for final labels. We observe strong agreement between evaluator and human scores, with correlations computed over per-class averages (Pearson’s r = 0.929, Spearman’s ρ = 0.936, Kendall’s τ = 0.772). To further test robustness and mitigate potential bias, we evaluated on Gemini-2.5-Flash outputs, which again yielded high consistency (r = 0.922, ρ = 0.928, τ = 0.764), confirming the reliability and generalizability of our protocol.

### 5 Experiment

#### 5.1 Experimental Setup

We evaluate on FIG-Eval by jointly considering image generators and retrieval agents. For generation, we select three representative LLM-based image generators with strong multimodal reasoning capacity: GPT-Image (GPT-Image-1), GeminiImage (Gemini-2.5-Flash-Image), and the opensource Qwen-Image. Each generator is tested under four knowledge conditions: Direct, using only the raw prompt; Oracle, conditioned on annotated gold textual and visual references as an upper bound; Prompt Enhanced, where prompts are expanded with parametric knowledge from the retrieval backbone LLM; and Retrieval, where prompts are augmented with retrieved evidence. Within the retrieval setting, we compare our ORIG agent against two open-source multimodal baselines, OpenManus (Liang et al., 2025) and OmniSearch (Li et al., 2024c), and additionally introduce two modality-restricted variants: ORIGImg, which retrieves only images, and ORIG-Txt, which retrieves only text. The retrieval agents are instantiated with two backbone LLMs, GPT-5 and Qwen2.5-VL-72B. Open retrieval is conducted via Google Search and Google Image Search with identical configurations across methods. All evaluation results are reported using GPT-5 as the evaluator to maintain consistency across settings.

#### 5.2 Main Results

As shown in Table 3, the Direct setting establishes the baseline: Gemini-Image (34.6%) and GPTImage (32.1%) outperform Qwen-Image (19.0%), reflecting inherent differences in their parametric knowledge. Performance further improves in the Prompt Enhanced setting (e.g., GPT-Image reaches 40.4%) by leveraging the retrieval backbone model’s reasoning ability; however, these gains remain limited, as the enhanced prompts depend solely on internal knowledge and fail to incorporate the detailed, up-to-date information provided by web-retrieval methods. In contrast, when augmented with our proposed ORIG framework, all three image generation models achieve the highest accuracies, with GPT-Image at 50.1%, GeminiImage at 51.4%, and Qwen-Image at 39.7%, consistently surpassing retrieval baselines such as OmniSearch and OpenManus. Moreover, the ORIG model outperforms both unimodal variants, confirming that joint multimodal retrieval effectively

- Table 3: Accuracy (%) comparison across 10 entity classes and 3 concept categories on the FIG-Eval dataset using GPT-5 as the retrieval backbone. The All Acc. columns report average score of all 10 categories with both GPT-5 and Qwen2.5-VL-72B backbones.

Entity Classes Concepts All Acc. An. Sp. Tr. La. Fo. Pe. Pl. Pr. Cu. Ev. PF CC TC GPT Qwen

Methods

Qwen-Image 16.4 9.3 19.3 31.3 12.8 16.7 23.3 12.3 30.2 17.7 21.5 18.2 14.0 19.0 Oracle 78.2 69.1 73.2 81.7 68.9 73.4 79.2 70.1 77.6 73.4 74.4 73.1 71.9 74.5 Prompt Enhanced 31.9 16.7 37.9 41.2 23.3 26.4 45.3 25.0 34.5 31.9 30.7 33.4 30.3 31.5 28.1 OpenManus 37.1 20.6 42.3 43.1 26.5 26.7 45.4 35.2 42.9 33.5 36.2 34.4 35.5 35.4 32.2 OmniSearch 35.4 19.1 41.8 42.4 26.4 26.1 44.8 34.5 41.9 33.1 35.6 34.6 32.3 34.7 31.3 ORIG 39.0 25.8 45.2 46.5 32.5 29.7 52.4 40.2 45.5 36.7 41.9 38.5 37.2 39.7 36.1 ORIG-Img 32.3 21.6 42.1 44.5 25.6 27.5 47.4 38.0 37.8 34.4 39.2 34.3 27.9 35.4 32.0 ORIG-Txt 37.4 21.7 41.0 42.8 28.0 27.3 50.8 33.9 43.3 34.9 36.1 37.1 36.4 36.5 32.9

Gemini-Image 47.0 20.0 32.3 45.2 31.0 22.3 51.3 22.6 42.5 30.4 34.9 34.4 35.1 34.6 Oracle 85.1 77.9 78.1 86.3 78.2 77.4 87.8 78.5 85.4 82.5 84.4 79.8 78.9 81.8 Prompt Enhanced 49.6 25.2 41.0 49.2 38.2 30.7 61.0 28.7 50.7 38.1 39.8 42.4 45.3 41.4 34.9 OpenManus 50.9 33.1 46.4 51.2 33.9 35.5 62.1 41.3 46.4 42.5 46.2 42.6 43.9 44.4 36.1 OmniSearch 49.5 31.6 44.8 50.4 33.4 34.2 61.5 38.4 45.7 42.1 44.0 43.1 41.6 43.3 35.3 ORIG 54.0 41.3 53.6 57.3 42.5 42.0 68.7 51.1 50.5 49.4 52.4 50.0 53.9 51.4 41.6 ORIG-Img 50.0 33.5 52.1 50.5 39.1 35.9 62.1 46.3 46.9 41.9 49.5 43.9 43.6 46.2 37.3 ORIG-Txt 52.9 36.5 46.1 53.9 39.8 35.0 64.8 42.4 47.7 48.1 46.2 48.1 47.2 46.9 37.8

GPT-Image 39.7 17.1 33.1 44.5 21.8 20.2 45.7 31.9 35.1 30.3 34.6 30.7 29.1 32.1 Oracle 83.3 75.2 78.3 85.2 74.6 76.7 87.2 79.6 85.1 82.1 83.1 79.0 78.3 80.8 Prompt Enhanced 48.1 27.4 41.7 47.2 36.1 30.0 59.8 26.5 45.5 39.5 39.5 40.7 43.5 40.4 32.5 OpenManus 50.5 32.4 45.8 50.1 33.2 34.7 60.8 39.9 45.1 42.2 45.1 42.0 43.1 43.6 35.5 OmniSearch 49.1 30.7 43.6 49.2 32.8 33.6 60.4 37.9 44.2 41.0 43.2 41.9 40.8 42.4 34.7 ORIG 53.0 40.2 51.6 56.4 40.4 41.0 67.0 50.5 49.5 47.6 51.5 48.9 50.6 50.1 40.5 ORIG-Img 49.3 32.5 49.3 49.0 37.7 33.9 61.6 44.8 46.2 40.6 48.8 42.3 41.0 44.9 36.1 ORIG-Txt 52.2 35.0 44.6 53.3 37.8 34.1 63.6 40.6 47.7 45.4 44.5 47.6 46.4 45.8 37.2

combines complementary visual and textual knowledge for superior performance. Nonetheless, even with Oracle knowledge, the models still struggle to capture fine-grained visual attributes, underscoring a persistent challenge in integrating multimodal information for faithful image synthesis.

Impact of retrieval backbones. The results reveal two distinct stages of improvement. The transition from the Direct to the Prompt Enhanced setting reflects each model’s ability to exploit internal parametric knowledge through reasoning-based prompt enrichment, whereas the subsequent gain from Prompt Enhanced to Retrieval captures its capacity to integrate external knowledge. Qwen2.5VL-72B shows a modest average 3.3% improvement in the first phase but a more substantial 7.6% gain with retrieval. In contrast, GPT-5 achieves stronger internal reasoning gains (+9.2%) and comparable retrieval gains (+9.3%), demonstrating a more balanced ability to leverage both parametric and external knowledge.

Performance on different concept categories. Across the three generation models, the Temporal Consistency category yielded the largest average improvement (+21.2%), underscoring the critical role of retrieval in supplying the temporal and dynamic knowledge that is often underrepresented in parametric memory. Interestingly, Perceptual Fidelity showed a slightly higher gain (+18.2%) than the more structurally complex Compositional Consistency category (+17.9%). This suggests that while compositional reasoning re-

mains a known challenge, integrating knowledge for multi-entity scenes is inherently more complex due to the difficulty of aligning diverse contextual cues. Conversely, retrieval of fine-grained perceptual attributes provides clearer and more targeted information that models can integrate.

Performance across different entity classes. Based on knowledge update frequency (Chen et al., 2021), entities are categorized into dynamic classes (e.g., Transportation, Events, Sports) and stable classes (e.g., Animals, Landmarks, Culture). On average improvement of ORIG across three generation models, dynamic categories show lower baseline accuracy but receive a substantial boost from retrieval, with large improvements observed in Products (+24.9%) and Sports (+20.3%). In contrast, stable categories exhibit stronger baseline performance and therefore experience smaller retrieval gains, such as in Animals (+14.3%) and Culture (+12.5%).

#### 5.3 Ablation Study

Contribution of Different Modality. As shown in Table 3, based on averages across the three generation models relative to the Direct baseline, ORIGImg yields the largest gains on Perception Fidelity (PF, +15.4%) but smaller gains on Compositional Consistency (CC, +12.4%) and Temporal Consistency (TC, +11.4%), reflecting weaker semantic constraints and harder alignment in multi-entity or cross-stage scenes (e.g., Events), where insufficient prompt control can steer the model toward

- Table 4: Retrieval performance comparison(%) based on Gemini-Image, with "Acc" as retrieval accuracy, "Token" for average token count in generated prompts, "Image Num" for the average number of images per prompt. "Iters" represent the total number of retrieval iterations, including the bootstrapping retrieval, +n-Round represents n iterations in the retrieval loop.

Methods Iters R. Acc Tokens Images G. Acc

OpenManus 2.3 63.4 483.3 1.3 44.4 OmniSearch 2.5 60.8 472.9 1.5 43.3

ORIG 2.8 74.7 519.1 2.1 51.4

- +1-Round 2.0 70.6 453.5 1.6 50.3

- +2-Round 3.0 75.0 536.4 2.3 51.4

- +3-Round 4.0 75.1 597.2 2.8 50.9

irrelevant elements within reference images; in such cases, image-only retrieval can even trail the Prompt-Enhanced results on CC and TC. By contrast, ORIG-Txt delivers steadier improvements on CC (+16.7%) and TC (+15.6%) by supplying attributes, relations, and procedural cues, though it is less effective on PF (+12.3%) due to missing fine-grained appearance detail. At the entity level, image retrieval provides larger marginal benefits for visually fast-evolving classes (Transportation +19.6%; People +20.7%), whereas text retrieval benefits appearance-stable classes (Animals +13.1%; Plants +19.6%). Finally, by jointly leveraging text for semantic constraints and images for appearance cues, and by enforcing coarse-to-fine filtering with prompt refinement, the ORIG framework mitigates insufficient semantic control and visual misalignment, yielding consistent gains across 10 entity classes and 3 concept categories.

Comparison of Multimodal Retrieval Methods. As shown in Table 4, our adaptive ORIG framework achieves a high retrieval accuracy of 74.7%, clearly outperforming baseline methods such as OpenManus (63.4%) and OmniSearch (60.8%). Further analysis shows that ORIG constructs richer prompts (averaging 519.1 tokens) and leverages more reference images (averaging 2.1), indicating its stronger ability to capture and synthesize relevant and detailed multimodal information, which in turn enhances the final generation quality. To further examine retrieval efficiency, we conducted a study on fixed-round strategies in the retrieval loop. The results show that increasing the retrieval rounds to three only slightly improves retrieval accuracy to 75.1%, while it leads to a small decrease in generation accuracy to 50.9%. This trade-off likely arises from two factors: first, excessive retrieval rounds tend to introduce overly fine-grained or redundant descriptions of minor de-

Table 5: Accuracy (%) based on Gemini-Image of different ablation variants. The values after the slash “/” denote the performance drop compared to the ORIG.

Methods GPT-5 Qwen2.5-VL

ORIG 51.4 41.6 w.o. Bootstrapping Retrieval 50.2/1.2 40.9/0.7 w.o. Knowledge Accumulation 47.4/2.8 39.1/2.5 w.o. Fine-grained Refinement 46.8/3.4 38.6/3.0 w.o. Prompt Extension 48.9/2.5 39.2/2.4

tails that are difficult for the image generator to reflect effectively; second, a larger set of reference images increases the likelihood of including irrelevant entities. Both factors can distract the model’s attention and hinder the generation process. Besides, considering token efficiency, the marginal improvement is negligible: total input tokens increase substantially from 2-Rounds to 3-Rounds, while the retrieval gain remains minimal.

Effectiveness of ORIG Components. Results in Table 5 indicate that removing Knowledge Accumulation and Fine-grained Multimodal Refinement leads to a notable decline in the generation accuracy of Gemini-Image, when using retrieval results from GPT-5 and Qwen2.5-VL-72B as inputs. Specifically, the generation accuracy decreases by 2.8% and 3.4% with GPT-based retrieval, and by 2.5% and 3.0% with Qwen-based retrieval. These results highlight the importance of the Coarse-toFine Filtering process in excluding irrelevant or noisy information before generation. Removing Prompt Extension also degrades performance, with 2.5% and 2.4% reductions under GPT- and Qwenbased retrieval, respectively, confirming its role in enriching prompts and aligning retrieved evidence with generation objectives. Furthermore, removing Bootstrapping Retrieval causes a smaller yet consistent drop in performance, with 1.2% and 0.7% decreases for GPT- and Qwen-based retrieval, respectively. This suggests that the warm-up retrieval helps stabilize early retrieval, particularly for rare or semantically ambiguous entities.

### 6 Conclusion

This work tackles the problem of factual inconsistency in image generation, where outputs often diverge from verifiable knowledge. We formalized Factual Image Generation (FIG) as a new task requiring both visual realism and factual grounding, and proposed ORIG, an open multimodal retrievalaugmented framework that dynamically integrates textual and visual evidence. We further built FIGEval, a benchmark for evaluating this task across perceptual, compositional, and temporal dimen-

sions. Experimental results show that ORIG improves factual consistency and generation quality over conventional approaches. Future work will explore deeper reasoning over retrieved knowledge and finer alignment between semantic intent and visual details.

### References

Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930.

Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, and 1 others. 2025. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506.

Wenhu Chen, Xinyi Wang, and William Yang Wang.

2021. A dataset for answering time-sensitive questions. arXiv preprint arXiv:2108.06314.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, and 1 others. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning.

Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. 2024. Guiding instruction-based image editing via multimodal large language models. In The Twelfth International Conference on Learning Representations.

Xingyu Fu, Muyu He, Yujie Lu, William Yang Wang, and Dan Roth. 2025. Commonsense-t2i challenge: Can text-to-image generation models understand commonsense? In First Conference on Language Modeling.

Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, and 1 others. 2025. Xomni: Reinforcement learning makes discrete autoregressive image generative models great again. arXiv preprint arXiv:2507.22058.

Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang Zhao, Xue Ben, Boqing Gong, William Cohen, and 1 others. 2024a. Instruct-imagen: Image generation with multi-modal instruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4754–4763.

Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang, and Nanyun Peng. 2024b. Mrag-bench: Vision-centric evaluation for retrievalaugmented multimodal models. In The Thirteenth

International Conference on Learning Representations.

Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. 2025a. T2icompbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 others. 2025b. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55.

Ziwei Huang, Wanggui He, Quanyu Long, Yandi Wang, Haoyuan Li, Zhelun Yu, Fangxun Shu, Long Chan, Hao Jiang, Fei Wu, and 1 others. 2024. T2ifactualbench: Benchmarking the factuality of textto-image models with knowledge-intensive concepts. arXiv preprint arXiv:2412.04300.

Adam Tauman Kalai and Santosh S Vempala. 2024. Calibrated language models must hallucinate. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 160–171.

Tero Karras, Samuli Laine, and Timo Aila. 2019. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410.

Black Forest Labs. 2024. Flux. https://github.com/ black-forest-labs/flux.

Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Xide Xia, Pengchuan Zhang, Graham Neubig, and Deva Ramanan. 2024a. Evaluating and improving compositional text-to-visual generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5290–5301.

Wei Li, Xue Xu, Jiachen Liu, and Xinyan Xiao. 2024b. Unimo-g: Unified image generation through multimodal conditional diffusion. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6173–6188.

Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao Zheng, Fei Huang, Jingren Zhou, and 1 others. 2024c. Benchmarking multimodal retrieval augmented generation with dynamic vqa dataset and self-adaptive planning agent. arXiv preprint arXiv:2411.02937.

Xinbin Liang, Jinyu Xiang, Zhaoyang Yu, Jiayi Zhang, Sirui Hong, Sheng Fan, and Xiao Tang. 2025. Openmanus: An open-source framework for building general ai agents.

Youngsun Lim, Hojun Choi, and Hyunjung Shim. 2025. Evaluating image hallucination in text-to-image generation with question-answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 26290–26298.

Han Liu, Yuhao Wu, Shixuan Zhai, Bo Yuan, and Ning Zhang. 2023. Riatig: Reliable and imperceptible adversarial text-to-image generation with natural prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20585–20594.

Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. 2025. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265.

Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. 2025. Dreambench++: A human-aligned benchmark for personalized image generation. In The Thirteenth International Conference on Learning Representations.

Leigang Qu, Haochuan Li, Tan Wang, Wenjie Wang, Yongqi Li, Liqiang Nie, and Tat-Seng Chua. 2025. Tiger: Unifying text-to-image generation and retrieval with large multimodal models. In The Thirteenth International Conference on Learning Representations.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3.

Adi Robertson. 2024. Google apologizes for ‘missing the mark’after gemini generated racially diverse nazis. the Verge, 21.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, and 1 others. 2022. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494.

Rotem Shalev-Arkushin, Rinon Gal, Amit H Bermano, and Ohad Fried. 2025. Imagerag: Dynamic image retrieval for reference-guided image generation. arXiv preprint arXiv:2502.09411.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, and 1 others. 2024. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.

Yuhan Wang, Siwei Yang, Bingchen Zhao, Letian Zhang, Qing Liu, Yuyin Zhou, and Cihang Xie. 2025. Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint

- arXiv:2507.21033.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, and 1 others. 2025. Qwen-image technical report. arXiv preprint

- arXiv:2508.02324.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903– 15935.

Xinhao Xu, Hui Chen, Mengyao Lyu, Sicheng Zhao, Yizhe Xiong, Zijia Lin, Jungong Han, and Guiguang Ding. 2025. Mitigating hallucinations in multi-modal large language models via image token attentionguided decoding. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1571–1590.

Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. 2025. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782.

Huaying Yuan, Ziliang Zhao, Shuting Wang, Shitao Xiao, Minheng Ni, Zheng Liu, and Zhicheng Dou. 2025. Finerag: Fine-grained retrieval-augmented text-to-image generation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 11196–11205.

Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. 2017. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 5907–5915.

Chuyang Zhao, Yuxing Song, Wenhao Wang, Haocheng Feng, Errui Ding, Yifan Sun, Xinyan Xiao, and Jingdong Wang. 2024. Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280.

Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. 2024. Transfusion: Predict the next token and diffuse images with one multi-modal model. In The Thirteenth International Conference on Learning Representations.

Table 6: Cost and time analysis across pipeline stages.

Bootstrapping retrieval

Knowledge Accumulation

Finegrained Refine

Query Planning

Prompt Extension

Metric

Total Retrieval 1.2 6.7 – – – Retrieval Time (s) 1.1 7.7 – – – Input Tokens 61.6 1,021.1 3,598.8 827.1 740.6 Output Tokens 102.6 138.8 827.1 740.6 519.1

### A More Implementation Details

During retrieval, textual and visual queries were processed separately to ensure modality-specific optimization. For text retrieval, we used the Serper API to access Google Search (region = US, language = English), retrieving the top 10 webpage results per query, each containing a URL and a short textual snippet. To prevent redundant crawling and enhance relevance, the snippets were evaluated by the retrieval backbone model based on semantic similarity, and the top 2 most relevant webpages were retained. The full content of these pages was then extracted using the Jina.ai Reader API, providing clean textual evidence for subsequent reasoning and grounding. For image retrieval, we used Serper’s Google Image Search endpoint to obtain the top 10 image results per query. We selected the top

- 5 accessible and unique images after removing broken or duplicate links, ensuring both coverage and diversity of visual evidence. The retrieval backbone was Qwen2.5-VL, applied in multi-scale variants (72B, 32B, and 7B), complemented by a GPT-5 (2025.10) retriever with default sampling parameters (temperature ≤ 0.2, top-p default). For image generation, we employed Gemini-Image (Gemini2.5-Flash-Image, “Nano-Banana” version), GPTImage-1, and Qwen-Image, all accessed through their official APIs with default configurations consistent with their released repositories, ensuring comparability across generation backbones. All instructions, prompts, evaluation scripts, and annotated references will be released in the anonymous GitHub repository.

#### A.1 Token Usage

Table 6 reports the cost and time statistics across different pipeline stages. Average Search denotes the total number of text and image retrieval calls, while Input Tokens correspond to the average number of tokens fed into the retrieval backbone at each stage. The Query Planning phase shows relatively high input token usage because each planning step must condition on previously accumulated information within the knowledge database to decide the next retrieval direction. The Knowledge Accumulation stage consumes the largest number of tokens,

Table 7: Retrieval cost with different iterations. Text retrieval represents the total retrieval times of Google Search, where Image Retrieval represents the retrieval times of Google Image, Input and Output Tokens is the length of input and output for prompt extension stage.

Text Retrieval

Image Retrieval

Input Tokens

Output Tokens

Iter.

- +1-Round 2.9 1.8 657.9 453.5
- +2-Round 4.6 2.7 784.8 536.4
- +3-Round 6.3 3.2 941.1 597.2 ORIG 4.5 2.5 740.6 527.1

as it involves processing and filtering the full multimodal content of the retrieved information.

Although the pipeline is token-intensive, this higher cost is intrinsic to the FIG task itself, which requires inherently heavy retrieval for multimodal, knowledge-grounded image generation. Such generation must render fine-grained scene details using complementary textual and visual evidence, thereby necessitating richer and more exhaustive retrieval than typical VQA-style tasks that seek only a single textual answer. To maintain practicality, several safeguards are integrated into the pipeline. Adaptive query generation dynamically produces one to five modality-specific retrieval queries per round, balancing coverage and efficiency (Table 7). Sufficiency-based iteration control halts retrieval once adequate evidence has been gathered, preventing redundant web expansion and mitigating downstream token growth.

### B Supplement to Experiments

#### B.1 Results Based on Qwen-Retrieval

As shown in Table 8, we present detailed results on the FIG-Eval dataset using the Qwen2.5-VL-72B retrieval backbone, covering ten entity classes and three concept categories. The overall trend aligns well with the main results in Table 3: all three image generation models (Qwen-Image, GeminiImage, and GPT-Image) exhibit substantial accuracy improvements after integrating retrieved external evidence, while the full ORIG framework consistently achieves the highest scores across nearly all categories. The most pronounced gains are observed in semantically complex and frequently updated domains such as Products, ransportation, and Events, highlighting the crucial role of factual and time-sensitive web evidence in enhancing the fidelity and contextual correctness of image generation. Furthermore, improvements in Temporal Consistency and Compositional Consistency demonstrate that retrieval-grounded generation better preserves coherent temporal and relational structures

- Table 8: Accuracy(%) comparison based on Qwen2.5-VL-72B retrieved evidence across 10 entity classes and 3 concept categories on the FIG-Eval dataset. All represents the average score of all 10 categories.

Methods An. Sp. Tr. La. Fo. Pe. Pl. Pr. Cu. Ev. PF CC TC All

Qwen-Image 16.4 9.3 19.3 31.3 12.8 16.7 23.3 12.3 30.2 17.7 21.5 18.2 14.0 19.0 Prompt Enhanced 33.5 12.7 25.3 39.2 19.7 21.9 37.8 24.0 37.6 27.3 28.4 27.2 30.4 28.1 ORIG 36.6 21.1 41.5 45.1 28.2 23.6 50.1 37.7 42.0 33.9 37.1 36.5 33.6 36.1 ORIG-Img 33.8 16.7 37.2 39.8 20.1 22.8 47.3 33.5 35.7 31.0 34.8 31.0 27.8 32.1 ORIG-Txt 35.6 18.7 35.3 40.6 24.1 23.5 48.5 28.7 40.8 31.2 33.3 34.0 30.9 32.9

Gemini-Image 47.0 20.0 32.3 45.2 31.0 22.3 51.3 22.6 42.5 30.4 34.9 34.4 35.1 34.6 Prompt Enhanced 38.4 24.0 33.8 44.0 22.9 28.0 51.6 28.3 41.3 34.9 35.4 33.8 37.2 34.9 ORIG 47.9 26.4 43.9 50.8 29.6 33.3 53.3 42.7 41.9 44.4 42.5 41.8 40.0 41.6 ORIG-Img 46.2 21.3 37.5 44.7 25.2 27.0 51.0 37.2 41.4 37.4 39.6 35.6 35.9 37.3 ORIG-Txt 48.0 24.6 38.1 45.6 23.9 28.2 52.4 32.4 41.3 40.3 37.3 38.6 38.8 37.8

GPT-Image 39.7 17.1 33.1 44.5 21.8 20.2 45.7 31.9 35.1 30.3 34.6 30.7 29.1 32.1 Prompt Enhanced 35.7 21.9 31.1 41.7 18.6 26.7 49.1 25.7 39.6 33.0 33.7 31.8 31.8 32.5 ORIG 44.6 25.1 40.2 49.8 26.1 33.8 51.6 45.3 41.4 44.4 42.6 38.4 40.5 40.5 ORIG-Img 42.7 22.1 38.4 42.6 23.9 30.8 47.9 33.9 39.1 35.8 38.7 33.8 35.1 36.1 ORIG-Txt 44.3 23.9 35.9 45.2 24.3 30.1 49.9 34.8 40.5 39.2 37.4 36.7 38.9 37.2

- Table 9: Accuracy(%) comparison with Qwen2.5-VL-72B/-32B/-7B retrieved evidence across 10 entity classes and 3 concept categories on the FIG-Eval dataset. All represents the average score of all 10 categories. The image generation model is Gemini-Image

Methods An. Sp. Tr. La. Fo. Pe. Pl. Pr. Cu. Ev. PF CC TC All Qwen2.5-VL-72B

- Prompt Enhanced 38.4 24.0 33.8 44.0 22.9 28.0 51.6 28.3 41.3 34.9 35.4 33.8 37.2 34.9 ORIG 47.9 26.4 43.9 50.8 29.6 33.3 53.3 42.7 41.9 44.4 42.5 41.8 40.0 41.6

Qwen2.5-VL-32B

- Prompt Enhanced 38.5 24.3 33.5 42.4 22.9 27.6 51.9 27.7 38.1 34.6 34.6 34.6 34.8 34.4 ORIG 43.1 26.1 37.4 47.7 28.4 33.7 53.0 38.0 41.3 42.3 40.6 39.5 37.3 39.5

Qwen2.5-VL-7B

Prompt Enhanced 38.2 22.7 33.0 40.2 22.7 26.7 50.4 26.3 36.5 33.8 34.4 32.6 33.1 33.4 ORIG 40.8 23.6 35.1 45.5 26.3 31.9 52.0 36.6 38.9 37.4 38.0 37.2 35.9 37.2

that purely parametric generation often fails to capture. A comparison between the ORIG-Img and ORIG-Txt variants further verifies the importance of multimodal fusion. Text-only retrieval improves context consistency but lacks perceptual grounding, whereas image-only retrieval contributes to visual alignment but fails to capture factual attributes. Their combination within the ORIG framework consistently outperforms both unimodal variants, demonstrating that Qwen2.5-VL-72B effectively fuses complementary textual and visual knowledge.

#### B.2 Results Across Different Model Scales

As shown in Table 9, the retrieval performance of the Qwen2.5-VL family scales consistently with model size. Larger variants exhibit stronger reasoning and evidence integration abilities, leading to clear accuracy gains across both entity and concept dimensions. Specifically, Qwen2.5-VL-72B achieves the highest overall accuracy (41.6%), surpassing the 32B and 7B models by 2.1% and 4.4%, respectively. The improvement mainly stems from enhanced cross-modal grounding and better utilization of retrieved knowledge, particularly in seman-

tically complex and frequently updated domains classes such as Products and Events.

#### B.3 Additional Generation Results

As shown in Table 10, we additionally evaluate several representative diffusion-based models on the FIG-Eval benchmark. The diffusion-based models such as Flux-Schnell (Labs, 2024), FluxDev (Labs, 2024), and SD-3.5-Large (Esser et al., 2024) demonstrate limited performance in factual grounding and multi-entity reasoning. Their accuracy across concept categories remains lower than LLM-based models like Gemini-Image or GPTImage, indicating that diffusion-based models possess limited intrinsic knowledge and lack the reasoning capabilities required to interpret knowledgeintensive or context-dependent visual semantics.

#### B.4 Evaluation of I-HallA Benchmark

I-HallAbench (Lim et al., 2025) (I-HallA) is a benchmark for evaluating image hallucination in text-to-image models, using QA pairs generated by GPT-4o to assess factual accuracy. It includes 1,000 multiple-choice sets to measure how well

- Table 10: Accuracy comparison (%) of four models on direct generation task across 10 entity classes and 3 concept categories using the FIG-Eval benchmark. All indicates the average performance across all entity categories.

Methods An. Sp. Tr. La. Fo. Pe. Pl. Pr. Cu. Ev. PF CC TC All

Flux-Schnell 13.4 4.0 18.7 22.1 6.1 12.6 9.5 14.9 13.5 24.6 14.6 13.2 12.7 13.8 Flux-Dev 9.6 10.5 12.9 27.1 5.0 15.0 12.5 20.4 16.7 21.6 18.2 13.1 9.8 14.9 SD-3.5-Large 19.3 8.4 16.3 31.1 10.7 9.9 17.3 10.5 11.5 20.4 16.8 17.5 14.8 16.8

- Table 11: Results (Accuracy %) on I-HallA, evaluation using I-HallA score. "w. ORIG" represents the ORIG augmented generation process.

of textual descriptions (e.g., model specifications, relative heights) and visual appearances from external sources. This design encourages retrievaldependent reasoning, ensuring that successful generation reflects both linguistic–visual alignment and factual consistency.

GeminiImage w. ORIG

GPTImage w. ORIG

QwenImage w. ORIG

GeminiImage

GPTImage

QwenImage

Metric

Science 0.826 0.854 0.818 0.852 0.728 0.788 History 0.806 0.834 0.796 0.822 0.667 0.712

Furthermore, as shown in Table 13, FIG-Eval adopts a more realistic retrieval setting compared to prior benchmarks. Earlier methods such as ImageRAG, TIGER, and RE-IMAGEN rely solely on image-based retrieval from static, locally stored datasets using CLIP embeddings. By contrast, our ORIG pipeline performs iterative, open-domain retrieval on the web (via Google), jointly leveraging textual and visual evidence with dynamic filtering and sufficiency evaluation. This multimodal retrieval setting not only expands the accessible knowledge space but also provides more faithful, up-to-date evidence for complex realworld prompts. Overall, FIG-Eval bridges the gap between conventional parametric generation and retrieval-augmented multimodal reasoning, providing a comprehensive benchmark for assessing factuality, compositionality, and knowledge grounding in text-to-image models.

models reflect factual information in generated images. We tested the performance of ORIG framework on the I-HallA benchmark using GeminiImage, GPT-Image and Qwen-Image as the image generation models. Although I-HallA focuses on factual domains where strong models such as GPT and Gemini have already internalized most knowledge, thereby reducing the marginal benefit of retrieval (compared to FIG-Eval), our approach still yields consistent and noticeable improvements in generation accuracy (Table 11), demonstrating strong generalizability beyond our own dataset.

### C Supplement to FIG-Eval

#### C.1 Comparison with Prior Benchmarks

As shown in Tables 12 and 13, FIG-Eval differs fundamentally from existing text-to-image benchmarks in its task definition, knowledge dependency, and retrieval formulation. Existing benchmarks primarily assess models’ ability to generate from internal (parametric) knowledge, relying on commonsense priors or style imitation. Reasoning, when required, typically operates within the model’s internal representation without external grounding. In contrast, FIG-Eval explicitly targets open-domain, multimodal knowledge integration. It evaluates whether models can retrieve and compose factual visual and textual evidence to address knowledgeintensive prompts that extend beyond their parametric memory.

#### C.2 Example of Retrieval Principles

Multi-hop sequential retrieval refers to a staged retrieval process in which the model first performs text retrieval to obtain factual and symbolic information, followed by image retrieval to supplement perceptual details. For example, when generating “the top-3 best-selling electric vehicles in North America this year”, the model first retrieves textual data to identify the specific models and brands (e.g., Tesla Model Y, Ford Mustang Mach-E), and then retrieves corresponding images to capture their visual appearance and structural features, achieving stepwise text-to-image grounding. In contrast, parallel co-retrieval conducts text and image retrieval simultaneously within the same iteration, integrating complementary multimodal evidence for knowledge alignment. For instance, when generating “a comparison image of Nintendo Switch and Switch 2”, the model jointly retrieves textual information (e.g., size differences and hardware specifications)

Table 12 compares representative prompts across benchmarks. While datasets such as InstructImagen, UNIMO-G, and DREAMBENCH++ are solvable through internal stylistic or compositional reasoning, FIG-Eval requires external factual grounding. For example, the prompt “Generate an image of a Tesla robot and a Unitree-G1 robot shaking hands” cannot be resolved using parametric priors alone, which demands retrieval

- Table 12: Representative prompt comparison between FIG-Eval and existing benchmarks. Unlike prior datasets solvable through parametric knowledge alone, FIG-Eval requires multimodal evidence-grounded reasoning.

Dataset Prompt Example Knowledge Source Instruct-Imagen Render an image of the [ref#1] vase that depicts the caption, adopting the style of [ref#2] style

Internal

image: a vase with flowers on top.

UNIMO-G <image#1> wearing <image#2> Internal DREAMBENCH++ A tiny kitten navigating a vast, otherworldly landscape on a leaf sailboat. Internal RISEBench Draw what they [ref image] will look like after being kept in a daily environment for a year. Internal, Reasoning Commonsense-T2I A lightbulb without electricity. Internal, Reasoning FIG-Eval Generate an image of a Tesla robot and a Unitree-G1 robot shaking hands. External, Reasoning

Table 13: Comparison of retrieval settings between FIG-Eval and prior benchmarks.

Method Prompt Example Modalities Retrieval Source

ImageRAG Grizzly bear in calculus class. Image CLIP Local dataset TIGER Mary Poppins flying with balloons. Image CLIP Local dataset RE-IMAGEN Braque Saint-Germain taking a shower. Image CLIP Local dataset FineRAG Cat with Santa mug. Image CLIP Local dataset ORIG (Ours) Tesla and UniTree robots shaking hands. Image + Text Google-based Open Web

###### Reference Image Direct Generation Generation with Text Generation with Image Generation with Multimodal

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Prompt: Generate an image of a Tesla Optimus and a UniTree G1 robot shaking hands.

Questions Dir Txt Img MM The picture shows two humanoid robots. False True True True

The two humanoid robots are interacting with each other and shaking False True True True hands. The Tesla Optimus robot in the picture resembles the one shown in True True True True

- <image_0>. The Unitree G-1 robot in the picture resembles the one shown in False False True True
- <image_1>.

The Tesla Optimus robot in the picture appears to be made of metal and True True True True plastic materials. The Unitree G-1 robot in the picture appears to be made of aluminum True True True True alloy and high-strength engineering plastic. The Tesla Optimus robot appears significantly taller than the Unitree G-1 True True False True robot.

Figure 3: Question-answering scores for different modality retrieval tasks (evaluated by GPT-5), with red indicating incorrect evaluations.

and visual information (device appearance images), enabling the generated output to accurately reflect real-world proportions and design variations.

#### C.3 Reference Content and Evaluation

In this section, we provide examples illustrating the structure of reference content in our dataset, including both textual and visual entries, as well as the evaluation questions derived from them. Each prompt is associated with manually curated reference text and images that reflect the intended semantics and visual context, as shown in Figure 6. Based on these references, we construct a set of fine-grained, true/false evaluation questions targeting specific visual attributes or factual elements. This structured annotation process enables consistent and interpretable assessment of generation quality across models and concept categories.

Reference for Prompt "Generate an image of a Tesla Optimus and a Unitree-G1 robot shaking hands."

Reference Text:

- 1. Two robots: Tesla humanoid robot (Optimus)

→ and Unitree G-1 robot.

- 2. Optimus and Unitree G-1 both have humanoid

→ structures: human -like form including two

→ arms , two legs , and a torso.

- 3. Unitree G-1:

- - Unitree G-1 robot is approximately 127 cm tall

→ and weighs around 35 kg.

- - It is made from aluminum alloy and high -

→ strength engineering plastic.

- - The head is a hollow structure with a light

→ ring.

- 4. Tesla Optimus:

- - Tesla Optimus is about 173 cm and weighs 73 kg

→ .

- - constructed using metal and plastic materials.

- - Optimus has a display screen on the face for

→ interaction and communication.

###### Reference Image

- 1. Img_0 (Tesla Optimus Picture)

- Title: .../- Url: .../- Local Path: ...

- 2. Img_1 (Unitree G-1 Picture) ...

### D Case Study

D.1 Comparison between Different Retrieval Methods

In addition to the quantitative evaluation presented in Section 5.3, we further compare our method with two representative agentic multimodal retrieval systems (OmniSearch, OpenManus) through case analysis. While Table 4 reports overall retrieval accuracy across the dataset, here we focus on illustrating the fundamental differences in retrieval objective, evidence usage, and resulting generation quality.

As shown in Figure 4, for the prompt “Generate a picture of a frog’s life cycle”, all three meth-

ods retrieve relevant external knowledge for expanding the prompt; however, only our approach successfully identifies fine-grained and visually aligned evidence that directly supports accurate image generation. In contrast, OmniSearch and OpenManus often retrieve semantically correct but visually underspecified content, resulting in images that omit critical visual cues, such as developmental stages, spatial configuration, or appearance transitions. This limitation arises because both systems were originally designed for multimodal question answering, where the retrieval target emphasizes semantic sufficiency rather than visual faithfulness. To further clarify the distinction, Table 14 provides a structured comparison among the three methods, highlighting their task focus, pipeline design, and knowledge granularity in the FIG-Eval setting. While OpenManus and OmniSearch also employ multimodal retrieval, ORIG fundamentally differs in its task formulation, evidence usage, and pipeline design, yielding more fine-grained and visually grounded outputs.

Task Difference. OmniSearch is optimized for producing a single textual answer, with images serving only as auxiliary or text-converted cues. As a result, its retrieved evidence remains coarsegrained and weakly linked to concrete visual attributes. Similarly, OpenManus focuses on explanatory reasoning, retrieving verbose yet abstract content that seldom translates into explicit visual constraints, rendering much of its retrieved prose redundant for image synthesis. In contrast, ORIG is specifically designed for knowledge-grounded textto-image generation, where factual and perceptual evidence must be jointly mapped to visual structures. Both textual and visual retrieval results are complementary and jointly essential for producing controllable, faithful images.

Pipeline Difference. ORIG introduces several design innovations absent in prior systems: ORIG introduces several design innovations absent in prior systems, which collectively enable finer-grained retrieval alignment and more coherent generation. First, it begins with bootstrapping retrieval for stable planning, performing an initial lightweight evidence search that helps stabilize sub-query decomposition under rare or ambiguous concepts. Next, it performs modality-aware knowledge accumulation and filtering, where textual evidence is dynamically refined based on the evolving knowledge state, and image retrieval is subsequently conditioned on the refined textual

[Figure 71]

[Figure 72]

[Figure 73]

###### OpenManus OmniSearch ORIG

Figure 4: The generation results for prompt "Generate a picture of a frog’s life cycle" with three retrieval methods

context. This step preserves complementary information across modalities and enhances cross-modal consistency. Finally, ORIG adopts a two-stage prompt construction process that integrates multimodal content refinement, through summarization and deduplication, along with visual–textual alignment, resulting in concise, generation-ready prompts. This structured pipeline avoids the redundancy and misalignment typical of raw retrieval concatenation, achieving both higher factual accuracy and improved visual coherence.

D.2 Effect of Retrieved Text length and details This section investigates how the length and granularity of retrieved textual information affect image generation quality. Specifically, we aim to examine whether generation models are capable of reflecting fine-grained details present in the input and whether increasing the level of description improves the output. To explore this, we present two case studies. For each case, we use GPT-5 to extract key content from the retrieved text and construct three image generation prompts at different granularity levels: coarse, normal, and fine. We then compare the generated results across these levels to evaluate the model’s ability to incorporate varying degrees of detail.

We employ GPT-Image as our generation model for all prompt variants. As shown in Figure 5, in most cases, increasing the granularity of textual descriptions leads to more accurate and visually aligned generations. The model is generally able to reflect fine-grained visual features, when these details are explicitly included in the prompt. However, we also observe notable failure cases. As illustrated in Figure 6, despite progressively refining the description to emphasize that the beetle emits a chemical spray from its rear as a defense mechanism, the model consistently fails to reproduce this behavior accurately in the generated images. Even when we provide an additional ground-truth reference image alongside the prompt, the output remains inconsis-

Table 14: Comparison of retrieval settings between FIG-Eval and prior benchmarks.

Method Task Components Knowledge Granularity

OmniSearch Knowledge-intensive VQA Planning; Adaptive retrieval Coarse OpenManus Knowledge-intensive VQA Planning; Adaptive retrieval Coarse ORIG FIG Planning; Adaptive retrieval; Content filtering; Prompt Expanding Fine-grained

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Fine Prompt

Original Prompt

Coarse Prompt

Normal Prompt

- Figure 5: Different granularity for prompt "Generate a picture of a Qin Dynasty emperor riding in a horsedrawn carriage."

[Figure 78]

[Figure 79]

[Figure 80]

Fine Prompt_1 Fine Prompt_2 Fine Prompt_3

- Figure 6: Different granularity for prompt "Generate a photo of a bombardier beetle attacking an enemy."

tent with the described action. This suggests that beyond a certain level of detail, the model’s ability to incorporate fine-grained semantics is limited not by the prompt itself, but by inherent limitations in the model’s capacity for fine-grained visual grounding and compositional fidelity.

#### D.3 Effect of Retrieved Images

In this section, we examine how the properties of retrieved reference images affect image generation quality under controlled settings. Using GPTImage as the generation model, we fix the prompt and accompanying control instructions, and vary four key image conditions: (1) presence of noisy entities, (2) image clarity, (3) cropping, and (4) the number of retrieved images. As illustrated in Figure 7, several important observations emerge.

We observe that the quality and structure of retrieved images significantly influence generation outcomes. First, images containing noisy entities, such as unrelated figures or text, can introduce semantic conflicts, leading the model to incorporate incorrect elements. Second, low-resolution or blurry images often cause hallucinations, as the model fills in missing details from prior knowledge, sometimes inaccurately. Third, cropped images that show only partial views (e.g., just the car front) may result in incomplete generations, while full-object images lead to more coherent outputs.

Lastly, although multiple images can enrich context, inconsistencies among them can confuse the model and reduce generation fidelity.

### E Annotators

We employed a group of trained human annotators through a controlled crowd-sourcing process for factual alignment and multimodal relevance verification in FIG-Eval. All annotators possessed at least TEM-4 or TEM-8 English proficiency and had prior experience in multimodal annotation, such as image–text alignment, caption verification, and factual consistency assessment. Before formal annotation, all participants underwent task-specific training to ensure their understanding of the evaluation criteria and annotation interface.

Annotation Guidelines and Criteria. To ensure annotation reliability and consistency, we prepared a detailed guideline document that provided comprehensive annotation instructions and examples. The guideline covered the following five aspects: (1) entity and concept classification, defining rules for identifying and labeling the ten entity classes and three concept dimensions (Perceptual, Compositional, and Temporal); (2) factual consistency criteria, specifying how to verify whether a generated image accurately reflects realworld facts, attributes, and relations described in the prompt; (3) visual–textual alignment, outlining procedures for assessing semantic and perceptual consistency between textual instructions and image content; (4) error categorization, defining common error types such as hallucinated objects, incorrect attributes, and temporal mismatches; and (5) annotation examples, providing both correct and incorrect examples with justification to standardize annotator judgment.

Training and Qualification Phase. All annotators participated in a trial annotation phase using a shared subset of data, which served both as a training exercise and qualification assessment. Only annotators who demonstrated strong agreement with expert annotations in this phase were retained for the main annotation task. During formal annotation, each generated image–prompt pair was independently reviewed by two annotators and

Reference Image with Noise Reference Image with Different Clarity

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Reference Image with Cropping Multiple Reference Images

- Figure 7: Effect of Retrieved Images, Prompt is "Generate a photo of a typical Xiaomi SU7 Ultra owner and his car. The owner is a 30-40 year old tech professional living in a major city in China, with an relative high annual income. The car should resemble the auto in the reference image <image#X>. Ignore unrelated details."

cross-checked for inter-annotator consistency. Any disagreement was resolved by senior annotators with prior experience in factual verification tasks.

Quality Control and Review Mechanism. To further ensure data quality, we implemented a multi-level review mechanism. Automatic filtering removed low-quality or incomplete generations based on resolution and object detection confidence. Human validation was conducted to eliminate NSFW or inappropriate content and to ensure factual and visual coherence. Additionally, nonauthor experts periodically audited random samples and provided feedback to maintain consistent annotation standards.

Ethical Considerations and Compensation. All annotators were fairly compensated at or above local academic assistant rates. All data used in FIG-Eval were collected from publicly available web sources strictly for academic research. No personally identifiable or sensitive information was included. For the human annotation component, all participants were informed of the research purpose and provided explicit consent prior to participation. Annotators acknowledged that their work would be used solely for dataset construction and model evaluation in research contexts.

Use of LLM Assistance in Annotation Support. In addition, we used ChatGPT as an auxiliary tool to assist in annotation guideline drafting, prompt rephrasing, and preliminary quality checking. ChatGPT was employed solely for nondecisive support tasks such as generating example instructions, verifying language clarity, and identifying potential annotation inconsistencies. All final annotations, factual judgments, and quality evaluations were conducted and verified by trained human annotators. No sensitive data were shared with the model, and all usage strictly adhered to OpenAI’s terms of service and institutional research ethics policies.

