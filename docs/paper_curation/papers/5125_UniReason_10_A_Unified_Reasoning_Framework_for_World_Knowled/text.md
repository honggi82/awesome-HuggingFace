

[Figure 2]

[Figure 3]

## UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing

[Figure 4]

# arXiv:2602.02437v4[cs.CV]20Feb2026

### Dianyi Wang1,2*, Chaofan Ma3*, Feng Han1,2, Size Wu4, Wei Song2,5, Yibin Wang1,2, Zhixiong Zhang2,3, Tianhang Wang2,5,

Siyuan Wang6†, Zhongyu Wei1,2†, Jiaqi Wang2† 1Fudan University, 2Shanghai Innovation Institute, 3Shanghai Jiao Tong University 4Nanyang Technological University, 5Zhejiang University, 6University of Southern California *Equal Contribution, †Corresponding Authors

### Abstract

Unified multimodal models often struggle with complex synthesis tasks that demand deep reasoning, and typically treat text-to-image generation and image editing as isolated capabilities rather than interconnected reasoning steps. To address this, we propose UniReason, a unified framework that harmonizes these two tasks through two complementary reasoning paradigms. We incorporate world knowledge-enhanced textual reasoning into generation to infer implicit knowledge, and leverage editing capabilities for fine-grained editing-like visual refinement to further correct visual errors via self-reflection. This approach unifies generation and editing within a shared architecture, mirroring the human cognitive process of planning followed by refinement. We support this framework by systematically constructing a large-scale reasoning-centric dataset (∼300k samples) covering five major knowledge domains (e.g., cultural commonsense, physics, etc.) for textual reasoning, alongside an agent-generated corpus for visual refinement. Extensive experiments demonstrate that UniReason achieves advanced performance on reasoning-intensive benchmarks such as WISE, KrisBench and UniREditBench, while maintaining superior general synthesis capabilities.

GitHub: https://github.com/AlenjandroWang/UniReason HuggingFace: https://huggingface.co/Alex11556666/UniReason

### 1 Introduction

Unified multimodal models have emerged as a promising paradigm for jointly handling visual understanding and generation tasks [1, 2, 3, 4, 5, 6]. By integrating perception and synthesis within a shared architecture, these models enable seamless interplay between comprehending visual content and producing new images conditioned on multimodal inputs. Among various capabilities, text-to-image (T2I) generation and image editing stand out as particularly challenging yet impactful applications. However, current unified models still struggle with complex scenarios, where tasks demand not only precise instruction following, but also world knowledge that extends beyond surface-level pixels, e.g., commonsense, physical laws, and spatial-temporal logic. Such challenges fundamentally demand reasoning capabilities to bridge the gap between abstract user

[Figure 5]

Figure 1 Illustrative cases of UniReason on image editing and T2I generation tasks. Given an instruction, the model first performs world knowledge-enhanced textual reasoning to generate grounded, fine-grained guidance for image synthesis. It then applies fine-grained editing-like visual refinement, correcting errors introduced during the initial generation and improving the synthesis quality.

intent and faithful visual output.

To enhance reasoning capabilities, a prominent line of work focuses on prompt enhancement or reprompting strategies [7, 8, 9]. These methods employ chain-of-thought (CoT) reasoning to expand abstract user prompts into explicit semantic and spatial guidance before generation. While effective in improving instruction alignment, these “reason-then-generate” approaches are inherently limited as reasoning occurs only before generation without access to visual feedback, preventing reflection on and correction of output errors. More recently, interleaved reasoning mechanisms [10, 11] alternate between textual reasoning and visual generation. By first generating an initial image, then performing textual reflection based on visual feedback, and finally refining the output, these approaches enable post-generation correction that was previously infeasible.

Despite this progress, existing methods still exhibit two key limitations. (1) Reasoning in these methods largely remains at the level of semantic reorganization, decomposing instructions into finer-grained descriptions or spatial layouts [7, 9, 11, 12]. This addresses only the explicit component of user intent, whereas faithful synthesis in practice demands world knowledge that is implicitly assumed rather than explicitly stated. Such knowledge must be inferred, not merely parsed from instructions. This creates a fundamental knowledge gap that surface-level decomposition cannot bridge. (2) Existing methods typically address text-to-image generation and image editing as separate tasks [10], leaving their inherent synergies within a unified interleaved framework untapped. We argue that these two tasks share substantial reasoning overlap and can mutually reinforce each other. Specifically, post-generation critique and refinement in interleaved reasoning is structurally analogous to editing. Isolating them therefore forgoes such synergy and leads to redundant learning.

To address these challenges, we propose UniReason, a unified reasoning framework that harmonizes text-to-image generation and image editing within a shared architecture, as illustrated in Fig. 1. Our framework supports two complementary reasoning paradigms. (1) World Knowledge-Enhanced Textual Reasoning aims to bridge the knowledge gap prior to synthesis. Given an underspecified instruction, the model performs textual reasoning to infer implicit world knowledge and produces grounded guidance that specifies fine-grained details for the subsequent image synthesis. To support this, we construct training data across five knowledge categories: cultural commonsense, natural science, spatial, temporal, and logical reasoning. We use large language models to generate reasoning traces and apply multi-dimensional filtering

to ensure high-quality supervision. (2) Fine-grained Editing-like Visual Refinement aims to improve synthesis quality after initial generation. Given the initial image and prior reasoning, the model performs self-reflection to identify discrepancies or missing details, then applies targeted corrections to produce a refined image. Observing that this process is structurally analogous to image editing, we jointly learn T2I generation and editing for mutual benefit. We design an agent pipeline that iterates through generation, verification, refinement, and comparison to construct high-quality training data. These two paradigms can be applied independently or jointly, offering flexibility across diverse synthesis scenarios.

We adopt a two-stage training strategy: the first stage strengthens foundational generation capability, and the second stage enables interleaved reasoning by jointly training the understanding and generation branches. Through this unified framework, we achieve comprehensive world knowledge-grounded reasoning capabilities for both T2I generation and image editing, with advanced performance on multiple benchmarks.

Our main contributions are summarized as follows:

- • We propose UniReason, a unified reasoning framework for both T2I generation and image editing. Our key insight is that refinement and editing share the same reasoning pattern, enabling bidirectional capability transfer.
- • We introduce two complementary reasoning paradigms: World Knowledge-Enhanced Textual Reasoning bridges the knowledge gap before synthesis, while Fine-grained Editing-like Visual Refinement enables iterative improvement after generation.
- • We systematically construct training data for both paradigms, including world knowledge-aligned data across five categories and an agent pipeline for refinement supervision, combined with a two-stage training strategy.
- • Extensive experiments demonstrate advanced performance on multiple benchmarks, including GenEval, WISE for T2I generation, and UniREditBench, KrisBench for image editing.

### 2 Related Work

Image Generation and Editing Image generation (T2I) and editing are two related tasks, depending on whether the conditional signals are textual descriptions or reference images. Recently, Diffusion Transformers [13, 14] (DiTs) have served as the backbone of state-of-the-art generation frameworks, with flow-matching [15, 16] adopted as the prevailing training scheme. Together with data and model scaling, these advances have enabled photo-realistic synthesis and substantially improved instruction following in T2I generation [15, 17, 18]. Building upon these powerful generators, recent image editing systems [17, 19] achieve precise content manipulation while preserving overall visual consistency. However, despite their generative prowess, these specialized models lack the intrinsic capacity for world comprehension and self-reflection, motivating the integration of reasoning and generation within a coherent unified framework.

Unified Multimodal Models Unified multimodal models [2, 3, 4, 5, 6, 20] aim to jointly support image understanding and generation within a single framework. Broadly, existing approaches can be grouped into two paradigms. A first, more modular paradigm aligns pretrained LMMs and DiTs via LLM hidden states [3, 21, 22] or learnable queries [5, 20, 23, 24]. Another line of work [1, 2, 6, 25] adopts a shared LLM architecture for perception and synthesis, encouraging a tight coupling between the two tasks. In our study, we focus on the second paradigm, since a shared backbone naturally supports interleaved reasoning between language and image generation in a unified inference process.

Reasoning in Unified Multimodal Models The structural convergence of understanding and generation within unified models unlocks the potential for grounding high-fidelity image synthesis in complex multimodal reasoning. Initial efforts primarily involve the adaptation of textual Chain-of-Thought (CoT) to image generation [7, 8, 9], following a “reason-then-generate” paradigm that expands user instructions into detailed descriptions prior to synthesis. More recently, interleaved reasoning mechanisms [10, 11] extend the process into iterative "reason-generate-reflect" cycles to incorporate visual feedback. Despite these advancements, existing methods are often confined to prompt reorganization and rigidly separate generation and editing tasks. In this work, we address these limitations by inferring implicit world knowledge rather than merely

parsing instructions. Furthermore, we exploit the inherent synergies between T2I generation and image editing within a unified reasoning framework.

### 3 Preliminary

Architecture We build upon Bagel [6] to develop a unified and interleaved reasoning framework for both T2I generation and image editing. Bagel adopts a Mixture-of-Transformers (MoT) architecture with a ViT encoder [26] to process multimodal inputs and enables unified image understanding and generation within a single foundation model.

Specifically, multimodal understanding is formulated as generating context-aware textual outputs via standard next-token prediction through a language modeling head. This process is conditioned on multimodal context inputs and handled by the understanding expert. Formally, the training objective minimizes the negative log-likelihood:

𝑇

log 𝑝𝜃(𝑥𝑡 | 𝑥<𝑡, 𝐶) , (1)

ℒtext = −

𝑡=1

where 𝑥𝑡 denotes the target text token, 𝑥<𝑡 is the preceding tokens and 𝐶 is the multimodal context.

Multimodal generation focuses on producing high-quality and semantically aligned images via a rectified flow process [27] in a VAE’s latent space [18], conditioned on multimodal inputs and handled by the generation expert. The training objective is to minimize the the latent flow-matching loss:

ℒimage = 𝔼𝑡∼𝒰(0,1) 𝑢𝜃(𝑧𝑡, 𝑡 ; 𝐶) − 𝑢★(𝑧𝑡, 𝑡) 22 , (2) where 𝑢★ denotes the target velocity, 𝑢𝜃 is the learned time-conditioned velocity field in the latent space.

Reasoning Paradigms Bagel’s unified architecture supports interleaving textual reasoning and visual synthesis in both T2I generation and image editing tasks. Specifically, T2I generation takes a textual instruction as input and outputs a sequence of intermediate reasoning tokens together with a synthesized image. For image editing, an existing image and a textual instruction are taken as input and the model outputs a reasoning text and the edited image. In this work, we formulate interleaved reasoning as an iterative process: (𝐼𝑘+1,𝑇𝑘+1) = ℱ 𝐼≤𝑘,𝑇≤𝑘, 𝐶 where 𝐼𝑘 and 𝑇𝑘 denote the image and reasoning text at iteration 𝑘, 𝐶 denotes the multimodal context, and ℱ is the unified model (𝑘 = 1 in our implementation). Under this formulation, each refinement step can be interpreted as an image editing operation conditioned on the reasoning trace. Therefore, we propose to jointly learn T2I generation and image editing within a unified interleaved reasoning framework, allowing the refinement process to benefit from editing learning and, conversely, enhance interleaved reasoning for both T2I generation and editing.

### 4 Method

In this section, we present UniReason, a unified multimodal reasoning framework for both T2I generation and image editing, as illustrated in Fig. 2. In practice, the framework operates in two phases, (1) World Knowledge-Enhanced Textual Reasoning for initial synthesis; (2) Fine-grained Editing-like Visual Refinement for iterative improvement. We introduce each phase along with its corresponding data creation pipeline in Sec. 4.1 and 4.2, respectively shown in Fig. 4, followed by the training strategy in Sec. 4.3.

#### 4.1 World Knowledge-Enhanced Textual Reasoning

Different from prior work [8, 9] that primarily focuses on re-organizing user instructions into more detailed visual descriptions, our core objective is to enable the unified multimodal model to not only expand raw user prompts but also understand the underlying implicit world knowledge. Specifically, UniReason utilizes textual reasoning to infer the world knowledge required to complete the visual synthesis, including commonsense, cultural context, time-spatial and natural science principles. This process provides explicit and structured guidance to ensure the initial generation is both instruction-aligned and knowledge-consistent, mirroring the conceptual planning that humans perform when outlining ideas for a drawing.

[Figure 6]

Figure 2 Overview of UniReason framework for two complementary reasoning paradigms in image synthesis.

Data Preparation To enable world knowledge-enhanced textual reasoning for the initial synthesis, we construct challenging input instructions for both T2I generation and image editing tasks that require complex world knowledge reasoning beyond complementing pixel-level details, along with their associated reasoning processes. Specifically, we cover five major categories of world knowledge and adopt post-generation filtering to ensure high-quality supervision.

- • Cultural Commonsense instructions require using shared cultural knowledge, such as historical events, iconic figures, social customs, and idiomatic expressions, to resolve unnamed or underspecified entities into explicit, contextually meaningful visual content, ensuring generated images aligned with real-world cultural understanding.
- • Natural Science instructions requires incorporating principles from physics, biology, medicine, or chemistry to ensure that generated images remain consistent with scientific laws, and reflect plausible real-world observations.
- • Spatial reasoning focuses on understanding correct spatial relationships among entities, including relative position, orientation, viewpoint, and camera transformations. Such instructions requires deriving precise spatial configurations from abstract descriptions to generate visuals consistent with real-world geometric logic.
- • Temporal reasoning models time-dependent relationships, such as event sequences, state transitions, and causal ordering. This type of instructions require inferring the temporal progression of events and ensuring that visual outputs reflect coherent and plausible temporal dynamics aligned with natural chronological flow.
- • Logical reasoning emphasizes causal coherence and logical consistency during image generation, such as in maze-solving or constraint satisfaction problems, by adhering to explicit or implicit logical structures. These instructions require applying deductive principles to translate abstract logical constraints into visually valid solutions.

For T2I generation in each category, we manually construct seed prompts based on Wikipedia, together with explicit category definitions, and use Gemini-2.5 Pro [28] to expand them into a larger prompt set. And Gemini-2.5 Pro is also employed to generate textual CoT reasoning for each prompt. All prompts with their corresponding CoTs are subsequently fed into Qwen-Image [17] for image rendering to form paired training samples. For image editing, we utilize data triples (original image, editing instruction, desired outcome) from UniREdit-Data-100K [29] that covers diverse knowledge dimensions, and expand them with textual reasoning traces generated by Gemini-2.5 Pro. Moreover, to ensure the training samples are generated without hallucinations, Gemini-2.5 Pro serves as a comprehensive evaluator to assess the generated images across three dimensions: instruction alignment, visual fidelity, and reasoning correctness. Only verified samples are retained to construct a high-quality training set for training visual synthesis with textual reasoning.

#### 4.2 Fine-grained Editing-like Visual Refinement

After the initial visual generation or editing, the draft already captures essential elements and semantically aligns with the input instruction and world knowledge, but inevitably contains imperfections that require fine-grained refinement. We therefore continue refining the results from the knowledge-enhanced initial synthesis. Specifically, the model reassesses the initial synthesized image considering prior textual reasoning, reflectively identifies and verbalizes inconsistencies and missing details. It then optionally incorporates a second round of textual reasoning, which accordingly refines semantic attributes, aesthetic details, stylistic coherence and instruction consistency to produce a polished image. This refinement process guided by textual reflection is structurally analogous to image editing, motivating us to create a synergistic loop for mutual improvement between T2I generation and image editing, by alternating knowledge-enhanced textual reasoning and editing-like visual refinement.

Data Preparation We design an agent pipeline to construct high-quality supervision data for training interleaved reasoning across both T2I generation and image editing tasks. The pipeline consists of (i) an initial generator (the base model) that produces a draft image with its textual reasoning from the input; (ii) a verifier (Gemini-2.5 Pro) that diagnoses caption–image mismatches and outputs structured, actionable edit directives across five dimensions: object presence, attribute accuracy, style consistency, realism, and aesthetic quality; (iii) a refinement teacher (Qwen-Image-Edit [17]) that applies the feedback and textual reasoning via instruction-guided image editing to obtain an improved image; and (iv) a final judge (Gemini-2.5 Pro) that performs comparative evaluation between the initial and refined images, retaining refined images only if they exhibit measurable improvements over the initial generation and faithfully reflects the verifier’s suggestion.

Specifically, we sample long-form captions from ShareGPT-4o-Image dataset [30] and short-form captions from midjourney prompts1 for T2I generation, and image-instruction pairs from UniREdit-Data-100K [29] for image editing. These inputs are fed to the initial generator for reasoning-augmented initial synthesis. The caption–image pairs then undergo the full verification, refinement and comparison cycle, resulting in a corpus of high-quality training data for image synthesis with multimodal interleaved reasoning.

#### 4.3 Two-stage Training Strategy

We adopt a simple yet effective two-stage supervised fine-tuning (SFT) strategy to first strengthen the foundational generation capability of the unified multimodal model, then train interleaved knowledgeenhanced reasoning and refinement capabilities across diverse image synthesis queries.

- Stage 1: Foundational Generation Strengthening In the first stage, we freeze the multimodal understanding branch of the base model and train only the generation branch. This stage focuses exclusively on image synthesis using existing T2I generation and image editing datasets without textual reasoning, aiming to enhance the instruction-following ability and foundational image synthesis capability.
- Stage 2: Interleaved Reasoning Tuning In the second stage, we unfreeze all model parameters and jointly train the understanding and generation branches using the curated interleaved reasoning data, including single-turn knowledge-enhanced reasoning samples and iterative visual refinement samples. This enables the model to perform world knowledge-enhanced reasoning and iteratively reflect and refine visual content. Specifically, for single-turn reasoning data, we supervise both the textual reasoning traces and the image synthesis outputs. For visual refinement data, we supervise textual reflections and refined images while leaving the initial reasoning text and visual draft unsupervised. The overall objective is formulated as

ℒ = 𝜆text ℒtext + 𝜆img ℒimg, (3)

where ℒtext denotes the text loss for supervising the reasoning tokens, and ℒimg denotes the image loss for supervising the synthesized images. 𝜆text and 𝜆img are scalar loss weights that balance the contributions of the text and image objectives, respectively.

- 1https://huggingface.co/datasets/vivym/midjourney-prompts

- Table 1 Evaluation of world knowledge-intensive text-to-image generation on the WISE [31] benchmark. "*" denotes generation with textual reasoning only, "†" denotes generation with both reasoning and refinement. The first block reports the performance of closed-source models. Bold entries represent the best performance among open-source models.

Model Cultural Time Space Biology Physics Chemistry Overall GPT-4o 0.81 0.71 0.89 0.83 0.79 0.74 0.80

Seedream 4.0 0.78 0.73 0.85 0.79 0.84 0.67 0.78 Unified Understanding and Generation w/o Reasoning. Harmon 0.38 0.48 0.52 0.37 0.44 0.29 0.41

Show-o 0.28 0.40 0.48 0.30 0.46 0.30 0.35 Janus Pro 0.30 0.37 0.49 0.36 0.42 0.26 0.35

MetaQuery-XL 0.56 0.55 0.62 0.49 0.63 0.41 0.55

BLIP3-o – – – – – – 0.62 UniWorld-V1 0.53 0.55 0.73 0.45 0.59 0.41 0.55

OmniGen2 0.42 0.52 0.64 0.43 0.50 0.34 0.47 Hunyuan-Image 3.0 0.58 0.57 0.70 0.56 0.63 0.31 0.57

Qwen-Image 0.62 0.63 0.77 0.57 0.75 0.40 0.62 Unified Understanding and Generation w/ Reasoning.

T2I-R1* 0.56 0.55 0.63 0.54 0.55 0.30 0.54 MindOmni* 0.75 0.70 0.76 0.76 0.72 0.52 0.71

IRG† 0.78 0.72 0.76 0.81 0.82 0.78 0.77 BAGEL* 0.76 0.69 0.75 0.65 0.75 0.58 0.70 UniCoT† 0.76 0.70 0.76 0.73 0.81 0.73 0.75

Ours† 0.80 0.68 0.79 0.77 0.83 0.81 0.78

- Table 2 Evaluation of knowledge-intensive image editing on KrisBench [32] and UniREditBench [33] benchmarks. "*" denotes textual reasoning only for editing, "†" denotes interleaved reasoning with both reasoning and refinement. Bold entries represent the best performance among open-source models.

|Model<br><br>|KrisBench Factual Conceptual Extract Procedural Overall<br><br>| |UniREditBench Real World Game World Overall| |
|---|---|---|---|---|
| | | | | |
|GPT-4o Gemini 2.0 Seedream 4.0<br><br>|79.80 81.37 78.32 65.26 59.65 62.90 – – –|80.09 62.41 –<br><br>|81.01 62.07 – – 66.22 45.38|73.39 – 55.77<br><br>|

Unified Understanding and Generation w/o Reasoning.

OmniGen2 57.36 44.20 47.79 49.71 53.69 33.14 43.41 Uniworld V1 47.71 44.80 47.92 50.27 – – – Lumina-DiMOO – – – – 51.44 45.61 48.54 LightFusion-World 66.69 63.50 52.38 61.85 – – –

Qwen-Image-Edit – – – – 70.95 41.92 56.52 Unified Understanding and Generation w/ Reasoning.

BAGEL* 66.18 61.92 49.02 60.18 56.80 45.10 50.96 UniCoT† 71.85 67.16 63.68 68.00 – – –

Ours† 70.67 72.38 56.89 68.23 74.82 65.30 70.06

### 5 Experiments

#### 5.1 Experimental Setup

Training Details In the first stage, the training corpus comprises nearly 7 million T2I samples and 500k image editing samples collected from open-source datasets including BLIP-3o [20], ShareGPT-4o-Image [30], Echo-4o-Image [34], OpenGPT4o-Image [35], Nano-banana-consist [36], and Pico-banana [37]. We train the model’s generation branch for 30,000 iterations using the Adam optimizer with a cosine learning rate schedule, including 3,000 warm-up steps, a maximum learning rate of 5 × 10−5 and a minimum learning rate of 1 × 10−5.

In the second stage, the training corpus consists of 150k self-constructed single-turn knowledge-enhanced reasoning samples for T2I generation, 100k image editing reasoning samples [29], and self-constructed interleaved reasoning samples, including 36k for T2I generation and 10k for image editing. We fine-tune all model parameters for 10,000 iterations with 1,000 warm-up steps, a maximum learning rate of 2 × 10−5 and a minimum learning rate of 1 × 10−6. Loss weights are set to 𝜆text = 2 and 𝜆img = 1, with a packed sequence length of 50k tokens.

- Table 3 Comparison of different models across general image generation and editing benchmarks. Bold entries represent the best performance among open-source models and underlined entries indicate the best performance among unified models with reasoning.

General T2I Generation General Image Editing

Type Model

GenEval DPGBench ImgEdit GEdit-EN

GPT-4o 0.84 85.15 4.20 7.53 Gemini 2.0 – – – 6.32

Closed-source

Seedream 4.0 0.84 88.25 4.18 7.68 Unified Understanding and Generation w/o Reasoning.

TokenFlow-XL 0.55 73.38 – – Harmon 0.76 – – –

Show-o 0.53 67.48 – – Janus Pro 0.80 84.19 – –

MetaQuery-XL 0.80 82.05 – – BLIP3-o 0.84 81.60 – –

UniWorld-V1 0.80 81.38 3.26 4.85

Open-source

Mogao 0.89 84.33 – – OmniGen2 0.80 83.57 3.43 6.41 MMaDA 0.63 69.97 – – Lumina-DiMOO 0.88 86.04 – – LightFusion-World – – 3.85 6.58 Hunyuan-Image 3.0 0.72 86.10 – –

Qwen-Image 0.87 88.32 – – Qwen-Image-Edit – – 4.27 7.56

Unified Understanding and Generation w Reasoning.

T2I-R1 – – – –

GoT 0.64 – – – Mind-Omni 0.83 82.50 – –

Open-source

IRG 0.85 – – – BAGEL 0.88 85.07 3.20 6.52 UniCoT 0.83 – – 6.74

Ours 0.90 86.21 4.06 6.94

Evaluation Setup We evaluate world knowledge reasoning and fine-grained semantic alignment for T2I generation using the WISE [31] benchmark, which comprises 1,000 world knowledge-informed prompts across culture, natural science, and spatial and temporal comprehension. For image editing, we use UniREditBench[29]with2,700meticulouslycuratedsamplescoveringbothreal-andgame-worldscenarios[38], and KrisBench [39] with 1,267 samples across factual, conceptual, and procedural knowledge to assess world knowledge reasoning and refinement capabilities. Additionally, we evaluate general compositional and instruction-followingabilities using GenEval [12]and DPGBench [40]for T2Igeneration, aswellasImgEdit [32] and GEdit-EN [33] for image editing.

#### 5.2 Main Results

We present a comprehensive comparison of our model against existing state-of-the-art unified multimodal models that support both generation and understanding in Tab. 1 and Tab. 2, for world knowledge-intensive T2I generation and image editing tasks, respectively. Detailed descriptions of the compared models are provided in Appendix A.1.

Our model achieves the best overall performance among open-source unified multimodal models, with or without explicit reasoning mechanisms, across knowledge-intensive image generation and editing tasks. Besides, it demonstrates comparable results to closed-source models, including Seedream 4.0 [41] and GPT-4o [42] on T2I generation, and even surpasses Gemini 2.0 [43] on KrisBench [39] and outperforms Seedream 4.0 [41] on UniREditBench [29]. These results highlight the effectiveness of our unified reasoning framework.

Moreover, as shown in the fine-grained breakdown of performance across different knowledge domains

- Table 4 Ablation study of UniReason. The base model is BAGEL [6]. “Two-Stage Training” refers to fine-tuning the base model using the two-stage training recipe, as described in Sec. 4.3.

Method WISE KrisBench UniREditBench Base Model 0.52 56.21 50.96

+ Two-Stage Training 0.58(+0.06) 61.53(+5.32) 63.37(+12.41) + Reasoning 0.73(+0.21) 64.12(+7.91) 67.30(+16.34) + Refinement 0.78 (+0.26) 68.23(+12.02) 70.06(+19.10)

in Tab. 1 and 2, our model exhibits broad and consistent world-knowledge coverage. Notably, it achieves the highest performance in Cultural Commonsense, Spatial Reasoning, Natural Science including Physics and Chemistry. For image editing tasks, it also demonstrates strong performance across diverse knowledge categories in both KrisBench and UniREditBench. Overall, our model’s knowledge-enhanced reasoning capabilities cover a wide range of tasks and domains.

#### 5.3 General Ability Retention

Beyond knowledge-intensive tasks, our model remains highly competitive on general image generation and editing benchmarks while improving knowledge-enhanced reasoning, demonstrating strong generalization capability. As shown in Tab. 3, on GenEval [12], our model surpasses leading systems, including QwenImage [17], GPT-4o [35], and Seedream 4.0 [41], without relying on any external LLM-based rewriting. On DPGBench [40], it achieves the best performance among models with reasoning mechanisms during generation, highlighting strong long-horizon instruction following. We further evaluate precise instructionfollowing image editing on ImgEdit [32] and GEdit-EN [33], which are essential for practical refinement. Our model delivers the strongest results among models with reasoning capability while remaining competitive with a broad range of existing approaches. These results indicate that our model is not only strong in reasoning-centric settings but also excels in general generation and editing, providing a robust and versatile unified foundation. Detailed results are shown in Appendix A.3 and case studies are shown in Appendix A.4.

#### 5.4 Ablation Study

We further investigate the contributions of the two-stage training strategy, as well as the reasoning and refinement mechanisms for image synthesis. On three knowledge-intensive generation and editing benchmarks, we compare three progressive settings built upon the BAGEL base model: (i) Two-Stage Training, which performs direct image generation after two-stage fine-tuning; (ii) + Reasoning, which elicits textual reasoning prior to image synthesis; and (iii) + Refinement, which further introduces an explicit reflection and refinement step to produce a final refined output.

Tab. 4 shows consistent improvement across all benchmarks as each component is added. The two-stage training alone effectively improves the base model’s instruction-following and synthesis capabilities. Then, introducing world knowledge-enhanced textual reasoning yields significant gains, especially on WISE with a +0.21 improvement. Finally, the visual refinement phase further improves the overall performance on all benchmarks. These results suggest that the two-stage training strategy injects both knowledge-enhanced reasoning and fine-grained refinement capabilities into the unified multimodal model, rather than merely enhancing surface-level visual composition. Moreover, the results highlight the importance of explicitly modeling implicit world knowledge during initial synthesis and performing fine-grained editing for further refinement.

#### 5.5 Correlation of Editing and Refinement

To show how image editing capability affects refinement effectiveness, we analyze performance gains with and without the refinement mechanism across models with varying editing capabilities. Specifically, we select different checkpoints during stage-1 training, each exhibiting different levels of editing proficiency, and apply identical stage-2 training to all checkpoints. We then evaluate performance on three knowledge-intensive benchmarks, measuring the gains achieved through refinement after initial textual reasoning. Fig. 3 plots these performance gains against the editing performance of each checkpoint on ImgEdit.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Figure 3 Correlation between image editing capability (ImgEdit score) and performance gains from refinement across three benchmarks. Higher editing proficiency leads to monotonically increasing refinement effectiveness.

The results reveal that performance gains from refinement increase monotonically with higher ImgEdit scores. This trend highlights the importance of jointly training image editing and T2I generation within a unified interleaved reasoning framework that integrates both textual reasoning and visual refinement. Since visual refinement relies on fine-grained and controllable editing, insufficient editing capacity can limit the effectiveness of reasoning-guided refinement.

### 6 Conclusion

In this paper, we introduce UniReason, a unified reasoning framework that harmonizes the text-to-image generation and image editing by exploiting their inherent structural synergies. Specifically, we proposed two complementary components: World Knowledge-Enhanced Textual Reasoning that infers implicit common sense and physical laws, and Fine-grained Editing-like Visual Refinement that enables iterative reflection and correction. By constructing high-quality datasets across five knowledge categories and employing a two-stage training strategy, UniReason demonstrates superior instruction following and visual fidelity. Extensive experiments on multiple benchmarks demonstrate that our unified reasoning approach achieves advanced performance across both T2I and editing tasks.

### 7 Impact Statement

This work focuses on improving reasoning and alignment in image generation and editing models. While such advances may benefit various creative and assistive applications, they may also introduce risks related to misuse of generated visual content. Addressing these risks requires system-level safeguards and responsible deployment practices beyond the scope of this paper.

### References

- [1] Jinheng Xie, Weĳia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhĳie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

- [2] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv: 2501.17811, 2025.

- [3] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025.

- [4] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

- [5] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

- [6] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv: 2505.14683, 2025.

- [7] Dongzhi Jiang, Ziyu Guo, Renrui Zhang, Zhuofan Zong, Hao Li, Le Zhuo, Shilin Yan, Pheng-Ann Heng, and Hongsheng Li. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv: 2505.00703, 2025.

- [8] Rongyao Fang, Chengqi Duan, Kun Wang, Linjiang Huang, Hao Li, Shilin Yan, Hao Tian, Xingyu Zeng, Rui Zhao, Jifeng Dai, Xihui Liu, and Hongsheng Li. Got: Unleashing reasoning capability of multimodal large language model for visual generation and editing. arXiv preprint arXiv: 2503.10639, 2025.

- [9] Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, and Ying Shan. Mindomni: Unleashing reasoning generation in vision language models with rgpo. arXiv preprint arXiv: 2505.13031, 2025.

- [10] Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, Junbo Qiao, Yue Guo, Yao Hu, Zhenfei Yin, Philip Torr, Yu Cheng, Wanli Ouyang, and Shaohui Lin. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv: 2509.06945, 2025.

- [11] Luozheng Qin, Jia Gong, Yuqing Sun, Tianjiao Li, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan, and Hao Li. Uni-cot: Towards unified chain-of-thought reasoning across text and vision. arXiv preprint arXiv: 2508.05606, 2025.

- [12] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. arXiv preprint arXiv: 2310.11513, 2023.

- [13] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [14] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [15] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

- [16] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eĳnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models wfith scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

- [17] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv: 2508.02324, 2025.

- [18] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [19] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [20] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv: 2505.09568, 2025.

- [21] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, Yatian Pang, and Li Yuan. Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv: 2506.03147, 2025.

- [22] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv: 2506.18871, 2025.

- [23] Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025.

- [24] Dianyi Wang, Ruihang Li, Feng Han, Chaofan Ma, Wei Song, Siyuan Wang, Yibin Wang, Yi Xin, Hongjian Liu, Zhixiong Zhang, Shengyuan Ding, Tianhang Wang, Zhenglin Cheng, Tao Lin, Cheng Jin, Kaicheng Yu, Jingjing Chen, Wenjie Wang, Zhongyu Wei, and Jiaqi Wang. Deepgen 1.0: A lightweight unified multimodal model for advancing image generation and editing. arXiv preprint arXiv: 2602.12205, 2026.

- [25] Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, and Jiaqi Wang. Autoregressive semantic visual reconstruction helps vlms understand better. arXiv preprint arXiv:2506.09040, 2025.

- [26] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv: 2502.14786, 2025.

- [27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv: 2209.03003, 2022.

- [28] Google. Gemini 2.5 pro. https://deepmind.google/models/gemini/pro/, 2025.
- [29] Feng Han, Yibin Wang, Chenglin Li, Zheming Liang, Dianyi Wang, Yang Jiao, Zhipeng Wei, Chao Gong, Cheng Jin, Jingjing Chen, and Jiaqi Wang. Unireditbench: A unified reasoning-based image editing benchmark. arXiv preprint arXiv: 2511.01295, 2025.

- [30] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv: 2506.18095, 2025.

- [31] Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv: 2503.07265, 2025.

- [32] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv: 2505.20275, 2025.

- [33] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv: 2504.17761, 2025.

- [34] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, and Weĳia Li. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025.

- [35] Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, Pengfei Wan, and Yi-Fan Zhang. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv: 2509.24900, 2025.

- [36] Nano-banana-150k. https://github.com/yejy53/Nano-banana-150k, 2024. GitHub repository.
- [37] Yusu Qian, Eli Bocek-Rivele, Liangchen Song, Jialing Tong, Yinfei Yang, Jiasen Lu, Wenze Hu, and Zhe Gan. Pico-banana-400k: A large-scale dataset for text-guided image editing. arXiv preprint arXiv: 2510.19808, 2025.

- [38] Jingqi Tong, Jixin Tang, Hangcheng Li, Yurong Mou, Ming Zhang, Jun Zhao, Yanbo Wen, Fan Song, Jiahao Zhan, Yuyang Lu, Chaoran Tao, Zhiyuan Guo, Jizhou Yu, Tianhao Cheng, Zhiheng Xi, Changhao Jiang, Zhangyue Yin, Yining Zheng, Weifeng Ge, Guanhua Chen, Tao Gui, Xipeng Qiu, Qi Zhang, and Xuanjing Huang. Game-rl: Synthesizing multimodal verifiable game data to boost vlms’ general reasoning. arXiv preprint arXiv: 2505.13886, 2025.

- [39] Yongliang Wu, Zonghui Li, Xinting Hu, Xinyu Ye, Xianfang Zeng, Gang Yu, Wenbo Zhu, Bernt Schiele, Ming-Hsuan Yang, and Xu Yang. Kris-bench: Benchmarking next-level intelligent image editing models. arXiv preprint arXiv: 2505.16707, 2025.

- [40] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv: 2403.05135, 2024.

- [41] ByteDance. Seedream 4.0, 2025. URL https://seed.bytedance.com/en/seedream4_0. Accessed: 2025-08.
- [42] OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/introducing-4o-image-generation/. Accessed: 2025.
- [43] Kat Kampf and Nicole Brichtova. Experiment with gemini 2.0 flash native image generation. Accessed: 05-08, 2025, March 2025. URL https://developers.googleblog.com/en/ experiment-withgemini-20-flash-native-image-generation/.
- [44] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation. Computer Vision and Pattern Recognition, 2024.

- [45] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, Jinbin Bai, Qian Yu, Dengyang Jiang, Yuandong Pu, Haoxing Chen, Le Zhuo, Junjun He, Gen Luo, Tianbin Li, Ming Hu, Jin Ye, Shenglong Ye, Bo Zhang, Chang Xu, Wenhai Wang, Hongsheng Li, Guangtao Zhai, Tianfan Xue, Bin Fu, Xiaohong Liu, Yu Qiao, and Yihao Liu. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv: 2510.06308, 2025.

- [46] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. arXiv preprint arXiv: 2505.15809, 2025.

- [47] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv: 2505.05472, 2025.

- [48] Siyu Cao, Hangting Chen, Peng Chen, Yĳi Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, Tiankai Hang, Duojun Huang, Jie Jiang, Zhengkai Jiang, Weĳie Kong, Changlin Li, Donghao Li, Junzhe Li, Xin Li, Yang Li, Zhenxi Li, Zhimin Li, Jiaxin Lin, Linus, Lucaz Liu, Shu Liu, Songtao Liu, Yu Liu, Yuhong Liu, Yanxin Long, Fanbin Lu, Qinglin Lu, Yuyang Peng, Yuanbo Peng, Xiangwei Shen, Yixuan Shi, Jiale Tao, Yangyu Tao, Qi Tian, Pengfei Wan, Chunyu Wang, Kai Wang, Lei Wang, Linqing Wang, Lucas Wang, Qixun Wang, Weiyan Wang, Hao Wen, Bing Wu, Jianbing Wu, Yue Wu, Senhao Xie, Fang Yang, Miles Yang, Xiaofeng Yang, Xuan Yang, Zhantao Yang, Jingmiao Yu, Zheng Yuan, Chao Zhang, Jian-Wei Zhang, Peizhen Zhang, Shi-Xue Zhang, Tao Zhang, Weigang Zhang, Yepeng Zhang, Yingfang Zhang, Zihao Zhang, Zĳian Zhang, Penghao Zhao, Zhiyuan Zhao, Xuefei Zhe, Jianchen Zhu, and Zhao Zhong. Hunyuanimage 3.0 technical report. arXiv preprint arXiv: 2509.23951, 2025.

- [49] Chenhui Gou, Zilong Chen, Zeyu Wang, Feng Li, Deyao Zhu, Zicheng Duan, Kunchang Li, Chaorui Deng, Hongyi Yuan, Haoqi Fan, Cihang Xie, Jianfei Cai, and Hamid Rezatofighi. Vq-va world: Towards high-quality visual question-visual answering. arXiv preprint arXiv: 2511.20573, 2025.

## Appendix

### A Appendix

#### A.1 Compared Baselines

We compared closed-source models including: GPT-4o [42], Gemini-2.0 [28], Seedream4.0 [41], as well as open-source advanced unified multimodal models models which support both multimodal understanding and high quality image generation including autoregressive unified models, such as Harmon [3], TokenFlowXL [44], and Janus-Pro [2]. Discrete diffusion-based approaches, including Lumina-DiMOO [45], MMaDA [46], and Show-o [1]. Another line of work connects VLMs and diffusion transformers via explicit connectors, exemplified by BLIP-3o [20], UniWorld-V1 [21], OmniGen2 [22], and the Qwen-Image series [17]. In contrast, deep fusion methods tightly integrate VLMs and DiTs within a unified architecture, such as Mogao [47], Hunyuan Image 3.0 [48] and LightFusion-World [49], the latter further enhanced with knowledge-centric fine-tuning.

Among open-source unified multimodal models that support naive reasoning, T2I-R1 [7], MindOmni [9], and BAGEL [6] primarily rely on textual reasoning to decompose abstract instructions into explicit semantic components that guide image generation. In contrast, GoT [8] introduces coordinate-based representations to provide explicit spatial guidance during synthesis. Another line of work, including IRG [10] and UniCoT [11], adopts interleaved reasoning mechanisms to reorganize semantics across modalities, progressively decomposing instructions into finer-grained and more structured descriptions for generation and refinement.

#### A.2 Data Preparation Details

To construct high-quality supervision for training UniReason across both text-to-image (T2I) generation and image editing tasks, we design a two-phase data construction pipeline that integrates world knowledge–enhanced textual reasoning with fine-grained editing-like visual refinement.

[Figure 7]

Figure 4 Overview of our data preparation framework.

- Phase I: World Knowledge–Enhanced Reasoning Data Construction We first build challenging instructions that require reasoning beyond pixel-level completion, covering five categories of world knowledge: (i) Cultural Commonsense, which resolves culturally grounded but underspecified entities using shared knowledge of history, customs, and symbols; (ii) Natural Science, which enforces consistency with physical, biological, medical, or chemical laws; (iii) Spatial Reasoning, which derives correct relative positions, orientations, viewpoints, and camera transformations; (iv) Temporal Reasoning, which models time-dependent state transitions and causal event sequences; and (v) Logical Reasoning, which translates explicit or implicit logical constraints into visually valid solutions. For T2I generation, we manually curate seed prompts grounded in Wikipedia and category definitions, then use Gemini-2.5 Pro [28] to expand them and generate corresponding textual CoT reasoning. Each prompt–reasoning pair is rendered into images using Qwen-Image [17], forming

- reasoning-grounded training samples. For image editing, we adopt triplets from UniREdit-Data-100K [29], augmented with Gemini-2.5 Pro–generated reasoning processes with category definitions. All samples are filtered by Gemini-2.5 Pro to ensure instruction alignment, visual fidelity, and knowledge-consistent reasoning, retaining only verified high-quality data.
- Phase II: Fine-grained Editing-like Visual Refinement Data Construction To further train interleaved reasoning and refinement capabilities, we design an agent-based pipeline to generate iterative refinement supervision. Given an input instruction, an initial generator produces a draft image along with textual reasoning. A verifier (Gemini-2.5 Pro) then diagnoses caption–image mismatches and outputs structured, actionable feedback along five dimensions: object presence, attribute accuracy, style consistency, realism, and aesthetic quality. A refinement teacher (Qwen-Image-Edit [17]) applies this feedback and textual reasoning via instruction-guided image editing to produce a refined image. Finally, a judge (Gemini-2.5 Pro) performs comparative evaluation between the initial and refined images, retaining refined results only if they demonstrate measurable improvements and faithfully reflect the suggested modifications. Concretely, we sample long-form captions from ShareGPT-4o-Image [30] and short-form captions from Midjourney prompts2 for T2I generation, and image–instruction pairs from UniREdit-Data-100K for image editing. These inputs undergo the full generation–verification–refinement–selection cycle, yielding a high-quality training set that jointly supports world knowledge–enhanced reasoning and fine-grained visual refinement.

#### A.3 Detailed Evaluation Results

We show the detailed evaluation results on general tasks include GenEval [12] shown in Tab. 5 and DPGBench [40] in Tab. 6 for T2I generation, as well as ImgEdit [32] and GEdit-EN [33] in Tab. 7 for image editing. The results show our model delivers the strongest results among models with reasoning while remaining competitive with a broad range of existing approaches. These results indicate that our model is not only strong in reasoning-centric settings but also excels in general generation and editing, providing a robust and versatile unified foundation.

Table 5 Evaluation of general text-to-image generation capabilities on GenEval [12] benchmark.

Model Single object Two object Counting Colors Position Attribution Overall GPT-4o 0.99 0.92 0.85 0.92 0.75 0.61 0.84

Seedream 4.0 0.99 0.92 0.72 0.91 0.76 0.74 0.84 Unified Understanding and Generation w/o Reasoning.

TokenFlow-XL 0.95 0.60 0.41 0.81 0.16 0.24 0.55 Harmon 0.99 0.86 0.66 0.85 0.74 0.48 0.76

Show-o 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Janus Pro 0.99 0.89 0.59 0.90 0.79 0.66 0.80

MetaQuery-XL – – – – – – 0.80

BLIP3-o – – – – – – 0.84 UniWorld-V1 0.99 0.93 0.79 0.89 0.49 0.70 0.80

Mogao 1.00 0.97 0.83 0.93 0.84 0.80 0.89 OmniGen2 1.00 0.95 0.64 0.88 0.55 0.76 0.80

MMaDA 0.99 0.76 0.61 0.84 0.20 0.37 0.63 Lumina-DiMOO 1.00 0.94 0.85 0.89 0.85 0.76 0.88

Hunyuan-Image 3.0 1.00 0.92 0.48 0.82 0.42 0.63 0.72 Qwen-Image 0.99 0.92 0.89 0.88 0.76 0.77 0.87

Unified Understanding and Generation w Reasoning.

GoT – – – – – – 0.64 Mind-Omni 0.99 0.94 0.71 0.90 0.71 0.71 0.83

IRG 0.98 0.94 0.83 0.86 0.74 0.73 0.85 BAGEL 0.98 0.95 0.84 0.95 0.78 0.77 0.88

Uni-CoT 0.99 0.96 0.84 0.92 0.57 0.71 0.83 Ours 1.00 0.96 0.82 0.90 0.88 0.82 0.90

- 2https://huggingface.co/datasets/vivym/midjourney-prompts

Table 6 Evaluation of general text-to-image generation capabilities on DPG [12] benchmark.

Model Global Entity Attribute Relation Other Overall

GPT-4o 88.89 88.94 89.84 92.63 90.96 85.15 Seedream 4.0 94.10 92.28 92.75 93.67 92.77 88.25

Unified Understanding and Generation w/o Reasoning.

TokenFlow-XL 78.72 79.22 81.29 85.22 71.20 73.38 Show-o – – – – – 67.48 Janus Pro 86.90 88.90 89.40 89.32 89.02 84.19 MetaQuery-XL – – – – – 82.05 BLIP3-o – – – – – 81.60 UniWorld-V1 83.64 88.39 88.44 89.27 87.22 81.38 Mogao 82.37 90.03 88.26 93.18 85.40 84.33 OmniGen2 88.81 88.83 90.18 89.37 90.27 83.57 MMaDA 77.81 78.48 81.74 84.79 63.20 69.97 Lumina-DiMOO 81.46 92.08 88.98 94.31 82.00 86.04 Hunyuan-Image 3.0 92.12 92.53 89.13 92.13 91.92 86.10 Qwen-Image 91.32 91.56 92.02 94.31 92.73 88.32

Unified Understanding and Generation w Reasoning.

Mind-Omni 89.10 – – – 89.20 82.50 BAGEL 88.94 90.37 91.29 90.82 88.67 85.07 Ours 91.78 91.23 90.76 91.12 92.27 86.21

Table 7 Evaluation of general image editing capabilities on ImgEdit [32] and GEdit-EN [33] benchmarks.

|Model|ImgEdit<br><br>| |GEdit-EN|
|---|---|---|---|
| |Add Adjust Extract Replace Remove Background Style Hybrid Action<br><br>|Overall<br><br>|G_SC G_PQ G_O|
|GPT-4o Gemini 2.0 Seedream 4.0<br><br>|4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 – – – – – – – – – 4.52 4.41 2.93 4.56 4.44 4.30 4.76 3.33 4.36|4.20 – 4.18<br><br>|7.85 7.62 7.53<br><br>6.73 6.61 6.32<br><br>8.24 8.08 7.68<br>|

Unified Understanding and Generation w/o Reasoning.

Janus 4o 3.60 – 2.28 3.27 2.28 3.32 4.47 2.74 4.13 3.26 – – – UniWorld-V1 3.82 3.66 2.31 3.45 3.02 2.99 4.71 2.96 2.74 3.26 4.93 7.43 4.85

OmniGen2 3.74 3.54 1.77 3.21 2.77 3.57 4.81 2.30 4.14 3.43 7.16 6.77 6.41 LightFusion-World 4.33 3.37 1.25 4.63 3.74 4.24 4.69 3.91 4.45 3.85 7.00 7.29 6.58

Qwen-Image-Edit 4.38 4.16 3.43 4.66 4.14 4.38 4.81 3.82 4.69 4.27 8.00 7.86 7.56

Unified Understanding and Generation w Reasoning.

BAGEL 3.56 3.31 1.88 2.62 2.88 3.44 4.49 2.38 4.17 3.20 7.36 6.83 6.52 UniCoT – – – – – – – – – – 7.91 6.24 6.74

Ours 4.14 4.06 2.49 4.42 4.31 4.23 4.65 2.58 4.68 4.06 7.46 7.66 6.94

#### A.4 Case Study

We present additional UniReason results on both T2I generation and image editing tasks in Fig. 5. The results demonstrate that, while maintaining high-quality T2I generation and image editing performance, UniReason exhibits strong reasoning capabilities, enabling it to handle complex scenarios such as maze navigation, temporal evolution, and spatial camera viewpoint transformations. Moreover, UniReason shows robust refinement ability, effectively correcting fine-grained details such as faces, text, and hand gestures, thereby improving the quality of the initial images and rectifying errors introduced during the initial generation.

[Figure 8]

###### Figure 5 Qualitative results of UniReason on both T2I generation (blue column) and image editing task (orange column).

