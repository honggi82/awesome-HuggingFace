[Figure 1]

## Scone: Bridging Composition and Distinction in Subject-Driven Image Generation via Unified Understanding-Generation Modeling

Yuran Wang1,2* Bohan Zeng1,2* Chengzhuo Tong1,2 Wenxuan Liu1 Yang Shi1,2 Xiaochen Ma4 Hao Liang1 Yuanxing Zhang2 Wentao Zhang1,3,5† 1Peking University 2Kling Team, Kuaishou Technology 3Zhongguancun Academy 4HKUST 5Beijing Key Laboratory of Data Intelligence and Security (Peking University)

# arXiv:2512.12675v3[cs.CV]9Jun2026

### Abstract

(a) Problem

(b) Challenge 1

| |
|---|

[Figure 2]

: Target

|Understanding Generation|
|---|

[Figure 3]

| |
|---|

[Figure 4]

The third boot from the right in Image 1 lies on a muddy path.

Subject-driven image generation has advanced from single- to multi-subject composition, while neglecting distinction, the ability to distinguish and generate the correct subject when inputs contain multiple candidates. This limitation restricts effectiveness in complex, realistic visual settings. We propose Scone, a unified understandinggeneration method that integrates composition and distinction. Scone enables the understanding expert to act as a semantic bridge, conveying semantic information and guiding the generation expert to preserve subject identity while minimizing interference. A two-stage training scheme first learns composition, then enhances distinction through semantic alignment and attention-based masking. We also introduce SconeEval, a benchmark for evaluating both composition and distinction across diverse scenarios. Experiments demonstrate that Scone outperforms existing opensource models in composition and distinction tasks on two benchmarks. Our model, benchmark, and training data are available at: https://github.com/Ryann-Ran/Scone.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

###### Scone (Ours)

GPT-4o

[Figure 9]

[Figure 10]

(c) Challenge 2

###### Und. Und. + Gen.

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

The baby lion in image 1 is gazing toward the horizon.

Success Subject error

Gemini-2.5 -Flash-Image

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

USO

[Figure 27]

[Figure 28]

[Figure 29]

Subject redundancy

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

The bird with a purple neck and a blue belly is walking on street.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Subject omission Subject redundancy

[Figure 46]

[Figure 47]

[Figure 48]

Success

Subject error

Figure 1. The distinction problem and challenges. (a) Problem. State-of-the-art methods have limitations in distinguishing target subjects specified by the instruction. (b) Challenge 1: semantic deficiency in generation. Reference image information from the understanding and generation experts in the unified model is used to compute semantic similarity with instruction. (c) Challenge 2: biased understanding and misaligned generation. “Und.” and “Und.+Gen.” indicate whether textual information from generation expert in the unified model is included to collaborate with understanding expert. The unified model is BAGEL [6].

### 1. Introduction

subject combinations while neglecting the ability to distinguish target subjects in complex contexts. As shown in Fig. 1(a), although current models can combine multiple subjects, they may fail to distinguish and generate the correct target subject when a reference image contains multiple candidates, leading to problems such as subject omissions (none of the candidate subjects appear) or errors (misidentification of the target subject). Real-world images often involve interference and intricate details [20, 34], further limiting practical performance. Thus, we emphasize examining the input subjects themselves, focusing on the model’s ability to distinguish the target subject within complex contexts and leverage this information for generation.

Image generation methods [8, 9, 38] have demonstrated exceptional capabilities, enabling the generation of desired images across diverse scenarios [37]. Subject-driven image generation has recently gained significant attention, with the focus evolving from single-subject to multi-subject generation, incorporating more input images. Existing methods [38, 39, 41, 42] can process two or more input images and combine subjects based on instructions. Moreover, methods such as [9, 46] extend this capability by accepting more than four images, showcasing potential for more complex composition tasks.

However, existing works primarily focus on expanding

A core challenge is extracting useful information from complex references, which remains difficult for generation models. Subject distinction relies on semantic under-

*Equal contribution †Corresponding author: wentao.zhang@pku.edu.cn

standing of instruction’s expression of references, where understanding models are more proficient [1, 18, 49]. As shown in Fig. 1(b), in a unified understanding-generation model consisting of an understanding expert and a generation expert, the information encoded by the understanding expert is more similar to the instruction, which means more aligned with instruction than that encoded by the generation expert. This reveals generation models’ deficiency and understanding model’s advantage in interacting with instructions and semantically understanding reference information. However, this semantic advantage of understanding models is not entirely reliable: understanding models often exhibit biases [15, 19, 33, 48], which become problematic when directly used to assist generation. As illustrated in Fig. 1(c), in a unified model, relying only on semantic information from understanding expert still struggles to prevent irrelevant subjects from appearing, and subject errors persist even with correct semantic information due to misalignment between generation and understanding experts.

Compared with generation models, unified models offer a clear advantage for subject-driven image generation because the understanding expert captures semantic cues earlier than the generation expert [51], as illustrated in Fig. 2(a). These early-layer semantics highlight instruction-relevant regions such as candidate subjects and enable more accurate distinction in complex reference images. Moreover, to alleviate bias introduced by the understanding expert, the unified architecture allows end-to-end collaboration, as shown in Fig. 2(b). The understanding expert refines its semantic interpretation through feedback from generation, and the generation expert aligns with these cues to better preserve subject-related details.

Based on these insights, we propose a subject-driven image generation method, Scone (Subject composition and distinction enhancement), built upon a unified understanding-generation model capable of handling subject composition and distinction. Our method leverages the strong understanding capabilities of the understanding expert to overcome the limitations of the generation expert in complex contexts involving reference images and instructions. Specifically, Scone enables the understanding expert to act as a semantic bridge conveying high-level semantic information to guide generation, which called understanding bridge strategy. In the first training stage, the model learns subject composition on single-candidate data (i.e. a reference image contains only one candidate subject). In the second stage, the understanding expert is trained to align visual and textual representations and filter instruction-irrelevant regions using a semantic mask derived from early layer, forming a robust semantic bridge. After this formation, the understanding expert provides semantic guidance to the generation expert, ensuring that subjectrelated information is emphasized while unrelated interfer-

Input Similarity and masked image

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

The baby lion in image 1 is gazing…

Understanding expert Generation expert

(a) Distinction with understanding guidance via unified modeling

Input Output

semantics

[Figure 56]

[Figure 57]

[Figure 58]

The baby lion in image 1 is gazing…

###### Und. Expert Gen. Expert

Feedback

[Figure 59]

[Figure 60]

(b) End-to-end understanding-generation collaboration

Figure 2. Our motivation. (a) visualizes the early similarity between image token from the understanding and generation experts and text token within the unified model, showing that the former attends to semantic regions while the latter is less sensitive. (b) illustrates the collaboration between the understanding and generation experts within the unified model through end-to-end training.

ence is suppressed. This design enables Scone to distinguish useful reference information and achieve precise subject composition in complex multi-subject contexts through internal understanding, without relying on external models, additional parameters, or test-time techniques. Unlike methods with external understanding modules, our unified architecture jointly learns semantics and generation, avoiding extra latency and preserving end-to-end optimization. As shown in Fig. 1(a), compared to existing methods, our method more accurately distinguishes relevant reference information and generates ideal results.

Furthermore, to evaluate whether existing models can genuinely distinguish subjects in reference images based on instructions and use relevant information to generate the correct target subject, we introduce a new benchmark, SconeEval, which includes subject-driven image generation tasks with varying difficulty levels, including composition, distinction, and distinction & composition. This benchmark provides a comprehensive evaluation from both composition and distinction perspectives.

Our main contributions are threefold:

- • We propose the Scone model, which supports multisubject composition and excels in subject distinction in complex contexts, ranking first among open-source models on OmniContext benchmark.
- • We introduce the understanding bridge strategy, which transforms the understanding expert into a semantic bridge, enabling early multimodal alignment and attention-based semantic masking to guide the generation expert, enhancing subject distinction and semantic fidelity without adding extra parameters.
- • We develop SconeEval, a challenging benchmark to evaluate subject-driven image generation from both composition and distinction perspectives.

### 2. Related work

Text Target image

…

- 2.1. Subject-driven image generation

Early subject-driven generation rely on fine-tuned diffusion models [36, 45, 47] with image conditions for flexible customization. With the rise of Diffusion Transformer [25], generation quality improves significantly. Recent methods [11, 14, 31, 42] extend single-subject to multi-subject composition, but they typically assume clean references and are prone to interference from irrelevant information under complex conditions, causing deviations from the target subject. Although methods like SSR-Encoder [50] aim to isolate features, they handle only simple prompts with a single reference, reflecting limited understanding and restricted effectiveness under complex instructions or noisy inputs.

- 2.2. Unified understanding-generation models

To advance general-purpose agents, several methods [4– 6, 16, 17, 29, 43, 44] integrate multimodal understanding and generation tasks within a unified architecture. By leveraging multimodal understanding, these methods enhance the stability of image generation when handling complex instructions. Some methods [2, 39, 46] use this capability for subject-driven generation. However, when reference images contain substantial irrelevant content, existing unified models lack effective mechanisms to prevent interference, often resulting in unwanted subjects. We address this gap by using understanding semantics to better distinguish target conditions and guide cleaner, more reliable generation.

- 3. The Scone model

VAE token ViT token

Text token Target token

###### 🔥 Und. Expert ❄ Gen. Expert

[Figure 61]

[Figure 62]

###### Step 1:

[Figure 63]

ViT hidden states 𝑺𝒊𝒎𝒊𝒍𝒂𝒓𝒊𝒕𝒚(·) Text hidden states

Early layer

Eq.(4)

Eq. (2)

Semantic mask

[Figure 64]

Later layer

ViT hidden states

VAE hidden states Target hidden states

[Figure 65]

Step 2: 🔥 Und. Expert 🔥 Gen. Expert

[Figure 66]

[Figure 67]

Figure 3. Understanding bridge strategy. Step 1: Understanding bridge formation. Early semantic alignment and attention masking enable the understanding expert to serve as the semantic bridge. Step 2: Understanding bridge guidance. The generation expert is optimized under the guidance of the semantic bridge, enabled by unified understanding-generation modeling.

The understanding expert refines its semantics through generation feedback, and the generation expert aligns with these cues to preserve subject-related details in complex reference images.

#### 3.2. Unified understanding-generation modeling

We adopt BAGEL [6] as base, a Mixture-of-TransformerExperts architecture that handles understanding and generation information through dedicated experts sharing multimodal attention. We optimize it using the original MSE loss, with no additional parameters. For subject-driven generation, image tokens from Vision Transformer (ViT) encoder and instruction tokens are handled by the understanding expert, while image tokens from the VAE model are processed by the generation expert. To improve distinction in complex contexts, the understanding expert acts as a semantic bridge that provides discriminative cues for generation.

We present Scone (Subject composition and distinction enhancement), which supports multi-subject composition and demonstrates strong distinction capability in complex contexts via unified understanding-generation modeling.

#### 3.1. Motivation and preliminaries

Distinction with understanding guidance via unified modeling Unified models outperform generation models in complex semantics due to stronger understanding ability and better text-image alignment [32]. The understanding expert captures instruction-relevant semantics earlier than the generation expert, before textual features emerge [32, 51]. As shown in Fig. 2(a), early-layer similarity with text token indicates that the understanding expert attends to key subject regions, while the generation expert is less semantically sensitive.

#### 3.3. Stage I: Composition training

We first finetune BAGEL on single-candidate data, where each reference image contains a single subject. The understanding expert and generation expert (including corresponding MLP connectors) are trained, while the ViT and VAE remain frozen. One epoch of base data enables both single- and multi-subject generation. A refined dataset is then used for another epoch to further enhance subject consistency. Training data details are provided in Sec. 5.1.

#### 3.4. Stage II: Distinction training with understanding bridge strategy

End-to-end understanding-generation collaboration The understanding expert may introduce semantic bias, leading to subject errors or redundancy. Unified modeling enables end-to-end collaboration, as shown in Fig. 2(b).

We propose the understanding bridge strategy, which enables the understanding expert to act as a semantic bridge

that transfers high-level semantic information for generation guidance, as shown in Fig. 3. It comprises two steps: forming the semantic bridge via multimodal alignment and guiding generation through this bridge. This design improves subject identity preservation, relevance discrimination, and contextual fidelity. Multi-candidate data are introduced for distinction-aware composition.

- Step 1: Understanding bridge formation The understanding expert jointly learns visual and textual semantics

to become the semantic bridge. Let hv = {hvi }N

v

i=1 and ht = {htj}N

t

j=1 denote early-layer visual and textual hidden states in understanding expert, respectively. We apply L2-normalization and compute the cosine similarities as:

S = Hˆ v(Hˆ t)⊤, Si,j = hˆvi · hˆtj =

hvi ∥hvi ∥2

·

htj ∥htj∥2

. (1) The semantic relevance for each visual token is defined as:

si =

1 Nt

Nt

j=1

Si,j. (2)

We construct a binary semantic mask M based on a threshold τ (with parameter study in Sec. 5.4), which influences the number of reference image tokens that remain invisible to the target image tokens in generation expert. Rather than discarding tokens, the mask modifies the attention logits. For logits A mapping target tokens to reference image tokens in subsequent layers, we apply the mask as follows:

A˜k,i = Ak,i + Mi, Mi =

0, si > τ, −∞, otherwise.

(3)

Tokens where Mi = −∞ receive zero attention, which allows target tokens to disregard irrelevant regions. This mechanism establishes the understanding expert as a semantic bridge to align representations and suppress semantic interference. We train the model for 1k steps.

- Step 2: Understanding bridge guidance Functioning as the semantic bridge, the understanding expert guides the generation expert. We train both experts for an additional 1k steps to align generation representations with the bridge and focus on key regions identified by the understanding expert. This phase enforces semantic consistency within complex compositional scenarios.

### 4. The SconeEval benchmark

#### 4.1. Overview

Existing benchmarks usually offer simple contexts where the reference image contains a single prominent subject referred to by a basic category term. Such settings fail to

reflect performance in real-world images with substantial interference and less structured compositions. They also mainly assess subject reproduction and composition using similarity metrics from models such as DINOv2 [24] and CLIP [27]. In multi-subject settings, averaging similarity across subjects cannot reliably measure generation quality, especially when subject omission or redundancy occurs.

To evaluate a model’s ability to distinguish and generate referred subjects in complex contexts, we introduce SconeEval, a benchmark with 409 test cases spanning character, object, and scene combinations and subject distinction. It covers 19 case types in Fig. 4(a) and 6 subtasks in Fig. 4(b), enabling comprehensive evaluation of subject distinction and feature utilization. Unlike traditional benchmarks emphasizing visual fidelity or text alignment, SconeEval focuses on cross-modal reasoning over complex reference images and instructions, requiring the model to decide whom to generate among multiple candidates. SconeEval includes three progressively challenging tasks, as shown in Fig. 4(c): composition, distinction, and distinction & composition. In composition, each reference image contains one subject, and one or more images are used for single- or multi-subject generation. In distinction, each reference image contains multiple subjects, and the model must generate the target one. The distinction & composition task combines both settings, requiring multi-subject generation from reference images that each contain multiple subjects. Tasks involving distinction include cross-category and intra-category cases, indicating whether candidate subjects in a reference image belong to the same category. As shown in Tab. 1, existing benchmarks mainly evaluate subject composition in simple contexts, while our benchmark targets more realistic scenarios.

#### 4.2. Construction pipeline

- Step 1: Image collection Images are collected from three sources. (1) Existing benchmarks: We use Qwen3-VL30B-A3B-Instruct [35] to filter images, recognize subjects, and classify subject categories, followed by manual verification to ensure each image contains only one subject. Samples are drawn from DreamBench++ [26] and OmniContext [39]. (2) T2I (text-to-image) synthesis and (3) Open access: To increase category diversity, we further supplement the benchmark with single-candidate images synthesized by Flux.1-dev [13] and additional open-access samples. Finally, we build a single-candidate image pool covering three categories, character, object, and scene, with 15 subcategories and at least 30 images per subcategory. The images are grouped into sets of 1 to 4 and split into two subsets for subsequent single- and multi-candidate data construction.
- Step 2: Multi-candidate editing We create multicandidate images by adding other subjects to single-

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Task 1: Composition Easy Intermediate

###### Task 2: Distinction

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Single-Subject Multi-Subject Cross-Category Intra-Category

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Reference Instruction Reference Reference

Instruction Reference Instruction Instruction

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

The minion figurine is sitting

The woman in figure 1 is

The bird in the first image flies

The black dog wearing a Santa

atop a stack of game cartridges.

standing beside the man in fig 2.

through the starlight.

hat standing in a snowy backyard.

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Difficult

19 Case types 6 Subtasks 3 Levels

###### Task 3: Distinction & Composition

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Cross-Category

Intra-Category

SconeEval

[Figure 109]

[Figure 110]

Reference Instruction Instruction

Reference

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

The silver hair dryer on far left from image 1 is resting beside the computer on

[Figure 118]

The Lego minifigure is perched on the back of the white stork.

[Figure 119]

[Figure 120]

/

Single-Subject Generation Multi-Subject Generation

[Figure 121]

the far left from image 2.

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

(a) Distribution of case type (b) Distribution of task type (c) Task demonstration

Figure 4. Overview of SconeEval. “Char”: character, “Obj”: object, “Sce”: scene. SconeEval evaluates target subject identification and generation in complex contexts, with 409 test cases across three domains, 19 case types, and 6 subtasks, covering composition, distinction, and distinction & composition tasks.

Table 1. Task comparison of existing benchmark for subjectdriven image generation.

candidate images with Qwen-Image-Edit-2509 [38], as shown in Fig. 5, followed by manual verification to ensure each subject is clearly recognizable.

Benchmark Composition Distinction Distinction & Composition DreamBench [28] ✓ ✗ ✗

- Step 3: Instruction construction We adopt a two-step decoupling strategy that separates visual understanding from text generation, reducing cross-image interference and improving subject identification and linguistic coherence. Instructions explicitly specify target indices or distinct features to avoid ambiguity (e.g. “Image 1”, “the man with green hair”). Step 1: Subject identification (imageto-text). Each image is processed independently by the VLM model Qwen3-VL-30B-A3B-Instruct [35] to identify its most prominent subject, minimizing mutual interference. For single-candidate images, we extract direct subject names (e.g. “woman”); for multi-candidate images, we generate referential names with distinctive cues such as attribute, size, and position (e.g. “woman on the left of the image”), guided by the corresponding single-candidate images. For scene images, we provide detailed descriptions to support interaction-related instructions (e.g. “place the bird on the shelf”). Step 2: Instruction generation (text-to-text). Only the subject names or scene descriptions from Step 1 are fed to the LLM model Qwen3-30B-A3B-Instruct2507 [35], without image inputs.

DreamBench++ [26] ✓ ✗ ✗

OmniContext [39] ✓ ✗ ✗ XVerseBench [3] ✓ ✗ ✗ SconeEval (Ours) ✓ ✓ ✓

Multi-candidate editing

[Figure 126]

[Figure 127]

"The pepper in Figure 1 is resting on…"

Cross-category

User

“Add a toy”

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Scone (Ours)

GPT-4o

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Singlecandidate

[Figure 140]

Multi-candidate

[Figure 141]

[Figure 142]

Subject redundancy

Success

[Figure 143]

[Figure 144]

Intra-category "The man with curly hair is tying his shoelaces…"

“Add two men”

User

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

Scone Single- (Ours)

GPT-4o

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Multi-candidate

candidate

[Figure 158]

[Figure 159]

Subject redundancy

Success

Figure 5. Multi-candidate editing in SconeEval construction. Task difficulty increases with image and instruction complexity.

#### 4.3. Evaluation protocol

sion, recall, and F1. Precision and recall reflect redundancy and omission, and the distinction score is defined as the average of accuracy and F1. To align with the composition scale, we rescale the distinction score from [0,1] to [0,10]. The overall score averages composition and distinction.

Following VIEScore [12] and OmniContext [39], we use GPT-4.1 [22] to score composition capability on a 0–10 scale with rationales, covering prompt following and subject consistency. Unlike automatic metrics such as VSM [7] and AlphaCLIP [30], which require additional masks, GPT-

### 5. Experiments

- 4.1 provides anchor-free evaluation for complex multicondition cases. Prompt for composition scoring is similar to OmniContext [39], but evaluates subject consistency only for the target subject. For distinction evaluation, GPT-4.1 judges whether the described reference subject appears in the target image, from which we compute accuracy, preci-

#### 5.1. Implementation details

Training data We collect a large-scale pool of opensource subject-driven generation datasets, including X2I [42], MUSAR-Gen [10], UNO-1M [41], and Echo-

- Table 2. Quantitative comparison of existing models on OmniContext [39] benchmark. “Char. + Obj.” indicates Character + Object. † indicates our base model. Best scores in each group are highlighted in bold.

Method

SINGLE ↑ MULTIPLE ↑ SCENE ↑ Character Object Character Object Char. + Obj. Character Object Char. + Obj. Average ↑ Closed-source model

Gemini-2.5-Flash-Image [9] 8.79 9.12 8.27 8.60 7.71 7.63 7.65 6.81 8.07 GPT-4o [23] 8.96 8.91 8.90 8.95 8.81 8.92 8.40 8.44 8.78

Generation model

FLUX.1 Kontext [dev] [14] 8.07 7.97 - - - - - - UNO [41] 7.15 6.72 3.56 6.46 4.90 2.72 4.89 4.76 5.14 USO [40] 8.03 7.55 3.32 6.10 4.56 2.77 5.38 5.09 5.35 UniWorld-V2 [16] 8.45 8.44 7.87 8.22 7.95 5.36 7.47 6.98 7.59 Qwen-Image-Edit-2509 [38] 8.56 8.41 7.92 8.37 7.79 5.23 7.70 6.86 7.60

Unified model

BAGEL† [6] 7.00 7.04 5.32 6.69 6.74 3.94 5.77 5.73 6.03 OmniGen2 [39] 8.17 7.63 7.26 7.03 7.56 7.02 6.90 6.64 7.28 Echo-4o [46] 8.34 8.27 8.13 8.14 8.11 7.07 7.73 7.77 7.95

Scone (Ours) 8.34 8.52 8.24 8.14 8.30 7.06 7.88 7.63 8.01

- Table 3. Quantitative comparison of existing models on our SconeEval benchmark. † indicates our base model. “COM”: Composition score. “DIS”: Distinction score. Best scores in each group are highlighted in bold.

Composition ↑ Distinction ↑ Distinction & Composition ↑ Single Multi Cross Intra Cross Intra Average ↑ COM COM COM DIS COM DIS COM DIS COM DIS COM DIS Overall

Method

Closed-Source Model

Gemini-2.5-Flash-Image [9] 8.87 7.94 9.12 9.15 9.00 8.50 8.27 8.87 8.17 8.85 8.56 8.84 8.70 GPT-4o [23] 8.92 8.51 9.18 8.55 9.45 9.01 8.83 8.49 8.99 9.56 8.98 8.90 8.94

###### Generation Model

FLUX.1 Kontext [dev] [14] 7.92 - 7.93 8.45 6.20 6.11 - - - - - - USO [40] 8.03 5.19 7.96 8.50 7.14 6.51 5.10 6.25 5.07 5.57 6.41 6.71 6.56 UNO [41] 7.53 5.38 7.27 7.90 6.76 6.53 5.27 7.02 5.61 6.27 6.31 6.93 6.62 UniWorld-V2 [16] 8.41 7.16 8.63 8.24 7.44 6.77 7.52 8.03 7.70 7.24 7.81 7.57 7.69 Qwen-Image-Edit-2509 [38] 8.54 6.85 8.85 8.57 7.32 6.86 7.53 8.13 7.49 7.02 7.76 7.65 7.70

###### Unified Model

BAGEL† [6] 7.14 5.55 7.49 7.95 6.93 6.21 6.44 7.38 6.87 7.27 6.74 7.20 6.97 OmniGen2 [39] 8.00 6.59 8.31 8.99 6.99 6.80 7.28 8.30 7.14 7.13 7.39 7.81 7.60 Echo-4o [46] 8.58 7.73 8.36 8.33 7.74 7.18 7.87 8.72 8.01 8.33 8.05 8.14 8.09

Scone (Ours) 8.52 7.40 8.98 9.73 7.97 7.74 8.20 9.25 8.21 8.44 8.21 8.79 8.50

4o-Image [46]. We further synthesize 15K samples with 3-4 input images to supplement missing data types. Data categories cover character, object, and scene with predefined object types and attributes. GPT-4o [21] generates prompts, instructions, and descriptions from random attribute combinations, FLUX.1-dev [13] produces input images, and Gemini-2.5-Flash-Image [9] generates the final outputs. (1) For training stage I: We randomly select 70K base single-candidate samples from the data pool and further filter 22K refined single-candidate samples using Qwen3-VL-30B-A3B-Instruct [35] by scoring subject consistency and instruction following. (2) For training stage II: Besides the refined single-candidate

data, we construct 20K multi-candidate samples from another filtered subset. For image acquisition, Qwen330B-A3B-Instruct-2507 [35] extracts subject names and scene descriptions from single-candidate instructions, and Qwen-Image-Edit-2509 [38] adds cross- or intra-category subjects to create multi-candidate images. The original image serves as the target. For instruction construction, original instructions are reused for cross-category data. For intra-category data, Qwen3-VL-30B-A3B-Instruct [35] identifies the new subject introduced during editing, and Qwen3-30B-A3B-Instruct-2507 [35] replaces the subject name in the original instruction (e.g. “the woman” → “the woman on the left of the image”).

Gemini-2.5 -Flash-Image

Qwen-lmage -Edit-2509

Reference image Instruction GPT-4o

Scone (Ours) OmniGen2 BAGEL

Echo-4o

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

I wish the person and the man would stare at each other.

[Figure 169]

The jaguar emblem is perched atop a weathered stone pedestal near an ornate door, the shiny black beetle crawls along the textured surface of the building's foundation.

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

The butterfly flutters gently above the vase of white flowers on the dining table, its vibrant wings contrasting with the rustic wood.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

The figure smiles warmly at the café table, while the seagull curiously observes the pastéis de nata from its perch nearby.

Figure 6. Qualitative comparison of existing models on OmniContext [39] benchmark.

- Table 4. Ablation results for stage I. Evaluated on OmniContext benchmark. “PF”: prompt following. “SC”: subject consistency.

Version Training data PF ↑ SC ↑ Overall ↑ BAGEL [6] - 6.74 5.73 6.03

- Stage I, step 1 Base single-candidate data (70K) 7.83 8.27 7.95
- Stage I, step 2 Refined single-candidate data (22K) 7.92 8.31 8.02

- Table 5. Ablation results for stage II. Evaluated on SconeEval benchmark. “COM”: composition. “DIS”: distinction.

Stage Version COM ↑ DIS ↑ Overall ↑ Base BAGEL [6] 6.74 7.20 6.97

- Stage I - 7.94 7.78 7.86

- Stage II

- (a) Direct 7.64 8.23 7.94
- (b) Two-step, w/o bridge 8.15 8.70 8.43
- (c) Two-step, w/ bridge (Ours) 8.21 8.79 8.50

- Table 6. Parameter study of threshold in stage II. Evaluated on SconeEval benchmark. “COM”: composition. “DIS”: distinction.

Evaluation settings We evaluate on OmniContext [39] and SconeEval. Images are sampled at 1024 × 1024 with each method’s default settings. To reduce randomness, we conduct 3 sampling rounds and score each round 3 times, yielding 9 result groups whose average is reported.

#### 5.2. Quantitative and qualitative evaluation

Quantitative evaluation On OmniContext, as shown in Tab. 2, our Scone achieves the highest average score among open-source methods, showing strong composition capability. Closed-source models GPT-4o [23] and Gemini2.5-Flash-Image [9] achieve the top two average scores, demonstrating leading performance. On SconeEval, as shown in Tab. 3, Scone achieves the best composition, distinction and overall scores among open-source models, showing strong composition and distinction performance. Unified models with lower composition scores, such as OmniGen2 [39], still outperform generation models like QwenImage-Edit-2509 [38] in distinction, highlighting the benefit of understanding for subject distinction. GPT-4o [23] and Gemini-2.5-Flash-Image [9] exhibit strong composition and distinction abilities, securing the top two overall scores, consistent with results on OmniContext. Moreover, generation in complex contexts remains difficult due to semantic/visual interference and unstable subject preservation. In Fig. 8, Scone achieves the lowest score standard deviation on SconeEval, indicating the best stability.

Threshold τ COM ↑ DIS ↑ Overall ↑

w/o bridge 8.15 8.70 8.43 0.82 8.18 8.73 8.46 0.85 8.19 8.75 8.47 0.88 8.21 8.79 8.50

omission. All results are sampled with the same seed.

#### 5.3. User study

Qualitative evaluation Results on OmniContext in Fig. 6 show that Scone generates natural compositions with strong subject consistency. Results on SconeEval in Fig. 7 further demonstrate that Scone can compose four subjects and distinguish the target subject among multiple candidates, reducing issues such as subject redundancy, blending, and

We validate the alignment of GPT-4.1 scores with human evaluation. Thirty evaluators, both professionals and nonprofessionals, assess cases from SconeEval. Each evaluator reviews 60 test samples, with 10 samples from each subtask, and compares the outputs of OmniGen2 [39], UniWorld-V2 [16] and our Scone side by side. Evaluators

Gemini-2.5 -Flash-Image

Qwen-lmage -Edit-2509

Reference image Instruction GPT-4o

Scone (Ours) OmniGen2 BAGEL

###### Echo-4o

| |
|---|

: Target subject

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

The man in Image 1 standing beside the woman in Image 2, both smiling as they watch the monkey in Image 3 swing playfully from a tree branch, while the dalmatian dog in Image 4 dashes joyfully.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

The food in the reference image is resting on a rustic wooden table.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

The woman on the far left of the image sitting by the riverbank, fishing with a smile, surrounded by lush greenery and calm water.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

The rose in img 1 is entwined with the pink flowers in img 2, blooming together in a sundrenched garden, their petals glowing softly as a breeze carries their scent across a stone archway.

| |
|---|

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

The man wearing a watch in Figure 1 is hanging the quilt in Figure 2.

| |
|---|

###### Figure 7. Qualitative comparison of existing models on SconeEval benchmark.

Effects of understanding bridge strategy in stage II We compare three versions, (a) direct fine-tuning of both the understanding and generation experts, and (b)(c) first finetuning the understanding expert and then both experts, differing in bridge usage. All are trained for 2k steps, with 1k per stage in the two-step setting. As shown in Tab. 5, the two-step strategy outperforms direct fine-tuning, and the bridge further improves subject distinction and overall robustness. Qualitative results in Fig. 9, spanning cross- and intra-category cases from the Distinction and Composition & Distinction tasks, further highlight the bridge’s benefit for more robust distinction. An additional user study, following Sec. 5.3, compares Echo-4o, Scone w/o bridge, and Scone w/ bridge, yielding scores of 0.27, 0.31, and 0.42, respectively, again showing the bridge version performs best.

OmniGen2 Echo-4o BAGEL Gemini-2.5-Flash-Image GPT-4o Scone (Ours)

0.13

STD()

| |
|---|

0.09

| |
|---|

0.05

| |
|---|

| |
|---|

0.01

| |
|---|

COM DIS Overall

- Figure 8. Stability measured by score standard deviation.

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

error redundancy redundancy blending

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

|[Figure 254]|
|---|

| |
|---|

: Target

Reference

w/ bridge (ours)

[Figure 255]

[Figure 256]

w/o bridge

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

- Figure 9. Qualitative comparison of bridge on SconeEval.

Parameter study of threshold in stage II As shown in Tab. 6, reducing irrelevant token interference steadily improves performance, showing robust semantic guidance.

select the best result based on instruction following, subject consistency, realism, and aesthetics. After normalization, the scores are: OmniGen2 0.27, UniWorld-V2 0.27, and Scone 0.46, confirming both the reasonableness of GPT-4.1 scores and the effectiveness of our method.

### 6. Conclusion

We introduce Scone, a unified understanding-generation framework that addresses a neglected problem in subjectdriven generation: distinguishing target subjects in multicandidate contexts. By using the understanding expert as a semantic bridge, it aligns semantics early and filters irrelevant content, guiding the generation expert toward accurate subject preservation and robust composition. Together with SconeEval, it offers a comprehensive benchmark for evaluating and improving both composition and distinction.

#### 5.4. Ablation study and parameter study

Effects of refined data in stage I As shown in Tab. 4, stage I significantly improves composition performance over BAGEL. The 70K base set brings major gains, and the refined 22K set further boosts both prompt following and subject consistency, highlighting that role of data quality.

### Acknowledgments

This work is supported by National Natural Science Foundation of China (92470121, 62402016), Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM113), National Key R&D Program of China (2024YFA1014003), Zhongguancun Academy (C20250204, C20250602), Beijing Major Science and Technology Project (Z251100008125043, Z251100008425023), and Highperformance Computing Platform of Peking University.

### References

- [1] Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, et al. Mc-llava: Multi-concept personalized vision-language model. arXiv preprint arXiv:2411.11706, 2024. 2
- [2] Ruichuan An, Sihan Yang, Renrui Zhang, Zijun Shen, Ming Lu, Gaole Dai, Hao Liang, Ziyu Guo, Shilin Yan, Yulin Luo, et al. Unictokens: Boosting personalized understanding and generation via unified concept tokens. arXiv preprint arXiv:2505.14671, 2025. 3
- [3] Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse: Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025. 5
- [4] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 3

- [5] Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, et al. Opengpt-4o-image: A comprehensive dataset for advanced image generation and editing. arXiv preprint arXiv:2509.24900, 2025.
- [6] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 1, 3, 6, 7, 12, 13
- [7] Abdelrahman Eldesokey, Aleksandar Cvejic, Bernard Ghanem, and Peter Wonka. Mind-the-glitch: Visual correspondence for detecting inconsistencies in subject-driven generation. arXiv preprint arXiv:2509.21989, 2025. 5
- [8] Shiran Ge, Chenyi Huang, Yuang Ai, Qihang Fan, Huaibo Huang, and Ran He. Expand and prune: Maximizing trajectory diversity for effective grpo in generative models. arXiv preprint arXiv:2512.15347, 2025. 1
- [9] Google. Introducing gemini 2.5 flash image, our state-ofthe-art image model, 2025. 1, 6, 7
- [10] Zinan Guo, Pengze Zhang, Yanze Wu, Chong Mou, Songtao Zhao, and Qian He. Musar: Exploring multi-subject customization from single-subject dataset via attention routing. arXiv preprint arXiv:2505.02823, 2025. 5
- [11] Chanran Kim, Jeongin Lee, Shichang Joung, Bongmo Kim, and Yeul-Min Baek. Instantfamily: Masked attention

- for zero-shot multi-id image generation. arXiv preprint arXiv:2404.19427, 2024. 3
- [12] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. Viescore: Towards explainable metrics for conditional image synthesis evaluation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12268–12290, 2024. 5
- [13] Black Forest Labs. Flux, 2024. 4, 6
- [14] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 3, 6

- [15] Jie Lei, Tamara Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 487–507,

2023. 2

- [16] Zongjian Li, Zheyuan Liu, Qihui Zhang, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Yang Ye, Wangbo Yu, Yuwei Niu, and Li Yuan. Uniworld-v2: Reinforce image editing with diffusion negative-aware finetuning and mllm implicit feedback. arXiv preprint arXiv:2510.16888, 2025. 3, 6, 7
- [17] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 3
- [18] Weifeng Lin, Xinyu Wei, Ruichuan An, Tianhe Ren, Tingwei Chen, Renrui Zhang, Ziyu Guo, Wentao Zhang, Lei Zhang, and Hongsheng Li. Perceive anything: Recognize, explain, caption, and segment anything in images and videos. arXiv preprint arXiv:2506.05302, 2025. 2
- [19] Wenxuan Liu, Yao Deng, Kang Chen, Xian Zhong, Zhaofei Yu, and Tiejun Huang. Sota: spike-navigated optimal transport saliency region detection in composite-bias videos. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, 2025. 2
- [20] Wenxuan Liu, Xian Zhong, Yihan Dai, Xuemei Jia, Zheng Wang, and Shin’Ichi Satoh. Motion-consistent representation learning for uav-based action recognition. IEEE Transactions on Intelligent Transportation Systems, 2025. 1
- [21] OpenAI. Hello gpt-4o, 2025. 6, 12
- [22] OpenAI. Introducing gpt-4.1 in the api, 2025. 5
- [23] OpenAI. Introducing 4o image generation, 2025. 6, 7
- [24] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 4
- [25] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 3

- [26] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. arXiv preprint arXiv:2406.16855, 2024. 4, 5
- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 4
- [28] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500– 22510, 2023. 5
- [29] Wei Song, Yuran Wang, Zijia Song, Yadong Li, Haoze Sun, Weipeng Chen, Zenan Zhou, Jianhua Xu, Jiaqi Wang, and Kaicheng Yu. Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324, 2025. 3
- [30] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Alphaclip: A clip model focusing on wherever you want. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13019–13029, 2024. 5
- [31] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14940–14950, 2025. 3
- [32] Bingda Tang, Boyang Zheng, Sayak Paul, and Saining Xie. Exploring the deep fusion of large language models and diffusion transformers for text-to-image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28586–28595, 2025. 3
- [33] Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, et al. Video understanding with large language models: A survey. IEEE Transactions on Circuits and Systems for Video Technology, 2025. 2
- [34] Jing Tao, You Li, Banglei Guan, Yang Shang, and Qifeng Yu. Simultaneous enhancement and noise suppression under complex illumination conditions. IEEE Transactions on Instrumentation and Measurement, 73:1–11, 2024. 1
- [35] Qwen Team. Qwen3 technical report, 2025. 4, 5, 6, 12, 13
- [36] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, Anthony Chen, Huaxia Li, Xu Tang, and Yao Hu. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 3
- [37] Yuran Wang, Zhijing Wan, Yansheng Qiu, and Zheng Wang. Devil is in details: Locality-aware 3d abdominal ct volume generation for self-supervised organ segmentation. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10640–10648, 2024. 1
- [38] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei

- Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 1, 5, 6, 7, 12
- [39] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 1, 3, 4, 5, 6, 7, 13
- [40] Shaojin Wu, Mengqi Huang, Yufeng Cheng, Wenxu Wu, Jiahe Tian, Yiming Luo, Fei Ding, and Qian He. Uso: Unified style and subject-driven generation via disentangled and reward learning. arXiv preprint arXiv:2508.18966, 2025. 6
- [41] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025. 1, 5, 6
- [42] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025. 1, 3, 5
- [43] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 3
- [44] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025. 3
- [45] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3

- [46] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, and Weijia Li. Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation, 2025. 1, 3, 6
- [47] Bohan Zeng, Shanglin Li, Yutang Feng, Ling Yang, Juan Zhang, Hong Li, Jiaming Liu, Conghui He, Wentao Zhang, Jianzhuang Liu, et al. Ipdreamer: Appearance-controllable 3d object generation with complex image prompts. In The Thirteenth International Conference on Learning Representations, 2024. 3
- [48] Huaxin Zhang, Xiaohao Xu, Xiang Wang, Jialong Zuo, Chuchu Han, Xiaonan Huang, Changxin Gao, Yuehuan Wang, and Nong Sang. Holmes-vad: Towards unbiased and explainable video anomaly detection via multi-modal llm. arXiv preprint arXiv:2406.12235, 2024. 2
- [49] Tao Zhang, Chenglin Zhu, Yanjun Shen, Wenjing Luo, Yan Zhang, Hao Liang, Fan Yang, Mingan Lin, Yujing Qiao, Weipeng Chen, et al. Cfbench: A comprehensive constraintsfollowing benchmark for llms. In Proceedings of the 63rd

- Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32926–32944, 2025. 2
- [50] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8069–8078, 2024. 3
- [51] Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. Cross-modal information flow in multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19781–19791, 2025. 2, 3, 12

[Figure 273]

## Scone: Bridging Composition and Distinction in Subject-Driven Image

## Generation via Unified Understanding-Generation Modeling Supplementary Material

### A. Additional details of motivation

[Figure 274]

###### Prompt for distinction scoring

The similarity visualizations of instruction and image tokens in Figs.1(b) and 2(a) of the main paper are based on our base model, BAGEL [6]. To better highlight high-similarity regions, we retain the top 50% and generate masked images. We group layers into four sets (0-7, 8-15, 16-23, 24-27) based on the layer function analysis from [51]. In Fig. 11, we further show representative similarity and masked images for each group.

### Given

- - A **subject description**.
- - The **first image** is the reference image.
- - The **second image** is the target image.

### Task Determine whether the described subject from the reference image **appears** in the target image.

- 1. **Identify** the subject in the reference image based on the given description.
- 2. **Judge presence** in the target image:

- - Focus strictly on **presence**, not on appearance similarity or instruction compliance.
- - Assign **1** if the subject is clearly identifiable in the target image.
- - Assign **0** if the subject is not identifiable. … ### IMPORTANT: Your response must be either 0 or 1. …

- Observation 1 (Comparison between understanding and generation experts) The understanding expert captures more distinct semantic information from the image than the generation expert, as its image-token hidden states exhibit higher similarity to the instruction-token hidden states. It attends more strongly to instruction-relevant regions, such as candidate subjects.
- Observation 2 (Comparison across layers of the understanding expert) Although similarity remains high at layers 16 and 24, region discrimination is strongest at layer 8, which provides more distinctive semantic cues for generation guidance. We therefore choose layer 8 to provide the semantic mask, and apply it to the later semantically discriminative layers 9–15.

Figure 10. Prompt for distinction scoring in SconeEval. It determines whether the described reference subject appears in the target image.

versing the reference and target images, so that the original reference becomes the target and the original target becomes the reference. This avoids the cost of generating new images. For instruction construction, Qwen3-30B-A3BInstruct-2507 [35] identifies subjects, provides distinctive descriptions, and generates instructions based on the prompt

- in Fig. 16(a). The final dataset contains 2 case types, each with cross-category and intra-category candidate subjects. Examples are shown in Fig. 16(b).

Multi-subject data Multi-subject data are constructed from single-candidate multi-subject data by editing a subset of the reference images. Specifically, we use GPT-4o [21] to generate prompts for subjects from different categories, and then add at least one subject to the reference images using Qwen-Image-Edit-2509 [38]. The instruction construction consists of two steps: (1) subject identification and (2) subject replacement. Subject identification follows the procedure described in Sec.4.2 of the main paper, with additional details provided in Sec. C. Subject replacement uses Qwen3-30B-A3B-Instruct-2507 [35] and the prompt

- in Fig. 17(a) to replace the original subject description with the distinct description obtained in Step 1. The final dataset contains 5 case types, each including both cross-category and intra-category candidate subjects in the reference images. Examples are shown in Fig. 17(b).

### B. Additional details of training data

#### B.1. Synthesized data for data pool

As described in Sec.5.1 of the main paper, we synthesize 15K samples with 3-4 input images to fill gaps in the data pool and improve the composition capability of Scone. Examples are shown in Figs. 13 and 14.

#### B.2. Data filtering for refined single-candidate data

As described in Sec.5.1 of the main paper, refined singlecandidate samples are filtered by scoring subject consistency and instruction following with the VLM model Qwen3-VL-30B-A3B-Instruct. Key prompt contents are shown in Fig. 15(a), with emphasis on facial identity, text, and quantity. Each sample is scored from 0 to 4, and only those with a score of 4 are retained, as shown in Fig. 15(b).

#### B.3. Details of multi-candidate data

Single-subject data Multi-candidate single-subject data are derived from single-candidate multi-subject data by re-

Layer 0 Layer 8 Layer 16 Layer 24

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

Understanding expert

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Generation expert

Figure 11. Representative similarity and masked images for each layer group. Similarity visualizations of instruction and image token hidden states from understanding and generation experts are based on our base model, BAGEL [6]. Masked images retain the top 50% of regions for clearer observation.

### C. Two-step decoupling instruction construction in SconeEval

As described in Sec.4.2 of the main paper, we adopt a twostep decoupling strategy that separates visual understanding from instruction generation, improving instruction stability and quality. As shown in Fig. 18, direct instruction construction often produces unusable results, such as incorrect image indices, ambiguous target subjects, and unrelated subjects. Our strategy first uses the vision-language model Qwen3-VL-30B-A3B-Instruct [35] to identify the target subject and generate a distinct description from the raw single-candidate and edited multi-candidate reference images, with prompt in Fig. 19(a). It then uses the language model Qwen3-30B-A3B-Instruct-2507 [35] to generate instructions solely from the subject descriptions, with prompt in Fig. 19(b).

### D. Limitation and future work

Our Scone still shares a common limitation with existing methods: unrealistic interactions. As shown in Fig. 12, the generated dog passes through the chair, violating physical laws. This issue appears in both our Scone and OmniGen2 [39]. Future work will also explore more efficient mechanisms to reduce redundant image tokens for scalable generation in complex scenarios.

Reference image Instruction

[Figure 291]

The small dog in the center of Image 1 is sitting calmly on

[Figure 292]

the wooden armchair, its paws resting on the dark gray fabric, with a striped pillow beside it.

Scone (Ours) GPT-4o

OmniGen2

[Figure 293]

[Figure 294]

[Figure 295]

###### Figure 12. Limitation of our Scone.

[Figure 296]

###### Instruction

###### Reference image Target image

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Combine in a picture

Character Character Object

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Place the researcher from Image 1, who is with short hair under a hair net, wearing the royal blue corduroy yoga pants from Image 2. Adorn the researcher with the large hoop earrings from Image 3 Character Object Object

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Position the Coach from photo 1 on the left, and the Lead scientist from photo 2 on the right. In the background, display the study wall map bristling with thumbtacks and notes from photo 3

Character Character Scene

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

Place the Slow Cooker from the first image in the foreground, with the Brooch from the second image elegantly placed near it. In the background, depict the scene of the fog-laced morning with a pagoda tower visible across a riverbend, as shown in the third image Object Object Scene

- Figure 13. Examples of synthesized data with 3 input images. These cover 4 case types, including character-character/object interactions, characters with multiple objects, characters in scenes, and objects placed in scenes.

[Figure 313]

[Figure 314]

[Figure 315]

Reference image Target image

Place the Adventurer in

- <|image_1|>, the Judge in
- <|image_2|>, the Friend

mediator in <|image_3|>, and the Bride in <|image_4|>

Instruction

Character Character

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Character Character

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Character Character

Character Object

Place the cutlery set from the fourth image on a dining table. The man from the first image is standing in the foreground. Nearby, the rider from the second image is standing. In the background, the man from the third image is seated using a computer.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Depict the traveler in <|image_1|>, alongside the security person in <|image_2|>, in a T-shirt and shorts, with dark skin. Place the mint oven with a non-stick interior from

- <|image_3|> in the background. Include the white cat from
- <|image_4|> sitting nearby.

Character Character

Object Object

A judge with a side-part hairstyle stands dignified, the court badge visible on their robe. Nearby, a gold rice cooker sits. In the scene, an elder butterfly in the midst of molting displays its pale, shedding shell. A playful orange kitten frolics around.

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Character

Object Object

Object

[Figure 336]

Combine these items in a picture.

Combine in a picture.

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Object Object

Object Object

Character Character

Character Scene

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

Character Character

Object Scene

The ambiance is softly lit by the gentle glow of fairy lights strung above a convenience truck. Nearby, a bystander stands casually. A player with curly hair and a windbreaker. A baby albino ferret with its distinctive white fur and red eyes peeks.

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Depict a teacher wearing a blouse and skirt (Image 1)

standing beside a cat (Image 2). Surround them with leaves that are dark green and vegetative (Image 3). In the background, an ancient city gate is glowing at sunset (Image 4).

Place the Jade Carving from

- Image 1 on a wooden table next to the Tennis Racket from
- Image 2. Position the Terracotta Saucer from Image 3 nearby, with the Violin and bow from

Image 4 resting on a windowsill in the background.

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

Character Object

Object Scene

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

Object Object

Object Scene

[Figure 360]

Reference image Instruction Target image Reference image Instruction Target image

- Figure 14. Examples of synthesized data with 4 input images. These cover 9 case types, including combinations of characters, objects, and scenes, as well as their interactions and mixed compositions.

[Figure 361]

[Figure 362]

###### Prompt for training data filtering

###### Reference image Target image

###### Instruction

[Figure 363]

[Figure 364]

[Figure 365]

You are an AI expert evaluating how well the subjects in the target image match the identities of those in the reference images, according to the given instruction. … ### Scoring Criteria

A man gripping a gleaming sword in one hand while the other rests on a Nintendo DS console tucked under his arm.

- - 0: Completely inconsistent or subject missing.
- - 1: Severely inconsistent; only minor similarities.
- - 2: Some similarities but many mismatches (moderate).
- - 3: Mostly consistent; only small differences (e.g., facial or text inconsistency).
- - 4: Perfect match with all references. ### Deduction Rules Deduct the score in any of the following cases:
- - Characters: The facial identity is inconsistent or unclear; eyes, nose, mouth, cheekbones, chin, wrinkles, makeup, hairstyle, hair color, or face/head structure do not match. If the instruction does not specify changes, clothing or hairstyle differs from the references. Body shape, skin tone, or other major physical traits are inconsistent.
- - Objects: The shape, pattern, or material is inconsistent with the references.
- - Text: The text differs from the references in content or appearance.
- - Quantities: The number of elements does not match (e.g., “three blue lights” vs. “one light”).
- - Physical integrity: Any subject in the image shows unnatural deformities, such as broken limbs, missing body parts, or distorted anatomy.
- - Instruction consistency: The target image fails to follow the instruction semantically. For example, misinterprets relationships, roles, or actions, adds or omits key elements, or does not represent the intended meaning of the instruction. …

Nintendo DS sword

###### Score: 0

(completely inconsistent)

[Figure 366]

[Figure 367]

[Figure 368]

A young woman stands confidently with a skateboard tucked under her arm. Large headphones rest around her neck. Score: 2

[Figure 369]

headphones skateboard

(moderate consistent)

[Figure 370]

[Figure 371]

Place the person from image_1 standing at a café counter, talking with the person from image_2, who is seated at a nearby

person person

###### table. Score: 4

(perfect match)

###### (a) Prompt for training data filtering (b) Results of training data filtering

- Figure 15. Data filtering for refined single-candidate data. (a) Prompt for filtering. Key prompt components are shown. (b) Filtering results. Samples are scored from 0 to 4, and only those with a score of 4 are retained.

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Reference image Target image

The woman with long black hair is now facing forward and looking directly at the camera, with a neutral expression

Instruction

The woman in black floral dress is now

standing against a white background, with hands in her pockets, looking directly at the camera

The man's wristwatch is now a green malachite stone bracelet with brown leather and beads, set against a plain white background

The red fox figurine in the front is now isolated on a plain white background

Character (cross)

[Figure 378]

Character (intra)

Object (cross)

Object (intra)

[Figure 379]

[Figure 380]

[Figure 381]

Prompt for instruction construction

### Task You are given two images:

- - The **first image** is the reference image.
- - The **second image** is the target image from a subjectdriven image generation task. Your goal is to produce an instruction that can accurately generate the **target image**.

- **Step 1:** Identify which subject in the reference image corresponds to the target subject in the target image. The target subject is the **most prominent subject** in the target image.
- **Step 2:** Describe the corresponding subject in the reference image with an **unambiguous, concise phrase** that uniquely distinguishes it from others **within the reference image**. Use 1–3 cues if necessary:

- - **Attribute:** age, color, clothing, hairstyle, facial feature, expression, posture, or species (e.g., “the woman with curly hair,” “the golden retriever dog”)
- - **Position:** absolute or relative position within the image (e.g., “on the far left,” “second from the right)
- - **Size:** relative scale within the image (e.g., “the tallest,” “the smallest”)

- **Step 3:** Write a short, fluent **subject-driven generation instruction** describing how the identified subject

**changes** from the reference image to match the target image. …

(a) Prompt for instruction construction (b) Example demonstration

- Figure 16. Multi-candidate single-subject data construction. (a) Prompt for instruction construction. The prompt instructs the vision-language model to identify subjects, provide distinct descriptions, and generate instructions. (b) Example demonstration. It includes 2 case types, Character and Object, each with cross-category and intra-category candidate subjects in the reference images.

[Figure 382]

[Figure 383]

###### Prompt for subject replacement

###### Reference image Target image

###### Instruction

[Figure 384]

[Figure 385]

[Figure 386]

The woman in the red dress from image_1 is walking down a red carpet. Beside her, the man with green hair and clown makeup from image_2 is standing with a playful grin, dressed in a vibrant suit. The woman looks over her shoulder with a smile. The man gestures towards the front.

You are an AI assistant skilled at revising subject-driven image generation instructions.

### Task Given an original instruction and a list of subject referring descriptions (each corresponding to an input image index) , replace the subject references in the instruction with the provided referring descriptions. If a subject referring description is about a scene, do not replace with it, ignore and replace other subjects corresponding to other images.

[Figure 387]

Character (intra)

Character

[Figure 388]

[Figure 389]

Position the woman in the white t-shirt and blue jeans from image_1 seated on the sofa set on the right from image_2. The person’s left arm should be casually draped over the armrest, and the right hand can be resting on the lap.

Clearly indicate which image each specified subject comes from.

Character (cross)

Object

[Figure 390]

[Figure 391]

[Figure 392]

Keep all other parts of the instruction unchanged, and ensure the revised instruction remains fluent and coherent.

An undivided, seamless, and harmonious picture with two objects in the snow, the yellow boat on the far left with large windows and the green collapsible bucket are placed together.

### Example 1 Input: Subject referring descriptions: the woman in a pink suit standing on the right side of the image; the person in the denim jacket standing in the jungle Instruction: "Position the person from image_1, dressed in a stylish pink suit and holding a cane, standing confidently on a stage. Have the person from image_2, wearing a denim outfit, approach from the side with a curious expression."

Object (intra)

Object

[Figure 393]

[Figure 394]

[Figure 395]

Position the woman in the white top in the center of image_1 standing near a colorful bookshelf in the scene from image_2, with her left hand gently touching one of the whimsical light fixture's crescent moon tips.

[Figure 396]

[Figure 397]

Character (intra)

Scene

[Figure 398]

Output: "Position the woman in a pink suit standing on the right side of image_1, dressed in a stylish pink suit

Place the container from image_1 on the small table at the end of the hallway in image_2. Ensure it is centered on the table and catches the light streaming through the blinds, creating a warm and inviting

and holding a cane, standing confidently on a stage. Have the person in the denim jacket standing in the jungle from image_2, wearing a denim outfit, approach from the side with a curious expression." …

atmosphere. Object (Cross) Scene

###### (a) Prompt for subject replacement (b) Example demonstration

- Figure 17. Multi-candidate multi-subject data construction. (a) Prompts for subject replacement. The prompts guide the language model to replace the original subject description with a new distinct description for the edited multi-candidate reference images. (b) Example demonstration. It includes 5 case types: Character+Character, Character+Object, Object+Object, Character+Scene, and Object+Scene, each with cross-category and intra-category candidate subjects in the reference images.

[Figure 399]

Subject description

Two-step decoupling instruction construction Subject identification image-to-text Instruction generation

the small dog text-to-text

[Figure 400]

[Figure 401]

in the center

Instruction

The small dog in the center from Image 1 is

sitting calmly on the wooden armchair…

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Direct instruction construction

[Figure 406]

[Figure 407]

Raw image1

Edited image1

[Figure 408]

[Figure 409]

Raw image2

[Figure 410]

Subject description

…a wooden Scene armchair…

Character /Object

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

Raw image1

Edited image1

[Figure 415]

[Figure 416]

Raw image2

[Figure 417]

[Figure 418]

: VLM

Instruction

The small dog in the center from img 1 should be running on the grassy path in the scene of img 3, positioned between the other two dogs.

- Figure 18. Comparison of two instruction construction strategies. The two-step decoupling strategy separates image-to-text and textto-text generation, reducing cross-image interference and avoiding errors in the direct strategy, such as incorrect image indices, ambiguous target subjects, and unrelated subjects.

[Figure 419]

[Figure 420]

###### Prompt for subject identification

Prompt for instruction generation

For Character/Object

For samples w/o Scene

You are an AI assistant skilled in composing subject-driven image generation instructions.

You are an AI assistant skilled at identifying a target subject in an image based on a reference image.

### Task Given a list of subject referring descriptions (each corresponding to an input image index), write a single, coherent instruction that places all subjects in a new state or scene.

### Task You are given two images.

- - The first image (reference image) contains the target subject.
- - The second image (image to identify) contains multiple subjects.

Your goal is to **identify and describe only the subject in the second image** that corresponds to the reference. Provide an **unambiguous, concise phrase** that uniquely distinguishes the target subject **within the second image**.

### Use cues such as:

- - **attribute**: …
- - **position**: …
- - **size**: … …

- - If there is only one description, describe that subject in a new state or scene.
- - If there are multiple descriptions, describe how the subjects appear or interact together naturally in a new scene. …

You are an AI assistant skilled in composing subject-driven image generation instructions.

### Task Given a list of descriptions (each corresponding to an input image index), write a single, coherent instruction that places all subjects in the specified scene.

- - The former description(s) refer to subjects.
- - The last description refers to the scene, showing the background or environment where the subjects should appear.

For samples w/ Scene

For Scene

### Task Given a scene image, describe the scene clearly and concisely.

- 1. Provide a one-sentence summary of the overall scene, including whether it's indoors or outdoors and the general setting (e.g., city street, park, living room).
- 2. List the main and most noticeable objects, such as furniture, people, animals. …

Construct a natural, realistic instruction describing how the subjects appear or interact within the scene (e.g., the bird is standing on the table from the scene). …

(a) Prompt for subject identification

(b) Prompt for instruction generation

- Figure 19. Prompts for instruction construction in SconeEval. (a) Prompt for subject identification. For Character or Object images, provide a concise subject description; for Scene images, describe the setting and key objects. (b) Prompt for instruction generation. Generate instructions from the subject descriptions, emphasizing subject-subject and subject-scene interactions.

