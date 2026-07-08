# arXiv:2602.20951v2[cs.CV]26Mar2026

## See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis

Jaehyun Park1,∗ Minyoung Ahn2,3,∗ Minkyu Kim3 Jonghyun Lee3 Jae-Gil Lee1 Dongmin Park3,† 1KAIST 2Seoul National University 3KRAFTON {jhpark813,jaegil}@kaist.ac.kr,{minkyu.kim,jonghyunlee,dongmin.park}@krafton.com,michellahn02@snu.ac.kr

#### Abstract

Despite recent advances in diffusion models, AI generated images still often contain visual artifacts that compromise realism. Although more thorough pre-training and bigger models might reduce artifacts, there is no assurance that they can be completely eliminated, which makes artifact mitigation a highly crucial area of study. Previous artifactaware methodologies depend on human-labeled artifact datasets, which are costly and difficult to scale, underscoring the need for an automated approach to reliably acquire artifact-annotated datasets. In this paper, we propose ArtiAgent, which efficiently creates pairs of real and artifactinjected images. It comprises three agents: a perception agent that recognizes and grounds entities and subentities from real images, a synthesis agent that introduces artifacts via artifact injection tools through novel patch-wise embedding manipulation within a diffusion transformer, and a curation agent that filters the synthesized artifacts and generates both local and global explanations for each instance. Using ArtiAgent, we synthesize 100K images with rich artifact annotations and demonstrate both efficacy and versatility across diverse applications. Code is available at link.

#### 1. Introduction

Recent advances in diffusion models have led to remarkable success in generating highly photorealistic images [45, 53]. While these models have demonstrated striking achievements in text-to-image alignment and aesthetic quality, they still suffer from producing visual artifacts, unintended distortions or anomalies in generated outputs. For example, in Figure 1(a), even state-of-the-art models such as NanoBanana [13] produce artifacts, e.g., six-fingered hands and fused entities, reducing user satisfaction, e.g., uncanny valley. Furthermore, mitigating such artifacts is especially critical in high-stakes diffusion applications where reliability is paramount, including image generation in medicine [23], robotics [17], autonomous driving [19], and so on.

* Equal contribution; work done during an internship at KRAFTON. † Corresponding author.

Meanwhile, in image understanding, vision-language models (VLMs) have made substantial progress, showing strong capabilities in visual question answering and scene description [54]. These advances have allowed VLMs to serve as automatic systems for many vision tasks, including medical analysis [16] and robotics control [24]. However, we find that VLMs face challenges when confronted with visual artifacts. As shown in Figure 1(a), even state-of-theart VLMs, such as GPT-5 [34] and Gemini-2.5-pro [8], exhibit limited ability to detect, localize, or explain artifacts in AI-generated images, nearly indistinguishable from random guessing (see § 6 for details). Consequently, VLMs cannot yet serve as reliable systems for artifact comprehension, which constrains their utility.

Recently, several approaches have been proposed to tackle visual artifacts in VLMs and diffusions[20, 48, 55]. However, existing efforts exhibit two major limitations: (1) they mainly target simple artifact types (e.g., Gaussian noise or blur), which were prevalent in early diffusion models such as SD1.0 [43], but are rarely observed in modern models, where more plausible physical distortions are predominant; and (2) they rely heavily on human-annotated artifacts (e.g., as many as 10K labels), which is costly and fundamentally limited in scalability to capture the full diversity of diffusion-generated artifacts. These limitations highlight the need for scalable, annotator-free methods to address plausible and extensive artifacts across diverse visual contexts produced by modern diffusion models.

To fill this gap, we introduce ArtiAgent (Figure 1(b)), a novel agentic framework that injects plausible artifacts without human intervention via inversion-restoration [9] by perturbing the attention of DiT (see § 4.2.2). ArtiAgent consists of three agents: (1) a perception agent, which identifies the most suitable objects or entities in a real image to perturb, (2) a synthesis agent, which selects and applies the tools to generate plausible artifacts, and (3) a curation agent, which filters low-quality results and refines their annotations. Through this agentic pipeline, ArtiAgent produces high-quality artifact-injected images with rich annotations, including binary labels, locations, and explanations suitable for detection, localization, and reasoning.

[Figure 1]

(a) Limitations of SOTA Diffusions & VLMs (b) Agentic Artifact-Aware Data Synthesis

[Figure 2]

Inversion-Injection Is there anything wrong with the image?

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Real Synthesized

Duplication

Distortion

The nose of the bear is distorted.

Distort

|[Figure 7]<br><br>Diffusion|
|---|

[Figure 8]

SD3.5, FLUX, Qwen-Image, Nano-Banana

There is an extra paw of the bear.

[Figure 9]

Add

[Figure 10]

[Figure 11]

FLUX.1-dev Qwen-Image

[Figure 12]

Omission Fusion

[Figure 13]

[Figure 14]

There is nothing wrong with the image.

ArtiAgent

The child’s hand is missing.

Remove

[Figure 15]

[Figure 16]

GPT-5, GPT-4o Gemini-2.5-Pro

###### VLM

Fuse

The teddy bear and the child are merged.

[Figure 17]

SD3.5 Nano-Banana

(c) VLM Artifact Comprehension

(d) Reward-Guided Generation

(e) VLM-Guided Image Correction

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

| |
|---|

[Figure 23]

[Figure 24]

Is there anything wrong with the image?

Detect Verify

[Figure 25]

Inpaint

[Figure 26]

[Figure 27]

There is a hand holding a phone with an extra VLM finger in <bbox>.

| |
|---|

[Figure 28]

Reward=0.2

Reward=0.8

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Real Synthesized

Figure 1. Overview of our challenges and approach. The red boxes indicate the regions with visual artifacts. (a) Examples of structural visual artifacts in state-of-the-art diffusion models and the inability of VLMs to recognize or explain them. (b) Overview of ArtiAgent, a novel agentic framework that synthesizes artifacts for arbitrary visual contexts without human intervention. (c) Example of VLMbased artifact comprehension via detection, explanation, and localization. (d) Application to reward-guided text-to-image generation. (e) Application to image correction, where artifact-aware VLM-guided inpainting removes the flawed regions.

For thorough validation, we present ArtiBench, a new benchmark of 1K AI-generated images produced by modern generative models such as FLUX-dev and Nano-Banana, where each image is carefully annotated by humans with binary artifact labels, their bounding box locations, and explanations. We evaluate ArtiAgent on ArtiBench as well as three existing benchmarks, showing that open-source VLMs (e.g., Qwen2.5-VL) fine-tuned on 100K training samples generated by ArtiAgent consistently outperform proprietary VLMs (e.g., GPT-5) and prior baselines across major artifact-perception tasks, including detection, localization, and reasoning. Furthermore, with these artifactaware VLM, we benefit two downstream applications: (1) guiding diffusion sampling toward artifact-free generation with VLM-based artifact reward, and (2) automatically editing diffusion outputs that contain artifacts, substantially improving the image generation pipelines.

challenging benchmark of 1K images generated by modern generative models with human labels.

• Experiments. Extensive experiments demonstrate the superiority of ArtiAgent by scaling up VLM performance on core artifact-perception tasks. Moreover, we show its utility in downstream diffusion-based applications: artifact-free image generation and editing.

#### 2. Related Work

Visual Artifact Datasets. Several training datasets have been introduced to supervise the understanding of visual artifacts in generative models. PAL[55] provides 10K images with pixel-level annotations of perceptual defects for segmentation-based training. SynthScars[20] provides 12K images with pixel-level masks and textual explanations. DiffDoctor[48] scales human labeling through semisupervised expansion from a 25K seed set. Although these datasets offer essential supervision for artifact detection and correction, their reliance on human annotation makes them costly and difficult to scale.

Our key contributions are summarized as follows:

- • Framework. We introduce ArtiAgent, an agentic data synthesis framework that produces diverse plausible artifacts at scale, enriched with high-quality annotations.
- • Tools. We develop agentic tools that inject artifacts via our novel inversion-injection method during image reconstruction, which can be used by any DiT model.
- • Datasets. We synthesize a large-scale training set with 100K examples via ArtiAgent, along with ArtiBench1, a

Moreover, several benchmark datasets have been released to evaluate models’ ability to understand visual artifacts. RichHF-18K[27] provides annotated artifact regions, while LOKI[51] adds natural-language explanations with bounding box labels. SynthScars also releases evaluation splits to test perception, localization, and explanation ca-

1https://huggingface.co/datasets/KRAFTON/ArtiBench

pabilities. These benchmarks serve as reference points for measuring a model’s artifact awareness; however, most of them rely on artifacts from earlier diffusion models and tend to focus on degenerate artifacts (e.g., Gaussian noise), limiting their relevance to the richer and more diverse failure modes of modern generative systems.

Table 1. Artifact frequency of modern diffusion models.

20%

37%

Generative Model Freq.

34%

SD3.5-Large 36% FLUX-schnell 28%

9%

duplication omission

Qwen-Image 17%

Handling Visual Artifacts. Using these datasets, different modeling strategies have been explored. PAL [55] trains segmentation models to localize artifact regions and enables automated correction via inpainting. RichHF-18K [27] is used to train multimodal models that predict humanlike feedback heatmaps, which can then refine diffusion models through preference learning. LEGION [20] trains GLaMM [39] with the SynthScars dataset that allows detection, localization, and explanation into a unified model. DiffDoctor [48] employs its dataset to train an artifact segmentation model, which is used for diffusion fine-tuning to alleviate artifact generation. While these studies emphasize the value of artifact detection and reasoning, they also reveal the necessity of reducing reliance on manual annotation to enable scalable and reliable dataset construction.

distortion fusion

FLUX-dev 12% Nano-Banana 5%

Figure 2. Artifact type distribution of diffusion models.

Qwen-Image [49], and Nano-Banana [13]. For each generated image, a human annotator manually inspected the outputs and identified whether they contained any artifacts, further categorizing them into the four structural types. Based on this evaluation, Table 1 summarizes the frequency of artifacts for each model, while Figure 2 presents the relative occurrence of different artifact types. These findings underscore the importance of addressing structural artifacts in modern generative modeling, highlighting the need for scalable methods to detect, analyze, and mitigate them.

#### 4. Agentic Pipeline for Artifact Synthesis

#### 3. Understanding Visual Artifacts

We propose ArtiAgent, a fully automated agentic pipeline that synthesizes visual artifacts in clean images. As shown in Figure 3, three agents (perception, synthesis, curation) select injection candidates, inject artifacts, and curate outputs with local and global explanations. Leveraging recent LLMs and our artifact-injection tools, ArtiAgent determines the artifact types and locations for each image and produces high-quality annotations. Full agent details and visualizations are in Appendices A.

###### 3.1. Problem Scope

As generative models have matured, the types of visual artifacts that frequently appear have also changed. Early GANs [15] and U-Net–based diffusions [42] predominantly exhibited naive distortions such as Gaussian-like noise or low-level pixel corruption [55]. Modern diffusion models, equipped with DiT-based architectures and trained on higher-quality datasets, have largely overcome these naive distortions. However, they still generate plausible structural visual artifacts, which are the main focus of this work.

###### 4.1. Perception Agent (Figure 3(a))

The perception agent aims to analyze a real image and decompose it into meaningful semantic units that can serve as reliable candidates for the synthesis agent.

Definition 3.1. (Informal) Structural visual artifacts refer to defects in which the inherent physical structures of objects are distorted in the generated image. That is, while the contents specified in the prompt are represented, their form violates common-sense plausibility (e.g., a generated dog with two noses). This definition excludes text-to-image misalignments where the prompt content itself is misrepresented (e.g., generating a cat given the prompt of a dog).

###### 4.1.1. Entity-Subentity Vocabulary Generation

The agent uses out-of-the-box VLMs to decompose the input image into a hierarchical vocabulary of entities (e.g., dog) and subentities (e.g., nose). These are grouped into two semantic levels: peripheral subentities, such as fingers or legs, and intermediate subentities, such as body or face. The prompt template for this module is in Appendix A.2.1.

###### 3.2.ArtifactAnalysisinModernGenerativeModels

We have categorized structural artifacts into four representative types: duplication, omission, distortion, and fusion. Figure 1(a) provides illustrative examples of each case, showing how diffusion models can compromise reliability while still producing visually high-quality pixels. To systematically analyze these artifacts, we randomly sampled 100 captions from the MS-COCO dataset [7] and used them as text-to-image prompts across five state-of-the-art diffusion models, including Stable Diffusion 3.5 [11], FLUX [5],

###### 4.1.2. Entity-Subentity Grounding

The agent then employs Grounded-SAM [41] to ground entities and subentities visible in the image. Once the segmentation masks for both entities (e.g., dog) and subentities (e.g., leg) are obtained, the agent performs a containment analysis to associate each subentity with its parent entity (e.g., a leg of a dog) by computing the overlap ratio.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

(a) Perception Agent (b) Synthesis Agent (c) Curation Agent

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Entity-Subentity Vocab.

Filtering

| |
|---|

Compare [real] and [injected] image.

Instruction

[Figure 45]

Entity Intermediate Peripheral

Decompose the image into entities and subentities.

[Figure 46]

[Figure 47]

LPIPS

Toolbox

| |
|---|

[Figure 48]

Intermediate Entity

Peripheral

Response

[Figure 49]

add remove distort fuse

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Explanation Generation

[“person”: {“peripheral”: [“hand”, “nose”], “intermediate”: [“face”, “ear”]}]

[Figure 55]

What is wrong with [bbox] of [injected]?

[Figure 56]

“<bbox1>”: “the face … distorted” “<bbox2>”: “… duplicate hand”

[Figure 57]

patch mapping

Grounding

[Figure 58]

[Figure 59]

Explain the [injected] image.

[Figure 60]

[Figure 61]

“There is a surfer with an extra hand. Also, the face is …”

[Figure 62]

[Figure 63]

[Figure 64]

Grounding-SAM

[Figure 65]

DiT

DiT

[Figure 66]

[Figure 67]

Inversion-Injection

Real

[Figure 68]

“type”: “distortion” “entity”: “person” “subentity”: “face”

| | |
|---|---|
| | |

“<bbox1>”: “The face has been distorted” “<bbox2>”: “There is a duplicate hand” “exp”: “There is a surfer with an extra hand. Also, the face is unnaturally warped …”

“type”: “duplication” “entity”: “person” “subentity”: “hand”

“person”: [“face”, “hand1”, “hand2”, “nose”]

Injected

- Figure 3. ArtiAgent consists of three coordinated agents: (1) the perception agent detects entities and subentities using Grounded-SAM;

(2) the synthesis agent injects artifacts through patch mapping tool and the inversion-injection paradigm; and (3) the curation agent filters low-quality results and generates localized and global textual explanations.

|[Figure 69]|[Figure 70]|[Figure 71]|[Figure 72]|
|---|---|---|---|
|[Figure 73]|[Figure 74]|[Figure 75]|[Figure 76]|
|[Figure 77]|[Figure 78]|[Figure 79]|[Figure 80]|

ReferenceTargetResult

(a) Add (b) Remove (c) Distort (d) Fuse

- Figure 4. Visualization of each target-reference patch mapping and its resulting artifact-injected image.

mapping. It applies add and remove to peripheral subentities, distort to intermediate subentities, and fuse to two overlapping entities. Figure 4 shows one example per tool. Detailed algorithms are in Appendix A.3.

- • Add. Reference patches are the original subentity region, while the tool chooses the best target-patch candidate from surrounding patches. Nearby patches without overlaps or subentity class conflicts are preferred.
- • Remove. Target patches are given as the original subentity region, while the tool creates reference patch mappings to replace them with the surrounding context.
- • Distort. Target patches are the original subentity region, while the tool applies kernels to target patches to generate distorted mappings to reference patches. We use three kernels: jitter for random displacement, strip for circular shifts over bands, and a random permutation kernel.
- • Fuse. The fuse tool generates mappings that blend content across multiple entities. The target patches contain the overlapping region between two entity instances, and the chunks of reference patches from one entity are assigned to the target patches of the other entity.

###### 4.2. Synthesis Agent (Figure 3(b))

The synthesis agent uses the perception agent’s grounding to inject artifacts into an image via two components: a toolbox that generates target-reference patch mappings and an inversion-injection module that applies them to the image.

###### 4.2.1. Target-Reference Patch Mapping Toolbox

4.2.2. Inversion-Injection Method for Artifact Synthesis We propose an inversion-injection module, which extends the inversion-restoration paradigm from image editing [9,

The toolbox has four artifact-injection tools: add, remove, distort, and fuse, each producing a target-reference patch

• Background Region. For a non-target patch pb ∈ PB, we keep their original positional information and reuse the Vinv(ℓ) values to maintain the context of the original image: Q˜(pℓ)

DiT Self-Attention

DiT Self-Attention

Background 𝑝 Target 𝑝

[Figure 81]

[Figure 82]

𝑄  ,    𝐾  ,    𝑉  ,   

𝑄  ,    𝐾  ,    𝑉  ,

[Figure 83]

[Figure 84]

Value

For 𝑃 , Reference 𝑝

[Figure 85]

[Figure 86]

= RoPE(Q(pℓ)

,pb), K˜p(ℓ)

= RoPE(Kp(ℓ)

,pb), and Vp(ℓ)

[Figure 87]

Inversion

Injection

b

b

b

b

b ← Vp(ℓ)

b,inv.

𝑄 ,    𝐾 ,    𝑉 ,

𝑄  ,    𝐾  ,    𝑉  ,   

[Figure 88]

[Figure 89]

Then, self-attention is performed as in Equation (3).

Value & PE

For 𝑃 & ℳ,

In general, PE injection controls where the model believes denoising is occurring, while value injection provides what semantic content fills that position. Their combination allows for local injections of realistic artifacts, while the background remains consistent with the original image. Although value injection [9, 47] is a well-established approach in image editing research, our method introduces the novel idea of manipulating their positions during injection. Furthermore, PE injection itself has not been used in previous image editing works. Combining PE and value injections offers a particularly effective methodology for generating structural artifacts, as it enables direct manipulation of spatial information during the reconstruction process. To prevent shortcut-feature learning (e.g., edge discontinuities), we restrict PE and value injections to earlyto-middle layers and disable them during the final denoising steps. We provide attention visualizations and ablation studies (including injection-step ablations) to support these design choices in Appendices A.4 and A.5. Furthermore, our training-free inversion-injection method is inherently model-agnostic, making it readily generalizable to other emerging DiT-based image generation models.

- Figure 5. Inversion-injection module. In this example, the right arm from the reference patches is added to target patches below it.

29, 31, 47]. Specifically, the inversion-injection module employs the target-reference patch mapping to manipulate the positional information in the self-attention layers of DiT, allowing realistic structural artifact synthesis.

Notations. Let X(ℓ) ∈ RN×d denote the input to a transformer layer ℓ, where N is the number of image patches and d is the embedding dimensionality. We denote the set of all patch indices as P = {1,...,N}. During artifact injection, we accept the target-reference patch mapping M = {(pt,pr) ∈ P × P} provided by the toolbox, where pt is a target patch to be modified and pr its reference patch. The set of all target patches is PT = {pt | (pt,pr) ∈ M}, the set of all reference patches is PR = {pr | (pt,pr) ∈ M}, and the set of background patches becomes PB = P \ PT. Inversion Stage. The inversion stage maps the input image into its corresponding noisy latent representation. Queries, keys, and values for the self-attention layer are

###### 4.3. Curation Agent (Figure 3(c))

Q(ℓ) = X(ℓ)WQ(ℓ), K(ℓ) = X(ℓ)WK(ℓ), V (ℓ) = X(ℓ)WV(ℓ),

(1)

The curation agent is the pipeline’s quality assurance and enrichment stage, which refines the synthesis agent’s output into training-ready datasets for downstream tasks. Given paired real and artifact-injected images, the curation agent performs data filtering and explanation generation. This paired input enables reliable filtering and explanation by contrasting artifact-injected regions with real counterparts.

where WQ(ℓ),WK(ℓ),WV(ℓ) ∈ Rd×d are learnable. We encode positions with rotary embeddings (RoPE) [46] such that

Q˜(pℓ) = RoPE(Q(ℓ),p), K˜p(ℓ) = RoPE(K(ℓ),p), p ∈ P.

(2) Then, the layer’s attention output is

###### 4.3.1. Data Filtering

Q ˜(ℓ)K˜(ℓ)⊤ √

Attn(ℓ)(X(ℓ)) = Softmax

V (ℓ). (3)

This stage applies one of two filtering methods depending on which tool was used for artifact injection.

d

While executing Equation (3), we cache the per-layer value embeddings Vinv(ℓ) ← V (ℓ) for use in the injection stage.

• LPIPS-Based Filtering. Distortion artifacts are validated using the LPIPS [56] metric, which measures the perceptual difference between two images by comparing their feature representations, producing a metric that aligns much more closely with human judgments. For each cropped original-injected pair, we compute the LPIPS score and retain the pair if it satisfies τ1 ≤ 1 − dLPIPS(xoriginal,xartifact) ≤ τ2, with τ2 filtering out pairs too similar with unidentifiable changes, while τ1 filters out severely corrupted samples with implausible damage in the region.

Injection Stage. As in Figure 5, during the denoising process, the injection stage injects structural artifacts into PT by borrowing spatial semantics from PR, while keeping the original semantics of PB.

- • Target Region. For pt ∈ PT mapped to pr, we replace the positional embedding(PE) and the value embedding

of pt with those of pr: Q˜(pℓ)

= RoPE(Q(pℓ)

,pr), K˜p(ℓ)

= RoPE(Kp(ℓ)

t

t

t

,pr), and Vp(ℓ)

t ← Vp(ℓ)

r,inv.

t

- • VLM-Based Filtering. For duplication, omission, and fusion artifacts, we employ a VLM to validate whether the injected change is perceptible and localized within the designated region. The VLM receives the original image with the target region masked out (to provide the global scene context), the original image cropped to the target region (to show the unaltered content), and the artifactinjected image cropped to the same region (to focus on the modification). From this triplet, the VLM makes a binary judgment, confirming whether a new instance has appeared (duplication), an expected object has become missing (omission), or two objects have been unnaturally merged (fusion). For a detailed prompt template used for VLM-based filtering, see Appendix A.2.2.

###### 4.3.2. Explanation Generation

- • Local Explanation. For each candidate region, the VLM receives the same triplet used in filtering. We prompt the VLM to synthesize a short local description by guiding the VLM to describe what is different in the artifact region compared to the real counterpart. For the detailed prompt template used, see Appendix A.2.3.
- • Global Explanation. After generating local explanations for all artifacts, the curation agent generates a global explanation for the whole image. The VLM accepts the artifact image and a list of artifact-injected bounding boxes and their corresponding local explanations. The VLM is prompted to explain why this artifact injected image is indeed an artifact. For the detailed prompt template used, see Appendix A.2.3.

###### 4.4. Data Collection

With our novel three-stage agentic pipeline, we collect 50K pairs of artifact-injected images and the corresponding original images, along with the metadata consisting of artifact explanation shown in the final output of Figure 3. Here, we reconstruct the source images using the inversionrestoration method to alleviate pairwise differences originating from diffusion-generated image traits. The source images are composed of four different datasets, COCO [7], Caltech-101 [12], 11K Hands [1], and Celeba HQ [21], broadening the distribution from diverse real-world scenes to specific single entity images. We employ GPT-4o [33] as the VLM that contributes to our agentic flow. In addition, the inversion-injection method in Section 4.2.2 uses FLUX.1-dev [5] for the DiT and FireFlow [9] for the inversion-injection module.

#### 5. ArtiBench: Artifact Detection Benchmark

Existing artifact detection benchmarks, such as RichHF [25], LOKI [51], and SynthScars [20] were built primarily using earlier diffusion models, including Stable Diffusion 1 or 2 and Midjourney. These datasets,

Table 2. Comparison of artifact benchmark datasets. If a benchmark reused another dataset as a source, we describe the generative models used in that source. The table with the full citations is provided in Appendix B.1.

Source Models

Benchmark

Sample Bin. Loc. Exp. Oldest Newest

Dreamlike

955 ✓ Photoreal

RichHF SD2.1

LOKI pix2pix FLUX 229 ✓ ✓ SynthScars Midjourney DALL·E3 1K ✓ ✓ ArtiBench SD3.5 Nano-Banana 1K ✓ ✓ ✓

though valuable at the time, no longer capture the characteristics of modern artifacts produced by current diffusion and multimodal transformers. As a result, the evaluation on these benchmarks may not accurately reflect the artifact-handling capabilities of today’s models.

To address this issue, we introduce ArtiBench, a new benchmark that reflects the current state of artifact phenomena in recent generative models. As summarized in Table 2, previous benchmarks were limited either in sample diversity, recency of generative sources, or task coverage. Our benchmark is designed to overcome these limitations by including data from recent models and providing comprehensive annotations for multiple artifact-related tasks.

We construct ArtiBench with 1K images generated by five state-of-the-art diffusion models, Stable Diffusion3.5 [11], FLUX-schnell/dev [5], Qwen-Image [49], and Nano-Banana [13], with the prompts sampled from three datasets, MS-COCO [7], PartiPrompts [52], and FuseCap [44]. For annotation, we involve 12 human annotators and label each image with: (1) a binary indicator denoting the presence or absence of artifacts, (2) bounding boxes for all artifact regions, and (3) concise descriptions of the observed abnormalities. The dataset is balanced with an equal ratio of artifact-free and artifact-containing samples. Further details on the construction and annotation pipeline are provided in Appendix B.2.

#### 6. Experiments

###### 6.1. Understanding Artifacts with VLMs

We assess the efficacy of ArtiAgent by training VLMs with visual question-answering(VQA) samples generated from ArtiAgent and measuring their performance on artifactaware benchmarks, including ArtiBench.

###### 6.1.1. Setup

We consider three tasks, artifact detection, localization, and explanation. For evaluation metrics, we use accuracy and F1 score for detection, mIoU and F1 score for localization, and ROUGE and CSS for explanation. More details of the evaluation protocol are provided in Appendix B.3. For baselines, we use three artifact segmentation algorithms,

###### Table 3. Artifact understanding performance across (a) binary detection, (b) localization, and (c) explanation. The benchmarks are listed in the order in which they were published.

(a) Binary detection (b) Localization (c) Explanation

Methods

ArtiBench RichHF LOKI SynthScars ArtiBench LOKI SynthScars ArtiBench Acc F1 mIoU F1 mIoU F1 mIoU F1 mIoU F1 ROUGE CSS ROUGE CSS ROUGE CSS PAL - - 0.079 0.028 0.021 0.037 0.035 0.053 0.040 0.066 - - - - - -

DiffDoctor - - 0.077 0.139 0.175 0.274 0.083 0.136 0.081 0.137 - - - - - LEGION - - 0.067 0.112 0.100 0.158 0.106† 0.152† 0.062 0.099 0.133 0.314 0.247† 0.589† 0.143 0.332

- GPT-4o 0.619 0.601 0.086 0.040 0.037 0.056 0.032 0.052 0.049 0.084 0.107 0.266 0.125 0.404 0.143 0.433

Gemini-2.5-Pro 0.582 0.575 0.061 0.091 0.109 0.169 0.064 0.101 0.095 0.147 0.097 0.358 0.103 0.474 0.159 0.420

- GPT-5 0.599 0.577 0.126 0.146 0.089 0.141 0.117 0.185 0.061 0.099 0.121 0.382 0.120 0.461 0.145 0.434

Qwen2.5-VL-7B 0.501 0.336 0.075 0.028 0.052 0.068 0.013 0.018 0.010 0.014 0.106 0.267 0.115 0.362 0.117 0.263 + ArtiAgent 0.627 0.627 0.119 0.198 0.129 0.198 0.137 0.214 0.111 0.168 0.169 0.454 0.196 0.578 0.233 0.643 InternVL3.5-8B 0.498 0.357 0.013 0.022 0.015 0.025 0.019 0.033 0.010 0.015 0.081 0.189 0.050 0.180 0.126 0.256 + ArtiAgent 0.630 0.620 0.100 0.170 0.126 0.196 0.140 0.217 0.119 0.176 0.137 0.401 0.179 0.513 0.226 0.625

†LEGION was trained on the SynthScars training dataset split.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| |(0.10)| | |
| | | | |
| | | | |

PAL [55], DiffDoctor [48], and LEGION [20]; and three proprietary VLMs, GPT-4o [33], Gemini-2.5-Pro [14], and GPT-5 [34]. PAL and DiffDoctor are evaluated only on the localization task as they output only segmentations, LEGION is evaluated on the localization and explanation tasks, and the three VLMs are evaluated on all three tasks. We fine-tune Qwen2.5-VL-7B [4] and InternVL3.5-8B [57] on a 100K training set generated by ArtiAgent. The detailed procedure for VQA generation is provided in Appendix B.4.

(0.43)

Figure 6. Scaling effect of data generated by ArtiAgent with Qwen2.5-VL-7B. We average the results of all the benchmarks.

Table 4. ArtiBench result of the VLM trained on a 1K dataset.

(a) Binary detection (b) Localization (c) Explanation Acc F1 mIoU F1 ROUGE CSS

Methods

###### 6.1.2. Main Results

SynthScars 0.555 0.548 0.094 0.147 0.156 0.521 ArtiAgent 0.555 0.550 0.074 0.121 0.222 0.606

Overall, fine-tuning open-source VLMs with synthetic data generated by ArtiAgent consistently enhances their ability to detect, explain, and localize visual artifacts. Across all three tasks, ArtiAgent-trained models not only outperform their vanilla counterparts but also match or exceed the performance of proprietary systems such as GPT-5 and Gemini-2.5-Pro. These gains highlight the quality and scalability of ArtiAgent-generated supervision, demonstrating that artifact synthesis through agentic data generation provides rich, transferable signals for both spatial grounding and semantic reasoning.

Artifact Explanation. Table 3(c) shows that training with the ArtiAgent’s generated dataset greatly strengthens the reasoning and description capabilities of VLMs. The finetuned models exhibit clear improvements in the ROUGE and CSS metrics across all benchmarks.

###### 6.1.3. Data Scaling Effect of ArtiAgent

Figure 6 shows how model performance grows as we increase the size of the training data with ArtiAgent. In all three tasks, we observe a clear upward trend, which means that more synthesized data consistently lead to better artifact understanding. Notably, for localization and explanation, the performance with subsets as small as 1K samples already surpasses that of GPT-5, showing that ArtiAgent provides highly sample-efficient supervision. In contrast, the binary detection task continues to improve up to the 100K scale, suggesting that detection benefits from larger and more diverse artifacts. These results highlight the rich supervision and the strong scaling potential of ArtiAgent.

Artifact Binary Detection. Table 3(a) shows that VLMs trained with ArtiAgent achieve clearly superior artifact detection performance. Specifically, ArtiAgent improves the accuracy of InternVL3.5-8B by 26.5%. At the same time, the overall low accuracy on ArtiBench highlights the difficulty and importance of understanding visual artifacts in AI-generated images, especially because modern diffusion models increasingly exhibit subtle, structured failures that current VLMs easily miss.

###### 6.1.4. Comparison with Human-Annotated Supervision

Artifact Localization. Table 3(b) shows that ArtiAgent consistently enhances the spatial grounding capability of open-source VLMs. Although DiffDoctor exhibits high accuracy in LOKI, it fails to generalize to more recent benchmarks, including ArtiBench. This outcome means that ArtiBench further extends the artifact detection field one step deeper, by capturing artifacts that prior artifact-detection models struggle to identify.

To compare the quality of human annotation and ArtiAgentgenerated annotations, we train Qwen2.5-VL-7B with 500 SynthScars samples and 500 clean samples from ArtiAgent and compare it with a counterpart trained on 1K ArtiAgent samples. Table 4 shows that ArtiAgent’s synthetic supervision remains competitive with human annotation: SynthScars is slightly better in localization, while ArtiA-

CVPR #2904

CVPR 2026 Submission #2904. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.

toward artifact-free images.

CVPR #2904

Figure 6. Scaling law experiment on Qwen2.5-VL-7B with ArtiAgent. We average the results of all the benchmarks for each task.

- 485 ArtiAgent provides highly sample-efﬁcient supervision. In
- 486 contrast, the binary detection task continues to improve up
- 487 to the 100K scale, suggesting that detection beneﬁts from
- 488 larger and more diverse artifacts. These results highlight the
- 489 rich supervision and strong scaling potential of ArtiAgent.
- 490 6.2. Mitigating Artifacts in Diffusion Models
- 491 6.2.1. Reward-Guided Artifact-Free Generation
- 492 A key strength of ArtiAgent is its pairwise data design: for
- 493 every instance, it provides two tightly matched images with
- 494 the same content—one clean and one with artifact. This
- 495 structure gives extremely rich supervision for learning arti-
- 496 fact preferences. Using these pairs, we adopt the reward-
- 497 guided test-time scaling framework [25] that steers diffu-
- 498 sion models toward producing artifact-free images.
- 499 Setup. We use CLIP as the backbone of the Bradley-
- 500 Terry [4] reward model. The model learns to assign a higher
- 501 score to the real image over the artifact image. With this
- 502 artifact-aware reward model, we apply test-time scaling to
- 503 FLUX-schnell. We run six search rounds each for 100
- 504 prompts sampled from MS-COCO and measure how much
- 505 the reward increases as the search progresses.
- 506 Results.

Table 4. Reward-guided generation. ArtiAgent can train a reward model that guides diffusion to generate artifact-free images.

! Reward

0 0.06±0.11 0.10±0.08 0.13±0.08 0.16±0.07 0.23±0.08 “a photo of a man holding a phone”

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

“a very small child is laying on the ground with oranges”

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

- 507 As can be seen in Figure 7, the reward steadily improves
- 508 throughout the search rounds, indicating that the diffusion
- 509 model continues to generate images with fewer artifacts.
- 510 Qualitatively, the examples in Figure 7 show clearer struc-
- 511 tures and reduced artifact patterns in later rounds, demon-
- 512 strating that the reward model successfully guides sampling

Round 1 Search rounds Round 6

Figure 7. Reward-guided generation. ArtiAgent can train a reward model that guides diffusion to generate artifact-free images.

gent matches in detection and performs better in explanation. We attribute the localization gap mainly to the patchlevel granularity of ArtiAgent labels. Overall, these results indicate that ArtiAgent is a cost-effective and scalable alternative to human annotation while maintaining comparable supervision quality.

8

###### 6.2. Mitigating Artifacts in Diffusion Models

###### 6.2.1. Reward-Guided Artifact-Free Generation

A key strength of ArtiAgent is its pairwise data design: for every instance, it provides two tightly matched images with the same content, one clean and one with artifact. This structure gives extremely rich supervision for learning artifact preferences. Using these pairs, we adopt the rewardguided test-time scaling framework [28] that steers diffusion models toward producing artifact-free images.

Setup. We use CLIP as the backbone of the BradleyTerry [6] reward model. The model learns to assign a higher score to the real image over the artifact image. With this artifact-aware reward model, we apply test-time scaling to FLUX-schnell. We run six search rounds, each for 100 prompts sampled from MS-COCO and measure how much the reward increases as the search progresses. The detailed training schema and the test-time scaling procedure are provided in the Appendix C.1.

Results. As can be seen in Figure 7, the reward steadily improves throughout the search rounds, indicating that the diffusion model continues to generate images with fewer artifacts. Qualitatively, the examples in Figure 7 show clearer structures and reduced artifact patterns in later rounds, demonstrating that the reward model has successfully learned the real-artifact preference and enables guidance towards artifact-free images.

###### 6.2.2. VLM-Guided Artifact Correction

Since our artifact-trained VLM can reliably detect and localize artifacts, we employ it to guide an image inpainting model to correct artifact regions in AI-generated images.

Figure 7. Reward-guided generation. ArtiAgent can train a reward model that guides diffusion to generate artifact-free images.

513

###### 6.2.2. VLM-Guided Artifact Inpainting 514

Since our artifact-trained VLM can reliably detect and lo- 515 calize visual artifacts, we leverage this capability to guide 516 an image inpainting model to correct artifact regions of AI- 517 generated image. 518

Setup. We use Qwen2.5-VL-7B trained with ArtiAgent to 519 determine whether an artifact is present in a given image 520 and to localize the artifact region. This region is then pro- 521 cessed by the FLUX inpainting pipeline, which synthesizes 522 a corrected version of the localized area. Next, the corrected 523 image is re-evaluated by the VLM to verify whether the ar- 524 tifact in the region has been fully resolved. If the VLM 525 continues to detect an artifact, the inpainting procedure is 526 repeated. This iterative loop continues until the VLM con- 527 ﬁrms the absence of artifacts. 528

Results. Figure 8 presents qualitative results of our image 529 correction pipeline. The VLM accurately identiﬁes the ar- 530 tifact

531 w 532 su 533 ca 534 ar 535

|region, and with natural and sults demonstrate cate<br><br>|and|
|---|
<br><br>correct artifact understan<br><br>[Figure 102]<br><br>Before|the inpainting structurally con that the propos artifacts, depic ding VLM.<br><br>[Figure 103]<br><br>After|model correct sistent content ed pipeline can ting the useful<br><br>[Figure 104]<br><br>Before|s the region . These rereliably loness of our<br><br>[Figure 105]<br><br>After|
|---|---|---|---|
|Conclusion<br><br>this work, w framework th<br><br>facts through pos<br><br>[Figure 106]<br><br>Before|e introduce Ar at automatical itional embedd<br><br>[Figure 107]<br><br>After|tiAgent, a sca ly synthesizes ing manipulati<br><br>[Figure 108]<br><br>Before|lable agenvisual artion in diffu-<br><br>[Figure 109]<br><br>After|
|sion transformers curation agents, annotated artifact periments showed achieved substant<br><br>[Figure 110]<br><br>Before|. By integrating our pipeline ge<br><br>datasets witho<br><br>that VLMs ﬁn ial gains in arti<br><br>[Figure 111]<br><br>After|perception, sy nerates large-s ut human supe<br><br>e-tuned on Art fact detection,<br><br>[Figure 112]<br><br>Before|nthesis, and cale, richly<br><br>rvision. Ex-<br><br>iAgent data localization,<br><br>[Figure 113]<br><br>After|

##### 7. 536

In 537 tic 538 fa 539 si 540 cu 541 an 542 pe 543 ac 544 and explanation, and that the resulting models can guide 545 diffusion sampling toward artifact-free generations and per- 546 form automated artifact correction. Together, these results 547

Figure 8. Image correction. The ArtiAgent-trained VLM can effectively guide image inpainting models to correct artifact regions.

Setup. We use Qwen2.5-VL-7B trained with ArtiAgent to determine whether an artifact is present in a given image and to localize the artifact region. This region is then processed by the FLUX inpainting pipeline [3], which synthesizes a corrected version of the localized area. Next, the corrected image is re-evaluated by the VLM to verify whether the artifact in the region has been fully resolved in the specified region. If the VLM continues to detect an artifact, the inpainting procedure is repeated. This iterative loop continues until the VLM confirms the absence of artifacts. The detailed procedure is provided in the Appendix C.2.

Results. Figure 8 presents qualitative results of our image correction pipeline. The VLM accurately identifies the artifact region and the inpainting model corrects the region with natural and structurally consistent content. These results demonstrate that the proposed pipeline can reliably locate and correct artifacts, depicting the usefulness of our artifact understanding VLM.

#### 7. Conclusion

In this work, we introduce ArtiAgent, a scalable agentic framework that automatically synthesizes visual artifacts through positional embedding manipulation in diffusion transformers. By integrating perception, synthesis, and curation agents, our pipeline generates large-scale, richly annotated artifact datasets without human supervision. Experiments showed that VLMs fine-tuned on the ArtiAgent datasets achieved substantial gains in artifact detection, localization, and explanation, and that the resulting models can guide diffusion sampling toward artifact-free generations and perform automated artifact correction. Together, these results demonstrate that agentic data synthesis provides an effective and general pathway to perceiving and mitigating visual artifacts in modern generative models.

#### Acknowledgements

This work was supported by Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No. RS2020-II200862, DB4DL: High-Usability and Performance In-Memory Distributed DBMS for Deep Learning, 50% and No. RS-2025-25442149, LG AI STAR Talent Development Program for Leading Large-Scale Generative AI Models in the Physical AI Domain, 50%).

#### References

- [1] Mahmoud Afifi. 11k hands: gender recognition and biometric identification using a large dataset of hand images. Multimedia Tools and Applications, 78:20835–20854, 2019. 6
- [2] Vatsal Agarwal, Matthew Gwilliam, Gefen Kohavi, Eshan Verma, Daniel Ulbricht, and Abhinav Shrivastava. Towards multimodal understanding via stable diffusion as a taskaware feature extractor. arXiv preprint arXiv:2507.07106,

2025. 1

- [3] AlimamaCreative. Flux.1-dev controlnet inpainting (beta). https://huggingface.co/alimama-creative/ FLUX.1- dev- Controlnet- Inpainting- Beta,

2024. Model weights released under the FLUX.1 [dev] NonCommercial License. 8, 18

- [4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, and Others. Qwen2. 5-vl technical report. Arxiv Preprint Arxiv:2502.13923, 2025. 7, 17
- [5] Blackforestlabs. Flux: a powerful tool for text generation. https://blackforestlabs.ai/, 2024. 3, 6, 16
- [6] Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952. 8, 17
- [7] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: data collection and evaluation server. Arxiv Preprint Arxiv:1504.00325, 2015. 3, 6
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and Others. Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Arxiv Preprint Arxiv:2507.06261, 2025. 1
- [9] Yingying Deng, Xiangyu He, Changwang Mei, Peisong Wang, and Fan Tang. Fireflow: fast inversion of rectified flow for image semantic editing. In Proceedings of the International Conference on Machine Learning (ICML), 2025. 1, 4, 5, 6
- [10] Dreamlike Art. Dreamlike photoreal 2.0. https:// huggingface.co/dreamlike-art/dreamlikephotoreal-2.0, 2023. 16
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis.

- In Proceedings of the International Conference on Machine Learning (ICML), 2024. 3, 6, 16
- [12] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. Computer Vision and Image Understanding, 106(1):59–70,

2007. 6

- [13] Alisa Fortin, Guillaume Vernade, Kat Kampf, and Ammaar Reshi. Introducing gemini 2.5 flash image. https : / / developers . googleblog . com / en / introducing-gemini-2-5-flash-image/, 2025. Google AI / DeepMind blog post, August 26 2025. 1, 3, 6, 16
- [14] Gemini Team, Google DeepMind. Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Technical Report, 2025. 7
- [15] Ian Goodfellow, Jean Pouget-abadie, Mehdi Mirza, Bing Xu, David Warde-farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 3
- [16] Iryna Hartsock and Ghulam Rasool. Vision-language models for medical report generation and visual question answering: a review. Frontiers in Artificial Intelligence, 7:1430984,

2024. 1

- [17] Xinyu Huang. A survey of domain adaptation in robotics using diffusion models. Applied and Computational Engineering, 179:1–8, 2025. 1
- [18] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1125–1134, 2017. 16
- [19] Max Jiang, Yijing Bai, Andre Cornman, Christopher Davis, Xiukun Huang, Hong Jeon, Sakshum Kulshrestha, John Lambert, Shuangyu Li, Xuanyu Zhou, and Others. Scenediffuser: efficient and controllable driving simulation initialization and rollout. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pages 55729– 55760, 2024. 1
- [20] Hengrui Kang, Siwei Wen, Zichen Wen, Junyan Ye, Weijia Li, Peilin Feng, Baichuan Zhou, Bin Wang, Dahua Lin, Linfeng Zhang, and Others. Legion: learning to ground and explain for synthetic image detection. Arxiv Preprint Arxiv:2503.15264, 2025. 1, 2, 3, 6, 7, 16
- [21] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. In Proceedings of the International Conference on Learning Representations (ICLR), 2018. 6
- [22] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8110–8119, 2020. 16
- [23] Firas Khader, Gustav M¨uller-Franzes, Soroosh Tayebi Arasteh, Tianyu Han, Christoph Haarburger, Maximilian Schulze-Hagen, Philipp Schad, Sandy Engelhardt, Bettina Baeßler, Sebastian Foersch, et al. Denoising

- diffusion probabilistic models for 3d medical image generation. Scientific Reports, 13(1):7303, 2023. 1
- [24] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, and Others. Openvla: an open-source vision-language-action model. Arxiv Preprint Arxiv:2406.09246, 2024. 1
- [25] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Arxiv Preprint Arxiv:2305.01569, 2023. 6
- [26] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning (ICML), pages 19730–19742. PMLR, 2023. 1
- [27] Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-tuset, Sarah Young, Feng Yang, and Others. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19401–19411, 2024. 2, 3, 16
- [28] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, YuChuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Others. Inference-time scaling for diffusion models beyond scaling denoising steps. Arxiv Preprint Arxiv:2501.09732, 2025. 8, 17
- [29] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. 5
- [30] Midjourney, Inc. Midjourney. https : / / www . midjourney.com/, 2022. 16
- [31] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6038–6047, 2023. 5
- [32] Openai. Dall·e 3. https://openai.com/research/ dall-e-3, 2023. 16
- [33] OpenAI. Gpt-4o system card. Technical Report arXiv:2410.21276, OpenAI, 2024. 6, 7, 1
- [34] Openai. Gpt-5: large language model. https:// openai.com/research/gpt-5, 2025. 1, 7
- [35] Ji-Hoon Park, Yeong-Joon Ju, and Seong-Whan Lee. Explaining generative diffusion models via visual analysis for interpretable decision-making process. Expert Systems with Applications, 248:123231, 2024. 1
- [36] Taesung Park, Alexei A Efros, Richard Zhang, and JunYan Zhu. Contrastive learning for unpaired image-to-image translation. In Proceedings of the European Conference on Computer Vision (ECCV), pages 319–345. Springer, 2020. 16
- [37] Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, and Youngjung Uh. Understanding the latent space of diffusion models through the lens of riemannian geometry. In

- Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pages 24129–24142, 2023. 1
- [38] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In Proceedings of the International Conference on Learning Representations (ICLR), 2024. 16
- [39] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13009–13018, 2024. 3
- [40] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3982–3992. Association for Computational Linguistics, 2019. 16
- [41] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, and Others. Grounded sam: assembling open-world models for diverse visual tasks. Arxiv Preprint Arxiv:2401.14159, 2024. 3
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 3, 16
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 1
- [44] Noam Rotstein, David Bensaid, Shaked Brody, Roy Ganz, and Ron Kimmel. Fusecap: leveraging large language models for enriched fused image captions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5689–5700, 2024. 6
- [45] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo lopes, Burcu Karagol ayan, Tim Salimans, and Others. Photorealistic text-to-image diffusion models with deep language understanding. In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pages 36479–36494, 2022. 1
- [46] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

- 2024. 5

[47] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. In Proceedings of the International Conference on Machine Learning (ICML),

- 2025. 5

- [48] Yiyang Wang, Xi Chen, Xiaogang Xu, Sihui Ji, Yu Liu, Yujun Shen, and Hengshuang Zhao. Diffdoctor: diagnos-

ing image diffusion models before treating. Arxiv Preprint Arxiv:2501.12382, 2025. 1, 2, 3, 7

- [49] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-Ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, and Others. Qwen-image technical report. Arxiv Preprint Arxiv:2508.02324, 2025. 3, 6, 16
- [50] Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, and Weidi Xie. A sanity check for AIgenerated image detection. In Proceedings of the International Conference on Learning Representations (ICLR),

2025. 16

- [51] Junyan Ye, Baichuan Zhou, Zilong Huang, Junan Zhang, Tianyi Bai, Hengrui Kang, Jun He, Honglin Lin, Zihao Wang, Tong Wu, and Others. Loki: a comprehensive synthetic data detection benchmark using large multimodal models. Arxiv Preprint Arxiv:2410.09732, 2024. 2, 6, 16
- [52] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. Featured Certification. 6
- [53] Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon. Text-to-image diffusion models in generative ai: a survey. Arxiv Preprint Arxiv:2303.07909, 2023. 1
- [54] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: a survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8):5625–5644, 2024. 1
- [55] Lingzhi Zhang, Zhengjie Xu, Connelly Barnes, Yuqian Zhou, Qing Liu, He Zhang, Sohrab Amirghodsi, Zhe Lin, Eli Shechtman, and Jianbo Shi. Perceptual artifacts localization for image synthesis tasks. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 7579–7590, 2023. 1, 2, 3, 7
- [56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 586–595, 2018. 5
- [57] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao, Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Internvl3: exploring advanced training and test-time recipes for open-source multimodal models. Arxiv Preprint Arxiv:2504.10479, 2025. 7, 17

## See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis

### Supplementary Material

#### A. ArtiAgent Pipeline Details (§ 4)

###### A.1. Implementation Details

Injection Stage. The details of the injection process in Section 4.2.2 were designed to introduce natural and coherent structural artifacts to the image. We accept and build on the coarse-to-fine manner of denoising processes, adapting the understanding that earlier denoising time steps contribute to overall structural context, while later steps focus on regions of fine-grained details [2, 35, 37].

- • PE Injection. Among the 25 denoising steps, we disable PE injection for selected final steps and apply PE injection only in the earlier steps. Duplication, distortion, and fusion artifacts disable five final steps, focusing on introducing coarse context of structural modifications while maintaining natural connection with the original scene. In contrast, the omission artifact is generated by disabling PE injection only in the final timestep, since it requires stronger perturbation on positional context to remove an existing feature from the image and distinguish the newly injected background from its original entity.
- • Value Injection. Value injection is performed only for the first 15 denoising steps. Reducing the injection steps for value injection ensures artifact injection while maintaining image quality. For the architecture consisting of sequential double stream blocks and single stream blocks of FLUX.1-dev, only the deeper single stream blocks (20-

38) carry out the value injection process.

Filtering. To ensure alignment with human visual perception, the distortion-type artifact filtering thresholds, τ1 and τ2, were heuristically established at 0.5 and 0.9, respectively. For each artifact region cropped from the original image and the artifact-injected image, a LPIPS distance was measured to ensure certain quality among the artifacts introduced. While malformed regions that are not acceptable as plausible structural artifacts are filtered out by a high LPIPS distance over 0.5, unsuccessful artifact injections with similar features are discarded by the low LPIPS distance, indicating high similarity.

###### A.2. Prompt Templates for ArtiAgent A.2.1. Entity-Subentity Vocabulary (§ 4.1.1)

We employ the capability of GPT-4o [33] to identify and recognize relationships between entities in images. To ensure that the extracted entity-subentity sets are clearly segmentable and valid for generating plausible artifacts, we set

strict rules and guidance as in Figure 12 and instruct the VLM model to respond with qualified sets of vocabulary in an explicit format. Figure 13 shows the precise prompt used to generate the entity-subentity vocabulary sets and an example response from the perception agent.

###### A.2.2. Data Filtering (§ 4.3.1)

The curation agent uses a set of deliberately described artifact types referring to the injection methods and detailed instructions to detect or explain the artifacts, shown in Figure 14. Utilizing the descriptions and criteria according to the type of artifact injected, we query the VLM model with the entity name and a triplet of images consisting of (1) the original image with the target part masked out, (2) the original image with only the target part cropped, and (3) the generated image with only the target part cropped.

Although modern out-of-the-box VLMs demonstrate limited understanding of structural artifacts, providing the models with the rich image context through the image triplet and detailed descriptions of the type of artifact enhances their reliability for this task. The prompt template is elaborated in Figure 15.

###### A.2.3. Explanation Generation (§ 4.3.2)

Similarly to Appendix A.2.2, we employ artifact type description in Figure 14 to generate rich explanations. The detailed prompt and an example of the local and global explanation generation process is shown in Figure 16 and Figure 17. By providing multiple images that contain both the global and local view, we maximize VLM capacity of understanding structural artifacts for reliable quality in explanations. Moreover, we employ BLIP2 [26] to generate concise caption describing the real image.

###### A.3. Synthesis Agent Tools (§ 4.2.1)

The artifact injection tools in the synthesis agent produce target–reference patch mappings that are subsequently consumed by the inversion–injection module in Section 4.2.2. Each tool follows a common interface but implements a different geometric prior tailored to a specific artifact type (duplication, omission, distortion, and fusion).

Notation. Let the image be discretized into a patch grid of size (hp,wp), and let Pall be the set of all patch coordinates on this grid. For a given tool call, the tool outputs a target–reference mapping

###### M = {(pt,pr)},

where pt ∈ Pall denotes a target patch to be modified, and pr ∈ Pall is its reference patch whose semantics will be injected at pt during the diffusion inversion–injection stage. We freely switch between linear indices and (y,x) grid coordinates using simple index–coordinate conversion routines.

For tools that operate on entity- or subentity-specific regions, we additionally assume access to patch sets such as the same-entity foreground Pent and same-subentity foreground Psub. Concretely, Pent contains all patches that belong to a single object instance (e.g., one person or one dog), while Psub ⊆ Pall marks patches of other instances of the same semantic part (e.g., hands of other people, paws of other dogs) that we want to avoid colliding with. These sets are provided by the synthesis agent’s perception stage (e.g., derived from instance/part segmentation) and are treated as fixed inputs when running the tools.

Whenever we say that we clip to the valid patch grid, we mean that any candidate coordinate (i,j) whose row or column index falls outside the range 0 ≤ i⟨hp or 0 ≤ j⟨wp is either discarded or projected back into the rectangular domain [0,hp − 1] × [0,wp − 1] by truncating it to the nearest boundary index, so that all patches used by the tools lie on valid positions of the patch grid.

Add Tool (duplication). The Add Tool (Algorithm 1) realizes duplication-type artifacts by creating an extra, plausibly placed copy of a subentity (e.g., an extra hand or paw adjacent to the original one). Given a set of reference patches PR for the original subentity, the tool first computes the subentity centroid (ci,cj) in patch space and constructs a perimeter band Pring of candidate locations around this centroid with Manhattan distance in [1,α]. For each candidate (i,j) ∈ Pring, it evaluates the score

###### S(i,j) = 3 − rself − rent − rsub gdist,

where rself, rent, and rsub measure the overlap ratio of the shifted subentity with (i) the original subentity region itself, (ii) other foreground patches of the same entity, and (iii) foreground patches belonging to other instances of the same subentity, respectively, and gdist is a distance-based decay term that penalizes large offsets. Intuitively, this prefers candidate locations that (i) stay close to the source entity, (ii) minimally overlap the original instance, and (iii) avoid collisions with other same-subentity regions. The tool then selects the best-scoring perimeter patch (i⋆,j⋆), computes the corresponding offset (∆⋆i ,∆⋆j), and builds the mapping

M = {((ri + ∆⋆i ,rj + ∆⋆j),(ri,rj)) : (ri,rj) ∈ PR}, which duplicates the entire subentity at the chosen location. Remove Tool (omission). The Remove Tool (Algorithm 2) implements omission-type artifacts by erasing a subentity and filling the region with nearby background context. The

target set PT is the set of patch coordinates belonging to the subentity being removed. The tool constructs a local neighborhood

Pnbr = {p ∈ Pall : ∥p − pt∥1 ≤ R,pt ∈ PT, p ∈/ PT }, discarding any coordinates outside the valid patch grid.

From this neighborhood it derives:

Pnbr-no-sub = Pnbr \ Psub, Pnbr-non-ent = Pnbr \ Pent. If sufficiently many true background patches exist, i.e.

|Pnbr-non-ent|⟩21|Pnbr-no-sub|,

the tool prioritizes them as the reference pool PRpool = Pnbr-non-ent; otherwise it uses PRpool = Pnbr-no-sub, which avoids collisions with other same-subentity patches but may include same-entity foreground.

Finally, each target patch pt ∈ PT selects the nearest reference patch under L1 distance:

∥pt − p∥1,

pr = arg min

p∈PRpool

and the Remove Tool outputs the mapping M = {(pt,pr)}. Distort Tool (distortion). The Distort Tool (Algorithm 3) performs structural perturbations within a subentity while keeping its global placement intact. Here, the target and reference sets are drawn from the same foreground region: PT indexes the original subentity patches, and PR is obtained by applying a distortion kernel. The tool supports three kernel types:

- • Shuffle kernel. The simplest kernel copies PT into PR and applies a random permutation. Each target patch is thus reassigned to a randomly chosen patch of the same subentity, breaking local structure while preserving appearance statistics.
- • Gaussian jitter kernel. For each (py,px) ∈ PT, the kernel repeatedly samples a discrete offset from a Gaussian

distribution in patch space, (δy,δx) ∼ N(0,σ2I). If a sampled location falls within the same-entity foreground Pent, it is accepted as the reference; otherwise, the kernel resamples up to a fixed budget and falls back to the nearest foreground (or self) patch if necessary. This yields small, local displacements that bend the entity’s internal geometry.

- • Strip kernel. This kernel first computes the bounding

box of PT and chooses a dominant direction (vertical or horizontal) by aspect ratio. It then partitions the subentity into S strips along that direction and, within each strip, imposes an ordering of patches. Each strip is circularly shifted by an integer offset ∆s (with alternating signs and magnitudes), and the shifted positions define the references. This produces band-like shearing or sliding artifacts within the entity.

After applying the chosen kernel to form PR, the tool returns the one-to-one mapping M = {(PT[i],PR[i])}, which causes the inversion–injection module to reconstruct a structurally distorted yet context-consistent object.

Fuse Tool (fusion). The Fuse Tool (Algorithms 4–5) introduces fusion artifacts along the interface of two overlapping entities with patch sets PA and PB. It first identifies the overlap region

Poverlap = PA ∩ PB

and the union foreground Pfg = PA ∪ PB, as well as the non-overlapping parts PA\B = PA \ Poverlap and PB\A = PB \Poverlap. If Poverlap = ∅, no fusion is applied. Given the overlap, the tool constructs a thin fusion band PT around Poverlap by dilating each overlap patch within an L1 radius R and intersecting with Pfg.

To avoid treating the band as a single global region, the tool selects up to K seeds on PT via farthest-point sampling and assigns each band patch to its nearest seed in L1 distance, forming local regions {Rs}. For each region, it then chooses an opposite-side pool Popp: if the region’s seed is closer (in L1) to PA\B, the pool is PB\A, and vice versa; if distances tie or a side is empty, the pool defaults to Pfg\PT. Over a discrete set of small integer offsets

Ω = {(∆i,∆j) : 1 ≤ |∆i| + |∆j| ≤ Roff},

the tool then selects the offset ∆⋆ which, when applied to patches in the region, maps the largest number of them onto valid, non-band patches in Popp. Each band patch is finally paired either with its offset-shifted opposite patch (if valid) or with the nearest patch in Popp in L1 distance. The resulting mapping M injects appearance from one entity into the boundary band of the other, producing visually plausible yet structurally implausible fusion along their interface. In practice, we optionally add a subset of reversed pairs (pr,pt) while ensuring that each target is unique, which yields more symmetric blending as visualized in Figure 4.

###### A.4. Qualitative Visualizations

To validate the coverage and generalization of our synthesized artifacts to natural ones, we analyze attention maps for InternVL3.5-8B before and after fine-tuning with ArtiAgent data. As shown in Figure 9, the fine-tuned model precisely attends to artifact regions in both synthetic (ArtiAgent) and real (ArtiBench) images. This comparison confirms that our ArtiAgent-trained VLM successfully learns general artifact features rather than relying on shortcut features such as edge discontinuities, thereby demonstrating strong distributional alignment.

Figure 10 and Figure 11 illustrate the sampled images generated by the ArtiAgent pipeline and its synthesized annotations. The bounding boxes highlight the target patch area where artifacts are injected by the synthesis agent.

[Figure 114]

[Figure 115]

(a) ArtiAgent (b) ArtiBench

[Figure 116]

[Figure 117]

Figure 9. Attention Visualization. We compare the attention maps of InternVL3.5-8B before (base) and after (fine-tuned) training on ArtiAgent. The fine-tuned model reliably focuses on genuine artifact features across both synthesized images and realworld images.

###### A.5. Ablation Studies

To justify the selected configuration of the synthesis agent in Appendix A.1, we conducted ablations on the PE injection steps along with the value injection blocks and visualized their results. Figure 18 shows how the choice of injection steps and value-injecting blocks affects artifact injection quality.

To investigate the effect of injection strength on artifact synthesis, we conducted an ablation study on the number of injection steps during the denoising process. Table 5 shows that VLM performance peaks when artifacts are injected for 15 steps. Downstream performance degrades if the steps are too few (due to failed artifact injection) or too many (due to overall image quality degradation), supporting our choice of 15 injection steps.

Table 5. Ablation study on injection steps. Performance of Qwen2.5-VL-7B on ArtiBench binary detection when trained with 1K ArtiAgent samples generated using varying numbers of injection steps out of 25 total steps.

Steps Acc F1

5/25 0.513 0.377 10/25 0.583 0.565 15/25 0.586 0.570 20/25 0.540 0.477

|[Figure 118]|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]|
|---|---|---|---|---|---|---|---|
|[Figure 126]|[Figure 127]|[Figure 128]|[Figure 129]|[Figure 130]|[Figure 131]|[Figure 132]|[Figure 133]|
|[Figure 134]|[Figure 135]|[Figure 136]|[Figure 137]|[Figure 138]|[Figure 139]|[Figure 140]|[Figure 141]|
|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|[Figure 146]|[Figure 147]|[Figure 148]|[Figure 149]|

Figure 10. Visualizations of artifacts injected with ArtiAgent. The first and third rows show the original images, whereas the second and fourth rows show the output images with artifacts injected.

[Figure 150]

[Figure 151]

“bbox”: [424, 317, 456, 397] “entity”: “person” “subentity”: “leg” “label”: “The person is missing a leg.” “explanation”: “The image depicts a person surfing, but there is an unusual absence of one leg, creating an empty space where the leg should be. This anomaly disrupts the natural appearance of the scene, as the person appears to be balancing on the surfboard with only one leg visible.” ”image_caption”: “a person is standing on a surfboard in the ocean.”

| |
|---|

Clean Artifact

Figure 11. An instance of ArtiAgent with the annotation.

###### <Output Format>

peripheral: {

- "<peripheral_entity1>": ["<peripheral_sub1>", "<peripheral_sub2>", "..."],
- "<peripheral_entity2>": ["<peripheral_sub1>", "<peripheral_sub2>", "..."],

...}, intermediate: {

- "<intermediate_entity1>": ["<intermediate_sub1>", "<intermediate_sub2>", "..."],
- "<intermediate_entity2>": ["<intermediate_sub1>", "<intermediate_sub2>", "..."],

...}

###### <Hard Rules>

- 1) Each returned entity MUST have at least one subentity in its dictionary. b return an empty list for any entity.
- 2) If you cannot name at least one clearly visible subentity for an entity in that layer, omit that entity from that dictionary.
- 3) Subentities must be clearly visible and reasonably segmentable in the image.
- 4) Exclude parts that are tightly bound to or visually fused with the torso/core body (e.g., arms pressed to sides, folded wings against body). Only include subentities with clear visual separation.
- 5) Do not invent parts that are occluded, cropped out, or ambiguous.
- 6) Use concise, lowercase nouns; de-duplicate terms. Prefer 1–6 subentities per entity per dictionary.
- 7) Return exactly one JSON object with exactly two top-level keys: "peripheral" and "intermediate" (not an array, not multiple objects).
- 8) Granularity rule (coarsity): Choose the most specific visible entity. If only a part is clearly visible (e.g., a leg without enough evidence of the full person), output that part as the entity (e.g., "leg") rather than its parent (e.g., "person"). Do not infer parent entities that are not clearly visible.
- 9) Peripheral variable-cardinality ban: In the peripheral dictionary, do not include variable-cardinality micro-parts (0..n multiplicity) such as leaves, hairs, feathers, scales, spikes, thorns, grains, pebbles, raindrops, confetti, crowd members, etc. Prefer distal parts with fixed or small bounded counts (e.g., hand, finger, ear, eye, wheel, mirror). If only variable-cardinality micro-parts are visible for an entity, omit that entity from the peripheral dictionary.

###### <Layering Guidance>

- - Peripheral dictionary (index 0): Include outermost parts with clear boundaries (e.g., arm, leg, wing, fin, hand, finger, nail, ear, eye, paw, tail, wheel, mirror, antenna).
- - For the peripheral dictionary, exclude variable-cardinality micro-parts (e.g., leaves, hairs, feathers, scales, spots/pattern dots, raindrops); select fixedcardinality or bounded-count distal parts instead.
- - Intermediate/Core dictionary (index 1): Include mid-level structural parts (e.g., arm, leg, face, door, window).
- - If a subentity fits both notions, prefer peripheral only if it is clearly distal and separable (e.g., "hand" → peripheral; "arm" → intermediate).
- - If a candidate layer has no valid entities with visible subentities, set that key to an empty object `[]` (e.g., `"peripheral": []` or `"intermediate": []`).

Figure 12. Guidance and rules for generating entity-subentity vocabulary sets.

###### <User Prompt>

You are given an image. Identify the visible entities and their subentities, split into two layers:

- - Peripheral (outermost) subentities first (e.g., hands, fingers, eyes, ears, paws, tails, wheels, mirrors).
- - Intermediate/Core subentities second (e.g., arms, legs, face, head, door, window). Output must return two dictionaries for each peripheral and intermediate types: {OUTPUT FORMAT} Hard rules: {HARD RULES} Layering guidance: {LAYERING GUIDANCE} Clarifications:
- - Examples of valid subentities:

- • person → face, arm, hand, leg, foot, ear, eye
- • hand → finger, nail, palm
- • dog → ear, snout, leg, tail, paw
- • car → wheel, door, window, mirror

- - Avoid generic torso-like regions. If no fine-grained parts are clearly separable for a candidate entity, omit the entity rather than returning an empty

list.

- - Granularity examples:

• If only a single leg is clearly visible:

{"peripheral": {"leg": ["knee", "ankle", "foot"]}, "intermediate": []}

- - Variable-cardinality micro-parts are not allowed in the peripheral dictionary. Examples to avoid: leaf/leaves (tree), hair/hairs (person/animal),

feather/feathers (bird), scale/scales (fish/reptile), spot/spots (dalmatian), petal/petals (flower), book/books (bookshelf), crowd/persons (crowd scene). Bad examples (NOT allowed):

- - Returning a single array or more than one top-level JSON object
- - {"dog": []} # empty subentities list
- - Multiple separate top-level JSON objects
- - "peripheral": [{ "entity": "tree", "subentities": ["leaf", "fruit"]

}] # variable-cardinality micro-parts in peripheral

Good format examples (illustrative only):

{ "peripheral": [{"entity": "person","subentities": ["hand", "finger", "leg", "arm"]}, {"entity": "car","subentities": ["wheel", "mirror"]}, {"entity": "dog", "subentities": ["ear", "paw", "leg", "tail"]}],

"intermediate": [{"entity": "person", "subentities": ["face", "palm"]},

{"entity": "car", "subentities": ["door", "window"]}, {"entity": "dog", "subentities": ["leg", "face"]}] }

{ "peripheral": [{"entity": "cat", "subentities": ["ear", "paw", "leg", "tail"]},

{"entity": "bicycle", "subentities": ["wheel", "pedal"]}], "intermediate": [{"entity": "cat", "subentities": ["face"]},

{"entity": "bicycle", "subentities": ["frame", "seat"]}] }

Edge-case examples:

- - Only distal parts visible: {"peripheral": [{"entity": "hand", "subentities": ["finger", "nail"]}], "intermediate": []}
- - Only intermediate parts visible: {"peripheral": [], "intermediate": [{"entity": "person", "subentities": ["face", "palm"]}]}
- - Multiple distal entities visible, no intermediate: {"peripheral": [{"entity": "hand", "subentities": ["finger", "nail"]},

{"entity": "dog", "subentities": ["ear", "paw", "tail", "leg"]}], "intermediate": []}

###### <Example>

Input: {image} {USER PROMPT} Output: {"peripheral": {"cannon": ["wheels", "barrel opening"]},

"intermediate": {"cannon": ["barrel", "carriage"]}}

Figure 13. Full prompt for generating entity-subentity vocabulary sets.

###### Algorithm 1 Add Tool

Require: Reference patches PR, patch grid (hp, wp), same-entity foreground Pent, same-subentity foreground Psub, ring thickness α, distance weight

λdist

Ensure: target-reference patch mapping M = {(pt, pr)}

- 1: (ci, cj) ← CENTROID(PR) /* subentity center in patch space */
- 2: Pring ← PERIMETERBAND(PR, (ci, cj), α, hp, wp) /* candidate centroid patches band around the subentity */
- 3: S ← ∅
- 4: for each (i, j) ∈ Pring do
- 5: ∆i ← i − ci, ∆j ← j − cj
- 6: Pshift ← {(ri + ∆i, rj + ∆j) : (ri, rj) ∈ PR}
- 7: rself ← |Pshift|P ∩PR|

shift|

- 8: rent ← |Pshift∩|P(Pent\PR)|

shift|

- 9: rsub ← |Pshift|P∩Psub|

shift|

- 10: dL1 ← |i − ci| + |j − cj|
- 11: gdist ← 1+λ1

distdL1

- 12: S(i, j) ← (3 − rself − rent − rsub) · gdist
- 13: (i⋆, j⋆) ← arg max(i,j)∈Pring S(i, j)
- 14: ∆⋆i ← i⋆ − ci, ∆⋆j ← j⋆ − cj
- 15: M ← {((ri + ∆⋆i , rj + ∆⋆j), (ri, rj)) : (ri, rj) ∈ PR}
- 16: return M

- 17: function CENTROID(PR)
- 18: n ← |PR|
- 19: ci ← n1 (ri,rj)∈PR ri

- 20: cj ← n1 (ri,rj)∈PR rj

- 21: ci ← round(ci), cj ← round(cj)
- 22: return (ci, cj)

- 23: function PERIMETERBAND(PR, (ci, cj), α, hp, wp)
- 24: Pring ← ∅
- 25: for i ← 0 to hp − 1 do
- 26: for j ← 0 to wp − 1 do
- 27: dL1 ← |i − ci| + |j − cj|
- 28: if 1 ≤ dL1 ≤ α then
- 29: Pring ← Pring ∪ {(i, j)}
- 30: return Pring

- Algorithm 2 Remove Tool

Require: Subentity patch set PT , patch grid (hp, wp), same-entity foreground Pent, same-subentity foreground Psub, neighborhood radius R Ensure: Target–reference mapping M

- 1: Pnbr ← LOCALNEIGHBORHOOD(PT , R, hp, wp) /* collect nearby non-target patches in the grid */
- 2: Pnbr-no-sub ← Pnbr \ Psub
- 3: Pnbr-non-ent ← Pnbr \ Pent
- 4: if |Pnbr-non-ent| > 12|Pnbr-no-sub| then

- 5: PRpool ← Pnbr-non-ent
- 6: else
- 7: PRpool ← Pnbr-no-sub
- 8: M ← ∅
- 9: for each pt ∈ PT do
- 10: pr ← arg minq∈Ppool R

∥pt − q∥1

- 11: M ← M ∪ {(pt, pr)}
- 12: return M

- 13: function LOCALNEIGHBORHOOD(PT , R, hp, wp)
- 14: Pnbr ← ∅
- 15: for each (ti, tj) ∈ PT do
- 16: for dy = −R to R do
- 17: for dx = −R to R do
- 18: if |dy| + |dx| ≤ R and (dy, dx) ̸= (0, 0) then
- 19: pi ← ti + dy, pj ← tj + dx
- 20: if 0 ≤ pi < hp and 0 ≤ pj < wp and (pi, pj) ∈/ PT then
- 21: Pnbr ← Pnbr ∪ {(pi, pj)}
- 22: return Pnbr

###### Algorithm 3 Distort Tool

Require: Subentity patches PT , patch grid (hp, wp), same-entity patches Pent, kernel type k ∈ {shuffle, jitter, strip} Ensure: target-reference patch mapping M = {(pt, pr)}

- 1: if k = shuffle then
- 2: PR ← SHUFFLEKERNEL(PT ) /* randomly permute subentity patches */
- 3: else if k = jitter then
- 4: PR ← GAUSSIANJITTERKERNEL(PT , σ, hp, wp, Pent) /* locally jitter patches within the entity */
- 5: else if k = strip then
- 6: PR ← STRIPSHIFTINGKERNEL(PT , S, hp, wp) /* shift patches along strips of the subentity */
- 7: M ← {(PT [i], PR[i]) : i = 1, . . . , |PT |}
- 8: return M

- 9: function SHUFFLEKERNEL(PT )
- 10: PR ← PT
- 11: RANDOMSHUFFLE(PR)
- 12: return PR

- 13: function GAUSSIANJITTERKERNEL(PT , σ, hp, wp, Pent)
- 14: PR ← ∅
- 15: for each (py, px) ∈ PT do
- 16: found ← False
- 17: for a ← 1 to Amax do
- 18: δy ∼ N(0, σ2), δx ∼ N(0, σ2)
- 19: ny ← clip(round(py + δy), 0, hp − 1)
- 20: nx ← clip(round(px + δx), 0, wp − 1)
- 21: if Pent = ∅ or (ny, nx) ∈ Pent then
- 22: PR ← PR ∪ {(ny, nx)}
- 23: found ← True
- 24: break
- 25: if not found then
- 26: (ny, nx) ← NEARESTFOREGROUNDORSELF(py, px, Pent)
- 27: PR ← PR ∪ {(ny, nx)}
- 28: return PR

- 29: function STRIPSHIFTINGKERNEL(PT , S, hp, wp)
- 30: Compute bounding box of PT in patch space
- 31: Determine direction d ∈ {vertical, horizontal} from aspect ratio
- 32: Partition PT into S strips {S1, . . . , SS} along d
- 33: For each strip Ss, sort patches to obtain an order (p(1s), . . . , p(nss))
- 34: Choose integer strip shifts {∆s}Ss=1 (alternating signs, increasing magnitudes)
- 35: PR ← list of length |PT |
- 36: for s ← 1 to S do
- 37: ∆ ← ∆s converted to patch units (circular shift)
- 38: for u ← 1 to ns do
- 39: v ← 1 + ((u + ∆ − 1) mod ns)
- 40: Set reference of p(us) to p(vs) in PR
- 41: return PR

- Algorithm 4 Fuse Tool (Main)

Require: Entity A patches PA, entity B patches PB, patch grid (hp, wp), band radius R, max offset Roff, number of seeds K Ensure: target-reference patch mapping M = {(pt, pr)}

- 1: Poverlap ← PA ∩ PB
- 2: if Poverlap = ∅ then
- 3: return ∅
- 4: Pfg ← PA ∪ PB
- 5: PA\B ← PA \ Poverlap, PB\A ← PB \ Poverlap
- 6: PT ← OVERLAPFUSIONBAND(Poverlap, Pfg, R, hp, wp) /* build foreground band around the overlap */
- 7: if PT = ∅ then
- 8: return ∅
- 9: S ← FARTHESTPOINTSAMPLING(PT , K) /* choose seeds that cover the band */
- 10: Initialize {Rs}s∈S as empty sets
- 11: for each p ∈ PT do
- 12: s⋆ ← arg mins∈S ∥p − s∥1
- 13: Rs⋆ ← Rs⋆ ∪ {p}
- 14: Ω ← {(∆i, ∆j) : 1 ≤ |∆i| + |∆j| ≤ Roff}
- 15: M ← ∅
- 16: for each seed s ∈ S do
- 17: R ← Rs
- 18: if R = ∅ then
- 19: continue
- 20: Popp ← OPPOSITEREGION(s, PA\B, PB\A, Pfg, PT ) /* decide which side to fuse from */
- 21: ∆⋆ ← BESTOFFSET(R, Popp, Ω, hp, wp, PT ) /* find best shared shift for this region */
- 22: for each p ∈ R do
- 23: pr ← OFFSETORNEAREST(p, ∆⋆, Popp, hp, wp, PT ) /* apply offset or nearest opposite patch */
- 24: M ← M ∪ {(p, pr)}
- 25: return M

- Algorithm 5 Fuse Tool (Helpers)

- 1: function OVERLAPFUSIONBAND(Poverlap, Pfg, R, hp, wp)
- 2: PT ← ∅
- 3: for each (oi, oj) ∈ Poverlap do
- 4: for dy ← −R to R do
- 5: for dx ← −R to R do
- 6: if |dy| + |dx| ≤ R then
- 7: vi ← oi + dy, vj ← oj + dx
- 8: if 0 ≤ vi < hp and 0 ≤ vj < wp and (vi, vj) ∈ Pfg then
- 9: PT ← PT ∪ {(vi, vj)}
- 10: return PT

- 11: function FARTHESTPOINTSAMPLING(P, K)
- 12: if P = ∅ then
- 13: return ∅
- 14: K ← min(K, |P|)
- 15: Choose initial seed s1 ∈ P (e.g., closest to centroid)
- 16: S ← {s1}; d(p) ← ∥p − s1∥1 for all p ∈ P
- 17: for m ← 2 to K do
- 18: sm ← arg maxp∈P d(p)
- 19: S ← S ∪ {sm}
- 20: for each p ∈ P do
- 21: d(p) ← min{d(p), ∥p − sm∥1}
- 22: return S

- 23: function OPPOSITEREGION(s, PA\B, PB\A, Pfg, PT )
- 24: dA ← +∞, dB ← +∞
- 25: if PA\B ̸= ∅ then
- 26: dA ← mina∈PA\B ∥s − a∥1
- 27: if PB\A ̸= ∅ then
- 28: dB ← minb∈PB\A ∥s − b∥1
- 29: if dA < dB then
- 30: return PB\A
- 31: else if dB < dA then
- 32: return PA\B
- 33: else
- 34: return Pfg \ PT

- 35: function BESTOFFSET(R, Popp, Ω, hp, wp, PT )
- 36: ∆⋆ ← None, m⋆ ← 0
- 37: for each (∆i, ∆j) ∈ Ω do
- 38: count ← 0
- 39: for each (vi, vj) ∈ R do
- 40: ri ← vi + ∆i, rj ← vj + ∆j
- 41: if 0 ≤ ri < hp and 0 ≤ rj < wp and (ri, rj) ∈ Popp and (ri, rj) ∈/ PT then
- 42: count ← count + 1
- 43: if count > m⋆ then
- 44: m⋆ ← count, ∆⋆ ← (∆i, ∆j)
- 45: return ∆⋆

- 46: function OFFSETORNEAREST(p, ∆⋆, Popp, hp, wp, PT )
- 47: (vi, vj) ← p
- 48: if ∆⋆ ̸= None then
- 49: (∆i, ∆j) ← ∆⋆
- 50: ri ← vi + ∆i, rj ← vj + ∆j
- 51: if 0 ≤ ri < hp and 0 ≤ rj < wp and (ri, rj) ∈ Popp and (ri, rj) ∈/ PT then
- 52: return (ri, rj)
- 53: return arg minu∈Popp ∥(vi, vj) − u∥1

|Duplication|
|---|
|<Type Description><br><br>Duplication artifacts occur when a part of an object is duplicated and placed adjacent to the original, creating anatomically or structurally implausible duplications (e.g., extra fingers, duplicate ears, duplicate wheels).<br><br><Detection Criteria><br><br>• Case A — Object present in target (Image 2): If the second image already contains the specified object, then the third image must show a plausible duplication or additional instance of the object. The new instance should be distinct from the original.<br>• Case B — No object in target (Image 2): If the second image does not contain the object, the third image must show a clearly new instance with a distinct boundary/contour. Reject if there is only subtle texture change, brightness shift, or local warping rather than a new instance.<br><br><br><Explanation Instructions><br><br>Focus on cues like duplicated parts, unnatural growths, or extra elements that conflict with normal anatomy or structure.|

|Omission|
|---|
|<Type Description><br><br>Omission artifacts occur when a part of an object is deleted and the area is inpainted with background, resulting in missing features or gaps where something should be present (e.g., missing fingers, absent ears).<br><br><Detection Criteria><br><br>• Definitive removal evidence : stump/ termination cues, disrupted silhouette, hollow/negative space, texture continuation/inpainting traces, mismatched shadows/reflections, or symmetry break.<br>• Ambiguity/occlusion rule: If the missing part could plausibly be merely hidden, set has_artifact = false.<br>• Anatomical plausibility check: If the scene still reads as anatomically correct and a typical pose could hide the part, set has_artifact = false.<br><br><br><Explanation Instructions><br><br>Focus on cues like missing structure, unnatural gaps, smoothed-over areas, or anatomical discontinuity where something appears to be absent.|

|Distortion|
|---|
|<Type Description><br><br>Distortion artifacts occur when parts are warped, creating unnatural geometry, irregular textures, or visual blending errors.<br><br><Explanation Instructions><br><br>Focus on cues like warped shapes, unnatural geometry, irregular textures, or visual blending errors that make the structure appear broken or malformed.|

|Fusion|
|---|
|<Type Description><br><br>Fusion artifacts occur when parts or distinct entities are unnaturally merged together, creating blurred boundaries, overlapped textures, or structural entanglement (e.g., two animals merged into one).<br><br><Detection Criteria><br><br>• Boundary visibility comparison: Compare Image 2 and Image 3. If a clear, continuous boundary between the two objects remains visible in Image 3 (similar to Image 2), set has_artifact = false.<br>• Fusion cues needed: boundary loss/softening across seams; cross-object texture/color blending; geometry interpenetration; inconsistent occlusion ordering.<br>• Not fusion: mere blur, minor warping, or lighting change that preserves recognizable boundary.<br><br><br><Explanation Instructions><br><br>Focus on cues like boundary loss/softening across seams, cross-object texture/color bleed, geometry interpenetration, and inconsistent occlusion ordering.|

Figure 14. Type descriptions and instructions by artifact type.

###### <User Prompt>

You are an expert at detecting [duplication, omission, fusion]– type artifacts in AI-generated images.

## Type Description You will be shown :

- 1. An original image without the target region
- 2. An original image with only the target region
- 3. An artifact image with only the target region
- 4. The object name that may be [added, removed, fused]

Your task is to determine if there is a successful [duplication, omission, fusion] artifact in the third image.

Detection criteria (be strict): ## Detection Criteria Return your analysis in the following format:

"has_artifact": true/false (whether the artifact is successfully present, always true for distortion) Important:

- Focus on visible evidence in the target region

<Example> Input: {images} {USER PROMPT[‘addition’]}

[Figure 152]

[Figure 153]

[Figure 154]

OUTPUT: “has_artifact”: false

Figure 15. Full prompt and example for data filtering.

###### <User Prompt>

You are an expert at describing [duplication, omission, distortion, fusion]–type artifacts in AI-generated images.

## Type Description You will be shown :

- 1. An original image without the target region
- 2. An original image with only the target region
- 3. An artifact image with only the target region
- 4. The object name that may be [added, removed, fused, distorted] Your task is to describe what looks wrong or unnatural in the target region. For explanation: ## Explanation Instructions

Return your analysis in the following format:

"explanation": "Detailed description of what looks wrong in the region, without reasoning about the artifact type or how the artifact is created."

"label": "Brief description of the artifact (empty string if no artifact)"

Important:

- - Focus on visible evidence in the target region
- - Use simple, clear language for explanations
- - Do not refer to images by number; say "the region" or "the area" instead
- - Make explanations understandable for non-experts

###### <Examples>

[Figure 155]

Input: {images} {USER PROMPT}

[Figure 156]

[Figure 157]

Output: “explanation”: “The elephant has an extra ear on its back, which is not naturally possible. Elephants typically have two ears on the sides of their head.”, “label”: “duplicated ear”

Figure 16. Full prompt and example for regional explanation generation.

###### <User Prompt>

You are an image artifact analyst. You will be given an image with artifacts and a list of injected artifact annotations, in the format of bbox:<(xmin, ymin, xmax, ymax)> description:<description of the artifact in that bbox region>.

Your job: Read ALL artifact annotations and write a single, holistic explanation the explanation of the anomalies in the image. Guidance:

- - Do NOT mention coordinates or the term "bbox". Use the annotations only to understand what is wrong.
- - Use commonsense knowledge about typical anatomy/structure (e.g., zebras normally have four legs).
- - If multiple issues appear, summarize the combined effect coherently rather than listing them mechanically.
- - Keep it concise (2-3 sentences) and easy to understand. Avoid making concluding sentences. Just focus on explaining the abnormality.

<Examples> Input: {images} {USER PROMPT}

[“bbox: [352, 240, 416, 336] description: "The elephant's trunk appears to blend unnaturally with the surrounding vegetation, creating a confusing visual where the trunk's shape is not clearly defined. The tusk also seems partially obscured by the blending, making it look irregular.", “bbox: [128, 160, 240, 288] description: 'The elephant has an extra ear on its back, which is not naturally possible. Elephants typically have two ears on the sides of their head.”]

[Figure 158]

[Figure 159]

Output: The image shows an elephant with an unusual anomaly where an extra ear is positioned on its back, which is not typical for elephants. Additionally, the trunk appears to blend with the surrounding vegetation, causing it to lose its distinct shape and making the tusk appear irregular.

Figure 17. Full prompt and example for global explanation generation.

Value Injection Steps

10 15 20 24

10 15 20 24

|[Figure 160]|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]<br><br>[Figure 165]|[Figure 166]<br><br>[Figure 167]|[Figure 168]<br><br>[Figure 169]|[Figure 170]<br><br>[Figure 171]|
|---|---|---|---|---|---|---|---|
|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]<br><br>[Figure 177]|[Figure 178]<br><br>[Figure 179]|[Figure 180]<br><br>[Figure 181]|[Figure 182]<br><br>[Figure 183]|
|[Figure 184]|[Figure 185]<br><br>|[Figure 186]|[Figure 187]|[Figure 188]|[Figure 189]|[Figure 190]|[Figure 191]|
|[Figure 192]|[Figure 193]|[Figure 194]|[Figure 195]|[Figure 196]|[Figure 197]|[Figure 198]|[Figure 199]|

10241520

PEInjectionSteps

Figure 18. Hyperparameter study on PE injection and value injection steps (1). The image in the red box shows our selected configuration.

#### B. Benchmarks and Evaluation Protocols (§ 5 & § 6.1)

###### B.1. Benchmark Datasets (§ 5)

Table 6 shows the fully cited artifact benchmark datasets and the sources used for generation. Details on the dataset’s metadata formats for artifact region representations and image explanations are elaborated below.

- • RichHF. The RichHF dataset, with 995 images sampled and annotated from the Pick-a-Pic dataset, provides the artifact map in a heatmap format, highlighting the probability of abnormal regions in the image. Annotations and scores are generated by the trained multi-modal transformer to predict human perception-aligned feedback.
- • LOKI. The LOKI benchmark is a multi-modal syntheticdata detection dataset, covering image, video, text, audio, and 3D content, and comprises roughly 18,000 curated questions across 26 subcategories. It includes coarse real vs. synthetic judgments, multiple-choice detection, anomaly / artifact region selection, and explanation tasks. We use the image artifact subset of the dataset, which consists of 229 images.
- • SynthScars. The SynthScars dataset consists of 12,236 fully synthetic images across four distinct content types (Human, Object, Scene, Animal) and three artifact categories (Physics, Distortion, Structure). Each image is annotated with pixel-level segmentation masks delimiting artifact regions, detailed textual explanations of the artifact(s), and artifact-category labels.

Visualizations. Visualizations of the evaluated benchmarks, including ArtiBench, are shown in Figure 19. It is clearly visible that even though our benchmark dataset shows overall better quality of image generation compared to the previous benchmarks, image artifacts are still visible in the generated images. We emphasize the importance of our timely benchmark that successfully represents structural artifacts remaining in the most recent diffusion models.

###### B.2. ArtiBench Generation (§ 5)

Image Generation. ArtiBench is built on a set of images generated by state-of-the-art diffusion models, reflecting the most timely artifacts appearing in current image generation models. We use five different models for image generation and three reliable prompt sources. After generating a set of images with the five models from randomly sampled prompts, the images were annotated according to the strictly guided annotation process.

Annotation Pipeline. With the set of diffusion-generated images, we construct ArtiBench following a guided 4-step annotation process: (1) classification, (2) bounding box labeling, (3) explanation generation, and (4) expert curation.

RichHF-18K LOKI SynthScars ArtiBench

|[Figure 200]|[Figure 201]|[Figure 202]|[Figure 203]|
|---|---|---|---|
|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|
|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|
|Figure 19. Artifa order of publishm|ct examples of ent.|the four benchm|arks, listed in|

F the o

- • Classification. Annotators are asked to meticulously observe the images and determine whether there are any artifacts visible in the image. The caption used for T2I generation is provided with the image for better understanding of generative intentions and respecting the model’s capability to follow instructions on generating abnormalities. All images classified as having artifacts proceed to the next step, where they are shuffled and redistributed to the annotators to mitigate human bias in the examination process.
- • Bounding Box Labeling. For the artifact-containing images from the classification step, annotators are expected to generate bounding boxes, identify the type of artifact, and write a short description about what abnormalities exist in the specified region.
- • Explanation Generation. The final annotations are used as the input of the VLM query to generate a comprehensive and polished explanation for the full image. With the prompt used for generation and the list of pairs of bounding box and captions as the input, the VLM is required to output a global explanation of the image regarding all artifacts mentioned in the captions.
- • Expert Curation. With the set of images and the metadata completed, we curate the images to build the final version of our 1K benchmark. Images are carefully selected to ensure a balanced set that provides a clear and straightforward view of plausible artifacts.

Table 6. Comparison of artifact benchmark datasets, their generative sources, and evaluation tasks. Highlighted entries denote dataset sources that were reused by subsequent benchmarks.

Benchmark Generative Sources Sample Bin. Loc. Exp. RichHF [27] SD2.1 [42], SDXL [38], Dreamlike Photoreal [10] 955 ✓ LOKI [51] FLUX [5], SD1.4–2.1 [42], Midjourney [30], StyleGAN [22], pix2pix [18], CUT [36] 229 ✓ ✓ SynthScars [20] RichHF [27], Chameleon [50], Midjourney [30], DALL·E 3 [32], SD1.x [42] 1K ✓ ✓ ArtiBench SD3.5 [11], FLUX [5], Qwen-Image [49], Nano-Banana [13] 1K ✓ ✓ ✓

###### B.3. Evaluation Protocol (§ 6.1)

###### B.3.1. Binary Classification

- • Accuracy. The rate of true predictions is measured to show basic performance of accurate classification.
- • Macro F1. In addition, we use the macro F1 score, to achieve fair comparison across unbalanced datasets and biased predictions. F1 scores are relatively calculated for positive and negative samples to be averaged, exposing the model’s capability to precisely capture both artifact presence and absence without random guessing.

###### B.3.2. Localization

We alleviate the unfairness between divergent representations of artifact regions, including bounding boxes, polygonal segmentation maps, and heatmaps, by mapping all representations to pixel-wise binary maps.

- • IoU. With the binary maps obtained, we calculate the pixel-wise IoU between the foreground region predictions and ground-truth areas. IoU scores prefer tight and highly overlapping predictions, providing intuitions on precise predictions but highly possible of penalizing bounding box representations for over-prediction.
- • F1. Pixel-wise predictions are measured to capture true predictions on a fine-grained basis. Unlike the macro F1 score used in binary detection tasks, F1 scores are measured only for positive ground truth regions to prioritize true detections. With each pixel prediction classified, we use the pixel count for true positive (TP), false positive (FP), and false negative (FN) predictions to use them for calculating the F1 score. This method lessens the penalty on loose region representations of bounding boxes over segmentation maps or heatmaps.

###### B.3.3. Explanation

- • ROUGE-L. The ROUGE-L score shows the proportion of the longest overlapping phrase among the full data. This captures the words or phrases that focus on specific artifact regions and objects, with higher scores showing that the model better understands the visual artifact.
- • CSS. Cosine similarity was measured on sentence embeddings generated by sentence-transformers [40]. CSS portrays the general similarity in context between the descriptions regarding the full scenery’s plausibility.

###### B.3.4. Evaluation Prompt of VLMs

Figure 20 shows the specific prompts used for the evaluation on VLMs. Artifact priors are shared for all tasks, providing understanding of the structural artifacts we are focusing on.

###### <Artifact Prior>

Image artifacts refer to unintended, implausible, or visibly corrupted regions within images generated by diffusion models. These artifacts often break the natural semantics or visual coherence of an image, such as a person with extra fingers, a car with warped wheels, or missing parts of animals, and can significantly degrade image quality or realism. Artifacts are a critical concern in both model evaluation and training.

There are four types of image artifacts: Duplication, Omission, Distortion, and Fusion.

- 1. Duplication: These artifacts appear as excessive or repeated components, leading to unrealistic objects or implausible duplication.
- 2. Omission: These artifacts appear as omitted components, leading to unrealistic or incomplete objects.
- 3. Distortion: These artifacts occur when objects have details that are not typical, making the object unrecognizable or visually broken, such as geometric inconsistencies, abnormal textures, unnatural asymmetry or twisted, warped, scrambled parts.
- 4. Fusion: These artifacts result from the combination of objects so their boundaries or interiors merge into one ambiguous, hybrid part, resulting in a visually incoherent region.

###### <User Prompt – Binary Classification>

{ARTIFACT PRIOR} Your task is to identify if there are any artifacts within this image. Are there any artifacts in this image? Use lowercase "true" or "false" (boolean values).

###### <User Prompt – Localization>

{ARTIFACT PRIOR} Identify and localize artifact regions in this image. For each artifact found, provide bounding box coordinates [x_min, y_min, x_max, y_max] where coordinates are in pixels. Do NOT output multiple bboxes that indicate the same region

###### <User Prompt – Explanation>

{ARTIFACT PRIOR} Analyze the artifacts visible in this image and provide a comprehensive explanation. If no artifacts are found, state "There are no artifacts in this image." If the image is inaccessible, state: "Image not available for analysis."

Figure 20. Evaluation prompts for VLMs.

###### B.4. VQA Dataset Structure (§ 6.1)

This section presents the structure of our multi-turn visual question answering (VQA) dataset, which provides the supervision used to train VLMs in artifact detection, localization, and explanation. Each data instance is derived from a paired clean–artifact image generated by ArtiAgent with the synthesized annotations as in Figure 11, enabling the construction of tightly aligned conversations that elicit the model’s ability to identify normal content and reason about artifact regions.

###### B.4.1. VQA Template

Each ArtiAgent instance yields two conversations: one for the clean reconstruction image and one for the corresponding artifact image. Tables 7 and 8 summarize the questionanswer templates implemented across these two settings.

Task Template Bin.

Q: Does this image contain any visual artifacts?

A: No.

- Loc. 1

Q: Locate the 〈entity〉’s 〈subentity〉.

A: 〈bbox〉

- Loc. 2

Q: Is there a 〈entity〉’s 〈subentity〉 in 〈bbox〉?

A: Yes / No Exp.

Q: Describe the clean image.

A: 〈image caption〉 Table 7. VQA templates for clean images.

Task Template Bin.

Q: Does this image contain any visual artifacts?

A: Yes.

- Loc. 1

Q: Provide bounding boxes for all artifact regions.

A: [〈bbox〉]

- Loc. 2

Q: Explain why region 〈bbox〉 is an artifact.

A: 〈label〉 Exp.

Q: Describe all artifacts in the image.

A: 〈explanation〉 Table 8. VQA templates for artifact-injected images.

###### B.4.2. VLM Training Configuration

We train the VLM using a two-stage supervised fine-tuning setup. We use Qwen2.5-VL-7B-Instruct [4] and Intern3.5VL-8B [57] as the backbone, and use the identical configurations. In the first stage, we fine-tune only the language model and multi-modal projector while keeping the vision encoder frozen, using a learning rate of 1×10−5, batch size 64, cosine decay scheduling, and one training epoch. In the second stage, we unfreeze the vision encoder and continue

training with a smaller learning rate of 1×10−6, with 200 steps. Both stages use the same VQA dataset.

#### C. Mitigating Artifacts in Diffusion Models (§ 6.2.1 & § 6.2.2)

###### C.1. Reward-guided generation (§ 6.2.1)

Verifier Training. We train an artifact verifier modeled with Bradley-Terry model [6] to score images by how likely they are to be artifact-free. The model is trained to assign higher scores to the clean image and lower scores to the artifact-injected image. The verifier uses a frozen ViT-B/16 encoder with a lightweight MLP head, and is optimized using AdamW (learning rate 1×10−3, weight decay 1×10−4, L2 regularization 1 × 10−4). Training uses a batch size of 32 for 5 epochs. Each batch additionally includes two crossscene negative pairs to encourage content-agnostic ranking.

Test-Time Scaling. At inference time, we use the verifier as a reward model in a compute-scaled best-of-N sampling procedure [28]. For each prompt, round r samples 2r independent latent noises, generates all corresponding images, and evaluates them with the verifier. The highest-scoring image is retained, and the search space doubles in the next round. This random-search strategy expands the candidate pool exponentially, enabling the diffusion model to reliably discover images with fewer artifacts without modifying the weights of the model.

###### C.2. Image Correction (§ 6.2.2)

In our artifact-correction pipeline, all VLM interactions are carried out using the Qwen2.5-VL-7B model fine-tuned with ArtiAgent supervision. This unified VLM is responsible for localizing artifact regions, generating artifact-free captions, and verifying whether the corrected content remains flawed. We describe each of the three prompts used in the loop below. The actual prompt is shown in Table 9.

Prompt Instruction Text Localization “Provide the bounding box for the artifact

region. Format as [x min, y min, x max, y max]. Only output the bounding box.”

Captioning “Describe what the image would look like if it had no artifacts. Provide a short caption of the clean scene.”

Verifying “Is there an artifact within 〈B〉? Answer

Yes or No.” Table 9. Prompts used by the VLM in the artifact-correction loop.

Localization prompt plocalize. This prompt instructs the VLM to identify the spatial extent of the artifact within the input image. The model is asked to produce a single

- Algorithm 6 VLM-Guided Artifact Correction Loop Require: Image I containing at least one artifact

- 1: /* localize artifact region */
- 2: B ← VLM(I,plocalize)
- 3: /* generate caption of the image */
- 4: c ← VLM(I,pcaption)
- 5: while true do
- 6: /* Inpaint region B using caption c */
- 7: I ← Inpaint(I,B,c)
- 8: /* check artifact existence inside B */
- 9: r ← VLM(I,pverify)
- 10: if not r then
- 11: return I

bounding box formatted as [x min, y min, x max, y max] without additional commentary. This strict output format ensures deterministic parsing. The resulting bounding box B is computed once at the beginning of the procedure and kept fixed across all subsequent iterations, providing spatial stability for the iterative refinement process.

Captioning prompt pcaption. To guide inpainting, the VLM is prompted to describe the overall image without specifying the artifacts in the image. Rather than summarizing the corrupted content, the model is explicitly instructed to generate a short caption that reflects the intended clean version of the scene. This caption serves as the semantic conditioning signal for the FLUX inpainting pipeline [3], which synthesizes corrected content inside the region B.

Verifying prompt pverify. After each inpainting step, the VLM is queried to assess whether the corrected region B still contains an artifact. The prompt restricts the judgment to the localized region, returning a binary response (Yes or No). This local verification prevents the algorithm from drifting to unrelated image regions and directly determines whether the loop terminates or continues.

Together, these three prompts coordinate the interaction between the ArtiAgent-trained VLM and the FLUX inpainting pipeline: the VLM identifies the corrupted area, provides semantic guidance for correction, and validates the result, while the inpainting model performs the pixel-level repair. This iterative coupling enables consistent and stable reduction of visual artifacts while preserving overall image semantics. The whole procedure is shown in Algorithm 6.

