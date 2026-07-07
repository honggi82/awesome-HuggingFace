### PixelSmile: Toward Fine-Grained Facial Expression Editing

[Figure 1]

# arXiv:2603.25728v1[cs.CV]26Mar2026

Jiabin Hua1,2,∗ Hengyuan Xu1,2,∗ Aojie Li2,† Wei Cheng2 Gang Yu2,‡ Xingjun Ma1,‡ Yu-Gang Jiang1

1 Fudan University 2 StepFun

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Project Page Code Model Benchmark Demo

[Figure 7]

[Figure 8]

Ⅰ. Fine-Grained Expression Editing Ⅱ. Extended Expression

|[Figure 9]|
|---|

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|
|---|---|---|---|---|---|
|[Figure 18]|[Figure 19]|[Figure 20]|[Figure 21]|[Figure 22]|[Figure 23]|
|[Figure 24]|[Figure 25]|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|
|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|

Contempt

Confused

Anxious

HappySadDisgustAngryFearSurprised

[Figure 36]

[Figure 37]

[Figure 38]

###### RealHumanAnime

|[Figure 39]|
|---|

|[Figure 40]|
|---|

|[Figure 41]|
|---|

confident

Sleepy

Shy

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Happy

Ⅲ. Expression Blending

[Figure 46]

|[Figure 47]|[Figure 48]|[Figure 49]|[Figure 50]|[Figure 51]|[Figure 52]|
|---|---|---|---|---|---|
|[Figure 53]|[Figure 54]|[Figure 55]|[Figure 56]|[Figure 57]|[Figure 58]|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Figure 1. Overview of PixelSmile. It enables 1) continuous and precise control of facial expression intensity across real-world and anime domains, 2) editing across 12 distinct expression categories, and 3) seamless blending of multiple expressions.

#### Abstract

Fine-grained facial expression editing has long been limited by intrinsic semantic overlap. To address this, we construct the Flex Facial Expression (FFE) dataset with continuous affective annotations and establish FFE-Bench to evaluate structural confusion, editing accuracy, linear controllability, and the trade-off between expression editing and identity preservation. We propose PixelSmile, a diffusion

∗ Equal contribution. † Project lead. ‡ Corresponding authors.

framework that disentangles expression semantics via fully symmetric joint training. PixelSmile combines intensity supervision with contrastive learning to produce stronger and more distinguishable expressions, achieving precise and stable linear expression control through textual latent interpolation. Extensive experiments demonstrate that PixelSmile achieves superior disentanglement and robust identity preservation, confirming its effectiveness for continuous, controllable, and fine-grained expression editing, while naturally supporting smooth expression blending.

#### 1. Introduction

Recent advances in diffusion-based image editing models [46, 73] and identity-consistent generation techniques [27, 34, 74] have significantly improved the ability to manipulate personal portraits using natural language. Despite this progress, fine-grained facial expression editing remains a challenging problem. Current models can generate clearly distinct expressions, such as happy versus sad, but struggle to delineate highly correlated, semantically overlapping expression pairs, such as fear versus surprise or anger versus disgust. Most existing methods rely on discrete expression categories, forcing inherently continuous human expressions into rigid class boundaries. As a result, these formulations fail to capture subtle expression boundaries, leading to structured cross-category confusion, limited control over expression intensity, and degraded identity consistency during editing.

To better understand this limitation, we analyze the semantic structure of facial expressions. As illustrated in Fig. 2, facial expressions lie on a continuous semantic manifold where semantically adjacent emotions naturally overlap. This overlap manifests as systematic confusion across multiple stakeholders: human annotators, classifiers, and generative models often fail to uniquely distinguish semantically adjacent expressions like fear versus surprise or anger versus disgust. When generative models are trained using discrete and potentially conflicting labels from such ambiguous samples, they are forced to learn entangled representations in the latent space. Consequently, this structural entanglement prevents precise control, resulting in unintended expression leakage, where editing one emotion inadvertently triggers the characteristics of another or even degrades identity consistency.

Addressing this challenge requires a new supervision paradigm for facial expression editing models. Conventional datasets often represent facial expressions using rigid one-hot labels, which fail to capture the nuanced structure of human affect and propagate semantic entanglement into the generative pipeline. To address this limitation, we introduce a new supervision paradigm based on continuous affective annotations. Specifically, we construct the Flex Facial Expression (FFE) dataset, which replaces discrete labels with continuous 12-dimensional affective score distributions. Based on this dataset, we further establish FFEBench to evaluate structural confusion, editing accuracy, linear controllability, and the trade-off between expression editing and identity preservation. By providing diverse expressions within the same identity and continuous affective ground truth across both real and anime domains, FFE breaks the one-hot supervision bottleneck, allowing models to learn the fine-grained boundaries of the expression manifold rather than disjoint categories, and enabling systematic evaluation of controllable expression editing.

Building upon this data-centric foundation, we propose PixelSmile, a diffusion-based editing framework that disentangles expression semantics. Our framework introduces a fully symmetric joint training paradigm to contrast confusing expression pairs identified in our analysis. Combined with a flow-matching-based textual latent interpolation mechanism, PixelSmile enables precise and linearly controllable expression intensity at inference time without requiring reference images. Through the synergy between continuous affective supervision and symmetric learning, PixelSmile achieves robust and controllable editing while preserving identity fidelity.

In summary, our contributions are threefold:

- • Systematic Analysis of Semantic Overlap. We reveal and formalize the structured semantic overlap between facial expressions, demonstrating that structured semantic overlap, rather than purely classification error, is a primary cause of failures in both recognition and generative editing tasks.
- • Dataset and Benchmark. We construct the FFE dataset—a large-scale, cross-domain collection featuring 12 expression categories with continuous affective annotations—and establish FFE-Bench, a multi-dimensional evaluation environment specifically designed to evaluate structural confusion, expression editing accuracy, linear controllability, and the trade-off between expression editing and identity preservation.
- • PixelSmile Framework. We propose a novel diffusionbased framework utilizing fully symmetric joint training and textual latent interpolation. This design effectively disentangles overlapping emotions and enables disentangled and linearly controllable expression editing.

#### 2. Related Work

Facial Expression Editing. Facial expression editing aims to modify facial expressions while preserving identity. Early approaches relied on conditional GANs [24], formulating the task as multi-domain image-to-image translation [10, 11, 15, 45, 59]. Subsequent works explored disentangled latent manipulation within StyleGAN-based architectures [28, 36, 37, 64, 65, 80] to identify semantic directions for continuous expression control. Another line of research incorporates explicit facial priors, such as Action Units or 3DMM parameters, to enable structured, interpretable manipulation. For instance, MagicFace [71] leverages such priors to guide diffusion models, while other works [13, 16, 22, 33, 59] explore similar structural constraints. Despite facilitating discrete expression transfers, these methods often struggle with fine-grained control, identity consistency, and generalization. More recently, diffusion models [30] have significantly advanced image generation and editing quality [4, 29, 49, 82]. Furthermore, large-scale multimodal pretraining has fueled signifi-

Key Observation

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

###### Surprised Fear Angry Disgust

[Figure 68]

[Figure 69]

Surprised!

Shared: Wide Eyes, Open Mouth Shared: Frown, Negative Affect

[Figure 70]

Human

Angry Disgust

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Fear: 0.95

Model

[Figure 77]

[Figure 78]

[Figure 79]

Recognition Confusion Editing Confusion Expression Semantic Overlap

Our Contribution

[Figure 80]

Real Human Samples

Fear Surprised Angry Disgust

Data Structure

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

{

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

“id”: ‘001136’, “category”: ‘angry’, “scores”: {

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

“happy”: 0.85, “angry”: 0.0, “sad”: 0.1, “surprised”: 0.02, “fear”: 0.0, “disgust”: 0.0, “anxious”: 0.01, “confident”: 0.55, “contempt”: 0.01, “confused”: 0.0, “shy”: 0.02, “sleepy”: 0.0,

Anime Samples

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

|[Figure 97]|[Figure 98]|[Figure 99]|[Figure 100]|[Figure 101]|[Figure 102]|
|---|---|---|---|---|---|

Angry

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

}, “box”:[53.21,…]

[Figure 107]

}

[Figure 108]

[Figure 109]

[Figure 110]

PixelSmile

FFE Dataset

- Figure 2. Observation of Expression Semantic Overlap. Inherent expression overlap causes systematic confusion across human annotators, recognition models, and generative models (top). We resolve this via the FFE dataset (bottom left) and PixelSmile framework (bottom right), utilizing continuous supervision and symmetric training for disentangled editing.

cant advancements in general-purpose editing. Large-scale foundation models, such as GPT-Image [54], Nano Banana Pro [25], Qwen-Image [73], and LongCat-Image [68], now demonstrate remarkable zero-shot flexibility and editing capabilities [5, 39, 46].

Continuously Controlled Generation. Prior works achieve continuous editing by leveraging interpolatable subspaces within generative models. ConceptSlider [20] interpolates LoRA weights, while subsequent methods [3, 7, 12, 21, 23, 26, 29, 32, 35, 63, 67, 77, 85] manipulate text embeddings or modulation features to achieve gradual semantic variation. More recently, SliderEdit [81], Kontinuous-Kontext [57], and concurrent works [72, 75, 79] extend continuous control to editing models built upon FLUX.1 Kontext [40]. Despite smoother transitions via reduced strength or pixel interpolation, these methods remain constrained by entangled latent spaces, leading to semantic ambiguity and identity drift at large magnitudes. By disentangling latent expression semantics, our structured formulation achieves fine-grained linear control and identity preservation across diverse manipulation strengths.

Facial Expression Datasets and Benchmarks. Highquality datasets and reliable benchmarks are essential for facial expression analysis. Early controlled datasets [41, 47, 48, 78] provide same-identity multi-expression samples for precise comparison but lack diversity, while large-scale inthe-wild datasets [2, 42, 50, 76, 83] enhance generalization but lack paired expressions for the same identity, hindering identity-expression disentanglement in generative edit-

ing. Recent efforts extend to video and multimodal settings. While video-based datasets [51, 60, 84] focus on temporal or cross-modal dynamics, the MEAD dataset [69] provides expressions with three distinct intensity levels, moving beyond purely categorical labels but still falling short of finegrained, continuous control and structured disentanglement in static editing contexts. Alongside these, benchmarks such as F-Bench [44] and SEED [87] evaluate facial generation using visual metrics and human preference. However, standard metrics (e.g., CLIP, SSIM, LPIPS) capture overall quality but offer limited insight into disentanglement and continuous control. To address these gaps, we propose FFE and FFE-Bench. By providing same-identity pairs with continuous affective annotations, our approach enables rigorous evaluation of fine-grained, linearly controllable, and disentangled expression editing.

#### 3. Dataset and Benchmark

To facilitate fine-grained and linearly controllable facial expression editing, we construct the FFE dataset and establish FFE-Bench, a dedicated evaluation benchmark. Existing datasets often lack same-identity expression diversity or provide only discrete expression labels, which limits the evaluation of controllable expression manipulation. Our dataset addresses these limitations by providing large-scale same-identity expression variations with continuous affective annotations, enabling systematic analysis of expression disentanglement and editing controllability.

[Figure 111]

[Figure 112]

###### Inference Stage

###### Training Stage

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|1. He’s Surprised(0.8)|[Figure 115]|
|---|---|
|2. He is in Fear(0.6)| |

###### Target Prompt

###### Neutral Prompt

Source GT Surprise(0.8)

GT Fear(0.6)

[Figure 116]

Text Encoder

Text Encoder

###### PixelSmile

[Figure 117]

[Figure 118]

Contrastive Loss

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Gen(Surprise)

Gen(Fear)

Interpolated Embedding

Encoder

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Text

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

𝑇𝑇𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖𝑖 = 𝑇𝑇𝑖𝑖𝑖𝑖𝑛𝑛𝑖𝑖 + 𝛼𝛼 ⋅ (𝑇𝑇target − 𝑇𝑇𝑖𝑖𝑖𝑖𝑛𝑛𝑖𝑖)

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

| | | |
|---|---|---|
| | | |

GT(Fear)

GT(Fear)

[Figure 136]

[Figure 137]

|[Figure 138]|
|---|

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

GT(Surprise)

GT(Surprise)

###### DiT Blocks

[Figure 144]

PixelSmile

VAE

- 1
- 2

𝑻𝑻 𝑮𝑮fear,𝑷𝑷fear, 𝑵𝑵surp + 𝑻𝑻 𝑮𝑮surp, 𝑷𝑷surp, 𝑵𝑵fear

Source Image

LoRA

|[Figure 145]|[Figure 146]|[Figure 147]|[Figure 148]|[Figure 149]|[Figure 150]|
|---|---|---|---|---|---|

[Figure 151]

[Figure 152]

ArcFace

|[Figure 153]|
|---|

ID Loss

[Figure 154]

|[Figure 155]<br><br>[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]|
|---|

1 − cos ,

[Figure 159]

[Figure 160]

Generated Fear

Generated Surprise

[Figure 161]

Flow Matching Loss

Neutral Angrier

- Figure 3. Framework Overview. (1) Inference Stage. We interpolate between the neutral and target expression embeddings in textual latent space using a controllable coefficient α, enabling continuous adjustment of expression intensity. (2) Training Stage. We adopt a joint fully symmetric training framework. Specifically, we sample a source image Psrc and a confusing expression pair (Pa, Pb) to construct a triplet. We first treat Pa as the positive and Pb as the negative to compute a joint loss, and then swap their roles to compute it again, yielding a symmetric training objective. The joint loss consists of three components: a Flow-Matching loss for intensity alignment, a contrastive loss for expression separation, and an identity preservation loss to maintain subject consistency.

##### 3.1. The FFE Dataset

FFE is constructed through a four-stage collect–compose– generate–annotate pipeline designed to ensure expression diversity, cross-domain coverage, and reliable annotations. The final dataset contains 60,000 images across real and anime domains, supporting both photorealistic and stylized facial expression editing.

Base Identity Collection. We first curate a set of highquality base identities from two domains: (1) Real domain: approximately 6,000 real-world portraits are collected from public portrait datasets [1, 66], covering diverse demographics and scene compositions, including both close-up and full-body images; (2) Anime domain: to enable cross-domain evaluation, we collect stylized portraits from 207 anime productions covering 629 characters, from which around 6,000 high-quality images are retained after quality filtering and automated face detection. For both domains, automated face detection followed by manual verification is applied to ensure identity clarity and image quality. These images form the identity backbone of FFE dataset.

Expression Prompt Composition. To obtain fine-grained expression variations, we construct a structured prompt library for 12 target expressions. The taxonomy consists of six basic emotions [19] and six extended emotions (Confused, Contempt, Confident, Shy, Sleepy, Anxious). Rather than relying solely on abstract expression labels, each expression is decomposed into facial attribute components (e.g., mouth shape, eyebrow movement, and eye openness). Candidate attribute combinations are automatically gener-

ated and filtered with a vision-language model to remove anatomically inconsistent or semantically conflicting descriptions, resulting in a validated library of fine-grained expression prompts.

Controlled Expression Generation. For each base identity, multiple target expressions with varying intensities are synthesized using a state-of-the-art image editing model, Nano Banana Pro. We adopt a dual-part prompt design that specifies both the global expression category and localized facial attributes, improving controllability and reducing ambiguity between semantically similar expressions. This process produces approximately 60,000 images in total (30,000 per domain), providing rich identity-preserving expression variations across diverse conditions.

Continuous Annotation and Quality Filtering. Departing from conventional one-hot expression labels, each image is annotated with a 12-dimensional continuous score vector v ∈ [0,1]12. The scores are predicted by a vision-language model, Gemini 3 Pro, which estimates the intensity of each expression category. A subset of samples is verified by human annotators to ensure reliability. This representation captures semantic overlap between facial expressions (e.g., fear and surprise), providing a faithful approximation of the affective manifold. We further perform consistency checks and manual spot verification to remove ambiguous or lowconfidence samples. The resulting dataset provides sameidentity expression variations with continuous soft labels, enabling fine-grained evaluation of expression disentanglement and controllable facial expression editing.

##### 3.2. The FFE-Bench Benchmark

Motivated by the intrinsic semantic entanglement among facial expressions, which leads to structured cross-category confusion, we design a unified benchmark to evaluate facial expression editing from four complementary aspects: structural confusion, the trade-off between expression editing and identity preservation, control linearity, and expression editing accuracy. All expression classifications and intensity scores are predicted by Gemini 3 Pro.

Mean Structural Confusion Rate (mSCR). To quantify structured confusion between semantically similar expressions, we define the directed confusion rate Ci→j and the bidirectional confusion rate (BCR) as follows:

Ci→j =

BCR(i,j) =

Ni

1 Ni

1(ˆyk(i) = j), (1)

k=1

- 1

- 2

(Ci→j + Cj→i), (2)

where Ni denotes the number of samples edited toward class i, and yˆk(i) is the predicted dominant expression. The mSCR is computed by averaging BCR(i,j) over predefined confusing pairs (e.g., Fear–Surprise and Angry–Disgust). A lower mSCR indicates reduced cross-category confusion and improved semantic disentanglement.

Harmonic Editing Score (HES). Facial expression editing requires both accurate expression transfer and identity preservation. We define the Harmonic Editing Score as

2 × SE × SID SE + SID

, (3)

HES =

where SE denotes the VLM-based target expression score, and SID is the cosine similarity between source and edited faces. Identity similarity is computed as the average cosine similarity from three face recognition models (including ArcFace [14], AdaFace [38], FaceNet [62]) for robustness. High HES is achieved only when both expression strength and identity fidelity are preserved.

Control Linearity Score (CLS). To evaluate continuous controllability, we feed uniformly spaced intensity coefficients α ∈ [0,αmax] during inference and compute the Pearson correlation between α and the VLM-predicted intensity scores. Higher CLS indicates more linear and predictable expression control.

Expression Editing Accuracy (Acc). We report the proportion of generated images whose predicted dominant expression matches the target instruction. This metric measures overall categorical editing success.

#### 4. Method

We present PixelSmile, a framework for fine-grained facial expression editing. As illustrated in Fig. 3, our method

Ours (Full)

0.0

0.0

KSlider

0.9

SliderEdit

Ours(w/o CL)

0.8

0.0

Narrower ID Impairment

3.0

IDSim

0.7

0.0

3.0

0.6

3.0

Wider Expression Range

0.5

3.0

0.4

0.1 0.2 0.3 0.4 0.5 0.6 0.7

Expression Score

Figure 4. Quantitative Evaluation of Linear Control Methods. Comparison of the trade-off between ID similarity and expression score across different models. PixelSmile achieves an optimal balance, providing a wider expression manipulation range while preserving identity fidelity.

builds upon a pretrained Multi-Modal Diffusion Transformer (MMDiT) [58] with LoRA adaptation [31]. To address intrinsic semantic entanglement and enable continuous intensity control, we introduce two key components: (1) a Flow-Matching-based textual interpolation mechanism [43] for smooth expression strength control; and (2) a Fully Symmetric Joint Training framework with a symmetric contrastive objective to reduce cross-category confusion while preserving identity and background consistency.

##### 4.1. Textual Latent Interpolation for Continuous Editing

Existing expression editing approaches typically rely on discrete labels or coarse reference signals [73], which limits fine-grained control over expression intensity. Instead, we perform linear interpolation in the textual latent space to enable continuous and smooth expression manipulation.

Textual Latent Interpolation. Given a neutral prompt Pneu and a target expression prompt Ptgt, the frozen MMDiT text encoder maps them to embeddings eneu and etgt, respectively. We define the residual direction

∆e = etgt − eneu, (4)

which captures the semantic shift from neutral to the target expression.

A continuous conditioning embedding is then constructed as

econd(α) = eneu + α · ∆e, α ∈ [0,1]. (5)

When α = 0, the conditioning corresponds to neutral expression; when α = 1, it recovers the full target expression. Intermediate values of α yield smoothly varying expression intensities. Importantly, the same direction also supports

Table 1. Quantitative Evaluation of General Editing Models. Best, second best, and third best results are indicated by , , and respectively.

Method mSCR ↓ Acc-6 ↑ Acc-12 ↑ ID Sim ↑

Seedream [5] 0.3725 0.5294 0.3737 0.7221 Nano Banana Pro [25] 0.1754 0.8431 0.6200 0.7107 GPT-Image [54] 0.1107 0.8039 0.6300 0.5056

Closed

FLUX-Klein [39] 0.2850 0.4510 0.3310 0.4146 LongCat [68] 0.1754 0.6275 0.4100 0.6036 Qwen-Edit [73] 0.2625 0.4510 0.2900 0.6938 Ours(w/o training) 0.2400 0.5294 0.3500 0.6769 Ours 0.0550 0.8627 0.6000 0.6522

Open

extrapolation: at inference time, α > 1 enables stronger expression transfer while maintaining structural consistency.

Score-Supervised Flow Matching. To enforce consistency between textual interpolation and visual intensity, we introduce score supervision during Flow Matching (FM) training. Each training image is associated with a ground-truth intensity coefficient αgt ∈ [0,1], derived from the continuous expression annotations. During LoRA fine-tuning, we set α = αgt and use econd(α) as the conditioning input to the dual-stream attention blocks. The score-supervised velocity loss is defined as

0,x1 vθ(xt,t,econd(α))−(x1−x0) 22 , (6)

LeditFM = Et,x

where x0 denotes the source image latent and x1 denotes the edited target latent. This objective explicitly couples the interpolation coefficient with the corresponding visual transformation. At inference, continuous control is achieved by varying α, without requiring reference images.

##### 4.2. Fully Symmetric Joint Training for Disentanglement

As stated in Sec. 1 and illustrated in Fig. 2, facial expressions lie on a continuous and highly overlapping semantic manifold. For example, Surprise and Fear share similar arousal and facial cues, leading to structural confusion near class boundaries when trained with discrete supervision only. Inspired by contrastive learning and the idea of symmetric learning [70], we introduce a Fully Symmetric Joint Training framework with a symmetric contrastive objective in the feature space.

Symmetric Construction. Given a pair of semantically overlapping expressions, (Ea,Eb), defined based on the confusion patterns observed in the FFE dataset, and an input image, the model performs two parallel generations, Ga and Gb, conditioned on prompts corresponding to Ea and Eb, respectively. For Ga, the ground-truth image with expression Ea, denoted as Pa, serves as the positive, while the image with expression Eb, denoted as Pb, is treated as a hard negative; the roles are reversed for Gb. This symmetric design avoids directional bias and enforces consistent separation between confusing expressions.

Table 2. Quantitative Evaluation of Linear Control Models. Best, second best, and third best results are indicated by , , and respectively.

Method CLS-6 ↑ CLS-12 ↑ ID Sim ↑ HES↑ SAEdit [35] -0.0183 0.0007 - ConceptSlider [20] 0.3161∗ - 0.6250 0.3656 AttributeControl [3] 0.2856∗ - 0.3609 0.2712 K-Slider [57] -0.0459 -0.0634 0.7974 0.3272 SliderEdit [81] 0.5599 0.5217 0.7414 0.3441 Ours(w/o training) 0.6892 0.5217 0.6769 0.4086 Ours 0.8078 0.7305 0.6522 0.4723

∗ Evaluated on CLS-2 (happy, surprised).

Symmetric Contrastive Loss. All images are encoded using a frozen CLIP image encoder to capture expression semantics. The symmetric loss is defined as

LSC =

- 1

- 2

[T (Ga,Pa,Pb) + T (Gb,Pb,Pa)], (7)

where T pulls the generated sample toward its target while pushing it away from the confusing expression.

We investigate three realizations of T , including hingebased [62], log-ratio [52], and InfoNCE-style [53] formulations. In practice, we primarily adopt the InfoNCE-style objective due to its stable optimization. Detailed formulations and ablations are provided in the Appendix A.

##### 4.3. Identity Preservation

Strong intensity extrapolation (α > 1) or contrastive forces may degrade identity consistency. To stabilize biometric features, we introduce an identity preservation loss based on a pretrained face recognition model. Specifically, we adopt ArcFace [14] as a frozen identity encoder Φarc(·). For generated images Ga,Gb and their corresponding ground truths Pa,Pb, the identity loss is defined as

- 1

- 2 i∈{a,b}

[1 − cos(Φarc(Gi),Φarc(Pi))], (8)

LID =

This term enforces identity consistency while allowing expression variation.

##### 4.4. Overall Training Objective

We fine-tune the LoRA parameters of the frozen MMDiT under a symmetric dual-branch training scheme, where a pair of confusing expressions (a,b) is optimized jointly for the same subject. The overall objective is defined as

- 1

- 2 LaFM + LbFM + λscLSC + λidLID, (9)

Ltotal =

where λsc and λid control the trade-off between disentanglement and identity preservation. This symmetric formulation jointly enforces continuous intensity control, expression separation, and identity consistency.

[Figure 162]

Nano Banana Pro

GPT-Image 1.5

Seedream4.5

LongCatImage-Edit

Qwen-ImageEdit-2511

FLUX.2 Klein

Ours

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

AngryDisgustAngryDisgustFearSurprisedFearSurprised

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

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

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

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

- Figure 5. Qualitative Comparison with General Editing Models. PixelSmile produces clearer expression changes while preserving facial identity, whereas existing editing models either weaken expression editing or degrade identity consistency.

#### 5. Experiment

##### 5.1. Experimental Setup

We implement PixelSmile based on Qwen-Image-Edit2511. To handle the distinct stylistic distributions of realworld and anime domains, we train two independent LoRA adapters for each. Following prior work [6, 88], for contrastive supervision, we adopt CLIP-ViT-L/14 [61] for the real domain and DanbooruCLIP [55] for anime. Identity

preservation is enforced using a pretrained ArcFace (antelopev2) model for the real domain. Additional implementation details are provided in Appendix B.

Baselines. To ensure a comprehensive and fair evaluation, we divide baselines into two groups according to their primary strengths in facial expression editing: general editing models, which are strong in overall expression editing quality, and linear control models, which are designed for continuous and predictable intensity control.

[Figure 225]

|Concept Slider|[Figure 226]|[Figure 227]|[Figure 228]|[Figure 229]|[Figure 230]|[Figure 231]|
|---|---|---|---|---|---|---|
|Attribute Control<br><br>|[Figure 232]|[Figure 233]|[Figure 234]|[Figure 235]|[Figure 236]|[Figure 237]|
| |[Figure 238]|[Figure 239]|[Figure 240]|[Figure 241]|[Figure 242]|[Figure 243]|
|SliderEdit<br><br>K-Slider<br><br>Ours| | | | | | |
| |[Figure 244]|[Figure 245]|[Figure 246]|[Figure 247]|[Figure 248]|[Figure 249]|
| |[Figure 250]|[Figure 251]|[Figure 252]|[Figure 253]|[Figure 254]|[Figure 255]|

|Concept Slider|[Figure 256]|[Figure 257]|[Figure 258]|[Figure 259]|[Figure 260]|[Figure 261]|
|---|---|---|---|---|---|---|
|Attribute Control<br><br>|[Figure 262]|[Figure 263]|[Figure 264]|[Figure 265]|[Figure 266]|[Figure 267]|
| |[Figure 268]|[Figure 269]|[Figure 270]|[Figure 271]|[Figure 272]|[Figure 273]|
|SliderEdit| | | | | | |
|K-Slider|[Figure 274]|[Figure 275]|[Figure 276]|[Figure 277]|[Figure 278]|[Figure 279]|
|Ours|[Figure 280]|[Figure 281]|[Figure 282]|[Figure 283]|[Figure 284]|[Figure 285]|

###### Figure 6. Qualitative Comparison with Linear Control Models. PixelSmile achieves smooth and monotonic expression transitions while preserving facial identity, whereas existing control methods either produce unstable responses or sacrifice identity consistency. The figure illustrates two representative expressions: happy (top row) and surprised (bottom row).

###### Group 1: General Editing Models. This group repre-

sents the strongest general-purpose text-guided image editing systems. We include three closed-source commercial systems: Nano Banana Pro, GPT-Image-1.5 (GPT-Image), Seedream-4.5 (Seedream), and three open-source models: Qwen-Image-Edit-2511 (Qwen-Edit), FLUX.2 Klein (FLUX-Klein), and LongCat-Image-Edit (LongCat). In the following, we refer to each model by the abbreviated name in parentheses. Although these models do not provide explicit mechanisms for fine-grained linear control, their strong generative priors make them competitive in overall expression editing quality. We therefore use them to evaluate expression editing accuracy and the ability to resolve structural confusion between semantically overlapping expressions.

###### Group 2: Linear Control Models. This group fo-

cuses on continuous attribute manipulation in latent space. We compare with recent control-oriented editing models including Kontinuous-Kontext (K-Slider), SliderEdit, and SAEdit. We also include earlier latent control approaches ConceptSlider and AttributeControl, using their officially recommended inversion strategies for real-image editing. While these earlier methods pioneered latent attribute control, they are often limited by narrow predefined attribute categories and information loss introduced by inversion. We therefore treat them as reference baselines rather than primary competitors in multi-category quantitative evaluation.

|src w/o ID Loss<br><br>| |Ours|
|---|---|---|
|[Figure 286]|[Figure 287]|[Figure 288]|
|[Figure 289]|[Figure 290]|[Figure 291]|
|[Figure 292]<br><br>[Figure 293]|[Figure 294]|[Figure 295]|
|[Figure 296]|[Figure 297]|[Figure 298]|

- Figure 7. Ablation on identity loss. Without ID loss, large expression intensities cause identity drift in hairstyle and skin texture. Our full method preserves identity consistently.

Evaluation Metrics. We adopt the benchmark protocol defined in Sec. 3.2. For Group 1, we evaluate editing accuracy and expression disentanglement using Acc-6, Acc-12, and mSCR. For Group 2, we evaluate linear intensity control and identity fidelity using CLS-6, CLS-12, and HES.

##### 5.2. Quantitative Evaluation

We quantitatively compare PixelSmile with both general editing and linear control models in Table 1 and Table 2.

Evaluation with General Editing Models. As shown in Table 1, we evaluate baselines on editing accuracy, structural confusion, and identity fidelity. For the six basic expressions, PixelSmile achieves the highest editing accuracy (0.8627), surpassing Nano Banana Pro (0.8431) and GPTImage (0.8039). On the twelve extended expressions, our method remains among the best-performing models. This partially reflects the bias of the VLM scoring model (Gemini 3 Pro), which is highly reliable on basic expressions but less consistent on extended expression categories. More importantly, PixelSmile achieves the lowest structural confusion rate (0.0550), significantly outperforming GPT-Image (0.1107) and Nano Banana Pro (0.1754), while most other models exceed 0.2000. A value approaching 0.5 indicates that the model tends to collapse the confusing expression pair into a single expression, reflecting poor disentanglement of overlapping expressions. In terms of identity fidelity, empirical observations in [74] suggest that realistic facial expression editing typically yields ID similarity values around 0.6–0.7. Scores above 0.8 often indicate rigid “copy-paste” behavior, while scores below 0.5 imply severe identity distortion. Some baselines fall into these extremes: Seedream maintains high ID similarity but suffers from large structural confusion due to limited edits, whereas FLUX-Klein drops below 0.5, significantly degrading identity consistency. In contrast, PixelSmile produces strong expressions while maintaining identity similarity within the natural range, achieving a better balance between expression strength and identity preservation.

Evaluation with Linear Control Models. PixelSmile demonstrates robust and consistent linear controllability across all metrics. ConceptSlider and AttributeControl are limited to editing only two expression attributes (happy and surprised) and produce weak editing effects; therefore, we report them as reference baselines with partial metrics (e.g., CLS-2) rather than full-category comparisons. SAEdit is a text-to-image method that does not explicitly support identity-preserving editing; therefore, we include it only for quantitative reference and do not provide detailed qualitative analysis. As shown in Table 2 and Figure 4, simply applying textual embedding interpolation to QwenEdit (zero-shot) already yields competitive controllability (CLS-6 0.6892, HES 0.4086), outperforming existing control baselines. With the proposed symmetric joint training,

###### w/o Contrastive Loss w/o Symmetrical framework Ours

|[Figure 299]|[Figure 300]|[Figure 301]|[Figure 302]|[Figure 303]|[Figure 304]|
|---|---|---|---|---|---|
|[Figure 305]|[Figure 306]|[Figure 307]|[Figure 308]|[Figure 309]|[Figure 310]|
|[Figure 311]|[Figure 312]|[Figure 313]|[Figure 314]|[Figure 315]|[Figure 316]|
|[Figure 317]|[Figure 318]|[Figure 319]|[Figure 320]|[Figure 321]|[Figure 322]|
|[Figure 323]|[Figure 324]|[Figure 325]|[Figure 326]|[Figure 327]|[Figure 328]|
|[Figure 329]|[Figure 330]<br><br>[Figure 331]|[Figure 332]|[Figure 333]|[Figure 334]|[Figure 335]|

- Figure 8. Ablation on symmetric contrastive learning. Both w/o Contrastive Loss and w/o Symmetric Framework suffer from expression confusion, while our full method achieves precise expression disentanglement. The upper three rows show angry and disgust, and the lower three rows show fear and surprised.

PixelSmile further improves performance and achieves the best results across all benchmarks (CLS-6 0.8078, CLS-12 0.7305, and HES 0.4723), indicating that explicitly modeling expression semantics is critical for stable and finegrained controllability. Figure 4 further reveals the limitations of existing methods. Although K-Slider and SliderEdit maintain high average ID similarity, this is largely because low editing intensities produce negligible changes, yielding ID similarity values close to 1.0. Specifically, KSlider exhibits negative CLS scores and irregular intensity fluctuations that never exceed ∼0.3, failing to establish lin-

ear controllability. SliderEdit shows increasing expression intensity but forces a rapid drop in ID similarity (down to ∼0.4) once expression scores approach 0.5, indicating a trade-off between editing strength and identity preservation. In contrast, PixelSmile achieves a monotonic response across a wide intensity range (expression scores reaching ∼0.8) while maintaining identity similarity within the natural 0.6–0.7 interval, effectively balancing controllability and fidelity. This behavior demonstrates that our method not only improves average performance but also ensures stable and predictable control across the intensity spectrum.

Table 3. Ablation Study. Best, second best, and third best results are indicated by , , and respectively.

Ablation mSCR ↓ ACC-6 ↑ ACC-12 ↑ CLS-6 ↑ CLS-12 ↑ HES ↑ ID Sim ↑

w/o Contrastive Loss 0.2725 0.6471 0.5889 0.6978 0.5889 0.4500 0.7018 w/o ID Loss 0.0550 0.8824 0.6500 0.8215 0.6874 0.4451 0.5749 w/o Sym. Frame. 0.1350 0.7843 0.4700 0.7939 0.6488 0.4253 0.6402

Loss

w/ Log-Ratio Constraint 0.1750 0.8039 0.5300 0.7917 0.6546 0.4933 0.6943 w/ Hinge Constraint 0.0950 0.8824 0.6600 0.7997 0.7228 0.4758 0.6280

Constraint

Data MEAD 0.2125 0.7647 0.4700 0.7047 0.6187 0.4235 0.5735 Ours Full Setting 0.0550 0.8627 0.6000 0.8078 0.7305 0.4723 0.6522

##### 5.3. Qualitative Comparison

We qualitatively compare PixelSmile with both general editing models and linear control baselines, as illustrated in Figure 5 and Figure 6.

Comparison with General Editing Models. As shown in Figure 5, existing general editing models struggle to simultaneously achieve clear expression editing and strong identity preservation. Several models, including Nano Banana Pro, Qwen-Edit, Seedream, and LongCat, preserve identity well but produce only weak expression changes, often resulting in barely noticeable edits. In contrast, GPTImage generates more visible expression differences but introduces moderate identity drift. FLUX-Klein performs the worst in both aspects, showing weak expression editing while severely degrading identity consistency. Compared with these methods, PixelSmile produces clear and recognizable expression changes while maintaining stable facial identity, achieving the best balance between semantic editing and identity preservation.

Comparison with Linear Control Models. Figure 6 compares continuous expression control across different methods. An ideal method should produce expression intensity that increases monotonically with the control parameter while preserving identity consistency. We first analyze the relatively simple expression Happy. ConceptSlider and AttributeControl show limited linear response but quickly degrade identity as editing strength increases. SliderEdit exhibits a step-like behavior: expressions remain nearly unchanged for most control values and suddenly increase at higher strengths, accompanied by significant identity degradation. K-Slider shows unstable behavior, where expression changes have little correlation with the control parameter. When moving to the more challenging expression Surprised, the linear response of these methods further deteriorates and identity preservation becomes worse. In contrast, PixelSmile maintains a stable monotonic increase in expression intensity while preserving identity across the entire control range. Even for more difficult expressions such as Disgust, our method continues to produce clear and controllable expression changes.

##### 5.4. Ablation Study

To validate the necessity of each component in PixelSmile, we conduct comprehensive ablation experiments, with quantitative results summarized in Table 3. Overall, the results reveal an inherent trade-off between expression editing capability and identity preservation: stronger editing often leads to identity degradation, while excessive identity constraints suppress effective expression transfer.

Ablation on Loss Framework. We first analyze the roles of the identity loss and the contrastive loss. Removing the identity loss improves expression editing and disentanglement but significantly degrades identity consistency. The model tends to modify facial attributes such as hairstyle or skin texture to match the target expression, especially at large editing intensities, leading to clear identity drift and inconsistent facial appearance across edits. As illustrated in Fig. 7, the full model maintains stable identity while the variant without ID loss shows noticeable identity changes, confirming the importance of identity supervision for preserving subject consistency.

0.60

0.14

0.55

0.12

0.50

TrainLoss

mSCR

0.10

0.45

0.40

0.08

w/o Sym. frame. (mSCR)

Ours (mSCR)

w/o Sym. frame. (Loss)

0.35

0.06

Ours (Loss)

1000 3000 5000 7000 9000

Steps

Figure 9. Training dynamics of symmetric contrastive learning. The asymmetric variant reduces loss faster in early training but leads to higher structural confusion, while the symmetric framework achieves lower and more stable mSCR.

Conversely, removing the contrastive loss yields the highest identity similarity but leads to the weakest editing accuracy and the highest structural confusion. Without the contrastive objective, the model collapses toward reconstructing the source image instead of performing meaningful expression edits. As shown in Fig. 8, the model without contrastive supervision fails to separate semantically similar expressions, resulting in severe expression confusion. These results demonstrate that the two losses play complementary roles: identity loss stabilizes facial identity, while contrastive loss enhances expression disentanglement.

Ablation on Symmetric Framework. We further compare the proposed symmetric training design with an asymmetric variant that applies contrastive supervision to only one branch. As shown in Fig. 8, removing the symmetric structure again leads to noticeable expression confusion. From the training dynamics in Fig. 9, the asymmetric model shows faster initial loss reduction but converges to worse solutions with lower editing accuracy and higher confusion rates. In contrast, the symmetric design acts as a structural regularizer: although it slows early convergence, the bidirectional constraints stabilize optimization and lead to better disentangled representations.

Ablation on Triplet Formulations. We also compare three triplet formulations: Log-ratio, Hinge, and InfoNCE. Logratio favors identity preservation but weakens expression editing, while Hinge maximizes editing strength at the cost of identity consistency. InfoNCE achieves the best balance between expression disentanglement and identity fidelity, and is therefore adopted as the default formulation.

Ablation on Dataset. Finally, we evaluate the impact of training data by training the same architecture on the widely used MEAD dataset [69], with preprocessing details provided in Appendix C. The MEAD-trained model consistently underperforms our full model across all metrics. This gap is mainly due to MEAD’s limited identity diversity and discrete intensity annotations, which restrict fine-grained expression modeling and semantic disentanglement. In contrast, our FFE provides richer identity variation and continuous soft-label supervision, enabling more precise and robust expression editing in the wild.

##### 5.5. User Study

We conducted a user study with 2,400 images and 10 trained annotators who ranked three continuous editing methods on expression continuity and identity consistency. Mean scores (continuity, identity) are: PixelSmile (4.48, 3.80); K-Slider (1.36, 4.06); SliderEdit (3.16, 1.14). As illustrated in Figure 10, human judgments are consistent with the machinebased evaluation. Overall, PixelSmile achieves the best balance, attaining the highest continuity while maintaining strong identity preservation.

##### 5.6. Expression Blend

Human facial behavior often involves compound expressions [17, 18, 56]. To explore whether such compositionality emerges in the learned representation, we perform pairwise linear interpolation among six basic expressions, producing 15 zero-shot combinations. As shown in Fig. 12 in the Appendix, several pairs generate perceptually coherent compound expressions, suggesting that the learned emotion manifold is continuous and compositional. However, some combinations collapse into a single dominant expression (e.g., Fear+Surprise) or produce unstable results due to physiological conflicts (e.g., Angry+Happy). Overall, 9 out of 15 combinations form plausible compound expressions, indicating that the learned representation supports linear composition while respecting implicit facial constraints and capturing meaningful compositional structure.

- 1
- 2
- 3
- 4
- 5

| |K-Slid|er| |Ours| |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | |S|liderE|dit| |
| | | | | | |
| | | | | | |

IdentityConsistency

1 2 3 4 5

Continuity

Figure 10. User study results. We show the trade-off between identity preservation and continuity of editing, annotated by human annotators. The size of the points indicates the HES scores of human annotators.

#### 6. Conclusion

In this paper, we present PixelSmile, a framework for addressing semantic entanglement in facial expression editing. By shifting from discrete supervision to the continuous expression manifold defined by FFE and evaluated through FFE-Bench, our approach enables precise and linearly controllable editing via symmetric joint training. Extensive experiments demonstrate effectiveness of PixelSmile in four dimensions: structural confusion, expression accuracy, linear controllability, and identity preservation. Overall, this work establishes a standardized framework for fine-grained facial expression editing and advances research toward continuous and compositional facial affect manipulation.

#### Ethics Statement

All data in FFE is collected from publicly available sources and used in compliance with their respective licenses and terms of use. The real-world subset is derived from existing public datasets (e.g., Human Images Dataset [66] and Matting Human Dataset [1], both distributed under the MIT License), while the anime subset consists of stylized fictional characters from publicly available media. We do not collect or use any private or login-restricted data. Facial expression editing is a dual-use technology that may pose risks, such as misuse in identity-related scenarios. Our work focuses on expression manipulation and is intended for noncommercial academic research. We do not aim to alter identity or enable deceptive applications. To mitigate potential risks, no personal metadata is retained, and the dataset is curated to exclude offensive content. We encourage responsible use of the dataset and models in compliance with applicable laws, regulations, and ethical guidelines.

#### References

- [1] aisegmentcn. Matting human datasets. https : //github.com/aisegmentcn/matting_human_ datasets, 2020. Accessed: 2026-03-03. 4, 13
- [2] Emad Barsoum, Cha Zhang, Cristian Canton Ferrer, and Zhengyou Zhang. Training deep networks for facial expression recognition with crowd-sourced label distribution. In ICMI, 2016. 3
- [3] Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Melvin Sevi, Vincent Tao Hu, and Bj¨orn Ommer. Continuous, subject-specific attribute control in t2i models by identifying semantic directions. In CVPR, 2025. 3, 6
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2
- [5] ByteDance. Seedream 4.5: Advanced ai image generation model. https://seed.bytedance.com/en/ seedream4_5, 2025. Accessed: 2026-03. 3, 6
- [6] Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arXiv:2506.07977,

2025. 7

- [7] Ta Ying Cheng, Prafull Sharma, Mark Boss, and Varun Jampani. Marble: Material recomposition and blending in clipspace. In CVPR, 2025. 3
- [8] Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, and Hongsheng Li. Generalizable neural performer: Learning robust radiance fields for human novel view synthesis. arXiv preprint arXiv:2204.11798, 2022. 16
- [9] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural

- actor repository for high-fidelity human-centric rendering. In ICCV, 2023. 16
- [10] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In CVPR, 2018. 2
- [11] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In CVPR, 2020. 2
- [12] Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. Fluxspace: Disentangled semantic editing in rectified flow transformers. arXiv preprint arXiv:2412.09611, 2024. 3
- [13] Radek Danˇeˇcek, Michael J Black, and Timo Bolkart. Emoca: Emotion driven monocular face capture and animation. In CVPR, 2022. 2
- [14] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 5, 6
- [15] Hui Ding, Kumar Sricharan, and Rama Chellappa. Exprgan: Facial expression editing with controllable expression intensity. In AAAI, 2018. 2
- [16] Zheng Ding, Xuaner Zhang, Zhihao Xia, Lars Jebe, Zhuowen Tu, and Xiuming Zhang. Diffusionrig: Learning personalized priors for facial appearance editing. In CVPR,

2023. 2

- [17] Shichuan Du and Aleix M Martinez. Compound facial expressions of emotion: from basic research to clinical applications. Dialogues in clinical neuroscience, 2015. 12
- [18] Shichuan Du, Yong Tao, and Aleix M Martinez. Compound facial expressions of emotion. PNAS, 2014. 12
- [19] Paul Ekman. An argument for basic emotions. Cognition & emotion, 1992. 4
- [20] Rohit Gandikota, Joanna Materzy´nska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In ECCV, 2024. 3, 6
- [21] Rohit Gandikota, Zongze Wu, Richard Zhang, David Bau, Eli Shechtman, and Nick Kolkin. Sliderspace: Decomposing the visual capabilities of diffusion models. In ICCV, 2025. 3
- [22] Nicola Garau, Niccolo Bisagno, Piotr Br´odka, and Nicola Conci. Deca: Deep viewpoint-equivariant human pose estimation using capsule autoencoders. In ICCV, 2021. 2
- [23] Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, and Tali Dekel. Tokenverse: Versatile multi-concept personalization in token modulation space. TOG, 2025. 3
- [24] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. NeurIPS, 2014. 2
- [25] Google. Nano banana pro: High-fidelity ai image generation and editing model. https : / / www.androidcentral.com/apps-software/ai/ googles-nano-banana-pro-and-more, 2025. Accessed: 2026-03. 3, 6
- [26] Julia Guerrero-Viu, Milos Hasan, Arthur Roullier, Midhun Harikumar, Yiwei Hu, Paul Guerrero, Diego Gutierrez, Belen Masia, and Valentin Deschaintre. Texsliders: Diffusion-

- based texture editing in clip space. In SIGGRAPH, 2024. 3
- [27] Zinan Guo, Yanze Wu, Chen Zhuowei, Peng Zhang, Qian He, et al. Pulid: Pure and lightning id customization via contrastive alignment. NeurIPS, 2024. 2
- [28] Erik H¨ark¨onen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. NeurIPS, 2020. 2
- [29] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 3
- [30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2
- [31] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 2022. 5
- [32] Rahul Jain, Amit Goel, Koichiro Niinuma, and Aakar Gupta. Adaptivesliders: User-aligned semantic slider-based editing of text-to-image model output. In CHI, 2025. 3
- [33] Wooseok Jang, Youngjun Hong, Geonho Cha, and Seungryong Kim. Controlface: Harnessing facial parametric control for face rigging. In CVPR, 2025. 2
- [34] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. In ICCV, 2025. 2
- [35] Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik, and Daniel Cohen-Or. Saedit: Tokenlevel control for continuous image editing via sparse autoencoder. arXiv preprint arXiv:2510.05081, 2025. 3, 6
- [36] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 2
- [37] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020. 2
- [38] Minchul Kim, Anil K Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition. In CVPR,

2022. 5

- [39] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025. 3, 6
- [40] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 3

- [41] Oliver Langner, Ron Dotsch, Gijsbert Bijlstra, Daniel HJ Wigboldus, Skyler T Hawk, and AD Van Knippenberg. Presentation and validation of the radboud faces database. Cognition and Emotion, 2010. 3
- [42] Shan Li, Weihong Deng, and JunPing Du. Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild. In CVPR, 2017. 3
- [43] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 5

- [44] Lu Liu, Huiyu Duan, Qiang Hu, Liu Yang, Chunlei Cai, Tianxiao Ye, Huayu Liu, Xiaoyun Zhang, and Guangtao Zhai. F-bench: Rethinking human preference evaluation metrics for benchmarking face generation, customization, and restoration. In ICCV, 2025. 3
- [45] Ming Liu, Yukang Ding, Min Xia, Xiao Liu, Errui Ding, Wangmeng Zuo, and Shilei Wen. Stgan: A unified selective transfer network for arbitrary image attribute editing. In CVPR, 2019. 2
- [46] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2, 3
- [47] Patrick Lucey, Jeffrey F Cohn, Takeo Kanade, Jason Saragih, Zara Ambadar, and Iain Matthews. The extended cohnkanade dataset (ck+): A complete dataset for action unit and emotion-specified expression. In 2010 ieee computer society conference on computer vision and pattern recognitionworkshops, 2010. 3
- [48] Daniel Lundqvist, Anders Flykt, and Arne Ohman.¨ Karolinska directed emotional faces. Cognition and Emotion, 1998. 3
- [49] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 2
- [50] Ali Mollahosseini, Behzad Hasani, and Mohammad H Mahoor. Affectnet: A database for facial expression, valence, and arousal computing in the wild. IEEE transactions on affective computing, 2017. 3
- [51] Arsha Nagrani, Joon Son Chung, Weidi Xie, and Andrew Zisserman. Voxceleb: Large-scale speaker verification in the wild. Computer Speech & Language, 2020. 3
- [52] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. Deep metric learning via lifted structured feature embedding. In CVPR, 2016. 6
- [53] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 6
- [54] OpenAI. Introducing gpt-image-1.5. https://openai. com/index/new- chatgpt- images- is- here/,

2025. Accessed: 2026-03. 3, 6

- [55] OysterQAQ. Danbooruclip. https://huggingface. co/OysterQAQ/DanbooruCLIP, 2023. Accessed: 2023-05-18. 7
- [56] Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, et al. Renderme-360: A large digital asset library and benchmarks towards high-fidelity head avatars. NeurIPS,

2023. 12, 16

- [57] Rishubh Parihar, Or Patashnik, Daniil Ostashev, R Venkatesh Babu, Daniel Cohen-Or, and Kuan-Chieh Wang. Kontinuous kontext: Continuous strength control for instruction-based image editing. arXiv preprint arXiv:2510.08532, 2025. 3, 6
- [58] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 5

- [59] Albert Pumarola, Antonio Agudo, Aleix M Martinez, Alberto Sanfeliu, and Francesc Moreno-Noguer. Ganimation: Anatomically-aware facial animation from a single image. In ECCV, 2018. 2
- [60] Zongyang Qiu, Bingyuan Wang, Xingbei Chen, Yingqing He, and Zeyu Wang. Emovid: A multimodal emotion video dataset for emotion-centric video understanding and generation. arXiv preprint arXiv:2511.11002, 2025. 3
- [61] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICLR, 2021. 7
- [62] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In CVPR, 2015. 5, 6
- [63] Prafull Sharma, Varun Jampani, Yuanzhen Li, Xuhui Jia, Dmitry Lagun, Fredo Durand, Bill Freeman, and Mark Matthews. Alchemist: Parametric control of material properties with diffusion models. In CVPR, 2024. 3
- [64] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in gans. In CVPR, 2021. 2
- [65] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of gans for semantic face editing. In CVPR, 2020. 2
- [66] snmahsa. Human images dataset (men and women). https://www.kaggle.com/datasets/snmahsa/ human-images-dataset-men-and-women, 2021. Accessed: 2026-03-03. 4, 13
- [67] Deepak Sridhar and Nuno Vasconcelos. Prompt sliders for fine-grained control, editing and erasing of concepts in diffusion models. In ECCV, 2024. 3
- [68] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025. 3, 6
- [69] Kaisiyuan Wang, Qianyi Wu, Linsen Song, Zhuoqian Yang, Wayne Wu, Chen Qian, Ran He, Yu Qiao, and Chen Change Loy. Mead: A large-scale audio-visual dataset for emotional talking-face generation. In ECCV, 2020. 3, 12, 16
- [70] Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric cross entropy for robust learning with noisy labels. In ICCV, 2019. 6
- [71] Mengting Wei, Tuomas Varanka, Xingxun Jiang, Huai-Qian Khor, and Guoying Zhao. Magicface: High-fidelity facial expression editing with action-unit control. arXiv preprint arXiv:2501.02260, 2025. 2
- [72] Alon Wolf, Chen Katzir, Kfir Aberman, and Or Patashnik. Continuous control of editing models via adaptive-origin guidance. arXiv preprint arXiv:2602.03826, 2026. 3
- [73] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 2, 3, 5, 6
- [74] Hengyuan Xu, Wei Cheng, Peng Xing, Yixiao Fang, Shuhan Wu, Rui Wang, Xianfang Zeng, Daxin Jiang, Gang

Yu, Xingjun Ma, et al. Withanyone: Towards controllable and id consistent image generation. arXiv preprint

- arXiv:2510.14975, 2025. 2, 9

- [75] Zhenyu Xu, Xiaoqi Shen, Haotian Nan, and Xinyu Zhang. Numerikontrol: Adding numeric control to diffusion transformers for instruction-based image editing. arXiv preprint

arXiv:2511.23105, 2025. 3

- [76] Yuqi Yang, Dongliang Chang, Yuanchen Fang, Yi-Zhe SonG, Zhanyu Ma, and Jun Guo. Controllable-continuous color editing in diffusion model via color mapping. arXiv preprint arXiv:2509.13756, 2025. 3
- [77] Weixin Ye, Hongguang Zhu, Wei Wang, Yahui Liu, and Mengyu Wang. All-in-one slider for attribute manipulation in diffusion models. arXiv preprint arXiv:2508.19195, 2025. 3
- [78] Lijun Yin, Xiaozhou Wei, Yi Sun, Jun Wang, and Matthew J Rosato. A 3d facial expression database for facial behavior research. In 7th international conference on automatic face and gesture recognition (FGR06), 2006. 3
- [79] Xingxi Yin, Jingfeng Zhang, Yue Deng, Zhi Li, Yicheng Li, and Yin Zhang. Instructattribute: Fine-grained object attributes editing with instruction. arXiv preprint arXiv:2505.00751, 2025. 3
- [80] O˘guz Kaan Y¨uksel, Enis Simsar, Ezgi G¨ulperi Er, and Pinar Yanardag. Latentclr: A contrastive learning approach for unsupervised discovery of interpretable directions. In ICCV,

2021. 2

- [81] Arman Zarei, Samyadeep Basu, Mobina Pournemat, Sayan Nag, Ryan Rossi, and Soheil Feizi. Slideredit: Continuous image editing with fine-grained instruction control. arXiv preprint arXiv:2511.09715, 2025. 3, 6
- [82] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2
- [83] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. From facial expression recognition to interpersonal relation prediction. IJCV, 2018. 3
- [84] Zhicheng Zhang, Weicheng Wang, Yongjie Zhu, Wenyu Qin, Pengfei Wan, Di Zhang, and Jufeng Yang. Videmo: Affective-tree reasoning for emotion-centric video foundation models. arXiv preprint arXiv:2511.02712, 2025. 3
- [85] Weizhi Zhong, Huan Yang, Zheng Liu, Huiguo He, Zijian He, Xuesong Niu, Di Zhang, and Guanbin Li. Mod-adapter: Tuning-free and versatile multi-concept personalization via modulation adapter. arXiv preprint arXiv:2505.18612, 2025. 3
- [86] Hao Zhu, Wayne Wu, Wentao Zhu, Liming Jiang, Siwei Tang, Li Zhang, Ziwei Liu, and Chen Change Loy. Celebvhq: A large-scale video facial attributes dataset. In ECCV,

2022. 16

- [87] Yule Zhu, Ping Liu, Zhedong Zheng, and Wei Liu. Seed: A benchmark dataset for sequential facial attribute editing with diffusion models. arXiv preprint arXiv:2506.00562, 2025. 3
- [88] Cailin Zhuang, Ailin Huang, Yaoqi Hu, Jingwei Wu, Wei Cheng, Jiaqi Liao, Hongyuan Wang, Xinyao Liao, Weiwei Cai, Hengyuan Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. arXiv preprint arXiv:2505.24862, 2025. 7

## Appendix

#### A. Details of the Symmetric Contrastive Loss

- A.1. Triplet Constraint Formulations

In this section, we provide detailed formulations of the triplet constraint function T (G,P,N) used in Sec. 4. All features are extracted using a frozen CLIP image encoder to represent expression semantics and are ℓ2-normalized before distance computation. For brevity, we denote dG,P = d(G,P) and dG,N = d(G,N) as cosine distances, and sG,P = sim(G,P) and sG,N = sim(G,N) as cosine similarities.

Hinge-based Formulation. The margin-based objective is

Thinge(G,P,N) = max 0, dG,P − dG,N + m , (10) where m is a fixed margin.

Log-Ratio Formulation. We adopt a smooth distanceratio objective:

Tratio(G,P,N) = log

dG,P + ϵ dG,N + ϵ

, (11) where ϵ is a small constant for numerical stability.

InfoNCE-style Formulation. The probabilistic contrastive objective is

Tnce(G,P,N) = −log

exp(sG,P/τ) x∈{P,N} exp(sG,x/τ)

, (12)

where τ is a temperature parameter.

- A.2. Implementation Details

Unless otherwise specified, we use the InfoNCE-style formulation with temperature τ = 0.07. For the hinge-based variant, the margin is set to m = 0.2. For the log-ratio formulation, we set ϵ = 10−6 for numerical stability. All variants are evaluated under identical training schedules.

#### B. Details of Experiment

To ensure reproducibility and clarity, we provide additional implementation details for PixelSmile. Training is conducted on 4 NVIDIA H200 GPUs.

LoRA Configuration. We apply LoRA to major attention and MLP components of the diffusion transformer. Key hyperparameters are: rank = 64, α = 128, and dropout = 0.

Training Hyperparameters. The models are optimized for 100 epochs using the AdamW optimizer with β1 = 0.9, β2 = 0.999, weight decay = 0.001, and ϵ = 1e−8. The learning rate is set to 1e−4 with cosine scheduling and 500 warmup steps. Mixed precision (bf16) is enabled to stabilize training. For the loss weights, we set λSC = 1.0 (InfoNCE mode, symmetric) and λID = 0.1. The batch size per GPU is 4 with gradient accumulation steps = 1.

#### C. Details of Dataset Ablation

##### C.1. Dataset Overview

Among human-centric dataset [8, 9, 56, 69, 86], we choose MEAD [69] to ablate on effectiveness of proposed dataset. The MEAD dataset [69] contains 7 discrete facial expressions captured from multi-view video sequences, with three intensity levels (low, medium, high). For our ablation, we only use the front-view subset and map its three intensity levels to continuous values 0.5,0.75,1.0 to match the input range of PixelSmile.

##### C.2. Preprocessing and Triplet Construction

Since MEAD provides video sequences, we uniformly sample frames to obtain independent images. From these sampled frames, we construct triplets (Pa,Pb,Iorig ) in the same manner as for FFE to train the symmetric contrastive framework. Each triplet consists of:

- • Iorig 1: the source frame.
- • Pa,Pb 2: two frames of the same subject with distinct expressions.

Finally, we construct triplet data pairs from the same identities to conduct the symmetric contrastive training under our default configuration.

#### D. Additional Qualitative Results

This section provides additional qualitative results for PixelSmile. We present more examples of linear expression editing across multiple expression categories, as well as additional expression blending results obtained through interpolation in the learned expression space.

##### D.1. Additional Linear Expression Editing Results

- Figure 11 presents additional linear editing results for the remaining ten expressions across both real and anime domains. As the control parameter increases, the expression intensity changes smoothly while the facial identity remains consistent.

D.2. Expression Blend Results

- Figure 12 shows the examples of expression blending obtained through pairwise interpolation between basic expressions.

[Figure 336]

|Anxious|[Figure 337]|[Figure 338]|[Figure 339]|[Figure 340]|[Figure 341]<br><br>[Figure 342]|[Figure 343]|
|---|---|---|---|---|---|---|
| | | | | | | |
|Contempt|[Figure 344]|[Figure 345]|[Figure 346]|[Figure 347]|[Figure 348]<br><br>[Figure 349]|[Figure 350]|
| | | | | | | |
| | | | | | | |
|Disgust<br><br>Fear<br><br>Sad|[Figure 351]|[Figure 352]|[Figure 353]|[Figure 354]|[Figure 355]<br><br>[Figure 356]|[Figure 357]|
| | | | | | | |
| | | | | | | |
| |[Figure 358]|[Figure 359]|[Figure 360]|[Figure 361]|[Figure 362]<br><br>[Figure 363]|[Figure 364]|
| | | | | | | |
| | | | | | | |
| |[Figure 365]|[Figure 366]|[Figure 367]|[Figure 368]|[Figure 369]<br><br>[Figure 370]|[Figure 371]|
| | | | | | | |

|Angry|[Figure 372]|[Figure 373]|[Figure 374]|[Figure 375]|[Figure 376]<br><br>[Figure 377]|[Figure 378]|
|---|---|---|---|---|---|---|
| | | | | | | |
|Confident|[Figure 379]|[Figure 380]|[Figure 381]|[Figure 382]|[Figure 383]<br><br>[Figure 384]|[Figure 385]|
| | | | | | | |
| | | | | | | |
|Confused<br><br>Shy<br><br>Sleepy|[Figure 386]|[Figure 387]|[Figure 388]|[Figure 389]|[Figure 390]<br><br>[Figure 391]|[Figure 392]|
| | | | | | | |
| | | | | | | |
| |[Figure 393]|[Figure 394]|[Figure 395]|[Figure 396]|[Figure 397]<br><br>[Figure 398]|[Figure 399]|
| | | | | | | |
| | | | | | | |
| |[Figure 400]|[Figure 401]|[Figure 402]|[Figure 403]|[Figure 404]<br><br>[Figure 405]|[Figure 406]|
| | | | | | | |

###### Figure 11. Additional linear expression editing results. We show the remaining ten expressions across both real and anime domains. The top row shows results on real images, while the bottom row shows results on anime images. Expression intensity increases from left to right for each expression.

[Figure 407]

|Happy|Surprised|Blended| |Happy|Sad|Blended|
|---|---|---|---|---|---|---|
|[Figure 408]|[Figure 409]|[Figure 410]| |[Figure 411]|[Figure 412]|[Figure 413]|

Disgust Happy

Blended Blended

Disgust Sad

|[Figure 414]|[Figure 415]|[Figure 416]|
|---|---|---|

|[Figure 417]|[Figure 418]|[Figure 419]|
|---|---|---|

Fear Happy Fear

Blended Blended

Sad

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

- Figure 12. Expression Blending Results. Visualizing compositional facial expressions generated by smoothly blending multiple emotional categories in PixelSmile.

#### E. Additional Dataset Details

This section provides supplementary details of FFE, including annotation and scoring prompts, together with additional dataset statistics.

##### E.1. Annotation and Scoring Prompts

We provide the prompt templates used in our annotation pipeline and expression scoring procedure. Table 4 and

- Table 5 present the prompts used for statistical annotation of the human and anime subsets, respectively. These two templates are designed to extract structured semantic attributes for dataset analysis and are both based on Qwen3VL-235B-A22B. Table 6 shows the prompt used to assign expression intensity scores to images, which is based on Gemini 3 Pro.

##### E.2. Dataset Statistics

We present the statistical analysis of FFE in Fig. 13 and Fig. 14, covering categorical distributions and textual description patterns across the real-world and anime domains.

From Fig. 13, the real-world subset is diverse but imbalanced, dominated by young adults (53.5%), with children, teens, and seniors forming smaller proportions. Similar trends are observed in other attributes, where female samples are more frequent and light-to-medium skin tones

constitute the majority, indicating that the dataset inherits non-uniform demographic characteristics. This bias reflects common patterns in portrait-centric internet images and introduces challenges for expression modeling. In contrast, the anime subset exhibits broader stylistic diversity, with CG and 2D anime each accounting for about 44%, along with additional styles such as chibi, manga, and sketch. Compared with the real-world subset, the anime subset also shows a flatter age distribution. However, it contains more unknown labels in attributes like gender and age, suggesting that stylized characters are inherently more ambiguous under real-world categorization schemes, which increases the difficulty of consistent annotation and evaluation.

Fig. 14 further reveals domain-specific textual patterns. The real-world subset emphasizes natural appearance cues such as clothing, hairstyle, and facial details, while the anime subset contains more stylized and visually distinctive descriptions. This difference highlights that expression editing in FFE involves both visual transformation and domain-dependent semantic interpretation, requiring models to generalize across heterogeneous distributions. Overall, these statistics indicate that FFE combines substantial diversity with realistic biases, making it a challenging and representative benchmark for fine-grained, diverse, and real-world facial expression editing.

53.5%

Young Adult

19.7%

Adult

12.4%

Middle Aged

5.6%

Child

5.0%

Teen

3.6%

Senior

0.1%

Unknown

0% 10% 20% 30% 40% 50% 60%

Percentage (%)

(a) Age distribution in the real-world domain

44.7%

Cg Anime

44.1%

2D Anime

4.0%

Chibi

2.7%

Manga

2.4%

Unknown

1.2%

Sketch

1.0%

Other

0% 10% 20% 30% 40% 50%

Percentage (%)

(b) Style distribution in the anime domain

- Figure 13. Statistical distributions of annotated data in FFE. The results provide insights into the underlying data characteristics across real-world and anime domains.

Table 4. Human Dataset Annotation Prompt Template.

Human Dataset Annotation Prompt System Context You are an image annotation assistant. For a single person image, output strict JSON only. Requirements

- • Describe only visible facts; do not infer identity, story, or intent.
- • categorical values must be selected from the provided enums.
- • All three fields in descriptions must be present.
- • Write all description sentences in English, each 8–25 words. JSON Schema

{

"categorical": { "gender": "male/female/androgynous/unknown", "age_group": "child/teen/young_adult/adult/middle_aged/senior/unknown", "skin_tone": "very_light/light/medium/dark/very_dark/unknown", "expression": "neutral/happy/sad/angry/surprised/fear/disgust/other/unknown"

[Figure 426]

[Figure 427]

(a) Real-world appearance descriptions (b) Anime-style appearance descriptions

- Figure 14. Visualization of appearance-related textual descriptions in FFE. The visualizations highlight the distribution and diversity of annotations across real-world and anime domains.

Table 5. Anime Dataset Annotation Prompt Template.

Anime Dataset Annotation Prompt System Context You are an anime image annotation assistant. For a single character image, output strict JSON only. Requirements

- • Describe only visible facts; do not infer identity, story, or intent.
- • categorical values must be selected from the provided enums.
- • All three fields in descriptions must be present.
- • Write all description sentences in English, each 8–25 words. JSON Schema

{

"categorical": { "gender": "male/female/androgynous/unknown", "age_group": "child/teen/young_adult/adult/middle_aged/senior/unknown", "expression": "neutral/happy/sad/angry/surprised/fear/disgust/other/unknown", "anime_style": "2d_anime/chibi/manga/sketch/cg_anime/other/unknown"

- Table 6. Facial Expression Scoring Prompt Template. The same prompt is applied to both human and anime domains, highlighting the domain-agnostic nature of our scoring pipeline.

Facial Expression Scoring Prompt Role and Task Definition You are an expert AI specialized in analyzing facial expressions in both photorealistic and anime-style images. Your task is to analyze the input image and determine the intensity score for specific expression categories. Target Emotion Definitions Analyze the image based on the following visual definitions. Crucially, pay close attention to the distinctions between similar expressions (e.g., Fear vs. Surprise, Anger vs. Disgust).

- • Happiness: Faintly smiling, cheerful, ecstatic; corners of the mouth raised or laughing.
- • Sadness: Somber, sorrowful, devastated; frowning with downcast eyes or tears.
- • Anger: Annoyed, angry, furious; furrowed brows and glaring eyes.
- • Fear: Wary, frightened, terrified; knitted brows and pupil constriction.
- • Surprise: Taken aback, amazed, stunned; eyes wide open or jaw dropped.
- • Disgust: Distasteful, repulsed, gagging; wrinkling the nose or raising the upper lip.
- • Embarrassment: A shy smile, blushing cheeks, avoiding eye contact, looking down, or covering the face with hands.
- • Confidence: Self-assured smile, corners of the mouth raised, sharp and firm gaze, slightly smug.
- • Confusion: Knitted brows, looking aside or rolling the eyes, head tilted in deep thought.
- • Drowsiness: Heavy drooping eyelids, yawning, lethargic or low-energy look.
- • Contempt: Asymmetrical smirk (one corner raised), sneering, looking down on someone.
- • Nervousness: Visible sweat drops, tense facial muscles, uneasy or restless gaze. Output Format Requirements
- • Return format: Output a standard JSON object.
- • Keys: Use the exact emotion labels provided above.
- • Values: The intensity score (a float between 0.00 and 1.00).
- • Constraint: Do not explain your reasoning. Output only the JSON object. Do not include Markdown formatting such as ‘‘‘json ... ‘‘‘.

###### Output Example

{

"Happiness": 0.90, "Surprise": 0.75, "Anger": 0.05,

... }

