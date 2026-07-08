## MagicMirror: ID-Preserved Video Generation in Video Diffusion Transformers

Yuechen Zhang*1 Yaoyang Liu*2 Bin Xia1 Bohao Peng1 Zexin Yan3 Eric Lo1 Jiaya Jia1,2,4 1CUHK 2HKUST 3CMU 4SmartMore

# arXiv:2501.03931v2[cs.CV]22Nov2025

Figure 1. MagicMirror generates text-to-video results given the ID reference image. Complete videos are available in https:// julianjuaner.github.io/projects/MagicMirror/.

### Abstract

We present MagicMirror, a framework for generating identity-preserved videos with cinematic-level quality and dynamic motion. While recent advances in video diffusion models have shown impressive capabilities in text-to-video generation, maintaining consistent identity while producing natural motion remains challenging. Previous methods either require person-specific fine-tuning or struggle to balance identity preservation with motion diversity. Built upon Video Diffusion Transformers, our method introduces three key components: (1) a dual-branch facial feature extractor that captures both identity and structural features, (2) a lightweight cross-modal adapter with Conditioned Adaptive Normalization for efficient identity integration, and (3) a two-stage training strategy combining synthetic identity pairs with video data. Extensive experiments demonstrate that MagicMirror effectively balances identity consistency with natural motion, outperforming existing methods across multiple metrics while requiring minimal parameters added. The code and model will be made publicly available.

### 1. Introduction

Recent advancements in image generation, particularly through Diffusion Models [4, 22, 48, 51], have propelled

personalized content creation to the forefront of computer vision research. While significant progress has been made in preserving personal identity (ID) in image generation [5, 19, 24, 34, 44, 56], achieving comparable fidelity in video generation remains challenging.

Existing ID-preserving video generation methods show promising results but face limitations. Approaches like Magic-Me and ID-Animator [20, 39], utilizing inflated UNets [18] for fine-tuning or adapter training, demonstrate some success in maintaining identity across frames. However, they are ultimately restricted by the generation model’s inherent capabilities, often failing to produce highquality videos (see Fig. 2). These approaches make a static copy-and-paste instead of generating dynamic facial motions. The co-current work ConsisID [69], implements identity-preserving video generation via a Diffusion Transformer (DiT) framework. While achieving notable performance in video quality, it still exhibits limitations in temporal smoothness and naturalistic facial motions. Another branch of methods combines image personalization methods with Image-to-Video (I2V) generation [60, 63, 64]. While these two-stage solutions preserve ID to some extent, they often struggle with stability in longer sequences and require a separate image generation step. To address current shortcomings, we present MagicMirror, a single-stage framework designed to generate high-quality videos while

maintaining strong ID consistency and dynamic natural facial motions. Our approach leverages native video diffusion models [64] to generate ID-specific videos, aiming to empower individuals as protagonists in their virtual narratives, and bridge the gap between personalized ID generation and high-quality video synthesis.

The generation of high-fidelity identity-preserving videos poses several technical challenges. A primary challenge stems from the architectural disparity between image and video generation paradigms. State-of-the-art video generation models, built on full-attention Diffusion Transformer (DiT) architectures [42, 64], are not directly compatible with conventional cross-attention conditioning methods. To bridge this gap, we introduce a lightweight identityconditioning adapter integrated into the Video DiT framework. Specifically, we propose a dual-branch facial embedding that simultaneously preserves high-level identity features and reference-specific structural information. Meanwhile, we observed that current video foundation models optimize for text-video alignment, often at the cost of spatial fidelity and generation quality. This trade-off manifests in reduced image quality metrics on benchmarks such as VBench [25], particularly introducing the challenge of preserving fine-grained identity features. To address it, we develop a Conditioned Adaptive Normalization (CAN) that effectively incorporates identity conditions, as a distribution prior, into the pre-trained foundation model. The CAN module, combined with a learnable cross-attention, enables identity conditioning through attention guidance and feature distribution guidance.

Another significant challenge lies in the acquisition of high-quality training data. While paired image data with consistent identity is relatively abundant, obtaining high quality [66] image-video pairs that maintain high-fidelity identity consistency remains scarce. To address this limitation, we develop a strategic data synthesis pipeline that leverages identity preservation models [34] to generate paired training data. Our training methodology employs a progressive approach: initially pre-training on image data to learn robust identity representations, followed by videospecific fine-tuning. This two-stage strategy enables effective learning of identity features while ensuring temporal consistency in facial expressions across video sequences.

We evaluate our method on multiple general metrics by constructing a human-centric video generation test set, and comparing it with the aforementioned competitive IDpreserved video generation methods. Extensive experimental and visual evaluations [54] demonstrate that our approach successfully generates high-quality videos with dynamic content and strong facial consistency, as illustrated in Fig. 1. By integrating identity preservation with natural facial motion in Video DiT frameworks without casespecific fine-tuning, MagicMirror advances personalized

ID-Animator ID-Preserving |PromptFollowing|Motion

|[Figure 1]<br><br>[Figure 2]<br><br>An old man is wearing a black suit, smiling and talking. The background is blurred,seems in a coffee shop.|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

Video ocean (close-sourced)

ID-Preserving | Prompt Following | Motion

[Figure 6]

[Figure 7]

[Figure 8]

Magic Mirror (Ours)

ID-Preserving | Prompt Following | Motion

[Figure 9]

[Figure 10]

[Figure 11]

Figure 2. MagicMirror generates dynamic facial motion. IDAnimator [20] and Video Ocean [37] exhibit limited motion range due to a strong identity-preservation constraint. MagicMirror achieves more dynamic facial expressions while maintaining reference identity fidelity.

video generation and enhances creative expression in digital storytelling.

Our main contributions are three-fold: (1) We introduce MagicMirror, a novel fine-tuning free framework with a dual-branch condition extractor for generating IDpreserving videos; (2) We design a lightweight adapter with a conditioned adaptive normalization module, for effective integration of identity features in full-attention Diffusion Transformer architectures; (3) We develop a dataset construction method that combines synthetic data generation with a progressive training strategy to address data scarcity challenges in personalized video generation.

### 2. Related Works

Diffusion Models. Since the introduction of DDPM [22], diffusion models have demonstrated remarkable capabilities across diverse domains, spanning NLP [16, 32], medical imaging [6, 7], and molecular modeling [8, 26]. In computer vision, following initial success in image generation [11, 29], Latent Diffusion Models (LDM) [48] significantly reduced computational requirements while maintaining generation quality. Subsequent developments in conditional architectures [40, 50] enabled fine-grained concept customization over the generation process.

Video Generation via Diffusion Models. Following the emergence of diffusion models, their superior controllability and diversity in image generation [62] have led to their prominence over traditional approaches based on GANs [17, 27, 28] and auto-regressive Transformers [13, 46, 67]. The Video Diffusion Model (VDM) [23] pioneered video generation using diffusion models by extending the traditional U-Net [49] architecture to process

temporal information. Subsequently, LVDM [21] demonstrated the effectiveness of latent space operations, while AnimateDiff [18] adapted text-to-image models for personalized video synthesis. A significant advancement came with the Diffusion Transformer (DiT) [42], which successfully merged Transformer architectures [12, 55] with diffusion models. Building on this foundation, Latte [38] emerged as the first open-source text-to-video model based on DiT. Following the breakthrough of SORA [41], several open-source initiatives including Open-Sora-Plan [35], Open-Sora [73], and CogVideoX [64] have advanced video generation through DiT architectures. While current research predominantly focuses on image-to-video translation [60, 63, 70] efficiency, and motion control [59, 61, 68, 72], the critical challenge of ID-preserving video generation remains relatively unexplored.

ID-Preserving Generation. ID-preserving generation, or identity customization, aims to maintain specific identity characteristics in generated images or videos. Initially developed in the GAN era [17] with significant advances in face generation [27, 47, 58], this field has evolved substantially with diffusion models, demonstrating enhanced capabilities in novel image synthesis [14, 50]. Current approaches to ID-preserving image generation fall into two main categories:

Tuning-based Models: These approaches fine-tune models using one or more reference images to generate identityconsistent outputs. Notable examples include Textual Inversion [14] and Dreambooth [50].

Tuning-free Models: Addressing the computational overhead problem, these models maintain high ID fidelity through additional conditioning and trainable parameters. Starting with IP-adapter [65], various methods such as InstantID, PhotoMaker [5, 19, 24, 34, 44, 56] have emerged to enable efficient and high quality personalized generation.

ID-preserving video generation introduces additional complexities, particularly in synthesizing realistic facial movements from static references while maintaining identity consistency. Current approaches include MagicMe [39], a tuning-based method requiring per-identity optimization, and ID-Animator [20], a tuning-free approach utilizing face adapters and decoupled Cross-Attention [65]. However, these methods face challenges in maintaining dynamic expressions while preserving identity, and are constrained by base model limitations in video quality, duration, and prompt adherence. The integration of Diffusion Transformers presents promising opportunities for advancing ID-preserving video generation. ConsisID [69] is the co-current work with the DiT architecture, but it faces the problem of video smoothness and face dynamics.

### 3. MagicMirror

An overview of MagicMirror is illustrated in Fig. 3. This dual-branch framework (Sec. 3.2) extracts facial identity features from one or more reference images r. They are subsequently processed through a DiT backbone augmented with a lightweight cross-modal adapter, incorporating Conditioned Adaptive Normalization (Sec. 3.3). This architecture enables MagicMirror to synthesize identity-preserved text-to-video outputs. The following sections elaborate on the preliminaries of diffusion models (Sec. 3.1) and each component of our method.

- 3.1. Preliminaries Latent Diffusion Models (LDMs) generate data by iteratively reversing a noise corruption process, converting random noise into structured samples. At time step t ∈ {0,...,T}, the model predicts latent state xt conditioned on xt+1:

pθ(xt|xt+1) = N(xt; µt, βtI), (1)

where θ represents the model parameters, µt denotes the predicted mean, and βt is the variance schedule.

The training objective typically employs a mean squared error loss Lnoise on the noise prediction ϵθ(xt,t,ctxt):

Lnoise = Et,c

txt,ϵ∼N(0,1) ∥ϵ − ϵθ(xt,t,ctxt)∥2 , (2) where ctxt denotes the text condition.

Recent studies on controllable generation [43, 56, 65, 71] extend this framework by incorporating additional control signals, such as image condition cimg. This is achieved through a feature extractor τimg that processes a reference image r: cimg = τimg(r). Consequently, the noise prediction function in Eq. (2) becomes ϵθ(xt,t,ctxt,cimg). In this paper, we denote the additional facial condition by cface.

- 3.2. Dual-Branch Facial Feature Extraction The facial feature extraction module in MagicMirror, depicted in the left part of Fig. 3, is designed to extract highlevel identity information and detailed structural features. Given an identity reference image r ∈ Rh×w×3, our model

extracts facial condition embeddings cface = {xface,xˆid} using a dual-branch perceiver architecture, where each branch specializes in a distinct aspect of facial representation.

To obtain these embeddings, we first extract dense feature maps f = Ffeat(r) from the input image, where Ffeat is a pre-trained CLIP ViT encoder [45] that captures rich facial semantics. The identity branch employs an identity perceiver τid to extract high-level identity features:

xid = τid(qid,f), (3) where qid = Fid(r) represents high-level identity-aware features extracted by an ArcFace encoder Fid [9, 34].

The structural branch utilizes a facial structure perceiver τface that focuses on fine-grained facial details, leveraging a learnable query embedding qface for facial structure

User text input (embedding)

Identitybranch

Multi-modal latent input

|Q|
|---|

IDEmbedding#!"

| |$%!"| |
|---|---|---|
| | | |

!"!"!

|!!"!|
|---|

|$(!"| |
|---|---|
| | |

Trainable modules

|IDEmbedding Extractor !!"|!|IDPerceiver "!"<br><br>| |$| |
|---|---|---|
| |!"| |
<br><br>Fusion MLP MLP!"#$ Distribution|
|---|---|---|
|DenseFeature Extractor !#$%&|K & V|FacePerceiver "#%'$<br><br>$#%'$<br><br>Attention<br><br>condition|

[Figure 12]

[Figure 13]

DiT Blocks (CogVideoX)

ID Reference

Generated Video

condition

Cross-modal adapter

|Q|
|---|

Structuralbranch

Learnablefeaturequery ##%'$

- Figure 3. Overview of MagicMirror. The framework employs a dual-branch feature extraction system with ID and face perceivers, followed by a cross-modal adapter (illustrated in Fig. 4) for DiT-based video generation. By optimizing trainable modules marked by the flame, our method efficiently integrates facial features for controlled video synthesis while maintaining model efficiency.

extraction:

which implements a cross-modal full-attention paradigm coupled with layer-wise distribution modulation experts. This architectural choice introduces additional complexity in adapting to new conditions beyond simple cross-attention augmentation. Under this constraint, we find that modulation layers, despite requiring extremely few parameters, play a crucial role in learning the data distribution. This point is discussed experimentally in the Appendix B.2.

xface = τface(qface,f). (4)

Both perceivers τface and τid implement the standard QFormer architecture [31] with distinct query conditions in Eqs. (3) and (4). The identity branch and structural branch serve complementary roles in ensuring high-quality ID-preserving video generation. The identity branch maintains consistent identity features across frames, guaranteeing that the generated video preserves the subject’s identity throughout the sequence. Conversely, the structural branch captures fine-grained facial details and facilitates dynamic natural facial motions, effectively preventing structural collapse during movement.

Leveraging mm-DiT’s layer-wise modulation [64], we propose a lightweight adapter that incorporates additional facial conditions. As illustrated in Fig. 4, facial embedding xface in Eq. (4) is concatenated with text and video features (xtxt and xvid) to feed into the full selfattention. CogVideoX employs modal-specific modulation, where factors mvid and mtxt are applied to their respective modalities through adaptive normalization, where modulation factors are extracted from MLP extractors φ{txt, vid}. To accommodate the facial modality, we introduce a dedicated adaptive normalization module, normalizing facial features preceding the self-attention and feed-forward network (FFN). The corresponding set of modulation factors for the facial modality mface is computed by a MLP φface:

To control these aspects effectively, we employ distinct feature injection mechanisms. Identity information is incorporated into the text representation, building upon recent advancements in personalized text-to-image generation [34, 50]. Specifically, identity embeddings are projected into the text embedding space via a fusion MLP MLPfuse, xˆid = MLPfuse(xid,mxtxt), where xtxt denotes the input textual prompt embedding, m is a token-level binary mask that selectively applies fusion at identity-relevant tokens (e.g., “man”, “woman”) within xtxt. This process results in a fused id representation xˆid. The final adapted text embedding for DiT is computed as a masked replacement:

mface = {µ1face,σface1 ,γface1 ,µ2face,σface2 ,γface2 } = φface(t,l),

(6) where t denotes the time embedding and l represents the layer index. Here, µ is the shift parameter to adjust the feature mean, σ is the scale parameter to modulating the feature amplitude, and γ is the gating parameter to control the influence of the modulation. Within each block, let ϕn denote the in-block operations—where ϕ1 represents the self-attention operation and ϕ2 represents the FFN. The feature transformation after operation n is computed as: x¯n = xn−1 ∗ (1 + σn) + µn, then xn = x¯n + γnϕn(¯xn), with modality-specific subscripts omitted for brevity.

xˆtxt = mxˆid + (1 − m)xtxt, (5)

Meanwhile, the structural embedding xface is utilized as direct conditioning signals to guide the generative process.

#### 3.3. Conditioned Adaptive Normalization

Having obtained the decoupled ID-aware conditions cface, we address the challenge of efficiently integrating these conditions into the video diffusion transformer. Traditional Latent Diffusion Models, as exemplified by Stable Diffusion [48], utilize isolated cross-attention mechanisms for condition injection, which allow for straightforward adaptation to new conditions via decoupled cross-attention [19, 65]. However, our framework is based on mm-DiTs [64],

Furthermore, we explore whether layer-wise distribution conditioned by ID features would make ID injection more effective. To enhance the distribution learning capability of text and video latents from specific reference IDs, we

#### 3.4. Data and Training

Trainable

Shift, Scale Gate

Training a zero-shot customization adapter presents unique data challenges compared to fine-tuning approaches, like Magic-Me [39]. Our model’s full-attention architecture, which integrates spatial and temporal components inseparably, necessitates a two-stage training strategy. As shown in Fig. 5, we begin by training on diverse, high-quality datasets to develop robust identity preservation capabilities.

!&'#(- !)*)-

|!!,!"-./0<br><br>|
|---|

!!"#$ !,-,

"!"#$+ ",-,+ "./0+

$!"#$+ %!"#$+ $,-,+ %,-,+ $./0+ %./0+

!!

Attention

|!&'&+|
|---|

!!"#$+

!()*+

"!"#$% ",-,% "./0%

$!"#$% %!"#$% $,-,% %,-,% $./0% σ./0%

!"

FFN

|!!"#$%|
|---|

|!&'&%|
|---|

|!()*%|
|---|

Our progressive training pipeline leverages diverse datasets to enhance model performance, particularly in identity preservation. For image pre-training, we first utilize the LAION-Face [53] dataset, which contains web-scale real images and provides a rich source for generating selfreference images. To further increase identity diversity, we utilize the SFHQ [3] dataset, which applies self-reference techniques with standard text prompts. To prevent overfitting and promote the generation of diverse face-head motion, we use the FFHQ [27] dataset as a base. From this, we random sample text prompts from a prompt pool of human image captions, and synthesize ID-conditioned image pairs using PhotoMaker-V2 [34], ensuring both identity similarity and facial motion diversity through careful filtering.

!!"

Face embedding

ID-Conditioned Adaptive Norm

"+,-% ,

text & video modulation factors

")*+, ",!"

"#$%"

AdaLN Zero AdaLN Cond

time embedding

Time embedding

Pair-wise addition

ID modulation factors

"&'#(

AdaLN Face

Attention inside the selected blocks

|!!"#$%|
|---|

|!&'&%|
|---|

|!()*%|
|---|

…

copy Q1

QSA

|!! QQ1SA|
|---|

###### SA + CA

|K1 #<br><br>KSA|
|---|
|./ VV1SA|

|KK2CA|
|---|

#$./

|VV2CA|
|---|

output

- Figure 4. Cross-modal adapter in DiT blocks. Top: Cross-modal modulation in mmDiTs. Bottom: The Conditioned Adaptive Normalization (CAN) for modal-specific feature modulation and decoupled attention integration.

For video post-training, we leverage the high-quality Pexels and Mixkit datasets [1, 2], along with a small collection of self-collected videos from the web. Similarly, synthesized image data corresponding to each face reference of keyframes are generated as references. The combined dataset offers rich visual content for training the model across images and videos.

introduce Conditioned Adaptive Normalization (CAN), inspired by class-conditioned DiT [42] and StyleGAN’s [27] approach to control with conditions. Based on φ{txt, vid}, CAN predicts distribution shifts for video and text modalities with a trainable MLP φcond:

The objective function combines identity-aware and general denoising loss: L = Lnoise+λ(1 − cos(qface,D(x0))), where D(·) represents the latent decoder for the denoised latent x0, and λ is the balance factor. Following PhotoMaker [34], we compute the denoising loss specifically within the face area for 50% of random training samples.

mˆ vid,mˆ txt = φcond(t,l,µ1vid,xid). (7) Here, µ1vid acts as a distribution identifier for a better initialization of the CAN module, and xid from Eq. (3) represents the identity distribution prior. The final modulation factors are computed via residual addition: mvid = mˆ vid + φvid(t,l),mtxt = mˆ txt + φtxt(t,l).

### 4. Experiments 4.1. Implementation Details

Complementing the conditioned normalization, we augment the joint full self-attention TSA with a cross-attention mechanism TCA [20, 65] to further enhance the aggregation of ID modality features. The final attention output is computed as:

Dataset preparation. As illustrated in Fig. 5, our training pipeline leverages both self-referenced and synthetically paired image data [3, 27, 53] for identity-preserving alignment in the initial training phase. For synthetic data pairs (denoted as C and D in Fig. 5), we employ ArcFace [9] for facial recognition and detection to extract key attributes including age, bounding box coordinates, gender, and facial embeddings. Reference frames are then generated using PhotoMakerV2 [34]. We implement a quality control process by filtering image pairs {a,b} based on their facial embedding cosine similarity, retaining pairs where cos(qfacea ,qfaceb ) > 0.65, qface means the facial embedding. For text conditioning, we utilize MiniGemini-8B [33] to caption all video data, to form a diverse prompt pool con-

xout = TSA(Wq(xfull), Wkv(xface)) + TCA(Wq(xfull), Wˆkv(xface)), (8)

where xfull denotes the complete aggregated feature representation produced by concatenating or integrating text, video, and facial features from preceding layers. TSA and TCA utilize the same query projection Wq(xfull), while the key-value projections Wˆkv in cross-attention are reinitialized and trainable.

Dynamic Degree↑

Text Alignment↑

Inception Score↑

Motion Smoothness↓

Average ID Similarity↑

Similarity Decay↓

Face Motion FMref↑

Face Motion FMinter↑

Overall Preference↑

Models

DynamiCrafter [60] 0.455 0.168 8.20 0.507 0.896 0.002 0.237 0.287 5.402 EasyAnimate-I2V [63] 0.155 0.177 9.55 0.482 0.903 0.022 0.262 0.278 5.935 CogVideoX-I2V [64] 0.660 0.213 9.85 0.497 0.901 0.029 0.413 0.532 6.985 ID-Animator [20] 0.140 0.211 7.57 0.515 0.923 0.005 0.652 0.181 5.693 ConsisID [69] 0.615 0.236 11.09 0.513 0.913 0.002 0.652 0.601 6.640 MagicMirror 0.705 0.240 10.59 0.484 0.922 0.002 0.730 0.610 7.315

- Table 1. Quantitative comparisons. We report results with Image-to-Video and ID-preserved models. ID similarities are evaluated on the corresponding face-enhanced prompts to avoid face missing caused by complex prompts. Arrows indicate the direction of improved performance for each metric. We highlight the best and the second best results for each metric.

LAION-Face (50K) Web-scale real images

[Figure 14]

- A. Self-reference images C. Filtered synthesized image pairs

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Generated

Generated

FFHQ (70K + 132K) Synthesized pairs

Captioner

Samplefrom prompt pool

[Figure 19]

[Figure 20]

- B. Self-reference with standard text prompt

SFHQ(120K) Enhance ID diversity

Pexels, Mixkit (29K + 120K) High Quality Video

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

prompt pool

Generated

[Figure 27]

Generated

Generated

Video Captioner

D. Filtered synthesized video-image pairs

- Figure 5. Overview of our training datasets. The pipeline includes image pre-training data (A-C) and video post-training data (D). We utilize both self-reference data (A, B) and filtered synthesized pairs with the same identity (C, D). Numbers of (images + synthesized images) are reported.

taining 29K prompts, while CogVLM [57] provides video descriptions in the second training stage. Detailed data collection procedures are provided in Appendix A.1.

Training Details. Our MagicMirror framework extends CogVideoX-5B [64] by integrating facial-specific modal adapters into alternating DiT layers (i.e., adapters in all layers with even index l). We adopt the feature extractor Ffeat and ID perceiver τid from a pre-trained PhotoMakerV2 [34]. In the image pre-train stage, we optimize the adapter components for 30K iterations using a global batch size of 64. Subsequently, we perform video fine-tuning for 5K iterations with a batch size of 8 to enhance temporal consistency in video generation. Both phases employ a decayed learning rate starting from 10−5. All experiments were conducted on a single compute node with 8 NVIDIA A800 GPUs.

Evaluation and Comparisons. We evaluate our approach against the state-of-the-art ID-consistent video generation model ID-Animator [20], ConsisID [69] and leading Imageto-Video (I2V) frameworks, including DynamiCrafter [60],

CogVideoX-I2V [64], and EasyAnimate [63]. Our evaluation leverages standardized VBench [25], for video generation assessment that measures motion quality and textmotion alignment. For identity preservation, we utilize facial recognition embedding similarity [15] and facial motion metrics. Our evaluation dataset consists of 40 singlecharacter prompts from VBench, ensuring demographic diversity, and 40 action-specific prompts for motion assessment. Identity references are sampled from 50 face identities from PubFig [30], generating four personalized videos per identity across varied prompts.

#### 4.2. Quantitative Evaluation

The quantitative results are summarized in Tab. 1. We evaluate generated videos using VBench’s and EvalCrafter’s general metrics [25, 36], including dynamic degree, textprompt consistency, and Inception Score [52] for video quality assessment. We also evaluate on smoothness using cross-frame optical flow consistency. For identity preservation, we introduce Average Similarity, measuring the distance between generated faces and the average similarity of a group of reference images with the same identity. This prevents models from achieving artificially high scores through naive copy-paste behavior, as illustrated in Fig. 2. Face motion is quantified using two metrics: FMref (relative distance to the reference face) and FMinter (inter-frame distance), computed using RetinaFace [10] landmarks after position alignment, and the L2 distance between the normalized coordinates is reported as the metric.

Our method achieves superior facial similarity compared to I2V approaches while maintaining competitive performance to ID-Animator and ConsisID. We demonstrate strong text alignment, video quality, and dynamic performance, attributed to our decoupled facial feature extraction and cross-modal adapter with CAN.

Besides, we analyze facial similarity drop across uniformly sampled frames from each video to assess temporal identity consistency, reporting as the similarity decay term in Tab. 1. Standard I2V models (CogVideoX-I2V [64], EasyAnimate [63]) exhibit significant temporal decay in identity preservation. Although DynamiCrafter [60] shows better stability due to its random reference strategy, it com-

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

An elderly man, with a determined expression, stands in a sunlit gym, wearing a gray tank top, his muscles taut as he grips a heavy kettlebell. The room is filled with natural light streaming through large windows, casting shadows on the polished wooden floor. His face shows concentration and strength, highlighting his commitment to fitness. As he lifts the kettlebell with steady hands, the camera captures the sweat glistening on his brow, emphasizing his effort and resilience. The background features neatly arranged gym equipment, adding to the atmosphere of dedication and perseverance.

[Figure 32]

ConsisID

[Figure 33]

[Figure 34]

[Figure 35]

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

ID-animator

EasyAnimate-I2V

ID-Animator

|[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>MagicMirror<br><br>[Figure 50]|
|---|

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

DynamiCrafter

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

A serene woman, dressed in a cozy oversized sweater and jeans, kneels on a lush green meadow, gently petting a friendly golden retriever. The dog's tail wags enthusiastically, its fur gleaming in the soft sunlight. Her face lights up with a warm smile as her hand moves tenderly over the dog's head and back. In the background, a picturesque landscape of rolling hills and blooming wildflowers enhances the tranquil scene. The golden retriever, with its tongue lolling out and eyes full of affection, leans into her touch, creating a heartwarming moment of connection and joy.

[Figure 59]

ConsisID

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

ID-Animator

EasyAnimate-I2V

[Figure 73]

[Figure 74]

[Figure 75]

|[Figure 76]<br><br>[Figure 77]<br><br>[Figure 78]<br><br>[Figure 79]<br><br>MagicMirror<br><br>[Figure 80]|
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

CogVideoX-5B-I2V

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

A bearded man in his thirties, wearing a plaid shirt, sits at a rustic wooden bar, surrounded by an array of beer taps and vintage brewery decor. He carefully lifts a frosty pint glass filled with amber beer, examining its color and clarity against the warm, ambient lighting. He takes a slow, appreciative sip, his eyes closing momentarily as he savors the complex flavors. The camera captures the subtle smile of satisfaction on his face, highlighting the rich foam on his upper lip. The background hum of soft

[Figure 90]

chatterand clinking glassesaddsto the cozy, inviting atmosphere of the pub. ConsisID

[Figure 91]

[Figure 92]

[Figure 93]

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

EasyAnimate-I2V

ID-Animator

[Figure 104]

[Figure 105]

[Figure 106]

|[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]<br><br>[Figure 110]<br><br>MagicMirror<br><br>[Figure 111]|
|---|

[Figure 112]

DynamiCrafter

Figure 6. Qualitative comparisons. Captions and reference identity images are presented in the top-left corner for each case.

promises fidelity. Both ConsisID [69] and MagicMirror maintain consistent identity preservation throughout the video duration.

- 4.3. Qualitative Evaluation Beyond the examples shown in Fig. 1, we present comparative results in Fig. 6. Our method maintains high text coherence, motion dynamics, and video quality compared to conventional CogVideoX inference. When compared to existing image-to-video approaches [34, 60, 63, 64], Mag-

icMirror demonstrates superior identity consistency across frames while preserving natural motion. Our method also achieves enhanced dynamic range and text alignment compared to ID-Animator [20] and ConsisID [69], which exhibits limitations in motion variety and prompt adherence.

To complement our quantitative metrics, we conducted a comprehensive user study to evaluate the perceptual quality of generated results. The study involved 173 participants who assessed the outputs across four key aspects: motion

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

𝑥 only

𝑥 only

Both 𝑥 , 𝑥

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Reference

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

w/o CAN

w/o image pretrain

Full MagicMirror

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

𝑥 only

𝑥 only

Both 𝑥 , 𝑥

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Reference

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

w/o CAN

w/o image pretrain

Full MagicMirror

Figure 7. Examples for ablation studies on modules and training strategies.

Visual Quality↑

Text Alignment↑

Dynamic Degree↑

ID Similarity↑

Models

DynamiCrafter [60] 6.03 7.29 4.85 5.87 EasyAnimate-I2V [63] 6.62 8.21 5.57 6.01 CogVideoX-I2V [64] 6.86 8.31 6.55 6.22 ID-Animator [20] 5.63 6.37 4.06 6.70 ConsisID [69] 6.43 8.35 6.23 5.55 MagicMirror 6.97 8.88 7.02 6.39

- Table 2. User study results. We highlight the best and the second best results for each metric.

Exp. xˆid xface mface mˆ Pretrain txt-align↑ FMinter↑ ID↑

- A [identity branch] ✓ 0.238 0.572 0.865

- B [structural branch] ✓ 0.240 0.584 0.869

- C [dual branch] ✓ ✓ 0.239 0.654 0.870

- D [Eq.6, mface] ✓ ✓ 0.241 0.563 0.872

- E [Eq.7, mˆ txt, mˆ vid] ✓ ✓ 0.242 0.696 0.875

- F [w/o CAN] ✓ ✓ ✓ 0.236 0.568 0.886

- G [w/o pretrain] ✓ ✓ ✓ ✓ 0.241 0.559 0.883 Full [MagicMirror] ✓ ✓ ✓ ✓ ✓ 0.240 0.665 0.911

- Table 3. Ablation study results on the same training scale across multiple settings. Some modules are interdependent.

dynamics, text-motion alignment, video quality, and identity consistency. Participants rated each aspect on a scale of 1-10, with results summarized in Tab. 2. As shown in the overall preference scores in Tab. 1, MagicMirror consistently outperforms baseline methods across most evaluated dimensions, demonstrating its superior perceptual quality in human assessment. Regarding video quality and ID similarity, we observed a gap between designed metrics and human-evaluated perceptual evaluation in ConsisID [69].

- 4.4. Ablation Studies

Condition-related Modules. We evaluate our key modules through ablation studies, shown in Tab. 3 and Fig. 7. To ensure a fair comparison, all ablations are conducted on the same training scale, with a half training iteration of the official setting. Experiments using single-branch facial embedding (Exp. A: identity branch only, Exp. B: structural branch only) exhibit a limited identity preservation (0.8650.869 ID similarity). In contrast, the dual-branch strategy (Exp. C) synergizes their complementary strengths, and achieves a higher motion metric. The Conditioned Adaptive Normalization (CAN) proves vital for distribution align-

ment, enhancing identity preservation across frames. The effectiveness of CAN for facial condition injection is further demonstrated in Exp. D, E, and F. Notably, complete CAN removal (Exp. F) causes significant performance degradation (0.911 → 0.886), underscoring its necessity for effective identity injection. An extended analysis of the CAN’s benefits for the training convergence and distribution alignment is provided in the Appendix B.1-B.2.

Training Strategy. Exp. G in Tab. 3 and Fig. 7 also illustrate the impact of different training strategies. Image pretraining is essential for robust identity preservation, while video post-training ensures temporal consistency. Our twostage training approach achieves optimal results by leveraging the advantages of both phases, generating high ID fidelity videos with dynamic facial motions. Appendix B.3 discusses more details about the training strategy.

### 5. Conclusion

In this work, we presented MagicMirror, a zero-shot framework for ID-preserving video generation. MagicMirror incorporates dual facial embeddings and Conditional Adaptive Normalization (CAN) into DiT-based architectures. Our approach enables robust identity preservation and stable training convergence. Extensive experiments demonstrate that MagicMirror generates high-quality personalized videos while maintaining identity consistency from a single reference image, outperforming existing methods across multiple benchmarks and human evaluations.

Limitations. While MagicMirror excels at ID-consistent video generation, challenges remain in supporting multiple identities and preserving fine-grained attributes beyond facial features, such as clothing, improvements necessary for practical customized video applications.

Acknowledgment. The study was supported in part by the Research Grants Council under the Areas of Excellence scheme grant AoE/E-601/22-R, Hong Kong General Research Fund (14208023), Hong Kong AoE/P-404/18, and the Center for Perceptual and Interactive Intelligence (CPII) Ltd under InnoHK supported by the Innovation and Technology Commission.

### Appendix

This appendix provides comprehensive technical details and additional results for MagicMirror, encompassing dataset preparation, architectural specifications, implementation, and extensive experimental validations. We include additional qualitative results and in-depth analyses to support our main findings. We strongly encourage readers to examine the project page https://julianjuaner. github.io/projects/Magic-Mirror/ for dynamic video demonstrations. The following contents are organized for efficient navigation.

### Appendix Contents

- A. Experiment Details 9

- A.1. Training Data Preparation . . . . . . . . . . . 9
- A.2. Test Data Preparation . . . . . . . . . . . . . 9
- A.3. Comparisons . . . . . . . . . . . . . . . . . 9
- A.4. Evaluation Metrics . . . . . . . . . . . . . . 11
- A.5. Implementation Details . . . . . . . . . . . . 12

- B. Additional Discussions 12

- B.1. Advantages of CAN . . . . . . . . . . . . . . 12
- B.2. Distribution Analysis and Its Impact . . . . . 13
- B.3. Two-Stage Training Analysis . . . . . . . . . 13
- B.4. Limitation Analysis . . . . . . . . . . . . . . 13

- C. Additional Results & Applications 14

- C.1. Additional Applications . . . . . . . . . . . . 14
- C.2. Image Generation Results . . . . . . . . . . . 14
- C.3. Video Generation Results . . . . . . . . . . . 15

- D. Acknowledgments 15 A. Experiment Details

#### A.1. Training Data Preparation

Our training dataset is constructed through a rigorous preprocessing pipeline, as illustrated in Fig. 8. For the image pretrain data, we start downloading 5 million images from LAION-face [53], then undergo strict quality filtering based on face detection confidence scores and resolution requirements. The filtered subset of 107K images is then processed through an image captioner [33], where we exclude images containing texts. This results in a curated set of 50K highquality face image-text pairs. To enhance identity diversity, we incorporate the synthetic SFHQ dataset [3]. To fit the model output, we standardize these images by adding black borders and pairing them with a consistent prompt template: ”A squared ID photo of ..., with pure black on two sides.” This preprocessing ensures uniformity while maintaining the dataset’s diverse identity characteristics.

For FFHQ [27], we leverage a state-of-the-art identitypreserving prior PhotoMakerV2 [34] to generate synthetic images of the same identity, but with different face poses.

We filter redundant identities using pairwise facial similarity metrics, with prompts sampled from our 50K video keyframe captions. We use the Pexels-400K and Mixkit datasets from [35] for construction of image-video pairs. The videos undergo a systematic preprocessing pipeline, including face detection and motion-based filtering to ensure high-quality dynamic content. We generate video descriptions using CogVLM video captioner [57]. Following our FFHQ processing strategy, we employ PhotoMakerV2 to synthesize identity-consistent images from the detected faces, followed by quality-based filtering.

#### A.2. Test Data Preparation

Face Images Preparation We construct a comprehensive evaluation set for identity preservation assessment across video generation models. Our dataset comprises 50 distinct identities across seven demographic categories: man, woman, elderly man, elderly woman, boy, girl, and baby. The majority of faces are sourced from PubFig dataset [30], supplemented with public domain images for younger categories. Each identity is represented by 1-4 reference images to capture variations in pose and expression.

Prompt Preparation Our test prompts are derived from VBench [25], focusing on human-centric actions. For detailed descriptions, we sample from the initial 200 GPT4-enhanced prompts and select 77 single-person scenarios. Each prompt is standardized with consistent subject descriptors and augmented with the img trigger word for model compatibility. We assign four category-appropriate prompts to each identity, ensuring demographic alignment. For the ”baby” category, which lacks representation in VBench, we craft four custom prompts to maintain evaluation consistency across all categories.

#### A.3. Comparisons

ID-Animator [20] We utilize enhanced long prompts for evaluation, although some undergo partial truncation due to CLIP’s 77-token input constraint.

In our main comparisons Tab. 1, we evaluated IDAnimator at a resolution of 480×720. This choice was made to ensure that SD-based ID-Animator comparisons used matching resolutions, thereby ensuring equal content capacity—a decision justified by the inherent resolution independence of the UNet architecture. To provide a fair and comprehensive evaluation, we additionally present some results at the default 512×512 resolution here in Tab. 4. These results confirm that our comparisons remain robust and consistent across different resolution settings.

Method Base Resolution txt-align↑ FMinter↑ ID↑ Smooth↓

ID-Animator SD1.5 (480, 720) 0.211 0.181 0.923 0.515 ID-Animator SD1.5 (512, 512) 0.217 0.179 0.921 0.501 MagicMirror CogVideoX (480, 720) 0.240 0.610 0.922 0.484

###### Table 4. ID-Animator resolution comparison.

Discard if

- (1) Resolution lower than 720P
- (2) Face is too small and not recognized

[Figure 171]

[Figure 172]

LAION-Face (50K) Web-scale real images

Filtering: Resolution + Face Detector, Crop

Discard if

Web

- (1) Specific ‘text’ in the image caption
- (2) No word related to identity

Download From Dataset (~5M)

Captioner (MiniGemini-8B)

[Figure 173]

|[Figure 174]|
|---|

[Figure 175]

SFHQ(120K) Enhance ID diversity standard text promptSelf-reference with

“A squared ID photo of a <man/woman/child…> with pure black on two sides”

[Figure 176]

Image from Dataset

[Figure 177]

FFHQ (70K + 139K) Generated Synthesized pairs

“A young woman img is smiling and looking to the camera.”

Discard if similarity is low

[Figure 178]

Filtering: Face Similarity

[Figure 179]

PhotoMakerV2

“A young woman img is standing in front of the Christmas tree,”

Prompt Pool

Get pairs for image pre-train

Image from Dataset

Generated

Sample from prompt pool

- (1) Get pairs for image pre-train
- (2) Get image for ID-reference of the video

[Figure 180]

Pexels, Mixkit (29K + 132 K) High Quality Video

Reference face from key-frames

[Figure 181]

[Figure 182]

[Figure 183]

PhotoMakerV2

[Figure 184]

[Figure 185]

Generated

Filtering: Face Detector, Clipping, Motion score…

Also used in FFHQ Filtering:

Prompt Pool

Web

pairwise Face Similarity

Download From Dataset (~600K)

Image Captioner (MiniGemini-8B) Video Captioner (CogVLM)

Discard if similarity is low

Discard unqualified videos

- Figure 8. Detailed training data processing pipeline. Building upon Fig. 5, we illustrate comprehensive filtering criteria, prompt examples, and processing specifications. The data flow is indicated by blue arrows, while filtering rules leading to data exclusion are marked with red arrows.

ConsisID [69] We utilize the CogVideoX-5B[64] version. Its base inference settings are aligned with those of our model, and enhanced long prompts are employed to fully leverage its capabilities.

CogVideoX-5B-I2V experiments.

DynamiCrafter [60] Due to model-specific resolution requirements, we create a dedicated set of reference images using PhotoMakerV2 that conform to the model’s specifications.

CogVideoX-5B-I2V [64] For this image-to-video variant, we first generate reference images using PhotoMakerV2 [34] for each prompt-identity pair. These images, combined with enhanced long prompts, serve as input for video generation.

In image-to-video baselines, through reference images generated by enhanced prompts, we deliberately use original short concise prompts for video generation. This choice stems from our empirical observation that image-to-video models exhibit a strong semantic bias when processing lengthy prompts. Specifically, these models tend to prior-

EasyAnimate [63] We evaluate using the same PhotoMakerV2-generated reference images as in our

###### CogVideoX-I2V：

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

A man playing golf

|[Figure 192]|
|---|

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

A focused man stands on a lush, emerald-green fairway, wearing a crisp white polo shirt, beige trousers, and a navy cap, with the sun casting a warm glow over the rolling hills. He is playing golf. The camera captures a close-up of their hands gripping the club, showcasing the precision and concentration in their stance. As he swing, the club arcs gracefully through the air, sending the golf ball soaring against a backdrop of clear blue sky and distant trees. The scene shifts to the golfer watching intently as the ball lands on the manicured green, the flag fluttering gently in the breeze, embodying the serene yet competitive spirit of the game.

EasyAnimate-I2V：

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

A shirtless man climbing

|[Figure 204]|
|---|

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

A shirtless man with a lean, muscular build ascends a rugged cliff face, his skin glistening with sweat under the bright sun. His determined expression and focused gaze reveal his concentration and skill as he navigates the challenging rock formations. The camera captures the intricate details of his movements, highlighting the tension in his muscles and the precision of his grip. The backdrop of the scene is a vast, open sky, with the distant horizon hinting at the expansive landscape below. As he climbs higher, the play of light and shadow across the rock surface adds depth and drama to the breathtaking ascent.

- Figure 9. Impact of prompt length on image-to-video generation. We demonstrate how image-to-video models perform differently with concise versus enhanced prompts. Frames with large artifacts are marked in red. First frame images are generated from enhanced prompts.

itize text alignment over reference image fidelity, leading to degraded video quality and compromised identity preservation. This trade-off is particularly problematic for our face preservation objectives. We provide visual evidence of this phenomenon in Fig. 9.

#### A.4. Evaluation Metrics

Our evaluation framework combines standard video metrics with face-specific measurements. From VBench [25], we utilize Dynamic Degree for motion naturality and Overall Consistency for text-video alignment. Video quality is assessed using Inception Score from EvalCrafter [36]. For facial fidelity, we measure identity preservation using facial recognition embedding similarity [15] and temporal stability through frame-wise similarity decay.

We propose a novel facial dynamics metric to address the limitation of static face generation in existing methods. As shown in Fig. 10, we extract five facial landmarks using RetinaFace [10] and compute two motion scores: FMref measures facial motion relative to the reference image (computed on aspect-ratio-normalized frames to eliminate positional bias), while FMinter quantifies maximized inter-

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Find landmarks

|[Figure 214]|
|---|

Resize

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Find landmarks

FM!"# = 1.46

Resize

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

|[Figure 226]|
|---|

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Find landmarks

Resize

[Figure 231]

[Figure 232]

FM!"# = 0.28

Figure 10. Face Motion (FM) calculation. FMinter follows a similar computation across consecutive video frames.

frame facial motion (computed on original frames to preserve translational movements). This dual-score approach enables a comprehensive assessment of facial dynamics.

Success rate & failcase analysis. Success rate metrics better demonstrate reliability. Our additional experiments with

Trainable

Shift, Scale Gate

!&'#(- !)*)-

|!!,!"-./0<br><br>|
|---|

!!"#$ !,-,

"!"#$+ ",-,+ "./0+

$!"#$+ %!"#$+ $,-,+ %,-,+ $./0+ %./0+

!!

Attention

|!&'&+|
|---|

!!"#$+

!()*+

"!"#$% ",-,% "./0%

$!"#$% %!"#$% $,-,% %,-,% $./0% σ./0%

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

!"

FFN

|!!"#$%|
|---|

|!&'&%|
|---|

|!()*%|
|---|

!!"

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Face embedding

ID-Conditioned Adaptive Norm

w/o CAN

"+,-% ,

text & video modulation factors

")*+, ",!"

"#$%"

AdaLN Zero AdaLN Cond

[Figure 241]

time embedding

Time embedding

Pair-wise addition

ID modulation factors

[Figure 242]

"&'#(

AdaLN Face

Reference Ground Truth

Inside the layer %

Attention inside the selected blocks

w/ CAN

Time embeddingt

!&'&%

|!!"#$% !|
|---|

|c]!()*%|
|---|

[2,

!"

…

copy Q1

Q1 K1 V1

Iterations:

[c2]

1K 3K 5K 16K

##### SA + CA

[2c]

concatenation

|K2|
|---|

V2

Figure 12. CAN speeds up the convergence. Without the Conditioned Adaptive Normalization, the model cannot fit the simplest appearance features like hairstyle in the image pre-train stage.

AdaLN #!"#，#$%$(MLP)

Dimension projection (Linear)

output

!!"#,!$%$

[12,c3]

[c1]

"#!"$

cial modulation factors mface, we employ a two-layer MLP architecture, following the implementation structure of the

concatenation Pairwise

[c1+ c2+ c3]

Addition !$!"#,!$$%$

|Conditioned AdaLN #&'(# (MLP)|
|---|

original normalization modules φ{vid, text}. The detailed implementation of CAN is illustrated in Fig. 11. Given the

"##!"$ [12,c3]

facial ID embedding xid ∈ R2×c containing two tokens, we first apply one global projection layer for dimensionality reduction, mapping it to dimension c1. Subsequently, in each adapted layer, we concatenate this projected embedding with the time embedding t and the predicted shift factor µ1vid along the channel dimension. An MLP then processes this concatenated representation to produce the final modulation factors. To ensure stable training, all newly introduced modulation predictors are initialized with zero.

Figure 11. Detailed implementation of Conditioned Adaptive Normalization. We present the expanded architecture of φcond (illustrated in the unmasked region above) with comprehensive annotations of input-output tensor dimensions at each transformation.

200 videos (50 prompts × 4 seeds) compared MagicMirror with identity preservation baselines on success rates for face recognition, identity check, motion, and text alignment. MagicMirror achieves improved SR across most dimensions, though failcase analysis reveals motion quality remains the primary limitation, which is predominantly model-dependent.

We also tried to directly use the prediction of CAN as the data distribution, this results in a bad initialization, comparing with the residual prediction, direct prediction leads to abnormal video generation quality.

### B. Additional Discussions

Method / SR motion quality text align face recongized identity check average

#### B.1. Advantages of CAN

ID-Animator 11.5% 60.0% 93.5% 82.5% 61.9% ConsisID 38.5% 73.5% 89.5% 74.5% 69.0% MagicMirror 44.0% 75.5% 98.0% 81.0% 74.6%

The benefits of CAN in facial condition injection are evident in its ability to enhance training convergence, particularly during the image pre-train stage. As illustrated in Fig. 12, models equipped with CAN achieve significantly improved identity information capture, enabling faster adaptation to appearance attributes. This acceleration in convergence highlights CAN’s effectiveness in preserving identity consistency throughout the training process.

Table 5. Success rate comparison.

#### A.5. Implementation Details

Decoupled Facial Embeddings. Our architecture employs two complementary branches: an ID embedding branch based on pre-trained PhotoMakerV2 [34] with two-token ID-embedding query qid, and a facial structural embedding branch that extracts detailed features from the same ViT’s penultimate layer. The latter initializes 32 token embeddings as facial query qface input. We use a projection layer to align facial modalities before diffusion model input.

Furthermore, we specifically design CAN and related modules to be lightweight and avoid altering any pretrained weights of the video DiT, thereby preserving the original model capacity. We evaluate GPU memory utilization, parameter count, and inference latency for generating a 49-frame 480P video. Compared to the baseline model, the additional parameters introduced by MagicMirror are primarily concentrated in the embedding extraction stage, which requires only a single forward pass. As summarized in Tab. 6, compared with ConsisID [69] and

Conditioned Adaptive Normalization. This paragraph elaborates on the design details of the Conditioned Adaptive Normalization (CAN) module, complementing the overview provided in Sec. 3.3 and Fig. 4. For predicting fa-

Norm Layers Finetuning on CelebV-Text

[Figure 243]

[Figure 244]

[Figure 245]

!!"#$ !!"#$ + !#!"#$ !%&%$ !%&%$ + !#%&%$ !'()*$

The same layer index

[Figure 246]

[Figure 247]

[Figure 248]

Figure 13. Different modalities’ scale distribution using t-SNE. Each point represents the scale with a unique timestep-layer index. We also illustrate a shift variant on text and video’s adaptive scale using different colors.

Norm Layers Finetuning on Pexels

[Figure 249]

[Figure 250]

[Figure 251]

CogVideoX [64] baseline, MagicMirror introduces minimal computational overhead, with only a slight increase in GPU memory consumption and inference time.

[Figure 252]

[Figure 253]

[Figure 254]

Model Video size Memory Params. Latency Batch×Iter. Data (I+V) GPU ID-Animator (16,512,512) 8.4 GiB 1.52B 11s 2×58K 0K+13K A100*1 CogVideoX-5B (49,480,720) 24.9 GiB 10.5B 204s (0.1-2K)×750K 2B+35M CogVideoX-I2V (49,480,720) 25.9 GiB 10.6B 213s - - ConsisID (49,480,720) 41.5 GiB 11.1B 213s 80×1.8K 0K+130K H100*40 MagicMirror (49,480,720) 28.6 GiB 12.8B 209s 8×9K* 570K +29K A800*8

Figure 14. Modulation layers reflect data distribution. Finetuning solely the modulation layer weights demonstrates adaptation to distinct data distributions, affecting both spatial fidelity and temporal dynamics.

Table 6. Computation overhead of MagicMirror. All computations are measured on the one A800 GPU.

namic motion score from 0.71 to 0.84. This result, similar to Experiments E-F in Tab. 3, further verifies the impact of the modulation module on dynamic facial motion.

#### B.2. Distribution Analysis and Its Impact

We begin by visualizing the predicted modulation scale factors σ using t-SNE [54] in Fig. 13. The results show that distinct modalities occupy characteristic distributions across different Transformer layers, and these distributions appear largely invariant to the specific timestep input. In particular, the face modality exhibits a unique pattern, while the conditioned residual σˆ introduces targeted shifts away from the baseline distribution. This shift empirically accelerates model convergence when incorporating ID conditions.

#### B.3. Two-Stage Training Analysis

In Fig. 15, we present additional ablation results that clarify how each training phase addresses a distinct aspect of identity-preserving video generation. Specifically, the image pre-training phase prioritizes robust identity encoding, ensuring that facial features remain consistent and accurately captured. However, training exclusively on image data leads to color-shift artifacts during video inference, caused by modulation factor inconsistencies across different training stages. By combining these two stages, our final approach aligns both identity representation and color distribution, resulting in dynamic and high-fidelity ID-preserving videos without the artifacts observed in single-stage alternatives.

Beyond the t-SNE visualization, we further investigate the critical role of distribution alignment by examining how modality-aware data distributions affect generation quality. Specifically, we fine-tuned only the normalization layers φvid,φtxt of the CogVideoX base model on two distinct datasets—CelebV-Text [66] and our Pexels video collection [2]. As illustrated in Fig. 14, this distribution-specific fine-tuning exerts a substantial influence on the spatial fidelity of generated videos. These observations underscore the importance of aligning modality distributions during training, and they also validate the high quality of our curated video dataset.

#### B.4. Limitation Analysis

As discussed in Sec. 5, our approach faces several limitations, particularly in handling multi-person scenarios and preserving fine-grained features. Fig. 16 illustrates two representative failure cases: incomplete transfer of reference character details (such as accessories) and motion artifacts caused by the base model. These limitations highlight critical areas for future research in controllable personalized

Additionally, we conducted another experiment using our Pexels dataset. We found that by using a dataset with twice the frame rate and training only the modulation layers, we achieved an improvement in the VBench [25] dy-

[Figure 256]

A woman as the main subject. She appears to be outdoors, as suggested by the natural lighting and the blurred background that suggests a natural setting. The woman has dark hair and is wearing sunglasses on top of her head. She is dressed in a patterned top with a blue and white scheme…

A young woman with curly hair and a professional demeanor is seen engaging in a phone conversation while seated at a desk. She wears a black headset with a microphone, a dark blazer, and a necklace with a heart pendant…

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 262]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 268]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

w/o facial embedding w/o adaptive condition Full model w/o video finetune w/o image pretrain Full model

- Figure 15. Examples for ablation studies on training strategies.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

(b) Video Quality:MotionArtifacts

(a) Fine-grained Feature Missing

- Figure 16. Limitations of MagicMirror. (a) Fine-grained feature preservation failure in facial details and accessories. (b) Motion artifacts in generated videos showing temporal inconsistencies.

video generation, particularly in maintaining temporal consistency and fine detail preservation.

### C. Additional Results & Applications

#### C.1. Additional Applications

Fig. 17 demonstrates two extended capabilities of MagicMirror. First, beyond realistic customized video generation, our framework effectively handles stylized prompts,

leveraging CogVideoX’s diverse generative capabilities to produce identity-preserved outputs across various artistic styles and visual effects. Furthermore, we show that our method can generate high-quality, temporally consistent multi-shot sequences when maintaining coherent character and style descriptions. We believe these capabilities have significant implications for automated video content creation.

###### “The synthwave retro, 80s style, sunset colors style, an elderly woman img is reading book.”

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

###### “Art nouveau, organic curves, floral patterns style a male police officer talking on the radio”

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

(a) Videos with style-specific prompts

[Figure 297]

A serene womanwithdelicate features,wearinga flowing whiteblouse

- • practices gentle yoga stretches…
- • sits at her kitchen counter bathed in morning light. ..
- • is working at her writing desk near a window…

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

(b) multi-shot videos with consistent character prompts

Figure 17. Additional applications of MagicMirror. We can generate identity-preserved videos across artistic styles and can generate multi-shot videos with consistent characters. More results are presented in the project page.

#### C.2. Image Generation Results

MagicMirror demonstrates strong capability in IDpreserving image generation with the image-pre-trained stage. Notably, it achieves even superior facial identity fidelity compared to video-finetuned variants, primarily due to optimization constraints in video training (e.g., limited

batch sizes and dataset scope). Representative examples are presented in Fig. 18.

#### C.3. Video Generation Results

Additional video generation results and comparative analyses are provided in Figs. 19 and 20, highlighting our method’s advantages. Fig. 19 specifically demonstrates the benefits of our one-stage approach over I2V, including superior handling of occluded initial frames, enhanced dynamic range, and improved temporal consistency during facial rotations. In Fig. 20, we provide more results with human faces on different scales.

### D. Acknowledgments

Social Impact. MagicMirror is designed to facilitate creative and research-oriented video content generation while preserving individual identities. We advocate for responsible use in media production and scientific research, explicitly discouraging the creation of misleading content or violation of portrait rights. As our framework builds upon the DiT foundation model, existing diffusion-based AI-content detection methods remain applicable.

Data Usage. The training data we used is almost entirely sourced from known public datasets, including all image data and most video data. All video data was downloaded and processed through proper channels (i.e., download requests). We implement strict NSFW filtering during the training process to ensure content appropriateness.

###### Figures 18-20 are presented on the following pages ↓

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

Ref-ID Generated Images Ref-ID Generated Images

- Figure 18. Image generation using MagicMirror. Model in the image pre-train stage captures ID embeddings of the reference ID (RefID), yet over-fits on some low-level distributions such as image quality, style, and background.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

Camera Motion

Dynamic Facial Movement

Camera Motion + Dynamic Facial Movement

- Figure 19. Advantages over I2V generation. MagicMirror successfully handles challenging scenarios including partially occluded initial frames and maintains identity consistency through complex facial dynamics, addressing limitations of traditional I2V approaches.

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

Ref-ID Generated Videos

- Figure 20. Video generation results. We demonstrate MagicMirror’s capability across varying facial scales and compositions. Additional examples and comparative analyses are available in the project page.

### References

- [1] Mixkit: Free assets for your next video project. 5
- [2] Pexels: Free 4k stock videos & full hd video clips to download. 5, 13
- [3] David Beniaguev. Synthetic faces high quality (sfhq) dataset,

2022. 5, 9

- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. Technical report, OpenAI, 2023. 1
- [5] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023. 1, 3
- [6] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated mri. Medical image analysis, 80: 102479, 2022. 2
- [7] Hyungjin Chung, Byeongsu Sim, and Jong Chul Ye. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction. In CVPR, pages 12413–12422, 2022. 2
- [8] Gabriele Corso, Hannes St¨ark, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock: Diffusion steps, twists, and turns for molecular docking. ICLR, 2023. 2
- [9] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, pages 4690–4699, 2019. 3, 5
- [10] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-level face localisation in the wild. In CVPR, pages 5203–5212,

- 2020. 6, 11

[11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 34:8780–8794,

- 2021. 2

- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021. 3
- [13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, pages 12873–12883, 2021. 2
- [14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. ICLR, 2023. 3
- [15] Adam Geitgey. face recognition, 2017. 6, 11

- [16] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. ICLR, 2022. 2
- [17] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and

- Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 2, 3
- [18] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. ICLR, 2024. 1, 3
- [19] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. NeurIPS, 2024. 1, 3, 4
- [20] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, Man Zhou, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024. 1, 2, 3, 5, 6, 7, 8, 9
- [21] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 3

- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. 1, 2
- [23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 35:8633–8646, 2022. 2
- [24] Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal fine-grained identity preserving. In arXiv preprint arXiv:2404.16771, 2024. 1, 3
- [25] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024. 2, 6, 9, 11, 13
- [26] Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, and Tommi Jaakkola. Torsional diffusion for molecular conformer generation. NeurIPS, 35:24240–24253, 2022. 2
- [27] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, pages 4401–4410, 2019. 2, 3, 5, 9
- [28] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, pages 8110–8119,

2020. 2

- [29] Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. NeurIPS, 34:21696– 21707, 2021. 2
- [30] Neeraj Kumar, Alexander C Berg, Peter N Belhumeur, and Shree K Nayar. Attribute and simile classifiers for face verification. In ICCV, pages 365–372. IEEE, 2009. 6, 9
- [31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 4
- [32] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. NeurIPS, 35:4328–4343, 2022. 2

- [33] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 5, 9

- [34] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, MingMing Cheng, and Ying Shan. Photomaker: Customizing realistic human photos via stacked id embedding. In CVPR, pages 8640–8650, 2024. 1, 2, 3, 4, 5, 6, 7, 9, 10, 12
- [35] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, Tanghui Jia, Junwu Zhang, Zhenyu Tang, Yatian Pang, Bin She, Cen Yan, Zhiheng Hu, Xiaoyi Dong, Lin Chen, Zhang Pan, Xing Zhou, Shaoling Dong, Yonghong Tian, and Li Yuan. Open-sora plan: Open-source large video generation model, 2024. 3, 9
- [36] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In CVPR, pages 22139– 22149, 2024. 6, 11
- [37] LuChen. Video ocean - filmmaking for everyone. 2
- [38] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 3
- [39] Ze Ma, Daquan Zhou, Chun-Hsiao Yeh, Xue-She Wang, Xiuyu Li, Huanrui Yang, Zhen Dong, Kurt Keutzer, and Jiashi Feng. Magic-me: Identity-specific video customized diffusion. arXiv preprint arXiv:2402.09368, 2024. 1, 3, 5
- [40] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. PMLR, 2021. 2
- [41] OpenAI. Video generation models as world simulators. 3
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 2, 3, 5
- [43] Bohao Peng, Jian Wang, Yuechen Zhang, Wenbo Li, MingChang Yang, and Jiaya Jia. Controlnext: Powerful and efficient control for image and video generation. arXiv preprint arXiv:2408.06070, 2024. 3
- [44] Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. Portraitbooth: A versatile portrait model for fast identity-preserved personalization. In CVPR, pages 27080–27090, 2024. 1, 3
- [45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763. PMLR, 2021. 3
- [46] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pages 8821–

8831. Pmlr, 2021. 2

- [47] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding

- in style: a stylegan encoder for image-to-image translation. In CVPR, pages 2287–2296, 2021. 3
- [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 1, 2, 4
- [49] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241. Springer, 2015. 2
- [50] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023. 2, 3, 4
- [51] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022. 1
- [52] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. NeurIPS, 29, 2016. 6
- [53] Christoph Schuhmann, Robert Kaczmarczyk, Aran Komatsuzaki, Aarush Katta, Richard Vencu, Romain Beaumont, Jenia Jitsev, Theo Coombes, and Clayton Mullis. Laion400m: Open dataset of clip-filtered 400 million image-text pairs. In NeurIPS Workshop Datacentric AI, number FZJ2022-00923. J¨ulich Supercomputing Center, 2021. 5, 9
- [54] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. JMLR, 9(11), 2008. 2, 13
- [55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017. 3
- [56] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 1, 3
- [57] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, et al. Cogvlm: Visual expert for pretrained language models. arXiv preprint arXiv:2311.03079, 2023. 6, 9
- [58] Xintao Wang, Yu Li, Honglun Zhang, and Ying Shan. Towards real-world blind face restoration with generative facial prior. In CVPR, pages 9168–9178, 2021. 3
- [59] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH, pages 1–11, 2024. 3
- [60] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. ECCV, 2024. 1, 3, 6, 7, 8, 10
- [61] Jinbo Xing, Long Mai, Cusuh Ham, Jiahui Huang, Aniruddha Mahapatra, Chi-Wing Fu, Tien-Tsin Wong, and Feng Liu. Motioncanvas: Cinematic shot design with controllable image-to-video generation. arXiv preprint arXiv:2502.04299, 2025. 3

- [62] Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. A survey on video diffusion models. ACM Computing Surveys, 2023. 2
- [63] Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Yunkuo Chen, Bo Liu, MengLi Cheng, Xing Shi, and Jun Huang. Easyanimate: A high-performance long video generation method based on transformer architecture. arXiv preprint arXiv:2405.18991,

2024. 1, 3, 6, 7, 8, 10

- [64] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2, 3, 4, 6, 7, 8, 10, 13
- [65] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

2023. 3, 4, 5

- [66] Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu. Celebv-text: A large-scale facial text-video dataset. In CVPR, pages 14805–14814, 2023. 2, 13
- [67] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. ICLR, 2

(3):5, 2024. 2

- [68] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 3
- [69] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identitypreserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024. 1, 3, 6, 7, 8, 10, 12
- [70] Yan Zeng, Guoqiang Wei, Jiani Zheng, Jiaxin Zou, Yang Wei, Yuchen Zhang, and Hang Li. Make pixels dance: Highdynamic video generation. In CVPR, 2024. 3
- [71] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3836–3847, 2023. 3
- [72] Yuechen Zhang, Jinbo Xing, Bin Xia, Shaoteng Liu, Bohao Peng, Xin Tao, Pengfei Wan, Eric Lo, and Jiaya Jia. Training-free efficient video generation via dynamic token carving. arXiv preprint arXiv:2505.16864, 2025. 3
- [73] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 3

