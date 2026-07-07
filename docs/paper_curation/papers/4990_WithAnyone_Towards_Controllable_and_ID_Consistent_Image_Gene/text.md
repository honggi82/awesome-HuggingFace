[Figure 1]

### WithAnyone: Towards Controllable and ID Consistent Image Generation

# arXiv:2510.14975v1[cs.CV]16Oct2025

Hengyuan Xu1,2 Wei Cheng2,† Peng Xing2 Yixiao Fang2 Shuhan Wu2 Rui Wang2 Xianfang Zeng2 Daxin Jiang2 Gang Yu2,‡ Xingjun Ma1,‡ Yu-Gang Jiang1

1 Fudan University 2 StepFun

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

###### Project Page MultiID-2M MultiID-Bench Models Code

|[Figure 7]| | | | |[Figure 8]| | |[Figure 9]| | | |[Figure 10]| | | |[Figure 11]| |[Figure 12]| | | | |[Figure 13]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 14]| | | |[Figure 15]| | | | | | | | | |[Figure 16]<br><br>[Figure 17]| | | | |[Figure 18]| | |[Figure 19]| | |
|[Figure 20]| | | | | | | | |[Figure 21]| | | | | | | | | | |[Figure 22]<br><br>[Figure 23]| | | | |
|[Figure 24]|[Figure 25]| | | | | |[Figure 26]| | | |[Figure 27]| |[Figure 28]| | | |[Figure 29]| | |[Figure 30]| | | |[Figure 31]|

Figure 1. Showcases of WithAnyone. WithAnyone is capable of generating high-quality, controllable, and ID-consistent images by leveraging ID-contrastive training on the proposed MultiID-2M dataset.

##### Abstract

Identity-consistent generation has become an important focus in text-to-image research, with recent models achieving notable success in producing images aligned with a reference identity. Yet, the scarcity of large-scale paired datasets containing multiple images of the same individual forces most approaches to adopt reconstruction-based training. This reliance often leads to a failure mode we term copy-paste, where the model directly replicates the reference face rather than preserving identity across natural variations in pose, expression, or lighting. Such over-similarity undermines controllability and limits the expressive power of generation. To address these limitations, we (1) construct a large-scale

† Wei Cheng leads this project; ‡Corresponding authors.

paired dataset MultiID-2M tailored for multi-person scenarios, providing diverse references for each identity; (2) introduce a benchmark that quantifies both copy-paste artifacts and the trade-off between identity fidelity and variation; and (3) propose a novel training paradigm with a contrastive identity loss that leverages paired data to balance fidelity with diversity. These contributions culminate in WithAnyone, a diffusion-based model that effectively mitigates copypaste while preserving high identity similarity. Extensive qualitative and quantitative experiments demonstrate that WithAnyone significantly reduces copy-paste artifacts, improves controllability over pose and expression, and maintains strong perceptual quality. User studies further validate that our method achieves high identity fidelity while enabling expressive controllable generation.

[Figure 32]

###### Face Similarity

##### 1. Introduction

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

0.77 0.46 0.46 0.30

With the rapid progress of generative artificial intelligence, controllable image generation via reference images or image prompting [16, 19, 44, 57, 59, 66] and identity-consistent (ID-consistent) generation [8, 14, 15, 21, 50, 64, 68] have achieved remarkable advances: modern models can synthesize portraits that closely match the provided individual. Recent efforts [4, 8] push resemblance toward near-perfect reproduction. While pursuing higher similarity seems natural, beyond a certain point, excessive fidelity becomes counterproductive.

Real Image Pairs Ours UniPortrait PuLID InstantID UMO

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

0.818

In real photographs of the same person, identity similarity varies substantially due to natural changes in pose, expression, makeup, and illumination (Fig. 2). By contrast, many generative models adhere to the reference image far more rigidly than this natural range of variation. Although such over-optimization may seem beneficial, it suppresses legitimate variation, reducing controllability and limiting practical usability. We term this failure mode the copy-paste artifact: rather than synthesizing an identity in a flexible, controllable manner, the model effectively copies the reference image into the output (see Fig. 2). In this work, we formalize this artifact, develop metrics to quantify it, and propose a novel training strategy to mitigate it.

0.788

0.717

0.667

0.616

0.586

0.0

0.2

0.4

0.6

0.8

FaceSimilarity

1.0

Prompt: A blonde lady, natural makeup

[Figure 37]

Input InstantID PULID WithAnyone GT

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Copy Copy Customized

Figure 2. Our Observation. Natural variations, such as head pose, expression, and makeup, may cause more face similarity decrease than expected. Copying reference image limits models’ ability to respond to expression and makeup adjustment prompts.

Mitigating copy-paste artifacts is fundamentally constrained by the lack of suitable training data. While numerous large-scale face datasets exist [9, 22, 29, 47, 51, 67, 70], they remain ill-suited for controllable multi-identity generation. Critically, few datasets provide paired references for each identity-multiple images of the same person across diverse expressions, poses, hairstyles, and viewpoints. As a result, most prior work resorts to single-person, reconstructionbased training [14, 50], where the reference and target coincide. This setup inherently promotes copying and exacerbates copy-paste artifacts. Constructing datasets with multiple references per identity, particularly in group photos, and developing methods to effectively exploit such data remain open challenges.

between ID similarity and copy-paste artifacts (see Fig. 5).

Furthermore, we present WithAnyone, a novel identity customization model built on the FLUX [27] architecture, as a step toward mitigating copy-paste artifacts. WithAnyone maintains state-of-the-art identity similarity (with regard to target image) while substantially reducing copy-paste, thereby breaking the long-observed trade-off between fidelity and artifacts. This advance is enabled by a paired-training strategy combined with an ID contrastive loss enhanced with a large negative pool, both made possible by our paired dataset. The labeled identities and their reference images enable the construction of an extended negative pool (images of different identities), which provides stronger discrimination signals during optimization.

In this work, we introduce a large-scale open-source Multi-ID dataset, MultiID-2M, together with a comprehensive benchmark, MultiID-Bench, designed for intrinsic evaluation of multi-identity image generation. MultiID2M contains 500k group photos featuring 1–5 recognizable celebrities. For each celebrity, hundreds of individual images are provided as paired references, covering diverse expressions, hairstyles, and viewing angles. In addition,

In summary, our main contributions are:

- • MultiID-2M: A large-scale dataset of 500k group photos containing multiple identifiable celebrities, each with hundreds of reference images capturing diverse variations, along with 1.5M additional unpaired group photos. This resource supports pre-training and evaluation of multiidentity generation models.
- • MultiID-Bench: A comprehensive benchmark with standardized evaluation protocols for identity customization, enabling systematic and intrinsic assessment of multi-

- 1.5M unpaired group photos without references are included. MultiID-Bench establishes a standardized evaluation protocol for multi-identity generation. Beyond widely adopted metrics such as ID similarity [11, 45], it quantifies copypaste artifacts by measuring distances between generated images, references, and ground truth. Evaluation on 12 stateof-the-art customization models highlights a clear trade-off

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

###### 4. Post-processing

###### 3. ID Image Paring

###### 2. Multi-ID

###### 1. Single-ID

[Figure 64]

[Figure 65]

###### DatasetsTrainingStagesDataConstruction

###### Filtering & Labelling

###### Embedding Retrieval

###### Collection

###### Collection & Clustering

[Figure 66]

[Figure 67]

[Figure 68]

+ <name>

[Figure 69]

[Figure 70]

[Figure 71]

collage❌ not aesthetic❌ OCR & cropping

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

+ 2/3/4 actors

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

- - <watermark>
- - alone - crowd

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

cos( , )

+ <name> <ID A>

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

[Figure 95]

[Figure 96]

[Figure 97]

<ID A> <ID B>

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

###### Open Dataset

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

HQ and Style Subset 10K

Single-ID Subset 1M Mult-ID Subset 2M Paired Subset 500K

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

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

[Figure 124]

[Figure 125]

###### 1. Reconstruction

[Figure 126]

###### 2. Reconstruction

###### 4. Quality Tuning

###### 3. Paired Tuning

###### Pretrain with Caption

Quality & Style Tuning

###### Mid-train with Pair & Recon.

###### Pretrain with Fixed Prompt

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

“A man with … and a lady with”

[Figure 136]

###### “two people”

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

High Quality Stylized

Reference(s) Ground Truth

Reference(s) Ground Truth

Reference(s) Ground Truth

- Figure 3. Overview of WithAnyone. It builds on a large-scale dataset, MultiID-2M, constructed through a four-step pipeline: (1) collect and cluster single-ID data based on identity similarity; (2) gather multi-ID data via targeted searches using desired identity names with negative keywords for filtering; (3) form image pairs by matching faces between single-ID and multi-ID data; and (4) apply post-processing for quality control and stylization. Training proceeds in four stages: (1) pre-train on single-ID, multi-ID, and open-domain images with fixed prompts; (2) train with image-caption supervision; (3) fine-tune with ID-paired data; and (4) perform quality tuning using a curated high-quality subset.

identity image generation methods.

and evaluation standards. Other general-purpose customization and editing models [2, 30, 36, 37, 53–56, 61] can also synthesize images containing multiple identities, but their ID similarity is often compromised for generality.

• WithAnyone: A novel ID customization model built on FLUX that achieves state-of-the-art performance, generating high-fidelity multi-identity images while mitigating copy-paste artifacts and enhancing visual quality.

ID-Centric Datasets and Benchmarks. Although there are numerous single-ID datasets [23, 51] and multi-ID collections [9, 22], paired reference images are scarce, so reconstruction remains the dominant training objective for multi-ID datasets. Representative datasets are listed in Table 4. Evaluation protocols are underdeveloped: several works (e.g., PuLID [14], UniPortrait [15], and others [60, 68]) construct test sets by sampling identities from CelebA [29], which undermines reproducibility. Recent efforts benchmark multiple reference generation [54, 71] while focusing on general customization. To address this, we release a curated multi-ID benchmark with standardized splits and comprehensive metrics to facilitate future research.

##### 2. Related Work

Single-ID Preservation. The generation of Identitypreserving images is a core topic in customized synthesis [5, 20, 35, 48, 49, 52, 58, 60, 63]. Many methods in the UNet/Stable Diffusion era inject learned embeddings (e.g., CLIP or ArcFace) via cross-attention or adapters [17, 40–43, 64]. With the rise of DiT-style backbones [13, 27, 38] (e.g., SD3, FLUX), progress in ID preservation like PuLID [14], also attracts great attention.

Multi-ID Preservation. Multi-ID preservation remains relatively underexplored. Some works target spatial control of multiple identities [15, 25, 68], while others focus on identity fidelity. Methods such as XVerse [4] and UMO [8] use VAE-derived face embeddings concatenated with model inputs, which can produce pixel-level copy-paste artifacts and reduce controllability. DynamicID [18]1 achieves improved controllability but is constrained by limited task-specific data

##### 3. MultiID-2M: Paired Multi-Person Dataset Construction

MultiID-2M is a large-scale multi-person dataset constructed via a four-stage pipeline: (1) collect single-ID images from the web and construct a clean reference bank by clustering ArcFace [11] embeddings, yielding ∼1M reference images across ∼3k identities (averaging 400 per identity); (2) retrieve candidate group photos via multi-name and scene-

1Excluded from our experiments due to unavailability of code and pretrained models.

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

ID Contrastive Loss GT-aligned ID Loss

[Figure 151]

Face Embedding Branch

Cross Attention

[Figure 152]

[Figure 153]

|[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]|[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]|
|---|---|
|[Figure 161]| |
|[Figure 162]<br><br>[Figure 163]| |

[Figure 164]

[Figure 165]

[Figure 166]

Extended negative samples

###### Detection

Target

GT-Landmark

Target

[Figure 167]

DiT Blocks

[Figure 168]

[Figure 169]

[Figure 170]

Negative Pool Embedding

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

Target

[Figure 180]

[Figure 181]

[Figure 182]

Predicted

[Figure 183]

Predicted

[Figure 184]

[Figure 185]

1- cos( , )

[Figure 186]

Image Embedding Branch

a Model Architecture

b Training Objectives

- Figure 4. (a) Architecture of WithAnyone: Each reference is encoded by both a face-recognition network and a general image encoder, yielding identity-discriminative signals and complementary mid-level features. Face embeddings are restricted to attend only to image tokens within their corresponding face regions. (b) Training Objectives of WithAnyone: In addition to the diffusion loss, we incorporate an ID contrastive loss and a ground-truth–aligned ID loss, which together provide consistent and accurate identity supervision.

aware queries and detect faces; (3) assign identities by matching ArcFace embeddings to single-ID cluster centers using cosine similarity (threshold 0.4); and (4) perform automated filtering and annotation, including Recognize Anything [69], aesthetic scoring [12], OCR-based watermark/logo removal, and LLM-based caption generation [1]. The final corpus comprises ∼500k identified multi-ID images with matched references from the reference bank, as well as ∼1.5M additional unidentified multi-ID images for reconstruction training, covering ∼25k unique identities, with diverse nationalities and ethnicities. Further details of the construction pipeline and dataset statistics are provided in Appendix B.

SimGT the similarity to the ground-truth identity described by the prompt as the primary metric. This design penalizes excessive copying when natural variations (e.g., pose, expression, occlusion) are expected, while rewarding faithful realization of the prompted scene.

We define the angular distance as θab = arccos(Sim(a,b)) (geodesic distance on the unit sphere). The Copy-Paste metric is given by

θgt − θgr

max(θtr, ε) ∈ [−1,1], (2) where ε is a small constant for numerical stability. The metric thus captures the relative bias of g toward the reference r versus the ground truth t, normalized by angular distance of r and t. A score of 1 means g fully coincides with the reference (perfect copy-paste), while −1 means full agreement with the ground truth.

MCP(g | t,r) =

##### 4. MultiID-Bench: Comprehensive ID Customization Evaluation

MultiID-Bench is a unified benchmark for group-photo (multi-ID) generation. It samples rare, long-tail identities with no overlap to training data, yielding 435 test cases. Each case consists of one ground-truth (GT) image containing 1–4 people, the corresponding 1–4 reference images as inputs, and a prompt describing the GT. Detailed statistics are provided in Appendix B.

We additionally report identity blending, prompt fidelity (CLIP I/T), and aesthetics; formal definitions and further details are provided in Appendix C.

##### 5. WithAnyone: Controllable and IDConsistent Generation

Evaluation considers both identity fidelity and generation quality. Let r,t,g denote the face embeddings of the reference identity, the target (ground-truth), and the generated image, respectively. We define similarity between two embeddings as Sim(a,b), specifically we term the generated image’s face similarity with regard to GT as SimGT, and to reference as SimRef,

Building on the scale and paired-reference supervision of the MultiID-2M, we devise training strategies and tailored objectives that transcend reconstruction to enable robust, identity-conditioned synthesis. This rich, identity-labeled supervision not only substantially improves identity fidelity but also suppresses trivial copy–paste artifacts and affords finer control over multi-identity composition. Motivated by these advantages, we introduce WithAnyone - a unified architecture and training recipe designed for controllable, high-fidelity multi-ID generation. Architectural schematics and implementation details are provided in Fig. 4 and Appendix E.

a⊤b ∥a∥∥b∥

, (1)

Sim(a,b) =

Specially, we denote SimRef = Sim(r,g) and SimGT = Sim(t,g). Prior works [8, 14, 15, 68] has largely reported only SimRef, which inadvertently favors trivial copy-paste: directly replicating the reference appearance maximizes the score, even when the prompt specifies changes in pose, expression, or viewpoint. In contrast, MultiID-Bench uses

###### 5.1. Training Objectives

Diffusion Loss. We adopt the mini-batch empirical flowmatching loss. For each batch, we sample a data latent

x1 ∼ pdata, Gaussian noise x0 ∼ N(0,I), and a timestep t ∼ U(0,1). We then form the interpolated latent xt = (1 − t)x0 + tx1 and regress the target velocity (x1 − x0):

Ldiff = vθ(x(ti),t(i),c(i)) − (x(1i) − x(0i)) 22, (3) where c(i) denotes the conditioning signal.

Ground-truth-Aligned ID Loss. Since ArcFace embedding requires landmark detection and alignment, directly extracting landmarks from Igen is unreliable because generated images are obtained through noisy diffusion or one-step denoising. Prior methods compromise: PortraitBooth [39] applies the loss only at low noise levels (t < 0.25), discarding supervision at higher noise, while PuLID [14] fully denoises generated results at significant computational cost. In contrast, we align the generated image using GT landmarks, thereby avoiding noisy landmark extraction. We minimize the cosine distance between GT-aligned ArcFace embeddings of the generated and ground-truth (GT) faces:

LID = 1 − cos(g,t) (4)

where g and t are ArcFace embeddings of the generated and GT images. This design (1) enables applying the ID loss across all noise levels, (2) incurs negligible overhead throughout training, and (3) implicitly supervises generated landmarks. Ablation studies (Sec. 6.3) demonstrate more accurate identity measurement and substantially improved identity preservation.

Denoting the face recognition model as f(·,·) (Arcface [11], in our case), and the coupled detection model as g(·) (RetinaFace [10]), the generated image as G, and the ground-truth image as T, a embedding extraction should be performed as follows:

t = f(g(T),T), (5)

where g(T) are the detected landmarks, and f(·,·) extracts the aligned face embedding. Instead of using g(G) as landmarks for G, our GT-aligned ID loss is computed as:

Lid = 1 − cos(f(g(T),G),f(g(T),T)). (6)

ID Contrastive Loss With Extended Negatives. To further strengthen identity preservation, we introduce an ID contrastive loss that explicitly pulls the generated image closer to its reference images in the face embedding space while pushing it away from other identities. The loss follows the InfoNCE [31] formulation:

exp(cos(g,r)/τ)

, (7)

LCL = −log

M j=1 exp(cos(g,nj))/τ)

where r is the embedding of a reference image of the same identity as the generated image, nj are embeddings of M negatives from different identities, and τ is a temperature hyperparameter. This formulation relies on ID-labeled datasets, which make it possible to draw thousands of negatives per sample from the reference bank, thereby greatly enriching the diversity of negative examples.

The overall training objective is a weighted sum of the above losses:

L = Ldiff + λIDLID + λCLLCL, (8)

where λID and λCL are hyper-parameters controlling the contributions of the ID loss and contrastive loss, respectively. Both are set to 0.1 across all training phases described below.

###### 5.2. Training pipeline

Copy–paste artifacts largely arise from reconstruction-only training, which encourages models to replicate the reference

- Table 1. Quantitative comparison on the single-person subset of MultiID-Bench and OmniContext. , , and indicate the first-, second-, and third-best performance, respectively. For Copy-Paste ranking, only cases with Sim(GT) > 0.40 are considered.

###### a MultiID-Bench

###### b OmniContext Single Character Subset

Identity Metrics Generation Quality

Quality Metrics Overall PF ↑ SC ↑ Overall ↑

Method

Method

Sim(GT) ↑ Sim(Ref) ↑ CP ↓ CLIP-I ↑ CLIP-T ↑ Aes ↑

DreamO 0.454 0.694 0.303 0.793 0.322 4.877 OmniGen 0.398 0.602 0.248 0.780 0.317 5.069 OmniGen2 0.365 0.475 0.142 0.787 0.331 4.991 FLUX.1 Kontext 0.324 0.408 0.099 0.755 0.327 5.319 Qwen-Image-Edit 0.324 0.409 0.093 0.776 0.316 5.056 GPT-4o Native 0.425 0.579 0.178 0.794 0.311 5.344 UNO 0.304 0.428 0.141 0.765 0.314 4.923 USO 0.401 0.635 0.286 0.790 0.329 5.077 UMO 0.458 0.732 0.359 0.783 0.305 4.850

DreamO 8.13 7.09 7.02 OmniGen 7.50 5.52 5.47 OmniGen2 8.64 8.50 8.34 FLUX.1 Kontext 7.72 8.60 7.94 Qwen-Image-Edit 7.66 8.16 7.51 GPT-4o Native 7.98 9.06 8.12 UNO 7.22 7.72 7.04 USO 6.96 7.88 6.70 UMO 6.56 7.92 6.79

UniPortrait 0.447 0.677 0.265 0.793 0.319 5.018 ID-Patch 0.426 0.633 0.231 0.792 0.312 4.900 InfU 0.439 0.630 0.233 0.772 0.328 5.359 PuLID 0.452 0.705 0.315 0.779 0.305 4.839 InstantID 0.464 0.734 0.337 0.764 0.295 5.255 Ours 0.460 0.578 0.144 0.798 0.313 4.783

UniPortrait 6.62 6.00 5.55 ID-Patch N/A N/A N/A InfU 7.69 4.62 4.70 PuLID 6.62 6.83 5.78 InstantID 4.89 5.49 4.35 Ours 7.43 7.04 6.52

GT 1.000 0.521 -0.999 N/A N/A N/A Ref 0.521 1.000 0.999 N/A N/A N/A

Ours

0.16

GPT-4o

0.1

OmniGen OmniGen2

Copy-Paste

Copy-Paste

Ours

ID-Patch InfU

0.24

UniPortrait

0.2

GPT-4o

Other Methods

Other Methods

DreamO

ID-Patch DreamO

UniPortrait

0.32

IntantID

PuLID

0.3

0.42 0.43 0.44 0.45 0.46

0.30 0.35 0.40

Sim(GT)

Sim(GT)

a Single-ID subset

b Multi-ID subset

- Figure 5. Trade-off between Face Similarity and Copy-paste. Except for WithAnyone, the other models fall roughly on a fitted curve, illustrating a clear trade-off between face similarity and copy-paste. Upper-right corner is desired.

image rather than learn robust identity-conditioned generation. Leveraging our paired dataset, we employ a four-phase training pipeline that gradually transitions the objective from reconstruction toward controllable, identity-preserving synthesis.

###### Phase 1: Reconstruction pre-training with fixed

prompt. We begin with reconstruction pre-training to initialize the backbone, as this task is simpler than full identityconditioned generation and can exploit large-scale unlabeled data. For the first few thousand steps, the caption is fixed to a constant dummy prompt (e.g., “two people”), ensuring the model prioritizes learning the identity-conditioning pathway rather than drifting toward text-conditioned styling. The full MultiID-2M is used in this phase, which typically lasts for 20k steps, at which point the model achieves satisfactory identity similarity. To further enhance data diversity, CelebAHQ [23], FFHQ [24], and a subset of FaceID-6M [51] are also incorporated.

###### Phase 2: Reconstruction pre-training with full cap-

tions. This phase aligns identity learning with textconditioned generation and lasts for an additional 40k steps, during which the model reaches peak identity similarity.

###### Phase 3: Paired tuning. To suppress trivial copy–paste

behavior, we replace 50% of the training samples with paired instances drawn from the 500k labeled images in MultiID-

- 2M. For each paired sample, instead of using the same image as both input and target, we randomly select one reference

image from the identity’s reference set and another distinct image of the same identity as the target. This perturbation breaks the shortcut of direct duplication and compels the model to rely on high-level identity embeddings rather than low-level copying.

Phase 4: Quality tuning. Finally, we fine-tune on a curated high-quality subset augmented with generated stylized variants to (i) enhance perceptual fidelity and (ii) improve style robustness and transferability. This phase refines texture, lighting, and stylistic adaptability while preserving the strong identity consistency established in earlier phases.

##### 6. Experiments

In this section, we present a comprehensive evaluation of baselines and our WithAnyone model on the proposed MultiID-Bench.

Baselines. We evaluate two categories of baseline methods: general customization models and face customization methods. The general customization models include OmniGen [61], OmniGen2 [54], Qwen-Image-Edit [53], FLUX.1 Kontext [2], UNO [56], USO [55], UMO [8], and native GPT-4o-Image [32]. The face customization methods include UniPortrait [15], ID-Patch [68], PuLID [14] (referring to its FLUX [27] implementation throughout this paper), and InstantID [50]. All models were evaluated on the singleperson subset of the benchmark, while only those supporting

- Table 2. Quantitative comparison on the multi-person subset of MultiID-Bench. , , and indicate the first-, second-, and third-best performance, respectively. For Copy-Paste ranking, only cases with Sim(GT) > 0.35 are considered. GPT exhibits prior knowledge of identities from TV series in subsets with more than two IDs, leading to abnormally high similarity scores.

###### a 2-people Subset

###### b 3-and-4-people Subset

Identity Metrics Generation Quality

Identity Metrics Generation Quality

Method

Method

Sim(GT) ↑ Sim(Ref) ↑ CP ↓ Bld ↓ CLIP-I ↑ CLIP-T ↑ Aes ↑

Sim(GT) ↑ Sim(Ref) ↑ CP ↓ Bld ↓ CLIP-I ↑ CLIP-T ↑ Aes ↑

DreamO 0.359 0.514 0.179 0.105 0.763 0.319 4.764 OmniGen 0.345 0.529 0.209 0.110 0.750 0.326 5.152 OmniGen2 0.283 0.353 0.081 0.112 0.763 0.334 4.547 GPT 0.332 0.400 0.061 0.092 0.774 0.328 5.676 UNO 0.223 0.274 0.043 0.082 0.735 0.325 4.805

DreamO 0.311 0.427 0.116 0.081 0.709 0.317 4.695 OmniGen 0.345 0.529 0.209 0.110 0.750 0.326 5.152 OmniGen2 0.288 0.374 0.099 0.071 0.734 0.329 4.664 GPT 0.445 0.484 0.048 0.044 0.815 0.320 5.647 UNO 0.228 0.276 0.046 0.065 0.717 0.319 4.880

UMO 0.328 0.491 0.176 0.111 0.743 0.316 4.772 UniPortrait 0.367 0.601 0.254 0.075 0.750 0.323 5.187 ID-Patch 0.350 0.517 0.183 0.085 0.767 0.326 4.671 Ours 0.405 0.551 0.161 0.079 0.770 0.321 4.883

UMO 0.318 0.465 0.180 0.070 0.717 0.309 4.946 UniPortrait 0.343 0.517 0.178 0.048 0.708 0.323 5.090 ID-Patch 0.379 0.543 0.195 0.059 0.781 0.329 4.547 Ours 0.414 0.561 0.171 0.045 0.771 0.325 4.955

GT Input Ref Kontext QwenImgEdit PuLID InstantID WithAnyone

###### UniPortrait

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Prompt: “a woman wearing a white hooded jacket with a black inner garment. Her hair is styled loosely, and she has minimal makeup. The woman is posing with her head slightly tilted, showcasing a calm and composed demeanor. Her expression is neutral.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Prompt: “a woman with long, dark hair flowing dynamically. She wears a white and blue geometric patterned top with a shawl-like drape. Her posture is poised, showcasing elegant jewelry and a subtle smile. The background features a blurred circular pattern in shades of gray.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Prompt: “a woman in a black leather jacket holding a red microphone. She is smiling and appears to be performing or speaking, with her head slightly tilted and her mouth open as if she is in the middle of talking. Her long brown hair is styled straight..

UMO

###### GT Input Ref DreamO GPT UniPortrait ID-Patch WithAnyone

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Prompt: “a man in a dark suit holding a coffee mug and a woman in a light blue sweater resting her head on her hand. They appear to be in a kitchen, looking concerned or surprised. The man is standing, while the woman is seated at a counter.

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Prompt: “A couple posing together. The woman wears a blue, sleeveless, V-neck dress, while the man dons a light blue, semi-buttoned shirt. Both are smiling and standing close, with the man's arm around the woman, indicating a friendly or intimate relationship.

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Prompt: “three people, two women and one man, posing closely together. The woman on the left wears a white blazer, while the younger woman in front has a strapless top. The man has a white shirt. All are smiling warmly at the camera.

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Prompt: “four people dressed in white shirts posing together. The group includes three males and two females, with one male and one female in the center. They are smiling and standing closely, suggesting a family or close-knit group. The attire is casual and coordinated.

- Figure 6. Qualitative Results of Different Generation Methods. The text prompt is extracted from the ground-truth image shown on the leftmost side.

multi-ID generation were additionally tested on the multiperson subset. Further implementation details are provided in Appendix F.1.

###### 6.1. Quantitative Evaluation

The quantitative results are reported in Tables 1 and 2. We observe a clear trade-off between face similarity and copy-paste artifacts. As shown in Fig. 5, most methods align closely with a regression curve, where higher face similarity generally coincides with stronger copy-paste. This indicates that many existing models boost measured similarity by directly replicating reference facial features rather than synthesizing the identity. In contrast, WithAnyone deviates substantially from this curve, achieving the highest face similarity with regard to GT while maintaining a markedly lower copy-paste score.

WithAnyone also achieves the highest score among IDspecific reference models on the OmniContext [54] benchmark. However, VLMs [1, 32] exhibit limited ability to distinguish individual identities and instead emphasize nonidentity attributes such as pose, expression, or background. Despite that general customization and editing models often outperform face customization models on OmniContext, WithAnyone still has best performance among face customization models.

###### 6.2. Qualitative Comparison

To complement the quantitative results, Fig. 6 presents qualitative comparisons between our method, state-of-the-art general customization/editing models, and face customization generation models.

It shows that identity consistency remains a significant weakness of general customization or editing models, consistent with our quantitative findings. Many VAE-based approaches where references are encoded through a VAE, such as FLUX.1 Kontext and DreamO tend to produce faces that either exhibit copy-paste artifacts or deviate markedly from the target identity. A likely reason is that VAE embeddings emphasize low-level features, leaving high-level semantic understanding to the diffusion backbone, which may not have been pre-trained for this task. ID-specific reference models also struggle with copy-paste artifacts. For example, they fail to make the subject smile when the reference image is neutral and often cannot adjust head pose or even eye gaze. In contrast, WithAnyone generates flexible, controllable faces while faithfully preserving identity.

###### 6.3. Ablation and User Studies

To better understand the contribution of each component in WithAnyone, we conduct ablation studies on the training strategy, the GT-aligned ID loss, the InfoNCE-based ID loss, and our dataset. Due to space constraints, we report

Table 3. Ablation Study. indicate the first, second, third performance respectively. We ablate paired data training (without stage 2, w/o s2), GT-Aligned landmark ID loss (Self-aligned, S.A.), extended negative samples in InfoNCE (w/o neg). And model trained on FFHQ is also compared.

Identity Metrics Generation Quality

Ablation

Sim(G) ↑ Sim(R) ↑ CP ↓ CLIP-I ↑ CLIP-T ↑ Aes ↑ Phases w/o Phase 3 0.406 0.625 0.239 0.755 0.307 4.955 Loss

w/o GT-Align 0.385 0.549 0.175 0.763 0.317 4.754 w/o Ext. Neg. 0.368 0.455 0.074 0.740 0.304 4.984

Data FFHQ only 0.224 0.246 0.027 0.658 0.330 5.039 Ours Full Setting 0.405 0.551 0.161 0.770 0.321 4.883

0.10 0.06

0.54 0.34

0.36 0.21

###### 0.71

Face Sim

###### 0.44

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

###### GT Lmk

Pred Lmk

Noise Level

0.2 0.4 0.6 0.8

1.0

IDLoss

0.5

0.0

Figure 7. Comparison of GT-aligned and Prediction-aligned landmarks.

the key results here, with additional analyses provided in Appendix G.

As shown in Table 3, the paired-data fine-tuning phase reduces copy-paste artifacts without diminishing similarity to the ground truth, while training on FFHQ performs significantly worse than on our curated dataset. Fig. 7 further demonstrates that the GT-aligned ID loss lowers denoising error at low noise levels and yields higher-variance, more informative gradients at high noise, thereby strengthening identity learning. By ablating extended negatives, leaving only 63 negative samples from the batch (originally extended to 4096), the effectiveness of ID contrastive loss is greatly reduced. More ablation results can be found in Appendix G.

We conduct a user study to evaluate perceptual quality and identity preservation. Ten participants were recruited and asked to rank 230 groups of generated images according to four criteria: identity similarity, presence of copy-paste artifacts, prompt adherence, and aesthetics. The results, shown in Fig. 8, indicate that our method consistently achieves the highest average ranking across all dimensions, demonstrating both stronger identity preservation and superior visual quality. Moreover, the copy-paste metric exhibits a moderate

positive correlation with human judgments, suggesting that it captures perceptually meaningful artifacts. Further details of the study design, ranking protocol, and statistical analysis are provided in Appendix H.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Aes

PF

CP

Sim

Ours UNO ID-Patch UniPortrait OmniGen

- Figure 8. User study. Bigger bubbles indicate higher ranking.

##### 7. Conclusion

Copy-paste artifacts are a common limitation of identity customization methods, and face-similarity metrics often exacerbate the issue by implicitly rewarding direct copying. In this work, we identify and formally quantify this failure mode through MultiID-Bench, and propose targeted solutions. We curate MultiID-2M and develop training strategies and loss functions that explicitly discourage trivial replication. Empirical evaluations demonstrate that WithAnyone significantly reduces copy-paste artifacts while maintaining and in many cases improving identity similarity, thereby breaking the long-standing trade-off between fidelity and copying. These results highlight a practical path toward more faithful, controllable, and robust identity customization.

##### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 4, 8, 19

- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, 2025. 3, 6, 13

- [3] Anthony Chen, Jianjin Xu, Wenzhao Zheng, Gaole Dai, Yida Wang, Renrui Zhang, Haofan Wang, and Shanghang Zhang. Training-free regional prompting for diffusion transformers. arXiv preprint arXiv:2411.02395,

2024. 15

- [4] Bowen Chen, Mengyi Zhao, Haomiao Sun, Li Chen, Xu Wang, Kang Du, and Xinglong Wu. Xverse: Consistent multi-subject control of identity and semantic attributes via dit modulation. arXiv preprint arXiv:2506.21416, 2025. 2, 3

- [5] Weifeng Chen, Jiacheng Zhang, Jie Wu, Hefeng Wu, Xuefeng Xiao, and Liang Lin. Id-aligner: Enhancing identity-preserving text-to-image generation with reward feedback learning. arXiv preprint arXiv:2404.15449, 2024. 3

- [6] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-rendering: A diverse neural actor repository for high-fidelity humancentric rendering. In ICCV, 2023. 14

- [7] Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, and Hongsheng Li. Generalizable neural performer: Learning robust radiance fields for human novel view synthesis. arXiv preprint arXiv:2204.11798, 2022. 14

- [8] Yufeng Cheng, Wenxu Wu, Shaojin Wu, Mengqi Huang, Fei Ding, and Qian He. Umo: Scaling multiidentity consistency for image customization via matching reward. arXiv preprint arXiv:2509.06818, 2025. 2, 3, 4, 6

- [9] Jiaming Chu, Lei Jin, Yinglei Teng, Jianshu Li, Yunchao Wei, Zheng Wang, Junliang Xing, Shuicheng Yan, and Jian Zhao. Uniparser: Multi-human parsing with unified correlation representation learning. TIP, 2024. 2, 3, 14

- [10] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-

shot multi-level face localisation in the wild. In CVPR,

2020. 5

- [11] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In CVPR, 2019. 2, 3, 5, 19

- [12] discus0434. aesthetic-predictor-v2-5. https: //github.com/discus0434/aestheticpredictor-v2-5, 2023. Accessed: 2025-05-12. 4, 13, 15
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 3

- [14] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. In NeurIPS,

2024. 2, 3, 4, 5, 6, 15

- [15] Junjie He, Yifeng Geng, and Liefeng Bo. Uniportrait: A unified framework for identity-preserving single-and multi-human image personalization. ICCV, 2025. 2, 3, 4, 6, 14

- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Promptto-prompt image editing with cross attention control. ICLR, 2023. 2

- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 3

- [18] Xirui Hu, Jiahao Wang, Hao Chen, Weizhan Zhang, Benqi Wang, Yikun Li, and Haishun Nan. Dynamicid: Zero-shot multi-id image personalization with flexible facial editability. ICCV, 2025. 3

- [19] Yuqi Hu, Longguang Wang, Xian Liu, Ling-Hao Chen, Yuwei Guo, Yukai Shi, Ce Liu, Anyi Rao, Zeyu Wang, and Hui Xiong. Simulating the real world: A unified survey of multimodal generative models. arXiv preprint arXiv:2503.04641, 2025. 2

- [20] Junha Hyung, Jaeyo Shin, and Jaegul Choo. Magicapture: High-resolution multi-concept portrait customization. In AAAI, 2024. 3

- [21] Liming Jiang, Qing Yan, Yumin Jia, Zichuan Liu, Hao Kang, and Xin Lu. Infiniteyou: Flexible photo recrafting while preserving your identity. ICCV, 2025. 2

- [22] Qing Jiang, Lin Wu, Zhaoyang Zeng, Tianhe Ren, Yuda Xiong, Yihao Chen, Qin Liu, and Lei Zhang. Referring to any person. 2025. 2, 3, 14
- [23] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. ICLR, 2018. 3, 6

- [24] Tero Karras, Samuli Laine, and Timo Aila. A stylebased generator architecture for generative adversarial networks. In CVPR, 2019. 6, 19

- [25] Chanran Kim, Jeongin Lee, Shichang Joung, Bongmo Kim, and Yeul-Min Baek. Instantfamily: Masked attention for zero-shot multi-id image generation. arXiv preprint arXiv:2404.19427, 2024. 3

- [26] Minchul Kim, Anil K Jain, and Xiaoming Liu. Adaface: Quality adaptive margin for face recognition. In CVPR,

2022. 19

- [27] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 3, 6, 13
- [28] Black Forest Labs. Flux.1 krea. https : //huggingface.co/black-forest-labs/ FLUX.1-Krea-dev, 2025. 13
- [29] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In ICCV, 2015. 2, 3, 14

- [30] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. SIGGRAPH Asia,

2025. 3

- [31] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 5

- [32] OpenAI. Addendum to gpt-4o system card: Native image generation, 2025. 6, 8
- [33] Maxime Oquab, Timothée Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin ElNouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. arXiv:2304.07193, 2023. 14

- [34] Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, et al. Renderme-360: A large digital asset library and benchmarks towards high-fidelity head avatars. NeurIPS, 2023. 14

- [35] Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Jiankang Deng, Bernhard Kainz, and Stefanos Zafeiriou. Arc2face: A foundation model for id-consistent human faces. In ECCV, 2024. 3

- [36] Gaurav Parmar, Or Patashnik, Kuan-Chieh Wang, Daniil Ostashev, Srinivasa Narasimhan, Jun-Yan Zhu, Daniel Cohen-Or, and Kfir Aberman. Object-level visual prompts for compositional image generation. arXiv preprint arXiv:2501.01424, 2025. 3

- [37] Or Patashnik, Rinon Gal, Daniil Ostashev, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen-Or. Nested

- attention: Semantic-aware attention values for concept personalization. In SIGGRAPH, 2025. 3
- [38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3

- [39] Xu Peng, Junwei Zhu, Boyuan Jiang, Ying Tai, Donghao Luo, Jiangning Zhang, Wei Lin, Taisong Jin, Chengjie Wang, and Rongrong Ji. Portraitbooth: A versatile portrait model for fast identity-preserved personalization. In CVPR, 2024. 5

- [40] Guocheng Qian, Kuan-Chieh Wang, Or Patashnik, Negin Heravi, Daniil Ostashev, Sergey Tulyakov, Daniel Cohen-Or, and Kfir Aberman. Omni-id: Holistic identity representation designed for generative tasks. CVPR, 2025. 3

- [41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 15

- [42] Xingyu Ren, Alexandros Lattas, Baris Gecer, Jiankang Deng, Chao Ma, and Xiaokang Yang. Facial geometric detail recovery via implicit representation. In FG, 2023. 13

- [43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015. 3

- [44] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 2

- [45] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In CVPR, 2015. 2, 19

- [46] Erich Schubert, Jörg Sander, Martin Ester, Hans Peter Kriegel, and Xiaowei Xu. Dbscan revisited, revisited: why and how you should (still) use dbscan. TODS,

2017. 13

- [47] Lorenzo Stacchio, Alessia Angeli, Giuseppe Lisanti, Daniela Calanca, and Gustavo Marfia. Imago: A family photo album dataset for a socio-historical analysis of the twentieth century. arXiv preprint arXiv:2012.01955, 2020. 2, 14

- [48] Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. Face0: Instantaneously conditioning a textto-image model on a face. In SIGGRAPH Asia, 2023. 3

- [49] Qinghe Wang, Xu Jia, Xiaomin Li, Taiqing Li, Liqian Ma, Yunzhi Zhuge, and Huchuan Lu. Stableidentity: Inserting anybody into anywhere at first sight. TMM,

2025. 3

- [50] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identity-

- preserving generation in seconds. arXiv preprint arXiv:2401.07519, 2024. 2, 6
- [51] Shuhe Wang, Xiaoya Li, Jiwei Li, Guoyin Wang, Xiaofei Sun, Bob Zhu, Han Qiu, Mo Yu, Shengjie Shen, Tianwei Zhang, et al. Faceid-6m: A large-scale, opensource faceid customization dataset. arXiv preprint arXiv:2503.07091, 2025. 2, 3, 6

- [52] Yibin Wang, Weizhong Zhang, Jianwei Zheng, and Cheng Jin. High-fidelity person-centric subject-toimage synthesis. In CVPR, 2024. 3

- [53] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 3, 6

- [54] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025. 3, 6, 8, 19

- [55] Shaojin Wu, Mengqi Huang, Yufeng Cheng, Wenxu Wu, Jiahe Tian, Yiming Luo, Fei Ding, and Qian He. Uso: Unified style and subject-driven generation via disentangled and reward learning. arXiv preprint arXiv:2508.18966, 2025. 6

- [56] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to-more generalization: Unlocking more controllability by in-context generation. ICCV, 2025. 3, 6, 14

- [57] Tong Wu, Yinghao Xu, Ryan Po, Mengchen Zhang, Guandao Yang, Jiaqi Wang, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Fiva: Fine-grained visual attribute dataset for text-to-image diffusion models. NeurIPS,

2024. 2

- [58] Yi Wu, Ziqiang Li, Heliang Zheng, Chaoyue Wang, and Bin Li. Infinite-id: Identity-preserved personalization via id-semantics decoupling paradigm. In ECCV, 2024. 3

- [59] Chufeng Xiao and Hongbo Fu. Customsketching: Sketch concept extraction for sketch-based image synthesis and editing. In Computer Graphics Forum. Wiley Online Library, 2024. 2

- [60] Guangxuan Xiao, Tianwei Yin, William T Freeman, Frédo Durand, and Song Han. Fastcomposer: Tuning-

- free multi-subject image generation with localized attention. IJCV, 2025. 3
- [61] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340,

2024. 3, 6

- [62] Hengyuan Xu, Liyao Xiang, Hangyu Ye, Dixi Yao, Pengzhi Chu, and Baochun Li. Permutation equivariance of transformers and its applications. In CVPR,

2024. 15

- [63] Yuxuan Yan, Chi Zhang, Rui Wang, Yichao Zhou, Gege Zhang, Pei Cheng, Gang Yu, and Bin Fu. Facestudio: Put your face everywhere in seconds. arXiv preprint arXiv:2312.02663, 2023. 3

- [64] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arxiv:2308.06721, 2023. 2, 3, 15

- [65] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 15, 19

- [66] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2

- [67] Ning Zhang, Manohar Paluri, Yaniv Taigman, Rob Fergus, and Lubomir Bourdev. Beyond frontal faces: Improving person recognition using multiple cues. In CVPR, 2015. 2, 14

- [68] Yimeng Zhang, Tiancheng Zhi, Jing Liu, Shen Sang, Liming Jiang, Qing Yan, Sijia Liu, and Linjie Luo. Id-patch: Robust id association for group photo personalization. In CVPR, 2025. 2, 3, 4, 6, 14, 19

- [69] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514, 2023. 4, 13

- [70] Yujie Zhong, Relja Arandjelovic, and Andrew Zisserman. Compact deep aggregation for set retrieval. In ECCV, 2018. 2, 14

- [71] Cailin Zhuang, Ailin Huang, Wei Cheng, Jingwei Wu, Yaoqi Hu, Jiaqi Liao, Hongyuan Wang, Xinyao Liao, Weiwei Cai, Hengyuan Xu, et al. Vistorybench: Comprehensive benchmark suite for story visualization. arXiv preprint arXiv:2505.24862, 2025. 3

## Appendix

names as search queries on Google Images. For each image, facial features were extracted with ArcFace [42], ensuring that only images containing exactly one face were retained. To remove outliers, DBSCAN [46] clustering was applied to the embeddings for each celebrity, resulting in a set of cluster centers and hundreds of reference images per identity. This process established a reliable reference set for each unique identity. Human review confirms the accuracy of the ID bank built in this step.

##### A. Family of WithAnyone

FLUX.1 comprises a family of models, including FLUX.1 [27], FLUX.1 Kontext [2] and FLUX.1 Krea [28]. Krea is a text-to-image model with improved real-person face generation, whereas Kontext is an image-editing model that excels at making targeted adjustments while preserving the rest of the image. However, as reported in Table 1, Kontext shows limited consistency with the reference face identity.

Multi-ID images. To achieve best searching efficiency, group photos were obtained using more complex queries that combined multiple celebrity names, keywords indicating the number of people (e.g., “two celebrities”), scene descriptors (e.g., “award ceremony”), and negative keywords to filter out irrelevant results. ArcFace embeddings were extracted for these images, yielding a large pool of candidate multi-ID images. At this stage, the dataset comprised more than 20 million images.

Our method, WithAnyone, can be seamlessly integrated into Kontext for the face customization downstream tasks like face editing. As illustrated in Fig. 9, WithAnyone effectively injects identity information from the reference images into the target image.

Retrieval. To provide ID reference for the multi-ID images, it is necessary to retrieve the IDs on it. All single-ID cluster centers were aggregated into an embedding matrix. For each detected face in every multi-ID image, its ArcFace embedding was compared to all single-ID cluster centers to determine identity. The similarity between two embeddings was calculated as:

The overall training pipeline follows the procedure described in Sec. 5, with a single modification: the input image provided to Kontext (whose tokens are concatenated with the noisy latent at each denoising step) is set to the target image with the face region blurred.

##### B. MultiID-2M Construction Details

###### sim(id1,id2) = cos(f(id1),f(id2)) (9)

To fill in the void left by the lack of publicly available multiID datasets, a data constraction pipeline is proposed to create a large-scale dataset of multi-person images with paired identity references for identities on the data record. Based on this pipeline, 500k group photo images are collected, featuring 3k identities, each with hundreds of single-ID reference images. Another 1M images that cannot be identified are also included in the dataset for image reconstruction training purpose for image reconstruction training purpose.

where id1 and id2 denote two faces, and f is the ArcFace embedding network.

Each face in a multi-ID image was assigned the identity of the single-ID cluster center with the highest similarity, provided the similarity exceeded a predefined threshold (0.5). This approach enabled accurate and automated identity assignment in group images and facilitated retrieval of corresponding reference images.

Filtering and labelling. To further improve dataset quality, a series of annotation and filtering steps were applied. The Recognize Anything model [69], an aesthetic score predictor [12], and other auxiliary tools were used for annotation. Images with low aesthetic scores or those identified as collages rather than genuine group photos were excluded.

###### B.1. Dataset Construction Pipeline

The pipeline contains four steps, as shown in Fig. 3. The detailed pipeline are as follows.

Single-ID images. To construct a ID reference set, singleID images were collected from the web using celebrity

|[Figure 254]<br><br>W|[Figure 255]|[Figure 256]<br><br>e|
|---|---|---|
| |ithAnyon Kontext| |
| |[Figure 257]| |

|[Figure 258]<br><br>W|[Figure 259]|[Figure 260]<br><br>e|
|---|---|---|
| |ithAnyon Kontext| |
| |[Figure 261]| |

- Figure 9. Application of WithAnyone-Kontext. Marrying editing models, WithAnyone is capable of face editing given customization references.

Range: 3 to 42738

USA

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

AppearanceCount

European descent

China

Multiple US

30k

Korea

Europe

Asian descent

Japan

15k

CN

Others

Single

Canada

Median: 2407

African descent

EU

0k Celebrities

CA

Count

0 600 1200

a ID Appearance.

b Nationality Distribution.

###### c Benchmark Distribution.

- Figure 10. Overview of Dataset Distributions. (a) ID appearance distribution for the subset of one nation: the x-axis represents celebrities, sorted by the number of images in which they appear. (b) Nationality distribution: celebrities in our dataset come from over 10 countries, with most data sourced from China and the USA. (c) Word cloud of the most frequent words in the captions.

Optical Character Recognition (OCR) tools detected watermarks and logos, which were cropped out when possible; otherwise, the images were discarded. Finally, descriptive captions were generated for the images using a large language model, enriching the dataset with textual information.

So far, a dataset with three parts is obtained: (1) 1M single-ID images as reference bank, or single-ID crosspaired training; (2) 500k paired multi-ID images with identified persons; (3) 1M unpaired multi-ID images, which can be used for training scenario without the need of references, such as reconstruction.

###### B.2. Dataset Statistics

Following prior arts [6, 7, 29, 34], comprehensive statistics of the dataset are provided in Fig. 11, including the distribution of nationalities, the count of appearances per identity, and a word cloud illustrating the most frequent terms in the generated image captions, offering insights into the diversity and richness of the dataset. A long-tail distribution is observed in the count of appearances per identity in Fig. 11a, with a few identities appearing frequently while many others are less common. This provide a diverse set of identities, as well as a perfect test dataset without identity interaction with the training set. Fig. 11b and Fig. 10c illustrate MultiID-2M’s nationality distribution and action diversity respectively. The comparison between the proposed dataset and existing multi-ID datasets are listed in Table 4, highlighting MultiID-2M’s outstanding volume and paired references.

##### C. Benchmark and Metrics Details

Most existing methods are evaluated on privately curated test sets that are seldom released, and even when datasets are shared, the accompanying evaluation protocols vary widely. For example, ID-Patch [68] and UniPortrait [15] measure identity similarity using ArcFace embeddings, whereas UNO [56] relies on DINO [33] and CLIP similarity scores. This

Table 4. Statistic comparison for multi-identity group photo datasets. #Img refers to total scale of the dataset; #Paired refers to paired group image number; #Img / ID indicates number of reference image for each single ID; #ID / Img means number of IDs appears on group photos.

Dataset #Img #Paired #Img / ID #ID / Img IMAGO [47] 80k 0 0 -

MHP [9] 5k 0 0 2 − 10 PIPA [67] 40k 40k cross 1 − 10

HumanRef [22] 36k 36 1+ 1 − 14+ Celebrity Together [70] 194k 0 0 1 − 5

MultiID-2M 1.5M 500k 100+ 1 − 5

heterogeneity together with the common practice of reporting only the cosine similarity between matched ArcFace embeddings fails to capture more nuanced insights and can even encourage degenerate behavior in which models produce images that are effectively “copy-pastes” of the reference photos.

In this work, MultiID-Bench is introduced as a unified and extensible evaluation framework for group photo (multiID) generation. It standardizes assessment along two complementary axes: (i) identity fidelity (preserving each target identity without unintended copying and blending), and (ii) generation quality (semantic faithfulness to the prompt/ground truth and overall aesthetic quality).

The data used in MultiID-Bench are drawn from the long-tail portion of MultiID-2M. We first select the least frequent identities and gather all images containing them. To prevent information leakage, the training split is filtered to ensure zero identity overlap with the benchmark set. The final benchmark contains 435 samples; each sample provides 1–4 reference identities (with their images), a corresponding ground-truth image, and a text prompt describing that ground-truth scene.

Identity Blending. In the similarity matrix, the offdiagonal elements correspond to the similarity between different identities. The average of the diagonal elements is

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

costume

others

Ancient

communicate

Professional

Dynamic

Interaction

Special

intimate

Upper

[Figure 266]

[Figure 267]

Clothes

Lower

Actions

Single

Normal

Static Expression

Top

a Clothes & Accessories Distribution.

b Action Distribution. Figure 11. Distribution of Clothes and Action Labels of Proposed Dataset.

used as the metric for identity fidelity, and the average of the off-diagonal elements serves as the metric for identity blending, as in Eq. 10.

N

N

1 N2 − N

MBld(xg,xt) =

cos(gi,tj) (10)

i=1

j=1,j̸=i

where gi is the embedding of the i-th face in the generated image xg, and tj is the embedding of the j-th face in the ground-truth image xt. A lower value indicates less unintended blending between different identities, which is desirable.

Generation quality. The overall generation quality is evaluated based on CLIP-I and CLIP-T, which are the de facto standards for evaluating the prompt-following capability [41], are employed to measure the cosine similarity in the CLIP embedding space between the generated image and the ground truth image or caption. Additionally, an aesthetic score model [12] is used to assess the aesthetic quality of the generated images.

##### D. Galleries of WithAnyone

We show more results of WithAnyone in Fig. 12, Fig. 13, and Fig. 14.

##### E. Model Framework Details

We follow prior work [14, 64] and integrate a lightweight identity adapter into the diffusion backbone. Identity embeddings are injected by cross-attention so that the base generative prior is preserved while controllable identity signals are added.

Face embedding. Each reference face is first encoded by ArcFace, producing a 1×512 identity embedding. To match the tokenized latent space of the DiT backbone, this vector is projected with a multi-layer perceptron (MLP) into 8

tokens of dimension 3072 (i.e., an 8 × 3072 tensor). This tokenization provides sufficient capacity for the cross-attention layers to integrate identity cues without overwhelming the generative context.

Controllable attribute retention. Completely suppressing copy-like behavior is not always desirable: users sometimes expect certain mid-level appearance attributes (e.g., hairstyle, accessories) to be preserved. ArcFace focuses on high-level, identity-discriminative geometry and texture cues but omits many mid-level semantic factors. To expose controllable retention of such attributes when needed, we optionally incorporate SigLIP [65] as a secondary encoder. SigLIP provides more semantically entangled representations, enabling selective transfer of style-relevant traits while ArcFace anchors identity fidelity.

Attention mask and location control. To further improve identity disentanglement and precise localization in the generated images, an attention mask and location control mechanism are incorporated [3, 62]. Specifically, groundtruth facial bounding boxes are extracted from the training data and used to generate binary attention masks. These masks are applied to the attention layers of the backbone model, ensuring that each reference token only attends to its corresponding face region in the image, providing location control at the same time.

Feature injection. After each transformer block of the DiT backbone, we inject face features through a crossattention modulation:

(HWQ)(EWK)⊤ √

H′ = H+λid softmax

+ M (EWV ),

d

(11) where H denotes the current hidden tokens, E the stacked face-embedding tokens, and WQ,WK,WV the projection matrices; d is the query/key dimension, and λid = 1.0 during training. When SigLIP is enabled, its tokens are processed by a parallel cross-attention with an independent scaling

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

#### Reference

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

#### Reference

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

#### Reference

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

#### Reference

[Figure 298]

[Figure 299]

[Figure 300]

- Figure 12. Galleries of Single-ID Generation.

|[Figure 301]|[Figure 302]| | |[Figure 303]| | |[Figure 304]<br><br>[Figure 305]| | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 306]| | | | | | | | | | | |
|[Figure 307]| |[Figure 308]| | |[Figure 309]<br><br>[Figure 310]| | | | | |[Figure 311]|
| | | | | | | | | | | | |
|[Figure 312]| | | | | | |[Figure 313]|[Figure 314]| |[Figure 315]| |
|[Figure 316]| | | | | | | |[Figure 317]| | | |
|[Figure 318]| | |[Figure 319]<br><br>[Figure 320]<br><br>| | | | | | | |[Figure 321]<br><br>[Figure 322]|
| | | |[Figure 323]<br><br>| | | | | | | | |
|[Figure 324]|[Figure 325]| | | |[Figure 326]| | | |[Figure 327]| | |
|[Figure 328]| | | | | | | | | | | |

[Figure 329]

###### Figure 13. Galleries of 2-person Generation.

| |[Figure 330]<br><br>[Figure 331]|[Figure 332]| | | |[Figure 333]| |[Figure 334]| |
|---|---|---|---|---|---|---|---|---|---|
|[Figure 335]| | | | | | | |[Figure 336]| |
| | | | | | | | | | |
|[Figure 337]| | |[Figure 338]<br><br>[Figure 339]|[Figure 340]|[Figure 341]| | | |[Figure 342]|
| | | | | | | | | |[Figure 343]|
| | | | |[Figure 344]| | | |[Figure 345]| |
|[Figure 346]| |[Figure 347]| | | |[Figure 348]|[Figure 349]| | |
|[Figure 350]| | | | | |[Figure 351]| | | |
|[Figure 352]| | | | | |[Figure 353]| | | |
|[Figure 354]<br><br>|[Figure 355]|
|---|
<br><br>[Figure 356]<br><br>| |
|---|
<br><br>[Figure 357]<br><br>| |
|---|
<br><br>[Figure 358]<br><br>| |
|---|
| | | | | | | | | |

###### Figure 14. Galleries of 3-to-4-person Generation.

coefficient.

##### F. Experimental Details

###### F.1. Implementation Details

WithAnyone is trained on 8 NVIDIA H100 GPUs, with a batch size of 4 on each GPU. The learning rate is set to 1e−4, and the AdamW optimizer is employed with a weight decay of 0.01. The pre-training phase runs for 60k steps, with a fixed prompt used during the first 20k steps. The subsequent paired-tuning phase lasts 30k steps: 50% of the samples use paired (reference, ground-truth) data, while the remaining 50% continue reconstruction training. Finally, a quality/style tuning stage of 10k steps is performed with a reduced learning rate of 1 × 10−5.

For the extended ID contrastive loss, the target is used as the positve sample, while other IDs from samples in the same batch serve as negative samples. With the global batch size of 32, this yields less than a hundred negative samples. Extended negative samples are drawn from reference bank. If this ID is identified as one of the 3k ID in the reference bank, we simply omit its own ID and draw the from other IDs. If this ID is not identified, then it makes things easier – all the IDs in the reference bank can be used as negative samples.

For other baseline methods, official implementations and checkpoints (or API) are used with default settings. Methods are tested on MultiID-Bench and real-human subset of OmniContext [54]. OmniContext uses Vision-Language Models (VLMs) to evaluate the prompt-following (PF) and subjectconsistency (SC) of generated images. For reproducibility, the VLM is fixed to Qwen2.5-VL [1]. ID-Patch [68] requires pose condition, and we use the ground-truth pose for it.

Single face embedding model may induce biased evaluation on ID similarity, thus we average three de-facto face recognition models’ consine similarity to compute the overall ID similarity metric, namely ArcFace [11], FaceNet [45], and AdaFace [26].

###### F.2. More Discussion on the Quantitative Results

The performance of GPT on our 3-and-4-people subset offers a useful validation of our copy-paste metric, as shown in Table 2. This subset largely comprises group photographs from TV series that GPT may have encountered during pretraining, so GPT attains unusually high identity-similarity scores both to the ground truth (GT) and to the reference images. Actually, in one case GPT even generates an ID from the TV series that is not present in the reference images. This behaviour approximates an idealized scenario in which a model fully understands and faithfully reproduces the target identity: similarity to GT and to references are both high, and the copy-paste measure the difference between distances to GT and to references approaches zero. These observa-

tions are consistent with our metric design and support its ability to distinguish true identity understanding from trivial copy-and-paste replication.

We report the experimental limit in Table 1. If one model completely copy the reference image, SimGT = 0.521, SimRef = 1.0, and copy-paste is 0.999, which aligns with the theoretical limit 1.0 of copy-paste.

The prompt-following ability is measured by CLIP-I and CLIP-T in our benchmark, and is judged by VLM in OmniContext. WithAnyonegains state-of-the-art performance in both metrics, and is ranked the highest in our user study. However, the credibility of CLIP scores and the aesthetic scores may be debated, as they are not always consistent with human perception.

##### G. Ablation Study Details

In this section, we systematically evaluate the impact of training strategy, GT-aligned ID-Loss, InfoNCE ID Loss, and our dataset construction. User study is also conducted to validate the consistency of the proposed metrics with human perception, as well as evaluate the human preference on different methods.

SigLIP signal. SigLIP [65] signal is introduced to retain copy-paste effect when user tend to retain the features from reference images like hairstyle, accessories, etc. As shown in Fig. 16, increasing the SigLIP signal weight effectively amplifies the copy-paste effect while simultaneously boosting ID similarity to the reference images exactly as expected, since stronger SigLIP guidance enforces tighter semantic alignment and transfers more fine-grained appearance cues (e.g., hairstyle, accessories, local textures).

Training strategy. We evaluate the effect of a paired-data fine-tuning stage. After an initial reconstruction training phase, we either continue training with paired (reference, ground-truth) data or keep training under the reconstruction objective for 10k steps. As shown in Table 3, continuing with paired data effectively reduces the copy-paste effect without compromising similarity to the ground truth.

Dataset construction. To validate the effectiveness of our dataset, we trained a model on FFHQ [24] using reconstruction training for the same number of steps. As shown in Table 3, the FFHQ-trained model performs poorly across all metrics. This likely stems from FFHQ’s limited diversity and size, as it contains only 70k face-only portrait images.

GT-aligned ID-Loss. We validate the GT-aligned IDLoss with a simple experiment that visualizes predicted faces at different denoising time steps during training. As shown in Fig. 7, at low noise levels the GT-aligned ID-Loss is substantially lower than the loss computed using predictionaligned landmarks, indicating that aligning faces to groundtruth landmarks reduces denoising error and yields a more accurate identity assessment. At high noise levels the GTaligned ID-Loss shows greater variance, producing stronger

0.7

IDLoss

0.6

0.0 0.1 0.1+

0.5

0.4

0 300 600

Step

Figure 15. ID Loss Curves with λ× InfoNCE Loss. 0.1 is 0.1× InfoNCE Loss without extended negative samples, and 0.1+ is 0.1× InfoNCE Loss with extended negative samples.

[Figure 359]

Figure 17. User Study Interface.

and more informative gradients that help the model learn identity features.

InfoNCE Loss. The InfoNCE loss with extended negative samples is crucial for the convergence in the early training stage. We conduct a toy experiment with 1000 training samples, and record ID Loss curves with no InfoNCE loss, 0.1× InfoNCE loss without extended negatives, and 0.1× InfoNCE loss with extended negatives. As shown in Fig. 15, ID loss fits a lot faster with InfoNCE loss with extended negatives, demonstrating its effectiveness in accelerating training convergence. It also largely increases the ID similarity score, as shown in Table 3.

0.15

Copy-Paste

0.20

0.0 * Sig 0.2 * Sig

0.4 * Sig 0.6 * Sig

- 0.8 * Sig

- 1.0 * Sig

0.25

0.55 0.60 0.62 0.64

Sim(Ref)

Figure 16. Trade-off Curves with λ× Siglip and (1−λ)× ArcFace signal.

##### H. User Study Details

Our user study is conducted with the same data samples and generated results in our quantitative experiments. Due to a tight financial budget, we randomly select 100 samples from single-person subset, 100 samples from 2-people subset, and all samples from 3-and-4 people subset. 10 participants are recruited for the study, all of whom are trained with a brief tutorial to understand the task and evaluation criteria.

We illustrate the interface used in our user study in Fig. 17.

- H.1. Correlation Analysis

We analyze the correlation between our proposed metrics and user study results. As shown in Table 5, our copy-paste metric shows a moderate positve correlation with user ratings on copy-paste effect.

- H.2. Participant Instructions

We provide the instructions for training the participants in the following table.

##### I. Prompts for Language Models

Large language models (LLMs) and vision-language models (VLMs) are used in various stages of our work, including dataset captioning and OmniContext evaluation.

###### I.1. Dataset Captioning

Besides the system prompt, we design 6 different prompts to generate diverse captions for each image. 1 prompt is randomly selected for each image during captioning.

Table 5. Correlation Statistics Between Machine Ranking and Human Ranking. Reported values include Pearson’s r, Spearman’s ρ, and Kendall’s τ with corresponding p-values.

###### Dimension (N) Pearson r (p) Spearman ρ (p) Kendall τ (p)

Copy-Paste 0.4417 (7.98e−48) 0.4535 (1.26e−50) 0.3405 (1.10e−46) ID Sim 0.3254 (1.54e−26) 0.3237 (2.91e−26) 0.2423 (1.11e−25)

Participant Instructions and Evaluation Procedure Data source and task overview. Five different methods generated images under the following conditions:

- • A single prompt that describes the “ground truth image.”
- • Between 1 and 4 people in the scene (most examples contain 1–2 people). For each trial you will be shown the ground truth image, input images, and a generation instruction. Then you will observe five generated group-photo results (one per method) and rank them according to several evaluation dimensions. Use a 5-star scale where 5 stars = best and 1 star = worst. Please read the input image(s) and the editing instruction carefully before inspecting the generated results.

Evaluation procedure (per-image ranking). Rank each generated image individually on the following criteria.

Identity similarity

- • How well do the person(s) in the generated image resemble the person(s) in the ground truth image?

- • Rank images by their resemblance to the ground truth image: the more the generated person(s) look like the original reference, the higher the rating.
- • Important: When judging identity similarity, ignore factors such as image quality, rendering artifacts, or general aesthetics. Focus only on how much the person(s) resemble the original reference(s). Also, try to assess resemblance to the ground truth image as a whole, rather than comparing to any single separate “reference person n.”

###### Copy-and-paste effect (excessive mimicry of the reference)

- • Generated images should resemble the original reference but should not be direct copies of an individual reference photo.
- • Evaluate whether the generated person appears to be directly copied from one of the reference images. Consider changes (or lack thereof) in expression, head pose and orientation, facial expression/demeanor, and lighting/shading.
- • The lower the degree of direct copying (i.e., the less it looks like a pasted replica), the better. Rank according to the amount of change observed in the person(s): more natural variation (less copy-paste) should be ranked higher.

###### Prompt following

- • Does the generated image reflect the content and constraints specified by the prompt/instruction?
- • Rank images by prompt fidelity: the more faithfully the image follows the prompt, the higher the ranking.

Aesthetics

- • Judge the overall visual quality and pleasantness of the generated image (e.g., smoothness of rendering, harmonious body poses and composition).
- • Rank images by aesthetic quality: higher perceived visual quality receives higher ratings.

Full Prompts for Dataset Captioning (6 variants) System Prompt: You are an advanced vision-language model tasked with generating accurate and comprehensive captions for images.

- Prompt 1: Please provide a brief description of the image based on these guidelines:

- 1. Describe the clothing, accessories, or jewelry worn by the people in detail.
- 2. Describe the genders, actions, and posture of the individual in detail, focusing on what they are doing.
- 3. The description should be concise, with a maximum of 77 words.
- 4. Start with ‘This image shows’

- Prompt 2: Offer a short description of the image according to these rules:

- 1. Focus on details about clothing, accessories, or jewelry.
- 2. Focus on the gender, activity, and pose, and explain what the people is doing.
- 3. Keep the description within 77 words.
- 4. Begin the description with ‘This image shows’

- Prompt 3: Please describe the image briefly, following these instructions:

- 1. Provide a detailed description of the clothing or jewelry the person may be wearing.
- 2. Provide a detailed description of the two persons’ gender, actions, and body position.
- 3. Limit the description to no more than 77 words.
- 4. Begin your description with ‘This image shows’

- Prompt 4: Describe the picture briefly according to these rules:

- 1. Provide a detailed description of the clothing, jewelry, or accessories of the individuals.
- 2. Focus on the two persons’ gender, what they are doing, and their posture.
- 3. Keep the description concise, within a limit of 77 words.
- 4. Start your description with ‘This image shows’

- Prompt 5: Provide a short and precise description of the image based on the following guidelines:

- 1. Describe what the person is wearing or any accessories.
- 2. Focus on the gender, activities, and body posture of the person.
- 3. Ensure the description is no longer than 77 words.
- 4. Begin with ‘This image shows’

- Prompt 6: Briefly describe the image according to these instructions:

- 1. Provide a precise description of the clothing, jewelry, or other adornments of the people.
- 2. Focus on the person’s gender, what they are doing, and their posture.
- 3. The description should not exceed 77 words.
- 4. Start with the phrase ‘This image shows’

###### Modified Prompt for OmniContext Evaluation (Face Identity Focus) Rate from 0 to 10:

Task: Evaluate how well the facial features in the final image match those of the individuals in the original reference images, as described in the instruction. Focus strictly on facial identity similarity; ignore hairstyle, clothing, body shape, background, and pose.

###### Scoring Criteria

- • 0: The facial features are completely different from those in the reference images.

- • 1–3: The facial features have minimal similarity with only one or two matching elements.

- • 4–6: The facial features have moderate similarity but several important differences remain.

- • 7–9: The facial features are highly similar with only minor discrepancies.

- • 10: The facial features are perfectly matched to those in the reference images. Pay detailed attention to these facial elements:

- • Eyes: Shape, size, spacing, color, and distinctive characteristics of the eyes and eyebrows.
- • Nose: Shape, size, width, bridge height, and nostril appearance.
- • Mouth: Lip shape, fullness, width, and distinctive smile characteristics.
- • Facial structure: Cheekbone prominence, jawline definition, chin shape, and forehead structure.
- • Skin features: Distinctive marks like moles, freckles, wrinkles, and overall facial texture.
- • Proportions: Overall facial symmetry and proportional relationships between features.

Example: If the instruction requests combining the face from one image onto another pose, the final image should clearly show the same facial features from the source image.

Important:

- • For each significant facial feature difference, deduct at least one point.
- • Ignore hairstyle, body shape, clothing, background, pose, or other non-facial elements.

- • Focus only on facial similarity, not whether the overall instruction was followed.

- • Scoring should be strict high scores should only be given for very close facial matches.
- • Consider the level of detail visible in the images when making your assessment. Editing instruction: <instruction>

