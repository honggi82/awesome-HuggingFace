## RefineAnything: Multimodal Region-Specific Refinement for Perfect Local Details

##### Dewei Zhou1, You Li1, Zongxin Yang2, and Yi Yang1

###### 1 RELER, CCAI, Zhejiang University, {zdw1999, uli2000, yangyics}@zju.edu.cn 2 DBMI, HMS, Harvard University, Zongxin_Yang@hms.harvard.edu

# arXiv:2604.06870v1[cs.CV]8Apr2026

Prompt Refined by Ours Gemini3-Image Refined by Ours

Reference Image GPT-Image 1.5

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Seen from a distance, a person wearing these shoes and two friends are buying something by an stand on the street. The person is fully visible.

| |
|---|

Prompt Reference Image Input Image

Gemini3-Image Qwen-Image Ours

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

Refine text on the box.

| |
|---|

Prompt

Input Image Ours Prompt

Input Image Ours

[Figure 22]

[Figure 23]

|[Figure 24]|
|---|
|[Figure 25]|
|[Figure 26]|
|[Figure 27]|

|[Figure 28]|
|---|
|[Figure 29]|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

| | |
|---|---|
| | |
| | |

| | |
|---|---|
| | |
| | |

Refine the face.

Refine text ‘HOPE accept’.

|[Figure 34]|
|---|

|[Figure 35]|
|---|

Fig. 1: RefineAnything restores fine-grained details (e.g., text, logos, and faces) in user-specified regions (indicated by the bounding boxes) for both reference-based and reference-free inputs, keeping the background unchanged.

Abstract. We introduce region-specific image refinement as a dedicated problem setting: given an input image and a user-specified region (e.g., a scribble mask or a bounding box), the goal is to restore finegrained details while keeping all non-edited pixels strictly unchanged. Despite rapid progress in image generation, modern models still frequently suffer from local detail collapse (e.g., distorted text, logos, and thin structures). Existing instruction-driven editing models emphasize coarse-grained semantic edits and often either overlook subtle local defects or inadvertently change the background, especially when the region of interest occupies only a small portion of a fixed-resolution input. We present RefineAnything, a multimodal diffusion-based refinement model that supports both reference-based and reference-free refinement. Building on a counter-intuitive observation that crop-and-resize

can substantially improve local reconstruction under a fixed VAE input resolution, we propose Focus-and-Refine, a region-focused refinementand-paste-back strategy that improves refinement effectiveness and efficiency by reallocating the resolution budget to the target region, while a blended-mask paste-back guarantees strict background preservation. We further introduce a boundary-aware Boundary Consistency Loss to reduce seam artifacts and improve paste-back naturalness. To support this new setting, we construct Refine-30K (20K reference-based and 10K reference-free samples) and introduce RefineEval, a benchmark that evaluates both edited-region fidelity and background consistency. On RefineEval, RefineAnything achieves strong improvements over competitive baselines and near-perfect background preservation, establishing a practical solution for high-precision local refinement. Project Page: https://limuloo.github.io/RefineAnything/.

Keywords: Image Generation · Image Editing · Multimodal Learning

### 1 Introduction

Image generation has advanced rapidly, and modern models offer substantially improved controllability [4,8,9,11,12,19–24,26–31,36,40,43,46,51,53–65,67]. Yet a practical failure mode still frequently blocks real-world deployment: local detail collapse. As shown in Fig. 1, fine-grained elements such as printed text, logos, and thin structures are often distorted or inconsistent, even when the global composition is plausible. This issue is particularly damaging in high-stakes applications where small details carry key information, such as e-commerce product images and advertisements, retail signage and packaging, or UI/infographics, where a single wrong character or broken stroke can undermine trust and usability.

This motivates region-specific image refinement as a dedicated problem setting: given an input image and a user-specified region, the goal is to improve local details while keeping the rest of the image strictly unchanged.

In this setting, a natural first attempt is to use today’s instruction-driven editing models to “fix” local defects with prompts. However, existing paradigms are not well-suited to refinement, as shown in Fig. 1 and Fig. 6, mainly due to three issues: (1) weak region controllability—it is difficult to precisely specify where to refine; (2) poor micro-detail recovery—subtle defects (e.g., broken text strokes) are often left unresolved; and (3) background drift—non-target regions may change unintentionally. In practice, users require a refinement tool that is simultaneously region-accurate, detail-effective, and background-preserving.

To achieve region controllability, we propose RefineAnything (Fig. 2), a region-aware refinement model that builds on recent multimodal editing models [43] and fine-tunes them with explicit region cues. RefineAnything injects region cues (scribbles or bounding boxes) into the model’s conditioning, enabling user-specified refinement in both reference-based and reference-free settings. Nevertheless, micro-detail recovery remains challenging when the target region is very small (see Fig. 8), since most modern diffusion models generate

in the VAE latent space and decoding from latents inevitably incurs information loss [17,35,43]; this loss becomes more pronounced when the region itself contains only a limited amount of effective pixel information. This motivates a counterintuitive yet impactful observation (Fig. 3): simply cropping a small target region and upsampling it to the same resolution as the full image—upsampling does not increase the amount of effective pixel information—can yield substantially better VAE reconstruction within the region than reconstructing the full image. Building on this, we introduce Focus-and-Refine (Fig. 4): we refine the focused crop and paste it back with a blended mask, improving refinement effectiveness and efficiency. Focus-and-Refine also naturally enforces background preservation: the blended-mask paste-back guarantees strict background consistency by construction. To further improve paste-back naturalness, we propose a Boundary Consistency Loss that strengthens training supervision near the edit boundary to reduce seam artifacts.

To support training and evaluation at scale, we build Refine-30K, a dataset of 30K samples (20K reference-based and 10K reference-free) constructed with VLM grounding, SAM-based segmentation, and controlled inpainting degradations while explicitly preserving the background. We also introduce RefineEval, a benchmark that evaluates both edited-region fidelity and background preservation in reference-based and reference-free settings.

Extensive experiments on RefineEval show that RefineAnything consistently outperforms the strongest baselines: it improves region fidelity with lower MSE/LPIPS [66] reconstruction errors (0.020/0.155 vs. 0.040/0.264), and strengthens semantic alignment with higher DINO [32, 50]/CLIP [34] similarities and SSIM [42] scores (0.793/0.885/0.591 vs. 0.675/0.807/0.436). Meanwhile, it achieves near-perfect background consistency with lower MSEbg/LPIPSbg errors and higher SSIMbg scores (0.000/0.000/0.9997 vs. 0.011/0.019/0.9660).

In summary, our contributions are three-fold:

- – We formulate region-specific image refinement as a new setting and present RefineAnything, a practical system that improves local details while keeping non-edited regions strictly unchanged.
- – We propose Focus-and-Refine and a boundary-aware Boundary Consistency Loss to enable high-quality refinement with seamless paste-back.
- – We construct Refine-30K and RefineEval to support training and evaluation in both reference-based and reference-free settings, and demonstrate strong improvements in refinement quality, semantic alignment, and background consistency.

### 2 Related Work

Image Generation Models. Image generation has progressed rapidly, delivering high-fidelity images with stronger controllability and instruction following. Modern models largely build upon diffusion models [14]. In particular, the Stable Diffusion family (SD1.5 [35], SDXL [33]) popularizes latent diffusion, where a variational autoencoder (VAE) [17] maps images into a compact latent space

for denoising, significantly accelerating training and sampling; many subsequent models adopt this VAE-based latent framework. Building on this foundation, the community has moved from UNet backbones to better-scaling Diffusion Transformers, such as Hunyuan-DiT [25], PixArt [7], SD3 [13], and FLUX [2]. More recently, multimodal generators (e.g., Qwen-Image [43] and Flux Klein [2]) incorporate VLM encoders (e.g., Qwen2.5-VL [1]) to jointly interpret text and images, broadening real-world applications. Nevertheless, even state-of-the-art models still struggle with fine-grained local details—text, logos, thin structuresmotivating a dedicated local refiner for region-level detail correction.

Image Editing Models. With increasingly capable generators, image editing has gained growing attention [3,5,26,37,41,45,49,52]. FLUX Kontext [18] extends the text-only FLUX.1-dev [2] by incorporating image inputs for editing. OmniGen2 [44] uses modality-separated decoding with non-shared parameters and a decoupled image tokenizer, improving performance and consistency across generation, editing, and context-aware synthesis. BAGEL [11] proposes a Mixture-of-Transformers (MoT) design that couples an understanding model with a generator to better transfer instruction understanding. Qwen-Edit [43] encodes the input image with a VLM and injects its last-layer hidden states into a generative DiT, while also using a VAE to provide fine-detail context. Nevertheless, existing editing models largely focus on coarse-grained manipulations and often struggle with reliable fine-grained local refinement, motivating RefineAnything for region-specific detail enhancement with strict background preservation.

### 3 Method

#### 3.1 Architecture

We propose RefineAnything for localized refinement. Given an input image I, an optional reference image Iref, a user-provided scribble mask M indicating the edit region, and a text instruction y, our goal is to refine the specified region while preserving the rest of the image.

As shown in Fig. 2, our overall framework builds on Qwen-Image [43] and consists of three components: (i) a frozen multimodal encoder (Qwen2.5-VL [1]) that produces refinement-guiding conditioning tokens; (ii) a VAE that maps images to a latent space, providing fine-grained visual context; and (iii) a diffusion backbone built from MMDiT blocks that denoises a target latent under both multimodal and latent conditioning.

High-level multimodal context (VLM). We encode the input (and optional reference) image, the region cue, and the instruction into multimodal conditioning tokens. Let Eϕ(·) denote the frozen Qwen2.5-VL encoder, then

c = Eϕ I, Iref, M, y , c ∈ RL×d, (1)

where L is the token length and d is the feature dimension. These tokens provide high-level guidance (e.g., semantics and instruction intent) to the denoiser via joint-attention [13,25,43,48,63].

|MMDIT Block<br><br>[Figure 36]|
|---|

###### N

…

[Figure 37]

|MMDIT Block<br><br>[Figure 38]|
|---|

Patchify

Patchify

[Figure 39]

| | |
|---|---|
| | |

t

Qwen2.5 VL

Noise

[Figure 40]

[Figure 41]

VAE Encoder

VAE Encoder

| | |
|---|---|
|[Figure 42]| |

| | |
|---|---|
|Refine the LOGO| |

| | |
|---|---|
|[Figure 43]| |

[Figure 44]

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

| |
|---|

| |
|---|

| |
|---|

Reference (Optional)

Reference (Optional)

Input Image User Prompt Target Image Input Image

Spatial Image

- Fig. 2: Architecture of RefineAnything. Given an input image and an optional reference image, the user specifies an edit region via a scribble mask; the images, region cue, and text instruction are encoded by a frozen Qwen2.5-VL encoder into multimodal conditioning tokens. Conditioned on these tokens, a diffusion backbone built from MMDiT blocks (trainable, e.g., via LoRA [15, 47]) denoises a VAE latent from timestep t to produce the locally refined result.

Low-level visual context (VAE latents). We encode the input and optional reference images into VAE latents as low-level fine-grained visual conditioning:

zI = Encψ(I), zref = Encψ(Iref) ∈ RC×H×W, (2)

where Iref is omitted if unavailable. These latents serve as additional conditioning branches (alongside the multimodal tokens c in Eq. 1). We pack them with the noisy target latent zt into patch token sequences and concatenate along the sequence dimension before feeding them into the MMDiT backbone.

Denoising backbone (Qwen-Image). We adopt the MMDiT denoiser from Qwen-Image [43]. It iteratively removes noise from the target latent zt conditioned on both the multimodal tokens c and the VAE latent branches.

Inference. At inference, given (I,Iref,M,y), we start from a noise latent zT and iteratively denoise under the scheduler to obtain z0, which is decoded by the VAE decoder Decψ into the output image I. Conditioning on M steers refinement to the specified region while preserving the rest of the image.

#### 3.2 Focus-and-Refine

Motivation. Under a fixed input pixel budget (e.g., on the order of 1024×1024 pixels for VAE-based pipelines), local refinement is inherently challenging: the model receives only a limited amount of effective pixel information about the fine

[Figure 55]

[Figure 56]

[Figure 57]

1024 resolution 1024 resolution 1024 resolution

| |
|---|

VAE Encoder

VAE Decoder

Worse Result

Crop

[Figure 58]

[Figure 59]

1024 resolution 1024 resolution

Resize 1024

[Figure 60]

VAE Encoder

VAE Decoder

Better Result

- Fig. 3: Motivation for Focus-and-Refine. We compare VAE reconstructing a local region (red box) from the full image versus first cropping the region and resizing it to original full image resolution before VAE encoding. Although the crop-and-resize step does not introduce new information, it substantially improves the reconstruction quality within the target region. This observation suggests that, under a fixed input resolution, directing the model to focus on the local area rather than the entire image leads to better detail recovery for region-specific refinement.

structures to be repaired, since subtle details (e.g., thin strokes) may correspond to only a small number of pixels in the resized input. A natural question is whether we should process the entire image under the same pixel budget, or instead focus the resolution budget on the region of interest.

Surprisingly, our experiments reveal a counter-intuitive phenomenon (Fig. 3): although cropping the target region and resizing it to the same fixed resolution does not introduce any new information, it substantially improves reconstruction quality within the region. In other words, simply re-parameterizing the input by zooming into the region—without changing the model, training data, or compute—already leads to sharper text strokes and cleaner local structures. This suggests that, for region-specific refinement, what limits quality is often not the availability of information, but whether the model is forced to allocate its fixedresolution capacity and attention to the right place. This observation motivates our Focus-and-Refine design

Method. Given an input image I ∈ RH×W×3, an optional reference image Iref, a text instruction y, and a scribble mask M ∈ {0,1}H×W, our goal is to generate a refined image I such that the edit is localized to the region while the rest of the content is preserved. As shown in Fig. 4, Focus-and-Refine consists of three steps: (i) region localization, (ii) focused generation, and (iii) seamless paste-back.

- (i) Region localization and focus crop. We first compute a tight bounding box around the scribble mask (or directly use the user-provided box),

B = BBox(M) = (x1, y1, x2, y2), (3)

###### Stage1: Region Localization and Focus Crop

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Refine text ‘Flower Dance’

Compute BBox

Expand BBox

Crop

| |
|---|

| |
|---|

[Figure 65]

###### Stage2: Focused Refinement with Spatial Conditionin

[Figure 66]

Refine text ‘Flower Dance’

[Figure 67]

[Figure 68]

RefineAnything Resize Model

10242 Pixels

###### Stage3: Seamless Paste-Back via a Blended Mask

[Figure 69]

[Figure 70]

[Figure 71]

|[Figure 72]|
|---|

[Figure 73]

Dilate & Blur

| |
|---|

[Figure 74]

Paste-Back

Fig. 4: Overview of Focus-and-Refine Method.

and expand it with a margin m to obtain the focus crop box

C = Expand(B, m) (4)

clipped to the image boundary. We then crop and resize the input (and the corresponding mask) to obtain the focused view:

Ic = Crop(I, C), Mc = Crop(M, C). (5)

The margin m provides local context (e.g., surrounding texture and illumination) while still concentrating most of the fixed-resolution budget on the target region.

- (ii) Focused generation with spatial conditioning. On the cropped view, we use

the cropped scribble mask Mc as the spatial cue and perform conditional generation on a multi-image input:

X = Ic, Iref, Mc , (6)

where Iref is omitted if unavailable. The model then produces a refined crop

Ic = G(X, y), (7)

where G denotes our RefineAnything Model (Fig. 2).

- (iii) Seamless paste-back via blended mask. Directly replacing the cropped area can introduce visible seams at the crop boundary. We therefore paste the refined

result back using a softened version of the cropped mask Mc. Specifically, we apply morphological dilation and Gaussian smoothing to obtain a blended mask:

Mc = Blur Dilate(Mc; r), k , (8)

where r is the dilate kernel size and k is the blur kernel. We then composite the refined crop with the original crop:

Ic = Mc ⊙ Ic + (1 − Mc) ⊙ Ic, (9)

with element-wise multiplication ⊙. Finally, we resize and paste Ic back to the full canvas at location C to obtain the output image I. This design yields highquality local refinement while maintaining global consistency, and the blended mask effectively suppresses boundary artifacts.

#### 3.3 Boundary Consistency Loss.

To improve paste-back naturalness, we upweight supervision near the edit boundary during training. We define a boundary band

Bc = Dilate(Mc; rout) − Erode(Mc; rin), (10) Following Qwen-Image [43], we adopt the flow-matching denoising objective on the focused crop in latent space. Let z0 denote the latent of the target crop, sample z1 ∼ N(0,I) and t ∈ [0,1], and construct zt = tz0+(1−t)z1 with target velocity vt = z0 −z1. Conditioning on the multimodal tokens c in Eq. 1 and the VAE latent branches zI (and zref if available), the model predicts vθ(zt,t,c,zI,zref), yielding a per-location base loss map ℓbase = vθ(zt,t,c,zI,zref) − vt 22 (summed over channels). We resize Bc to match the spatial resolution of ℓbase and define the boundary-weighted objective as

Lboundary = E ∥ℓbase ⊙ (1 + αBc)∥1 . (11)

#### 3.4 Implementation Details

Training. We fine-tune Qwen-Image-Edit [43] (2509 version) with LoRA [15] on attention projections only (to_q, to_k, to_v, to_out.0): rank 256, lora_alpha 256; only LoRA parameters are optimized. We use AdamW [16] (lr 2 × 10−4, β1 0.9, β2 0.999, weight decay 0.01, ϵ 10−8) with a constant schedule, BF16, batch size 8, and train for 20K steps. Focus-and-Refine. Crop margin m = 64; paste-back mask uses Eq. 8 with dilation kernel size r = 7 and Gaussian blur kernel size k = 11; boundary band uses Eq. 10 with dilation/erosion kernel sizes rout = rin = 16; boundary weighting uses Eq. 11 with α = 9.

### 4 Refine-30K Dataset

We collect Refine-30K, a dataset of 30K samples for training our RefineAnything model. Refine-30K consists of two subsets. The first subset contains 20K reference-based refine pairs: as illustrated in Fig. 6, the model is provided with both a refinement instruction and a reference image, and is expected to refine the input while following the visual style/appearance cues from the reference. The remaining 10K reference-free refine samples are instruction-only: as shown in Fig. 7, users provide only the refinement text to specify how the input should be refined.

Grounding Result Instance Mask Scribble Mask Degraded Image

###### Reference Target Image

|[Figure 75]|
|---|

|[Figure 76]|
|---|

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

christmas tree

| |
|---|

VLM SAM3 Generate Scribble Inpainting

- Fig. 5: Overview of Reference-Based Refine Data Construction Pipeline.

#### 4.1 Reference-Based Refine Data

We build the reference-based subset by converting each collected image pair into a supervised refinement sample (Fig. 5). Each pair consists of a reference image Iref and a target image I⋆, where I⋆ contains the main subject depicted in Iref. Our pipeline produces a degraded input image I to be refined, the corresponding ground-truth target I⋆, a spatial cue mask M, and a text instruction y specifying the refinement goal. We construct each sample in four steps:

- (i) Cross-image grounding. Given (Iref,I⋆), we apply a visual-language model (Gemini3 [39]) to identify the single most salient subject in Iref, verify that the same subject appears in I⋆, and localize it in I⋆ with a bounding box B. To ensure high precision, we enforce strict subject-consistency checks and keep only pairs for which the VLM confidently confirms a match and outputs a valid box.
- (ii) Mask generation with SAM. The bounding box may still include background clutter. We therefore refine localization by segmenting the subject region in I⋆. Specifically, we run SAM (SAM3 [6]) on the target image, conditioned on the

VLM box and a short textual description, and obtain an object mask Mobj. We restrict to a single-instance mask to avoid ambiguous segmentations.

- (iii) Scribble degradation via inpainting. To synthesize challenging refine inputs, we generate local artifacts within the localized subject region. We first sample random scribble strokes and constrain them to lie inside a dilated version of Mobj, yielding the final inpainting mask M. We then inpaint the target image to obtain a degraded image I:

I = Inpaint(I⋆,M). (12)

This step introduces realistic local corruptions while keeping the degradation spatially controlled, and we apply a light paste-back blending to ensure the final input differs from I⋆ only within the edited region.

- (iv) Instruction and outputs. For each sample, we store (I,Iref,I⋆,M,y). The instruction y is derived from the VLM description and explicitly refers to the localized region to align with our region-conditioned refinement setting.

#### 4.2 Reference-Free Refine Data

We construct a reference-free subset from single images, using only a refinement instruction and a spatial cue (no external reference). We synthesize a degraded input while keeping the original as ground truth, and employ a VLM to

###### Input Reference OminiGen2 BAGEL Kontext Qwen-Edit Ours

[Figure 83]

Refine the white cloth of the person

[Figure 84]

Refine the shoe of the person

[Figure 85]

Refine the text logo

[Figure 86]

Refine the face

[Figure 87]

Refine the face

Fig. 6: Qualitative Result on Reference-Based Refinement.

filter implausible or unrecognizable degradations to keep the task well-defined. We construct each sample in four steps:

- (i) Salient object localization. Given a single image I⋆, we first apply a VLM (Gemini3 [39]) to detect salient objects and produce a set of candidate bounding

boxes {Bi} along with short textual descriptions. We then randomly sample one object B to diversify the edited regions across categories and scales.

- (ii) Masking and degradation. We then follow the same segmentation and scribblebased inpainting degradation pipeline as in the reference-based subset: we obtain an object mask Mobj using SAM3 [6], sample a scribble mask M inside a dilated Mobj, and synthesize a degraded input I from I⋆ via inpainting with a light paste-back blending so that I and I⋆ differ only in the intended region.
- (iii) VLM-based defect validation. Not all synthetic degradations lead to meaningful refinement tasks. We therefore employ a VLM to judge whether the degraded image I exhibits noticeable defects (e.g., artifacts, missing structures, or unnatural textures) and whether the degradation is logically plausible given the scene. We discard samples that are judged as (a) having no obvious defect or

Table 1: Evaluation on Reference-Based Image Refinement.

Method MSE↓ LP↓ VGG↓ DINO↑ CLIP↑ SSIM↑ MSEbg↓ LPbg↓ SSIMbg↑

Gemini2.5 0.049 0.250 0.592 0.717 0.817 0.423 0.201 0.103 0.7662 Gemini3 0.031 0.178 0.431 0.771 0.855 0.510 0.029 0.052 0.9061 GPT4o 0.083 0.370 0.918 0.620 0.801 0.302 0.815 0.309 0.6001 OmniGen2 0.155 0.602 1.691 0.384 0.717 0.219 2.094 0.624 0.4300 BAGEL 0.045 0.253 0.611 0.682 0.803 0.494 0.033 0.046 0.9360 Kontext 0.040 0.264 0.540 0.685 0.785 0.538 0.011 0.019 0.9660 Qwen-Edit 0.049 0.287 0.676 0.675 0.807 0.436 0.454 0.148 0.7530

Ours 0.020 0.155 0.401 0.793 0.885 0.591 0.000 0.000 0.9997 ↓: Smaller is better, ↑: Larger is better. Gemini2.5 represents Gemini2.5 flash image, Gemini3 represents Gemini3-pro. LP stands for the LPIPS metric, and DINO stands for the DINOv2Large metric.

(b) being semantically inconsistent/ill-posed, which improves data quality and stabilizes training.

- (iv) Instruction and outputs. Each sample is stored as (I,I⋆,M,y), where y is a reference-free refinement instruction generated from the VLM description of the selected object/region (e.g., “Refine {object} in the masked region”). This subset complements reference-based data by teaching the model to follow textonly refinement cues while maintaining strict background consistency.

### 5 Experiment

#### 5.1 Benchmarks

To evaluate the image refinement capabilities of our model, we construct RefineEval. RefineEval includes two settings: Reference-Based Image Refinement and Reference-Free Image Refinement. The former focuses on identity-sensitive content such as specific logo text, products, and person IDs, while the latter covers common structures including human bodies, generic objects, faces, and text. Each RefineEval case provides a clean target image, a localized edit region, and a refinement instruction (and additionally a reference image in the reference-based setting). We curate 67 cases from open-source websites and manually annotate the regions to be degraded/refined. Following the data construction protocol in Sec. 4, we synthesize degraded inputs via inpainting within the annotated regions, using Flux-fill [2], SDXL [33], and Qwen-Edit [43] to cover varying degradation patterns. For each inpainting method, we generate candidates with 5 randomly sampled scribble masks across 3 different seeds and manually select 2 representative degraded images for evaluation. This results in 402 degraded inputs in total (67 cases × 3 methods × 2 images), including 31 reference-based cases and 36 reference-free cases.

###### Input OminiGen2 BAGEL Kontext Qwen-Edit Ours

[Figure 88]

Refine the face of man

[Figure 89]

Refine the face of woman

[Figure 90]

Refine the hand and cup

[Figure 91]

Refine text ‘STARTING 8|7c’

[Figure 92]

Refine text ‘⽜ ⼤⼈台湾⽕锅 吃到饱’

Fig. 7: Qualitative Result on Reference-Free Refinement.

#### 5.2 Evaluation Metrics

Reference-Based Image Refinement. When a reference image is provided, we evaluate (i) edited-region fidelity and (ii) background preservation. For the edited region, we compare the refined image with the ground-truth (GT) image using MSE, SSIM, LPIPS, VGG, and feature similarities via DINO and CLIP; for the background, we compare the refined image with the input image using MSEbg, LPIPSbg, and SSIMbg. Foreground/background regions are defined by the object bounding box annotations in the benchmark. We use dino-v2 large [32] for DINO and clip-vit-large-patch14-336 [10] for CLIP.

Reference-Free Image Refinement. In the absence of a reference image, refinement is inherently open-ended. We therefore adopt a VLM-based evaluator (Gemini2.5-Pro) and score the expanded foreground crop for each case on five dimensions: visual quality (VQ), naturalness (Nat.), aesthetics (Aes.), finedetail fidelity (Det.), and instruction faithfulness (Faith.). Scores are in [1,5] with decimals allowed (higher is better); prompts are provided in the appendix.

#### 5.3 Baselines

We compare our method with several representative open-source and closedsource approaches for image editing and instruction-based generation, including GPT4o [31], Gemini 3-pro-image-preview [39], Gemini 2.5-flash-image [38],

Table 2: Evaluation on the Reference-Free Image Refinement.

Method VQ↑ Nat.↑ Aes.↑ Det.↑ Faith.↑

OmniGen2 2.501 2.500 2.461 2.348 2.586 BAGEL 3.018 3.000 2.959 2.851 3.135 kontext 1.716 2.114 1.982 1.690 1.750 Qwen-Edit 3.081 3.110 3.105 2.975 3.214 Ours 3.806 3.868 3.876 3.720 3.644

Table 3: Ablation on Focus-and-Refine and Boundary Consistency Loss.

Method MSE↓ LP↓ VGG↓ DINO↑ CLIP↑ SSIM↑ MSEbg↓ LPbg↓ SSIMbg↑

w/o focus 0.021 0.177 0.449 0.779 0.869 0.578 0.005 0.022 0.9601 w/o loss 0.023 0.191 0.482 0.736 0.858 0.563 0.000 0.000 0.9997 Ours 0.020 0.155 0.401 0.793 0.885 0.591 0.000 0.000 0.9997

Qwen-Image-Edit [43], BAGEL [11], OmniGen2 [44], and Kontext [18]. More details are provided in the supplementary material.

#### 5.4 Quantitative Results

Tab. 1 shows that our method achieves the best overall performance on referencebased refinement, jointly improving edited-region fidelity and background preservation. Compared to the best open-source baseline (Kontext), we reduce MSE by 0.020 (50%), LPIPS by 0.109 (41%), and VGG by 0.139 (26%), and improve DINO and CLIP by +0.108 and +0.100, respectively. Meanwhile, we deliver near-perfect background consistency (MSEbg = 0.000, LPbg = 0.000, SSIMbg = 0.9997), eliminating background drift (e.g., Kontext: MSEbg = 0.011; Qwen-Edit: MSEbg = 0.454). For reference-free refinement, Tab. 2 reports MLLM-based subjective scores across five dimensions (VQ, Nat., Aes., Det., and Faith.). Our method ranks first on all criteria, surpassing the strongest open-source baseline (Qwen-Edit) by +0.725, +0.758, +0.771, +0.745, and +0.430 on VQ, Nat., Aes., Det., and Faith., respectively, indicating more natural, detailed, and instructionfaithful refinements even without a reference image.

#### 5.5 Qualitative Results

- Fig. 6 and Fig. 7 present a qualitative comparison between our method and state-of-the-art baselines on reference-based and reference-free refinement. Prior methods often suffer from poor background preservation, weak responsiveness to the instruction/reference, and limited ability to recover fine details. In contrast, empowered by our Focus-and-Refine strategy, our approach not only restores subtle details more effectively but also keeps the background strictly unchanged, substantially improving practicality and real-world usability.

###### Input Reference w/o w/ Input w/o w/

[Figure 93]

[Figure 94]

Refine Text

Refine logo

‘SHE COULD’

###### Fig. 8: Ablation of the Focus-and-Refine strategy.

[Figure 95]

###### Input Reference w/o w/ Input w/o w/

[Figure 96]

Refine logo

Refine hand

Fig. 9: Ablation of the Boundary Consistency Loss.

#### 5.6 Ablation Study

Focus-and-Refine. As shown in Fig. 8 and Tab. 3, removing the focusing step leads to weaker refinements, often leaving subtle errors unresolved and occasionally introducing artifacts. In contrast, Focus-and-Refine allocates the model’s capacity to the target region, producing sharper local details while keeping the surrounding background unchanged.

Boundary Consistency Loss. As shown in Fig. 9 and Tab. 3, removing the Boundary Consistency Loss leads to poor coherence between the locally refined region and its surrounding context. This often manifests as visible seams, color inconsistencies, and structurally implausible stitching along object boundaries.

### 6 Conclusion

We introduced RefineAnything, the first framework tailored for region-specific image refinement—improving fine-grained local details while keeping non-edited regions strictly unchanged. Motivated by the observation that crop-and-resize can significantly boost local reconstruction under a fixed input resolution, we proposed Focus-and-Refine, which concentrates model capacity on the target region and then pastes the refined result back with a blended mask for seamless integration. To further enhance paste-back naturalness, we introduced a boundaryaware Boundary Consistency Loss that encourages seam-consistent refinements during training. We also built Refine-30K and the RefineEval benchmark to support training and evaluation in both reference-based and reference-free settings. Extensive experiments demonstrate that our approach improves local detail fidelity and semantic alignment while achieving near-perfect background preservation. We hope RefineAnything and our datasets will facilitate future research on practical, high-precision refinement for real-world image generation and editing workflows.

### References

- 1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)
- 2. BlackForest: Black forest labs; frontier ai lab (2024), https://blackforestlabs. ai/
- 3. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18392–18402 (2023)
- 4. Cai, Q., Chen, J., Chen, Y., Li, Y., Long, F., Pan, Y., Qiu, Z., Zhang, Y., Gao, F., Xu, P., et al.: Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705 (2025)
- 5. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 22560–22570 (October 2023)
- 6. Carion, N., Gustafson, L., Hu, Y.T., Debnath, S., Hu, R., Suris, D., Ryali, C., Alwala, K.V., Khedr, H., Huang, A., et al.: Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719 (2025)
- 7. Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wang, Z., Kwok, J.T., Luo, P., Lu, H., Li, Z.: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In: ICLR. OpenReview.net (2024)
- 8. Chen, Z., Li, Y., Wang, H., Chen, Z., Jiang, Z., Li, J., Wang, Q., Yang, J., Tai, Y.: Ragd: Regional-aware diffusion model for text-to-image generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19331–19341

(2025)

- 9. Chen, Z., Zhu, J., Chen, X., Zhang, J., Hu, X., Zhao, H., Wang, C., Yang, J., Tai, Y.: Dip: Taming diffusion models in pixel space. arXiv preprint arXiv:2511.18822

(2025)

- 10. Chen, Z., Liu, G., Zhang, B.W., Ye, F., Yang, Q., Wu, L.: Altclip: Altering the language encoder in clip for extended language capabilities. arXiv preprint arXiv:2211.06679 (2022)
- 11. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., et al.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025)
- 12. Du, N., Chen, Z., Gao, S., Chen, Z., Chen, X., Jiang, Z., Yang, J., Tai, Y.: Textcrafter: Accurately rendering multiple texts in complex visual scenes. arXiv preprint arXiv:2503.23461 (2025)
- 13. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: ICML (2024)
- 14. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. In: NeurIPS. pp. 6840–6851 (2020)
- 15. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)
- 16. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization (2017)
- 17. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

- 18. Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al.: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 (2025)
- 19. Li, D., Zhang, H., Wang, S., Li, J., Wu, Z.: Seg2any: Open-set segmentationmask-to-image generation with precise shape and semantic control. arXiv preprint arXiv:2506.00596 (2025)
- 20. Li, M., Yang, T., Kuang, H., Wu, J., Wang, Z., Xiao, X., Chen, C.: Controlnet++: Improving conditional controls with efficient consistency feedback. arXiv preprint arXiv:2404.07987 (2024)
- 21. Li, M., Yang, T., Kuang, H., Wu, J., Wang, Z., Xiao, X., Chen, C.: Controlnet++: Improving conditional controls with efficient consistency feedback. In: European Conference on Computer Vision. pp. 129–147. Springer (2025)
- 22. Li, Y., Ma, F., Yang, Y.: Anysynth: Harnessing the power of image synthetic data generation for generalized vision-language tasks. arXiv preprint arXiv:2411.16749

(2024)

- 23. Li, Y., Ma, F., Yang, Y.: Imagine and seek: Improving composed image retrieval with an imagined proxy. In: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR). pp. 3984–3993 (June 2025)
- 24. Li, Y., Zhou, D., Ma, F., Li, F., He, D., Yang, Y.: Foleydirector: Fine-grained temporal steering for video-to-audio generation via structured scripts. arXiv preprint arXiv:2603.19857 (2026)
- 25. Li, Z., Zhang, J., Lin, Q., Xiong, J., Long, Y., Deng, X., Zhang, Y., Liu, X., Huang, M., Xiao, Z., Chen, D., He, J., Li, J., Li, W., Zhang, C., Quan, R., Lu, J., Huang, J., Yuan, X., Zheng, X., Li, Y., Zhang, J., Zhang, C., Chen, M., Liu, J., Fang, Z., Wang, W., Xue, J., Tao, Y., Zhu, J., Liu, K., Lin, S., Sun, Y., Li, Y., Wang, D., Chen, M., Hu, Z., Xiao, X., Chen, Y., Liu, Y., Liu, W., Wang, D., Yang, Y., Jiang, J., Lu, Q.: Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding (2024)
- 26. Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., et al.: Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761 (2025)
- 27. Lu, S., Lian, Z., Zhou, Z., Zhang, S., Zhao, C., Kong, A.W.K.: Does flux already know how to perform physically plausible image composition? arXiv preprint arXiv:2509.21278 (2025)
- 28. Lu, S., Liu, Y., Kong, A.W.K.: Tf-icon: Diffusion-based training-free cross-domain image composition. In: ICCV (2023)
- 29. Lu, S., Wang, Z., Li, L., Liu, Y., Kong, A.W.K.: Mace: Mass concept erasure in diffusion models. CVPR (2024)
- 30. Lu, S., Zhou, Z., Lu, J., Zhu, Y., Kong, A.W.K.: Robust watermarking using generative priors against image editing: From benchmarking to advances. arXiv preprint arXiv:2410.18775 (2024)
- 31. OpenAI: Gpt-4o. https : / / openai . com / index / introducing - 4o - image generation/ (2025)
- 32. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 33. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023)

- 34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML. pp. 8748–8763 (2021)
- 35. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models (2022)
- 36. Shi, X., Li, B., Han, X., Cai, Z., Yang, L., Lin, D., Wang, Q.: Consistcompose: Unified multimodal layout control for image composition. arXiv preprint arXiv:2511.18333 (2025)
- 37. Shi, Y., Wang, P., Huang, W.: Seededit: Align image re-generation to image editing

(2024), https://arxiv.org/abs/2411.06686

- 38. Team, G.: Gemini 2.5 flash & gemini 2.5 flash image model card (2025)
- 39. Team, G.: Gemini 3.0 pro & gemini 3.0 pro image model card (2025)
- 40. Team, S., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al.: Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427 (2025)
- 41. Wang, Y., Yang, S., Zhao, B., Zhang, L., Liu, Q., Zhou, Y., Xie, C.: Gptimage-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033 (2025)
- 42. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)
- 43. Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S.m., Bai, S., Xu, X., Chen, Y., et al.: Qwen-image technical report. arXiv preprint arXiv:2508.02324 (2025)
- 44. Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al.: Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871 (2025)
- 45. Xie, J., Darrell, T., Zettlemoyer, L., Wang, X.: Reconstruction alignment improves unified multimodal models. arXiv preprint arXiv:2509.07295 (2025)
- 46. Xu, R., Zhou, D., Ma, F., Yang, Y.: Contextgen: Contextual layout anchoring for identity-consistent multi-instance generation. arXiv preprint arXiv:2510.11000

(2025)

- 47. Xu, Y., He, Z., Shan, S., Chen, X.: Ctrlora: An extensible and efficient framework for controllable image generation. arXiv preprint arXiv:2410.09400 (2024)
- 48. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)
- 49. Ye, Y., He, X., Li, Z., Lin, B., Yuan, S., Yan, Z., Hou, B., Yuan, L.: Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275

(2025)

- 50. Zhang, H., Li, F., Liu, S., Zhang, L., Su, H., Zhu, J., Ni, L.M., Shum, H.Y.: Dino: Detr with improved denoising anchor boxes for end-to-end object detection. arXiv preprint arXiv:2203.03605 (2022)
- 51. Zhang, H., Duan, Z., Wang, X., Chen, Y., Zhang, Y.: Eligen: Entity-level controlled image generation with regional attention. arXiv preprint arXiv:2501.01097 (2025)
- 52. Zhang, H., Duan, Z., Wang, X., Chen, Y., Zhao, Y., Zhang, Y.: Nexus-gen: A unified model for image understanding, generation, and editing. arXiv preprint arXiv:2504.21356 (2025)
- 53. Zhang, H., Hong, D., Gao, T., Wang, Y., Shao, J., Wu, X., Wu, Z., Jiang, Y.G.: Creatilayout: Siamese multimodal diffusion transformer for creative layout-to-image generation. arXiv preprint arXiv:2412.03859 (2024)

- 54. Zhang, H., Hong, D., Yang, M., Cheng, Y., Zhang, Z., Shao, J., Wu, X., Wu, Z., Jiang, Y.G.: Creatidesign: A unified multi-conditional diffusion transformer for creative graphic design. arXiv preprint arXiv:2505.19114 (2025)
- 55. Zhang, Z., Xie, J., Lu, Y., Yang, Z., Yang, Y.: Enabling instructional image editing with in-context generation in large scale diffusion transformer. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)
- 56. Zhao, C., Cai, W., Dong, C., Hu, C.: Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. CVPR

(2024)

- 57. Zhao, C., Cai, W., Dong, C., Zeng, Z.: Toward sufficient spatial-frequency interaction for gradient-aware underwater image enhancement. In: ICASSP 20242024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 3220–3224. IEEE (2024)
- 58. Zhao, C., Chen, J., Li, H., Kang, Z., Lu, S., Wei, X., Zhang, K., Yang, J., Tai, Y.: Luve: Latent-cascaded ultra-high-resolution video generation with dual frequency experts. arXiv preprint arXiv:2602.11564 (2026)
- 59. Zhao, C., Chen, Z., Xu, Y., Gu, E., Li, J., Yi, Z., Wang, Q., Yang, J., Tai, Y.: From zero to detail: Deconstructing ultra-high-definition image restoration from progressive spectral perspective. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 17935–17946 (2025)
- 60. Zhao, C., Ci, E., Xu, Y., Fan, T., Guan, S., Ge, Y., Yang, J., Tai, Y.: Ultrahr100k: Enhancing uhr image synthesis with a large-scale high-quality dataset. arXiv preprint arXiv:2510.20661 (2025)
- 61. Zhao, C., Dong, C., Cai, W.: Learning a physical-aware diffusion model based on transformer for underwater image enhancement. arXiv preprint arXiv:2403.01497

(2024)

- 62. Zhou, D., Li, M., Yang, Z., Lu, Y., Xu, Y., Wang, Z., Huang, Z., Yang, Y.: Bidedpo: Conditional image generation with simultaneous text and condition alignment. arXiv preprint arXiv:2511.19268 (2025)
- 63. Zhou, D., Li, M., Yang, Z., Yang, Y.: Dreamrenderer: Taming multi-instance attribute control in large-scale text-to-image models. In: ICCV (2025)
- 64. Zhou, D., Li, Y., Ma, F., Zhang, X., Yang, Y.: Migc: Multi-instance generation controller for text-to-image synthesis. In: CVPR (2024)
- 65. Zhou, D., Xie, J., Yang, Z., Yang, Y.: 3dis: Depth-driven decoupled instance synthesis for text-to-image generation. arXiv preprint arXiv:2410.12669 (2024)
- 66. Zhou, D., Yang, Z., Yang, Y.: Pyramid diffusion models for low-light image enhancement. In: IJCAI (2023)
- 67. Zhou, Z., Lu, S., Leng, S., Zhang, S., Lian, Z., Yu, X., Kong, A.W.K.: Dragflow: Unleashing dit priors with region based supervision for drag editing. arXiv preprint arXiv:2510.02253 (2025)

