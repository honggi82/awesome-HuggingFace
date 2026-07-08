## ControlLight: Towards Controllable, Consistent, and Generalizable Low-Light Enhancement

Yufeng Yang1 Jianzhuang Liu1,† Jisheng Chu1 Yuqi Peng1 Xianfang Zeng2 Jiancheng Huang1 Shifeng Chen1,‡ 1Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences 2Zhejiang University

# arXiv:2605.25569v2[cs.CV]26May2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

#### Project Page Models Light100K Code

Enhancement Strength Input s = 0.25 s = 0.5 s = 0.75 s = 1.00 0 1

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

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

Figure 1: Given a low-light input, ControlLight supports continuous adjustment of the enhancement strength from s = 0 to s = 1, producing smooth and controllable restoration consistent results across real-world scenes.

### Abstract

Existing deep learning-based low-light enhancement methods are typically trained on limited datasets with single enhancement targets, which restricts their generalization ability and controllability in real-world applications. To overcome these limitations, we propose ControlLight, a controllable, consistent, and generalizable framework for low-light enhancement. We first construct a large-scale dataset of real-world degraded images with continuous illumination-strength supervision. To further ensure consistent outputs under different control strengths, we introduce a misalignment-aware weighted flow matching loss that preserves image structure across continuous enhancement strengths. ControlLight allows users to edit real-world degraded low-light images toward satisfactory enhancement results by flexibly controlling the strength while preserving visual consistency and realism. Extensive experiments show that ControlLight achieves state-of-the-art performance against existing low-light enhancement approaches while demonstrating strong continuous controllability and generalization to real-world scenarios.

†leads this project; ‡Corresponding authors.

Preprint.

### 1 Introduction

Low-light enhancement aims to recover degraded images captured under low-light conditions by restoring details in dark regions while suppressing noise. With the development of deep learning, many methods [2, 4, 28, 45, 57, 43, 58, 40] have demonstrated strong capability in low-light image restoration. However, most existing datasets typically provide only a single supervision target for each low-light image, forcing the model to learn a fixed enhancement strength without controllability. This limitation is critical in practical applications, where users often need to freely adjust the enhancement strength according to different images and personal preferences.

Meanwhile, large-scale image editing models [5, 22, 46, 35], such as Nano Banana Pro [34] and FLUX.2-klein [18], have demonstrated strong generalization ability across both high-level and low-level vision tasks. Trained on massive image–text paired data, these models possess powerful generative priors and can recover visually plausible details while largely preserving the overall scene structure, making them promising for low-light enhancement. However, most large image editing models provide a single enhancement strength by giving instructions and may introduce hallucinated textures or structural distortions due to their generative nature, which limits their fine-grained controllability and reliability in practical low-light enhancement scenarios.

To address these issues, we construct Light100K, a continuous low-light enhancement dataset containing real degraded low-light images and structure-consistent pseudo-enhanced targets with different illumination strengths. This dataset provides fine-grained supervision for controllable enhancement.

We further observe that diffusion-generated pseudo targets, despite offering strong appearance supervision, may contain subtle edge misalignment with the input images. Directly applying flow matching to such targets can cause the model to inherit and amplify these offsets, leading to structural artifacts. To mitigate this issue, we propose a Misalignment-Aware Weighted Flow Matching Loss, which down-weights unreliable target-edge regions and encourages structure preservation from the input image.

Based on Light100K and the proposed Misalignment-Aware Weighted Flow Matching Loss, we train ControlLight on FLUX.2-klein-9B with LoRA [11]. By conditioning on the LoRA strength, ControlLight enables continuous and fine-grained low-light enhancement, producing smooth illumination changes while preserving scene structure.

In summary, our contributions are threefold:

- • We construct Light100K, a continuous low-light enhancement dataset containing training groups, providing fine-grained supervision for controllable low-light enhancement.
- • We reveal that visually plausible diffusion-generated pseudo pairs can still contain subtle edge misalignment, and propose a Misalignment-Aware Weighted Flow Matching Loss that anchors the enhanced output edges to the input image structure while down-weighting unreliable target-edge regions.
- • We develop ControlLight, a continuous low-light enhancement model that produces smoothly controllable enhancement results and achieves state-of-the-art performance compared with both continuous and non-continuous low-light enhancement methods.

### 2 Related Work

#### 2.1 Low-light Enhancement Methods

Many deep learning-based low-light enhancement methods [40, 41, 6] incorporate classical imaging priors, especially Retinex theory [19], to restore image brightness. With paired datasets such as LOL [51] and LSRW [10], these methods learn mappings from low-light inputs to normallight outputs. EnlightenGAN [15] learns enhancement from unpaired normal-light images with a GAN-based framework [59], while Retinexformer [2] uses illumination information to guide a Transformer [37]. CIDNet [49] further revisits brightness restoration from the HSV color space.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

a) Low-light Enhancement

b) Sequence Generation

c) Filtering

Input I0 Enhanced I1

Input I0

Enhanced I1

Image sequence

[Figure 26]

[Figure 27]

L(x,y)

L(x,y)

L0

L1

###### Interpolate

Retinex Decomp.

Retinex Decomp.

Semantic Filtering

Li Ri

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Retinex Recon.

R1

R0

R(x,y)

R(x,y)

Degradation Filtering

L(x,y) · R(x,y)

Flux2 Klein

Edge Consistency Filtering

"Restore this low-quality image, recover its normal brightness and clarity, and keep the structure the same."

I0 strength s1 strength s2 strength s3 strength s4 I1

- Figure 2: Main data construction pipeline of Light100K. FLUX.2-klein-9B is used to generate normal-light references from real low-light images. We then apply Retinex-inspired decomposition and selective interpolation to construct highly consistent and continuous pseudo-paired data from each pair (I0, I1).

These methods are generally trained under fixed supervision and therefore tend to produce results with a single enhancement strength. This limitation makes them less suitable for scenarios where flexible brightness control is required.

To address this issue, several works have investigated controllable low-light enhancement. ReCoRo [48] adopts GANs to learn enhancement from images with different brightness levels. CLE Diffusion [53] employs a conditional diffusion model that uses brightness alpha blending target images as guidance, enabling controllable enhancement to some extent. Nevertheless, limited by model capacity and the relatively simple interpretation of the training data construction, CLE Diffusion often struggles to generalize to real-world continuous low-light enhancement and may produce noticeable artifacts.

- 2.2 Image Editing Methods and Continuous Control

Large-scale image editing models [18, 22, 46, 35, 12, 8, 32, 39, 31] have shown strong potential for restoration by leveraging semantic priors learned from massive image–text pairs. However, their generative nature can introduce hallucinations, pixel shifts, and structural deformation, which are undesirable for restoration tasks requiring content consistency. Although high-quality data and consistency reward models [16] can alleviate this issue, existing instruction-based editing methods still lack reliable continuous control.

Recent methods [1, 7, 24, 55, 33, 27] achieve continuous editing through interpolatable text embeddings, modulation features, or low-rank adaptors, but are limited by scarce continuous supervision. Kontinuous Kontext (KSlider) [24] synthesizes continuous samples via morphing [3], which is difficult to keep consistent for global restoration tasks. ConceptSlider [7] learns controllable LoRA directions, but its control can be unstable without intermediate supervision. To address these limitations, we use Retinex theory to construct continuous pseudo-paired supervision and train a controllable LoRA on FLUX.2-klein-9B. We further propose a Misalignment-Aware Weighted Flow Matching Loss to reduce pixel-level inconsistency during continuous enhancement.

- 3 Method

- 3.1 Light100K: Continuous Pseudo-Paired Data Construction

To address the limited availability of real paired training data, we construct paired data from realworld low-light images rather than relying solely on synthetic degradation generated from traditional single-degradation models.

Specifically, we collect high-quality images from open-source image websites, including Pexels and Pinterest, using low-light-related keywords. To build a high-quality real-world degradation dataset, we conduct low-light semantic and degradation filtering. After filtering, we obtain approximately 30K high-quality low-light images.

Given a low-light image I0, we use a fixed enhancement prompt and the pretrained FLUX.2-klein-9B model to generate its enhanced counterpart I1. To avoid supervision from structurally inconsistent pseudo pairs, we remove severely mismatched samples using an edge-consistency filtering strategy,

I0 I1 Alpha blending Ours

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Figure 3: Visual comparison of intermediate pseudo-GT construction at s = 0.5. Retinex-based interpolation yields more natural illumination transitions and better local contrast than direct alpha blending, making it better suited for continuous low-light enhancement.

leaving approximately 20K high-quality paired samples. For each retained pair (I0,I1), we further construct a continuous pseudo-paired training group:

##### G = {I0,I0.2,I0.4,I0.6,I0.8,I1},

where Is denotes the pseudo ground-truth image at enhancement strength s ∈ {0.2,0.4,0.6,0.8}.

A straightforward strategy is alpha blending [53], i.e., Isalpha = (1 − s)I0 + sI1. However, direct RGB-space averaging mixes illumination, reflectance, color, and local contrast, making it suboptimal for continuous low-light enhancement where the target should mainly follow a gradual illumination transition.

To construct a more illumination-consistent trajectory, we propose a Retinex-inspired interpolation strategy as shown in Figure 2. The key idea is to use the Retinex [19] image formation model I = R ⊙ L, where R represents reflectance-related scene content and L represents illumination. Under this model, continuous enhancement is primarily modeled as a transition in the illumination component rather than as a direct interpolation of the whole image appearance.

Specifically, we first convert I0 and I1 from sRGB to linear RGB, and use the same notation for simplicity. We compute their luminance maps by Y = 0.2126R + 0.7152G + 0.0722B, and

estimate illumination maps using edge-preserving smoothing (bilateral filter): L0 = Smooth(Y0) and L1 = Smooth(Y1), since illumination is assumed to be spatially smooth. The reflectance maps are then estimated according to the Retinex model as R0 = I0/L0 and R1 = I1/L1.

We interpolate only the illumination maps in the log domain rather than I0 and I1 in RGB space:

Ls = exp((1 − s)log(L0) + slog(L1)). (1)

This is equivalent to a multiplicative interpolation Ls = L10−sLs1, which is more consistent with the Retinex assumption than additive image-space averaging. In parallel, we conservatively interpolate

the reflectance as Rs = (1 − βs)R0 + βsR1, where βs = 0.5s. This design avoids relying only on R0, which may contain amplified low-light noise, while also avoiding excessive dependence on R1, which may inherit artifacts or subtle structural deviations from the diffusion-generated target. The intermediate pseudo-GT is finally reconstructed as:

Is = clip(Rs ⊙ Ls,0,1). (2) The reconstructed image is then converted back to sRGB space for training. More details about the data contrsuction pipeline and the Light100K is provided in the Appendix A.

In Figure 3, the direct RGB-space averaging of Alpha blending flattens shadows and textures by weakening local contrast, while the nonlinear illumination transition of Retinex interpolation preserves local shading, scene depth, and contrast variations. Thus, our use of Retinex interpolation provides a more illumination-aware pseudo-GT trajectory for continuously controllable low-light enhancement.

#### 3.2 Misalignment-Aware Weighted Flow Matching

Although the filtered pseudo pairs {I0,I1} are visually well aligned, they may still contain subtle pixel-level edge misalignment. Such misalignment is difficult to observe directly in RGB space, as

I0 ℒ𝐹𝑀 ℒ𝑤𝐹𝑀 I1

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Structural

Edge Edge-

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

difference

- Figure 4: Visualization of edge misalignment and the effect of weighted flow matching. Compared with

standard flow matching, the proposed LwFM produces enhanced results with weaker edge-difference responses and better structural alignment to the input.

the dominant differences between I0 and I1 mainly arise from brightness and color variations. After normalizing illumination, however, the remaining high-frequency residuals reveal local structural edge discrepancies. As shown in Figure 4, even a pair that satisfies our matching criterion can still exhibit non-negligible edge differences. When FLUX.2-klein-9B is fine-tuned with the standard flow matching loss [21, 5, 26], these misaligned edges may be inherited and amplified, leading to visible structural drift in the enhanced output I1FM. To address this issue, we introduce a misalignment-aware weighted flow matching loss that reduces the supervision strength in unreliable target-edge regions across the continuous pseudo-paired sequence generated from the same degraded image.

To visualize edge misalignment, we employ a structural edge-difference map that focuses on illumination-invariant features. Specifically, we first convert the images to the log-luminance domain and remove slow-varying brightness by subtracting a smoothed version (via a bilateral filter) to isolate the high-pass structural component H(I). We then compute a high-frequency edge response defined as E(I) = ∥∇H(I)∥1, where ∇ denotes the gradient operator. Finally, the edge-difference map between any two images A and B is calculated as Iedge-diff(A,B) = |E(A) − E(B)|. This operation effectively suppresses low-frequency illumination and color discrepancies, ensuring the resulting response primarily reflects local structural misalignments rather than brightness variations.

As shown in Figure 4, the columns correspond to the input I0, the output trained with LFM (I1FM), our output (I1wFM), and the pseudo target I1. The second row shows the extracted structural edge maps, while the third row shows the edge-difference maps computed with respect to I0. Compared with standard flow matching, our weighted loss produces fewer edge-difference responses, indicating better preservation of the input structure.

In standard flow matching, given a target image Is at enhancement strength s, we encode it into the latent space as z1, sample a noise latent z0, and construct an intermediate latent zt = (1 − t)z0 + tz1, where t ∈ [0,1]. The model predicts a velocity field vθ(zt,I0,s), and the standard objective is:

##### LFM = ∥vθ(zt,I0,s) − v∗∥22 , (3)

where v∗ = z1 − z0. This objective treats all spatial regions of the pseudo target equally. Therefore, if Is contains misaligned edges, the model is still encouraged to reproduce those unreliable structures.

We instead assign lower weights to unreliable target-edge regions. For each pseudo target Is, we compute binary edge maps B0 and Bs from I0 and Is, respectively. We then compute the distance transform D0 to the nearest edge pixel in B0. A target edge pixel is regarded as unreliable if it is far from any input edge:

##### Ms(p) = 1[Bs(p) = 1 and D0(p) > d], (4)

where d is a distance threshold. We dilate Ms slightly to cover the neighborhood around the mismatched edge and obtain a soft weight map:

##### Ws(p) = clip(1 − αMs(p),wmin,1). (5)

(a) Inference Pipeline

Text Prompt

"Please restore this low-q y …"

[Figure 54]

VAE Encoder

I0

Frozen

LoRA

[Figure 55]

Transformer

Single-StreamBlock

[Figure 56]

[Figure 57]

I0 Edge

[Figure 58]

[Figure 59]

Edge Diff

- Figure 5: Overview of the proposed framework. (a) During training, the low-light input image and a fixed restoration prompt are encoded and fed into FLUX.2-klein, where LoRA is used for efficient fine-tuning. The enhancement strength s modulates both the LoRA scaling factor and the pseudo ground-truth selection. (b) The

edge mask is generated offline from input and target edges, producing the weight map Ws. At inference time, s can be set to any value in [0, 1].

Details of the weight map generation and the hyperparameters d, α, and wmin are provided in Appendix B.

The image-space weight map Ws is resized to the latent resolution as Ws, which is then applied over latent spatial locations u, to reweight the flow matching objective:

Ws(u)∥vθ(zt,I0,s)(u) − v∗(u)∥22 u Ws(u)

. (6)

LwFM = u

Here, Ws(p) remains positive even in unreliable regions, so the model still receives weak appearance supervision but is no longer forced to exactly fit misaligned pseudo-target edges.

#### 3.3 ControlLight

Given the continuous pseudo-paired dataset (Section 3.1) and the misalignment-aware loss (Eq. 6), we now describe how s is incorporated into the model. The Retinex formulation I = R⊙L suggests that continuous enhancement is primarily a smooth transition along the illumination axis, approximately linear in some parameter subspace. This motivates using s directly as the LoRA scaling factor:

W′ = W + s · AB, (7) where W is frozen and A, B are learnable low-rank matrices.

This formulation resembles ConceptSlider [7], but the training regimes differ critically. Concept Sliders optimize a LoRA direction via text-guided score matching between opposing prompts, with the scaling factor s applied only at inference time. The linearity of control is assumed but never enforced. In contrast, our s enters the training loop: each s ∈ {0.2,0.4,0.6,0.8,1.0} is paired with a pseudo ground truth Is, and LwFM is computed against that target. The LoRA direction is therefore calibrated against a physically grounded illumination trajectory with per-strength supervision, which is the key reason ControlLight achieves substantially better trajectory smoothness than Concept Sliders (Table 3).

During training, the input image and fixed text prompt are encoded by Flux2-VAE [18] and Qwen3VL [36], respectively. Since the prompt remains fixed, the Qwen3-VL text encoder can be offloaded

during inference. The weight maps Ws are precomputed offline. We train at 1024 × 1024 resolution with a fixed learning rate of 1 × 10−4 and a global batch size of 16. The LoRA modules contain about 300M trainable parameters. Additional implementation details are provided in Appendix C.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Input Retinexformer CIDNet CLE Diffusion

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

LLFormer DarkIR QuadPrior Ours

- Figure 6: Visual comparison on DICM Benchmark. The arrows indicate increasing enhancement strength from left to right. Compared with CLE Diffusion, our method produces smoother and more continuous enhancement transitions while better preserving natural color and scene structure.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

Input Retinexformer CIDNet CLE Diffusion

LLFormer DarkIR QuadPrior Ours

- Figure 7: Visual comparison on RealIR-Bench. The arrows indicate increasing enhancement strength from left to right. Compared with traditional methods, our method achieves more natural restoration and better preserves scene structure. The arrows indicate increasing enhancement strength, showing that our model provides continuous and controllable low-light enhancement.

### 4 Experiments

#### 4.1 Quantitative Metrics and Evaluation Protocol

ControlLight is compared with two baseline groups: low-light enhancement methods and universal continuous image editing methods. For low-light enhancement, we evaluate on five benchmarks: LOL [51] and LWSR [10] with paired reference, as well as real-world DICM [20], LIME [9], and RealIR-Bench [52] with non-reference. As generative restoration models such as SUPIR [54] may synthesize perceptually plausible details that are penalized by reference-based metrics like PSNR and SSIM [44], we mainly report non-reference perceptual metrics, including CLIP-IQA [38], MUSIQ [17], NIQE [23], and MANIQA [50]. To further evaluate Linear Control, we compare ControlLight with universal continuous image editing methods on real-world non-reference test sets,

- as they can potentially perform continuous low-light enhancement. We assess the smoothness and directionality of the enhancement trajectory using δsmooth [24] and CLIP-Dir [25], respectively.

#### 4.2 Low-light Enhancement Evaluation

We compare ControlLight with several state-of-the-art low-light enhancement methods on both paired and unpaired benchmarks. For paired evaluation, we use LOL-v1 [51], which contains 15 testing images, and the LWSR test set [10], which contains 50 testing images. For LWSR, we report the average performance over the Huawei and Nikon subsets. The compared methods include Retinexformer [2], HVI-CIDNet [49], LLFormer [40], DarkIR [6], CLE Diffusion [53], and QuadPrior [42].

Since ControlLight is a continuous enhancement model and does not rely on a single fixed enhancement level, we evaluate it at four enhancement strengths, i.e., s ∈ {0.25,0.50,0.75,1.00}, and report the average score. For CLE Diffusion, in paired testing scenarios, the method can use the ground-truth reference to guide result selection. For test sets without ground-truth references, we evaluate CLE Diffusion under the same four-strength setting as ControlLight for a fair comparison.

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Input Retinexformer CIDNet LLFormer

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

DarkIR QuadPrior CLE Diffusion GT

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Ours s=0.25

Ours s=0.50

Ours s=0.75

Ours s=1.00

- Figure 8: Visual comparison on LOL-v1 benchmark. Our zero-shot results may deviate from the ground truth in color appearance, but they provide natural visual quality with preserved structures and textures. The outputs

- at different enhancement strengths s show smooth and approximately linear low-light enhancement control.

- Table 1: Quantitative comparison on paired enhancement benchmarks: LOL-v1 [51] and LWSR [10]. The best and second-best results are highlighted with yellow and purple backgrounds, respectively.

LOL-v1 [51] LWSR [10] NIQE↓ CLIPIQA↑ MANIQA↑ MUSIQ↑ NIQE↓ CLIPIQA↑ MANIQA↑ MUSIQ↑

Method

Retinexformer [2] 3.455 0.429 0.383 63.16 3.778 0.420 0.401 58.48 CIDNet [49] 4.110 0.488 0.511 71.91 3.708 0.415 0.387 56.25 LLFormer [40] 3.580 0.331 0.317 60.77 3.791 0.394 0.360 57.44 DarkIR [6] 5.335 0.389 0.41 70.69 4.103 0.462 0.431 64.45 QuadPrior [42] 5.184 0.367 0.295 58.81 5.045 0.345 0.358 58.91 CLE Diffusion [53] 4.893 0.581 0.435 68.84 4.265 0.491 0.388 62.63

ControlLight (Ours) 4.567 0.553 0.512 70.20 4.232 0.589 0.494 68.39

As shown in Table 1 and Table 2, our method achieves the best results on most metrics among domainspecific methods on paired benchmarks, and consistently outperforms all baselines on real-world benchmarks. This demonstrates its strong generalization capability under real-world degradations.

- Figure 8 further illustrates that our method produces more natural textures and colors. Although such perceptually plausible outputs may deviate from the reference image and slightly affect referencebased metrics (the cat color in Figure 8), they better match real-world visual preference. While due to limited training data and the absence of large-scale generative priors, traditional methods struggle to generalize to realistic low-light degradations, as illustrated in Figure 7. Moreover, our model shows strong linear controllability for low-light enhancement on both paired and real-world benchmarks.

#### 4.3 Linear Control Evaluation

Following the evaluation protocol of KSlider [24], we report δsmooth to measure the smoothness of the continuous enhanment trajectory based on LPIPS feature distances. We also report CLIP-Dir to evaluate whether the enhancement trajectory consistently moves away from dark or underexposed semantics. We compare with several universal continuous image editing methods, including ConceptSlider [7], AttributeControl [1], KSlider [24], SliderEdit [55], and CLE Diffusion [53]. For a fair comparison, all methods are evaluated at the same four control strengths, s ∈ {0.25,0.50,0.75,1.00}, by mapping each method’s control variable linearly to this range. As shown in Table 3, our method achieves the highest CLIP-Dir score, demonstrating that its enhancement trajectory is more semantically aligned with the increasing enhancement strength and exhibits stronger linear controllability. More Qualitative Results is provide in the Appendix D.

###### 4.4 Ablation Study To exmain the effectiveness of our mehtod, we conduct more ablation studies:

- Table 2: Quantitative comparison on real-world and unpaired datasets: DICM [20], LIME [9], and RealIR-Bench [52].

Method

DICM [20] LIME [9] RealIR-Bench [52] NIQE↓ CLIPIQA↑ MANIQA↑ MUSIQ↑ NIQE↓ CLIPIQA↑ MANIQA↑ MUSIQ↑ NIQE↓ CLIPIQA↑ MANIQA↑ MUSIQ↑

Retinexformer [2] 3.962 0.377 0.291 54.27 4.300 0.394 0.367 59.41 4.200 0.286 0.277 52.98 CIDNet [49] 3.657 0.501 0.384 57.90 4.182 0.439 0.399 60.72 4.129 0.377 0.353 62.41 LLFormer [40] 3.943 0.435 0.274 55.03 4.392 0.382 0.297 57.39 3.866 0.236 0.250 49.14 DarkIR [6] 3.869 0.463 0.345 57.44 4.523 0.441 0.373 61.73 5.097 0.374 0.358 63.30 QuadPrior [42] 4.797 0.488 0.315 58.21 5.310 0.396 0.292 58.92 4.659 0.305 0.270 51.86 CLE Diffusion [53] 4.368 0.390 0.218 47.38 5.317 0.433 0.279 57.76 3.887 0.423 0.347 60.10

ControlLight (Ours) 3.522 0.698 0.505 68.22 3.638 0.576 0.526 67.68 3.748 0.550 0.491 67.96

- Table 3: Quantitative comparison of controllable editing performance across three datasets. We focus

on the trajectory smoothness (δsmooth ↓) and semantic directional consistency (CLIP-Dir ↑). To ensure fairness, all the methods are evaluated using aligned four-point control strengths.

Method

RealIR-Bench DICM LIME

δsmooth ↓ CLIP-Dir ↑ δsmooth ↓ CLIP-Dir ↑ δsmooth ↓ CLIP-Dir ↑

ConceptSlider [7] 0.9237 -0.0530 0.8700 0.3872 0.8589 0.0256 AttributeControl [1] 0.7262 0.3520 0.7928 0.3593 0.8176 0.3605 KSlider [24] 0.1956 0.0901 0.3570 0.4488 0.0485 0.0434 SliderEdit [55] 0.3840 -0.3125 0.4818 0.1768 0.3741 -0.1061 CLE Diffusion [53] 0.7503 -0.2624 0.7063 -0.2946 0.6643 0.1830

ControlLight (Ours) 0.2195 0.9138 0.2382 0.9012 0.1786 0.9159

Misalignment-Aware Weighted Flow Matching Loss. To assess the contribution of the proposed misalignment-aware weighted flow matching loss, we train a baseline model with the standard flow matching objective and evaluate both models on the low-light subset of RealIR-Bench. For consistency evaluation, we adopt LI-LPIPS [53] from CLE Diffusion, an edge-aware and colornormalized perceptual distance that is more stable than the original LPIPS [56] for measuring continuous-output consistency. We further report non-reference image quality assessment metrics to evaluate perceptual enhancement quality.

Table 4: Ablation study of LwFM on the RealIR-Bench. The best results are marked in bold.

Ablation LI-LPIPS ↓ NIQE ↓ MANIQA ↑ MUSIQ ↑ CLIPIQA ↑

LFM 0.2237 5.6242 0.3384 55.2252 0.5232 LwFM 0.2148 4.5367 0.4180 62.5262 0.6112

- Table 4 shows that LwFM effectively reduces structural inconsistency while also improving the perceptual quality of the low-light enhancement results.

Data Interpolation Methods. We conduct a no-reference quality assessment on the five-level interpolation results between Retinex-based interpolation and alpha blending interpolation. As shown in Figure 3, while both methods yield visually plausible results, they differ significantly in their illumination modeling. Specifically, Retinex-based interpolation more faithfully reflects real-world low-light degradation. To evaluate whether Retinex-based interpolation is superior for training, we analyze various no-reference metrics. Table 6 in Appendix indicate that Retinex-based interpolation provides richer degradation cues. It exhibits a more pronounced and physically reasonable quality gradient from I1 to I0, which is essential for the model to effectively learn the enhancement mapping.

- 5 Conclusions

We introduced a Retinex-inspired interpolation strategy and a high-quality dataset, Light100K, to facilitate real-world low-light enhancement. To tackle hallucinations and inconsistencies in outputs, we also developed the Misalignment-Aware Weighted Flow Matching Loss with Offline Edge-Mask Generation, which suppresses the effects of edge shifts during training. By fine-tuning the FLUX.2klein-9B model with LoRA using our proposed LwFM, ControlLight establishes new state-of-the-art performance. It outperforms existing enhancement and continuous editing methods, delivering superior consistency, controllability, and generalization in real-world scenarios.

### References

- [1] Stefan Andreas Baumann, Felix Krause, Michael Neumayr, Nick Stracke, Melvin Sevi, Vincent Tao Hu, and Björn Ommer. Continuous, subject-specific attribute control in t2i models by identifying semantic directions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13231– 13241, 2025.
- [2] Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, and Yulun Zhang. Retinexformer: One-stage retinex-based transformer for low-light image enhancement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12504–12513, 2023.
- [3] Yukang Cao, Chenyang Si, Jinghao Wang, and Ziwei Liu. Freemorph: Tuning-free generalized image morphing with diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18111–18120, 2025.
- [4] Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun. Learning to see in the dark. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3291–3300, 2018.
- [5] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [6] Daniel Feijoo, Juan C. Benito, Alvaro Garcia, and Marcos V. Conde. Darkir: Robust low-light image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 10879–10889, June 2025.
- [7] Rohit Gandikota, Joanna Materzy´nska, Tingrui Zhou, Antonio Torralba, and David Bau. Concept sliders: Lora adaptors for precise control in diffusion models. In European Conference on Computer Vision, pages 172–188. Springer, 2024.
- [8] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.
- [9] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. IEEE Transactions on Image Processing, 26(2):982–993, 2016.
- [10] Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. R2rnet: Low-light image enhancement via real-low to real-normal network. Journal of Visual Communication and Image Representation, 90:103712, 2023.
- [11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [12] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Liangliang Cao, and Shifeng Chen. Diffusion model-based image editing: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [13] Bernd Jähne. Digital image processing. Springer, 2005.
- [14] Kui Jiang, Zhongyuan Wang, Zheng Wang, Chen Chen, Peng Yi, Tao Lu, and Chia-Wen Lin. Degrade is upgrade: Learning degradation for low-light image enhancement. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 1078–1086, 2022.
- [15] Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. Enlightengan: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing, 30:2340–2349, 2021.
- [16] Zhangqi Jiang, Zheng Sun, Xianfang Zeng, Yufeng Yang, Xuanyang Zhang, Yongliang Wu, Wei Cheng, Gang Yu, Xu Yang, and Bihan Wen. Geditbench v2: A human-aligned benchmark for general image editing. arXiv preprint arXiv:2603.28547, 2026.
- [17] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5148–5157, 2021.
- [18] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- [19] Edwin H Land. The retinex theory of color vision. Scientific american, 237(6):108–129, 1977.
- [20] Chulwoo Lee, Chul Lee, and Chang-Su Kim. Contrast enhancement based on layered difference representation of 2d histograms. IEEE Transactions on Image Processing, 22(12):5372–5384, 2013.
- [21] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [22] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [23] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.
- [24] Rishubh Parihar, Or Patashnik, Daniil Ostashev, R Venkatesh Babu, Daniel Cohen-Or, and Kuan-Chieh Wang. Kontinuous kontext: Continuous strength control for instruction-based image editing. arXiv preprint arXiv:2510.08532, 2025.
- [25] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2085–2094, October 2021.
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [27] Yuqi Peng, Lingtao Zheng, Yufeng Yang, Yi Huang, Mingfu Yan, Jianzhuang Liu, and Shifeng Chen. Tara: Token-aware lora for composable personalization in diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 8385–8393, 2026.
- [28] Stephen M Pizer. Contrast-limited adaptive histogram equalization: Speed and effectiveness stephen m. pizer, r. eugene johnston, james p. ericksen, bonnie c. yankaskas, keith e. muller medical image display research group. In Proceedings of the first conference on visualization in biomedical computing, Atlanta, Georgia, volume 337, page 2, 1990.
- [29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PmLR, 2021.
- [30] Sudarshan Rajagopalan, Nithin Gopalakrishnan Nair, Jay N Paranjape, and Vishal M Patel. Gendeg: Diffusion-based degradation synthesis for generalizable all-in-one image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28144–28154, 2025.
- [31] Team Seedance, De Chen, Liyang Chen, Xin Chen, Ying Chen, Zhuo Chen, Zhuowei Chen, Feng Cheng, Tianheng Cheng, Yufeng Cheng, et al. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint arXiv:2604.14148, 2026.
- [32] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.
- [33] Prafull Sharma, Varun Jampani, Yuanzhen Li, Xuhui Jia, Dmitry Lagun, Fredo Durand, Bill Freeman, and Mark Matthews. Alchemist: Parametric control of material properties with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24130– 24141, 2024.
- [34] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [35] Meituan LongCat Team, Hanghang Ma, Haoxian Tan, Jiale Huang, Junqiang Wu, Jun-Yan He, Lishuai Gao, Songlin Xiao, Xiaoming Wei, Xiaoqi Ma, Xunliang Cai, Yayong Guan, and Jie Hu. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.
- [36] Qwen Team. Qwen3 technical report, 2025.
- [37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [38] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 2555–2563, 2023.
- [39] Peng Wang, Yichun Shi, Xiaochen Lian, Zhonghua Zhai, Xin Xia, Xuefeng Xiao, Weilin Huang, and Jianchao Yang. Seededit 3.0: Fast and high-quality generative image editing. arXiv preprint arXiv:2506.05083, 2025.
- [40] Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, and Tong Lu. Ultra-high-definition low-light image enhancement: A benchmark and transformer-based method. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 2654–2662, 2023.
- [41] Wenjing Wang, Huan Yang, Jianlong Fu, and Jiaying Liu. Zero-reference low-light enhancement via physical quadruple priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26057–26066, 2024.
- [42] Wenjing Wang, Huan Yang, Jianlong Fu, and Jiaying Liu. Zero-reference low-light enhancement via physical quadruple priors, 2024.
- [43] Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, and Alex Kot. Low-light image enhancement with normalizing flow. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 2604–2612, 2022.
- [44] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.
- [45] Jiangwei Weng, Zhiqiang Yan, Ying Tai, Jianjun Qian, Jian Yang, and Jun Li. Mamballie: Implicit retinex-aware low light enhancement with global-then-local state space. Advances in neural information processing systems, 37:27440–27462, 2024.
- [46] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025.
- [47] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, and Weisi Lin. Q-bench: A benchmark for general-purpose foundation models on low-level vision. In ICLR, 2024.
- [48] Dejia Xu, Hayk Poghosyan, Shant Navasardyan, Yifan Jiang, Humphrey Shi, and Zhangyang Wang. Recoro: Re gion-co ntrollable ro bust light enhancement with user-specified imprecise masks. In Proceedings of the 30th ACM International Conference on Multimedia, pages 1376–1386, 2022.
- [49] Qingsen Yan, Yixu Feng, Cheng Zhang, Guansong Pang, Kangbiao Shi, Peng Wu, Wei Dong, Jinqiu Sun, and Yanning Zhang. Hvi: A new color space for low-light image enhancement. In Proceedings of the computer vision and pattern recognition conference, pages 5678–5687, 2025.
- [50] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1191–1200, 2022.
- [51] Wenhan Yang, Wenjing Wang, Haofeng Huang, Shiqi Wang, and Jiaying Liu. Sparse gradient regularized deep retinex network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30:2072–2086, 2021.
- [52] Yufeng Yang, Xianfang Zeng, Zhangqi Jiang, Fukun Yin, Jianzhuang Liu, Wei Cheng, Shiyu Liu, Yuqi Peng, Gang YU, Shifeng Chen, et al. Realrestorer: Towards generalizable real-world image restoration with large-scale image editing models. arXiv preprint arXiv:2603.25502, 2026.
- [53] Yuyang Yin, Dejia Xu, Chuangchuang Tan, Ping Liu, Yao Zhao, and Yunchao Wei. Cle diffusion: Controllable light enhancement diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8145–8156, 2023.

- [54] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photo-realistic image restoration in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25669–25680, 2024.
- [55] Arman Zarei, Samyadeep Basu, Mobina Pournemat, Sayan Nag, Ryan Rossi, and Soheil Feizi. Slideredit: Continuous image editing with fine-grained instruction control. arXiv preprint arXiv:2511.09715, 2025.
- [56] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.
- [57] Yonghua Zhang, Jiawan Zhang, and Xiaojie Guo. Kindling the darkness: A practical low-light image enhancer. In Proceedings of the 27th ACM international conference on multimedia, pages 1632–1640, 2019.
- [58] Dewei Zhou, Zongxin Yang, and Yi Yang. Pyramid diffusion models for low-light image enhancement. arXiv preprint arXiv:2305.10028, 2023.
- [59] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE International Conference on Computer Vision, pages 2223–2232, 2017.

## Appendix

### A Continuous Pseudo-Paired Data Construction Details

During the construction of Light100K, we first collect high-resolution low-light images from opensource image websites, including Pexels and Pinterest, using low-light-related keywords. We then use the CLIP text encoder [29] to compute the cosine similarity between each image and darknessrelated prompts, such as “a dark photo”, “underexposed”, and “low illumination”, in order to filter images with relevant low-light semantics. Next, we employ Qwen3-VL-8B-Instruct [36] to assess the degradation level [47], ensuring that the retained images contain sufficient degradation cues for model learning [14, 30]. After semantic and degradation filtering, we obtain 27,529 high-quality low-light images, all with resolutions higher than 1024 × 1024.

We then use FLUX.2-klein-9B to generate restored normal-light references for the collected low-light images. To ensure pairwise structural consistency, we apply Sobel edge detection [13] to the low-light and restored images and filter out pairs with obvious edge shifts or structural misalignment. This process yields 17,809 high-consistency low-/normal-light image pairs.

Finally, we apply Retinex-inspired interpolation with enhancement strengths s ∈ {0.2,0.4,0.6,0.8} to construct intermediate pseudo targets. The resulting Light100K is a high-quality, real-world, continuous pseudo-paired dataset for controllable low-light enhancement.

### B Misalignment Analysis and Offline Edge-Mask Generation

During the construction of Light100K, subtle visual misalignment may still remain between lowlight inputs and their paired enhanced images, even after edge-consistency filtering. Although such misalignment is below the filtering threshold and is often visually negligible, the generative nature of the base model makes it problematic during training. Under the standard flow matching loss,

LFM = ∥vθ(zt,I0,s) − v∗∥22 ,

The model is encouraged to fit all target regions equally, which may introduce additional randomness when learning from slightly misaligned pseudo targets and lead to inconsistent outputs.

Our key insight is to preserve the structural edges of the input image while learning the desired illumination enhancement. To this end, we compute illumination-normalized log-luminance representations instead of directly comparing RGB values, so that images with different brightness levels can still share similar structural responses. We then apply a gradient operator to extract the main structural edges.

For each pair in Light100K, we compute a structural edge-difference map and use it to generate a spatial mask that guides flow matching with adaptive weights:

Ws(p) = clip(1 − αMs(p),wmin,1).

The resulting weight map is resized to the latent resolution as Ws and used in the weighted flow matching objective:

Ws(u)∥vθ(zt,I0,s)(u) − v∗(u)∥22 u Ws(u)

LwFM = u

.

In practice, we set d = 3 pixels, α = 0.8, and wmin = 0.2. To improve training efficiency, all weight maps are generated offline and cached before training.

### C Implementation Details

During training, we fine-tune only the DiT blocks with LoRA, while freezing the VAE and text encoders. LoRA layers are applied to both the single-stream and double-stream DiT blocks [26] with

a rank of 64. The bucket resolution is fixed at 1024 × 1024, and the global batch size is set to 16. Detailed hyperparameters are provided in Table 5. All experiments are conducted on 4 NVIDIA A6000 GPUS.

Table 5: Training hyperparameters for ControlLight fine-tuning. Hyperparameters

LoRA Setting Rank=64, Alpha=64 Trainable Parameters 317M Learning Rate 1 × 10−4 Optimizer AdamW 8-bit Precision BFloat16 (BF16) Scheduler Flow Matching Global Batch Size 16 Training Steps 3,000 Resolution 1024 × 1024

### D More Qualitative Results and Ablation Study Details

#### D.1 More Qualitative Results

Additional qualitative comparisons with low-light enhancement methods and general continuous image editing methods are presented in Figure 9 and Figure 10.

#### D.2 More Ablation Study Details

We report NIQE and MUSIQ to evaluate the image-quality trajectories of different interpolation methods across four intermediate enhancement strengths using 200 randomly sampled images from Light100K. Since the low-light image I0 is expected to have lower perceptual quality than the normal-light image I1, a desirable interpolation method should produce a smooth and monotonic quality transition between them. Table 6 shows that Retinex-based interpolation yields more natural image-quality trends and provides richer degradation cues for continuous enhancement learning.

Table 6: Ablation study on interpolation strategies for Light100K. Our Retinex-based interpolation preserves the intrinsic degradation at low enhancement levels, whereas Alpha Blending yields artificially high scores that deviate from the low-light distribution.

Metric Strategy I0 I0.2 I0.4 I0.6 I0.8 I1

Alpha Blending 4.588 3.931 3.561 3.356 3.461 3.695 Ours 4.588 4.171 3.649 3.315 3.419 3.695

NIQE ↓

Alpha Blending 55.936 62.620 66.289 70.047 68.469 70.019 Ours 55.936 58.780 60.889 67.626 67.716 70.019

MUSIQ ↑

LOL

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Input

Retinexformer

CIDNet

LLFormer

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

DarkIR

QuadPrior

CLE Diffusion

GT

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Ours s=0.25

Ours s=0.50

Ours s=0.75

Ours s=1.00

LSWR

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Input

Retinexformer

CIDNet

LLFormer

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

DarkIR

QuadPrior

CLE Diffusion

GT

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Ours s=0.25

Ours s=0.50

Ours s=0.75

Ours s=1.00

- Figure 9: Qualitative comparison with state-of-the-art low-light enhancement methods on the paired benchmarks LOL-v1 and LSWR.

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

CLE Diffusion

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

ConceptSlider

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

AttributeControl

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

KSlider

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

SliderEdit

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Ours

- Figure 10: Qualitative comparison with universal continuous image editing methods on RealIR-Bench for low-light enhancement. All methods are evaluated under the same four-point control strengths.

Input s = 0.25 s = 0.5 s = 0.75 s = 1.00

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

RealIR-BenchLWSRDICMLIMELOL

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

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

- Figure 11: Qualitative results of ControlLight on the reported benchmarks under four continuous enhancement strengths.

