# arXiv:2512.04504v1[cs.CV]4Dec2025

## UltraImage: Rethinking Resolution Extrapolation in Image Diffusion Transformers

Min Zhao1,2, Bokai Yan3, Xue Yang1,2, Hongzhou Zhu1,2, Jintao Zhang1,2, Shilong Liu4, Chongxuan Li3, Jun Zhu1,2 1Dept. of Comp. Sci. & Tech., BNRist Center, Tsinghua University. 2ShengShu. 3Gaoling School of Artificial Intelligence, Renmin University of China. 4 Princeton University. gracezhao1997@gmail.com, dcszj@tsinghua.edu.cn

#### Abstract

Recent image diffusion transformers achieve high-fidelity generation, but struggle to generate images beyond these scales, suffering from content repetition and quality degradation. In this work, we present UltraImage, a principled framework that addresses both issues. Through frequencywise analysis of positional embeddings, we identify that repetition arises from the periodicity of the dominant frequency, whose period aligns with the training resolution. We introduce a recursive dominant frequency correction to constrain it within a single period after extrapolation. Furthermore, we find that quality degradation stems from diluted attention and thus propose entropy-guided adaptive attention concentration, which assigns higher focus factors to sharpen local attention for fine detail and lower ones to global attention patterns to preserve structural consistency. Experiments show that UltraImage consistently outperforms prior methods on Qwen-Image and Flux (around 4K) across three generation scenarios, reducing repetition and improving visual fidelity. Moreover, UltraImage can generate images up to 6K×6K without low-resolution guidance from a training resolution of 1328p, demonstrating its extreme extrapolation capability. Project page is available at https://thu-ml.github.io/ultraimage.github.io/.

#### 1. Introduction

Building upon the expressive power of image diffusion transformers [1, 27], recent advances in text-to-image generation [6, 29, 38, 40, 41] have enabled high-fidelity image synthesis. Despite these advances, these models still struggle to generate ultra-resolution images beyond their trained spatial scale [22, 39, 48], a task we refer to as resolution extrapolation. This capability is critical for practical applications such as large-format printing, high-resolution content creation, and detailed visual simulations, where gener-

ating images at scales larger than the training resolution is required.

To investigate the challenges of resolution extrapolation in image diffusion transformers, we first conduct experiments on two representative models, Flux [22] and QwenImage [39]. We identify two typical failure modes: content repetition, where visual elements repeat periodically across the image, and quality degradation, where fine details are blurred. Existing positional extrapolation methods either fail to mitigate repetition or reduce repetition at the cost of introducing over-smoothed textures. We argue that these failures stem from an incomplete understanding of frequency components in position encoding, leading to flawed modifications or targeting incorrect frequencies.

In this paper, we establish a principled guideline for which frequency components should be modified and how to modify them. Through systematic analysis, we find that the high-frequency component, with short period, primarily governs local textures, and interpolating it leads to blurring. The low-frequency component has minimal impact. In contrast, the mid-band frequency—whose period is comparable to the training resolution—controls global structure; we refer to it as dominant frequency. By ensuring that the dominant frequency remain within a single period after extrapolation, repetition can be effectively mitigated. Furthermore, due to dynamic-resolution training, some models may contain multiple dominant frequencies near the training resolution. To address this, we apply a recursive dominant frequency correction procedure to the dominant frequency until repetition is resolved.

To address the second challenge, quality degradation, our analysis finds that the observed blurring stems from flattened attention distributions that dilute focus. Applying a single global focus factor sharpens local details but disrupts long-range dependencies, creating a trade-off between fine textures and structural consistency. This tradeoff arises from the functional specialization of attention pat-

[Figure 1]

Figure 1. Generated results of UltraImage. Starting from the base Qwen-Image model trained at 1328p resolution, UltraImage can generate high-quality images up to 6K × 6K without any low-resolution guidance, demonstrating its extreme extrapolation capability. All prompts used in this paper are provided in the Appendix.

terns: global patterns (high-entropy) require lower focus factor to preserve structural coherence, while local patterns (low-entropy) benefit from higher focus factor to enhance fine details. To resolve this, we propose an entropy-guided adaptive concentration strategy, which assigns a distinct focus factor to each attention pattern based on its entropy. Furthermore, to handle high-resolution images, we implement a custom Triton kernel that computes the softmax in a block-wise, online fashion, avoiding out-of-memory issues while dynamically applying the pattern-specific focus.

Extensive experiments on Flux [22] and QwenImage [39] demonstrate the effectiveness of UltraImage. Our method consistently outperforms state-of-the-art baselines [4, 5, 8, 11, 28] across almost all metrics including FID, KID, and CLIP score around 4K, in three generation scenarios—direct resolution extrapolation, guided resolution extrapolation, and guided view extrapolation. Qualitative results further confirm that UltraImage reduces repe-

tition and improves visual fidelity, validating its practicality for generating ultra-resolution outputs beyond the training scale. Moreover, UltraImage can generate images up to 6K×6K from a training resolution of 1328p, demonstrating its extreme extrapolation capability.

#### 2. Background

Rotary position embedding (RoPE) in image diffusion transformers. RoPE [36] is a widely adopted method in transformers for encoding relative positional information. RoPE works by rotating pairs of feature dimensions according to their positions, effectively introducing structured, position-dependent modulation. For a one-dimensional sequence, given an input vector x ∈ Rd at position p, the

embedding is computed as

cos(pθj) −sin(pθj) sin(pθj) cos(pθj)

x2j x2j+1

fRoPE(x,p,θ)j =

,

(1) where j = 1,...,d′/2, θ ∈ Rd

′/2 defines the frequency for each dimension, and b controls the base of the exponential schedule θj = b−2(j−1)/d

′

. For images with height and width, existing works use two independent RoPE embeddings: one along the height axis with frequencies θih, and one along the width axis with frequencies θiw [22, 39]. Each axis applies the standard single-axis RoPE, and the resulting embeddings are concatenated to obtain the final positional encoding.

Attention mechanism. Given an input image with height H and width W, let Q,K ∈ RHW×D and V ∈ RHW×D

′

denote the queries, keys, and values after RoPE. The attention logits, scores, and output are then computed as

S = QK⊤, P = softmax

S √

D

###### , O = PV , (2)

where S ∈ RHW×HW is the attention logits, P stores the pairwise similarities between queries and keys, and O ∈ RHW×D

′

is the final attended output.

Length extrapolation in image generation. Extending sequence lengths in transformers is challenging due to positional encoding limitations. Position Interpolation (PI) [8] rescales RoPE frequencies to match the target sequence length, θ′ = θ/s, where s = L′/L and L, L′ is the sequence length for training and inference, respectively. NTK-Aware Scaled RoPE (NTK) [4, 48] adjusts the base frequency b for all dimensions to extrapolate high-frequency components while interpolate lowfrequency ones:

′

′/(d′−2),j = 1,...,d′/2.

θjNTK = (λb)−2(j−1)/d

,λ = sd

(3) YaRN [28] further refines this via frequency-wise grouping and gradual interpolation–extrapolation. For images, these adjustments can be applied independently along height and width, termed Vision NTK and Vision YaRN [25].

In early U-Net-based image generation, extending the spatial resolution often leads to object repetition due to the limited convolutional receptive field [14]. Methods such as ScaleCrafter [14] or DemoFusion [10] incorporating global perceptive layers have been proposed to enlarge the effective receptive field and mitigate repetition. Further analysis in FouriScale attribute repetitive artifacts to frequency misalignment across scales [18], highlighting challenges in naive length extrapolation. Recent transformer-based approaches leverage low-resolution cues for high-resolution synthesis, e.g., I-Max [11] and HiFlow [5] project or guide

high-resolution flows using low-resolution information, enabling training-free or guided super-resolution. These methods focus on external guidance, whereas the intrinsic extrapolation ability of image diffusion transformers remains underexplored. Recent work on video length extrapolation [46] closely aligns with our finding in image generation. See more work in the related work section in supplementary materials.

#### 3. Problem Setting and Challenges

10242 32002 40962

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Figure 2. Challenges of resolution extrapolation in image diffusion transformers. Top row: Flux (training resolution with 20482). Bottom row: Qwen-Image (training resolution 1328p). Both models exhibit typical failure modes: content repetition and quality degradation at higher resolutions.

In this paper, we address the task of resolution extrapolation for image diffusion transformers. Given a model trained on a resolution h × w, the goal is to enable it to generate high-quality and spatially coherent images beyond its trained spatial scale, at larger resolutions H × W, without any re-training. We define the extrapolation ratios as sh = H/h and sw = W/w, where sh ≥ 1,sw ≥ 1.

When directly applying diffusion transformers such as Flux [22] and Qwen-Image [39] beyond their training resolution, we identify two typical failure modes (see Fig. 2). The first is content repetition, where visual content repeats periodically across regions. The second is quality degradation, where texture fidelity declines and images become overly blurred, The second is quality degradation, where texture fidelity declines and images become overly blurred, especially under large extrapolation. In the following sections, we analyze and address them separately.

#### 4. Understanding and Solving Repetition from Positional Encoding

In this section, we first focus on the problem of spatial repetition, analyzing its origin from positional encoding in

Role of each frequency in RoPE Validation of the repetition cause

high frequency (T0h ≪ h)

mid-band frequency (T8h ≈ h)

low frequency (T20h ≫ h)

reference

H > T8h H = T8h H < T8h

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

(a) (b) (c) (d) (e) (f) (g)

- Figure 3. Cause of content repetition. Left: (a) Height extrapolation baselines. (b) High-frequency interpolation blurs local textures. (c) The mid-band dominant frequency, whose period aligns with the training height h, governs global structure and introduces repetition. (d) Low-frequency components have minimal effect. Right: Validation: repetition appears when the extrapolated height H (e) exceeds the dominant period Tkh, and disappears when H ≤ Tkh (f,g).

- Sec. 4.1 and then proposing a frequency-based solution in
- Sec. 4.2 to solve it.

- 4.1.FrequencyAnalysisofRoPEinImageDiffusion Transformers

NTK PI Ours

[Figure 15]

[Figure 16]

[Figure 17]

- Figure 4. Failure modes of existing positional extrapolation. NTK suffers from repetition during extrapolation, while PI reduces it but causes over-smoothed textures. In contrast, our method mitigates repetition without sacrificing image fidelity by correctly identifying and adjusting the relevant frequency components.

leads to suboptimal results. In this paper, we aim to establish a principled guideline by answering two key questions:

- 1. Which frequencies in RoPE are primarily responsible for governing image structure and causing repetition?
- 2. How should these specific frequencies be adjusted to mitigate repetition while preserving image fidelity?

The role of each frequency component. To answer the above questions, we analyze different RoPE frequency components by interpolating one component at a time along a spatial dimension (e.g., height), where θih is scaled by 1/sh. As shown in Fig. 3, different frequency components control visual features at different scales. The High-frequency component primarily affects local textures—interpolating it causes blurring but preserves overall layout (Fig. 3b). The low-frequency components have negligible impact when modified (Fig. 3d). In contrast, the mid-band frequency governs global structure, and adjusting it effectively eliminates content repetition (Fig. 3c). These results indicate that spatial repetition mainly arises from the mid-band frequency responsible for global structure.

Inspired by advances in language models, prior works have attempted to mitigate spatial repetition by modifying positional encodings [11, 25, 48]. For example, NTK alleviates repetition via frequency scaling, while PI interpolates all RoPE components for structural preservation. However, as shown in Fig. 4, NTK still shows noticeable repetition, and PI introduces over-smoothed textures similar to those in Sec. 3. We argue that these issues stem from targeting incorrect components or modifying them improperly, which

We posit that these phenomena originate from a unified mechanism: the intrinsic periodicity of RoPE. As introduced in Sec. 2, RoPE encodes spatial positions using periodic functions cos(·) and sin(·), where each frequency component i corresponds to an intrinsic period

2π θih

Tih =

. (4)

###### Static attention scores Reference HF interpolation Refocused HF

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

(a) (b) (c) (d)

- Figure 5. Cause of quality degradation. Previous interpolation of a single high frequency (HF) in Fig. 3b leads to similar quality loss during extrapolation. (a) Comparing attention maps (b) pre- and (c) post-interpolation, the distribution becomes significantly flattened. Sharpening the attention in (d) restores the lost details, confirming that the degradation originates from reduced attention concentration.

When the relative positional distance exceeds this period, the encoding becomes ambiguous:

cos(θih(p+Tih)) = cos(θihp), sin(θih(p+Tih)) = sin(θihp).

(5) Due to this periodicity, positions p and p + Tih share identical encodings, leading the model to treat them as equivalent and thus produce repeated content. This interpretation aligns with our observations: high-frequency components (with short periods, Tih ≪ h) control local textures; low-frequency components (Tih ≫ h) remain unaffected; and mid-band frequencies (Tih ≈ h) govern global structure. Consequently, the structure-level repetition reported in Sec. 3 emerges once the extrapolated height H exceeds the intrinsic period of such a mid-band frequency.

Formally, we define the dominant frequency as the one whose period is closest to the training height h:

|Tih − h|. (6)

kh = arg min

i

To avoid repetition, the dominant frequency must remain within a single period during extrapolation, satisfying the following non-repetition condition:

Tk′h ≥ H, θk′h ≤

2π H

. (7)

As illustrated in Fig. 3e, settings where H > Tkh exhibit repetition, while H ≤ Tkh avoid it (Fig. 3f, g). Under this perspective, NTK-based scaling fails to satisfy the non-repetition condition in Eq. (7), thereby causing structure repetition; conversely, PI effectively amplifies highfrequency components, leading instead to blurred images.

##### 4.2. Recursive Dominant Frequency Correction

Building on the above analysis, we directly modify the dominant frequencies for height and width as

2π H

, θk′w =

θk′h =

2π W

. (8)

Empirically, this modification largely mitigates structurelevel repetition. However, in some models (e.g., QwenImage), we observe a residual, weaker repetition with a spatial period close to Tk. We hypothesize that this arises because dynamic-resolution training around Tk introduces multiple frequency components that can align with the training length, yielding several competing dominant frequencies.

To address this, we propose a Recursive Dominant Frequency Correction (RDFC), which iteratively adjusts the dominant frequency until the remaining repetition is eliminated and the dominant component is constrained within a single period after extrapolation. The complete procedure is summarized in Algorithm 1.

#### 5. Understanding and Solving Quality Degradation in Attention

With the content repetition artifacts resolved, we now address the second key challenge identified in Sec. 3: quality degradation.

Algorithm 1 Recursive Dominant Frequency Correction Require: Training resolution (h,w), frequencies {θih,θiw}

in RoPE, target resolution (H,W)

- 1: Initialize observed repetition period Nh ← h, Nw ← w
- 2: while Nh < H or Nw < W do ▷ Repetition persists
- 3: for i = 1 to d

′

2 do

- 4: Tih = 2θπ

- h
- i

, Tiw = θ2wπ

i

▷ Compute period

- 5: end for
- 6: kh = arg mini |Tih − Nh|, kw = arg mini |Tiw − Nw| ▷ Identify dominant frequency
- 7: θkh

h

← 2Hπ, θkw

w

← 2Wπ ▷ Correct dominant frequency

- 8: Detect new repetition period Nh,Nw
- 9: end while

###### baseline λ = 1.1 λ = 1.2

[Figure 22]

[Figure 23]

[Figure 24]

- Figure 6. Trade-off in attention concentration. Increasing the focus factor λ (e.g., 1.1) during extrapolation sharpens attention and enhances details, but further increase (e.g., 1.2) leads to spatial inconsistencies despite higher sharpness.

##### 5.1. Analyzing Quality Degradation in Attention

Comparing the one-dimensional static attention scores before and after interpolation (Fig. 5) reveals a noticeably flatter distribution. To quantify this effect, we introduce a global focus factor λ (an inverse temperature) [19] applied to the attention logits:

Sij′ = λ · Sij, (9)

where Sij denotes the original attention score and λ > 1. As shown in Fig. 5, sharpening with a larger λ restores fine details, confirming that the degradation stems from reduced attention concentration.

We hypothesize that a similar flattening occurs during resolution extrapolation, where the expanded token field dilutes attention weights. Indeed, applying a focus factor λ improves detail quality (Fig. 6); however, excessive sharpening introduces new repetition artifacts and spatial inconsistency. This behavior can be explained by the attention mechanism in Eq. (2): a flat distribution averages diverse features, producing blur, while an overly sharp one restricts long-range dependencies, breaking spatial coherence.

Visualizing multi-head attention further reveals distinct functional roles: some heads capture global structure with broad attention, while others focus on local textures. A single global λ therefore affects them unevenly—values that benefit local attention pattern often over-concentrate global ones (Fig. 7). To address this, we require an adaptive focus factor λα for each attention head α, assigning smaller values to global attention patterns to preserve consistency and larger ones to local attention patterns to enhance details.

##### 5.2. Entropy-guided Adaptive Attention Concentration

The design principle introduces two challenges: (1) quantifying the dispersion of each attention pattern, and (2) mapping this metric to an appropriate focus factor λα. We measure dispersion using the Shannon entropy Hα, which reflects how concentrated an attention distribution is—high

type generated image attention pattern

[Figure 25]

[Figure 26]

local

[Figure 27]

[Figure 28]

global

Figure 7. Effects of the focus factor on different attention patterns. A single factor λ may under-sharpen local attention (top) but over-suppress global attention (bottom), disrupting structural coherence and motivating adaptive focus for each pattern.

entropy indicates a global pattern, while low entropy corresponds to a local one. Specifically, for the α-th attention pattern:

HW

1 HW

Hα = −

i=1

HW

Pi,jα log Pi,jα , (10)

j=1

where Pα ∈ RHW×HW is the attention map and HW is the number of query tokens.

To assign focus factors, we construct a function that increases monotonically as entropy decreases, giving stronger sharpening to more concentrated patterns:

p

Hmax − Hα Hmax − Hmin

, (11)

λα = λmin + (λmax − λmin)

where λmin and λmax are the lower and upper bounds of scaling, and the exponent p controls the mapping curvature—smaller p yields more aggressive sharpening of lowentropy heads, while larger p provides smoother adjustment. See more details in Appendix.

Empirically, we find that the functional role of each attention head remains consistent across prompts and timesteps, allowing us to determine pattern types and corresponding focus factors from a single diffusion step (requiring only ∼ 2% of total inference cost for 50 steps).

Triton-based implementation. The proposed method introduces two practical challenges: (1) computing Eq. (10) requires materializing the full attention map, which is memory-prohibitive—e.g., a 4096×4096 token input yields a 40K × 40K attention matrix, consuming over 80GB in

Algorithm 2 Entropy-Guided Adaptive Attention Concentration Require: Bounds λmin, λmax, exponent p. Matrices Q, K, V ∈

RN×d, block size bq, bkv. Stage 1: Entropy computation for the first denoising step

- 1: Initialize Hmin ← +∞, Hmax ← −∞
- 2: for each block-wise attention pattern α do
- 3: Hα = Eq. 10 (softmax(QK⊤/

√

d)) ;

- 4: Hmin ← min(Hmin, Hα);
- 5: Hmax ← max(Hmax, Hα);
- 6: Cache Hα ;
- 7: end for
- 8: for each block-wise attention pattern α do
- 9: λα = Eq. 11 (cached Hα, α, Hmax, Hmin )
- 10: Cache λα for later use
- 11: end for Stage 2: Head-wise adaptive concentration
- 12: Divide Q into Tm = N/bq blocks {Qi}, and divide K, V into Tn = N/bkv blocks {Ki} and {Vi};
- 13: for i in [1, Tm] do
- 14: for j in [1, Tn] do
- 15: Sij = QiKjT ;
- 16: mji = max(mji−1, rowmax(Sij)) ;
- 17: Pij = exp(Sij − mji) ;
- 18: for each attention pattern α during generation do
- 19: Retrieve cached λα ;
- 20: Scale attention map: Pij = Softmax(λα × Pij) ;
- 21: end for
- 22: lij = em

j−1

i −mji lij−1 + rowsum( Pij) ;

- 23: Oij = diag(em

j−1

i −mji )Oij−1 + PijVj ;

- 24: end for
- 25: Oi = diag(liTn)−1OiTn ;
- 26: end for
- 27: return O = {Oi};

bf16; (2) standard attention kernels do not support patterndependent modification of attention logits.

To address these two challenges, we develop a custom online softmax kernel in Triton. For the first challenge, we adopt a block-wise streaming strategy (e.g., block size bq = 128,bkv = 128) that computes attention scores in-

###### baseline +RDFC full version

[Figure 29]

[Figure 30]

[Figure 31]

- Figure 8. Ablation studies. Our RDFC effectively mitigates image repetition without compromising quality, while adaptive attention concentration further enhances visual fidelity. See the Appendix for quantitative ablation studies.

crementally without materializing the full attention matrix, ensuring memory efficiency. For the second challenge, our Triton kernel supports head-wise dynamic scaling, where each attention head retrieves its pre-computed focus factor λα during computation. This implementation fully realizes adaptive concentration with negligible overhead. The full algorithm is shown in Algorithm 2.

#### 6. Experiments

##### 6.1. Setup

Experiment Setup. We evaluate UltraImage under three generation scenarios. (1) Direct resolution extrapolation directly generates ultra-resolution images without any guidance, testing the model’s ability to extrapolate beyond the training resolution (4096 × 4096 for Qwen, 3600 × 3600 for Flux). (2) Guided resolution extrapolation follows prior work [5, 11]: we first generate images at the training resolution (1024×1024), upsample them to the target resolution with 3600×3600 on Flux), and use the upsampled image as low-resolution guidance to generate the final high-resolution output. (3) Guided view extrapolation generates a 1024×1024 image at the training resolution and places it at the center of a target resolution of 3600×3600 on Flux, following the SDEdit [26], allowing the model to expand content around the central low-resolution image. See more experimental details in the Appendix. The dominant frequency k = 9 for Flux and k = 8,9 for Qwen-Image. λmin = 1.0,λmax = 1.3,p = 2 for all models.

Evaluation. To evaluate image generation, we randomly select 1K high-quality captions from LAION-5B covering diverse scenarios, following the general setup of prior work [14]. We conduct evaluations on Qwen-Image [39] and Flux models. Prompt-following ability is measured using the CLIP score [31], while overall image quality is assessed via Frechet Inception Distance (FID) [15] and Kernel Inception Distance (KID) [3]. The generated images are compared against 10K high-quality real images from LAION-5B [34].

##### 6.2. Results

Main Results. Quantitative and qualitative comparisons are presented in Tab. 1 and Fig. 9. Our method consistently outperforms baselines across almost all metrics and tasks. In direct resolution extrapolation, UltraImage achieves substantial FID improvements of 113.41 over NTK and 41.31 over PI on Flux, demonstrating notable gains in visual fidelity and a significant reduction of repetition artifacts. In guided resolution extrapolation, where low-resolution images are upsampled and used as guidance, our approach effectively improves image quality, surpassing prior methods in FID, KID, and CLIP score. In guided view extrapola-

Table 1. Quantitative comparison across three generation scenarios. extra. denote extrapolation. All methods are evaluated at 4096 × 4096 on Qwen-Image (training resolution 1368p) and 3600 × 3600 on Flux (training resolution ranging from 2562 to 20482).

Direct extra. on Qwen-Image FID↓ KID↓ CLIP Score↑ FID↓ KID↓ CLIP Score↑

Direct extra. on Flux

Method

Method

PE 206.2 0.1133 0.2280 PE 86.93 0.0162 0.3257 PI [8] 124.5 0.0391 0.2789 PI [8] 94.03 0.0217 0.3310 NTK [4] 196.6 0.1055 0.2345 NTK [4] 86.94 0.0144 0.3246 YaRN [28] 186.2 0.0947 0.2391 YaRN [28] 96.47 0.0174 0.3304 Entropy [19] 117.3 0.0334 0.2708 Entropy [19] 89.92 0.0164 0.3399 Ours 83.19 0.0114 0.3083 Ours 78.15 0.0086 0.3337

Guided view extra. on Flux FID↓ KID↓ CLIP Score↑ FID↓ KID↓ CLIP Score↑

Guided resolution extra. on Flux

Method

Method

HiFlow [5] 73.13 0.0085 0.3375 NTK [4] 118.6 0.0399 0.2291 I-Max [11] 72.00 0.0078 0.3392 YaRN [28] 111.7 0.0352 0.2339 Ours 68.98 0.0076 0.3417 Ours 104.7 0.0276 0.2842

[Figure 32]

- Figure 9. Qualitative comparison across three generation scenarios. Our method outperforms baselines across all tasks by delivering high visual quality while mitigating content repetition.

tion, which expands content around a central low-resolution image, UltraImage preserves structural consistency while recovering fine details, outperforming baselines across all metrics. Qualitative results in Fig. 9 further support these findings. Overall, these results confirm that UltraImage can generate ultra-resolution images that are both visually coherent and rich in detail, demonstrating its versatility across diverse resolution extrapolation scenarios.

Ablation Studies. As shown in Fig. 8, compared with the direct extrapolation baseline, our Recursive Dominant Fre-

quency Correction strategy effectively mitigates image repetition without sacrificing quality. Building upon this, the Entropy-Guided Adaptive Attention Concentration further enhances visual quality. Additional qualitative examples are presented in Fig. 1. More ablation of the hyperparameters λmin,λmax,p is provided in the Appendix.

#### 7. Conclusion

We propose UltraImage, a principled framework for generating ultra-resolution images beyond the training scale. It tackles two key challenges in extrapolation: content repeti-

tion and quality degradation. UltraImage introduces a recursive dominant-frequency correction to eliminate repetition and an entropy-guided adaptive attention concentration to restore sharpness lost during extrapolation. Together, these components preserve global coherence and fine details, enabling high-fidelity generation at extreme resolutions.

#### References

- [1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 1
- [2] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023. 2
- [3] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 7
- [4] bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023. 2, 3, 8
- [5] Jiazi Bu, Pengyang Ling, Yujie Zhou, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Hiflow: Training-free high-resolution image generation with flow-aligned guidance. arXiv preprint arXiv:2504.06232, 2025. 2, 3, 7, 8
- [6] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 1, 2
- [7] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024. 2
- [8] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023. 2, 3, 8
- [9] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [10] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising highresolution image generation with no $$$. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6159–6168, 2024. 3, 2
- [11] Ruoyi Du, Dongyang Liu, Le Zhuo, Qin Qi, Hongsheng Li, Zhanyu Ma, and Peng Gao. I-max: Maximize the resolution potential of pre-trained rectified flow transformers with projected flow. 2024. 2, 3, 4, 7, 8
- [12] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang, et al. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers. arXiv preprint arXiv:2405.05945, 2024. 2
- [13] Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adapta-

- tion. In European Conference on Computer Vision, pages 39–55. Springer, 2024. 2
- [14] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2023. 3, 7, 2
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 7
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [17] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 2
- [18] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. In European conference on computer vision, pages 196–212. Springer, 2024. 3, 2
- [19] Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. Trainingfree diffusion model adaptation for variable-sized text-toimage synthesis. Advances in Neural Information Processing Systems, 36:70847–70860, 2023. 6, 8
- [20] Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. Trainingfree diffusion model adaptation for variable-sized text-toimage synthesis. Advances in Neural Information Processing Systems, 36, 2024. 2
- [21] Younghyun Kim, Geunmin Hwang, Junyu Zhang, and Eunbyung Park. Diffusehigh: Training-free progressive highresolution image synthesis through structure guidance. arXiv preprint arXiv:2406.18459, 2024. 2
- [22] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 1, 2, 3
- [23] Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems, 36:50648–50660, 2023. 2
- [24] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024. 2
- [25] Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai. Fit: Flexible vision transformer for diffusion model. International Conference on Machine Learning., 2024. 3, 4
- [26] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021. 7
- [27] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 2

- [28] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. International Conference on Learning Representations., 2023. 2, 3, 8
- [29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 2
- [30] Haonan Qiu, Shiwei Zhang, Yujie Wei, Ruihang Chu, Hangjie Yuan, Xiang Wang, Yingya Zhang, and Ziwei Liu. Freescale: Unleashing the resolution of diffusion models via tuning-free scale fusion. arXiv preprint arXiv:2412.09626,

2024. 2

- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 7
- [32] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024. 2
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2
- [34] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 7
- [35] Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering highresolution image generation via structural and fine-grained guidance. arXiv preprint arXiv:2406.16476, 2024. 2
- [36] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021. 2
- [37] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 2
- [38] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 1
- [39] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 1, 2, 3, 7
- [40] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-

- Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM computing surveys, 56(4): 1–39, 2023. 1
- [41] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 1
- [42] Ruonan Yu, Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Ultra-resolution adaptation with ease. 2025. 2
- [43] Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models. arXiv preprint arXiv:2503.18352, 2025. 2
- [44] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Zhenyuan Chen, Yao Tang, Yuhao Chen, Wengang Cao, and Jiajun Liang. Hidiffusion: Unlocking high-resolution creativity and efficiency in low-resolution trained diffusion models. arXiv preprint arXiv:2311.17528, 2023. 2
- [45] Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609–3623, 2022. 2
- [46] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025. 3
- [47] Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, and Hang Xu. Any-sizediffusion: Toward efficient text-driven synthesis for any-size hd images. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7571–7578, 2024. 2
- [48] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems., 2024. 1, 3, 4, 2

## UltraImage: Rethinking Resolution Extrapolation in Image Diffusion Transformers

### Supplementary Material

#### 8. Related Work

##### 8.1. Text-to-Image Generation

Diffusion-based generative models have become the dominant paradigm for text-to-image synthesis due to their strong visual fidelity and semantic controllability. Early diffusion formulations such as DDPM [16] and Guided Diffusion [9] demonstrated that iterative denoising can produce high-quality images. The introduction of latent-space diffusion in LDM [33] greatly improved efficiency by operating in a compressed representation space, leading to widely adopted systems including Stable Diffusion and SDXL [29].

More recent approaches explore transformer-based diffusion architectures, which enhance global reasoning and scale more effectively than U-Net backbones. DiT [27], PixArt-α/Σ [6, 7], and the Lumina series [12, 48] exemplify this trend. Rectified-flow models such as Flux [22] further refine the generative process through stable ODEbased formulations. In addition to architectural advancements, a parallel line of work focuses on improving controllability through energy-based guidance. Energy-Guided SDEs (EGSDE) [45] introduce an elegant framework for unpaired image-to-image translation by shaping the diffusion trajectory with learned energy functions.

In this work, we adopt Flux.1.0-dev [22] as our text-toimage backbone due to its strong image quality and stable large-scale behavior.

##### 8.2. High-Resolution Image Generation

Synthesizing images at resolutions far beyond the training scale remains a longstanding challenge for diffusion models, primarily due to the limited availability of highresolution datasets and the significant computational demands of modeling large spatial grids. Existing research can be broadly divided into two categories.

Training or fine-tuning with high-resolution data. A number of approaches [13, 17, 24, 32, 37, 42, 43, 47] directly train or adapt diffusion models on higher-resolution datasets. While these methods can capture fine details, they require substantial computational resources and are limited by the scarcity of high-quality high-resolution data.

Training-free resolution expansion. To bypass costly retraining, many works modify the inference process to scale

resolution while keeping model weights unchanged. Patchbased fusion strategies such as MultiDiffusion [2] and SyncDiffusion [23] combine overlapping denoising paths to enlarge the output canvas. DemoFusion [10] extends this idea with global layout perception. Other methods reshape or reinterpret intermediate features to align with larger spatial grids, as demonstrated in HiDiffusion [44] and I-Max [11]. Receptive-field manipulation with dilated convolutions (ScaleCrafter [14]), frequency-domain alignment (FouriScale [18]), wavelet-based structural guidance [21], and training-free detail enhancement [20, 30, 35] offer additional routes.

Table 2. Quantitative ablation studies.

Method FID↓ KID↓ CLIP↑ baseline 206.2 0.1133 0.2280 +RDFC 107.81 0.0257 0.2829 full version 83.19 0.0114 0.3083

#### 9. Experiment Setup

For the guided resolution extrapolation setting, building upon I-max, we first generate a 1024 × 1024 image and then upsample it to the target high resolution, using the upsampled result as guidance during high-resolution generation. For the guided view extrapolation setting, we first generate a 1024×1024 low-resolution image. During highresolution generation, we replace the central region of the high-resolution x0-prediction with noisy versions of the low-resolution image obtained via forward diffusion, ensuring consistent view alignment across scales.

#### 10. More Results

As shown in Fig. 11, we provide additional UltraImage samples generated without any low-resolution guidance. Despite being trained only at 1328p, UltraImage can produce much higher-resolution images on Qwen-Image, demonstrating its strong capability to extrapolate far beyond the training resolution.

#### 11. Ablation Studies

Ablation studies for the hyperparameters λmin,λmax and p. As shown in Fig. 10, different exponent values p correspond to different degrees of concentration. A smaller

[Figure 33]

###### Figure 10. Visualization of λ with different p in Eq. (11)., with

- λmin = 1.0 and λmax = 1.3. Smaller p produces stronger and broader attention focusing, while larger p results in weaker concentration.

p applies stronger focusing to a larger portion of the attention patterns, leading to more concentrated attention. As shown in Fig. 12, when p is too small (e.g., p = 0.2), the attention becomes overly concentrated, leading to oversharpening and structural artifacts; when p is too large (e.g., p = 5.0), the attention is insufficiently focused and the visual quality degrades. An intermediate value yields the best trade-off. Similarly, Fig. 13 shows that a small λmax = 1.1 does not provide enough concentration, leading to suboptimal visual quality, while a large λmax = 1.6 results in oversharpening. An intermediate value yields the best trade-off. For λmin (see Fig. 14), even a relatively small value such as

- λmin = 1.1 introduces structural inconsistency, indicating that not all attention patterns should be concentrated. This further supports our design of selectively applying stronger focusing only to local patterns.

Ablation studies for entropy-guided adaptive attention concentration. As shown in Fig. 15, using a single global concentration factor leads to a clear trade-off: a smaller focus factor results in poor visual quality, whereas a larger one introduces structural inconsistencies. Moreover, even a small focus factor (e.g., λ = 1.1) still causes noticeable structural mismatches. In contrast, our entropy-guided adaptive attention concentration strategy applies a mild focus factor to the global attention pattern to preserve structural consistency, while assigning a stronger concentration factor to the local attention pattern to enhance visual quality, thereby simultaneously achieving high-quality generation and strong structural consistency.

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

###### Figure 11. Results of UltraImage at 4096 × 4096, where training resolution is 1328p.

###### p = 5.0 p = 2.0 p = 0.2

[Figure 40]

[Figure 41]

[Figure 42]

- Figure 12. Ablation on p. Small p (e.g., 0.2) causes overly concentrated attention and over-sharpening, while large p (e.g., 5.0) leads to insufficient focus and degraded visual quality. An intermediate value yields the best trade-off.

λmax = 1.1 λmax = 1.3 λmax = 1.6

[Figure 43]

[Figure 44]

[Figure 45]

- Figure 13. Ablation on λmax. A small λmax (e.g., 1.1) yields insufficient concentration and suboptimal quality, while a large value (e.g., 1.6) causes over-sharpening. An intermediate setting achieves the best balance.

λmin = 1.0 λmin = 1.1 λmin = 1.2

[Figure 46]

[Figure 47]

[Figure 48]

- Figure 14. Ablation on λmin. Even a small λmin (e.g., 1.1) introduces structural inconsistencies, showing that not all attention patterns should be concentrated and supporting our choice to apply stronger focusing only to local patterns.

global factor λ = 1.0 global factor λ = 1.1 global factor λ = 1.2 our entropy-guided factor

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

- Figure 15. Ablation studies. Using a single global concentration factor produces a trade-off: small factors (e.g., λ = 1.0, 1.1) hurt visual quality, while large factors introduce structural inconsistency (e.g., λ = 1.2), and even mild values (e.g., λ = 1.1) cause noticeable mismatches. In contrast, our entropy-guided strategy applies weak focusing to global patterns and stronger focusing to local patterns, achieving both high visual quality and stable structure.

PE PI NTK YaRN Entropy Ours

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

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

- Figure 16. Qualitative comparison in direct resolution extrapolation on Flux. Our method outperforms baselines by delivering high visual quality while mitigating content repetition.

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

HiFlow I-Max Ours

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

- Figure 17. Qualitative comparison on guided resolution extrapolation. Our method outperforms baselines across by delivering high visual quality.

NTK YaRN Ours

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- Figure 18. Qualitative comparison on guided view extrapolation. Our method outperforms baselines by delivering high visual quality while mitigating content repetition.

