## One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models

[Figure 1]

Aleksandr Razin∗

razin.ad@edu.spbstu.ru

Kazantsev Danil∗

311495@niuitmo.ru

Ilya Makarov

iamakarov@hse.ru

# arXiv:2511.10629v2[cs.CV]21Nov2025

### Abstract

Diffusion models struggle to scale beyond their training resolutions, as direct high-resolution sampling is slow and costly, while post-hoc image super-resolution (ISR) introduces artifacts and additional latency by operating after decoding. We present the Latent Upscaler Adapter (LUA), a lightweight module that performs super-resolution directly on the generator’s latent code before the final VAE decoding step. LUA integrates as a drop-in component, requiring no modifications to the base model or additional diffusion stages, and enables high-resolution synthesis through a single feed-forward pass in latent space. A shared Swin-style backbone with scale-specific pixel-shuffle heads supports ×2 and ×4 factors and remains compatible with imagespace SR baselines, achieving comparable perceptual quality with nearly 3× lower decoding and upscaling time (adding only +0.42s for 1024px generation from 512px, compared to 1.87s for pixel-space SR using the same SwinIR architecture). Furthermore, LUA shows strong generalization across the latent spaces of different VAEs, making it easy to deploy without retraining from scratch for each new decoder. Extensive experiments demonstrate that LUA closely matches the fidelity of native high-resolution generation while offering a practical and efficient path to scalable, high-fidelity image synthesis in modern diffusion pipelines. Code: github.com/vaskers5/LUA.

### 1. Introduction

Diffusion models have transformed image synthesis, progressing from pixel-space formulations [16, 33] to Latent Diffusion Models (LDMs), which shift computation into compact latent representations [31]. This latent formulation underpins contemporary systems for image generation, editing, and translation [2, 11]. Despite these advances, current models are effectively constrained by the spatial resolutions seen during training (typically 5122 or 10242). Na¨ıvely sampling beyond these scales often yields repeti-

∗ Equal Contribution.

[Figure 2]

Figure 1. Our proposed lightweight Latent Upscaler Adapter (LUA) integrates into diffusion pipelines without retraining the generator/decoder and without an extra diffusion stage. The example uses a FLUX [2] generator: it produces a 64×64 latent for a 512px image (red dashed path decodes directly). Our path (green dashed) upsamples the same latent to 128×128 (×2) or 256×256 (×4) and decodes once to 1024px or 2048px, adding only +0.42s (1K) and +2.21s (2K) on an NVIDIA L40S GPU. LUA outperforms multi-stage high-resolution pipelines while avoiding their extra diffusion passes, and achieves efficiency competitive with image-space SR at comparable perceptual quality, all via a single final decode.

tion, geometric distortions, and texture breakdown [14, 40]. While retraining or high-resolution fine-tuning can reduce such artifacts [23], these remedies demand substantial compute and data.

A pragmatic alternative is to generate at the native resolution and subsequently upsample. Two main paradigms implement this strategy. Pixel-space super-resolution (SR) applies an external SR model to the decoded image. This approach is simple but reconstructs fine structure purely from pixels, which encourages oversmoothing, semantic drift, and a computational cost that grows quadratically with output size [6, 38]. Latent-space upsampling instead enlarges the latent representation prior to decoding, reducing inference cost and better preserving semantics. Recent reference-based methods, such as DemoFusion-style pipelines [10] and LSRNA [17], first generate a low-

[Figure 3]

Figure 2. Upscaling FLUX outputs [2] from 10242→20482. Columns: (1) base decode, (2) bicubic latent, (3) SwinIR image-space SR, (4) LUA latent-space SR. Top: runtime overhead vs. (1). Middle (8× crops): bicubic blurs/aliases; SwinIR sharpens but adds noise/texture drift; LUA preserves eyelashes and skin with stable edges. Bottom: Laplacian-variance maps (darker=less noise) with means—LUA attains the lowest residual noise and the smallest overhead via single-decode latent upscaling.

resolution reference, upsample it, and then run a second diffusion stage guided by the upsampled latent. Although effective, these pipelines require multi-stage inference, auxiliary noise or guidance branches, and tight coupling to specific VAEs, which increases latency and limits generality across model families.

In practice, the upsampling step is the principal bottleneck. Bicubic resizing or na¨ıve latent interpolation departs from the manifold of valid latents, producing unnatural textures after decoding (Fig. 2, column 2). Conversely, pixelspace SR applied after decoding can improve fidelity but incurs higher latency and may introduce noise (Fig. 2, column 3). The core challenge is to increase latent resolution while preserving manifold geometry and high-frequency latent details that decode into realistic images—without invoking an additional diffusion process.

We address this challenge with the Latent Upscaler Adapter (LUA), a lightweight module inserted between the generator and VAE decoder. Given a latent z ∈ Rh×w×C, LUA predicts an upscaled latent zˆ ∈ Rαh×αw×C for α ∈ 2,4, which is decoded once to produce the final image. Because VAE decoders typically expand spatial dimensions by stride s=8, a ×2 latent upscaling yields a 16× increase in pixel count with no extra denoising. LUA employs a shared SwinIR-style backbone [24] with lightweight, scale-specific pixel-shuffle heads and is trained directly in the latent space with complementary latent- and pixel-domain objectives to preserve high-frequency consistency and enable photoreal-

istic decoding (Fig. 2, column 4). Our study focuses on three questions:

- 1. Can a simple latent adapter deliver higher-quality images at lower cost than native high-resolution synthesis or pixel-space SR?
- 2. Can a single adapter handle multiple upscaling factors within one unified framework?
- 3. Can an adapter trained for one model’s VAE transfer to other generators with minimal fine-tuning? We evaluate LUA across backbones and resolutions us-

ing FID/KID/CLIP and local-detail metrics, with ablations on architecture, objectives, and scale handling. Preserving latent microstructure via a single upscaling stage narrows the gap to native high-res synthesis while reducing complexity and latency relative to multi-stage diffusion and pixel-space SR.

Our contributions are threefold:

- • We train only a latent upscaler adapter with a multi-stage curriculum and show that latent-space super-resolution attains quality comparable to modern high-resolution diffusion pipelines while being more efficient and less noisy than pixel-space SR.
- • We design a single model that supports multiple scale factors (×2, ×4) via a shared backbone with jointly trained, scale-specific heads—avoiding retraining from scratch.
- • We demonstrate cross-VAE generalization: the same backbone operates across SD3 [11], SDXL [28], and FLUX [2] by changing only the first layer to match input channels with minimal fine-tuning.

### 2. Related Work

This section reviews three lines of work relevant to highresolution image synthesis: (i) efficient diffusion-based generation at large scales, (ii) super-resolution in pixel and latent spaces, and (iii) multi-scale (discrete vs. continuous) super-resolution. For each, we outline representative methods and the limitations that motivate our approach.

Efficient high-resolution generation with diffusion models. Diffusion models in compressed latent spaces [31] enable controllable synthesis, yet sampling beyond the training scale (5122–10242) often yields repetition, distortions, or texture loss [14, 40]. Direct high-resolution training has been demonstrated in large systems such as SDXL [28], SD3 [11], and related frameworks [23], but it demands massive datasets and compute. Inference-time strategies avoid full retraining: tiling/blending in MultiDiffusion [1] preserves locality but risks seams; receptive-field expansions via adaptive/dilated or sparse convolutions [12] and step-reduction by distillation/consistency methods [27, 34] improve speed but can reduce fidelity at extreme resolutions; progressive pipelines such as HiDiffusion and Scale-

Crafter [14, 40] iteratively upsample and refine; referencebased approaches like DemoFusion [10] synthesize a lowresolution latent, upsample it, and re-diffuse. These families rely on additional diffusion passes with method-specific schedules and step counts, increasing latency and coupling execution to particular resolution settings.

Super-resolution in image and latent spaces. Pixelspace SR advanced from CNN regressors (SRCNN, EDSR) [9, 25] to adversarial/perceptual (SRGAN, ESRGAN) [20, 37] and transformer models (SwinIR, HAT) [7, 24]; diffusion-based SR (SR3, SRDiff, SeeSR, StableSR, DiffBIR, SUPIR) [22, 26, 32, 36, 38, 39] further boosts perceptual fidelity. However, all denoise at the target resolution, incurring quadratic compute/memory with image size and risking semantic drift in fine textures. Latent-space SR reduces cost by upsampling before decoding, but na¨ıve latent interpolation (e.g., bicubic/linear) departs from the generative manifold, and learned mappings such as LSRNA or the latent guidance used in DemoFusion [10, 17] still depend on a subsequent diffusion stage, yielding multi-stage, latency-heavy pipelines.

Discrete vs. continuous multi-scale SR. Discrete-factor SR (e.g., ×2, ×4) commonly trains separate networks or employs a shared backbone with scale-specific heads (MDSR, SwinIR) [24, 25], which is effective but resource intensive to train and store for multiple scales. Arbitraryscale methods (LIIF, LTE, CiaoSR) [4, 8, 21] predict continuous coordinates from learned features, yet often underperform direct upscaling on fine textures where high-frequency structures dominate.

In contrast, we target a single-pass latent upscaler that avoids extra diffusion stages, side-steps the quadratic cost and drift of pixel-space SR, and replaces na¨ıve latent resizing with a dedicated training curriculum. The design supports multiple scales via a shared backbone with lightweight, scale-specific heads rather than separate perscale models or weaker arbitrary-scale decoders. This aims to deliver high-resolution fidelity with substantially lower latency and practical deployment characteristics.

### 3. Proposed Method

We target high-resolution synthesis without an extra diffusion stage or retraining the generator/VAE by inserting a single-pass Latent Upscaler Adapter (LUA) between the generator and the frozen decoder: LUA enlarges the latent, then one decode produces the final image with near–basedecode overhead. In Sec. 3.1 we formalize the upscaling operator Uα, establish its computational efficiency relative to image-space SR and multi-stage diffusion, and show cross-model generalization (FLUX, SD3, SDXL) via

[Figure 4]

Figure 3. Cross-model 2× latent upscaling with a single adapter. For SDXL [28], SD3 [11], and FLUX [2], a 128×128 latent is upscaled to 256×256 by the same LUA and decoded once by each model’s native VAE to yield 20482 images. SD3 and FLUX share C=16 latents; SDXL (C=4) is supported by changing only the first convolution. Insets show artifact-free detail preservation; green boxes mark ×8 zooms.

changing only the first convolution layer and minimal finetuning. We then describe the multi-scale architecture—a shared backbone with scale-specific heads for ×2 and ×4 (Sec. 3.2)—and the multi-stage training strategy that preserves latent microstructure and stabilizes decoded appearance (Sec. 3.3).

#### 3.1. Latent Upscaling

Formulation. Given text condition c and noise ϵ, a pretrained generator G produces a latent z ∈ Rh×w×C:

z = G(c,ϵ). (1)

A frozen VAE decoder D with spatial stride s (typically s=8) maps z to an RGB image x ∈ R(sh)×(sw)×3:

x = D(z). (2)

We introduce a deterministic latent upscaler Uα with scale factor α ∈ {2,4} that maps z ∈ Rh×w×C to zˆ ∈ Rαh×αw×C:

zˆ = Uα(z). (3) A single decode yields the high-resolution image:

###### xˆ = D(ˆz). (4)

Here, h,w are the latent spatial dimensions, C is the latent channel width, s is the decoder stride, c is the conditioning (e.g., text embedding), and ϵ is the generator noise. All generative stochasticity resides in G; Uα is a feed-forward operator trained to remain on the latent manifold and to preserve the fine-scale statistics required for photorealistic decoding.

Computational efficiency. Pixel-space SR operates on (sh)×(sw) positions; LUA operates on h×w positions and still decodes once. The cost ratio scales as

O((sh)(sw)) O(hw) ≈ s2, (5)

[Figure 5]

Figure 4. Architecture of the Latent Upscaler Adapter (LUA). A SwinIR-style backbone [24] is shared across scales; a 1×1 input conv adapts the VAE latent width (C=16 for FLUX/SD3; C=4 for SDXL). Scale-specific pixel-shuffle heads output ×2 or ×4 latents. At inference, the path selects the input adapter, runs the shared backbone, and activates the requested head. The schematic shows FLUX/SD3 ×2 and SDXL ×4.

so for typical s=8 our upscaler touches about 1/64 as many spatial elements as image-space SR. In addition, unlike progressive or reference-based pipelines, LUA does not add a second diffusion pass or any full-resolution refinement stage. Empirically, this yields lower wall-clock overhead and memory traffic while achieving comparable highresolution fidelity (Sec. 4).

Cross-model generalization. LUA acts on latents, not backbone internals, and therefore does not require training from scratch for each VAE. The same model generalizes across FLUX, SD3, and SDXL by changing only the first convolution layer to match the input channel count and finetuning the adapter on a small set of latents from the target model. The backbone and scale heads remain unchanged. Architectural details are given in Sec. 3.2; the fine-tuning protocol and data sizes are in Sec. 4.

#### 3.2. Architecture

We adopt a Swin Transformer restoration backbone in the spirit of SwinIR [24], which has proven effective for super-resolution in RGB space, and consistent with latentdomain adaptations such as LSRNA [17]. Windowed selfattention provides long-range context while preserving locality, matching the spatial–statistical structure of VAE latents. Given an input latent z ∈ Rh×w×C, the backbone ϕ(·) extracts features through an encoder–decoder with residual connections. Upscaling is realized with explicit SR heads—shallow convolutions followed by pixelshuffle—that directly predict an enlarged latent; compared with implicit coordinate decoders (e.g., LIIF [8]), this better preserves high-frequency latent microstructure that decodes into sharper textures.

To support multiple scale factors without duplicating capacity, a single backbone ϕ is shared across scale-specific heads for ×2 and ×4 (denoted U×2 and U×4). Joint training

with balanced sampling encourages the backbone to learn scale-agnostic representations while each head specializes to its factor’s aliasing and artifact profile. At inference, the backbone runs once and the head corresponding to the requested scale α ∈ {2,4} is applied:

xˆ = D U×α(ϕ(z)) , α ∈ {2,4}, (6)

where D is the frozen VAE decoder and z is the input latent with channel width C. The overall module layout is depicted in Fig. 4.

To operate across VAEs with different latent channel widths (e.g., SD3/FLUX with C=16 and SDXL with C=4), we change only the first convolution layer to match the input channels; the shared backbone and the scale heads are reused unchanged. A brief fine-tuning on a small set of latents from the target model aligns statistics and enables cross-VAE transfer without training from scratch. The finetuning protocol and data sizes are detailed in Sec. 4, and cross-model results are shown in Fig. 3.

#### 3.3. Multi-stage Training Strategy

Single-domain objectives are insufficient for latent SR: optimizing only in latent space preserves coarse structure but yields decoded images with residual grid/blur and spurious high-frequency noise, while optimizing only in pixel space is unstable—gradients backpropagated through the frozen VAE decoder interact with unnormalized latents and fail to converge. To address both issues, we adopt a progressive three-stage curriculum that first secures latent structural and spectral alignment, then couples latent fidelity to decoded appearance with high-frequency emphasis, and finally refines edges in pixel space without re-diffusion. Throughout this section, z and zˆ denote input and upscaled latents, zHR the reference HR latent (from the frozen VAE encoder), and x=D(z), xˆ=D(ˆz), xHR=D(zHR) their decoded images via the frozen decoder D. Superscripts z and x indicate losses computed in latent and pixel domains, respectively.

Stage I — Latent-domain structural alignment. This stage learns a stable mapping zˆ = Uα(z) that matches highresolution latent structure and spectra while avoiding oversmoothing. The objective is

LSI = α1 LzL1 + β1 LzFFT, (7)

where α1,β1 ≥ 0 balance reconstruction and spectral alignment in latent space, with

LzL1 = z ˆ − zHR 1, (8)

LzFFT = F(ˆz) − F(zHR) 1. (9) Here zˆ ∈ Rαh×αw×C and F(·) is the channel-wise 2D FFT magnitude [13]. LzL1 enforces element-wise correspondence; LzFFT aligns high-frequency latent statistics to preserve microstructure [18].

- Stage II — Joint latent–pixel consistency. This stage

links latent fidelity to decoded appearance by adding imagedomain constraints while retaining the Stage I latent terms:

LSII = α2 LzL1 + β2 LzFFT + γ2 LxDS + δ2 LxHF, (10)

- with α2,β2,γ2,δ2 ≥ 0 and

LxDS = ↓d (ˆx)− ↓d (xHR) 1, (11) LxHF = x ˆ − Gσ(ˆx) − xHR − Gσ(xHR)

1

. (12)

Here ↓d (·) denotes bicubic downsampling [29] with d=2 for ×2 and d=4 for ×4, and Gσ(·) is a Gaussian blur with σ=1.0 (applied channel-wise). LxDS enforces coarse appearance consistency at a common reduced scale; LxHF emphasizes edges and textures by matching high-frequency residuals.

Stage III — Edge-aware image refinement. The final stage sharpens edges and suppresses residual ringing/grids in pixel space, without any additional denoising:

LSIII = α3 LxL1 + β3 LxFFT + γ3 LxEAGLE, (13)

- with α3,β3,γ3 ≥ 0 and

LxL1 = x ˆ − xHR 1, (14) LxFFT = F(ˆx) − F(xHR) 1, (15)

where LxEAGLE is an edge-aware gradient localization loss that enforces crisp boundaries and reduces staircase artifacts [35].

Weight selection. Weights were set via grid search: ℓ1 and FFT terms (latent and image domains) receive the largest coefficients as primary reconstruction and texture-sensitive objectives, while downsampling and edge-aware terms use smaller auxiliary regularizers. Please see the appendix for detailed grid-search settings and per-stage loss weights.

Our multi-stage latent–pixel curriculum adapts LUA to the latent domain and the frozen decoder. Stage I aligns the upscaled latent with high-resolution structure and spectra (Fig. 5, col. 2). Stage II introduces joint latent–image supervision (downsampled and high-frequency terms) to normalize latents before decoding and suppress noise (col. 3).

- Stage III trains in pixel space only, selectively removing redundant noise and grid artifacts while preserving necessary high-frequency detail (col. 4). The curriculum enables coherent cross-domain training and yields single-decode synthesis without an extra diffusion stage; per-stage loss weights are provided in the appendix.

### 4. Experiments

We evaluate LUA on OpenImages with a unified train/val setup, comparing to representative high-resolution baselines and reporting fidelity (FID/KID and patch variants),

[Figure 6]

Figure 5. Effect of the three-stage curriculum on latent reconstruction and decoded appearance (FLUX backbone). The 2×4 grid shows top: latent feature maps (channel 10, min–max normalized); bottom: corresponding 8× zoomed decodes. Columns: (1) original low-resolution latent (1282) and decode; (2–4) LUA upscaled latents to 2562 after Stage I–III with their decodes. Yellow boxes mark the zoomed region. From (2) to (4), decodes become less noisy and more structured; Stage III concentrates high-frequency energy around details, indicating that controlled latent noise aids faithful VAE decoding.

text–image alignment (CLIP), and wall-clock latency. We also study cross-model and multi-scale generalization and ablate the curriculum and architecture. Further implementation details and extended results are provided in the appendix.

#### 4.1. Experiment Settings

Training data. Following LSRNA [17], we use OpenImages [19]; photos with both sides ≥ 1440px are tiled into non-overlapping 512×512 crops. HR/LR pairs are made by bicubic downsampling at the target scale (×2, ×4). Crops are encoded with the FLUX VAE (stride s=8, C=16) to produce latent: for ×2, 16×32×32 → 16×64×64; for ×4, 16×16×16 → 16×64×64. The final set contains 3.8M pairs.

Multi-stage training. We train the three stages (Sec. 3.3) with Adam (lr 2×10−4, weight decay 0), EMA 0.999, grad-clip 0.4, and MultiStepLR (milestones 62.5k/93.75k/112.5k, γ=0.5); each stage runs 125k steps. Hyperparameters follow SwinIR [24]. We keep the same initial lr across stages; Stage III uses a short warm-up. Effective batch sizes: 2,048 (Stage I) and 32 (Stages II–III). Losses match Sec. 3.3.

Scalability: multi-scale and cross-VAE. A single Swin backbone is shared by the ×2/ × 4 heads (U×2, U×4) and trained jointly with balanced sampling (Sec. 3, Fig. 4). For cross-VAE transfer, we build 500k LR/HR latent pairs for SDXL (C=4) and SD3 (C=16), replace only the first convolution to match channels, and perform a brief Stage III–style fine-tune, reusing the backbone and heads without retraining from scratch.

- Table 1. OpenImages validation. Metrics follow Sec. 4.2 (FID/pFID, KID/pKID, CLIP) and runtime (median s). H100, batch size 1. For 1K (10242), SDXL samples at 5122 then upscales to 10242 via LUA (latent) or SwinIR (pixel). For 2K/4K, images are sampled at 10242 then upscaled to 20482/40962. LUA achieves the lowest latency at all resolutions and the strongest fidelity at 2K/4K; best marked in bold. Resolution Method FID ↓ pFID ↓ KID ↓ pKID ↓ CLIP ↑ Time (s)

HiDiffusion 232.55 230.39 0.0211 0.0288 0.695 1.54 DemoFusion 195.82 193.99 0.0153 0.0229 0.725 2.04 LSRNA–DemoFusion 194.55 192.73 0.0151 0.0228 0.734 3.09 SDXL (Direct) 194.53 192.71 0.0151 0.0225 0.731 1.61 SDXL + SwinIR 210.40 204.23 0.0313 0.0411 0.694 2.47 SDXL + LUA (ours) 209.80 191.75 0.0330 0.0426 0.738 1.42

1024×1024

HiDiffusion 200.72 114.30 0.0030 0.0090 0.738 4.97 DemoFusion 184.79 177.67 0.0030 0.0100 0.750 28.99 LSRNA–DemoFusion 181.24 98.09 0.0019 0.0066 0.762 20.77 SDXL (Direct) 202.87 116.57 0.0030 0.0086 0.741 7.23 SDXL + SwinIR 183.16 100.09 0.0020 0.0077 0.757 6.29 SDXL + LUA (ours) 180.80 97.90 0.0018 0.0065 0.764 3.52

2048×2048

HiDiffusion 233.65 95.95 0.0158 0.0214 0.698 122.62 DemoFusion 185.36 177.89 0.0043 0.0113 0.749 225.77 LSRNA–DemoFusion 177.95 62.07 0.0023 0.0071 0.757 91.64 SDXL (Direct) 280.42 101.89 0.0396 0.0175 0.663 148.71 SDXL + SwinIR 183.15 65.71 0.0018 0.0103 0.756 7.29 SDXL + LUA (ours) 176.90 61.80 0.0015 0.0152 0.759 6.87

4096×4096

#### 4.2. Evaluation Protocol

Validation data. We evaluate on 1,000 held-out highresolution OpenImages photos (disjoint from training). Prompts are obtained via captioning and shared across methods; see the appendix for details.

Metrics. We report FID [15], KID [3], and CLIP [30]. Images are synthesized at the target resolution (10242/20482/40962). To better capture fine detail, we also report patch metrics (pFID/pKID) on random 1024×1024 crops with an identical crop policy across methods, following LSRNA and Anyres-GAN [5]. Additional metrics and computation details are provided in the appendix.

Baselines. We compare ScaleCrafter [14], HiDiffusion [40] (progressive), DemoFusion [10], LSRNA–DemoFusion [17] (reference-based re-diffusion), SDXL (Direct), and SDXL+SwinIR [24] (pixel-space SR). Samplers, steps, guidance, and prompts are held constant where applicable; see the appendix for configuration details.

Our method. SDXL+LUA: generate at 10242, upscale in latent space, decode once. We report SDXL-anchored comparisons in Table 1 and summarize cross-model/multiscale results (FLUX/SD3/SDXL at ×2/ × 4) in Table 2.

Runtime measurement. Per-image wall-clock time is the median over N runs after 20 warm-ups on a single GPU with AMP and batch size 1. Composite pipelines include generation, upscaling, and decode.

#### 4.3. Quantitative Results

Table 1 summarizes fidelity–efficiency trade-offs across resolutions. At 10242, SDXL+LUA attains the lowest latency (1.42s) but lags strongest single-stage baselines on fidelity (FID 209.80 vs. 194.53 for SDXL (Direct) and 194.55 for LSRNA–DemoFusion). We attribute this gap to lowresolution input latent (64×64 for 512px, stride 8), which constrains recoverable detail in ×2 setting; notably, our patch fidelity is competitive (pFID 191.75, best in row), indicating preserved local structure despite lower scores.

Beyond 1K, LUA consistently improves both quality and speed. At 20482, SDXL+LUA delivers the best fidelity among single-decode pipelines while remaining the fastest: FID 180.80, pFID 97.90, KID 0.0018, CLIP 0.764 in 3.52s, outperforming SDXL+SwinIR (183.16 / 100.09 / 0.0020 / 0.757 in 6.29s) and substantially undercutting multi-stage re-diffusion (LSRNA–DemoFusion: FID 181.24 in 20.77s; DemoFusion: 184.79 in 28.99s). At 40962, SDXL+LUA again sets the best single-pass fidelity (FID 176.90; pFID 61.80) with the lowest runtime (6.87s), surpassing SDXL+SwinIR (183.15; 65.71; 7.29s) and avoiding the quality collapse of direct high-res sampling (SDXL (Direct) FID 280.42). LSRNA–DemoFusion attains slightly lower pKID (0.0071) but is an order of magnitude slower (91.64s), underscoring LUA’s favorable accuracy–latency frontier. These results highlight the efficacy of LUA for high-resolution synthesis.

Table 2 further shows that a single backbone generalizes across models and scales with minimal adaptation: at

- Table 2. Cross-model, multi-scale results for LUA. Metrics, crop protocol, runtimes, and hardware match Table 1. We evaluate ×2/ × 4 latent upscaling (10242→20482/40962) on FLUX, SDXL, and SD3; best results are in bold. Scale Diffusion Model FID ↓ pFID ↓ KID ↓ pKID ↓ CLIP ↑ Time (s)

FLUX + LUA 180.99 100.40 0.0020 0.0079 0.773 29.829 SDXL + LUA 183.15 101.18 0.0020 0.0087 0.753 3.52 SD3 + LUA 184.58 103.94 0.0022 0.0083 0.768 20.292

×2

FLUX + LUA 181.06 62.30 0.0018 0.0085 0.772 31.908 SDXL + LUA 182.42 71.92 0.0015 0.0152 0.754 6.87 SD3 + LUA 183.34 67.25 0.0016 0.0095 0.769 21.843

×4

×2, FLUX+LUA reaches FID 180.99 and CLIP 0.773; at ×4, SDXL+LUA yields KID 0.0015 while FLUX+LUA attains the strongest pFID (62.30). These results confirm robust transfer across SDXL, SD3, and FLUX and consistent gains at both ×2 and ×4 without retraining the generator or adding diffusion stages. These findings highlight the adaptability of LUA across models and scales.

#### 4.4. Qualitative Results

Figure 6 presents side-by-side comparisons at 20482 (top two rows) and 40962 (bottom two), starting from the same 10242 SDXL bases (identical seeds/prompts). Direct highresolution sampling (SDXL Direct) exhibits canonical largescale failures: duplicated structures and geometry drift in the crab legs and sand granules (row 1), brittle clumps in dog fur and snow particles (row 2), and warped taillights/reflections in the street scene (row 3). HiDiffusion shows similar breakdowns at 4K (row 3), indicating that training-free escalation struggles to maintain global layout at extreme scales. In contrast, SR-from-base approaches avoid these hallucinations by inheriting the cleaner 10242 generation.

Among SR methods, pixel-space SwinIR sharpens but introduces ringing/haloing and plastic textures: overshot specular ridges on the crab shell (row 1), halos at fur boundaries (row 2), glare around car edges (row 3), and granular noise on petals (row 4). DemoFusion+LSRNA restores rich texture but requires a second diffusion stage with much higher latency. SDXL+LUA (ours) preserves edge continuity and microstructure with fewer artifacts: crisp eyelashes and shell ridges without halos (row 1), distinct yet stable fur strands (row 2), sharp panel seams and reflections without brittleness (row 3), and coherent high-frequency petal detail (row 4). Per-image runtimes overlaid above each panel align with Table 1: LUA attains comparable or better visual quality at the lowest latency via single-decode latent upscaling. Additional examples are provided in the appendix.

#### 4.5. Ablation Studies

Multi-stage training effectiveness. We assess the curriculum by removing individual stages and comparing against a latent ℓ1–only baseline. Table 3 shows that the

- Table 3. Ablation on the multi-stage latent–pixel curriculum for ×2 (512→1024) and ×4 (256→1024). Best results are in bold.

Configuration PSNR ↑ LPIPS ↓ PSNR ↑ LPIPS ↓ Latent ℓ1 28.53 0.198 26.16 0.236

- I+II (w/o III) 28.96 0.172 26.67 0.213
- I+III (w/o II) 31.05 0.150 27.10 0.198 II+III (w/o I) 31.60 0.145 27.40 0.192 Full (I+II+III) 32.54 0.138 27.94 0.184

- Table 4. Ablation of upsampling strategies for ×2 (512→1024) and ×4 (256→1024). Best results are in bold.

Variant PSNR ↑ LPIPS ↓ PSNR ↑ LPIPS ↓ LIIF 29.10 0.210 26.10 0.235 Separated-×2 31.92 0.150 – – Separated-×4 – – 27.71 0.189 Joint Multi-Head 32.54 0.138 27.94 0.184

full configuration (I+II+III) yields the best fidelity at both scales. At ×2, the full model attains PSNR 32.54 / LPIPS 0.138 versus 28.96 / 0.172 for I+II (w/o III). At ×4, II+III (w/o I) reaches 27.40 / 0.192 compared to 27.94 / 0.184 for the full model. Removing Stage III reduces edge sharpness and perceptual quality; removing Stage II weakens latent–pixel consistency and increases artifacts. A pixel-only variant fails to converge with a frozen VAE. Please, follow appendix for training settings details.

Multi-scale super-resolution adaptation. We compare three design choices per scale factor: (i) an implicit, coordinate-based upsampler (LIIF), (ii) per-scale specialist models trained only for a single factor (×2 or ×4), and (iii) a joint multi-head model that shares a backbone and uses scale-specific heads. Table 4 reports within-scale results. The joint multi-head design attains the best PSNR/LPIPS at both ×2 and ×4, while also consolidating capacity in a single backbone, reducing storage and training overhead (see appendix for resource analysis and experiments settings).

- 5. Discussion

Our approach inherits limitations from its adapter nature: errors or biases in the generator’s latent are faithfully upscaled, so artifacts in the base sample can per-

[Figure 7]

- Figure 6. Qualitative comparison at 20482 and 40962 starting from the same 10242 SDXL base generations. Each row uses identical seeds and prompts (GPT-generated captions from OpenImages validation). Red boxes indicate 12× magnified crops; titles report per-image runtime. For visual clarity we show the DemoFusion+LSRNA variant in place of plain DemoFusion. The column with SDXL+LUA (ours) achieves the lowest latency and produces clean, stable textures without the high-resolution artifacts (e.g., repetition, gridding) seen in direct high-res sampling, and without the sharpening noise typical of pixel-space SR.

sist at higher resolution. A promising direction is joint refinement–and–upscaling directly in latent space before decoding, e.g., lightweight consistency modules to suppress artifacts while preserving semantics, with uncertaintyaware gating to invoke refinement only when needed. Beyond text-to-image, the same mechanism can serve imageto-image tasks requiring high-resolution outputs—such as depth-to-image or semantic maps, where preserving structure during upscaling is critical. Finally, extending the adapter to video with temporal consistency (e.g., recurrent latent refinement or temporal priors) remains essential for practical high-resolution synthesis in dynamic settings.

### 6. Conclusion

We introduced LUA, a single-pass latent upscaler inserted between a pretrained generator and a frozen VAE decoder, and demonstrated that latent-space upscaling is train-

able via a multi-stage latent–pixel curriculum with scalespecific ×2/ × 4 heads. On OpenImages, at 20482 and 40962, SDXL+LUA achieves state-of-the-art single-decode fidelity (FID 180.80 / 176.90; pFID 97.90 / 61.80) while remaining the fastest; at 20482 it runs in 3.52s versus 7.23s for SDXL (Direct), outperforms pixel-space SR (SDXL+SwinIR), and approaches multi-stage re-diffusion quality at a fraction of the runtime. At 10242, performance trails the strongest baselines due to the 64×64 input latent constraint, though patch-level fidelity remains competitive. A single backbone transfers across SDXL, SD3, and FLUX with only a first-layer channel change and brief fine-tuning, demonstrating robust cross-VAE and multi-scale generalization without modifying the generator or adding diffusion stages. Overall, these results establish single-decode latent upscaling as a practical alternative to multi-stage highresolution pipelines.

### Acknowledgements

The authors are grateful to colleagues at CAIRO, Technical University of Applied Sciences W¨urzburg–Schweinfurt, Germany, for their sustained support throughout this project. We especially thank Pavel Chizhov for insightful discussions and assistance with manuscript revisions, which substantially improved the quality and clarity of this work.

### References

- [1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation.(2023). URL https://arxiv. org/abs/2302.08113,

2023. 2

- [2] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 1, 2, 3

- [3] Mikołaj Bi´nkowski, Danica J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans, 2021. 6
- [4] Jiezhang Cao, Qin Wang, Yongqin Xian, Yawei Li, Bingbing Ni, Zhiming Pi, Kai Zhang, Yulun Zhang, Radu Timofte, and Luc Van Gool. Ciaosr: Continuous implicit attention-inattention network for arbitrary-scale image super-resolution,

2023. 3

- [5] Lucy Chai, Michael Gharbi, Eli Shechtman, Phillip Isola, and Richard Zhang. Any-resolution training for highresolution image synthesis, 2022. 6
- [6] Weimin Chen, Yuqing Ma, Xianglong Liu, and Yi Yuan. Hierarchical generative adversarial networks for single image super-resolution. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 355– 364, 2021. 1
- [7] Xiangyu Chen, Xintao Wang, Wenlong Zhang, Xiangtao Kong, Yu Qiao, Jiantao Zhou, and Chao Dong. Hat: Hybrid attention transformer for image restoration, 2025. 3
- [8] Yinbo Chen, Sifei Liu, and Xiaolong Wang. Learning continuous image representation with local implicit image function, 2021. 3, 4
- [9] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence, 38(2):295–307, 2015. 3
- [10] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising highresolution image generation with no $. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6159–6168, 2024. 1, 3, 6
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 1, 2, 3

- [12] Mohsen Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei Jafari, Sunando Sengupta, Hamid Reza Vaezi Joze, Eric Sommerlade, Hamed Pirsiavash, and J¨urgen Gall. Adaptive token sampling for efficient vision transformers. In European conference on computer vision, pages 396–414. Springer, 2022. 2
- [13] Dario Fuoli, Luc Van Gool, and Radu Timofte. Fourier space losses for efficient perceptual image super-resolution, 2021. 4
- [14] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2023. 1, 2, 3, 6
- [15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. 6
- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [17] Jinho Jeong, Sangmin Han, Jinwoo Kim, and Seon Joo Kim. Latent space super-resolution for higher-resolution image generation with diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2355–2365, 2025. 1, 3, 4, 5, 6
- [18] Xingyu Jiang, Xiuhui Zhang, Ning Gao, and Yue Deng. When fast fourier transform meets transformer for image restoration. In European Conference on Computer Vision, pages 381–402. Springer, 2024. 4
- [19] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020. 5
- [20] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network, 2017. 3
- [21] Jaewon Lee and Kyong Hwan Jin. Local texture estimator for implicit representation function, 2022. 3
- [22] Haoying Li, Yifan Yang, Meng Chang, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single image superresolution with diffusion probabilistic models, 2021. 3
- [23] Hao Li, Yang Zou, Ying Wang, Orchid Majumder, Yusheng Xie, R Manmatha, Ashwin Swaminathan, Zhuowen Tu, Stefano Ermon, and Stefano Soatto. On the scalability of diffusion-based text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9400–9409, 2024. 1, 2
- [24] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF inter-

national conference on computer vision, pages 1833–1844,

2021. 2, 3, 4, 5, 6

- [25] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution, 2017. 3
- [26] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior, 2024. 3
- [27] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14297–14306, 2023. 2
- [28] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 2, 3
- [29] Mohammad Saeed Rad, Thomas Yu, Claudiu Musat, Hazim Kemal Ekenel, Behzad Bozorgtabar, and JeanPhilippe Thiran. Benefiting from bicubically down-sampled images for learning real-world image super-resolution, 2020. 5
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 6
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2
- [32] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement, 2021. 3
- [33] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1
- [34] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. 2
- [35] Yipeng Sun, Yixing Huang, Zeyu Yang, Linda-Sophie Schneider, Mareike Thies, Mingxuan Gu, Siyuan Mei, Siming Bayer, Frank G Z¨ollner, and Andreas Maier. Eagle: an edge-aware gradient localization enhanced loss for ct image reconstruction. Journal of Medical Imaging, 12(1):014001– 014001, 2025. 5, 11
- [36] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C. K. Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution, 2024. 3
- [37] Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, and Xiaoou Tang. Esrgan: Enhanced super-resolution generative adversarial networks, 2018. 3
- [38] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution, 2024. 1, 3

- [39] Fanghua Yu, Jinjin Gu, Zheyuan Li, Jinfan Hu, Xiangtao Kong, Xintao Wang, Jingwen He, Yu Qiao, and Chao Dong. Scaling up to excellence: Practicing model scaling for photorealistic image restoration in the wild, 2024. 3
- [40] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Yuhao Chen, Yao Tang, and Jiajun Liang. Hidiffusion: Unlocking higherresolution creativity and efficiency in pretrained diffusion models. In European Conference on Computer Vision, pages 145–161. Springer, 2024. 1, 2, 3, 6

## One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models

Supplementary Material

### A. Training and Resource Analysis

This section details how LUA is trained in practice. We first specify the exact loss compositions and weights used in the three-stage curriculum (Sec. A.1), then summarize the optimization and scheduler settings (Sec. A.2), and finally discuss why the staged design is important for stability with a frozen VAE decoder (Sec. A.3). We conclude with the multi-scale training protocol and the associated resource usage (Sec. A.4).

#### A.1. Per-stage Loss Weights

Let z and zˆ denote the input and upscaled latents, and zHR the reference high-resolution latent (from the frozen FLUX encoder). Let x = D(z), xˆ = D(ˆz), and xHR = D(zHR) be the corresponding decoded images via the frozen decoder D. Superscripts z and x indicate whether a loss is computed in latent or pixel space, respectively. For each scale factor α ∈ {2,4} we use identical coefficients. Figure 7 provides a visual overview of all objective terms, their error maps, and their usage across stages.

##### Stage I (latent-domain alignment). Stage I uses the la-

tent ℓ1 and latent FFT losses from Eqs. (8)–(9) for each scale:

L(SIα) = α1 Lz,L1(α) + β1 Lz,FFT(α), (16)

where LzL1 is the element-wise ℓ1 distance between zˆ and zHR, and LzFFT is the channel-wise FFT magnitude loss in latent space. We set

α1 = 1.0, β1 = 0.1,

so that latent reconstruction is the primary term and spectral alignment acts as a lower-weight regularizer. Rows 1–2 of Fig. 7 visualize these losses in latent space.

###### Stage II (joint latent–pixel consistency). Stage II augments the Stage I objective with pixel-domain terms (Eqs. (11)–(12)):

L(SIIα) = α2 Lz,L1(α)+β2 Lz,FFT(α)+γ2 Lx,DS(α)+δ2 Lx,HF(α). (17)

Here, LxDS is the bicubic downsample-consistency loss between xˆ and xHR, computed after downsampling both images by a factor of 2; and LxHF is the Gaussian highfrequency ℓ1 loss (kernel size 5, σ = 1.0) on the residuals (x − Gσ(x)). We use

α2 = 1.0, β2 = 0.1, γ2 = 0.1, δ2 = 0.05.

Rows 3–4 of Fig. 7 show the corresponding downsampleconsistency and high-frequency residual error maps.

Stage III (edge-aware pixel refinement). Stage III operates only in pixel space as in Eqs. (14)–(16):

L(SIIIα) = α3 Lx,L1(α) + β3 Lx,FFT(α) + γ3 Lx,EAGLE(α) , (18)

where LxL1 is the pixel-wise ℓ1 loss between xˆ and xHR, and LxFFT is the FFT magnitude loss in image space (defined analogously to LzFFT but applied to RGB images). The edge-aware term LxEAGLE follows Sun et al. [35] and is defined as

LxEAGLE = H ⊙ |F(Vxout)| − |F(Vxtgt)| 1

(19)

+ H ⊙ |F(Vyout)| − |F(Vytgt)| 1,

where F(·) is the 2D FFT, Vxout and Vxtgt are the per-patch variance maps of the horizontal image gradients of xˆ and

xHR (and analogously for Vyout,Vytgt in the vertical direction), and H is a Gaussian high-pass mask in the frequency domain,

u2 + v2 2fc2

, (20)

H(u,v) = 1 − exp −

with normalized spatial frequencies (u,v) and cutoff fc = 0.5. Gradients are computed with 3×3 Scharr filters, and Vx,Vy are obtained by non-overlapping variance pooling over 3×3 patches. In our implementation, EAGLE is applied to luminance for RGB images and reduced with an ℓ1 norm over all frequencies. We use

α3 = 10.0, β3 = 1.0, γ3 = 5 × 10−5,

so that the strong ℓ1 term stabilizes Stage III, while LxEAGLE acts as a light regularizer that sharpens edges and suppresses grid artifacts. Rows 5–7 of Fig. 7 illustrate the corresponding pixel reconstruction, pixel-frequency, and EAGLE texture losses.

Hyperparameter selection. We select all coefficients (αk,βk,γk,δk) using a small held-out latent dataset (10k training pairs, 100 validation images). We first perform a coarse grid search over discrete weights {0,0.05,0.1,0.5,1,5,10}, measuring a scalar objective combining patch FID and LPIPS on the validation set. We then refine the best region with Optuna-based TPE search, using continuous ranges (e.g., α2 ∈ [0.5,2], β2,γ2 ∈ [0.05,0.5], δ2 ∈ [0.01,0.1]). The configuration above minimizes the validation objective averaged over both scale heads and is reused for the full training runs.

Latent GT zGT (Avg) Latent Error jz^¡ zGTj

Latent Pred z^ (Avg)

[Figure 8]

[Figure 9]

[Figure 10]

Latent Reconstruction

[Figure 11]

[Space: Latent] (Stages I, II)

1.75

1.50

LzL1 = kz^¡ zGTk1

1.25

1.00

Pixel-wise L1 loss in latent space. Ensures generated embeddings structurally match the encoder output.

0.75

0.50

16 £ 128 £ 128

16 £ 128 £ 128

Spectrum jF(z^)j Spectrum jF(zGT)j Latent Spectral Diff

[Figure 12]

[Figure 13]

[Figure 14]

Latent Freq Loss

[Figure 15]

1.0

[Space: Latent] (Stages I, II)

0.8

LzFFT / kjF(z^)j ¡ jF(zGT)jk1

0.6

0.4

Penalizes global frequency discrepancies in latents to preserve encoded texture information.

0.2

Downsampled # s(x^)

Downsampled # s(xHR) Low-Freq Error

[Figure 16]

[Figure 17]

[Figure 18]

Downsample Consistency

[Figure 19]

0.10

[Space: Pixel] (Stage II)

0.08

LxDS = k # s(x^) ¡ # s(xHR)k1

0.06

0.04

Ensures model learns low-frequency structure via bicubic downsampled consistency.

0.02

3 £ 512 £ 512

3 £ 512 £ 512

HF Residual jx^¡ G(x^)j HF Residual jx ¡ G(x)j High-Freq Error

[Figure 20]

[Figure 21]

[Figure 22]

0.07

HF Residual Loss

[Figure 23]

[Space: Pixel] (Stage II)

0.06

0.05

LxHF = kHF(x^) ¡ HF(xHR)k1

0.04

0.03

Loss on residuals after Gaussian blur (¾ = 1:0). Forces focus on edges and fine details.

0.02

0.01

Ground Truth xHR Pixel Error jx^¡ xHRj

Prediction x^

[Figure 24]

[Figure 25]

[Figure 26]

Pixel Reconstruction

[Figure 27]

0.12

[Space: Pixel] (Stage III)

0.10

LxL1 = kx^¡ xHRk1

0.08

0.06

Standard pixel-wise L1. Enforces global consistency but may over-smooth.

0.04

0.02

3 £ 1024 £ 1024

3 £ 1024 £ 1024

0.00

Spectrum jF(x^)j Spectrum jF(xHR)j Pixel Spectral Diff

[Figure 28]

[Figure 29]

[Figure 30]

Pixel Freq Loss

[Figure 31]

0.035

[Space: Pixel] (Stage III)

0.030

LxFFT / kjF(x^)j ¡ jF(xHR)jk1

0.025

0.020

0.015

Penalizes global frequency discrepancies in RGB, suppressing periodic artifacts.

0.010

0.005

Variance Vout Variance Vtgt EAGLE Spec Diff

[Figure 32]

[Figure 33]

[Figure 34]

EAGLE Texture Loss

[Figure 35]

0.7

[Space: Pixel] (Stage III)

0.6

LxEAGLE = XkH ¯ ¢jF(V)jk1

0.5

0.4

0.3

Edge-aware refinement. Computes spectral difference of gradient variance maps to sharpen edges.

0.2

0.1

0.0

- Figure 7. Visualization of all losses in the three-stage curriculum: latent reconstruction/frequency (Rows 1–2), downsample consistency and high-frequency residuals (Rows 3–4), and pixel reconstruction, pixel frequency, and EAGLE texture (Rows 5–7). Columns show prediction, ground truth, the corresponding error map, and the loss formula; brighter regions indicate larger gradient contributions.

#### A.2. Optimization and Scheduler Settings

All stages employ Adam with learning rate 2 × 10−4 and zero weight decay. Stage I and Stage II use momentum parameters (β1,β2) = (0.9,0.995), while Stage III uses (β1,β2) = (0.9,0.99) to adapt more quickly once supervision moves entirely to pixel space. Each stage is trained for 125,000 steps with a MultiStepLR scheduler (milestones at 62,500, 93,750, and 112,500 steps; decay factor γ = 0.5) and no warm-up. We maintain an exponential moving average of the generator parameters with decay 0.999 and use the EMA weights for validation and final reporting.

To prevent gradient explosion, we apply global ℓ2-norm

Pixel space train only visual artifacts

[Figure 36]

[Figure 37]

[Figure 38]

Figure 8. Artifacts from a pixel-only baseline trained without latent normalization. Decoded crops show saturated streaks and “broken” pixels caused by out-of-range latents passed to the frozen VAE decoder. These failures are avoided by the proposed threestage latent–pixel curriculum.

clipping with max norm = 0.4 to the generator in all stages. Stage I uses a per-GPU batch size of 64, combined with gradient accumulation to reach an effective batch of approximately 2,048 latent samples. Stages II and III use a perGPU batch size of 8 with accumulation, leading to an effective batch of 32 decoded samples for the more expensive pixel-domain losses. All experiments use distributed data parallelism with NCCL and mixed-precision training on 8 GPUs.

#### A.3. Latent Normalization and Limitations

Pixel-only training of LUA with a frozen VAE decoder is numerically unstable: when supervision is applied directly in image space from scratch, latent activations drift outside the decoder’s typical range, leading to saturated codes, localized “broken pixels” after decoding, and eventual loss divergence. The three-stage curriculum described in Sec. 3.3 mitigates this by first aligning latent statistics to those of the encoder and only then transitioning to pixeldomain optimization. In ablations, removing the intermediate (latent–pixel) stage or training the final pixel-only stage from random initialization consistently causes instability and pronounced grid-like artifacts.

A remaining limitation is the absence of an explicit bound on the latent values produced by LUA. Under out-ofdistribution inputs, the adapter can still emit latents that decode to localized corruptions (Fig. 8). Introducing explicit latent-domain regularization or lightweight post-processing (e.g., bounded residual corrections) is therefore a natural direction for improving robustness under distribution shift.

#### A.4. Multi-scale Training and Resource Usage

LUA is trained as a single multi-scale model that supports both ×2 and ×4 latent upscaling. For each training sample we form two latent pairs sharing the same high-resolution target: an intermediate-to-high pair for the ×2 head and a coarse-to-high pair for the ×4 head. In the early phase of each stage, both heads are supervised at every iteration so that the shared Swin backbone learns features useful for

both scales. In the final 50,000 steps, we switch to stochastic head selection, sampling either the ×2 or ×4 head with equal probability per iteration to encourage head-specific specialization while keeping the backbone shared. This protocol is reused across all three stages; only the loss composition changes (latent-only, joint latent–pixel, and pixel-only refinement).

All experiments were run on 8 NVIDIA H100 80GB GPUs. The three stages required approximately 16 hours (Stage I), 7.2 hours (Stage II), and 11 hours (Stage III), for a total of about 34.1 hours, i.e.,

34.1h × 8GPUs ≈ 2.7 × 102 GPU-hours.

This yields a single model that handles both scale factors. Training two independent models, one specialized for ×2 and one for ×4, with the same three-stage schedule would roughly double the budget to ≈ 5.4 × 102 GPUhours. The multi-head design therefore saves on the order of 2.7 × 102 GPU-hours (about a factor of two in training cost) while producing a single checkpoint deployable across both scales.

### B. Validation Prompts and Captions

This section describes how we obtain text prompts for the 1k-image validation set and how they are used for multiresolution evaluation.

#### B.1. Prompt Generation Procedure

For each image in the validation set (Sec. 4.2) we generate a single caption with a vision–language model based on GPT. The model is queried once per image with a fixed decoding configuration and instructed to return a short, self-contained description suitable for text-to-image generation. We query a GPT-based captioner with a fixed, structured instruction:

You are an image captioning model for text-to-image diffusion. Describe the image in 1-2 sentences, focusing on concrete objects, their attributes, layout, and photographic style. Return JSON as { "caption": "<caption here>" }.

The resulting "caption" field is used as the prompt for all methods and resolutions (10242, 20482, 40962).

#### B.2. Prompt Set Statistics and Examples

The prompts typically consist of one or two sentences (about 25–40 tokens), mentioning the main objects, their spatial arrangement, and basic style cues (e.g., lighting, depth of field). They cover a broad range of content, including natural scenes, animals, indoor objects, and people, and are used uniformly across all compared methods.

Figure 9 shows representative examples: for each validation image we display the original photograph, the generated caption, and synthesized outputs at 1024×1024, 2048×2048, and 4096×4096 resolutions driven by the same prompt. As resolution increases, some pipelines exhibit typical high-resolution hallucinations such as object duplication and overly dense textures, despite the caption remaining fixed. This observation motivates the strategy of generating a clean base image at a moderate resolution and then applying upsampling in latent or pixel space, rather than sampling directly at 2K–4K. We analyze this trade-off and compare direct high-resolution sampling with 1K+LUA upscaling in Sec. D.2.

### C. Baseline and Reproduction Details

This section summarizes the main configuration choices for all baselines in Sec. 4 and the common evaluation protocol used to ensure a fair comparison.

#### C.1. Diffusion and Super-resolution Baselines

Unless otherwise stated, all SDXL-based diffusion experiments use the same caption-derived positive prompts (Sec. B), a fixed negative prompt

blurry, ugly, duplicate, poorly drawn face, deformed, mosaic, artifacts, bad limbs

classifier-free guidance scale 7.5, global seed 42, and the stabilityai/stable-diffusion-xl-base-1.0 checkpoint with a DDIM sampler in half precision.

SDXL (Direct). SDXL (Direct) uses the standard SDXL pipeline without any explicit upscaling or refinement. For each target resolution (10242, 20482, 40962) we sample with 30 DDIM steps and η = 1.0.

HiDiffusion. HiDiffusion is evaluated by applying the official HiDiffusion modification to the same SDXL pipeline. For each prompt and target resolution we use 30 DDIM steps, η = 1.0, guidance scale 7.5, and seed 42.

DemoFusion. DemoFusion is instantiated as a DemoFusion–SDXL pipeline using the SDXL base model and the VAE madebyollin/sdxl-vae-fp16-fix. For all resolutions we use 40 sampling steps, guidance scale 7.5, and seed 42. DemoFusion-specific hyperparameters are: view batch size 4, stride 64 pixels, cosine scales (3.0,1.0,1.0), and noise level σ = 0.8.

LSRNA–DemoFusion. For LSRNA–DemoFusion we augment the DemoFusion–SDXL pipeline with the official LSRNA latent super-resolution module (swinir-liif-latentsdxl.pth), keeping the SDXL backbone and DDIM sampler

###### Original Input Generated Caption

###### Generated (1024px) Generated (2048px) Generated (4096px)

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

“a stunning yellow flower in close-up, petals glowing in sunlight with a soft green background, capturing the beauty of nature, shot on a macro lens”

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

“two snow leopards lounging on rocks in a sunny habitat, vibrant greenery behind them, tranquil mood, natural light, shot on 50mm dslr”

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

“elegant grand piano with a glass lid showcasing its intricate woodwork, glinting chrome details, softly lit by studio lights, capturing a modern aesthetic”

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

“close-up of an 8 ball on a green felt pool table, warm indoor lighting, shallow depth, a white ball nearby, capturing the game's mood”

Figure 9. Caption-conditioned multi-resolution generation on the validation set. For each held-out photograph (left), we obtain a structured caption via the procedure in Sec. B.1 (second column) and synthesize SDXL images at 10242, 20482, and 40962 from the same prompt (right). While global semantics are preserved, native high-resolution sampling introduces hallucinations such as object multiplication and exaggerated detail, motivating the 1K+upscaling strategy in Sec. D.2.

unchanged and enabling tiled VAE decoding. For all resolutions we use 50 DDIM steps, guidance scale 7.5, and seed 42. Additional hyperparameters are: view batch size 8, stride ratio 0.5, cosine scales (3.0,1.0,1.0), noise level σ = 0.8, latent standard deviation range [0.0,1.2], and inversion depth 30.

FLUX backbone. For the FLUX experiments we use FLUX.1-Krea-dev as implemented in the public FluxPipeline. For each prompt and target resolution, FLUX latents are generated with 12 denoising steps, guidance scale 4.5, and bfloat16 precision, and decoded with the native FLUX VAE. These latents serve as inputs to LUA for the FLUX-based evaluations in Sec. 4.

Pixel-space super-resolution (SwinIR). For pixel-space super-resolution we start from the official SwinIR checkpoints for ×2 and ×4 SR and fine-tune them on the same high-resolution image collection as LUA using the official training hyperparameters, with bicubically downsampled inputs and a combination of ℓ1 and perceptual losses. At test time, SDXL first generates a 10242 image, which is then either upscaled to 20482 by the ×2 SwinIR model or to 40962 by the ×4 SwinIR model, depending on the target resolution. No additional post-processing is applied.

#### C.2. Shared Evaluation Protocol

To make the comparison as controlled as possible, all methods share the following settings:

- • Prompts. For each validation image, all methods at all resolutions use the same caption-derived prompt from Sec. B together with the fixed negative prompt above.
- • Backbone and guidance. All SDXL-based diffusion methods use the same SDXL checkpoint, DDIM sampler, guidance scale 7.5, and mixed-precision computation; the only differences lie in the high-resolution strategy (direct sampling, HiDiffusion, DemoFusion, LSRNA, or LUA).
- • Randomness. For SDXL-based methods, we fix the seed to 42 for each method and resolution, ensuring deterministic sampling and identical noise realizations across baselines.
- • Runtime measurement. For diffusion and SR pipelines we measure the wall-clock time of a full forward pass per image and report the mean per-image time for each resolution, as in Sec. 4.

### D. Additional Ablations

This section provides an additional analysis that complements the ablation studies in Sec. 4.5, focusing on the statistical differences between latent and pixel super-resolution and how LUA behaves with respect to these distributions.

#### D.1. Latent vs. Pixel Super-resolution Statistics

We analyze whether latent super-resolution is statistically more demanding than pixel-space super-resolution and how far LUA shifts low-resolution statistics toward highresolution ones. For each validation image we compute channel-aggregated means and standard deviations in latent space (per latent channel) and in pixel space (per RGB channel). These per-image statistics are pooled over the validation set, and kernel density estimation (KDE) is used to estimate probability densities for low-resolution (LR) inputs, high-resolution (HR) targets, and LUA outputs. Figure 10 shows the resulting distributions; Table 5 quantifies the LR–HR gap using Wasserstein distance (W1), Jensen–Shannon divergence (JSD), and relative variance gap (∆σ2), capturing global shift, density-shape differences, and second-order mismatch, respectively.

In latent space (top row of Fig. 10), HR latents exhibit rich, multimodal structure with several distinct peaks and shoulders, while LR latents are much flatter, collapsing many modes into broad envelopes and indicating a marked loss of structural specificity at lower resolution. In pixel space (bottom row), LR and HR distributions retain almost identical shapes and differ mainly by small shifts in mean and scale, suggesting that pixel SR operates in a comparatively smooth statistical regime.

Table 5 makes this contrast explicit: the Wasserstein distance between LR and HR distributions is about 40× larger in latent space (0.119) than in pixel space (0.003), with JSD and variance gap following the same trend. Thus, latent SR must reconstruct missing modes of the latent manifold,

[Figure 55]

Figure 10. KDEs of channel-aggregated mean and standard deviation in latent (top) and pixel (bottom) domains for low-resolution inputs (blue), high-resolution targets (orange), and LUA outputs (green). Latent distributions are complex and multimodal, with low-resolution latents collapsing high-resolution modes (red arrows), whereas pixel distributions remain simple and stable.

Table 5. LR–HR domain gap in pixel and latent spaces. We report Wasserstein distance W1, Jensen–Shannon divergence (JSD), and relative variance gap ∆σ2 between low- and high-resolution statistics (Sec. D.1). Latent space exhibits a substantially larger shift across all metrics.

Domain W1 ↓ JSD ↓ Variance gap Pixel space 0.003 0.019 1.1% Latent space 0.119 0.044 6.4%

whereas pixel SR primarily refines an already well-aligned distribution.

LUA’s distributions (green curves in Fig. 10) do not perfectly match the HR ones, which is expected because LR inputs are obtained by downscaling HR generations to enforce identical semantics, rather than arising from a natural generation process. Within this setting, LUA nonetheless shifts LR latents closer to the HR manifold, and in pixel space the remaining discrepancies are further reduced by the VAE decoder, which smooths residual latent errors. Overall, latent SR involves a much larger distributional shift than pixel SR; LUA partially narrows this gap and leverages the VAE to absorb remaining inconsistencies in image space.

#### D.2. Hallucinations in High-resolution Sampling

Diffusion models are known to exhibit visual hallucinations when sampled at very high resolutions, including duplicated structures, malformed anatomy, and physically implausible geometry. We quantify how such artifacts scale with resolution by auditing SDXL as a representative text-to-image backbone.

We generate 100 images at each target resolution 10242, 20482, and 40962 using the caption-derived prompts from Sec. B. Each image is analyzed by a GPT-based vi-

Merged Objects Incorrect Anatomy Merged Limbs

[Figure 56]

[Figure 57]

[Figure 58]

Unnatural Highlights Incorrect Limb Count Awkward Framing

[Figure 59]

[Figure 60]

[Figure 61]

- Figure 11. Representative hallucinations in native high-resolution SDXL sampling. For SDXL generations at 20482 and 40962, we show cropped examples corresponding to the most frequent artifact types identified by the auditor: Merged Objects, Incorrect Anatomy, Merged Limbs, Unnatural Highlights, Incorrect Limb Count, and Awkward Framing. These crops illustrate characteristic failure modes that become increasingly common as the native sampling resolution grows (Table 6).

sion–language auditor in a two-step zero-shot protocol: (i) determine whether any non-trivial hallucination is present; (ii) if so, assign one or more labels from a fixed taxonomy. The auditor operates with the following fine-grained artifact types:

Incorrect Anatomy: Unrealistic body shape or pose. Merged Limbs: Limbs fused or intersecting unnaturally. Merged Objects: Separate objects blending into one. Distorted Face: Strongly warped facial structure. Blurry Background: Background blur beyond intent. Incorrect Limb Count: Too many or too few limbs. Awkward Framing: Unnatural crop or viewpoint. Malformed Hands: Implausible hand or finger shapes. Inconsistent Shadows: Shadows contradict lighting. Unnatural Highlights: Physically implausible highlights. Minor Inconsistencies: Small local visual errors. Impossible Perspective: Geometrically impossible view. Incoherent BG: Background inconsistent with the scene. Cluttered: Overly dense layout hiding the subject. Floating Objects: Objects not supported by anything. Blurry Patches: Local unexpected blur in sharp areas. Over-smooth: Region overly smooth, missing texture.

In this audit, no hallucinations are detected at 10242, whereas 9% of images at 20482 and 19% at 40962 contain at least one artifact. Table 6 reports, for each fine-grained type, how many images at each resolution receive that label; the most frequent failures are Merged Objects, Incorrect Anatomy, and Merged Limbs, followed by Unnatural Highlights and Incorrect Limb Count, with representative

Table 6. Fine-grained hallucination counts vs. sampling resolution. For each artifact type, we report the number of images (out of 100) at each resolution that are flagged by the auditor; multiple labels may be assigned to the same image. The Overall row aggregates the total number of hallucination tags, illustrating how both the frequency and variety of artifacts increase with resolution.

Artifact type 1024px 2048px 4096px Total Merged Objects 0 8 8 16 Incorrect Anatomy 0 6 7 13 Merged Limbs 0 5 6 11 Unnatural Highlights 0 0 8 8 Incorrect Limb Count 0 3 4 7 Awkward Framing 0 2 3 5 Inconsistent Shadows 0 1 2 3 Distorted Face 0 1 1 2 Malformed Hands 0 1 1 2 Minor Inconsistencies 0 0 2 2 Impossible Perspective 0 0 2 2 Incoherent Background 0 0 2 2 Blurry Background 0 1 0 1 Cluttered 0 0 1 1 Floating Objects 0 0 1 1 Blurry Patches 0 0 1 1 Unnatural Smoothness 0 0 1 1 Overall 0 28 50 78

examples in Fig. 11. This pattern is consistent with common training regimes in which diffusion backbones are optimized primarily at moderate resolutions and only weakly exposed to very large image sizes, so native sampling at 2K–4K occurs outside the best calibrated regime and exhibits a higher rate and diversity of hallucinations. These findings support a two-stage design in which images are first generated near 10242 and then upscaled by a dedicated super-resolution module; our latent upscaler adapter follows this strategy, leveraging stable low-resolution generation and performing resolution enhancement in latent space to reduce high-resolution hallucinations relative to direct sampling.

### E. Additional FLUX-based Visualizations

To complement the cross-model experiments in Sec. 4, we provide additional qualitative examples for the FLUX backbone, whose C=16-channel latents offer a rich per-location representation on which LUA performs particularly well. In all cases (Figs. 12–13), FLUX generates a 128×128 latent with 12 denoising steps in 46.57s on a single NVIDIA L40S GPU, LUA upsamples it to 256×256 in 0.63s, and the native FLUX decoder produces the final 20482 image in 3.40s, so latent upscaling adds only about one second of overhead beyond decoding while enhancing spatial detail.

| |
|---|

| |
|---|

|[Figure 66]|
|---|

|[Figure 67]|
|---|

[Figure 68]

[Figure 69]

|[Figure 70]|
|---|

|[Figure 71]|
|---|

[Figure 72]

[Figure 73]

|[Figure 74]|
|---|

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

- Figure 12. FLUX+LUA qualitative examples at 20482 (set 1). For each prompt, FLUX generates a 128×128 latent (12 denoising steps), LUA upsamples it to 256×256, and the native FLUX VAE decodes once to a 2048×2048 image. Insets show ×8 zooms highlighting fine structures (hair, fabric, foliage) preserved by latent upscaling. Timings (generation: 46.57s; upscaling: 0.63s; decoding: 3.40s) are measured on a single NVIDIA L40S GPU.

| |
|---|

| |
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

[Figure 84]

[Figure 85]

|[Figure 86]|
|---|

|[Figure 87]|
|---|

[Figure 88]

[Figure 89]

|[Figure 90]|
|---|

|[Figure 91]|
|---|

[Figure 92]

[Figure 93]

- Figure 13. FLUX+LUA qualitative examples at 20482 (set 2). The pipeline is identical to Fig. 12: FLUX samples a 128×128 latent with 12 denoising steps, LUA performs a single 2× latent upscaling step, and the FLUX decoder produces the final 2048×2048 image. Examples cover diverse content (portraits, indoor scenes, complex textures) and illustrate that the C=16 FLUX latents provide sufficient capacity for LUA to add high-frequency detail with minimal runtime overhead.

