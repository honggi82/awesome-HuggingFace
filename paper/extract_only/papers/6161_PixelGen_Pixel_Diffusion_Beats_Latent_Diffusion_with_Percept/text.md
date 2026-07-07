# arXiv:2602.02493v2[cs.CV]7May2026

## PixelGen: Improving Pixel Diffusion with Perceptual Supervision

Zehong Ma1, Ruihan Xu1, Shiliang Zhang1 1State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University

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

Figure 1: Text-to-image samples from PixelGen, pretrained with only 6 days on 8×H800 GPUs.

### Abstract

Pixel diffusion generates images directly in pixel space, avoiding the VAE artifacts and representational bottlenecks of two-stage latent diffusion. Recent JiT further simplifies pixel diffusion with x-prediction, where the model predicts clean images rather than velocity. However, the standard pixel-wise diffusion loss treats all pixels equally, spending model capacity to perceptually insignificant signals and often leading to blurry samples. We propose PixelGen, an end-to-end pixel diffusion framework that augments x-prediction with perceptual supervision. Specifically, PixelGen introduces two complementary perceptual losses on top of x-prediction: an LPIPS loss for local textures and a P-DINO loss for global semantics. To preserve sample coverage, PixelGen further proposes a noise-gating strategy that applies these losses only at lower-noise timesteps. On ImageNet-256 without classifier-free guidance, PixelGen achieves an FID of 5.11 in 80 training epochs, surpassing the latent diffusion baselines. Moreover, PixelGen scales efficiently to text-to-image generation, reaching a GenEval score of 0.79 with only 6 days of training on 8×H800 GPUs. These results show that perceptual supervision substantially narrows the gap between pixel and latent diffusion while preserving a simple one-stage pipeline. Codes are available at https://github.com/Zehong-Ma/PixelGen.

### 1 Introduction

Diffusion models [1–3] have achieved remarkable success in high-fidelity image generation, offering exceptional quality and diversity. Research in this field generally follows two main directions: latent diffusion and pixel diffusion. Latent diffusion models [4–7] split generation into two stages. As illustrated in Fig. 2(a), a VAE first compresses images into a latent space, and a diffusion model then performs denoising in that space. The performance of latent diffusion is largely constrained by the VAE, where the reconstruction quality limits the upper bound of generation, and the learned latent distribution further affects the convergence of diffusion training [8, 9]. These works have shown that VAEs introduce low-level artifacts and representational bottlenecks for latent diffusion models.

Pixel diffusion models avoid these limitations by modeling raw pixels directly. This end-to-end pipeline removes the need for latent representations and eliminates VAE-induced artifacts. However,

Preprint.

[Figure 11]

[Figure 12]

VAE Decoder

LPIPS Loss

P-DINO Loss

Latent on Loss

| |Diffusion|
|---|---|
|Velo|city|
| | |
|Diffusio|n Model|
| | |

Pixel Diffusion Loss

Perceptual Loss

Image

[Figure 13]

[Figure 14]

Diffusion Model

VAE Encoder

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

(a) Latent Diffusion (b) Pixel Diffusion (PixelGen) (c) FID 50K without CFG

- Figure 2: Perceptual supervision improves pixel diffusion. (a) Latent diffusion is affected by VAE bottlenecks. (b) Standard pixel diffusion relies on uniform pixel-wise supervision, whereas PixelGen introduces perceptual losses to provide more effective guidance. (c) In the no-CFG setting, PixelGen surpasses latent diffusion baselines on ImageNet using only 80 epochs.

it is difficult for the diffusion model to learn a high-dimensional and complex velocity field in pixel space. Recently, JiT [10] has improved pixel diffusion by predicting the clean image, i.e., x-prediction, rather than velocity or noise. Since the prediction target lies directly on the image manifold, x-prediction substantially simplifies optimization and improves generation quality.

Despite this progress, a notable performance gap persists between JiT and latent diffusion. Beyond the prediction target, another key bottleneck lies in the training objective itself. Although x-prediction nominally constrains the output to lie on the image manifold, pixel diffusion still struggles to fit it. The standard pixel-wise loss supervises every pixel uniformly, so model capacity is largely spent on perceptually irrelevant components rather than salient structures. As shown in Fig. 3, this produces blurry samples and a clear spectral gap to real images across all frequency bands. To enable more efficient optimization, pixel diffusion needs perceptual supervision that emphasizes perceptually significant components instead of learning all pixels uniformly.

Based on this insight, we propose PixelGen, a simple and effective training framework that augments x-prediction with perceptual supervision, as illustrated in Fig. 2 (b). Since x-prediction directly estimates the clean image, it offers a natural interface to perceptual encoders such as LPIPS [11] and DINO [12], which are pretrained on clean images and only operate meaningfully in image space. PixelGen introduces three coupled designs. First, an LPIPS-based perceptual loss [11] is applied on the predicted image to enhance local textures and fine details. Second, a DINO-based perceptual loss, termed P-DINO, aligns last-layer features of a DINOv2 encoder to provide global semantic guidance. Third, at high-noise timesteps, the predicted image is still blurry and lacks fine details, so forcing perceptual alignment with the clean image can over-constrain early denoising and reduce sample coverage, as reflected by lower recall [13] and further analyzed in Section 3.4. We thus introduce a simple noise-gating strategy that enables the perceptual losses only at lower-noise timesteps, where the prediction is closer to a clean image. PixelGen requires no latent representations, no VAEs, and no auxiliary stages.

We evaluate PixelGen on both class-to-image and text-to-image generation. PixelGen achieves a leading FID score of 5.11 on ImageNet 256 without classifier-free guidance (CFG) using only 80 epochs, with only a small training overhead. In this no-CFG setting, it surpasses the strong latent diffusion model REPA [14], which achieves an FID of 5.90 with 800 training epochs. With CFG, PixelGen remains competitive, clearly improving over prior pixel diffusion models. Beyond better metrics, the radial power spectrum in Fig. 3(b) shows that perceptual supervision pulls the generated distribution closer to real images, although a residual gap remains. For text-to-image generation, PixelGen is efficiently pretrained from scratch in only 6 days on 8×H800 GPUs and reaches a GenEval score of 0.79.

In summary, our contributions are as follows. i) We propose PixelGen, a simple end-to-end pixel diffusion framework that augments x-prediction with perceptual supervision. ii) We introduce three complementary designs for perceptual supervision: a local LPIPS loss, a global P-DINO loss, and a noise-gating strategy that disables high-noise supervision. iii) PixelGen substantially narrows the gap between pixel and latent diffusion on ImageNet, and reaches GenEval 0.79 on text-to-image generation with only 6 days of training on 8×H800 GPUs.

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

+LPIPS Loss

JiT (Baseline) +LPIPS Loss

+P-DINO Loss FID: 23.67 FID: 10.00 FID: 7.46

(a) Effect of Perceptual Supervision (b) Normalized Radial Frequency

- Figure 3: (a) LPIPS sharpens local textures over blurry samples of JiT, and P-DINO further improves global semantics. (b) PixelGen better matches the real-image radial spectrum on 50,000 samples.

### 2 Related Work

This work is closely related to latent diffusion, pixel diffusion, and perceptual supervision.

Latent Diffusion. Latent diffusion trains diffusion models in a compact latent space learned by a VAE [4]. Compared with raw pixel space, this latent space significantly reduces spatial dimensionality, easing optimization and lowering computational cost [4, 15]. Consequently, VAEs have become a fundamental component of diffusion models [5, 16–23]. However, VAE training often involves adversarial objectives, which complicates the overall pipeline [24]. Poorly trained VAEs can also introduce decoding artifacts [25, 26] and representational bottlenecks. Architecturally, DiT [5] replaced the U-Net [27, 3] used in earlier latent diffusion, and SiT [6] further validated DiT-style backbones with linear flow diffusion. Later works strengthen latent diffusion through representation alignment and joint optimization: REPA [14] and REG [28] align intermediate features to pretrained DINOv2 [12], which is compatible with our framework and used in both our baseline and PixelGen. REPA-E [9] jointly optimizes the VAE and DiT, but a changing latent space may be unstable during the diffusion training, whereas pixel diffusion keeps denoising targets fixed in pixel space. In other directions, VAVAE [8] and RAE [29] improve autoencoders for faster training. DDT [30] proposes decoupled diffusion transformers. MeanFlow [31] and Improved MeanFlow [32] pursue one-step generation with average velocity fields.

Pixel Diffusion. Pixel diffusion avoids latent bottlenecks but has advanced more slowly because pixel space is high dimensional [3, 33–36]. Early methods used multi-resolution diffusion. Relay Diffusion [34] trains separate scale-specific models, while PixelFlow [26] shares one model across scales but needs a complex denoising schedule. Recent work explores alternative designs, including fractal generation [37], transformer-based normalizing flows [38, 39], neural-field velocity prediction [24], self-supervised pretraining [40], and pixel decoders for high-frequency details [41–43]. JiT [10] predicts clean images instead of velocity to learn the low-dimensional image manifold. BiFlow [44] learns bidirectional data-noise mappings. pMF [45] targets one-step pixel diffusion through pixel-space mean flows.

Perceptual Supervision. Perceptual supervision replaces pixel-wise losses with feature-space objectives, emphasizing perceptually meaningful structure over exact RGB matches. It is widely used in autoencoders and GANs to reduce blur and sharpen details. LPIPS [11] is a common choice for improving local textures, while recent self-supervised encoders such as DINOv2 [12] can provide semantic features for global structure. Adversarial losses [46] can improve realism but are unstable and difficult to optimize for pixel diffusion, so we do not use them. Note that latent diffusion’s VAE is also trained with LPIPS and GAN losses [4]. In traditional latent diffusion, some early works have explored perceptual losses. Self-Perceptual loss [47] uses the diffusion model itself as the perceptual encoder, LPL [48] uses VAE’s decoder features and applies the loss at lower-noise timesteps, and Diffusion2GAN [49] combines perceptual and adversarial losses to distill diffusion models into conditional GANs. PixelGen shows that local LPIPS and global P-DINO supervision are complementary. We also find that, unlike LPL’s stable recall across timesteps in latent diffusion, perceptual loss at high-noise timesteps hurts recall in pixel diffusion. It is because predictions are far from clean targets and can be over-constrained by feature matching.

### 3 Methodology

In this section, we introduce PixelGen, a simple and effective training framework that augments x-prediction with perceptual supervision. We first present an overview of PixelGen in Section 3.1. We then introduce two complementary perceptual losses: an LPIPS loss for local textures in Section 3.2 and a P-DINO loss for global semantics in Section 3.3. Finally, we describe a simple noise-gating strategy in Section 3.4.

#### 3.1 Overview

In pixel diffusion, the model output can be parameterized as noise (ϵ), velocity (v), or image (x), corresponding to ϵ-, v-, and x-prediction, respectively. JiT [10] recently simplified the target by replacing the widely used v-prediction with x-prediction, substantially improving pixel diffusion.

However, a clear gap remains between JiT and strong latent diffusion models. We attribute this gap in part to the inefficiency of uniform pixel-wise supervision, which spends capacity on perceptually insignificant signals such as sensor noise and imperceptible details [4] while giving limited emphasis to structures that determine visual quality. As a result, generated samples can still drift from the real image manifold, as shown in Fig. 3. Our key insight is that pixel diffusion should not learn all pixels equally. Instead, it should receive perceptual supervision that prioritizes visually meaningful components and pulls predictions toward the real image manifold. Since x-prediction directly estimates the clean image, it provides a natural interface for perceptual encoders pretrained on clean images. We next introduce image prediction with flow matching, and then present the proposed perceptual supervision.

Image Prediction. Following JiT [10], we adopt image prediction, i.e., x-prediction, to provide a stable target across noise levels. Given a noisy image xt at time t ∈ [0,1], the diffusion transformer netθ predicts the clean image xθ as:

xθ = netθ(xt,t,c), (1) where c denotes conditional information, such as class labels or text embeddings. The noisy input xt is constructed by linearly interpolating between the ground-truth image x and Gaussian noise ϵ ∼ N(0,I):

xt = tx + (1 − t)ϵ. (2)

Velocity Conversion. To retain the sampling advantages of flow matching, we convert the predicted image xθ into a velocity vθ following JiT [10]:

xθ − xt 1 − t

, (3) while the ground-truth velocity v can be represented as:

vθ =

x − xt 1 − t

= x − ϵ. (4) The resulting flow matching objective is:

v =

2

xθ − x 1 − t

LFM = Et,x,ϵ ∥vθ − v∥2 = Et,x,ϵ

. (5) This formulation combines the target of x-prediction with the sampling advantages of flow matching. Perceptual Supervision. Although x-prediction simplifies the training objective, the diffusion loss still supervises pixels uniformly and is therefore misaligned with perceptual quality. PixelGen addresses this by applying two complementary perceptual losses directly to the predicted clean image xθ: LPIPS for local textures and fine details, and P-DINO for global semantics. This image-space supervision is most useful when xθ is sufficiently close to a clean image, because clean-image encoders can over-constrain high-noise predictions. We therefore introduce a simple gate g(t) that enables perceptual losses only in the low-noise regime. Together with the widely used REPA loss [14], which encourages alignment of intermediate representations with a pretrained DINOv2 encoder, the final training objective is:

L = LFM + λ1g(t)LLPIPS + λ2g(t)LP-DINO + LREPA, (6)

where λ1 and λ2 balance the diffusion objective and perceptual supervision. The gate g(t) disables perceptual losses at high-noise timesteps, as detailed in Section 3.4. This end-to-end training enables PixelGen to better fit the real image manifold without VAEs.

Input

[Figure 27]

Velocity Conversion

[Figure 28]

###### -pred

[Figure 29]

[Figure 30]

P-DINO Loss

LPIPS Loss

-pred

DINO VGG

[Figure 31]

###### Perceptual Loss

###### Perceptual Supervision

Image manifold

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Diffusion Model

FM Loss

[Figure 36]

[Figure 37]

Noisy Image Target Image

Predicted Image

off manifold on manifold

(a) Manifold View of Perceptual Supervision (b) Overview of PixelGen

- Figure 4: Overview of PixelGen. (a) v-prediction and pixel-wise x-prediction struggle to fit the image manifold, while perceptual supervision guides x-predictions toward it. (b) PixelGen combines flow matching with LPIPS for local texture and P-DINO for global semantics.

#### 3.2 LPIPS Loss

High-quality image generation requires sharp details and realistic local textures, which are not well captured by pixel-wise losses. To address this, we incorporate the Learned Perceptual Image Patch Similarity (LPIPS) loss [11]. LPIPS measures perceptual similarity by comparing multi-level feature activations extracted from a frozen pretrained VGG network fVGG. The LPIPS loss can be written as:

wl ⊙ fVGGl (xθ) − fVGGl (x) 22 , (7)

LLPIPS =

l

where l indexes VGG layers, and wl denotes the learned per-channel weighting vector for layer l. For simplicity, we omit spatial averaging over feature maps and channel-wise normalization in the formulation.

By minimizing LLPIPS, PixelGen learns perceptually important local patterns instead of matching exact pixel values. As shown in Fig. 3(a), adding LPIPS to the JiT baseline sharpens local textures and fine details. Quantitatively, it reduces FID from 23.67 to 10.00 on ImageNet without classifier-free guidance, indicating that local perceptual supervision is much more effective than uniform pixel-wise matching for correcting over-smoothed predictions.

#### 3.3 P-DINO Loss

While LPIPS provides strong local perceptual supervision, local patterns alone are insufficient for high-fidelity generation. We therefore introduce a Perceptual DINO (P-DINO) loss to provide global semantic guidance.

Specifically, we extract patch-level features using a frozen DINOv2-B [12] encoder fDINO. Let fDINOp (·) denote the feature of patch p. We align the predicted image xθ and the ground-truth image x using cosine similarity:

1 |P| p∈P

1 − cos fDINOp (xθ),fDINOp (x) , (8)

LP-DINO =

where P denotes the set of all patches. The P-DINO loss provides global semantic guidance by aligning high-level representations, encouraging the predicted image to be consistent with the overall scene layout and object semantics. Together with the local LPIPS loss, it enables PixelGen to balance global semantics and local realism, pulling the predicted images closer to the real image manifold.

P-DINO complements LPIPS by improving the global semantics that local texture supervision alone cannot fully capture. As shown in Fig. 3(a), adding P-DINO on top of LPIPS yields clearer object semantics and more coherent global structures, further reducing FID from 10.00 to 7.46. With these two complementary perceptual losses, as illustrated in Fig. 3(b), spectral alignment is consistently improved over the JiT baseline, suggesting that perceptual supervision pulls pixel diffusion closer to the real image manifold.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Figure 5: Class-to-image samples from PixelGen with CFG on ImageNet 256 × 256.

#### 3.4 Noise Gating

Although LPIPS and P-DINO provide useful perceptual supervision, applying them uniformly across all timesteps can be harmful. At high-noise timesteps, the predicted image xθ is still blurry and lacks fine details, so clean-image feature matching can over-constrain early denoising. We provide an empirical analysis of perceptual gradients across timesteps in Appendix A, which suggests that high-noise perceptual gradients are comparatively large relative to the flow matching gradient.

We therefore use a simple timestep gate for perceptual supervision:

g(t) = 1[t ≥ τ], (9)

where t = 0 corresponds to pure noise and t = 1 corresponds to clean data under the interpolation in Section 3.1. We set τ = 0.3, disabling perceptual losses during the first 30% high-noise timesteps and activating them during the later 70% low-noise timesteps. This strategy does not change the diffusion objective or the sampling process. It only limits perceptual supervision when xθ is too noisy.

The ablation in Table 5f shows that applying perceptual losses at high-noise timesteps hurts recall [13], which measures the coverage of real images by generated samples. Noise gating improves recall with only a small trade-off in FID and precision, suggesting that perceptual losses are most useful after the prediction enters a reasonably clean regime. Together with the LPIPS and P-DINO analyses above, this supports the central design of PixelGen, which applies perceptual supervision at low-noise timesteps and relies on the flow matching objective elsewhere.

### 4 Experiments

We evaluate PixelGen on ImageNet 256×256 and GenEval [50]. Section 4.1 compares PixelGen with latent and pixel diffusion baselines under a 200K-step no-CFG setting on ImageNet, and Section 4.4 ablates the main components. Section 4.2 reports class-to-image results on ImageNet with FID [51], Inception Score, precision, and recall [13]. Section 4.3 reports text-to-image results on GenEval.

#### 4.1 Comparison with Baselines

Setup. We first compare PixelGen with latent and pixel diffusion baselines under the same training setting. All models are trained on ImageNet 256 × 256 for 200K training steps with a DiT-L backbone [5]. Following prior work [5, 24, 41], we use a global batch size of 256, AdamW with a constant learning rate of 1 × 10−4, and log-normal timestep sampling. For fair comparison, we apply REPA loss [14] to all models except DiT-L/2 and PixelFlow-L/4 [26]. The DiT patch size is 16, and JiT [10] with REPA loss serves as our baseline. For inference, we use 50 Euler steps without CFG. All experiments are run on one node with 8×H800 GPUs.

Detailed Comparisons. Table 1 shows that PixelGen outperforms both latent and pixel-based diffusion baselines under the same 200K training setting. Compared with the JiT baseline [10], PixelGen reduces FID from 23.67 to 7.53. This result suggests that perceptual losses provide more effective supervision than the standard pixel diffusion loss, guiding predicted images closer to the real image manifold. PixelGen also improves over recent pixel diffusion methods such as PixNerd [24]

- Table 1: ImageNet 256 results after 200K training steps without CFG. The inference adopts 50 Euler steps. REPA loss is used for all models except DiT-L/2 and PixelFlow-L. Latent diffusion models require an extra VAE with 86M parameters. † denotes online VAE encoding.

Training Inference Generation Metrics

Method Params Speed (s/it) Hours Mem (GB) (s/image) FID↓ IS↑ Prec.↑ Rec.↑

|Latent|DiT-L/2 † [5]<br><br>REPA-L/2 [14] DDT-L/2 † [30]<br><br>|458+86M 458+86M 458+86M|0.34 18.9 28.5 0.17 9.7 22.0 0.28 15.5 30.1<br><br>|0.43 0.43 0.43|41.93 36.5 0.52 0.59 16.14 87.3 0.65 0.63 10.00 112.9 0.67 0.65<br><br>|
|---|---|---|---|---|---|
|Pixel|PixNerd-L/16 [24]<br><br>PixelFlow-L/4 [26]<br><br>DeCo-L/16 [41]<br><br>JiT-L/16 (Baseline) [10] PixelGen-L/16 (100K) PixelGen-L/16<br><br>|459M 459M 426M 459M 459M 459M<br><br>|0.20 11.0 29.2<br><br>1.27 70.8 73.2<br><br><br>0.19 10.6 27.5<br><br>0.19 10.6 29.5<br><br>0.20 5.5 33.9<br><br><br>0.20 11.1 33.9<br><br><br>|0.33 4.60 0.32 0.32 0.32 0.32<br><br>|37.49 43.0 0.46 0.62 54.33 24.7 0.43 0.58 31.35 48.4 0.51 0.65 23.67 64.1 0.55 0.63 10.50 106.2 0.70 0.58<br><br>7.53 131.7 0.72 0.60<br><br>|

LatentPixel

and DeCo [41] under the same protocol. Under the 200K-step no-CFG setting, PixelGen is also competitive with strong latent diffusion baselines, achieving an FID of 7.53 compared with 10.00 for DDT-L/2 [30] and 16.14 for REPA-L/2 [14]. Since latent diffusion can also benefit from perceptual supervision [48, 47], we do not claim that pixel diffusion is intrinsically superior. Our goal is instead to show that perceptual losses can be effectively integrated into a pixel diffusion framework, improving one-stage end-to-end generation. The comparison is fair, since latent diffusion’s VAE is also trained with LPIPS and GAN perceptual losses [4].

These gains come with small overhead. As shown in Table 1, perceptual supervision increases per-step time from 0.19s to 0.20s and memory from 29.5GB to 33.9GB over JiT, while keeping the same inference cost of 0.32s/image. PixelGen reaches 10.50 FID in 5.5 hours and further improves to 7.53 FID at 200K steps. Compared with latent diffusion, PixelGen removes VAE training and storage while achieving lower FID and faster inference than DiT-L/2, with 0.32s/image versus 0.43s/image.

#### 4.2 Class-to-Image Generation

Setup. For class-to-image generation on ImageNet, we train PixelGen-XL with 676M parameters at 256 × 256 resolution for 160 epochs. During inference, we follow JiT and use the Heun sampler [52] with 50 steps. When CFG is enabled, we use guidance interval [53].

Results without CFG. Table 2 reports class-to-image performance without CFG. This setting directly tests whether the model captures the underlying image distribution. PixelGen-XL/16 achieves an FID of 5.11 using only 80 training epochs, improving over strong latent diffusion baselines such as REPA-XL/2 with FID 5.90 at 800 epochs and DDT-XL/2 with FID 6.27 at 400 epochs. It also improves over prior pixel diffusion models. For example, DeCo-XL/16 reaches FID 14.88 with 320 epochs, while PixelGen reduces FID by more than 60% using one quarter as many epochs. PixelGen also obtains competitive Inception Score, precision, and recall. These results suggest that, in our no-CFG setting, perceptual supervision helps end-to-end pixel diffusion fit the real image manifold without a separately trained VAE.

Results with CFG. Table 3 reports results with CFG. With 160 training epochs and 50 Heun inference steps, PixelGen achieves an FID of 1.83. It improves over recent pixel diffusion models such as DeCo-XL/16 [41] and JiT-H/16 [10], which use 320 and 600 training epochs. A small gap remains to the leading latent baseline REPA-XL/2, which reaches FID 1.42 at 800 epochs. We attribute this gap partly to the interaction between CFG and perceptual losses in sample coverage. CFG improves precision while reducing recall by moving samples away from low-density regions [54, 53], and our perceptual losses push coverage in the same direction as shown in Table 5f. We leave improved pixel-space samplers and CFG strategies tailored to perceptual supervision for future work.

Qualitative Results. Fig. 5 shows samples generated by PixelGen-XL/16 on ImageNet 256 × 256 with CFG, which exhibit accurate class semantics, sharp textures, and coherent global structures across animals, objects, and natural scenes.

- Table 2: Class-to-image generation performance without CFG on ImageNet 256. PixelGen follows JiT [10] and uses 50 Heun [52] steps, while latent diffusion models use 250 Euler steps.

Method Epochs FID↓ IS↑ Prec.↑ Rec.↑

LDM [4] 170 10.56 103.5 0.71 0.62 DiT-XL/2 [5] 1400 9.62 121.5 0.67 0.67 SiT-XL/2 [6] 1400 8.61 131.7 0.68 0.67 FlowDCN [18] 400 8.36 122.5 0.69 0.65 FasterDiT [22] 400 7.91 131.3 0.67 0.69 DDT-XL/2 [30] 80 6.62 135.2 0.69 0.67 DDT-XL/2 [30] 400 6.27 154.7 0.69 0.67 MDT [21] 1300 6.23 143.0 0.71 0.65 REPA-XL/2 [14] 800 5.90 157.8 0.70 0.69 MaskDiT [23] 1600 5.69 177.9 0.74 0.60

Latent

ADM [3] 400 10.94 - 0.69 0.63 PixelFlow-XL [26] 320 12.23 103.3 0.63 0.66 PixNerd-XL [24] 320 15.61 88.9 0.59 0.68 DeCo-XL/16 [41] 320 14.88 88.2 0.60 0.68 PixelGen-XL/16 80 5.11 159.2 0.72 0.63

Pixel

Table 3: Class-to-image generation performance with CFG on ImageNet 256.

Method Epochs FID↓ IS↑ Prec.↑ Rec.↑

MaskDiT [23] 1600 2.28 276.6 0.80 0.61 DiT-XL/2 [5] 1400 2.27 278.2 0.83 0.57 SiT-XL/2 [6] 1400 2.06 284.0 0.83 0.59 MDT [21] 1300 1.79 283.0 0.81 0.61 REPA-XL/2 [14] 200 1.96 264.0 0.82 0.60 REPA-XL/2 [14] 800 1.42 305.7 0.80 0.64

Latent

ADM [3] 400 4.59 186.7 0.82 0.52

SimpleDiffusion [36] 800 2.44 256.3 - -

RDM [34] 400 1.99 260.4 0.81 0.58 FractalMAR-H 600 6.15 348.9 0.81 0.46 EPG-XL [40] 800 2.04 283.2 0.80 0.61 PixelFlow-XL [26] 320 1.98 282.1 0.81 0.60 DiP-XL/16 [42] 320 1.98 282.9 0.80 0.62 PixNerd-XL [24] 320 1.95 298.0 0.80 0.60 DeCo-XL/16 [41] 320 1.90 303.0 0.80 0.61 JiT-H/16 [10] 600 1.86 303.4 0.78 0.62 PixelGen-XL/16 160 1.83 293.6 0.79 0.63

Pixel

Table 4: Text-to-image generation on GenEval [50] at a 512×512 resolution.

Diffusion GenEval Method Params Sin.Obj. Two.Obj Counting Colors Pos Color.Attr. Overall↑

|PixArt-α [55] SD3 [56] FLUX.1-dev [7] DALL-E 3 [57] OmniGen2 [58]<br><br>|0.6B 8B 12B 4B<br><br>|0.98 0.50 0.44 0.80 0.08 0.07 0.48<br><br>0.98 0.84 0.66 0.74 0.40 0.43 0.68<br>0.99 0.81 0.79 0.74 0.20 0.47 0.67 0.96 0.87 0.47 0.83 0.43 0.45 0.67<br><br><br>1 0.95 0.64 0.88 0.55 0.76 0.80|
|---|---|---|
|PixelFlow-XL/4 [26] PixNerd-XXL/16 [24] PixelGen-XXL/16<br><br>|882M 1.2B 1.1B<br><br>|- - - - - - 0.60 0.97 0.86 0.44 0.83 0.71 0.53 0.73 0.99 0.88 0.59 0.90 0.70 0.70 0.79<br><br>|

#### 4.3 Text-to-Image Generation

Setup. For text-to-image generation, we train on about 36M pretraining images and 60k highquality instruction-tuning samples from BLIP3o [59]. We use Qwen3-1.7B [60] as the text encoder. Following Fluid [61], we train several transformer layers on top of the frozen text features to improve feature alignment [61]. The total batch size is 1536 for 256 × 256 pretraining and 512 for 512 × 512 pretraining. Following previous work [24], we pretrain PixelGen-XXL at 256 × 256 for 200K steps and then at 512×512 for 80K steps. We further fine-tune it on BLIP3o-60k for 40K steps at 512×512. We use gradient clipping for stable training, adopt the Adams-2nd solver with 25 steps as the default sampler, and set the CFG scale to 4.0. The full training takes about 6 days on 8×H800 GPUs.

Main Results. We evaluate PixelGen on text-to-image generation to test its scalability and generalization. Quantitative results on GenEval are reported in Table 4. PixelGen-XXL achieves an overall GenEval score of 0.79, which is competitive with recent large-scale diffusion models such

- as FLUX.1-dev [7] with 0.67 while using fewer parameters and less compute. We do not claim superiority over these systems, since training data, model size, and compute differ substantially. Instead, this result indicates that end-to-end pixel diffusion with perceptual supervision can reach a competitive GenEval score under a much smaller training cost. PixelGen also improves over recent pixel diffusion methods such as PixNerd, suggesting that perceptual supervision is useful for end-to-end pixel diffusion beyond ImageNet class-to-image generation.

Practical efficiency. PixelGen-XXL is pretrained from scratch in only 6 days on 8×H800 GPUs, without an extra VAE, providing a simple framework for future end-to-end pixel diffusion research.

#### 4.4 Ablation Experiments

We study the key components of PixelGen on ImageNet 256×256 under the same setup as Section 4.1. We first ablate each component and then analyze the main hyperparameters.

- Table 5: Ablation experiments of PixelGen on ImageNet 256 × 256. Blue background indicates the default configuration. All models are trained for 200K steps with the same setup as Section 4.1.

(a) Effectiveness of each component.

Method FID↓ IS↑ Prec.↑ Rec.↑

Baseline JiT 23.67 64.13 0.55 0.63 + LPIPS Loss 10.00 113.16 0.70 0.59

+ P-DINO 7.46 137.95 0.73 0.58 + Noise-Gate 7.53 131.70 0.72 0.60

###### (b) Weight λ1 of LPIPS Loss.

Weight FID↓ IS↑ Prec.↑ Rec.↑ 0.05 10.89 106.95 0.68 0.61

0.1 10.00 113.16 0.70 0.59

- 0.5 9.36 122.34 0.71 0.58

- 1.0 10.12 117.75 0.71 0.57

###### (c) Weight λ2 of P-DINO Loss.

Weight FID↓ IS↑ Prec.↑ Rec.↑ 0.005 8.11 128.86 0.72 0.59

- 0.01 7.46 137.95 0.73 0.58

- 0.02 6.84 149.23 0.74 0.57 0.04 6.62 157.78 0.73 0.57

(d) Compatibility with REPA [14].

Method REPA FID↓ IS↑ Prec.↑ Rec.↑

Baseline ✓ 23.67 64.13 0.55 0.63 PixelGen ✓ 7.53 131.70 0.72 0.60 Baseline × 34.85 42.45 0.50 0.61 PixelGen × 11.81 105.30 0.67 0.60

###### (e) Selected DINO layer.

Depth FID↓ IS↑ Prec.↑ Rec.↑

6 12.65 95.92 0.68 0.58 9 10.01 111.51 0.71 0.58

12 7.46 137.95 0.73 0.58 6,9,12 10.01 111.50 0.71 0.58

###### (f) Noise-Gating threshold.

Thres. FID↓ IS↑ Prec.↑ Rec.↑

- 0.0 7.46 137.95 0.73 0.58

- 0.1 7.42 136.95 0.72 0.58 0.3 7.53 131.71 0.72 0.60 0.6 10.72 109.50 0.69 0.60

Effectiveness of each component. Table 5a summarizes the contribution of each component. Starting from JiT, adding LPIPS substantially improves FID and IS, supporting the importance of local perceptual supervision in pixel space. Adding P-DINO further improves global structure and brings additional gains. However, applying perceptual losses at all timesteps reduces recall [13], suggesting a fidelity-coverage trade-off when high-noise predictions are strongly aligned to cleanimage features. We therefore use noise gating to disable perceptual losses during the first 30% high-noise timesteps. This improves recall with only a small trade-off in FID and precision.

Loss weight of LPIPS. We vary the LPIPS weight λ1 in Table 5b. A small weight of 0.05 gives weaker FID, while larger weights of 0.5 slightly reduce recall. We set λ1 = 0.1 as a balanced choice.

Loss weight of P-DINO. We ablate the P-DINO weight λ2 in Table 5c. Increasing λ2 generally improves FID, but larger weights can reduce recall. We set λ2 = 0.01 to balance FID and recall.

Compatibility with REPA. Table 5d studies compatibility with the REPA [14] loss. REPA improves JiT from FID 34.85 to 23.67 and PixelGen from 11.81 to 7.53, while PixelGen without REPA already outperforms JiT+REPA. Thus, perceptual supervision provides an effective objective for pixel diffusion, and it is complementary to REPA.

Selected DINO layer. Table 5e compares DINOv2-B feature depths for the P-DINO loss. Shallow layers at depths 6 and 9 are less effective, likely because they mainly encode low-level appearance. The last layer at depth 12 performs best, suggesting that P-DINO benefits more from high-level semantic features than from low-level appearance cues. Using multiple layers performs worse, likely due to conflicting supervision across feature levels.

Threshold of the Noise-Gating Strategy. Table 5f studies the noise-gating threshold. With threshold 0.0, perceptual losses are applied at all timesteps, including high-noise timesteps where predictions are still far from clean images. This gives strong FID and precision but lower recall. A small threshold of 0.1 has limited effect. Threshold 0.3 gives a better balance by applying perceptual losses only during the last 70% low-noise timesteps, where predicted images are more accurate. A large threshold of 0.6 removes too much supervision and substantially hurts FID and IS.

### 5 Conclusions

We presented PixelGen, a simple end-to-end pixel diffusion framework that augments x-prediction with perceptual supervision through a local LPIPS loss, a global P-DINO loss, and a noise-gating strategy. Without VAEs, PixelGen substantially narrows the gap between pixel and latent diffusion. PixelGen surpasses strong latent baselines on ImageNet in the no-CFG setting, and on text-to-image generation it reaches a GenEval score of 0.79 with only 6 days of training on 8×H800 GPUs.

PixelGen is still inferior to the strongest latent baseline under CFG and relies on pretrained perceptual encoders. Future work includes designing better pixel-space samplers and CFG strategies, and incorporating richer perceptual objectives such as adversarial losses to further improve performance.

### References

- [1] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [2] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, October 2020. URL https://arxiv.org/abs/2010.02502.
- [3] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [5] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [6] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740, 2024.
- [7] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [8] Jingfeng Yao and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. arXiv preprint arXiv:2501.01423, 2025.
- [9] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.
- [10] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise, 2025. URL https://arxiv.org/abs/2511.13720.
- [11] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [12] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [13] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.
- [14] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [15] Junyu Chen, Han Cai, Junsong Chen, Enze Xie, Shang Yang, Haotian Tang, Muyang Li, Yao Lu, and Song Han. Deep compression autoencoder for efficient high-resolution diffusion models. arXiv preprint arXiv:2410.10733, 2024.
- [16] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24174–24184, 2024.
- [17] Xiaoyu Yue, Zidong Wang, Zeyu Lu, Shuyang Sun, Meng Wei, Wanli Ouyang, Lei Bai, and Luping Zhou. Diffusion models need visual priors for image generation. arXiv preprint arXiv:2410.08531, 2024.
- [18] Shuai Wang, Zexian Li, Tianhui Song, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Exploring dcn-like architecture for fast image generation with arbitrary resolution. Advances in Neural Information Processing Systems, 37:87959–87977, 2024.

- [19] Yao Teng, Yue Wu, Han Shi, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Dim: Diffusion mamba for efficient high-resolution image synthesis. arXiv preprint arXiv:2405.14224, 2024.
- [20] Tianhui Song, Weixin Feng, Shuai Wang, Xubin Li, Tiezheng Ge, Bo Zheng, and Limin Wang. Dmm: Building a versatile image generation model via distillation-based model merging. arXiv preprint arXiv:2504.12364, 2025.
- [21] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23164–23173, 2023.
- [22] Jingfeng Yao, Cheng Wang, Wenyu Liu, and Xinggang Wang. Fasterdit: Towards faster diffusion transformers training without architecture modification. Advances in Neural Information Processing Systems, 37:56166–56189, 2024.
- [23] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Masked diffusion transformer is a strong image synthesizer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23164–23173, 2023.
- [24] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025.
- [25] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.
- [26] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025.
- [27] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.
- [28] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025.
- [29] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [30] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.
- [31] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=uWj4s7rMnR.
- [32] Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J Zico Kolter, and Kaiming He. Improved mean flows: On the challenges of fastforward generative models. arXiv preprint arXiv:2512.02012, 2025.
- [33] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. Advances in Neural Information Processing Systems, 36:65484–65516, 2023.
- [34] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023.
- [35] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315–11325, 2022.

- [36] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–

13232. PMLR, 2023.

- [37] Tianhong Li, Qinyi Sun, Lijie Fan, and Kaiming He. Fractal generative models. arXiv preprint arXiv:2502.17437, 2025.
- [38] Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, David Berthelot, Jiatao Gu, Huangjie Zheng, Tianrong Chen, Miguel Angel Bautista, Navdeep Jaitly, and Josh Susskind. Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329, 2024.
- [39] Guangting Zheng, Qinyu Zhao, Tao Yang, Fei Xiao, Zhijie Lin, Jie Wu, Jiajun Deng, Yanyong Zhang, and Rui Zhu. Farmer: Flow autoregressive transformer over pixels. arXiv preprint arXiv:2510.23588, 2025.
- [40] Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. There is no vae: End-to-end pixel-space generative modeling via self-supervised pretraining. arXiv preprint arXiv:2510.12586, 2025.
- [41] Zehong Ma, Longhui Wei, Shuai Wang, Shiliang Zhang, and Qi Tian. Deco: Frequencydecoupled pixel diffusion for end-to-end image generation. arXiv preprint arXiv:2511.19365, 2025.
- [42] Zhennan Chen, Junwei Zhu, Xu Chen, Jiangning Zhang, Xiaobin Hu, Hanzhen Zhao, Chengjie Wang, Jian Yang, and Ying Tai. Dip: Taming diffusion models in pixel space. arXiv preprint arXiv:2511.18822, 2025.
- [43] Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, and Jiebo Luo. Pixeldit: Pixel diffusion transformers for image generation. arXiv preprint arXiv:2511.20645, 2025.
- [44] Yiyang Lu, Qiao Sun, Xianbang Wang, Zhicheng Jiang, Hanhong Zhao, and Kaiming He. Bidirectional normalizing flow: From data to noise and back. 2025. URL https://arxiv. org/abs/2512.10953.
- [45] Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, and Kaiming He. One-step latent-free image generation with pixel mean flows. arXiv preprint arXiv:2601.22158, 2026.
- [46] Axel Sauer, Katja Schwarz, and Andreas Geiger. Stylegan-xl: Scaling stylegan to large diverse datasets. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.
- [47] Shanchuan Lin and Xiao Yang. Diffusion model with perceptual loss. 2025. URL https: //arxiv.org/abs/2401.00110.
- [48] Tariq Berrada, Pietro Astolfi, Melissa Hall, Marton Havasi, Yohann Benchetrit, Adriana RomeroSoriano, Karteek Alahari, Michal Drozdzal, and Jakob Verbeek. Boosting latent diffusion with perceptual objectives. In The Thirteenth International Conference on Learning Representations,

2025. URL https://openreview.net/forum?id=y4DtzADzd1.

- [49] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling Diffusion Models into Conditional GANs. In European Conference on Computer Vision (ECCV), 2024.
- [50] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [51] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [52] Karl Heun et al. Neue methoden zur approximativen integration der differentialgleichungen einer unabhängigen veränderlichen. Z. Math. Phys, 45:23–38, 1900.

- [53] Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. arXiv preprint arXiv:2404.07724, 2024.
- [54] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [55] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.
- [56] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.
- [57] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra, Prafulla Dhariwal, Casey Chu, Yunxin Jiao, and Aditya Ramesh. Improving image generation with better captions. OpenAI Technical Report, 2023. URL https://cdn.openai.com/papers/dall-e-3.pdf.
- [58] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.
- [59] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [60] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [61] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.
- [62] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [63] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [64] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [65] Zidong Wang, Lei Bai, Xiangyu Yue, Wanli Ouyang, and Yiyuan Zhang. Native-resolution image synthesis. arXiv preprint arXiv:2506.03131, 2025.
- [66] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025.
- [67] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.
- [68] Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.

### A Empirical Analysis of Noise Gating

This section provides additional empirical observations that support the noise-gating design in Section 3.4. Following the convention in Section 3.1, t = 0 corresponds to pure noise and t = 1 to clean data, so small t denotes high-noise timesteps. The statistics in Fig. 6(b,c) are computed on the pretrained JiT baseline in Table 1 with 50,000 ImageNet samples. Concretely, we sweep t over a uniform grid in [0,1]. For each t, we sample a clean image x and Gaussian noise ϵ ∼ N(0,I), build the noisy input xt = tx + (1 − t)ϵ, and obtain the prediction xθ = netθ(xt,t,c). We then compute the gradient of each loss with respect to xθ and report its ℓ2 norm averaged over the 50,000 samples.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

t=0.1 t=0.2

[Figure 55]

[Figure 56]

Ground Truth

t=0.3 t=0.5

X Predictions

(a) Predicted Image (b) Grad Norm (c) Grad Norm Ratio

- Figure 6: Empirical analysis of perceptual gradients across timesteps. (a) Predicted clean images xθ at different timesteps. At high-noise timesteps (e.g., t = 0.1,0.2), the predictions are blurry and lack fine details. (b) Mean gradient norm of the perceptual losses on xθ across different t. (c) Ratio between the gradient norm of the perceptual losses and that of the flow matching loss.

Predicted images at high-noise timesteps. Fig. 6(a) visualizes xθ across timesteps. At high-noise timesteps such as t = 0.1 and t = 0.2, the predicted images are noticeably blurry and miss fine details. Directly aligning such predictions with the clean ground-truth image in a perceptual feature space requires the model to match details that are difficult to recover from highly noisy inputs, which may push samples away from the natural image distribution and reduce sample coverage.

Magnitude of perceptual gradients. Fig. 6(b) reports the mean gradient norm of the perceptual losses on xθ as a function of t, computed as described above. The gradient norm is noticeably larger at high-noise timesteps (e.g., t < 0.3) than at low-noise timesteps. Combined with Fig. 6(a), this suggests that the comparatively large perceptual gradients at high-noise timesteps are partly driven by the large gap between the blurry xθ and the clean ground-truth, and may therefore be less reliable than at low-noise timesteps.

Relative scale to the flow matching gradient. Fig. 6(c) plots the ratio between the gradient norm of the perceptual losses and that of the flow matching loss on xθ. The ratio is also larger at high-noise timesteps, indicating that the perceptual term tends to contribute a relatively larger share of the overall gradient there. Since xθ is still far from a clean image in this regime, such gradients may over-emphasize clean-image feature matching during early denoising. It may reduce the sample coverage of generated images.

### B More Implementation Details

#### B.1 Baseline Comparisons

In this subsection, we summarize the settings used for all baseline comparisons. In the baseline comparisons, all diffusion models are trained on ImageNet at 256×256 resolution for 200k iterations using a large DiT variant. Following previous works [5, 24], we use a global batch size of 256 and the AdamW optimizer with a constant learning rate of 1e-4. Both baseline and PixelGen adopt SwiGLU [62, 63], RoPE2d [64], and RMSNorm, and are trained with lognorm sampling and REPA [14]. The patch size of DiT’s input is set to 16 for both baseline and our PixelGen. The only modification on the baseline is to add two complementary perceptual losses. For inference, we use 50 Euler steps without classifier-free guidance [54] (CFG) for all models except PixelFlow [26], which requires 100 steps. The timeshift is set to 1.0 for all experiments in Table 1.

Table 6: Configurations of Experiments.

PixelGen-L PixelGen-XL PixelGen-XXL

architecture DiT depth 22 28 16 hidden dim 1024 1152 1536 heads 16 16 24 params 459M 676M 1.1B bottleneck dim 128 - 256 dropout 0.0 0.1 0.0 image size 256 256 512 patch size 16 training optimizer AdamW, β1, β2 = 0.9, 0.999 batch size 256 256 1536/512 learning rate 1e-4 lr schedule constant weight decay 0 ema decay 0.9999 time sampler logit(t)∼N(µ, σ2), µ = -0.8, σ = 0.8 noise scale 1.0 clip of (1-t) 0.05 sampling ODE solver Euler Heun Adams-2nd ODE steps 50 50 25 time steps linear in [0, 1.0] timeshift 1.0 2.0 3.0 CFG scale - 2.25 4.0 CFG interval [0.1, 0.9] (if used)

#### B.2 Class-to-Image Generation

This subsection describes additional implementation details for class-to-image generation. The batch size and learning rate follow the default settings previously described. We use a global batch size of 256 and the AdamW optimizer with a constant learning rate of 1e-4. The time sampler uses logit-normal distribution over t: logit(t)∼N(−0.8,0.82), which aligns with JiT [10]. We train the PixelGen-XL for 160 epochs, and use an autograd operation to balance gradients between flowmatching loss and perceptual losses after 80 epochs. We set the CFG scale to 2.25. The guidance interval [53] is set to (0.1, 0.9). For evaluation, we use a Heun sampler with 50 inference steps following JiT [10]. The timeshift is set to 2.0 to match the time sampler.

#### B.3 Text-to-Image Generation

We adopt Qwen3-1.7B [60] as the text encoder. To improve the alignment of frozen text features [61], we jointly train several transformer layers on the frozen text features similar to Fluid [61]. The total batch size is 1536 for 256 × 256 resolution pretraining and 512 for 512 × 512 resolution pretraining. Following PixNerd [24], we pretrain PixelGen on 256 × 256 resolution for 200K steps and pretrain on 512 × 512 resolution for 80K steps. We further fine-tune the pretrained PixelGen on BLIP3o-60k with 40k steps at the 512 × 512 resolution following PixNerd. We adopt gradient clipping to stabilize training. The whole training only takes about 6 days on 8× H800 GPUs. We use the Adams-2nd solver with 25 steps as the default choice for sampling. The CFG scale is set to 4.0. We leave the native resolution [65] or native aspect training [66–68] as future work.

#### B.4 Experiment Configurations

- Table 6 summarizes the experiment configurations for PixelGen-L/16, PixelGen-XL/16, and PixelGenXXL/16. In practice, we follow the training setups from previous works such as DiT [5], SiT [6], and PixNerd [24].

- Algorithm 1 Training step

# netθ: DiT network # x: training batch # c: class label or textual prompt

t = sample_t() ϵ = randn_like(x)

xt = t * x + (1 − t) * ϵ v = (x - xt) / (1 - t)

xθ = netθ(xt, t, c) vθ = (xθ - xt) / (1 - t) gt = (t ≥ τ)

lossFM = l2_loss(vθ - v) lossLPIPS = LPIPS(xθ, x) lossP-DINO = P-DINO(xθ, x)

loss = lossFM + gt * (λ1lossLPIPS + λ2lossP-DINO) + lossREPA

- Algorithm 2 Sampling step

# xt: current samples at t

xθ = netθ(xt, t, c) vθ = (xθ - xt) / (1 - t)

xt_next = xt + (t_next - t) * vθ

### C Text-to-Image Prompts

Below, we list the prompts used for text-to-image generation in Fig. 1. These prompts cover a mix of animals, people, and scenes to evaluate semantic understanding and visual detail generation.

- • A baby cat stands on two legs, wearing a chothes.
- • A kungfu panda is wielding a sword in realistic style.
- • A fox sleeping inside a large transparent lightbulb.
- • An extremely happy American Cocker Spaniel is smiling and looking up at the camera with his head tilted to one side.
- • A man sipping coffee on a sunny balcony filled with potted plants, wearing linen clothes and sunglasses, basking in the morning light.
- • A beautiful girl with hair flowing like a cascading waterfall.
- • Dreamlike portrait with soft neon glow and painterly textures.
- • A raccoon wearing a detective’s hat, observing something with a magnifying glass.
- • Close-up of an aged man with weathered features and sharp blue eyes peering wisely from beneath a tweed flat cap.

### D Pseudocode for PixelGen

In Algorithm 1, we provide the pseudocode for the training step of PixelGen. PixelGen follows the pipeline of JiT [10] and additionally introduces two complementary perceptual losses on the predicted image xθ. A REPA [14] loss is used in both our Baseline and PixelGen. Algorithm 2 provides the pseudocode for the sampling steps.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

蒸汽朋克风格的长城，巨大空 中飞艇，黄昏史诗光影 (Steampunk Great Wall with huge airships, epic dusk light)

破碎的青花瓷风少女面孔，细 腻纹理，超现实主义 (Shattered blue-and-white porcelain girl’s face, fine texture, surreal)

赛博朋克大熊猫，佩戴机械义 肢，在霓虹街道漫步 (Cyberpunk panda with mechanical prosthetics walking through neon streets)

透明冰晶材质的龙，盘踞雪山 之巅 (A transparent ice-crystal dragon coiled atop a snowy peak)

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Minimalist interior design, natural light, modern architectural style

Cute kitten wearing astronaut helmet, floating in space

Humanoid robot standing in rain with a city background

A cyberpunk woman with glowing tattoos and a mechanical arm

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Old fisherman holding a

Viking warrior wearing a red

A beautiful girl with hair flowing like a cascading waterfall.

Portrait of a woman made entirely of porcelain and gold.

galaxy inside a glass jar.

sweater

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

Pixel art sunset over a retro synthwave city.

A highway leading straight into a giant purple moon with a person.

A lovely horse stands in the bedroom.

A lion made of burning charcoal and glowing embers.

- Figure 7: More qualitative results of text-to-image generation at a 512×512 resolution. Our PixelGen supports multiple languages with the Qwen3 text encoder, such as Chinese and English.

### E More Visualizations

In this section, we provide more visualizations, including text-to-image generation in Fig. 7, and classto-image generation at a 256×256 resolution in Fig. 8. Our PixelGen supports multiple languages with the Qwen3 text encoder after pretraining on the BLIP3o dataset [59], such as Chinese and English.

[Figure 73]

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

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

##### Figure 8: More qualitative results of class-to-image generation at a 256×256 resolution with CFG. The CFG scale is set to 3.0.

### NeurIPS Paper Checklist

- 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes].

Justification: The abstract and Section 1 state the main contributions, including perceptual supervision for pixel diffusion, the method details, and the ImageNet and GenEval results. The experimental sections report the corresponding quantitative results.

Guidelines:

- • The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

- 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes].

Justification: The conclusion includes a limitations and future work paragraph noting the remaining gap to the strongest latent baseline under CFG and the reliance on frozen perceptual encoders. Section 4.2 also discusses the CFG-related performance gap and leaves improved samplers and CFG strategies for future work.

Guidelines:

- • The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate “Limitations” section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

- 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]. Justification: The paper is empirical and methodological, and does not introduce formal theoretical results or theorem/proof statements. Guidelines:

- • The answer [N/A] means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes].

Justification: Sections 3 and 4.1 to 4.3 describe the model objectives, losses, datasets, optimization settings, sampling procedures, and evaluation metrics. The appendix further provides baseline settings, class-to-image and text-to-image training details, experiment configurations, and pseudocode for training and inference.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes].

Justification: We use open-access evaluation benchmarks such as ImageNet and GenEval, and provide the text-to-image prompts used for qualitative experiments in the appendix. We are organizing our code and reproduction instructions, and plan to open-source both soon.

Guidelines:

- • The answer [N/A] means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes].

Justification: The experiments section specifies the datasets, metrics, backbone sizes, optimizers, batch sizes, learning rates, training durations, CFG settings, samplers, and inference steps. Additional hyperparameters and model configurations are summarized in the appendix.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No].

Justification: The paper reports standard generation metrics such as FID, Inception Score, precision/recall, and GenEval, but does not include error bars or statistical significance tests. Multiple full training runs would be computationally expensive at the reported model and dataset scales.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- • If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes].

Justification: Table 1 reports training speed, wall-clock hours, memory, and inference time for the baseline comparison, and Sections 4.1 and 4.3 report the use of 8×H800 GPUs and the 6-day text-to-image training cost. The appendix also lists the relevant training and sampling configurations.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes].

Justification: To the best of our knowledge, the research conforms to the NeurIPS Code of Ethics. The work uses standard image-generation benchmarks and publicly described datasets/models, and does not involve human-subject experiments.

Guidelines:

- • The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

- • If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes].

Justification: The paper discusses that PixelGen can efficiently train a text-to-image model in only 6 days on 8×H800 GPUs, which may make high-quality generative modeling more accessible and computationally efficient.

Guidelines:

- • The answer [N/A] means that there is no societal impact of the work performed.
- • If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.
- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]. Justification: We do not release any data or models. We do not believe that the algorithm for pixel diffusion has a high risk for misuse Guidelines:

- • The answer [N/A] means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]. Justification: All existing models and datasets used in this work are properly credited with appropriate references, and their licenses and terms of use have been fully respected. Guidelines:

- • The answer [N/A] means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]. Justification: The paper does not currently introduce or release a new dataset, codebase, or model checkpoint as a new asset. Guidelines:

- • The answer [N/A] means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]. Justification: The paper does not involve crowdsourcing experiments or research with human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]. Justification: The paper does not involve crowdsourcing or human-subject research, so IRB approval or equivalent review is not applicable. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes].

Justification: The text-to-image setup in Section 4.3 and the appendix explicitly state that Qwen3-1.7B is used as the text encoder, with additional transformer layers trained on frozen text features. This LLM component is part of the text-to-image system rather than merely a writing or formatting aid.

Guidelines:

- • The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.

