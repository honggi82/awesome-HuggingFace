arXiv:2504.02154v3[cs.CV]29May2025

# FreSca: Scaling in Frequency Space Enhances Diffusion Models

### Chao Huang1, Susan Liang1, Yunlong Tang1, Jing Bi1, Li Ma2, Yapeng Tian3, Chenliang Xu1 1University of Rochester, 2HKUST, 3The University of Texas at Dallas

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

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Figure 1: FreSca: A plug-and-play enhancement for diffusion models. Without retraining, FreSca refines Marigold [1] depth predictions to recover fine details (top); enables precise, promptaligned generation over SD3 [2] (middle) ; and boosts motion, detail, and temporal consistency in VideoCrafter2 [3] video generation (bottom) .

## Abstract

Latent diffusion models (LDMs) have achieved remarkable success in a variety of image tasks, yet achieving fine-grained, disentangled control over global structures versus fine details remains challenging. This paper explores frequency-based control within latent diffusion models. We first systematically analyze frequency characteristics across pixel space, VAE latent space, and internal LDM representations. This reveals that the “noise difference” term, ∆ϵt, derived from classifier-free guidance at each step t, is a uniquely effective and semantically rich target for manipulation. Building on this insight, we introduce FreSca, a novel and plug-andplay framework that decomposes ∆ϵt into low- and high-frequency components and applies independent scaling factors to them via spatial or energy-based cutoffs. Essentially, FreSca operates without any model retraining or architectural change, offering model- and task-agnostic control. We demonstrate its versatility and effectiveness in improving generation quality and structural emphasis on multiple architectures (e.g., SD3, SDXL) and across applications including image generation, editing, depth estimation, and video synthesis, thereby unlocking a new dimension of expressive control within LDMs.

Preprint. Under review.

## 1 Introduction

Latent diffusion models (LDMs) [4] have emerged as a dominant force in generative modeling, capable of producing images of unprecedented quality and diversity from textual prompts [5, 2, 6, 7] or other conditioning signals [8]. Despite their power, achieving nuanced control beyond the initial conditioning remains an active area of research. Users often desire to modulate specific image characteristics, such as the prominence of fine textures versus coarse shapes, or to impart particular artistic styles, in a more direct and disentangled manner. Existing control mechanisms might involve complex model modifications, additional training, or offer only coarse-grained adjustments.

The frequency domain offers a natural and powerful paradigm for image manipulation [9], where low frequencies typically represent global structures and smooth variations, while high frequencies encode fine details such as edges and textures. This fundamental separation has been exploited in classical image processing for tasks like sharpening [10], denoising [11], and style transfer [12]. We hypothesize that by extending frequency-domain manipulations to the internal workings of LDMs, we can unlock more intuitive and fine-grained control over the synthesis process. However, the iterative nature of diffusion and its operation within a learned noisy latent space raise critical questions: How do frequency characteristics translate from pixel space to the VAE latent space? And, more importantly, which specific component or stage within the diffusion model’s denoising trajectory is most amenable and effective for frequency-based interventions?

In this paper, we systematically investigate these questions. We begin by comparing frequency decompositions in pixel space versus the VAE latent space (as shown in Fig. 2), highlighting differences in semantic content and sensitivity. Grounded by the observations on the VAE latent space, we then explore various internal representations within the diffusion model, including the noisy latents xt, the noise prediction ϵt, and the crucial “noise difference” term ∆ϵt arising from classifierfree guidance (CFG) [13]. Interestingly, our analysis reveals that ∆ϵt is particularly rich in semantic information among others, and thereby can serve as an ideal target for frequency manipulation.

Based on these insights, we propose FreSca, a versatile framework that operates by decomposing the noise prediction into its low- and high-frequency components at each step of the denoising process. FreSca then applies distinct scaling factors to these components, allowing for independent amplification or suppression of global structures and fine details. To further enhance adaptability, FreSca supports both spatial- and energy-based frequency cutoffs for band separation. As FreSca operates directly in the common noise space used by nearly all diffusion models, it is inherently modeland task-agnostic, avoiding the architectural constraints of prior frequency-aware methods [14, 15]. We validate this versatility across a variety of models (e.g., SDXL [5], SD3 [2]) and tasks such as diffusion-based depth estimation [1, 16], image generation [2, 5], image editing [17, 18], and video synthesis [3].

In summary, our contributions to the community are

- 1. A comparative analysis of frequency representations in pixel space, VAE latent space, and key internal states of latent diffusion models.
- 2. The identification of the CFG-derived noise-difference term, ∆ϵt, as a highly effective and semantically meaningful target for frequency-based manipulation in LDMs.
- 3. The FreSca framework, a plug-and-play method providing disentangled control over low- and high-frequency image characteristics without model retraining or architectural changes. We demonstrate its efficacy through qualitative and quantitative experiments on diverse tasks and models, highlighting its ability to produce varied stylistic effects and modulate detail levels.

## 2 Related Works

Controls in Diffusion Models. The quest for greater control over diffusion model outputs has spurred various approaches. Prompt engineering [19] is the most direct method but often lacks fine-grained control over specific visual attributes. Classifier-Free Guidance [13] significantly improved sample quality and adherence to prompts by amplifying the guidance signal. Beyond prompt-based control, structural guidance methods like ControlNet [8] and T2I-Adapters [20] enable conditioning on spatial inputs like edge maps or pose, typically by introducing trainable modules or fine-tuning parts of the UNet. Other approaches focus on adapting pre-trained models using lightweight finetuning techniques

[Figure 42]

[Figure 43]

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

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Figure 2: (a) Frequency decomposition of an RGB image (Il,Ih) and its SD3 [2]/SDXL [5] VAE encodings (xl,xh) with r0 = 0.05 (pixel) and r0 = 0.5 (latent). (b) Cutoff-radius sensitivity in pixel vs. latent space.

such as LoRA [21] for domain-specific generation. While powerful, many of these methods may require auxiliary networks [8, 20], per-instance optimization [22–24], or are not primarily focused on disentangled frequency control. In contrast, FreSca differs by offering a zero-shot, plug-and-play mechanism that directly targets frequency bands during the denoising process of diffusion models.

Frequency and Spectral Methods in Diffusion Models. Frequency and spectral analyses have long illuminated deep models’ behavior, from CNNs’ spectral bias [25] to distribution discrepancies in GANs [26]. Yet, despite these insights and analogous explorations in neural networks, explicit frequency-domain control within diffusion processes remains nascent. A handful of recent works have sought to manipulate spectral components, e.g., tuning the frequency behavior of U-Net skip connections and backbone features [14], applying filters to noisy latents for artistic effects [27, 28], and modulating frequency content in temporal attention maps [15]. However, these approaches tend to be model- or task-specific and do not generalize across diffusion variants. In contrast, FreSca offers a unified, model- and task-agnostic framework to decompose and dynamically scale the classifier-free guidance noise difference by frequency, providing direct, interpretable control over both global structure and fine detail.

- 3 Method

In this section, we begin by analyzing the differences between frequency decomposition in pixel space versus the latent space of Variational Autoencoders (VAEs). Subsequently, we investigate frequency decomposition applied to various intermediate representations within diffusion models to pinpoint an effective basis for frequency manipulation. We then examine the denoising trajectory, observing the step-wise dynamics of different frequency bands. Building on these insights, we introduce FreSca, a novel framework for unified frequency scaling in latent diffusion models.

Preliminaries. Latent diffusion models (LDMs) operate by first encoding images into a latent space using a VAE, and then performing the diffusion process within this space. An LDM typically consists of: (i) a VAE with an encoder E and a decoder D. Given an RGB image I, the encoder maps it to an initial latent representation x = E(I). The decoder reconstructs the image from a latent code

as Iˆ = D(x). (ii) A time-conditional denoising network ϵθ that operates in the latent space. The diffusion model involves a forward noising process and a reverse denoising process over T timesteps.

Starting with an initial latent x0, the forward process corrupts it into a squence of noisy latent {x}Tt=1 by gradually adding Gaussian noise according to predefined schedule (see, e.g., [29]). At each time

step t, the denoising network ϵθ(xt,t) is trained to predict the added noise ϵt, enabling a reverse denoising process that recovers x0 from pure noise. In what follows, xt denotes the latent at timestep t, and ϵθ the noise predictor–our primary handle for frequency-based control.

- Table 1: Experiment configuration: Frequency operations (Eqs. (1) to (3) ) applied across different feature spaces. Operation Pixel VAE Diffusion Model Space

Noisy Latents Combined Noise Noise Difference Eqs. (1) to (3) I x x1:T = {xt}Tt=1 ϵ1:T = {ϵt}Tt=1 ∆ϵ1:T = {∆ϵt}Tt=1

### 3.1 Frequency Decomposition in Pixel vs. Latent Space

Frequency decomposition is a cornerstone of image processing, enabling insights into both classical algorithms and modern neural networks. Typically, an image can be separated into low-frequency components, capturing global structures and smooth variations, and high-frequency components, encoding fine details like edges and textures.

While this concept is well-established in pixel space, its extension to the latent representations learned by VAEs (and subsequently used by LDMs) requires investigation. To this end, we define a unified frequency decomposition operator. Given an input signal u ∈ {I (RGB image), x (VAE latent)}, we compute its channel-wise 2D Fourier transform:

U = F(u), u = F−1(U). (1)

Let the spatial dimensions of u be H × W. We define a cutoff ratio r0 ∈ [0,1], which the actual cutoff radius Rc in the frequency domain is then Rc = r0 · min(H/2,W/2). This ensures the ratio r0 has comparable effect across different spatial resolutions. We then define binary low-pass Ml and high-pass Mh masks over frequency coordinates (kx,ky):

Ml(kx,ky) =

1, if kx2 + ky2 ≤ Rc, 0, otherwise,

Mh(kx,ky) = 1 − Ml(kx,ky). (2)

The low- and high-frequency components of u are then obtained by applying these masks in the Fourier domain:

ul = F−1 Ml ⊙ U , uh = F−1 Mh ⊙ U , (3) with ⊙ denoting element-wise multiplication.

By applying this decomposition to both the pixel image I and its VAE encoding x, we obtain pairs (Il,Ih) and (xl,xh). Visual comparisons (see Fig. 2(a)) indicate that in both domains, low frequencies correspond to coarse structures and high frequencies to details. However, we identify two key distinctions: (i) Semantic richness in latent high frequencies. The high-frequency components of x tend to preserve more abstract semantic patterns, such as object contours and characteristic textures. This reflects the VAE’s ability to learn meaningful representations. (ii) Threshold sensitivity. Pixel-space details (edges, textures) diminish rapidly as r0 increases (e.g., beyond 0.1). In contrast, VAE latent features often reveal significant structural and textural information even at higher r0 value (see Fig. 2(b)). These observations highlight both the conceptual alignment and the practical differences in frequency content between pixel and VAE latent spaces.

### 3.2 Frequency Decomposition for Diffusion Models

Having analyzed frequency characteristics in the VAE latent space, we now investigate where frequency-specific manipulations can be most effectively applied during the iterative denoising process of LDMs. For conditional generation (e.g., text-to-image), LDMs typically employ ClassifierFree Guidance [13]. The effective noise prediction ϵt at timestep t is:

ϵt = ϵθ(xt,t) + ω · ∆ϵt, ∆ϵt = ϵθ(xt,c,t) − ϵθ(xt,t). (4)

Here ϵθ(xt,c,t) and ϵθ(xt,t) denote the conditional and unconditional noise estimates, and ω is the classifier-free guidance scale.

We consider three primary candidate representations within the diffusion process for applying frequency decomposition (using Eqs. (1) to (3)), as outlined in Tab. 1. To determine the most suitable

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

- Figure 3: (a) SDXL outputs (left) and results of frequency decomposition on various diffusion representations (right); top: high-frequency components, bottom: low-frequency components; cutoff

r0 = 0.5. (b) Temporal average over T steps for each representation, highlighting the semantic richness of the noise-difference term.

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 4: Relative log amplitudes of Fourier over all T denoising steps for (a) the latent variables

xt, (b) the noise prediction ϵt, and (c) the noise-difference term ∆ϵt. Each curve corresponds to a timestep, illustrating how low and high frequencies changes in each representation.

candidate, we apply either a low-pass or a high-pass filter (using a fixed r0 = 0.5) to the chosen representation at each denoising step t. The final generated image D(x0) allows us to assess the impact. Our experiments (visualized in Fig. 3(a)) reveal that manipulating the frequency components of the noise difference term ∆ϵ1:T yields the most semantically meaningful and controllable results. For instance, removing high-frequency components from ∆ϵ1:T results in minimal degradation to the overall image structure, while selectively preserving only its high-frequency components can produce interesting stylization effects, capturing low-level textures of patterns like “dragon,” “cloud,” and “mountain.”

We hypothesize that ∆ϵ1:T inherently encodes crucial semantic structures. To support this, we normalize each of the three candidate sequences (per-channel min-max normalization at each step t) and then time-average them, yielding x¯, ϵ¯, and ∆¯ϵ. As shown in Fig. 3(b), ∆¯ϵ exhibits clearer semantic structures compared to the others, suggesting it is a more potent target for frequency-based operations. Further examples in Fig. 5 corroborate the significant role of frequency components within ∆ϵt.

Step-wise Frequency Dynamics. Based on our analysis of the three diffusion representations, we further examine their evolution of spectral profiles throughout the denoising trajectory (see Fig. 4). Our key observations are:

- 1. The spectrum of the noisy latent xt shows that low-frequency structures quickly converge in early step, and emerge more clear in later steps as the high-frequency noise is attenuated.
- 2. The specturm of ϵ1:T shows more flutations, and no consistent trend is found across different t.
- 3. ∆ϵ1:T evolves from a more low-pass characteristic at early, high-noise stages towards a broader, flatter spectrum at later stages. Furthermore, as t decreases, its magnitude generally increases, signifying that the guidance becomes more influential in refining details during later steps.

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

- Figure 5: Examples of original SDXL generations (top) and the generation results by applying high-pass (middle) and low-pass filters (bottom) on ∆ϵ1:T.

[Figure 92]

[Figure 93]

[Figure 94]

Figure 6: Overview of FreSca. We introduce scaling factors l and h to decompose the control mechanisms in the Fourier domain.

### 3.3 FreSca: Versatile Frequency Scaling in Diffusion Models

Building on the finding that the noise difference term ∆ϵt is a semantically rich and suitable candidate for frequency manipulation, we introduce FreSca, a framework for versatile frequency scaling within

LDMs. FreSca operates by decomposing ∆ϵt into its low- and high-frequency components and then applying independent scaling factors to each.

Let Ut be the Fourier transform of the noise difference term at timestep t. Using the low-pass (Ml) and high-pass (Mh) masks defined in Eq. (2) (which depend on a cutoff choice, see below), we define the modified noise difference term ∆ˆϵt as:

∆ˆϵt = F−1 l · Ml ⊙ Ut + h · Mh ⊙ Ut , (5) where we introduce two scaling factors l and h that allow for independent amplification or suppression of different frequency bands. This modified ∆ˆϵt then replaces ∆ϵt in Eq. (4). Generally, FreSca offers several advantages:

- • Flexibility: Independent scaling of low and high frequencies enables effects from fine-detail enhancement (h > 1,l = 1) to smoothing (l > 1,h < 1) or targeted stylization of specific bands.
- • Faithfulness: When l = h = 1, FreSca losslessly reduces to the original CFG mechanism.
- • Generality: As it operates on the noise difference term, a ubiquitous component of conditional diffusion, FreSca applies seamlessly across architectures (e.g., SDXL, SD3) and tasks.

Dynamic Cutoff Determination. The effectiveness of FreSca can be further enhanced by dynamically adjusting the frequency separation (i.e., the cutoff radius Rc used in Ml,Mh) at each timestep t. We propose two strategies for determining Rc(t):

- 1. Spatial-Ratio Cutoff: The cutoff radius Rc(t) is determined based on a predefined ratio r0: Rc(t) = r0 · min(Ht/2,Wt/2), (6)

where Ht and Wt are the spatial dimension of Ut.

- 2. Energy-Based Cutoff: Let Etot(t) = k

x,ky Ut(kx,ky) . We choose the smallest integer R such that the cumulative magnitude within radius R reaches a fraction r0 of Etot(t):

Ut(kx,ky) ≥ r0 Etot(t) . (7)

Rc(t) = min R ∈ N0 |

√

kx2+ky2 ≤ R

This tailors “low” versus “high” frequencies to the spectral energy distribution at each step.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Figure 7: Samples generated by SDXL [5] and SD3 [2] with or without FreSca.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Figure 8: Ablation of cutoff strategies: (a) original SDXL output; FreSca applied with (b) spatial-ratio cutoff and (c) energy-based cutoff (both h = 1.5). The adaptive energy-based cutoff yields the closest alignment to the prompt.

Figure 9: Cumulativeenergy curve that tells how r0 affects cutoff radius at timestep t.

## 4 Experiments: FreSca is Vesatile, Model-Agnostic, and Task-Agnostic

### 4.1 Task: Text to Image Generation

Generalization Across Models. To demonstrate FreSca ’s model-agnostic versatility, we incorporate it into two distinct image generation methods: SDXL [5], which uses a U-Net backbone, and SD3 [2], a multimodal diffusion transformer. In Fig. 7, both setups employ a high-frequency scaling factor

- h = 1.5 with an energy-based cutoff r0 = 0.9. In each case, FreSca enhances prompt fidelity and overall image quality, producing outputs that better match the text description while exhibiting fewer distortions.

Ablation on Cutoff Strategy. In Fig. 8, we compare the baseline SDXL output against FreSca using spatial-ratio and energy-based cutoffs. The energy-based variant, with its adaptive radius schedule shown in Fig. 9, produces generations that more closely match the prompt.

More image generation results and ablations on the effect of scaling factors h, l, and different cutoff ratio can be found in the supplementary materials.

### Table 2: Zero-shot depth estimation on DIODE, KITTI, and ETH3D. We compare Marigold and

Marigold + FreSca using AbsRel (lower better) and δ1 (higher better); bold denotes best, underline represents second best. Our method consistently improves both indoor and outdoor results. †Official Marigold implementation.

DIODE [30] KITTI [31] ETH3D [32]

Method Ensemble

AbsRel↓ δ1 ↑ AbsRel↓ δ1 ↑ AbsRel↓ δ1 ↑

Marigold† ✗ 31.0 77.2 10.5 90.4 7.1 95.1 Marigold† ✓ 30.8 77.3 9.9 91.6 6.5 96.0 Marigold† w/ FreSca ✓ 30.2 77.8 9.8 91.7 6.4 95.9

[Figure 120]

Figure 10: FreSca sharpens depth predictions. From top to bottom: input RGB, Marigold + FreSca, and Marigold. Red arrows highlight where our method recovers clearer shapes and reduces blur.

### 4.2 Task: Monocular Depth Estimation

Monocular depth estimation recovers 3D scene geometry from a single imag – a key capability for autonomous driving, robotics, and augmented reality. Despite its intrinsic 2D to 3D ambiguity, latent diffusion methods like Marigold [1], which fine-tunes only the denoising U-Net of Stable Diffusion [33] on synthetic RGB-D data, achieve strong zero-shot performance on real-world benchmarks without ever using real depth maps.

While it generalizes well, it can miss fine details and misestimate distant objects. To address this, we equip Marigold with FreSca boosting its high-frequency noise components (h > 1, l = 1) while leaving the low frequencies intact. Specifically, Marigold’s predictor ϵt = ϵθ(dt,x,t) [1] runs with fixed classifier-free guidance (ω = 1) and relies solely on the conditional branch. Therefore, we apply FreSca directly to the predicted noise ϵt, since this noise encodes the semantic information necessary for accurate, detail-rich depth estimation.

As Tab. 2 shows, integrating FreSca consistently outperforms Marigold baselines (with or without ensemble) on DIODE [30], KITTI [31], and ETH3D [32], achieving leading AbsRel and δ1 metrics. Unlike ensembling, which can oversmooth, our frequency-based adjustment yields more deterministic, accurate depth maps, recovering fine structures and sharp edges (see Fig. 10).

### 4.3 Task: Text-guided Image Editing

Dataset and Baselines. We conduct experiments on the public image editing dataset TEdBench [34], which comprises 40 images from diverse categories paired with various editing prompts. FreSca can be seamlessly integrated into existing image editing frameworks without altering their core architectures. Accordingly, we benchmark our approach against training-free methods, including LEdits++[17] and Edited-Friendly DDPM Inversion[18], strictly following their prescribed settings.

Evaluation Protocol. For quantitative comparison, we fix the CFG ω = 15 for all methods. In our framework, we set l = 1 for both, while applying h = 1.2 for Edited-Friendly DDPM Inversion and

- h = 2.0 for LEdits++, using a spatial cutoff radius of 20 in both cases. Further discussion on l and h choices can be found in the supp. We measure editing fidelity with the CLIP-text similarity [35]

- Table 3: Image editing results evaluated by both generative metrics (FID-30k and CLIP-text) and human-centric VLM metrics (Success Rate and Quality).

FID-30k ↓ CLIP-text (%) ↑ Success Rate (%) ↑ Quality

Edited-Friendly DDPM [18] 255.5 31.35 75.0 4.23 DDPM [18] w/ FreSca 253.4 31.54 80.0 4.18

LEdits++ [17] 255.3 31.08 72.5 4.08 LEdits++ [17] w/ FreSca 255.0 31.34 72.5 4.18

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

- Figure 11: Editing results from LEdits++ [17] and DDPM inversion [18] with or without FreSca.

against the target prompt, and assess overall image quality via FID-30k [36]. Additionally, we perform qualitative evaluation using the large vision–language model InternVL2.5-8B [37].

Results. As reported in Tab. 3, integrating FreSca into both Edited-Friendly DDPM Inversion and LEdits++ consistently boosts CLIP-text scores and reduces FID, demonstrating that selective amplification of high-frequency detail strengthens the target edit, preserves image fidelity, and increases the editing success rate. Qualitative examples in Fig. 11 further illustrate these enhancements.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

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

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

- Figure 12: FreSca enhances VideoCrafter2’s [3] video generation quality at no additional cost.

### 4.4 Task: Text to Video Generation

FreSca’s applicability is not limited to static image tasks; we demonstrate its effectiveness in the dynamic domain of video generation. We integrate FreSca into VideoCrafter2 [3], an open-source video diffusion model. By modulating solely the high-frequency components of the predicted noise, we achieve improvements in video quality and fidelity without any model retraining. As illustrated in Figs. 1 and 12, FreSca enhances motion coherence, preserves intricate details, and mitigates text-video misalignment. This underscores FreSca’s significant potential and versatility across diverse diffusion models.

## 5 Conclusion

This paper introduced FreSca, a novel framework enabling fine-grained, disentangled control over latent diffusion models through frequency-domain manipulation. By targeting the semantically rich classifier-free guidance noise difference ∆ϵt, FreSca decomposes it into frequency bands, applying scaled adjustments with dynamic cutoffs. This model-agnostic, plug-and-play approach is shown to effectively control visual attributes across various models (SDXL, SD3) and tasks (image generation, editing, depth estimation, video synthesis). FreSca not only provides practical creative control but also contributes to understanding frequency components in LDMs. Future work could explore advanced spectral techniques and learned control strategies.

## References

- [1] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [2] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.
- [3] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310–7320, 2024.
- [4] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [5] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [6] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.
- [7] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 35:36479–36494, 2022.
- [8] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, pages 3836–3847, 2023.
- [9] Edward H Adelson, Charles H Anderson, James R Bergen, Peter J Burt, and Joan M Ogden. Pyramid methods in image processing. RCA engineer, 29(6):33–41, 1984.
- [10] Eduardo SL Gastal and Manuel M Oliveira. Domain transform for edge-aware image and video processing. In ACM SIGGRAPH 2011 papers, pages 1–12. 2011.
- [11] Burhan Ergen. Signal and image denoising using wavelet transform. InTech London, UK, 2012.
- [12] Xin Deng, Ren Yang, Mai Xu, and Pier Luigi Dragotti. Wavelet domain style transfer for an effective perception-distortion tradeoff in single image super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3076–3085, 2019.
- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [14] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In CVPR, 2024.
- [15] Jiazi Bu, Pengyang Ling, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Broadway: Boost your text-to-video generation model in a training-free way. arXiv preprint arXiv:2410.06241, 2024.
- [16] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. arXiv preprint arXiv:2406.01493, 2024.

- [17] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinário Passos. Ledits++: Limitless image editing using text-toimage models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8861–8870, 2024.
- [18] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469–12478, 2024.
- [19] Sam Witteveen and Martin Andrews. Investigating prompt engineering in diffusion models. arXiv preprint arXiv:2211.15462, 2022.
- [20] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 4296–4304, 2024.
- [21] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In ICLR, 2022.
- [22] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023.
- [23] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, pages 22500–22510, 2023.
- [24] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [25] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.
- [26] Ricard Durall, Margret Keuper, and Janis Keuper. Watch your up-convolution: Cnn based generative deep neural networks are failing to reproduce spectral distributions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7890–7899, 2020.
- [27] Daniel Geng, Inbum Park, and Andrew Owens. Visual anagrams: Generating multi-view optical illusions with diffusion models. Computer Vision and Pattern Recognition (CVPR), 2024.
- [28] Daniel Geng, Inbum Park, and Andrew Owens. Factorized diffusion: Perceptual illusions by noise decomposition. European Conference on Computer Vision (ECCV), 2024.
- [29] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.
- [30] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.
- [31] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE, 2012.
- [32] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017.

- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.
- [34] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.
- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021.
- [36] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [37] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

- Figure 13: Visual effects of varying cutoff thresholds and scaling factors (r0, l,h) using the SpatialRatio Cutoff strategy.

## A Analysis on Text to Image Generation

### A.1 Analysis of Frequency Scaling Parameters and Cutoff Strategies

Effects of Frequency Scaling Factors l,h, and Cutoff Ratio r0. We investigate the impact of our low-frequency scaling factor l, high-frequency scaling factor h, and the cutoff ratio r0 under two distinct frequency cutoff strategies.

Spatial-Ratio Cutoff Strategy. This strategy defines the cutoff frequency based on a spatial frequency ratio r0, where a fixed proportion of the lowest spatial frequencies are low-frequency components (Fig. 13).

- • Impact of Cutoff Ratio r0: Low r0 (0.1) results in most frequencies being treated as highfrequency, leading to strong detail amplification and potential noise with high h. Increasing r0 towards 0.3 ∼ 0.5 designates a larger portion as low-frequency, yielding a more balanced mix of

structure and detail enhancement. High r0 means most frequencies are low-frequency, resulting in smoother images with subtle detail “pop” even with high h, as fewer high-frequency components exist.

- • Impact of Low-Frequency Scaling Factor l: Varying l scales coarse structures (fixed r0,h = 1.0). Low l (0.2) heavily suppresses coarse forms, emphasizing edges and textures. l values 0.5 ∼ 0.8 attenuate coarse structures to a lesser degree, balancing form and detail. l ≥ 1.5 enhances coarse structures, potentially overpowering fine details.
- • Impact of High-Frequency Scaling Factor h: Varying h scales fine details and textures (fixed

r0,l). Low h (0.2) suppresses details, making the image less sharp. h = 1.5 provides noticeable sharpening without significant artifacts. Very high h (= 4) causes strong, often artifact-prone sharpening, potentially useful for stylized effects but detrimental to realism.

Energy-based Cutoff Strategy. This strategy defines the cutoff frequency based on the cumulative energy spectrum, with r0 as the energy threshold (Fig. 14).

• Impact of Cutoff Ratio r0: Varying r0 redistributes spectral energy between the low and highfrequency bands. For low r0 (e.g., 0.1 ∼ 0.5), most energy is in the high-frequency band; scaling the low-frequency components has minimal impact, highlighting the energy distribution.

For high r0 (0.5 ∼ 0.9), most energy is low-frequency. In this case, scaling factors become more influential, particularly a high h which enhances finer details within the remaining high

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

#### Figure 14: Visual effects of varying cutoff thresholds and scaling factors (r0, l,h) using the Energybased Cutoff strategy.

frequencies. The sensitivity of local structures to low h (e.g., 0.2) further demonstrates the crucial role of high frequencies.

- • Impact of High-Frequency Scaling (h): (e.g., r0 = 0.7, l = 1.0) Increasing h amplifies fine details. h = 1.0 is baseline. h = 1.5 − 2.0 yields mild to clear detail enhancement without significant artifacts. h ≥ 2.5 leads to strong sharpening and potential artifacts. Photographic realism is best achieved with a moderate h ∈ [1.5,2.5]; h > 3.0 suits stylized effects.
- • Impact of Low-Frequency Scaling (l): (e.g., r0 = 0.7, h = 1.0) Varying l scales coarse structures. l < 1 can negatively impact prompt adherence, while l > 1 appears to improve it. Changes to local structure and style from varying l are generally more subtle than those from h.

Summary. Optimal parameter selection balances structural preservation and frequency component scaling. The Energy-based cutoff strategy offers good interpretability, and the significant role of high frequencies allows for diverse applications.

- A.2 Fine-grained Changes with Different Cutoff Thresholds for Pixel & VAE Spaces To complement Fig. 2, we provide a more detailed change as visualized in Fig. 15.

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

#### Figure 15: Frequency decomposition with different cut-off thresholds (a more fine-grained version of Fig. 2.

- Table 4: Ablation study of FreSca applied to SD3, showing the effect of slightly varying the hyperparameters h, l, and r0 on the two generation evaluation metrics FID and CLIP-text scores.

Method h l r0 FID ↓ CLIP Score (%) ↑ SD3 (baseline) – – – 219.96 16.24 SD3 + FreSca

- w/ FreSca 1.0 1.1 0.9 219.70 ▼ 16.23 ▼
- w/ FreSca 1.1 1.0 0.9 220.47 ▲ 16.25 ▲ w/ FreSca 1.1 1.0 0.7 220.57 ▲ 16.30 ▲ w/ FreSca 1.1 1.0 0.5 219.98 ▲ 16.23 ▼

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

- Figure 16: Visualization of FreSca applied to SD3, showing the effect of varying the hyperparameters

h, l, and r0. Red box shows the region of interest, where increasing high freqeuncy bring higher image-prompt alignment compared to the baseline, while improving low-frequency marginally improve the generation FID.

### A.3 Quantative and Visual Effects of Frequency Scaling Parameters (h, l, r0)

We quantitatively analyze the effects of varying the hyperparameters h, l, and r0 using the energybased cutoff strategy) on the SD3 base model, as presented in Tab. 4.

Our observations reveal that enhancing high-frequency components effectively improves image-text alignment (higher CLIP score), though it slightly degrades the generation FID. Conversely, enhancing low-frequency components yields inverse effects: a lower (better) FID but a diminished (worse) CLIP score.

As visually demonstrated in Fig. 16, the enhancement of high-frequency components is crucial as it significantly improves prompt alignment and facilitates better instruction following. Importantly, the minor quantitative degradation in FID (e.g., from 219.96 to 220.47) does not noticeably impact the subjective quality of the generated images.

Therefore, frequency scaling proves to not only be a useful technique for controlling image characteristics, but also in affecting diffusion-based image representations. Optimizing the combination of different hyperparameter sets for specific generation goals is an important direction for future work.

### A.4 Understanding Step-wise Dynamics of High-Frequency Scaling

Beyond the observation that enhancing high-frequency components improves image-prompt alignment, we explore whether the scaling factor h benefits from a time-dependent schedule. As illustrated in Fig. 4, the high-frequency components of ∆ϵ intensify as the denoising process progresses.

- Table 5: Ablation study of FreSca applied to SD3, showing the effect of varying the high-frequency scaling schedule on FID and CLIP-text scores.

### Method h l r0 FID ↓ CLIP Score (%) ↑

w/ FreSca 1.1 1.0 0.9 220.47 16.25 w/ FreSca Linear Growth 1.1 1.0 0.9 220.60 ▲ 16.19 ▼ w/ FreSca Linear Decay 1.1 1.0 0.9 219.69 ▼ 16.23 ▼

[Figure 211]

[Figure 212]

[Figure 213]

Figure 17: Visual results comparing different high-frequency scaling schedules.

[Figure 214]

Figure 18: Illustration of Linear Decay and Linear Growth schedules for the high-frequency scaling factor h over denoising steps t.

To investigate the importance of this dynamic, we introduce two scheduling strategies for h, defined over the total 50 denoising steps (t ∈ [0,49]), as shown in Fig. 18:

Linear Decay: h(t) = 4949−t ·(hmax −1)+1 Linear Growth: h(t) = hmax − 4949−t ·(hmax −1) where hmax is the maximum high-frequency scaling factor (e.g., 1.1 in our ablation).

As detailed in Tab. 5 (using hmax = 1.1), adopting a Linear Decay strategy for h yields better FID while slightly reducing the CLIP score compared to a constant h. This suggests that attenuating high-frequency factors in earlier steps provides better image preservation, as the higher magnitude of high-frequency components in later steps makes their scaling more impactful. Conversely, the Linear Growth strategy did not contribute positively to either metric.

As verified qualitatively in Fig. 17, dynamic adjustment of high-frequency components can lead to more faithful results. This time-aware scheduling serves as an optional strategy for practical implementation, warranting further investigation in future work.

- A.5 More Image Generation Results We show more generation results from SDXL and SD3 with or without FreSca below.

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

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

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

Figure 19: More generation examples from SDXL and SD3 with or without FreSca.

## B Results on Text to Video Generation

We have created a project page to illustrate our method and showcase our results. We strongly encourage readers to visit this webpage.

## C More Details on the Editing Task

Details about TEdBench [34] Dataset The complete list of image names and their target text prompt mappings we used for evaluation are shown in Tab. 6.

|Image Name<br><br>|Target Text|
|---|---|
|dog2_standing.png tennis_ball.jpeg zebra.jpeg red_car.jpeg bird.jpeg box.jpeg cat.jpeg cat_3.jpeg dog_with_shirt.jpg dog_01.jpeg vase_01.jpeg door.jpeg couple_beach.jpeg open_book.jpeg empty_street.jpeg black_shirt.jpeg bear3.jpeg milk_cookie.jpeg chibi.jpeg giraffe.jpeg apples.jpeg new_cat_3.jpeg chair_1.jpeg flamingo.jpeg banana_1.jpeg cake_1.jpeg tree_1.jpeg teddy_1.jpeg<br><br>white_horse1.png<br>white_horse2.png prague.png<br><br><br>bird.png goat_and_cat.jpg elephant.jpeg road1.png egg_tree.jpeg two_dogs_with_checkered_shirts1.jpg pizza1.png drinking_horse.png bird-g83440b9c4_1920.jpg|A photo of a sitting dog.<br><br>A photo of a tomato in a blue tennis court.<br><br>A photo of a horse.<br><br>A photo of a car in Manhattan.<br><br>A photo of a bird spreading wings.<br><br>A photo of an open box.<br><br>A photo of a cat wearing a hat. A photo of a cat wearing a hat. A dog smoking a cigar.<br><br>A photo of a sitting dog.<br><br>A photo of a vase of red roses.<br><br>A photo of an open door.<br><br>A photo of a couple holding hands on a beach.<br><br>A photo of a closed book.<br><br>A busy congested street.<br><br>A person with crossed arms.<br><br>A black bear walking in the grass next to red flowers.<br><br>A cookie next to a glass of juice.<br><br>Image of a cat wearing a floral shirt.<br><br>A giraffe with a short neck.<br><br>A basket of oranges.<br><br>A photo of a sleeping cat.<br><br>A knocked down chair.<br><br>A sitting flamingo.<br><br>A photo of a sliced banana. A photo of a birthday cake. A photo of a dead tree.<br><br>A photo of a teddy bear doing pushups.<br><br>A white horse in a grass field.<br><br>A jumping horse.<br><br>A cyclist riding in a street. A bird looking backwards. A goat and a cat hugging.<br><br>A person riding on an elephant.<br><br>An image of a post-apocalyptic road.<br><br>A cracked egg.<br><br>Two dogs growling at each other.<br><br>Pizza with pepperoni.<br><br>A horse raising its head.<br><br>Two kissing parrots.<br><br>|

Table 6: Image names and their corresponding target texts.

Success Rate & Quality Metic in Tab. 3. We further evaluate the edited images using the large vision-language model InternVL2.5-8B [37]. This model provides a binary decision (0 or 1) to indicate whether the editing was successful and assigns a qualitative score on a scale of 1 to 5—where 1 denotes poor quality and 5 reflects excellent performance in both concept fidelity and overall image

[Figure 239]

Figure 20: Prompts designed for LVLM evaluation.

quality. As shown in Tab. 3, incorporating FreSca not only improves the overall quality of the edited outputs but also increases the editing success rate. This demonstrates the effectiveness of our approach in achieving high-quality, semantically faithful edits. The prompt design for obtaining these metrics are shown in Fig. 20.

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

- Figure 21: Key components for image editing: (a) latent vector xT is obtained through inversion techniques; (b) the visualizations of ∆ϵ at the first editing step; (c) the final edited output.

Role of ∆ϵ in Image Editing Intuitively, the noise difference term ∆ϵt in image editing encodes spatial regions corresponding to the target prompt. We validate this through three distinct editing scenarios depicted in Fig. 21: replacement editing, self-editing, and using an unrelated prompt. As illustrated, the ∆ϵt maps clearly show activation in prompt-relevant areas for semantically related prompts, while exhibiting diffuse or random patterns for unrelated ones. This analysis leads to three key observations about the editing process: 1. Semantically Rich Inversion: The inverted initial latent xT preserves essential input semantics, aligning with the target prompt c′. 2. ∆ϵt As Prompt Proxy: The noise prediction difference ∆ϵt effectively isolates and spatially localizes the representation of the target prompt. 3. ω Modulates Edit Strength and Direction: Given that ∆ϵt

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

- Figure 22: Frequency scaling effects on the image editing task: We set the target prompt to increasing the size of stones and apply three different scaling strategies in the frequency domain: uniform scaling (h = l = 1.5), low-frequency scaling (l = 1.5, h = 1), and high-frequency scaling (h = 1.5, l = 1). Each approach produces distinct effects.

represents the target concept, the scalar factor ω directly modulates the strength and determines the enhancement or suppression direction of the edit.

C.1 Understanding Editing Dynamics via Fourier Analysis

Here, we further study the roles of ω and ∆ϵt in image editing, understand how frequency scaling works for this task. We analyze ∆ϵt in the Fourier domain, decomposing it into low (∆ϵlt) and high (∆ϵht ) frequency components using a spatial-ratio cutoff threshold (with r0 = 0.3). We question if low- and high-frequency dynamics are equivalent. By introducing independent scaling factors l and h, we found distinct roles. Fig. 22 shows that asynchronous scaling (e.g., l = 1.5,h = 1) primarily affects structure, while h = 1.5,l = 1 adds texture. Also, the relative Fourier log-amplitude patterns are different when choosing different combinations of l and h. Therefore, it reveals that low and high frequencies are not always synchronous, suggesting a need for flexible scaling.

Transition with varying h. For a given image, altering h from values below 1 to values above 1 produces an intriguing transition. When h < 1, gradually increasing h (e.g., from 0.5 to 0.8) introduces fundamental structural details, as evidenced by the appearance of the “riding horse person.” In contrast, when h > 1, further increases enhance edges, contours, and other high-frequency features. These findings indicate that h spans a scaling space that governs both high-frequency patterns and the underlying structural composition of the image, demonstrating that FreSca offers superior controllability compared to prior scaing space.

The role of high-frequency scaling factors h. As demonstrated Fig. 24, adjusting the high-frequency scaling factor h produces two distinct effects: when h > 1, the representation of shape, structure, and contour is enhanced, while setting h < 1 introduces a counter-effect that pulls the edited result closer

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

- Figure 23: Continuous adjustment of high-frequency components. We scale the h from 0.5 to 2 to examine its impacts on the editing performance.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

| |
|---|

| |
|---|

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 24: Scaling up the high-frequency parts (h = 2.0) effectively enhances the editing fidelity.The red hat is successfully injected, and the edge of the LEGO flowers is sharpened. (a) input image, (b) results from FreSca with different h being set, and (c) results from LEdits++.

to the original image. This creates a practical trade-off between inducing more pronounced shape changes and better preserving the original structure. FreSca decouples these components, achieve varying levels of subtle control on h without altering the primary editing direction.

## D Experiment Configuration

Here, we summarize the default configurations for getting results for different task in the main paper. Note that we do not massively search for the best combination of h, l, and r0, but rather empircally pick a set for each task. Even without grid-search, FreSca works as an effective plug-and-play module for different models and different method. To observe, all task favors enhancing the highfrequency components while keeping its low-freq the same. In the following sections, we will show the effect of different roles for adjusting h, l, and etc.

Table 7: Configuration settings for each task in the main paper. Task Baseline h l r0 Cutoff strategy Dataset Text-to-Image Generation

SDXL [5] 1.5 1 0.9

Energy-based N/A SD3 [2] 1.2 1 0.9

1.5 1 0.3

DIODE [30] 1.2 1 0.3 KITTI [31] 1.1 1 0.3 ETH3D [32]

Monocular Depth Prediction Marigold [1]

Spatial-ratio

LEDits++ [17] 2.0 1 0.3

Text-guided Image Editing

Spatial-ratio TEdBench [34] DDPM Inversion [18] 1.2 1 0.3

Text-to-Video Generation VideoCrafter2 [3] 1.5 1 0.9 Energy-based N/A

- E Simple Pytorch Implementation Please refer to Fig. 25 for an example implementation of FreSca with energy-based cutoff method.

[Figure 280]

Figure 25: A simple pytorch implementation of our FreSca in less than 70 lines of code.

