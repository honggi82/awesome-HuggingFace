arXiv:2506.19713v1[cs.LG]24Jun2025

# Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales

Seyedmorteza Sadat1, Tobias Vontobel1, Farnood Salehi2, Romann M. Weber2 1ETH Zürich, 2DisneyResearch|Studios {votobias, ssadat}@ethz.ch {farnood.salehi, romann.weber}@disneyresearch.com

## Abstract

Classifier-free guidance (CFG) has become an essential component of modern conditional diffusion models. Although highly effective in practice, the underlying mechanisms by which CFG enhances quality, detail, and prompt alignment are not fully understood. We present a novel perspective on CFG by analyzing its effects in the frequency domain, showing that low and high frequencies have distinct impacts on generation quality. Specifically, low-frequency guidance governs global structure and condition alignment, while high-frequency guidance mainly enhances visual fidelity. However, applying a uniform scale across all frequencies—as is done in standard CFG—leads to oversaturation and reduced diversity at high scales and degraded visual quality at low scales. Based on these insights, we propose frequency-decoupled guidance (FDG), an effective approach that decomposes CFG into low- and high-frequency components and applies separate guidance strengths to each component. FDG improves image quality at low guidance scales and avoids the drawbacks of high CFG scales by design. Through extensive experiments across multiple datasets and models, we demonstrate that FDG consistently enhances sample fidelity while preserving diversity, leading to improved FID and recall compared to CFG, establishing our method as a plug-and-play alternative to standard classifier-free guidance.

CFG FDG (Ours)

[Figure 1]

[Figure 2]

- Figure 1: Low classifier-free guidance produces images with good diversity and color composition but often results in low-quality and blurry generations. We propose frequency-decoupled guidance (FDG), a novel modification to the CFG update rule in the frequency domain that significantly improves image quality at low guidance scales while, by design, avoiding common issues at high CFG scales such as reduced diversity. Images are generated using class-conditional DiT-XL/2 [41] with a guidance scale of 1.2.

## 1 Introduction

Diffusion models [53, 18, 56] are a class of generative models that learn the data distribution by reversing a forward process that gradually corrupts data with Gaussian noise until it resembles pure noise. While the theory suggests that simulating this reverse process should yield high-quality samples, in practice, unguided sampling often produces low-quality images that poorly match the input condition. To address this, classifier-free guidance (CFG) [17] has become a standard technique in modern diffusion models for improving both output quality and alignment with the conditioning signal—though often at the cost of reduced diversity [17, 47] and excessive oversaturation [49].

Typically, diffusion models rely on high guidance scales to achieve better image quality and prompt alignment. However, high guidance scales degrade sample diversity and introduce color saturation artifacts [17]. Conversely, low CFG scales tend to produce more diverse samples with natural color compositions but often suffer from poor global structure and lower visual fidelity. To address these trade-offs, several empirical strategies have been proposed to balance diversity and quality of CFG [28, 47, 49, 60]. Despite this progress, a systematic understanding of how CFG improves image quality and prompt alignment remains limited. Existing works do not fully explain the internal mechanisms of CFG or explore how to improve generation quality at low guidance scales.

In this paper, our objective is to advance the understanding of how CFG works and improve image quality at low CFG scales, thereby avoiding the detrimental effects associated with high guidance scales. We begin by analyzing the CFG update rule in the frequency domain and show that CFG enhances image quality and prompt alignment through distinct frequency components. Specifically, we find that the low-frequency components primarily govern the global structure and condition alignment, while the high-frequency components mainly contribute to the visual quality and details, with minimal impact on the overall composition of the image. We also observe that excessive guidance in the low-frequency domain leads to reduced diversity and oversaturation, whereas high-frequency components usually benefit from higher guidance scales. Since standard CFG applies a uniform scale across all frequencies, this explains why low CFG scales degrade quality, and high CFG scales boost detail at the cost of diversity and oversaturation.

Building on this insight, we propose a new classifier-free guidance scheme in the frequency domain, termed frequency-decoupled guidance (FDG), which disentangles the guidance scales applied to the low- and high-frequency components of the CFG signal. We argue that low-frequency components should be guided more conservatively to avoid low-diversity and oversaturated generations, while high-frequency components can benefit from stronger guidance to enhance image quality. By assigning separate guidance strengths to these components, FDG improves image quality while retaining the diversity typically associated with low CFG scales. FDG is among the first approaches to systematically enhance the quality of low CFG scales, thereby avoiding the drawbacks of high CFG scales by design.

FDG introduces practically no additional sampling cost and can be applied to any pretrained diffusion model without extra training or fine-tuning. Through extensive experiments, we show that FDG consistently improves image quality and maintains diversity across a range of datasets, models, and metrics. Moreover, FDG offers a deeper understanding of how CFG enhances image quality and prompt alignment by separately analyzing the roles of low- and high-frequency components in the CFG update rule. Accordingly, we consider FDG as a superior plug-and-play alternative to standard classifier-free guidance for conditional diffusion models.

## 2 Related work

Score-based diffusion models [55, 56, 53, 18] learn the data distribution by reversing a forward diffusion process that progressively corrupts the data with Gaussian noise. These approaches have rapidly surpassed previous generative modeling techniques in terms of fidelity and diversity [39, 11], achieving state-of-the-art performance across a wide range of applications, including unconditional image synthesis [11, 22], text-to-image generation [43, 51, 3, 44, 42, 64, 12], video synthesis [5, 4, 14], image-to-image translation [50, 32], motion synthesis [58, 59], and audio synthesis [9, 26, 21].

Recent studies have introduced numerous enhancements to the original DDPM framework [18], including improved network architectures [19, 23, 41, 11], novel sampling strategies [54, 22, 34, 35, 52], and better training techniques [39, 22, 56, 52, 44]. Despite these advancements, guidance

methods—such as classifier-free guidance [17]—remain essential for enhancing sample quality and improving alignment between conditioning information and generated outputs [40].

In modern diffusion models, high guidance scales are essential for improving image quality and enhancing alignment between the generated output and conditioning inputs. However, this comes at the cost of reduced diversity [17, 47] and undesirable effects such as oversaturation [49]. Our work addresses this issue by disentangling the benefits of high CFG from its detrimental effects. Leveraging frequency decomposition, we propose a novel approach that improves image quality at lower CFG scales, inherently avoiding the trade-offs associated with high guidance.

Several recent works have aimed to improve diffusion model training for unguided generation [23, 65, 29, 20]. While unguided generation is theoretically expected to produce high-quality, diverse samples without the drawbacks of CFG, in practice, strong CFG guidance is still often necessary for high-quality generation even with such improved models. In contrast, our method enhances sample quality at low CFG scales without retraining, making it readily applicable to any pretrained model.

Frequency decomposition techniques, such as Laplacian pyramids [8] and wavelet transforms [6], have recently been employed in generative models to boost both quality and efficiency [10, 13,

- 2, 48, 1, 62]. However, their use in improving the sampling characteristics of diffusion models remains underexplored. We demonstrate that applying the CFG update rule in the frequency domain significantly enhances generation quality while avoiding the common issues of high CFG scales.
- 3 Background

Diffusion models Let x ∼ pdata(x) denote a data sample, and let t ∈ [0,1] represent a continuous time variable. The forward diffusion process adds noise to the data as zt = x + σ(t)ϵ, where σ(t) is a time-dependent noise schedule. This schedule controls the degree of corruption, with σ(0) = 0 (no noise) and σ(1) = σmax (maximum noise). As shown by Karras et al. [22], this forward process corresponds to the following ODE:

dz = −σ˙(t)σ(t)∇zt

log pt(zt)dt, (1)

where pt(zt) is the distribution over noisy samples at time t, with p0 = pdata and p1 = N 0,σmax2 I . Given access to the time-dependent score function ∇zt

log pt(zt), one can sample from the original data distribution by integrating the ODE in reverse from t = 1 to t = 0. Since this score function is unknown, it is approximated by a neural denoiser Dθ(zt,t), which is trained to recover the clean data x from its noisy counterpart zt. Conditional generation is enabled by augmenting the denoiser with an auxiliary input y, such as class labels or text prompts, yielding Dθ(zt,t,y).

Classifier-free guidance Classifier-free guidance (CFG) is an inference technique designed to enhance the quality of generated samples by interpolating between conditional and unconditional model predictions [17]. Let ynull = ∅ denote a null condition representing the unconditional case. CFG modifies the denoiser output at each sampling step as follows:

DˆCFG(zt,t,y) = Dθ(zt,t,ynull) + w(Dθ(zt,t,y) − Dθ(zt,t,ynull)), (2) where w = 1 corresponds to the unguided case. The unconditional model Dθ(zt,t,ynull) is trained by randomly replacing the condition y with ynull during training. Alternatively, the unconditional score can be learned using a separate model as in Karras et al. [23]. Similar to the truncation trick used in GANs [7], CFG enhances image quality, but often at the cost of sample diversity [38].

Frequency decompositions Multi-level frequency decompositions, such as Laplacian pyramids and wavelet transforms, are commonly used to separate an image into different frequency bands. These techniques decompose an image into coarse, low-frequency structures and fine, high-frequency details. The low-frequency components capture the global characteristics of the image, including object placement, overall geometry, and color distribution, while the high-frequency components represent localized information such as edges, textures, and fine structural details.

- 4 Classifier-free guidance in the frequency domain

We now describe how guidance in the frequency domain can enhance the characteristics of CFG. Our first goal is to understand how different frequency components in the CFG prediction DˆCFG(zt,t,y)

###### wlow = 1.5, whigh = 1.5

###### wlow = 7, whigh = 7

###### wlow = 7, whigh = 1.5

###### wlow = 1.5, whigh = 7

low quality, high diversity

high quality, low diversity

low quality, low diversity

high quality, high diversity

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 2: Illustration of how different frequency components of CFG affect generation. Sampling with low CFG scales results in diverse generations but lower overall quality. Increasing the CFG scale improves quality but reduces diversity. We show that the low-frequency component of the CFG signal primarily drives the reduction in diversity, while the high-frequency component contributes to quality enhancement without affecting diversity.

influence the final generation. Let ψ[·] denote a linear and invertible frequency transformation, such as a Laplacian pyramid or wavelet transform, which decomposes each input x into low- and highfrequency components, denoted by ψlow[x] and ψhigh[x], respectively. For notational convenience, we define Dc(zt)

.

.

.

= DˆCFG(zt,t,y) to represent the conditional, unconditional, and CFG outputs, respectively. As a consequence of the assumed properties of ψ, the CFG update rule can be expressed as

= Dθ(zt,t,ynull), and DˆCFG(zt)

= Dθ(zt,t,y), Du(zt)

#### DˆCFG(zt) = ψ-1[ψ[DˆCFG(zt)]] (3)

= ψ-1[ψ[Du(zt)] + w(ψ[Dc(zt)] − ψ[Du(zt)])]. (4) This implies that, in the frequency domain, the CFG update affects both the low- and high-frequency components of ψ[DˆCFG(zt)] = {DˆCFGlow (zt),DˆCFGhigh(zt)} as follows:

DˆCFGlow (zt) = ψlow[Du(zt)] + w(ψlow[Dc(zt)] − ψlow[Du(zt)]), (5) DˆCFGhigh(zt) = ψhigh[Du(zt)] + w(ψhigh[Dc(zt)] − ψhigh[Du(zt)]). (6)

Thus, in standard CFG, both low- and high-frequency components are guided using the same scale w throughout the sampling process. However, we argue that this approach is suboptimal, as the low- and high-frequency components exhibit different behaviors and influence distinct aspects of

the generated image. We find that strong guidance on the low-frequency component DˆCFGlow (zt) leads to oversaturation and reduced diversity, whereas high guidance on the high-frequency component

DˆCFGhigh(zt) primarily enhances image quality. On the other hand, low scales for wlow keeps diversity and realistic color composition while low values for whigh degrade the visual details of the image and result in reduced sample quality. Motivated by this, we propose a generalized CFG scheme,

called frequency-decoupled guidance (FDG), that employs separate guidance scales—wlow for lowfrequency and whigh for high-frequency components.

- Figure 2 illustrates the effect of low- and high-frequency components on generated images. For CFG

(wlow = whigh), low guidance scales lead to poor global structure and visual degradation, while high guidance reduces diversity. We observe that strong guidance on the low-frequency signal (i.e., a large

wlow) primarily causes diversity issues, whereas increasing whigh enhances quality without adverse effects on diversity. These findings highlight the limitations of using a single scale w in standard CFG: low w produces blurry or incoherent outputs, while high w reduces diversity and causes oversaturation. Our results therefore advocate for asymmetric guidance, where we set wlow < whigh.

Implementation details We use Laplacian pyramids [8] as the frequency transform ψ in our experiments. The complete algorithm for applying FDG as well as the pseudocode of our method is provided in the appendix. Notably, FDG requires only minor modifications to the standard CFG sampling procedure, introduces no significant computational overhead, and is readily compatible with all pretrained diffusion models.

CFG FDG (Ours) CFG FDG (Ours)

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

CFG FDG (Ours) CFG FDG (Ours)

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

- Figure 3: Class-conditional generation results using EDM2 with w = wlow = 1.25. FDG enhances image quality while maintaining the diversity of low CFG scales.

[Figure 15]

CFG FDG (Ours) CFG FDG (Ours)

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

CFG FDG (Ours) CFG FDG (Ours)

[Figure 20]

[Figure 21]

[Figure 22]

- Figure 4: Text-to-image generation results using Stable Diffusion XL with w = wlow = 2. FDG enhances the details of the CFG image while maintaining the overall structure and color palette.
- 5 Experiments

Setup We primarily conduct experiments on text-to-image generation using Stable Diffusion models [44, 42, 12], and class-conditional ImageNet [46] generation using EDM2 [23] and DiT-XL/2 [41]. In all cases, we adopt the default diffusion samplers provided by each model (e.g., Euler scheduler for Stable Diffusion XL), and utilize the official pretrained checkpoints and codebases to maintain consistency with the original implementations. More details on the experimental setup and the hyperparameters used for each experiment are given in the appendix.

Evaluation metrics For class-conditional models, we adopt Fréchet Inception Distance (FID) [16]

- as the main metric to assess both image quality and diversity, given its strong correlation with human perception. To account for FID’s sensitivity to implementation details, we evaluate all models under a consistent setup. Additionally, we report precision [27] as a supplementary quality metric and recall

- Table 1: Quantitative comparison between CFG and FDG. FDG consistently improves FID and recall, showing high generation quality while maintaining good diversity.

Model Guidance FID ↓ Precision ↑ Recall ↑ EDM2-S [23]

CFG 9.77 0.85 0.52 FDG (Ours) 5.44 0.83 0.68

EDM2-XXL [23]

CFG 8.65 0.83 0.56

- FDG (Ours) 4.99 0.82 0.68

DiT-XL/2 [41]

CFG 9.31 0.89 0.54

- FDG (Ours) 5.33 0.84 0.65

Stable Diffusion 2.1 [44]

CFG 24.99 0.68 0.44

- FDG (Ours) 23.33 0.69 0.49

Stable Diffusion XL [42]

CFG 25.23 0.64 0.49

- FDG (Ours) 24.60 0.62 0.52

Stable Diffusion 3 [12]

CFG 30.32 0.76 0.37 FDG (Ours) 27.68 0.76 0.42

- Table 2: Quantitative comparison of CFG and FDG across various evaluation metrics for text-to-image models. FDG consistently outperforms CFG on all metrics and models.

Benchmark Model Guidance ImageReward ↑ HPSv2 ↑ PickScore ↑ CLIP Score ↑

- Stable Diffusion 2.1

CFG −0.112 0.273 0.45 30.82 FDG (Ours) 0.101 0.279 0.55 31.81

Stable Diffusion XL

CFG 0.323 0.277 0.38 31.99 FDG (Ours) 0.595 0.286 0.62 32.78

- Stable Diffusion 3

DrawBench [51]

CFG 0.185 0.273 0.33 31.26 FDG (Ours) 0.803 0.288 0.67 33.08

- Stable Diffusion 2.1

CFG 0.109 0.270 0.42 31.55 FDG (Ours) 0.332 0.277 0.58 32.09

Stable Diffusion XL

CFG 0.442 0.276 0.48 32.22 FDG (Ours) 0.663 0.283 0.52 32.81

- Stable Diffusion 3

Parti Prompts [64]

CFG 0.527 0.272 0.31 31.84 FDG (Ours) 0.988 0.286 0.69 32.83

- Stable Diffusion 2.1

CFG 0.035 0.269 0.35 32.73 FDG (Ours) 0.282 0.276 0.65 33.56

Stable Diffusion XL

CFG 0.603 0.278 0.35 33.99

- FDG (Ours) 0.824 0.286 0.65 34.77

Stable Diffusion 3

CFG 0.470 0.272 0.29 32.22

- FDG (Ours) 1.00 0.288 0.71 33.73

HPS Prompts [61]

[27] to capture diversity. For text-to-image tasks, we further use ImageReward [63], HPSv2 [61], PickScore [25], and CLIP Score [15] to evaluate both image quality and prompt alignment.

### 5.1 Main results

Qualitative results Figures 3 and 4 qualitatively compare the generations of FDG and CFG for the EDM2 and Stable Diffusion XL models at low CFG scales. FDG enhances generation quality while preserving the overall structure and color palette of the CFG output. Thus, FDG improves the quality

- at low CFG scales while avoiding the issues caused by high CFG values.

Quantitative results Table 1 provides the metrics for FDG and CFG across several models at guidance values typically used in practice. We observe that FDG consistently improves FID and recall while largely maintaining the precision of the CFG outputs. We therefore conclude that FDG enhances quality while preserving diversity, leading to significantly better FID. Additionally, we evaluate generation quality at lower CFG values in Table 2 using metrics specifically designed to reflect human preferences for text-to-image models. As FDG boosts quality without introducing the detrimental effects of high CFG scales, it significantly outperforms CFG across all quality metrics for the Stable Diffusion models.

Recall

Saturation

Precision

FID

- 5
- 6
- 7
- 8
- 9
- 10
- 11

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

wlow = whigh = w

0.46

wlow = w, whigh = 1 wlow = 1, whigh = w

0.7

0.44

0.84

0.65

0.42

0.82

0.4

0.6

0.38

0.8

0.55

0.36

0.78

0.5

0.34

1 1.5 2 2.5 3 3.5 4

1 1.5 2 2.5 3 3.5 4

1 1.5 2 2.5 3 3.5 4

1 1.5 2 2.5 3 3.5 4

w

w

w

w

Figure 5: Illustrating the effect of different frequency components on CFG behavior. Although CFG improves quality, it rapidly restricts diversity and increases saturation, leading to higher FID and lower recall. Note that the low-frequency component is mainly responsible for these adverse effects, whereas the high-frequency component enhances quality while preserving diversity and color composition, resulting in better FID and recall.

### 5.2 Effect of different frequency components of CFG on the generated distribution

To directly measure the effect of different frequency components on CFG outputs, we compared CFG with two sampling variants that use only DˆCFGlow (zt) or DˆCFGhigh(zt). This was achieved by setting whigh = 1 or wlow = 1 in FDG. As shown in Figure 5, increasing wlow is the main cause of the reduced diversity in CFG, as evidenced by the higher FID and lower recall values. In contrast, increasing whigh improves precision while maintaining recall, leading to significantly better FID across most guidance scales. These results suggest that excessive low-frequency guidance is the main contributor to the adverse effects of high CFG, whereas increasing high-frequency guidance generally enhances generation quality. Additionally, we observed that the low-frequency component is the primary cause of oversaturation, explaining why high guidance scales result in color artifacts. Therefore, our method adopts low values for wlow and high values for whigh.

### 5.3 Frequency analysis of prompt alignment in CFG

We next demonstrate how different frequency components of the CFG signal influence the alignment between generated images and the input condition. Figure 6 shows that although both low- and high-frequency components improve alignment as the guidance scale increases, the low-frequency component is the primary driver of this effect. Therefore, FDG can achieve comparable or superior prompt alignment to CFG by appropriately combining low- and high-frequency components, as shown in Table 2.

CLIP Score

- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33 FDG (Ours)

wlow = whigh = w

wlow = w, whigh = 1 wlow = 1, whigh = w

0 2 4 6 8 10 12 14 16

w

Figure 6: Effect of frequency components of CFG on prompt alignment.

### 5.4 Additional experiments

Relation to variable guidance scale Several works have explored the use of time-dependent guidance scales to balance diversity and quality in classifier-free guidance [47, 28, 60]. For example, guidance interval (GI) [28] applies the guidance scale only during a limited range of sampling steps identified via grid search. We argue that the improvement in diversity offered by these methods is closely related to the frequency decomposition of the guidance signal. To support this, we provide the norms of the low- and high-frequency components of the guidance signal across sampling steps in Figure 7. Note that the norm of the high-frequency component increases over sampling, while the norm of the low-frequency component decreases. We observe that GI starts applying guidance when the norms of the two components become roughly equal. This suggests that applying guidance in the mid-stages, as GI does, implicitly increases the effective guidance on the high-frequency component relative to the low-frequency component. This analysis also provides a principled way to select an interval for applying GI, avoiding costly grid searches. Additionally, GI may still lead to quality degradation due to the absence of guidance at the beginning of sampling. As shown in Table 3, FDG outperforms GI in terms of image quality for Stable Diffusion XL by maintaining high-frequency guidance during the early sampling steps.

Frequency analysis of Autoguidance Autoguidance [24] proposed a modified version of CFG that replaces the unconditional prediction with a degraded version of the main diffusion model.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- 4

6

σ(t)

∥∆Dt∥ for CFG

| | | |Lo|w frequency| |
|---|---|---|---|---|---|
| | | |Hig|h frequenc|y|
| | | | | | |
| | | | | | |
| | | | | | |

0 20 40 60 80

0

2

4

6

σ(t)

∥∆Dt∥ for Autoguidance

Figure 7: Illustration of how the norm of the guidance signal ∆Dt behaves in the frequency domain for CFG and Autoguidance. For CFG, low-frequency components are dominant during most early steps (high σ(t)), which can be harmful. Guidance interval [28] improves this by limiting CFG to the shaded region (found by grid search), i.e., the steps where high-frequency components are dominant. In contrast, Autoguidance maintains strong high-frequency components throughout the sampling process, making the signal useful at all steps as observed by Karras et al. [24].

Table 3: Comparing guidance interval [28] with FDG based on Stable Diffusion XL [42]. FDG achieves better quality metrics due to having high-frequency guidance in earlier sampling steps.

Method ImageReward ↑ HPSv2 ↑ PickScore ↑ CLIP Score ↑

Guidance interval [28] 0.437 0.282 0.34 32.77 FDG (Ours) 0.595 0.286 0.66 32.78

Table 4: Quantitative comparison between CFG and FDG using SDXL-Lightning [30] as an example of a distilled model that uses fewer sampling steps. FDG outperforms sampling both with and without CFG, achieving better quality and good prompt alignment.

Method ImageReward ↑ HPSv2 ↑ CLIP Score ↑

w/o CFG 0.535 0.282 31.77 CFG 0.573 0.286 32.44 FDG (Ours) 0.672 0.292 32.44

- Figure 7 presents a frequency analysis of the update provided by Autoguidance, showing that both low- and high-frequency components remain strong throughout the sampling process, unlike CFG where low-frequency response dominates. This likely explains why the authors found Autoguidance effective at all steps and why it provides a better guidance direction compared to CFG.

Compatibility with distilled models CFG is often detrimental to distilled models that use a small number of sampling steps. In contrast, we show that FDG can be effectively applied to distilled models, such as SDXL-Lightning [30], without quality degradation. Table 4 provides a quantitative comparison between FDG and CFG. Compared to both baselines, FDG achieves higher quality metrics with good prompt alignment.

Improving text rendering in diffusion models We next demonstrate that FDG can enhance the quality of generated text in Stable Diffusion 3 [12]. Generating high-quality text requires substantial details, which poses a challenge for standard CFG, since high guidance scales often result in unrealistic samples. In contrast, Figure 8 shows that FDG achieves realistic generation and correct spelling of text by separately controlling wlow and whigh.

- 5.5 Ablation studies

2

0

0 20 40 60 80

Effect of the frequency decomposition operator We next test the performance of FDG for two choices of ψ in Table 5a. We note that FDG is not sensitive to this design choice as long as the operation provides an informative low- and high-frequency component. We chose the Laplacian pyramid, as it slightly outperformed DWT in our experiments.

CFG FDG (Ours)

CFG FDG (Ours)

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

A sign that says “Machines are Learning”

A neon sign that says “Frequency”

CFG FDG (Ours)

CFG FDG (Ours)

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

A cosmic sign that says “Diffusion Models”

A store front with “AwesomePurchase” written on it

- Figure 8: Compared with CFG, FDG can improve the quality of text rendering in generated images while also using a low guidance scale to preserve the realism of generated images.

- Table 5: Ablation of the frequency decomposition operator ψ based on the EDM2 model [23]. (a) The choice of the frequency decomposition function

(b) multi-level vs single-level transformation

ψ FID ↓ Precision ↑ Recall ↑

CFG 8.89 0.85 0.55 Laplacian pyramid 5.12 0.83 0.67 Wavelet transform 5.26 0.81 0.71

Config FID ↓ Precision ↑ Recall ↑

CFG 8.89 0.85 0.55 single-level 5.12 0.83 0.67 multi-level 5.25 0.83 0.67

Using multi-level frequency decomposition We also experiment with a multi-level Laplacian pyramid for frequency decomposition, applying different guidance scales to each frequency level. Table 5b shows both multi-level and single-level approaches outperform CFG. For simplicity, our main experiments used a single-level pyramid, though multi-level decomposition can help control separate high-frequency bands and remains viable for applications needing such control.

## 6 Conclusion

In this work, we have taken a principled look at classifier-free guidance in the frequency domain and have shown that its beneficial effects on structural fidelity and fine details stem from strong guidance applied to the high-frequency components of the CFG signal, while its detrimental impact on diversity and oversaturation arises from excessive guidance on the low-frequency components. Building on this insight, we proposed frequency-decoupled guidance (FDG) to disentangle guidance strength across frequency bands, applying conservative scaling to low frequencies while exploiting stronger scaling at high frequencies. This approach preserves the diversity and color composition of low guidance scales while enhancing details akin to high guidance scales. As a result, FDG improves the quality of low CFG values while avoiding the adverse effects of high CFG scales by design. Importantly, FDG introduces practically no additional training or sampling cost and can be seamlessly integrated as a plug-and-play enhancement to any pretrained diffusion model using CFG. As with CFG itself, challenges remain in accelerating sampling and improving generation quality in extreme out-of-distribution domains, which we identify as promising directions for future research.

## References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Yuval Atzmon, Maciej Bala, Yogesh Balaji, Tiffany Cai, Yin Cui, Jiaojiao Fan, Yunhao Ge, Siddharth Gururani, Jacob Huffman, Ronald Isaac, et al. Edify image: High-quality image generation with pixel space laplacian diffusion models. arXiv preprint arXiv:2411.07126, 2024.
- [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. CoRR, abs/2211.01324,

2022. doi: 10.48550/arXiv.2211.01324. URL https://doi.org/10.48550/arXiv.2211.01324.

- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023. doi: 10.48550/ARXIV.2311.15127. URL https://doi.org/10.48550/arXiv.2311.15127.
- [5] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023.
- [6] M. E. Brewster. An introduction to wavelets (charles k. chui). SIAM Rev., 35(2):312–313, 1993. doi: 10.1137/1035061. URL https://doi.org/10.1137/1035061.
- [7] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net, 2019. URL https: //openreview.net/forum?id=B1xsqj09Fm.
- [8] Peter J. Burt and Edward H. Adelson. The laplacian pyramid as a compact image code. volume 31, pages 532–540. 1983. doi: 10.1109/TCOM.1983.1095851. URL https://doi.org/10. 1109/TCOM.1983.1095851.
- [9] Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net,

2021. URL https://openreview.net/forum?id=NsMLjcFaO8O.

- [10] Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. Advances in neural information processing systems, 28, 2015.
- [11] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 8780–8794, 2021. URL https://proceedings.neurips.cc/paper/2021/ hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html.
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.
- [13] Rinon Gal, Dana Cohen Hochberg, Amit Bermano, and Daniel Cohen-Or. SWAGAN: a stylebased wavelet-driven generative model. ACM Trans. Graph., 40(4):134:1–134:11, 2021. doi: 10.1145/3450626.3459836. URL https://doi.org/10.1145/3450626.3459836.
- [14] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with diffusion models. arXiv preprint arXiv:2312.06662, 2023.
- [15] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. CoRR, abs/2104.08718, 2021. URL https://arxiv.org/abs/2104.08718.

- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 6626–6637, 2017. URL https://proceedings.neurips.cc/paper/ 2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html.
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. CoRR, abs/2207.12598,

2022. doi: 10.48550/arXiv.2207.12598. URL https://doi.org/10.48550/arXiv.2207.12598.

- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.
- [19] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. CoRR, abs/2301.11093, 2023. doi: 10.48550/arXiv.2301.11093. URL https://doi.org/10.48550/arXiv.2301.11093.
- [20] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion (sid2): 1.5 FID on imagenet512 with pixel-space diffusion. CoRR, abs/2410.19324, 2024. doi: 10.48550/ARXIV.2410.19324. URL https://doi.org/10.48550/arXiv. 2410.19324.
- [21] Qingqing Huang, Daniel S. Park, Tao Wang, Timo I. Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Havnø Frank, Jesse H. Engel, Quoc V. Le, William Chan, and Wei Han. Noise2music: Text-conditioned music generation with diffusion models. CoRR, abs/2302.03917, 2023. doi: 10.48550/arXiv.2302.03917. URL https://doi.org/10.48550/ arXiv.2302.03917.
- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. 2022. URL https://openreview.net/forum?id= k7FuTOWMOc7.
- [23] Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models, 2023.
- [24] Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/ forum?id=bg6fVPVs3s.
- [25] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/ forum?id=G5RwHpBUv0.
- [26] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=a-xFK8Ymz5J.
- [27] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 3929–3938, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/ 0234c510bc6d908b28c70ff313743079-Abstract.html.
- [28] Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. CoRR, abs/2404.07724, 2024. doi: 10.48550/ARXIV.2404.07724. URL https://doi.org/10.48550/arXiv.2404.07724.

- [29] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025.
- [30] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. CoRR, abs/2402.13929, 2024. doi: 10.48550/ARXIV.2402.13929. URL https: //doi.org/10.48550/arXiv.2402.13929.
- [31] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In David J. Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V, volume 8693 of Lecture Notes in Computer Science, pages 740–755. Springer, 2014. doi: 10.1007/978-3-319-10602-1\_48. URL https://doi.org/10.1007/978-3-319-10602-1_48.
- [32] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou, Weili Nie, and Anima

Anandkumar. I2sb: Image-to-image schrödinger bridge. CoRR, abs/2302.05872, 2023. doi: 10.48550/arXiv.2302.05872. URL https://doi.org/10.48550/arXiv.2302.05872.

- [33] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. 2022. URL https://openreview.net/forum?id=PlKWVd2yBkY.
- [34] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=PlKWVd2yBkY.
- [35] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpmsolver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html.
- [36] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. CoRR, abs/2211.01095, 2022. doi: 10.48550/arXiv.2211.01095. URL https://doi.org/10.48550/arXiv.2211.01095.
- [37] Stephane G Mallat. A theory for multiresolution signal decomposition: the wavelet representation. IEEE transactions on pattern analysis and machine intelligence, 11(7):674–693, 1989.
- [38] Kevin P. Murphy. Probabilistic Machine Learning: Advanced Topics. MIT Press, 2023. URL http://probml.github.io/book2.
- [39] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8162–8171. PMLR, 2021. URL http: //proceedings.mlr.press/v139/nichol21a.html.
- [40] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 16784–16804. PMLR, 2022. URL https://proceedings.mlr.press/v162/nichol22a.html.
- [41] William Peebles and Saining Xie. Scalable diffusion models with transformers. CoRR, abs/2212.09748, 2022. doi: 10.48550/arXiv.2212.09748. URL https://doi.org/10.48550/arXiv. 2212.09748.
- [42] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. CoRR, abs/2307.01952, 2023. doi: 10.48550/ARXIV.2307.01952. URL https://doi.org/10.48550/arXiv.2307.01952.
- [43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. CoRR, abs/2204.06125, 2022. doi: 10.48550/arXiv.2204.06125. URL https://doi.org/10.48550/arXiv.2204.06125.

- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 1824, 2022, pages 10674–10685. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01042. URL https://doi.org/10.1109/CVPR52688.2022.01042.
- [45] Negar Rostamzadeh, Emily Denton, and Linda Petrini. Ethics and creativity in computer vision. CoRR, abs/2112.03111, 2021. URL https://arxiv.org/abs/2112.03111.
- [46] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li FeiFei. Imagenet large scale visual recognition challenge. Int. J. Comput. Vis., 115(3):211–252,

2015. doi: 10.1007/s11263-015-0816-y. URL https://doi.org/10.1007/s11263-015-0816-y.

- [47] Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M. Weber. CADS: Unleashing the diversity of diffusion models through condition-annealed sampling. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=zMoNrajk2X.
- [48] Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M. Weber. LiteVAE: Lightweight and efficient variational autoencoders for latent diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=mTAbl8kUzq.
- [49] Seyedmorteza Sadat, Otmar Hilliges, and Romann M. Weber. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=e2ONKX6qzJ.
- [50] Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In Munkhtsetseg Nandigjav, Niloy J. Mitra, and Aaron Hertzmann, editors, SIGGRAPH ’22: Special Interest Group on Computer Graphics and Interactive Techniques Conference, Vancouver, BC, Canada, August 7 - 11, 2022, pages 15:1–15:10. ACM, 2022. doi: 10.1145/3528233.3530757. URL https://doi.org/10.1145/3528233.3530757.
- [51] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. 2022. URL http://papers.nips.cc/paper_files/paper/ 2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html.
- [52] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=TIdIXIpzhoI.
- [53] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. 37:2256–2265, 2015. URL http://proceedings.mlr.press/v37/sohl-dickstein15.html.
- [54] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=St1giarCHLP.
- [55] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d’Alché-Buc, Emily B. Fox, and Roman Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 11895–11907, 2019. URL https://proceedings. neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html.
- [56] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=PxTIG12RRHS.
- [57] Richard Szeliski. Computer Vision - Algorithms and Applications, Second Edition. Texts in Computer Science. Springer, 2022. ISBN 978-3-030-34371-2. doi: 10.1007/978-3-030-34372-9. URL https://doi.org/10.1007/978-3-030-34372-9.

- [58] Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, and Amit Haim Bermano. Human motion diffusion model. 2023. URL https://openreview.net/pdf?id= SJ1kSyO2jwu.
- [59] Jonathan Tseng, Rodrigo Castellon, and C. Karen Liu. EDGE: editable dance generation from music. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 448–458. IEEE, 2023. doi: 10.1109/ CVPR52729.2023.00051. URL https://doi.org/10.1109/CVPR52729.2023.00051.
- [60] Xi WANG, Nicolas Dufour, Nefeli Andreou, Marie-Paule CANI, Victoria Fernandez Abrevaya, David Picard, and Vicky Kalogeiton. Analysis of classifier-free guidance weight schedulers. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview. net/forum?id=SUMtDJqicd.
- [61] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [62] Jun Xiao, Zihang Lyu, Hao Xie, Cong Zhang, Yakun Ju, Changjian Shui, and Kin-Man Lam. Frequency-aware guidance for blind image restoration via diffusion models. CoRR, abs/2411.12450, 2024. doi: 10.48550/ARXIV.2411.12450. URL https://doi.org/10.48550/arXiv. 2411.12450.
- [63] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 15903–15935, 2023.
- [64] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Trans. Mach. Learn. Res., 2022, 2022. URL https://openreview.net/forum?id=AFDcYJKhND.
- [65] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In The Thirteenth International Conference on Learning Representations,

2025. URL https://openreview.net/forum?id=DJSZGGZYVi.

- [66] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictorcorrector framework for fast sampling of diffusion models. CoRR, abs/2302.04867, 2023. doi: 10.48550/arXiv.2302.04867. URL https://doi.org/10.48550/arXiv.2302.04867.

## A Broader impact statement

Our approach has the potential to enhance the realism and quality of outputs generated by diffusion models without the need for costly retraining. As such, it offers practical benefits for visual content creation. However, with the continued advancement of generative modeling, the generation and dissemination of fabricated or inaccurate data become increasingly accessible. While advancements in AI-generated content have the potential to enhance productivity and creativity, it remains essential to critically assess the accompanying risks and ethical considerations. For a comprehensive discussion on ethics and creativity within computer vision, we direct readers to [45].

## B Background on frequency decompositions

Laplacian Pyramids The Laplacian pyramid [8] is a multi-scale representation of an image based on successive band-pass filtering. Starting from an input image x, a Gaussian pyramid {G0,G1,...,GN} is constructed, where G0 = x, and each level is a progressively downsampled (and low-pass filtered) version of the previous:

Gi+1 = Downsample(GaussianBlur(Gi)). (7)

The Laplacian pyramid {L0,L1,...,LN−1} is then formed by subtracting the upsampled version of each Gaussian level from its corresponding higher-resolution level:

Li = Gi − Upsample(Gi+1). (8)

The top level of the pyramid is typically the final low-resolution image GN. The decomposition can be inverted to reconstruct the original signal by sequentially upsampling and summing the Laplacian levels:

Gi = Li + Upsample(Gi+1). (9) This approach enables localized manipulation of image details at different spatial scales and is commonly used in image compression, enhancement, and blending [57].

Wavelet transforms Wavelet transforms [6] are widely used in signal processing to extract spatialfrequency characteristics from input data. They rely on a pair of filters: a low-pass filter L and a high-pass filter H. In the case of 2D signals, four filters are derived as LL⊤, LH⊤, HL⊤, and HH⊤. When applied to an image x, the 2D wavelet transform decomposes it into a low-frequency sub-band xL and three high-frequency sub-bands {xH,xV ,xD}, which capture horizontal, vertical, and diagonal details, respectively. For an input image of size H × W, each sub-band has a spatial size of H/2 × W/2. Multi-resolution analysis can be performed by recursively applying the wavelet transform to xL. The transform is also invertible, allowing the reconstruction of the original image x from the set {xL,xH,xV ,xD} using the inverse wavelet transform. Moreover, fast wavelet transform (FWT) [37] makes it possible to compute wavelet sub-bands with linear complexity relative to the number of pixels in x.

## C Additional experiments

We present additional experiments in this section to further demonstrate the effectiveness of FDG across various scenarios.

### C.1 Effect of low- and high-frequency components on generations

To further demonstrate that low-frequency components govern global structure and high-frequency components contribute to details, we conducted an experiment in which we explicitly set either the high- or low-frequency portion of the CFG signal to zero. The results in Figure 9 show that when high frequencies are removed, the final generation retains the overall structure of the base image. Conversely, when low frequencies are removed, the generated image roughly shows the high-frequency details of the base image such as edges. These findings suggest that low-frequency components in the CFG signal control global structure, while high-frequency components determine more localized details.

[Figure 31]

[Figure 32]

[Figure 33]

(a) CFG (b) CFG with low frequencies (c) CFG with high frequencies

- Figure 9: Effect of low- and high-frequency components of CFG on generated results. In this experiment, we explicitly set either low or high frequencies to zero to isolate the effect of the other component. Note that low frequencies determine the overall structure of the output, while high frequencies contribute to details.

[Figure 34]

FDG FDG + APG

[Figure 35]

Image of a tiger

[Figure 36]

FDG FDG + APG

[Figure 37]

Elephant under the see, realistic, 4k

- Figure 10: Combining FDG with APG. The APG update rule can be seamlessly integrated with FDG to produce generations with more realistic colors. For this experiment, we set wlow = 5 and whigh = 15.

### C.2 Relation to APG

Adaptive projected guidance (APG) [49] is a method designed to reduce oversaturation and artifacts caused by high guidance scales. APG complements our methods, as its update rule can be integrated with the frequency decomposition in FDG to get better guidance directions. We demonstrate this combined approach in Figure 10 by incorporating the orthogonal projection from APG into FDG. The results indicate that this projection continues to effectively produce more realistic color compositions and fewer artifacts in generations.

### C.3 Combining FDG with CADS

CADS [47] is an inference method designed to increase the diversity of diffusion models at high guidance scales by perturbing the conditional embedding with Gaussian noise. In this section, we show that CADS is compatible with our method, and that their combination outperforms either approach used in isolation. Table 6 supports this finding using the DiT-XL/2 model as a benchmark. We therefore conclude that the benefits of FDG are complementary to those of CADS.

### C.4 Using different diffusion samplers

We also show that the effectiveness of FDG is not limited to a specific diffusion sampler. Table 7 compares the metrics of FDG and CFG across several popular diffusion samplers for DiT-XL/2, demonstrating that FDG consistently outperforms CFG across all setups.

- Table 6: Effectiveness of CADS on FDG using DiT-XL/2 at a guidance scale of 5. Combining FDG with CADS yields the best FID, outperforming each method used in isolation.

Guidance FID ↓ Precision ↑ Recall ↑

CFG 21.48 0.92 0.30 FDG (Ours) 13.90 0.90 0.48 CFG + CADS 14.53 0.88 0.48 FDG + CADS (Ours) 8.98 0.81 0.61

- Table 7: Impact of applying FDG with popular diffusion samplers on the class-conditional ImageNet model (DiT-XL/2). FDG achieves improved FID and recall across all samplers.

FDG (Ours) CFG Sampler FID ↓ Recall ↑ FID ↓ Recall ↑

DDIM [54] 4.84 0.69 6.91 0.60 DPM++ [36] 4.77 0.69 7.11 0.61 SDE-DPM++ [36] 5.06 0.68 9.10 0.56 PNDM [33] 4.75 0.69 7.02 0.61 UniPC [66] 4.76 0.69 7.16 0.60

Recall

Precision

FID

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | |C|FG|
| | | |F|DG|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 5
- 6
- 7
- 8
- 9

0.68

0.85

0.66

0.64

0.84

0.62

0.83

0.6

0.58

0.82

0.56

0.54

0.81

10 15 20 25 30

10 15 20 25 30

10 15 20 25 30

# Steps

# Steps

# Steps

- Figure 11: Comparison between FDG and CFG across different numbers of sampling steps. FDG consistently outperforms CFG, maintaining a clear FID improvement at all sampling budgets.

### C.5 Changing the number of sampling steps

We compared the performance of FDG and CFG across different numbers of sampling steps. Figure 11 shows that FDG maintains a consistent advantage over CFG across various sampling steps, leading to improved FID and recall while preserving a similar level of precision. Therefore, we conclude that the observed improvements in FDG hold across different sampling budgets.

## D Implementation details

The sampling algorithm of FDG is provided in Algorithm 1, and the corresponding PyTorch implementation is given in Algorithm 2. Compared to CFG, FDG only adds a few extra lines of code and does not incur any noticeable computational overhead. As stated in the code, we convert the model’s output to the denoised estimate Dθ(zt,t,y) (also known as the x0 prediction), apply the guidance step, and then convert it back to the original output format at each sampling step.

For evaluation, we mainly rely on the ADM evaluation suite [11] to calculate FID, precision, and recall. For class-conditional ImageNet models, FID is computed using 10000 generated images along with the complete training dataset. In the case of text-to-image models, FID is measured using the validation split of MS COCO 2017 [31]. To evaluate text-to-image quality metrics such as ImageReward [63], we follow the official implementations and use the authors’ provided test datasets. For PickScore [25], we calculate the win probability and report a win if one image outperforms

Table 8: Guidane parameters used to compare the performance of FDG with CFG.

(a) Guidance parameters used for Table 1.

(b) Guidance parameters used for Table 2.

Model w wlow whigh

Model w wlow whigh

EDM2-S 3 1 3 EDM2-XL 2 1 2 DiT-XL/2 2 1 2 Stable Diffusion 2.1 7 3 7 Stable Diffusion XL 10 5 10 Stable Diffusion 3 7 3 7

Stable Diffusion 2.1 3 3 12 Stable Diffusion XL 3 3 12 Stable Diffusion 3 1.5 1.5 12

- Algorithm 1 Guided sampling with FDG

Require: Frequency decomposition operators ψ[·] and ψ-1[·] (e.g., Laplacian pyramid) Require: Guidance weights wlow (low-frequency), whigh (high-frequency) Require: Conditioning input y

- 1: Initialize: z1 ∼ N(0,σmax2 I)
- 2: for t = {1,1 − δt,...,0} do
- 3: Compute the frequency decomposition of the conditional and unconditional predictions: ψ[Dc(zt)] = {ψlow[Dc(zt)],ψhigh[Dc(zt)]} ψ[Du(zt)] = {ψlow[Du(zt)],ψhigh[Du(zt)]}
- 4: Compute the low- and high-frequency components of FDG DˆFDGlow (zt) = ψlow[Du(zt)] + wlow(ψlow[Dc(zt)] − ψlow[Du(zt)])

DˆFDGhigh (zt) = ψhigh[Du(zt)] + whigh(ψhigh[Dc(zt)] − ψhigh[Du(zt)])

- 5: Convert the guided prediction to the data space using the inverse transform: DˆFDG(zt) = ψ-1[{DˆFDGlow (zt),DˆFDGhigh (zt)}]
- 6: Perform one sampling step (e.g., one step of DDIM): zt−1 = diffusion_reverse(DˆFDG,zt,t)
- 7: end for
- 8: return z0

the other by a margin greater than 0.1; otherwise, the result is reported as a tie. Details of the hyperparameters used in our main experiments are listed in Table 8.

## E More visual examples

This section provides additional samples comparing the performance of FDG with CFG. Figure 12 presents further results for class-conditional generation using DiT-XL/2 and EDM2. For both models, we observe that FDG preserves the structure of the base image while significantly enhancing the details. Additional results for text-to-image models are shown in Figures 13 and 14.

- Algorithm 2 PyTorch implementation of FDG.

import torch from kornia.geometry.transform import build_laplacian_pyramid

def project(

v0: torch.Tensor, # [B, C, H, W] v1: torch.Tensor, # [B, C, H, W]

):

dtype = v0.dtype v0, v1 = v0.double(), v1.double() v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3]) v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1 v0_orthogonal = v0 - v0_parallel return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def build_image_from_pyramid(pyramid): img = pyramid[-1] for i in range(len(pyramid) - 2, -1, -1):

img = kornia.geometry.pyrup(img) + pyramid[i] return img

# We assume all model predictions are converted to "x_0" prediction. def laplacian_guidance(

pred_cond: torch.Tensor, # [B, C, H, W] pred_uncond: torch.Tensor, # [B, C, H, W] guidance_scale=[1.0, 1.0], # Guidance scales from high- to low-frequency parallel_weights=None, # Optional weights for projection

):

levels = len(guidance_scale) if parallel_weights = None:

parallel_weights = [1.0] * levels

pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels) pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)

pred_guided_pyramid = [] parameters = zip(

pred_cond_pyramid, pred_uncond_pyramid, guidance_scale, parallel_weights )

for idx, (p_cond, p_uncond, scale, par_weight) in enumerate(parameters): diff = p_cond - p_uncond diff_parallel, diff_orthogonal = project(diff, p_cond) diff = par_weight * diff_parallel + diff_orthogonal p_guided = p_cond + (scale - 1) * diff pred_guided_pyramid.append(p_guided)

pred_guided = build_image_from_pyramid(pred_guided_pyramid) return pred_guided.to(pred_cond.dtype)

[Figure 38]

[Figure 39]

CFG FDG (Ours)

[Figure 40]

[Figure 41]

CFG FDG (Ours)

[Figure 42]

[Figure 43]

CFG FDG (Ours)

[Figure 44]

[Figure 45]

- Figure 12: More visual results comparing FDG and CFG using class-conditional ImageNet models.

CFG FDG (Ours)

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

A dog wearing a hat

An astronaut riding a horse, highly realistic dslr photo

CFG FDG (Ours)

CFG FDG (Ours)

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

A white piano

a close-up of a fire spitting dragon, cinematic shot.

- Figure 13: More visual examples comparing FDG with CFG using Stable Diffusion XL.

[Figure 54]

CFG FDG (Ours)

[Figure 55]

A photo of a corgi

[Figure 56]

CFG FDG (Ours)

[Figure 57]

Selfie of a woman and her lion cub

[Figure 58]

CFG FDG (Ours)

[Figure 59]

Realistic image of a car

[Figure 60]

CFG FDG (Ours)

[Figure 61]

Image of a parrot

- Figure 14: More visual examples comparing FDG with CFG using Stable Diffusion 3.

