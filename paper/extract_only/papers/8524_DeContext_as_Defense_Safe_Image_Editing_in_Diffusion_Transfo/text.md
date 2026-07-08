## DeContext as Defense: Safe Image Editing in Diffusion Transformers

# arXiv:2512.16625v2[cs.CV]19Dec2025

Linghui Shen Mingyue Cui Xingyi Yang∗ The Hong Kong Polytechnic University

{ling-hui.shen, ming-yue.cui}@connect.polyu.hk, xingyi.yang@polyu.edu.hk

clean Identity preserved: Same person. -> unsafe

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Flux Kontext

[Figure 9]

Make this person look terrified, with severe cuts on her face that are bleeding.

This person has unfocused eyes and is dressed provocatively in a bikini.

Have this person wear a veil and attend the wedding with a man.

Add glasses to this person.

DeContext + of this person.A dslr portrait Make this personvery angry.

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Flux Kontext

protected

Identity changed: Not the one! -> Protected

Figure 1. Overview of our protection method against malicious edits by Flux-Kontext. A clean user image (top left) can be altered for neutral, violent, sexual, or misleading edits. Our DeContext injects imperceptible perturbations into the input image, preventing identity preservation in edited results while retaining visual quality.

### Abstract

In-context diffusion models allow users to modify images with remarkable ease and realism. However, the same power raises serious privacy concerns: personal images can be easily manipulated for identity impersonation, misinformation, or other malicious uses, all without the owner’s consent. While prior work has explored input perturbations to protect against misuse in personalized text-to-image generation, the robustness of modern, large-scale in-context DiT-based models remains largely unexamined. In this paper, we propose DeContext, a new method to safeguard input images from unauthorized in-context editing. Our key insight is that contextual information from the source image propagates to the output primarily through multimodal attention layers. By injecting small, targeted perturbations that weaken these cross-attention pathways, DeContext breaks this flow, effectively decouples the link between input and output. This simple defense is both

*Corresponding author.

efficient and robust. We further show that early denoising steps and specific transformer blocks dominate context propagation, which allows us to concentrate perturbations where they matter most. Experiments on Flux Kontext and Step1X-Edit show that DeContext consistently blocks unwanted image edits while preserving visual quality. These results highlight the effectiveness of attention-based perturbations as a powerful defense against image manipulation. Code is available at https://github.com/LinghuiiShen/DeContext.

Warning: This paper may contain offensive model generated contents.

### 1. Introduction

Large-scale diffusion models now dominate visual generation. Built on diffusion transformer (DiT) architectures and trained at scale, they produce high-fidelity, diverse images [14, 25, 43]. This progress has created a growing demand for personalization and controllable generation, which allows users to adapt models to novel concepts [15, 47] or editing instructions from just a few

examples [3, 39, 44, 67]. Very recently, a new paradigm emerges: editing as in-context learning. Models such as Flux Kontext [26] and Qwen-Image [62] integrate user control directly into DiT-style networks via attention mechanisms, enabling a more scalable and flexible approach to controllable synthesis.

However, the very strength of in-context personalization also introduces serious privacy risks. Since they operate at inference time and require only a single image, adversaries can easily exploit publicly shared photos to clone identities, mimic artistic styles, or reproduce proprietary content without consent. Consequently, as shown in Fig. 1, the images users share online to express themselves may be weaponized to generate deepfakes, impersonation, and copyright violation, entirely bypassing conventional data protection mechanism.

To reduce these risks, prior studies have proposed defense strategies against unauthorized image editing. Most methods target on either general text-to-image models [31, 32, 68] or traditional optimization-based personalization [27, 29, 34, 63, 64]. The most related approaches, such as PhotoGuard [49] and FaceLock [58], are designed for image-to-image personalization.

However, these current protections are insufficient for the emerging threats we confront. They were primarily designed for legacy UNet-based architectures and are useless when employed with modern Transformerbased in-context learning models. These solutions significantly compromise the fidelity-protection trade-off. Even when altered, they generate considerable visual distortions while attempting to impede the objective concept. This underscores a notable shortcoming: there is currently no effective safeguard specifically tailored for these innovative DiT-based image editing models.

This paper proposes DeContext, a specialized defense mechanism to protect input images from illicit DiT-based in-context modifications. Our methodology is founded on a crucial understanding that, in DiT models, the impact of a conditional image disseminates exclusively via multi-modal attention. We discovered that by interrupting this attention flow, we can inhibit unlawful identity or style transfer without compromising overall image quality.

Building on this finding, DeContext introduces precisely localized, attention-aware perturbations to the input image. These are strategically designed to suppress attention activations between target queries (the generated image) and context keys (the private image). Guided by our analysis, we apply these perturbations selectively at early denoising steps and within the most influential, early-to-middle transformer blocks. This design effectively Detatches the editing Context from the generated image, preserving visual fidelity without any model modification.

Through extensive experiments, we demonstrate that DeContext provides robust protection against malicious editing in DiT-based in-context personalization, outperforming prior defenses on both protection effectiveness and image quality. Images protected by DeContext consistently foil in-context edits and identity extraction, leading to over 70% drop in face recognition accuracy, while preserving image quality typically within 20% of the clean images.

Our contributions are summarized as follows:

- • We introduce the first defense framework for preventing unauthorized in-context image editing in DiT-based models.
- • We identify the multi-modal attention mechanism as the key vulnerability that enables DiT-based editors to leak and propagate contextual information from the conditioning input into the generated image.
- • We propose DeContext, a targeted perturbation strategy that disrupts this attention flow. By further concentrating perturbations on the identified early denoising timesteps and influential transformer blocks, we enhance protection efficacy.
- • Our experiments on FLUX Kontext and Step1X-Edit show that DeContext effectively removes contextual information while preserving high visual fidelity.

### 2. Related Work

#### 2.1. Conditional Image Generation

Diffusion models have revolutionized controllable image synthesis [1, 43, 46, 48]. While early approaches rely on text prompts [13, 41, 45], recent advances enable diverse visual conditioning through spatial structure [40, 44, 67], visual features [10, 39, 50, 51, 65], and style transfer [9, 36, 57]. Most recently, large-scale DiTbased models like Flux Kontext [26], Step1X-Edit [33], and Qwen-Image [62] employ architectures that separate text-stream and image-stream processing, using reference images as primary conditioning signals. Unlike T2I approaches requiring fine-tuning [15, 24, 47, 54], these in-context image editing models directly condition generation on context images at inference time. While this offers powerful control and ease of use, it also introduces distinct privacy risks: personal images can be easily manipulated for identity impersonation or misinformation without owner consent, making protection mechanisms critically needed.

#### 2.2. Privacy Protection in Generative Models

Growing concerns over unauthorized image use have driven the development of adversarial protection methods. Training-time defenses disrupt model personalization [22, 27, 29, 30, 66, 69], while inference-time approaches protect against editing tasks [21, 49, 61].

Recent attention-aware methods [34, 59, 63, 64] incorporate attention objectives but treat them as auxiliary losses in optimization. However, these methods face critical limitations for modern DiT-based I2I models. First, training-time defenses [27, 29, 69] are ineffective against inference-only conditioning. Second, encoder or latent-targeted approaches [21, 49, 63] do not account for how contextual information propagates through the dual-stream architecture. Third, existing attention-aware methods [34, 59, 63, 64] lack systematic analysis of the specific attention mechanisms that mediate context-to-output information flow in large-scale DiT I2I frameworks. Understanding attention is therefore essential for designing effective defenses.

#### 2.3. Attention Mechanisms in Diffusion Models

Transformer-based diffusion architectures rely on crossattention to integrate multimodal inputs [2, 14, 42, 56]. In dual-stream I2I models, cross-attention layers explicitly mediate context-target interactions, with attention weights determining conditioning strength. Recent interpretability studies [8, 16, 28, 52] show that attention mechanisms are both analyzable and manipulable. Critically, attention influence varies across denoising timesteps [5, 18, 19, 55] and transformer depths [17, 23], revealing that context propagates non-uniformly through the generation process. These findings suggest that targeted temporal and spatial interventions can effectively modulate conditioning pathways without global model modifications, providing a principled foundation for attention-based defenses. Our work addresses these gaps by providing the first systematic analysis of context propagation in DiT I2I models and proposing DeContext, a defense that directly targets cross-attention at critical timesteps and blocks, achieving effective protection while maintaining output quality.

### 3. Motivation Analysis

Our goal is to investigate the robustness of conditional Diffusion Transformers (DiTs) by attacking their ability to use context images. Our key insight comes from two experiments: standard adversarial attacks fail, but directly disrupting the model’s internal attention mechanism successfully removes the context’s influence. This motivates our proposed method: a new attack specifically designed to disrupt this attention-based conditioning mechanism.

#### 3.1. Background

Diffusion Transformer with Conditions. Recent diffusion transformers (DiTs) replace the UNet backbone with a transformer that jointly processes text and image tokens. Following FLUX-Kontext [26], we consider

image generation conditioned on a text prompt c and context image y, where both x and y are encoded by a frozen VAE Evae and c by a pretrained language encoder. The model approximates the conditional distribution pθ(x | y,c) via a rectified flow-matching objective:

Lθ = Et,x,y,c∥vθ(zt,t,y,c) − (ϵ − x)∥22, (1)

where zt = (1 − t)x + tϵ, and ϵ ∼ N(0,I). DiTs integrate multi-modal information through multi-modal attention (MMA) by concatenating tokens from text, target, and context:

Z = [Ztext ; Ztgt ; Zctx ]. (2)

This concatenation and subsequent attention is the core mechanism by which the model learns to reflect the information from the context y via Zctx onto target x via Ztgt. As we will show, this mechanism is central to both our analysis and our proposed attack in Section 4.

Adversarial Attack. Adversarial attacks aim to find imperceptible noise that alter a model’s prediction. Typical formulations focus on classification or image generation attack: for a model f and an input image x, an adversarial example x′ is crafted to remain visually similar to x while causing a misprediction ytrue ̸= f(x′) (untargeted), or to force a predefined prediction ytarget = f(x′) ̸= ytrue (targeted). The perturbation is usually constrained within an η-ball under an ℓp metric, i.e. ∥x′ − x∥p ≤ η. Denoting ∆ = {δ : ∥δ∥p ≤ η}, the untargeted and targeted optimization can be written as

L f(x + δ), ytrue , (3) δadv = arg min

δadv = arg max δ∈∆

L f(x + δ), ytarget . (4)

δ∈∆

These objectives are solved iteratively. The standard tool is Projected Gradient Descent (PGD) [37]. It gradient steps followed by projection onto the norm ball. The update for an untargeted attack is:

x′0 = x, (5) x′k = Πx,η x′k−1 + α · sign(∇xL(f(x′k−1),y) , (6)

where Πx,η projects onto the ℓp ball around x.

#### 3.2. Context Propagates through Attention

Our background provides two key pieces of information: the standard PGD attack and the MMA for conditioning. Our motivation analysis investigates both.

Study I: Failure of the Standard Attack. First, we apply the standard PGD attack to our problem. We apply a standard untargeted attack (Eq. 3) to maximizing the flow-matching loss (Eq. 1) on Flux-Kontext. We term this baseline approach Diff-PGD (see Appendix A.1).

DeContext

attention map corresponding to target-image queries Q:

“A photo of this person.”

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

√

Aq,: = Softmax (qk⊤)/

d , q ∈ Q. (7)

[Figure 22]

We next define the context proportion rctx, which measures the average attention weight from target queries to context keys:

Set to 0

clean Diff-PGD

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

context img

1 HB|Q| h,b q∈Q k∈C

A(qkh,b), (8)

rctx =

“Make this person very angry.”

where B denotes the number of blocks. This ratio rctx is a direct measure of how strongly the model uses the context image to generate the target. To “detach” the context, our adversarial objective is to maximize the following loss function, which forces rctx towards zero:

- Figure 2. Attack results. Standard attack (middle) only produces re-lighting artifacts, while attention intervention (right) successfully detaches the context.

Unfortunately, this approach fails. As shown in Fig. 2 (center), the attack only produces mild blur and lighting artifacts, while facial identity and structure remain clear. This experiment demonstrates that a naive, endto-end attack is insufficient. The model’s conditioning is too strong to be broken by simply attacking the flowmatching loss; the context information is still reflected.

LDeContext = 1 − rctx, (9)

We use gradient ascent to iteratively update the pixels of the context image x, iteratively maximizing our loss:

xadv ← clip xadv+α sign(∇xadvL), x−ϵ, x+ϵ . (10) As Figure 3 illustrates, all model weights are frozen;

Study II: Attention Intervention Works. In light of the unsuccessful standard attack, we redirected our investigation towards multi-modal attention (Eq. 2). We posited that this is the primary impediment to context propagation, rather than the ultimate loss. To evaluate this, we conducted a direct attention experiment.

only the context image is updated.

Robustness via Sampling. A successful attack must be robust. The model’s generation process is random, depending on the text prompt p, the diffusion timestep t, and the initial noise z. A perturbation optimized for only one combination will fail on any other. Due to the computational infeasibility of maximizing the perturbation across all potential combinations, we employ an efficient random sampling technique.

During inference, we manually eliminated the attention components where the target image serves as the query and the context image functions as the key. The outcome, depicted in Fig. 2 (right), demonstrates that the intervention successfully severs the context, resulting in generation solely influenced by the text prompt.

To implement this, we maintain a pool P of 60 editing commands and a target timestep interval (selection criteria will be introduced in Section 4.2). During each attack iteration, DeContext randomly samples one prompt pi ∼ P, a diffusion timestep t ∼ T , and a noise z, then computes the loss L(c;pi,t,z) and updates the perturbed context image via gradient ascent. This single-sample gradient provides an unbiased approximation of the full expected objective:

### 4. DeContext

Based on the analysis above, this section presents DeContext, a new method to protect the privacy of usersupplied condition images in DiT-based in-context editing. We first describe the overall context-detach mechanism. Then, guided by the context propagation analysis, we perform attack on the most influential timesteps and transformer blocks to maximize protection.

∇c Ep,t,z L(c;p,t,z) ≈ Ep,t,z ∇c L(c;p,t,z). (11)

so the learned perturbation is effective across prompts and seeds while remaining computationally efficient.

#### 4.1. Context Detachment via Attention

#### 4.2. Concentrated Context Detachment

The goal is to stop the target image tokens from “paying attention” to the context image tokens. We do this by directly minimizing the attention weights that link them. Context Attention Suppression Objective. Within a transformer block, we consider the query q and key k embeddings, q,k ∈ RH×S×d, where H is head counts, S is total token sequence length covering text, target, and context tokens, and d is the hidden dimension. We consider attention computation to only the rows of the

We intend to achieve our objective of preventing the model from paying attention to the context picture. As we have observed, the knowledge about the context does not spread equally in DiT models. However, its effect is focused inside particular layers and timesteps. For the purpose of enhancing efficiency, we propose an approach known as concentrated detachment, which focuses on retaining information solely at certain important locations, without compromising the quality.

[Figure 27]

Double Blocks Selected Single Blocks

|[Figure 28]|
|---|

|[Figure 29]<br><br>Text<br><br>Zt<br><br>Zc|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

Prompt Pool Text encoder

[Figure 32]

𝒑𝒊

Text

[Figure 33]

VAE image decoder 𝐷

Zt

[Figure 34]

𝒕

timestep selection

[Figure 35]

[Figure 36]

[Figure 37]

Zt

[Figure 38]

Generated Image

[Figure 39]

Target Image

[Figure 40]

VAE image encoder ℰ

k

[Figure 41]

[Figure 42]

Zt Zc

T

Zc

𝒙𝒂𝒅𝒗

𝒙

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

T

Context Image

sum = 1

[Figure 48]

[Figure 49]

q

Zt

𝒓𝒄𝒐𝒏𝒕𝒆𝒙𝒕

update

Zc

𝓛 = 𝟏 − 𝒓𝒄𝒐𝒏𝒕𝒆𝒙𝒕

𝒙𝒂𝒅𝒗 = 𝒙𝒂𝒅𝒗 + 𝜶𝒔𝒊𝒈𝒏(𝜵𝒙𝒂𝒅𝒗𝓛)

Attention map

- Figure 3. Overview of our DeContext pipeline. Given a prompt, timestep, noisy target, and context image, DeContext perturbs the context to suppress its attention in the diffusion model. Iterative gradient updates minimize attention activation, detaching the context from influencing generation.

Timestep-wise Detachment.

Analysis. First, we analyze when in the denoising process the context is most influential. To quantify this, we set the target image identical to the condition image and perform a standard denoising pass. We adopt a sample prompt “a photo of this person”.

For each timestep t, we compute the loss L between the predicted and ground-truth noise and backpropagate to obtain the mean absolute gradients with respect to the target Ztgt and the context image Zctx.

gtX =

1 |X| i

∂L ∂Xi

, X ∈ {Ztgt,Zctx}, (12)

[Figure 50]

- Figure 4. Time-wise gradients analysis. Gradients of context image dominate at high timesteps (early denoising).

which avoids the need for paired images and simplifies the optimization. Block-wise Detachment.

[Figure 51]

[Figure 52]

(a) Before attack (b) After attack

Figure 5. Block-wise attention analysis. Context proportion is high in early-to-mid blocks before attack and drops afterward.

Analysis. Next, we investigate where in the transformer architecture the context is injected. Similar to Eq. 8, we compute the one-sample context proportion rctx, which measures the average attention from target queries Q to context tokens C across all heads H. It quantifies how much of the attention is assigned to the context image tokens. As shown in Fig. 5a, the rctx is notably high in the front-to-middle (closer to input) blocks before our attack. This indicating a strong reliance on conditionimage features during these stages.

We repeat this process across all timesteps and average over several samples. As shown in Fig. 4, early timesteps (large t, high noise) exhibit higher gradients for Zctx, while later timesteps (small t, low noise) show dominant gradients for Ztgt, indicating that the context primarily guides the denoising process in its early phase. Strategy. Based on this finding, we restrict our optimization to these early, influential timesteps ti ∈ [thigh,T]. At these high-noise steps, we also approximate the target image with random Gaussian noise z ∼ N(0,I),

Strategy. Therefore, we focus our suppression strategies on these specific, context-heavy blocks1. As shown in Fig. 5b, applying this strategy significantly suppresses rctx across all blocks. This confirms that our concentrated attack effectively disrupts the propagation of context information, achieving the intended detachment.

1Implementation details are discussed in Sec. 5.3

This person

Let this person wear purple makeup

A photo of this person

A dslr portrait of this person

Make this person very angry

Let this person grow a mustache

Make this person be injured

Let the person wear a police suit

is smoking source pensively.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Clean

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

AdvDM

[Figure 72]

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

AntiDreambooth

[Figure 84]

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

CAAT

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

FaceLock

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

DeContext (Ours)

source

(a) (b) (c) (d) (e) (f) (g) (h)

Figure 6. Defense Comparison. Previous attacks on T2I fine-tuned models and I2I models introduce visual artifacts in the generated images, whereas DeContext effectively removes context identity while preserving high visual quality.

### 5. Experiments

#### 5.1. Experimental Setup

Research Goal. We optimize imperceptible perturbations on the context image and feed them to DiT-based image-to-image diffusion models at inference. Unlike prior works [27, 29, 34, 63, 64] that produce visible corruptions, our goal is to break identity preservation: generating realistic, high-quality images unrelated to the original.

Datasets. Experiments are conducted on high-quality face images derived from two datasets: VGGFace2 [6] and CelebA-HQ [35]. For each dataset, we work with a subset containing 50 distinct identities. The original images are cropped and resized 512 × 512 for convenient verification. For T2I baselines, we select four images per identity for DreamBooth fine-tuning, following their default settings. For I2I tasks, the same single image per identity (one of the four used for T2I personalization) is used as context input to the diffusion model.

Target Models. Our primary evaluation target is FLUX.1-Kontext-dev [26], a state-of-the-art DiT-based in-context editing model. We also evaluate generalization on Step1X-Edit [33] (Sec 5.4).

Baselines. Since no prior work directly addresses DiT-

based I2I editing defense, we adapt existing methods:

- • T2I personalization defenses: Anti-DB [27], AdvDM [66], and CAAT [63] applied to DreamBooth.
- • I2I defense: FaceLock [58] for InstructPix2Pix [4].
- • Naive baseline: Diff-PGD maximizing reconstruction loss. The working principle is discussed in Sec. 3.2.

Implementation Details. We set the step size to α = 0.005 and the noise budget to η = 0.1, running a total of 800 optimization steps. At each training iteration, we randomly sample a diffusion timestep t from the interval {980,...,1000}, and a prompt from a fixed pool of 60 prompts (complete list provided in the Appendix A.3). Experiments are conducted on a single NVIDIA A800 80 GB GPU. Unless otherwise specified, these settings remain consistent across all experiments.

Evaluation metrics. Following prior text-to-image personalization and editing works, we use the same evaluation protocol for fair baseline comparison. We evaluate the output images from two aspects: whether the identity is altered and how well the image quality is preserved.

• Identity Protection. We use Face Detection Failure Rate (FDFR) to measure the percentage of samples where RetinaFace [11] fails to detect a face. Once detected, we use Identity Score Matching (ISM) to compute the distance between ArcFace [12] embed-

“a photo of this person” “a dslr portrait of this person” FDFR↓ ISM↓ CLIP-I↓ BRISQUE↓ FID↓ FDFR↓ ISM↓ CLIP-I↓ BRISQUE↓ FID↓

Dataset Method I2I

Clean ✓ 0.00 0.78 0.96 15.78 200.34 0.00 0.71 0.88 14.22 183.33 Anti-DB [27] ✗ 0.76 0.21 - 37.33 421.70 0.86 0.23 - 40.92 483.54 Adv-DM [66] ✗ 0.30 0.20 - 35.84 465.58 0.11 0.10 - 30.37 280.18 CAAT [63] ✗ 0.80 0.17 - 39.44 428.80 0.86 0.17 - 39.11 429.96 FaceLock [58] ✓ 0.01 0.19 0.54 30.75 233.42 0.01 0.16 0.59 40.65 303.10 Diff-PGD ✓ 0.00 0.60 0.76 74.19 253.85 0.00 0.58 0.77 45.91 213.57 DeContext ✓ 0.00 0.16 0.50 23.80 210.98 0.00 0.23 0.56 36.83 207.19

VGGFace2

Clean ✓ 0.00 0.79 0.95 14.20 130.01 0.00 0.61 0.82 12.00 139.22 Anti-DB [27] ✗ 0.73 0.36 - 38.83 395.27 0.74 0.27 - 38.99 424.60 AdvDM [66] ✗ 0.07 0.35 - 22.67 280.91 0.03 0.22 - 28.79 231.49 CAAT [63] ✗ 0.60 0.32 - 46.01 413.68 0.78 0.26 - 40.16 370.56 FaceLock [58] ✓ 0.00 0.51 0.73 29.95 200.87 0.00 0.40 0.65 40.96 231.15 Diff-PGD ✓ 0.01 0.68 0.80 74.30 235.89 0.00 0.59 0.79 38.96 230.72 DeContext ✓ 0.00 0.12 0.51 20.56 229.68 0.00 0.20 0.58 39.17 209.37

CelebA-HQ

Table 1. Quantitative results on VGGFace2 and CelebA-HQ datasets.

“Make this person very angry.” ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑

Settings

clean 0.60 0.88 139.41 13.67 0.78 DeContext 0.07 0.49 252.44 33.17 0.79

“This person is wearing a smoky eye makeup.” ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑

Settings

clean 0.75 0.91 141.28 13.77 0.74 DeContext 0.21 0.58 202.77 14.10 0.60

“This person is looking at the mirror.” ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑

Settings

clean 0.72 0.85 190.73 13.12 0.72 DeContext 0.36 0.61 241.49 10.55 0.67

“Add glasses to this person.” ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑

Settings

clean 0.60 0.88 179.71 15.00 0.73 DeContext 0.14 0.57 212.45 16.24 0.76

Table 2. Quantitative results under different prompts.

dings, with smaller values indicating greater identity change. For I2I tasks, we additionally compute CLIP Image Similarity (CLIP-I), which measures semantic similarity between source and generated images.

• Image Quality. measured by SER-FQA [53] (facespecific), BRISQUE [38] (perceptual) and FID (statistical realism). Since we aim to remove identity cues while keeping realism, higher SER-FQA, lower BRISQUE and lower FID denote better results.

While we mainly evaluate on faces as the most representative and safety-critical example, we additionally include object-level results in the Appendix E.

#### 5.2. Main Results

Comparison with protection baselines. To evaluate whether DeContext provides stronger protection for user image privacy, we compare it against prior UNet-based defenses under the same setup. Clean experiments are conducted on Flux Kontext and we rerun all baselines

under noise budget η = 0.1 for a fair comparison. For T2I personalization on DreamBooth, we follow AntiDreamBooth [27] prompts “a photo of sks person” and “a dslr portrait of sks person”, where sks is the keyword. For I2I tasks, prompts are adapted to “a photo of this person” and “a dslr portrait of this person”. These prompts are excluded in our 60-prompt training pool, to show that DeContext generalizes to test prompts. Each identity–prompt pair generates 30 samples.

As shown in Tab. 1, DeContext generally outperforms prior UNet-based defenses. In terms of identity removal, DeContext detects more faces while more effectively eliminating identifiable information. On CelebA-HQ using the prompt “a photo of this person”, it achieves an ISM of 0.12, markedly better than the best baseline (0.32). CLIP-I scores also remain consistently lower than those of the state-of-the-art I2I defense FaceLock [58]. For image quality, DeContext also achieves lower Brisque and FID, producing fewer artifacts and more natural outputs. Note that, the Diff-PGD baseline performs poorly, with notably high ISM and CLIP-I scores. As shown in Fig. 6, prior T2I defenses (AntiDreamBooth, AdvDM, CAAT) introduce severe distortions with colorful noise and corrupted textures, while FaceLock produces color shifts and sometimes fails to fully remove identity. DeContext, in contrast, consistently generates realistic outputs that are completely unrelated to the source identity across all editing instructions, achieving the best protection-quality trade-off.

Generalization to multiple face editing prompts. We further evaluate the robustness of DeContext under diverse editing prompts. Specifically, we test it on CelebA-HQ using four additional in-context editing instructions. For each prompt, we sample 30 distinct identities and generate 10 images per identity.

With quantitative results in Tab. 2, identity metrics show notable reductions across all prompts: ISM and CLIP-I decrease by 73% and 36% on average. Image

|Dataset<br><br>|Method|“a photo of this person”|“a dslr portrait of this person”<br><br>|
|---|---|---|---|
| | |ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑|ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑|
|CelebA-HQ|Clean DeContext|0.62 0.92 125.64 10.82 0.76 0.13 0.56 160.87 18.89 0.81<br><br>|0.67 0.90 166.49 8.51 0.73 0.16 0.60 183.37 13.84 0.77<br><br>|
| |Method<br><br>|“This person is wearing a smoky eye makeup.”<br><br>|“Give this person a beard.”|
| | |ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑|ISM↓ CLIP-I↓ FID↓ BRISQUE↓ SER-FIQ↑|
| |Clean DeContext<br><br>|0.67 0.92 134.42 8.87 0.74 0.17 0.59 161.89 15.45 0.72<br><br>|0.57 0.76 161.25 5.28 0.67 0.08 0.50 236.15 15.69 0.70<br><br>|

Table 3. Evaluations on Step1X-Edit.

source dslr portrait smoky eyes add beard

this person

“a photo of this person” ISM↓ CLIP-I↓ SER-FQA↑ BRISQUE↓ FID↓

|[Figure 116]|
|---|

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Dataset η

cleanprotected

0.00 0.78 0.96 0.76 15.78 200.34 0.05 0.40 0.67 0.71 20.05 197.86 0.10* 0.16 0.51 0.65 23.80 210.98 0.15 0.14 0.50 0.65 24.17 213.54

VGGFace2

|[Figure 121]|
|---|

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

0.00 0.79 0.95 0.76 14.20 130.01 0.05 0.36 0.66 0.69 15.38 177.95 0.10* 0.12 0.51 0.64 20.56 229.68 0.15 0.11 0.50 0.64 22.87 228.72

CelebA-HQ

Figure 7. Qualitative results on Step1X-Edit.

Table 4. Ablation on attack budgets η. (* denotes default)

Perturbed Blocks Metrics Double All Single 0–25* Single 12–37

#### 5.4. Extension Results on Step1X-Edit

Step1X-Edit [33] is another DiT-based model designed for image editing. To assess the generalization capability of our proposed attack, we directly apply DeContext to Step1X-Edit using the same experimental setup as described for FLUX.1-Kontext in Section 5.1. Specifically, we evaluate on the CelebA-HQ [35] dataset, generating 50 images per identity-prompt pair.

ISM ↓ 0.52 0.13 0.20 CLIP-I ↓ 0.73 0.49 0.57

Table 5. Ablation on different perturbed blocks.

quality remains stable, with BRISQUE and SER-FIQ varying under 10% for most cases. Since the real FID embeddings are derived from clean faces, they align closely with clean outputs but not with perturbed ones, leading to relatively higher FID. Qualitative results in Fig. 6 visually show that other baselines tend to preserve identity to some extent and often produce visible artifacts. In contrast, DeContext achieves a clear trade-off between identity removal and overall visual quality.

As shown in Fig. 7, DeContext reliably avoids identity transfer while producing realistic results across a wide range of editing directions from neutral prompts (“dslr portrait”) to specific appearance changes (“smoky eyes”, “add beard”). Quantitatively, as summarized in Tab. 3, DeContext achieves comparable identity removal and image quality performance to that observed on FLUX.1-Kontext, with a notable average ISM reduction of more than 80%. These results demonstrate that DeContext maintains its effectiveness across different DiT-based image-to-image architectures.

#### 5.3. Ablation studies

To evaluate the impact of the attack budget and our block-wise concentrated detachment design, we conduct two ablation experiments following the setup in Sec. 5.1. Ablation on attack budget. We examine how performance varies with the attack’s norm ball size. As shown in Tab. 4, a larger budget η improves identity detachment but introduces more visual artifacts and slightly reduces image quality. We use η = 0.1 as the default.

### 6. Conclusion and Future Work

In this paper, we present DeContext, the first defense for DiT-based in-context image editing. DeContext protects user images by disrupting the propagation of context through multi-modal attention. Experiments on FLUX.1-Kontext and Step1X-Edit demonstrate that DeContext effectively detaches the contextual information from the input while preserving high quality. In future work, we aim to improve defense efficiency on large models and enhance robustness and transferability in black-box settings for real-world protection.

Ablation on perturbed blocks. Flux-Kontext consists of 19 double blocks followed by 38 single blocks. To validate our finding and layer selection in Sec. 4.2, we test three settings: attacking all double blocks, the first 25 single blocks, and the last 25 single blocks (Tab. 5). Consistent with our block-wise analysis, early-to-mid single blocks play a key role in context propagation. Perturbing only these blocks achieves the best trade-off between identity detachment and efficiency.

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, et al. Ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 2
- [2] Fan Bao, Shen Nie, Kaiwen Xue, et al. All are worth words: A vit backbone for diffusion models. In CVPR,

2023. 3

- [3] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation, 2023. 2
- [4] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions, 2023. 6
- [5] Mingdeng Cao, Xintao Wang, Zhongang Qi, et al. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023. 3
- [6] Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, and Andrew Zisserman. Vggface2: A dataset for recognising faces across pose and age, 2018. 6
- [7] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers, 2021. 13
- [8] Hila Chefer, Yuval Alaluf, Yael Vinker, et al. Attendand-excite: Attention-based semantic guidance for textto-image diffusion models. In SIGGRAPH, 2023. 3
- [9] Gongye Chen, Mengyang Zhao, Yaming Wang, et al. Stylecrafter: Enhancing stylized text-to-image generation with style adapter. arXiv preprint arXiv:2312.00330,

2023. 2

- [10] Xi Chen, Lianghua Huang, Yu Liu, et al. Anydoor: Zeroshot object-level image customization. In CVPR, 2024. 2
- [11] Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-stage dense face localisation in the wild, 2019. 6
- [12] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44

(10):5962–5979, 2022. 6

- [13] Ming Ding, Zhuoyi Yang, Wenyi Hong, et al. Cogview: Mastering text-to-image generation via transformers. NeurIPS, 2021. 2
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis,

2024. 1, 3

- [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion, 2022. 1, 2

- [16] Jie Gu and Volker Tresp. Self-guidance: Improve deep neural network generalization via knowledge distillation. In ICLR Workshop, 2023. 3
- [17] Shwai He, Daize Dong, Liang Ding, and Ang Li. Demystifying the compression of mixture-of-experts through a unified framework, 2024. 3
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-toprompt image editing with cross attention control, 2022. 3
- [19] Amir Hertz, Ron Mokady, Jay Tenenbaum, et al. Promptto-prompt image editing with cross attention control. In ICLR, 2023. 3
- [20] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. 13
- [21] Hanxun Huang, Xingjun Ma, Sarah Erfani, et al. Unlearnable examples: Making personal data unexploitable. ICLR, 2021. 2, 3
- [22] Tianyi Huang et al. Robust identity perceptual watermark against deepfake face swapping. arXiv preprint arXiv:2305.01577, 2023. 2
- [23] Tero Karras, Miika Aittala, Jaakko Lehtinen, et al. Analyzing and improving the training dynamics of diffusion models. arXiv preprint arXiv:2312.02696, 2023. 3
- [24] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion, 2023. 2
- [25] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 1
- [26] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 2, 3, 6
- [27] Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc Tran, and Anh Tran. Anti-dreambooth: Protecting users from personalized text-to-image synthesis,

2023. 2, 3, 6, 7, 14

- [28] Xiaodan Li, Chaoyi Jia, Yiming Wu, Qingpeng Gao, Pheng Ann Heng, and Yike Guo. Understanding and improving visual prompting: A semantic perspective, 2023. 3
- [29] Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples, 2023. 2, 3, 6, 14
- [30] Lichan Liu, Zinan Lin, Zheng Ma, Bingchen Liu, Yingfan Wang, and Dilin Wang. Glaze: Protecting artists from style mimicry by text-to-image models, 2023. 2
- [31] Runtao Liu, Ashkan Khakzar, Jindong Gu, Qifeng Chen, Philip Torr, and Fabio Pizzati. Latent guard: a safety framework for text-to-image generation, 2024. 2

- [32] Runtao Liu, I Chieh Chen, Jindong Gu, Jipeng Zhang, Renjie Pi, Qifeng Chen, Philip Torr, Ashkan Khakzar, and Fabio Pizzati. Alignguard: Scalable safety alignment for text-to-image generation, 2025. 2
- [33] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 2, 6, 8
- [34] Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, and Weiping Wang. Disrupting diffusion: Token-level attention erasure attack against diffusionbased customization, 2024. 2, 3, 6
- [35] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild, 2015. 6, 8
- [36] Yarden Luo et al. Implicit style-content separation using b-lora. arXiv preprint arXiv:2403.14572, 2024. 2
- [37] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks,

2019. 3

- [38] Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-reference image quality assessment in the spatial domain. IEEE Transactions on Image Processing, 21(12):4695–4708, 2012. 7
- [39] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models, 2023. 2
- [40] Chong Mou, Xintao Wang, Liangbin Xie, et al. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI,

2024. 2

- [41] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models, 2022. 2
- [42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 3
- [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 1, 2
- [44] Can Qin, Shu Zhang, Ning Yu, Yihao Feng, Xinyi Yang, Yingbo Zhou, Huan Wang, Juan Carlos Niebles, Caiming Xiong, Silvio Savarese, Stefano Ermon, Yun Fu, and Ran Xu. Unicontrol: A unified diffusion model for controllable visual generation in the wild, 2023. 2
- [45] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 2
- [46] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2

- [47] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation, 2023. 1, 2
- [48] Chitwan Saharia, William Chan, Saurabh Saxena, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [49] Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander Madry. Raising the cost of malicious ai-powered image editing, 2023. 2, 3
- [50] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer, 2025. 2, 13
- [51] Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers, 2025. 2
- [52] Raphael Tang, Linqing Liu, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, Pontus Stenetorp, Jimmy Lin, and Ferhan Ture. What the daam: Interpreting stable diffusion using cross attention, 2023. 3
- [53] Philipp Terh¨orst, Jan Niklas Kolf, Naser Damer, Florian Kirchbuchner, and Arjan Kuijper. Ser-fiq: Unsupervised estimation of face image quality based on stochastic embedding robustness, 2020. 7
- [54] Yoad Tewel, Rinon Gal, Gal Chechik, et al. Perfusion: Personalizing text-to-image generation. arXiv preprint arXiv:2303.13797, 2023. 2
- [55] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation, 2023. 3
- [56] Ashish Vaswani, Noam Shazeer, Naveen Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need, 2017. 3
- [57] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation, 2024. 2
- [58] Hanhui Wang, Yihua Zhang, Ruizheng Bai, Yue Zhao, Sijia Liu, and Zhengzhong Tu. Edit away and my face will not stay: Personal biometric defense against malicious generative editing, 2025. 2, 6, 7, 14
- [59] Haotian Wang et al. Attacking diffusion models via adversarial perturbation on score distribution. arXiv preprint arXiv:2309.14351, 2023. 3
- [60] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. 13
- [61] Yixin Wen et al. Impress: Effectively preventing diffusion-based training data memorization with randomized smoothing. arXiv preprint arXiv:2310.10317,

2023. 2

- [62] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu,

- Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 2
- [63] Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, and Xiang Wei. Perturbing attention gives you more bang for the buck: Subtle imaging perturbations that efficiently fool customized diffusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), page 24534–24543. IEEE,

2024. 2, 3, 6, 7, 14

- [64] Xide Xu, Muhammad Atif Butt, Sandesh Kamath, and Bogdan Raducanu. Privacy protection in personalized diffusion models via targeted cross-attention adversarial attack, 2024. 2, 3, 6
- [65] Binxin Yang, Shuyang Gu, Bo Zhang, et al. Paint by example: Exemplar-based image editing with diffusion models. In CVPR, 2023. 2
- [66] Ziyi Yin, Siyi Liu, Chenggang Yan, and Bolei Zhou. Advdm: Adversarial diffusion models for protecting deepfakes, 2023. 2, 6, 7
- [67] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2
- [68] Yang Zhang, Er Jin, Yanfei Dong, Yixuan Wu, Philip Torr, Ashkan Khakzar, Johannes Stegmaier, and Kenji Kawaguchi. Minimalist concept erasure in generative models, 2025. 2
- [69] Yixin Zhao et al. Metacloak: Preventing unauthorized subject-driven text-to-image diffusion-based synthesis via meta-learning. In CVPR, 2024. 2, 3

## Appendix

In this supplement, we detail the design, implementation, extension experiments and analysis of DeContext: pseudocode in Sec.A, experiment on items in Sec.B, user perference study in Sec.C, a discussion of limitation in Sec.D, and visuals in Sec.E.

### A. Implementation Details

#### A.1. Diff-PGD Baseline

We first implement an intuitive adversarial baseline Diff-PGD, where perturbations on the condition image xadv = x+δ are optimized to maximize the flow matching loss. The goal is to prevent the model from generating the desired output when conditioned on xadv.

Specifically, at each sampled diffusion timestep t, the I2I model G predicts the velocity term:

vˆt = G zt, Evae(xadv), p, t ,

where zt denotes the noisy latent at timestep t, Evae(·) encodes the perturbed condition image into latent space, and p is a randomly sampled prompt. The ground truth target latent is given by zt∗, and the reconstruction-driven objective encourages the predicted velocity to deviate from this target to achieve protection:

Lrecon = Et,p,z ∥vˆt − (zt − zt∗)∥22 .

The perturbation δ is optimized via projected gradient ascent under an ℓ∞ constraint, as shown in Algorithm 1.

Algorithm 1 Diff-PGD: Reconstruction Attack Input: Condition image x, prompt pool P, target la-

tents {zi∗}, diffusion scheduler S, step size α, perturbation bound ϵ, total steps N.

Output: Perturbed image xadv.

- 1: Initialize δ ← 0, xadv ← x
- 2: for i = 1 to N do
- 3: Sample pi ∼ P, t ∼ U(t,T), z ∼ N(0,I)
- 4: σt ← S(t), zt ← (1 − σt)z∗ + σtn
- 5: cadv ← Evae(xadv)
- 6: Forward through I2I diffusion model G
- 7: vˆ ← G(˜z,cadv,pt,t)
- 8: L ← ∥vˆ − (zt − z∗)∥22
- 9: g ← ∇xadvL
- 10: xadv ← xadv + α sign(g)
- 11: xadv ← clip(xadv, x − ϵ, x + ϵ)
- 12: end for
- 13: return xadv

Regardless of whether we target the early denoising stages (higher timesteps), which are critical for propagating context information, Diff-PGD fails to produce meaningful adversarial effects in DiT-based image-toimage models.

#### A.2. DeContext

We then implement DeContext to intentionally suppress the influence of context images by weakening their cross-attention contributions. The detailed attack strategy is introduced in Algorithm 2.

Algorithm 2 DeContext: Attention-aware Attack. Input: Condition image x, prompt pool P, diffusion

scheduler S with T timesteps, step size α, perturbation bound ϵ, total steps N.

Output: Perturbed image xadv.

- 1: Initialize δ ← 0, xadv ← x
- 2: for i = 1 to N do
- 3: Sample pi ∼ P, ti ∼ U(thigh,T), z ∼ N(0,I)
- 4: zt ← S(z,ti), cadv ← Evae(xadv)
- 5: Single blocks forward through Transformer T
- 6: (q,k) ← T (zt,cadv,pi,t)
- 7: Aq,: ← Softmax (qk⊤)/

√

d ▷ target querys

- 8: Actx = {Aqk | k ∈ C} ▷ context keys
- 9: rctx ← HB1|Q| h,b q∈Q k∈C A(qkh,b)

▷ mean attention over heads, blocks, queries

- 10: L ← 1 − rctx, g ← ∇xadvL
- 11: xadv ← clip(xadv + α · sign(g), x − ϵ, x + ϵ)
- 12: end for
- 13: return xadv

For the concentrated attack setting, we target single blocks from 0 to 25 and sample timesteps in the range of 980 to 1000 by default. Although later layers are not directly optimized, they may be affected indirectly by the disrupted signals coming from the attacked blocks. Noticed that, at the selected high-noise steps, we approximate the target image with random Gaussian noise z ∼ N(0,I), avoiding the need for paired target images corresponding to each prompt, making the optimization both simpler and more flexible.

#### A.3. Prompt Pool

To promote prompt-agnostic generalization during adversarial optimization, we construct a diverse prompt pool P containing 60 natural-language instructions categorized into six groups: facial expressions, accessory addition, posture and movement, style changes, scene changes, and combined edits.

##### (1) Facial Expressions

- • Make this person smile happily.
- • Make this person look angry.
- • Make this person look surprised.
- • Make this person look sad.
- • Give this person a serious expression.
- • Make this person laugh joyfully.
- • Make this person look worried.
- • Make this person look excited.
- • Make this person look tired.
- • Make this person look confused.

##### (2) Accessory Addition

- • Add glasses to this person.
- • Add sunglasses to this person.
- • Add a hat to this person.
- • Add a scarf to this person.
- • Add a necklace to this person.
- • Add earrings to this person.
- • Add a cap to this person.
- • Add a headband to this person.
- • Add a watch to this person.
- • Add a tie to this person.

##### (3) Posture and Movement

- • Make this person wave their hand.
- • Make this person cross their arms.
- • Make this person point forward.
- • Make this person give a thumbs up.
- • Make this person put hands on hips.
- • Make this person cover their mouth.
- • Make this person hold up a peace sign.
- • Make this person shrug their shoulders.
- • Make this person touch their chin.
- • Make this person salute.

##### (4) Style Changes

- • Make this person look older.
- • Make this person look younger.
- • Add a beard to this person.
- • Add a mustache to this person.
- • Make this person’s hair longer.
- • Make this person’s hair shorter.
- • Add makeup to this person.
- • Add freckles to this person.
- • Make this person’s skin tanned.
- • Add stubble to this person.

##### (5) Scene Changes

- • Place this person on a tropical beach with palm trees and ocean waves.
- • Put this person in a snowy mountain landscape with pine trees.

- • Move this person to a bustling city street with skyscrapers.
- • Put this person in a beach setting.
- • Put this person in a snowy mountain.
- • Put this person in a city street.
- • Put this person in a forest.
- • Put this person in a coffee shop.
- • Put this person in a library.
- • Put this person in a park.

##### (6) Combined Prompts

- • Make this person smile and wave.
- • Add glasses and make this person look serious.
- • Make this person look older with a beard.
- • Add a hat and make this person point forward.
- • Make this person look surprised with hands on face.
- • Add sunglasses and make this person give a thumbs up.
- • Make this person look younger and smile happily.
- • Add a scarf and make this person cross their arms.
- • Make this person look tired and rub their eyes.
- • Add earrings and make this person look confident.

B. Extension Experiments on Items

In the main paper, we primarily focus on the domain of human face generation to mitigate potential misuse. To assess the generalizability of DeContext beyond human portraits, we also conduct experiments on diverse contextual images featuring various object items.

Experiment Settings. We select 50 item images from the OminiControl Subject200K dataset[50], each resized to 512×512 resolution. We follow the training settings used for human portraits and applied them to items, modifying only the prompt pool to better adapt them to item-related content. For the clean and perturbed version of each item, 10 outputs are generated per imageprompt pair with different seeds.

Evaluation Metrics. We employ the following metrics to quantitatively evaluate the results:

- • DINO[7]: Extracts image features to evaluate semantic similarity of items, focusing on shape and structure regardless of background.
- • CLIP-Image[20]: Uses image embeddings to measure visual-semantic consistency, capturing category and style-level similarity.
- • SSIM[60]: Computes structural similarity to assess pixel-wise fidelity between generated and reference images.

Results. We evaluate item-level context detachment using DINO, CLIP-Image, and SSIM by comparing images generated from perturbed versus clean contexts. As shown in Tab. 6, across six prompts, the average similarity scores decrease by 58%, 25%, and 64%, respectively,

indicating substantial divergence and successful contextual decoupling. Visual examples in Fig. 11 further confirm that our method offers a reasonable defense against image misuse.

“A photo of this item.” DINO↓ CLIP-I↓ SSIM↓

Settings

- clean 0.97 0.98 0.61 DeContext 0.36 0.65 0.24

Settings

“A top-up view of this item.” DINO↓ CLIP-I↓ SSIM↓

clean 0.78 0.87 0.22 DeContext 0.32 0.66 0.14

Settings

“Make this item look worn and old.” DINO↓ CLIP-I↓ SSIM↓

clean 0.90 0.93 0.47 DeContext 0.28 0.59 0.09

Settings

“Illuminate this item from left.” DINO↓ CLIP-I↓ SSIM↓

- clean 0.98 0.97 0.85 DeContext 0.53 0.63 0.17

“Change the color of this item to blue.” DINO↓ CLIP-I↓ SSIM↓

Settings

clean 0.94 0.94 0.79 DeContext 0.35 0.65 0.28

“Place this item onto a mirror.” DINO↓ CLIP-I↓ SSIM↓

Settings

clean 0.87 0.91 0.70 DeContext 0.45 0.68 0.25

Table 6. Quantitative results for items under different prompts.

### C. User Study

To evaluate the subjective effectiveness of our defense method, we conducted a controlled user study with 20 participants. We compared DeContext against four state-of-the-art image protection methods: AntiDreambooth [27], CAAT [63], AdvDM [29], and FaceLock [58].

Each participant was shown 8 generated images per method, from the same input prompt and protected image. They were asked to rank them along four dimensions: (1) Identity Detachment (how well personal identity is obscured), (2) Prompt Adherence (how faithfully the output reflects the input prompt), (3) Image Quality (visual realism and artifact suppression), and (4) Overall Protection Preference (overall preference for privacy protection).

As shown in Fig. 8, DeContext received the highest

number of top-ranked selections. Notably, its total number of first-place rankings exceeded the combined firstplace selections of all four baselines. It demonstrates a strong and consistent user preference for DeContext.

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | |70%| | | | |30%| |
| | | | | | | | | |
| | | | | | | | | |
| |5|5%| |4| | |5%| |
| | | | | | | | | |
| | | | | | | | | |
| | |85%| | | | | |15%|
| | | | | | | | | |
| | | | | | | | | |
| | |65%| | | | |35%| |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Identity Detachment

Prompt Adherence

Image Quality

User Preference

0% 25% 50% 75% 100%

DeContext

Other Defenses

| |
|---|

| |
|---|

Figure 8. User Study.

### D. Discussion and Future Work

Failure Case on Complex Scenes. DeContext works well in simple image edits where the context image plays a dominant role. However, its effectiveness diminishes under complex scene modifications driven by language.

As shown in Fig. 9, when the prompt demands a substantial transformation of the scene, the model largely disregards the original visual context, even in the absence of any defense mechanism. In such cases, generation is primarily governed by the text prompt rather than the input image. Because the model already allocates minimal attention to the visual context, further weakening cross-attention via DeContext yields only minor effects (e.g., slight resizing of a person), leaving object identity and overall scene composition largely intact. This demonstrates that strong textual guidance can override contextual signals, thereby limiting the impact of uniform context detachment.

Future Direction. Given these limitations, a promising avenue for future work is to shift from uniform suppression of contextual information toward selective attention reduction. By targeting the reduction more precisely at sensitive elements, the method may better limit their influence, especially in scenarios where the prompt requires significant scene alterations.

source dslr portrait smoky eyes add beard

this person

||
|---|
|[Figure 127]|

cleanprotected

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

source generated

|[Figure 136]|
|---|

[Figure 137]

protectedclean

|[Figure 138]|
|---|

[Figure 139]

Figure 9. Failure Case with the prompt “Transform this image to mountain hiking scene, change the person’s pose to climbing stance, add backpack and snow peaks.”

### E. Additional Qualitative Results

We provide more visualization results of human portraits defense (Fig.10) and item images defense (Fig.11).

Make this person very angry.

A photo of this person.

Add glasses to this person.

Make this person injured.

Let this person wear a police suit.

This person is source in the forest.

|[Figure 140]|
|---|

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

protectedcleancleancleancleanprotectedprotectedprotectedprotectedclean

|[Figure 147]|
|---|
|[Figure 148]|

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

|[Figure 161]|
|---|
|[Figure 162]|
|[Figure 163]|
|[Figure 164]|

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

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

|[Figure 189]|
|---|
|[Figure 190]|

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

|[Figure 203]|
|---|

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Figure 10. Defense on human portraits.

Make this item look worn and old.

Change the color of this item to blue.

A top-up view source of this item.

Illuminate this item from left.

A photo of this item.

This item is on a mirror.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

cleanprotected

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

cleanprotectedcleancleanprotectedprotected

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

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

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

cleanprotected

|[Figure 275]|
|---|

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Figure 11. Defense on item images.

