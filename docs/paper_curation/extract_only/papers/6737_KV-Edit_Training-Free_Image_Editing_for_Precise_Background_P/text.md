## KV-Edit: Training-Free Image Editing for Precise Background Preservation

Tianrui Zhu1*, Shiyi Zhang1*, Jiawei Shao2, Yansong Tang1† 1Shenzhen International Graduate School, Tsinghua University 2Institute of Artificial Intelligence (TeleAI), China Telecom

xilluill070513@gmail.com,sy-zhang23@mails.tsinghua.edu.cn shaojw2@chinatelecom.cn,tang.yansong@sz.tsinghua.edu.cn https://xilluill.github.io/projectpages/KV-Edit/

# arXiv:2502.17363v3[cs.CV]12Mar2025

mask& input edited image mask& input edited image mask& input edited image

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Removeobject

|a lion in suit … with a laptop|
|---|

|a lion in suit … with a laptop<br><br>|
|---|

|a girl and dog … in the forest|
|---|

|a girl and dog … in the forest|
|---|

|a paraglider … the mountain|
|---|

|a paraglider … the mountain<br><br>|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Addobject

|A girl wearing a straw hat …|
|---|

|a small owl with a mushroom cap|
|---|

|a small owl wears sunglasses …|
|---|

|a girl sits on a cozy couch|
|---|

|a girl … holding her cat doll|
|---|

|A girl stands in a field of flowers|
|---|

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Changeobject

|an … van with surfboards on top|
|---|

|an … van with flowers on top|
|---|

|… neon sign that reads "I love here"|
|---|

|… neon sign that reads "I love nana"|
|---|

|a cat sitting in the grass|
|---|

|a cat sitting in the rocks|
|---|

Figure 1. We propose KV-Edit to address the challenge of background preservation in image editing, thereby enhancing the practicality of AI editing. Rather than designing complex mechanisms, we achieve impressive results by simply preserving the key-value pairs of the background. Our method effectively handles common semantic editing operations, including adding, removing, and changing objects.

### Abstract

ity to the original image and generating content that aligns with the target. Here, we propose KV-Edit, a training-free approach that uses KV cache in DiTs to maintain background consistency, where background tokens are preserved rather than regenerated, eliminating the need for complex mechanisms or expensive training, ultimately generating new content that seamlessly integrates with the background

Background consistency remains a significant challenge in image editing tasks. Despite extensive developments, existing works still face a trade-off between maintaining similar-

*Equal contribution. †Corresponding author.

within user-provided regions. We further explore the memory consumption of the KV cache during editing and optimize the space complexity to O(1) using an inversion-free method. Our approach is compatible with any DiT-based generative model without additional training. Experiments demonstrate that KV-Edit significantly outperforms existing approaches in terms of both background and image quality, even surpassing training-based methods.

### 1. Introduction

Recent advances in text-to-image (T2I) generation have witnessed a significant shift from UNet [43] to DiT [39] architectures, and from diffusion models (DMs) [11, 48, 52] to flow models (FMs) [1, 23, 64]. Flow-based models, such as Flux [1], construct a straight probability flow from noise to image, enabling faster generation with fewer sampling steps and reduced training resources. DiTs [39], with their pure attention architecture, have demonstrated superior generation quality and enhanced scalability compared to UNetbased models. These T2I models [1, 14, 42] can also facilitate image editing, where target images are generated based on source images and modified text prompts.

In the field of image editing, early works [12, 16, 36, 50] proposed the inversion-denoising paradigm to generate edited images, but they struggle to maintain background consistency during editing. One popular approach is attention modification, such as HeadRouter [57] modifying attention maps and PnP [50] injecting original features during the denoising process, aiming to increase similarity with the source image. However, there remains a significant gap between improved similarity and perfect consistency, as it is challenging to control networks’ behavior as intended. Another common approach is designing new samplers [37, 38] to reduce errors during inversion. Nevertheless, errors can only be reduced but not completely eliminated and both training-free approaches above still require extensive hyperparameter tuning for different cases. Meanwhile, exciting training-based inpainting methods [26, 65] can maintain background consistency but suffer from expensive training costs and potential degradation of quality.

To overcome all the above limitations, we propose a new training-free method that preserves background consistency during editing. Instead of relying on regular attention modification or new inversion samplers for similar results, we implement KV cache in DiTs [39] to preserve the keyvalue pairs of background tokens during inversion and selectively reconstruct only the editing region. Our approach first employs a mask to decouple attention between background and foreground regions and then inverts the image into noise space while caching KV values of background tokens at each timestep and attention layer. During the subsequent denoising process, only foreground tokens are pro-

cessed, while their keys and values are concatenated with the cached background information. Effectively, we guide the generative model to maintain new content continuity with the background and keep the background content identical to the input. We call this approach KV-Edit.

To further enhance the practical utility of our approach, we conduct an analysis of the removal scenario. This challenge arises from the residual information in surrounding tokens and the object itself which sometimes conflict with the editing instruction. To address this issue, we introduce mask-guided inversion and reinitialization strategies as two enhancement techniques for inversion and denoising separately. These methods further disrupt the information stored in surrounding tokens and self tokens respectively, enabling better alignment with the text prompt. In addition, we apply KV-Edit to the inversion-free method [23, 56], which no longer caches key-value pairs for all timesteps, but uses KV immediately after one step, significantly reducing the memory consumption of the KV cache.

In summary, our key contributions include: 1) A new training-free editing method that implements KV cache in DiTs, ensuring complete background consistency during editing with minimal hyperparameter tuning. 2) Maskguided inversion and reinitialization strategies that extend the method’s applicability across various editing tasks, offering flexible choices for different user needs. 3) Using the inversion-free method to optimize the memory overhead of our method and enhance its usefulness on PC. 4) Experimental validation demonstrating perfect background preservation while maintaining generation quality comparable to direct T2I synthesis.

### 2. Related Work

#### 2.1. Text-guidanced Editing

Image editing approaches can be broadly categorized into training-based and training-free methods. Training-based methods [1, 7, 20, 22, 26], have demonstrated impressive editing capabilities through fine-tuning pre-trained generative models on text-image pairs, achieving controlled modifications. Training-free methods have emerged as a flexible alternative, with pioneering works [12, 16, 36, 50] establishing the two-stage inversion-denoising paradigm. Attention modification has become a prevalent technique in these methods [4, 9, 25, 49, 57], specially Add-it [49] broadcast features from inversion to denoising process to maintain source image similarity during editing. Some other work [21, 27, 37, 38, 53] focused on a better inversion sampler such as the RF-solver [53] designs a second-order sampler. The methods most similar to ours [3, 10, 29, 49] attempt to preserve background elements by blending source and target images at specific timesteps using masks. A common consensus is that accurate masks are crucial for better

inverse N steps

mask& input

|| |optional|
|---|---|
|optional| |
<br><br>text fg bg<br><br>textfgbg|
|---|

[Figure 19]

textfgbg

x

MM-DiT

MM-DiT

MM-DiT

MM-DiT

MM-DiT

MM-DiT

[Figure 20]

split

x

| | | | | | | |
|---|---|---|---|---|---|---|
| |KV| |KV| |KV| |

| | | | | | | |
|---|---|---|---|---|---|---|
| |KV| |KV| |KV| |

x x

x

(1) attention mask

KV

KV

KV

KV

KV

KV

“ a cat sitting on a wooden chair ”

“ a dog sitting on a wooden chair ”

x

[Figure 21]

(2) reinitialization

MM-DiT

MM-DiT

MM-DiT

MM-DiT

MM-DiT

MM-DiT

|optional<br><br>x +noise z x z<br><br>|
|---|

C

optional

z 𝑧 𝑧

denoise N steps

- Figure 2. Overview of our proposed KV-Edit. Given an input image and mask, we separate the image into foreground and background. Here, x and z denote intermediate results in inversion and denoising processes respectively. Starting from x0, we first perform inversion to obtain predicted noise xN while caching KV pairs. Then, we choose the input zfgN and generate edited foreground content zfg0 based on a new prompt. Finally, we concatenate it with the original background xbg0 to obtain the edited image with preserved background.

ability of our method across diverse scenarios.

quality, where user-provided inputs [20, 26] and segmentation models [6, 18, 34, 35, 41, 58, 59] prove to be more effective choices compared to masks derived from attention layers in UNet [43]. However, the above methods frequently encounter failure cases and struggle to maintain perfect background consistency during editing, while trainingbased methods [1, 7, 20, 22, 26, 65] face the additional challenge of computational overhead.

#### 3.1. Preliminaries

Deterministic diffusion models like DDIM [46] and flow matching [28] can be modeled using ODE [47] to describe the probability flow path from noise distribution to real distribution. The model learns to predict velocity vectors that transform Gaussian noise into meaningful images. During the denoising process, x1 represents noise, x0 is the final image, and xt represents intermediate results.

#### 2.2. KV cache in Attention Models

KV cache is a widely-adopted optimization technique in Large Language Models (LLMs) [5, 8, 30, 54] to improve the efficiency of autoregressive generation. In causal attention, since keys and values remain unchanged during generation, recomputing them leads to redundant resource consumption. KV cache addresses this by storing these intermediate results, allowing the model to reuse key-value pairs from previous tokens during inference. This technique has been successfully implemented in both LLMs [5, 8, 30, 54] and Vision Language Models (VLMs) [2, 17, 24, 31, 60– 62]. However, it has not been explored in image generation and editing tasks, primarily because image tokens are typically assumed to require bidirectional attention [13, 15].

- 1

- 2

g2(t)∇xt

dxt = f(xt,t) −

log p(xt) dt,t ∈ [0,1].

(1) where sθ(x,t) = ∇xt

log p(xt) predicted by networks. Both DDIM [46] and flow matching [28] can be viewed as special cases of this ODE function. By setting f(xt,t) =

dt , g2(t) = 2αtβtdtd β

θ(x,t)

αt , and sθ(x,t) = −ϵ

xt αt

dαt

βt , we obtain the discretized form of DDIM:

t

xt − β¯tϵθ(xt,t) α¯t

+ β¯t−1ϵθ(xt,t) (2)

xt−1 = α¯t−1

Both forward and reverse processes in ODE follow Eq. (1), describing a reversible path from Gaussian distribution to real distribution. During image editing, this ODE establishes a mapping between noise and real images, where noise can be viewed as an embedding of the image, carrying information about structure, semantics, and appearance.

### 3. Method

In this section, we first analyze the reasons why the inversion-denoising paradigm [12, 16] faces challenges in background preservation. Then, we introduce the proposed KV-Edit method, which achieves strict preservation of background regions during the editing process according to the mask. Finally, we present two optional enhancement techniques and an inversion-free version to improve the us-

Recently, Rectified Flow [32, 33] constructs a straight path between noise distribution and real distribution, training a model to fit the velocity field vθ(x,t). This process can be simply described by the ODE:

dxt = vθ(x,t)dt,t ∈ [0,1]. (3)

|MSE×𝟏𝟎|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

- 2.2

|𝑡15|
|---|

𝑡24

0.07

0.4

|𝑡28|
|---|

|𝑡0|
|---|

Step

[Figure 26]

Source Image

[Figure 27]

|𝑡15|
|---|

[Figure 28]

|𝑡24|
|---|

[Figure 29]

|𝑡28|
|---|

step N = 28 guidance = 1.5

Reconstruction error

Figure 3. The reconstruction error in the inversionreconstruction process. Starting from the original image xt0, the inversion process proceeds to xtN . During inversion process, we use intermediate images xti to reconstruct the original image and calculate the MSE between the reconstructed image x′t0 and the original image xt0.

Due to the reversible nature of ODEs, flow-based models can also be used for image editing through inversion and denoising in less timesteps than DDIM [46].

- 3.2. Rethinking the Inversion-Denoising Paradigm

edit

|a slanted rusty mountain bicycle|
|---|

|a slanted mountain bicycle|
|---|

|a slanted mountain bicycle|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

edit

|a slanted rusty mountain bicycle|
|---|

|a slanted mountain bicycle|
|---|

|a slanted mountain bicycle|
|---|

Figure 4. Analysis of factors affecting background changes. The four images on the right demonstrate how foreground content and condition changes influence the final results.

these definitions, the background denoising process is:

,ti) = vθ(C,zfgt

,zbgt

,ti) (6)

vθ(C,zt

i

i

i

The inversion-denoising paradigm views image editing as an inherent capability of generative models without additional training, capable of producing semantically different but visually similar images. However, empirical observations show that this paradigm only achieves similarity rather than perfect consistency in content, leaving a significant gap compared to users’ expectations.This section will analyze the reasons for this issue into three factors.

zbgt

= zbgt

+ (ti−1 − ti)vθ(C,zfgt

,zbgt

,ti) (7)

i−1

i

i

i

According to these formulas, when generating edited results, the background will be influenced by both the new condition C and new foreground zfgt

. Fig. 4 demonstrates that background regions change when only modifying the prompt or foreground noise. In summary, uncontrollable background changes can be attributed to three factors: error accumulation, new conditions, and new foreground content. In practice, any single element will trigger all three effects simultaneously. Therefore, this paper will present an elegant solution to address all these issues simultaneously.

i

Taking Rectified Flow [32, 33] as an example, based on Eq. (3), we can derive the discretized implementation of inversion and denoising. The model takes the original image xt

and Gaussian noise xt

N ∈ N(0,I) as path endpoints. Given discrete timesteps t = {tN,...,t0}, the model predictions vθ(C,xt

0

denote intermediate states in inversion and denoising respectively, as described by the following equations:

,ti),i ∈ {N,··· ,1}, where xt

and zt

i

i

i

#### 3.3. Attention Decoupling

Traditional inversion-denoising paradigms process background and foreground regions simultaneously during denoising, causing undesired background changes in response to foreground and condition modifications. Upon deeper analysis, we observe that in UNet [43] architectures, the extensive convolutional networks lead to the fusion of background and foreground information, making it impossible to separate them. However, in DiT [39], which primarily relies on attention blocks [51], allows us to use only foreground tokens as queries, generating foreground content separately and then combined with the background.

,ti) (4) zt

+ (ti − ti−1)vθ(C,xt

##### xt

= xt

i−1

i

i

,ti) (5) Ideally, zt

+ (ti−1 − ti)vθ(C,zt

= zt

i−1

i

i

when directly reconstructed from xt

should be identity with xt

0

0

. However, due to discretization and causality in the inversion process, we can only estimate using vθ(C,Xt

N

,ti), introducing cumulative errors. Fig. 3 shows that with a fixed number of timesteps N, error accumulation increases as inversion timesteps approach tN, preventing accurate reconstruction.

,tt−1) ≈ vθ(C,Xt

t−1

i

Moreover, directly generating foreground tokens often results in discontinuous or incorrect content relative to the background. Therefore, we propose a new attention mechanism where queries contain only foreground information, while keys and values incorporate both foreground and

In addition, consistency is affected by condition. We can divide the image into regions we wish to edit zfgt

and regions we want to preserve zbgt

0

, where “fg” and “bg” represent foreground and background respectively. Based on

0

- Algorithm 1 Simplified KV cache during inversion

- 1: Input: ti, image xt

i

, M-layer block {lj}Mj=1, foreground region mask, KV cache C

- 2: Output: Prediction vector Vθt

i

, KV cache C

- 3: for j = 0 to M do
- 4: Q,K,V = WQ(xt

i

),WK(xt

i

),WV (xt

i

)

- 5: Kijbg,Vijbg = K[1 − mask > 0],V [1 − mask > 0]
- 6: C ← Append(Kijbg,Vijbg)
- 7: xt

i

= xt

i

+ Attn(Q,K,V )

- 8: end for
- 9: Vθt

i

= MLP(xt

i

,ti)

- 10: Return Vθt

i

, C

- Algorithm 2 Simplified KV cache during denosing

Source Ours +Inf.

memory 22.1 /GB

[Figure 33]

[Figure 34]

[Figure 35]

w/o Inf.

19.4

w/ Inf.

16.2

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

1.9 1.92

1.9

step

24 28 32

- Figure 5. Demonstration of inversion-free KV-Edit. The right panel shows three comparative cases including a failure case, while the left panel illustrates inversion-free approach Significantly optimizes the space complexity to O(1).

background information. Excluding text tokens, the imagemodality self-attention computation can be expressed as:

- 1: Input: ti, foreground ztfg

i

, M-layer block {lj}Mj=1, KV cache C

- 2: Output: Prediction vector Vθtfg

i

- 3: for j = 0 to M do
- 4: Qfg,Kfg,V fg = WQ(ztfg

i

),WK(ztfg

i

),WV (ztfg

i

)

- 5: Kijbg,Vijbg = CK[i,j],CV [i,j]
- 6: K,V = Concat(Kijbg,Kfg),Concat(Vijbg,V fg)
- 7: ztfg

i

= ztfg

i

+ Attn(Qfg,K,V )

- 8: end for
- 9: Vθtfg

i

= MLP(ztfg

i

,ti)

- 10: Return Vθtfg

QfgKT √

Att(Qfg,(Kfg,Kbg),(Vfg,Vbg)) = S(

##### )V

d

(8) where Qfg represents queries containing only foreground tokens, (Kfg,Kbg) and (Vfg,Vbg) denote the concatenation of background and foreground keys and values in their proper order (equivalent to the complete image’s keys and values), and S represents the softmax operation. Notably, compared to conventional attention computations, Eq. (8) only modifies the query component, which is equivalent to performing cropping at both input and output of the attention layer, ensuring seamless integration of the generated content with the background regions.

i

#### 3.4. KV-Edit

sistency, effectively circumventing the three influencing factors discussed in Sec. 3.2.

Building upon Eq. (8), achieving background-preserving foreground editing requires providing appropriate key-value pairs for the background. Our core insight is that background tokens’ keys and values reflect their deterministic path from image to noise. Therefore, we implement KV cache during the inversion process, as detailed in Algorithm 1. This approach records the keys and values at each timestep and block layer along the probability flow path, which are subsequently used during denoising as shown in Algorithm 2. We term this complete pipeline “KV-Edit” as shown in Fig. 2 where “KV” means KV cache.

Previous works [9, 12, 16] often fail in object removal tasks when using image captions as guidance, as the original object still aligns with the target prompt. Through our in-depth analysis, we reveal that this issue stems from the residual information of the original object, which persists both in its own tokens and propagates to surrounding tokens through attention mechanisms, ultimately leading the model to reconstruct the original content.

To address the challenge in removing objects, we introduce two enhancement techniques. First, after inversion, we replace zt

Unlike other attention injection methods [4, 49, 50], KVEdit only reuses KV for background tokens while regenerating foreground tokens, without requiring specification of particular attention layers or timesteps. Rather than using the source image as injected information, we treat the deterministic background as context and the foreground as content to continue generating, analogous to KV cache in LLMs. Since the background tokens are preserved rather than regenerated, KV-Edit ensures perfect background con-

with fused noise z′t

N ·(1−tN) to disrupt the original content information. Second, we incorporate an attention mask during the inversion process, as illustrated in Fig. 2, to prevent foreground content from being incorporated into the KV values, further reducing the preservation of original content. These techniques serve as optional enhancements to improve editing capabilities and performances in different scenarios as shown in Fig. 1.

= noise·tN+zt

N

N

Source P2P RF Inv. RF Edit FlowEdit BrushEdit FLUX Fill Ours

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

RemoveobjectChangeobjectAddobject

a girl and a dog sitting in the forest → a girl sitting in the forest

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

a golden retriever holding a flower sitting on the ground … → a golden retriever sitting on the ground …

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

a woman holding a straw hat → a woman holding a straw hat and a tennis racket

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

a dog looking at the camera → a dog with a red dog collar looking at the camera

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

white plate with fruits on it → white plate with pizza on it

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

a round cake on a wooden plate → a square cake on a wooden plate

- Figure 6. Qualitative results on PIE-Bench. Unlike existing methods, our method demonstrates superior performance by strictly maintaining background consistency and simultaneously following users’ text prompt. The comparison also showcases a user-friendly workflow.

#### 3.5. Memory-Efficient Implementation

cussion about inversion-free in supplementary materials.

Inversion-based methods require storing key-value pairs for N timesteps, which can pose significant memory constraints when working with large-scale generative models (e.g., 12B parameters [1]) on personal computers. Fortunately, inspired by [23, 56], we explore an inversion-free approach. The method performs denoising immediately after each inversion step, computing the vector difference between the two results to derive a probability flow path in the t0 space. This approach allows immediate release of KV cache after use, reducing memory complexity from O(N) to O(1).

However, the inversion-free method may occasionally result in content retention artifacts as shown in Fig. 5 and FlowEdit [23]. Since our primary focus is investigating background preservation during editing, we leave more dis-

### 4. Experiments

#### 4.1. Experimental Setup

Baselines. We compare our method against two categories of approaches: (1) Training-free methods including P2P [16], MasaCtrl [9] based on DDIM [46], and RFEdit [53], RF-Inversion [44] based on Rectified Flow [33]; (2) Training-based methods including BrushEdit [26] and FLUX-Fill [1], which are based on DDIM and Rectified Flow respectively. In total, we evaluate against six prevalent image editing and inpainting approaches.

Datasets. We evaluate our method and baselines on nine tasks from PIE-Bench [21], which comprises 620 images

Image Quality Masked Region Preservation Text Align HPS ×102 ↑ AS ↑ PSNR ↑ LPIPS ×103 ↓ MSE ×104 ↓ CLIP Sim ↑ IR×10 ↑

Method

VAE∗ 24.93 6.37 37.65 7.93 3.86 19.69 -3.65 P2P [16] 25.40 6.27 17.86 208.43 219.22 22.24 0.017 MasaCtrl [9] 23.46 5.91 22.20 105.74 86.15 20.83 -1.66 RF Inv. [44] 27.99 6.74 20.20 179.73 139.85 21.71 4.34 RF Edit [53] 27.60 6.56 24.44 113.20 56.26 22.08 5.18 BrushEdit [26] 25.81 6.17 32.16 17.22 8.46 22.44 3.33 FLUX Fill [1] 25.76 6.31 32.53 25.59 8.55 22.40 5.71 Ours 27.21 6.49 35.87 9.92 4.69 22.39 5.63 +NS+RI 28.05 6.40 33.30 14.80 7.45 23.62 9.15

- Table 1. Comparison with previous methods on PIE-Bench. VAE∗ denotes the inherent reconstruction error through direct VAE reconstruction. P2P and MasaCtrl are DDIM-based methods, while RF Inversion and RF Edit are flow-based. BrushEdit and FLUX fill represent training-based methods. NS indicates there is no skip step during inversion. RI indicates the addition of reinitialization strategy. Bold and underlined values denote the best and second-best results respectively.

Method

Image Quality Text Align HPS ×102 ↑ AS ↑ CLIP Sim ∗ ↓ IR∗

×10 ↓

KV Edit (ours) 26.76 6.49 25.50 6.87 +NS 26.93 6.37 25.05 3.17 +NS+AM 26.72 6.35 25.00 2.55 +NS+RI 26.73 6.34 24.82 0.22 +NS+AM+RI 26.51 6.28 24.90 0.90

- Table 2. Ablation study for object removal task. CLIP Sim∗ and IR∗ represent alignment between source prompt and new image through CLIP [40] and Image Reward [55] to evaluate whether remove particular object from image. NS indicates there is no skip step during inversion. RI indicates the addition of reinitialization strategy. AM indicates that using attention mask during inversion.

across three dimensions to evaluate our method. For image quality, we report HPSv2 [63] and aesthetic scores [45]. For background preservation, we measure PSNR [19], LPIPS [63], and MSE. For text-image alignment, we report CLIP score [40] and Image Reward [55]. Notably, while Image Reward was previously used for quality assessment, we found it particularly effective at measuring text-image alignment, providing negative scores for unedited images. Based on this observation, we also utilize Image Reward to evaluate the successful removal of objects.

#### 4.2. Editing Results

We conduct experiments on PIE-Bench [21], categorizing editing tasks into three major types: removing, adding, and changing objects. For practical applications, these tasks prioritize background preservation and text alignment, followed by overall image quality assessment.

with corresponding masks and text prompts. Following [26, 56], we exclude style transfer tasks from PIE-Bench [21] as our primary focus is background preservation in semantic editing tasks such as object addition, removal, and change.

Quantitative Comparison. Sec. 4 presents quantitative results including baselines, our method, and our method with the reinitialization strategy. We exclude results with the attention mask strategy, as it shows improvements only in specific cases. Our method surpasses all others in Masked Region Preservation metrics. Notably, as shown in Fig. 6, methods with PSNR below 30 fail to maintain background consistency, producing results that merely resemble the original. RF-Inversion [44], despite obtaining high image quality scores, generates entirely different backgrounds. Our method achieves the third-best image quality, which has been higher than the original images, and perfectly preserving the background at the same time. With the reinitialization process, we achieve optimal text alignment scores, as the injected noise disrupts the original content, enabling more effective editing in certain cases (e.g., object removal and color change). Even compared to training-based inpainting methods [1, 26], our approach better preserves

Implementation Details. We implement our method based on FLUX.1-[dev] [1], following the same framework as other Rectified Flow-based methods [23, 44, 53]. We maintain consistent hyperparameters with FlowEdit [23], using 28 timesteps in total, skipping the last 4 timesteps (N = 24) to reduce cumulative errors, and setting guidance values to 1.5 and 5.5 for inversion and denoising processes respectively. NS in tables and charts represent no skip step (N = 28). Other baselines retain their default parameters or use previously published results. Unless otherwise specified, “Ours” in tables refers to the inversion-based KV-Edit without the two optional enhancement techniques proposed in Sec. 3.4. All experiments are conducted on two NVIDIA 3090 GPUs with 24GB memory.

Metrics Following [20, 21, 26], we use seven metrics

Source Ours +NS +RI +AM

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

a large brown sand area with clouds on the sky

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

a husky dog running on a path in the woods

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

two cups of coffee with a leaf design on the top and coffee beans

- Figure 7. Ablation study of different optional strategies on object removal task. From left to right, applying more strategies leads to stronger removal effect and the right is the best.

backgrounds while following user intentions.

Qualitative Comparison. Fig. 6 demonstrates our method’s performance against previous works across three different tasks. For removal tasks, the examples shown require both enhancement techniques proposed in Sec. 3.4. Previous training-free methods fail to preserve backgrounds, particularly Flow-Edit [23] which essentially generates new images despite high quality. Interestingly, training-based methods like BrushEdit [26] and FLUXFill [1] exhibit notable phenomena in certain cases (first and third rows in Fig. 6). BrushEdit [26], possibly limited by generative model capabilities, produces meaningless content. FLUX-Fill [1] sometimes misinterprets text prompts, generating unreasonable content like duplicate subjects. In contrast, our method demonstrates satisfactory results, successfully generating text-aligned content while preserving backgrounds, eliminating the traditional trade-off between background preservation and foreground editing.

#### 4.3. Ablation Study

We conduct ablation studies to illustrate the impact of two enhancement strategies proposed in Sec. 3.4 and the no-skip step on our method’s object removal performance. Tab. 2 presents the results in terms of image quality and text alignment scores. Notably, for text alignment evaluation, we compute the similarity between the generated results and the original prompt using CLIP [40] and Image Reward [55] models. This metric proves more discriminative in removal tasks, as still presenting of specific objects in the final images significantly increases the similarity scores.

As shown in Tab. 2, the combination of NS (No-skip) and RI (Reinitialization) achieves the optimal text alignment scores. However, we observe a slight decrease in im-

ours vs. Quality↑ Background↑ Text↑ Overall↑

Random∗ 50.0% 50.0% 50.0% 50.0% RF Inv. [44] 61.8% 94.8% 79.6% 85.1% RF Edit [53] 54.5% 90.5% 75.0% 73.6% BrushEdit [26] 71.8% 66.7% 68.7% 70.2% FLUX Fill [1] 60.0% 53.7% 58.6% 61.9%

Table 3. User Study. We compared our method with four popular baselines. Participants were asked to choose their preferred option or indicate if both methods were equally good or not good based on four criteria. We report the win rates of our method compared to baseline excluding equally good or not good instances. Random∗ denotes the win rate of random choices.

age quality metrics after incorporating these components. We attribute this phenomenon to the presence of too large masks in the benchmark, where no-skip, reinitialization, and attention mask collectively disrupt substantial information, leading to some discontinuities in the generated images. Consequently, these strategies should be viewed as optional enhancements for editing effects rather than universal solutions applicable to all scenarios.

Fig. 7 visualizes the impact of these strategies. In the majority of cases, reinitialization alone suffices to achieve the desired results, while a small subset of cases requires additional attention masking for enhanced performance.

#### 4.4. User Study

We conduct an extensive user study to compare our method with four baselines, including the training-free methods RFEdit [53], RF-Inversion [44], and the training-based methods BrushEdit [26] and Flux-Fill [1]. We use 110 images from the “random class” in the PIE-Bench [21] (excluding style transfer task, images without backgrounds, and controversial content). More than 20 participants are asked to compare each pair of methods based on four criteria: image quality, background preservation, text alignment, and overall satisfaction. As shown in Tab. 3, our method significantly outperforms the previous methods, even surpassing Flux-Fill [1], which is the official inpainting model of FLUX [1]. Additionally, users’ feedback reveals that background preservation plays a crucial role in their final choices, even if RF-Edit [53] achieves high image quality but finally fails in satisfaction comparison.

### 5. Conclusion

In this paper, we introduce KV-Edit, a new training-free approach that achieves perfect background preservation in image editing by caching and reusing background keyvalue pairs. Our method effectively decouples foreground editing from background preservation through attention mechanisms in DiT, while optional enhancement strategies and memory-efficient implementation further improve

its practical utility. Extensive experiments demonstrate that our approach surpasses both training-free methods and training-based inpainting models in terms of both background preservation and image quality. Moreover, we hope that this straightforward yet effective mechanism could inspire broader applications, such as video editing, multi-concept personalization, and other scenarios.

### References

- [1] Flux. https://github.com/black- forestlabs/flux/. 2, 3, 6, 7, 8, 12, 13, 14
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 3

- [3] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. TOG, 42(4):1–11, 2023. 2
- [4] Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel CohenOr. Stable flow: Vital layers for training-free image editing. arXiv preprint arXiv:2411.14430, 2024. 2, 5, 12
- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3
- [6] Sule Bai, Yong Liu, Yifei Han, Haoji Zhang, and Yansong Tang. Self-calibrated clip for training-free open-vocabulary segmentation. arXiv preprint arXiv:2411.15869, 2024. 3
- [7] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, pages 18392–18402, 2023. 2, 3
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877– 1901, 2020. 3
- [9] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, pages 22560–22570, 2023. 2, 5, 6, 7
- [10] Zhennan Chen, Yajie Li, Haofan Wang, Zhibo Chen, Zhengkai Jiang, Jun Li, Qian Wang, Jian Yang, and Ying Tai. Region-aware text-to-image generation via hard binding and soft refinement. arXiv preprint arXiv:2411.06558, 2024. 2
- [11] Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, and Yansong Tang. Motionlcm: Real-time controllable motion generation via latent consistency model. In ECCV, pages 390–408, 2024. 2
- [12] Wenkai Dong, Song Xue, Xiaoyue Duan, and Shumin Han. Prompt tuning inversion for text-driven image editing using diffusion models. In ICCV, pages 7430–7440, 2023. 2, 3, 5, 14

- [13] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2
- [15] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 16000–16009, 2022. 3
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 2, 3, 5, 6, 7, 14
- [17] Wenke Huang, Jian Liang, Zekun Shi, Didi Zhu, Guancheng Wan, He Li, Bo Du, Dacheng Tao, and Mang Ye. Learn from downstream and be yourself in multimodal large language model fine-tuning. arXiv preprint arXiv:2411.10928, 2024. 3
- [18] Xiaoke Huang, Jianfeng Wang, Yansong Tang, Zheng Zhang, Han Hu, Jiwen Lu, Lijuan Wang, and Zicheng Liu. Segment and caption anything. In CVPR, pages 13405– 13417, 2024. 3
- [19] Quan Huynh-Thu and Mohammed Ghanbari. Scope of validity of psnr in image/video quality assessment. Electronics letters, 44(13):800–801, 2008. 7
- [20] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In ECCV, pages 150–168, 2024. 2, 3, 7
- [21] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. In ICLR, 2024. 2, 6, 7, 8, 12, 14
- [22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, pages 6007–6017, 2023. 2, 3
- [23] Vladimir Kulikov, Matan Kleiner, Inbar HubermanSpiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024. 2, 6, 7, 8, 13
- [24] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, pages 19730–19742, 2023. 3
- [25] Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang. Stylediffusion: Prompt-embedding inversion for text-based editing. arXiv preprint arXiv:2303.15649, 2023. 2
- [26] Yaowei Li, Yuxuan Bian, Xuan Ju, Zhaoyang Zhang, Ying Shan, and Qiang Xu. Brushedit: All-in-one image inpainting and editing. arXiv preprint arXiv:2412.10316, 2024. 2, 3, 6, 7, 8
- [27] Haonan Lin, Mengmeng Wang, Jiahao Wang, Wenbin An, Yan Chen, Yong Liu, Feng Tian, Guang Dai, Jingdong Wang,

- and Qianying Wang. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. arXiv preprint arXiv:2410.18756, 2024. 2
- [28] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 3
- [29] Aoyang Liu, Qingnan Fan, Shuai Qin, Hong Gu, and Yansong Tang. Lipe: Learning personalized identity prior for non-rigid image editing. arXiv preprint arXiv:2406.17236,

2024. 2

- [30] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024. 3
- [31] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 36, 2024. 3
- [32] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022. 3, 4
- [33] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2022. 3, 4, 6
- [34] Yong Liu, Sule Bai, Guanbin Li, Yitong Wang, and Yansong Tang. Open-vocabulary segmentation with semantic-assisted calibration. In CVPR, pages 3491–3500, 2024. 3
- [35] Yong Liu, Cairong Zhang, Yitong Wang, Jiahao Wang, Yujiu Yang, and Yansong Tang. Universal segmentation at arbitrary granularity with language instruction. In CVPR, pages 3459–3469, 2024. 3
- [36] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 2
- [37] Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models. arXiv preprint arXiv:2305.16807, 2023. 2
- [38] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, pages 6038–6047,

2023. 2

- [39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 2, 4, 12, 15
- [40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pages 8748–8763, 2021. 7, 8
- [41] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image syn-

- thesis with latent diffusion models. In CVPR, pages 10684– 10695, 2022. 2
- [43] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241, 2015. 2, 3, 4
- [44] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 6, 7, 8
- [45] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. NeurIPS, 35:25278– 25294, 2022. 7
- [46] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3, 4, 6
- [47] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [48] Siao Tang, Xin Wang, Hong Chen, Chaoyu Guan, Zewen Wu, Yansong Tang, and Wenwu Zhu. Post-training quantization with progressive calibration and activation relaxing for text-to-image diffusion models. In ECCV, pages 404–420,

2024. 2

- [49] Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. Add-it: Training-free object insertion in images with pretrained diffusion models. arXiv preprint arXiv:2411.07232, 2024. 2, 5
- [50] Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In CVPR, pages 1921–1930,

2023. 2, 5, 14

- [51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. page 6000–6010,

2017. 4

- [52] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate post-training quantization for diffusion models. In CVPR, pages 16026– 16035, 2024. 2
- [53] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024. 2, 6, 7, 8
- [54] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 3
- [55] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. NeurIPS, 36:15903–15935, 2023. 7, 8
- [56] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with language-guided diffusion models. In CVPR, pages 9452–9461, 2024. 2, 6, 7

- [57] Yu Xu, Fan Tang, Juan Cao, Yuxin Zhang, Xiaoyu Kong, Jintao Li, Oliver Deussen, and Tong-Yee Lee. Headrouter: A training-free image editing framework for mmdits by adaptively routing attention heads. arXiv preprint arXiv:2411.15034, 2024. 2
- [58] Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Lavt: Language-aware vision transformer for referring image segmentation. In CVPR, pages 18155–18165, 2022. 3
- [59] Zhao Yang, Jiaqi Wang, Xubing Ye, Yansong Tang, Kai Chen, Hengshuang Zhao, and Philip HS Torr. Languageaware vision transformer for referring segmentation. TPAMI,

2024. 3

- [60] Xubing Ye, Yukang Gan, Yixiao Ge, Xiao-Ping Zhang, and Yansong Tang. Atp-llava: Adaptive token pruning for large vision language models. arXiv preprint arXiv:2412.00447,

2024. 3

- [61] Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, Ying Shan, and Yansong Tang. Voco-llama: Towards vision compression with large language models. arXiv preprint arXiv:2406.12275, 2024.
- [62] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Jifeng Dai, and Xiaojie Jin. Flash-vstream: Memorybased real-time understanding for long video streams. arXiv preprint arXiv:2406.08085, 2024. 3
- [63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, pages 586–595,

2018. 7

- [64] Yixuan Zhu, Wenliang Zhao, Ao Li, Yansong Tang, Jie Zhou, and Jiwen Lu. Flowie: Efficient image enhancement via rectified flow. In CVPR, pages 13–22, 2024. 2
- [65] Junhao Zhuang, Yanhong Zeng, Wenran Liu, Chun Yuan, and Kai Chen. A task is worth one word: Learning with task prompts for high-quality versatile image inpainting. In ECCV, pages 195–211, 2024. 2, 3

### Appendix

In this supplementary material, we provide more details and findings. In Appendix A, we present additional experimental results and implementation details of our proposed KVEdit. Appendix B provides further discussion and data regarding our inversion-free methodology. Appendix C details the design and execution of our user study. Moreover, In Appendix D, we discuss potential future directions and current limitations of our work.

### A. Implementation and More Experiments

Implementation Details. Our code is built on Flux [1], with modifications to both double block and single block to incorporate KV cache through additional function parameters. Input masks are first downsampled using bilinear interpolation, then transformed from single-channel to 64channel representations following the VAE in Flux [1]. In the feature space, the smallest pixel unit is 16 dimensions rather than the entire 64-dimensional token. Therefore, in addition to KV cache, we preserve the intermediate image features at each timestep to ensure fine-grained editing capabilities. In our experiment, inversion and denoising can be performed independently, allowing a single image to be inverted just once and then edited multiple times with different conditions, further enhancing the practicality of this workflow.

Experimental Results. Due to space constraints in the main paper, we only present results on the PIE-Bench [21].

Here, we provide additional examples demonstrating the effectiveness of our approach. To further showcase the flexibility of our method, Fig. A and Fig. B present various editing target applied to the same source image, without explicitly labeling the input masks because each case corresponds to a different mask. Fig. D illustrates the impact of steps and reinitialization strategy on the color changing tasks and inpainting tasks.

When changing colors, as the number of skip-steps decreases and reinitialization strategy is applied, the color information in the tokens is progressively disrupted, ultimately achieving successful results. In our experiments, the optimal number of steps to skip depends on image resolution and content, which can be adjusted based on specific needs and feedback. Unlike previous training-free methods, our approach even can be applied to inpainting tasks after employing reinitialization strategy, as demonstrated in the third row of Fig. D. The originally removed regions in inpainting tasks can be considered as black objects, thus requiring reinitialization strategy to eliminate pure black information and generate meaningful content. We plan to further extend our method to inpainting tasks in future work, as there are currently very few training-free methods available for this application.

Attention Scale When dealing with large masks (e.g., background changing tasks), our original method may produce discontinuous images including conflicting content, as illustrated in Fig. C. Stable-Flow [4] demonstrated that during image generation with DiT [39], image tokens primarily at-

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

red cup black mug empty bottle white mug red mug bowl

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

cola lemonade coffee necktie tartan bow polka-dot

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

sunglasses cat-eye glasses wire-rimmed straw hat baseball cap beret

- Figure A. Additional editing results of KV-Edit. Our method demonstrates robust performance with strict background preservation and high image quality. Users can achieve creative designs by simply adjusting text prompts and masks according to their needs.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

cowboy hat straw hat graduation cap flowers teddy bear Christmas wreath

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

cat dog toy car vase alarm clock flower arrangement

- Figure B. Additional editing results of KV-Edit. Our method demonstrates robust performance with strict background preservation and high image quality. Users can achieve creative designs by simply adjusting text prompts and masks according to their needs.

tend to their local neighborhood rather than globally across most layers and timesteps.

Consequently, although our approach treats the background as a condition to guide new content generation, large masks can introduce generation bias which ignore existing content and generate another objects. Based on this analysis, we propose a potential solution as shown in Fig. C. We directly increase the attention weights from masked regions to unmasked regions in the attention map (produced by query-key multiplication), effectively mitigating the bias impact. This attention scale mechanism enhances content coherence by strengthening the influence of preserved background on new content.

B. More Discussions on Inversion-Free

We implement inversion-free editing on Flux [1] based on the code provided by FlowEdit [23]. As noted in FlowEdit [23], adding random noise at each editing step may introduce artifacts, a phenomenon we also demonstrate in the main paper. In this section, we primarily explore the

text mask unmask

mask

0 0 0 0

0 0 0 0

1 1 1 1

scale× +

[Figure 140]

[Figure 141]

Ours

+scale

- Figure C. Implementation of attention scale. The scale can be adjusted to achieve optimal results.

Source Ours +SKIP=1 +NS +RI

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

a man wearing black → white shirt

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

a man wearing white → blue T-shirt

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

a round cake with orange frosting on a wooden plate

Figure D. Additional ablation studies on two tasks. The first and second rows demonstrate the impact of timesteps and reinitialization strategy (RI) on color changing. The third row demonstrates the impact of timesteps and RI on the inpainting tasks.

impact of inversion-free methods on memory consumption.

Algorithm A demonstrates the implementation of inversion-free KV-Edit, where “KV-inversion” and “KVdenoising” refer to single-step noise prediction with KV cache. KV cache is saved during a one-time inversion process and immediately utilized in the denoising process. The final vector can be directly added to the original image without first inversing it to noise. This strategy ensures that the space complexity of KV cache remains O(1) along the time dimension. Moreover, resolution has a more significant impact on memory consumption as the number of image tokens grows at a rate of O(n2).

We conducted experiments across various resolutions and time steps, reporting memory usage in Tab. A. When processing high-resolution images and more timesteps, personal computers struggle to accommodate the mem-

512 × 512 768 × 768 Ours +Inf. Ours +Inf.

timesteps

24 steps 16.2G 1.9G 65.8G 3.5G 28 steps 19.4G 1.9G 75.6G 3.5G 32 steps 22.1G 1.9G 86.5G 3.5G

Table A. Memory usage at different resolutions and timesteps. Our approach has a space complexity of O(n) along the time dimension, while inversion-free methods achieve O(1).

Algorithm A Simplified Inf. version KV-Edit

- 1: Input: ti, real image xsrc0 , foreground ztfg

i

,foreground region mask, KV cache C

- 2: Output: Prediction vector Vθtfg

i

- 3: Nt

i ∼ N(0,1)

- 4: xsrct

i

= (1 − ti)xsrct

0

+ tiNt

i

- 5: Vθtsrc

i

,C = KV-Inverison(xsrct

i

,ti,C)

- 6: ztfg

i

= ztfg

i

+ mask · (xsrct

i

− xsrc0 )

- 7: Vθtfg

i

,C = KV-Denosing( ztfg

i

,ti,C)

- 8: Return Vθtfg

= Vθtfg

− Vθtsrc

i

i

i

ory requirements. Nevertheless, we still recommend the inversion-based KV-Edit approach for several reasons:

- 1. Current inversion-free methods occasionally introduce artifacts.
- 2. Inversion-based KV-Edit enables multiple editing attempts after a single inversion, significantly improving usability and workflow efficiency.
- 3. Large generative models inherently require substantial GPU memory, which presents another challenge for personal computers. Therefore, we position inversion-based KV-Edit as a server-side technology.

### C. User Study Details

We conduct our user study in a questionnaire format to collect user preferences for different methods. We observe that in most cases, users struggle to distinguish the background effects of training-based inpainting methods (e.g., FLUX-Fill [1] sometimes increases grayscale tones in images). Therefore, we allowed participants to select “equally good” regarding background quality.

Additionally, PIE-Bench [21] contains several challenging cases where all methods fail to complete the editing tasks satisfactorily. Consequently, we allow users to select “neither is good” for text alignment and overall satisfaction metrics, as illustrated in Fig. E.

We implement a single-blind mechanism where the corresponding method for each question is randomly sampled,

[Figure 157]

Figure E. User study. We provide a sample where participants were presented with the original image, editing prompts, results from two different methods for comparison and four questions from four aspects.

ensuring fairness in the comparison. We collect over 2,000 comparison results and calculate our method’s win rate after excluding cases where both methods are rated equally.

### D. Limitations and Future Work

In this section, we outline the current challenges faced by our method and potential future improvements. While our approach effectively preserves background content, it struggles to maintain foreground details. As shown in Fig. D, when editing garment colors, clothing appearance features may be lost, such as the style, print or pleats.

Typically, during the generation process, early steps determine the object’s outline and color, with specific details and appearance emerging later. In the contrast, during inversion, customized object details are disrupted first and subsequently influenced by new content during denoising. This represents a common challenge in the inversion-denoising paradigm [12, 16, 50].

In future work, we could employ trainable tokens to preserve desired appearance information during inversion and inject it during denoising, still without fine-tuning of the base generative model. Furthermore, our method could be

adapted to other modalities, such as video and audio editing, image inpainting tasks. We hope that “KV cache for editing” can be considered an inherent feature of the DiT [39] architecture.

