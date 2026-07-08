### FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion

Haonan Qiu1, Shiwei Zhang2, , Yujie Wei3, Ruihang Chu2, Hangjie Yuan2, Xiang Wang2, Yingya Zhang2, Ziwei Liu1,

1Nanyang Technological University 2Alibaba Group 3Fudan University

# arXiv:2412.09626v2[cs.CV]11Jul2025

###### Project Page: http://haonanqiu.com/projects/FreeScale.html

[Figure 1]

[Figure 2]

[Figure 3]

|[Figure 4]|
|---|

(2048×4096)

[Figure 5]

|[Figure 6]|
|---|

|[Figure 7]|
|---|

16× Resolution (4096×4096) (2048×4096)

[Figure 8]

[Figure 9]

[Figure 10]

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

16× Resolution (4096×4096)

64× Resolution (8192×8192)

(4096×2048) (4096×2048)

Figure 1. Gallery of FreeScale. Original SDXL [40] can only generate images with a resolution of up to 10242 without losing quality, while FreeScale successfully extends SDXL to generate 81922 images without any fine-tuning. All generated images are produced using a single A800 GPU. Best viewed ZOOMED-IN.

##### Abstract

tackle this challenge, we propose FreeScale, a tuning-free inference paradigm to enable higher-resolution visual generation via scale fusion. Specifically, FreeScale processes information from different receptive scales and then fuses it by extracting desired frequency components. Extensive experiments validate the superiority of our paradigm in extending the capabilities of higher-resolution visual generation for both image and video models. Notably, compared with previous best-performing methods, FreeScale unlocks the 8k-resolution text-to-image generation for the first time.

Visual diffusion models achieve remarkable progress, yet they are typically trained at limited resolutions due to the lack of high-resolution data and constrained computation resources, hampering their ability to generate high-fidelity images or videos at higher resolutions. Recent efforts have explored tuning-free strategies to exhibit the untapped potential higher-resolution visual generation of pre-trained models. However, these methods are still prone to producing low-quality visual content with repetitive patterns. The key obstacle lies in the inevitable increase in high-frequency information when the model generates visual content exceeding its training resolution, leading to undesirable repetitive patterns deriving from the accumulated errors. To

##### 1. Introduction

Diffusion models have revolutionized visual generation [10, 11, 40, 51, 55, 57], empowering individuals without any

artistic expertise to effortlessly create distinctive and personalized designs, graphics, and short films using specific textual descriptions. Nonetheless, current visual diffusion models are generally trained on data with limited resolution, such as 5122 for SD 1.5 [43], 10242 for SDXL [40], and 320 × 512 for VideoCrafter2 [10], hampering their ability to generate high-fidelity images or videos at higher resolutions. Given the scarcity of high-resolution visual data and the substantially greater model capacity required for modeling such data, recent efforts have focused on employing tuning-free strategies for high-resolution visual generation to inherit the strong generation capacities of existing pretrained diffusion models.

Despite the advances achieved by existing methods, they are still prone to producing low-quality images or videos, particularly manifesting as repetitive object occurrences and unreasonable object structures. ScaleCrafter [20] puts forward that the primary cause of the object repetition issue is the limited convolutional receptive field and uses dilated convolutional layers to achieve tuning-free higherresolution sampling. But the generated results of ScaleCrafter still suffer from the problem of local repetition. Inspired by MultiDiffusion [2] fusing the local patches of the whole images, DemoFusion [14] designed a mechanism by fusing the local patches and global patches, almost eliminating the local repetition. Essentially, this solution just transfers the extra signal of the object to the background, leading to small object repetition generation. FouriScale [25] reduces those extra signals by removing the high-frequency signals of the latent before the convolution operation. Although FouriScale completely eliminates all types of repetition, the generated results always have weird colors and textures due to its violent editing on the frequency domain.

To generate satisfactory visual contents without any unexpected repetition, we propose FreeScale, a tuning-free inference paradigm that enables pre-trained image and video diffusion models to generate vivid higher-resolution results. Building on past effective modules [16, 20], we first propose tailored self-cascade upscaling and restrained dilated convolution to gain the basic visual structure and maintain the quality in higher-resolution generation. To further eliminate all kinds of unexpected object repetitions, FreeScale processes information from different receptive scales and then fuses it by extracting desired frequency components, ensuring both the structure’s overall rationality and the object’s local quality. This fusion is smoothly integrated into the original self-attention layers, thereby bringing only minimal additional time overhead. Finally, we demonstrate the effectiveness of our model on both the text-to-image model and the text-to-video model, pushing the boundaries of image generation even up to an 8k resolution.

Our contributions are summarized as follows:

• We propose FreeScale, a tuning-free inference paradigm

to enable pre-trained diffusion models to generate vivid higher-resolution results via fusing the information from different scales.

- • We empirically evaluate our approach on both the text-toimage model and the text-to-video model, demonstrating the effectiveness of our model.
- • Compared to other state-of-the-art tuning-free methods, we unlock the 8k-resolution text-to-image generation for the first time.

##### 2. Related Work

Diffusion Models for Visual Generation. The advent of diffusion models has transformed the landscape of image and video generation by enabling the production of exceptionally high-quality outputs [10, 11, 40, 47, 51, 54–57]. Initial breakthroughs like DDPM [22] and Guided Diffusion [13] demonstrated that diffusion processes could yield remarkable image quality. To enhance computational efficiency, LDM [43] introduced latent space diffusion, which operates in a compressed space, significantly lowering the computational burden and training demands; this method laid the groundwork for Stable Diffusion. Building on this, SDXL [40] further advanced high-resolution image synthesis. Inspired by DiT [39], Pixart-alpha [11] adopted a transformer-based architecture, achieving both high fidelity and cost-effective image generation.

For video generation, VDM [23] pioneered the application of diffusion in this domain, followed by LVDM [19], which extended the method to propose a hierarchical latent video diffusion framework capable of generating extended video sequences. To bridge text-to-image and text-to-video (T2V) capabilities, Align-Your-Latents [6] and AnimateDiff [17] introduced temporal transformers into existing T2I models. VideoComposer [53] then offered a controllable T2V generation approach, allowing precise management of spatial and temporal cues. VideoCrafter [9, 10] and SVD [5] scaled these latent video diffusion models to handle extensive datasets. Lumiere [3] proposed temporal downsampling within a space-time U-Net for greater efficiency. Finally, CogVideoX [55] and Pyramid Flow [28] two recent highly regarded open-source models, showcase impressive video generation capabilities, demonstrating the superior performance of DiT structure in video generation.

Since the DiT structure models often take up more memory, achieving high-resolution generation on a single GPU is difficult even in the inference phase. Therefore, we still use the U-Net structure models in this work. We chose SDXL [40] as our pre-trained image model, and VideoCrafter2 [10] as our pre-trained video model.

Higher-Resolution Visual Generation. High-resolution visual synthesis is a classic challenge in the generative field due to the difficulty of collecting plenty of highresolution data and the requirement of substantial compu-

Tailored Self-Cascade Upscaling Scale Fusion

[Figure 15]

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | |[Figure 16]| | | |
| |[Figure 17]| |[Figure 18]| |[Figure 19]| | | |

## 𝒟

Upsample

[Figure 20]

[Figure 21]

| |[Figure 22]| |[Figure 23]| |[Figure 24]| | | |
|---|---|---|---|---|---|---|---|---|
| | | | | |[Figure 25]| | | |
| | | | | | | | | |

Self-Attention Layer

## ℰ

[Figure 26]

|[Figure 27]| |
|---|---|
| | |

(Skipped)

|Gaussian Blur|
|---|

[Figure 28]

Detail

−

[Figure 29]

Control Copy

High Frequency

[Figure 30]

Mask (or Scalar)

Global Attention

[Figure 31]

+

[Figure 32]

[Figure 33]

| | | | |
|---|---|---|---|

[Figure 34]

Low Frequency

| |[Figure 35]| |[Figure 36]| |[Figure 37]| | | |
|---|---|---|---|---|---|---|---|---|
| | | | | |[Figure 38]| | | |
| | | | | | | | | |

|Gaussian Blur|
|---|

𝒟

[Figure 39]

Local Attention

(Skipped)

𝒟 VAEDecoder ℰ VAEEncoder

− / + Element Sub/Add

UNet Block with Dilated Convolution

Weighted Add

- Figure 2. Overall framework of FreeScale. (a) Tailored Self-Cascade Upscaling. FreeScale starts with pure Gaussian noise and progressively denoises it using the training resolution. An image is then generated via the VAE decoder, followed by upscaling to obtain a higher-resolution one. We gradually add noise to the latent of this higher-resolution image and incorporate this forward noise into the denoising process of the higher-resolution latent with the use of restrained dilated convolution. Additionally, for intermediate latent steps, we enhance high-frequency details by applying region-aware detail control using masks derived from the image. (b) Scale Fusion. During denoising, we adapt the self-attention layer to a global and local attention structure. By utilizing Gaussian blur, we fuse high-frequency details from global attention and low-frequency semantics from local attention, serving as the final output of the self-attention layer.

lutional layers to enlarge the convolutional receptive field. Although successful in removing small object repetition, ScaleCrafter suffers from a new problem of local repetition. FouriScale [25] concludes that all types of repetitions are from the non-alignment of frequency domain on different scales. FouriScale removes the high-frequency signals of the latent prior to convolution operation and achieves no repetition at all. But this violent editing operation on the frequency domain leads to strange results with unnatural colors and textures. Another solution is directly removing the text semantics from unexpected areas in the input level [36, 38]. However, it only works for small object repetition and will suffer information leakage through the temporal layers in the video generation. With the additional pose as input, BeyondScene [30] has achieved 8k human image generation. However, its scope is limited to human image generation due to the requirement of additional pose input. FreeScale is the first 8k-resolution text-to-image generation method without these constraints.

tational resources. Recent methods for higher-resolution generation can mainly be divided into two categories: 1) training/tuning methods with high-resolution data and large models [12, 16, 24, 37, 42, 49, 59], or 2) tuning-free methods without any additional data requirement [7, 18, 27, 29, 31, 34, 35, 58]. Training with high-resolution data on larger models should be a more fundamental solution. However, high-resolution visual data only accounts for a small proportion. Meanwhile, targeting for modeling higherresolution data demands a notably increased requirement in model capacity. Based on current data and calculation resources, tuning-free approaches are more achievable for high-resolution generation.

One straightforward approach is to generate visual patches of the same resolution as the training data and then stitch them together. Although eliminating the traininginference gap, this method results in disconnected and incoherent patches. MultiDiffusion [2] addresses this problem by fusing patches smoothly during the denoising process. DemoFusion [14] utilizes this mechanism and adds global perception to ensure the rationality of the overall layout. However, this solution easily leads to the generation of small object repetition. ScaleCrafter [20] argues that the object repetition issue is mainly caused by the limited convolutional receptive field and uses dilated convo-

##### 3. Methodology

###### 3.1. Preliminaries

Latent Diffusion Models (LDM) first encodes a given image x to the latent space z via the encoder of the pre-

trained auto-encoder E: z = E(x). Then a forward diffusion process is used to gradually add noise to the latent data z0 ∼ p(z0) and learn a denoising model to reverse this process. The forward process contains T timesteps, which gradually add noise to the latent sample z0 to yield zt through a parameterization trick:

q(zt|zt−1) = N(zt; 1 − βtzt−1,βtI), q(zt|z0) = N(zt;√α¯tz0,(1 − α¯t)I),

(1)

where βt is a predefined variance schedule, t is the timestep, α¯t = ti=1 αi, and αt = 1 − βt. The reverse denoising process obtains less noisy latent zt−1 from the noisy input zt at each timestep:

pθ (xt−1 | xt) = N (xt−1;µθ (zt,t),Σθ (zt,t)), (2)

where µθ and Σθ are determined through a noise prediction network ϵθ (zt,t) with learnable parameters θ.

###### 3.2. Tailored Self-Cascade Upscaling

Directly generating higher-resolution results will easily produce several repetitive objects, losing the reasonable visual structure that was originally good. To address this issue, we utilize a self-cascade upscaling framework from previous works [14, 16], which progressively increases the resolution of generated results:

√α¯Kϕ(zr0),√1 − α¯KI , (3)

z˜2Kr ∼ N

where z˜ means the noised intermediate latent, r is the resolution level (1 represents original resolution, 2 represents the twice height and width), and ϕ is an upsampling operation. Specifically, FreeScale will denoise using the training resolution. The intermediate results will then be gradually up-sampled. In the higher resolution, blurry details from the upsampling will be removed by adding noise (to the level of timestep K) and denoising. In this way, the framework will generate a reasonable visual structure in low resolution and maintain the structure when generating higherresolution results.

There are two options for ϕ: directly upsampling in latent (ϕ(z) = UP(z)) or upsampling in RGB space (ϕ(z) = E(UP(D(z))), where E and D are the encoder and decoder of pre-trained VAE, respectively. Upsampling in RGB space is closer to human expectations but will add some blurs. We empirically observe that these blurs will hurt the video generation but help to suppress redundant overfrequency information in the image generation. Therefore, we adopt upsampling in RGB space for higher-solution image generation and latent space upsampling in highersolution video generation.

Flexible Control for Detail Level. Different from superresolution tasks, FreeScale will endlessly add more details as the resolution grows. This behavior will hurt the

generation when all reasonable details are generated. To control the level of newly generated details, we modify pθ (zt−1 | zt) to pθ (zt−1 | zˆt) with:

zˆrt = c × z˜rt + (1 − c) × zrt, (4)

where c = 1 + cos TT−t × π /2 α is a scaled cosine decay factor with a scaling factor α.

Even in the same image, the detail level varies in different areas. To achieve more flexible control, α can be a 2D-tensor and varies spatially. In this case, users can assign different values for different semantic areas according to D (zr0) calculated in the previous process already.

###### 3.3. Restrained Dilated Convolution

ScaleCrafter [20] observes that the primary cause of the object repetition issue is the limited convolutional receptive field and proposes dilated convolution to solve it. Given a hidden feature map h, a convolutional kernel k, and the dilation operation Φd(·) with factor d, the dilated convolution can be represented as:

fkd(h) = h⊛Φd(k),(h ⊛ Φd(k))(o) =

h(p)·k(q),

s+d·t=p

(5) where o, p, and q are spatial locations used to index the feature or kernel. ⊛ denotes convolution operation.

To avoid catastrophic quality decline, ScaleCrafter [20] only applies dilated convolution to some layers of UNet while still consisting of several up-blocks. However, we find that dilated convolution in the layers of up-blocks will bring many messy textures. Therefore, unlike previous works, we only apply dilated convolution in the layers of down-blocks and mid-blocks. In addition, the last few timesteps only render the details of results and the visual structure is almost fixed. Therefore, we use the original convolution in the last few timesteps.

###### 3.4. Scale Fusion

Although tailored self-cascade upscaling and restrained dilated convolution can maintain the rough visual structures and effectively generate 4× resolution images, generating 16× resolution images still leads to artifacts such as local repetition, e.g., additional eyes or noses. This issue arises because dilated convolution weakens the focus on local features. DemoFusion [14] addresses this by using local patches to enhance local focus. However, although the local patch operation mitigates local repetition, it brings small object repetition globally. To combine the advantages of both strategies, we design Scale Fusion, which fuses information from different receptive scales to achieve a balanced enhancement of local and global details.

Regarding global information extraction, we utilize global self-attention features. The reason is that the

Table 1. Image quantitative comparisons with other baselines. FreeScale achieves the best or second-best scores for all quality-related metrics with negligible additional time costs. The best results are marked in bold, and the second best results are marked by underline.

20482 40962

Method

FID ↓ KID ↓ FIDc ↓ KIDc ↓ IS ↑ Time (min) ↓ FID ↓ KID ↓ FIDc ↓ KIDc ↓ IS ↑ Time (min) ↓ SDXL-DI [40] 64.313 0.008 31.042 0.004 10.424 0.648 134.075 0.044 42.383 0.009 7.036 5.456 ScaleCrafter [20] 67.545 0.013 60.151 0.020 11.399 0.653 100.419 0.033 116.179 0.053 8.805 9.255 DemoFusion [14] 65.864 0.016 63.001 0.024 13.282 1.441 72.378 0.020 94.975 0.045 12.450 11.382 FouriScale [25] 68.965 0.016 69.655 0.026 11.055 1.224 93.079 0.029 128.862 0.068 8.248 8.446 Ours 44.723 0.001 36.276 0.006 12.747 0.853 49.796 0.004 71.369 0.029 12.572 6.240

self-attention layer enhances the patch information based on similarity, making it easier for the subsequent crossattention layer to aggregate semantics into a complete object. This can be formulated as:

QKT √

hglobalout = SelfAttention(hin) = softmax

V, where Q = LQ(hin),K = LK(hin),V = LV (hin).

d′

(6)

In this formulation, query Q, key K, and value V are calculated from hin through the linear layer L, and d′ is a scaling coefficient for the self-attention.

After that, the self-attention layer is independently applied to these local latent representations via hout, n = SelfAttention(hin, n). And then Houtlocal = [hout, 0 ··· ,hout, n ··· ,hout, N] is reconstructed to the original size with the overlapped parts averaged as hlocalout = Rlocal Houtlocal , where Rlocal denotes the reconstruction process.

Regarding local information extraction, we follow previous works [2, 14, 41] by calculating self-attention locally to enhance the local focus. Specifically, we first apply a shifted crop sampling, Slocal(·), to obtain a series of local latent representations before each self-attention layer, i.e., Hinlocal = Slocal (hin) = [hin, 0 ··· ,hin, n ··· ,hin, N],hin, n ∈ Rc×h×w, where N = (Hd−h)

+ 1 × (Wd−w)

+ 1 , with dh and dw representing the vertical and horizontal stride, respectively. After that, the self-attention layer is independently applied to these local latent representations via hout, n = SelfAttention(hin, n). The resulting outputs Houtlocal = [hout, 0 ··· ,hout, n ··· ,hout, N] are then mapped back to the original positions, with the overlapped parts averaged to form hlocalout = Rlocal Houtlocal , where Rlocal denotes the reconstruction process.

w

h

While hlocalout tends to produce better local results, it can bring unexpected small object repetition globally. These artifacts mainly arise from dispersed high-frequency signals, which will originally be gathered to the right area through global sampling. Therefore, we replace the highfrequency signals in the local representations with those from the global level hglobalout :

hfusionout = hglobalout − G hglobalout

high frequency

+ G hlocalout

, (7)

low frequency

where G is a low-pass filter implemented as a Gaussian blur, and hglobalout − G hglobalout acts as a high pass of hfusionout .

##### 4. Experiments

Experimental Settings. We conduct experiments based on an open-source T2I diffusion model SDXL [40] and an open-source T2V diffusion model VideoCrafter2 [10]. Considering the computing resources that can be afforded, we evaluate the image generation at resolutions of 20482 and 40962, and video generation at resolutions of 640 × 1024. All experiments are produced using a single A800 GPU.

Evaluation Metrics. Since higher-resolution inference methods are intended to maintain the quality of the original resolution outputs, we calculate all metrics between the originally generated low-resolution images/videos and the corresponding high-resolution outputs. To evaluate the quality of generated images, we report Frechet Image Distance (FID) [21], Kernel Image Distance (KID) [4] and IS (Inception Score) [44]. FID and KID need to resize the images to 299 before the comparison and this operation may cause quality loss for high-resolution images. Inspired by previous work [8], we also use cropped local patches to calculate these metrics without resizing, termed FIDc and KIDc. We use Frechet Video Distance (FVD) [50] to evaluate the quality of video generation. In addition, we test dynamic degree and aesthetic quality from the VBench [26] to evaluate the dynamics and aesthetics.

###### 4.1. Higher-Resolution Image Generation

We compare FreeScale with other higher-solution image generation methods: (i) SDXL [40] direct inference (SDXL-DI) (ii) ScaleCrafter [20] (iii) DemoFusion [14], and (iv) FouriScale [25]. FreeU [47] is used if compatible.

Qualitative comparison results are shown in Figure 3. We observe that direct generation often results in multiple duplicated objects and a loss of the original visual structure. ScaleCrafter tends to produce localized repetitions, while DemoFusion generates isolated small objects nearby. FouriScale can drastically alter the style for certain prompts. In contrast, the proposed FreeScale is capable of generating high-quality images without any unexpected repetition.

The quantitative results also confirm the superiority of FreeScale. As shown in Table 1, SDXL-DI achieves the

SDXL 1024 ×1024

SDXL-DI ScaleCrafter DemoFusion FouriScale Ours

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

Prompt: A blue unicorn flying over a mystical land.

×4Resolution

×20482048 ×16Resolution

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

Prompt: Astronaut on Mars during sunset.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Prompt: A chihuahua in an astronaut suit floating in space, cinematic lighting, glow effect.

×40964096

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Prompt: Stunning feminine body, commercial image, a beautiful girl from Spain, holographic photography shoots, large body of water sprayed, ……

- Figure 3. Image qualitative comparisons with other baselines. Our method generates both 20482 and 40962 vivid images with better content coherence and local details. Best viewed ZOOMED-IN.

|[Figure 80]|
|---|
|[Figure 81]|

|[Figure 82]|
|---|
|[Figure 83]|

[Figure 84]

[Figure 85]

[Figure 86]

1× Result

[Figure 87]

[Figure 88]

Semantic Mask

Weighted with Scalar

Weighted with Mask

- Figure 4. Results of flexible control for detail level. A better result will be generated by adding the coefficient weight in the area of Griffons and reducing the coefficient weight in the other regions. Best viewed ZOOMED-IN.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- 1× Result No Editing Hair Editing Face Editing

- Figure 5. Results of local semantic editing. FreeScale makes the hair purple or edits the face to make this person look more Japanese in the higher-resolution (40962).

Latent Space Upsampling Ours

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

w/o Scale Fusion Dilated Up-Blocks

- Figure 6. Qualitative image comparisons with ablations. Our full method performs the best. The resolution of results is 40962 for better visualizing the difference between the various strategies.

Table 2. Video quantitative comparisons with baselines. FreeScale achieves the best scores for all metrics.

Method FVD ↓ Dynamic Degree ↑ Aesthetic Quality ↑ Time (min) ↓

VC2-DI [10] 611.087 0.191 0.580 4.077 ScaleCrafter [20] 723.756 0.104 0.584 4.098 DemoFusion [14] 537.613 0.342 0.614 9.302 Ours 484.711 0.383 0.621 3.787

Local Control. FreeScale provides flexible control for detail level in generated results. Figure 4 shows a demo of changing the detail level of different semantic areas. During the process of tailored self-cascade upscaling, we will get 1× results as intermediates. Although more details will be added or modified in the later higher-resolution stages, the overall structure and main content of the image have been determined in the 1× results. It is easy to calculate semantic masks [32] and assign different α for each region in Equation 4. As shown in Figure 4, we will obtain a better result when we add the coefficient weight in the area of Griffons and reduce the coefficient weight in other regions.

In addition, this mechanism can even be extended to local semantic editing. Utilizing semantic mask from 1× results, we can inject different text semantics into different regions in the layers of cross-attention. As shown in Figure 5, FreeScale successfully edits the hair and face in the higher-resolution results.

###### 4.2. Higher-Resolution Video Generation

We compare FreeScale with other tuning-free highersolution video generation methods: (i) VideoCrafter2 [10] direct inference (VC2-DI) (ii) ScaleCrafter [20], and (iii) DemoFusion [14]. FouriScale [25] is not evaluated since its bundled FreeU [47] does not work well in video generation.

best FIDc and KIDc. The reason is that SDXL-DI tends to generate multiple duplicated objects and its crop may be closer to the reference images. However, this behavior will sacrifice the visual structure thus SDXL gains the worst FID, KID and IS in the resolution of 40962. Overall, our approach achieves the best or second-best scores for all quality-related metrics with negligible additional time costs.

As shown in Figure 7, the behavior of VC2-DI and ScaleCrafter are similar to the corresponding version in image generation, tending to generate duplicated whole objects and local parts, respectively. However, DemoFusion has completely unexpected behavior in the video generation. Its Dilated Sampling mechanism brings strange patterns all

A bear running in the ruins, photorealistic, 4k, high definition.

A chihuahua in astronaut suit floating in space, cinematic lighting, glow effect.

VideoCrafter2

(320512)× Scale

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

VC2-DIFusion

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Crafter DemoOurs

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

[Figure 126]

- Figure 7. Video qualitative comparisons with other baselines. While other baselines fail in video generation, FreeScale effectively generates higher-resolution videos with high fidelity. Best viewed ZOOMED-IN. Table 3. Image quantitative comparisons with other ablations. Our final FreeScale achieves better quality-related metric scores in all experiment settings. The best results are marked in bold.

20482 40962

Method

FID ↓ KID ↓ FIDc ↓ KIDc ↓ IS ↑ Time (min) ↓ FID ↓ KID ↓ FIDc ↓ KIDc ↓ IS ↑ Time (min) ↓

w/o Scale Fusion 75.717 0.017 76.536 0.026 12.743 0.614 68.115 0.012 100.065 0.037 12.415 4.566 Dilated Up-Blocks 75.372 0.017 76.673 0.025 12.541 0.861 67.447 0.011 98.558 0.035 12.543 6.245 Latent Space Upsampling 72.454 0.015 71.793 0.023 12.210 0.840 65.081 0.009 88.632 0.029 11.307 6.113 Ours 44.723 0.001 36.276 0.006 12.747 0.853 49.796 0.004 71.369 0.029 12.572 6.240

over the frames and Skip Residual operation makes the whole video blur. In contrast, our FreeScale effectively generates higher-resolution videos with high fidelity. Table 2 exhibits that our method achieves the best FVD, dynamic degree and aesthetic quality. In addition, the time cost saved by skipping certain timesteps near pure noise (transparent blocks in Figure 2) even outweighs the extra time caused by other modules in FreeScale.

ated results due to small repetition problems. Table 3 shows that our final FreeScale achieves better quality-related metric scores in all experimental settings.

##### 5. Conclusion

This study introduces FreeScale, a tuning-free inference paradigm designed to enhance high-resolution generation capabilities in pre-trained diffusion models. By leveraging multi-scale fusion and selective frequency extraction, FreeScale effectively addresses common issues in highresolution generation, such as repetitive patterns and quality degradation. Experimental results demonstrate the superiority of FreeScale in both image and video generation, surpassing existing methods in visual quality while also having significant advantages in inference time. FreeScale not only eliminates various forms of visual repetition but also ensures detail clarity and structural coherence in generated visuals. Additional local control capabilities provide users with more flexibility. Eventually, FreeScale achieves unprecedented 8k-resolution text-to-image generation.

###### 4.3. Ablation Study

The proposed FreeScale mainly consists of three components: (i) Tailored Self-Cascade Upscaling, (ii) Restrained Dilated Convolution, and (iii) Scale Fusion. To visually demonstrate the effectiveness of these three components, we conducted ablations on the SDXL generating 20482 and 40962 images. First, we show the advantage of upsampling in RGB space. As shown in Figure 6, upsampling in latent space brings certain artifacts in the lion’s eyes. Then dilating the convolution in up-blocks or removing Scale Fusion will cause some cluttered textures that appear in the gener-

##### 6. Acknowledgements

This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG2-PhD-2022-01-035T), the Ministry of Education, Singapore, under its MOE AcRF Tier

- 2 (MOE-T2EP20221-0012, MOE-T2EP20223-0002), and Alibaba Group. Special thanks to Yingqing He and Lanqing Guo for sharing their invaluable experience and advice in getting us started quickly in the higher-resolution visual generation task. References

- [1] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In IEEE International Conference on Computer Vision, 2021. 1
- [2] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. arXiv preprint arXiv:2302.08113, 2023. 2, 3, 5
- [3] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, et al. Lumiere: A spacetime diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. 2
- [4] Mikołaj Bi´nkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. arXiv preprint arXiv:1801.01401, 2018. 5
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [6] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2
- [7] Boyuan Cao, Jiaxin Ye, Yujie Wei, and Hongming Shan. Ap-ldm: Attentive and progressive latent diffusion model for training-free high-resolution image generation. arXiv preprint arXiv:2410.06055, 2024. 3
- [8] Lucy Chai, Michael Gharbi, Eli Shechtman, Phillip Isola, and Richard Zhang. Any-resolution training for highresolution image synthesis. In European Conference on Computer Vision, pages 170–188. Springer, 2022. 5
- [9] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [10] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 1, 2, 5, 7

- [11] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,

2023. 1, 2

- [12] Jiaxiang Cheng, Pan Xie, Xin Xia, Jiashi Li, Jie Wu, Yuxi Ren, Huixia Li, Xuefeng Xiao, Min Zheng, and Lean Fu. Resadapter: Domain consistent resolution adapter for diffusion models. 2024. 3
- [13] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [14] Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising highresolution image generation with no $$$. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6159–6168, 2024. 2, 3, 4, 5, 7, 1
- [15] Ruoyi Du, Dongyang Liu, Le Zhuo, Qin Qi, Hongsheng Li, Zhanyu Ma, and Peng Gao. I-max: Maximize the resolution potential of pre-trained rectified flow transformers with projected flow. arXiv preprint arXiv:2410.07536, 2024. 5
- [16] Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation. arXiv preprint arXiv:2402.10491, 2024. 2, 3, 4
- [17] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2
- [18] Moayed Haji-Ali, Guha Balakrishnan, and Vicente Ordonez. Elasticdiffusion: Training-free arbitrary size image generation through global-local content separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6603–6612, 2024. 3
- [19] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity video generation with arbitrary lengths. arXiv preprint arXiv:2211.13221, 2022. 2
- [20] Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higherresolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 4, 5, 7, 1
- [21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 5
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2
- [23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2

- [24] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–13232. PMLR, 2023. 3
- [25] Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. arXiv preprint arXiv:2403.12963, 2024. 2, 3, 5, 7, 1
- [26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023. 5
- [27] Juno Hwang, Yong-Hyun Park, and Junghyo Jo. Upsample guidance: Scale up diffusion models without training. arXiv preprint arXiv:2404.01709, 2024. 3
- [28] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. 2024. 2
- [29] Zhiyu Jin, Xuli Shen, Bin Li, and Xiangyang Xue. Trainingfree diffusion model adaptation for variable-sized text-toimage synthesis. Advances in Neural Information Processing Systems, 36:70847–70860, 2023. 3
- [30] Gwanghyun Kim, Hayeon Kim, Hoigi Seo, Dong Un Kang, and Se Young Chun. Beyondscene: Higher-resolution human-centric scene generation with pretrained diffusion. In European Conference on Computer Vision, pages 126–142. Springer, 2024. 3
- [31] Younghyun Kim, Geunmin Hwang, Junyu Zhang, and Eunbyung Park. Diffusehigh: Training-free progressive highresolution image synthesis through structure guidance. arXiv preprint arXiv:2406.18459, 2024. 3
- [32] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 7
- [33] Black Forest Labs. Flux.1 : An advanced state-of-the-art generative deep learning model. Technical report, Black Forest Labs, 2024. 5
- [34] Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems, 36:50648–50660, 2023. 3
- [35] Mingbao Lin, Zhihang Lin, Wengyi Zhan, Liujuan Cao, and Rongrong Ji. Cutdiffusion: A simple, fast, cheap, and strong diffusion extrapolation method. arXiv preprint arXiv:2404.15141, 2024. 3
- [36] Zhihang Lin, Mingbao Lin, Meng Zhao, and Rongrong Ji. Accdiffusion: An accurate method for higher-resolution image generation. arXiv preprint arXiv:2407.10738, 2024. 3
- [37] Songhua Liu, Weihao Yu, Zhenxiong Tan, and Xinchao Wang. Linfusion: 1 gpu, 1 minute, 16k image. 2024. 3
- [38] Xinyu Liu, Yingqing He, Lanqing Guo, Xiang Li, Bu Jin, Peng Li, Yan Li, Chi-Min Chan, Qifeng Chen, Wei

- Xue, et al. Hiprompt: Tuning-free higher-resolution generation with hierarchical mllm prompts. arXiv preprint arXiv:2409.02919, 2024. 3
- [39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [40] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 1, 2, 5, 3
- [41] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023. 5
- [42] Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra-high-resolution image synthesis to new peaks. arXiv preprint arXiv:2407.02158, 2024. 3
- [43] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 2
- [44] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 5
- [45] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2024. 3

- [46] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022. 1
- [47] Chenyang Si, Ziqi Huang, Yuming Jiang, and Ziwei Liu. Freeu: Free lunch in diffusion u-net. In CVPR, 2024. 2, 5, 7
- [48] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 1
- [49] Jiayan Teng, Wendi Zheng, Ming Ding, Wenyi Hong, Jianqiao Wangni, Zhuoyi Yang, and Jie Tang. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023. 3
- [50] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 5
- [51] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report, 2023. 1, 2
- [52] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with

- pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1905–1914, 2021. 1
- [53] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. Videocomposer: Compositional video synthesis with motion controllability. NeurIPS, 2023. 2
- [54] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024. 2
- [55] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2, 5
- [56] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. In CVPR, 2024.
- [57] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 1, 2
- [58] Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Yuhao Chen, Yao Tang, and Jiajun Liang. Hidiffusion: Unlocking higherresolution creativity and efficiency in pretrained diffusion models. In European Conference on Computer Vision, pages 145–161. Springer, 2024. 3
- [59] Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, and Hang Xu. Any-sizediffusion: Toward efficient text-driven synthesis for any-size hd images. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 7571–7578, 2024. 3

### FreeScale: Unleashing the Resolution of Diffusion Models via Tuning-Free Scale Fusion

#### Supplementary Material

Overview. In the supplementary material, we introduce implementation details in Section A, show more evaluations in Section B, exhibit more results in Section C, and finally, discuss limitations and future work in Section D.

##### A. Implementation Details

During sampling, we perform DDIM sampling [48] with 50 denoising steps, setting DDIM eta to 0. For image generation, the base inference resolution of SDXL is 1024 × 1024 pixels, and the scale of the classifier-free guidance is set to 7.5. For video generation, the base inference resolution of VideoCrafter2 is 320 × 512, the video length is 16 frames, and the scale of the classifier-free guidance is set to 12.0.

For tailored self-cascade upscaling, we set K = 700 in Equation 3 for all experiments. And in Equation 4, α is set as a scaler, 2, by default. To avoid excessive and messy textures in generating 8k images, α is reduced to 1. In Figure 4, α is 3 and 0.5 in the targeted and other areas, respectively. Users can further adjust these parameters according to the detailed requirements of different images. For restrained dilated convolution, the dilation factor d in Equation 5 is equal to the resolution level (1 represents original resolution, 2 represents the twice height and width). For scale fusion, the kernel size is 2 × height × width ÷ (1024 × 1024) − 1 and the standard deviation is 1 in Equation 7.

Datasets. We evaluate image generation on the LAION5B dataset [46] with 1024 randomly sampled captions. Specifically, to better align with human preference, we randomly selected prompts from the LAION-Aesthetics-V26.5plus dataset to evaluate image generation. The LAIONAesthetics-V2-6.5plus is a subset of the LAION 5B dataset, characterized by its high visual quality, where images have scored 6.5 or higher according to aesthetic prediction models. Regarding the evaluation of video generation, we use randomly sampled 512 captions from the WebVid-10M dataset [1].

##### B. More Evaluation B.1. Comparison with Super-Resolution

Different from traditional super-resolution (SR) tasks. Higher-resolution generation aims to tap the potential of the pre-trained model itself. Therefore, the performance of the higher-resolution generation method is based on the base model rather than another additional SR model. We compare our method with a super-resolution post-processing setting: SDXL+Real-ESRGAN [52]. As shown in Table 4,

- Table 4. Image quantitative comparisons with superresolution. Compared to super-resolution post-processing setting SDXL+Real-ESRGAN, FreeScale also achieves competitive performance. As reported in most previously published related works, higher-resolution generation methods are hard to beat SR methods completely on quantitative metrics due to the difference in difficulty between the two tasks.

Method FID ↓ KID ↓ FIDc ↓ KIDc ↓ IS ↑

SDXL+Real-ESRGAN [52] 43.476 0.000 73.524 0.024 12.599 Ours 49.796 0.004 71.369 0.029 12.572

- Table 5. User study. Users are required to pick the best one among our proposed FreeScale with the other baseline methods in terms of image-text alignment, image quality, and visual structure.

Method Text Alignment Image Quality Visual Structure

SDXL-DI [40] 0.87% 0.00% 0.00% ScaleCrafter [20] 7.83% 5.22% 7.83% DemoFusion [14] 17.39% 14.35% 18.26% FouriScale [25] 2.17% 2.61% 1.74% Ours 71.74% 77.83% 72.17%

- Table 6. User study for Video Generation. Users are required to pick the best one among our proposed FreeScale with the other baseline methods in terms of text alignment, cover quality, and video quality.

Method Text Alignment Cover Quality Video Quality

VC2-DI 5.38% 4.62% 3.85% ScaleCrafter 4.62% 5.38% 0.77% DemoFusion 30.00% 26.92% 30.77% Ours 60.00% 63.08% 64.62%

FreeScale achieves competitive performance in quantitative metrics. As reported in most previously published related works [14, 20], higher-resolution generation methods are hard to beat SR methods completely on quantitative metrics due to the difference in difficulty between the two tasks. However, Figure 8 shows that FreeScale is not inferior to SDXL+Real-ESRGAN in visual quality, and adds more details. In addition, SR methods will faithfully follow the lowresolution input while FreeScale can regenerate the original blurred areas based on the prior knowledge that the model has learned (the eyes and logos in Figure 8).

###### B.2. User Study

In addition, we conducted a user study to evaluate our results on human subjective perception. Users are asked to watch the generated images of all the methods, where each

[Figure 127]

[Figure 128]

[Figure 129]

|[Figure 130]|[Figure 131]|
|---|---|

|[Figure 132]|[Figure 133]|
|---|---|

|[Figure 134]|[Figure 135]|
|---|---|

SDXL (1024 × 1024) Real-ESRGAN (8192 × 8192) FreeScale (8192 × 8192)

- Figure 8. Image qualitative comparisons with super-resolution. FreeScale is not inferior to SDXL+Real-ESRGAN in visual quality, and adds more details. In addition, SR methods will faithfully follow the low-resolution input while FreeScale can regenerate the original blurred areas based on the prior knowledge that the model has learned. Best viewed ZOOMED-IN.

Table 7. Video quantitative comparisons with other ablations. Our final setting achieves the best or second-best scores for all metrics. The best results are marked in bold, and the second best results are marked by underline.

[Figure 136]

[Figure 137]

Method FVD ↓ Dynamic Degree ↑ Aesthetic Quality ↑ Time (min) ↓

3× Resolution (1024×3072)

[Figure 138]

Dilated Up-Blocks 523.323 0.363 0.611 3.788 RGB Upsampling 422.245 0.381 0.604 3.799 Ours 484.711 0.383 0.621 3.787

example is displayed in a random order to avoid bias, and then pick the best one in three evaluation aspects. A total of 23 users were asked to pick the best one according to the image-text alignment, image quality, and visual structure, respectively. As shown in Table 5, our approach gains the most votes for all aspects, outperforming baseline methods by a large margin.

3× Resolution (3072×1024)

6× Resolution (2048×3072)

Figure 9. Flexible aspect ratio generation. FreeScale can directly achieve a flexible aspect ratio (the resolution must be a multiple of 512) without any adaptation.

We also add a human study for video generation. Users were asked to pick the best one according to the text alignment, cover quality, and video quality, respectively. As shown in Table 6, our method still gains the most votes for all aspects, outperforming baseline approaches significantly.

##### C. More Results

###### C.1. Flexible Aspect Ratio Generation

###### B.3. Ablation Study for Video Generation

As shown in Figure 9, FreeScale can directly achieve a flexible aspect ratio (the resolution must be a multiple of 512) without any adaptation. We also add quantitative experiments for 2048 × 4096 resolution. As shown in Table 8, FreeScale still achieves the best or second-best scores for all metrics.

We also conduct an ablation study for higher-solution video generation. As discussed in the method part, we adopt latent space upsampling in video generation. Table 7 shows that our final setting achieves the best or second-best scores for all metrics.

|[Figure 139]|
|---|
|[Figure 140]|

|[Figure 141]|
|---|
|[Figure 142]|

[Figure 143]

[Figure 144]

|[Figure 145]|
|---|
|[Figure 146]|

|[Figure 147]|
|---|
|[Figure 148]|

[Figure 149]

[Figure 150]

###### Resolution: 1024 × 1024 Resolution: 8192 × 8192

- Figure 10. Zoomed in details for the 8k image. FreeScale may regenerate the original blurred areas at low resolution based on the prior knowledge that the model has learned. As shown in the bottom row, two originally chaotic and blurry faces are clearly outlined at 8k resolution. Best viewed ZOOMED-IN.

[Figure 151]

[Figure 152]

[Figure 153]

SDXL (50 steps) SDXL-Turbo (4 steps) SDXL-Turbo (2 steps)

- Figure 11. Fast generation with SDXL-Turbo. FreeScale can help SDXL-Turbo generate results at 20482 resolution with even 2 timesteps.

Table 8. Image quantitative comparisons with baselines in 2048 × 4096 resolution. FreeScale still achieves the best or second-best scores for all metrics.

Method FID ↓ KID ↓ FIDc ↓ KIDc ↓ IS ↑ SDXL-DI 97.493 0.026 38.273 0.009 7.258 ScaleCrafter 97.235 0.032 107.582 0.050 8.001 DemoFusion 72.196 0.019 91.264 0.044 10.622 FouriScale 95.891 0.032 118.306 0.061 8.422 Ours 54.704 0.004 65.584 0.025 11.323

###### C.3. Gallery of 8k Images

###### C.2. Fast Generation with SDXL-Turbo

Figure 12 illustrates the effectiveness of FreeScale on generating ultra-high-resolution images (i.e., 8k-resolution images). As shown in Figure 10, FreeScale effectively enhances local details without compromising the original visual structure or introducing object repetitions. Different from simple super-resolution, FreeScale may regenerate the original blurred areas at low resolution based on the prior knowledge that the model has learned. In Figure 10, two

FreeScale can easily be compatible with other models with similar structures. SDXL-Turbo [45] is a distilled version of SDXL [40] and can produce similar quality results with 2 ∼ 4 timesteps. However, SDXL-Turbo can only generate results at 5122 resolution due to the knowledge loss during distillation. As shown in Figure 11, FreeScale can help SDXL-Turbo generate results at 20482 resolution.

[Figure 154]

[Figure 155]

[Figure 156]

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

[Figure 160]

[Figure 161]

[Figure 162]

|[Figure 163]|
|---|

|[Figure 164]|
|---|

|[Figure 165]|
|---|

[Figure 166]

[Figure 167]

[Figure 168]

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

[Figure 172]

[Figure 173]

[Figure 174]

|[Figure 175]|
|---|

|[Figure 176]|
|---|

|[Figure 177]|
|---|

###### Figure 12. Gallery of generated 8k images. We place the original-resolution result in the lower right corner for reference. FreeScale effectively enhances local details without compromising the visual structure or introducing object repetitions. Best viewed ZOOMED-IN.

[Figure 178]

[Figure 179]

[Figure 180]

SDXLFLUX

[Figure 181]

[Figure 182]

[Figure 183]

1024x1024 2048x2048 4096x4096

- Figure 13. Structure gap. UNet-based LDMs and DiT-based LDMs will face different challenges in the higher-resolution generation task. UNet-based LDMs face repetition problems while DiT-based LDMs face blur problems.

originally chaotic and blurry faces are clearly outlined at 8k resolution.

Visual Enhancement. FreeScale also supports using existing images to replace the intermediate 1× result. Compared to SDXL [40], FLUX [33] is better in visual text generation. In the center of Figure 12, we first use FLUX to generate the intermediate 1× result, a dragon with “FreeScale”. Then we utilize the remaining pipeline of FreeScale to generate the final 8k-resolution result. In this sense, FreeScale is also a tool to upscale resolution and enhance detail.

##### D. Limitations and Future Work

Inference Cost. We employ the scale fusion only in the self-attention layers thus bringing negligible time cost. And the omitted time steps almost offset the additional cost of tailored self-cascade upscaling. As a result, the inference cost of FreeScale is close to the direct inference by the base model. However, the inference cost is still huge for ultrahigh-resolution generation. In future work, when users require image generation at resolutions exceeding 8k, memory constraints may be mitigated through multi-GPU inference strategies, while computational efficiency can be enhanced by employing inference acceleration techniques.

Knowledge Limitation. Even ignoring the limitations of the computation, there is a limit to the upscaling capability of FreeScale. When the desired resolution is beyond the prior knowledge that the model has learned, no more details can be reasonably added. In other words, the endless higher-resolution result will have either the same level of detail or unnatural messy detail. In addition, as a tuningfree framework, FreeScale’s performance relies heavily on base models. During the tailored self-cascade process, the intermediate 1× result is equivalent to direct inference with base models. Some artifacts caused by inherently flawed (e.g., extra legs), will be inherited in further upscaling.

Structure Gap. DiT-based LDMs (e.g., FLUX [33] and CogVideoX [55]), have showcased impressive visual generation capabilities recently. However, UNet-based LDMs and DiT-based LDMs will face different challenges in the higher-resolution generation task. As shown in Figure 13, UNet-based LDMs face repetition problems while DiTbased LDMs face blur problems. Most previous higherresolution generation methods either support the UNetbased LDMs (DemoFusion [14], and FouriScale [25]) or DiT-based LDMs (I-MAX [15]), in line with the common sense that different problems require different strategies to solve. To support the DiT based structure, FreeScale also needs to be customized specifically.

