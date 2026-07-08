### StyleMe3D: Stylization with Disentangled Priors by Multiple Encoders on 3D Gaussians

Cailin Zhuang1,2,3 Yaoqi Hu3 Xuanyang Zhang2† Wei Cheng2 Jiacheng Bao1 Shengqi Liu2 Yiying Yang2 Xianfang Zeng2 Gang Yu2 Ming Li4‡

1ShanghaiTech University 2StepFun 3AIGC Research 4Guangming Laboratory https://styleme3d.github.io/

# arXiv:2504.15281v2[cs.CV]5Feb2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

##### Next-View Style Out-painting ✕ Hierarchical Disentangled Priors

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

|[Figure 20]<br><br>[Figure 21]|
|---|

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

Style Reference Image

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

Low-levelOnly

|[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>Previous|
|---|

[Figure 84]

[Figure 85]

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

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

|[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>Our|
|---|

[Figure 115]

Multi-level

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

|[Figure 119]|
|---|

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Hierarchical Stylization Stylization with the Same Style Stylization on the Same Object

Figure 1. Our StyleMe3D enables hierarchical, versatile and high-quality 3D stylization across diverse styles.

cess, we propose a multi-modal alignment strategy using the CLIP latent space: a CLIP-based style stream evaluator (Contrastive Style Descriptor) that enforces middlelevel stylistic similarity, and a CLIP-based content stream evaluator (3D Gaussian Quality Assessment) that acts as a global regularizer to mitigate typical GS quality degradation. Finally, a VGG-based Simultaneously Optimized Scale module is integrated to refine fine-grained texture details at the low-level. Extensive experiments demonstrate that our method consistently preserves intricate geometric details and achieves coherent stylistic effects across entire scenes, significantly surpassing state-of-the-art baselines in both qualitative and quantitative evaluations.

#### Abstract

Current 3D Gaussian Splatting stylization approaches are limited in their ability to represent diverse artistic styles, frequently defaulting to low-level texture replacement or yielding semantically inconsistent outputs. In this paper, we introduce StyleMe3D, a novel hierarchical framework that achieves comprehensive, high-fidelity stylization by disentangling multi-level style representations while preserving geometric fidelity. The cornerstone of StyleMe3D is Dynamic Style Score Distillation (DSSD), which harnesses latent priors from a style-aware diffusion model to provide high-level semantic guidance, ensuring robust and expressive style transfer. To further refine this distillation pro-

Project lead: †X. Zhang; Corresponding author: ‡M. Li.

#### 1. Introduction

The advent of 3D Gaussian Splatting (GS) [19] has revolutionized 3D scene representation, delivering high reconstruction fidelity and real-time rendering. This advancement has broadened the horizons of 3D stylization, enabling direct scene editing towards given styles. Nevertheless, current methods remain notably limited in their capacity for expressive artistic control. They can be mainly divided into two categories, i.e., single-template color matching methods and texture transfer counterparts. A typical method of the former is Ref-NPR [55], which depends solely on global style exemplars and consequently fails to capture fine-grained textures. The texture transfer methods [7, 10, 25, 52] adapt basic color and patterns using VGG-based representations [34] but lack semantic understanding, resulting in an inability to preserve coherent semantics. These shortcomings are particularly pronounced with abstract styles and complex scene layouts, as existing methods focus exclusively on distilling a single level of style representation.

To address these limitations, we introduce a novel hierarchical framework that establishes a new paradigm for semantic-aware GS stylization. Our central innovation is the integration of multi-level style understanding and transfer within a single, unified framework. By disentangling stylistic features across low, middle, and high semantic levels, our framework achieves expressive artistic transformation while strictly upholding the structural integrity of the underlying 3D content.

At the heart of our framework lies a suite of specialized modules, each tailored to address different facets of 3D stylization. For high-level representation, we leverage the latent space of a style-enhanced diffusion model [33] to achieve robust, content-aware alignment between style exemplars and 3D geometry, marking the first application of such semantic priors in GS stylization. To complement this core engine, we introduce a CLIP-based dual-stream alignment mechanism that operates on the multi-modal latent space: (1) a Style Stream, implemented via the Contrastive Style Descriptor (CSD) [35], which is pre-trained on diverse style datasets and employs a CLIP-based evaluator to enforce mid-level stylistic consistency; and (2) a Content Stream, realized through 3D Gaussian Quality Assessment (3DG-QA) [42], which leverages contrastive aesthetic priors and acts as a semantic regularizer to mitigate geometric degradation. Finally, a Simultaneously Optimized Scale (SOS) [10] module is incorporated to refine low-level texture details. Collectively, these innovations establish a comprehensive and flexible solution that substantially advances the expressive power and quality of GS stylization.

We validate our method on both 3D object dataset (i.e., NeRF synthetic [27]) and scene datasets (i.e., tandt db [19] and mip-NeRF 360 [1]), demonstrating its generalization

ability across various geometric complexities and artistic styles. Our approach consistently achieves superior stylization fidelity compared to state-of-the-art (SOTA) baselines. Qualitatively, our method excels at preserving fine geometric structures while establishing scene-level stylistic coherence throughout diverse content. Quantitatively, our approach achieves remarkable improvements in CLIPbased style similarity and LPIPS/PSNR/SSIM metrics over prior methods.

Our main contributions are summarized as follows:

- • Hierarchical 3D Stylization: We present the first multilevel prior-disentangled 3D Gaussian Splatting framework, enabling geometry-preserving stylization that captures low-level, mid-level, and high-level artistic attributes.
- • Semantic-Aware Style Transfer: By leveraging a styleenhanced diffusion latent space prior (DSSD), our approach enables robust semantic alignment and expressive style transfer, surpassing the limitations of conventional VGG-based methods.
- • CLIP-based Dual-stream Alignment: We introduce a aesthetic-driven synergistic alignment strategy comprising a CLIP-based Style Stream (CSD) for attribute consistency and a CLIP-based Content Stream (3DGQA) for maintaining structural and semantic fidelity.

#### 2. Related Works

###### 2.1. 2D Generation and Stylization

2D generation has rapidly advanced across generative modeling, customization, conditional control, editing, and stylization. Initial breakthroughs in 2D synthesis with VAEs and GANs [2, 14, 18] were furthered by diffusion models [21, 33, 48, 51], enhancing image quality and diversity for complex manipulation. Stylization advances emphasize style-content separation, with cross-attention-based transfer [6, 40, 50] and shared attention mechanisms for coherence [47]. Frequency-domain techniques aid diffusion control [12], while Aligning style with textual cues [23], cross-domain fusion [31] and FFT-based transfer [15] expand style applications. In this paper, we aim to style 3D GS and these 2D methods give us a lot of insights and priors that can be reused in the 3D field.

###### 2.2. 3D Generation and Stylization

Text-guided 3D generation has also advanced with Score Distillation Sampling (SDS) [32] and its variants [39, 44], enabling controllable, diverse 3D synthesis. These techniques support artistic scene generation [22] and multimodal inputs (text and image) [36, 37, 45, 49]. Recent improvements in latent diffusion models further enhance the expressiveness and creative potential of text-to-3D generation [49, 56], and more and more multi-view [3, 4, 30] and

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Dynamic Style Score Distillation (DSSD)

Contrastive Style Descriptor (CSD)

###### Style Purification

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

+Style Purification

StyleContent

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

###### Style Adapter

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Latents

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Style Ref

Style Ref Render Img

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

###### Middle-levelCLIP-basedDual-streamAlignPrior

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

GPT4/Qwen

[Figure 178]

[Figure 179]

[Figure 180]

StylizationwithHierarchicalPriors

Diffusion Model

###### CSD-CLIP Vision

[Figure 181]

[Figure 182]

Style Ref

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Style Decoupling

+Style Purification

[Figure 191]

[Figure 192]

[Figure 193]

Style: cartoonish, vibrant colors, bold outlines

[Figure 194]

[Figure 195]

VAE

[Figure 196]

[Figure 197]

[Figure 198]

DSSD Loss

[Figure 199]

[Figure 200]

[Figure 201]

Content : wooden path, green forest, mountains, sky, clouds, trees

[Figure 202]

Render Img

[Figure 203]

Ref-Style Render-Style

High-level Diffusion-based Semantic Prior

[Figure 204]

Cosine Similarity

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

Simultaneously Optimized Scale (SOS)

Style Outpainting

[Figure 214]

[Figure 215]

[Figure 216]

GaussianRenderer

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

###### Render-Content Quality Text

[Figure 227]

[Figure 228]

Contentfeatures

[Figure 229]

VGG Grammatrices

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

SOS Loss

[Figure 234]

[Figure 235]

[Figure 236]

VGG

CLIP Image CLIP Text

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

Negative Prompts

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Positive Prompts

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Original Img

Style Ref

[Figure 261]

[Figure 262]

[Figure 263]

Render Img

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Render Img

(3DG-QA) 3D Gaussian Quality Assessment

Next-view Stylization

Low-level VGG-based Visual Prior

- Figure 2. Overview of our StyleMe3D framework. (a) Style Purification: Extracts and refines stylistic representations within the CLIP latent space by decoupling style from the content of the reference image. (b) Hierarchical Stylization: The Dynamic Style Score Distillation (DSSD) module employs dynamic noise scheduling and adaptive style guidance, integrating latent losses to achieve consistent stylization step by step. Integrates three specialized components within DSSD: Simultaneously Optimized Scale (SOS): Refines low-level texture details through adaptive noise-scale optimization. Contrastive Style Descriptor (CSD): Serves as the CLIP-based Style Stream to separates mid-level style and content via contrastive learning for style similarity score. 3D Gaussian Quality Assessment (3DGQA): Acts as the CLIP-based Content Stream for quality-guided refinement, utilizing antonymic semantic prompts to preserve structural integrity. (c) Progressive Next-view Optimization (Style Outpainting): Progressive outpainting achieves novel view style propagation. Ensures global coherent through iterative latent alignment, eliminating multi-view dependencies.

#### 3. Method: StyleMe3D

3D dataset [8, 46] still stimulate the development of this field.

In this section, we elaborate on our comprehensive algorithmic framework for 3D style transfer using 2D priors. We first formally define our core task: performing style transfer on reconstructed 3D Gaussian Splatting (3D GS) representations while preserving structural fidelity in Sec.3.1. To address the inherent challenges in cross-dimensional style adaptation, we propose StyleMe3D, a systematic framework comprising mixture of four encoders that collectively resolve critical challenges in 3D style migration from Sec.3.3 to Sec.3.6 and is unified in Sec.3.7.

For 3D stylization, methods like [17, 28] embed styles directly into 3D structures, while radiance field-based methods [29, 38, 53] achieve style transfer through optimization for enhanced scene realism. Though HyperNet [5] enables arbitrary style embedding in MLPs, it suffers from slow rendering and detail loss, while StyleRF [24] offers zero-shot stylization by transforming radiance field features but lacks adaptability and control.

###### 3.1. Preliminaries

Recent advances in 3D stylization have explored various techniques to embed artistic styles into 3D content, with reference-based methods like [26, 55] for controlled stylization and arbitrary reference techniques [25, 52] for flexible style transfer. Scalable 3D style transfer brings the 3d stylized resolution up to 4K by SOS Loss [10]. Stylized Score Distillation [20] and 3D-aware diffusion models [45] further expand these capabilities.

We define initial 3D GS Reconstruction as a pre-trained task, while redefining 3D GS stylization as a post-training task. Unlike conventional 3D generation tasks that begin from scratch, our approach applies stylization to prereconstructed 3D gausion for both 3D objects and scenes, allowing for enhanced control over style application while preserving the underlying geometry. Firstly, we define the 3D GS reconstruction process as:

Unlike previous works, we conduct a systematic analysis of 3D GS stylization and propose a comprehensive framework to achieve hierarchical, multi-granular style transfer.

N

1 N

MSE(R(Cv;Θ),Ivgt) (1)

min

Θ

v=1

Gaussians

where Θ = {(ui,Σi,αi,ci,0,(ci,j,k)j,k))}N

i=1

represents the 3D gaussian, ci,0 is the main color and ci,j,k is the coefficient. R(Cv;Θ) means render 3D gaussian and Iigt means the ground truth image from the viewpoint Cv respectively.

After obtaining the optimized 3D gaussian, we further formulate the 3D gaussian style transfer process with 2D prior as follows:

N

1 N

L(R(Cv;Θ);ϕ,R) (2)

min

Θ

v=1

where ϕ means the 2D prior and R means the reference prompt, like text prompts or image prompts. L means the loss function to further optimize the 3D gaussian which is initialized with Θ from Eq.1.

In the style transfer task, we aim to only change the 3D gaussian stylization rather than the geometry content. We achieve geometry-style decoupling in 3D gaussian by leveraging the inherent separation of geometric and color parameters in its parametric representation. Specifically, our style transfer framework exclusively optimizes the color parameters Θcolor while maintaining frozen geometric attributes during the stylization process as:

1 N

min

Θcolor

N

L(R(Cv;Θ);ϕ,R) (3)

v=1

We further discuss how to instantiate the L, ϕ and R with different formulations and jointly improve the stylization effectiveness in the following sections.

###### 3.2. Next-view Stylization: Progressive Style Outpainting

Progressive Style Outpainting (PSO) is a novel style guidance method for consistent and detailed style propagation in multi-view 3D stylization (see Fig. 2). Using 2D style priors provided by an image stylization diffusion model [11], we redefine multi-view guidance as a progressive outpainting task. By integrating sparse-view RGB loss with dense-view SDS loss, PSO ensures consistent 3D stylization across views. Instead of random view selection, our method incrementally propagates style information to adjacent views, enhancing style coherence with each step. Specifically, PSO consists of two primary guidance modes, namely gobal guidance and local guidance.

Global Guidance. In the global guidance stage, a uniform noise level is applied to all views before stepwise reduction, defined as:

istep nview mod nopt

αstep =

nopt

(4)

where nview represents the total number of rendering views and nopt denotes the required optimizations per view, managed iteratively by istep.

Local Guidance. Local guidance focuses on single-view optimization, maximizing stylization quality for individual views, albeit at the potential expense of global consistency. The local guidance schedule is defined as:

istep mod nopt nopt

(5)

αstep =

The effectiveness of these modes in balancing stylization strength and consistency is discussed in Sec. 4.3. To maximize the stylization outcome, we combine both guidance modes for complementary strengths.

###### 3.3. High-level Guidance: Dynamic Style Score Distillation

In this section, we distill the prior from the 2D stable diffusion model [33] and use both text and image prompt for style transfer.

Style Purification. Inspired by InstantStyle [40], we use a pre-trained CLIP model for Style Purification to isolate pure style information. In CLIP space, we filter out styleirrelevant details by subtracting content descriptors from style embeddings. Specifically, descriptions of the style reference image are generated using a captioning model (e.g., GPT-4V) to distinguish content-related descriptors. The CLIP Text Encoder extracts a Content Text Embedding (or both content and style) from these descriptors, while the CLIP Image Encoder produces a Style Image Embedding. Subtracting Content Text Embedding from Style Image Embedding (and adding Style Text Embedding) yields a Final Style Embedding containing only style-related information. The style purification process is shown in Fig. 2.

Fine Timestep Sampling. Fine timestep sampling enhances temporal resolution by focusing on low-noise intervals for more granular optimization, with noise progressively decreasing from high to low levels. This sampling strategy is formulated as:

t = Round((1 − αstep0.5) · T).clip(Tmin,Tmax) (6)

where T denotes the total timesteps, with Tmin and Tmax setting the bounds. Higher noise initialization effectively eliminates outlier gaussian, refining the stylization outcome.

Style Distillation. As shown in Fig. 2(b). DSSD further extends score distillation by applying a dynamic CFG (Classifier-Free Guidance) [16] scale coefficient to optimize the intensity of style guidance. Fixed CFG values can lead to oversmoothing (low CFG) or oversaturation (high CFG). To counter this, we introduce a dynamic guidance coefficient that adaptively balances fixed CFG values throughout optimization. The adaptive coefficient is defined as:

∆λ = max 7.5, λmax · αstep2 (7)

With this method, we extend the SSD proposed by [20], and define the style loss in latent space as:

model (variants of ViT [9], like ViT-B and ViT-L) for the representation of the image style. The ViT is trained with both self-supervised learning and supervised objectives, which can extract image descriptors with concise and effective style information. CSD leverage the ViT to extract style feature from rendered images and reference image respectively and then calculate the pairwise cosine silimarity score. Finally, the CSD loss term reduces to:

DSSDz2D = (1−∆λs)ϵϕ2D(zt

s|y,ts)+∆λsϵˆϕ2D(zt

s|y,s,ts)−ϵs,

(8) where ϵϕ2D(·) is the predicted noise by the style-based 2D diffusion prior ϕ.

Our Dynamic Style Score Distillation (DSSD) objective function as follows:

N

1 N

∇ΘcolorLDSSD(x = R(Cv;Θcolor);ϕ,R) = Etz s,ϵzs ω(t) DSSDz2D

(1 − cos(ϕViT(Iv),ϕViT(Iref))) (12)

LCSD =

v=1

∂x ∂θ

(9)

###### 3.6. CLIP-based Content Stream: 3D Gaussian Quality Assessment

where ω(t) is a weighting function regulating timestep contributions.

In addition to preserving the original content and migration style of 3D gaussian, we also need to ensure the overall aesthetics quality between the migrated style and content. CLIP-IQA [41] has been developed to evaluate the look or quality of an image. CLIP-IQA leverages CLIP for perception assessment and calculate the cosine similarity between the feature embeddings of the given text promt and image as follows:

Further, we optimize the stylized multi-view image Irgb and the associated mask Imask for alignment with the input data. If required, additional loss terms such as SSIM loss [43] or LPIPS loss [54] may be integrated to enhance alignment. Thus, our final objective function is:

Lstyle = λDSSDLDSSD + λRGBLRGB + λmaskLmask (10) This setup ensures multi-view consistency in 3D styliza-

x ⊙ t ||x|| ∗ ||t||

(13)

s =

tion, achieving refined style expression and geometric fidelity through the dynamic coefficient adjustment and adaptive optimization strategy.

where x ∈ RC and t ∈ RC represents the image embedding and text embedding, C is the embedding channel dimension. CLIP-IQA further introduces antonym prompts (e.g., “Good photo.” and “Bad photo.”) to address the linguistic ambiguity. t1 and t2 are obtained from text prompts with good quality and bad quality respectively and the si can be obtained with the corresponding ti, then the final CLIP-IQA score can be formulate as:

###### 3.4. Low-Level Refinement: Simultaneously Optimized Scale

To further enhance the texture details of 3D gaussian, multiscale stylization strategy is introduced into the optimization process, called Simultaneously Optimized Scale (SOS) [10]. Following the silimar approach from [13], we employ VGG-19 [34] to extract high-resolution texture features through its shallow convolutional layers. We use N rendered images (each image is represented as Iv) from the source 3D gaussian and style reference image Iref to compute multi-scale Gram matrix correlations and formulate the style objective as follows:

es

1

(14)

s =

es1 + es2

We adopt the CLIP-IQA property and extend CLIP-IQA to the 3D style transfer field to ensure the perception quality of 3D gaussian. More specifically, we define the 3D gaussian Quality Assessment (3DG-QA) as a objective term as:

N

1 N

∥G(ϕlVGG(Iv)) − G(ϕlVGG(Iref))∥22

LSOS =

v=1 l∈Ls

(11)

where G(·) denotes Gram matrix computation,ϕlVGG represents features from the l-th VGG layer and Ls = {ReLU k 1,k ∈ 1,3,5}.

###### 3.5. CLIP-based Style Stream: Contrastive Style Descriptor

To further align to the style between the 3D gaussian and the given reference image, we utilize Contrastive Style Descriptor (CSD) [35], which aims to build a high-performance

1 N

L3DG-QA =

N

(1 − sv) (15)

v=1

where v means the viewpoint index rendered from the 3D gaussian representation.

###### 3.7. Stylization with Hierarchical Priors

StyleMe3D systematically addresses five fundamental aspects in 3D gaussian stylization: (1) Style-content decoupling, (2) Adaptive style conditioning, (3) Multi-scale feature alignment, (4) Texture detail enhancement, and (5) Global aesthetic optimization with four principal components. The DSSD stablishes effective style conditioning

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

###### 3D Object & Style-Ref Stylized 3D Object 3D Scene & Style-Ref Stylized 3D Scene

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

|[Figure 283]|
|---|

[Figure 284]

[Figure 285]

|[Figure 286]|
|---|

[Figure 287]

[Figure 288]

|[Figure 289]|
|---|

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

|[Figure 295]|
|---|

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

|[Figure 309]|
|---|

[Figure 310]

[Figure 311]

[Figure 312]

|[Figure 313]|
|---|

[Figure 314]

[Figure 315]

|[Figure 316]|
|---|

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

|[Figure 328]|
|---|

[Figure 329]

[Figure 330]

|[Figure 331]|
|---|

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

|[Figure 338]|
|---|

[Figure 339]

[Figure 340]

[Figure 341]

|[Figure 342]|
|---|

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

|[Figure 348]|
|---|

[Figure 349]

[Figure 350]

|[Figure 351]|
|---|

[Figure 352]

[Figure 353]

[Figure 354]

|[Figure 355]|
|---|

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

|[Figure 368]|
|---|

[Figure 369]

[Figure 370]

|[Figure 371]|
|---|

[Figure 372]

|[Figure 373]|
|---|

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

- Figure 3. Stylization Visualizations with various reference images. Demonstration of our method’s performance across five styles (vangogh wheat field, star night, fire nezha, colorful oil, and lighting tiger) applied to five objects(chair, ship, hotdog, lego and mic) and two scenes (man face and train). The results illustrate our model’s capability to handle two main categories of styles: (1) Non-photorealistic Art Styles (e.g.cartoon, drawing), showcasing traditional artistic expressions, and (2) State-based Styles (e.g.fire, oil), which capture physical properties. This figure demonstrates our method’s versatility and semantic-aware ability in stylizing 3D models while preserving style fidelity and geometric consistency across diverse artistic and physical characteristics. For Example, semantic separation of the legs of the chair from the seat cushion, detail texture of chair, texture of the fire on the hot dog, and metallic sheen on the mic are all effectively preserved.

through high-level semantic alignment, leveraging scorebased stable diffusion to extract and transfer domaininvariant style features. SOS addresses low-level feature alignment via multi-scale optimization, preserving stylistic textures through scale-aware importance sampling and geometric consistency constraints. CSD facilitates mid-level style-content harmonization using contrastive learning to disentangle and recompose style attributes while maintaining content integrity. At last, 3DG-QA enhances global aesthetic quality through metric-guided refinement, employing perceptual quality evaluation to optimize both local textural coherence and global visual appeal.

We integrate the whole optimization goal as:

###### Lfinal = Lstyle + LSOS + LCSD + L3DG-QA (16)

In summary, this multi-faceted approach ensures semantic-aware style, fine-grained style, style fidelity and global aesthetics quality.

As demonstrated in Sec. 4.3, the combined losses enable simultaneous preservation of geometric integrity and artistic expression while suppressing common artifacts like overstylization and texture flickering.

#### 4. Experiment

###### 4.1. Comparison Studies

Qualitative Result. we show objects and scene stylization comparisons in Fig.4 and Fig.5 respectively. For objects, we applied vangogh, fire nezha, and sketch styles to chair, hotdog and mic. For scene stylization, we select truck and train from tandt db dataset using landscope and lighting tiger styles. We evaluate our method against others, including SGSST [10], StyleGaussian [25] and ARF [53]. The horizontal axis lists competing methods and the vertical axis denotes datasets.

Different from traditional methods based only on VGG networks like SGSST [10], StyleGaussian [25] and ARF [53], which focus on simple style transfer, our approach prioritizes vivid, expressive and semantic-aware stylization. They relies on VGG networks [34] with empirical-based style decoupling, which limiting style extraction with customized references, our diffusion-based and multi-expert method, pre-trained on large-scale style image-text data, captures style features with greater fidelity. Moreover, training on image-text data enhances semantic understanding, allowing content filtering in CLIP space for precise style extraction.

Unlike ARF [53], which depends on carefully pre-

Input Style Ours

ARF

StyleGaussian

SGSST

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

|[Figure 398]|
|---|

|[Figure 399]|
|---|

|[Figure 400]|
|---|

|[Figure 401]|
|---|

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

|[Figure 432]|
|---|

|[Figure 433]|
|---|

|[Figure 434]|
|---|

|[Figure 435]|
|---|

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

|[Figure 458]|
|---|

|[Figure 459]|
|---|

|[Figure 460]|
|---|

|[Figure 461]|
|---|

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

- Figure 4. Qualitative Comparisons on Object Level Stylization. We compare our method against other SOTA on nerf synthetic dataset (selected chair, hotdog, and mic) using vangogh wheat field, fire nezha, and sketch styles. The horizontal axis represents the compared methods, and the vertical axis displays different data. Our method effectively retains semantic and details of original model and style feature of reference image, such as semantic separation of the legs of the chair from the seat cushion, texture of the fire on the hot dog, and metallic sheen on the mic. Compared to others, our method exhibits stronger semantic understanding, clearly distinguishing elements like the cushions, backrest and legs on the chair.

stylized views for effective color matching and risks texture drift if the initial view is misaligned, our method only requires a single arbitrary style reference image. While we incorporate pre-stylized multi-views, they serve solely for pixel-level style guidance in our outpainting process rather than relying on single-view matching, establishing a distinct way from that of ARF.

Quantitative Evaluation We evaluate our method with three standard image quality metrics: Peak Signal-toNoise Ratio (PSNR), Structural Similarity Index Measure (SSIM) [43], and Learned Perceptual Image Patch Similarity (LPIPS) [54]. PSNR quantifies pixel-level accuracy, indicating how closely the stylized image matches the original. SSIM measures structural similarity, capturing perceptual features like textures and edges. LPIPS assesses perceptual quality based on deep network features, emphasizing visual similarity as perceived by humans.

As shown in Tab. 1, our method achieves significantly higher SSIM and PSNR scores, demonstrating enhanced structural and perceptual fidelity compared to SGSST, StyleGaussian and ARF. Our higher PSNR and SSIM score indicates better fidelity in color and texture reproduction while preserving structural details. Furthermore, the

LPIPS score, measuring perceptual similarity, supports our method’s superior style consistency and stylization quality across multiple viewpoints.

###### 4.2. Visualizations of Various Object-Style Pairs

As shown in Fig. 3, we applied six styles to showcase our experimental results on both object and scene datasets (NeRF synthetic dataset [27] and tandt db [19]). The style references fall into two main categories: nonphotorealistic art styles (e.g.vangogh, cartoon, sketch, hand-drawing, watercolor, painting) and state-based styles (e.g.fire, water, clouds, hair). These categories highlight our method’s ability to handle traditional art styles and capture realistic physical characteristics in 3D. To highlight our method’s advantage in preserving detail textures and shadows, we zoom in on details like the legs and detail texture of the chair, texture of the fire on the hot dog, and metallic sheen on the mic. Experimental results indicate that Gaussian Splatting effectively enhances non-photorealistic and state-based style representations, showing strong adaptability in diverse stylized scenarios. Additional results are provided in the Supplementary.

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

Style Input

ARF StyleGaussian SGSST Ours

|[Figure 474]<br><br>|[Figure 475]|
|---|
|
|---|

|[Figure 476]<br><br>[Figure 477]<br><br>|[Figure 478]|
|---|
|
|---|

|[Figure 479]<br><br>[Figure 480]<br><br>|[Figure 481]|
|---|
|
|---|

|[Figure 482]<br><br>[Figure 483]|
|---|

|[Figure 484]<br><br>[Figure 485]<br><br>|[Figure 486]|
|---|
|
|---|

[Figure 487]

|[Figure 488]|
|---|

|[Figure 489]|
|---|

|[Figure 490]|
|---|

|[Figure 491]|
|---|

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

|[Figure 498]<br><br>|[Figure 499]|
|---|
|
|---|

|[Figure 500]<br><br>[Figure 501]<br><br>|[Figure 502]|
|---|
|
|---|

|[Figure 503]<br><br>[Figure 504]<br><br>|[Figure 505]|
|---|
|
|---|

|[Figure 506]<br><br>[Figure 507]|
|---|

|[Figure 508]<br><br>[Figure 509]<br><br>|[Figure 510]|
|---|
|
|---|

[Figure 511]

|[Figure 512]|
|---|

|[Figure 513]|
|---|

|[Figure 514]|
|---|

|[Figure 515]|
|---|

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

|[Figure 520]<br><br>|[Figure 521]|
|---|
|
|---|

|[Figure 522]<br><br>[Figure 523]|
|---|

|[Figure 524]<br><br>[Figure 525]<br><br>|[Figure 526]|
|---|
|
|---|

|[Figure 527]<br><br>[Figure 528]<br><br>|[Figure 529]|
|---|
|
|---|

|[Figure 530]<br><br>[Figure 531]<br><br>|[Figure 532]|
|---|
|
|---|

[Figure 533]

|[Figure 534]|
|---|

|[Figure 535]|
|---|

|[Figure 536]|
|---|

|[Figure 537]|
|---|

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

|[Figure 544]<br><br>|[Figure 545]|
|---|
|
|---|

|[Figure 546]<br><br>[Figure 547]<br><br>|[Figure 548]|
|---|
|
|---|

|[Figure 549]<br><br>[Figure 550]<br><br>|[Figure 551]|
|---|
|
|---|

|[Figure 552]<br><br>[Figure 553]|
|---|

|[Figure 554]<br><br>[Figure 555]<br><br>|[Figure 556]|
|---|
|
|---|

[Figure 557]

|[Figure 558]|
|---|

|[Figure 559]|
|---|

|[Figure 560]|
|---|

|[Figure 561]|
|---|

- Figure 5. Qualitative Comparisons on Scene Level Stylization. We compare our method against other SOTA on tandt db dataset (selected truck and train) using landscope and lighting tiger styles. The horizontal axis represents the compared methods, and the vertical axis displays different data. Our method effectively retains semantic and details of original model and style feature of reference image, such as the truck wheel and train fence (as shown in Zoom-in). Compared to others, our method exhibits stronger semantic understanding, clearly distinguishing elements like the fence, tire and rail.

- Table 1. Quantitative comparison with competing methods. CLIP(S) means CLIP-based Style Similarity and CLIP(C) means CLIP-based Content Similarity.

Method PSNR↑ SSIM↑ LPIPS↓ CLIP(S)↑ CLIP(C)↑

ARF 17.537 0.802 0.188 0.269 0.701 Ref-NPR 14.047 0.655 0.331 0.531 0.664 SGSST 11.963 0.678 0.306 0.354 0.595 StyleGaussian 7.279 0.129 0.558 0.320 0.610 Ours 18.015 0.830 0.174 0.583 0.691

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

Local mode

Global+local mode

|[Figure 569]|
|---|

[Figure 570]

|[Figure 571]|
|---|

[Figure 572]

w style outpainting

w/o style Input outpainting

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

|[Figure 579]|
|---|

|[Figure 580]|
|---|

Style

(a) (b)

|[Figure 581]|
|---|

[Figure 582]

|[Figure 583]|
|---|

[Figure 584]

|[Figure 585]|
|---|

[Figure 586]

|[Figure 587]|
|---|

[Figure 588]

|[Figure 589]|
|---|

[Figure 590]

|[Figure 591]|
|---|

[Figure 592]

[Figure 593]

Input

|[Figure 594]|
|---|

|[Figure 595]|
|---|

|[Figure 596]|
|---|

|[Figure 597]|
|---|

|[Figure 598]|
|---|

|[Figure 599]|
|---|

[Figure 600]

Style

Input

- Figure 6. Ablation study on style outpainting guidance mode. (a) Baseline without style outpainting exhibits limited stylization scope and view-dependent artifacts (red boxes). (b) Local Guidance enables single-view enhancement but causes multi-view inconsistencies. Global-Local Fusion achieves cross-view style propagation through adaptive attention weighting, improving style consistency while preserving view-specific details.

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

low scale dynamic scale high scale

Style Input

Figure 7. Ablation study on dynamic noise scheduling. Low Scale (7.5) produces incomplete stylization with missing texture details. High Scale (50) introduces oversaturation artifacts and structural distortions. Dynamic Scale (7.5-30) adaptively balances detail preservation and style intensity.

###### 4.3. Ablation Study

We conducted ablation studies to assess the impact of various components and parameters in our method, focusing on style outpainting mode, DDSD and multi-expert module.

Ablation on Style Outpainting. As shown in Fig. 6. We present an ablation study on the impact of Style Outpaint-

- Table 2. Quantitative comparison with V1 (DSSD version), V2 (DSSD+SOS version), V3 (DSSD+SOS+CSD version) and V4 (DSSD+SOS+CSD+IQA version). V4 is the final Multi-Expert version.

###### Method PSNR↑ SSIM↑ LPIPS↓ CLIP(S)↑ CLIP(C)↑

Ours (V1) 17.270 0.776 0.181 0.280 0.811 Ours (V2) 17.650 0.800 0.178 0.285 0.805 Ours (V3) 17.900 0.820 0.175 0.325 0.695 Ours (V4) 18.015 0.830 0.174 0.331 0.701

[Figure 611]

[Figure 612]

[Figure 613]

gradient optimization directions - DSSD’s tendency toward over-smoothing is counterbalanced by SOS’s capacity for detail enhancement, while SOS’s potential over-emphasis on low-level features is constrained by DSSD’s semantic guidance.

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

|[Figure 622]|
|---|

|[Figure 623]|
|---|

|[Figure 624]|
|---|

[Figure 625]

|[Figure 626]|
|---|

|[Figure 627]|
|---|

|[Figure 628]|
|---|

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

|[Figure 640]|
|---|

|[Figure 641]|
|---|

|[Figure 642]|
|---|

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

Subsequent integration of CSD and 3DG-QA implements knowledge-driven perceptual assessment through CLIP-space cosine similarity metrics. The CSD module specializes in style authenticity evaluation through learned artistic aesthetics criteria, while 3DG-QA provides qualityfocused guidance via antonymic text prompts. Quantitative analysis shows this combined approach achieves remarkable improvement in human perceptual quality scores compared to baseline configurations (see Table 2).

DSSD DSSD + SOS DSSD + SOS + CSD&IQA

Style

Figure 8. Ablation study loss design. (a) DSSD-only initialization yields semantically coherent but texture-deficient results with color shifts (see missing curvilinear patterns in Van Gogh stylization). (b) DSSD+SOS achieves texture-geometry equilibrium through gradient mutual regularization, recovering fine details while suppressing over-smoothing. (c) Full Model enhances perceptual quality via knowledge-driven style assessment, achieving remarkable improvement over baseline (Table 2).

#### 5. Conclusion

ing. Without it, the degree of stylization is visibly limited, whereas applying Style Outpainting allows effective style propagation across views. We compares different guidance schemes: local mode & global-local mode. Local mode shows inconsistencies, resulting in artifacts and missing details in certain views. In contrast, global-local mode enhances stylization intensity and detail refinement, achieving more coherent stylization across views.

Ablation on DSSD. As shown in Fig. 7. We conducted an ablation study on the effectiveness of dynamic guidance scale in DSSD. Comparing results at a low scale of 7.5, a high scale of 50, and a dynamic scale ranging from 7.5 to 30, we observed that the dynamic scale approach consistently outperforms static setting.

Ablation on Multi-Expert. As shown in Fig. 8, we analyze the impact of SOS, CSD and 3DG-QA on stylization quality. our analysis reveals that initial stylization using DSSD alone produces semantically coherent results but suffers from two critical limitations: 1) Insufficient low-level texture details (e.g., missing curvilinear patterns in Van Gogh-inspired wheat field renderings), and 2) Systematic color deviation artifacts. The introduction of SOS loss establishes a dual-optimization framework where DSSD and SOS operate concurrently within single-view projections. This configuration enables mutual regularization of their

We redefine the 3D Gaussian Splatting (3D GS) stylization task through comprehensive analysis and propose the StyleMe3D framework, establishing a novel paradigm for artistic 3D scene stylization. StyleMe3D enables artistic 3D Gaussian Splatting stylization via Diffusion-guided score distillation (DSSD), CLIP-based dual-stream (style and content) alignment (CSD&3DG-QA), and multi-scale optimization (SOS). Experiments show superior detail retention, style consistency across various objects and scenes.

#### References

- [1] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR,

2022. 2

- [2] Andrew Brock. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096, 2018. 2
- [3] Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, and Hongsheng Li. Generalizable neural performer: Learning robust radiance fields for human novel view synthesis. arXiv preprint arXiv:2204.11798, 2022. 2
- [4] Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. Dna-

- rendering: A diverse neural actor repository for highfidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19982–19993, 2023. 2
- [5] Pei-Ze Chiang, Meng-Shiun Tsai, Hung-Yu Tseng, Wei-Sheng Lai, and Wei-Chen Chiu. Stylizing 3d scene via implicit representation and hypernetwork. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1475–1484,

2022. 3

- [6] Jiwoo Chung, Sangeek Hyun, and Jae-Pil Heo. Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8795–8805, 2024. 2
- [7] Valentin De Bortoli, Agn`es Desolneux, Alain Durmus, Bruno Galerne, and Arthur Leclaire. Maximum entropy methods for texture synthesis: theory and practice. SIAM Journal on Mathematics of Data Science, 3(1):52–82, 2021. 2
- [8] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663,

2023. 3

- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 5
- [10] Bruno Galerne et al. Sgsst: Scaling gaussian splatting styletransfer. arXiv preprint arXiv:2412.03371, 2024. 2, 3, 5, 6
- [11] Junyao Gao, Yanchen Liu, Yanan Sun, Yinhao Tang, Yanhong Zeng, Kai Chen, and Cairong Zhao. Styleshot: A snapshot on any style. arXiv preprint arXiv:2407.01414, 2024. 4, 13
- [12] Xiang Gao, Zhengbo Xu, Junhan Zhao, and Jiaying Liu. Frequency-controlled diffusion model for versatile text-guided image-to-image translation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 1824–1832, 2024. 2
- [13] Leon A Gatys, Alexander S Ecker, Matthias Bethge, Aaron Hertzmann, and Eli Shechtman. Controlling perceptual factors in neural style transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3985–3993, 2017. 5

- [14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [15] Feihong He, Gang Li, Mengyuan Zhang, Leilei Yan, Lingyu Si, Fanzhang Li, and Li Shen. Freestyle: Free lunch for text-guided style transfer using diffusion models. arXiv preprint arXiv:2401.15636, 2024. 2
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598,

2022. 4

- [17] Hsin-Ping Huang, Hung-Yu Tseng, Saurabh Saini, Maneesh Singh, and Ming-Hsuan Yang. Learning to stylize novel views. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13869–13878, 2021. 3
- [18] Tero Karras, Samuli Laine, and Timo Aila. A stylebased generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 2
- [19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 2, 7, 13
- [20] Hubert Kompanowski and Binh-Son Hua. Dream-instyle: Text-to-3d generation using stylized score distillation. arXiv preprint arXiv:2406.18581, 2024. 3, 5
- [21] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024. 2
- [22] Pengzhi Li, Chengshuai Tang, Qinxuan Huang, and Zhiheng Li. Art3d: 3d gaussian splatting for textguided artistic scenes generation. arXiv preprint arXiv:2405.10508, 2024. 2, 13
- [23] Wen Li, Muyuan Fang, Cheng Zou, Biao Gong, Ruobing Zheng, Meng Wang, Jingdong Chen, and Ming Yang. Styletokenizer: Defining image style by a single instance for controlling diffusion models. arXiv preprint arXiv:2409.02543, 2024. 2
- [24] Kunhao Liu, Fangneng Zhan, Yiwen Chen, Jiahui Zhang, Yingchen Yu, Abdulmotaleb El Saddik, Shijian Lu, and Eric P Xing. Stylerf: Zero-shot 3d style transfer of neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8338–8348, 2023. 3
- [25] Kunhao Liu, Fangneng Zhan, Muyu Xu, Christian Theobalt, Ling Shao, and Shijian Lu. Stylegaussian:

- Instant 3d style transfer with gaussian splatting. arXiv preprint arXiv:2403.07807, 2024. 2, 3, 6
- [26] Yiqun Mei, Jiacong Xu, and Vishal M Patel. Reference-based controllable scene stylization with gaussian splatting. arXiv preprint arXiv:2407.07220,

2024. 3

- [27] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99–106, 2021. 2, 7, 13
- [28] Fangzhou Mu, Jian Wang, Yicheng Wu, and Yin Li. 3d photo stylization: Learning to generate stylized novel views from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16273–16282, 2022. 3
- [29] Thu Nguyen-Phuoc, Feng Liu, and Lei Xiao. Snerf: stylized neural implicit representations for 3d scenes. arXiv preprint arXiv:2207.02363, 2022. 3
- [30] Dongwei Pan, Long Zhuo, Jingtan Piao, Huiwen Luo, Wei Cheng, Yuxin Wang, Siming Fan, Shengqi Liu, Lei Yang, Bo Dai, et al. Renderme-360: a large digital asset library and benchmarks towards high-fidelity head avatars. Advances in Neural Information Processing Systems, 36:7993–8005, 2023. 2
- [31] Kien T Pham, Jingye Chen, and Qifeng Chen. Tale: Training-free cross-domain image composition via adaptive latent manipulation and energy-guided optimization. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 3160–3169,

2024. 2

- [32] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 13
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4
- [34] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 2, 5, 6
- [35] Gowthami Somepalli, Ayan Bansal, Micah Goldblum, Jonas Geiping, Tom Goldstein, et al. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024. 2, 5
- [36] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653, 2023. 2
- [37] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-

- view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2025. 2
- [38] Can Wang, Ruixiang Jiang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Nerf-art: Text-driven neural radiance fields stylization. IEEE Transactions on Visualization and Computer Graphics, 2023. 3
- [39] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12619–12629, 2023. 2
- [40] Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards stylepreserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 2, 4
- [41] Jianyi Wang, Kelvin C.K. Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. arXiv preprint arXiv:2207.12396, 2022. 5
- [42] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023. 2
- [43] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 5, 7
- [44] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024. 2
- [45] Zhenwei Wang, Tengfei Wang, Zexin He, Gerhard Hancke, Ziwei Liu, and Rynson WH Lau. Phidias: A generative model for creating 3d content from text, image, and 3d conditions with reference-augmented diffusion. arXiv preprint arXiv:2409.11406, 2024. 2, 3, 13
- [46] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, et al. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 803–814, 2023. 3
- [47] Zongze Wu, Yotam Nitzan, Eli Shechtman, and Dani Lischinski. Stylealign: Analysis and applications of aligned stylegan models. arXiv preprint arXiv:2110.11323, 2021. 2

- [48] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. Svgdreamer: Text guided svg generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4546–4555,

2024. 2

- [49] Zizheng Yan, Jiapeng Zhou, Fanpeng Meng, Yushuang Wu, Lingteng Qiu, Zisheng Ye, Shuguang Cui, Guanying Chen, and Xiaoguang Han. Dreamdissector: Learning disentangled text-to-3d generation from 2d diffusion priors. arXiv preprint arXiv:2407.16260, 2024. 2, 13
- [50] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023. 2, 13
- [51] Yu Zeng, Vishal M Patel, Haochen Wang, Xun Huang, Ting-Chun Wang, Ming-Yu Liu, and Yogesh Balaji. Jedi: Joint-image diffusion models for finetuning-free personalized text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6786–6795, 2024. 2
- [52] Dingxi Zhang, Yu-Jie Yuan, Zhuoxun Chen, Fang-Lue Zhang, Zhenliang He, Shiguang Shan, and Lin Gao. Stylizedgs: Controllable stylization for 3d gaussian splatting. arXiv preprint arXiv:2404.05220, 2024. 2, 3
- [53] Kai Zhang, Nick Kolkin, Sai Bi, Fujun Luan, Zexiang Xu, Eli Shechtman, and Noah Snavely. Arf: Artistic radiance fields. In European Conference on Computer Vision, pages 717–733. Springer, 2022. 3, 6
- [54] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5, 7
- [55] Yuechen Zhang, Zexin He, Jinbo Xing, Xufeng Yao, and Jiaya Jia. Ref-npr: Reference-based nonphotorealistic radiance fields for controllable scene stylization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4242–4251, 2023. 2, 3
- [56] Junsheng Zhou, Weiqi Zhang, and Yu-Shen Liu. Diffgs: Functional gaussian splatting diffusion. arXiv preprint arXiv:2410.19657, 2024. 2, 13

## Appendix

#### A. Preliminary

###### A.1. Style-aware Image Customization

In recent advancements in style transfer, StyleShot [11] and IP-Adapter [50] represent two prominent techniques, each employing distinct methods to transfer style from a reference image to a target image. StyleShot emphasizes the extraction of detailed style features using a style-aware encoder, which leverages multi-scale patch partitioning to capture both low-level and high-level style cues. Specifically, StyleShot divides the reference image into nonoverlapping patches of three sizes, corresponding to different scales. For each patch scale, there is a dedicated ResBlock at different depths.

The following are the key formulas for style injection in StyleShot:

Attention(Q,Ks,Vs) = softmax

QKsT

d · Vs (S1)

√

where Q is the query projected from the latent embeddings f, and Ks and Vs are the keys and values, respectively, that the style embeddings fs are projected onto through independent mapping functions WK

. The attention outputs of the text embeddings ft and style embeddings fs are then combined into new latent embeddings f′, which are fed into subsequent blocks of Stable Diffusion:

and WV

s

s

f′ = Attention(Q,Kt,Vt) + λAttention(Q,Ks,Vs)

(S2) where λ represents the weight balancing the two components.

###### A.2. Score Distillation Sampling for 3D Generation

Text-guided 3D generation has gained significant attention due to advancements in methods such as Score Distillation Sampling (SDS) [32], which facilitates the optimization of 3D representations using pre-trained diffusion models. SDS optimizes the parameters θ of a 3D model g(θ) by distilling gradients from a diffusion model ϕ, ensuring that 2D projections generated from g(θ) align with a target text prompt. The gradient of the SDS loss is defined as:

∂x ∂θ

∇θLSDS(ϕ,x = g(θ)) = Et,ϵ ω(t)(ˆϵϕ(zt;y,t) − ϵ)

,

(S3) where ϵˆϕ(zt;y,t) represents the predicted noise residual from the pre-trained diffusion model, ϵ is the actual noise used in the forward process, zt is the latent variable at timestep t, and ω(t) is a timestep weighting function.

These have been extended to artistic scene generation [22] and combined input conditions, including text and images [45, 49]. Recent advances leveraging latent diffusion models have improved the scope and expressiveness of textto-3D synthesis [49, 56], supporting more nuanced and creative 3D outputs.

###### A.3. 3D Gaussian Splatting

3D Gaussian Splatting (3D GS) [19] represents a 3D scene using a collection of spatial Gaussians. Each Gaussian gi is defined by a mean position µi ∈ R3 and a covariance matrix Σi ∈ R3×3, which determines its shape and orientation. The Gaussian’s influence on a point x is given by:

- 1

- 2(x−µi)⊤Σ−i 1(x−µi) (S4)

G(x) = e−

where Σi = RSS⊤R⊤ is decomposed into a rotation R and scaling S matrices. Each Gaussian has an opacity αi and a view-dependent color ci.

During rendering, Gaussians are projected to 2D and blended using alpha compositing. The final pixel color C is calculated as:

n

C =

i=1

- i−1
- j=1

(1 − αj) (S5)

ciαi

Here, αi is the effective opacity of the i-th Gaussian in sorted depth order. Gaussian Splatting enables real-time, differentiable rendering and can reconstruct scenes with multi-view supervision.

Compared to NeRF [27], 3D Gaussian Splatting is significantly more efficient in both time and memory usage. By representing scenes with Gaussian primitives rather than dense neural networks, it allows for faster rendering and lower computational costs, making it more suitable for realtime applications.

#### B. Implementation Details

Computational Environment: All experiments were conducted on a single NVIDIA L40S GPU with 46GB of VRAM.

Dataset: NeRF synthetic dataset [27] and tandt db [19], was used for all experiments.

###### B.1. Details of Dynamic Style Score Distillation (DSSD)

1. Backbone Models: For the style-aware diffusion model, we adopt StyleShot, which builds on IP-Adapter and incorporates a style-aware encoder to enhance style representation, enabling robust style transfer through score distillation guidance.

- 2. Fine Timestep Sampling: We employ a fine-grained timestep sampling strategy with a timestep constant T =

1000. Minimum and maximum timesteps were set as Tmin = 0.02 · T and Tmax = 0.75 · T, respectively. The noise intensity was dynamically reduced to high, medium, and low levels to stabilize the updates during training.

- 3. Dynamic Guidance Coefficients: The dynamic guidance coefficient ∆λ was tuned to adapt to varying scales of the dataset and style variations. For the NeRF Syn-

thetic dataset, we selected λmax = 20 and confined ∆λ within [7.5,20].

- 4. Guidance Modes and Outpainting Strategy: A total of 2800 steps were employed, segmented into specific guidance modes:

- • Main RGB Loss (Local Mode): Steps 100 to 600.
- • Adaptive Iteration (Global Mode): Steps 1 to 1000, alternating between global RGB and global SDS losses.
- • Fixed or Free Global Modes: Steps 1000 to 1900, alternating between global-fix and global-free modes.
- • Local Mode: Steps 1900 to 2800. This hybrid strategy begins with global optimization before transitioning to local refinement, requiring 1800 iterations for SDS loss.

- 5. Iteration Time and Cost Analysis:

- • Average Time Per Iteration: Single-view RGB loss averaged 0.1 seconds, while SDS loss averaged 2.5 seconds.
- • Total Iteration Count and Convergence: Using RGB loss for the initial 1000 steps and SDS loss for the subsequent 2000 steps, convergence was achieved in approximately 2600 seconds. For enhanced local convergence, an additional 500 to 1000 SDS iterations were applied.

###### B.2. Details of Simultaneously Optimized Scale (SOS)

###### 1. VGG Feature Extraction

- • Style layers: [’r11’,’r21’,’r31’,’r41’,’r51’]
- • Content layer: [’r42’]
- • Gram matrix weights: [1e3/64², 1e3/128², 1e3/256², 1e3/512², 1e3/512²]

###### 2. Two-Phase Optimization

- • Pretraining phase:

- – Trigger: optimize iteration=10000 and current iter < 10000

- – Fixed scale: optimize size=0.5 (uses minimum resize images if unspecified)

- – Downsampling: Bilinear interpolation mode="bilinear"

- • Full multi-scale phase: Activates all

[Figure 649]

Figure S1. Optimization Pathways for Pre-training vs. Posttraining. The plot illustrates the optimization pathways for pretraining (blue solid line) and post-training (orange dashed line), highlighting the optimization gap (gray shaded area) between 3D reconstruction and stylization. The pre-training pathway shows smooth, steady convergence, while the post-training pathway oscillates due to inherent uncertainty in stylization. The optimization gap represents misalignment between the stages, emphasizing the need for alignment techniques, such as style-aware priors and dynamic guidance, to achieve stable and consistent 3D stylization.

resize images scales

###### B.3. Details of Contrastive Style Descriptor (CSD)

• Deployed CSD ViT-L style encoder pretrained on LAION-Styles dataset.

###### B.4. Details of 3D Gaussian Quality Assessment (3DG-QA)

- • Integrated CLIP-ViT-B with antonymic prompts: ”Good, Sharp, Colorful” vs ”Bad, Blurry, Dull”, prompts=(”quality”, ”sharpness”, ”colorfullness”)
- • loss = 1 - (0.4*scores[’quality’]

+ 0.4*scores[’sharpness’] + 0.2*scores[’colorfullness’]).mean()

, where wq = 0.4, ws = 0.4, wc = 0.2 denote quality, sharpness, and colorfulness weights respectively.

#### C. Additional Method Analysis

The challenges of directly transferring 3D generation techniques to 3D stylization stem from the optimization gap between pre-training and post-training stages. This section provides a theoretical and visual analysis of this gap.

###### C.1. Misalignment in Optimization Pathways

• Pre-training Objective: The goal of 3D reconstruction during pre-training is to capture geometric and photometric properties accurately. This optimization process is typically smooth and guided by explicit ground truth data.

- • Post-training Objective: In the post-training phase, the focus shifts to aesthetic alignment using style-aware guidance, which lacks explicit supervision and introduces higher uncertainty.
- • Disjoint Loss Landscapes: The loss landscapes for pre-training and post-training differ significantly. Pretraining minimizes reconstruction errors, while stylization involves abstract priors from style information, leading to potential misalignment.

The optimization pathways during pre-training and posttraining can be represented as two distinct loss functions:

Lpre = Lrecon(Gpre(x),xgt), (S6)

where Lrecon minimizes geometric and photometric errors between the predicted Gpre(x) and ground truth xgt, and:

Lpost = Lstyle(Gpost(x),sref), (S7)

where Lstyle aligns the generated results Gpost(x) with a style reference sref using abstract priors.

The optimization gap can then be formulated as:

∆L = |Lpre − Lpost|, (S8)

where ∆L quantifies the divergence between the loss landscapes, reflecting the mismatch in optimization objectives.

###### C.2. High Uncertainty in Style Information

- • Multi-modal Style Representations: Styles are inherently diverse and lack well-defined ground truth, making the optimization process less predictable.
- • Temporal Instability: Stylization optimization pathways often exhibit oscillations due to conflicts between style priors and geometric constraints.

The uncertainty in style optimization can be modeled as the variance in style priors:

σstyle2 = Var(sref), (S9)

where sref represents multi-modal style representations. Temporal oscillations in optimization can be expressed as:

δt = |∇Lpost,t+1 − ∇Lpost,t|, (S10)

where δt measures the instability between consecutive timesteps t and t + 1.

###### C.3. Visualization Analysis

The graph (Figure S1) visualizes the optimization gap between pre-training and post-training:

• Pre-training pathway (blue solid line) shows smooth convergence, reflecting steady optimization for geometric fidelity.

- • Post-training pathway (orange dashed line) exhibits oscillations, driven by the abstract and subjective nature of style priors.
- • Optimization gap (gray shaded area) represents the divergence between the two pathways, indicating the challenges of transitioning between the stages.

To bridge the optimization gap, alignment strategies must minimize:

∆L + λconsLconsistency, (S11)

min

G

where Lconsistency enforces multi-view consistency, and λcons is a weighting factor to balance consistency with style fidelity.

###### C.4. Key Observations and Insights

- 1. Mismatch in Optimization: The smooth convergence of pre-training contrasts with the oscillatory adjustments in post-training, reflecting the differences in objectives—geometric accuracy vs. subjective style transfer.

The loss landscapes Lpre and Lpost differ fundamentally in their curvature:

κpre ≪ κpost, (S12)

where κ represents the curvature, indicating smoother optimization for pre-training compared to post-training.

- 2. Impact of the Gap: The optimization gap introduces challenges such as:

- • Optimization Instability: Misaligned pathways can lead to instability during post-training.
- • Inconsistent Stylization: Divergent trajectories may result in geometric distortions or incomplete stylization.

Misaligned pathways can exacerbate:

- • Instability: ∆L leads to higher gradients: ∇Lpost ≫ ∇Lpre. (S13)
- • Inconsistency: Variance in style priors σstyle2 introduces inconsistencies in multi-view stylization.

- 3. Bridging the Gap: Effective strategies such as styleaware diffusion priors, dynamic style score distillation, and progressive style outpainting are critical to aligning pathways and ensuring robust stylization. Introducing regularization terms:

Lalign = λpriorLstyle + λgeoLrecon, (S14)

where λprior and λgeo balance style fidelity and geometric preservation, helps align the pathways.

###### C.5. Conclusion

This analysis highlights the inherent challenges in aligning pre-training and post-training optimization pathways. The

visualization emphasizes the need for dedicated techniques to bridge the gap, ensuring high-fidelity and consistent stylization while maintaining geometric coherence.

#### D. More Visual Result

As shown in Figure S2, S3, and S4, we demonstrate our method’s performance across nine distinct styles (sky painting, cartoon, watercolor, fire, cloud, Wukong, drawing, color oil, and sketch) on three datasets (chair, hotdog, and mic).

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

###### Figure S2. More visual results. Demonstration of our method’s performance across nine distinct styles (sky painting, cartoon, watercolour, fire, cloud, Wukong, drawing, color oil, and sketch) applied to chair.

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

###### Figure S3. More visual results. Demonstration of our method’s performance across nine distinct styles (sky painting, cartoon, watercolour, fire, cloud, Wukong, drawing, color oil, and sketch) applied to hotdog.

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

###### Figure S4. More visual results. Demonstration of our method’s performance across nine distinct styles (sky painting, cartoon, watercolour, fire, cloud, Wukong, drawing, color oil, and sketch) applied to mic.

