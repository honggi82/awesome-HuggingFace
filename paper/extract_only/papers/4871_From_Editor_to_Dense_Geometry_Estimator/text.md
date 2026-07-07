# arXiv:2509.04338v2[cs.CV]24Mar2026

## FE2E: From Editor to Dense Geometry Estimator

Jiyuan Wang1,2 Chunyu Lin1,* Lei Sun2,† Rongying Liu1 Lang Nie3 Mingxing Li2 Kang Liao4 Xiangxiang Chu2 1BJTU 2Alibaba Group 3CQUPT 4NTU

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Input Image

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

FE2E Normal

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

FE2E Depth

Zero-shot Depth Es ma on

Zero-shot Normal Es ma on

[Figure 16]

[Figure 17]

Figure 1. We present FE2E, a DiT-based foundation model for monocular dense geometry prediction. Trained with limited supervision, FE2E achieves promising performance improvements in zero-shot depth and normal estimation. Bar length indicates the average ranking across all metrics from multiple datasets, where lower values are better. ★ represents the amount of training data used.

### Abstract

Pre-trained text-to-image (T2I) generative priors have shown success in depth and normal prediction. However, dense prediction is inherently an image-to-image task, suggesting that image editing models, rather than T2I generative models, may be a more suitable foundation for finetuning. Motivated by this, we conduct a systematic analysis of the fine-tuning behaviors of both editors and generators for dense geometry estimation. Our findings show that editing models possess inherent structural priors, which enable them to converge more stably by “refining” their innate features, and ultimately achieve higher performance than their generative counterparts. Based on these findings, we introduce FE2E, a framework that pioneeringly adapts an advanced editing model based on Diffusion Transformer (DiT)

*Corresponding author. †Project leader.

architecture for dense geometry prediction. Specifically, to tailor the editor for this deterministic task, we reformulate the editor’s original flow matching loss into the “consistent velocity” training objective. And we use logarithmic quantization to resolve the precision conflict between the editor’s native BFloat16 format and the high precision demand of our tasks. Additionally, we repurpose the editor’s discarded region for a cost-free joint estimation of depth and normals, which improves the inference efficiency. Without scaling up the training data, FE2E achieves impressive performance improvements in zero-shot monocular depth and normal estimation across multiple datasets. Notably, it achieves over 35% performance gains on the ETH3D dataset and outperforms the DepthAnything series, which is trained on 100× data.

### 1. Introduction

Depth and normal estimation are typical dense geometry prediction tasks and crucial for a wide range of applications such as augmented reality [43], and 3D reconstruction [34]. Estimating pixel-level geometric attributes from a single image is an ill-posed problem and can only be solved with the help of prior knowledge, such as typical object shapes and sizes, occlusion patterns, etc. Based on this observation, recent works ingeniously leverage the priors from pre-trained text-to-image (T2I) generators, typically Stable Diffusion [56], for zero-shot depth prediction [29], yielding impressive results with limited training data.

However, the pre-trained generative models are initially designed for T2I generation, limiting their ability to capture the geometric cues. In contrast, image editing models have recently risen to be a universal framework to solve more diversified image-to-image (I2I) tasks, such as semantic segmentation and depth estimation [73]. We argue that these editing models not only align with the dense estimation paradigm but also possess a deep understanding of input images while maintaining the generative advantages, and offer a more suitable foundation for dense geometry prediction.

Motivated by this intuition, we systematically analyze the fine-tuning process of the image editing models versus their generative counterparts. Our analysis reveals that the features of editing models are inherently aligned with geometric structures, and the fine-tuning process only requires to “refine” and “focus” this perceptual ability for dense estimation tasks. In contrast, although generative models can gradually acquire this capability from scratch, this process leads to substantial feature reshaping and cannot fundamentally bridge this gap (Sec. 3.1). Therefore, in this paper, we explore this editing option and propose From Editor to Estimator (FE2E) (Fig. 2), a diffusion transformer (DiT) model built upon the current SoTA editor Step1X-Edit [38], along with a fine-tuning protocol to adapt it for dense geometry prediction tasks.

However, the direct adaptation of an image editing model for geometric dense prediction is often suboptimal, due to the inherent differences between the two tasks. First, geometry prediction is more deterministic as only one unique ground-truth (GT) exists. Therefore, we analyze the flow matching loss of Step1X-Edit and extend the deterministic prediction [20] to this paradigm, reformulating the original instantaneous velocity training objective to consistent velocity and setting a fixed starting point for stable training. Second, image editing models like Step1X-Edit are typically trained with BF16 precision, which is sufficient for RGB outputs, whereas depth estimation demands much higher numerical precision. This discrepancy does not occur in previously adopted foundation models like Stable Diffusion v1.5/v2, which offer FP32 checkpoints. To address this limitation and improve computational efficiency,

we analyze the GT quantization strategies and adopt a logarithmic quantization to alleviate precision-related artifacts. Third, we take advantage of the parallel output in DiT-based editing models, and reformulate it as a joint estimation with no additional computational cost.

Extensive experiments demonstrate that our model achieves notable zero-shot performance improvements compared to previous state-of-the-art (SoTA) models. Our contributions can be summarized as follows:

- • We systematically analyze the fine-tuning process of image editors and generators, revealing that editing models are more suitable for dense geometry prediction. Based on this, we introduce FE2E, a novel framework that, for the first time, successfully adapts a pre-trained image editing model for this task.
- • We identify and address the challenges that arise from this paradigm shift: 1) Reformulate the training objective to align with the deterministic nature of dense prediction; 2) Adopt a logarithmic quantization to resolve the precision conflict. 3) Design a cost-free joint estimation to improve the efficiency.
- • Based on above enhancements, FE2E achieves impressive performance gains, including 35% AbsRel improvement on the ETH3D dataset. Even when using only 0.2% of the geometric GT data for training, FE2E outperforms the data-driven models like DepthAnything v1/v2.

### 2. Related Work

Image Generative and Editing models In the field of image generation, Stable Diffusion series [56] and FLUX series [32, 39] models have basically become the community standard. Both are trained on massive datasets and demonstrate extremely high generation quality. Meanwhile, the field of image editing is also evolving rapidly. Recent advancements include Step1X-Edit [38], a model fine-tuned from FLUX, demonstrating superior instructionfollowing and image understanding capability; The multimodal Qwen-Image [73] Editor combined with LLM, attempting to expand the editor into a unified computer vision framework; The concurrent work FLUX-Kontext [32] unifies editing and generation with robust character consistency. We conduct a more detailed review [2, 9, 33, 51, 57] in the appendix Sec E.

Dense Geometry Estimation, encompassing tasks like depth and normal estimation[66, 67], is a cornerstone of 3D computer vision. Early research predominantly focused on supervised learning paradigms, where models were trained and evaluated on specific datasets [12, 13]. A significant shift occurred with MiDaS [53], which pioneered crossdataset generalization for dense estimation. This line of work was extended by models like DPT [54] and Omnidata [11], which further improved zero-shot performance. There are also some point-based methods[14, 68, 72]. More

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

DiT 𝑓

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Es matorEditor

[Figure 33]

[Figure 34]

ℰ 𝒟

C

[Figure 35]

[Figure 36]

z ~𝒩(0,1)

𝑡 ∈ 0,1 , 𝑖𝑛𝑠truction

discard

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

𝑡 𝑖𝑛𝑠truction Consist Velocity

Logarithmic Quantization

z = 𝟎

[Figure 46]

[Figure 47]

𝒅

Fixed Origin

retain

[Figure 48]

[Figure 49]

[Figure 50]

| | | | |
|---|---|---|---|
|[Figure 51]| | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 52]| | |
| | | | |

[Figure 53]

❄ ❄

[Figure 54]

y

[Figure 55]

[Figure 56]

[Figure 57]

| | | | |
|---|---|---|---|
| | | | |
|[Figure 58]| | | |
| | | | |

[Figure 59]

[Figure 60]

[Figure 61]

ℰ 𝒟

[Figure 62]

[Figure 63]

C

𝒙

DiT 𝑓

𝑝 𝑝

𝑧

[Figure 64]

y

[Figure 65]

𝒏

| | | | |
|---|---|---|---|
| |[Figure 66]| | |
| | | | |
| | | | |

[Figure 67]

Cost-Free Joint Estimation

[Figure 68]

Only Inference

𝑧

- Figure 2. FE2E Adaptation Pipeline. The grey background shows the original editor’s workflow, while the other details FE2E: ① A pre-trained VAE encodes the logarithmically quantized depth d, input image x, and normals n into latent space. ② The DiT fθ learns

a constant velocity v from a fixed origin zy0 to the target latent zy1, independent of t or instructions. ③ By repurposing the discarded output region, FE2E jointly predicts depth and normals without extra computation. Training loss is computed in the latent space, with final predictions decoded by VAE only at inference.

#### 3.1. Fine-tuning Analysis of Editor and Generator

recently, the field has witnessed the rise of data-driven models such as the Depth Anything series [36, 78, 79] and the Metric3D series [23], which leverage massive datasets to train powerful, general-purpose geometric estimators.

In this paper, we select Step1X-Edit as the editor and FLUX as the generator, owing to their shared DiT architecture and SoTA performance in their respective tasks. To facilitate a fair comparison, we adapt the FLUX model to the same input structure (as in Fig. 2) and use the same training settings for this analysis (details are provided in Appendix Sec B). Through this systematic analysis, we identify three key advantages of editing-based models over generative predecessors for dense estimation tasks.

Generative Models for Dense Estimation. In parallel to the trend of scaling up data, an alternative approach [20] emerged by leveraging the rich priors of pretrained generative models. Works like Marigold [29] and GeoWizard [15] showed that fine-tuning diffusion models on limited data could yield remarkable performance, effectively harnessing the models’ learned world knowledge. This paradigm was further refined by GenPercept [74], StableNormal [80], Diffusion-E2E-FT [41], DepthFM [19], and Jasmine [69]. These studies identified and addressed the limitations of standard diffusion formulations, developing the end-to-end denoising architecture to boost performance. DICEPTION and pixel-perfect-depth [75, 95] further extend this paradigm to DiT architecture. In this paper, we also build FE2E with limited data, posit that the I2I editing models are inherently better than the T2I models (Stable Diffusion [56]) for dense estimation.

First, the editing model possesses a superior inductive bias for image-to-image dense estimation tasks, providing a much stronger starting point for finetuning. This is evident in Fig. 3 (a1 v.s. a2), in the early stage and blocks, the editor’s internal features already align with the input image’s geometric structures, while the generative ones are abstract and unstructured. The loss difference in Fig. 4 ★ shows the same conclusion.

Second, the above difference directly impacts the learning dynamics: as illustrated in Fig. 4, the editor achieves a more stable convergence, in contrast to the oscillations seen in the generative ones. This difference can be further explained in Fig. 3, the fine-tuning process significantly reshapes the characteristics of generative models, while the editor’s features are more like a “refinement” and “focusing”. After 30 epochs of fine-tuning, the generative model learned highly structured and semantic features (a4, b4, c4) from chaotic states (a2, b2, c2), achieving a qualitative leap. Whereas the editing ones make the well-structured features (a1, b1, c1) clearer and task-oriented (a3, b3, c3), with their features being incrementally honed rather than fundamen-

### 3. Methods

We first conduct an investigation into the fine-tuning process between the editor and generator in Sec 3.1. Then, we adapt the editor into an estimator (Sec 3.2) by introducing three key contributions to the training objective (Sec 3.3), GT quantization(Sec 3.4), and joint estimation (Sec 3.5).

Step1xEdit-Based: Epoch 1

###### Editing

Step1xEdit-Based: Epoch 30

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

- (a1) (b1) (c1) (d1)

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

- (a2) (b2) (c2) (d2)

- (a3) (b3) (c3) (d3)

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

- (a4) (b4) (c4) (d4)

Input Image

Generative

FLUX-Based: Epoch 30

FLUX-Based: Epoch 1

[Figure 88]

[Figure 89]

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

Block Error High

[Figure 102]

Activate

Inactive

Low

Block 1 Block 20 Block 35 Pred Error

Block 1 Block 20 Block 35 Pred Error

- Figure 3. Comparison between the Generative and Editing foundation models. We analyze the feature evolution at both the initial (Epoch 1) and final (Epoch 30) stages of fine-tuning, resulting in 4 groups. Each group presents: the DiT features at the input end (Block1), middle layers (Block20), output end (Block35), and the depth prediction’s AbsRel (Absolute Relative error). Visual implementation detailed in Sec B.

[Figure 103]

TrainingLoss

Training Epoch

0.10

0.09

0.08

0.073

- Figure 4. Quantitative comparison of the training loss between Generative and Editing foundation models. The main plot details the convergence loss from epoch 5 to 30, while the inset displays the steep initial loss reduction during the first 10 epochs, which occurs on a different scale.

Initially, we take the input image x ∈ RH×W×3 as the editing source and the geometric annotation y ∈ RH×W×3 as the expected editing results. First, the VAE, which consists an encoder E(·) and a decoder D(·), is used to encode the input image x into a latent representation zx = E(x) ∈ Rh×w×c. Then, the editing process is modeled as a flow path from a noise vector zy0 ∼ N(0,I) to the target latent representation zy1 = E(y). The trajectory is defined as:

zyt = tzy1 + (1 − t)zy0, t ∈ [0,1]. (1)

The DiT backbone, denoted as fθ, is trained to predict the velocity vector of this flow, which is simply v = dz

y t

dt = zy1 − zy0. The model is optimized by minimizing the flow matching loss [37]:

1,zy0∥v − fθ(zx,zyt ,t)∥2. (2)

L = Et,zy

In the inference stage, we predict the editing target zˆy1 by solving the following ordinary differential equation:

tally altered.

Third, the “structured learning” and “characteristics reshaping” mentioned above are unable to address the shortcomings of the generative model. As shown in Fig. 4 (epochs 20-30, especially ♦), the generative model’s training loss meets a bottleneck around 0.08, while the editing ones can reduce to 0.073. The Table 4 (ID6 vs. ID7) further verifies that this bottleneck persists at test time, resulting in a significant performance gap.

1

zˆy1 = zy0 +

fθ(zx,zyt ,t)dt, (3)

0

and the final dense geometry predictions are given by yˆ = D(zˆy1). More details of the flow matching process are provided in Appendix Sec D.

#### 3.3. Consistent Velocity Flow Matching with Deterministic Departure

In summary, the analysis of feature evolution, training dynamics, and performance consistently demonstrates that editing models provide a more stable, effective, and promising foundation for dense geometry estimation. Note that this conclusion can also generalize to other DiT-based editing models (see Sec 4.4).

Flow matching has been widely adopted in modern generative and editing modeling. While efforts have gone into adapting traditional denoising-based models for deterministic prediction [20], our work focuses on analyzing and adapting the flow matching paradigm inherent to our editor.

#### 3.2. From Editor To Estimator

As shown in Fig. 5, MeanFlow [17] has identified that, since the model learns the velocity over all possible flow paths in Eq 1, the global instantaneous velocity field becomes inherently non-linear and typically induces a curved

As shown in Fig. 2, we pose monocular dense estimation as an image editing task and use the Flow Matching Loss for supervision.

𝑡

[Figure 104]

𝑣(𝑡,𝑧 ,z )

Table 1. Quantization errors at BF16 precision on Virtual KITTI dataset. Calculation details in Appendix Sec C.

[Figure 105]

Instant

𝑣

z ~𝒩(0,1)

𝑧

(a)Uniform (b)Inverse (c)Logarithmic

𝐴𝑏𝑠𝑅𝑒𝑙 = 5.6

Error Absrel Error Absrel Error Absrel 80m 16cm 0.002 125m 1.563 1.04m 0.013

(a) (b)

0.1m 16cm 1.600 0.2mm 0.002 1.3mm 0.013

[Figure 106]

[Figure 107]

[Figure 108]

𝑣(𝑧 )

[Figure 109]

BF16Precision

𝑣

Consist

𝑧

z = 𝟎

𝐴𝑏𝑠𝑅𝑒𝑙 = 4.8

(c) (d)

- Figure 5. Left: GT velocity field for network training. The gray dots represent different Gaussian noise (top) or zero starting point (bottom), the red dots represent data samples. Right: Instantaneous velocity v determines the tangent direction and creates errors in the cumulative path (top); The constant speed path is a straight line.

trajectory. During inference, as shown in Fig. 5 (b), the ideal integration path in Eq. 3 is approximated by a discrete numerical solver, which introduces a non-trivial approximation error for high-precision tasks such as dense geometric estimation.

Figure 6. Illustration of BF16 quantization error. (b) and (d) show GT depth visualized with BF16 precision. For clarity, (a) and (c) use Maigold’s VAE regularization, mapping max/min values to 1/1, respectively.

An intuitive idea is to find a straight integration path, which means the velocity direction always remains the same. In this paper, we further require the velocity magnitude to be consistent so that the velocity is completely independent of t, and redefine the loss as:

precision saves the training cost, but also for their typical outputs, RGB images; this precision is entirely sufficient. Specifically, a normalized BF16 value is represented as:

##### V = (−1)S × 2(E−127) × (1.F)2,

1,zy0∥v − fθ(zx,zy0)∥2. (4) Additionally, for deterministic dense prediction tasks, the stochastic nature of generative/editing models is unnecessary. Therefore, we simplify the objective by reducing it from an expectation of all zy0 ∼ N(0,1) to the fixed zy0 = 0. For simplicity, we omit this term and rewrite the loss as:

L = Ezy

where S is the sign (1 bit), E is the exponent (8 bits), and F is the fraction (7 bits). Due to the [-1,1] data range of VAE encoded input [38], the worst-case precision occurs at ±[0.5, 1.0], which is 2126−127×2−7 = 1/256 and perfectly satisfies the RGB range of 0-255.

Uncritically finetuning these models with FP32, as done in Marigold or Lotus, not only increases training/inference costs, but also leads to suboptimal inheritance of the baseline model’s priors, and restricts the capabilities of BF16only models like Step1X-Edit. Therefore, finetuning with BF16 is necessary.

∥v − fθ(zx)∥2, (5) and the inference process can be simplified as:

L = Ezy

1

1

zy1 = zy0 +

fθ(zx)dt = 0 + (1 − 0)fθ(zx) = fθ(zx).

0

(6) Overall, as shown in Fig. 5 (c) and (d), our refined flow matching not only eliminates the errors introduced by discretized curved trajectories and random starting points, but also significantly reduces inference time, achieving simultaneous improvements in both performance and efficiency.

However, as shown in Fig. 6 (a,b), when meeting the depth annotations in the Virtual KITTI dataset, the valid depth range is 0-80m. Uniformly regularizing to [-1,1] requires a reduction of 40 times, and the accuracy of 1/256 is reflected in the original depth with a significant error of 40/256≈0.16m. These errors result in an AbsRel of 1.6 at 0.1m (Table 1 (a)) and make the finetune process unfeasible. Previous works have employed an inverse quantization scheme, which means converting the reciprocal of the depth, or disparity, to BF16 precision (Fig. 6 (c, d)). As

#### 3.4. Logarithmic Annotation Quantization

Modern generative/editing models are almost exclusively trained with BF16 precision. This is not only because BF16

shown in Table 1 (b), despite offering extremely high precision at close ranges, this scheme becomes entirely unusable at greater distances, and even makes 39m and 78m correspond to the same value. The principles and calculations are detailed in Appendix Sec C.

After numerous attempts and explorations, we use the logarithmic depth quantization to achieve good precision at both near and far ranges (Table 1(c)) while reducing training and inference costs. Specifically, we first perform the logarithmic quantization with Dlog = ln(DGT + 1e − 6), then follow and refine Marigold’s depth normalization strategy, defining the supervision label yD as:

yD =

(Dlog − Dlog,2) (Dlog,98 − Dlog,2) − 0.5 × 2 , (7)

where Dlog,i corresponds to the i% percentiles of Dlog, and ⟨·⟩ is the BF16 precision truncation.

#### 3.5. Cost-Free Joint Estimation

Joint estimation of depth and normals can benefit from their potential connections. Unlike GeoWizard[15] that introduces additional cross-attention and switchers, we leverage the DiT’s global attention to perform joint estimation in a single forward pass.

As shown in Fig. 2 grey part, Step1X-Edit and other DiT-based editing works have found that, the DiT architecture can effectively guide image generation by horizontally concatenating the noise and condition latents, which means the input is formulated as: zx+Θ = concat(zx,zΘ) ∈ Rh×2w×c, where zΘ is the noise latents, shown in Fig. 2. However, after processing by the DiT, although the model’s output has the same shape as the input, fθ(zx+Θ) = [pl,pr] ∈ Rh×2w×c, supervision is only applied to the region corresponding to the original noise, i.e., L = ∥v − pr∥2, where pl,pr ∈ Rh×w×c. This means pl is computed but ultimately discarded, and creates a 50% computational waste. Based on this observation, without introducing any additional training or inference costs, we further incorporate another task’s supervision on pl during finetuning, extending Eq. 5 for both tasks as:

(∥vD − pl∥2 + ∥vN − pr∥2), (8)

Lfm = Ezy

1

where vD/vN are velocity training objectives for depth/normal task, respectively.

In addition to computational efficiency, we observe that this joint estimation strategy can also yield consistent performance gain. We hypothesize that this benefit may stem from the DiT’s global attention, which allows implicit information exchange and helps resolve challenging regions in both tasks’ features, as qualitatively suggested in Fig. 8.

### 4. Experiments

#### 4.1. Implementation Details

We build FE2E upon the Step1X-Edit v1.0 framework [38]. To further enhance the DiT’s representational power, we also introduce an auxiliary dispersion loss that encourages features from different samples to spread out in the hidden space, which is detailed in Appendix Sec A.1. During finetuning, all parameters except for the DiT module are frozen, and the language control input is left blank. The process employs LoRA [22] with rank = 64 and scale factor α = 32. We trained for 30 epochs using the AdamW optimizer [40] with an initial learning rate of 1 × 10−4. With gradient checkpoint enabled, the model can be trained on a single RTX 4090 GPU, but to accelerate experimentation, training was conducted on NVIDIA H20 GPUs, completing in approximately 1.5 days.

#### 4.2. Training Datasets

We train our model for joint depth and normal estimation on a mixture of two synthetic datasets: Hypersim [55] and Virtual KITTI [4]. For Hypersim, a photorealistic indoor dataset, we use its official training split after filtering out samples with over 1% invalid pixels, resulting in approximately 51k images at a 1024×768 resolution. For Virtual KITTI, a synthetic street view dataset, we utilize four driving scenarios, totaling around 20k samples at a 1216×352 resolution with a maximum depth of 80m. Following Marigold, each training batch is constructed by sampling from Hypersim and Virtual KITTI with probabilities of 90% and 10%, respectively.

The evaluation datasets [10, 16, 58, 62, 64] and metrics also follow Marigold and are detailed in Appendix Sec A.2, A.3, respectively.

#### 4.3. Quantitative Evaluation

Zero-shot Depth Estimation Comparison As presented in Table 2, FE2E significantly outperforms recent SoTA methods across five challenging benchmarks. Notably, on the ETH3D and KITTI datasets, it reduces the AbsRel error by 35% and 10% respectively, compared to the 2ndbest method. Remarkably, despite being trained on only 0.071M images, FE2E’s average rank surpasses that of the DepthAnything series, which was trained on a massive 62.6M image dataset. This highlights the effectiveness of our strategy: inheriting the editing model priors rather than simply scaling up training data. Furthermore, qualitative comparisons in Fig. 1 and 7 demonstrate that FE2E produces superior results in challenging lighting conditions (extreme-light, low-light, etc.) and better preserves distant details, which reveal the core advantages that contribute to FE2E’s superior performance. We provide further comparisons with concurrent unified works in Appendix Sec F.

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

RGB

###### RGB

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Lotus-D

Lotus-D

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

DSINE

Depth Anything V2

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

FE2E

FE2E

Figure 7. Quantitative comparison on zero-shot depth and normal estimation. The 1st row shows the input, the 2nd, 3rd rows are previous SoTA methods results, and the 4th row is ours prediction. White arrows highlight the regions we significantly improved.

- Table 2. Quantitative comparison on zero-shot affine-invariant depth estimation between FE2E and SoTA methods. The best and second best performances are highlighted. ⋆ denotes the method relies on pre-trained Stable Diffusion.

Method

Training NYUv2 (Indoor) KITTI (Outdoor) ETH3D (Various) ScanNet (Indoor) DIODE (Various) Avg Data↓ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ Rank↓

MiDaS [53] 2M 11.1 88.5 23.6 63.0 18.4 75.2 12.1 84.6 33.2 71.5 10.6 GeoWizard [15] 280K 5.6 96.3 14.4 82.0 6.6 95.8 6.4 95.0 33.5 72.3 8.4 GenPercept [74] 74K 5.6 96.0 13.0 84.2 7.0 95.6 6.2 96.1 35.7 75.6 7.8 Marigoldv1.1⋆ [29] 74K 5.8 96.1 11.0 88.8 7.0 95.5 6.6 95.3 30.4 77.3 7.6 Marigold⋆ [29] 74K 5.5 96.4 9.9 91.6 6.5 95.9 6.4 95.2 30.8 77.3 6.3 DepthAnything V2 [79] 62.6M 4.5 97.9 7.4 94.6 13.1 86.5 - - 26.5 73.4 5.4 Lotus-G⋆ [20] 59K 5.4 96.8 8.5 92.2 5.9 97.0 5.9 95.7 22.9 72.9 4.7 Diffusion-E2E-FT⋆ [41] 74K 5.4 96.5 9.6 92.1 6.4 95.9 5.8 96.5 30.3 77.6 4.6 Lotus-D⋆ [20] 59K 5.1 97.2 8.1 93.1 6.1 97.0 5.5 96.5 22.8 73.8 3.7 DepthAnything [78] 62.6M 4.3 98.1 7.6 94.7 12.7 88.2 4.3 98.1 26.0 75.9 3.5 FE2E 71K 4.1 97.7 6.6 96.0 3.8 98.7 4.4 97.5 22.8 81.2 1.4

- Table 3. Quantitative comparison on zero-shot surface normal estimation between FE2E and SoTA methods. ‡refers to the Marigold normal model as detailed in their repository.

Training NYUv2 (Indoor) ScanNet (Indoor) iBims-1 (Indoor) Sintel (Outdoor) Avg. Data↓ MeanErr↓ 11.25◦↑ MeanErr↓ 11.25◦↑ MeanErr↓ 11.25◦↑ MeanErr↓ 11.25◦↑ Rank

Method

Marigold‡⋆ [29] 74K 20.9 50.5 21.3 45.6 18.5 64.7 - - 9.5 GeoWizard⋆ [15] 280K 18.9 50.7 17.4 53.8 19.3 63.0 40.3 12.3 8.9 GenPercept⋆ [74] 74K 18.2 56.3 17.7 58.3 18.2 64.0 37.6 16.2 7.4 StableNormal⋆ [80] 250K 18.6 53.5 17.1 57.4 18.2 65.0 36.7 14.1 7.2 Lotus-G∗ [20] 59K 16.5 59.4 15.1 63.9 17.2 66.2 33.6 21.0 5.2 DSINE [1] 160K 16.4 59.6 16.2 61.0 17.1 67.4 34.9 21.5 4.6 Lotus-D⋆ [20] 59K 16.2 59.8 14.7 64.0 17.1 66.4 32.3 22.4 3.0 Diffusion-E2E-FT⋆ [41] 74K 16.5 60.4 14.7 66.1 16.1 69.7 33.5 22.3 2.6 Marigoldv1.1 ∗ [29] 77K 16.1 60.5 14.5 66.1 16.3 68.5 - - 2.0 FE2E∗ 71K 16.2 59.6 13.8 67.2 15.1 70.6 31.2 22.3 1.6

Zero-shot Normal Estimation Comparison As presented in Table 3, FE2E also achieves SoTA performance on the zero-shot normal estimation task, outperforming the

methods in the recent 2 years across four benchmarks. This quantitative superiority stems from its ability to handle complex geometries. As illustrated in Fig. 1, 7, FE2E excels

- Table 4. Ablation studies of our adaptation protocol. Here we show the results in depth estimation. CV: Consistent Velocity; FS: Fixed Start; JE: Joint Estimation; Quant: Quantization (sub-items same with Table 1). DirectAdapt refers to the formulation in Sec 3.2, and Improved setting denotes all improvements except JE (output dimensions of FLUX cannot support JE).

KITTI ETH3D AbsRel↓ δ1 ↑ AbsRel↓ δ1 ↑

ID Note Foundation CV FS JE Quant

- 1 DirectAdapt FLUX Direct 9.7 91.2 6.0 96.0

- 2 DirectAdapt Step1X-Edit Direct 9.5 91.4 5.6 96.2

- 3 Step1X-Edit ✓ Direct 8.8 93.2 5.0 97.2

- 4 Step1X-Edit ✓ ✓ Direct 8.6 94.0 4.8 97.3

- 5 Step1X-Edit ✓ ✓ Inverse 6.9 95.1 4.6 98.2

- 6 Improved Step1X-Edit ✓ ✓ Logarithmic 6.8 95.6 3.9 98.6

- 7 Improved FLUX ✓ ✓ Logarithmic 7.1 94.9 4.5 97.8

- 8 FE2E (full) Step1X-Edit ✓ ✓ ✓ Logarithmic 6.6 96.0 3.8 98.7

- 9 Extension FLUX-Kontext ✓ ✓ ✓ Logarithmic 6.7 96.1 3.6 98.8

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

RGB w/ Joint Estimation w/o Joint Estimation

at reconstructing intricate details such as surface folds and small objects, which are often challenging for other models.

#### 4.4. Ablation Study

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Effect of Foundation Model. FE2E is based on the new foundation model Step1X-Edit, the direct adaptation protocol (ID2) establishes a strong baseline, outperforming Marigold (based on SD v2) by 8% and 4% on ETH3D and KITTI datasets, respectively (AbsRel, the same below). Building on this, FE2E (ID8) further reduces the AbsRel by 32.1% on ETH3D and 30.5% on KITTI, which confirms the effectiveness of our proposed techniques.

[Figure 151]

Figure 8. Qualitative comparison on the Joint Estimation. The ‘w/o Joint Estimation’ shows two models’ results.

Effect of Editing Priors. We trained the FLUX-based model under both DirectAdapt and Improved settings, corresponding to ID1 and ID7 in Table 4. Compared with their counterparts (ID2 and ID6), the editing-based models consistently outperform the generative models, regardless of equipping our proposed improvements. These results, together with the findings in Sec 3.1, highlight the effectiveness of leveraging editing model priors for dense prediction tasks.

notable improvements in challenging scenarios such as flat butterfly structures and distant buildings.

Extensibility to Other Editors. We apply our adaptation protocol to the concurrent FLUX-Kontext model. The finetuned model (ID9) achieves comparable or even superior performance to its Step1X-Edit counterpart (ID8), likely due to the stronger editing priors in FLUX-Kontext, which confirms the broad applicability and high potential of our approach.

Effect of Improved Flow Matching. Adopting the consistent velocity training objective effectively eliminates accumulated inference errors from the original paradigm, leading to notable performance gains of 7% on KITTI and 10% on ETH3D (ID2 v.s. ID3). Introducing a fixed starting point further eases optimization and brings additional improvements (ID3 v.s. ID4).

### 5. Conclusion

In this paper, our systematic analysis shows that editors provide a more stable and effective foundation than their generative counterparts. Based on this, we introduce FE2E, a novel framework that successfully adapts a pre-trained editing model for depth and normal geometry prediction. To bridge the gap between these tasks, we proposed a consistent velocity training objective for stable convergence and logarithmic quantization to resolve precision conflicts. We also repurpose the editor’s discarded region to design a costfree joint estimation strategy, improving the inference efficiency. FE2E achieves SoTA performance and validates the ‘From Editor to Estimator’ paradigm, showcasing that harnessing the inherent abilities of editing models is an effective and data-efficient approach for dense geometry prediction.

Effect of Data Quantization. Supervision leads to substantial performance gains, with ID6 outperforming ID4 by 19% and 13% on the KITTI and ETH3D datasets, respectively. Notably, inverse quantization (ID5) generally outperforms uniform quantization, typically because there are more valid pixels nearby.

Effect of Joint Estimation. While the motivation is to improve computational efficiency, the results from ID6 and ID8 demonstrate this “cost-free” strategy can also achieve consistent performance gains. This synergistic effect is more clearly illustrated in Fig. 8, where joint training yields

### Acknowledgment

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant (62573039, U2441242) and Graduate Research Innovation Project under Grant (KKYJS25001536).

### References

- [1] Gwangbin Bae and Andrew J. Davison. Rethinking inductive biases for surface normal estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022.
- [3] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 611–

625. Springer, 2012.

- [4] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020.
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [6] Chong Cheng, Yu Hu, Sicheng Yu, Beizhen Zhao, Zijian Wang, and Hao Wang. Reggs: Unposed sparse views gaussian splatting with 3dgs registration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8100–8109, 2025.
- [7] Chong Cheng, Xianda Chen, Tao Xie, Wei Yin, Weiqiang Ren, Qian Zhang, Xiaoyuang Guo, and Hao Wang. Longstream: Long-sequence streaming autoregressive visual geometry. arXiv preprint arXiv:2602.13172, 2026.
- [8] Xiangxiang Chu, Jianlin Su, Bo Zhang, and Chunhua Shen. Visionllama: A unified llama backbone for vision tasks. In European Conference on Computer Vision, pages 1–18. Springer, 2024.
- [9] Xiangxiang Chu, Renda Li, and Yong Wang. Usp: Unified self-supervised pretraining for image generation and understanding. In ICCV, 2025.
- [10] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.
- [11] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multitask mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10786–10796, 2021.
- [12] David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In Proceedings of the IEEE inter-

- national conference on computer vision, pages 2650–2658, 2015.
- [13] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014.
- [14] Xianze Fang, Jingnan Gao, Zhe Wang, Zhuo Chen, Xingyu Ren, Jiangjing Lyu, Qiaomu Ren, Zhonglei Yang, Xiaokang Yang, Yichao Yan, and Chengfei Lyu. Dens3r: A foundation model for 3d geometry prediction. arXiv preprint arXiv:2507.16290, 2025.
- [15] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. arXiv preprint arXiv:2403.12013, 2024.
- [16] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE, 2012.
- [17] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling, 2025.
- [18] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014.
- [19] Ming Gui, Johannes Schusterbauer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Depthfm: Fast monocular depth estimation with flow matching, 2024.
- [20] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and YingCong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124, 2024.
- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020.
- [22] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
- [23] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. arXiv preprint arXiv:2404.15506, 2024.
- [24] Yu Hu, Chong Cheng, Sicheng Yu, Xiaoyang Guo, and Hao Wang. Vggt4d: Mining motion cues in visual geometry transformers for 4d scene reconstruction. arXiv preprint arXiv:2511.19971, 2025.
- [25] Dongyang Jin, Ryan Xu, Jianhao Zeng, Rui Lan, Yancheng Bai, Lei Sun, and Xiangxiang Chu. Semantic context matters: Improving conditioning for autoregressive models. arXiv preprint arXiv:2511.14063, 2025.
- [26] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.

- [27] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020.
- [28] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34:852–863, 2021.
- [29] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492– 9502, 2024.
- [30] Zong Ke, Yuqing Cao, Zhenrui Chen, Yuchen Yin, Shouchao He, and Yu Cheng. Early warning of cryptocurrency reversal risks via multi-source data. Finance Research Letters, page 107890, 2025.
- [31] Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Korner. Evaluation of cnn-based single-image depth estimation methods. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018.
- [32] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025.
- [33] Rui Lan, Yancheng Bai, Xu Duan, Mingxing Li, Lei Sun, and Xiangxiang Chu. Flux-text: A simple and advanced diffusion transformer baseline for scene text editing, 2025.
- [34] Changbai Li, Haodong Zhu, Hanlin Chen, Juan Zhang, Tongfei Chen, Shuo Yang, Shuwei Shao, Wenhao Dong, and Baochang Zhang. Hrgs: Hierarchical gaussian splatting for memory-efficient high-resolution 3d reconstruction, 2025.
- [35] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025.
- [36] Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [37] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2022.
- [38] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- [39] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022.
- [40] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [41] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025.
- [42] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations, 2022.
- [43] Shervin Minaee, Xiaodan Liang, and Shuicheng Yan. Modern augmented reality: Applications, trends, and future directions, 2022.
- [44] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [45] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 Conference Proceedings, 2023.
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [47] Bowen Ping, Chengyou Jia, Minnan Luo, Changliang Xia, Xin Shen, Zhuohang Dang, and Hangwei Qian. Pacorl: Advancing reinforcement learning for consistent image generation with pairwise reward modeling. arXiv preprint arXiv:2512.04784, 2025.
- [48] Bowen Ping, Chengyou Jia, Minnan Luo, Hangwei Qian, and Ivor Tsang. Flow-factory: A unified framework for reinforcement learning in flow-matching models. arXiv preprint arXiv:2602.12529, 2026.
- [49] Xiangyan Qu, Zhenlong Yuan, Jing Tang, Rui Chen, Datao Tang, Meng Yu, Lei Sun, Yancheng Bai, Xiangxiang Chu, Gaopeng Gou, Gang Xiong, and Yujun Cai. From scale to speed: Adaptive test-time scaling for image editing, 2026.
- [50] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks, 2016.
- [51] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.
- [52] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022.

- [53] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular

- depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020.
- [54] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.
- [55] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021.
- [56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [57] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.
- [58] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017.
- [59] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278–25294, 2022.
- [60] Wenhao Shen, Wanqi Yin, Xiaofeng Yang, Cheng Chen, Chaoyue Song, Zhongang Cai, Lei Yang, Hao Wang, and Guosheng Lin. Adhmr: Aligning diffusion-based human mesh recovery via direct preference optimization. arXiv preprint arXiv:2505.10250, 2025.
- [61] Wenhao Shen, Hao Wang, Wanqi Yin, Fayao Liu, Xulei Yang, Chao Liang, Zhongang Cai, and Guosheng Lin. Vlmguided group preference alignment for diffusion-based human mesh recovery. arXiv preprint arXiv:2602.19180, 2026.
- [62] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12, pages 746–760. Springer, 2012.
- [63] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer, 2025.
- [64] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z Dai, Andrea F Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R Walter, et al.

- Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463, 2019.
- [65] Honglie Wang, Yan-Ming Zhang, Wangzi Yao, Fei Yin, and Cheng-Lin Liu. Learning to generate stylized handwritten text via a unified representation of style, content, and noise. In The Fourteenth International Conference on Learning Representations, 2026.
- [66] Jiyuan Wang, Chunyu Lin, Lang Nie, Shujun Huang, Yao Zhao, Xing Pan, and Rui Ai. Weatherdepth: Curriculum contrastive learning for self-supervised depth estimation under adverse weather conditions. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 4976–4982. IEEE, 2024.
- [67] Jiyuan Wang, Chunyu Lin, Lang Nie, Kang Liao, Shuwei Shao, and Yao Zhao. Digging into contrastive learning for robust depth estimation with diffusion models. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 4129–4137, 2024.
- [68] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.
- [69] Jiyuan Wang, Chunyu Lin, Cheng Guan, Lang Nie, Jing He, Haodong Li, Kang Liao, and Yao Zhao. Jasmine: Harnessing diffusion prior for self-supervised depth estimation, 2025.
- [70] Jiyuan Wang, Chunyu Lin, Lei Sun, Zhi Cao, Yuyang Yin, Lang Nie, Zhenlong Yuan, Xiangxiang Chu, Yunchao Wei, Kang Liao, et al. Geometry-guided reinforcement learning for multi-view consistent 3d scene editing. arXiv preprint arXiv:2603.03143, 2026.
- [71] Runqian Wang and Kaiming He. Diffuse and disperse: Image generation with representation regularization, 2025.
- [72] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning. arXiv e-prints, pages arXiv–2507, 2025.
- [73] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025.
- [74] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Diffusion models trained with large data are transferable visual models. arXiv preprint arXiv:2403.06090, 2024.
- [75] Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi, Haiyang Sun, Bing Wang, et al. Pixel-perfect depth with semantics-prompted diffusion transformers. arXiv preprint arXiv:2510.07316, 2025.

- [76] Ryan Xu, Dongyang Jin, Yancheng Bai, Rui Lan, Xu Duan, Lei Sun, and Xiangxiang Chu. Scalar: Scale-wise controllable visual autoregressive learning. arXiv preprint arXiv:2507.19946, 2025.
- [77] Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attngan: Finegrained text to image generation with attentional generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1316– 1324, 2018.
- [78] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024.
- [79] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024.
- [80] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. arXiv preprint arXiv:2406.16864, 2024.
- [81] Junliang Ye, Fangfu Liu, Qixiu Li, Zhengyi Wang, Yikai Wang, Xinzhou Wang, Yueqi Duan, and Jun Zhu. Dreamreward: Text-to-3d generation with human preference, 2024.
- [82] Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. Shapellm-omni: A native multimodal llm for 3d generation and understanding. arXiv preprint arXiv:2506.01853, 2025.
- [83] Junliang Ye, Shenghao Xie, Ruowen Zhao, Zhengyi Wang, Hongyu Yan, Wenqiang Zu, Lei Ma, and Jun Zhu. Nano3d: A training-free approach for efficient 3d editing without masks, 2025.
- [84] Qian Yu, Zong Ke, Guofu Xiong, Yu Cheng, and Xiaojun Guo. Identifying money laundering risks in digital asset transactions based on ai algorithms. In 2024 4th International Conference on Electronic Information Engineering and Computer Communication (EIECC), pages 1081–1085. IEEE, 2024.
- [85] Zhenlong Yuan, Chengxuan Qian, Jing Tang, Rui Chen, Zijian Song, Lei Sun, Xiangxiang Chu, Yujun Cai, Dapeng Zhang, and Shuo Li. AutoDrive-R2: Incentivizing reasoning and self-reflection capacity for vla model in autonomous driving. arXiv preprint arXiv:2509.01944, 2025.
- [86] Zhenlong Yuan, Xiangyan Qu, Chengxuan Qian, Rui Chen, Jing Tang, Lei Sun, Xiangxiang Chu, Dapeng Zhang, Yiwei Wang, Yujun Cai, et al. Video-star: Reinforcing openvocabulary action recognition with tools. arXiv preprint arXiv:2510.08480, 2025.
- [87] Jianhao Zeng, Dan Song, Weizhi Nie, Hongshuo Tian, Tongtong Wang, and An-An Liu. Cat-dm: Controllable accelerated virtual try-on with diffusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8372–8382, 2024.
- [88] Jianhao Zeng, Yancheng Bai, Ruidong Chen, Xuanpu Zhang, Lei Sun, Dongyang Jin, Ryan Xu, Nannan Zhang, Dan

- Song, and Xiangxiang Chu. Eevee: Towards close-up high-resolution video-based virtual try-on. arXiv preprint arXiv:2511.18957, 2025.
- [89] Ziyao Zeng, Jingcheng Ni, Daniel Wang, Patrick Rim, Younjoon Chung, Fengyu Yang, Byung-Woo Hong, and Alex Wong. Iris: Integrating language into diffusionbased monocular depth estimation. arXiv preprint arXiv:2411.16750, 2024.
- [90] Ziyao Zeng, Yangchao Wu, Hyoungseob Park, Daniel Wang, Fengyu Yang, Stefano Soatto, Dong Lao, Byung-Woo Hong, and Alex Wong. Rsa: Resolving scale ambiguities in monocular depth estimators through language descriptions. Advances in neural information processing systems, 37: 112684–112705, 2024.
- [91] Ziyao Zeng, Jingcheng Ni, Ruyi Liu, and Alex Wong. Coffee: Controllable diffusion fine-tuning. arXiv preprint arXiv:2511.14113, 2025.
- [92] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 5907– 5915, 2017.
- [93] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan++: Realistic image synthesis with stacked generative adversarial networks. IEEE transactions on pattern analysis and machine intelligence, 41(8):1947–1962, 2018.
- [94] Han Zhang, Jing Yu Koh, Jason Baldridge, Honglak Lee, and Yinfei Yang. Cross-modal contrastive learning for text-toimage generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 833–842, 2021.
- [95] Canyu Zhao, Yanlong Sun, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks. arXiv preprint arXiv:2502.17157, 2025.
- [96] Heng Zhou, Jing Tang, Jusheng zhang, Yanshu Li, Canran Xiao, Liwei Hou, Zong Ke, and Jiawei Yao. Comem: Compositional concept-graph memory for vision–language adaptation. In The Fourteenth International Conference on Learning Representations, 2026.

## FE2E: From Editor to Dense Geometry Estimator Supplementary Material

In this appendix, we provide more implementation details, experiments, analysis, and discussions for a comprehensive evaluation and understanding of FE2E. Detailed contents are listed as follows:

- A. Experiment Settings 13

- A.1. Auxiliary Dispersion Loss . . . . . . . . . . 13
- A.2. Evaluation Datasets . . . . . . . . . . . . . . 13
- A.3. Evaluation Metrics . . . . . . . . . . . . . . 13

- B. Training Details of Finetune Analysis 13

- B.1. Improved Experiment Setup . . . . . . . . . 13
- B.2. Implementation Details of Generative-based Models . . . . . . . . . . . . . . . . . . . . 14

- C. Quantization Error Calculation Details 14

- C.1. Uniform Quantization . . . . . . . . . . . . . 14
- C.2. Inverse Quantization . . . . . . . . . . . . . 14
- C.3. Logarithmic Quantization . . . . . . . . . . . 14

- D. Preliminaries of Flow Matching 15
- E. Reviews of Related Generative and Editing Models 15
- F . Addition Experiments Results 16
- G. Limitations and Future Work 16 A. Experiment Settings

#### A.1. Auxiliary Dispersion Loss

Following Diffuse-and-Disperse [71], we apply this loss to the output of the 9th block:

Ldisp = log Ei,j exp(−∥ηi − ηj∥22/τ) , (9)

where ηi,j are the output features for the i-th and j-th samples in a batch, respectively, and temperature τ = 1. Finally, the training loss is defined as: Ltrain = Lfm + λLdisp,λ = 0.5. The choices of λ, τ, and block all follow the optimal hyperparameters identified in the experiments from Diffuse-and-Disperse.

Table 5. Ablation study on Disperse Loss. The baseline is the ID4 model in the main paper, Table 4.

KITTI ETH3D AbsRel↓ δ1 ↑ AbsRel↓ δ1 ↑

Method

Baseline (CV + FS) 8.6 94.0 4.8 97.3 + Disperse Loss (DL) 8.4 94.4 4.5 97.6

For integrity, we also conducted ablation studies on this dispersed loss. The performance gains observed in Table 5

confirm that this loss is also effective for dense geometric estimation tasks.

#### A.2. Evaluation Datasets

We evaluate our model on two tasks: Zero-shot AffineInvariant Depth Estimation. We evaluate on five standard benchmarks: NYUv2 [62], ScanNet [10], KITTI [16], ETH3D [58], and DIODE [64]. Following standard practice, we report the Absolute Relative error (AbsRel) and δ1 accuracy. Surface Normal Prediction. We evaluate on NYUv2, ScanNet, iBims-1 [31], and Sintel [3] benchmarks. The evaluation metrics are the mean angular error (MeanErr) and the percentage of pixels with an angular error below 11.25◦.

#### A.3. Evaluation Metrics

For zero-shot depth estimation, similar to [29], we employ the following evaluation metrics:

- • AbsRel: |M1

vl| d∈Mvl |d − dgt|/dgt;

- • a1: percentage of d such that max(dd

, ddgt) < 1.25 ;

gt

where dgt and d denote the GT and estimated pixel depth, Mvl is the valid mask (mask rules are consistent with [20]).

For zero-shot normal estimation, we use the following evaluation metrics:

- • MeanErr

= |M1

vl| n∈Mvl

180

π arccos(clamp(n · ngt,−1,1));

- • 11.25◦: The percentage of n where the angular error is less than 11.25◦;

where ngt and n are GT and estimated normal vector.

### B. Training Details of Finetune Analysis

#### B.1. Improved Experiment Setup

For clarity, we term the direct adaptation of the original editing/generative formulation as “DirectAdapt” (Sec 3.2), and Table 4 shows that DirectAdapt fails to achieve satisfactory performance. To address this, we introduce two key improvements on training objective (Sec 3.3) and GT quantization(Sec 3.4). They can benefit both editing and generative models, and these improved models are better for analyzed our core motivation (Sec 3.1), as they isolate the error from training data and the denoising process. We finally introduce joint training on the editing-based model to obtain FE2E.

#### B.2. Implementation Details of Generative-based Models

Step1X-Edit is fine-tuned from the generative model FLUX, and both share an almost identical DiT architecture. To further reduce confounding factors, we follow the Step1X-Edit protocol and replace the original FLUX input with a horizontally concatenated noise and RGB image. All hyperparameters, including LoRA settings, optimizer, and training data, are kept exactly the same as those used for FE2E indepth estimation.

FLUX consists of 38 block layers, each producing outputs of consistent dimensions. After rearrangement, the feature map has the shape B × 192 × H/8 × W/8, where B is the batch size, H and W are the height and width of the input image. Typically, the output from the final block is projected to 16 channels and passed to the VAE for reconstruction to B ×3×H ×W. For visualization, we chose 1, 20, and 35 blocks, operate on the B × 192 × H/8 × W/8 feature map, normalize it to B × 1 × H/8 × W/8 using the L2 norm, upsample it to B × 1 × H × W, and finally visualize it using the Rainbow colormap. The visualization of depth and normals follows the approach of Lotus.

Since our experimental comparisons are conducted using the improved model, only one single “denoising” step is performed during inference. Consequently, the output from the VAE decoder directly represents the depth map (the B ×3×H ×W output mentioned before was averaged to obtain a 1-channel depth map), which makes it easier to visualize meaningful features.

### C. Quantization Error Calculation Details

The following calculations are based on the effective depth range of 0-80m from the Virtual KITTI dataset. The normalization scheme consistently maps an input domain X to the VAE’s mandatory input range of [-1, 1] using the standard min-max scaling formula:

X − Xmin Xmax − Xmin − 1

V = 2 ×

. While other mapping schemes from [0m, 80m] to [-1, 1] may exist, they are not explored in this work. All calculations use the worst-case precision of BF16 over the [-1, 1] interval, which corresponds to a single quantization step of ∆V ≈ 1/256.

C.1. Uniform Quantization

In this scheme, the depth value D is linearly mapped to the [-1, 1] interval. The depth range is [Dmin,Dmax] = [0m,80m]. The mapping function is V = 2 × 80D−−00 − 1 =

###### D 40 − 1. A quantization step of ∆V = 1/256 in the normalized space corresponds to an error ∆D in the real-world

depth space. This error is constant across the entire depth range:

1 256 ≈ 0.15625

∆D = 40 × ∆V = 40 ×

At 80m: Error ≈ 16cm. AbsRel = 080.16mm = 0.002. At 0.1m: Error ≈ 16cm. AbsRel = 00..161mm = 1.600. This method yields an unacceptably large relative error

at close distances.

#### C.2. Inverse Quantization

This scheme quantizes the reciprocal of depth, i.e., disparity P = 1/D. We consider an effective depth range of [0.1m,80m] to avoid division by zero. The corresponding disparity range is [Pmin,Pmax] = [1/80,1/0.1] = [0.0125,10]. The disparity P is linearly mapped to [-1, 1]. The quantization step in disparity, ∆P, is constant:

∆V 2 ≈ 0.0195.

∆P = (Pmax − Pmin) ×

The relationship between depth error ∆D and disparity error ∆P is given by ∆D ≈ |d(1dP/P)|∆P = P12 ∆P = D2∆P.

At 80m: Error = (80m)2 × 0.0195 = 6400 × 0.0195 ≈ 124.8m ≈ 125m. AbsRel = 12580mm ≈ 1.563.

At 0.1m: Error = (0.1m)2 × 0.0195 = 0.01 × 0.0195 =

- 0.000195m ≈ 0.2mm. AbsRel = 0.00002.1mm = 0.002. As mentioned in the main text, the disparities for 39m

and 78m are 1/39 ≈ 0.0256 and 1/78 ≈ 0.0128, respectively. Their difference is ≈ 0.0128, which is smaller than the disparity quantization step ∆P ≈ 0.0195, making them indistinguishable after quantization. This scheme fails completely at large distances.

C.3. Logarithmic Quantization

This scheme quantizes the logarithmic depth, Dlog = ln(D). We again consider the depth range [0.1m,80m]. The corresponding log-depth range is [ln(0.1),ln(80)] ≈ [−2.30,4.38]. The log-depth Dlog is linearly mapped to [-1,

- 1]. The quantization step in log-depth, ∆Dlog, is constant:

∆V

∆Dlog = (ln(80) − ln(0.1)) ×

2 ≈ 0.013. The relationship between depth error ∆D and log-depth error ∆Dlog is given by ∆D ≈ |d(e

Dlog)

dDlog |∆Dlog = eD

∆Dlog = D · ∆Dlog. This implies that the absolute relative error, AbsRel = ∆D/D, is approximately constant and equal to ∆Dlog ≈ 0.013.

log

At 80m: AbsRel ≈ 0.013. Error = 80m × 0.013 = 1.04m.

At 0.1m: AbsRel ≈ 0.013. Error = 0.1m × 0.013 = 0.0013m = 1.3mm.

This method maintains a reasonable and nearly constant relative error across both near and far ranges, making it a well-balanced and effective solution. The percentile-based normalization used in the main text is a more robust implementation of this fundamental principle.

### D. Preliminaries of Flow Matching

Flow Matching [37] is a highly effective framework for training Continuous Normalizing Flows (CNFs). The core idea is to smoothly transform a simple prior distribution p0 (e.g., the standard Gaussian distribution N(0,I)) into a complex target data distribution p1 over a continuous time variable t ∈ [0,1].

This transformation process can be described by an Ordinary Differential Equation (ODE), where the velocity at any time t and point z is defined by a vector field vt(z). However, estimating this marginal vector field vt(z) directly from data samples is challenging. The Flow Matching framework elegantly bypasses this issue by regressing a much simpler and easier-to-compute conditional vector field ut(z|z0,z1) instead.

Specifically, we first sample a pair of points, (z0,z1), from the prior distribution p0 and the target distribution p1, respectively. We then define a simple path zt from z0 to z1 and its corresponding conditional vector field ut = dz

dt . It has been proven that if a neural network fθ(z,t) is trained to approximate this simple conditional vector field ut, then in expectation over all sample pairs (z0,z1) and time t, the network fθ will converge to the complex marginal vector field vt that we truly wish to learn.

t

Rectified Flow [39] presents a particularly simple and powerful instance of Flow Matching. It defines the path between z0 and z1 as a straight line:

zt = tz1 + (1 − t)z0, t ∈ [0,1].

The derivative of this path is trivial, yielding a constant velocity vector that is independent of both time and space:

dzt dt

= z1 − z0.

v =

Consequently, the training objective (loss function) becomes exceedingly simple: aligning the neural network’s prediction with this constant velocity vector v:

1,z0∥(z1 − z0) − fθ(zt,t)∥2.

L = Et,z

Application in DirectAdapt In this paper, we adapt this framework for a conditional image editing task. Our goal is not to learn an unconditional generative model, but rather a flow from noise zy0 to the target geometry latent zy1, guided

by the input image x (encoded as zx). Therefore, our velocity prediction model fθ must take zx as an additional condition. As shown in Eq. 2 in the main text, our loss function is:

1,zy0∥(zy1 − zy0) − fθ(t,zx)∥2.

L = Et,zy

During inference, we generate the target latent zˆy1 by solving the following ODE, with zx serving as the guiding condition:

dzˆyt dt

= fθ(t,zx), with initial value zˆy0 ∼ N(0,I).

By integrating from t = 0 to t = 1 using a numerical ODE solver (e.g., Euler method), we can obtain the final prediction zˆy1.

### E. Reviews of Related Generative and Editing Models

The fields of image generation and image editing have always been complementary, and they have undergone several paradigm shifts. The first major breakthrough was the Generative Adversarial Network (GAN) [18], which introduced a novel adversarial training process. Then, key advancements in this era include architectural refinements like DCGAN [50], the development of conditional and text-toimage GANs such as the StackGAN series [92, 93], AttnGAN [77], and Cross-Modal Contrastive Learning based models [94]. The StyleGAN series [26–28] marked a high point for GANs, achieving unprecedented photorealistic high-resolution image synthesis and offering fine-grained control over visual attributes through a disentangled latent space, which became a cornerstone for many subsequent editing techniques.

More recently, the field has transitioned to Denoising Diffusion Models [21], which have become the state-of-theart for their superior image quality and textual coherence. A series of influential diffusion-based methods were introduced, including GLIDE [44], DALL·E [51] and its successor DALL·E 2 [52], Imagen [57], and PIXART-α [5]. The open-source Stable Diffusion (SD) [56] model, trained on the large-scale LAION-5B dataset [59], further democratized high-quality image generation and quickly became a community standard. A growing body of evidence suggests that Diffusion Transformers [8, 9, 32, 46] outperform UNets, motivating the shift toward training modern diffusion models with Transformer architectures.

Building on these powerful generative foundations, the domain of image editing [33] (generalized editing) has also advanced rapidly. Early diffusion-based methods like SDEdit [42] demonstrated that real images could be edited by adding noise and then denoising with a new text prompt. A significant leap was made with instruction-guided editing, pioneered by InstructPix2Pix [2], which enabled edits

Table 6. Quantitative comparison on zero-shot affine-invariant depth estimation between FE2E and the concurrent unified model.

NYUv2 (Indoor) KITTI (Outdoor) ETH3D (Various) ScanNet (Indoor) DIODE (Various) Avg

Method

AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ AbsRel↓ δ1↑ Rank↓

Qwen-Image 5.5 96.7 7.8 95.1 6.6 96.2 4.7 97.4 19.7 83.2 2.6 DINOv3 4.3 98.0 7.3 96.7 5.4 97.5 4.4 98.1 25.6 82.2 1.8 FE2E 4.1 97.7 6.6 96.0 3.8 98.7 4.4 97.5 22.8 81.2 1.6

based on natural language commands. The field has since diversified with numerous innovative approaches. For instance, DragGAN [45] introduced a novel point-based interaction, allowing users to “drag” pixels to precisely deform object shapes. OmniControl [63] further enhances controllability by creating a unified framework that accepts diverse spatial guidance signals for both synthesis and editing. This trend towards more powerful and versatile models is also reflected in large-scale systems like UniWorld [35], which uses a unified transformer for multi-modal understanding and generation, Step1X-Edit [38], fine-tuned from the FLUX architecture for superior instruction following, and multi-modal editors like Qwen-Image [73], which leverage Large Language Models (LLMs) to build more comprehensive visual editing frameworks.

We also note several recent studies from our broader collaborators on adjacent topics, including visual geometry and 4D reconstruction [6, 7, 24], 3D generation, editing, and multimodal understanding [70, 81–83], controllable generation, virtual try-on, and adaptive inference for image editing [25, 49, 65, 76, 87, 88], controllable diffusion and language-guided depth estimation [89–91], preference alignment or reinforcement learning for diffusion and flow-based models [47, 48, 60, 61], and a broader set of downstream applications such as compositional visionlanguage adaptation, autonomous driving, action recognition, and risk analysis [30, 84–86, 96].

### F. Addition Experiments Results

Comparison with Concurrent Unified Model The field of dense geometry estimation is advancing rapidly, with the task of depth estimation particularly fast. Recently, several concurrent works have been explored to unify the visual tasks, which also include depth estimation benchmarks. As shown in Table 6, our method consistently achieves the top average ranking, even though they are trained with extremely huge data compared to FE2E (e.g., Qwen Image utilizes billions of samples, and DINO v3 is trained on 1.7 billion images).

Additional Qualitative Comparison Fig. 9 presents a qualitative comparison between FE2E and other methods. The results demonstrate that our approach produces more

refined and accurate depth predictions, particularly in structurally complex regions that may not be fully captured by quantitative metrics. Furthermore, as illustrated in Fig. 10, FE2E consistently delivers precise surface normal predictions, effectively handling intricate geometries and diverse environments. These results highlight the robustness of our method in fine-grained prediction tasks.

Table 7. Performance comparison of different models.

|Methods|Marigold Lotus-D Qwen Image DINO v3 FE2E|
|---|---|
|MACs RunTime AbsRel<br><br>|133T 2.65T 2.13P 14.5T 28.9T 9.67s 212ms 63.4s 632ms 1.78s 6.5 6.1 6.6 5.4 3.8<br><br>|

### G. Limitations and Future Work

Large computational load We present the inference latency and computational complexity of the FE2E model in Table 7, alongside comparisons with previous SD-based and unified methods. Although incorporating DiT does lead to a notable increase in computational complexity relative to other self-supervised approaches, FE2E strikes a trade-off between performance and computational efficiency.

Diversifying foundation models The field of image editing is evolving rapidly, and our approach is designed to be model-agnostic. In future work, we plan to incorporate a broader range of editing models to further substantiate the motivation and conclusions presented in this paper.

Scaling up the training data While a key contribution of this work is demonstrating strong generalization performance with a limited amount of training data, we still anticipate that scaling up the training dataset could further improve the model’s capabilities. This direction is meaningful for domains that are not sensitive to computational complexity but require extremely high prediction accuracy. We leave the exploration for future research.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

RGB Depth Anything V2 Lotus-D FE2E

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

[Figure 173]

[Figure 174]

[Figure 175]

- Figure 9. Additional qualitative comparison on zero-shot affine-invariant depth estimation. FE2E achieves more accurate depth predictions, particularly in structurally complex regions. White arrows highlight these improvements.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

RGB DSINE Lotus-D FE2E

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

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

- Figure 10. Additional qualitative comparison on zero-shot surface normal estimation. FE2E offers improved accuracy, particularly in detailed and complex regions.

