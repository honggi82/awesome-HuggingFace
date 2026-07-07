arXiv:2506.03147v4[cs.CV]18Jun2025

[Figure 1]

## UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation

#### Bin Lin1,3, Zongjian Li1,3, Xinhua Cheng1,3, Yuwei Niu1,3, Yang Ye1,3, Xianyi He1,3, Shenghai Yuan1,3, Wangbo Yu1,3, Shaodong Wang1,3, Yunyang Ge1,3, Yatian Pang1, Li Yuan1,2,

†

1 Peking University, Shenzhen Graduate School, 2 Peng Cheng Laboratory, 3 Rabbitpre AI

Code: https://github.com/PKU-YuanGroup/UniWorld-V1

Models: https://huggingface.co/LanguageBind/UniWorld-V1

Data: https://huggingface.co/datasets/LanguageBind/UniWorld-V1

Image Perception

[Figure 2]

Reference Detection Segment Canny Depth Sketch Normal

|[Figure 3]|
|---|

|[Figure 4]|
|---|

|[Figure 5]|
|---|

|[Figure 6]|
|---|

|[Figure 7]|
|---|

|[Figure 8]|
|---|

|[Figure 9]|
|---|

###### Image Manipulation

Edit Reference Remove Replace Background Adjust Style Add

|[Figure 10]|
|---|

|[Figure 11]|
|---|

|[Figure 12]|
|---|

|[Figure 13]|
|---|

|[Figure 14]|
|---|

|[Figure 15]|
|---|

|[Figure 16]|
|---|

Extract: Crop out everything except the <object>

Restoration: Remove the blur

Apply hues to the image

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

Try-on: Integrate the <object> into the person's look

Style reference: Adapt <img1> to fit the style of <img2>

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

###### +

- Figure 1: Showcase of UniWorld-V1’s versatile capabilities. The left two panels illustrate image perception and manipulation examples, and the right panel presents comparisons with state-of-the-art models and training data resources.

### Abstract

Although existing unified models achieve strong performance in vision-language understanding and text-to-image generation, they remain limited in addressing image perception and manipulation—capabilities increasingly demanded in practical applications. Recently, OpenAI introduced the powerful GPT-4o-Image model, which showcases advanced capabilities in comprehensive image perception and manipulation, sparking widespread interest. Through carefully designed experiments, we observe that GPT-4o-Image likely relies on semantic encoders rather than VAEs for feature extraction, despite VAEs being commonly regarded as crucial for image manipulation tasks. Inspired by this insight, we propose UniWorld-V1, a unified generative framework built upon semantic features extracted from powerful multimodal large language models and contrastive semantic encoders. Using only 2.7M training data, UniWorld-V1 achieves impressive performance across diverse tasks, including image understanding, generation, manipulation, and perception. We fully open-source the UniWorld-V1 framework, including model weights, training and evaluation scripts, and datasets to promote reproducibility and further research.

### 1 Introduction

Unifying image understanding and generation has demonstrated remarkable capabilities on multimodal models. Recent studies [40, 55, 44] demonstrate that carefully designed architectures can be jointly optimized to perform well on both understanding and generation benchmarks. The remarkable visual understanding and generation capabilities exhibited by the recent GPT-4o-Image model have further energized the open-source community, leading to a surge of new unified models [4, 9] aiming to replicate its performance. GPT-4o-Image achieves superior generation performance on various image-to-image tasks across different domains, which can be divided into two categories, including image perception tasks [36, 8] (e.g., detection) and image manipulation [48] (e.g., editing) tasks.

Nevertheless, most unified models are limited to image-to-language understanding tasks and languageto-image generation tasks, with very few addressing image-to-image perception and manipulation tasks. However, attempting to tackle image perception and manipulation tasks in one model is difficult because such an all-in-one model requires multiple superior capabilities, including (1) the textual and visual unified understanding ability for correctly addressing user intention, (2) the pixel-level information maintaining ability for image reconstruction and region editing, and (3) the semantic extraction ability for cross-domain perception and visual conception composition.

Therefore, the requirements of various capabilities impose specific designs for the unified generation model, especially for visual feature injection. Recent attempts including Step1X-Edit [26] and FLUXKontext [15] introduce variational autoencoders (VAEs) for extracting visual features and perform well on individual image editing tasks. However, their methods encounter challenges when extended to multiple perception and manipulation tasks simultaneously due to the visual features encoded by VAEs, which imply a wealth of low-frequency information, thereby limiting their performance when facing semantic-level tasks.

Inspired by the success of GPT-4o-Image, we investigate the integration of visual features into unified generative models for image manipulation tasks, a direction that remains insufficiently explored in current research. Therefore, we carefully construct experiments on GPT-4o-Image to infer the visual feature extraction manner that GPT-4o-Image likely adopts, and we infer that GPT-4o-Image employs visual features extracted by semantic encoders rather than VAEs by observing from the experimental results.

Based on our essential observation shown in Section 2, we propose a unified generative model named UniWorld-V1 for both image perception and manipulation tasks, which consists of pre-trained multi-modal large models for providing auto-regressive understanding tokens and pre-trained highresolution contrastive semantic encoders for extracting visual features with both pixel-level local information and semantic-level global conceptions.

As a result, UniWorld-V1, trained on only 2.7M samples, consistently outperforms BAGEL [9] (trained on 2665M samples) on the ImgEdit-Bench [48] for image manipulation. It also surpasses the specialized image editing model Step1X-Edit across multiple dimensions, including add, adjust, and extract on ImgEdit-Bench. Additionally, for text-to-image generation, UniWorld-V1 outperforms BAGEL on WISE [29] and achieves performance comparable to GPT-4o-Image on GenEval [14]. Furthermore, beyond its strong capabilities in the aforementioned tasks, UniWorld-V1 also excels in Image Perception tasks such as detection, segmentation, and depth prediction. In summary, we are the first open-source model to achieve such comprehensive and powerful capabilities across the multimodal domain.

We summarize our primary contributions as follows:

- • Provide insights into unified architecture design by hypothesizing, through extensive observation of GPT-4o-Image, that it likely does not adopt a VAE-based structure.
- • Propose UniWorld-V1, a unified architecture that leverages a high-resolution semantic encoder to deliver reference-image control signals. UniWorld-V1 achieves performance comparable to BAGEL using only 2.7M training samples.
- • Collect, curate, and open-source high-quality training data. We fully release the code, model weights, datasets, and training & evaluation scripts to support and advance future research.

[Figure 35]

[Figure 36]

Denoise this image

Denoise this image

[Figure 37]

Input:

[Figure 38]

[Figure 39]

User User

Turn the advertisement on the back of the bus blue

User

Input

Input

[Figure 40]

[Figure 41]

[Figure 42]

Generating image

Generating image

[Figure 43]

[Figure 44]

GPT-4o GPT-4o

Respond

Respond

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

(b) Visual Manipulation (0.4×noise)

[Figure 49]

This is a dog. Specifically, it appears to be a Siberian Husky, based on its size, fur pattern, and characteristic markings.

[Figure 50]

What animal?

[Figure 51]

GPT-4o

User

[Figure 52]

Respond:

The image shows a dog runing on a sandy surface, likely a beach. Baed on the appearance, the dog resembles a siberian Husky or a similar breed with a thick coat and distinctive facial features. The dog has a mix of dark and light fur, which is characteristic of many northern breeds.

[Figure 53]

Turn the advertisement on the back of the bus blue

Input

GPT-4o

Qwen2.5 VL-32B

[Figure 54]

- (d) Visual Understanding (0.4×noise)

[Figure 55]

(c) Visual Manipulation (0.6×noise)

[Figure 56]

- (e) Visual Understanding (0.6×noise)

[Figure 57]

The image appears to be heavily pixelated or noisy, making it very difficult to clearly identify the animal. However, based on the shape and general outline visible through the noise, it looks like it might be a deer or elk standing in a natural environment.

[Figure 58]

What animal?

[Figure 59]

GPT-4o

User

The image appears to be quite pixelated, making it dificut to identify the animals with ertainty. However,based on the visible shapes and forms,itseems to depict a group of animals that could be wowes ora simlar type of large canine, The blured and indistinct nature of the image makesitchalenging to provide a defnitive identifcation. ifyou havmore details or a clearer image, it would be easier to identify the animals accurately.

[Figure 60]

[Figure 61]

Input

Qwen2.5 VL-32B

[Figure 62]

(a) Local Editing

- Figure 2: Empirical Observations of GPT-4o-Image. We verify local consistency of edits in (a). We explore the relationship between model comprehension and generation in (b)–(e), conducting observations within the GPT-4o architecture and across architectures using Qwen2.5VL-32B.

### 2 Observation

GPT-4o-Image [1] achieves impressive performance in various image tasks in a generative way, and we divide supported image tasks into two categories: Image Perception (detection, segmentation, depth prediction, etc.) and Image Manipulation (editing [26, 58, 6], reference style transfer [38], subject consistency generation [52, 51], etc.). It is generally believed within the community that GPT-4o-Image demonstrates the necessity of unifying understanding and generation by integrating an auto-regressive understanding module [60, 19, 21] with a diffusion-based generation module [47] for addressing various image tasks with complex requirements. However, we consider that additional visual features beyond auto-regressive tokens from understanding models should be injected to maintain image information, since we fail to train an effective generation model from solely VLM outputs. Recent notable unified image manipulation approaches, exemplified by Step1X-Edit [26] and FLUX-Kontext [15], simultaneously introduce variational autoencoders (VAEs) to extract image featuresa as reference image control. Although their models perform well on individual editing tasks, our experiments show that they fail to converge when extended to multiple image perception and manipulation tasks, which indicates that the manner of additional visual feature injection remains underexplored. To explore the visual feature injection method that GPT-4o-Image utilizes, we construct two groups of experiments and obtain several key observations (see Figure 2), from which we infer that GPT-4o-Image likely employs visual features extracted by semantic encoders rather than VAEs.

Editing Experiment. We first require GPT-4o-Image to execute a local image editing task with the instruction: “Turn the advertisement on the back of the bus blue”, as shown in Figure 2 (a). Before the editing, both yellow-labeled and green-labeled texts are in the top-right corner of the bus. However, the yellow-labeled text is on the right, and the green-labeled text is on the bottom right after editing. We claim that if GPT-4o-Image leverages VAE features that strongly preserve the low-frequency

information for visual injection, the positions of texts would remain virtually unchanged, while GPT-4o-Image renders both texts into inconsistent positions.

Denoising Experiment. We then corrupt a dog image with noise levels of 0.4× and 0.6× for GPT-4o-Image to execute an image denoising task with the instruction: “Denoise this image”, as shown in Figure 2 (b) & (c). We observe that GPT-4o-Image performs normally when the noise is small, but wrongly denoises the image as a deer when the noise level is 0.6×. We claim that VAE features preserve the low-frequency components of the input (e.g., global structures and contours) and lead to correct denoise results. Besides, to figure out why GPT-4o-Image denoises the dog image as a deer, we query two multi-modal understanding models, including GPT-4o and Qwen2.5-VL, as shown in Fig. 2 (d) & (e). Interestingly, both understanding models caption the 0.6×-noised dog image as a deer, which demonstrates GPT-4o-Image is based on the prior of powerful multi-modal understanding models.

In summary, our experiments demonstrate that GPT-4o-Image more likely employs visual features extracted by semantic encoders rather than VAEs since the low-level information of the source image is not preserved after the manipulation. Inspired by such observations, we design the architecture of our unified image perception and manipulation methods named UniWorld-V1.

### 3 Method

[Figure 63]

Trainable parameter

Image Generation （Flow Matching）

Non-trainable

Frozen in Stage I Trained in Stage II

# DiT

Optional path

Understanding （Auto-Regressive）

High-level semantics Low-level control Image branch

MLP

| || | |
|---|---|
|ML|P|
| | |
| | |
<br><br>|
|---|---|
|SigLIP<br><br>| |

| | |
|---|---|
|ML|P|
| | |
| | |

[Figure 64]

T5 (Optional)

VLM

[Figure 65]

Segment the giraffe from the

[Figure 66]

- Figure 3: Model architecture. The model consists of a VLM, SigLIP, DiT [33], and MLP connector. High-level semantics and historical state are provided by the VLM, while low-level image features are controlled by SigLIP. The understanding part uses a frozen VLM with an autoregressive approach, while the generation part is trained with flow matching. T5 (originally used for conditional injection) is optional during training or generation.

#### 3.1 Model Architecture

Based on our empirical observations, we replace the VAE-based low-level control signal with the SigLIP encoder [54, 41] (Figure 3), a contrastive vision-language model that demonstrates superior performance. We use the largest resolution variant, which is SigLIP2-so400m/14 with a fixed resolution of 512. For visual understanding, we follow prior work [32, 4] and use the pretrained Qwen2.5-VL-7B [2] as the base module. The reference image is processed by both Qwen2.5-VL-7B and SigLIP, and their outputs are concatenated as the input to the text branch of FLUX [16].

#### 3.2 Training Recipe

During training, the T5 [35] features (the original condition used in FLUX) are optional. However, we observe that incorporating T5 features in the early stages often leads to convergence to poor local minima. Therefore, we do NOT recommend using T5 features early in training.

- Stage 1: Pretraining for Semantic Alignment Due to a feature gap between VLM representations and the FLUX text branch, stage 1 focuses on aligning VLM features to T5 features. During this stage, only the MLP mapping from VLM to FLUX is trainable, while all other parameters remain frozen. Moreover, since stage 1 is solely dedicated to aligning VLM semantic features, SigLIP features are excluded. After stage 1 pretraining, the model can perform text-to-image generation and produce images that differ from the reference based on editing instructions.
- Stage 2: Fine-Tuning for Consistent Generation We load the weights of the VLM to FLUX MLP trained in stage 1 and the MLP weights from FLUX-Redux [15], which align SigLIP features to the text branch. We unfreeze all learnable parameters in the FLUX image branch while keeping all text branch parameters frozen. Although stage 1 aligns VLM to FLUX, early in stage 2, the model still takes a shortcut by directly reconstructing the reference image. After 5,000 to 10,000 training steps, the model begins learning how to use SigLIP features as reference cues to generate images according to instructions.

#### 3.3 ZeRO-3 EMA

EMA offers more stable and consistent weight averaging during training and is stored in FP32 to preserve numerical precision. This approach ensures that weight fluctuations are smoothed over time, which improves convergence behavior and promotes better generalization. Because our model is extremely large, storing an extra FP32 copy on each GPU would strain computational resources and potentially limit overall batch size. As shown in Figure 4, the training model (DiT) operates under ZeRO-2, while the EMA model is sharded across GPUs using ZeRO-3. By leveraging ZeRO-3 for the EMA, each GPU holds only a fraction of the full FP32 parameters, enabling efficient memory utilization. For instance, a 20B model sharded across N GPUs requires each GPU to hold only 20×4 GiB, which minimizes redundant storage. We updateNthe

EMA (ZeRO-3) DiT (ZeRO-2)

- GPU-0

- GPU-1

- GPU-2

- GPU-3

Figure 4: Zero-3 EMA. EMA model is initialized with Zero-3-style sharding across GPUs to reduce overhead. During each step, each GPU updates only its own shard.

EMA every step, which reduces its computation to N1 and ensures that computation cost remains low as the number

of GPUs increases. This scheme also supports running the training model under ZeRO-3, further decreasing memory overhead and allowing larger effective batch sizes.

#### 3.4 Training Data

We use almost identical data in two stages. This includes open-source high-quality data, self-generated data, and filtered open-source data. Data types include:

- 1. Image Perception: canny, mlsd, hed, depth, sketch, segmentation (mask), detection (bounding box). Most data originates from (1) Graph200k [18], (2) COCO2017 [23]. Although perception maps usually reach 1024 × 1024 resolution, over 90% of reference images are limited to lower resolutions (e.g., 512 × 512). Since perception maps and reference images differ substantially, the mask weighting strategy is unnecessary. Image perception data amounts to approximately 1.4M.
- 2. Image Manipulation: common edit types such as add, remove, replace, etc. Main sources are ImgEdit [48] and SEED-X [12]. ImgEdit provides over 1M editing samples. We use the subset with high score, totaling 724k higher-quality samples. In SEED-X, we select part 3 with resolutions of at least 1024 × 1024. We also collect style transfers from Graph200k given reference style images, as well as virtual try-on and product extraction data. Since most open-source data lack editing masks, we generate edit masks following Section 3.5. Image manipulation data amounts to approximately 1M.

- 3. Text-to-Image Generation: sources include BLIP3-o [4] and internal images from the Open-Sora Plan [20]. Open-Sora Plan images receive dense captions from Qwen2-VL72B [42], all with resolutions of at least 1024 × 1024 and aesthetic score at least 6.0. Text-to-image generation data amounts to approximately 300k.

Reference Image (1) (2) (3) (4)

|[Figure 67]<br><br>Aedit|
|---|

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

| |
|---|

Target Image Weight Function

|[Figure 72]|
|---|

|[Figure 73]|
|---|

|[Figure 74]|
|---|

|[Figure 75]|
|---|

|[Figure 76]|
|---|

- Figure 5: Pipeline for mask generation. Given a reference image and a target image, the mask is obtained through (1) pixel-wise differencing, (2) dilation, (3) connected component filtering, and

(4) max-pooling downsampling. The bottom right shows four different weighting functions. We highlight the limitations of steps (1), (2), and (3), which are addressed in the next stage.

#### 3.5 Adaptive Editing Region Weighting Strategy

In image editing tasks, the edited region typically occupies only a small portion of the image. If uniform loss weighting is applied across the whole image, the loss signal from the relatively small edited area may be overwhelmed by that from the much larger unedited region. This imbalance can cause the model to underfit the edited content, failing to capture fine-grained or user-intended changes.

A straightforward approach is to use the mask region (i.e., the edited area) as a weight. However, not all edited data comes with masks. Therefore, we obtain masks through the following four steps:

- (1) Pixel-wise Differencing: Compute the pixel-wise difference between the reference and target images. We set a tolerance threshold to determine whether a pixel region is edited or unedited. However, as shown in Figure 5 (1), this produces many noisy differences.
- (2) Dilation: Expand each edited pixel using a dilation factor to reduce noise, though many isolated pixels may still remain.
- (3) Connected Component Filtering: Remove small connected components to eliminate spurious edits, but this does not address bubbles within larger edited regions.
- (4) Max-pooling Downsampling: Apply max-pooling to remove internal noise within connected regions. Finally, we obtain the edited area size, denoted as Aedit.

To prevent small edited regions from being overwhelmed by the background during training, we design a weighting strategy that assigns higher loss weights to edited pixels. The weight is a function of the relative area ratio between the full image and the edited region: x = A

Aedit , where the total image area is denoted as Atotal. We require the weighting function w(x) to satisfy w(1) = 1, so that when the entire image is edited (i.e., text-to-image or style transfer, Aedit = Atotal), the loss reverts to uniform weighting. We design and compare four candidate functions:

total

- (1) Linear: w(x) = x
- (2) Exponential Root: w(x) = 2

√x−1

- (3) Logarithmic: w(x) = log2(x) + 1
- (4) Quadratic Root: w(x) = (√x − 1)2 + 1

All four functions ensure w(1) = 1, and increase as the edit area shrinks (x → ∞). Among them, the logarithmic function (3) grows moderately, avoids instability for extremely small regions, and

Table 1: Comparison between different models on Understanding & Generation & Editing benchmarks. † refer to the methods using LLM rewriter. “×” indicates the model is incapable of performing the task.

Understanding Image Generation Image Editing

Model

MMBV MMBI MMMU MM-Vet GenEval WISE Overall Add Adjust Extract Replace Remove Hybird

###### Image Understanding

LLaVA-1.5 [25] × 36.4 67.8 36.3 × × × × × × × × × LLaVA-NeXT [57] × 79.3 51.1 57.4 × × × × × × × × ×

###### Image & Video Understanding

Video-LLaVA [22] 1.05 60.9 32.8 32.0 × × × × × × × × × LLaVA-OV [17] 0.94 80.8 48.8 57.5 × × × × × × × × ×

###### Text-to-Image Generation

SDXL [34] × × × × 0.55 0.55 × × × × × × × FLUX.1 Dev [16] × × × × 0.66 0.50 × × × × × × ×

###### Image Editing

MagicBrush [56] × × × × × × 1.83 2.84 1.58 1.51 1.97 1.58 1.62 Instruct-P2P [3] × × × × × × 1.88 2.45 1.83 1.44 2.01 1.50 1.20 AnyEdit [49] × × × × × × 2.45 3.18 2.95 1.88 2.47 2.23 1.56 UltraEdit [59] × × × × × × 2.70 3.44 2.81 2.13 2.96 1.45 1.91 Step1X-Edit [27] × × × × × × 3.06 3.88 3.14 1.76 3.40 2.41 2.64

Unified Understanding & Generation Show-o [46] × - 27.4 - 0.68 0.35 × × × × × × × Janus [44] × 69.4 30.5 34.3 0.61 0.18 × × × × × × × Janus-Pro [7] × 75.5 36.3 39.8 0.80 0.35 × × × × × × × Emu3 [43] - 58.5 31.6 37.2 0.66† 0.39 - - - - - - MetaQuery-XL [32] - 83.5 58.6 66.6 0.80† 0.55 - - - - - - BAGEL [9] - 85.0 55.3 67.2 0.88† 0.52 3.20 3.56 3.31 1.70 3.30 2.62 2.38 UniWorld-V1 1.79 83.5 58.6 67.1 0.84† 0.55 3.26 3.82 3.64 2.27 3.47 3.24 2.96

maintains a good balance between sensitivity and robustness. Therefore, we adopt the logarithmic weighting function w(x) = log2(x) + 1 in our final implementation.

### 4 Experiment

#### 4.1 Main Results

- Table 1 presents our main comparative results. We compare the performance of our UniWorld-V1 model against other advanced models across three core benchmarks: image understanding, image generation, and image editing. The experimental results comprehensively demonstrate UniWorldV1’s exceptional performance across these three major categories, proving its powerful unified capabilities. We achieved state-of-the-art or near state-of-the-art performance in every sub-category.

Specific comparisons for text-to-image generation will be provided in Section 4.2, image editing results in Section 4.3, visual understanding comparisons in Section 4.4, and image perception capabilities will be showcased through sampling examples in Section 4.5.

#### 4.2 Text-to-Image Generation

This section presents the performance of our UniWorld-V1 in text-to-image generation, primarily focusing on two aspects: the models’ fundamental object-focused text-to-image generation abilities, as evaluated by the GenEval [14], and their world knowledge reasoning capabilities for image generation, as assessed by the WISE [29].

Evaluation results on GenEval. Table 2 showcases the models’ evaluation results on GenEval. Our UniWorld-V1 achieves a strong performance with an overall score of 0.79. Furthermore, we observe that many models, such as MetaQuery, BLIP3-o, and BAGEL, utilize LLM rewriters for prompt rewriting to facilitate generation. For a fair comparison, we used the rewrite prompt used by blip3o for additional testing. UniWorld-V1 ultimately achieves an impressive score of 0.84. This result is remarkably close to the top-performing BAGEL’s 0.88, while UniWorld-V1 only utilizes 2.7M training data compared to BAGEL’s 2665M data. This stark difference in data efficiency powerfully highlights the superiority of our architecture.

Evaluation results on WISE. As presented in Table 3, our proposed UniWorld-V1 exhibits exceptionally strong performance as a unified model, achieving an overall score of 0.55. This makes UniWorld-V1 highly competitive among unified models in leveraging and integrating world knowledge for generating semantically rich and accurate images. Notably, UniWorld-V1 achieves an

- Table 2: Comparison results on GenEval. “Gen. Only” refers to pure generation models, while “Unified” indicates models capable of both understanding and generation. † refers to the methods using LLM rewriter. ‡: Results of GPT-4o-Image are tested by [47]. Model Single Obj.↑ Two Obj.↑ Counting↑ Colors↑ Position↑ Color Attribute↑ Overall↑

Gen. Only

PixArt-α [5] 0.98 0.50 0.44 0.80 0.08 0.07 0.48 Emu3-Gen [43] 0.98 0.71 0.34 0.81 0.17 0.21 0.54 SDXL [34] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 DALL-E 3 [37] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 SD3-Medium [10] 0.99 0.94 0.72 0.89 0.33 0.60 0.74 FLUX.1-dev† [16] 0.98 0.93 0.75 0.93 0.68 0.65 0.82

Unified

Janus [44] 0.97 0.68 0.30 0.84 0.46 0.42 0.61 Emu3-Gen†[43] 0.99 0.81 0.42 0.80 0.49 0.45 0.66 Show-o [46] 0.98 0.80 0.66 0.84 0.31 0.50 0.68 Janus-Pro-7B [7] 0.99 0.89 0.59 0.90 0.79 0.66 0.80 MetaQuery-XL† [32] - - - - - - 0.80 BLIP3-o [4] - - - - - - 0.84 BAGEL [9] 0.99 0.94 0.81 0.88 0.64 0.63 0.82 BAGEL† [9] 0.98 0.95 0.84 0.95 0.78 0.77 0.88 GPT-4o-Image‡ 0.99 0.92 0.85 0.92 0.75 0.61 0.84 UniWorld-V1 0.99 0.93 0.79 0.89 0.49 0.70 0.80 UniWorld-V1† 0.98 0.93 0.81 0.89 0.74 0.71 0.84

- Table 3: Comparison results on WISE. WISE evaluates the model’s capacity for complex semantic understanding and world knowledge in text-to-image generation. “Gen. Only” refers to pure generation models, while “Unified” indicates models capable of both understanding and generation. †: Results of GPT-4o-Image are tested by [47].

Model Cultural↑ Time↑ Space↑ Biology↑ Physics↑ Chemistry↑ Overall↑ Gen. Only

SDXL [34] 0.43 0.48 0.47 0.44 0.45 0.27 0.43 SD3.5-large [10] 0.44 0.50 0.58 0.44 0.52 0.31 0.46 PixArt-Alpha [5] 0.45 0.50 0.48 0.49 0.56 0.34 0.47 playground-v2.5 [24] 0.49 0.58 0.55 0.43 0.48 0.33 0.49 FLUX.1-dev [16] 0.48 0.58 0.62 0.42 0.51 0.35 0.50

###### Unified

Janus [44] 0.16 0.26 0.35 0.28 0.30 0.14 0.23 Show-o [46] 0.28 0.40 0.48 0.30 0.46 0.30 0.35 Janus-Pro-7B [7] 0.30 0.37 0.49 0.36 0.42 0.26 0.35 Emu3 [43] 0.34 0.45 0.48 0.41 0.45 0.27 0.39 MetaQuery-XL [32] 0.56 0.55 0.62 0.49 0.63 0.41 0.55 BAGEL [9] 0.44 0.55 0.68 0.44 0.60 0.39 0.52 GPT-4o-Image† 0.81 0.71 0.89 0.83 0.79 0.74 0.80 UniWorld-V1 0.53 0.55 0.73 0.45 0.59 0.41 0.55

impressive 0.73 in the Space category. This score is particularly significant as it is the closest to GPT-4o-Image’s 0.89 in this category, positioning UniWorld-V1 as the top performer among all other evaluated models (excluding GPT-4o-Image) in capturing and utilizing spatial world knowledge.

#### 4.3 Image Manipulation

As illustrated in Table 4, we compare UniWorld-V1 with other open-source models and GPT-4o-Image on editing capabilities using the ImgEdit-Bench [48] benchmark. Our UniWorld-V1 model achieves the best overall performance among all open-source models, with a total score of 3.26, outperforming other strong open-source competitors such as BAGEL (3.20) and Step1X-Edit (3.06). This highlights UniWorld-V1’s robust and consistent image editing ability across a wide spectrum of tasks. Notably, UniWorld-V1 ranks first among open-source models in several key categories, including Adjust

- Table 4: Comparison results on ImgEdit-Bench. “Overall” is calculated by averaging all scores across tasks. We use GPT-4.1 for evaluation.

Model Add Adjust Extract Replace Remove Background Style Hybrid Action Overall↑

MagicBrush [56] 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.83 Instruct-P2P [3] 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit [49] 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit [59] 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 Step1X-Edit [27] 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 BAGEL [9] 3.56 3.31 1.70 3.30 2.62 3.24 4.49 2.38 4.17 3.20 GPT-4o-Image 4.61 4.33 2.90 4.35 3.66 4.57 4.93 3.96 4.89 4.20 UniWorld-V1 3.82 3.64 2.27 3.47 3.24 2.99 4.21 2.96 2.74 3.26

(3.64), Extract (2.27), Replace (3.47), Remove (3.24), Background (2.99), and Hybrid (2.96). These results reflect UniWorld-V1’s strong capability in tasks that require nuanced attribute adjustment, structural modification, and complex multi-operation editing. While GPT-4o-Image continues to lead with a remarkable total score of 4.20, UniWorld-V1 emerges as the closest open-source contender, significantly narrowing the performance gap. This demonstrates UniWorld-V1’s notable progress towards achieving high-fidelity and generalizable image editing quality, comparable to state-of-the-art proprietary models.

To validate out-of-domain generalization, we use GEdit-Bench [27] and report scores on the English dataset as shown in Table 5. UniWorldV1’s perceptual score on GEdit-Bench (G_PQ) is relatively high (7.43), whereas its instructionfollowing score (G_SC) is relatively low (4.93). This is because most of our editing data originates from ImgEdit [48] and is limited in quantity, resulting in lower instruction diversity than BAGEL or Step1X-Edit. Additionally, text editing is a key evaluation metric in GEdit-Bench. Our dataset contains no text-editing samples, so it performs poorly in this aspect. Moreover, text editing is challenging for a 512-resolution SigLIP. We believe instruction-following capability can be improved by incorporating more data and fine-tuning the VLM. Regarding text-editing capability, we continue collecting relevant data and raising SigLIP’s resolution. Finally, we reiterate that our core contribution is finding that high-resolution SigLIP enables control over reference-image consistency.

Table 5: Comparison results on GEdit-Bench. G_SC, G_PQ, and G_O refer to the metrics evaluated by GPT-4.1.

#### Model G_SC↑ G_PQ↑ G_O↑

Instruct-P2P [3] 3.58 5.49 3.68 MagicBrush [56] 4.68 5.66 4.52 AnyEdit [49] 3.18 5.82 3.21 OmniGen [45] 5.96 5.89 5.06 Step1X-Edit [26] 7.09 6.76 6.70 BAGEL [9] 7.36 6.83 6.52 Gemini 2.0 [13] 6.73 6.61 6.32 GPT-4o [30] 7.85 7.62 7.53 UniWorld-V1 4.93 7.43 4.85

- 4.4 Visual Understanding

By benefiting from our strategy of freezing the Multimodal Large Language Model component, we successfully inherited the robust multimodal understanding capabilities of Qwen2.5-VL-7B without the need for retraining. This significantly reduces our resource consumption, specifically in terms of data and computational power, and also prevents potential degradation of understanding performance that could arise from training on generative tasks. As presented in Table 6, this architectural choice enables UniWorld-V1 to achieve remarkable results, significantly surpassing models like Janus, Show-o, and Emu3 across multiple metrics, and achieves highly competitive performance against the latest advanced models such as BAGEL.

Table 6: Comparison between different models on Visual Understanding benchmarks. × indicates the model is incapable of performing the task.

Model MMBV [11]MMBI [28]MMMU [53]MM-Vet [50] Image & Video Understanding

LLaVA-1.5 [25] × 36.4 67.8 36.3 Video-LLaVA [22] 1.05 60.9 32.8 32.0

Unified Understanding & Generation Show-o [46] × - 27.4 Janus [44] × 69.4 30.5 34.3 Janus-Pro [7] × 75.5 36.3 39.8 Emu3 [43] - 58.5 31.6 37.2 BLIP3-o [4] - 83.5 50.6 66.6 MetaQuery [32] - 83.5 58.6 66.6 BAGEL [9] - 85.0 55.3 67.2 GPT-4o 2.15 - 72.9 76.9 UniWorld-V1 1.79 83.5 58.6 67.1

Detection Canny Depth Normal HED Segmentation Sketch

|[Figure 77]|
|---|

|[Figure 78]|
|---|

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

|[Figure 83]|
|---|

UniWorld

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

|[Figure 87]|
|---|

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

GPT-4o

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

|[Figure 95]|
|---|

|[Figure 96]|
|---|

|[Figure 97]|
|---|

UniWorld

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

GPT-4o

- Figure 6: Showcase of UniWorld-V1’s perception capabilities. This figure presents a qualitative comparison of UniWorld-V1’s perception tasks against GPT-4o, using randomly selected examples. Green boxes indicate correct responses, while Red boxes highlight instances where the model’s output deviates from the expected result.

#### 4.5 Image Perception

As the first to integrate visual understanding, image perception and manipulation into a unified model, there is currently no suitable benchmark to comprehensively evaluate our UniWorld-V1 model’s full image perception capabilities. Therefore, we conducted a qualitative comparison using sampled examples against GPT-4o-Image, as detailed in Figure 6. It can be observed that UniWorld-V1 demonstrates highly competitive, and in many scopes superior, performance across various perception tasks. Specifically, UniWorld-V1 distinctly showcases stronger instruction understanding and task execution capabilities than GPT-4o-Image in canny edge detection, normal map generation, HED, segmentation, and sketch generation. This highlights that UniWorld-V1’s integrated architecture effectively enables a broad and accurate range of image perception functionalities, positioning it as the first open-source unified model capable of such diverse and high-fidelity visual analyses.

### 5 Conclusion

In summary, UniWorld-V1 demonstrates that a unified architecture anchored by a high-resolution semantic encoder, which address both image perception and manipulation tasks with state-of-the-art efficiency. By leveraging only 2.7M training samples, UniWorld-V1 achieves superior performance against much larger cost models on diverse benchmarks, confirming that semantic encoders provide richer and more versatile visual representations than traditional VAE-based approaches. This work establishes a foundation for future research in unified visual generation. We release all code, model weights, and datasets to foster continued innovation and collaboration within the community.

Limitation Despite UniWorld-V1’s remarkable training efficiency, the following shortcomings remain:

- • Insufficient instruction generalization. Limited training data and lack of VLM fine-tuning require specific instruction templates to outperform BAGEL.
- • Inadequate reference-image consistency. Reference images are processed at 512 × 512 resolution, which is insufficient to generate all details at 1024 × 1024 scale.
- • Incomplete benchmarks. DPG-Bench and GenAI-Bench with scores above a certain threshold often fail to reflect human preference, as verified through manual inspection.

Some samples in GenEval forcibly bind two objects that rarely co-occur in the real world, ignoring the natural distribution of images. ImgEdit-Bench and GEdit-Bench lack sufficient sensitivity to the reference regions.

#### Future Work

- • Continue collecting data and perform joint training with a VLM.
- • Integrate higher-resolution semantic encoders or adopt VLM techniques to increase inputimage resolution, such as multi-scale image gridding.

#### Failed Attempts

- • We empirically attempt to replace SigLIP with other encoders such as DINO V2 [31] and RADIO V2.5 [39], but the attempts are unsuccessful.
- • We attempt to use Qwen2.5VL’s visual output (picking the image feature of outputs while abandoning text feature) directly as the reference-image control signal. However, consistency between generated and reference images remains poor. This issue arises from the intrinsic gap between VLM training objectives and contrastive training. Contrastive learning focuses on global semantic features that saturate as resolution increases, whereas VLM training demands both global and local semantic information. As a result, the model capacity does not preserve sufficient low-level control signals, which likely causes the failure.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [4] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025.
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.
- [6] Liang Chen, Shuai Bai, Wenhao Chai, Weichu Xie, Haozhe Zhao, Leon Vinci, Junyang Lin, and Baobao Chang. Multimodal representation alignment for image generation: Text-image interleaved control is easier than you think. arXiv preprint arXiv:2502.20172, 2025.
- [7] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [8] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16901–16911, 2024.
- [9] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [11] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. Advances in Neural Information Processing Systems, 37:89098–89124, 2024.
- [12] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.
- [13] Google Gemini2. Experiment with gemini 2.0 flash native image generation, 2025.
- [14] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [15] Black Forest Labs. Flux. https://bfl.ai/announcements/24-11-21-tools, 2024.
- [16] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [17] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [18] Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li, Peng Gao, Zhanyu Ma, and MingMing Cheng. Visualcloze: A universal image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960, 2025.
- [19] Jiaqi Liao, Yuwei Niu, Fanqing Meng, Hao Li, Changyao Tian, Yinuo Du, Yuwen Xiong, Dianqi Li, Xizhou Zhu, Li Yuan, et al. Langbridge: Interpreting image as a combination of language embeddings. arXiv preprint arXiv:2503.19404, 2025.
- [20] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024.
- [21] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Yatian Pang, Munan Ning, et al. Moe-llava: Mixture of experts for large vision-language models. arXiv preprint arXiv:2401.15947, 2024.
- [22] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
- [23] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pages 740–755. Springer, 2014.
- [24] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.
- [25] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.
- [26] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

- [27] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.
- [28] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [29] Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.
- [30] OpenAI. Introducing 4o image generation, 2025.
- [31] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [32] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.
- [33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [35] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [36] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [37] Zhan Shi, Xu Zhou, Xipeng Qiu, and Xiaodan Zhu. Improving image captioning with better use of captions. arXiv preprint arXiv:2006.11807, 2020.
- [38] Yiren Song, Cheng Liu, and Mike Zheng Shou. Omniconsistency: Learning style-agnostic consistency from paired stylization data. 2025.
- [39] RADIOv2.5 Team. Flux. https://github.com/NVlabs/RADIO/blob/main/RADIOv2.5_ tech_report.md, 2024.
- [40] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.
- [41] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.
- [42] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.
- [43] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

- [44] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024.
- [45] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.
- [46] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.
- [47] Zhiyuan Yan, Junyan Ye, Weijia Li, Zilong Huang, Shenghai Yuan, Xiangyang He, Kaiqing Lin, Jun He, Conghui He, and Li Yuan. Gpt-imgeval: A comprehensive benchmark for diagnosing gpt4o in image generation. arXiv preprint arXiv:2504.02782, 2025.
- [48] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.
- [49] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024.
- [50] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [51] Shenghai Yuan, Xianyi He, Yufan Deng, Yang Ye, Jinfa Huang, Bin Lin, Chongyang Ma, Jiebo Luo, and Li Yuan. Opens2v-nexus: A detailed benchmark and million-scale dataset for subject-to-video generation. arXiv preprint arXiv:2505.20292, 2025.
- [52] Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyuan Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, and Li Yuan. Identity-preserving text-to-video generation by frequency decomposition. arXiv preprint arXiv:2411.17440, 2024.
- [53] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556–9567, 2024.
- [54] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.
- [55] Jihai Zhang, Tianle Li, Linjie Li, Zhengyuan Yang, and Yu Cheng. Are unified visionlanguage models necessary: Generalization across understanding and generation. arXiv preprint arXiv:2505.23043, 2025.
- [56] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023.
- [57] Y Zhang, B Li, H Liu, Y Lee, L Gui, D Fu, J Feng, Z Liu, and C Li. Llava-next: A strong zero-shot video understanding model. 2024.
- [58] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with in-context generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025.
- [59] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024.

##### [60] Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. arXiv preprint arXiv:2310.01852, 2023.

