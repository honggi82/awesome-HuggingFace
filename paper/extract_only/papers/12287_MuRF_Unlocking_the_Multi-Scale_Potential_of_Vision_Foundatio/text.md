## arXiv:2603.25744v2[cs.CV]2Apr2026

# MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models

Bocheng Zou†, Mu Cai†, Mark Stanley∗, Dingfu Lu∗, and Yong Jae Lee 1 1University of Wisconsin-Madison

Vision Foundation Models (VFMs) have become the cornerstone of modern computer vision, offering robust representations across a wide array of tasks. While recent advances allow these models to handle varying input sizes during training, inference typically remains restricted to a single, fixed scale. This prevalent single-scale paradigm overlooks a fundamental property of visual perception: varying resolutions offer complementary inductive biases, where low-resolution views excel at global semantic recognition and high-resolution views are essential for fine-grained refinement. In this work, we propose Multi-Resolution Fusion (MuRF), a simple yet universally effective strategy to harness this synergy at inference time. Instead of relying on a single view, MuRF constructs a unified representation by processing an image at multiple resolutions through a frozen VFM and fusing the resulting features. The universality of MuRF is its most compelling attribute. It is not tied to a specific architecture, serving instead as a fundamental, training-free enhancement to visual representation. We empirically validate this by applying MuRF to a broad spectrum of critical computer vision tasks across multiple distinct VFM families—primarily DINOv2, but also demonstrating successful generalization to contrastive models like SigLIP2.

Date: March 26, 2026

Projects: https://MuRF-VFM.github.io

Code Repository: https://github.com/orgs/MuRF-VFM

Contact: bochengz@cs.wisc.edu

### 1. Introduction

The landscape of computer vision has been reshaped by standardizing visual representation learning through large-scale Vision Foundation Models (VFMs) (Oquab et al., 2024). These models, pre-trained on vast datasets, provide widely transferable features that serve as the bedrock for numerous downstream applications. A key axis of evolution in this domain has been handling image resolution. While early ViT-based approaches required rigid, fixed-size inputs (Dosovitskiy et al., 2021), recent paradigms like “multi resolution” training (e.g., DINOv2 (Oquab et al., 2024)) have enabled models to process images of arbitrary aspect ratios and scales flexibly.

Despite this flexibility in training, standard inference protocols remain surprisingly rigid. Typically, an input image is resized to a single “optimal” resolution before being processed by the VFM. We argue that this single-scale inference discards a wealth of information inherent in the multi-scale nature of visual data. As shown in Figure 1, it is well-understood that a “division of labor” exists across different viewing scales: lowresolution inputs, by virtue of their larger relative patch sizes, excel at capturing globally coherent semantic context (recognition). High-resolution inputs are indispensable for resolving fine-grained high-frequency details and precise boundaries (refinement). By restricting inference to any single scale, current methods inevitably compromise either global coherence or local precision.

Corresponding author(s): †: Equal Contribution, ∗: Equal Second Author

Input 266 518 784 Input 266 518 784

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- Figure 1: The “Recognition vs. Refinement” Dynamic. The feature map obtained when the input is resized to 266, 518,

784. At lower resolutions, the representation is globally coherent, enabling robust recognition. At higher resolutions, boundary details are sharper, enabling precise refinement, but the object’s interior becomes noisy, risking incomplete segmentation. Our work is motivated by synergizing these two roles.

We posit that unlocking the full potential of pre-trained VFMs requires actively aggregating these complementary views at inference time. To this end, we introduce Multi-Resolution Fusion (MuRF), a general and straightforward strategy designed to create a scale-robust visual representation. Rather than modifying the VFM backbone, MuRF treats it as a frozen feature extractor. It processes the same input image across a pyramid of resolutions and fuses the resulting features into a unified representation. This simple operation effectively synergizes the recognition strengths of low-resolution views with the refinement capabilities of high-resolution views, without requiring expensive multi-scale training of the backbone itself.

The universality of MuRF is its most compelling attribute. It is not tied to a specific architecture or task but serves as a fundamental enhancement to the visual representation. We empirically validate this by applying MuRF to a broad spectrum of critical computer vision tasks, all using frozen VFM backbones:

- • Dense Prediction (Probing): In semantic segmentation and depth estimation, simple linear heads trained on MuRF representations significantly outperform those trained on standard single-scale features, proving that multi-scale fusion enriches basic feature quality.
- • Multimodal Understanding: When used as the visual encoder for Multimodal Large Language Models (MLLMs) in tasks like Visual Question Answering (VQA), MuRF provides LLMs with a more comprehensive visual context, enabling them to reason about both macroscopic scenes and microscopic details simultaneously.
- • Unsupervised Anomaly Detection: In this training-free regime, MuRF effectively resolves the tradeoff between detecting large structural anomalies and tiny surface defects, achieving state-of-the-art performance on demanding benchmarks like MVTec AD 2 TESTpriv,mix (Heckler-Kram et al., 2025).

Our results suggest that effectively leveraging multi-resolution views at inference time is a general principle that consistently improves VFM-based systems. MuRF offers a simple, unified, and highly effective implementation of this principle.

### 2. Related Work

Our work explores unlocking the full potential of frozen VFMs by revisiting and modernizing multi-resolution inference strategies. We review the evolution of VFMs regarding standard input resolutions and discuss historical and contemporary approaches to multi-scale representation learning.

Segmentation

|[Figure 9]|
|---|

Upsample

|[Figure 10]|
|---|

[Figure 11]

VQA

[Figure 12]

|[Figure 13]|
|---|

[Figure 14]

Upsample

|[Figure 15]|
|---|

|[Figure 16]|
|---|

[Figure 17]

##### DINOv2

Visual Grounding

|[Figure 18]|
|---|

[Figure 19]

Merged Features

Depth Estimation

Resized Image

Separate Features

⃛

- Figure 2: Overview of Multi-Resolution Fusion (MuRF). An input image is resized to multiple resolutions and each view is processed by a frozen DINOv2 encoder to produce separate feature maps. These features are upsampled to a shared spatial resolution and fused into a single multi-resolution representation, which can then be used by lightweight task-specific heads for semantic segmentation, depth estimation, visual question answering, visual grounding, and other downstream tasks. Background is removed from the PCA figures.

###### 2.1. Vision Foundation Models and Input Resolution

The field of computer vision has been fundamentally altered by the rise of Vision Foundation Models (VFMs). Initially dominated by Convolutional Neural Networks (CNNs) like ResNet (He et al., 2016) trained via supervised learning, the paradigm shifted toward Vision Transformers (ViTs) (Dosovitskiy et al., 2021) leveraging large-scale self-supervision. Models such as CLIP (Radford et al., 2021) and DINO/DINOv2 (Caron et al., 2021, Oquab et al., 2024) now serve as ubiquitous feature extractors across diverse tasks.

Traditionally, ViTs have been constrained by rigid input resolutions (e.g., 224 × 224) due to their reliance on fixed positional embeddings. This limitation often necessitates aggressive resizing or cropping, potentially discarding critical visual information. Recent advancements, such as DINOv2 (Oquab et al., 2024), NaViT (Dehghani et al., 2023) and FlexViT (Beyer et al., 2023), have introduced multi resolution or native resolution training, allowing models to process images in varying sizes during training.

While multi resolution training enhances model flexibility, standard inference protocols typically revert to utilizing a single, fixed scale, often the highest resolution practically affordable. This single-scale inference overlooks the inherent “division of labor" in visual perception: low-resolution views excel at global semantic recognition, while high-resolution views are necessary for fine-grained refinement. Our Multi-Resolution Fusion (MuRF) strategy addresses this by explicitly aggregating features from multiple resolutions at inference time, ensuring that both global context and local details are preserved without retraining the backbone.

###### 2.2. Multi-Scale Representations

Handling objects of varying scales is a classical challenge in computer vision. Early approaches relied on image pyramids (Adelson et al., 1984), where an image is repeatedly resized and processed. While effective, this was computationally prohibitive for heavy older models. The deep learning era introduced feature pyramids, most notably Feature Pyramid Networks (FPN) (Lin et al., 2017), which efficiently construct multi-scale representations within the network’s forward pass.

In the era of frozen VFMs and Multimodal Large Language Models (MLLMs), explicitly constructing FPNs

often requires costly task-specific training. Consequently, recent trends have shifted back towards input-level manipulations. For handling high-resolution images in MLLMs (e.g., GPT-4V (OpenAI, 2023), LLaVANeXT (Liu et al., 2024b)), a common technique is dividing the image into smaller, fixed-resolution tiles (patches) processed independently, often supplemented by a single low-resolution global view. Similar tiling strategies have been applied in token efficiency for multimodal LLM, such as S2 (Shi et al., 2025).

While tiling allows processing arbitrarily large images, it artificially breaks continuity, making it difficult for models to reason about objects that span splitting boundaries. Furthermore, simple upsampling techniques for VFM features, like FeatUp (Fu et al., 2024) and JAFAR (Couairon et al., 2025), may recover high-frequency details but do not inherently add new information absent from the original single-scale forward pass. What’s more, all of them require certain amount of training, which may lead to generalizability issues. MuRF revisits the classical image pyramid but modernizes it for the VFM era. By fusing multi-resolution views in the feature space, we avoid the boundary artifacts of tiling while capturing a richer, more holistic representation than single-scale upsampling, all in a completely training-free manner for the backbone.

### 3. Method

Our goal is to develop a universal, scale-robust visual representation from a frozen Vision Foundation Model (VFM) that can benefit a wide array of downstream tasks. To this end, we propose Multi-Resolution Fusion (MuRF), a simple yet highly effective strategy for feature extraction at inference time. As illustrated in Figure 2, our method first constructs a rich, multi-scale feature map and then adapts this representation to specific tasks using lightweight, trainable heads.

###### 3.1. Multi-Resolution Feature Fusion

The core of our approach is motivated by the observation that different input resolutions provide complementary visual information: low resolutions capture global context for robust recognition, while high resolutions provide fine-grained detail for precise refinement. MuRF explicitly harnesses this synergy by building a feature pyramid from the input space.

Given an input image x ∈ RH×W×C, we first create an input pyramid by resizing it to a set of different scaling factors, Sres = {s1, s2, . . . , sk}. This yields a collection of images {xs}s∈Sres.

Each resized image xs is then passed through a frozen VFM encoder, which we denote as Φ. We extract the feature map from the encoder (typically the last layer or a specific target block). For each resolution s, we obtain a patch-level feature map:

ℱs = Φ(xs) ∈ RHs×Ws×d (1)

where (Hs,Ws) are the spatial dimensions of the feature map for scale s, and d is the feature (channel) dimension.

To create a single, unified representation, we fuse these individual feature maps. Each map ℱs is first upsampled to a common target spatial resolution (H′,W′), typically the original input size, using bilinear interpolation. These spatially-aligned feature maps are then concatenated along the channel dimension. This process yields the final MuRF representation, ℱMuRF:

ℱMuRF = Concats∈Sres (Upsample(ℱs)) ∈ RH

′×W′×D (2) The total channel dimension D is the sum of the dimensions of the feature maps across all scales, D = ∣Sres∣×d.

This final tensor, ℱMuRF, serves as a powerful, frozen representation that is spatially rich, semantically deep, and robust to scale variations.

Why Channel-wise Concatenation? While one might consider alternative feature fusion operations—such as element-wise addition, mean pooling, or learnable attention—we intentionally select channel-wise concatenation. ViT features are highly localized and scale-dependent semantic tokens. Summation or mean pooling risks destructive interference, blending orthogonal scale-specific activations (e.g., a macroscopic semantic feature with a microscopic edge feature) into an ambiguous representation. By concatenating, we project the features into a higher-dimensional space (D = ∣Sres∣ × d), preserving the strict independence of the “recognition” and “refinement” signals. This allows the lightweight downstream head to adaptively select and route the appropriate scale information without requiring a heavy, parameterized fusion network.

###### 3.2. Task-Specific Adaptation

The ℱMuRF representation is task-agnostic. We adapt it to various downstream tasks by attaching lightweight, task-specific heads. This allows the frozen VFM backbone to be leveraged efficiently across different domains.

Dense Prediction Tasks. For tasks like semantic segmentation and depth estimation, which require a pixel-wise prediction, we attach a simple dense prediction head, Headdense(⋅). This head typically consists of one or more convolutional layers (e.g., a 1 × 1 convolution) that project the D-dimensional feature channels to the desired number of output channels (e.g., number of semantic classes or a single depth value). The final prediction map Yˆ ∈ RH×W×Cout is obtained by:

Yˆ = Upsample(Headdense(ℱMuRF)) (3) During training, only the parameters of Headdense are updated, making the process highly efficient.

Unsupervised Anomaly Detection. This task follows a training-free paradigm. Instead of fusing features into a single tensor, we extend the nearest-neighbor approach. For each resolution s, we construct a dedicated memory bank ℳs from anomaly-free training images. At inference time, we compute a separate anomaly score map Sˆs for each feature map ℱs by calculating the L2 distance of each feature vector to its nearest neighbor in ℳs. The final, robust anomaly score map Sˆ is produced by averaging all individual score maps after upsampling them to the original image dimensions:

Upsample(Sˆs) (4)

1 ∣Sres∣

∑

Sˆ =

s∈Sres

This fusion of scores leverages the strengths of all views for both recognition and refinement.

Multimodal Language Models. In the context of MLLMs for tasks like Visual Question Answering (VQA) and visual grounding, the ℱDINOv2 representation serves as the visual input to the language model. The rich spatial feature map ℱMuRF is first passed through a perception module or projection layer, HeadMLLM(⋅), which maps the visual features into the word embedding space of the LLM.

Evisual = HeadMLLM(ℱMuRF) (5)

These visual embeddings Evisual are then treated as a sequence of “visual tokens” and prepended to the text token embeddings. By providing the LLM with a representation that is rich in both global and local detail, we empower it to answer questions that require reasoning across multiple scales. Only the MLLM’s projection layer and other designated parameters are trained, while the MuRF feature extractor remains frozen.

### 4. Experiments

To validate the universality and effectiveness of our proposed Multi-Resolution Fusion (MuRF) representation, we conduct a comprehensive set of experiments across four diverse and fundamental computer vision tasks: semantic segmentation, depth estimation, visual question answering, and anomaly detection. Our evaluation aims to demonstrate that MuRF consistently improves performance over strong, single-scale baselines, regardless of the downstream application.

- 4.1. Experimental Setup VFM Backbone and MuRF Configuration. Unless otherwise specified, our MuRF configuration is as follows:

- • Backbone: Across all experiments, we use the publicly available, pre-trained DINOv2-ViT-B/14 (Oquab et al., 2024) as our frozen VFM encoder. This ensures a fair comparison and highlights the gains attributable solely to our multi-resolution fusion strategy. We additionally experiment on SigLIP2-Base (Patch16 NaFlex version) (Tschannen et al., 2025) to verify the applicability of our proposed approach.
- • Resolutions (Sres): We are using 5 resolutions for anomaly detection, and 3 resolutions for the rest of tasks including segmentation, depth estimation, and PCA, and 2 resolutions for MLLM. We adhere to the resolution of the original implementation including LLaVA-1.5 (Liu et al., 2024a) and DINOv2 (Oquab et al., 2024).

The final MuRF representation is generated by upsampling all extracted feature maps to a common resolution and concatenating them channel-wise, as described in Section 3.1.

Training Details. For all downstream tasks (segmentation, depth estimation, and VQA), we exclusively train the task-specific heads while keeping both the VFM backbone and the MuRF fusion module frozen. We adhere to the optimization configurations—including optimizer, learning rate, and schedule—established in the respective original papers (Oquab et al., 2024, Liu et al., 2024b). Additional details of the experiment for reproducibility can be found in the Supplementary Material.

###### 4.2. Semantic Segmentation

Setup. We evaluate on the challenging ADE20k (Zhou et al., 2019) and PASCAL VOC (Everingham et al., 2010) benchmark using the standard mean Intersection over Union (mIoU) metric. Our baseline for comparison is a linear probing setup using the same frozen DINOv2 encoder but with features extracted from a single input resolution, which represents the standard inference paradigm. All experiments are performed using DINOv2-ViT-B/14.

Results. Table 1 presents the main results. Our MuRF-based representation provides a significant performance boost over the strong single-scale baseline. Figure 3 qualitatively illustrates this improvement. This

- Table 1: Combined downstream evaluation. Semantic segmentation: performance in mIoU (%) on ADE20K and PASCAL VOC (higher is better). Depth estimation: performance on NYU Depth V2 and SUN RGB-D reporting RMSE (lower is better). MuRF significantly outperforms single-scale baselines, highlighting the benefit of our multi-scale representation for dense prediction and geometric reasoning. Bold indicates the best performance, and underline indicates the second best. "Rel. Improv" measures the relative improvement over original DINOv2 (high resolution for semantic segmentation, and medium resolution for depth estimation)

Semantic Segmentation ↑ Depth Estimation ↓ Method Arch. ADE20K PASCAL VOC NYU Depth V2 SUN RGB-D

lin. 1 lin. 4 lin. 1 lin. 4 OpenCLIP (Cherti et al., 2023) ViT-G/14 39.3 71.4 0.541 0.510 0.537 0.476 MAE (He et al., 2022) ViT-H/14 33.3 67.6 0.517 0.483 0.545 0.523 DINO (Caron et al., 2021) ViT-B/8 31.8 66.4 0.555 0.539 0.553 0.541 iBOT (Zhou et al., 2022) ViT-L/16 44.6 82.3 0.417 0.387 0.447 0.435 Low Resolution ViT-B/14 40.6 71.2 0.423 0.408 0.463 0.439 Medium Resolution ViT-B/14 45.5 78.9 0.389 0.373 0.432 0.416 High Resolution ViT-B/14 46.1 82.5 0.394 0.380 0.445 0.426 MuRF (Ours) ViT-B/14 47.4 83.5 0.361 0.358 0.419 0.407 Rel. Improv - +2.8% +1.2% +7.2% +4.0% +3.0% +2.2%

demonstrates that by combining global context from low resolutions and fine details from high resolutions, MuRF produces a feature map that is inherently better suited for dense prediction. For PASCAL VOC segmentation under the linear probing protocol, MuRF requires approximately 1.3× the training time of the single-resolution counterpart.

- 4.3. Depth Estimation

Setup. We test the in-domain learning capability on NYU Depth V2 (Silberman et al., 2012) dataset and the transfer learning capability on SUN RGB-D dataset (Song et al., 2015). Following standard protocols, we report performance using Root Mean Squared Error (RMSE, lower is better). Our evaluation employs two linear probing configurations: Lin. 1, which utilizes features from the final transformer layer concatenated with the [CLS] token, and Lin. 4, which follows the same protocol but concatenates tokens from layers l = {3,6,9,12}. For our MuRF method, these feature sets are extracted at multiple resolutions and fused. In both cases, we compare against a baseline frozen DINOv2 backbone with a single-scale input.

- Table 2: Computational Cost and Efficiency Analysis on DINOv2-ViT-B/14. We compare the single-scale baselines against our multi-scale MuRF in terms of training latency, downstream head parameter count, and performance (RMSE on NYU Depth V2). MuRF achieves the best performance with a reasonable trade-off in computation.

###### Method Resolution Latency (ms/iter) Head Params (M) VRAM (GB) RMSE ↓

- Single-Scale 0.5× 11.36 0.39 0.36 0.423
- Single-Scale 1.0× 22.61 0.39 0.43 0.389 Single-Scale 1.5× 32.55 0.39 0.54 0.394 MuRF (Ours) {0.5,1.0,1.5}× 58.35 1.18 0.56 0.361

Input

266

518

784

MuRF

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

windowpane

windowpane

windowpane

windowpane

curtain

wall

wall

wall

curtain

wall

curtain

curtain

stove pot

stove pot

stove pot

pot

cabinet table

book

cabinet table

table

book

book

book

table

cabinet

cabinet

chair

chair

chair

chair

floor

floor

floor

floor

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

sky

sky

sky

sky

fountain

fountain

mountain

mountain

mountain

mountain

crt_screen

crt_screen

crt_screen

crt_screen

boat

boat

boat

boat

river water

water

water

water

fountain

(a) Comparison of semantic segmentation results on ADE20K.

Input

140

266

518

MuRF

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

person

person

person

person

chair

diningtable

diningtable

diningtable

diningtable

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

sofa sofa

sofa

sofa

(b) Comparison of semantic segmentation results on PASCAL VOC

- Figure 3: Qualitative comparison of semantic segmentation results on ADE20K (top) and PASCAL VOC (bottom) with different input resolutions. All images are resized to a square shape before being fed into DINOv2, and the subtitle above each image indicates the corresponding input resolution (side length in pixels).

[Figure 40]

Input 0.5x 1.0x 1.5x MuRF

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

(a) Comparison of depth estimation results on NYUd.

[Figure 50]

Input 0.5x 1.0x 1.5x MuRF

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

(b) Comparison of depth estimation results on SUN RGB-D

- Figure 4: Qualitative depth estimation results on NYUd (left) and SUN RGB-D (right). We compare single-scale DINOv2 predictions at 0.5×, 1.0×, and 1.5× input resolutions with our MuRF fusion. By aggregating multi-resolution features, MuRF better preserves global scene structure while sharpening local geometry, producing smoother and more accurate depth maps. Labels 0.X× indicate that the image fed into DINOv2 is resized to 0.X of the original image height and width.

Results. As shown in Table 1 and Figure 4, the model using our MuRF representation achieves substantially lower error rates. This indicates that the fusion of multi-scale features allows the prediction head to better reason about both the overall scene geometry (from low resolutions) and object boundaries (from high resolutions), leading to more accurate depth predictions. We have also measured the performance impact of MuRF, as demonstrated in Table 2.

###### 4.4. Visual Question Answering

Setup. We integrate MuRF into an MLLM framework for VQA. The original vision encoder in LLaVA 1.5 is CLIP, which doesn’t have natural multi-resolution support. To verify how MuRF can support MLLMs, we apply MuRF on multiple variants of LLaVA 1.5, where we keep the same training recipe while changing the vision encoder.

In LLaVA’s DINOv2 variant, we use a single resolution input (336 × 336) in DINOv2 as a baseline. For MuRF, we process resolutions of 224 and 336. In LLaVA’s SigLIP2 variant, we use a single resolution input (384×384) in SigLIP2 as a baseline. For MuRF, we process resolutions of 256 and 384.

For all experiments, to prevent a computational bottleneck in the LLM, we must strictly avoid increasing the token sequence length (all experiments will have 576 visual tokens). Instead of appending multi-resolution

- Table 3: LLaVA-MuRF VQA Performance on MME(Fu et al., 2025), VLMsAreBiased (Bias)(Vo et al., 2026), V∗(Wu and Xie, 2024), MME RealWorld (MR)(Zhang et al., 2025b), RealWorld QA (RW)(xAI, 2024), GQA(Hudson and Manning, 2019), the total score obtained on four subset of mmbench (MMB)(Liu et al., 2025) ("cn_cc", "cn_dev", "en_dev", "ru_dev"), and the accuracy metric of POPE(Li et al., 2023). Bold indicates the best performance.

MME

Vision Encoder Res.

Bias V∗ RW MR GQA MMB POPE

Percept. Cogn. CLIP (official LLaVA 1.5) 336 1511.4 347.1 16.2 50.3 56.1 26.5 62.0 195.4 86.9 DINOv2

336 1291.6 278.6 17.3 38.7 52.3 26.2 62.1 172.4 87.1 224+336 (Ours)

1357.1

366.4

17.7

40.3

62.4

173.1

87.1

53.6

26.1

(+65.5)

(+87.8)

(+0.4)

(+1.6)

(+1.3)

(-0.1)

(+0.3)

(+0.7)

(0.0)

336 1403.4 243.9 15.8 48.7 53.6 31.5 62.2 194.2 86.4 224+336 (Ours)

Clip+DINOv2

62.9

198.8

1471.2

281.4

48.2

56.7

###### 87.4

31.8

16.3

(+67.8)

(+37.5)

(+0.5)

(-0.5)

(+3.1)

(+0.3)

(+0.7)

(+4.6)

(+1.0)

384 1529.3 355.4 19.4 44.0 58.2 33.1 64.1 211.7 87.1 256+384 (Ours)

SigLIP2

###### 19.7

###### 216.9

42.9

86.7

###### 1545.7

###### 371.4

###### 58.4

###### 64.5

###### 33.3

(+16.4)

(+16.0)

(+0.3)

(-1.1)

(+0.2)

(+0.2)

(+0.4)

(+5.2)

(-0.4)

tokens sequentially, features from the low resolution (224 for DINOv2 and 256 for SigLIP2) are spatially up-sampled and concatenated patch-wise along the channel dimension with the high resolution (336 for DINOv2 and 384 for SigLip2) features. The LLM’s visual projector then maps these high-channel-dimension tokens back to the standard LLM hidden dimension. Crucially, this ensures the number of visual tokens fed into the LLM remains exactly the same as the single-resolution baseline, incurring zero extra sequence-length computational cost for the LLM.

Results. The results in Table 3 show that equipping the MLLM with our MuRF representation strongly improves the multimodal understanding capacity regardless of whether DINOv2 or SigLIP2 is used as vision encoder. This suggests that the rich visual context provided by MuRF captures both holistic scene understanding and fine-grained details, empowers the language model to answer a wider range of visual questions more accurately. MuRF maintains comparable VRAM usage, as well as training and inference latency, relative to the single-resolution baseline. The Single-resolution baseline takes 71 minutes to pretrain, and 270 minutes to finetune, while MuRF takes 72 minutes to pretrain, and 274 minutes to finetune.

###### 4.5. Unsupervised Anomaly Detection

Setup. Finally, we validate MuRF in a training-free setting on the MVTec AD 2 (Heckler-Kram et al., 2025) benchmark, a challenging industrial inspection dataset. We use the pixel-level AU-PRO0.05 score as the primary metric and compare against state-of-the-art methods like PatchCore (Roth et al., 2022) and SuperAD (Zhang et al., 2025a).

For this experiment, we use a set of five scaling factors relative to the original image size: {0.3,0.4,0.5,0.6,0.7}.

Results. Table 4 confirms that our method achieves highly competitive, and in several cases state-of-the-art, performance. The ability of MuRF to handle anomalies of vastly different scales—from microscopic scratches to large, structural defects—is a direct result of its multi-resolution design, proving its effectiveness even without any parameter tuning.

Qualitatively, we sampled some results and presented them in Figure 5. The final “Merged" anomaly maps successfully combine the strengths of different input resolutions. Low-resolution views (e.g., 0.3×) are

- Table 4: Anomaly detection performance (AU-PRO0.05 in %) on the MVTec AD 2 dataset. MuRF demonstrates stateof-the-art results on TESTpriv,mix subset, showcasing its robustness in a challenging training-free scenario. "Training" means if the method involves parameter tuning within a neural network. Bold indicates the best performance, and underline indicates the second best.

Method Training? TESTpriv TESTpriv,mix

PatchCore (Roth et al., 2022) ✗ 62.3 52.6 SuperAD (Zhang et al., 2025a) ✗ 61.2 59.3 RoBiS (Li et al., 2025) ✓ 67.3 59.7 MuRF (Ours) ✗ 66.0 62.3 ↑+2.6

Input Ground Truth 0.3x 0.5x 0.7x MuRF

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Input Ground Truth 0.3x 0.5x 0.7x MuRF

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Input Ground Truth 0.3x 0.5x 0.7x MuRF

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Input Ground Truth 0.3x 0.5x 0.7x MuRF

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

- Figure 5: The visualization of anomaly detection on MVTec AD 2

TESTpub dataset. Our merged result (MuRF) successfully combines the robust detection from low-resolution views (e.g., 0.3× correctly identifies the anomaly’s presence but with a coarse mask) and the sharp boundaries from high-resolution views (e.g., 0.7×).

Input 266 518 784 MuRF

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

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

Figure 6: The PCA of feature embedding. It can be seen that lower resolution images tend to have better internal representation, but have worse boundaries due to low resolution. Higher resolution images tend to have sharper boundary but have holes in their internal. MuRF achieves a good balance that generates high-quality representation in both.

adept at robustly identifying the presence of an anomaly, though the resulting detection masks are coarse. Conversely, high-resolution views (e.g., 0.7×) provide sharp and precise boundaries for these anomalies. However, those views contain lots of noise or holes inside large anomaly areas (e.g., in walnuts). By fusing these different views, the MuRF-AD method produces more accurate and complete segmentation masks than single-resolution approaches like SuperAD (Zhang et al., 2025a).

###### 4.6. Qualitative comparison

We present the PCA of DINOv2’s feature embedding on multiple resolutions, as well as our method’s feature embedding.

We have presented the results in Figure 6. It’s easy to see that high resolution images on the right provide sharper, more accurate boundaries. However, this focus on local detail comes at a cost: the model can lose global context, resulting in “holes” within the segmented region, while the low resolution images on the left’s internal is smoother. Our proposed approach MuRF (on the right) achieved a good balance.

###### 4.7. Analysis of Resolution and Feature Concatenation

To show the benefits of MuRF, we conduct a controlled comparison against a standard feature concatenation baseline. To ensure a fair comparison over feature representation, we compare aggregating three distinct resolutions (MuRF) against aggregating three distinct encoder layers (Lin. 3).

The results, presented in Table 5, reveal a complementary relationship between multi-scale and multi-layer features. MuRF achieves superior performance on the in-domain NYU Depth V2 dataset (0.361 vs. 0.376), suggesting that scaling effectively captures the fine-grained structural details required for precise metric depth estimation. Conversely, the multi-layer approach (Lin. 3) performs comparably to MuRF (0.418 vs. 0.419). This aligns with the intuition that intermediate Transformer layers retain robust, generic semantic abstractions.

Crucially, the best performance is achieved by combining both methodologies. This demonstrates that MuRF and multi-layer feature concatenations are not mutually exclusive; rather, they offer orthogonal benefits for complementary performance.

Table 5: Linear probing comparison for depth estimation reporting RMSE. We evaluate on NYU Depth V2 (in-domain) and SUN RGB-D (zero-shot). Lin. 1 utilizes only the final layer. Lin. 3 utilizes layers {4,8,12}. Lower scores indicate better performance. Bold indicates the best performance, underline indicates the second best.

Table 6: Semantic segmentation performance in mIoU (%) when using SigLIP2 as backbone. Bold indicates the best performance, underline indicates the second best.

Resolution ADE20K↑

Method Resolutions Layers NYUd SUN RGB-D

256 32.04 512 35.27 768 34.46

Lin. 1 1.0 12 0.389 0.432 MuRF {0.5,1.0,1.5} 12 0.361 0.419

- Lin. 3 0.5 4,8,12 0.412 0.443
- Lin. 3 1.0 4,8,12 0.376 0.418 Lin. 3 1.5 {4,8,12 0.380 0.428 Lin. 3 + MuRF {0.5,1.0,1.5} {4,8,12} 0.357 0.409

MuRF (Ours) 37.10 ↑+1.83

###### 4.8. Ablation Study

We studied how the number of resolutions fused into MuRF has an impact on the performance on two downstream tasks: depth estimation and anomaly detection.

###### 4.8.1. Depth Estimation

To isolate the contribution of each scale, we conduct an ablation study on the depth estimation task. We evaluate performance on NYUd using the Lin. 1 protocol (reporting RMSE) while varying the specific resolutions used in our fusion. The results in Table 7 compare single-scale baselines against two-scale and our full three-scale MuRF method.

All three single-scale settings are clearly suboptimal compared to any multi-scale fusion: the best single-scale baseline (1.0×) achieves an RMSE of 0.389, whereas all two-scale variants already reduce the error to the 0.373–0.366 range, and the full MuRF configuration further improves it to 0.361. While the medium and high resolutions (1.0× and 1.5×) individually outperform the low-resolution setting (0.5×), including the coarsest view in the fusion remains beneficial. Comparing the higher-resolution pair (1.0×–1.5×, RMSE

Table 7: MuRF ablation results for depth estimation. We compare single-scale baselines and two-scale fusions against the full three-scale MuRF. Performance is measured in RMSE (lower is better). The performance drops (indicated in red) are relative to the Full MuRF setting.

Table 8: Ablation study on Multi-Resolution Settings. Performance measured by AU-PRO0.05 on MV Tec AD 2 TESTpub dataset is reported. We analyze the impact of different resolution combinations. The performance drops are relative to the Full MuRF setting.

Resolutions Setting 0.5× 1.0× 1.5× RMSE ↓ Single Resolution

- 1 ✓ 0.423 ↑0.062
- 2 ✓ 0.389 ↑0.027
- 3 ✓ 0.394 ↑0.033 Fused Resolutions

- 4 ✓ ✓ 0.373 ↑0.011
- 5 ✓ ✓ 0.366 ↑0.005
- 6 ✓ ✓ 0.368 ↑0.007 MuRF ✓ ✓ ✓ 0.361

Resolutions Setting 0.3× 0.4× 0.5× 0.6× 0.7× AU-PRO0.05 ↑ Single Resolution

- 1 ✓ 52.29 ↓5.03
- 2 ✓ 53.42 ↓3.90
- 3 ✓ 55.39 ↓1.93
- 4 ✓ 55.23 ↓2.09
- 5 ✓ 54.87 ↓2.45 Fused Resolutions

- 6 ✓ ✓ ✓ 56.60 ↓0.72
- 7 ✓ ✓ ✓ ✓ 57.29 ↓0.03
- 8 ✓ ✓ ✓ 57.05 ↓0.27 MuRF ✓ ✓ ✓ ✓ ✓ 57.32

0.366) to full MuRF (0.5×–1.0×–1.5×, RMSE 0.361) shows that adding the low-resolution view still yields a measurable gain, even when two relatively strong scales are already present.

This pattern supports our hypothesis that low-resolution inputs provide complementary global geometric context that cannot be fully compensated by higher-resolution views alone. Overall, performance improves monotonically as we increase the number of fused resolutions, with diminishing but consistent returns, confirming that MuRF’s benefits stem from genuine multi-scale complementarity rather than from any single particularly “good” resolution.

###### 4.8.2. Semantic Segmentation

We performed experiments on SigLIP2-Base-Patch16-NaFlex(Tschannen et al., 2025) in addition to DINOv2Base to verify the applicability of our proposed approach. Since SigLIP2 uses patch size of 16 instead of 14, we used the following combination of resolutions: Sres = {256,512,768}. Results are presented in Table 6.

###### 4.8.3. Anomaly Detection

We evaluated different combinations of resolutions to test how the number of resolutions and how the choice of different resolutions have an impact on performance. We used AU-PRO0.05 on the MV Tec AD 2 TESTpub dataset to evaluate the performance.

- As with depth estimation, single-scale models are consistently weaker than their multi-scale counterparts,

- shown in Table 8. The best single resolution is 0.5× (55.39 AU-PRO0.05), while both lower (0.3×, 52.29) and higher (0.7×, 54.87) scales alone underperform, indicating that extremely coarse views lose too much detail and extremely fine views become overly sensitive to local noise. Fusing multiple nearby resolutions mitigates these issues: a three-scale fusion of {0.3,0.5,0.7} already improves performance to 56.60, and adding additional intermediate scales ({0.3,0.4,0.6,0.7}) further raises it to 57.29, nearly matching the full

five-scale MuRF (57.32). Notably, different multi-scale subsets with overlapping ranges (e.g., {0.5,0.6,0.7}) perform similarly well, suggesting that what matters most is covering a spectrum of coarse-to-fine views rather than any particular “magic” resolution.

These trends align with our qualitative findings in Fig. 5: coarse resolutions excel at reliably localizing anomalous regions but yield coarse masks, while finer resolutions sharpen boundaries but may miss parts of large defects. The progressive improvement from single- to multi-resolution settings confirms that MuRF’s anomaly detection gains arise from combining these complementary behaviors across scales, rather than from a specific hand-picked input size.

### 5. Conclusion

In this work, we introduced Multi-Resolution Fusion (MuRF), a simple yet powerful inference-time strategy to enhance Vision Foundation Models representations. By constructing a feature pyramid from the input space and fusing the resulting representations, MuRF effectively synergizes the global context from low-resolution views with the fine-grained detail from high-resolution ones. Our extensive experiments demonstrate that this single, unified approach yields consistent and significant performance gains across a diverse set of fundamental vision tasks, including dense prediction, multimodal reasoning, and unsupervised anomaly detection. These results establish multi-resolution aggregation not as a task-specific trick, but as a general principle for unlocking the full potential of pre-trained visual encoders.

### 6. Acknowledgment

This work was supported in part by NSF IIS2404180, KLA, and the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration), (No. RS-202200187238, Development of Large Korean Language Model Technology for Efficient Pretraining), and (No. RS-2025-2543949. Environment-Aware and Domain-Adaptive Multimodal Embodied AI for Real-World Interaction).

### References

Edward H Adelson, Charles H Anderson, James R Bergen, Peter J Burt, and Joan M Ogden. Pyramid methods in image processing. RCA engineer, 29(6):33–41, 1984.

Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, and Filip Pavetic. Flexivit: One model for all patch sizes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14496–14506, June 2023.

Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4009–4018, June 2021.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9650–9660, October 2021.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2818–2829, June 2023.

Paul Couairon, Loick Chambon, Louis Serrano, Jean-Emmanuel Haugeard, Matthieu Cord, and Nicolas Thome. Jafar: Jack up any feature at any resolution. arXiv preprint arXiv:2506.11136, 2025.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey Gritsenko, Mario Lucic, and Neil Houlsby. Patch n’ pack: Navit, a vision transformer for any aspect ratio and resolution. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 2252–2274. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 06ea400b9b7cfce6428ec27a371632eb-Paper-Conference.pdf.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips.cc/paper_files/paper/2014/file/ 91c56ce4a249fae5419b90cba831e303-Paper.pdf.

Mark Everingham, Luc Gool, Christopher K. Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. Int. J. Comput. Vision, 88(2):303–338, June 2010. ISSN 0920-5691. doi: 10.1007/s11263-009-0275-4. URL https://doi.org/10.1007/s11263-009-0275-4.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. MME: A comprehensive evaluation

benchmark for multimodal large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/ forum?id=DgH9YCsqWm.

Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. Featup: A model-agnostic framework for features at any resolution. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=GkJiNn2QDF.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, June 2022.

Lars Heckler-Kram, Jan-Hendrik Neudeck, Ulla Scheler, Rebecca König, and Carsten Steger. The mvtec ad 2 dataset: Advanced scenarios for unsupervised anomaly detection. arXiv preprint arXiv:2503.21622, 2025.

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535–547, 2019.

Xurui Li, Zhonesheng Jiang, Tingxuan Ai, and Yu Zhou. Robis: Robust binary segmentation for high-resolution industrial images. arXiv preprint arXiv:2505.21152, 2025.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.20. URL https://aclanthology.org/2023.emnlp-main.20/.

Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b. URL https://llava-vl.github.io/ blog/2024-01-30-llava-next/.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 216–233, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-72658-3.

OpenAI. Gpt-4v(ision) system card, 2023.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ radford21a.html.

Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14318–14328, 2022.

Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When do we not need larger vision models? In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, pages 444–462, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-73242-3.

Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Andrew Fitzgibbon, Svetlana Lazebnik, Pietro Perona, Yoichi Sato, and Cordelia Schmid, editors, Computer Vision – ECCV 2012, pages 746–760, Berlin, Heidelberg, 2012. Springer Berlin Heidelberg. ISBN 978-3-642-33715-4.

Shuran Song, Samuel P. Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 567–576, 2015. doi: 10.1109/CVPR.2015.7298655.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, and Daeyoung Kim. Vision language models are biased. In The Fourteenth International Conference on Learning Representations,

#### 2026. URL https://openreview.net/forum?id=DG4S2OlGQA.

Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13084–13094, June 2024.

xAI. Grok-1.5, 2024.

Huaiyuan Zhang, Hang Chen, Yu Cheng, Shunyi Wu, Linghao Sun, Linao Han, Zeyu Shi, and Lei Qi. Superad: A training-free anomaly classification and segmentation method for cvpr 2025 vand 3.0 workshop challenge track 1: Adapt & detect. arXiv preprint arXiv:2505.19750, 2025a.

YiFan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, Liang Wang, and Rong Jin. MME-realworld: Could your multimodal LLM challenge high-resolution real-world scenarios that are difficult for humans? In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum? id=k5VHHgsRbi.

Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3): 302–321, 2019.

Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Image BERT pre-training with online tokenizer. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=ydopy-e6Dg.

### A. Implementation Details

###### A.1. Semantic Segmentation

For semantic segmentation, we follow a consistent strategy with the DINOv2 segmentation protocol. We evaluate our method on both the ADE20K dataset∗ and the PASCAL VOC 2012 dataset. For MuRF , we selected Sres = {266,518,784} for ADE20K dataset, and Sres = {140,266,518} for PASCAL VOC 2012 dataset since images in PASCAL VOC 2012 dataset generally has lower resolution.

All experiments are performed using a frozen DINOv2-Base as backbone. The objective function is the standard pixel-wise Cross-Entropy loss. The segmentation head is optimized using the AdamW optimizer with an initial learning rate η = 1 × 10−3, weight decay λwd = 1 × 10−4, and momentum parameters β1 = 0.9, β2 = 0.999.

For ADE20K dataset training, we applied a joint transformation 𝒯 (x, y) to each RGB input image x and its corresponding segmentation mask y. When data augmentation is enabled, 𝒯 is given by the following composition of operations. We first rescale the image and mask with an isotropic resize operator ℛ such that the smaller spatial dimension of the image does not exceed 784 pixels, while preserving the aspect ratio. From the resized image we sample a random square crop xcrop ∈ Rc×c of size 784 × 784, and extract the corresponding crop of the mask ycrop. With probability p = 0.5 we apply a horizontal flip to both the cropped image and mask. The (possibly flipped) crop is then normalized channel-wise using a fixed mean µ ∈ R3 and standard deviation σ ∈ R3, µ = (123.675255 , 116.280255 , 103.530255 ), σ = (58.395255 , 57.120255 , 57.375255 ) and each RGB channel is transformed as x˜ = x−σµ. We train for a total of 50 epochs with a batch size of 32.

For ADE20K dataset testing, we normalized the image and resize then to the largest possible scale (784). We directly evaluated the output of the model without applying sliding window. The model is evaluated on the original ADE20K validation set.

For PASCAL VOC 2012 dataset training, we utilize the augmented training set combined with the standard training set, which consists of 12,031 images. We apply a composition of geometric and photometric transformations 𝒯 (x) to input images x. First, images are resized such that their shortest spatial dimension is 512 pixels, followed by a random multi-scale resizing operation 𝒮(x, s) where the scale ratio s is sampled uniformly from [0.5,2.0]. From the rescaled image, we extract a random content-aware crop xcrop ∈ R512×512. random sampling strategy. To prevent class imbalance during training, this strategy rejects crops where a single valid category occupies more than 75% of the spatial area. We next apply a horizontal flip with probability p = 0.5, followed by photometric distortions 𝒫(⋅). Specifically, we sequentially apply random brightness shifts (δ ∈ [−32,32]), contrast and saturation scaling (factor ∈ [0.5,1.5]), and hue shifts (δ ∈ [−18,18]), where each transformation is applied with a probability of p = 0.5. Finally, inputs are normalized using the ImageNet mean µ and standard deviation σ. We train for a total of 40,000 iterations with a batch size of 16. We employ a sequential learning rate schedule: a linear warm-up is applied for the first 1,500 iterations (starting from η = 1 × 10−6), followed by a polynomial decay schedule with power p = 1.0 for the remainder of the training.

For PASCAL VOC 2012 dataset testing, we employ a sliding window evaluation strategy to handle varying image aspect ratios. We use a window size of 512 × 512 with a stride of 341 × 341 to generate the final segmentation maps. The model is evaluated on the original PASCAL VOC 2012 validation set.

∗The ADE20K dataset can be obtained at https://data.csail.mit.edu/places/ADEchallenge/ ADEChallengeData2016.zip

The code for semantic segmentation experiments has been uploaded. The semantic segmentation experiment was conducted on a single server equipped with a RTX A6000 GPU.

###### A.2. Depth Estimation

We adopt a training strategy consistent with the DINOv2 depth estimation protocol. The model is trained on the NYU Depth V2 dataset (𝒟train) and evaluated in a zero-shot setting on SUN RGB-D (𝒟test). we selected Sres = {0.5s,1.0s,1.5s}, where s is the original resolution. Training is performed using mixed-precision floating point arithmetic. We minimize the objective using the AdamW optimizer with an initial learning rate η = 1 × 10−4, momentum parameters β1 = 0.9, β2 = 0.999, and weight decay λwd = 0.01. The training spans 38,400 iterations with a batch size of 2. We employ a cosine annealing schedule, decaying η to ηmin = 1×10−6, and a linear warm-up period of 1,000 iterations is applied to stabilize early-stage convergence.

We test two feature extraction configurations on the frozen DINOv2-Base encoder Φ: a single-layer setup (Lin. 1) and a multi-layer fusion setup (Lin. 4). In the Lin. 1 configuration, features are extracted solely from the final transformer layer. In the Lin. 4 configuration, we designate a set of intermediate layers Slayer = {3,6,9,12}. For both configurations, for every utilized layer l, we concatenate the layer-specific global classification token [CLS]l to each spatial patch token. In the multi-layer case, the final representation is obtained by concatenating these upsampled features across all l ∈ Slayer along the channel dimension before passing them to the decoding head.

Following the methodology of Bhat et al. (2021), the decoding head treats depth estimation as a per-pixel classification task. We divide the depth prediction range into 256 uniformly distributed bins over a depth range of [1 × 10−3,10].

We apply a composition of geometric and photometric transformations 𝒯 (x) to input images x ∈ RH×W×C. We first apply the standard NYU Eigen crop (Eigen et al., 2014) to remove invalid border regions. We then apply a random rotation R(θ) with zero-padding, where θ ∼ 𝒰(−2.5◦,2.5◦) with probability p = 0.5, followed by a random horizontal flip with p = 0.5. Subsequently, we extract a random crop xcrop ∈ R416×544 from the transformed image. We also employ photometric distortions with probability p = 0.5, specifically color jittering defined by random adjustments to gamma γ ∈ [0.9,1.1], brightness β ∈ [0.75,1.25], and per-channel RGB scaling sc ∈ [0.9,1.1]. Finally, all inputs are normalized using the ImageNet mean µ and standard deviation σ.

The network parameters are optimized by minimizing a composite loss function ℒtotal, consisting of a scaleinvariant log loss ℒdepth with a warm-up schedule (Bhat et al., 2021), and a multi-scale gradient matching term ℒgrad:

ℒtotal = λdepthℒdepth(dˆ, d) + λgradℒgrad(dˆ, d) (6)

where dˆ and d denote the predicted and ground-truth depth maps, respectively. The term ℒgrad enforces gradient consistency across four spatial scales to preserve structural details. We set the balancing coefficients

to λdepth = 1.0 and λgrad = 0.5.

During evaluation, we process inputs at their native resolution (480 × 640). To further improve predictive stability, we employ Test-Time Augmentation (TTA). The final prediction dˆfinal is computed as the average of the direct prediction and the inverse-transformed prediction of the horizontally flipped input:

- 1

- 2 (ℳ(x) + flip−1(ℳ(flip(x)))) (7)

dˆfinal =

The depth estimation experiment was conducted on a single server equipped with a RTX A6000 GPU.

Input

266

518

784

MuRF

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

chandelier

lamp

lamp

chandelier

wall

wall

wall

windowpane

blind

mirror

mirror

mirror

wall

mirror

buffet

buffet

blind countertop

windowpane

door

door

door

door

countertop

countertop

sink

sink

sink

cabinet

cabinet

cabinet

cabinet

fireplace

fireplace

fireplace

fireplace

screen_door

floor

floor

floor

floor

tank

tank

tank

tank

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

sky

sky

sky

sky

building

building

building

building

tree

grass

path

grass

grass

grass

path sidewalk

sidewalk

path

road

sidewalk

sidewalk

- (a) Comparison of semantic segmentation results on ADE20K. Input

[Figure 114]

[Figure 115]

horse

140

[Figure 116]

horse

266

[Figure 117]

horse

518

[Figure 118]

horse

MuRF

[Figure 119]

[Figure 120]

train

[Figure 121]

train train train

[Figure 122]

[Figure 123]

- (b) Comparison of semantic segmentation results on PASCAL VOC

- Figure 7: Additional segmentation visualizations from ADE20K and PASCAL VOC.

###### A.3. LLaVA-style MLLM training

To verify how MuRF can support MLLMs, we replace the vision encoder of LLaVA 1.5 with DINOv2 and SigLIP2.

For the DINOv2 variant’s baseline, we resize all images into squares of 336 × 336, and put them into DINOv2. For MuRF, we used Sres = {224,336}, and after that, we resize the feature embedding generated by images of size 224 × 224 to 24 × 24 (the same as the feature embedding generated by images of size 336 × 336) using bilinear interpolation.

For the SigLIP2 variant’s baseline, we resize all images into squares of 384 × 384, and put them into SigLIP2. We choose 384 because SigLIP2 has a patch size of 16 and 384 × 384 image can result in 576 tokens, which is the same for the DINOv2 variant and the same with the original Clip variant. For MuRF, we used Sres = {256,384}, and after that, we resize the feature embedding generated by images of size 256×256 to 24×24 (the same as the feature embedding generated by images of size 384×384) using bilinear interpolation.

After that, we perform the token-wise concatenation for all two features embedding, so the length per visual token becomes two times longer but the number of total number of visual token remains the same. We changed the input dimension of the two-layer multi-projection layer accordingly, but keep the output dimension the same. Therefore, the number of tokens inputted into the LLM is the same.

We followed the LLaVA 1.5 two-stage training pipeline with the same hyperparameter.

Input 0.5x 1.0x 1.5x MuRF

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

- (a) Additional comparison of depth estimation results on NYUd. Input x0.5 x1.0 x1.5 MuRF

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

- (b) Additional comparison of depth estimation results on SUN RGB-D.

- Figure 8: Additional visualizations of depth estimation results on NYUd and SUN RGB-D.

All experiments were conducted on a single server equipped with eight NVIDIA H100 GPUs.

###### A.4. Anomaly Detection

For anomaly detection, we followed the Embedding-based Anomaly Detection Paradigm as Roth et al. (2022), Zhang et al. (2025a).

Specifically, for a given input image x ∈ RH×W×C, we first generate a set of resized versions {xs}s∈Sres, where Sres is a predefined set of scaling factors. Each version xs is then processed by a frozen DINOv2-Base encoder, which we denote as Φ.

To further enrich the representation with hierarchical semantic information, we also extract features from multiple intermediate layers of the encoder, Slayer. This process yields a comprehensive collection of patchlevel feature maps {ℱl,s ∣ l ∈ Slayer, s ∈ Sres}, where each feature map is defined as:

ℱl,s = Φl(xs) ∈ RHl,s×Wl,s×d (8)

Here, Φl represents the feature output at layer l, (Hl,s,Wl,s) are the spatial dimensions of the feature map for scale s, and d is the feature dimension. Each feature map ℱl,s thus captures a unique combination of spatial scale and semantic level.

- At inference time, for a given test image x∗, we extract its feature maps {ℱl∗,s}. For each feature map ℱl∗,s, we compute a corresponding anomaly score map Sˆl,s by scoring each of its feature vectors. The score for a

Input 266 518 784 MuRF Input 266 518 784 MuRF

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

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

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

- Figure 9: Additional PCA visualization.

feature vector f∗ ∈ ℱl∗,s is its L2 distance to the nearest neighbor in the dedicated memory bank ℳl,s: S(f∗) = min

∥f∗ − f∥2 (9)

f∈ℳl,s

Finally, to produce the single output anomaly score map Sˆ, we fuse all individual score maps {Sˆl,s}. Each map is first up-sampled to the original image dimensions (H,W) via bilinear interpolation, and then aggregated through element-wise averaging:

Upsample(Sˆl,s) (10)

1 ∣Slayer∣∣Sres∣

∑

∑

Sˆ =

s∈Sres

l∈Slayer

To streamline the process in different resolution through a standard scaling factor, we preprocess images in different categories of the MVTec AD 2 dataset by resizing them to comparable total number of pixels, as

- shown in Table 9.

Table 9: Resolution conversion table for MVTec AD v2 categories to standardize the total number of pixels for our experiments.

Object Original Resolution # of Pixels New Resolution New # of Pixels

Can 1024x2232 2,285,568 1536x3348 5,142,528 Fruit Jelly 1520x2100 3,192,000 1900x2625 4,987,500 Vial 1900x1400 2,660,000 2470x1820 4,495,400 (... other categories with no change are omitted for brevity ...)

For all experiments, we used features from layer 7, layer 9 and layer 11, and features from resolution 0.3, 0.4, 0.5, 0.6, 0.7. The image is resized to the nearest multiple of patch size (14) before entering the model.

We use the IndexIVFFlat approximate nearest neighbor search algorithm from Faiss Johnson et al. (2019). For all experiments, we used nlist = 512 and nprobe = 32.

All experiments were conducted on a single server equipped with an NVIDIA A100-SXM4-80GB GPU and two AMD EPYC 7713 64-Core Processors. A full run of our method on a single MVTec AD v2 category takes less than 3 hours.

### B. Additional Visualization

We provided additional visualization for Semantic Segmentation (Figure 7), Depth Estimation (Figure 8) and PCA Qualitative comparison (Figure 9).

