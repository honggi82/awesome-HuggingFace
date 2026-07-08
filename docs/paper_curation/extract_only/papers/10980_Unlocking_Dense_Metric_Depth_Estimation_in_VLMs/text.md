## Unlocking Dense Metric Depth Estimation in VLMs

Hanxun Yu1,2∗† Xuan Qu1,2† Yuxin Wang2,3 Jianke Zhu1,4 Lei Ke2 1Zhejiang University 2Tencent Hunyuan LLM 3HKUST 4Shenzhen Loop Area Institute

Project Page: https://depthvlm.github.io/

# arXiv:2605.15876v3[cs.CV]20May2026

###### Dense Geometry Prediction 3D Spatial Reasoning

###### User Input

[Figure 1]

depth (m)

[Figure 2]

- Q1: Given this image,

generate its metric depth map (in meters).

- Q2: Which one is

4.82

[Figure 3]

[Figure 4]

DepthMapPointCloud

4.07

3.31

m

m 3.

2.56

06

[Figure 5]

1.38

1.81

1.05

[Figure 6]

[Figure 7]

: The black monitor is closer.

Inference Time

(a 256×192 input image)

[Figure 8]

DepthLM

13h

| |
|---|

Youtu-VL 2.48s

| |
|---|

closer to me? The black monitor or the office chair?

[Figure 9]

: Here is the metric depth map.

Ours

0.42s

| |
|---|

Figure 1: Our method serves as a unified foundation model for both low-level dense geometry prediction and high-level multimodal understanding, while achieving substantially faster inference compared with existing VLM-based approaches such as DepthLM [9] and Youtu-VL [64].

[Figure 10]

🤖

[Figure 11]

👤

Frame_107

Faster Speed

### Abstract

Vision–Language Models (VLMs) excel at 2D tasks such as grounding and captioning, yet remain limited in 3D understanding. A key limitation is their text-only supervision paradigm, which under-constrains fine-grained visual perception and prevents the recovery of dense geometry. Prior methods either distill geometry from external vision models, introducing error accumulation, or enable direct prediction with inefficient per-pixel query or coarse token-level outputs. In this paper, we propose DepthVLM, a simple yet effective framework that transforms a single VLM into a native dense geometry predictor while preserving its multimodal capability. By attaching a lightweight depth head to the LLM backbone and training under a unified vision–text supervision paradigm with a two-stage schedule, DepthVLM generates full-resolution depth maps alongside language outputs in a single forward pass. We further introduce a unified indoor–outdoor metric depth benchmark in a VLM-compatible format. Experiments show that DepthVLM significantly outperforms existing VLMs with higher inference efficiency, surpasses leading pure vision models, and improves complex 3D spatial reasoning, moving toward a truly unified multimodal foundation model.

*Work done during an internship at Tencent Hunyuan LLM. †Equal contribution.

Preprint.

Teaser_v1

[Figure 12]

Text-Only Supervision

Unified Vision-Text Supervision

MetricDepth

0.533

ZoeDepth

[Figure 13]

Models VLMs

Depth Head

0.546

Depth Pro

0.812

Metric3Dv2

0.823

UniDepthV2

[Figure 14]

[Figure 15]

Large Language Model

Large Language Model

0.877

DepthAnythingV3

0.201

InternVL3.5-38B

0.405

GPT-5.5

Vision Encoder

Vision Encoder

0.609

Youtu-VL-4B

0.706

DepthLM-12B

0.884

Ours-4B

Image Text

Image

Text

0.0 0.2 0.4 0.6 0.8 1.0

(c) Accuracy 𝛿1(↑) over 5 datasets (Waymo, NuScenes, ETH3D, sunRGBD, IBims-1).

(a) Prevailing VLMs (b) DepthVLM (ours)

- Figure 2: Comparison of prevailing VLMs with our method. (a) Prevailing VLMs are typically supervised solely in the text space, leaving dense 3D geometry out of reach. (b) DepthVLM introduces a unified vision–text supervision paradigm by integrating a lightweight depth head, natively enabling a single VLM backbone to generate dense geometry alongside language responses. (c) While even advanced VLMs such as GPT-5.5 [41] struggle to infer 3D structure from 2D inputs, our model significantly outperforms prior VLMs and even surpasses leading specialized pure vision models.

### 1 Introduction

With the rapid advancement of Large Language Models (LLMs) [15, 35, 54, 71], growing efforts have extended them beyond pure text understanding, giving rise to Vision-Language Models (VLMs) [30, 78, 81] that tackle diverse multimodal tasks. Despite strong performance on 2D tasks such as visual reasoning and image captioning, current VLMs remain limited in complex 3D understanding [12, 40, 42, 72], which is crucial for applications like AR/VR, autonomous driving, and embodied robotics.

A fundamental limitation of prevailing VLMs is their text-only supervision paradigm: visual signals are consumed only as inputs, while outputs are generated as autoregressive text. This design inherently under-constrains fine-grained visual perception and prevents explicit modeling of dense scene geometry, as shown in Figure 2(a). To address this, prior works [18, 66, 84] inject geometric signals (e.g., depth maps or point clouds) from pretrained 3D models to augment VLMs, but such pipelines rely on knowledge distillation from external vision experts and inevitably suffer from error accumulation. More recent works [25, 69, 70] instead explore direct geometric prediction from RGB inputs within VLMs. DepthLM [9] first demonstrates that VLMs can match pure vision models on metric depth estimation, but its single-pixel query per inference makes dense prediction prohibitively slow, while its text-heavy supervision substantially degrades the VLM’s general VQA capability. Youtu-VL [64] further enables full-image depth prediction in one pass, yet its token-level outputs remain coarse and require post-hoc interpolation for pixel-level detail. Moreover, its from-scratch training recipe demands massive data and compute, limiting direct adaptation to existing VLMs.

These observations raise a natural question: can a VLM serve as a native dense geometry predictor with minimal architectural change, while preserving its general multimodal capability? Focusing on dense metric depth estimation, a fundamental task in 3D understanding, we propose DepthVLM, a simple yet effective framework that enables a single VLM backbone to jointly generate dense pixel-level depth maps and language responses. As shown in Figure 2(b), we attach a lightweight depth head to the LLM backbone, taking processed visual tokens as input, and fine-tune the model under a unified vision–text supervision paradigm. In a single forward pass, DepthVLM predicts fullimage depth for all pixels without post-processing, reducing DepthLM’s O(HW) inference cost to O(1). Moreover, unlike fixed-resolution vision models [57], DepthVLM inherits the native-resolution flexibility of VLMs and can be seamlessly integrated into the standard instruction tuning stage.

Since extending VLMs to other tasks often degrades their general multimodal capability [16, 83], we adopt a two-stage training strategy: Stage-1 trains only the added depth head to establish initial depth prediction ability, and Stage-2 fine-tunes the full model end-to-end. We further introduce DepthVLMBench, a unified benchmark that aggregates public indoor and outdoor depth datasets into a VLMcompatible format, enabling both effective training and fair comparison with pure vision models. Interestingly, we find that equipping VLMs with dense geometry prediction improves downstream 3D spatial reasoning performance, further highlighting the value of a unified foundation model that jointly excels at low-level dense geometry prediction and high-level multimodal understanding.

In summary, our contributions are threefold:

- • We find that a VLM can serve as a native dense geometry predictor and propose a lightweight recipe that yields a unified foundation model for both dense geometry generation and multimodal interaction, seamlessly compatible with the standard instruction-tuning stage.
- • We devise a two-stage training strategy that preserves the VLM’s original multimodal capability, and present DepthVLM-Bench, a unified indoor–outdoor benchmark that enables VLM training and direct comparison with pure vision models on metric depth estimation.
- • Extensive experiments across diverse datasets show that DepthVLM significantly outperforms existing VLMs with higher inference efficiency, surpasses state-of-the-art pure vision models on metric depth estimation, and further improves 3D spatial reasoning performance.

### 2 Related Work

#### 2.1 Dense Metric Depth Estimation

Dense metric depth estimation aims to recover per-pixel absolute depth values from RGB images, which is fundamental for 3D scene understanding. Early methods [4, 17] rely on single-domain supervision, producing models specialized to either indoor rooms [50] or outdoor scenes [20] with limited cross-domain generalization. To improve robustness, MiDaS [47] and DPT [46] introduce affine-invariant prediction across diverse datasets, but only provide relative depth without metric scale. To resolve scale ambiguity, ZoeDepth [5] combines relative and metric depth via domainspecific heads, while Metric3D [77, 24] unifies inputs in a canonical camera space. More recently, UniDepth [43, 42] jointly estimates depth and camera intrinsics in a self-promptable manner, and DepthAnything [73, 74, 34] leverages large-scale synthetic supervision for zero-shot generalization. Despite their strong geometric accuracy, these pure vision models focus solely on low-level geometric prediction and lack high-level language interaction, limiting their applicability to 3D reasoning tasks.

#### 2.2 VLMs for 3D Spatial Understanding

Spatial-Enhanced VLMs. To bridge the gap between 2D semantics and 3D spatial intelligence, a line of research augments VLMs [3, 82, 59] with external geometric signals. One direction [23, 13, 85, 79, 86, 27, 26, 44, 61] directly feeds explicit 3D data (e.g., point clouds, voxels, or depth maps) from sensors into LLMs via projectors. While effective on 3D VQA benchmarks [2, 39], these methods rely on sparse and costly 3D data and are largely limited to indoor scenes. Another direction elicits spatial reasoning purely from 2D inputs. SpatialVLM [11] and SpatialRGPT [14] convert vision outputs into textual supervision, while Ross3D [56] introduces multi-view reconstruction as an auxiliary objective. More recent works [66, 18, 84, 28, 68] further distill geometric priors from 3D reconstruction [57, 60, 58] or video diffusion models [55, 6] into VLMs to improve spatial reasoning. However, these methods rely on external vision experts, making them prone to error accumulation, and are still limited to textual outputs without enabling dense, pixel-level geometry prediction.

Geometry-Generative VLMs. Recent studies [70] instead treat the VLM as a unified foundation model that directly generates dense geometry from RGB inputs. Multi-SpatialMLLM [69] and Seed1.5-VL [22] explore pixel-level metric depth estimation while lagging behind pure vision models. G2VLM [25] adopts a Mixture-of-Experts architecture for unified modeling, yet focuses on relative depth. DepthLM [9] matches advanced vision models in accuracy, but predicts only one pixel per inference and its text-heavy supervision severely degrades general performance. Youtu-VL [64] enables full-image depth prediction in one pass, but produces coarse token-level outputs and relies on costly from-scratch training. In contrast, our method lightweightly equips existing VLMs with dense metric depth estimation while preserving their general capability. Inheriting native-resolution processing, it enables flexible inputs and can be seamlessly integrated into standard instruction tuning.

### 3 Methodology

Our goal is to develop a unified foundation model that natively supports both low-level dense geometry prediction and high-level multimodal understanding within a single VLM backbone. As illustrated in Figure 3, we (i) augment the standard VLM with a lightweight DPT-style [46] depth head to jointly

Pipeline_v2

Depth Prediction Head

[Figure 16]

Input Image (𝑰) Multi-Layer Visual Features in ViT

Training Stage 1 + Training Stage 2

[Figure 17]

[Figure 18]

𝐻LLMℳ𝓋

[Figure 19]

[Figure 20]

DepthPrediction Head

[Figure 21]

Reassemble32

Fusion

[Figure 22]

Vision Encoder

𝜙 ℎ ℓ

LargeLanguageModel

Reassemble16

Fusion

…

…

|[Figure 23]|
|---|

𝜙 ℎ ℓ

Reassemble8

Fusion

- F1 Fusion

F4

F3

- F2

Metric Depth Map ( 𝑫)

unify focal length

Input Tokens Generated Tokens

𝜙 ℎ ℓ

Reassemble4

TextTokenizer

Text Prompt (𝑻) Text Response ( 𝑻)

TextDecoding

A dense per-pixel metric depth map (in meters) has been generated for the input image.

Given this image, estimate the metric depth (in meters) for every pixel of the image.

Multi-Layer Features in ViT

Deep

Head

Shallow

[Figure 24]

LLM Output Visual Feature

- Figure 3: Overview of our proposed DepthVLM. We extend the standard VLM architecture with a lightweight DPT-style [46] depth prediction head, and adopt a two-stage training strategy to preserve the backbone’s general VQA capability. In addition, input images are normalized to a unified focal length, eliminating camera-induced ambiguity across heterogeneous dataset domains.

produce dense metric depth map and language responses; (ii) employ a two-stage training strategy to preserve the VLM’s inherent multimodal capability; and (iii) leverage a multi-source training corpus together with focal-length normalization to mitigate camera-induced ambiguity across heterogeneous sensors, yielding strong cross-dataset generalization.

#### 3.1 Model Architecture

Preliminaries. A standard VLM comprises three components: a vision encoder Ev that tokenizes an input image I ∈ R3×H×W into Nv vision tokens, a projector ϕ that maps them into the LLM embedding space, and an autoregressive language model FLLM that processes the joint multimodal sequence to generate text. Given an image I and a text prompt T, the VLM produces hidden states as

HLLM = FLLM([ϕ(Ev(I)); T ]) ∈ R(N

v+Nt)×d. (1)

Motivation: VLM as a Native Dense Predictor. Prior works on 2D dense understanding [67] typically augment the VLM with region-level encoders [48] or task-specific tokens [53], inevitably fragmenting the architecture and complicating the training and inference pipelines. Inspired by recent

- 3D foundation models [57, 60] that derive dense geometry directly from transformer tokens, we instead ask: is a standard VLM already a dense predictor? We answer this affirmatively by showing that dense geometry can be decoded directly from the VLM’s own vision tokens using a lightweight DPT-style [46] head over multi-scale visual features, without altering its text generation pathway.

Unified Architecture for Dense Geometry. A key observation is that the vision encoder Ev naturally provides a hierarchy of representations—from low-level appearance cues in shallow layers to highlevel semantics in deeper layers—that inherently form a multi-scale pyramid well suited for dense prediction. Let {h(ℓ)}L

ℓ=1 denote the per-layer hidden states of the ViT and HLLM the last-layer hidden states of the LLM. We extract four feature maps from the VLM: three intermediate ViT layers {ℓ1,ℓ2,ℓ3} together with the LLM’s final hidden states at image-token positions:

v

v×d, k = 1,2,3, (ViT intermediate layers) HLLMM

k) ∈ RN

ϕ h(ℓ

(2)

Fk =

v×d, k = 4, (LLM final layer)

∈ RN

v

where Mv selects LLM hidden states at image-token positions. F1,2,3 capture purely visual features with increasing abstraction, while F4 encodes vision-language contextualized representations.

Unlike the original DPT [46] that operates on native ViT features, visual tokens in a VLM are already downsampled by the patch merger [3]. We therefore avoid additional downsampling and instead construct a bottom-up pyramid via upsampling, assigning higher spatial resolution to earlier ViT layers. Specifically, each Fk is projected with a 1×1 convolution and resampled to a layer-specific resolution, yielding finer spatial details for shallower features. The resulting multi-scale features are fused with RefineNet blocks [33] and decoded into a dense metric depth map at the input resolution:

Dˆ = DPT(F1,F2,F3,F4) ∈ RH×W, D, ˆ Tˆ = DepthVLM(I,T), (3)

where a final Softplus activation ensures strictly positive depth values. In this way, our model jointly generates dense metric geometry Dˆ and text response Tˆ within a unified foundation model.

#### 3.2 Two-Stage Training Strategy

To introduce dense geometry prediction while preserving the original multimodal understanding, we adopt a two-stage training strategy. In the first stage, we train only the depth head to initialize dense depth prediction capability. In the second stage, we unfreeze the LLM backbone and fine-tune the model end-to-end, enabling tighter integration of geometric prediction with multimodal reasoning.

- Stage-1: Depth Head-Only Training. Since the introduced depth head is randomly initialized, directly training it with the VLM can lead to noisy gradients that may disrupt pretrained knowledge. We therefore freeze the entire VLM and train only the depth head. Following standard practice [24, 74], we supervise the predicted depth map Dˆ using the scale-invariant logarithmic (SILog) loss [17]:

Ldepth =

1 |Ω| i∈Ω

di2 − λ

1 |Ω| i∈Ω

di

2

, di = log Dˆi − log Di∗, (4)

where Ω denotes pixels with valid ground-truth depth D∗ and λ provides a balanced inductive bias, preserving metric supervision while reducing sensitivity to dataset-specific scale variations.

- Stage-2: End-to-End Fine-Tuning. To further strengthen geometric prediction in synergy with the VLM’s inherent language interaction capability, we unfreeze the LLM backbone and perform end-to-end fine-tuning on a mixture of instruction-following data. The overall objective is a weighted combination of the autoregressive language modeling loss and the depth loss defined in Stage-1:

Ljoint = Ltext + α Ldepth, Ltext = −

t

log pθ T ˆt T ˆ<t, I, T , (5)

where Ltext is the standard cross-entropy loss over response tokens and α balances the two objectives.

#### 3.3 Mixed-Source Data Curation

Eliminating Camera Ambiguity. Joint training across datasets suffers from camera-induced scale ambiguity in metric depth estimation. Images with different focal lengths can depict similar scenes but correspond to inconsistent metric depths, leading to conflicting supervision and poor generalization. We address this by adopting focal-length normalization following prior works [9, 43], rescaling all images to a unified focal length fc to remove dataset-specific biases and enforce consistent pixel-to-metric mapping. Formally, given an image I with focal length f and depth map D, we apply:

s = fc / f, I˜ = Rs(I), D˜ = Rs(D), (6)

where Rs(·) denotes isotropic bilinear resizing. After normalization, all samples are aligned to a virtual camera with focal length fc. This removes cross-dataset scale discrepancies and enables the model to learn a focal-invariant mapping that generalizes well to open-world images.

DepthVLM-Bench. We assemble a diverse set of widely used public datasets for metric depth estimation into a unified benchmark that supports training VLMs for dense geometry prediction and enables direct comparison with pure vision models under a consistent protocol.

Training split. We mix the training set of 8 datasets covering indoor and outdoor scenes. For indoor data, we use ScanNet++ [76], Taskonomy [80], HM3D [45], and Matterport3D [10]; for outdoor data, we use Argoverse2 [65], Waymo [52], DDAD [21], and NuScenes [8]. In contrast to pure vision models [7, 34], which often rely on more than 20 datasets with extensive synthetic data, our model achieves comparable performance with an order of magnitude less data.

Evaluation split. We evaluate on 9 datasets across domains, all disjoint from the training set: 4 indoor (ScanNet++, sunRGBD [51], IBims-1 [31], NYUv2 [50]), 4 outdoor (Argoverse2, Waymo, DDAD, NuScenes), and ETH3D [49] containing both indoor and outdoor scenes. For each dataset, we sample 1k images and 10 pixels per image (10k pixels total), oversampling smaller datasets when needed.

- Table 1: Comparison with existing VLMs on metric depth estimation across diverse indoor and outdoor datasets. For VLMs not explicitly trained for this task, we adopt the prompting strategy proposed in DepthLM [9] to elicit their best performance. Even the state-of-the-art GPT-5.5 [41]

attains a δ1 of only around 0.4, highlighting the difficulty of the task for prevailing VLMs. Bold and underlined values denote the best and second-best results, respectively.

δ1(↑) of various methods

Outdoor Out+In Indoor

Avg. Argoverse2 Waymo DDAD NuScenes ETH3D ScanNet++sunRGBD IBims-1 NYUv2

Naive Prediction with Constant Answers Always Output 2.0m 0.002 0.004 0.002 0.010 0.112 0.261 0.380 0.269 0.373 0.157 General-Purpose VLMs

GPT-4o [29] 0.141 0.139 0.193 0.174 0.296 0.369 0.358 0.393 0.394 0.273 GPT-5.5 [41] 0.378 0.368 0.304 0.276 0.369 0.432 0.527 0.483 0.525 0.407 Qwen3-VL-4B [3] 0.033 0.006 0.086 0.040 0.085 0.208 0.220 0.037 0.155 0.097 Qwen3-VL-8B [3] 0.119 0.061 0.243 0.118 0.175 0.274 0.286 0.081 0.194 0.172 Qwen3-VL-32B [3] 0.029 0.017 0.138 0.049 0.150 0.434 0.536 0.105 0.435 0.210 InternVL3.5-8B [59] 0.139 0.110 0.177 0.111 0.185 0.395 0.459 0.214 0.431 0.247 InternVL3.5-14B [59] 0.086 0.045 0.128 0.084 0.183 0.390 0.440 0.235 0.445 0.226 InternVL3.5-38B [59] 0.131 0.129 0.219 0.110 0.181 0.400 0.431 0.155 0.423 0.242

Spatial-Enhanced VLMs

SpaceLLaVA-13B [11] 0.006 0.002 0.001 0.006 0.107 0.050 0.044 0.172 0.087 0.053 SpatialRGPT-8B [14] 0.045 0.064 0.096 0.116 0.133 0.124 0.084 0.044 0.070 0.086 Cambrian-S-7B [75] 0.006 0.019 0.038 0.033 0.069 0.145 0.073 0.057 0.063 0.056

VLMs Trained on Metric Depth Estimation

Youtu-VL-4B [64] 0.663 0.473 0.342 0.698 0.286 0.522 0.734 0.856 0.849 0.603 DepthLM-12B [9] 0.761 0.588 0.654 0.736 0.666 0.756 0.785 0.754 0.866 0.730 Ours-4B 0.810 0.879 0.818 0.821 0.924 0.861 0.882 0.912 0.908 0.868 Ours-8B 0.798 0.865 0.813 0.831 0.928 0.901 0.889 0.936 0.920 0.876

- Table 2: Comparison with specialized pure vision models on metric depth estimation across indoor and outdoor datasets. Despite being a unified VLM that preserves strong multimodal capabilities, our method can outperform state-of-the-art pure vision specialists, demonstrating that dense geometry prediction can emerge natively within a single vision-language foundation model.

Outdoor Out+In Indoor

Avg. Waymo NuScenes ETH3D sunRGBD IBims-1

δ1(↑) of various methods

ZoeDepth [5] 0.639 0.196 0.345 0.769 0.718 0.533 Depth Pro [7] 0.255 0.389 0.355 0.852 0.880 0.546 Metric3D [77] 0.879 0.721 0.373 0.222 0.796 0.598 Metric3Dv2 [24] 0.923 0.747 0.851 0.813 0.726 0.812 UniDepth [43] 0.670 0.858 0.149 0.907 0.158 0.548 UniDepthV2 [42] 0.730 0.872 0.657 0.911 0.941 0.823 DepthAnything [73] 0.739 0.205 0.277 0.847 0.854 0.584 DepthAnythingV2 [74] 0.715 0.168 0.111 0.697 0.887 0.516 DepthAnythingV3 [34] 0.885 0.790 0.843 0.913 0.955 0.877

Ours-4B 0.879 0.821 0.924 0.882 0.912 0.884 Ours-8B 0.865 0.831 0.928 0.889 0.936 0.890

### 4 Experiment

#### 4.1 Experimental Settings

Baselines and Metrics. We compare our model against VLMs and pure vision models. Baselines include four groups: (i) general-purpose VLMs: Qwen3-VL [3], InternVL3.5 [59], GPT-4o [29], GPT-

##### 5.5 [41]; (ii) spatially-enhanced VLMs: SpaceLLaVA-13B [11], SpatialRGPT-8B [14], Cambrian-S-

- 7B [75]; (iii) depth-specialized VLMs: Youtu-VL-4B [64], DepthLM-12B [9]; and (iv) pure vision models: ZoeDepth [5], Depth Pro [7], UniDepth [43, 42], Metric3D [77, 24], DepthAnything [73, 74,

34]. Following standard practice, we report δ1 accuracy, the percentage of predictions within 25% relative error of ground truth. All models are evaluated on the DepthVLM-Bench evaluation split.

Implementation Details. We adopt Qwen3-VL [3] (4B/8B) as the default VLM backbone, and integrate a lightweight DPT-style [46] head with 34M parameters (<1% of the LLM). Models are trained in PyTorch on 4.4M samples from the training split of DepthVLM-Bench with uniform

- Table 3: Evaluation on broad visual benchmarks, covering general VQA, document understanding, multi-image reasoning, counting, and hallucination. Empowered by our lightweight depth head and two-stage training strategy, our method natively gains the ability to generate dense geometry without sacrificing the general multimodal capability of the underlying VLM, in sharp contrast to prior text-heavy supervision approach [9] that typically incurs substantial capability degradation. Methods MMB-EN MMB-CN MMStar ScienceQA BLINK OCRBench CountBench POPE

Pixtral-12B [1] 78.2 73.7 52.0 88.8 49.2 660 69.5 85.5 Qwen3-VL-4B [3] 83.4 81.1 60.9 91.2 63.8 817 97.7 89.8 Qwen3-VL-8B [3] 84.7 82.4 63.4 92.8 65.0 833 98.3 88.8

DepthLM-12B [9]† N/A N/A N/A N/A N/A N/A N/A N/A

Ours-4B 82.9 (↓ 0.5) 81.8 (↑ 0.7) 60.4 (↓ 0.5) 91.3 (↑ 0.1) 63.3 (↓ 0.5) 832 (↑ 15) 98.0 (↑ 0.3) 89.9 (↑ 0.1) Ours-8B 84.6 (↓ 0.1) 82.3 (↓ 0.1) 63.8 (↑ 0.4) 93.1 (↑ 0.3) 64.8 (↓ 0.2) 862 (↑ 29) 98.2 (↓ 0.1) 89.1 (↑ 0.3)

† Due to its text-dominant supervised fine-tuning on single-pixel depth query, DepthLM [9] collapses to always emitting a

depth value regardless of the input instruction, making it incompatible with standard VQA evaluation protocols.

sampling. Intermediate ViT features are taken from layers 5, 11, and 17 for 4B, and 8, 16, and 24 for ratios of 0.04 and 0.05 for Stage-1 and Stage-2. The balance factors λ and α are set to 0.5 and 1.0.

- 8B. We use AdamW with a cosine schedule, learning rates of 3.5×10−4 and 2×10−5, and warmup

Demo

Point Depth Estimation Pair Distance Measurement Depth Order Ranking 3D Object Estimation

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

| |
|---|

Q: What is the distance of the point marked by the red

Q: What is the shortest distance between the frontleft vehicle and the yellow detour sign on the right?

Q: What is the depth order of the ①trash can, ②sink, and ③electric kettle from nearest to farthest?

Q: What are the 3D

dimensions (length, width, and height) of the washing machine?

arrow along the forward direction of the camera?

GT: ② ③ ① GPT-5.5: ② ① ③ Ours: ② ③ ①

GT: 31.1m GPT-5.5: 21.0m Ours: 32.3m

GT: 4.3m GPT-5.5: 11.2m Ours: 4.5m

GT: (0.8,0.6,0.9)m GPT-5.5: (0.6,0.4,0.6)m Ours: (0.8,0.6,1.0)m

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- Figure 4: Qualitative results on more complex 3D tasks. Beyond dense metric depth estimation, our model further supports a variety of downstream 3D reasoning tasks, demonstrating that native dense geometry prediction serves as a solid foundation for high-level spatial reasoning in VLMs.

真实的模型输出：

GT: ② ③ ① GPT-5.5: ② ③ ① Ours: ② ① ③

GT: 31.1m GPT-5.5: 21.0m Ours: 12.0m

GT: (0.8, 0.6, 0.9)m GPT-5.5: (0.6, 0.6, 0.8)m Ours: (0.7, 0.7, 0.8)m

GT: 4.3m GPT-5.5: 11.2m Ours: 2.8m

[Figure 37]

❌

#### 4.2 Main Results

Comparison with Other VLMs. To evaluate metric depth estimation in existing VLMs, we follow DepthLM [9] by prompting models with an arrow-marked pixel to predict its depth. As shown in Table 1, general-purpose VLMs perform poorly—especially in outdoor driving scenes—with Qwen3VL-32B [3] achieving δ1 = 0.21 and GPT-5.5 [41] only 0.41 on average, revealing a substantial gap to reliable 3D understanding. Even spatially enhanced VLMs, despite depth and calibration supervision, underperform a constant-depth baseline. In contrast, our model consistently excels across indoor and outdoor settings, significantly outperforming both larger and task-specific VLMs.

Comparison with Pure Vision Models. Table 2 further compares our model with leading specialized pure vision models on indoor and outdoor metric depth estimation. Since both pure vision models and DepthVLM produce dense metric depth maps, we evaluate them on the same sampled pixels used in the VLM setting for a fair comparison. Despite being a unified model with strong multimodal capabilities, our method not only significantly outperforms most vision specialists, including UniDepthV2 [42] and Metric3Dv2 [24], but also surpasses the state-of-the-art DepthAnythingV3 [34].

Evaluation on General Visual Benchmarks. To verify that dense geometry prediction does not compromise multimodal understanding, we evaluate on broad visual benchmarks in Table 3. Our models match their original VLM backbones and even improve on OCRBench [37] and POPE [32]. In contrast, prior depth-specialized VLMs such as DepthLM [9] often overfit to text-heavy supervision

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

DepthLMYoutu-VLRGB&GTOurs

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

[Figure 61]

- Figure 5: Qualitative comparison with others. Our results show finer structural details and improved semantic consistency across diverse scenes. Depth is color-coded from near ( ) to far.

[Figure 62]

- Table 4: Ablation of depth head designs. “Multiscale” indicates aggregation of visual features from intermediate ViT layers. Our lightweight DPTstyle [46] head with multi-scale fusion performs best due to the design tailored to VLM features. Depth Head Multi-scale Waymo NuScenes sunRGBD IBims-1

Table 5: Ablation of feature sources for the depth head. “Inter.” denotes three intermediate layers and “Final” the last-layer output. Fusing multi-scale ViT features with the LLM final feature yields the best depth fidelity.

3 5 6 2 3e6ceea56c, frame_002460

29b607c6d5, frame_001320

02455b3d20, frame_001020

3db0a1c8f3, frame_004260

Feature Source Waymo NuScenes sunRGBD IBims-1

Two-layer MLP ✗ 0.547 0.444 0.533 0.695 Two-layer MLP ✓ 0.723 0.776 0.727 0.806 Original DPT [46] ✓ 0.866 0.826 0.856 0.895 Ours (Lightweight DPT) ✓ 0.879 0.821 0.882 0.912

ViT Inter. + ViT Final 0.758 0.712 0.798 0.777 LLM Inter. + LLM Final (Stage-1 only) 0.720 0.710 0.763 0.744 LLM Inter. + LLM Final (Two-stage) 0.812 0.770 0.846 0.829 ViT Inter. + LLM Final (Two-stage) 0.879 0.821 0.882 0.912

and lose general-purpose capabilities. These results underscore the effectiveness of our unified design, which supports both accurate dense geometry prediction and strong multimodal understanding.

Evaluation on Spatial Reasoning Tasks. We further find that enabling a VLM to act as a native 3D dense geometry predictor also improves spatial reasoning performance. Figure 4 demonstrates more complex 3D reasoning tasks beyond metric depth estimation, where even pioneering GPT-5.5 [41] may fail. These results suggest that strong native dense geometry prediction capabilities provide a solid foundation for high-level spatial reasoning in VLMs.

Qualitative Visualizations. As shown in Figure 5, we compare the depth maps and corresponding 3D point clouds generated by Youtu-VL-4B [64] and DepthLM-12B [9] across diverse scenes. Youtu-VL produces noisy and fragmented point clouds with poor geometric continuity, while DepthLM maintains better semantic coherence but loses fine structural details. In contrast, our method significantly improves generation quality, preserving both semantic consistency and detailed spatial structure.

#### 4.3 Ablation Studies

In this section, we perform ablation experiments on the 4B model to thoroughly evaluate the effectiveness of the components.

Ablation of Depth Head Variants. We compare different depth prediction heads in Table 4. A two-layer MLP performs worst due to its overly simple architecture, while incorporating multi-scale ViT features already brings clear improvements. The original DPT [46] head remains suboptimal because downsampling the LLM final visual feature discards high-level semantic information. In contrast, our lightweight DPT-style head constructs a bottom-up feature pyramid through upsampling, assigning higher spatial resolution to earlier ViT layers, and achieves the best overall accuracy.

Ablation of Multi-Layer Feature Sources. Table 5 studies the impact of feature inputs to the DPTstyle head. Using only ViT features underperforms due to limited learnable parameters. LLM-only features improve results but remain suboptimal, as they lack fine-grained geometry details from early ViT layers. Combining multi-scale ViT features with the LLM final hidden state yields the best performance, effectively integrating high-level semantic information with detailed geometry cues.

- Table 6: Ablation of training strategies. We compare four variants: (i) Stage-1 Only, training only the depth head with the VLM frozen; (ii) Stage-2 Only, directly fine-tuning the full model; (iii) Stage-1

+ Stage-2‡, where the vision encoder is unfrozen in Stage-2; and (iv) our full strategy. Unfreezing the vision encoder yields marginal depth gains but degrades general multimodal performance, whereas our design achieves strong depth estimation accuracy while preserving the VLM’s general capability.

Training Strategy

Depth Estimation (δ1 ↑) General Visual Benchmarks Waymo NuScenes sunRGBD IBims-1 MMB-EN MMStar BLINK OCRBench

- Stage-1 Only 0.737 0.742 0.782 0.753 83.23 60.36 63.55 840

- Stage-2 Only 0.784 0.762 0.826 0.805 81.44 57.21 57.77 793 Stage-1 + Stage-2‡ (unfreeze ViT) 0.884 0.837 0.893 0.900 82.13 54.60 59.47 769 Stage-1 + Stage-2 (freeze ViT) 0.879 0.821 0.882 0.912 82.93 60.42 63.25 832

- Table 7: Ablation of focal-length normalization. We compare training on raw mixedsource images (w/o normalization) with canon-

Table 8: Efficiency comparison with others. We report the end-to-end cost of generating a 256×192 depth map. DepthLM [9] uses per-pixel queries, while Youtu-VL [64] predicts a sparse grid with upsampling. In contrast, our model outputs a pixel-aligned depth map in one pass, achieving higher efficiency without post-processing.

icalizing inputs to a shared focal length fc ∈ {800, 1000, 1200}. Normalization consistently

improves performance, with fc=1000 achieving the best results across diverse benchmarks.

#Fwd. / image

Output Pattern

Postproc.

Latency (ms) ↓

Training Setting Waymo NuScenes sunRGBD IBims-1

Method

Raw mixed-source 0.802 0.715 0.770 0.630 Canonical fc=800 0.833 0.824 0.865 0.883 Canonical fc=1000 0.879 0.821 0.882 0.912 Canonical fc=1200 0.858 0.840 0.837 0.856

DepthLM-12B [9] H×W point-wise queries none 13h Youtu-VL-4B [64] 1 sparse patch-level bilinear ↑ 2.48s Ours-4B 1 dense pixel-level none 0.42s

One-Stage vs. Two-Stage Training. We compare different training strategies in Table 6, including single-stage and two-stage variants. Stage-1 Only preserves multimodal reasoning but yields limited depth gains. Stage-2 Only improves geometry prediction, yet suffers from unstable optimization and reduced multimodal performance. The full two-stage strategy achieves a better balance. Unfreezing the vision encoder in Stage-2 further improves depth accuracy but harms general multimodal ability. Overall, our final design balances dense geometry prediction with the VLM’s general capability.

Effect of Focal-Length Normalization. We validate the focal-length normalization strategy in Table 7. Training on raw mixed-source data suffers from camera-induced scale ambiguity, as identical scenes with different focal lengths yield inconsistent metric depth supervision. Normalizing inputs to a shared focal length fc aligns projective geometry across datasets and mitigates this issue. We sweep fc∈{800, 1000, 1200}: smaller fc loses image details, while larger fc amplifies interpolation artifacts. Empirically, fc=1000 performs best across datasets, consistent with DepthLM [9].

Inference Efficiency Analysis. We compare the efficiency of VLM-based methods in Table 8 using 256×192 inputs and reporting end-to-end runtime. DepthLM [9] formulates depth estimation as per-pixel text queries, requiring H×W forward passes, making inference prohibitively slow. YoutuVL [64] reduces this to one pass but predicts sparse patches that require upsampling, introducing artifacts and overhead. In contrast, our method directly decodes multi-scale features into a pixelaligned depth map in one pass without post-processing. This efficiency stems from treating the VLM

- as a native dense predictor, enabling efficient dense geometry prediction within a unified framework.

### 5 Conclusion and Limitations

In this paper, we present DepthVLM, a unified foundation model that jointly supports low-level dense geometry prediction and high-level multimodal understanding. We integrate a lightweight depth head into the VLM backbone and adopt a two-stage training strategy under the unified vision-text supervision, enabling geometry prediction and language response in a single forward pass. Extensive experiments show that DepthVLM achieves leading performance across diverse datasets with higher inference efficiency, surpasses strong pure vision models, and improves complex 3D spatial reasoning. Limitations. This work mainly focuses on dense metric depth estimation and does not yet explore broader 3D perception tasks such as object detection and pose estimation. Extending the framework toward a unified model for holistic 3D perception and reasoning remains future work.

### References

- [1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024.
- [2] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19129–19139, 2022.
- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [4] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4009–4018, 2021.
- [5] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023.
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [7] Aleksei Bochkovskii, AmaãG, l Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024.
- [8] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.
- [9] Zhipeng Cai, Ching-Feng Yeh, Hu Xu, Zhuang Liu, Gregory Meyer, Xinjie Lei, Changsheng Zhao, ShangWen Li, Vikas Chandra, and Yangyang Shi. Depthlm: Metric depth from vision language models. arXiv preprint arXiv:2509.25413, 2025.
- [10] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017.
- [11] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024.
- [12] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In European conference on computer vision, pages 202–221. Springer, 2020.
- [13] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26428–26438, 2024.
- [14] An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. Advances in Neural Information Processing Systems, 37:135062–135093, 2024.
- [15] Wei-Lin Chiang, Zhuohan Li, Ziqing Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2(3):6, 2023.
- [16] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.
- [17] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. Advances in neural information processing systems, 27, 2014.

- [18] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025.
- [19] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pages 148–166. Springer, 2024.
- [20] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE, 2012.
- [21] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 3d packing for selfsupervised monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2485–2494, 2020.
- [22] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1.5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.
- [23] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023.
- [24] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zeroshot metric depth and surface normal estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10579–10596, 2024.
- [25] Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, and Jiangmiao Pang. G2vlm: Geometry grounded vision language model with unified 3d reconstruction and spatial reasoning. arXiv preprint arXiv:2511.21688, 2025.
- [26] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. Advances in Neural Information Processing Systems, 37:113991–114017, 2024.
- [27] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, SongChun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023.
- [28] Xiaohu Huang, Jingjing Wu, Qunyi Xie, and Kai Han. 3drs: Mllms need 3d-aware representation supervision for scene understanding. arXiv preprint arXiv:2506.01946, 2025.
- [29] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [30] Xinqi Jin, Hanxun Yu, Bohan Yu, Kebin Liu, Jian Liu, Keda Tao, Yixuan Pei, Huan Wang, Fan Dang, Jiangchuan Liu, et al. Streamingassistant: Efficient visual token pruning for accelerating online video understanding. arXiv preprint arXiv:2512.12560, 2025.
- [31] Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, and Marco Korner. Evaluation of cnn-based singleimage depth estimation methods. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops, pages 0–0, 2018.
- [32] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305, 2023.
- [33] Guosheng Lin, Anton Milan, Chunhua Shen, and Ian Reid. Refinenet: Multi-path refinement networks for high-resolution semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1925–1934, 2017.
- [34] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

- [35] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [36] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.
- [37] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xucheng Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023.
- [38] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.
- [39] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474, 2022.
- [40] Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16488–16498, 2024.
- [41] OpenAI. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.
- [42] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.
- [43] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024.
- [44] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428, 2025.
- [45] Santhosh Kumar Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. Habitat-matterport 3d dataset (HM3d): 1000 large-scale 3d environments for embodied AI. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2021.
- [46] René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021.
- [47] René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020.
- [48] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S Khan. Glamm: Pixel grounding large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13009–13018, 2024.
- [49] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multicamera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017.
- [50] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012.
- [51] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015.

- [52] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020.
- [53] Hao Tang, Chenwei Xie, Haiyang Wang, Xiaoyi Bao, Tingyu Weng, Pandeng Li, Yun Zheng, and Liwei Wang. Ufo: A unified approach to fine-grained visual perception via open-ended language interface. arXiv preprint arXiv:2503.01342, 2025.
- [54] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [55] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [56] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9275–9286, 2025.
- [57] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.
- [58] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10510–10522, 2025.
- [59] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.
- [60] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347, 2025.
- [61] Yuxin Wang, Lei Ke, Boqiang Zhang, Tianyuan Qu, Hanxun Yu, Zhenpeng Huang, Meng Yu, Dan Xu, and Dong Yu. N3d-vlm: Native 3d grounding enables accurate spatial reasoning in vision-language models. arXiv preprint arXiv:2512.16561, 2025.
- [62] Hui Wei, Hao Tang, Xuemei Jia, Zhixiang Wang, Hanxun Yu, Zhubo Li, Shin’ichi Satoh, Luc Van Gool, and Zheng Wang. Physical adversarial attack meets computer vision: A decade survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9797–9817, 2024.
- [63] Hui Wei, Hanxun Yu, Kewei Zhang, Zhixiang Wang, Jianke Zhu, and Zheng Wang. Moiré backdoor attack (mba): A novel trigger for pedestrian detectors in the physical world. In Proceedings of the 31st ACM International Conference on Multimedia, pages 8828–8838, 2023.
- [64] Zhixiang Wei, Yi Li, Zhehan Kan, Xinghua Jiang, Zuwei Long, Shifeng Liu, Hongze Shen, Wei Liu, Xiaoyu Tan, Haojia Lin, et al. Youtu-vl: Unleashing visual potential via unified vision-language supervision. arXiv preprint arXiv:2601.19798, 2026.
- [65] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.
- [66] Diankun Wu, Fangfu Liu, Yi-Hsin Hung, and Yueqi Duan. Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747, 2025.
- [67] Jiannan Wu, Muyan Zhong, Sen Xing, Zeqiang Lai, Zhaoyang Liu, Zhe Chen, Wenhai Wang, Xizhou Zhu, Lewei Lu, Tong Lu, et al. Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks. Advances in Neural Information Processing Systems, 37:69925–69975, 2024.
- [68] Xianjin Wu, Dingkang Liang, Tianrui Feng, Kui Xia, Yumeng Zhang, Xiaofan Li, Xiao Tan, and Xiang Bai. Generation models know space: Unleashing implicit 3d priors for scene understanding. arXiv preprint arXiv:2603.19235, 2026.

- [69] Runsen Xu, Weiyao Wang, Hao Tang, Xingyu Chen, Xiaodong Wang, Fu-Jen Chu, Dahua Lin, Matt Feiszli, and Kevin J Liang. Multi-spatialmllm: Multi-frame spatial understanding with multi-modal large language models. arXiv preprint arXiv:2505.17015, 2025.
- [70] Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, and Weidi Xie. Omnistream: Mastering perception, reconstruction and action in continuous streams. arXiv preprint arXiv:2603.12265, 2026.
- [71] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [72] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. In CVPR, 2025.
- [73] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10371–10381, 2024.
- [74] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.
- [75] Shusheng Yang, Jihan Yang, Pinzhi Huang, Ellis Brown, Zihao Yang, Yue Yu, Shengbang Tong, Zihan Zheng, Yifan Xu, Muhan Wang, Daohan Lu, Rob Fergus, Yann LeCun, Li Fei-Fei, and Saining Xie. Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670, 2025.
- [76] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.
- [77] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3d: Towards zero-shot metric 3d prediction from a single image. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9043–9053, 2023.
- [78] Hanxun Yu, Wentong Li, Xuan Qu, Song Wang, Junbo Chen, and Jianke Zhu. Visiontrim: Unified vision token compression for training-free mllm acceleration. arXiv preprint arXiv:2601.22674, 2026.
- [79] Hanxun Yu, Wentong Li, Song Wang, Junbo Chen, and Jianke Zhu. Inst3d-lmm: Instance-aware 3d scene understanding with multi-modal instruction tuning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14147–14157, 2025.
- [80] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3712–3722, 2018.
- [81] Boqiang Zhang, Kehan Li, Zesen Cheng, Zhiqiang Hu, Yuqian Yuan, Guanzheng Chen, Sicong Leng, Yuming Jiang, Hang Zhang, Xin Li, et al. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106, 2025.
- [82] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.
- [83] Zheng Zhang, Yeyao Ma, Enming Zhang, and Xiang Bai. Psalm: Pixelwise segmentation with large multi-modal model. In European Conference on Computer Vision, pages 74–91. Springer, 2024.
- [84] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625, 2025.
- [85] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8995–9006, 2025.
- [86] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024.

Technical Appendices and Supplementary Material

- A Statistics of DepthVLM-Bench

We summarize the data composition of DepthVLM-Bench in Tables 9 and 10.

For training, we construct a large-scale and diverse dataset by sampling from the training splits of multiple public benchmarks spanning both autonomous driving and indoor scenes, including Argoverse2 [65], Waymo [52], DDAD [21], NuScenes [8], ScanNet++ [76], Taskonomy [80], HM3D [45], and Matterport3D [10]. As many RGB images are extracted from videos and thus contain highly redundant, near-duplicate frames, we apply uniform sampling to reduce redundancy. Most datasets contribute approximately 800K images, resulting in a relatively balanced distribution across domains and helping mitigate dataset bias. Smaller datasets, such as DDAD and Matterport3D, are included at their original scales to further enrich diversity. In total, the training set comprises 4.4M images, covering a wide range of scenes, viewpoints, and depth distributions. We train for a single epoch. The 8B variant is trained for four days on 80 NVIDIA H20 GPUs, while the 4B variant is trained for two days using the same computational resources.

For evaluation, we assemble a comprehensive benchmark using the validation or test splits of the same public datasets, covering both outdoor and indoor scenarios. When possible, each dataset contributes around 1K images to ensure a balanced and consistent evaluation protocol across sources. In addition, we include several standard indoor benchmarks, including ETH3D [49], sunRGBD [51], IBims1 [31], and NYUv2 [50], to further assess generalization across diverse environments and capture varying depth characteristics. This design enables a fair and reliable evaluation of cross-domain generalization and robustness for dense geometry prediction.

- Table 9: Training data statistics. Number of training images sampled from each source dataset in our setting.

Dataset Argoverse2 Waymo DDAD NuScenes ScanNet++ Taskonomy HM3D Matterport3D # Images 800K 800K 76K 206K 800K 800K 800K 158K

- Table 10: Benchmark data statistics. Number of evaluation images sampled from each dataset. Dataset Argoverse2 Waymo DDAD NuScenes ETH3D ScanNet++ sunRGBD IBims-1 NYUv2 # Images 1K 1K 1K 1K 454 1K 1K 100 654

- B Evaluation of VLMs on Metric Depth Estimation

To fairly compare the metric depth estimation capabilities of existing VLMs with our model, we design a standardized evaluation protocol described below.

Following the paradigm introduced by DepthLM [9], given a single RGB image, a specific pixel location is indicated by a red arrow marker drawn directly on the image. The model is asked to estimate the metric depth (in meters), i.e., the distance along the camera’s optical axis from that point to the camera, as illustrated in Figure 6.

Visual Marker Size. DepthLM uses a default red arrow marker of 5 pixels. However, through our experiments, we observe that off-the-shelf VLMs that have not been specifically trained for this task cannot reliably detect a 5 pixels marker. When not explicitly constrained to output only a number, these models frequently respond with statements such as:

“There is no red arrow in the image. Therefore, the distance cannot be determined.”

To ensure fairness and to maximally elicit each model’s depth estimation capability, we increase the red arrow marker size to 20 pixels for all evaluated VLMs. This larger marker is clearly visible in all tested scenes, ensuring that the evaluation measures depth estimation ability rather than marker detection ability.

Z-Depth

|Prompt: Given this image, how far is the point pointed by the red arrow from the camera in the forward backward direction? Please respond with only a number in meters, rounded to two decimal places, without any unit or explanation.<br><br>[Figure 63]<br><br>[Figure 64]<br><br>|[Figure 65]|
|---|
<br><br>| |
|---|
|
|---|

Figure 6: VLM evaluation setup for metric depth estimation. A red arrow marker (20 pixels) is drawn on the input image to indicate the query point. The model receives the annotated image along with a text prompt asking for the metric depth (in meters). The zoomed-in region shows the arrow marker in detail.

Image Resolution Handling. Since different datasets are captured at varying resolutions, we downscale images with a longest edge over 1024 pixels to this limit before adding the red arrow marker, while leaving smaller images unchanged. This provides a consistent input condition across all models and datasets, and ensures that the marker remains clearly visible at the given image resolution.

With the above protocol, all VLMs are evaluated under identical conditions, ensuring a fair comparison with our method. For VLMs such as DepthLM [9] and Youtu-VL [64], which are specifically trained for metric depth estimation, we directly adopt their provided evaluation scripts.

### C Asset License and Consent

Committed to openness and transparency to mitigate misinformation concerns [62, 63], we provide the licenses and URLs of all public datasets and benchmarks used in this work. These resources span a wide range of multimodal tasks [38, 32, 37, 36, 19], including depth estimation, general multimodal understanding, spatial reasoning, document understanding, multi-image reasoning, visual grounding, and hallucination evaluation. Table 11 summarizes all datasets and benchmarks used in this study along with their corresponding licenses and URLs.

Table 11: Open-source resources utilized in this paper.

Name License URL ScienceQA MIT License https://github.com/lupantech/ScienceQA POPE MIT License https://github.com/AoiDragon/POPE OCRBench MIT License https://github.com/Yuliang-Liu/MultimodalOCR MMBench Apache License 2.0 https://github.com/open-compass/MMBench BLINK Apache License 2.0 https://github.com/zeyofu/BLINK_Benchmark Argoverse2 Apache License 2.0 https://github.com/argoverse/argoverse-api Waymo Apache License 2.0 https://github.com/waymo-research/waymo-open-dataset NuScenes Apache License 2.0 https://github.com/nutonomy/nuscenes-devkit MMStar No license specified https://github.com/MMStar-Benchmark/MMStar CountBenchQA No license specified https://huggingface.co/datasets/vikhyatk/CountBenchQA ScanNet++ ScanNet++ Terms of Use https://scannetpp.mlsg.cit.tum.de/scannetpp/ DDAD Creative Commons Attribution 4.0 https://github.com/TRI-ML/DDAD ETH3D Creative Commons Attribution 4.0 https://www.eth3d.net/datasets

