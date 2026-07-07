# arXiv:2511.19320v2[cs.CV]18May2026

## SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation

Jiaming Zhang1,‡∗ Shengming Cao2,‡ Rui Li2,‡ Xiaotong Zhao2 Yutao Cui2 Xinglin Hou Gangshan Wu1 Haolan Chen2 Yu Xu2 Limin Wang1,3,† Kai Ma2,† 1State Key Laboratory for Novel Software Technology, Nanjing University 2Platform and Content Group (PCG), Tencent 3Shanghai AI Lab

[Figure 1]

Figure 1: We introduce SteadyDancer, an Image-to-Video (I2V) paradigm animation framework to achieve harmonized and coherent animation with first-frame preservation.

#### Abstract

Preserving first-frame identity while ensuring precise motion control is a fundamental challenge in human image animation. The Image-to-Motion Binding process of the dominant Reference-to-Video (R2V) paradigm overlooks critical spatiotemporal misalignments common in real-world applications, leading to failures such as identity drift and visual artifacts. We introduce SteadyDancer, an Image-toVideo (I2V) paradigm-based framework that achieves harmonized and coherent animation and is the first to ensure first-frame preservation robustly. Firstly, we propose a Condition-Reconciliation Mechanism to harmonize the two conflicting conditions, enabling precise control without sacrificing fidelity. Secondly, we design Synergistic Pose Modulation Modules to generate an adaptive and coherent pose representation that is highly compatible with the reference image. Finally, we employ a Staged Decoupled-Objective Training Pipeline that hierarchically optimizes the model for motion fidelity, visual quality, and temporal coherence. Experiments demonstrate that SteadyDancer achieves state-of-the-art performance

∗Work is done during internship at Tencent PCG. ‡ Equal contribution. † Corresponding author (lmwang@nju.edu.cn).

Preprint.

in both appearance fidelity and motion control, while requiring significantly fewer training resources than comparable methods. The model has been publicly released at https://mcg-nju.github.io/steadydancer-web.

#### 1 Introduction

Human image animation, which aims to generate video from a single static image with controllable motion, has emerged as a prominent research frontier in video generation and holds immense potential for applications like film production, advertising, and video game development. While significant progress has been made, the breakthroughs of diffusion models [16, 28] have recently unlocked new capabilities in generating high-fidelity and temporally coherent videos.

Most existing methods adhere to the Reference-to-Video (R2V) paradigm [14], as illustrated in the left part of Fig. 2 (b). Specifically, the R2V paradigm defines the animation task as binding the reference image onto the driven pose, which inherently relaxes the alignment constraints between the inputs, thereby reducing the objective to only achieving a natural appearance transfer. However, inherent discrepancies exist between the image and pose inputs in practical scenarios as shown in Fig. 2 (a), manifesting as spatial misalignments limb structure and proportion) and temporal misalignments (movement type and amplitude). In such scenarios, the relaxation of alignment constraints within the R2V paradigm leads to unacceptable results, including severe artifacts, appearance distortions, and temporal incoherence. Moreover, in the common start-gap scenario due to the temporal misalignments, the R2V paradigm instantly binds the reference image to the first pose, completely omitting any smooth transition from the reference state. In real-world applications, including VFX post-production, keyframe-based animation, and digital human activation, the generated video must start exactly from the reference frame. However, this abrupt jump, combined with the appearance distortion, makes it unsuitable for these applications that require high visual and temporal fidelity.

In contrast, the Image-to-Video (I2V) paradigm, which inherently guarantees first-frame preservation by generating consistent and coherent videos starting from the initial frame, maximizes fidelity and emerges as a preferable solution. However, I2V-based animation research remains scarce and technically challenging, due to its stringent requirement for first-frame coherence, which demands that all input conditions and generated results adhere to the initial frame. Specifically, it requires the pose to be modulated into a suitable control signal for the reference image, necessitating much tighter alignment than R2V. Failure to achieve such strict alignment will severely impact performance, especially when there are insufficient resources to train high-capacity models for learning control.

In this paper, we propose SteadyDancer, an animation model built on the I2V paradigm with firstframe preservation. Firstly, to resolve the trade-off between appearance preservation and motion control within the I2V paradigm, we propose Condition-Reconciliation Mechanism that achieves precise motion-driven control without sacrificing first-frame fidelity by optimizing at three levels, including condition fusion, injection, and augmentation. Secondly, to address the spatio-temporal misalignment between the reference image and the driving pose, we introduce Synergistic Pose Modulation Modules to extract pose representations that are better adapted to the reference image, comprising the Spatial Structure Adaptive Refiner, the Temporal Motion Coherence Module, and the Frame-wise Attention Alignment Unit. Finally, to achieve efficient and stable model training, we propose a Staged Decoupled-Objective Training Pipeline, including an Action Supervision Stage to establish precise motion control, a Condition-Decoupled Distillation Stage to enhance the generated video quality, and a Motion Discontinuity Mitigation Stage that aims to generate smooth and continuous results.Ultimately, building on these strategies, SteadyDancer successfully activates the ability of first-frame preservation in the human animation task, enabling it to robustly handle misalignments and generate videos with high fidelity and accurate motion. Moreover, it achieves superior performance over existing methods while requiring substantially fewer training resources. Meanwhile, we propose the first non-homogeneous benchmark, X-Dance, to test model performance when the reference and action sources are inconsistent.

In summary, the main contribution of this work lies in: (i) We propose SteadyDancer, a novel highfidelity animation framework that achieves first-frame preservation firstly and significant training resource efficiency; (ii) To address the conflict and mismatch between the reference image and the driven pose, and to improve training efficiency, we propose a Condition-Reconciliation Mechanism, Synergistic Pose Modulation Modules, and a Staged Decoupled-Objective Training Pipeline; (iii)

[Figure 2]

[Figure 3]

(a) (b)

- Figure 2: (a) The spatio-temporal misalignment in practical scenarios. (b) The illustration of Reference-to-Video (R2V) and Image-to-Video (I2V) paradigms for human image animation. While R2V only cares about how to bind the reference image to driven motion, I2V additionally needs to carefully deal with the misalignment from the driven motion to the reference image.

Extensive quantitative and qualitative results on multiple benchmarks validate the superiority and effectiveness of our proposed method.

#### 2 Related Work

Diffusion for video generation Diffusion-based models have become state-of-the-art for generative tasks, achieving remarkable success in both image [20, 37] and video generation [36, 4]. With the introduction of OpenAI’s Sora [2], which leverages the Diffusion Transformer (DiT) architecture [19], DiT-based approaches have since supplanted UNets [21], as their pure Transformer structure facilitates massive parameter scaling and has become the mainstream technical route. Concurrently, to efficiently handle video data, many recent DiT models [28, 16, 35] adopt 3D VAEs over standard 2D VAEs [20, 15] to compress data across both spatial and temporal dimensions. The proliferation of powerful, open-source foundational models has further accelerated this field. Consequently, human image animation, as one of downstream tasks, directly benefits from the enhanced power and fidelity of these new models.

Human Image Animation. Earlier works in image animation primarily relied on warping-based feature representations and GAN-based architectures [22, 23, 41]. Recently, this field has pivoted to diffusion models, yielding significant performance improvements. Early diffusion-based methods, such as DisCo [29], leveraged ControlNet [37] for pose guidance and integrated motion modules to enhance cross-frame consistency. A key breakthrough came with Animate Anyone [9] and subsequent studies [34, 44, 26], which utilize a UNet-based ReferenceNet to inject appearance features, achieving excellent identity preservation. To further enhance controllability, other works [44, 10, 42, 32, 40] incorporated camera parameters and rich 3D geometric guidance, such as depth, SMPL, hand. Mirroring the trend in general video generation, DiT-based architectures [19] have recently been adapted for human animation [39, 33, 30, 43, 17, 5, 14], leading to substantial enhancements in realism and temporal continuity. However, most approaches follow the Reference-to-Video (R2V) paradigm. This paradigm focuses on binding Image-to-Pose naturally, which inherently ignores critical input misalignments, frequently leading to unsatisfactory results.

#### 3 Method

Given a reference image Ic along with a pose sequence Pm = {p0,...,pT}, the animation task aims to generate a video that preserves the appearance of the reference while maintaining adherence to the pose sequence. As shown in Fig. 3, we first review the Image-to-Video (I2V) generation model (Sec. 3.1) and introduce a Naïve I2V Baseline to highlight its limitations (Sec. 3.2). We then present the core technical components, including the Condition-Reconciliation Mechanism (Sec. 3.3) and the Synergistic Pose Modulation Modules (Sec. 3.4). Finally, we describe the Staged Decoupled-Objective Training Pipeline (Sec. 3.5).

[Figure 4]

- Figure 3: An overview of SteadyDancer, a Image-to-Video (I2V) paradigm-based Human Image Animation framework. First, it employs a Condition-Reconciliation Mechanism to reconcile appearance and motion conditions, achieving precise control without sacrificing first-frame preservation. Second, it utilizes Synergistic Pose Modulation Modules to resolve critical spatio-temporal misalignments.

##### 3.1 Preliminaries

A foundational Image-to-Video (I2V) model [28, 35, 1] conditions synthesis on a static image Ic, used as the first frame, concatenated with zero-filled frames, and encoded by a VAE encoder E [15] into

zc. At denoising timestep t, the Diffusion Transformer (DiT) [19] receives the channel-concatenated noisy latent zˆt, binary mask m, and condition latent zc:

zt = ChannelConcat(ˆzt,m,zc). (1)

The DiT predicts the denoised latent from zt, while global context cclip and text conditions ctxt are injected through decoupled cross-attention for spatial-semantic alignment.

##### 3.2 Naïve I2V Baseline

To introduce pose control Pm, we build a Naïve Baseline that treats pose and image conditions equivalently. The pose sequence reuses the image VAE encoder to obtain zp in the same feature space as zc, then fuses them by element-wise addition for denoising:

zt = ChannelConcat(ˆzt,m,zc + zp). (2) , which achieves both appearance preservation and motion control by simple element-wise addition.

##### 3.3 Condition-Reconciliation Mechanism

The Naïve Baseline uses simple additive fusion to preserve appearance and motion, but this conflates static image details with driving-pose dynamics, a conflict especially acute in I2V. The fused signal often favors control and weakens appearance retention. We therefore propose the ConditionReconciliation Mechanism, a three-aspect design for precise control with first-frame preservation.

Condition Fusion. We identify that the element-wise addition is a critical bottleneck of Naïve Baseline, which conflates the static appearance (zc) and dynamic pose (zp) signals, leading to information loss and mutual interference. To resolve this, we replace it with channel-wise concatenation as:

zt = ChannelConcat(ˆzt,m,zc,zp), (3) which keeps conditions distinct, improving appearance preservation and motion control.

Condition Injection. Parameter-intensive injection, e.g., adapters or decoupled cross-attention, increases trainable parameters and attention cost, risking interference with the pre-trained generator. We instead inject the pose latent zp with the image condition and apply LoRA fine-tuning, enhancing motion control while preserving generation capacity and first-frame fidelity.

Condition Augmentation. To further reinforce the preservation of the first frame, we introduce two augmentation strategies. First, we augment the DiT input at the temporal level. We take the channel-concatenated latent zcond from the Condition Fusion and temporally concatenate it with the image latent zc

and the first-frame pose latent zp

as:

0

0

zcond = ChannelConcat(ˆzt,m,zc,zp), zt = TemporalConcat(zcond,zc

(4)

,zp

).

0

0

This provides the model with an explicit, clean reference to the starting appearance and pose. Second, we enhance the global context cclip by concatenating it with the CLIP feature of the first pose frame. This provides the model with a richer, pose-aware semantic embedding. These two augmentations work synergistically to improve identity preservation and visual consistency.

##### 3.4 Synergistic Pose Modulation Modules

While our Condition-Reconciliation Mechanism improves the fidelity-control balance (Fig. 2), spatiotemporal misalignment remains. Spatial misalignment arises from source-pose disparities in skeleton or identity, causing structural changes, identity drift, and detail loss. Temporal misalignment stems from noisy poses and abrupt source-to-pose transitions, producing jitter and realism degradation; we therefore design modulation modules for precise alignment.

Spatial Structure Adaptive Extractor. To address spatial misalignment, we propose the Spatial Structure Adaptive Refiner PSSAE with dynamic convolution, which generates pose-dependent parameters from zp to modulate latent representations. Using global context to predict scaling factors and a transformation matrix, its dual-path restructuring produces image-compatible pose features that reduce fusion interference and preserve fine details.

Temporal Motion Coherence Module. For temporal misalignment, we introduce the Temporal Motion Coherence Module PTMCM to model continuous dynamics from the discrete pose zp. Stacked factorized blocks use depthwise spatial convolutions for intra-frame structures and pointwise temporal convolutions for smoothed inter-frame dynamics. This suppresses erratic-pose artifacts and yields coherent motion guidance.

Frame-wise Attention Alignment Unit. To enforce fine-grained pose-appearance correspondence, we introduce the lightweight Frame-wise Attention Alignment Unit PFAAU via cross-attention, where denoising latent zˆt (Query) attends to pose latent (Key / Value). It yields appearance-calibrated pose features for subsequent fusion.

Hierarchical Aggregation. Ultimately, we combine the aforementioned three modules using a hierarchical aggregation strategy. First, the base pose feature is processed in parallel by the spatial (PSSAE) and temporal (PTMCM) modules. Their outputs are then integrated with the base feature via a residual connection to construct a high-quality, spatio-temporally coherent representation. This intermediate representation is immediately calibrated by the Alignment Unit (PFAAU). This synergistically refined pose feature achieves precise appearance alignment, which then serves as an additional, high-quality control condition for Eq. 4. The aggregation process can be formalized as:

zp∗ = zp + PSSAE(zp) + PTMCM(zp), zp† = PFAAU(q = zˆt,kv = zp∗),

(5)

zcond = ChannelConcat(ˆzt,m,zc,zp∗,zp†).

##### 3.5 Staged Decoupled-Objective Training Pipeline

Our training pipeline is divided into three distinct stages to achieve efficient and precise training.

Action Supervision. This stage instills motion-control capability. For each training video, the first frame is reference, while the full video provides the motion condition and supervision target. We use LoRA-based [8] fine-tuning to preserve the model’s generative priors.

Condition-Decoupled Distillation. To recover quality lost during motion-control learning, the second stage aims to improve detail while retaining first-stage pose controllability. We use the original pre-trained I2V model as the teacher model θ, a stationary manifold for the unconditional data distribution, and the first-stage model as the student model ϕ, a conditional flow estimator. We decompose velocity prediction into unconditional and conditional components:

vϕ(xt,t,c) = vuϕ(xt,t)

###### + vcϕ(xt,t,c)

, (6)

unconditional component

conditional component

where Ldistill = ∥vuϕ(xt,t) − vuθ(xt,t)∥2 aligns the unconditional component with the frozen teacher, and Lfidelity = ∥vϕ(xt,t,c) − v∗∥2 regresses the ground-truth velocity field v∗ like the first stage. Consequently, the teacher’s unconditional manifold is injected into the student without biasing pose-specific conditions, eliminating the distribution shift when a conditional network mimics an unconditional target, thereby improving video quality.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Reference Driven Video Iter 1000 Iter 1500 Iter 12000

- Figure 4: Pose Simulation in Motion Discontinuity Mitigation of Staged Decoupled-Objective Training Pipeline, which randomly replaces the second pose with an interpolated pose.

Figure 5: Model performance across various training steps. The results indicate that it rapidly acquires motion-control in the early steps, while the later steps focus more on detail.

Table 1: Comparison of Extra Inputs, Pre-Trained Model, and Training Requirements.

Method Extra Inputs Pre-Trained Model Training Step Training Data UNet-based

Disco [CVPR24] [29] Mask Stable Diffusion 70k TikTok (350) Animate Anyone [CVPR24] [9] × Stable Diffusion [30k, 10k] 5k character video clips MagicAnimate [CVPR24] [34] DensePose Stable Diffusion - TikTok (350), TED-talks (1,203)

Champ [ECCV24] [44] Depth, SMPL Stable Diffusion [60k, 20k] 5k human videos HumanVid [NeurIPS24] [32] Camera Stable Diffusion [30k, 10k] 20k real, 50k synthetic RealisDance [arxiv24] [42] SMPL, HaMeR Real Vision [200k, 100k] about 16k videos

StableAnimator [CVPR25] [26] Face Stable Video Diffusion 20 ep (∼15k) 3K videos (60-90 seconds long)

X-Dyna [CVPR25] [3] Face Stable Diffusion [5ep, 2ep] 107,546, 30-second videos MIMO [CVPR25] [18] Depth, Mask Stable Diffusion 50K 5k real, 2k synthetic videos

Animate-X [ICLR25] [25] × Stable Diffusion - 9k human videos

MimicMotion [ICML25] [40] × Stable Video Diffusion 20 ep (∼11k) 4,436 human dancing videos DiT-based

Dreamactor-M1 [ICCV25] [17] Face Seaweed APTs [20k, 20k, 30k] 500-hour videos

VACE [ICCV25] [14] Mask Wan-2.1 T2V 14B 200k 1M FlexiAct [SIGGRAPH25] [39] × CogVideoX-I2V [40K, 1.5K] -

X-UniMotion [SIGGRAPHASIA25] [24] Face, Hands Seaweed-7b 40k 200-hour videos UniAnimate-DiT [arxiv25] [30] × Wan-2.1 I2V 14B - ∼10K human dance videos RealisDance-DiT [arxiv25] [43] SMPL Wan-2.1 I2V 14B - 1M high-quality videos HyperMotion [arxiv25] [33] × Wan-2.1 I2V 14B 20k 19,597 video clips Wan-Animate [arxiv25] [5] Face Wan-2.1 I2V 14B - -

SteadyDancer (Ours) × Wan-2.1 I2V 14B [12k, 2k, 0.5k] 7338 videos (10.2 hours)

Motion Discontinuity Mitigation. At test time, reference–pose gaps can cause abrupt start-frame jumps. Training lacks this start-gap because the reference image and first pose are identical (v0), while random jumps destabilize learning. We therefore propose Pose Simulation (Fig. 4): for a smooth sequence {p0,p1,...,pT}, we sample (p0,pT∗) with T∗ ∈ {2,3,4}, interpolate {p˜1,...,p˜T∗−1}, and replace p1 with p˜1 to form {p0,p˜1,...,pT}. This strategy needs only a few hundred fine-tuning steps and no architectural changes, resolving over 80% of discontinuities without post-processing.

#### 4 Experiments

##### 4.1 Experimental Setups

Implementation Details. We initialize from a pre-trained Wan-2.1 I2V 14B video model [28] and conduct all experiments on 8 NVIDIA H800s. Each sample contains 81 frames, with center resolution 640 for training and 768 for inference. Our three stages run for 12,000, 2,000, and 500 steps, totaling only 14,500 steps. As Table 1 shows, this is significantly fewer training steps than other DiT-based methods, while still delivering strong motion control, quality, and smoothness. Fig. 5 further shows that the first stage quickly learns control before focusing on details.

Training Dataset. We collect a proprietary human motion dataset with 7,338 five-second clips, totaling 10.2 hours. It mainly contains human dance sequences, with a few slow-motion documentarystyle shots, and excludes extreme or complex movements. As shown in Table 1, the dataset scale is significantly smaller than comparable works, demonstrating that the I2V paradigm leverages image priors to reduce training overhead over R2V.

Evaluation Metrics. For the TikTok [13] dataset, we follow the settings in HumanVid [32]. The metrics consist of image quality (SSIM [31], LPIPS [38], PSNR [7], and FID [6]) and video fidelity (FVD [27]). For the RealisDance-Val [43] dataset, we utilize Vbench-I2V [11, 12] to facilitate finegrained and objective evaluation. For high-performance models, low-level metrics are insufficient

[Figure 11]

- Figure 6: Qualitative comparisons between SteadyDancer and other methods on the X-Dance. Each example displays the evolution (starting from the first frame), highlighting our model’s superior high-fidelity, coherence, and first-frame preservation.

Table 2: Quantitative Results on the TikTok, RealisDance-Val, and our X-Dance. .

I2V Subject↑

I2V Background↑

Subject Consistency↑

Background Consistency↑

Temporal Flickering↑

Motion Smoothness↑

Aesthetic Quality↑

Imaging Quality↑

I2V Subject↑

I2V Background↑

Subject Consistency↑

Background Consistency↑

Temporal Flickering↑

Motion Smoothness↑

Aesthetic Quality↑

Imaging Quality↑

Model

FVD↓

Model

Method SSIM↑ PSNR↑ LPIPS↓ FID↓ FVD↓ Moore-AnimateAnyone [9] 0.752 16.79 0.288 52.26 935.6 MagicAnimate [34] 0.748 17.89 0.270 56.84 876.0 Champ [44] 0.778 18.43 0.267 50.76 736.1 Animate-X [25] 0.740 16.71 0.280 32.77 508.6 HumanVid [32] 0.778 18.76 0.247 41.35 691.8 Realisdance-DiT [43] 0.717 17.55 0.260 30.39 458.8 SteadyDancer (Ours) 0.749 17.67 0.263 30.65 451.3

Moore-AnimateAnyone [9] 94.00 94.69 94.65 94.90 97.16 98.07 51.56 66.34 748.38 HumanVid [32] 94.72 95.27 93.69 94.94 97.87 98.52 55.58 67.45 624.33 MimicMotion [40] 90.78 92.52 92.21 93.60 97.46 98.61 52.09 59.67 785.73 Animate-X [25] 95.68 96.22 93.39 95.11 97.79 98.68 51.72 60.91 679.66 Hyper-Motion [33] 94.76 95.71 93.58 94.97 98.19 99.01 52.97 65.52 568.14 UniAnimate-DiT [30] 93.15 93.95 94.56 95.44 98.78 99.24 52.18 65.52 599.03 VACE [14] 87.39 88.58 93.56 95.03 96.74 98.25 57.81 70.61 911.72 Wan-Animate [5] 93.81 94.61 93.06 94.52 98.42 98.96 54.47 66.87 386.87 SteadyDancer (Ours) 94.68 95.38 93.48 95.18 97.99 99.02 56.80 68.45 326.49

Moore-AnimateAnyone [9] 85.56 86.04 90.38 92.00 95.05 96.78 48.89 69.66 HumanVid [32] 86.42 85.67 89.83 92.15 97.30 98.19 52.99 64.73 MimicMotion [40] 87.27 88.26 90.05 91.56 96.95 98.29 54.26 61.58 Animate-X [25] 93.58 94.53 92.47 93.42 96.76 97.97 59.86 66.74 UniAnimate-DiT [30] 91.35 91.75 92.81 93.00 97.52 98.45 58.18 70.85 VACE [14] 75.92 78.29 89.51 92.15 97.00 98.01 53.01 60.74 Wan-Animate [5] 88.71 89.23 90.65 92.67 97.41 98.40 56.57 62.79 SteadyDancer (Ours) 96.17 96.92 91.61 93.38 97.10 98.37 61.57 71.74

(a) TikTok dataset.

(c) X-Dance dataset.

(b) RealisDance-Val dataset.

proxies for generative quality, as they over-penalize minor shifts, miss semantic coherence, and are inapplicable without ground truth. Although VBench provides a supplement, it lacks motion-driving accuracy measures; thus, qualitative visual effectiveness better reflects real-world performance, pending stronger metrics.

##### 4.2 Comparison with the State-of-the-Art methods

Quantitative results. As shown in Table 2, we evaluate TikTok with low-level metrics and RealisDance-Val/X-Dance with multi-dimensional Vbench-I2V metrics. SteadyDancer achieves highly competitive results, especially on representative FID, FVD, and VBench metrics, including our real-world-oriented, non-homogeneous X-Dance benchmark.

Qualitative comparisons. To evaluate spatio-temporal misalignments beyond same-source benchmarks, we propose X-Dance, a different-source benchmark with diverse references (male/female/cartoon and upper-/full-body shots) and challenging driving videos with blur, occlusion, spatial-structural inconsistencies, and temporal start-gaps. As shown in Fig. 6, competing methods often fail to preserve identity or follow driving motion under these challenges, whereas our model maintains first-frame identity with precise, coherent motion control. We also evaluate RealisDance-Val, which includes daily/dance motions and complex Human-Object Interactions that test pose following and interaction potential. As shown in Fig. 7 and Fig. 8, our model synthesizes interacting objects with physically plausible motion and deformation while preserving appearance, unlike competing methods that often produce static artifacts or shape collapse.

Generalization to Multi-person and Animal Scenarios. We further explore our model’s zero-shot generalization in multi-person (with a multi-person pose estimator) and animal scenarios. As shown

[Figure 12]

- Figure 7: Visualization on RealisDance-Val about complex Human-Object Interactions. Even when driven solely by human pose, our model successfully synthesizes the interacting objects with physically plausible motion and deformation.

[Figure 13]

[Figure 14]

- Figure 10: Zero-shot generalization to multi-person and animal scene.

[Figure 15]

- Figure 11: First-frame preservation with first-last-frame generation post-processing.

Figure 8: Comparison on RealisDance-Val, showing that our model achieves both precise control and reasonable interaction with objects, causing them to produce reasonable movements.

[Figure 16]

Figure 9: Training-free long-video generation result.

in Fig. 10, our model seamlessly handles multi-person pose inputs, maintaining the same performance of identity preservation and motion fidelity as in single-person. Remarkably, our framework achieves this performance without any multi-person/animal training data or specialized inference-time tuning.

Training-free Long-Video Generation. Although our default training and inference setting uses 81 frames, SteadyDancer can be extended to minute-long video generation in a training-free manner using overlapping context-window inference, with 81-frame sliding windows, FreeNoise initialization, and linear overlap fusion. As shown in Fig. 9, the generated long video maintains strong identity preservation and coherent motion over an extended duration.

First-frame preservation without post-processing. R2V methods can also achieve first-frame preservation by first-last-frame post-processing, which uses the reference frame and generated first frame to synthesize the shaded segment in Fig. 11, but this adds computation and still fails to preserve the reference appearance during the animation generation part. In contrast, SteadyDancer achieves first-frame preservation intrinsically through I2V generation, making it not only a native capability but also an effective means of preserving identity.

##### 4.3 Ablation Study

Condition-Reconciliation Mechanism. To achieve precise control with minimal data while preserving I2V priors, condition integration is crucial. As shown in Fig. 12, channel concatenation better preserves identity than element-wise addition (Row 1) or adapter-based injection (Row 2), and removing Condition Augmentation (Row 3) degrades appearance fidelity.

Synergistic Pose Modulation Modules. Robust Motion-to-Image Alignment remains challenging in I2V due to pose inaccuracies and spatio-temporal misalignments, and Fig. 13 validates the distinct roles of our modules. For spatial issues, SSAE handles pose errors (Row 1) and FAAU addresses

[Figure 17]

Table 3: Ablation on Synergistic Pose Modulation Modules and the training stage.

|RealisDance-Val<br><br>|X-Dance|
|---|---|
|IS ↑ IB ↑ SC ↑ TF ↑ FVD ↓|IS ↑ IB ↑ SC ↑ TF ↑|

Settings

Stage 1 94.49 95.09 93.24 97.00 330.47 93.22 93.70 87.74 96.46 Stage 1 w/o SSAE 94.48 95.09 93.17 96.34 349.98 92.69 94.65 87.41 96.35

- Stage 1 w/o FFAU 94.36 95.08 93.21 96.06 339.55 92.44 93.91 87.74 96.41

- Stage 1 w/o TMCM 94.37 95.01 93.24 96.19 333.52 92.25 93.41 87.82 96.43

- Stage 2 w C-D Distillation 94.66 95.28 93.31 97.56 329.55 93.44 94.11 87.74 96.51

- Stage 2 w Normal Distillation 79.13 81.10 88.98 97.12 723.91 - - - -

- Stage 3 94.68 95.38 93.48 97.99 326.49 96.17 96.92 91.61 97.10

Figure 12: Ablation on ConditionReconciliation Mechanism.

[Figure 18]

[Figure 19]

- Figure 14: Ablation on Condition-Decoupled Distillation. Red lines are incorrectly artifacts in Normal Distillation.

[Figure 20]

- Figure 15: Ablation study on Motion Discontinuity Mitigation.

Figure 13: Ablation on Synergistic Pose Modulation Modules.

large structural disparities (Row 2) with multi-scale and reference-adaptive representations. For temporal issues, TMCM models missing or contradictory poses (Row 3) for smooth guidance, with VBench results in Table 3 further confirming all three modules.

Condition-Decoupled Distillation. To recover generation quality after motion-control learning, we introduce Condition-Decoupled Distillation in Sec. 3.5 and compare it with the first-stage model and normal MSE distillation. As shown in Figure 14, our design improves video fidelity while maintaining stability, whereas conventional distillation collapses. This failure stems from conflicts between unconditional distillation gradients and pose-conditional optimization, which gradually desensitize the model to conditional inputs. By decoupling these objectives, our method mitigates inter-branch interference, as further supported by Table 3.

Motion Discontinuity Mitigation. To address start-gap misalignment between the reference image and first pose frame, we propose Pose Simulation in Sec. 3.5. As shown in Figure 15, this discrepancy causes an abrupt baseline artifact between the second and third frames. In contrast, our Pose Imitation fine-tuning synthesizes the hand-raising motion as a smooth, plausible transition. This validates its ability to mitigate motion discontinuities, also reflected by the Temporal Flickering gains in Table 3.

#### 5 Conclusion

We present SteadyDancer, a framework for harmonized, coherent human animation that leverages first-frame preservation. It resolves the core I2V challenge of harmonizing fidelity with motion control and ensuring coherence via our novel Condition-Reconciliation Mechanism and Synergistic Pose Modulation Modules. Our Staged Decoupled-Objective Training pipeline efficiently optimizes for motion, quality, and continuity with minimal resources. Quantitative and qualitative results, especially on our X-Dance benchmark, show SteadyDancer significantly outperforms competitors. We believe these innovations provide a solid, efficient method for future robust human animation.

#### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023.
- [2] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.
- [3] Di Chang, Hongyi Xu, You Xie, Yipeng Gao, Zhengfei Kuang, Shengqu Cai, Chenxu Zhang, Guoxian Song, Chao Wang, Yichun Shi, Zeyuan Chen, Shijie Zhou, Linjie Luo, Gordon Wetzstein, and Mohammad Soleymani. X-dyna: Expressive dynamic human image animation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 5499–5509. Computer Vision Foundation / IEEE, 2025.
- [4] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation. CoRR, abs/2310.19512, 2023.
- [5] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, Zhen Shen, Yafei Song, Ke Sun, Linrui Tian, Feng Wang, Guangyuan Wang, Qi Wang, Zhongjian Wang, Jiayu Xiao, Sheng Xu, Bang Zhang, Peng Zhang, Xindi Zhang, Zhe Zhang, Jingren Zhou, and Lian Zhuo. Wan-animate: Unified character animation and replacement with holistic replication. CoRR, abs/2509.14055, 2025.
- [6] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [7] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010.
- [8] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.
- [9] Li Hu. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 8153–8163. IEEE, 2024.
- [10] Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. Animate anyone 2: High-fidelity character image animation with environment affordance. arXiv preprint arXiv:2502.06145, 2025.
- [11] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [12] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024.
- [13] Yasamin Jafarian and Hyun Soo Park. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12753–12762, 2021.
- [14] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025.
- [15] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

- [16] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Daquan Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models. CoRR, abs/2412.03603, 2024.
- [17] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, and Tianshu Hu. Dreamactor-m1: Holistic, expressive and robust human image animation with hybrid guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11036–11046, 2025.
- [18] Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. MIMO: controllable character video synthesis with spatial decomposed modeling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 21181–21191. Computer Vision Foundation / IEEE, 2025.
- [19] William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4172–4182. IEEE, 2023.
- [20] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 10674–10685. IEEE, 2022.
- [21] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015

- 18th International Conference Munich, Germany, October 5 - 9, 2015, Proceedings, Part III, pages 234–241. Springer, 2015.

- [22] Aliaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019.
- [23] Aliaksandr Siarohin, Oliver J Woodford, Jian Ren, Menglei Chai, and Sergey Tulyakov. Motion representations for articulated animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13653–13662, 2021.
- [24] Guoxian Song, Hongyi Xu, Xiaochen Zhao, You Xie, Tianpei Gu, Zenan Li, Chenxu Zhang, and Linjie Luo. X-unimotion: Animating human images with expressive, unified and identity-agnostic motion latents. CoRR, abs/2508.09383, 2025.
- [25] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.
- [26] Shuyuan Tu, Zhen Xing, Xintong Han, Zhi-Qi Cheng, Qi Dai, Chong Luo, and Zuxuan Wu. Stableanimator: High-quality identity-preserving human image animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21096–21106, 2025.
- [27] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.
- [28] Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Xiaofeng Meng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. CoRR, abs/2503.20314, 2025.
- [29] Tan Wang, Linjie Li, Kevin Lin, Yuanhao Zhai, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. Disco: Disentangled control for realistic human dance generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9326–9336, 2024.

- [30] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. CoRR, abs/2504.11289, 2025.
- [31] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.
- [32] Zhenzhi Wang, Yixuan Li, Yanhong Zeng, Youqing Fang, Yuwei Guo, Wenran Liu, Jing Tan, Kai Chen, Tianfan Xue, Bo Dai, and Dahua Lin. Humanvid: Demystifying training data for camera-controllable human image animation. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 15, 2024, 2024.
- [33] Shuolin Xu, Siming Zheng, Ziyi Wang, HC Yu, Jinwei Chen, Huaqi Zhang, Bo Li, and Peng-Tao Jiang. Hypermotion: Dit-based pose-guided human image animation of complex motions. CoRR, abs/2505.22977, 2025.
- [34] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 1481–1490. IEEE, 2024.
- [35] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28,

2025. OpenReview.net, 2025.

- [36] Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G. Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. MAGVIT: masked generative video transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 10459–10469. IEEE, 2023.
- [37] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 3813–3824. IEEE, 2023.
- [38] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.
- [39] Shiyi Zhang, Junhao Zhuang, Zhaoyang Zhang, Ying Shan, and Yansong Tang. Flexiact: Towards flexible action control in heterogeneous scenarios. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2025.
- [40] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. CoRR, abs/2406.19680, 2024.
- [41] Jian Zhao and Hui Zhang. Thin-plate spline motion model for image animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3657–3666, 2022.
- [42] Jingkai Zhou, Benzhi Wang, Weihua Chen, Jingqi Bai, Dongyang Li, Aixi Zhang, Hao Xu, Mingyang Yang, and Fan Wang. Realisdance: Equip controllable character animation with realistic hands. arXiv preprint arXiv:2409.06202, 2024.
- [43] Jingkai Zhou, Yifan Wu, Shikai Li, Min Wei, Chao Fan, Weihua Chen, Wei Jiang, and Fan Wang. Realisdance-dit: Simple yet strong baseline towards controllable character animation in the wild. CoRR, abs/2504.14977, 2025.
- [44] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision, pages 145–162. Springer, 2024.

[Figure 21]

- Figure 16: Examples from the X-Dance benchmark. The second and third rows display driving video sequences, comprising both intricate, high-dynamic dance movements and low-amplitude simple activities. The first row presents reference images, which were specifically curated relative to these driving videos to evaluate the model with real-world misalignment challenges.

### Appendix

#### A X-Dance

Standard benchmarks, such as TikTok and RealisDance, source both the reference image and pose sequence from the same video. This idealized setup fails to reflect the spatio-temporal misalignment challenges prevalent in real-world applications. As shown in Fig. 16, to more robustly evaluate the model’s generalization capabilities in such scenarios, we curated and introduced a new evaluation dataset, X-Dance. We first collected 12 distinct driving videos, comprising 8 sequences of intricate, high-dynamic dance movements and 4 sequences of low-amplitude daily activities. These sequences are replete with non-ideal real-world factors, such as motion blur, severe occlusion, and drastic pose changes. Tailored to these motions, we specifically curated a diverse set of reference images to simulate real-world misalignments. This specially designed collection contains: (1) anime characters to introduce stylistic domain gaps; (2) half-body shots to represent compositional inconsistencies;

- (3) cross-gender or anime characters to simulate significant skeletal structural discrepancies; and
- (4) subjects in distinct postures to maximize the initial action gap. By systematically pairing these reference images with the 12 driving videos, we simulate two critical real-world challenges: (1) Spatial pose-structure inconsistency (e.g., an anime character driving a real-world pose); and (2) Temporal discontinuity, specifically the significant gap between the reference pose and the initial driving pose.

#### B Model Details

##### B.1 Motion Discontinuity Mitigation.

As discussed in the main text, to address the abrupt transition between the reference frame and the initial pose frame, we propose Pose Simulation to explicitly replicate this discontinuity within the

[Figure 22]

- Figure 17: Performance comparison of four Motion Discontinuity Mitigation methods, showing that the Pose Simulation approach generates smooth and natural transitions. Notably, this method achieves this without introducing additional modules or extra inference latency.

training data. Specifically, given a smooth training sequence {p0,p1,...,pT}, we first construct synthetic pairs (p0,pT∗), where the target timestamp T∗ is randomly sampled from {2,3,4} to ensure diversity. We then interpolate between this pair to generate intermediate poses, selecting the first interpolated frame p˜1. With a probability of 0.5, we replace the original p1 with this synthetic p˜1, yielding a pseudo-training sample {p0,p˜1,p2,...,pT}. This strategy effectively mimics realistic discontinuities while preserving the integrity of the motion control signal. By directly utilizing these synthetic trajectories as training samples, the model is exposed to realistic jump patterns without requiring any architectural modifications. Notably, fine-tuning on these synthetic samples for just a few hundred steps resolves over 80% of extreme jump scenarios, all while maintaining no additional module parameters and zero additional inference latency.

Beyond our proposed strategy, we explored several alternative approaches to address the motion discontinuity:

- • Pose Warping: We inserted an explicit pose-interpolation submodule designed to generate intermediate poses bridging the reference pose and the first driving frame, effectively implementing pose warping.
- • Pose Generator: We devised a lightweight pose-sequence generator trained to synthesize intermediate frames conditioned on the start and end poses.
- • Feature Mapping: We designed a feature-mapping module that enhances each pose latent by aggregating features from its adjacent temporal neighbors.

As shown in Fig. 17, a comparison of the four methods reveals that Pose Simulation yields the best results, producing smooth and natural transitions superior to the alternatives. Furthermore, all three approaches necessitate the introduction of additional network modules, which proved difficult to

[Figure 23]

- Figure 18: Model performance using Decoupled-Condition Classifier-Free Guidance (DC-CFG). From top to bottom, the rows display the original pose (positive condition), the perturbed pose (negative condition), generation results without DC-CFG, and generation results with DC-CFG, showing our DC-CFG improves pose control and effectively suppresses generation artifacts.

optimize effectively given our limited training data. In contrast, Pose Simulation mitigates this issue from a data-centric perspective. It is perfectly suited to our limited data regime and imposes zero additional parameter overhead.

##### B.2 Implementation Details.

Training Details. As detailed in the main text, our Staged Decoupled-Objective Training Pipeline employs distinct training configurations to optimize different objectives across separate stages, all leveraging the same dataset.

- • Stage 1: Action Supervision. We utilize LoRA-based training with a learning rate of 1e-4 for 12,000 steps. Model selection in this phase prioritizes motion adherence and basic image quality.
- • Stage 2: Condition-Decoupled Distillation. We switch to full-parameter fine-tuning with a reduced learning rate of 1e-6 for 2,000 steps. The selection of the optimal checkpoint is governed by visual preference.
- • Stage 3: Motion Discontinuity Mitigation. We revert to LoRA-based training with a learning rate of 1e-4 for a brief 500 steps, focusing on leveraging the augmented training data to mitigate motion discontinuity artifacts.

Training Dataset. We curated a proprietary human motion dataset consisting of 7,338 five-second video clips, totaling 10.2 hours. Notably, this dataset scale is significantly smaller than that of comparable works, highlighting the data efficiency of our design. To ensure quality, these clips were rigorously filtered from the Internet based on aesthetic scores, motion smoothness, subject prominence, and action types. The final collection is composed of two parts: ∼2,000 clips sourced from high-quality footage (e.g., documentaries, YouTube) to introduce diversity in aspect ratios and motion dynamics; and ∼5,000 vertical videos collected primarily from social media, focusing

[Figure 24]

Figure 19: More visualization comparison on X-Dance.

predominantly on dance sequences. We purposefully excluded extreme or complex actions to maintain training stability.

##### B.3 Decoupled-Condition Classifier-Free Guidance.

Most of the existing video generation models typically employ Classifier-Free Guidance (CFG) to synthesize high-quality samples that strictly adhere to the provided conditional guidance. Specifically, during the sampling process, the model leverages its capability to predict both conditional and unconditional noise. At each denoising timestep t, we perform two forward passes through the DiT using the current noisy latent xt, one conditioned on y (ϵθ(xt,t,y)), and another conditioned on a null embedding ∅ (ϵθ(xt,t,∅)). The standard CFG noise prediction is formulated as:

ϵˆθ(xt,t,y,w) = ϵθ(xt,t,∅)

(7)

+w(ϵθ(xt,t,y) − ϵθ(xt,t,∅))

where the difference between the two predictions represents a vector in noise space that steers the generation towards the condition y. The scalar w denotes the guidance scale, determining the strength of the shift from the unconditional distribution toward the conditional one. In practice, it is common to replace the null condition ∅ with a specific negative prompt (ynegtxt ) to provide a more explicit negative constraint, thereby improving generation quality.

Within our framework, the pose signal ypose serves as a critical condition. Inspired by textual negative prompting, we propose an innovative Decoupled-Condition Classifier-Free Guidance (DC-CFG) to further enhance pose controllability. Specifically, we apply scale and shift perturbations to the extracted pose signal to construct a negative pose condition ynegpose. This explicitly simulates the

[Figure 25]

Figure 20: More visualization comparison on X-Dance.

misalignment issues we aim to avoid. Based on this, we obtain a prediction guided by the negative pose: ϵθ(xt,t,ynegpose). To effectively integrate these multiple conditions, we decouple the guidance and adjustment processes as follows:

∆ϵθ(xt,t,y,ynegpose) = ϵθ(xt,t,y) − ϵθ(xt,t,ynegpose) ∆ϵθ(xt,t,y,ynegtxt ) = ϵθ(xt,t,y) − ϵθ(xt,t,ynegtxt ) ϵˆθ(xt,t,y,wpose,wtxt) = ϵθ(xt,t,ynegtxt )

(8)

+wpose · ∆ϵθ(xt,t,y,ynegpose) + wtxt · ∆ϵθ(xt,t,y,ynegtxt )

where wpose and wtxt denote the guidance scales for the pose and text prompt, respectively. By leveraging a pose-conditioned CFG, we compel the generator to not only strictly adhere to the target pose but also actively diverge from imprecise or ambiguous neighboring poses. This mechanism significantly sharpens motion precision, effectively suppressing common artifacts such as limb blurring, motion ghosting, or regression to the mean pose, ultimately achieving higher-fidelity control of the driving motion.

Furthermore, we carefully designed a temporal scheduling strategy for pose-guided CFG, capitalizing on the inherent coarse-to-fine generation characteristic of diffusion models. The core philosophy is to modulate the guidance strength dynamically. In the early denoising stages, the model primarily constructs low-frequency components, such as global contours and spatial layout. Imposing strong guidance at this phase ensures that the character’s global pose structure is precisely anchored.

[Figure 26]

Figure 21: More visualization comparison on X-Dance.

Conversely, as the process advances to the later stages, the model shifts its focus to rendering highfrequency details, such as texture and lighting. Here, it is crucial to attenuate or remove the guidance. This grants the model sufficient degrees of freedom to synthesize photorealistic details faithful to the source appearance, avoiding visual artifacts like structural rigidity or unnatural deformations caused by over-guidance. This temporal mechanism effectively decouples structural pose control from appearance detail generation. By maximizing realism and naturalness while ensuring motion precision, it serves as a pivotal trade-off strategy for achieving high-fidelity pose-driven video generation. In practice, we apply pose-conditioned CFG exclusively within the normalized timestep interval of [0.1,0.4]. The guidance scales are configured as wpose = 1.0 for the pose condition and wtxt = 5.0 for the text prompt. Fig. 18 presents the positive and perturbed negative pose conditions along with generation results, illustrating the impact of our DC-CFG. As shown, our approach improves pose control and effectively suppresses generation artifacts.

##### B.4 Efficiency Comparison

To illustrate the comparison in terms of efficiency, we also provide a comparative analysis among Wan-based methods in the Table 4. Moreover, we support multi-GPU parallel inference and provide a near-lossless FP16 version integrated with LoRA acceleration from LightX2V to facilitate practical deployment. We will also release quantized models to further reduce overhead.

Table 4: Efficiency comparison among Wan-based Model.

|Single GPU|Two-GPU parallel|
|---|---|
|81 frame (min) Memory (GB)<br><br>|81 frame (min) Memory (GB)|

Model Params (B) UniAnimate-DiT 16.40 20.5 24 - -

VACE 17.33 OOM OOM 11.7 79 Wan-Animate 17.27 11.9 62 6.7 56

Ours 16.39 17.2 43 10.2 41 Oursoffload,FP16,LoRAblockaccelerationswap 16.39 3.7 9.2 - -

#### C More Qualitative Results

We present qualitative comparisons between our method and state-of-the-art approaches, including HumanVid [32], Hyper-Motion [33], MinicMotion [40], and Wan-Animate [5], on our challenging X-Dance dataset in Fig. 19, Fig. 20, and Fig. 21. Notably, our method not only achieves superior generation quality with first-frame preservation but also requires significantly fewer training resources.

#### D Limitation and Future Work

Despite the promising results achieved by SteadyDancer in harmonized and coherent animation, several limitations remain to be addressed. 1) Domain Gap in Stylized Images. While our model delivers visually pleasing and coherent results for anime reference frames, its performance remains slightly inferior to the exceptional fidelity achieved on real-world images, occasionally exhibiting minor degradation in stylistic consistency. This limitation stems from the under-representation of anime samples in our current training corpus. In future work, we aim to augment the training data with specialized anime datasets to bridge this domain gap and enhance the model’s generalization across different artistic styles. 2) Extreme Motion Discontinuities. Our proposed strategy effectively mitigates motion discontinuities in the vast majority of scenarios. However, in cases of extreme pose discrepancies between the reference frame and the initial driving pose, as the model prioritizes precise motion control, it may generate transitions that appear accelerated or slightly unnatural. We believe that developing more sophisticated temporal modeling architectures and scaling up the training data will be instrumental in further resolving this start-gap challenge. 3) Advanced Motion Representation. The current architecture relies heavily on the accuracy of the input pose sequence; for instance, consecutive pose estimation errors can lead to irreversible artifacts in the generated video. We argue that an ideal animation system should balance precise controllability with high error tolerance. Therefore, designing a more refined, diverse, and semantically rich motion representation remains a promising direction for future research.

#### E Broader Impact

SteadyDancer can benefit creative applications such as film production, advertising, game development, virtual content creation, and digital human animation by reducing the resources required for high-fidelity motion-driven video generation. However, as with other human image animation and video generation technologies, it may also be misused to animate a person’s image without consent or to create misleading synthetic media. We therefore encourage responsible use of the released code and model resources, including obtaining appropriate consent for identity-driven generation, avoiding deceptive or harmful applications, and following applicable data, copyright, and platform usage policies. The released code and model resources will be provided under a usage license that prohibits unauthorized identity animation, deceptive synthetic media generation, and other harmful applications.

#### F Existing Assets

We use and cite existing models, datasets, benchmarks, and evaluation tools, including Wan-2.1, TikTok, RealisDance-Val, VBench/VBench++, and baseline methods. These assets are used for research and evaluation purposes following their respective licenses and terms of use. Our proprietary training data are not released and are used under applicable data usage restrictions.

