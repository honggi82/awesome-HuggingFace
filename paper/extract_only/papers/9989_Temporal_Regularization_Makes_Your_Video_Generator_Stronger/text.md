# arXiv:2503.15417v1[cs.CV]19Mar2025

## Temporal Regularization Makes Your Video Generator Stronger

Harold Haodong Chen1,2 Haojian Huang1,4 Xianfeng Wu1,2 Yexin Liu1,2 Yajing Bai1,2 Wen-Jie Shu1,2 Harry Yang1,2 Ser-Nam Lim1,3

1Everlyn AI 2HKUST 3UCF 4HKU

Project page:https://haroldchen19.github.io/FluxFlow/

Static Dynamic Blur and Wrong Clear and Correct

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

CogVideoX

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

w/ FluxFlow

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

CogVideoX

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

w/ FluxFlow

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Small Motion Large Motion Implausible Plausible

Figure 1. FLUXFLOW improves the temporal quality of video generators. Captions: (Top) A dog chasing a butterfly in a garden, with the butterfly flying in random directions. (Bottom) A person is running along a beach with waves crashing in the background.

#### Abstract

Temporal quality is a critical aspect of video generation, as it ensures consistent motion and realistic dynamics across frames. However, achieving high temporal coherence and diversity remains challenging. In this work, we explore temporal augmentation in video generation for the first time, and introduce FLUXFLOW for initial investigation, a strategy designed to enhance temporal quality. Operating at the data level, FLUXFLOW applies controlled temporal perturbations without requiring architectural modifications. Extensive experiments on UCF-101 and VBench benchmarks

demonstrate that FLUXFLOW significantly improves temporal coherence and diversity across various video generation models, including U-Net, DiT, and AR-based architectures, while preserving spatial fidelity. These findings highlight the potential of temporal augmentation as a simple yet effective approach to advancing video generation quality.

#### 1. Introduction

The pursuit of photorealistic video generation faces a critical dilemma: while spatial synthesis (e.g., SD-series [9, 28],

AR-based [22, 40]) has achieved remarkable fidelity, ensuring temporal quality remains an elusive target. Modern video generators, whether diffusion [4, 20, 21, 44, 48] or autoregressive [7, 15, 37], frequently produce sequences plagued by temporal artifacts, e.g., flickering textures, discontinuous motion trajectories, or repetitive dynamics, exposing their inability to model temporal relationships robustly (see Figure 1).

These artifacts stem from a fundamental limitation: despite leveraging large-scale datasets, current models often rely on simplified temporal patterns in the training data (e.g., fixed walking directions or repetitive frame transitions) rather than learning diverse and plausible temporal dynamics. This issue is further exacerbated by the lack of explicit temporal augmentation during training, leaving models prone to overfitting to spurious temporal correlations (e.g., “frame #5 must follow #4”) rather than generalizing across diverse motion scenarios.

Unlike static images, videos inherently require models to reason about dynamic state transitions rather than isolated frames. While spatial augmentations (e.g., cropping, flipping, or color jittering) [45, 46] have proven effective for improving spatial fidelity in visual generation, they fail to address the temporal dimension, making them inadequate for video generation. As a result, video generation models often exhibit two key issues (as shown in Figure 1): ❶ Temporal inconsistency: Flickering textures or abrupt transitions between frames, indicating poor temporal coherence (see Figure 4). ❷ Similar temporal patterns: Overreliance on simplified temporal correlations leads to limited temporal diversity, where generated videos struggle to distinguish between distinct dynamics, such as fast and slow motion, even with explicit prompts (see Figure 5). Addressing these challenges requires balancing spatial realism—textures, lighting, and object shapes—with temporal plausibility—coherent and diverse transitions. While modern architectures leverage large-scale image priors for spatial realism [9, 28, 34], they struggle with complex temporal relationships, relying heavily on architectural modifications [1, 10, 14] or constraint engineering [5, 15, 38]. However, data-level augmentation, proven effective in video understanding [2, 18, 42, 49], remains underexplored, highlighting the untapped potential of temporal data augmentation for improving video generation.

To make an initial exploration of this issue, in this paper, we propose FLUXFLOW, a data augmentation strategy that injects controlled temporal perturbations into video generation training. Inspired by human cognition—where we infer missing frames or reorder events—FLUXFLOW operates on a simple principle: disrupting fixed temporal order to force the model to learn disentangled motion/optical flow dynamics. Specifically, FLUXFLOW introduces two levels of temporal perturbations for investigation:

###### Temporal Quality

99.64

- 95

- 96

- 97

- 98

- 99

- 100

VideoCrafter2

99.28

98.82

w/ FluxFlow

98.64

98.22 98.41

Scores(%)

97.73

96.85

Subject Consistency

Background Consistency

Temporal Flickering

Motion Smoothness

###### Frame-wise & Overall Quality

84.48

85

VideoCrafter2

82.36

82.20

80.44

w/ FluxFlow

80

Scores(%)

75

70

67.94

67.22

63.55

65

63.13

60

Aesthetic Quality

Imaging Quality

Quality Score

Total Score

Figure 2. Comparison of VideoCrafter2 with FLUXFLOW using VBench metrics for Temporal Quality (Top) and Frame-wise and Overall Quality (Bottom). FLUXFLOW significantly enhances the temporal quality of generated videos while maintaining or even improving frame-wise and overall quality.

➮ Frame-Level: Randomly shuffle individual frames to disrupt fixed temporal order, encouraging the model to infer plausible temporal relationships.

➮ Block-Level: Reorder contiguous-frame blocks to simulate realistic temporal disruptions while preserving coarse motion patterns.

By training on disordered sequences, the generator learns to recover plausible trajectories, effectively regularizing temporal entropy. FLUXFLOW bridges the gap between discriminative and generative temporal augmentation, offering a plug-and-play enhancement solution for temporally plausible video generation while improving overall quality (see Figure 1 and 2). Unlike existing methods that introduce architectural changes or rely on post-processing, FLUXFLOW operates directly at the data level, introducing controlled temporal perturbations during training. To summarize, our contributions are as follows:

- • We introduce FLUXFLOW, the first dedicated temporal augmentation strategy for video generation, which introduces controlled temporal perturbations without requiring architectural modifications.
- • We identify and formalize the challenge of temporal brittleness in video generation, highlighting the lack of explicit temporal augmentation in existing methods and demonstrating the potential of temporal augmentation as a simple yet viable solution.
- • Extensive experiments across UCF-101 [33] and VBench [19] benchmarks on diverse video generators (U-Net [5], DiT [44], and AR-based [7]) demonstrate that FLUXFLOW enhances temporal coherence without compromising spatial fidelity.

We hope this work inspires broader explorations of temporal augmentation strategies in video generation and beyond.

Frame Order

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

1 2

[Figure 41]

[Figure 42]

- 1 2 3 4 5 6 ... N-2 N-1 N

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

- 1 3 4 5 6 ... N-2 N

InputOutput

[Figure 53]

[Figure 54]

3 N

[Figure 55]

...

[Figure 56]

FluxFlow-Frame N-1

[Figure 57]

[Figure 58]

[Figure 59]

Video Generator

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

2

[Figure 67]

- 2×1

2×4

- 2×2

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

...

- N-2

[Figure 72]

1 3

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

6 ... N-2

[Figure 78]

[Figure 79]

[Figure 80]

...

N-3

5 2

7 N

[Figure 81]

[Figure 82]

1

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

5 6 ... N

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

FluxFlow-Block

4

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

- N-3 6 ... 2 4

[Figure 97]

2

[Figure 98]

3 4 3 6 ... N 5

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

N-1

4×1

- (a) Regular Video Generation Training
- (b) Video Generation Training w/ FluxFlow

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

1 2 3 4 5 6 7 ... N

[Figure 107]

[Figure 108]

[Figure 109]

×1

[Figure 110]

2

FluxFlow

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

1

[Figure 115]

[Figure 116]

N 3 2

[Figure 117]

InputOutput

[Figure 118]

[Figure 119]

2 N-2 N-1

...

3 N-2 N-1 N 5

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Video Generator

3

### ...

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

...

...

[Figure 130]

... ...

[Figure 131]

...

[Figure 132]

...

[Figure 133]

...

... ...

[Figure 134]

[Figure 135]

[Figure 136]

... ...

4×

(c) FluxFlow: Frame-level and Block-level

4

Figure 3. Overview of FLUXFLOW. (a) Standard video generation trains on fixed frame orders, which may limit the model’s ability to learn temporal dynamics. (b) FLUXFLOW introduces controlled temporal perturbations during training as a plug-and-play augmentation strategy. (c) This study explores FLUXFLOW at two levels: frame-level (top) and block-level (bottom). In frame-level, Num × 1 denotes the number of individual frames shuffled. In block-level, Num1 × Num2 represents a block comprising Num2 consecutive frames.

#### 2. Related Work

based, and Autoregressive (AR)-based. This section provides an overview of (Latent) Diffusion Models for U-Net and DiT, and Next Token Prediction for AR-based methods.

Video Generation. Advancements in video generation span T2V [5, 10, 14, 15, 20, 21, 36] and I2V [1, 20, 23, 41]. T2V generates videos aligned with textual descriptions, while I2V focuses on temporally coherent output conditioned on images. Beyond per-frame quality, ensuring temporal quality remains a key challenge.

Diffusion Models (DMs) [12, 32] are probabilistic generative frameworks that gradually corrupt data x0 ∼ pdata(x) into Gaussian noise xT ∼ N(0,I) via a forward process, and subsequently learn to reverse this process through denoising. The forward process q(xt|x0,t), defined over T timesteps, progressively adds noise to the original data x0 to obtain xt, leveraging a parameterization trick. Conversely, the reverse process pθ(xt−1|xt,t) denoises xt to recover xt−1 using a denoising network ϵθ (xt,t). The training objective is formulated as follows:

Temporal Refinement for Video Generation. Modern approaches to temporal refinement can be categorized into three main paradigms: (i) Architecture-Centric Modeling: Spatiotemporal transformers [14, 21], hybrid 3D convolutions [1], and motion-decoupled architectures [10] improve long-range coherence but increase computational cost. (ii) Physics-Informed Regularization: Techniques like optical flow warping [38], surface normal prediction [5], and motion codebooks [15] ensure realistic motion through physical priors. (iii) Training Dynamics Optimization: Temporal contrastive loss [47], curriculum frame sampling [23], and dynamic FPS sampling [20] enhance robustness and consistency. While these methods have advanced architectural designs and constraint engineering, they often overlook the potential of systematic temporal augmentation within video data itself. Our work addresses this gap by introducing simple yet effective temporal augmentation strategies, paving the way for improved temporal quality in video generation.

data,ϵ∼N(0,I)∥ϵ − ϵθ (xt;c,t)∥22, (1) where ϵ represents the ground-truth noise, θ denotes the learnable parameters, and c is an optional conditioning input. Once trained, the model generates data x0 by iteratively denoising a random Gaussian noise xT.

Et,x∼p

min

θ

Latent Diffusion Models (LDMs) [13, 28] extend DMs by operating in a compact latent space, significantly improving computational efficiency. Instead of performing the diffusion process in the pixel space, LDMs encode the input video x ∈ RL×3×H×W into a latent representation z = E(x) using an autoencoder E, where z ∈ RL×C×h×w. The diffusion process zt = p(z0,t) and the denoising process zt = pθ(zt−1,c,t) are then conducted in the latent space. The training objective is similar to DMs but applied to the latent representation:

#### 3. Methodology

##### 3.1. Preliminaries

Modern video generation models fall into three main paradigms: U-Net-based, Diffusion Transformer (DiT)-

data,ϵ∼N(0,I)∥ϵ − ϵθ (E(xt);c,t)∥22, (2)

Et,x∼p

Finally, the generated latent representation z is decoded back into the pixel space using the decoder D, yielding the generated video xˆ = D(z).

Next Token Prediction. AR video generation can be formulated as next-token prediction, similar to language modeling. A video is converted into a sequence of discrete video tokens T = {t1,t2,...,tn} by tokenizers. Similar to LLMs, the next video token is predicted using past video tokens as context. Specifically, the training objective is to minimize the following negative log-likelihood (NLL) loss:

−log P(ti|t1,t2,...,ti−1;Θ), (3)

LNLL =

i

where the conditional probability P of the predicted next ti is modeled by a transformer decoder with parameters Θ.

- 3.2. FLUXFLOW While spatial augmentations (e.g., flipping, cropping) are commonly employed to enhance spatial robustness, the temporal dimension remains under-regularized in video generation. To address this gap, we propose FLUXFLOW, a data-level temporal augmentation strategy that perturbs the temporal structure of video sequences during training. In this initial exploration, FLUXFLOW operates in two modes: Frame-level and Block-level Perturbations, each targeting distinct temporal scales, as demonstrated in Figure 3.

Frame-Level Perturbations. FLUXFLOW-FRAME introduces fine-grained disruptions by shuffling individual frames within a sequence. As shown in Figure 3(c) (top), given a video sequence V = {F1,F2,...,FN}, we randomly shuffle a subset of frames, controlled by the perturbation ratio α. Formally:

Vframe = Shuffle({Fi | i ∈ S}) + {Fj | j ∈/ S}, (4) where S is a randomly selected subset of frames with |S| = ⌊αN⌋. Frames outside S remain in their original positions, maintaining partial temporal consistency. This perturbation forces the model to reconstruct plausible temporal relationships, enhancing its ability to generalize beyond deterministic frame-to-frame dependencies.

Block-Level Perturbations. FLUXFLOW-BLOCK operates at a coarser scale by reordering contiguous blocks of frames, as illustrated in Figure 3(c) (bottom). The input sequence V is divided into M non-overlapping blocks of size k, such that:

Vblock = {B1,B2,...,BM}, (5)

where Bm = {F(m−1)k+1,...,Fmk}. A subset B of these blocks is then randomly reordered with a probability β, producing:

Vblockperturbed = Reorder({Bm | m ∈ B}) + {Bn | n ∈/ B}.

(6) Block-level perturbations simulate realistic temporal disruptions, such as changes in motion speed or direction,

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

###### w/ FluxFlow

[Figure 145]

- 0.8
- 1.0
- 1.2

1.4

1.6

AngleDifference

w/ FluxFlow CogVideoX

High Variance

CogVideoX

Figure 4. Illustration of FLUXFLOW in enhancing temporal coherence. (Top) Example frames from CogVideoX, without and with FLUXFLOW, showcasing larger motion dynamics in the latter. (Bottom) Comparison of temporal angle differences across frames. FLUXFLOW achieves consistently lower angle differences, indicating improved temporal coherence over the base model. Caption: A skateboarder performing tricks in a skatepark, with fastpaced movements and dynamic camera angles.

while preserving coarse motion patterns.

Implementation. FLUXFLOW is implemented as a preprocessing strategy applied during training. Each perturbation (frame-level or block-level) is independently applied to evaluate its impact on temporal quality. Figure 3(b) illustrates the combined training pipeline. A concrete illustration of the algorithm is given in the pseudocode below.

Algorithm 1 FLUXFLOW Pseudocode

Require: Video V = {F1,F2,...,FN}, perturbation type mode ∈ {frame,block}, perturbation ratio α (for frame-level), block size k and perturbation probability β (for block-level)

Ensure: Perturbed sequence VFluxFlow

- 1: if mode = frame: ▷ Frame-Level Perturbations
- 2: Select subset S of frames with |S| = ⌊αN⌋
- 3: Shuffle frames in S to obtain VFluxFlow
- 4: else if mode = block: ▷ Block-Level Perturbations
- 5: Divide V into M = ⌊N/k⌋ blocks {B1,B2,...,BM}
- 6: Select subset B of blocks with |B| = ⌊βM⌋
- 7: Reorder blocks in B to obtain VFluxFlow
- 8: end if
- 9: Output: VFluxFlow

- 3.3. What does model learn with FLUXFLOW? To better understand the impact of FLUXFLOW on the model’s temporal learning capabilities, we evaluate its effect on temporal coherence and temporal diversity. For this purpose, we select three groups of text prompts with varying temporal dynamics: static, slow, and fast (details can be found in Appendix §A). Our observations are as follows:

0 10 20 30 40

Frame Index

[Figure 146]

[Figure 147]

0.5

[Figure 148]

Static Fast Slow

[Figure 149]

Static Fast Slow

0.5

0.0

- -3.0
- -2.0

0.0

2.0 1.5 1.0 0.5 0.0 -0.5 -1.0 -1.5 -2.0 PC1

- -1.0 -1.0
- -1.5

- -2.0
- -2.5
- -3.0
- -2.5
- -2.0
- -1.5
- -1.0
- -0.5

- -0.5

- -1.0
- -1.5
- -2.5
- -2.5
- -2.0
- -1.5

PC3

PC3

PC2

PC2

2.0 1.5 1.0 0.5 0.0 -0.5 -1.0 -1.5 PC1

(a) w/o FluxFlow (b) w/ FluxFlow

Figure 5. Illustration of FLUXFLOW in improving temporal feature diversity. (a) Without FLUXFLOW, the model trained on fixed original frame sequences fails to distinguish features across different temporal paradigms. (b) With FLUXFLOW, features are more distinctly separated, reflecting enhanced temporal representation.

Obs.❶ FLUXFLOW enhances temporal coherence. As shown in Figure 4, we analyze videos generated from one of the “fast” prompts. Videos generated without FLUXFLOW exhibit abrupt and unstable temporal changes, reflecting inconsistent motion dynamics. In contrast, the videos generated with FLUXFLOW demonstrate significantly larger and smoother motion dynamics. Quantitative analysis of angular differences further supports this observation. By comparing angular differences between consecutive frames, we observe that the base model produces high variance in these differences, reflecting erratic temporal transitions. In comparison, FLUXFLOW achieves consistently lower angular differences, indicating its ability to stabilize temporal changes while maintaining the intended motion dynamics.

Obs.❷ FLUXFLOW improves temporal diversity. Figure 5 demonstrates the generated videos’ temporal feature representations. Without FLUXFLOW (Figure 5(a)), the features of videos generated from different temporal prompts (static, slow, and fast) are largely overlapped, indicating the model struggles to distinguish between distinct temporal paradigms. This lack of separation reflects the baseline model’s inability to capture diverse temporal dynamics. In contrast, with FLUXFLOW (Figure 5(b)), the temporal features are more distinctly separated across the three temporal paradigms, reflecting the model’s enhanced ability to represent diverse temporal patterns.

These findings highlight the critical role of FLUXFLOW in improving the temporal capabilities of baseline models, allowing them to generate temporally consistent and diverse videos that align more closely with the intended motion dynamics of the input prompts.

#### 4. Experiment

In this section, we conduct extensive experiments to answer the following research questions (RQ): RQ1: Can FLUXFLOW improve temporal quality while

maintaining spatial fidelity?

- RQ2: Does FLUXFLOW facilitate the learning of motion/optical flow dynamics?
- RQ3: Can FLUXFLOW maintain temporal quality in extraterm generation?
- RQ4: How sensitive is FLUXFLOW to its key hyperparameters?

##### 4.1. Experimental Settings

Base Models. To comprehensively evaluate the effectiveness of FLUXFLOW, we apply it to three distinct video generation architectures: (i) U-Net-based: VideoCrafter2 [41]. (ii) AR-based: NOVA-0.6B [7]. (iii) DiT-based: CogVideoX-2B [44]. To ensure fair and consistent comparisons, we fine-tune base models using FLUXFLOW as an additional training stage with one epoch on OpenVidHD0.4M [27], following their default configurations (e.g., resolution, frame length). The results are compared with models trained under identical settings but without temporal augmentation (i.e., w/o FLUXFLOW). Notably, FLUXFLOW is model-agnostic and can be seamlessly integrated into the training pipeline of any video generation architecture.

Evaluations. We evaluate FLUXFLOW on two widelyused benchmarks for video generation, focusing on both temporal coherence and overall video quality:

- • UCF-101 [33]: A large-scale human action dataset containing 13,320 videos across 101 action classes. We utilize the following metrics:

- (i) Fr´echet Video Distance (FVD) [35] for temporal coherence and motion realism.
- (ii) Inception Score (IS) [29] for frame-level quality and diversity.

- • VBench [19]: A comprehensive benchmark designed to evaluate video generation quality across 16 dimensions. To specifically assess temporal and frame-level quality, we focus on the following key dimensions:

- (i) Temporal Quality: Subject Consistency, Background Consistency, Temporal Flickering, Motion Smoothness, and Dynamic Degree.
- (ii) Frame-Wise Quality: Aesthetic Quality and Imaging Quality.
- (iii) Overall Quality: Total Score, Quality Score, and Semantic Score.

These benchmarks and metrics provide a comprehensive evaluation, allowing us to rigorously assess the impact of FLUXFLOW on both temporal dynamics and spatial fidelity.

##### 4.2. Quality and Fidelity Enhancement (RQ1)

We present the quantitative comparison of FLUXFLOWFRAME and FLUXFLOW-BLOCK on VideoCrafter2 (VC2), NOVA, and CogVideoX (CVX) in Tab. 1 and 2 and qualitative comparison in Fig. 6. Each model is evaluated with three settings based on its default frame length. Specifically, FLUXFLOW-FRAME is shown on VC2, NOVA, and CVX

Table 1. Evaluation of FLUXFLOW-FRAME. “+ Original” refers to training without FLUXFLOW, while “+ Num × 1” indicates the use of different FLUXFLOW-FRAME strategies. We shade the best results and underline the second-best results for each model.

UCF-101 VBench FVD↓ IS↑ Subject↑ Back.↑ Flicker↑ Motion↑ Dynamic↑ Aesthetic↑ Imaging↑ Quality↑ Semantic↑ Total↑

Method

U-Net-based: 16F×320×512 VideoCrafter2 [5] 463.80 36.57 96.85 98.22 98.41 97.73 42.50 63.13 67.22 82.20 73.42 80.44

+ Original 468.32↑4.52 37.13↑0.56 97.02↑0.17 97.89↓0.33 97.17↓1.24 97.78↑0.05 41.24↓1.26 63.87↑0.74 68.01↑0.79 81.81↓0.39 73.14↓0.28 80.08↓0.36

- + 2 × 1 444.59↓19.21 37.89↑1.32 98.82↑1.97 99.28↑1.06 99.64↑1.23 98.63↑0.90 49.58↑7.08 63.55↑0.42 67.94↑0.72 84.48↑2.28 73.89↑0.47 82.36↑1.92

+ 4 × 1 451.43↓12.37 37.02↑0.45 97.90↑1.05 99.15↑0.93 98.66↑0.25 98.66↑0.93 50.00↑7.50 61.74↓1.39 65.76↓1.46 83.31↑1.11 73.39↓0.03 81.33↑0.89 + 8 × 1 457.21↓6.59 37.92↑1.35 97.93↑1.08 98.71↑0.49 98.69↑0.28 98.92↑1.19 47.25↑4.75 60.97↓2.16 66.20↓1.02 83.11↑0.91 72.37↓1.05 80.96↑0.52 AR-based: 33F×480×768

NOVA [7] 428.12 38.44 94.71 94.81 96.38 96.34 54.35 54.52 66.21 78.96 76.57 78.48

+ Original 427.42↓0.70 39.49↑1.05 95.12↑0.41 94.54↓0.27 95.88↓0.50 96.45↑0.11 52.23↓2.12 54.89↑0.37 67.04↑0.83 78.84↓0.12 76.87↑0.30 78.37↓0.11

- + 2 × 1 420.17↓7.95 38.71↑0.27 96.18↑1.47 95.56↑0.75 96.87↑0.49 97.40↑1.06 58.64↑4.29 54.22↓0.30 66.86↑0.65 80.52↑1.56 76.11↓0.46 79.64↑1.16

+ 4 × 1 413.45↓14.67 39.31↑0.87 96.76↑2.05 96.24↑1.43 97.45↑1.07 97.21↑0.87 57.88↑3.53 54.96↑0.44 66.50↑0.29 80.91↑1.95 76.84↑0.27 80.10↑1.62 + 16 × 1 423.09↓5.03 39.24↑0.89 95.24↑0.53 94.57↓0.24 97.12↑0.74 97.52↑1.18 56.54↑2.20 54.18↓0.34 65.69↓0.52 79.97↑1.01 75.28↓1.29 79.03↑0.55 DiT-based: 49F×480×720

CogVideoX [44] 347.59 44.32 96.78 96.63 98.89 97.73 59.86 60.82 61.68 82.18 75.83 80.91

+ Original 349.34↑1.75 45.91↑1.59 96.82↑0.04 95.34↓1.29 98.83↓0.06 97.31↓0.42 60.16↑0.30 58.52↓2.30 62.25↑0.57 81.43↓0.76 75.96↑0.13 80.34↓0.57

- + 2 × 1 343.23↓4.36 44.12↓0.20 97.32↑0.54 97.15↑0.52 99.14↑0.25 98.20↑0.47 61.26↑1.40 60.74↓0.08 61.96↑0.28 82.88↑0.70 75.98↑0.15 81.50↑0.59

+ 8 × 1 329.41↓18.18 46.09↑1.77 98.35↑1.57 97.98↑1.35 99.62↑0.73 98.24↑0.51 61.14↑1.28 61.54↑0.72 62.02↑0.34 83.58↑1.40 76.09↑0.26 82.08↑1.17 + 24 × 1 345.19↓2.40 44.98↑0.66 98.04↑1.26 97.09↑0.46 98.96↑0.07 98.11↑0.38 62.15↑2.29 59.82↓1.00 60.21↓1.47 82.53↑0.35 74.29↓1.54 80.88↓0.03

Table 2. Evaluation of FLUXFLOW-BLOCK. “+ Num1 × Num2” indicates the use of different FLUXFLOW-BLOCK strategies.

Method

UCF-101 VBench FVD↓ IS↑ Subject↑ Back.↑ Flicker↑ Motion↑ Dynamic↑ Aesthetic↑ Imaging↑ Quality↑ Semantic↑ Total↑

U-Net-based: 16F×320×512 VideoCrafter2 [5] 463.80 36.57 96.85 98.22 98.41 97.73 42.50 63.13 67.22 82.20 73.42 80.44

+ Original 468.32↑4.52 37.13↑0.56 97.02↑0.17 97.89↓0.33 97.17↓1.24 97.78↑0.05 41.24↓1.26 63.87↑0.74 68.01↑0.79 81.81↓0.39 73.14↓0.28 80.08↓0.36

- + 2 × 2 449.30↓14.50 37.76↑1.19 98.32↑1.47 98.72↑0.50 99.27↑0.86 98.63↑0.90 48.85↑6.35 63.84↑0.71 68.17↑0.95 84.14↑1.94 73.45↑0.03 82.00↑1.56

+ 2 × 4 457.39↓14.41 37.86↑1.29 97.59↑0.74 99.18↑0.96 98.88↑0.47 98.85↑1.12 47.24↑4.74 63.24↑0.11 67.64↑0.42 83.76↑1.56 73.67↑0.25 81.74↑1.30 + 4 × 4 460.31↓14.67 36.41↓0.16 97.23↑0.38 98.45↑0.23 98.92↑0.51 98.63↑0.90 47.90↑5.40 62.86↓0.27 66.90↓0.32 83.32↑1.12 72.08↓1.34 81.08↑0.64 AR-based: 33F×480×768

NOVA [7] 428.12 38.44 94.71 94.81 96.38 96.34 54.35 54.52 66.21 78.96 76.57 78.48

+ Original 427.42↓0.70 39.49↑1.05 95.12↑0.41 94.54↓0.27 95.88↓0.50 96.45↑0.11 52.23↓2.12 54.89↑0.37 67.04↑0.83 78.84↓0.12 76.87↑0.30 78.37↓0.11

- + 2 × 2 423.19↓4.93 39.12↑0.68 96.24↑1.53 95.48↑0.67 96.89↑0.51 97.05↑0.71 56.53↑2.18 54.24↓0.28 66.54↑0.33 80.13↑1.17 76.22↓0.35 79.35↑0.87

+ 4 × 4 417.99↓10.13 39.14↑0.70 96.89↑2.18 95.68↑0.87 97.21↑0.83 96.89↑0.55 57.12↑2.78 55.02↑0.50 66.29↑0.08 80.47↑1.51 76.97↑0.75 79.76↑1.28 + 4 × 8 + 1 425.04↓3.08 39.38↑0.94 96.53↑1.82 95.92↑1.11 97.04↑0.66 97.24↑0.90 56.86↑2.34 54.74↑0.22 66.86↑0.65 80.59↑1.63 76.75↑0.18 79.82↑1.34 DiT-based: 49F×480×720

CogVideoX [44] 347.59 44.32 96.78 96.63 98.89 97.73 59.86 60.82 61.68 82.18 75.83 80.91

+ Original 349.34↑1.75 45.91↑1.59 96.82↑0.04 95.34↓1.29 98.83↓0.06 97.31↓0.42 60.16↑0.30 58.52↓2.30 62.25↑0.57 81.43↓0.76 75.96↑0.13 80.34↓0.57

- + 2 × 2 341.78↓5.81 45.65↑1.33 97.14↑0.36 97.25↑0.62 99.21↑0.32 98.05↑0.32 61.36↑1.50 59.32↓1.50 62.86↑1.18 82.74↑0.56 75.85↑0.02 81.36↑0.45

+ 4 × 8 336.27↓11.32 45.93↑1.61 98.38↑2.05 97.81↑1.18 99.04↑0.15 98.46↑0.73 61.85↑1.99 61.02↑0.20 62.18↑0.50 83.42↑1.24 76.36↑0.53 82.01↑1.10 + 4 × 12 + 1 345.98↓1.61 45.19↑0.87 97.47↑0.69 97.04↑0.41 99.18↑0.29 98.68↑0.95 61.19↑1.33 59.64↓1.18 61.98↑0.30 82.98↑0.80 75.92↑0.09 81.57↑0.66

with 2 × 1, 4 × 1, and 8 × 1 in the qualitative comparisons, respectively. We give the following observations:

Obs.❸ FLUXFLOW improves temporal quality with preserved spatial fidelity. Both FLUXFLOW-FRAME and FLUXFLOW-BLOCK significantly improve temporal quality, as evidenced by the metrics in Tabs. 1, 2 (i.e., FVD, Subject, Flicker, Motion, and Dynamic) and qualitative results in Fig. 6. For instance, the motion of the drifting car in VC2, the cat chasing its tail in NOVA, and the surfer riding a wave in CVX become noticeably more fluid with FLUXFLOW. Importantly, these temporal improvements are achieved without sacrificing spatial fidelity, as evidenced by the sharp details of water splashes, smoke trails, and wave textures, along with spatial and overall fidelity metrics.

Obs.❹ Optimal temporal perturbation strength is model-specific. The ideal perturbation strength depends

on the base model’s default frame length. For example, in Tab. 1, the 16-frame VC2 performs best with the 2×1 strategy, while the 49-frame CVX benefits most from 8 × 1. Excessive perturbation, however, may disrupt spatial consistency, highlighting the importance of selecting modelspecific perturbation during training.

Obs.❺ Frame-level perturbations outperform blockLevel. While both frame-level and block-level perturbations improve temporal quality, frame-level generally delivers better results. This can be attributed to their finer granularity, which allows for more precise temporal adjustments. In contrast, block-level perturbations may introduce excessive noise due to stronger spatiotemporal correlations within blocks, limiting their effectiveness. As a result, frame-level strategies yield smoother and more coherent motion transitions.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

VideoCrafter2

VideoCrafter2

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

###### w/ FluxFlow

w/ FluxFlow

"A person jumping off a diving board into a pool, creating a big splash."

"A car drifting around a sharp corner on a racetrack, with smoke and dust trailing behind."

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

NOVA

NOVA

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

w/ FluxFlow

w/ FluxFlow

"A basketball player dribbling the ball, performing a crossover, and taking a jump shot."

"A cat chasing its tail in circles, repeating the motion over and over."

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

###### CogVideoX

CogVideoX

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

w/ FluxFlow

w/ FluxFlow

"Two children playing with a ball in a park, tossing it back and forth while running around."

"A surfer riding a large wave, with spray flying off the crest and seagulls circling overhead."

Figure 6. Qualitative results of FLUXFLOW on VideoCrafter2 [5] (Top), NOVA [7] (Middle), and CogVideoX [44] (Bottom).

##### 4.3. User Study with Temporal Dynamics (RQ2)

To answer RQ2, we first refer to Fig. 4, which highlights FLUXFLOW’s ability to capture smooth and coherent optical flow changes, particularly in complex motion scenarios, and Fig. 6, which demonstrates its superior motion realism and dynamics. Building on these findings, we further conduct a user study (Fig. 8) on 20 video-pairs to evaluate subjective perceptions of motion quality across five dimensions: Motion Diversity, Motion Realism, Motion Smoothness, Temporal Coherence, and Optical Flow Consistency, using prompts of two types: Action Speed (Fast & Slow) and Motion Pattern (Linear & Nonlinear). We observe that:

Obs.❻ FLUXFLOW significantly facilitates temporal dynamics learning. As shown in Fig. 8, FLUXFLOW effectively disentangles and learns motion dynamics, excelling in complex trajectories and rapid temporal variations. Specifically, (i) Motion Diversity: Broader and more varied motion trajectories, particularly in dynamic or nonlinear scenarios. (ii) Optical Flow Consistency: Smoother and more coherent transitions, reducing abrupt changes and artifacts. (iii) Motion Realism and Smoothness: More natural and fluid motion, especially in intricate and complex trajectories. (iv) Temporal Coherence: Stable frame-to-frame dynamics without compromising other dimensions.

##### 4.4. Extra-term Temporal Quality (RQ3)

To answer RQ3 and evaluate whether FLUXFLOW can maintain temporal quality in extreme conditions, we specifically use the 16-frame VC2 to generate 128-frame videos, as shown in Fig. 9. This allows us to verify whether FLUXFLOW can overcome the cumulative error and temporal instability challenges commonly observed in longsequence generation. We give the following observations:

Obs.❼ FLUXFLOW effectively preserves temporal quality under extreme conditions. As shown in Fig. 9, the qualitative comparison (top) demonstrates that FLUXFLOW maintains dynamic background consistency and generates smoother transitions, while the baseline (VC2) exhibits temporal artifacts, e.g., flickering and motion inconsistency. Quantitatively (bottom), the gray regions highlight score drops relative to the original 16-frame generation. FLUXFLOW significantly reduces these drops, achieving superior subject consistency, background consistency, temporal flickering, and motion smoothness scores, ensuring high temporal quality in extra-term scenarios.

##### 4.5. Ablation & Sensitivity Analysis (RQ4)

To better investigate the effectiveness of FLUXFLOW, we conduct two ablation studies to assess its sensitivity to shuffle interval constraints and perturbation degrees in Fig. 7: (i)

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

FluxFlow-Frame FluxFlow-Block

VC2 w/ FluxFlow NOVA w/ FluxFLow

2x1 4x1 8x1 12x1 16x1 2x1 4x1 16x1 24x1 33x1

(a) FluxFlow-Frame Interval (b) FluxFlow-Block Interval (c) 16-Frame Perturbation (d) 33-Frame Perturbation

- Figure 7. Ablation and sensitivity analysis on FLUXFLOW with VBench temporal metrics. (a, b) Impact of shuffle interval constraints on VC2 using 2 × 1 and 2 × 2 configurations. (c, d) Impact of perturbation degrees on 16-frame VC2 and 33-frame NOVA.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

w/ FluxFlow

CogVideoX

[Figure 211]

[Figure 212]

[Figure 213]

- Figure 8. User study results comparing CVX and w/ FLUXFLOW. (Top) Examples frames from a non-linear motion pattern, where FLUXFLOW demonstrates superior handling of complex trajectories. Caption: A fish swims in circular loops in a clear blue pond. (Bottom) User ratings across temporal dynamics evaluation criteria. For more details please refer to Appendix §A.

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

###### VideoCrafter2

#F1 #F42 #F84 #F128

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

w/ FluxFlow

#F1 #F42 #F84 #F128

[Figure 222]

[Figure 223]

Figure 9. Performance comparison under extra-term conditions. (Top) Example frames from 16-frame VC2 generating 128-frame, without and with FLUXFLOW, showcasing dynamic background consistency in the latter. Caption: A dog running along a beach, splashing water as it moves through the waves. (Bottom) Comparison of temporal quality metrics on VBench, where the gray regions indicate the performance drop under extra-term scenarios.

on 16-frame VC2 and 33-frame NOVA, as illustrated in Fig. 7(c,d). The results indicate that performance begins to decline significantly when the perturbation degree exceeds half of the total frames. This observation aligns with Obs❹, which highlights that perturbing more than half of the frames disrupts the model’s ability to infer the correct temporal order due to insufficient contextual information.

Inter-frame/block Interval, and (ii) Perturbation Degree.

Inter-frame/block Interval Analysis. We analyze the impact of shuffle interval constraints on frame-level (FLUXFLOW-FRAME) and block-level (FLUXFLOWBLOCK). The shuffle interval defines the minimum distance between shuffled frames or blocks. For example, in a 2 × 1 frame-level shuffle with an interval of 8 frames, any two shuffled frames must be separated by at least 8 frames. As demonstrated in Fig. 7(a,b), ablations on VC2 using 2 × 1 and 2 × 2 shuffle configurations reveal that removing interval constraints (0.0% interval ratio) achieves the best performance across all metrics. Larger constraints (e.g., 25% or 50%) lead to noticeable performance degradation. This suggests that allowing free shuffle without interval constraints enables the model to better leverage temporal information, supporting the hypothesis that excessive constraints reduce the diversity of temporal patterns learned by the model.

#### 5. Conclusion

In this work, we propose FLUXFLOW, a pioneering temporal data augmentation strategy aimed at enhancing temporal quality in video generation. This initial exploration introduces two simple yet effective ways: framelevel (FLUXFLOW-FRAME) and block-level (FLUXFLOWBLOCK). By addressing the limitations of existing methods that focus primarily on architectural designs and conditioninformed constraints, FLUXFLOW bridges a critical gap in the field. Extensive experiments demonstrate that integrating FLUXFLOW significantly improves both temporal coherence and overall video fidelity. We believe FLUXFLOW sets a promising foundation for future research in temporal augmentation strategies, paving the way for more robust and temporally consistent video generation.

Perturbation Degree Analysis. We further examine whether excessive perturbation would cause significant performance degradation. We performed frame-level ablation

#### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [2] Haodong Chen, Haojian Huang, Junhao Dong, Mingzhe Zheng, and Dian Shao. Finecliper: Multi-modal fine-grained clip for dynamic facial expression recognition with adapters. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 2301–2310, 2024. 2
- [3] Haodong Chen, Yongle Huang, Haojian Huang, Xiangsheng Ge, and Dian Shao. Gaussianvton: 3d human virtual tryon via multi-stage gaussian splatting editing with image prompting. arXiv preprint arXiv:2405.07472, 2024. 11
- [4] Haodong Chen, Lan Wang, Harry Yang, and Ser-Nam Lim. Omnicreator: Self-supervised unified generation with universal editing. arXiv preprint arXiv:2412.02114, 2024. 2
- [5] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7310– 7320, 2024. 2, 3, 6, 7, 13
- [6] Jin Chen, Kaijing Ma, Haojian Huang, Jiayu Shen, Han Fang, Xianghao Zang, Chao Ban, Zhongjiang He, Hao Sun, and Yanmei Kang. Bovila: Bootstrapping video-language alignment via llm-based self-questioning and answering. arXiv preprint arXiv:2410.02768, 2024. 11
- [7] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169,

2024. 2, 5, 6, 7, 13

- [8] Haoyu Deng, Zijing Xu, Yule Duan, Xiao Wu, Wenjie Shu, and Liang-Jian Deng. Exploring the low-pass filtering behavior in image super-resolution. arXiv preprint arXiv:2405.07919, 2024. 11
- [9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 2
- [10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized textto-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 2, 3
- [11] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024. 11
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3

- [13] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [14] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 2, 3
- [15] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 2, 3
- [16] Haojian Huang, Xiaozhennn Qiao, Zhuo Chen, Haodong Chen, Bingyu Li, Zhe Sun, Mulin Chen, and Xuelong Li. Crest: Cross-modal resonance through evidential deep learning for enhanced zero-shot learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5181–5190, 2024. 11
- [17] Haojian Huang, Chuanyu Qin, Zhe Liu, Kaijing Ma, Jin Chen, Han Fang, Chao Ban, Hao Sun, and Zhongjiang He. Trusted unified feature-neighborhood dynamics for multiview classification. arXiv preprint arXiv:2409.00755, 2024. 11
- [18] Yongle Huang, Haodong Chen, Zhenbang Xu, Zihan Jia, Haozhou Sun, and Dian Shao. Sefar: Semi-supervised fine-grained action recognition with temporal perturbation and learning stabilization. arXiv preprint arXiv:2501.01245,

2025. 2

- [19] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2, 5
- [20] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3
- [21] Aonian Li, Bangwei Gong, Bo Yang, Boji Shan, Chang Liu, Cheng Zhu, Chunhao Zhang, Congchao Guo, Da Chen, Dong Li, et al. Minimax-01: Scaling foundation models with lightning attention. arXiv preprint arXiv:2501.08313, 2025. 2, 3
- [22] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 2
- [23] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 3
- [24] Yexin Liu and Lin Wang. Mycloth: Towards intelligent and interactive online t-shirt customization based on user’s pref-

- erence. In 2024 IEEE Conference on Artificial Intelligence (CAI), pages 955–962. IEEE, 2024. 11
- [25] Yexin Liu, Zhengyang Liang, Yueze Wang, Muyang He, Jian Li, and Bo Zhao. Seeing clearly, answering incorrectly: A multimodal robustness benchmark for evaluating mllms on leading questions. arXiv preprint arXiv:2406.10638, 2024.
- [26] Kaijing Ma, Haojian Huang, Jin Chen, Haodong Chen, Pengliang Ji, Xianghao Zang, Han Fang, Chao Ban, Hao Sun, Mulin Chen, et al. Beyond uncertainty: Evidential deep learning for robust video temporal grounding. arXiv preprint arXiv:2408.16272, 2024. 11
- [27] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 5
- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3
- [29] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 5
- [30] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2616–2625,

2020. 11

- [31] Wen-Jie Shu, Hong-Xia Dou, Rui Wen, Xiao Wu, and LiangJian Deng. Cmt: Cross modulation transformer with hybrid loss for pansharpening. IEEE Geoscience and Remote Sensing Letters, 21:1–5, 2024. 11
- [32] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 3
- [33] K Soomro. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402,

2012. 2, 5

- [34] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2
- [35] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 5
- [36] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 3
- [37] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2

- [38] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2, 3
- [39] Xianzu Wu, Xianfeng Wu, Tianyu Luan, Yajing Bai, Zhongyuan Lai, and Junsong Yuan. Fsc: Few-point shape completion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26077– 26087, 2024. 11
- [40] Xianfeng Wu, Yajing Bai, Haoze Zheng, Harold Haodong Chen, Yexin Liu, Zihao Wang, Xuran Ma, Wen-Jie Shu, Xianzu Wu, Harry Yang, et al. Lightgen: Efficient image generation through knowledge distillation and direct preference optimization. arXiv preprint arXiv:2503.08619, 2025. 2
- [41] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. arXiv preprint arXiv:2310.12190, 2023. 3, 5
- [42] Zhen Xing, Qi Dai, Han Hu, Jingjing Chen, Zuxuan Wu, and Yu-Gang Jiang. Svformer: Semi-supervised video transformer for action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18816–18826, 2023. 2
- [43] Yibo Yan, Haomin Wen, Siru Zhong, Wei Chen, Haodong Chen, Qingsong Wen, Roger Zimmermann, and Yuxuan Liang. Urbanclip: Learning text-enhanced urban region profiling with contrastive language-image pretraining from the web. In Proceedings of the ACM Web Conference 2024, pages 4006–4017, 2024. 11
- [44] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 5, 6, 7, 13
- [45] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6023–6032, 2019. 2
- [46] Hongyi Zhang. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017. 2
- [47] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. Motiondirector: Motion customization of text-tovideo diffusion models. arXiv preprint arXiv:2310.08465,

2023. 3

- [48] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 2
- [49] Yuliang Zou, Jinwoo Choi, Qitong Wang, and Jia-Bin Huang. Learning representational invariances for dataefficient action recognition. Computer Vision and Image Understanding, 227:103597, 2023. 2

## Temporal Regularization Makes Your Video Generator Stronger Supplementary Material

#### A. Detailed Analysis Settings

This section details the prompt information for the analysis in the main text.

##### A.1. Temporal Diversity Analysis

As shown in Figure 5 in Section 3.3, we analyze temporal diversity within generated videos by evaluating three groups of text prompts with distinct temporal dynamics: Static, Slow, and Fast. These prompts are designed to capture varying levels of motion and temporal changes across the generated videos. Below, we provide the complete prompt details for each group:

Static

- 1. "A serene mountain landscape at sunrise, with no movement, only a slight shimmer of sunlight on the lake."
- 2. "A still life of a fruit bowl on a wooden table, with soft sunlight streaming through a window, no movement."
- 3. "A quiet library with shelves full of books, no people, only a faint light flickering from a reading lamp."
- 4. "A frozen winter forest, with snow-covered trees, no wind, and complete stillness."
- 5. "A close-up of a flower in a garden, with no movement, just soft natural lighting."

Slow

- 1. "A slowly flowing river in a forest, with leaves gently drifting on the surface."
- 2. "A time-lapse of a sunset over the ocean, with the colors of the sky slowly changing."
- 3. "A person walking slowly through a quiet park, with trees swaying gently in the breeze."
- 4. "A candle flame flickering gently in a dark room, with soft shadows moving on the wall."
- 5. "A snail crawling slowly on a leaf, with tiny raindrops falling in the background."

Fast

- 1. "A high-speed car racing through a winding mountain road, with trees and scenery blurring past."
- 2. "A flock of birds flying rapidly across a stormy sky, with clouds moving fast and lightning flashing."
- 3. "A skateboarder performing tricks in a skatepark, with fast-paced movements and dynamic camera angles."
- 4. "A rocket launching into space, with flames and smoke bursting out rapidly."
- 5. "A bustling city street at night, with cars and people moving quickly, and neon lights flashing."

These prompts ensure a comprehensive assessment of the model’s ability to generate videos with diverse temporal characteristics.

##### A.2. User Study with Temporal Dynamics

In Section 4.3, we conduct user studies to evaluate the perceived quality of temporal dynamics in the generated videos. The study involves two key aspects: Action Speed (Fast & Slow) and Motion Pattern (Linear & Nonlinear). Below, we provide the full prompt details used for each category:

Action Speed: Fast

- 1. "A person sprints across a track at full speed."
- 2. "A cheetah runs rapidly through the savannah, chasing its prey."
- 3. "A cyclist pedals quickly downhill on a steep mountain trail."
- 4. "A soccer ball is kicked hard and flies quickly across the field."
- 5. "A train speeds through the station without stopping." Action Speed: Slow

- 1. "A person strolls leisurely along a quiet beach."
- 2. "A cat stretches slowly under the morning sunlight."
- 3. "A snail crawls gently across a leaf, moving inch by inch."
- 4. "A balloon ascends slowly into the sky on a calm day."
- 5. "A candle flame flickers gently in the still air."

Motion Pattern: Linear

- 1. "A car drives steadily along a straight highway under the evening sky."
- 2. "A person walks directly across a pedestrian crossing in a straight line."
- 3. "A paper airplane glides smoothly in a straight trajectory across the room."
- 4. "A rocket launches vertically into the sky, moving in a straight path."
- 5. "A ball rolls straight down a ramp without deviating." Motion Pattern: Nonlieaner

- 1. "A butterfly flutters in a zigzag pattern through a flower garden."
- 2. "A skier carves smooth curves as they descend a snowy slope."
- 3. "A fish swims in circular loops in a clear blue pond."
- 4. "A drone hovers and moves in an unpredictable spiral pattern above the city."
- 5. "A kite dances in the wind, swaying back and forth in an irregular motion."

Ten participants were asked to evaluate the generated videos from Motion Diversity, Motion Realism, Motion Smoothness, Temporal Coherence, and Optical Flow Consistency, scoring from 0 to 5 for each dimension. The results of this study provide valuable insights into the effectiveness of FLUXFLOW in capturing different temporal dynamics. Examples are shown in Figure 10.

#### B. Limitations

Deep learning [3, 6, 8, 11, 16, 17, 24–26, 30, 31, 39, 43] has revolutionized video generation by enabling models

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

CogVideoX

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

w/ FluxFlow

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Figure 10. User study examples. Each video is provided with its optical flow to assess the Optical Flow Consistency. Caption: A skier carves smooth curves as they descend a snowy slope.

to learn complex spatiotemporal patterns from large-scale data. While our work introduces FLUXFLOW as a pioneering exploration of temporal data augmentation in video generation, it is limited to two strategies: frame-level shuffle and block-level shuffle. These methods, while effective, represent only an initial step in this direction. Future work could explore more advanced temporal augmentation techniques, such as motion-aware or context-sensitive strategies, to further enhance temporal coherence and diversity. We hope this study inspires broader research into temporal augmentations, paving the way for more robust and expressive video generation models.

#### C. More Experimental Results

We provide more comparison results here in Figure 11.

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

VideoCrafter2

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

w/ FluxFlow

"A person tossing a frisbee to a dog, with the dog jumping to catch it mid-air."

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

VideoCrafter2

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

w/ FluxFlow

"A fish jumping out of the water and splashing back in."

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

NOVA

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

w/ FluxFlow

"A butterfly fluttering erratically in a flower garden."

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

NOVA

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

w/ FluxFlow

"A firework exploding in the night sky, scattering sparks in all directions."

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

CogVideoX

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

###### w/ FluxFlow

"A cat chasing a laser pointer controlled by a person, darting across the floor."

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

CogVideoX

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

###### w/ FluxFlow

"A person passing a football to a teammate, who catches it mid-run."

Figure 11. More comparison of FLUXFLOW on VideoCrafter2 [5], NOVA [7], and CogVideoX [44].

