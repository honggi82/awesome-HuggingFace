## PersonaLive! Expressive Portrait Image Animation for Live Streaming

Zhiyuan Li1,2,3 Chi-Man Pun1,* Chen Fang2 Jue Wang2 Xiaodong Cun3,* 1 University of Macau 2 Dzine.ai 3 GVC Lab, Great Bay University https://github.com/GVCLab/PersonaLive

# arXiv:2512.11253v1[cs.CV]12Dec2025

[Figure 1]

1st frame / 1s 300th frame / 20s 600th frame / 39s 950th frame / 60s

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|[Figure 15]|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

###### 22×

7×

2"×

76×

Figure 1. An overview of generated portraits and inference speed of PersonaLive. PersonaLive produces high-quality, temporally stable portrait animations over long sequences, while achieving real-time streaming performance with substantially lower latency than prior diffusion-based approaches.

### Abstract

trait animation with multi-stage training recipes. Specifically, we first adopt hybrid implicit signals, namely implicit facial representations and 3D implicit keypoints, to achieve expressive image-level motion control. Then, a fewer-step appearance distillation strategy is proposed to eliminate appearance redundancy in the denoising process, greatly improving inference efficiency. Finally, we introduce an autoregressive micro-chunk streaming generation paradigm

Current diffusion-based portrait animation models predominantly focus on enhancing visual quality and expression realism, while overlooking generation latency and real-time performance, which restricts their application range in the live streaming scenario. We propose PersonaLive, a novel diffusion-based framework towards streaming real-time por-

equipped with a sliding training strategy and a historical keyframe mechanism to enable low-latency and stable longterm video generation. Extensive experiments demonstrate that PersonaLive achieves state-of-the-art performance with up to 7-22× speedup over prior diffusion-based portrait animation models.

### 1. Introduction

Influencers’ live streaming has become one of the hottest areas in short-video social media. The Internet provides us with a chance to disguise ourselves as virtual beings. Early 3D avatar approaches [3, 24, 64] cannot reenact expressive movements and rely on expensive motion capture devices. In contrast, the portrait animation algorithms [52, 54, 60, 63] animates a static portrait image according to the motions (i.e., detailed expression, pose) captured from a driving video, which shows great potential.

Recently, diffusion-based portrait animation methods [52, 54, 60, 63] have emerged as a dominant paradigm due to their strong generative capabilities. However, directly using these models in a live streaming scenario has two key obstacles: (i) the high computational cost. Current methods primarily focus on improving visual quality and motion consistency while overlooking inference efficiency. Most of them require over 20 denoising steps [39] and rely on the CFG technique [16] to enhance visual fidelity and expression control, which hinders their practical application; (ii) the limitations of chunk-wise processing. Due to computational and memory constraints, current methods divide long videos into multiple fixed-length chunks and process them independently. To improve temporal consistency across chunks, several methods [52, 54, 55, 63] introduce trainingfree overlapping frames between adjacent chunks, resulting in redundant computation and increased latency. Other methods [20, 41, 53] reuse the last few frames from the previously generated chunk to enhance cross-chunk consistency, which inevitably causes error accumulation during long video generation.

We posit that portrait animation primarily involves modeling motion changes across highly similar frames, a task that may not necessitate extensive denoising steps. Furthermore, in contrast to independent chunk-wise generation, we can directly train the model for longer and continuous generation conditioned on previously generated frames’ intermediate latents and contexts.

We thus propose PersonaLive, a diffusion-based portrait animation framework for real-time, streamable motiondriven animation. Building upon the recent success of ReferenceNet-based diffusion animation method [29, 52, 63], we incorporate several novel components. (i) Motion Transfer with Hybrid Control. For portrait animation, effective motion control is essential to ensure realistic and

expressive synthesis. In this work, we adopt hybrid motion signals, composed of implicit facial representations [63] and 3D implicit keypoints [11, 47], to achieve simultaneous control of both facial dynamics and head movements. Compared with the 2D landmarks [4, 17] and motion frames [52, 56] used in existing methods, 3D implicit keypoints provide a more flexible and controllable representation of head motion. (ii) Fewer-Step Appearance Distillation. We observe that portrait animation exhibits appearance redundancy in the denoising process. Specifically, the structural layout and motion are established in the initial denoising steps, whereas numerous subsequent iterations are inefficiently spent on gradually refining appearance details such as texture and illumination. To address this inefficiency, we introduce an appearance distillation strategy that adapts the pretrained diffusion model to a compact sampling schedule, significantly improving inference efficiency without compromising visual quality. (iii) Micro-chunk Streaming Video Generation. After accelerating the denoising process with the previous strategy, we further aim to enable low-latency and temporally coherent video generation for real-time streaming applications. In contrast to chunk-wise generation [13], which relies on latents with uniform noise levels, we adopt an autoregressive micro-chunk streaming paradigm [5] that assigns progressively higher noise levels across micro chunks with each denoising window, enabling continuous video generation. To mitigate exposure bias [30, 36] inherent in the autoregressive paradigm, we design a Sliding Training Strategy (ST) to eliminate the discrepancy between the training and inference stages and an effective Historical Keyframe Mechanism (HKM) that adaptively selects historical frames as auxiliary references, effectively mitigating error accumulation during streaming generation. Extensive quantitative and qualitative results show that PersonaLive achieves state-ofthe-art performance with up to 7-22× speedup over prior diffusion-based portrait animation models.

The contributions of this paper can be summarized as:

- • We propose PersonaLive, a few-step diffusion-based framework for real-time, streamable portrait animation that achieves low-latency and stable long-term quality.
- • We design hybrid motion signals combining implicit facial representations and 3D implicit keypoints to enable the simultaneous control of both fine-grained facial dynamics and head movements. Furthermore, we introduce a fewer-step appearance distillation strategy to eliminate appearance redundancy in denoising, greatly improving inference efficiency without compromising visual fidelity.
- • We design an autoregressive micro-chunk streaming generation paradigm equipped with a sliding training strategy and a historical keyframe mechanism, effectively mitigating exposure bias and error accumulation for stable long-term generation.
- • Extensive experiments demonstrate that our method

- (a) Stage 1: Image-level Hybrid Motion Training
- (b) Stage 2: Fewer-step Appearance Distillation

(c) Stage 3: Micro-chunk Streaming Video Generation

|[Figure 20]<br><br>[Figure 21]||[Figure 22]<br><br>[Figure 23]|
|---|
|[Figure 24]<br><br>[Figure 25]|[Figure 26]<br><br>|
|---|---|---|---|

[Figure 27]

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]<br><br>|[Figure 31]|
|---|---|

|[Figure 32]|
|---|

|[Figure 33]| |
|---|---|
| | |

[Figure 34]

Spatial Module

· ·

··· ···

Motion Module

[Figure 35]

Motion Extractor

slide

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Pose Guider

[Figure 43]

[Figure 44]

|[Figure 45]|
|---|

|[Figure 46]<br><br>|
|---|

|[Figure 47]<br><br>|[Figure 48]<br><br>[Figure 49]|
|---|---|

|[Figure 50]<br><br>[Figure 51]|
|---|

|[Figure 52]<br><br>[Figure 53]|[Figure 54]<br><br>|
|---|---|

|[Figure 55]|
|---|

| | |
|---|---|
|[Figure 56]| |

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

· ·

··· ···

3D Implicit keypoints

Unet fea.

HKM

Temporal Module Real Frames Generated Frames Frozen Weight Tuning Weight

Q

Reference fea.

K V

| |
|---|

| | |
|---|---|
| | |

[Figure 61]

History fea.

[Figure 62]

| |
|---|

Motion fea.

concat

[Figure 63]

|[Figure 64]|
|---|

[Figure 65]

Stop grad

Discriminator

[Figure 66]

History Bank

if True

[Figure 67]

[Figure 68]

Motion bank

Motion Extractor

[Figure 69]

|[Figure 70]| |
|---|---|
| | |

[Figure 71]

|[Figure 72]|
|---|

|[Figure 73]<br><br>|
|---|

|[Figure 74]<br><br>|[Figure 75]<br><br>[Figure 76]|
|---|---|

|[Figure 77]|
|---|

|[Figure 78]<br><br>|[Figure 79]<br><br>|
|---|---|

|[Figure 80]|
|---|

···

VAE

[Figure 81]

· ·

1~4 steps

[Figure 82]

[Figure 83]

Figure 2. Overview of the three-stage pipeline of PersonaLive. (a) Image-level hybrid motion training: Learns expressive motion control using implicit facial representations and 3D implicit keypoints. (b) Fewer-step appearance distillation: Eliminates appearance redundancy in the denoising process, improving inference efficiency without compromising visual quality. (c) Micro-chunk streaming video generation: An autoregressive micro-chunk paradigm, equipped with sliding training and historical keyframes, enables low-latency and temporally coherent real-time video generation.

achieves state-of-the-art performance while achieving significantly higher efficiency.

### 2. Related Work

Diffusion-based Portrait Animation. Diffusion models [15, 39, 40] have demonstrated strong generative capabilities, with Latent Diffusion Models (LDMs) [33] further improving efficiency by performing the denoising process in a lowerdimensional latent space. Building upon this foundation, several works [46, 52, 54, 63] extend pre-trained diffusion models [1, 33, 45] to controllable and high-fidelity portrait animation with explicit structural conditions, such as facial keypoints [14, 29, 31], facial mesh renderings [12, 26], and original driving video [52, 55, 56]. These methods typically employ ControlNet [61] or PoseGuider [17] to incorporate motion constraints into the generation process. To model fine-grained facial dynamics, recent works [28, 46, 54, 63] introduce implicit facial representations. This strategy enhances the preservation of intricate facial expression details, enabling more flexible and realistic animation. However, the above methods primarily focus on improving visual quality and motion consistency while overlooking inference efficiency. In this work, we address this limitation by introducing a real-time, streamable diffusion framework that enables efficient and temporally coherent portrait animation.

Long-term Portrait Animation. With the rapid advancement of animation methods and rising user expectations, producing temporally coherent long-term videos has be-

come critical. Due to computational constraints, existing diffusion-based methods [29, 52–56, 63] are trained on short clips and rely on inference-time extension for longer sequences. X-Portrait [52] and X-NeMo [63] adopt the prompt traveling technique [42] to enhance temporal smoothness across chunk boundaries. Follow-your-emoji [29] design a coarse-to-fine progressive strategy that generates intermediate frames through keyframe-guided interpolation. Sonic [19] builds global inter-clip connections through the time-aware shifted windows that bridge the preceding clip along the timesteps axis. Despite these advances, existing approaches remain unsuitable for real-time streaming generation. While several methods [20, 41, 53] leverage “motion frames” to enable chunk-wise streaming generation of long videos, they introduce additional training overhead and inevitable error accumulation [43]. In contrast, we introduce an autoregressive micro-chunk framework to enable streaming and temporally coherent long-term portrait animation.

Diffusion Model Acceleration. Despite their strong performance, the high computational cost of diffusion models keeps them far from real-time applications. Existing acceleration strategies can be broadly categorized into model quantization [9, 25, 50] and sampling step reduction [18, 27, 57– 59]. ADD [35] combines an adversarial and a score distillation objective to efficiently distill diffusion models. Viewing the guided reverse diffusion process as solving an augmented probability flow ODE (PF-ODE), LCMs [27] directly predict the solution of such ODE in latent space, mitigating the need

for numerous iterations. DMD [58] and DMD2 [57] distill a many-step diffusion model into a few-step generator by minimizing the approximate Kullback-Liebler (KL) divergences between the diffused target and generator output distributions. Despite recent advances, little attention has been paid to the application of the distillation technique in portrait animation. In this paper, we explore diffusion distillation for real-time portrait animation.

### 3. Method

Streaming portrait animation aims to generate long-term, temporally coherent animation streams from a given reference image and driving video, in a real-time and low-latency manner. Formally, given a reference portrait image IR and a continuous stream of S driving frames {ID1 ,ID2 ,...,IDS}, the objective of streaming portrait animation is to synthesize an animation sequence A{1,2,...,S} in a streaming paradigm, where each frame is rendered in real time by combining the appearance information from IR with the motion cues extracted from {ID1 ,ID2 ,...,IDS}, which is formulated as:

Ai = D(M(IDi ),R(IR)), i = 1,2,...,S, (1)

where D is the denoising backbone, M is the motion extractor, and R is the appearance extractor. As shown in Fig. 2, we achieve expressive and coherent streaming animation through a three-stage pipeline. We first employ hybrid motion control to achieve expressive and robust motion transfer (Sec. 3.1). Then, a fewer-step appearance distillation strategy is introduced to compress the redundant appearance refinement process (Sec. 3.2). Finally, to ensure low-latency and stable long-term generation, we propose a micro-chunk streaming generation paradigm equipped with a sliding training strategy and a historical keyframe mechanism (Sec. 3.3).

#### 3.1. Image-level Hybrid Motion Training

As shown in Fig. 2 (a), we leverage a pretrained diffusion model D as the denoising backbone and a reference network R for appearance conditioning. To achieve expressive and robust motion control, we adopt hybrid conditioning signals composed of implicit facial representations and 3D implicit keypoints. Specifically, we first crop the face region from the driving image ID and use a face motion extractor Ef [63] to encode it into 1D facial motion embeddings mf = Ef(ID), which are then injected into D via cross-attention layers. Since the implicit facial representations focus solely on local facial dynamics, we further introduce 3D implicit keypoints to capture global pose, position, and scale information. We use an off-the-shelf method Ek [11] to extract 3D parameters from the driving image ID and the source image IR:

kc,d, Rd, td, sd = Ek(ID), kc,s, Rs, ts, ss = Ek(IR),

(2)

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

999 949 899 849

Source image ···

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

199 149 99 49

Driving image Denoising trajectory, 20 steps

Figure 3. The denoising trajectory without CFG [16].

where kc represents the canonical keypoints, R, t, and s represent the rotation, translation, and scale parameters, respectively. The driving 3D implicit keypoints kd are transformed as follows:

kd = sd · kc,sRd + td. (3)

Finally, the extracted 3D implicit keypoints kd are mapped to the pixel space and injected into D via PoseGuider [17].

#### 3.2. Fewer-step Appearance Distillation

Building upon the hybrid motion control, we observe that in portrait animation, the motion and structural layout of each frame are largely determined during the earliest denoising step, while subsequent iterations primarily refine appearance details, as shown in Fig. 3. This observation reveals substantial redundancy in the denoising process, motivating us to develop a distillation strategy that significantly reduces sampling steps without compromising visual fidelity.

Based on the above motivation, we introduce a fewer-step appearance distillation strategy to compress the redundant refinement process into a compact sampling schedule {ti}Ni=1, as shown in Fig. 2 (b). Specifically, starting from a Gaussian noise latent znoise ∼ N(0,I), we randomly sample a denoising step n ∈ [1,N] and perform n denoising iterations to obtain an intermediate noise-free state zˆ0, which is then decoded into the pixel space as xˆ = Vd(ˆz0). The predicted image xˆ is supervised by the corresponding ground-truth frame xgt using a hybrid objective that combines MSE loss, LPIPS loss [62] and adversarial loss [10]:

Ldistill = L2(ˆx,xgt)+λlpipsLlpips(ˆx,xgt)+λadvLadv(ˆx),

(4) where λlpips and λadv are balancing coefficients. Backpropagating through the entire diffusion process would result in excessive memory consumption. To improve computational efficiency, we propagate gradients only through the final denoising step, while stochastic step sampling ensures that all middle timesteps receive supervision throughout training.

#### 3.3. Micro-chunk Streaming Video Generation

To extend the image animation model for video generation, we integrate a temporal module [13] into the denoising back-

bone D. However, instead of assigning a uniform noise level to all frames within a denoising window as in conventional methods, we divide each denoising window into multiple micro-chunks with progressively higher noise levels, as shown in Fig. 2(c). Formally, the denoising window at step s is defined as a collection of N micro-chunks:

Ws = {Cs1,Cs2,...,CsN}, (5) Csn = {zt

i |i = 1,2,...,M}, t1 < t2 < ··· < tN, (6) where Csn denotes the n-th micro-chunk consisting of M frames. After each denoising step, all chunks are shifted to lower noise levels, with the first chunk yielding M clean frames ready for emission. Subsequently, the denoising window slides forward by one chunk, and a new noisy chunk Cnoise = {ϵi}Mi=1 is appended at the end, initialized with Gaussian noise. This streaming processing paradigm enables continuous frame generation without overlapping regions, ensuring both temporal coherence and low latency. Despite its efficiency, streaming generation still suffers from exposure bias [30, 36] and error accumulation when generating long video sequences. To address this, we design a sliding training strategy and a historical keyframe mechanism to jointly stabilize long-range generation and enhance temporal coherence. Below, we give the details of each method.

n

Sliding Training Strategy. The exposure bias in streaming generation primarily stems from the discrepancy between training and inference: during training, the model learns from inputs derived from ground-truth frames. However, during inference, it must rely on its own generated predictions, which inevitably deviate from the distribution of ground-truth data and lead to accumulated temporal errors. To mitigate this issue, we simulate the streaming generation process during training, forcing the model to encounter and learn from its own prediction errors. As shown in Fig. 2 (c), the first denoising window is constructed from noisy ground-truth frames. For n = 1,2,...,N − 1, we define:

zigt + 1 − α¯t

C0n = { α¯t

ϵi}Mi=1, (7) where ϵi ∼ N(0,I), αt

n

n

is a noise scheduling parameter, and α¯t

n

= ti=1n αi. The final chunk C0N is initialized with a random noisy chunk Cnoise. After each denoising step, the denoising window slides forward by one chunk, and a new noisy chunk is appended at the end, which is completely consistent with the inference procedure. To reduce computational overhead, we compute gradients for only a subset of denoising windows and propagate them through a single denoising step. The overall training objective remains consistent with the appearance distillation stage. As shown in Fig. 4, interpolating the implicit motion signals enables a smooth transition from the source motion to the driving motion. Leveraging this property, we introduce a MotionInterpolated Initialization (MII) strategy, which constructs

n

Source image Motion interpolation Driving image

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

Figure 4. Motion interpolation for the first denoising window initialization.

the first denoising window using the reference image IR combined with interpolated implicit motion signals, to align the inference procedure with the training setup.

Historical Keyframe Mechanism. When synthesizing regions not explicitly constrained by the reference image (e.g., occluded areas), the stochasticity inherent in diffusion sampling can introduce subtle appearance variations across frames. In a streaming generation setting, these inconsistencies may gradually accumulate, leading to temporal drift and degraded visual stability over time. To mitigate this, we introduce historical keyframes, i.e., representative frames from previously generated results, as auxiliary references, providing the model with stable historical cues to preserve appearance consistency during long-term streaming synthe-

sis. As shown in Fig. 2(c), we maintain a history bank Bhis and a motion bank Bmot. The history bank stores reference features {h0,h1,...} extracted from historical keyframes, while the motion bank stores their corresponding motion embeddings {m0,m1,...}. After each denoising step, given the current motion embedding mf of the first frame, we measure its similarity to Bmot as:

∥mf − mi∥2. (8)

d = min

i=0,1,...

If d > τ, where τ denotes a predefined motion threshold, the current frame is identified as a keyframe. Its reference fea-

tures hf and motion embedding mf are then added to Bhis and Bmot, respectively. During subsequent inference, these selected historical features are concatenated with the source image feature h0 and injected into the diffusion backbone via the spatial module to enhance temporal consistency.

### 4. Experiments

We train our method on the VFHQ [51], NerSemble [23] and DH-FaceVid-1K [8] datasets. All data are uniformly processed at 25 fps and cropped to a 512 × 512 resolution. For the discriminator, we employ the StyleGAN2 [22] architecture, initialized with weights pretrained on the FFHQ [21] dataset. The denoising steps in stage 2 and 3 are set to N = 4.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

HunyuanPortraitLivePortraitX-NeMoDriving&Ref.Megactor-ΣFollowYEX-Portrait

[Figure 114]

[Figure 115]

1 400 543 1094

###### 741 1 250 500 750 1500

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

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

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

[Figure 184]

[Figure 185]

Ours

Figure 5. Qualitative comparisons. PersonaLive achieves high-quality portrait animation using significantly fewer denoising steps, while preserving identity, expression fidelity, and facial detail.

The chunk size in stage 3 is set to M = 4. The motion threshold in HKM is set to τ = 17. The training is conducted on 8 Nvidia H100 GPUs using the AdamW optimizer with a learning rate of 1 × 10−5 and a weight decay of 0.01. Following [11], we evaluate our model on the official test split of the TalkingHead-1KH dataset [47]. To further assess performance on long-term portrait animation, we build a benchmark comprising 100 in-the-wild reference portraits and 100 unseen long videos (most of them longer than one minute), referred to as LV100. More details about implementations can be found in the supplementary materials.

#### 4.1. Evaluations and Comparisons

Baselines and Metrics. We compare our method against state-of-the-art video-driven portrait animation baselines, including the GAN-based LivePortrait [11] and Diffusionbased X-Portrait [52], Follow-your-Emoji [29], Megactor-

Σ [56], X-NeMo [63], and HunyuanPortrait [54]. RAIN [37] adopts the diffusion forcing framework [5] for streaming generation on anime portrait data. However, it does not address essential challenges such as exposure bias and error accumulation, and the anime portrait domain is overly simplified for real-world portrait animation. Thus, we exclude RAIN from our comparisons. For self-reenactment, experiments are conducted on the TalkingHead-1KH dataset [47]. We evaluate the performance by computing L1, structural (SSIM [48]), perceptual (LPIPS [62]), and temporal (tLP [6]) differences to assess image quality, motion accuracy, and temporal consistency, respectively. For cross-reenactment, we evaluate on our collected LV100 benchmark, which contains diverse identities and long video sequences. We utilize the ArcFace Score [7] as the identity similarity (ID-SIM) metric. Motion accuracy is calculated as the average L1 distance between extracted expression (AED [38]) and pose parame-

Table 1. Quantitative comparisons. Numbers in red and blue indicate the best and the second-best results, respectively. tLP multiplied by 10−3. All speed measurements are conducted on a single NVIDIA H100 GPU. * LivePortrait [11] is a frame-wise method using GAN. While it runs significantly faster than diffusion-based approaches, its generated portraits often lack fine-grained details.

Self-Reenactment Cross-Reenactment Efficiency L1 ↓ SSIM ↑ LPIPS ↓ tLP ↓ ID-SIM ↑ AED ↓ APD ↓ FVD ↓ tLP ↓ FPS ↑ Latency ↓

Method

LivePortrait* [11] 0.043 0.821 0.137 20.40 0.723 0.729 0.027 557.2 13.51 – – X-Portrait [52] 0.049 0.777 0.173 25.87 0.678 0.823 0.061 587.8 24.52 0.851 14.10 FollowYE [29] 0.045 0.803 0.144 26.92 0.773 0.911 0.043 696.5 35.13 1.558 7.793 Megactor-Σ [56] 0.055 0.766 0.183 23.55 0.606 0.855 0.079 585.3 28.86 2.216 6.918 X-NeMo [63] 0.077 0.689 0.267 25.11 0.691 0.679 0.022 639.1 18.10 1.281 15.32 HunyuanPortrait [54] 0.043 0.801 0.137 22.33 0.644 0.804 0.069 620.4 16.84 1.443 14.91

###### Ours 0.039 0.807 0.129 21.31 0.698 0.703 0.030 520.6 12.83 15.82 0.253

ters (APD [38]) of the generated and driving images using SMIRK [32], with lower values indicating better expression and pose similarity. FVD [44] and tLP [6] are used to evaluate temporal coherence. Furthermore, we report Frames Per Second (FPS) and the average inter-chunk latency to assess the efficiency of diffusion-based models.

Self-Reenactment. For each test video, the first frame is used as the reference image, and the remaining frames serve as the driving inputs and ground-truth targets for sequence generation. As shown in Table 1, despite using significantly fewer denoising steps, PersonaLive achieves competitive or superior performance across all reconstruction metrics.

Cross-Reenactment. As evidenced in our qualitative comparisons in Fig. 5, PersonaLive achieves competitive or superior visual fidelity compared to existing methods. It consistently reconstructs facial details and maintains temporal stability across long sequences, while other baselines may exhibit texture smoothing, identity drift, or motion inconsistency in challenging cases. Quantitatively, as reported in Table 1, PersonaLive achieves performance comparable to existing methods in identity preservation (ID-SIM) and accurate motion transfer (AED/APD), while achieving the best FVD and tLP scores. These results indicate that PersonaLive provides improved long-term temporal coherence and superior overall perceptual quality.

Efficiency. As shown in Table 1, the proposed method achieves a substantial improvement in inference efficiency, running at 15.82 FPS with an average latency of only 0.253 s, far surpassing existing diffusion-based baselines. Moreover, by replacing the standard VAE decoder with the TinyVAE [2] decoder, PersonaLive can further boost the inference speed to 20 FPS. For all diffusion-based competitors, latency is reported without using overlapping frames between chunks. Although this setting allows them to perform chunk-wise streaming generation, the lack of overlapping regions inevitably leads to weaker temporal consistency across chunks. In contrast, PersonaLive maintains both real-time performance and stable long-term temporal coherence.

Source & Driving w/o distill w/ distill, w/o GAN Ours

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

Figure 6. Ablation on appearance distillation strategy. All results are generated using 4 denoising steps without the CFG technique.

#### 4.2. Ablation Studies

To validate the effectiveness of our key components, we conduct comprehensive ablation studies on both the fewer-step appearance distillation strategy and micro-chunk streaming generation paradigm.

Appearance Distillation. As shown in Fig. 6, directly reducing the number of sampling steps without distillation (w/o distill) leads to significant degradation in visual quality. Incorporating the appearance distillation strategy (w/ distill, w/o GAN) effectively improves reconstruction quality; however, the outputs still lack high-frequency details and appear overly smooth. Although applying CFG can enhance fidelity, it substantially reduces inference speed (only 9.5 FPS). In contrast, introducing an adversarial loss enables the model to generate more realistic results without relying on CFG, achieving both high visual fidelity and efficient inference.

Micro-chunk Streaming Generation. To assess the contribution of our streaming design, we examine how each component affects temporal stability and long-range consistency, as shown in Table 2 and Fig. 7. Below, we give the detailed introduction:

(1) Sliding Training Strategy. Removing the sliding training strategy (w/o ST) causes the model to train only on

Table 2. Ablation study on micro-chunk streaming generation.

setting ID-SIM↑ AED↓ APD↓ FVD↓ tLP↓ w/ ChunkAttn 0.689 0.709 0.032 537.0 12.83 ChunkSize=2 0.660 0.713 0.031 520.2 12.14 w/o MII 0.680 0.703 0.031 511.5 13.06 w/o HKM 0.728 0.710 0.031 535.6 13.27 w/o ST 0.549 0.785 0.040 678.8 10.05 Ours 0.698 0.703 0.030 520.6 12.83

|[Figure 196]<br><br>300|
|---|

[Figure 197]

[Figure 198]

[Figure 199]

1 600 1200

w/oSTw/oHKMChunkSize=2w/oMIIOurs

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Figure 7. Ablation study on the core components of the microchunk streaming generation paradigm.

GT-constructed noisy inputs, leading to a train-inference mismatch. Since the model never learns to correct its own prediction drift, errors rapidly accumulate and produce severe temporal collapse, as reflected by the huge drop in IDSIM (0.549) and other metrics. Visual artifacts in Fig. 7 (last row) clearly show temporal degradation.

(2) Historical Keyframe. As shown in Fig. 7 (w/o HKM), removing the historical keyframe mechanism leads to noticeable temporal drift in regions not constrained by the reference portrait (e.g., the clothing area). These inconsistencies accumulate over long sequences, ultimately reducing temporal stability. In contrast, incorporating historical keyframes (highlighted in the yellow box) effectively suppresses such drift and stabilizes long-term generation. Although ID-SIM exhibits a slight decrease, since historical cues partially weaken the reliance on the reference portrait,

Source & Driving PersonaLive Source & Driving PersonaLive

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

Figure 8. Failure cases. Some details of our method may fail when the given reference images are out of the training domain.

this trade-off is acceptable given the substantial improvement in temporal coherence.

- (3) Motion-Interpolation Initialization. To isolate the effect of motion-interpolation initialization, we remove it and instead adopt the variable-length initialization strategy [49]. As shown in Fig. 7 (w/o MII), removing MII introduces noticeable appearance distortions at the beginning of the sequence. These artifacts arise from the mismatch between training and inference, as the model is forced to transition abruptly from the reference motion to the driving motion.
- (4) Chunk Size and Attention. We further examine the influence of micro-chunk structure. Reducing the chunk size from 4 to 2 (ChunkSize=2) slightly improves temporal consistency but noticeably degrades identity similarity. This occurs because a smaller chunk size lowers intra-window variation, which helps stabilize short-term dynamics, but it also narrows the effective temporal receptive field, limiting the model’s ability to maintain identity information across longer sequences. As shown in Fig. 7, a smaller chunk size leads to more artifacts in later frames. Replacing the bidirectional attention with chunk-wise causal attention (w/ ChunkAttn) results in similar motion accuracy but a mild decrease in identity similarity.

### 5. Conclusion

We present PersonaLive, an efficient diffusion-based framework for streaming portrait animation via a three-stage strategy. Firstly, we introduce a diffusion-based image animation framework based on hybrid control. Then, by introducing an appearance distillation strategy and a micro-chunk streaming generation paradigm, PersonaLive enables real-time and lowlatency portrait animation. Furthermore, we design a sliding training strategy and a historical keyframe mechanism to alleviate exposure bias and error accumulation, ensuring stable long-term generation and enhanced temporal coherence. We conduct comprehensive experiments to demonstrate the advantages of the proposed methods in terms of visual quality, temporal coherence, and inference efficiency.

Limitation & Future Work. While our method achieves real-time and temporally coherent streaming portrait animation, there remain two primary limitations. First, the current framework does not explicitly exploit temporal redundancy

across consecutive frames, which could potentially improve inference efficiency and enable longer denoising windows for streaming generation. Second, our model is trained primarily on human facial data and thus struggles to generalize to out-of-domain portraits with non-human appearances, such as cartoon characters or animals, which may lead to artifacts like blurred or distorted eyes and mouths, as shown in Fig. 8. These limitations suggest promising directions for future research in enhancing the scalability and applicability of portrait animation models in real-world scenarios.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3, 14
- [2] Ollin Boer Bohan. Tinyvae. 2023. 7, 14
- [3] Chen Cao, Tomas Simon, Jin Kyu Kim, Gabe Schwartz, Michael Zollhoefer, Shunsuke Saito, Stephen Lombardi, ShihEn Wei, Danielle Belko, Shoou-I Yu, et al. Authentic volumetric avatars from a phone scan. ACM Transactions on Graphics (TOG), 41(4):1–19, 2022. 2
- [4] Di Chang, Yichun Shi, Quankai Gao, Jessica Fu, Hongyi Xu, Guoxian Song, Qing Yan, Yizhe Zhu, Xiao Yang, and Mohammad Soleymani. Magicpose: Realistic human poses and facial expressions retargeting with identity-aware diffusion. arXiv preprint arXiv:2311.12052, 2023. 2
- [5] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081– 24125, 2024. 2, 6
- [6] Mengyu Chu, You Xie, Jonas Mayer, Laura Leal-Taix´e, and Nils Thuerey. Learning temporal coherence via selfsupervision for gan-based video generation. ACM Transactions on Graphics (TOG), 39(4):75–1, 2020. 6, 7
- [7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019. 6
- [8] Donglin Di, He Feng, Wenzhang Sun, Yongjia Ma, Hao Li, Wei Chen, Lei Fan, Tonghua Su, and Xun Yang. Dh-facevid1k: A large-scale high-quality dataset for face video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12124–12134, 2025. 5
- [9] Weilun Feng, Haotong Qin, Chuanguang Yang, Zhulin An, Libo Huang, Boyu Diao, Fei Wang, Renshuai Tao, Yongjun Xu, and Michele Magno. Mpq-dm: Mixed precision quantization for extremely low bit diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 16595–16603, 2025. 3
- [10] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 4

- [11] Jianzhu Guo, Dingyun Zhang, Xiaoqiang Liu, Zhizhou Zhong, Yuan Zhang, Pengfei Wan, and Di Zhang. Liveportrait: Efficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168, 2024. 2, 4, 6, 7, 12
- [12] Mingtao Guo, Guanyu Xing, and Yanli Liu. High-fidelity relightable monocular portrait animation with lightingcontrollable video diffusion model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 228–238, 2025. 3
- [13] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. In The Twelfth International Conference on Learning Representations, 2024. 2, 4
- [14] Yue Han, Junwei Zhu, Keke He, Xu Chen, Yanhao Ge, Wei Li, Xiangtai Li, Jiangning Zhang, Chengjie Wang, and Yong Liu. Face-adapter for pre-trained diffusion models with finegrained id and attribute control. In European Conference on Computer Vision, pages 20–36. Springer, 2024. 3
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 2, 4
- [17] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 2, 3, 4
- [18] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009,

2025. 3

- [19] Xiaozhong Ji, Xiaobin Hu, Zhihong Xu, Junwei Zhu, Chuming Lin, Qingdong He, Jiangning Zhang, Donghao Luo, Yi Chen, Qin Lin, et al. Sonic: Shifting focus to global audio perception in portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 193–203,

2025. 3

- [20] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming audio-driven portrait avatar with long-term motion dependency. In The Thirteenth International Conference on Learning Representations,

2025. 2, 3

- [21] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 5
- [22] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8110–8119, 2020. 5
- [23] Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter, and Matthias Nießner. Nersemble: Multi-view radiance field reconstruction of human heads. ACM Transactions on Graphics (TOG), 42(4):1–14, 2023. 5
- [24] Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun

- Zhou, and Lin Gu. Talkinggaussian: Structure-persistent 3d talking head synthesis via gaussian splatting. In European Conference on Computer Vision, pages 127–145. Springer, 2024. 2
- [25] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17535–17545, 2023. 3
- [26] Yukang Lin, Hokit Fung, Jianjin Xu, Zeping Ren, Adela SM Lau, Guosheng Yin, and Xiu Li. Mvportrait: Text-guided motion and emotion control for multi-view vivid portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26242–26252, 2025. 3
- [27] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 3
- [28] Yuxuan Luo, Zhengkun Rong, Lizhen Wang, Longhao Zhang, and Tianshu Hu. Dreamactor-m1: Holistic, expressive and robust human image animation with hybrid guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11036–11046, 2025. 3
- [29] Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–12, 2024. 2, 3, 6, 7
- [30] Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, and Itir Onal Ertugrul. Elucidating the exposure bias in diffusion models. In The Twelfth International Conference on Learning Representations, 2024. 2, 5
- [31] Di Qiu, Zhengcong Fei, Rui Wang, Jialin Bai, Changqian Yu, Mingyuan Fan, Guibin Chen, and Xiang Wen. Skyreels-a1: Expressive portrait animation in video diffusion transformers. arXiv preprint arXiv:2502.10841, 2025. 3
- [32] George Retsinas, Panagiotis P. Filntisis, Radek Danˇeˇcek, Victoria F. Abrevaya, Anastasios Roussos, Timo Bolkarr, and Petros Maragos. 3d facial expressions through analysis-byneural-synthesis. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2490– 2501, 2024. 7
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3, 12, 14
- [34] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer,

2015. 12

- [35] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer,

2024. 3

- [36] Florian Schmidt. Generalization in generation: A closer look at exposure bias. EMNLP-IJCNLP 2019, page 157, 2019. 2, 5

- [37] Zhilei Shu, Ruili Feng, Yang Cao, and Zheng-Jun Zha. Rain: Real-time animation of infinite video stream. arXiv preprint arXiv:2412.19489, 2024. 6
- [38] Aliaksandr Siarohin, St´ephane Lathuili`ere, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. First order motion model for image animation. Advances in neural information processing systems, 32, 2019. 6, 7
- [39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 2, 3

- [40] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 3
- [41] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive generating expressive portrait videos with audio2video diffusion model under weak conditions. In European Conference on Computer Vision, pages 244–260. Springer, 2024. 2, 3
- [42] Jonathan Tseng, Rodrigo Castellon, and Karen Liu. Edge: Editable dance generation from music. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 448–458, 2023. 3
- [43] Shuyuan Tu, Yueming Pan, Yinming Huang, Xintong Han, Zhen Xing, Qi Dai, Chong Luo, Zuxuan Wu, and Yu-Gang Jiang. Stableavatar: Infinite-length audio-driven avatar video generation. arXiv preprint arXiv:2508.08248, 2025. 3
- [44] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 7
- [45] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3
- [46] Qiang Wang, Mengchao Wang, Fan Jiang, Yaqi Fan, Yonggang Qi, and Mu Xu. Fantasyportrait: Enhancing multicharacter portrait animation with expression-augmented diffusion transformers. arXiv preprint arXiv:2507.12956, 2025. 3
- [47] Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. One-shot free-view neural talking-head synthesis for video conferencing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10039–10049, 2021. 2, 6
- [48] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [49] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6322–6332,

2025. 8

- [50] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In The Thirteenth International Conference on Learning Representations, 2025. 3

- [51] Liangbin Xie, Xintao Wang, Honglun Zhang, Chao Dong, and Ying Shan. Vfhq: A high-quality dataset and benchmark for video face super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 657–666, 2022. 5
- [52] You Xie, Hongyi Xu, Guoxian Song, Chao Wang, Yichun Shi, and Linjie Luo. X-portrait: Expressive portrait animation with hierarchical motion attention. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2, 3, 6, 7
- [53] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao, and Siyu Zhu. Hallo: Hierarchical audio-driven visual synthesis for portrait image animation. arXiv preprint arXiv:2406.08801, 2024. 2, 3
- [54] Zunnan Xu, Zhentao Yu, Zixiang Zhou, Jun Zhou, Xiaoyu Jin, Fa-Ting Hong, Xiaozhong Ji, Junwei Zhu, Chengfei Cai, Shiyu Tang, et al. Hunyuanportrait: Implicit condition control for enhanced portrait animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15909–15919, 2025. 2, 3, 6, 7
- [55] Shurong Yang, Huadong Li, Juhao Wu, Minhao Jing, Linze Li, Renhe Ji, Jiajun Liang, and Haoqiang Fan. Megactor: Harness the power of raw video for vivid portrait animation. arXiv preprint arXiv:2405.20851, 2024. 2, 3
- [56] Shurong Yang, Huadong Li, Juhao Wu, Minhao Jing, Linze Li, Renhe Ji, Jiajun Liang, Haoqiang Fan, and Jin Wang. Megactor-sigma: Unlocking flexible mixed-modal control in portrait animation with diffusion transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 9256–9264, 2025. 2, 3, 6, 7
- [57] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455– 47487, 2024. 3, 4
- [58] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 4
- [59] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 3
- [60] Bohan Zeng, Xuhui Liu, Sicheng Gao, Boyu Liu, Hong Li, Jianzhuang Liu, and Baochang Zhang. Face animation with an attribute-guided diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 628–637, 2023. 2
- [61] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023. 3
- [62] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. 4, 6

- [63] Xiaochen Zhao, Hongyi Xu, Guoxian Song, You Xie, Chenxu Zhang, Xiu Li, Linjie Luo, Jinli Suo, and Yebin Liu. X-nemo: Expressive neural motion reenactment via disentangled latent attention. In The Thirteenth International Conference on Learning Representations, 2025. 2, 3, 4, 6, 7
- [64] Yufeng Zheng, Wang Yifan, Gordon Wetzstein, Michael J Black, and Otmar Hilliges. Pointavatar: Deformable pointbased head avatars from videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21057–21067, 2023. 2

### A. Preliminary: Latent Diffusion Model

Latent Diffusion Models (LDMs) [33] conduct the diffusion process in a compact latent space for improving efficiency. Given an image x ∈ RH×W×3, an encoder Ve first maps it to a latent representation z = Ve(x). After that, the latent representation is progressively corrupted by Gaussian noise ϵ as:

zt = Ψ(z,ϵ,t) = √α¯tz + √1 − α¯tϵ, (9)

where α¯t is a pre-defined noise schedule within a finite time horizon t ∈ [0,1000]. A U-Net [34] denoiser ϵθ is trained to predict and remove the added noise from zt. The training objective is formulated as:

∥ϵ − ϵθ(zt,t,c)∥22, (10)

LLDM = E

t,z,ϵ

where c represents the conditioning input. In portrait animation, the input is a multi-frame latent window {zti}Mi=1, c contains appearance features extracted from the reference portrait and motion features derived from the driving video.

### B. Experimental Details

More implementation details. Our training pipeline progresses through three stages. In Stage 1, we conduct imagelevel hybrid motion training. Specifically, we randomly sample paired reference and driving images from the training videos, enabling the model to learn appearance conditioning from the reference portrait and motion conditioning from the driving input. This stage is trained for 30K iterations with a batch size of 32. During this stage, all model parameters are updated. After this initialization, Stage 2 performs fewer-step appearance distillation following Algorithm 1. We adopt a compact sampling schedule [0,333,666,999], enabling the model to learn to reconstruct high-quality frames from only a few denoising steps. This stage is trained for 30K iterations with a batch size of 32, and all model parameters remain trainable. Stage 3 focuses on temporal modeling: we train the temporal attention layers following Algorithm 2. At each iteration, we slide over a 40-frame sequence and perform three model updates. Only the temporal attention layers are trainable in this stage, while all other parameters remain frozen. This stage is trained for 10K iterations with a batch size of 8. For Stage 2 and 3, λlpips and λadv are set to 2.0 and 0.05, respectively.

Details of LV100. For long-term portrait animation evaluation, we collect 100 unseen videos (≥1 minute, 25 FPS) from various online platforms, including YouTube, TikTok, and BiliBili. Additionally, we compile 100 in-the-wild reference portraits from ChatGPT-5, Doubao, and Pexels, covering a broad range of facial structures, appearances, and styles. Representative examples from the LV100 benchmark are shown in Fig. 9.

Algorithm 1: Fewer-step Appearance Distillation Input: Reference image IR; Driving image ID; Animation model Gθ; VAE decoder Vd; Sampling schedule {ti}Ni=1

Output: Updated parameters θ for each iteration do

/* Sample initial noisy latent

and random step count n */ Sample znoise ∼ N(0,I) Sample n ∼ Uniform(1,N) Set zt

N ← znoise /* Perform n denoising steps */ for i = N to N − n + 1 do

if i > N − n + 1 then Disable gradient computation Set zˆ0 ← Gθ(zt

;ti,IR,ID) Sample ϵ ∼ N(0,I) Set zt

i

i−1 ← Ψ(ˆz0,ϵ,ti−1) else

Enable gradient computation Set zˆ0 ← Gθ(zt

;ti,IR,ID)

i

/* Decode prediction */ Set xˆ ← Vd(ˆz0) /* Update model */ Update θ via Distillation loss Ldistill

##### return θ

Implicit 3D keypoints. As shown in Fig. 10, the implicit 3D keypoint extractor [11] produces 21 canonical keypoints (left), from which we select a subset of stable landmarks (right) to encode global head pose, scale, and spatial configuration. These selected keypoints serve as an effective global motion prior in our hybrid motion control.

Motion-interpolation initialization. Given the reference image IR and first driving frame ID1 , we construct the first denoising window W0 = {C1,C2,...CN} using noisy reference latent zref = Ve(IR):

Cn = { α¯t

n

ϵi}Mi=1, (11)

zref + 1 − α¯t

n

where ϵi ∼ N(0,I). Subsequently, we interpolate the motion signals between the reference image and the first driving

frame. Let mf,s and m1f,d denote the implicit facial motion embeddings extracted from IR and ID1 , respectively. For the i-th frame in the initial window:

mf,i = (1 − ωi)mf,s + ωim1f,d, (12)

where ωi = MNi−1−1 is the interpolation factor. For implicit 3D keypoints, we interpolate the 3D transformation parame-

Algorithm 2: Sliding Training Strategy

Input: Reference image IR; Driving video {IDi }Si=1; Animation model Gθ; VAE encoder Ve; VAE decoder Vd; Micro-chunk size M; Sampling schedule {ti}Ni=1

Output: Updated parameters θ for each iteration do

/* Construct the initial denoising window W0 */

Set z1:M(N−1) ← Ve ID1:M(N−1) for n = 1 to N − 1 do

Sample ϵ ∼ N(0,I) Set Cn ← Ψ z(n−1)M+1:nM,ϵ,tn

Sample CN ∼ N(0,I) ; /* Last chunk

is pure noise */ Initialize window W0 = {C1,C2,...,CN} /* Sliding generation and

training */ for s = 0 to MS − N do

Set Vs ← IDsMN+1:(s+1)MN if s mod (N − 1) ̸= 0 then

Disable gradient computation Set Wˆs ← Gθ(Ws,t1:N,IR,Vs)

##### else

Enable gradient computation Set Wˆs ← Gθ(Ws,t1:N,IR,Vs) /* Decode sequence

prediction */ Set xˆseq ← Vd(Wˆs) /* Update model */

Update θ via Distillation loss Disable gradient computation

/* Slide window forward */ Set Ws+1 ← {Cˆs+2,Cˆs+3,...,Cˆs+N} Sample ϵs+1 ∼ N(0,I) Set Ws+1 ← Ψ(Ws+1,ϵs+1,t1:N−1) Sample Cs+N+1 ∼ N(0,I) Set Ws+1 ← {Ws+1,Cs+N+1}

##### return θ

ters:

Ri = R (1 − ωi)θs + ωiθd1 , si = (1 − ωi)ss + ωis1d, ti = (1 − ωi)ts + ωit1d,

(13)

where θ = (pitch,yaw,roll) denotes Euler angles and R(θ) denotes the rotation matrix constructed from Euler angles. The interpolated keypoints are then computed as:

ki = si · kc,sRi + ti. (14)

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

PortraitImageDrivingVideo

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

Figure 9. Examples from LV100.

all keypoints selected keypoints all keypoints selected keypoints

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

- Figure 10. The implicit 3D keypoints used in our hybrid motion control.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Source & Driving

[Figure 255]

[Figure 256]

- Figure 11. Effect of implicit 3D keypoints and facial motion embedding.

### C. More Ablations

In this section, we provide additional ablations on some network and hyperparameters.

Hybrid motion signals. As shown in Fig. 11, the implicit 3D keypoints kd control global head movements, including rotation, translation, and scale. In contrast, the implicit facial

- Table 3. Ablation study on motion threshold τ.

τ ID-SIM↑ AED↓ APD↓ FVD↓ tLP↓

- 15 0.6924 0.7043 0.0309 510.9 12.69
- 16 0.6940 0.7039 0.0306 516.6 12.78
- 17 0.6983 0.7028 0.0305 520.6 12.83
- 18 0.7015 0.7047 0.0306 522.9 12.93
- 19 0.7097 0.7084 0.0304 526.5 13.05
- 20 0.7159 0.7099 0.0303 529.2 13.18

- Table 4. Ablation study on VAE decoder.

decoder ID-SIM↑ AED↓ APD↓ tLP↓ FPS↑

SVD VAE [1] 0.6920 0.7452 0.0489 12.97 11.4 SD VAE [33] 0.6983 0.7028 0.0305 12.83 15.8 TinyVAE [2] 0.6758 0.7593 0.0489 14.66 20.0

motion embedding mf,d primarily controls fine-grained facial expressions. Although mf,d contains some pose-related cues, these signals have lower priority compared to the implicit 3D keypoints, as reflected in the result of ks + mf,d.

Motion threshold. We evaluate the effect of the motion threshold τ in our historical keyframe mechanism. As shown in Table 3, a smaller τ triggers more frequent history bank updates, providing richer historical information that helps stabilize long-term temporal consistency (lower FVD and tLP). However, more historical frames weakens the influence of the reference image IR, leading to slight ID drift and consequently lower ID-SIM. Conversely, a larger τ better preserves identity but slightly degrades temporal stability due to fewer historical keyframes. Overall, we set τ = 17 as it offers the best trade-off between identity preservation and temporal coherence.

VAE decoder. We further analyze the impact of different VAE decoders. As shown in Table 4, TinyVAE [2] significantly accelerates inference (up to 20 FPS) but introduces noticeable degradation in visual quality. SVD VAE [1] provides no improvement in temporal consistency and even reduces runtime efficiency. In contrast, SD VAE [33] achieves the best overall performance while maintaining competitive inference speed.

### E. Ethics Statement.

Our work focuses on advancing portrait animation technology and is developed solely for academic and creative research. While the method itself is not intended for malicious use, we acknowledge its potential misuse in generating deceptive or non-consensual synthetic media. To promote transparency and responsible use, all generated content should be clearly marked as artificial, and the technology should be applied in accordance with ethical and legal standards.

### D. More results

As shown in Fig. 12, Fig. 14, Fig. 16, and Fig. 18, we present additional visualization results under self-reenactment and cross-reenactment setting, further demonstrating the robustness and generalization ability of PersonaLive. Fig. 13, Fig. 15, Fig. 17, and Fig. 19 show long avatar videos synthesized by PersonaLive, highlighting its stability and consistency over long-term sequences.

|[Figure 257]<br><br>1|
|---|

|[Figure 258]<br><br>1|
|---|

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

187 395

600 800

Driving&Ref.X-PortraitMegactor-ΣHunyuanPortraitLivePortraitFollowYEX-NeMoOurs

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

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

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

200 400 600 764

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

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

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

Figure 12. More visualizations of self-reenactment comparison (1/2). The images with red borders are the reference images.

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

Driving&Ref.PersonaLivePersonaLiveDriving&Ref.

[Figure 347]

1 50 100 150 200 250 300 400 500 600

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

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

[Figure 368]

1 200 400 600 800 1000 1200 1400 1600 1800

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

Figure 13. Long avatar video results (1/4).

|[Figure 379]<br><br>1|
|---|

|[Figure 380]<br><br>1|
|---|

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

192 500 733 1000

Driving&Ref.X-PortraitMegactor-ΣHunyuanPortraitLivePortraitFollowYEX-NeMoOurs

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

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

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

120 200 330 1000

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

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

[Figure 458]

Figure 14. More visualizations of self-reenactment comparison (2/2). The images with red borders are the reference images.

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Driving&Ref.PersonaLivePersonaLiveDriving&Ref.

[Figure 469]

1 300 600 900 1200 1500 1800 2100 2400 2700

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

1 300 600 900 1200 1500 1800 2100 2400 2700

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

Figure 15. Long avatar video results (2/4).

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

Driving&Ref.X-PortraitMegactor-ΣHunyuanPortraitLivePortraitFollowYEX-NeMoOurs

[Figure 511]

[Figure 512]

1 500 1000 1600 2100 1 20 500 1086 1500

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

Figure 16. More visualizations of cross-reenactment comparison (1/2).

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

Driving&Ref.PersonaLivePersonaLiveDriving&Ref.

[Figure 593]

1 300 600 900 1200 1500 1800 2100 2400 2700

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

1 300 600 900 1200 1500 1800 2100 2400 2700

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

Figure 17. Long avatar video results (3/4).

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

LivePortraitFollowYEX-NeMoOursDriving&Ref.X-PortraitMegactor-ΣHunyuanPortrait

[Figure 630]

[Figure 631]

1 150 500 2320 2897

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

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

1 200 505 1023 1414

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

Figure 18. More visualizations of cross-reenactment comparison (2/2).

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

Driving&Ref.PersonaLivePersonaLiveDriving&Ref.

[Figure 717]

1 200 400 600 800 1000 1200 1400 1600 1800

[Figure 718]

[Figure 719]

[Figure 720]

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

1 150 300 450 600 750 900 1050 1200 1350

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

Figure 19. Long avatar video results (4/4).

