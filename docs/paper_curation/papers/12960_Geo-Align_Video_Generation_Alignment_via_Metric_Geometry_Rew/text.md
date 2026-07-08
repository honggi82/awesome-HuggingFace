# arXiv:2605.23903v1[cs.CV]22May2026

## Geo-Align: Video Generation Alignment via Metric Geometry Reward

#### Zizun Li1,2 Haoyu Guo2∗ Runzhe Teng1,2 Chunhua Shen2,3 Tong He2

##### 1USTC 2Shanghai AI Lab 3ZJU https://lizizun.github.io/geo-align-page/

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|
|---|---|---|---|---|---|
|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|
|[Figure 19]|[Figure 20]|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|
|[Figure 25]|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|[Figure 36]|

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Figure 1: Given a conditioning video, Geo-Align synthesizes a novel view video according to the target camera trajectory.

### Abstract

Camera-controlled video generation has achieved remarkable progress in recent years. However, existing video-to-video re-rendering methods primarily rely on Supervised Fine-Tuning using synthetic datasets. At present, there is an extreme scarcity of synchronized, multi-view real-world video data. Consequently, the prevailing paradigm often exhibits limited generalization when processing outof-distribution real-world videos, with models struggling to accurately adhere to physical scales and camera trajectories. To bridge this gap, we propose Geo-Align, the first Reinforcement Learning framework specifically designed for cameracontrolled video re-rendering. Built upon a pretrained model, we optimize the model through a scale-aware perceptual reward mechanism. Specifically, we introduce a metric 3D estimator to extract precise camera trajectories from generated videos, explicitly penalizing deviations in rotation and translation. Furthermore, we meticulously designed a data pipeline strategy based on real-world conditioning videos and target camera trajectories derived from synthetic data, eliminating the reliance on paired data. Extensive experiments demonstrate that Geo-Align consistently outperforms existing supervised learning baselines in both precise camera controllability and visual fidelity, indicating the effectiveness of our method.

∗Corresponding author.

Preprint.

### 1 Introduction

Camera controllability plays a vital role in video generation, particularly in fields such as film production and game engine rendering. In this paper, we focus on the video retake task. Formulated as a video-to-video generation problem, this task requires a model to synthesize a novel-view video along a target camera trajectory, given a conditioning video and the target trajectory as inputs. Recent methods such as ReCamMaster [1] and ReDirector [2], have successfully re-rendered dynamic scenes from input videos along new camera trajectories by training on synthetic datasets generated via engines like Unreal Engine. While TrajectoryCrafter [3] and CogNVS [4] are methods based on reconstruction, warping, and subsequent completion. However, the current supervised learning paradigm for generating videos with novel camera trajectories faces two core bottlenecks:

Data Scarcity: Unlike camera-controlled video generation conditioned on a single initial frame, video retake requires multi-view video data for supervised training. Given the scarcity of such real-world data, implicit condition methods [1, 2] predominantly rely on synthetic datasets, while warping-based methods [3, 5, 4] rely on point cloud renderings to synthesize target videos, constructing such data is highly non-trivial. While fine-tuning on synthetic data yields impressive results, these models often exhibit significant domain shift when performing inference on real-world scenes.

Metric Ambiguity: Camera pose annotations for existing real-world videos are often scale-less. Even the MultiCam-Video data constructed by ReCamMaster [1] only provides metric information for synthetic data. Standard SFT loss functions focus on pixel-level or feature-level reconstruction rather than explicitly optimizing for physically meaningful, metric-level camera alignment, frequently leading to scale drift in generated trajectories.

To address these challenges, we propose Geo-Align, a framework that introduces Reinforcement Learning (RL) to directly optimize the physical alignment and visual quality of camera movements. Unlike previous SFT paradigms [1, 2] that rely on time-synchronized ground-truth videos from multiple camera angles, reinforcement learning methods do not require video data corresponding to the target camera trajectory. Since real-world conditioning videos are easily obtainable, we can post-train the model via RL as long as we have the target camera trajectory. We adopt a fusion strategy combining real and synthetic data. During RL training, the conditioning videos are realworld captures. For the target camera trajectories, we sample from OmniWorld [6] gaming data, which provides a rich variety of natural camera movements. Since gaming trajectories are typically non-metric, we perform rescaling using Truncated Gaussian Sampling. Specifically, we sample the maximum values for rotation and translation between adjacent frames within defined thresholds and rescale the camera trajectories to reasonable scales accordingly.

We utilize a Verifiable Geometry Reward to train our model, which compares the camera trajectories estimated from the generated video (via MapAnything [7]) against the target trajectories. A metric evaluator is introduced to mitigate metric-related reward hacking during the reinforcement learning process. This effectively penalizes degenerate solutions—such as the model producing a shapepreserving but slow-moving trajectory in response to a rapid target trajectory. To prevent visual degradation during geometric optimization and preserve the model’s priors, we also incorporate aesthetic rewards, utilizing VideoAlign [8] and HPSv3 [9] as the reward models. We freeze the majority of the model’s parameters, training only the self-attention layers.

We evaluate our model on the DAVIS [10] datasets across the ten target camera trajectory categories defined by ReCamMaster [1]. Results demonstrate that our RL-trained model not only improves accuracy in following target trajectories on real-world data but also outperforms the original model across various aesthetic evaluation metrics. Our core contributions are as follows:

- • Reinforcement Learning for Video Retake: We utilize metric geometry model to extract rotation and translation errors. This enables our model to better align with geometric constraints and achieve more accurate metric scaling in real-world conditioning videos. Furthermore, we incorporate aesthetic rewards to enhance the overall quality of the generated videos.
- • Fusion Data Strategy: We leverage MapAnything [7] to extract camera poses from Citywalk [11] dataset as real-world conditioning priors. By combining this with Truncated Gaussian Sampling to rescale target trajectories from gaming data, we enhance training diversity and bridge the scale gap between source videos and target trajectories. Furthermore, it circumvents the necessity of paired multi-view video data.

- • State-of-the-Art (SOTA) Performance: Our RL-trained model achieves SOTA performance on the DAVIS [10] dataset across ReCamMaster’s [1] 10 trajectory types, consistently improving both camera trajectory fidelity and overall visual aesthetics. Qualitative comparisons further demonstrate a noticeable improvement in the quality of the generated videos.

### 2 Related Work

#### 2.1 Camera-Controlled Video Retake

Camera-controlled video retake [12–15] aims to synthesize novel views from existing footage by redirecting camera trajectories through generative models. Early approaches predominantly rely on explicit geometric transformations, utilizing external depth estimators [16, 17] and point trackers [18, 19] to warp input frames before refining them with video diffusion models [20–22], as seen in methods like TrajectoryCrafter [3] and CogNVS [4]. However, these explicit methods frequently suffer from warping artifacts that propagate directly into the synthesized output, particularly under dynamic camera motions or complex scene structures. To bypass explicit warping, implicit methods [4, 23–25] such as Generative Camera Dolly (GCD) [5] and ReCamMaster [1] condition models directly on camera extrinsic parameters, internalizing multi-view geometry through synthetic datasets. While recent advancements like ReDirector [2] extend this implicit paradigm to handle variablelength inputs and dynamic motions via Rotary Camera Encoding (RoCE). ll these frameworks fundamentally rely on supervised fine-tuning, where the primary bottleneck is the severe scarcity of time-synchronized multi-view video data. Since constructing such datasets from real-world footage is exceedingly difficult, existing SFT methods [1, 2] are forced to rely heavily on synthetic data.

#### 2.2 Feed-Forward 3D Reconstruction

Recent feed-forward models directly predict scene geometry without traditional SfM optimization [26–29]. DUSt3R [30] pioneered this by regressing dense point maps from unconstrained images. To handle continuous visual streams, methods [31–34] such as CUT3R [35] and WinT3R [36] introduced stateful memory and sliding-window mechanisms for efficient online perception. Concurrently, models [37–41] like VGGT [42], π3 [43], and Depth Anything 3 [44] have scaled into unified foundational architectures capable of jointly inferring multi-view geometry, cameras, and depth. Despite these advances, achieving accurate metric-scale reconstruction remains challenging. To address this, MapAnything [7] introduces a universal framework specifically for metric 3D reconstruction. By employing a factored representation that decouples camera poses and depth into scale-invariant components and explicit global scales, MapAnything [7] robustly maps local geometry into a unified metric space without test-time optimization.

#### 2.3 Group Relative Policy Optimization in Generative Models

Group Relative Policy Optimization (GRPO) [45] has recently emerged as a powerful online reinforcement learning framework for aligning generative models [46–48]. In flow-matching [49] domains, Flow-GRPO [50] enables online RL via ODE-to-SDE conversion, while MixGRPO [51] further improves optimization efficiency by introducing a mixed ODE-SDE sliding window sampling mechanism. This paradigm has similarly advanced video generation: GrndCtrl [52] utilizes GRPO for physically grounded world modeling, and recent frameworks [53–55] adopt verifiable geometry rewards to optimize precise camera-controlled video generation. Another line of work enhance synthesis quality by incorporating explicit [56–58] or implicit [59–62] geometric constraints as reward signals to enforce multi-view consistency. Furthermore, LongCat-Video [63] demonstrates robust multi-reward RLHF in foundational video models by introducing crucial stabilization techniques—specifically, employing max group standard deviation to bound reward variances within groups, and utilizing policy and KL loss reweighting to dynamically balance optimization and prevent reward hacking. Building upon these advancements, our method synergistically integrates the efficient mixed sampling framework of MixGRPO [51] with the max group standard deviation and policy/KL loss reweighting strategies from LongCat-Video [63], achieving highly stable and computationally efficient policy optimization.

SDE Sampling

3D Evaluator

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

|Rollout|Rotation Error (°)|Translation Error （m）|
|---|---|---|
|1-st|2.73|0.028|
|2-nd|1.91|0.047|
|…|…|…|
|g-th|3.45|0.056|

[Figure 47]

[Figure 48]

- 1-st Rollout

- 2-nd Rollout

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Video Diffusion Model

[Figure 56]

[Figure 57]

[Figure 58]

Input condition video

[Figure 59]

[Figure 60]

Rescaled Trajectory

|[Figure 61]|
|---|

[Figure 62]

[Figure 63]

Advantage aggregation

[Figure 64]

[Figure 65]

(combine rotation & translation errors)

[Figure 66]

Scale Factor

𝑠 ,𝑠

Rescale

###### GRPO Update

[Figure 67]

[Figure 68]

g-th Rollout

|[Figure 69]|
|---|

[Figure 70]

dataset raw trajectory

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

rescaled trajectory

evaluated trajectory

Camera-annotated data Truncated Gaussian Sampling

- Figure 2: Geo-Align pipeline. Given a conditioning video, we sample a camera trajectory from other camera-annotated data and scale it to a plausible range, with the scaling factor drawn from a truncated Gaussian distribution. After the model generates a set of rollout videos, a metric 3D evaluator assesses the camera trajectory of each sample to compute geometry rewards. Finally, the model is optimized via Group Relative Policy Optimization [45].

### 3 Methodology

#### 3.1 Overview

Given an input conditioning video and a user-specified, unseen camera trajectory, we aim to re-render and generate a novel view video sequence. Formally, let x1:N denote the conditioning video of length N, and c be the corresponding text prompt. To guide the generation process along a designated path, the model is additionally conditioned on a target camera trajectory Ptgt1:N, including target camera intrinsic parameters Ktgt1:N and extrinsic parameters Etgt1:N.

Our framework is built upon a pretrained video world model, denoted as Wθ. During the iterative generation process (e.g., diffusion or flow matching), the model predicts the denoised representation

(or velocity vector) given a noisy latent zt at timestep t. The conditional generation process can be formulated as:

vˆθ = Wθ(zt,t,C), (1)

where C = {x1:N,c,Ptgt1:N} encapsulates all the multimodal conditioning signals. Although finetuning video foundation models multi-view videos offers a viable solution to this task, the inherent

scarcity of such data remains a significant bottleneck. Relying solely on supervised fine-tuning often leads to geometric inconsistencies and suboptimal camera control. Therefore, the adoption of RL frees us from multi-view data dependencies, unlocking the potential to train on vastly larger and more diverse data. Our goal is to optimize the model parameters θ to maximize a composite reward function R, which comprehensively evaluates the alignment between the generated video y1:N and the target trajectory Ptgt1:N, as well as the overall video quality. The RL objective is defined as:

1:N∼Wθ(·|C) R(y1:N,Ptgt1:N) . (2)

Ey

max

θ

By directly optimizing this reward, the model is guided to strictly adhere to the prescribed target trajectory while maintaining superior spatiotemporal fidelity.

#### 3.2 Multi-Dimensional Reward Design

Verifiable Geometry Reward. To enforce rigorous spatial alignment between the generated video y1:N and the designated target trajectory, we introduce a verifiable Geometry Reward. We construct

our 3D evaluator upon MapAnything [7], a metric feed-forward 3D reconstruction model. By feeding the generated video into the 3D evaluator, we extract the predicted camera trajectory, comprising translations ˆt1:N and rotations Rˆ 1:N. The geometric discrepancy is then quantified against the input target trajectory (t1:N,R1:N) across two dimensions. Specifically, we compute the weighted Euclidean deviation for translation and the angular deviation for rotation:

Dtrans =

N

wi∥ti − ˆti∥2, (3)

i=1

N

Tr(R⊤i Rˆ i) − 1 2

, (4)

Drot =

wi arccos

i=1

where wi represents the temporal weight for the i-th frame. A key empirical observation motivates this weighting scheme: pretrained video generative models typically exhibit strong adherence to the conditioning trajectory in the initial frames, but suffer from severe error accumulation and spatial drift in the latter frames. Because the latter frames more accurately reflect the model’s true predictive capability and are the primary bottleneck in trajectory control, we design wi as a monotonically increasing function of time i (e.g., w1 < w2 < ··· < wN). This temporally progressive weighting mechanism explicitly penalizes long-term drift and forces the RL process to prioritize the optimization of challenging latter frames.

Perceptual and Aesthetic Reward. Optimizing solely for geometric alignment can inadvertently lead to reward hacking, resulting in perceptual degradation or unnatural artifacts. To preserve and enhance the visual fidelity of the synthesized video, we incorporate multidimensional aesthetic and quality rewards. First, we leverage the VideoAlign [8] evaluator to assess sequence-level dynamics, yielding a visual quality score (svis) and a motion quality score (smot). Furthermore, to guarantee superior single-frame visual aesthetics and high-frequency details, we utilize HPSv3 [9] to evaluate the perceptual quality of each individual frame.

#### 3.3 Flow Matching Optimization via GRPO

To efficiently optimize the pretrained flow matching model Wθ for trajectory-controlled generation, we employ Group Relative Policy Optimization [45]. Traditional PPO [64] relies on a memoryintensive value model for baseline estimation. GRPO resolves this memory constraint by removing the value model and leveraging the relative scores within a group of outputs to compute the advantage. Given the prohibitively long group sampling time of video generation models, we adopt the slidingwindow sampling strategy from MixGRPO [51]. This mechanism restricts stochastic sampling and gradient updates strictly to an active temporal window, significantly accelerating convergence. Furthermore, since directly summing multi-dimensional rewards is mathematically unstable, we aggregate the feedback in the advantage space. Following the max group standard deviation strategy (as in LongCat Video [63]), we robustly normalize each reward dimension k ∈ {rot,trans,vis,mot,hps} within a group of G sampled rollouts to prevent the over-amplification of low-variance noise:

rk(j) − µk max(σk,ϵ)

Aˆ(kj) =

, (5)

where µk and σk are the group mean and standard deviation. The total advantage A(totalj) is then formulated as:

A(totalj) =

λiAˆ(ij). (6)

i∈k

Standard GRPO incorporates a KL-divergence penalty to anchor the policy to the pretrained model. However, to maximize the model’s exploratory capability on entirely novel, out-of-distribution target camera trajectories, we remove this KL penalty. Incorporating a timestep-aware policy loss weight wt to balance gradients across diffusion stages as in LongCat Video [63], our final objective function is:

  1

 . (7)

G

wt min ρ(tj)A(totalj) ,clip ρ(tj),1 − ϵc,1 + ϵc A(totalj)

J (θ) = Et,z

t,C

G

j=1

where ρ(tj) denotes the policy probability ratio, and ϵc is the clipping hyperparameter.

#### 3.4 Metric-Aware Data Sampling Pipeline

Benefiting from the RL framework, our approach eliminates the reliance on paired ground-truth videos, unlocking the ability to train on large-scale, in-the-wild data.

Specifically, for the conditioning inputs, we utilize in-the-wild CityWalk [11] videos, which encompass a diverse array of static and dynamic scenes across both indoor and outdoor environments. The source camera trajectories for these uncalibrated conditioning videos are estimated using MapAnything [7]. Conversely, to inject a rich and complex repertoire of camera motions into the model, we sample the target trajectories from the OmniWorld [6] gaming dataset. However, drawing target trajectories directly from gaming data introduces critical optimization bottlenecks: these trajectories lack an absolute physical metric scale and frequently exhibit severe rotation. To guarantee the physical plausibility and kinematic stability of the target camera poses during RL training, we introduce a rescaling mechanism. First, we calculate the maximum frame-to-frame translation speed vtransmax and rotation speed vrotmax of the raw target trajectory Ptgt1:N:

vtransmax = max

∥ti+1 − ti∥2, (8)

i∈[1,N−1]

∥log(R⊤i Ri+1)∨∥2, (9)

vrotmax = max

i∈[1,N−1]

where ti and Ri denote the translation vector and rotation matrix at frame i, respectively, and (·)∨ maps the skew-symmetric matrix in the Lie algebra so(3) to its corresponding rotation vector. To ensure the trajectory speeds fall within a reasonable physical bound while maintaining data diversity, we sample target maximum speeds, τtrans and τrot, from Truncated Gaussian Distributions:

τtrans ∼ Ntrunc(µt,σt2,[at,bt]), (10) τrot ∼ Ntrunc(µr,σr2,[ar,br]), (11)

where [at,bt] and [ar,br] define the strict physical bounds for translation and rotation speeds, concentrating the sampling probability around natural human-walking or steady-cam speeds. Finally,

we compute the rescaling factors for translation and rotation, denoted as strans and srot respectively:

τtrans vtransmax + ϵ

τrot vrotmax + ϵ

, (12)

, srot =

strans =

where ϵ is a small constant to prevent division by zero. The target trajectory is then uniformly rescaled to yield the modified physical-aware trajectory P˜tgt1:N:

˜ti = stransti, (13)

R˜ i = exp(srot log(Ri)). (14) This rescaling protocol effectively eliminates unnatural camera jumps and aligns the synthetic gaming trajectories with real-world metric scales, significantly stabilizing the RL optimization landscape.

### 4 Experiments

#### 4.1 Implementation Details

We adopt ReDirector [2] which is based on Wan2.1 [65] 1.3B as our foundational pretrained video generation model. Following our proposed metric-aware data sampling pipeline, we continuously draw conditioning videos from the CityWalk [11] dataset and physically rescaled target trajectories from the OmniWorld [6] dataset. For the verifiable geometric reward, MapAnything [7] is employed as the frozen 3D evaluator. To preserve the strong spatiotemporal prior of the pretrained base model while enabling precise spatial control, we employ a parameter-efficient fine-tuning strategy: during the RL optimization, we solely update the weights of the self-attention layers, keeping all other network components strictly frozen. The model is configured to generate video sequences of N = 81 frames at a spatial resolution of 480 × 832. During inference and RL sampling, the continuous flow matching generation process is discretized into T = 25 denoising timesteps. For the GRPO [45] reinforcement learning framework, we follow the efficient mixed sampling framework of MixGRPO [51] and set the group size to G = 12 video rollouts per condition to compute the robust groupnormalized advantages. The network is optimized for a total of 140 RL iterations using a constant learning rate of η = 1 × 10−4. The post-training process is distributed across 64 NVIDIA A800 GPUs, consuming about 130 hours.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

InputVideoReCamMasterReDirectorOursInputVideoReCamMasterReDirectorOurs

[Figure 83]

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

[Figure 114]

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

- Figure 3: Qualitative results on the DAVIS [10] dataset. Geo-Align demonstrates superior capabilities in maintaining geometric consistency between the foreground subject and the background, whereas other methods suffer from varying degrees of distortion.

- Table 1: Quantitative comparison results on different metrics. Through the proposed reinforcement learning framework, our model yields marked enhancements across all quantitative metrics.

Visual Quality ↑ Geometric Consistency Camera Accuracy Subject Background Aesthetic Imaging Temporal Motion

Method Type

Consistency Consistency Quality Quality Flickering Smoothness Dyn-MEt3R ↑ MEt3R ↓ TransErr ↓ RotErr ↓

CogNVS [4] explicit warping 0.9134 0.9342 0.4830 0.6398 0.9460 0.9768 0.8037 0.2736 0.0367 6.9499 TrajectoryCrafter [3] explicit warping 0.9145 0.9331 0.5252 0.6394 0.9432 0.9800 0.8244 0.2136 0.0293 10.4340

Recammaster [1] implicit condition 0.9050 0.9190 0.5146 0.6611 0.9644 0.9867 0.7971 0.3485 0.0245 2.3175 Redirector [2] implicit condition 0.9098 0.9150 0.5141 0.6821 0.9537 0.9856 0.8497 0.3130 0.0149 1.4635 Ours implicit condition 0.9151 0.9179 0.5168 0.6842 0.9548 0.9862 0.8573 0.3077 0.0129 1.3645

#### 4.2 Baselines

We compare our method against two categories of state-of-the-art baselines. The first category comprises explicit warping-based methods, specifically TrajectoryCrafter [3] and CogNVS [4]. As these models are limited to generating fewer than 49 frames during inference. The second category consists of models conditioned on implicit camera extrinsics, including ReCamMaster [1] and ReDirector [2], which are capable of generating 81 or more frames.

###### Table 2: Quantitative comparison results across different camera speeds. Our model consistently outperforms baseline method across varying camera speeds.

Visual Quality ↑ Geometric Consistency Camera Accuracy Subject Background Aesthetic Imaging Temporal Motion

Method Camera Speed

Consistency Consistency Quality Quality Flickering Smoothness Dyn-MEt3R ↑ MEt3R ↓ TransErr ↓ RotErr ↓

Redirector [2] 1.0 0.9130 0.9143 0.5168 0.6755 0.9540 0.9857 0.8555 0.3206 0.0140 1.3689 Ours 1.0 0.9178 0.9174 0.5191 0.6782 0.9552 0.9862 0.8638 0.3150 0.0123 1.2156

- Redirector [2] 1.5 0.8922 0.9004 0.5085 0.6713 0.9486 0.9838 0.8368 0.3354 0.0157 1.5476

- Ours 1.5 0.8983 0.9035 0.5111 0.6741 0.9495 0.9844 0.8458 0.3307 0.0129 1.5491

Redirector [2] 2.0 0.8726 0.8881 0.4984 0.6624 0.9457 0.9820 0.8249 0.3496 0.0161 1.9246

- Ours 2.0 0.8783 0.8911 0.5017 0.6668 0.9469 0.9828 0.8338 0.3444 0.0153 1.8821

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

- Figure 4: More visualization results on CityWalk [11] dataset. For each example, the top row illustrates the input video, whereas the bottom row visualizes our results following the target trajectory.

#### 4.3 Evaluation protocol

We follow the evaluation protocol of ReDirector [2], using 50 videos from the DAVIS dataset. By applying 10 ReCamMaster [1] camera trajectories per video, we construct 500 test cases with lengths varying from tens to nearly a hundred frames. We restrict TrajectoryCrafter [3] and CogNVS [4] to a maximum of 49 frames to prevent performance degradation; for the other methods, the evaluated frame length matches the dataset defaults.

For our metrics, we use ViPE [66] to extract camera parameters to compute TransErr and RotErr. We also apply MEt3R [67] for input video consistency, Dyn-MEt3R [68] for geometric consistency, and VBench [69] for comprehensive aesthetic evaluation. Moreover, to evaluate complex trajectories, we compare our method with our base model (ReDirector [2]) under different camera speeds. Since large camera movements can produce consecutive featureless frames (e.g., sky, water or non-textured

###### Table 3: Qualitative ablation results on DAVIS [10] dataset. Beyond improving geometric consistency and camera accuracy, the geometry reward also yields improvements in visual quality.

Visual Quality ↑ Geometric Consistency Camera Accuracy Subject Background Aesthetic Imaging Temporal Motion

Method

Consistency Consistency Quality Quality Flickering Smoothness Dyn-MEt3R ↑ MEt3R ↓ TransErr ↓ RotErr ↓

baseline (Redirector [2]) 0.9098 0.9150 0.5141 0.6821 0.9537 0.9856 0.8497 0.3130 0.0149 1.4635 Video quality reward 0.9130 0.9174 0.5145 0.6818 0.9549 0.9861 0.8531 0.3117 0.0147 1.6082 Full reward 0.9147 0.9181 0.5163 0.6828 0.9548 0.9863 0.8550 0.3082 0.0140 1.3895

ground) that lead to ViPE [66] estimation failures, we perform this speed-wise comparison on a reliable subset of 40 DAVIS [10] videos to ensure fairness, applying 10 ReCamMaster [1] camera trajectories per video.

#### 4.4 Main Results

Table 1 demonstrates the effectiveness of our reinforcement learning framework. Compared to the baseline [2], our RL-trained model exhibits notable improvements in Geometric Consistency, Camera Accuracy, and overall video quality. The gains in Geometric Consistency and Camera Accuracy are particularly substantial, validating that the incorporation of the geometry reward successfully guides the model to follow target trajectories with higher precision. Furthermore, Table 2 presents the evaluation results across varying camera speeds. Our model consistently outperforms the baseline under more complex target trajectories, clearly illustrating the enhanced robustness across diverse trajectory conditions achieved through our RL training.

Qualitatively, Figure 3 visualizes our results on the DAVIS [10] dataset. Our method demonstrates superior capabilities in maintaining geometric consistency between the foreground subject and the background. Notably, under large camera motions, baseline methods such as ReCamMaster [1] and ReDirector [2] frequently suffer from severe degradation, including subject disappearance and background blurring. In contrast, our approach significantly mitigates these collapse scenarios, robustly preserving both subject and background details. We attribute this enhanced stability to our RL training paradigm, which effectively exposes the model to a broader and more complex distribution of camera trajectories.

#### 4.5 Ablation Study

To validate the effectiveness of our proposed mechanisms, we conduct an ablation study by retraining the model under different reward configurations. All models are trained for 140 steps on 16 A800 GPUs and evaluated against the baseline (ReDirector [2]) using the identical evaluation protocol and dataset as in Table 1. Specifically, we compare the full reward formulation (incorporating both aesthetic and geometry rewards) against an ablated variant trained exclusively with the aesthetic reward. As shown in Table 3, relying solely on the aesthetic reward yields marginal overall performance improvements and even leads to a noticeable degradation in rotation accuracy. In contrast, integrating the geometry reward not only substantially enhances camera accuracy but also contributes to better visual quality and geometric consistency.

### 5 Conclusion

In this work, we introduce Geo-Align, a reinforcement learning framework designed for cameracontrolled video retake. First, we design a reward mechanism using a metric 3D evaluator to explicitly optimize how accurately the generated video follows the target camera trajectory. Second and crucially, we propose a hybrid data sampling strategy combining real videos and scaled camera trajectories from synthetic data. This effectively mitigates the field’s reliance on scarce time-synchronized multi-view video data, allowing the model to train on far more diverse scenes and complex trajectories. Empirical results validate that our approach consistently enhances camera control precision, consistency, and visual quality over existing baselines.

### References

- [1] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14834–14844, 2025.
- [2] Byeongjun Park, Byung-Hoon Kim, Hyungjin Chung, and Jong Chul Ye. Redirector: Creating any-length video retakes with rotary camera encoding. arXiv preprint arXiv:2511.19827, 2025.
- [3] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 100–111, 2025.
- [4] Kaihua Chen, Tarasha Khurana, and Deva Ramanan. Reconstruct, inpaint, test-time finetune: Dynamic novel-view synthesis from monocular videos. arXiv preprint arXiv:2507.12646, 2025.
- [5] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer, 2024.
- [6] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, et al. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025.
- [7] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025.
- [8] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025.
- [9] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.
- [10] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.
- [11] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025.
- [12] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22875–22889, 2025.
- [13] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.
- [14] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024.
- [15] Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim, Soonwoo Kwon, and Changick Kim. Splatflow: Multi-view rectified flow model for 3d gaussian splatting synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21524–21536, 2025.
- [16] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2005–2015, 2025.
- [17] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22831–22840, 2025.

- [18] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European conference on computer vision, pages 18–35. Springer, 2024.
- [19] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20406–20417, 2024.
- [20] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [22] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [23] Hyeonho Jeong, Suhyeon Lee, and Jong Chul Ye. Reangle-a-video: 4d video generation as video-tovideo translation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11164–11175, 2025.
- [24] Dongyue Lu, Ao Liang, Tianxin Huang, Xiao Fu, Yuyang Zhao, Baorui Ma, Liang Pan, Wei Yin, Lingdong Kong, Wei Tsang Ooi, et al. See4d: Pose-free 4d generation via auto-regressive video inpainting. arXiv preprint arXiv:2510.26796, 2025.
- [25] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2050–2062, 2025.
- [26] Bill Triggs, Philip F McLauchlan, Richard I Hartley, and Andrew W Fitzgibbon. Bundle adjustment—a modern synthesis. In International workshop on vision algorithms, pages 298–372. Springer, 1999.
- [27] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3d. In ACM SIGGRAPH 2006 Papers, pages 835–846, 2006.
- [28] Changchang Wu. Towards linear-time incremental structure from motion. In 2013 International Conference on 3D Vision-3DV 2013, pages 127–134. IEEE, 2013.
- [29] Johannes L Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016.
- [30] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 20697–20709, 2024.
- [31] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. Stream3r: Scalable sequential 3d reconstruction with causal transformer, 2025. URL https://arxiv.org/abs/2508.10893.
- [32] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory, 2024. URL https://arxiv. org/abs/2408.16061.
- [33] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer, 2026. URL https://arxiv.org/abs/2507.11539.
- [34] Yuzheng Liu, Siyan Dong, Shuzhe Wang, Yingda Yin, Yanchao Yang, Qingnan Fan, and Baoquan Chen. Slam3r: Real-time dense scene reconstruction from monocular rgb videos, 2025. URL https: //arxiv.org/abs/2412.09401.
- [35] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10510–10522, 2025.
- [36] Zizun Li, Jianjun Zhou, Yifan Wang, Haoyu Guo, Wenzheng Chang, Yang Zhou, Haoyi Zhu, Junyi Chen, Chunhua Shen, and Tong He. Wint3r: Window-based streaming reconstruction with camera token pool. arXiv preprint arXiv:2509.05296, 2025.

- [37] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views, 2026. URL https://arxiv.org/abs/2502.12138.
- [38] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass, 2025. URL https://arxiv.org/abs/2501.13928.
- [39] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. Vggt-long: Chunk it, loop it, align it – pushing vggt’s limits on kilometer-scale long rgb sequences, 2026. URL https://arxiv.org/abs/2507.16443.
- [40] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r, 2024. URL https://arxiv.org/abs/2406.09756.
- [41] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion,

2025. URL https://arxiv.org/abs/2410.03825.

- [42] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.
- [43] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Permutation-Equivariant Visual Geometry Learning. arXiv preprint arXiv:2507.13347, 2025.
- [44] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.
- [45] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [46] Senyu Fei, Siyin Wang, Li Ji, Ao Li, Shiduo Zhang, Liming Liu, Jinlong Hou, Jingjing Gong, Xianzhong Zhao, and Xipeng Qiu. Srpo: Self-referential policy optimization for vision-language-action models, 2025. URL https://arxiv.org/abs/2511.15605.
- [47] Zigang Geng, Yibing Wang, Yeyao Ma, Chen Li, Yongming Rao, Shuyang Gu, Zhao Zhong, Qinglin Lu, Han Hu, Xiaosong Zhang, Linus, Di Wang, and Jie Jiang. X-omni: Reinforcement learning makes discrete autoregressive image generative models great again, 2025. URL https://arxiv.org/abs/ 2507.22058.
- [48] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation, 2025. URL https://arxiv.org/abs/2505.07818.
- [49] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [50] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [51] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.
- [52] Haoyang He, Jay Patrikar, Dong-Ki Kim, Max Smith, Daniel McGann, Ali-akbar Agha-mohammadi, Shayegan Omidshafiei, and Sebastian Scherer. Grndctrl: Grounding world models via self-supervised reward alignment. arXiv preprint arXiv:2512.01952, 2025.
- [53] Zhaoqing Wang, Xiaobo Xia, Zhuolin Bie, Jinlin Liu, Dongdong Yu, Jia-Wang Bian, and Changhu Wang. Taming camera-controlled video generation with verifiable geometry reward. arXiv preprint arXiv:2512.02870, 2025.
- [54] Wenhang Ge, Guibao Shen, Jiawei Feng, Luozhou Wang, Hao Lu, Xingye Tian, Xin Tao, and Ying-Cong Chen. Campilot: Improving camera control in video diffusion model with efficient camera reward feedback,

###### 2026. URL https://arxiv.org/abs/2601.16214.

- [55] Weijie Wang, Xiaoxuan He, Youping Gu, Yifan Yang, Zeyu Zhang, Yefei He, Yanbo Ding, Xirui Hu, Donny Y. Chen, Zhiyuan He, Yuqing Yang, and Bohan Zhuang. World-r1: Reinforcing 3d constraints for text-to-video generation, 2026. URL https://arxiv.org/abs/2604.24764.
- [56] Fan Wu, Jiacheng Wei, Ruibo Li, Yi Xu, Junyou Li, Deheng Ye, and Guosheng Lin. Ic-world: In-context generation for shared world modeling, 2025. URL https://arxiv.org/abs/2512.02793.
- [57] Orest Kupyn, Fabian Manhardt, Federico Tombari, and Christian Rupprecht. Epipolar geometry improves video generation models, 2025. URL https://arxiv.org/abs/2510.21615.
- [58] Tengjiao Yin, Jinglei Shi, Heng Guo, and Xi Wang. Vigor: Video geometry-oriented reward for temporal generative alignment, 2026. URL https://arxiv.org/abs/2603.16271.
- [59] Zhaochong An, Orest Kupyn, Théo Uscidda, Andrea Colaco, Karan Ahuja, Serge Belongie, Mar GonzalezFranco, and Marta Tintore Gazulla. Vggrpo: Towards world-consistent video generation with 4d latent reward, 2026. URL https://arxiv.org/abs/2603.26599.
- [60] Tianyi Yan, Wencheng Han, Xia Zhou, Xueyang Zhang, Kun Zhan, Cheng zhong Xu, and Jianbing Shen. Rlgf: Reinforcement learning with geometric feedback for autonomous driving video generation, 2025. URL https://arxiv.org/abs/2509.16500.
- [61] Hongyang Du, Junjie Ye, Xiaoyan Cong, Runhao Li, Jingcheng Ni, Aman Agarwal, Zeqi Zhou, Zekun Li, Randall Balestriero, and Yue Wang. Videogpa: Distilling geometry priors for 3d-consistent video generation, 2026. URL https://arxiv.org/abs/2601.23286.
- [62] Haoyu Wu, Diankun Wu, Tianyu He, Junliang Guo, Yang Ye, Yueqi Duan, and Jiang Bian. Geometry forcing: Marrying video diffusion and 3d representation for consistent world modeling, 2025. URL https://arxiv.org/abs/2507.07982.
- [63] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.
- [64] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.
- [65] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL https://arxiv.org/abs/2503.20314.
- [66] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025.
- [67] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring multi-view consistency in generated images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6034–6044, 2025.
- [68] Byeongjun Park, Hyojun Go, Hyelin Nam, Byung-Hoon Kim, Hyungjin Chung, and Changick Kim. Steerx: Creating any camera-free 3d and 4d scenes with geometric steering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27326–27337, 2025.
- [69] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

### A Appendix

#### A.1 Limitations

Our reinforcement learning approach improves the accuracy of camera trajectory adherence in generated videos while simultaneously enhancing overall video quality. However, the model remains susceptible to failure when faced with excessively fast rotations, large translations, or large foreground objects close to the camera. Furthermore, inputs dominated by dynamic objects frequently lead to artifacts, such as the flickering or vanishing of those dynamic elements. Thirdly, the RL training process is highly time-consuming because it requires sampling the model to generate multiple complete videos for each batch, and the video generation process itself is inherently slow. Therefore, accelerating the RL training process remains a compelling direction for future exploration.

#### A.2 Assets and Licenses

We summarize the assets used in our research, including their licenses and accessibility, in Table 4. All assets are used in accordance with their respective terms.

Table 4: Summary of used assets (datasets, models, and code).

Asset Citation License Link/Source

ReCamMaster [1] MIT License https://github.com/KlingAIResearch/ReCamMaster ReDirector [2] Apache License 2.0 https://github.com/byeongjun-park/ReDirector TrajectoryCrafter [3] Apache License 2.0 https://github.com/TrajectoryCrafter/TrajectoryCrafter MixGRPO [51] Non-Commercial License v1.1.1 https://github.com/Tencent-Hunyuan/MixGRPO MapAnything [7] Apache License 2.0 https://github.com/facebookresearch/map-anything OmniWorld [6] BY-NC-SA 4.0 https://github.com/yangzhou24/OmniWorld CityWalk [11] No commercial license https://github.com/Lixsp11/sekai-codebase ViPE [66] Apache License 2.0 https://github.com/nv-tlabs/vipe MEt3R [67] MIT License https://github.com/mohammadasim98/met3r Dyn-MEt3R [68] CC BY 4.0 https://github.com/byeongjun-park/SteerX VBench [69] Apache License 2.0 https://github.com/Vchitect/VBench

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

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

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

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

###### Figure 5: More qualitative comparison on DAVIS [10] dataset. For each example, the top row illustrates the input video, while the second and third rows present the results of ReDirector [2] and our model, respectively.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

InputVideoOurs

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

###### Figure 6: Failure Case. the model remains susceptible to failure when faced with excessively fast rotations, large translations, or large foreground objects close to the camera.

