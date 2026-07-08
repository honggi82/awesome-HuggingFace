# arXiv:2601.05239v1[cs.CV]8Jan2026

## Plenoptic Video Generation

Xiao Fu1,2, Shitao Tang1, Min Shi1,3, Xian Liu1, Jinwei Gu1, Ming-Yu Liu1, Dahua Lin2, Chen-Hsuan Lin1 1NVIDIA, 2The Chinese University of Hong Kong, 3Georgia Institute of Technology

###### InputCameras Frames

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Input Video

[Figure 6]

Prompt: a humanoid robot with a sleek, silver body and black joints performs a series of dance-like movements

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Rotation Right

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Arc Left

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Azimuth Left

[Figure 24]

Figure 1. We present PlenopticDreamer, a generative framework that re-renders input video under novel camera trajectories while preserving long-term spatio-temporal memory in hallucinated regions across overlapping views, thereby producing coherent plenoptic functions (see robot’s right side, highlighted in red dashed boxes across three trajectories). Please refer to our website for more results.

### Abstract

Camera-controlled generative video re-rendering methods, such as ReCamMaster, have achieved remarkable progress. However, despite their success in single-view setting, these works often struggle to maintain consistency across multi-view scenarios. Ensuring spatio-temporal coherence in hallucinated regions remains challenging due to the inherent stochasticity of generative models. To address it, we introduce PlenopticDreamer, a framework that synchronizes generative hallucinations to maintain spatiotemporal memory. The core idea is to train a multi-insingle-out video-conditioned model in an autoregressive manner, aided by a camera-guided video retrieval strategy that adaptively selects salient videos from previous generations as conditional inputs. In addition, Our training incorporates progressive context-scaling to improve convergence, self-conditioning to enhance robustness against long-range visual degradation caused by error accumulation, and a long-video conditioning mechanism to support extended video generation. Extensive experiments on the

Basic and Agibot benchmarks demonstrate that PlenopticDreamer achieves state-of-the-art video re-rendering, delivering superior view synchronization, high-fidelity visuals, accurate camera control, and diverse view transformations (e.g., third-person → third-person, and headview → gripper-view in robotic manipulation). Project page: https://research.nvidia.com/labs/ dir/plenopticdreamer/.

### 1. Introduction

Video generation [3, 9, 20, 21, 23, 26, 34, 45, 51, 52, 54] has become increasingly prevalent in content creation and social media. As video frames result from camera projections of scene radiance, they can be interpreted as discrete samples of the underlying plenoptic function [8, 39]. Consequently, effective control of camera motion [4, 24, 46, 58] is essential for shaping the captured light field, emphasizing visual focus, and guiding the viewer’s attention.

Recently, camera-controlled generative video re-

rendering, which aims to synthesize novel videos along arbitrary camera trajectories while preserving the original content, has attracted significant attention, supporting applications such as immersive content creation and embodied AI. Representative methods, including ReCamMaster [6] and TrajectoryCrafter [66], achieve promising results on their curated real-world or synthetic datasets. However, these methods primarily succeed in the single-view setting and struggle in multi-view scenarios, which are essential for reconstructing a holistic representation of the scene. Specifically, they fail to maintain consistent spatio-temporal hallucinations in regions unseen from the source view. The inherent stochasticity of diffusion models, combined with their limited long-range spatial memory, leads to geometric misalignment and view desynchronization across different camera-conditioned generations.

To address it, we present PlenopticDreamer, a cameracontrolled generative video re-rendering framework that explicitly enforces spatio-temporal memory for consistent scene generation. Unlike prior single-shot methods that generate each view independently, PlenopticDreamer adopts an autoregressive, multi-in–single-out formulation. At each step, it retrieves a set of previously generated video–camera pairs from a memory bank and conditions the next generation on these retrieved contexts. This design enables synchronized hallucinations across time and viewpoints while preserving scene geometry and motion dynamics. Video context retrieval is guided by a 3D field-of-view (FOV) mechanism that evaluates spatial co-visibility to select the most relevant past video segments.

In addition, PlenopticDreamer introduces two training strategies that substantially improve robustness and convergence. First, progressive context-scaling stabilizes optimization by gradually increasing the number of conditioning videos during training, enabling the model to learn context-aware reasoning across short to long temporal horizons. Second, self-conditioned training mitigates error accumulation in autoregressive generation by fine-tuning the model on its own synthesized outputs. Together, these strategies facilitate stable video synthesis while preserving spatial alignment and temporal consistency. We further propose a long-video conditioning mechanism to extend the model’s capability to render longer video sequences.

We evaluate our method on two benchmarks: (1) a Basic benchmark covering diverse in-the-wild scenes, and (2) an Agibot benchmark [10] focusing on robotic manipulation. Experimental results show that PlenopticDreamer achieves state-of-the-art performance in view synchronization while maintaining accurate camera control and highfidelity visual quality. In summary, our contributions are:

- 1. We present PlenopticDreamer, the first cameracontrolled generative video re-rendering framework with long-term spatio-temporal memory.

- 2. We propose an autoregressive architecture with a 3D FOV–based video retrieval mechanism for scalable, coherent multi-camera generation. We incorporate progressive context-scaling and self-conditioned training strategies to enhance stability and long-term consistency.
- 3. Extensive experiments show that our method achieves state-of-the-art performance in video re-rendering, including view synchronization, camera accuracy, and visual fidelity. It supports diverse camera transformations (e.g., third-person → third-person, head-view → gripper-view) and enables long video generation.

### 2. Related Work

Camera-Controlled Video Generation. Effective control of camera motion has received significant attention in video generation. Existing approaches can be broadly categorized into three directions: (1) single-view generation: works rely on explicit 6DoF camera poses [46, 58] or pixel-wise Pl¨ucker raymaps [4, 5, 24, 63, 68] to guide text- or image-to-video synthesis, achieving view control. Methods [27, 28, 40, 64] explore training-free strategies for camera manipulation, and a few [19, 57] extend these mechanisms to control object motion via camera trajectories. (2) multi-view video generation: methods aim to maintain cross-view consistency, ranging from object-level synthesis [36, 62] to scene-level reconstruction [7, 35]. (3) video-to-video re-rendering: some approaches [6, 53] perform implicit re-rendering with minimal 3D supervision, whereas others [22, 46, 61, 62, 66] project video context into 3D representations to synthesize novel views. Despite these advances, none integrate memory mechanisms to maintain long-term spatio-temporal coherence across multiple views. In contrast, our PlenopticDreamer introduces the first memory-based framework for generative video-tovideo re-rendering, achieving coherent multiview synthesis. Memory Mechanism for Video Generation. Building long-term memory is essential for coherent video generation [33, 50]. Existing approaches can be broadly categorized into four types: (1) frame-level memory: methods such as [13, 14, 47, 60, 65] store key historical frames and retrieve the top-k relevant ones via camera-pose similarity for conditioning; (2) latent-level memory: approaches including [11, 37, 43, 44, 67] maintain hierarchical memory capturing long-term coarse tokens and short-term finegrained tokens, adaptively retrieving salient features during inference; (3) 3D-level memory: works like VMem [38] and SPMem [59] reconstruct 3D structures (e.g., surfels or point clouds) to store video context and render geometry-aware representations for novel-view synthesis; (4) network-level memory: TTT-Video [16] leverages Test-Time Training (TTT) layers to record input tokens and update model weights. In contrast, our PlenopticDreamer introduces aN explicit video-based retrieval mechanism conditioning gen-

Trainable Frozen

Prompt

Loss

Feed-Forward Feed-Forward

DiT Blocks

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Cross-Attention Cross-Attention

Add Noise

Self-Attention Self-Attention

Plücker Coords. & Patchify

[Figure 33]

[Figure 34]

Patchify

Cameras

Context Parallelism (N Splits)

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Addition

PlückerRaymap

(Temporal Concatenation)

Top-k Retrieval (3D FOV-based)

[Figure 43]

[Figure 44]

Target Video

Context Videos

(c) DiT Block

(a) Video&Camera Memory Bank (b) Autoregressive Multi-Camera Video Generator

Figure 2. PlenopticDreamer Framework. Its core is an autoregressive multi-camera video generator that retrieves k video–camera pairs {(Pn, Vn)}kn=1 from the memory bank using a 3D FOV–based retrieval strategy. Conditioned on these retrieved pairs and the target camera Pk+1, the model performs noisy scheduling and learnable reconstruction to generate the target video Vk+1. To enable long video generation, a portion of the preceding frames in Vk+1 is preserved as clean inputs at a certain ratio during training. Within each DiT block, temporal concatenation is applied to form video tokens x as in-context condition.

eration on camera-guided selection of past video segments.

### 3. Method

Our goal is to enable generative video-to-video re-rendering with spatio-temporal memory, interpretable as generating the space-time dependent plenoptic function of a scene. We first introduce the preliminaries and task formulation in Sec. 3.1, followed by our autoregressive modeling paradigm and video retrieval mechanism in Sec. 3.2. Finally, we describe the enhanced training strategies in Sec. 3.3. The overall framework is illustrated in Fig. 2.

#### 3.1. Preliminary and Problem Definition

Flow-based Video Diffusion Transformer (DiT). We conduct experiments using a video diffusion transformer model under the flow-matching paradigm [18, 41]. Given a data sample x0 ∼ p(x), a noise sampler ϵ ∼ N (0,I), and a continuous time variable t ∈ [0,1], the forward process linearly interpolates between data and noise distribution, i.e.,

xt = (1 − t)x0 + tϵ,vt = ϵ − x0 (1)

where vt is the GT velocity field. The denoising process is solved by an ordinary differential equation (ODE):

###### dxt = vΘ (xt,t,c)dt (2)

where vΘ(·) represents the predicted velocity function parameterized by a transformer-based network [45], and c is the conditional signal (e.g., video context). The model is optimized using the following flow-matching objective:

L(Θ) = Ex,ϵ,c,t ∥vΘ (xt,t,c) − vt∥2 (3)

During training, the timestep t can be biased toward higher noise levels to encourage robust reconstruction under degraded spatio-temporal correlations.

Task Formulation and Notations. Given a source video Vs ∈ RF×C×H×W and a set of N target camera trajectories {Pnt }Nn=1, each Pt specified by extrinsic parameters Ct = [Rt;Tt] ∈ RF×3×4 and intrinsics Kt ∈ R3×3, our objective is to synthesize N target videos {Vtn}Nn=1, with each Vtn ∈ RF×C×H×W sharing the same context as the input video while corresponding to a distinct virtual camera trajectory. The generated videos are required to maintain the source content fidelity and exhibit synchronized spatialtemporal consistency across viewpoints, particularly in hallucinated regions. We employ a standard pinhole camera model (zero horizontal and vertical skew) for generation. The overall generative process f(·) is formulated as

f(·) : c,Vs,Ps,{Pnt }Nn=1 → {Vtn}Nn=1 (4)

where c denotes the video caption and Ps is the camera trajectory of the source video. Additionally, a variational autoencoder with encoder E(·) and decoder D(·) is employed to map the videos between pixel-space and latent-space (Vs ↔ zs,{Vtn}Nn=1 ↔ {znt }Nn=1, where z ∈ Rf×h×w×c).

#### 3.2. Injecting Conditions into Video DiT

To effectively guide the video model with target conditions, including the source video and target camera trajectories, we adopt an in-context conditioning strategy. [6, 25, 32, 65]. Native Solution. A straightforward approach is to enlarge the context window by extending the number of input

videos from 1 to N, following ReCamMaster [6]:

xs = patchify (zs),xnt = patchify (znt ),n = 1,...,N x = [xs,x1,...,xN]frame-dim ∈ R[(N+1)×f]×(h×w)×c,

(5) where x denotes the input video tokens fed into the DiT block. While this strategy can be effective for small N (e.g.,

- 2 or 3) under low-resolution settings (≤480p), it rapidly becomes computationally prohibitive and prone to out-ofmemory (OOM) failures as N or video resolution increases. Autoregressive Generation Paradigm. Inspired by the effectiveness of the autoregressive paradigm for long-context modeling [1, 12, 30, 56, 65], we reformulate video generation as a sequential process instead of performing singleshot inference. Specifically, we generate one video at a time and produce all videos in a sequential manner. Accordingly, we rewrite Eq. 4 as:

f(·) : c,{(Pn,Vn)}kn=1,Pk+1 → Vk+1,k = 1,...,N − 1

(6)

where {(Pn,Vn)}kn=1 denotes previously generated videos and their cameras and (Ps,Vs) is regarded as (P1,V1), and k is the model context size. We adopt temporal concatenation strategy to form conditional video tokens:

x = [x1,...,xk+1]frame-dim ∈ R[(k+1)×f]×(h×w)×c (7) Camera Conditioning. To encode camera information, we represent it using Pl¨ucker raymaps [48], mapping pixels to 6D ray representations: Pn = (Cn ∈ Rf×3×4,K ∈ R3×3) → P¨n ∈ Rf×H×W×6,n = 1,...,k + 1. These raymaps are then temporally concatenated and patchified. A camera projection layer Ecam(·) is introduced to align the raymap dimensionality with that of the video latents. The resulting raymap tokens are channel-wise added to the video tokens before self-attention layer, enabling DiT to integrate camera pose information.

- 3D FOV–based Video Retrieval. A key challenge in this autoregressive framework lies in selecting the most salient k videos from the previously video pool as conditioning inputs for the next generation step. Given that each video corresponds to a distinct camera trajectory, we adopt a 3D Field-of-View (FOV) retrieval mechanism to identify the most relevant candidates. Specifically, we compute videolevel similarity via spatial co-visibility across all frames, and select the top-k context videos, as illustrated in Algorithm 1 and Fig. 3. When the number of context videos is less than k, we replicate input video-camera pair (P1,V1) to match the required context length. Furthermore, when the number of retrieved videos l exceeds the model’s context capacity k, a divide-and-conquer inference strategy is employed to cover as diverse viewpoints as possible and minimize viewpoint overlap, as described in Algorithm 2. Here the trajectory fusion process results in a merged trajectory roughly spanning the FOV of all inputs.

[Figure 45]

[Figure 46]

###### (a) Frame-level (previous) (b) Video-level (ours)

Figure 3. FOV-based Retrieval Comparison. Unlike prior frame-level retrieval methods [60, 65], ours computes robust video-level similarity by averaging frame-wise similarities.

Algorithm 1 Video Retrieval Algorithm Input:

- • Memory bank of K videos {(Vn,Pn)}Kn=1
- • Target camera trajectory PK+1
- • Maximum retrieved video number k
- • Near/Far plane distances Dn,Df
- • Monte Carlo sampling points P

Output: Top-k retrieved videos

- 1: Initialize similarity set S ← ∅
- 2: for n = 1 to K do
- 3: Initialize similarity Sn ← 0
- 4: for f = 1 to F do
- 5: Construct f-th camera frustum of Pn and PK+1
- 6: Perform Monte Carlo sampling within near/far planes of each frustum
- 7: Count visible points Pn, PK+1 in other’s frustum
- 8: Update similarity: Sn ← Sn + P

n+PK+1 2P×F

- 9: end for
- 10: Append Sn to S
- 11: end for
- 12: Select indices of the top-k values in S
- 13: return retrieved k videos

Autoregressive Long Video Generation. For input video exceeding the model’s temporal window, we partition them into overlapping sub-chunks {Vsm}Mm=1, where consecutive chunks share a set of frames from the latter portion of the previous chunk to preserve temporal continuity. Unlike the formulation in Eq. 6, where Vk+1 is generated from pure noise, we incorporate the overlapping frames as additional conditioning:

f(·) : c,{(Pn,m,Vn,m)}kn=1,Pk+1,m,V˜ k+1,m → Vk+1,m

(8) where k = 1,...,N − 1,m = 1,...,M. Here V˜ k+1,m ∈ RF˜×C×H×W contains the F˜ overlapping frames, and m indexes the m-th sub-chunk Vsm from the source video. During inference, we sequentially generate the videos as follow:

→ V1,2 → V2,2... Start the next chunk Vs2

→ VN,M

V1,1 → V2,1 → ... → VN,1 Finish the first chunk Vs1

(9)

Algorithm 2 Divide-and-Conquer Inference Algorithm Input:

- • Top-l retrieved videos {(Vn,Pn)}ln=1 (sorted by ascending camera similarity)
- • Model context video size k
- • Target camera trajectory PL+1

Output: Target video VL+1.

- 1: while l > k do
- 2: Select the first m = min(l − k,k) videos to form context set V
- 3: while m < k do
- 4: Append (Ps,Vs) to V
- 5: m ← m + 1
- 6: end while
- 7: Merge trajectories in V to form Pmerge
- 8: Infer merged video Vmerge using (V,Pmerge)
- 9: Replace the first m elements with (Vmerge,Pmerge)
- 10: l ← l − m + 1
- 11: end while
- 12: Perform final inference to obtain VL+1
- 13: return target video VL+1

#### 3.3. Training Strategy

Progressive Training. The training objective corresponding to Eq. 6 is defined as:

L(Θ) = Eϵ,c,P,V,t vΘ {(Pn,Vn)}k+1n=1,t,c − vt 2

(10) We also incorporate the extended prediction V˜ k+1 from Eq. 8 into the loss at a certain ratio. In our empirical experiments, we observe that directly training the model with a large context size often leads to unstable convergence. To address this, we adopt a progressive training strategy: the model is first trained with a small context size (e.g., 1) and gradually scaled up as training stabilizes, until reaching the target context size k. This progressive scheme significantly improves convergence stability and accelerates training in later stages with larger contexts.

Self-conditioned Training. When the total generation length N becomes large, multiple inference steps are required, where previously generated videos are repeatedly used as conditioning inputs. This recursive dependency can lead to error accumulation due to propagation of imperfect generations. To alleviate this issue, we adopt a selfconditioned training strategy. Specifically, in the first training stage of Eq. 10, all conditioning videos are ground-truth samples. After convergence, the model is used to generate synthetic outputs from training-set input, which then replace the ground-truth conditions in the second training round. This iterative refinement improves model robustness to imperfect inputs during long-range inference.

### 4. Experiment

#### 4.1. Experiment Setting

Implementation Details. We adopt Cosmos-Predict2.52B [3] as the backbone. The generated videos have a resolution of 432×768 with 93 frames. We employ context parallelism [2, 3] to alleviate memory overhead and set the parallelism size to 8. Finetuning is conducted on 32 NVIDIA H100 GPUs with batch size 1 and a learning rate 2e-5. During finetuning, only the self-attention layers and camera encoder are updated, while all other parameters remain frozen. We post-train two model variants in Sec. 4.2 and Sec. 4.3.

Evaluation Metrics. We evaluate models from three aspects: 1) Visual Quality: PSNR and FVD measure pixeland frame-level fidelity, respectively. 2) Camera Accuracy: TransErr and RotErr [24] quantify translation and rotation errors. Dynamic poses are evaluated with ViPE [29], while static novel views (e.g., azimuth/elevation shifts) accessed with VGGT [55] for relative pose estimation. 3) Video Synchronization: RoMa [17] computes the number of matched pixels above a confidence threshold, denoted as Mat. Pix.

Baselines. We compare the proposed PlenopticDreamer with state-of-the-art camera-controlled generative video re-rendering methods: ReCamMaster [6], TrajectoryCrafter [66], and Trajectory-Attention [61]. All baselines are used with their best-performing settings from the official open-sourced models. For a fairer comparison, we also retrain ReCamMaster on Cosmos-Predict2.5 with Pl¨ucker raymaps on the same datasets, denoted as ReCamMaster*.

#### 4.2. Experiment on Basic Benchmark Experiment Details.

- • Functionality: The model performs third-view to thirdview transformations, such as left/right rotations, azimuth and elevation shifts, distance variations, and dynamic focal length changes.

- • Training Dataset: We use MultiCamVideo [6] and SynCamVideo [7], large-scale synthetic datasets comprising approximately 136K and 34K episodes, respectively, depicting human motion captured under dynamic and static camera trajectories across 40 synthetic 3D environments.

- • Training Details: The model context size k is set to 4. In the first stage, it is progressively trained for 10K, 4K, 1K, and 1K steps with context sizes 1–4, respectively. In the second stage, the model generates synthetic data from 1,000 scenes and is further trained for 2K steps.

- • Benchmark: We construct a Basic benchmark of 100 inthe-wild videos and 12 sequential camera trajectories.

Qualitative and Quantitative Comparison. As shown in Fig. 4 and Tab. 1, PlenopticDreamer achieves superior view synchronization with high-fidelity visuals compared to all baselines (see the painting and electrical outlet on the wall in the first example, the traffic light in the sec-

- Table 1. Quantitative Comparison on the Basic Benchmark. Ours consistently outperforms all baselines in view synchronization across all shots while maintaining high-fidelity visual quality and accurate camera accuracy. ReCamMaster* denotes a retrained version on Cosmos-Predict2.5 with Pl¨ucker raymaps, using the same combined datasets (MultiCamVideo + SyncCamVideo) for a fair comparison.

Visual Quality Camera Accuracy View Synchronization (Mat. Pix.(K) ↑)

Model FVD ↓ TransErr ↓ RotErr (rad) ↓ 3 Shots 6 Shots 9 Shots 12 Shots Trajectory-Attention [61] 734.1 0.77 0.26 22.7 26.9 28.8 29.1

TrajectoryCrafter [66] 665.9 0.65 0.27 31.2 29.3 35.3 36.2 ReCamMaster [6] 731.6 0.72 0.23 32.1 29.0 30.9 27.6

ReCamMaster* [6] 675.4 0.52 0.22 24.6 20.2 29.7 31.2 PlenopticDreamer (Ours) 425.8 0.54 0.21 41.4 40.8 45.4 41.2

(Frame77) (Frame93) (Frame93)

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Input Video& Cameras

[Figure 51]

[Figure 52]

RotationLeft ArcRight AzimuthLeft RotationRight TiltUp TranslationDown(Rot)

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Ours

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

ReCamMaster

ReCamMaster*

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

TrajectoryCrafter

- Figure 4. Qualitative Comparison on the Basic Benchmark. PlenopticDreamer generates high-fidelity visuals with consistent hallucinations from different camera trajectories. In contrast, ReCamMaster and TrajectoryCrafter fail to preserve spatio-temporal consistency while maintaining visual quality, especially under large-angle viewpoint changes, such as leftward azimuth shifts.

ond, and the eave above the robot in the third). TrajectoryCrafter and Trajectory-Attention leverage 3D point tracking to extract dynamic cues from the source video and feed them as conditional inputs to the generator. However, without updating the 3D memory using newly rendered content, they fail to maintain consistent cross-view synthesis. Moreover, the off-the-shelf checkpoints of these baselines exhibit low camera accuracy, especially in translation, due to poor performance on static novel-view synthesis under large-angle viewpoint changes (e.g., azimuth and elevation shifts). When retrained on the same datasets, ReCamMaster* achieves comparable camera accuracy. Notably, the original ReCamMaster does not employ Pl¨ucker raymaps; we integrate them to ensure a fair comparison.

- 4.3. Experiment on Agibot Benchmark Experiment Details.

- • Functionality: The model supports head-view to gripperview transformations in robotic manipulation.

- • Training Dataset: We use Agibot [10], a large-scale robotic dataset with about 1M episodes. We sample 145,820 episodes, each containing three synchronized video views (one head-view and two gripper-views) with precise camera pose annotations.

- • Training Details: The model context size k is set to 2, and it is trained for 15K steps with merely the first stage, requiring ∼5 days.

- • Benchmark: We build an Agibot benchmark using 200 test videos, covering head-to-hand and hand-to-hand camera transformations.

Qualitative and Quantitative Comparison. As illustrated in Fig. 5 and Tab. 2, PlenopticDreamer can perform headview→gripper-view and gripper-view→gripper-view trans-

###### Frames

[Figure 77]

[Figure 78]

[Figure 79]

Head View(Input)

[Figure 80]

[Figure 81]

[Figure 82]

Left-gripperView

Right-gripperView

The robotic arms push the door closed after retrieving the jeans

[Figure 83]

[Figure 84]

[Figure 85]

Head View(Input)

[Figure 86]

[Figure 87]

[Figure 88]

Left-gripperView

Right-gripperView

The robotic arms glide a garment steamer on a neatly hung shirt

[Figure 89]

[Figure 90]

[Figure 91]

Input(head-view)

Ours

ReCamMaster*

- Figure 5. Qualitative Results on the Agibot Benchmark. Given a head-view manipulation video in Agibot, PlenopticDreameragibot (Ours) can generate temporally consistent videos from the left and right gripper viewpoints.

formation in an autoregressive manner. Specifically, given a head-view manipulation video, it generates temporally consistent videos from both left and right gripper viewpoints across diverse manipulation tasks. In contrast, ReCamMaster* (also retrained on the Agibot dataset) fails to maintain view synchronization and high visual quality (see the blackboard eraser marked in the red dashed box).

- Table 2. Quantitative Comparison on the Agibot Benchmark. Visual quality and view synchronization are assessed on 2 shots (left and right gripper viewpoints).

PSNR ↑ View Sync. (Mat. Pix.(K) ↑) ReCamMaster* 13.84 13.2

Ours 14.54 15.3

#### 4.4. Ablation Study

Training Strategy. As shown in Fig. 6 and Tab. 3, removing the progressive training strategy (w/o Progressive

(Frame93)

[Figure 92]

[Figure 93]

Input Video& Cameras

RotationRight ArcLeft ZoomOut

[Figure 94]

[Figure 95]

[Figure 96]

Full Model

[Figure 97]

[Figure 98]

[Figure 99]

w/o Self-Cond.Training

[Figure 100]

[Figure 101]

[Figure 102]

w/ RandomContext Videos

w/o ProgressiveTraining

[Figure 103]

[Figure 104]

[Figure 105]

Figure 6. Ablation Study. Qualitative visualization of effects from different training strategies and context retrieval method.

Training) leads to a notable degradation in camera accuracy (0.54→0.63 in TransErr), and an occluded man becomes erroneously visible under the “Rotation Right” case. This indicates that progressively enlarging the context size effectively stabilizes model convergence and enhances camera performance. When the self-conditioned training is removed, the generated videos exhibit pronounced artifacts and over-exposure, particularly in long-shot sequences. Correspondingly, both FVD and IQ (Image Quality in VBench [31]) metrics worsen, verifying that training with imperfect inputs enhances model robustness and mitigates error accumulation over time.

Video Retrieval Strategy. Replacing the proposed retrieval mechanism with random selection leads to a significant decline in view synchronization across all shots, with inconsistent hallucinations (highlighted with red dashed boxes in Fig. 6). We further examine the impact of context size in Tab. 4: increasing the number of retrieved contexts from 4 to 6 enhances multi-view consistency by offering richer spatial cues. However, further enlarging the context brings diminishing gains due to compounded trajectory fusion errors and accumulated generative noise.

#### 4.5. Application

Long Video Generation. With the proposed long-video conditioning strategy, PlenopticDreamer supports coherent long-context video re-rendering, as shown in Fig. 7. Given a leftward rotation trajectory, our method produces temporally consistent long video segments while preserving spatial coherence across adjacent chunks. In contrast, removing this conditioning results in visible inconsistencies, as

- Table 3. Ablation Study on the Basic Benchmark. Evaluation is conducted on the full set, comprising a total of 1,200 generated videos.

Visual Quality Camera Accuracy View Synchronization (Mat. Pix.(K) ↑)

Model FVD ↓ IQ↑ TransErr ↓ RotErr (rad) ↓ 3 Shots 6 Shots 9 Shots 12 Shots w/o Self-Cond. Training 464.3 56.7 0.54 0.23 40.9 40.2 45.1 40.7

w/ Random Context Retrieval 520.5 58.3 0.56 0.20 33.6 33.4 36.5 32.4 w/o Progressive Training 453.8 57.2 0.63 0.23 39.6 40.6 43.6 39.4

###### Full Model 425.8 58.5 0.54 0.21 41.4 40.8 45.4 41.2

Frame1(Chunk1) Frame93(Chunk1) Frame94(Chunk2) Frame165(Chunk2) Frame166(Chunk3)

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Input Video

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

w/o LVG Cond.

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

w/ LVG Cond.

Figure 7. Long Video Generation. Given a leftward rotation camera trajectory, ours (w/ LVG Cond.) preserves spatial consistency across adjacent video chunks, yielding seamless transitions at their boundaries (highlighted by red dotted lines).

###### Table 4. Ablation on Retrieved Context Video Number.

View Synchronization (Mat. Pix.(K) ↑) Video Num. 3 Shots 6 Shots 9 Shots 12 Shots

4 58.1 51.2 52.1 42.7 6 52.1 53.1 43.6 8 50.2 41.0

10 40.8

illustrated in the second row.

Focal Length Effect. Varying the input focal length leads to corresponding depth-of-field changes, as shown in Fig. 8 under a “zoom-in” camera trajectory. This enables finer control over camera behavior and visual emphasis, offering users greater flexibility in camera-aware video generation.

### 5. Conclusion

We introduce PlenopticDreamer, a camera-controlled generative video re-rendering framework enforcing spatiotemporal consistency. It employs a multi-in-singleout, autoregressive diffusion model conditioned on spatiotemporal memory, retrieved via 3D FOV strategy for coherent hallucinations along trajectories. By incorporating progressive context-scaling and self-conditioning during training, the method improves stability and reduces error accumulation in long-range video generation. Evaluation on Basic and Agibot benchmarks shows state-of-the-art view syn-

###### Frames

[Figure 121]

[Figure 122]

[Figure 123]

Input Video

[Figure 124]

[Figure 125]

|[Figure 126]|
|---|

focal: 18mm

[Figure 127]

[Figure 128]

|[Figure 129]|
|---|

focal: 50mm

[Figure 130]

[Figure 131]

|[Figure 132]|
|---|

focal: 100mm

Figure 8. Focal Length Effect. Our method simulates varying depth-of-field effects corresponding to different focal lengths (18mm→100mm) under a “zoom-in” camera trajectory.

chronization, high fidelity, and precise camera control.

Limitations. Despite self-conditioned training, ours still exhibits occasional failures, including over-exposure and distortion in long-shot videos. Future work could explore a Self-Forcing–style [15, 30, 42] paradigm. We also observe degraded performance in complex human motions, such as dancing, likely from pretraining data biases in Cosmos.

### A. More Experimental Details

#### A.1. Implementation

Video Retrieval Algorithm. To construct the view frustum for a single camera pose, we fix the horizontal and vertical fields of view to 90◦ and 60◦, respectively, and set the near and far clipping planes to 0 and 10. For mesh sampling, we uniformly sample 8 points along the width and 6 points along the height on the plane.

Choice of Context Number k. A larger value for k allows for the retrieval of more contextual information and reduces the number of required inference iterations, but it also increases computational overhead. but comes at the cost of increased computational overhead. To balance context visibility and computation, we adopt video consistency as the selection criterion, as shown in Tab. R1, and choose k = 4 as the appropriate setting.

- Table R1. Ablation on In-context Video Number k. View Synchronization is evaluated on the Basic Benchmark.

N Shots/k videos 2 3 4 5 6

6 38.9 39.7 40.3 40.8 40.2 9 43.6 44.5 45.6 45.4 44.7

12 40.8 40.2 41.2 40.9 40.4

Long Video Generation. During training, we adopt 6 overlapped latent frames (corresponding to 21 decoded frames) and set the conditioning ratio to 0.45. Although the model is trained on 81-frame multi-view datasets, it generalizes effectively to long-form video generation. During inference, we produce a 93-frame initial chunk, followed by subsequent chunks of 71 frames.

Self-Conditioned Training. For the second training stage of the model evaluated on the Basic benchmark, we randomly sample 900 scenes from the MultiCamVideo dataset and 100 scenes from the SynCamVideo dataset. For each scene, we synthesize 1–5 videos, yielding roughly 3.5K training samples in total. In our current setup, the clean ground-truth video is used as the context for generating its noisy pseudo–GT counterpart. We did not observe clear performance gains when incorporating long-shot autoregressive generation into the synthetic video pipeline, as reported in Tab. R2. For each N-shot setting, we generate the same number of synthetic videos and post-train the model for 2K steps to ensure a fair comparison.

- Table R2. Ablation on N-Shots Synthetic Video Generation. Evaluation is conducted on the Basic Benchmark.

1 Shot 2 Shots 3 Shots 4 Shots FVD↓ 425.8 441.3 436.4 460.2 IQ↑ 58.5 57.6 57.2 56.5

InputVideo AzimuthLeft

[Figure 133]

[Figure 134]

Input Image

[Figure 135]

[Figure 136]

Matched Result

RotationRight ArcLeft(w/Rotation)

[Figure 137]

[Figure 138]

Input Image

[Figure 139]

[Figure 140]

Matched Result

Figure S1. Image Matching Result. The red points indicate the matched pixel correspondences across the input images..

#### A.2. Evaluation

FVD (Fr´echet Video Distance). We use StyleGAN-V [49] as the feature extractor backbone, sample 49 frames per video interval, resize each frame to 432 × 768, and include all frames for evaluation. For videos with substantial camera motion (low FOV overlap) relative to the input video, we select alternative video with similar camera trajectories for evaluation. Thus this metric can provide an indirect measure of video similarity.

##### TransErr (Camera Translation Error).

n

TransErr =

i=1

Tigt − Tipred 22 (11)

where Tigt and Tipred are the ground-truth and predicted translation vectors for the i-th frame. In our experiments,

we first align the translation scale of cameras estimated by ViPE [29] or VGGT [55] with the input cameras before computing TransErr.

##### RotErr (Camera Rotation Error).

RotErr =

tr RigtRiTpred − 1 2

n

arccos

i=1

(12)

where Rigt and Ripred are the ground-truth and predicted rotation matrices for the i-th frame, and tr(·) denotes the

matrix trace. We report this metric in radians.

##### Mat. Pix. (Matched Pixels in Video Synchronization).

K

Mat. Pix. =

1 Ci ≥ τ (13)

i=1

where K is the total number of pixels, Ci is the confidence score of the i-th pixel, τ is the confidence threshold, and 1(·) is the indicator function. Mat. Pix. counts pixels with confidence above the threshold. The qualitative matching results are presented in Fig. S1.

In our experiments, we set τ = 0.5, resize frames to 432×768, and average all frames. As illustrated in Fig. S2, the sequential 12-shot trajectory is: (1) Rotation Left → (2) Arc Right (w/ Rot.) → (3) Azimuth Right → (4) Rotation Right → (5) Arc Left (w/ Rot.) → (6) Azimuth Left → (7) Tilt Up → (8) Translate Down (w/ Rot.) → (9) Tilt Down → (10) Translate Up (w/ Rot.) → (11) Elevation Up → (12) Zoom Out. View synchronization is computed for the video pairs shown in Tab. R3.

- Table R3. Video Pairs for Multi-shot Video Synchronization Calculation on the Basic Benchmark.

N-Shots Calculated Video Pairs 3 Shots

(Rotation Left, Arc Right (w/ Rot.)) (Rotation Left, Azimuth Right)

(Rotation Left, Arc Right (w/ Rot.)) (Rotation Left, Azimuth Right) (Rotation Right, Arc Left (w/ Rot.)) (Rotation Right, Azimuth Left)

6 Shots

(Rotation Left, Arc Right (w/ Rot.)) (Rotation Left, Azimuth Right) (Rotation Right, Arc Left (w/ Rot.)) (Rotation Right, Azimuth Left) (Tilt Up, Translate Down (w/ Rot.)) (Tilt Down, Translate Up (w/ Rot.))

9 Shots

(Rotation Left, Arc Right (w/ Rot.)) (Rotation Left, Azimuth Right) (Rotation Right, Arc Left (w/ Rot.)) (Rotation Right, Azimuth Left) (Tilt Up, Translate Down (w/ Rot.)) (Tilt Down, Translate Up (w/ Rot.)) (Translate Up (w/ Rot.), Elevation Up) (Translate Up (w/ Rot.), Zoom Out)

12 Shots

### B. More Qualitative Results

We present additional qualitative results on the Basic Benchmark in Fig. S3, along with comparative visualizations in Fig. S4. Further results on the Agibot Benchmark are shown in Fig. S5, with corresponding comparisons in Fig. S6. We also include extended long video generation results in Fig. S7 and illustrate the focal-length effect in Fig. S8.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 4

- [2] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 5
- [3] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025. 1, 5
- [4] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In CVPR, 2025. 1, 2
- [5] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. Vd3d: Taming large video diffusion transformers for 3d camera control. In ICLR, 2025. 2
- [6] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. In ICCV, 2025. 2, 3, 4, 5, 6
- [7] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. In ICLR, 2025. 2, 5
- [8] James R Bergen and Edward H Adelson. The plenoptic function and the elements of early vision. Computational models of visual processing, 1991. 1
- [9] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024. 1
- [10] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In IROS, 2025. 2, 6
- [11] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al. Mixture of contexts for long video generation. arXiv preprint arXiv:2508.21058, 2025. 2
- [12] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024. 4
- [13] Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, et al. Deepverse: 4d autoregres-

- sive video generation as a world model. arXiv preprint arXiv:2506.01103, 2025. 2
- [14] Taiye Chen, Xun Hu, Zihan Ding, and Chi Jin. Learning world models for interactive video generation. arXiv preprint arXiv:2505.21996, 2025. 2
- [15] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 8
- [16] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In CVPR, 2025. 2
- [17] Johan Edstedt, Qiyu Sun, Georg B¨okman, M˚arten Wadenb¨ack, and Michael Felsberg. Roma: Robust dense feature matching. In CVPR, 2024. 5
- [18] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 3
- [19] Xiao Fu, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for multientity motion in video generation. In ICLR, 2025. 2
- [20] Xiao Fu, Xintao Wang, Xian Liu, Jianhong Bai, Runsen Xu, Pengfei Wan, Di Zhang, and Dahua Lin. Learning video generation for robotic manipulation with collaborative trajectory control. arXiv preprint arXiv:2506.01943, 2025. 1
- [21] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 1

- [22] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, et al. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. In SIGGRAPH, 2025. 2
- [23] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR,

2024. 1

- [24] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for video diffusion models. In ICLR, 2025. 1, 2, 5
- [25] Xuanhua He, Quande Liu, Zixuan Ye, Weicai Ye, Qiulin Wang, Xintao Wang, Qifeng Chen, Pengfei Wan, Di Zhang, and Kun Gai. Fulldit2: Efficient in-context conditioning for video diffusion transformers. arXiv preprint arXiv:2506.04213, 2025. 3
- [26] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In ICLR, 2023. 1
- [27] Chen Hou and Zhibo Chen. Training-free camera control for video generation. arXiv preprint arXiv:2406.10126, 2024. 2

- [28] Teng Hu, Jiangning Zhang, Ran Yi, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, and Lizhuang Ma. Motionmaster: Training-free camera motion transfer for video generation. arXiv preprint arXiv:2404.15789, 2024. 2
- [29] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025. 5, 9
- [30] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. In NeurIPS, 2025. 4, 8
- [31] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 7
- [32] Xuan Ju, Weicai Ye, Quande Liu, Qiulin Wang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Qiang Xu. Fulldit: Multi-task video generative foundation model with full attention. In ICCV, 2025. 3
- [33] Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Martin Grayson, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao, et al. World and human action models towards gameplay ideation. Nature, 638(8051):656–663, 2025. 2
- [34] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1
- [35] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas J Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with camera control. In NeurIPS, 2024. 2
- [36] Bing Li, Cheng Zheng, Wenxuan Zhu, Jinjie Mai, Biao Zhang, Peter Wonka, and Bernard Ghanem. Vivid-zoo: Multi-view video generation with diffusion model. In NeurIPS, 2024. 2
- [37] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201, 2025. 2
- [38] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. In ICCV, 2025. 2
- [39] Zhengqi Li, Wenqi Xian, Abe Davis, and Noah Snavely. Crowdsampling the plenoptic function. In ECCV, 2020. 1
- [40] Pengyang Ling, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, Huaian Chen, Jiaqi Wang, and Yi Jin. Motionclone: Training-free motion cloning for controllable video generation. arXiv preprint arXiv:2406.05338, 2024. 2
- [41] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 3
- [42] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 8

- [43] Zhiheng Liu, Xueqing Deng, Shoufa Chen, Angtian Wang, Qiushan Guo, Mingfei Han, Zeyue Xue, Mengzhao Chen, Ping Luo, and Linjie Yang. Worldweaver: Generating longhorizon video worlds via rich perception. arXiv preprint arXiv:2508.15720, 2025. 2
- [44] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744, 2025. 2
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 1, 3
- [46] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In CVPR, 2025. 1, 2
- [47] Manuel-Andreas Schneider, Lukas H¨ollein, and Matthias Nießner. Worldexplorer: Towards generating fully navigable 3d scenes. arXiv preprint arXiv:2506.01799, 2025. 2
- [48] Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering.

2021. 4

- [49] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In CVPR, 2022. 9
- [50] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 2
- [51] Google Deepmind Veo Team. Veo 3. 2025. 1
- [52] Kling Team. Kling ai 2.5 turbo. 2025. 1
- [53] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In ECCV,

2024. 2

- [54] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1
- [55] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 5, 9
- [56] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 4
- [57] Zhouxia Wang, Yushi Lan, Shangchen Zhou, and Chen Change Loy. Objctrl-2.5 d: Training-free object control with camera poses. arXiv preprint arXiv:2412.07721,

2024. 2

- [58] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In SIGGRAPH, 2024. 1, 2

- [59] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. In NeurIPS, 2025. 2
- [60] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Longterm consistent world simulation with memory. In NeurIPS,

2025. 2, 4

- [61] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for fine-grained video motion control. In ICLR, 2025. 2, 5, 6
- [62] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. In ICLR, 2025. 2
- [63] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 2
- [64] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In SIGGRAPH, 2024. 2
- [65] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In SIGGRAPH Asia, 2025. 2, 3, 4
- [66] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025. 2, 5, 6
- [67] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025. 2
- [68] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. arXiv preprint arXiv:2410.15957, 2024. 2

###### InputCameras Frames

[Figure 141]

Input Video

[Figure 142]

Prompt: Photorealistic view of a busy urban intersection captured from the perspective of a driver traveling down a wide city street

[Figure 143]

Rotation Left

[Figure 144]

[Figure 145]

Arc Right(w/ Rotation)

[Figure 146]

Azimuth Right

[Figure 147]

[Figure 148]

[Figure 149]

Rotation Right

[Figure 150]

[Figure 151]

Arc Left(w/ Rotation)

[Figure 152]

[Figure 153]

Azimuth Left

[Figure 154]

[Figure 155]

[Figure 156]

Tilt Up

[Figure 157]

Translate Down(w/ Rotation)

[Figure 158]

[Figure 159]

[Figure 160]

Tilt Down

[Figure 161]

[Figure 162]

Translate Up(w/ Rotation)

[Figure 163]

Elevation Up

[Figure 164]

[Figure 165]

[Figure 166]

Zoom Out

- Figure S2. Full Camera Trajectories on the Basic Benchmark. The sequence proceeds as: (1) Rotation Left→(2) Arc Right (w/ Rot.)→(3) Azimuth Right→(4) Rotation Right→(5) Arc Left (w/ Rot.)→(6) Azimuth Left→(7) Tilt Up→(8) Translate Down (w/ Rot.)→(9) Tilt Down→(10) Translate Up (w/ Rot.)→(11) Elevation Up→(12) Zoom Out.

[Figure 167]

Frames

[Figure 168]

Input Video

Prompt:Amaninplaidsuitsmokes,gestures,checkswatchbesidepolicecarinstaticscene.

Rotation Right

[Figure 169]

[Figure 170]

Arc Left

Azimuth Left

[Figure 171]

[Figure 172]

Frames

[Figure 173]

Input Video

Prompt:Amonkeyinaredcapandbluevestsitsonabench,contemplatingachessmove.

[Figure 174]

Tilt Down

Translate Up

[Figure 175]

[Figure 176]

Elevation Up

[Figure 177]

Frames

Input Video

[Figure 178]

Prompt:AphotographerwithavintagecamerastandsbyaneoclassicalcolumnandancientstatueinPlaka’sgoldenlight.

Rotation Left

[Figure 179]

[Figure 180]

Arc Right

[Figure 181]

Zoom Out

- Figure S3. More Visual Results on the Basic Benchmark. Our method generates consistent hallucinated context in unseen region.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

###### Figure S4. More Qualitative Comparison on the Basic Benchmark. The figures above and below correspond to frames 54 and 88, respectively. Please check full videos on the website provided in the website.

###### Frames

[Figure 189]

Head View(Input Video)

[Figure 190]

Left-gripperView

Right-gripperView

Prompt: The robot arm pours water from green kettle into oatmeal, with fresh fruits and pastries nearby.

[Figure 191]

Head View(Input Video)

[Figure 192]

Left-gripperView

Right-gripperView

Prompt: The robot arm scoops peas with a spoon into a bowl while stabilizing it, then returns the spoon.

[Figure 193]

Head View(Input Video)

[Figure 194]

Left-gripperView

Right-gripperView

Prompt: One robot arm holds soy sauce bottle while the other unscrews the cap and sets aside.

[Figure 195]

Head View(Input Video)

[Figure 196]

Left-gripperView

Right-gripperView

Prompt: The robot arms manipulate plush toy octopus, tilting it onto a patterned blanket for stability.

- Figure S5. More Visual Results on the Agibot Benchmark. The sequence proceeds as: (1) Left-gripper View→(2) Right-gripper View.

[Figure 197]

[Figure 198]

###### Figure S6. More Qualitative Comparison on the Agibot Benchmark. The figures above and below correspond to frames 24 and 93, respectively. Compared to our method, ReCamMaster* exhibits noticeably stronger object distortion and inconsistency.

Frames

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Input Video

(Frame1) (Frame309)

Prompt:Amanandwomansitonasunnyparkbench,readingseparately.

CameraTrajectory:RotationRight(Dynamic)

[Figure 204]

- Chunk 1
- Chunk 2
- Chunk 3
- Chunk 4

[Figure 205]

[Figure 206]

[Figure 207]

Frames

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Input Video

(Frame1) (Frame453)

Prompt:Ayoungcouplestandsatasteamshipbowatsunset

CameraTrajectory:DistanceAwary(Static)

[Figure 217]

- Chunk 1
- Chunk 2
- Chunk 3
- Chunk 4
- Chunk 5
- Chunk 6

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

###### Figure S7. More Long Video Generation Results under Dynamic and Static Novel-Camera Settings.

Frames

[Figure 223]

Input Video

Prompt:Amanreadshandwrittenpaperbyruralstonewall,seatedcalmlywithhatandjacket.

CameraTrajectory:ZoomIn

[Figure 224]

18mm

[Figure 225]

24mm

[Figure 226]

50mm

[Figure 227]

100mm

CameraTrajectory:ZoomOut

[Figure 228]

18mm

[Figure 229]

24mm

[Figure 230]

50mm

[Figure 231]

100mm

Figure S8. More Focal Length Effect Results. Our method synthesizes depth-of-field variations across focal lengths (18mm→100mm). Shorter focal lengths produce more greater changes in the resulting field-of-view (FOV).

