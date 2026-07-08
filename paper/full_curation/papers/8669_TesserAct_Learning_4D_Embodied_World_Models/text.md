[Figure 1]

## TesserAct: Learning 4D Embodied World Models

Haoyu Zhen1 ∗ Qiao Sun1 ∗ Hongxin Zhang1 Junyan Li1 Siyuan Zhou2 Yilun Du3 Chuang Gan1

1UMass Amherst 2HKUST 3Harvard University

https://TesserActWorld.github.io

# arXiv:2504.20995v1[cs.CV]29Apr2025

[Figure 2]

[Figure 3]

Unseen scene, unseen object

Input Instruction: Pick up the brown cup – Franka Panda Robot RGB-DN Video Generation 4D Scene Reconstruction & Explicit Actions

Put shoes in box – Franka Panda Robot In Domain: RLBench (Sim)

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Point Cloud

Reach the target

[Figure 12]

[Figure 13]

1. Open Box

Lift

RGB

Input Image

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Depth

2. Put Shoes

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Normal Mesh

[Figure 26]

[Figure 27]

Cross Domain

Move coke can near bottle - Google Robot In Domain: Fractal Data (Real)

Input Instruction: Pick up the apple from white bowl – Trossen WidowX 250 RGB-DN Video Generation 4D Scene Reconstruction & Explicit Actions

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Point Cloud

[Figure 35]

[Figure 36]

RGB Grasp

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Depth

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Normal

Point Cloud

Input Image

Figure 1. We propose TesserAct, the 4D Embodied World Model, which takes an input image and text instruction to generate RGB, depth, and normal videos, reconstructing a 4D scene and predicting actions. Our model not only achieves strong performance on in-domain data (right) but also generalizes effectively to unseen scenes, novel objects (top left), and cross-domain scenarios (bottom left).

### Abstract

we first extend existing robotic manipulation video datasets with depth and normal information leveraging off-the-shelf models. Next, we fine-tune a video generation model on this annotated dataset, which jointly predicts RGB-DN (RGB, Depth, and Normal) for each frame. We then present an algorithm to directly convert generated RGB, Depth, and Normal videos into a high-quality 4D scene of the world. Our method ensures temporal and spatial coherence in 4D scene predictions from embodied scenarios, enables novel view synthesis for embodied environments, and facilitates policy learning that significantly outperforms those derived from prior video-based world models.

This paper presents an effective approach for learning novel 4D embodied world models, which predict the dynamic evolution of 3D scenes over time in response to an embodied agent’s actions, providing both spatial and temporal consistency. We propose to learn a 4D world model by training on RGB-DN (RGB, Depth, and Normal) videos. This not only surpasses traditional 2D models by incorporating detailed shape, configuration, and temporal changes into their predictions, but also allows us to effectively learn accurate inverse dynamic models for an embodied agent. Specifically,

∗Equal contribution.

### 1. Introduction

Learned world models [21, 64, 67, 77], which simulate environmental dynamics, play a crucial role in enabling embodied intelligent agents. Such models enable flexible policy synthesis [15, 40], data simulation and generation [67, 80], and long-horizon planning [14, 28, 74]. However, while the physical world is three-dimensional in nature, existing world models operate in the space of 2D pixels. This limitation leads to an incomplete representation of spatial relationships, impeding tasks that require precise depth and pose information. For instance, without accurate depth and 6-DoF pose estimations, robotic systems struggle to determine the exact position and orientation of objects. Furthermore, existing 2D models can produce unrealistic results, such as inconsistent object sizes and shapes across time steps, which limits their use in data-driven simulations and robust policy learning.

In this paper, we explore how we can instead learn a 4D embodied world model, TesserAct, which simulates the dynamics of a 3D world. This 4D embodied world model allows us to generate realistic 3D interactions, such as grasping objects or opening drawers, with a level of detail that traditional 2D-based models cannot achieve. By modeling spatial and temporal dimensions, our model provides the depth and pose information essential for robotic manipulation. However, the task of learning a 4D embodied world model is challenging as the dynamics of the world are extremely computationally expensive to train and learn, requiring models to generate outputs in three-dimensional space and time. To efficiently represent and predict the dynamics of the world, we propose a substantially more lightweight representation of the 4D world, consisting of predicting a sequence of RGB, depth, and normal maps of the scene. This combined representation accurately captures the appearance, geometry, and surface of a scene while being substantially lower dimensional than explicitly predicting world dynamics. Furthermore, such a representation shares substantial similarities to existing video models, allowing us to directly use the generative capabilities of existing video models to effectively construct our 4D world model.

Given this intermediate representation, we present an efficient algorithm to reconstruct accurate 4D scenes from generated RGB-DN videos. For each frame, we use a combination of both depth and normal prediction to integrate a smooth 3D surface of the scene. We then use optical flow between generated frames to distinguish between background and dynamic regions in the reconstructed 3D scene across frames and introduce two novel loss functions to enforce consistency across scenes over time. As shown in Figure 1, our 4D Embodied World Model enables the construction of highfidelity 4D-generated scenes, facilitating strong-performance action planning for downstream tasks.

A key challenge for training TesserAct is a lack of access to existing large-scale datasets with high-quality 4D

annotations, or the RGB image, depth, and normal information needed to train our approach. To overcome this, we collect a 4D embodied video dataset consisting of synthetic data with ground truth depth and normal information and real-world data with estimated depth and normal information with off-the-shelf estimators.

Overall, our paper has the following contributions:

- • We collect a 4D embodied video dataset with compact and high-quality RGB, depth, and normal annotations and learn a 4D embodied world model.
- • We present an algorithm to convert the generated RGB-DN video into high-quality 4D scenes coherent across both time and space.
- • Extensive experiments demonstrate our 4D embodied world model can predict high-fidelity 4D scenes and achieve superior performance in downstream embodied tasks compared to traditional video-based world models.

### 2. Related Work

Embodied Foundation Models A flurry of recent work has focused on constructing foundation models for general purpose agents [18, 68]. One line of work has focused on constructing multimodal language models that operate over images [13, 20, 29, 39, 51, 63, 73] as well as 3D inputs [24, 25] and output text describing the actions of an agent. Other works have focused on the construction of vision-languageaction (VLA) models that directly output action tokens [7, 34, 76]. Both of the previous approaches aim to construct foundation model policies (over text or continuous actions). In contrast, our work aims to instead construct a foundation 4D world model for embodied agents, which can then be used for downstream applications such as planning [14, 74] or policy synthesis [15, 40].

Learning World Models Learning dynamics model of the world given control inputs has been a long-standing challenge in system identification [42], model-based reinforcement learning [56], and optimal control [4, 81]. A large body of work focused on learning world models in the low dimensional state space [1, 17, 38], which while being efficient to learn, is difficult to generalize across many environments. Other works have explored how world models may be learned over pixel-space images [11, 12, 21, 22, 43], but such models are trained on simple game environments. With advances in generative modeling, a large flurry of recent research has focused on using video models as foundation world models [8, 46, 64, 67, 77, 79] but such models operate over the space of 2D pixels which does not fully simulate the 3D world. Most similar to our work are [76], which predicts only the 3D goal state for robotic tasks, and [57], an geometric-aware world model trained purely on synthetic data without language grounding or downstream robotic tasks; in contrast, our approach models the 4D scene from RGB-DN videos and supports language-conditioned control.

Dataset Domain Depth Source Normal Source Embodiment # of videos RLBench [26] Synthetic Simulator Depth2Normal [2] Franka Panda 80k RT1 Fractal Data [6]

Google Robot 80k Bridge [61] WidowX 25k SomethingSomethingV2 [19] Human Hand 100k

Real Rolling Depth [31] Marigold [32]

Table 1. Overview of the 4D embodied video datasets.

4D Video Generation The task of 4D video generation has gained increasing attention in recent years [55, 71], driven by advancements in diffusion models [23, 47, 54], neural radiance fields [44], and 3D Gaussian splatting [33]. However, existing methods often suffer from slow optimization due to hybrid frameworks [3, 41, 65, 75] and the convergence challenges of SDS loss [30, 53]. Instead, we represent 4D scenes using RGB-DN videos, which offer more efficient training and provide high-accuracy 3D information crucial for embodied tasks. Furthermore, our approach is the first to directly predict 4D scenes from the current frame and the embodied agent’s action described in the text.

### 3. Preliminaries

#### 3.1. Latent Video Diffusion Models

Diffusion models [23, 54] are capable of learning the data distribution p(x) by progressively adding noise to the data until it resembles a Gaussian distribution through a forward process. During inference, a denoiser ϵ is trained to recover the data from this noisy state. Latent video diffusion models [77] utilize a Variational Autoencoder (VAE) [35, 60] to encode the data in the latent space, maintaining high-quality outputs while more efficiently modeling the data distribution.

We formulate the task of RGB V, depth D, and normal N video generation as a conditional denoising generation task, i.e., we model the distribution p(v,d,n|v0,d0,n0,T ), where v,d,n represent the predicted future latent sequences of RGB, depth, and normal maps, respectively; the conditions v0,d0,n0,T are the latent of RGB image, depth and normal maps, and embodied agent’s action in text.

The forward diffusion process adds Gaussian noise to the latent z ∈ {v,d,n} over T timesteps, defined as:

q(zt|zt−1) = N (zt;√αtzt−1,(1 − αt)I) (1)

where t ∈ {1,2,...,T} denotes the diffusion step, αt is a parameter controlling the noise influence at each step, and I is the identity matrix. In the reverse process, the model aims to recover the original latent from the noise. Let x = [v,n,d] denoting the concatenation of v,n,d, a denoising network ϵθ(xt,t,x0,T ) with learning parameters θ is trained to predict the noise added at each timestep. The

reverse process is defined as:

pθ(xt−1|xt,x0,T ) = N xt−1;µθ(xt,t,x0,T ),Σθ(xt,t) (2)

Once the denoised latent x0 is obtained, the model maps it back to the pixel space to obtain the final RGB-DN video.

#### 3.2. Depth Optimization via Normal Integration

As discussed in [9, 66], normal maps provide essential information about surface orientation, which is vital for enforcing geometric constraints and imposing surface smoothness and continuity during depth optimization. This spatial optimization leads to more accurate and reliable depth estimates that closely align with the true 3D geometry and capture fine surface details.

To formalize the process, we use the perspective camera model to set constraints on the depth and surface normal. In the coordinate system of the 2D image at frame i, a pixel position is given as u = (u,v)T ∈ Vi, and its corresponding depth scalar, normal vector is d ∈ Di,n = (nx,ny,nz) ∈ Ni. Under the assumption of a perspective camera with focal length f and the principal point (cu,cv)T, as proposed by [16], the log-depth d˜= log(d) should satisfy the following equations: n˜z∂ud˜+ nx = 0 and n˜z∂vd˜+ ny = 0 where n˜z = nx(u − cu) + ny(v − cv) + nzf. In addition, we can add the assumption that all locations are smooth surfaces [9]. We can convert the above constraint to the quadratic loss function, allowing us to find the optimized depth map:

(˜nz∂ud˜+ nx)2 + (˜nz∂ud˜+ ny)2dudv. (3)

min

d Ω

Following [9], we can convert the above objective to an iteratively optimized loss objective. At iteration step t, we can compute the matrix W(d˜t) and iteratively optimize for a refined depth prediction d˜t+1:

d˜t+1 = arg mind˜(Ad˜− b)TW(d˜t)(Ad˜− b) def= arg minD˜ Ls(D˜,Ni)

where A and b are defined by normals and camera intrinsic.

### 4. Learning a 4D Embodied World Model

Learning how 3D scenes may change over time based on the current observation and action is crucial for embodied agents.

2

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

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Training Objective

Depth

User Instruction 𝑇

Projector

|Predicted noise 𝜖𝑣,𝜖𝑑, 𝜖𝑛|
|---|

Ƹ Ƹ Ƹ

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

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

RGB

RGB Out Projector

Add

DiT

Projector

[Figure 87]

[Figure 88]

DN Out

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Projector

Normal Projector

[Figure 94]

[Figure 95]

[Figure 96]

CogVideoX Backbone Zero Initialization

[Figure 97]

[Figure 98]

concat Conv3D

Figure 2. Architecture and Training Overview of TesserAct.

We propose to learn a 4D embodied world model, TesserAct, by training on RGB-DN videos and reconstructing the 4D scenes from it. We first introduce the 4D embodied video dataset we collected in Sec. 4.1, then discuss the model architecture and training strategy in Sec. 4.2. In Sec. 4.3, we propose an efficient optimization algorithm with two novel loss functions to convert the generated RGB-DN videos into 4D scenes. Finally, in Sec. 4.4, we demonstrate how the 4D world model can help downstream embodied tasks.

state-of-the-art video depth estimator RollingDepth [31] to annotate the videos with affine-invariant depth. As affineinvariant depth map does not directly yield normal map as metric depth does and a reliable video normal estimator is currently unavailable, we annotate normal maps using Temporal-Consistent Marigold-LCM-normal1. These two approaches allow us to obtain high-quality, sharp, and temporally consistent video depth and normal annotations. Specifically, we select two high-quality datasets from OpenX [58], the Fractal data [6], and the Bridge [61] dataset. Moreover, to further increase the diversity of the instructions, we incorporated the human-object interaction dataset, Something Something V2 [19]. Detailed statistics are shown in Table 1.

#### 4.1. 4D Embodied Video Dataset

Learning 4D embodied world models requires large-scale 4D datasets, which are expensive to collect in the real world. In this section, we present a data collection and annotation pipeline that enables us to automatically construct 4D datasets from existing video datasets.

#### 4.2. Model Architecture and Training Strategy

Training a diffusion model to generate temporal RGB-DN data is challenging. To effectively train RGB video models, large-scale video datasets containing billions of high-quality samples are typically employed [69, 77]. In contrast, our RGB-DN dataset, even with automatic annotation, comprises only about 200k data points, which is insufficient to train a world model from scratch. To overcome this limitation, we modify and fine-tune CogVideoX [69] to serve as our RGBDN prediction model, leveraging the pre-trained knowledge within it to effectively bootstrap our 4D model.

Simulator-synthesized data provide ground truth depth information, so we first selected 20 tasks of relatively high difficulty from RLBench [26] and generated 1000 instances captured from 4 different views for each task, making 80k synthetic 4D embodied videos in total. Although the simulator provides metric depth information, it lacks surface normal data. To estimate normals, we use the depth2normal function from DSINE [2]. To enhance the generalization, we adopt the scene randomization techniques from the Colosseum data generation pipeline [48], which alters the background, table texture and light of the scene.

Our architecture is illustrated in Figure 2. First, we utilize the 3D VAE [35, 60] from CogVideoX to separately encode the RGB (v), depth (d), and normal (n) videos, without any additional fine-tuning of the VAE. These latent

While synthetic data provides depth and normal data of high quality, their diversity is limited, resulting in a gap compared to real-world scenarios. To bridge this gap, we also incorporate real-world video datasets. As most of these datasets lack depth and normal annotations, we employ the

https : / / huggingface . co / docs / diffusers / en / using- diffusers/marigold_usage#frame- by- framevideo-processing-with-temporal-consistency

w/ consistency loss

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

wo/ consistency loss

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

w/

wo/

regularization

regularization

- Figure 3. Effect of consistency and regularization loss on 4D scene reconstruction. The red boxes highlight the inconsistent regions.

#### 4.3. 4D Scene Reconstruction

representations are perturbed with noise to get xt, and are then fed into our model along with the corresponding image latent head x0. For the input design, we introduce three separate projectors for each modality to extract the embeddings: fz = InputProj(zt,z0), where z ∈ {v,d,n}. DiT then takes the sum of these embeddings as input, conditioned on the textual input T and denoising step t, to obtain the hidden state: h = DiT( fz,t,T ). To distinguish between different robot arms, T is defined as [action instruction] + [robot arm name], e.g., “pick up apple google robot”. On the output side, we retain the original RGB output method: ϵ∗v = OutputProj(h). However, we introduce additional modules for depth and normal prediction. A Conv3D layer is used to encode the concatenation of the input latents and the predicted RGB denoised output. These are then combined with the hidden states produced by the DiT backbone and passed through the output projector to obtain the denoised predictions for depth and normal: ϵ∗d,n = DNProj(h,Conv3D(ϵ∗v,[zt;z0]z∈{v,d,n})). To preserve the pretrained knowledge, we initialize our model with the CogVideoX weights. All other modules are initialized with zeros, ensuring that the RGB output at the beginning of training matches the output of CogVideoX. During training, we randomly select samples from the 4D embodied dataset (V,D,N,T ) constructed above and apply Eq.1 to add noise ϵv, ϵd, and ϵn to the RGB-DN data at timestep t, minimizing the following objective:

After obtaining the RGB-DN video, we further optimize the depth and reconstruct the 4D scene. Similar to prior works [32, 70], our depth representation for each image is given by a relative map in the range [0,1], and thus cannot directly reconstruct the entire scene. While past work has sidestepped this by assuming either a default scale for depth [72] or by directly predicting metric depth [10, 76], such reconstructions from depth are often coarse and often cause reconstructed planes or walls to be tilted.

With the normal maps Ni, we can optimize the depth maps Di via normal integration for refined depth maps Dˆi as introduced in Sec. 3.2 with a loss term Ls for spatial consistency. However, this approach optimizes depth frame by frame, which lacks temporal consistency across the dynamic scene. To address this, we compute optical flow between frames [59] F = RAFT(V) and enforce consistency of depth across frames. We define the static regions of each frame as the pixels with the magnitude of optical flow smaller than threshold c and obtain its mask by Mis = ∥Fi∥ ≤ c. We then define the dynamic parts of an image as Mid = ¬Mis. We further define the background of an image as static regions that are fixed across image frames, Mib = Mis ∩ Mis−1

Since optical flow represents the movement of objects in the 2D-pixel space, we can retrieve the depth at any position from the previous frame to impose consistency constraints. To compute the depth values from the previous frame at positions corresponding to the current frame, we utilize the optical flow Fi→(i−1). For each pixel (u,v) in frame i, the optical flow provides the displacement (∆u,∆v), allowing us to find the corresponding pixel in frame i − 1 at position (u − ∆u,v − ∆v). Based on this mapping, we define the

2

0,T ,t,ϵ [ϵv,ϵd,ϵn] − ϵθ(xt,t,x0,T )

(4)

L = Ev

RGB Depth Normal Point Cloud

Domain Method

FVD ↓ SSIM ↑ PSNR ↑ AbsRel ↓ δ1 ↑ δ2 ↑ Mean ↓ Median ↓ 11.25◦ ↑ Chamfer L1 ↓

4D Point-E - - - - - - - - - 0.2211 OpenSora 23.67 71.31 19.25 31.41 60.39 79.97 41.82 32.15 13.58 0.3013 CogVideoX 20.64 79.38 22.39 26.17 64.82 81.62 19.53 10.09 22.70 0.2191 TesserAct (Ours) 21.59 75.86 20.27 22.07 66.80 82.60 15.74 7.32 27.80 0.2030

Real

4D Point-E - - - - - - - - - 0.1086 OpenSora 54.11 65.90 19.28 18.40 65.02 91.20 12.94 7.58 25.02 0.2570 CogVideoX 41.23 76.60 20.87 19.81 60.07 80.16 20.36 10.47 26.04 0.2884 TesserAct (Ours) 40.01 77.59 19.73 16.02 69.26 93.03 14.75 6.34 36.85 0.0811

Synthetic

- Table 2. Main results on the 4D scene generation. All metrics are averaged over 10 runs for each of the samples. The best results are in bold, and the second best are in underlined. TesserAct predicts the depth and normal maps most accurately without hurting RGB much and thus achieves the best accuracy of reconstructed 4D point clouds across real and synthetic image domains.

Di→(i−1) such that: Di→(i−1)(u,v) = Di−1(u − ∆u,v − ∆v). We then introduce the consistency loss over the dynamic and background region of the image Lc:

2

Lc(D˜,Dˆi−1,Fi,Fi−1) =λcd D ˜i ◦ Mid − Di→(i−1) ◦ Mid

+

2 (5)

λcb D ˜i ◦ Mib − Di→(i−1) ◦ Mib

In addition to the consistency loss, we also incorporate the regularization loss Lr, enforcing that optimized depths are similar to the generated depth map Di:

(6) The overall loss objective we optimize is given by:

2

2

Lr(D˜,Di) = λrd D ˜i ◦ Mid − Di ◦ Mid

+ λrb D ˜i ◦ Mib − Di ◦ Mib

arg minD˜ Ls(D˜,Ni) + Lc(D˜,Dˆi−1,Fi,Fi−1) + Lr(D˜,Di) (7) Starting from the first frame, we iteratively refine the

depth map by optimizing the loss above, and initialize the depth map at frame i with the generated depth map D˜0 = Di. With refined depth maps D˜ and RGB Images V, we can reconstruct 4D point clouds representing the world that are consistent over both space and time.

#### 4.4. Embodied Action Planning with 4D Scenes

After generating 4D scenes, which encapsulate both spatial and temporal information, we extract geometric details that can significantly enhance downstream tasks in robotics. The detailed geometry captured by these scenes plays a crucial role in tasks including robotic grasping.

To achieve this, we employ an inverse dynamics model built on the 4D point clouds, predicting the appropriate robot action ai based on the current state si, the predicted future state si+1 and the instruction T . Mathematically, this relationship is expressed as ai = ID(si,si+1,T ), where si represents the scene at the time step i. Specifically, we use PointNet [49] to encode the point cloud and extract 3D features. These features are then combined with the instruction text embeddings and further processed by an MLP to generate the final 7-DoF action.

### 5. Experiments

We first evaluate the quality of 4D scene predictions from our model across real and synthetic datasets in Sec. 5.1, then conduct experiments in RLBench to demonstrate how the 4D information helps the embodied tasks in Sec. 5.2. We provide more qualitative results and videos in the Supplementary and the website.

#### 5.1. 4D Scene Prediction

##### 5.1.1. Setup

Datasets. We conduct experiments on the real domain with 400 unseen samples in RT1 Fractal [6] and Bridage [61] dataset where depth and normal are estimated as in Sec. 4.1, and the synthetic domain with 200 unseen samples in RLBench [26], where ground truth depth and normal maps are directly accessible.

Metrics. We evaluate the video quality with FVD, SSIM, and PSNR; depth quality with AbsRel, δ1, and δ2; normal maps quality with Mean, Median, and 11.25◦; reconstructed point cloud quality with the L1 Chamfer Distance. We generate 10 times per sample and report the average.

Baselines. We compare our method to two video diffusion models and a 4D point cloud diffusion model.

- • OpenSora [77], video diffusion model fine-tuned with LoRA on the same dataset without depth and normal annotations. To construct the full 4D scene, depth and normal are additional estimated given the predicted video with Rolling Depth and Marigold.
- • CogVideoX [69], video diffusion model fine-tuned with LoRA on the same dataset without depth and normal annotations. The 4D scene is obtained similarly to the above.
- • 4D Point-E, since no prior work directly generates dynamic scenes from the first frame and text inputs, we implemented a 4D point cloud diffusion model as the baseline, where we modify the Point-E [45] model by conditioning it on the mean of CLIP [50] features extracted from both text and image inputs, outputting T point clouds

close box

open drawer

open jar

open microwave

put knife

sweep to dustpan

lid off

weighing off

water plants

Methods

Image-BC 53 4 0 5 0 0 12 21 0 UniPi∗ 81 67 38 72 66 49 70 68 35 4DWM (Ours) 88 80 44 70 70 56 73 62 41

- Table 3. TesserAct boosts the performance of action planning. We report the success rate averaged over 100 episodes for each task here.

CLIP Score

CLIP Aesthetic

Time Costs

Method PSNR SSIM LPIPS

SoM[62] 10.94 24.02 73.82 66.67 3.61 ∼2 hours Ours 12.99 42.62 60.51 83.02 3.73 ∼ 1 mins

Table 4. Novel view synthesis results on RLBench.

of size n, where T is set to 4 and n is set to 8192 due to computational constraints.

Implementation Details. We train our model on the collected 4D embodied video dataset using a multi-resolution training approach, predicting 49 frames at a time. For more details, please kindly refer to the Supplementary Materials.

##### 5.1.2. Results and Analysis

TesserAct predicts high quality 4D scenes. As shown in Table 2, our model accurately predicts the depth and normal maps compared to video diffusion models with postestimation, verifying the effectiveness of our learned model. With better depth and normal maps, the 4D point clouds reconstructed with our method achieve the lowest Chamfer distances across real and synthetic datasets. The 4D Point-E method performs better than video diffusion models, particularly on RLBench, but still lags behind our approach. Additionally, training directly with point clouds is computationally expensive, restricting the number of frames used. In contrast, our model leverages an efficient representation with RGB-DN videos to generate more precise 4D scenes, particularly in capturing fine-grained details in dynamic scenes. We also show qualitative results in Figure 4 (a).

TesserAct synthesizes novel views efficiently. Our method could also perform monocular video to 4D tasks by predicting depth and normal sequences and generating point clouds. We conduct experiments on the task of novel view synthesis on RLBench and compare with Shape of Motion [62], a state-of-the-art video reconstruction approach that utilizes Gaussian splatting [33]. The input is a monocular front camera video, and we compare results from the overhead and left shoulder cameras. We report PSNR (reconstruction accuracy), SSIM (structural similarity), LPIPS (perceptual difference), CLIP Score (semantic match) [78], CLIP aesthetic (visual quality) [37], and Time costs in Table 4. The results show our method can synthesize novel views of higher visual quality and better alignment in signif-

icantly less time. A qualitative result on the Bridge dataset is shown in Figure 4 (c).

Consistency and Regularization Loss are effective. We conduct ablation experiments on our newly designed loss terms in Sec. 4.3. The results are shown in Figure 3. The first two rows demonstrate the effect of the consistency loss, where we render frames from the same camera view at different time steps. The results show that the robot arm’s movements are more coherent with the consistency loss applied. The last row highlights the role of the regularization loss. We display images of the same frame from three different views, revealing that this loss term helps improve the geometric accuracy of the reconstruction.

TesserAct shows generalization across scenes and embodiments. Our model demonstrates strong generalization capabilities. Benefiting from the knowledge of CogVideoX, the model achieves good generation on unseen scenes and unseen objects. Additionally, it performs well in crossembodiment scenarios, such as using the Bridge dataset robotic arm in the RT-1 environment. Figure 4 (b) shows a generalization result on unseen scenes and objects. More results can be found in the Supplementary Materials.

#### 5.2. Embodied Action Planning

Dataset. We select 9 challenging tasks from RLBench [26] including tasks requiring high-precision grasping. Metric. The success rate averaged over 100 episodes. Baseline. We compare our method to a behavior cloning agent and a video-based world model.

- • Image-BC [27]: a behavior cloning agent that takes in an image and task instruction and outputs the 7-DoF actions.
- • UniPi∗ [15]: a method that takes the task instruction and current image, predicts the future video, and uses a 2Dbased inverse dynamic policy to predict actions. For this baseline, we re-implement it by replacing the backbone with fine-tuned CogVideoX [69] for fair comparison.

Implementation Details. We collected 500 samples for each task to train the inverse dynamic model. Given an initial state during inference, we first predict and record all future keyframes. In subsequent actions, we only query the inverse dynamic model to obtain the corresponding actions by the current state and the predicted future state. We post-trained the model from Sec. 5.1, allowing the model to predict only the keyframes for each task in RLBench. Our maximum

[Figure 121]

(a) In-domain 4D Generation Results

Input instruction: Place the red fork to the left of the left burner – Trossen WidowX 250

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

RGB

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Depth

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Normal

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

PCD

[Figure 158]

(b) Out-of-domain 4D Generation Results

Input instruction: Pick up brown cup – Franka Panda Robot

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

RGB

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Depth

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

Normal

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

PCD

[Figure 195]

(c) Novel View Synthesis

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Input View Novel View

- Figure 4. Qualitative results of (a) In-domain 4D generation results. (b) Generalization over unseen scenes and objects. (c) Novel view synthesis.

frame length is 13, with a fixed resolution of 512 × 512.

##### 5.2.1. Results and Analysis

The results are shown in Table 3, where our method outperforms video diffusion models and image behavior cloning agents in most of the tasks. This is because, in most tasks, 4D point clouds can reveal the geometry of objects, providing better spatial guidance for robotics planning, as seen in tasks like close box and open jar. At the same time,

- 3D information can assist with tool use, such as in tasks like sweep to dustpan and water plants. However, in the open microwave and weighing off tasks, the performance is not as good as the baseline, possibly because these tasks already have sufficient information in the 2D front image. Overall, these results highlight the potential of combining 4D scene prediction with inverse dynamic models to improve robotics task execution.

6. Conclusion

We learn a 4D generative world model, TesserAct, using a collected 4D embodied video dataset, which consists of robotic manipulation videos annotated with depth and normal information. To ensure both temporal and spatial consistency in scene reconstruction, we introduce two novel loss terms. Our experiments across synthetic and real-world datasets demonstrate that our model generates high-quality

- 4D scenes and significantly enhances the performance of downstream embodied tasks by leveraging 3D information. We believe that such world models will become increasingly powerful and essential, serving as a foundation for simulating the physical world and advancing the development of intelligent embodied agents. These models will enable fully offline policy training in the real world and facilitate planning through imagined rollouts within the learned world representation.

### 7. Limitations

While our RGB-DN representation of a 4D world model is cheap and easy to predict, it only captures a single surface of the world. To construct a more complete 4D world model, it may be interesting in the future to have a generative model that generates multiple RGB-DN views of the world, which can then be integrated to form a more complete 4D world model.

### Acknowledgements

We are extremely grateful to Pengxiao Han for assistance with the baseline code, and to Yuncong Yang, Sunli Chen, Jiaben Chen, Zeyuan Yang, Zixin Wang, Lixing Fang, and many other friends for their helpful feedback and insightful discussions.

### References

- [1] Alessandro Achille and Stefano Soatto. A separation principle for control in the age of deep learning. Annual Review of Control, Robotics, and Autonomous Systems, 1:287–307,

2018. 2

- [2] Gwangbin Bae and Andrew J. Davison. Rethinking inductive biases for surface normal estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2024. 3, 4

- [3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006,

2024. 3

- [4] Dimitri P. Bertsekas. Dynamic Programming and Optimal Control. Athena Scientific, 1995. 2
- [5] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. Zoedepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 14
- [6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 3, 4, 6, 13, 18
- [7] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 2
- [8] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack ParkerHolder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 2
- [9] Xu Cao, Hiroaki Santo, Boxin Shi, Fumio Okura, and Yasuyuki Matsushita. Bilateral normal integration. In European Conference on Computer Vision, pages 552–567. Springer,

2022. 3

- [10] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14455–14465,

2024. 5

- [11] Chang Chen, Yi-Fu Wu, Jaesik Yoon, and Sungjin Ahn. Transdreamer: Reinforcement learning with transformer world models. arXiv preprint arXiv:2202.09481, 2022. 2
- [12] Silvia Chiappa, S´ebastien Racaniere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. arXiv preprint arXiv:1704.02254, 2017. 2
- [13] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An

- embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 2
- [14] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. arXiv preprint arXiv:2310.10625, 2023. 2
- [15] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in Neural Information Processing Systems, 36, 2024. 2, 7
- [16] Jean-Denis Durou, Jean-Franc¸ois Aujol, and Fr´ed´eric Courteille. Integrating the normal field of a surface in the presence of discontinuities. In International Workshop on Energy Minimization Methods in Computer Vision and Pattern Recognition, pages 261–273. Springer, 2009. 3
- [17] Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite markov decision processes. In UAI, pages 162–169,

2004. 2

- [18] Roya Firoozi, Johnathan Tucker, Stephen Tian, Anirudha Majumdar, Jiankai Sun, Weiyu Liu, Yuke Zhu, Shuran Song, Ashish Kapoor, Karol Hausman, et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, page 02783649241281508, 2023. 2
- [19] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz MuellerFreitag, et al. The” something something” video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017. 3, 4
- [20] Maitrey Gramopadhye and Daniel Szafir. Generating executable action plans with environmentally-aware language models. arXiv preprint arXiv:2210.04964, 2022. 2
- [21] David Ha and J¨urgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2018. 2
- [22] Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021. 2
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [24] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023. 2
- [25] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023. 2
- [26] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters,

2020. 3, 4, 6, 7, 13

- [27] Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn. Bc-z: Zero-shot task generalization with robotic imitation learning. In Conference on Robot Learning, pages 991–1002. PMLR, 2022. 7
- [28] Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022. 2
- [29] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. arXiv preprint arXiv:2210.03094, 2(3):6, 2022. 2
- [30] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360 {\deg} dynamic object generation from monocular video. arXiv preprint arXiv:2311.02848,

2023. 3

- [31] Bingxin Ke, Dominik Narnhofer, Shengyu Huang, Lei Ke, Torben Peters, Katerina Fragkiadaki, Anton Obukhov, and Konrad Schindler. Video depth without video models, 2024. 3, 4
- [32] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9492–9502,

2024. 3, 5

- [33] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 3, 7
- [34] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2
- [35] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3, 4
- [36] Frank Klinker. Exponential moving average versus moving exponential average. Mathematische Semesterberichte, 58: 97–107, 2011. 13
- [37] LAION-AI. Aesthetic predictor. https://github.com/ LAION-AI/aesthetic-predictor, 2022. 7
- [38] Timoth´ee Lesort, Natalia D´ıaz-Rodr´ıguez, Jean-Franois Goudou, and David Filliat. State representation learning for control: An overview. Neural Networks, 108:379–392,

2018. 2

- [39] Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Aky¨urek, Anima Anandkumar, et al. Pre-trained language models for interactive decision-making. Advances in Neural Information Processing Systems, 35:31199–31212, 2022. 2
- [40] Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, and Carl Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. arXiv preprint arXiv:2406.16862, 2024. 2
- [41] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align your gaussians: Text-to-4d

- with dynamic 3d gaussians and composed diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8576–8588, 2024. 3
- [42] Lennart Ljung and Torkel Glad. Modeling of dynamic systems. Prentice-Hall, Inc., 1994. 2
- [43] Vincent Micheli, Eloi Alonso, and Fran¸cois Fleuret. Transformers are sample efficient world models. arXiv preprint arXiv:2209.00588, 2022. 2
- [44] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 3
- [45] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 6
- [46] OpenAI. Video generation models as world simulators. https://openai.com/index/videogeneration- models- as- world- simulators/,

2023. Accessed: 2024-10-01. 2

- [47] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

- 2023. 3

[48] Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox. The colosseum: A benchmark for evaluating generalization for robotic manipulation.

- 2024. 4

- [49] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. Advances in neural information processing systems, 30, 2017. 6
- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 6
- [51] Shreyas Sundara Raman, Vanya Cohen, Eric Rosen, Ifrah Idrees, David Paulius, and Stefanie Tellex. Planning with large language models via corrective re-prompting. arXiv preprint arXiv:2211.09935, 2022. 2
- [52] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 14
- [53] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. arXiv preprint arXiv:2312.17142, 2023. 3
- [54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3
- [55] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea

- Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280, 2023. 3
- [56] Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin, 2(4):160– 163, 1991. 2
- [57] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, et al. Aether: Geometric-aware unified world modeling. arXiv preprint arXiv:2503.18945, 2025. 2
- [58] Open X-Embodiment Team. Open X-Embodiment: Robotic learning datasets and RT-X models. https://arxiv. org/abs/2310.08864, 2023. 4
- [59] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020. 5
- [60] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 3, 4
- [61] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe HansenEstruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023. 3, 4, 6, 13, 18
- [62] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764,

2024. 7

- [63] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: interactive planning with llms enables open-world multi-task agents. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. 2
- [64] Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu Feng, Tianhua Tao, Shibo Hao, Yemin Shi, et al. Pandora: Towards general world model with natural language actions and video states. arXiv preprint

- arXiv:2406.09455, 2024. 2

[65] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint

- arXiv:2407.17470, 2024. 3

- [66] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, and Michael J. Black. ECON: Explicit Clothed humans Optimized via Normal integration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2023. 3

- [67] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 2
- [68] Sherry Yang, Ofir Nachum, Yilun Du, Jason Wei, Pieter Abbeel, and Dale Schuurmans. Foundation models for decision making: Problems, methods, and opportunities. arXiv preprint arXiv:2303.04129, 2023. 2

- [69] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 4, 6, 7, 13
- [70] Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal. arXiv preprint arXiv:2406.16864, 2024. 5
- [71] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4dgen: Grounded 4d content generation with spatial-temporal consistency. arXiv preprint arXiv:2312.17225, 2023. 3
- [72] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. Wonderworld: Interactive 3d scene generation from a single image. arXiv:2406.09394, 2024. 5
- [73] Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. arXiv preprint arXiv:2307.02485,

2023. 2

- [74] Hongxin Zhang, Zeyuan Wang, Qiushi Lyu, Zheyuan Zhang, Sunli Chen, Tianmin Shu, Yilun Du, and Chuang Gan. Combo: Compositional world models for embodied multiagent cooperation. arXiv preprint arXiv:2404.10775, 2024.

- 2

[75] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023.

- 3

- [76] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024. 2, 5, 14
- [77] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 2, 3, 4, 6
- [78] SUN Zhengwentai. clip-score: CLIP Score for PyTorch. https://github.com/taited/clipscore, 2023. Version 0.1.1. 7
- [79] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024. 2
- [80] Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. Irasim: Learning interactive realrobot action simulators. arXiv preprint arXiv:2406.14540,

2024. 2

- [81] Karl J. Astr¨˚ om and Bj¨orn Wittenmark. Adaptive control of linear time-invariant systems. Automatica, 9(6):551–564,

1973. 2

## TesserAct: Learning 4D Embodied World Models Supplementary Material

### 1. Implementation Details

Reference RGB & Normal Image Untextured Rendering Detailed Front View Side Perspective View

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

#### 1.1. Video Diffusion Model Details

We trained an RGB-DN video diffusion model using the CogVideoX [69] architecture. On the input side, our depth normal projector and RGB projector shared the same architecture. On the output side, our Conv3DNet consisted of three layers, while the MLP had two layers, both with a dimension of 1024. The model outputs videos with 49 frames, utilizing gradient checkpointing to optimize memory usage. We set a global batch size of 16 and used bf16 precision to accelerate. For sampling, we employed the DDPM scheduler across 50 steps and set a classifier-free guidance scale of 7.5.

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

- Figure 5. Visualization of the optimized 3D robotic scene reconstruction using our method. The untextured renderings show enhanced detail (green box) and improved surface smoothness (red box). The side perspective view highlights accurate shape and geometry optimization, including the perpendicular alignment of the wall and table (red boxes).

1.3. Implementation Details for Robotics Planning

For the RLBench training, we adopted the same architecture and methods as our video diffusion model, with the primary difference being that we used 13 frames and fine-tuned the model.

For the action prediction stage, we first filter out the background and floor from the data, focusing only on the points of the table and the objects manipulated by the robotic arm, and then sample 8192 points from the filtered point cloud. In our inverse dynamic model, the PointNet extracts features from this point cloud, concatenated with the instruction’s language embedding, and passed into a 4-layer MLP, finally outputting the 7DoF actions. To augment the data and better adapt to the output of video diffusion models, we add significant Gaussian noise (with a relative magnitude of 20%) to both the image and point cloud coordinates.

Reference RGB & Depth Image Ours 3D-VLA Generated Data

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

- Figure 6. Comparison of point cloud generation quality between our method and 3D-VLA

The training spanned 40,000 iterations with an initial learning rate of 1 × 10−4, a gradient clipping of 1.0, and a 1,000-step warmup. The optimizer used was Adam with ϵ set to 1×10−15, and an exponential moving average (EMA [36]) decay of 0.99 was applied to stabilize training.

#### 1.2. 4D Scene Generation

The parameters for the loss term in Eq.12 are set differently for the RT-1 [6], Bridge [61] and RLBench [26] datasets, as shown in the table below. It is worth noting that the selection of λ varies for different scenarios. In practice, achieving the best performance requires tuning these parameters.

Dataset λd λb λg1 λg2 RT-1, Bridge 20 200 20 20

RLBench 20 200 2 2

Table 5. Loss Term Parameters for RT-1 and RLBench Datasets

In Figure 5, we present a visualization of the 3D robotic scene reconstruction optimized using our proposed method in the BridgeV2 [61] dataset. After estimating the depth and normal with the estimator, we refine the outputs to reconstruct the scene accurately. The figure includes untextured rendering and texture-rendered views, where the wall textures are significantly enhanced due to normal optimization. The side perspective view shows the improved shape and geometry reconstruction. Notably, the wall and table surfaces are well-aligned, appearing perpendicular to each other, further validating the effectiveness of our optimization process in capturing accurate spatial relationships.

[Figure 217]

Out-of-domain 4D Generation Results

Input instruction: Pick coke can– Franka Panda Robot

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

RGB

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

Depth

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Normal

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

PCD

[Figure 254]

(b) Out-of-domain 4D Generation Results

Figure 7. Out-of-domain 4D generation results

Input instruction: Pick up brown cup – Franka Panda Robot

### 2. More Qualitative Results

#### 2.4. Explicit Action Planning

[Figure 255]

One potential application of our generated mesh is to extract action trajectories directly. As illustrated in Figure 8, we track the robotic arm in the video to capture its motion path. This trajectory is subsequently lifted into 3D space, enabling the reconstruction of the robot arm’s action trajectory. The red line in the visualization represents the captured action trajectory.

#### 2.1. Data Annotation

[Figure 256]

In this section, we first compare our data generation method with 3D-VLA [76]. They use ZoeDepth [5] for depth map estimation and directly map them into 3D space. The comparison results, shown in Figure 6, evaluate the quality of point cloud generation for both methods, with cubes replacing vertices for rendering. Our generated data demonstrates higher realism, while 3D-VLA exhibits noticeable shape distortion. Figure 12 showcases some of the RGB, depth, and normal images from the datasets we used, along with the corresponding natural language instructions.

[Figure 257]

RGB

[Figure 258]

[Figure 259]

[Figure 260]

Depth

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Beginning Mid End

[Figure 271]

[Figure 272]

Normal

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

- View 1

[Figure 278]

[Figure 279]

- View 2

[Figure 280]

#### 2.2. 4D Generation

As shown in Figure 7, we present our out-of-domain results. We used DALL-E [52] to generate an image and prompted the 4D world model for generation. Our video diffusion model directly produces the RGB, depth, and normal maps, while the point clouds are rendered from the reconstructed outputs. These results highlight the robustness of our method across different visual modalities. Additional video results can be found in the supplementary materials folder for further analysis and evaluation.

[Figure 281]

[Figure 282]

PCD

###### Figure 8. Tracking and visualization of robotic arm action trajectories on the Bridge dataset

#### 2.3. Video Generation

We present the video generation results on the RT1, Bridge and RLBench datasets in Figure 11, Figure 9 and Figure 10. More videos can be found in our supplementary folder.

[Figure 283]

In-domain Video Generation Results

Move the bowl to the left of the blue towel - Trossen WidowX 250

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

RGB

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

Depth

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Normal

Put broccoli in pot or pan - Trossen WidowX 250

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

RGB

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

Depth

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

Normal

Put carrot on plate - Trossen WidowX 250

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

RGB

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

Depth

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

Normal

Figure 9. In-domain RGB-DN Video generation results on Bridge dataset

[Figure 347]

In-domain Video Generation Results

Close middle drawer – Google Robot

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

RGB

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Depth

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

Normal

Place Red Bull can upright – Google Robot

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

RGB

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

Depth

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Normal

Place green Jalapeno chip bag into top drawer – Google Robot

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

RGB

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Depth

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

Normal

Figure 10. In-domain RGB-DN Video generation results on RT1 dataset

[Figure 411]

In-domain Video Generation Results

Open wine bottle - Franka

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

RGB

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

Depth

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

Normal

Set up chess - Franka

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

RGB

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

Depth

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Normal

Take off weighing scales - Franka

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

RGB

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

Depth

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

Normal

Figure 11. In-domain RGB-DN Video generation results on RLBench dataset

[Figure 475]

###### Sample Frames from the Datasets.

Bridge: Put stuffed duck in pan.

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

RGB

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

Depth

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

Normal

Bridge: Put pan on stove and put stuffed pig in pan.

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

RGB

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

Depth

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

Normal

RT-1: Place redbull can into middle drawer.

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

RGB

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

Depth

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

Normal

RT-1: Place green can upright.

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

RGB

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

Depth

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

Normal

Figure 12. Some sample frames extracted from the datasets Bridge [61] and RT-1 [6].

