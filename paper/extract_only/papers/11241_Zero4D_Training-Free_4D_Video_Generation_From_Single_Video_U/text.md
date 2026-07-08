## Zero4D: Training-Free 4D Video Generation From Single Video Using Off-the-Shelf Video Diffusion Models

# arXiv:2503.22622v4[cs.CV]4Dec2025

Jangho Park Taesung Kwon Jong Chul Ye KAIST

{jhq1234, star.kwon, jong.ye}@kaist.ac.kr

[Figure 1]

Figure 1. Zero4D is a training-free multi-view synchronized video generation framework that takes a single monocular video and generates a grid of camera-time consistent frames. It first utilizes a depth estimation model to warp target view frames from the input video (top-left), then repurposes the image-to-video diffusion model to sample multi-view frames synchronized in both camera and temporal dimensions (top-right). Using an off-the-shelf video diffusion model without training, our approach can generate multi-view videos for both synthesized and real-world footage. Video results are available on our project page: zero4dvid.github.io.

### Abstract

Multi-view and 4D video generation have emerged as important topics in generative modeling. However, existing approaches face key limitations: they often require orchestrating multiple video diffusion models with additional training, or involve computationally intensive training of full 4D diffusion models—despite limited availability of real-world 4D datasets. In this work, we propose a novel trainingfree 4D video generation method that leverages off-the-shelf video diffusion models to synthesize multi-view videos from a single input video. Our approach consists of two stages.

First, we designate the edge frames in a spatio-temporal sampling grid as key frames and synthesize them using a video diffusion model, guided by depth-based warping to preserve structural and temporal consistency. Second, we interpolate the remaining frames to complete the spatio-temporal grid, again using a video diffusion model to maintain coherence. This two-step framework allows us to extend a single-view video into a multi-view 4D representation along novel camera trajectories, while maintaining spatio-temporal fidelity. Our method is training-free, requires no access to multi-view data, and fully utilizes existing generative video models offering a practical and effective solution for 4D video generation.

### 1. Introduction

Since the introduction of the diffusion and foundation models [9, 19, 32], 3D reconstruction has advanced significantly, leading to unprecedented progress in representing the real world in 3D models. Combined with generative models, this success drives a renaissance in 3D generation, enabling more diverse and realistic content creation. These advancements extend beyond static scene or object reconstruction and generation, evolving toward dynamic 3D reconstruction and generation that aims to capture the real world. Previous works [2, 3, 22, 39, 41] leverage video diffusion models (VDM) and Score Distillation Sampling (SDS) to enable dynamic 3D generation. However, most existing approaches primarily focus on generating dynamic objects in blank or simplified backgrounds (e.g., text-to-4D generation), leaving the more challenging task of reconstructing or generating real-world scenes from text prompts, reference images, or input videos largely unaddressed. In contrast to the abundance of high-quality datasets for 3D and video tasks, 4D datasets with multiview, temporally synchronized video remain extremely scarce. As a result, a core challenge in training 4D generative models for real-world scenes lies in the lack of comprehensive, large-scale multi-view video datasets. To overcome these limitations, recent works such as 4DiM [29] propose a joint training diffusion model with 3D and video with a scarce 4D dataset. CAT4D [30] proposes training multi-view video diffusion models by curating a diverse collection of synthetic 4D data, 3D datasets, and monocular video sources. DimensionX [23] trains the spatial-temporal diffusion model independently with multiple LoRA, achieving multi-view videos via an additional refinement process. Despite several approaches, the scarcity of high-quality 4D data makes it difficult to generalize to complex real-world scenes and poses fundamental challenges in training large multi-view video models.

To address these challenges, we introduce Zero4D—a novel zero-shot framework for 4D video generation. Zero4D generates synchronized multi-view 4D video from a single monocular input video by leveraging an off-the-shelf video diffusion model [5], without requiring any additional training. Building upon the prior observations [26, 30] that 4D video is composed of multiple video frames arranged along the spatio-temporal sampling grid (i.e., camera view and time axes), generating a 4D video can be regarded as populating the sampling grid with consistent spatio-temporal frames. Consequently, our approach achieves this through two key steps: (1) We first designate the boundary frames of the spatio-temporal sampling grid as key frames and synthesize them using a video diffusion model. To ensure structural fidelity, we incorporate a depth-based warping technique as guidance, encouraging the generated frames to conform to the underlying scene geometry. (2) We repurpose the interpolation capabilities of a video diffusion model to fill in the

Table 1. Comparison of camera controllable VDM and 4D VDM. Unlike prior approaches, Zero4D can generate 4D-consistent videos with camera control without requiring additional training.

Model Training-Free Camera Control 4D Consistency

Camera VDM [8, 31, 38] ✗ ✓ ✗ 4D VDM [23, 26, 30, 32] ✗ ✓ ✓ Zero4D (Ours) ✓ ✓ ✓

remaining frames through bidirectional diffusion sampling, resulting in a fully populated and temporally coherent 4D grid. Throughout both stages, our method enforces spatial and temporal consistency across the entire grid.

Our main contributions can be summarized as follows:

- • We propose a novel framework that can generate 4D video from a single video via an off-the-shelf video diffusion model without any training or large-scale datasets. To the best of our knowledge, our approach is the first interpolation based training-free method to generate synchronized multi-view video— previously regarded as infeasible.
- • This is made possible by a novel synchronization mechanism, which guarantees high-quality outputs while maintaining global spatio-temporal consistency. Specifically, we alternate bidirectional video interpolation across both the camera and temporal axes to align motion and appearance throughout the sequence.
- • Our framework outperforms baselines in maintaining global spatio-temporal consistency and demonstrates robust 4D video generation capability, achieving competitive performance across diverse quantitative and qualitative evaluations even without additional training.

### 2. Related work

Video generation with camera control. Several studies try to train a multi-view diffusion model for spatially consistent image generation [6, 12, 16, 18, 21, 27]. ReCapture [40] trains the novel camera trajectory video diffusion model from a single reference video with existing scene motion. They train LoRA layers with camera labels to regenerate the anchor video into a novel view. Camco[33] fintunes pre-trained video diffusion model with injecting Pl¨ucker embedding vector into a specific layer in the model. CameraCtrl [8] proposes a plug-and-play camera module in the video diffusion model to control video generation with precise and smooth camera viewpoints. TrajectoryCrafter [38] and TrajectoryAttention [31] fine-tune video diffusion models to generate novel-view videos along a given camera trajectory using depth-based warping. These approaches can be categorized as camera-controllable video diffusion models. However, although these models can synthesize novel views conditioned on warped videos, they fail to produce 4D-consistent videos that ensure global consistency across multiple views and multiple time steps (see Table 1).

4D generation. Recent advancements in 4D generation have

TrajectoryCrafter TrajectoryAttention

Ours CameraCtrl

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

x-tsliceTime2Time1

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

𝑥

𝑥

𝑥

𝑥

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

𝑡

𝑡

𝑡

𝑡

- Figure 2. Qualitative comparison. We compare our method with baseline models in terms of novel-view video generation and global spatio-temporal consistency. Given a single input video, both baselines and ours generate outputs across multiple views and time steps. To evaluate global consistency, we leverage baselines to produce bullet-time videos at all input frames and re-align them to a fixed viewpoint. We also visualize x–t slices (red lines) to highlight temporal coherence. While baselines exhibit inconsistencies across views and time, our method preserves spatio-temporal coherence and yields high-quality multi-view videos.

been driven by numerous pioneering works exploring various conditioning methods. Several approaches have leveraged score distillation sampling in conjunction with video diffusion models or multi-view image diffusion models to generate 4D content from text prompts [3, 22, 39]. However, these approaches largely focus on generating dynamic objects in blank backgrounds. A notable example is CAT4D [30], which synthesizes 4D videos conditioned on multiple input modalities using a multi-view video model trained on a curated synthetic multi-view dataset. Similarly, [25] introduces a framework for novel-view synthesis of dynamic 4D scenes from a single video. This method is trained on synthetic multi-view video data with corresponding camera poses, enabling high-fidelity 4D reconstructions. Concurrently, 4Real [37] proposes text-to-4D scene generation pipelines that integrate video diffusion models with canonical 3D Gaussian Splatting (3DGS) [14], ensuring spatiotemporal consistency in the generated 4D outputs. Furthermore, 4Real-Video [26] enhance video diffusion models by introducing a parallel camera-temporal token stream and a learnable synchronization layer, which effectively fuses independent tokens to maintain camera and temporal consistency across generated frames. While these 4D video diffusion models enable camera control and maintain multi-view and temporal consistency, they rely on training a large diffusion model with 4D data, which is limited in availability and costly to obtain (see Table 1).

### 3. Zero4D

Let x[i,j] ∈ RH×W,i = 1,··· ,N,j = 1,··· ,F denotes the image at the i-th camera viewpoint and the j-th temporal frame, where H and W denote the height and width of the image, respectively (see Fig. 3(a)). Then, the input video captured from a single camera viewpoint c is denoted as x[c,:], whereas the multi-view images at the temporal frame f are represented by x[:,f]. The goal of Zero4D is then to populate the spatio-temporal video grid (or camera-time grid) x[:,:] by generating frames across multiple camera poses. The key innovation is that the spatio-temporal grid can be populated entirely at inference time, without any training—a task once thought impossible. As illustrated in Fig. 3, the overall reconstruction pipeline of Zero4D is composed of two steps: 1) key frame generation and 2) spatio-temporal bidirectional interpolation along the time and camera axes in an alternating manner. In this section, we describe each in detail.

#### 3.1. Key frame generation

As shown in Fig. 3(a), the key frame generation is achieved through three steps. Specifically, given a input video denoted by x[1,:], we first perform novel-view synthesis, followed by end-view video frame generation. These two steps are achieved through diffusion sampling, guided by warped views. Finally, we complete the rightmost column using diffusion-based interpolation sampling.

Novel view synthesis (a1). First, we synthesize novel view

(a) Key Frame Generation (b) Spatio-Temporal Bidirectional Interpolation

Time

[Figure 14]

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Camera

Input video (a1) Novel view synthesis

[Figure 15]

Initial noise

Spatio-temporally consistent videos

[Figure 16]

Camera axis interpolation Time axis interpolation

𝑡 = 𝑇 Diffusion time step 𝑡 = 0

(a2) End view video generation

(a3) End frame view synthesis

- Figure 3. Generation pipeline of Zero4D: (a) Key frame generation step: Starting from the input video(shown as the gray-shaded row), we sequentially generate boundary frames—novel view synthesis, end-view video generation, and end-frame view synthesis—where each step leverages the results of the previous one. (b) Spatio-temporal bidirectional interpolation step: Starting from the noisy frames, we alternately perform camera-axis and time-axis interpolation, each conditioned on boundary frames, to progressively denoise the 4D grid. Through this bidirectional process, noisy latents are refined into globally coherent spatio-temporal videos. The detailed algorithm is described in Algorithm 1.

video x[:,1] from the first frame image x[1,1] using the I2V diffusion model. Here, we incorporate the warped frames xw[:,1] as guidance to ensure the generated novel views align with the warped images from input video. The warped frames xw[:,:] are computed as follows. Given an input video x[1,:], we generate novel views by first estimating a per-frame depth map D[1,:] using a video depth estimation model [10], which ensures a shared scale and shift across all frames. This depth information enables depth-based geometric warping, wherein each frame of the input video is unprojected into 3D space and reprojected into a target viewpoint in p(n) ∈ PN where PN defines the desired set of camera views. This produces the warped frames:

consistency within the 4D video grid. This can be considered as conditional sampling under the condition of the warped image, occlusion mask, and the input video conditioning. For the case of novel-view synthesis at the temporal frame index j = 1, this corresponds to

##### x[:,1] ∼ pθ (x[:,1] | xw[:,1],mw[:,1],c[1,1]), (3)

where pθ corresponds to the conditional distribution from the trained diffusion model, mw[:,1] is an occlusion mask that identifies missing pixels, and c[1,1] is conditioned embedding vector from x[1,1]. The specific details of conditional video diffusion sampling will be described in Section 3.3.

End view video generation (a2). Similarly, we can synthesize the end-view video x[N,:] from the generated view x[N,1] utilizing warp-guided diffusion sampling.

##### xw[n,i] = W (x[1,i],D[1,i],p(n),K), i = 1,...,F,

(1) for n = 1,··· ,N, where K is the intrinsic camera matrix. The warping function W(·) unprojects each pixel using its estimated depth and reprojects it into the target view. Formally, for each pixel location ri in the i-view, the warped pixel location rj in the novel-view at the j-th camera location is computed as:

##### x[N,:] ∼ pθ (x[N,:] | xw[N,:],mw[N,:],c[N,1]). (4)

This process follows the same video sampling approach as first-frame novel-view synthesis; however, it differs in that it synthesizes the video from the final camera position.

End frame novel-view synthesis (a3). Finally, we generate video at the end-frame novel-view x[:,F], which constitutes the rightmost column of the 4D grid in Fig. 3(a). Given that we already have x[1,F] from the input video and the synthesized end-view frame x[N,F] derived from x[N,:], we incorporate both images to enhance consistency. To this end, we repurpose a video interpolation method that simultaneously conditions on both c[1,F] and c[N,F] for novel-view synthesis. During interpolation, we further incorporate the warped image and its mask to fully exploit the

rj = KPi→jDi(ri)K−1ri, (2) where Pi→j is the transformation from the input to the novelview, and Di(ri) is the depth at ri. Since rj may not align exactly with integer pixel locations, interpolation is applied to assign pixel values. However, missing regions (e.g., occlusions from depth-based projection) often appear in xw. To address this, we utilize a video diffusion model[5] parameterized by θ to inpaint the missing regions and ensure

Algorithm 1: Zero4D overall pipeline

Input: Input video x[1, :], warped views xw[:, :], masks mw[:, :],

Interpolator Iθ

Output: 4D video grid x0[:, :] ∈ RN×F

- 1 Stage A — Boundary/Keyframe generation
- 2 x[:, 1] ∼ pθ(x[:, 1] | xw[:, 1], mw[:, 1], c[1, 1]) // (a1)
- 3 x[N, :] ∼ pθ(x[N, :] | xw[N, :], mw[N, :], c[N, 1]) // (a2)
- 4 for t ← T to 0 do // (a3)

- 5 xt−1[:, F] ← Iθ(xt[:, F], σt, c[1, F], c[N, F], xw[:, F])
- 6 end
- 7 c[:, 1], c[N, :], c[:, F] ← Encode({x[:, 1], x[N, :], x[:, F]})
- 8 Stage B — Spatio–temporal bidirectional interpolation
- 9 xT[:, :] ∼ N(0, I)
- 10 for t ← T to 1 do

- 11 for i ← 1 to F do // Camera-axis interpolation

- 12 xt−1[:, i] ← Iθ(xt[:, i], σt, c[1, i], c[N, i], xw[:, i], mw[:, i]) xt[:, i] ← xt−1[:, i] + σt2 − σt2−1 ϵ

- 13 end
- 14 for j ← 1 to N do // Time-axis interpolation

- 15 xt−1[j, :]←Iθ(xt[j, :], σt, c[j, 1], c[j, F], xw[j, :], mw[j, :])
- 16 end
- 17 end
- 18 return x0[:, :]

available prior information. In particular, we synthesize the last column x[:,F] leveraging video diffusion interpolation method [34]:

xt−1[:,F] = Iθ(xt[:,F], σt, c[1,F], c[N,F], xw[:,F])

for t = T → 0.

(5) where Iθ denotes the one-step denoising using video interpolation. The final novel-view frame x[:,F] is obtained iteratively by applying Iθ over diffusion time steps t = T → 0. The detailed implementation of the interpolation process is provided in Algorithm 2 of Appendix A.4

#### 3.2. Spatio-temporal bidirectional interpolation

As shown in Fig. 3(b), once the keyframes are generated, the remaining task is to fill in the missing sampling grid at the center so the final resulting 4D video remains consistent across both the camera and time axes. Accordingly, it is essential to perform conditioned sampling using the key frames and adjacent frames from the camera and temporal axes. However, a naive image-to-video diffusion model can only condition on a single or two end frames. To address this challenge, we first repurpose a video interpolation approach to generate spatio-temporally consistent samples under multi-view conditions. The key idea is to alternate

interpolation along both the camera and time axes, thereby guiding the overall diffusion trajectory to satisfy the multiple constraints from the keyframes. In this work, we leverage ViBiDSampler [35] as the interpolator, with implementation details provided in Appendix A.4

Camera axis interpolation. Starting from the initial noise xT[:,:] ∼ N(0,I), we select a specific frame in the 4D grid (a column) xt[:,i], and perform an interpolation denoising process(6) using the edge-frame conditions c[1,i] and c[N,i]:

xt−1[:,i]← Iθ(xt[:,i],σt,c[1,i],c[N,i],xw[:,i]) (6)

Here, the image condition c[1,i] is applied first, along with the warped view to guide the diffusion denoising step. The video is then perturbed with noise again, flipped along the camera axis, and subjected to another diffusion denoising step using c[N,i] as the condition. Through these two conditioning steps, xt[:,i] integrates information from both c[1,i] and c[N,i], enabling interpolation-based denoising that preserves consistency across the camera axis. Before proceeding to time axis interpolation, we apply a re-noising step to ensure smooth transitions across generated frames.

Time axis interpolation. After ensuring spatial consistency across the camera axis, we interpolate frames along the time axis to maintain temporal coherence. For each row xt[j,:] in the 4D grid, we perform an interpolation denoising (7) using the start and end frame conditions c[j,1] and c[j,F].

xt−1[j,:]← Iθ xt[j,:],σt,c[j,1],c[j,F],xw[j,:] (7)

Initially, c[j,1] is applied along with the warped view to guide the diffusion denoising step. The frame is then perturbed with noise, flipped along the time axis, and another diffusion denoising step is performed using c[j,F] as the condition. Through this bidirectional conditioning process, xt[j,:] effectively integrates information from both c[j,1] and c[j,F], facilitating interpolation-based denoising that ensures smooth transitions along the time axis. Throughout the diffusion steps, we perform denoising by alternating interpolation along the camera axis and time axis. This approach maintains global coherence while ensuring consistency in multi-view video generation.

#### 3.3. Details of conditional video diffusion

Our work is built upon Stable Video Diffusion (SVD) [5], an image-to-video diffusion model that follows the principles of the EDM framework [13]. SVD utilizes an iterative denoising approach based on an Euler step method, which progressively transforms a Gaussian noise sample xT into a clean signal x0:

σt−1 σt

xt−1(xt;σt,c) := xˆc(xt) +

xt − xˆc(xt) , (8)

Input video Camera orbit controls Input video Dolly and transition controls

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

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Figure 4. Result from Zero4D. Our model generates high-quality 4D videos from a single input video, enabling diverse camera motions such as orbit, transition, and dolly movements. As illustrated, the synthesized videos maintain spatial and temporal consistency across multiple views and frames, effectively rendering novel perspectives that are not present in the original input. Best viewed with Acrobat Reader. Click first two rows’ images to play the video clip.

where the initial noise is xT ∼ N(0,I), xˆc(xt) is the denoised estimate by Tweedie’s formula using the score function trained by the neural network parameterized by θ, and σt is the discretized noise level for each timestep t ∈ [0,T].

Now, we describe how to modify SVD to enable conditional sampling under the condition on warped image xw, occlusion mask m, and conditioning input c. For convenience, we refer to xt[:,:] as xt. From the formulation of the reverse diffusion sampling process in Eq. (8), the reverse diffusion process can be modulated by conditioning on a known scene-prior xknown [17]:

x¯c(xt) = xˆc(xt) · m + xknown · (1 − m), (9) where m is a mask that determines which parts of the scene are known, guiding the denoising process by preserving the warped pixels while allowing the diffusion model to inpaint the missing areas. In our approach, rather than relying on an externally defined scene-prior xknown, we leverage the warped frames xw obtained from depth-based warping as the conditional guidance. Specifically, we redefine the denoising process by replacing xknown with xw and substituting m with the occlusion mask mw:

x¯c(xt) = xˆc(xt) · mw + xw · (1 − mw). (10)

Here, the occlusion mask mw ensures that the visible regions in xw directly guide the denoising process, while the unseen parts are inpainted using the learned prior. By incorporating this modified formulation into the reverse diffusion process, we obtain the following sampling update:

σt−1 σt

xt−1(xt;σt,c) ← x¯c(xt) +

xt − xˆc(xt) , (11)

where the target camera viewpoints influence the generated frames through the depth-warped observations xw, ensuring geometric consistency during video synthesis. Throughout the reverse sampling, we iteratively apply this procedure. Additionally, following the approach of [15, 17], we incorporate resampling annealing to further enhance output quality.

### 4. Experiments

We used the SVD [5] as an I2V model without additional training. The image resolution was fixed at 576×1024, with 25 cameras and a sequence length of 25 frames, a total of multi-view video frames are 625=252. All frames were generated to form a multi-view video following the target camera trajectory. For depth-based warping, we utilized offthe-shelf depth models [10] with various camera movements,

- Table 2. Quantitative result in novel view video generation. We evaluate our method against baselines on VBench, comparing multi-view video results based on novel-view generation from a fixed camera view. Our method achieves the best performance in both frame consistency across videos and image quality of individual frames. (* denotes baselines evaluated with bullet-time re-alignment)

Method

Subject Consistency ↑

Background Consistency ↑

Temporal Flickering ↑

Motion Smoothness ↑

Dynamic Degree ↓

Image Quality ↑

Aesthetic Quality ↑

SV4D [32] 88.76% 91.36% 94.21% 95.28% 49.20% 46.89% 34.36% GCD [24] 90.31% 94.13% 96.14% 93.21% 19.23% 45.77% 32.98% TrajectoryAttention [31] 88.83% 91.42% 96.86% 97.89% 59.50% 42.98% 37.92% TrajectoryCrafter [38] 93.47% 96.93% 98.42% 99.26% 21.00% 52.10% 44.41% Ours 95.55% 95.75% 97.48% 98.34% 27.50% 51.12% 38.22%

CameraCtrl* [8] 91.71% 91.05% 89.98% 91.03% 98.00% 40.12% 35.86% TrajectoryAttention* [31] 94.72% 94.93% 97.61% 98.28% 27.50% 47.75% 42.88% TrajectoryCrafter* [38] 94.71% 94.48% 94.74% 96.81% 32.50% 48.81% 35.86% Ours 95.55% 95.75% 97.48% 98.34% 27.50% 51.12% 38.22%

- Table 3. Quantitative ablation. Ablation studies on generated videos show that incorporating all components yields the best performance.

Subject Consistency ↑

Background Consistency ↑

Temporal Flickering ↑

Motion Smoothness ↑

Dynamic Degree ↓

Image Quality ↑

Aesthetic Quality ↑

Method ATE (m,↓) RPE-T (m, ↓) RPE-R (deg, ↓)

Ours 0.190 0.142 0.53 95.55% 95.75% 97.48% 98.34% 27.50% 51.12% 38.22% w/o STBI 0.175 0.149 0.34 93.23% 92.63% 93.28% 95.24% 100% 52.38% 43.21% w/o warp 0.501 0.251 0.89 93.73% 93.38% 93.98% 96.12% 47.29% 43.79% 36.11%

including orbit controls (right, left), dolly in/out, and vertical transitions (up, down), with further details on the camera movements provided in Appendix A.3.

Baseline models. We compare against state-of-the-art video generation models that support either camera control or multi-view generation: (1) CameraCtrl [8] is a cameracontrollable video diffusion model. Given a single input image, it can synthesize bullet-time videos by following a predefined camera trajectory. (2) TrajectoryCrafter [38], a representative baseline, synthesizes novel-view and bullettime videos from warped frames aligned to a target trajectory. (3) TrajectoryAttention [31] leverages warped video frames from the input video to generate both novel-view and bullettime videos. (4) SV4D [32] is an image-to-video diffusion model capable of generating multiple novel-view videos from a single input video. (5) GCD [24] also takes a single video as input and generates novel views of dynamic 4D scenes by controlling azimuth and elevation angles. Several related works [23, 26, 30, 37] provide no full implementation code, so they are omitted from direct comparison.

Evaluation protocol. We evaluate our method in two categories: (1) fixed novel-view video generation and (2) bullettime video generation. For novel-view evaluation, we adopt VBench [11], which measures seven aspects of video quality, including identity retention, motion coherence, and temporal consistency. For bullet-time evaluation, we assess 3D consistency using pose errors (ATE, RPE-T, RPE-R) [7] obtained via COLMAP [20] and MEt3R [1], a recent metric based on DUSt3R [28] that quantifies geometric consistency

from unposed frames. We conducted all experiments on 50 videos randomly sampled from Webvid-10M [4], comparing ours with baseline models. We also conduct a user study, presented in Appendix A.1, which shows that our method achieves superior human evaluation scores compared to the baselines.

#### 4.1. Fixed novel-view video generation

We evaluate our method in two settings: (1) novel-view generation for video quality, and (2) spatio-temporal consistency for coherence across views and time.

Evaluation of direct novel-view generation. We assess the quality of novel-view videos from fixed target viewpoints using VBench [11]. Zero4D retrieves x[n,:] corresponding to a target camera viewpoint p(n) from the 4D video grid x[:,:] synthesized from the input video x[1,:], while baselines directly generate x[n,:] at viewpoint p(n). For this experiment, we consider baselines capable of direct novel-view generation at viewpoint n, SV4D, GCD, TrajectoryAttention, and TrajectoryCrafter. As shown in the upper part of Table 2, Zero4D, despite being training-free, achieves the highest score in subject consistency and ranks second in five other categories. This demonstrates that ours achieves robust novel-view video generation performance, comparable to models pretrained on large-scale datasets.

Evaluation of global spatio-temporal consistency. To examine whether models maintain global 4D consistency, we construct re-aligned videos at a fixed viewpoint from generated bullet-time videos. For each input frame x[1,i] (i = 1,...,F), baselines generate a bullet-time sequence

- Table 4. Bullet-time video quantitative comparisons. We report results on (1) direct bullet-time generation for spatial coherence and (2) multi-view consistency by re-aligning outputs at fixed time steps. (* denotes baselines evaluated with novel-view re-alignment)

Method ATE (m, ↓) RPE-T (↓) RPE-R (deg ↓) MEt3R ↓

CameraCtrl [8] 0.185 0.155 0.57 0.0264 TrajectoryAttention [31] 0.182 0.113 0.25 0.0202 TrajectoryCrafter [38] 0.170 0.140 2.26 0.0224 Ours 0.190 0.142 0.53 0.0307

TrajectoryAttention* [31] 5.582 3.377 1.65 0.1000 TrajectoryCrafter* [38] 0.211 0.251 3.61 0.0930 Ours 0.190 0.142 0.53 0.0307

x[:,i] along a predefined trajectory. These sequences are aggregated into a 4D grid x[:,:], from which the fixed-view sequence x[n,:] at viewpoint p(n) is extracted. We consider three baseline models capable of bullet-time video generation: CameraCtrl, TrajectoryAttention, and TrajectoryCrafter. In contrast, Zero4D directly retrieves x[n,:] from its generated 4D grid without requiring bullet-time realignment. As shown in the Table 2 (below the horizontal separator), ours achieves the highest scores in five VBench categories and second-best in the remaining two. This strong performance indicates that spatio-temporal interpolation enables Zero4D to preserve global consistency across views and time, whereas baseline models, unable to sample jointly across multi-view and multi-time dimensions, yield inferior consistency. Although baseline models generate plausible bullet-time results at individual time steps, re-alignment to a fixed viewpoint exposes frequent inconsistencies, particularly in the background and the x–t slices shown in Figure 2, which clearly reveal the inconsistencies.

#### 4.2. Bullet-time video generation

We design two evaluations for bullet-time video generation: (1) direct generation along a camera trajectory to assess spatial coherence, and (2) multi-view alignment at fixed time steps to measure global 4D consistency.

Evaluation of direct bullet-time generation. We compare Zero4D against baselines (CameraCtrl, TrajectoryAttention, TrajectoryCrafter) capable of bullet-time generation. Given an input video x[1,:], these models generate bullet-time sequences x[:,i] by smoothly moving the camera along a predefined trajectory at fixed time i. This setting provides a direct evaluation of each model’s ability to produce spatially coherent bullet-time videos from the input. As shown in Table 4 (upper part), Zero4D attains comparable scores to baselines that are explicitly trained for novel-view video generation, despite being a training-free approach.

Evaluation of multi-view consistency in bullet-time. To further assess global 4D consistency, we construct bullettime videos by re-aligning novel-view outputs at a fixed time step. For baseline models, novel-view videos x[n,:] are generated at each target viewpoint p(n) along the predefined camera trajectory, and the frames corresponding to the same

time index are re-aligned to form a bullet-time sequence x[:,:]. In contrast, ours directly retrieves the corresponding sequence x[:,i] from its generated 4D grid x[:,:], without requiring re-alignment. As shown in Table 4 (below part), Zero4D maintains global coherence across views and time, thereby achieving better accuracy in pose estimation (ATE, RPE-T, RPE-R) and lower MEt3R scores, surpassing the performance of baseline approaches.

Ablation. We performed ablation studies under two settings: (1) Without warped frame guidance: removing warped frames from the input degrades image fidelity and weakens structural details. (2)Without spatio-temporal bidirectional interpolation (STBI): generating each novel-view independently breaks multi-view coherence. Table 3, evaluated with ATE, RPE-T, RPE-R in the bullet-time setting and VBench [11] for fixed novel-view, shows that both components are essential for maintaining fidelity and global consistency. Qualitative ablation results are provided in Appendix A.5

#### 4.3. Limitations

Our method can generate high-fidelity 4D videos without any training, yet several limitations remain. First, generating wide camera trajectories, such as full 360-degree rotations, remains challenging because monocular video inputs provide limited geometric cues. Second, when the depth estimation model produces inaccurate depth predictions in challenging cases, suboptimal 4D videos may be generated. In addition, since the 4D generation process is guided by the prior knowledge embedded in a pre-trained video diffusion model, our method may also inherit limitations that are commonly observed in generative models.

### 5. Conclusion

In this work, we introduced a novel training-free approach for synchronized multi-view video generation using an offthe-shelf video diffusion model. Our method generates highquality 4D video through depth-based warping and spatiotemporal bidirectional interpolation, ensuring structural consistency across both spatial and temporal domains. Unlike prior methods that rely on extensive training with video or 4D datasets, our framework achieves competitive performance without additional training. Experiments demonstrate that our approach produces synchronized multi-view videos with superior subject consistency, smooth motion trajectories, and temporal stability. This makes our framework a practical solution for multi-view video generation, particularly in scenarios where large-scale 4D datasets and powerful computational resources are limited. Future work may investigate extensions to more complex dynamic scenes, adaptive interpolation strategies, or fusion with other generative models to further enhance realism and flexibility.

### References

- [1] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring multi-view consistency in generated images, 2024. 7
- [2] Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectoryconditioned text-to-4d generation. In European Conference on Computer Vision, pages 53–72. Springer, 2024. 2
- [3] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7996–8006,

2024. 2, 3

- [4] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-toend retrieval. In IEEE International Conference on Computer Vision, 2021. 7
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 4, 5, 6, 12
- [6] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 2
- [7] Puneet Goel, Stergios I Roumeliotis, and Gaurav S Sukhatme. Robust localization using relative and absolute position estimates. In Proceedings 1999 IEEE/RSJ International Conference on Intelligent Robots and Systems. Human and Environment Friendly Robots with High Intelligence and Emotional Quotients (Cat. No. 99CH36289), pages 1134–1140. IEEE,

1999. 7

- [8] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 7, 8, 11
- [9] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. ArXiv, abs/2006.11239, 2020. 2
- [10] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. arXiv preprint arXiv:2409.02095, 2024. 4, 6, 13
- [11] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7, 8
- [12] Yash Kant, Aliaksandr Siarohin, Ziyi Wu, Michael Vasilkovsky, Guocheng Qian, Jian Ren, Riza Alp Guler, Bernard Ghanem, Sergey Tulyakov, and Igor Gilitschenski.

- Spad: Spatially aware multi-view diffusers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10026–10038, 2024. 2
- [13] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 5
- [14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023. 3
- [15] Kunhao Liu, Ling Shao, and Shijian Lu. Novel view extrapolation with video diffusion priors. arXiv preprint arXiv:2411.14208, 2024. 6, 12, 13
- [16] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298– 9309, 2023. 2
- [17] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11461–11471, 2022. 6, 13
- [18] Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and Filippos Kokkinos. Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation. arXiv preprint arXiv:2402.08682, 2024. 2
- [19] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021. 2
- [20] Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 7
- [21] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2
- [22] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280,

2023. 2, 3

- [23] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 2, 7
- [24] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer, 2024. 7
- [25] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer, 2024. 3

- [26] Chaoyang Wang, Peiye Zhuang, Tuan Duc Ngo, Willi Menapace, Aliaksandr Siarohin, Michael Vasilkovsky, Ivan Skorokhodov, Sergey Tulyakov, Peter Wonka, and Hsin-Ying Lee. 4real-video: Learning generalizable photo-realistic 4d video diffusion. arXiv preprint arXiv:2412.04462, 2024. 2, 3, 7
- [27] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023. 2
- [28] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024. 7
- [29] Daniel Watson, Saurabh Saxena, Lala Li, Andrea Tagliasacchi, and David J Fleet. Controlling space and time with diffusion models. arXiv preprint arXiv:2407.07860, 2024. 2
- [30] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. arXiv preprint arXiv:2411.18613, 2024. 2, 3, 7
- [31] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for finegrained video motion control. In The Thirteenth International Conference on Learning Representations, 2025. 2, 7, 8, 11
- [32] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 2, 7
- [33] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 2
- [34] Serin Yang, Taesung Kwon, and Jong Chul Ye. Vibidsampler: Enhancing video interpolation using bidirectional diffusion sampler. arXiv preprint arXiv:2410.05651, 2024. 5
- [35] Serin Yang, Taesung Kwon, and Jong Chul Ye. VibiDSampler: Enhancing video interpolation using bidirectional diffusion sampler. In The Thirteenth International Conference on Learning Representations, 2025. 5, 12
- [36] Meng You, Zhiyu Zhu, Hui Liu, and Junhui Hou. Nvs-solver: Video diffusion model as zero-shot novel view synthesizer. arXiv preprint arXiv:2405.15364, 2024. 13
- [37] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Laszlo A Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4real: Towards photorealistic 4d scene generation via video diffusion models. arXiv preprint arXiv:2406.07472, 2024. 3, 7
- [38] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025. 2, 7, 8, 11
- [39] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. Stag4d: Spatial-temporal anchored generative 4d gaussians. In European Conference on Computer Vision, pages 163–179. Springer, 2024. 2, 3
- [40] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou,

Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. arXiv preprint arXiv:2411.05003, 2024. 2

[41] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603, 2023. 2

### A. Appendix

#### A.1. User study.

To evaluate our approach, we conducted a user study comparing Ours, TrajectoryCrafter [38], TrajectoryAttention [31], and CameraCtrl [8] across four key metrics: View Angle, General Quality, Smoothness, and Background Quality. Participants viewed generated videos and selected the most visually appealing results for each criterion, providing subjective feedback on the overall quality and realism. As shown in Table 5, our method consistently achieved the highest user preference, particularly excelling in General Quality (36%) and Background Quality (39%), which highlights its superior fidelity and ability to preserve scene details. The View Angle metric (30%) confirms accurate and convincing novel-view synthesis, while Smoothness (33%) indicates our approach produces fluid transitions with minimal distortion or artifacts. These results collectively demonstrate that our method offers a more immersive and visually coherent experience compared to competing techniques.

Table 5. User study. Winning rates across four evaluation metrics. Our method consistently outperforms the baselines, particularly in General Quality and Background Quality.

Method View Angle General Quality Smoothness BG Quality

Ours 30% 36% 33% 39% TrajectoryCrafter 32% 30% 27% 28% TrajectoryAttention 27% 26% 34% 23% CameraCtrl 11% 8% 6% 10%

#### A.2. Pre-trained model checkpoints

Zero4D is developed based on publicly available, pre-trained generative models for both images and videos. For transparency and reproducibility, we specify below the exact versions of each model employed in our framework:

- • Depth estimation model: Depthcrafter
- • Image-to-Video generation model: stable-video-diffusion-img2vid-xt

#### A.3. Camera trajectory control

We support various camera motions for novel view synthesis, leveraging depth information for realistic scene transformation:

Orbit left Orbit right Transition up

- Complex Trajectory 1
- Complex Trajectory 2

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Camera orbit rotation: Horizontal camera movement around the subject, creating a side-to-side viewing effect. The depth map guides proper parallax by determining each pixel’s displacement based on its relative depth.

Transition down Dolly in Dolly out

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Dolly movement: Forward/backward camera translation that adjusts focal length to maintain subject size. For dolly-in, foreground elements remain stable while the background compresses; for dolly-out, the background expands naturally.

Figure 5. Camera trajectory visualization. With a monocular depth estimation model, our approach can generate various novel view videos with spatio-temporal synchronized videos.

Elevation transition: Vertical camera movement that rotates the viewpoint up or down. Depth information ensures accurate perspective shifts as the camera changes height, maintaining geometric consistency.

Complex trajectory: We also conducted experiments on complex camera trajectories. In this setting, the camera moves along a combined path in the x, y, and z axes, first moving inward toward the subject and then moving outward, forming a complex trajectory. In addition, we generated another variant, referred to as complex trajectory 2, where the camera first moves outward and then moves back inward. Our system utilizes monocular video depth estimation to construct a pseudo-3D dynamic representation of the scene. This depth map is crucial for maintaining geometric consistency during novel view synthesis, allowing for convincing parallax effects and occlusion handling. By projecting pixels according to their estimated depth values, we achieve realistic scene transformations without explicit 3D reconstruction.

Algorithm 2: Iθ: A sampling step of extended ViBiDSampler for bidirectional interpolation Function Iθ(xt,σt,cstart,cend,xw):

start ← Dθ(xt; σt,cstart) // EDM denoising

xˆc

- 1 x¯c

start ← xˆc

start · m + xw · (1 − m)

- 2 xt−1,c

start ← x¯c

start

+ σ

t−1

σt (xt − xˆ∅)

- 3 (xt,cstart) ← xt−1,c

start

+ σt2 − σt2−1 ϵ // Re-noise

- 4 (xt,cstart) ← flip(xt,cstart) // Time reverse
- 5 xˆ′c

end

← Dθ(x′t,cstart; σt,cend) // EDM denoising

- 6 x¯′c

end

← xˆ′c

end

· m + xw · (1 − m)

- 7 x′t−1 ← x¯′c

end

+ σ

t−1

σt (x′t − xˆ′∅)

- 8 x′t−1 ← flip(x′t−1) // Time reverse return xt−1

Algorithm 3: Novel view synthesis and end-view video generation algorithm from [15] Input: Warped frames xw, opacity mask m Output: Input video x0

- 1 xT ∼ N(0,1)
- 2 for t ← T to 1 do

- 3 if t > T − Tguide then

- 4 for r ← 1 to R do

- 5 xˆ0 ← Predict(xt)
- 6 if r ≤ Rguide then

- 7 xˆ0 ← Dθ(xt; σt,cx

0

)

- 8 x¯0 ← xˆ0 · m + xw · (1 − m)
- 9 else

- 10 x¯0 ← xˆ0
- 11 end
- 12 xt−1 ← x¯0 + σ

t−1

σt (xt − xˆ0)

- 13 if r < R then

- 14 xt ∼ N(¯x0,σt)
- 15 end
- 16 end
- 17 else

- 18 xˆt−1 ← Dθ(xt; σt,cx

0

)

- 19 xt−1 ← x¯0 + σ

t−1

σt (xt − xˆ0)

- 20 end
- 21 end
- 22 return x0

#### A.4. Details of Zero4D Implementation

Details of interpolation. To generate globally consistent 4D videos, we adapt the interpolation strategy during spatio-temporal video generation. Specifically, we leverage ViBiDSampler [35] as the interpolator Iθ. ViBiDSampler is a state-of-the-art training-free video interpolation method designed for image-to-video diffusion models. Given two conditioning frames, it alternates denoising along the temporal axis to synthesize intermediate frames. In our framework, we extend this process by incorporating warped-frame guidance (see Algorithm 2), which provides additional geometric cues. This modification refines the interpolation process, leading to more faithful structure preservation and improved global spatio-temporal coherence across the generated 4D video grid.

Novle-view synthesis. Algorithm 3 outlines the process for generating novel-view videos from a single monocular video. We first apply novel view synthesis to the initial frame using an I2V diffusion model [5] to produce the novel view x[:,1]. For this,

[Figure 49]

- Figure 6. Input Video Warping. Given a single video, we utilize an off-the-shelf depth estimation model to generate warped frames from novel viewpoints.

depth-based warping priors from the input video are incorporated to enable inpainting-based synthesis. Specifically, using an off-the-shelf depth estimation model [10], we warp the original frame to novel viewpoints, as illustrated in Figure 5. As shown in Fig. 6, occluded regions from the warp operation appear black, allowing us to extract an opacity mask. Inspired by [15, 17, 36], we adopt a mask inpainting approach, where inpainting is performed on the estimated noisy frame xˆ0[:,1]. Rather than applying inpainting at every denoising step, as in [15], we utilize a re-noising process within the diffusion model’s denoising step to refine the final synthesis by reducing artifacts and enhancing structural coherence. A detailed description is provided in Algorithm 3.

A.5. Additional Results

Oursw/oSTBIw/owarp

[Figure 50]

- Figure 7. Ablation results. Removing spatio-temporal bidirectional interpolation (STBI) or warping guidance leads to broken consistency and geometric artifacts (red boxes). In contrast, our full method preserves spatial structure and temporal coherence across views.

Ablation (detailed analysis). Figure 7 qualitatively illustrates the role of each component in maintaining global consistency. Without spatio-temporal bidirectional interpolation (STBI), each frame is synthesized independently, which causes temporal flickering and background inconsistencies across views. For example, in the water-pouring sequence (left), the liquid surface fails to remain temporally stable, as highlighted by the red boxes. Similarly, without warping guidance, the model struggles with geometric alignment. In the motorcycle example (middle), artifacts appear in the generated human figure, leading to distorted or incomplete shapes. Finally, in the clock sequence (right), the absence of warping or spatio-temporal interpolation leads to visible structural mismatches and background inconsistencies. In contrast, our full model effectively aggregates global information through STBI and enforces geometric consistency via warped-frame guidance, resulting in coherent and high-quality multi-view videos across both spatial and temporal dimensions.

Table 6. Quantitative results under low-light and camera-motion blur. We evaluate our method in challenging scenarios where the depth estimation model may fail. We perturb the input videos using motion-blur and low-light filters. The top three rows show bullet-time videos(*), while the bottom three rows present novel-view videos.

Subject Consistency ↑

Background Consistency ↑

Temporal Flickering ↑

Motion Smoothness ↑

Dynamic Degree ↓

Image Quality ↑

Aesthetic

Method

Quality ↑ Ours (Low-light)* 95.03% 95.41% 98.80% 99.31% 2.00% 34.12% 33.46% Ours (Motion blur)* 94.62% 92.88% 94.28% 94.32% 2.22% 37.75% 28.81 Ours * 95.73% 94.81% 96.88% 98.76% 1.00% 38.81% 38.14% Ours (Low-light) 94.28% 96.03% 95.98% 99.22% 30.13% 36.11% 33.76% Ours (Motion blur) 94.76% 94.68% 95.49% 95.99% 32.35% 39.11% 29.21 Ours 95.55% 95.75% 97.48% 98.34% 27.50% 51.12% 38.22%

Quantitative comparison under difficult scenarios. We evaluated our method under difficult scenarios (low-light, blurred camera) to evaluate the robustness of our method. We evaluated the robustness of our proposed method under challenging scenarios, including low-light conditions and camera motion-blurred settings. We use 50 WebVid-10M videos with a low-light filter(30% brightness) and a motion blur filter (20px, 45°). As shown in Table 6, despite minor drops in aesthetic quality, performance remains stable, especially in bullet-time, where static objects are well preserved.

