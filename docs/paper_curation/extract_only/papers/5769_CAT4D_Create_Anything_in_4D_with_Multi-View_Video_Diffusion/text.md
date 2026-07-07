## CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models

Rundi Wu1,2 Ruiqi Gao1 Ben Poole1 Alex Trevithick1,3 Changxi Zheng2 Jonathan T. Barron1 Aleksander Hoły´nski1

1Google DeepMind 2Columbia University 3UC San Diego

“A cat kneading dough”

[Figure 1]

[Figure 2]

[Figure 3]

|[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>[Figure 7]<br><br>|
|---|

|[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]<br><br>[Figure 11]|
|---|

[Figure 12]

[Figure 13]

[Figure 14]

# arXiv:2411.18613v2[cs.CV]18Dec2024

[Figure 15]

[Figure 16]

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

Sparse dynamic capture to 4D Real video to 4D

Text-to-video-to-4D

Figure 1. CAT4D enables 4D scene creation from any number of real or generated images or video frames.

### Abstract

could enable applications in robotics, film-making, video games, and augmented reality.

We present CAT4D, a method for creating 4D (dynamic 3D) scenes from monocular video. CAT4D leverages a multi-view video diffusion model trained on a diverse combination of datasets to enable novel view synthesis at any specified camera poses and timestamps. Combined with a novel sampling approach, this model can transform a single monocular video into a multi-view video, enabling robust 4D reconstruction via optimization of a deformable 3D Gaussian representation. We demonstrate competitive performance on novel view synthesis and dynamic scene reconstruction benchmarks, and highlight the creative capabilities for 4D scene generation from real or generated videos. See our project page for results and interactive demos: cat-4d.github.io.

Advances in 3D reconstruction have enabled the creation of accurate 3D and 4D models when certain careful capture requirements are followed [8, 11, 14, 28, 35, 43, 46, 53, 74]. For static 3D reconstruction, this requires having a large set of multi-view consistent images of the 3D scene. This can be achieved through careful capture and restricted settings, which is challenging to accomplish in most environments, but often feasible with enough time and practice. Reconstructing 4D content, however, requires synchronized multiview video, which is nearly impossible for a typical user.

To reduce the dependence on these challenging capture requirements, several methods have turned to data-driven approaches leveraging learned 3D generative priors to improve the quality of 3D reconstruction [17, 22, 58, 75]. Core to the success of these methods is the availability of highquality multi-view image datasets that enable learning generative priors. While this data exists in large quantities for static environments, the same cannot be said for 4D, where the complexity of building a synchronized multi-view capture setup constrains both the amount and diversity of available data, and therefore the ability to build a data-driven 4D prior. Instead, methods have attempted to regularize

### 1. Introduction

While the world is a dynamic 3D environment, the images and videos we capture represent only a partial snapshot of this reality. Transforming this limited information into an accurate model of the dynamically changing 3D world remains an open research challenge, and progress in this space

| | | |
|---|---|---|
| |Camera Movement| |

[Figure 27]

[Figure 28]

| | |
|---|---|
| | |
| | |

Time

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

Figure 2. Qualitative results: CAT4D can generate high-quality dynamic 3D scenes from a single input monocular video. For each example, we show four rendered images, varying in time along the vertical axis, and varying in viewpoint along the horizontal axis. We also show a depth map (bottom right) and a frame from the input video (top right) at the same timestamp as the second column of renders.

Time

4D reconstruction via geometric constraints [48, 49], estimated depth maps [15, 81], optical flow fields [37, 38], or long-term 2D tracks [33, 72]. But even with all these additional signals, the most effective methods still show noticeable artifacts when viewed from novel viewpoints, especially when inputs only partially observe the target scene.

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Camera

Video Multi-View Camera Control Multi-View Video

Figure 3. What is a multi-view video model? Given one or several input images (grey), different generative models have the ability to create novel images (orange) at certain collections of camera viewpoints and timestamps. Video models generate frames at all timestamps, but without control over camera. Multi-view models generate at controllable cameras but at a fixed timestamp. Camera-controlled video models enable the choice of camera per timestamp, but cannot generate multiple cameras per timestamp. Multi-view video models can generate all views at all timestamps.

In this work, we leverage a variety of real and synthetic data sources to learn a generative prior that enables 4D reconstruction. Inspired by the success of generative priors for sparse-view 3D reconstruction of static scenes [17, 58, 75], we train a multi-view video diffusion model that can transform an input monocular video into multi-view videos, which we then use to reconstruct the dynamic 3D scene. Specifically, our model accepts any number of input images captured at different viewpoints and times, and synthesizes as output the scene’s appearance at any specified novel viewpoints and novel times (see Fig. 3). Because very little real-world multiview training data of dynamic scenes exists, we train our model with a bespoke mixture of multi-view images of static scenes [36, 55, 84, 95], fixed-viewpoint videos containing dynamics, synthetic 4D data [12, 19], and additional data-sources augmented by a pre-trained video model [7] and a multi-view image synthesis model [17]. Our model is trained to produce a consistent collection of frames at specified viewpoints and timestamps, but due resource constraints, it was only trained to generate a finite number of frames at once (16). Since high-quality 4D reconstruction requires many frames at many novel viewpoints, we additionally propose a sampling strategy to generate an unbounded collection of consistent multi-view videos beyond the diffusion model’s native output length. Finally, we use these generated multi-view videos to reconstruct a dynamic

### 2. Related Work

Dynamic 3D Reconstruction Reconstruction and novel view synthesis of dynamic 3D scenes is a challenging research problem. Many existing methods require multiple synchronized videos captured from different viewpoints as input [3, 11, 14, 30, 35, 43, 48, 49, 71, 74], while some more recent works address the more practical setting of casually captured monocular videos as input [32, 33, 37, 38, 72, 89, 94]. These works tend to use radiance field models such as NeRF [46] or 3DGS [28] as the underlying static 3D representations, alongside a deformation field that models the scene motion. Most recently, Shape-of-Motion [72] and MoSca [33] greatly improved reconstruction quality by utilizing extra supervision signals estimated from large vision models, such as segmentation masks [29, 76], depth maps [50, 77] and long-term 2D tracks [13, 27]. However, these methods are not fully automatic (e.g., requiring user clicks for the object mask) and cannot reconstruct regions that are never observed in any input frames. In contrast, we use a diffusion model to transform an input monocular video to multi-view videos, and directly optimize a deformable 3D Gaussians representation [74] without the need of additional supervision signals. 4D Generation Early work for 4D generation used Score Distillation Sampling (SDS) [52] to distill a 4D representation from image and video diffusion models [63]. Follow up work [4, 5, 26, 40, 56, 80, 82, 85, 90, 93] improved generation quality by performing SDS with a combination of video diffusion models [9, 44, 62] and multi-view image diffusion models [41, 42, 61]. Recent work sidestepped costly SDS optimization by crafting a sampling scheme of diffusion models that generate multi-view videos [34, 47, 79], fine-tuning a video model for 360 spins [39] or training a feed-forward 4D reconstruction model [57]. However, most prior work focuses on generating a single dynamic object. Concurrent work DimensionX [66] achieved 4D scene generation by leveraging multiple LoRAs [24] trained on a video model, each tailored to a specific type of camera mo-

- 3D model by optimizing a deformable 3D Gaussian representation [74] with a photometric reconstruction loss.

We evaluate our system, CAT4D, on a variety of tasks: We first showcase our model’s ability to generate images at novel viewpoints and times with disentangled camera and time control, given sparse-view input images at different timestamps. We then demonstrate our model’s ability to perform sparse-view static 3D reconstruction in the presence of scene motion, which is a common failure mode for prior sparse-view reconstruction methods. We then evaluate on the task of 4D reconstruction from monocular videos with moving cameras, and show that our results are comparable to those from existing state-of-the-art models that critically depend on multiple priors and external sources of information — all of which our model does not need or use. Lastly, we show that our model is able to produce compelling 4D generation results from fixed-viewpoint videos that are scene-scale (e.g., multiple dynamic objects within dynamic environments, see Fig. 2), whereas most previous works focus on generating singular objects.

tion, to generate multi-view videos, which were then used to reconstruct a 4D scene. However, it is constrained to a fixed set of predefined camera trajectories and only limited results are shown. In contrast, we train a single multi-view video diffusion model that transforms a monocular video into multi-view videos at controllable novel viewpoints.

Observed input images (a monocular video)

Space

[Figure 39]

Time

[Figure 40]

| | |
|---|---|
|Sample from diﬀu|sion model|
| | |

[Figure 41]

Video Generation Models with Camera Control Our work also relates to video generation models with controllable camera trajectories [6, 20, 21, 23, 31, 66, 69, 78, 83, 86, 91]. However, these models cannot be used to generate multi-view videos that are consistent with each other.

|[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Dynamic 3D Gaussians<br><br>Generated images from observed and novel views at all moments in time<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>[Figure 54]<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>|
|---|

- 4DiM[73] trained a diffusion model for synthesizing images under novel views and timestamps, but it’s unclear whether their model can produce consistent multi-view videos. Our system is not intended to be a video generation model, but is instead a dynamic 3D reconstruction and generation system built around a multi-view video diffusion model. As such, our system has several capabilities that video models lack: exact control over camera location and time by rendering the 4D model, real-time rendering, and enabling various visual effects such as object insertion, 3D editing, etc.

Figure 4. Illustration of the method: Given a monocular video (top), we generate the missing frames (orange frames) of virtual stationary video cameras positioned at all input poses (gray circles) and novel poses (blue circles) using our multi-view video diffusion model. These frames are then used to reconstruct the dynamic 3D scene as deforming 3D Gaussians. Note that although the input trajectory is visualized with changing viewpoints, our method also works for fixed-viewpoint videos.

### 3. Method

CAT4D is a two-stage approach for creating a dynamic 3D scene from a monocular input video (see Fig. 4). First, we transform the monocular video into a multi-view video using a multi-view video diffusion model. Then, we reconstruct a dynamic 3D scene by optimizing a deformable 3D Gaussian representation from these generated multiview videos. We first describe our multi-view video diffusion model (Sec. 3.1) and our carefully curated training datasets (Sec. 3.2). We then describe how our model can be used to perform few-view “bullet-time” 3D reconstruction (Sec. 3.3). Lastly, we describe our sampling strategy for generating nearly-consistent multi-view videos (Sec. 3.4), and our reconstruction method that yields exactly-consistent dynamic Gaussian models (Sec. 3.5).

an identical architecture while injecting our additional time conditioning into the model. Specifically, we encode each time condition t ∈ Tcond ∪ Ttgt via a sinusoidal positional embedding [70] followed by a two-layer MLP, and add that encoding to the diffusion timestep embedding, which is then projected and added to each residual block in the U-Net.

To enable separate guidance for camera and time, we randomly drop out either cT = (Tcond,Ttgt) or both cT and cI = (Pcond,Icond) during training, with a probability 7.5% for each case. We sample from the model using classifierfree guidance as in [73]:

#### 3.1. Multi-view Video Diffusion Model

We train a diffusion model that takes in a set of views of a dynamic 3D scene (where a “view” is an image and its corresponding camera parameters and time) and generates target frames at specified viewpoints and times. Given M input conditional views with a corresponding set of images Icond, camera parameters Pcond and times Tcond, the model is trained to learn the joint distribution of N target images Itgt given their camera parameters Ptgt and times Ttgt,

ϵθ ztgt(i),Ptgt,∅,∅

+sI· ϵθ ztgt(i),Ptgt,cI,∅ − ϵθ ztgt(i),Ptgt,∅,∅

+sT· ϵθ ztgt(i),Ptgt,cI,cT − ϵθ ztgt(i),Ptgt,cI,∅ ,

where ϵθ is the denoising network, ztgt(i) are the latents of all target images at diffusion timestep i, and sI and sT are two guidance scale hyperparameters. Intuitively, sT is to enhance the time alignment of the generated samples, while sI is to encourage consistency with the other conditioning information besides time. We initialize the model with the checkpoints from CAT3D, and train with M = 3 input

p Itgt | Icond,Pcond,Tcond,Ptgt,Ttgt . (1)

This model is built on top of CAT3D’s diffusion model [17], which is a multi-view latent diffusion model that applies 3D self-attention to connect all image latents. We adopt

views and N = 13 target views. For all of our experiments, we set sI = 3.0 and sT = 4.5. Please refer to the supplementary materials for more model and training details.

#### 3.2. Dataset Curation

To fully disentangle camera and time controls (such that Ptgt only controls camera motion and Ttgt only controls scene motion), we would ideally train on a large-scale dataset of multi-view videos that capture a dynamic 3D scene from multiple perspectives. However, this kind of dataset does not yet exist at scale due to the expense of collecting video captures from multiple synchronized cameras [18, 25, 59]. Rendering such a dataset from synthetic assets is straightforward, but existing synthetic 4D datasets [12, 19, 54, 92] by themselves are insufficiently diverse or realistic to train a model that generalizes to challenging real-world scenes. Therefore, we carefully curate available datasets for training to maximally cover different combinations of camera and scene motions. See Table 1 for an overview of our real and synthetic training datasets, grouped by the characteristics of motions in input and target views.

Fortunately, there are many real-world multi-view images of static scenes [36, 55, 84, 95] and real-world monocular videos taken at a static viewpoint, each of which corresponds to one of our control signals: camera or time. We therefore use a mixture of synthetic 4D datasets [12, 19], multi-view image datasets [36, 55, 84, 95], and monocular video datasets. We filter the video dataset to contain only videos of a static viewpoint, in order to prevent the model from confusing time control Ttgt with camera motion (as we lack camera annotation for these videos). We perform this filtering by checking if the four corner patches of each video are constant over time (see supplement).

Until this point, we still do not have real data corresponding to the two cases we particularly care about — both camera and scene moves in the input views but one of them stays static in the target views. To address this issue, we perform generative data augmentations [68]. Specifically, on the one hand, we take samples from CO3D dataset [55], and prompt a video generation model (Lumiere [7]) to animate the input frames without moving the camera. On the other hand, we take samples from the video dataset and run CAT3D [17] to generate novel views of the input frames. While augmented images are often not perfect (e.g., unrealistic deformation, incorrect scene scale), we found them to help the model better learn disentangled camera and time control.

#### 3.3. Sparse-View Bullet-Time 3D Reconstruction

One application of our model is sparse-view static 3D reconstruction in the presence of dynamic scene motion. Given only a few posed images of a dynamic scene, we want to create a “bullet-time” effect by reconstructing a static 3D scene corresponding to the time of one input view.

Input views Target views Camera Real data Synthetic data motion

Scene motion

Camera motion

Scene motion

CO3D [55], MVImgNet [84] Re10K [95], MC4K [36]

✔ ✘ ✔ ✘

✘ ✔ ✘ ✔ static-view videos

Kubric [19], Objaverse [12]

- ✔ ✔ ✔ ✔ -

- ✔ ✔ ✔ ✘

CO3D augmented with Lumiere [7]

- ✔ ✔ ✘ ✔

static-view videos augmented with CAT3D [17]

- ✔ ✔ ✘ ✘ single image

Table 1. Our training datasets: Datasets grouped based on whether each has camera or scene motion in the input or target views. The “single image” row corresponds to randomly (with 1% probability) setting all target views to be one of the input views when drawing samples from all above datasets. For Objaverse, we only use the filtered animated assets [39]. See Sec. 3.2 for details.

To achieve this, we follow a similar two-step approach to CAT3D [17] — we first use the diffusion model with an anchored sampling strategy to generate K novel views at some target time, and then run a robust 3D reconstruction pipeline on those images.

Concretely, we first generate N anchor views by specifying all target times Ttgt to be the same as one of the inputs t ∈ Tcond. Then, we split all K target views into batches of N views and generate each batch by conditioning on the nearest M anchor views. Finally, we reconstruct the 3D scene using a standard 3DGS model [28] with an additional perceptual loss LPIPS [88]. See the supplement for details.

#### 3.4. Generating Consistent Multi-view Videos

To enable 4D reconstruction, we transform the input monocular video to a multi-view video. When there is sufficient camera movement in the monocular input video, we aim to generate multi-view video at a subset of the observed cameras. Given an input video of L frames I1:L with cameras P1:L, we start by picking K camera viewpoints {Pk

i}Ki=1 ⊆ P1:L by farthest point sampling. Our goal is to generate multi-view videos at these K cameras, i.e., a K×L grid of images GK,L. Recall that our model is trained to generate N frames only, yet usually KL ≫ N. To generate such a grid of images, we design a sampling strategy that alternates between multi-view sampling that independently generates each column of the image grid and temporal sampling that independently generates each row of the image grid. Here we use a restricted setting of our multi-view video diffusion model where the number of conditioning frames M is equal to the number of target frames N + 1. In order to condition on more input frames, we use a fine-tuned version of our trained model with M = 9 input views and N = 8 output views. See Fig. 5 for an illustrative example of this procedure.

Multi-view sampling For each time t, we generate images G·,t at all K cameras by first generating each sliding

P2,T2 P3,T3 P4,T4

[Figure 69]

[Figure 70]

[Figure 71]

Time

Time (T)

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Multi-view sampling

Temporal sampling

Multi-view sampling

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Camera

(P)Camera

Input

SDEdit SDEdit

|[Figure 72]<br><br>Multi-view sampling<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>median next column<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
|
|---|

- P1,T3

- P2,T1

- P1,T3

- P2,T1

- P1,T3

- P2,T1

[Figure 73]

[Figure 74]

[Figure 75]

Varying ViewFixed Time

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

|[Figure 76]<br><br>Temporal sampling<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
<br><br>median next row<br><br>| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
<br><br>|
|---|

Figure 5. Illustrating our alternating sampling strategy: Given a diffusion model that generates N output views (here, N = 3), we use SDEdit [45] to alternate between multi-view and temporal sampling to generate a grid of images at K cameras and L time steps (top, here K = 4 and L = 4). In multi-view sampling, we generate each sliding window of size 3 for each column and take the median of the results (middle). Temporal sampling follows a similar process for rows (bottom). Generations for each column or row can be executed in parallel.

[Figure 77]

[Figure 78]

[Figure 79]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Fixed ViewVarying Time

P5,T0

P5,T0

P5,T0

[Figure 80]

[Figure 81]

[Figure 82]

Varying ViewVarying Time

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

window of size N and then taking the pixel-wise median of the results. The generation of the j-th window is conditioned on N input frames at corresponding target cameras {Ic | c ∈ {ki mod K}ij=+jN} and 1 input frame at the target time It. We choose such conditional frames as they provide the most useful information. Taking the pixel-wise median of multiple windows aggregates information from more conditional frames and also helps reduce the variance. Results from multi-view sampling are largely multi-view consistent, but are not temporally consistent. Temporal sampling For each camera Pk

4DiM Ours GT

Figure 6. Qualitative comparison, disentangled control: The camera-time grid on the left shows the positions of three input images (gray cells, images visualized in top row) and output images (green) in three different target sampling settings. One frame from each setting (orange cell) is visualized in each row, comparing our model with 4DiM [73] and ground truth.

to sufficiently constrain 4D reconstruction. To do this, we first generate K novel views at t = 0, add these generated frames to the set of input frames, and then run the previously described alternating sampling strategy.

, we generate images Gi,· at all L times using a similar sliding window approach as multi-view sampling. The generation of the j-th window is conditioned on N input frames at corresponding target timestamps {It mod L}tj=+jN and 1 input frame at target camera Ik

i

Dense view sampling To further increase the coverage of our generated multi-view videos, for each timestep we condition on the K generated views and generate K′ more views using the nearest-anchoring strategy presented in Sec. 3.3. For our experiments, we set K = 13 and K′ = 128. We use MonST3R [87] to get camera parameters for unposed input videos. More details (e.g., sampled camera trajectories) are described in the supplement.

. Results from temporal sampling are largely temporally consistent, but are not multi-view consistent.

i

Alternating strategy To achieve both multi-view and temporal consistency, we alternate between multi-view sampling and temporal sampling with SDEdit [45] from the previous iteration and use 25 DDIM steps for sampling from random noise and a reduces number of steps when initializing from lower noise levels. We run three iterations of sampling: (1) multi-view sampling from random noise, (2) temporal sampling initialized from the previous iteration at noise level 16/25 (3) multi-view sampling initialized from the previous iteration at noise level 8/25. Multi-view videos generated with this strategy are sufficiently consistent for accurate 4D reconstruction, provided the input video has enough coverage of the scene (recall that the K cameras are picked from the input cameras P1:L).

#### 3.5. 4D Reconstruction

The alternating sampling method (Sec. 3.4) generates multiview videos that are sufficiently consistent to be used in existing 4D reconstruction pipelines. We build on 4DGS [74], which represents a dynamic 3D scene as a set of canonical-space 3D Gaussians that are moved by a deformation field parameterized with K-Planes [14]. See the supplement for details.

### 4. Experiments

Stationary videos For input videos with little to no camera movement, we must generate images from novel viewpoints

Separate Control over Camera and Time We first examine our model’s ability to separately control camera view-

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Fixed Viewpoint Varying Time

Varying Viewpoint Fixed Time

Varying Viewpoint Varying Time

Input

Method

PSNRSSIMLPIPS PSNRSSIMLPIPS PSNRSSIMLPIPS

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

CAT3D-1cond CAT3D-3cond

CAT3D-1cond CAT3D-3cond

4DiM[73] 19.77 0.540 0.195 18.81 0.428 0.219 17.28 0.378 0.256 Ours 21.97 0.683 0.121 21.68 0.588 0.105 19.73 0.533 0.155

- Table 2. Quantitative comparison, disentangled control: We compare with 4DiM [73] on the NSFF dataset [37], evaluating how well the time and viewpoint can be independently manipulated.

Method PSNR↑ SSIM↑ LPIPS ↓

CAT3D-1cond [17] 15.33 0.379 0.527 CAT3D-3cond [17] 20.19 0.568 0.258 Ours 20.79 0.576 0.160

- Table 3. Quantitative comparison, sparse-view “bullet-time”: We compare with CAT3D [17] on the NSFF dataset [37], evaluating the ability to reconstruct a consistent static 3D scene from input images containing scene motion.

Sparse-View Bullet-Time 3D Reconstruction On this task, we again evaluate on the NSFF dataset [37] with 3 input images. For each scene, we test on 4 different sets of input images randomly sampled from the captured sequence. We compare against two versions of CAT3D [17], with either all 3 input images (“CAT3D-3cond”) or only one input image at the target time (“CAT3D-1cond”). Qualitative and quantitative comparison results are presented in Fig. 7 and Table 3, respectively. CAT3D-3cond is unable to resolve inconsistencies in the inputs, resulting in blurry novel view renderings, especially in dynamic parts of the scene. CAT3D-1cond, which is given only one image at the target time, does not need to resolve any dynamic inconsistencies, but it is unable to leverage available information in other input images to determine the global scene scale and correct scene content. Our multi-view video diffusion model reconstructs the 3D scene at the target time most accurately.

4D Reconstruction We evaluate the task of dynamic scene reconstruction from monocular video on the DyCheck dataset [16], which contains video sequences with 200-500 frames of challenging real world scene motion. Following [72], we test on 5 scenes that have two synchronized static cameras for novel-view synthesis evaluation. In

- Table 4 and Fig. 9, we compare to the baseline 4D-GS [74] (without using our generated images) and state-of-the-art dynamic reconstruction methods Shape-of-Motion [72] and MoSca [33]. Our method greatly improves the reconstruction quality over the baseline 4D-GS representation that we used. Our results are roughly on par with those of Shape-ofMotion and MoSca, but those models use additional supervisory signals (e.g., depth maps, 2D tracks, user-provided foreground masks). We use no additional signals and supervise with only straightforward photometric losses.

NovelView

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Ours GT

Ours GT

- Figure 7. Qualitative comparison, sparse-view “bullet-time”

- 3D reconstruction: The three input images are shown on the top, where the first one is the target bullet-time frame.

Image

Independent multi-view

Independent temporal

Multi-view sampling Temporal sampling

Alternating sampling

Ground Truth

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- Figure 8. Qualitative comparison, sampling strategies: A comparison of different sampling strategies using space-time slices, where the vertical axis represents time and the horizontal axis shows a spatial slice of the image (red line). Our alternating sampling strategy best matches the ground truth motion.

point and scene dynamics. Given 3 input images, we generate three types of output sequences using our diffusion model: 1) fixed viewpoint and varying time, 2) varying viewpoint and fixed time, and 3) varying viewpoint and varying time. We evaluate on the NSFF dataset [37] (a subset of the Nvidia Dynamic Scenes Dataset [81]), which provides ground truth frames from 12 synchronized cameras in eight real-world dynamic scenes. For each scene, we test on 8 different sets of input images that are randomly sampled from the dataset. We compare against 4DiM [73], a recent method that trains a cascaded diffusion model to synthesize images under specified viewpoints and timestamps. Qualitative and quantitative results are shown in Fig. 6 and Table 2, respectively. We observe that 4DiM conflates control over camera and scene motion: even when the model is instructed to only change the camera viewpoint, dynamic objects still move. In contrast, our model exhibits decoupled control over camera and scene motion control, in addition to generating overall higher quality images.

4D-GS Shape-of-Motion MoSca Ours GT Training View

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

- Figure 9. Qualitative comparison, 4D reconstruction: We compare 4D reconstructions on the DyCheck dataset [16]. The rightmost column shows the input frame, at the same timestamp as the desired target image. Evaluation-excluded co-visibility masks are highlighted green. The visual improvement of our rendering over 4D-GS [74] (which our system leverages) demonstrates the value of our proposed multi-view video model. Renderings from Shape-of-Motion [72] and MoSca [33] were graciously provided by the authors.

Ablation Study The multi-view and temporal consistency of our samples is essential for high-quality 4D creation. We compare our alternating sampling strategy (Sec. 3.4) to four simpler options: 1) multi-view sampling only, 2) temporal sampling only, 3) independent multi-view sampling (i.e., without the sliding-window overlap), and 4) independent temporal sampling (i.e., without the slidingwindow overlap). We evaluate different strategies on the NSFF dataset [37] for generating multi-view videos. The results in Table 5 and Fig. 8 shows that our alternating sampling strategy improves sample quality. We also present an ablation study of our training datasets in the supplement.

Method mPSNR↑ mSSIM↑ mLPIPS ↓

4D-GS [74] 16.54 0.594 0.347 Shape-of-Motion [72] 16.72 0.630 0.450 Ours 17.39 0.607 0.341

MoSca [33]† 19.54 0.738 0.244 Ours† 18.24 0.666 0.227

- Table 4. Quantitative comparison, 4D reconstruction: Following prior work [16, 33, 72], we report co-visibility masked image metrics on the DyCheck dataset [16]. † indicates methods trained on images at half the original resolution.

Sampling strategy PSNR ↑ SSIM ↑ LPIPS ↓

Independent multi-view 20.27 0.525 0.136 Independent temporal 21.63 0.615 0.130 Multi-view sampling 22.34 0.609 0.217 Temporal sampling 23.36 0.681 0.145 Alternating sampling 22.15 0.633 0.108

- Table 5. Quantitative comparison, sampling strategies: We compare our sampling strategy to four simpler variants by comparing to ground-truth images from the NSFF dataset [37].

### 5. Discussion

We present CAT4D, an approach for creating 4D scenes from captured or generated monocular videos. Our multiview video diffusion model transforms monocular inputs into consistent multi-view videos, enabling reconstruction as deformable 3D Gaussians.

CAT4D has several limitations: the diffusion model struggles with temporal extrapolation beyond input frames and cannot fully disentangle camera viewpoint from temporal progression, especially in challenging cases where dynamic objects become occluded. The alternating sampling strategy we propose is effective at increasing the number of frames we can generate, but training larger-scale multi-view video models that can accomplish this directly is an exciting future area. Additionally, while our generated 4D scenes appear plausible from novel viewpoints, the recovered 3D motion fields may not be physically accurate. Incorporating supervision signals like depth or motion estimates could improve the quality of our results but would reduce the applicability of our method to dense video captures.

- 4D Creation In addition to reconstructing of real, captured dynamic 3D scenes, CAT4D can also be used to create 4D scenes of generated content, by first using a text-to-video or image-to-video model [1, 2, 10, 51, 60, 67] to create a monocular video sequence. Unlike real captures, these generated videos sometimes contain inconsistent scene elements that defy physical 3D constraints and typically lack the camera motion that most state-of-the-art reconstruction methods require. Fig. 2 shows a collection of these results, but we strongly recommend viewing the supplementary video instead.

Acknowledgments We would like to thank Arthur Brussee, Philipp Henzler, Daniel Watson, Jiahui Lei, Hang Gao, Qianqian Wang, Stan Szymanowicz, Jiapeng Tang, Hadi Alzayer and Angjoo Kanazawa for their valuable contributions. We also extend our gratitude to Shlomi Fruchter, Kevin Murphy, Mohammad Babaeizadeh, Han Zhang and Amir Hertz for training the base text-to-image latent diffusion model.

### References

- [1] Kling AI. Kling. https://klingai.com/, . 8
- [2] Luma AI. Dream Machine. https://lumalabs.ai/ dream-machine/, . 8
- [3] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil Kim. HyperReel: High-Fidelity 6-DoF Video with RayConditioned Sampling. CVPR, 2023. 3
- [4] Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. TC4D: Trajectory-Conditioned Text-to-4D Generation. ECCV,

2024. 3

- [5] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling. CVPR, 2024. 3
- [6] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, David B. Lindell, and Sergey Tulyakov. VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control. arXiv:2407.12781, 2024. 4
- [7] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Yuanzhen Li, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A Space-Time Diffusion Model for Video Generation. arXiv:2401.12945,

2024. 3, 5, 2

- [8] Mojtaba Bemana, Karol Myszkowski, Hans-Peter Seidel, and Tobias Ritschel. X-fields: Implicit neural view-, lightand time-image interpolation. SIGGRAPH, 2020. 1
- [9] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv:2311.15127, 2023. 3
- [10] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 8

- [11] Ang Cao and Justin Johnson. HexPlane: A Fast Representation for Dynamic Scenes. CVPR, 2023. 1, 3
- [12] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana

- Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A Universe of Annotated 3D Objects. CVPR, 2023. 3, 5, 1, 2
- [13] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement. ICCV, 2023. 3
- [14] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. KPlanes: Explicit Radiance Fields in Space, Time, and Appearance. CVPR, 2023. 1, 3, 6
- [15] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic View Synthesis from Dynamic Monocular Video. ICCV, 2021. 3
- [16] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. NeurIPS, 2022. 7, 8, 1
- [17] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. CAT3D: Create Anything in 3D with Multi-View Diffusion Models. NeurIPS,

2024. 1, 3, 4, 5, 7, 2

- [18] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives. CVPR, 2024. 5
- [19] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. CVPR, 2022. 3, 5, 1, 2
- [20] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. arXiv:2307.04725, 2023. 4
- [21] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling Camera Control for Text-to-Video Generation. arXiv:2404.02101, 2024. 4
- [22] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large Reconstruction Model for Single Image to 3D. ICLR, 2024. 1
- [23] Chen Hou, Guoqiang Wei, Yan Zeng, and Zhibo Chen. Training-free Camera Control for Video Generation. arXiv:2406.10126, 2024. 4
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022. 3
- [25] Yifei Huang, Guo Chen, Jilan Xu, Mingfang Zhang, Lijin Yang, Baoqi Pei, Hongjie Zhang, Lu Dong, Yali Wang, Limin Wang, et al. EgoExoLearn: A Dataset for Bridging Asynchronous Ego-and Exo-centric View of Procedural Activities in Real World. CVPR, 2024. 5

- [26] Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video. ICLR, 2024. 3
- [27] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker: It is Better to Track Together. ECCV, 2024. 3
- [28] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH, 2023. 1, 3, 5
- [29] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment Anything. ICCV, 2023. 3
- [30] Agelos Kratimenos, Jiahui Lei, and Kostas Daniilidis. DynMF: Neural Motion Factorization for Real-time Dynamic View Synthesis with 3D Gaussian Splatting. ECCV,

2024. 3

- [31] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas Guibas, and Gordon Wetzstein. Collaborative Video Diffusion: Consistent Multi-video Generation with Camera Control. arXiv:2405.17414, 2024. 4
- [32] Yao-Chih Lee, Zhoutong Zhang, Kevin Blackburn-Matzen, Simon Niklaus, Jianming Zhang, Jia-Bin Huang, and Feng Liu. Fast view synthesis of casual videos with soup-ofplanes. ECCV, 2024. 3
- [33] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds. arXiv:2405.17421,

2024. 3, 7, 8

- [34] Bing Li, Cheng Zheng, Wenxuan Zhu, Jinjie Mai, Biao Zhang, Peter Wonka, and Bernard Ghanem. Vivid-zoo: Multi-view video generation with diffusion model. arXiv preprint arXiv:2406.08659, 2024. 3
- [35] Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. Neural 3D Video Synthesis from Multi-view Video. CVPR, 2022. 1, 3
- [36] Zhengqi Li, Tali Dekel, Forrester Cole, Richard Tucker, Noah Snavely, Ce Liu, and William T Freeman. Learning the Depths of Moving People by Watching Frozen People. CVPR, 2019. 3, 5, 1, 2
- [37] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes. CVPR, 2021. 3, 7, 8, 2
- [38] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. DynIBaR: Neural Dynamic Image-Based Rendering. CVPR, 2023. 3
- [39] Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models. arXiv:2405.16645, 2024. 3, 5, 1
- [40] Huan Ling, Seung Wook Kim, Antonio Torralba, Sanja Fidler, and Karsten Kreis. Align Your Gaussians: Text-to-4D with Dynamic 3D Gaussians and Composed Diffusion Models. CVPR, 2024. 3

- [41] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot One Image to 3D Object. ICCV, 2023. 3
- [42] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. SyncDreamer: Generating Multiview-consistent Images from a Single-view Image. ICLR, 2024. 3
- [43] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis. 3DV, 2024. 1, 3
- [44] Willi Menapace, Aliaksandr Siarohin, Ivan Skorokhodov, Ekaterina Deyneka, Tsai-Shien Chen, Anil Kag, Yuwei Fang, Aleksei Stoliar, Elisa Ricci, Jian Ren, et al. Snap Video: Scaled Spatiotemporal Transformers for Text-toVideo Synthesis. CVPR, 2024. 3
- [45] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. ICLR, 2022. 6
- [46] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV, 2020. 1, 3
- [47] Zijie Pan, Zeyu Yang, Xiatian Zhu, and Li Zhang. Efficient4D: Fast Dynamic 3D Object Generation from a Singleview Video. arXiv:2401.08742, 2024. 3
- [48] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable Neural Radiance Fields. ICCV, 2021. 3
- [49] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo MartinBrualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. SIGGRAPH Asia, 2021. 3
- [50] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal Monocular Metric Depth Estimation. CVPR, 2024. 3
- [51] Pika. Pika 1.5. https://pika.art/. 8
- [52] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. ICLR,

2023. 3

- [53] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. CVPR, 2021. 1
- [54] Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, et al. Infinite photorealistic worlds using procedural generation. CVPR, 2023. 5
- [55] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction. ICCV, 2021. 3, 5, 1, 2
- [56] Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. DreamGaussian4D: Generative 4D Gaussian Splatting. arXiv:2312.17142, 2023. 3

- [57] Jiawei Ren, Kevin Xie, Ashkan Mirzaei, Hanxue Liang, Xiaohui Zeng, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, et al. L4GM: Large 4D Gaussian Reconstruction Model. arXiv:2406.10324, 2024. 3
- [58] Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, et al. ZeroNVS: Zero-shot 360degree View Synthesis from a Single Real Image. CVPR,

2024. 1, 3

- [59] Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao. Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities. CVPR, 2022. 5
- [60] Abhishek Sharma, Adams Yu, Ali Razavi, Andeep Toor, Andrew Pierson, Ankush Gupta, Austin Waters, A¨aron van den Oord, Daniel Tanis, Dumitru Erhan, Eric Lau, Eleni Shaw, Gabe Barth-Maron, Greg Shaw, Han Zhang, Henna Nandwani, Hernan Moraldo, Hyunjik Kim, Irina Blok, Jakob Bauer, Jeff Donahue, Junyoung Chung, Kory Mathewson, Kurtis David, Lasse Espeholt, Marc van Zee, Matt McGill, Medhini Narasimhan, Miaosen Wang, Mikołaj Bi´nkowski, Mohammad Babaeizadeh, Mohammad Taghi Saffar, Nando de Freitas, Nick Pezzotti, Pieter-Jan Kindermans, Poorva Rane, Rachel Hornung, Robert Riachi, Ruben Villegas, Rui Qian, Sander Dieleman, Serena Zhang, Serkan Cabi, Shixin Luo, Shlomi Fruchter, Signe Nørly, Srivatsan Srinivasan, Tobias Pfaff, Tom Hume, Vikas Verma, Weizhe Hua, William Zhu, Xinchen Yan, Xinyu Wang, Yelin Kim, Yuqing Du, and Yutian Chen. Veo. 2024. 8
- [61] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. MVDream: Multi-view Diffusion for 3D Generation. arXiv:2308.16512, 2023. 3
- [62] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-A-Video: Text-to-Video Generation without Text-Video Data. ICLR, 2023. 3
- [63] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv:2301.11280, 2023. 3
- [64] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: exploring photo collections in 3D. SIGGRAPH,

2006. 1

- [65] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 1
- [66] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion. arXiv:2411.04928, 2024. 3, 4
- [67] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 8
- [68] Alex Trevithick, Roni Paiss, Philipp Henzler, Dor Verbin, Rundi Wu, Hadi Alzayer, Ruiqi Gao, Ben Poole, Jonathan T Barron, Aleksander Holynski, et al. Simvs: Simulating world inconsistencies for robust view synthesis. arXiv preprint arXiv:2412.07696, 2024. 5, 2

- [69] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis. ECCV,

2024. 4

- [70] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, et al. Attention is all you need. NeurIPS, 2017. 4
- [71] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Jingyi Yu, and Lan Xu. Fourier PlenOctrees for Dynamic Radiance Field Rendering in Real-time. CVPR, 2022. 3
- [72] Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of Motion: 4D Reconstruction from a Single Video. arXiv:2407.13764, 2024. 3, 7, 8
- [73] Daniel Watson, Saurabh Saxena, Lala Li, Andrea Tagliasacchi, and David J Fleet. Controlling Space and Time with Diffusion Models . arXiv:2407.07860, 2024. 4, 6, 7
- [74] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. CVPR, 2024. 1, 3, 6, 7, 8
- [75] Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. ReconFusion: 3D Reconstruction with Diffusion Priors. CVPR, 2024. 1, 3
- [76] Jinyu Yang, Mingqi Gao, Zhe Li, Shang Gao, Fangjing Wang, and Feng Zheng. Track Anything: Segment Anything Meets Videos. arXiv:2304.11968, 2023. 3
- [77] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. CVPR, 2024. 3
- [78] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion. arXiv:2402.03162, 2024. 4
- [79] Zeyu Yang, Zijie Pan, Chun Gu, and Li Zhang. Diffusion2: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models. arXiv:2404.02148, 2024. 3
- [80] Yuyang Yin, Dejia Xu, Zhangyang Wang, Yao Zhao, and Yunchao Wei. 4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency. arXiv:2312.17225, 2023. 3
- [81] Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera. CVPR, 2020. 3, 7, 2
- [82] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Laszlo A Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4Real: Towards Photorealistic 4D Scene Generation via Video Diffusion Models. arXiv:2406.07472, 2024. 3
- [83] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video

- diffusion models for high-fidelity novel view synthesis. arXiv:2409.02048, 2024. 4
- [84] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. MVImgNet: A Large-scale Dataset of Multi-view Images. CVPR, 2023. 3, 5, 1, 2
- [85] Yifei Zeng, Yanqin Jiang, Siyu Zhu, Yuanxun Lu, Youtian Lin, Hao Zhu, Weiming Hu, Xun Cao, and Yao Yao. STAG4D: Spatial-Temporal Anchored Generative 4D Gaussians . ECCV, 2024. 3
- [86] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E. Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. ReCapture: Generative Video Camera Controls for User-Provided Videos using Masked Video Fine-Tuning. arXiv:2411.05003, 2024. 4
- [87] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion. arXiv:2410.03825,

2024. 6, 1

- [88] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. CVPR, 2018. 5
- [89] Xiaoming Zhao, R Alex Colburn, Fangchang Ma, Miguel Angel´ Bautista, Joshua M Susskind, and Alex Schwing. Pseudo-Generalized Dynamic View Synthesis from a Video. ICLR, 2024. 3
- [90] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating One Image to 4D Dynamic Scene. arXiv:2311.14603, 2023. 3
- [91] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee, and Lijuan Wang. Genxd: Generating any 3d and 4d scenes. arXiv preprint arXiv:2411.02319, 2024. 4
- [92] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking. ICCV,

2023. 5

- [93] Yufeng Zheng, Xueting Li, Koki Nagano, Sifei Liu, Otmar Hilliges, and Shalini De Mello. A Unified Approach for Text- and Image-guided 4D Scene Generation. CVPR, 2024. 3
- [94] Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu, Yiyuan Yang, Andrew Markham, and Niki Trigoni. Dynpoint: Dynamic neural point for view synthesis. NeurIPS,

2024. 3

- [95] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo Magnification: Learning View Synthesis using Multiplane Images. SIGGRAPH, 2018. 3, 5, 1, 2

## CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models Supplementary Material

### A. Method Details

Diffusion Model We initialize our model from CAT3D [17], with the additional MLP layers in the new timestamp embedding randomly initialized. All timestamps t ∈ Tcond ∪ Ttgt (of each set of M + N frames) are normalized within range [0,1] and are relative to the first timestamp Tcond0 . We fine-tune the full latent diffusion model (i.e., the denoising U-Net) with M = 3 input views and N = 13 target views for 2.0M iterations with a batch size of 128 and a learning rate of 5 × 10−5. For the 4D reconstruction application, in order to condition on more input frames, we further fine-tune the model with M = 9 input views and N = 8 target views for 20K iterations.

Sampling For all of our experiments, we use DDIM [65] with 25 sampling steps and classifier-free guidance weights s1 = 3.0,s2 = 4.5. Our alternating sampling strategy takes about 1 minute to generate all K′ = 128 views for each timestamp, when executed in parallel on 16 A100 GPUs. We note that CAT3D originally use 50 DDIM steps, yet for our model we found that 25 steps work just as well—and using fewer steps reduces the runtime of our sampling strategy.

Camera Trajectory Selection The choice of camera trajectories where we generate novel views has a large impact on the quality of 4D creation. In principle, the camera trajectories should cover the viewpoints where we want to render the scene after reconstruction. We design the novelview camera trajectory based on the camera trajectory of the input video:

- • For input videos with sufficient view coverage (e.g., videos from DyCheck [16] whose cameras are centered around a focus point), we simply sample views on the input camera trajectory.
- • For input videos with a forward-moving camera trajectory, we sample novel views from a spiral path around the input camera trajectory.
- • For input videos with little or no camera movement, we sample novel views from either a spiral path that moves into and out of the scene or a orbit path that spins around the central object. For each example, we run both, and select the one which is most appropriate for the given scene.

See Fig. 10 for an illustration of different types of camera trajectories.

Sparse-View Bullet-Time 3D Reconstruction The conditional times Tcond should in principal be the actual timestamps of the input frames, but they may not be known for unstructured in-the-wild datasets. We found that in practice

the model works well if we just set the timestamps for the bullet-time frame as 0 and other frames as 1.

4D Reconstruction We build our reconstruction pipeline on top of 4D-GS [74] with several extensions. We use a combination of L1, DSSIM and LPIPS for the photometric reconstruction loss, with weighting factors 0.8, 0.2 and 0.4 respectively, and keep all regularization terms from [74] as is. We set the 3D Gaussians’ densification threshold (magnitude of view-space position gradients) to 0.0004, and use a batch size of 4 (images). We initialize the 3D Gaussians with points from SfM [64] or MonST3R [87]. When SfM or MonST3R points are not available, e.g., input videos of a static viewpoint, we use uniformly random points for initialization. We first optimize only the canonical-space 3D Gaussians with all generated images at t = 0 for 2000 iterations, then jointly optimize both the 3D Gaussians and the deformation field with images at all timestamps for 18000 iterations. After the first 2000 iterations, we linearly anneal the multiplier of reconstruction loss for our generated images from 1.0 to 0.5 while keeping the multiplier for real input images fixed to 1.0. The optimization takes about 25 minutes on a single A100 GPU.

### B. Datasets Details

For Objaverse [12], we use only the animated assets filtered by [39] (around 42k in total). We render each asset under 4 different lighting conditions ( randomly sampled environment maps). For each lighting condition, we render synchronized videos of 8 frames at 8 evenly spaced viewpoints on the 360◦ orbit path. For Kubric [19], we randomly generate 4k scenes using their generator. For each scene, we render synchronized videos of 8 frames at 8 viewpoints evenly spaced on a smooth camera path that is randomly sampled on the upper hemisphere. When drawing samples from these two synthetic 4D datasets, we sample the input and target views according to different combinations in Table 1 with equal probability.

For data samples from all multi-view image datasets [36, 55, 84, 95], we set all timestamps Tcond ∪ Ttgt to zero.

For our video dataset, we filter it to contain only videos of at static viewpoint. We perform this filtering by checking if the four corner patches (size 10 × 10) of each video are nearly constant over time. Concretely, we compute for each corner patch the L2 distance between consecutive frames (averaged over time), and then check if the maximum of the four is smaller than 0.05. While this simple strategy sometimes yields false positives, it’s sufficiently effective and can be run on the fly. For samples from this dataset, we ran-

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

a) videos with sufficient view coverage (DyCheck) b) videos with a forward-moving camera path

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

c) videos of a static viewpoint, forward d) videos of a static viewpoint, orbit

- Figure 10. Camera trajectories (where we generate novel views) for different types of input videos. Within each panel, we show the trajectories from two different viewpoints. The input views are colored red, and the anchoring sample views are colored blue with the remaining sample views are colored by their index. For videos with sufficient view coverage (a), we only generate anchor views picked from the input camera trajectory.

{Icond0 }∪{Vki

}Mi=1−1 and Pcond ← {Pcond0 }∪{P˜ki

}Mi=1−1. We obtain around 160k sequences in total with this augmentation.

domly shuffle the order of frames and set all camera parameters Pcond∪Ptgt to be the same with a central principal point and a random focal length sampled from [0.8·512,1.2·512].

i

i

We use Lumiere [7] to augment the CO3D dataset [55] (4-th row in Table 1) following SimVS [68]. For each sampled sequence (Icond,Pcond,Tcond,Itgt,Ptgt,Ttgt) from CO3D, we animate each of the M input images (except the first one) using Lumiere [7], resulting in M − 1 videos {VLi}Mi=1−1 of length L. Then we randomly sample one frame (index ki) from each video, and treat them as pseudo ground truth of original input images at another timestamps, i.e. Icond ← {Icond0 } ∪ {Vki

We mix all the datasets (Objaverse [12], Kubric [19], Re10K [95], MVImgNet [84], CO3D [55], MQ4K [36], static-view video data, augmented CO3D and augmented static-view video data) with weights 2.5, 2.5, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0 and 1.0, respectively.

### C. Baselines Details

}Mi=1−1 and Tcond ← {Tcond0 } ∪ { k

For the evaluation of sparse-view bullet-time 3D reconstruction, we run CAT3D baselines with one or three input images. For CAT3D-3cond, we use the same camera trajectory as ours (Fig. 10 (c)) for generating novel views. For CAT3D-1cond, we use their default camera trajectory (a forward-facing spiral path similar to Fig. 10 (c)) for generating novel views, and manually adjust the global scene scale such that it roughly matches the actual scene scale of the dataset.

i

L−1}Mi=1−1. We obtain around 24k sequences in total with this augmentation.

i

We use CAT3D [17] to augment our static-view video dataset as follows (5-th row in Table 1). For each sampled sequence (Icond,Pcond,Tcond,Itgt,Ptgt,Ttgt) from the video dataset, we use CAT3D to generate 7 novel views for each of the M input images (except the first one), resulting in M − 1 image sets {VLi}Mi=1−1(L = 7) at viewpoints {P˜i}Mi=1−1. Then we randomly sample one frame (index ki) from each image set, and treat them as pseudo ground truth of original input images at another viewpoints, i.e. Icond ←

For the ablation study of different sampling strategies, we evaluate the quality of the generated multi-view videos on the NSFF dataset [37]. As in the prior work [37, 81], we

Input Synthetic only No augmentation All datasets

Fig. 11). For the model trained only on synthetic 4D datasets, it already gives surprisingly good control over camera and time, but the generated scene motions are often unnatural and the generated novel views usually look worse. This is likely a generalization issue. For the model trained without the two augmented datasets, its main failure mode is “fixed viewpoint, varying time” — in many cases, the camera still moves even when the model is instructed to only change the scene dynamics. This is potentially caused by our imperfect filtering of the video data, where some videos of non-static viewpoints are treated as static-view videos.

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

- Figure 11. A comparison of models trained with different datasets on in-the-wild input images. The three input images are shown on the left-most column. Top: space-time slices of generated videos of “fixed viewpoint, varying time”. Pixels of static background should be straight vertical lines on the slices and pixels of dynamic object should be smooth curves on the slices. Bottom: one frame of generated videos of “varying viewpoint, fixed time”.

simulate a moving monocular camera by extracting images from each of the 12 camera viewpoints at different timestamps (24 in total) and compare the generated multi-view videos at all 12 viewpoints to the ground truth.

### D. Ablation Study of Training Data

Fixed Viewpoint Varying Time

Varying Viewpoint Fixed Time

Varying Viewpoint Varying Time

Training Data

PSNRSSIMLPIPS PSNRSSIMLPIPS PSNRSSIMLPIPS

Synthetic only 22.19 0.745 0.123 21.41 0.547 0.123 19.50 0.523 0.173 No augmentation 20.84 0.596 0.135 22.03 0.602 0.104 19.41 0.519 0.160 All datasets 22.49 0.749 0.110 21.86 0.599 0.105 19.74 0.546 0.152

Table 6. A ablation study of training data, evaluated on the NSFF dataset [37]. All datasets: using all of our training datasets. No augmentation: dropping the two augmented datasets (CO3D augmented with Lumiere and static-view video data augmented with CAT3D). Synthetic only: dropping all real-world datasets and using only synthetic 4D data (Kubric and Objaverse).

We also perform an ablation study of our training datasets. We train our model with 1) all datasets listed in Table 1, 2) all datasets except the two augmented datasets (CO3D augmented with Lumiere and static-view video data augmented with CAT3D), 3) synthetic 4D datasets only (Kubric and Objaverse). For all three versions, we train the model for 60k iterations, and evaluate its ability for separate camera and time control on the NSFF dataset [37].

The quantitative results are presented in Table 6. While the numbers themselves do not show a large gap (as most of the evaluated pixels are static background), we observe more clear visual differences on in-the-wild data (see

