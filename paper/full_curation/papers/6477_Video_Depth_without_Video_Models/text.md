# arXiv:2411.19189v2[cs.CV]17Mar2025

## Video Depth without Video Models

Bingxin Ke1 Dominik Narnhofer1 Shengyu Huang1 Lei Ke2 Torben Peters1 Katerina Fragkiadaki2 Anton Obukhov1 Konrad Schindler1 1ETH Zurich 2Carnegie Mellon University

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

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Figure 1. The RollingDepth model takes an unconstrained video and reconstructs a corresponding depth video. Unlike methods that rely on video diffusion models, it extends a single-image monodepth estimator such that it can process short snippets. To account for temporal context, snippets with varying frame rates are sampled from the video, processed, and reassembled through a global alignment algorithm to obtain long, temporally coherent depth videos. Depth is colour-coded near far.

[Figure 19]

### Abstract

Video depth estimation lifts monocular video clips to 3D by inferring dense depth at every frame. Recent advances in single-image depth estimation, brought about by the rise of large foundation models and the use of synthetic training data, have fueled a renewed interest in video depth. However, naively applying a single-image depth estimator to every frame of a video disregards temporal continuity, which not only leads to flickering but may also break when camera motion causes sudden changes in depth range. An obvious and principled solution would be to build on top of video foundation models, but these come with their own limitations; including expensive training and inference, imperfect 3D consistency, and stitching routines for the fixed-length (short) outputs. We take a step back and demonstrate how to turn a single-image latent diffusion model (LDM) into a state-of-the-art video depth estimator. Our model, which we call RollingDepth, has two main ingredients: (i) a multiframe depth estimator that is derived from a single-image LDM and maps very short video snippets (typically frame

triplets) to depth snippets. (ii) a robust, optimization-based registration algorithm that optimally assembles depth snippets sampled at various different frame rates back into a consistent video. RollingDepth is able to efficiently handle long videos with hundreds of frames and delivers more accurate depth videos than both dedicated video depth estimators and high-performing single-frame models. Project page: rollingdepth.github.io.

### 1. Introduction

Inferring 3D scene structure from a video stream is a fundamental capability of a vision system. Besides its scientific relevance as an elementary building block of machine perception, it has a broad range of applications, including mobile robotics and autonomous driving, augmented reality, media production, and content creation.

Traditionally, a video would be converted into a 3D world model by recovering the camera trajectory with structure-from-motion (SfM) techniques [17, 56], then ap-

plying multi-view reconstruction based on either stereo triangulation [15, 74] or, more recently, inverse volume rendering [26, 44]. That approach has the attractive property that it delivers a full 3D scene model in a common coordinate frame. The price to pay is that it is only feasible under narrowly defined conditions: the camera motion must be just right, and the scene must have a static background with cooperative texture and lighting conditions. In practice, both SfM and multi-view reconstruction fail more often than not when applied to in-the-wild videos.

This is where video depth comes in. Not all applications require full-scale 3D reconstruction, and it turns out that information about the scene structure can be recovered much more reliably if one aims for a more modest goal: augment every video frame with a dense 2.5D depth map, in such a way that those depth maps are consistent through time. The past years have witnessed tremendous progress in depth estimation from a single image, sidestepping camera pose estimation (and often also calibration of the focal length) [5, 25, 52, 72]. A common thread is that recent methods build on foundation models trained on internetscale data, such as DINOv2 [46] or StableDiffusion [54], and fine-tune them for depth estimation, often using predominantly synthetic RGB+depth image pairs that can be generated in large quantities and have accurate depth. The underlying, rich visual priors afford these depth estimators excellent zero-shot generalization across scene types, imaging, and lighting conditions.

In general, applying a single-image depth estimator to a video frame-by-frame does not yield satisfactory results, but leads to depth flicker and drift. These artifacts are caused by multiple factors. Most obviously, neither the model training nor the inference procedure have any notion of temporal coherence between adjacent frames. Moreover, monodepth estimation requires scene understanding, which may also suffer from the lack of temporal context (e.g., when a partially visible object only becomes recognizable after zooming out). What is more, in a video the depth range between nearby and distant scene parts may change all of a sudden (e.g., when a foreground object enters the viewfield, or when the camera pans to a window), making consistent monodepth estimation difficult.

Some authors [24, 58] have explored the idea of repurposing generative video models like Stable Video Diffusion [4] for depth prediction. These methods enable information exchange along the time axis and acquire a strong flow and motion prior during training, hence they achieve excellent local consistency through time. On the downside, video LDMs – besides being computationally demanding – are trained for fixed, short sequence lengths and cannot be applied directly to uncurated footage of varying lengths. To be practically useful, the diffusion routine must be wrapped into a partitioning scheme that splits the input video for pro-

cessing and stitches the depth estimates back together, often resulting in low-frequency flickering and gradual drift. We also find that current LDM-based video depth models tend to be less accurate on distant scene parts.

Rather than design more refined video LDMs, which require huge resources for training, we take a step back and re-examine how far one can take video depth estimation with augmented single-image LDMs. We design a set of measures that, taken together, extend a per-image monodepth framework like Marigold [25] in a way that enables it to handle video input. Importantly, these measures greatly improve local and global consistency across time while maintaining a constant memory footprint such that one can process long sequences. Specifically, we employ a “rolling” inference with a sliding window of a few frames (typically three, but other numbers are possible). Those snippets are sampled from the video with varying spacing, i.e., they can be immediately adjacent but also dilated along the timeline to cover long-range context. They are then fed into a multi-frame LDM fine-tuned from a single-frame model, with a modified cross-frame self-attention mechanism to enable information exchange. To reassemble the snippets, we propose a robust optimization-based global coalignment, followed by averaging the aligned frames. Optionally, the resulting video can be degraded with moderate random noise and denoised again with the same per-snippet LDM to further refine spatial details.

To summarize, our approach estimates accurate and temporally consistent video depth without resorting to cumbersome video diffusion models. To that end, we contribute:

- 1. an LDM for monocular depth estimation in video snippets of a few frames, adapted from the Marigold [25] single-frame model but able to capture temporal patterns across frames via self-attention;
- 2. a rolling inference scheme that operates on snippets with multiple different (temporal) resolutions and enables efficient propagation of contextual information through video sequences of arbitrary length (up to minutes);
- 3. a global alignment procedure, based on robust optimization, to recompose the snippets into a depth video whose depth values remain consistent over long time periods;
- 4. an optional refinement of the final output with another round of multi-frame diffusion, where the same LDM is applied starting from a moderately degraded video.

### 2. Related Work

#### 2.1. Monocular Depth Estimation

Monocular depth estimation is a dense regression task. The pioneering work by Eigen et al. [12] showed that metric depth values can be recovered from single sensors. Successive advancements include including various parameterizations (ordinals, bins, planar guidance maps, piecewise pla-

narity, CRFs, etc.) [2, 13, 31, 33, 39, 45, 48, 79, 85], switching CNN backbones to vision transformers [1, 3, 34, 69], considering camera intrinsics [20, 23, 49, 50, 78], and patch-wise processing [5, 36, 37]. To handle “in-the-wild” settings, extensive internet photo collections are used for training [32, 77]. MiDaS [52] improves the generality by training on a mixture of multiple datasets. Depth Anything [71, 72] takes data scaling to the next level by relying on DINOv2 [46], a foundational model trained on 142M images in a self-supervised manner, and subsequently training with 62M pseudo-labels, 1M real depth annotations, and 0.5M synthetic ones. Recent trends leverage generative models, particularly diffusion models [22, 59], for depth estimation [11, 55, 84, 84]. Marigold [25] proposed to finetune Stable Diffusion [54], a generative text-to-image latent diffusion model (LDM) trained with LAION-5B [57], towards affine-invariant depth using 74k samples. This approach has been improved in many aspects including fewer steps [16, 18, 21, 67], finer details [81], and more modalities [14, 21].

#### 2.2. Video Depth Estimation

Video depth estimation calls for dedicated mechanisms to ensure smoothness of adjacent frames, and correct handling of varying depth range. Existing approaches can be grouped into three main categories: test-time optimization, feed-forward prediction, and diffusion-based. Test-time optimization methods [7, 29, 43, 82] often rely on camera poses or optical flow and perform optimization for each new video during inference. While these methods can produce depth estimates that are temporally consistent, their dependence on camera poses and long processing time hamper their application to open-world video scenarios. Feedforward prediction methods estimate depth sequences directly from input videos [35, 61, 63, 75, 76, 80]. For example, DeepV2D [61] integrates camera motion estimation with depth prediction, MAMO [75] adopts memory attention mechanisms, and NVDS [64, 65] introduces a stabilization network as a post-processing module. However, the generalization of these methods to in-the-wild videos is often constrained by the limited diversity of training data and model capacity.

Very recently, concurrent with our work, several authors have investigated the use of video diffusion models, in particular SVD [4], for video depth. ChronoDepth [58] DepthCrafter [24] and DepthAnyVideo [70] all modify video diffusion for conditional generative depth prediction. From the underlying video diffusion model they inherit high training and inference costs, and a restriction to short video clips of at most ≈100 frames. In contrast, in RollingDepth we explore how to turn an image diffusion model into a temporally consistent depth estimator, which can handle long videos of 1000 frames or more.

#### 2.3. Image Diffusion Models for Video Tasks

Image diffusion models have been employed in various video inverse problems, such as video generation, inpainting, and super-resolution [10, 30, 86]. A large amount of work [51, 73, 83] focusses on video editing, either by fine-tuning text-to-image diffusion models on video data [40, 66] or through training-free approaches using cross-frame attention and latent fusion [6, 27]. However, these works [6, 40, 66] predominantly address video-tovideo translation tasks, where both the input and output reside in RGB space. In contrast, our approach leverages image diffusion priors to generate consistent depth videos, with the additional challenge to accommodate large variations of the depth range, as the near and far planes change – often suddenly – due to camera and object motion. Implementation tricks when using single-image models, like fixing the initial noise or blending consecutive latent representations, can somewhat mitigate the lack of knowledge w.r.t. temporal coherence, but do not solve it [25].

### 3. Method

Let x ∈ RN

F×3×H×W be an RGB video of length NF, the goal of a monocular video depth estimator is to predict a depth video d ∈ RN

F×H×W. All frames in that depth video should share a common depth scale and shift, i.e., depth values should not drift unless the associated pixel moves relative to the camera. In the following, we present our RollingDepth framework for predicting d from x. The proposed approach is based on a per-snippet LDM, test-time depth co-alignment, and an optional refinement of the resulting video, as illustrated in Fig. 2.

#### 3.1. Marigold Monocular Depth Recap

Several recent methods [11, 25, 55], including our base model Marigold [25], cast monocular depth estimation as conditional image generation, where a pre-trained LDM is retargeted to generate the depth map given the input image. To that end, the model progressively adds noise to depth samples di and learns to reverse that degradation, to approximate the conditional distribution p(di|xi).

In detail, the model is trained to predict the added noise ϵ at each step by minimizing the objective

0,xi)∼Pdi,xi,t∼U,ϵ∼N ϵ − ϵθ(dit,xi,t) 2 .

L(θ) = E(di

At inference time the model starts from the input xi and pure Gaussian noise diT ∼ N(0,I), and gradually maps the latter to a depth map di0 by iteratively applying the learned denoising step. For computational efficiency, the denoising process operates a low-dimensional latent space Z, with an auto-encoder to map images to latent embeddings, and depth maps back to image space [54].

Step 1: Initial Prediction

Step 2: Co-Alignment

Snippet Inference

Video Sequence

Optimization

|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|

Input: RGB

|a|a|a| | | | |
|---|---|---|---|---|---|---|
| |b|b|b| | | |
|…| | | | | | |
|k| | |k| | |k|

× 𝑠Ƹ𝑎 + 𝑡Ƹ𝑎 × 𝑠Ƹ𝑏 + 𝑡Ƹ𝑏

[Figure 20]

|a|a|a| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| |b|b|b| | | | | | | | |
| | |c|c|c| | | | | | | |
| | | |d|d|d| | | | | | |
| | | | | | | | | | | | |
| | | | | | | | |i|i|i| |
|Dilation=3 Dilation=3| | | | | | | | |j|j|j|
|k| | |k| | |k| | | | | |
| |l| | |l| | |l| | | | |
| | |m| | |m| | |m| | | |
| | | |n| | |n| | |n| | |
| | | | |o| | |o| | |o| |
| | | | | |p| | |p| | |p|

|𝑠𝑎|𝑡𝑎|
|---|---|
|𝑠𝑏|𝑡𝑏|
|𝑠𝑐|𝑡𝑐|
|𝑠𝑑|𝑡𝑑|
|…| |
|𝑠𝑖|𝑡𝑖|
|𝑠𝑗|𝑡𝑗|
|𝑠𝑘|𝑡𝑘|
|𝑠𝑙|𝑡𝑙|
|𝑠𝑚|𝑡𝑚|
|𝑠𝑛|𝑡𝑛|
|𝑠𝑜|𝑡𝑜|
|𝑠𝑝|𝑡𝑝|

      

Average

1

4

…

…

7

× 𝑠Ƹ𝑑 + 𝑡Ƹ𝑑

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

…

…

[Figure 27]

…

[Figure 28]

|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|

[Figure 29]

Self-Attention

Output

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Co-Aligned Depth Sequence

[Figure 34]

     

k

k k

(Optional) Step 3: Refinement

Refined Depth Sequence

Output: Depth

|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|

Output

Scale Shift

is ith frame), we construct NT overlapping snippets using a dilated rolling kernel with varying dilation rates, and perform 1-step inference to obtain initial depth snippets ( k k k ). Next, depth co-alignment optimizes NT pairs of scale and shift values to achieve globally consistent depth throughout the full video. An optional refinement step further enhances details by applying additional, snippet-based denoising steps.

###### Figure 2. Overview of the RollingDepth Inference Pipeline. Given a video sequence x (with

|i|
|---|

#### 3.2. Extension to Snippets

Dilated Rolling Kernel. We construct multi-scale snippets using the dilated rolling kernel. For instance, for 3-frame snippets with dilation rate (frame spacing) g and stride h, the kernel picks frames (xi−g,xi,xi+g) from the input video, where i ∈ {g +1,g +1+h,g +1+2h,...}. By varying the dilation rate, we sample snippets with different frame rates, in order to capture temporal dependencies at different time scales. For each snippet of n frames, we then predict depth using the multi-frame LDM, to obtain a corresponding n-frame depth snippet.

Inspired by multi-view diffusion models [28, 41], we extend Marigold [25] to handle multiple frames by modifying its self-attention layers. In each self-attention block, we flatten tokens from all frames in a snippet into a single sequence, such that the attention mechanism operates across frames and captures spatial and temporal interactions. Unlike video diffusion models with factorized spatial-temporal attention, this approach can handle frames with varying temporal spacing, which makes it possible to sample snippets at lower frame rates and capture long-range dependencies, an advantage when processing long videos.

Depth Co-alignment. At this stage we have generated NT depth snippets. Each of them has its own scale and shift parameters {(sk,tk), k ∈ 1...T}, which are shared across its constituent frames. Our goal is to jointly compute NT scale and shift values such that they optimally align all snippets into a consistent video. At a given frame xi, there are Ni different individual depth maps {dij,j = 1...Ni} originating from different snippets, where Ni can vary from frame to frame. Let k(i,j) be an indexing function that retrieves the snippet index k for the j-th depthmap at frame i. To estimate the best alignment, we minimize the L1 loss over all individual depth predictions,

The original Marigold model predicts (affine-invariant) depth between image-specific near and far planes. This parametrization poses problems for video depth estimation, where the depth range can vary over time. We therefore retrain Marigold to predict inverse depth (like several other monodepth estimators [52, 72]), which is less sensitive to such variations, particularly in the far field.

#### 3.3. From Snippets to Video

Our multi-frame depth estimator operates on short snippets of n frames, where n ≪ NF. As these snippets are processed independently, each has its own scale and shift – which are arbitrary in the case of affine-invariant methods [18, 25, 52] including Marigold, but will in practice not be perfectly aligned even when using a metric depth estimator [37, 49, 78]. To resolve that ambiguity, we construct overlapping snippets with different temporal dilation rates. The frames shared between different snippets are subsequently used to align all depth predictions to one common scale and shift.

 

 , (1)

Ni

NF

sk(i,j)dij + tk(i,j) − di

min

sk>0,tk

i=1

j=1

with the mean depth

Ni

1 Ni

sk(i,j)dij + tk(i,j) . (2)

di =

j=1

The solution to eq. (1) is found with gradient descent, stabilized by putting more emphasis on snippets with high di-

Co-Aligned Depth Sequence

malized on a per-snippet basis, using the 2nd and 98th percentiles for robustness. We found it important to jointly normalize the values within each snippet rather than normalizing each frame individually. In this way, the same frame is normalized differently depending on the context it appears in, and normalized depths remain comparable within a snippet, enabling the model to understand and correctly handle rapid changes in the depth range, which routinely appear in longer video sequences.

|1|2|3|4|5|6|7|8|9|10|11|12| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

Latent Space Add Noise

|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|

Refine

| | | | | | | | | | |[Figure 35]<br><br>| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | |Average|
| | | | | | | | | | | | |
| | | | | | | | | | | | |

Average

w/ Reduced Dilation

|1|2|3|4|5|6|7|8|9|10|11|12| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | |

|1|2|3|4|5|6|7|8|9|10|11|12|
|---|---|---|---|---|---|---|---|---|---|---|---|

Output

### 4. Experiments

Refined Depth Sequence

#### 4.1. Implementation Details

- Figure 3. Depth Refinement encodes the co-aligned depth video into latent space, contaminates it with a moderate amount of noise, then denoises it with a series of reverse diffusion steps with decreasing snippet dilation rate. After each step, overlapping latents are averaged to propagate information between snippets.

Training Datasets. To finetune the snippet LDM, we use TartanAir [62], a synthetic video dataset with various (indoor and outdoor) scenes, styles, and camera motions. We visually inspect the scenes and select 18 scenes consisting of 369 sequences. Training snippets are randomly sampled from a sequence, with a minimum overlap ratio of 30%. To increase the diversity of scenes and avoid a significant simto-real gap we additionally use Hypersim [53], a photorealistic single-image dataset containing 365 diverse scenes, treating images as 1-frame snippets.

lation rates, and additional regularization. Once the depth snippets have been aligned in a common frame with the estimated scale and shift values, the depth maps at every frame xi are obtained by taking the pixel-wise mean di, resulting in a single, consistent depth video with one depth map per frame. See Sec. A.1 for further details.

Training Settings. Training images are resized to 480×640 for efficiency, with random horizontal flipping as data augmentation. To align with the refinement setting, we employ depth range augmentation, where we randomly squeeze the normalized depth snippets to a smaller range and then slightly rescale and shift the depth range in each frame. As optimizer, we use AdamW [42] with a learning rate of 3 × 10−5 and exponential decay. Training is run on four Nvidia A100 GPUs with a batch size of 32 and takes approximately 18k iterations or two days to converge.

Depth Refinement. To enhance visual quality and capture finer details, we optionally apply a diffusion-based refinement step to the merged depth video d, as illustrated in Fig. 3. The video is again encoded into latent space frame by frame, and contaminated with a moderate amount of noise, corresponding to step T/2 of the diffusion schedule, halfway between the clean latent and pure noise. The degraded video is again split into snippets with the dilated rolling kernel and each snippet is denoised individually with the same LDM as above. To integrate information across overlapping snippets, the latent embeddings of every frame are averaged after every denoising step. We find that this partial (reverse) diffusion works best when applied in a coarse-to-fine manner in time, starting with a large snippet dilation rate and gradually decreasing it along the denoising process. The refinement process enhances high-frequency detail without altering the global scene depth layout, at the cost of increased inference time due to the additional round of denoising diffusion.

Inference Settings. During inference, we fix the snippet length to n = 3, with three different dilation rates g ∈ {1,10,25} to capture short- to mid-range temporal relations. For each snippet we perform 1-step inference. Long-range temporal relations are covered by the depth coalignment, which is initialized with sk =1 and tk =0 and optimized with 2000 steps of gradient descent, using the Adam optimizer. For the optional refinement, we start at timestep T/2 of the diffusion trajectory and perform 10 denoising steps, gradually reducing the dilation rate from 6 to 1. Input images are resized to a maximum side length of 768 pixels. For evaluation, the final result is up-sampled to match the original resolution in the dataset.

#### 3.4. Multi-Frame Training

We exploit the flexible design of the multi-frame selfattention mechanism to fine-tune the model with varying snippet lengths. Training snippets are randomly picked to have one, two, or three frames, making sure that the motion between frames is small enough to have overlapping view frustra. To fully utilize the value range of the diffusion model for best performance, inverse depth values are nor-

#### 4.2. Evaluation

Evaluation Datasets. We evaluate RollingDepth on four datasets that include both static and dynamic scenes with varying camera and scene motions: PointOdyssey [87] is a synthetic dataset with individually animated characters that move independently, designed for long-term tracking.

Table 1. Quantitative comparison of RollingDepth with baseline methods on zero-shot benchmarks. Bold numbers are the best, underscored second best, numbers in the bracket after each dataset denote video sequence length. RollingDepth demonstrates superior performance across both short and long video sequences, despite being an image-based model.

PointOdyssey (250) ScanNet (90) Bonn (110) DyDToF (200) DyDToF (100) Abs Rel↓ δ1 ↑ Abs Rel↓ δ1 ↑ Abs Rel↓ δ1 ↑ Abs Rel↓ δ1 ↑ Abs Rel↓ δ1 ↑

Marigold∗ [25] 14.9 80.4 14.9 78.3 10.5 86.7 25.3 55.5 16.4 73.5 DepthAnything [71] 16.3 76.0 12.9 84.0 9.9 89.4 25.4 54.3 16.4 75.6 DepthAnythingv2 [72] 14.4 81.4 13.3 82.6 10.5 87.4 24.8 55.9 16.0 76.6

Single

frame

NVDS (DPT-Large) [64] 26.6 68.2 18.5 67.7 10.5 88.1 24.7 56.0 18.8 69.3 ChronoDepth [58] 51.7 71.2 16.8 73.8 10.9 86.9 26.9 53.2 19.9 66.5 DepthCrafter [24] 36.3 75.0 12.7 84.3 6.6 96.7 22.1 60.7 16.2 74.7

Video

RollingDepth (ours, fast)† 9.6 90.4 10.1 89.7 7.9 93.6 17.7 69.6 12.7 81.6 RollingDepth (ours) 9.6 90.5 9.3 91.6 7.9 93.9 17.3 71.7 12.3 83.0

∗Inverse depth version, retrained with the original training code. †Run at half-precision (fp16), with dilation rates {1, 25}, without refinement.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

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

- Figure 4. Qualitative comparison between different methods. RollingDepth excels at preserving fine-grained details (cf. the chandelier in the first sample and the tripod in the third sample) and recovering accurate scene layout (cf. the far plane in the second sample).

We filter out overly simplistic toy scenes and retain 35 sequences. For each sequence, we follow the videodepth literature and exclude frames with camera zoom, then select the first 250 frames in each sequence. ScanNet v2 [8] is an indoor dataset of static scenes recorded with the Kinect RGB-D sensor. We use its test set of 100 sequences, taking the first 270 frames of each sequence and downsample the frame rate by a factor 3, Bonn RGBD [47] is an RGB-D dataset of moving people in indoor spaces. Following [24], we use frames 30-140 from five different dynamic scenes. DyDToF [60] is a photorealistic synthetic dataset featuring moving objects including people and animals. It has several videos per scene, we always take the first video and create two subsets of different lengths from it, by clipping frames 50-250, respectively frames 50-150.

Evaluation Protocol. We extend the affine-invariant depth evaluation protocol [52] to videos, i.e., depth predictions dˆ are aligned to the ground truth with a scale and shift found

with least squares fitting, where we fit one pair of transformation parameters per video, i.e., all frames in a video share a common scale and shift. We quantify the depth estimation accuracy with two standard metrics [24, 25, 52, 64]: the absolute mean relative error (AbsRel), defined as

M j=1 |dˆj − dj|/dj, where M is the total number of pixels; and the δ1-accuracy, which measures the fraction of pixels for which max(dˆj/dj,dj/dˆj) < 1.25. Metrics are always given as percentages. We provide additional temporal smoothness evaluation in the supplementary material.

1 M

#### 4.3. Comparison with Other Methods

We compare RollingDepth against six state-of-the-art methods for zero-shot monocular depth estimation: Marigold [25], DepthAnything [71] and DepthAnythingv2 [72], which are single-frame methods; as well as NVDS [65], ChronoDepth [58], and DepthCrafter [24], which are video-based approaches.

RollingDepth (Ours) DepthCrafter ChronoDepth DepthAnythingV2

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Frame 10

Frame 190

Frame10Frame190

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

- Figure 5. AbsRel error over time: The line plot (left) shows the depth error at every individual frame, end-of-line numbers are the average error across the video. The images (right) display error maps (low high) for two specific frames. RollingDepth achieves the lowest error overall, competing methods recover scene layout less faithfully and tend to be biased towards the foreground or the background.

[Figure 73]

[Figure 74]

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

[Figure 82]

|[Figure 83]|
|---|

[Figure 84]

|[Figure 85]|
|---|

[Figure 86]

|[Figure 87]|
|---|

[Figure 88]

|[Figure 89]|
|---|

[Figure 90]

|[Figure 91]|
|---|

[Figure 92]

|[Figure 93]|
|---|

[Figure 94]

|[Figure 95]|
|---|

[Figure 96]

|[Figure 97]|
|---|

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

[Figure 101]

|[Figure 102]|
|---|

|[Figure 103]|
|---|

[Figure 104]

|[Figure 105]|
|---|

- Figure 6. Qualitative comparison of depth predictions (near far) from in-the-wild videos. To graphically show temporal consistency, we display temporal profiles (red box) for a fixed column (marked with a red line). RollingDepth picks up subtle details like accessories and wrinkled cloth, and mitigates spurious depth discontinuities (cf. background in temporal profile of the first sample) in time.

[Figure 106]

Quantitative Comparison. As shown in Tab. 1, RollingDepth outperforms both single-frame and videobased approaches across multiple datasets and different sequence lengths, often by considerable margins. We attribute this to its ability to combine the accuracy of imagebased models with the temporal coherence afforded by our snippet-based inference and global depth co-alignment. On PointOdyssey, which includes many challenging scenes with highly variable depth ranges, RollingDepth achieves by far the best result. Methods based on video models struggle on this dataset, and are in fact even unable to match the performance of single-frame methods. We observe that the performance of video models drops especially in scenes with sudden, large changes in the depth range (e.g. a hand gesture in front of the camera). We hypothesize that the underlying video prior is too rigid and prevents a correct adaptation to the rapid change, see Sec. B.4 for details. Also on

DyDToF, RollingDepth greatly reduces the error compared to other methods, again underscoring its ability to handle dynamic scenes and variations of the near and far planes. Still, the good performance is not limited to dynamic scenes with strong depth variations. RollingDepth also performs well on indoor data, reaching the lowest error on the static ScanNet scenes and the second-lowest error on the Bonn data. Here DepthCrafter shines – we observe that it generally tends to do well in scenes dominated by foreground objects, particularly humans.

Qualitative Comparison. To make our findings more tangible, we provide qualitative comparisons both on evaluation data and on in-the-wild examples. Figure. 4 confirms that RollingDepth consistently produces high-quality depth maps that preserve fine detail, both near the camera in the distance. DepthCrafter and ChronoDepth produce locally smooth videos with little frame-to-frame flicker, but have a

tendency to distort the overall scene layout in a way that certain objects are segmented well but placed at incorrect (relative) depths. Single-image estimators are seemingly more accurate in that respect, but suffer from flickering and a lack of temporal coherence. We further illustrate these trends in

- Fig. 5, where we plot per-frame errors, as well as per-pixel errors for selected frames.

To demonstrate generalization to real-world video clips,

- Fig. 6 shows depth predictions for videos collected from the internet. Also in these cases, RollingDepth accurately recovers fine details and maintains long-term coherence. To better illustrate the evolution of the depth estimate over time, we extract temporal profiles for fixed image columns. They exhibit no significant high-frequency variations along the time axis that would indicate frame-to-frame flicker. We also do not observe drift or unwarranted jumps in the depth values that would indicate systematic biases. DepthCrafter for the most part also recovers plausible depth, but misses depth variations within the main segments and sometimes exhibits instabilities along the time axis. Chronodepth recovers depth boundaries rather well, but delivers billboardlike, layered depth maps.

#### 4.4. Ablation Studies

We validate our main hyper-parameters and design choices on a subset of 10 sequences from the PointOdyssey test set and 20 sequences from the ScanNet test set.

Dilation of Initial Predictions. We start by ablating the arguably most crucial hyper-parameter of RollingDepth, the dilation rate for snippet sampling, see Tab. 2. The base setting uses only dilation rate {1} for minimal information exchange and smoothness between adjacent frames. Having a high dilation rate {1,25} gives the model access to longerterm motion patterns on the order of 1 second and greatly stabilizes the co-alignment step, which in turn reduces the AbsRel error by >6 percept points on PointOdyssey and by >2 percent points on the (static) ScanNet. This is what we use in our fast setting (c.f. Tab. 1), which takes 81s for a 768×432 video of 250 frames (ChronoDepth: 121s, DepthCrafter: 284s). An additional, intermediate dilation rate {1,10,25} further intensifies the information exchange across time. This further boosts the quality of the estimated depth maps, but as expected yields diminishing returns.

Effectiveness of Co-Alignment and Refinement. We further isolate the effect of the RollingDepth’s components, see Tab. 3. The snippet diffusion step is mandatory to obtain any depth estimates at all and cannot be left out. For the experiment we switch on and off the two remaining steps, co-alignment and refinement, and test all combinations. Simply merging overlapping latents without prior alignment proves to be insufficient, i.e., their individually estimated depth ranges are too inconsistent to average them into a coherent sequence. The refinement step cannot fix

- Table 2. Ablation of dilation rates for snippet prediction. We report values before the optional refinement step. The minimal base setting uses only dilation rate 1. Adding a high dilation rate 25 brings a marked performance gain. Yet another dilation rate 10 gives a further, smaller boost.

PointOdyssey ScanNet

Dilation rates Abs Rel↓ δ1 ↑ Abs Rel↓ δ1 ↑ {1} 16.7 75.5 12.8 83.2 {1, 25} 10.2 89.5 10.6 88.8 {1, 10, 25} 10.2 89.6 9.9 90.1

- Table 3. Ablation of components. Depth co-alignment is a crucial functionality for the snippet-based strategy of RollingDepth, whereas the additional refinement has only a small effect on the performance metrics, despite visibly enhanced image detail.

PointOdyssey ScanNet

Co-Alignment Refinement Abs Rel↓ δ1 ↑ Abs Rel↓ δ1 ↑ × × 13.0 84.4 12.4 84.3 × ✓ 13.0 84.6 12.3 84.8 ✓ × 10.2 89.6 9.9 90.1 ✓ ✓ 10.2 89.8 9.8 90.2

that problem. Conversely, the co-alignment does the heavy lifting to fuse depth snippets with different scales and shifts into a coherent video and contributes the lion’s share of the improvement. Subsequent refinement of the aligned video only results in a marginal increase of the performance metrics, but visibly improves the result by recovering sharp details that have been missed or blurred in the preceding steps.

### 5. Conclusion

We have introduced RollingDepth, a novel method for monocular video depth estimation that is derived from a single-image (latent) diffusion model. The core components of our method (i) are a monodepth estimator for short snippets, sampled at various frame rates to capture temporal context at different time scales; (ii) an optimization-based co-alignment procedure that optimally registers all snippets of a video into a common depth range; and (iii) an optional refinement step, again based on the same denoising diffusion scheme for snippets, that enhances fine details in the depth video. RollingDepth strikes a favorable balance between accurate per-frame depth prediction and temporal coherence, and can process long video with hundreds of frames. It empirically delivers best-in-class performance across multiple datasets, also outperforming alternatives derived from full-blown video diffusion models. That being said, the RollingDepth framework is flexible and offers the possibility to replace individual components. For instance, an interesting avenue for future work would be to swap out the snippet-based refinement and replace it with a generative video model or a flow-based method for even better motion reconstruction.

Acknowledgments. We thank Yue Pan, Shuchang Liu, Nando Metzger, and Nikolai Kalischek for fruitful discussions. We are grateful to redmond.ai for providing GPU resources.

### References

- [1] Shubhra Aich, Jean Marie Uwabeza Vianney, Md Amirul Islam, Mannat Kaur, and Bingbing Liu. Bidirectional attention network for monocular depth estimation. In ICRA, 2021. 3
- [2] Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. AdaBins: Depth estimation using adaptive bins. In CVPR,

2021. 3

- [3] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. ZoeDepth: Zero-shot transfer by combining relative and metric depth. arXiv:2302.12288, 2023. 3
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv:2311.15127, 2023. 2, 3
- [5] Aleksei Bochkovskii, Ama¨el Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R Richter, and Vladlen Koltun. Depth Pro: Sharp monocular metric depth in less than a second. arXiv:2410.02073, 2024. 2, 3
- [6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2Video: Video editing using image diffusion. In ICCV,

2023. 3

- [7] Yuhua Chen, Cordelia Schmid, and Cristian Sminchisescu. Self-supervised learning with geometric constraints in monocular video: Connecting flow, depth, and camera. In ICCV, 2019. 3
- [8] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Niessner. ScanNet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 6
- [9] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017. 12
- [10] Giannis Daras, Weili Nie, Karsten Kreis, Alex Dimakis, Morteza Mardani, Nikola Borislavov Kovachki, and Arash Vahdat. Warped diffusion: Solving video inverse problems with image diffusion models. NeurIPS, 2024. 3
- [11] Yiqun Duan, Xianda Guo, and Zheng Zhu. DiffusionDepth: Diffusion denoising approach for monocular depth estimation. arXiv:2303.05021, 2023. 3
- [12] David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using a multi-scale deep network. NeurIPS, 2014. 2
- [13] Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, and Dacheng Tao. Deep ordinal regression network for monocular depth estimation. In CVPR, 2018. 3
- [14] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. GeoWizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. In ECCV, 2024. 3

- [15] Yasutaka Furukawa, Carlos Hern´andez, et al. Multi-view stereo: A tutorial. Foundations and Trends® in Computer Graphics and Vision, 9(1-2):1–148, 2015. 2
- [16] Gonzalo Martin Garcia, Karim Abou Zeid, Christian Schmidt, Daan de Geus, Alexander Hermans, and Bastian Leibe. Fine-tuning image-conditional diffusion models is easier than you think. arXiv:2409.11355, 2024. 3, 12
- [17] Carsten Griwodz, Simone Gasparini, Lilian Calvet, Pierre Gurdjos, Fabien Castan, Benoit Maujean, Gregoire De Lillo, and Yann Lanthony. AliceVision Meshroom: An opensource 3d reconstruction pipeline. In ACM Multimedia,

2021. 1

- [18] Ming Gui, Johannes S Fischer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. DepthFM: Fast monocular depth estimation with flow matching. arXiv:2403.13788, 2024. 3, 4
- [19] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 3d packing for self-supervised monocular depth estimation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 13
- [20] Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares, Ambrus,, and Adrien Gaidon. Towards zero-shot scale-aware monocular depth estimation. In ICCV, 2023. 3
- [21] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Liu, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv:2409.18124, 2024. 3
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 3
- [23] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3D v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. arXiv:2404.15506, 2024. 3
- [24] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. DepthCrafter: Generating consistent long depth sequences for open-world videos. arXiv:2409.02095, 2024. 2, 3, 6, 12
- [25] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 2, 3, 4, 6, 12
- [26] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d Gaussian splatting for real-time radiance field rendering. ACM TOG, 42(4):139–1, 2023. 2
- [27] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2Video-Zero: Textto-image diffusion models are zero-shot video generators. In ICCV, 2023. 3
- [28] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. EscherNet: A generative model for scalable view synthesis. In CVPR, 2024. 4
- [29] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In CVPR, 2021. 3

- [30] Taesung Kwon and Jong Chul Ye. Solving video inverse problems using image diffusion models. arXiv:2409.02574,

2024. 3

- [31] Jin Han Lee, Myung-Kyu Han, Dong Wook Ko, and Il Hong Suh. From big to small: Multi-scale local planar guidance for monocular depth estimation. arXiv:1907.10326, 2019. 3
- [32] Zhengqi Li and Noah Snavely. MegaDepth: Learning singleview depth prediction from internet photos. In CVPR, 2018. 3
- [33] Zhenyu Li, Xuyang Wang, Xianming Liu, and Junjun Jiang. BinsFormer: Revisiting adaptive bins for monocular depth estimation. arXiv:2204.00987, 2022. 3
- [34] Zhenyu Li, Zehui Chen, Xianming Liu, and Junjun Jiang. DepthFormer: Exploiting long-range correlation and local information for accurate monocular depth estimation. Machine Intelligence Research, pages 1–18, 2023. 3
- [35] Zhaoshuo Li, Wei Ye, Dilin Wang, Francis X Creighton, Russell H Taylor, Ganesh Venkatesh, and Mathias Unberath. Temporally consistent online depth estimation in dynamic scenes. In WACV, 2023. 3
- [36] Zhenyu Li, Shariq Farooq Bhat, and Peter Wonka. PatchFusion: An end-to-end tile-based framework for highresolution monocular metric depth estimation. In CVPR,

2024. 3

- [37] Zhenyu Li, Shariq Farooq Bhat, and Peter Wonka. PatchRefiner: Leveraging synthetic data for real-domain highresolution monocular metric depth estimation. In ECCV,

2024. 3, 4

- [38] Shanchuan Lin, Bingchen Liu, Jiashi Li, and Xiao Yang. Common diffusion noise schedules and sample steps are flawed. In WACV, 2024. 12
- [39] Ce Liu, Suryansh Kumar, Shuhang Gu, Radu Timofte, and Luc Van Gool. VA-DepthNet: A variational approach to single image depth prediction. In ICLR, 2023. 3
- [40] Shaoteng Liu, Yuechen Zhang, Wenbo Li, Zhe Lin, and Jiaya Jia. Video-P2P: Video editing with cross-attention control. In CVPR, 2024. 3
- [41] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. SyncDreamer: Generating multiview-consistent images from a single-view image. arXiv:2309.03453, 2023. 4
- [42] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 5
- [43] Xuan Luo, Jia-Bin Huang, Richard Szeliski, Kevin Matzen, and Johannes Kopf. Consistent video depth estimation. ACM Transactions on Graphics, 39(4), 2020. 3
- [44] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [45] Jia Ning, Chen Li, Zheng Zhang, Chunyu Wang, Zigang Geng, Qi Dai, Kun He, and Han Hu. All in tokens: Unifying output space of visual tasks via soft token. In ICCV,

2023. 3

- [46] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

- Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. arXiv:2304.07193, 2023. 2, 3
- [47] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguere, and Cyrill Stachniss. ReFusion: 3d reconstruction in dynamic environments for RGB-D cameras exploiting residuals. In IROS, 2019. 6
- [48] Vaishakh Patil, Christos Sakaridis, Alexander Liniger, and Luc Van Gool. P3Depth: Monocular depth estimation with a piecewise planarity prior. In CVPR, 2022. 3
- [49] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal monocular metric depth estimation. In CVPR,

2024. 3, 4

- [50] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. UniDepthV2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110, 2025. 3
- [51] Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. FateZero: Fusing attentions for zero-shot text-based video editing. In ICCV, 2023. 3
- [52] Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE TPAMI, 2020. 2, 3, 4, 6
- [53] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In ICCV, 2021. 5
- [54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [55] Saurabh Saxena, Abhishek Kar, Mohammad Norouzi, and David J Fleet. Monocular depth estimation using diffusion models. arXiv:2302.14816, 2023. 3
- [56] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 1
- [57] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS, 2022. 3
- [58] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. arXiv:2406.01493, 2024. 2, 3, 6, 12
- [59] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021. 3
- [60] Zhanghao Sun, Wei Ye, Jinhui Xiong, Gyeongmin Choe, Jialiang Wang, Shuochen Su, and Rakesh Ranjan. Consistent direct time-of-flight video depth super-resolution. arXiv:2211.08658, 2022. 6, 12
- [61] Zachary Teed and Jia Deng. Deepv2d: Video to depth with differentiable structure from motion. In ICLR, 2020. 3

- [62] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. TartanAir: A dataset to push the limits of visual SLAM. In IROS, 2020. 5
- [63] Yiran Wang, Zhiyu Pan, Xingyi Li, Zhiguo Cao, Ke Xian, and Jianming Zhang. Less is more: Consistent video depth estimation with masked frames modeling. In ACM MM,

2022. 3, 12

- [64] Yiran Wang, Min Shi, Jiaqi Li, Zihao Huang, Zhiguo Cao, Jianming Zhang, Ke Xian, and Guosheng Lin. Neural video depth stabilizer. In ICCV, 2023. 3, 6
- [65] Yiran Wang, Min Shi, Jiaqi Li, Chaoyi Hong, Zihao Huang, Juewen Peng, Zhiguo Cao, Jianming Zhang, Ke Xian, and Guosheng Lin. NVDS+: Towards efficient and versatile neural stabilizer for video depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 3, 6
- [66] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-A-Video: One-shot tuning of image diffusion models for text-to-video generation. In ICCV, 2023. 3
- [67] Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Diffusion models trained with large data are transferable visual models. arXiv:2403.06090, 2024. 3
- [68] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. GMFlow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8121– 8130, 2022. 12
- [69] Guanglei Yang, Hao Tang, Mingli Ding, Nicu Sebe, and Elisa Ricci. Transformer-based attention networks for continuous pixel-wise prediction. In ICCV, 2021. 3
- [70] Honghui Yang, Di Huang, Wei Yin, Chunhua Shen, Haifeng Liu, Xiaofei He, Binbin Lin, Wanli Ouyang, and Tong He. Depth any video with scalable synthetic data. arXiv:2410.10815, 2024. 3
- [71] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024. 3, 6
- [72] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. arXiv:2406.09414, 2024. 2, 3, 4, 6
- [73] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia, 2023. 3
- [74] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. MVSNet: Depth inference for unstructured multiview stereo. In ECCV, 2018. 2
- [75] Rajeev Yasarla, Hong Cai, Jisoo Jeong, Yunxiao Shi, Risheek Garrepalli, and Fatih Porikli. MAMo: Leveraging memory and attention for monocular video depth estimation. In ICCV, 2023. 3
- [76] Rajeev Yasarla, Manish Kumar Singh, Hong Cai, Yunxiao Shi, Jisoo Jeong, Yinhao Zhu, Shizhong Han, Risheek Garrepalli, and Fatih Porikli. FutureDepth: Learning to predict

- the future improves video depth estimation. In European Conference on Computer Vision, pages 440–458. Springer, 2024. 3
- [77] Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi Tian, Songcen Xu, Changming Sun, and Dou Renyin. DiverseDepth: Affine-invariant depth prediction using diverse data. arXiv:2002.00569, 2020. 3
- [78] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3D: Towards zero-shot metric 3d prediction from a single image. In ICCV, 2023. 3, 4
- [79] Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, and Ping Tan. NeWCRFs: Neural window fully-connected CRFs for monocular depth estimation. In CVPR, 2022. 3
- [80] Haokui Zhang, Chunhua Shen, Ying Li, Yuanzhouhan Cao, Yu Liu, and Youliang Yan. Exploiting temporal consistency for real-time video depth estimation. In ICCV, 2019. 3
- [81] Xiang Zhang, Bingxin Ke, Hayko Riemenschneider, Nando Metzger, Anton Obukhov, Markus Gross, Konrad Schindler, and Christopher Schroers. BetterDepth: Plug-and-play diffusion refiner for zero-shot monocular depth estimation. NeurIPS, 2024. 3
- [82] Zhoutong Zhang, Forrester Cole, Richard Tucker, William T Freeman, and Tali Dekel. Consistent depth of moving objects in video. ACM Transactions on Graphics (TOG), 40(4):1– 12, 2021. 3
- [83] Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, and Luoqi Liu. Towards consistent video editing with text-to-image diffusion models. NeurIPS, 2024. 3
- [84] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing text-to-image diffusion models for visual perception. arXiv:2303.02153, 2023. 3
- [85] Zixiang Zhao, Jiangshe Zhang, Shuang Xu, Zudi Lin, and Hanspeter Pfister. Discrete cosine transform network for guided depth map super-resolution. In CVPR, pages 5697– 5707, 2022. 3
- [86] Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, and Luc Van Gool. DDFM: Denoising diffusion model for multi-modality image fusion. In ICCV, 2023. 3
- [87] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. PointOdyssey: A large-scale synthetic dataset for long-term point tracking. In ICCV,

2023. 5, 12

## Supplementary Material

This supplementary material includes additional implementation details and experimental results.

### A. Implementation Details

#### A.1. Depth Co-Alignment

As discussed in Sec. 3.3, let k(i,j) denote an indexing function that returns the snippet index k corresponding to the jth depthmap of i-th frame. To make the optimization more robust, we include an additional loss term in depth space while predicting inverse depth. We further scale the loss terms by their respective mean absolute value per frame to increase the numerical stability. Additionally, soft constraints on sk,tk are applied:

 

 

−1

Ni

NF

dij − di µi

dij

− di µi

min

+

sk>0,tk

i=1

j=1

+ λ1 max(0,1 − sk(i,j))2 + λ2tk(i,j), (3)

where dij = sk(i,j)dij + tk(i,j). The mean depth and mean inverse depth are defined as

1 Ni

di =

Ni

Ni

1 Ni

dij di =

j=1

j=1

−1

, (4)

dij

with the corresponding mean absolute values per frame given by

HW

HW

1 HW

1 HW

di . (5)

di µi =

µi =

We found that λ1 = 10−1, λ2 = 101 work well in practice.

#### A.2. Additional Training and Inference Details

During training, we follow Marigold to use MSE loss on the latents. We apply gradient accumulation to increase the effective batch size, to 32. To better mix the samples with varying snippet lengths, every mini-batch is sampled randomly and can have different snippet lengths. For the initial depth prediction, we apply the same random Gaussian noise to all frames. When applying refinement, the same noise is used to perturb the (encoded) co-aligned depth sequence. The denoising process then starts from timestep T/2.

#### A.3. Evaluation Datasets

PointOdyssey [87] contains several sequences that feature overly simplified toy scenes, as well as some with smoke, for which depth estimation is ambiguous (cf. Fig. S1). We exclude these sequences from the test dataset, a detailed list of selected frames will be provided with the code. For evaluation, pixels on windows are excluded due to inconsistent depth labels.

In ScanNet [9], the RGB images and depth labels include a thin black border. Following DepthCrafter [24], we crop the RGB images by removing 8 pixels from the top and bottom and 12 pixels from the left and right. Similarly, we crop the depth maps by removing 4 pixels from the top and bottom and 6 pixels from the left and right.

For DyDToF [60], we exclude depth values beyond 23m, corresponding to less than 1% of the depth values.

[Figure 107]

[Figure 108]

Figure S1. Examples of PointOdyssey toy scenes (left) and scenes with smoke (right).

#### A.4. Baseline Methods

We evaluate baseline methods using their recommended default settings. For DepthCrafter [24], the inference is performed with 25 diffusion steps, using an overlap of 25 frames for videos longer than 110 frames. For ChronoDepth [58], inference comprises 10 denoising steps, with a window size of 10 (referred to as “num-frames” in the code) and a stride of 9 (referred to as “denoise-steps” in the code).

For Marigold [25], we retrained an inverse depth version using the trailing scheduler setting [16, 38]. Under this configuration, 1-step inference with a single model achieves performance comparable to the original configuration with multi-step inference and ensembling, so we utilize the former, more efficient setting.

### B. Additional Experiment Results B.1. Temporal smoothness evaluation

We further quantitatively evaluate the temporal smoothness using optical-flow-based warping loss (OPW) [63] on PointOdyssey and ScanNet datasets and report the results in Tab. S1. The optical flow is estimated using GMFlow [68].

Table S1. Temporal smoothness (OPW↓) comparison. All values are ×103, lower is better. ∗ denotes catastrophic failures on some sequences. Numbers in brackets are evaluated on subsets that exclude those cases.

PointOdyssey ScanNet

Marigold 3.52∗ (4.00) 0.48 DepthAnything 3.92∗ (4.21) 0.32 NVDS 3.50∗ (2.97) 0.29 ChronoDepth 8.98∗ (2.99) 0.29 DepthCrafter 7.75∗ (1.30) 0.25 RollingDepth (ours) 1.42∗ (1.63) 0.20

We notice that ChronoDepth and DepthCrafter have

catastrophic failure in some cases of PointOdyssey (cf. Sec. B.4), leading to large errors, as denoted by ∗. We manually exclude these failure cases. The re-calculated average OPW is reported in the brackets. Overall, RollingDepth shows good smoothness, on par with DepthCrafter, while being more robust than DepthCrafter and ChronoDepth against occasional failures.

We point out that OPW only evaluates the “smoothness” between adjacent frames while ignoring the longterm smoothness and geometric consistency. As shown in Tab. S2, with larger dilation rates, the geometric accuracy shows a clear improving trend, while the trend of OPW is unclear. We hypothesize that with a larger dilation rate, geometric accuracy is improved at a cost of minor local smoothness decrease when merging the aligned snippets.

Table S2. Extended table of dilation rate ablation study (Tab. 2). Values are ×103.

PointOdyssey ScanNet Dilation rates Abs Rel↓ δ1 ↑ OPW ↓ Abs Rel↓ δ1 ↑ OPW ↓

{1} 16.7 75.5 1.22 12.8 83.2 0.24 {1, 25} 10.2 89.5 2.06 10.6 88.8 0.29 {1, 10, 25} 10.2 89.6 1.98 9.9 90.1 0.29

#### B.2. Evaluation on DDAD dataset

We further evaluate the model performance on the DDAD [19] dataset, which is a driving-scene dataset with sparse depth annotation. We use the 100-frame sequences on the test set.

As shown in Tab. S3, RollingDepth outperforms other methods in terms of accuracy and smoothness.

Table S3. Evaluation on DDAD dataset.

Abs Rel↓ δ1 ↑ OPW ↓

×10−2 ×10−2 ×10−3

NVDS 30.8 57.2 0.39 ChronoDepth 34.2 46.9 0.21 DepthCrafter 19.3 74.8 0.28 RollingDepth (ours) 12.8 83.2 0.19

Table S4. Inference speed and peak GPU memory usage comparison on a 768×432 video of 250 frames. By increasing the batch size of processing, RollingDepth† can trade memory for speed.

Time (s) Peak GPU Memory (GB)

NVDS 284 17.6 ChronoDepth 121 15.0 DepthCrafter 284 13.6 RollingDepth (ours) 105 16.2 RollingDepth† (ours) 181 40.1

sudden changes require rapid alterations of the depth range, both before and after the event. Video models tend to produce incorrect overall scene layout in such cases, we hypothesize that they ”try too hard” to equalize the depth range throughout the scene.

#### B.5. Failure Cases of RollingDepth

While our proposed method handles changing depth range more robustly than video models, it also has certain limitations. Two examples are shown in Fig. S3. RollingDepth sometimes misjudges the depth of cloudy skies. Another source of error is transparent surfaces such as glass windows, where subtle variations of transparency or reflections may cause the depth to oscillate between the glass and the scene behind it – a common issue of depth estimators.

#### B.3. Inference efficiency

We report the inference efficiency comparison in Tab S4. The benchmarking is done on the same machine with a single RTX3090 GPU. For each method, we run 10 repeated inferences after a warm-up iteration, with the model loaded on GPU, and calculated the mean run time and peak memory footage of each iteration.

#### B.4. Failure cases of video models on PointOdyssey

We provide further examples from the PointOdyssey dataset where video-based methods struggle. Figure S2 features scenes with large depth changes, such as hand gestures in front of the camera or objects entering the near field. These

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

- Figure S2. Examples of PointOdyssey samples that challenge video models. In the cases above, the (inverse) depth range varies significantly across frames. The arrows highlight situations where video models yield distorted depth maps. In the first two rows, this occurs in regions where the depth deviates significantly from the surrounding scene. In the last row, the depth predictions get drawn towards the near plane to match the object close to the camera, biasing the depth in the far field.

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

- Figure S3. The two samples on the left show incorrect depth predictions in the cloudy sky. The two samples on the right show inconsistencies between different frames of the same video, where the depth at the glass windows fluctuates between the solid and transparent states.

