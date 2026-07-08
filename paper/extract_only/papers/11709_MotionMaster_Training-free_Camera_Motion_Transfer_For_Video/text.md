# MotionMaster: Training-free Camera Motion Transfer For Video Generation

Teng Hu∗, Jiangning Zhang∗, Ran Yi†, Yating Wang, Hongrui Huang, Jieyu Weng, Yabiao Wang, Lizhuang Ma

## arXiv:2404.15789v2[cs.CV]1May2024

Abstract—The emergence of diffusion models has greatly propelled the progress in image and video generation. Recently, some efforts have been made in controllable video generation, including text-to-video, image-to-video generation, video editing, and video motion control, among which camera motion control is an important topic. However, existing camera motion control methods rely on training a temporal camera module, and necessitate substantial computation resources due to the large amount of parameters in video generation models. Moreover, existing methods pre-define camera motion types during training, which limits their flexibility in camera control, preventing the realization of some specific camera controls, such as various camera movements in films. Therefore, to reduce training costs and achieve flexible camera control, we propose MotionMaster, a novel training-free video motion transfer model, which disentangles camera motions and object motions in source videos, and transfers the extracted camera motions to new videos. We first propose a one-shot camera motion disentanglement method to extract camera motion from a single source video, which separates the moving objects from the background and estimates the camera motion in the moving objects region based on the motion in the background by solving a Poisson equation. Furthermore, we propose a few-shot camera motion disentanglement method to extract the common camera motion from multiple videos with similar camera motions, which employs a window-based clustering technique to extract the common features in temporal attention maps of multiple videos. Finally, we propose a motion combination method to combine different types of camera motions together, enabling our model a more controllable and flexible camera control. Extensive experiments demonstrate that our training-free approach can effectively decouple camera-object motion and apply the decoupled camera motion to a wide range of controllable video generation tasks, achieving flexible and diverse camera motion control. More details can be referred in https://sjtuplayer.github.io/projects/MotionMaster.

Index Terms—Video Generation, Video Motion, Camera Motion, Disentanglement

✦

#### 1 INTRODUCTION

In recent years, the rapid development of generative models [1], [2] has led to significant advancements in the field of image and video generation. Among video generation, diffusion models [3]–[6] have emerged as powerful tools for generating high-quality videos with high diversity. Meanwhile, the demand for controllable video generation has grown significantly, especially in applications such as film production, virtual reality, and video games, where researchers have devoted much effort to controllable generation tasks including text-to-video generation [3], [5]–[7], image-to-video generation [3], [4], video motion control [8]– [11], and video editing [12], [13]. Since video is composed of a sequence of images with consistent and fluent motions, the control of video motion has become an important topic in controllable video generation.

For video motion control, 1) most of the existing methods [8], [10], [12], [14] focus on modeling the object motion and use trajectory or a source video to guide the movement of the objects, but usually lack the ability to model the camera motion. 2) To enable the control of

- • Teng Hu, Ran Yi, Yating Wang, Jieyu Weng, and Lizhuang Ma are with the Department of Computer Science and Engineering, Shanghai Jiao Tong University, Shanghai 200240, China (email: {huteng,ranyi,wyating 0929,w.jerry,ma-lz}@sjtu.edu.cn).

- • Jiangning Zhang and Yabioa Wang are with the Youtu Lab, Tencent, Shanghai 200233, China (email: {vtzhang, caseywang}@tencent.com).
- • Hongruihuang is with the Faculty of Computing, Harbin Institute of Technology, Harbin 150001, (email: 2022212016@stu.hit.edu.cn).

the camera motion, AnimateDiff [3] trains temporal LoRA modules [15] on a collected set of videos with the same camera motion. To control different camera motions using one model, MotionCtrl [16] labels a large number of videos with corresponding camera pose parameters to train a camera motion control module. In contrast, Direct-avideo [11] utilizes a self-supervised training process by manually constructing camera motions along x, y, and z axis, reducing the training resources to some extent. However, all the existing camera motion control methods rely on training a temporal camera module to control the camera motion, which poses a significant requirement to the computational resources due to the large number of parameters in video generation models. Moreover, these methods can only achieve simple camera motion control and cannot handle some complex and professional camera motions in films, such as Dolly Zoom (zoom in or out the camera while keeping the object still) and Variable-Speed Zoom (zoom with variable speed).

To achieve complex camera motion control and reduce the training costs, we propose MotionMaster, a novel training-free camera motion transfer model, which disentangles camera motions and object motions in source videos and then transfers the extracted camera motions to new videos. Firstly, we observe that the temporal attention maps in diffusion-based video generation models contain the information of video motions, and find that the motions are composed of two motion types, camera motions and object motions. We then propose two methods to disentangle the

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Zoom In

Optical Flow

- (a) Prompt: A beautiful butterfly on the flowers.
- (b) Prompt: A peaceful garden with ponds and flowers.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Zoom In+Pan Right

Optical Flow

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

Dolly Zoom Zoom In+Object Still

Optical Flow

(c) Prompt: A cute rabbit sits in the forest.

- Fig. 1: Flexible and diverse camera motion control of our training-free MotionMaster. MotionMaster can control (a) one camera motion or (b) combine several camera motions in one video. Moreover, MotionMaster enables control of different camera motions in different regions, which can achieve professional Dolly Zoom with zooming motions in the background and fixed motion in the foreground (c).

camera motions and object motions in temporal attention maps. 1) In one-shot camera motion disentanglement, we decompose camera and object motions from a single source video. We regard the motion in the background as only containing camera motion, while the motion in the foreground as containing both camera and object motions. We employ a segmentation model to separate the moving objects and background regions, and then predict the camera motion in foreground region from background motion by solving a Poisson equation. 2) To further enhance the disentanglement ability, we propose a few-shot camera motion disentanglement method to extract the common camera motion from several videos with similar camera motions, which employs a novel window-based clustering method to extract the common features from temporal attention maps of multiple videos. Finally, we investigate the additivity and positional composition ability of camera motions, and propose a camera motion combination method to achieve flexible camera control, which can enable combining different kinds of camera motions into a new motion, and apply different camera motions in different regions.

- • We propose a novel one-shot camera-object motion disentanglement method. By separating the moving objects and the background regions and estimating the camera motion in the moving objects region by solving a Poisson equation, our model can effectively disentangle the camera motion from object motion in a single video.
- • We further propose a few-shot camera-object motion disentanglement method, which employs a novel window-based clustering method to extract the common camera motion from several given videos with similar camera motions, effectively dealing with scenarios with overly complex and diverse object motions.
- • We propose a camera motion combination method to achieve flexible camera motion control, which enables the model to combine different camera motions into a new motion and apply different camera motions in different regions.

2 RELATED WORK

Extensive experiments demonstrate the superior performance of our model in both one-shot and few-shot camera motion transfer. With the camera motion combination and the disentanglement between the camera motion and position, our model substantially improve the controllability and flexibility of camera motions.

2.1 Text-to-Video Generation

Generative models have rapidly advanced and achieved tremendous success in text-driven video generation tasks, which mostly rely on generative adversarial networks (GANs) [17]–[20] and diffusion models [3]–[7], [21]–[23] Among these methods, diffusion models have emerged as a powerful tool due to their ability to generate diverse and high-quality contents. Early text-driven video generation models [5], [23], [24] perform diffusion in pixel space, requiring cascaded generation and significant computational resources to generate high-resolution videos. Recent research papers have implemented diffusion in the latent space

The main contributions can be summarized as follows:

• We propose MotionMaster, a training-free camera motion transfer method based on Camera-Object Motion Disentanglement, which can transfer the camera motion from source videos to newly generated videos.

[3], [4], [22], [25]–[27], achieving high-quality and longduration video generation. Additionally, researchers are exploring more controllable video generation approaches. For instance, [28]–[30] introduce spatial and geometric constraints to generative models, [31] generates videos of desired subject, and [8], [16] govern motion in generated videos. These methods enable users to finely control various attributes of videos, resulting in generated outcomes that better align with user preferences and requirements.

###### 2.2 Motion Controllable Video Generation

Object Motion Control. Many researches [9], [31]–[36] have been conducted to control object motions to better align with user preferences. Some methods [10], [11] enable users to control the motion of objects by dragging bounding boxes, while some other works [16], [33] allow control over the trajectory of the object. VideoComposer [37] provides global motion guidance by conditioning on pixel-wise motion vectors. Besides, some video editing methods [9], [12], [13], [38] also enable motion editing through text-driven or manually specified motions, which requires motion consistency between adjacent frames. In summary, all these works focus more on controlling the object motions rather than camera motions, which operates at a local, high semantic level.

Camera Motion Control. There have been relatively few researches in camera motion control. AnimateDiff [3] employs temporal LoRA modules [15] trained on a collected set of videos with similar camera motion. Thus a single LoRA module is capable of controlling only a specific type of camera motion. MotionCtrl [16] constructs a video dataset annotated with camera poses to learn camera motions, but requires substantial manual effort. Direct-a-video [11] adds camera motion along coordinate axes to existing videos, which can reduce annotation costs. However, all of these works require fine-tuning pretrained video generation models, consuming a large amount of computation resources and limiting the style of camera motion to the training data. In contrast, our model enables flexible camera motion control with any target camera motions without re-training the model, which brings a much wider application for camera control in video generation.

#### 3 METHOD

Our MotionMaster model aims to disentangle the camera motion and object motion in a single or several videos, and then transfer the disentangled camera motion to the newly generated videos. We first observe that the temporal attention maps in diffusion-based video generation models contain the information of videos motions, and find that the motion are composed of two motion types, camera motions and object motions. We then propose two methods to decompose the temporal attention map Attn into object motion Attno and camera motion Attnc, as shown in Fig. 2. By substituting the temporal attention map with the temporal attention of the target camera motion, we can enable the video generation models to generate videos with the desired camera motion.

Specifically, to disentangle the camera motion from the object motion, we propose to extract the camera motions

from either a single video or a few (5-10) videos. 1) In oneshot camera motion disentanglement, we aim to extract camera motion from a single video (Fig. 2 top). Considering the motion in background region only contains camera motion, while motion in the foreground region contains both camera motion and object motion, we first separate background and foreground regions. We employ SAM [39] to segment the moving objects, and decompose the given video into moving object region M and background region M˜ = 1 − M. Then we regard the motion in the background region M˜ as only containing camera motion. With the observation that the camera motion is smooth and continuous, and the neighboring pixels share similar motions [40]–[43], we construct a Poisson equation to estimate the camera motions in the moving objects region M based on the given camera motions in the background region M˜ , achieving camera-object motion disentanglement for a single video.

2) When the object motions are too complex to disentangle from a single video, we propose a few-shot camera motion disentanglement method to extract common camera motion from m (5-10) videos with similar camera motions (Fig. 2 bottom). To extract common camera motion of m videos, we regard the common feature of the temporal attention maps of these videos as the feature of the common camera motion. We then propose a window-based clustering method for each pixel of the temporal attention map to extract the common camera motion and filter out outliers. Specifically, we regard the neighboring pixels in a k × k window share similar camera motions and cluster the k2neighboring pixels of each pixel in the m temporal attention maps with DBSCAN clustering method, where the centroid of the largest cluster can be used to represent the common camera motion.

Finally, we investigate the additivity and positional composition ability of camera motions. We propose a camera motion combination method to achieve flexible camera motion control, which can combine different camera motions into a new motion and apply different camera motions in different regions, substantially improving the controllability and flexibility of camera motions.

3.1 Camera Motion Extraction Based on Temporal Attention

Preliminaries of temporal attention module. Most of the current video generation models [3], [4], [22] are built on a pretrained text-to-image diffusion model [2], which employs spatial attention module to model the image generation process. To extend the image generation models to generate videos, temporal attention module [3], [22] is proposed to enable the pretrained image generation models with the ability to model the temporal relationship between each frame of the video. Specifically, the temporal attention mechanism is a self-attention module, which takes the feature map fin of t frames (b × t × c × h × w) as input, and reshapes it to a (b×h×w)×t×c feature map f. Then, a self-attention module is employed to capture the temporal relationships between t frames, and output a feature map with temporal relationships between each frame, which is formulated as follows:

QKT √c

), fout = AttnV, (1)

Attn = Softmax(

Successive Over Relaxation

Mask

Object Tracking

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Union

 . ., ​ =  ​

[Figure 33]

One-shot Camera Motion Disentangle

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

DDIM Inversion

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Disentangled Camera Motion Attention Map

Solving Poisson Equation

Inverted Latents

Masked Attention Map

Temporal Attention Map

Source Video

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Text Prompt

[Figure 52]

Camera Motion Transfer

Video Diffusion Model

Output Video with the target camera motion

Random Latents

[Figure 53]

DBSCAN Clustering

[Figure 54]

Centroid

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

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

DDIM Inversion

[Figure 74]

Few-shot Camera Motion Disentangle

[Figure 75]

T-SNE

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

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

DBSCAN Clustering

Centroid

Disentangled Camera Motion Attention Map

Video Gallery (n videos)

n×Inverted Latents

n×Temporal Attention Map

- Fig. 2: Main framework of our method: Our model can extract the camera motion from a single video or several videos that share similar camera motions. 1) One-shot camera motion disentanglement: We first employ SAM [39] to segment the moving objects in the source video and extract the temporal attention maps from the inverted latents. To disentangle the camera and object motions, we mask out the object areas of the attention map and estimate the camera motion inside the mask by solving a Poisson equation. 2) Few-shot camera motion disentanglement: we extract the common camera motion from the temporal attention maps of several given videos. For each position (x,y), we employ all of its k-neighboring attention map values across each video for clustering. Then, we use the centroid of the largest cluster to represent the camera motions in position (x,y).

where Q = WQf, K = WKf and V = WV f, and WQ, WK and WV are learnable query, key and value matrices.

video can usually be divided into two parts: the foreground region and the background region. Considering the motion in background region mainly contains camera motion, while the motion in foreground region contains both camera motion and object motion, we first extract camera motion in background region, and then predict camera motion in foreground based on the background camera motion.

Extracting motion information from temporal attention map. UniEdit [12] found that the temporal attention modules model the inter-frame dependency and motion information1, and use the temporal attention for video motion editing tasks, where the global motion of video is edited guided by text. However, it lacks a deep analysis of how the temporal attention module models the interframe dependency. In this paper, we find that the attention maps Attn of the temporal attention layer are composed of two motion types, which are camera motions and object motions. We propose two methods to decouple motion in temporal attention map into camera and object motions (Sec. 3.2 and 3.3), where we disentangle the temporal attention map Attn extracted from a video into camera motion attention Attnc and object motion attention Attno. After decoupling camera motion from object motion, we can easily transfer camera motion from a source video vs to a target video vt, by replacing the temporal attention map of vt with the temporal attention map Attncs that corresponds to the camera motion of vs:

Specifically, we first employ segment-anything model to segment the moving objects and the background, and then take the temporal attention map from the background region as the camera motions. Based on the observation that the camera motions are continuous and the neighboring pixels have similar camera motions, we construct a Poisson equation to estimate the camera motions inside the moving object region based on the camera motions outside, thereby achieving the camera-object motion disentanglement.

Obtaining temporal attention map of a video by DDIM inversion. First of all, to obtain the temporal attention map of the source video vs, we apply DDIM inversion on the source video to invert it into a T-step latent xT. Then, by denoising xT with the video diffusion model, we obtain a series of attention maps {AttnT,Attnt−1 ···Attn1} in different timesteps. Different from the spatial attention maps (in spatial attention modules), which model different spatial relationships in different timesteps, the temporal attention maps model the temporal motion of the video, and we find they are similar in different timesteps. Therefore, we can use one representative temporal attention map Attn = Attnt at timestep t to model the temporal motion, which can effectively reduce the computation resources to

fout = AttncsV. (2)

- 3.2 One-shot Camera Motion Disentanglement

In this section, we propose a method to disentangle camera motions from object motions in a single source video. A

1. Our experiments also validate this finding, shown in #Suppl.

1

T of using all timesteps. We adopt a medium timestep t, since when t is large, there are too many noises in the video

feature; while when t is small, the denoising has almost been completed and the overall motion has already been determined, thus the motion information in the temporal attention map at small t is not sufficient.

Extracting camera motion in background region. With the obtained temporal attention map Attn from the source video, we employ segment anything model (SAM) to obtain the mask of the moving objects in each frame Mi = SAM(vi),i = 1,··· ,t, where vi denotes the i-th frame of the source video vs. Then, we merge the masks of t frames into one mask M = U(M1,M2 ···Mt). Since the motion in the background region mainly comes from the camera motion, we regard the masked temporal attention map in the background region Attnm = Attn ⊙ (1 − M) as the camera motion attention map that only controls the camera motion. Although currently the masked attention map Attnm has no value inside the moving objects mask M, we can estimate the camera motion inside the mask based on the camera motion outside. To estimate the camera motion inside the mask M, we transform the motion estimation problem into solving a Poisson equation, which is introduced below.

Predicting camera motion in foreground region. Video processing tasks such as video compression, optical flow estimation, and video interpolation, share a common assumption that the changes between video frames are smooth and continuous [40]–[43], and the motions of the pixels in a local neighborhood are similar. Based on this assumption, we posit that the camera motion is also continuous and has local coherence, i.e., the camera motions in a local region are almost the same. Therefore, we assume the gradient of the camera motion attention map inside the mask region is quite small, and the values of the attention map on both sides of the mask boundary are almost the same. Denote the camera motion attention map inside the mask M as Ain (to be estimated), and the camera motion attention map outside the mask as Aout (which we already have Aout = Attnm). And we denote the positions of each pixel inside the mask as Ω ∈ R2, and the mask boundary as ∂Ω. Then, we have ∇Ain ≈ 0, and Ain|∂Ω = Aout|∂Ω. Since we already know Aout, we can estimate Ain by solving the following optimization problem:

A∗in = argmin

∥∇Ain∥2. s.t.Ain|∂Ω = Aout|∂Ω.

Ain Ω

(3)

Therefore, the camera-motion estimation problem is converted into a Poisson blending problem. By setting the gradient inside the mask to be 0, we can employ Successive Over Relaxation algorithm [44] for Poission Blending to find the optimal solution A∗in. Finally, we obtain the complete camera motion attention map Attnc = {Ain∗,Aout}, which is disentangled with the object motion. With the disentangled Attnc, we can employ the camera motion transfer method in Sec. 3.1 to transfer the camera motion from a single source video to target videos.

###### 3.3 Few-shot Camera Motion Disentanglement

When the object motions are overly complex to disentangle, e.g., moving objects may occupy nearly all the pixels, it may be difficult to disentangle camera motion and object motion from a single video. To improve the disentanglement performance for videos with complex object motions, we relax the input conditions from one shot to few shot. I.e., we aim to extract the common camera motion from several videos {v1,v2 ···vm} with similar camera motions.

Extracting common feature in temporal attention as common camera motion. In Sec. 3.2, we decompose the temporal attention maps of a single video into camera motion and object motion. Since the given m videos {v1,v2 ···vm} share similar camera motions, we regard the common feature of the temporal attention maps as the feature of camera motion. Therefore, we calculate common camera motion by extracting a common feature from the temporal attention maps of m videos. Since the motion at different locations may be different (e.g., zoom in/out), we model the motion at pixel level. Denote the temporal attention map of each video as {A1,A2 ···Am}, where Ai ∈ RW×H×t×t and t is the number of frames. For each pixel (x,y) in video vi, we denote its motion as Ai(x,y) ∈ Rt×t. Next, we aim to extract the common feature for each pixel (x,y) from m temporal attention maps.

Local coherence assumption for camera motion. To extract the common feature for each pixel (x,y), only using the attention values at the location (x,y) in m temporal attention maps may not be adequate, especially when the object motions in the given m video are complex and diverse. Therefore, based on the assumption of local coherence, we regard that the neighboring pixels in a window centered at pixel (x,y) share similar camera motion as the center pixel. In other words, we extract the common camera motion for the pixel (x,y) by considering the attention values of neighboring pixels in a k × k window Nk(x,y) in each of the m temporal attention maps (m × k2 pixels in total), whose attention values form a tensor A(x,y) = {Ai(Nk(x,y)),i = 1···m}∈ Rm×k

2×t×t.

Extracting common camera motion by window-based clustering. For each pixel (x,y), to extract the common camera motion from the attention values A(x,y) in its k ×k neighboring window, we first reshape the attention values A(x,y) to R(m×k

2)×(t×t). We then employ t-SNE [45] to reduce the dimension from (t × t) to 2, for better clustering in the subsequent steps. After dimension reduction, we compute the centroid of the m × k2 pixels as the representation of the common camera motion. Directly computing the mean value of all the m × k2 pixels is a possible solution to compute the centroid, but has inferior accuracy of the extracted motion when the camera motions in some of the samples are severely entangled with object motion. Therefore, we employ DBSCAN [46] to cluster all the pixels, which can effectively distinguish the outliers. After clustering, we have nc clusters, with each cluster containing part of the attention values. We regard the centroid of the largest cluster as the common camera motion, since it is the most common motion among the m × k2 pixels. With the extracted camera motion map Attnc, we can transfer the camera motions to new videos.

###### 3.4 Camera Motion Combination

Camera motion combination. In Sec. 3.2 and 3.3, we extract the camera motion Attnc from a single or several videos. These camera motions can work separately by transferring one extracted camera motion to a target video. One natural question is whether we can combine different camera motions to enable a more complex and flexible camera motion control. To achieve this, in this section, we explore different ways to combine camera motions, which enables 1) combining different camera motions into a new motion; 2) applying different camera motions in different areas; and 3) preserving part of the contents while transferring the camera motion.

Additivity of the camera motions. We first explore how to combine different camera motions together. We are delighted to discover that the camera motions extracted from Sec. 3.2 and 3.3 are additive. By adding the attention maps {Attnci}ni=1 corresponding to different camera motions, we can obtain a new camera motion that includes all the combined camera motions at the same time. And by assigning different weights {wi}ni=1 to different camera motions, we can control the intensity of each camera motion by:

Attncnew =

wi × Attnci, (4)

i∈Sub({1···n})

where Sub({1···n}) is an arbitrary subset of {1···n}.

Position-specified motion transfer. The camera motion transfer methods in previous sections can only transfer the camera motions in a global manner, while lacking the ability to transfer the camera motions in a local region. Therefore, to enable our model with the ability to control the camera motions in a local manner, we propose a segmentationbased local camera motion control method. We segment local regions by SAM, and assign different camera motions to different local regions of the generated video, by applying the mask Mi on the camera motion attention map Attnci as follows:

Attncnew =

Mi ⊙ Attnci. (5)

i

Local content-preserving camera motion transfer. To better preserve specific content within the target video, we first utilize SAM to segment the object region M we aim to keep unchanged and then modify the temporal attention calculation. We find that in diffusion-based video generation models, the appearance and motions are well disentangled in the temporal attention modules, where the temporal attention maps represent the temporal motions, while the Value V represents the appearance. Therefore, when we need to transfer the camera motions from a source video vs to a target video vt while keeping the appearance in region M of vt unchanged, we modify the temporal attention calculation by keeping the Value inside M the same as the Value Vt of the target video, and substituting the temporal attention map by the camera motion attention map Attncs of the source video, which can be formulated as follows:

V ′ = Vt ⊙ M + V ⊙ (1 − M), fout = AttncsV ′. (6)

4 EXPERIMENTS

- 4.1 Implementation Details

Experiment details and hyperparameters. In our experiments, we adopt AnimateDiff [3] as the baseline method for motion disentanglement and control, which is one of the state-of-the-art text-to-video models. The generated video size is 512 × 512, with each video composed of 16 frames with 8 FPS. When generating videos, we employ 25-step DDIM [47] for inference and choose the temporal attention maps in the 15-th step to extract the camera motions. Moreover, for few-shot camera motion extraction, we compute the neighborhood size k by k = ⌈size16 ⌉ × 2 + 1, where size is the width and height of the temporal attention maps.

Evaluation metrics. To evaluate the generation quality, diversity and camera motion accuracy, we employ three evaluation metrics: 1) FVD [48]: Fr´echet Video Distance measures the quality and authenticity by calculating the Fr´echet distance between real and generated videos; 2) FIDV [49]: Video-level FID uses a 3D Resnet-50 model to extract video features for video-level FID scoring, measuring the quality and diversity of the generated videos; and 3) Optical Flow Distance [50] assesses the camera movement accuracy by computing distance between the flow maps from the generated and ground truth videos.

- 4.2 Camera Motion Transfer

Qualitative comparison with the state-of-the-arts. To validate the effectiveness of our model, we compare our model with the state-of-the-art camera motion control methods on four types of basic camera motions: 1) zoom in, 2) zoom out,

- 3) pan left, and 4) pan right (in #Suppl). We compare with two motion control methods: 1) AnimateDiff [3] employs the temporal LoRA [15] module to learn the motions from given videos with target camera motions. We train motion LoRA modules on AnimateDiff with one-shot and few-shot data, and compare them with our model. 2) Moreover, we also compare with MotionCtrl [16]. Since the training code is not open-sourced, we employ the officially provided model, which is pretrained on a large scale of camera-labeled data.

The comparison results are shown in Fig. 3 (video comparison results are provided in #Suppl). It can be seen that in one-shot condition, AnimateDiff tends to overfit to the given video; while in the few-shot condition, AnimateDiff tends to mix the features of the training videos, which cannot generate correct videos corresponding to the given prompts. MotionCtrl can generate videos that better align with the prompts, but may cause shape distortions and logical inconsistencies when controlling camera motion. In contrast, our model can generate high-quality and diverse videos with only one-shot or few-shot data, without the need for training.

Quantitative comparison. We also compare with these models quantitatively, using FVD, FID-V, and Optical Flow distance to evaluate the generation quality, diversity, and camera motion accuracy. For each method, we generate 1,000 videos for each type of camera motion and compute FVD and FID-V with 1,000 collected high-quality videos. We also compute the average Optical Flow Distance between the generated videos and given videos. The results are

###### One-Shot Camera Motion Transfer

###### One-Shot Camera Motion Transfer

###### Few-Shot Camera Motion Transfer

Prompt：A beekeeper inspecting hives in an apiary

###### Prompt：A group of friends birdwatching in a nature reserve

Prompt：A florist arranging flowers in the flower shop

[Figure 94]

[Figure 95]

[Figure 96]

Pan Right Zoom In

Zoom Out

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

Source Video

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

Ours

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

AnimateDiff +Lora

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

MotionCtrl (Trained on Large-Scale Dataset)

- Fig. 3: The comparison on one-shot and few-shot camera motion transfer with AnimateDiff+Lora [3] and MotionCtrl [16]. AnimateDiff+Lora tends to overfit to the training data while MotionCtrl suffers from shape distortions and logical inconsistencies when controlling camera motion, even though it is trained on large-scale data. In contrast, our MotionMaster generates high-quality videos with accurate camera motions.

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

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

Zoom In+Pan Right Zoom In+Pan Left Zoom Out+Pan Right Zoom Out+Pan Left

[Figure 190]

[Figure 191]

[Figure 192]

- Fig. 4: Camera motion combination results: The extracted camera motions can be combined to form new camera motions. The newly constructed camera motions in this figure contain both zoom and pan camera motions at the same time.

shown in Tab. 1, where our model achieves the best FIDV and FVD, demonstrating superior generation quality and diversity. Since AnimateDiff overfits to the training data, it get a lower Flow distance, but suffers from the worst generation diversity. In summary, our model achieves the best FVD and FID-V, while also ensuring a good camera transfer accuracy compared to MotionCtrl.

###### 4.3 Flexible Motion Control

Motion combination. In this section, we evaluate the additivity of our disentangled camera motion attention maps. We employ the extracted camera motions including zoom in, zoom out, pan left and pan right in Sec. 4.2 and combine two of them into a new camera motion by Eq.(4). The results are shown in Fig. 4. It can be seen that when combining the zooming motions with the panning motions, the camera zooms and pans at the same time, which demonstrates

that our model can successfully combine different kinds of camera motions together while ensuring generation quality.

More professional camera motions. In this section, we show more professional camera motions in the real film industry, including variable-speed zoom and dolly zoom. For variable-speed zoom, where the camera firstly zooms in fast and then zooms in slowly, we crop a video clip from films with this kind of motion, and achieve this motion control by one-shot camera motion disentanglement (Sec. 3.2). For dolly zoom, where the camera in the background region zooms while the camera in the foreground fixes, we employ the local content-preserving camera motion transfer method (Sec. 3.4) to realize it. The results are shown in Fig. 5. It can be seen that our model transfers the variable-speed zoom motion in the given video well, and achieves good generation results in both dolly zoom in and dolly zoom out motion controls.

###### Prompt:A family of der grazing peacefuly ina meadow

- TABLE 1: Quantitative comparison results with the state-ofthe-art methods on FVD, FID-V and Optical Flow Distance. Note that AnimateDiff+Lora [3] overfits to the training data, thereby achieving the lowest flow distance. But FVD and FID-V demonstrate its worst generation diversity. In contrast, our model achieves the best FVD and FID-V, while also ensuring a good camera transfer accuracy compared to MotionCtrl [16].

Zoom Out

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

MotionMaster

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

MotionMaster with Average Attention

Data and Method Pan Right Zoom In Data Scale Method FID-V ↓ FVD ↓ Flow Dis ↓ FID-V ↓ FVD ↓ Flow Dis ↓

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

AnimateDiff 382.40 4956.42 19.76 482.58 6322.46 6.91 MotionMaster 54.45 921.95 37.92 61.45 863.24 12.11 Large Scale MotionCtrl 95.83 1207.52 38.18 80.58 935.08 13.12

One shot

MotionMaster w/o Window

- (a) Comparison results on one-shot camera motion control.

Data and Method Pan Left Zoom Out Data Scale Method FID-V ↓ FVD ↓ Flow Dis ↓ FID-V ↓ FVD ↓ Flow Dis ↓

Few shot

AnimateDiff 268.29 4629.08 14.76 251.44 3975.41 3.12 MotionMaster 61.38 1092.09 38.94 52.90 910.76 5.10

Large Scale MotionCtrl 98.04 1196.54 55.25 80.12 928.41 7.88

- (b) Comparison results on few-shot camera motion control.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

MotionMaster w/o Dimension Reduction

Zoom In Fast Zoom In Slowly

- Fig. 6: Ablation study on one-shot camera motion disentanglement. The model without motion disentanglement generated artifacts in the region of the moving rabbit.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Source Video With Masks

Prompt:A be polinating flowers ina vibrant garden

MotionMaster

MotionMaster w/o Motion Disentangle

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Pan Right

- Fig. 7: Ablation study on few-shot camera motion disentanglement. All the ablated models generate videos with unnatural movements shown in the red boxes which are caused by the inaccurate extracted camera motions.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Source Video with VariableSpeed Zoom

Pompt:A charming vlage

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

quare witha bustling market

Prmpt:A peacful tranqui with ponds and flowers

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Dolloy Zoom

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Source Video with Masks

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Prompt:A painter capturinga scenic landscape on canvas

Dolly Zoom Out

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Prmpt:A fmly of der grazg peacfuly ina meadow

Dolly Zoom In

ablated versions on zoom-out camera motion: 1) MotionMaster with Average Attention: the model without DBSCAN clustering and directly averages the camera motions from all the videos; 2) MotionMaster w/o Window: the model without the window-based clustering, which only uses the m pixels at the same location for clustering; and 3) MotionMaster w/o Dimension Reduction: the model without t-SNE to reduce the dimension. The comparison results are shown in Fig. 7. It can be seen that all the ablated models generate unnatural movements shown in the red boxes where certain objects abruptly appear or vanish, or suffer from shape distortions. In contrast, our model achieves the highest generation quality and transfers the camera motions correctly.

- Fig. 5: Camera motion control results on professional camera motions, including variable-speed zoom and dolly zoom.

###### 4.4 Ablation Study

Ablation on one-shot camera motion disentanglement. We first validate the effectiveness of our one-shot camera motion disentanglement method. We compare our model with the ablated version that directly transfers the temporal attention map from the source video to the target video, which does not disentangle the camera and object motions. The results are shown in Fig. 6. It can be seen that when transferring the pan right camera motion entangled with the object motion of the moving rabbit, the model without motion disentanglement tends to generate artifacts in the region of the rabbit, which is clearer in the video of #Suppl.

#### 5 CONCLUSION

In this paper, we propose MotionMaster, a training-free camera motion transfer method based on camera-motion disentanglement. We find that the temporal attention map in the video diffusion model is composed of both camera

Ablation on few-shot camera motion disentanglement. We then validate the effectiveness of our few-shot camera motion disentanglement method. We experiment on three

motion and object motion. We then propose two methods to disentangle the camera motions from object motions for a single or several videos. Moreover, with the extracted camera motions, we further propose a camera motion combination method to enable our model a more flexible and controllable camera control. Extensive experiments demonstrate the superior camera motion transfer ability of our model and show our great potential in controllable video generation.

#### REFERENCES

- [1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020. 1
- [2] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695. 1, 3
- [3] Y. Guo, C. Yang, A. Rao, Y. Wang, Y. Qiao, D. Lin, and B. Dai, “Animatediff: Animate your personalized text-to-image diffusion models without specific tuning,” arXiv preprint arXiv:2307.04725,

2023. 1, 2, 3, 6, 7, 8, 11, 12, 13, 14

- [4] A. Blattmann, T. Dockhorn, S. Kulal, D. Mendelevitch, M. Kilian, D. Lorenz, Y. Levi, Z. English, V. Voleti, A. Letts et al., “Stable video diffusion: Scaling latent video diffusion models to large datasets,” arXiv preprint arXiv:2311.15127, 2023. 1, 2, 3
- [5] U. Singer, A. Polyak, T. Hayes, X. Yin, J. An, S. Zhang, Q. Hu, H. Yang, O. Ashual, O. Gafni et al., “Make-a-video: Text-to-video generation without text-video data,” arXiv preprint arXiv:2209.14792, 2022. 1, 2
- [6] H. Chen, Y. Zhang, X. Cun, M. Xia, X. Wang, C. Weng, and Y. Shan, “Videocrafter2: Overcoming data limitations for highquality video diffusion models,” arXiv preprint arXiv:2401.09047,

2024. 1, 2

- [7] H. Chen, M. Xia, Y. He, Y. Zhang, X. Cun, S. Yang, J. Xing, Y. Liu, Q. Chen, X. Wang et al., “Videocrafter1: Open diffusion models for high-quality video generation,” arXiv preprint arXiv:2310.19512,

2023. 1, 2

- [8] T.-S. Chen, C. H. Lin, H.-Y. Tseng, T.-Y. Lin, and M.-H. Yang, “Motion-conditioned diffusion model for controllable video synthesis,” arXiv preprint arXiv:2304.14404, 2023. 1, 3, 11
- [9] S. Tu, Q. Dai, Z.-Q. Cheng, H. Hu, X. Han, Z. Wu, and Y.-G. Jiang, “Motioneditor: Editing video motion via content-aware diffusion,” arXiv preprint arXiv:2311.18830, 2023. 1, 3
- [10] C. Chen, J. Shu, L. Chen, G. He, C. Wang, and Y. Li, “Motion-zero: Zero-shot moving object control framework for diffusion-based video generation,” arXiv preprint arXiv:2401.10150, 2024. 1, 3
- [11] S. Yang, L. Hou, H. Huang, C. Ma, P. Wan, D. Zhang, X. Chen, and J. Liao, “Direct-a-video: Customized video generation with user-directed camera movement and object motion,” arXiv preprint arXiv:2402.03162, 2024. 1, 3
- [12] J. Bai, T. He, Y. Wang, J. Guo, H. Hu, Z. Liu, and J. Bian, “Uniedit: A unified tuning-free framework for video motion and appearance editing,” arXiv preprint arXiv:2402.13185, 2024. 1, 3, 4
- [13] C. Qi, X. Cun, Y. Zhang, C. Lei, X. Wang, Y. Shan, and Q. Chen, “Fatezero: Fusing attentions for zero-shot text-based video editing,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 15932–15942. 1, 3
- [14] J. Z. Wu, Y. Ge, X. Wang, S. W. Lei, Y. Gu, Y. Shi, W. Hsu, Y. Shan, X. Qie, and M. Z. Shou, “Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7623–7633. 1
- [15] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021. 1, 3, 6
- [16] Z. Wang, Z. Yuan, X. Wang, T. Chen, M. Xia, P. Luo, and Y. Shan, “Motionctrl: A unified and flexible motion controller for video generation,” arXiv preprint arXiv:2312.03641, 2023. 1, 3, 6, 7, 8, 11, 12, 13, 14
- [17] C. Vondrick, H. Pirsiavash, and A. Torralba, “Generating videos with scene dynamics,” Advances in neural information processing systems, vol. 29, 2016. 2

- [18] T.-C. Wang, M.-Y. Liu, A. Tao, G. Liu, J. Kautz, and B. Catanzaro, “Few-shot video-to-video synthesis,” arXiv preprint arXiv:1910.12713, 2019. 2
- [19] M. Saito, E. Matsumoto, and S. Saito, “Temporal generative adversarial nets with singular value clipping,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2830–

2839. 2

- [20] J. Zhang, C. Xu, L. Liu, M. Wang, X. Wu, Y. Liu, and Y. Jiang, “Dtvnet: Dynamic time-lapse video generation via single still image,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16. Springer, 2020, pp. 300–315. 2
- [21] R. Girdhar, M. Singh, A. Brown, Q. Duval, S. Azadi, S. S. Rambhatla, A. Shah, X. Yin, D. Parikh, and I. Misra, “Emu video: Factorizing text-to-video generation by explicit image conditioning,” arXiv preprint arXiv:2311.10709, 2023. 2
- [22] A. Blattmann, R. Rombach, H. Ling, T. Dockhorn, S. W. Kim, S. Fidler, and K. Kreis, “Align your latents: High-resolution video synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 22563–22575. 2, 3
- [23] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, “Video diffusion models,” Advances in Neural Information Processing Systems, vol. 35, pp. 8633–8646, 2022. 2
- [24] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., “Imagen video: High definition video generation with diffusion models,” arXiv preprint arXiv:2210.02303, 2022. 2
- [25] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10684–10695. 3
- [26] D. Zhou, W. Wang, H. Yan, W. Lv, Y. Zhu, and J. Feng, “Magicvideo: Efficient video generation with latent diffusion models,” arXiv preprint arXiv:2211.11018, 2022. 3
- [27] J. Wang, H. Yuan, D. Chen, Y. Zhang, X. Wang, and S. Zhang, “Modelscope text-to-video technical report,” arXiv preprint arXiv:2308.06571, 2023. 3
- [28] W. Chen, J. Wu, P. Xie, H. Wu, J. Li, X. Xia, X. Xiao, and L. Lin, “Control-a-video: Controllable text-to-video generation with diffusion models,” arXiv preprint arXiv:2305.13840, 2023. 3
- [29] P. Esser, J. Chiu, P. Atighehchian, J. Granskog, and A. Germanidis, “Structure and content-guided video synthesis with diffusion models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 7346–7356. 3
- [30] Y. Guo, C. Yang, A. Rao, M. Agrawala, D. Lin, and B. Dai, “Sparsectrl: Adding sparse controls to text-to-video diffusion models,” arXiv preprint arXiv:2311.16933, 2023. 3
- [31] Y. Wei, S. Zhang, Z. Qing, H. Yuan, Z. Liu, Y. Liu, Y. Zhang, J. Zhou, and H. Shan, “Dreamvideo: Composing your dream videos with customized subject and motion,” arXiv preprint arXiv:2312.04433,

2023. 3

- [32] H. Jeong, G. Y. Park, and J. C. Ye, “Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models,” arXiv preprint arXiv:2312.00845, 2023. 3
- [33] Y. Jain, A. Nasery, V. Vineet, and H. Behl, “Peekaboo: Interactive video generation via masked-diffusion,” arXiv preprint arXiv:2312.07509, 2023. 3
- [34] Y. Teng, E. Xie, Y. Wu, H. Han, Z. Li, and X. Liu, “Draga-video: Non-rigid video editing with point-based interaction,” arXiv preprint arXiv:2312.02936, 2023. 3
- [35] R. Wu, L. Chen, T. Yang, C. Guo, C. Li, and X. Zhang, “Lamp: Learn a motion pattern for few-shot-based video generation,” arXiv preprint arXiv:2310.10769, 2023. 3
- [36] R. Zhao, Y. Gu, J. Z. Wu, D. J. Zhang, J. Liu, W. Wu, J. Keppo, and M. Z. Shou, “Motiondirector: Motion customization of textto-video diffusion models,” arXiv preprint arXiv:2310.08465, 2023. 3
- [37] X. Wang, H. Yuan, S. Zhang, D. Chen, J. Wang, Y. Zhang, Y. Shen, D. Zhao, and J. Zhou, “Videocomposer: Compositional video synthesis with motion controllability,” Advances in Neural Information Processing Systems, vol. 36, 2024. 3
- [38] Y. Deng, R. Wang, Y. Zhang, Y.-W. Tai, and C.-K. Tang, “Dragvideo: Interactive drag-style video editing,” arXiv preprint arXiv:2312.02216, 2023. 3
- [39] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment

- anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026. 3, 4
- [40] S. Zhou, X. Jiang, W. Tan, R. He, and B. Yan, “Mvflow: Deep optical flow estimation of compressed videos with motion vector prior,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 1964–1974. 3, 5
- [41] D. Fleet and Y. Weiss, “Optical flow estimation,” in Handbook of mathematical models in computer vision. Springer, 2006, pp. 237–

257. 3, 5

- [42] H. Zhang, D. Liu, Q. Zheng, and B. Su, “Modeling video as stochastic processes for fine-grained video representation learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 2225–2234. 3, 5
- [43] Z. Gharibi and S. Faramarzi, “Multi-frame spatio-temporal superresolution,” Signal, Image and Video Processing, vol. 17, no. 8, pp. 4415–4424, 2023. 3, 5
- [44] D. Young, “Iterative methods for solving partial difference equations of elliptic type,” Transactions of the American Mathematical Society, vol. 76, no. 1, pp. 92–111, 1954. 5
- [45] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal of machine learning research, vol. 9, no. 11, 2008. 5
- [46] M. Ester, H.-P. Kriegel, J. Sander, X. Xu et al., “A density-based algorithm for discovering clusters in large spatial databases with noise,” in kdd, vol. 96, no. 34, 1996, pp. 226–231. 5, 11
- [47] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020. 6
- [48] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Towards accurate generative models of video: A new metric & challenges,” arXiv preprint arXiv:1812.01717, 2018. 6, 11
- [49] Y. Balaji, M. R. Min, B. Bai, R. Chellappa, and H. P. Graf, “Conditional gan with discriminative filter generation for text-tovideo synthesis.” in IJCAI, vol. 1, no. 2019, 2019, p. 2. 6, 11
- [50] G. Farneb¨ack, “Two-frame motion estimation based on polynomial expansion,” in Image Analysis: 13th Scandinavian Conference, SCIA 2003 Halmstad, Sweden, June 29–July 2, 2003 Proceedings 13. Springer, 2003, pp. 363–370. 6, 11
- [51] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” arXiv preprint arXiv:2010.02502, 2020. 11

#### APPENDIX

In this supplementary material, more details about the proposed MotionMaster and more experimental results are provided, including:

- • More implementation details (Sec. A);
- • Solving Poisson Equation (Sec. A);
- • Temporal attention maps determines the video motion (Sec. A)
- • More Comparison Results (Sec. A);
- • More Experiments on the hyperparameters (Sec. A);
- • User Study (Sec. A).

To see the generated results more clearly, you can refer to https://sjtuplayer.github.io/projects/MotionMaster, which includes all the videos in the experiments.

We conduct experiments based on AnimateDiff-v2 [3]. we use DDIM [51] to accelerate the generation process with 25 denoising steps. Moreover, to decrease the computation cost, we employ the temporal attention maps in timestep t = 15 to represent the video motions in different timesteps as illustrated in Sec. 3.2 of the main paper. Furthermore, for few-shot camera motion disentanglement, we specified a video count of 5 and configured DBSCAN clustering [46] with an Eps-neighborhood of 4 and core points of 3.

We complete the temporal attention map inside the object-moving region by solving a Poisson equation. The gradients within object-moving regions of the completed attention map are assumed to be zero and the boundary values should match those of the original attention map. We choose the parallel red-black ordering Gauss-Seidel iteration method to solve the Poisson equation. Initially, we label the pixels with red-black ordering, ensuring that each pixel and its neighboring pixels alternate between being labeled red and black. Next, while ensuring that the values of boundary nodes remain unchanged, we update the red and black pixels alternately until reaching a specified number of iterations or until the residual falls below a predefined threshold. The iteration process is illustrated by the pseudo code. This algorithm can be accelerated using parallel computing frameworks like CUDA.

The foundation of our method comes from the observation that the temporal attention map determines the motions in the generated videos, including camera motions and object motions. To validate this, we conduct an experiment to swap the temporal attention maps between two videos, where one of them contains only camera motion while the other one contains only object motion. The results are shown in Fig. 8. It can be seen that after swapping the temporal attention map, the contents of the two videos are similar and the motions are totally swapped. The source videos of

- (a) keep the camera fixed while moving the bus from left to right and (b) keep the object fixed while zooming out the camera. After swapping the temporal attention maps, the second row of (a) keeps the bus fixed while zooming out the camera and (b) keeps the camera fixed, but a shadow of a bus moves from left to right. Therefore, the temporal attention maps determine both the camera and object motions and by swapping the temporal attention map, the motions can be transferred to a new video.

Qualitative comparison. In this section, we show more results on one-shot and few-shot camera motion transfer

Algorithm 1 Solving Poisson Equation

- 1: function POISSON SOLVING(u, f)

- 2: u: RGB, f: gradient
- 3: choose an initial guess u(0)
- 4: while not converge do:
- 5: for (i, j) is red node do:
- 6: u(i,jk+1) = 14(fi,j + u(i+1k) ,j + u(i−k)1,j + u(i,jk)+1 + u(i,jk)−1)

- 7: end for
- 8: for (i,j) is black node do:
- 9: u(i,jk+1) = 14(fi,j + u(i+1k+1),j + u(i−k+1)1,j + u(i,jk+1)+1 + u(i,jk+1)−1 )

- 10: end for
- 11: end while
- 12: end function

results, where both one-shot and few-shot methods are employed to transfer zoom-in, zoom-out, pan-left, and panright camera motions. The qualitative comparison results are shown in Fig. 9 and 10. In the one-shot scenario, AnimateDiff+Lora [3] appears prone to overfitting to the provided video, whereas in the few-shot scenario, it tends to amalgamate features from the training videos, leading to inaccurate video generation in response to the given prompts. MotionCtrl [16] exhibits improved alignment with prompts in video generation; however, it may introduce shape distortions and logical inconsistencies in camera motion control. In contrast, our model achieves high-quality and high-diversity generation with only one-shot or fewshot data, without the need for training.

Quantitative comparison. To further validate the effectiveness of our model, we conduct quantitative comparisons on the four basic camera motions with one-shot and fewshot data. The comparison results are shown in Tab. 2. It shows that our model achieves the best FVD [48] and FIDV [49] scores, indicating the best generation quality and diversity of our model. Since Animatediff is overfitted to the training data, it has the minimum optical flow distance [50], but it suffers from much worse FVD and FID-V. In summary, our model achieves the best FVD and FID-V, while also ensuring a good camera transfer accuracy compared to MotionCtrl. (Note that the header of Tab. 1(b) in the main paper should be ”Pan Left” and ”Zoom Out” and Tab. 2 here is the correct version)

Comparison on the computation cost. Moreover, we also compare the computation cost including computation time and GPU memory requirement between MotionMaster, Animatediff+Lora [3] and MotionCtrl [8]. Since our model is a training-free method, we compute the time for disentangling the camera-object motions as our training time. To ensure the fairness of the experiment, we compute the time on the same NVIDIA A100 GPU. Meanwhile, we compare the GPU memory required for all the methods. The comparison results are shown in Tab. 3. It can be seen that our model can accomplish camera motion disentanglement in a few minutes while the other methods require a much longer training time. Moreover, both AnimateDiff+Lora and MotionCtrl require more than 30G GPU memory, while our model only needs 13G GPU memory which is the only

###### (a) Prompt:A yelow schol bus puling intoa driveway (b) Prompt:A countryside barn surouned by sunflowers

12

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

Source Videos

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Swap Temporal Attention Map

- Fig. 8: After swapping the temporal attention maps of the first-row videos, we have the second-row videos which swap the motions. The source videos of (a) moves the bus while fixing the camera, and (b) keeps the objects fixed while zooming out the camera. After swapping the temporal attention map, (a) keeps the bus fixed while zooming out the camera. And

- (b) fixed camera, but there is a shadow of a moving bus in the generated video (it is clearer in demo.html).

- TABLE 2: Quantitative comparison results with the state-of-the-art methods on FVD, FID-V, and Optical Flow Distance. Note that AnimateDiff+Lora [3] overfits to the training data, thereby achieving the lowest flow distance. But FVD and FID-V demonstrate its worst generation diversity. In contrast, our model achieves the best FVD and FID-V, while also ensuring a good camera transfer accuracy compared to MotionCtrl [16].

Camera Motion Pan Right Pan Left Zoom In Zoom Out Data Scale Method FVD ↓ FID-V ↓ Flow Dis ↓ FVD ↓ FID-V ↓ Flow Dis ↓ FVD ↓ FID-V ↓ Flow Dis ↓ FVD ↓ FID-V ↓ Flow Dis ↓

One shot

Animatediff 382.4 4956.42 19.76 382.04 5939.96 15.22 482.58 6322.46 6.91 396.96 7767.33 5.78

MotionMaster (ours) 54.45 921.95 37.92 64.43 933.77 35.64 61.45 863.24 12.11 55.23 862.9 6.93 Largescale MotionCtrl 95.83 1207.52 38.18 98.04 1196.54 55.25 80.58 935.08 13.12 80.12 928.41 7.88

- (a) Comparison results on one-shot camera motion control. Bold and underline represent optimal and sub-optimal results, respectively.

Camera Motion Pan Right Pan Left Zoom In Zoom Out Data Scale Method FVD ↓ FID-V ↓ Flow Dis ↓ FVD ↓ FID-V ↓ Flow Dis ↓ FVD ↓ FID-V ↓ Flow Dis ↓ FVD ↓ FID-V ↓ Flow Dis ↓

Few shot

Animatediff 290.86 5198.78 25.61 268.29 4629.08 14.76 281.73 4333.26 5.72 251.44 3975.41 3.12

MotionMaster (ours) 55.94 1153.27 35.98 61.38 1092.09 38.94 51.97 847.08 12.93 52.90 910.76 5.10 Largescale MotionCtrl 95.83 1207.52 38.18 98.04 1196.54 55.25 80.58 935.08 13.12 80.12 928.41 7.88

- (b) Comparison results on few-shot camera motion control. Bold and underline represent optimal and sub-optimal results, respectively.

- TABLE 3: Comparison on the computation resources. Our training-free MotionMaster requires much less time to control the camera motions and it is the only method that is capable of running on a single NVIDIA 24G 3090/4090 GPU.

TABLE 4: User Study from 28 volunteers.

Method Animatediff+Lora MotionCtrl MotionMaster (Ours) Percentage of Ranking First (%) ↑

0.65% 25.22% 74.13% Average Rank ↓ 2.93 1.80 1.27

Method Computation Time GPU Memory Animatediff+Lora [3] ≈10 hours 52G

(t ≤ 3), the generated video cannot be correctly generated since the temporal attention map contains too little motion information, which fails to guide the video generation process with accurate camera motions. The intermediate timesteps 5 < t < 20 all generate good results. Therefore, we choose timestep t = 15 as our default hyperparameter.

MotionCtrl [16] > 10 days 32G One-shot MotionMaster (Ours) ≈ 60 seconds 13G Few-shot MotionMaster (Ours) ≈150 seconds 13G

In this section, we conduct a user study to evaluate the effectiveness of our MotionMaster. We have invited 28 volunteers in related research areas to rank the generated results from AnimateDiff+Lora [3], MotionCrtl [16] and our MotionMaster considering the generation quality, diversity, and the camera transfer accuracy. Specifically, each volunteer ranked 20 sets of results, where each basic camera motion (pan left, pan right, zoom in, and zoom out) contains 5 videos. We compute the average ranking and the percentage of ranking first of the three methods, which is shown in Tab. 4. It can be seen that our model ranks first in 74.13% situations and achieves the best average rank of 1.27, demonstrating the superior performance of our MotionMaster in camera motion transfer.

method that can be implemented on a single 24G 3090/4090 GPU.

In section 3.2 of the main paper, we propose that we can employ the temporal attention map in one intermediate step t to represent the motions in different timesteps. We find that the timestep t cannot either be too large or too small, since the temporal attention map in a too large t contains too much noise and the temporal attention map in a too small t contains little temporal information. To validate this, we conduct one-shot camera motion transfer experiments on different timesteps t, which are shown in Fig. 11. It can be seen that when t is too large (t ≥ 22), the output videos suffer from heavy artifacts due to the noise in the temporal attention map. And when t is too small

[Figure 261]

[Figure 262]

##### One-Shot Camera Motion Transfer

Prompt：A beekeeper inspecting hives in an apiary

Pan Right Zoom In

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Source Video

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

Ours

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

AnimateDiff +Lora

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

MotionCtrl (Trained on Large-Scale Dataset)

[Figure 295]

[Figure 296]

One-Shot Camera Motion Transfer

Prompt：A yellow school bus pulling into a driveway

Pan Left Zoom Out

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Source Videos

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

Ours

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

AnimateDiff +Lora

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

MotionCtrl (Trained on Large-Scale Dataset)

- Fig. 9: Comparison on one-shot camera motion transfer on four basic camera motions: pan right, pan left, zoom in and zoom out. AnimateDiff+Lora [3] overfits the training video. Even if MotionCtrl [16] is trained on a large-scale dataset, it still suffers from inaccurate camera motion control and some artifacts in the generated videos. In contrast, our model accurately transfers the camera motions while ensuring good generation quality and diversity.

[Figure 329]

[Figure 330]

###### Few-Shot Camera Motion Transfer

Prompt：A magical fantasy forest with a hidden path

Pan Right Zoom In

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

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

[Figure 347]

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

Source Video

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

Ours

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

AnimateDiff +Lora

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

MotionCtrl (Trained on Large-Scale Dataset)

Few-Shot Camera Motion Transfer

[Figure 387]

[Figure 388]

Prompt：A florist arranging flowers in the flower shop

Pan Left Zoom Out

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

Source Videos

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

Ours

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

AnimateDiff +Lora

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

MotionCtrl (Trained on Large-Scale Dataset)

- Fig. 10: Comparison on few-shot camera motion transfer on four basic camera motions: pan right, pan left, zoom in and zoom out. AnimateDiff+Lora [3] overfits to the training videos, which generate videos with mixed features from the training data. Even if MotionCtrl [16] is trained on a large-scale dataset, it still suffers from inaccurate camera motion control and some artifacts in the generated videos. In contrast, our model accurately transferred the camera motions while ensuring good generation quality and diversity.

### (b)A peaceful vineyard wit rows of grapevines and roling hils

### (a)A charming seaside vilge with clorful fishing boats

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

t = 24

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

t = 22

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

t = 20

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

t = 15 (Ours)

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

t = 10

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

t = 5

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

t = 3

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

t = 1

- Fig. 11: Ablation on the hyperparameter timestep t. When it t is too large (t ≥ 22), there is too much noise in the temporal attention map, which causes the artifacts in the generated videos. When t is too small (t ≤ 3), the latent zt is too close to the denoised z0, where the temporal attention module contains less motion information. Therefore, the camera motions in the generated results are not as obvious as others. We choose the medium timestep t = 15, whose temporal attention maps capture the video motions accurately.

