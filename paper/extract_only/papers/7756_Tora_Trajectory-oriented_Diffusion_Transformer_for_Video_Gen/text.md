## arXiv:2407.21705v4[cs.CV]14Mar2025

### Tora: Trajectory-oriented Diffusion Transformer for Video Generation

Zhenghao Zhang1*, Junchao Liao1*, Menghao Li1, ZuoZhuo Dai1, Bingxue Qiu1, Siyu Zhu2, Long Qin1, Weizhi Wang1 1 Alibaba Cloud Computing 2 Fudan University

|[Figure 1]|
|---|

|[Figure 2]|
|---|

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

A delicate porcelain teacup floats upward on a linen tablecloth.

A stylish fox wearing sunglasses walks across the whimsical landscape of a candy-filled Mars.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

A warm rustic lantern floats upward in a dimly lit room.

A vibrant apple bobs gently up and down on a hanging tree branch during a calm autumn afternoon.

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Butterflies flutter over rocks, set against an erupting volcano.

Two roses, one purple, one yellow, sway together before a snow-covered mountain range.

Visual & Trajectory

Figure 1. Tora is capable of generating videos guided by arbitrary trajectories, images, texts, or combinations thereof. Our proposed motion modules integrate seamlessly with the scalability of DiT, ensuring that the generated movements not only adhere precisely to the specified trajectory but also effectively emulate physical world dynamics.

#### Abstract

Recent advancements in Diffusion Transformer (DiT) have demonstrated remarkable proficiency in producing highquality video content. Nonetheless, the potential of transformer-based diffusion models for effectively generating videos with controllable motion remains an area of limited exploration. This paper introduces Tora, the first trajectory-oriented DiT framework that concurrently integrates textual, visual, and trajectory conditions, thereby enabling scalable video generation with effective motion guidance. Specifically, Tora consists of a Trajectory Extractor (TE), a Spatial-Temporal DiT, and a Motion-guidance Fuser (MGF). The TE encodes arbitrary trajectories into hierarchical spacetime motion patches with a 3D motion compression network. The MGF integrates the motion patches into the DiT blocks to generate consistent videos that accurately follow designated trajectories. Our design aligns seamlessly with DiT’s scalability, allowing precise control of video content’s dynamics with di-

*Equal contribution

verse durations, aspect ratios, and resolutions. Extensive experiments demonstrate that Tora excels in achieving high motion fidelity compared to the foundational DiT model, while also accurately simulating the complex movements of the physical world. Code is made available at https://github.com/alibaba/Tora .

#### 1. Introduction

Diffusion models [10, 32] have demonstrated their capability to generate diverse and high-quality images or videos. Previously, video diffusion models [4, 14, 51] predominantly employed UNet-based architectures [34], focusing primarily on synthesizing videos of limited duration, typically around two seconds, and were constrained to fixed resolutions and aspect ratios. Recently, Sora [5], a text-to-video generation model leveraging Diffusion Transformer (DiT) [26], has showcased video generation capabilities that significantly outstrip current state-of-the-art methods. Sora excels not only in the production of high-quality videos ranging from 10 to 60 seconds, but also distinguishes

|[Figure 31]<br><br>|
|---|

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

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

A Chinese woman in traditional attire is turning from side to side.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

On a bright, sunny day, two adorable kittens walk side by side along the golden sands of a serene beach.

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

A crucian carp swims gracefully across the red, rocky surface of Mars.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Two rabbits in diving gear explore an underwater world of colorful corals and playful fish.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

A flock of seagulls soars gracefully through the vibrant underwater world filled with colorful coral reefs.

[Figure 69]

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

A dynamic aerial shot showcasing a mountain waterfall cascades down in the early morning light.

- Figure 2. More generated samples. Tora effectively manages trajectories to precisely manipulate various objects and backgrounds. In the realm of image-to-video synthesis, it can craft dynamic camera movements in accordance with textual descriptions and the designated starting trajectory points, such as common backgrounds. Furthermore, Tora supports video generation across different aspect ratios, resolutions, and durations, ensuring flexible content creation.

itself through its capacity to handle diverse resolutions, various aspect ratios, adherence to the laws of actual physics.

Video generation requires consistent motion across image sequences, underscoring the importance of motion control. Previous works, such as VideoComposer [39] and DragNUWA [48], have implemented generalized motion manipulation through motion vectors and trajectories. Building on this foundation, MotionCtrl [41] innovates by independently managing camera and object motions,

thereby expanding the diversity of achievable motion patterns. Despite their promising controllable motion quality, UNet-based methods are restricted to generating videos of only 16 frames at a fixed, lower resolution. This limitation hinders the smooth portrayal of motion, particularly during significant positional shifts in the provided trajectory, leading to distortion and unnatural movements, such as parallel drifting, which diverge from real-world dynamics. Consequently, there is an urgent need for a model capable of

producing longer videos with robust motion control and detailed physical representations.

To address these challenges, we present Tora, the first DiT model that simultaneously integrates text, images, and trajectories, enabling scalable video generation with robust motion control. Notably, our work adopts OpenSora [55], an open-source version of Sora, as the foundational DiT model. To align motion control with the scalability of the DiT framework, we propose two novel modules: the Trajectory Extractor (TE), which converts arbitrary trajectories into hierarchical spacetime motion patches, and the Motionguidance Fuser (MGF), designed to seamlessly integrate these patches within the stacked DiT blocks. More specifically, TE initially converts positional displacements along trajectory into the RGB domain via flow visualization techniques. These visualized displacements undergo Gaussian filtering to mitigate scattered issues. Subsequently, a 3D Variational Autoencoder (VAE) [19] encodes trajectory visualizations into spacetime motion latents, which share the same latent space with video patches. The motion latents are then decomposed into multiple levels of motion conditions via stacked lightweight modules. Our VAE architecture is inspired by MAGVIT-v2 [50] but simplified by omitting codebook dependencies. The MGF integrates adaptive normalization layers [27] to infuse multi-level motion conditions into the corresponding DiT blocks. We explored various adaptations of transformer blocks including adaptive layer normalization, cross-attention, and extra channel connections to inject the motion conditions. Among these, adaptive layer normalization emerged as the most effective to generate consistent videos following the trajectory.

During training, we adapt OpenSora’s workflow to generate high-quality video-text pairs and utilize an optical flow estimator [44] for trajectory extraction. We also integrate a motion segmentor [54] with a camera detector1 to filter out instances dominated by camera motion, thereby improving our tracking of object trajectories. This careful selection process results in a dataset of high-quality videos with consistent motion. With an adapter-like strategy [24], we solely train the temporal blocks, together with the TE and MGF. This strategy seamlessly integrates DiT’s inherent generative knowledge with external motion signals.

The main contributions of our work are as follows:

- • We introduce Tora, the first trajectory-oriented DiT model for video generation. As illustrated in Figure 2, Tora enables the creation of motion-manipulable videos with varying aspect ratios, extending up to 204 frames and 720p resolution.
- • We propose a novel trajectory extractor and a motionguidance fusion mechanism to facilitate motion control that aligns with the scalability of DiT. Additionally, we

1https://github.com/antiboredom/camera-motion-detector

ablate several architectural choices and scaling capabilities to offer empirical baselines for future research.

• Experiments demonstrate that Tora achieves state-of-theart accuracy in controlling object motions. Furthermore, it demonstrates superiority in simulating movements within the physical world.

#### 2. Related Work

##### 2.1. Diffusion models for Video Generation

Diffusion models have demonstrated an impressive capability to generate high-quality video samples. Previous research [13, 14, 17, 36, 52] commonly use video diffusion models (VDMs) that incorporate temporal convolutional and attention layers into the pre-trained image diffusion models. Subsequently, VideoCrafter [7] and SVD [4] expand the application of video diffusion models to larger datasets, while TF-T2V [40] directly learns from extensive text-free videos. Nonetheless, these methods encounter limitations in generating long videos, owing to the inherent constraints on capacity and scalability within the UNet design. Conversely, DiT-based models [3, 5, 29, 47, 55] can directly generate videos extending up to tens of seconds. Sora [5] converts visual data into a unified representation, facilitating large-scale training and enabling the generation of 1-minute high-definition video. Vidu [3] is capable of generating both realistic and imaginative videos in various aspect ratios. CogVideoX [47] introduces an expert diffusion transformer model that generates videos from text prompts or images, along with an effective text-video data processing pipeline that enhances video caption quality.

##### 2.2. Motion control in Video Generation

To better control motion in generated videos, a multitude of studies have endeavored to introduce diverse motion signals in VDMs. Pioneering works like MotionDirector [53] and VMC [16] have utilized reference videos to extract motion patterns applicable to diverse video generations. VideoComposer [39] expands upon this by adopting depth maps, sketches, or motion vectors from references as conditional inputs for motion control. Nonetheless, these methodologies are limited to reproducing existing motion patterns. Conversely, approaches that leverage trajectories [12, 20, 23, 35, 41, 48] or bounding boxes [9, 41, 48] in video generation promise greater adaptability and user accessibility. DragNUWA [48] breaks new ground by integrating trajectory-based conditioning into VDMs, facilitating complex camera and object movements. AnimateAnything [9] employs motion masks for precise control over the moving regions. DragAnything [43] uses the object mask to generate entity representations for achieving motion control. MotionCtrl [41] facilitates more flexible control, allowing separate adjustment of both camera movements and

|x<br><br>[Figure 85]| |
|---|---|
| | |

+ Add

Trajectory Extractor

[Figure 86]

###### Noise patch

| | |
|---|---|
| | |
| | |
| | |
|[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]| |
| | |
| | |
| | |

[Figure 92]

Motion patch

Spatial-Temporal DiT Block

[Figure 93]

Motion-guidanceFuser

TemporalSelf-Attention

[Figure 94]

SpatialSelf-Attention

[Figure 95]

Cross-Attention

Cross-Attention

3D VAE

[Figure 96]

[Figure 97]

[Figure 98]

FeedForward

FeedForward

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

+ +

###### + +

…

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

…

…

[Figure 111]

[Figure 112]

ResBlock

+

Video T5 Decoder

A vibrant monarch butterfly flutters gracefully above a lush, colorful coral reef.

ResBlock

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

###### +

Trajectory Extractor

- Figure 3. Overview of the Tora Architecture. We introduce two novel modules: the Trajectory Extractor and the Motion-guidance Fuser. The Trajectory Extractor uses a 3D motion VAE to embed trajectory vectors into the same latent space as video patches, preserving motion information across frames. It then employs stacked convolutional layers to extract hierarchical motion features. The Motion-guidance Fuser utilizes adaptive normalization layers to integrate these multi-level motion conditions into the corresponding DiT blocks, ensuring that generated videos consistently follow defined trajectories. Our method leverages the scalability of DiT, enabling the creation of motioncontrollable videos of extended duration.

individual object motions. However, all of them yield noticeable artifacts in both motion consistency and visual presentation when applied to longer sequences. In contrast,

- our method first integrates trajectories into DiT architecture, which enables closer adherence to the physical world.

#### 3. Methodology

##### 3.1. Preliminary

Latent Video Diffusion Model (LVDM). The LVDM enhances the stable diffusion model [32] by integrating a 3D UNet, thereby empowering efficient video data processing. This 3D UNet design augments each spatial convolution with an additional temporal convolution and follows each spatial attention block with a corresponding temporal attention block. It is optimized by employing a noise-prediction objective function:

lϵ = ||ϵ − ϵθ(zt,t,c)||22, (1) Here, ϵθ(·) signifies the 3D UNet’s noise prediction function. The condition c is guided into the UNet using crossattention for adjustment. Meanwhile, zt denotes the noisy hidden state, evolving like a Markov chain that progressively adds Gaussian noise to the initial latent state z0:

zt = √α¯tz0 + √1 − α¯tϵ, ϵ ∼ N(0,I), (2)

where α¯t = ti=1(1 − βt) and βt is a coefficient that controls the noise strength in step t.

Diffusion Transformer (DiT). The DiT [26] introduces a novel architecture that merges the strengths of diffusion models with transformer architectures [38]. This integration aims to address the limitations of traditional UNet-based latent diffusion models (LDMs), improving their performance, versatility, and scalability. While keeping the overall framework consistent with existing LDMs, the key shift lies in replacing the UNet with a transformer architecture for learning the denoising function ϵθ(·), thereby marking a pivotal advance in the realm of generative modeling.

##### 3.2. Tora

Tora employs the Spatial-Temporal Diffusion Transformer (ST-DiT) from OpenSora as its foundational model. To facilitate user-friendly motion control while aligning with the scalability of DiT, Tora integrates two novel motionprocessing components: the Trajectory Extractor (TE) and the Motion-guidance Fuser (MGF). An overview of Tora’s workflow is illustrated in Figure 3.

Spatial-Temporal DiT. The ST-DiT architecture incorporates two distinct block types: the Spatial DiT Block (SDiT-B) and the Temporal DiT Block (T-DiT-B), arranged in an alternating sequence. The S-DiT-B comprises two attention layers, each performing Spatial Self-Attention (SSA) and Cross-Attention sequentially, succeeded by a pointwise feed-forward layer that serves to connect adjacent TDiT-B block. Notably, the T-DiT-B modifies this schema solely by substituting SSA with Temporal Self-Attention

(TSA), preserving architectural coherence. Within each block, the input, upon undergoing normalization, is concatenated back to the block’s output via skip-connections. By leveraging the ability to process variable-length sequences, the denoising ST-DiT can handle videos of variable durations.

During processing, a video autoencoder [49] is first employed to diminish both spatial and temporal dimensions of videos. To elaborate, it encodes the input video X ∈ RL×H×W×3 into video latent z0 ∈ Rl×h×w×4, where L denotes the video length and l = L/4,h = H/8,w = W/8. z0 is next “patchified”, resulting in a sequence of input tokens I ∈ Rl×s×d. Here, s = hw/p2 and p denotes the patch size. In both SSA and TSA, standard Attention is performed using Query (Q), Key (K), and Value (V ) matrices:

###### Q = WQ · Inorm;K = WK · Inorm;V = WV · Inorm, (3)

Here, Inorm is the normalized I, WQ,WK,WV are learnable matrices. The textual prompt is embedded with a T5 encoder and integrated using a cross-attention mechanism.

Trajectory Extractor. Trajectories have proven to be a more user-friendly method for controlling the motion of generated videos. Specifically, given a trajectory traj = {(xi,yi)}Li=0−1, where (xi,yi) denotes the spatial position (x,y) at the i-th frame the trajectory passes through. Previ-

- ous studies primarily encode the horizontal offset u(xi,yi) and the vertical offset v(xi,yi) as the motion condition:

###### u(xi,yi) = xi+1 − xi; v(xi,yi) = yi+1 − yi, (4)

However, the DiT model employs a video autoencoder and a patchification process to convert the video into patches. Here, each patch is derived across multiple frames, rendering it inappropriate to straightforwardly employ frame-toframe offsets. To address this, our TE converts the trajectory into motion patches, which inhabit the same latent space as the video patches. Particularly, we begin by transforming the traj into a trajectory map g ∈ RL×H×W×2, enhanced with a Gaussian Filter to mitigate scatter. Notably, the first frame employs a fully-zero map. Afterward, the trajectory map g is translated into the RGB color space, producing gvis ∈ RL×H×W×3 through a flow visualization technique. We use a 3D VAE to compress trajectory maps, achieving an 8x spatial and 4x temporal reduction, aligning with OpenSora framework. Our VAE is based on the MAGVIT-v2 architecture, with spatial compression initialized using the VAE of SDXL [28] to accelerate convergence. We train the model using reconstruction loss to obtain the compact motion latent representation gm ∈ Rl×h×w×4 from the gvis.

To match the size of the video patches, we use the same patch size on gm and encode it using a series of convolutional layers, resulting in spacetime motion patches f ∈ Rl×s×d

′

. Here d′ is the dimension of motion patches. The

Concatenation Layer Norm MLP MLP

MLP

MLP Scale, Shift

Cross-Attention

Temporal SelfAttention

Temporal SelfAttention

Temporal SelfAttention

Extra Channel Adaptive Norm Cross-Attention

Figure 4. Different designs of the Motion-guidance Fuser for incorporating trajectory conditioning. Adaptive Norm demonstrates the best performance.

output of each convolutional layer is skip-connected to the input of the next layer to extract multi-level motion features:

fi = Convi (fi−1) + fi−1, (5) where fi is the motion condition for i-th ST-DiT block.

Motion-guidance Fuser. To incorporate DiT-based video generation with the trajectory, we explore three variants of fusion architectures that inject motion patches into each STDiT block. These designs are illustrated in Figure 4.

- - Extra channel connections. Denote hi ∈ Rl×s×d as the resultant output from the i-th block of the ST-DiT. Following the widespread use of concatenation in GANbased LVDM, the motion patches are simply concate-

nated with the previous hidden state hi−1 along the channel dimension. An additional MLP is then added to maintain the same latent size:

hi = MLP ([hi−1,fi]) + hi−1, (6)

- - Adaptive Norm layer. Inspired by the adaptive normalization layers employed in GANs, we initially convert fi into scale γi and shift βi by adding two zero-initialized convolution layers into the ST-DiT block. Subsequently,

γi and βi are integrated into hi through a straightforward linear projection:

hi = γi · hi−1 + βi + hi−1, (7)

- - Cross-Attention layer. The ST-DiT block has been modified to include an additional Cross-Attention layer following the SSA or TSA, with the motion patches serving as the key and value to integrate with the hidden state h:

###### hi = CrossAttn ([hi−1,fi]) + hi−1, (8)

We evaluate three types of fusion architectures and find that the adaptive norm yields the best performance and computational efficiency. For the remainder of the paper, MGF employs the adaptive norm layer unless otherwise specified.

- 3.3. Data Processing and Training Strategy

Data Processing. We employ a structured data processing method to obtain high-quality training videos with consistent object motion. Initially, raw videos are segmented into shorter clips based on scene detection2. Subsequently, we remove invalid videos, such as those with encoding errors, zero duration, or low resolution. Furthermore, we utilize aesthetic3 and optical flow scores [44] to filter out low-quality videos. To concentrate on the motion of primary objects, we implement camera motion filtering, using results from motion segmentation [54] and camera detection to exclude instances predominantly exhibiting camera movement. Dramatic object motions in certain videos can lead to significant optical flow deviations, which may interfere with trajectory training. To address this, we retain these videos based on a probability of (1 − flow score/100). For eligible videos, we generate captions using the PLLaVA model [45]. During inference, we utilize GPT-4o [25] to refine prompts, ensuring alignment with training process.

Motion condition training. Inspired by DragNUWA and MotionCtrl, we adopt a two-stage training approach for trajectory learning. In the first stage, we extract dense optical flow [44] from the training video as the trajectory, providing richer information to enhance motion learning. In the second stage, we adjust the model from complete optical flow to more user-friendly trajectories by randomly selecting 1 to N object trajectories based on motion segmentation results and flow scores. To improve the scattered nature of sparse trajectories, we apply a Gaussian filter for refinement. After completing the two-stage training, Tora facilitates flexible motion control using arbitrary trajectories.

Image condition training. We employ a mask strategy to support visual conditioning. Specifically, we randomly unmask frames during training, ensuring that the video patches of unmasked frames are not subjected to any noise.

- 4. Experiments

- 4.1. Experimental Setup

Implementation Details. Tora is initialized with OpenSora v1.2 weights, and training videos have resolutions from 144p to 720p and frame counts ranging from 51 to 204. To balance training FLOP and memory usage, we adjust the batch size from 1 to 50. We use Adam Optimizer [18] with a learning rate of 2 × 10−5 on 8 NVIDIA A100. The 3D VAE is initially trained on datasets [6, 21, 22, 33] for optical flow estimation and then frozen during Tora training. We train Tora for 2 epochs with dense optical flow and fine-tune for 1 epoch with sparse trajectories. The maximum number of sampling trajectories N is set to 16. The inference step and the guidance scale are set to 30 and 7.0, respectively.

- 2https://github.com/Breakthrough/PySceneDetect
- 3https://github.com/christophschuhmann/improved-aesthetic-predictor

Dataset. Our training videos are sourced from four datasets: 1) Panda-70M [8], from which we use the training10M subset containing high-quality videos; 2) Mixkit [11]; 3) Pexels [2]; and 4) Internal videos. The internal videos are manually annotated, with each clip labeled to include object masks and camera movement. Following our data processing pipeline, we select about 630k eligible videos for the training dataset. For inference, we curate 185 clips with diverse motion trajectories and scenes, to serve as a new benchmark for evaluating the motion controllability.

Metrics. We leverage standard metrics including Fr´echet Video Distance (FVD) [37], and CLIP Similarity (CLIPSIM) [42] to quantitatively evaluate video quality. For assessing motion controllability, we utilize the Trajectory Error (TrajError) metric, which calculates the average L1 distance between the generated and predefined trajectories.

##### 4.2. Results

We evaluate against motion-guided video generation methods using 16-frame, 64-frame, and 128-frame configurations. Trajectories are scaled proportionally for different durations. UNet-based approaches utilize sequential inference for extended generation. Since there are no DiT-based methods, we adapt DragNUWA’s motion trajectory design to our foundation model as an additional baseline. Specifically, we implement its official convolutional motion feature extraction and linear projection-based injection. To ensure compatibility with base DiT model, we perform downsampling across both spatial and temporal dimensions.

As shown in Table 1, UNet-based methods all exhibit increasing deviations with longer sequences, causing motion blur and object deformation. In contrast, Tora leverages the transformer’s scaling capabilities to maintain robustness, achieving 3-5 times higher trajectory accuracy and approximately 30-40% better FVD in 128-frame tests. Figure 6 shows a comparative qualitative analysis of Tora against these mainstream motion control techniques.

While the OpenSora achieves high visual quality, its inability to incorporate motion control leads to random object trajectories, resulting in significantly elevated TrajError. Notably, Tora demonstrates dual superiority over OpenSora in both motion control efficacy and visual performance at most settings. We find Tora’s exceptional capacity in suppressing temporal artifacts and motion blurring compared with OpenSora (see Appendix for visual comparisons), yielding enhanced temporal coherence and video stability. While OpenSora-based DragNUWA improves motion controllability over UNet-based methods, its motion representations exhibit intrinsic incompatibility with DiT’s latent space. This architectural incompatibility induces suboptimal feature integration during training, consequently degrading visual quality by about 5% below baseline.

Figure 5 presents an analysis of Trajectory Error across

FVD (↓) CLIPSIM (↑) TrajError (↓)

Method

16-frame 64-frame 128-frame 16-frame 64-frame 128-frame 16-frame 64-frame 128-frame UNet-based method

VideoComposer [39] 529 668 856 0.2335 0.2284 0.2236 15.11 29.14 58.76 DragNUWA [48] 475 593 784 0.2385 0.2341 0.2305 10.04 17.33 41.25

AnimateAnything [9] 487 602 775 0.2399 0.2342 0.2313 13.39 27.28 51.33 TrailBlazer [20] 459 581 756 0.2403 0.2351 0.2322 11.68 19.47 44.10 MotionCtrl [41] 463 572 731 0.2412 0.2376 0.2331 9.42 16.46 38.39

###### DiT-based method

OpenSora [55] 430 476 533 0.2452 0.2433 0.2411 286.43 321.52 373.17 OpenSora-based DragNUWA* 451 504 565 0.2430 0.2419 0.2393 10.11 13.88 21.75

###### Tora(Ours) 438 460 494 0.2447 0.2435 0.2418 7.23 8.45 11.72

- Table 1. Quantitative comparisons with motion-controllable video generation models. As the number of generated frames increases, Tora’s performance advantage over UNet-based methods becomes more pronounced. Specifically, Tora not only enhances motion fidelity but also improves the visual quality of the foundational model. Comparisons with OpenSora-based DragNUWA highlight the strengths of our proposed motion modules, which integrate seamlessly with DiT’s architecture.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0

2

4

6

8

10

12

14

16

240p 360p 480p 720p

Trajectory Error vs. Resolution & Duration

34-frame 51-frame 102-frame 204-frame

- Figure 5. Comparison of Trajectory Error across various resolutions and durations.

Method FVD (↓) CLIPSIM (↑) TrajError (↓)

Sampling Frame 581 0.2304 27.61 Average Pooling 558 0.2325 20.97

3D VAE 513 0.2358 14.25

- Table 2. Evaluation of the impact of different trajectory compression methods.

Method FVD (↓) CLIPSIM (↑) TrajError (↓)

Extra Channel 542 0.2329 21.07 Cross Attention 526 0.2354 18.36 Adaptive Norm 513 0.2358 14.25

Table 3. Different variants of motion fusion blocks employed in MGF. Adaptive Norm works best.

approach is sub-optimal for motion control due to potential flow estimation errors during rapid movements or occlusions, and the increased dissimilarity between patches complicates learning. The second method employs average pooling to gather information from successive frames. Although this captures a general sense of movement, it sacrifices precision by averaging the trajectory’s direction and magnitude, diluting important motion details. Our method utilizes a 3D motion VAE to extract the global context of successive trajectory intervals. Extensive training on a large dataset of trajectory videos with this method yields the best results, highlighting the effectiveness of our customized VAE approach for compression.

different resolutions and durations. Unlike UNet models, Tora shows a gradual increase in error as duration extends. This aligns with the decrease in video quality observed in the DiT model. The results demonstrate that our method effectively maintains trajectory control over longer durations.

Block design and integrated position of MGF. We train the three variant MGF blocks as previously described, with the results presented in Table 3. Notably, the adaptive norm block achieves lowest FVD and Trajectory Error, while also exhibiting the highest computational efficiency. This advantage arises from its ability to dynamically adapt features based on varying conditions without strict alignment, a common cross-attention challenge. It also maintains temporal consistency by modulating conditional information over time, essential for incorporating motion cues. In contrast, channel concatenation can lead to information congestion, making motion signals less effective. We find that initializing the normalization layer as the identity function is vital

##### 4.3. Ablation study

Trajectory Compression. To validate the alignment of motion patches with the DiT inputs, we explore three different methods for trajectory compression, as summarized in Table 2. The first method samples the mid-frame as a keyframe for successive 4-frame intervals and uses PatchUnshuffle [15] for spatial compression. While simple, this

|[Figure 118]|
|---|

Two men cycled on the highway under a clear, sunny sky

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

DragNUWA

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

MotionCtrl

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Ours

|[Figure 134]|
|---|

Two red lanterns sway in the wind under the serene night sky

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

- Figure 6. Qualitative Comparisons on Trajectory Control. In the bicycle scenario, Tora realistically captures pedaling motions, while other methods show legs in an unnatural, nearly horizontal position. In another case, DragNUWA causes significant deformation of the lanterns, and MotionCtrl fails to accurately depict two lanterns. Overall, Tora not only adheres precisely to the specified trajectory but also produces smoother movement that conforms to the physical world.

Motion-guidance FVD (↓) CLIPSIM (↑) TrajError (↓)

Dense Flow 601 0.2307 39.34 Sparse Flow 556 0.2334 24.73

Hybrid 513 0.2358 14.25

Table 4. Ablation of the type of training trajectories. “Hybrid” denotes the two-stage training strategy.

for optimal performance. Additionally, placing the MGF module within the Temporal DiT block significantly enhances trajectory motion control, evidenced by a drop in Trajectory Error from 23.39 to 14.25.

Training Strategies. We evaluate the two-stage training approach, with results in Table 4. Training only with dense optical flow is ineffective, as it fails to capture the details of sparse trajectories. Conversely, using only sparse trajectories provides limited information, complicating the learning process. In contrast, our two-stage strategy demonstrates better adaptability and versatility in managing various motion patterns, leading to improved overall performance.

Scaling motion-control ability. We investigate scaling laws in motion-controllable video generation by examining model parameter size and training volume. To leverage existing multi-size foundational models, we transfer our motion modules to CogVideoX architectures (2B/5B parameters), resulting in Tora-CogVideoX2B (2.5B) and ToraCogVideoX5B (6.3B). Our implementation retains base VAE for motion compression while inserting MGF prior to Expert Transformer’s full attention module. Figure 7

TrajError

25

| | |23.8|2| | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | |15.6|5<br><br>13|.28| |
| | |15.3|7| | | | | |
| | | | | |11.4|7 10|.52| |
| | | | | | | | | |

20

|Tora-CogVideoX2B Tora-CogVideoX5B<br><br>|
|---|

15

10

5

0 200 400 600 800

Training Volume (k)

Figure 7. Scaling behavior of motion control ability in Tora.

demonstrates that increasing model scale and training data improves motion control well, confirming our modules’ seamless compatibility with DiT’s scalability framework.

#### 5. Conclusion

This paper introduces Tora, the first trajectory-oriented Diffusion Transformer framework for video generation. Tora effectively encodes arbitrary trajectories into spacetime motion patches, which align well with the scaling properties of DiT, thereby enabling more realistic simulations of physical world movements. By employing a two-stage training process, Tora achieves motion-controllable video generation across a wide range of durations, aspect ratios, and resolutions. Remarkably, it can generate high-quality videos that adhere to specified trajectories, producing up to 204 frames at 720p resolution. This capability underscores Tora’s versatility and robustness in handling diverse motion patterns while maintaining high visual fidelity. We hope our work provides a strong baseline for future research in motionguided Diffusion Transformer methods.

# Appendix

This supplementary material offers additional results, comprehensive dataset information, and thorough analyses that bolster the findings and conclusions outlined in the main text. It is organized as follows:

- • Additional qualitative results.
- • User Study with DiT-based methods.
- • Data pre-processing method.
- • Dataset details, regarding total quantity, total durations, etc.
- • Prompt refinement method.
- • Motion VAE training.

#### 6. Qualitative Comparisons

While the main text focuses on quantitative comparisons with the motion-controllable video generation models and ablation studies on different designs for Trajectory Extractor and Motion-guidance Fuser, here we provide further visual comparisons.

##### 6.1. Compare with OpenSora

Despite OpenSora’s impressive accomplishments, it faces challenges when creating long videos featuring complex motions, such as simultaneous movement of multiple objects, swinging, or circling. This often leads to incoherent or distorted foreground objects, negatively impacting visual quality. To our delight, we discovered that incorporating appropriate trajectory control into the DiT model offers a more effective constraining signal. This improvement markedly enhances video fluidity and preserves object fidelity, as demonstrated in Figure 8.

In scenarios where a teddy bear is oscillating side to side on a skateboard or a rose is swirling in circular motions, OpenSora, which relies solely on textual directives for motion control, exhibits noticeable object deformations. In contrast, Tora excels at maintaining the inherent shape of the objects. Additionally, when managing the motion of multiple entities, such as a pair of jellyfish—one moving upward while the other moves downward, OpenSora demonstrates noticeable flickering, underscoring its limitations in handling complex movements. In conclusion, the integration of Tora’s motion signaling mechanism enhances both the controllability and stability of the synthesized video output.

##### 6.2. Comparison of Different Trajectory Compression Methods

We train our proposed trajectory extractor using the various trajectory compression methodologies previously discussed. The comparisons of these methods are visually illustrated in Figure 9.

In key-frame sampling, while it successfully captures essential motion, it frequently leads to misalignment between video patches and motion patches, especially during rapid motion sequences. This misalignment hinders the generated objects from accurately tracking their trajectories, negatively impacting visual fluidity and overall quality. On the other hand, average pooling smooths out minor variations, resulting in a more consistent motion representation. However, in complex trajectories, such as S-shaped turns where consecutive frame directions are inconsistent, this approach may introduce artifacts because the physical relevance of optical flow decreases. In contrast, our proposed 3D VAE approach effectively compresses trajectory information into the video’s latent space. By training the 3D VAE on the large dataset with flow annotations, it successfully extracts the most relevant motion features for guidance, preserving the movement of successive frames to a significant extent. As evidenced in the results, this method significantly enhances the fluidity and coherence of the generated movements, producing visually compelling sequences that closely resemble natural motion.

#### 7. User Study with DiT-based methods

We conduct a user study to compare OpenSora-v1.2, CogVideoX-2B [47], Vidu [3], and Kling v1.0 [1], assessing the effectiveness of our method using our evaluation dataset. 10 human volunteers participate in evaluating quality based on three criteria: physics simulation, sensory quality, and instruction adherence. For Tora, participants draw appropriate trajectories in response to given text prompts. The experiment employs a pairwise comparison approach, where evaluators choose the superior output from each pair of generated results based on the same input. The resulting win rates appear in Table 5. Our method outperforms both OpenSora and CogVideoX-2B across all metrics, affirming the superiority of our proposed modules and data processing methods. Compared to the closed-source method, Vidu, we achieve competitive results. Kling demonstrates remarkable capabilities, and we hope that Tora can work to close the performance gap in future iterations.

Model Phys. Simu. Sens. Qual. Inst. Foll. Tora vs. OpenSora-v1.2 71% 61% 64%

Tora vs. CogVideoX 53% 56% 52% Tora vs. Vidu 54% 48% 47% Tora vs. Kling 45% 43% 41%

Table 5. Win rates of Tora compared to OpenSora-v1.2, CogVideoX, Vidu, and Kling in terms of Physics Simulation, Sensory Quality, and Instruction Following.

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

OpenSoraTora

OpenSoraTora

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

Prompt: Two jellyfish gracefully swimming underwater. The left one moves upward while the right one moves downward.

Prompt: A rose gracefully swirling in circular motions, captured through a soft-focus lens, with the vibrant backdrop of a bustling city park.

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

OpenSoraTora

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Prompt: A teddy bear gently oscillating side to side on a skateboard. The scene is set against a picturesque, dreamy landscape, withdistant trees silhouetted against a starry sky. Soft, twinkling lights from a nearby village or park create a magical ambiance, enhancing the whimsical charm of the teddy bear's playful ride.

- Figure 8. Qualitative comparison between Tora and OpenSora. All results are generated under the same text and image conditions. Tora employs an appropriate trajectory that simulates real-world physics, leading to more coherent and stable motion.

#### 8. Data Pre-processing

During the processing of the video datasets, constructing a high-quality training set is crucial as it significantly impacts the quality of the generated videos. The following is a detailed description of our data processing workflow, which includes steps such as invalid videos removal, resolution filtering, camera motion filtering, and assessing the degree of object motion.

Initially, during the dataset preparation phase, we remove invalid videos. This step aims to identify and discard videos that do not meet our established criteria, including those with encoding errors, a duration of zero, or low quality. We identify encoding errors and zero-duration videos by directly decoding them. Furthermore, we predict both the aesthetic score4 and the optical flow score [44] for each

4https://github.com/christophschuhmann/improved-aesthetic-predictor

video. A video is deemed valid only if its aesthetic score exceeds 5.5 and its flow score is greater than 3.

Next, we perform resolution filtering. To ensure the effectiveness of subsequent study, we establish a minimum resolution standard of 720p. By checking the resolution of each video, we can eliminate those that fall below this threshold, thereby ensuring that the videos in our dataset possess adequate clarity and detail.

Subsequently, we perform camera motion filtering using a camera motion detector5 and a motion segmentor [54] to filter out videos with significant camera movement, which may distort the model’s ability to focus on the motion of the primary subjects. More specifically, the zoom detection threshold is set between 0.4 and 0.6. The detected camera movement angles, calculated based on the background from the motion segmentation results, are valid as

5https://github.com/antiboredom/camera-motion-detector

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

- (a)
- (b)
- (c)

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Prompt: A red helium balloon floating slowly up to the sky over a desert.

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

(a) (b)

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

(c)

Prompt: Feather floats gently down in a quiet meadow.

- Figure 9. Generated videos employing different trajectory compression methods: (a) Sampling Keyframe; (b) Average Pooling; (c) 3D VAE.

follows:[0◦,20◦],[160◦,200◦],[340◦,360◦].

Finally, we analyze the magnitude score of the optical flow within the foreground, excluding those scenes that are mostly static or exhibit minimal movement. Moreover, dramatic object motions in some videos can cause significant optical flow deviations, interfering with trajectory training. Consequently, we retain these videos with a probability of (1 − flow score/100).

Through these rigorous filtering and processing steps, we successfully construct a high-quality video dataset suitable for subsequent training, providing a solid foundation for our study.

#### 9. Dataset Details

This section offers an overview of the dataset used in this study, covering its origin and composition. We employ histograms and descriptive statistics to illustrate the dataset’s structure and distribution.

##### 9.1. Training Data

The video data is sourced from the Panda-70M subset, Mixart, and internal videos. We initially collect 2.6M videos and apply the data preprocessing pipeline to filter the content, resulting in 631k eligible videos for training.

120000

100000

80000

Frequency

60000

40000

20000

0

40 80 120 160 200 240 280 Caption Lengths

(a) Histogram of Caption Lengths.

200000

160000

Frequency

120000

80000

40000

0

0 8 16 24 32 40 48 56 64 Duration (seconds)

(b) Histogram of Video Durations.

|resolution<br><br># frames|51 102 204 408|
|---|---|
|144p 240p 360p 480p 720p|50 25 12 6 20 10 5 2<br><br>8 4 2 1 4 2 1 2 1 - -<br><br>|

(c) The training batch size of every bucket (resolution, duration).

Figure 10. Overview of the training data distributions and batch sizes.

An overview of the training dataset is presented in Table 6, which details the durations, resolution and FPS.

# Videos Clips 631053 Total Durations (hours) 2952.93 Average Shorter Edge Length 965.11 Average FPS 29.23

Table 6. Statistical information about the training data.

Additionally, Table 7 summarizes the mean and standard deviation for the durations, number of frames, and caption lengths. We also present histogram to show the distribution of the caption lengths and the durations of all video clips, as shown in the Figure 10a and Figure 10b.

mean std

Durations (seconds) 16.85 19.58 #Frames 506.22 644.38 Caption Length (#word) 125.52 24.22

Table 7. Statistics of training set, regarding durations, number of frames, and caption lengths.

Drawing inspiration from OpenSora, we employ a multiscale and mixed-duration training strategy, which involves training videos of various resolutions and lengths together. Specifically,we establish predefined buckets, each defined by a unique combination of (video resolution, duration). Videos are then assigned to the appropriate bucket according to their specific attributes. Note that videos of any aspect ratio will fall into these buckets if their total pixel count is within the specified statistical intervals. The parameter settings for the buckets adhere to the principle that lower resolutions correspond to longer durations, enabling Tora to adapt to videos of varying lengths. Notably, our preprocessing steps ensure that the shorter edge of each training video exceeds 720 pixels. To enable training across various

scales, we shuffle the dataset and randomly select videos for downsampling to lower resolutions. Additionally, we employ different batch sizes for each bucket to balance the GPU load. The details of the buckets are presented in Figure 10c.

##### 9.2. Evaluation Data

Our evaluation dataset is primarily sourced from video object segmentation datasets [30, 31, 46], which offer robust object motion critical for our analysis. To enhance the quality of our evaluation, we implement a camera motion filtering technique to select videos where the camera remains predominantly stable. This filtering process allows us to concentrate on where object motion is distinctly pronounced, thereby improving the reliability of our assessments. For each frame, we utilize the center of the annotated object masks as trajectory points, providing precise references for evaluating motion dynamics. Figure 11 presents several examples from our evaluation dataset, highlighting the diversity and relevance of the selected video sequences.

#### 10. Prompt Refinement

We encourage users to provide detailed text prompts to achieve satisfactory video results. To ensure consistency in the distribution of text prompts during both training and testing phases, we utilize GPT-4o to refine simple testing prompts. The process of learning refined prompts for GPT4o involves two key components. The first component is the task description, which clearly outlines the objectives for the model in generating enhanced content:

You need to refine user’s input prompt. The user’s input prompt is used for video generation task. You need to refine the user’s prompt to make it more suitable for the task. Here are some examples of refined prompts: ↓ a close-up shot of a woman applying makeup. she is using a black brush to

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

Figure 11. Visualization of Our Evaluation Dataset, highlighting 0%, 20%, 40%, 60%, 80%, and 100% of the total duration. Each center point of the annotated object masks is treated as a trajectory point. The number of trajectories in the tested video matches the number of annotated objects.

apply a dark powder to her face. the woman has blonde hair and is wearing a black top. the background is black, which contrasts with her skin tone and the makeup. the focus is on her face and the brush, with the rest of her body and the background being out of focus. the lighting is soft and even, highlighting the texture of the makeup and the woman’s skin. there are no texts or other objects in the video. the woman’s expression is neutral, and she is looking directly at the camera. the video does not contain any action, as it is a still shot of a woman applying makeup. the relative position of the woman and the brush is such that the brush is in her hand and is being used to apply the makeup to her face. the video does not contain any other objects or actions. the woman is the only person in the video, and she is the main subject. the video does not contain any sound. the description

is based on the visible content of the video and does not include any assumptions or interpretations. ↓ a professional setting where a woman is presenting a slide from a presentation. she is standing in front of a projector screen, which displays a bar chart. the chart is colorful, with bars of different heights, indicating some sort of data comparison. the woman is holding a pointer, which she uses to highlight specific parts of the chart. she is dressed in a white blouse and black pants, and her hair is styled in a bun. the room has a modern design, with a sleek black floor and a white ceiling. the lighting is bright, illuminating the woman and the projector screen. the focus of the image is on the woman and the projector screen, with the background being out of focus. there are no texts visible in the image. the relative positions of the objects suggest that

the woman is the main subject of the image, and the projector screen is the object of her attention. the image does not provide any information about the content of the presentation or the context of the meeting. ↓ a serene scene in a park. the sun is shining brightly, casting a warm glow on the lush green trees and the grassy field. the camera is positioned low, looking up at the towering trees, which are the main focus of the image. the trees are dense and full of leaves, creating a canopy of green that fills the frame. the sunlight filters through the leaves, creating a beautiful pattern of light and shadow on the ground. the overall atmosphere of the video is peaceful and tranquil, evoking a sense of calm and relaxation. ↓ a moment in a movie theater. a couple is seated in the middle of the theater, engrossed in the movie they are watching. the man is dressed in a casual outfit, complete with a pair of sunglasses, while the woman is wearing a cozy sweater. they are seated on a red theater seat, which stands out against the dark surroundings. the theater itself is dimly lit, with the screen displaying the movie they are watching. the couple appears to be enjoying the movie, their attention completely absorbed by the on-screen action. the theater is mostly empty, with only a few other seats visible in the background. the video does not contain any text or additional objects. the relative positions of the objects are such that the couple is in the foreground, while the screen and the other seats are in the background. the focus of the video is clearly on the couple and their shared experience of watching a movie in a theater. ↓ a scene where a person is examining a dog. the person is wearing a blue shirt with the word "volunteer" printed on it. the dog is lying on its side, and the person is using a stethoscope to listen to the dog’s heartbeat. the dog appears to be a golden retriever

and is looking directly at the camera. the background is blurred, but it seems to be an indoor setting with a white wall. the person’s focus is on the dog, and they seem to be checking its health. the dog’s expression is calm, and it seems to be comfortable with the person’s touch. the overall atmosphere of the video is calm and professional. ↓ The refined prompt should pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. The refined prompt should be in English.

Following that, GPT-4o is supplied with the testing captions for processing. This allows it to refine the prompts based on the initial task description, ensuring that the provided captions are more detailed and aligned with our objectives:

Generate the refined prompts for following inputs: ↓ A man rides on a huge fish, flying from the water into the sky. ↓ Two Jedi cats are fighting with each other in the forest. ↓ A polar bear with a black hat is walking on the Great Wall. ↓ A woman and a golden retriever are playing on the beach at sunset. ↓ Two roses sway together before a snow-covered mountain range.

#### 11. Motion VAE Training

Given the absence of pre-existing networks tailored for video optical flow compression, training such a network from scratch presents significant challenges. Directly transferring the motion vectors to the image domain and applying a pretrained 3D VAE may hinder the model’s ability to effectively encode motion features, primarily due to domain discrepancies. To overcome this issue, we refine a motion-specific 3D VAE that is initialized from a pretrained model. Specifically, our motion 3D VAE is specifically initialized using the architecture of OpenSora’s VAE, which adapts the structure of Magvit-v2. This VAE has a substantial parameter count of 384 million, effectively leveraging the capabilities of a well-established network. Our training data is sourced from a combination of datasets annotated with optical flow information [6, 21, 22, 33]. We fine-tune the motion 3D VAE for 200,000 iterations with a batch size of 1. The training video size is set to a ran-

dom number of frames, capped at 34. This setting aligns with the OpenSora video VAE, improving compatibility between the motion VAE and the video VAE and ensuring a cohesive training process. We utilize PSNR, SSIM and Trajectory Error to evaluate reconstruction quality and motioncontrollable ability. The performance differences between the pure video VAE and our fine-tuned model are presented in Table 8.

Model PSNR↑ SSIM↑ TrajError↓ Pure Video VAE 27.34 0.842 17.09

Our VAE 28.76 0.860 14.25

Table 8. The performance comparison of different 3D VAE.

#### References

- [1] Kling ai. Accessed June, 2024 [Online] https:// klingai.com. 9
- [2] Pexels. https://www.pexels.com/. 6
- [3] Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024. 3, 9
- [4] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 3
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. https://openai.com/research/videogeneration - models - as - world - simulators,

2024. 1, 3

- [6] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual KITTI 2. arXiv preprint arXiv:2001.10773, 2020. 6, 14
- [7] Haoxin Chen, Menghan Xia, Yin-Yin He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao-Liang Weng, and Ying Shan. VideoCrafter1: Open diffusion models for highquality video generation. arXiv preprint arXiv:2310.19512,

2023. 3

- [8] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, and Sergey Tulyakov. Panda-70M: Captioning 70M videos with multiple cross-modality teachers. In CVPR, pages 13320–13331. IEEE, 2024. 6
- [9] Zuozhuo Dai, Zhenghao Zhang, Yao Yao, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Fine-grained open domain image animation with motion guidance. arXiv preprint

arXiv:2311.12886, 2023. 3, 7

- [10] Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In NeurIPS, pages 8780–8794, 2021. 1
- [11] Elements Envato. Mixkit: Free assets for your next video project. https://mixkit.co, 2024. 6
- [12] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, Chen Sun, Oliver Wang, Andrew Owens, and Deqing Sun. Motion prompting: Controlling video generation with motion trajectories. arXiv preprint arXiv:2412.02700, 2024. 3
- [13] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [14] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 1, 3
- [15] Hyemi Jang, Junsung Park, Dahuin Jung, Jaihyun Lew, Ho Bae, and Sungroh Yoon. PUCA: Patch-unshuffle and channel attention for enhanced self-supervised image denoising. In NeurIPS, 2023. 7
- [16] Hyeonho Jeong, Geon Yeong Park, and Jong Chul Ye. VMC: Video motion customization using temporal attention adaption for text-to-video diffusion models. In CVPR, pages 9212–9221. IEEE, 2024. 3
- [17] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2Video-Zero: Textto-image diffusion models are zero-shot video generators. In ICCV, pages 15908–15918. IEEE, 2023. 3
- [18] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 6
- [19] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 3
- [20] Wan-Duo Kurt Ma, John P. Lewis, and W. Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. In SIGGRAPH Asia, pages 97:1–97:11. ACM,

2024. 3, 7

- [21] Nikolaus Mayer, Eddy Ilg, Philip H¨ausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In CVPR, pages 4040–4048. IEEE Computer Society, 2016. 6, 14
- [22] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A high-resolution highdetail dataset and benchmark for scene flow, optical flow and stereo. In CVPR, pages 4981–4991. IEEE, 2023. 6, 14
- [23] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. ReVideo: Remake a video with motion and content control. In NeurIPS, 2024. 3
- [24] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In AAAI, pages 4296–4304. AAAI Press,

2024. 3

- [25] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 6
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, pages 4195–4205, 2023. 1, 4
- [27] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, pages 3942–3951,

2018. 3

- [28] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR. OpenReview.net, 2024. 5
- [29] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam S. Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali K. Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dmitry Vengertsev, Edgar Sch¨onfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 3

- [30] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 12
- [31] Jiyang Qi, Yan Gao, Yao Hu, Xinggang Wang, Xiaoyu Liu, Xiang Bai, Serge Belongie, Alan Yuille, Philip HS Torr, and Song Bai. Occluded video instance segmentation: A benchmark. IJCV, 130(8):2022–2039, 2022. 12
- [32] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125,

2022. 1, 4

- [33] Anurag Ranjan, David T. Hoffmann, Dimitrios Tzionas, Siyu Tang, Javier Romero, and Michael J. Black. Learning multihuman optical flow. IJCV, 128(4):873–890, 2020. 6, 14
- [34] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015. 1

- [35] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Motion-I2V: Consistent and controllable image-to-video generation with explicit motion modeling. In SIGGRAPH, page 111. ACM, 2024. 3
- [36] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 3

- [37] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, pages 5998–6008, 2017. 4
- [39] Xiang Wang, Hangjie Yuan, Shiwei Zhang, Dayou Chen, Jiuniu Wang, Yingya Zhang, Yujun Shen, Deli Zhao, and Jingren Zhou. VideoComposer: Compositional video synthesis with motion controllability. In NeurIPS, 2023. 2, 3, 7
- [40] Xiang Wang, Shiwei Zhang, Hangjie Yuan, Zhiwu Qing, Biao Gong, Yingya Zhang, Yujun Shen, Changxin Gao, and Nong Sang. A recipe for scaling up text-to-video generation with text-free videos. In CVPR, 2024. 3
- [41] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Tianshui Chen, Menghan Xia, Ping Luo, and Yin Shan. MotionCtrl: A unified and flexible motion controller for video generation. In SIGGRAPH, 2024. 2, 3, 7
- [42] Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. GODIVA: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021. 6
- [43] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. DragAnything: Motion control for anything using entity representation. In ECCV, pages 331–348. Springer, 2024. 3
- [44] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE TPAMI, 45(11):13941– 13958, 2023. 3, 6, 10
- [45] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See-Kiong Ng, and Jiashi Feng. PLLaVA : Parameter-free LLaVA extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 6
- [46] Linjie Yang, Yuchen Fan, Yang Fu, and Ning Xu. The 3rd large-scale video object segmentation challenge - video instance segmentation track, 2021. 12
- [47] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3, 9
- [48] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. DragNUWA: Fine-grained

- control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2, 3, 7
- [49] Lijun Yu, Yong Cheng, Kihyuk Sohn, Jos´e Lezama, Han Zhang, Huiwen Chang, Alexander G Hauptmann, MingHsuan Yang, Yuan Hao, Irfan Essa, et al. MAGVIT: Masked generative video transformer. In CVPR, 2023. 5
- [50] Lijun Yu, Jos´e Lezama, Nitesh Bharadwaj Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion - tokenizer is key to visual generation. In ICLR. OpenReview.net, 2024. 3
- [51] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2VGen-XL: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 1
- [52] Yabo Zhang, Yuxiang Wei, Dongsheng Jiang, Xiaopeng Zhang, Wangmeng Zuo, and Qi Tian. ControlVideo: Training-free controllable text-to-video generation. In ICLR. OpenReview.net, 2024. 3
- [53] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, and Mike Zheng Shou. MotionDirector: Motion customization of text-to-video diffusion models. In ECCV, pages 273–290. Springer, 2024. 3
- [54] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. ParticleSfM: Exploiting dense point trajectories for localizing moving cameras in the wild. In ECCV, pages 523–542, 2022. 3, 6, 10
- [55] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 3, 7

