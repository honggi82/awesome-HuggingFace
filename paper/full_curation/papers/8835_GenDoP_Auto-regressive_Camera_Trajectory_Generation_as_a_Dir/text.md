## GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography

Mengchen Zhang1,2, Tong Wu3 , Jing Tan4, Ziwei Liu5, Gordon Wetzstein3, Dahua Lin2,4

1Zhejiang University, 2Shanghai Artificial Intelligence Laboratory, 3Stanford University, 4The Chinese University of Hong Kong, 5Nanyang Technological University

zhangmengchen@zju.edu.cn, {wutong16,gordon.wetzstein}@stanford.edu

# arXiv:2504.07083v2[cs.CV]10Apr2025

{tj023,dhlin}@ie.cuhk.edu.hk, ziwei.liu@ntu.edu.sg

[Figure 1]

LLM

[Directorial Caption]: (Movements + Scene + Intent)

The camera starts static, yawing right to capture the extensive area and reveal more equipment as it pivots. The shot continues to hold steady, allowing for clear observation of the surroundings, including dirt inclines and the still water reflecting nearby trees. The camera then smoothly moves forward to the water's edge, keeping focus on the calm water surface.

[Motion Caption]: The camera remains static while yawing right, stays static, and then moves forward.

DatasetConstructionTrajectoryGeneration&Applications

LLM

[Motion Tags]: (translation + rotation) Frames 0-47: static + yaw right Frames 48-67: static + static Frames 68-119: move forward + static

Pre-processing and Trajectory extraction

[Figure 2]

[Figure 3]

Motion Tagging

[Figure 4]

[Figure 5]

time

Multi-modal Input Trajectory Gen Camera-Controlled Video

4

Text

[Figure 6]

###### + A sunflower swaying in the wind.

|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|
|---|---|---|---|---|

[Motion Caption] The camera stays static while rolling right, then moves forward continuing to roll right.

Text + RGBD

[Directorial Caption] The camera moves forward along the valley, showcasing the rolling hills and yellow fields.

[Figure 12]

|[Figure 13]|
|---|

[Figure 14]

+

|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|
|---|---|---|---|

Figure 1. Overview. Top: DataDoP data construction. Given RGB video frames, we extract RGBD images and camera poses, then tag the pose sequence with different motion categories (in different colors). With LLM, we generate two types of captions from motion tags and RGBD inputs: Motion Caption describes the camera movements, while Directorial Caption describes the camera movements along with their interaction with the scene and directorial intent. Bottom: Our GenDoP method supports multi-modal inputs for trajectory creation. The generated camera sequence can be easily applied to various video generation tasks, including text-to-video (T2V) [13] and image-to-video (I2V) generation [15]. GenDoP paves the way for future advancements in camera-controlled video generation.

### Abstract

tory generation remain limited: Traditional approaches rely on geometric optimization or handcrafted procedural systems, while recent learning-based methods often inherit structural biases or lack textual alignment, constraining creative synthesis. In this work, we introduce an auto-regressive model inspired by the expertise of Directors of Photography to generate artistic and expressive camera trajectories. We first introduce DataDoP,

Camera trajectory design plays a crucial role in video production, serving as a fundamental tool for conveying directorial intent and enhancing visual storytelling. In cinematography, Directors of Photography meticulously craft camera movements to achieve expressive and intentional framing. However, existing methods for camera trajec-

a large-scale multi-modal dataset containing 29K realworld shots with free-moving camera trajectories, depth maps, and detailed captions in specific movements, interaction with the scene, and directorial intent. Thanks to the comprehensive and diverse database, we further train an auto-regressive, decoder-only Transformer for high-quality, context-aware camera movement generation based on text guidance and RGBD inputs, named GenDoP. Extensive experiments demonstrate that compared to existing methods, GenDoP offers better controllability, finer-grained trajectory adjustments, and higher motion stability. We believe our approach establishes a new standard for learningbased cinematography, paving the way for future advancements in camera control and filmmaking. Our project website: https://kszpxxzmc.github.io/GenDoP/.

### 1. Introduction

In video production, the camera serves as the window of observation, playing a crucial role in presenting scene content, conveying the director’s intent, and achieving visual effects. In recent years, video generation technology has advanced [1, 3, 25, 50], and several cutting-edge studies have explored camera-controlled video generation [13, 15, 31, 39]. However, these works often rely on predefined, simplistic camera trajectories to demonstrate their results. The generation of artistic, expressive, and intentional camera movements remains largely unexplored.

Trajectory generation has been a long-standing problem. Traditional approaches include optimization-based camera motion planning [4, 11, 29] and learning-based camera control [5, 9, 18, 21]. However, these techniques demand geometric modeling or cost function engineering for each motion, which limits creative synthesis. Meanwhile, oversimplified procedural systems impede precise text control. Recent advances in diffusion-based camera trajectory generation [8, 22, 26] have expanded creative possibilities for text-driven cinematography. However, CCD [22] and E.T. [8] inherit structural biases from human-centric tracking datasets, constraining camera movements to oversimplified character-relative motion patterns. Director3D [26] introduces object/scene-centric 3D trajectories from multiview datasets [47, 51], but the lack of trajectory-level captions limits text-to-motion alignment. As a result, the generated paths are driven by geometric plausibility rather than directorial intent. These dataset constraints hinder the creation of artistically coherent free-moving trajectories that interpret creative vision without relying on specific subjects.

In this work, we tackle the problems above with several key designs. First, we introduce DataDoP Dataset, a multimodal, free-moving camera motion dataset extracted from real video clips, which includes accurate camera trajectories

extracted by state-of-the-art and scene compositions. We extract camera trajectories and corresponding depth maps using MonST3R [48], and employ GPT-4o to generate comprehensive descriptions of the camera trajectories and scene focus, capturing both motion dynamics and directorial intent. DataDoP comprises over 29K shots, totaling 11M frames, with corresponding camera trajectories and diverse textual descriptions. Furthermore, given the inherently sequential nature of camera trajectories, we propose GenDoP, which treats camera parameters as discrete tokens and leverages an auto-regressive model for camera trajectory generation. Our model incorporates multi-modal condition as inputs, including fine-grained textual descriptions and optionally RGBD information from the first frame, to produce stable, complex, and instruction-aligned camera movements.

We conduct rigorous human validation to ensure the dataset quality. Extensive experiments confirm that GenDoP outperforms state-of-the-art methods [8, 22, 26] across fine-grained textual controllability, motion stability, and complexity, while exhibiting enhanced robustness. As AIdriven video creation evolves, multi-modal camera trajectory generation emerges as a timely and crucial direction. We believe that this work paves the way for future advancements in camera-controlled video generation and a wide range of trajectory-related downstream applications.

### 2. Related Work

Camera trajectory datasets. While several existing datasets document camera trajectories, their cinematographic expressiveness remains constrained. Datasets such as MVImgNet [47], RealEstate10K [51], and DL3DV10K [28] provide calibrated trajectories through structured capture methods, but predominantly focus on basic paths around static objects or scenes. These datasets lack the sophisticated cinematographic language necessary for narrative-driven sequencing and intentional viewpoint control. CCD [22] and E.T. [8] emphasize human-centric tracking but are confined to reactive tracking mechanisms. In contrast, DataDoP’s camera movement is driven by the compositional logic of the scene and the narrative demands. We underscore DataDoP’s unique contribution to the field of artistic camera trajectory generation.

Camera trajectory generation. Early efforts in trajectory generation generally consist optimization-based motion planning [4, 11, 29, 30] and learning-based camera control [5, 9, 18, 21]. Recent progress focuses on integrating camera motion with scene and character dynamics. CCD [22] introduced a camera diffusion model using text and keyframe controls, generating motion in character-centric coordinates. E.T. [8] improves to incorporate both character trajectories and camera-character text descriptions as control and generates trajectories in the global coordinates. On the other hand, Director3D [26]

Caption Statistics

Dataset Traj Type Domain

Traj Scene Intent #Vocab #Sample #Frame #Avg (s)

MVImgNet [47] Object/Scene-Centric Captured × × × - 22K 6.5M 10 RealEstate10k [51] Object/Scene-Centric Youtube × × × - 79K 11M 5.5 DL3DV-10K [28] Object/Scene-Centric Captured × × × - 10K 51M 85

CCD [22] Tracking Synthetic ✓ × × 48 25K 4.5M 7.2 E.T. [8] Tracking Film ✓ × × 1790 115K 11M 3.8 DataDoP (Ours) Free-Moving Film ✓ ✓ ✓ 8698 29K 11M 14.4

- Table 1. DataDoP Dataset. We compare the DataDoP dataset to other datasets containing camera trajectories. DataDoP is a large dataset focusing on artistic, free-moving trajectories, each accompanied by high-quality caption annotations. The provided captions detail the camera movements, their interactions with scene content, and the underlying directorial intent. To capture more intricate camera movements, each video clip spans 10-20 seconds, averaging 14.4 seconds.

trains DiT-based framework on object/scene-level multiview datasets to generate object/scene-centric camera trajectories. NWM [2] employs conditional DiT to plan camera trajectory via agents’ egocentric views. Concurrent work [16] employs an auto-regressive transformer to predict the next frame’s camera movement based on past camera paths and images in aerial videography. Our approach goes further by incorporating both text instructions and RGBD spatial information, enabling precise control in generating camera trajectories for cinematic storytelling.

Auto-regressive models. Auto-regressive (AR) modeling employs tokenizers to transform inputs into discrete tokens and formulates generation as a next-token prediction task with transformers. In recent years, great advancements are witnessed in auto-regressive modeling in image [12, 33, 41, 45], video [24, 42, 44], and 3D generation [7, 38, 40]. Early approaches [33, 45] serialize images into patch tokens and train a transformer to autoregressively model the text and image tokens in a sequential data stream. VAR [41] reformulates auto-regressive image generation as coarse-to-fine next-scale prediction. VideoPoet[24] leverages bidirectional attention for multimodal input conditioning in auto-regressive video generation. Our work extends auto-regressive modeling to camera trajectory generation controlled by text and geometry cues, leveraging the discrete nature of camera tokens. Compared to diffusion-based methods, our model generates precise, coherent, and intricately detailed artistic trajectories for long camera pose sequences.

### 3. DataDoP Dataset

We introduce DataDoP, a camera trajectory dataset extracted from long shots in artistic films, including both movies and documentaries, designed to capture freemoving, intricate, and expressive camera movements. As shown in Fig. 1, each sample in DataDoP consists of a shotlevel camera trajectory, accompanied by the corresponding RGBD images and two types of trajectory captions: Motion captions, which accurately describe the camera motion alone, and Directorial captions, which detail the camera movements, their interaction with the scene, and the directorial intent. We describe the data construction pipeline in

Sec. 3.1 and the dataset statistics in Sec. 3.2.

##### 3.1. Dataset Construction

Pre-processing. We curate and filter artistic videos from the internet, which are then segmented into shots using PySceneDetect 1. Captions are removed using VSR 2, after which the shots are merged with a publicly available subset from MovieShots [34]. A filtering process is applied to retain shots between 10 and 20 seconds in length, while removing those that are excessively dark or nearly static. Since our dataset focuses on free-moving camera trajectories, which enable unrestricted 3D camera motion within scenes and events, rather than tracking moving people or objects, we specifically filter for this category of data. GPT4o [19] was used to categorize the shots, removing those with static cameras or object-tracking motion. For details, please refer to Appendix A.2.

Trajectory extraction. We then utilize MonST3R [48] to estimate the geometry of dynamic scenes. Camera trajectories are extracted along with the corresponding depth maps. The trajectories are subsequently cleaned, smoothed, and interpolated into fixed-length sequences.

Motion tagging. We then partition the camera trajectories into segments of motion tags. Compared to existing datasets [8, 22], our captions explicitly incorporate descriptions of camera rotation, enabling more fine-grained characterization of camera movements. As a result, our motion tags include both translation and rotation components (see Fig. 2a). For camera translation, excluding the static state, we consider six fundamental motions across three degrees of freedom: lateral (left/static/right), vertical (up/static/down), and depth (forward/static/backward). Each translation motion can be categorized into one, two, or three motions, resulting in a total of 27 possible combinations. For camera rotation, aside from the static state, we consider six fundamental motions across three degrees of freedom: pitch (up/down), yaw (left/right), roll(left/right), resulting in 7 base actions. We do not consider the combination of these rotations, as in practical scenarios, rotation typically involves only one of these basic motions at a time.

- 1https://github.com/Breakthrough/PySceneDetect
- 2https://github.com/YaoFANGUK/video-subtitle-remover

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

The camera continuously moves backward throughout the entire sequence.

The camera moves continuously to the right while yawing left throughout the entire sequence.

[Figure 23]

(a) Distribution of Translation and Rotation Motion Tags.

(b) Diverse Trajectories.

- Figure 2. Dataset Statistics. (a) The figure illustrates the composition and distribution of 27 translation motions (left) and 7 rotation motions (right), emphasizing the complexity and diversity of trajectories in our DataDoP dataset. (b) Based on the same caption, our dataset includes diverse trajectories that still conform to the given caption. As shown in the figure, the trajectories exhibit variations in terms of length, direction, and speed, effectively showcasing the diversity within our dataset.

Moving. Static shots keep the camera fixed. Object/SceneCentric shots capture multi-view data focusing on specific objects or scenes. Tracking shots track a moving subject. Free-Moving shots allow unrestricted 3D camera motion, enabling complex scene exploration and dynamic framing, crucial for cinematic storytelling and creative expression. As shown in Tab. 1, DataDoP stands out by uniquely focusing on artistic, free-moving trajectories, capturing the director’s creative vision and offering significant cinematic and artistic value. Unlike tracking shots, where the camera follows a specific object, free-moving shots fluidly navigate the scene, enhancing visual storytelling without constraints.

Alignment Video-Traj Traj-Motion Traj-Directorial Quality

Score

Acc 0.863 0.913 0.858 0.945 Kappa 0.642 0.530 0.502 0.551

- Table 2. Dataset User Study. Our user study demonstrates that our dataset exhibits excellent quality and human-alignment, with proven reliability of the results.

We simplify by assuming that camera translation and rotation are completely independent, which results in a total of 27 × 7 possible combinations for camera motion tags.

We adopt the motion tagging method from E.T. [8] to process the camera trajectories. For translation, we use an initial velocity threshold and velocity difference thresholds in different directions to determine the dominant velocity direction combinations. For rotation, we use an initial rotational velocity threshold to identify the unique dominant rotational direction. Finally, we combine the translation and rotation information to generate the complete tags, and apply smoothing to remove noise and sparse tags. These methods provide a coarse temporal description of the camera trajectories, as shown in Fig. 1.

Caption generation. Finally, we generate two types of trajectory captions based on the motion tags obtained in the previous stage, as shown in Fig. 1. First, we structure the motion tags by incorporating context, instructions, constraints, and examples, and then leverage GPT-4o to generate Motion captions that describe the camera motion alone. Next, we extract 16 evenly spaced frames from the shots to create a 4 × 4 grid and prompt GPT-4o to consider both the previous caption and the image sequence. This enables GPT-4o to generate Directorial captions that describe the camera movement, the interaction between the camera and the scene, as well as the directorial intent. Further details can be found in Appendix A.2.

- 3.2. Dataset Statistics

Data scale. DataDoP is built on long shots from the Internet. As shown in Tab. 1, it consists of 29K samples, spanning 12M frames and totaling 113 hours of footage, all with high-quality trajectory annotations. The dataset focuses on long shots averaging 14.4 seconds, capturing more complex camera movements compared to other datasets. While DL3DV-10K [28] has a longer average duration, its camera trajectories lack directorial intent, emphasizing scene-level consistency rather than creative camera work.

Statistics. We present the dataset statistics across four dimensions: Alignment, Quality, Complexity, and Diversity. To evaluate Alignment and Quality, we conducted a user study with 8 experts. We selected 100 samples, including original videos, camera trajectories, and two captions: Motion and Directorial. The samples were split into two sets, each labeled by four users. For Alignment, we assess the consistency between the trajectory and video (VideoTraj), the motion caption and trajectory (Traj-Motion), and the directorial caption with both the trajectory and video scene (Traj-Directorial). For Quality, we assess whether the camera trajectory is free of breaks, roughness, or jitter. We use Fleiss’ Kappa [10] to measure inter-rater agreement among multiple users. As shown in Tab. 2, our dataset achieves high accuracy in both Alignment and Quality, with all Kappa values exceeding 0.4, confirming the reliability of the results. For Complexity, as illustrated in Fig. 2a,

Trajectory types. We classify camera trajectories into four types: Static, Object/Scene-Centric, Tracking, and Free-

[Figure 24]

[Figure 25]

The camera remains static, then moves right, followed by moving forward while yawing right, and finally moving left and forward

Encoder

Latent code

Pose tokens

Text

Text

[Figure 26]

[Figure 27]

De-tokenizer

#### Trajectory

[Figure 28]

Decoder

while continuing to yaw right.

Encoder

concat

Image

[Figure 29]

RGBD (optional)

[Figure 30]

Camera Trajectory Tokenization

Encoder

Depth

R R R R T T T F F S

[Figure 31]

Trainable Network

N × Quaternion Translation Intrinsic Scale

- Figure 3. Our Auto-regressive Generation Model. Our model supports multi-modal inputs and generates trajectories based on these inputs. By treating the task as an auto-regressive next-token prediction problem, the model sequentially generates trajectories, with each new pose prediction influenced by previous camera states and input conditions.

we present the composition and distribution of motion tags within the dataset. For Diversity, as shown in Fig. 2b, the trajectories, while remaining consistent with the caption, exhibit significant variations in length, direction, and speed, effectively showcasing the diversity within our dataset.

- 4. Method

is detailed in Sec. 4.2, while the generation method is comprehensively described in Sec. 4.3.

##### 4.2. Camera Trajectory Tokenization

Auto-regressive models commonly process information as discrete token sequences, making compact tokenization essential for efficient representation without sacrificing accuracy. Videos are naturally serialized into discrete frames and in this sense, camera trajectories from videos can be easily tokenized into discrete camera pose xi = [Ri|ti|Ki] at each frame. This simplicity facilitates efficient tokenization, enabling a compact encoding of the trajectory.

##### 4.1. Overview

We introduce GenDoP here, an auto-regressive method for camera trajectory generation. Previous trajectory generation methods [2, 17, 22, 26, 27, 40] largely relied on diffusion models [35], which often result in discontinuous and unstable trajectories (See Fig. 4). In contrast, we pioneer the application of auto-regressive models to trajectory generation. Auto-regressive models are well-suited for this task due to their ability to capture sequential dependencies. In trajectory, each pose’s position and orientation depend on the previous one, making the framework ideal for modeling the temporal and spatial continuity of trajectories. By conditioning each pose on its predecessor, the model effectively generates realistic and coherent 3D camera trajectories.

Canonical normalization. We first establish a scaleinvariant trajectory representation via canonical normalization. The camera frame is aligned as the world reference, setting Rnorm0 = I and tnorm0 = 0. Subsequent poses are relativized through rigid transformation: Rnormi = R⊤0 Ri, ˆti = R⊤0 (ti − t0) for i ∈ [1,N). Scale normalization then computes s = max1≤i<N ∥ˆti∥2 and projects translations to unit space via tnormi = ˆti/(s+ϵ) with ϵ = 10−5, maintaining geometric consistency and numerical stability.

Trajectory tokenization. For the resulting normalized parameters Rnormi and tnormi , we compute the corresponding quaternion representation for Rnormi and normalize all parameters to the range [0,1], resulting in the vector (r1,r2,r3,r4,t1,t2,t3). Subsequently, the focal lengths fx,fy and the scale size s are also normalized, yielding (f1,f2,s), which are then concatenated with the previously computed values. Finally, these parameters are multiplied by the discrete bin size B and converted into integer values. Thus, for each xi, we can tokenize it into an integer vector of length 10, where the values are within the range [0,B]. As a result, each camera trajectory can be tokenized into an integer vector of length 10N.

GenDoP automatically constructs the camera’s 3D motion path based on an input caption or appearance and geometry from the initial frame, capturing changes in both position and orientation. As illustrated in Fig. 3, GenDoP takes a text description T, optionally combined with the initial frame’s RGBD image (I0,D0), as input and generates the corresponding camera trajectory C. A camera trajectory C = {x0,x1,...,xN−1} is defined as a sequence of N consecutive camera poses, where each pose xi = [Ri|ti|Ki] comprises a rotation matrix Ri (orientation), a translation vector ti (position), and an intrinsic matrix Ki (projection parameters). The intrinsic matrix Ki can be simplified to (fx,fy), assuming a fixed principal point (cx,cy) and image dimensions (H,W). Our goal is to derive an auto-regressive generation function f such that C = f(T[,(I0,D0)]). The trajectory tokenization process

Auxiliary tokens. Similar to prior auto-regressive approaches [7, 38, 40], we prepend a BOS token at the beginning of a trajectory sequence, append an EOS token at the end, and use PAD tokens to fill the necessary positions.

Text-Trajectory Alignment Trajectory Quality User Study (AUR)

Condition Method Dataset

F1-Score↑ CLaTr-CLIP↑ Coverage↑ CLaTr-FID↓ Alignment↑ Quality↑ Complexity↑

CCD [22] Pre-trained 0.297 5.288 0.332 357.822 3.013 3.022 3.273 E.T. [8] Pre-trained 0.330 2.450 0.020 609.906 1.227 1.067 1.067 Director3D [26] Pre-trained 0.058 0.000 0.171 542.385 2.313 3.110 2.453 Director3D [26] DataDoP 0.391 31.689 0.839 31.979 3.753 3.260 3.493 GenDoP(Ours) DataDoP 0.400 36.179 0.872 22.714 4.693 4.573 4.713

Motion

CCD [22] Pre-trained 0.315 4.247 0.416 240.216 2.950 3.050 3.217 E.T. [8] Pre-trained 0.319 0.000 0.014 758.923 1.309 1.092 1.184 Director3D [26] Pre-trained 0.126 0.000 0.348 348.312 2.333 2.867 2.342 Director3D [26] DataDoP 0.361 23.505 0.802 35.538 3.808 3.467 3.683 GenDoP(Ours) DataDoP 0.399 32.408 0.854 34.275 4.617 4.557 4.575

Directorial

RGBD & Text GenDoP (Ours) DataDoP 0.388 30.231 0.855 33.653 - - -

- Table 3. Quantitative Results. We present the quantitative results of our GenDoP across two text-conditional generation tasks and an RGBD & Text-conditioned task, comparing it with human-tracking methods CCD [22] and E.T. [8], as well as the object/scenecentric method Director3D [26]. Our model consistently outperforms all baselines across all metrics and caption subsets, confirming the effectiveness of both our dataset and auto-regressive framework, positioning GenDoP as a state-of-the-art trajectory generation model.

During the tokenization process, we obtain B + 1 integer values. Consequently, this tokenized representation can be discretized through a learnable codebook V ∈ R(B+4)×L, where L is the latent dimension.

##### 4.3. Auto-regressive Generation

We employ a transformer-based auto-regressive architecture to establish a bidirectional mapping between fixedlength camera trajectories and their compact latent representations. Although raw camera trajectories may vary in length, their spatial paths remain consistent after interpolation, allowing us to target fixed-length camera trajectories.

Text-conditioned encoder. Our architecture comprises a text encoder ET and an auto-regressive decoder D for the base text-conditioned model, as shown in Fig. 3. The text encoder ET utilizes the pretrained and learnable text encoder from Stable Diffusion 2.1 (SD2.1) [35] to extract semantic features, which are then processed through an MLP to generate a textual latent code ZT ∈ RM

T×L, where MT denotes the textual latent size and L is the latent dimension. RGBD-conditioned encoder. For the RGBD-conditioned model, we introduce two separate encoders: EI for RGB image and ED for depth. Specifically, we expand the depth to R3×H×W to ensure it can be processed by the encoder. Both encoders use the pretrained and learnable CLIP Vision Model [20, 32, 36] to extract features, which are then passed through MLPs to generate latent codes ZI ∈ RM

I×L and ZD ∈ RM

D×L. The final latent representation is the concatenation of the textual, RGB, and depth codes:

###### Z = [ZT;ZI;ZD] ∈ RM×L,M = MT + MI + MD. (1)

This combined representation integrates both visual and geometric modalities, conditioning the trajectory generation on the accompanying textual information.

Auto-regressive decoder. The decoder D is an autoregressive transformer designed to generate a trajectory token sequence from the latent code Z and previously token

IDs. We adopt the OPT architecture [49] as the decoder, as utilized in prior works [7, 40]. The latent code Z is prepended to the input sequence, positioned before the BOS token. For each token prediction, the decoder D queries the learnable codebook V ∈ R(B+4)×L using the previous token IDs y0:P−1, producing the corresponding continuous token embeddings V [y0:P−1] ∈ RP×L, where P denotes the length of the previous token sequence. The input embeddings for the decoder are then computed as:

XP = PosEmbed([Z;V [y0:P−1]]) ∈ R(M+P)×L. (2)

Stacked causal self-attention layers are then employed to predict the next feature based on XP. A linear projection is applied to map the predicted feature to classification logits, which are subsequently used to retrieve the corresponding token ID yP. This process ultimately generates a fixedlength trajectory token sequence.

Loss function. The model is optimized with a weighted sum of cross-entropy loss and a regularization term:

L = CrossEntropy(S[1 :],Sˆ[:,−1]) + λ∥Z∥22, (3)

where S is the one-hot ground truth token sequence, Sˆ is the predicted logits, and Z is the latent code.

### 5. Experiments

##### 5.1. Experimental Setting

Our GenDoP framework implements three conditional generation paradigms: (1) Motion captions for isolated camera motions, (2) Directorial captions for scene-synchronized trajectories, and (3) RGBD & Text, a novel approach that integrates images and depth maps with Directorial captions through hierarchical feature fusion.

All experiments for both training and inference are carried out with an Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz and a single NVIDIA A100-SXM4-80GB GPU. We maintain consistency in parameters and strategies

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Director3D (DataDoP)

Director3D (Pre-trained)

E.T. (Pre-trained)

Ours (DataDoP)

CCD (Pre-trained)

Input

The camera starts by moving left while yawing left, remains static, and then moves left and forward while yawing right.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

The camera initially moves left while yawing left, then

moves forward while continuing to yaw left, and finally becomes static.

- Figure 4. Qualitative Results of Text-conditioned Trajectory Generation. We offer a comparative analysis of text-conditioned trajectory generation in the figure. Our model’s trajectories (color-coded to highlight text alignment) remain stable and closely follow the instructions, while other models exhibit significant jitter or fail to match the instructions well.

throughout training to ensure uniformity across the experimental setup. The image resolution is set to W = 512, with a trajectory length of N = 60, discrete bin size B = 256, and latent dimension L = 1024. The textual latent size is MT = 77, the image latent size is MV = 257, and the depth latent size is MD = 257. The backbone OPT Transformer consists of 12 layers with 12 attention heads each. Training converges after 8 hours on a single A100 GPU, yielding an inference throughput of approximately 3 seconds per trajectory. For evaluation, 3k samples are randomly selected from the DataDoP dataset as the test set, with the remaining data forming the training corpus. Implementation specifics are detailed in Appendix C.1.

##### 5.2. Quantitative Results

Metrics. We obtain the Contrastive Language-Trajectory embedding (CLaTr) [8] by leveraging the DataDoP dataset with a CLIP-like approach [32]. A random subset of 3k samples is selected as a test set, from which CLaTr embeddings for both ground truth (GT) and generated data are extracted. Using these embeddings, we evaluate the model with two main metrics. (1) Text-Trajectory Alignment: We measure the similarity between text and trajectory embeddings using the CLaTr-CLIP (analogous to CLIP-Score [20]). We also replicate the motion tagging step from Sec. 3.1 to obtain the GT motion tags, which are then compared with the generated tags. Classifier F1-Score is computed by verifying the generated motion tags against the GT labels. (2) Trajectory Quality: The alignment between GT and generated trajectories is evaluated using CLaTr-FID (analogous to FID [14]). Additionally, Coverage evaluates how well the generated data spans the range of real data, with higher values indicating a broader representation of the data distribution.

Results. We report the quantitative results of our GenDoP across two text-conditional generation paradigms (Motion / Directorial) in Tab. 3. We compare it with previous trajectory generation methods. For the human-tracking methods, CCD [22] and E.T. [8], we assume the character remains static to simplify the camera trajectory inference process.

For the object/scene-centric, text-only conditioned method, Director3D [26], in addition to the pretrained model, we also train a version using DataDoP to emphasize the significance and effectiveness of our dataset for camera trajectory generation tasks. For the RGBD & Text-conditioned task, a novel paradigm introduced by us, we present only the metric results for GenDoP.

Our model demonstrates consistent superiority across all metrics and caption subsets, primarily due to the enhanced trajectory complexity and trajectory-aware captions in our dataset. This innovation enables more precise motion representation, significantly enhancing text-trajectory alignment. This is demonstrated by Director3D models trained on our dataset, which show a dramatic leap in CLaTr-CLIP scores from 0 to over 30, transitioning from object-centric to trajectory-enriched training. Despite sharing the same training data, GenDoP outperforms DataDoP-trained Director3D by 4.5 (Motion) and 9.1 (Directorial) for CLaTrCLIP, while reducing CLaTr-FID by 9.3 (Motion) and 1.3 (Directorial), as confirmed by user studies. These results validate the effectiveness of our auto-regressive framework. Additionally, GenDoP demonstrates exceptional versatility in handling RGBD & Text-conditional tasks, showing strong multi-modal integration for high-quality trajectory generation under complex constraints. Collectively, the experiments confirm the effectiveness of both our dataset and auto-regressive framework, establishing GenDoP as a stateof-the-art model for trajectory generation.

User study. To establish human-aligned evaluation metrics, we engaged 27 domain experts in a user study centered on three critical dimensions: Alignment (trajectory consistency with input text), Quality (smoothness, logical coherence, and seamless connectivity between sequential actions), and Complexity (kinematic sophistication of motion sequences under input constraints). We employed the Average User Ranking (AUR) metric to evaluate model performance, where domain experts assigned ranking scores (1-5) to the five competing models per task. Higher ranking scores indicate superior performance. We comparatively assessed five models on text-conditioned tasks (with 10 sam-

###### Input RGBD & Text-conditioned

###### Text-conditioned

[Figure 42]

[Figure 43]

The camera remains still, then glides left and forward, highlighting wall displays and a central table, before settling static.

|[Figure 44]<br><br>|
|---|

[Figure 45]

[Figure 46]

The camera moves forward to capture a group conversing, stays static for clarity, moves closer to highlight details, and holds still to focus on their interactions.

[Figure 47]

- Figure 5. Qualitative Results of RGBD & Text-conditioned Generation. This figure compares the impact of incorporating RGBD input on trajectory generation under identical text conditions. While both models generate command-compliant trajectories, the RGBD & Text-conditioned model demonstrates superior scene adaptation by utilizing RGBD data to integrate geometric and contextual constraints.

Ablation Text-Traj Alignment Trajectory Quality Encoder Norm F1-Score CLaTr-CLIP Coverage CLaTr-FID

✓ ✓ 0.400 36.179 0.872 22.714 ✓ × 0.322 14.917 0.766 68.590 × ✓ 0.389 31.420 0.866 22.841

- Table 4. Ablation Study. We conduct an ablation study to evaluate the effectiveness of canonical normalization (see Sec. 4.2) and the trainability of the encoder (see Sec. 4.3).

ples per task), excluding RGBD & Text-conditional scenarios with single-model baselines. As evidenced in Tab. 3, our approach outperformed others across all metrics, with results closely matching the earlier quantitative findings, validating its perceptual and technical coherence.

- 5.3. Qualitative Results

Text-conditioned Generation. We present comparative analysis of Text-conditioned trajectory generation in Fig. 4. Our model not only achieves superior text-trajectory alignment but also maintains high-quality trajectory generation. Furthermore, the intricate input conditions highlight its capacity to produce sophisticated outputs with high-level complexity. In contrast, the DataDoP-trained Director3D captures basic motion patterns but exhibits trajectory jitter and instability. Furthermore, its object-centric variant pre-trained on [47, 51] generates orbit-dominated trajectories that exhibit no text correspondence, despite improved smoothness. Other baselines exhibit notably inferior performance in both text-trajectory alignment and quality.

RGBD & Text-conditioned Generation. We conduct a comparative analysis of our trajectory generation model un-

der varying input conditions, as shown in Fig. 5. The results demonstrate that both models generate command-compliant trajectories when given identical textual inputs. However, the RGBD & Text-conditioned model shows superior scene adaptation by leveraging RGBD to incorporate geometric and contextual constraints. Specifically, as shown in the first row of Fig. 5, the spatial information from RGBD effectively mitigates ambiguities, i.e., “left and forward” in the text. This multimodal conditioning enables precise alignment with the 3D scene structure.

##### 5.4. Ablation Studies

Canonical normalization. We experiment with an alternative strategy that skips canonical normalization, directly using trajectories from Monst3r [48] with scale normalization for tokenization feasibility. These trajectories are scenecentered, with the 3D space focused around the scene. In contrast, canonical normalization transforms them into firstperson tracking paths. As shown in the table Tab. 4, applying canonical normalization significantly improves both alignment and quality, providing more consistent camera movements align with the instructions.

Trainable Encoder. Contrary to conventional practice in text/image-conditional generation where pretrained encoders remain frozen to preserve prior knowledge, our experiments demonstrate comprehensive performance gains (see Tab. 4) by employing trainable encoders. This improvement arises from the encoders’ ability to adapt and bridge cross-modal gaps: through joint optimization, the visual encoder creates geometry-aware trajectory embeddings, while the text encoder learns motion-semantic relationships, resulting in more accurate alignment between text and camera movements.

### 6. Conclusion

We propose DataDoP, a pioneering dataset of expressive, free-moving camera trajectories from artistic videos, and GenDoP, an auto-regressive multimodal model for trajectory generation. Our approach innovatively incorporates RGBD information as input, enabling spatial data to guide trajectory supervision. This sets a new benchmark, achieving state-of-the-art performance with superior controllability and intent alignment compared to existing methods.

Limitations and future work. Currently, our multimodal approach combines text and first-frame RGBD to generate trajectories. Meanwhile, our dataset also extracts 4D point cloud during the extraction process but remains underexplored. Looking ahead, we aim to incorporate more modalities to enhance the adaptability and contextual awareness of the generated trajectories. In addition, we plan to unify trajectory and camera-controlled video creation for iterative creation of both trajectories and video content, establishing a seamless pipeline for automated, artistic film production.

### Appendix

The appendix provides detailed supplementary material on the DataDoP dataset and GenDoP method. It outlines data availability, ensuring compliance with YouTube’s policies and detailing our data sharing practices. The dataset construction process is described in detail, including shot collection, quality filtering, and semantic categorization using GPT-4o [19]. Additionally, we provide further dataset statistics. The appendix also explains the tokenization details of camera trajectory data for model processing. Finally, it includes information on the experimental setup, along with additional ablation studies

### A. DataDoP Dataset

##### A.1. Data Availability Statement and Clarification

We are dedicated to upholding transparency and compliance in our data collection and sharing practices. Please take note of the following:

- • Publicly Available Data: The data utilized in our studies is sourced from publicly available repositories. We do not access any exclusive or private data sources.
- • Data Sharing Policy: Our data sharing policy is in line with established practices from previous works, such as [6]. Instead of providing raw data, we furnish YouTube video IDs essential for accessing the content.
- • Usage Rights: The data we release is exclusively meant for research purposes. Any commercial use is not permitted under this agreement.
- • Compliance with YouTube Policies: Our data collection and sharing practices strictly adhere to YouTube’s data privacy and fair use policies. We ensure that user data and privacy rights are respected throughout the process.
- • Data License: The data is distributed under Creative Commons Attribution 4.0 International License (CC BY 4.0).

Furthermore, the DataDoP dataset is provided solely for informational purposes. The copyright for the original video content remains with the respective owners. All DataDoP videos are sourced from the internet and are not owned by our institution. We disclaim responsibility for the content and interpretation of these videos. In relation to the future open-source version, researchers must agree not to reproduce, duplicate, sell, trade, resell, or exploit any portion of the videos or derived data for commercial purposes, and refrain from copying, publishing, or distributing any part of the DataDoP dataset.

##### A.2. Construction Details

Pre-processing. Our dataset construction involves a multistage curation process:

• Shot Collection: A curated collection of cinematographically significant films and documentaries forms the foun-

dation of our dataset. Using PySceneDetect1, we extract 43k initial shots through content-aware boundary detection. An optimized VSR pipeline2 is employed to eliminate textual overlays while maintaining visual integrity. To enhance processing speed and reduce misclassification, we focus the check area on the lower 1/5 of the frame. Finally, we merge this dataset with a publicly available subset of MovieShots [34] to further diversify stylistic elements.

- • Quality Filtering: We retained only shots with a duration between 10 and 20 seconds. Statistics can be seen in Fig. R1a. Then we exclude sequences with static frames and low-light conditions. For each shot, we calculate the pixel-wise similarity between all pairs of consecutive frames. The similarity between two frames F1 and F2 is defined as:

S(F1,F2) = i,j

I(F1(i,j) = F2(i,j)) H × W

,

where F1(i,j) and F2(i,j) represent the pixel values at position (i,j) in frames F1 and F2, respectively, and H and W are the height and width of the frames. To identify static frames, we compute the average similarity S between all consecutive frame pairs in the shot. Specifically, for a shot with N frames, the average similarity is calculated as:

S =

1 N − 1

N−1

k=1

S(Fk,Fk+1),

where Fk and Fk+1 are consecutive frames in the shot, and N −1 is the number of consecutive frame pairs. If the average similarity S exceeds a threshold (e.g., S > 0.6), the entire shot is considered static and excluded. For each shot, the average brightness (mean gray value) for all frames is computed using the following formula:

B =

1 N × H × W

N

k=1

H

i=1

W

j=1

Fk(i,j),

where N is the total number of frames in the shot, H and W are the height and width of each frame, and Fk(i,j) represents the pixel value at position (i,j) in frame k. If the average brightness of a shot is below a predefined threshold (e.g., B < 15), the shot is classified as too dark and excluded from the dataset.

- • Semantic Filtering: We developed an automated categorization pipeline using GPT-4o. Following the definitions in Sec. 3.2, shots are classified into categories. Leveraging GPT-4o [19], we automate the categorization of shots into Static, Free-Moving, and Tracking. Shots

- 1https://github.com/Breakthrough/PySceneDetect
- 2https://github.com/YaoFANGUK/video-subtitle-remover

[Figure 48]

[Figure 49]

(a) Shot Length. (b) Trajectory Scale.

Figure R1. Dataset Statistics in terms of video shot length and trajectory scale.

classified as Object/Scene-Centric, which are common in multi-view datasets, are not considered in this study. We then discard the Static and Tracking shots. The detailed process and examples are shown in Fig. R2.

Trajectory Extraction. We use MonST3R [48] to estimate the geometry of dynamic scenes, generating a time-varying dynamic point cloud along with per-frame camera poses and intrinsics in a feed-forward manner. This enables efficient video depth estimation and reconstruction [43]. Camera trajectories, along with the corresponding depth maps, are extracted for further processing. These trajectories undergo a series of steps including cleaning, smoothing, interpolation, and standardization into fixed-length sequences, ensuring their suitability for subsequent training.

- • Cleaning the Trajectories: To clean the camera trajectories, we first extract the camera translations from the transformation matrices and compute the velocities between consecutive frames. A threshold is determined based on the 95th percentile of the velocity distribution, with an outlier exclusion factor α (set to 18.0). Frames with velocities exceeding this threshold are discarded. The remaining valid frames are then grouped into consecutive segments, ensuring that each segment contains at least 5 frames.
- • Smoothing the Trajectories: After the trajectories have been cleaned, a Kalman [23] filter, based on a Constant Velocity model, is applied to smooth the valid frames within each segment. The smoothing process is performed using process and measurement noise standard deviations of 0.5 and 1.0, respectively. The smoothed segments are subsequently recombined with the original poses, resulting in a cleaned and smoothed camera trajectory. This smoothing step serves to reduce noise and enhance the stability and accuracy of the trajectory, facilitating more reliable analysis in subsequent stages.

• Interpolation into Fixed-Length Sequences: To ensure consistency across the trajectory data for downstream deep learning tasks, we standardize trajectories of varying lengths into fixed-length input sequences, addressing issues related to inconsistent time steps. First, spherical linear interpolation (SLERP) [37] is applied to the rotational component, while the translational component is interpolated linearly, ensuring smooth transitions between frames. The interpolated data is then padded to a fixed length of 120 frames, ensuring uniform time steps across all trajectory samples. This process guarantees that the input sequences are consistent in length and temporal structure, providing stable and reliable training data for deep learning models.

Motion Tagging. We present the distribution of translation and rotation combinations in Fig. R3. As shown, simpler motion combinations are more frequent, but motion tags still exhibit high diversity and complexity.

Caption Generation. In Fig. R4, we present the specific prompts and cases for generating two types of captions.

### B. GenDoP Method

##### B.1. Camera Trajectory Tokenization Details

The camera trajectory tokenization process converts continuous camera parameters into discrete tokens. For the normalized parameters Rnormi (rotation) and tnormi (translation) obtained after canonical normalization, the rotation matrix Rnormi is converted into a unit quaternion q = (r1′ ,r2′ ,r3′ ,r4′ ), and the translation values tnormi = (t′1,t′2,t′3) are processed. This results in the combined vector (r1′ ,r2′ ,r3′ ,r4′ ,t′1,t′2,t′3). Subsequently, the focal lengths fx, fy, and the scale factor s are also acquired. The process involves three key stages:

###### • Rotation and Translation Normalization: The rotation

###### [Prompt]

The images above are 16 frames sequentially extracted from a video, arranged in a 4x4 grid, from left to right and top to bottom. Answer the following questions based on the sequence of images.

Describe the trajectory of the camera’s movement in the first paragraph. Provide a detailed and accurate description of how camera changes in position, orientation, angle, and distance throughout the scene.

In the second paragraph, classify the video into one of the following three categories:

- 1. The camera is stationary (with possible slight shaking), while the object change its position, or both may remain stationary.
- 2. The camera moves relative to the object/scene it is focused on, encompassing changes in position, orientation, angle, and distance, while the object/scene itself remains static (part of it may move slightly in place).
- 3. The camera tracks the movement of the objects as they changes positions and adjusts its focus accordingly.

Output 1/2/3 according to the above standard. Then, explain the reasoning behind the classification. If the camera is stationary, please make sure to choose option 1. If the camera tracks an object’s motion from a fixed relative position, please choose option 3. For example, '2, In the video, the camera gradually approaches the object, causing it to become larger in the center of the screen.‘

###### [Output] – Free-Moving

[Figure 50]

The sequence of images suggests that the camera initially starts with a close-up shot focused on the eyes and face of a character. As the frames progress, the camera appears to zoom out, revealing more of the character's face and upper body. The camera's position gradually shifts from a tight close-up to a medium shot, showing the character in more detail and eventually capturing a surrounding environment and additional figures in the background. The framing transitions smoothly, indicating a steady camera movement rather than sudden shifts.

- 2, The video is classified into this category because the initial focus is on the character’s face, and as the sequence progresses, the camera moves back to encompass more of the scene and reveal other elements. This indicates that the camera is moving relative to the character to show a wider view, rather than the character staying stationary with a static camera or the camera following a moving character across a space.

[Figure 51]

[Output] – Static

Based on the sequence of images, the camera appears to stay relatively static, focusing on a subject who is using a communication device. There are no noticeable shifts in the camera’s position relative to the subject, indicating that the camera does not move significantly. The subject seems to remain more or less in the same position throughout the frames, suggesting a stationary setup with only slight changes in the framing due to minor movements by the subject rather than the camera.

1, The video likely falls into the first category: "Camera is stationary (with possible slight shaking), and the object moves." This is because the camera remains fixed on the subject, who appears to be the one making subtle movements, such as talking or holding an object to their mouth, while the camera itself does not shift its position.

[Figure 52]

[Output] – Tracking

In the sequence of images, the camera follows the character closely, initially positioned behind her as she moves toward a doorway (frames 1-4). The camera then maintains its focus on the character from a rear perspective as she goes down a stairway (frames 5-8), showing a steady orientation and slight adjustments in angle to maintain the character in view. As the character exits into an open area, the camera's position shifts to a broader scene, capturing her from back and slightly to the side (frames 9-12), transitioning smoothly to a more distant angle that reveals more of the environment. The final frames (13-16) involve a dramatic upward tilt, panning to capture the collapsing buildings above from a low angle, indicating a significant change in camera orientation and distance from the character, while still emphasizing the surrounding chaos.

- 3, The video shows the camera following the character’s movement throughout the scene. It adjusts its position and angle to maintain focus as she exits a building and enters an open space. The camera moves in relation to her, maintaining a dynamic perspective as the scene unfolds and dramatic events occur..

- Figure R2. Semantic Filtering. Following the definitions in Sec. 3.2, shots are classified. Leveraging GPT-4o [19], we automate shot categorization into Static, Free-Moving, and Tracking. Shots categorized as Object/Scene-Centric, common in multi-view datasets, are not considered in films.

and translation components are tokenized as follows:

rk′ + 1 2

, k ∈ {1,...,4},

rk =

t′k + 1 2

, k ∈ {1,...,3}.

tk =

This normalization maps values from the range [−1,1] to [0,1], while preserving the constraints on both rotation

and translation.

###### • Focal Length Adaptation: Normalize focal lengths

(fx,fy) relative to the principal points (cx,cy):

fx 10cx

fy 10cy

f1 =

,f2 =

.

Here, cx and cy typically represent half the image dimensions. The factor of 10 ensures that the focal length values remain within the range (0,1), accommodating typi-

[Figure 53]

[Figure 54]

[Figure 55]

- Figure R3. Tag Distribution. The distribution of Translation and Rotation combinations is shown in the figure. Different tag modes are represented by shades of yellow, ranging from deep to light: Static, Translation only, Rotation only, and both Translation and Rotation.

cal focal lengths.

- • Scale Parameter Transformation: Apply logarithmic compression to the scale factor s:

s =

log10(s′) + 2 4

.

This transformation enables a linear representation of multiplicative scale changes across three orders of magnitude (0.01 ≤ s′ ≤ 100).

- • Parameter Clamping and Discretization: All normalized parameters are clamped to the range [0,1] before discretization into N bins:

ptoken = ⌊p · N⌋, ∀p ∈ r1,...,t3,f1,f2,s.

This process generates a compact 10-dimensional token that preserves the relative geometric relationships between parameters. The hyperparameter N controls the trade-off between quantization error and codebook size.

### C. Experiments

##### C.1. Experiments Setting Details

We train GenDoP with a batch size of 16 using the AdamW optimizer, with a learning rate of 1e-5, (β1,β2) =

(0.9,0.95), and a weight decay of 0.01. The KL loss weight is set to 1e-8. We use a gradient accumulation step of 1. Training is performed using bfloat16 mixed precision. The model converges with the best results at the 100th epoch.

##### C.2. Additional Qualitative Results

In the supplementary video, we present additional cases with text prompts randomly generated by the LLM model [19]. These text prompts have never been seen in the training set, creating a certain gap compared to the captions in our dataset. However, as shown in the video, despite the differences between the generated prompts and the training data, our model is still able to generate precise, high-quality, complex, and artistic trajectories.

Furthermore, we use TrajectoryCrafter [46] to showcase how our trajectory generation method can be applied to camera-controlled video generation. This allows for the creation of videos that align with the camera descriptions provided, ensuring the generated video sequences match the specific visual and motion criteria described by the camera control inputs.

###### [Prompt to generate Motion captions]

You act as a camera operator writing a technical script for camera motion descriptions. Given a rough outline of the camera motion, write the description of camera movement + rotation. The sentence should be short and factual. Use natural language. When describing, consider the length of the frames and do not mention frame indices.

# Examples Outline: Total frames 119. [Camera motion] Between frames 0 and 20: move_right - yaw_left; Between frames 21 and 43: static - yaw_left; Between frames 44 and 86: move_right - yaw_left; Between frames 87 and 118: move_right-move_backward - yaw_left. Description: The camera continuously yawing left while moving right, remaining static, moving right again, and moving right and backward. # End of examples

###### [Structured motion tags]

Outline: Total frames 120. [Camera motion] Between frames 0 and 119: move backward

###### [Motion caption]

###### [Prompt to generate Directorial captions]

You act as a camera operator writing a technical script for camera motion descriptions. The images above are 16 frames sequentially extracted from a video, arranged in a 4x4 grid, from left to right and top to bottom. Combining the description of camera movement, describe how the camera interacts with objects in the video. The sentence should be short and factual. Use natural language. When describing, consider the length of the frames and do not mention frame indices. The description of the camera trajectory should conform to the description of the Movement while describing the specific objects in the video. Provide two descriptions, one detailed and the other concise.

The camera continuously moves backward throughout the sequence.

[Figure 56]

Output format: **Detailed**: XXX

**Concise**: XXX”

Movement: The camera continuously moves backward throughout the sequence.

###### [Directorial caption]

"Detailed": "The camera begins by focusing on a rooftop scene, capturing two figures looking out over a city skyline. As it moves backward, the perspective widens, revealing more of the surrounding buildings and details of the rooftop. The figures remain centered as they continue to gaze at the horizon. Gradually, the camera exposes the depth of the urban landscape, showcasing the interplay of various architectural structures. The backward motion allows for an enhanced view of the expansive skyline, emphasizing the scale of the scene.", "Concise": "The camera moves backward, revealing two figures on a rooftop while expanding the view of the city skyline and surrounding buildings."

- Figure R4. Caption Generation. We structure the motion tags by incorporating context, instructions, constraints, and examples, and then leverage GPT-4o to generate Motion captions that describe the camera motion alone. Next, we extract 16 evenly spaced frames from the shots to create a 4 × 4 grid, prompting GPT-4o to consider both the previous caption and the image sequence. This enables GPT-4o to generate Directorial captions that describe the camera movement, the interaction between the camera and scene, and the directorial intent.

##### C.3. Additional Ablation Studies

for the small size, the latent dimension is L = 512, with the backbone OPT Transformer consisting of 8 layers and 8 attention heads per layer. For the base size, the latent dimension is L = 1024, with 12 layers and 12 attention heads. For the large size, the latent dimension is L = 1536, with 16 layers and 24 attention heads.

We conduct ablation experiments on several hyperparameters, as shown in Tab. S1, including the number of discrete bins, trajectory length, and model size. These parameters correspond to the discrete bin size B, the trajectory length, and the model size (as detailed in Sec. 5.1). Specifically,

The results indicate that optimal performance is achieved

Text-Trajectory Alignment Trajectory Quality F1-Score↑ CLaTr-Score↑ Coverage↑ CLaTr-FID↓

Ablation

64 0.394 33.594 0.751 49.854 128 0.409 35.824 0.851 24.748 256 0.400 36.179 0.872 22.714 512 0.391 35.201 0.882 23.633

Discrete bins

1024 0.393 34.277 0.884 24.979

15 0.398 34.576 0.863 22.238 30 0.400 36.179 0.872 22.714 60 0.393 34.523 0.864 26.307

Traj length

small 0.389 32.868 0.880 25.604 base 0.400 36.179 0.872 22.714 large 0.398 33.843 0.888 20.474

Model size

Table S1. Ablation Study on Hyperparameters. We conduct ablation experiments on several hyperparameters, including the number of discrete bins, trajectory length, and model size. These parameters correspond to the discrete bin size B, the trajectory length N, and the model size (as detailed in Sec. 5.1). The results show that the optimal performance is achieved when the number of discrete bins is set to 256, the trajectory length to 30, and the model size to base.

when the number of discrete bins is set to 256, the trajectory length to 30, and the model size to base. Notably, when the model size is set to large, although the performance in Text-texttt Alignment decreases, the Trajectory Quality improves. We speculate that this may be due to the larger model’s tendency to overfit, learning better trajectory quality while failing to follow the text instructions effectively.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, Daniel Dworakowski, Jiaojiao Fan, Michele Fenzi, Francesco Ferroni, Sanja Fidler, Dieter Fox, Songwei Ge, Yunhao Ge, Jinwei Gu, Siddharth Gururani, Ethan He, Jiahui Huang, Jacob Samuel Huffman, Pooya Jannaty, Jingyi Jin, Seung Wook Kim, Gergely Kl´ar, Grace Lam, Shiyi Lan, Laura Leal-Taix´e, Anqi Li, Zhaoshuo Li, Chen-Hsuan Lin, Tsung-Yi Lin, Huan Ling, Ming-Yu Liu, Xian Liu, Alice Luo, Qianli Ma, Hanzi Mao, Kaichun Mo, Arsalan Mousavian, Seungjun Nah, Sriharsha Niverty, David Page, Despoina Paschalidou, Zeeshan Patel, Lindsey Pavao, Morteza Ramezanali, Fitsum Reda, Xiaowei Ren, Vasanth Rao Naik Sabavat, Ed Schmerling, Stella Shi, Bartosz Stefaniak, Shitao Tang, Lyne Tchapmi, Przemek Tredak, WeiCheng Tseng, Jibin Varghese, Hao Wang, Haoxiang Wang, Heng Wang, Ting-Chun Wang, Fangyin Wei, Xinyue Wei, Jay Zhangjie Wu, Jiashu Xu, Wei Yang, Lin Yen-Chen, Xiaohui Zeng, Yu Zeng, Jing Zhang, Qinsheng Zhang, Yuxuan Zhang, Qingqing Zhao, and Artur Z´olkowski. Cosmos world foundation model platform for physical AI. CoRR, abs/2501.03575, 2025. 2
- [2] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. CoRR, abs/2412.03572, 2024. 3, 5
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel

- Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. CoRR, abs/2311.15127, 2023. 2
- [4] James F. Blinn. Where am i? what am I looking at? [cinematography]. IEEE Computer Graphics and Applications, 8

(4):76–81, 1988. 2

- [5] Rogerio Bonatti, Wenshan Wang, Cherie Ho, Aayush Ahuja, Mirko Gschwindt, Efe Camci, Erdal Kayacan, Sanjiban Choudhury, and Sebastian A. Scherer. Autonomous aerial cinematography in unstructured environments with learned artistic decision-making. J. Field Robotics, 37(4):606–641,

2020. 2

- [6] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. CoRR, abs/2411.00769, 2024. 9
- [7] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, and Chi Zhang. Meshanything: Artistcreated mesh generation with autoregressive transformers. CoRR, abs/2406.10163, 2024. 3, 5, 6
- [8] Robin Courant, Nicolas Dufour, Xi Wang, Marc Christie, and Vicky Kalogeiton. E.T. the exceptional trajectories: Text-to-camera-trajectory generation with character awareness. In ECCV (4), pages 464–480. Springer, 2024. 2, 3, 4, 6, 7
- [9] Steven Mark Drucker, Tinsley A. Galyean, and David Zeltzer. CINEMA: A system for procedural camera movements. In SI3D, pages 67–70. ACM, 1992. 2
- [10] Joseph L. Fleiss. Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5):378–382, 1971. 4
- [11] Quentin Galvane, Marc Christie, Christophe Lino, and R´emi Ronfard. Camera-on-rails: automated computation of constrained camera paths. In MIG, pages 151–157. ACM, 2015. 2

- [12] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. CoRR, abs/2412.04431, 2024. 3
- [13] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 1, 2
- [14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NIPS, pages 6626–6637, 2017. 7
- [15] Chen Hou, Guoqiang Wei, Yan Zeng, and Zhibo Chen. Training-free camera control for video generation. CoRR, abs/2406.10126, 2024. 1, 2
- [16] Yunzhong Hou, Liang Zheng, and Philip Torr. Learning camera movement control from real-world drone videos. arXiv preprint, 2024. 3
- [17] Biaozhang Huang, Xinde Li, Chuanfei Hu, and Heqing Li. Stochastic human motion prediction using a quantized conditional diffusion model. Knowl. Based Syst., 309:112823,

2025. 5

- [18] Chong Huang, Chuan-En Lin, Zhenyu Yang, Yan Kong, Peng Chen, Xin Yang, and Kwang-Ting Cheng. Learning to film from professional human motion videos. In CVPR, pages 4244–4253. Computer Vision Foundation / IEEE, 2019. 2
- [19] Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander Madry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos, Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam, Amadou Crookes, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy, Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler, Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg, Brydon Eastman, Camillo Lugaresi, Carroll L. Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson, Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea Voss, Chen Ding, Cheng Lu, Chong Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei, Constantin Koumouzelis, and Dane Sherburn. Gpt-4o system card. CoRR, abs/2410.21276, 2024. 3, 9, 11, 12
- [20] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave,

- Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. If you use this software, please cite it as below. 6, 7
- [21] Hongda Jiang, Bin Wang, Xi Wang, Marc Christie, and Baoquan Chen. Example-driven virtual cinematography by learning camera behaviors. ACM Trans. Graph., 39(4):45,

2020. 2

- [22] Hongda Jiang, Xi Wang, Marc Christie, Libin Liu, and Baoquan Chen. Cinematographic camera diffusion model. Comput. Graph. Forum, 43(2):i–iii, 2024. 2, 3, 5, 6, 7
- [23] Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. 1960. 10
- [24] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, Krishna Somandepalli, Hassan Akbari, Yair Alon, Yong Cheng, Joshua V. Dillon, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, Mikhail Sirotenko, Kihyuk Sohn, Xuan Yang, Hartwig Adam, MingHsuan Yang, Irfan Essa, Huisheng Wang, David A. Ross, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In ICML. OpenReview.net, 2024. 3
- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Daquan Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models. CoRR, abs/2412.03603,

2024. 2

- [26] Xinyang Li, Zhangyu Lai, Linning Xu, Yansong Qu, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Director3d: Real-world camera trajectory and 3d scene generation from text. In NeurIPS, 2024. 2, 5, 6, 7
- [27] Jing Liang, Amirreza Payandeh, Daeun Song, Xuesu Xiao, and Dinesh Manocha. DTG : Diffusion-based trajectory generation for mapless global navigation. In IROS, pages 5340–

5347. IEEE, 2024. 5

- [28] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. DL3DV-10K: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, pages 22160–22169. IEEE, 2024. 2, 3, 4
- [29] Christophe Lino and Marc Christie. Intuitive and efficient camera control with the toric space. ACM Trans. Graph., 34

(4):82:1–82:12, 2015. 2

- [30] Xinyi Liu, Tianyi Zhang, Matthew Johnson-Roberson, and Weiming Zhi. Splatraj: Camera trajectory generation with

- semantic gaussian splatting. CoRR, abs/2410.06014, 2024. 2
- [31] Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B. Lindell. Sg-i2v: Selfguided trajectory control in image-to-video generation. arXiv preprint arXiv:2411.04989, 2024. 2
- [32] Alec Radford, Jong Wook Kim, Chris Hallacy, A. Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 6, 7
- [33] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, pages 8821–

8831. PMLR, 2021. 3

- [34] Anyi Rao, Jiaze Wang, Linning Xu, Xuekun Jiang, Qingqiu Huang, Bolei Zhou, and Dahua Lin. A unified framework for shot type classification based on subject centric lens. In ECCV (11), pages 17–34. Springer, 2020. 3, 9
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10674–

10685. IEEE, 2022. 5, 6

- [36] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. 6
- [37] Ken Shoemake. Animating rotation with quaternion curves. In Proceedings of the 12th annual conference on Computer graphics and interactive techniques, pages 245–254, 1985. 10
- [38] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoder-only transformers. In CVPR, pages 19615–

19625. IEEE, 2024. 3, 5

- [39] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 2
- [40] Jiaxiang Tang, Zhaoshuo Li, Zekun Hao, Xian Liu, Gang Zeng, Ming-Yu Liu, and Qinsheng Zhang. Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. CoRR, abs/2409.18114, 2024. 3, 5, 6
- [41] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In NeurIPS, 2024. 3
- [42] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. CoRR, abs/2410.02757, 2024. 3
- [43] Tong Wu, Jiaqi Wang, Xingang Pan, Xudong Xu, Christian Theobalt, Ziwei Liu, and Dahua Lin. Voxurf: Voxel-based

- efficient and accurate neural surface reconstruction. In ICLR. OpenReview.net, 2023. 10
- [44] Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using VQ-VAE and transformers. CoRR, abs/2104.10157, 2021. 3
- [45] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation. Trans. Mach. Learn. Res., 2022,

2022. 3

- [46] Mark YU, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models, 2025. 12
- [47] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A largescale dataset of multi-view images. In CVPR, pages 9150–

9161. IEEE, 2023. 2, 3, 8

- [48] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. CoRR, abs/2410.03825,

2024. 2, 3, 8, 10

- [49] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: open pre-trained transformer language models. CoRR, abs/2205.01068, 2022. 6
- [50] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all, 2024. 2
- [51] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: learning view synthesis using multiplane images. ACM Trans. Graph., 37(4): 65, 2018. 2, 3, 8

