## ReDirector: Creating Any-Length Video Retakes with Rotary Camera Encoding

#### Byeongjun Park1 Byung-Hoon Kim1,2 Hyungjin Chung1† Jong Chul Ye3† 1 EverEx 2 Yonsei University 3 KAIST

| | |
|---|---|
| | |

|[Figure 1]|[Figure 2]|[Figure 3]|[Figure 4]|[Figure 5]|[Figure 6]|
|---|---|---|---|---|---|
|[Figure 7]|[Figure 8]|[Figure 9]|[Figure 10]|[Figure 11]|[Figure 12]|
|[Figure 13]|[Figure 14]|[Figure 15]|[Figure 16]|[Figure 17]|[Figure 18]|
|[Figure 19]|[Figure 20]|[Figure 21]|[Figure 22]|[Figure 23]|[Figure 24]|
|[Figure 25]|[Figure 26]|[Figure 27]|[Figure 28]|[Figure 29]|[Figure 30]|
|[Figure 31]|[Figure 32]|[Figure 33]|[Figure 34]|[Figure 35]|[Figure 36]|

- ① Input Video
- ② Target Retake

- ① Input Video
- ② Target Retake

# arXiv:2511.19827v1[cs.CV]25Nov2025

①

②

②

57

49

①

- ① Input Video
- ② Target Retake

- ① Input Video
- ② Target Retake

VideoLength

###### VideoLength

①

②

②

81

81 ①

- ① Input Video
- ② Target Retake

- ① Input Video
- ② Target Retake

① ②

###### 261

241

②

①

Figure 1. ReDirector for video retake generation. Given any-length input videos (①), ReDirector generates realistic retakes (②) along the target camera trajectories even with dynamic camera motion in the input video. ReDirector is capable of accurately localizing dynamic objects while preserving static backgrounds throughout the sequence, leading to multi-view consistent retakes spanning hundreds of frames.

### Abstract

### 1. Introduction

Camera-controlled video generation has attracted broad interest because it enables precise camera control over scene progression and visual dynamics, such as in filmmaking and virtual production [41]. In this context, video retake generation has made significant progress by using video-to-video generative models to redirect the camera trajectories of existing videos, enabling video captures at physically impractical viewpoints and stabilizing shaky footage, thereby reducing production costs and providing immersion [4, 40].

We present ReDirector, a novel camera-controlled video retake generation method for dynamically captured variablelength videos. In particular, we rectify a common misuse of RoPE in previous works by aligning the spatiotemporal positions of the input video and the target retake. Moreover, we introduce Rotary Camera Encoding (RoCE), a cameraconditioned RoPE phase shift that captures and integrates multi-view relationships within and across the input and target videos. By integrating camera conditions into RoPE, our method generalizes to out-of-distribution camera trajectories and video lengths, yielding improved dynamic object localization and static background preservation. Extensive experiments further demonstrate significant improvements in camera controllability, geometric consistency, and video quality across various trajectories and lengths.

Recent warping-based frameworks [7, 25, 39, 66, 72, 73] achieve high-quality video retake generation by leveraging video generative models [29, 61, 68] with two conditions, an input video and target camera trajectories. These frameworks resort to explicit geometric transformations (i.e., perframe warping) using video depth estimation models [8, 22] and point tracking models [28, 65], after which the video

†Corresponding authors

Project page: https://byeongjun-park.github.io/ReDirector/

generative model refines and inpaints warped frames. However, the external geometry estimators frequently degrade under dynamic camera motion and complex structures, allowing warping artifacts to propagate directly into the video generative model, where they are inadequately addressed.

To address these limitations, another line of works [4, 60] extend camera-controlled text-/image-to-video generation methods [2, 14, 18, 62] that utilize camera parameters to implicitly synthesize video frames without a warping process. In practice, the representations of the input video and the target camera trajectory are combined with those of the target video via concatenation or addition along channel or token axes, fine-tuning video generative models to internalize the multi-view geometry between input and target videos. While such designs may be sufficient for general conditioning in video-to-video generation tasks [26], critical drawbacks remain. Namely, the existing methods, including the previous warping-based approaches, assume fixed-length inputs with minimal camera motion, and hence degrade quickly outside these regimes. A key open challenge, therefore, is to handle variable-length inputs with dynamic camera motion by seamlessly integrating the two control signals to encode multi-view relationships across both videos and along their camera trajectories.

In this work, we introduce ReDirector, a novel cameracontrolled video retake generation method for dynamically captured variable-length input videos, as shown in Fig. 1. We leverage Rotary Position Embedding (RoPE) [57], a relative positional encoding known to generalize across sequence lengths [6, 19, 61, 68]. Our key idea is to employ the same RoPE to both input and target videos to encode the length-agnostic relative positions, then inject camera conditions as physically grounded positional signals that encode multi-view geometry within and across the two videos. This design views shared RoPE indices as tightly aligned spatiotemporal positions, and the camera condition encourages discrimination between the input and target videos at the corresponding positions. More concretely, we present Rotary Camera Encoding (RoCE), which integrates camera pose parameters into RoPE through a cameraconditioned phase shift. RoCE is initialized with zero phase, disabling camera conditioning at the start, and gradually learns nonzero phases during fine-tuning, ensuring stable incorporation of the camera signal. Applied across multiple RoPE frequencies with head-specific phase shifts, RoCE enhances geometric consistency by strengthening attention at the same physical positions across views, enabling multiview-consistent retakes even over long temporal gaps.

In parallel, amid growing interest in camera conditions

- as positional encoding [5, 47], recent geometry-aware attention methods [30, 34, 42] use explicit geometric transforms directly into attention layers, which are trained from scratch on static scenes. Given their design, these methods are

ill-suited for fine-tuning video generative models. Instead, we replace explicit geometric transforms with our cameraconditioned phase shifts, which apply a SO(2) phase shift to values before attention weighting and the inverse phase shift after the value aggregation. This preserves geometry awareness in the attention layers, remains compatible with fine-tuning of video generative models, and improves the disentanglement of dynamic objects from static backgrounds, resulting in physically plausible video retakes.

Through extensive experiments on video examples from the DAVIS dataset [53] and trajectories from ReCamMaster [4], our method consistently outperforms previous methods in camera control, geometric consistency, and visual quality across diverse target trajectories and video lengths. Evaluations on the iPhone dataset [12] further validate the strong generalizability of our method to out-of-distribution camera trajectories, video lengths, and resolutions.

### 2. Related Works

##### 2.1. Video Retake Generation

Recent progress in diffusion transformers [17, 46, 48, 51] and video generative models [29, 59, 61, 68] enables the incoporation of camera control signals into text-to-video generation [14, 15] and image-to-video generation [2, 3, 18, 21, 43, 54–56, 62, 67]. These methods exploit camera poses either through explicit geometric transformation (i.e., perframe warping) or by embedding Pl¨ucker rays [52] to encourage the model to implicitly learn the multi-view geometry. Building on these advances, video retake generation, which repurposes the input video by redirecting its camera trajectory using video-to-video generative models, has been widely explored in both explicit and implicit forms.

Explicit geometric transformation. The dominant approach for video retake generation is warping-based methods. Most pipelines first estimate scene geometry with video depth models [8, 22] and backproject into 3D point clouds, then reproject each input frame to the target camera trajectories to synthesize a geometrically aligned proxy of the retake. A video generative model subsequently refines and inpaints these proxies using warped frames and the input video, either via fine-tuning [7, 25, 39, 66, 72, 73] or in a training-free manner [50, 69, 70]. While effective, these methods depend heavily on accurate video depths, and scale alignment between the estimated input geometry and the target camera trajectory typically relies on handcrafted heuristics. Moreover, under dynamic camera motion and complex scene structure, warping artifacts from external geometric estimators propagate directly into the generated retakes, substantially degrading quality. The per-frame warping process further undermines the disentanglement of dynamic objects from static backgrounds and prevents encoding of a spatio-temporally aligned scene representation.

Implicit geometric transformation. Recently, implicit approaches [4, 60] construct large-scale synthetic datasets of input video, target camera trajectory, and target retake triplets, using rendering pipelines such as Kubric [16] and Unreal Engine 5 [9]. Instead of explicit warping, these methods incorporate both control signals directly into the video generative models by conditioning on camera extrinsic parameters and VAE-encoded input latent, relying on dataset supervision. This encourages the models to internalize multi-view geometry and to inpaint occlusions without hand-crafted preprocessing. Nevertheless, performance remains sensitive to the training data and can degrade under out-of-distribution trajectories and their video lengths. We extend these implicit methods by seamlessly integrating the two control signals to exhibit robust generalization to outof-distribution trajectories and sequence lengths.

##### 2.2. Camera encoding in vision transformers

Camera parameters provide physically grounded positional information that token indices do not capture. Most prior works [13, 27, 58, 63, 64] therefore adopt pixel-aligned (token-level) encodings that attach per-pixel geometry to each token, such as spherical angles or Pl¨ucker rays. By concatenating these camera encodings with tokens, the model is conditioned on both intrinsics and extrinsics and can learn fine-grained, token-level multi-view relationships. However, this encoding requires specifying a reference frame and a global world coordinate system, which introduces frame dependence and can hinder generalization, even with the scene scale normalization [14, 31].

In parallel, frame-aligned (camera-level) encodings utilize the camera pose of each frame by leveraging the characteristics of SE(3) and SO(3), typically encoding relative transformations into the attention scores. These signals modulate attention to inject geometry awareness and tend to generalize well across variable sequence lengths; they do not require a consistent global frame and have been shown to improve the novel view synthesis performance [30, 42]. Nonetheless, camera-level encoding lacks per-pixel granularity, which undermines fine multi-view correspondence and visibility reasoning (e.g., occlusions and thin structures). Concurrently, PRoPE [34] combines both camera encodings and trains a multi-view transformer from scratch on static scenes. In contrast, we propose a novel camera encoding that integrates with RoPE as phase shifts and is tailored for fine-tuning video generative models on dynamic scenes.

### 3. Preliminary

Rotary Position Embedding (RoPE). Recent video diffusion models [29, 59, 61, 68] are trained to generate arbitrary-length video sequences and multiple resolutions, despite not being explicitly trained for these settings. This generalizability is mostly derived from Rotary Position Em-

bedding (RoPE) [57] used in their attention layers. After the VAE encoder and patchify layers, each attention layer takes latent tokens x ∈ RN×d, where N = fhw is the total number of tokens and f,h,w are the compressed resolution of frame, height, and width, respectively. Linear projections are then applied to obtain the query and the key vectors, q,k ∈ RN×d

head, where dhead is the dimension of a single head of the multi-head attention (MHA). RoPE treats the vectors as a set of complex vectors by grouping elements, i.e., the even-indexed and subsequent odd-indexed channels are the real and imaginary parts, respectively

q,k  → q¯,k¯ ∈ CN×(d

head/2). (1)

Complex rotation can then be applied with a unitary rotation matrix R, which can be effectively implemented via

q¯′ = q¯ ◦ R,k¯′ = k¯ ◦ R, (2) where R ∈ CN×(d

head/2) is the matrix that implements the rotation with Hadamard product ◦, where each element R(n,c) = eiθ

cn is defined following Euler’s formula with exponentially decaying multiple frequency θc as:

c−1 dhead/2

θc = 10000−

,where c ∈ {1,2,...,dhead/2}. (3)

For (n,m)-th query and key vectors, this modulates each element in the attention matrix A′ ∈ RN×Nas follows:

A′(n,m) = Re[q¯′nk¯′∗m] = Re[q¯n(k¯∗m ◦ eiθ

c(n−m))], (4)

where Re[·] extracts the real part and ∗ denotes the complex conjugates. Therefore, relative positions are encoded as integer multiples of per-channel phase offsets determined by the token indices, and multiple frequency allows the attention matrix to cover a wide range of relative positions.

3D RoPE. In video diffusion models, spatiotemporal relative positions are commonly encoded using 3D RoPE, which consists of three complex rotation matrices for the frame, height, and width axes as follows:

R¯ f = Rf ⊗ 1h ⊗ 1w ∈ Cf×h×w×(d

- head/6), (5)

R¯ h = 1f ⊗ Rh ⊗ 1w ∈ Cf×h×w×(d

- head/6), (6)

R¯ w = 1f ⊗ 1h ⊗ Rw ∈ Cf×h×w×(d

- head/6), (7)

where ⊗ is Kronecker product and 1l ∈ Cl×1 denotes the all-ones matrix and Rl ∈ Cl×(d

head/6) is the complex rotation matrix for each axis l ∈ {f,h,w}. Then, we reshape each R¯ l to R˜ l ∈ CN×(d

head/2) and apply channel-wise concatenation to create 3D RoPE R as follows:

R = R˜ f ⊕ R˜ h ⊕ R˜ w ∈ CN×(d

head/2). (8)

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

Figure 2. Overview of ReDirector. (a) ReDirector is fine-tuned on Wan-I2V-CamCtrl [61], which incorporates camera control signals into image-to-video generation. Our goal is to reconstruct a video retake V′t conditioned on target camera trajectories Pt, input video Vs, and its poses Ps. (b) Following ReCamMaster [4], we train only self-attention layers while keeping the remaining modules frozen. (c) We insert RoCE into self-attention layers, whose outputs are used as camera-conditioned RoPE phase shifts. First, ϕqv is applied to queries and keys, providing physically grounded rotary position encoding. Second, ϕvo modulates the value path by applying −ϕvo before attention weighting and +ϕvo after value aggregation, enabling geometry-aware attention. For clarity, text prompts in cross-attention are omitted.

### 4. Methods

In this section, we present ReDirector, a camera-controlled video retake generation method for variable-length inputs captured with dynamic camera motion. In Sec. 4.1, we explain how we rectify common misuse of RoPE and camera encodings in prior work, enabling arbitrary-length input videos to be retaken with precise camera control. Then, in Sec. 4.2, we introduce RoCE, a physically grounded positional encoding that integrates into RoPE, further improving geometric consistency and achieving better disentanglement of dynamic objects from static backgrounds.

##### 4.1. Video-to-Video Generative Model

Architectural design. Given an input video Vs and target camera trajectories Pt, our objective is to generate a video retake Vt that follows the target trajectories. To this end, we start from the pretrained camera-controlled image-tovideo generative model [61] and fine-tune it into a video-tovideo generative model. We specifically revisit and correct common misuse in prior work regarding RoPE and camera encodings. Existing methods make only limited use of positional encodings for the input video, typically applying absolute positional encodings or using only specific axes in 3D RoPE, which forces them to assume fixed-length inputs. Moreover, unlike mainstream camera-controlled text/image-to-video generation methods that leverage Pl¨ucker rays as implicit representations, these methods predominantly rely on explicit geometric transformations, and even implicit variants [4, 60] often encode only target camera extrinsic parameters, resulting in coarse camera conditioning and limited token-level multi-view relationships.

In contrast, we apply a shared 3D RoPE to both the input and target videos, aligning their spatiotemporal positions and preserving length-agnostic positional encodings for the input sequence. This significantly improves generalization to longer video sequences, even when such lengths are absent from the training data. We further adopt Pl¨ucker rays as camera pose representations to employ token-level camera encodings. For the input video geometry, we use its camera trajectories Ps obtained from the training dataset, or estimate them at inference using ViPE [23]. By jointly conditioning on the input and target trajectories along with their video frames, the generative model is fine-tuned to represent detailed multi-view geometry under variable-length input videos. This overall setup is illustrated in Fig. 2-(a).

Training loss. We adopt the rectified flow setting [10, 38] and conditional flow matching loss [37] to train our model uθ, following ReCamMaster [4] to update weights only in the self-attention layers while keeping all other components frozen, as shown in Fig. 2-(b). We define a noise distribution p1 ∈ N(0,I) and a mapping between the data distribution p0 (i.e., target video) and the noise distribution as an ordinary differential equation (ODE):

dzt = uθ(zt,t)dt, (9) zt = tz1 + (1 − t)z0, (10)

where z0 ∼ p0 and z1 ∼ p1, and t ∈ [0,1]. Then, the conditional flow matching loss is defined as:

0,p1 ∥(z1 − z0) − uθ(zt,t)∥2 . (11)

LCFM = Et,p

After the training, video retakes are generated by solving the ODE from t = 1 to t = 0 at the inference stage.

Training strategy. To further enhance the performance, we adopt several training strategies. First, we include identity-retake pairs where the input and target videos share the same camera trajectories, {Vs,Ps} = {Vt,Pt}. This encourages the model to learn tighter alignment between tokens that correspond to the same 3D RoPE and RoCE. In addition, we augment the training dataset with timereversed versions of the videos, exposing the model to a wider variety of camera trajectories and enabling retake generation from other viewpoints, even at the first frame.

Token index

|[Figure 168]| |
|---|---|
| | |
| | |
| | |
| | |

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

LastframeFirstframe

[Figure 173]

[Figure 174]

[Figure 175]

Figure 3. Attention of RoCE. We visualize the attention from the colored dot in the leftmost column to the first frame (left), and across five uniformly sampled frames (right). Within each frame, attention varies with pixel coordinates, whereas differences in relative pose have a more pronounced impact on the attention scores.

##### 4.2. Rotary Camera Encoding

head/2) be the RoPE rotation matrix encoding the spatiotemporal location of the video for the target and the source, respectively, with Rt = Rs. Further let c = [ct,cs] ∈ R2N×d be the Pl¨ucker ray tokens encoding the camera parameters of the target and the source. Our goal is to further encode the camera information through relative positional embeddings to the spatial part, a method we call RoCE, as shown in Fig. 2-(c). Specifically,

Let R = [Rt,Rs] ∈ C2N×(d

phase shifts. We inject geometric awareness directly into the value aggregation process by leveraging SO(2) characteristics of phase shifts. We apply R˜ −vo11 before the attention weighting, and apply R˜ vo after the value aggregation as:

 A′

 v¯ ◦ R˜ −vo1

 

  ◦ R˜ vo. (18)

o¯′ =

- at each DiTBlock, we have ϕqk = [0,MLPqk(c)] ∈ R2N×(d

v ¯′

- head/2), (12)

ϕvo = [0,MLPvo(c)] ∈ R2N×(d

- head/2), (13)

This enables an efficient implementation of a learnable geometry-aware attention similar to GTA [42] when finetuning video generative models, whereas previous methods require training from scratch on static scenes. Furthermore, the model better separates dynamic objects from static backgrounds, as tokens in static regions stay multiview consistent, while tokens on moving objects break this pattern. This yields more geometrically consistent retakes.

where 0 ∈ R2N×(d

head/6) is the all-zeros matrix to prevent it from affecting the temporal positions. Similar to RoPE, we construct the learnable camera rotation matrix by grouping and applying it to the query and the key vectors

###### R˜ qk = eiϕ

∈ C2N×(d

head/2), (14) R˜ vo = eiϕ

qk

### 5. Experimental Results

∈ C2N×(d

head/2), (15) and subsequently

vo

##### 5.1. Experimental Setup

Implementation details. We fine-tune ReDirector on the MultiCamVideo dataset [4] for 20K steps at a resolution of 480 × 832 with a learning rate of 0.0001 and a batch size of 8. All training videos contain 81 frames. Training takes about 90 hours on eight RTX Pro 6000 Blackwell GPUs.

###### q¯′ = q¯ ◦ R ◦ R˜ qk, k¯′ = k¯ ◦ R ◦ R˜ qk. (16)

For the (n,m)-th query and key vectors, the attention matrix now reads

A(n,m)′ = Re[q¯′n(k¯′∗m ◦ ei(θ

c(n−m)+ϕqk(n,c)−ϕqk(m,c))].

Baselines. We compare our method with state-of-the-art video retake generation approaches [4, 7, 60, 72]. GCD [60] and ReCamMaster [4] implicitly generate video retakes conditioned on camera extrinsic parameters, and mainly focus on retakes of 14 and 81 frames, respectively. TrajectoryCrafter [72] and CogNVS [7] are the explicit warpingbased methods, which primarily focus on 49-frame retakes.

(17) Notably, as shown in Fig. 3, we observe that the attention matrix formed purely by the RoPE phase shifts (i.e., Re[ei(ϕ

qk(n,c)−ϕqk(m,c)]) indicates that RoCE behaves similarly to RoPE within each frame, but is more sensitive to relative poses, substantially suppressing attention between distant viewpoints. This enables the model to align distant frames and consistently preserve background regions at corresponding positions, even when they are far apart in time.

Evaluation protocol. We construct an evaluation set using 50 videos from the DAVIS dataset [53] and 10 target camera trajectories from ReCamMaster [4], yielding 500 test cases. Note that video lengths range from a few dozen

Geometry-aware attention. We extend geometry-aware attention frameworks [30, 34, 42], which treat token-wise camera encoding and geometry-aware attention independently, by explicitly coupling these components via RoPE

1With a slight abuse of notation, R−1 refers to the matrix that applies the complex rotation in the negative direction.

Camera

②

①

②

①

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

InputVideoGCDReCamMaster Trajectory

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

Crafter OursCogNVS

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

Tilt Up Arc Right

- Figure 4. Qualitative results on the DAVIS dataset [53]. ReDirector generates realistic video retakes (②) from dynamically captured input video (①), achieving better camera control, dynamic object localization, and background preservation.

Visual quality↑ Geometric Consistency Camera Accuracy Subject Consistency

Method

Background Consistency

Aesthetic Quality

Imaging Quality

Temporal Flickering

Motion Smoothness

Dyn-MEt3R↑ MEt3R↓ TransErr↓ RotErr↓

GCD [60] 0.7117 0.8400 0.3998 0.4928 0.9526 0.9639 0.6898 0.4438 0.1062 22.853 ReCamMaster [4] 0.9008 0.9212 0.5064 0.6461 0.9673 0.9881 0.7857 0.3472 0.0292 2.347 TrajectoryCrafter [72] 0.8846 0.9174 0.5046 0.6071 0.9364 0.9727 0.7338 0.3272 0.0697 9.115 CogNVS [7] 0.8929 0.9055 0.2160 0.4300 0.9637 0.9721 0.6845 0.4036 0.0768 10.878

Ours (ReDirector) 0.9043 0.9171 0.5149 0.6668 0.9548 0.9867 0.8477 0.3073 0.0165 1.666

Table 1. Quantitative results on the DAVIS dataset [53]. ReDirector significantly improves geometric consistency and camera control. Some visual quality metrics are lower because ReDirector explores larger scene scales under the same camera motion, whereas methods with milder camera movement naturally yield higher scores on metrics that favor consistent backgrounds and smoother motion.

frames to about 100 frames, and the target trajectories include rotations and translations along multiple axes. We extract camera poses for the input videos using ViPE [23]. For evaluation, we use multiple metrics from VBench [24] to assess visual quality, Dyn-MEt3R [49] to measure geometric consistency of the generated retakes, and per-frame MEt3R [1] to quantify consistency with the input video. We also report TransErr and RotErr [14, 15, 18, 74], which measure errors in relative translation and rotation for every frame pair by estimating input camera poses using ViPE.

##### 5.2. Main Results

Any-length video retake generation. Figure 4 shows the qualitative results on the DAVIS dataset [53]. We observe that previous methods degrade when the input video length deviates from their assumed setting. Also, explicit approaches [7, 72] fail to faithfully preserve the input content, even though they condition on warped frames, while implicit methods [4, 60] often produce blurry objects and sometimes cause dynamic objects to disappear. As the se-

- ①
- ②

①

Camera

②

②

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

InputVideoRoCE+GTARoCEAddition

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

Translate Up Zoom In Pan Right

- Figure 5. Qualitative ablation of conditioning type on DAVIS [53]. Compared to simple additive conditioning, RoCE enhances geometric consistency, while the additional geometry-aware attention further boosts retake quality and achieves accurate dynamic object localization.

Visual quality Geometric Consistency Camera Accuracy Aesthetic Quality↑ Imaging Quality↑ Dyn-MEt3R↑ MEt3R↓ TransErr↓ RotErr↓

Method Condition Type

ReCamMaster [4] Addition 0.5064 0.6461 0.7857 0.3472 0.0292 2.347 Ours (ReDirector) Addition 0.5121 0.6643 0.8378 0.3159 0.0202 1.975 Ours (ReDirector) RoCE 0.5139 0.6656 0.8341 0.3164 0.0193 1.897 Ours (ReDirector) RoCE + GTA [42] 0.5149 0.6668 0.8477 0.3073 0.0165 1.666

- Table 2. Quantitative ablations of conditioning type on DAVIS [53]. All components consistently improve the overall performance.

quence progresses and the visibility of previously seen views decreases, especially under dynamic camera motion, they increasingly fail to localize dynamic objects and maintain consistent backgrounds. In contrast, ReDirector produces high-quality video retakes that remain stable even under dynamic camera motion, accurately localizes dynamic objects over time, and consistently preserves static backgrounds across distant viewpoints. It better preserves fine details, avoids warping artifacts, and maintains a coherent scene layout that closely follows the target camera trajectories. Table 1 further demonstrates the superiority of our method, with particularly large gains in geometric consistency and camera controllability. We attribute these improvements to the seamless integration of camera encoding with 3D RoPE and the incorporation of geometry-aware attention, which together enable more reliable multi-view reasoning and physically accurate video retakes.

Ablation study. We conduct ablation studies to disentangle the contributions of each component in ReDirector. First, we correct the misuse of RoPE and camera encodings by introducing a shared 3D RoPE for input and target

videos and employing token-wise encodings. Next, we replace simple additive conditioning with RoCE, a cameraconditioned RoPE phase shift in spatial axes. Finally, we incorporate geometry-aware attention on top of RoCE.

Figure 5 and Table 2 summarize the results. Compared to ReCamMaster [4], correcting RoPE and camera encodings improves all metrics, showing that tight alignment between input and target videos and proper integration of camera signals into positional encodings are crucial. Introducing RoCE yields further gains in visual quality and camera accuracy, indicating that phase shift in the complex domain is more effective than na¨ıve addition for camera encoding. However, we observe a drop in geometric consistency metrics, suggesting that RoCE improves coarse alignment but still struggles to maintain fine-grained, multi-view consistent details. Finally, adding geometry-aware attention delivers the best performance across both geometric and visual metrics, confirming that the combination of aligned 3D RoPE, RoCE-based camera encoding, and geometry-aware attention is key to accurate camera control, better geometric consistency, and strong camera controllability.

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

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

GCD [60] ReCamMaster [4] TrajectoryCrafter [66] CogNVS [7] Ours GT

- Figure 6. Qualitative results on the iPhone dataset [12]. For each method, the left and right columns are retakes with 161 and 241 frames, respectively. All previous methods degrade severely on long videos, whereas our method remains robust and produces high-quality retakes.

81 frame 161 frame 241 frame PSNR↑ LPIPS↓ PSNR↑ LPIPS↓ PSNR↑ LPIPS↓

Method

GCD [60] 9.56 0.697 9.63 0.758 9.41 0.765 ReCamMaster [4] 10.69 0.678 10.03 0.762 10.37 0.772 TrajectoryCrafter [72] 8.79 0.733 9.58 0.744 10.35 0.741 CogNVS† [7] 10.56 0.720 10.63 0.741 10.81 0.720

Ours (ReDirector) 10.82 0.655 11.56 0.631 11.85 0.611

- Table 3. Quantitative results on iPhone dataset [12]. CogNVS† uses MoSca [32] and LiDAR depths to align scene scales.

Generalization to out-of-distribution conditions. Our method seamlessly integrates the camera control signal into RoPE phase shifts, and geometry-aware attention further enhances the multi-view consistency. To assess effectiveness, we evaluate the dynamic novel view synthesis performance [11, 32, 35, 45], confirming generalization to out-ofdistribution trajectories, video lengths, and resolutions.

Figure 6 shows that GCD [60] and ReCamMaster [4] often produce blurry, speckled artifacts and fail at novel view synthesis on long video sequences. TrajectoryCrafter [72] struggles to infer scene scale without LiDAR-based metric depth alignment, due to its reliance on hand-crafted heuristics in the warping process. CogNVS [7], despite using a strong 4D reconstruction method [32] and LiDARbased metric depth alignment, frequently fails to refine and can even degrade video quality. In contrast, our method remains robust on long sequences and produces realistic novel views without ground-truth metric depth or external geometry models. As longer input videos naturally cover larger portions of the scene, our approach exploits this to reconstruct additional regions (e.g., the black stand in the first example) and to recover scene scales that align more closely with the ground truth. Table 3 further shows that our method achieves the best novel view synthesis performance, with consistent improvements as input video length increases.

Geometric Consistency Camera Accuracy Dyn-MEt3R↑ MEt3R↓ TransErr↓ RotErr↓

Train Iter

20K 0.8477 0.3073 0.0165 1.666 50K 0.8491 0.2954 0.0154 1.521

Table 4. Quantitative ablations of training iterations on the DAVIS dataset [53]. We observe that additional training consistently improves the geometric consistency and camera accuracy.

Effect of additional training iterations. We investigate the effect of additional training iterations in Table 4. The results show steady gains in camera accuracy and geometric consistency, suggesting that further training progressively strengthens the model to internalize multi-view geometry purely from data, without explicit inductive biases.

### 6. Conclusion

In this paper, we have introduced ReDirector, a cameracontrolled video retake generation method for dynamically captured variable-length input videos. We first rectify a prevalent misuse of RoPE and camera encodings in prior work. Specifically, we apply the same RoPE for input and target videos, and use Pl¨ucker rays as camera pose representations, enabling length-agnostic conditioning and encoding token-level multi-view relationships. Then, we introduce RoCE, a camera-conditioned RoPE phase shift with geometry-aware attention, which improves geometric consistency, better dynamic object localization, and static background preservation, even when the input video contains dynamic camera motion. Extensive experiments verify the effectiveness of our method in camera control, geometric consistency, and visual quality, with strong generalization to out-of-distribution trajectories, lengths, and resolutions.

### References

- [1] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring multi-view consistency in generated images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6034–6044, 2025. 6
- [2] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22875–22889, 2025. 2
- [3] Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, David B. Lindell, and Sergey Tulyakov. Vd3d: Taming large video diffusion transformers for 3d camera control. Proc. ICLR, 2025. 2
- [4] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. Recammaster: Camera-controlled generative rendering from a single video. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 14834–14844, 2025. 1, 2, 3, 4, 5, 6, 7, 8, 12
- [5] Yunpeng Bai, Haoxiang Li, and Qixing Huang. Positional encoding field. arXiv preprint arXiv:2510.20385, 2025. 2
- [6] Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Veliˇckovi´c. Round and round we go! what makes rotary positional encodings useful? arXiv preprint arXiv:2410.06205, 2024. 2
- [7] Kaihua Chen, Tarasha Khurana, and Deva Ramanan. Reconstruct, inpaint, test-time finetune: Dynamic novel-view synthesis from monocular videos. In Advances in Neural Information Processing Systems (NeurIPS), 2025. 1, 2, 5, 6, 8
- [8] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22831–22840, 2025. 1, 2, 13
- [9] Epic Games. Unreal engine 5. https://www. unrealengine.com/en-US/unreal-engine-5,

2022. 3

- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 4

- [11] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5712–5721, 2021. 8
- [12] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. In NeurIPS, 2022. 2, 8
- [13] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan,

- Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 3
- [14] Hyojun Go, Byeongjun Park, Jiho Jang, Jin-Young Kim, Soonwoo Kwon, and Changick Kim. Splatflow: Multi-view rectified flow model for 3d gaussian splatting synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21524–21536,

2025. 2, 3, 6

- [15] Hyojun Go, Byeongjun Park, Hyelin Nam, Byung-Hoon Kim, Hyungjin Chung, and Changick Kim. Videorfsplat: Direct scene-level text-to-3d gaussian splatting generation with flexible pose and multi-view joint modeling. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 26706–26717, 2025. 2, 6
- [16] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761, 2022. 3
- [17] Seokil Ham, Sangmin Woo, Jin-Young Kim, Hyojun Go, Byeongjun Park, and Changick Kim. Diffusion model patching via mixture-of-prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 17023–17031,

2025. 2

- [18] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 6
- [19] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision, pages 289–305. Springer, 2024. 2
- [20] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 12
- [21] Chen Hou and Zhibo Chen. Training-free camera control for video generation. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [22] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2005–2015, 2025. 1, 2, 13
- [23] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, and Sanja Fidler. Vipe: Video pose engine for 3d geometric perception. In NVIDIA Research Whitepapers, 2025. 4, 6, 12
- [24] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6
- [25] Hyeonho Jeong, Suhyeon Lee, and Jong Chul Ye. Reangle-avideo: 4d video generation as video-to-video translation. In

- Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11164–11175, 2025. 1, 2
- [26] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17191–17202,

2025. 2

- [27] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024. 3
- [28] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European conference on computer vision, pages 18–35. Springer, 2024. 1, 13
- [29] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 2, 3
- [30] Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9503–9513, 2024. 2, 3, 5
- [31] Jon´aˇs Kulh´anek, Erik Derner, Torsten Sattler, and Robert Babuˇska. Viewformer: Nerf-free neural rendering from few images using transformers. In European Conference on Computer Vision, pages 198–216. Springer, 2022. 3
- [32] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. Mosca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. arXiv preprint arXiv:2405.17421, 2024. 8
- [33] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 12
- [34] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 2025. 2, 3, 5
- [35] Zhengqi Li, Qianqian Wang, Forrester Cole, Richard Tucker, and Noah Snavely. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4273– 4284, 2023. 8
- [36] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10486–10496, 2025. 12
- [37] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4

- [38] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [39] Dongyue Lu, Ao Liang, Tianxin Huang, Xiao Fu, Yuyang Zhao, Baorui Ma, Liang Pan, Wei Yin, Lingdong Kong, Wei Tsang Ooi, et al. See4d: Pose-free 4d generation via auto-regressive video inpainting. arXiv preprint arXiv:2510.26796, 2025. 1, 2
- [40] Yawen Luo, Jianhong Bai, Xiaoyu Shi, Menghan Xia, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Tianfan Xue. Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140,

2025. 1

- [41] Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869,

2025. 1

- [42] Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger. GTA: A geometry-aware attention mechanism for multi-view transformers. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 5, 7
- [43] Norman M¨uller, Katja Schwarz, Barbara R¨ossle, Lorenzo Porzi, Samuel Rota Bul`o, Matthias Nießner, and Peter Kontschieder. Multidiff: Consistent novel view synthesis from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10258–10268, 2024. 2
- [44] Hyelin Nam, Hyojun Go, Byeongjun Park, Byung-Hoon Kim, and Hyungjin Chung. Generating human motion videos using a cascaded text-to-video framework. arXiv preprint arXiv:2510.03909, 2025. 13
- [45] Byeongjun Park and Changick Kim. Point-dynrf: Pointbased dynamic radiance fields from a monocular video. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 3171–3181, 2024. 8
- [46] Byeongjun Park, Sangmin Woo, Hyojun Go, Jin-Young Kim, and Changick Kim. Denoising task routing for diffusion models. arXiv preprint arXiv:2310.07138, 2023. 2
- [47] Byeongjun Park, Hyojun Go, and Changick Kim. Bridging implicit and explicit geometric transformation for singleimage view synthesis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(9):6326–6340, 2024. 2
- [48] Byeongjun Park, Hyojun Go, Jin-Young Kim, Sangmin Woo, Seokil Ham, and Changick Kim. Switch diffusion transformer: Synergizing denoising tasks with sparse mixture-ofexperts. In European Conference on Computer Vision, pages 461–477. Springer, 2024. 2
- [49] Byeongjun Park, Hyojun Go, Hyelin Nam, Byung-Hoon Kim, Hyungjin Chung, and Changick Kim. Steerx: Creating any camera-free 3d and 4d scenes with geometric steering. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 27326–27337, 2025. 6
- [50] Jangho Park, Taesung Kwon, and Jong Chul Ye. Zero4d: Training-free 4d video generation from single video using off-the-shelf video diffusion. arXiv preprint arXiv:2503.22622, 2025. 2

- [51] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2

- [52] Julius Plucker. Xvii. on a new geometry of space. Philosophical Transactions of the Royal Society of London, (155): 725–791, 1865. 2
- [53] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbel´aez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 2, 5, 6, 7, 8, 12, 13
- [54] Stefan Popov, Amit Raj, Michael Krainin, Yuanzhen Li, William T. Freeman, and Michael Rubinstein. Camctrl3d: Single-image scene exploration with precise 3d camera control. In 2025 International Conference on 3D Vision (3DV), pages 649–658, 2025. 2
- [55] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6121– 6132, 2025.
- [56] Junyoung Seo, Kazumi Fukuda, Takashi Shibuya, Takuya Narihira, Naoki Murata, Shoukang Hu, Chieh-Hsin Lai, Seungryong Kim, and Yuki Mitsufuji. Genwarp: Single image to novel views with semantic-preserving generative warping. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2
- [57] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 2, 3

- [58] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024. 3
- [59] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 2, 3
- [60] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer, 2024. 2, 3, 4, 5, 6, 8
- [61] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 4
- [62] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [63] Sangmin Woo, Byeongjun Park, Hyojun Go, Jin-Young Kim, and Changick Kim. Harmonyview: Harmonizing consistency and diversity in one-image-to-3d. In Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10574–10584, 2024. 3
- [64] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26057–26068, 2025. 3
- [65] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20406–20417, 2024. 1, 13
- [66] Zeqi Xiao, Wenqi Ouyang, Yifan Zhou, Shuai Yang, Lei Yang, Jianlou Si, and Xingang Pan. Trajectory attention for fine-grained video motion control. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2, 8
- [67] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 2
- [68] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2, 3
- [69] Hidir Yesiltepe and Pinar Yanardag. Dynamic view synthesis as an inverse problem. arXiv preprint arXiv:2506.08004,

2025. 2

- [70] Meng YOU, Zhiyu Zhu, Hui LIU, and Junhui Hou. NVSsolver: Video diffusion model as zero-shot novel view synthesizer. In The Thirteenth International Conference on Learning Representations, 2025. 2
- [71] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141,

2025. 13

- [72] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 100–111, 2025. 1, 2, 5, 6, 8
- [73] David Junhao Zhang, Roni Paiss, Shiran Zada, Nikhil Karnad, David E Jacobs, Yael Pritch, Inbar Mosseri, Mike Zheng Shou, Neal Wadhwa, and Nataniel Ruiz. Recapture: Generative video camera controls for user-provided videos using masked video fine-tuning. arXiv preprint arXiv:2411.05003,

2024. 1, 2

- [74] Jason Y Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In International Conference on Learning Representations (ICLR), 2024. 6
- [75] Siyuan Zhou, Yilun Du, Yuncong Yang, Lei Han, Peihao Chen, Dit-Yan Yeung, and Chuang Gan. Learning 3d persistent embodied world models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems,

2025. 13

## ReDirector: Creating Any-Length Video Retakes with Rotary Camera Encoding Supplementary Material

Geometric Consistency Camera Accuracy Dyn-MEt3R↑ MEt3R↓ TransErr↓ RotErr↓

- ①

Zoom Out

- ②

Camera Pose

MegaSaM [36] 0.8357 0.3041 0.0210 1.942 ViPE [23] 0.8477 0.3073 0.0165 1.666

[Figure 269]

[Figure 270]

[Figure 271]

Table 5. Effect of input trajectories on the DAVIS dataset [53]. ViPE [23] and MegaSaM [36] approximately take 30 seconds and 3 minutes for a hundred-length video, respectively.

InputVideoOurs+MegaSaMOurs+ViPE

[Figure 272]

[Figure 273]

[Figure 274]

### A. Evaluation Details

For all experiments, we set the inference steps to 50 and the CFG scale [20] to 5. We use BLIP [33] for text captioning.

[Figure 275]

[Figure 276]

[Figure 277]

DAVIS dataset. We use the camera intrinsics estimated by ViPE [23] when converting the target camera trajectories into Pl¨ucker rays. Since the 10 test camera trajectories in ReCamMaster [4] are originally defined over 81 frames, we reparameterize them as per-frame updates. Specifically, for each trajectory, we first accumulate the original 81-frame updates into a single total rotation and translation. Then, given an input video with F frames, we divide this total camera motion by F − 1 to obtain constant per-frame updates. This per-frame parameterization allows us to apply the same camera patterns to videos of arbitrary length in our experimental setting. The resulting per-frame camera trajectories are detailed below:

- Figure 7. Qualitative ablation on input trajectories.

InputVideoOurs(20K)Ours(50K)

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

- Figure 8. Qualitative ablation on training iterations.

- • Pan right/left: The camera rotates by F20−1 degrees about the yaw axis per frame, with no translation.

- • Tilt up/down: The camera rotates by F10−1 degrees about the pitch axis per frame, with no translation.

- • Zoom in/out: The camera translates by F−2 1(m) along the z-axis per frame, with no rotation.

- • Translate up/down: For each frame, the camera tilts down/up by F14−1 degrees and translates F−1 1(m) along the y-axis (i.e., up/down), respectively; in both cases, it additionally moves F0.−121(m) outward (i.e., zoom out).

- • Arc left/right: For each frame, the camera pans right/left

### B. Additional Results

##### B.1. Ablative Results

Effect of input trajectories. We evaluate the impact of input camera trajectories on retake performance by replacing ViPE with MegaSaM [36]. ViPE predicts in metric scale, whereas MegaSaM operates in a scale-ambiguous space and provides more accurate camera poses with an optimization process. Because the target camera trajectories are also defined in metric scale, Table 5 shows that metrically aligned poses yield better performance than more accurate but scale-indefinite poses. Consistently, Figure 7 demonstrates that poses from ViPE produce a geometrically accurate video retake, whereas poses from MegaSaM cause the model to fail to properly localize the person.

by F30−1 degrees and translates F−2 1(m) along the x-axis (i.e., left/right), respectively; in both cases, it additionally

moves F0.−011(m) inward (i.e., zoom in).

iPhone dataset. We do not use any video pose estimation methods or other external models. We rely solely on the provided camera trajectories for each input video and the specified viewpoint for each test case. Even without external models and LiDAR depth, ReDirector achieves the best performance in dynamic novel view synthesis compared to previous video retake generation methods.

Effect of further training. As shown in Fig. 8, we qualitatively verify that additional training improves both geometric consistency and dynamic object localization.

①

Camera

②

②

①

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

InputVideoGCDReCamMaster Trajectory

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

[Figure 309]

[Figure 310]

Crafter OursCogNVS

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

Arc Left Translate Down

Figure 9. Additional qualitative results on the DAVIS dataset [53].

##### B.2. More Results and Applications

We provide additional qualitative comparisons in Fig. 9. In the first example, previous methods produce unrealistic retakes, and TrajectoryCrafter renders the person’s legs inconsistently with the input video. In the second example, we observe that previous methods either produce inconsistent water textures or fail to achieve precise camera control. In contrast, our method consistently preserves both dynamic objects and backgrounds, and faithfully follows the target camera trajectories even under textureless backgrounds.

We present additional video retakes generated by ReDirector in Fig. 11-(a). The results confirm that our method generates high-quality, geometrically accurate, and temporally coherent video retakes across various trajectories and lengths. Although the training data consists of synchronized videos whose first frames are fully overlapped, we additionally include time-reversed versions of these videos during training. This enables our model to generate retakes with non-overlapping first frames, as shown in Fig. 11-(b). Finally, since our method is well aligned with metrically scaled camera trajectories, we can stretch the target trajectories; Fig. 11-(c) shows video retakes when the camera trajectories are scaled by a factor of two.

[Figure 323]

[Figure 324]

InputVideoOurs

[Figure 325]

[Figure 326]

Figure 10. Failure case.

### C. Limitations and Discussions

As shown in Fig. 10, our method still struggles in challenging scenarios with extreme object and camera motion, particularly when large dynamic objects dominate the frame, making it difficult to reliably infer multi-view relationships. A promising future direction is to integrate our framework with next-frame video generation models [71, 75] in the context of world models, or to combine it with external geometry models [8, 22, 28, 44, 65] to further improve geometric robustness and fidelity.

|[Figure 327]|[Figure 328]|[Figure 329]|
|---|---|---|
|[Figure 330]|[Figure 331]|[Figure 332]|

|[Figure 333]|[Figure 334]|[Figure 335]|
|---|---|---|
|[Figure 336]|[Figure 337]|[Figure 338]|

InputTarget

TargetInput

| | |
|---|---|

Translate Down Arc Right

|[Figure 339]|[Figure 340]|[Figure 341]|
|---|---|---|
|[Figure 342]|[Figure 343]|[Figure 344]|

|[Figure 345]|[Figure 346]|[Figure 347]|
|---|---|---|
|[Figure 348]|[Figure 349]|[Figure 350]|

InputTarget

TargetInput

Zoom Out Translate Up

(a) Overlapped first frames

|[Figure 351]|[Figure 352]|[Figure 353]|
|---|---|---|
|[Figure 354]|[Figure 355]|[Figure 356]|

|[Figure 357]|[Figure 358]|[Figure 359]|
|---|---|---|
|[Figure 360]|[Figure 361]|[Figure 362]|

InputTarget

TargetInput

Frame-wise Translate Up Frame-wise Translate Down

|[Figure 363]|[Figure 364]|[Figure 365]|
|---|---|---|
|[Figure 366]|[Figure 367]|[Figure 368]|

|[Figure 369]|[Figure 370]|[Figure 371]|
|---|---|---|
|[Figure 372]|[Figure 373]|[Figure 374]|

InputTarget

TargetInput

Frame-wise Zoom In Frame-wise Zoom Out

(b) Non-overlapped first frames

|[Figure 375]|[Figure 376]|[Figure 377]|
|---|---|---|

|[Figure 378]|[Figure 379]|[Figure 380]|
|---|---|---|

Input

Input Target

|[Figure 381]|[Figure 382]|[Figure 383]|
|---|---|---|

|[Figure 384]|[Figure 385]|[Figure 386]|
|---|---|---|

CamSpeed=1

- CamSpeed=1

Target

- CamSpeed=2

Target

|[Figure 387]|[Figure 388]|[Figure 389]|
|---|---|---|

|[Figure 390]|[Figure 391]|[Figure 392]|
|---|---|---|

CamSpeed=2

Target

Arc Left Zoom In

(c) Varying camera speed

###### Figure 11. Additional qualitative results and applications of ReDirector.

