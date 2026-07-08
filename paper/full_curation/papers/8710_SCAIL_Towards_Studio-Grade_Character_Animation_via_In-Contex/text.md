## SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations

Wenhao Yan1∗† Sheng Ye1∗† Zhuoyi Yang1†‡ Jiayan Teng1† ZhenHui Dong1 Kairui Wen1 Xiaotao Gu2 Yong-Jin Liu1§ Jie Tang1§ 1Tsinghua University 2Z.ai

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

# arXiv:2512.05905v3[cs.CV]23Mar2026

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

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Figure 1. We propose SCAIL, a character animation framework that enables high-fidelity character animation under diverse and challenging conditions, including large motion variations, stylized characters, and multi-character interactions.

#### Abstract

establish a comprehensive benchmark for systematic evaluation. Experiments show that SCAIL achieves state-of-theart performance and advances character animation toward studio-grade controlling. Code and model are available at zai-org/SCAIL.

Achieving controllable character animation that meets studio-grade standards remains challenging despite recent progress. Existing approaches can transfer motion from a driving video to a reference image, but often fail to preserve structural fidelity and temporal consistency in wild scenarios involving complex motion and cross-identity animations. In this work, we present SCAIL (a framework toward Studio-grade Character Animation via In-context Learning), which is designed to address these challenges from two key innovations. First, we propose a novel 3D pose representation, providing a robust and flexible motion signal. Second, we introduce a full-context pose injection mechanism within a diffusion-transformer, enabling effective spatio-temporal reasoning over full motion sequences. To align with studio-grade requirements, we develop a curated data pipeline ensuring both diversity and quality, and

#### 1. Introduction

High-fidelity character animation has tremendous potential for film production. Conventional filmmaking pipelines rely on a complicated workflow—involving motion capture, rigging, rendering—that demands expensive hardware and significant expert labor. Recently, the emergence of video generation models [2, 13, 36, 47] introduces a new paradigm: given a reference image and a driving video, such models can directly synthesize animations that follow the target motion in driving video. Similar to real-world production pipelines, existing methods [4, 15, 30, 31, 38, 44, 49, 52] typically begin by extracting skeletal motion sequences from the driving video as a form of “motion capture”, and then inject this information into a video generation model to perform “rigging” and “rendering”. However, these methods often struggle with challenging scenarios, such as com-

*Equal contribution. †Work done during internship at Z.ai. ‡Project leader. §Corresponding author: jietang@tsinghua.edu.cn,

liuyongjin@tsinghua.edu.cn

plex motions (e.g., turning, rolling, flipping), multi-person interactions (e.g., dancing, hugging, fighting), and crossdomain animation where the reference and driving subjects differ significantly in appearance or body shape, exhibiting distorted appearance or implausible limb occlusions. In this work, we identify these challenges as the primary bottleneck to achieving studio-grade controllable animation. These limitations reveal that current skeletal pose representations and pose-driving generation methods fail to adequately capture 3D structure of driving motions, intercharacter spatial and occlusion relationships, and temporal correlations of motion sequences.

First, for motion representation, prior works [4, 15, 17, 31, 38, 49] typically rely on 2D keypoints (extracted from DWPose [46], ViTPose [43], etc.), which suffer from noisy predictions and cannot encode occlusion. Some works [37, 52] adopt SMPL [20] mesh controls, which offer strong 3D human priors but cause severe identity leakage. Second, regarding pose injection, a common approach in diffusion transformer (DiT) [23] model is to concatenate conditions channel-wise [4, 51]. However, we find that this method provides more local motion cues than capturing global motion dependencies.

To address the above limitations, we present SCAIL, a framework that revisits two core technical bottlenecks: (1) how to build a pose representation that effectively bridges driving and generated videos with unambiguous and accurate motion encoding, and (2) how to inject pose control in a way that enables the model to capture spatio-temporal motion structures. We propose a novel 3D pose representation that rasterizes bones as spatial cylinders into the pixel plane, such representation can be augmented and retargeted to enable seamless controlling across diverse characters and scenarios. To prevent model from following local motion cues as a shortcut, we propose a full-context pose injection mechanism within a DiT [23]-based architecture. This design allows the model to attend to the entire pose sequence when generating each frame, enabling it to reason about motion context across time, and better capture high-level motion semantics.

To obtain reliable video-pose pairs, we apply a segmentation-and-extraction approach to get 3D keypoints in human interactions. Building on that, a data curation pipeline ensuring both character diversity and motion complexity is established, with automatic filtering based on human presence and motion-based metrics and a final manual stage to select a finetuning set of superior quality. We also observe that current evaluations lack a comprehensive benchmark that adequately reflects production-level requirements. To address this gap, we propose Studio-Bench. It consists of two parts: the first evaluates motion adherence and structural integrity under complex single- and multiperson actions, while the second measures model general-

ization when the reference image and driving video differ in identity or domain. Our Studio-Bench covers challenging real-world cases and provides a realistic and rigorous measure of studio-level generation capability.

Our main contributions can be summarized as follows: (1) We propose an identity-agnostic 3D pose representation that serves as motion-driving signal suitable for complex motions and multi-human interactions. (2) We inject driving-pose controls via in-context reasoning to enable effective spatiotemporal motion modeling, yielding superior results in complex and multi-person scenarios. (3) We construct a pipeline for curating high-quality, diverse training data, and establish a comprehensive Studio-Bench for systematic evaluation. (4) Our SCAIL framework achieves state-of-the-art performance over existing baselines, and advances character image animation toward production-level readiness.

#### 2. Related Work

##### 2.1. Diffusion Models for Video Generation

Diffusion models [11, 28] effectively overcome the training instability and mode collapse of generative adversarial networks (GANs) [8], and emerge as the dominant paradigm in visual content generation. The success of the Stable Diffusion (SD) [24, 25] in image synthesis has naturally inspired the extension from image to video. Subsequently, the Diffusion Transformer (DiT) [23] architecture, combined with RoPE [29] for position encoding, offers superior modeling capability and scalability, becoming the leading backbone for high-quality video generation [18, 22, 36, 47]. More recently, diffusion-based video generation has advanced toward controllability, enabling control over camera viewpoints [9, 42], motion trajectories [6, 50], and scene structures [5, 19, 41]. In this work, we focus on character video generation with precise pose control.

##### 2.2. Controllable Character Image Animation

Character image animation aims to generate photorealistic and temporally coherent video, where the appearance remains consistent with a given reference image and the motion follows a driving video. AnimateAnyone [15] extracts 2D skeletons from the driving video as motion guidance, and designs Pose Guider and ReferenceNet modules to control motion and appearance. Subsequent works explore various improvements. Champ [52] integrates multiple motion signals, including depth maps, SMPL [20] normals, and 2D skeletons. Animate-X [30] introduces a skeleton augmentation strategy to enable animation of anthropomorphic characters with large body-ratio discrepancies. Moreover, MimicMotion [49] address hand and facial distortions via regional loss. DanceTogether [3] extends this line of research to multi-person animation. UniAnimate-DiT [38],

[Figure 28]

SMPL Mesh Representation

###### 2D Pose Representation

###### Ambigous and Noisy Pose

[Figure 29]

◮

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Train

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

◬

[Figure 40]

[Figure 41]

[Figure 42]

Train

Reshape Pool 𝐴𝑢𝑔 𝒫 = 𝒫

𝒫

[Figure 43]

Train

[Figure 44]

Mesh Renderer

Test

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

2D-3D Mismatch Noise

Identity Leakage

[Figure 51]

[Figure 52]

◬

Generation

[Figure 53]

Test

[Figure 54]

𝒫

Test

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

2D Retarget

[Figure 59]

[Figure 60]

𝑅𝑒𝑡 𝒫,𝒫 = 𝒫˜

[Figure 61]

[Figure 62]

[Figure 63]

◬

𝒫

𝑇

𝑇 + 1

◬ 2D Keypoints Estimator ◮ SMPL Estimator

Rasterizer

###### 3D-Consistent Pose Representation (Ours)

Identity-Agonostic Motion & Accurate Occulusion

3D-Consistent Adaptation

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Figure Rescale Camera Manipulation

[Figure 72]

[Figure 73]

◮

Train

𝒥

𝒥 = 𝐴𝑢𝑔(S,𝒥)

Train

Train S = 𝑓 𝐽 ,  ,𝐽 ,  ∣ m,𝑛 ∈ ℬ

𝑇

𝐴𝑢𝑔 Cam ∗  = Cam ∗ 

[Figure 74]

Cam Pool

[Figure 75]

…

Figure Pool

[Figure 76]

Camera Zoom In/Out Camera Move ↑ ↓← →

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Test

𝑇 + 1

𝑇

[Figure 82]

###### Sample S Principle Point ↑ ↓← → fromFigure Pool

[Figure 83]

[Figure 84]

[Figure 85]

◮

[Figure 86]

[Figure 87]

[Figure 88]

𝒥

Generation

[Figure 89]

[Figure 90]

…

S S

𝑇 + 1

[Figure 91]

[Figure 92]

𝑇

…

Cam Cam Cam

[Figure 93]

…

…

[Figure 94]

Or Generate S

Test

𝑅𝑒𝑡 𝒥,𝒫 ,Cam ∗  = Cam˜  ∗ 

[Figure 95]

[Figure 96]

◬

S S

…

𝒫

Optimize

𝑇 + 1

𝒥 = 𝒥 ℒproj = ∑ Π Cam˜  ∗ 𝐽 ,  −𝑃 ref

Test

- Figure 2. Overview of the proposed 3D-consistent pose. For scaling implementation, we take the clavicle or the pelvis as the central reference, applying scaling from proximal to distal along each limb in bones set B. Aug(·) denotes augmentation in training, Ret(·) denotes

retargeting in inference, and Pref = {Prefj | 1 ≤ j ≤ N} denotes N estimated 2D keypoints in the reference image. We further incorporate hand and face controls by overlaying 2D hand and face keypoints onto the rendered sequences, and align them with the projection of 3D joints during augmentation or retargeting. For better clarity, we omit the drawing process of 2D hand and face in the figure.

VACE [17] and Wan-Animate [4] replace U-Net with Transformer as the denoising backbone, significantly improving the generation quality. Despite progress, current models still struggle with complex scenarios and cross-pair animations, which hinders their deployment in real production.

works propose Diffusion Transformer (DiT) [23] architecture, which supports variable input resolution and sequence length. The DiT combines the generative power of diffusion models with the flexibility of Transformers [34], enabling better scalability and stronger modeling capacity.

##### 3.2. 3D-Consistent Pose Conditioning

#### 3. Method

3D-Consistent Pose Representation. Recent methods [4, 15, 17, 30, 38] that employ 2D keypoint-based skeletons as motion controls perform well on simple actions. However, they often struggle under complex studio-level production scenarios, leading to abnormal human kinematics. As there exists discrepancy between the reference image and the driving pose (e.g. body shape variations), previous 2D-based methods [30, 38] mitigate such gap through 2D skeleton scaling during training or heuristic retargeting at test time. However, such adaptation inherently suffers from deformation due to inconsistency with 3D motion dynamics, further amplifying estimation noise and inaccuracies.

##### 3.1. Preliminaries

Latent Diffusion Models. Latent diffusion models [1, 25] reduce the computational cost of pixel-space diffusion [11, 12] by operating in a compressed latent space. Given an input video x, a pretrained VAE encoder [36] E first maps it into a latent representation z0 = E(x). During the forward diffusion process, Gaussian noise is progressively added over T timesteps, formulated as:

q(zt|zt−1) = N zt; 1 − βt zt−1, βtI , (1)

We design our 3D pose representation to retain depth and occlusion, deliver accurate motion and remain identityagnostic. To achieve that, we employ NLFPose [26] to estimate 3D body keypoints J = {Ji,j}T,Ni=1,j=1, where ˜Ji,j denotes the predicted 3D joint j at frame i, and connect them according to the skeletal topology in 3D space. We discard SMPL mesh-like representations, which are inherently human-specific and offer limited controllability for

where βt denotes the noise schedule. The denoising model εθ(zt,t,c) learns to predict the added noise conditioned on optional input c (e.g., pose, text), and is optimized by:

t,ε∼N(0,I) ∥ε − εθ(zt,t,c)∥22 . (2)

L = Ez

Diffusion Transformer (DiT). Traditional diffusion models typically rely on U-Net as denoising backbones. Recent

###### Video Input Pose-shifted RoPE

###### Pose Control

###### Reference Images

[Figure 97]

[Figure 98]

[Figure 99]

𝐻

[Figure 100]

[Figure 101]

[Figure 102]

𝑇

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- (3,0,5) (3,0,6) (3,0,7) (3,0,8)
- (3,1,5) (3,1,6) (3,1,7) (3,1,8)
- (3,2,5) (3,2,6) (3,2,7) (3,2,8)

- (3,0,0) (3,0,1) (3,0,2) (3,0,3)
- (3,1,0) (3,1,1) (3,1,2) (3,1,3)
- (3,2,0) (3,2,1) (3,2,2) (3,2,3)

[Figure 107]

[Figure 108]

- (2,0,0) (2,0,1) (2,0,2) (2,0,3)
- (2,1,0) (2,1,1) (2,1,2) (2,1,3)
- (2,2,0) (2,2,1) (2,2,2) (2,2,3)

- (2,0,5) (2,0,6) (2,0,7) (2,0,8)
- (2,1,5) (2,1,6) (2,1,7) (2,1,8)
- (2,2,5) (2,2,6) (2,2,7) (2,2,8)

[Figure 109]

[Figure 110]

| | |
|---|---|
| | |

- (1,0,5) (1,0,6) (1,0,7) (1,0,8)
- (1,1,5) (1,1,6) (1,1,7) (1,1,8)
- (1,2,5) (1,2,6) (1,2,7) (1,2,8)

- (1,0,0) (1,0,1) (1,0,2) (1,0,3)
- (1,1,0) (1,1,1) (1,1,2) (1,1,3)
- (1,2,0) (1,2,1) (1,2,2) (1,2,3)

[Figure 111]

3D VAE Encoder

- (0,0,0) (0,0,1) (0,0,2) (0,0,3)
- (0,1,0) (0,1,1) (0,1,2) (0,1,3)
- (0,2,0) (0,2,1) (0,2,2) (0,2,3)

[Figure 112]

Add Noise

𝐻

I2V Mask concatenation

…

…

…

𝑊

[Figure 113]

[Figure 114]

Video Patchify Pose Patchify

𝑊

𝑊

𝑊

[Figure 115]

###### … Pose-shifted RoPE

Optional Text Input

Video Output

…

…

…

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Trainable Parameters

[Figure 120]

…

3D VAE Decoder

[Figure 121]

[Figure 122]

Dit Transformer Block

×𝑁

[Figure 123]

Frozen Parameters

- Figure 3. Overview of SCAIL’s model architecture. SCAIL builds upon I2V model and incorporate pose control as an explicit context for the model to learn spatial-temporal motion. To accommodate to the training setting where reference image and video input are sampled from different parts of the video, we modify the I2V model’s input structure by concatenating the reference image at the beginning of the sequence and initiating generation from T = 1, using the original I2V pattern to inject the reference CLIP feature. To help the model better distinguish the conditional tokens and the noisy video sequence, we leverage the original mask mechanisim of Wan-I2V model architecture, applying an all-one mask for the reference image and the driving sequence, and an all-zero mask for the noisy video sequence.

jective to refine the camera projection matrix Cam˜ 3×3:

augmentation or retargeting, and render 3D skeletons as cylindrical segments to provide driving signals. We distinguish person identities by directly rendering skeletons with different hues instead of relying on hardcoded network designs [3].

N

2 2

1 N

Π Cam ˜ 3×3˜J0,j − Prefj

, (3)

Lproj =

j=1

where Π(·) is the perspective projection, N is joint number, and Prefj is the estimated 2D joints of the reference frame.

As the alignment stage in inference may introduces approximation error, we simulates such error in training by directly applying such aligning method using the reference image and the first frame of the driving sequence (the reference image and driving sequence are sampled from different parts of the video) and exerting modest disturbance on camera parameters Cam3×3 to enhance the model’s robustness towards camera variance. Such design strikes a balance between the controllability of location and robust motion transfer.

To address the disparity between the reference image and the driving pose, we introduce a two-stage optimization strategy leveraging the 3D prior of our representation. Unlike previous methods [30] that applies 2D scaling in training augmentation and non-robust skeleton scaling rules in inference retargeting, we design training pose to be fully identity agnostic with accurate bone angles, and in inference we avoid altering the original estimated skeleton and only optimize the camera instead. To achieve identity decoupling during training, we synthesize a set of scale parameters S, and randomly sample Si ∈ S during training, each Si denoting a set of scaling parameter of each bones, simulating binding motion to characters with diverse body shapes. Since the underlying estimated skeleton topology and relative bone directions remain consistent, the scaled skeletons is temporally coherent and still faithfully represent the same motions.

Full-Context Driving Pose Injection. Our model builds upon a DiT-based Image-to-Video (I2V) model [36] and inject pose control signals for motion guidance. The common practice for pose information injection is through channel concat [4, 51] or using a pose adapter [38], where the driving pose sequence is embedded and added to the noisy video latents. We first implement the channel concat approach as illustrated in Figure 4, following [51]. While this method shows decent pose-following capability in generation, we find it tend to generate unnatural human pose when the motion is complex despite conditioned on accurate and unambiguous pose. A typical example is the turning motion, where the model often fails to correctly distinguish between

In inference, we perform spatial alignment using a projection loss (Lproj) to remap the predicted 3D driving poses to the reference 2D frame to help the model anchor where the motion should be transferred to, and optimize the training process to Specifically, we optimize the following ob-

front and back views, generating awkward postures. We assume that this limitation stems from the per-frame addition scheme, which fails to provide sufficient temporal context for motion understanding. Since motion is inherently sequential, many actions can only be interpreted correctly within a temporal context. Therefore, we design full-context pose injection, which enables the model to reason over the entire pose sequence. Specifically, we directly concatenates conditional pose tokens with noisy video tokens to facilitate spatial-temporal interactions between the two modality. As shown in Figure 4, this strategy handles motion correctly in complex scenarios.

Video token Pose token

[Figure 124]

…

…

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

…

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Pooling

+

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

… …

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

…

Generation

Generation

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

𝑇 𝑇 + 1 𝑇 + 2 𝑇 𝑇 + 1 𝑇 + 2

(a) Full-Context Injection Strategy (b) Channel Concat Strategy

Figure 4. Exploration of different strategies for pose injection.

To mitigate the sequence length increase introduced by our context-aware injection scheme, we apply spatial downsampling to the pose video. Empirically, we find that with 2× downsampling the pose following ability is nearly unaffected. Thus, we adopt this as the default setting in both training and testing, achieving balance between generation quality and efficiency.

##### 3.3. Data Pipeline

Pose Extraction. We adopt [26] for 3D keypoints extraction. However, the default extraction pipeline often fails to detect correct limbs under occlusion, especially in multi-human interactions. To address so, we apply a segmentation-and-extraction approach to obtain reliable poses in human interactions. To be more specific, we employ Samurai [45] to track and split the mask of each character, generating multiple single-human video splits. Subsequently, 3D keypoints are extracted for each video that contain one main character. Finally, The 3D skeletons are represented as cylindrical segments in 3D space. Those cylindrical segments are then composed together in 3D space and rasterized on the 2D canvas. This procedure preserves interperson occlusion relationships, benefiting from NLFPose’s accurate depth estimation. In practice, we find that our proposed multi-stage pipeline provides more accurate estimation than direct multi-person pose estimation methods like PromptHMR [39], especially in wild cases involving complex interaction. We follow the same extraction pipeline in inference.

Shifted RoPE for Pose Context. Conventional pipelines typically extract poses that are spatially aligned with the original video. However, in our setting, augmentations such as scaling and camera transformations often prevent the extracted pose context to be spatially aligned with the driving video. To address this misalignment, we introduce PoseShifted RoPE mechanism, which enables the model to effectively retrieve driving signals from the augmented fullsequence pose representation.

As illustrated in Figure 3, conventional 3D RoPE encodes position in the form (t,h,w), where t is the temporal frame index and h,w denote spatial dimensions, each starting from zero. In our design, Pose-Shifted RoPE applies a shift along the width dimension specifically to pose representations. The positional vectors for each driving pose token d are defined as:

Data Curation. To meet the requirements for generating studio-grade character animation, we have established the following criteria for data filtering: (1) the character should be the primary focus of each frame, (2) the person should exhibit explicit motion, and (3) the pose should be visually complete, covering either the upper body or the full body. We first detect human presence using YOLO [33] and discard clips where characters are not the main subjects. Next, 2D keypoints are extracted using DWPose [46], retaining only half-body or full-body videos. After this stage, multiperson videos can be further separated based on the number of bounding boxes detected by YOLO [33]. We further employ a large language model [7] to analyze captions, complementing the complex rule-based logic of multi-person motion bounding box checking. To compel the model to learn from complex motion dependencies, we compute the motion speed from the estimated 3D keypoints and discard samples with limited motion, resulting in a clean and

###### Pos = [t, h, Wmax : Wmax + shiftW], (4)

where shiftW is a constant shift magnitude and Wmax is the width of the reference image. As show in Figure 3, this design helps the model to distinguish driving pose tokens from reference image tokens and noisy video tokens in conjunction with the modified I2V Mask and reference image token injection strategy, enhancing the overall performance. Considering the modality gap between the noisy video tokens and the driving pose tokens, we observe that the model performs best particularly when the shift distance is a relatively large constant. To accommodate different downsampling ratios for pose context, mean-pooling on the 3D-RoPE [29] frequencies is performed according to the applied ratio.

- 3) SAMURAI Track & Split

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

- 4) NLFPose 3D-Keypoints Detection &Taichi Rasterization

[Figure 151]

###### Original Videos

𝑴𝒂𝒔𝒌

[Figure 152]

𝑺𝒑𝒍𝒊𝒕

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

Bboxes + Keypoints as Initial Prompt

VLM Captioning

𝑺𝒑𝒍𝒊𝒕

𝑴𝒂𝒔𝒌

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Mask Tracking

[Figure 170]

[Figure 171]

[Figure 172]

- 1) YOLO Detection

|[Figure 173]<br><br>[Figure 174]|
|---|

No Character

|[Figure 175]<br><br>[Figure 176]|
|---|

Too Small Character(s)

|[Figure 177]<br><br>[Figure 178]|
|---|

|[Figure 179]<br><br>[Figure 180]|
|---|

Single Character

Multi Characters

[Figure 181]

[Figure 182]

|[Figure 183]<br><br>[Figure 184]|
|---|

Too Many Characters

- 2) DWPose Detection

…

[Figure 185]

𝑺𝒑𝒍𝒊𝒕𝒏 …

𝑴𝒂𝒔𝒌 …

Multi Characters

LLM Caption Recheck

###### Single Character Splits

A man and a woman dance in a room with wooden floors …… their bodies get close during interaction yet apart when leaping.

Single Character

𝒞 = Cylinder 𝐉 , , ,𝐉 , , ,𝑟 ｜ 1 ≤ 𝑖 ≤ 𝑇,1 ≤ 𝑝 ≤ 𝑃, 𝑚 𝑛 ∈ ℬ}

𝒥

𝑺𝒑𝒍𝒊𝒕

###### Motion Speed Check

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

1 𝑇

Final Rendering

˜ J − J˜

𝑣 =

𝒥

[Figure 191]

𝑺𝒑𝒍𝒊𝒕

[Figure 192]

|[Figure 193]<br><br>[Figure 194]|
|---|

|[Figure 195]<br><br>[Figure 196]|
|---|

Joints Detection

[Figure 197]

|[Figure 198]<br><br>[Figure 199]|
|---|

|[Figure 200]<br><br>[Figure 201]|
|---|

|[Figure 202]<br><br>[Figure 203]|
|---|

|[Figure 204]<br><br>[Figure 205]|
|---|

…

…

𝒥

𝑺𝒑𝒍𝒊𝒕

Limited Motion Dynamic Motion

Incomplete Body

Portrait WholeBody

- Figure 5. The data curation pipeline. We perform character filtering and motion-speed filtering to construct high-quality training data.

reference image and driving video differ in identity or domain. We manually select a set of real reference images to test generalization, and further construct additional reference images using an image generation tool [27] to introduce variations in style and body shape, obtaining 120 single-character pairs and 10 multi-character pairs. As we simulate in-the-wild reference-driving discrepancy, pose retarget should be enabled for single-character pairs in this subset. We incorporate the retarget logic of Unianimatedit [38] for VACE [17] which do not natively support retargeting. For this subset, we conduct user study to assess the generation quality. All quantitative results on the subset are based on the 120 single-character pairs to ensure fairness considering some baselines [35] are incompatible with multi-character settings. Evaluation samples are strictly excluded from the training data.

motion-rich training set. The motion speed v is calculated as:

T−1

1 T

J ¯t+1,j − J¯t,j 2 (5)

v =

t=1 j

where j refers to the joints in the screen and ¯Jt = Jt − Jroott denotes the 3D human joint positions relative to the body center at frame t.

We collect around 250K high-quality motion-rich videopose pairs using the filtering pipeline, among which 20K pairs featuring multiple characters. To further enhance model’s performance on complex motions, we select 12K samples with the highest motion speed from dance, general motion, and obtain a finetuning set of 4,000 high-dynamic videos with minimal blur after manual clarity inspection.

##### 3.4. Studio-Bench for Evaluation

#### 4. Experiments

To evaluate the model capabilities under studio-grade standards, we propose Studio-Bench, a new evaluation benchmark tailored for Studio-Grade character animation. Previous evaluations [15, 31, 38, 44, 49, 52] primarily focus on simple actions, which fail to capture the challenges present in film production, such as complex and dynamic motions, multi-person interactions, and cross-domain animations. Specifically, our benchmark consists of two parts: Self-Driven and Cross-Driven. The first part evaluates motion adherence under complex actions. We examine whether the generated videos maintain correct body structure during large-scale motions, and whether the model can accurately capture inter-person relationships in multicharacter scenes, totally containing 130 clips. Quantitative metrics can be computed by directly comparing generated results with paired ground-truth videos for this subset. The second part measures the model’s transferability when the

##### 4.1. Implementation Details

We train two versions (1.3B and 14B) of our model. The 1.3B model is finetuned from the Wan2.1-1.3B-Fun-Inp backbone on our pretraining dataset for 6,000 steps with a batch size of 96 and a learning rate of 1e-5, using 32 NVIDIA H100 GPUs for approximately two days. For the larger 14B model, we finetune it from the Wan2.1-I2V-14B backbone in two stages: during the pretraining stage, we train for 8,000 steps with a batch size of 96 at a learning rate of 1e-5; after convergence, we perform an additional finetuning stage for 400 steps with the same batch size and a reduced learning rate of 4e-6. Training of the 14B model is conducted on 128 NVIDIA H100 GPUs for over four days with sequence parallelism enabled. All models are optimized using AdamW [21]. During inference, we set the

Self-Driven Animation Cross-Driven Animation

Methods

PSNR↑ SSIM↑ LPIPS↓ FVD↓ Mot-Acc↑ Kin-Consis↑ Phy-Consis↑ ID-Sim↑

UniAnimate-DiT [38] 17.79 0.637 0.242 362.27 2.5% 1.7% 0.8% 1.7% VACE [17] 16.73 0.588 0.263 264.71 9.2% 14.2% 18.3% 32.5% Wan-Animate [4] 18.54 0.648 0.221 187.61 35.0% 28.3% 24.2% 20.0%

###### SCAIL-14B (Ours) 19.22 0.660 0.206 176.16 53.3% 55.8% 56.7% 45.8%

Table 1. Quantitative comparison for SCAIL-14B and baselines. All compared methods are built upon 14B Wan [36] foundation models.

|Mot-Acc<br><br>Kin-Consis<br><br>Phy-Consis<br><br>ID-Sim<br><br>89.2% 10.0%<br><br>80.8% 15.8%<br><br>77.5% 5.8% 16.7%<br><br>48.3% 20.0% 31.7%<br><br>SCAIL-14B Vs VACE<br><br>Mot-Acc<br><br>Kin-Consis<br><br>Phy-Consis<br><br>ID-Sim<br><br>36.7% 30.0% 33.3%<br><br>45.0% 24.2% 30.8%<br>46.7% 28.3% 25.0%<br><br><br>25.0% 49.2% 25.8%<br><br>SCAIL-14B Vs Wan-Animate<br><br>0% 20% 40% 60% 80% 100%<br><br>Mot-Acc<br><br>Kin-Consis<br><br>Phy-Consis<br><br>ID-Sim<br><br>21.7% 54.2% 24.2%<br><br>34.2% 34.2% 31.7%<br><br>85.0% 9.2% 5.8%<br><br>63.3% 18.3% 18.3%<br><br>SCAIL-14B Vs Viggle<br><br>Win Rate Tie Rate Lose Rate<br><br>|
|---|

- Figure 6. User study for comparing our model with popular community and commercial projects.

classifier-free guidance (CFG) scale [10] to 4, offering a favorable balance between pose following and video fidelity.

##### 4.2. Quantitative Evaluation

We conduct a quantitative comparison with state-of-the-art methods on our proposed Studio-Bench. For self-driven part we employed several widely-used quantitative metrics, including PSNR [14], SSIM [40], LPIPS [48], and FVD [32]. To evaluate the generated results in the second subset, we design four metrics: (1) Motion Accuracy, which measures how faithfully the generated motion follows the driving signal in a frame-by-frame manner. (2) Kinesiology Consistency, evaluating whether joint rotations and body movements remain anatomically feasible and temporally coherent, penalizing sudden twists or physically impossible poses that break natural motion continuity. (3) Physical Consistency, assessing whether the generated motions comply with basic physical constraints such as gravity, support, and momentum conservation, penalizing unrealistic behaviors like hovering in midair. (4) Identity Similarity, measuring the consistency of the subject’s appearance with the reference image. We convey a detailed blinded user study to collect these metrics, letting users vote for the best in baseline comparison. We also conduct a Win/Tie/Lose study to evaluate our performance between two commonly used

open-source frameworks, as well as the closed-source commercial product Viggle [35], omitting UniAnimate-DiT[38] due to obvious artifacts under studio-grade demandings. Viggle is widely believed to rely on a 3D foundation model rather than video diffusion and can be a strong baseline for frame-by-frame motion accuracy under complex scenarios.

To ensure methodological rigor, we note that the user study for selecting the best-performing model in Table 1 and the win/tie/lose evaluation in Figure 6 were conducted on different batches of participants, allowing the two sets of results to serve as cross-validations. Results in Table 1 show that our model performs better than current open-source works on Studio-Bench. Under the best-model selection scheme, our method shows significant improvements across all metrics, and the win/tie/lose study further highlights its advantages in motion-transfer accuracy over other video diffusion methods and state-of-the-art motion naturalness in the metrics of Kinesiology Consistency and Physical Consistency.

##### 4.3. Qualitative Evaluation

- Figure 7 shows qualitative comparisons across both selfdriven and cross-driven animation settings for singlecharacter animation. In self-driven cases, SCAIL produces animations with stable structure and accurate limb articulation, particularly in challenging motions such as spinning and bending. Baseline methods often exhibit incomplete pose following or limb abnormalities such as incorrect legs in ballet dancing. In cross-driven cases, our method still demonstrates stronger motion transferability even when large discrepancies exist between the character image and the first frame of the driving video. In contrast, prior methods produce artifacts such as body-shape drift, inconsistent textures, and structural distortions like arm–body interpenetration during acrobats.
- Figure 8 presents qualitative comparisons for multi-

character animations, where issues caused by occlusion and estimation error become more pronounced. In relatively clean cases without severe occlusions—such as the dancing scenario in the castle—baseline methods mainly struggle with identity and orientation ambiguities during character turning, similar to their failure patterns in single-character

###### Self-Driven Animation Cross-Driven Animation

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

Wan-AnimateInputsUniAnimateOursVACE

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

[Figure 269]

- Figure 7. Qualitative comparison for single-character animation. Rendered pose in Cross-Driven Animation are omitted for clarity. Zoom in for better visualization.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Ours2D-Pose3D-PoseVACEDrivingVideoWan-Animate

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

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

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Figure 8. Qualitative comparison for dual-character animation. Zoom in for better visualization.

1.3B w/o full context

Reference Image Driving Pose 1.3B Full 1.3B w/o PRoPE

Self-Driven Animation PSNR↑ SSIM↑ LPIPS↓ FVD↓

Methods

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

- (1)
- (2)
- (3)

SCAIL-1.3B (Ours) 18.08 0.639 0.249 228.62 SCAIL-1.3B w/ 2D Pose 17.08 0.619 0.284 295.36 SCAIL-1.3B w/o P-RoPE 17.79 0.637 0.280 269.35 Channel Concat-1.3B 17.69 0.626 0.262 263.63 Channel Concat-1.3B w/ 2D Pose 17.12 0.624 0.282 296.23 MimicMotion [49] 17.01 0.630 0.314 334.24

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

Table 2. Ablation study conducted on SCAIL-1.3B.

Figure 9. Visualization of ablation study. The w/o full-context denotes using the channel-concatenation strategy.

that our context-aware injection strategy enables the model to capture global motion dependencies and reason plausible human poses. Specifically, case (2) in 9 shows that through this strategy, when the estimator misidentifies the left part and the right part and extracts a posture like a forward lunge in a running driving video, our model with full-context pose injection can still generate correct running posture based on the global motion, as conditions from the reference frame and estimations in other frames clearly reflect the semantic context of running along the river.

animation. However, as complexity increases, as illustrated by the last two cases of Figure 8, pose overlapping makes it difficult for the baselines to correctly distinguish human limbs, causing the models to generate severe artifacts where body parts of different characters merge during motion. Furthermore, in multi-character settings, both 2D and 3D pose estimations may occasionally fail due to heavy occlusions (in the shown case they detect only the front character in certain frame), posing significant challenges for temporal and spatial reasoning. In contrast to the baselines which are unable to infer coherent actual poses in such scenarios, our model captures global pose-identity relationships and generates plausible results. These results highlight SCAIL’s strong ability to handle both large motion variations and cross-domain appearance gaps, producing videos that are more natural and visually appealing.

Ablation on Pose-Shifted RoPE. Removing Pose-Shifted RoPE (P-RoPE) leads to noticeable performance degradation (Table 2), especially in LPIPS and FVD, confirming that precise pose-aware positional encoding is crucial for maintaining image quality and temporal motion smoothness. Figure 9 shows that P-RoPE can strengthen the correspondence between motion cues and spatial structure, resulting in more accurate hand articulation and more reasonable foot grounding. Results from case (1) demonstrate that with full-context injection, P-RoPE yields the strongest disentanglement of character identity and pose guidance, enabling the model to faithfully preserve the subject’s appearance while following motion patterns.

##### 4.4. Ablation Study

We conduct ablation studies on SCAIL-1.3B model to evaluate the contribution of each component. For reference, we also compare against the U-Net based MimicMotion [49], highlighting the advantage of our DiT-based architecture. More ablations would be provided in Suppl. A2.

Ablation on the Pose Representation. To validate the effectiveness of our 3D-consistent pose representation, we compare with a 2D keypoints-based pipeline (extracted using DWPose [46] and denoted as w/ 2D Pose). Implementation details of the 2D keypoints-based baseline and qualitative comparisions are also provided in Suppl.A2. Results shows the pose representation is critical for the performance gain, and such improvements are more pronounced under full-context pose injection than under channel-concat scheme.

#### 5. Conclusion

In this work, we present SCAIL, a novel framework for studio-grade character image animation. By introducing a scalable and robust 3D pose representation and leveraging a novel full-context pose injection with shifted RoPE, we enhances spatiotemporal reasoning within a DiT architecture, allowing the model to generate structurally accurate and temporally consistent animations under challenging scenarios. With curated data pipeline and comprehensive StudioBench, we push character animation towards productionlevel standards. Experiments show that SCAIL achieves state-of-the-art performance in both self-driven and crossdriven animation. We believe this work provides a solid step toward practical, production-ready character animation.

Ablation on Full-Context Driving Pose Injection. As shown in Table 2, replacing channel concatenation with the proposed full-context injection consistently improves all the metrics when providing the model with temporally coherent and information-rich conditions. Qualitatively, We observe that generated videos exhibit fewer artifacts like limb tearing and structural collapse (shown in Figure 9), indicating

#### References

- [1] Jie An, Songyang Zhang, Harry Yang, Sonal Gupta, Jia-Bin Huang, Jiebo Luo, and Xi Yin. Latent-shift: Latent diffusion with temporal shift for efficient text-to-video generation. arXiv preprint arXiv:2304.08477, 2023. 3
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1
- [3] Junhao Chen, Mingjin Chen, Jianjin Xu, Xiang Li, Junting Dong, Mingze Sun, Puhua Jiang, Hongxiang Li, Yuhang Yang, Hao Zhao, et al. Dancetogether! identity-preserving multi-person interactive video generation. arXiv preprint arXiv:2505.18078, 2025. 2, 4
- [4] Gang Cheng, Xin Gao, Li Hu, Siqi Hu, Mingyang Huang, Chaonan Ji, Ju Li, Dechao Meng, Jinwei Qi, Penchong Qiao, et al. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055, 2025. 1, 2, 3, 4, 7
- [5] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7346–7356, 2023. 2
- [6] Xiao Fu, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 3dtrajmaster: Mastering 3d trajectory for multientity motion in video generation. In The Thirteenth International Conference on Learning Representations, ICLR, 2025. 2
- [7] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024. 5
- [8] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014. 2
- [9] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2
- [10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 7
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3
- [12] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 3
- [13] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video dif-

- fusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 1
- [14] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010. 7
- [15] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 1, 2, 3, 6
- [16] Yuanming Hu. Taichi: An open-source computer graphics library. arXiv preprint arXiv:1804.09293, 2018. 1
- [17] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025. 2, 3, 6, 7, 1
- [18] Kuaishou AI Team. Kling. https : / / kling . kuaishou.com/en, 2024. 2
- [19] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: layout-guided multi-view driving scenarios video generation with latent diffusion model. In European Conference on Computer Vision, pages 469–485. Springer, 2024. 2
- [20] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: a skinned multiperson linear model. ACM Trans. Graph., 34(6):248:1– 248:16, 2015. 2
- [21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR, 2019. 6
- [22] OpenAI. Sora. https://openai.com/index/ sora/, 2024. 2
- [23] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 2, 3

- [24] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, ICLR, 2024. 2
- [25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3
- [26] Istv´an S´ar´andi and Gerard Pons-Moll. Neural localizer fields for continuous 3d human pose and shape estimation. Advances in Neural Information Processing Systems, 37: 140032–140065, 2024. 3, 5
- [27] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward nextgeneration multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 6
- [28] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR, 2021. 2

- [29] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 2, 5

- [30] Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. Animate-x: Universal character image animation with enhanced motion representation. In The Thirteenth International Conference on Learning Representations, ICLR, 2025. 1, 2, 3, 4
- [31] Shuyuan Tu, Zhen Xing, Xintong Han, Zhi-Qi Cheng, Qi Dai, Chong Luo, and Zuxuan Wu. Stableanimator: Highquality identity-preserving human image animation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21096–21106, 2025. 1, 2, 6
- [32] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 7
- [33] Rejin Varghese and M Sambath. Yolov8: A novel object detection algorithm with enhanced performance and robustness. In 2024 International conference on advances in data engineering and intelligent computing systems (ADICS), pages 1–6. IEEE, 2024. 5
- [34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [35] Viggle AI. Viggle. https://viggle.ai/, 2024. 6, 7
- [36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 4, 7
- [37] Qilin Wang, Zhengkai Jiang, Chengming Xu, Jiangning Zhang, Yabiao Wang, Xinyi Zhang, Yun Cao, Weijian Cao, Chengjie Wang, and Yanwei Fu. Vividpose: Advancing stable video diffusion for realistic human image animation. arXiv preprint arXiv:2405.18156, 2024. 2
- [38] Xiang Wang, Shiwei Zhang, Longxiang Tang, Yingya Zhang, Changxin Gao, Yuehuan Wang, and Nong Sang. Unianimate-dit: Human image animation with large-scale video diffusion transformer. arXiv preprint arXiv:2504.11289, 2025. 1, 2, 3, 4, 6, 7
- [39] Yufu Wang, Yu Sun, Priyanka Patel, Kostas Daniilidis, Michael J. Black, and Muhammed Kocabas. Prompthmr: Promptable human mesh recovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1148–1159, 2025. 5
- [40] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 7
- [41] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE

Transactions on Visualization and Computer Graphics, 31

(2):1526–1541, 2024. 2

- [42] Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash Vahdat. Camco: Cameracontrollable 3d-consistent image-to-video generation. arXiv preprint arXiv:2406.02509, 2024. 2
- [43] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems, 35:38571–38584, 2022. 2
- [44] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. Magicanimate: Temporally consistent human image animation using diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1481–1490, 2024. 1, 6
- [45] Cheng-Yen Yang, Hsiang-Wei Huang, Wenhao Chai, Zhongyu Jiang, and Jenq-Neng Hwang. Samurai: Adapting segment anything model for zero-shot visual tracking with motion-aware memory. arXiv preprint arXiv:2411.11922,

2024. 5

- [46] Zhendong Yang, Ailing Zeng, Chun Yuan, and Yu Li. Effective whole-body pose estimation with two-stages distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4210–4220, 2023. 2, 5, 9, 1
- [47] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, ICLR, 2025. 1, 2
- [48] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 7
- [49] Yuang Zhang, Jiaxi Gu, Li-Wen Wang, Han Wang, Junqi Cheng, Yuefeng Zhu, and Fangyuan Zou. Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance. In International Conference on Machine Learning, 2025. 1, 2, 6, 9
- [50] Zhenghao Zhang, Junchao Liao, Menghao Li, Zuozhuo Dai, Bingxue Qiu, Siyu Zhu, Long Qin, and Weizhi Wang. Tora: Trajectory-oriented diffusion transformer for video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2063–2073, 2025. 2
- [51] Jingkai Zhou, Yifan Wu, Shikai Li, Min Wei, Chao Fan, Weihua Chen, Wei Jiang, and Fan Wang. Realisdance-dit: Simple yet strong baseline towards controllable character animation in the wild. arXiv preprint arXiv:2504.14977, 2025. 2, 4
- [52] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision, pages 145–162. Springer, 2024. 1, 2, 6

## SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations

### Supplementary Material

#### A1. Details on Pose Conditioning

Reference Image 3D Driving Pose

Driving Frame 2D Driving Pose

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

Input

Rendering Details. To preserve spatial relationships between joints, we represent the human skeleton using cylindrical structures. For efficient rendering of multiple skeletons, we employ a 3D rendering pipeline that first converts the cylinders into spatial voxels, followed by ray marching implemented via [16]. This strategy is highly optimized for modern GPUs, introducing negligible computational overhead. To facilitate person discrimination in multi-person motion sequences, we assign distinct color schemes to each individual. This enables the model to directly learn how to distinguish characters from the representation. We observe that this design effectively alleviates identity switching when the positions of the people interchange.

- (1)

Generated

- (2)

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Input (2)

SCAIL 1.3B w/ 2D Pose

SCAIL 1.3B (Ours)

[Figure 341]

[Figure 342]

|[Figure 343]|
|---|

Generated (1)

[Figure 344]

[Figure 345]

| | |[Figure 346]|
|---|---|---|
| | | |

Augmentation Details. As our augmentation strategy is designed to maximally preserve the original motion, we use a high augmentation rate of 0.8 to achieve the balance between the pose-following accuracy and motion transfer robustness. During training, the overlaid 2D hand and face keypoints extracted by DWPose [46] will go through additional augmenation after the 3D adaptation process where

Figure A1. Ablation studies on pose representation. Anomalies in the human body or deviations from correct posture are boxed.

#### A2. More Ablation Studies

We conduct further ablation studies on SCAIL-1.3B model to evaluate the contribution of our proposed components. All training settings, including learning rate, data, batch size, and training steps, are kept exactly the same across all models in our ablation studies. Additional user studies are conducted to collect users’ preference towards different configurations for the SCAIL-1.3B model.

- 2D keypoints are shifted to match the reformed 3D skeleton after body rescaling and camera manipulation. This step is to minimize the unintended influence of the 2D facial and hand signals on the 3D pose representation. Performance Tradeoff. For the injection of the representation in full-context settings, we observe that the rendered pose sequence is relatively sparse as most frame areas consist of non-informative black pixels. In our method, spatial pooling to pose sequence can serve as a simple workaround to effectively reduce the contextual pose tokens to 1/4 and preserves accurate pose-following capability. In addition, full-context pose injection introduces no new parameters except the additional patchify layer to the original model, offering a more streamlined architecture compared to stacking additional DiT layers in residual context-tuning methods [17]. For quantitative comparison, we compare the inference costs of the two injection schemes (512 * 896 resolution, 81 frames, 20 diffusion steps) on an H100 GPU. In general, considering the significant motion error reduction, we conclude that the modest efficiency trade-off is acceptable, particularly in studio-grade scenarios which prioritize stringent accuracy and stability.

##### A2.1. Ablation details on the Pose Representation

As mentioned in the main paper, we implement the 2D augmentation strategy as close as to the 3D version, with similar figure settings and same augmentation ratio of 0.8. Faces and hands are also augmented identically to the 3D version and retargeting logic from [38] are applied during inference. As shown in Table 2, the estimation noise and the 2D-3D mismatch noise introduced by the pipeline significantly undermine the model’s performance in the self-driven subset of our challenging Studio-Bench, which involves a high ratio of complex motions.

For the cross-driven subset, 2D pose can easily lead to distorted limbs in generation especially when the 1.3B model have difficulty transferring motion to a significant different reference image, as seen in case (1) of Figure A1. Case (2) indicates that when model needs to distinguish the front and back of limbs, the inherent ambiguity of 2D pose can result in incorrect pose interpretations. Qualitative re-

Methods (all w/ CFG) Inference Time (s) FPS Memory (GB)

14B w/ channel-concat 286.11 0.283 61.7 14B w/ full-context (Ours) 380.78 (+33.1%) 0.213 (-24.9%) 68.5 (+11.0%)

|0% 20% 40% 60% 80% 100%<br><br>Mot-Acc Kin-Consis<br><br>Phy-Consis ID-Sim<br><br>27.5% 52.5% 20.0% 29.2% 54.2% 16.7% 41.7% 37.5% 20.8% 39.2% 45.0% 15.8%<br><br>SCAIL-1.3B (Ours) Vs SCAIL-1.3B w/o augmentation<br><br>Win Rate Tie Rate Lose Rate<br><br>|
|---|

3D Driving Pose w/o Retarget

3D Driving Pose w/ Retarget

SCAIL-1.3B (Ours)

Reference Image

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

3D Retarget

Driving Frame

- Figure A3. User study results of ablation on 3D Augmentation.

Reference Image Driving Frame Driving Pose 1.3B Full 1.3B w/o augmentation

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

- Figure A4. Qualitative results of ablation on 3D Augmentation.

| | |
|---|---|
| | |

[Figure 356]

2D Driving Pose w/o Retarget

2D Driving Pose w/ Retarget

SCAIL-1.3B w/ 2D Pose

First Driving Frame

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

2D Retarget

(To Compare)

Figure A2. Ablation studies on pose retargeting. 2D Retarget are visualized for comparison. Regions where the body proportions deviate from the reference image are boxed.

In the case of cross-driven animation, 3D augmentation significantly mitigated identity leakage, particularly notable in characters with substantial body shape differences, as illustrated in Figure A4. To quantify this improvement, we conduct a user study comparing two groups: with and without 3D augmentation. The results reveal that 3D augmentation clearly enhances the metric of Physical Consistency and Identity Similarity, endowing our model with better generalization capabilities when handling in-the-wild characters.

sults from the cross-driven subset and quantitative results from the self-driven subset together demonstrate the overall effectiveness of our proposed 3D-based solution. Figure A2 can also demonstrate the unreasonable scaling factors introduced by 2D-based retargeting process during the inference step, which we will discuss in the next section of retarget ablations.

##### A2.2. Ablation on 3D-Consistent Adaptation

#### A3. Data Source

Furthermore, we validate the key component in our pose representation: 3D-Consistent Adaptation. 3D-Consistent Adaptation includes 3D Retarget in cross-driven inference and 3D Augmentation in training.

Our training dataset is composed of three primary data sources: (1) samples retrieved from our internal base model training data and other downstream tasks, (2) a large collection of high-resolution dance videos from Bilibili and YouTube, and (3) additional sports videos such as gymnastics and figure skating. To ensure diversity, we maintain a certain proportion of stylized content, including 3D and 2D

Ablation on 3D Retarget. Figure A2 shows the comparison of the driving pose with and without 3D Retarget. 3D Retarget helps transfer the motion to the person without introducing position change. Compared to 2D Retarget, 3D Retarget keeps the original motion without introducing limb length distortion. Note that in our Studio-Bench, we only include cases where 2D Retarget works well for a fair comparison of the the model itself’s performance against other baselines. In wild scenarios, however, 3D information and camera parameters can help create highly robust retarget rules that are suitable for production-level use.

- animations from source (1), as well as MMD and Live2D
- animations from source (2).

#### A4. Details on Studio-Bench

To comprehensively evaluate studio-grade scenarios, we curate an diverse set of motion sequences in our StudioBench, as illustrated in Figure A5. The motion collection in our dataset emphasizes complex human-body configurations, covering a wide spectrum of challenging inputs, including dance, sports, martial-arts, acrobats and so on. In addition to isolated single-person motions, the test set also contains a small portion of interactions between the person and the environment, as well as several cases of multiperson interactions like dual dancing. We also include certain portion of fine-grained motions which are commonly featured in advertisement poses and iconic movie gestures,

Ablation on 3D Augmentation. 3D augmentation is central to our method’s ability to adapt to different characters. We conducted experiments to evaluate the impact of using

- 3D augmentation in the context of self-driven animation. The results indicated that even with a high augmentation ratio, there was no significant difference in the metrics compared to when augmentation was not used. This is because 3D augmentation effectively preserves motion information by only altering figure shape and maintaining the temporal motion semantics.

Environment Interaction

Generic Dance

Dual-dance

HandMovements

Fighting

Stage-performing

...

...

Turning

BodyLanguage

Hosting&Emceeing

Ballet

Interaction

MovieIconic

Dance

MotionsBasic

Swing

Ad-acts

Rotating

...

Skating

Acrobatics &Skills

Bending

Handstanding

Jumping

Kneeing Jumping

Fitness& MartialArts

Flipping

...

Rotating

Acrobat

Gym-acts

Turning

Taekwondo

Yoga

...

Sports

Kungfu

Figure A5. Visualization of the distribution of data annotations in Studio-Bench. We categorize and annotate videos based on motion types. The annotation of a single video can contain multiple tags, such as ”turning” and ”ballet”.

to evaluate the model’s all-around capability.

For the construction of cross-driven cases, we intentionally select reference character images to cover diverse figure shapes and different facial characteristics. On top of these real human references, we additionally introduce approximately 40 non-real characters. Most of them originate from 3D animated productions, while others include 2D animated characters, plush toys, anime figurines, and various stylized representations.

##### A4.1. More Examples on Studio-Bench

Example from the main paper demonstrate that our model can handle both fine-grained motions and highly complex in-the-wild motions. Figure A6 and Figure A7 will provide more cases with nonstandard figures to show our model’s generalization ability across a wide range of subjects and artistic styles. Figure A6 demonstrates that SCAIL can produce accurate limb motions that respect the features of nonstandard figures such as thin-limb anime characters. Furthermore, when the driving image is drastically different from the standard human figure (such as a plush toy with a very short body), SCAIL avoids the undesirable changes in body proportions that often plague baseline models. When the dual challenges of complex motions and non-standard character figures appear together in Figure A7, the issues with baseline models become even more pronounced. Another point worth noting is the scenario of reverse driving, where an anime character’s motion is used to drive a real person. While most user inputs involve a real person driv-

ing another figure, this less common use case presents demand of making a real person mimic an anime posture. We found that previous methods produce strange proportions under such inputs in Figure A8 while our model is still capable of handling this task. These comparisons highlight the strong potential of our approach for studio-grade applications where character compatibility for diverse motion types are critical requirements.

#### A5. Discussion

##### A5.1. Limitations and Future Work

Although we have adopted a relatively effective multiperson pose extracting pipeline, the accuracy of multiperson pose estimation is still not as precise as that for single-person scenarios. While we are able to make the model more robust to inaccurate poses through tailored model architecture design and sufficient data, we still look forward to advances in the field of multi-person pose estimation to further improve the fidelity of motion replication.

Moreover, our SCAIL model currently relies on facial landmarks to achieve face control. Such a representation is inherently limited in fine-grained facial expression. As our work primarily targets addressing the challenges including instability and motion artifacts in studio-grade video generation, enhancing the expressiveness of facial control is left for future exploration. Specifically, future work will focus on improving the accuracy and fidelity of fine-grained details, such as hands and facial expressions, to further elevate the model’s overall quality.

##### A5.2. Ethical Considerations

Our approach is designed to produce studio-grade, highfidelity character animation, enabling professional workflows in virtual production and cinematic pipelines. As our method advances character animation to a new level of realism and expressiveness, the potential for misuse, particularly in generating misleading or harmful digital content, becomes an important consideration. Despite these concerns, we believe that fully open-sourcing our model will bring substantial value to the community. By promoting transparency and broad accessibility, we aim to encourage a wide range of responsible, creative, and innovative works.

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

Input

ref ref

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Ours

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

VACE

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

Wan-Animate

Figure A6. Comparison of model’s ability to preserve body structure for non-standard character figures.

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

Input

ref

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

Ours

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

VACE

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Wan-Animate

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

Viggle

Figure A7. Visualization of our model’s performance under both high-dynamic motion and non-standard character figures.

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

Input

ref

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

Ours

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

VACE

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

Wan-Animate

Figure A8. Visualization of our model’s performance under reverse driving settings.

