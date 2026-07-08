## Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation

CHENJIE CAO, Alibaba DAMO Academy, Fudan University, Hupan Lab, China JINGKAI ZHOU, Alibaba DAMO Academy, Hupan Lab, China SHIKAI LI, Alibaba DAMO Academy, Hupan Lab, China JINGYUN LIANG, Alibaba DAMO Academy, Hupan Lab, China CHAOHUI YU∗, Alibaba DAMO Academy, Hupan Lab, China FAN WANG, Alibaba DAMO Academy, China YANWEI FU†, Fudan University, Shanghai Innovation Institute, China XIANGYANG XUE, Fudan University, China

# arXiv:2504.14899v2[cs.CV]20Sep2025

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

(a)Cameracontrol

(b)Unifiedcameraandhumanmotioncontrol

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

First view and depth Camera trajectories Upper: conditions, below: generated video

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

(c)Motiontransfer

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Target trajectories

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Reference motion

First view and depth Camera trajectories Upper: conditions, below: generated video

Fig. 1. Given a single-view image across various domains (e.g., real-world, text-to-image, animation), we first extract the monocular depth and focal length of it via Depth-Pro [Bochkovskii et al. 2024] and then achieve point clouds. Then, the proposed Uni3C can generate impressive videos under arbitrary (a) camera trajectories, (b) human motion characters (SMPL-X [Pavlakos et al. 2019]), or both of these conditions. (c) Uni3C further supports the camera-controlled motion transfer. Please review the videos in the supplementary for more details.

Camera and human motion controls have been extensively studied for video generation, but existing approaches typically address them separately, suffering from limited data with high-quality annotations for both aspects. To overcome this, we present Uni3C, a unified 3D-enhanced framework for precise control of both camera and human motion in video generation. Uni3C includes two key contributions. First, we propose a plug-and-play control module trained with a frozen video generative backbone, PCDController, which utilizes unprojected point clouds from monocular depth to achieve

accurate camera control. By leveraging the strong 3D priors of point clouds and the powerful capacities of video foundational models, PCDController shows impressive generalization, performing well regardless of whether the inference backbone is frozen or fine-tuned. This flexibility enables different modules of Uni3C to be trained in specific domains, i.e., either camera control or human motion control, reducing the dependency on jointly annotated data. Second, we propose a jointly aligned 3D world guidance for the inference phase that seamlessly integrates both scenic point clouds and SMPL-X characters to unify the control signals for camera and human motion, respectively. Extensive experiments confirm that PCDController enjoys strong robustness in driving camera motion for fine-tuned backbones of video generation. Uni3C substantially outperforms competitors in both camera controllability and human motion quality. Additionally, we collect tailored validation sets featuring challenging camera movements and human actions to validate the effectiveness of our method. Codes are released at https://github.com/alibaba-damo-academy/Uni3C.

∗Project lead. †Corresponding author. Prof. Yanwei Fu is with the Institute of Trustworthy Embodied Al, and the School of Data Science, Fudan University.

Authors’ Contact Information: ChenJie Cao, Alibaba DAMO Academy, Fudan University, Hupan Lab, China, ccjdurandal422@163.com; JingKai Zhou, Alibaba DAMO Academy, Hupan Lab, China; ShiKai Li, Alibaba DAMO Academy, Hupan Lab, China; JingYun Liang, Alibaba DAMO Academy, Hupan Lab, China; ChaoHui Yu, Alibaba DAMO Academy, Hupan Lab, China; Fan Wang, Alibaba DAMO Academy, China; Yanwei Fu, Fudan University, Shanghai Innovation Institute, China, yanweifu@fudan.edu.cn; XiangYang Xue, Fudan University, China.

CCS Concepts: • Networks → Network algorithms; • Computing methodologies → Computer vision.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Additional Key Words and Phrases: Camera control, Human animation, Video generation, Generative models

#### ACM Reference Format:

ChenJie Cao, JingKai Zhou, ShiKai Li, JingYun Liang, ChaoHui Yu, Fan Wang, Yanwei Fu, and XiangYang Xue. 2025. Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM 1557-7368/2025/9-ART https://doi.org/10.1145/nnnnnnn.nnnnnnn

ACM Trans. Graph. 1, 1 (September 2025), 14 pages. https://doi.org/10.1145/ nnnnnnn.nnnnnnn

1 Introduction

Recentadvancementsinfoundational video diffusion models (VDMs) [Blattmann et al. 2023; Brooks et al. 2024; Kong et al. 2024; Kuaishou 2024; RunwayML 2024; Wang et al. 2025a; Yang et al. 2025] have unlocked unprecedented capabilities in creating dynamic and realistic video content. A significant challenge in this field is achieving controllable video generation, a feature with broad applications in virtual reality, film production, and interactive media. In this paper, we focus on two aspects of controllable video generation: camera control [Bahmani et al. 2025a; He et al. 2025a; Ren et al. 2025; Yang

- et al. 2024; Yu et al. 2024; Zheng et al. 2025a] and human motion control [Hu 2024; Hu et al. 2025; Tan et al. 2025; Wang et al. 2024a; Xu et al. 2024b; Zhou et al. 2024]—both of which are critical and interdependent in real-world scenarios.

Recent pioneering works have extensively studied controlling cameratrajectoriesforVDMsthrough explicit conditions like Plücker ray [Bahmani et al. 2025a; He et al. 2025a,b; Liang et al. 2024; Zheng et al. 2025a], point clouds [Feng et al. 2025; Li et al. 2025b; Popov

- et al. 2025; Ren et al. 2025; Yu et al. 2024]. Concurrently, controllable human animation also attracted a lot of attention based on poses [Hu

- 2024; Hu et al. 2025; Tan et al. 2025; Xu et al. 2024b] or SMPL characters [Zhou et al. 2024, 2025b; Zhu et al. 2024]. Despite these advancements, several challenges remain: 1) Most approaches deeply hack the inherent capacities of VDMs, which have been trained with domain-specific data and conditions, inevitably undermining the generalization to handle out-of-distribution scenarios. 2) Very few works investigate the joint control of both camera trajectories and human motions. This requires diverse camera trajectories in humancentric videos with high-quality annotations [Wang et al. 2024a], which are often expensive to obtain. 3) There is a lack of explicit and synchronized guidance that incorporates strong 3D-informed priors to control both camera movements and human motions concurrently. Relying on separate conditions, like point clouds and SMPL, struggles to represent physically reasonable interactions and positional relations between characters and environments, resulting in conflicting guidance and suboptimal outcomes.

To address these challenges, we present Uni3C, a novel framework that Unifies precise 3D-enhanced camera and human motion Controls for video generation as shown in Figure 1 via two key innovations. Firstly, we propose the insight that controlling camera trajectories for powerful foundational VDMs can be achieved by lightweight, trainable modules with rich informed guidance and reasonable training strategies. By avoiding hacking the underlying capacities of VDMs, our camera-controlling model can be directly generalized to versatile downstream tasks rather than costly joint training and extensive Structure from Motion (SfM) labeling [Li et al. 2025a; Schönberger et al. 2016; Zhao et al. 2022]. Secondly, we claimed that camera and human motion controls are inherently interdependent. Therefore, we propose to align their conditions into a global 3D world during the inference phase, enabling 3D consistent generation across both domains.

Formally, our Uni3C is built upon the foundational VDM—Wan2.1 [Wang et al. 2025a]. For the control of camera trajectories, we propose PCDController, a plug-and-play control module with only 0.95B parameters (compared to 14B of Wan2.1) that operates on unprojected 3D point clouds derived from monocular depth estimation [Bochkovskii et al. 2024]. Thanks to the rich geometric priors of point clouds, PCDController is capable of fine-grained camera control, even when trained on constrained multi-view images and videos with a frozen backbone. Furthermore, PCDController can be compatible with fine-tuned VDM backbones for versatile downstream tasks. This surprising factor supports domain-specific training, i.e., camera and human motion modules can be trained independently without jointly annotated data. For the global 3D world guidance, we align scenic point clouds (for camera control) and SMPL-X characters [Pavlakos et al. 2019] (for human animation) into the same world-coordinate space via the rigid transformation [Umeyama 1991], while the 2D keypoints [Xu et al. 2023] bridge the relation of two presentations. Note that our alignment enables complicated motion transfer, covering disparate characters, positions, and viewpoints as verified in the last row of Figure 1.

Extensive experiments validate the efficacy of Uni3C. To evaluate the remarkable generalization of PCDController, we collect an outof-distribution test set across different domains, where each image has four different camera trajectories. For the joint controllability, we build a comprehensive test set of in-the-wild human videos. GVHMR [Shen et al.2024] is used to extract SMPL-X as the condition, while three complex and random camera trajectories are assigned for each video. VBench++ [Huang et al. 2024] is employed to verify the overall performance of our method. Uni3C significantly outperforms other competitors, both quantitatively and qualitatively.

We highlight the key contributions of Uni3C as:

- • PCDController. A robust, lightweight camera control module is proposed, which enjoys strong 3D priors from point clouds, compatible with both frozen or adapted VDMs.
- • Global 3D World Guidance. A unified inference framework that aligns scene geometry (point clouds) and human characters (SMPL-X) for 3D-coherent video control.
- • Comprehensive Validation. We propose new benchmarks and datasets to evaluate challenging camera-human interaction scenarios, demonstrating Uni3C’s superiority over existing approaches.

2 Related Work 2.1 Camera Control for VDMs

Controlling camera trajectories in video generation has garnered significant attention recently. Some works focused on injecting camera parameters into VDMs to achieve camera controllability [Bahmani et al. 2025a,b; He et al. 2025a,b; Liang et al. 2024; Wang et al. 2024b; Zheng et al. 2025a], typically utilizing the Plücker ray presentation. For instance, VD3D [Bahmani et al. 2025b] designed a tailored framework for Diffusion Transformer (DiT) [Peebles and Xie 2023], while AC3D [Bahmani et al. 2025a] further emphasized the generalization with fewer trainable parameters. Moreover, DimensionX [Sun et al. 2024] further decoupled the spatial and temporal control with different LoRAs [Hu et al. 2022]. Despite the progress made by these

Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation • 3

methods, they cannot control the camera movements precisely, particularly when the case is beyond the training domains with an unknown metric scale. Thus, other recent works have employed point cloud conditions via training-based [Feng et al. 2025; Li et al.

- 2025b; Popov et al. 2025; Ren et al. 2025; Yu et al. 2024] and trainingfree [Hou et al. 2025; You et al. 2025] manners. However, they fail to accommodate the generalized model design and training strategy to handle imperfect point clouds or out-of-distribution data, especially in motional scenarios involving humans or animals.

2.2 Unified Control for VDMs

Recent works have unified multiple conditions to guide video generation [Chen et al. 2025; Feng et al. 2025; Geng et al. 2025; Gu et al. 2025; Wang et al. 2024a,b; Zheng et al. 2025b]. MotionCtrl [Wang et al. 2024b] integrated the camera and object motion controls through separate pose and trajectory injections. Subsequently, researchers further explored the presentation of conditions, such as point trajectories [Feng et al. 2025], point tracking [Geng et al. 2025; Gu et al. 2025], and 3D-aware signals [Chen et al. 2025]. Moreover, VidCraft3 [Zheng et al. 2025b] considered the lighting control to VDMs for the first time. For the control of both cameras and human motions, some works focus on the behavior transfer based on camera and human motion estimation [Jiang et al. 2024; Kocabas et al. 2024]. Humanvid [Wang et al. 2024a] first unified them in video generation. However, it requires both camera and human guidance from the same source video; otherwise, the human pose would conflict with the camera movements, leading to limited flexibility for the decoupled control. Despite the promising performance demonstrated by these pioneering approaches, they often rely heavily on joint training under various conditions and well-labeled datasets. Furthermore, there has been limited discussion on unified control within foundational VDMs that exceed 10B parameters. Our work offers a solution to address these issues: unifying existing models for different downstream tasks without the need for costly fine-tuning or fully annotated datasets. This strategy is particularly well-suited for large VDMs, enabling models to focus more on enhancing performance within their specific domains.

3 Preliminary: Video Diffusion Models

We briefly review VDMs and Wan2.1 [Wang et al. 2025a] in this section as preliminary knowledge. VDMs are mainly based on the latent diffusion model [Rombach et al. 2022], modeling the conditional distribution 𝑝(𝑧0|𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔), where 𝑧0 indicate clean video latent features encoded by 3D-VAE; 𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔 denote the text condition for Text-to-Video (T2V) and the optional image condition for Image-to-Video (I2V), respectively. The training of VDM involves reversing the diffusion process by the noise estimator 𝜖𝜃 as:

E𝑧0,𝑡,𝜖,𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔 [∥𝜖𝜃 (𝑧𝑡,𝑡,𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔) − 𝜖∥2], (1)

min

𝜃

where 𝜖 ∼ N(0,𝐼) indicates Gaussian noise; timestep 𝑡 ∈ [0,𝑇𝑚𝑎𝑥]; 𝑧𝑡 is the intermediate noisy latent state of timestep 𝑡. Recently, most VDMs have employed Flow Matching (FM) [Lipman et al. 2022] as the improved diffusion process with faster convergence and more stable training. Based on the ordinary differential equations (ODEs), FM formulates the linear interpolation between 𝑧0 and 𝑧1,

###### Uni3C

Control Branch Main Backbone

PCDController (Camera Control) RealisDance-DiT (Human Animation)

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Camera Point clouds Reference image SMPL-X Hamer

- Fig. 2. The overview of Uni3C, which adopts multi-modal conditions. The camera, point clouds, and reference image are assigned to the camera control module called PCDController, while the reference image, SMPLX [Pavlakos et al. 2019], and Hamer [Pavlakos et al. 2024] are assigned to human animation called RealisDance-DiT [Zhou et al. 2025b].

[Figure 48]

[Figure 49]

Cam- PCDController Encoder

[Figure 50]

Wan-DiT

[Figure 51]

WanDecoder

[Figure 52]

Image CLIP

umT5

“A cartoonish bear sitting at a school desk in classroom.”

Firstframe

Image latent 𝑐

Mask

Noisy latent 𝑧

0-

padding

[Figure 53]

Point clouds

Render

𝑉

𝐼

WanEncoder

Point latent 𝑐

[Figure 54]

[Figure 55]

[Figure 56]

Plücker ray 𝑷

C

C

S

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Generated video

[Figure 62]

Element-wise addition S Stitching along length C Concatenation along channels

zero-linear

𝑐

|[Figure 63]|[Figure 64]|[Figure 65]|
|---|---|---|

…

[Figure 66]

RealisDance-DiT

(Inference with unified control)

[Figure 67]

(Train or camera control only)

or

- Fig. 3. Pipeline of PCDController, which is built as a lightweight DiT trained from scratch. We first obtain point clouds via monocular depth from the first view. Then, the point clouds are warped and rendered into the video 𝑉𝑝𝑐𝑑. Input conditions for PCDController comprise rendered 𝑉𝑝𝑐𝑑, Plücker ray P, and the noisy latent 𝑧𝑡. Only the PCDController and camera encoder are trainable in our framework. For inference of unified control over camera and human motions, we directly replace the frozen Wan backbone with RealisDance-DiT [Zhou et al. 2025b] without joint fine-tuning.

i.e., 𝑧𝑡 = 𝑡𝑧1 + (1 − 𝑡)𝑧0, where 𝑡 ∈ [0, 1] is sampled from the logit-normal distribution. The velocity prediction 𝑣𝜃 is written as:

min

𝜃

E𝑧0,𝑡,𝜖,𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔 [∥𝑣𝜃 (𝑧𝑡,𝑡,𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔) − 𝑣𝑡 ∥2], (2)

where the ground truth velocity denotes 𝑣𝑡 = 𝑑𝑧𝑑𝑡𝑡 = 𝑧1 − 𝑧0. Additionally, recent foundational VDMs [Brooks et al. 2024; Kong et al.

2024; Wang et al. 2025a; Yang et al. 2025] are built with DiT [Peebles and Xie 2023] to achieve more capacities for video generation. Wan2.1 [Wang et al. 2025a] is an open-released VDM with DiT architecturetrainedwithflowmatching[Lipman et al.2022]. umT5[Chung et al. 2023] is utilized as the multi-language text encoder to inject textual features into Wan2.1 through cross-attention. For image-tovideo, Wan-I2V further incorporates features from CLIP’s image encoder [Radford et al. 2021] to improve the results. Uni3C is primarily designed for Wan-I2V with 14B parameters, but we empirically find that it is compatible with the Wan-T2V version as verified in Section 5.4, showing convincing generalization.

- 4 Method Overview. We first present the overview of Uni3C in Figure 2.

Given a reference view 𝐼𝑖𝑚𝑔 ∈ R3×ℎ×𝑤, camera trajectories {𝑐𝑐𝑎𝑚𝑖 }𝑖𝑁=1 of 𝑁 target views, and textual condition, Uni3C uses PCDController

to produce the target video under specified camera trajectories. This process can be formulated as 𝑝(𝑧0|𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔,𝑐𝑐𝑎𝑚), where 𝑧0 indicates the clean latent video feature, 𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔 are textual features from umT5 and image latent condition encoded from 𝐼𝑖𝑚𝑔, respectively. We show the pipeline of PCDController in Figure 3 as a core component of Uni3C for generalized camera control (Section 4.1) across downstream tasks. In this work, we focus on human animation (Section 4.2). Subsequently, we introduce the global 3D world guidance illustrated in Figure 5 to unify both camera and human characters into a consistent 3D space for inference (Section 4.3).

- 4.1 PCDController with 3D Geometric Priors

Architecture. Following AC3D [Bahmani et al. 2025a], PCDController is designed within a simplified DiT module rather than copying modules and weights from the main backbone as ControlNet [Zhang et al. 2023]. To preserve the generalization of Wan-I2V, we follow the insight of training as few parameters as possible once the effective camera control has been achieved. Formally, we reduce the hidden size of PCDController from 5120 to 1024, while zeroinitialized linear layers are used to project the hidden size back to 5120 before being added to Wan-I2V. Moreover, as investigated in [Bahmani et al. 2025a; Liang et al. 2024], VDMs mainly determine camera information through shallow layers. Thus, we only inject camera-controlling features into the first 20 layers of Wan-I2V to further simplify the model. Additionally, we discard the textual condition for PCDController to alleviate intractable hallucination and remove all cross-attention modules. In this way, the overall number of trainable parameters for PCDController is reduced to 0.95B, a significant reduction compared to Wan-I2V (14B).

3D Geometric Priors. In contrast to merely utilizing Plücker ray as the camera embedding [Bahmani et al. 2025a; Liang et al. 2024], we incorporate much stronger 3D geometric priors to compensate the

model simplification, i.e., videos {𝑉𝑝𝑐𝑑𝑖 }𝑖𝑁=1 ∈ R𝑁×3×ℎ×𝑤 rendered from point clouds under given camera trajectories. Specifically, we first use Depth-Pro [Bochkovskii et al. 2024] to extract the monocular depth map from the reference view. We then align this depth map into a metric representation using SfM annotations [Schönberger et al. 2016] or multi-view stereo [Cao et al. 2024]. Following [Cao et al. 2025], we employ RANSAC to derive the rescale and shift coefficients, preventing the collapse of constant depth outcomes. Subsequently, the point clouds 𝑋𝑝𝑐𝑑 ∈ Rℎ𝑤×3 are got by unprojecting all 2D pixels from 𝐼𝑖𝑚𝑔 into the world coordinate via its metric depth 𝐷ˆ𝑖𝑚𝑔 as follows:

𝑋𝑝𝑐𝑑(𝑥) ≃ 𝑅𝑐→𝑤𝐷ˆ𝑖𝑚𝑔(𝑥)𝐾−1𝑥,ˆ (3)

where 𝑥 denotes the 2D coordinates of 𝐼𝑖𝑚𝑔, while 𝑥ˆ refers to the homogeneous coordinates; 𝐾,𝑅𝑐→𝑤 mean the intrinsic and extrinsic cameras of the reference view, respectively. After that, we render

{𝑉𝑝𝑐𝑑𝑖 }𝑖𝑁=2 for the remaining (𝑁 − 1) views by PyTorch3D according to their respective camera intrinsics and extrinsics. Note that the

first rendering corresponds to the reference image, i.e., 𝑉𝑝𝑐𝑑1 = 𝐼𝑖𝑚𝑔, to confirm the identity. We apply 𝑉𝑝𝑐𝑑 to the 3D-VAE of Wan2.1 to achieve 𝑐𝑝𝑐𝑑 as the point latent condition. To further handle the significant viewpoint changes, which may extend beyond the point clouds’ visibility of the first frame, PCDController also includes

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

(a) First view (b) Point & Generation (frame-22) (c) Point & Generation (frame-68)

Fig. 4. Results of PCDController with imperfect point clouds. Benefiting from the well-preserved capacity of VDM, PCDController enjoys robust generation with inferior point clouds.

Plücker ray embedding [Xu et al. 2024a], {P𝑖}𝑖𝑁=1 ∈ R𝑁×6×ℎ×𝑤, as the auxiliary condition. {P𝑖}𝑖𝑁=1 is encoded by a small camera encoder, comprising causal convolutions and a 4-8-8 downsampling factor to keep the same sequential length as 3D-VAE outputs. The distribution modeling of PCDController is 𝑝(𝑧0|𝑐𝑡𝑥𝑡,𝑐𝑖𝑚𝑔,𝑐𝑝𝑐𝑑, P).

Discussion. We empirically find that our method demonstrates robust performance even when working with imperfect point clouds obtained through monocular depth unprojection. In this context, the point clouds serve as the primary camera control signal, facilitating the convergence of training rather than dominating the multiview geometric and textural generations, as illustrated in Figure 4. Moreover, the lightweight PCDController retains its precise camera control capability over versatile fine-tuned Wan backbones, even without joint training. This flexibility allows for a range of downstream applications, showcasing the robustness of our approach.

- 4.2 Human Animation

In this paper, we explore the unified control through two human animation approaches, both of which are built on the Wan2.1 framework, targeting I2V and T2V, respectively. While these methods are not the primary focus of our work, we provide a brief introduction here. Formally, the concurrently pioneering work, RealisDanceDiT [Zhou et al. 2025b], directly replaces the Wan-I2V backbone for high-quality human animation during inference. RealisDance-DiT incorporates SMPL-X [Pavlakos et al. 2019] and Hamer [Pavlakos et al. 2024] as additional input conditions through newly zeroinitialized layers to guide human motions, while camera-control features are added via the external PCDController as shown in Figure 2. To ensure the flexibility for motion transfer, RealisDance-DiT randomly selects the reference frame in the video sequence, which is not perfectly aligned with the given SMPL-X. RealisDance-DiT only trains self-attention modules and patchify encoders to confirm the generalization. However, we clarify that integrating the control branch trained within different backbones is still challenging, requiring a generalized control branch like our proposed PCDController. Furthermore, we tried another version, RealisDance-DiT-T2V, based on Wan-T2V without reference image conditions to explore the generalization of PCDController. Remarkably, PCDController adapts successfully to Wan-T2V, empowering it with impressive I2V ability.

- 4.3 Global 3D World Guidance

Definition. As illustrated in Figure 2, Uni3C adopts multi-modal conditions, including camera, point clouds, reference image, SMPLX [Pavlakos et al. 2019], and Hamer [Pavlakos et al. 2024]. The first three conditions are used for PCDController, while the latter three

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation • 5

[Figure 78]

[Figure 79]

[Figure 80]

𝐼 with 2D landmarks Point clouds in 𝑊 Aligned SMPL-X in 𝑊

SMPL-X in 𝑊

Gravityofpointclouds

[Figure 81]

Rigid transform

[Figure 82]

GeoCalib

[Figure 83]

[Figure 84]

[Figure 85]

𝑠̃ 𝑅 𝑡̃

[Figure 86]

Gravity of SMPL-X

[Figure 87]

[Figure 88]

Horizontal plane of point clouds

(b) Gravity calibration under 𝑊

(a) Aligning SMPL-X from 𝑊 to 𝑊

[Figure 89]

[Figure 90]

[Figure 91]

(c) Camera controlled global 3D world guidance

Camera trajectories

- Fig. 5. Overview of global 3D world guidance. (a) We first align SMPL-X characters from the human world space𝑊ℎ𝑢𝑚 to the environment world space 𝑊𝑒𝑛𝑣 with dense point clouds. (b) GeoCalib [Veicht et al. 2024] is used to calibrate the gravity direction of SMPL-X. (c) Rigid transformation coefficients 𝑠,˜ 𝑅,˜ 𝑡˜ are employed to align the whole SMPL-X sequence. We re-render all aligned conditions under specific camera trajectories as the global 3D world guidance.

are used for human animations. We employ GVHMR [Shen et al. 2024] to recover SMPL-X characters. One can also retrieve desired motion sequences from motion datasets [Guo et al. 2022; Lin et al. 2023; Plappert et al. 2016; Punnakkal et al. 2021] or generate new motions through text-to-motion models [Barquero et al. 2024; Guo et al. 2024; Jiang et al. 2023]. Although most of the conditions above are formulated as 3D presentations, they stay in two different world coordinates. We define the point cloud world coordinate as the environmental world space 𝑊𝑒𝑛𝑣, which is the main world space controlled by cameras, while the SMPL-X is placed in the human world space 𝑊ℎ𝑢𝑚. It is non-trivial to control the camera across two different world spaces consistently. For example, determining the initial camera position within𝑊ℎ𝑢𝑚 is particularly ambiguous, especially for tasks involving motion transfer. Moreover, to facilitate flexible control over separate camera and human motions without conflicts, human conditions should be re-rendered under new camera trajectories. Therefore, we propose the global 3D world guidance that places the human condition into the “environment”, i.e., aligning SMPL-X from𝑊ℎ𝑢𝑚 to𝑊𝑒𝑛𝑣 as shown in Figure 5(b).

SMPL-X sequences under the assumption that they share the same rigid transformation. However, even minor orientation errors can accumulate, leading to physically unrealistic motion trajectories, such as ascending into the sky or descending into the ground. To address this, we adopt GeoCalib [Veicht et al. 2024] to estimate the gravity direction in𝑊𝑒𝑛𝑣, which is then employed to calibrate the SMPL-X to ensure parallel gravity directions, as illustrated in Figure 5(b). For the alignment of Hamer [Pavlakos et al. 2024], which shares common vertices with the hand parts of SMPL-X, Hamer can also be aligned to𝑊𝑒𝑛𝑣 through the rigid transformation (Equation (4)). Additionally, we address the issue of hand occlusion for Hamer by masking occluded hand parts based on the rendered depth from SMPL-X. After the alignments of SMPL-X and Hamer sequences, we place all conditions into𝑊𝑒𝑛𝑣, establishing the global 3D world guidance that allows for rendering concurrently controlled conditions under arbitrary camera trajectories and human motions as shown in Figure 5(c). Finally, these re-rendered conditions are sent to PCDController and RealisDance-DiT for generated outcomes, as shown in Figure 2.

Multi-Modal Alignment. Fortunately, 2D human pose keypoints subtly bridge𝑊ℎ𝑢𝑚 and𝑊𝑒𝑛𝑣. Formally, we first estimate 17 human keypoints {k𝑖2𝐷}𝑖17=1 ∈ R17×2 from the reference view 𝐼𝑖𝑚𝑔 by ViTPose++ [Xu et al. 2023]. Then, we unproject 2D keypoints into𝑊𝑒𝑛𝑣 to obtain 3D keypoints {k𝑖𝑒𝑛𝑣}𝑖17=1 ∈ R17×3 through metric monocular depth 𝐷ˆ𝑖𝑚𝑔 and the intrinsic camera of the reference image. For SMPL-X in 𝑊ℎ𝑢𝑚, the COCO17 regressor [Shen et al. 2024] is utilized to project the first frame’s SMPL-X character into {kℎ𝑢𝑚𝑖 }𝑖17=1 corresponding to the same human keypoints. Consequently, a leastsquares estimation [Umeyama 1991] based rigid transformation can be used to align kℎ𝑢𝑚 to k𝑒𝑛𝑣 as follows:

5 Experiments 5.1 Implementation Details

PCDController is trained with the frozen Wan-I2V [Wang et al. 2025a] on multi-resolution images scaled from [480×768, 512×720, 608 × 608, 720 × 512, 768 × 480] of 81 frames. The learning rate is warmed up to 1e-5 for 400 steps and then fixed. We train the model for 6,000 steps with batch size 32, while more training steps would slightly hurt Wan-I2V’s generalization. The training is accomplished with 64 H100 GPUs for 40 hours. We also provided results based on CogVideoX-5B-I2V [Yang et al. 2025], training for 20k steps with batch size 16. During the training, we randomly drop 10% texts, as well as 5% point cloud renderings and Plücker embeddings. For inference, we set the classifier-free guidance scale to 5.0 for textual conditions, keeping other guidance on the default scale 1.

### ∑︁17

𝑤𝑖∥(𝑠˜𝑅˜(kℎ𝑢𝑚𝑖 )𝑇 + 𝑡˜)𝑇 − k𝑖𝑒𝑛𝑣∥2, (4)

min

𝑠,˜ 𝑅,˜ 𝑡˜

𝑖=1

where 𝑠,˜ 𝑅,˜ 𝑡˜ indicate the optimized scaling factor, rotation matrix, and translationvector,respectively.𝑤𝑖 denotestheconfidenceweight of 2D keypoint k𝑖2𝐷. We discard any keypoints with confidence below 0.7, as they typically degrade alignment quality. Once the transformation parameters 𝑠,˜ 𝑅,˜ 𝑡˜ are determined, we apply them to all other

Datasets. To ensure the generalization of PCDController, we collect large-scale training data for camera control, including DL3DV [Ling et al. 2024], RE10K [Zhou et al. 2018], ACID [Liu et al. 2021], Co3Dv2 [Reizenstein et al. 2021], Tartainair [Wang et al. 2020],

Map-Free-Reloc [Arnold et al. 2022], WildRGBD [Xia et al. 2024], COP3D [Sinha et al. 2023], UCo3D [Liu et al. 2025]. This comprehensive dataset encompasses various scenarios, featuring static and dynamic scenes, as well as object-level and scene-level environments. Furthermore, all datasets are annotated with metric-aligned monocular depth through the way proposed in [Cao et al. 2025] or are provided with ground-truth depth.

- 5.2 Results of Camera Control

Benchmark. To evaluate camera control ability, we built an outof-distribution benchmark with 32 images across various domains, including text-to-image generation1, real-world [Barron et al. 2022; Knapitsch et al. 2017], object-centric, and human scenes as in Figure 17. Each image has four distinct camera trajectories, resulting in 128 test samples. Moreover, we collect a larger benchmark of 500 samples, comprising 50 scene images from megascenes [Tung et al. 2024] and 50 human scenes collected from the Internet. For each image, 5 random trajectories are assigned. We used VBench++ [Huang

- et al. 2024] to evaluate the video quality, while absolute translation error (ATE), relative translation error (RPE), and relative rotation error (RRE) are used to verify the camera precision. We utilize VGGT [Wang et al. 2025b] to produce extrinsic cameras for the generated images, which are then evaluated against the predefined cameras after trajectory alignment.

Analysis. We present the quantitative results in Table 1, comparing our model with ViewCrafter [Yu et al. 2024], SEVA [Zhou

- et al. 2025a], and the CogVideoX [Yang et al. 2025] version of our framework. The qualitative outcomes are shown in Figure 7. Our experiments demonstrate that point clouds significantly enhance the controllability of both the Wan2.1 and CogVideoX, as verified by the improvement of ATE, RPE, and RRE. Although SEVA achieves precise camera trajectories, it requires massive training with static multi-view data (0.8M iterations), struggling to handle dynamic outof-distribution scenarios, such as humans and animals, as illustrated in Figure 7. We should clarify that our baseline, Wan-I2V with only Plücker ray, suffers from inferior camera movements. While this setting achieves a strong VBench overall score, it compromises with poor camera metrics. Overall, the proposed PCDController achieves the optimal balance between video quality and camera precision. By integrating both Plücker rays and point clouds, it further enhances the performance in challenging scenes featuring substantial viewpoint changes, as validated in Table 1 and Figure 8. Furthermore, our method also performs well in the large benchmark on megascene and Internet human images as listed in Table 2.

5.3 Results of Unified Camera and Human Motion Control

Benchmark. We have developed a new benchmark of 50 videos of 720p, each featuring a person performing challenging motions. For guidance, SMPL-X is extracted using GVHMR [Shen et al. 2024]. Each video is assigned three different types of camera trajectories, resulting in a total of 150 test cases. To ensure that the person remains within the camera’s viewpoint, we employ a follow shooting technique for all test cases, adjusting the camera’s position based

1https://github.com/black-forest-labs/flux

on the movement of the SMPL-X center. These noisy and subtle movements further increase the difficulty of camera control. We follow the same camera metrics as mentioned in Section 5.2. To facilitate generalization for motion transfer, RealisDance-DiT is not specifically designed to perfectly recover the first frame aligned with the reference image. Consequently, we remove the metrics that heavily depend on the reference view (I2V subject and background) in Table 3 for fairness. Note that our method performs well in 720p generation, even though it was trained using 480p data.

Analysis. We show quantitative results in Table 3, while qualitative results are displayed in Figure 6 and Figure 9. To our knowledge, there are currently very few available models that effectively address the challenge of unified camera and human motion controls. We begin by comparing to CamAnimate [Wang et al. 2024a], which enables simultaneous control of both conditions. However, due to the misalignment of body poses extracted relative to predefined camera trajectories, the results of CamAnimate suffer from conflicting human motions and backgrounds. Moreover, we evaluate Uni3C against the baseline of RealisDance-DiT [Zhou et al. 2025b] and various ablation versions of our model. Notably, while RealisDance-DiT, which focuses solely on human control, achieves the best visual quality, it struggles to produce accurate camera trajectories, resulting in poor camera metrics. In contrast, Uni3C shows good VBench scores alongside impressive camera metrics. Note that the proposed PCDController can also be generalized to the T2V model with comparable quality, featuring the robust generalization of PCDController. Moreover, an interesting insight is revealed from Figure 9 and Table 3 that the aligned SMPL-X characters can further strengthen the camera controllability, while the PCDController trained with I2V formulation also enhances the visual quality and consistency of RealisDance-DiT. This illustrates the complementary features of these two components. Additionally, we show that the proposed Uni3C enables control of detailed hand motions under various camera trajectories as in Figure 10.

5.4 Ablation Study and Exploratory Discussions

Plücker Ray vs Point Clouds. As shown in Table 1, point clouds enjoy significantly more precise camera trajectories. We further clarify that video camera control trained with point clouds achieves much faster convergence with lower training loss, as illustrated in Figure 13. Only 1,000 training iterations can hold the general camera trajectory. We empirically find that timing large-scale VDM like Wan-I2V through Plücker ray is difficult. Maybe enabling more trainable parameters would improve the performance, potentially hindering the generalization, which is not considered in this work. Moreover, as verified in the last two rows of Table 1, using both Plücker ray and point clouds improves the results of very challenging camera trajectories.

Point Clouds of Humans. Human point clouds are always frozen in world space without any motion, which conflicts with the human motion conditions provided by SMPL-X. Eliminating this “redundant” information is a straightforward idea to improve motion quality. However, as verified in Table 3, while Uni3C-T2V with human-masked point clouds achieves slightly better visual quality,

- Table 1. Quantitative results of camera control. VBench++ scores (%) are normalized (higher is better). Injected camera features are divided as Plücker ray and point clouds (Pcd). † denotes the results with challenging 360◦ camera rotations. Results of MotionCtrl and CameraCtrl are tested with the I2V version re-trained by [Zheng et al. 2025a]. Methods in bold are the final setting of PCDController.

Camera Overall Subject Bg Aesthetic Imaging Temporal Motion I2V I2V Plücker Pcd Score Consist Consist Quality Quality Flicker Smooth Subject Bg ATE↓ RPE↓ RRE↓

MotionCtrl [Wang et al. 2024b] 82.48 89.42 91.64 54.99 55.02 91.93 95.85 89.92 91.11 0.345 0.263 2.547 CameraCtrl [He et al. 2025a] ✓ 83.69 90.17 92.21 55.71 54.13 93.32 96.65 93.34 94.01 0.354 0.239 2.306 CamI2V [Zheng et al. 2025a] ✓ 86.52 91.45 92.65 60.91 66.22 93.34 97.01 95.17 95.45 0.322 0.247 1.653 ViewCrafter [Yu et al. 2024] ✓ 85.39 89.69 91.68 55.13 64.33 92.94 97.66 95.59 96.11 0.210 0.117 0.873 SEVA [Zhou et al. 2025a] ✓ 87.39 91.86 93.36 56.79 68.43 95.74 98.59 97.08 97.23 0.077 0.029 0.223

- Ours (CogVideoX) ✓ 86.48 93.17 93.02 55.00 66.75 95.10 98.36 94.45 95.94 0.356 0.162 1.280

- Ours (CogVideoX) ✓ 87.22 91.26 92.44 56.90 69.53 94.79 98.47 96.60 97.79 0.123 0.045 0.346 Ours (Wan-I2V) ✓ 89.16 94.71 94.93 60.42 72.20 96.51 98.51 97.74 98.29 0.402 0.095 0.728 Ours (Wan-I2V) ✓ 87.95 91.71 92.97 58.52 71.12 95.51 98.55 97.24 97.96 0.091 0.028 0.211 Ours (Wan-I2V) ✓ ✓ 88.27 92.20 93.37 58.99 71.96 95.56 98.66 97.38 98.01 0.102 0.031 0.246

Ours (Wan-I2V)† ✓ 82.70 82.16 88.72 53.67 66.95 91.41 95.45 90.80 92.47 1.327 0.551 6.334 Ours (Wan-I2V)† ✓ ✓ 82.82 82.31 88.75 53.56 66.25 90.81 95.23 92.12 93.49 1.010 0.416 4.428

[Figure 92]

[Figure 93]

Uni3C(I2V)CamAnimateUni3C(T2V)ConditionPCDControllerRealisDanceDiT

Fig. 6. Results of unified camera and human motion controls. Leftmost images are reference views; the first row indicates aligned 3D world guidance.

- Table 2. Camera control results on large benchmark with 500 samples collected from megascenes [Tung et al. 2024] and Internet human scenes. ‘Overall’ means the average score across all metrics of VBench++.

such as animation and real-world scenes. Meanwhile, Uni3C can be further extended to generate vivid videos based on other conditions, like text-to-motion guidance or retrieved motions from motion databases. To prove this point, we randomly integrate several motion clips generated text-to-motion [Barquero et al. 2024] trained on BABEL [Punnakkal et al. 2021] and use Uni3C to control both motion and camera, as illustrated in Figure 12. More results are shown in our supplementary.

Overall↑ ATE↓ RPE↓ RRE↓

CamI2V [Zheng et al. 2025a] 83.62 0.8081 0.6858 2.5803 ViewCrafter [Yu et al. 2024] 84.26 0.6224 0.4098 1.2642 SEVA [Zhou et al. 2025a] 84.83 0.2473 0.0832 0.5329 PCDController 86.62 0.2613 0.0828 0.4612

this masking adversely affects camera precision, particularly when humans occupy a significant image area. Therefore, retaining the point clouds of humans is essential for effective camera control. Given that Wan2.1 [Wang et al. 2025a] is trained on videos featuring substantial motion, it can generate natural and smooth movements even when the foreground point clouds remain fixed.

Gravity Calibration. As mentioned in Section 4.3, gravity calibration is critical for aligning the global 3D world space. Results shown in Figure 14 verify that the calibration can correct the SMPL-X characters aligned with skewed human point clouds and eliminate the error accumulation for humans’ long-distance movements.

Motion Transfer. We present motion transfer results achieved by the Uni3C framework in Figure 11. Our model effectively controls both camera trajectories and human motions, even when reference motions are sourced from different videos or distinct domains,

Ablation of PCDController. We present some ablation results of PCDController with various external DiT layers in Table 4. Each setting injects conditional features into the main Wan-I2V backbone from top to bottom with different layers. While the DiT branch with 30 layers shows slightly improved camera metrics, it ultimately compromises overall visual quality and generalization. We reveal that solely applying more external layers suffers from overfitting point cloud conditions as shown in Figure 15, resulting in suboptimal outcomes with distorted point clouds. Thus, we determine that employing 20 DiT layers achieves an effective balance between controllability and generalization within the PCDController.

5.5 User Study

We conducted user studies for camera control (128 samples) and unified control (150 samples) on our respective benchmarks in Table 5. Formally, 20 unrelated volunteers are invited. For the camera

- Table 3. Quantitative results of unified camera and human motion controls. “Aligned” control means re-rendering human conditions under new camera trajectories in the environmental world space. † denotes masking out the foreground point clouds of humans. Method in bold is the final setting of Uni3C.

Control Overall Subject Bg Aesthetic Imaging Temporal Motion Camera Human Aligned Score Consist Consist Quality Quality Flicker Smooth ATE↓ RPE↓ RRE↓

CamAnimate [Wang et al. 2024a] ✓ ✓ 82.45 89.20 90.52 57.42 67.40 93.94 96.20 0.619 0.419 2.035 RealisDance-DiT [Zhou et al. 2025b] ✓ ✓ 85.21 93.03 95.34 57.89 68.71 97.44 98.82 0.549 0.195 0.547 PCDController ✓ 83.19 89.08 91.63 57.23 68.27 95.22 97.71 0.256 0.092 0.661 Uni3C (T2V)† ✓ ✓ ✓ 83.34 88.45 91.45 57.45 69.84 95.21 97.63 0.296 0.098 1.167 Uni3C (T2V) ✓ ✓ ✓ 83.16 88.67 91.38 56.79 69.42 95.14 97.57 0.262 0.083 0.606 Uni3C (I2V, unaligned) ✓ ✓ 82.98 89.16 92.40 56.02 66.78 95.65 97.84 0.270 0.111 0.639 Uni3C (I2V) ✓ ✓ ✓ 83.43 89.45 93.05 57.25 67.28 95.70 97.86 0.251 0.093 0.490

- Table 4. Ablation of PCDController. Different DiT layers are employed from top to bottom. PCDController adopts 20-layer as the final setting.

External DiT Overall Layer Param Score ATE↓ RPE↓ RRE↓

10 0.48B 87.75 0.136 0.0398 0.559 20 0.95B 88.27 0.102 0.0313 0.246 30 1.39B 88.08 0.092 0.0302 0.197 40 1.85B 87.90 0.115 0.0339 0.216

- Table 5. User studies on camera and unified (camera&human) control. Camera CamI2V ViewCrafter SEVA PCDController VideoQuality↑ 0.0391 0.0855 0.2328 0.6457 CameraPrecision↑ 0.0359 0.1003 0.2429 0.6281 Camera&Human CamAnimate RealisDance-DiT PCDController Uni3C VideoQuality↑ 0.0170 0.2466 0.0453 0.6920 CameraPrecision↑ 0.0676 0.2766 0.5266 0.8976 HumanPose↑ 0.0716 0.5863 0.1730 0.8983

control, volunteers are requested to vote for the best result, considering video quality and camera precision, while the quality of human pose is additionally required for the unified control. If two methods perform very similarly, volunteers are allowed to vote for both of them as top-2. We finally average the score for each metric.

- 6 Limitation and Future Work

Although Uni3C supports flexible and diverse unified control and motion transfer, it operates under the constraints of predefined camera trajectories and human characters (SMPL-X). Consequently, Uni3C struggles to produce physically plausible outcomes when human motions conflict with environmental conditions, as shown in Figure 16. For instance, if a human’s movement trajectory is blocked by walls, barriers, or other objects, the generated results may exhibit artifacts such as distortion, clipping, or sudden disappearance. This limitation could be mitigated by employing a more advanced human motion generation method that accounts for physical obstructions within the environment. Besides, though our method can produce plausible results within inaccurate SMPL or world coordinates, it will result in undesired pose or motion direction.

- 7 Conclusion

efficiently manage camera trajectories without compromising the inherent capacities of foundational VDMs. This not only enhances generalization but also facilitates versatile downstream applications without joint training. Furthermore, by aligning multi-modal conditions, including both environmental point clouds and human characters in a global 3D world space, we established a coherent framework for jointly controlling camera movements and human animations. Our comprehensive experiments validated the efficacy of Uni3C across diverse datasets, showcasing superior performance in both quantitative and qualitative assessments compared to existing approaches. The significance of our contributions lies not only in improving the state-of-the-art in controllable video generation but also in proposing a robust way to inject multi-modal conditions without the requirements of heavily annotated data. We believe that Uni3C paves the way for advanced controllable video generation.

Acknowledgments

This work was supported by the Science and Technology Commission of Shanghai Municipality (No. 24511103100).

This paper introduced Uni3C, a framework that unifies 3D-enhanced camera and human motion controls for video generation. We first propose the PCDController, demonstrating that lightweight, trainable modules, and rich geometric priors from 3D point clouds can

[Figure 94]

Viewcrafter

[Figure 95]

SEVA

Ours (Plücker only)

Ours (full)

[Figure 96]

[Figure 97]

Viewcrafter

SEVA

Ours (Plücker only)

Ours (full)

[Figure 98]

[Figure 99]

Viewcrafter

SEVA

Ours (Plücker only)

Ours (full)

[Figure 100]

[Figure 101]

Viewcrafter

SEVA

Ours (Plücker only)

Ours (full)

- Fig. 7. Qualitative results of camera control on our benchmark. We compare the proposed PCDController to ViewCrafter [Yu et al. 2024], SEVA [Zhou et al. 2025a], and our model without point cloud guidance. The leftmost image is the reference condition. “full” indicates using both Plücker ray and point clouds as conditions.

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

- Fig. 8. Results of the challenging orbital 360◦ rotations from PCDController. The leftmost images are the reference views.

[Figure 113]

Condition

CamAnimate

RealisDance-DiT

PCDController

Uni3C (T2V)

Uni3C (I2V)

[Figure 114]

Condition

CamAnimate

RealisDance-DiT

PCDController

Uni3C (T2V)

Uni3C (I2V)

[Figure 115]

Condition

CamAnimate

RealisDance-DiT

PCDController

Uni3C (T2V)

Uni3C (I2V)

- Fig. 9. More results of unified camera and human motion controls. Leftmost images are reference views; the first row indicates aligned 3D world guidance.

RealisDance-DiTConditionUni3C(T2V)Uni3C(I2V)HamerPCDController

[Figure 116]

- Fig. 10. Results of unified camera, human motion, and Hamer controls. The leftmost images are the reference views, while the first and second rows indicate the aligned 3D world guidance and Hamer rendering.

[Figure 117]

- ReferenceReference
- Fig. 11. Results of motion transfer. The first row indicates the reference video, while others show our generated videos transferring motions from the reference sequence.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

- Fig. 12. Results transferred from randomly integrated motion clips generated from text-to-motion [Barquero et al. 2024] trained on BABEL [Punnakkal et al. 2021]. The motion sequences are listed on the left, which are executed from light to dark colors.

GTPlückerPointcloud

[Figure 124]

(a) Training losses of different camera conditions (b) Results of 1,000 iterations with different camera conditions

[Figure 125]

[Figure 126]

[Figure 127]

- Fig. 13. Ablation results of Plücker ray and point clouds during training phase. Point clouds enjoy highly accurate camera control against Plücker ray.

[Figure 128]

[Figure 129]

NoCalib.WithCalib.NoCalib.WithCalib.

[Figure 130]

[Figure 131]

[Figure 132]

#### Fig. 14. Rendering results with and without gravity calibration by GeoCalib [Veicht et al. 2024].

[Figure 133]

Layer30ConditionLayer20

[Figure 134]

##### Fig. 15. Camera control results of PCDController with challenging point clouds. The model with fewer external DiT layers enjoys superior conditional robustness compared to the one with more DiT layers.

[Figure 135]

##### Fig. 16. Failed cases generated by Uni3C. These results are primarily limited by the conflict between human motions and environments.

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

##### Fig. 17. Our out-of-distribution benchmark for camera control. The validation set includes generative, human, scene-level, and object-level images with diverse aspect ratios.

References

Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo Garcia-Hernando, Aron Monszpart, Victor Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann. 2022. Map-free visual relocalization: Metric pose relative to a single image. In European Conference on Computer Vision. Springer, 690–708.

Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. 2025a. AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Sherwin Bahmani, Ivan Skorokhodov, Aliaksandr Siarohin, Willi Menapace, Guocheng Qian, Michael Vasilkovsky, Hsin-Ying Lee, Chaoyang Wang, Jiaxu Zou, Andrea Tagliasacchi, et al. 2025b. Vd3d: Taming large video diffusion transformers for 3d camera control. In International Conference on Learning Representations.

German Barquero, Sergio Escalera, and Cristina Palmero. 2024. Seamless Human Motion Composition with Blended Positional Encodings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman.

2022. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5470–5479. Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127 (2023).

Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. 2024. Depth Pro: Sharp Monocular Metric Depth in Less Than a Second. arXiv (2024). https://arxiv.org/abs/2410.02073

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. 2024. Video generation models as world simulators. (2024). https: //openai.com/research/video-generation-models-as-world-simulators

Chenjie Cao, Xinlin Ren, and Yanwei Fu. 2024. MVSFormer++: Revealing the Devil in Transformer’s Details for Multi-View Stereo. In International Conference on Learning Representations.

Chenjie Cao, Chaohui Yu, Shang Liu, Fan Wang, Xiangyang Xue, and Yanwei Fu. 2025. MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yingjie Chen, Yifang Men, Yuan Yao, Miaomiao Cui, and Liefeng Bo. 2025. Perceptionas-Control: Fine-grained Controllable Image Animation with 3D-aware Motion Representation. arXiv preprint arXiv:2501.05020 (2025).

Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. 2023. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151 (2023). Wanquan Feng, Jiawei Liu, Pengqi Tu, Tianhao Qi, Mingzhen Sun, Tianxiang Ma, Songtao Zhao, Siyu Zhou, and Qian He. 2025. I2VControl-Camera: Precise Video Camera Control with Adjustable Motion Strength. In International Conference on Learning Representations.

Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, Chen Sun, Oliver Wang, Andrew Owens, and Deqing Sun. 2025. Motion Prompting: Controlling Video Generation with Motion Trajectories. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.

Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, et al. 2025. Diffusion as Shader: 3D-aware Video Diffusion for Versatile Video Generation Control. arXiv preprint arXiv:2501.03847 (2025). Chuan Guo, Yuxuan Mu, Muhammad Gohar Javed, Sen Wang, and Li Cheng. 2024. Momask: Generative masked modeling of 3d human motions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1900–1910.

Chuan Guo, Shihao Zou, Xinxin Zuo, Sen Wang, Wei Ji, Xingyu Li, and Li Cheng. 2022. Generating diverse and natural 3d human motions from text. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5152–5161.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2025a. Cameractrl: Enabling camera control for text-to-video generation. In International Conference on Learning Representations.

Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, and Hongsheng Li. 2025b. CameraCtrl II: Dynamic Scene Exploration via Camera-controlled Video Diffusion Models. arXiv preprint arXiv:2503.10592 (2025).

Chen Hou, Guoqiang Wei, Yan Zeng, and Zhibo Chen. 2025. Training-free camera control for video generation. In International Conference on Learning Representations.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Li Hu. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8153–8163.

Li Hu, Guangyuan Wang, Zhen Shen, Xin Gao, Dechao Meng, Lian Zhuo, Peng Zhang, Bang Zhang, and Liefeng Bo. 2025. Animate Anyone 2: High-Fidelity Character Image Animation with Environment Affordance. arXiv preprint arXiv:2502.06145 (2025).

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al. 2024. Vbench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503 (2024).

Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. 2023. Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems 36 (2023), 20067–20079.

Xuekun Jiang, Anyi Rao, Jingbo Wang, Dahua Lin, and Bo Dai. 2024. Cinematic behavior transfer via nerf-based differentiable filming. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6723–6732.

Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG) 36, 4 (2017), 1–13.

Muhammed Kocabas, Ye Yuan, Pavlo Molchanov, Yunrong Guo, Michael J Black, Otmar Hilliges, Jan Kautz, and Umar Iqbal. 2024. PACE: Human and Camera Motion Estimation from in-the-wild Videos. In 2024 International Conference on 3D Vision (3DV). IEEE, 397–408.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

Kuaishou. 2024. Kling. https://klingai.kuaishou.com (2024). Teng Li, Guangcong Zheng, Rui Jiang, Tao Wu, Yehao Lu, Yining Lin, Xi Li, et al. 2025b.

RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control. arXiv preprint arXiv:2502.10059 (2025).

Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. 2025a. Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Hanwen Liang, Junli Cao, Vidit Goel, Guocheng Qian, Sergei Korolev, Demetri Terzopoulos, Konstantinos N Plataniotis, Sergey Tulyakov, and Jian Ren. 2024. Wonderland: Navigating 3D Scenes from a Single Image. arXiv preprint arXiv:2412.12091 (2024).

Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. 2023. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems 36 (2023), 25268–25280.

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. 2024. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22160–22169.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022). Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. 2021. Infinite nature: Perpetual view generation of natural scenes from a single image. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 14458–14467.

Xingchen Liu, Piyush Tayal, Jianyuan Wang, Jesus Zarzar, Tom Monnier, Konstantinos Tertikas, Jiali Duan, Antoine Toisoul, Jason Y Zhang, Natalia Neverova, et al. 2025. UnCommon Objects in 3D. arXiv preprint arXiv:2501.07574 (2025).

Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. 2019. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10975–10985.

Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. 2024. Reconstructing Hands in 3D with Transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024. 9826– 9836.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205. Matthias Plappert, Christian Mandery, and Tamim Asfour. 2016. The kit motionlanguage dataset. Big data 4, 4 (2016), 236–252.

Stefan Popov, Amit Raj, Michael Krainin, Yuanzhen Li, William T Freeman, and Michael Rubinstein. 2025. CamCtrl3D: Single-Image Scene Exploration with Precise 3D Camera Control. arXiv preprint arXiv:2501.06006 (2025).

Abhinanda R Punnakkal, Arjun Chandrasekaran, Nikos Athanasiou, Alejandra QuirosRamirez, and Michael J Black. 2021. BABEL: Bodies, action and behavior with english labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 722–731.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.

Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. 2021. Common Objects in 3D: Large-Scale Learning and

Evaluation of Real-life 3D Category Reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin NimierDavid, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. 2025. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 10684–10695.

RunwayML. 2024. Gen-3 Alpha. https://runwayml.com/research/ introducing-gen-3-alpha

(2024). Johannes Lutz Schönberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm.

2016. Pixelwise View Selection for Unstructured Multi-View Stereo. In European conference on computer vision.

Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng, Zechen Hu, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou. 2024. World-Grounded Human Motion Recovery via GravityView Coordinates. In SIGGRAPH Asia 2024 Conference Papers. 1–11.

Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ignacio Rocco, Natalia Neverova, Andrea Vedaldi, and David Novotny. 2023. Common pets in 3d: Dynamic new-view synthesis of real-life deformable categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4881–4891.

Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. 2024. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928 (2024).

Shuai Tan, Biao Gong, Xiang Wang, Shiwei Zhang, Dandan Zheng, Ruobing Zheng, Kecheng Zheng, Jingdong Chen, and Ming Yang. 2025. Animate-x: Universal character image animation with enhanced motion representation. In International Conference on Learning Representations.

Joseph Tung, Gene Chou, Ruojin Cai, Guandao Yang, Kai Zhang, Gordon Wetzstein, Bharath Hariharan, and Noah Snavely. 2024. Megascenes: Scene-level view synthesis at scale. In European conference on computer vision. Springer, 197–214.

Shinji Umeyama. 1991. Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis & Machine Intelligence 13, 04 (1991), 376–380.

Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. 2024. GeoCalib: Learning Single-image Calibration with Geometric Optimization. In European Conference on Computer Vision. Springer, 1–20.

Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. 2025a. Wan: Open and Advanced LargeScale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. 2025b. VGGT: Visual Geometry Grounded Transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.

Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. 2020. TartanAir: A Dataset to Push the Limits of Visual SLAM. (2020).

Zhenzhi Wang, Yixuan Li, Yanhong Zeng, Youqing Fang, Yuwei Guo, Wenran Liu, Jing Tan, Kai Chen, Tianfan Xue, Bo Dai, et al. 2024a. Humanvid: Demystifying training data for camera-controllable human image animation. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024b. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers. 1–11.

Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. 2024. RGBD objects in the wild: scaling real-world 3D object learning from RGB-D videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22378–22389. Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, and Kai Zhang. 2024a. DMV3D: Denoising Multi-View Diffusion using 3D Large Reconstruction Model. In International Conference on Learning Representations.

Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. 2023. Vitpose++: Vision transformer for generic body pose estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence 46, 2 (2023), 1212–1230.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2024b. MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model. In IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1481–1490.

Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. 2024. Direct-a-video: Customized video generation with user-directed camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers. 1–12.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2025. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations.

Meng You, Zhiyu Zhu, Hui Liu, and Junhui Hou. 2025. Nvs-solver: Video diffusion model as zero-shot novel view synthesizer. In International Conference on Learning Representations.

Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. 2024. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024).

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 3836–3847.

Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. 2022. ParticleSfM: Exploiting Dense Point Trajectories for Localizing Moving Cameras in the Wild. In European conference on computer vision (ECCV).

Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. 2025a. Cami2v: Camera-controlled image-to-video diffusion model. In International Conference on Learning Representations.

Sixiao Zheng, Zimian Peng, Yanpeng Zhou, Yi Zhu, Hang Xu, Xiangru Huang, and Yanwei Fu. 2025b. VidCRAFT3: Camera, Object, and Lighting Control for Image-toVideo Generation. arXiv preprint arXiv:2502.07531 (2025).

Jingkai Zhou, Benzhi Wang, Weihua Chen, Jingqi Bai, Dongyang Li, Aixi Zhang, Hao Xu, Mingyang Yang, and Fan Wang. 2024. RealisDance: Equip controllable character animation with realistic hands. arXiv preprint arXiv:2409.06202 (2024).

Jingkai Zhou, Yifan Wu, Shikai Li, Min Wei, Chao Fan, Weihua Chen, Wei Jiang, and Fan Wang. 2025b. RealisDance-DiT: Simple yet Strong Baseline towards Controllable Character Animation in the Wild. arXiv preprint arXiv:2504.14977 (2025).

Jensen Jinghao Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. 2025a. STABLE VIRTUAL CAMERA: Generative View Synthesis with Diffusion Models. arXiv e-prints (2025), arXiv–2503.

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. Stereo magnification: learning view synthesis using multiplane images. ACM Trans. Graph. 37, 4, Article 65 (July 2018), 12 pages. doi:10.1145/3197517.3201323

Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Zilong Dong, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. 2024. Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance. In European Conference on Computer Vision, Vol. 15113. 145–162.

- A Border Impact

This study delves into the realm of controllable video generation. The remarkable generative capabilities of AI-generated content (AIGC) can inadvertently lead to the creation of misleading information or fabricated visuals. Consequently, we sincerely urge users to remain vigilant regarding these issues. Moreover, issues of privacy and consent must be taken into account, as generative models are always developed using extensive datasets. It’s also crucial to acknowledge that such models can potentially reinforce biases in the training data, which may result in unjust outcomes. Therefore, we advise users to be responsible and inclusive when utilizing these generative models. It is important to mention that our method concentrates solely on technical aspects. All pre-trained models and training videos referenced in this study are publicly available.

- B Details of Uni3C B.1 Architecture Details of PCDController

As mentioned in the main paper, our PCDController is built upon a simplified DiT module. We apply detailed ablation studies and demonstrate that the 20-layer external DiT branch with 1024 hidden size enjoys a good balance between both controllability and generalization. Detailed architecture of PCDController is illustrated in Figure 18. We follow the DiT design in CogVideoX [Yang et al. 2025], including AdaLayerNorm, 3D attention, and Feed Forward Network (FFN), without the textual branch. The output features are added to the main branch of Wan-I2V block by block via zero-initialized projection layers.

Table 6. Inference efficiency of Uni3C. Our model only increases a little inference time compared to the baseline method.

…

[Figure 168]

ExDiT Block i

Inference Times 480×768 720×1280

Methods

AdaLayerNorm

RealisDance-DiT 50.5s 182.7s Uni3C 62.3s (+23%) 213.4s (+17%)

3D Full Attention

Table 7. Efficiency of other components of Uni3C. PyTorch3D-Rendering 2D-Keypoint+Alignment GeoCalib

Gated Addition

AdaLayerNorm

4.341s 0.577s 1.459s

Feed Forward

…

- B.3 High-Resolution Inference

While PCDController is trained under 480p multi-view images (480× 768, 512 × 720, 608 × 608, 720 × 512, 768 × 480), we find that this model also performs well under 720p video generation (720 × 1280, 800 × 1152, 960 × 960, 1152 × 800, 1280 × 720) without specific fine-tuning, as empirically verified in our qualitative results in the supplementary. Specifically, for the inference of 720p, we first render the conditional videos of warped point clouds with PyTorch3D under 480p. Then we replace the first frame with a 720p reference image as the high-resolution guidance. Thus, PCDController can produce high-quality 720p videos extended from the high-resolution reference, while the low-resolution point cloud renderings are only considered as auxiliary camera signals.

- B.4 Inference Efficiency

[Figure 169]

Gated Addition

Wan DiT Block i

Zero-initialized Linear

[Figure 170]

[Figure 171]

Wan DiT Block i+1

ExDiT Block i+1

…

…

Fig. 18. Detailed model architecture of PCDController.

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

We evaluate the inference efficiency of Uni3C in comparison to the baseline model, RealisDance-DiT [Zhou et al. 2025b], across two different resolutions, as detailed in Table 6. Note that the efficiency of RealisDance-DiT [Zhou et al. 2025b] is almost the same as the basic VDM, Wan-I2V [Wang et al. 2025a], while only a few convolution layers are newly incorporated to encode additional conditions. For this analysis, we utilize the official settings from Wan2.1 [Wang et al. 2025a], which include a 40-step denoising process and a classifierfree guidance scale of 5. The inference environment is configured using a sequence parallel based on 8 Nvidia H100 GPUs. We further provide other detailed time costs of Uni3C in Table 7. Overall, Uni3C demonstrates impressive efficiency in this setup.

Reference image

SMPL-X Hamer

[Figure 179]

[Figure 180]

[Figure 181]

Noise Patchfier

Ref Patchfier

Pose Patchfier

[Figure 182]

Zero Proj

C

[Figure 183]

Self-Attention Block

TextEmbedding

[Figure 184]

×N

Cross-Attention Block

C Details of Datasets

[Figure 185]

FeedForward Layer

We summarized our training data setting in Table 8. A sampling strategy across epochs ensures balanced sample scales over different data domains. Formally, we pay more attention to the data with high-quality images and complex camera trajectories of real-world scenarios like DL3DV [Ling et al. 2024] and UCo3D [Liu et al. 2025].

- Fig. 19. Detailed architecture of Realisdance-DiT [Zhou et al. 2025b].

B.2 Architecture Details of Realisdance-DiT

We show the model architecture of Realisdance-DiT [Zhou et al. 2025b] in Figure 19 employed for our unified controls for both camera and human poses. Formally, Realisdance-DiT is fine-tuned from the pre-trained Wan-I2V, while features of human condition signals (SMPL-X and Hamer) are added to the inputs, and the reference image feature is spatially concatenated to the inputs to confirm fine-grained details. During the training of Realisdance-DiT, only convolution-based pose and reference patchfiers and self-attention layers of DiT are trainable.

D Inference Details of Camera Control

During the inference phase, we begin by extracting the monocular depth of the reference view using Depth-Pro [Bochkovskii et al. 2024]. We then establish the foreground depth medium, defined as the rotation radius, to determine the placement of the initial camera. For extracting the foreground mask, we utilize CarveKit 2. In cases

2https://github.com/OPHoperHPO/image-background-remove-tool

[Figure 186]

Uni3Cresult1Uni3Cresult2Inputvideo

- Fig. 20. Results of Uni3C under the two-person scenario. We operate two different camera trajectories, while their human motions follow the input video.

Table 8. Dataset details of training PCDController. The training datasets include Co3Dv2 [Reizenstein et al. 2021], DL3DV [Ling et al. 2024], RE10K [Zhou et al. 2018], ACID [Liu et al. 2021], Tartainair [Wang et al. 2020], Map-Free-Reloc [Arnold et al. 2022], WildRGBD [Xia et al. 2024], COP3D [Sinha et al. 2023], and UCo3D [Liu et al. 2025]. We dynamically sample subsets for each dataset across training epochs.

Scene Number Train Validation Epoch

Indoor Outdoor Object Synthetic

Co3Dv2 ✓ ✓ ✓ 24,437 53 2,252 DL3DV ✓ ✓ 9,808 250 19,868 RE10K ✓ ✓ 20,259 50 7,528 ACID ✓ 2,575 20 1,047 Tartainair ✓ ✓ ✓ 2,834 50 2,834 Map-Free-Reloc ✓ ✓ 892 18 1,784 WildRGBD ✓ ✓ ✓ 22,105 46 2,300 COP3D ✓ ✓ 2,109 20 2,109 UCo3D ✓ ✓ ✓ 161,591 99 10,000

where no foreground is detected, the entire image is treated as the foreground. Our camera control is based on two primitives:

- • Rotation. We define the rotation along azimuth and elevation, respectively. The rotation radius is used to control the distance to the foreground, serving as the center of rotation.
- • Translation: We implement translation along the x, y, and z axes. The translation values are constrained within the range

of [0, 1.0], which are then multiplied by the rotation radius to ensure that translations do not become excessively large or small.

Overall, the camera control mechanism proposed in this paper is flexible and effective enough to handle most downstream tasks.

E Multiple-Person Discussion

Uni3C builds upon RealisDance-DiT [Zhou et al. 2025b], which is explicitly tailored for single-person scenarios. Notably, we observe that RealisDance-DiT retains functionality even in two-person scenarios, as verified by the case presented in [Zhou et al. 2025b] and visualized in Figure 20. Formally, we separately align two persons’ SMPL-X representations, guided by their respective 3D keypoints. However, injecting additional control signals from more people would hinder the performance. This outcome is unsurprising, as multi-person control is significantly outside the scope of RealisDance-DiT training. As a plug-and-play module, Uni3C can be integrated with other backbones that are natively designed for addressing multi-person control in the future.

