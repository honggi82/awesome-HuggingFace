# arXiv:2503.23368v3[cs.CV]4Apr2025

## VLIPP: Towards Physically Plausible Video Generation with Vision and Language Informed Physical Prior

Xindi Yang1*, Baolu Li2*, Yiming Zhang2, Zhenfei Yin4,5†, Lei Bai3†, Liqian Ma6, Zhiyong Wang5, Jianfei Cai1, Tien-Tsin Wong1, Huchuan Lu2, Xu Jia2†

1Monash University 2Dalian University of Technology 3Shanghai Artificial Intelligence Laboratory 4Oxford University 5The University of Sydney 6ZMO AI

#### Abstract

Video diffusion models (VDMs) have advanced significantly in recent years, enabling the generation of highly realistic videos and drawing the attention of the community in their potential as world simulators. However, despite their capabilities, VDMs often fail to produce physically plausible videos due to an inherent lack of understanding of physics, resulting in incorrect dynamics and event sequences. To address this limitation, we propose a novel two-stage image-to-video generation framework that explicitly incorporates physics with vision and language informed physical prior. In the first stage, we employ a Vision Language Model (VLM) as a coarse-grained motion planner, integrating chain-of-thought and physics-aware reasoning to predict a rough motion trajectories/changes that approximate real-world physical dynamics while ensuring the inter-frame consistency. In the second stage, we use the predicted motion trajectories/changes to guide the video generation of a VDM. As the predicted motion trajectories/changes are rough, noise is added during inference to provide freedom to the VDM in generating motion with more fine details. Extensive experimental results demonstrate that our framework can produce physically plausible motion, and comparative evaluations highlight the notable superiority of our approach over existing methods. More video results are available on our Project Page: https://madaoer.github.io/projects/ physically_plausible_video_generation/.

#### 1. Introduction

Video diffusion models (VDMs) trained on large-scale video datasets have made remarkable progress in terms of realism, demonstrating significant potential for various content creation applications. Despite the absence of explicit geometric modeling, the generated videos still exhibit co-

∗ Equal contribution † Corresponding author

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Gen-3OursKling1.6OursLumaOurs

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Two small balls on the table move towards each other and collide.

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

Clean water is poured into a transparent glass.

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

A basketball fell onto the clean floor.

Figure 1. Existing commercial closed-source VDMs fail to generate physically plausible motion, whereas our video generation framework is able to achieve this by incorporating external physical prior knowledge.

herent spatial relationships among objects, rich textured details, and realistic lighting effects, including reflections and shadows. Such qualities often make the generated videos nearly indistinguishable from real-world footages. This drives the research community to explore the potential of VDMs as world models. However, they still struggle with understanding the physical laws of the real world and generating videos that adhere to these principles.

Although existing VDMs can produce visually realistic videos, they fail to mimic the real-dynamic physical motions. As shown in Fig 1, even the current commercial closed-source VDMs struggle with the task of generating videos that conform to physical laws. PhyT2V [58] refines text prompts by incorporating detailed descriptions of phys-

ical processes to guide VDMs in generating physically plausible videos. However, despite being pretrained on internetscale real-world video–text pairs, VDMs do not inherently understand physical laws. This limitation arises from the gap and ambiguity between text descriptions and the actual motion in the video[33]. Moreover, VDMs tend to overfit the training data rather than developing a general understanding of physical laws[21]. Inspired by the success of graphics-based physical rendering, some methods have guided VDMs to generate physically plausible videos using simulations from graphics engines[18, 29, 32, 56, 64]; however, these approaches rely on the physical effects that graphics engines can simulate and incur high computational costs.

The gap and ambiguity between text and real-world motion makes it difficult to enable physically plausible video generation through detailed text descriptions alone. Moreover, it is challenging to gather scalable physical data for training due to the abstractness and diversity of physical phenomena. Consequently, a viable approach could be to model abstract physical laws as conditions for diffusion models. However, it is less practical to explicitly model the physics equation for every kind of motion. Instead, we resort to current large foundation models for their ability to “understand” basic physics[6] and reason about physical phenomena based on the knowledge they extract. For example, given two colliding balls, the Large Language Model (LLM) can approximately predict the paths of the balls after collision. Inspired by this observation, we propose a novel video generation framework that employs a Vision Language Model (VLM) to predict the path/change during a physics event, described by a given image and a text prompt.

In this paper, we propose VLIPP, a two-stage approach to incorporate physics as conditions into VDM, enabling the generation of physically plausible motion. In the first stage, the VLM serves as a coarse-level motion planner, while a VDM serves as a fine-level motion synthesizer. The idea of stage one is to utilize the chain-of-thought and the physics-aware reasoning of VLM planning to ensure that coarse-level motion trajectories approximately follow realworld physics dynamics. In stage two, we can generate finelevel motion using an image-to-video diffusion model conditioned by the approximated path/change planned by VLM from stage one. Note that the approximated path/changes are not in the level to tell the speed or acceleration of the motion. We choose an existing image-to-video model [7] to accept our coarse-level path/change, by injecting noise to the motion path during both the training and inference phases. Notably, during the VLM planning stage, generating entire physically plausible motion trajectories is not required. Instead, we leverage the generative priors of VDM to produce fine-level physically plausible videos based on coarse-level motion trajectories provided by the VLM. So

that the detail-level motion such as speed, acceleration, and vibration are left to the VDM to synthesize.

We evaluate our physically plausible video generation framework with two major video physics benchmarks and achieved satisfactory results. Furthermore, we discuss and analyze multiple insightful design choices in our video generation framework, such as employing a motion planner tailored for different physics categories, and enhancing the robustness of diffusion model to noisy trajectories. Our contributions are summarized as follows:

- 1. We introduce a novel image-to-video generation framework for generating physically plausible videos by leveraging the VLM and VDM priors, significantly outperforming the contemporary competitors.
- 2. We propose a novel chain-of-thought and physics-aware reasoning approach in VLM, along with random noise injection in the latent space during video generation, which effectively improves both the generation quality and physical plausibility.
- 3. We conduct a comprehensive experiments and user studies to demonstrate the effectiveness and generalization of our framework in physically plausible videos generation.

#### 2. Related Work

###### 2.1. Physically Plausible Visual Content Generation

Generating physically plausible videos offers substantial value to real-world applications such as scientific simulations[43], robotics [4, 59], and autonomous driving [10, 48]. Traditional graphics pipelines rely on simulation systems to model physical phenomena [36, 42]. Inspired by these approaches, recent studies [18, 29] have performed dynamic simulations in image space based on physical engines. Furthermore, some methods [56, 64] incorporate physical priors into 3D representations to enable the synthesis of physically plausible motions. However, these rule-based or solver-based simulators face limitations in expressiveness, efficiency, generalizability, and parameter tuning. Furthermore, these simulators require significant expertise, rendering them inaccessible and unfriendly for users.

In addition, Some studies have explored VDMs for generating physically plausible videos. Li et al. [24] models natural oscillations and swaying in frequency-domain. A downstream rendering module then animates static images based on the generated motion information. PhysDiff [62] introduces physical simulator as constrain into the diffusion process by projecting denoised motion of a diffusion step into a physically plausible motion. These methods mainly focus only on specific types of physical motion and do not establish a generalizable approach for generating physically plausible videos.

Overview of Pipeline Stage1：Coarse-level Motion Planning

|[Figure 37]|
|---|
|[Figure 38]|
|[Figure 39]|
|[Figure 40]|
|[Figure 41]|

|[Figure 42]| |
|---|---|
|[Figure 43]<br><br>| |
|---|
<br><br>| |
|---|
| |

[Figure 44]

[Figure 45]

###### VLM

Grounding-SAM

First Frame

|[Figure 46]| |
|---|---|
| | |

|[Figure 47]|
|---|

MotionTrajectory

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

###### Chain of Thought

[Figure 52]

Coarse-level Motion Planner

Fine-level Motion Synthesis

Step1 Step2 Step3 ···

[Figure 53]

Time

P

Prompt A wooden ball collides

Context Information

Tag: Momentum Conservation

with an iron cube.

Video Prompt

System Instruction

Stage2：Fine-level Motion Synthesis

[Figure 54]

First Frame

|[Figure 55]|
|---|
|[Figure 56]|
|[Figure 57]|

|[Figure 58]|
|---|
|[Figure 59]|
|[Figure 60]|

|[Figure 61]|
|---|

|[Figure 62]|
|---|

###### MotionAnimationModule

###### NoiseWarpingModule

[Figure 63]

[Figure 64]

###### Motion Controllable I2V Diffusion Model

[Figure 65]

First Frame

Time

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Structured Noise

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

A wooden ball collides with an iron cube.

Motion Trajectory

Optical Flow Prompt

Synthetic Motion Video

Random Noide

- Figure 2. The illustration of our physically plausible image-to-video generation pipeline. Our pipeline consists of two stages. In the first stage, the VLM generates a coarse-grained, physically plausible motion trajectory based on the provided input conditions. In the second stage, We simulate a synthetic video using the predicted trajectory to provide the motion condition. We then extract the optical flow from this video and convert it into structured noise. These conditions are fed into a motion controllable image-to-video diffusion model, and ultimately generates a physically plausible video.

###### 2.2. Motion Controllable Video Generation

Existing studies commonly provide one of the following three types of motion control: bounding box control [19, 23, 30, 46, 52], point trajectory control [34, 37, 40, 50, 54] and camera control [2, 15, 47, 61]. Bounding box control provides object motion guidance by generating a sequence of bounding boxes that track the object’s position over time. Point-trajectory control offers motion cues through pointbased trajectories, enabling drag-style manipulation. Camera motion control guides video generation using explicit 3D camera parameters, ensuring consistency and realistic viewpoint changes. However, these approaches prioritize motion control but often overlook physical plausibility. To address the limitation, we propose a novel framework for physically plausible video generation that incorporates physics as conditions into video diffusion models.

###### 2.3. Generation based on VLMs Planning

VLMs have exhibited robust capabilities in visual understanding and planning [27, 39, 65]. Their strong performance in domains such as robot path planning and video understanding shows their ability in understanding the real physical world. Prior work has successfully leveraged LLMs to guide the layout of images or videos, yielding promising results [25, 53]. VideoDirectorGPT [26] leverages LLMs for fine-grained scene-by-scene planning, explicitly controlling spatial layout to generate temporally consistent long videos. Pandora [55] utilizes LLMs for realtime control through free-text action commands, achieving

domain generality, video consistency, and controllability. However, these efforts have yet to address interactions with real-world physical phenomena, such as collision, fall, and melting.

Moreover, the absence of visual information can cause severe hallucination issues in language models for spatial planning tasks, leading to problems like overlapping object boundaries, disproportionate scaling, and incorrect planning [20, 57]. In this paper, we propose utilizing VLMs as coarse-level motion planners within the image space and incorporate physics-aware reasoning and Chain of Thought (CoT) [51] into the inference process.

#### 3. Method

Task Fomulation. In this paper, our goal is to enable an image-to-video diffusion model to generate physically plausible videos. Since VDMs rely more on memory and casebased imitation and struggle to understand general physical rules[21], the key challenge is how to incorporate physical laws into the models. To achieve this, we need to identify a method to incorporate physical principles into the video diffusion framework. Given an image I ∈ RH×W×C(H is height, W is width and C is the number of channels) and a text description d of possible events based on image I, our framework should infer a physics-compliant guidance as the input condition and synthesize a video that adheres to both physical laws and real-world dynamics.

Overall Pipeline. Overall pipeline of VLIPP is illustrated

[Figure 75]

System Instruction (CoT)

Context Information

Gravity Momentum Conservation

Visual Information Video Caption

Optics Thermodynamics

- Step1: Analyze caption and determine the physical laws.

The theme is two rigid balls move towards each other and collide, so the movement satisfy Newton‘s laws of motion, and the law of conservation of momentum is satisfied when colliding….

- Step2: Analyze the impact of this physical laws.

According to the provided caption and the width and height of the two balls, we know that the two balls are made of the same material and have the same volume…. Left and right ball move towards each other with uniform acceleration. The law of conservation of momentum is satisfied

when a collision occurs…

- Step3: Analyze the impact on box coordinates.

Magnetism

Fluid Mechanics

Context Information Base

[Figure 76]

- • Movement : Newton's laws of motion.
- • Collision : Conservation of Momentum. Tags: Momentum Conservation.

VLM

Before the collision, the two balls accelerate uniformly, the x coordinate of the left ball will increase, and the x coordinate of the right ball will decrease, and the acceleration process will gradually increase the distance between each frame. At the moment of collision, the boundaries of the two balls overlap...

[Figure 77]

##### LLM

[Figure 78]

- F1: Obj1 (148,157,100,100), Obj2 (487,157,100,100)
- F2: Obj1 (151,157,100,100), Obj2 (474,157,100,100)
- F3: Obj1 (160,157,100,100), Obj2 (458,157,100,100)
- F4: Obj1 (176,157,100,100), Obj2 (436,157,100,100)

Query:

Two balls on the table

...

What physical laws are satisfied?

move towards each

other and collide.

[Figure 79]

[Figure 80]

- Figure 3. The illustration of chain-of-thought reasoning in the VLM for generating a coarse-grained motion trajectory. First off, the VLM determines the corresponding physical laws and its context for the given scene. Then, the VLM performs step-by-step reasoning to predict the physically plausible motions of objects in image space, leveraging physical context and chain-of-thought prompting. Finally, the VLM predicts bounding boxes according to real-world physics.

in Figure 2. In the first stage, the VLM conducts semantic analysis and physical attribute analysis on the given image I and a description d to obtain the bounding boxes of the objects in the scene, denoted as b1,b2,...,bn, along with the applicable physical laws l. Next, the VLM infers possible future physical scenarios in the current scene to derive the coarse-level motion trajectories of p1,p2,...,pt in the image space. Finally, we utilize an image-to-video diffusion model to synthesize the detailed dynamics in the video.

###### 3.1. VLM as a Coarse-Level Motion Planner

Our motivation is to incorporate physical laws as constraints into a video diffusion model to enhance the physical plausibility of the generated videos. To achieve this, we must identify a method to inject physical laws into the video diffusion model. Given a video description and the first frame, the task at this stage is to generate coarse-level motion trajectory aligned with physical laws.

Scene Understanding. In the real world, most physical phenomena arise from interactions among objects and their motion trajectories. We first initiate the process by identifying and locating objects within a scene. Inspired by the recent studies [1, 9, 28] in VLMs for scene understand-

ing, we employ GPT-4o [38] to recognize all objects that could be involved in physical phenomena as described in the text description d. These objects are subsequently detected and segmented using Grounded-SAM2 [41], yielding their bounding boxes. By leveraging the pretrained knowledge and common-sense reasoning capabilities of foundation models, we effectively determine the relevant objects in the scene.

Physical-Aware Recognition. To perform more effective reasoning in predicting the motion, it is necessary to determine what specific physical principle to apply in the given context. We utilize the pretrained prior of the LLM to determine the physical laws applicable to the current scene. Following the configuration in the physical benchmark [3, 31, 33], we currently classify common physical phenomena in videos into six categories: gravity, momentum conservation, optics, thermodynamics, magnetism, and fluid mechanics. Note that such list can be easily extended within our framework. Given a video description d, the LLM infers the physical law l that governs the current scene. We provide the specific physical context information for VLM to enhance its understanding of physical laws [11]. Detailed context design is presented in the Appendix.

Chain of Thought Reasoning in VLM. Given the physical law l, an image I and a video description d for the scene, we prompt the VLM to predict the future bounding box positions of objects within the image-space. We choose to predict in the image space for two primary reasons. Firstly, motion in image space aligns more with our subsequent video synthesizer. Secondly, image space dynamics can effectively represent a wide range of real-world motions [29].

At a given time t, the predicted position of i-th object bounding box bti is denoted as [xit,yti,wti,hit], where (xit,yti) represents its top-left coordinate; wti & hit denote its width and height, respectively. Governed by the physical law, the four values of the bounding box may change over time. The VLM reasons the bounding box positions of N future frames for every object oi based on the condition. To help VLM better understand physical laws, we adapt a chain-ofthought [51] into its reasoning, to significantly enhance its reasoning capabilities. As shown in Figure 3, we formulate our analysis of physical phenomena in videos as step-bystep reasoning: beginning with broad conceptual ideas and progressing to a detailed and practical examination:

- 1. Given the physical law l and context information, the VLM analysis video caption and detail the physical law.
- 2. The VLM analyzes the potential interactions and movement of each object within the scene;
- 3. The VLM predicts the detailed changes in position and shape of the bounding box corresponding to each object over time. Through the structured planning process, the VLM plans

coarse-level motion trajectories for the objects, approximating real-world physics dynamics. In particular, our VLM infers the changes of object bounding boxes for next 12 frames, constrained by the token length limitation. To be compatible with the generation process of the chosen VDM in the next stage, these inferred 12 frames are further linearly interpolated to produce a total of 49 frames.

###### 3.2. VDM Serves as a Fine-Level Motion Synthesizer

In the previous stage, the motion trajectory planned by the VLM is neither precise nor fully compliant with physical laws. On the other hand, while VDM may not be able to produce realistic global motion trajectories, it is able to generate sound motion in finer scale. In this stage, our key insight is that the VDM can refine the coarse-level motion to produce physically plausible motion that aligns with realworld dynamics with its powerful generative prior.

Motion Animation. To incorporate physical laws into the video diffusion model, we use the inferred coarse motion trajectory to guide the generation process of the diffusion model. Optical flow provides a unified representation of motion, and recent studies [7, 12, 13] have demonstrated

its effectiveness in guiding diffusion models. Accordingly, we leverage the coarse-level motion trajectory to animate a synthetic motion sequence and derive the corresponding optical flow. Specifically, for each object oi , we extract its bounding box from the first frame and move it to the bounding box location bi specified by the motion trajectory. To animate the change of shape (e.g., due to compression or expansion), we resize object oi according to the difference between oi and oi+1 during inpainting. The synthetic motion video is generated as follows:

Vˆ(t) = Animation(B,rs(o00,b0t)...rs(oi0,bit)) (1) where Vˆ(t) denotes the corresponding frame of inpainted video at timestep t, B denotes the inpainted background with the foreground object removed, o0i denotes the i-th object at timestep 0, bti represents the i-th bounding box at timestep t, and rs denotes resize function.

Structured Noise from Synthetic Video. Optical flow is an effective representation for guiding VDMs [12, 13]. Follow prior work [7, 8], we employ RAFT[44] to extract optical flow from the synthetic video and formulate it as structural noise, which retains Gaussian properties. Given the synthetic video Vˆ(t) ∈ RF×C×H×W, we calculate its per-frame optical flow to get a structured noise tensor Q ∈ RF×C×H×W. The structured noise enables the VDM to generate videos that exhibit motion patterns closely aligne with those in the optical flow, thereby improving the realism of the output.

Noise Injection in Video Synthesis. We adopt Go-withthe-Flow [7] as our video synthesis model, a fine-tuned CogVideoX [60], which is designed to accept structured noise Q as input and synthesize videos that adhere to the implicit optical flow. The vanilla Go-with-the-Flow tends to tightly follow the provided structured noise Q. However, our Q is derived from a coarse-level motion trajectory and may not be sufficiently accurate to follow the physical laws of the real world. To address this limitation, we inject noise during the inference phase to give more flexibility to the VDM to generate detail-level motion changes as

(1 − γ)Qi + ζγ (1 − γ)2 + γ2

(2)

Qi =

where Qi is structured noise at i-th frames, ζ ∈ RC×H×W is Gaussian noise and γ ∈ [0,1]. We set γ = 0.4 for even frame index and γ = 0.6 for odd frame index.

With this approach, the VDM is able to generate motion deviate from the coarse-level motion trajectory whenever necessary for producing high-quality fine-level motion.

#### 4. Empirical Analysis and Disscusion

In this section, we conduct extensive experiments to demonstrate the effectiveness of our video generation framework

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

OursOursCogVXCogVXSVD-XTSVD-XTLTXLTX

OursOursCogVXCogVXSVD-XTSVD-XTLTXLTX

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

Input Frame Input Frame

A rubber ball falls freely under gravity, accelerating

A piece of

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

iron is gently

placed on the surface of the water in a tank filled with water.

as it descends.

Upon hitting the ground, it undergoes an elastic collision.

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

Input Frame Input Frame

A whiskey glass on a wooden table in tavern, and amber-colored whiskey is

A blue marker is used to write on the smooth, white surface of a

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

###### poured into it

whiteboard

from a bottle.

- Figure 4. Visual comparisons of physically plausible video generation results from our framework, CogVideoX-I2V-5B [60], LTX-VideoI2V [14] and SVD-XT [5].

Model Mechanics(↑) Optics(↑) Thermal(↑) Material(↑) Average(↑) CogvideoX-T2V-5B 0.43 0.55 0.40 0.42 0.45

LTX-Video-T2V 0.35 0.45 0.36 0.38 0.39 OpenSora 0.43 0.50 0.44 0.37 0.44 PhyT2V 0.49 0.61 0.49 0.47 0.52 LLM-Grounding Video Diffusion 0.32 0.41 0.26 0.24 0.31

CogvideoX-I2V-5B 0.48 0.69 0.43 0.41 0.52 SVD-XT 0.46 0.68 0.48 0.41 0.52 LTX-Video-I2V 0.47 0.65 0.46 0.37 0.50 SG-I2V 0.52 0.69 0.51 0.39 0.54

###### Ours 0.55 0.71 0.60 0.53 0.60

Table 1. Quantitative results of VDMs on PhyGenBench.

compared to existing methods. We evaluate our approach on two established benchmarks for physically plausible video generation. Our framework consistently achieves superior performance across all benchmarks.

###### 4.1. Implementation Details

We propose a two-stage physically plausible image-tovideo generation framework. In the first stage, we utilize ChatGPT-4o as the coarse-level motion planner. In the second stage, we utilize an open-source I2V model, Go-withthe-Flow [7], as a fine-level motion synthesizer. Unless otherwise specified, in all experiments, we generate each video with a resolution of 720 × 480 and 49 frames.

###### 4.2. Benchmarks and Models

Traditional metrics in the visual domain, such as the Peak Signal-to-Noise Ratio (PSNR)[17], the Structural Similarity Index (SSIM)[49], the Learned Perceptual Image

Patch Similarity (LPIPS)[63], the Fr´echet Inception Distance (FID)[16] and the Fr´echet Video Distance (FVD)[45], do not account for the physical realism of the generated videos [31, 33]. Recent studies have begun to address this limitation by developing benchmarks and metrics that evaluate physical realism. In this work, we adopt two benchmarks, described below.

PhyGenBench [31] categorizes physical properties into four domains: mechanics, optics, thermal, and material. It includes 27 physical phenomena, each governed by real world physical laws, reflected in 160 carefully designed text prompts. As PhyGenbench provides only text prompts, we adapt it to our image-to-video setting by generating a corresponding first frame for each prompt with FLUX[22]. We adhere to the predefined benchmark evaluation protocol, i.e., employing GPT-4o to assess the physical realism of the generated videos.

Cogvideo-I2V-5B 30.4 29.8 16.7 13.3 8.5 27.1 SVD-XT 21.9 20.5 6.8 8.4 17.1 19.1 LTX-Video-I2V 30.2 29.8 15.9 13.2 8.4 26.8 SG-I2V 34.6 31.2 15.9 13.1 8.4 29.7 Ours 42.3 34.1 16.9 13.4 8.8 34.6

Table 2. Quantitative results of physically plausible video generaion on Physics-IQ Benchmark. S.M. refers to Solid Mechanics, and F.D. refers to Fluid Dynamics.

Physics-IQ [33] comprises 396 real-world videos spanning 66 distinct physical scenarios. For each scenario, videos are recorded from three different perspectives and filmed twice under identical conditions to eliminate randomness. This benchmark evaluates real-world physical phenomena, including collisions, object continuity, occlusion, object permanence, and fluid dynamics. This benchmark assesses physical realism from semantic and temporal perspectives, using semantic metrics and visual metrics to compare generated videos against the real-world reference videos.

Compared Models. In the context of text-to-video generation, we compare our framework with CogVideoX-T2V5B [60], LTX-Video-T2V [14], and OpenSora [66]. Moreover, we evaluate our framework against PhyT2V [58], which enhances physical realism by iteratively refining the prompt. For the image-to-video generation scenario, our framework is evaluated alongside CogVideoX-I2V-5B [60], SVD-XT [5], and LTX-Video-I2V [14]. Additionally, we conducted experiments in the motion-controllable setting. In this setting, we leverage the motion trajectory predicted by the VLM as a condition to guide VDM generation. We benchmark our approach against image-to-video motion controllable model, SG-I2V [35] and text-to-video motion controllable model LLM-grounded Video Diffusion Models [25]. The experimental details are presented in the Appendix.

###### 4.3. Quantitative Evaluation

We begin with an empirical study on PhyGenBench and Physics-IQ, comparing our framework against widely adopted open-source models in the research community. Based on different physical properties, we categorize the benchmark samples accordingly. Additionally, we classify VDMs into text-to-video (T2V) diffusion models and image-to-video (I2V) diffusion models based on the input conditions.

In Table 1, we present our experimental results on PhyGenBench, evaluating different video generation models following its evaluation protocol. The results show that our framework achieves state-of-the-art performance across four different physical phenomena. Our framework outperforms the best T2V method by an average of 15.3% and the best I2V method by 11.1%. Specifically, our

framework demonstrates significant advantages in the Mechanics, Thermal, and Material domains, outperforming the best I2V method by 5.7%, 17.6%, and 35.8%, respectively. These advantages are particularly evident in these three types of physical phenomena, which involve more substantial changes in motion, volume, or shape. Our framework is better equipped to understand and reason about bounding box sequences to represent these changes effectively.

Similarly, for the Physics-IQ benchmark, we evaluate the performance of different video generation models following its evaluation protocol. Our framework achieves the best results across four different physical phenomena, with improvements of 22.2% in Solid Mechanics and 9.2% in Fluid Dynamics compared to the second-best models. These significant improvements demonstrate the effectiveness of our framework in generating physically plausible videos.

###### 4.4. Qualitative Evaluation

Figures 4 and 5 demonstrate a qualitative comparison between our video generation framework and baseline methods. Among all evaluated approaches, our framework consistently produces videos with the highest degree of physical realism. In the ball falling sample in Figure 4, while CogVideoX shows a bouncing effect, artifacts are present in the video; LTX-Video and SVD-XT exhibit motions that do not adhere to the laws of physics. In Figure 5, we analyze two examples from Physics-IQ. In the pouring water example, the baseline methods fail to show the simultaneous decrease in water level of the glass beverage dispenser and the increase in water level of the glass below; in the ball collision example, none of the baseline methods correctly depict the collision of balls. More videos are provided in the supplement.

###### 4.5. Ablation Study

We perform an ablation study to evaluate the contributions of key components in our framework. We design four variants to analyze the effectiveness of different components in our framework.

1. Ours w/o VLM Planner: To assess the overall functionality of our framework, we replace the structured noise input of the VDM with random noise to evaluate the ef-

Ours 42.3 34.1 16.9 13.4 8.8 34.9 w/o VLM Planner. 16.3 20.8 13.4 5.8 5.6 16.2

w/o C.I 26.3 28.1 16.9 11.2 8.4 24.3 w/o CoT 21.4 26.9 16.1 8.6 6.9 21.0 w/o C.C 18.7 22.4 14.9 7.2 6.1 18.1

Table 3. Ablation study on VLM, in-context learning and COT. S.M. refers to Solid Mechanics, and F.D. refers to Fluid Dynamics.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

SVD-XTLTXOursCogVX

[Figure 171]

First Frame

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

A glass

beverage dispenser filled with a bright red liquid is set up on a woven basket

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

###### and ispouring

the liquid into a clear glass on a wooden table.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

SG-I2V

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

LTXCogVXSVD-XTSG-I2VOurs

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

First Frame

A table with two tennis balls, one

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

orange and one

brown, placed near thecenter back to back. A grey tennis ball rolls out of a black pipe sitting on the table and

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

towards

theother two balls.

- Figure 5. Visual comparisons of physically plausible video generation results from our framework, CogVideoX-I2V-5B, LTXVideo-I2V, SVD-XT and SG-I2V [35] in the Physics-IQ dataset.

fectiveness of the VLM planner.

- 2. Ours w/o CI: Keeping the overall structure unchanged, we remove the in-context information from the VLM.
- 3. Ours w/o CoT: Similarly, while keeping other components unchanged, we remove the CoT reasoning process from the VLM.
- 4. Ours w/o CC: Lastly, we remove both the in-context information and the CoT reasoning process from the VLM planner while maintaining all other components. Table 3 presents a quantitative comparison between our

full method and these variants. Among all variants, Ours w/o VLM Planner shows the most significant performance drop, as removing the planner completely eliminates our ability in understanding the physical laws, leading to nearly random results. Notably, Ours w/o CoT exhibits a more pronounced decline compared to Ours w/o CI, indicating that the reasoning process in CoT enhances the understanding of physics. While in-context information contributes to the physical reasoning ability of VLM, compared to CoT it

is less effective in preventing errors caused by VLM hallucination.

###### 4.6. User Study

To complement our above evaluations, we conduct a user study to assess the subjective human perception of the generated videos. We follow the gold standard experimental approaches from psychophysics, a 2AFC paradigm, which means two-alternative-forced-choice [33]. In our case, participants completed a questionnaire in which they were presented with pairs of videos and asked to select the one that better aligned with their expectations of physical realism. Responses from 50 participants are summarized in Table 4. The result indicates a strong preference for videos generated by our framework over those from competitors. A detailed analysis of these findings follows in the subsequent discussion.

Model P.P.(↑) V.R.(↑) CogVideoX-I2V-5B 34% 40%

LTX-Video 22% 18% Ours 52% 48%

Table 4. User study statistics of the preference rate for Physical Plausibility (P.P.) & Visual Realism (V.R.).

###### 4.7. Limitations

Although our framework can generate physically plausible videos, its performance remains constrained by the base model. Firstly, we cannot model physical events that cannot be represented by image space bounding box trajectories. For example, phenomena that involve intrinsic state changes of objects such as solid fragmentation and gas solidification. Moreover, our pipeline lacks 3D spatial perception. It is unable to understand the spatial relationships within the scene. Finally, the optical flow of small objects is prone to noise interference. This will cause our framework to generate ambiguous content. With the recent progress in video generation model, we anticipate that our framework will be further improved in generating videos under more challenging physical conditions.

#### 5. Conclusion

Recently, VDMs have achieved great empirical success and are receiving considerable attention in computer vision and computer graphics. However, due to the lack of understanding of physical laws, VDMs are unable to generate physically plausible videos. In this paper, we introduce VLIPP, a novel two-stage physically plausible video generation framework that incorporates physical laws into video diffusion models through vision and language informed physical prior. Our experimental results demonstrate the effectiveness of our method compared to existing approaches.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 4

- [2] Sherwin Bahmani, Ivan Skorokhodov, Guocheng Qian, Aliaksandr Siarohin, Willi Menapace, Andrea Tagliasacchi, David B Lindell, and Sergey Tulyakov. Ac3d: Analyzing and improving 3d camera control in video diffusion transformers. arXiv preprint arXiv:2411.18673, 2024. 3
- [3] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, KaiWei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024. 4
- [4] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. arXiv preprint arXiv:2412.03572, 2024. 2
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 6, 7, 1
- [6] S´ebastien Bubeck, Varun Chadrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023. 2
- [7] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, et al. Go-with-the-flow: Motioncontrollable video diffusion models using real-time warped noise. arXiv preprint arXiv:2501.08331, 2025. 2, 5, 6
- [8] Pascal Chang, Jingwei Tang, Markus Gross, and Vinicius C Azevedo. How i warped your noise: a temporally-correlated noise prior for diffusion models. In The Twelfth International Conference on Learning Representations. OpenReview, 2024. 5
- [9] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. Science China Information Sciences, 67(12):220101,

2024. 4

- [10] Boyang Deng, Richard Tucker, Zhengqi Li, Leonidas Guibas, Noah Snavely, and Gordon Wetzstein. Streetscapes: Large-scale consistent street view generation using autoregressive video diffusion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2
- [11] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022. 4
- [12] Daniel Geng and Andrew Owens. Motion guidance: Diffusion-based image editing with differentiable motion estimators. arXiv preprint arXiv:2401.18085, 2024. 5
- [13] Daniel Geng, Charles Herrmann, Junhwa Hur, Forrester Cole, Serena Zhang, Tobias Pfaff, Tatiana Lopez-Guevara, Carl Doersch, Yusuf Aytar, Michael Rubinstein, et al. Motion prompting: Controlling video generation with motion trajectories. arXiv preprint arXiv:2412.02700, 2024. 5
- [14] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103,

2024. 6, 7, 1

- [15] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3
- [16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6
- [17] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010. 6
- [18] Hao-Yu Hsu, Zhi-Hao Lin, Albert Zhai, Hongchi Xia, and Shenlong Wang. Autovfx: Physically realistic video editing from natural language instructions. arXiv preprint arXiv:2411.02394, 2024. 2
- [19] Hsin-Ping Huang, Yu-Chuan Su, Deqing Sun, Lu Jiang, Xuhui Jia, Yukun Zhu, and Ming-Hsuan Yang. Fine-grained controllable video generation via object appearance and context. arXiv preprint arXiv:2312.02919, 2023. 3
- [20] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025. 3
- [21] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385, 2024. 2, 3
- [22] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 6, 1
- [23] Pengxiang Li, Kai Chen, Zhili Liu, Ruiyuan Gao, Lanqing Hong, Guo Zhou, Hua Yao, Dit-Yan Yeung, Huchuan Lu, and Xu Jia. Trackdiffusion: Tracklet-conditioned

- video generation via diffusion models. arXiv preprint arXiv:2312.00651, 2023. 3
- [24] Zhengqi Li, Richard Tucker, Noah Snavely, and Aleksander Holynski. Generative image dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24142–24153, 2024. 2
- [25] Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models. arXiv preprint arXiv:2309.17444, 2023. 3, 7, 1
- [26] Han Lin, Abhay Zala, Jaemin Cho, and Mohit Bansal. Videodirectorgpt: Consistent multi-scene video generation via llm-guided planning. arXiv preprint arXiv:2309.15091,

2023. 3

- [27] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023. 3
- [28] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024. 4
- [29] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded imageto-video generation. In European Conference on Computer Vision, pages 360–378. Springer, 2024. 2, 5
- [30] Wan-Duo Kurt Ma, John P Lewis, and W Bastiaan Kleijn. Trailblazer: Trajectory control for diffusion-based video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024. 3
- [31] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024. 4, 6, 1
- [32] Antonio Montanaro, Luca Savant Aira, Emanuele Aiello, Diego Valsesia, and Enrico Magli. Motioncraft: Physicsbased zero-shot video generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. 2
- [33] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models learn physical principles from watching videos? arXiv preprint arXiv:2501.09038, 2025. 2, 4, 6, 7, 8, 1
- [34] Chong Mou, Mingdeng Cao, Xintao Wang, Zhaoyang Zhang, Ying Shan, and Jian Zhang. Revideo: Remake a video with motion and content control. Advances in Neural Information Processing Systems, 37:18481–18505, 2024. 3
- [35] Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B Lindell. Sg-i2v: Self-guided trajectory control in image-to-video generation. arXiv preprint arXiv:2411.04989, 2024. 7, 8, 1
- [36] Andrew Nealen, Matthias M¨uller, Richard Keiser, Eddy Boxerman, and Mark Carlson. Physically based deformable models in computer graphics. In Computer graphics forum, pages 809–836. Wiley Online Library, 2006. 2
- [37] Muyao Niu, Xiaodong Cun, Xintao Wang, Yong Zhang, Ying Shan, and Yinqiang Zheng. Mofa-video: Controllable image animation via generative motion field adaptions in frozen image-to-video diffusion model. In European Conference on Computer Vision, pages 111–128. Springer, 2024. 3

- [38] Openai. https://openai.com/index/hello-gpt-4o/, 2024. 4
- [39] Chenbin Pan, Burhaneddin Yaman, Tommaso Nesti, Abhirup Mallik, Alessandro G Allievi, Senem Velipasalar, and Liu Ren. Vlp: Vision language planning for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14760–14769, 2024. 3
- [40] Haonan Qiu, Zhaoxi Chen, Zhouxia Wang, Yingqing He, Menghan Xia, and Ziwei Liu. Freetraj: Tuning-free trajectory control in video diffusion models. arXiv preprint arXiv:2406.16863, 2024. 3
- [41] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159,

2024. 4

- [42] Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, and Andrew Selle. A material point method for snow simulation. ACM Transactions on Graphics (TOG), 32(4): 1–10, 2013. 2
- [43] Weixiang Sun, Xiaocao You, Ruizhe Zheng, Zhengqing Yuan, Xiang Li, Lifang He, Quanzheng Li, and Lichao Sun. Bora: Biomedical generalist video generation model. arXiv preprint arXiv:2407.08944, 2024. 2
- [44] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part II 16, pages 402–419. Springer,

2020. 5

- [45] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Rapha¨el Marinier, Marcin Michalski, and Sylvain Gelly. Fvd: A new metric for video generation. 2019. 6
- [46] Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and controllable motions for video synthesis. arXiv preprint arXiv:2402.01566, 2024. 3
- [47] Xi Wang, Robin Courant, Marc Christie, and Vicky Kalogeiton. Akira: Augmentation kit on rays for optical video generation. arXiv preprint arXiv:2412.14158, 2024. 3
- [48] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-worlddrive world models for autonomous driving. In European Conference on Computer Vision, pages 55–72. Springer,

2024. 2

- [49] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 6
- [50] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
- [51] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022. 3, 5

- [52] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. arXiv preprint arXiv:2406.17758, 2024. 3
- [53] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llm-controlled diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6327– 6336, 2024. 3
- [54] Weijia Wu, Zhuang Li, Yuchao Gu, Rui Zhao, Yefei He, David Junhao Zhang, Mike Zheng Shou, Yan Li, Tingting Gao, and Di Zhang. Draganything: Motion control for anything using entity representation. In European Conference on Computer Vision, pages 331–348. Springer, 2024. 3
- [55] Jiannan Xiang, Guangyi Liu, Yi Gu, Qiyue Gao, Yuting Ning, Yuheng Zha, Zeyu Feng, Tianhua Tao, Shibo Hao, Yemin Shi, et al. Pandora: Towards general world model with natural language actions and video states. arXiv preprint arXiv:2406.09455, 2024. 3
- [56] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physicsintegrated 3d gaussians for generative dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4389–4398, 2024. 2
- [57] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817, 2024. 3
- [58] Qiyao Xue, Xiangyu Yin, Boyuan Yang, and Wei Gao. Phyt2v: Llm-guided iterative self-refinement for physics-grounded text-to-video generation. arXiv preprint arXiv:2412.00596, 2024. 1, 7
- [59] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 1(2):6, 2023. 2
- [60] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint

- arXiv:2408.06072, 2024. 5, 6, 7, 1

[61] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint

- arXiv:2409.02048, 2024. 3

- [62] Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. In Proceedings of the IEEE/CVF international conference on computer vision, pages 16010–16021, 2023. 2
- [63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [64] Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interac-

tion with 3d objects via video generation. In European Conference on Computer Vision, pages 388–406. Springer, 2024.

- 2

- [65] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024.

3

- [66] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 7

## VLIPP: Towards Physically Plausible Video Generation with Vision and Language Informed Physical Prior

### Supplementary Material

#### A. Coarse-Level Motion Planning Details

In this section, we present the experimental setting and details for reproducing the results. The main principle of our experimental setting is to fairly compare different Video Diffusion Models(VDMs) in generating physically plausible videos. Our adapt well-known open source model to serve as Compared Models. We disscuss these models in details.

- 1. CogVideoX[60]: CogVideoX is capable of performing both text-to-video generation and image-to-video generation. It provides two model variants, featuring 2 billion and 5 billion parameters, respectively. In our experiments, we configured CogVideoX to generate 49 frames with a resolution of 720×480.
- 2. LTX-Video[14]: LTX-Video is also capable of performing both text-to-video generation and image-to-video generation. In our experiments, we compared two versions of LTX-Video with corresponding methods. It can generate videos with 49 frames with a resolution of 768×512.
- 3. SVD-XT[5]: SVD-XT is capable of performing imageto-video generation. In our experiments, we configured SVD-XT to generate 25 frames with a resolution of 1024×576.
- 4. SG-I2V[35]: SG-I2V is a motion trajectory-guided image-to-video generation model. It is capable of generating bounding box-controllable videos with 14 frames with a resolution of 1024×576.
- 5. LLM-grounded Video Diffusion Models[25]: LLMgrounded Video Diffusion Models are capable of predicting future frame bounding boxes based on input prompts and injecting the box information in a trainingfree manner. In our experiments, we configured LLMgrounded to generate 24 frames with a resolution of 576×320.

We additionally present the Reasoning Template utilized during the stage 1 Coarse-Level Motion Planning process, as shown in Fig 6 and Fig 7. This includes system instructions to ensure the proper functioning of the chain of thought and provides the VLM with context information to guarantee the accuracy of predictions.

#### B. Experiment Details

In this section, we present the experimental details of our benchmark, PhyGenBench[31] and Physics-IQ[33].

PhyGenBench comprises 160 prompts, spanning four

domains of physical knowledge: Mechanics (40), Optics (50), Thermal (40), and Material (20), along with 27 types of physical laws. It also includes 165 objects and 42 actions. The evaluation focuses on two aspects: semantic alignment and physical commonsense alignment. The degree of semantic alignment is assessed by extracting objects and actions from the prompts using a Vision-Language Model (VLM), determining whether the objects appear, and evaluating based on the presence of objects and the occurrence of actions. The degree of physical commonsense alignment is determined through a three-step process: detecting whether the physical phenomena occur and whether the order of occurrence is correct; and finally conducting an overall naturalness evaluation.

Physic-IQ categorizes real-world physical laws into Solid Mechanics, Fluid Dynamics, Optics, Magnetism, and Thermodynamics, encompassing 114, 45, 24, 6, and 9 videos, respectively. The evaluation approach is twofold, focusing on physical comprehension and visual authenticity. Physical comprehension is determined by identifying the timing, location, and frequency of actions, ultimately calculating the mean squared error between corresponding pixels in the generated and real frames to derive a physical comprehension score. Visual authenticity is evaluated using a Vision-Language Model (VLM), employing the gold standard experimental method from psychophysics. The VLM receives pairs of real and generated videos of the same scene in random order and is tasked with identifying the real scene, a design intended to reflect visual authenticity.

During the experimental phase of this paper, we utilized the prompts provided by the PhyGenBench dataset to infer the initial frame’s prompts using an LLM, which were then generated by FLUX[22]. To ensure fairness in comparison, all I2V models were supplied with the same initial frame image. Given that different models produce videos with varying numbers of frames, a uniform sampling ratio was applied during the testing phase to extract key frames consistently across all models.

#### C. More Qualitative Results

In this section, we further demonstrate examples of the proposed framework across various scenarios using prompts from the PhyGenBench dataset. Fig 8, 9, and 10 show video generation results driven by physical conditions like fluid dynamics and thermodynamics, while Fig 11 highlights effects in light reflection scenarios.

###### Physics-Aware Reasoning Template (Gravity/Moumentum)

###### System Instruction

You are an expert in real-world physical motion. Your task is to predict video bounding boxes. You will receive a reference image, its segmentation map, the initial bounding box where motion starts, and a textual description of the video content. Your task is to predict the bounding box coordinates for the next 13 frames. When predicting, ensure that different bounding boxes do not overlap or exceed the frame boundaries.

Each frame should be represented as: `[{'id': unique object identifier, 'name': object name, 'box': [box top-left x-coordinate, box top-left y-coordinate, box width, box height]}, ...]`.Your predicted frames must cover the entire described scene. Critical moments in the movement must be included in the predicted sequence, such as the moment of collision.You need to consider the material and properties of the moving objects in the video. You only need to focus on the

objects provided to you in the first frame, and no need to pay attention to anything else. Your predictions should infer which physical laws apply, such as Gravity,

Conservation of Momentum, Fluid Mechanics, Thermodynamics, Magnetic Force, etc. For example, under the action of gravity, the ball will accelerate and rebound after colliding with the ground. When two balls move towards each other, the material, volume, density, etc. of each ball must be considered, and the collision process satisfies the law of conservation of momentum. Assume objects move and interact based on real-world physics. The camera follows perspective geometry. Even if an object disappears and reappears, its bounding box should maintain the same ID across frames. If necessary, you may make reasonable assumptions.Before generating bounding box coordinates for each frame, provide step-by-step reasoning.

Context Information

[Figure 217]

Caption: Generate a high-quality video of a wooden ball dropping from a height of about 1 meter onto a smooth, solid table.Capture the physics of the ball's descent, including its acceleration due to gravity, the moment of impact with the table, and the subsequent bounce. Show the ball compressing slightly upon impact, then rebounding, losing a bit of height with each bounce until it comes to rest. Include subtle details like the sound of the impact, the slight vibration of the table, and the ball's rotation as it moves. initial_boxes: [{'id': 0, 'name': 'ball', 'box': [303.0, 82.2, 108, 108]}]

User:

[Figure 218]

Reasoning:

Assistant:

###### Step1-Analysis Caption and determine the physical laws to be satisfied: As can be seen from the caption, the topic of this study is about the free fall of a

small ball, so this topic needs to satisfy Newton's laws of motion, focusing on the impact of gravity.

###### Step2-Analyze the impact of this physical law: In accordance with Newton's laws of motion, gravity will cause the small ball to perform free fall motion,

vertically accelerate and hit the table, and rebound after the collision. Due to energy loss, the height of each rebound will decrease, which is in line with the law of conservation of energy and momentum.

###### Step3-Analyze the impact on box coordinates: The horizontal motion of the ball is negligible, so the x coordinate remains basically unchanged. Due to the

vertical acceleration, the y coordinate of the small ball will gradually increase and the interpolation of the y coordinate between each two frames will also gradually increase. After colliding with the table, the small ball rebounds, the y coordinate decreases, and the small ball decelerates vertically upward. Subsequently, the small ball continues to accelerate vertically and collides with the table for the second time. After the third collision, the small ball is stationary on the table, and the y coordinate remains unchanged. Since the ball is a rigid body, it will be slightly compressed at the moment of collision with the desktop.

Therefore, the width in the box coordinates remains unchanged, the height decreases slightly, and then quickly returns to normal. The width and height will not

change for the rest of the time.

Predict the coordinates for the next 13 frames:

- Frame 1: [{'id': 0, 'name': 'wooden ball', 'box': [303, 82, 108, 108]}]
- Frame 2: [{'id': 0, 'name': 'wooden ball', 'box': [303, 137, 108, 108]}]
- Frame 3: [{'id': 0, 'name': 'wooden ball', 'box': [303, 184, 108, 108]}]
- Frame 4: [{'id': 0, 'name': 'wooden ball', 'box': [303, 313, 108, 108]}]
- Frame 5: [{'id': 0, 'name': 'wooden ball', 'box': [303, 255, 108, 108]}]
- Frame 6: [{'id': 0, 'name': 'wooden ball', 'box': [303, 284, 108, 104]}]
- Frame 7: [{'id': 0, 'name': 'wooden ball', 'box': [303, 313, 108, 108]}]
- Frame 8: [{'id': 0, 'name': 'wooden ball', 'box': [303, 298, 108, 108]}]
- Frame 9: [{'id': 0, 'name': 'wooden ball', 'box': [303, 305, 108, 108]}]
- Frame 10: [{'id': 0, 'name': 'wooden ball', 'box': [303, 313, 108, 104]}]
- Frame 11: [{'id': 0, 'name': 'wooden ball', 'box': [303, 313, 108, 108]}]
- Frame 12: [{'id': 0, 'name': 'wooden ball', 'box': [303, 313, 108, 108]}]
- Frame 13: [{'id': 0, 'name': 'wooden ball', 'box': [303, 313, 108, 104]}]

Figure 6. Physics-Aware Reasoning Template for Rigid Body Motion

###### Physics-Aware Recognition Template (Fluid)

###### System Instruction

You are an expert in real-world physical motion. Your task is to predict video bounding boxes. You will receive a reference image, its segmentation map, the initial bounding box where motion starts, and a textual description of the video content. Your task is to predict the bounding box coordinates for the next 13 frames. When predicting, ensure that different bounding boxes do not overlap or exceed the frame boundaries.

Each frame should be represented as: `[{'id': unique object identifier, 'name': object name, 'box': [box top-left x-coordinate, box top-left y-coordinate, box width, box height]}, ...]`.Your predicted frames must cover the entire described scene. Critical moments in the movement must be included in the predicted sequence, such as the moment of collision.You need to consider the material and properties of the moving objects in the video. Your predictions should infer which physical laws apply, such as Gravity, Conservation of Momentum, Fluid Mechanics, Thermodynamics, Magnetic Force, etc. For example, under the influence of fluid mechanics, when pouring water into a cup, the water level in the cup should rise and ripples should occur. The top boundary of a liquid should move when it's being poured, while the bottom edge of the liquid (for instance, in a container) should stay fixed. When an object (such as a liquid) is in a container and is being poured, the bottom boundary of the liquid typically stays fixed. Assume objects move and interact based on real-world physics. The camera follows perspective geometry. Even if an object disappears and reappears, its bounding box should maintain the same ID across frames. If necessary, you may make reasonable assumptions.Before generating bounding box coordinates for each frame, provide step-by-step reasoning.

Context Information

[Figure 219]

Caption: The video shows a close-up of a clear glass being filled with tea. The tea is being poured from above, and we can see the stream of tea hitting the bottom of the glass and causing ripples and splashes. The background is a plain white surface, which contrasts with the transparency of the glass and the clarity of the tea. The glass is cylindrical in shape and appears to be of a standard size for a drinking glass.

User:

initial_boxes: [{'id': 0, 'name': 'tea', 'box': [241.8, 335.3, 165, 88]}] Reasoning:

[Figure 220]

Assistant:

###### Step1-Analysis Caption and determine the physical laws to be satisfied: As can be seen from the caption, the theme of this session is to pour tea into a

glass teacup. Therefore, the tea is affected by gravity during its fall, and the state of the tea in the cup satisfies fluid mechanics.

###### Step2-Analyze the impact of this physical law: Tea falls into the cup under the influence of gravity. According to fluid mechanics, the water level in the cup

will gradually rise, and the water surface will ripple due to the pouring of water.

###### Step3-Analyze the impact on box coordinates: When tea is poured into the cup, under the restriction of the cup, the x coordinate of the tea box will not

change much. As the water level rises, the y coordinate will gradually decrease. As the water level rises, the overall volume of the tea gradually increases, the height of the box will also increase, and the width will change with the boundary of the cup.

Predict the coordinates for the next 13 frames:

- Frame 1: [{'id': 0, 'name': 'tea', 'box': [241.8, 335.3, 165, 88]}]
- Frame 2: [{'id': 0, 'name': 'tea', 'box': [241.2, 330.0, 165, 94]}]
- Frame 3: [{'id': 0, 'name': 'tea', 'box': [241.0, 322.2, 165, 102]}]
- Frame 4: [{'id': 0, 'name': 'tea', 'box': [241.3, 317.6, 165, 105]}]
- Frame 5: [{'id': 0, 'name': 'tea', 'box': [240.6, 313.7, 166, 111]}]
- Frame 6: [{'id': 0, 'name': 'tea', 'box': [239.1, 305.6, 167, 118]}]
- Frame 7: [{'id': 0, 'name': 'tea', 'box': [237.7, 301.1, 170, 123]}]
- Frame 8: [{'id': 0, 'name': 'tea', 'box': [236.8, 295.3, 172, 129]}]
- Frame 9: [{'id': 0, 'name': 'tea', 'box': [235.7, 290.2, 172, 134]}]
- Frame 10: [{'id': 0, 'name': 'tea', 'box': [235.1, 283.3, 173, 140]}]
- Frame 11: [{'id': 0, 'name': 'tea', 'box': [234.0, 278.3, 175, 145]}]
- Frame 12: [{'id': 0, 'name': 'tea', 'box': [233.1, 272.0, 177, 152]}]
- Frame 13: [{'id': 0, 'name': 'tea', 'box': [231.7, 268.0, 179, 156]}]

Figure 7. Physics-Aware Reasoning Template for fluid dynamics and thermodynamics.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

SVD-XTLTXCogVXOurs

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

First Frame

A clear glass of oil is gently poured into a glass of water.

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

SVD-XTLTXCogVXOurs

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

First Frame

A timelapse captures the

reaction as

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

concentrated sulfuric acid is poured onto a cotton ball.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

Figure 8. More examples of generated videos related to fluid dynamics and thermodynamics.

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

SVD-XTLTXCogVXOurs

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

First Frame A timelapse captures the gradual transformation

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

of a candle as

the temperature rises significantly.

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

SVD-XTLTXCogVXOurs

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

First Frame

A timelapse captures the gradual

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

transformation

of butter as the temperature rises significantly.

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

Figure 9. More examples of generated videos related to thermodynamics.

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

SVD-XTLTXCogVXOurs

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

First Frame

A stone is gently placed

on the surface

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

of a pool filled with water.

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

SVD-XTLTXCogVXOurs

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

First Frame

A piece of wood block is gently placed on the surface of a bowl filled with water.

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

Figure 10. More examples of generated videos related to fluid dynamics.

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

SVD-XTLTXCogVXOurs

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

First Frame

A clear glass of

oil is gently

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

poured into a glass of water.

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

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

SVD-XTLTXCogVXOurs

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

First Frame

A timelapse captures the reaction as

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

concentrated

sulfuric acid is poured onto a cotton ball.

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

Figure 11. More examples of generated videos related to optics.

