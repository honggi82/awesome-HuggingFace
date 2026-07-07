## : Creating New Games with Generative Interactive Videos

[Figure 1]

Jiwen Yu1∗† Yiran Qin1∗ Xintao Wang2‡ Pengfei Wan2 Di Zhang2 Xihui Liu1‡

1 The University of Hong Kong 2 Kuaishou Technology

https://yujiwen.github.io/gamefactory/

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

W A S D Space

W A S D Space

W A S D Space

W A S D Space

W A S D Space

[Figure 13]

W A S D Space

# arXiv:2501.08325v4[cs.CV]30Oct2025

(1) Prompt: Standing in the cherry blossom forest with a gun in first person perspective.

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

W A S D Space

[Figure 20]

[Figure 21]

[Figure 22]

W A S D Space

W A S D Space

W A S D Space

W A S D Space

W A S D Space

[Figure 23]

[Figure 24]

[Figure 25]

(2) Prompt: An emerald green velvet accent chair sits prominently in the center of a minimalist room.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

W A S D Space

W A S D Space

W A S D Space

W A S D Space

W A S D Space

[Figure 32]

W A S D Space

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

(3) Prompt: A large Saint Bernard walks steadily across the vast, snow-covered expanse of a mountain range.

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

W A S D Space

W A S D Space

W A S D Space

W A S D Space

W A S D Space

W A S D Space

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

(4) Prompt: In a Renaissance palace, from a first person perspective, one can see a lion in front.

Figure 1. We propose GameFactory, a framework that leverages the powerful generative capabilities of pre-trained video models for the creation of new games. By learning action control from a small-scale first-person Minecraft dataset, this framework can transfer these control abilities to open-domain videos, ultimately allowing the creation of new games within open-domain scenes. As illustrated in (1)(4), GameFactory supports action control across diverse, newly generated scenes in open domains, paving the way for the development of entirely new game experiences. The yellow buttons indicate pressed keys, and the arrows represent the direction of mouse movements.

### Abstract

autoregressive generation for unlimited-length interactive videos. More importantly, GameFactory tackles the critical challenge of scene-generalizable action control, which most existing methods fail to address. To enable the creation of entirely new and diverse games beyond fixed styles and scenes, we leverage the open-domain generative priors from pre-trained video diffusion models. To bridge the domain gap between open-domain priors and small-scale game datasets, we propose a multi-phase training strategy with a domain adapter that decouples game style learning from action control. This decoupling ensures that action control learning is no longer bound to specific game styles, thereby achieving scene-generalizable action control. Experimental

Generative videos have the potential to revolutionize game development by autonomously creating new content. In this paper, we present GameFactory, a framework for actioncontrolled scene-generalizable game video generation. We first address the fundamental challenge of action controllability by introducing GF-Minecraft, an action-annotated game video dataset without human bias, and developing an action control module that enables precise control over both keyboard and mouse inputs. We further extend to support

‡Corresponding authors. ∗Equal contribution. †Part of the work done during internship at KwaiVGI, Kuaishou Technology.

###### Inference

###### Pretraining

scene-generation priors presents a more feasible path to scene generalization (upper blue section in Fig. 2). This observation motivates us to ask: With the video generation models pre-trained on large-scale open-domain videos, along with small-scale action-annotated game videos, can we empower scene-generalizable video generation with action control? This is a nontrivial problem, because directly fine-tuning the pre-trained video generation models with action-annotated game videos will lead to degraded scene generalizability (i.e., the model will collapse to the specific domain of the game videos).

[Figure 50]

OpenDomain Results

[Figure 51]

OpenDomain Data

Large Video Generation Models

[Figure 52]

Large-Scale No Action Labeled

Powerful Prior

Generalize

Plug

[Figure 53]

[Figure 54]

ActionControlled Results

Action Control Module

[Figure 55]

[Figure 56]

[Figure 57]

Game Data

[Figure 58]

[Figure 59]

Finetuning

Small-Scale Action Labeled

CreateNew Games!

- Figure 2. A schematic of our GameFactory creating new games based on pre-trained large video generation models. The upper blue section shows the generative capabilities of the pretrained model in an open-domain, while the lower green section demonstrates how the action control module, learned from a small amount of game data, can be plugged in to create new games.

In this work, we present GameFactory, a framework for action-controlled and scene-generalizable game video generation. To achieve action-controlled video generation, we firstly collect an action-annotated game video dataset GFMinecraft. This dataset, sourced from the Minecraft platform [13], features an unbiased action distribution, diverse scenes, and text annotations (Sec. 4.1). Secondly, we design action control mechanisms for continuous mouse movements and discrete keyboard inputs, respectively. The action control modules can be seamlessly injected into the pretrained video diffusion models without affecting the video generation abilities of the pre-trained models (Sec. 4.2). Lastly, we extend our model to autoregressive long video generation by allowing varying noise levels across different frames, continuously conditioning on previously generated frames to generate new video frames (Sec. 4.3).

results demonstrate that GameFactory effectively generates open-domain action-controllable game videos, representing a significant step forward in AI-driven game generation.

### 1. Introduction

Video diffusion models have demonstrated impressive capabilities in video generation [26, 31, 50, 52]. These models have the potential to become promising candidates for generative game engines [2, 6, 11, 12, 42], which could transform game development by not only significantly reducing manual work in the traditional game industry through automatic content creation, but also enabling the generation of unlimited game content for players to explore. Given this promising direction, it is important to explore how to build a qualified generative game engine.

However, directly fine-tuning the pretrained model on Minecraft data not only imparts action control capabilities, but also risks embedding a distinct Minecraft style into the generated games, thereby compromising the model’s opendomain generalization. To address this challenge, our key idea is to disentangle the learning of game style and action control with different modules and parameters, so that we can remove the game style without affecting the action controllability for generating open-domain videos with action control (Sec. 5.1). Specifically, we propose a domain adapter to fit Minecraft style, combined with a multiphase decoupled training strategy (Sec. 5.2).

Generative game engines are typically implemented as video generation models with action controllability, enabling responses to user inputs like keyboard and mouse interactions [11, 12, 42]. Current research works [2, 7, 11, 42, 47] mainly focus on specific games such as DOOM [42, 47], Atari [2], CS:GO [2], Super Mario Bros [47], and Minecraft [11], or use limited game-specific datasets [7]. This game-specific approach lacks scene generalization capability, preventing models from creating content beyond existing games and limiting their potential for developing new games. Thus, scene generalization remains a key challenge in advancing generative game engines.

In summary, our key contributions include:

- • We propose GameFactory for action-controlled and scene-generalizable video generation, enabling diverse interactive game generation beyond existing games.
- • We introduce GF-Minecraft, an action-annotated game video dataset featuring unbiased action distributions, diverse scenes, and text descriptions. We also design dedicated mechanisms for action control and autoregressive long video generation.
- • We propose a domain adapter with multiphase decoupled training strategy that disentangles game style from action control, enabling open-domain video generation while preserving action controllability.

To achieve scene generalization in game videos, collecting large-scale action-annotated video datasets would be the most direct approach. With sufficient data covering all possible scenarios, it could enable arbitrary game scene generation. However, such action annotations are prohibitively expensive and impractical for open-domain videos. In contrast, open-domain videos are abundant on the Internet, and video generation models trained on these videos contain rich generative priors. Leveraging these

### 2. Related Work

#### 2.1. Video Diffusion Models

With the rise of diffusion models [19, 40, 41], significant progress has been made in visual generation, spanning both image [10, 34, 38] and video generation [5, 9, 26, 50, 52]. In particular, recent advancements in video generation have seen a shift from the U-Net architecture to Transformerbased architectures, enabling video diffusion models to generate highly realistic, longer-duration videos [31, 33]. The quality of these generated videos has led to the belief that video diffusion models are capable of understanding and simulating real-world physical rules [31, 45, 49], suggesting their potential use as world models in areas such as autonomous driving and robotics.

#### 2.2. Controllable Video Generation

Text descriptions alone often provide limited control in textto-video models, leading to ambiguous outputs. To enhance control, several methods have introduced additional control signals. For instance, approaches such as [16, 30, 44] incorporate images as control signals for the video generator, improving both video quality and temporal relationship modeling. Direct-a-Video [48] uses a camera embedder to adjust camera poses, but its reliance on only three camera parameters limits control to basic movements, like left panning. In contrast, MotionCtrl [43] and CameraCtrl [17] offer more complex and nuanced control over camera poses in generated videos.

#### 2.3. Game Video Generation

Early works [24, 25, 27–29] began exploring game generation using generative models like GAN, but were primarily limited by the generative capability of these models. Recently, given the promising capabilities of video generation models, researchers have begun exploring their application in game generation [2, 6, 7, 11, 12, 14, 23, 42, 47]. Genie [6] proposes a foundation model for playable world based on video generation. DIAMOND [2], GameNGen [42], Oasis [11] and PlayGen [47] leverage diffusion-based world modeling for specific games like Atari [2], CS:GO [2], DOOM [42, 47], Minecraft [11] and Super Mario Bro [47]. GameGenX [7] introduces OGameData for game video generation and control. However, these works suffer from overfitting to specific games or datasets, exhibiting limited scene generalization capabilities. Recent works like Genie 2 [12] and Matrix [14] have made valuable progress towards control generalization in game video generation. While Genie 2 demonstrates impressive results through extensive action-labeled data collection, and Matrix shows promising generalization capabilities in racing games, there remains room for improvement in achieving broader scene generalization with more diverse action controls.

In contrast, our proposed GameFactory addresses scene generalization by utilizing pretrained video model priors and the easily-accessible and low-cost game dataset GFMinecraft, with experiments validating its effectiveness across diverse open-world scenarios and a much more complex action space (including forward/backward/left/right movement, jumping, and acceleration/deceleration).

### 3. Preliminaries

We adopt a transformer-based latent video diffusion model [31, 33] as the backbone. Let X represent a video sequence. To reduce modeling complexity, an Encoder E(·) compresses the video spatially and temporally into a latent representation Z = E(X). With temporal compression ratio r, a (1 + rn)-frame video is compressed into (1 + n) latent frames. Denoting the i-th frame as xi and i-th latent as zi, we have X = [x0,x1,...,xrn] and Z = [z0,z1,...,zn]. During training, noise is added to the clean latent Z0 to get noisy latent Zt at timestep t. ϵ is the added random noise. When considering action control, the action A = [a1,a2,...,arn], where ai represents the action taken at timestep (i − 1) to transfer from xi−1 to xi. The corresponding action-conditioned loss function is:

La(ϕ) = E[||ϵϕ(Zt,p,A,t) − ϵ||22], (1)

where ϕ means the model parameters, and p is the prompt input. During inference, we can sample clean latent Z0 from a noisy latent ZT. The predicted latent Z0 is then decoded back to video X through D(·): X = D(Z0).

### 4. Action-Controlled Video Generation

This section introduces implementing an action-controlled video generation model, which is the foundation for Sec. 5. Sec. 4.1 describes our collected game video dataset and its advantages; Sec. 4.2 introduces how to implement the action control module for responding to player actions; Sec. 4.3 presents the method for autoregressive long video generation, which is the key to creating playable game.

#### 4.1. GF-Minecraft Dataset

For our action-controllable video generation model to simulate real game engines, the training data should satisfy three key requirements: (1) easily accessible with customizable action inputs to enable cost-effective data collection; (2) action sequences free from human bias, allowing extreme and low-probability action combinations to support arbitrary action inputs; (3) diverse game scenes with corresponding textual descriptions to learn scene-specific physical dynamics. Existing datasets like VPT [3] are collected from human gameplay videos, which inherently contain human behavioral biases and lack text descriptions of the scenes. To address these limitations, we introduce our GF-Minecraft

[[0.8, -0.9], [-0.3,0.4],...]

0 5

- 1

- 2

- 3

- 4

- 9

- 10

- 11

- 12

- 13

- 14

- 15

- 16

VideoFrames

Spatial Self-Attention

C

- 6

- 7

- 8

###### +

Group Repeat

[Figure 60]

MLP

...

Spatial-Temporal Self-Attention

Temporal Self-Attention

+

i i-th Video Frame

Prompt Cross-Attention

+

+

[[1,0,0,1,1,0,0], [0,1,0,1,1,0,0],...]

Compressed Video Latents

i i-th Latent / Feature

Action Control Module

...

0 1 2 3 4

V K Q

Embed Group

[Figure 61]

i i-th Action

Key Action Coss-Attention

MLP

VAE Compression

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 13

- 14

- 15

- 16

- 9

- 10

- 11

- 12

+ +

Actions

Grouped Action Fusion

###### ...

(a) Transformer Block with Action Control Module

(b) Achitecture of Action Control Module

- Figure 3. (a) Integration of Action Control Module into transformer blocks of the video diffusion model. (b) Different control mechanisms for continuous mouse and discrete keyboard inputs.

dataset, where ‘GF’ stands for our method GameFactory and ‘Minecraft’ refers to the game name. The advantages of our dataset are summarized as follows and its details can be found in the supplementary materials:

❑ Minecraft as an Accessible Data Source. We leverage Minecraft as our data collection platform for its comprehensive API that captures detailed environmental snapshots, enabling large-scale data collection with action annotations. The game also offers extensive scenes, navigable areas, diverse action space, and an open-world environment. By executing predefined action sequences, we collected 70 hours of gameplay video as GF-Minecraft Dataset.

❑ Collecting Videos with Unbiased Action. Existing Minecraft datasets, such as VPT [3], are collected from real human gameplay, resulting in biased action distributions that favor common human behaviors. Models trained on these datasets overlook rare action combinations, such as moving backward, jumping in place, or standing still with mouse movements. To eliminate such biases, we decompose keyboard and mouse inputs into atomic actions and ensure their balanced distribution. We also randomize the frame duration of each atomic action to avoid temporal bias. ❑ Diverse Scenes with Textual Descriptions. To enhance dataset diversity, we captured videos across different scenes, weather conditions, and times of day. We segmented the videos and annotated them with textual descriptions using an efficient multimodal LLM, MiniCPM [51].

- 4.2. Action Control Module

group window size

Figure 4. Due to temporal compression (compression ratio r = 4), the number of latent features differs from the number of actions, causing granularity mismatch during fusion. Grouping aligns these sequences for fusion. Additionally, the i-th latent feature can fuse with action groups within a previous window (window size w = 3), accounting for delayed action effects (e.g., ‘jump’ key affects several subsequent frames).

the input action includes both continuous mouse movement action M ∈ Rrn×d

1, and discrete keyboard action K ∈ Rrn×d

2. The intermediate feature in the Transformer is denoted as F ∈ R(n+1)×l×c. The total number of video frames is (1 + rn), compressed to (1 + n) latent frames, where r denotes the temporal compression ratio. d1 and d2 denote the dimension numbers of actions, l is the length of token sequence and c is the number of feature channels.

❑ Grouping Actions with a Sliding Window. Due to the temporal compression ratio r, the number of actions (rn) differs from the number of features (n + 1), creating a granularity mismatch for action-feature fusion. As shown in Fig. 4, we address this by grouping actions using a sliding window of size w. For the i-th feature fi, we consider actions within [ar×(i−w+1),...,ari]. This window design captures delayed action effects, such as how a jump command influences multiple subsequent frames. For out-of-range indices, boundary actions are used as padding. For mouse movement M, the grouped action is Mgroup ∈ R(n+1)×rw×d

1. As for keyboard actions K, we first learn the embedding of actions and add the positional encoding. After that, we perform a grouping operation on the action embeddings to get Kgroup ∈ R(n+1)×rw×c.

We incorporate action control modules into the transformer blocks of the video diffusion model to enable actioncontrollable generation. The architecture of the transformer block is shown in Fig. 3 (a), and the structure of the action control module is shown in Fig. 3 (b). Suppose that

❑ Mouse Movements Control. To fuse the grouped mouse action Mgroup with feature F, we first reshape it from R(n+1)×rw×d

1. Then we repeat it in the dimension of the token sequence length to get Mrepeat ∈

1 to R(n+1)×1×rwd

(a) Training Stage (b) Inference Stage

History Video Latents

0 ... k k+1 ... N

+ +

Autoregressive Generation

...

0 ... k ...

Video Diffusion Model (1 step)

...

Video Diffusion Model (T steps)

k+1 ... N

i i-th Clean Latent Noise

Training Loss

- Figure 5. Illustration of autoregressive video generation. The frames from index 0 to k serve as conditional frames, while the remaining N − k frames are for prediction, with k randomly selected. (a) Training stage: Loss computation and optimization focus only on the noise of predicted frames. (b) Inference stage: The model iteratively selects the latest k + 1 frames as conditions to generate N − k new frames, enabling autoregressive generation.

1, and concatenate it with F along with the channel dimension to get Ffused ∈ R(n+1)×l×(c+rwd

R(n+1)×l×rwd

1). After that, further learning on Ffused is conducted through a layer of MLP and a layer of temporal self-attention.

❑ Keyboard Actions Control. For discrete keyboard control, we perform a cross-attention calculation between the grouped action embeddings Kgroup ∈ R(n+1)×rw×c and F ∈ R(n+1)×l×c, similar to the prompt cross-attention between text and F. Specifically, Kgroup serves as the key and value in the attention, while F functions as the query.

#### 4.3. Autoregressive Generation of Long ActionControllable Game Videos

Current video generation methods introduced in Sec. 3 are limited to fixed-length outputs, which is inadequate for practical game applications requiring continuous video streams. To address this, we develop an autoregressive approach that generates multiple frames per step based on previous outputs, enabling efficient long video generation.

While video diffusion transformer [26, 31, 33, 52] offers superior generation quality, it typically operates in a full-sequence manner. Inspired by Diffusion Forcing [8], we modify the video diffusion transformer to enable autoregressive generation. Specifically, unlike standard diffusion models that require identical noise levels across frames, our approach allows different noise levels where later frames can have more noise while depending on earlier frames with less noise. This varying noise schedule ensures that earlier frames are generated first, allowing subsequent frames to be conditioned on them in an autoregressive manner.

As shown in Fig. 5 (a), during training with N +1 frame latents (from 0 to N), we randomly select the first k + 1 frames as conditions without adding noise, while adding noise only to the remaining N − k frames for noise prediction training. Although the first k+1 frames are input to the model, since they are assumed to be already generated, their predicted noise outputs are not utilized. To improve train-

ing efficiency, we focus on computing training losses only for the N − k frames that require prediction. As for inference, as shown in Fig. 5 (b), after full-sequence generation of the first N + 1 frame latents, we can autoregressively generate new N − k frames by selecting the most recent k + 1 frames from history video latents as conditions for each subsequent generation, and merge them into the history video latents. This process can be repeated to achieve infinite-length video generation. Unlike conventional nextframe generation methods [8, 11, 42], our approach supports multi-frame generation in one step, greatly reducing the time for long video generation.

### 5. Open-Domain Game Scene Generalization

This section introduces how to generalize learned action control capabilities to open-domain scenes. As analyzed in Sec. 1, while pre-trained video models offer rich generative priors for open-domain generation, directly fine-tuning them with game data leads to undesired style bias: the outputs inherit the visual style of training data while learning action control. To address this, we propose a Style-Action Decoupled Model with Domain Adapter (Sec. 5.1) and a multi-phase training strategy (Sec. 5.2).

#### 5.1. Style-Action Decoupling with Domain Adapter

To prevent action control capabilities from being bound to specific game styles, our key insight is to disentangle the learning of game style and action control through different modules and parameters. Specifically, while using the proposed action control module to learn action controllability, we employ an independent domain adapter to capture game-specific visual styles. The domain adapter is implemented using LoRA [21], which efficiently learns specific styles and can be plugged in and out without affecting the open-domain generation priors of the original model. To effectively train these decoupled components, we need a carefully designed training strategy that ensures the style adaptation and action control learning remain independent by learning them in different training phases. We detail this multi-phase training approach in the next section.

#### 5.2. Multi-Phase Training Strategy

As illustrated in Fig. 6, our training process consists of Phase #0 (model pretraining) and the following phases:

- ❑ Phase #1: Tune LoRA to Fit Game Videos. We finetune the pre-trained video diffusion model using LoRA to adapt it to specific game video while preserving most original parameters. This produces a model specialized for the target game domain. Better style adaptation in this phase allows the next phase to focus purely on action control, reducing style-control entanglement.
- ❑ Phase #2: Tune Action Control Module. We freeze both pre-trained parameters and LoRA, only training the

##### Phase #1 Phase #2 Phase #3

Phase #0

[Figure 62]

[Figure 63]

[Figure 64]

OpenDomain Results

[Figure 65]

OpenDomain Data

Action Control

[Figure 66]

Game Data

[Figure 67]

[Figure 68]

Game Data

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Trainable Frozen

[Figure 74]

LoRA

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Pretrained Model

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 6. Phase #0: pretraining a video generation model on open-domain data. Phase #1: finetuning with LoRA for game video data. Phase #2: training the action control module while fixing other parameters. Phase #3: inference for action-controlled opendomain generation. To decouple style learning from action control, Phase #1 learns game-specific style while Phase #2 focuses on style-independent action control. This design preserves the open-domain capabilities from Phase #0, enabling generalization in Phase #3.

DIAMOND [2] GameNGen [42] GameGenX [7] Oasis [11] Matrix [14] Genie 2 [12] GameFactory

|Release Time|NeurIPS 2024<br><br>|ICLR 2025|ICLR 2025<br><br>|2024.10.31<br><br>|2024.12.4|2024.12.4<br><br>|-|
|---|---|---|---|---|---|---|---|
|Game Sources<br><br>|Atari, CS:GO<br><br>|DOOM<br><br>|AAA Games<br><br>|Minecraft|AAA Games<br><br>|Unknown|Minecraft|
|Resolution<br><br>|280 × 150|240p|720p<br><br>|640 × 360|720p<br><br>|720p|640 × 360|
|Control Granularity|Frame-level<br><br>|Frame-level|Video-level|Frame-level|Frame-level<br><br>|Frame-level<br><br>|Frame-level|
|Technical Paper|✔<br><br>|✔|✔|✗<br><br>|✔<br><br>|✗|✔|
|Testable Model<br><br>|✔|✗|✗|✗<br><br>|✗|✗<br><br>|-|
|Available Dataset|✗<br><br>|✗<br><br>|✔|✗<br><br>|✗|✗<br><br>|✔|
|Action Space<br><br>|18 Keys<br><br>|Key|Instruction<br><br>|Key + Mouse|4 Keys<br><br>|Key+Mouse<br><br>|7 Keys+Mouse|
|Scene Generalizable<br><br>|✗|✗<br><br>|✗<br><br>|✗|✔<br><br>|✔|✔|

- Table 1. Comparison with recent related works. Scene-generalizable action control is the core contribution of GameFactory, being the only technical paper that validates on complex action space and conducts extensive testing across open-domain scenarios.

rate of the VAE is r = 4.

action control module with game videos and action signals. Since Phase #1 has handled style adaptation through LoRA, the training loss now focuses less on style learning. This allows the model to concentrate on action control learning, as it becomes the main contributor to minimizing the diffusion loss. Such separation enables style-independent control that can generalize to open-domain scenarios.

❑ Training and Inference Setting. Each phase of finetuning or training requires about two-four days of training on 8 A100 GPUs with a batch size of 64. The hyper parameters for LoRA finetuning can be referenced as rank = 128, with a learning rate of 1e-4. The learning rate for training the action control module is 1e-5. We only apply classifier free guidance [18] to the text prompt conditional input and use DDIM [39] sampling with 50 sampling steps.

❑ Phase #3: Inference on Open Domain. During inference, we remove the LoRA weights for game style adaptation, keeping only the action control module parameters. Thanks to the decoupling in previous phases, the action control module can now work independently of specific game styles, enabling controlled game video generation across open-domain scenarios.

❑ Evaluation Setting. We retained 5% of our collected, segmented dataset as a testset, excluding it from training, and selected three subsets for ablation study: (1) onlykey: contains only keyboard actions, designed to test the model’s ability to follow discrete actions; (2) mouse-small: includes small-scale continuous mouse movements; and (3) mouse-large: includes large-scale continuous mouse movements. Additionally, for qualitative experiments, we support custom combinations of input actions, allowing us to test complex or rare action combinations. We use these evaluation metrics: (1) Flow: calculates the optical flow of the generated video to reflect its dynamics, assessing action-following ability by measuring mean square error to the optical flow of the reference video; (2) Cam: com-

### 6. Experiments

#### 6.1. Implementation Details

❑ Pretrained Model Setting. Our experiments are based on an internal 1B-sized transformer-based text-to-video diffusion model for research purpose, which is distilled from a larger pretrained video diffusion model and possesses strong generative priors in the open domain. The resolution of the game videos is 360×640. The temporal compression

Control Module Only-Key Mouse-Small Mouse-Large Key Mouse Cam↓ Flow↓ CLIP↑ FID↓ FVD↓ Cam↓ Flow↓ CLIP↑ FID↓ FVD↓ Cam↓ Flow↓ CLIP↑ FID↓ FVD↓

Cross-Attn Cross-Attn 0.0527 8.67 0.3313 107.13 814.05 0.0798 20.46 0.3137 125.67 1203.29 0.1362 325.18 0.3103 167.37 1383.92

Concat Concat 0.0853 22.37 0.3277 103.89 786.50 0.0756 19.18 0.3159 133.42 1151.71 0.1179 258.93 0.3123 145.74 1405.47 Cross-Attn Concat 0.0439 7.79 0.3292 105.28 795.03 0.0685 18.64 0.3184 127.84 1032.98 0.1021 249.54 0.3107 139.91 1420.89

- Table 2. Results of the ablation study on action control mechanisms. The findings indicate that an optimal approach for the action control module is to use cross-attention for discrete action control and concatenation for continuous action control.

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

Mouse (Concat), Key (Cross-Attn)

Mouse (Cross-Attn), Key (Cross-Attn)

Mouse (Concat), Key (Concat) (BAD Results)

W A S D

W A S D

W A S D

W A S D

W A S D

W A S D

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

Figure 7. Qualitative comparison of key input control performance. The yellow buttons indicate pressed keys.

Strategy Domain Cam↓ Flow↓ Dom↑ CLIP↑ FID↓ FVD↓ Multi-Phase In- 0.0839 43.48 - - - Multi-Phase Open- 0.0997 54.13 0.7565 0.3181 121.18 1256.94

One-Phase Open- 0.1134 76.02 0.7345 0.3111 167.79 1323.58

- Table 3. Quantitative results of evaluation on scene generalization.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Original Model

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

Multi-Phase Training (Open-Domain Result)

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

One-Phase Training (Open-Domain Result)

Prompt: Standing in the cherry blossom forest with a gun in first person perspective.

Figure 8. Qualitative comparison of multi-phase training with onephase training for scene generalization. The arrows represent the direction of mouse movements.

and discrete keyboard control, comparing two typical methods: cross-attention and concatenation. The results are presented in Tab. 2 and Fig. 7. For discrete keyboard inputs, cross-attention outperforms concatenation. As shown in Fig. 7, concatenating the action control signal with the input leads to poor action following performance. This suggests that category-based signal control benefits from similaritybased cross-attention, as is often used in text-based control. In contrast, for continuous mouse movement, concatenation is more effective than cross-attention. This may be due to cross-attention’s similarity computation, which tends to reduce the influence of the control signal’s magnitude, thereby affecting the final result. Additionally, the values of Cam and Flow reveal that mouse movements have a larger impact on the visual output than keyboard inputs, particularly in the mouse-large test set where movement magnitude is greater. For the metrics of CLIP, FID and FVD, there is little difference between the methods. This is primarily because our multi-phase training strategy decouples visual style learning into Phase #1, where style learning is consistent across all methods. As a result, metrics assessing semantic consistency and generation quality yield similar performance across different methods.

putes the Euclidean distance between camera poses extracted from predicted videos and those extracted from reference videos, where both sets of camera poses are obtained using GLOMAP [32]. (3) CLIP: computes feature similarity in the CLIP [37] space to evaluate semantic relevance to the given text prompt; (4) FID, FVD: measures distribution differences between the generated videos and the reference, providing an assessment of generation quality.

❑ Comparison with Related Works. As shown in Tab. 1, given the diverse sources of game datasets, varying video resolutions, and different levels of control across methods, establishing a unified benchmark for comparison is challenging. Regarding our main contribution of scenegeneralizable action control, the only capable technical paper [14] validated their approach on a simple action space (left turn, right turn, acceleration in racing games). In contrast, we validate our method on a much more comprehensive action space (forward/backward/left/right movement, jumping, acceleration/deceleration, mouse movement) and provide extensive results in the supplementary materials.

#### 6.3. Scene Generalization

❑ Create new games in generalized scenes. In Fig. 1, we showcase a variety of newly generated game videos across open-domain scenes. These results inspire a future where generative game engines emerge as a new form of gaming, allowing players or game creators to generate and interact with anything they can imagine at minimal cost.

#### 6.2. Action Controllability

❑ Ablation Study. We perform ablation studies on the control mechanisms for continuous mouse movement

Dataset Cam↓ Flow↓ CLIP↑ FID↓ FVD↓ VPT [3] 0.1324 107.67 0.3174 156.69 1233.15

GF-Minecraft (ours) 0.0839 43.48 0.3135 125.85 1047.59

Loss Scope Cam↓ Flow↓ CLIP↑ FID↓ FVD↓ All frames 0.1547 148.73 0.2965 176.07 1592.43

Only predicted frames 0.0924 85.45 0.3190 136.95 1154.45

- Table 4. Comparison with Minecraft dataset with human bias.

Table 6. Ablation study on loss scope for long video generation.

Dataset W A S D Space Shift Ctrl VPT [3] 50.11% 4.03% 0.32% 3.45% 20.37% 0.14% 19.58%

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Ours 13.56% 13.56% 13.56% 13.56% 15.25% 15.25% 15.25%

Frame 1 Frame 26 Frame 51 Frame 76 Frame 101 Frame 126 Frame 151

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Table 5. The proportion of key inputs across different datasets.

Frame 176 Frame 201 Frame 226 Frame 251 Frame 276 Frame 301 Frame 326

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Figure 10. Demonstration of key frames in generated long video.

OursVPTOursVPT

remain still

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

pears over 100 times more frequently than the backward movement key (S), reflecting typical human gameplay patterns. Tab. 4 compares the action control capabilities of models trained on both datasets through in-domain evaluation, demonstrating clear advantages of our GF-Minecraft dataset in action following performance. Fig. 9 illustrates two representative examples: jumping in place and moving backward (both actions rarely performed by human players). The model trained on GF-Minecraft successfully follows these actions, while the VPT-trained model fails to execute them properly. Specifically, when commanded to jump in place, the VPT model incorrectly combines forward movement with jumping, and when instructed to move backward, it simply remains stationary.

Action: Hold the S key to move backward

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Walk and jump simultaneously

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Action: Press the Space key to jump in place several times

Figure 9. Compare the dataset on actions that are less commonly used by human players to test the effect of human bias in dataset.

❑ Comparison Results. As shown in Tab. 3, we evaluate the action following capability using Cam and Flow metrics on in-domain as baseline. The results demonstrate that our multi-phase training strategy achieves better action following performance closer to the baseline compared to the one-phase approach. Furthermore, the multi-phase strategy shows superior performance across text-alignment and generation quality metrics including CLIP, FID, and FVD. The Dom metric, which measures the CLIP space similarity between videos generated by the original model and the finetuned model, indicates that the multi-phase trained model maintains a domain closer to the original model. This is visually confirmed in Fig. 8, where the multi-phase strategy preserves the original model’s generation domain without style leakage. Additionally, the one-phase approach degrades the generated video quality with noticeable artifacts.

#### 6.5. Evaluation for Long Video Generation

In Tab. 6, we compare different scopes for computing loss functions in long video generation training: calculating losses across all frames versus only on frames that need to be predicted. The results demonstrate that computing losses solely on frames requiring prediction yields better performance. This improvement can be attributed to the elimination of noise interference from previously generated frames, as learning from such noise is irrelevant for future video generation. As demonstrated in Fig. 5, our model successfully generates long videos exceeding 300 frames in length. Additional examples of long video generation can be found in the supplementary materials.

#### 6.4. Evaluation for GF-Minecraft Dataset

In Sec. 4.1, we introduced our GF-Minecraft dataset, which eliminates human bias by ensuring uniform distribution of atomic actions. This design enables models trained on GF-Minecraft to effectively respond to actions that human players rarely perform. We compare our dataset with the VPT [3] dataset, which contains recordings of human gameplay and inherent human biases. Specifically, we selected video clips from VPT’s Find Cave dataset, as it closely matches our setting by primarily excluding inventory management and block modification operations. Tab. 5 compares the usage proportions of keyboard inputs across both datasets. The VPT dataset shows highly skewed distributions. For instance, the forward movement key (W) ap-

### 7. Conclusion

In this paper, we propose GameFactory, a framework that uses generative interactive videos to create new games, addressing significant gaps in existing research, particularly in scene generalization. The study of generative game engines still faces many challenges, including the design of diverse levels and gameplay, player feedback systems, ingame object manipulation, long-context memory, and realtime game generation. GameFactory marks our first effort in this field, and we aim to continue progressing toward the realization of a fully capable generative game engine.

### References

- [1] Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning. Advances in Neural Information Processing Systems, 36, 2024. 14

- [2] Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Fran¸cois Fleuret. Diffusion for world modeling: Visual details matter in atari. In Thirty-eighth Conference on Neural Information Processing Systems, 2024. 2, 3, 6

- [3] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (VPT): Learning to act by watching unlabeled online videos. In Advances in Neural Information Processing Systems, 2022. 3, 4, 8

- [4] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. arXiv preprint arXiv:2412.03572, 2024. 13

- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3

- [6] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 2, 3

- [7] Haoxuan Che, Xuanhua He, Quande Liu, Cheng Jin, and Hao Chen. Gamegen-x: Interactive open-world game video generation. In International Conference on Learning Representations, 2025. 2, 3, 6

- [8] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion, 2024. 5
- [9] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024. 3
- [10] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023. 3

- [11] Etched Decart. Oasis: A universe in a transformer. https: //oasis-model.github.io/, 2024. 2, 3, 5, 6
- [12] Google DeepMind. Genie 2: A large-scale foundation world model. https://deepmind.google/discover/blog/genie2-a-large-scale-foundation-world-model/, 2024. 2, 3, 6
- [13] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In Thirty-sixth Conference on Neural Information

- Processing Systems Datasets and Benchmarks Track, 2022. 2, 12
- [14] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024. 3, 6, 7

- [15] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024. 13

- [16] Yuwei Guo, Ceyuan Yang, Anyi Rao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Sparsectrl: Adding sparse controls to text-to-video diffusion models. In European Conference on Computer Vision, pages 330–348. Springer, 2025. 3

- [17] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 3

- [18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6

- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 2020. 3

- [20] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 13

- [21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations,

2022. 5

- [22] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023. 14

- [23] Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Martin Grayson, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao, et al. World and human action models towards gameplay ideation. Nature, 638(8051):656–663, 2025. 3

- [24] Seung Wook Kim, Yuhao Zhou, Jonah Philion, Antonio Torralba, and Sanja Fidler. Learning to simulate dynamic environments with gamegan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1231–1240, 2020. 3

- [25] Seung Wook Kim, Jonah Philion, Antonio Torralba, and Sanja Fidler. Drivegan: Towards a controllable highquality neural simulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2021. 3

- [26] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan. https://github.com/PKU-YuanGroup/OpenSora-Plan, 2024. 2, 3, 5

- [27] Willi Menapace, Stephane Lathuiliere, Sergey Tulyakov, Aliaksandr Siarohin, and Elisa Ricci. Playable video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10061– 10070, 2021. 3

- [28] Willi Menapace, St´ephane Lathuili`ere, Aliaksandr Siarohin, Christian Theobalt, Sergey Tulyakov, Vladislav Golyanik, and Elisa Ricci. Playable environments: Video manipulation in space and time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3584–3593, 2022.

- [29] Willi Menapace, Aliaksandr Siarohin, St´ephane Lathuili`ere, Panos Achlioptas, Vladislav Golyanik, Sergey Tulyakov, and Elisa Ricci. Promptable game models: Text-guided game simulation via masked diffusion models. ACM Transactions on Graphics, 2024. 3

- [30] Haomiao Ni, Changhao Shi, Kai Li, Sharon X Huang, and Martin Renqiang Min. Conditional image-to-video generation with latent flow diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18444–18455, 2023. 3

- [31] OpenAI. Creating video from text. https://openai. com/index/sora/, 2024. 2, 3, 5
- [32] Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Johannes L Sch¨onberger. Global structure-from-motion revisited. In European Conference on Computer Vision, 2024. 7

- [33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023. 3, 5

- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 3

- [35] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072, 2024. 13

- [36] Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, and Jing Shao. Mp5: A multi-modal open-ended embodied system in minecraft via active perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16307–16316, 2024. 14

- [37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, 2021. 7

- [38] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022. 3

- [39] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 6

- [40] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 2019. 3

- [41] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. International Conference on Learning Representations, 2021. 3

- [42] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024. 2, 3, 5, 6

- [43] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, 2024. 3

- [44] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors,

2023. 3

- [45] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 3

- [46] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 2023. 13

- [47] Mingyu Yang, Junyou Li, Zhongbin Fang, Sheng Chen, Yangbin Yu, Qiang Fu, Wei Yang, and Deheng Ye. Playable game generation. arXiv preprint arXiv:2412.00887, 2024. 2, 3

- [48] Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video: Customized video generation with userdirected camera movement and object motion. In ACM SIGGRAPH 2024 Conference Papers, pages 1–12, 2024. 3

- [49] Sherry Yang, Jacob C Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. Position: Video as the new language for real-world decision making. In Proceedings of the 41st International Conference on Machine Learning, 2024. 3

- [50] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3

- [51] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 4, 12

- [52] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. https://github.com/hpcaitech/OpenSora, 2024. 2, 3, 5

[53] Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. Irasim: Learning interactive realrobot action simulators. arXiv preprint arXiv:2406.14540, 2024. 13

Additional results of action control in Minecraft and open-domain scenarios can be found in our homepage at https: //yujiwen.github.io/gamefactory/.

### Appendix A. Details of Minecraft Dataset

- A.1. Basic Information

For data collection, we use Minedojo [13] to obtain Minecraft snapshots which contain a wide array of details within the agent’s surroundings. To mitigate biases introduced by human player habits, we uniformly sampled the frequency and duration of each action and randomly combined them to enhance action generalization (e.g., pressing two keys simultaneously or combining keyboard and mouse operations). Subsequently, the agent was connected to the MineDojo simulation environment to execute the action sequences and collect corresponding observations.

To further enhance the diversity and generalization of the generated game scenes, we preconfigured three biomes (forest, plains, desert), three weather conditions (clear, rain, thunder), and six different times of day (”Starting of a day,” ”Noon, sun is at its peak,” ”Sunset,” ”Beginning of night,” ”Midnight, moon is at its peak,” ”Beginning of sunrise”). This process resulted in the generation of 2,000 video clips with action annotations, each containing 2,000 frames.

- A.2. Data PreProcessing

After obtaining the full video, we preprocess it in two steps: (1) Slicing: Randomly sample consecutive sequences of k frames from the full video to form new video clips. Sampling continues until the total number of sampled frames reaches n times the total frame count of the original video. In our experimental setup, k = 81, which is slightly larger than the generation length of the video diffusion model (77), and n = 3; (2) Text Annotation: We use the open-source multimodal large language model MiniCPM-V [51] to annotate the sliced video clips with text. The corresponding prompts and annotation examples are shown in Fig. 11.

Q: Describe the scenery and objects in the Minecraft video. A: The video showcases a desert biome in the game Minecraft, characterized by sandy terrain and sparse vegetation. The environment includes sand blocks forming the ground and cacti scattered around as the only plant life visible. In the background, there are rock formations that add to the arid landscape's ruggedness. A notable feature is a green, pixelated entity resembling a zombie-like figure with outstretched arms, which remains stationary throughout the video. The sky transitions from orange hues at the horizon to darker shades above, suggesting either sunrise or sunset. The player's HUD (Heads-Up Display) is consistently present, indicating health and inventory status, but no actions such as mining or crafting are depicted within these frames.

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Figure 11. An example of video clip annotation, where words describing scenes and objects are highlighted in red and bolded.

- A.3. Details of Action Space

We use the part of the action space of Minedojo [13] which encompasses nearly all actions available to human players, including keypresses, mouse movements, and clicks. We used keypresses and mouse movements as our control signal. The specific binary actions used in our setup are listed in Tab 7. Interface1 to Interface5 represent different MineDojo interfaces, where mutually exclusive actions (e.g., moving forward and backward) are assigned to the same interface. Mouse movements are represented by the offset of the mouse pointer from the center of the game region. For each frame in the video, we calculate the cumulative offset relative to the first frame, and this absolute offset is used as input to the model.

### Appendix B. Supplementary Experimental Results

#### B.1. Interaction in generative Minecraft videos.

For navigation agents, collision is one of the most critical physical interactions in a simulator. Since our data collection process involves randomly generated scenes, our dataset naturally includes numerous examples of collisions. In these cases,

Table 7. Details of Action Space. The term Control Signal refers to the raw input signals utilized for training purposes, while the Action Interface represents the corresponding interface in the MineDojo platform that maps these input signals to actionable commands.

|Behavior<br><br>|Control Signal<br><br>|Action Interface|
|---|---|---|
|forward<br><br>|W key<br><br>|Interface1|
|back|S key<br><br>|Interface1|
|left<br><br>|A key|Interface2|
|right|D key|Interface2|
|jump<br><br>|space key|Interface3|
|sneak<br><br>|shift key<br><br>|Interface3|
|sprint<br><br>|ctrl key|Interface3|
|vertical perspective movement|mouse movement(yaw)<br><br>|Interface4|
|horizontal perspective movement|mouse movement(pitch)<br><br>|Interface5|

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

collision & stop

Move left and encounter an obstacle

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

collision & stop

Move forward and hit a wall

- Figure 12. Demonstration of the learned response to collision, one of the most common interactions in Minecraft navigation. Note that the text below each video frame is a descriptive label of the content, not a text prompt provided to the model.

even when action inputs are provided, the agent’s behavior should ideally remain stationary. Such corner cases can impact the learning of the action control module, especially when data volume is limited. However, during inference, we observed that our model has developed collision detection ability and provides appropriate interaction feedback, as shown in Fig. 12.

#### B.2. More Inspiration from Examples of Racing Games

In Fig. 13, we discuss an intriguing example. Since our collected Minecraft data is from a first-person perspective, the learned action space primarily generalizes to first-person scenes. Here, we experimented with a racing game scenario using a prompt for a car. Interestingly, we observed that the model’s learned yaw control for the mouse seamlessly generalized to steering control in the racing game. Additionally, certain directional controls, such as moving backward or sideways, were diminished, an adaptation that aligns well with typical controls in a racing game, where these actions are rarely needed. This example not only highlights the strong generalization capabilities of our method but also raises the question: could there exist a larger, more versatile action space that encompasses a wider range of game controls, extending beyond firstperson and racing games? This example also leads us to wonder whether the racing game scenario could have applications in autonomous driving. If we were to collect an action dataset from an autonomous driving simulation environment, could a pre-trained model then generate unlimited open-domain autonomous driving data? Our exploration of scene generalization within generative game engines may hold valuable insights for other fields as well.

### Appendix C. Potential of Generalizable World Model

We propose that the GameFactory we have developed is not merely a tool for creating new games but a Generalizable World Model with far-reaching implications. This model has the capability to generalize physical knowledge learned from small-scale labeled datasets to open-domain scenarios, addressing challenges in areas like autonomous driving [15, 20] and embodied AI [4, 35, 46, 53], which also face limitations due to the lack of large-scale action-labeled datasets. The generalizable world model has two key applications from different perspectives:

• As a data producer: It transfers knowledge from small labeled datasets to open-domain scenarios, enabling the generation of diverse unlimited action-annotated data that closely approximates real-world complexity.

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

W

A S D Space

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

W

A S D Space

Prompt: On a racing track, from a first person perspective, one can see holding a steering wheel.

- Figure 13. Our model demonstrates the ability to generalize to a different game type, a racing game. Interestingly, the yaw control learned in Minecraft seamlessly transfers to steering control in the racing game, while unrelated actions, such as moving backward, left, or right, and pitch angle adjustments, automatically diminish.

• As a simulator: It provides an environment for directly training agents to perform real-world tasks [1, 22, 36], closely approximating real-world conditions. By enabling controlled and diverse scenario generation, including extreme situations that are difficult to capture in real-world data collection, it facilitates the development of policy models that are exposed to a wide range of environments and interactions, thereby improving their robustness and generalizability, and aiding in overcoming the challenges of sim-to-real transfer.

