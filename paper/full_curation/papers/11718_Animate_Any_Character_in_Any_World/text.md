# arXiv:2512.17796v2[cs.CV]24Jun2026

## CustomX: Unified Character, Action, and Scene Customization in Video World Models

Yitong Wang1,2⋆ , Fangyun Wei2⋆ , Hongyang Zhang3 , Bo Dai4† , and Yan Lu2

1Fudan University 2Microsoft Research 3University of Waterloo 4The University of Hong Kong wangyitong23@m.fudan.edu.cn {fawe, yanlu}@microsoft.com hongyang.zhang@uwaterloo.ca bdai@hku.hk https://snowflakewang.github.io/CustomX_Page/

Consistent Environment and Character Fidelity Rich Action Repertoire

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

Any Character Any Scene

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

[Figure 28]

[Figure 29]

User

Long-Horizon, Temporally Coherent Interaction Controllable Camera Behavior

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Run Forward Run to the Right Play a Guitar

Fig. 1: CustomX enables users to provide a 3D or multi-view character together with a 3DGS scene, supporting iterative, multi-condition control of character behaviors and active environment exploration through natural language–guided video generation. The system features: (1) Consistent Environment and Character Fidelity, ensuring visual and spatial coherence with the user-provided scene and character; (2) a Rich Action Repertoire covering a wide range of behaviors, including locomotion, gestures, and object-centric interactions; (3) Long-Horizon, Temporally Coherent Interaction, enabling iterative user interaction while maintaining continuity across generated clips; and (4) Controllable Camera Behavior, which explicitly incorporates camera control, analogous to navigating 3DGS views, to produce accurate, user-specified viewpoints.

Abstract. Recent advances in world models have greatly enhanced interactive environment simulation. Existing methods mainly fall into two categories: (1) static world generation models, which construct 3D environments without active agents, and (2) controllable-entity models,

⋆ Equal contribution. † Corresponding author.

which allow a single entity to perform limited actions in an otherwise uncontrollable environment. In this work, we introduce CustomX, leveraging the realism and structural grounding of static world generation while extending controllable-entity models to support user-specified characters capable of performing open-ended actions. Users can provide a 3DGS scene and a character, then use natural language to direct the character to perform diverse behaviors, ranging from basic locomotion to objectcentric interactions, while freely exploring the environment. CustomX synthesizes temporally coherent video clips that preserve visual fidelity with the provided scene and character, formulated as a conditional autoregressive video generation problem. Built upon a pre-trained video generator, our training strategy significantly enhances motion dynamics while maintaining generalization across actions and characters. Our evaluation covers a broad range of aspects, including visual quality, character consistency, action controllability, and long-horizon coherence.

Keywords: World Models · Controllable Video Generation · Video Generation Post-Training

### 1 Introduction

Recent advances in world models have led to substantial progress in simulating dynamic and interactive environments. Existing methods generally fall into two categories: (1) static world generation approaches [12, 58, 87], which construct explorable 3D environments but lack active agents; and (2) controllable-entity approaches [47, 89], which allow a single agent to execute only a limited set of actions, such as steering a vehicle along a predefined path [25,35,73], while leaving the environment itself uncontrollable. In this work, we propose an alternative framework that combines the strengths of both paradigms: leveraging the realism and structural grounding of static world generation while extending controllable-entity models to support user-specified characters capable of performing open-ended actions.

Specifically, users can provide a 3D Gaussian Splatting (3DGS) scene [44], which represents either a synthetic environment or a real-world reconstruction, along with a 3D or multi-view character. Through natural language instructions, users can control the character’s behavior and enable active exploration within the environment. At each iteration, the model generates a video clip that captures the evolving states of both the character and the environment, resulting in coherent and temporally consistent generation. We name our method CustomX. As illustrated in Fig. 1, CustomX exhibits several key capabilities:

- 1. Consistent Environment and Character Fidelity. The visual contents appearing in the generated video clips exhibit strong consistency in visual identity and spatial layout with the user-provided scene and character.
- 2. Rich Action Repertoire. Unlike previous works [18,35,47] that limit the controllable entity to basic locomotion, our model enables the character to perform up to hundreds of distinct actions, encompassing not only navigation

- behaviors (e.g., “moving forward”, “turning left”) but also body-language gestures (e.g., “waving hands”, “saluting”) and object-centric interactions (e.g., “making a phone call”, “playing a guitar”).
- 3. Text Instruction as the Interface. Users can directly guide the character through natural language commands.
- 4. Long-Horizon, Temporally Coherent Interaction. Users can interact with the model iteratively, generating new video clips that remain temporally consistent with previously produced sequences.
- 5. Controllable Camera Behavior. Our model supports flexible and intuitive camera control, allowing behaviors such as following a character’s trajectory or orbiting around it to achieve user-specified viewpoints. Unlike previous methods [33,34] that encode camera trajectories into Plücker embeddings [71] and inject them as conditioning signals into the generation network, our approach achieves camera control in a more geometrically grounded manner. Specifically, given a user-provided 3DGS scene and a defined camera path, we directly render a projection scene video along the specified trajectory. This rendered video serves as an explicit conditioning input, enabling the generation model to produce videos that accurately follow the desired camera motion.

We formulate the entire process as a conditional autoregressive video generation problem. Concretely, the objective is to synthesize a video clip at each iteration, conditioned on a set of multi-modal inputs including: (a) the userprovided scene and character, which establish the spatial and visual grounding; (b) a text instruction, which specifies the intended behavior of the character; and (c) the previously generated video clips, which serve as temporal references to ensure consistency across iterations.

Additionally, we adopt a pre-trained video generator [37] as the foundation of our framework. We find that fine-tuning it on a small dataset containing basic locomotion actions across diverse characters not only preserves the generalization ability of the pre-trained model but also enhances overall motion quality compared to the original generator. This phenomenon is analyzed in Sec. 4.2.

In our experiments, we comprehensively evaluate the proposed model from multiple perspectives, including: (a) visual quality, assessed using the WorldScore benchmark [22]; (b) character consistency, which measures the alignment between the character appearing in the generated video and the user-provided reference; (c) action control success rate, which quantifies how accurately the character’s behavior follows the input text instruction across a diverse set of up to around 150 actions; and (d) long-horizon generation quality, which evaluates the model’s ability to maintain temporal coherence and visual fidelity over extended interaction sequences. We compare CustomX with both video generation foundation models [45, 77, 88] and dedicated world models [18, 35, 47]. Experimental results show that our method consistently outperforms both categories across nearly all evaluation metrics.

### 2 Related Works

Controllable Video Generation. Recent foundation models for video generation [1,6,7,15,31,45,77,88] have greatly improved modality alignment across text, image, and video, enabling large-scale pre-training for both text-to-video and image-to-video synthesis. Building on these advances, subsequent research has pursued finer-grained controllability by introducing mechanisms such as explicit subject control [17, 20, 24, 40–42, 54, 57] and camera control [14, 26, 28, 30,60,68,79,84,85,94,95,101]. For subject control, a typical approach [37,52] is to extract visual embeddings from reference images and use them within a Multimodal Diffusion Transformer [23,64] to guide the generated subjects to remain consistent with the reference appearance. For camera control, one common practice [33,34] is to convert the camera path into Plücker embeddings [71] and inject them into the main network, guiding the synthesized video to follow the specified trajectory. In contrast, our model controls the camera by navigating through 3DGS views: given a 3DGS scene and a camera path, we render a projection video along the path, which conditions the generator to follow the desired motion faithfully.

Memory Mechanism in Video Generation. Recent works incorporate memory mechanisms to improve long-term spatial and temporal consistency in video generation. These approaches retrieve generation history to localize relevant content across modalities such as RGB [92] and depth [18], often using surfel-indexed view selection [48] or camera FOV overlap [81,92]. Other methods [38,39,80,91] maintain a global point cloud map during generation, enabling the model to identify and reuse the most relevant spatial regions, thereby maintaining coherence across continuously generated scenes. In our work, the video generation with memory mechanism is realized by conditioning on both the character and the 3DGS scene. The 3DGS scene serves as a spatial memory that explicitly encodes the geometric and appearance information of the environment, while the character provides dynamic cues for motion and interaction.

World Models for Static Scene Creation. Existing world models that generate static yet explorable environments can be broadly categorized into two types. The first type [5,13,16,21,25,35,43,47,61,73,75,76,83,89,93,98] stores the world implicitly within neural networks, using video generation models [6,45,77,88] to visualize it. Users provide navigation commands (e.g., “camera forward”) drawn from a predefined set of camera trajectories, and the model synthesizes new frames along this path while maintaining spatial consistency with past generations. The second type [50,70] explicitly constructs a 3DGS world, where multiview images are optimized to form a manipulable 3D representation, allowing users to render novel views from arbitrary camera poses. Further developments extend this paradigm by using panoramic inputs [87,99] or directly generating 3DGS representations from text or a single image through feed-forward networks [27,53,58,74,86], while others [19,32,72,78,97] integrate video generation to streamline and accelerate 3DGS creation. In this work, users can either create or specify a 3DGS scene before generation. When users do not provide one, we adopt Marble [12] to automatically generate a static 3DGS world.

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

F L B R

[Figure 44]

[Figure 45]

[Figure 46]

Video VAE

|F|
|---|

|B|
|---|

|L|
|---|

|R|
|---|

[Figure 47]

Encoder

[Figure 48]

Multi-View Character Tokens

Multi-View 2D Character

3D Character

Animation with Game Engine

: Addition : Concatenation

C

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Decoder

Denoised

Video VAE Encoder

Target Tokens

| |
|---|

| |
|---|

| |
|---|

###### …

| |
|---|

| |
|---|

| |
|---|

…

Target Tokens

Rendered Video

Segmentation and Inpainting

[Figure 55]

[Figure 56]

Multi-Modal Diffusion Transformer LoRA

[Figure 57]

[Figure 58]

[Figure 59]

…

|F|
|---|

|B|
|---|

|L|
|---|

|R|
|---|

| |
|---|

| |
|---|

| |
|---|

…

| |
|---|

| |
|---|

| |
|---|

Video VAE

| |
|---|

| |
|---|

| |
|---|

###### …

Noisy Target Tokens

Encoder

Scene Tokens

Scene Video

LLaVA Encoder

Projector

[Figure 60]

[Figure 61]

[Figure 62]

Video VAE

| |
|---|

| |
|---|

| |
|---|

###### …

| |
|---|

| |
|---|

| |
|---|

…

Encoder

Mask Tokens

###### Conditional Noisy

C

Text: “The Character

=

| |
|---|

| |
|---|

| |
|---|

…

Character Mask

is running.”

Tokens

| | |
|---|---|
|(a) Training Data Preprocessing|(b) Architecture<br><br>|

- Fig. 2: (a) Each training sample consists of a 3D character and a video depicting the character performing an action described by a short text. Through segmentation and inpainting, we obtain the corresponding scene video and character mask sequence. The VAE encoder is then applied to encode these inputs into tokens. (b) CustomX predicts target video tokens conditioned on scene, mask, text, and multi-view character tokens within a Multi-Modal Diffusion Transformer, trained using Flow Matching [55]. Refer to Fig. 3 for the training process of the auto-regressive mode, which enables iterative interaction with CustomX, and Fig. 4 for the inference.

### 3 Method

In this section, we illustrate the detailed design of our method. Given a pregenerated or real-world reconstructed 3DGS scene S and a user-specified 3D character C, CustomX enables users to iteratively control the character C via text instructions T within the scene S, generating long-horizon, temporally coherent video clips that remain visually consistent with both S and C.

We formulate this process as a conditional autoregressive video generation problem. At each iteration, CustomX synthesizes the current video clip conditioned on multiple multi-modal inputs: the previous generated clip, character representations, scene representations, and the current text instruction. Overviews of the training and inference pipelines are shown in Fig. 2 and Fig. 4, respectively.

#### 3.1 Training Data Pre-Processing

Training Set Construction. As shown in Fig. 2(a), our training data is GTAV [9], a game where players can control a character to perform basic actions such

as “run forward”. We record gameplay sequences and segment them into short video clips, ensuring that each clip (1) contains only a single action and (2) has a fixed length of 129 frames. For each clip V , we apply the following steps:

- 1. Character Segmentation. We use Grounded-SAM-2 [67] to segment characters and extract their bounding-box mask sequences, denoted as M.
- 2. Scene Inpainting. We remove the segmented characters and apply DiffuEraser [49] to fill the missing regions, yielding the inpainted scene video S.
- 3. Action Labeling. Each clip is then annotated with a concise text label T describing the action performed by the character, such as “The character is running forward”.

GTA-V also provides access to 3D character models used in the game. To ease the character modeling, we represent each character using four viewpoint renders [96]—front, left, right, and back—denoted as C = {CF,CL,CR,CB}.

Finally, each processed training sample is represented as a tuple {V ,S,M, T,C}, forming a structured dataset.

Token Extraction. As shown in Fig. 2(a), given a training sample {V ,S,M, T,C}, we adopt the video VAE encoder from HunyuanCustom [37] to extract tokens for the video V , scene S, and mask M. The resulting token sequences are denoted as TV , TS and TM, respectively. The video VAE encoder operates with a spatial downsampling rate of 8 and a temporal downsampling rate of 4.

Note that the video VAE encoder can also be applied to single images. Therefore, for the multi-view character C = {CF,CL,CR,CB}, we use the same encoder to extract tokens from each view, resulting in the multi-view character token set TC = {TCF

,TCB}.

,TCL

,TCR

Finally, to extract text tokens, following HunyuanCustom [37], we employ the multi-modal encoder LLaVA [56], which takes both the text instruction T and character tokens TC as input. The resulting encoded text tokens are denoted as TT . Implementation details are provided in the appendix.

Now, the training sample {V ,S,M,T,C} is fully encoded into the latent space as {TV ,TS,TM,TT ,TC}.

#### 3.2 Architecture

Training Objective. Fig. 2(b) illustrates the architecture of CustomX, whose backbone consists of a stack of full-attention Transformer blocks. We adopt Flow Matching [55] for model training, conditioned on multiple inputs (i.e., TS, TM, TT , and TC), to guide the generation process from pure noise to TV .

Concretely, given target video tokens TV , we first sample t ∈ [0,1] from a logit-normal distribution and initialize the noise x0 ∼ N(0,I) following Gaussian distribution. The intermediate sample xt = (1 − t)x0 + tTV is then obtained via linear interpolation. The model is trained to predict the velocity ut = dxt/dt by minimizing the mean squared error between the predicted velocity vt and the ground-truth velocity ut:

0,TV ∥vt − ut∥2 . (1)

L = Et,x

| |
|---|

| |
|---|

| |
|---|

Denoised Target Tokens …

Multi-Modal Diffusion Transformer

… …

…

|F|
|---|

|B|
|---|

|L|
|---|

|R|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Text

Multi-View

###### Augmented

Conditional

Tokens

Character Tokens

Preceding Video Tokens

Noisy Tokens

- Fig. 3: Illustration of the auto-regressive mode. The only difference from the original architecture in Fig. 2 is the addition of an extra conditioning input, i.e., the preceding video tokens. Note that a misalignment exists between training and inference: during training, the preceding video tokens are derived from ground-truth videos, whereas during inference, they are generated by the model itself. To mitigate this discrepancy, we add a small Gaussian noise to the preceding video tokens during training and refer to the resulting tokens as augmented preceding video tokens.

Condition Modeling. We incorporate multiple conditioning signals to guide the learning process, including text tokens TT , multi-view character tokens TC, scene tokens TS, and mask tokens TM. As illustrated in Fig. 2(b), to inject scene and mask priors, we directly fuse TS and TM into the noisy target tokens xt via: x′t = xt + Projector([TS;TM]), where [ ; ] denotes channel-wise concatenation. The Projector maps the concatenated tokens [TS;TM] to the same dimensionality as xt using a lightweight linear layer. We refer to the resulting x′t as the conditional noisy tokens.

At last, to integrate the text tokens TT and multi-view character tokens TC,

we concatenate TT , TC, and x′t along the sequence dimension. The concatenated sequence is then fed into the backbone, which is implemented as a stack of full-

attention Transformer blocks, to denoise xt under the supervision of Eq. (1).

Positional Embeddings. Following HunyuanCustom [37], no positional embeddings are added to the text tokens TT . A standard 3D-RoPE (over time, height, and width dimensions) is applied to the conditional noisy tokens x′t. For the multi-view character tokens TC = {TCF

,TCB}, each TCF

, TCL

, TCR

,TCL

,TCR

, and TCB

represents single-view character tokens. For each view, a shifted3D-RoPE is applied. Taking TCF

as an example, PET

(i,j) = 3D-RoPE(−1,i + w,j + h), (2) where (w,h) denotes the spatial size of TCF

CF

, and the shifts along the temporal and spatial dimensions are −1 and (w,h), respectively. For TCL

, TCR

and TCB

,

the spatial shift remains the same, while the temporal shifts are set to −2, −3, and −4, respectively.

#### 3.3 Auto-Regressive Mode

CustomX supports multi-round user interaction while maintaining temporal continuity and semantic coherence between adjacent video clips. To achieve this, we

[Figure 63]

Iterative

(a) User Configuration

(b) Scene Video Rendering

(c) CustomX Inference

Instruction: “The character is running forward.” Instruction Parsing

3DGS Scene

[Figure 64]

Instruction

[Figure 65]

Character

CustomX

Locomotion Camera Behavior

Gesture Object-Centric Action

Scene Video Anchor

[Figure 66]

User

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

|[Figure 73]<br><br>| |
|---|
<br><br>|
|---|

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Virtual Camera

[Figure 81]

[Figure 82]

Previously Generated Clip

Output

Camera Path Scene Video

Character

Character Anchor

- Fig. 4: Inference of CustomX. (a) Users first specify the inputs, including the character, 3DGS scene, virtual camera location, and character anchor. (b) The user-provided text instruction is parsed, and a corresponding camera path is generated. Applying this path to the 3DGS scene produces a rendered scene video. (c) CustomX then takes multiple inputs as conditions to generate the final output. Steps (b) and (c) can be performed iteratively, enabling temporally consistent, long-horizon interactions.

extend CustomX into an auto-regressive mode. Specifically, we divide the target video tokens TV along the temporal dimension into two parts: the first quarter, denoted as TV 1

, serves as the preceding video tokens, and the remaining three quarters, denoted as TV [2:4]

, serve as the new target video tokens. The model is trained to generate TV [2:4]

conditioned on both the preceding video tokens TV 1

and the other conditioning signals introduced in Sec. 3.2. As shown in Fig. 3, to incorporate the newly added condition TV 1

, we prepend it to the conditional noisy tokens. The fusion strategy for the other conditioning signals remains unchanged, as described in Sec. 3.2.

#### 3.4 Inference and Acceleration

Inference. Fig. 4 illustrates the inference pipeline of CustomX, which consists of three main stages:

- 1. User Configuration. Users first specify a character and a 3DGS scene. They can place a virtual camera at any desired viewpoint within the scene and define a single-frame bounding-box mask (i.e., an anchor) to indicate where the character should appear in the generated video. This anchor is automatically propagated to all subsequent frames, forming per-frame anchors and keeping the manual effort minimal. Users may also leverage existing tools to create the 3D character (e.g., Hunyuan3D [46]) or the 3DGS scene (e.g., World Labs Marble [12]).
- 2. Scene Video Rendering. Next, users provide a text instruction. The instruction is parsed into four categories: (a) locomotion, (b) gesture, (c) objectcentric action, and (d) camera behavior. Each category determines a different camera path. For (a), CustomX generates a camera trajectory consistent with the motion described in the text, for example, for “The character is running forward”, the camera follows a forward-moving path. For (b) and

- (c), the camera remains stationary. For (d), CustomX generates a trajectory matching the specified camera motion, for instance, for “The camera orbits around the character”, the camera follows a circular path around the character. CustomX then renders a scene video clip along the corresponding camera trajectory.
- 3. CustomX Inference. Finally, CustomX encodes the text instruction, character, scene, character anchor, and the previously generated clip (optional, for auto-regressive mode) into tokens using the encoders illustrated in Fig. 2. These tokens are then fed into the CustomX model to generate the final output.

Note that Steps 2 and 3 can be performed iteratively, enabling temporally consistent, long-horizon generation. Acceleration. To accelerate inference, we adopt DMD2 [90] to distill the original 30-step denoising model into a more efficient 4-step version.

- 4 Experiment

#### 4.1 Experimental Settings

Training Details. Our model is initialized from HunyuanCustom [37], which contains 13B parameters. We freeze the LLaVA encoder and the scene condition projector, and apply LoRA-style [36] fine-tuning to the backbone with a rank of 64. Two separate models are trained for 360P and 720P data using the AdamW optimizer with a learning rate of 1e-4, each for 5,000 steps under the ZeRO-2 strategy. The 360P model is trained on 8× NVIDIA H100 GPUs, while the 720P model, owing to its higher resolution, is trained on 8× NVIDIA B200 GPUs. Further details on training and acceleration are provided in the appendix.

Training Data. Following the data curation procedure described in Sec. 3.1, we construct a training set comprising 2,084 video samples featuring five characters. Each sample is annotated with text labels describing either locomotion actions ({“run forward”, “run to the left”, “run to the right”, “run backward”}) or camera behaviors ({“orbit”, “follow” }). A key observation of CustomX is that post-training on such simple locomotion and camera-behavior data can substantially enhance pre-trained models, improving both action dynamics and camera control capability. This is analogous to large language models [29, 82], where extensive pre-training endows rich world knowledge, while post-training, which is conducted with a much smaller dataset, serves to activate specific capabilities (e.g., in GPT-3 [63], post-training data is orders of magnitude smaller than the pre-training corpus). For each sample, we prepare two resolutions, 360P and 720P, to train models of different quality levels. Unless otherwise specified, the 360P version is used by default. Additional details are provided in the appendix. Evaluation. We evaluate our model across four aspects: (1) visual quality, (2) camera controllability, (3) action control capability and generalization to novel actions, and (4) character consistency on novel characters. For (1) and (2), we adopt the WorldScore [22] metrics to assess the generated samples. For (3), we

- Table 1: WorldScore metrics for evaluating generation quality, categorized into three groups: (1) controllability, (2) quality, and (3) dynamics. The static and dynamic scores are computed by aggregating metrics from these three groups. † denotes dedicated world models. Ctrl: Controllability; Align: Alignment; Const: Consistency; Photo: Photometric; Acc: Accuracy; Mag: Magnitude; Smooth: Smoothness.

WorldScore Controllability Quality Dynamics Static

Dynamic

Camera Ctrl

Object Ctrl

Content Align

3D Const

Photo Const

Style Const

Subjective Quality

Motion Acc

Motion Mag

Motion Smooth

Method

CogVideoX1.5-I2V (5B) [88] 60.08 56.77 42.13 100.00 31.12 68.65 81.43 77.86 19.39 54.92 24.58 67.62 HunyuanVideo-I2V (13B) [45] 56.43 55.14 27.30 100.00 13.70 58.24 93.22 91.97 10.58 59.36 24.46 72.58

- Wan2.1-I2V (14B) [77] 57.91 55.87 37.32 98.33 26.34 81.88 83.77 65.03 12.73 59.59 28.60 65.07

- Wan2.2-I2V (14B) [77] 54.52 51.74 24.79 100.00 24.03 57.95 59.44 98.91 16.51 59.60 38.93 37.26 Wan-VACE (14B) [41] 51.54 52.03 21.29 100.00 30.39 27.53 54.18 99.02 17.39 56.73 34.78 67.98 Wan-VACE (1.3B) [41] 50.75 49.38 31.12 100.00 11.54 29.66 61.00 97.67 24.26 55.87 29.48 53.17 HunyuanCustom (13B) [37] 62.64 61.11 47.19 100.00 72.07 48.06 31.40 97.47 25.84 59.04 24.05 89.56 DeepVerse† [18] 52.63 47.63 52.48 75.00 18.80 35.47 83.16 92.39 11.09 33.30 33.61 40.97 Hunyuan-GameCraft† [47] 69.92 57.77 77.45 83.33 51.16 85.91 82.12 93.39 16.11 16.65 31.83 39.77 Matrix-Game-2.0† [35] 52.26 43.98 15.10 99.17 12.38 60.29 68.35 97.60 12.92 3.07 47.36 23.59 CustomX (Ours) 84.64 77.22 88.91 100.00 75.73 83.57 87.68 98.91 57.72 61.08 24.12 94.47

measure the control success rate via human evaluation and report the CLIP [65] text-to-image similarity score, covering both the four seen locomotion actions and up to 142 novel actions. For (4), we assess character similarity between the ground-truth character and the generated one using 30 novel characters, evaluated by DINOv2 [62] and CLIP [65] scores. By default, we use the 360P version of our model. At each iteration, our model generates 96 frames, using the previously generated 33 frames as conditions when available. Unless otherwise noted, evaluations are conducted on the first generated clip.

#### 4.2 Main Results

Visual Quality Evaluation. We adopt the metrics proposed by WorldScore [22] to evaluate the visual quality of generated videos. WorldScore defines a suite of metrics categorized into three groups: Controllability, Quality, and Dynamics. The first two primarily assess visual fidelity in static regions, while the last evaluates motion quality in dynamic regions (i.e., the character in our case).

We compare our model with dedicated world models, including DeepVerse [18], Hunyuan-GameCraft [47], and Matrix-Game-2.0 [35], that focus on controlling the main entity, as well as with video generation base models listed in Tab. 1.

For each model, we use the first two metric groups to evaluate 60 generated videos covering 30 different characters and two camera behaviors, {“orbit”,“follow”}. The third metric group is evaluated on 146 videos, where each video features the same character performing a distinct action, either one of four locomotion actions {“run forward”, “run to the right”, “run to the left”, “run backward”} or one of 142 novel actions unseen during training. Note that (1) to the best of our knowledge, our model is the first framework to jointly integrate 3D or multi-view characters, action text prompts, scenes, and long-horizon generation, which necessitates an input configuration that naturally diverges from established baselines; (2) both foundation and dedicated models are evaluated

- Table 2: Action control and character consistency evaluation on general foundation models and dedicated world models†. Locomotion actions include “run forward”, “run to the left”, “run to the right”, and “run backward”, while richer actions encompass 142 gesture and object-centric actions. Note that the three world models restrict action control to locomotion only.

Action Control Character Consistency Success Rate (%) CLIP Text-Image Score

Method

DINOv2 Score

CLIP Score

Locomotion Actions

Richer Actions

Locomotion Actions

Richer Actions

CogVideoX1.5-I2V (5B) [88] 23.3 21.1 0.261 0.273 0.594 0.611 HunyuanVideo-I2V (13B) [45] 26.7 35.2 0.272 0.293 0.645 0.709

- Wan2.1-I2V (14B) [77] 26.7 64.8 0.267 0.302 0.627 0.678

- Wan2.2-I2V (14B) [77] 53.3 74.6 0.272 0.303 0.650 0.704 Wan-VACE (14B) [41] 43.3 73.2 0.270 0.303 0.398 0.541 Wan-VACE (1.3B) [41] 26.7 13.4 0.261 0.269 0.504 0.548 HunyuanCustom (13B) [37] 56.7 51.4 0.273 0.297 0.558 0.665 DeepVerse† [18] 6.7 - 0.259 - 0.291 0.523 Hunyuan-GameCraft† [47] 16.7 - 0.255 - 0.329 0.529 Matrix-Game-2.0† [35] 3.3 - 0.255 - 0.339 0.524 CustomX (Ours) 100.0 80.7 0.279 0.305 0.698 0.721

using their original control mechanisms for character and camera, without any modification; (3) the three dedicated world models only support locomotion actions, thus they are evaluated solely on those; and (4) for models requiring an initial image input, we use Google Gemini [4] to render the corresponding character within the scene as the initialization. The results are shown in Tab. 1.

Action Control and Generalization. In Tab. 2, we evaluate the model’s action control capability on both seen actions (four locomotion actions) and 142 novel actions (referred to as “richer actions”). The novel actions cover both gesture-based behaviors (e.g., “wave hands”) and object-centric interactions (e.g., “play a guitar”). For each model, we conduct 30 evaluations on locomotion actions and 142 evaluations on distinct novel actions using the same set of characters. We report the action control success rate via human evaluation and the CLIP textimage similarity score, computed as the average frame-wise text-image score.

The results in Tab. 2 reveal that our model outperforms the base model, HunyuanCustom [37]. This phenomenon can be interpreted through the lens of post-training in large language models [29,63,66,82], where fine-tuning typically does not disrupt the pre-trained representation space; rather, it adjusts the response style, for example, to make the outputs more helpful or harmless, while preserving the extensive knowledge acquired during pre-training. In our case, the structurally simple fine-tuning data, composed primarily of fundamental locomotion behaviors, serve to refine motion dynamics and align human embodiment representations, rather than to redefine the model’s generative space. Consequently, our fine-tuning strategy enhances motion fidelity and behavioral

[Figure 83]

- Fig. 5: Screenshot visualizations of videos generated by CustomX, showcasing different characters performing various novel actions across two scenes. Additional examples are provided in the appendix and supplementary videos.

[Figure 84]

- Fig. 6: Screenshot visualizations of a long video generated by CustomX, showcasing a character iteratively exploring a scene with diverse actions. Additional long-form results are available in the appendix and supplementary videos.

coherence while maintaining the broad semantic and generative generalization inherited from large-scale pre-training.

Character Consistency Evaluation. A key feature of our model is its ability to maintain consistent visual identity between the provided character and the one appearing in the generated videos. In Tab. 2, we evaluate character consistency using DINOv2 and CLIP scores. Both metrics measure the similarity between the generated and ground-truth characters. During evaluation, we crop the character region from each generated frame and compute the similarity score; the final result is averaged across all frames. Since our method takes multi-view inputs, each frame is scored by its maximum similarity to the ground-truth views. Each model is evaluated 30 times with different characters performing locomotion.

- Table 3: Using multi-view character inputs improves character consistency. By default, we employ all four views.

Character View DINOv2 Score CLIP Score Front 0.556 0.628

+Back 0.613 0.678 +Right and Left 0.698 0.721

Table 4: Using per-frame character anchors helps the model distinguish dynamic entities from the static scene, leading to higher DINOv2 and CLIP character consistency scores.

Character Anchor Type DINOv2 Score CLIP Score

w/o Anchor 0.477 0.529 w/ First-Frame Anchor 0.597 0.645

w/ Per-Frame Anchor 0.698 0.721

Visualization. Fig. 5 shows screenshots from the generated videos, featuring different characters performing various actions across diverse scenes. Fig. 6 illustrates a long-horizon generation example in which users interact with the model four times. The resulting video follows the specified character, scene, and text instructions, while maintaining temporal coherence across generated clips. Additional examples are provided in the appendix and supplementary videos.

#### 4.3 Ablation Study

Multi-View Character Condition. We compare our four-view charactercondition model with two baselines: a single-view model and a front–back-view model. To evaluate character consistency, we generate videos by instructing characters to run toward the front, left, right, and back, which naturally reveals their appearance from multiple viewpoints. We then compute DINOv2 and CLIP scores following the protocol in Sec. 4.2. We use the same dataset as that used in Tab. 2. As shown in Tab. 3, character consistency improves as more view inputs are used.

Character Anchor Condition. As illustrated in Fig. 2, we introduce an additional condition, namely a character mask, to enable the model to distinguish dynamic entities from static scenes. In the default training setting, a boundingbox mask, referred to as an anchor, is extracted for each frame. During inference, as shown in Fig. 4, users only need to specify a single-frame anchor, which is then propagated to all subsequent frames to form per-frame anchors. Tab. 4 compares our default model, which uses per-frame anchors, with two variants: (1) a model without any anchor during both training and inference, and (2) a model that uses only the first-frame anchor, without propagating it to subsequent frames, throughout training and inference.

Visual Conditions Enhance Long-Horizon Generation. We introduce two types of visual conditions, multi-view character and 3DGS scene, to ensure both character and scene consistency during generation. Beyond improving spatial coherence, these visual conditions also alleviate the issue of visual quality degradation over long-horizon generation. To validate this, we compare CustomX operating in the auto-regressive mode (see Sec. 3.3) against two variants: (1) using only the multi-view character condition, where the 3DGS scene condition is replaced with textual scene descriptions; and (2) using only the 3DGS scene

[Figure 85]

- Fig. 7: Using both visual conditions, namely the 3DGS scene and the multi-view character, significantly improves longhorizon visual quality across diverse video clips.

[Figure 86]

Fig. 8: Evaluation of the hybrid data training strategy. (a) Training solely on game data causes the model to inherit a game-engine rendering style in synthesized characters. (b) Incorporating real-world data improves photorealism and enables the model to capture physical effects, such as dynamic clothing wrinkles, which are absent in GTA-V.

condition, where the multi-view character condition is replaced with textual character descriptions. Fig. 7 presents a comparison with the two variants, using the CLIP-Aesthetic and DINOv2 scores as metrics to evaluate visual quality and character consistency, respectively. The evaluation protocol consists of 10 trials per model. In each trial, a different character is used to autoregressively generate 10 video clips, forming an iterative long-horizon generation process.

Game-Real Hybrid Data Enhances Real-World Character Fidelity and Physics. CustomX trained solely on GTA-V videos tends to inherit the gameengine rendering style, leading to stylized outputs even when conditioned on realworld characters. We therefore investigate whether incorporating real-world data into training can improve generation realism. Specifically, we collect additional 400 real-world videos using the same data collection pipeline as for GTA-V, except that the videos are captured from the real world rather than rendered by a game engine. As shown in Fig. 8, the model trained on the hybrid dataset achieves higher realism and captures physical effects present in the real world but absent in games, such as dynamic clothing wrinkles. Additional details of the real-world dataset and further results are provided in the appendix.

Acceleration. Our 13B-parameter base model generates a 93-frame 360P video in 121s on a single NVIDIA H100 using a 30-step denoising schedule. By applying DMD2 [90], we distill it into a 4-step version, reducing latency to 21s with only slight drops in DINOv2 (0.698→0.669) and CLIP-Aesthetic (5.665→5.583) scores. Latencies for higher resolutions are provided in the appendix.

### 5 Conclusion

We present CustomX, a novel framework that allows users to provide a character and a 3DGS scene, enabling iterative interaction for both character control and world exploration. Unlike prior controllable-entity models that restrict the agent to a small set of predefined actions, CustomX supports open-ended control over a wide range of behaviors through natural language commands. CustomX delivers substantial improvements in motion dynamics and character consistency over its base model, as validated across a broad set of metrics including visual quality, action controllability, character fidelity, and long-horizon generation capability. Limitations. As a controllable video generation framework, CustomX shares common limitations with existing video generation models. Although inference has been accelerated, it still does not operate in real time. Furthermore, the current model is limited to single-subject scenarios and cannot generate videos involving multiple simultaneously active characters.

### Acknowledgements

This work is supported in part by Microsoft Research, the HKU Startup Fund, and HKU Musketeers Foundation Institute of Data Science.

## Supplementary Material

- A More Implementation Details

- A.1 Text Token Extraction

Following HunyuanCustom [37], we use the multi-modal encoder LLaVA [56] to extract text tokens while incorporating the multi-view character images. Concretely, given the text instruction T and the character views C = {CF,CL,CR, CB}, we construct the following prompt:

“A character is [Action]. <SEP>The character front view looks like E(CF). <SEP>The character left view looks like E(CL). <SEP>The character back view looks like E(CB). <SEP>The character right view looks like E(CR).”

Here, [Action] denotes the action description, E(·) is the LLaVA image encoder, and <SEP>is the separation token used to distinguish text and visual modalities.

- A.2 More Training Details

The model is initialized from the pre-trained weights of HunyuanCustom [37], a multi-modal subject-driven variant of HunyuanVideo [45]. Core components, which include the LLaVA encoder, the scene-condition projector, and the MMDiT [23, 64] backbone, are kept frozen. Trainable parameters are introduced by injecting LoRA modules [36] with a rank of 64 into the attention query, key, value, and projection matrices, as well as into the fully connected layers. We optimize the model using AdamW [59] with a learning rate of 1e-4 and 500 warm-up steps. The scene condition is randomly dropped with a probability of 0.3, encouraging the model to rely more heavily on text and multi-view character references.

Our model supports two generation modes: (1) First-clip generation. The model generates 93 frames, corresponding to 24 video latents (because the VAE encoder temporally downsamples a video of N frames into (N − 1)/4 + 1 video latents). In this setting, no preceding clip is provided. (2) Auto-regressive clip generation. When previous clips already exist, the model conditions on the last 33 frames (9 video latents) of the preceding clip and generates 96 new frames (24 video latents).

- A.3 Inference Acceleration

To accelerate inference, we adopt the DMD2 distillation framework [90], employing teacher, student, and fake-score models initialized from our trained model. The teacher model remains fully frozen, while the student and fake-score models are fine-tuned using LoRA modules with a rank of 64. Following the DMD2 protocol, the fake-score model is updated every iteration, whereas the student model is updated once every five iterations. This setup—instantiating three 13B models while training two sets of LoRA parameters—incurs a substantial GPU memory overhead. As a result, DMD2 distillation is applied only to the 360P model and is trained for 4,000 steps on 8× NVIDIA B200 GPUs, using the ZeRO-3 optimization strategy to manage memory efficiently.

Table A1: Hybrid training with game and real-world data helps the model disentangle game-engine rendering from real-world visual characteristics, yielding higher DINOv2 and CLIP character consistency scores.

Training Data Type DINOv2 Score CLIP Score Game Data 0.686 0.718

+Real-World Data 0.755 0.729

### B More Experiments

#### B.1 Game-Real Hybrid Data Enhances Real-World Character Fidelity and Physics

As described in the main paper, training solely on GTA-V [9] videos introduces a challenge: the model tends to inherit the game-engine rendering style, causing synthesized characters to appear stylized even when conditioned on real-world multi-view characters at inference time. To improve the realism of generated real-world characters—while retaining the diverse and dynamic motion patterns learned from game data—we curate an additional real-world dataset and jointly train CustomX on both sources. Specifically, we record 400 video clips of real individuals performing the same set of locomotion actions as in the game dataset. These videos are processed using the same pipeline described in the main paper and standardized to 360P resolution.

The model is then trained on a hybrid dataset combining the aforementioned GTA-V and newly collected real-world videos. To help the model differentiate between rendered and real-world visual styles, we apply a simple datatagging strategy [100]: GTA-V samples are tagged with the keyword “rendered” (e.g., “The rendered character is running forward”), while real-world samples are tagged with “real” (e.g., “The real character is running forward”). Aside from this tagging mechanism, the overall training procedure remains identical to that described in the main paper.

To evaluate the effectiveness of the mixed-data strategy, we collect multi-view captures of two unseen real-world individuals. Following the evaluation protocol in the main paper, we compute DINOv2 [62] and CLIP [65] scores to evaluate the character consistency. As shown in Tab. A1, training on hybrid data produces measurable improvements in real-world character fidelity.

#### B.2 Inference Acceleration

To qualitatively assess the effectiveness of our inference acceleration strategy, Fig. A1 compares three variants: (1) the original model with a 30-step denoising schedule (no acceleration), (2) the accelerated model with a 4-step denoising schedule, and (3) the original model also restricted to 4 denoising steps (no acceleration but fewer steps). As illustrated in Fig. A1(a) and Fig. A1(b), our 4-step

[Figure 87]

- Fig. A1: Qualitative comparison of three models: (a) the original model with a 30-step denoising schedule (no acceleration), (b) the accelerated model with a 4-step schedule, and (c) the original model restricted to 4 steps (no acceleration but fewer steps). The results show that our 4-step model matches the visual quality of the original model while achieving a 7.5× inference speedup.

distilled model maintains visual fidelity on par with the original framework, while facilitating a 7.5× inference speedup. Conversely, executing the non-accelerated original model with a simple 4-step denoising results in a pronounced degradation of visual quality, as evidenced in Fig. A1(c).

#### B.3 Latencies for Higher-Resolution Inference

We further report the inference cost and performance for generating higherresolution outputs using 8× NVIDIA H100 GPUs. When producing a 93-frame video clip at 720P, the 720P model outperforms the 360P model in both DINOv2 [62] (0.698 → 0.704) and CLIP-Aesthetic [69] (5.665 → 5.887) metrics, with an observed latency of 159 seconds.

#### B.4 Overall Inference Latencies

While the main paper and the appendix Sec. B.3 focus on the inference overhead of CustomX, this section provides a comprehensive temporal analysis under cold-start conditions—specifically, scenarios where 3DGS scene assets and scene video rendering have not been pre-computed. Evaluated on a single NVIDIA H100 GPU, the total latency for synthesizing a 96-frame video at 360P resolution is partitioned into: (1) an initial one-time 3DGS scene generation phase (approximately 300 seconds via World Labs Marble [12] or roughly 9 seconds via FlashWorld [51]); (2) 3DGS rendering (approximately 0.695 seconds); (3) the accelerated model inference (approximately 21 seconds). We emphasize that: (a) Step (1) can be bypassed if pre-existing 3DGS assets are utilized; (b) Step (1) represents a one-time computational cost, whereas Steps (2) and (3) are executed iteratively; and (c) in the absence of our proposed acceleration framework, the identical 96-frame generation requires approximately 121 seconds.

[Figure 88]

- Fig. A2: Visualization of a character performing 84 randomly selected novel actions.

#### B.5 Computational Cost of 720P Video Generation

Owing to the substantial token count, generating high-resolution (720P) videos remains computationally demanding for current open-source video generation

foundation models. Specifically, Transformer-based architectures exhibit significant inference latency when processing long-sequence spatiotemporal data. For instance, synthesizing a 96-frame video at 720P resolution requires the processing of approximately 85,800 spatiotemporal tokens; this incurs a temporal overhead of roughly 151 seconds for HunyuanCustom-13B [37] and 182 seconds for Wan2.2-14B [77] on 8× NVIDIA H100 GPUs. We acknowledge this limitation for both existing video generators and our method. As detailed in the main paper, we provide models trained at both 360P and 720P resolutions. The 360P variant serves as the default configuration—facilitated by DMD2 [90] for accelerated inference—whereas the 720P model is reserved for scenarios requiring enhanced visual quality.

### C More Visualizations

#### C.1 Action Control and Generalization

In the main paper, we quantitatively report the success rate of controlling a character to perform 142 novel actions. In Fig. A2, we provide qualitative results by visualizing 84 randomly selected actions using the same character. Fig. A3 further illustrates the model’s generalization by showing 25 randomly selected actions—with text annotations—performed by a different character.

#### C.2 Scene Customization

Our model supports flexible scene customization. Using state-of-the-art 3DGS scene generators, users can create diverse environments and control any character to explore these worlds. In this work, most 3DGS scenes are sourced from World Labs Marble [12]. Fig. A4 shows examples of characters navigating a variety of generated worlds. The proposed model also exhibits compatibility to real-world environments. Fig. A5 demonstrates visualization built upon a publicly available reconstructed real-world 3DGS scene.

#### C.3 Character Customization

Our model demonstrates strong generalization in controlling previously unseen characters. Leveraging mature 3D character-generation tools—such as Hunyuan3D [10], Tripo [11], Meshy [8], and Rodin [2]—or sourcing assets from online repositories like Sketchfab [3], diverse 3D characters can be easily acquired and used directly for inference. Fig. A6 and Fig. A7 illustrate examples of these characters performing locomotion actions.

#### C.4 Long-Horizon Generation

Our model supports auto-regressive generation, enabling the creation of temporally coherent video sequences that build upon previously generated clips. This capability allows for extended, long-horizon user–model interactions. Fig. A8 and Fig. A9 present two examples of long-horizon generation.

[Figure 89]

- Fig. A3: Visualization of a character performing 25 randomly selected novel actions with text annotations.

[Figure 90]

Fig. A4: Visualization of a character exploring various 3DGS worlds.

### D Potential Negative Societal Impact

In line with existing video generation models, our framework may be repurposed to produce malicious content, highlighting the importance of ethical safeguards.

[Figure 91]

- Fig. A5: Visualization of a character performing novel actions in a publicly available reconstructed real-world 3DGS scene.

[Figure 92]

- Fig. A6: Visualization of diverse characters performing locomotion actions (Part 1).

### References

- 1. ByteDance Seed: Seedance 2.0. https://seed.bytedance.com/en/seedance2_0

(2026)

- 2. Deemos Technologies: Rodin. https://hyper3d.ai/ (2026)

[Figure 93]

- Fig. A7: Visualization of diverse characters performing locomotion actions (Part 2).

- 3. Epic Games: Sketchfab. https://sketchfab.com/feed (2026)
- 4. Google: Gemini. https://gemini.google.com/app (2026)
- 5. Google: Genie 3: A new frontier for world models. https://deepmind.google/ blog/genie-3-a-new-frontier-for-world-models/ (2026)
- 6. Google: Veo. https://deepmind.google/models/veo/ (2026)
- 7. Kuaishou: Kling ai. https://klingai.com/global/ (2026)
- 8. Meshy: Meshy ai. https://www.meshy.ai/ (2026)
- 9. Rockstar Games: Grand theft auto v. https://www.rockstargames.com/gta-v

(2026)

- 10. Tencent: Tencent hunyuan3d. https://3d.hunyuan.tencent.com/ (2026)
- 11. VAST AI Research: Tripo. https://www.tripo3d.ai/ (2026)
- 12. World Labs: Marble. https://marble.worldlabs.ai/ (2026)
- 13. World Labs: Rtfm: A real-time frame model. https://rtfm.worldlabs.ai/

(2026)

[Figure 94]

Fig. A8: Visualization of long-horizon generation (Example 1).

- 14. Bai, J., Xia, M., Fu, X., Wang, X., Mu, L., Cao, J., Liu, Z., Hu, H., Bai, X., Wan, P., et al.: Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647 (2025)
- 15. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023)

[Figure 95]

Fig. A9: Visualization of long-horizon generation (Example 2).

- 16. Bruce, J., Dennis, M.D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al.: Genie: Generative interactive environments. In: Forty-first International Conference on Machine Learning

(2024)

- 17. Chen, J., Li, X., Bai, X., Ma, T., Zhang, P., Chen, Z., Li, G., Liu, L., Zhao, S., Li, B., et al.: Omniinsert: Mask-free video insertion of any reference via diffusion transformer models. arXiv preprint arXiv:2509.17627 (2025)

- 18. Chen, J., Zhu, H., He, X., Wang, Y., Zhou, J., Chang, W., Zhou, Y., Li, Z., Fu, Z., Pang, J., et al.: Deepverse: 4d autoregressive video generation as a world model. arXiv preprint arXiv:2506.01103 (2025)
- 19. Chen, L., Zhou, Z., Zhao, M., Wang, Y., Zhang, G., Huang, W., Sun, H., Wen, J.R., Li, C.: Flexworld: Progressively expanding 3d scenes for flexiable-view synthesis. arXiv preprint arXiv:2503.13265 (2025)
- 20. Chen, T.S., Siarohin, A., Menapace, W., Fang, Y., Lee, K.S., Skorokhodov, I., Aberman, K., Zhu, J.Y., Yang, M.H., Tulyakov, S.: Multi-subject open-set personalization in video generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6099–6110 (2025)
- 21. Decart, E., McIntyre, Q., Campbell, S., Chen, X., Wachen, R.: Oasis: A universe in a transformer (2024)
- 22. Duan, H., Yu, H.X., Chen, S., Fei-Fei, L., Wu, J.: Worldscore: A unified evaluation benchmark for world generation. arXiv preprint arXiv:2504.00983 (2025)
- 23. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for highresolution image synthesis. In: Forty-first international conference on machine learning (2024)
- 24. Fei, Z., Li, D., Qiu, D., Wang, J., Dou, Y., Wang, R., Xu, J., Fan, M., Chen, G., Li, Y., et al.: Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436 (2025)
- 25. Feng, R., Zhang, H., Yang, Z., Xiao, J., Shu, Z., Liu, Z., Zheng, A., Huang, Y., Liu, Y., Zhang, H.: The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568 (2024)
- 26. Feng, W., Liu, J., Tu, P., Qi, T., Sun, M., Ma, T., Zhao, S., Zhou, S., He, Q.: I2vcontrol-camera: Precise video camera control with adjustable motion strength. arXiv preprint arXiv:2411.06525 (2024)
- 27. Go, H., Park, B., Jang, J., Kim, J.Y., Kwon, S., Kim, C.: Splatflow: Multi-view rectified flow model for 3d gaussian splatting synthesis. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 21524–21536 (2025)
- 28. Gu, Z., Yan, R., Lu, J., Li, P., Dou, Z., Si, C., Dong, Z., Liu, Q., Lin, C., Liu, Z., et al.: Diffusion as shader: 3d-aware video diffusion for versatile video generation control. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. pp. 1–12 (2025)
- 29. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)
- 30. Guo, Y., Yang, C., Rao, A., Liang, Z., Wang, Y., Qiao, Y., Agrawala, M., Lin, D., Dai, B.: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023)
- 31. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., Bibi, O.: Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103 (2024)
- 32. Hao, J., Wang, P., Wang, H., Zhang, X., Guo, Z.: Gaussvideodreamer: 3d scene generation with video diffusion and inconsistency-aware gaussian splatting. arXiv preprint arXiv:2504.10001 (2025)
- 33. He, H., Xu, Y., Guo, Y., Wetzstein, G., Dai, B., Li, H., Yang, C.: Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101 (2024)

- 34. He, H., Yang, C., Lin, S., Xu, Y., Wei, M., Gui, L., Zhao, Q., Wetzstein, G., Jiang, L., Li, H.: Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models. arXiv preprint arXiv:2503.10592 (2025)
- 35. He, X., Peng, C., Liu, Z., Wang, B., Zhang, Y., Cui, Q., Kang, F., Jiang, B., An, M., Ren, Y., et al.: Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009 (2025)
- 36. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al.: Lora: Low-rank adaptation of large language models. ICLR 1(2), 3

(2022)

- 37. Hu, T., Yu, Z., Zhou, Z., Liang, S., Zhou, Y., Lin, Q., Lu, Q.: Hunyuancustom: A multimodal-driven architecture for customized video generation. arXiv preprint arXiv:2505.04512 (2025)
- 38. Huang, J., Hu, X., Han, B., Shi, S., Tian, Z., He, T., Jiang, L.: Memory forcing: Spatio-temporal memory for consistent scene generation on minecraft. arXiv preprint arXiv:2510.03198 (2025)
- 39. Huang, T., Zheng, W., Wang, T., Liu, Y., Wang, Z., Wu, J., Jiang, J., Li, H., Lau, R.W., Zuo, W., Guo, C.: Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation. arXiv preprint arXiv:2506.04225 (2025)
- 40. Jiang, Y., Wu, T., Yang, S., Si, C., Lin, D., Qiao, Y., Loy, C.C., Liu, Z.: Videobooth: Diffusion-based video generation with image prompts. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6689–6700 (2024)
- 41. Jiang, Z., Han, Z., Mao, C., Zhang, J., Pan, Y., Liu, Y.: Vace: All-in-one video creation and editing. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17191–17202 (2025)
- 42. Ju, X., Ye, W., Liu, Q., Wang, Q., Wang, X., Wan, P., Zhang, D., Gai, K., Xu, Q.: Fulldit: Multi-task video generative foundation model with full attention. arXiv preprint arXiv:2503.19907 (2025)
- 43. Kanervisto, A., Bignell, D., Wen, L.Y., Grayson, M., Georgescu, R., Valcarcel Macua, S., Tan, S.Z., Rashid, T., Pearce, T., Cao, Y., et al.: World and human action models towards gameplay ideation. Nature 638(8051), 656–663 (2025)
- 44. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics 42(4) (July 2023)
- 45. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)
- 46. Lai, Z., Zhao, Y., Liu, H., Zhao, Z., Lin, Q., Shi, H., Yang, X., Yang, M., Yang, S., Feng, Y., et al.: Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint arXiv:2506.16504 (2025)
- 47. Li, J., Tang, J., Xu, Z., Wu, L., Zhou, Y., Shao, S., Yu, T., Cao, Z., Lu, Q.: Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition. arXiv preprint arXiv:2506.17201 (2025)
- 48. Li, R., Torr, P., Vedaldi, A., Jakab, T.: Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903

(2025)

- 49. Li, X., Xue, H., Ren, P., Bo, L.: Diffueraser: A diffusion model for video inpainting. arXiv preprint arXiv:2501.10018 (2025)
- 50. Li, X., Lai, Z., Xu, L., Qu, Y., Cao, L., Zhang, S., Dai, B., Ji, R.: Director3d: Real-world camera trajectory and 3d scene generation from text. Advances in neural information processing systems 37, 75125–75151 (2024)

- 51. Li, X., Wang, T., Gu, Z., Zhang, S., Guo, C., Cao, L.: Flashworld: High-quality 3d scene generation within seconds. arXiv preprint arXiv:2510.13678 (2025)
- 52. Li, Z., Qian, D., Su, K., Diao, Q., Xia, X., Liu, C., Yang, W., Zhang, T., Yuan, Z.: Bindweave: Subject-consistent video generation via cross-modal integration. arXiv preprint arXiv:2510.00438 (2025)
- 53. Liang, H., Cao, J., Goel, V., Qian, G., Korolev, S., Terzopoulos, D., Plataniotis, K.N., Tulyakov, S., Ren, J.: Wonderland: Navigating 3d scenes from a single image. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 798–810 (2025)
- 54. Liang, S., Yu, Z., Zhou, Z., Hu, T., Wang, H., Chen, Y., Lin, Q., Zhou, Y., Li, X., Lu, Q., et al.: Omniv2v: Versatile video generation and editing via dynamic content manipulation. arXiv preprint arXiv:2506.01801 (2025)
- 55. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)
- 56. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023)
- 57. Liu, L., Ma, T., Li, B., Chen, Z., Liu, J., Li, G., Zhou, S., He, Q., Wu, X.: Phantom: Subject-consistent video generation via cross-modal alignment. arXiv preprint arXiv:2502.11079 (2025)
- 58. Liu, Y., Min, Z., Wang, Z., Wu, J., Wang, T., Yuan, Y., Luo, Y., Guo, C.: Worldmirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726 (2025)
- 59. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)
- 60. Luo, Y., Bai, J., Shi, X., Xia, M., Wang, X., Wan, P., Zhang, D., Gai, K., Xue, T.: Camclonemaster: Enabling reference-based camera control for video generation. arXiv preprint arXiv:2506.03140 (2025)
- 61. Mao, X., Lin, S., Li, Z., Li, C., Peng, W., He, T., Pang, J., Chi, M., Qiao, Y., Zhang, K.: Yume: An interactive world generation model. arXiv preprint arXiv:2507.17744 (2025)
- 62. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 63. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022)
- 64. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205

(2023)

- 65. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021)
- 66. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems 36, 53728–53741 (2023)
- 67. Ren, T., Liu, S., Zeng, A., Lin, J., Li, K., Cao, H., Chen, J., Huang, X., Chen, Y., Yan, F., et al.: Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159 (2024)

- 68. Ren, X., Shen, T., Huang, J., Ling, H., Lu, Y., Nimier-David, M., Müller, T., Keller, A., Fidler, S., Gao, J.: Gen3c: 3d-informed world-consistent video generation with precise camera control. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6121–6132 (2025)
- 69. Schuhmann, C.: Clip+mlp aesthetic score predictor (2026)
- 70. Shriram, J., Trevithick, A., Liu, L., Ramamoorthi, R.: Realmdreamer: Textdriven 3d scene generation with inpainting and depth diffusion. arXiv preprint arXiv:2404.07199 (2024)
- 71. Sitzmann, V., Rezchikov, S., Freeman, B., Tenenbaum, J., Durand, F.: Light field networks: Neural scene representations with single-evaluation rendering. Advances in Neural Information Processing Systems 34, 19313–19325 (2021)
- 72. Sun, W., Chen, S., Liu, F., Chen, Z., Duan, Y., Zhang, J., Wang, Y.: Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928 (2024)
- 73. Sun, W., Wei, F., Zhao, J., Chen, X., Chen, Z., Zhang, H., Zhang, J., Lu, Y.: From virtual games to real-world play. arXiv preprint arXiv:2506.18901 (2025)
- 74. Szymanowicz, S., Zhang, J.Y., Srinivasan, P., Gao, R., Brussee, A., Holynski, A., Martin-Brualla, R., Barron, J.T., Henzler, P.: Bolt3d: Generating 3d scenes in seconds. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 24846–24857 (2025)
- 75. Team, R., Gao, Z., Wang, Q., Zeng, Y., Zhu, J., Cheng, K.L., Li, Y., Wang, H., Xu, Y., Ma, S., Chen, Y., Liu, J., Cheng, Y., Yao, Y., Zhu, J., Meng, Y., Zheng, K., Bai, Q., Chen, J., Shen, Z., Yu, Y., Zhu, X., Shen, Y., Ouyang, H.: Advancing open-source world models. arXiv preprint arXiv:2601.20540 (2026)
- 76. Valevski, D., Leviathan, Y., Arar, M., Fruchter, S.: Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837 (2024)
- 77. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,

- X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
- Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

- 78. Wang, H., Liu, F., Chi, J., Duan, Y.: Videoscene: Distilling video diffusion model to generate 3d scenes in one step. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16475–16485. IEEE (2025)
- 79. Wang, Z., Yuan, Z., Wang, X., Li, Y., Chen, T., Xia, M., Luo, P., Shan, Y.: Motionctrl: A unified and flexible motion controller for video generation. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–11 (2024)
- 80. Wu, T., Yang, S., Po, R., Xu, Y., Liu, Z., Lin, D., Wetzstein, G.: Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284 (2025)
- 81. Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., Pan, X.: Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369 (2025)
- 82. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)
- 83. Yang, M., Li, J., Fang, Z., Chen, S., Yu, Y., Fu, Q., Yang, W., Ye, D.: Playable game generation. arXiv preprint arXiv:2412.00887 (2024)

- 84. Yang, S., Hou, L., Huang, H., Ma, C., Wan, P., Zhang, D., Chen, X., Liao, J.: Direct-a-video: Customized video generation with user-directed camera movement and object motion. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–12

(2024)

- 85. Yang, X., Xu, J., Luan, K., Zhan, X., Qiu, H., Shi, S., Li, H., Yang, S., Zhang, L., Yu, C., et al.: Omnicam: Unified multimodal video generation via camera control. arXiv preprint arXiv:2504.02312 (2025)
- 86. Yang, Y., Shao, J., Li, X., Shen, Y., Geiger, A., Liao, Y.: Prometheus: 3d-aware latent diffusion models for feed-forward text-to-3d scene generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2857–2869

(2025)

- 87. Yang, Z., Ge, W., Li, Y., Chen, J., Li, H., An, M., Kang, F., Xue, H., Xu, B., Yin, Y., et al.: Matrix-3d: Omnidirectional explorable 3d world generation. arXiv preprint arXiv:2508.08086 (2025)
- 88. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)
- 89. Ye, D., Zhou, F., Lv, J., Ma, J., Zhang, J., Lv, J., Li, J., Deng, M., Yang, M., Fu, Q., et al.: Yan: Foundational interactive video generation. arXiv preprint arXiv:2508.08601 (2025)
- 90. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T.: Improved distribution matching distillation for fast image synthesis. In: NeurIPS (2024)
- 91. Yu, H.X., Duan, H., Hur, J., Sargent, K., Rubinstein, M., Freeman, W.T., Cole, F., Sun, D., Snavely, N., Wu, J., et al.: Wonderjourney: Going from anywhere to everywhere. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6658–6667 (2024)
- 92. Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., Liu, X.: Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141 (2025)
- 93. Yu, J., Qin, Y., Wang, X., Wan, P., Zhang, D., Liu, X.: Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325

(2025)

- 94. YU, M., Hu, W., Xing, J., Shan, Y.: Trajectorycrafter: Redirecting camera trajectory for monocular videos via diffusion models. arXiv preprint arXiv:2503.05638

(2025)

- 95. Yu, W., Xing, J., Yuan, L., Hu, W., Li, X., Huang, Z., Gao, X., Wong, T.T., Shan, Y., Tian, Y.: Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024)
- 96. Zhang, L., Wang, Z., Zhang, Q., Qiu, Q., Pang, A., Jiang, H., Yang, W., Xu, L., Yu, J.: Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43(4), 1–20 (2024)
- 97. Zhang, S., Li, J., Fei, X., Liu, H., Duan, Y.: Scene splatter: Momentum 3d scene generation from single image with video diffusion model. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 6089–6098 (2025)
- 98. Zhang, Y., Peng, C., Wang, B., Wang, P., Zhu, Q., Kang, F., Jiang, B., Gao, Z., Li, E., Liu, Y., et al.: Matrix-game: Interactive world foundation model. arXiv preprint arXiv:2506.18701 (2025)
- 99. Zhang, Z., Hold-Geoffroy, Y., Hašan, M., Chen, Z., Luan, F., Dorsey, J., Hu, Y.: Generating 360◦ video is what you need for a 3d scene. arXiv preprint arXiv:2504.02045 (2025)

- 100. Zhao, Q., Ni, X., Wang, Z., Cheng, F., Yang, Z., Jiang, L., Wang, B.: Synthetic video enhances physical fidelity in video synthesis. arXiv preprint arXiv:2503.20822 (2025)
- 101. Zhou, J., Gao, H., Voleti, V., Vasishta, A., Yao, C.H., Boss, M., Torr, P., Rupprecht, C., Jampani, V.: Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489 (2025)

