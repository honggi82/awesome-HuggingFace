# arXiv:2601.05241v1[cs.CV]8Jan2026

## RoboVIP: Multi-View Video Generation with Visual Identity Prompting Augments Robot Manipulation

Boyang Wang1*, Haoran Zhang4*, Shujie Zhang1,2*, Jinkun Hao1,3, Mingda Jia1, Qi Lv1, Yucheng Mao1, Zhaoyang Lyu1, Jia Zeng1, Xudong Xu1†, Jiangmiao Pang1 1Shanghai AI Laboratory 2Tsinghua University 3Shanghai Jiao Tong University 4University of Michigan Homepage: https://robovip.github.io/RoboVIP/

### Abstract

The diversity, quantity, and quality of manipulation data are critical for training effective robot policies. However, due to hardware and physical setup constraints, collecting large-scale real-world manipulation data remains difficult to scale across diverse environments. Recent work uses text-prompt conditioned image diffusion models to augment manipulation data by altering the backgrounds and tabletop objects in the visual observations. However, these approaches often overlook the practical need for multi-view and temporally coherent observations required by state-ofthe-art policy models. Further, text prompts alone cannot reliably specify the scene setup. To provide the diffusion model with explicit visual guidance, we introduce visual identity prompting, which supplies exemplar images as conditioning inputs to guide the generation of the desired scene setup. To this end, we also build a scalable pipeline to curate a visual identity pool from large robotics datasets. Using our augmented manipulation data to train downstream vision-language-action and visuomotor policy models yields consistent performance gains in both simulation and real-robot settings.

### 1. Introduction

High-quality and diverse visual data remains fundamental to progress in robotic manipulation and policy learning [7, 25, 42]. However, collecting such data in the real world is notoriously challenging: each episode requires precise mechanical setups, calibrated camera rigs, and reliable synchronization across sensing devices. These constraints make it difficult to scale manipulation datasets in both quantity and environmental diversity. As a complementary solution, recent work has turned to generative models [18, 40] to synthesize additional data, offering a promising alternative to labor-intensive data collection [11, 52, 53].

*Indicates co–first author. †Corresponding author.

A growing line of work [11, 52, 53] augments visual observations in manipulation data while keeping the underlying action trajectory fixed. They segment the robot arm and the interacted objects, then apply text prompt-guided image generative models [37] to inpaint masked regions, diversifying backgrounds and table-top contents. However, these approaches typically operate in a single-frame, single-view setting, which diverges from the needs of modern policy models [7, 12, 25, 42]. First, many manipulation tasks inherently require reasoning over longer temporal histories, rather than relying solely on a single observation. Consider a policy model executing a push-button task where the preinteraction and post-interaction states of the button appear visually identical. With only one historical frame, the policy model cannot tell whether it has already pushed the button or is about to, often producing indecisive behaviors, even action loops. Second, multi-view observations are increasingly adopted in visuomotor policy models [12, 56] and VLA systems [7, 25, 42], and are usually provided in recent robotics datasets [14, 24], as they provide richer spatial cues and better cross-view generalization. As a result, a practical augmentation framework should operate at the video level and support multi-view generation.

In this work, we present a multi-view video generation augmentation framework—featuring dynamically moving wrist-camera views—that enables diversification of backgrounds and tabletop scenes in a fully plug-and-play manner using only raw videos as input. This framework necessitates an automatic segmentation pipeline to segment out both the robot and the objects being interacted with. In practice, the interacted object may be invisible in early wristcamera frames, coupled with rapid camera motion, narrow field of view, long trajectories, and limited robotics-specific training; thus, directly applying off-the-shelf models like vision-language models [4, 6] (VLM) often fail to localize the target object reliably. To address these issues, we propose an automated segmentation pipeline that leverages action information to mitigate segmentation failures from the off-the-shelf model, especially on the wrist-camera. Specif-

[Figure 1]

##### 1. Data Curation

2. Diverse Visual Augmentation

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Video Segmentation Visual Identity Pool

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

###### RoboVIP

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

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

###### Task:

[Figure 41]

[Figure 42]

A robot arm is reaching into a cabinet…

Visual Identity Curation

Action - Oriented Segmentation

3. VLA Model Training

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Real-World Data Augmented Data

[Figure 47]

[Figure 48]

“Pick the bowl” “Grab the cup”

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

VLA Model / Visuomotor Policy

[Figure 55]

Action = [Δx, Δy, Δz, Rx , Ry , Rz , Gripper State]

Real-World Video-Action Pair

Figure 1. Overview of our RoboVIP Workflow. (1) We extract observation videos from robotics manipulation data with corresponding action data to segment the robot arm and interacted objects for inpainting-based augmentation. (2) A large-scale pool of visual identity prompts is curated from robotics datasets and used as conditioning inputs for our multi-view video diffusion model to conduct diverse visual augmentation. (3) The augmented videos, paired with action information from original robotics manipulation data, are utilized for downstream VLA and visuomotor policy training.

In summary, we present RoboVIP, a multi-view inpainting-based video diffusion model with visual identity prompting as conditions to augment the visual observations of the robotics manipulation data. Our approach integrates an action-oriented pipeline for multi-view robot and object segmentation, a scalable curation pipeline to construct a large-scale visual identity pool, and a video diffusion model capable of generating temporally consistent multiview sequences with visual identity prompting as conditions. To assess the effectiveness at scale, we augment 12K BridgeV2 [45] trajectories for training mainstream VLA models, including π0 [7] and Octo [42]. We further evaluate RoboVIP on a real-world robot dataset (100 trajectories) for training a visuomotor policy, Diffusion Policy [12]. Across both simulation and real-world robot evaluations, RoboVIP delivers consistent gains in success rate, demonstrating its practicality for large-scale VLA training as well as low-data policy learning.

ically, we use the 1D gripper state to identify the time window that the robot arm actually interacts with the target object, which narrows the search space.

Furthermore, we observe that solely relying on textprompt–guided generation, as done in prior works [2, 3, 11, 52, 53], imposes limitations. The text prompts provided by existing datasets [14, 24] are typically overly simplistic and lack detailed table-top descriptions. Even if we apply SoTA VLM [6] to caption, the generated descriptions frequently suffer from hallucinations and misalignment. More importantly, text prompts can not capture low-level details. To mitigate these issues, we introduce visual identity prompting, conditioning the video diffusion model on one or more exemplar images to synthesize both semantically and lowlevel consistent content in the inpainted regions. This conditioning will force the model to enrich the table-top and background contents. Further, we modify our video diffusion model to incorporate visual identity prompting conditions in the multi-view paradigm and propose an efficient scheme to embrace multiple identities at the same time. To preserve the plug-and-play nature of our framework, visual identities are not manually provided by humans, as in general video generation methods [16, 32]. Instead, we propose an agentic curation and filtering pipeline that automatically constructs a million-scale visual identity pool from largescale robotics datasets. The full pipeline is shown in Fig. 1.

### 2. Related Works

#### 2.1. Conditioned Video Generation

Video generation synthesizes realistic, temporally coherent sequences conditioned on text, images, or videos [2, 27, 46, 51]. Video-to-video models transform an input clip into another consistent sequence, enabling style transfer [27], en-

###### Gripper-Action-Triggered Keyframe Detection

hancement [59], and content editing [23]. Beyond pixelaligned cues, identity references [16, 32, 47] have emerged as a way to inject explicit visual attributes into generation. Although video generation is increasingly used for robot planning [13, 20, 29] and controllable video generation is gaining attention [48, 60], its use for visual augmentation remains underexplored: most existing approaches rely on image-based diffusion [52, 53] or support only single-view conditioning [2], leaving multi-view and richer conditioning largely unexamined.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

...

...

...

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

...

... ...

[Figure 64]

###### Wrist-view Keyframes

Outlier Filter

Video Segmentation SAM2

Open-Vocab Segmentation EVF-SAM

“robot”

Video-Reasoning Model Cosmos-Reason1

[Figure 65]

#### 2.2. Visual Augmentation on Robotics

Clean Mask

Merged for Training

Outlier Filter

Object Naming

Open-Vocab Segmentation EVF-SAM

[Figure 66]

Traditional augmentations like cropping, rotation, and resizing offer limited benefit for robot policy training and do not resolve data scarcity. As a result, practitioners resort to applying learning-based methods to increase visual diversity. GreenAug [43] augments the dataset by manually setting the real-world robot manipulation environment with the green screen and generating the background by post-effects. ReBot [15] and RoboSplat [50] conduct a real-to-sim-toreal paradigm, which converts the real-world robot information to a simulation environment and then manually adds objects, changes views, or poses of the objects to create a hand-crafted dataset. However, these methods require significant manual effort and do not generalize to new tasks or environments in a plug-and-play manner. To achieve plugand-play, Cosmos-Transfer [2, 3] and RoboTransfer [33] use pixel-aligned conditions, like edges, depth, and segmentation, to drive the diffusion model on appearance-level editing. Although effective, this condition design limits the ability to introduce new semantic content for the diffusion model. Instead, Rosie [52] and RoboEngine [53] apply masking to the background and table-top contents and then apply a generative model to inpaint masked areas. This unleashes the power of generative models beyond appearancelevel editing.

Text Condition

Video Segmentation SAM2

“bowl” “cube” “drawer”

[Figure 67]

…

[Figure 68]

Figure 2. Segmentation Pipeline. Our segmentation pipeline comprises two parallel streams: one for robot-arm segmentation and one for interacted-object segmentation. We first use the gripper-action signal to identify accurate keyframe ranges, which is helpful to locate the interacted objects that are not visible in the first or last frame. We then leverage off-the-shelf models such as Cosmos-Reason1 [4] and SAM2 [36], together with several heuristic refinements, to obtain accurate masks in a fully plug-andplay manner.

remain scarce because real-world collection is slow, costly, and rarely long-horizon or synchronized [9]. This gap motivates data augmentation that generates additional supervision while preserving frame-to-frame and view-to-view consistency.

### 3. Method

#### 3.1. Problem Formulation

As shown in Fig. 1, starting from a robotic manipulation episode with multi-view video sequences and its corresponding action information on the end-effector state, we segment the robot arm and the objects being interacted with to preserve the fundamental 6-DoF Cartesian end-effector delta pose and gripper-state information. For the remaining masked regions—including background, foreground, and tabletop objects—we adopt a video diffusion model that is conditioned on the text prompt y, the masked multiview videos M = {M0,...,MN}, and the proposed visual identity prompting f = {f1,...,fk}. The model is trained to learn the conditional joint distribution pθ(I0,...,IN | M0,...,MN,y,f1,...,fk), where I = {I0,...,IN} denotes the generated latent video frames that adhere to all conditioning signals. The inpainted multi-view videos serve as augmented observations for policy model training, while the original action sequences are directly reused.

#### 2.3. Manipulation Models

Research on robot manipulation has progressed from early visuomotor policies to unified VLA architectures. Classic visuomotor systems [12, 28] map images to actions with supervised or reinforcement learning; however, task-specific demonstrations limit generalization. Large vision-language models [1, 35] broadened representational capacity, which in turn motivated VLAs that jointly encode vision, language, and action. Two design axes are now prominent: first, temporal conditioning, where models range from short history windows (RT-1 [8], Octo[42], OpenVLA[42]) to sequence encoders; second, viewpoint, where many systems use a single egocentric stream, whereas newer ones incorporate multi-view inputs to improve 3D reasoning (e.g., π0[7]). As a result, training increasingly requires temporally coherent and cross-view-aligned data, yet such data

Generated Video Freezed Parameters LoRA Finetune Full Finetune Clean Latent Noisy Latent

#### 3.2. Action-guided Segmentation of Robots and Interacted Objects

[Figure 69]

[Figure 70]

[Figure 71]

VAE Decoder

The action information generally includes the 6-DoF pose of the end-effector ∆x, ∆y, ∆z, ∆roll, ∆pitch, ∆yaw, and a 1-D gripper state. The gripper state indicates when the robot arm closes or opens, which provides decisive cues for most robot manipulation tasks. In a long video sequence, an effective grasp by the robot arm occurs only within a very short time window, and the moment when the gripper state changes can be used to localize the interacted object from the wrist-view perspective. Specifically, as shown in Fig. 2, we first identify the frame range corresponding to gripper-closure intervals in the wrist view, which marks the preparation and execution of interaction. The extracted video clip is then fed into a video-reasoning VLM [4] to infer the object’s semantic label, enabling accurate object naming directly from the wrist view.

###### Padding

###### Diffusion Transformer

Patchify Layer Text Encoder

Channel-wise Concat

c Frame-wise Concat

[Scene Setup]: Place the spice jar, salt shaker, sauce bottle, and water bottle around the robot arm on the wooden surface.

VAE Encoder

VAE Encoder

VAE Encoder

[Action Description]: The arm is moving colorful cubes, stacking and rearranging them with precision.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Visual Identity Prompting

Ground Truth Segmented Video

Figure 3. Video Diffusion Model Architecture. Our video diffusion model is conditioned on the segmented multi-view video sequence, structured text prompt, and visual identity prompting to achieve consistent visual augmentation.

When processing other third-person camera views, the object name obtained from the wrist view is directly reused. The identified object name is fed into an open-vocabulary segmentation model [53, 55] to obtain a reliable mask for the corresponding frame. We separately extract the masks for the robot and the interacted object, followed by median blurring to filter out outlier pixels. At this stage, we locate accurate temporal range and mask locations. To further refine temporal consistency, we perform k-means sampling on the masks, and the sampled points will be taken as prompting for the video segmentation model [36] to track the complete video segmentation for the robot and interacted objects. The robot and the object mask are processed independently and then merged into one at the end. The resulting video segmentation provides high-quality mask conditions that can be directly used in the training process.

trainable but not part of LoRA fine-tuning. Since our training objective shifts from single-image conditioning in the base model to the masked video sequences as conditions, we enable the patchification encoder for training as well. Empirically, we find that fine-tuning this additional layer beyond the LoRA setup tends to yield slight improvements in performance.

For multi-view inputs, inspired by [22], we adopt a structured vertical stitching strategy, which concatenates masked frames from different views at the same timestamp. The ground-truth sequences are processed in the same manner, ensuring that the learning objective remains view-aligned and encourages the video diffusion model to capture crossview spatial consistency and correspondence. Accordingly, we modify the base model’s input structure by replacing the single-image padding with channel-wise concatenation of the full video sequence, which achieves a minimally invasive yet effective formulation of a video-conditioned objective. The overall model structure can be found in Fig. 3.

#### 3.3. Multi-View Inpainting Video Diffusion Model

We aim to transfer the generation quality and conditionalignment capability of state-of-the-art video diffusion models to robotic tasks. To this end, our base model is the Wan2.1 [46] image-to-video variant with 14 billion parameters. However, directly fine-tuning such a large model is computationally infeasible. More critically, it leads to severe overfitting collapse, causing the model to rapidly forget its original visual generation stability. To address this, we adopt a Low-Rank Adaptation [19] (LoRA) strategy to enable feasible and memory-efficient fine-tuning.

#### 3.4. Visual Identity Prompting

For robotic downstream tasks, we aim for a pipeline that can autonomously select appropriate and necessary visual identities without any human intervention. To achieve this, we design an agentic inference pipeline, as shown in Fig. 4, that automatically constructs a massive, rich, and diverse visual identity pool. We find that adopting a panoptic segmentation [26] approach is the most straightforward way to achieve this goal. Panoptic segmentation simultaneously provides mask localization and corresponding label classification. Based on the classification label, we select common objects that are needed and do not consider background-

Recent video diffusion models are predominantly built upon the Diffusion Transformer [34] architecture, where attention blocks serve as the primary computational units. LoRA injects trainable low-rank adapters into the linear projections, typically applied to the query and value matrices within attention layers. Apart from the attention blocks, the patchification layer—implemented as a convolutional layer that transforms latent images into patches—is fully

[Figure 81]

[Figure 82]

[Figure 83]

Input (Multi-view) Images

ambiguity in identity prompting.

[Figure 84]

To incorporate visual identity prompting into the video diffusion model, we adopt a frame-wise concatenation strategy, following the design of [47, 58]. As illustrated in Fig. 3, before entering the video diffusion transformer, the packed identity images are first encoded by a shared causal VAE encoder [46] and concatenated with the latent video segmentation inputs along the frame dimension. The noisy frame latent is zero-padded for temporal alignment and then channel-wise concatenated with the conditional inputs. After the diffusion transformer processes all layers, the identity tokens are dropped and excluded from loss computation, ensuring they serve purely as contextual guidance rather than optimization targets. During inference, newly encoded identity images are injected at each diffusion timestep to continuously guide generation.

[Figure 85]

[Figure 86]

[Figure 87]

Panoptic Segmentation Model

[Figure 88]

❌Table, Wall, Door, Sink, Banner …

✅ Cup, Bottle, Bowl, Orange, Chair …

Class Type Selection

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Image Quality Assessment

CLIP Completenes

Resolution Filter

Clarity Filter

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

Random Select

Random Resize

### 4. Experiment

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Packing

[Figure 129]

[Figure 130]

#### 4.1. Video Diffusion Model Implementation Details

[Figure 131]

[Figure 132]

Data Curation. For all videos, we first discard sequences that are too short (fewer than 25 frames). For overly long sequences (more than 550 frames), we perform temporal cropping to mitigate segmentation failures induced by excessively extended inputs. Since the captions provided by the original dataset are often noisy, we re-caption all videos using Qwen2.5-VL 32B [6], employing a multi-view vertical stitching strategy to ensure consistent and accurate textual descriptions across different viewpoints. The text prompt is composed of the scene setup and the action description. Then, for the robot and object segmentation described in the method section, we adopt OneFormer [21] for the panoptic segmentation. The open-vocab segmentation model is the EVF-SAM [55] model from RoboEngine [53]. The video segmentation is done by SAM2 [36].

###### Conditioning for VDM

Visual Identity Pool

- Figure 4. Visual Identity Curation and Processing Pipeline. Our visual identity is curated by panoptic segmentation from the large-scale robotics dataset [14, 24, 45], followed by several scoring criteria filters. In augmentation, we randomly select some from the pool and pack them into one image frame to serve as conditioning for our video diffusion model.

related large objects, like the table and the wall. Using these labels, we can naturally classify both tabletop objects and background elements, ultimately forming a comprehensive visual identity pool. To this end, we constitute a millionscale visual identity pool.

We observe that objects obtained by straightforward segmentation are often of suboptimal quality. Moreover, some of these segmented objects are partially occluded and thus cannot serve as semantically complete visual identity references. To address this, we crop the corresponding visual identity image predicted by the panoptic segmentation model [21] and then apply several filtering criteria, including image quality assessment [49], sharpness clarity assessment, CLIP-based text–image scoring [35], and resolution size filtering. The CLIP text embedding is derived from the panoptic segmentation class label, serving as an effective proxy to assess the semantic completeness of each object.

Training Details. We train our video diffusion models on Bridge V1 [14] and V2 [45] for the following downstream VLA tasks, which provide one to three third-person views. Further, we train Droid [24] for the visual quality comparisons and real-robot augmentation, which includes a wristmounted camera and two third-person views. To support variable-length sequences, we adopt a batch size of 1 per GPU and use gradient accumulation to achieve an effective batch size of 4 per GPU, which costs around 70GB per GPU in training. This strategy enables dynamic frame sampling without incurring unnecessary computation from padding or attention mask overhead. Since current VLA models cannot condition on long observation histories, we train on at most 49 frames. We train for 15K iterations on 8 GPUs whose per-GPU memory is 144GB, resulting in a total cumulative batch size of 32. Each view is trained at 256×256 resolution for Bridge and 320×416 for Droid. When an instance contains only a single view, the conditioning input

Unlike previous approaches that inject only a single identity reference per frame, we adopt a packing scheme to efficiently accommodate multiple visual identity references within a single frame, thereby reducing computational overhead. To prevent overfitting to fixed scale ratios, each identity image is randomly resized before encoding. During training under multi-view supervision, all visual identity references are sampled from a single view to avoid view-

“A cluttered workspace with tools, a table holding a bowl, cup, sponge, and cloth. Robot arm moves towards the table, picks up an object.”

“Kitchen counter with stacked cups, plates, and utensils. Person in background. Clean, organized workspace. Robot arm moves towards the counter, picks up an object.”

|[Figure 133]|[Figure 134]|[Figure 135]|[Figure 136]|[Figure 137]|[Figure 138]|[Figure 139]|
|---|---|---|---|---|---|---|

|[Figure 140]|[Figure 141]|[Figure 142]|[Figure 143]|[Figure 144]|[Figure 145]|[Figure 146]|
|---|---|---|---|---|---|---|

GroundTruth

|[Figure 147]|[Figure 148]|[Figure 149]|[Figure 150]|[Figure 151]|[Figure 152]|[Figure 153]|
|---|---|---|---|---|---|---|

|[Figure 154]|[Figure 155]|[Figure 156]|[Figure 157]|[Figure 158]|[Figure 159]|[Figure 160]|
|---|---|---|---|---|---|---|

RoboEngine

2.5

|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|[Figure 166]|[Figure 167]|
|---|---|---|---|---|---|---|

|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|[Figure 172]|[Figure 173]|[Figure 174]|
|---|---|---|---|---|---|---|

Cosmos-Transfer

|[Figure 175]|[Figure 176]|[Figure 177]|[Figure 178]|[Figure 179]|[Figure 180]|[Figure 181]|
|---|---|---|---|---|---|---|

|[Figure 182]|[Figure 183]|[Figure 184]|[Figure 185]|[Figure 186]|[Figure 187]|[Figure 188]|
|---|---|---|---|---|---|---|

Ours

- Figure 5. Qualitative comparisons of different models on Droid [24]. Our method produces temporally consistent and visually diverse results, outperforming RoboEngine [53], which is a single-image-based method, and Cosmos-Transfer2.5 [3], which struggles to generalize beyond appearance-level edge conditioning. Zoom in for the best view.

[Figure 189]

Visual Identity Prompt

Visual Identity Prompt Segmentation Generated Video

for the missing view is zero-padded with black pixels to distinguish 255-value white pixels of the segmentation masks.

Segmentation Generated Video

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

In practice, we observe that generative quality correlates with the model’s pretrained resolution; therefore, the cumulative stitched width and height for multi-view must be lower than the pretrained setting. Guided by this finding, we employ the Wan2.1-I2V [46] 720p variant to support diverse generation settings, rather than the lower-resolution 480p model. We set the LoRA [19] rank to 128 and 256 for the Bridge and Droid configurations, respectively. To maximize data utilization, we randomly sample two views from the three available in Bridge V2. For Droid, we fix the wrist-mounted camera as the first view and select the second view from the two third-person perspectives.

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Figure 6. Augmented BridgeV2 Data by our RoboVIP for VLA Training. Our visual identity prompting enriches tabletop contents and introduces additional distractors to create more challenging settings for the policy model. The visual identity is randomly selected from our proposed pools. Zoom in for the best view.

#### 4.2. Video Generation Results

Our goal is to develop a scalable, plug-and-play multi-view inpainting-based generative framework that serves as an effective augmentation solution for robotic manipulation. To this end, we evaluate against Cosmos-Transfer2.5 [3], a video diffusion model designed for real-to-real generation, and RoboEngine [53], an inpainting-based approach similar to ours, on the held-out test subset of the Droid [24] dataset consisting of 300 test cases. We consider both a wrist-mounted view and a third-person view for augmentation. For RoboEngine, we use identical robot and object segmentation masks as our method for a fair comparison. For Cosmos-Transfer2.5, we evaluate its edgeconditioned variant, using its native 720p setup, and the model is conditioned on the same Qwen2.5-VL [6] captioned text prompts like ours. For our RoboVIP, we apply visual identity prompting. For all methods, we generate at

most 49 frames per episode, starting from the first frame.

We report standard generative video metrics: Fr´echet Inception Distance [17] (FID), Fr´echet Video Distance [44] (FVD), and Learned Perceptual Image Patch Similarity [54] (LPIPS). FID measures single-frame visual quality via distributional differences, while FVD captures temporal coherence and video-level dynamics. LPIPS quantifies imagelevel perceptual similarity between generated outputs and ground truth in deep feature space rather than pixel space. These metrics measure on vertically stitched inputs due to the multi-view setting. To reflect the multi-view nature of our setting, we follow prior works [5, 33] and evaluate cross-view correspondence by counting matched feature

- Table 1. Comparison of evaluation results on the WidowX robot in SIMPLERENV. We evaluate two variants of our RoboVIP: a textprompt–conditioned multi-view inpainting video diffusion model, and another version with additional visual identity prompting conditions (denoted as ID). Each task is performed on 100 trials. Each entry shows Grasp/Put, where Put is the conditional success rate given a successful Grasp (Put = Success/Grasp), and the Success column reports overall task success. Bold and underlined numbers in the Average Success column indicate the best and second-best performance.

Model

Put spoon on towel

Put carrot on plate

Stack green cube on yellow cube

Put eggplant in basket Average

Grasp/Put Success Grasp/Put Success Grasp/Put Success Grasp/Put Success Grasp/Put Success

Octo [42] (Zero-Shot) 34% / 26% 9% 35% / 20% 7% 28% / 0% 0% 65% / 51% 33% 40.5% / 30.1% 12.2% Octo (Bridge V2 SFT) 52% / 41% 29% 32% / 37% 14% 47% / 4% 3% 60% / 11% 5% 47.5% / 23.0% 12.8% Octo+RoboEngine [53] 67% / 21% 14% 43% / 37% 16% 43% / 5% 2% 0% / 0% 0% 38.2% / 20.9% 8.0%

Octo+RoboVIP (Text prompt) 59% / 7% 4% 69% / 46% 32% 55% / 9% 5% 63% / 17% 11% 61.5% / 21.1% 13.0% Octo+RoboVIP (Text prompt with ID) 59% / 63% 37% 37% / 62% 23% 47% / 15% 7% 37% / 19% 7% 45.0% / 41.1% 18.5%

π0 [7] (Zero-Shot) 52% / 63% 33% 0% / 0% 0% 28% / 7% 2% 31% / 42% 13% 27.75% / 43.2% 12% π0 (Bridge V2 SFT) 57% / 63% 36% 44% / 43% 19% 30% / 7% 2% 29% / 41% 12% 40% / 43.1% 17.25% π0+RoboEngine [53] 61% / 70% 43% 33% / 30% 10% 61% / 11% 7% 31% / 45% 14% 46.5% / 39.8% 18.5%

π0+RoboVIP (Text prompt) 74% / 84% 62% 52% / 40% 21% 49% / 14% 7% 36% / 64% 23% 52.75% / 55.0% 29% π0+RoboVIP (Text prompt with ID) 73% / 64% 47% 49% / 41% 20% 52% / 13% 7% 53% / 70% 37% 56.75% / 48.9% 27.75%

- Table 2. Generative Model Comparisons on 300 test cases of Droid [24]. Cosmos refers to Cosmos-Transfer2.5 [3]. The best is highlighted.

mance [57]. SimplerEnv enables a consistent benchmark of manipulation policies under common robot setups.

We evaluate our pipelines on two recent visionlanguage-action models: Octo-base [42] and π0 [7]. Specifically, we adopt the Octo-base model as a multi-frame conditioned policy (with 2 history frames) and the π0 model as a single-frame VLA policy. We fine-tune both Octo and π0 on 8 NVIDIA GPUs with 48GB of memory each. All experiments use a global seed of 42 and identical data processing and augmentation settings following their official preprocessing pipeline.Both Octo and π0 are evaluated under three training regimes:

Method FID↓ FVD↓ LPIPS↓ MV-Mat.↑

Cosmos [3] 47.43 325.4 0.353 1583.4 RoboEngine [53] 62.77 1788.8 0.598 1301.9 RoboVIP (Ours) 39.97 138.4 0.409 2242.1

points between two generated views (MV-Mat.). A higher count indicates better spatial consistency and generative stability. We use GIM [38] as the correspondence model, keeping all confidence thresholds and hyperparameters identical to its demo configuration.

- • Zero-shot: both Octo-base and π0 are directly deployed without any further supervised fine-tuning in the SimplerEnv tasks.
- • Supervised Fine-Tuning (SFT) on BridgeDataV2: we fine-tune each model using the open-ended instructionconditioned dataset BridgeDataV2 [45] and then deploy.
- • Mixed-policy baseline vs. our method: We mix BridgeDataV2 with the augmented data produced by RoboEngine [53] and our proposed RoboVIP. We evaluate two variants of our multi-view inpainting video diffusion model. The first variant uses only text prompts as the generative condition. The second variant augments the same architecture with our visual identity prompting, which provides exemplar images as additional conditioning signals as shown in Fig. 6.

As shown in Tab. 2, our method consistently outperforms prior approaches on most quantitative metrics. The improvement can be attributed to the fact that RoboEngine operates under a single-frame, single-view setting, while Cosmos-Transfer2.5 overlooks the requirements of multiview generation. In supplementary, we will also include a human study for the visual identity prompting-oriented comparisons. As shown in Fig. 5, compared to RoboEngine, our RoboVIP performs distinguished temporal consistency. Compared to Cosmos-Transfer2.5, ours unleashes diverse scene generation, which is not limited by the pixelaligned conditions, like edges or depth. This is thanks to our inpainting design choices. Further, none of the methods achieve multi-view consistent generation.

Tab. 1 presents quantitative comparisons across the SimplerEnv tasks. For the Octo family, our RoboVIP (Text+ID) achieves an average success rate of 18.5%, improving upon Octo zero-shot experiment (12.2%) and Bridge SFT version (12.8%), and our text-prompt-only variant (13.0%). For π0, our RoboVIP (Text-only) configuration yields the highest overall success at 29.0%, outperforming both the SFT base-

#### 4.3. Simulation Results

To evaluate performance in a reproducible and scalable way, we employ the simulation environment suite SimplerEnv [30], which shows realistic textures in the simulation environment like the real-world and has been shown to correlate well with real-world robot manipulation perfor-

|[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>1<br><br>2<br><br>3<br><br>[Figure 205]<br><br>Open Space Scene Cluttered Scene| |
|---|---|
|[Figure 206]<br><br>DP DP + RoboVIP<br><br>[Figure 207]<br><br>Stack Cube Task in Varied Scenes|[Figure 208]<br><br>Working Space<br><br>Third view Camera<br><br>Setup<br><br>Wrist Camera|

45

RoboEngine (Baseline)

| |
|---|

40

RoboVIP (Ours)

35

30

28.0%

Successrate(%)

25

20

18.5%

14.8%

15

10.2%

9.5%

10

8.0%

- 5

1.5%

0.0%

0

History = 1 History = 2 History = 4 History = 6 Conditioned history length (frames)

Figure 8. Real Robot Experiment on Diffusion Policy. Both policies were trained using identical parameter settings and measured over 10 trials.

Figure 7. Policy success rate vs. conditioned history length (frames). Bars show task-averaged success rates for RoboEngine (baseline) and our RoboVIP (Text prompt with ID) on Octo [42]. The error bars denote standard deviation across tasks. The bar plot indicates that RoboVIP consistently outperforms RoboEngine, whose average success falls to zero at six history frames.

under six-frame conditioning, our RoboVIP still preserves meaningful success rates, underscoring its robustness to longer temporal contexts. This trend suggests that videolevel generative augmentation—not image diffusion—is a more scalable and forward-looking direction, which is applicable for future long-horizon needs on VLA training.

line (17.25%) and RoboEngine (18.5%). The Text+ID variant performs similarly at 27.75%, confirming that both visual identity prompting and temporally consistent generative augmentation contribute to stronger generalization.

#### 4.4. Real-World Robot Results

A closer look at the decomposition into Grasp and conditional Put success (Put = Success / Grasp) reveals the source of these gains. For Octo, our RoboVIP (Text+ID) attains the best average Put success at 41.1%, significantly higher than the 23.0% achieved by Octo SFT. In π0, the text-only RoboVIP obtains the highest Put success of 55.0%, exceeding the SFT baseline (43.1%) and RoboEngine (39.8%). These results indicate that our method not only improves task initiation (grasping) but also strengthens the more challenging post-grasp “Put” phase, demonstrating enhanced closed-loop control and task completion reliability.

To specifically validate the effectiveness of our RoboVIP augmentation pipeline against real-world background distractors, we conduct experiments using a 7-DoF Franka Research 3 robotic arm equipped with a Robotiq gripper. We design a cube stacking task, which requires grasping a blue cube and stacking it onto the red cube. All experiments are conducted using Diffusion Policy (DP) [12]. We established two experimental settings to test robustness against background distractors: Open space: A clean background with no distractors. Cluttered: A scene with 4 different distractor objects. We compare the performance of two policies:

- • DP: A DP model trained solely on 100 real-world demonstration trajectories.
- • DP + RoboVIP (Text+ID) : A DP model trained on a mixed dataset of 200 trajectories, consisting of the 100 original demonstrations and 100 additional trajectories augmented by our RoboVIP framework.

The observed improvements arise from RoboVIP’s ability to generate temporally consistent, multi-view scenes that closely approximate real data distributions. For models such as Octo, which condition on multiple frames, our generated sequences provide realistic motion continuity that mitigates frame inconsistency issues present in RoboEngine. For π0, the multi-view setup aligns with its pretraining configuration, reducing the gap between synthetic and real data distributions. Moreover, the use of visual identity prompts enriches scene diversity (as shown in Fig. 6) and introduces controlled clutter, which empirically benefits learning in visually complex settings. As a result, generative data from RoboVIP can closely approach—or even surpass—the effectiveness of real fine-tuning data.

As shown in Figure 8, the baseline DP model’s success rate drops from 7/10 in the Open space setting to 0/10 in the Cluttered setting. In contrast, the DP + RoboVIP model achieves perfect 10/10 success in the open space setting and maintains a robust 9/10 success rate in the cluttered setting. This demonstrates that our augmentation pipeline significantly enhances the policy’s generalization and robustness to real-world visual distractors. More details and experiments are presented in the supplementary.

Further, we compare the influence of history length on VLA success in Fig. 7. We retrained and tested Octo on different numbers of history observation frames. Across all historical lengths, our RoboVIP maintains consistently higher success rates than RoboEngine (baseline). Notably, while RoboEngine’s performance collapses to nearly zero

### 5. Conclusion

In this work, we introduce RoboVIP, a multi-view inpainting video diffusion model with visual identity prompting to augment visual observations of the robotic manipula-

tion data in a plug-and-play manner. We augment largescale data and demonstrate its effectiveness in both visionlanguage-action and visuomotor policy models on both the simulation environment and the real-world robot deployment.

Limitation. Although our method can automate largescale visual data augmentation and we prove its effectiveness on VLA and visuomotor policy learning, several limitations stem from current tools. State-of-the-art video segmentation [36] still struggles with gripper localization and flickering; VLM reasoning [4, 6] often fails to identify interactive objects; and the open-vocabulary segmentation [53, 55] frequently produces incorrect masks and does not produce consistent results in multi-view inputs. Furthermore, although our real-world experiments leverage multi-view observations, the SimplerEnv[30] benchmark only supports single-view image inputs; therefore, more extensive simulation studies are needed to fully evaluate the benefits of multi-view consistency training.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Jin, et al. Flamingo: A visual language model for few-shot learning. In NeurIPS, 2022. 3
- [2] Hassan Abu Alhaija, Jose Alvarez, Maciej Bala, Tiffany Cai, Tianshi Cao, Liz Cha, Joshua Chen, Mike Chen, Francesco Ferroni, Sanja Fidler, et al. Cosmos-transfer1: Conditional world generation with adaptive multimodal control. arXiv preprint arXiv:2503.14492, 2025. 2, 3
- [3] Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025. 2, 3, 6, 7, 15, 16, 17
- [4] Alisson Azzolini, Junjie Bai, Hannah Brandon, Jiaxin Cao, Prithvijit Chattopadhyay, Huayu Chen, Jinju Chu, Yin Cui, Jenna Diamond, Yifan Ding, et al. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint arXiv:2503.15558, 2025. 1, 3, 4, 9, 12
- [5] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760, 2024. 6
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 1, 2, 5, 6, 9, 12
- [7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1, 2, 3, 7
- [8] Anthony Brohan, Yevgen Chen, Karol Hausman, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 3

- [9] Anthony Brohan, Noah Brown, Daniel Rifai, et al. Open x-embodiment: Robotic learning datasets and rt-x models. arXiv preprint arXiv:2310.08864, 2023. 3
- [10] Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, Steven Palma, Pepijn Kooijmans, Michel Aractingi, Mustafa Shukor, Dana Aubakirova, Martino Russi, Francesco Capuano, Caroline Pascal, Jade Choghari, Jess Moss, and Thomas Wolf. Lerobot: State-ofthe-art machine learning for real-world robotics in pytorch. https://github.com/huggingface/lerobot,

2024. 15

- [11] Zoey Chen, Zhao Mandi, Homanga Bharadhwaj, Mohit Sharma, Shuran Song, Abhishek Gupta, and Vikash Kumar. Semantically controllable augmentations for generalizable robot learning. The International Journal of Robotics Research, 44(10-11):1705–1726, 2025. 1, 2
- [12] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44 (10-11):1684–1704, 2025. 1, 2, 3, 8, 15, 16, 17
- [13] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023. 3
- [14] Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine. Bridge data: Boosting generalization of robotic skills with cross-domain datasets. arXiv preprint arXiv:2109.13396, 2021. 1, 2, 5, 12
- [15] Yu Fang, Yue Yang, Xinghao Zhu, Kaiyuan Zheng, Gedas Bertasius, Daniel Szafir, and Mingyu Ding. Rebot: Scaling robot learning with real-to-sim-to-real robotic video synthesis. arXiv preprint arXiv:2503.14526, 2025. 3
- [16] Zhengcong Fei, Debang Li, Di Qiu, Jiahua Wang, Yikun Dou, Rui Wang, Jingtao Xu, Mingyuan Fan, Guibin Chen, Yang Li, et al. Skyreels-a2: Compose anything in video diffusion transformers. arXiv preprint arXiv:2504.02436, 2025. 2, 3, 16
- [17] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. 6
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 4, 6
- [20] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024. 3
- [21] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer

- to rule universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2989–2998, 2023. 5, 12
- [22] Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, Kaushil Kundalia, Yen-Chen Lin, et al. Dreamgen: Unlocking generalization in robot learning through neural trajectories. arXiv e-prints, pages arXiv–2505, 2025. 4
- [23] Xuan Ju, Tianyu Wang, Yuqian Zhou, He Zhang, Qing Liu, Nanxuan Zhao, Zhifei Zhang, Yijun Li, Yuanhao Cai, Shaoteng Liu, et al. Editverse: Unifying image and video editing and generation with in-context learning. arXiv preprint arXiv:2509.20360, 2025. 3
- [24] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024. 1, 2, 5, 6, 7, 12, 13, 16
- [25] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1
- [26] Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr Doll´ar. Panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9404–9413, 2019. 4
- [27] Max Ku, Cong Wei, Weiming Ren, Harry Yang, and Wenhu Chen. Anyv2v: A tuning-free framework for any video-tovideo editing tasks. arXiv preprint arXiv:2403.14468, 2024. 2
- [28] Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. In Journal of Machine Learning Research, pages 1–40, 2016. 3
- [29] Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200, 2025. 3
- [30] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024. 7, 9, 13, 16
- [31] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023. 15
- [32] Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Gen Li, Siyu Zhou, Qian He, and Xinglong Wu. Phantom: Subject-consistent video generation via crossmodal alignment. arXiv preprint arXiv:2502.11079, 2025. 2, 3, 16
- [33] Liu Liu, Xiaofeng Wang, Guosheng Zhao, Keyu Li, Wenkang Qin, Jiaxiong Qiu, Zheng Zhu, Guan Huang, and Zhizhong Su. Robotransfer: Geometry-consistent video diffusion for robotic visual policy transfer. arXiv preprint arXiv:2505.23171, 2025. 3, 6

- [34] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 4

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3, 5, 13, 16
- [36] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3, 4, 5, 9, 15
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1
- [38] Xuelun Shen, Zhipeng Cai, Wei Yin, Matthias M¨uller, Zijun Li, Kaixuan Wang, Xiaozhi Chen, and Cheng Wang. Gim: Learning generalizable image matcher from internet videos. arXiv preprint arXiv:2402.11095, 2024. 7
- [39] Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 16
- [40] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1
- [41] Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Viswesh Nagaswamy Rajesh, Yong Woo Choi, Yen-Ru Chen, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025. 15
- [42] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 1, 2, 3, 7, 8
- [43] Eugene Teoh, Sumit Patidar, Xiao Ma, and Stephen James. Green screen augmentation enables scene generalisation in robotic manipulation. arXiv preprint arXiv:2407.07868,

2024. 3

- [44] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [45] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Con-

- ference on Robot Learning, pages 1723–1736. PMLR, 2023. 2, 5, 7, 12, 13
- [46] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 4, 5, 6, 13
- [47] Boyang Wang, Xuweiyi Chen, Matheus Gadelha, and Zezhou Cheng. Frame in-n-out: Unbounded controllable image-to-video generation. arXiv preprint arXiv:2505.21491, 2025. 3, 5
- [48] Boyang Wang, Nikhil Sridhar, Chao Feng, Mark Van der Merwe, Adam Fishman, Nima Fazeli, and Jeong Joon Park. This&that: Language-gesture controlled video generation for robot planning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 12842–12849. IEEE, 2025. 3
- [49] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, pages 2555–2563, 2023. 5, 12
- [50] Sizhe Yang, Wenye Yu, Jia Zeng, Jun Lv, Kerui Ren, Cewu Lu, Dahua Lin, and Jiangmiao Pang. Novel demonstration generation with gaussian splatting enables robust one-shot manipulation. arXiv preprint arXiv:2504.13175, 2025. 3
- [51] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 13
- [52] Tianhe Yu, Ted Xiao, Austin Stone, Jonathan Tompson, Anthony Brohan, Su Wang, Jaspiar Singh, Clayton Tan, Jodilyn Peralta, Brian Ichter, et al. Scaling robot learning with semantically imagined experience. arXiv preprint arXiv:2302.11550, 2023. 1, 2, 3
- [53] Chengbo Yuan, Suraj Joshi, Shaoting Zhu, Hang Su, Hang Zhao, and Yang Gao. Roboengine: Plug-and-play robot data augmentation with semantic robot segmentation and background generation. arXiv preprint arXiv:2503.18738, 2025. 1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 17
- [54] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6
- [55] Yuxuan Zhang, Tianheng Cheng, Lianghui Zhu, Rui Hu, Lei Liu, Heng Liu, Longjin Ran, Xiaoxin Chen, Wenyu Liu, and Xinggang Wang. Evf-sam: Early vision-language fusion for text-prompted segment anything model. arXiv preprint arXiv:2406.20076, 2024. 4, 5, 9, 15
- [56] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023. 1
- [57] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances

- spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024. 7
- [58] Yong Zhong, Zhuoyi Yang, Jiayan Teng, Xiaotao Gu, and Chongxuan Li. Concat-id: Towards universal identity-preserving video synthesis. arXiv preprint arXiv:2503.14151, 2025. 5
- [59] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-a-video: Temporalconsistent diffusion model for real-world video superresolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2535– 2545, 2024. 3
- [60] Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong. Irasim: A fine-grained world model for robot manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9834– 9844, 2025. 3

## Supplementary Material

### 6. Overview

This supplementary material provides more implementation and technical details, additional quantitative comparisons with analysis, and more qualitative visualization to complement the main manuscript. In Sec. 7, we present more details of our video diffusion model, including dataset curation, caption generation, segmentation workflows, and visual identity filtering. We then elaborate on the simulation setup used for evaluating augmented data in tabletop manipulation tasks, followed by details of our real-world robot experiments. In Sec. 8, we provide real-world robot baseline comparisons and a user study to prove our proposed visual identity prompting. Finally, in Sec. 9, we provide more visualizations of rollouts and generated videos to support the qualitative findings.

### 7. More Experimental Details

#### 7.1. More Video Generation Details

In the dataset curation, the text prompt is composed of the scene setup description and the robot arm action description. The scene setup caption for VLM [6] is: Describe the scene setup in the video (exclude the robot arm) within 15 words. The input is a vertically stitched multi-view video. Only provide information with high confidence. The action description caption for VLM is: Describe the action of the robot arm briefly in the video within 15 words. The input is a vertically stitched multi-view video. Only provide information with high confidence. Additionally, for longduration datasets such as Droid [24], we split the video into shorter frame chunks rather than using the entire sequence, preventing the diffusion model using text prompt from describing events that do not occur in the segment. The caption for VLM, instead, is: Describe the action of the robot arm briefly in the video within 10 words. Do not predict what will happen. Just focus on what has been done. The input is a vertically stitched multi-view video. Only provide information with high confidence.

Our segmentation pipeline is illustrated in Fig. 2 of the main paper and consists of two parallel streams: one for robot-arm segmentation and one for interacted-object segmentation. The gripper state annotations vary across datasets. For Bridge V1 [14] and V2 [45], the gripper open/close state is provided as a Boolean indicating whether the gripper is closed. For Droid [24], the gripper open/close state is a continuous value representing the physical distance between the two fingertips. We convert this to a Boolean state through thresholding to maintain compatibility with the Bridge format. Meanwhile, we apply a tem-

poral buffer when converting gripper states, adding a constant offset of 5 frames before the detected gripper-close event and 5 frames after the gripper-open event. This enlarges the effective action window and mitigates annotation edge cases. Next, open-vocab segmentation needs to input an accurate query. For the robot-arm stream, we simply use the fixed query “robot.” Additionally, we uniformly sample five frames from each video and select the frame with the largest robot arm masking region estimated by the segmentation model. This is because we observe that the robot arm does not always appear in the first frame or the last frame; the consideration of middle frames is needed. For the wrist camera view, we additionally leverage off-theshelf VLM reasoning models, Cosmos-Reason1 [4] (7B), which are trained on large-scale robotics datasets to enable plug-and-play labeling. All videos are resized to a unified resolution of 320×448 to ensure stable and consistent segmentation performance. Afterward, we apply an outlier filter and sample segmentation keypoints using K-means, following the procedure described in the main manuscript.

In the visual identity curation, the class detected by OneFormer [21] has 133 classes, but not all of them are what we need in the robotics scene. Thus, we manually select classes that correlate to the robotics table-top and background scene: bus, boat, sheep, cow, elephant, bear, zebra, giraffe, backpack, handbag, suitcase, frisbee, skis, snowboard, sports ball, baseball bat, baseball glove, tennis racket, bottle, wine glass, cup, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, cake, chair, laptop, remote, keyboard, cell phone, book, clock, scissors, teddy bear, blanket, cardboard, flower, fruit, pillow, towel, food-other-merged, microwave, oven, toaster, refrigerator, potted plant, couch, banner, net, platform, bench, mirror-stuff, bed, vase, tent, paper-merged, cabinet-merged, curtain.

We crop each visual identity into a square bounding box and mask the background. We discard identities whose masked background covers more than 60% of the bounding box, as overly large masks often lead to visually abnormal or distorted identity crops.

To curate the visual identity pool, we apply a series of quality-control filters. First, we assess overall image quality using CLIP-IQA [49] and remove the lowest 50%. Second, to exclude extremely small or disproportionately large

objects, we discard crops falling in the smallest and largest 10% by resolution size. Third, we measure image sharpness using the variance of the Laplacian on grayscale images and filter out the lowest 30%. Finally, for CLIP [35] text–image alignment, we remove 60% of samples with the weakest similarity scores.

In the video diffusion model training, the learning rate is set to 5e-5 with 100 steps of warm-up. We use 8-bit Adam to save GPU memory in the training. In the training, the maximum number of frames is set to 33 for Droid [24] and 49 for Bridge [45], which presents similar training time. This is because most Bridge datasets are shorter than 49 frames, but almost all Droid datasets are longer than 49 frames. Since our pretrained model, Wan2.1 [46], has a CLIP-based image encoder for the first frame, we directly set that with our first segmented frame without doing other modifications. The frames are sampled in their original fps without applying acceleration like general video generation [46, 51]. Our visual-identity input is randomly resized within a scale range of [0.8,1.2] to avoid fixed scale learning for the visual identity reference. To reduce computation during both training and inference, we use at most one packed identity frame. An additional frame consumes computation equivalent to four temporal frames in Wan2.1 [46] due to its temporal compression ratio of 4, and the attention cost scales quadratically with sequence length (O(N2)). If an instance does not require visual identity prompting, we simply omit this identity frame rather than inserting any padding as placeholder.

For the evaluation of RoboEngine on the Droid dataset, we apply the same segmentation masks condition as in our method. RoboEngine does not provide an automatic caption system and assumes ideal text-prompt inputs by the user; however, using VLM-generated captions may introduce hallucinated descriptions of background elements such as tables. For example, the prompt might be “put the apple on the table”. Their system will directly input the word “table” into the open-vocab segmentation model. If large background regions (e.g., the table) are mistakenly included in the segmented video as conditions, the diffusion model would be exposed to an unrealistically large portion of ground-truth pixels, thereby inflating the bias in the evaluation. Compared to largely modifying their segmentation workflow, we choose to directly use our segmented results as their condition, which induces a more direct comparison of the generative model capability. For the Bridge [45] augmentation used in simulation, we directly use the task descriptions provided by the dataset on RoboEngine. These captions are short, concise, and generally more reliable than those for Droid, enabling a fair comparison of end-to-end augmentation capabilities.

#### 7.2. More Simulation Setup Details

As discussed in Sec. 4.3, we conduct experiments on finetuning vision–language–action (VLA) models. To demonstrate the effectiveness of our data augmentation method, we evaluate the performance of the fine-tuned models on the SimplerEnv simulation benchmark [30].

SimplerEnv provides lightweight tabletop manipulation environments that closely mirror the scenes in BridgeData V2 [45]. Each environment consists of a fixed camera observing a planar workspace, a WidowX 250 6-DoF robot arm, and a small number of colored objects or receptacles (e.g., towel, plate, cubes, basket). The robot starts from a reset joint configuration, with the target object placed in a random but task-consistent location on the table, and the target receptacle placed in a distinct region of the workspace. Overall success is defined by the final spatial relationship between the manipulated object and its target receptacle (e.g., on top of, inside). Grasp success is defined by the target object is grasped during execution.

Since our data augmentation method is applied to the BridgeData V2 dataset, which is collected using a WidowX 250 6-DoF robot arm, we conduct our evaluation in the corresponding SimplerEnv environments that emulate these real-world scenes. The four tasks we consider are summarized below.

###### 1. Put Spoon on Tablecloth: The scene contains a

metallic spoon and a blue cloth patch (towel) placed on the tabletop. At the start of each episode, the spoon is initialized at a random position away from the cloth. The goal is to grasp the spoon and place it such that it lies on top of the blue cloth region, indicating successful completion of the task.

###### 2. Put Carrot on Plate: This environment includes an

orange carrot object and a green plate on the table. The carrot is initially positioned off the plate, with random variation in its pose and distance to the target. The objective is to pick up the carrot and place it so that it rests on the surface of the green plate, without requiring a specific orientation.

###### 3. Stack Green Block on Yellow Block: The scene

consists of two cubic blocks: a green block and a yellow block. The yellow block serves as the base and is placed in a fixed region of the workspace, while the green block starts at a separate, randomly sampled position. The goal is to precisely place the green block on top of the yellow block, forming a stable stack. This task requires both accurate vertical placement and alignment of the cube centers.

###### 4. Put Eggplant in Basket: In this setting, the tabletop

contains a purple eggplant-shaped object and an open basket. The eggplant is initialized outside the basket at varying positions, while the basket remains in a fixed area. The robot must grasp the eggplant and place it inside the basket volume (rather than merely touching the rim). Note that BridgeData V2 does not contain an exact instruction and

[Figure 209]

###### Figure 9. π0 rollouts in SimplerEnv. Visualization of rollouts for the π0 policy on the same four tasks and in the same order as Figure 10. As before, each row corresponds to a single task and shows frames sampled uniformly in time from left (initial state) to right (final state). Comparing this figure with Figure 10 provides a qualitative view of the differences in behavior and convergence between the two policies.

[Figure 210]

###### Figure 10. Octo rollouts in SimplerEnv. Each row shows a rollout of the Octo policy on one task. From top to bottom: spoon on towel, carrot on plate, stack cube, and put eggplant in basket. Within each row, frames are sampled uniformly in time from an episode and arranged from left (start of the rollout) to right (end of the rollout), illustrating how the policy gradually moves the object toward the goal region.

Table 3. Generative Model Comparisons on Real-World Experiments. Success rates averaged over 10 trials. The best is highlighted.

Method Open Space Cluttered

Diffusion Policy [12] (DP) 7/10 0/10 DP + RoboEngine [53] 8/10 2/10 DP + Cosmos-Transfer2.5 [3] 3/10 3/10 RoboVIP (Ours) 10/10 9/10

demonstration set for this specific “eggplant in basket” configuration; instead, SimplerEnv provides a scene that is only semantically related to the original data. As a result, this task primarily evaluates the policy’s ability to generalize to a novel but conceptually similar goal.

Since augmenting a large-scale multi-view video dataset is computationally expensive, we subsample episodes using task-relevant keywords for the main manuscript comparisons. Specifically, we filter episodes using the keywords Spoon, Cloth, Carrot, Cube, Eggplant, and Basket, resulting in a curated set of 12k episodes. For Octo SFT, we evaluate the 80k checkpoint, and for fine-tuning with mixed data (e.g., with RoboEngine and RoboVIP) we use the 100k checkpoint to ensure the same number of epochs. For π0, we use the 20k checkpoint for supervised fine-tuning and the 25k checkpoint for evaluation. Since the amount of data is increased by roughly 20%, we also increase the number of training iterations by approximately 20% to enable a fair comparison. For evaluation in SimplerEnv, we only evaluate each policy on a single static camera view, since SimplerEnv does not provide a wrist-camera view.

During Octo training, we enable its built-in data augmentation. Specifically, we use a uniform goal relabeling strategy with a subsample length of 100, and a task augmentation strategy that deletes task conditioning while keeping each image with probability 0.5. All Octo-Base models are trained on eight Nvidia GPUs with 48GB memory each and a global batch size of 128. The same hardware setup is used for π0, and all other training hyperparameters follow the defaults in their official repositories.

We also experimented with LIBERO [31] and ManiSkill3 [41]. However, we found that the off-the-shelf segmentation model [36, 55] performed poorly in these simulators. Furthermore, these simulation environments have texture distributions that differ significantly from those in our real-world training data. In contrast, SimplerEnv is designed to closely match the real-world scenes in BridgeData V2, and thus provides visual distributions that are better aligned with policies trained on real robot data. For this reason, we focus our quantitative evaluation on SimplerEnv.

#### 7.3. Real-World Robot Setup

Experiment Environments. We conduct all real-world experiments using a 7-DoF Franka Research 3 robotic arm equipped with a Robotiq gripper. The table also features a multi-camera system, comprising a wrist-mounted Intel RealSense D435 and a third-person Intel RealSense D455. Both cameras capture RGB frames with a resolution of 640 × 480 at 30 fps.

Training Data. Demonstrations were gathered via teleoperation using a 3Dconnexion SpaceMouse. Frankcontroller is utilized for low-level communication with the robot and gripper during both data collection and policy evaluation. Our dataset for the open-space experiment comprises 100 real human demonstrations, each episode has nearly 180 frames. In addition, we synthesized 100 augmented demonstrations based on these curated trajectories using our proposed RoboVIP and each baseline method. For the comparison of augmentation methods, we constructed training sets of 200 episodes by combining the synthesized demonstrations with the original real-world data.

Data Augmentation. For our real-world robot data augmentation by video diffusion model, video conditions are split into chunks of up to 33 frames before synthesis. Chunks shorter than 33 frames are padded to the nearest 4N+1 length required by the temporal compression scheme of the video diffusion model.

Policy Training. We train and evaluate RoboVIP and 2 additional demonstration augmentation methods, including RoboEngine [53] and Cosmos-Transfer2.5 [3], on enhancing the vanilla Diffusion Policy [12]. For the experiment setting without augmentation methods, the models are trained for 2500 steps, and 5000 steps for all augmented settings. The batch size is set to 64. We utilize Lerobot [10] as the dataloader and trainer. During the data preprocessing, we resize all the different views of observations to 224×224 and downsample the videos to 10 FPS during both the training process and evaluation as the model input. We also filter a few episodes that have a total frame length than 300. During training, we set the horizon, observation steps, and action steps of Diffusion Policy [12] to 8, 2, and 4 for all the models. When training with RoboEngine and CosmosTransfer2.5, we keep the default settings reported in their manuscripts.

Policy Evaluation. We designed a Cube Stacking task to evaluate the policy’s precision and robustness. The objective is to grasp a target blue cube and stack it onto a red cube. A trial is considered successful only if the blue cube is stably placed atop the red cube. A trial is recorded as a failure if any of the following occur: (1) the robot fails to grasp the blue cube; (2) the cube is dropped during transport; or (3) the blue cube is not successfully placed on the red cube or fails to remain stable after the gripper releases it.

We propose two different settings for our real-world experiments, including Open space: A clean background with no distractors. and Cluttered: A scene with 4 different unseen distractor objects.

### 8. Additional Results and Analysis

#### 8.1. Real-World Robot Baseline Comparisons

Real-World Experiment Results. We extend the experiments presented in Sec. 4.4 to compare our RoboVIP against two additional generative methods. The quantitative results, measured by success rates over 10 trials, are listed in Tab. 3. The results demonstrate that our RoboVIP achieves the best performance in enhancing Diffusion Policy across both seen tasks and unseen environments. Notably, Diffusion Policy trained with RoboVIP attains a significantly higher success rate in the Cluttered setting (9/10) compared to other augmentation methods that rarely succeed, while maintaining stable performance under normal and unseen settings.

Visualizations. We visualize the policy rollouts in the cluttered scene setting. As shown in Fig. 11, baseline methods—including vanilla Diffusion Policy and policies trained with Cosmos-Transfer2.5 [3] or RoboEngine [53] augmentations—struggle to generalize in the presence of background distractors. Specifically, vanilla Diffusion Policy and RoboEngine frequently fail to correctly localize the target object, resulting in grasp failures on the blue cube, while Cosmos-Transfer2.5 fails to place the cube stably. In contrast, the policy augmented by our RoboVIP demonstrates robust generalization capabilities. It successfully executes the full horizon of the task: accurately grasping the blue cube amidst distractors, stably transporting it, and precisely stacking it onto the red cube without the cube falling.

#### 8.2. User Study on Visual Identity Prompting

Since our visual identities correspond to small tabletop objects, it is difficult to rely on DINO [39] or CLIPbased [35] metrics, as in general video generation [16, 32] to reliably retrieve these small objects’ features amid the overwhelming background semantics. To more faithfully demonstrate how our proposed visual identity prompting enriches the generated tabletop scenes, we conduct an additional user study comparing our visual-identity-conditioned model against our own text-only variant. We randomly select 50 videos for each rater and asked three anonymous human raters to perform pairwise comparisons between two methods. The instruction provided to the raters is:

You will be shown two videos generated by two different models, along with a visual identity image. For each comparison, please answer the following two questions and select the option that best matches the intended criterion: 1. Which video more faithfully incorporates the provided vi-

sual identity image into the scene? 2. Which video presents a tabletop that is richer in visual content?

For Question (1), raters preferred the visualidentity–conditioned generation in 97.3% of the comparisons, indicating a strong advantage in identity preservation. For Question (2), visual identity prompting was preferred in 80.0% of the comparisons, showing that this feature encourages richer and more detailed tabletop content in the augmentation. These results demonstrate that conditioning on visual identity prompting not only improves identity alignment but also leads to more complex and content-rich scene compositions.

### 9. More Visualization

#### 9.1. Rollout Visualization

To complement the quantitative success rates reported in Sec. 4.3 of the main manuscript, we visualize representative rollouts of the fine-tuned policies in SimplerEnv [30]. In Fig.9 and Fig.10, each row corresponds to one of the four evaluation tasks, and each image is a temporal strip constructed by sampling frames uniformly from the beginning to the end of an episode. Thus, the rollout progresses from left to right within each row, showing how the robot moves the object from its initial pose toward the target configuration.

#### 9.2. Real-World Robot Augmented Videos

As shown in Fig. 12, we include long-horizon generation results by our RoboVIP on real-world robot episodes collected in our lab. These generated videos are directly used for downstream visuomotor policy training, Diffusion Policy [12]. It is worth noting that the hardware configuration in our lab cannot be perfectly aligned with the Droid [24] training data. In particular, while Droid provides gripper views that are almost perfectly centered and symmetric, our wrist-mounted camera exhibits a noticeable left-shift bias. As a result, these real-world rollouts constitute genuine zero-shot cases for our video diffusion model.

Beyond this domain mismatch, our model demonstrates richer tabletop and scene variations under zero-shot deployment. For example, if you closely inspect the wrist-view generations, you will notice that the tabletop textures continuously change across videos, and these textures appear highly realistic. Furthermore, in the third-person view, each generation presents a different background configuration, and even the geometry of the table varies from chunk to chunk. Thanks to our proposed visual identity prompting, the generated tabletop scenes become significantly more enriched with diverse and realistic objects. We recommend readers check our website for more vivid videos.

[Figure 211]

[Figure 212]

Vanilla Diffusion Policy (Grasp Fail)

[Figure 213]

[Figure 214]

RoboEngine (Grasp Fail)

[Figure 215]

[Figure 216]

Cosmos-Transfer2.5 (Grasp Fail)

[Figure 217]

[Figure 218]

RoboVIP (Success)

- Figure 11. Real-World Robot Rollout Results. Baseline methods [3, 12, 53] fail during the grasping stage under the cluttered setup, whereas our approach achieves a successful grasp and final placement. More samples are available on our website.

[Figure 219]

- Figure 12. Real-World Robot Zero-Shot Long-Horizon Augmentation by our RoboVIP. We showcase long-horizon real-world video augmented by our proposed RoboVIP. Since real robot videos contain far more frames than what current video diffusion models can process directly, we split the video into 33-frame chunks and generate them sequentially with different visual identities selected. The figure presents equally sampled frames across the full trajectory. More samples are available on our website.

