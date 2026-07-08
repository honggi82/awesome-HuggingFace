[Figure 1]

### MotionEdit: Benchmarking and Learning Motion-Centric Image Editing

Yixin Wan1,2*, Lei Ke1, Wenhao Yu1, Kai-Wei Chang2, Dong Yu1 1Tencent AI, Seattle 2University of California, Los Angeles

Project Page: https://motion-edit.github.io

# arXiv:2512.10284v2[cs.CV]14Dec2025

|[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>MotionNFT (Ours)|
|---|

Original Nano-Banana GPT-Image-1

###### Ground Truth

###### Seedream Hunyuan Image

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

“Change the girl's right arm pose from an extended gesture to a hand-on-hip pose, keeping her winking expression, blue hair, and outfit unchanged.”

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

“Make the right foot step on the red apple, crushing it.”

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

“Have the female dental professional turn away from the male patient and bend down to adjust the dental chair.”

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

“Make the orange car fall off the cliff, positioning it mid-air with the front angled downwards towards the water.”

Figure 1. Case studies of MotionNFT against closed-source commercial baselines: Nano-Banana [8], GPT-Image-1 [22], Seedream [26], and Hunyuan Image [3]. Red circles highlight failure regions (e.g., failing to displace the car in the bottom row, generating an artifact ‘foot” in the second row). While commercial models maintain high visual quality, they struggle to ground complex motion changes or maintain visual consistency. MotionNFT follows these dynamic instructions, ensuring geometric alignment with the ground truth.

#### Abstract

troduce MotionEdit-Bench, a benchmark that challenges models on motion-centric edits and measures model performance with generative, discriminative, and preferencebased metrics. Benchmark results reveal that motion editing remains highly challenging for existing state-of-the-art diffusion-based editing models. To address this gap, we propose MotionNFT (Motion-guided Negative-aware FineTuning), a post-training framework that computes motion alignment rewards based on how well the motion flow between input and model-edited images matches the groundtruth motion, guiding models toward accurate motion transformations. Extensive experiments on FLUX.1 Kontext and Qwen-Image-Edit show that MotionNFT consistently improves editing quality and motion fidelity of both base models on the motion editing task without sacrificing general editing ability, demonstrating its effectiveness.

We introduce MotionEdit, a novel dataset for motioncentric image editing—the task of modifying subject actions and interactions while preserving identity, structure, and physical plausibility. Unlike existing image editing datasets that focus on static appearance changes or contain only sparse, low-quality motion edits, MotionEdit provides highfidelity image pairs depicting realistic motion transformations extracted and verified from continuous videos. This new task is not only scientifically challenging but also practically significant, powering downstream applications such as frame-controlled video synthesis and animation.

To evaluate model performance on the novel task, we in-

*Work done during internship at Tencent AI Lab in Seattle, contact email: elaine1wan@cs.ucla.edu

###### MotionEdit (Ours)

###### Existing Image Editing Benchmarks

[Figure 30]

||
|---|

Bottleneck 1: Lack of Motion Edit Data / Categorization

Bottleneck 2: Poor Data Quality for Motion Editing

Solution: High-Quality, Targeted Motion Editing Data

[Figure 32]

[Figure 33]

[Figure 34]

###### OmniEdit

###### InstructP2P

###### MagicBrush

MotionEdit (Ours)

UltraEdit

[Figure 35]

[Figure 36]

Input Image

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

“have them holding a wand”

Ground truth image is unfaithful: the subject is not holding the wand, and half of the wand is nonexistent.

Target Image

Edit Prompt

“turn the color of toolbox to red”

“make Doraemon and the boy turn their bodies and head to face each other for direct interaction.”

“Change the baseball to a fireball”

“Make the woman sip from her coffee cup, looking down.”

“have them holding a rifle”

“make the cow lift its leg up”

[Figure 46]

Ground truth image not faithful: the subject not holding the rifle.

Require mask input as guidance.

[Figure 47]

[Figure 48]

Only static change in color, no motion / action change

No motion / action change.

Ground truth image not faithful: leg not lifted. Ground truth image is inconsistent: change in body &

[Figure 49]

[Figure 50]

High-quality data triplets: input image, edit prompt, and ground truth edited image from continuous sequences

Ground truth image not of subjects.

Ground truth image is inconsistent: Artifacts on subject face, missing / hallucinated details, ...

faithful: no fireball, distortion / artifacts on appearance, etc..

[Figure 51]

[Figure 52]

No interaction change between subjects.

[Figure 53]

[Figure 54]

leg color, missing rope, ... Rich data in different motion categories with subject & objects

- Figure 2. Comparison of existing image editing benchmarks with MOTIONEDIT. Prior datasets lack motion-edit supervision, either focusing on appearance edits or offering low-quality action changes with artifacts. MOTIONEDIT fills this gap by providing instructionfollowing motion edits with paired input–target image data, enabling evaluation and training of motion-aware image editing models.

#### 1. Introduction

and MagicBrush [39] examples in Figure 2).

Instruction-guided image editing models have made remarkable progress recently [7, 14, 15, 22, 34], capable of transforming images based on natural language commands. While recent image editing models excel at performing appearance-only static edits that simply adjust color, texture, or object presence, they oftentimes fall short in accurately, faithfully, and naturally editing the motion, posture, or interaction between subjects in images. In this work, we aim at addressing this limitation in existing models through systematically formulating and studying motion editing as an independent and important image editing task.

We formally define the new task of motion image editing—editing that modifies the action, pose, or interaction of subjects and objects in an image according to a textual instruction, while preserving visual consistency in characters and scene. Motion editing aims at changing how subjects move, act, or interact, which is essential for applications such as frame-controlled video generation and character animation. However, existing image editing datasets and benchmarks suffer from two major bottlenecks in approaching the motion image editing task: First, they primarily focus on static editing tasks like appearance modification or replacement (e.g. OmniEdit [33] and UltraEdit [40] examples in Figure 2), neglecting the important aspect of motion editing in their data at all. Second, datasets that do include motion edits offer only a small amount of low-quality data, often with unfaithful or incoherent edit ground-truth that fail to execute the intended motion (e.g. InstructP2P [2]

To bridge this research gap, we curate MOTIONEDIT, a high-quality dataset and benchmark specifically targeting motion editing, consisting of paired input–target image examples extracted and validated from continuous highresolution video frames to ensure accurate, natural, and coherent motion changes. As shown in Figure 2, MOTIONEDIT captures realistic action and interaction changes that preserve identity, background, and style, in contrast to prior datasets where edit data is either static, unfaithful, or visually inconsistent. Moreover, our data is sourced from a large set diverse video sequences, ensuring the assessment of diverse sub-categories of motion image editing, such as posture, orientation, and interaction changes in Figure 5. Beyond constructing high quality editing data, we also devise evaluation metrics to evaluate motion edit performances of models. For discriminative evaluation, we by comparing the optical flow [11, 28, 30, 35, 36]—which captures the magnitude and direction of motion changebetween the input and model-edited images against the input–ground truth flow. For generative evaluation, we adopt Multimodal Large Language Model (MLLM)-based metrics to assess the fidelity, preservation, coherence, and overall quality of edited images. Additionally, we report pairwise win rates through head-to-head comparisons between overall edit quality of different models to reflect preference performance. Both quantitative and qualitative results across state-of-the-art image editing models on MOTIONEDIT-BENCH show that motion image editing re-

###### Original UniWorld-V1 BAGEL FLUX.1 [Kontext] Dev Ours

Ground Truth

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Prompt: “make Doraemon and the boy turn their bodies and head to face each other for direct interaction.”

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

Qwen-Image-Edit

Prompt: “Have the female dental professional turn away from the male patient and bend down to adjust the dental chair.”

- Figure 3. Qualitative comparison of state-of-the-art image editing models on MOTIONEDIT. Existing models fail to execute the required motion edits (e.g. UNIWORLD-V1 fail to edit subject postures and FLUX.1 KONTEXT produces severe identity distortions), while our MotionNFT-trained model accurately performs the intended motion edit that closely matches the ground-truth.

mains a challenging task for the majority of open-source image editing models.

replacement), they struggle with motion-related edits that require modifying actions or interactions (e.g., “make the man drink from the cup”). This gap largely stems from limitations in existing editing datasets. First, most benchmarks focus on static transformations—local texture changes, object replacement, or style transfer [2, 33, 40]—with little coverage of motion edits. Second, datasets containing motion edits are small and of low quality: motion categories are unclear, and the provided target edits are often unfaithful or physically implausible [2, 16, 39]. As shown in Fig. 2, these models frequently fail to achieve intended action changes and introduce visual artifacts, undermining both training supervision and evaluation reliability. These limitations underscore a key challenge in motion image editing: building datasets with precise motion-edit instructions and high-quality, faithful edited targets that preserve appearance and scene context while accurately reflecting the intended action changes.

To improve existing image editing models on the motion editing task, we further propose Motion-guided Negativeaware FineTuning (MOTIONNFT), a post-training framework for motion editing that extends DiffusionNFT [41] to incorporate motion-aware reward signals. MotionNFT leverages the motion alignment measurement between input-edit and input-ground truth optical flows to construct a reward scoring framework, providing targeted guidance on motion direction and magnitude in training. As illustrated in Figure 3, MotionNFT enables models to perform accurate, geometrically consistent motion edits. Quantitative results in Table 1 further shows substantial improvement across all metrics over prior approaches. For instance, MOTIONNFT achieves over 10% improvement in overall quality and over 12% on pairwise win rates when applied on FLUX.1 Kontext [14]).

The key contributions of our paper are three-fold:

Motion Estimation in Images. Motion estimation is a long-standing problem in computer vision. Modern approaches rely on optical flow, which predicts per-pixel displacement between two images [11, 28, 30, 35]. Recent work such as UniMatch [36] further advances largedisplacement estimation by formulating optical flow as a global matching problem unified with stereo tasks. Inspired by the effectiveness of optical flow in capturing finegrained motion changes, we propose a motion-centric reward framework based on optical flow, which quantitatively measures how accurately a model performs the intended motion edit in synthesized images.

- • We systematically define and study the novel task of motion image editing.
- • We construct MOTIONEDIT, a high quality dataset and benchmark for motion image editing, containing diverse and accurate edit data sourced from video frames.
- • We propose MOTIONNFT, a post-training framework that integrates optical flow–based rewards into DiffusionNFT to guide motion edit improvements.1

#### 2. Related Works

Image Editing. Recent advances in text-to-image (T2I) diffusion models have greatly improved text-guided image editing [2, 14, 20, 22, 34, 39]. While current models handle static appearance edits well (e.g., color changes or object

Reinforcement Learning for Image Generation. Policygradient methods such as PPO [23, 25] and GRPO [17, 27] have been explored for improving image generation. More recently, DiffusionNFT [41] introduces negativeaware finetuning, which contrasts positive and negative gen-

1Dataset, code, and evaluation toolkit will be released upon acceptance.

erations during the forward diffusion process to obtain an implicit policy improvement direction, steering the model toward high-reward outcomes while repelling low-reward ones. UniWorld-V2 [15] extends DiffusionNFT by integrating an MLLM-based online scoring pipeline for rating editing aspects like prompt compliance and style fidelity. However, current RL-based post-training frameworks remain motion-agnostic: they emphasize semantic correctness and visual details, yet offer no supervision on how subjects and objects should move for motion-centric edits.

#### 3. Dataset Construction

##### 3.1. Problem Definition and Categorization

The task of motion image editing has not been comprehensively explored in prior works. Therefore, we first provide a systematic definition of this novel task.

Motion Image Editing. Given an input image and a natural-language instruction specifying a target motion change (e.g. “make the woman drink from the cup”), the goal is to synthesize an edited image where: (1) the edited motion faithfully reflects the intended action; (2) the resulting pose or interaction is physically plausible and respects articulated constraints (e.g., “slightly open his eyes”); (3) non-edited factors like appearance, background, and viewpoint remains consistent. Unlike traditional appearancefocused editing, motion editing requires models to interpret the instructed motion and translate it into coherent spatial changes in the image, requiring fine-grained spatial and kinematic understanding.

##### 3.2. Dataset Construction Pipeline

As discussed in Section 2, existing image editing datasets and benchmarks lack reliable ground-truth targets that correctly execute the instructed motion while preserving subject identity and scene context. Prior datasets either introduce artifacts and hallucinations, alter appearance, or unintentionally shift viewpoint or scale. Sourcing high-quality motion edit ground truth remains a challenging problem. Instead of synthesizing edited targets as in prior work [2, 39], we propose a video-driven data construction pipeline that mines paired frames from dynamic video sequences to produce high-quality (input image, edit instruction, target image) triplets. These data reflect naturally occurring and coherent motion transitions grounded in video kinematics. Full details on dataset construction are in the “Additional Dataset Construction Details” Appendix section.

Video Collection To obtain frame pairs capturing clean motion transitions, we first explored conventional human action datasets such as HAA500 [6] and K400 [12]. Although diverse, these datasets often suffer from problems like low resolution, motion blur, rapid viewpoint shifts, etc., making them unsuitable for extracting faithful pre-/post-edit pairs

that preserve identity and background consistency.

In contrast, recent Text-to-Video (T2V) models (e.g. Veo-3 [9], Kling-AI [13]) produce visually sharp, temporally smooth videos with stable subjects and backgrounds. We therefore draw from two publicly released T2V video collections—ShareVeo3 [31] and the KlingAI Video Dataset [21]—as our initial pool of candidate videos. We then apply further processing to extract high-quality frame pairs for our MOTIONEDIT dataset.

Frame Extraction and Automatic Validation Given the video pool, our goal is to identify frame pairs that exhibit meaningful motion changes while preserving all nonmotion factors. We segment each video into 3-second windows and sample the first and last frame of each segment, providing a broad and efficient set of candidate motion transitions. However, many sampled pairs are unusable due to camera motion, subject disappearance, environmental changes, or visual degradation. Motivated by the recent success of LLM/MLLM-based data filtering [4, 5, 10, 32], we leverage Google’s Gemini [29] model to automatically filter these cases at scale. We prompt Gemini to evaluate each frame pair along three critical dimensions:

- • Setting Consistency. Verify that background, viewpoint, and lighting remain stable despite subject motion.
- • Motion and Interaction Change. Identify interaction states in each frame and summarize the primary motion transition (e.g., “not holding cup → drinking”). The model also judges whether the change is significant enough to constitute a meaningful motion edit.
- • Subject Integrity and Quality. Ensure the main subjects are present, identifiable, and artifact-free, avoiding cases with occlusion, shrinkage, hallucinations, and distortions.

Based on these criteria, the MLLM outputs a binary keep/discard decision. A pair is accepted only if (1) the scene remains stable, (2) the motion change is non-trivial, (3) subjects are consistent and coherent, and (4) both frames maintain high visual quality. This filtering is essential for obtaining high-quality motion edit triplets for our dataset.

##### 3.3. Editing Prompt Construction

While the validated frame pairs provide reliable visual reference, their corresponding edit instructions must be clear, natural, and semantically faithful to the observed change. We convert the MLLM-generated motion-change summaries into user-style editing prompts by following the prompt refinement procedure of Wu et al. [34]. This step removes unnecessary analysis details and standardizes prompts into imperative form (e.g. “Make the woman turn her head toward the dog.”), ensuring alignment between the described edit and the actual motion transition in data.

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

###### Editing Instruction Refinement

MLLM Data Quality Judge

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Raw Video

Interaction Summary: "The large rabbit, ridden by the woman, transitions from a static pose to an active running motion."

[Figure 84]

[Figure 85]

Frame Pair 1

Frame Pair 4

[Figure 86]

###### MLLM Rewrite

Chunking

Setting Consistency: False Interaction Change Significance: Minor

Setting Consistency: True Interaction Change Significance: Significant

“Make the large white rabbit run forward dynamically, with its legs lifted in a powerful stride. The woman riding should lean slightly forward, her hair flowing backward, adapting to the fast motion.”

Analyze the frames according to the criteria:

Interaction Summary: The rabbit and rider have moved forward slightly ...

Interaction Summary: The large rabbit transitions from a static pose to running... Quality Analysis: Character Consistency - True ... Final Reasoning: “The visual setting is consistent, and the interaction shows a significant change ...”

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

First & End Frame Extraction

- 1. Visual Setting Consistency
- 2. Interaction Analysis
- 3. Quality & Character Analysis

###### ...

Quality Analysis: Character Consistency - True ... Final Reasoning: “The setting shows a camera adjustment (zoom/pan) ...”

Final Decision: Drop

Final Decision: Keep

...

- Figure 4. MotionEdit’s data construction pipeline. We segment raw videos, extract frame pairs, and automatically filter them using an MLLM data quality judge. For all kept pairs, we use a MLLM rewrite module to generate clean, motion-focused editing instructions. Our pipeline enables scalable construction of high-quality motion editing data and can be extended to much larger video corpora.

Pose / Posture Subject-Object Interaction

[Figure 91]

[Figure 92]

Inter-Subject

Interaction

[Figure 93]

[Figure 94]

Orientation / Viewpoint

Pose/Posture Locomotion/

[Figure 95]

[Figure 96]

Distance

Object State / Formation Inter-Subject Interaction

[Figure 97]

[Figure 98]

“Open the man's eyes, make him look directly at the camera, and add a subtle smile”.

“Start vacuuming the colorful confetti on the carpet.”

“Make Doraemon and the boy turn their bodies and heads to face each other for direct interaction..”

“Change the character's pose: straighten body from a slight crouch to an upright, active stance, extending the arm to spray graffiti on the wall.”.

“Turn the character to face away from the camera.”

“Have the female dental professional turn away from the male patient and bend down to adjust the dental chair.”

“Crush the red apple under the foot, breaking it into pieces and scattering ..."

“Reduce the large fireworks display above the building to a few falling embers.”

[Figure 99]

“Make the seagull land on the water, causing splashes.”

Locomotion / Distance

[Figure 100]

[Figure 101]

[Figure 102]

“Reduce the large fireworks display above the building to a few falling embers.”

[Figure 103]

[Figure 104]

“Turn the character to face away from the camera.”

[Figure 105]

[Figure 106]

“Start vacuuming the colorful confetti on the carpet.”

[Figure 107]

[Figure 108]

“Have the female dental professional turn away from the male patient and bend down to adjust the dental chair.”

- Figure 5. Example categories of data in MOTIONEDIT. Drawing from diverse video sources, our dataset captures a broad spectrum of motion transformations, including pose shifts, locomotion, viewpoint changes, and both subject–object and inter-subject interactions.

##### 3.4. Dataset Statistics

Our final MOTIONEDIT dataset consists of 10,157 motioneditable frame pairs, sourced from both Veo-3 and KlingAI video collections. Specifically, we obtain 6,006 samples from Veo-3 and 4,151 samples from KlingAI. We perform a random 90/10 train-test split, resulting in 9,142 training data and 1,015 evaluation data that constitutest MOTIONEDITBENCH. Each sample includes a source or input image, a target image exhibiting a real motion transition from the original video, and a precise motion edit instruction. As shown in Figure 5, data in MOTIONEDIT can be generally categorized into six motion edit types:

- • Pose / Posture: Changes in body configuration position (e.g. raising hand) while keeping identity and scene fixed.
- • Locomotion / Distance: Changes in subject’s spatial position or distance relative to the camera or environment.
- • Object State / Formation: Changes in the physical form or condition of an object (e.g., deformation, expansion).
- • Orientation / Viewpoint: Changes in subject’s facing direction or angle without position change.
- • Subject–Object Interaction: Changes in how a person or agent physically interacts with an object (e.g., holding).

• Inter-Subject Interaction: Changes in the coordinated motion between two or more subjects (e.g., facing).

##### 3.5. Data Motion Magnitude Comparison

To quantify and compare the amount of motion present in before-after editing pairs between MOTIONEDIT and other editing datasets, we randomly select 100 data from each dataset and calculate the overall pixel-level motion displacement between each input image and its corresponding edited target. We measure motion changes in the image pairs with optical flow, the calculation of which is explained later in Section 4.

Figure 6 reports the average input-target motion magnitude across 6 editing datasets. Prior datasets such as MagicBrush [39], AnyEdit [38], InstructPix2Pix [2], UltraEdit [40], and OmniEdit [33] contain relatively modest motion changes (typically around 0.05), whereas our MOTIONEDIT dataset exhibits substantially larger motion differences (0.19), representing 5.8× greater motion than MagicBrush and OmniEdit and 3× that of UltraEdit. This highlights our contribution of a significantly more challenging motion editing dataset with substantial motion transformation.

Avg.Input-TargetMotionDifference

0.20

0.19

Since our objective is to evaluate how accurately a model applies the intended action to subjects and objects, our reward function must quantify the alignment between modelpredicted motion and the ground-truth motion edit. Inspired by the use of optical flow for measuring motion between consecutive video frames, we adopt an optical-flow–based motion-centric scoring framework that treats each input–edit pair as an implicit “before–after” sequence.

0.15

0.10

0.07

0.06

0.04

0.05

0.03

0.03

0.00

MagicBrush AnyEdit InstructPix2Pix UltraEdit OmniEdit MotionEdit (Ours)

Dataset Name

Given a triplet X = (Iorig,Iedited,Igt), we compute optical flow fields using a pretrained estimator [36]. The predicted motion is Vpred = F(Iorig,Iedited) and the groundtruth motion is Vgt = F(Iorig,Igt), where each flow lies in RH×W×2. We normalize both flows by the image diagonal to ensure scale consistency across resolutions.

- Figure 6. Comparison of motion difference between before- and post-edit images in different datasets [2, 33, 38–40]. Our MOTIONEDIT dataset achieves the most significant motion changes.

#### 4. Learning Motion Image Editing

##### 4.1. Preliminaries

Motion magnitude consistency. We measure the deviation between flow magnitudes using a robust ℓ1 distance: Dmag = HW1 i,j(∥V˜ pred(i,j) − V˜ gt(i,j)∥1 + ε)q, where q ∈ (0,1) is a constant term to suppress outliers.

Flow Matching Models Recent progress in diffusion models has shifted from Denoising Diffusion Probabilistic Models (DDPMs) [24] to Flow Matching Models (FMMs) [14]. Given noisy sample zt and conditioning c, FMMs reformulate the noise prediction process in DDPMS by estimates a deterministic velocity field v that transports zt toward its clean counterpart. As a result, inference for FMMs reduces to the ODE dzt = vθ(zt,t,c)dt, which enables efficient generation compared to DDPM sampling.

Motion direction consistency. We compute cosinebased directional error between the unit flow vectors edir(i,j) = 12 1 − vˆpred(i,j)⊤vˆgt(i,j) , and weight each pixel by its relative ground-truth motion magnitude. The directional misalignment is Ddir = i,j w(i,j)edir(i,j)

i,j w(i,j)+ε .

Movement regularization. To prevent trivial edits that make almost no motion, we compare the average predicted and ground-truth magnitudes: Mmove = max{0, τ+12m¯ gt− m¯ pred}, where τ is a small positive margin and m¯ denotes the spatial mean.

Diffusion Negative-aware Finetuning (NFT) DiffusionNFT [41] enhances FMM reward training by learning not only a positive velocity v+(xt,c,t) that the model should move toward, but also a negative velocity v−(xt,c,t) that it should avoid. The training objective is:

Combined reward. We aggregate the three terms into a composite distance Dcomb = α Dmag +β Ddir +λmoveMmove where α, β, and λmove are constants that balances term scales and weightings. The composite distance is then normalized and clipped: D˜ = clip (Dcomb − Dmin∗ )/(Dmax − Dmin∗ ),0,1 , and converted into a continuous reward rcont = 1 − D˜. Finally, we quantize it into 6 discrete reward levels: rmotion = 15 round(5rcont) ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. The resulting scalar reward is used to compute optimality rewards and update the policy model vθ under the DiffusionNFT objective (Eq. 4). Figure 7 illustrates the MotionNFT reward pipeline.

L(θ) =Ec, πold(x0|c), t r ∥vθ+(xt, c, t) − v∥22

(1)

+ (1 − r) ∥vθ−(xt, c, t) − v∥22 ,

where v is the target velocity and vθ+,vθ− are defined as linear combinations of the old and current policies:

vθ+(xt, c, t) = (1 − β) vold(xt, c, t) + β vθ(xt, c, t), vθ−(xt, c, t) = (1 + β) vold(xt, c, t) − β vθ(xt, c, t).

(2)

A key challenge is obtaining a calibrated reward r that accurately reflects whether a sample should be treated as “positive”. Since raw rewards may differ in scale or distribution, DiffusionNFT transforms them into an optimality reward:

#### 5. Experiments

##### 5.1. Experimental Setup

rraw(x0, c) − Eπold(·|c) rraw(x0, c) Zc

- 1

- 2

- 1

- 2

, −1, 1 ,

r(x0, c) =

+

clip

We provide important details of our experimental setups. Full details are reported in the Additional Experiment Details Appendix section.

(3) where Zc is a normalization factor (e.g., the global reward standard deviation). This normalization stabilizes learning and ensures consistent positive/negative assignment across prompts and reward models.

MotionNFT Training We use FLUX.1 KONTEXT [DEV] [14] and QWEN-IMAGE-EDIT [34] as base models for MotionNFT training. Following Lin et al. [15]’s implementation, we use Fully Sharded Data Parallelism (FSDP) for text encoder and apply gradient checkpointing in training for GPU memory usage optimization. To im-

##### 4.2. MotionNFT: Motion-Aware Reward for NFT

We introduce MotionNFT, a motion-aware reward framework designed for NFT training on motion-editing tasks.

MotionEdit-Bench Overall↑ Fidelity↑ Preservation↑ Coherence↑ Motion Alignment Score (MAS)↑ Win Rate↑

Model

Instruct-P2P [2] 1.30 1.32 1.29 1.29 34.15 16.09 AnyEdit [38] 1.31 1.32 1.32 1.30 35.11 16.88 MagicBrush [39] 1.50 1.58 1.47 1.44 44.24 19.51 UltraEdit [40] 2.42 1.88 2.09 2.13 47.18 28.33 UniWorld-V1 [15] 2.87 2.96 2.76 2.88 55.37 41.14 Step1X-Edit [18] 4.02 4.04 3.99 4.02 52.98 61.14 BAGEL [7] 4.10 4.24 4.01 4.06 51.83 61.46

FLUX.1 Kontext [Dev] [14] 3.84 3.89 3.79 3.83 53.73 57.71

+MOTIONNFT (Ours) 4.25 4.33 4.16 4.25 55.45 64.95 Qwen-Image-Edit [34] 4.65 4.70 4.59 4.66 56.46 72.80

###### +MOTIONNFT (Ours) 4.72 4.79 4.63 4.74 57.23 73.67

- Table 1. Quantitative results on MOTIONEDIT-BENCH. Among existing methods, Step1X-Edit and BAGEL achieve the strongest motionediting performance, while diffusion-based editors such as AnyEdit and MagicBrush perform poorly across both generative and discriminative metrics. FLUX.1 Kontext and Qwen-Image-Edit models trained with MotionNFT yields the best overall results: for both models, applying MotionNFT boosts all generative metrics, MAS and pairwise win rate.

###### Candidate Sampling

##### 5.2. Evaluation Metrics

Optical Flow

MotionNFT Reward

[Figure 109]

V ∈R ^(H, W, 2)

Generative Metrics. Following Luo et al. [19] and Lin et al. [15], we use an MLLM to evaluate edited images with four generative metrics: Fidelity, Preservation, Coherence, and their Overall average. We choose to use Google’s Gemini [29] as the MLLM evaluator and use evaluation prompts adapted from the “action” category of Luo et al. [19].

Motion Magnitude Term: 0.06

UniMatch

Motion Direction Consistency: 0.12

“Make the woman actively bite the burrito she is holding, with her mouth closed...”

[Figure 110]

NFT Training

Movement Regularization Term: 0.19

FMM

Ground Truth

Discriminative Motion Alignment Score (MAS). To complement the MLLM generative metric with deterministic assessment, we introduce an optical flow–based Motion Alignment Score (MAS) to measure how well the model understands and performs the correct motion change in images. MAS combines the motion magnitude consistency term Dmag and the motion direction consistency term Ddir from Section 4 into a single motion alignment metric: Dovl = α Dmag + (1 − α)Ddir, where α is a constant term balancing scales. Then, we normalize Dovl and convert it into: MAS = 100.00 · (1 − clip (Dovl − dmin)/(dmax − dmin), 0,1 ). Higher scores indicate closer alignment. If the predicted motion is nearly static compared to ground truth, i.e., E[mpred]/E[mgt] < ρmin, we assign MAS = 0.

[Figure 111]

[Figure 112]

[Figure 113]

MotionNFT Reward: 0.80

UniMatch

- Figure 7. MotionNFT’s Reward Scoring pipeline. For each sampled model-edited image, we measure the alignment between the input-generated optical flow and the input-ground truth optical flow, obtaining the final reward score.

prove models’ motion image editing capabilities while preserving general image editing ability, we employ a multiscore reward formulation with a weighted combination of

- (i) 50% our optical flow-based Motion Reward rmotion and
- (ii) 50% MLLM reward proposed by Lin et al. [15]. For MLLM-based evaluation, we serve a QWEN2.5-VL-32BINSTRUCT [1] model via vLLM on a separate node that performs online scoring throughout training. The optical flow component of our reward leverages a lightweight UniMatch model (335.6M parameters), which we run directly on the training nodes to provide efficient motion-level guidance.

##### 5.3. Quantitative Evaluation Results

Table 1 reports quantitative performance of 9 image editing models on MOTIONEDIT-BENCH. The first 4 columns shows MLLM generative ratings on a 0–5 scale. Our optical flow-based MAS metric measures motion consistency on a 0–100 scale. The Win Rate reflects the percentage of pairwise comparisons in which a model’s output received a higher average MLLM score than a competing one.

Benchmarked Image Editing Models We evaluate 9 opensource models on MOTIONEDIT-BENCH: Instruct-P2P [2], MagicBrush [39], AnyEdit [38], UltraEdit [40], Step1XEdit [18], BAGEL [7], UniWorld-V1 [15], FLUX.1 Kontext [Dev] [14], and Qwen-Image-Edit [34].

#1: Improved Motion Editing Quality. Across both base models, MotionNFT consistently improves all aspects of generation quality on motion editing, as measured by the generative evaluator. When applied to FLUX.1 KON-

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Image

Input

Submerge the orca into the water, with its dorsal fin and upper body visible, and position it to obscure the polar bear.

Move Santa Claus from the fireplace to stand on the wooden floor to the left of the fireplace. Remove the surrounding smoke, adjust his pose to standing and waving ...

Prompt

Make the man charge forward aggressively. Give him an angry expression and a clenched right fist, removing the object he was previously holding.

Reposition the convoy of white semi-trucks further to the right. Add several white semi-trucks parked at the factory loading docks in the background.

Have the female dental professional turn away from the male patient and bend down to adjust the dental chair.

Change the woman's hand posture: bring her hands together, clasped at chest height.

Make Doraemon and the boy turn their bodies and heads to face each other for direct interaction.

Edit

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

Qwen-Image-

Edit

|[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>[Figure 134]|
|---|

+MotionNFT

(Ours)

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Ground

Truth

- Figure 8. Qualitative examples of our MotionNFT. The baseline QWEN-IMAGE-EDIT [34] model often fails to execute the instructed motion (circled regions), producing edits that do not match the required action change (red underlines). With MotionNFT training, the model succeeds in performing precise motion edits that closely align with the ground-truth transformations.

# 1: Existing models struggle to perform correct motion edits. We observe that even state-of-the-art opensourced image editing models like FLUX.1 Kontext and Qwen-Image-Edit struggle to correctly perform motioncentric changes like turning body directions. These models often preserve the original pose or only apply superficial appearance changes. This highlights the crucial bottleneck in translating motion-related language instructions into coherent image subject / object transformations.

TEXT, MotionNFT raises the Overall score from 3.84 to

- 4.25 (+10.68%), with notable gains in Fidelity (+0.44) and Coherence (+0.42). For QWEN-IMAGE-EDIT, MotionNFT also improves the Overall score from 4.65 to 4.72.

- #2: Enhanced Motion Alignment. MotionNFT yields substantial improvements in MAS, highlighting its effectiveness in producing motion changes consistent with the ground-truth edits. On FLUX.1 KONTEXT, MotionNFT increases MAS from 53.73 to 55.45, while on QWENIMAGE-EDIT, MAS improves from 56.46 to 57.23. These gains are achieved despite the strong baselines and show that our flow-based reward provides meaningful guidance for learning spatial and motion-aware transformations.
- #3: Strong Pairwise Preference Performance. MotionNFT also achieves higher win rates relative to all evaluated models. For FLUX.1 KONTEXT, MotionNFT boosts win rate from 57.97% to 65.16% (+12.40%), and from 72.99% to 73.87% for QWEN-IMAGE-EDIT. These results show that MotionNFT produces more accurate motion edits that are more frequently preferred over outputs of other models.

- 5.4. Qualitative Evaluation Results

#2: MOTIONNFT improves motion editing capability. Training with MOTIONNFT enables Qwen-ImageEdit to produce outputs that more faithfully reflect the intended motion, e.g. rotating character directions, adjusts limb and torso positions to reflect bending or turning actions. Additionally, the resulting edits preserve identity and scene context while achieving the targeted motion change, closely matching the ground-truth transformations. These observations validates the effectiveness of incorporating motion-centric guidance in MotionNFT to execute meaningful, structure-aware motion edits that current image editing models consistently fall short in achieving.

Figure 3 and Figure 8 illustrate representative qualitative results on MOTIONEDIT.

##### 5.5. Ablation Studies

General Image Editing Performance To verify that MotionNFT preserves a model’s general editing ability, we follow previous work [15] and conduct evaluation on ImgEditBench [37], a comprehensive benchmark covering 8 editing subtasks. Table 2 shows that MotionNFT consistently improves or maintains performance across all categories for both FLUX.1 KONTEXT and QWEN-IMAGE-EDIT, even yielding higher overall scores. Results confirm that MotionNFT can enhance models’ motion editing performance without trading off general editing quality.

Model ImgEdit-Bench

Add Adj. Rpl. Rem. Bck. Stl. Hyb. Act. Ovl.↑ FLUX.1 Kontext 3.54 2.90 3.73 2.89 3.59 3.96 2.90 2.56 3.26

- + MOTIONNFT 3.71 3.28 3.93 3.05 3.72 4.41 2.99 2.85 3.50

Qwen-Image-Edit 4.20 3.70 4.22 4.20 4.17 4.60 3.55 4.03 4.08

- + MOTIONNFT 4.31 3.72 4.46 4.30 4.21 4.67 3.96 3.87 4.20

- Table 2. Results on ImgEdit-Bench [37] MotionNFT not only preserves, but oftentimes boosts general editing performances.

Comparison with MLLM-only Reward To verify the effect of MotionNFT’s supervision, we compare MotionNFT against the MLLM-only RL framework in UniWorldV2 [15]. Table 3 shows that while MLLM-only training yields modest improvements over the base models, MotionNFT consistently achieves higher overall edit quality, better motion alignment, and superior win rates across both base models. These results demonstrate that incorporating optical flow-based motion guidance yields more targeted and effective motion-editing capabilities.

Model

MotionEdit-Bench

Overall. ↑ MAS ↑ Win Rate ↑ FLUX.1 Kontext 3.84 53.73 57.71

+ UniWorld-V2[15] 4.20 54.58 63.76 +MOTIONNFT (Ours) 4.25 55.45 64.95

Qwen-Image-Edit 4.65 56.46 72.80 + UniWorld-V2[15] 4.70 56.46 72.56 +MOTIONNFT (Ours) 4.72 57.23 73.67

- Table 3. Comparison to training with MLLM-based reward [15] only. Incorporating MotionNFT yields noticeable improvements MLLM-scored Overall editing quality, optical flow-based Motion Alignment Score, and the pairwise Win Rate across all models.

#### 6. Conclusion

We introduced MOTIONEDIT, a high-quality dataset and benchmark for the novel motion image editing task, aiming at correct modifying subject actions and interactions in images while preserving identity and scene consistency. To improve model performance on this challenging task, we proposed MOTIONNFT, a motion-guided negative-aware

finetuning framework that integrates an optical-flow–based motion reward for training. MotionNFT provides supervision on motion magnitude and direction, enabling models to understand and perform motion transformations that existing models consistently struggle with. Both quantitative and qualitative experiment results demonstrate that MotionNFT delivers consistent gains across generative quality, motion alignment, and preference win rate on two strong base models, FLUX.1 Kontext and Qwen-Image-Edit.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 7
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022. 2, 3, 4, 5, 6, 7
- [3] Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025. 1, 5, 9
- [4] Derin Cayir, Renjie Tao, Rashi Rungta, Kai Sun, Sean Chen, Haidar Khan, Minseok Kim, Julia Reinspach, and Yue Liu. Refine-n-judge: Curating high-quality preference chains for llm-fine-tuning. arXiv preprint arXiv:2508.01543, 2025. 4
- [5] Ruibo Chen, Yihan Wu, Lichang Chen, Guodong Liu, Qi He, Tianyi Xiong, Chenxi Liu, Junfeng Guo, and Heng Huang. Your vision-language model itself is a strong filter: Towards high-quality instruction tuning with data selection. In Findings of the Association for Computational Linguistics ACL 2024, pages 4156–4172, 2024. 4
- [6] Jihoon Chung, Cheng hsin Wuu, Hsuan ru Yang, Yu-Wing Tai, and Chi-Keung Tang. Haa500: Human-centric atomic action dataset with curated videos. In ICCV 2021. 4
- [7] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2, 7, 5, 8
- [8] Google. Gemini image generation api, 2025. https: //ai.google.dev/gemini-api/docs/imagegeneration/. 1, 5, 9
- [9] Google DeepMind. Veo 3. https://deepmind. google/models/veo/, 2025. Accessed: 2025-11. 4
- [10] Erik Henriksson, Otto Tarkka, and Filip Ginter. Finerweb10bt: Refining web data with llm-based line-level filtering. In Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025), pages 258–268, 2025. 4

- [11] Shihao Jiang, Dylan Campbell, Yao Lu, Hongdong Li, and Richard Hartley. Learning to estimate hidden motions with global motion aggregation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9772– 9781, 2021. 2, 3
- [12] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950,

2017. 4

- [13] Kuaishou Technology. Kling ai. https://app. klingai.com/global/image-to-video/, 2025. Accessed: 2025-11. 4
- [14] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 2, 3, 6, 7, 4, 5, 8

- [15] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang, Yunyang Ge, et al. Uniworld: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147, 2025. 2, 4, 6, 7, 9, 3, 5, 8
- [16] Haonan Lin, Yan Chen, Jiahao Wang, Wenbin An, Mengmeng Wang, Feng Tian, Yong Liu, Guang Dai, Jingdong Wang, and Qianying Wang. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. Advances in Neural Information Processing Systems, 37:115712–115756, 2024. 3
- [17] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 3
- [18] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 7, 3
- [19] Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, et al. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909, 2025. 7, 2
- [20] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations. 3
- [21] Nyuuzyou. Klingai video dataset. https : / / huggingface . co / datasets / nyuuzyou / klingai, 2025. Accessed: 2025-05. 4

- [22] OpenAI. Image generation api, 2025. https://openai. com/index/image-generation-api/. 1, 2, 3, 5, 9
- [23] Allen Z Ren, Justin Lidard, Lars Lien Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy optimization. In CoRL 2024 Workshop on Mastering Robot Manipulation in a World of Abundant Data. 3
- [24] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022. 6
- [25] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. 3
- [26] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward nextgeneration multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 1, 5, 9
- [27] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3
- [28] Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz. Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8934–8943,

2018. 2, 3

- [29] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 4, 7
- [30] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020. 2, 3
- [31] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models.

2024. 4

- [32] Weizhi Wang, Yu Tian, Linjie Yang, Heng Wang, and Xifeng Yan. Open-qwen2vl: Compute-efficient pre-training of fully-open multimodal llms on academic resources. arXiv preprint arXiv:2504.00595, 2025. 4
- [33] Cong Wei, Zheyang Xiong, Weiming Ren, Xinrun Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. arXiv preprint arXiv:2411.07199, 2024. 2, 3, 5, 6
- [34] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu,

Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 2, 3, 4, 6, 7, 8

- [35] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8121– 8130, 2022. 2, 3
- [36] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 2, 3, 6, 1
- [37] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 9
- [38] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. arXiv preprint arXiv:2411.15738, 2024. 5, 6, 7, 3
- [39] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. In Advances in Neural Information Processing Systems, 2023. 2, 3, 4, 5, 7
- [40] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale, 2024. 2, 3, 5, 6, 7
- [41] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025. 3, 6, 1

[Figure 142]

### MotionEdit: Benchmarking and Learning Motion-Centric Image Editing

## Supplementary Material

In this supplementary material, we present additional details on our method design in Section 7. We also present additional experimental setups, metric design, and human validation of metric in Section 8. Furthermore, we provide results of ablation experiments in Section 9.1 to verify the effectiveness of our MOTIONNFT method. Lastly, we visualize qualitative examples comparing our method to both open-source and closed-source commercial models in 9.2, highlighting the failure cases in these models and pointing towards future research direction.

#### 7. Additional Method Details

##### 7.1. Preliminaries

In T2I diffusion models, the forward noising process perturbs clean data x0 from real distribution π0 by adding a scheduled Gaussian noise ϵ ∼ N(0,I). The model then learns to reverse the noise and output clean images. The shift from Denoising Diffusion Probabilistic Models (DDPMs) to Flow Matching models (FMMs) is essentially a change in the prediction target of the model, from predicting the added noise itself (DDPM) to estimating a “velocity field” from the noise sample to the clean sample (FMM).

In mathematics formulation, FMMs define zt = αtx0 + σtϵ to be a noisy interpolated latent at timestep t between the initial clean x0 and the noise ϵ, where αt and σt defines the scheduled noise at t. Then, for noisy sample zt and textual context c, a FMM vθ is trained to directly approximate the target constant velocity field v = dα

dt x0 + dσ

dt ϵ by minimizing the objective:

t

t

0∼π0,ϵ∼N(0,I) ∥vθ(xt,t,c) − v∥22 .

LFM(θ) = Et,x

This velocity prediction allows for efficient inference by solving the deterministic Ordinary Differential Equation (ODE), dzt = vθ(zt,t,c)dt for the forward process.

##### 7.2. Diffusion Negative-aware Finetuning (NFT)

DiffusionNFT [41] aims at finding not only the “positive velocity” v∗(xt,t,c) = v+(xt,t,c) that the model learns to predict, but also identifying the “negative velocity” v−(xt,t,c) component that the model should steer away from. The training objective of DiffusionNFT is:

L(θ) = Ec,πold(x0|c),t r vθ+(xt, c, t) − v 22

(4)

+ (1 − r) vθ−(xt, c, t) − v 22

Where v is the target velocity, vθ+ and vθ− are the implicit positive policy and implicit negative policy, defined as com-

binations of the old policy and current training policy:

vθ+(xt, c, t) := (1 − β) vold(xt, c, t) + β vθ(xt, c, t), vθ−(xt, c, t) := (1 + β) vold(xt, c, t) − β vθ(xt, c, t).

(5)

Naturally, we need an optimal reward r to accurately estimate the likelihood of the current action to fall into the “positive” subset of all samples. However, real-world reward models might differ in score distributions and scales. To this end, DiffusionNFT transforms raw rewards rraw into the optimality reward:

rraw(x0, c) − Eπold(·|c)rraw(x0, c) Zc

- 1

- 2

- 1

- 2

, −1, 1

+

clip

r(x0, c) :=

(6) Where Zc is a normalizing factor (e.g. standard deviation of global rewards).

##### 7.3. MotionNFT: Motion-Aware Reward for NFT

We propose a optical flow-based motion-centric reward scoring framework for our MotionNFT method to compute how closely model-predicted motion matches the ground-truth motion. Our reward scoring process is illustrated as follows:

###### 7.3.1. Motion Calculation

Optical Flow Calculation Given two images I0 and I1, an optical flow estimation model [36] F quantifies motion flow between them with V = F(Iorig,Iedited) ∈ RH×W×2, where V is a vector field that represents the motion of each pixel with a 2D vector. In our case, given an input triplet X = (Iorig,Iedited,Igt) containing triplets of the original image Iorig, the model-edited image Iedited, and the ground truth image Igt, we first calculate the optical flow between the input image and the model-edited image: Vpred = F(Iorig,Ipred). Then, we construct the motion reward rm(X) to quantify the level of alignment between Vpred and the ground truth motion flow derived from the input and the ground-truth edited image Vgt = F(Iorig,Igt) with three consistency terms: a motion magnitude consistency term, a motion direction consistency term, and an movement regularization term.

Flow normalization. Let Vpred(i,j) ∈ R2 and Vgt(i,j) ∈ R2 denote the optical flow vectors at pixel (i,j) for the model-edited and ground-truth edited images, respectively. For an image of height H and width W, we normalize the flows by the image diagonal to make the displacement magnitude comparable across resolutions:

Vgt(i,j) √H2 + W2

Vpred(i,j) √H2 + W2

V˜ pred(i,j) =

,V˜ gt(i,j) =

.

###### 7.3.2. Reward Calculation

Motion Magnitude Consistency Term. We first measure how closely the predicted flow magnitudes match the ground truth using a robust ℓ1 distance. Let d(i,j) = V˜ pred(i,j) − V˜ gt(i,j), magnitude deviation Dmag can be calculated as:

H

W

1 HW

∥d(i,j)∥1 + ε q,

Dmag =

i=1

j=1

where ε > 0 is a small constant used for numerical stability and the exponent q ∈ (0,1) enables a robust variant of the ℓ1 distance that suppresses the influence of large outliers in the flow field while still preserving sensitivity to semantically meaningful deviations. Empirically, we set q = 0.4, which provides a stable trade-off between robustness and sensitivity for the motion-editing task.

Motion Direction Consistency Term. We additionally measure directional alignment between the two flow fields, while focusing on regions with non-trivial motion. Let:

mgt(i,j) = V ˜ gt(i,j) 2, mpred(i,j) = V ˜ pred(i,j) 2, and define unit flow directions:

V˜ pred(i,j) ∥V˜ pred(i,j)∥2 + ε

V˜ gt(i,j) mgt(i,j) + ε

, vˆpred(i,j) =

vˆgt(i,j) =

.

We compute a cosine-based directional error per pixel:

cos(i,j) = vˆpred(i,j)⊤vˆgt(i,j), edir(i,j) = 12 1−cos(i,j) , and weight each pixel by the relative ground-truth motion magnitude:

mgt(i,j) maxu,v mgt(u,v) + ε · 1 mgt(i,j) > τm ,

w(i,j) =

where τm is a small motion threshold and 1[·] is the indicator function. The directional misalignment Ddir can be calculated as:

w(i,j)edir(i,j) i,j w(i,j) + ε

Ddir = i,j

.

Movement Regularization Term. While Dmag and Ddir encourage the predicted flow to match the ground-truth motion, they do not by themselves prevent the model from collapsing to a nearly static edit. To discourage models from demonstrating this degeneration, we introduce a movement regularization term that compares the average motion magnitude of the predicted flow to that of the ground truth. We obtain the spatial means of mgt and mpred:

1 HW i,j

1 HW i,j

mgt(i,j), m¯ pred =

mpred(i,j),

m¯ gt =

and define the anti-identity hinge term to be:

Mmove = max 0, τ + 21 m¯ gt − m¯ pred , where τ > 0 is a small margin. Intuitively, Mmove penalizes trivial edits that keep the image nearly identical to the input.

Final: Motion-Centric Reward for training. Finally, we convert the optical flow-based alignment measure into a scalar reward for NFT training. We combine the 3 terms to obtain:

Dcomb = α Dmag + β Ddir + λmove Mmove, where α, β, and λmove are hyper-parameters that balance the scales between magnitude and directional alignments, as well as assigning a small proportion to discouraging undermotion. We normalize and clip the combined term:

D˜ = clip Dcomb − Dmin∗ Dmax − Dmin∗

, 0, 1 ,

where Dmin∗ is the lower bound of magnitude and directional terms calculated from a pair of duplicated inputs. We then construct the continuous optical flow-based reward:

rcont = 1 − L˜ ∈ [0,1], so that higher reward corresponds to better alignment with the ground-truth motion edit. Finally, to approximate discrete human ratings of edited images following [15, 19], we quantize the reward to 6 equally spaced levels:

1 5

round 5rcont ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0},

rfinal =

which is the final scalar reward signal for MotionNFT. During training, this raw reward score is further transformed to optimality rewards through group-wise normalization, and used to update the policy model vθ by optimizing the DuffusionNFT objective in Equation 4.

#### 8. Additional Evaluation Experiment Details

##### 8.1. Hyperparameter Setting

8.1.1. Reward and Metric MotionNFT Reward When calculating reward used for MOTIONNFT, we utilize three hyper-parameters to balance the three reward terms: Dcomb = α Dmag + β Ddir + λmove Mmove In our experiments, we set α = 0.7, β = 0.2, and λmove = 0.1. Not only does this balance the scales between magnitude and directional alignments, as well as assigning a small proportion to discouraging under-motion. MAS Calculation When quantifying the MAS between model-edited images and ground truth targets, we punish degenerate cases where the predicted motion is nearly static compared to the ground-truth motion as a hard failure case and assign the minimum score MAS = 0. This happens when EE[m[mpred]

gt] < ρmin), where ρmin is a parameter determining how harsh the punishment threshold would be. In our experiments, we set ρmin = 0.01.

###### 8.1.2. Model Training

Following the training setup in Lin et al. [15], we train all models with learning rate set to 3e − 4. For FLUX.1 Kontext [Dev] [14] as base model, we report results for 300 steps. For Qwen-Image-Edit [34] as base model, we report results for 210 steps. Due to computational limits, we set batch size to 2. For NFT training, during sampling, we set sampling inference steps to 6, number of images per prompt to 8, and number of groups to 24; for training, we set KL loss’ weight to 0.0001 and guidance strength to 1.0. For group filtering, we set the ban mean threshold to 0.9 and the standard deviation threshold to 0.05.

###### 8.1.3. Model Inference

During inference for Qwen-Image-Edit [34] and the trained checkpoints, we set number of inference steps to 28, true cfg scale to 4.0, and guidance scale to 1.0. For inference of FLUX.1 Kontext [Dev] [14] and its trained checkpoints, we set the same number of inference steps. For inference of other open-sourced models, we follow the hyper-parameter setup in the official repositories. For UniWorld [15], we set number of inference steps to 25 and guidance scale to 3.5. For AnyEdit [38], we set guidance scale to 3, number of inference steps to 100, and original image guidance scale to 3. For UltraEdit [40], we set number of inference steps to 50, image guidance scale to 1.5, and guidance scale to

- 7.5. For Step-1X [18], we set number of inference steps to 28 and true cfg scale to 6.0. For all other models, we set guidance scale to 3.5 and number of inference steps to 28.
- 8.2. Human Evaluation for Generative Metrics

To evaluate the alignment between the MLLM-based generative metric and human judgment on motion editing quality, we conduct a human annotation study. Our annotators are a group of voluntary participants who are college-level or graduate-level students based in the United States. All annotators are proficient in English and have prior familiarity with AI research, ensuring that they understand the evaluation criteria and the purpose of the study. Prior to beginning, all annotators were informed that the anonymized results of their annotations may be used for research purposes only.

###### 8.2.1. Annotation Interface and Instructions

We randomly sample 100 entries from MOTIONEDITBENCH for human evaluation, for which we further conduct random selection of outputs from 2 different models for comparison. To ensure consistency, all annotators completed the same set of comparison tasks. Each annotation instance consists of five visual components: (1) the Input Image to be edited, (2) the Ground Truth Edited Image demonstrating the ideal motion change, (3) a Text Editing Instruction, and (4–5) two model-generated edited outputs (labeled Model 1 and Model 2). The annotators were

asked to select which of the two model outputs better fulfill the requested motion edit, preserve the subject’s identity, and maintain overall visual coherence. Annotators were reminded that the Ground Truth serves as a reference only, not something to be matched pixel-wise. They were encouraged to evaluate edits based on correctness of motion transformation and appearance preservation of the final image. If both outputs appear to be comparably good, annotators were instructed to select the one that is slightly better.

Annotation Instruction. Before beginning annotation, participants read the following notice and instructions:

Warning: The set of model-synthesized images displayed below might contain explicit, sensitive, or biased content.

Thank you for being a human annotator for our study on the motion image editing task! By completing this form, you confirm voluntary participation in our research and agree to share your annotation data for research purposes only.

For each example, you will see: the Input Image, the Ground Truth Edited Image, an Editing Instruction, and two model-generated outputs. Your task is to determine which model output better follows the editing instruction while preserving the identity and appearance of the subject. Consider whether the edit is applied correctly, whether the subject remains consistent with the input, and whether the final image appears coherent and natural. You may consider the Ground Truth Image to be a “reference answer” of the ideal edit. If both outputs are similar in quality, choose the one you feel is slightly better.

###### 8.2.2. Human Evaluation Results.

Since all annotatros complete the same set of comparison tasks, each pair of model comparison was labeled by three independent annotators. Inter-annotator agreement between all human annotators, as measured by Fleiss’ κ, is 0.607, indicating good agreement among human raters. The aggregated agreement between human annotators and decisions made by the overall generative metric (averaged over Fidelity, Preservation, and Coherence) achieves a Fleiss’ κ score of 0.574, similarly demonstrating substantial alignment between human judgment and our metric. These results support the use of the MLLM-based generative evaluation metric as a practical and human-consistent measure of motion editing quality.

#### 9. Additional Evaluation Results 9.1. Ablation Studies

Balancing MLLM and Optical Flow-Based Rewards We investigate the optimal balancing strategy between our pro-

posed optical flow-based motion alignment reward (rmotion) and the MLLM-based semantic reward (rmllm) introduced in Uniworld-v2. Specifically, we adopt multi-objective reward NFT training with different weights for each reward. Table 1 summarizes the editing performance on our MOTIONEDIT-BENCH across varying balancing weights. We observe that relying solely on the motion reward (1.0 * Motion) leads to a performance degradation, indicating that geometric motion cues alone are insufficient for maintaining semantic fidelity. Conversely, while the pure MLLM reward (1.0 * MLLM) provides a strong baseline, it is consistently outperformed by the combined approach. The results demonstrate that the two objectives are complementary. The balanced configuration (λ = 0.5) yields the highest performance across all metrics for both FLUX.1 Kontext [Dev] [14] and Qwen-Image-Edit [34] backbones (achieving 4.25 and 4.72 Overall scores, respectively). This suggests that the optical flow reward effectively regularizes the MLLM guidance, improving motion alignment without compromising semantic coherence.

MotionEdit-Bench Ovl. ↑ Fid. ↑ Pre. ↑ Coh. ↑

Model

FLUX.1 Kontext 3.84 3.89 3.79 3.83 + 1.0 * Motion 3.60 3.62 3.60 3.59 + 0.3 * MLLM + 0.7 * Motion 4.22 4.29 4.15 4.23

- + 0.7 * MLLM + 0.3 * Motion 4.16 4.23 4.08 4.16

- + 1.0 * MLLM 4.20 4.28 4.11 4.21

###### + 0.5 * MLLM + 0.5 * Motion 4.25 4.33 4.16 4.25

Qwen-Image-Edit 4.65 4.70 4.59 4.66

###### + 1.0 * Motion 4.60 4.65 4.55 4.61

+ 0.3 * MLLM + 0.7 * Motion 4.72 4.81 4.61 4.74

- + 0.7 * MLLM + 0.3 * Motion 4.71 4.78 4.62 4.73

- + 1.0 * MLLM 4.70 4.80 4.57 4.73

###### + 0.5 * MLLM + 0.5 * Motion 4.72 4.79 4.63 4.74

Table 4. Ablation experiments on different weights for balancing the MLLM-based reward proposed by [15] and our optical flowbased motion alignment reward. Results show that combining both rewards on a 0.5:0.5 scale achieves best performance, outperforming MLLM-only reward training.

MLLM-only Reward vs. MotionNFT Figures 9 and 10 visualize the evolution of the Motion Alignment Score (MAS) during training for the MLLM-only reward in Lin et al. [15] and our MOTIONNFT reward. As explained in previous sections, MAS utilizes optical flow to quantify magnitude and directional alignment level between model-edited motion and ground truth motion achieved in the target image. We observe that relying solely on the MLLM-based semantic reward results in suboptimal motion alignment; for Qwen-Image-Edit (Fig. 10), the MAS even degrades significantly during the mid-training phase. In contrast, MotionNFT demonstrates robust and consistent improvement in MAS across both backbone models. By incorporating explicit motion guidance, our method prevents the model from overfitting to semantic cues at the expense of geomet-

[Figure 143]

- Figure 9. MAS vs. Training Steps on FLUX.1 Kontext [Dev] [14]. MAS quantifies the fidelity of the generated motion by calculating the optical flow alignment (considering both magnitude and direction) between the model’s edit and the ground truth target edit. While the MLLM-only baseline (blue) begins to regress after approximately 150 steps, MotionNFT (red) demonstrates steady improvement throughout training, ultimately achieving superior motion grounding by leveraging explicit motion guidance.

[Figure 144]

- Figure 10. MAS vs. Training Steps on Qwen-Image-Edit [34]. Results on other base models again shows that relying solely on semantic MLLM rewards leads to training regression in motion alignment. MotionNFT maintains prevents overfitting to semantic cues and achieving higher final MAS.

ric accuracy, achieving a significantly higher final MAS.

Qualitative Examples Figures 11 and 12 compare MotionNFT against the base models (Qwen-Image-Edit [34] and FLUX.1 Kontext [Dev] [14]) and their MLLM reward [15]guided counterparts. A recurring failure mode in the baselines is lack of “motion awareness”, where the model fails to interpret and execute the desired motion subject, direction, and magnitude from the editing prompts. For instance, in Figure 11 (Row 2), the Qwen baseline and the UniWorldV2 baseline fails to correctly move the subject’s right hand to operate the joystick, but rather placing both hands on it. In Row 6, both baselines mistakenly flip the caterpillar’s body direction when moving it towards the center of the flower. In contrast, MotionNFT successfully executes both edits, matching the ground truth desired motion.

Additionally, we observe that another failure mode in baseline methods is the preservation of setting and subject

identity. In Figure 11 (Row 5), both baselines completely remove the milk jar despite it being a main subject in the image. In the last row, both baselines remove the photo frame surrounding the woman that was in the original image, failing to preserve setting consistency. Similarly, in Figure 12 row 2, we observe that using [15]’s MLLM-only reward on FLUX.1 Kontext changes the penguin’s beak in to a black color, failing to preserve its appearance while also not correctly performing the motion edit. MotionNFT, on the other hand, achieves good preservation of subject’s appearance and setting consistency.

##### 9.2. Model Comparison

###### 9.2.1. Comparison with Open-Source Models

We compare MotionNFT against leading open-source editing models, including UniWorld-V1 [15], BAGEL [7], and FLUX.1 Kontext [Dev] [14]. Visual comparisons in Figure 13 reveal that these baselines frequently struggle with precise motion controllability:

- • Editing Inertia: Existing models may fail to execute significant geometric transformations, defaulting to the original pose. For instance, in the ”car cliff” scenario (Row 6), UniWorld-V1 fails to displace the vehicle, leaving it on the ledge with a flipped direction, while BAGEL and FLUX.1 lift the car but fail to capture the ”downward angled” physics of the fall. Similarly, in the ”lion” example (Row 2), all baselines fail to fully lower the head to the requested ”looking downwards” pose, whereas MotionNFT achieves accurate alignment with the ground truth.
- • Motion Misalignment: Existing models may fail to interpret and execute the subject part and direction of the motion change. For instance, in the gorilla example (Row 3), FLUX.1 Kontext fails to put the right hand into a fist. In the robot example (row 5), all baseline models fail to move the robot’s left arm but move the right one instead. MotionNFT, on the other hand, performs the correct motion change on the correct subject part.
- • Structural Distortion: When baselines do attempt large edits, they often introduce anatomical or semantic artifacts. In the ”gorilla” example (Row 3), FLUX.1 Kontext distort the hand structure when attempting the “fist” gesture. In the jug drinking example (Row 4), the baselines leave residual artifacts that distorts the jug, while our method cleanly executes the edit without artifacts.

###### 9.2.2. Comparison with Closed-source Models

We conduct selective case studies that compares MotionNFT with Qwen-Image-Edit as base model against leading closed-source commercial models, including NanoBanana [8], GPT-Image-1 [22], Seedream [26], and Hunyuan Image [3]. As visualized in Figure 14, these models still exhibit distinct failure modes for motion editing:

1. Semantic Hallucination and Structural Distortion:

When complex pose changes are required, baselines often introduce artifacts or unwanted semantic changes. In the ”apple” example (Row 2), Nano-Banana introduces artifact by creating an additional “feet” that steps on the apple. MotionNFT avoids these structural collapses, successfully executing the editing instructions with high anatomical fidelity.

2. Motion Misalignment: Even strong closed-source commercial models suffer from correctly identifying the subject part, direction, and magnitude of the motion change. For instance, in the anime girl example (row 1), Nano-banana demonstrates editing inertia where the motion of the subject remains the same, while GPT-Image 1, Seedream, and Hunyuan Image fail to execute the correct edit on the girl’s arm.

###### 9.2.3. Failure Analysis and Limitations

While MotionNFT demonstrates robust performance across a wide range of editing cases, we observe specific scenarios where it, alongside leading commercial models, encounters difficulties. Figure 15 illustrates these common failure modes that highlights persisting challenges:

- • Multi-Subject Interactions: Challenging editing settings with multiple involving and non-involving subjects in images pose a major challenge for existing models. For instance, in the orca example (row 1), all models fail to position the orca in front of the polar bear while executing the motion change to make it submerge in water. Similarly, in the crew member example (row 3), changing only the direction and the motion of one subject among a couple is difficult for existing models.
- • Identity Preservation Existing models still struggle to preserve subjects and their identities in complex scenes. For instance, in the chicken example (row 2), 3 models fail to preserve the chicken’s appearance. In the tent example (last row), models fail to preserve additional subjects in the scene not involved in the motion change.

These cases suggest that future work incorporating stronger physics-based priors or motion guidance could further resolve the remaining challenges.

##### 9.3. Speed and Inference Cost

MotionNFT is designed to be lightweight and computationally efficient. A key advantage of our method is that it can be seamlessly integrated with base models such as FLUX.1 Kontext Dev and Qwen-Image-Edit with no additional inference-time cost. All experiments were conducted on a single NVIDIA GPU. Using 28 sampling steps for a single entry, inference requires approximately 48 seconds with the FLUX.1 backbone and 98 seconds with QwenImage-Edit. This confirms that MotionNFT enhances generation capabilities without compromising the speed or hardware requirements of the original models.

|[Figure 145]<br><br>[Figure 146]<br><br>[Figure 147]<br><br>[Figure 148]<br><br>[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>+MotionNFT (Ours)|
|---|

Input Image Qwen-Image-Edit Ground Truth

###### + UniWorld-V2

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Change the person's pose: lift the torso to an upright position, extend the right arm straight out to the side, and extend the left arm forward. Keep the current outfit and facial expression.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Change the woman's right hand to operate the joystick on the control panel.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Move the white mug from the woman's hands to the wooden table, positioned to her right.

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Replace the man's clasped hands with his right hand gently touching the woman's left hand, comforting her.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

Replace the milk jar's drawn smile with an open mouth shape, holding a piece of the chocolate chip cookie taken from the cookie on the right. The right chocolate chip cookie should visibly have a bite missing.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Show the man actively manipulating the puppet, with one hand touching its leg, giving the puppet slightly bent knees, and the man a focused expression.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Move the green caterpillar from the pink petal to the yellow center of the flower.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

Change the polar bear's pose to lower its head and sniff the snow.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Change the girl's right arm pose from an extended gesture to a hand-on-hip pose, keeping her winking expression, blue hair, and outfit unchanged.

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Move the watermelon from on the woman's head to in front of her lower body.

- Figure 11. Qualitative comparison of our method to Qwen-Image-Edit [34] and the MLLM-only reward training in Lin et al. [15]. The base model frequently fails to demonstrate correct motion awareness for the edit (e.g. fail to move the subject’s arms in the first row, and failing to displace the watermelon in the last row). While the MLLM-only approach improves semantic adherence, it often lacks geometric precision (e.g., caterpillar’s orientation in row 7). MotionNFT leverages optical flow to bridge this gap, achieving precise motion alignment and high fidelity to the editing instructions.

|[Figure 195]<br><br>[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>+MotionNFT (Ours)|
|---|

###### Input Image FLUX.1 Kontext + UniWorld-V2 Ground Truth

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Turn the character to face away from the camera.

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Make the penguin's pose passive and still, with its wings relaxed and mouth closed.

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Make the boy on the left stop cleaning the wheel and look towards the girl. Make the girl stop working on the bicycle and look towards the boy on the left. The boy on the right's action remains unchanged.

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

Replace the robot's firing projectile weapon with an activated energy sword held in its hands.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Change the person's pose: lift the torso to an upright position, extend the right arm straight out to the side, and extend the left arm forward. Keep the current outfit and facial expression.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Remove the handshake between the two characters.

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Have the female professional help the elderly woman stand up by holding her hand. Remove the man on the right.

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Change the black cat's pose from lying on its back on the blue pillow to standing on all fours on the pillow.

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

Change the woman's focused expression to a relaxed, subtle smile, and lower her right hand from the chess piece onto the table.

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Change the girl's right arm pose from an extended gesture to a hand-on-hip pose, keeping her winking expression, blue hair, and outfit unchanged.

- Figure 12. Qualitative comparison of our method to FLUX.1 Kontext [Dev] [14] and the MLLM-only reward training in Lin et al. [15]. The base model often exhibits editing inertia, failing to execute structural changes such as removing the handshake (row 6) or changing the subjects’ directions (row 3). MLLM-only baseline also frequently hallucinates incorrect poses (e.g., the distorted limb placement in row 5) or fails to preserve subject identity (row 2). MotionNFT is able to interpret and execute complex motion edit instructions while preserving subject appearance and visual setting.

|[Figure 245]<br><br>[Figure 246]<br><br>[Figure 247]<br><br>[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]<br><br>MotionNFT (Ours)|
|---|

Original UniWorld-V1 BAGEL

###### Ground Truth

###### FLUX.1 Kontext

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

“Reduce the large fireworks display above the building to a few falling embers.”

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

“Adjust the lion's head pose and facial expression to be calm, looking slightly downwards.”

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

“Change the gorilla's pose to an aggressive gesture, raising its left arm and forming a fist.”

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

“The man is drinking from the brown jug.”

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

“Make the robot raise its left arm to wave.”

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

“Make the orange car fall off the cliff, positioning it mid-air with the front angled downwards towards the water.”

- Figure 13. We compare MotionNFT against state-of-the-art baselines: UniWorld-V1 [15], BAGEL [7], and FLUX.1 Kontext [Dev] [14]. Red circles highlight failure regions. Baseline models exhibit different failure modes like editing inertia (e.g., failing to change the lion’s pose in row 2), or motion misalignment (e.g., raising the robot’s right arm instead of left arm in row 5). While baselines often struggle to execute challenging motion edits, MotionNFT achieves superior geometric grounding, accurately following semantic instructions and maintaining high motion fidelity to the ground truth.

|[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>MotionNFT (Ours)|
|---|

Original Nano-Banana GPT-Image-1

###### Ground Truth

###### Seedream Hunyuan Image

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

“Change the girl's right arm pose from an extended gesture to a hand-on-hip pose, keeping her winking expression, blue hair, and outfit unchanged.”

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

“Make the right foot step on the red apple, crushing it.”

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

“Reposition the convoy of white semi-trucks further to the right. Add several white semi-trucks parked at the factory loading docks in the background.”

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

“Change the man's pose from seated to standing, his expression from intense screaming to distressed, and his action from reacting to the laptop to disengaging from it, looking down.”

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

“Have the female dental professional turn away from the male patient and bend down to adjust the dental chair.”

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

“Make the orange car fall off the cliff, positioning it mid-air with the front angled downwards towards the water.”

- Figure 14. We conduct selective case studies of MotionNFT against leading closed-source commercial baselines: Nano-Banana [8], GPTImage-1 [22], Seedream [26], and Hunyuan Image [3]. Red circles highlight failure regions where baselines exhibit spatial inertia (e.g., failing to displace the car in the bottom row) or structural hallucination (e.g., generating an artifact “foot” in the second row). While commercial models generally maintain high visual quality, they frequently struggle to ground complex motion changes or maintain visual consistency. MotionNFT accurately follows these dynamic instructions, ensuring geometric alignment with the ground truth.

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

[Figure 333]

[Figure 334]

[Figure 339]

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

Original Nano-Banana GPT-Image-1 MotionNFT (Ours)

“Submerge the orca into the water, with its dorsal fin and upper body visible, and position it to obscure the polar bear.”

Ground Truth

“Turn the standing crew member to face the holographic display and the seated crew member on the left. Make them point their right hand at the holographic display.”

“Add an adult in a yellow shirt. Both adults collaboratively lift and shape the large white sheet into a tent structure.”

“Change the woman's pose and action: have her actively scooping seeds from the girl's pumpkin with a spoon, looking down into the pumpkin with an engaged expression.”

Seedream Hunyuan Image

“Shift the giant chicken to the left and slightly upwards, revealing more of its body, and turn its head to gaze horizontally instead of downwards.“

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

- Figure 15. Additional failure cases of our model and closed-source commercial models. We observe that instructions involving multiple involving and non-involving subjects (e.g. the orca example in row 1, which requires complex 3D spatial edit) remain challenging for all evaluated methods. Current models, including ours and commercial baselines, struggle to correctly generate accurate and targeted motions on the correct subject part with the correct direction and magnitude in challenging scenarios.

