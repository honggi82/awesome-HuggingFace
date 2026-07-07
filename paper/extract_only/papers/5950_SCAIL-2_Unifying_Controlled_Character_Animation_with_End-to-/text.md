## SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning

### Wenhao Yan1*, Fengjia Guo1*, Zhuoyi Yang1†, Jie Tang1‡

1Tsinghua University 2Z.ai

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

|[Figure 8]|
|---|

Human2Any

|[Figure 9]|
|---|

[Figure 10]

[Figure 11]

[Figure 12]

# arXiv:2606.10804v2[cs.CV]10Jun2026

[Figure 13]

[Figure 14]

Replacement

Animation

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Any2Any

|[Figure 23]|
|---|

|[Figure 24]|
|---|

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Replacement

Animation

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Complex Interactions

Egocentric Actions

|[Figure 35]|
|---|

|[Figure 36]|
|---|

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Figure 1: SCAIL-2 unifies various character animation tasks within an end-to-end paradigm.

##### Abstract

as well as model weights will be released at our project page: https://teal024.github.io/SCAIL-2/.

Controlled character animation requires transferring motion from a driving sequence to a reference character. Prior works heavily rely on intermediate representations, including pose skeletons to represent motion or masked background to represent environment, which inevitably leads to information loss. To address this, we present SCAIL-2, a framework that bypasses those intermediates and achieves end-to-end character animation. By directly concatenating driving videos to the sequence, the model can obtain all the required visual information from the input video. To address the lack of endto-end data, we unify sub-tasks of character animation with decoupled conditions and then curate a pipeline to synthesize MotionPair-60K, an end-to-end motion transfer dataset containing heterogeneous tasks of character animation. To achieve the unification, we utilize in-context mask conditioning and mode-specific RoPE as soft guidance beyond textual instructions and raw visual information. To address synthetic discrepancy in detailed regions, we propose Bias-Aware DPO to construct preference items to mitigate the errors. Extensive experiments demonstrate that our method substantially outperforms existing state-of-the-art approaches in various character animation tasks. A large subset of synthetic data

### 1 Introduction

Controlled character animation (Hu 2024; Cheng et al. 2025; Yan et al. 2025) has tremendous potential for film production and entertainment use since the development of video diffusion models (VDMs) (Blattmann et al. 2023; Yang et al. 2025; Wan et al. 2025). Existing arts primarily rely on intermediate motion representations as conditions for VDMs to transfer the movements. For the motion representation, current works typically use off-the-shelf pose estimators to draw skeleton maps (Hu 2024; Cheng et al. 2025) or apply self-supervised bottleneck designs (Song et al. 2025; Fang et al. 2026) to obtain motion embeddings. Despite current progress, skeleton maps suffer from inherent ambiguity under complex scenarios, while a bottleneck-design encoder loses spatial information essential for multi-character interactions. Recent works (Tan et al. 2025; Yan et al. 2025; Shi et al. 2025) explore universal character animation to drive any characters, but still rely on exocentric human skeletons and thus cannot handle driving sources like animals.

*Contributed equally. Work done during internship at Z.ai. †Tech lead. ‡Corresponding author: jietang@tsinghua.edu.cn

Sub-tasks for character animation face the same issue. Character replacement, typically defined as animation with

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Empirically, our model learns end-to-end driving capability under diverse scenarios including complex interactions and non-human inputs where pose estimators struggle or completely fail. The model shares such capability across tasks, showing superior generalization especially in crossidentity motion following and environment integration. Our main contributions can be summarized as follows:

[Figure 48]

[Figure 49]

User Inputs

Ref Image

Ref Character

Character Replacement Animation

Driving Motion Driving Motion

Character Image Animation

[Figure 50]

[Figure 51]

[Figure 52]

###### Character Masking

Ambiguous Skeleton

Intermediates

Unified End-to-end Animation Model

Unified End-to-end Animation Model

[Figure 53]

[Figure 54]

Pose-Driven Model

Pose-Driven Model

[Figure 55]

[Figure 56]

Outputs

- 1. We propose an end-to-end conditioning paradigm to unify different tasks in character animation.
- 2. We introduce a motion pair synthetic pipeline to synthesize MotionPair-60K, a heterogeneous dataset to support the end-to-end driven paradigm.
- 3. We apply a novel DPO-based post-training mechanism to refine detailed end-to-end motion capture.
- 4. We release SCAIL-2, an open-source end-to-end animation model. Extensive experiments demonstrate that SCAIL-2 outperforms current SoTA methods in various animation tasks, unlocking emerging applications.

[Figure 57]

✅

[Figure 58]

✅

[Figure 59]

❌ ❌

[Figure 60]

Figure 2: SCAIL-2 adopts end-to-end driving paradigm to bypass unreliable animation intermediates.

environment affordance (Hu et al. 2025), is often framed as a pose-driven inpainting task (Hu et al. 2025; Cheng et al. 2025), using cropped background or objects as intermediates. Such intermediates extracted from ground truth videos are by design suboptimal, as changing the character may also affect the interacted objects and background. Furthermore, the character mask limits the body shape and hinders cross-body-shape replacement. Another important sub-task is multi-character animation, where pose-driven approaches suffer from misinterpretation of interaction when depth-ambiguous skeletons overlap. Existing methods for the task (Chen et al. 2025a; Hu et al. 2026) also apply masking to alleviate the issues, but sacrifice the shape adaptability needed for universal characters as well.

### 2 Related Works

Character Image Animation. Character image animation refers to animating the character within its original background. Following Wan-Animate (Cheng et al. 2025), hereafter we denote Animation Mode to be this specific task. Existing methods for pose-guided character animation (Hu 2024; Hu et al. 2025; Zhu et al. 2024; Cheng et al. 2025; Li et al. 2026) typically begin by extracting skeletal motion sequences from the driving video as a form of “motion capture”, and then inject this information into a video generation model to perform “rigging” and “rendering”. SCAIL (Yan et al. 2025) introduces an identity-agnostic 3D skeleton representation rendered with different hues to enable multi-character animation, but still suffers from limited information in the pose especially under interactions. The most relevant work adopting end-to-end animation is closedsource DreamActor-M2 (Luo et al. 2026), which aligns endto-end capability on a pose-driven model. Our work focuses on unifying a wider range of sub-tasks including complex interactions by directly training with heterogeneous synthetic data.

End-to-end character animation directly provides the driving context instead of passing intermediates, therefore faithfully preserving visual information including occlusions and environments. However, such a paradigm relies on paired data: one video requires a pairing sequence where totally different character(s) perform the same set of movements in the same environment or different environments. To address the lack of such data, we adopt pose-driven models to generate synthetic videos of the same motion, and leverage an agentic synthetic loop to curate diverse high-quality animation pairs, and then reversely use the generated data as driving videos.

Under this paradigm, we decouple the sub-tasks into endto-end animation with task-specific conditions. To model their distinctions and enhance the inputs, we introduce incontext mask conditioning and mode-specific context RoPE as a unified interface under the reverse driving training paradigm. The in-context mask contains an environment switch that works together with mode-specific context RoPE to support task unification, and further incorporates character binding slots that describe motion–character binding. This unification not only supports more diverse forms of user input, but also allows different sub-tasks to be composed within the reverse interface and surpass the performance of original generators. As a result, the model can address compositional tasks for which constructing dedicated data is difficult. A further challenge in end-to-end training is the bias of synthetic data, which we find most pronounced in detailed finger movements. We therefore design Bias-Aware DPO, a post-training scheme for better end-to-end motion capture in the hand regions.

Character Replacement. Character replacement still animates the reference character, but using the driving background. Following Wan-Animate, hereafter Replacement Mode is for this task. Previous methods (Hu et al. 2025; Cheng et al. 2025) achieve this by background-inpainting pose-driven animation. A recent advance, MoCha (Xu et al. 2026), trains an end-to-end character replacement model based on data rendered by Unreal Engine 5. Still, it struggles in generalizing to characters with large gaps in body shape or complex object interactions due to renderer limitation. In this work, we overcome the generalization limitations of this task by end-to-end unification.

### 3 Methods

#### 3.1 Preliminary

General Task Formulation. Given an input video x, a latent video diffusion model (Wan et al. 2025) first encodes it

###### Exclude C & Reselect

[Figure 61]

[Figure 62]

VLM: Given	 irst	frame F of𝐲as	posture reference,	yield	best-matching C ∈	𝒫char

Anima on Mo on: Single

[Figure 63]

###### Mo on Pool 𝒫mo on

VLM:	Given F and C,	envision generated	character,	posture, background,	yield	description D

[Figure 64]

Posture Bestmatching

|[Figure 65]|
|---|

[Figure 66]

Candidate Selector

[Figure 67]

Image Animation

Prompt Weaver

[Figure 68]

…

[Figure 69]

An anime girl with twintail stands in

[Figure 70]

[Figure 71]

[Figure 72]

a low, wide stance in front of a temple, with left hand raised overhead and right hand clenched at the chest.

[Figure 73]

[Figure 74]

[Figure 75]

No Matched Return

[Figure 76]

[Figure 77]

[Figure 78]

Sample 𝐲~𝒫mo on

[Figure 79]

Nano Banana Pro: Given	posture	ref F,	char	ref C, desc D, form	edit	instruction D’ and	yield	edited	frame F’

[Figure 80]

[Figure 81]

Pose Editor

[Figure 82]

[Figure 83]

Anima on Character:

ℳ

[Figure 84]

[Figure 85]

###### Character Pool 𝒫char

Edit Image A to follow the exact pose of Image B. The final image features an anime girl …

Replacement Mo on: Single/Mul 

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

Quality Checker VLM: Given F, C, Fʹ ,	judge	pose	accuracy,	environment	and	overall quality

[Figure 94]

[Figure 95]

[Figure 96]

Leaked / Implausible Env

Edit Env

[Figure 97]

[Figure 98]

[Figure 99]

|[Figure 100]|
|---|

[Figure 101]

[Figure 102]

Inaccurate Pose: Pass Totally Fail F’ -> F, Re-edit

Sample {Ci}⊂ 𝒫char

[Figure 103]

[Figure 104]

[Figure 105]

Synthetic Animation Generator 𝒢 𝒢(𝐲, Fʹ) → 𝐲̃

Synthetic Replacement Generator 𝒢 𝒢(𝐲, Fʹ) → 𝐲̃

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

###### Replacement Character: White-BG Standing Human (Pre-edited) Replace One

Or One-by-One (𝒢(𝐲̃, Fʹ) -> 𝐲̃)

Yield training Pair (𝐲̃, 𝐲) a er video check

Figure 3: The overview of our synthetic pipeline for curating diverse high-quality cross-identity motion pairs.

into a latent representation z0 = E(x) via a pretrained VAE encoder E. A forward diffusion process then progressively

- O2 Environment Weaving — read the prescribed environ-

ment source E from the context, and integrate the characters into the scene from either the reference (EI) or the driving video (Ey) to generate a coherent composition;

- O3 Universal Transfer — disentangle pose from identity

corrupts z0 by adding Gaussian noise over T timesteps:

q(zt|zt−1) = N zt; 1 − βt zt−1, βtI , (1)

where βt denotes the noise schedule. A denoising model εθ(zt,t,c) is trained to recover the added noise conditioned on auxiliary input c (e.g., pose, text), with the objective:

so that motion extracted from any driving character transfers to any target in a physically plausible manner without identity leakage.

t,ε∼N(0,I) ∥ε − εθ(zt,t,c)∥22 . (2)

L = Ez

#### 3.2 End-to-end Data Synthesis

For character animation, the condition c comprises a text prompt ctext, a reference image I containing a character set CI = {C1,...,CN} within an environment EI, and a motion signal derived from a driving video y containing characters Cy = {C1driv,...,CMdriv} within environment Ey. Pose-driven methods first extract an explicit pose sequence cpose = P(y) via an off-the-shelf estimator P, then encode it into latent space as the motion condition. In an end-to-end solution, the driving video is directly encoded, i.e., zdriv = E(y), bypassing explicit pose estimation while still operating in the shared VAE latent space.

Animation Synthetic Loop. To achieve end-to-end character image animation, we need a synthetic engine to produce a synthetic video y˜ from a ground-truth driving video y and a provided character reference image I, through an animation generator G:

y˜ = G(y,I). (3)

Given a fixed G and a sampled driving sequence y, our pipeline synthesizes optimal I for animation generators. To reduce synthetic cost, improve character diversity and reduce the generation of unreasonable data, we propose an agentic editing loop to generate plausible reference images directly from random human-centric datasets. The generation loop combines a Candidate Selector, a Prompt Weaver, a Quality Checker and a strong multi-reference image generation model M (Google DeepMind 2025). Each iteration we directly sample one driving video and several character images. The Selector chooses the best character candidate, then provides M with the first frame of the video as the posture reference besides the character image. The Prompt Weaver is designed to pre-plan the desired character, background, and posture, bypassing context hallucination of M’s innate planner and generator. We apply multiple turns of editing under the supervision of the Quality Checker to obtain better results. Additional editing is optionally applied to environment elements to prevent potential leakage and improve human-object-interaction (HOI) generalization.

Sub-Tasks Formulation. We unify the sub-tasks of character image animation by a binding map π : Cy → CI and an environment source E ∈ {EI,Ey}.

Character Image Animation: Single and Multi correspond to |Cy| = |CI| = 1 and >1 respectively, both with E = EI, where each driving character Cidriv transfers its motion to π(Cidriv).

Character Replacement: it shares the same binding formulation for both single- and multi-character scenarios, but takes E = Ey.

We cast all sub-tasks as a single problem of reading different dimensions of information from the context and composing them into a plausible final result, and decompose the optimization into three objectives that the model should learn accordingly:

O1 Motion Binding — extract motions from the driving video while identifying their respective character origins, and route them solely to their bound targets π(Cidriv);

For the choice of G, we adopt pose-driven models (Yan et al. 2025; Cheng et al. 2025) as the animation generator.

SCAIL generates the majority of data in pretraining as it is robust towards large body shape gaps and complex motions. With the combination of our synthetic pipeline and the generator choice, we control the discard rate of generated videos to be less than 30% when applying VLM (Gemini Team, Google 2023) to check the synthetic data.

Replacement Data. We adopt a renderer-trained singlecharacter replacement model (Xu et al. 2026) as the replacement generator. Replacement is meaningful not only because it supports the task itself, but also because it can supplement data for animation. Multi-character animation pairs are hard to collect even with the designed loop, as the task complexity is significantly higher even for models optimized for it (Yan et al. 2025; Hu et al. 2026). We instead substitute multi-character animation with multi-character replacement, as replacement is more tractable and can be performed in a character-by-character manner. From the perspective of optimization objectives, the two sub-tasks only differ in O2 (EI vs. Ey) according to our formulation. Notably, the difference is already covered by single-character animation and replacement pairs, while the challenge of the multi-character setting, namely learning the binding π for O1 and extracting and transferring motion under heavy inter-character occlusions for O3, is equally exercised by replacement.

Data Composition. The full pipeline yields MotionPair60K, an end-to-end motion-transfer dataset with animation mode data and replacement mode data in a ratio around 3:1. In training, we also randomly sample from pose-driven datasets in SCAIL’s pose format for data diversity. We apply random augmentations for the driving video in animation mode: for the end-to-end driving sequences we use cropping and stretching, while for the pose we apply random skeleton scaling. Detailed data composition and sampling ratios in training are shown in Appendix A.

Reverse Driving. We use data in a reverse manner: designated characters in a real video y are re-synthesized from I via pose transfer or one-by-one replacement, yielding a synthetic video y˜. The synthetic y˜ then serves as the driving input while the original real video y serves as the denoising target x, alongside a reference frame I sampled from y, to form a training triplet without introducing artifacts or renderer-bias by G. For instance, a renderer-trained generator such as MoCha often suffers from inaccurate character texture and physically implausible object interactions; under the reverse scheme the driving input is merely to convey motion, while the supervised target preserves faithful, physically consistent interactions.

#### 3.3 Model Design

Model Architecture. Our model adopts the In-Context Driving design (Yan et al. 2025), where the condition is concatenated to the denoised sequence rather than injected into the denoising embedding via channel concat (Cheng

- et al. 2025) or pose-guider (Wang et al. 2025). Concretely, the I2V backbone (Wan et al. 2025) receives the input of [zref; zt; zdriv], the concatenation of the reference, noisy video, and driving tokens. Following SCAIL, zdriv carries a fixed spatial offset ∆W along the w axis so that it stays spatially detached from the video tokens.

Table 1: 3D RoPE coordinates assigned to zref, zt and zdriv. The video latent has shape Tv×Hv×Wv.

t h w Animation Mode zref 0 [0, Hv) [0, Wv)

zt [1, Tv] [0, Hv) [0, Wv) zdriv [1, Tv] [0, Hv) [∆W, ∆W+Wv) Replacement Mode zref 0 [∆refH, ∆refH+Hv) [0, Wv)

zt [0, Tv−1] [0, Hv) [0, Wv) zdriv [0, Tv−1] [0, Hv) [∆W, ∆W+Wv)

In-Context Mask Conditioning. We propose In-Context Mask Conditioning to simultaneously model the difference between sub-tasks and enhance the original raw visual inputs for certain tasks, as shown in Fig.4. To distinguish character image animation with replacement and optimize objective O2, we add 1 additional channel as environment switch, indicating whether the environment should be derived from the reference image or the video. To optimize O1, we further introduce K channels as the binding slots. The binding slots explicitly describe that the motion should flow exclusively within characters sharing the same channel. Single-character animation activates one random slot, while multi-character activates several, and allowing two far-away characters to share one channel can support more than K characters.

In training, all valid mask signals are derived from reference images and driving sequences, without injecting signals from ground truth (the denoising latents keep an allzero mask), which constitutes the fundamental distinction between our approach and prior works. The extraction is performed by a robust segment model SAM3 (Carion et al. 2025) with rule-based matching. To align with the latent grid, the mask is downsampled spatially and stacked temporally along the channel dimension, producing 4(K+1) channels concatenated to the context. The introduction of such signals serves to provide enhanced guidance on top of the visual context to avoid confusion, rather than to alter it; the end-to-end nature is therefore preserved, as the model still observes the complete visual information.

Mode-Specific Shifted RoPE. To better model the difference between Animation Mode and Replacement Mode, we adopt Mode-Specific Shifted RoPE. We notice that animation mode will regenerate a new starting frame from the visual elements in the reference while replacement mode needs identical background from the first driving frame and only regenerate the character for the first frame. To model such difference, we design Animation Mode’s denoising latent and reference with a temporal difference (T = 0 and T = 1), then set Replacement Mode with a spatial difference, where an extra spatial RoPE shift ∆refH is applied for zref, as shown in Tab.1. Empirically, the collaboration of in-context mask conditioning and RoPE differentiation prevents training conflicts, allowing the two tasks to share the optimization of universal target O3 and convey the trained universal capability to compositional tasks.

- MR - Ch0
- MR - Ch1

[Figure 113]

[Figure 114]

- MR - Ch2

- MD - Ch0
- MD - Ch1
- MD - Ch2

Anima on Mode

Reference Image I Synthetic Driving Video 𝐲̃ Real Video 𝐲

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Reference Image

Driving Video 𝐲̃

Real Video 𝐲

I

[Figure 119]

| | |
|---|---|
| | |

[Figure 120]

[Figure 121]

[Figure 122]

3D VAE Encoder

[Figure 123]

…

[Figure 124]

[Figure 125]

###### Add Noise

… … …

- MR - Ch0
- MR - Ch1

[Figure 126]

[Figure 127]

- MR - Ch2

- MD - Ch0
- MD - Ch1
- MD - Ch2

Replacement Mode

Driving Mask MD All-Zero Mask Mzero

Reference Mask MR

[Figure 128]

…

…

- Ch0
- Ch1

Reference Image

Driving Video 𝐲̃

Real Video 𝐲

…

… … …

I

[Figure 129]

Chk … … …

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

…

Video Patchify

[Figure 136]

[Figure 137]

|Sequence Concat| |
|---|---|
| |…|

Text Input

… 3DVAE

Flow Matching Loss v-Prediction of 𝐲

[Figure 138]

… Dit Transformer Block

[Figure 139]

…

…

×𝑁

Decoder

- Figure 4: Overview of our model architecture and the context mask signal. Ch0 means the mask channel for environment control, while Ch1 to ChK denote K channels for character binding. The environment mask is the complement of the union of either the driving or the reference character masks.

#### 3.4 Post Training

P′′, where we can select less accurate estimators as P′ or P′′:

Bias-Aware DPO. Although end-to-end modeling enables the model to handle scenarios where pose extraction fails, the errors introduced by pose-driven synthetic data mean that the motion accuracy of end-to-end training data will be affected by pose estimation and animation generators. We notice that subtle movements in the hand region provide the most obvious evidence, where finger joints are often incorrectly articulated or simply neglected. To mitigate this bias, we propose Bias-Aware DPO. Specifically, we bootstrap synthetic data to directly simulate the errors introduced by pose estimators, and frame these errors as negative preferences for the model, thereby optimizing error correction during training.

r− = G P′′(G(P′(y),R)),R , (6) To construct a preference data item, we randomly sample one frame from r as the reference image R1 and use s as the driving video, forming a preference tuple:

s,R1,r,r− , (7)

where (s,R1) serves as the conditioning input, r is the preferred sample, and r− the less preferred sample. With the preference tuple we can adopt DPO-based methods (Wallace et al. 2024) to optimize the preference. Unlike the main training stage which is driven in a reverse manner, the preference optimization here takes the synthesized r directly as the preferred target. Further implementation details of post training are provided in Appendix B.

Preference Dataset Construction. Our target is to construct positive–negative sample pairs that share the same reference identity (i.e. the same character) and follow a consistent overall pose, but where the negative sample is always slightly less accurate than the positive one in fine-grained details including finger articulation. Given a motion video y, a pose estimator P, and an animation generator G, we extract a pose p = P(y) and synthesize two videos under different reference images R and S:

### 4 Experiments

#### 4.1 Implementation Details

We train the model on a 14B I2V Backbone Wan2.1-14BI2V: during the pretraining stage, we full-finetune the backbone for 3,500 steps with a batch size of 128 at a learning rate of 1e-5; after convergence, we perform DPO Post Training for another 400 steps. For the in-context conditioning, K is set at 6, so 28 additional channels are stacked to the model. For long video generation, we follow Wan-Animate to randomly replace the first 2 latents to be conditional history latents. Overall training of the 14B model is conducted on 64 NVIDIA H100 GPUs for around a week using FSDP2 (Zhao et al. 2023).

r = G(p,R), s = G(p,S). (4)

Sharing the same pose sequence, r and s form a basic pair: we take s as the driving video and r as the positive sample. The negative sample is then obtained by passing r through one more round of error propagation along the same pipeline—re-extracting the pose from the synthesized video and regenerating under the same reference image:

r− = G P(r),R . (5)

#### 4.2 Evaluation Metrics

where r− inherits one extra round of error and is therefore less accurate than r in details. The gap can be further widened by performing the two extraction steps with P′ and

We evaluate the animation performance of our methods using Studio-Bench (Yan et al. 2025) and X-Dance Benchmark (Zhang et al. 2026) and establish a new benchmark

###### Motion Consistency

Win Rate (%)

68.3% 28.3%

vs SCAIL

65.0% 6.7% 28.3%

vs Wan-Animate

36.7% 23.3% 40.0%

vs Kling 3.0

0 20 40 60 80 100

###### Physical Plausibility

Win Rate (%)

68.3% 6.7% 25.0%

vs SCAIL

78.3% 16.7%

vs Wan-Animate

46.7% 16.7% 36.7%

vs Kling 3.0

0 20 40 60 80 100

###### Identity Consistency

Win Rate (%)

46.7% 18.3% 35.0%

vs SCAIL

68.3% 11.7% 20.0%

vs Wan-Animate

53.3% 8.3% 38.3%

vs Kling 3.0

0 20 40 60 80 100

SCAIL-2 (Ours) Win

Tie

Competitor Win

| |
|---|

| |
|---|

| |
|---|

- Figure 5: Single-character human evaluation on StudioBench. Kling 3.0 denotes Kling 3.0 Motion Control (Team

- et al. 2026).

###### Motion Consistency

Win Rate (%)

57.1% 28.6% 14.3%

vs Mocha

71.4% 25.0%

vs Wan-Animate

0 20 40 60 80 100

###### Environment Integration

Win Rate (%)

67.9% 25.0%

vs Mocha

67.9% 28.6%

vs Wan-Animate

0 20 40 60 80 100

###### Identity Consistency

Win Rate (%)

66.7% 26.7%

vs Mocha

73.3% 23.3%

vs Wan-Animate

0 20 40 60 80 100

SCAIL-2 (Ours) Win

Tie

Competitor Win

| |
|---|

| |
|---|

| |
|---|

Figure 7: Human evaluation on Studio-Bench for character replacement.

###### Motion Consistency

Win Rate (%)

50.0% 30.0% 20.0%

vs SCAIL

76.7% 13.3% 10.0%

vs Wan-Animate

93.3% 6.7%

vs MultiAnimate

0 20 40 60 80 100

###### Identity Isolation

Win Rate (%)

56.7% 36.7%

vs SCAIL

90.0% 10.0%

vs Wan-Animate

86.7% 10.0%

vs MultiAnimate

0 20 40 60 80 100

###### Identity Consistency

Win Rate (%)

26.7% 60.0% 13.3%

vs SCAIL

60.0% 16.7% 23.3%

vs Wan-Animate

93.3% 6.7%

vs MultiAnimate

0 20 40 60 80 100

SCAIL-2 (Ours) Win

Tie

Competitor Win

| |
|---|

| |
|---|

| |
|---|

Figure 6: Multi-character human evaluation on StudioBench.

Method SSIM↑ PSNR↑ LPIPS↓ FVD↓ Ours

+ SAM3D-Body Mesh 0.6453 19.09 0.2231 287.11 + NLF-Pose Skeleton 0.6370 18.76 0.2285 282.85

SCAIL

+ SAM3D-Body Skeleton 0.6407 19.08 0.2212 309.63 + NLF-Pose Skeleton 0.6378 19.08 0.2212 312.79

Wan-Animate 0.6340 18.62 0.2269 305.31 SteadyDancer 0.6386 18.40 0.2311 332.20 Onetoall-Animation 0.6138 17.25 0.2667 448.06 UniAnimate-DiT 0.6367 18.52 0.2747 480.15 VACE 0.5942 17.09 0.2883 387.52

Table 2: Single-character animation metrics on the singlecharacter split of Studio-Bench’s pose-driven partition.

for the replacement mode following the standards of StudioBench, which emphasizes real-world cross-identity scenarios to test the performance of models under complex motion and interactions when appearance or body shape changes. As the end-to-end paradigm is for cross-identity inference settings, we adopt GSB(Good/Same/Bad) subjective evaluations for such scenarios as prior work (Cheng et al. 2025). As our model also supports pose-driven, low level metrics like SSIM (Wang et al. 2004), PSNR (Hore and Ziou 2010), LPIPS (Zhang et al. 2018), FVD (Unterthiner et al. 2018) can be calculated under this setting where the character’s own pose serves as the condition. We also employ VideoBench (Han et al. 2025) as the automatic evaluator following prior works (Luo et al. 2026) to judge the overall video quality.

#### 4.3 Quantitative Evaluation

Cross-Identity Results. Results from Fig. 5 show that for single-character animation our model wins over leading open-source works in all human-evaluation metrics and re-

Video-Bench Evaluations Imaging Quality ↑

Method

Motion Smoothness ↑

Temporal Consistency ↑

Appearance Consistency ↑

Wan-Animate 3.80 3.89 4.03 4.23 Onetoall-Animation 3.98 3.72 3.99 4.05 SteadyDancer 4.41 3.97 4.08 4.17 SCAIL 4.27 3.90 4.21 4.25

Ours 4.43 3.89 4.18 4.38

Table 3: Automatic evaluation of video quality on X-dance using Video-Bench.

mains close to proprietary services like Kling 3.0. For multicharacter animation our model also beats previous methods with clear advantages, especially in terms of Identity Isolation, as shown in Fig. 6. Notably, the multi-character animation results are zero-shot, and this advantage validates the soundness of our data construction and training strategy.

###### Animation Mode Replacement Mode

Reference Frame

SCAIL Mocha Ours

Wan-Animate Kling 3.0 Ours Wan-Animate

Reference Character

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Driving Sequence

Driving Sequence

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

- (a)
- (b)

(d)

(f)

- (c)

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Reference Character

Wan-Animate Ours

Reference Frame

SCAIL

Wan-Animate Mocha Ours

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Driving Sequence

[Figure 168]

Driving Sequence

(e)

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

| |
|---|

|[Figure 173]|
|---|

[Figure 174]

[Figure 175]

Reference Frame

SCAIL MultiAnimate Ours

[Figure 176]

Reference Character

Wan-Animate Mocha Ours

[Figure 177]

[Figure 178]

[Figure 179]

Driving Sequence

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

Driving Sequence

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

Figure 8: Qualitative comparison against baselines under cross-identity inputs.

Pose-driven Results. Even though the model is trained with rather limited amount of pose pairs, results in pose-driven video metrics (Tab. 2) have interesting outcomes. Under the challenging Studio-Bench with complex rotations and intense movements, using our model as a skeleton-driven generator yields mediocre metrics in SSIM/PSNR. However, adopting human mesh from SAM3D-Body (Yang et al. 2026) clearly improves those metrics, even when it’s completely zero-shot (the model has never seen such representation in training). We attribute the performance gain to the richer information that a precise mesh provides, and believe it demonstrates the advantage of end-to-end animation in extracting more information from the driving sequence. In replacement mode (Fig. 7) our model beats inpainting-based Wan-Animate and surpasses MoCha, our generator to create replacement pairs, proving the effectiveness of unification under the reverse driving paradigm.

the inherent advantages of end-to-end modeling approach. Fig. 8(c) indicates our model guided by explicit in-context character binding signals isolates identities while keeping characters’ original body shape.

For the replacement part, Fig. 8(d) showcases that in replacement mode our model still preserves the combination of precise motion and character generalization. Furthermore, Fig. 8(e) presents a highly challenging scenario that requires maintaining both the character’s identity and the hand-instrument interaction amidst a crossing crowd. In this case, MoCha completely loses the instrument, while WanAnimate produces noticeable dark artifacts around the person due to its inpainting mechanism. Fig. 8(f) further exposes the inherent limitations of Wan-Animate’s inpainting approach; as evident from the shoe reflections, our model achieves the most natural integration into the surrounding environment.

#### 4.5 Ablation Studies

#### 4.4 Qualitative Evaluation

Ablation on Driving Modes. Fig. 9(a) shows the comparison of different driving modes of our model. Without visual information of how the two characters fight, the model is confused about their interactions. Together with comparisons against other pose-driven methods, this ablation further proves that the performance gain in complex interactions stems from the end-to-end paradigm itself.

Fig. 8 shows qualitative comparisons against SoTA baselines under cross-identity inputs. In the comparison presented in Fig. 8(a), our model yields accurate motions, exhibiting superior identity consistency and highly precise human-object interactions (e.g., handling the ball). Fig. 8(b) is a more complex case further demonstrating our framework’s advantage over pose-driven baselines, which are unable to faithfully synthesize intricate character movements involving object interactions without the visual information, highlighting

Ablation on Network Modules. Fig. 9(c) indicates both the environment switch and Mode-Specific RoPE are essential

(b) Replacement Mode

###### (a) Animation Mode

Driving Video Ours, inference w/o character mask

w/o Binding Slots Ours w/o Binding Slots Ours

Driving Video Reference Image Driving Video Ours w/ Pose Driven Ours

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

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

(d) Animation Mode

###### (e) Replacement Mode

(c) Animation Mode

Driving Video w/o Replacement Data Ours Driving Video w/o Animation Data Ours

Reference Image w/o Mode-Specific Ours RoPE

w/o Environment Switch

Driving Video

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

Figure 9: Ablation studies on network modules and data.

Driving Video Base Model w/ SFT w/ DPO

to unify Animation Mode and Replacement Mode. Without the environment switch, the model generates an arbitrary background, as it struggles to distinguish the two modes from textual cues alone; without Mode-Specific RoPE, the model changes shadowed areas in the reference image to textured white, suggesting that the model is disturbed by certain spurious patterns in the reference image without proper disambiguation. Fig. 9(b) demonstrates the effectiveness of Binding Slots. Inference without the character mask fails to maintain identity when pedestrians pass through; training without Binding Slots forces the model into innate tracking but affects the pedestrian’s outfit. In rotating scenarios, the slots also help steady identity assignment. This shows that an additional mask signal remains important even atop an end-to-end formulation: while end-to-end modeling maximally exploits the model’s prior, tasks inherently hard for the model—such as preserving identity after characters swap positions in an I2V backbone—require stronger conditioning, which Binding Slots provide. At the same time, this does not compromise the end-to-end nature, as the mask merely supplies more information rather than altering the visual context. Quantitative gains from Slots and data on multi-character scenarios are provided in Appendix D.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Driving Video Base Model w/ SFT w/ DPO

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Figure 10: Qualitative comparison of our Bias-Aware DPO against the SFT variant and the base model.

the base model on other fine details such as the mouths and shoulders, which we will further discuss in the Appendix B.

### 5 Limitations and Discussion

Ablation on Data Composition. Fig. 9(d) and (e) demonstrate the synergistic effect of character image animation data and replacement data. Without replacement data, the model fails to extract correct motion when characters overlap; without animation data, the model struggles to maintain motion consistency when body shape changes drastically, as it is beyond the replacement generator’s domain. This validates the effectiveness of unification: the replacement data equips the model with the ability to handle complex overlaps among multiple characters, while the animation data complements replacement by covering cross-body-shape cases.

Limitations. While end-to-end designs feed the model complete visual information that is naturally richer, the fundamental limitation lies in a strict dependence on largescale, high-quality paired training data. While our synthetic pipeline largely resolves the data-scarcity problem, the fidelity of the constructed data still hinges on the capability of these generators. We use Bias-Aware DPO to model the preference against bias, but reliable positive samples for finegrained regions remain hard to obtain. Future works could adopt more advanced models or more efficient pipeline to synthesize data of higher quality to extend the framework to more tasks, such as lip-syncing and detailed expressions in facial regions.

Ablation on Bias-Aware DPO. Fig. 10 shows the comparison among our Bias-Aware DPO, SFT, and the base model. While SFT can strengthen hand learning by adding an explicit hand loss, its hand optimization remains insufficient for lack of negative samples. In contrast, our Bias-Aware DPO explicitly models the error, enabling it to capture finer hand details. Moreover, although preference is modeled only on the hands region, in some cases the policy also refines

Discussion. Our framework demonstrates two kinds of gains from the unified end-to-end conditioning. The first comes from end-to-end training itself: by training a DiT with strong priors to extract and convert information from visual contexts, the model generalizes to a broader range of zeroshot inputs. The second comes from unification under the

reverse-driving end-to-end pipeline: through reverse driving and concept decoupling, the model extracts distinct types of information and acquires strong compositional ability; and because the real videos always serve as authentic supervision, the optimization is steered toward composing these abilities in a plausible way, thereby surpassing the data generators. Our framework is positioned to benefit from, rather than be obsoleted by, future advances in data synthesis methods and supervision strategies.

### 6 Conclusions

In this work, we present SCAIL-2, an end-to-end framework for character animation. We design an end-to-end data curation pipeline that synthesizes a dataset spanning diverse animation tasks, making end-to-end animation feasible at scale. Building on this dataset, we unify several subtasks under a single end-to-end driving paradigm and observe a clear synergistic effect between the curated data and the unified network design. To further enhance finegrained motion transfer, we introduce a novel Bias-Aware DPO post-training scheme. Extensive experiments demonstrate that SCAIL-2 achieves state-of-the-art performance, particularly in cross-identity motion following, environment integration, and multi-character interactions, while generalizing well across diverse tasks. We believe SCAIL-2 offers a practical and extensible paradigm towards production-ready character animation.

### References

Blattmann, A.; Dockhorn, T.; Kulal, S.; Mendelevitch, D.; Kilian, M.; Lorenz, D.; Levi, Y.; English, Z.; Voleti, V.; Letts, A.; et al. 2023. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127.

Carion, N.; Gustafson, L.; Hu, Y.-T.; Debnath, S.; Hu, R.; Suris, D.; Ryali, C.; Alwala, K. V.; Khedr, H.; Huang, A.; et al. 2025. Sam 3: Segment anything with concepts. arXiv preprint arXiv:2511.16719.

Chen, J.; Chen, M.; Xu, J.; Li, X.; Dong, J.; Sun, M.; Jiang, P.; Li, H.; Yang, Y.; Zhao, H.; et al. 2025a. Dancetogether! identity-preserving multi-person interactive video generation. arXiv preprint arXiv:2505.18078.

Chen, L.; Ma, T.; Liu, J.; Li, B.; Chen, Z.; Liu, L.; He, X.; Li, G.; He, Q.; and Wu, Z. 2025b. HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning. arXiv:2509.08519.

Cheng, G.; Gao, X.; Hu, L.; Hu, S.; Huang, M.; Ji, C.; Li, J.; Meng, D.; Qi, J.; Qiao, P.; et al. 2025. Wan-animate: Unified character animation and replacement with holistic replication. arXiv preprint arXiv:2509.14055.

Contributors, L. 2025. LightX2V: Light Video Generation Inference Framework. https://github.com/ModelTC/ lightx2v.

Fang, Z.; He, X.; Tang, S.; Zhang, H.; Li, Q.; Liu, X.; Wan, P.; and Gai, K. 2026. 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation. arXiv preprint arXiv:2602.03796.

Ferguson, A.; Osman, A. A. A.; Bescos, B.; Stoll, C.; Twigg, C.; Lassner, C.; Otte, D.; Vignola, E.; Prada, F.; Bogo, F.; Santesteban, I.; Romero, J.; Zarate, J.; Lee, J.; Park, J.; Yang, J.; Doublestein, J.; Venkateshan, K.; Kitani, K.; Kavan, L.; Farra, M. D.; Hu, M.; Cioffi, M.; Fabris, M.; Ranieri, M.; Modarres, M.; Kadlecek, P.; Khirodkar, R.; Abdrashitov, R.; Pr´evost, R.; Rajbhandari, R.; Mallet, R.; Pearsall, R.; Kao, S.; Kumar, S.; Parrish, S.; Yu, S.-I.; Saito, S.; Shiratori, T.; Wang, T.-L.; Tung, T.; Xu, Y.; Dong, Y.; Chen, Y.; Xu, Y.; Ye, Y.; and Jiang, Z. 2025. MHR: Momentum Human Rig. arXiv:2511.15586.

Gemini Team, Google. 2023. Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805.

Google DeepMind. 2025. Nano Banana Image Generation via Gemini API. https://ai.google.dev/gemini-api/docs/ image-generation. Accessed: 2026-05-20.

Han, H.; Li, S.; Chen, J.; Yuan, Y.; Wu, Y.; Leong, C. T.; Du, H.; Fu, J.; Li, Y.; Zhang, J.; Zhang, C.; jia Li, L.; and Ni, Y. 2025. Video-Bench: Human-Aligned Video Generation Benchmark. arXiv:2504.04907.

Hore, A.; and Ziou, D. 2010. Image quality metrics: PSNR vs. SSIM. In 2010 20th international conference on pattern recognition, 2366–2369. IEEE.

Hu, L. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8153–8163.

Hu, L.; Wang, G.; Shen, Z.; Gao, X.; Meng, D.; Zhuo, L.; Zhang, P.; Zhang, B.; and Bo, L. 2025. Animate anyone 2: High-fidelity character image animation with environment affordance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 10207–10217.

Hu, Y.; Gong, H.; Yang, C.; An, Z.; Xu, Y.; and Liu, S. 2026. MultiAnimate: Pose-Guided Image Animation Made Extensible. arXiv preprint arXiv:2602.21581.

Li, W.; Gao, Y.; Hassan, M.; Feng, L.; Pan, W.; Luan, P.C.; and Alahi, A. 2026. EverAnimate: Minute-Scale Human Animation via Latent Flow Restoration. arXiv:2605.15042. Liang, S.; He, J.; Wang, C.; Liao, L.; Zhang, G.; Chen, Y.; and Yuan, Y. 2026. SDPose: Exploiting Diffusion Priors for Out-of-Domain and Robust Pose Estimation. arXiv:2509.24980.

Luo, M.; Liang, S.; Rong, Z.; Luo, Y.; Hu, T.; Hou, R.; Chang, H.; Li, Y.; Zhang, Y.; and Gao, M. 2026. DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning. arXiv preprint arXiv:2601.21716.

Shi, S.; Xu, J.; Li, Z.; Peng, C.; Yang, X.; Lu, L.; Hu, K.; and Zhang, J. 2025. One-to-All Animation: AlignmentFree Character Animation and Image Pose Transfer. arXiv preprint arXiv:2511.22940.

Song, G.; Xu, H.; Zhao, X.; Xie, Y.; Gu, T.; Li, Z.; Zhang, C.; and Luo, L. 2025. X-UniMotion: Animating Human Images with Expressive, Unified and Identity-Agnostic Motion Latents. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, 1–11.

Tan, S.; Gong, B.; Wang, X.; Zhang, S.; Zheng, D.; Zheng, R.; Zheng, K.; Chen, J.; and Yang, M. 2025. Animate-X: Universal Character Image Animation with Enhanced Motion Representation. In The Thirteenth International Conference on Learning Representations.

Team, K.; Chen, J.; Ding, Y.; Fang, Z.; Gai, K.; He, K.; He, X.; Hua, J.; Lao, M.; Li, X.; Liu, H.; Liu, J.; Liu, X.; Shi, F.; Shi, X.; Sun, P.; Tang, S.; Wan, P.; Wen, T.; Wu, Z.; Zhang, H.; Zhao, R.; Zhang, Y.; and Zhou, Y. 2026. KlingMotionControl Technical Report. arXiv:2603.03160.

Unterthiner, T.; Van Steenkiste, S.; Kurach, K.; Marinier, R.; Michalski, M.; and Gelly, S. 2018. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717.

Wallace, B.; Dang, M.; Rafailov, R.; Zhou, L.; Lou, A.; Purushwalkam, S.; Ermon, S.; Xiong, C.; Joty, S.; and Naik, N. 2024. Diffusion Model Alignment Using Direct Preference Optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8228–8238.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Wang, X.; Zhang, S.; Tang, L.; Zhang, Y.; Gao, C.; Wang, Y.; and Sang, N. 2025. UniAnimate-DiT: Human Image Animation with Large-Scale Video Diffusion Transformer. arXiv:2504.11289.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

- Xu, Y.; Zhang, J.; Zhang, Q.; and Tao, D. 2022. ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation. In Advances in Neural Information Processing Systems.
- Xu, Z.; Ma, J.; Wang, Z.; Peng, Z.; Liang, J.; and Li, J.

2026. End-to-End Video Character Replacement without Structural Guidance. arXiv preprint arXiv:2601.08587.

Yan, W.; Ye, S.; Yang, Z.; Teng, J.; Dong, Z.; Wen, K.; Gu, X.; Liu, Y.-J.; and Tang, J. 2025. SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations. arXiv preprint arXiv:2512.05905.

Yang, X.; Kukreja, D.; Pinkus, D.; Sagar, A.; Fan, T.; Park, J.; Shin, S.; Cao, J.; Liu, J.; Ugrinovic, N.; Feiszli, M.; Malik, J.; Dollar, P.; and Kitani, K. 2026. SAM 3D Body: Robust Full-Body Human Mesh Recovery. arXiv:2602.15989. Yang, Z.; Teng, J.; Zheng, W.; Ding, M.; Huang, S.; Xu, J.; Yang, Y.; Hong, W.; Zhang, X.; Feng, G.; et al. 2025. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, volume 2025, 83048–83077.

Zhang, J.; Cao, S.; Li, R.; Zhao, X.; Cui, Y.; Hou, X.; Wu, G.; Chen, H.; Xu, Y.; Wang, L.; and Ma, K. 2026. SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation. arXiv:2511.19320.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, 586–595.

Zhao, Y.; Gu, A.; Varma, R.; Luo, L.; Huang, C.-C.; Xu, M.; Wright, T.; Shojanazeri, H.; Puglia, M.; Chen, S.; et al. 2023. PyTorch FSDP: experiences on scaling fully sharded data parallel. Proceedings of the VLDB Endowment, 16(12): 3848–3860.

Zhu, S.; Chen, J. L.; Dai, Z.; Dong, Z.; Xu, Y.; Cao, X.; Yao, Y.; Zhu, H.; and Zhu, S. 2024. Champ: Controllable and consistent human image animation with 3d parametric guidance. In European Conference on Computer Vision, 145– 162. Springer.

## SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning

Appendix

A Details on Data Composition

Even though we adopt SCAIL as the major animation generator for its optimization of cross-body-shape scenarios and complex motion, the model is not suitable for closeup shots and slow motion. To complement this, we adopt Wan-Animate as a supplement generator for those requirements. Our pipeline generates diverse pairing reference images and driving videos for the two generators to compose varied pairs, as shown in Fig. 11. To further improve data diversity as the synthetic driving videos are more diverse according to the distribution of reference image, we select around 5% motion pairs to not work in the reverse driving pattern, instead directly utilize the synthetic y˜ as the denoising target. Source videos are collected from internal datasets as well as public datasets (Chen et al. 2025b).

We employ LightX2V (Contributors 2025) to speed up generation of the pretraining synthetic dataset. The final composition of our MotionPair-60K, a dataset comprising 59,376 end-to-end motion-transfer pairs, is shown in Tab. 4.

Distribution of reference image I

Distribution of source video y

9%

33%

57%

10%

86%

3D animation 2D animation Real human

Figure 11: Distribution of data source.

Table 4: Composition of MotionPair-60K, along with the additional pose-driven dataset, and their corresponding sampling ratios used during training.

Source Construction #Pairs Sampling Ratio SCAIL Single

31,895

60% Wan-Animate Single

animation

13,847

animation

Single replacement

9,249

MoCha

20%

Multi replacement

4,385

- Single/Multi Pose Extraction

∼ 100,000 20%

### B Details on Post Training

#### B.1 Details of Preference Dataset

As noted in Section 3.4, the synthesized r is taken directly as the preferred target, so its fidelity propagates straight into the optimization signal. We therefore generate r with accurate estimator SDPose (Liang et al. 2026) to generate a clean positive sample and adopt strict filtering standards for those samples. To amplify error for the negative sample, we degrade the extra extraction passes P′ or P′′ in Eq. (3) with the less accurate ViTPose (Xu et al. 2022). Since both branches share the same reference image R and general global motion, the resulting gap between r and r− is concentrated in fine-grained articulation rather than in identity or global motion. For the animation generator G used in post-training we adopt Wan-Animate, which is also used to synthesize closeup shots as pretraining pairs. The final pipeline yields around 1K pairs of preference data.

#### B.2 Bias-Aware DPO Implementation

General Formulation. Following Diffusion-DPO (Wallace et al. 2024), we optimize a trainable flow matching model vθ against a frozen reference model vref. Given a preference dataset D = {(x+0 ,x−0 )} consisting of preferred and dispreferred samples, the DPO objective is

LDPO = − E(x+0 ,x−0 )∼D, x1∼N(0,I), t σ −

β 2

∆(x+0 ,x1,t) − ∆(x−0 ,x1,t) , (8)

where x+0 and x−0 denote the preferred and dispreferred samples, respectively. The term ∆(·,·,·) measures the relative

flow-matching error between the trainable model and the frozen reference model.

∆(x0,x1,t) = ∥vθ(xt,t) − v∥22 − ∥vref(xt,t) − v∥22 (9) xt = tx1 + (1 − t)x0 (10) v = x1 − x0 (11)

Regional DPO. As described in Section 3.4, the difference of the preference pairs comes from the estimation error of pose estimators, which is most significant in hand movements. To highlight such difference and avoid distraction brought by other factors, we apply the DPO objective only within hand regions.

Specifically, given a hand mask M, the per-sample DPO score is computed using masked velocity prediction errors:

∆M(x0,x1,t) = ∥M ⊙ (vθ(xt,t) − v)∥22− ∥M ⊙ (vref(xt,t) − v)∥22, (12)

where ⊙ denotes element-wise multiplication. We design M as the union of hand bounding boxes of positive and negative

samples at each frame, and directly downsample it into a mask in latent space.

We replace ∆ in Eq. (8) with ∆M when computing the DPO loss, and the DPO term over a preference pair with regional mask becomes

LMDPO(x+0 ,x−0 ,x1,t)

β 2

∆M(x+0 ,x1,t) − ∆M(x−0 ,x1,t) . (13)

=σ −

SFT Anchor. Optimizing the DPO objective alone leads to unstable training. We adopt a common approach, adding a supervised fine-tuning (SFT) item over positive samples to mitigate the problem:

LSFT(x+0 ,x1,t) = ∥vθ(x+t ,t) − v+∥22 (14) x+t = tx1 + (1 − t)x+0 (15) v+ = x1 − x+0 (16)

To stabilize training, we jointly optimize the SFT objective with the DPO objective:

L =E(x+0 ,x−0 ,M)∼D, x1∼N(0,I), t LSFT(x+0 ,x1,t) + λLMDPO(x+0 ,x−0 ,x1,t) , (17)

where λ = 0.01 is used to balance the scales of the two objectives. The SFT term serves as an anchor that prevents excessive divergence during preference optimization.

Training Details. During post-training, we freeze the backbone parameters and optimize only LoRA adapters inserted into the transformer layers, with rank 128. We use a learning rate of 1×10−4 and a batch size of 24. The DPO temperature parameter is set to β = 5000.

Discussions. Although the DPO loss is computed only within hand regions, it updates the policy globally rather than locally: the hand mask merely up-weights the most salient errors instead of confining optimization to the masked region. As shown in Fig. 10, this leads to improvements on other fine details such as the mouth. Compared with weighted SFT, which similarly emphasizes the hands but only fits positive samples, our Bias-Aware DPO yields visibly better fine-grained quality on both hand regions and other regions.

### C Evaluation Details

Pose-Driven Metrics. For self-driven part we employed several widely-used quantitative metrics, including PSNR (Hore and Ziou 2010), SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018), and FVD (Unterthiner et al. 2018). For the mesh adopted in Tab. 2, we adopt standard MHR-format Grey Mesh (Ferguson et al. 2025).

Human Evaluation Metrics. To evaluate the generated results in cross-identity settings, human evaluation is still the most convincing method. To conduct reasonable subjective studies, we design several metrics:

(1) Motion Accuracy, which measures how faithfully the generated motion follows the driving signal in a frame-byframe manner.

- (2) Identity Consistency, measuring the consistency of the

subject’s appearance with the reference image. For single character image animation, we measure:

- (3) Physical Plausibility, assessing whether the gener-

ated motions comply with basic physical constraints such as gravity, support, and momentum conservation, especially when object interactions are involved. This metric penalizes unrealistic behaviors like hovering in midair, objects morphing, or objects penetrating into human body.

For multi character image animation, we measure:

- (4) Identity Isolation, making sure that one character’s

limbs do not unnaturally merge with another character’s body, and their clothing remains strictly separated.

For replacement scenarios, we measure:

- (5) Environment Integration. This metric evaluates

whether the newly replaced characters fit naturally into the original scene and how well the reference video’s environment is maintained in the generated output, including the consistency of the background and the preservation of character-object interactions.

Automatic Metrics. For cases where the original quality of the video can work as a strong indicator of the performance, we adopt VideoBench’s human-aligned automatic protocol following DreamActor-M2 (Luo et al. 2026), focusing on four key perceptual dimensions: Imaging Quality, Motion Smoothness, Temporal Consistency, and Appearance Consistency. The protocol evaluates all those dimensions in a 1–5 scale (1=very poor, 2=poor, 3=moderate, 4=good, 5=excellent).

### D Quantitative Ablations

We conduct quantitative ablations on Binding Slots and Replacement Data on the cross-identity multi-character animation part of Studio-Bench. As shown in Table 5, both modules contribute positively to multi-character animation. Removing Binding Slots causes a clear drop in Appearance Consistency, demonstrating that the slots are key to keeping each character’s identity intact. Replacement Data, on the other hand, is crucial to alleviate generation of implausible scenes when characters overlap, as shown in Fig. 9.

Video-Bench Evaluations Imaging Quality ↑

Method

Temporal Consistency ↑

Appearance Consistency ↑

w/o Binding Slots 4.47 4.17 3.90 w/o Replacement 3.90 4.13 4.10

Full Model (Ours) 4.63 4.23 4.13

Table 5: Quantitative ablations on multi-character animation.

### E More Examples

We provide additional visualization results to further demonstrate the generalization of SCAIL-2 across challenging scenarios in Fig. 12, Fig. 13 and Fig. 14.

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Inputs

ref

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

SCAIL

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

Wan-Animate

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

Ours

- Figure 12: Examples of complex cross-body-shape character image animation. Our method maintains decent character consistency under complex motions.

ref

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Wan-Animate

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Mocha

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Ours

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

Inputs

- Figure 13: Examples requiring fine-grained HOI. Our method simultaneously preserves correct character identity and finegrained objects (e.g., thin sticks) during interaction. Zoom-in for better details.

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Inputs

ref

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Multi-Animate

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

SCAIL

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Wan-Animate

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Ours

- Figure 14: Examples of complex multi-character interactions. Our method accurately captures the interaction relationships among multiple characters with proper identity isolation.

