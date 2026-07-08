2026-6-8

[Figure 1]

## ViVa: A Video-Generative Value Model for Robot Reinforcement Learning

# arXiv:2604.08168v2[cs.RO]5Jun2026

Jindi Lv1,2*, Hao Li1*, Jie Li1, Fankun Kong1, Yang Wang1, Pengfei Yi1, Yifei Nie1, Xiaofeng Wang1,3, Zheng Zhu1†, Chaojun Ni1, Qiuping Deng1, Hengtao Li1, Jiancheng Lv2†, Guan Huang1 1GigaAI 2Sichuan University 3Tsinghua University * Equal Contribution † Corresponding Authors Project Page: https://viva-value-model.github.io/

###### Current Images

###### Scalar Value

= τ + : [

]

Progress towards successful task completion

[Figure 2]

[Figure 3]

[Figure 4]

 = 

[Figure 5]

 + − 

′ +   +  − 

,  =

Advantage ϵ {0,1}

′= 

Left wrist (t) Right wrist(t)

High (t)

∈ 14 (endeffector pose or joint angles)

 +  ∈ 14 (endeffector pose or joint angles)

Current Proprioception

Future Proprioception

ViVa

[Figure 6]

[Figure 7]

Figure 1: Illustration of ViVa. Given multi-view observations and proprioception, ViVa jointly predicts future proprioception and a task-progress value. By grounding value estimation in anticipated embodiment dynamics, it leverages spatiotemporal priors to couple value with foresight.

### Abstract

Vision-language-action (VLA) models have advanced robot manipulation through large-scale pretraining, but real-world deployment remains challenging due to partial observability and delayed feedback. Reinforcement learning addresses this via value functions, which assess task progress and guide policy improvement. However, existing value models built on vision-language models (VLMs) struggle to capture temporal dynamics and physical interactions, undermining reliable value estimation in long-horizon tasks. In this paper, we propose ViVa, a video-generative value model that repurposes a pretrained video generator to jointly predict future proprioception and a scalar value. By grounding value estimation in anticipated embodiment dynamics, ViVa leverages spatiotemporal priors to intrinsically couple value with foresight beyond static snapshots. ViVa achieves state-of-the-art results in metric-based evaluation across three tasks, producing reliable value signals that accurately track task progress and detect execution errors. Integrated into RECAP, it achieves an average success rate of 80%, highlighting the promise of video-generative models for value estimation.

### 1. Introduction

Building robots that can perceive, reason, and act in the physical world remains a central challenge in embodied artificial intelligence (Li et al., 2026; Sapkota et al., 2025). Vision-language-action (VLA) models (Intelligence et al., 2025; Kim et al., 2024; Li et al., 2025; Team et al., 2026; Zitkovich et al., 2023) have made significant strides by leveraging large-scale pretraining to enable general-purpose manipulation across diverse tasks. Yet success in real-world settings requires more than static scene understanding: robotic interaction unfolds under partial observability and delayed feedback, where the consequences of decisions only manifest over extended horizons (Huang et al., 2022; Zitkovich et al., 2023). Learning to connect present behavior with future outcomes thus remains a fundamental challenge for real-world robotics.

© 2026 GigaAI. All rights reserved.

This challenge calls for an ability that assesses whether ongoing interaction is progressing toward successful task completion. Such progress awareness allows robots to distinguish beneficial behaviors from undesirable ones and improve through experience. In reinforcement learning (RL) (Sutton et al., 1998), this capability is formalized by the value function, which estimates expected future outcomes and provides a learning signal for policy improvement. Recent VLA frameworks such as 𝜋0*.6 (Intelligence et al., 2025) highlight this importance: their RL with Experience and Corrections via Advantage-conditioned Policies (RECAP) pipeline relies on a value function for advantage estimation and policy refinement, demonstrating that learning performance strongly depends on value model quality.

Motivated by this importance, recent works have explored leveraging vision-language models (VLMs) (Bai et al., 2025; Chen et al., 2024; Comanici et al., 2025; Li et al., 2024; Marafioti et al., 2025; Zhu et al., 2025) for value estimation, framing value prediction as classification (Intelligence et al., 2025) or temporal ordering problems (Ma et al., 2024). While promising, these approaches inherit a key limitation: VLMs are primarily trained on static image–text data for semantic understanding rather than explicitly modeling how scenes evolve over time. Accordingly, they capture what is present in a scene but struggle to represent how interactions dynamically transform the environment. This mismatch limits their ability to support reliable value estimation in temporally extended robotic tasks.

The above limitations reveal a key insight: value estimation is inherently a problem of anticipating how the future will unfold. In contrast to discriminative models trained on static data, video generative models are explicitly optimized to capture temporal evolution, learning how scenes change as interactions unfold. This makes them a natural foundation for value estimation, as the ability to imagine future outcomes directly enables assessing whether current behavior progresses toward task completion. Guided by this observation, we reformulate value learning as future prediction and develop a video-generative value model.

In this paper, we propose Video-generative Value model (ViVa), a novel approach that repurposes a pretrained video generator as a value function for robotic reinforcement learning. By leveraging the spatiotemporal priors learned from large-scale video corpora, our model captures rich dynamics about how scenes evolve over time. Taking the current observation together with robot proprioception as input, ViVa jointly predicts future proprioception and a scalar value for the current state (Figure 1). Grounding value estimation in anticipated embodiment dynamics enables ViVa to incorporate predictive structure beyond static snapshots, intrinsically coupling value with foresight. This design provides more reliable value signals for advantage computation, leading to improved policy optimization in robotic manipulation tasks.

We conduct extensive experiments across three long-horizon manipulation tasks. ViVa demonstrates strong sensitivity to fine-grained execution errors and robust generalization to novel objects. Integrated into RECAP for real-robot evaluation, it delivers consistent performance gains, highlighting the potential of video-generative models as a new paradigm for value estimation in robotics.

We highlight the main contributions of this paper below:

- • We identify robotic value estimation as fundamentally a future anticipation problem, for which video generative models offer a more natural foundation than discriminative VLMs.
- • We introduce ViVa, a video-generative value model that intrinsically couples value with foresight by jointly predicting future embodiment dynamics alongside the current value.
- • ViVa achieves state-of-the-art performance across three long-horizon tasks in metric-based evaluation and real-robot experiments, with sensitivity to execution errors and robust generalization to novel objects.

### 2. Related Works

- 2.1. Value Functions in Robot Learning Value functions are central to reinforcement learning for robotics, providing learning signals under delayed and sparse feedback (Ross et al., 2011; Sutton et al., 1998). Prior works have applied reinforcement learning to robotic manipulation, spanning offline Q-learning (Huang et al., 2025; Kalashnikov et al., 2018; Levine et al., 2020; Mandlekar et al., 2020), autonomous real-world interaction (Lampe et al., 2024; Luo et al., 2024; Mendonca et al., 2023; Sharma et al., 2023), and end-to-end value-guided policy optimization (Ghasemipour et al., 2025; Zhai et al., 2025). To enable reinforcement learning for VLA models (Cheang et al., 2024; Kim et al., 2024; Li et al., 2024; Liu et al., 2024; O’Neill et al., 2024; Team et al., 2024), recent works have explored VLM-based value estimation (Frans et al., 2025; Ma et al., 2023, 2024). For instance, GVL (Ma et al., 2024) directly queries VLMs to assess task progress by framing value prediction as temporal ordering over shuffled video frames. TopReward (Chen et al., 2026) instead leverages VLM token probabilities as zero-shot reward

signals. 𝜋0*.6 (Intelligence et al., 2025) trains a dedicated VLM-based value model with dense supervision as a core component of the RECAP pipeline for advantage-conditioned policy refinement.

Despite their differing designs, these methods share a fundamental limitation: they all rely on VLMs trained on static images, which capture per-frame semantics but lack explicit modeling of temporal dynamics and physical interactions. This motivates leveraging video generative models, which learn spatiotemporal dynamics directly from large-scale video data and offer a natural foundation for value estimation in long-horizon tasks.

- 2.2. Video Generation Models for Robot Manipulation Video generation models (Blattmann et al., 2023; Kong et al., 2024; Yang et al., 2024; Zheng et al., 2024) are explicitly optimized to capture temporal evolution, commonly via diffusion Transformers (Bao et al., 2023; Peebles and Xie, 2023) conditioned on language (Blattmann et al., 2023; Singer et al., 2022; Villegas et al.,

2022) or visual (Ceylan et al., 2023; Qi et al., 2023) inputs. These properties make them well-suited for anticipating visual dynamics.

In robotics, video generation has been applied to planning via world models (Zhou et al., 2024), where generated futures simulate action outcomes for decision making. It has also been integrated into policy learning, by extracting actions via inverse dynamics (Du et al., 2023; Yang et al., 2023), conditioning policies on goal frames (Du et al., 2023; Zhang et al., 2025), or jointly generating video and actions (Cheang et al., 2024; Wu et al., 2023; Ye et al., 2026,). More recently, human-to-robot transfer has been explored by synthesizing human-object interaction videos (Bharadhwaj et al., 2024; Kareer et al., 2025; Zhao et al., 2025).

Despite these advances, existing methods primarily leverage video generation to produce or guide actions. In contrast, we investigate a complementary role: value estimation. We propose ViVa, a video-generative value model that repurposes a pretrained video generator to predict scalar values, grounding value estimation in anticipated embodiment dynamics.

- 3. Method

##### 3.1. Problem Formulation We formalize robotic manipulation as a Markov decision process. At each time step 𝑡, the agent receives an observation comprising multi-view RGB images o𝑡 and proprioception q𝑡; we denote the joint observation as x𝑡 = (o𝑡,q𝑡). The value function

𝑉 𝜋(x𝑡) = E𝜋[︁∑︁𝑇 𝑘=𝑡

𝑟𝑘 ⃒ x𝑡]︁

estimates the expected future return under policy 𝜋. We argue that accurate value estimation should be grounded in understanding temporal dynamics and physical interactions, which static image-based models

###### Input Injection

###### Architecture of ViVa

decode

scalar value (0~1)

future proprio (14-dim)

 + 

current proprio (14-dim)

scalar value

Repeat & Truncate

Broadcast

###### Wan DiT (Diffusion Transformer)

Target Frames (noised → denoised)

Conditioning Frames (clean)

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

1 2 3

 + 

injected proprio

injected value

- Figure 2: Overall architecture of ViVa. Left: Proprioception and the scalar value are embedded into latent frames via repeat-padding and broadcast. Right: The latent sequence comprises clean conditioning frames (blank token, proprioception, multi-view images) and noisy target frames (future proprioception and value). The diffusion Transformer denoises the targets conditioned on the clean prefix, jointly predicting future embodiment state and value.

struggle to capture. To address this, we propose a video-generative value model that leverages spatiotemporal priors from video data, jointly predicting future proprioception alongside the current value to ground value estimation in anticipated physical dynamics.

- 3.2. Overall Architecture We build our video-generative value model upon Wan2.2 (Wan et al., 2025), a pretrained video diffusion Transformer that originally generates future frames conditioned on an initial image and text. To adapt it for value estimation, we extend its input and output modalities via latent injection (Agarwal et al., 2025; Liang

- et al., 2025), without modifying the core architecture. The overall architecture of ViVa is illustrated in Figure 2.

Latent encoding of modalities. All input and output modalities are mapped to latent frames of shape (𝐻′,𝑊′,𝐶′), where 𝐻′,𝑊′ are the spatial dimensions after VAE downsampling and 𝐶′ is the latent channel dimension. We use a pretrained spatiotemporal VAE to encode images: each camera view o𝑖𝑡 is independently compressed into a latent frame zo𝑖

. For low-dimensional vectors such as the proprioceptive state q𝑡 and the scalar value 𝑣𝑡, we design specialized injection procedures. Both are first normalized to [−1,1] to match the latent space statistics. The proprioceptive state q𝑡 is embedded via repeat-padding: we repeat its elements to match the latent frame size 𝐻′𝑊′𝐶′ and reshape to (𝐻′,𝑊′,𝐶′), producing zq

𝑡

. The scalar value 𝑣𝑡 is embedded via broadcast: we set every element of a latent frame to the same normalized value, yielding z𝑣

𝑡

.

𝑡

Latent sequence during training. During training, we assemble a fixed-length sequence of latent frames that includes both conditioning and target frames. Let 𝐾 denote a fixed prediction horizon. The sequence is:

[zblank, zq

###### , zo1

###### , zo2

###### , zo3

###### , zq

, z𝑣

],

𝑡

𝑡

𝑡+𝐾

𝑡

𝑡

𝑡

where zblank is a zero-initialized placeholder required by the causal VAE. The first five frames (blank, current proprioception, and current images) serve as clean conditioning, while the remaining two frames (future proprioception zq

) are corrupted with Gaussian noise at a randomly sampled level 𝜎. The denoiser 𝐷𝜃 learns to recover the clean targets from the noisy ones, conditioned on the clean prefix. Latent sequence during inference. At inference time, only the conditioning frames are available. We encode

and value z𝑣

𝑡

𝑡+𝐾

the current observations (images and proprioception) into their respective latent frames, form the same prefix [zblank,zq

. The predicted value 𝑣ˆ𝑡 is obtained by averaging all elements of zˆ𝑣

], and run reverse diffusion to generate the target frames zˆq

and zˆ𝑣

###### ,zo3

###### ,zo2

###### ,zo1

𝑡

𝑡

𝑡+𝐾

𝑡

𝑡

𝑡

and rescaling from [−1,1] back to [0,1]. To recover the future proprioceptive state qˆ𝑡+𝐾, we apply the inverse of the repeat-padding injection: flatten zˆq

𝑡

, split into consecutive chunks of size equal to the original proprioception dimension, average each chunk, and rescale to the original range.

𝑡+𝐾

Training objective. We adopt the flow matching formulation as in Wan2.2 (Wan et al., 2025). Let z0 denote a clean latent frame (either zq

), and let z1 ∼ 𝒩(0,I) be a Gaussian noise latent of the same shape. We construct a linear interpolation path

or z𝑣

𝑡

𝑡+𝐾

z𝜏 = (1 − 𝜏)z0 + 𝜏z1, 𝜏 ∈ [0,1].

The model 𝑣𝜃(z𝜏;𝜏,c) is trained to predict the constant velocity z1 − z0 along this path. The overall objective is a weighted combination:

[︀‖𝑣𝜃(zq𝜏;𝜏,c) − (z1 − zq0)‖22

]︀

ℒ = 𝜆propEzq

0∼𝑝data,z1∼𝒩(0,I),𝜏∼𝒰[0,1]

[︀‖𝑣𝜃(z𝑣𝜏;𝜏,c) − (z1 − z𝑣0)‖22

]︀

+ 𝜆valEz𝑣

.

0∼𝑝data,z1∼𝒩(0,I),𝜏∼𝒰[0,1]

where zq𝜏 and z𝑣𝜏 are modality-specific interpolated latents, c denotes the clean conditioning frames, and 𝜏 ∼ 𝒰[0,1] is the flow time step. We also experimented with jointly predicting future visual latents, but observed a degradation in value estimation accuracy. We hypothesize that this is due to the inherent difficulty mismatch between the two tasks: visual generation requires capturing high-dimensional spatial structure, while the value latent has a much simpler structure and is more susceptible to interference from the visual reconstruction objective during joint optimization.

By treating all modalities as latent frames, our architecture repurposes a powerful video generator for value estimation while preserving its spatiotemporal priors. The inclusion of future proprioceptive prediction serves two purposes: it forces the model to internalize the robot’s own dynamics, which is essential for tasks requiring precise limb coordination, and it provides an implicit measure of motion that complements visual cues for value estimation.

- 3.3. Reward Definition and Value Training We now define the learning targets for our video-generative value model. Each episode in the training data is annotated with a binary success label indicating the final task outcome. For an episode of length 𝑇, we define the step-wise reward 𝑟𝑡 to encode both temporal progress and completion status:

⎧ ⎪⎨

1 𝑇 , if 𝑡 < 𝑇,

(1)

- 0, if 𝑡 = 𝑇 and success,
- 1, if 𝑡 = 𝑇 and failure,

𝑟𝑡 =

⎪⎩

∑︀𝑇

where 𝑡 = 1,...,𝑇. Under this formulation, the cumulative return 𝐺𝑡 =

𝑘=𝑡 𝑟𝑘 provides a discriminative supervision signal that distinguishes outcomes through distinct value ranges:

⎧ ⎨

𝑇 , if success, 𝑇−𝑡

𝑇−𝑡

(2)

𝐺𝑡 =

⎩

𝑇 + 1, if failure.

Under this formulation, 𝐺𝑡 reflects normalized task progress within [0,1) for successful episodes, while failed

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

BoxPackagingRollOrganizingPantsFoldingShirtFolding

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

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- Figure 3: Illustration of the three real-world tasks. For each task, we show the initial state (left), key intermediate stages (middle), and the final successful state (right).

episodes are shifted to [1,2) by the terminal penalty. This ensures a constant margin of 1.0 between outcomes at any temporal stage, effectively resolving the ambiguity between progress and failure in value estimation.

The return 𝐺𝑡 serves as the supervision signal for the value latent z𝑣

, which is treated as the clean target in the flow matching objective described in Sec. 3.2. This formulation provides a consistent and outcome-aware supervision signal across episodes of varying lengths. By jointly predicting the return and future proprioception, the model learns to capture both task-level integrity and the robot’s embodied dynamics, effectively grounding value estimation in anticipated embodiment evolution.

𝑡

### 4. Experiments

#### 4.1. Experimental Setup

- 4.1.1. Evaluation Tasks Our evaluations and real-robot experiments span three long-horizon manipulation tasks: shirt folding, box packaging, and paper roll organization, summarized below and illustrated in Figure 3.

Shirt folding. This task evaluates dual-arm coordination for manipulating highly deformable textiles. The robot must first take a garment from a basket, flatten it on the table, execute a folding sequence (sleeves and sides inward, then longitudinal and cross folds), and return the folded shirt to the basket. Success requires neat folding and placement in the basket within 300 seconds. Failure occurs if entanglement damages the garment, fold collapses, or return is improper.

Box packaging. This task evaluates long-horizon dual-arm coordination through a multi-stage manipulation sequence. The robot picks a target item, places it into a cardboard box, then folds the side flaps and closes the lid. Success requires the item to be fully enclosed in a structurally sound box with all tabs interlocked within 300 seconds. Failure occurs if the item is dropped, the box is damaged, or the box cannot be fully sealed.

Paper roll organization. This task evaluates precise, multi-stage manipulation of flexible paper. The robot must grasp and tear off a single sheet, discarding it into a receptacle, then collaboratively rewind the remaining loose end until it is flush with the roll. Finally, a sealing sticker is applied to secure the end. Success requires completing the tear, disposal, and sealing within 240 seconds. Excessive tearing or failure to secure the sticker is recorded as a failure.

5.6%

7.1%

9.1%

7.1%

11.2%

18.2%

44.5%

45.5%

12.0%

Shirt Folding

Box Packaging

Paper Roll Organization

50.0%

35.7%

27.3%

26.7%

Failed to spread flat Poor spreading quality Grasp failure Basket placement failure

Side Panel Operation Box Operation Ear Operation Object A/B Operation Lay-down

Label and Tear Operation Roll Retrieval Deviation Roll Retrieval Slip Roll Displacement

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

| |
|---|

| |
|---|

| |
|---|

- Figure 4: Error type distribution for shirt folding, box packaging, and paper roll organization. Each pie chart shows the relative frequency of failure modes, based on 50 held-out episodes per task.

- 4.1.2. Evaluation Dataset We use 50 held-out episodes per task for evaluation. Milestone and error frames are pre-defined per task and manually annotated for each episode. Milestone frames mark key task stages and appear in every episode; error frames, by contrast, are annotated only for episodes in which the corresponding failure mode actually occurs. Figure 4 reports the error type distribution for each task.

Shirt folding. Failures are concentrated in the flattening stage, with Failed to spread flat (45.5%) and Poor spreading quality (27.3%) being the most frequent, followed by Grasp failure and Basket placement failure. The key challenge lies in reliably flattening a highly deformable textile without residual wrinkles or misalignment.

Box packaging. Failures span multiple sub-tasks, dominated by Side panel operation (44.5%) and Box operation (26.7%), with additional errors in Ear operation, Object A/B operation, and Lay-down. The difficulty stems from the need to coordinate a sequential, multi-stage assembly where each step must be executed precisely before the next can begin.

Paper roll organization. The majority of failures involve tearing and retrieval: Label and tear operation (50.0%) and Roll retrieval deviation (35.7%), along with Roll retrieval slip and Roll displacement. The core challenge is fine-grained manipulation of flexible paper, where even small deviations in tearing angle or retrieval trajectory can ruin the product.

- 4.1.3. Implementation Details

For the VLM-based value model, we follow the same design as 𝜋0*.6 (Intelligence et al., 2025), formulating value estimation as a 201-way classification problem over return bins. Both this baseline and our ViVa-based variant are trained within the identical RECAP pipeline. We conduct two rounds of rollouts: the first round collects 150 successful and 50 failed episodes per task, and the second round collects 100 successful and 50 failed episodes per task. The rollout data are combined with demonstration data from all three tasks. All models are trained for one epoch with batch size 192. The prediction horizon 𝐾 is set to 50 for all tasks, aligning with the default horizon used for advantage estimation in the RECAP framework. For ViVa, the loss weights for future proprioception and value prediction are set to 𝜆prop = 0.5 and 𝜆val = 1.0, respectively. During inference, we use 1 denoising step for ViVa with DDIM sampling. All experiments are conducted on 8 NVIDIA A800 GPUs.

- 4.2. Evaluation Metrics Event Response Score (ERS). For each evaluation episode, we annotate a set of key frames 𝒦, each labeled with an expected direction 𝑑𝑘 ∈ {+1,−1}: 𝑑𝑘 = +1 for milestones and 𝑑𝑘 = −1 for execution errors. Let 𝑉before(𝑘)

and 𝑉after(𝑘) denote the mean predicted value over a window of 𝑤 frames immediately preceding and following the annotated frame. The ERS for key frame 𝑘 is defined as

ERS𝑘 = 𝑑𝑘 ·

𝑉after(𝑘) − 𝑉before(𝑘) 𝑉before(𝑘) + 𝜖

, (3)

where 𝜖 is a small positive constant to avoid division by zero. We then define Milestone Sensitivity (MS) as the average ERS over all milestone frames (𝑑𝑘 = +1), and Error Sensitivity (ES) as the average ERS over all error frames (𝑑𝑘 = −1). Positive values indicate correct directional responses, with larger magnitudes reflecting stronger sensitivity; negative values indicate incorrect ones.

Event Detection Rate (EDR). Based on the ERS, we consider an annotated event frame 𝑘 as detected if ERS𝑘 > 0. Using this binary detection criterion, we define the Milestone Detection Rate (MDR) and Error Detection Rate (ErrDR) as the proportions of correctly detected frames within each event type:

MDR =

𝑁milestone+ 𝑁milestone

, ErrDR =

𝑁error+ 𝑁error

, (4)

where 𝑁milestone+ and 𝑁error+ denote the numbers of milestone frames (𝑑𝑘 = +1) and error frames (𝑑𝑘 = −1) with positive ERS values, respectively, and 𝑁milestone, 𝑁error are the total counts of milestone and error frames.

- 4.3. Value Model Evaluation

Evaluation setup. We benchmark ViVa against recent state-of-the-art value models on three tasks: shirt folding, box packaging, and paper organization.

GVL (Ma et al., 2025) and TopReward (Chen

Table 1: Comparison with robotics SOTA value models. Sparse-sampled methods show low or negative correlation. Dense-supervised approaches perform much better, and ViVa establishes new SOTA on two of three tasks.

- et al., 2026) leverage VLMs for sparse-sampled

value estimation, while 𝜋0*.6 (Intelligence et al., 2025) and ViVa are trained on ∼20 hours of mixed demonstration data from all three tasks. We use 50 held-out episodes per task and manually annotate milestone and error frames for evaluation. See Sec. 4.1.2 for dataset details.

Shirt Box Paper Method

Pearson Spearman Pearson Spearman Pearson Spearman Sparse-sampled

GVL 0.276 0.263 -0.081 -0.114 -0.099 -0.096 TopReward 0.118 0.090 0.503 0.499 0.318 0.241

Quantitative comparison. Table 1 reports Pearson and Spearman correlation as a global measure of value quality. GVL and TopReward yield low or even negative correlations, while

Dense-supervised

𝜋0*.6 0.740 0.720 0.946 0.945 0.926 0.936 ViVa 0.984 0.982 0.868 0.850 0.952 0.948

𝜋0*.6 and ViVa achieve consistently strong correlations, with ViVa leading on two of three tasks and 𝜋0*.6 ahead on box packaging. This performance gap stems from GVL and TopReward being training-free methods that query VLMs at sparse intervals, lacking the dense temporal supervision needed to reliably associate static frames with task progress. In contrast, dense supervision on every frame enables 𝜋0*.6 and ViVa to capture the continuous evolution of manipulation behaviors.

For 𝜋0*.6 and ViVa, however, higher global correlation does not necessarily imply better value quality. Nearuniform predictions collapse advantages to zero, providing no RL signal. We therefore evaluate on four

fine-grained event-level metrics for a more faithful comparison. Tables 2 and 3 show ViVa outperforming 𝜋0*.6

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

uneven force misaligns the box

improper manipulation tilts the box

missed grasp misaligns the box

premature release prevents box closure

asynchronous lifting tilts the box

- Figure 5: Detection of subtle manipulation errors by ViVa. The model produces clear value drops at failure moments, confirming sensitivity to fine-grained execution errors.

across nearly all metrics. On shirt folding, 𝜋0*.6 yields negative MS (−0.266) and ES (−0.299), indicating that its predictions move in the wrong direction at key events, while ViVa achieves positive values on both. For

detection, ViVa attains MDR of 0.990 and ErrDR of 1.000, nearly double 𝜋0*.6’s 0.407 and 0.500. This gap persists across tasks, and Fig. 5 qualitatively confirms ViVa’s sensitivity to subtle execution errors.

Table 2: Event-sensitivity comparison. MS (milestone sensitivity) and ES (error sensitivity) measure directional response quality. ViVa achieves substantially higher sensitivity than the VLM baseline across all three tasks.

Table 3: Event-detection comparison. MDR (milestone detection rate) and ErrDR (error detection rate) measure the proportion of correctly detected event frames. ViVa consistently achieves higher detection rates across all tasks.

Shirt Box Paper Method Backbone

Shirt Box Paper Method Backbone

MS ES MS ES MS ES

MDR ErrDR MDR ErrDR MDR ErrDR

𝜋0*.6 VLM -0.266 -0.299 0.013 0.006 0.023 0.028 ViVa Video-gen 0.095 0.012 0.034 0.026 0.039 0.055

𝜋0*.6 VLM 0.407 0.500 0.765 0.506 0.704 0.667 ViVa Video-gen 0.990 1.000 0.733 0.612 0.762 0.778

Trajectory visualization. Figure 6 visualizes the predicted value trajectories of all four methods on the three in-domain tasks. GVL and TopReward, which query VLMs at sparse intervals, produce noisy and flat value curves that fail to reflect task progression. 𝜋0*.6 captures the overall trend with dense supervision, but its responses at fine-grained events are often delayed or muted. In contrast, ViVa is more sensitive to fine-grained manipulation events, with clear value increases at milestones and decreases at execution errors. This suggests that grounding value in anticipated dynamics contributes to more precise progress awareness.

#### 4.4. Object Generalization

Evaluation setup. We further evaluate 𝜋0*.6 and ViVa on an out-of-domain pants folding task comprising 50 held-out episodes annotated with milestone frames. We focus on milestones, as error detection on novel objects remains challenging without large-scale pretraining.

GVL TopReward 0.6* ViVa

Box packaging

Shirt folding

Paper roll organization

Pants Folding

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

0.6

Progress(0-1)

0.4

0.2

0.0

1.0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Normalized time

- Figure 6: Value trajectory visualization across all tasks. Trajectories for 𝜋0*.6 and ViVa are smoothed by 20%. ViVa tracks task progress with clear responses at key events and maintains sensitivity on the unseen task, demonstrating stronger generalization.

Quantitative comparison. Table 4 reports object generalization results on the out-of-domain pants folding task. ViVa outperforms 𝜋0*.6 across all four metrics, with the largest gains on fine-grained event metrics: MS more than doubles (0.037 vs. 0.016) and MDR improves by over 60% (0.643 vs. 0.397) on this outof-domain task.

Table 4: Object generalization on the out-of-domain pants folding task. ViVa consistently outperforms 𝜋0*.6 across all evaluation metrics.

Method Pearson Spearman MS MDR

𝜋0*.6 0.740 0.721 0.016 0.397 ViVa 0.819 0.799 0.037 0.643

Trajectory visualization. The pants folding panel in Figure 6 shows value trajectories on the out-of-domain task. GVL and TopReward remain unresponsive, consistent with their general inability to capture temporal progress. 𝜋0*.6 captures the overall trend but misses key milestones. In contrast, ViVa consistently responds to milestone events, demonstrating that its spatiotemporal priors provide a more transferable representation for generalization.

#### 4.5. Real-Robot Experiments

Setup. We evaluate on three real-world tasks: shirt folding, box packaging, and paper roll organization. We compare imitation learning baselines Gigabrain-0 (Team et al., 2025) and 𝜋0.5 (Intelligence et al., 2025) against two RECAP (Intelligence et al., 2025) variants, both built on Gigabrain-0 with different value models: a VLM-based value function versus ViVa. Both RECAP variants undergo two rounds of rollouts. See Sec. 4.1.3.

Table 5: Real-robot experiment results. Success rates (%) on three manipulation tasks. RECAP (ViVa) outperforms both imitation learning baselines and RECAP (VLM) on all tasks.

Method Shirt (%) Box (%) Paper (%) Avg (%) Imitation learning

𝜋0.5 60.0 40.0 40.0 46.7

Gigabrain-0 50.0 50.0 20.0 40.0 Reinforcement Learning

RECAP (VLM) 70.0 60.0 60.0 63.3 RECAP (ViVa) 90.0 70.0 80.0 80.0

Table 6: Ablation on architecture design for box packaging. Replacing the VLM backbone with VG alone improves milestone metrics, and adding future state prediction further boosts error sensitivity and detection. VG denotes video generator.

Input State

Future

Method Backbone

State MS ES MDR ErrDR 𝜋0*.6 VLM ✗ ✗ 0.013 0.006 0.765 0.506

VG-only VG ✗ ✗ 0.048 0.004 0.853 0.449 ViVa w/o fut VG ✓ ✗ 0.047 0.008 0.778 0.527

ViVa VG ✓ ✓ 0.034 0.026 0.733 0.612

Main results. Table 5 reports the success rates on three long-horizon manipulation tasks. Both RECAP variants substantially outperform the imitation learning baselines, confirming the benefit of reinforcement learning. Replacing the VLM-based value function with ViVa yields further gains across all tasks, demonstrating that better value estimation translates to stronger policy improvement.

Rollout-wise performance evolution. Figure 7 shows the success rates across rollout rounds. ViVa improves substantially faster from R0 to R1: on shirt folding, it jumps from 50% to 80% while VLM stagnates at 60%; on

100

100

100

RECAP (VLM) RECAP (ViVa)

RECAP (VLM) RECAP (ViVa)

RECAP (VLM) RECAP (ViVa)

SuccessRate(%)

SuccessRate(%)

SuccessRate(%)

80

80

80

| |
|---|

| |
|---|

| |
|---|

60

60

60

40

40

40

20

20

20

0

0

0

rollout-0 rollout-1 rollout-2

rollout-0 rollout-1 rollout-2

rollout-0 rollout-1 rollout-2

(a) Shirt folding. (b) Box packaging. (c) Paper roll organization.

- Figure 7: Rollout-wise performance evolution across three manipulation tasks. Both methods improve over successive rounds, with RECAP (ViVa) achieving the best overall performance.

Table 7: Ablation on architecture design for shirt folding. VG-only outperforms the VLM baseline across all metrics, with further gains from future state prediction. VG denotes video generator.

Input State

Future

Method Backbone

State MS ES MDR ErrDR 𝜋0*.6 VLM ✗ ✗ -0.266 -0.299 0.407 0.500

VG-only VG ✗ ✗ 0.137 -0.026 1.000 0.500 ViVa w/o fut VG ✓ ✗ 0.093 0.006 0.998 1.000 ViVa VG ✓ ✓ 0.095 0.012 0.990 1.000

Table 8: Ablation on architecture design for paper roll organization. ViVa with full architecture achieves the best performance across all metrics. VG denotes video generator.

Input State

Future

Method Backbone

State MS ES MDR ErrDR 𝜋0*.6 VLM ✗ ✗ 0.023 0.028 0.704 0.667

VG-only VG ✗ ✗ 0.016 -0.011 0.651 0.444 ViVa w/o fut VG ✓ ✗ 0.019 -0.043 0.644 0.444 ViVa VG ✓ ✓ 0.039 0.055 0.762 0.778

paper roll organization, it surges from 20% to 80% vs. VLM’s 20% to 50%. From R1 to R2, VLM plateaus while ViVa keeps improving, reaching 90% on shirt and extending its lead on box. Overall, ViVa converges faster to stronger final performance.

#### 4.6. Ablation Studies

Effect of video generator backbone. We conduct architecture ablation on all three tasks, as reported in Tables 6, 7 and 8. Across the board, replacing the VLM backbone with a video generator brings consistent gains. On box packaging, the video backbone alone lifts MS from 0.013 to 0.048 and MDR from 0.765 to 0.853. The improvement is even more pronounced on shirt folding, where the VLM baseline suffers from negative MS (−0.266) and ES (−0.299), meaning its predictions move in the wrong direction at key events. Switching to a video generator reverses this to a positive MS of 0.137 and lifts MDR from 0.407 to 1.000. On paper roll organization, the video backbone alone yields mixed results, with the full architecture providing the clearest gains (discussed next). These results confirm that video generative priors offer a substantially stronger foundation for value estimation than static VLM representations.

Effect of input and future state. Building on the video generator backbone, we further examine the effect of incorporating proprioceptive state and future dynamics prediction across all three tasks (Tables 6, 7, 8). On box packaging, adding input state lifts ErrDR from 0.449 to 0.527, and future state prediction further raises ES to 0.026 and ErrDR to 0.612. On shirt folding, the full architecture brings ES from −0.026 (VG-only) to 0.012, turning a negative error sensitivity into a positive one, while maintaining near-perfect MDR (0.990) and ErrDR (1.000). On paper roll organization, the full ViVa achieves the strongest results across all metrics (MS 0.039, ES 0.055, MDR 0.762, ErrDR 0.778), while the VG-only variant yields negative ES (−0.011). These consistent improvements confirm that future dynamics prediction sharpens error awareness beyond what the video backbone alone provides, making the value model more sensitive to task-relevant execution events.

Table 9: Ablation on pretrained weights for box packaging. Removing pretrained weights degrades performance across all metrics, confirming the importance of spatiotemporal priors.

Method Pretrained MS ES MDR ErrDR ViVa w/o pretrain ✗ 0.009 0.023 0.523 0.524

ViVa ✓ 0.034 0.026 0.733 0.612

Effect of pretrained weights. We ablate the pretrained video weights on all three tasks, as reported in Tables 9, 10 and 11. Across the board, removing pretrained weights causes substantial degradation. On box packaging, MS drops from 0.034 to 0.009 and MDR falls from 0.733 to 0.523. The impact is even more severe on shirt

Table 10: Ablation on pretrained weights for shirt folding. Removing pretrained weights degrades performance, showing spatiotemporal priors’ value.

Table 11: Ablation on pretrained weights for paper roll organization. ViVa with pretrained weights outperforms its non-pretrained counterpart.

Method Pretrained MS ES MDR ErrDR ViVa w/o pretrain ✗ 0.117 -0.524 0.770 0.500

Method Pretrained MS ES MDR ErrDR ViVa w/o pretrain ✗ 0.034 0.034 0.489 0.667

ViVa ✓ 0.095 0.012 0.990 1.000

###### ViVa ✓ 0.039 0.055 0.762 0.778

folding, where ES plummets from 0.012 to −0.524, effectively turning a reliable error detector into one that signals in the wrong direction at error events. On paper roll organization, MDR degrades from 0.762 to 0.489 and ErrDR drops from 0.778 to 0.667. These consistent results confirm that spatiotemporal priors from large-scale video pretraining are indispensable for building value models that generalize across tasks and reliably detect execution errors.

#### 4.7. Hyperparameter Analysis

Effect of loss weight. We vary 𝜆prop (the loss weight for future proprioception prediction) from 0.1 to 0.75, as reported in Figure 8. Low values, such as 𝜆prop = 0.1, lead to negative error sensitivity (ES) on shirt folding and paper roll organization, and yield zero error detection (ErrDR) on shirt folding. This indicates that insufficient proprioception modeling weakens embodiment grounding for reliable error awareness. Increasing 𝜆prop to 0.5 yields positive ES across all three tasks and achieves high ErrDR (1.0 on shirt folding, 0.78 on paper roll organization, and 0.61 on box packaging). Further increasing to 0.75, however, degrades performance on paper roll organization (ES becomes negative again) and slightly reduces ErrDR on box packaging. Consequently, 𝜆prop = 0.5 offers the most robust and balanced performance across tasks and is adopted for all experiments.

MS ES MDR ErrDR

Box packaging

Shirt folding

Paper roll organization

0.12

0.08

0.045

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.08

0.030

0.04

0.04

0.015

0.00

0.00

0.000

0.04

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.8

0.76

0.8

0.7

0.72

0.4

0.6

0.68

0.0

0.5

0.1 0.25 0.5 0.75

0.1 0.25 0.5 0.75

0.1 0.25 0.5 0.75

prop prop prop

- Figure 8: Loss weight analysis for balancing value and future proprioception predictions. Setting 𝜆prop = 0.5 achieves the best trade-off and is adopted across all experiments.

Effect of prediction horizon. We vary the prediction horizon 𝐾 across {10,25,50,75}, as shown in Figure 9. At 𝐾 = 10, error detection collapses on shirt folding (ErrDR = 0.000, ES = −0.002), indicating that short horizons fail to capture sufficient temporal context. At 𝐾 = 25, milestone sensitivity improves (MS reaches 0.055 on box and 0.062 on paper) but ES on paper remains near zero (0.001). At 𝐾 = 50, ES becomes positive across all three tasks and ErrDR reaches 1.0 on shirt folding, achieving the best overall balance. At 𝐾 = 75, MS peaks on shirt (0.113) but ES turns negative (−0.002) and ErrDR drops to 0.500, suggesting overly long horizons introduce noise. Thus 𝐾 = 50 is adopted for all experiments.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.000

0.015

0.030

0.045

Box packaging

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.00

0.04

0.08

0.12

Shirt folding

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

0.00

0.03

0.06

Paper roll organization

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

10 25 50 75

0.5

0.6

0.7

0.8

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

10 25 50 75

0.0

0.4

0.8

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

10 25 50 75

0.56

0.64

0.72

0.80

Horizon Horizon Horizon

MS ES MDR ErrDR

- Figure 9: Hyperparameter analysis on the prediction horizon 𝐾 for future proprioception prediction. Setting 𝐾 = 50 achieves the best trade-off across tasks and is adopted for all experiments.

- 4.8. Efficiency Comparison Table 12 compares the computational cost of three value model variants. The VLM-based baseline follows the

lightweight design of 𝜋0*.6 but incurs the highest training cost of 6 GPU·days and inference latency of 0.32 seconds per frame, primarily due to its SigLIP (Zhai et al., 2023) visual encoder. The video-based variant predicting value alone achieves the fastest inference at 0.11 seconds and the lowest training cost of 3 GPU·days, yet omitting future proprioception compromises prediction accuracy. Our full ViVa model strikes a favorable balance, training in 4 GPU·days, 1.5× faster than the VLM baseline, while running at 0.18 seconds per frame. The additional proprioceptive prediction enriches the learning signal with minimal computational overhead.

Table 12: Computational cost comparison. Training time (GPU·days) and inference time (s) are reported. ViVa achieves faster training and inference than the VLM-based baseline.

Method Backbone Training cost (GPU·days) Inference cost (s / frame)

𝜋0*.6 VLM 6 0.32 ViVa w/o future-state VG 3 0.11

ViVa VG 4 0.18

### 5. Conclusion

In this work, we introduced ViVa, a video-generative value model that grounds value estimation in predicted future dynamics. By jointly predicting future proprioception alongside value, ViVa learns temporally grounded representations that reliably track task progress and detect execution errors. ViVa achieves strong results across all three real-robot tasks, demonstrating the importance of spatiotemporal priors and embodiment-aware prediction for value learning.

Limitation. Due to resource constraints, we have not yet pretrained the value model at scale. We believe this would enable strong cross-task generalization, which we leave for future work.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025. 4
- [2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 2
- [3] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 3
- [4] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. arXiv preprint arXiv:2409.16283, 2024. 3
- [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 3
- [6] Duygu Ceylan, Chun-Hao P Huang, and Niloy J Mitra. Pix2video: Video editing using image diffusion. In Proceedings of the IEEE/CVF international conference on computer vision, pages 23206–23217, 2023. 3
- [7] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024. 3
- [8] Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455–14465, 2024. 2
- [9] Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J Ratliff, Jiafei Duan, Dieter Fox, and Ranjay Krishna. Topreward: Token probabilities as hidden zero-shot rewards for robotics. arXiv preprint arXiv:2602.19313, 2026. 3, 8
- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 2
- [11] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. arXiv preprint arXiv:2310.10625,

2023. 3

- [12] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023. 3
- [13] Kevin Frans, Seohong Park, Pieter Abbeel, and Sergey Levine. Diffusion guidance is a controllable policy improvement operator. arXiv preprint arXiv:2505.23458, 2025. 3
- [14] Seyed Kamyar Seyed Ghasemipour, Ayzaan Wahid, Jonathan Tompson, Pannag Sanketi, and Igor Mordatch. Self-improving embodied foundation models. arXiv preprint arXiv:2509.15155, 2025. 3

- [15] Dongchi Huang, Zhirui Fang, Tianle Zhang, Yihang Li, Lin Zhao, and Chunhe Xia. Co-rft: Efficient fine-tuning of vision-language-action models through chunked offline reinforcement learning. arXiv preprint arXiv:2508.02219, 2025. 3
- [16] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022. 1
- [17] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace

Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. 𝜋0*.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025. 2, 3, 7, 8, 10

- [18] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan

Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. 𝜋0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025. 1, 10

- [19] Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al. Scalable deep reinforcement learning for vision-based robotic manipulation. In Conference on robot learning, pages 651–673. PMLR, 2018. 3
- [20] Simar Kareer, Karl Pertsch, James Darpinian, Judy Hoffman, Danfei Xu, Sergey Levine, Chelsea Finn, and Suraj Nair. Emergence of human to robot transfer in vision-language-action models. arXiv preprint arXiv:2512.22414, 2025. 3
- [21] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 1, 3
- [22] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 3
- [23] Thomas Lampe, Abbas Abdolmaleki, Sarah Bechtle, Sandy H Huang, Jost Tobias Springenberg, Michael Bloesch, Oliver Groth, Roland Hafner, Tim Hertweck, Michael Neunert, et al. Mastering stacking of diverse shapes with large-scale iterative reinforcement learning on real robots. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 7772–7779. IEEE, 2024. 3
- [24] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020. 3
- [25] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 2
- [26] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024. 3
- [27] Wei Li, Renshan Zhang, Rui Shao, Jie He, and Liqiang Nie. Cogvla: Cognition-aligned vision-languageaction model via instruction-driven routing & sparsification. arXiv preprint arXiv:2508.21046, 2025. 1
- [28] Xinghang Li, Peiyan Li, Long Qian, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Xinlong Wang, Di Guo, et al. What matters in building vision–language–action models for generalist robots. Nature Machine Intelligence, pages 1–15, 2026. 1

- [29] Junbang Liang, Pavel Tokmakov, Ruoshi Liu, Sruthi Sudhakar, Paarth Shah, Rares Ambrus, and Carl Vondrick. Video generators are robot policies. arXiv preprint arXiv:2508.00795, 2025. 4
- [30] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024. 3
- [31] Jianlan Luo, Zheyuan Hu, Charles Xu, You Liang Tan, Jacob Berg, Archit Sharma, Stefan Schaal, Chelsea Finn, Abhishek Gupta, and Sergey Levine. Serl: A software suite for sample-efficient robotic reinforcement learning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 16961–16969. IEEE, 2024. 3
- [32] Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. Liv: Languageimage representations and rewards for robotic control. In International Conference on Machine Learning, pages 23301–23320. PMLR, 2023. 3
- [33] Yecheng Jason Ma, Joey Hejna, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny Driess, Ted Xiao, et al. Vision language models are in-context value learners. In The Thirteenth International Conference on Learning Representations, 2024. 2, 3
- [34] Yecheng Jason Ma, Joey Hejna, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny Driess, Ted Xiao, et al. Vision language models are in-context value learners. In International Conference on Learning Representations, volume 2025, pages 33984–34009, 2025. 8
- [35] Ajay Mandlekar, Fabio Ramos, Byron Boots, Silvio Savarese, Li Fei-Fei, Animesh Garg, and Dieter Fox. Iris: Implicit reinforcement without interaction at scale for learning control from offline robot manipulation data. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 4414–4420. IEEE,

2020. 3

- [36] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, et al. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025. 2
- [37] Russell Mendonca, Shikhar Bahl, and Deepak Pathak. Alan: Autonomously exploring robotic agents in the real world. arXiv preprint arXiv:2302.06604, 2023. 3
- [38] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024. 3
- [39] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 3
- [40] Qiaosong Qi, Le Zhuo, Aixi Zhang, Yue Liao, Fei Fang, Si Liu, and Shuicheng Yan. Diffdance: Cascaded human motion diffusion model for dance generation. In Proceedings of the 31st ACM International Conference on Multimedia, pages 1374–1382, 2023. 3
- [41] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627–635. JMLR Workshop and Conference Proceedings, 2011. 3
- [42] Ranjan Sapkota, Yang Cao, Konstantinos I Roumeliotis, and Manoj Karkee. Vision-language-action (vla) models: Concepts, progress, applications and challenges. arXiv preprint arXiv:2505.04769, 2025. 1

- [43] Archit Sharma, Ahmed M Ahmed, Rehaan Ahmad, and Chelsea Finn. Self-improving robots: End-to-end autonomous visuomotor reinforcement learning. arXiv preprint arXiv:2303.01488, 2023. 3
- [44] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022. 3
- [45] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998. 2, 3
- [46] GigaBrain Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jie Li, Jiagang Zhu, Lv Feng, et al. Gigabrain-0: A world model-powered vision-language-action model. arXiv preprint arXiv:2510.19430, 2025. 10
- [47] GigaBrain Team, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, Lv Feng, et al. Gigabrain-0.5 m*: a vla that learns from world model-based reinforcement learning. arXiv preprint arXiv:2602.12099, 2026. 1
- [48] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. 3
- [49] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 3
- [50] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 4, 5
- [51] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. arXiv preprint arXiv:2312.13139, 2023. 3
- [52] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. arXiv preprint arXiv:2310.06114, 1(2):6, 2023. 3
- [53] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 3
- [54] Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. Gigaworld-policy: An efficient action-centered world–action model. arXiv preprint arXiv:2603.17240, 2026. 3
- [55] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. 3
- [56] Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, and Jiangmiao Pang. A vision-language-action-critic model for robotic real-world reinforcement learning. arXiv preprint arXiv:2509.15937, 2025. 3
- [57] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986,

2023. 13

- [58] Hongyin Zhang, Pengxiang Ding, Shangke Lyu, Ying Peng, and Donglin Wang. Gevrm: Goal-expressive video generation model for robust visual manipulation. arXiv preprint arXiv:2502.09268, 2025. 3
- [59] Hongxiang Zhao, Xingchen Liu, Mutian Xu, Yiming Hao, Weikai Chen, and Xiaoguang Han. Tasterob: Advancing video generation of task-oriented hand-object interaction for generalizable robotic manipulation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 27683– 27693, 2025. 3
- [60] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 3
- [61] Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, and Chuang Gan. Robodreamer: Learning compositional world models for robot imagination. arXiv preprint arXiv:2404.12377, 2024. 3
- [62] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 2
- [63] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 1

