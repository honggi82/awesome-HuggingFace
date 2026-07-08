## arXiv:2605.26105v1[cs.CV]25May2026

[Figure 1]

[Figure 2]

[Figure 3]

2026-05-26

# On-Policy Adversarial Flow Distillation for Autoregressive Video Generation

###### Yang Luo1, Shengju Qian2†, Xiaohang Tang3, Zirui Zhu1, Yong Liu1, Xin Wang2, Yang You1

1National Univesity of Singapore 2LIGHTSPEED 3University College London yang_luo@u.nus.edu † Corresponding Author

Abstract Autoregressive video generators are attractive for streaming, long-horizon, and interactive applications, but distilling strong black-box teachers into causal students remains difficult. The student must learn under its own rollout distribution, whereas practical teachers may expose only promptconditioned completed videos and may differ in architecture, capacity, temporal design, and sampling schedule. This interface makes supervised fine-tuning off-policy, score-based distillation inapplicable, and direct adversarial imitation too sparse for denoising-time credit assignment. We propose Adversarial Flow Distillation (AFD), an on-policy framework for heterogeneous black-box video distillation. AFD queries the teacher and rolls out the current student on the same prompts, trains a prompt-paired Bradley–Terry discriminator to estimate clean-sample teacher–student discrepancy, and converts the resulting on-policy advantage into forward-process flow-matching updates on the student’s own noised states. Thus, AFD provides dense velocity-field supervision while requiring no teacher scores, latents, denoising trajectories, step alignment, or reverse-chain reinforcement learning. Experiments across two causal AR student families show that AFD consistently improves motion- and physics-sensitive generation while preserving general video quality, and ablations validate the importance of adaptive on-policy feedback and forward-process credit assignment. The method requires only clean teacher videos and student rollouts, providing a practical route for distilling proprietary or heterogeneous video generators into efficient autoregressive students.

### 1. Introduction

Efficient deployment of modern video generators increasingly depends on distilling large diffusion, score-based, or flow-matching models into smaller students [Chern et al., 2025, Ho et al., 2020, Kim et al., 2025, Lipman et al., 2023, Liu et al., 2022, Salimans and Ho, 2022, Song et al., 2021, 2023, Yin et al., 2024]. In practice, many strong video teachers are accessible only as black-box samplers that return completed clips, without exposing scores, logits, latents, sampler states, or denoising trajectories [Brooks et al., 2024, Kling Team, 2025, Polyak et al., 2024]. This is especially restrictive for autoregressive (AR) students such as Self-Forcing [Huang et al., 2025]: a black-box teacher may use a different architecture, temporal conditioning scheme, and long denoising schedule, while the deployable AR student generates causally with few denoising steps. Existing distillation recipes therefore lack an aligned supervision channel from the teacher’s hidden generation process to the student’s rollout distribution and intermediate noised states.

A straightforward adaptation strategy is supervised fine-tuning (SFT) on teacher-generated videos. Although SFT does not require teacher scores, it trains the AR student under teacher-induced prefixes rather than under the student’s own rollout distribution. At inference time, the student conditions on its previously generated frames; local errors can therefore shift future denoising states and accumulate over the video horizon [Bengio et al., 2015]. A preliminary SFT sweep in Figure 1 confirms this mismatch: longer off-policy training fails to consistently improve either VBench [Huang et al., 2024] or VideoPhy-2 [Bansal et al., 2026].

The SFT mismatch motivates on-policy adaptation, yet existing on-policy objectives still assume

supervision unavailable in black-box video distillation. DMD-style objectives require teacher scores, density ratios, or compatible noised states [Yin et al., 2024], none of which are available from a sampling-only video teacher. Reward- and preference-based diffusion alignment can optimize models from sample-level feedback [Black et al., 2023, Liu et al., 2025a,b, Wallace et al., 2024, Xue et al., 2025], but a scalar score on a completed video does not identify which frames, motion patterns, or denoising states account for the teacher–student discrepancy. The useful signal is observable only at completed videos, while the object being trained is a time-dependent vector field evaluated on the student’s own noised AR states. The key challenge is therefore not merely to assign a reward to a video, but to lift black-box video evidence into dense flow-matching supervision for the student’s causal denoising process.

We propose Adversarial Flow Distillation (AFD), an on-policy distillation framework for autoregressive video generation under heterogeneous black-box teacher access. AFD factorizes the problem into clean-sample distribution-ratio estimation and forward-process vector-field regression. For each prompt batch, it queries the teacher for completed videos, rolls out the current student under autoregressive selfconditioning, and trains a prompt-conditioned discriminator on teacher samples versus current student samples. The discriminator produces an on-policy advantage score, normalized against the current batch or prompt distribution, which is identifiable from black-box samples and coevolves with the student.

[Figure 4]

Figure 1 | Preliminary SFT trend for an AR video student. VBench and VideoPhy-2 scores decrease rather than improve monotonically as SFT steps increase.

AFD then uses this advantage inside a reward-weighted flow-matching objective rather than a reversetrajectory policy-gradient objective. Student rollouts are forward-noised by the student’s own schedule, producing the same kind of intermediate states on which its velocity field is trained. Within each on-policy batch, high-advantage rollouts form positive examples and low-advantage rollouts form negative examples; the student is then trained to move its vector field toward the forward velocity of higher-scoring rollouts and away from lower-scoring rollouts, with the discriminator advantage controlling the strength of this contrastive correction. This turns a black-box video-level signal into denoising-time updates on the student’s own noised states. Consequently, AFD requires no teacher scores, no teacher–student step alignment, and no stored reverse trajectories, while still providing dense supervision beyond video-level adversarial training. Our contributions are as follows.

- • We identify black-box heterogeneous on-policy distillation as a core obstacle for autoregressive video students, and show that off-policy SFT, score-based DMD, and direct video-level adversarial training are mismatched to the limited teacher interface.
- • We introduce Adversarial Flow Distillation (AFD), a score-free distillation framework that estimates teacher–student discrepancy from completed videos and converts it into dense forward-process flow-matching updates on the student’s own noised rollouts.
- • We evaluate AFD on two causal autoregressive video backbones, showing consistent gains on motion- and physics-sensitive metrics under black-box teacher access, together with ablations on domain adaptation and discriminator design.

### 2. Related Work

Video diffusion and flow models. DDPM [Ho et al., 2020] and score-based SDEs [Song et al., 2021] established denoising-time generation, while Flow Matching [Lipman et al., 2023] and Rectified Flow [Liu et al., 2022] recast generation as continuous-time vector-field learning. DiT [Peebles and Xie, 2023] improved transformer diffusion scalability and now underlies video systems including Video Diffusion Models [Ho et al., 2022], Lumiere [Bar-Tal et al., 2024], CogVideoX [Yang et al., 2024], Movie Gen [Polyak et al., 2024], Sora [Brooks et al., 2024], Kling-Omni [Kling Team, 2025], and Wan [Team Wan et al., 2025]. Recent video distillation work such as V.I.P. [Kim et al., 2025] and LiveTalk [Chern et al., 2025] studies online or on-policy recipes for efficient video generation. AFD focuses on a different transfer interface: an AR student learns on its own rollouts while a sampling-only teacher provides only completed videos.

Black-box on-policy distillation. On-policy distillation [Lu and Thinking Machines Lab, 2025, Ye et al., 2025] keeps training on the student’s own trajectory distribution while using a teacher for supervision. In language models, this often means teacher feedback on student-generated prefixes, and Rethinking OPD [Li et al., 2026b] analyzes when such feedback succeeds or fails. VLA-OPD [Zhong et al., 2026] and Video-OPD [Li et al., 2026a] apply the same on-policy idea to action and temporal grounding tasks. Our setting differs from LLM OPD because the teacher cannot provide token-level logits or local probability ratios: a sampling-only video teacher returns completed clips, while the student must learn a continuous-time video flow. AFD therefore estimates teacher–student distributional discrepancy with a discriminator and projects that signal to denoising-time states with DiffusionNFT.

Adversarial and preference-guided alignment. DDPO [Black et al., 2023], Diffusion-DPO [Wallace

- et al., 2024], and VideoAlign [Liu et al., 2025b] show that learned or human feedback can guide diffusion models, but reverse-trajectory policy gradients are expensive for video. DiffusionNFT [Zheng
- et al., 2025] instead optimizes on the forward process from clean generated samples, avoiding likelihood estimation and reverse-trajectory storage; Astrolabe [Zhang et al., 2026] adapts this view to distilled AR video alignment. Continuous Adversarial Flow Models [Lin et al., 2026] further suggest that learned criteria can improve finite-capacity flow post-training. AFD uses these ideas for teacher distillation rather than generic reward maximization: feedback comes from a co-evolving teacher–student discriminator evaluated on on-policy student videos.

### 3. Method

Let 𝜋𝑇(𝑥0|𝑦) denote a black-box teacher that returns a video for prompt 𝑦, and let 𝜋𝜃 denote a causal autoregressive student flow model with velocity field 𝑓𝜃(𝑥𝑡, 𝑡, 𝑦). The teacher may differ from the student in architecture, capacity, latent representation, and sampling schedule. We assume no access to teacher parameters, scores, latents, or denoising trajectories; the only shared interface is the completed prompt-conditioned video. At each iteration, prompts are sampled from a distribution 𝑦 ∼ Y, the teacher returns 𝑥0𝑇 ∼ 𝜋𝑇(·|𝑦), and the current student produces an on-policy rollout 𝑥ˆ0 ∼ 𝜋𝜃(·|𝑦) under autoregressive self-conditioning. As shown in Figure 2, AFD consists of an adaptive video discriminator that estimates teacher–student distributional discrepancy on student rollouts and a DiffusionNFT update that transfers this sample-level signal to the student’s forward noising process.

###### 3.1. Adaptive Teacher–Student Discrimination

We train a prompt-conditioned spatiotemporal discriminator 𝐷𝜙(𝑥0, 𝑦) that scores how likely a video 𝑥0 is to be a teacher sample given prompt 𝑦. For each prompt, we treat the teacher sample as preferred

###### 1 Prompt Sampling 2 Adaptive Feedback 3 Diffusion-native On-policy Update

Forward-Process Update with DiffusionNFT

Black-box Teacher

###### Adaptive TeacherStudent Discrimination

Forward Noising (from student rollout)

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

sampling-only API

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

teacher videos

[Figure 19]

Linear Head

VideoAlign

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

###### Prompt Batch

[Figure 25]

[Figure 26]

Sampled from

LoRA

[Figure 27]

broadcast terminal discriminator feedback to denoising time / forward-process states

[Figure 28]

[Figure 29]

AR Student Flow

Contrastive Flow Matching

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Teacher-dependent Advantage

[Figure 36]

autoregressive self-conditioning

[Figure 37]

[Figure 38]

student rollouts (on-policy)

[Figure 39]

[Figure 40]

###### Post-trained Student

[Figure 41]

[Figure 42]

Student Objective

[Figure 43]

updated autoregressive flow model

[Figure 44]

- Figure 2 | Overview of AFD for black-box on-policy video distillation. A sampling-only teacher provides prompt-conditioned completed videos, while the current AR student provides on-policy rollouts. A coevolving discriminator estimates teacher–student distributional discrepancy from completed samples, and DiffusionNFT converts this black-box sample-level signal into dense forward-process flow updates on the student’s own noised states.

over the current student rollout, yielding the Bradley–Terry (BT) loss:

0,𝑥ˆ0,𝑦) log𝜎 𝐷𝜙(𝑥0𝑇, 𝑦) − 𝐷𝜙(𝑥ˆ0, 𝑦) , (1) where 𝜎(·) is the logistic function. This pairwise objective is standard in preference modeling: it increases the discriminator margin between teacher samples and current student samples under the same prompt, without requiring calibrated absolute rewards. In practice, 𝐷𝜙 can be initialized from a video preference or text-video alignment model such as VideoAlign [Liu et al., 2025b] and adapted with LoRA. Rather than defining a fixed global reward, the discriminator estimates the current discrepancy between teacher and student samples under the shared prompt distribution, providing on-the-fly feedback for on-policy rollouts when teacher scores are unavailable.

###### L𝐷(𝜙) := −𝔼(𝑥𝑇

The discriminator induces a discrepancy signal, which we use as an adaptive reward function: 𝑟𝜙(𝑥ˆ0, 𝑦) = sg 𝐷𝜙(𝑥ˆ0, 𝑦) − 𝑏(𝑦) , (2)

where sg(·) denotes stop-gradient and 𝑏(𝑦) is a batch or prompt-level baseline. Unlike a fixed reward model, this signal co-evolves with the student model and measures whether current student videos remain distinguishable from teacher videos under the current prompt distribution.

###### 3.2. Diffusion-Native On-Policy Update

The discriminator signal is sample-level because it scores completed videos. Directly optimizing this signal with reverse-trajectory policy gradients would require treating the reverse denoising chain as an RL trajectory, which is costly for long, high-resolution videos. We instead adopt a forward-process diffusion optimization method that optimizes diffusion models using only clean samples and the

forward noising process [Choi et al., 2026], namely DiffusionNFT [Zheng et al., 2025]. For a student rollout 𝑥ˆ0, timestep 𝑡 ∈ [0, 1], and noise 𝜖 ∼ N(0, 𝐼), define the forward-noised sample and its corresponding flow-matching target as

𝑥𝑡 = 𝛼𝑡𝑥ˆ0 + 𝜎𝑡𝜖, 𝑣 = 𝛼𝑡𝑥ˆ0 + 𝜎𝑡𝜖. (3)

We use the discriminator to score on-policy rollouts with 𝜋𝜃. We then determine the weights for positive and negative policy optimization. Concretely, for a minibatch of student videos {𝑥ˆ0,𝑖}𝑖𝐵=1, let 𝐴𝑖 = 𝑟𝜙(𝑥ˆ0,𝑖, 𝑦𝑖) and normalize it to a weight 𝑤𝑖 ∈ [0, 1] within the batch. We apply the forward process on this batch of videos following Equation 3 to obtain noised samples {𝑥𝑡,𝑖}𝑖𝐵=1. Denoting the parametrized velocity as 𝑣𝜃(𝑥𝑡,𝑖, 𝑡, 𝑦𝑖), we define two operators for positive and negative policy optimization under the on-policy sampling setting:

𝑣𝜃+ := (1 − 𝛽) sg(𝑣𝜃) + 𝛽𝑣𝜃, 𝑣𝜃− := (1 + 𝛽) sg(𝑣𝜃) − 𝛽𝑣𝜃. the negative-aware fine-tuning (NFT) loss whose optimum implicitly drives 𝑣𝜃 towards 𝑣+ is

###### ∑︁ 𝐵

𝑤𝑖 · ∥𝑣𝜃+(𝑥𝑡,𝑖, 𝑡, 𝑦𝑖) − 𝑣∥2 + (1 − 𝑤𝑖) · ∥𝑣𝜃−(𝑥𝑡,𝑖, 𝑡, 𝑦𝑖) − 𝑣∥2 , (4)

LNFT(𝜃) = 𝔼𝑡,𝑥

𝑡,𝑖

𝑖=1

Additionally, we include a regularization term in the objective function to avoid catastrophic forgetting Lprior = 𝔼𝑡,𝑥

𝐵 𝑖=1 𝑤𝑖∥𝑣𝜃(𝑥𝑡,𝑖, 𝑡, 𝑦𝑖) − 𝑣ref(𝑥𝑡,𝑖, 𝑡, 𝑦𝑖)∥2, where the reference model 𝑣ref is the initial pre-

𝑡,𝑖

trained diffusion model. The objective of our method AFD is finally defined as:

LAFD(𝜃) := LNFT(𝜃) + 𝜆priorLprior(𝜃). (5)

###### 3.3. Black-Box Teacher Interface

AFD is motivated by the measurability constraint imposed by a black-box video teacher. The teacher exposes only the clean-sample channel

O𝑇 : 𝑦 ↦→ 𝑥0𝑇 ∼ 𝜋𝑇(·|𝑦), (6)

not ∇𝑥 log 𝑝𝑇(𝑥𝑡|𝑦), teacher latents, teacher reverse states, or a teacher transition kernel. Any admissible distillation signal must therefore be a function of completed teacher videos and completed student videos. This constraint is restrictive for an AR flow student, because the trained object is not a video classifier but a vector field 𝑓𝜃(𝑥𝑡, 𝑡, ℎ𝑘, 𝑦) evaluated at noised states along student-induced histories ℎ𝑘 = 𝑥0(<𝑘).

Let 𝑥0 = [𝑥0(1), . . . , 𝑥0(𝐾)] denote a sequence of video blocks. The AR student induces

𝐾

𝜋𝜃 𝑥0(𝑘)|𝑥0(<𝑘), 𝑦 , (7)

𝜋𝜃(𝑥0|𝑦) =

𝑘=1

so student errors alter the future conditioning distribution. A bidirectional black-box teacher, however, is observed only through 𝜋𝑇(𝑥0|𝑦). Its hidden sampler may contain 𝑀𝑇 denoising evaluations, whereas the student may use 𝑀𝑆 ≪ 𝑀𝑇. Without teacher trajectories or a shared latent space, there is no observable alignment operator

A𝑚,𝑠 : 𝑧𝑇𝑚 ↦→ (ℎ𝑘, 𝑥𝑠𝑆) (8) from a teacher step 𝑚 to a student AR denoising state 𝑠. The absence of such an operator clarifies the limitations of two common baselines. SFT trains under teacher-induced prefixes rather than the

student’s rollout distribution, so exposure bias accumulates along the video horizon [Bengio et al., 2015]. DMD-style objectives require either a teacher score or a teacher density ratio at the student’s noised state,

𝜋𝑇(𝑥ˆ0|𝑦) 𝜋𝜃(𝑥ˆ0|𝑦)

∇𝑥 log 𝑝𝑇(𝑥𝑡|𝑦) or log

. (9)

The score term is unobservable, and the clean-sample ratio can only be estimated from samples at 𝑥0. The relevant objective is therefore not to reconstruct the teacher’s hidden diffusion path, but to transfer completed-video evidence to the student’s own noised states. DiffusionNFT matches this interface: it requires only clean samples from the current policy, a scalar score on those samples, and the student’s known forward noising kernel.

This differs from GRPO-style diffusion RL. Methods such as Flow-GRPO [Liu et al., 2025a] and DanceGRPO [Xue et al., 2025] make diffusion or flow policies amenable to online RL by casting the reverse denoising process as an MDP, introducing stochastic reverse trajectories, and optimizing group-relative policy objectives on sampled rollouts. This interface is well suited to reward-model alignment, where the optimized model’s reverse sampler defines the environment. It is less suitable for black-box teacher distillation: the teacher provides no reverse actions, no teacher trajectory probabilities, and no step-level supervision on the student’s reverse chain. Applying a reverse-MDP objective therefore introduces a synthetic credit-assignment layer that is not supported by the teacher interface. AFD instead keeps the teacher-dependent signal at the clean-sample level, where it is identifiable, and uses the forward noising kernel to induce the dense denoising-time structure.

###### 3.4. Forward-Process Credit Assignment

AFD follows this design by separating teacher evidence extraction from denoising-time credit assignment. First, the discriminator estimates the density-ratio information identifiable from black-box clean samples. The optimal discriminator that minimizes the BT loss satisfies

𝜙(𝑥ˆ0, 𝑦) 1 − 𝐷∗

𝐷∗

𝜋𝑇(𝑥ˆ0|𝑦) 𝜋𝜃(𝑥ˆ0|𝑦)

𝜌∗𝜙(𝑥ˆ0, 𝑦) := log

= log

. (10)

𝜙(𝑥ˆ0, 𝑦)

This ratio is defined on completed videos sampled from the current student distribution; it requires no teacher score, teacher timestep, or architectural alignment.

Second, AFD converts this clean-sample ratio into a vector-field target by applying the student’s forward noising map. Let F𝑡 denote this teacher-agnostic forward-noising operator. The key compatibility condition is that F𝑡 is entirely student-side: it depends on the student’s noising schedule and generated clean video, not on any teacher state. A generic sample-level advantage 𝑟(𝑥ˆ0, 𝑦) defines the tilted clean-video law following DiffusionNFT:

𝜋+(𝑥ˆ0 | 𝑦) ∝ 𝑟(𝑥ˆ0, 𝑦)𝜋𝜃(𝑥ˆ0 | 𝑦). (11)

Applying the student’s forward noising kernel 𝑝(𝑥𝑡|𝑥ˆ0) following Eq. 3 to this tilted clean distribution, it can induce a corresponding distribution over noisy states according to Bayes’ Rule:

𝜃(·|𝑦) [𝑟(𝑥ˆ0, 𝑦)𝑝(𝑥𝑡|𝑥ˆ0)] . (12)

𝜋+𝑡 (𝑥𝑡|𝑦) ∝ 𝔼𝑥ˆ0∼𝜋

Then the optimal positive flow-matching vector field for these noised marginals is the conditional average forward velocity,

𝑣+(𝑥𝑡, 𝑡, 𝑦) = 𝔼𝑥ˆ0∼𝜋+(·|𝑦), 𝑝(𝑥𝑡|𝑥ˆ0) [𝑣 | 𝑥𝑡, 𝑦] . (13) where 𝑣 is the forward-path velocity in Eq. 3. Thus, a sample-level reward over completed videos induces a dense denoising-time vector-field target after being propagated through the student’s

forward process. This identity provides the forward-noising bridge: sample-level preferences define a denoising-time vector field after being propagated through F𝑡. The black-box restriction and AFD therefore operate on compatible information structures: all teacher-dependent information is measurable from completed videos, while denoising-time structure is induced by the student’s forward process. AFD converts completed-video teacher evidence into dense corrections on the student’s own noised states without reconstructing the teacher’s hidden trajectory or storing the student’s reverse trajectory. Appendix 10 provides further theoretical derivation.

###### 3.5. Training Procedure

The optimization alternates between on-policy data collection, discriminator updates, and student velocity-field updates, as summarized in Algorithm 1. We implement the prior regularizer Lprior(𝜃) as a velocity-regression penalty against a frozen reference student to preserve the student’s base capabilities.

Algorithm 1 Adversarial Flow Distillation (AFD) Input: Teacher API 𝜋𝑇, Student Flow Model 𝑓𝜃, Discriminator 𝐷𝜙, Prompt distribution Y Output: Post-trained Student Model 𝑓𝜃 Initialize student parameters 𝜃 and EMA parameters 𝜃¯ ← 𝜃, and keep 𝑓ref frozen Initialize discrimi-

nator parameters 𝜙 (e.g., via LoRA on VideoAlign) while not converged do

- // 1. On-Policy Data Collection Sample batch of prompts 𝑦 ∼ Y

Query teacher for target videos: 𝑥0𝑇 ∼ 𝜋𝑇(·|𝑦) Sample student rollouts: 𝑥ˆ0 ∼ 𝜋𝜃(·|𝑦)

- // 2. Adaptive Teacher-Student Discrimination

Compute discriminator loss L𝐷(𝜙) comparing 𝑥0𝑇 and 𝑥ˆ0 Update discriminator 𝜙 ← 𝜙 − 𝜂𝐷∇𝜙L𝐷(𝜙) Compute baseline 𝑏(𝑦) (e.g., batch mean of 𝐷𝜙(𝑥ˆ0, 𝑦)) Compute advantage scores 𝑟𝜙(𝑥ˆ0, 𝑦) = sg(𝐷𝜙(𝑥ˆ0, 𝑦) − 𝑏(𝑦))

- // 3. Diffusion-Native Student Update Sample denoising timesteps 𝑡 ∼ U[0, 1] and noise 𝜖 ∼ N(0, 𝐼) Construct forward-noised student states 𝑥𝑡 = 𝛼𝑡𝑥ˆ0 + 𝜎𝑡𝜖 Form positive/negative rollout sets from

𝑟𝜙 and evaluate LNFT(𝜃) using 𝑥𝑡 Evaluate prior regularization Lprior(𝜃) against reference student 𝑓ref Update student 𝜃 ← 𝜃 − 𝜂𝜃∇𝜃 LNFT + 𝜆priorLprior

- // 4. EMA Update Update target network 𝜃¯ ← 𝛽𝜃¯ + (1 − 𝛽)𝜃

### 4. Experiments

###### 4.1. Setup

Experimental setup. We evaluate AFD on two causal autoregressive student families, Self-Forcing and Causal-Forcing. The teacher is Seedance 2.0 [ByteDance Seed, 2026], accessed only as a promptconditioned video sampling API. The main experiment is a continual adaptation setting: a pretrained AR student is adapted to a physics-oriented target domain while preserving general video quality. We sample 200 examples from the VideoPhy-2 physics benchmark and query the teacher on the

same prompts to obtain black-box adaptation videos. Unless otherwise specified, discriminators are initialized from VideoAlign [Liu et al., 2025b], adapted with LoRA, and updated online against the current student. We report VBench [Huang et al., 2024] dimensions grouped into Physics and General categories. Physics includes temporal flickering, motion smoothness, dynamic degree, human action, and spatial relationship; General is the mean over the remaining VBench dimensions. We also evaluate target-domain adaptation with VideoAlign Motion Quality (VideoAlign-MQ) [Liu et al., 2025b] and VideoPhy-2 Physical Consistency (VideoPhy-2-PC) [Bansal et al., 2026]. Full hyperparameters are provided in Appendix 7.

Baselines. We consider four baselines:

- • Base: the pretrained AR student before teacher adaptation.
- • SFT: supervised fine-tuning on teacher-generated videos without on-policy student rollouts.
- • GAN: adversarial video-level training with the teacher–student discriminator, excluding the forward-process policy update.
- • Score-free DMD: a DMD-style training scaffold with the score-based distribution-matching term removed, isolating sample-level supervision under the same black-box access constraints.

###### 4.2. Main Results

Tables 1 and 2 show that AFD improves physics-sensitive generation under limited target-domain data while preserving general video generation capability. On Self-Forcing, AFD reaches the best Physics VBench Total (87.55), VideoAlign-MQ (0.605), and VideoPhy-2-PC (4.20), with a General Total close to the best baseline (60.83 vs. 60.95). On Causal-Forcing, AFD gives the best Physics VBench Total (88.52) and VideoPhy-2-PC (4.24), ties the best VideoAlign-MQ (0.661), and keeps the General Total close to the highest score (59.83 vs. 60.32). Compared with the adapted baselines, AFD gives the strongest dynamic degree on both student families, indicating better motion and physical behavior without a large drop in general quality.

- Table 1 | Main VBench results after black-box continual adaptation. Physics reports all dimensions and their average; General reports the group average.

Physics (VBench) General (VBench)

Model Method

Temporal

Motion

Dynamic

Human

Spatial

Total↑

Rel. ↑ Total↑

Flick. ↑

Smooth.↑

Degree ↑

Action ↑

Base 68.49 98.27 97.19 91.67 46.00 9.32 36.51 SFT 58.69 96.64 97.40 44.44 34.00 20.97 37.62 GAN 83.83 99.28 98.58 61.11 79.00 81.18 60.95 DMD 81.88 99.01 98.95 52.78 81.00 77.66 59.98

Self-Forcing

- AFD 87.55 98.78 98.30 79.17 80.00 81.50 60.83

Causal-Forcing

Base 86.76 98.20 97.90 87.50 76.00 74.20 59.03 SFT 76.24 99.31 99.05 29.17 78.00 75.69 59.40 GAN 83.07 98.58 98.08 66.67 76.00 76.00 59.44 DMD 82.52 99.16 98.75 58.33 77.00 79.35 60.32

- AFD 88.52 98.41 97.89 88.89 80.00 77.39 59.83

- Figure 3 supports the trends in Tables 1 and 2. Across both AR student families, AFD keeps the main prompt content and visual quality while improving motion-related behavior. This matches the method design: the discriminator scores completed on-policy rollouts, and the DiffusionNFT update transfers this signal to the student’s forward-noised states. The examples therefore support the quantitative gains in dynamic degree, motion quality, and physical consistency. Additional examples are provided

- Table 2 | Physics evaluations after small-sample physics-domain adaptation.

Motion Quality↑

Physical Consistency↑

Model Method

Base 0.341 4.04 SFT 0.296 3.72 GAN 0.541 4.10 DMD 0.420 3.72 AFD 0.605 4.20

Self-Forcing

Base 0.520 4.16 SFT 0.499 4.17 GAN 0.582 4.16 DMD 0.661 4.14 AFD 0.661 4.24

Causal-Forcing

Self-Forcing

Self-Forcing

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

+GAN

+GAN

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

+AFD

+AFD

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Causal-Forcing

Causal-Forcing

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

+GAN

+GAN

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

+AFD

+AFD

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 3 | Qualitative visualizations of AFD in the main experiment. The first row shows Self-Forcing examples, and the second row shows Causal-Forcing examples. Each example shows representative teacher–student comparison frames under the same prompt, illustrating how the on-policy AFD update changes the student’s generated video distribution. Prompts are given in Table 4. in Appendix 8.

###### 4.3. Ablations

Continual learning on a stylized domain. We further test AFD on a shifted visual domain using 200 Disney-style animation prompts. Figure 4 shows that the adapted student follows the new style while preserving prompt alignment and coherent motion. This suggests that AFD is not limited to the physics prompt distribution used in the main experiment. It supports continual learning by using the teacher–student discriminator to provide feedback on the current target domain and the forward-process update to transfer this feedback to the student’s denoising states, without teacher scores or architectural alignment.

Discriminator learning rate. Because AFD’s supervision is generated by a co-evolving discriminator, the discriminator update rate 𝜂𝐷 directly controls how informative the student’s reward signal is over training. We sweep 𝜂𝐷 ∈ {0, 1×10−6, 5×10−6, 1×10−5, 5×10−5} with all other hyperparameters

Self-Forcing

Self-Forcing

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

+AFD

###### +AFD

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

- Figure 4 | Qualitative examples of AFD on a Disney-style animation dataset for continual learning. Detailed prompts are provided in Table 5. fixed, and track the discriminator reward 𝑟𝜙 on student rollouts.
- Figure 5 shows that both under-updating and over-updating the discriminator degrade the quality of the rollout reward. When 𝜂𝐷≤1×10−6, the discriminator lags behind the evolving student distribution: 𝑟𝜙 rapidly approaches 1, indicating saturation of a stale scoring function rather than reliable reduction of the teacher–student gap. When 𝜂𝐷 = 5×10−5, the discriminator becomes overly strong early in training; 𝑟𝜙 remains near 0 across the batch, reducing the contrast between positive and negative rollout weights and weakening the student update. Intermediate rates (5×10−6 and 1×10−5) avoid these saturation regimes and produce a gradual increase in 𝑟𝜙, which is consistent with the intended role of 𝐷𝜙 as an adaptive estimator of the current teacher–student discrepancy rather than a fixed reward model.

Discriminator loss. We next replace the Bradley– Terry discriminator objective from Section 3 with a GAN-style binary classification loss while keeping the same on-policy data, discriminator architecture, and forward-process student update. Figure 6 compares the resulting VBench motion and physics dimensions. BT improves dynamic degree by 9.72 points over the GAN loss (88.89 vs. 79.17) and slightly improves motion smoothness (98.89 vs. 98.53), while temporal flickering, human action, and spatial relation remain effectively tied.

1.0

Reward hacking (no adversarial signal)

- lrD = 5×10 5

- lrD = 1×10 5

lrD = 5×10 6

- lrD = 1×10 6

0.8

DiscriminatorReward

lrD = 0 (no disc.)

0.6

0.4

0.2

Suppressed learning (discriminator too strong)

0.0

20 40 60 80 100

Training Step

Figure 5 | Discriminator learning-rate ablation. The rollout reward 𝑟𝜙 is most informative at intermediate 𝜂𝐷; overly small or large update rates lead to reward hacking or suppressed learning.

This pattern matches the role of the discriminator in AFD. The student update does not need a globally calibrated real/fake probability; it needs a reliable on-policy ranking signal for forming positive and negative rollouts in the forward-process objective. The BT loss compares teacher and student videos under the same prompt and optimizes the margin 𝐷𝜙(𝑥0𝑇, 𝑦) − 𝐷𝜙(𝑥ˆ0, 𝑦), so prompt difficulty, content category, and static appearance biases are largely cancelled inside each pair. A GAN-style loss instead trains an absolute classifier over teacher and student samples, which can be dominated by easy distributional cues and can sat-

###### VBench Comparison: BT Loss vs GAN Loss

| |98.89 98.53 98.41 99.07<br><br>+0.36 0.66<br><br>BT Loss<br><br>GAN Loss| | | | | |
|---|---|---|---|---|---|---|
| |88.89<br><br>79.17 80.00 80.00 79.39 79.53<br><br>+9.72<br><br>0.00 0.14<br><br>| | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

100

VBenchScore(higherisbetter)

80

60

40

20

0

Motion Smoothness Dynamic Degree Temporal Flickering Human Action Spatial Relation

Figure 6 | VBench ablation of the discriminator loss. Replacing the prompt-paired Bradley–Terry objective with a GAN-style binary loss mainly reduces dynamic degree, while the other motion and physics dimensions remain close.

urate when the discriminator becomes too confident. BT therefore provides a smoother relative advantage for AFD, yielding stronger motion improvement while keeping the general video generation capability of the student.

### 5. Conclusion

AFD suggests that black-box video distillation should be formulated around the information interface available to the student, rather than around the teacher’s hidden generation process. For causal AR students, the missing ingredient is not access to the teacher trajectory, but a principled way to connect observable sample-level discrepancies to the student’s own denoising-time states. By separating these two roles into adaptive discrimination and forward-process credit assignment, AFD turns completed-video supervision into an on-policy training signal that improves motion and physical behavior without sacrificing general quality. This perspective offers a practical and extensible route for transferring capabilities from powerful sampling-only video models to efficient autoregressive generators.

### 6. Limitations

Our experiments focus on two causal autoregressive student backbones and a small set of target domains, mainly physics-oriented prompts and stylized animation prompts. Broader studies across more teachers, longer videos, and more diverse prompt distributions would further test the generality of AFD. In addition, AFD uses an online discriminator, so its performance can depend on discriminator update rate and reward calibration; our ablations provide initial guidance, but a more systematic tuning study is left for future work. Evaluation on broader reasoning-oriented benchmarks such as V-ReasonBench [Luo et al., 2025] is left for future work.

### References

Hritik Bansal, Clark Peng, Yonatan Bitton, Roman Goldenberg, Aditya Grover, and Kai-Wei Chang. VideoPhy-2: A challenging action-centric physical commonsense evaluation in video generation. In International Conference on Learning Representations, 2026. URL https://openreview.net/f orum?id=HA8KSQW7SO.

Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. URL https://arxiv.org/abs/2401.129 45.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in Neural Information Processing Systems, 28: 1171–1179, 2015. URL https://proceedings.neurips.cc/paper/2015/hash/e995f98 d56967d946471af29d7bf99f1-Abstract.html.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023. URL https://arxiv.org/ abs/2305.13301.

Tim Brooks, Bill Peebles, et al. Video generation models as world simulators. OpenAI technical report,

##### 2024. URL https://openai.com/research/video-generation-models-as-world-sim ulators.

ByteDance Seed. Seedance 2.0: Advancing video generation for world complexity. arXiv preprint

##### arXiv:2604.14148, 2026. URL https://arxiv.org/abs/2604.14148.

Ethan Chern, Zhulin Hu, Bohao Tang, Jiadi Su, Steffi Chern, Zhijie Deng, and Pengfei Liu. LiveTalk: Real-time multimodal interactive video diffusion via improved on-policy distillation. arXiv preprint arXiv:2512.23576, 2025. URL https://arxiv.org/abs/2512.23576.

Jaemoo Choi, Yuchen Zhu, Wei Guo, Petr Molodyk, Bo Yuan, Jinbin Bai, Yi Xin, Molei Tao, and Yongxin Chen. Rethinking the design space of reinforcement learning for diffusion models: On the importance of likelihood estimation beyond loss design. arXiv preprint arXiv:2602.04663, 2026.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020. URL https: //proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4 b-Abstract.html.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235 c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self Forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. URL https://arxiv.org/abs/2506.08009.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. URL https://vchitect.github.io/VBench-project/.

Jisoo Kim, Wooseok Seo, Junwan Kim, Seungho Park, Sooyeon Park, and Youngjae Yu. V.I.P.: Iterative online preference distillation for efficient video diffusion models. arXiv preprint arXiv:2508.03254,

2025. URL https://arxiv.org/abs/2508.03254. Kling Team. Kling-Omni technical report. arXiv preprint arXiv:2512.16776, 2025. URL https: //arxiv.org/abs/2512.16776.

Jiaze Li, Hao Yin, Haoran Xu, Boshen Xu, Wenhui Tan, Zewen He, Jianzhong Ju, Zhenbo Luo, and Jian Luan. Video-OPD: Efficient post-training of multimodal large language models for temporal video grounding via on-policy distillation. arXiv preprint arXiv:2602.02994, 2026a. URL https: //arxiv.org/abs/2602.02994.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026b. URL https://arxiv.org/abs/2604.13016.

Shanchuan Lin, Ceyuan Yang, Zhijie Lin, Hao Chen, and Haoqi Fan. Continuous adversarial flow models. arXiv preprint arXiv:2604.11521, 2026. URL https://arxiv.org/abs/2604.11521.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. arXiv preprint arXiv:2505.05470, 2025a. URL https://arxiv.org/abs/2505.05470.

Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, Xiaohong Liu, Fei Yang, Pengfei Wan, Di Zhang, Kun Gai, Yujiu Yang, and Wanli Ouyang. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025b. URL https://arxiv.org/abs/2501.13918.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. URL https://arxiv. org/abs/2209.03003.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism,

##### 2025. doi: 10.64434/tml.20251026. URL https://thinkingmachines.ai/blog/on-polic y-distillation.

Yang Luo, Xuanlei Zhao, Baijiong Lin, Lingting Zhu, Liyao Tang, Yuqi Liu, Ying-Cong Chen, Shengju Qian, Xin Wang, and Yang You. V-reasonbench: Toward unified reasoning benchmark suite for video generation models, 2025. URL https://arxiv.org/abs/2511.16668.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings

##### of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Dif fusion_Models_with_Transformers_ICCV_2023_paper.html.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, and Zijian He. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024. URL https://arxiv.org/abs/2410.13720.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/f orum?id=TIdIXIpzhoI.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=Px TIG12RRHS.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252, 2023. URL https://proceedings.mlr. press/v202/song23a.html.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. URL https://arxiv.org/abs/2503.20314.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. URL https://openaccess.thecvf.com/content/CV PR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Op timization_CVPR_2024_paper.html.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. DanceGRPO: Unleashing GRPO on visual generation. arXiv preprint arXiv:2505.07818, 2025. URL https://arxiv.org/abs/2505.07818.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. URL https://arxiv.org/abs/ 2408.06072.

Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-box on-policy distillation of large language models. arXiv preprint arXiv:2511.10643, 2025. URL https: //arxiv.org/abs/2511.10643.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. URL https: //arxiv.org/abs/2311.18828.

Songchun Zhang, Zeyue Xue, Siming Fu, Jie Huang, Xianghao Kong, Yue Ma, Haoyang Huang, Nan Duan, and Anyi Rao. Astrolabe: Steering forward-process reinforcement learning for distilled autoregressive video models. arXiv preprint arXiv:2603.17051, 2026. URL https://arxiv.org/ abs/2603.17051.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DiffusionNFT: Online Diffusion Reinforcement with Forward Process. arXiv preprint arXiv:2509.16117, 2025. URL https://arxiv.org/abs/2509 .16117.

Zhide Zhong, Haodong Yan, Junfeng Li, Junjie He, Tianran Zhang, and Haoang Li. VLA-OPD: Bridging offline SFT and online RL for vision-language-action models via on-policy distillation. arXiv preprint arXiv:2603.26666, 2026. URL https://arxiv.org/abs/2603.26666.

### 7. Hyperparameters

We provide detailed hyperparameters in Table 3. The table lists the model, optimization, diffusion, regularization, and rollout settings used for AFD training.

- Table 3 | Comprehensive hyperparameters for AFD training. We detail the configurations used across model and video specifications, LoRA fine-tuning, optimization, diffusion process, selective regularization, and streaming rollout.

Module Hyperparameter Value

Base Architecture Self-Forcing / Causal-Forcing Video Resolution (𝐻 × 𝑊) 480 × 832

Model & Video Specs

Rank (𝑟) 256 Scaling Factor (𝛼) 256 Dropout Rate 0.0 Gradient Checkpointing Enabled

LoRA Fine-Tuning

Precision Mode bfloat16 Optimizer AdamW (𝛽1 = 0.9, 𝛽2 =

0.999, 𝜖 = 1𝑒 − 8)

Learning Rate (𝜂) 1e-5 Weight Decay 1e-4 Max Gradient Norm 1.0

Optimization

Interpolation Strength (𝛽) 0.1 Prior Loss Weight (𝜆prior) 1e-4 Advantage Clip Max 5.0 Reward Normalization Global Std with Per-Prompt

Selective Regularization

Tracking EMA Decay Rate (𝛾) 0.99

- 8. Additional Qualitative Visualizations

We provide additional AFD visualizations in img 7–9. These examples further illustrate the same pattern observed in the main experiments: AFD improves motion evolution and physical plausibility while preserving the prompt-level semantics and visual fidelity of the base autoregressive video student.

- 9. Prompts for Qualitative Visualizations We present the prompts used for the qualitative examples in img 3 and 4.
- 10. Theoretical Insights

In this section, we provide theoretical analysis on the proposed method Adversarial Flow Distillation (AFD). We reveal the connection between policy optimization in AFD to the KL-divergence on-policy distillation from teacher.

Let 𝜋𝑇(𝑥0|𝑦) denote the teacher distribution accessible only through clean samples, and 𝜋𝜃(𝑥0|𝑦) the student parameterized by velocity field 𝑣𝜃(𝑥𝑡, 𝑦, 𝑡). Let 𝑣old(𝑥𝑡, 𝑦, 𝑡) denote the EMA data-collection policy, with induced distribution 𝜋old. The forward process is

𝑥𝑡 = 𝛼𝑡𝑥0 + 𝜎𝑡𝜖, 𝜖 ∼ N(0, 𝐼), (14) with sample-level forward velocity

𝑣(𝑥0, 𝜖, 𝑡) = 𝛼𝑡𝑥0 + 𝜎𝑡𝜖. (15)

Self-Forcing

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

+GAN

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

+AFD

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

- Figure 7 | Additional qualitative visualization of AFD. Prompt: A breathtaking tightrope walker, clad in a sleek black outfit and a striking red helmet, navigates a narrow tightrope suspended high above a lush green landscape. The camera captures a close-up of their focused face, framed by a dark visor, as they expertly manipulate a wooden block along the rope using a long, slender balance pole. The block glides smoothly, creating a subtle groove in the taut rope, a testament to their precision and control. The background blurs into a vibrant tapestry of greenery, enhancing the sense of isolation and danger. The tightrope, a thin line of tension, contrasts sharply with the serene environment, while the camera remains steady, allowing the viewer to fully immerse in the walker’s intense concentration and the delicate dance of balance and skill.

Self-Forcing

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

+GAN

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

+AFD

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- Figure 8 | Additional qualitative visualization of AFD. Prompt: A majestic dragon kite soars through a narrow urban canyon, framed by towering, weathered brick buildings. The kite, intricately crafted with a vibrant red body and a fierce, golden head, glides effortlessly against a backdrop of deep blue skies, its tail adorned with a cascade of colorful ribbons that flutter in the breeze. The camera captures the kite’s dynamic flight, showcasing its graceful turns and the subtle interplay of light and shadow as it navigates the tight space between the buildings. The scene is enhanced by a soft, golden-hour glow, casting a warm hue over the textured brick facades, while the kite’s movement creates a striking contrast against the static architecture. This cinematic moment celebrates the artistry of kite flying, blending the beauty of nature with the urban landscape.

Causal-Forcing

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

+GAN

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

+AFD

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

- Figure 9 | Additional qualitative visualization of AFD. Prompt: In a breathtaking underwater tableau, two swimmers glide through the crystal-clear water, their synchronized breaststroke movements creating a mesmerizing dance of rhythm and grace. The camera, positioned at a slight angle, captures the swimmers’ powerful legs and arms as they propel themselves forward, their bodies slicing through the water with precision. The first swimmer, clad in a sleek black swimsuit, leads the pack, while the second, in a vibrant red suit, trails closely, their synchronized movements echoing a fierce competition. The water’s surface shimmers with sunlight, casting dynamic reflections that enhance the scene’s ethereal quality. The static camera allows for a focused view of their synchronized strokes, highlighting the swimmers’ strength and agility, while the serene underwater environment amplifies the beauty of their synchronized performance.

The discriminator 𝐷𝜙(𝑥ˆ0, 𝑦) ∈ (0, 1) is trained to distinguish teacher samples 𝑥0𝑇 ∼ 𝜋𝑇 from student rollouts 𝑥ˆ0 ∼ 𝜋𝜃, with optimum

𝜙(𝑥ˆ0, 𝑦) 1 − 𝐷∗

𝐷∗

𝜋𝑇(𝑥ˆ0|𝑦) 𝜋𝑇(𝑥ˆ0|𝑦) + 𝜋𝜃(𝑥ˆ0|𝑦)

𝜋𝑇(𝑥ˆ0|𝑦) 𝜋𝜃(𝑥ˆ0|𝑦)

𝐷𝜙∗ (𝑥ˆ0, 𝑦) =

, 𝜌∗𝜙(𝑥ˆ0, 𝑦) := log

= log

. (16)

𝜙(𝑥ˆ0, 𝑦)

In the AFD setting, if we set advantage-form reward dependent on the density ratio estimated by discriminator:

𝑟𝜙(𝑥0, 𝑦) := exp 𝜌𝜙(𝑥0, 𝑦) . (17) Recall from DiffusionNFT Section 3.1:

𝑟(𝑥0, 𝑐) 𝑝𝜋old(𝑜 = 1|𝑐)

𝜋+(𝑥0|𝑐) := 𝜋old(𝑥0|𝑜 = 1, 𝑐) =

𝜋old(𝑥0|𝑐). (18)

This is the conditional distribution of 𝑥0 given that the optimality variable 𝑜 equals 1 — i.e., the reward-reweighted version of 𝜋old that puts more mass on high-reward samples. The tilted policy can also be written as

𝜋+(𝑥0|𝑦) ∝ exp 𝜌𝜙 · 𝜋old(𝑥0|𝑦), (19) and at optimal 𝜙∗ due to the on-policy setting 𝜋old = 𝜋𝜃, this reduces to

𝜋+(𝑥0|𝑦) ∝ 𝜋𝑇(𝑥0). (20)

Equation 20 demonstrates that the tilted policy 𝜋+ under the optimal discriminator recovers the teacher policy. In the following proposition, we show that the policy optimization process in AFD pushes the policy towards the teacher model.

Proposition 1. Under an optimal discriminator, the policy improvement step from 𝜋old to the reward-tilted distribution 𝜋+ is equivalent to on-policy reverse-KL distillation from the teacher 𝜋𝑇.

Proof. The tilted policy 𝜋+ is the solution to the optimization problem:

𝔼𝑥0∼𝜋[𝜌(𝑥0, 𝑦)] − KL 𝜋 ∥ 𝜋old . (21) Plugging the optimal discriminator into 𝜌, we obtain that the optimization problem becomes

max

𝜋

𝜋𝑇(𝑥0) 𝜋𝜃(𝑥0)

− KL 𝜋 ∥ 𝜋old (22)

max

𝔼𝑥0∼𝜋 log

𝜋

−KL(𝜋 ∥ 𝜋𝑇) + KL(𝜋 ∥ 𝜋𝜃) − KL 𝜋 ∥ 𝜋old (23) ⇐⇒max

###### ⇐⇒max

𝜋

−KL(𝜋 ∥ 𝜋𝑇) (24) Equation 24 finally becomes reverse-KL distillation from the teacher model. □

𝜋

- Table 4 | Prompts used for Figure 3.

Example Prompt

- physics1 In a breathtaking display of aquatic athleticism, a skilled skier glides effortlessly across the shimmering surface of a tranquil lake, propelled by a powerful ski jet. The scene opens with the skier, clad in a vibrant red and black wetsuit, gracefully poised on the water, their skis slicing through the glassy surface. The ski jet, a sleek black vessel with a striking red stripe, glides alongside, its engine humming with precision as it maintains a steady pace. The camera captures the dynamic interplay between the skier and the jet, showcasing the skier’s agility and balance as they lean into the turns, their movements fluid and controlled. The backdrop features a serene landscape of lush greenery and distant mountains, enhancing the tranquil ambiance. As the skier gains momentum, the camera remains fixed, allowing viewers to appreciate the exhilarating dance of speed and skill on the water, culminating in a thrilling display of aquatic prowess.
- physics2 In a serene bathroom setting, a person clad in a light blue shirt and dark pants stands before a sleek white sink, the soft glow of natural light filtering through a nearby window. The camera captures a close-up of their hands, one gently squeezing a vibrant blue toothpaste tube, while the other holds a pristine white toothbrush, poised to receive the creamy substance. As the tube is squeezed, a steady stream of toothpaste flows onto the bristles, creating a mesmerizing cascade of blue that contrasts beautifully with the white brush. The scene is bathed in a warm, inviting light, enhancing the textures of the toothpaste and the brush, while the background remains softly blurred, drawing the viewer’s focus to the meticulous act of oral hygiene. The static camera angle ensures a clear view of the process, capturing the rhythmic motion of the hands and the satisfying result of a perfectly coated toothbrush.
- physics3 In a serene, softly lit room, a pair of hands gracefully manipulates knitting needles, their rhythmic movements creating a mesmerizing dance of craftsmanship. The camera captures a close-up of the hands, framed against a blurred backdrop, enhancing the focus on the intricate process. The needles glisten under the warm, diffused light, casting gentle shadows that accentuate the texture of the yarn. As the hands deftly weave the yarn, the knitted piece grows visibly longer, each stitch a testament to the skill and patience of the unseen artisan. The static camera angle allows for an intimate exploration of the technique, while the subtle color grading enhances the cozy atmosphere, inviting viewers to immerse themselves in the artistry of knitting.
- physics4 A vibrant kite, adorned with whimsical cartoon characters, soars against a backdrop of lush greenery and a clear blue sky, its colorful design capturing the essence of childhood joy. The kite, featuring a playful character with a wide smile and a distinctive hat, is held by a young child in a bright yellow shirt, who eagerly runs in a circular motion, their laughter echoing through the air. The camera, positioned at a low angle, captures the dynamic interplay between the child’s enthusiasm and the kite’s graceful ascent, as it dances on the breeze. The scene is bathed in soft, natural light, enhancing the vivid hues of the kite and the child’s clothing, while the gentle rustle of leaves and the distant hum of nature create an immersive atmosphere. This moment encapsulates the simple pleasures of outdoor play, inviting viewers to share in the exhilaration of flight and the boundless energy of youth.r

- Table 5 | Prompts used for Figure 4.

Example Prompt

- anime1 A brave cartoon rabbit exploring a glowing mushroom forest at dusk, expressive 3D cartoon character, magical storybook atmosphere, warm cinematic lighting, whimsical fantasy environment, soft polished textures, original fairy-taleinspired animation style.
- anime2 A cartoon whale floating through a dreamy sky filled with stars and clouds, whimsical fantasy sequence, polished 3D cartoon family-animation look, cinematic soft glow, original fairy-tale-inspired animation style.

