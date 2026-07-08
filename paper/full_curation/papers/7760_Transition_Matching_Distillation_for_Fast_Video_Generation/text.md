[Figure 1]

2026-1-16

## Transition Matching Distillation for Fast Video Generation

Weili Nie1*, Julius Berner1*, Nanye Ma2, Chao Liu1, Saining Xie2, Arash Vahdat1 1NVIDIA 2NYU *equal contribution

# arXiv:2601.09881v1[cs.CV]14Jan2026

[Figure 2]

Figure 1 | Generated examples from TMD. Four frames of 5s 480p videos generated from two text prompts using our TMD method (distilled from Wan2.1 14B T2V) with two different (effective) number of function evaluations (NFE) (see the definition of effective NFE in Eq. (16)).

Abstract. Large video diffusion and flow models have achieved remarkable success in high-quality video generation, but their use in real-time interactive applications remains limited due to their inefficient multi-step sampling process. In this work, we present Transition Matching Distillation (TMD), a novel framework for distilling video diffusion models into efficient few-step generators. The central idea of TMD is to match the multi-step denoising trajectory of a diffusion model with a few-step probability transition process, where each transition is modeled as a lightweight conditional flow. To enable efficient distillation, we decompose the original diffusion backbone into two components: (1) a main backbone, comprising the majority of early layers, that extracts semantic representations at each outer transition step; and (2) a flow head, consisting of the last few layers, that leverages these representations to perform multiple inner flow updates. Given a pretrained video diffusion model, we first introduce a flow head to the model, and adapt it into a conditional flow map. We then apply distribution matching distillation to the student model with flow head rollout in each transition step. Extensive experiments on distilling Wan2.1 1.3B and 14B text-to-video models demonstrate that TMD provides a flexible and strong trade-off between generation speed and visual quality. In particular, TMD outperforms existing distilled models under comparable inference costs in terms of visual fidelity and prompt adherence. Project page: https://research.nvidia.com/labs/genair/tmd

© 2026 NVIDIA. All rights reserved.

### 1. Introduction

Recent progresses in large-scale diffusion models [19, 45] have significantly advanced the frontier of video generation [57, 25, 5, 1, 49, 15]. Open-sourced models (such as HunyuanVideo [25], Wan [49] and Cosmos [1]) and commercial text-to-video (T2V) systems (such as Sora, Veo and Kling) demonstrate remarkable capabilities in synthesizing coherent and photorealistic videos from text prompts. Despite their success, sampling inefficiency remains a central bottleneck. Standard diffusion models rely on a multi-step denoising process, often requiring hundreds of iterative steps, to progressively transform noise into realistic outputs [25, 49]. This iterative nature leads to high inference latency and computational cost, rendering large diffusion models impractical for interactive applications such as real-time video generation, content editing, or world modeling for agent training. Accelerating diffusion sampling without sacrificing visual quality thus becomes a key open challenge.

A growing body of research has explored diffusion distillation to compress long denoising trajectories into a small number of inference steps. Existing approaches can be broadly categorized into two families: (1) trajectory-based distillation, which includes knowledge distillation [36, 40] and consistency models [46, 16, 35, 17] that directly regress the teacher’s denoising trajectories; and (2) distribution-based distillation, encompassing adversarial [42, 41] and variational score distillation [59, 70, 58] methods that align the student and teacher distributions. These techniques can reduce the sampling process to as few as one or two steps in the image domain. However, extending them to video diffusion models presents unique challenges. Videos exhibit high spatiotemporal dimensionality and complex inter-frame dependencies, making it difficult to preserve both global motion coherence and fine-grained spatial details during distillation. Most existing methods treat the diffusion network as a monolithic mapping [28, 64, 69], neglecting the hierarchical structure and semantic progression inherent in large video diffusion backbones.

To address these limitations, we propose Transition Matching Distillation (TMD) to distill large video diffusion models into few-step generators (e.g., less than 4 steps; see Figure 1 for an example). Inspired by Transition Matching [44], TMD approximates the many-step denoising process with a compact few-step probability transition process, where each transition captures the distributional evolution of video samples across widely separated noise levels, enabling the student to take large transition steps that match the teacher model’s distribution. To model the transition process, we introduce a decoupled architecture for the

student model with two components: (1) a main backbone, comprising the majority of early layers, that extracts high-level semantic representations at each outer transition step; and (2) a flow head, consisting of the final few layers, that conditions on these representations and refines the fine-grained visual details through multiple inner flow updates.

This hierarchical decomposition allows the student to share representations from the main backbone with the flow head. The flow head then performs a few lightweight inner refinement steps within each outer transition step, providing a flexible mechanism to balance sampling efficiency and visual fidelity. First, we pretrain the student via trajectory-based distillation for the flow head using an adaption of MeanFlow [17]. Then, we formulate distillation as a distribution matching problem between the teacher denoising process and the student transition process using an improved version of DMD2 [58]. By unrolling the flow head during training, we align the student’s probability transitions with the teacher’s multi-step diffusion distribution, capturing both semantic evolution and fine-grained visual details.

We evaluate TMD in distilling Wan2.1 1.3B and 14B T2V models, using VBench [22] and a user preference study. Our experiments show that TMD consistently outperforms existing distilled methods under comparable inference budgets, achieving better visual fidelity and prompt adherence. In particular, our distilled 14B model achieves an overall score of 84.24 on VBench in near-one-step generation (NFE=1.38).

Our main contributions are summarized as follows:

- • A novel distillation framework for video diffusion: We propose Transition Matching Distillation (TMD), which distills long denoising trajectories into a compact few-step probability transition process.
- • A decoupled diffusion backbone design: We decompose the teacher model into a semantic backbone and a recurrent flow head, enabling hierarchical distillation with flexible inner flow refinement.
- • A two-stage training strategy: (1) transition matching adaption that converts the flow head into a conditional flow map and (2) distribution matching distillation with flow head rollout in each transition step.
- • Comprehensive empirical validation: We demonstrate the effectiveness of TMD in distilling Wan2.1 1.3B and 14B T2V models, achieving state-of-the-art trade-offs between speed and quality in few-step video generation.

### 2. Background

Transition Matching. Transition matching (TM) [44] is a continuous-state generative model, that can be viewed as a generalization of flow matching to discrete time. It uses the rectified flow schedule

𝑥𝑡 = (1 − 𝑡)𝑥 + 𝑡𝑥1, 𝑡 ∈ [0,1], (1)

interpolating between the data 𝑥 and a standard normal 𝑥1 ∼ 𝒩(0,𝐼). Flow matching approximates the instantaneous velocity

𝑣(𝑥,𝑡) := E[𝑥˙𝑡|𝑥] = E[𝑥1 − 𝑥|𝑥], (2)

requiring many small time-steps to generate data 𝑥 from noise 𝑥1 [34, 2, 32]. In contrast, TM directly models the probabilistic transition between two states 𝑥𝑡

, where 0 = 𝑡0 < 𝑡1 < ··· < 𝑡𝑀 = 1 is a given time discretization. For every transition the model is learned to predict an auxiliary latent variable 𝑦, such that 𝑥𝑡

to 𝑥𝑡

𝑖−1

𝑖

is easy to sample (e.g., deterministic) given 𝑦 and 𝑥𝑡

𝑖−1

. For instance, Difference Transition Matching (DTM) considers1

𝑖

𝑦 := 𝑥1 − 𝑥 (3) such that

𝑖 − (𝑡𝑖 − 𝑡𝑖−1)𝑦. (4) To predict 𝑦 given 𝑥𝑡

= 𝑥𝑡

#### 𝑥𝑡

𝑖−1

, transition matching considers an “inner” rectified flow schedule

𝑖

𝑦𝑠 = (1 − 𝑠)𝑦 + 𝑠𝑦1, 𝑠 ∈ [0,1], (5) where 𝑦1 ∼ 𝒩(0,𝐼). In practice, at every step 𝑥𝑡

, the main backbone predicts features and a lightweight head is learned using flow matching to approximate the inner velocity

𝑖

):=E[𝑦˙𝑠|𝑦,𝑥𝑡

𝑣(𝑦,𝑠;𝑥𝑡

]= E[𝑦1−(𝑥1−𝑥)|𝑦,𝑥𝑡

] (6)

𝑖

𝑖

𝑖

While DTM provided improved performance in image generation, it still needs around 30 transition steps in sampling.

MeanFlow. To accelerate the sampling of diffusion models, MeanFlow [17] proposes to learn a flow map 𝑓(𝑦𝑠,𝑠,𝑟) [4], which maps the point 𝑦𝑠 at arbitrary time 𝑠 on the PF-ODE trajectory to any preceding point 𝑥𝑟 for 𝑟 < 𝑠. Formally,

𝑓(𝑦𝑠,𝑠,𝑟) := 𝑦𝑠 − ∫︀𝑠𝑟 𝑣(𝑦𝜏,𝜏)d𝜏 (7) where 𝑣 denotes the instantaneous velocity as in (6). MeanFlow parameterizes this mapping using the average velocity along the trajectory segment from 𝑦𝑠 to

1While TM uses a probabilistic model to predict 𝑦, flow matching predicts the conditional expectation of 𝑦 (Eq. (2)).

𝑦𝑟, expressed as 𝑓(𝑦𝑠,𝑠,𝑟) = 𝑦𝑠 + (𝑠 − 𝑟)𝑢(𝑦𝑠,𝑠,𝑟), where 𝑢 represents the average velocity. Combining

with (7), we have (𝑠−𝑟)𝑢(𝑦𝑠,𝑠,𝑟) = −∫︀𝑠𝑟 𝑣(𝑦𝜏,𝜏) d𝜏. This inspires the key insight from MeanFlow, where

differentiating both sides w.r.t 𝑠 obtains the following identity:

𝑢(𝑦𝑠,𝑠,𝑟) + (𝑠 − 𝑟)dd𝑠𝑢(𝑦𝑠,𝑠,𝑟) = 𝑣(𝑦𝑠,𝑠), (8)

where dd𝑠𝑢(𝑦𝑠,𝑠,𝑟) refers to the total derivative. This identity motivates the following practical training

objective: ℒ(𝜃) := E𝑠,𝑟,𝑦

[︀‖𝑢𝜃(𝑦𝑠,𝑠,𝑟) − 𝑢^‖2]︀

(9) with

𝑠

(︀

)︀

𝑣(𝑦𝑠,𝑠) − (𝑠 − 𝑟)dd𝑠𝑢𝜃(𝑦𝑠,𝑠,𝑟)

𝑢^ := sg

, (10)

where sg(·) refers to the stop gradient operator, the instantaneous velocity 𝑣(𝑦𝑠,𝑠) is approximated by conditional velocity 𝑦1 − 𝑦, and the total derivative

d d𝑠𝑢(𝑦𝑠,𝑠,𝑟) can be computed using either forwardmode automatic differentiation or a finite-difference

approximation [47, 52].

DMD. Flow maps like MeanFlow must learn mappings between points along the ODE trajectory, which is difficult to scale to video generation due to high dimensionality and large trajectory curvature (see Appendix B). In contrast, distribution distillation methods such as DMD2 [58] offer more flexibility by matching only the output distribution to that of the teacher or data—using a GAN [18] loss for the latter and a variational score distillation (VSD) [59] loss, i.e., reverse KL divergence, for the former. This can be represented as

[︀

(︀

)︀𝑇

]︀

𝐷(^𝑥𝑡,𝑡)

𝑥^

, (11) where 𝑥^ = 𝑔𝜃(𝑥𝑡

𝑤(𝑡)sg

ℒ(𝜃) = E𝑡

𝑖,𝑥𝑡𝑖,𝑡,𝑥^𝑡

,𝑡𝑖) denotes the student output from the noisy input 𝑥𝑡

𝑖

, and 𝑡𝑖 is sampled from a given student time discretization, 𝑤 is a weighting function, 𝐷 is the difference between the scores of the student and teacher distributions, 𝑥^𝑡 = (1 − 𝑡)𝑥^ + 𝑡𝑥1 is a noisy sample from the forward process in (1) that is passed to score functions.

𝑖

Since we do not have access to the score of the student distribution, a so-called fake score is initialized with the teacher parameters and trained on data from the student using flow matching. Up to a timedependent weighting, 𝐷 equals the difference between the velocities of the teacher and fake score. In practice, both the fake score and GAN discriminator are updated for a given number of iterations in between updates of the student and the discriminator consists of a lightweight head operating on intermediate features of the fake score or teacher.

Noise Data

[Figure 3]

[Figure 4]

[Figure 5]

yr

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

r

c

[Figure 14]

[Figure 15]

[Figure 16]

Flow Head

s

t

Fusion

mt

ys

Unrolling flow head for N steps

c

ys

ys

ys

c

j−1

0

ys

j

Main backbone

N

mt

t

c

Main Backbone

t

Fusion Flow Head

xt

xt

(a) Decoupled architecture (b) Transition process with flow head rollout

- Figure 2 | Overview of our TMD method. (a) Decoupled architecture for TMD student, where the main backbone takes the noisy sample 𝑥𝑡, timestep 𝑡 and text conditioning 𝑐 as inputs and outputs the main feature 𝑚𝑡, and with a simple fusion layer, flow head conditions on 𝑚𝑡, 𝑐 and predicts the less noisy target 𝑦𝑟 from the more noisy 𝑦𝑠 (𝑠 ≥ 𝑟). (b) Top: Transition process maps noise to data with a few transition steps. Bottom: In each step, flow head rollout is performed during both distillation and sampling. We omit the timestep inputs 𝑠 and 𝑟 to the flow head for simplicity.

### 3. Method

We introduce our method – Transition Matching Distillation (TMD), including two training stages: (1) transition matching pretraining to initialize a flow head capable of iteratively refine features extracted from the main backbone; and (2) distillation with flow head where we introduce DMD2-v that advances DMD2 in few-step video generation and apply it with flow head rollout at each transition step. For ease of presentation, we drop additional conditioning of the teacher model, such as text conditioning, in our notation. Below, we start with introducing our student architecture before presenting our two-stage training.

Decoupled architecture. Our method follows the general formulation of transition matching explained in Section 2. Different from TM, we aim to approximate many small denoising steps of a teacher model with the large transition step of the distilled student. To efficiently predict 𝑦 in every transition step 𝑡𝑖, we decouple the pretrained teacher architecture into a main backbone 𝑚𝜃, that acts as a feature extractor, and a lightweight flow head 𝑓𝜃 that iteratively predicts 𝑦 given these features, i.e.,

(︀

)︀

,𝑠𝑗,𝑠𝑗−1;𝑚𝜃(𝑥𝑡

,𝑡𝑖)

, (12)

𝑗−1 ≈ 𝑓𝜃

𝑦𝑠

#### 𝑦𝑠

𝑗

𝑖

where 0 = 𝑠0 < 𝑠1 < ··· < 𝑠𝑁 = 1 is a given time discretization for the inner flow; see also Figure 2.

While such decoupling has been successfully used for training generative models [61, 44, 50, 27, 67], it requires a careful design to minimally disrupt the pretrained model. Our design considers two key factors:

Algorithm 1 TMD inference

𝑥 ∼ 𝒩(0, 𝐼) for 𝑖 = 𝑀 to 1 do

𝑚 = 𝑚𝜃(𝑥, 𝑡𝑖) ◁ Main backbone 𝑥 = 𝑥 − (𝑡𝑖 − 𝑡𝑖−1)InnerFlow(𝑚) ◁ See Eq. (4)

return 𝑥 ◁ Generated data def InnerFlow(𝑚)

𝑦 ∼ 𝒩(0, 𝐼) for 𝑗 = 𝑁 to 1 do

𝑦 = 𝑓𝜃(𝑦, 𝑠𝑗, 𝑠𝑗−1; 𝑚) ◁ Flow head return 𝑦

- (1) flow head target 𝑦. We find that the DTM formula-

tion 𝑦 = 𝑥1 − 𝑥 outperforms other target types, such as the sample prediction 𝑦 = 𝑥 (see Appendix B).

- (2) fusion layer. We use a time-conditioned gating

mechanism to fuse the main feature 𝑚𝑡

and the noisy flow head target 𝑦𝑠

𝑖

, ensuring the student’s initial forward pass matches that of the teacher. Additionally, we reuse the patch embedding of the main input 𝑥𝑡

𝑗

. See Appendix A for details. We provide pseudo-code for our inference in Algorithm 1.

for the inner flow input 𝑦𝑠

𝑖

𝑗

3.1. Stage 1: Transition matching pretraining Given the decoupled architecture, we convert the flow head into a flow map for iterative refinement before distillation. Similar to TM, we can directly use a flow matching loss in Eq. (6) to train the flow head to approximate the velocity of the inner flow. However, in theory, this would still require many inner steps to approximate 𝑦. Thus, we leverage MeanFlow to obtain a few-step flow head.

Transition matching MeanFlow. On a highlevel, our pretraining algorithm, termed Transition Matching MeanFlow (TM-MF), uses a MeanFlow objective as in Eq. (9), conditioned on the main feature 𝑚 = 𝑚𝜃(𝑥𝑡

,𝑡𝑖); see Algorithm 2 for the pseudocode. Specifically, we parametrize the conditional inner flow map using an average velocity as

𝑖

𝑓𝜃(𝑦𝑠,𝑠,𝑟;𝑚) := 𝑦𝑠 + (𝑠 − 𝑟)𝑢𝜃(𝑦𝑠,𝑠,𝑟;𝑚). (13)

Note that we do not detach the main backbone features during training as this would limit the flexibility needed in pretraining. Naively training the flow head to predict the average velocity 𝑢𝜃 performs poorly. Our hypothesis is that the flow head’s output should remain close to the pretrained teacher’s output. Since the teacher predicts the velocity of the outer flow in Eq. (2), the flow head should instead predict E[𝑥1 − 𝑥 | 𝑥] to stay aligned with the teacher. From the inner velocity definition in Eq. (5), we obtain

𝑢𝜃(𝑦𝑠,𝑠,𝑠;𝑚) ≈ E[𝑦1 − (𝑥1 − 𝑥)|𝑦,𝑥𝑡] Thus, we parametrize the average velocity as

𝑢𝜃(𝑦𝑠,𝑠,𝑟;𝑚) := 𝑦1 − head𝜃(𝑦𝑠,𝑠,𝑟;𝑚), (14)

where head𝜃 is the head of our decoupled architecture (initialized from the teacher as outlined in Ap-

pendix A). With this parameterization, the head𝜃 output approximates the teacher velocity prediction in the limit of 𝑟 → 𝑠.

For improved performance and stability, we follow the original MeanFlow to (1) perform flow matching (more precisely, transition matching in our setting) for a part of the batch, (2) use classifier-free guidance (CFG) (adapting the conditional velocity 𝑣(𝑦𝑠,𝑠)), drop the text condition with a certain probability, and (3) use adaptive loss normalization. Since JVP computation in Eq. (10) requires custom implementations to be compatible with large-scale training for video generation (e.g., flash attention [12], Fully Sharded Data Parallel (FSDP) [66], or context parallelism [23]), we use a finite-difference approximation of the JVP to make our algorithm agnostic of the underlying architectures and training techniques [47, 52]; see Appendix A for details.

Since we do not have direct access to the inner flow velocity, we use the conditional velocity 𝑣(𝑦𝑠,𝑠) = 𝑦1 − 𝑦 in the objective (9). We note that for specific 𝑦, one can also derive representations of the inner velocity in terms of the pretrained teacher velocity [20], which we leave for future work. Finally, we also observed that TM can provide competitive results as a pretraining strategy; see Section 4.3 for an ablation. In particular, TM pretraining can be understood as a special case of MeanFlow in Eq. (9) for 𝑟 = 𝑠 when using the conditional velocity.

Algorithm 2 TMD student update step (simplified)

Given 𝑥 ∼ 𝑝data, 𝑥1 ∼ 𝒩(0, 𝐼), 𝑡𝑖 ∼ Unif({𝑡1, . . . , 𝑡𝑀})

𝑥𝑡𝑖 = (1 − 𝑡𝑖)𝑥 + 𝑡𝑖𝑥1 ◁ See Eq. (1) 𝑚 = 𝑚𝜃(𝑥𝑡𝑖, 𝑡𝑖) ◁ Main backbone if stage_one then

𝑦 = 𝑥1 − 𝑥 ◁ See Eq. (3) 𝑦1 ∼ 𝒩(0, 𝐼), (𝑠, 𝑟) ∼ 𝑝𝑠,𝑟 𝑦𝑠 = (1 − 𝑠)𝑦 + 𝑠𝑦1 ◁ See Eq. (5) 𝑢 = 𝑢𝜃(𝑦𝑠, 𝑠, 𝑟; 𝑚) ◁ Avg. velocity 𝑣 = 𝑦1 − 𝑦 ◁ Conditional velocity ℒ = MeanFlow(𝑢, 𝑣, 𝑠, 𝑟) ◁ See Eq. (9)

###### else

𝑥^ = 𝑥1 − InnerFlow(𝑚) ◁ See Eq. (15) & Algorithm 1 ℒ = VSD(^𝑥) + 𝜆 · Discriminator(^𝑥) ◁ See Eq. (11)

𝜃 = step(𝜃, ∇𝜃ℒ) ◁ Gradient step

#### 3.2. Stage 2: Distillation with flow head

After the TM-MF pretraining, we apply distribution distillation to align the student and teacher distributions. We significantly improve the baseline DMD2 for video models (see Appendix B) and leverage our optimized implementation for our TMD method.

DMD2-v. DMD2 was originally proposed for distilling image diffusion models. Thus, its design choices may not be optimal in the video domain. We identify three key factors that improve DMD2 for videos (termed DMD2-v), which serve as the default setting for our TMD training: (1) GAN discriminator architecture. We find that using Conv3D layers in the GAN discriminator outperforms other architectures, implying the importance of localized, spatiotemporal features for the GAN loss; (2) Knowledge distillation (KD) warm-up. We find that KD warmup improves the performance in one-step distillation, however, in multi-step generation, it tends to introduce coarse-grained artifacts that can be hardly fixed by the DMD2 training (see Figure 10 in Appendix). Therefore, we only apply KD warm-up in one-step distillation for DMD-v. (3) Timestep shifting. When we sample a timestep for the outer transition step or for adding noise to generated samples in the VSD loss, we find that applying a shifting function 𝑡 = 𝛾𝑡

′

(𝛾−1)𝑡′+1 with 𝛾 ≥ 1 to a uniformly sampled 𝑡′ improves the performance and prevents mode collapse.

Flow head rollout. During distillation, we unroll the inner flow and treat the resulting architecture as a sample generator 𝑔𝜃(𝑥𝑡

,𝑡𝑖;𝑦1) at each transition step 𝑡𝑖 (see Figure 2b). Considering the flow head target 𝑦 = 𝑥1 − 𝑥 defined in Eq. (3), the unrolled student output is given by

𝑖

,𝑡𝑖;𝑦1):=𝑥1 − InnerFlow(𝑚𝜃(𝑥𝑡

,𝑡𝑖)) (15) where 𝑦^0 ≈ InnerFlow(𝑚𝜃(𝑥𝑡

#### 𝑔𝜃(𝑥𝑡

𝑖

𝑖

,𝑡𝑖)) denotes the final prediction of the flow head after 𝑁 inner refinement steps, following Eq. (12).

𝑖

Overall Quality Semantic

Overall Quality Semantic

Method NFE

Method NFE

score score score Diffusion models

score score score Diffusion models

VideoCrafter2 1.4B [7] 50×2 80.44 82.20 73.42 CogVideoX-2B [57] 50×2 81.55 82.48 77.81 Seaweed-7B [43] 25×2 82.15 84.36 73.31 Wan2.1 1.3B [49] 50×2 84.26 85.30 80.09

Hunyuan Video 13B [25] 50×2 83.43 85.07 76.88 Wan2.1 14B [49] 50×2 86.22 86.67 84.44

Distilled models

rCM [69] 4 84.92 85.43 82.88 DMD2-v 4 84.52 85.37 81.11 rCM [69] 2 85.05 85.57 82.95 DMD2-v 2 84.79 85.78 80.83 TMD-N4H5 (Ours) 2.75 84.62 85.09 82.72 rCM [69] 1 83.02 83.57 80.81 DMD2-v 1 83.69 84.46 80.61 TMD-N4H5 (Ours) 1.38 84.24 84.89 81.65

Distilled models

DOLLAR [13]‡ 4 82.57 83.83 77.51 T2V-Turbo-v2 [28]* 4 82.34 83.93 75.97 rCM [69] 4 84.43 85.38 80.63 DMD2-v 4 84.60 86.03 79.87 APT [29]† 2 81.85 84.39 71.70 rCM [69] 2 84.09 84.90 80.86

- DMD2-v 2 84.39 85.65 79.32
- DMD2-v 3 84.48 85.71 79.58 TMD-N2H5 (Ours) 2.33 84.68 85.71 80.55 TMD-N4H5 (Ours) 3.00 84.67 85.72 80.47 APT [29]† 1 82.00 84.21 73.15 rCM [69] 1 82.65 83.60 78.82 DMD2-v 1 83.24 84.28 79.10 TMD-N2H5 (Ours) 1.17 83.80 85.07 78.69 TMD-N4H5 (Ours) 1.50 83.79 85.03 78.83

Table 1 | VBench results of our method and baselines when distilling Wan2.1 1.3B into a few-step generator (*The teacher model is VideoCrafter2 1.4B [7]; †The teacher model is Seaweed-7B [43]; ‡The teacher model is a variant of CogVideoX [57]). DMD2-v denotes our improved version of DMD2 for video generation. “N2H5” (or “N4H5”) means two (or four) denoising steps and five DiT blocks in the flow head. NFE in our method is represented by the effective forward pass of the whole network as in Eq. (16).

Applying the VSD loss in Eq. (11) from DMD2-v to the unrolled student output naturally backpropagate gradients through all 𝑁 inner flow steps. Since the flow head is lightweight, this remains efficient: for instance, if we choose 5 final DiT blocks from a 30block DiT and unroll 2 steps, it adds less than 17% extra computation in updating the student network’s parameters. In the terminology of DMD2, this can be viewed as “backward simulation” [58] of the inner flow, but without detaching the gradient computation. Distillation with flow head rollout effectively avoids the mismatch between training and inference, and thus improves the distillation performance.

Summary: We provide pseudocode for the student updates for both stages in Algorithm 2.

- 4. Experiments

Table 2 | VBench results of our method and baselines when distilling Wan2.1 14B into a few-step generator. DMD2-v denotes our improved version of DMD2 for video generation.

[21,60,104], which gets decoded to 81 frames with pixel resolution 480 × 832. We use a dataset of 500k text and video pairs, where text prompts are sampled from the VidProM dataset [51] (and extended by Qwen-2.5 [55]) and videos are generated by the Wan2.1 14B T2V model. We defer more implementation details and hyperparameters to Appendix A.

Evaluation metrics. To evaluate our method and baselines, we use VBench [22] (where we report total score, quality score and semantic score) and a user preference study to assess visual quality and prompt adherence. We consider the effective number of function evaluations (NFE) to be the total number of DiT blocks used during generation divided by 𝐿, the number of blocks in the teacher architecture; for the baselines this corresponds to the number of steps 𝑀, and for our TMD models, this corresponds to

(︀

𝐿 )︀

1 + (𝑁−1)𝐻

Effective NFE := 𝑀

, (16)

where 𝑁 is the number of inner flow steps and 𝐻 is the number of blocks in the flow head. We note that 𝐿 = 30 for Wan2.1 1.3B and 𝐿 = 40 for Wan2.1 14B.

#### 4.2. Comparison with existing methods

Our TMD method is built upon the improved version of DMD2 for video generation (termed DMD2-v). Here, we compare TMD with DMD2-v and existing baselines in distilling video diffusion models. We provide a visual comparison in Figure 3 and refer to Appendix B for further examples. In Table 1, we show the VBench results of distilling Wan2.1 1.3B (or video models with a similar size) into a few-step generator, where we group the distilled models based on the number of student denoising steps 𝑀. When 𝑀 = 2, TMD-N2H5 with effective NFE = 2.33 (i.e., 2 denoising steps and 5 DiT blocks in flow head)

#### 4.1. Experiment setup

Implementation. We use Wan2.1 1.3B and 14B T2V-480p [49] as teacher video diffusion models and distill them to the same-sized student models with a decoupled architecture. All experiments are performed on the latent resolution of [𝑇,𝐻,𝑊] =

[Figure 17]

[Figure 18]

- Figure 3 | Visual comparison. We compare three frames (and zoomed-in regions of interest) of the outputs of TMD and DMD2-v on exemplary prompts for Wan2.1 1.3B (left) and Wan2.1 14B (right). TMD can improve visual quality at comparable cost to our DMD2-v baseline. Extended prompts can be found in Appendix A.

[Figure 19]

[Figure 20]

- Figure 4 | Visual comparison. We compare three frames of TMD and DMD2-v on exemplary prompts for Wan2.1 14B. TMD can improve prompt adherence at comparable cost to our DMD2-v baseline. Extended prompts can be found in Appendix A.

NFE = 1.38 significantly outperforms all other onestep distillation methods with an overall score of 84.24. It improves over one-step rCM by +1.22 while adding only minimal inference cost. In Figure 4, we show TMD improves prompt adherence, supporting the numerical improvement. Besides, TMD eliminates the need for the computationally expensive KD warm-up required by one-step DMD2-v.

User preference study. We perform a blinded two-alternative forced choice (2AFC) user preference study comparing TMD-N4H5 with DMD2-v when distilling Wan2.1 14B under two settings: (1) 𝑀 = 2 (two-step generation) and (2) 𝑀 = 1 (one-step generation). We randomly sample 60 challenging prompts from VBench [22] and generate 5 videos with different seeds for each prompt and model. For each prompt, raters are presented with side-by-side videos from the our model and the baseline (in random order and with random seed) and asked to perform independent pairwise comparisons for two separate criteria: visual quality and prompt alignment. Detailed instructions are provided in Figure 21.

obtains an overall score 84.68 that outperforms all other distilled models, including the strongest baseline rCM with NFE = 4 (overall score 84.43). When 𝑀 = 1, TMD-N2H5 with NFE = 1.17 also outperforms all other one-step distillation methods with an overall score 83.80, closing the gap from two-step distillation counterparts. With TMD, we can have more finegrained control over the quality-efficiency tradeoff by allowing for fractional NFEs.

As shown in Figure 5, users consistently preferred TMD over DMD2-v in both the one-step and twostep generation settings. The advantage is even more significant for prompt alignment, echoing the qualitative results in Figure 4 and underscoring the role of iterative flow-head refinement in largely improving prompt adherence.

#### 4.3. Ablation studies

In Table 2, we show the VBench results of distilling Wan2.1 14B into a few-step generator. When 𝑀 = 2, TMD-N4H5 with NFE = 2.75 does not outperform 2-step baselines (although it outperforms 4-step DMD-v). When 𝑀 = 1, TMD-N4H5 with

Three design choices in DMD2-v. Tables 3-5 show the impact of the discriminator head, KD warmup and timestep shifting in DMD2-v, respectively. In Table 3, we observe that adding the GAN loss

Visual Quality Prompt Alignment

Overall Quality Semantic score score score

100

KD warm-up

WinRate(%)

80

71.9

63.3 63.2

One-step w/ KD 83.24 84.28 79.10 One-step w/o KD 83.06 84.72 76.42

60

51.8

40

Two-step w/ KD 83.79 84.67 80.28 Two-step w/o KD 84.39 85.65 79.32

20

0

- Table 4 | Impact of the KD warm-up, where we distill Wan2.1 1.3B into a one-step or two-step student, respectively. For KD warm-up, we use teacher model to generate 10k noise-data pairs.

Timestep shifting

Overall Quality Semantic score score score

𝑡dmd w/ shift (𝛾 = 5, one-step) 83.24 84.28 79.10 𝑡dmd w/o shift (one-step)* 83.22 84.29 78.94

𝑡student w/ shift (𝛾 = 10, two-step) 84.39 85.65 79.32 𝑡student w/o shift (two-step) 83.44 84.96 77.33

- Table 5 | Impact of the timestep shifting for 𝑡dmd that controls the noise level in the DMD loss and

One-Step (𝑀 = 1) Two-Step (𝑀 = 2)

- Figure 5 | User preference study results. Comparison of TMD-N4H5 (ours) against DMD2-v under one-step (𝑀 = 1) and two-step (𝑀 = 2) distillation regimes. Values indicate the percentage of times users preferred our method over the baseline DMD2-v and the dashed line at 50% represents parity.

Overall Quality Semantic score score score

Discriminator head

Conv3D 83.24 84.28 79.10 Conv1D-2D 82.32 83.54 77.44 Attention 82.36 83.32 78.51

𝑡student that controls the denoising steps in the multistep student, where we distill Wan2.1 1.3B into a one-step or two-step student, respectively. *Note that the setting “𝑡dmd w/o shift” leads to the severe mode collapse that VBench scores cannot capture (see Fig. 10 in Appendix).

w/o GAN 81.63 82.70 77.35

Table 3 | Impact of discriminator head design in DMD2-v for one-step distillation of Wan2.1 1.3B. We compare three heads: (1) Conv3D, jointly processing spatio-temporal features; (2) Conv1D–2D, separating temporal and spatial convolutions (e.g., [64]); and (3) Attention, flattening features into tokens processed by self-attention (with pooling downsampling).

to highlight the impact of MeanFlow (TM-MF). Table 6 shows that TM-MF consistently achieves better distillation performance than TM, suggesting that TM-MF offers a superior initialization for the secondstage distillation training.

improves the distillation performance, and Conv3D outperforms the other two discriminator head architectures. In Table 4, the overall score on VBench increases with KD warm-up in one-step DMD2 but decreases with KD warm-up in two-step DMD2. It implies that we better only apply KD warm-up in one-step generation. In Table 5, we observe that applying timestep shifting to 𝑡dmd that controls the noise level in the DMD loss and 𝑡student that controls the denoising steps in the multi-step student improves the distillation performance, respectively.

Flow head rollout in distillation. It is crucial to close the gap between training and inference, by allowing gradients from the distillation objective to backpropagate through the unrolled inner flow trajectory. Figure 7 shows that applying flow head rollout in distillation largely leads to faster training convergence and improved performance.

### 5. Related works

Quality-efficiency tradeoff. The number of inner steps 𝑁 and flow head layers 𝐻 control the computational cost of inner flows. We vary 𝑁 and 𝐻 to more comprehensively analyze TMD’s performanceefficiency tradeoff. In Figure 6, we observe that the overall VBench score generally improves as the effective NFE increases. This justifies the fine-grained flexibility that our method offers in balancing generation speed and visual quality.

Accelerating diffusion models. Recent efforts to accelerate diffusion models for few-step generation generally follow two directions: Trajectory matching methods directly learn mappings along the underlying ODE trajectory [40, 46, 17, 47, 68, 4, 14, 35, 16, 26, 39, 8], while distribution matching methods train a student model to match the teacher’s distribution with fewer steps [41, 58, 54, 59, 42, 70, 37]. In trajectory matching, Progressive Distillation [40] and Shortcut Models [14] gradually reduce sampling steps during training, while Consistency Models [46, 35, 16] directly map intermediate states to clean data. Sub-

MeanFlow vs. flow matching. In transition matching pretraining, we replace the MeanFlow objective with the vanilla flow matching objective (TM)

| | |N2H5| | |N|4H5|
|---|---|---|---|---|---|---|
| | | | | | | |
| |N|2H3<br><br>N|N2H8 4H2|N4H3|DM|D2 v|
|DM|D2 v| | | | | |
| | | | | | | |

84.64

Overall Score

84.56

84.48

84.40

2.0 2.2 2.4 2.6 2.8 3.0

Effective NFE

- Figure 6 | Performance-efficiency tradeoff of TMD. We compare the overall VBench score and effective NFE of TMD, when distilling Wan2.1 1.3B with 𝑀 = 2 for different number of inner steps 𝑁 and flow head layers 𝐻 against 2- and 3-step DMD2-v. TMD can provide consistent performance gains for increasing NFE.

N4H5 N4H5 w/o rollout

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |

84.0

Overall Score

82.0

500 1,000 1,500 2,000

Iteration

Figure 7 | Convergence and rollout ablation. We compare the overall VBench score over iterations for the second-stage TMD training with and without flow head rollout. While TMD generally converges within a only few thousand iterations, we observe faster convergence and improved performance when using rollouts.

Overall Quality Semantic

Setting Pretraining

score score score N2H5

TM-MF 84.68 85.71 80.55 TM 84.61 85.64 80.46

TM-MF 84.67 85.72 80.47 TM 84.29 85.45 79.68

N4H5

Table 6 | Impact of the first stage pretraining method on the final performance when we distill Wan2.1 1.3B with 𝑀 = 2.

sequent works [17, 39, 14, 4, 24] extend this idea to learn mappings (or flow maps) between any pair of timesteps. They often face training instability when scaled to larger models or higher resolutions [35, 9, 69]. Distribution matching approaches, in contrast, align the student’s few-step sampling distribution with the teacher’s via distributional divergence [59, 58, 54]. This provides stable supervision at the distribution level and is less sensitive to the curvature of ODE trajectories, with adversarial objectives further improving sample fidelity.

As video generation grows in prominence—with its much higher sampling cost—recent works have extended these acceleration techniques to the video domain [69, 64, 29, 13]. Our method advances this line by introducing a decoupled architecture that more effectively leverages the teacher’s learned semantic representations.

Efficient video generation. Beyond reducing sampling steps, other research aims to enhance the overall efficiency of video diffusion models. Autoregressive–diffusion hybrids achieve this by training causal models that generate frames sequentially instead of predicting the entire sequence at once [60, 21, 11, 30, 48, 6, 62]. This design significantly reduces attention complexity per forward pass through shorter temporal contexts. Meanwhile, system-level optimizations

such as feature caching and efficient attention further reduce sampling cost by reusing intermediate activations [33, 65, 38] or by sparsifying or linearizing the attention operation [56, 53, 10]. These techniques are complementary to ours and can be combined with TMD for even greater video generation efficiency.

Decoupled backbone. Recent studies have demonstrated the advantages of decoupling diffusion backbones into a feature extractor (encoder) and a head (decoder) [61, 44, 50, 27, 67]. For example, aligning intermediate features with those from pretrained visual encoders improves training convergence and condition adherence [61, 3, 63]. Decoupled Diffusion Transformer (DDT) [50] further accelerates sampling by reusing extracted features across denoising steps, though such encoder sharing is not explicitly enforced during training. Transition Matching (TM) [44] explicitly models an inner flow loop for the head via flow matching [31]. Our work extends TM in two key aspects: (1) scaling it to large-scale video generation, and (2) distilling pretrained video diffusion models into few-step generators instead of training a multi-step TM model from scratch.

### 6. Conclusions

In this work, we introduced Transition Matching Distillation (TMD), a novel framework designed to address the significant inference latency of large-scale video diffusion models. The core of our method lies in a decoupled student architecture, which separates a main backbone for semantic feature extraction from a lightweight, recurrent flow head for iterative refinement. This design is coupled with a two-stage training strategy involving transition matching pretraining and distribution-based distillation. Our experiments on distilling state-of-the-art Wan2.1 models demonstrated that TMD offers fine-grained flexibility in

balancing the trade-off between generation speed and video quality. TMD can outperform existing distillation techniques, achieving superior visual fidelity and prompt adherence at comparable or even reduced computational budgets. Future directions include (1) unifying the two training stages into a single-stage pipeline and (2) integrating TMD with system-level optimizations, such as efficient attention or feature caching, to further accelerate video generation.

### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.
- [2] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.
- [3] Aritra Bhowmik, Denis Korzhenkov, Cees GM Snoek, Amirhossein Habibian, and Mohsen Ghafoorian. MoAlign: Motion-centric representation alignment for video diffusion models. arXiv preprint arXiv:2510.19022, 2025.
- [4] Nicholas M Boffi, Michael S Albergo, and Eric Vanden-Eijnden. Flow map matching. arXiv preprint arXiv:2406.07507, 2024.
- [5] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. openai.com/research/video-generation-models-asworld-simulators.
- [6] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets fullsequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.
- [7] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter2: Overcoming data limitations for highquality video diffusion models, 2024.
- [8] Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, and Sai Bi. pi-flow: Policy-based few-step generation via imitation distillation, 2025.
- [9] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025.

- [10] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sanavideo: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025.
- [11] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-forcing++: Towards minutescale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.
- [12] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344– 16359, 2022.
- [13] Zihan Ding, Chi Jin, Difan Liu, Haitian Zheng, Krishna Kumar Singh, Qiang Zhang, Yan Kang, Zhe Lin, and Yuchen Liu. DOLLAR: Few-step video generation via distillation and latent reward optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17961–17971, 2025.
- [14] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.
- [15] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.
- [16] Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. arXiv preprint arXiv:2406.14548, 2024.
- [17] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.
- [18] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139– 144, 2020.
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [20] Peter Holderrieth, Uriel Singer, Tommi Jaakkola, Ricky TQ Chen, Yaron Lipman, and Brian Karrer. GLASS flows: Transition sampling for alignment of flow and diffusion models. arXiv preprint arXiv:2509.25170, 2025.
- [21] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
- [23] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint

- arXiv:2309.14509, 2023.

[24] Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint

- arXiv:2310.02279, 2023.

- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [26] Sangyun Lee, Yilun Xu, Tomas Geffner, Giulia Fanti, Karsten Kreis, Arash Vahdat, and Weili Nie. Truncated consistency models, 2024.
- [27] Jiachen Lei, Keli Liu, Julius Berner, Haiming Yu, Hongkai Zheng, Jiahong Wu, and Xiangxiang Chu. Advancing end-to-end pixel space generative modeling via self-supervised pre-training. arXiv preprint arXiv:2510.12586, 2025.
- [28] Jiachen Li, Qian Long, Jian Zheng, Xiaofeng Gao, Robinson Piramuthu, Wenhu Chen, and William Yang Wang. T2v-turbo-v2: Enhancing video model post-training through data, reward, and conditional guidance design. In The Thirteenth International Conference on Learning Representations, 2025.
- [29] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316, 2025.
- [30] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial posttraining for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.
- [31] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [32] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching

- for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.
- [33] Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, Juan-Manuel Perez-Rua, and Jürgen Schmidhuber. Faster diffusion via temporal attention decomposition. arXiv preprint arXiv:2404.02747, 2024.
- [34] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.
- [35] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.
- [36] Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.
- [37] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. Advances in Neural Information Processing Systems, 37:115377– 115408, 2024.
- [38] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching. Advances in Neural Information Processing Systems, 37:133282–133304,

- 2024.

[39] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603,

- 2025.

- [40] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [41] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.
- [42] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer, 2024.
- [43] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, et al. Seaweed-7b: Cost-effective training of video generation foundation model. arXiv preprint arXiv:2504.08685, 2025.
- [44] Neta Shaul, Uriel Singer, Itai Gat, and Yaron Lipman. Transition matching: Scalable and flexible generative modeling. arXiv preprint arXiv:2506.23589, 2025.

- [45] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [46] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252. PMLR, 2023.
- [47] Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025.
- [48] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.
- [49] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [50] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. DDT: Decoupled diffusion transformer. arXiv preprint arXiv:2504.05741, 2025.
- [51] Wenhao Wang and Yi Yang. Vidprom: A millionscale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37:65618–65642, 2024.
- [52] Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint arXiv:2509.04394, 2025.
- [53] Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.
- [54] Yilun Xu, Weili Nie, and Arash Vahdat. One-step diffusion models with 𝑓-divergence distribution matching. arXiv preprint arXiv:2502.15681, 2025.
- [55] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.
- [56] Shuo Yang, Haocheng Xi, Yilong Zhao, Muyang Li, Jintao Zhang, Han Cai, Yujun Lin, Xiuyu Li, Chenfeng Xu, Kelly Peng, et al. Sparse videogen2: Accelerate video generation with sparse attention via semantic-aware permutation. arXiv preprint arXiv:2505.18875, 2025.

- [57] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In The Thirteenth International Conference on Learning Representations, 2025.
- [58] Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024.
- [59] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024.
- [60] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025.
- [61] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [62] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025.
- [63] Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, and Yu Cheng. VideoREPA: Learning physics for video generation through relational alignment with foundation models. arXiv preprint arXiv:2505.23656, 2025.
- [64] Zhixing Zhang, Yanyu Li, Yushu Wu, yanwu xu, Anil Kag, Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, Junli Cao, Dimitris N. Metaxas, Sergey Tulyakov, and Jian Ren. SF-v: Single forward video generation model. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [65] Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. arXiv preprint arXiv:2408.12588, 2024.
- [66] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

- [67] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [68] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, and Anima Anandkumar. Fast sampling of diffusion models via operator learning. In International conference on machine learning, pages 42390–42402. PMLR, 2023.
- [69] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuoustime consistency. arXiv preprint arXiv: 2510.08431, 2025.
- [70] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Fortyfirst International Conference on Machine Learning, 2024.

#### A.2. TM-MF

g

Pretraining recipe. To better align TM-MF pretraining with the multi-step inference strategy used in our distillation setup, we modify the sampling scheme of (𝑠,𝑟) compared to the continuous formulation in [17]. First, analogous to 𝑡student in the outer flow of DMD2-v, we also use a shifting function with 𝛾 = 10 for 𝑠student, i.e., the values defining the time grid2 0 = 𝑠0 < 𝑠1 < ··· < 𝑠𝑁 = 1. Secondly, analogous to sampling the timestep 𝑡dmd in DMD2-v, we also sample 𝑠mf during TM-MF training uniformly and shift it (using 𝛾 = 3). Then, we pick 𝑟 = 𝑠𝑘 where 𝑘 := max{𝑗 : 𝑠𝑗 ≤ 𝑠mf}. Empirically, this discrete sampling scheme is not only tailored to our Stage 2 distillation but also substantially stabilizes TM-MF pretraining.

g

Linear GELU Linear AdaLN

Linear

s

c

FFN

mt

mt

ys

ys

(a) Fusion type: “gated” (b) Fusion type: “concat”

- Figure 8 | Visualization of our mechanisms to fuse the main backbone’s features 𝑚(𝑥,𝑡) with the flow head input 𝑦𝑠: (a) “gated” fusion, where the flow head input 𝑦𝑠 is passed to an FFN block (conditioned on the flow head timestep 𝑠) and then added to the main feature 𝑚𝑡 via a gated operator; and (b) “concat” fusion, where the the flow head input 𝑦𝑠 and the main feature 𝑚𝑡 are concatenated in the channel dimension and then passed to the linear projection layer.

Our remaining design choices largely follow Geng et al. [17]. We set 𝑟 = 𝑠 for 75% of training batches to stabilize optimization under the flow-matching loss, apply condition dropout (using the negative prompt in Table 8) during training, and construct the velocity field using classifier-free guidance. We observed that mixing conditional and unconditional network-predicted velocities (Eq. (18) in [17]) offers no clear benefit in our setting, so we adopt the standard classifier-free guidance formulation throughout. Moreover, we also use an adaptive loss normalization, such that our final loss derived from (9) is given by

### A. Implementation details

#### A.1. Flow head conditioning

Time conditioning. For the flow head, we re-use the default time embedding to embed 𝑠 and instantiate an additional zero-initialized time embedding that embeds the difference 𝑠 − 𝑟 between two time steps of the flow map (see Section 2). The two embeddings are summed and used as the conditioning input to the adaptive normalization layers in the DiT blocks of the flow head.

[︃ ‖𝑢𝜃(𝑦𝑠,𝑠,𝑟) − 𝑢^‖2 sg

]︃,

E𝑠,𝑟,𝑦

(︀‖𝑢𝜃(𝑦𝑠,𝑠,𝑟) − 𝑢^‖2)︀

+ 𝑐

𝑠

where we choose 𝑐 = 𝑑 for Wan2.1 1.3B and 𝑐 = 10𝑑5 for Wan2.1 14B (where 𝑑 is the dimension of 𝑦𝑠).

Finite difference approximation. Computing the JVP term in the MeanFlow objective, namely

Inner flow fusion. To fuse the main backbone’s features 𝑚(𝑥,𝑡) with the flow head input 𝑦𝑠, we first process 𝑦𝑠 with the patch embedding layer of the main backbone. Then we process the patches using a randomly initialized AdaLN-style block (analogous to the ones used in the teacher model, i.e., consisting of adaptive layer normalization, an MLP, and a gated residual connection) conditioned on the embedding of 𝑠 and re-using the global scale and shift values. Finally, we fuse this processed flow head input with the main backbone’s features using a gated interpolation, controlled by a global gate derived from a sigmoidactivated learnable parameter. This makes the flow head’s contribution both time-aware and dynamically gated at the feature level before the final global fusion. We visualize the fusion mechanism in Figure 8 and provide an ablation w.r.t. other conditioning types in Section B.3.

d d𝑠𝑢𝜃(𝑦𝑠,𝑠,𝑟), using forward-mode automatic differentiation in PyTorch is currently incompatible with

system optimizations, such as flash attention and FSDP; however, these optimizations are crucial to avoid out-of-memory issues due to the long video sequence length. To address this and make our method agnostic of different training techniques, we approximate the JVP using a central difference scheme. For a step size 𝛿, we have

𝑢𝜃(𝑦𝑠+𝛿,𝑠 + 𝛿,𝑟) − 𝑢𝜃(𝑦𝑠−𝛿,𝑠 − 𝛿,𝑟) 2𝛿

d d𝑠𝑢𝜃(𝑦𝑠,𝑠,𝑟) ≈

with 𝑦𝑠±𝛿 = 𝑦𝑠 ± 𝛿𝑣(𝑦𝑠,𝑠), where we use the conditional velocity for 𝑣. At the boundaries, we fall back to using one-sided finite-difference approximations.

2Note that, in practice, we set 𝑡𝑀 = 𝑠𝑁 = 0.999 instead of 1 to align with the pretraining of Wan.

###### Hyperparameter Wan2.1 1.3B (T2V) Wan2.1 14B (T2V)

General Settings Resolution 480 × 832 (480P) Frame count 81 Frames (5s) Latent dimension (𝑇 × 𝐻 × 𝑊) 21 × 60 × 104 Dataset size 500k (479k after filtering) Dataset prompts VidProm extended by Qwen2.5-7B Dataset videos Generated by Wan2.1 14B (T2V) Optimizer (weight decay, betas, epsilon) AdamW (0.01, (0.9,0.99), 10−8) Global batch-size 64 Timestep shift 𝛾 for 𝑡student (see Section 3.2) 10 Multi-step sampling (see Section A.3) deterministic Precision BF16 (time 𝑡 in FP64) GPU type NVIDIA A100 NVIDIA H100 Parallelism Strategy DDP FSDP

Baselines KD Number of trajectories 10k CFG scale 5 Skip-layer 10 Learning rate 7e-5 Max. iterations 10k 5k

DMD2-v Fake score architecture same as teacher Discriminator architecture (see Section 4.3) Conv3D Discriminator parameters 68M 172M CFG scale 5 Layer for discriminator features (see Section A.3) (15,22,29) (19,29,39) Discriminator loss weight 𝜆 (see Algorithm 2) 0.03 Learning rates (student, discriminator, fake score) (10−5,10−5,10−5) Timestep shift 𝛾 for 𝑡dmd (see Section 3.2) 5 Min./max. for 𝑡dmd and 𝑠mf (0.001,0.999) Student update frequency every 5-th iteration Max. iterations 4k (𝑀 ≥ 2), 12k (𝑀 = 1) 1k (𝑀 ≥ 2), 5k (𝑀 = 1) TMD Stage 1: TM-MF CFG scale 3 Flow head fuse type (see Section A.1) gated Timestep shift 𝛾 for 𝑠student (see Section A.2) 10 Timestep shift 𝛾 for 𝑠mf (see Section A.2) 3 JVP finite-difference 𝛿 (see Section A.2) 5 · 10−3 Condition dropout probability (see Section A.2) 0.1 Loss normalization constant 𝑐 (see Section A.2) 𝑑 10−5𝑑 Learning rate 3 · 10−5 1 · 10−5 Max. iterations 3k 3k

Stage 2: DMD2-v with flow head All hyperparameters as in DMD2-v above

- Table 7 | Default hyperparameters if not specified otherwise in the experiments.

We empirically found that setting 𝛿 = 0.005 yields a satisfactory estimation to the JVP term and fix it throughout our experiments.

#### A.3. DMD2

Initialization. Since we consider video diffusion models that are trained to approximate the instantaneous velocity 𝑣 in (2), we parametrize the student network as

𝑔𝜃student(𝑥𝑡,𝑡) = 𝑥𝑡 − 𝑡𝑣𝜃(𝑥𝑡,𝑡) (17)

for our DMD2-v baseline, which initially approximates the conditional expectation E[𝑥|𝑥𝑡].

Multi-step distillation and inference. Compared to the original multi-step DMD2 [58], we did

not use backward simulation for the outer loop for multi-step distillation and efficiently sample 𝑥𝑡

using real data.

𝑖

Moreover, during inference, we use deterministic sampling from the conditional flow, i.e.,

= (︂1 −

)︂𝑥𝑡

𝑡𝑖+1 𝑡𝑖

𝑡𝑖+1 𝑡𝑖

𝑔student(𝑥𝑡

+

,𝑡𝑖), (18)

#### 𝑥𝑡

𝑖+1

𝑖

𝑖

which samples from the correct distribution as long as 𝑥 ≈ Student(𝑥𝑡

,𝑡𝑖). Compared to the standard resampling scheme in DMD2, i.e.,

𝑖

= (1 − 𝑡𝑖+1)𝑔student(𝑥𝑡

,𝑡𝑖) + 𝑡𝑖+1𝑥1, (19)

#### 𝑥𝑡

𝑖+1

𝑖

the noise 𝑥1 of the conditional flow is inferred from 𝑔student(𝑥𝑡

in Eq. (18) and sampled independently in Eq. (19).

,𝑡𝑖) and 𝑥𝑡

𝑖

𝑖

#### A.4. VBench evaluation

In the style of Vincent van Gogh, an astronaut with a helmet and spacesuit rides a cow on a sandy beach at sunset. The astronaut sits confidently on the cow, which is walking steadily towards the viewer.

[Figure 21]

[Figure 22]

[Figure 23]

We follow the official evaluation protocol in VBench [22] to test our method. During video sampling, we use standard VBench prompt lists to ensure fair comparisons among different methods. In particular, we rewrite the test prompts (i.e., prompt augmentation) using Qwen/Qwen2.5-7BInstruct [55], following the prior work [21]. Similarly, almost all the baselines, including Wan2.1 base models, also enhance VBench prompts, making them longer and more descriptive without altering their original meaning. We then calculate scores for 16 Text-to-Video (T2V) evaluation dimensions, respectively, and summarize them into quality score, semantic score and overall score.

0s 2.5s 5.0s

A cute, fluffy white dog running through a green meadow filled with wildflowers. The dog has bright, expressive brown eyes and a wagging tail. It is playing fetch with a red ball, jumping up to catch it in mid-air.

[Figure 24]

[Figure 25]

[Figure 26]

0s 2.5s 5.0s

Retro Ghibli-inspired scene, featuring a charming morning dove going about its daily life. The dove is depicted in a minimalist, flat 2D style with bold lines and minimal shading.

[Figure 27]

[Figure 28]

[Figure 29]

0s 2.5s 5.0s

In figures from the main text, we only show the short prompts due to the space limit. In Table 8, we show the corresponding extended prompts in Figures 1-4 that are actually passed to the model.

- Figure 9 | Mode collapse without time-shifting. We show videos generated by the one-step student distilled

from DMD2 in the setting “𝑡dmd w/o shift” (see Table 5), where the other hyperparameters use the default values. We can see that all generated videos have the main characters consistently appear on the left side of the pixel space, which is a sign of the severe mode collapse happening during distillation.

#### A.5. Hyperparameters

We provide an overview of our default hyperparameters in Table 7.

For TMD, the multi-step inference is given in Algorithm 1. In particular, the outer transitions are also deterministic, but additional independent noise 𝑦1 ∼ 𝒩(0,𝐼) is used for the inner flow.

### B. Additional experiments

#### B.1. Quality-efficiency tradeoff

Teacher, fake score, and discriminator. While it would be possible to distill Wan2.1 1.3B using the 14B model as teacher in DMD2, we use the 1.3B model for fair comparisons. Moreover, we use (unpatchified) teacher features at different layers as inputs to the discriminator, which we found more stable than using fake score features. The discriminator is trained using an average minimax log-likelihood objective over separate heads for each teacher feature, where the teacher is evaluated at noisy inputs (^𝑥𝑡,𝑡) (fake data

Extending the experiments in Section 4.3, Figure 11 shows TMD’s performance-efficiency tradeoff when varying both the number of outer and inner steps 𝑀 and 𝑁 and the flow head layers 𝐻. It shows a general trend of achieving better performance with a larger effective NFE and TMD offers a more fine-grained control over this performance-efficiency tradeoff than DMD2-v.

#### B.2. Curvature of Wan trajectories

- as defined in Section 2) or (𝑥𝑡,𝑡) (real data). The generator loss is given as the average (non-saturating) negative log-likelihood. We initialize the fake score 𝑔fake using the teacher parameters, parametrize it analogous to (17), and use denoising score matching

Inspired by Liu et al. [34], we define the curvature of Wan’s sampling trajectory at time 𝑡 as

𝐶(𝑣,𝑡) = ‖𝑣(𝑥𝑡,𝑡) − (𝑥 − 𝑥0)‖2.

to train it on noisy fake data (^𝑥𝑡,𝑡). Both the discriminator and fake score are trained for several iterations in between student updates. Finally, we can write our VSD objective in (11) as

Since the model is evaluated on a fixed time grid during sampling, we adopt a discretized version of the curvature in practice, i.e.,

2

𝐶(𝑣,𝑡𝑖) = ⃦𝑥˜𝑡𝑡𝑖−𝑥˜𝑡𝑖−1

𝑖−𝑡𝑖−1 − (𝑥1 − 𝑥)⃦

)︁𝑇𝑥^]︂,

[︂sg(︁ 𝑔fake(^𝑥𝑡,𝑡) − 𝑔teacher(^𝑥𝑡,𝑡) ‖𝑔fake(^𝑥𝑡,𝑡) − 𝑔teacher(^𝑥𝑡,𝑡)‖1

,

##### E𝑡

𝑖,𝑥𝑡𝑖,𝑡,𝑥^𝑡

where 𝑡𝑖 denotes the timesteps from Wan’s default 50-step sampling schedule and 𝑥˜𝑡

represents the corresponding intermediate samples along the trajectory.

where sg(·) denotes the stop-gradient operation, and the teacher is also parametrized analogous to Eq. (17), including CFG with the negative prompt given in Table 8.

𝑖

As shown in Figure 12, the sampling trajectories of the Wan model exhibit extremely large curvature

A romantic scene featuring a happy couple dancing salsa in the rain under a large colorful umbrella. They are dressed in elegant evening attire, with the man in a black suit and the woman in a flowing red dress. Both have joyful expressions as they move gracefully to the music, their feet barely touching the wet pavement. Raindrops fall softly around them, creating a misty atmosphere.

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

0s 2.5s 5.0s 0s 2.5s 5.0s

- (a) Two-step DMD2 w/o KD warm-up (Left: Iteration 0, Right: Iteration 1000)

- (b) Two-step DMD2 w/ KD warm-up (Left: Iteration 0, Right: Iteration 1000)

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

0s 2.5s 5.0s 0s 2.5s 5.0s

- Figure 10 | Effect of KD initialization. We compare the two-step DMD2 results of distilling Wan2.1 1.3B in two settings: (a) with and (b) without KD warm-up, where (left) “Iteration 0” means videos generated in the beginning of DMD2 training and (Right) “Iteration 1000” means videos generated after training DMD2 for 1k iterations. From left to right, we show the first, middle and the last frames in each video. We can see that the KD warm-up initially can generate better videos, but it also introduces coarse-grained artifacts. For instance, it generates an extra man besides a couple specified in the prompt. After training for 1k iterations, two-step DMD2 cannot remove these artifacts, leading to the worse generation quality than two-step DMD2 without KD warm-up.

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | |DM|N2H3 D2 v|N2H5<br><br>N4H2|N2H8 N4H3<br><br>|N<br><br>DM|4H5<br><br>D2 v|
| | | | | | | | | | |
| |N2H5|N4H3<br><br>N|4H5| | | | | | |
|DM|N2H3 N4H2<br><br>D2 v|N2H8<br><br>| | | | | | | |
| | | | | | | | | | |

1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00

Effective NFE

83.2

83.6

84.0

84.4

84.8

Overall Score

- Figure 11 | Performance-efficiency tradeoff of TMD. Extension of Figure 6 to include the 𝑀 = 1 settings.

near 𝑡 = 1 (i.e., the high noise regime), making it difficult for trajectory matching methods to learn the mapping along the ODE path. This also motivates our choice of larger 𝛾 in the time-shifting function for 𝑟𝑖 in TM-MF (see Section A.2) and 𝑡student in DMD2-v (see Section 3.2).

#### B.3. Flow head conditioning

Our main experiments use the gating mechanism explained in Section A.1 to condition the flow head on the main backbone’s features 𝑚 = 𝑚(𝑥,𝑡). Another version is to concatenate 𝑚 and the patch embeddings of 𝑦𝑠 along the hidden dimension and then project to the original hidden size using a linear layer. To minimize the impact on the pretrained model, we initialized the weights of the linear layer as identity for the coordinates corresponding to 𝑚 and as a Gaussian with small standard deviation (0.01) for the remaining coordinates. In Table 9, we see that TMD can achieve strong performance across fusion types.

While convergence was more stable using the gating mechanism used for our main experiments (see Figure 13), concatenation is a strong alternative in terms of final performance.

Overall Quality Semantic

Setting Fusion type

score score score N2H5

Gated (see Section A.1) 84.68 85.71 80.55 Concat 84.76 85.77 80.71

Gated (see Section A.1) 84.67 85.72 80.47 Concat 84.66 85.79 80.14

N4H5

Table 9 | Impact of the fusion type on the final performance when distilling Wan2.1 1.3B with 𝑀 = 2.

#### B.4. Inner flow targets

In principle, the auxiliary latent variable 𝑦 in the inner flow can be chosen arbitrarily as long as 𝑥𝑡

is easy to sample given 𝑦 and 𝑥𝑡

𝑖−1

. While we chose the

𝑖

- 0

- 1

- 2

- 3

- 4

- 5

- 6

Curvature

0.2 0.4 0.6 0.8 1.0 Timestep

- Figure 12 | Curvature of Wan trajectories. Curvature of different sampling trajectories for the Wan2.1 1.3B model. Large trajectory curvatures are observed near 𝑡 = 1 (i.e., the high noise regime).

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

500 1,000 1,500 2,000 2,500 3,000 3,500 4,000

Iteration

50.0

60.0

70.0

80.0

Overall Score

N2H5 N2H5 concat

- Figure 13 | Convergence and fusion ablation. We compare the overall VBench score over iterations for the second-stage TMD training with our gating mechanism (see Section A.1) and concatenation of features (see Section B.3). While concatenation yields very competitive results, we observed that it introduces some training instabilities.

DTM formulation 𝑦 := 𝑥1 − 𝑥 in (3) for our main experiments, we also experimented with other versions. While other formulations did not improve the overall performance, they could still achieve competitive performance. For instance, Table 10 provides results for the simple choice 𝑦 := 𝑥. Similar to (14) and (17), we parametrize the average velocity as

𝑢𝜃(𝑦𝑠,𝑠,𝑟;𝑚) :=

𝑦𝑠 − (𝑥𝑡

𝑖 − 𝑡𝑖head𝜃(𝑦𝑠,𝑠,𝑟;𝑚)) 𝑠

.

where 𝑚 = 𝑚𝜃(𝑥𝑡

𝑖

,𝑡𝑖) denote the main features.

- B.5. Impact of recurrence in flow head

To evaluate the effect of recurrence in the flow head, we manually restrict the inner flow to a single step

- at inference (N1H5) and compare the results with the standard recurrent setting (N4H5). As shown in

- Figure 14, when distilling Wan2.1 1.3B into a twostep generator, the non-recurrent variant produces noticeably lower-quality videos, exhibiting stronger

artifacts and blurriness. This highlights the importance of iterative refinement within the flow head for high-fidelity generation.

Overall Quality Semantic score score score

Setting Target 𝑦

N2H5 𝑥1 − 𝑥 (DTM) 84.68 85.71 80.55 𝑥 84.18 85.15 80.33

N4H5 𝑥1 − 𝑥 (DTM) 84.67 85.72 80.47 𝑥 84.44 85.36 80.77

- Table 10 | Impact of the inner flow target when distilling Wan2.1 1.3B with 𝑀 = 2.

- B.6. DMD2 comparisons

In Section 3.2, we observed that KD pretaining is only helpful for one-step distillation (see Table 4 and Figure 9) and that timestep shifting of 𝑡dmd and 𝑡student is crucial for good performance for one- and two-step DMD2-v (see Table 5 and Figure 10). In this section, we provide additional ablations on our DMD2-v baseline when using ≥ 3 steps. While the timestep shifting for 𝑡dmd and 𝑡student remain important, we observed that slightly lower shift values for 𝑡student can attain better scores when using more steps. In particular, for 4 steps, we picked a shift value of 𝛾 = 5 for 𝑡student, outperforming the DMD2 baseline in [69]; see Table 11.

Method shift 𝑡student NFE

Overall Quality Semantic score score score

DMD2 [69] 4 84.56 85.58 80.50 DMD2-v 10 4 84.53 85.95 78.84 DMD2-v 5 4 84.60 86.03 79.87

DMD2-v 10 3 84.48 85.71 79.58 DMD2-v 5 3 84.47 85.55 80.15

Table 11 | VBench results of the DMD2 version in [69] and our DMD2-v for 3 and 4 steps with different timestep shifts for 𝑡student.

- B.7. More visual comparison results

We provide further visual comparisons between the 50-step teacher models with classifier-free guidance (CFG), DMD2-v, and TMD in Figures 15 to 20.

A joyful child, with a big smile and arms spread wide, swings energetically on a rusty old swing set in a sunlit backyard. The swing set, with peeling paint and creaking chains, contrasts against the vibrant green grass and blooming flowers surrounding it. The child's laughter echoes as they swing higher and higher, their feet barely touching the ground at the bottom of each arc.

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

N4H5

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

N1H5

0s 1.6s 3.3s 5.0s

Impressionist style, a single yellow rubber duck gently floating on undulating waves at sunset. The duck has a cheerful smile painted on its face, with sunlight casting warm, golden hues across the water. The sky is filled with soft, pastel shades of orange, pink, and purple, reflecting off the rippling surface. The water appears calm yet lively, with subtle waves carrying the duck along smoothly.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

N4H5

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

N1H5

0s 1.6s 3.3s 5.0s

- Figure 14 | Impact of flow head recurrence. We show the impact of recurrence in the flow head by setting the number of flow head steps to 1 only at inference (i.e., N1H5) when distilling Wan2.1 1.3B with 𝑀 = 2 and the N4H5 setting for flow head (i.e., 4 denoising steps and 5 DiT blocks in flow head). We observe that the videos generated without recurrence (marked by N1H5) are of much lower quality (e.g., more artifacts and blurriness) than ones with recurrence (marked by N4H5), implying the importance of the fine-grained iterative refinement on our method.

Figure 1 Short prompt A fat rabbit wearing a purple robe walking through a fantasy landscape.

Long prompt A plump, fluffy rabbit donning a voluminous purple robe walks gracefully through a vibrant fantasy landscape. The rabbit has large, expressive eyes and a gentle, curious expression. Its fur is soft and thick, and the robe drapes elegantly over its body. The landscape features rolling hills covered in lush green grass, colorful wildflowers, and towering magical trees with shimmering leaves. In the distance, there are sparkling waterfalls and mystical castles. The scene is bathed in warm, golden sunlight. Medium shot, focusing on the rabbit’s walk through the picturesque environment.

Short prompt A person drinking coffee in a cafe.

Long prompt A cozy, warm café setting with soft ambient lighting and wooden furnishings. A young adult, casually dressed in a sweater and jeans, sits at a small round table. They hold a steaming cup of coffee in their hand, taking a sip while looking pensively out the window. The café is moderately busy with other patrons engaged in conversations. The background showcases various coffee drinks and pastries displayed on a counter. The person’s expression is relaxed and content. Medium shot focusing on the person’s face and the coffee cup, capturing the intimate atmosphere of the café.

- Figure 3 Short prompt A person is laughing. Long prompt A joyful person is laughing heartily, with a broad smile and crinkled eyes, conveying pure happiness. They are

standing upright with arms spread wide, as if embracing the world around them. The scene is set outdoors in a sunny park, surrounded by lush greenery and blooming flowers. The background includes a clear blue sky with fluffy clouds, adding to the cheerful atmosphere. Medium shot capturing the full body of the person, focusing on their animated facial expressions and gestures.

Short prompt A steam train moving on a mountainside.

Long prompt A vintage steam locomotive chugging along a winding track on a mountainous terrain. The train is covered in soot, with steam billowing from its smokestack as it navigates the rugged landscape. The surrounding mountains are steep and lush, with patches of snow visible at higher elevations. The train cars sway gently as they follow the curving tracks, and the scenery outside the windows shows dense forests and rocky cliffs. The camera follows the train from a medium distance, capturing the train’s movement and the dramatic backdrop.

- Figure 4 Short prompt A boat sailing leisurely along the Seine River with the Eiffel Tower in background. Long prompt A serene, picturesque scene of a small wooden boat gently gliding along the Seine River in Paris, France. The

boat is rowed leisurely by a middle-aged man in a casual striped shirt and khaki pants, who rows smoothly with rhythmic strokes. The Eiffel Tower stands majestically in the background, partially visible through the misty morning air. The riverbank is lined with lush green trees and quaint buildings, reflecting off the calm waters. The overall atmosphere is peaceful and tranquil, capturing the essence of a lazy summer day. Wide shot, static camera.

Short prompt An astronaut flying in space, zoom out.

Long prompt Astronaut floating in space with a helmet visor reflecting Earth below. The astronaut, wearing a full spacesuit with the American flag on the shoulder, is performing a spacewalk, arms extended as if in motion. The background shows the vastness of space with stars twinkling and Earth in the distance. The scene begins with a close-up of the astronaut and gradually zooms out to reveal the enormity of space surrounding them. Wide shot, showcasing the astronaut against the backdrop of the universe.

Negative prompt

Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards

- Table 8 | Extended prompts used in the figures from the main text and negative prompt used for CFG (in teacher sampling, KD, and DMD2-v) and condition dropout (in TM-MF), taken from the official Wan repository.

A person enjoying a juicy burger, with a satisfied smile on their face. They are seated at a casual dining table, surrounded by napkins and a drink. The burger is topped with lettuce, tomato, and cheese, and the person is taking a bite, showcasing the delicious layers inside. The scene has a warm, inviting atmosphere with soft lighting and a cozy background. Medium close-up shot focusing on the person’s hand holding the burger and their facial expressions as they savor each bite.

[Figure 58]

A person is skydiving from a plane, descending towards the ground. They are mid-air, arms spread wide, with a parachute deployed and open, ensuring a smooth descent. The skydiver is wearing a jumpsuit and helmet, with a determined and exhilarated expression on their face. The background shows a clear blue sky with fluﬀy clouds and the landscape below stretching out, including patches of green fields and distant mountains. The scene captures the moment just after exiting the plane, with the parachute fully inflated, showcasing the thrill and freedom of skydiving. Mid-shot, focusing on the skydiver against the expansive sky.

[Figure 59]

- Figure 15 | Visual comparison on Wan2.1 14B. We compare the outputs of the teacher, TMD, and DMD2-v on exemplary prompts.

A serene African savanna scene with tall grasses and scattered trees. A tall giraﬀe, with distinctive brown spots on its creamy white coat, bends its long neck gracefully to drink from a calm river. The giraﬀe’s gaze is focused intently on the water as it lowers its head, revealing its long eyelashes and gentle expression. The river reflects the golden hues of the late afternoon sun, casting a warm glow over the scene. The background shows the vast expanse of the savanna with distant hills. The video is a medium close-up, capturing the giraﬀe’s elegant movement and the tranquil environment.

[Figure 60]

A large great white shark is swimming gracefully through the vast, deep blue ocean. Its sleek, muscular body cuts through the water as it propels forward with powerful tail strokes. The shark’s dorsal fin slices through the surface, while smaller fish dart around it. The camera begins at a wide shot of the shark and the surrounding ocean, then smoothly zooms in to focus closely on the shark’s sharp teeth and piercing eyes. The scene is filled with sunlight filtering through the water, creating a dynamic interplay of light and shadow. Close-up underwater perspective.

[Figure 61]

- Figure 16 | Visual comparison on Wan2.1 14B. We compare the outputs of the teacher, TMD, and DMD2-v on exemplary prompts.

A happy, playful Corgi running and jumping in a park during sunset, captured in black and white. The Corgi has a friendly face with floppy ears and a wagging tail as it moves through the grassy area. The sky behind the dog shows soft gradients of orange and pink fading into shades of gray and black. The park includes a few trees and benches in the background, adding depth to the scene. The Corgi is in motion, emphasizing its joyful playfulness. Medium close-up shot, focusing on the Corgi’s expressive face and body language.

[Figure 62]

A stormtrooper from the Star Wars universe, clad in pristine white armor with a black helmet, is meticulously vacuuming a sandy beach. He bends down slightly, moving the vacuum cleaner back and forth across the sand with purposeful motions. His gloved hand firmly grips the handle of the vacuum as he navigates around rocks and debris. The sun sets behind him, casting long shadows and giving the scene a dramatic, golden glow. The background shows crashing waves and seagulls flying overhead. Medium close-up shot, focusing on the stormtrooper’s actions and the sweeping motion of the vacuum.

[Figure 63]

- Figure 17 | Visual comparison on Wan2.1 14B. We compare the outputs of the teacher, TMD, and DMD2-v on exemplary prompts.

A person is roller skating in an urban park, moving smoothly across the paved path. They wear a black helmet with a visor and knee pads, elbow pads, and wrist guards for safety. The skater has medium-length brown hair tied back in a ponytail and wears a bright yellow shirt with a graphic design and black shorts. They maintain a slight crouch posture as they glide, their feet making fluid movements, propelling them forward eﬀortlessly. The background shows other park-goers walking dogs and children playing, adding a lively atmosphere. Medium shot focusing on the skater from a side angle, capturing the motion and environment.

[Figure 64]

A koala bear playing a grand piano in a lush, dense forest. The koala has soft, grey fur and large, round ears. It sits upright on the piano bench, paws delicately placed on the keys, creating gentle melodies. The forest background is filled with tall eucalyptus trees, dappled sunlight filtering through the leaves, and a carpet of green moss beneath the piano. The scene is calm and serene, with the koala focused intently on its performance. Medium close-up shot, capturing the koala and part of the forest surroundings.

[Figure 65]

- Figure 18 | Visual comparison on Wan2.1 1.3B. We compare the outputs of the teacher, TMD, and DMD2-v on exemplary prompts.

A large, hairy Bigfoot creature walking through a heavy snowstorm. The Bigfoot stands at least eight feet tall, covered in shaggy brown fur, with muscular limbs and a stooped posture. Snowflakes swirl around him as he moves slowly and deliberately through the dense forest, his feet sinking slightly into the deep snow. The landscape is bleak and desolate, with bare trees and thick snowdrifts. The atmosphere is eerie and quiet, with only the sound of crunching snow and howling wind. The scene is captured in a mid-shot, focusing on the Bigfoot’s powerful form as he trudges through the storm.

[Figure 66]

An animated scene of a panda drinking coﬀee in a cozy café in Paris. The panda is sitting at a small table with a steaming cup of coﬀee, holding a spoon delicately. The café has vintage decor with wooden furniture, soft lighting, and a few other patrons in the background. The panda has a relaxed and content expression, sipping the coﬀee slowly. The atmosphere is warm and inviting, with the soft hum of conversation in the background. Medium shot focusing on the panda and the coﬀee cup.

[Figure 67]

- Figure 19 | Visual comparison on Wan2.1 1.3B. We compare the outputs of the teacher, TMD, and DMD2-v on exemplary prompts.

A dramatic and intense scene featuring an erupting volcano. The volcano is spewing lava and ash into the air, creating a vivid orange glow against a dark night sky filled with billowing smoke clouds. The ground trembles as molten rock flows down the sides of the volcano, lighting up the surrounding landscape. In the foreground, a few scattered trees and rocks are illuminated by the fiery eruption. The camera remains fixed on the volcano, capturing the powerful motion and scale of the event. Nighttime, wide shot.

[Figure 68]

A realistic classroom scene set during a typical school day. The classroom has rows of desks facing a chalkboard at the front. Students are engaged in various activities; some are reading books, others are writing in notebooks, and a few are quietly talking. The teacher stands at the front of the class, holding a book and addressing the students. The room is well-lit with sunlight streaming in from large windows, casting soft shadows across the desks. The walls are adorned with educational posters and motivational quotes. Medium shot capturing the full classroom environment.

[Figure 69]

- Figure 20 | Visual comparison on Wan2.1 1.3B. We compare the outputs of the teacher, TMD, and DMD2-v on exemplary prompts.

[Figure 70]

###### Figure 21 | User study interface. Screenshot of our user preference study interface explained in Section 4.2.

