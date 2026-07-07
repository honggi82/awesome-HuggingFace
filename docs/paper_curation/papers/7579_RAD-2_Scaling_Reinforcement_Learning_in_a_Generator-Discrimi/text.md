[Figure 1]

[Figure 2]

## RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework

# arXiv:2604.15308v1[cs.CV]16Apr2026

Hao Gao1 Shaoyu Chen2,† Yifan Zhu2 Yuehao Song1 Wenyu Liu1 Qian Zhang2 Xinggang Wang1,

1 Huazhong University of Science & Technology 2 Horizon Robotics

Project page: https://hgao-cv.github.io/RAD-2

### Abstract

High-level autonomous driving requires motion planners capable of modeling multimodal future uncertainties while remaining robust in closed-loop interactions. Although diffusion-based planners are effective at modeling complex trajectory distributions, they often suffer from stochastic instabilities and the lack of corrective negative feedback when trained purely with imitation learning. To address these issues, we propose RAD-2, a unified generatordiscriminator framework for closed-loop planning. Specifically, a diffusion-based generator is used to produce diverse trajectory candidates, while an RL-optimized discriminator reranks these candidates according to their long-term driving quality. This decoupled design avoids directly applying sparse scalar rewards to the full high-dimensional trajectory space, thereby improving optimization stability. To further enhance reinforcement learning, we introduce Temporally Consistent Group Relative Policy Optimization, which exploits temporal coherence to alleviate the credit assignment problem. In addition, we propose On-policy Generator Optimization, which converts closed-loop feedback into structured longitudinal optimization signals and progressively shifts the generator toward high-reward trajectory manifolds. To support efficient large-scale training, we introduce BEV-Warp, a high-throughput simulation environment that performs closed-loop evaluation directly in Bird’s-Eye View feature space via spatial warping. RAD-2 reduces the collision rate by 56% compared with strong diffusion-based planners. Real-world deployment further demonstrates improved perceived safety and driving smoothness in complex urban traffic.

## 1 Introduction

Achieving robust, safe, and human-like motion planning is a key goal in high-level autonomous driving systems. A variety of learning-based approaches have been explored for

† Project leader. Corresponding author.

- a) Stabilizing RL Optimization: Bridging High-Dimensional Trajectory with Low-Dimensional Reward

Low-Dim

Scalar Reward

High-Dim Physically-

Constrained Trajectory

Direct Gradient Update

Unstable Optimization

Longitudinal

Component

Shape-Preserving

Discriminator Extraction

Low-Dim

Trajectory Score

Stable Optimization Stable Optimization

- b) Scaling RL with Efficient Closed-Loop Simulation: Realistic Environment and High-Throughput Training

[Figure 3]

Limitations:

Limitations:

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Per-scene reconstruction

Sim-to-real gap

[Figure 8]

[Figure 9]

Heavy simulation pipeline

Naive actor behavior

Game-engine Simulator Reconstruction-based Simulator

Limitations: Slow multi-view generation Long-horizon rollout drift

Advantages: Higher feature fidelity Efficient and lightweight

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Generative Simulator

BEV-Warp Simulator

Figure 1. Scaling RL in a Generator-Discriminator framework. (a) Stabilizing RL Optimization: RAD-2 projects the highdim trajectory space into low-dim score and longitudinal components, ensuring stable policy updates. (b) Efficient Simulation: BEV-Warp enables high-throughput, feature-level closedloop training, overcoming the limitations of existing simulators.

this purpose [3, 19, 47]. Regression-based planners [12, 16, 49] predict trajectories deterministically, thus collapsing multimodal behaviors and producing mean-biased outputs. Selection-based planners [2, 24, 30, 32] rely on a discrete set of candidates, limiting their ability to represent the full range of feasible trajectories. To address these limitations, diffusion-based imitation learning (IL) planners [36, 61, 63, 67] have emerged as a promising approach in autonomous driving and embodied AI, providing a generative framework capable of modeling multimodal continuous trajectories and adapting them to the diverse conditions encountered in complex driving scenarios. An architectural comparison of different multimodal planning paradigms is provided in Fig. 2.

However, diffusion-based IL planners still face key chal-

a) Vocabulary-based Trajectory Scoring

b) Diffusion-based Trajectory Generation

c) Ours: Diffusion-Generator with RL-Discriminator Synergy

[Figure 16]

[Figure 17]

Sensor Data

Sensor Data

Noise

Sensor Data

Noise

Planning Vocabulary

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Human Driving IL

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

Demonstrations

Encode

Encode

Sample

Encode

Encode

Sample

[Figure 32]

Generated

Denoise

Trajectory Decoder

Diffusion Decoder

Diffusion Decoder

Trajectory

[Figure 33]

[Figure 34]

[Figure 35]

Score

Denoise & Score

Scene

p=1 p=0

[Figure 36]

Confidence Score

Human Driving

IL IL

Human Driving

p=1 p=0

[Figure 37]

Representation

[Figure 38]

Confidence Score

Demonstrations

Demonstrations

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

Open-loop

Open-loop

Close-loop

RL

RL Score

RL

Trajectory

simulator

simulator

simulator

Discriminator

[Figure 43]

[Figure 44]

Score of

Generated

RL

Each Anchor

Trajectory

Information Flow Optional (Method-Dependent)

Figure 2. The comparison of multimodal trajectory planning paradigms. (a) Vocabulary-based scoring with fixed anchors [2, 30]. (b) Diffusion-based multimodal generation [36]. (c) RAD-2: synergizing diffusion generator with RL-discriminator in closed-loop training.

lenges when applied to real-world autonomous driving. Real driving datasets contain noise and uneven distribution, which leads the diffusion model to learn certain regions of the trajectory distribution less effectively, resulting in occasional low-quality or unstable trajectories [5, 21]. This is especially critical for safety-sensitive planning. The high dimensionality of continuous trajectories further complicates optimization [26, 62], and imitation-only training provides no negative feedback to suppress unrealistic behaviors. In addition, IL introduces structural limitations: it suffers from causal confusion, learning correlations between states and actions instead of the underlying causal factors, which can lead to shortcut behaviors, and its openloop training paradigm causes a mismatch with the closedloop nature of real-world driving [7, 39].

Several recent works have explored combining reinforcement learning (RL) with imitation learning (IL) to improve policy learning [7, 13, 29, 35, 38, 39, 66]. While RL provides task-level feedback beyond imitation, directly applying it to generative planners is challenging. The reward signal in RL is usually a low-dimensional scalar, whereas the action space involves high-dimensional, temporally structured trajectories, complicating credit assignment and making optimization unstable [37]. RL training for autonomous driving requires closed-loop interaction, usually in simulation for safety and cost reasons. However, despite recent advances, existing simulation frameworks still exhibit several practical limitations. Game-engine simulators [4, 6] often simplify agent behavior, reconstructionbased simulators [7, 20, 40, 64] are complex and costly, and learned world models [23, 50–53] struggle with longhorizon, multi-view generation. These limitations pose practical challenges for scaling RL to high-dimensional trajectory optimization.

To scale RL training, we propose a unified architecture that integrates a generator–discriminator design with a scalable simulation environment, termed BEV-Warp. As illustrated in Fig. 1, this design enables stable policy updates and efficient closed-loop interaction.

Instead of directly optimizing the high-dimensional output of generator using sparse scalar rewards, which is challenging, we restrict reinforcement learning to the discriminator, whose output space naturally aligns with the lowdimensional reward signal. Trained via closed-loop interaction, the discriminator learns to evaluate trajectories based on long-term outcomes, progressively improving its ability to distinguish high-quality trajectories from undesirable ones, thereby enhancing safety, efficiency, and comfort.

Improving system performance requires both accurate trajectory evaluation and refinement of the trajectory distribution. While the discriminator prioritizes higher-quality candidates, the generator determines the diversity and quality of behaviors that can be realized. To this end, we introduce On-policy Generator Optimization (OGO). Rather than applying reinforcement learning to the highdimensional trajectory space, OGO constrains the optimization to the longitudinal component, where scalar rewards can optimize the policy without compromising stability.

The framework forms a self-improving closed loop. The discriminator D evaluates candidate trajectories, prioritizing higher-quality behaviors, while the generator G refines its trajectory distribution through on-policy optimization. By operating on the same interaction data, the two components achieve efficient joint optimization without additional simulation cost. Formally, let τ denote a future trajectory. The diffusion generator Gθ(τ|o), parameterized by θ, models a broad distribution of feasible actions conditioned on the context o. The RL-trained discriminator Dϕ(τ|o,C),

parameterized by ϕ, provides a reranking distribution over a set of candidate trajectories C = {τ1,...,τM} sampled from the generator, i.e., C ∼ Gθ(·|o). These two components define the joint policy distribution as Πθ,ϕ(τ|o) = EC∼G

θ(·|o)[Dϕ(τ|o,C)], aligning with the probabilistic inference framework for optimal control [22]. This architecture inherently supports inference-time scaling [15], where increasing the sample count M allows the discriminator to explore a denser action space and identify higher-quality solutions without retraining. Through iterative optimization, the generator and discriminator jointly optimize the overall policy, progressively shifting the distribution toward safer and more efficient behaviors. This mutual improvement raises the system’s performance upper bound without relying on additional expert supervision.

In contrast to Large Language Models [1, 8], autonomous driving involves high-dimensional continuous action spaces characterized by weak instantaneous rewardaction correlations. This discrepancy leads to a severe credit assignment problem, as sparse scalar rewards fail to effectively distinguish which specific variations within a sampled group contribute to superior outcomes [25]. To address this, we propose a Temporally Consistent (TC) sampling and optimization paradigm. During rollout collection, we employ a latched execution strategy, where a selected trajectory is reused over a fixed horizon to ensure behavioral coherence. Building upon this, we restructure the sampling process to enforce temporal dependencies across consecutive decision steps, ensuring that candidate trajectories are evaluated within a persistent behavioral context. This structured sampling allows us to introduce Temporally Consistent Group Relative Policy Optimization (TCGRPO), which effectively denoises the advantage signals and stabilizes the policy gradient. Together, these mechanisms enable the generator–discriminator system to iteratively optimize trajectory quality and enhance closed-loop planning performance.

The contributions of RAD-2 are summarized as follows:

- • We propose a unified generator-discriminator framework that stabilizes motion planning by decoupling diffusionbased trajectory exploration and RL-based reranking, ensuring robust performance in complex scenarios.
- • We introduce a joint policy optimization strategy that integrates TC-GRPO to ensure temporally consistent candidate reranking and OGO to iteratively shift the trajectory distribution via structured longitudinal optimizations.
- • We develop an RL pipeline based on BEV-Warp, a highthroughput, feature-level simulation environment leveraging spatial equivariance for efficient policy iteration. RAD-2 reduces the collision rate by over 56% on largescale benchmarks and significantly improves perceived safety during real-vehicle testing, yielding stable and comfortable planning behaviors.

## 2 Related Work

### 2.1 Discriminator for Autonomous Driving

Trajectory scoring and selection have emerged as pivotal techniques for enhancing the reliability of autonomous driving systems [9, 36, 46, 63, 67]. Early works like VADv2 [2] and Hydra-MDP [24, 30] rely on predefined trajectory vocabularies or rule-based teachers to guide selection. DriveSuprim [54] further refines this paradigm with a coarse-to-fine filtering framework combined with self-distillation. Recent advances such as DriveDPO [43] and GTRS [33] incorporate preference optimization and dynamic candidate evaluation to improve flexibility. However, these discriminative approaches typically operate in an open-loop manner, often resulting in suboptimal decisions because they neglect long-term downstream consequences and are constrained by the diversity of discrete candidate sets. In contrast, RAD-2 synergizes a continuous diffusionbased generator with a closed-loop trained discriminator, enabling robust planning over extended horizons by evaluating a more expressive manifold of future possibilities.

### 2.2 RL for Autonomous Driving

Reinforcement learning (RL) [42, 44, 55, 59] has been widely explored to mitigate the causal confusion and poor generalization issues of imitation learning. While recent works integrate RL with 3DGS-based digital twins [7], reasoning-oriented fine-tuning [18], or GRPO-based generation [29, 34, 66], optimizing high-dimensional driving outputs (e.g., raw trajectories) under sparse rewards remains notoriously difficult due to severe credit assignment challenges [13, 29, 35, 38, 39, 66]. Unlike these direct optimization approaches, we utilize RL rewards to train a lowdimensional trajectory discriminator, effectively reformulating the complex planning task into a tractable preference learning problem. This decoupling, empowered by TCGRPO, leverages temporal coherence as a physical prior to stabilize the RL search space and ensure behavioral consistency. Finally, the generator is iteratively optimized via Onpolicy Generator Optimization to align with high-reward manifolds, achieving a superior balance between exploration and learning stability.

### 2.3 Closed-loop Simulation Environment

Closed-loop simulation is fundamental for the training and validation of RL-based driving policies. Traditional simulators like CARLA [4] and SMARTS [65] offer interactive environments but often suffer from significant sim-toreal gaps due to their reliance on game engines. To bridge this gap, reconstruction-based simulators such as RAD [7] and ReconDreamer-RL [40] leverage 3D Gaussian Splatting (3DGS) [20] and video-diffusion priors to provide photorealistic training feedback. Furthermore, generative world

models [11, 28, 48, 57, 58] have been developed to synthesize future driving scenes or Bird’s-Eye View (BEV) representations [31, 41] for trajectory evaluation. While generative approaches offer high fidelity, they are often computationally intensive and susceptible to cumulative temporal drift. To address these limitations, we introduce BEV-Warp, which enables high-throughput simulation by directly warping BEV features around the ego vehicle thereby bypassing the expensive image-level rendering process.

## 3 Method

### 3.1 Generator-Discriminator Framework

As illustrated in Fig. 2 (c), our framework decomposes trajectory planning into two components: a diffusion-based generator that produces a diverse set of candidate trajectories, and a discriminator that evaluates and reranks these candidates. Together, they define a structured policy where generation and evaluation are jointly optimized.

#### 3.1.1 Diffusion-based Generator

The generator models a multimodal distribution over future trajectories and produces a set of candidate trajectories conditioned on the current observation ot, formally Gθ(τ | ot). Scene Encoding. The observation ot is first encoded into BEV features Tb, capturing the spatial layout of the scene. Static scene elements (e.g., lane geometry, road boundaries, and other map structures) and dynamic agents (e.g., surrounding vehicles and pedestrians with their motion states) are extracted from this representation to provide a comprehensive understanding of the environment. Let Xmap, Xagent, and Xnav denote the predicted static map elements, dynamic agents, and provided navigation inputs (e.g., waypoints or reference path), respectively. Token embeddings are then obtained via lightweight encoders:

Tm = Em(Xmap), Ta = Ea(Xagent), Tn = En(Xnav). (1)

These embeddings are fused with Tb through a learnable module F(·) to obtain the unified scene embedding:

Escene = F(Tb,Tm,Ta,Tn), (2)

which conditions the DiT-based trajectory generator via cross-attention.

Trajectory Generation. For M independent modes, an initial noise trajectory τ(0,m) ∼ N(0,I) is iteratively denoised for K steps:

##### τ(k,m) = G τ(k−1,m),Escene,k , k = 1,...,K, (3)

where G denotes the conditional denoising network [10] that predicts the next trajectory sample given the current

noisy sample τ(k−1,m), the scene embedding Escene, and the current step k. The final set of candidate trajectories is

T = τ ˆtm:t+H Mm=1, τˆtm:t+H = τ(K,m) ∼ Gθ(τ | ot),

(4) where τˆtm:t+H is a continuous (x,y) trajectory over the planning horizon H. These trajectories define the candidate set of the policy and are passed to the discriminator for evaluation and selection.

#### 3.1.2 RL-based Discriminator

The discriminator evaluates candidate trajectories based on their expected outcomes under the current scene, providing a preference over the candidate set.

Trajectory Encoding. Each point in the trajectory is embedded via a shared MLP, yielding a sequence of embeddings {ei}Hi=1. The trajectory sequence, prepended with a learnable [CLS] token, is processed by a Transformer encoder. The [CLS] output aggregates information from the entire trajectory and serves as the trajectory-level query Qτ. Scene Conditioning. The discriminator constructs a scene representation from the same inputs Xmap and Xagent as the generator. Its static and dynamic encoders, E∗m and E∗a, share the same architecture as the corresponding generator encoders Em and Ea, but maintain independent parameters:

Tm∗ = E∗m(Xmap), Ta∗ = E∗a(Xagent). (5)

Trajectory–Scene Interaction. The trajectory query Qτ aggregates multi-source scene context via cross-attention Ψ(Q,KV ):

Om =Ψ(Qτ,Tm∗ ),Ob = Ψ(Qτ,Tb),Oa = Ψ(Qτ,Ta∗), Ta∩m = Ψ(Ta∗,Tm∗ ),Oa∩m = Ψ(Qτ,Ta∩m).

(6) These embeddings are aggregated as Efusion:

Efusion = Φ([Om;Ob;Oa;Oa∩m]), (7) where Φ(·) denotes concatenation followed by an MLP.

Trajectory Scoring and Reranking. For each candidate trajectory τˆt:t+H ∈ T , the discriminator produces a scalar score via the sigmoid activation σ applied to the fused representation:

s(ˆτt:t+H) = σ(Efusion) ∈ [0,1]. (8)

Optionally, the scores can be normalized across the candidate set to induce a distribution over trajectories for reranking. In our implementation, we directly use the sigmoid outputs for prioritization.

### 3.2 Closed-Loop Simulation Environment and Controller

The proposed policy is evaluated in an efficient BEV-Warp environment, which enables high-throughput interaction by transforming BEV features. To verify the generalization across varied scene representations, validation is also conducted in a 3D Gaussian Splatting (3DGS) environment.

BEV-Warp Environment. BEV-Warp constructs a simulation environment by directly manipulating BEV features over time, bypassing redundant image-level rendering. The simulation is initialized from recorded real-world sequences, including multi-view observations, ego-vehicle trajectories, and scene context.

For each simulation step t, the system extracts the BEV feature Btref and loads the recorded pose Pt ∈ SE(2) as the state. The planner generates candidate trajectories T , from which an optimal trajectory τˆ∗ is selected and tracked to update the agent’s simulated pose Pt+1.

As illustrated in Fig. 3, to realign the logged environment with the simulated ego-state, we derive a warp matrix Mt+1 = (Pt+1)−1Ptref+1 ∈ R3×3. Assuming constant altitude and neglecting rotations (i.e., pitch and roll), the synthesized feature for the next timestep is obtained via:

Bt+1 = W Btref+1,Mt+1 , (9)

where W(·) is implemented via bilinear interpolation [14]. This synthesized representation, coupled with Pt+1, enables a continuous interaction for closed-loop evaluation.

3DGS Environment. To ensure performance consistency across different scene representations, the policy is further validated in a photorealistic simulation world built via 3D Gaussian Splatting (3DGS). Unlike the feature-level warping, this setup renders raw multi-view observations ot directly from the scene G using the agent’s current pose Pt: ot = R(G,Pt), where R(·) is the differential rendering operator. This serves as a testbed for verifying the policy’s robustness against explicit image-based rendering.

Trajectory-Based Controller. Both environments employ an iLQR-based controller [27] to track the planned trajectory τˆ∗. Given the non-linear vehicle dynamics xk+1 = f(xk,uk) (e.g., a kinematic bicycle model), the controller minimizes a quadratic cost function over the horizon H:

t+H

u∗t:t+H = arg min

u

k=t

∥xk − xˆrefk ∥2Q + ∥uk∥2R , (10)

where xˆrefk is the reference state derived from τˆ∗. The iLQR algorithm iteratively refines the control sequence via lo-

cal linearization of f(·) and quadratic approximation of the cost. Specifically, the first optimal command u∗t is applied to update the vehicle state, yielding the new ego pose Pt+1 for the subsequent simulation step.

T Warping Begin

T+1

T+2

New Pose

Ref. Pose

Ref. BEV Feature

New Pose

Ref. Pose

Ref. BEV Feature

Pose BEV Feature

|Warp Matrix| |
|---|---|
| | |

|Warp Matrix| |
|---|---|
| | |

Grid Sample

Grid Sample

New Pose

Warped BEV Feature

New Pose

Warped BEV Feature

Perception

Perception

Perception

Planning

Planning

Planning

Figure 3. Schematic of the BEV-Warp simulation environment. The closed-loop evaluation is driven by a recursive featurewarping mechanism. At each timestep, a transformation matrix Mt is derived from the relative pose deviation between the simulated agent Pt and the logged reference Ptref. This matrix is then applied to the reference BEV feature Btref via spatial warping, synthesizing a high-fidelity observation Bt for the subsequent planning cycle without expensive image-level rendering.

### 3.3 Joint Policy Optimization

To optimize the policy within the BEV-Warp environment, we propose a joint optimization framework that synergizes the diffusion generator Gθ with an RL-trained discriminator Dϕ through a cyclic structured pipeline. The global objective is to minimize the KL-divergence between the hybrid policy Πθ,ϕ and the ideal risk-neutral, high-efficiency distribution Π∗:

DKL Πθ,ϕ(τ | o) ∥ Π∗(τ | o) . (11)

min

θ,ϕ

As illustrated in Fig. 4, this objective is realized via a threestage iterative process: (i) Temporally Consistent Rollout, which collects stable closed-loop interaction data to ground exploratory behaviors (Sec. 3.3.1); (ii) Discriminator Optimization, where ϕ is optimized via trajectory-level RL to internalize sparse rewards and provide precise ranking feedback (Sec. 3.3.2) ; and (iii) Generator Optimization, which employs On-policy Generator Optimization to concentrate the density of Gθ onto high-reward manifolds (Sec. 3.3.3).

#### 3.3.1 Temporally Consistent Rollout

While the planning head generates multimodal trajectory hypotheses at each timestep, re-sampling and switching between diverse modes at high frequency often disrupts the continuity of the agent’s motion intentions. In the context of reinforcement learning, such frequent mode-switching decuples the correlation between a specific policy decision

[Figure 45]

IL Training Data Collected Rollout

[Figure 46]

[Figure 47]

EP=0

EP=0 TTC=Inf

IL

Trajectory Discriminator

[Figure 48]

Diffusion-based

TTC=Inf

[Figure 49]

- T0
- T1
- T2

- T0
- T1
- T2

Trajectory Generator

[Figure 50]

[Figure 51]

…

…

[Figure 52]

[Figure 53]

- EP=0.77 TTC=Inf

- EP=1.01 TTC=Inf

- EP=0.72 TTC=Inf

- EP=1.09 TTC=Inf

[Figure 54]

[Figure 55]

Discriminator

…

[Figure 56]

[Figure 57]

RL

Optimization

…

…

Rollout

[Figure 58]

Weight

Weight Update

Replication

[Figure 59]

[Figure 60]

Sensor Data

Cycle

[Figure 61]

[Figure 62]

[Figure 63]

Reward = 0.96 Reward = 1.00

IL

Diffusion-based

RL

| | | |
|---|---|---|
| | | |
| | | |

Trajectory Generator

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Generator Optimization

[Figure 68]

Navigation Expert Trajectory

Scene representation Navigation Reward Selected Trajectory

- Figure 4. The RAD-2 training pipeline. Our approach synergizes a Diffusion-based generator G and a Transformer-based discriminator D within a multi-stage optimization loop: (a) Pre-training Stage: G is initialized via imitation learning to capture multi-modal trajectory priors from expert demonstrations. (b) Closed-loop Rollout: The joint policy, integrating G for generation and D for selection, interacts with the high-throughput BEV-Warp environment to generate diverse rollout data. (c) Discriminator Optimization: D is optimized via Temporally Consistent Group Relative Policy Optimization, leveraging closed-loop feedback to enhance its scoring precision. (d) Generator Optimization: Through On-policy Generator Optimization, G is optimized via structured longitudinal optimizations derived from low-reward rollouts, effectively shifting its distribution toward safer and more efficient driving behaviors.

and its long-term driving outcome, leading to inefficient credit assignment and unstable policy improvement [56].

To mitigate these issues, we implement a trajectory reuse mechanism to maintain short-term behavioral consistency. Once an optimal trajectory τˆt∗ is selected at time t, it is converted into a structured control sequence u∗t:t+H following the iLQR optimization defined in Eq. (10). Instead of replanning at t + 1, the policy reuses the pre-computed controls over a fixed execution horizon Hreuse < H. Specifically, for each relative offset h ∈ {0,...,Hreuse − 1}, the corresponding control command u∗t+h is executed sequentially, ensuring that the vehicle state evolves consistently along the committed trajectory. While Hreuse is set as a fixed hyperparameter to ensure sampling efficiency, this mechanism natively supports asynchronous termination if safety constraints are violated, ensuring that the trajectory reuse does not compromise the agent’s reactive capabilities in dynamic environments. By stabilizing the exploratory direction, this mechanism ensures that the cumulative reward more accurately reflects the quality of the selected trajectory τˆt∗, thereby facilitating effective policy gradients.

#### 3.3.2 Discriminator Optimization via RL

To guide the discriminator in capturing long-horizon driving semantics, we establish a multi-objective reward function to evaluate closed-loop outcomes and optimize the policy via a Temporally Consistent Group Relative Policy Optimization (TC-GRPO) framework.

Reward Modeling. The reinforcement learning environ-

ment is built upon a collection of long-duration clips. Within each clip, the policy Πθ,ϕ interacts with the environment in a closed-loop manner to generate multiple rollouts. Each rollout represents an integrated decision-making chain, for which we define two complementary rewards to characterize safety and efficiency:

(i) Safety-Criticality Reward (rcoll): We quantify collision risks via counterfactual interpolation. Specifically, at each simulation step t, the Time-To-Collision Tt is defined as the earliest moment of intersection between the ego-vehicle’s projected occupancy and the environment’s ground-truth state:

Tt = inf{k ∈ [0,Tmax] | Bego(k;t) ∩ Venv(t + k) ̸= ∅},

(12) where Tmax is the safety detection threshold, Bego(k;t) denotes the ego-vehicle’s occupancy at look-ahead k (projected from step t), and Venv(t+k) represents the collective ground-truth occupancy of the environment at absolute time t+k. If no intersection is detected, Tt defaults to Tmax. The sequence-level reward is then formulated as the worst-case temporal margin:

(Tt − Tmax), (13)

rcoll = min

1≤t≤L

where L is the rollout horizon. This bottleneck-style formulation ensures that any momentary safety violation within the rollout dominates the sequence-level reward, thereby enforcing a strictly conservative driving policy.

(ii) Navigational Efficiency Reward (reff): To synchronize the traversal pacing with expert demonstrations, we

evaluate the sequence-level Ego Progress (EP), denoted as ρ, at the conclusion of each rollout. The reward reff is formulated as a stability window penalty to anchor the vehicle’s progress within a target efficiency interval [ρlow,ρhigh]:

reff = min(ρ − ρlow,0) + min(ρhigh − ρ,0) + 1, (14)

where ρlow = 1.05 and ρhigh = 1.10 define the normalized progress boundaries relative to the clipped expert trajectory. This formulation treats navigational efficiency as a bounded synchronization task, penalizing both sluggish under-performance and overly-aggressive deviations to enforce a human-like traversal pace.

Together, they balance safety and driving efficiency in closed-loop driving.

Optimization Objective. To address the credit assignment problem in continuous driving space, we propose Temporally Consistent Group Relative Policy Optimization (TCGRPO). This framework introduces a structured rollout and reward assignment mechanism specifically designed for the temporal dependencies of motion planning, where sampled trajectory hypotheses are persisted over a short horizon to ensure behavioral coherence. By ensuring that the sparse environment reward is directly attributed to the specific trajectory hypothesis sustained within each persistent interval, our approach prevents the reinforcement signal from being diluted by high-frequency switching between disparate motion intentions. This temporal alignment effectively denoises the policy gradient and stabilizes the iterative optimization of the generator-discriminator system.

For a rollout Oi with sequence-level reward ri, the standardized advantage Ai is computed relative to the group {Oi}Gi=1 generated from the same initial state:

ri − mean({r1,...,rG}) std({r1,...,rG})

. (15)

Ai =

We define Ki as the set of timesteps in rollout Oi where

a new trajectory τˆi,t∗ is sampled to commence a latchexecution interval. The clipped objective for these sparse decision points is formulated as:

= min ρi,tAi, clip(ρi,t,1 − ε,1 + ε)Ai , (16)

Li,t∈Ki

∗ i,t|oi,t)

where ρi,t = Dϕ(ˆτ

Dϕold(ˆτi,t∗ |oi,t) is the importance sampling ratio. This formulation ensures that the advantage signal reinforces the coherent trajectory hypothesis rather than independent action samples that lack temporal coherence.

To mitigate premature convergence and sigmoid saturation, we introduce an adaptive entropy regularization mechanism with temperature-based control. Concretely, let Hi,t denote the policy entropy at timestep t of the i-th rollout, and H¯ denote its batch-wise average. The adaptive regularization weight is computed based on the average entropy:

β = exp(λ) · [H¯<H¯target], (17)

where λ is a learnable temperature parameter and H¯target denotes the target entropy level. This formulation activates the entropy regularization exclusively when the average entropy falls below the target. The temperature parameter λ is adaptively tuned via gradient descent only under such deficit conditions.

The RL objective for the discriminator is formulated as:

|Ki|

G

1

Li,t + βHi,t . (18)

JRL(ϕ) = E

G i=1 |Ki|

t=1

i=1

By integrating adaptive entropy regularization with Li,t, the discriminator ensures stable exploration without compromising policy stability.

#### 3.3.3 On-policy Generator Optimization

The hybrid policy Πθ,ϕ is defined by the composition of generator Gθ and discriminator Dϕ. While Sec. 3.3.2 optimizes Dϕ for candidate selection, the mode coverage of Gθ constrains the explorable policy space. To expand this boundary, we optimize Gθ via On-policy Generator Optimization, which transforms closed-loop feedback into structured longitudinal optimization signal and repositions the generator’s probability mass toward regions with favorable long-term outcomes.

Reward-Guided Longitudinal Optimization. At each closed-loop time step t, we obtain a raw lookahead trajectory segment τtraw = {(xt+h,yt+h,vt+h,at+h)}Hh=0 over a horizon H. Based on the closed-loop feedback, we identify potential collision risks or insufficient progress and optimize the longitudinal component accordingly. This process adjusts the acceleration profile by applying a constant offset to the original values (with sign-aware handling) and re-integrating. This yields an optimized trajectory τtopt that preserves the spatial path of τtraw while optimizing its temporal progression to better align with the reward signals.

- (i) Safety-driven Deceleration: When the Time-to-

Collision Tt at the current time step falls below a threshold γsafe, we apply temporal compression by reducing the travel distance over the horizon H by a fixed ratio ρ ∈ (0,1).

- (ii) Efficiency-driven Acceleration: When the ego vehi-

cle lags behind the reference trajectory (indicating insufficient progress) and the instantaneous Time-to-Collision Tt exceeds a safety threshold (i.e., no collision risk), we apply temporal extension by increasing the travel distance over the horizon H by a fixed ratio ρ′ > 1.

By converting closed-loop reward signals into structured optimizations, this mechanism enables the generator to iteratively shift its output distribution toward safer and more efficient behaviors.

On-policy Distribution Shifting. The optimized trajectory segments τtopt are aggregated into an on-policy dataset Dopt = {τtopt}, which provides structured supervision for

fine-tuning the generator Gθ via a mean squared error loss over the prediction horizon H:

Lop(θ) = Eτopt∼Dopt

H

k=0

τ ˆt+k − τtopt+k 22 . (19)

This distribution shift is inherently gradual and stable, as the target trajectories in Dopt are constructed through on-policy interaction and dimension-specific optimization.

## 4 Experiment

### 4.1 Dataset Details

Generator Pretraining. For diffusion generator pretraining, we leverage approximately 50,000 hours of real-world driving data that provide ego-vehicle trajectories across diverse traffic scenarios. These trajectories are used to pretrain the motion generator, enabling the model to capture the multimodal distribution of human driving behaviors before downstream optimization.

Closed-loop Training and Evaluation in the BEV Warp Environment. We first collect 50k clips from real-world logs, each representing a continuous driving segment of 10–20 seconds. These clips cover diverse driving conditions and are categorized based on high-level driving objectives, including safety-oriented scenarios (e.g., interactions with elevated collision risk) and efficiency-oriented scenarios (e.g., driving under varying traffic conditions with suboptimal efficiency). The clips are evaluated in closed-loop simulation within the BEV Warp environment, where the pre-trained multimodal trajectory generator produces multiple candidate trajectories at every simulation step, from which one is randomly sampled to advance the ego vehicle. Based on the resulting outcomes, clips are filtered to retain safety-critical scenarios with elevated collision risk and efficiency-related scenarios with suboptimal driving performance. From these filtered scenarios, we curate two separate training sets corresponding to safety-oriented and efficiency-oriented objectives, each containing 10k clips, which are used for closed-loop RL training. For evaluation, we similarly construct two disjoint subsets of challenging clips corresponding to safety-oriented and efficiencyoriented scenarios, respectively, each containing 512 clips, for closed-loop assessment.

Closed-loop Training and Evaluation in the 3DGS Environment. In addition to the BEV Warp environment, we further train and evaluate the trajectory discriminator in the photorealistic 3D Gaussian Splatting (3DGS) simulation benchmark introduced in Senna-2 [45], which focuses on safety-oriented driving scenarios involving high-risk interactions. Among them, 1,044 clips are used for training and 256 clips are reserved for closed-loop evaluation.

Open-loop Evaluation Scenarios. We also adopt the open-loop evaluation dataset from Senna-2 [45]. The benchmark covers six representative driving scenarios: carfollowing start, car-following stop, lane changing, intersections, curves, and heavy braking. These scenarios span diverse driving conditions, including longitudinal control, lane-level interactions, and complex road geometries, enabling a comprehensive evaluation of planning quality across different driving contexts.

### 4.2 Experiment Setup

Baseline. We adopt the same perception backbone as RAD [7], which encodes multi-view images into BEV features and extracts static scene and dynamic agent tokens, providing inputs to the planning modules. The planning head follows the modeling formulation of ResAD [63], which is based on a standard diffusion framework for generative trajectory prediction. This planning head is used as the baseline trajectory generator in our experiments.

We compare our method with several representative planning approaches, including waypoint regression planners [3, 16, 60], diffusion-based generative planners [63], and selection-based planners [2]. All methods are implemented and trained on our dataset using the same perception backbone, differing only in the planning head used to model future trajectories.

Training Stages. RAD-2 follows a multi-stage training:

- (i) Pre-training: Following RAD [7], the model first per-

forms perception and planning pre-training. In the perception stage, the backbone encodes multi-view images into BEV features and extracts static scene and dynamic agent tokens, forming structured scene representations for downstream planning. Crucially, these BEV features exhibit strong spatial equivariance, ensuring that geometric transformations in the feature space correspond strictly to physical movements in the real world. As demonstrated in Fig. 5, applying a warp M to the reference feature Bref results in a predictable shift of the decoded perception outputs (e.g., 3D bounding boxes) that remains consistent with the simulated ego-motion while maintaining stability in the global coordinate system. In the planning stage, these representations are fed into a diffusion-based planning head trained on expert demonstrations to establish a baseline for humanaligned trajectory generation.

- (ii) Reinforcement Learning: After pre-training, the

model is further optimized in simulation by collecting rollouts using the current policy. The collected trajectories are stored in a shared buffer for coordinated training of both components: the discriminator is trained via reinforcement learning to assign higher scores to safe and efficient trajectories, while the generator is iteratively optimized using structured longitudinal signals derived from the collected rollouts. By leveraging this shared self-generated

[Figure 69]

[Figure 70]

[Figure 71]

Ref. BEV Feature Warped BEV Feature

[Figure 72]

[Figure 73]

Warp Matrix

|cos(θ)|−sin(θ)|𝑇𝑥|
|---|---|---|
|sin(θ)|cos(θ)|𝑇𝑦|
|0|0|1|

[Figure 74]

[Figure 75]

Perception

Perception

[Figure 76]

[Figure 77]

Same Region

in World Coordinates

- Figure 5. Spatial equivariance in BEV-Warp. We evaluate the reliability of warped features via a frozen pre-trained perception head. (Left) Reference feature Bref and its corresponding perception outputs. (Right) Under ego-displacement (e.g., θ = 1.08◦, Tx = 10.658m, Ty = 2.387m), the warped features yield perception results that maintain precise spatial alignment. This consistency across significant spatial transformations confirms that BEV-Warp preserves complex semantic geometry (e.g., dynamic agents and lane topologies) requisite for high-fidelity closed-loop simulation.

data, the system progressively enhances trajectory quality, safety, and closed-loop performance.

Reinforcement Learning Implementation Details. During the reinforcement learning stage, multiple driving rollouts are collected under the current policy for each clip. Trajectory reuse is employed to preserve short-term behavioral consistency (as described in Sec. 3.3.1), ensuring stable exploration during training. For each clip, we compute the mean and standard deviation of rollout rewards and discard the entire clip if the standard deviation falls below a predefined threshold. This filtering ensures that only clips with sufficiently diverse trajectory outcomes contribute to reinforcement learning. All valid rollouts are stored in a fixed-length replay buffer of 8, maintained in a first-in-firstout (FIFO) manner. Fig. 6 provides a comprehensive visualization of the data flow and the balanced data composition within the FIFO buffer.

During training, batches of rollouts are sampled from the buffer in groups of 4, and the standardized advantage is computed across each group.

The discriminator model is initialized by loading weights from the planning head for components that share the same

[Figure 78]

Rollout Collection

[Figure 79]

Driving

Close-loop

- • Trajectory reuse
- • Reward-based filtering
- • Scenario balancing

simulator

Policy

Safety Scenario Rollout Efficiency Scenario Rollout

A New batch rollout

[Figure 80]

New batch in Old batch out

Replay Buffer (FIFO)

New batch is added Buffer is fully refreshed

|Trigger|
|---|

|Trigger|
|---|

Hybrid Data

Trajectory

Trajectory

Joint optimization

Discriminator

Generator

Update policy Trigger new rollout collection

Figure 6. Replay buffer management and closed-loop optimization workflow. Driving rollouts are archived in a FIFO replay buffer that maintains a curated balance between safety-critical (yellow) and efficiency-oriented (green) scenarios. The system employs an asynchronous dual-trigger optimization strategy: (i) the trajectory discriminator is updated upon each batch ingestion to maintain sensitivity to recent rollout quality; (ii) the trajectory generator undergoes optimization only after a full buffer circulation (every 8 batches).

architecture, such as the static and dynamic encoders, as detailed in Sec. 3.1.2. Trajectory-leve policy gradients are applied to optimize the discriminator, with an entropy regularization term included to prevent the collapse of trajectory scores to extreme values.

The optimization alternates between the discriminator and the generator in a closed-loop fashion. Each time a new batch of rollouts is added, the discriminator is optimized using the current buffer contents. Once the buffer has been fully refreshed with 8 batches of new rollouts, a generator optimization is performed, resulting in an approximate 8:1 training frequency between the discriminator and generator. This ensures continuous co-adaptation of the discriminator and generator while maintaining diverse training data.

Metrics. We evaluate the closed-loop performance of our framework across two fundamental dimensions, safety and efficiency, using metrics derived directly from the simulated outcomes. To assess collision avoidance and risk management, we report the Collision Rate (CR) and the At-Fault Collision Rate (AF-CR), where the latter specifically accounts for incidents attributable to ego-vehicle decision errors. The robustness of safety margins is further quantified by Safety@1s and Safety@2s, representing the proportion of clips where the minimum Time-to-Collision (TTC) remains above 1 and 2 seconds, respectively. Concurrently, navigation effectiveness is evaluated through the

- Table 1. Closed-loop performance comparison in the BEV-Warp simulation environment. Our proposed approach achieves the lowest collision rates and highest navigation efficiency, demonstrating superior closed-loop interactive capabilities.

Method

Safety-oriented Scenario Efficiency-oriented Scenario

CR ↓ AF-CR ↓ Safety@1 ↑ Safety@2 ↑ EP-Mean ↑ EP@1.0 ↑ EP@0.9 ↑

TransFuser [3] 0.563 0.275 0.400 0.346 0.897 0.244 0.531 VAD [16] 0.594 0.299 0.371 0.312 0.904 0.252 0.623 GenAD [60] 0.592 0.305 0.363 0.309 0.930 0.467 0.736 ResAD [63] 0.533 0.264 0.418 0.281 0.970 0.516 0.894 RAD-2 0.234 0.092 0.730 0.596 0.988 0.736 0.984

- Table 2. Evaluation on photorealistic 3DGS benchmark. Our method outperforms IL and RL baselines in safety-critical scenarios. Notably, it achieves superior risk mitigation and safety margins, as reflected by the lower Collision Rate (CR) and higher Safety@1/2 scores.

Table 3. Open-loop benchmark of Senna-2. Our approach achieves state-of-the-art trajectory accuracy with the lowest ADE and FDE. The consistently minimal collision risk across diverse scenarios further validates the precision of our generative distribution in aligning with expert driving priors.

Method CR ↓ AF-CR ↓ Safety@1 ↑ Safety@2 ↑

TransFuser [3] 0.435 0.269 0.531 0.454 VAD [16] 0.502 0.280 0.458 0.362 VADv2 [2] 0.422 0.199 0.514 0.458 GenAD [60] 0.557 0.244 0.402 0.332 ResAD [63] 0.509 0.288 0.469 0.399 Senna [17] 0.310 0.111 0.638 0.539 Senna-2 [45] 0.269 0.077 0.667 0.565 RAD [7] 0.281 0.113 0.613 0.543 RAD-2 0.250 0.078 0.723 0.644

Ego Progress Mean (EP-Mean), which measures the average ratio of distance traveled relative to the reference route. We also include EP@1.0 and EP@0.9 as indicators of task completion reliability, denoting the percentage of scenarios where the vehicle successfully fulfills 100% and 90% of its assigned navigation goals, respectively.

For open-loop evaluation, we adopt the metrics from Senna-2 [45], including Final Displacement Error (FDE) and Average Displacement Error (ADE) for trajectory accuracy, along with Collision Rate (CR), Dynamic Collision Rate (DCR), and Static Collision Rate (SCR) for safety assessment of predicted trajectories against dynamic and static obstacles.

### 4.3 Experimental Results

We evaluate our method across both closed-loop and openloop benchmarks to assess safety, efficiency, and trajectory quality under diverse driving scenarios and simulation environments. Closed-loop evaluations focus on interactive driving behaviors, while open-loop tests examine the intrinsic accuracy of predicted trajectories. The following sections present detailed results in each setting.

Closed-loop performance in the BEV Warp environment. Closed-loop evaluation is critical for assessing the real driving behavior of planning systems. Tab. 1 reports

Method FDE (m) ↓ ADE (m) ↓ CR (%) ↓ DCR (%) ↓ SCR (%) ↓

TransFuser [3] 0.844 0.297 0.981 0.827 0.154 VAD [16] 0.722 0.262 0.621 0.554 0.067 GenAD [60] 0.806 0.290 0.520 0.491 0.030 ResAD [63] 0.634 0.234 0.378 0.367 0.011 Senna [17] 0.633 0.236 0.294 0.286 0.008 Senna-2 [45] 0.597 0.225 0.288 0.283 0.005 RAD-2 0.553 0.208 0.142 0.138 0.004

results in the BEV Warp simulation environment, evaluated on 512 held-out safety-oriented scenarios and 512 efficiency-oriented scenarios.

In safety-oriented scenarios, our method reduces the collision rate (CR) from 0.533 (ResAD) to 0.234 and the atfault collision rate (AF-CR) from 0.264 to 0.092, while increasing Safety@1/2 from 0.418/0.281 to 0.730/0.596. For efficiency-oriented scenarios, the Ego Progress Mean (EP-Mean) improves from 0.970 to 0.988, with route completion metrics EP@1.0/0.9 rising from 0.516/0.894 to 0.736/0.984. These results indicate that the proposed framework substantially enhances both safety and efficiency in closed-loop driving.

Closed-loop evaluation in the 3DGS environment. Tab. 2 reports results in the photorealistic 3D Gaussian Splatting (3DGS) benchmark, focusing on safety-oriented driving scenarios. Compared with recent systems including Senna, Senna-2, and RAD, our method achieves the lowest collision rate (0.250) and the highest Safety@1 and Safety@2 (0.723 and 0.644, respectively) among the evaluated methods. These results indicate that the trajectory scoring and reinforcement learning framework is also effective in photorealistic simulation environments.

Open-loop trajectory evaluation. Tab. 3 presents openloop comparisons on the Senna-2 evaluation benchmark, which includes six driving scenarios: car-following start, car-following stop, lane changing, intersections, curves, and

Table 4. Ablation study of the training pipeline. We evaluate the contribution of Imitation Learning (IL), On-policy Generator Optimization, and Discriminator Optimization. The results demonstrate that while IL provides essential priors, the synergistic optimization of the generator and discriminator is critical for balancing the safety-efficiency trade-off.

Gen. Pre-training Gen. Fine-tuning Disc. Training Safety-oriented Scenario Efficiency-oriented Scenario IL RL IL RL CR ↓ AF-CR ↓ Safety@1 ↑ Safety@2 ↑ EP-Mean ↑ EP@1.0 ↑ EP@0.9 ↑

ID

- 1 ✓ 0.533 0.264 0.418 0.281 0.970 0.516 0.894

- 2 ✓ ✓ 0.287 0.104 0.682 0.582 0.955 0.391 0.824

- 3 ✓ ✓ ✓ 0.403 0.197 0.555 0.434 0.973 0.527 0.936

- 4 ✓ ✓ 0.337 0.166 0.615 0.496 0.987 0.728 0.986

- 5 ✓ ✓ ✓ ✓ 0.234 0.092 0.730 0.596 0.988 0.736 0.984

Gen. = Generator, Disc. = Discriminator.

heavy braking. Our method substantially reduces collision rates, with the overall CR decreasing to 0.142%, compared to 0.288% from the previous best (Senna-2). Both dynamic and static collision components are also lower. FDE and ADE are also reduced to 0.553m and 0.208m, respectively, further supporting the improved trajectory quality in these open-loop scenarios.

### 4.4 Scaling Behavior Analysis

To study the scaling behavior of reinforcement learning performance with the cumulative number of environment timesteps, we compare three training strategies: discriminator-only optimization, sequential two-stage training, and synergistic joint optimization. As shown in Fig. 7, discriminator-only optimization exhibits limited scaling benefits, as the fixed generator constrains the quality and diversity of rollouts. The two-stage pipeline improves performance by first optimizing the generator and then training the discriminator, but suffers from suboptimal data utilization due to decoupled updates. In contrast, joint optimization achieves superior scaling efficiency, yielding a steeper performance curve and better final performance. Under this paradigm, the generator and discriminator are updated in a tightly coupled manner, enabling mutual improvement and progressively enhancing rollout quality. Moreover, both components are trained on shared rollouts, resulting in improved data efficiency and faster convergence.

### 4.5 Analysis of RL Training Design

Ablation on Training Pipeline. We ablate the training pipeline to examine the contributions of generator finetuning and discriminator training, as shown in Tab. 4. Incorporating On-policy Generator Optimization improves safety, reducing the collision rate from 0.533 to 0.287, but slightly decreases cruise efficiency. Introducing an IL finetuning stage during generator optimization restores cruise performance while maintaining these safety gains. Training the discriminator with RL further enhances both safety and efficiency, lowering the collision rate to 0.337 and in-

Two-stage

Joint Optimization

2.2

Disc. Optimization Only

| |
|---|

| |
|---|

2.0

Performance

1.8

1.6

Gen. Optimization Disc. Optimization

1.4

0 100000 200000 300000 400000 500000

Environment Timesteps

Figure 7. Scaling behavior of training paradigms. Performance (defined as 2 × Safety@1 + EP@1.0) is evaluated against the cumulative number of environment timesteps. The vertical dashed line marks the transition from on-policy generator optimization to discriminative optimization.

creasing EP-Mean to 0.987. By jointly applying RL training to the discriminator and fine-tuning the generator, the proportion of feasible trajectories is increased, allowing discriminator to fully exploit these trajectories and achieve the lowest collision rate (0.234) and highest Safety@1 (0.730) while maintaining high driving efficiency.

Analysis of RL Design Choices. We conduct ablations to analyze several design choices across the RL training pipeline, including data collection, sample filtering, model initialization, and optimization objectives.

Table 5. Sensitivity to execution horizon Hreuse. An execution horizon of 8 provides the optimal balance between stable credit assignment and reactive flexibility for stable RL optimization.

Hreuse CR ↓ Safety@1 ↑ EP@1.0 ↑

2 0.355 0.580 0.701 4 0.324 0.627 0.604 8 0.337 0.615 0.728 16 0.332 0.596 0.744

- Table 6. Effectiveness of reward-variance-based clip filtering. By discarding low-variance clips, the model focuses on scenarios with informative training signals, leading to significantly higher efficiency (EP@1.0) without compromising safety.

Clip Filtering CR ↓ Safety@1 ↑ EP@1.0 ↑

Disabled 0.340 0.617 0.662 Enabled 0.337 0.615 0.728

We evaluate the execution horizon Hreuse in Tab. 5. Lower values (e.g., 2) increase mode-switching frequency, which weakens the correlation between policy decisions and long-term outcomes, leading to unstable gradients. Conversely, a larger horizon (16) enhances temporal consistency but may limit reactivity. An execution horizon of 8 provides an optimal balance between credit assignment and reactive flexibility, achieving the best performance.

Next, we evaluate the impact of reward-based clip filtering for training data selection. As previously detailed, scenarios exhibiting low reward variance across rollouts are discarded, as they offer limited exploratory value. Tab. 6 demonstrates that this filtering mechanism enhances efficiency without compromising safety. Furthermore, as illustrated by the training dynamics in Fig. 8, the exclusion of non-informative clips fosters more stable convergence. This confirms that prioritizing scenarios with high-variance outcomes provides denser optimization signals.

0 100000 200000 300000 400000 500000

Environment Timesteps

1.4

1.5

1.6

1.7

1.8

1.9

2.0

Performance

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

w/o Clip Filtering

w/ Clip Filtering

- Figure 8. Training stability with and without clip filtering. Compared to the baseline (blue), clip filtering (orange) significantly stabilizes training dynamics and improves performance by prioritizing more informative training signals.

- Table 7. Ablation on discriminator model initialization strategies. Initializing the discriminator from the pre-trained planning head provides a robust structural prior, which is crucial for achieving high safety scores compared to random initialization.

Discriminator Initialization CR ↓ Safety@1 ↑ EP@1.0 ↑

Random 0.426 0.512 0.710 From Planning Head 0.337 0.615 0.728

We further investigate the impact of discriminator initial-

- Table 8. Effect of group size on TC-GRPO optimization. A group size of 4 achieves the best balance between reward normalization and safety performance.

Group Size CR ↓ Safety@1 ↑ EP@1.0 ↑

2 0.313 0.639 0.729 4 0.234 0.730 0.736 8 0.291 0.676 0.756

ization on optimization stability. As shown in Tab. 7, initializing the discriminator from the planning head significantly improves both safety and efficiency, whereas random initialization results in a drop in safety and a slight degradation in efficiency, indicating imbalanced optimization behavior. This performance gap stems from the structural prior provided by the pre-trained weights, enabling more reliable scene understanding and trajectory evaluation.

In addition, we evaluate the number of rollouts per group within the TC-GRPO in Tab. 8. Under our current setup, a group size of 4 yields the best performance, specifically achieving the lowest collision rate and highest Safety@1. While increasing the group size to 8 marginally improves efficiency (EP@1.0), it results in a performance drop across safety metrics. Consequently, we adopt a group size of 4 to ensure a robust balance between safety and efficiency.

Finally, we examine the role of the entropy term H in the RL objective. Including H prevents trajectory scores from collapsing to extreme values near 0 or 1, producing a more balanced distribution that facilitates stable RL optimization, as illustrated in Fig. 9. This improved stability translates into lower collision rates and higher Safety@1 and EP@1.0, as reported in Tab. 9.

- Table 9. Ablation on entropy regularization. The entropy term H in RL objective maintains a diverse scoring distribution and prevents policy collapse.

RL Objective CR ↓ Safety@1 ↑ EP@1.0 ↑

JRL (w/o H) 0.254 0.697 0.727 JRL (w/ H) 0.234 0.730 0.736

Ablation on Training Scenario Composition. We investigate the impact of training data distribution by comparing models trained on Mixed scenarios (integrating safety- and efficiency-oriented clips) against those trained on singleobjective subsets (Safety-only or Efficiency-only) and a Baseline configuration (Fig. 10). Empirical results show that the Mixed configuration achieves the most balanced performance, effectively navigating the trade-off between collision avoidance and navigation progress. In contrast, training exclusively on Efficiency-oriented data maximizes the imitation accuracy (EP@1.0) but suffers from substantially degraded safety robustness, indicating a failure to

0.7

Entropyoftrajectoryscore

w/o

| |
|---|

| |
|---|

w/

0.6

| |
|---|

| |
|---|

0.5

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

0.4

| |
|---|

| |
|---|

0.3

0.2

0 100000 200000 300000 400000 500000 600000

Environment Timesteps

- Figure 9. Entropy stability analysis. The inclusion of H (purple) prevents a rapid decline in trajectory score entropy, maintaining a more diverse distribution for stable optimization compared to the baseline without regularization (blue).

handle high-risk interactions. Conversely, Safety-oriented training prioritizes risk mitigation at the expense of efficiency. These findings suggest that single-objective training induces a significant performance bias, whereas incorporating a diverse scenario composition enables the model to learn a more robust driving policy that generalizes across competing objectives.

Mixed Efficiency-only Safety-only Baseline

0%

20%

40%

60%

80%

100%

Metric

| |
|---|

1-CR Safety@1

| |
|---|

EP@1.0

- Figure 10. Impact of scenario composition on closed-loop performance. Compared to single-objective training (Safety-only or Efficiency-only), the Mixed strategy achieves a superior trade-off across all metrics. Models trained on biased subsets exhibit significant performance drops in complementary tasks (e.g., Efficiencyonly lacks safety robustness), highlighting the necessity of diverse scenario curation for balanced driving policies.

Inference-time Scaling Analysis. We investigate the computational scalability of RAD-2 by varying the candidate trajectory count M at inference (Tab. 10). While the model is trained with M = 32, increasing M consistently scales the navigation efficiency (EP@1.0) from 0.667 to 0.814. Although safety metrics exhibit minor fluctuations due to the increased complexity of the expanded search space, the stable Collision Rate (CR) at higher M values demonstrates a robust inference-time scaling effect. This confirms that the discriminator can effectively leverage additional test-time

computation to identify higher-quality trajectories without further retraining.

Table 10. Inference-time scaling with candidate count M. Expanding the candidate pool during inference consistently improves planning performance by enabling a more comprehensive search of the multimodal action space.

M CR ↓ Safety@1 ↑ EP@1.0 ↑

8 0.275 0.693 0.667 16 0.266 0.689 0.711 32 0.234 0.730 0.736 64 0.252 0.699 0.816 128 0.234 0.719 0.814

### 4.6 Qualitative Comparisons.

We provide qualitative comparisons of closed-loop performance in Fig. 11 and Fig. 12. As shown in Fig. 11, RAD-2 exhibits superior safety in critical interactions by executing proactive deceleration to avoid collisions that the baseline fails to mitigate. Furthermore, Fig. 12 demonstrates that our model achieves higher driving efficiency in dynamic traffic through agile lane-changing maneuvers. These results validate the effectiveness of the proposed framework in balancing safety and progress within interactive environments.

## 5 Limitations and Future Work

Despite the substantial improvements in closed-loop stability and safety, several limitations of the proposed framework merit further discussion and provide avenues for future research.

Representation Specificity. The efficiency of our BEVWarp simulation environment is fundamentally rooted in the manipulation of BEV feature maps. While this design facilitates high-throughput policy iteration for systems that explicitly rely on BEV-centric perception, its applicability is constrained for architectures that utilize raw camera pixels or unified latent embeddings without explicit spatialequivariant grid structures. In such cases, the geometric warping mechanism would require a more generalized transformation module or a direct latent-space world model. Transition to Generative World Models. A promising extension of this work is the integration of our optimization pipeline with Generative World Models (WM). Although current WM offer superior flexibility and photorealism compared to feature-level warping, they often suffer from significant computational overhead and temporal drift during long-horizon generation, which limits their utility for large-scale RL training. Future efforts will focus on optimizing the inference efficiency and temporal consistency of latent-based world models. By porting our framework into a more flexible generative simulator, we aim to further

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Ego Vehicle

Ego Vehicle

Keyframe2Keyframe1StartFrame

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Ego Vehicle

Ego Vehicle

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Ego Vehicle

Ego Vehicle

- Figure 11. Qualitative comparison of closed-loop safety interaction. (1) Reference Input displays the front-view camera feed. (2) Baseline and (3) Ours each show the warped BEV input (left) and the corresponding perception and planning output (right). Starting from the same initial state at the Start Frame, the two models exhibit diverging behaviors as the interaction unfolds. By Keyframe 1, the Baseline fails to mitigate the risk and results in a collision, while our model performs proactive deceleration to maintain safety. At Keyframe 2, our vehicle resumes stable navigation after the threat clears, validating the robustness of our learned policy.

scale the diversity of training scenarios and bridge the remaining fidelity gap between simulation and the complex, open-world dynamics of real-world driving.

## 6 Conclusion

In this work, we presented RAD-2, a unified generator–discriminator framework for stable reinforcement learning in diffusion-based motion planning. We introduced Temporally Consistent Group Relative Policy Optimization to improve credit assignment through temporally coherent sampling, and On-policy Generator Optimization to progressively refine trajectory distributions using structured feedback. To support efficient large-scale training, we further introduce BEV-Warp, a feature-level simulation pipeline enabling scalable closed-loop learning. Extensive experiments demonstrate that RAD-2 consistently improves both safety and efficiency across diverse benchmarks, achieving substantial reductions in collision rates while maintaining reliable closed-loop navigation performance. Ablation studies further verify the effectiveness of

the proposed generator–discriminator formulation and temporally consistent optimization strategy.

## 7 Acknowledgement

We would like to acknowledge Hui Sun, Zhihao Guan, Songlin Yang, Qingjie Wang, Zhengqing Chen, Xiaoyang Guo, Xinbang Zhang and Nuoya Zhou for valuable discussions and assistance, Xinhui Bai, Wei Li and Pipi Ke for real-world closed-loop evaluation, and Zehua Li and Cheng Chi for data infra support.

## References

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3
- [2] Shaoyu Chen, Bo Jiang, Hao Gao, Bencheng Liao, Qing Xu, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang

[Figure 96]

[Figure 97]

[Figure 98]

EP=0

EP=0

Ego Vehicle

Ego Vehicle

Keyframe2Keyframe1StartFrame

[Figure 99]

[Figure 100]

[Figure 101]

- EP=0.75

- EP=1.09

- EP=0.77

- EP=1.01

Ego Vehicle

Ego Vehicle

[Figure 102]

[Figure 103]

[Figure 104]

Ego Vehicle

Ego Vehicle

###### Figure 12. Qualitative comparison of driving efficiency in dynamic traffic. (1) Reference Input displays the front-view camera feed.

(2) Baseline and (3) Ours each show the warped BEV input (left) and the perception and planning output with Efficiency Progress (EP) (right). Starting from the same initial state, both models encounter a vehicle merging from the right at Keyframe 1. Despite sufficient clearance in the adjacent lane, the Baseline exhibits overly conservative behavior, decelerating to wait behind the merging vehicle. In contrast, our model executes a proactive lane change to bypass the slow-moving agent. By Keyframe 2, our model achieves superior navigational progress (EP = 1.09) compared to the Baseline (EP = 1.01), validating its ability to make efficient tactical decisions in interactive environments.

Wang. Vadv2: End-to-end vectorized autonomous driving via probabilistic planning. arXiv preprint arXiv:2402.13243, 2024. 1, 2, 3, 8, 10

- [3] Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, and Andreas Geiger. Transfuser: Imitation with transformer-based sensor fusion for autonomous driving. IEEE transactions on pattern analysis and machine intelligence, 45(11):12878–12895, 2022. 1, 8, 10
- [4] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017. 2, 3
- [5] Lang Feng, Pengjie Gu, Bo An, and Gang Pan. Resisting stochastic risks in diffusion planners with the trajectory aggregation tree. arXiv preprint arXiv:2405.17879, 2024. 2
- [6] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Hongwei Xie, Bing Wang, Guang Chen, Dingkang Liang, and Xiang Bai. Minddrive: A vision-language-action model for autonomous driving via online reinforcement learning. arXiv preprint arXiv:2512.13636, 2025. 2

- [7] Hao Gao, Shaoyu Chen, Bo Jiang, Bencheng Liao, Yiang Shi, Xiaoyang Guo, Yuechuan Pu, Haoran Yin, Xiangyu Li, Xinbang Zhang, et al. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144, 2025. 2, 3, 8, 10
- [8] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3
- [9] Ke Guo, Haochen Liu, Xiaojun Wu, Jia Pan, and Chen Lv. ipad: Iterative proposal-centric end-to-end autonomous driving. arXiv preprint arXiv:2505.15111, 2025. 3
- [10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 4
- [11] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

- 4
- [12] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, et al. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17853–17862, 2023. 1
- [13] Zhiyu Huang, Jingda Wu, and Chen Lv. Efficient deep reinforcement learning with imitative expert priors for autonomous driving. IEEE Transactions on Neural Networks and Learning Systems, 34(10):7391–7403, 2022. 2, 3
- [14] Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. Advances in neural information processing systems, 28, 2015. 5
- [15] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024. 3
- [16] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. Vad: Vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023. 1, 8, 10
- [17] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024. 10
- [18] Bo Jiang, Shaoyu Chen, Qian Zhang, Wenyu Liu, and Xinggang Wang. Alphadrive: Unleashing the power of vlms in autonomous driving via reinforcement learning and reasoning. arXiv preprint arXiv:2503.07608, 2025. 3
- [19] Alex Kendall, Jeffrey Hawke, David Janz, Przemyslaw Mazur, Daniele Reda, John-Mark Allen, Vinh-Dieu Lam, Alex Bewley, and Amar Shah. Learning to drive in a day. In 2019 international conference on robotics and automation (ICRA), pages 8248–8254. IEEE, 2019. 1
- [20] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2, 3

- [21] Kyowoon Lee, Seongun Kim, and Jaesik Choi. Refining diffusion planner for reliable behavior synthesis by automatic detection of infeasible plans. Advances in Neural Information Processing Systems, 36:24223–24246, 2023. 2
- [22] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909, 2018. 3
- [23] Bohan Li, Zhuang Ma, Dalong Du, Baorui Peng, Zhujin Liang, Zhenqiang Liu, Chao Ma, Yueming Jin, Hao Zhao, Wenjun Zeng, et al. Omninwm: Omniscient driving navigation world models. arXiv preprint arXiv:2510.18313, 2025. 2
- [24] Kailin Li, Zhenxin Li, Shiyi Lan, Yuan Xie, Zhizhong Zhang, Jiayi Liu, Zuxuan Wu, Zhiding Yu, and Jose M Alvarez. Hydra-mdp++: Advancing end-to-end driving via expert-guided hydra-distillation. arXiv preprint arXiv:2503.12820, 2025. 1, 3

- [25] Qiyang Li, Zhiyuan Zhou, and Sergey Levine. Reinforcement learning with action chunking. arXiv preprint arXiv:2507.07969, 2025. 3
- [26] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025. 2
- [27] Weiwei Li and Emanuel Todorov. Iterative linear quadratic regulator design for nonlinear biological movement systems. In First International Conference on Informatics in Control, Automation and Robotics, pages 222–229. SciTePress, 2004. 5
- [28] Yingyan Li, Yuqi Wang, Yang Liu, Jiawei He, Lue Fan, and Zhaoxiang Zhang. End-to-end driving with online trajectory evaluation via bev world model. arXiv preprint arXiv:2504.01941, 2025. 4
- [29] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025. 2, 3
- [30] Zhenxin Li, Kailin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Yishen Ji, Zhiqi Li, Ziyue Zhu, Jan Kautz, Zuxuan Wu, et al. Hydra-mdp: End-to-end multimodal planning with multitarget hydra-distillation. arXiv preprint arXiv:2406.06978,

2024. 1, 2, 3

- [31] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2020–2036,

2024. 4

- [32] Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Jingde Chen, Nadine Chang, Maying Shen, Jingyu Song, Zuxuan Wu, Shiyi Lan, et al. Ztrs: Zero-imitation end-to-end autonomous driving with trajectory scoring. arXiv preprint arXiv:2510.24108, 2025. 1
- [33] Zhenxin Li, Wenhao Yao, Zi Wang, Xinglong Sun, Joshua Chen, Nadine Chang, Maying Shen, Zuxuan Wu, Shiyi Lan, and Jose M Alvarez. Generalized trajectory scoring for end-to-end multimodal planning. arXiv preprint arXiv:2506.06664, 2025. 3
- [34] Zhexi Lian, Haoran Wang, Xuerun Yan, Weimeng Lin, Xianhong Zhang, Yongyu Chen, and Jia Hu. Fine-tuning is not enough: A parallel framework for collaborative imitation and reinforcement learning in end-to-end autonomous driving. arXiv preprint arXiv:2603.13842, 2026. 3
- [35] Xiaodan Liang, Tairui Wang, Luona Yang, and Eric Xing. Cirl: Controllable imitative reinforcement learning for vision-based self-driving. In Proceedings of the European conference on computer vision (ECCV), pages 584–599,

2018. 2, 3

- [36] Bencheng Liao, Shaoyu Chen, Haoran Yin, Bo Jiang, Cheng Wang, Sixu Yan, Xinbang Zhang, Xiangyu Li, Ying Zhang, Qian Zhang, et al. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12037–12047, 2025. 1, 2, 3

- [37] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015. 2
- [38] Haochen Liu, Tianyu Li, Haohan Yang, Li Chen, Caojun Wang, Ke Guo, Haochen Tian, Hongchen Li, Hongyang Li, and Chen Lv. Reinforced refinement with self-aware expansion for end-to-end autonomous driving. arXiv preprint arXiv:2506.09800, 2025. 2, 3
- [39] Yiren Lu, Justin Fu, George Tucker, Xinlei Pan, Eli Bronstein, Rebecca Roelofs, Benjamin Sapp, Brandyn White, Aleksandra Faust, Shimon Whiteson, et al. Imitation is not enough: Robustifying imitation with reinforcement learning for challenging driving scenarios. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7553–7560. IEEE, 2023. 2, 3
- [40] Chaojun Ni, Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Wenkang Qin, Xinze Chen, Guanghong Jia, Guan Huang, and Wenjun Mei. Recondreamer-rl: Enhancing reinforcement learning via diffusion-based scene reconstruction. arXiv preprint arXiv:2508.08170, 2025. 2, 3
- [41] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In European conference on computer vision, pages 194–210. Springer, 2020. 4
- [42] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [43] Shuyao Shang, Yuntao Chen, Yuqi Wang, Yingyan Li, and Zhaoxiang Zhang. Drivedpo: Policy learning via safety dpo for end-to-end autonomous driving. arXiv preprint arXiv:2509.17940, 2025. 3
- [44] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. 3
- [45] Yuehao Song, Shaoyu Chen, Hao Gao, Yifan Zhu, Weixiang Yue, Jialv Zou, Bo Jiang, Zihao Lu, Yu Wang, Qian Zhang, et al. Senna-2: Aligning vlm and end-to-end driving policy for consistent decision making and planning. arXiv preprint arXiv:2603.11219, 2026. 8, 10
- [46] Wenchao Sun, Xuewu Lin, Keyu Chen, Zixiang Pei, Xiang Li, Yining Shi, and Sifa Zheng. Sparsedrivev2: Scoring is all you need for end-to-end autonomous driving. arXiv preprint arXiv:2603.29163, 2026. 3
- [47] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024. 1
- [48] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024. 4

- [49] Xinshuo Weng, Boris Ivanovic, Yan Wang, Yue Wang, and Marco Pavone. Para-drive: Parallelized architecture for realtime autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15449–15458, 2024. 1
- [50] Tianze Xia, Yongkang Li, Lijun Zhou, Jingfeng Yao, Kaixin Xiong, Haiyang Sun, Bing Wang, Kun Ma, Guang Chen, Hangjun Ye, et al. Drivelaw: Unifying planning and video generation in a latent driving world. arXiv preprint arXiv:2512.23421, 2025. 2
- [51] Tianyi Yan, Tao Tang, Xingtai Gui, Yongkang Li, Jiasen Zhesng, Weiyao Huang, Lingdong Kong, Wencheng Han, Xia Zhou, Xueyang Zhang, et al. Ad-r1: Closed-loop reinforcement learning for end-to-end autonomous driving with impartial world models. arXiv preprint arXiv:2511.20325, 2025.
- [52] Pengxuan Yang, Ben Lu, Zhongpu Xia, Chao Han, Yinfeng Gao, Teng Zhang, Kun Zhan, XianPeng Lang, Yupeng Zheng, and Qichao Zhang. Worldrft: Latent world model planning with reinforcement fine-tuning for autonomous driving. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 11649–11657, 2026.
- [53] Pengxuan Yang, Yupeng Zheng, Deheng Qian, Zebin Xing, Qichao Zhang, Linbo Wang, Yichen Zhang, Shaoyu Guo, Zhongpu Xia, Qiang Chen, et al. Dreamerad: Efficient reinforcement learning via latent world model for autonomous driving. arXiv preprint arXiv:2603.24587, 2026. 2
- [54] Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M Alvarez, and Zuxuan Wu. Drivesuprim: Towards precise trajectory selection for end-to-end planning. arXiv preprint arXiv:2506.06659, 2025. 3
- [55] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025. 3
- [56] Haichao Zhang, Wei Xu, and Haonan Yu. Generative planning for temporally coordinated exploration in reinforcement learning. arXiv preprint arXiv:2201.09765, 2022. 6
- [57] Guosheng Zhao, Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Xueyang Zhang, Yida Wang, Guan Huang, Xinze Chen, Boyuan Wang, Youyi Zhang, et al. Drivedreamer4d: World models are effective data machines for 4d driving scene representation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12015–12026, 2025. 4
- [58] Guosheng Zhao, Xiaofeng Wang, Chaojun Ni, Zheng Zhu, Wenkang Qin, Guan Huang, and Xingang Wang. Recondreamer++: Harmonizing generative and reconstructive models for driving scene representation. arXiv preprint arXiv:2503.18438, 2025. 4
- [59] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. 3
- [60] Wenzhao Zheng, Ruiqi Song, Xianda Guo, Chenming Zhang, and Long Chen. Genad: Generative end-to-end autonomous driving. In European Conference on Computer Vision, pages 87–104. Springer, 2024. 8, 10

- [61] Yinan Zheng, Ruiming Liang, Kexin Zheng, Jinliang Zheng, Liyuan Mao, Jianxiong Li, Weihao Gu, Rui Ai, Shengbo Eben Li, Xianyuan Zhan, et al. Diffusion-based planning for autonomous driving with flexible guidance. arXiv preprint arXiv:2501.15564, 2025. 1
- [62] Yinan Zheng, Tianyi Tan, Bin Huang, Enguang Liu, Ruiming Liang, Jianlin Zhang, Jianwei Cui, Guang Chen, Kun Ma, Hangjun Ye, et al. Unleashing the potential of diffusion models for end-to-end autonomous driving. arXiv preprint arXiv:2602.22801, 2026. 2
- [63] Zhiyu Zheng, Shaoyu Chen, Haoran Yin, Xinbang Zhang, Jialv Zou, Xinggang Wang, Qian Zhang, and Lefei Zhang. Resad: Normalized residual trajectory modeling for end-toend autonomous driving. arXiv preprint arXiv:2510.08562,

2025. 1, 3, 8, 10

- [64] Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, and Yiyi Liao. Hugsim: A real-time, photo-realistic and closed-loop simulator for autonomous driving. arXiv preprint arXiv:2412.01718, 2024. 2
- [65] Ming Zhou, Jun Luo, Julian Villella, Yaodong Yang, David Rusu, Jiayu Miao, Weinan Zhang, Montgomery Alban, Iman Fadakar, Zheng Chen, et al. Smarts: Scalable multi-agent reinforcement learning training school for autonomous driving. arXiv preprint arXiv:2010.09776, 2020. 3
- [66] Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A visionlanguage-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757, 2025. 2, 3
- [67] Jialv Zou, Shaoyu Chen, Bencheng Liao, Zhiyu Zheng, Yuehao Song, Lefei Zhang, Qian Zhang, Wenyu Liu, and Xinggang Wang. Diffusiondrivev2: Reinforcement learningconstrained truncated diffusion modeling in end-to-end autonomous driving. arXiv preprint arXiv:2512.07745, 2025. 1, 3

