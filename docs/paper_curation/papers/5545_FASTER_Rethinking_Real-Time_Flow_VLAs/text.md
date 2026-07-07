# arXiv:2603.19199v3[cs.RO]16May2026

## FASTER: Rethinking Real-Time Flow VLAs

##### Yuxiang Lu1,2 Zhe Liu1,∗ Xianzhe Fan1 Zhenya Yang1 Jinghua Hou1 Junyi Li1 Kaixin Ding1 Hengshuang Zhao1,† 1The University of Hong Kong 2ACE Robotics ∗Project Leader †Corresponding Author https://innovator-zero.github.io/FASTER

Dynamic Task Slow Reaction

Action Sampling w/ Constant Schedule

[Figure 1]

[Figure 2]

|VLM prefill|step1|step2|step3|step4|step5|step6|step7|step8|step9|step10|
|---|---|---|---|---|---|---|---|---|---|---|

22.4ms 76.7ms

10x Faster Sampling

[Figure 3]

[Figure 4]

22.4ms 7.8ms

Fast Action Sampling for ImmediaTE Reaction

|VLM prefill|step1|
|---|---|

Fast Reaction

Figure 1: We propose FASTER to alleviate the reaction latency bottleneck in action chunking flow policies. By compressing the sampling iterations of the immediate reaction into a single step, FASTER (bottom) achieves a 10× acceleration compared to original π0.5 and X-VLA (top). This enables real-time responsiveness in highly dynamic real-world tasks such as playing table tennis. Moreover, FASTER is a plug-and-play solution for flow-based VLAs that integrates seamlessly into the standard fine-tuning pipeline and requires no architectural modifications.

#### Abstract

Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (e.g., in π0.5 and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks substantially improved real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.

#### 1 Introduction

The paradigm of robot learning is undergoing a profound transformation with the advent of VisionLanguage-Action (VLA) models [114, 77, 64, 76]. By formulating continuous motor control as a

Preprint.

generative sequence modeling problem, recent approaches leveraging diffusion models [30, 83] and flow matching [51] for action chunking have achieved unprecedented capabilities in dexterous robotic manipulation tasks [34, 2, 119, 101]. As research focus shifts from simulation to real-world physical deployment, real-time capability has become increasingly paramount.

Existing real-time execution methods primarily address the “stop-and-wait” issue in standard synchronous inference [16] for action chunking policies [81]. By introducing an asynchronous pipeline, the robot can initiate the next inference request before the current action chunk is exhausted, thereby eliminating inter-chunk pauses and enhancing motion continuity [4]. While state-of-the-art advances in asynchronous inference strategy [5, 93, 88, 58, 118] further reinforce trajectory smoothness, these methods largely overlook another essential dimension of real-time embodied intelligence: reaction. Beyond smooth execution, a practical VLA system must promptly and precisely respond to dynamically changing physical environments. Delayed reactions to unexpected perturbations create a perilous “blind spot” in closed-loop control, limiting the effectiveness of generalist policies in open-world scenarios.

Our in-depth analysis of the inference pipeline in Section 2 reveals that reaction time is not a trivial constant determined by inference latency. Instead, it should be modeled as a random variable following a uniform distribution, due to the stochastic timing of external events relative to robot controller. We further illustrate that existing asynchronous methods are inherently limited, and a collaborative enhancement in both perception-execution latency and the frequency of inferenceexecution cycle is entailed to acquire truly responsive behavior.

We then revisit a common design in flow-based VLAs: a constant timestep schedule across the action chunk, which allocates an equal number of sampling steps to every action. Under this scheme, the full multi-step denoising process must be completed before any action can be dispatched, severely inflating the reaction delay. Considering the intrinsic causal structure of physical interaction, nearterm actions are more tightly coupled with current observations and typically lie in a significantly narrower solution space. Our pilot study in Section 3.2 supports this intuition. We clearly observe that early actions follow straighter interpolation paths and attain precise estimation of clean actions within only a few sampling steps, whereas a constant schedule over-samples these dimensions. This naturally raises a key question: since earlier actions are easier to predict than later ones, can flow-based VLAs generate these latency-critical actions with fewer sampling steps for immediate reaction?

To address these challenges, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER), a simple yet effective method applicable to flow-based VLAs [3, 119] without architectural modifications or additional training cost. As shown in Figure 1, FASTER aims to accelerate the sampling process of leading actions, as quantified by the newly introduced Time to First Action (TTFA) metric for reactivity. Concretely, a Horizon-Aware Schedule (HAS) is incorporated to decouple the local denoising timestep for each frame within the chunk. HAS adaptively allocates more aggressive sampling steps to near-term actions while maintaining a slower schedule for long-horizon ones. Consequently, the model can output the immediate action as fast as one-step sampling, while largely preserving long-term trajectory accuracy.

Beyond algorithmic acceleration, FASTER also catalyzes a paradigm shift from conventional asynchronous pipeline to a streaming client-server interaction, wherein early actions can be dispatched to the robot controller instantly upon completion. While the robot executes these initial movements, the VLA model continues refining subsequent actions in parallel and progressively replenishes the client’s action buffer. Real-world evaluations on two GPU platforms (i.e., RTX 4060 and RTX 4090) demonstrate that FASTER substantially reduces inference latency, as reflected by lower TTFA, while simultaneously boosting inference-execution cycle frequency through the synergization of streaming output and early-stopping strategies. Real-robot experiments further confirm the superior reaction capability of FASTER, even when deployed on resource-constrained GPUs, offering a general and promising path toward genuinely real-time VLAs.

Our contributions are summarized as: (1) We present a systematic analysis of reaction attributes in action chunking VLA policies, revealing the inherent limitations of existing methods for real-time responsiveness. (2) We propose the FASTER framework, capitalizing on a Horizon-Aware Schedule that prioritizes immediate actions during flow matching sampling, effectively compressing TTFA to one-step sampling without sacrificing prediction quality. (3) We design a streaming client-server interface with early stopping, jointly trimming the delay and accelerating the closed loop of inference-

Sync Inference

Infer Interval

Infer Latency

Policy Server

Infer

Infer

time

- (a)
- (b)

Robot Client

Action Chunk Pause Action Chunk Pause

Action Chunk

Execution Duration

Async Inference

Infer Interval Infer Latency

Policy Server

Infer

Infer

Infer

time

Robot Client

Action Chunk Action Chunk

Action Chunk

Action Chunk

Execution Duration

C-S Communication

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Best Reaction Case

| | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Worst Reaction Case

delayed executed discarded

- Figure 2: Temporal pipelines of (a) synchronous and (b) asynchronous inference in a robotic system composed of an action chunking policy server and a robot client. As indicated by the best and worst cases, reaction time depends on both inference latency and the interval between consecutive inference-execution cycles. We also illustrate the decomposition of two adjacent action chunks to

clarify the discretized inference delay d and the execution horizon s (bounded by smin and H − d) in the asynchronous client.

execution. (4) Extensive experiments on real robots demonstrate significant improvements in reaction capability and promising performance in dexterous action generation for manipulation tasks.

#### 2 Analysis on Action Chunking Policy Inference

Action chunking is a standard method in VLA policies [16, 117, 3]. Given a policy π, the model processes an observation ot at real-world time t to predict a sequence of future actions At = [at,at+1,...,at+H−1], where H denotes the prediction horizon specified by the policy. In practice, instead of executing the entire action chunk, it is common to execute only s actions, then trigger a new inference and discard the remaining actions. s is referred to as the execution horizon [4].

Deploying a VLA policy on a physical robot typically utilizes a client-server architecture, consisting of a policy server for model inference and a robot client for motor control. After initialization, the server remains active to process incoming requests from the client and returns predictions with a certain latency, giving rise to two interaction paradigms: synchronous and asynchronous inference.

We first define several time quantities to assist analysis of the pipeline:

- • Control period ∆tctrl := 1/f. In robotic systems, the controller operates at a specific frequency f (e.g., 30Hz), corresponding to a fixed period between consecutive operations, such as executing action or triggering inference.
- • Inference latency ∆tinfer. This is defined as the time interval between transmitting an observation and receiving the predicted actions on the client side. It encompasses model inference, network communication, pre- and post-processing, memory I/O, and other system overheads. For analytical convenience, we model the total latency as a constant. Following prior work [4], we also define the discretized inference delay as d := ⌊∆tinfer/∆tctrl⌋.
- • Execution duration ∆texec := s · ∆tctrl. This denotes the time required for the robot client to execute s actions.

Synchronous Inference. The system operates synchronously by default, as shown in Figure 2(a). After completing execution of the preceding chunk at time t, the client sends the observation ot to the server to request a new inference. After the inference latency, the server returns the predicted chunk.

Table 1: Reaction characteristics of synchronous and asynchronous inference. U(a,b) denotes a uniform distribution with lower and upper bounds a and b, so the expectation of ∆treact equals the midpoint (a + b)/2.

Mode Infer Interval ∆treact ∼ Dreact E[∆treact] smin Sync ∆tinfer + ∆texec U(∆tinfer, 2 ∗ ∆tinfer + ∆texec) 1.5 ∗ ∆tinfer + 0.5 ∗ ∆texec –

Async ∆texec U(∆tinfer, ∆tinfer + ∆texec) ∆tinfer + 0.5 ∗ ∆texec ⌈∆tinfer/∆tctrl⌉

During this period, the robot controller pauses and resumes only when the new actions arrive at t+∆tinfer. To achieve uninterrupted execution, the condition ∆tinfer < ∆tctrl (i.e., d = 0) should hold, meaning the next chunk is available within a single control step. In practice, however, this requirement is hardly satisfied, resulting in non-smooth trajectories and degraded task performance [4, 88].

Asynchronous Inference. A natural strategy to tackle inter-chunk pauses is asynchronous inference [81]. The core idea is to initiate inference of the next chunk before the current chunk is fully executed, as depicted in Figure 2(b). Specifically, once inference is triggered at time t, the robot continues executing the remaining actions in the ongoing chunk. By time t + ∆tinfer, when the final action is completed, the newly predicted chunk is expected to be available, thereby enabling seamless execution without halt.

However, asynchronous execution incurs the problem of perception-execution gap [49, 103]. The observation is captured at time t, but when the new chunk becomes available, the environment and the robot state may have changed due to actions executed during the interval (t,t + ∆tinfer). A naive strategy of discarding the first d delayed actions in the new chunk and switching to the remaining ones [d,d + s) can lead to unstable and discontinuous motion, and this issue becomes increasingly severe

- as the delay d grows [4, 88]. Recent approaches mitigate inter-chunk discontinuity by incorporating the overlapping actions ([s,d + s) in previous chunk) as part of the model input [4, 5, 93, 58, 9]. This paper follows RTC [4, 5], where the overlapping actions are treated as prefix conditions during action generation, guiding the new actions to transition smoothly.

Smoothness vs. Reaction. Existing real-time VLAs primarily focus on improving inter-chunk smoothness. Nevertheless, they often overlook or misunderstand another fundamental aspect of real-time performance: reaction. In this work, we revisit the notion of reaction in action chunking policy inference and provide a systematic analysis.

Reaction time ∆treact is defined as the interval between the occurrence of a sudden event and the response produced by the robot. We summarize the reaction characteristics of synchronous and asynchronous clients in Table 1. A key insight is that reaction time is susceptible to the dual influence of inference latency and frequency. Given that policy inference is performed periodically, with consecutive inference triggers at time t and t′ in Figure 2. When a new event occurs, the system can only respond after the next inference cycle is completed. Therefore, the lower bound of ∆treact is ∆tinfer, corresponding to the case where the event happens just before an inference starts. In the worst case, the event occurs immediately after inference begins; the reaction will then only be reflected at t′ + ∆tinfer, shaping an upper bound equal to ∆tinfer plus the inference interval.

As events occur stochastically in the physical world, the reaction time can be modeled as following a uniform distribution, denoted by Dreact. An important finding from the expectation E[∆treact] in

- Table 1 is that the gain by upgrading from synchronous to asynchronous inference is potentially limited, as it reduces the expected reaction time by only 0.5 ∗ ∆tinfer.

Reducing the execution horizon s is an intuitive idea to increase the inference frequency. In the asynchronous setting, s has a minimum value smin := ⌈∆tinfer/∆tctrl⌉ to guarantee that the inference interval exceeds the latency (i.e., ∆texec ≥ ∆tinfer). Under this configuration, inference is triggered every smin control steps, achieving optimal reaction performance [103].

Time to First Action. As observed in Table 1, reaction time heavily depends on inference latency. More importantly, if actions are not generated simultaneously, responsiveness is determined solely by how quickly the system can produce the first action. Since the robot does not need the entire action chunk to begin moving, later actions, while essential for task accuracy, do not directly affect immediate responsiveness. Therefore, we introduce Time to First Action (TTFA) as a more precise metric for measuring reactivity, analogous to Time to First Token (TTFT) in large language models [32, 120]. TTFA explicitly captures the earliest moment at which the robot can initiate movement, making it the true bottleneck of reaction speed. This paper presents a novel asynchronous pipeline that

0.07

0.07

| |Pick Beverage<br><br>Fold Towel| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| |[Figure 5]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |[Figure 6]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

[Figure 7]

[Figure 8]

- 0 10 20 30 40 50 action index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

- 0 10 20 30 40 50 action index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

0.12

0.06

0.06

0.10

SAstraightness()

0.05

0.05

←samplingstep

←samplingstep

0.08

0.04

0.04

0.06

0.03

0.03

0.04

0.02

0.02

0.02

0.01

0.01

0.00

0.00

0.00

0 10 20 30 40 50

action index

(a)

(b)

(c)

- Figure 3: Visualizations of (a) straightness S(A) of the denoising path during sampling of the action

chunk, and (b,c) differences between the intermediate clean action estimates A˜ τt→0 at each sampling timestep τ and the final output A0t, using the tasks “Pick Beverage” and “Fold Towel”, respectively.

jointly minimizes TTFA and increases inference frequency, leading to substantially improved reaction capability in action chunking policies.

#### 3 Methodology

##### 3.1 Preliminaries

We adopt the widely used flow-based VLA structure [3, 35, 2, 119]. The model consists of a VLM backbone and an action expert (AE), learning a velocity field that transports a noise sample to the target action chunk using conditional flow matching [51, 52]. Training follows the optimal transport formulation [90, 56], which assumes a linear interpolation path between Gaussian noise ϵ ∼ N(0,I)

and the ground-truth actions Aˆ t: Aτt = τϵ + (1 − τ)Aˆ t, where τ ∈ (0,1) is the continuous timestep of the flow. The objective is to regress the velocity field along this path with network vθ:

L(θ) = Eτ∼U(0,1) vθ(ot,Aτt ,τ) − (ϵ − Aˆ t)

2

. (1)

During inference, actions are generated by initializing from Gaussian noise A1t ∼ N(0,I) at τ = 1, and progressively integrating the learned velocity field toward τ = 0 using an ODE solver such as the Euler method:

Aτt+∆τ = Aτt + vθ(ot,Aτt ,τ)∆τ, (2) where ∆τ = −1/N is related to the number of sampling steps N, a typical value of 10 in practice.

##### 3.2 Pilot Study on Action Chunk Sampling

Existing flow-based VLAs treat the entire action chunk as an indivisible unit and apply a constant timestep schedule across all action indexes. As a result, every action within the chunk undergoes the same number of denoising steps during inference. The immediate next action At, which is urgently required for execution, is therefore forced to share the same schedule as the most distant future action At+H−1. Consequently, the entire multi-step denoising procedure has to be completed before any individual action can be issued, which constitutes a dominant bottleneck of the overall latency [108].

Nevertheless, action chunks exhibit an inherent temporal structure. Given current observations and proprioceptive states, early-stage actions are subject to stronger causal constraints and thus lie in a substantially narrower search space compared to future actions. Intuitively, this makes shortterm predictions easier and more certain. Furthermore, when asynchronous methods incorporate action prefixes as input [5, 58, 88], these extra priors provide additional conditioning that constrains subsequent predictions. This further reduces the uncertainty of the immediate actions and lowers the complexity of generation.

We validate this hypothesis through a quantitative analysis of the sampling dynamics in flow-based VLAs. Specifically, we adopt the straightness metric [56], which is defined for any continuously

differentiable process Z = {Zτ} evolving from Z0 to Z1 as S(Z) = 0 1 E ∥(Z1 − Z0) − Z˙τ∥2 dτ, where Z˙τ = ddτ Zτ denotes the instantaneous velocity at time τ. In our context, the VLA denoising

prediction horizon

action index

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | |
|---|---|---|---|---|---|---|---|

noise

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

samplingsteps

samplingstep

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

clean

| | | | | | | | |
|---|---|---|---|---|---|---|---|

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

dispatched

hit time

output

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

(a) Constant Schedule (b) Horizon-Aware Schedule

- Figure 4: Illustration of (a) constant timestep schedule used in conventional flow sampling and (b) Horizon-Aware Schedule (HAS) used in FASTER that allocates adaptive hit times across the action chunk and accelerates the sampling of early actions, enabling streaming output. process is discretized and the straightness can be formulated as:

S(A) =

1

Et (A1t − A0t) − vθ(ot,Aτt ,τ) 2 · (−∆τ), (3)

τ=0

where A0t represents the final actions obtained via Equation (2). A value of S(A) = 0 indicates a perfectly straight path. Smaller S(A) corresponds to paths closer to linear interpolation, which in turn can be accurately integrated with fewer steps [56]. We also investigate the estimated clean actions

A˜ τt→0 at each denoising step {1,1 + ∆τ,...,−∆τ}, obtained via the following extrapolation: A˜ τt→0 = Aτt − vθ(ot,Aτt ,τ)τ. (4)

We measure their deviation from the final output A0t using ℓ2 norm ∥A˜ τt→0−A0t∥2. During sampling, this deviation is expected to decrease and reaches zero at the final step. A smaller deviation suggests that the model provides a more accurate estimate of the results at current timestep.

We conduct a pilot study by fine-tuning a pretrained π0.5 model on our real-world robotic tasks. As visualized in Figure 3, we can find that both the straightness metric and the estimate deviation exhibit non-uniformity across the temporal dimension (action index) of the action chunk. In particular, early actions (approximately the first 1–10 frames) demonstrate lower straightness values and smaller

variations in A˜ τt→0 throughout the sampling iterations. This empirical observation provides strong evidence supporting our hypothesis.

##### 3.3 FASTER

Motivated by the insight that near-term actions within a chunk are easier to generate under flow matching, we propose to prioritize the sampling of these latency-critical actions with a Horizon-Aware Schedule (HAS).

Horizon-Aware Schedule. Unlike conventional flow-based VLAs, which employ a constant time schedule across the entire chunk (Figure 4(a)), we design a horizon-aware time allocation mechanism that accelerates the denoising of near-term actions during inference, while allowing later-horizon actions to follow a comparatively slower schedule, mimicking the straightness plot in Figure 3(a).

Inspired by Diffusion Forcing [10], the flow matching timestep is made index-dependent and represented as a vector τ = {τi}, where i ∈ [0,H − 1] denotes the action index. As illustrated in Figure 4(b), we use ρj ∈ (0,1) to represent the global sampling progress at j-th step, which still iterates from 1 to 0 following the standard flow matching procedure. Each action reaches completion

- at a distinct hit time ui determined by its index: ui = (1 − (i/(H − 1))α) · u0 i ∈ [1,H − 1], (5)

where the predefined u0 specifies the global timestep at which the first action is finalized. The hyperparameter α ∈ (0,1] controls how the hit times vary across the action index. When α = 1, the

hit times decrease uniformly from u0 to 0. When α < 1, early actions reach their hit times rapidly, while future actions have hit times closer to 0. This design adaptively allocates more denoising steps

- Table 2: Comparison of reaction capability on RTX 4090 and RTX 4060 GPUs. “Speedup” denotes the relative gain of our method over the asynchronous baseline.

RTX 4090 RTX 4060

Model Method

TTFA ↓ smin ↓ E[∆treact] ↓ TTFA ↓ smin ↓ E[∆treact] ↓

Sync 80.0±1.6ms 3 170.0ms 303.3±0.8ms 10 621.6ms Async 80.0±1.6ms 3 130.0ms 303.3±0.8ms 10 470.0ms

π0.5

FASTER 62.1±3.1ms 3 112.1ms 238.6±1.9ms 8 371.9ms

- Speedup 1.29× – 1.16× 1.27× 1.25× 1.26×

X-VLA

Sync 113.7±0.8ms 4 237.2ms 399.5±8.5ms 12 799.2ms Async 113.7±0.8ms 4 180.4ms 399.5±8.5ms 12 599.5ms

FASTER 44.8±0.3ms 2 78.1ms 129.2±2.4ms 6 229.2ms

- Speedup 2.54× 2× 2.31× 3.09× 2× 2.62×

to future actions due to compounding uncertainty, preserving generation quality and minimizing deviation from the original schedule inherited from pretraining.

Given the global timestep ρj, the local time schedule for each action index is formulated as:

τij = max 0,(ρj − ui)/(1 − ui) . (6)

Under this schedule, once ρj reaches u0, the first action is fully denoised and can be dispatched immediately, while the remaining actions continue refining. We set u0 = (N −1)/N, thus guaranteeing that the first action is ready with a single sampling step. With ρj further decreasing, subsequent actions are progressively completed when ρj = ui.

Fine-tuning with Mixed Schedule. Directly fine-tuning a pretrained VLA using HAS may introduce two challenges. First, existing pretrained models are optimized under a constant timestep schedule, thus naively switching to an index-dependent schedule can enlarge the fine-tuning gap, in addition to the distribution shift. Second, when randomly selecting ρ ∼ U(0,1), there exists a high probability that the corresponding local timestep for near-term actions becomes zero. This could collapse the learning of these actions since the inputs are constantly ground-truth and potentially deteriorating rollout performance.

To address these issues, we introduce a mixed scheduling strategy to augment fine-tuning. Concretely, given a mixing probability p, each action sample in a training batch utilizes HAS with probability p, and retains the original constant schedule with probability 1 − p. The mixed schedule forces the model to learn the flow matching velocity field under both timestep parameterizations, thereby improving robustness to the schedule variation across the action horizon. It is worth noting that the proposed scheduling methodology can be readily incorporated into the standard fine-tuning pipeline of flow-based VLAs, without any architecture modifications or additional training cost.

Synergization with Action Conditioning. HAS naturally synergizes with the action conditioning technique proposed in Training-time RTC [5], which conditions both training and sampling on the action prefixes by treating them as fully denoised. Under our adaptive schedule, early actions frequently receive timesteps close to zero, while future actions are assigned progressively larger timesteps. This pattern is inherently consistent with the principle of action conditioning. Moreover, it encourages the model to learn a more structured mapping between timestep values and the degree of noise interpolation relative to the target actions, thereby strengthening its understanding of temporally conditioned generation. Action conditioning is integrated into HAS with an offset on the index, and we provide a detailed description in Appendix D.1.

Streaming Client-Server Interface. With FASTER, actions are generated progressively during the denoising iterations. To exploit this property in real-world robotic systems, we implement a streaming client-server interface. On the server side, newly finalized actions from the policy are dispatched immediately, while the model proceeds to generate the remaining actions concurrently. On the client side, the controller continuously listens for incoming packets and appends received actions to the robot’s action buffer for execution, without waiting for the entire chunk to complete. As long as the action acquisition rate exceeds the robot control frequency—which is achievable even on consumer-level GPUs—the robot can operate without interruption (shown in Appendix D.2).

Enhancing Reaction Capability. We analyze how FASTER improves reaction time by tightening both its lower and upper bounds. For conventional flow-based VLAs with N sampling steps, TTFA is approximately ∆tVLM + N∆tAE, where ∆tVLM and ∆tAE denote the inference time of the VLM and

- Table 3: Comparison of reaction speed from a probabilistic perspective on RTX 4090 and RTX 4060 GPUs. “vs. Sync” denotes the probability that a given method has a shorter reaction time than Sync, and “vs. Async” is defined analogously.

Model Method

RTX 4090 RTX 4060 vs. Sync vs. Async vs. Sync vs. Async

π0.5

Async 0.72 - 0.74 -

- FASTER 0.81 0.66 0.88 0.77

X-VLA

Async 0.73 - 0.75 -

- FASTER 1.00 1.00 1.00 1.00

AE, respectively. Equipped with FASTER, TTFA is shortened to ∆tVLM + ∆tAE, as the first action requires only a single AE sampling step.

Responsiveness can be further improved by increasing the inference frequency, i.e., reducing the execution horizon s, whose minimum value smin must satisfy ∆texec ≥ ∆tinfer. When a high frequency is desired, a relatively small s is selected (e.g. 4 out of H = 50), implying that only a small portion of the chunk is useful while the other actions are discarded. Our method enables an early-stopping strategy: once all actions within the execution horizon are finalized, the remaining sampling steps can be skipped. This avoids completing all AE iterations and effectively reduces the overall latency. Consequently, we can apply a smaller feasible smin compared with conventional asynchronous inference, tightening the upper bound of ∆treact without sacrificing execution smoothness.

- 4 Experiments

We focus primarily on real-world experiments, simulation benchmarks and ablation studies are provided in Appendix F.3 and F.4 due to space limitations.

##### 4.1 Experimental Analysis on Reaction Speed

Experimental Setup. To investigate improvements in reaction capability, we first compare the inference latency—measured by TTFA—and the expected reaction time of our method against the synchronous and asynchronous baselines. Since prior real-time VLA approaches [4, 5, 88] follow the same naive asynchronous paradigm in these aspects, we do not report them separately. Experiments are conducted on two hardware platforms: a high-performance RTX 4090 GPU, and a consumer-grade RTX 4060 GPU. We utilize two representative flow-based VLAs, π0.5 [35] and X-VLA [119], and adhere to their default configurations. The robot control frequency is set to f = 30Hz, corresponding to a control period ∆tctrl = 33.3ms. To reflect the optimal achievable reaction speed, we set the execution horizon to smin, thereby maximizing the inference frequency.

Results. As detailed in Table 2, our method manifests substantial acceleration in reaction performance across all scenarios. As the design of X-VLA incurs a higher computational cost in its action expert, FASTER delivers particularly significant gains, achieving a 3× boost in TTFA on RTX 4060. Notably, the early-stopping strategy in our approach effectively decreases the feasible inference interval by a smaller smin, which minimizes the inference-execution cycle and contributes to additional enhancements in responsiveness.

As highlighted in Section 2, reaction time should be treated as a random variable, since event occurrences are inherently stochastic. Accordingly, Table 3 presents a probabilistic analysis that more faithfully reflects real-world conditions, measuring the probability that one method attains a faster reaction time than another. FASTER surpasses both baselines by a clear margin, with a larger advantage in the more resource-constrained scenario. Notably, on X-VLA, our method is deterministically superior: its upper bound of reaction time is lower than the baselines’ lower bound, establishing a strict performance dominance.

##### 4.2 Real-world Experiments

Experimental Setup. To examine the real-robot reaction capability in a highly dynamic environment, we design a task that demands both rapid response and accurate motion execution. Specifically, we train VLA to play table tennis using a racket mounted on a 6-DoF Piper robot arm in the AgileX

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Sync Training-time RTC FASTER

Naive Async

SyncTraining-timeRTCFASTER

1.0

0.80

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

0.8

NaiveAsync

0.53

0.6

Score

0.4

0.20

0.2

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

0.00

0.0

RTX 4090

1.0

0.8

0.6

Score

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

0.47

0.4

0.30

0.20

0.2

0.00

0.0

RTX 4060

- Figure 5: Comparison of real-world reaction speed on the table tennis task. Left: Visualization of rollouts on RTX 4090, the third column corresponds to the contact moment, and the interval between each image in a row is 166.7ms (5 frames). Right: Quantitative completion scores on two GPUs.

Pick Beverage Fold Towel

0.75

0.80

0.85

0.90

0.95

1.00

Score

0.879

0.957 0.950 0.957

0.788

0.825

0.888

0.963

Pick Beverage Fold Towel

0

5

10

15

20

25

Duration(s)

13.0 12.5 11.9 12.0

24.7 24.0

20.7 20.5

Sync Naive Async Training-time RTC FASTER

- Figure 6: Comparison of real-world performance and completion duration on two additional tasks.

Cobot Magic platform. We collect approximately 14 minutes of demonstration data via human teleoperation to fine-tune the π0.5 models. In addition, we include two tasks that place less emphasis on real-time reaction: (1) “Pick Beverage”: picking up a beverage and placing it in a basket, focusing on object and localization generalization; (2) “Fold Towel”: folding a towel twice with dual arms, representing deformable object manipulation. Details are provided in Appendix E.1.

Results. We compare synchronous inference (“Sync”), naive asynchronous inference (“Naive Async”), and the state-of-the-art Training-time RTC [5] with our FASTER. Representative rollouts are visualized on the left of Figure 5, with quantitative results presented in the right panels. We observe that Sync fails to respond to incoming balls in our trials due to its drastically slow reaction speed. Naive Async and Training-time RTC share the same temporal pipeline and thus similar reaction capability. However, Training-time RTC improves the smoothness of racket swing through its action conditioning mechanism, leading to moderately higher scores.

Our method exhibits noticeably faster reaction than all baselines, which can be intuitively observed from the racket angle at the moment of ball contact. To successfully return the ball, the robot should begin adjusting its arm posture in advance to reach an appropriate hitting position with sufficient swing velocity. If the reaction is delayed, there is insufficient time left to move, leading to suboptimal contact angles, as seen in the baselines. In contrast, faster reactions allow the robot to initiate motion earlier, providing adequate time to rotate the racket and build up swing speed, resulting in a more powerful and controlled hit. Therefore, the racket angle serves as a clear and visually interpretable indicator of reaction speed. Overall, FASTER achieves the best performance across both hardware settings. The advantage is particularly pronounced on RTX 4060, where inference runs at merely 3Hz, highlighting the compounded benefits of reduced inference latency and increased frequency.

We report the average completion score and duration for the additional tasks in Figure 6. The results show that asynchronous methods outperform Sync by a clear gap, and FASTER achieves better or comparable scores across both tasks. This highlights that task performance is not determined solely by action accuracy, but also by the real-time interaction with the physical world. Though our method may slightly compromise action prediction due to accelerated sampling, it strikes a more effective balance between responsiveness and accuracy.

Consistent with previous findings [5, 88], the synchronous method exhibits the longest task duration due to frequent inter-chunk pauses. In contrast, Training-time RTC and FASTER effectively reduce completion time, yielding greater efficiency for downstream applications.

#### 5 Conclusion

In this paper, we revisit reaction capability in action chunking VLA policies and identify the constant timestep schedule in flow-based VLAs as a key bottleneck of real-time responsiveness. We propose FASTER, which leverages a Horizon-Aware Schedule to adaptively accelerate action sampling. It enables single-step generation of the immediate action, without compromising overall trajectory quality. Capitalizing on a streaming client-server interface with early stopping, FASTER jointly reduces TTFA and speeds up closed-loop control. Real-robot experiments confirm that FASTER offers a robust, general, and plug-and-play path toward real-time embodied intelligence, particularly on edge devices.

#### References

- [1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [2] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T N1: an open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.
- [3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai,

Lachy Groom, Karol Hausman, Brian Ichter, et al. π0: A vision-language-action flow model for general robot control. In RSS, 2025.

- [4] Kevin Black, Manuel Y Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. In NeurIPS, 2025.
- [5] Kevin Black, Allen Z Ren, Michael Equi, and Sergey Levine. Training-time action conditioning for efficient real-time chunking. arXiv preprint arXiv:2512.05964, 2025.
- [6] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Xindong He, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. In IROS, 2025.
- [7] Paweł Budzianowski, Wesley Maa, Matthew Freed, Jingxiang Mo, Winston Hsiao, Aaron Xie, Tomasz Młoduchowski, Viraj Tipnis, and Benjamin Bolte. Edgevla: Efficient vision-language-action models. arXiv preprint arXiv:2507.14049, 2025.
- [8] Remi Cadene, Simon Alibert, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, Mustafa Shukor, Jess Moss, Alexander Soare, Dana Aubakirova, Quentin Lhoest, Quentin Gallouédec, and Thomas Wolf. Lerobot: An opensource library for end-to-end robot learning. In ICLR, 2026.
- [9] Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, Fei Ma, Kun Ma, et al. Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution. arXiv preprint arXiv:2602.12684, 2026.
- [10] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In NeurIPS, 2024.
- [11] Haojun Chen, Minghao Liu, Chengdong Ma, Xiaojian Ma, Zailin Ma, Huimin Wu, Yuanpei Chen, Yifan Zhong, Mingzhi Wang, Qing Li, and Yaodong Yang. Falcon: Fast visuomotor policies via partial denoising. In ICML, 2025.
- [12] Jiahong Chen, Jing Wang, Long Chen, Chuwei Cai, and Jinghui Lu. Nanovla: Routing decoupled visionlanguage understanding for nano-sized generalist robotic policies. arXiv preprint arXiv:2510.25122, 2025.
- [13] Yang Chen, Xiaoguang Ma, and Bin Zhao. Mean-flow based one-step vision-language-action. arXiv preprint arXiv:2603.01469, 2026.

- [14] Yuxuan Chen and Xiao Li. Rlrc: Reinforcement learning-based recovery for compressed vision-languageaction models. arXiv preprint arXiv:2506.17639, 2025.
- [15] Zhuoqun Chen, Xiu Yuan, Tongzhou Mu, and Hao Su. Responsive noise-relaying diffusion policy: Responsive and efficient visuomotor control. TMLR, 2025.
- [16] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion Policy: Visuomotor policy learning via action diffusion. In RSS, 2023.
- [17] Yuntao Dai, Hang Gu, Teng Wang, Qianyu Cheng, Yifei Zheng, Zhiyong Qiu, Lei Gong, Wenqi Lou, and Xuehai Zhou. Actionflow: A pipelined action acceleration for vision language models on edge. arXiv preprint arXiv:2512.20276, 2025.
- [18] Haoran Ding, Noémie Jaquier, Jan Peters, and Leonel Rozo. Fast and robust visuomotor riemannian flow matching policy. IEEE Transactions on robotics, 2025.
- [19] Zhenchen Dong, Jinna Fu, Jiaming Wu, Shengyuan Yu, Fulin Chen, and Yide Liu. Hybridflow: A two-step generative policy for robotic manipulation. arXiv preprint arXiv:2602.13718, 2026.
- [20] Yufei Duan, Hang Yin, and Danica Kragic. Real-time iteration scheme for diffusion policy. In IROS, pages 11758–11764, 2025.
- [21] Xianzhe Fan, Shengliang Deng, Xiaoyang Wu, Yuxiang Lu, Zhuoling Li, Mi Yan, Yujia Zhang, Zhizheng Zhang, He Wang, and Hengshuang Zhao. Any3d-vla: Enhancing vla robustness via diverse point clouds. arXiv preprint arXiv:2602.00807, 2026.
- [22] Hengyu Fang, Yijiang Liu, Yuan Du, Li Du, and Huanrui Yang. Sqap-vla: A synergistic quantizationaware pruning framework for high-performance vision-language-action models. arXiv preprint arXiv:2509.09090, 2025.
- [23] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In ICLR, 2025.
- [24] Zipeng Fu, Tony Z. Zhao, and Chelsea Finn. Mobile ALOHA: Learning bimanual mobile manipulation using low-cost whole-body teleoperation. In CoRL, 2024.
- [25] Juntao Gao, Feiyang Ye, Jing Zhang, and Wenjing Qian. Compressor-vla: Instruction-guided visual token compression for efficient robotic manipulation. arXiv preprint arXiv:2511.18950, 2025.
- [26] Yuxuan Gao, Yedong Shen, Shiqi Zhang, Wenhao Yu, Yifan Duan, Jiajia Wu, Jiajun Deng, Yanyong Zhang, et al. Drift-based policy optimization: Native one-step policy learning for online robot control. arXiv preprint arXiv:2604.03540, 2026.
- [27] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In NeurIPS, 2025.
- [28] Weifan Guan, Qinghao Hu, Aosheng Li, and Jian Cheng. Efficient vision-language-action models for embodied manipulation: A systematic survey. arXiv preprint arXiv:2510.17111, 2025.
- [29] Xiaoshuai Hao, Lei Zhou, Zhijian Huang, Zhiwen Hou, Yingbo Tang, Lingfeng Zhang, Guang Li, Zheng Lu, Shuhuai Ren, Xianhui Meng, et al. Mimo-embodied: X-embodied foundation model technical report. arXiv preprint arXiv:2511.16518, 2025.
- [30] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.
- [31] Sigmund H Høeg, Yilun Du, and Olav Egeland. Streaming diffusion policy: Fast policy synthesis with variable noise diffusion models. arXiv preprint arXiv:2406.04806, 2024.
- [32] Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, et al. Deepspeedfastgen: High-throughput text generation for llms via mii and deepspeed-inference. arXiv preprint arXiv:2401.08671, 2024.
- [33] Yuting Huang, Leilei Ding, Zhipeng Tang, Zenghuan Zhu, Jiajun Deng, Xinrui Lin, Shuo Liu, Haojie Ren, Jianmin Ji, and Yanyong Zhang. Environment-aware adaptive pruning with interleaved inference orchestration for vision-language-action models. arXiv preprint arXiv:2602.00780, 2026.
- [34] Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace

Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. π0∗.6: a vla that learns from experience. arXiv preprint arXiv:2511.14759, 2025.

- [35] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan

Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

- [36] Jason Jabbour, Dong-Ki Kim, Max Smith, Jay Patrikar, Radhika Ghosal, Youhui Wang, Ali Agha, Vijay Janapa Reddi, and Shayegan Omidshafiei. Don’t run with scissors: Pruning breaks vla models but they can be recovered. arXiv preprint arXiv:2510.08464, 2025.
- [37] Boseong Jeon, Yunho Choi, and Taehan Kim. Shallow-π: Knowledge distillation for flow-based vlas. arXiv preprint arXiv:2601.20262, 2026.
- [38] Bofang Jia, Pengxiang Ding, Can Cui, Mingyang Sun, Pengfang Qian, Siteng Huang, Zhaoxin Fan, and Donglin Wang. Score and distribution matching policy: Advanced accelerated visuomotor policies via matched distillation. arXiv preprint arXiv:2412.09265, 2024.
- [39] Jindou Jia, Gen Li, Xiangyu Chen, Tuo An, Yuxuan Hu, Jingliang Li, Xinying Guo, and Jianfei Yang. Action-to-action flow matching. arXiv preprint arXiv:2602.07322, 2026.
- [40] Titong Jiang, Xuefeng Jiang, Yuan Ma, Xin Wen, Bailin Li, Kun Zhan, Peng Jia, Yahui Liu, Sheng Sun, and Xianpeng Lang. The better you learn, the smarter you prune: Towards efficient vision-language-action models via differentiable token pruning. arXiv preprint arXiv:2509.12594, 2025.
- [41] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In RSS, 2024.
- [42] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In RSS, 2025.
- [43] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-languageaction model. In CoRL, 2024.
- [44] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [45] Shaolong Li, Lichao Sun, and Yongchao Chen. One-step flow policy: Self-distillation for fast visuomotor policies. arXiv preprint arXiv:2603.12480, 2026.
- [46] Wei Li, Renshan Zhang, Rui Shao, Zhijian Fang, Kaiwen Zhou, Zhuotao Tian, and Liqiang Nie. Semanticvla: Semantic-aligned sparsification and enhancement for efficient robotic manipulation. arXiv preprint arXiv:2511.10518, 2025.
- [47] Ye Li, Yuan Meng, Zewen Sun, Kangye Ji, Chen Tang, Jiajun Fan, Xinzhu Ma, Shutao Xia, Zhi Wang, and Wenwu Zhu. Sp-vla: A joint model scheduling and token pruning approach for vla model acceleration. arXiv preprint arXiv:2506.12723, 2025.
- [48] Zhuoling Li, Xiaoyang Wu, Zhenhua Xu, and Hengshuang Zhao. Train once, deploy anywhere: Realize data-efficient dynamic object manipulation. arXiv preprint arXiv:2508.14042, 2025.
- [49] Aileen Liao, Dong-Ki Kim, Max Olan Smith, Ali-akbar Agha-mohammadi, and Shayegan Omidshafiei. Delay-aware diffusion policy: Bridging the observation-execution gap in dynamic tasks. arXiv preprint arXiv:2512.07697, 2025.
- [50] Tao Lin, Yilei Zhong, Yuxin Du, Jingjing Zhang, Jiting Liu, Yinxinyu Chen, Encheng Gu, Ziyan Liu, Hongyi Cai, Yanwen Zou, et al. Evo-1: Lightweight vision-language-action model with preserved semantic alignment. arXiv preprint arXiv:2511.04555, 2025.
- [51] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023.
- [52] Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024.
- [53] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. In NeurIPS, pages 44776–44791, 2023.

- [54] Jiaming Liu, Mengzhen Liu, Zhenyu Wang, Pengju An, Xiaoqi Li, Kaichen Zhou, Senqiao Yang, Renrui Zhang, Yandong Guo, and Shanghang Zhang. Robomamba: Efficient vision-language-action model for robotic reasoning and manipulation. In NeurIPS, pages 40085–40110, 2024.
- [55] Songming Liu, Bangguo Li, Kai Ma, Lingxuan Wu, Hengkai Tan, Xiao Ouyang, Hang Su, and Jun Zhu. RDT2: Exploring the scaling limit of umi data towards zero-shot cross-embodiment generalization. arXiv preprint arXiv:2602.03310, 2026.
- [56] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.
- [57] Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Max Du, and Chelsea Finn. Bidirectional decoding: Improving action chunking via guided test-time sampling. In ICLR, 2025.
- [58] Yufeng Liu, Hang Yu, Juntu Zhao, Bocheng Li, Di Zhang, Mingzhu Li, Wenxuan Wu, Yingdong Hu, Junyuan Xie, Junliang Guo, et al. Learning native continuation for action chunking flow policies. arXiv preprint arXiv:2602.12978, 2026.
- [59] Zhe Liu, Runhui Huang, Rui Yang, Siming Yan, Zining Wang, Lu Hou, Di Lin, Xiang Bai, and Hengshuang Zhao. Drivepi: Spatial-aware 4d mllm for unified autonomous driving understanding, perception, prediction and planning. In CVPR, 2026.
- [60] Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, and Bo Zhao. Vla-pruner: Temporal-aware dual-level visual token pruning for efficient vision-language-action inference. arXiv preprint arXiv:2511.16449, 2025.
- [61] Guanxing Lu, Zifeng Gao, Tianxing Chen, Wenxun Dai, Ziwei Wang, Wenbo Ding, and Yansong Tang. Manicm: Real-time 3d diffusion policy via consistency model for robotic manipulation. arXiv preprint arXiv:2406.01586, 2024.
- [62] Wuyang Luan, Junhui Li, Weiguang Zhao, Wenjian Zhang, Tieru Wu, and Rui Ma. Snapflow: One-step action generation for flow-matching vlas via progressive self-distillation. arXiv preprint arXiv:2604.05656, 2026.
- [63] Xiaoyu Ma, Zhengqing Yuan, Zheyuan Zhang, Kaiwen Shi, Lichao Sun, and Yanfang Ye. Blurr: A boosted low-resource inference for vision-language-action models. arXiv preprint arXiv:2512.11769, 2025.
- [64] Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-languageaction models for embodied ai. arXiv preprint arXiv:2405.14093, 2024.
- [65] Yunchao Ma, Yizhuang Zhou, Yunhuan Yang, Tiancai Wang, and Haoqiang Fan. Running vlas at real-time speed. arXiv preprint arXiv:2510.26742, 2025.
- [66] Michael Matthews, Michael Beukman, Chris Lu, and Jakob Nicolaus Foerster. Kinetix: Investigating the training of general agents through open-ended physics-based control tasks. In ICLR, 2025.
- [67] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. RAL, 2022.
- [68] Chaojun Ni, Cheng Chen, Xiaofeng Wang, Zheng Zhu, Wenzhao Zheng, Boyuan Wang, Tianrun Chen, Guosheng Zhao, Haoyun Li, Zhehao Dong, et al. Swiftvla: Unlocking spatiotemporal dynamics for lightweight vla models at minimal overhead. arXiv preprint arXiv:2512.00903, 2025.
- [69] Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. In ICRA, 2024.
- [70] Seongmin Park, Hyungmin Kim, Wonseok Jeon, Juyoung Yang, Byeongwook Jeon, Yoonseon Oh, and Jungwook Choi. Quantization-aware imitation-learning for resource-efficient robotic control. arXiv preprint arXiv:2412.01034, 2024.
- [71] Seongmin Park, Hyungmin Kim, Sangwoo Kim, Wonseok Jeon, Juyoung Yang, Byeongwook Jeon, Yoonseon Oh, and Jungwook Choi. Saliency-aware quantized imitation learning for efficient robotic control. In ICCV, pages 13140–13150, 2025.
- [72] Xiaohuan Pei, Yuxing Chen, Siyu Xu, Yunke Wang, Yuheng Shi, and Chang Xu. Action-aware dynamic pruning for efficient vision-language-action manipulation. In ICLR, 2026.

- [73] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.
- [74] Moritz Reuss, Hongyi Zhou, Marcel Rühle, Ömer Erdinç Ya˘gmurlu, Fabian Otto, and Rudolf Lioutikov. Flower: Democratizing generalist robot policies with efficient vision-language-action flow policies. In CoRL, 2025.
- [75] Tommoro Robotics, Jesoon Kang, Taegeon Park, Jisu An, Soo Min Kimm, Jaejoon Kim, Jinu Pahk, Byungju Kim, Junseok Lee, Namheon Baek, et al. Habilis-beta: A fast-motion and long-lasting on-device vision-language-action model. arXiv preprint arXiv:2602.18813, 2026.
- [76] Ranjan Sapkota, Yang Cao, Konstantinos I Roumeliotis, and Manoj Karkee. Vision-language-action (vla) models: Concepts, progress, applications and challenges. arXiv preprint arXiv:2505.04769, 2025.
- [77] Rui Shao, Wei Li, Lingsen Zhang, Renshan Zhang, Zhiyang Liu, Ran Chen, and Liqiang Nie. Large vlm-based vision-language-action models for robotic manipulation: A survey. arXiv preprint arXiv:2508.13073, 2025.
- [78] Juyi Sheng, Ziyi Wang, Peiming Li, and Mengyuan Liu. Mp1: Meanflow tames policy learning in 1-step for robotic manipulation. In AAAI, 2026.
- [79] Modi Shi, Li Chen, Jin Chen, Yuxiang Lu, Chiming Liu, Guanghui Ren, Ping Luo, Di Huang, Maoqing Yao, and Hongyang Li. Is diversity all you need for scalable robotic manipulation? arXiv preprint arXiv:2507.06219, 2025.
- [80] Yiran Shi, Dongqi Guo, Tianchen Zhao, Feng Gao, Liangzhi Shi, Chao Yu, ZhiJian Mo, Qihua Xiao, XiaoShuai Peng, Qingmin Liao, et al. Streamingvla: Streaming vision-language-action model with action flow matching and adaptive early observation. arXiv preprint arXiv:2603.28565, 2026.
- [81] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al. Smolvla: A vision-languageaction model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.
- [82] Andreas Sochopoulos, Nikolay Malkin, Nikolaos Tsagkas, Joao Moura, Michael Gienger, and Sethu Vijayakumar. Fast flow-based visuomotor policies via conditional optimal transport couplings. In CoRL, 2025.
- [83] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2021.
- [84] Wenxuan Song, Jiayi Chen, Pengxiang Ding, Yuxin Huang, Han Zhao, Donglin Wang, and Haoang Li. Ceed-vla: Consistency vision-language-action model with early-exit decoding. arXiv preprint arXiv:2506.13725, 2025.
- [85] Wenxuan Song, Jiayi Chen, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, and Haoang Li. Accelerating vision-language-action model integrated with action chunking via parallel decoding. In IROS, 2025.
- [86] Mingzhen Sun, Weining Wang, Gen Li, Jiawei Liu, Jiahui Sun, Wanquan Feng, Shanshan Lao, SiYu Zhou, Qian He, and Jing Liu. Ar-diffusion: Asynchronous video generation with auto-regressive diffusion. In CVPR, pages 7364–7373, 2025.
- [87] Amir Taherin, Juyi Lin, Arash Akbari, Arman Akbari, Pu Zhao, Weiwei Chen, David Kaeli, and Yanzhi Wang. Cross-platform scaling of vision-language-action models from edge to cloud gpus. arXiv preprint arXiv:2509.11480, 2025.
- [88] Jiaming Tang, Yufei Sun, Yilong Zhao, Shang Yang, Yujun Lin, Zhuoyang Zhang, James Hou, Yao Lu, Zhijian Liu, and Song Han. Vlash: Real-time vlas via future-state-aware asynchronous inference. arXiv preprint arXiv:2512.01031, 2025.
- [89] Gemini Robotics Team, Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, et al. Gemini Robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. arXiv preprint arXiv:2510.03342, 2025.
- [90] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. TMLR, 2024.

- [91] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. BridgeData v2: A dataset for robot learning at scale. In CoRL, 2023.
- [92] Hanzhen Wang, Jiaming Xu, Jiayi Pan, Yongkang Zhou, and Guohao Dai. Specprune-vla: Accelerating vision-language-action models via action-aware self-speculative pruning. arXiv preprint arXiv:2509.05614, 2025.
- [93] Haoxuan Wang, Gengyu Zhang, Yan Yan, Yuzhang Shang, Ramana Rao Kompella, and Gaowen Liu. Real-time robot execution with masked action chunking. In ICLR, 2026.
- [94] Hongyu Wang, Chuyan Xiong, Ruiping Wang, and Xilin Chen. Bitvla: 1-bit vision-language-action models for robotics manipulation. arXiv preprint arXiv:2506.07530, 2025.
- [95] Songsheng Wang, Rucheng Yu, Zhihang Yuan, Chao Yu, Feng Gao, Yu Wang, and Derek F Wong. Spec-vla: speculative decoding for vision-language-action models with relaxed acceptance. In EMNLP, pages 26916–26928, 2025.
- [96] Yihao Wang, Pengxiang Ding, Lingxiao Li, Can Cui, Zirui Ge, Xinyang Tong, Wenxuan Song, Han Zhao, Wei Zhao, Pengxu Hou, et al. Vla-adapter: An effective paradigm for tiny-scale vision-language-action model. In AAAI, 2025.
- [97] Zhendong Wang, Max Li, Ajay Mandlekar, Zhenjia Xu, Jiaojiao Fan, Yashraj Narang, Linxi Fan, Yuke Zhu, Yogesh Balaji, Mingyuan Zhou, Ming-Yu Liu, and Yu Zeng. One-step diffusion policy: Fast visuomotor policies via diffusion distillation. In ICML, 2025.
- [98] Yujie Wei, Jiahan Fan, Jiyu Guo, Ruichen Zhen, Rui Shao, Xiu Su, Zeke Xie, and Shuo Yang. Learning to accelerate vision-language-action models through adaptive visual token caching. arXiv preprint arXiv:2602.00686, 2026.
- [99] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. TinyVLA: Towards fast, data-efficient vision-language-action models for robotic manipulation. RAL, 2025.
- [100] Justin Williams, Kishor Datta Gupta, Roy George, and Mrinmoy Sarkar. Lite vla: Efficient visionlanguage-action control on cpu-bound edge robots. arXiv preprint arXiv:2511.05642, 2025.
- [101] Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, et al. A pragmatic vla foundation model. arXiv preprint arXiv:2601.18692, 2026.
- [102] Yiming Wu, Huan Wang, Zhenghao Chen, Jianxin Pang, and Dong Xu. On-device diffusion transformer policy for efficient robot manipulation. In ICCV, pages 14073–14083, 2025.
- [103] Haozhe Xie, Beichen Wen, Jiarui Zheng, Zhaoxi Chen, Fangzhou Hong, Haiwen Diao, and Ziwei Liu. Dynamicvla: A vision-language-action model for dynamic object manipulation. arXiv preprint arXiv:2601.22153, 2026.
- [104] Zheng Xiong, Kang Li, Zilin Wang, Matthew Jackson, Jakob Foerster, and Shimon Whiteson. Hypervla: Efficient inference in vision-language-action models via hypernetworks. arXiv preprint arXiv:2510.04898,

- 2025.

[105] Chongyang Xu, Yixian Zou, Ziliang Feng, Fanman Meng, and Shuaicheng Liu. Ada3drift: Adaptive training-time drifting for one-step 3d visuomotor robotic manipulation. arXiv preprint arXiv:2603.11984,

- 2026.

- [106] Siyu Xu, Yunke Wang, Chenghao Xia, Dihao Zhu, Tao Huang, and Chang Xu. Vla-cache: Towards efficient vision-language-action model via adaptive token caching in robotic manipulation. In NeurIPS, 2025.
- [107] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [108] Yantai Yang, Yuhao Wang, Zichen Wen, Luo Zhongwei, Chang Zou, Zhipeng Zhang, Chuan Wen, and Linfeng Zhang. EfficientVLA: Training-free acceleration and compression for vision-language-action models. In NeurIPS, 2025.
- [109] Wencheng Ye, Tianshi Wang, Lei Zhu, Fengling Li, and Guoli Yang. Actdistill: General action-guided self-derived distillation for efficient vision-language-action models. arXiv preprint arXiv:2511.18082, 2025.

- [110] Yifan Ye, Jiaqi Ma, Jun Cen, and Zhihe Lu. Token expand-merge: Training-free token compression for vision-language-action models. arXiv preprint arXiv:2512.09927, 2025.
- [111] Wenda Yu, Tianshi Wang, Fengling Li, Jingjing Li, and Lei Zhu. Ac2-vla: Action-context-aware adaptive computation in vision-language-action models for efficient robotic manipulation. arXiv preprint arXiv:2601.19634, 2026.
- [112] Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan Zhang, Ji Zhang, Lianli Gao, Jingkuan Song, Nicu Sebe, and Heng Tao Shen. A survey on efficient vision-language-action models. arXiv preprint arXiv:2510.24795, 2025.
- [113] Yang Yue, Yulin Wang, Bingyi Kang, Yizeng Han, Shenzhi Wang, Shiji Song, Jiashi Feng, and Gao Huang. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. In NeurIPS, pages 56619–56643, 2024.
- [114] Dapeng Zhang, Jing Sun, Chenghui Hu, Xiaoyan Wu, Zhenlong Yuan, Rui Zhou, Fei Shen, and Qingguo Zhou. Pure vision language action (vla) models: A comprehensive survey. arXiv preprint arXiv:2509.19012, 2025.
- [115] Qinglun Zhang, Zhen Liu, Haoqiang Fan, Guanghui Liu, Bing Zeng, and Shuaicheng Liu. Flowpolicy: Enabling fast and robust 3d flow-based policy via consistency flow matching for robot manipulation. In AAAI, pages 14754–14762, 2025.
- [116] Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Yuan Du, and Shanghang Zhang. Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation. In AAAI, 2026.
- [117] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In RSS, 2023.
- [118] Yongsheng Zhao, Lei Zhao, Baoping Cheng, Gongxin Yao, Xuanzhang Wen, and Han Gao. Vla-rail: A real-time asynchronous inference linker for vla models and robots. arXiv preprint arXiv:2512.24673, 2025.
- [119] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. In ICLR, 2026.
- [120] Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, et al. A survey on efficient inference for large language models. arXiv preprint arXiv:2404.14294, 2024.
- [121] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In CoRL, pages 2165–2183, 2023.

### Appendix

We provide the following contents in the appendix:

- • Section A reviews related work on VLA models, real-time VLAs, diffusion acceleration approaches in VLAs, and discusses closely related directions.
- • Section B provides a fine-grained analysis of the asynchronous inference pipeline and clarifies the relationship between inference latency, delay, and minimal execution horizon.
- • Section C presents additional pilot study results that further support the non-uniform sampling difficulty across action indices.
- • Section D gives additional methodological details, including the integration with action conditioning, the streaming client-server interface, and pseudo-code for training and inference.
- • Section E summarizes implementation details for real-world experiments, simulation benchmarks, policy training, and robot deployment.
- • Section F reports additional experimental results, including reaction analysis, extra real-world results, simulation benchmark results, and ablation studies.
- • Section G discusses the limitations of this work and potential directions for future research.
- • Section H discusses potential positive and negative societal impacts.

- A Related Work

Vision-Language-Action Models. Vision-Language-Action (VLA) models [43, 3, 2, 59, 29] extend large-scale vision-language pretraining from Vision-Language Models (VLMs) [1, 107, 44] to embodied action learning, and have demonstrated impressive performance in robotic manipulation. By pretraining on large-scale vision-text-action corpora [6, 69, 91, 41], VLAs enable robots to map multimodal observations and language instructions directly to low-level motor commands, facilitating dexterous manipulation across a wide range of tasks and promoting generalization to diverse and complicated environments [79, 35, 34, 21, 89].

Early approaches such as RT-2 [121] and OpenVLA [43] discretize robot actions into tokens, making them compatible with the auto-regressive objective of VLMs. Subsequent effort explores diffusion- or flow-matching-based action generation [16], adopting continuous action representations to model the multimodal distribution. Represented by methods including π0 [3] and GR00T [2], these approaches incorporate a dedicated action expert alongside the VLM backbone, generating high-quality actions conditioned on vision-language features.

Real-Time VLAs. In contrast to VLMs operating purely in cyberspace, VLAs interact with the physical world and are therefore highly sensitive to real-time interaction [48, 103]. Consequently, improving the efficiency of VLAs has become an active research topic [112, 28]. A straightforward strategy is to reduce model inference latency, with representative directions including:

- • Replacing the VLM backbone (typically 3 ∼ 7B parameters) with smaller models consisting of fewer than 1B parameters [119, 99, 68, 50, 12, 104, 74, 81, 54, 96, 7];
- • Compressing the LLM backbone through mechanisms such as layer selection and early exiting [116, 108, 81, 14, 113, 37, 109, 36, 84, 2, 111];
- • Accelerating action decoding, mainly for auto-regressive VLAs [95, 84, 42, 85, 73];
- • Pruning visual tokens, as multi-view image inputs account for a large proportion of tokens while often introducing perceptual redundancy [110, 60, 25, 46, 72, 22, 92, 40, 47, 106, 33, 98, 111];
- • Applying low-level inference optimizations or quantization techniques [17, 63, 100, 65, 87, 22, 102, 14, 94, 71, 70].

Another line of work seeks to eliminate inter-chunk pauses introduced by standard action chunking and synchronous inference paradigm. By introducing asynchronous execution, VLA models can generate the next action chunk concurrently while the current one is being executed, resulting in nonstop trajectories [81, 118, 103, 9, 58]. However, naively switching between chunks may cause abrupt multimodal transitions and jerky motions, a phenomenon known as inter-chunk discontinuity [4, 49]. To mitigate this issue, RTC [4] inpaints the next action chunk conditioned on the current chunk, while Training-time RTC [5], REMAC [93], and VLASH [88] condition the model on the predicted actions.

Different from prior work, FASTER is the first real-time VLA that explicitly targets responsiveness by accelerating the sampling of immediate actions. Notably, it requires no architectural modifications,

Policy Server

Infer

Infer

time

Robot Client

| | |
|---|---|
| | |

- Figure 7: Temporal pipeline of asynchronous inference at a fine-grained level. Suppose the inference latency ∆tinfer is 2.5 times the controller period ∆tctrl, resulting in an inference delay of d = 2 and a minimal execution horizon of smin = 3.

rendering it orthogonal and complementary to aforementioned efficient VLA techniques, such as LLM compression and token pruning.

Diffusion Acceleration in VLAs. A closely related line of work aims to distill multi-step diffusion or flow matching computation in policies into one-step models, or to train one-step models directly. We focus on the VLA context rather than conventional diffusion policies [78, 82, 115, 11, 38, 18, 97, 61, 26, 19, 45, 105, 39]. Distillation-based methods [55, 62, 75] typically involve two stages: first training a standard multi-step model and then distilling it into a one-step model, which significantly increases training cost. Direct one-step training [13, 80] often requires model modifications, such as additional inputs in Shortcut Model [23] or alternative objectives in MeanFlow [27]. These changes make it difficult to fine-tune pretrained VLAs while preserving their strong capabilities. In contrast, FASTER requires no architectural modifications and can be seamlessly integrated into the standard fine-tuning pipeline of flow-based VLAs with the same optimization objective.

Another related direction explores streaming policies, where the immediate action is also produced within a single sampling step [20, 15, 31]. However, these approaches update the observation input at every step, which introduces a substantial inference burden when applied to VLAs. Updating observations requires a forward pass through the VLM backbone at each step, making real-time VLA deployment impractical. In contrast, our method is specifically designed for VLAs: the VLM backbone is forwarded only once per action chunk, thereby avoiding the primary computational bottleneck in VLA inference. The immediate action is generated through one-step sampling of the action expert after VLM prefilling, while subsequent actions are progressively completed through iterations of the action expert based on the same VLM representations.

#### B Details of Asynchronous Inference Pipeline

Here we provide a fine-grained illustration of the asynchronous inference pipeline. Since the robot controller typically operates at a fixed frequency, all operations occur with an interval of control period ∆tctrl. As shown in Figure 7, the inference latency generally does not align with the controller timesteps in real-world robotic systems. Therefore, if inference starts at time t0, the predicted actions can only be executed at time t0 + ⌈∆tinfer/∆tctrl⌉ · ∆tctrl. However, the next action is expected to be available at t0 + ∆tctrl (one control step later), resulting in a delay of d actions:

d := ⌈∆tinfer/∆tctrl⌉ − 1 = ⌊∆tinfer/∆tctrl⌋. (7)

Note that ∆tinfer almost never lies exactly on a control-period boundary. Once the current inference finishes, the next inference can only be triggered at the subsequent controller timestep (e.g., t3 in Figure 7). Consequently, the inference interval is at least ⌈∆tinfer/∆tctrl⌉ · ∆tctrl. As discussed in Table 1, the inference interval is related to the execution duration ∆texec, and the minimum feasible execution horizon is therefore smin := ⌈∆tinfer/∆tctrl⌉.

0.08

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.07

0.06

SAstraightness()

0.05

0.04

0.03

0.02

0.01

0.00

0 5 10 15 20 25 30

action index

(a)

×10−5

1.2

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

1.0

SAstraightness()

0.8

0.6

0.4

0.2

0.0

0 5 10 15 20 25 30

action index

(c)

×10−6

- 0
- 1
- 2
- 3
- 4
- 5

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

SAstraightness()

0 5 10 15 20 25 30

action index

(e)

0.07

| |[Figure 25]| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 26]

- 0 5 10 15 20 25 30 action index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

0.06

0.05

←samplingstep

0.04

0.03

0.02

0.01

0.00

(b)

0.004

| |[Figure 27]| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 28]

- 0 5 10 15 20 25 30 action index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

0.003

←samplingstep

0.002

0.001

0.000

(d)

| |[Figure 29]| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 30]

- 0 5 10 15 20 25 30 action index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10

0.008

←samplingstep

0.006

0.004

0.002

0.000

(f)

- Figure 8: Additional visualizations of (a,c,e) straightness S(A) and (b,d,f) differences between the

intermediate clean action estimates and the final output. Panels (a,b) are computed using a π0.5 model fine-tuned on Pick Beverage task with prediction horizon H = 30; (c,d) use an X-VLA model fine-tuned on the LIBERO simulation dataset with H = 30; and (e,f) use an X-VLA model fine-tuned on the CALVIN ABC simulation dataset with H = 30. The shaded regions in (a,c,e) denote the ±2 SEM across 200 random samples.

#### C Additional Results in Pilot Study

We present additional results from our pilot study in Section 3.2. We analyze the effect of using a shorter prediction horizon, H = 30, in Figure 8(a,b), compared with the default setting of H = 50 in Figure 3(a,b) using π0.5. To validate generalization across models and data sources, we further report results using an X-VLA model on the LIBERO simulation dataset in Figure 8(c,d) and on the CALVIN ABC dataset in Figure 8(e,f). Although the straightness metrics exhibit larger variance under certain conditions, all metrics reveal a similar non-uniform pattern across action indices, consistent with Figure 3. These results further support our observation that earlier actions are easier to sample than later ones, which forms the foundation for the motivation behind FASTER.

#### D Additional Methodological Details

##### D.1 Horizon-Aware Schedule with Action Conditioning

We provide details on how the Action Conditioning strategy [5] is integrated into our Horizon-Aware Schedule. It is also worth noting that action prefix unimodalizes the distribution for the immediate next action. Because the subsequent action is heavily constrained by the previous overlapping chunk, it generally does not demand multi-step expressivity.

Table 4: Illustration of inference latency vs. sequential execution. “Index” denotes the index of valid actions in the chunk, excluding delayed actions. “Time Req.” is the time at which an action is required by the robot controller, while “Time Rec.” is the time at which an action is received from the policy server via the streaming interface, reported as mean ± standard deviation over 20 trials. All timestamps are measured relative to the start of inference.

GPU π0.5 X-VLA

Index Time Req. Time Rec. Index Time Req. Time Rec.

- 1 66.7ms 62.1±3.1ms 1 66.7ms 44.8±0.3ms

- 2 100.0ms 70.2±3.1ms 2 100.0ms 52.0±0.1ms

- 3 133.3ms 77.1±3.2ms 3 133.3ms 59.6±0.3ms

RTX 4090

- 7 266.7ms 238.6±1.9ms 3 133.3ms 129.2±2.4ms

- 8 300.0ms 253.3±2.5ms 4 166.7ms 159.0±3.6ms

- 9 333.3ms 266.9±2.6ms 5 200.0ms 186.8±4.3ms

RTX 4060

In the main paper, we use the superscript j in ρj and τij to denote the j-th sampling step. Here we omit j, as the following description applies to both training and sampling. Specifically, given an

action prefix of length d < H, we set the local flow matching timesteps of the prefix actions to zero1 (i.e., τi ≡ 0 for i < d) and apply an offset −d to the action indices. The hit time formulation is accordingly updated from Equation (5) to:

α

i − d max(H − 1 − d,1)

· ud, i ∈ [d + 1,H − 1], (8)

u = {ui} = 1 −

where ud replaces u0 as the predefined global timestep at which the first valid action (now the action at index d) is finalized. Consequently, the immediate action still reaches its hit time ud = (N −1)/N with a single sampling step, ensuring low latency while preserving the temporal consistency of flow matching. During inference, we simply set d to the real inference delay, where the action prefix corresponds to the overlapping actions [s,s + d − 1] from the previous chunk.

During training, d is sampled uniformly from 0 to dmax to simulate varying Time to First Action (TTFA) on distinct devices. An action prefix mask m is introduced to mask the loss function:

m = {mi} = 1(i ≥ d), i ∈ [0,H − 1], (9) where 1(·) denotes the indicator function. The overall learning objective is formulated as:

0, i < d, max(0,(ρ − ui)/(1 − ui)), i ≥ d,

(10)

τ = {τi} =

2

m ⊙ vθ(ot,Aτt ,τ) − (ϵ − Aˆ t)

, (11)

L(θ) = Eρ∼U(0,1), d∼U{0,d

max}

∥m∥1

where ⊙ is element-wise multiplication and the ℓ1 norm ∥m∥1 denotes the number of elements equal to one. Therefore, the loss is computed only over the suffix actions.

##### D.2 Streaming Client-Server Interface

Unlike conventional action chunking pipelines that transmit an entire action chunk as a single large payload, the streaming client-server interface in FASTER employs a progressive streaming mechanism, decomposing the transmission into a sequence of high-frequency, smaller packets. While this mechanism increases the total completion time for a full chunk due to network communication (particularly for the final action), it aligns naturally with the sequential characteristic of robotic execution.

Suppose the i-th action ai is executed at controller time t. The system only requires the (i + 1)-th action ai+1 to be available at t + ∆tctrl. As long as the latency for acquiring ai+1 remains below the control period ∆tctrl, the controller avoids pipeline stalls. We provide detailed illustrations in Table 4, including latency measurements for two VLAs on RTX 4090 and RTX 4060 GPUs, respectively.

1The notation of timestep is opposite in Training-time RTC [5], so they set the timesteps to one in their formulation.

Under this paradigm, the network latency incurred by subsequent actions is effectively masked by the execution time of preceding actions. Consequently, the marginal delay for actions at the end of a chunk becomes functionally negligible and imperceptible during real-time execution.

##### D.3 Algorithms

The pseudo-code for VLA policy training and inference in FASTER are provided in Algorithm 1 and Algorithm 2, respectively.

- Algorithm 1 FASTER policy training

Input: Training dataset D, pretrained VLA model vθ, prediction horizon H, HAS parameters α,ud,

mixing probability p, max prefix length dmax Output: Fine-tuned VLA model vθ′

- 1: for each training step do
- 2: Sample: data (ot,Aˆ t) ∼ D, noise ϵ ∼ N(0,I), global timestep ρ ∼ U(0,1), prefix length d ∼ U{0,dmax}, schedule type z ∼ Bernoulli(p)
- 3: Compute action prefix mask m: m = {mi} = 1(i ≥ d), i ∈ [0,H − 1],
- 4: if z = 1 then ▷ Use Horizon-Aware Schedule
- 5: Compute hit times u:

u = {ui} = 1 −

i − d max(H − 1 − d,1)

α

· ud, i ∈ [d + 1,H − 1],

- 6: Compute local timesteps τ:

τ = {τi} =

0, i < d, max(0,(ρ − ui)/(1 − ui)), i ≥ d,

- 7: else ▷ Use Constant Schedule with Action Conditioning
- 8: Set local timesteps τ:
- 9:

τ = {τi} =

0, i < d, ρ, i ≥ d,

- 10: end if
- 11: Construct noisy actions: Aτt = τ ⊙ ϵ + (1 − τ) ⊙ Aˆ t
- 12: Compute masked loss:

L(θ) =

m ⊙ vθ(ot,Aτt ,τ) − (ϵ − Aˆ t)

2

∥m∥1

,

- 13: Update model parameters θ using ∇θL(θ)
- 14: end for

- Algorithm 2 FASTER policy inference

Input: Fine-tuned VLA model vθ′ , sampling steps N, prediction horizon H, HAS parameters α,ud,

observation ot, delay d, action prefix Apre, execution horizon s

- 1: Initialize At ∼ N(0,I)
- 2: Compute hit times u

u = {ui} = 1 −

i − d max(H − 1 − d,1)

α

· ud, i ∈ [d + 1,H − 1],

- 3: Forward VLM backbone with ot
- 4: for j = 1 to N do
- 5: Set global timestep ρj = (N − j + 1)/N
- 6: Compute local timesteps τj w.r.t. ρj:

τj = {τij} =

0, i < d, max 0,(ρj − ui)/(1 − ui) , i ≥ d,

- 7: Similarly, compute next local timesteps τj+1 w.r.t. ρj+1 ▷ ρN+1 := 0
- 8: Overwrite actions prefix At[0 : d − 1] with Apre
- 9: Predict velocity vj = vθ(ot,At,τj) ▷ Forward AE only
- 10: Compute delta timestep ∆τ = τj+1 − τj
- 11: Euler update At ← At + vj ⊙ ∆τ
- 12: for each valid action index i ∈ [d,H − 1] do
- 13: if τij+1 = 0 and action i has not been streamed then
- 14: Dispatch at+i to client immediately ▷ Streaming Interface
- 15: end if
- 16: end for
- 17: if all actions within execution horizon [d,d + s − 1] have been finalized then
- 18: break ▷ Early stopping
- 19: end if
- 20: end for

[Figure 31]

[Figure 32]

- Figure 9: AgileX Cobot Magic robotic platform with Piper arms.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

FoldTowelPickBeverage

- Figure 10: Visualization of Pick Beverage and Fold Towel tasks.

#### E Implementation Details

##### E.1 Real-world Experiments

Hardware Setup. We use the AgileX Cobot Magic robotic platform shown in Figure 9, designed in the Mobile Aloha style [117, 24]. It is equipped with four 6-DoF Piper robot arms: two leader arms for human teleoperation and two follower arms for data collection and policy inference. A multi-camera setup is used, consisting of one front camera (RealSense D455) and two wrist cameras (RealSense D435), each mounted on a follower arm.

Tasks. We evaluate three real-robot tasks: “Table Tennis”, “Pick Beverage”, and “Fold Towel”. The visualization of Table Tennis task is already provided in the main paper, while illustrations of the other two tasks are shown in Figure 10. We collect 335 demonstration episodes (approximately 14 minutes) for Table Tennis task and 150 episodes for the other two tasks via human teleoperation. All data is recorded at 30 FPS.

The language instructions for the tasks are as follows:

- • Table Tennis: “Hit the table tennis ball to the opponent.”
- • Pick Beverage: “Pick up the beverage and put it in the plastic basket.”
- • Fold Towel: “Fold the towel.”

Evaluation. The tasks are evaluated with a fixed number of rollouts: 15 trials for Table Tennis, 35 trials for Pick Beverage, and 10 trials for Fold Towel. For Pick Beverage and Fold Towel, we follow predefined test cases with fixed object types, positions, and orientations to ensure fair comparison. For Table Tennis, we regulate the ball speed to the best of our ability.

We define fine-grained evaluation metrics to more precisely assess the real-world performance of the models under a limited number of rollouts. Results are reported by averaging scores across sub-steps and rollouts. The scoring criteria are as follows:

- 1. Table Tennis

• Step 1: Hitting the table tennis ball with the racket

- – 0 point: The robot misses the ball.
- – 0.5 point: The robot returns the ball but produces a weak hit due to reaction latency; the ball travels only a short distance before landing on the table. The distance is measured using a marked line on the table.
- – 1 point: The robot performs a powerful return, and the ball travels a significant distance before landing on the table.

- 2. Pick Beverage

- • Step 1: Grasping the beverage

- – 0 point: The robot fails to grasp the object.
- – 0.5 point: The robot grasps the object after multiple (up to five) attempts and recovers from errors.
- – 1 point: The robot successfully grasps the object on the first attempt.

- • Step 2: Placing the beverage into the basket

- – 0 point: The object is not placed into the basket or is dropped.
- – 0.5 point: The robot places the object into the basket but causes a collision.
- – 1 point: The robot successfully places the object into the basket on the first attempt.

- 3. Fold Towel

- • Step 1: Grasping the towel

- – 0 point: The robot fails to grasp both sides of the towel.
- – 0.5 point: The robot grasps both sides after multiple (up to five) attempts and recovers from errors.
- – 1 point: The robot successfully grasps both sides on the first attempt.

- • Step 2: Forward folding the towel

- – 0 point: The robot fails to fold the towel forward (e.g., slippage from the grippers).
- – 0.5 point: The towel is folded forward but not well aligned.
- – 1 point: The robot folds the towel forward perfectly.

- • Step 3: Grasping the towel again

- – 0 point: The robot fails to grasp both sides of the towel.
- – 0.5 point: The robot grasps both sides after multiple (up to five) attempts and recovers from errors.
- – 1 point: The robot successfully grasps both sides on the first attempt.

- • Step 4: Backward folding the towel

- – 0 point: The robot fails to fold the towel backward (e.g., slippage from the grippers).
- – 0.5 point: The towel is folded backward but not well aligned.
- – 1 point: The robot folds the towel backward perfectly.

##### E.2 Simulation Benchmarks

LIBERO. The LIBERO benchmark [53] consists of four task suites: Spatial, Object, Goal, and 10 (Long), each targeting distinct aspects of embodied capabilities. Following prior work [35, 119], we train a single policy across all four suites. For training π0.5, we use the dataset provided by OpenVLA2 [43] and convert it to the LeRobot format [8] using the script in openpi. While for

2HuggingFace: openvla/modified_libero_rlds

Table 5: Hyperparameter settings for fine-tuning VLA models.

Model π0.5 X-VLA

Task AgileX Simulation AgileX Simulation Prediction Horizon H 50 10 30 30

Action Space Relative Joint Delta EEF Abs EE6D Abs EE6D Global Batch Size 128 256 128 128

Training Steps 50k 30k 50k 60k

Optimizer AdamW AdamW AdamW AdamW Weight Decay 0 0 0 0

Betas (0.9, 0.95) (0.9, 0.95) (0.9, 0.95) (0.9, 0.95)

Base LR 2.5e-5 5e-5 1e-4 1e-4 LR Scheduler Cosine Decay Cosine Decay Constant Constant Warmup Steps 1k 10k 2k 2k

Grad Norm Clip 1.0 1.0 1.0 1.0 EMA Decay 0.99 0.999 N/A N/A

- Table 6: Settings of inference delay d and execution horizon s on RTX 4090 and RTX 4060 GPUs. “Async” refers to Naive Async and Training-time RTC.

RTX 4090 RTX 4060

Task Method

d s d s

Sync 4 5 10 11 Async 4 5 10 11

Table Tennis

FASTER 3 4 8 10

Sync 4 50 10 50 Async 4 46 10 40

Pick Beverage, Fold Towel

FASTER 3 47 8 42

training X-VLA, we use the HDF5-format dataset provided by X-VLA authors3, which aligns with the EE6D action space. Each suite contains 10 tasks, and every task is evaluated over 50 trials.

CALVIN. The CALVIN benchmark [67] comprises 34 tasks with unconstrained language instructions spanning diverse manipulation skills. We adopt the widely used ABC→D evaluation setting and use the LeRobot-format dataset4. Each model is evaluated on 1,000 unique instruction chains, where each chain consists of five consecutive tasks. Performance is measured by the average number of successfully completed tasks per chain.

Kinetix. The Kinetix benchmark5 is proposed by RTC [4] and contains 12 dynamic tasks in the Kinetix simulation environment [66]. Since these tasks involve dynamic motions with force-based control, they are suitable for evaluating asynchronous execution methods under varying inference delays. However, the policy used in this benchmark is a simple 4-layer MLP rather than a VLA model, so it is not the primary focus of our study.

Following prior work [4, 5], the policy uses a prediction horizon of H = 8. During training, the simulated delay is randomly sampled from 0 to 4 with exponentially decreasing weights. We resume training from the 24th epoch and fine-tune the policy for 8 epochs using our mixed schedule. The number of sampling steps is set to 5, and accordingly we set ud = 0.8. Performance is evaluated over 2,048 rollouts, and we report the average success rates.

- 3HuggingFace: 2toINF/Libero-XVLA-format
- 4HuggingFace: InternRobotics/InternData-Calvin_ABC 5GitHub: Physical-Intelligence/real-time-chunking-kinetix

- Table 7: Detailed comparison of reaction capability on RTX 4090 and RTX 4060 GPUs. The inference latency ∆tinfer corresponds to TTFA, and execution duration ∆texec is computed as smin · ∆tctrl. The distribution of reaction time Dreact is U(∆tinfer,2 ∗ ∆tinfer + ∆texec) for Sync, and U(∆tinfer,∆tinfer + ∆texec) for Async and FASTER.

RTX 4090 RTX 4060

Model Method

∆tinfer ∆texec Dreact ∆tinfer ∆texec Dreact

Sync 80.0ms 100.0ms U(80.0, 260.0) 303.3ms 333.3ms U(303.3, 939.9) Async 80.0ms 100.0ms U(80.0, 180.0) 303.3ms 333.3ms U(303.3, 636.6)

π0.5

FASTER 62.1ms 100.0ms U(62.1, 162.1) 238.6ms 266.7ms U(238.6, 505.3)

Sync 113.7ms 133.3ms U(113.7, 360.7) 399.5ms 400.0ms U(399.5, 1199.0) Async 113.7ms 133.3ms U(113.7, 247.0) 399.5ms 400.0ms U(399.5, 799.5)

X-VLA

FASTER 44.8ms 66.7ms U(44.8, 111.5) 129.2ms 200.0ms U(129.2, 329.2)

##### E.3 Training Setup

We use the official codebases of π0.56 and X-VLA7 to fine-tune the VLA models. We follow their default or recommended configurations, with key hyperparameters listed in Table 5. Our real-robot dataset is recorded in the absolute joint space. Actions are converted to the relative joint space by computing offsets from the proprioceptive states, and to the absolute EE6D space using forward kinematics. All models are trained on 8 NVIDIA A800 GPUs with 80GB VRAM from cloud provider, requiring 1-2 days for π0.5 and 6-7 hours for X-VLA.

Regarding the hyperparameters of our proposed Horizon-Aware Schedule, we use α = 0.6 to control the hit times and p = 0.5 for the mixed schedule by default. When training X-VLA on LIBERO and CALVIN, we set α to 0.7; ablation studies are provided in Section F.4. Since both VLAs use 10 sampling steps for flow matching, we set u0 = 0.9 to ensure that the immediate action requires only a single step, as described in the main paper.

Although the prediction horizon is 50 or 30, we set the maximum prefix length dmax to 10 during fine-tuning real-world tasks, which simulate TTFA up to 333.3ms. This design is motivated by practical considerations: inference delays exceeding 333.3ms can render real-time reactive control functionally infeasible for certain tasks. By limiting the maximum prefix length, we prevent the model from wasting capacity on physically unrealistic scenarios. This range already provides sufficient coverage even for an RTX 4060 GPU, thereby ensuring better sample efficiency and smoother flow matching in the critical low-latency regime.

##### E.4 Robot Deployment Setup

As mentioned in the main paper, the ROS system on the robot runs at a fixed 30Hz frequency, i.e., with a control period ∆tctrl = 33.3ms. The robot is connected to policy server via LAN and communicates through websocket protocol. We define the inference delay as d := ⌊∆tinfer/∆tctrl⌋ and the minimum execution horizon as smin := ⌈∆tinfer/∆tctrl⌉. To account for additional overhead from network communication, local processing, memory I/O, and other system costs, we set these values slightly larger in real-robot deployment, as listed in Table 6.

For the table tennis task, we set the execution horizon to smin to achieve optimal reaction capability. For other tasks that do not heavily depend on high-frequency control, we set the execution horizon to the number of valid actions (H for Sync, H − d for Async and FASTER) to ensure smooth action trajectories.

- 6Github: Physical-Intelligence/openpi
- 7Github: 2toinf/X-VLA

Sync Naive Async Training-time RTC FASTER

1.0

23.4

25

21.7 22.5 21.5

0.8

20

Duration(s)

0.6

0.513

15

Score

0.488

11.4 10.2 11.1 9.9

0.371 0.336

0.313 0.338

0.4

10

0.293

0.243

0.2

5

0

0.0

Pick Beverage Fold Towel

Pick Beverage Fold Towel

- Figure 11: Comparison of real-world performance and task completion duration on two real-world tasks using X-VLA. Note that the duration is computed only from successful rollouts, and therefore is not directly comparable to the results in Figure 6.

- Table 8: Performance statistics for the Table Tennis task. The 96% confidence intervals are computed using bootstrap resampling.

RTX 4090 RTX 4060

Model Method

Mean 96% CI Mean 96% CI

Sync 0.000 (0.000, 0.000) 0.000 (0.000, 0.000) Naive Async 0.200 (0.075, 0.325) 0.200 (0.067, 0.333)

π0.5

Training-time RTC 0.533 (0.300, 0.767) 0.300 (0.133, 0.500) FASTER 0.800 (0.667, 0.933) 0.467 (0.333, 0.567)

#### F Additional Experimental Results

##### F.1 Additional Analysis on Reaction Speed

We provide Table 7 to complement the reaction time details in Table 2. The probabilities reported in Table 3 are also derived from the distributions Dreact, defined as the probability that one independently sampled ∆treact from a method is smaller than the other method. As discussed in the main paper, FASTER is deterministically superior on X-VLA, since its upper bound of reaction time is lower than the baselines’ lower bound: 111.5ms vs. 113.7ms on RTX 4090, and 329.2ms vs. 399.5ms on RTX 4060.

##### F.2 Additional Real-world Experiments

We evaluate FASTER on real-world Pick Beverage and Fold Towel tasks using the X-VLA [119] model and present the results in Figure 11. We observe that though X-VLA performs significantly worse than the state-of-the-art VLA π0.5, our method consistently achieves better or comparable results in terms of both completion scores and task duration. It is worth noting that Naive Async often fails due to unstable actions caused by inter-chunk discontinuities.

We supplement the performance statistics of Figure 5 and Figure 6 in Table 8 and Table 9, respectively.

##### F.3 Simulation Benchmarks

We conduct experiments on two widely used simulation benchmarks in VLA research: LIBERO [53] and CALVIN [67]. Although these simulation environments are not directly affected by inference latency, they provide a controlled protocol for evaluating whether FASTER preserves the original model performance. As shown in Table 10, HAS maintains competitive performance on both benchmarks, with limited degradation despite its aggressive action sampling strategy. These results suggest that, although FASTER may slightly affect long-horizon accuracy in simulation, it provides a favorable trade-off for latency-sensitive real-world settings.

We further evaluate on the Kinetix benchmark [4] to compare with state-of-the-art real-time methods. We include additional baselines: Bidirectional Decoding (BID) [57], Inference-time RTC [4], VLASH [88], and REMAC [93]. Since flow-matching sampling is applied to the entire model in this benchmark, FASTER reduces the inference latency of the immediate action by 5×. We conduct fair comparisons under the same wall-clock inference budget: given the maximum supported delay is 4,

- Table 9: Performance statistics for the Pick Beverage and Fold Towel task. The 96% confidence intervals are computed using bootstrap resampling.

Model Method

Pick Beverage Fold Towel

Score Duration Score Duration

Mean 96% CI Mean Std Mean 96% CI Mean Std

π0.5

Sync 0.879 (0.786, 0.950) 13.0 0.9 0.788 (0.600, 0.925) 24.7 0.5 Naive Async 0.957 (0.886, 1.000) 12.5 1.2 0.825 (0.613, 0.988) 24.0 2.4

Training-time RTC 0.950 (0.879, 0.993) 11.9 2.4 0.888 (0.700, 1.000) 20.7 0.4 FASTER 0.957 (0.886, 1.000) 12.0 0.9 0.963 (0.925, 1.000) 20.5 0.4

- Table 10: Comparison of performance on simulation benchmarks LIBERO and CALVIN. “∗” indicates experimental results evaluated on our cluster with official checkpoints.

LIBERO CALVIN ABC→D

Method

Spatial Object Goal 10 Avg. 1 2 3 4 5 Avg. Len π0.5 98.8 98.2 98.0 92.4 96.9 94.2 88.7 85.7 83.2 79.5 4.313

π0.5+FASTER 98.6 97.8 97.8 91.6 96.5 95.1 89.1 85.0 81.9 78.1 4.292

X-VLA∗ 97.8 99.4 97.8 96.8 98.0 95.7 89.8 82.4 77.0 70.2 4.151 X-VLA+FASTER 97.8 99.4 98.2 93.0 97.1 98.1 91.4 82.4 72.9 64.5 4.093

we compare the baselines at d = 4 with FASTER at d = 1 under the same execution horizon. As shown in Table 11, FASTER outperforms all baselines by a clear margin, highlighting the importance of reducing reaction latency in asynchronous execution, even in simulated environments.

##### F.4 Ablation Study

We conduct ablation studies to analyze the impact of our proposed methods on task performance. Since real-world experiments are difficult to repeat with a large number of rollouts and are easily affected by environmental factors, we use simulation benchmarks for more controlled evaluation. Moreover, as the LIBERO benchmark is highly saturated, we focus on CALVIN to better highlight long-horizon performance.

We first investigate the factor α in the Horizon-Aware Schedule, which controls the decreasing trend of hit times across the action index. As shown in Figure 12, a smaller α leads to a faster decay of hit times, allocating more denoising steps to future actions. From Table 12, we observe that HAS is robust to different values of α, except when α = 1.0, with the largest difference in Avg. Len being only 0.18.

We further analyze the influence of the mixed schedule in FASTER training. In addition to varying the mixing probability p between HAS and the conventional constant schedule (p = 0), we also include the independent time schedule proposed in Diffusion Forcing [10], where the timestep is sampled independently across the action index during training while HAS is used for sampling.

As shown in Table 13, a small value of p leads to degraded performance, since the inference-time schedule is observed less frequently during training. As discussed in the main paper, fine-tuning a pretrained VLA primarily with HAS, i.e., using a large value of p, can also introduce severe adverse effects, particularly on long-horizon performance. Although training with the independent schedule achieves reasonable accuracy, it is not directly comparable to our training strategy: independently sampling timesteps for each action in training enlarges the search space and can induce inconsistencies between training and inference [86]. These results highlight the importance of the mixed schedule in FASTER. We therefore use the default value p = 0.5 for all reported results and do not further tune p.

We also evaluate different values of α in real-world experiments with two models and observe no clear performance differences. This suggests that task-specific hyperparameter tuning is not necessary for HAS, and that our default configuration generalizes well across different settings.

##### F.5 Error Analysis

We include an error analysis on real-world task to better understand how accurately FASTER generates actions. We use fine-tuned π0.5 models with the constant schedule and HAS, and conduct open-loop

Table 11: Comparison of performance on Kinetix benchmark under the same inference budget.

Method d s Solve Rate Naive Async 4 4 0.492

BID [57] 4 4 0.553 Inference-time RTC [4] 4 4 0.614 Training-time RTC [5] 4 4 0.726

REMAC [93] 4 4 0.779 VLASH [88] 4 4 0.813

FASTER 1 4 0.869

- α=0.4

- α=0.5

- α=0.6

- α=0.7

- α=0.8

- α=0.9

Table 12: Ablation study of factor α in the Horizon-Aware Schedule, using X-VLA on the CALVIN benchmark. Mixing probability p is set to 0.5.

0.8

0.6

α=1.0

uhittime

CALVIN ABC→D

α

1 2 3 4 5 Avg. Len

0.4

- 0.4 96.7 91.3 82.5 73.1 63.5 4.071

- 0.5 95.1 88.4 79.3 69.6 58.7 3.911

- 0.6 97.5 89.5 80.2 71.2 60.7 3.991

- 0.7 98.1 91.4 82.4 72.9 64.5 4.093

- 0.8 95.9 88.7 79.7 71.2 61.5 3.970

- 0.9 99.0 88.1 75.2 70.3 59.4 3.921 1.0 94.7 84.0 71.7 61.3 51.8 3.635

0.2

0.0

0 5 10 15 20 25 30

action index

- Figure 12: Hit times used in ablation study, with factor α from 0.4 to 1.0.

tests to compare the inferred actions against ground-truth demonstrations. Specifically, we measure the normalized mean absolute error for each action dimension across different action chunk indices.

- As shown in Figure 13, both methods exhibit an overall increase in prediction error as the action index moves farther into the future, which echoes our main insight that early actions are easier to predict. Compared with the baseline, FASTER shows moderately higher errors on some action dimensions, but these errors are distributed more uniformly across chunk indices, reflecting the accuracy-latency trade-off introduced by accelerated sampling. Nevertheless, the error remains relatively low for near-term actions, which are directly executed for immediate reaction. This suggests that HAS preserves sufficient precision for latency-critical actions while enabling substantially faster response in real-time control.

#### G Limitations and Future Work

Despite the improved reaction capability demonstrated by FASTER, several limitations remain. First, our method is primarily applicable to flow-based or diffusion-based VLA models whose action generation process involves iterative sampling. The actual latency reduction also depends on the implementation and hardware. For example, we observe that in JAX implementations, the forwardpass runtime is not always proportional to the number of iterations, which may limit the practical speedup obtained.

Second, FASTER is a deployment-oriented inference scheduling and streaming framework rather than a fundamental modeling advance. Thus, it does not by itself resolve limitations inherited from the underlying VLA model, such as perception failures, language grounding errors, or inaccurate action generation. The robot may still fail if the base policy produces incorrect actions for the task.

Third, our reaction time analysis relies on a simplified timing model. In particular, we assume that sensing, communication, pre-processing, model inference, and action dispatch can be approximated by a single effective latency term. These assumptions lead to a uniform reaction-time distribution that provides a useful first-order characterization of responsiveness, but may not fully capture all

Table 13: Ablation study of mixing probability p in the mixed schedule, using X-VLA on the CALVIN benchmark. Factor α is set to 0.7.

CALVIN ABC→D

p

1 2 3 4 5 Avg. Len Baseline 95.7 89.8 82.4 77.0 70.2 4.151

0.3 93.7 85.2 76.0 65.5 55.2 3.756 0.5 98.1 91.4 82.4 72.9 64.5 4.093 0.7 89.6 76.7 63.2 50.8 40.3 3.206

1.0 (w/o mixed) 89.0 74.7 60.4 49.0 38.1 3.112 Independent 91.4 82.6 74.0 64.6 54.5 3.671

Baseline

###### FASTER

1.0

1.0

| |[Figure 45]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

|[Figure 46]| |
|---|---|
| | |
| | |
| | |
| | |

| |[Figure 47]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

|[Figure 48]| |
|---|---|
| | |
| | |
| | |
| | |

- 0 10 20 30 40 50 action chunk index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12

- 0 10 20 30 40 50 action chunk index
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12

0.8

0.8

intra-actionindex

intra-actionindex

0.6

0.6

0.4

0.4

0.2

0.2

0.0

0.0

- Figure 13: Open-loop action prediction error on the Fold Towel task. We compare the π0.5 baseline with FASTER. The heatmaps report the mean absolute error for each action dimension and action chunk index, normalized by the maximum error of each action dimension computed over 200 random samples and shared across both models.

deployment conditions. In practice, system latency may vary due to CPU/GPU scheduling, network jitter, memory contention, or OS overhead.

Finally, the Horizon-Aware Schedule introduces a trade-off between responsiveness and action accuracy. Although our experiments show that FASTER largely preserves task performance and often improves real-world execution by reducing delay, aggressive early sampling may slightly perturb the original generation trajectory, especially for tasks that require precise long-horizon coordination. Better adaptive schedules that depend on uncertainty, task phase, or online feedback may further improve this trade-off.

#### H Broader Impacts

FASTER aims to improve the real-time responsiveness of VLA models for physical robot control. Its positive impacts include enabling more reliable robot behavior in dynamic environments, reducing the hardware requirements for deploying generalist policies, and making real-time embodied AI systems more accessible on consumer-grade GPUs.

- At the same time, it introduces potential risks. More reactive robots can execute actions faster in the physical world, which may amplify the severity of failures when VLAs make incorrect predictions, misinterpret instructions, or encounter out-of-distribution observations. The method could also contribute to broader automation capabilities. While this can improve productivity, it also raises concerns about labor displacement or misuse.

Our experiments are conducted in controlled research settings. We do not recommend deploying such robot systems in safety-critical or unsupervised human-facing environments without additional risk assessment, safety monitors, and appropriate physical safeguards.

