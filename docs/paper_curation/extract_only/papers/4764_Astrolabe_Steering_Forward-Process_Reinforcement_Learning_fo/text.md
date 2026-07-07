## Astrolabe: Steering Forward-Process Reinforcement Learning for Distilled Autoregressive Video Models

Songchun Zhang1, Zeyue Xue2,3, Siming Fu2, Jie Huang2, Xianghao Kong1, Yue-Ma1, Haoyang Huang2, Nan Duan2§, and Anyi Rao1§

# arXiv:2603.17051v1[cs.CV]17Mar2026

1HKUST 2JD Explore Academy 3HKU Project Page

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Reward-Forcing Self-Forcing Self-Forcing + Ours

[Figure 7]

A fish driving a tiny submarine, exploring an underwater city

[Figure 8]

Single-prompt Short Video

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

Causal Forcing Causal Forcing

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

+Ours +Ours

[Figure 29]

Single-prompt Long Video

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

LongLive LongLive

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

+Ours

+Ours

[Figure 50]

Multi-prompt Long Video

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Inf-RoPE Inf-RoPE

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

+Ours +Ours

Fig. 1: Astrolabe efficiently aligns distilled streaming video models with human preferences without re-distillation, enhancing baselines (e.g., Causal Forcing [64], LongLive [51] and Infinite-RoPE [53]) by mitigating artifacts and improving temporal consistency. We demonstrate boosted perceptual quality across: (Top) single-prompt short, (Middle) single-prompt long, and (Bottom) multi-prompt long video generation.

Abstract. Distilled autoregressive (AR) video models enable efficient streaming generation but frequently misalign with human visual preferences. Existing reinforcement learning (RL) frameworks are not naturally suited to these architectures, typically requiring either expensive

re-distillation or solver-coupled reverse-process optimization that introduces considerable memory and computational overhead. We present Astrolabe, an efficient online RL framework tailored for distilled AR models. To overcome existing bottlenecks, we introduce a forward-process RL formulation based on negative-aware fine-tuning. By contrasting positive and negative samples directly at inference endpoints, this approach establishes an implicit policy improvement direction without requiring reverse-process unrolling. To scale this alignment to long videos, we propose a streaming training scheme that generates sequences progressively via a rolling KV-cache, applying RL updates exclusively to local clip windows while conditioning on prior context to ensure long-range coherence. Finally, to mitigate reward hacking, we integrate a multi-reward objective stabilized by uncertainty-aware selective regularization and dynamic reference updates. Extensive experiments demonstrate that our method consistently enhances generation quality across multiple distilled AR video models, serving as a robust and scalable alignment solution.

Keywords: Video Generation · Distilled Autoregressive Models · Reinforcement Fine-tuning

### 1 Introduction

Recent advances in diffusion models [4,5,8,14,17,18,42,48,59,61,63] have enabled unprecedented quality in video synthesis, yet deploying these systems for realtime interactive applications remains challenging. Conventional video diffusion models rely on extensive multi-step denoising processes, resulting in prohibitive generation latencies. Furthermore, the bidirectional attention mechanism employed by most architectures processes all frames jointly, precluding streaming generation, wherein frames must be produced sequentially. These constraints have motivated a paradigm shift toward efficient, autoregressive alternatives.

To overcome these constraints, several distilled autoregressive models [9,20, 22,23,37,51] have emerged. These methods distill pretrained bidirectional video diffusion models into efficient autoregressive models via distribution matching distillation (DMD) [54]. The resulting models leverage KV-caching for streaming inference, enabling real-time generation with the potential to support long video generation. However, while distillation ensures the student mimics the teacher’s distribution, it lacks optimization for human preference. Consequently, the generated outputs frequently exhibit artifacts and unnatural motion dynamics, remaining misaligned with human preferences.

Concurrently, online RL has demonstrated high efficacy in aligning LLMs with human preferences [12, 33]. This success motivates a natural question: can online RL be applied to align distilled streaming video models with human visual expectations without reverting to computationally expensive pre-training or re-distillation pipelines? Aligning these models via existing methods introduces non-trivial challenges. Previous attempts at reward-guided distillation [30] merely bias the supervised distillation loss by prioritizing samples with higher rewards. While this shifts the output distribution toward high-reward regions,

it lacks a mechanism for active exploration and fails to penalize suboptimal generation samples. On the other hand, applying online RL via reverse-process optimization [28,49] requires log-probability estimation along the sampling trajectory. This couples the algorithm to specific solvers and necessitates storing intermediate trajectory states, adding substantial memory and computational overhead that erodes the efficiency advantages of streaming models.

We present Astrolabe, an efficient and stable online RL framework for distilled AR video models, as shown in Figure 2. Firstly, to bypass the limitations of reward-weighted distillation and the overhead of reverse-process RL, we introduce a trajectory-free alignment strategy tailored for distilled AR video generation. Drawing on the principles of negative-aware fine-tuning [62], our approach contrasts positive and negative generations to establish an implicit policy improvement direction. Requiring only clean inference endpoints, our method sidesteps solver-specific unrolling and full trajectory storage, better preserving the efficiency inherent to streaming architectures. Then, while this resolves per-clip alignment efficiently, scaling to long videos remains challenging: naively unrolling and backpropagating through extended sequences is prohibitively expensive. To address this, we introduce a streaming training scheme that generates videos progressively while applying RL updates only to short segments, conditioning on prior context to retain long-range coherence. Furthermore, to prevent models from reward hacking at the expense of overall aesthetics, the framework employs a multi-reward formulation covering visual quality, motion dynamics, and text alignment. This optimization process is further stabilized by an uncertainty-aware selective regularization strategy that restricts KL penalties to samples lacking auxiliary consensus, alongside a dynamic reference update mechanism that accommodates shifting distributions during online learning.

Extensive experiments on various distilled AR models validate the effectiveness of our method. Figure 1 showcases a diverse set of representative results, demonstrating that the proposed framework consistently enhances generation quality across different settings. Comprehensive evaluations demonstrate improvements across multiple benchmarks. In summary, the primary contributions of our work are as follows: (1) Astrolabe, an online reinforcement learning framework formulated to align distilled streaming video models with human visual preferences; (2) a streaming training scheme that enables scalable alignment of long videos via segment-wise optimization under historical context; and (3) a suite of stabilization techniques, encompassing multi-reward optimization and dynamic regularization, to mitigate reward hacking.

### 2 Related Work

#### 2.1 Video Generative Models

Diffusion models achieve remarkable success in video synthesis [4,5,8,14,17,18, 38,48,57,59,61]. The strong scalability of Diffusion Transformers (DiTs) [2,34] facilitates the emergence of large-scale models [3,27,42,52] that generate highquality content by jointly denoising all frames. However, this full-sequence gen-

eration requires simultaneous processing of all frames, which incurs substantial latency and precludes real-time interaction. Consequently, autoregressive approaches [10,19,26,45–47,50,58,60] emerge to enable streaming generation by producing frames sequentially.

#### 2.2 Autoregressive Video Generation

To circumvent the limitation of bidirectional diffusion models, autoregressive (AR) approaches enable streaming generation by producing frames sequentially. While AR models are inherently suitable for real-time applications, early methods [11,21] relying on Teacher Forcing (TF) suffer from severe error accumulation during long-video synthesis. Recent studies explore novel training paradigms to resolve this train-test misalignment. Diffusion Forcing [6] introduces conditioning at arbitrary noise levels, while CausVid [55] employs block causal attention and distills bidirectional teacher via DMD [54]. More recently, Self-Forcing [22] and its successors [9,13,30,51,53] establish post-training frameworks that systematically mitigate error accumulation. Identifying an architectural gap in the initial ODE distillation phase of these frameworks, Causal Forcing [64] reveals that distilling from a bidirectional teacher violates frame-level injectivity. By employing an AR teacher for initialization instead, it theoretically bridges this gap to achieve superior real-time generation.

#### 2.3 Reinforcement Learning for Generative Models

Recent successes in large language models [12,33] highlight the efficacy of onpolicy reinforcement learning via memory-efficient algorithms like GRPO [36]. For diffusion models, DiffusionDPO [41] utilizes off-policy pairs, while DanceGRPO [49] and Flow-GRPO [28] perform alignment by estimating reversetrajectory log-probabilities. These reverse-process methods inherently couple the training objective to specific solvers and demand full trajectory storage. To bypass this, DiffusionNFT [62] introduces solver-agnostic forward-process policy optimization. Building on this, WorldCompass [44] recently adapted NFT to autoregressive world models [39]. However, their framework directly optimizes heavy pre-distilled teacher models. Extending RL to highly efficient distilled AR video models remains an open problem.

### 3 Methodology

Given a distilled autoregressive video diffusion model optimized for real-time generation, our goal is to further align it with human preferences through online reinforcement learning in the post-training stage. We propose Astrolabe, a memory-efficient framework combining streaming rollout with forward-process RL optimization. Section 3.1 reviews the foundations of AR video diffusion and forward-process RL. Section 3.2 details our memory-efficient streaming rollout

###### Reward Design & Regularization

###### Memory-Efficient Streaming Rollout

###### Online RL Optimization

Candidate Clip

[Figure 71]

Multi-Reward System

Distilled AR Video Model

[Figure 72]

Prompts

Shared Context

Frozen

Motion Quality (MQ)

Visual Quality (VQ)

Text Alignment (TA)

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Velocity Predictor

[Figure 84]

[Figure 85]

Rolling KV Cache

Evicted Frames Candidate

Noise Clip

Clip

Aggregated Reward

Shared Context

Frozen

Implicit Policies

Target Velocity

[Figure 86]

[Figure 87]

Reward Hacking Mitigation

Forward-Process RL loss Auxiliary Reward Ensemble

Primary Reward

[Figure 88]

[Figure 89]

Group-Wise Sampling

Inference Engine

Stream Tuning

Uncertainty Score

[Figure 90]

[Figure 91]

If

Distilled AR Video Model

Historical Context Current Clip Detached Grad. Enabled Grad.

KL Penalty

Candidate Clip

- Fig. 2: Overview of Astrolabe. We propose a memory-efficient RL framework for distilled streaming video models. The method combines group-wise streaming rollout using a rolling KV cache for efficient group-wise sampling (see left), and clip-level forward-process RL for solver-agnostic optimization (see middle). To scale to long videos, we utilize Streaming Long Tuning with detached historical gradients. Furthermore, a multi-reward formulation paired with uncertainty-based selective regularization is employed to effectively mitigate reward hacking during training (see right). The pseudocode of the algorithm can be found in the supplementary materials.

mechanism for scalable exploration. Section 3.3 presents the online RL optimization strategy, encompassing clip-level forward-process RL and streaming long tuning. Finally, Section 3.4 formulates our multi-reward design and selective regularization approach to mitigate reward hacking.

#### 3.1 Preliminaries

Autoregressive Video Diffusion Models. AR video model factorizes the joint distribution as p(x1:N) = Ni=1 p(xi|x<i). Following the flow matching formulation, each conditional p(xi|x<i) is modeled by defining a probability path xti = (1 − t)xi + tϵi, where ϵi ∼ N(0,I) and t ∈ [0,1]. The model predicts the velocity field vθ conditioned on text c and the KV cache of preceding frames. Training paradigms such as Teacher Forcing (TF) and Diffusion Forcing (DF) minimize the frame-wise MSE between the predicted and true targets. In TF, timesteps t are shared across frames with clean ground-truth context x<i, whereas in DF, independent timesteps ti are sampled for each frame using noisy context xtj<ij . Both suffer from exposure bias due to the mismatch between training context and inference-time generation. To mitigate this, Self-Forcing [22] employs autoregressive rollouts {xθ1:N} ∼ Ni=1 pθ(xi|x<i) to simulate inference dynamics. The objective aligns the velocity predictions of the model on these self-generated trajectories with the scores provided by teacher model.

Forward-Process Reinforcement Learning. To avoid the likelihood estimation challenges of reverse-process RL, DiffusionNFT [62] optimizes diffusion models by applying rewards directly to the forward process. Given a clean generated sample x with a normalized reward r˜ ∈ [0,1], a noisy version xt is constructed for timestep t ∈ [0,1]. Using the current (vθ) and old (vθ

) velocity predictors, implicit positive and negative policies are defined via interpolation:

old

+ βvθ, v− = (1 + β)vθ

v+ = (1 − β)vθ

old − βvθ (1) where β controls the interpolation strength. The policy loss contrasts these implicit policies against the target forward velocity vtarget:

old

Lpolicy = r˜∥v+ − vtarget∥22 + (1 − r˜)∥v− − vtarget∥22 (2)

This trajectory-free formulation requires only clean generated samples, enabling highly efficient, solver-agnostic training.

#### 3.2 Memory-Efficient Streaming Rollout

Standard RL paradigms rely on sequence-level rollouts with global rewards. For autoregressive (AR) video generation, this introduces two critical bottlenecks: the temporal credit assignment problem, where sparse global scores fail to isolate localized visual degradation, and the prohibitive memory overhead of maintaining independent KV caches for long sequences. To overcome these limitations, we propose a group-wise streaming rollout strategy.

Rolling KV Cache with Frame Sinks. We maintain a rolling KV cache to bound memory usage. Let the sequence of generated clips be denoted as x1,x2,...,xN. At generation step n, naïvely caching the full history x<n incurs a KV memory cost that grows linearly with video length, quickly becoming prohibitive for long-horizon rollouts. To resolve this, we construct a restricted visual context window Cn comprising two components: a frame sink of S permanently retained frames that anchors global semantic context to prevent long-range drift, and a rolling window of the L most recent frames that provides fine-grained local conditioning. The model attends exclusively to the KV cache of Cn to generate the next clip xn ∼ πθ(·|Cn,c). Since S and L are fixed hyperparameters independent of total video length N, the resident KV memory remains constant regardless of how long the video grows, enabling real-time streaming rollout.

Clip-level Group-wise Sampling. Rather than generating G independent long trajectories from scratch, we autoregressively sample the visual history exactly once and freeze its KV cache as a shared prefix. At the n-th step, utilizing the memory-efficient KV states of Cn, the model decodes G independent candidate clips in parallel:

x(ni) ∼ πθ(·|Cn,c), for i ∈ {1,...,G} (3)

This clip-level rollout restricts the generation overhead to the local chunk rather than the full sequence. By sharing the frozen context prefix across all G candidates, the additional cost of group-wise sampling is incurred only once per

step rather than once per trajectory, substantially reducing rollout time and eliminating out-of-memory bottlenecks during reinforcement learning.

#### 3.3 Online RL Optimization

Clip-level Forward-Process RL. For each candidate x(ni), we evaluate a composite reward R(x(ni),c) and compute its advantage A(i) via group-wise meancentering:

G

1 G

A(i) = R(x(ni),c) −

R(x(nj),c) (4)

j=1

This advantage is then normalized as r˜i = clip(A(i)/Amax)/2+0.5. For our T = 4 distilled model, the timestep t is sampled from Tdistill. Crucially, we discard the adaptive loss weighting of DiffusionNFT [62], as it triggers gradient explosion under large discretization gaps in distilled AR settings. Conditioned on text c and the shared KV cache Cn, we construct the noised sample xt,n(i) to predict velocities vθ and vθ

. The model is optimized directly via the implicit policy loss

old

Lpolicy (Eq. 2) by substituting xt,n(i) to derive vtarget. To further mitigate reward hacking, this objective is complemented by an uncertainty-aware selective KL penalty (Section 3.4).

Streaming Long Tuning. Distilled AR models suffer from a train-short/testlong mismatch, where accumulated prediction errors cause inevitable long-horizon degradation. To address this, our training paradigm strictly simulates the dynamics of long-sequence inference while decoupling the forward rollout from gradient computation. Specifically, we first perform a full forward pass to accumulate the KV cache up to the target step. Upon reaching the active training window xn, the KV cache of all preceding frames x<n is explicitly detached from the computation graph. This detached cache serves as historical context, mimicking the progressively noisy conditions encountered during autoregressive generation. Gradients are then backpropagated through the active window. This exact formulation inherently bounds the training memory usage, circumventing the cost of backpropagation through extended trajectories.

#### 3.4 Reward Design and Regularization

Multi-reward Formulation. Scalar reward functions obscure specific quality dimensions and often inadvertently encourage the model to exploit one attribute over others. To address this, we formulate a composite reward integrating three distinct axes: Visual Quality (VQ), Motion Quality (MQ), and Text-Video Alignment (TA). We compute the Visual Quality (VQ) reward as the mean HPSv3 [31] score over the top 30% of frames. Excluding lower-scoring frames prevents transient motion blur from disproportionately penalizing the overall aesthetic assessment. For the Motion Quality (MQ) reward, we evaluate temporal consistency using a pre-trained VideoAlign [29] strictly on grayscale inputs; removing color forces the metric to focus on motion dynamics rather than texture. Finally, the

Text Alignment (TA) reward employs the standard RGB VideoAlign to measure the semantic correspondence between the text and the generated video

Uncertainty-Aware Penalty. To prevent uniform KL regularization from indiscriminately suppressing high-quality generations, we introduce a selective KL penalty targeting reward hacking via reward rank disagreement [16]. For each candidate x(ni), we quantify sample uncertainty as the rank discrepancy between the primary reward model p and M − 1 auxiliary models: ∆(ranki) = rank(pi) −

M−1 m̸=p rank(mi). High positive values indicate likely reward hacking lacking ensemble consensus. We mask these risky samples using M(i) = [∆ (ranki) > τ], where τ is the (1 − ρ)-th percentile of positive discrepancies (with risk ratio ρ). The total objective L = Lpolicy + λKLLKL applies the KL penalty strictly to masked samples, preserving optimization flexibility for clean data. Furthermore, to mitigate distributional shifts during online RL, the policy θold follows an EMA update, and the reference policy conditionally resets (θref ← θ) when policy deviation surpasses τKL or epochs reach Kmax.

1

### 4 Experiments

#### 4.1 Experimental Setup

Implementation Details. To validate the effectiveness of our method, we evaluate Astrolabe on distilled autoregressive models. We adopt base models trained via Self-Forcing [22], Causal-Forcing [64], and LongLive [51] as our primary baselines. Training prompts are sampled from the VidProM dataset [43], specifically utilizing the filtered subset introduced in DanceGRPO [49]. We employ LowRank Adaptation (LoRA) with rank r = 256 and scaling factor α = 256 for parameter-efficient fine-tuning. To maximize memory efficiency during optimization, we do not store separate full-parameter copies for the current policy vθ and the old policy vθ

. Instead, both policies share a single frozen base model, and we switch between their respective lightweight LoRA during the forward pass, reducing GPU memory overhead. Training operations are distributed across 48 NVIDIA H200 GPUs. Each epoch processes 48 prompts, maintaining a group size of G = 24 candidate clips per prompt. For reward computation, we integrate VideoAlign [29] and HPSv3 [31] into our pipeline. More details can be found in supplementary material.

old

#### 4.2 Short-Video Single-Prompt Generation

We first validate our method under the short-video, single-prompt setting. Following VBench protocols [24], we evaluate models using 946 standard prompts. To ensure a fair comparison with Self-Forcing, we utilize the augmented prompt test set during sampling, where prompts are expanded via Qwen2.5-7B-Instruct [1] using Wan2.1 [42] system prompts. We integrate Astrolabe with various distilled AR models, comparing them against native AR models and bidirectional diffusion baselines. Quantitative results in Table 1 show that Astrolabe consistently

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

+Ours

+Ours

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Self-forcing

Self-forcing

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Reward-Forcing

Reward-Forcing

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

CausVid

CausVid

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

+Ours

+Ours

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

Self-forcing

Self-forcing

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Reward-Forcing

Reward-Forcing

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

CausVid

CausVid

- Fig. 3: Qualitative comparison under the short-video, single-prompt setting. We evaluate our framework (+Ours) against other baselines. Visual results confirm that our method generates videos with significantly sharper textures and superior motion coherence, aligning better with human preferences. More results can be found in supplementary material.

enhances performance across all Self-Forcing variants. Similar gains observed in LongLive [51] and Causal-Forcing [64] further demonstrate the framework’s generalizability across different base architectures. To further assess alignment with human preferences, we curate 100 diverse prompts from MovieGenBench [35] for evaluation. We compute HPSv3 and Motion Quality scores to quantify improvements in aesthetic appeal and temporal consistency. Results indicate that our RL-tuned models outperform their base versions in these metrics while maintaining the exact inference speed of the original checkpoints. Qualitative results in Figure 3 further confirm that Astrolabe yields sharper textures and superior motion coherence without sacrificing system throughput.

#### 4.3 Long-Video Single-Prompt Generation

Under the single-prompt long-video generation setting, we evaluate our method using VBench-Long protocols. For each prompt in the official dataset, we generate a 30-second video and subsequently partition it into localized clips using

- Table 1: Quantitative results on VBench benchmarks. Integrating our approach into existing distilled models yields consistent improvements in motion quality, semantic alignment, and overall generation quality.

Method Total↑ Quality↑ Semantic↑ HPSv3↑ MQ↑ Throughput↑ Diffusion Models

LTX-Video [15] 80.00 82.30 70.79 8.32 1.34 8.98 Wan2.1 [42] 84.26 85.30 80.09 9.26 1.62 0.78

AR Models

SkyReels-V2 [7] 82.67 84.70 74.53 9.08 1.59 0.49 MAGI-1 [40] 79.18 82.04 67.74 7.95 1.52 0.19 NOVA [10] 80.12 80.39 79.05 8.21 1.63 0.88 PyramidFlow [25] 81.72 84.74 69.62 8.76 1.50 6.70

Distilled AR Models

CausVid [55] 81.20 84.05 69.80 7.56 1.22 17.0 Reward Forcing [30] 84.13 84.84 81.32 8.74 1.65 23.1 Self-Forcing [22] 83.74 84.48 80.77 9.36 1.65 17.0

- + Ours 83.79+.05 84.51+.03 80.92+.15 10.72+1.36 1.71+.06 17.0

LongLive [51] 83.22 83.68 81.37 9.38 1.51 20.7

- + Ours 84.93+1.71 85.83+2.15 81.36-.01 11.03+1.65 1.64+.13 20.7

Causal Forcing [64] 84.04 84.59 81.84 9.48 1.69 17.0 + Ours 84.46+.42 85.15+.56 81.72-.12 10.84+1.36 1.80+.11 17.0

- Table 2: Quantitative results on VBench-Long benchmarks. Integrating our method consistently improves the performance of long video generation baselines across both video quality and human preference metrics.

###### Method Total↑ Quality↑ Semantic↑ HPSv3↑ MQ↑

SkyReels-V2 [7] 75.29 80.77 53.37 8.72 1.54 FramePack [56] 81.95 83.61 75.32 8.94 1.58

Self-Forcing [22] 81.59 83.82 72.70 9.12 1.61

+ Ours 82.03 84.36 72.71 10.38 1.72 LongLive [51] 83.52 85.44 75.82 9.21 1.48

+ Ours 84.07 86.12 75.87 10.67 1.64 Causal Forcing [64] 82.87 84.36 76.91 9.28 1.65

###### + Ours 84.24 86.18 76.48 10.52 1.74

the standard VBench-Long evaluation scripts. Notably, while LongLive natively supports long-video generation, Self-Forcing and Causal-Forcing are exclusively trained on 5-second sequences. To enable long-horizon generation for these shortcontext models, we integrate the Infinity-RoPE [53] to extrapolate their positional embeddings. Furthermore, we rigorously benchmark these configurations against open-source solutions, including SkyReels-V2 [7] and FramePack [56]. Quantitative results in Table 2 report standard VBench-Long metrics measuring long-horizon quality and temporal consistency. Results indicate that our RL framework can also improve performance across long-video benchmarks, demonstrating that alignment optimizations conducted on short videos can effectively extrapolate to extended temporal horizons. Qualitative results in Figure 4 further

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

LongLive

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

+Ours

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

LongLive

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

+Ours

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

LongLive

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

+Ours

- Fig. 4: Qualitative results under the single-prompt long-video setting. Our framework (+Ours) effectively translates alignment optimizations from short videos to extended temporal horizons. Our approach delivers enhanced visual details and more stable throughout the sequence.

confirm that Astrolabe yields sharper textures and superior motion coherence over extended durations.

#### 4.4 Long-Video Multi-Prompt Generation

To demonstrate that our framework effectively improves human preference alignment, we evaluate Astrolabe in the setting of interactive multi-prompt long-video generation. We apply our method directly to the baselines, demonstrating how Astrolabe further enhances their capabilities. Following established protocols from LongLive [51], we curate 100 groups of narrative scripts. Each group comprises six successive 10-second prompts, yielding 60-second long-form videos. To ensure fair comparisons, short-context baselines (Self-Forcing, Causal-Forcing) are adapted for multi-prompt generation via prompt switching during the autoregressive rollout. LongLive, conversely, natively supports generative extrapolation with interactive instructions. We segment the generated videos at prompt boundaries to evaluate text alignment. CLIP scores are subsequently computed at 10-second intervals to measure clip-wise semantic adherence. Quantitative results in Table 3 show that Astrolabe improves overall generation quality, with noticeable gains in visual aesthetics and long-range motion consistency. Qualitative examples in Figure 6 further illustrate these enhancements during extended

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

- Fig. 5: Performance improvements across different models. We evaluate our method on three models. The dashed grey lines indicate the baseline performance of the respective base models. The results demonstrate that our approach consistently improves both HPSv3 and MQ scores across all three models.

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

[Figure 214]

[Figure 215]

[Figure 216]

+Ours

[Figure 217]

LongLive

[Figure 218]

LongLive

[Figure 219]

+Ours

- Fig. 6: Qualitative comparison of multi-prompt long-video generation. We compare the LongLive [51] with our method. The generated sequences exhibit noticeable improvements in visual aesthetics and fine details during complex narrative transitions.

video generation. These results suggest that our framework enhances both framelevel aesthetics and temporal consistency in complex multi-prompt setting.

#### 4.5 Ablation Studies

We conduct ablation studies to validate each component of our method. All ablations are performed on Causal-Forcing with short-video alignment unless otherwise specified. Additional discussion and details can be found in the Supplementary Material.

Streaming Training Scheme. Table 4 compares different rollout and optimization strategies for 30-second video generation. Sequence-level rollout with full backpropagation causes out-of-memory errors. Our clip-level group-wise sampling with detached context achieves the best trade-off: it reduces memory con-

- Table 3: Quantitative evaluation on long video generation. We compare these overall metrics alongside CLIP Scores evaluated across 10-second intervals (0-60s).

CLIP Score↑ 0-10 10-20 20-30 30-40 40-50 50-60

Method Quality Score↑ Consistency Score↑ Aesthetic Score↑

SkyReels-V2 [7] 81.55 94.72 56.83 25.31 23.40 22.50 21.62 21.67 20.91 FramePack [56] 84.40 96.77 59.44 26.51 22.60 22.18 21.53 21.98 21.62

Self-Forcing [22] 83.94 95.74 58.45 26.24 24.87 23.46 21.92 22.05 21.07

- + Ours 84.72 95.98 59.62 26.42 24.75 23.95 22.40 21.85 21.50 LongLive [51] 84.28 96.05 59.89 26.63 25.77 24.65 23.99 24.52 24.11

- + Ours 85.15 96.16 60.75 26.80 26.15 24.45 24.55 24.30 24.65

Causal-Forcing [64] 84.12 95.88 59.15 26.45 25.60 23.98 22.85 22.48 22.45 + Ours 84.95 95.63 60.32 26.58 25.12 23.85 23.40 23.10 22.95

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

TA

[Figure 224]

[Figure 225]

[Figure 226]

- beta=0.1

[Figure 227]

- beta=1

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

TA + VQ

beta=0.1

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

TA + MQ + VQ

beta=1

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

W/o KL reg.

(a) Multi-reward design

(b) Different β values

- Fig. 7: Ablation studies on reward formulation and interpolation strength. (a) Optimizing a single objective induces reward hacking and degrades other quality dimensions, whereas our aggregated formulation balances visual aesthetics and motion consistency. (b) The parameter β controls the implicit contrast between positive and negative samples, with β = 1.0 yielding the optimal trade-off for quality.

sumption by ≈ 2× compared to clip-level full backpropagation while improving both HPSv3 and MQ. The efficiency gains stem from sharing historical context across candidate clips reducing redundant computation.

Reward Design and Regularization. Table 4 ablates our objective formulation. Single-reward optimization induces hacking: VQ-only training collapses into static frames, improving HPSv3 but degrading MQ. Our multi-reward formulation (VQ+MQ+TA) prevents this single-objective overfitting, yielding balanced improvements. Furthermore, uniform KL regularization over-constrains learning, while its omission causes instability and early MQ plateaus (Figure 8a). Our selective KL penalty with EMA reference updates resolves this by adaptively penalizing only high-uncertainty predictions. This targeted approach preserves optimization freedom for confident samples, effectively balancing exploration with stable convergence across all metrics.

Removing Adaptive Weighting. DiffusionNFT [62] scales the loss using a self-normalized x0 denominator. However, Figure 8b demonstrates this adaptive weighting destabilizes distilled AR setting. Under large discretization gaps, this dynamic denominator becomes volatile, causing the predicted x0 norm to explode

- Table 4: Ablation studies on each component. (a) and (b) show the impact of the streaming strategy and selective KL regularization. (c) compares different reward combinations.

(a) Streaming Training Config HPSv3↑ MQ↑ Mem↓ Seq + Full BP OOM OOM >140 Seq + Detach 10.21 1.72 96.4 Clip + Full BP 10.58 1.76 112.3 Clip+Detach 10.84 1.80 54.3

(b) Selective KL Reg. Strategy HPSv3↑ MQ↑ TA↑

No KL 10.67 1.74 -0.068 Uni. (λ = 1e−4) 10.52 1.71 0.012 Uni. (λ = 5e−4) 10.28 1.68 0.028 Sel.+EMA 10.84 1.80 0.065

(c) Multi-Reward Formulation Reward HPSv3↑ MQ↑ TA↑ VB↑ Reward HPSv3↑ MQ↑ TA↑ VB↑ Baseline 9.48 1.69 -0.015 84.04 VQ + MQ 10.67 1.74 -0.068 83.95 VQ only 10.92 1.58 -0.075 83.21 VQ + TA 10.71 1.62 0.055 84.12 MQ only 9.31 1.82 -0.058 83.67 MQ + TA 9.45 1.78 0.051 84.08

TA only 9.42 1.62 0.082 84.25 All (Ours) 10.84 1.80 0.065 84.46

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

=1e 4 =5e 4 =5e 4

(a) Ablation on KL penalty.

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
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
| | | | | |
| | | | | |

(b) Ablation on adaptive weight.

Fig. 8: Ablation study of stabilization techniques in Astrolabe. (a) Effect of the selective KL penalty. (b) Impact of the dynamic adaptive weight mechanism.

after 50 steps and triggering a sharp collapse in reward. Conversely, removing this scaling factor bounds the x0 norm and ensures steady, monotonic reward improvements.

Impact of β. Figure 7 ablates the parameter β, which determines the scale of the implicit guidance direction integrated into the old policy. Empirical results indicate that varying β directly influences the temporal dynamics of the generated sequences. In our experimental setup, setting β = 1 yields higher overall visual and motion quality compared to a smaller value such as β = 0.1. Consequently, we adopt β = 1 as the default configuration to maintain generation stability.

- 5 Conclusion

We present Astrolabe, an online RL framework for aligning distilled autoregressive video models with human preferences. Utilizing a memory-efficient, forwardprocess RL formulation, our method eliminates the trajectory storage overhead of reverse-process alternatives. For long-video scalability, we introduce a streaming training scheme with local-window optimization, achieving constant peak memory. To prevent reward hacking, we implement a multi-reward formulation

coupled with an uncertainty-aware selective KL penalty. Extensive experiments across multiple distilled streaming architectures and benchmarks validate the effectiveness and generality of our approach.

### References

- 1. Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al.: Qwen technical report. arXiv preprint arXiv:2309.16609 (2023) 8
- 2. Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., Zhu, J.: All are worth words: A vit backbone for diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22669–22679 (2023) 3
- 3. Bao, F., Xiang, C., Yue, G., He, G., Zhu, H., Zheng, K., Zhao, M., Liu, S., Wang, Y., Zhu, J.: Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233 (2024) 3
- 4. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023) 2, 3

- 5. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22563–22575 (2023) 2, 3
- 6. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems 37, 24081–24125 (2024) 4
- 7. Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al.: Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074 (2025) 10, 13
- 8. Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., et al.: Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023) 2, 3
- 9. Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., Hsieh, C.J.: Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283 (2025) 2, 4
- 10. Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., Wang, X.: Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169 (2024) 4, 10
- 11. Gao, K., Shi, J., Zhang, H., Wang, C., Xiao, J., Chen, L.: Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375 (2024) 4
- 12. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025) 2, 4
- 13. Guo, Y., Yang, C., He, H., Zhao, Y., Wei, M., Yang, Z., Huang, W., Lin, D.: End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702 (2025) 4
- 14. Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Li, F.F., Essa, I., Jiang, L., Lezama, J.: Photorealistic video generation with diffusion models. In: European Conference on Computer Vision. pp. 393–411. Springer (2024) 2, 3

- 15. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al.: Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103 (2024) 10
- 16. He, H., Ye, Y., Liu, J., Liang, J., Wang, Z., Yuan, Z., Wang, X., Mao, H., Wan, P., Pan, L.: Gardo: Reinforcing diffusion models without reward hacking. arXiv preprint arXiv:2512.24138 (2025) 8
- 17. He, Y., Yang, T., Zhang, Y., Shan, Y., Chen, Q.: Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221 (2022) 2, 3
- 18. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. Advances in neural information processing systems 35, 8633– 8646 (2022) 2, 3
- 19. Hong, W., Ding, M., Zheng, W., Liu, X., Tang, J.: Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868

(2022) 4

- 20. Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., Hold-Geoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al.: Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040 (2025) 2
- 21. Hu, J., Hu, S., Song, Y., Huang, Y., Wang, M., Zhou, H., Liu, Z., Ma, W.Y., Sun, M.: Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720 (2024) 4
- 22. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009

(2025) 2, 4, 5, 8, 10, 13

- 23. Huang, Y., Guo, H., Wu, F., Zhang, S., Huang, S., Gan, Q., Liu, L., Zhao, S., Chen, E., Liu, J., et al.: Live avatar: Streaming real-time audio-driven avatar generation with infinite length. arXiv preprint arXiv:2512.04677 (2025) 2
- 24. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807–21818 (2024) 8
- 25. Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954 (2024) 10
- 26. Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu, M.C., et al.: Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 (2023) 4
- 27. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024) 3
- 28. Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., Ouyang, W.: Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470 (2025) 3, 4
- 29. Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., Zheng, M., Wu, X., Wang, Q., Xia, M., Wang, X., et al.: Improving video generation with human feedback. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025) 7, 8
- 30. Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K.L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., et al.: Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678

(2025) 2, 4, 10

- 31. Ma, Y., Wu, X., Sun, K., Li, H.: Hpsv3: Towards wide-spectrum human preference score. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15086–15095 (2025) 7, 8
- 32. Millon, E.: Krea realtime 14b: Real-time video generation (2025), https://github. com/krea-ai/realtime-video 4
- 33. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, 27730–27744 (2022) 2, 4
- 34. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023) 3
- 35. Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.Y., Chuang, C.Y., et al.: Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024) 9
- 36. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024) 4
- 37. Shin, J., Li, Z., Zhang, R., Zhu, J.Y., Park, J., Shechtman, E., Huang, X.: Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266 (2025) 2
- 38. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without textvideo data. arXiv preprint arXiv:2209.14792 (2022) 3
- 39. Sun, W., Zhang, H., Wang, H., Wu, J., Wang, Z., Wang, Z., Wang, Y., Zhang, J., Wang, T., Guo, C.: Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614 (2025) 4
- 40. Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W., Luo, W., et al.: Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211 (2025) 10
- 41. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., Naik, N.: Diffusion model alignment using direct preference optimization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8228–8238 (2024) 4
- 42. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025) 2, 3, 8, 10
- 43. Wang, W., Yang, Y.: Vidprom: A million-scale real prompt-gallery dataset for textto-video diffusion models. Advances in Neural Information Processing Systems 37, 65618–65642 (2024) 8
- 44. Wang, Z., Wang, T., Zhang, H., Zuo, X., Wu, J., Wang, H., Sun, W., Wang, Z., Cao, C., Zhao, H., et al.: Worldcompass: Reinforcement learning for long-horizon world models. arXiv preprint arXiv:2602.09022 (2026) 4
- 45. Weissenborn, D., Täckström, O., Uszkoreit, J.: Scaling autoregressive video models. arXiv preprint arXiv:1906.02634 (2019) 4
- 46. Wu, C., Huang, L., Zhang, Q., Li, B., Ji, L., Yang, F., Sapiro, G., Duan, N.: Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806 (2021) 4
- 47. Wu, C., Liang, J., Ji, L., Yang, F., Fang, Y., Jiang, D., Duan, N.: Nüwa: Visual synthesis pre-training for neural visual world creation. In: European conference on computer vision. pp. 720–736. Springer (2022) 4

- 48. Xing, J., Xia, M., Zhang, Y., Chen, H., Yu, W., Liu, H., Liu, G., Wang, X., Shan, Y., Wong, T.T.: Dynamicrafter: Animating open-domain images with video diffusion priors. In: European Conference on Computer Vision. pp. 399–417. Springer (2024) 2, 3
- 49. Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al.: Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818 (2025) 3, 4, 8
- 50. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021) 4
- 51. Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al.: Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622 (2025) 1, 2, 4, 8, 9, 10, 11, 12, 13
- 52. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024) 3
- 53. Yesiltepe, H., Meral, T.H.S., Akan, A.K., Oktay, K., Yanardag, P.: Infinity-rope: Action-controllable infinite video generation emerges from autoregressive selfrollout. arXiv preprint arXiv:2511.20649 (2025) 1, 4, 10
- 54. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6613– 6623 (2024) 2, 4
- 55. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22963– 22974 (2025) 4, 10
- 56. Zhang, L., Cai, S., Li, M., Wetzstein, G., Agrawala, M.: Frame context packing and drift prevention in next-frame-prediction video diffusion models. arXiv preprint arXiv:2504.12626 (2025) 10, 13
- 57. Zhao, M., Bao, F., Li, C., Zhu, J.: Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems 35, 3609–3623 (2022) 3
- 58. Zhao, M., He, G., Chen, Y., Zhu, H., Li, C., Zhu, J.: Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894

(2025) 4

- 59. Zhao, M., Wang, R., Bao, F., Li, C., Zhu, J.: Controlvideo: conditional control for one-shot text-driven video editing and beyond. Science China Information Sciences 68(3), 132107 (2025) 2, 3
- 60. Zhao, M., Zhu, H., Wang, Y., Yan, B., Zhang, J., He, G., Yang, L., Li, C., Zhu, J.: Ultravico: Breaking extrapolation limits in video diffusion transformers. arXiv preprint arXiv:2511.20123 (2025) 4
- 61. Zhao, M., Zhu, H., Xiang, C., Zheng, K., Li, C., Zhu, J.: Identifying and solving conditional image leakage in image-to-video diffusion model. Advances in Neural Information Processing Systems 37, 30300–30326 (2024) 2, 3
- 62. Zheng, K., Chen, H., Ye, H., Wang, H., Zhang, Q., Jiang, K., Su, H., Ermon, S., Zhu, J., Liu, M.Y.: Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117 (2025) 3, 4, 6, 7, 13, 1
- 63. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404 (2024) 2

- 64. Zhu, H., Zhao, M., He, G., Su, H., Li, C., Zhu, J.: Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214 (2026) 1, 4, 8, 9, 10, 13

- A Hyperparameters We provide detailed hyperparameter in Table S1.

Table S1: Comprehensive Hyperparameters for Astrolabe Training. We detail the configurations used across the model architecture, optimization, diffusion process, reinforcement learning, and streaming rollout.

Module Hyperparameter Value Model & Video Specs

Base Architecture Causal Wan 2.1 Video Resolution (H × W) 480 × 832

LoRA Fine-Tuning

Rank (r) 256 Scaling Factor (α) 256 Dropout Rate 0.0 Gradient Checkpointing True

Optimization

Hardware 48 × NVIDIA GPUs Precision Mode bf16 (Mixed Precision) Optimizer AdamW (β1 = 0.9, β2 = 0.999) Learning Rate (η) 1e-5 Weight Decay 1e-4 Epsilon (ϵ) 1e-8 Max Gradient Norm 1.0

Diffusion Process

Distillation Timesteps (T) 4 (sampled from [1000, 750, 500, 250]) Timestep Shift 5.0 Forward Process Noise Level 0.7

Selective Regularization

Interpolation Strength (β) 1.0 KL Penalty Weight (λKL) 1e-4 Advantage Clip Max 5.0 Reward Normalization Global Std with Per-Prompt Tracking EMA Decay Rate (γ) 0.9 EMA Update Interval 1 step

Streaming Rollout

Window Selection Strategy Random Choice Rolling Window Size (L) 21 (Self-Forcing) / 15 (Longlive) Frame Sink Size (S) 3

- B Theoretical Proofs and Analysis

We extend the theoretical foundation of DiffusionNFT [62] from global, nonautoregressive continuous-time diffusion to the settings of autoregressive (AR) streaming generation and few-step distilled models. Specifically, we prove the optimality of local advantage guidance (Theorem 1), and establish a reward lower bound via selective KL penalty (Theorem 2).

#### B.1 Conditional Improvement via Advantage Guidance

- Theorem 1 (Conditional Improvement via Advantage Guidance). Let Cn be the frozen context at step n, r˜(xn,Cn) ∈ [0,1] the normalized advantage,

and α ≜ α(xtn,Cn) = Eπ

[˜r | xtn,Cn] the posterior positive probability. Define the implicit positive/negative distributions:

old

r˜· πold E[˜r]

(1 − r˜) · πold E[1 − r˜]

, π−(xn | Cn) =

π+(xn | Cn) =

and the local policy loss (β > 0 controls negative repulsion strength):

(5)

L(policyn) (θ) = Et, xt

n

∥vθ − v+∥2 + β∥vθ − v−∥2 (6)

where v± are conditional flow matching targets under π±. Then the optimal velocity field satisfies:

1 − α(1 + β) (1 + β)(1 − α)

vθ∗ = vold +

v+ − vold (7)

which strictly shifts the local transition toward the advantage-weighted policy in the typical regime α(1 + β) < 1.

Proof. Since the joint distribution factorizes as p(x1:N) = n p(xn | Cn), the velocity field at step n is strictly conditioned on Cn. By Bayes’ theorem, the posterior decomposes as:

πold(x0n | xtn,Cn) = α π+(x0n | xtn,Cn) + (1 − α)π−(x0n | xtn,Cn) (8) By the law of total expectation, the baseline velocity inherits this mixture:

vold = α v+ + (1 − α)v− (9)

Let δ+ = v+ − vold. From Eq. (9), v− − vold = −1−ααδ+. Substituting into the loss and writing u = vθ − vold:

L = E ∥u − δ+∥2 + β u + 1−ααδ+ 2 (10) Setting ∂L/∂u = 0:

1 − α(1 + β) 1 − α

βα 1 − α

(1 + β)u∗ = δ+ −

δ+ =

δ+ (11)

which yields the stated result. When α > 0 and α(1 + β) < 1, the shift aligns with v+ − vold, pushing the model toward the high-advantage region. As β → 0, this degrades to standard reward-weighted regression. Thus, optimizing the local advantage strictly improves πθ(xn | Cn) without requiring gradients across the full trajectory.

- B.2 Performance Lower Bound with Selective Trust Region

- Theorem 2 (Performance Lower Bound with Selective Trust Region). Let Rˆ(x) be the proxy reward, R∗(x) the true preference with |R∗| ≤ Rmax, and

U = {x | ∆rank(x) > τ} the high-uncertainty region. Assume:

- – (A3) For x ∈/ U: |Rˆ(x) − R∗(x)| ≤ ϵsafe.
- – (A4) Under πref: Eπ

[|Rˆ − R∗| | x ∈ U] ≤ ϵrisk, with ϵrisk ≫ ϵsafe. Define the selective velocity MSE in U:

ref

1

∥vθ(xt,t) − vref(xt,t)∥2 σ2(t)

dt (12)

LKL(θ) = Ex∼π

θ, x∈U

0

Then the true reward of πθ is lower-bounded by: Eπ

[Rˆ] − ϵsafe − πθ(U) ϵrisk + 2Rmax LKL (13) Proof. Step 1: Regional decomposition.

[R∗] ≥ Eπ

θ

θ

[R∗] = Eπ

[R∗ | x ∈/ U] · πθ(UC) + Eπ

[R∗ | x ∈ U] · πθ(U) (14)

Eπ

θ

θ

θ

[Rˆ | x ∈/ U] − ϵsafe.

#### Step 2: Safe region. By (A3), Eπ

[R∗ | x ∈/ U] ≥ Eπ

θ

θ

##### Step 3: Risk region. Since (A4) holds for πref, transferring to πθ requires

bounding the distribution shift. By the TV dual representation with |f| ≤ 2Rmax: Eπ

[Rˆ | U] − ϵrisk − 4Rmax · DTV(πθ∥πref | U) (15)

[R∗ | U] ≥ Eπ

θ

θ

##### Step 4: KL–TV connection. By Girsanov’s theorem, the KL divergence

restricted to U equals 12LKL. Pinsker’s inequality then gives:

DTV ≤ 12DKL = 21 LKL (16)

##### Step 5: Assembly. Combining Steps 1–4 with πθ(UC) ≤ 1 yields the stated

bound.

Theoretical Insight. Unlike global KL penalties that indiscriminately constrain all exploration, Astrolabe’s selective penalty acts only through the πθ(U)· 2Rmax√

LKL term. In safe regions (x ∈/ U), the policy explores freely; the trust region activates only when πθ drifts into the high-uncertainty region U, naturally balancing exploration and stability.

### C Algorithm

We provide the detailed pseudocode for the training procedures of the Astrolabe framework. Algorithm 1 outlines the core forward-process reinforcement learning pipeline, termed NFT Training with Selective KL Regularization. It details the multi-reward evaluation, uncertainty quantification via rank disagreement, and the selective application of the KL penalty to mitigate reward hacking

during short-clip optimization. To scale this alignment to long-sequence generation without exceeding memory constraints, Algorithm 2 presents the Streaming Training for Long Video. This algorithm leverages a rolling KV cache and detaches historical context, applying gradient updates exclusively to local active windows while maintaining long-range temporal coherence.

### D More Qualitative Results

We present additional qualitative comparisons on 30-second long video generation under the single-prompt setting in Figures S2–S29. Qualitative comparisons showing improvements over baseline distilled models and competing methods. As shown in Figure S1, our method consistently produces higher-quality and more temporally coherent videos compared to the Krea 14B baseline [32], demonstrating that our approach yields notable improvements even when applied on top of large-scale 14B models.

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

Krea 14B

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

+Ours

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Krea 14B

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

+Ours

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

Krea 14B

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

+Ours

###### Fig. S1: Qualitative comparison between Krea 14B (odd rows) and with our method (+Ours, even rows)

### E Discussion

While Astrolabe establishes a robust and memory-efficient paradigm for aligning distilled autoregressive video models, several limitations warrant further investigation.

Reliance on the Capability of Reward Models. Astrolabe fundamentally depends on the accuracy of the selected reward models. Currently, open-source video evaluation models (e.g., VideoAlign) exhibit strong proficiency in assessing short-term aesthetics and text-video alignment. However, their ability to evaluate complex physics, long-horizon causality, and multi-entity interactions in minute-scale videos remains limited. Consequently, if the reward model fails to penalize subtle temporal hallucinations in extended sequences, Astrolabe cannot explicitly correct them. Developing more robust, physics-aware reward models for long-form video represents a critical next step for the community.

Inherent Bottlenecks of the Base Architecture. As a post-training RL framework, Astrolabe excels at shifting the generation distribution toward highreward regions and mitigating error accumulation. However, reinforcement learning cannot arbitrarily instantiate capabilities that are entirely absent from the distilled base model. For instance, if the original streaming models (e.g., SelfForcing or LongLive) inherently lack the capacity to render highly complex spatial geometries or specific domain knowledge due to extreme distillation, our method can optimize the presentation of existing knowledge but cannot overcome the fundamental capacity ceiling of the base architecture.

Algorithm 1 NFT Training with Selective KL Regularization

Require: Policy πθ, behavior policy πθold, KL reference πθref; reward functions

{Rm}Mm=1; prompts D; hyperparameters β, λKL, Kmax, τKL Ensure: Optimized policy πθ

- 1: Initialize θold ← θ, θref ← θ
- 2: Initialize risk buffer B ← ∅, KL ratio ρ ← ρ0, klast ← 0
- 3: for each epoch k do
- 4: // Phase 1: Rollout and multi-reward evaluation
- 5: Sample prompts {ci}Bi=1 ∼ D
- 6: Generate samples {x(0i)} using πθold
- 7: Compute rewards {rm(i) = Rm(x(0i), ci)} for each reward model m
- 8: Compute advantages {A(i)} from primary reward
- 9: // Compute reward uncertainty via rank disagreement
- 10: for each reward model m do
- 11: rank(mi) ← Rank({rm(j)}Bj=1) ▷ Rank samples by reward m
- 12: end for
- 13: ∆(ranki) ← rank(primaryi) − M1−1 m̸=primary rank(mi)

- 14: // Adaptive threshold via risk compensation
- 15: ρ ← UpdateRiskRatio(B, {∆(ranki) }, ρ)
- 16: τ ← Percentile({∆(ranki) |∆(ranki) ≥ 0}, 100 − 100ρ)
- 17: M(i) ← [∆ (ranki) > τ] ▷ High-uncertainty mask
- 18: // Phase 2: Forward process optimization
- 19: for each mini-batch (x0, A, M) do
- 20: Sample t ∼ U(T )
- 21: xt ← (1 − t)x0 + tϵ, ϵ ∼ N(0, I)
- 22: vθ, vθold, vθref ← πθ(xt, t), πθold(xt, t), πθref(xt, t)
- 23: // Policy loss
- 24: v+ ← β · vθ + (1 − β) · vθold
- 25: v− ← (1 + β) · vθold − β · vθ
- 26: r˜ ← clip(A/Amax)/2 + 0.5
- 27: Lpolicy ← r˜∥v+ − x0∥2 + (1 − r˜)∥v− − x0∥2
- 28: // Selective KL: only on high-uncertainty samples
- 29: LKL ← |M|1 i:M(i)=1 ∥vθ(i) − vθ(i)

ref

∥2

- 30: L ← Lpolicy + λKL · LKL
- 31: Update: θ ← θ − η∇θL
- 32: end for
- 33: // Adaptive reference update based on KL divergence
- 34: if LKL > τKL or k − klast > Kmax then
- 35: θref ← θ ▷ Reset KL reference
- 36: klast ← k
- 37: end if
- 38: θold ← γ · θold + (1 − γ) · θ
- 39: end for

Algorithm 2 Streaming Training for Long Video

Require: Policy πθ, behavior policy πθold, KL reference πθref; rewards {Rm}; prompts

D; window size W, total frames F Ensure: Optimized policy πθ

- 1: Initialize θold ← θ, θref ← θ, risk buffer B ← ∅, ρ ← ρ0
- 2: for each epoch k do
- 3: // Select training window (synced across GPUs)
- 4: s ← SelectWindow(F, W), Freq ← s + W
- 5: // Phase 1: Efficient rollout with multi-reward
- 6: Sample prompts {ci}Bi=1 ∼ D
- 7: Generate partial videos {x(0i)[0 : Freq]} using πθold
- 8: Extract window: x(0i,W) ← x(0i)[s : s + W]
- 9: Compute rewards {rm(i) = Rm(x(0i,W) , ci)} for each model m
- 10: Compute advantages {A(i)}
- 11: // Reward uncertainty via rank disagreement
- 12: for each reward model m do
- 13: rank(mi) ← Rank({rm(j)}Bj=1)
- 14: end for
- 15: ∆(ranki) ← rank(primaryi) − M1−1 m̸=primary rank(mi)

- 16: ρ ← UpdateRiskRatio(B, {∆(ranki) }, ρ)
- 17: τ ← Percentile({∆(ranki) ≥ 0}, 100 − 100ρ)
- 18: M(i) ← [∆ (ranki) > τ]
- 19: // Phase 2: Window-focused optimization with selective KL
- 20: for each mini-batch (x0, A, M) do
- 21: Sample t ∼ U(T ), xt ← (1 − t)x0 + tϵ
- 22: vθ, vθold, vθref ← StreamingNFT(xt, t, s, W)
- 23: Extract: vθ,W, vθold,W, vθref,W, x0,W
- 24: // Policy loss on window
- 25: v+ ← β · vθ,W + (1 − β) · vθold,W
- 26: v− ← (1 + β) · vθold,W − β · vθ,W
- 27: r˜ ← clip(A/Amax)/2 + 0.5
- 28: Lpolicy ← r˜∥v+ − x0,W∥2 + (1 − r˜)∥v− − x0,W∥2
- 29: // Selective KL on high-uncertainty samples
- 30: LKL ← |M|1 i:M(i)=1 ∥vθ,W(i) − vθ(i)

ref,W∥2

- 31: Update: θ ← θ − η∇θ(Lpolicy + λKLLKL)
- 32: end for
- 33: // Adaptive reference update
- 34: if LKL > τKL or k − klast > Kmax then
- 35: θref ← θ, klast ← k
- 36: end if
- 37: θold ← γ · θold + (1 − γ) · θ
- 38: end for

[Figure 288]

###### Fig. S2: Qualitative comparison on long video generation (30s) under the singleprompt setting (Set 1).

[Figure 289]

[Figure 290]

###### Fig. S4: Additional qualitative comparison on long video generation (30s) under the single-prompt setting (Set 3).

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

