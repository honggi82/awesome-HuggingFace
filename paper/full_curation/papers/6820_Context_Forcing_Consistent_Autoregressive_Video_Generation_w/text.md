## Context Forcing: Consistent Autoregressive Video Generation with Long Context

#### Shuo Chen*1 Cong Wei*2 Sun Sun2 Ping Nie2 Kai Zhou3 Ge Zhang4 Ming-Hsuan Yang1 Wenhu Chen2

# arXiv:2602.06028v1[cs.CV]5Feb2026

###### Website: https://chenshuo20.github.io/Context Forcing Code: https://github.com/TIGER-AI-Lab/Context-Forcing

Frame ID 800, 50s Frame ID 0, 0s Quality vs. Context Length

Frame ID 0, 0s

Frame ID 800, 50s

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

LongLive

3.0sctx LongLive

DINOScoreClip-FScore

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

LongLive

3.75sctx

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

5.25sctx Ours

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

20s+ctx

Video Length(sec)

Figure 1. Context Forcing mitigates the forgetting–drifting dilemma. (1) State-of-the-art models are limited by short context windows (3.0–9.2s), which leads to poor long-term consistency (Forgetting). (2) For streaming long-context tuning baselines (e.g., LongLive), enlarging the context window during inference (3.0 → 5.25s) causes error accumulation and distribution shift (Drifting). In contrast, Context Forcing supports 20s+ context while maintaining strong long-term consistency.

### Abstract

Recent approaches to real-time long video generation typically employ streaming tuning strategies, attempting to train a long-context student using a short-context (memoryless) teacher. In these frameworks, the student performs long rollouts but receives supervision from a teacher limited to short 5-second windows. This structural discrep-

*Equal contribution 1Department of EECS, University of California, Merced, USA 2University of Waterloo, Canada 3Netmind.AI 4M-A-P. Correspondence to: Ming-Hsuan Yang <mhyang@ucmerced.edu>, Wenhu Chen <wenhuchen@uwaterloo.ca>.

Preprint. February 6, 2026.

ancy creates a critical student-teacher mismatch: the teacher’s inability to access long-term history prevents it from guiding the student on global temporal dependencies, effectively capping the student’s context length. To resolve this, we propose Context Forcing, a novel framework that trains a long-context student via a long-context teacher. By ensuring the teacher is aware of the full generation history, we eliminate the supervision mismatch, enabling the robust training of models capable of long-term consistency. To make this computationally feasible for extreme durations (e.g., 2 minutes), we introduce a context management system that transforms the linearly growing context into a Slow-Fast Mem-

ory architecture, significantly reducing visual redundancy. Extensive results demonstrate that our method enables effective context lengths exceeding 20 seconds—2–10× longer than state-of-theart methods like LongLive and Infinite-RoPE. By leveraging this extended context, Context Forcing preserves superior consistency across long durations, surpassing state-of-the-art baselines on various long video evaluation metrics.

### 1. Introduction

In recent years, video diffusion models based on architectures such as the Denoising Diffusion Transformer(DiT) (Peebles & Xie, 2023) have achieved remarkable success in generating photorealistic videos (Wan et al., 2025). While bidirectional models perform well for short clips, their computational cost limits long-form generation. To address this, the field is moving toward causal video architectures (Yin et al., 2024c; Huang et al., 2025), which, like Large Language Models, can theoretically generate infinite-length videos by predicting future frames from past context.

Despite this promise, current causal video models struggle to maintain coherence over long-term contexts. Effective context is often limited to just a few seconds (Cui et al., 2025; Yang et al., 2025; Zhang & Agrawala, 2025; Huang et al., 2025; Yesiltepe et al., 2025), beyond which identity shifts and temporal inconsistencies emerge. We identify the root cause as a fundamental student-teacher mismatch. As illustrated in Figure 2(b), current methods typically train a student to perform long rollouts using supervision from a memoryless teacher limited to short windows (e.g., 5 seconds). The teacher’s inability to access long-term history prevents it from guiding the student on global temporal dependencies, effectively capping the student’s learnable context length.

This mismatch results in a critical challenge for realtime long-context video generation, which we term the Forgetting-Drifting Dilemma (Figure 1). Existing methods face an unavoidable trade-off:

- • Forgetting: Restricting the model to a short memory window minimizes error accumulation but causes the model to lose track of previous subjects and scenes during long rollout.
- • Drifting: Maintaining a long context preserves identity but exposes the model to its own accumulated errors. Without a teacher capable of correcting these long-term deviations, the video distribution progressively drifts away from the real manifold.

To address these challenges, we propose Context Forcing,

a framework that distills a long-context teacher into a longcontext student. Our approach resolves the context-drifting dilemma by bridging the capability gap between teacher and student. We first leverage a Context Teacher pretrained on video continuation tasks, which is capable of processing long-context inputs. This teacher guides the student via Contextual Distribution Matching Distillation, explicitly transferring the ability to model long-term dependencies and ensuring global consistency. Furthermore, by exposing the student to imperfect, self-generated contexts during training, we enable it to actively recover from accumulated artifacts. The resulting robustness allows for 2−10× longer duration Key-Value (KV) cache management (maintaining 20+ seconds of history) compared to prior SOTA (1.5–9.2 seconds of history) during inference, effectively addressing the forgetting-drifting trade-off and enabling consistent, long-form video generation.

The contributions of this work are:

- • We introduce Context Forcing, a novel framework that mitigates the student-teacher mismatch in training real-time long video models. By distilling from a longcontext teacher aware of the full generation history, we enable the robust training of a long-context student capable of long-term consistency.
- • To support this, we design a context management system that transforms the linearly growing context into a Slow-Fast Memory architecture, significantly reducing visual redundancy. This mechanism enables effective context lengths exceeding 20 seconds—2–10× longer than state-of-the-art methods.
- • We demonstrate that, equipped with these extended context lengths, our model preserves superior consistency across long durations, surpassing state-of-the-art baselines on various long video evaluation metrics.

### 2. Related Work

Long Video Generation. The high computational cost of Diffusion Transformers (DiTs) (Kong et al., 2024; Wan et al., 2025; Peebles & Xie, 2023; Yang et al., 2024) has limited video generation to short clips. To extend temporal horizons, many works combine diffusion with autoregressive (AR) prediction (Kim et al., 2024; Lin et al., 2025; Gu et al., 2025), including NOVA (Deng et al., 2024), PyramidFlow (Jin et al., 2024), and MAGI-1 (Teng et al., 2025). Other approaches improve efficiency via causal or windowed attention and KV caching (Yin et al., 2024c; Huang et al., 2025; Kodaira et al., 2025), or extend context through training-free positional encoding modifications (Lu et al., 2024; Lu & Yang, 2025; Zhao et al., 2025). However, most methods still struggle with global consistency beyond 10-20 seconds. A key challenge of long video generation is error accumulation (drifting), addressed either during training by

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|[Figure 22]|
|---|

[Figure 23]

[Figure 24]

[Figure 25]

Short Teacher Short Teacher

Long Teacher

Memory

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

|[Figure 32]|
|---|

|[Figure 33]|
|---|

Long Student

Short Student

Long Student

Memory

|[Figure 34]<br><br>[Figure 35]|
|---|

|[Figure 36]<br><br>[Figure 37]|
|---|

[Figure 38]

|[Figure 39]<br><br>[Figure 40]|
|---|

|[Figure 41]<br><br>[Figure 42]|
|---|

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

(c) Context Long Tuning(ours) long teacher → long student

(a) Self-Forcing Training short teacher → short student

(b) Memoryless Long Tuning short teacher → long student

- Figure 2. Training paradigms for AR video diffusion models. (a) Self-forcing: A student matches a teacher capable of generating only 5s video using a 5s self-rollout. (b) Longlive (Yang et al., 2025): The student performs long rollouts supervised by a memoryless 5s teacher on random chunks. The teacher’s inability to see beyond its 5s window creates a student-teacher mismatch. (c) Context Forcing (Ours): The student is supervised by a long-context teacher aware of the full generation history, resolving the mismatch in (b).

exposing models to drifted inputs (Cui et al., 2025; Chen et al., 2024; 2025) or during inference via recaching, sampling strategies, or feedback (Yang et al., 2025; Zhang & Agrawala, 2025; Li et al., 2025a). To enable real-time generation, recent works distill multi-step diffusion into few-step models (Valevski et al., 2024; Liu et al., 2023; Luo et al., 2023; Sauer et al., 2024), including Distribution Matching Distillation (DMD/DMD2) (Yin et al., 2024b;a; Wang et al.,

mechanisms are key to extending temporal context and maintaining consistency in long-horizon generation. WorldPlay (Sun et al., 2025), Context as Memory (Yu et al., 2025), and WorldMem (Xiao et al., 2025) and Framepack (Zhang & Agrawala, 2025) introduce explicit memory structures to accumulate scene or contextual information over time, while RELIC (Hong et al., 2025) employs recurrent latent states for efficient long-range dependency modeling. PFP (Zhang et al., 2026) compress long videos into short context by training a novel compression module.

- 2023) and Consistency Models (CM) (Song et al., 2023; Wang et al., 2024).

Causal Video Generation. Causal video generation synthesizes video sequences under strict temporal ordering constraints, thereby enabling streaming inference and longhorizon synthesis. Although early autoregressive models (Vondrick et al., 2016; Kalchbrenner et al., 2017) generated frames or tokens sequentially, they often suffered from error accumulation and poor scalability. Recent diffusionbased frameworks have improved visual fidelity by incorporating causal architectural priors, such as the blockwise causal attention introduced in CausVid (Yin et al.,

- 2024c). To mitigate distribution shift, Self-Forcing (Huang et al., 2025), LongLive (Yang et al., 2025) and SelfForcing++ (Cui et al., 2025) align training with inference by conditioning on prior outputs via KV caching and rolloutbased objectives. InfinityRoPE (Yesiltepe et al., 2025) achieve a reduction of error accumulation by modifying positional encodings. Further research has addressed efficient long-context inference through windowed attention, as seen in StreamDiT (Kodaira et al., 2025). Memory Mechanism for Video Generation Memory

### 3. Methodology

We operate within the causal autoregressive framework, where the generation of a long video X1:N is decomposed into a sequence of conditional steps over frames or short chunks Xt. State-of-the-art methods, such as CausVid (Yin et al., 2024c) and Self-Forcing (Huang et al., 2025), enforce strict temporal causality via block-wise attention, modeling the distribution as t p(Xt | X<t). These approaches typically employ Distribution Matching Distillation (DMD) (Yin et al., 2024b) to distill a high-quality bidirectional teacher into a causal student. Building on these foundations, we introduce Context Forcing.

Our goal is to train a causal video diffusion model, parameterized by θ, whose induced distribution over long videos pθ(X1:N) matches the real data distribution pdata(X1:N). Here, N represents a duration spanning tens or hundreds of seconds. The objective is to minimize the global longhorizon KL divergence:

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

1 2 3 4 5 6 7 8 9 10 11 RoPE Index

[Figure 51]

Memory Long Student

| |
|---|

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Sink Slow Memory

Fast Memory Initialization

|1|
|---|

Fill Memory

|8|
|---|

|9|
|---|

|10|
|---|

|11|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

|5|
|---|

|6|
|---|

|7|
|---|

|1|
|---|

[Figure 57]

Memory Fake score function

[Figure 58]

Update Fast Mem

|9|
|---|

|10|
|---|

|11|
|---|

|12|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

|5|
|---|

|6|
|---|

|7|
|---|

|1|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

|8|
|---|

###### DMD Loss

Update Slow Mem

|9|
|---|

|10|
|---|

|11|
|---|

|12|
|---|

|2|
|---|

|3|
|---|

|4|
|---|

|5|
|---|

|6|
|---|

|12|
|---|

|1|
|---|

|11 ,|
|---|

|12|
|---|

) > 𝜏

sim(

[Figure 62]

|7|
|---|

Memory

Real score function

[Figure 63]

- Figure 3. Context Forcing and Context Management System. We use KV Cache as the context memory, and we organize it into three parts: sink, slow memory and fast memory. During contextual DMD training, the long teacher provides supervision to the long student by utilizing the same context memory mechanism.

diffused version of x at timestep t. The gradient is given by:

KL pθ(X1:N) ∥ pdata(X1:N) . (1)

Lglobal = min

θ

∂Gθ(z) ∂θ

∇θLlocal ≈ Ez,t,x

Directly optimizing Eq. (1) ensures long-term coherence but is computationally intractable for large N. By applying the chain rule of KL divergence, we decompose the global objective into two components:

wtαt sθ(xt,t) − sT(xt,t)

,

t

(4) where sθ and sT are the student and teacher scores, respectively, and wt is a weighting function. This stage ensures pθ(X1:k) ≈ pdata(X1:k), providing high-quality contexts for the subsequent stage.

##### Lglobal = KL pθ(X1:k)∥pdata(X1:k)

Llocal: Local Dynamics

#### 3.2. Stage 2: Contextual Distribution Matching

##### + EX

1:k∼pθ KL pθ(Xk+1:N|X1:k)∥pdata(Xk+1:N|X1:k)

Stage 2 targets Lcontext, the second term of Eq. (2). This term requires minimizing the divergence between the student’s continuation pθ(·|X1:k) and the true data continuation pdata(·|X1:k).

Lcontext: Global Continuation Dynamics

(2) This decomposition motivates our two-stage curriculum:

- • Stage 1 (Optimizing Llocal): We match the distribution of short windows (X1:k) to the real data distribution to learn local dynamics.
- • Stage 2 (Optimizing Lcontext): We match the model’s continuation predictions (Xk+1:N) with the temporal evolution of real data to learn long-term dependencies.

However, pdata is not directly accessible for arbitrary contexts generated by the student. To solve this, we employ a pretrained Context Teacher T, which provides a reliable proxy distribution pT(Xk+1:N | X1:k). We rely on two key assumptions to justify using the teacher as a target:

#### Assumption 1 (Teacher reliability near student contexts).

Whenever the student context X1:k ∼ pθ(X1:k) remains close to the real data manifold, the teacher’s continuation pT(Xk+1:N | X1:k) is accurate. This holds whenever the teacher is well-trained on real video prefixes.

#### 3.1. Stage 1: Local Distribution Matching

The first stage warms up the causal student by minimizing Llocal. Given a teacher distribution pT(X1:k) (approximately the real data), we optimize:

#### Assumption 2 (Approximate real prefixes). Stage 1 suc-

cessfully aligns pθ(X1:k) with pdata(X1:k). This ensures that student rollouts remain within the teacher’s reliable region during Stage 2 training.

Llocal = KL pθ(X1:k)∥pT(X1:k) , (3)

where k corresponds to a 1–5 second window. We estimate the distribution matching gradient follow DMD (Yin et al.,

Under these assumptions, we approximate pdata ≈ pT in the second term of Eq. (2), yielding the Contextual DMD

- 2024b). Let x = Gθ(z) for noise z, and let xt be the

Frame ID 0, 0s Frame ID 160, 10s Frame ID 320, 20s Frame ID 480, 30s Frame ID 640, 40s Frame ID 800, 50s Frame ID 960, 60s

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

(Context3.0s) InfinityRoPE

LongLive

RollingForcing

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

(Context6.0s) FramePack-F1

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

(Context1.5s) Ours

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

(Context9.2s)

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

(Context20+s)

- Figure 4. Comparison on 1-min Video Generation. Our method keeps both the background and subject consistent across 1-min video, while other baselines have different levels drifting or identity shift.

(CDMD) objective: LCDMD = EX

1:k∼pθ(X1:k)

KL pθ(Xk+1:N | X1:k)∥pT(Xk+1:N | X1:k)

(5) Crucially, the expectation is over X1:k ∼ pθ, ensuring the student is trained on its own rollouts, thereby mitigating exposure bias.

Score-based CDMD Gradient. We estimate the gradient of Eq. (5) using a conditional variant of the DMD gradient. Let xcont = Gθ(zcont | X1:k) be the generated continuation, and xt,cont be its diffused version. Running both fake score and real score models on the same student-generated context produces scores sθ(· | X1:k) and sT(· | X1:k). The gradient is:

∇θLCDMD ≈ EX

wtαt sθ(xt,cont,t | X1:k)

1:k∼pθ zcont,t

(6)

∂Gθ(zcont | X1:k) ∂θ

− sT(xt,cont,t | X1:k)

.

By descending Eq. (6), we align the student’s long-term autoregressive dynamics with the teacher’s robust priors.

Long Self-Rollout Curriculum. Minimizing Lcontext requires the context horizon k to approach the full sequence length N. However, sampling X1:k ∼ pθ for large k early in training causes severe distribution shift due to accumulated drift. To mitigate this, we employ a dynamic horizon schedule Nmax(t) that grows linearly with training step t. At each iteration, the rollout length is sampled as k ∼ U(kmin,Nmax(t) ).

This curriculum initializes training in the stable Stage 1 regime (k ≈ kmin) and progressively exposes the model to long-range dependencies.

Clean Context Policy. Self Forcing (Huang et al., 2025) typically generates rollouts using a random timestep selection strategy to ensure supervision across all diffusion steps. We retain this random exit policy for the target frames Xk+1:N to preserve gradient coverage, but enforce that the context frames X1:k are fully denoised. We apply a complete few-step denoising process to the context. This decoupling ensures the context remains informative and aligned with the teacher’s training distribution but also maintains supervision for every diffusion step.

#### 3.3. Context Management System

Our teacher and student models share an identical architecture; both are autoregressive generative models augmented with a memory module for context retention. We utilize KV caches to represent the context X1:k. To maintain efficiency as the sequence length k grows, we design a KV cache management system inspired by dual-process memory theories. Specifically, the cache M is partitioned into three functional components: an Attention Sink, Slow Memory (Context), and Fast Memory (Local). Both the student and teacher are equipped with this system.

Cache Partitioning. The total cache is defined as the union of disjoint sets:

##### M = S ∪ Cslow ∪ Lfast.

Frame ID 0, 0s Frame ID 160, 10s Frame ID 320, 20s Frame ID 480, 30s Frame ID 640, 40s Frame ID 800, 50s Frame ID 960, 60s

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

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

- Figure 5. Qualitative Results of Context Forcing. Our method enables minute-level video generation with minimal drifting and high consistency across diverse scenarios.

- • Attention Sink (S): Retains initial Nstokens to stabilize attention, following StreamingLLM (Yang et al., 2025; Shin et al., 2025).
- • Slow Memory (Cslow): A long-term buffer of up to Nc tokens, storing high-entropy keyframes and updating only with significant new information.
- • Fast Memory (Lfast): A rolling FIFO queue of size Nl, capturing immediate local context with short-term persistence.

Surprisal-Based Consolidation. Upon generating a new token xt and enqueuing it into the Fast Memory Lfast, we evaluate its informational value relative to the immediate temporal context. We postulate that tokens exhibiting high similarity to their predecessors carry redundant information (low surprisal), whereas dissimilar tokens indicate significant state transitions or visual changes (high surprisal).

To capture these high-information moments efficiently, we compare the key vector kt of the current token with that of the immediately preceding token kt−1. The consolidation policy π(xt) determines whether xt is promoted to Slow Memory Cslow:

Consolidate if sim(kt,kt−1) < τ, Discard otherwise,

(7)

π(xt) =

where τ is a similarity threshold. This criterion ensures that Cslow prioritizes storing temporal gradients and distinctive events rather than static redundancies. As with standard

cache management, if |Cslow| > Nc after consolidation, the oldest entry is evicted to maintain fixed memory complexity.

Bounded Positional Encoding. Unlike standard autoregressive video models (Huang et al., 2025; Cui et al., 2025), where positional indices grow unbounded (pt = t → ∞), leading to distribution shifts on long sequences, we adopt Bounded Positional Indexing. All tokens’ temporal RoPE positions are constrained to a fixed range Φ = [0,Ns+Nc+ Nl − 1] regardless of generation step t:

 

- i ∈ [0,Ns − 1] if x ∈ S,
- j ∈ [Ns,Nc − 1] if x ∈ Cslow,
- k ∈ [Nc,Nc + Nl − 1] if x ∈ Lfast.

(8)

ϕ(x) =



This creates a static attention window where recent history (Fast) slides through high indices, while salient history (Slow) is compressed into lower indices, stabilizing attention over long sequences.

#### 3.4. Robust Context Teacher Training

Standard training conditions the model on ground-truth context, but inference relies on self-generated history, creating a distribution shift known as exposure bias. To ensure our Context Teacher provides robust guidance even when the student drifts, we adopt Error-Recycling Fine-Tuning (ERFT) (Li et al., 2025a).

Rather than training on clean history X1:k, we inject realistic

accumulated errors into the teacher’s context. We construct a perturbed context X˜1:k = X1:k + I · edrift, where edrift is sampled from a bank of past model residuals and I is a Bernoulli indicator. The teacher is optimized to recover the correct velocity vtarget from X˜1:k. This active correction capability ensures pT(· | X1:k) remains a reliable proxy for pdata even when the student’s context X1:k degrades.

### 4. Experiments

Implementation Details. We implement the robust context teacher using Wan2.1-T2V-1.3B (Wan et al., 2025) as the base model. To construct the training dataset, we filter the Sekai (Li et al., 2025b) and Ultravideo (Xue et al., 2025) collections to retain high-quality videos exceeding 10 seconds in duration, yielding a total of 40k clips. The robust context teacher is trained for 8k steps with a batch size of 8. During training, frames are sampled uniformly from the 5–20 second interval of the video data to serve as context.

The student model also utilizes the Wan2.1-T2V-1.3B model. In Stage 1, we employ 81-frame video clips from the VidProM (Wang & Yang, 2024) dataset and train for 600 iterations with a batch size of 64. In Stage 2, which focuses on context distillation, we extend the rollout horizon to video lengths of 10–30 seconds to address short-term memory limitations. This phase is similarly trained on the VidProM dataset for 500 iterations using the same batch size. For both teacher and student models, we set the KV cache size to 21 latent frames, and set Ns = 3,Nc = 12,Nl = 6,τ = 0.95. We implement Surprisal-Based Consolidation at 2-chunk intervals. Upon chunk consolidation, we retain only the first latent, effectively extending the context beyond 20s.

Baselines. We evaluate our method against three distinct categories of baselines. The first category comprises bidirectional diffusion models, specifically LTX-Video (HaCohen et al., 2024) and Wan2.1 (Wan et al., 2025). The second category includes autoregressive models such as SkyReels-V2 (Chen et al., 2025), MAGI-1 (Teng et al.,

- 2025), CausVid (Yin et al., 2024c), NOVA (Deng et al.,

- 2024), Pyramid-Flow (Jin et al., 2024), and Self Forcing (Huang et al., 2025). The third category consists of recent methods targeting long video generation within autoregressive frameworks. These include LongLive (Yang et al., 2025) with a context length of 3 seconds, Self Forcing++ (Cui et al., 2025), Rolling Forcing (Liu et al., 2025) with a context length of 6 seconds, and Infinity-RoPE (Yesiltepe et al., 2025) with a context length of 1.5 seconds. Finally we include a long context baseline Framepack (Zhang & Agrawala, 2025) with a context length of 9.2 seconds.

Evaluation. We report performance on VBench (Zheng et al., 2025) following (Huang et al., 2025; Yang et al.,

- 2025). Beyond standard benchmarks, we assess fine-grained

Student Generate Frames Context Teacher Generated Frames

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Figure 6. Video Continuation with Robust Context Teacher. Context teacher can generate next segment videos with context generated by student.

consistency using DINOv2 (Oquab et al., 2023) (structural identity), CLIP-F (Radford et al., 2021) (semantic context), and CLIP-T (prompt alignment). To improve robustness against temporal artifacts, we implement window-based sampling: for any timestamp t, we compute the average cosine similarity between the first frame (V0) and frames within [t − 0.5s,t + 0.5s]. We average results over five random seeds per prompt to ensure statistical reliability. This approach effectively measures long-term subject and background consistency.

#### 4.1. Video Continuation with Robust Context Teacher

To evaluate the context teacher, we feed the teacher model with videos generated by the student model after Stage 1 training. We then assess the consistency of the complete sequence, which comprises the initial context with the generated continuation. Evaluation is performed using 100 text prompts randomly sampled from MovieGenBench (Polyak et al., 2024). As illustrated in Figure 6, the context teacher effectively synthesizes the subsequent video segment, providing empirical support for Assumptions 1 and 2. Furthermore, we quantitatively evaluate the performance of the context teacher using student-generated videos as input, reporting subject and background consistency on VBench, as well as DINOv2, CLIP-F, and CLIP-T scores. The consistency metrics for the complete 10-second sequence are presented in Table 1, further demonstrating that the context teacher consistently produces reliable continuations from student-generated contexts.

Table 1. Single-prompt 60-second long video consistency evaluation.

Model Context Length ↑ Dino Score ↑ Clip-F Score ↑ Clip-T Score ↑ Background Subject 10s 20s 30s 40s 50s 60s 10s 20s 30s 40s 50s 60s 10s 20s 30s 40s 50s 60s Consistency↑ Consistency↑

FramePack-F1 9.2s 81.86 77.95 78.84 73.10 69.52 68.50 95.41 93.42 93.92 92.44 92.47 89.36 36.36 34.67 33.75 33.84 34.77 32.30 91.61 89.15 LongLive 3.0s 91.25 89.55 89.12 86.51 87.83 86.26 95.74 94.25 92.80 91.50 93.12 94.82 36.95 35.80 36.17 36.58 35.92 37.13 94.92 93.05 Infinate RoPE 1.5s 91.18 88.17 85.37 79.80 81.10 83.72 94.09 91.71 91.13 89.71 86.11 88.88 35.26 35.03 35.88 32.56 32.29 32.28 92.42 90.11 Ours, teacher 20+s 87.61 - - - - - 95.52 - - - - - 35.93 - - - - - 95.24 94.87 Ours, student 20+s 91.45 89.25 89.66 87.45 88.33 87.89 95.82 94.75 95.05 93.88 94.92 95.35 37.12 36.25 36.72 37.15 37.08 37.66 95.95 95.68

Table 2. Comparison of video generation models across architecture families.

Model #Params Throughput (FPS) ↑ Evaluation scores on 5s ↑ Evaluation scores on 60s ↑ Background Subject Background Subject

Total Quality Semantic Consistency Consistency Total Quality Semantic Consistency Consistency Bidirectional models

LTX-Video 1.9B 8.98 80.00 82.30 70.79 95.30 95.01 - - - - Wan2.1 1.3B 0.78 84.26 85.30 80.09 96.96 95.99 - - - - -

Autoregressive models SkyReels-V2 1.3B 0.49 82.67 84.70 74.53 96.83 96.07 70.47 75.30 51.15 89.95 84.99 MAGI-1 4.5B 0.19 79.18 82.04 67.74 96.83 95.83 69.87 76.12 44.87 87.76 79.46 CausVid 1.3B 17.0 81.20 84.05 69.80 95.12 95.96 71.04 76.80 48.01 89.85 86.75 NOVA 0.6B 0.88 80.12 80.39 79.05 95.16 93.38 65.25 70.25 45.24 88.06 77.50 Pyramid Flow 2B 6.7 81.72 84.74 69.62 96.09 96.08 - - - - Self Forcing, chunk-wise 1.3B 17.0 84.31 85.07 81.28 95.98 96.29 71.86 77.20 50.51 87.84 83.60

Long autoregressive models LongLive 1.3B 20.7 84.87 86.97 76.47 96.55 95.82 83.64 84.53 74.97 94.62 93.88 Self Forcing++ 1.3B 17.0 83.11 83.79 80.37 - - - - - - Rolling Forcing 1.3B 15.8 81.22 84.08 69.78 96.11 96.02 79.31 81.87 67.69 94.12 93.10 Infinity-RoPE 1.3B 17.0 81.79 83.27 75.87 96.34 95.14 79.99 80.81 74.30 94.21 93.05

Ours, student model 1.3B 17.0 83.44 84.98 77.29 97.38 96.84 82.45 83.55 76.10 95.34 94.88

#### 4.2. Text-to-Short Video Generation

Quantitative Results. We quantitatively compare our method against baselines. We evaluate 5-second video generation on the VBench dataset using its official extended prompts. The results summarized in Table 2 demonstrate that our method achieves performance comparable to the baselines on short video generation.

#### 4.3. Text-to-Long Video Generation

Qualitative Results. We evaluate our proposed method against baseline models on 60-second video generation, with qualitative results illustrated in Figure 2. By leveraging a slow-fast memory architecture with a KV cache size of 21 and a context span exceeding 20s, our method achieves superior consistency and effectively mitigates content drifting compared to the baselines.

Quantitative Results. We evaluate 60-second video generation performance on the VBench with results summarized in Table 2, using its offical extened prompts. Additionally, we report DINOv2, CLIP-F, and CLIP-T scores in Table 1, using 100 text prompts randomly sampled from MovieGenBench (Polyak et al., 2024), following the same experimental protocol as in Section 4.1. Both tables demonstrate that our method achieves high consistency, particularly during extended video sequences. Notably, while LongLive also achieves competitive scores, qualitative inspection reveals that it frequently exhibits abrupt scene resets and cyclic motion patterns, shown in Figure 8 in Appendix.

Table 3. Ablation study on Slow Memory Sampling Strategy, Context DMD, and Bounded Positional Encoding (evaluated on 60s).

Total Score ↑

Quality Score ↑

Semantic Score ↑

Background Consistency ↑

Subject Consistency ↑

Dynamic Degree ↑

Model

Slow Memory Sampling Strategy

- Uniform sample, interval 1 80.82 82.20 75.32 92.45 92.10 52.15

- Uniform sample, interval 2 81.11 82.61 75.12 93.12 92.85 55.30

Contextual Distillation w/o. Contextual Distillation 80.36 82.28 72.70 93.55 93.20 48.12

Bounded Positional Encoding

w/o. Bounded Positional Encoding 73.52 75.44 65.82 84.68 79.24 27.45 Ours 82.45 83.55 76.10 95.34 94.88 58.26

#### 4.4. Ablation Studies

Slow Memory Sampling Strategy Our method employs a selection strategy based on key-vector similarity to sample context from slow memory. Unlike fixed uniform sampling, this strategy dynamically selects historical chunks that exhibit low similarity to the current generation window, thereby preserving critical semantic information over time. We compare our approach against alternative baselines, specifically uniform sampling with intervals of 1 and 2 chunks. As summarized in Table 3, the results demonstrate the effectiveness of similarity-based selection in maintaining long-term consistency.

Context DMD Distillation We evaluate the contribution of Contextual Distribution Matching Distillation by comparing our full model against a training-free baseline. In the latter, our context management system is applied directly after Stage 1 training without the DMD process. The results in Table 3 indicate that removing Context DMD leads to a degradation in both semantic and temporal con-

Context Teacher Generate Frames(0-30s)

[Figure 165]

[Figure 166]

[Figure 167]

w.ERFTw/o.ERFT

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

- Figure 7. Ablation on Error-Recycling Fine-Tuning (ERFT). With ERFT, context teacher is more robust to accumulate error.

sistency, highlighting its critical role in enabling coherent, long-horizon video generation.

Error-Recycling Fine-Tuning (ERFT). We test the context teacher by taking 5s videos from the video dataset as input for autoregressive rollout. As shown in Figure 7, the visualization of 30s generation results indicates that with robust context training, the context teacher produces videos with fewer artifacts. This results in a better distribution for further contextual distillation.

Bounded Positional Encoding. We investigate the impact of Bounded Positional Encoding by excluding it during inference, with quantitative results presented in Table 3. In the absence of this encoding, we observe a significant performance drop in both background stability and subject consistency. This demonstrates its essential role in stabilizing long-range attention and mitigating temporal drift during the generation process.

### 5. Conclusion

In this work, we introduced Context Forcing, a framework designed to overcome the fundamental student-teacher mismatch in long-horizon causal video generation. By ensuring the teacher model maintains awareness of long-term history, our approach eliminates the supervision gap that limits existing streaming-tuning methods. To handle the computational demands of extreme durations, we proposed a Slow-Fast Memory architecture that effectively reduces visual redundancy. Extensive experiments demonstrate that Context Forcing achieves effective context lengths of 20+ seconds, a 2–10× improvement over current state-of-the-art baselines. While our method significantly mitigates drifting errors and enhances temporal coherence, the current memory compression strategy still leaves room for optimization regarding information density. Future work can focus on learnable context compression and adaptive memory mechanisms to further improve efficiency and semantic retention for even more complex, open-ended video synthesis.

### Impact Statement

This paper contributes to the advancement of generative AI by enhancing temporal consistency in long video generation. Our work enables the creation of more coherent and realistic visual sequences, which has significant positive potential in digital storytelling, filmmaking, world model and professional video editing. However, we acknowledge that the ability to generate highly consistent long-form videos also increases the risk of creating sophisticated synthetic media or deepfakes that could be used for misinformation. To mitigate these ethical concerns, we advocate for the integration of digital watermarking and provenance standards in downstream applications. We believe that fostering transparency and developing robust detection mechanisms are essential as video generation technology continues to mature.

### References

Chen, B., Mart´ı Mons´o, D., Du, Y., Simchowitz, M., Tedrake, R., and Sitzmann, V. Diffusion forcing: Nexttoken prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, pp. 24081– 24125, 2024.

Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al. Skyreelsv2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.

Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., and Hsieh, C.-J. Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025.

Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., and Wang, X. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024.

Gu, Y., Mao, W., and Shou, M. Z. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al. Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., HoldGeoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al. Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

Huang, X., Li, Z., He, G., Zhou, M., and Shechtman, E. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., and Lin, Z. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024.

Kalchbrenner, N., Van Den Oord, A., Simonyan, K., Danihelka, I., Vinyals, O., Graves, A., and Kavukcuoglu, K. Video pixel networks. In International Conference on Machine Learning, 2017.

Kim, J., Kang, J., Choi, J., and Han, B. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, pp. 89834–89868, 2024.

Kodaira, A., Hou, T., Hou, J., Tomizuka, M., and Zhao, Y. Streamdit: Real-time streaming text-to-video generation. arXiv preprint arXiv:2507.03745, 2025.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Li, W., Pan, W., Luan, P.-C., Gao, Y., and Alahi, A. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025a.

Li, Z., Li, C., Mao, X., Lin, S., Li, M., Zhao, S., Xu, Z., Li, X., Feng, Y., Sun, J., et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025b.

Lin, S., Yang, C., He, H., Jiang, J., Ren, Y., Xia, X., Zhao, Y., Xiao, X., and Jiang, L. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.

Liu, K., Hu, W., Xu, J., Shan, Y., and Lu, S. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

Liu, X., Zhang, X., Ma, J., Peng, J., et al. Instaflow: One step is enough for high-quality diffusion-based text-toimage generation. In International Conference on Learning Representations, 2023.

Lu, Y. and Yang, Y. Freelong++: Training-free long video generation via multi-band spectralfusion. arXiv preprint arXiv:2507.00162, 2025.

Lu, Y., Liang, Y., Zhu, L., and Yang, Y. Freelong: Trainingfree long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, pp. 131434–131455, 2024.

Luo, S., Tan, Y., Patil, S., Gu, D., Von Platen, P., Passos, A., Huang, L., Li, J., and Zhao, H. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2023.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.-Y., Chuang, C.-Y., et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 8748–8763. PmLR, 2021.

Sauer, A., Boesel, F., Dockhorn, T., Blattmann, A., Esser, P., and Rombach, R. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, pp. 1–11, 2024.

Shin, J., Li, Z., Zhang, R., Zhu, J.-Y., Park, J., Schechtman, E., and Huang, X. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. 2023.

Sun, W., Zhang, H., Wang, H., Wu, J., Wang, Z., Wang, Z., Wang, Y., Zhang, J., Wang, T., and Guo, C. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W., Luo, W., et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

Valevski, D., Leviathan, Y., Arar, M., and Fruchter, S. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.

Vondrick, C., Pirsiavash, H., and Torralba, A. Generating videos with scene dynamics. Advances in Neural Information Processing Systems, 2016.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, F.-Y., Huang, Z., Bergman, A., Shen, D., Gao, P., Lingelbach, M., Sun, K., Bian, W., Song, G., Liu, Y., et al. Phased consistency models. Advances in Neural Information Processing Systems, pp. 83951–84009, 2024.

Wang, W. and Yang, Y. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, pp. 65618–65642, 2024.

Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, pp. 8406–8441, 2023.

Xiao, Z., Lan, Y., Zhou, Y., Ouyang, W., Yang, S., Zeng, Y., and Pan, X. Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.

Xue, Z., Zhang, J., Hu, T., He, H., Chen, Y., Cai, Y., Wang, Y., Wang, C., Liu, Y., Li, X., et al. Ultravideo: Highquality uhd video dataset with comprehensive captions. arXiv preprint arXiv:2506.13691, 2025.

Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al. Longlive: Realtime interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Yesiltepe, H., Meral, T. H. S., Akan, A. K., Oktay, K., and Yanardag, P. Infinity-RoPE: Action-controllable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649, 2025.

Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, B. Improved distribution matching distillation for fast image synthesis. Advances in Neural Information Processing Systems, pp. 47455–47487, 2024a.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.

Yin, T., Zhang, Q., Zhang, R., Freeman, W. T., Durand, F., Shechtman, E., and Huang, X. From slow bidirectional to fast causal video generators. arXiv e-prints, pp. arXiv– 2412, 2024c.

Yu, J., Bai, J., Qin, Y., Liu, Q., Wang, X., Wan, P., Zhang, D., and Liu, X. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025.

Zhang, L. and Agrawala, M. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, (3):5, 2025.

Zhang, L., Cai, S., Li, M., Zeng, C., Lu, B., Rao, A., Han, S., Wetzstein, G., and Agrawala, M. Pretraining frame preservation in autoregressive video memory compression, 2026. URL https://arxiv.org/abs/2512.

23851.

Zhao, M., He, G., Chen, Y., Zhu, H., Li, C., and Zhu, J. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Gu, L., Zhang, Y., He, J., Zheng, W.-S., et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Frame ID 0

Frame ID 525 …

Frame ID 524 Frame ID 524 Frame ID 526

Frame ID 523

|[Figure 175]|
|---|

|[Figure 176]|
|---|

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

- Figure 8. Visual artifacts in LongLive. The model exhibits a sudden flashback artifact, where the video abruptly resets to the initial frame after 524 frames, disrupting temporal continuity.

### A. Preliminaries

Causal Autoregressive Models. Causal autoregressive models generate videos at the frame or short-chunk level (Xt) while enforcing strict temporal causality. Methods such as CausVid (Yin et al., 2024c) and Self-Forcing (Huang et al., 2025) adopt block-wise causal attention, allowing bidirectional self-attention within each chunk Xt but restricting information flow across chunks. Video generation is formulated as P(Xt | X<t). In Self-Forcing, the student model is stochastically conditioned on its own generated outputs Xˆ<t during training. These models typically employ Distribution Matching Distillation (DMD) (Yin et al., 2024b) to distill knowledge from a bidirectional teacher into a causal student.

### B. Visual artifacts in LongLive.

While LongLive achieves respectable quantitative scores, we observe that it frequently suffers from abrupt scene resets and repetitive, cyclic motion patterns, as illustrated in Figure 8.

### C. Algorithm of Context Forcing. Algorithm block of context forcing.

Algorithm 1 Contextual DMD Require: Denoise timesteps {t1,..,tT} Require: Pre-trained teacher sreal Require: Checkpoints from stage 1, student score function sfake, AR diffusion model Gϕ Require: Text prompt dataset D, rollout decay step sd, rollout range (L0,L1), context window c, teacher length l, local

attention size a

- 1: Initialize, step s = 0
- 2: Initialize model output X ← []
- 3: Initialize KV cache C ← []
- 4: while training do
- 5: Sample prompt p ∼ D
- 6: Sample rollout length L = Uniform(L0, ss

d

× (L1 − L0) + L0 + 1)

- 7: Sample random exit r = Uniform(1,2,...,T)
- 8: for i = 1,...,L do
- 9: Initialize xit ∼ N(0,I)
- 10: if L − r − l ≤ i < L − l then
- 11: r′ = T
- 12: else
- 13: r′ = r
- 14: end if
- 15: for j = 1,...,r′ do
- 16: if j = r′ then
- 17: Enable gradient computation
- 18: xˆi0 ← Gϕ(xit

j

,tj,C)

- 19: X.append(ˆxi0)
- 20: Disable gradient computation
- 21: C ← GCϕ (ˆxi0,0,C)
- 22: else
- 23: Disable gradient computation
- 24: xˆi0 = Gϕ(xit

j

,tj,C)

- 25: Sample ϵ ∼ N(0,I)
- 26: Set xit

j−1

← addnoise(ˆxi0,ϵ,tj−1)

- 27: end if
- 28: end for
- 29: context video vc = X[L − r − l : L − l], target noise vt = addnoise(X[L − l :],t)
- 30: Compute Contextual DMD Loss with sfake(vt,t,vc) and sreal(vt,t,vc)
- 31: end for
- 32: end while

