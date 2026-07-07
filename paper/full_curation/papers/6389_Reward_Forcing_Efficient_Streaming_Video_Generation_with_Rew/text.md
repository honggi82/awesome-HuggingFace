## Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation

Yunhong Lu1,2 Yanhong Zeng†,2 Haobo Li2,4 Hao Ouyang2 Qiuyu Wang2 Ka Leong Cheng2 Jiapeng Zhu2 Hengyuan Cao1 Zhipeng Zhang4

Xing Zhu2 Yujun Shen2 Min Zhang∗,1,3 1Zhejiang University 2Ant Group 3SIAS-ZJU 4SJTU

# arXiv:2512.04678v2[cs.CV]28Dec2025

https://reward-forcing.github.io/

Enhanced object motion dynamics

[Figure 1]

𝑝 𝑝

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

|[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>[Figure 10]|
|---|

⋯

###### Better motion!

Re-DMD

DMD

Immersive scene navigation dynamics

[Figure 11]

[Figure 12]

|[Figure 13]<br><br>[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
|---|

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

⋯

[Figure 24]

23.1 FPS in Real-time!

0 min

1 min

Figure 1. We propose Reward Forcing to distill a bidirectional video diffusion model into a few-step autoregressive student model that enables real-time (23.1 FPS) streaming video generation. Instead of using vanilla distribution matching distillation (DMD), Reward Forcing adopts a novel rewarded distribution matching distillation (Re-DMD) that prioritizes matching towards high-reward regions, leading to enhanced object motion dynamics and immersive scene navigation dynamics in generated videos.

ing while maintaining long-horizon consistency. Second, to better distill motion dynamics from teacher models, we propose a novel Rewarded Distribution Matching Distillation (Re-DMD). Vanilla distribution matching treats every training sample equally, limiting the model’s ability to prioritize dynamic content. Instead, Re-DMD biases the model’s output distribution toward high-reward regions by prioritizing samples with greater dynamics rated by a vision-language model. Re-DMD significantly enhances motion quality while preserving data fidelity. We include both quantitative and qualitative experiments to show that Reward Forcing achieves state-of-the-art performance on standard benchmarks while enabling high-quality streaming video generation at 23.1 FPS on a single H100 GPU.

#### Abstract

Efficient streaming video generation is critical for simulating interactive and dynamic worlds. Existing methods distill few-step video diffusion models with sliding window attention, using initial frames as sink tokens to maintain attention performance and reduce error accumulation. However, video frames become overly dependent on these static tokens, resulting in copied initial frames and diminished motion dynamics. To address this, we introduce Reward Forcing, a novel framework with two key designs. First, we propose EMA-Sink, which maintains fixed-size tokens initialized from initial frames and continuously updated by fusing evicted tokens via exponential moving average as they exit the sliding window. Without additional computation cost, EMA-Sink tokens capture both long-term context and recent dynamics, preventing initial frame copy-

#### 1. Introduction

The scaling of video diffusion transformers (DiTs) [53, 72] has advanced text-to-video generation, producing realistic

* Corresponding author, † Project leader

videos with intricate dynamics [2, 4, 11, 76]. However, their simultaneous denoising of all frames using bidirectional attention hinders interactive applications, which demand streaming generation over extended horizons under strict latency constraints. Achieving both low latency and high visual-dynamic fidelity remains the central challenge.

To achieve efficient streaming video generation, recent advances distill slow pre-trained bidirectional diffusion models into efficient few-step autoregressive student models [7, 30, 89]. In these models, each frame attends only to previous frames with sliding window attention, enabling real-time streaming inference through key-value (KV) cache mechanisms. However, they often suffer from the well-known error accumulation issue [6, 30], as each frame depends on potentially corrupted previous outputs, causing errors to propagate progressively.

To mitigate error accumulation, recent works have adopted attention sink mechanisms that retain initial tokens in the KV cache [45, 62, 82]. Such a design largely recovers the performance of sliding window attention and alleviates long-horizon drifting. However, a new challenge arises: by consistently preserving initial tokens throughout generation, models develop a strong bias toward the starting frame, leading to over-attention on initial content. This manifests as diminished motion dynamics, where subsequent frames fail to evolve naturally, and frequent visual flashbacks that revert to the first frame’s appearance. While classical distribution matching distillation [86, 87] minimizes the divergence between student and teacher output distributions to transfer knowledge, this strategy struggles to address the over-attention issue. The degraded samples, despite their motion deficiencies, usually exhibit good visual quality and already fall close to the teacher distribution, making them difficult to distinguish and optimize.

In this paper, we propose Reward Forcing, a novel framework with two key technical innovations to ensure both high visual and dynamic fidelity for efficient streaming video generation. During training, Reward Forcing generates video chunks autoregressively by conditioning on previously self-generated outputs through KV cache mechanisms to bridge the train-test gap, following Self Forcing [30]. Instead of using static initial tokens as sink tokens in the KV cache, we introduce EMA-Sink, a novel state packaging mechanism for ultra-long video sequences. The core idea of EMA-Sink is to maintain fixed-size tokens initialized from initial frames while continuously updating them by fusing evicted tokens via exponential moving average as they exit the sliding window. Without additional cost, this design not only compresses effective global context to maintain attention performance, but also introduces recent dynamics to prevent over-attending to initial frames. To better distill motion dynamics from teacher models, we introduce Rewarded Distribution Matching Distillation (Re-

DMD). Instead of treating all samples equally as in vanilla distribution matching distillation, Re-DMD is able to distinguish samples with diminished motion dynamics and prioritizes matching with samples exhibiting greater dynamics. To this end, Re-DMD uses a powerful vision-language model as reward function to rate samples according to their motion quality, then uses these scores to weight distribution matching gradients. This effectively biases the distribution matching toward high-quality regions while preserving high data fidelity, leading to enhanced motion dynamics in streaming video generation. Comprehensive experimental evaluation on both short and long video benchmarks demonstrates that Reward Forcing achieves state-of-the-art video quality at 23.1 FPS on a single H100 GPU.

#### 2. Related Works

Autoregressive long video generation. Video diffusion models have advanced short video generation, yet most state-of-the-art models are limited to 5–10 second clips. To reduce the high cost of bidirectional denoising, recent studies have adopted autoregressive diffusion modeling for long video generation [18, 24, 39, 41, 64, 84, 91, 94]. Among these, Pyramidal-flow employs multi-scale flow matching to alleviate computational burden [33], while SkyReels-V2 integrates diffusion forcing [6] with structural planning and multi-modal control [7]. FAR combines short and longterm contexts via flexible positional encoding [20], and MAGI-1 utilizes chunk-wise prediction for scalable autoregressive generation [69]. CausVid reformulates bidirectional diffusion as causal generation through distribution matching distillation [86, 87] to reduce denoising steps [89]. Self-Forcing builds on this framework to mitigate train-test discrepancy by simulating inference conditions [30], which is further extended by LongLive through KV recaching and stream-based fine-tuning for long video generation [82], and by Rolling-Forcing via joint denoising for simultaneous multi-frame processing [45]. However, these methods consistently trade off motion dynamics against visual quality, often introducing cumulative artifacts.

Reinforcement learning for video generation. Reinforcement learning [67] addresses optimizing non-differentiable metrics and temporally extended outcomes, enabling video generation models to better align with human preferences. Research has diverged into two strands. The first develops specialized datasets and reward models [44, 57, 79] for video evaluation. The second integrates RL algorithms into generation pipelines. Some approaches use rewards [56, 90] to directly supervise generative models, while direct preference optimization (DPO) methods [48, 49, 60] implicitly learn preferences from datasets without explicit reward modeling, showing strong robustness [46, 93]. Additionally, policy optimization [92] techniques, such as SelfForcing++ [11], incorporate Flow-GRPO [43] into DMD-

distilled models to improve long-term temporal smoothness. However, this method depends on pre-distilled models, with performance inherently tied to base model.

Here, x0 denotes the output, r represents the reward model, c represents the conditioning input, p and q are distributions, and β acts as the regularization term.

#### 3. Method

(b) Sliding Window w/ Attention Sink (c) Ours (EMA-Sink)

(a) Window Attention

##### 3.1. Preliminaries

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Autoregressive video diffusion models. In autoregressive video diffusion models, an N-frame sequence x1:N follows p(x1:N) = Ni=1 p(xi|x<i). Self Forcing [30] introduces an autoregressive self-rollout mechanism aligning training with inference. During training, each frame xi undergoes iterative denoising conditioned on previously generated clean frames and its noisy state, sampling from the autoregressive distribution. A few-step diffusion model Gθ approximates each conditional p(xi|x<i). Given timesteps {t0,t1,··· ,tT}, denoising at step tj for frame i processes noisy frame xit

sink token

evicted tokens

cached tokens

EMA update

⋯ ⋯ ⋯ ⋯

⋯ ⋯

Figure 2. Comparison of EMA Sink with Existing Methods. Long video generation models typically extrapolate beyond their training sequence length during inference. (a) Window Attention caches only recent tokens for efficient inference but suffers performance degradation. (b) Sliding Window with attention sinks retains initial tokens for stable attention computation and recent tokens for extrapolation. However, discarding intermediate frames causes over-reliance on the first frame, leading to “frame copying” and stiff transitions. (c) EMA Sink preserves full history through exponential moving average (EMA) updates of all historical frames, maintaining stable and consistent performance in long video extrapolation without increasing computational cost.

conditioned on x<i, then reintroduces controlled Gaussian noise via forward process Ψ to yield xit

j

for the next step. The model distribution is: pθ(xi|x<i) = fθ,t

j

(xit

1 ◦ fθ,t

2 ◦ ··· ◦ fθ,t

) where fθ,t

T

T

,tj,x<i),tj−1) and xit

(xit

) = Ψ(Gθ(xit

∼ N(0,I). For longer sequences, LongLive [82] uses sink tokens [78] with p(xi|x1,xi−w+1:i−1) (window size w) to model p(xi|x<i), but over-relies on the initial frame, limiting dynamic variation and smooth transitions. While models can output multi-frame chunks per step [30, 69, 89], we term each chunk a “frame” for simplicity.

j

j

j

T

Distribution matching distillation. DMD [86, 88] distills multi-step diffusion models into a few-step generator G by minimizing reverse KL divergence between real preal(x) and generated distributions pfake(x) across timesteps:

##### 3.2. EMA-Sink: state packaging for long video

Problem formulation. Efficient streaming video generation aims to create indefinitely long videos while maintaining strict temporal and causal consistency. Although sliding window attention is widely adopted in autoregressive models to reduce computational cost, current approaches fail to retain historical context beyond their limited attention windows [89]. As generation progresses, earlier frames are discarded, creating an information bottleneck that diminishes global awareness and leads to temporal inconsistencies and quality drift over time. To address this, we introduce EMASink, a novel state-packaging mechanism that compresses history to support efficient autoregressive generation. Our approach preserves global context in a compact, computationally efficient form throughout the streaming process.

∇θLDMD ≜ Et(∇θDKL(pfake,t(xt)||preal,t(xt))) ≈ −Et (sreal(Ψ(Gθ(ϵ),t),t) − sfake(Ψ(Gθ(ϵ),t),t))

(1)

dGθ(ϵ) dθ

dϵ .

where ϵ ∼ N(0,I), Ψ denotes forward diffusion at timestep t. In diffusion models, the score function is defined as:

xt − αtµreal(xt, t) σt2

, (2)

sreal(xt, t) = ∇xt log preal,t(xt) = −

where µreal is the denoised estimate, and αt,σt are noise schedule parameters [25, 34, 52]. DMD freezes pre-trained µreal (teacher) and updates µfake on generator outputs.

For further illustration, given a noise schedule T = {tj}Tj=0 consisting of distinct noise levels, the model processes each intermediate noisy frame xit

Reinforcement learning. A unified RL [67] fine-tuning objective is established by maximizing the evidence lower bound for optimal video generation x0, culminating in an RL objective that makes an explicit trade-off between reward maximization and fidelity to the original model:

at denoising step tj and frame index i, incorporating earlier clean frames Xi,w = xi−w+1:i−1 where w denotes the window size used during video extrapolation (i > w). It first estimates a denoised version of the frame, then applies the forward diffusion operator Ψ to reintroduce a lower level of Gaussian noise, producing xit

j

- p(x0|c)

- q(x0|c)

r(x0,c) β − log

. (3)

JRL(p,q) = E

for subsequent denoising:

j−1

[Figure 25]

[Figure 26]

Frame 0 Frame n

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Decoder

⋯

⋯

VAE

[Figure 31]

[Figure 32]

Causal Student DiT

Output video

cacheupdate

CleanKV

[Figure 33]

[Figure 34]

[Figure 35]

Next KV cache

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Function

Reward

EMAupdate

Teacher Gradient

𝛼

Add noise

[Figure 40]

[Figure 41]

1 − 𝛼 Updated KV cache

∇𝒥

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Current KV cache

[Figure 47]

[Figure 48]

QKV Proj.

[Figure 49]

A dynamic motocross bike speeding out of a tight turn on a rugged dirt track, its wheels spinning furiously as it gains momentum.

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Real Fake

[Figure 55]

[Figure 56]

[Figure 57]

⋯

⋯

weighted

[Figure 58]

[Figure 59]

[Figure 60]

Text Prompt 𝑐

[Figure 61]

[Figure 62]

Streaming noisy tokens

- Figure 3. Pipeline of Reward Forcing. In a streaming text-to-video generation, noisy tokens in the current stream are first projected to produce new key-value pairs (green blocks), which are appended to the KV cache for attention computation. When the current KV cache reaches its maximum attention window size, sink tokens initialized from start frames (yellow blocks) are updated via exponential moving average using evicted tokens (pink blocks). During training, hallucinated tokens are decoded into videos to compute a reward score via a reward function. This score is then used to weight the distribution matching gradient from the teacher model.

∼ N(0,I). As the window advances to frame i + 1, the oldest frame xi−w+1 is removed from immediate access and is permanently discarded, thereby creating an information bottleneck [30].

,tj,Xiw),tj−1), where xit

handle the spatial-temporal nature of video while maintaining causal relationships, we employ a rotary position embedding (ROPE) [65] when calculating attention. The position encoding is applied causally, ensuring that each position can only attend to previous positions in the sequence.

Ψ(Gθ(xit

j

T

EMA-Sink mechanism. Rather than discarding evicted frames, EMA-Sink maintains compressed global states Si∗ in the KV-cache through an exponential moving average. When frame xi−w is evicted from the sliding window, its key-value pair (Ki−w,V i−w) is continuously fused into the compressed sink states Si∗:

##### 3.3. Rewarded distribution matching distillation

Problem formulation. DMD [86, 88] offers an effective framework for converting multi-step diffusion models into efficient single-step generators by enforcing alignment between the fake and real distributions:

SiK = α · SiK−1 + (1 − α) · Ki−w, (4) SiV = α · SiV−1 + (1 − α) · V i−w. (5)

pfake(x0|c) preal(x0|c)

JDMD = Ep(c)p

. (8)

fake(x0|c) log

Despite its success in preserving sample fidelity, DMD has a fundamental limitation: it treats all regions of the target distribution uniformly, lacking any mechanism to prioritize high-quality outputs according to task-specific metrics. This becomes particularly problematic in video generation, where models progressively produce increasingly static frames during training. This observation motivates a key question: Can we incorporate motion awareness into the distillation process while maintaining distributional fidelity? We address this challenge by integrating RL principles [67] to bias the distillation toward high-reward regions of the output space, thereby generating content with enhanced properties without sacrificing data fidelity.

Here α ∈ (0,1) is the momentum decay factor controlling compression rate, providing smooth temporal compression where recent information dominates while preserving a fading memory of distant history. During attention computation [70], we prepend the compressed sink states to the local window context:

Kiglobal = SiK;Ki−w+1:i , (6) V iglobal = SiV ;V i−w+1:i , (7)

where Ki−w+1:i and V i−w+1:i represent the key and value states from the current sliding window. This formulation allows each query to attend to both the fine-grained local context and the coarse-grained global history, effectively breaking the information bottleneck of the fixed window size. To

Re-DMD mechanism. We introduce Rewarded Distribution Matching Distillation (Re-DMD), which reweights the

Prompt: A kangaroo wearing boxing gloves, sparring with a punching bag in a gym.

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Ours

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Long Live

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Self Forcing

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

CausVid

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

SkyReels V2

t=0s t=10s t=20s t=30s t=40s t=50s t=60s

- Figure 4. Qualitative comparison on dynamic complexity of long video generation. Reward Forcing excels in both text alignment and motion dynamics while baselines exhibit diminished dynamics and weaker alignment.

distribution matching objective according to sample motion quality. Our approach builds on the Reward-Weighted Regression framework [16, 38, 44, 54], which reformulates the reinforcement learning problem as probabilistic inference via the Expectation-Maximization (EM) algorithm [51].

In the E-step [51, 54], we solve Eq. (3) as a constrained optimization problem, obtaining the optimal solution:

1 Z(c)

p(x0|c) =

q(x0|c)exp

r(x0,c) β

, (9)

p(x0|c)exp(r(x

0,c) β ).We assign the

where Z(c) = x

0

distributions in Eq. (3) as q = p′fake and p = p′real. In the M-step [51, 54], we project the nonparametric op-

timal model p = p′real onto the parametric model by maximizing expected log-likelihood Eq. (3) with respect to pfake:

pfake(x0|c) preal(x0|c)

exp(r(x0, c)/β) Z(c)

JRe-DMD = Ep(c)p′

log

.

fake(x0|c)

(10)

Computing the probability density to estimate this loss is generally intractable. However, when training the generator via gradient descent, we only need to obtain the gradient with respect to θ. By differentiating Eq. (10), we obtain:

∇θJRe-DMD =Et ∇θEpc

fake(xt)

exp(rc(xt)/β) Z(c)

pcfake(xt) pcreal(xt)

log

≈ − Et exp(rc(xt)/β) · (sreal(Ψ(Gθ(ϵ), t), t)

dGθ(ϵ) dθ

dϵ .

− sfake(Ψ(Gθ(ϵ), t), t))

(11)

where ϵ is random Gaussian noise, and Gθ is a generator parameterized by θ. sreal and sfake represent the score functions trained on the data and the generator’s output distribution, respectively, using a denoising objective. In addition, rc(xt) is estimated by rc(x0). This approach stabilizes training and accelerates convergence by bypassing the intractable normalization constant and alleviating the need to compute the reward function’s gradient.

##### 3.4. Efficiency analysis

Theoretical properties. The EMA-Sink enables token eviction in O(1) time with low overhead. While attention remains O(w2) in window size, it becomes independent of sequence length. By compressing history into a fixedsize sink, our method achieves constant memory usage relative to sequence length while retaining global context. The differentiable EMA enables gradient propagation through compression, supporting end-to-end learning of compression strategies. The Re-DMD objective implicitly optimizes a constrained reward maximization problem (maximizing expected reward under a distribution matching constraint), ensuring systematic quality improvement without distributional collapse. Notably, our approach avoids typical RL computational costs: the reward serves as a static weighting factor, eliminating backpropagation through reward models and preventing instability from noisy reward gradients.

Real-time long video inference. Long video generation faces quadratic complexity with dense causal attention, hindering real-time synthesis. Local window attention confines complexity to window size, independent of sequence

Prompt: A dramatic underwater photograph captures a man performing an intense drumming session.

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Ours

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Long Live

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Self Forcing

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

CausVid

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

SkyReels V2

t=0s t=10s t=20s t=30s t=40s t=50s t=60s

- Figure 5. Qualitative comparison on long-range temporal consistency. Reward Forcing maintains superior coherence over long-horizon, while baselines suffer from noticeable quality degradation and inconsistency over time.

length. With KV cache scaling by window dimension rather than video length, smaller windows accelerate inference and significantly improve efficiency.

#### 4. Experiments

Implementation details. Reward Forcing is built upon Wan2.1-T2V-1.3B [72] to generate 5-second videos at 832×480 resolution. The model is first trained on 16k ODE solution pairs sampled from the base model, initialized with causal attention masking, following CausVid [89]. Text prompts are drawn from the filtered and LLM-augmented VidProM [73] dataset. We use VideoAlign’s [44] motion quality as the reward function with β = 12. During training, denoising is applied chunk-wise using 3 latent frames per chunk, with denoising steps set to [1000, 750, 500, 250] and an attention window size of 9. Training runs for 600 steps on 64 H200 GPUs with a total batch size of 64 (3˜ hours). The AdamW optimizer is adopted with learning rates of 2.0 × 10−6 for the generator Gθ and 4.0 × 10−7 for the fake score sfake, updating the generator every 5 steps and adjusting the fake score sfake accordingly.

##### 4.1. Comparison with state-of-the-art

Short video generation. We generate 5-second videos using 946 official VBench [31, 32] prompts rewrited using Qwen/Qwen2.5-7B-Instruct [1] following Self Forcing [30], each sampled with 5 different seeds for comprehensive quality assessment. We benchmark our method against relevant open-source video generation models of comparable scale, including LTXVideo [13], Wan2.1 [72],

Table 1. Short video performance comparison with baselines. The comparison includes representative open-source models of comparable scale. Best results in bold, second-best underlined.

VBench evaluation scores ↑ Total Quality Semantic

Model Params FPS↑

Diffusion

LTX-Video [13] 1.9B 8.98 80.00 82.30 70.79 Wan-2.1 [72] 1.3B 0.78 84.26 85.30 80.09

Autoregressive

SkyReels-V2 [7] 1.3B 0.49 82.67 84.70 74.53 MAGI-1 [69] 4.5B 0.19 79.18 82.04 67.74 NOVA [13] 0.6B 0.88 80.12 80.39 79.05 Pyramid Flow [33] 2B 6.7 81.72 84.74 69.62 CausVid [89] 1.3B 17.0 82.88 83.93 78.69 Self Forcing [30] 1.3B 17.0 83.80 84.59 80.64 LongLive [82] 1.3B 20.7 83.22 83.68 81.37 Rolling Forcing [45] 1.3B 17.5 81.22 84.08 69.78

Ours 1.3B 23.1 84.13 84.84 81.32

SkyReels-V2 [7], MAGI-1 [69], CausVid [89], NOVA [13], Pyramid Flow [33], Self Forcing [30], LongLive [82], and Rolling Forcing [45]. The overall score of VBench comprises both quality and semantic components. As shown in Tab. 1, our method achieves an overall score of 84.13 on the 5-second clips, surpassing all existing baselines and demonstrating superior video generation quality. Notably, our approach employs the smallest attention window while achieving the fastest inference speed among all compared

###### Table 2. Long video performance comparison with key baselines. The best results are highlighted in bold.

Qwen3-VL Score ↑ Total Subject Background Smoothness Dynamic Aesthetic Imaging Quality Visual Dynamic Text

VBench Long Evaluation Scores ↑ Drift↓

Model

Diffusion Forcing SkyReels-V2 [7] 75.94 96.43 96.59 98.91 39.86 50.76 58.65 7.315 3.30 3.05 2.70

Distilled Causal

CausVid [89] 77.78 97.92 96.62 98.47 27.55 58.39 67.77 2.906 4.66 3.16 3.32 Self Forcing [30] 79.34 97.10 96.03 98.48 54.94 54.40 67.61 5.075 3.89 3.44 3.11 LongLive [82] 79.53 97.96 96.50 98.79 35.54 57.81 69.91 2.531 4.79 3.81 3.98

Ours 81.41 97.26 96.05 98.88 66.95 57.47 70.06 2.505 4.82 4.18 4.04

methods. Specifically, we attain a real-time generation speed of 23.1 FPS, representing a 47.14× speedup over SkyReels-V2 and a 1.36× speedup over Self Forcing.

Long video generation. Qualitative analysis of the results confirms the effectiveness of our approach. Specifically, Figure 4 illustrates its capability to produce more dynamic sequences for long video generation, and Figure 5 validates its improved temporal consistency. For long video quantitative evaluation, we use the first 128 prompts from MovieGen (consistent with CausVid [89]), extending generation duration to 60 seconds. We employ VBenchLong [32] metrics, including subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality, normalized and weighted using standard VBench [31] coefficients to compute the total score. To quantify drift in long video generation, we compute the standard deviation of imaging quality across 30 segments (2 seconds each) from 60-second videos. As shown in Tab. 2, our method achieves a total score of 81.41, significantly surpassing the state-of-the-art baseline LongLive (79.53). Notably, we observe substantial improvement in the dynamic metric (66.95), representing an 88.38% boost in dynamic amplitude while minimizing quality drift, demonstrating our method’s effectiveness with comparable performance on other metrics. Additionally, we employ Qwen3-VL-235B-A22B-Instruct [1] to evaluate long video generation quality at 55–60 seconds, assessing visual quality, motion dynamics, and text alignment (see more details in the supplements). Each of the 128 videos is scored from 1 to 5, with averaged results showing our model achieves the best performance across all three metrics. We also include a user study for comprehensive comparison in the supplementary material, which demonstrates that our method consistently outperforms all key baselines.

##### 4.2. Ablation studies

Impact of EMA-Sink and Re-DMD. We show the effectiveness of Reward Forcing through qualitative and quantitative comparisons. Qualitatively, as presented in Fig. 6,

Table 3. Ablation studies on key components. The best results for the “Improvement” module are indicated in bold.

VBench Evaluation Scores ↑ Drift↓ Background Smooth Dynamic Quality

Model

Improvement

Ours 95.07 98.82 64.06 70.57 2.51 w/o Re-DMD 95.85 98.91 43.75 71.42 1.77 w/o EMA 95.61 98.64 35.15 70.50 2.65 w/o Sink 94.94 98.56 51.56 69.92 5.08

Impact of α

α = 0.99 95.90 98.96 65.15 70.81 2.52 α = 0.9 95.80 99.09 63.15 71.37 3.23 α = 0.5 94.57 98.89 64.37 71.11 3.78

Impact of β

β = 1 95.14 98.31 54.68 71.73 2.63 β = 2/3 95.02 98.46 60.93 70.61 1.91 β = 1/3 94.94 98.43 58.59 69.29 2.02 β = 1/5 92.40 96.40 94.53 68.26 3.13

our method maintains smooth transitions and high dynamism when generating 850–950 frames (approximately 1 minute), with clearly perceptible fluidity between consecutive frames. Without Re-DMD training, long video generation preserves high consistency with the initial frame and smooth scene transitions, but exhibits significantly reduced dynamism with the dynamic score drops from 64.06 to 43.75 ( Tab. 3). As illustrated in Fig. 6, removing the EMA Sink module results in considerable inconsistency with the first frame and minimal dynamism, reflected quantitatively by declining motion smoothness (98.91 to 98.64 in Tab. 3) and dynamic score (43.75 to 35.15). Ablating the sink token leads to noticeable quality degradation.

Impact of EMA update weight α. An appropriately EMA coefficient α ensures smooth scene transitions in long videos, while a suitable α value effectively balances motion fluidity and temporal consistency. In our implementation, α is set to 9e−3. We can observe from Tab. 3 that α = 0.99

smooth transition, high dynamics

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Ours

Ours

⋯

Self Forcing

smooth transition, low dynamics

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

Long Live

w/o Re-DMD

⋯

stiff transition, low dynamics

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

w/o EMA Sink

Ours

⋯

Long Live

quality drift

Self Forcing

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

w/o Sink Token

⋯

frame=0 frame=850 frame=900 frame=950

- Figure 6. (Left) Ablation study on our proposed module, showing qualitative improvement. (Right) Top: Reward Forcing training leads to a steady rise in the dynamic score. Bottom: The plot of attention size versus FPS underscores the source of our inference efficiency.

0~5s: An empty glass on a rustic wooden table.

achieves a motion smoothness of 98.96 with a corresponding drift of 2.52. Conversely, reducing α to 0.9 improves motion smoothness to 99.09 but increases drift to 3.23.

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Impact of reward weight β. The parameter β modulates the reward term’s influence, with smaller values assigning higher reward weight. As illustrated in Tab. 3, an excessively small β (e.g., 1/5) yields an overly high dynamic score (94.53) at the expense of background consistency (92.40), motion smoothness (96.40), and image quality (68.26). Conversely, an overly large β (e.g., 1) produces an insufficient dynamic score (54.68). Therefore, we select β = 1/2 to optimally balance these metrics.

5~10s: Dark coffee streams into the empty glass.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Figure 7. Interactive video generation. Reward Forcing supports real-time prompt interaction with seamless transitions.

method supports interactive video generation, allowing users to modify prompts during generation to control output content. This is achieved by clearing the previous crossattention cache and recomputing it with the new prompt. Our EMA Sink mechanism ensures seamless prompt transitions while maintaining high temporal consistency.

##### 4.3. Analysis

Dynamic enhancement of Re-DMD. We employ the VBench dynamic score (averaged over the first 128 prompts) to evaluate training effectiveness. An inspection of Fig. 6 reveals that the dynamic score increases steadily with training time while requiring modest computational resources (under 200 GPU hours). Our method surpasses LongLive (high consistency, low dynamism) after only 100 GPU hours and exceeds Self-Forcing (high dynamism, severe drift) after 150 GPU hours. Our model achieves high dynamism while maintaining strong consistency.

#### 5. Conclusion

We presented Reward Forcing, which tackles the problem of motion stagnation in efficient streaming video generation. Our solution is built on two pillars: the EMA-Sink mechanism, which dynamically maintains context to prevent over-dependence on initial frames and ensures longterm coherence, and Re-DMD, which enhances motion dynamics by prioritizing high-reward samples during distillation. Our experiments confirmed that the proposed method achieves state-of-the-art performance on standard benchmarks. By successfully balancing high visual fidelity

Impact of attention window size. The attention window size is a critical factor affecting the speed of real-time generation. Figure 6 demonstrates that inference FPS is inversely proportional to the size of the attention window.

Interactive video generation. As shown in Fig. 7, our

with strong dynamic motion, Reward Forcing enables highquality streaming video generation in real-time. This work establishes a new benchmark for performance and efficiency in generating dynamic, interactive virtual worlds.

#### 6. Acknowledgments

This work was supported by the National Major Science and Technology Projects (the grant number 2022ZD0117000) and the National Natural Science Foundation of China (grant number 62202426). This work was supported by Ant Group Research Intern Program.

#### References

- [1] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 6, 7
- [2] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, Christos Kaplanis, Marjorie Limont, Matt McGill, Yanko Oliveira, Jack Parker-Holder, Frank Perbet, Guy Scully, Jeremy Shar, Stephen Spencer, Omer Tov, Ruben Villegas, Emma Wang, Jessica Yung, Cip Baetu, Jordi Berbel, David Bridson, Jake Bruce, Gavin Buttimore, Sarah Chakera, Bilva Chandra, Paul Collins, Alex Cullum, Bogdan Damoc, Vibha Dasagi, Maxime Gazeau, Charles Gbadamosi, Woohyun Han, Ed Hirst, Ashyana Kachra, Lucie Kerley, Kristian Kjems, Eva Knoepfel, Vika Koriakin, Jessica Lo, Cong Lu, Zeb Mehring, Alex Moufarek, Henna Nandwani, Valeria Oliveira, Fabio Pardo, Jane Park, Andrew Pierson, Ben Poole, Helen Ran, Tim Salimans, Manuel Sanchez, Igor Saprykin, Amy Shen, Sailesh Sidhwani, Duncan Smith, Joe Stanton, Hamish Tomlinson, Dimple Vijaykumar, Luyu Wang, Piers Wingfield, Nat Wong, Keyang Xu, Christopher Yew, Nick Young, Vadim Zubov, Douglas Eck, Dumitru Erhan, Koray Kavukcuoglu, Demis Hassabis, Zoubin Gharamani, Raia Hadsell, A¨aron van den Oord, Inbar Mosseri, Adrian Bolton, Satinder Singh, and Tim Rockt¨aschel. Genie 3: A new frontier for world models. 2025. 2
- [3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 6
- [4] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators.

2024. 2, 6

- [5] Hengyuan Cao, Yutong Feng, Biao Gong, Yijing Tian, Yunhong Lu, Chuang Liu, and Bin Wang. Dimension-reduction attack! video generative models are experts on controllable image synthesis. arXiv preprint arXiv:2505.23325, 2025. 6
- [6] Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024. 2, 6
- [7] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. 2, 6, 7
- [8] Nan Chen, Mengqi Huang, Yihao Meng, and Zhendong Mao. Longanimation: Long animation generation with dynamic global-local memory. arXiv preprint arXiv:2507.01945, 2025. 6
- [9] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023. 6
- [10] Jiale Cheng, Ruiliang Lyu, Xiaotao Gu, Xiao Liu, Jiazheng Xu, Yida Lu, Jiayan Teng, Zhuoyi Yang, Yuxiao Dong, Jie Tang, et al. Vpo: Aligning text-to-video generation models with prompt optimization. arXiv preprint arXiv:2503.20491,

2025. 6

- [11] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 2
- [12] Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang. One-minute video generation with test-time training. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17702–17711, 2025. 6
- [13] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169,

2024. 6

- [14] Xueji Fang, Liyuan Ma, Zhiyang Chen, Mingyuan Zhou, and Guo-jun Qi. Inflvg: Reinforce inference-time consistent long video generation with grpo. arXiv preprint arXiv:2505.17574, 2025. 6
- [15] Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. The matrix: Infinite-horizon world generation with real-time moving control. arXiv preprint arXiv:2412.03568, 2024. 6
- [16] Hiroki Furuta, Heiga Zen, Dale Schuurmans, Aleksandra Faust, Yutaka Matsuo, Percy Liang, and Sherry Yang. Improving dynamic object interactions in text-to-video generation with ai feedback. arXiv preprint arXiv:2412.02617,

2024. 5, 6

- [17] Chongkai Gao, Haozhuo Zhang, Zhixuan Xu, Zhehao Cai, and Lin Shao. Flip: Flow-centric generative planning as general-purpose manipulation world model. arXiv preprint arXiv:2412.08261, 2024. 6
- [18] Jianxiong Gao, Zhaoxi Chen, Xian Liu, Jianfeng Feng, Chenyang Si, Yanwei Fu, Yu Qiao, and Ziwei Liu. Longvie: Multimodal-guided controllable ultra-long video generation. arXiv preprint arXiv:2508.03694, 2025. 2, 6
- [19] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113,

2025. 6

- [20] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Longcontext autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 2
- [21] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. arXiv preprint arXiv:2503.10589, 2025. 6
- [22] Haoran He, Yang Zhang, Liang Lin, Zhongwen Xu, and Ling Pan. Pre-trained video generative models as world simulators. arXiv preprint arXiv:2502.07825, 2025. 6
- [23] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 6

- [24] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2568–2577, 2025. 2, 6
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [26] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 6
- [27] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022. 6
- [28] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 6
- [29] Panwen Hu, Nan Xiao, Feifei Li, Yongquan Chen, and Rui Huang. A reinforcement learning-based automatic video editing method using pre-trained vision-language model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 6441–6450, 2023. 6
- [30] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 2, 3, 4, 6, 7, 1, 5

- [31] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 6, 7, 5
- [32] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, YingCong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench++: Comprehensive and versatile benchmark suite for video generative models. arXiv preprint arXiv:2411.13503, 2024. 6, 7
- [33] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954,

2024. 2, 6

- [34] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022. 3
- [35] Jisoo Kim, Wooseok Seo, Junwan Kim, Seungho Park, Sooyeon Park, and Youngjae Yu. Vip: Iterative online preference distillation for efficient video diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17235–17245, 2025. 6
- [36] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 6
- [37] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 6
- [38] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning textto-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023. 5
- [39] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025. 2
- [40] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 6
- [41] Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350,

2025. 2

- [42] Wenfeng Lin, Renjie Chen, Boyuan Liu, Shiyue Yan, Ruoyu Feng, Jiangchuan Wei, Yichen Zhang, Yimeng Zhou, Chao Feng, Jiao Ran, et al. Contentv: Efficient training of video

- generation models with limited compute. arXiv preprint arXiv:2506.05343, 2025. 6
- [43] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025. 2
- [44] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025. 2, 5, 6
- [45] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025. 2, 6
- [46] Runtao Liu, Haoyu Wu, Ziqiang Zheng, Chen Wei, Yingqing He, Renjie Pi, and Qifeng Chen. Videodpo: Omnipreference alignment for video diffusion generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8009–8019, 2025. 2, 6
- [47] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37:131434–131455, 2024. 6
- [48] Yunhong Lu, Qichao Wang, Hengyuan Cao, Xierui Wang, Xiaoyin Xu, and Min Zhang. Inpo: Inversion preference optimization with reparametrized ddim for efficient diffusion model alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28629–28639, 2025. 2
- [49] Yunhong Lu, Qichao Wang, Hengyuan Cao, Xiaoyin Xu, and Min Zhang. Smoothed preference optimization via renoise inversion for aligning diffusion models with varied human preferences. arXiv preprint arXiv:2506.02698, 2025. 2
- [50] Yintai Ma, Diego Klabjan, and Jean Utke. Video to video generative adversarial network for few-shot learning based on policy gradient. arXiv preprint arXiv:2410.20657, 2024. 6
- [51] T.K. Moon. The expectation-maximization algorithm. IEEE Signal Processing Magazine, 13(6):47–60, 1996. 5
- [52] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 3

- [53] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 1, 6

- [54] Jan Peters and Stefan Schaal. Reinforcement learning by reward-weighted regression for operational space control. In Proceedings of the 24th International Conference on Machine Learning, page 745–750, New York, NY, USA, 2007. Association for Computing Machinery. 5
- [55] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 1

- [56] Mihir Prabhudesai, Russell Mendonca, Zheyang Qin, Katerina Fragkiadaki, and Deepak Pathak. Video diffusion alignment via reward gradients. arXiv preprint arXiv:2407.08737,

2024. 2, 6

- [57] Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072, 2024. 2
- [58] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023. 6
- [59] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 6
- [60] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023. 2
- [61] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 6
- [62] Joonghyuk Shin, Zhengqi Li, Richard Zhang, Jun-Yan Zhu, Jaesik Park, Eli Schechtman, and Xun Huang. Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025. 2
- [63] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792,

2022. 6

- [64] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 2, 6
- [65] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063,

2024. 4

- [66] Yanxiao Sun, Jiafu Wu, Yun Cao, Chengming Xu, Yabiao Wang, Weijian Cao, Donghao Luo, Chengjie Wang, and Yanwei Fu. Swiftvideo: A unified framework for few-step video generation through trajectory-distribution alignment. arXiv preprint arXiv:2508.06082, 2025. 6
- [67] R.S. Sutton and A.G. Barto. Reinforcement learning: An introduction. IEEE Transactions on Neural Networks, 9(5): 1054–1054, 1998. 2, 3, 4
- [68] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025. 6
- [69] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang,

- Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 2, 3, 6
- [70] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4, 6
- [71] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399, 2022. 6
- [72] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 6
- [73] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37: 65618–65642, 2024. 6
- [74] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. International Journal of Computer Vision, 133(5):3059–3078, 2025. 6
- [75] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025. 6
- [76] Thadd¨aus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners. arXiv preprint arXiv:2509.20328, 2025. 2
- [77] Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al. Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025. 6
- [78] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023. 3
- [79] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024. 2, 6
- [80] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025. 6
- [81] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 2
- [82] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive

long video generation. arXiv preprint arXiv:2509.22622,

2025. 2, 3, 6, 7, 1, 5

- [83] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 6
- [84] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649,

2025. 2

- [85] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023. 6
- [86] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 2, 3, 4
- [87] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 2
- [88] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fr´edo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024. 3, 4
- [89] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025. 2, 3, 6, 7, 1, 5
- [90] Hangjie Yuan, Shiwei Zhang, Xiang Wang, Yujie Wei, Tao Feng, Yining Pan, Yingya Zhang, Ziwei Liu, Samuel Albanie, and Dong Ni. Instructvideo: Instructing video diffusion models with human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6463–6474, 2024. 2, 6
- [91] Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, et al. Lumos-1: On autoregressive video generation from a unified model perspective. arXiv preprint arXiv:2507.08801, 2025. 2, 6
- [92] Hui Zhang, Zuxuan Wu, Zhen Xing, Jie Shao, and Yu-Gang Jiang. Adadiff: Adaptive step selection for fast diffusion. arXiv preprint arXiv:2311.14768, 2023. 2
- [93] Jiacheng Zhang, Jie Wu, Weifeng Chen, Yatai Ji, Xuefeng Xiao, Weilin Huang, and Kai Han. Onlinevpo: Align video diffusion model with online video-centric preference optimization. arXiv preprint arXiv:2412.15159, 2024. 2
- [94] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2(3):5, 2025. 2, 6

- [95] Kaifeng Zhao, Gen Li, and Siyu Tang. Dartcontrol: A diffusion-based autoregressive motion model for real-time text-driven motion control. arXiv preprint arXiv:2410.05260, 2024. 6
- [96] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025. 6

## Reward Forcing: Efficient Streaming Video Generation with Rewarded Distribution Matching Distillation

### Supplementary Material

#### S1: More Video Results

Please check the videos in the project page https://reward-forcing.github.io/. These videos are compressed to approximately 40% of their original file size without significant quality degradation.

Comparison with state-of-the-art methods. In addition to Fig. 1, Fig. 4, and Fig. 5 in the main paper, we provide additional video results in our supplementary materials for a more comprehensive evaluation of long videos (approximately 1 minute) generated by different methods. This page includes comparative studies with state-of-the-art methods. We randomly sample prompts from MovieGenBench [55], focusing on Scene Navigation and Object Motion. As demonstrated in the videos, our Reward Forcing preserves high visual fidelity while exhibiting superior motion dynamics over ultra-long horizon, which is crucial for simulating dynamic environments.

Interactive videos. In addition to Fig. 7 in the main paper, we include more interactive video results in the “Reward Forcing.html” page, demonstrating that our Reward Forcing enables user interaction during streaming generation. Specifically, by switching prompts and resetting the cross-attention cache, the model can introduce new events into the ongoing video.

#### S2: User Studies

Experimental setup. To comprehensively evaluate the performance of our proposed method in long video generation, we conducted a user study with 20 participants. Each participant was presented with 20 video groups, where each group contained four videos generated by different methods: CausVid [89], Self-Forcing [30], LongLive [82], and Reward Forcing (ours). The videos were randomly labeled as A, B, C, and D to avoid bias. In total, we collected 1,600 evaluations (20 participants × 20 video groups × 4 videos).

Evaluation protocol. Participants are asked to evaluate each video for three key criteria using a 4-point Likert scale (1-4):

- • Long-Range Temporal Consistency: This metric assesses whether the video maintains visual quality and coherence throughout its entire duration without experiencing visual drift, artifacts, or inconsistencies. Participants evaluated how well each video preserved semantic and structural consistency from start to finish.
- • Dynamic Complexity: This metric measures the naturalness, richness, and engagement of motions and changes in the video. Participants assessed whether the generated content exhibited realistic and diverse dynamics rather than static or repetitive patterns.
- • Overall Preference: This metric captures the holistic quality and appeal of each video, combining factors such as visual fidelity, coherence, motion quality, and subjective viewing experience.

For each criterion, participants assigned scores ranging from:

- • 4 (Good): High quality with no noticeable issues.
- • 3 (Borderline Accept): Acceptable quality with minor issues.
- • 2 (Borderline Reject): Below acceptable quality with noticeable issues
- • 1 (Poor): Unacceptable quality with major issues.

Results and analysis. The user study results unequivocally demonstrate the superiority of our proposed Reward Forcing method over the baseline models across all evaluation criteria (Tab. 4). Our method achieved the highest scores, nearing the “Good” (4) benchmark on the Likert scale, with 3.60 for Temporal Consistency, 3.72 for Dynamic Complexity, and 3.75 for Overall Preference. This indicates that participants consistently rated our videos as high-quality with no noticeable issues. These results validate that Reward Forcing sets a new state-of-the-art for coherent and engaging long video generation.

#### S3: More Quantitative Results and Details

Quality drift. We report the quality drift evaluation results in Tab. 2 and Tab. 3 in the main paper. To quantify the variability in long video imaging quality, we calculate the quality score drift along the temporal horizon using standard deviation,

[Figure 159]

Prompt: A dramatic underwater photograph captures a man performing an intense drumming session. He is submerged in clear blue water, with his face partially obscured by bubbles. His arms move rhythmically, striking the drums with powerful strokes. The drums, made of durable material, are suspended above him, reflecting the vibrant underwater environment. The background features a colorful coral reef with fish swimming around, adding to the vividness of the scene. The water has a soft, ethereal quality, creating a mesmerizing effect. A dynamic low-angle shot from below the surface, emphasizing the man's energetic movements and the aquatic surroundings.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

A B C D

|Reference Values：1=Poor, 2= Borderline Reject, 3= Borderline Accept, 4=Good<br><br>| |A|B|C|D|
|---|---|---|---|---|
|Consistency| | | | |
|Dynamic| | | | |
|Overall| | | | |
|
|---|

Figure 8. User study instruction screenshots.

Table 4. Average User Rating.

Models Temporal Consistency Dynamic Complexity Overall Preference

CausVid [89] 1.81408 1.72676 1.87324 Self Forcing [30] 1.19437 1.75493 1.27042 LongLive [82] 2.78873 2.38310 2.74648

###### Reward Forcing 3.60282 3.72113 3.75493

inspired by Zhang et al. [94]. Each one-minute video is divided into M clips where M = 30, each lasting 2 seconds. For any given long video clip Vi, we compute the drift as follows:

Drift(Vi) =

M

1 M − 1

(si,j − s¯i), (12)

j=1

where Ci,j represent clip j from video i, si,j be the imaging quality score of clip Ci,j. The overall drift across all videos is the mean of individual video drifts:

N

1 N

Drift =

Drift(Vi), (13)

i=1

where N is the total number of videos. Our results show that this metric effectively reflects video quality over long horizons, demonstrating a strong correlation between lower drift scores and more consistent visual fidelity throughout the sequences.

Qwen3-VL evaluation details. We use a powerful vision-language model, Qwen3-VL-235B-A22B-Instruct [81], for a more comprehensive evaluation and report the results in Tab. 2 in the main paper. We include the evaluation template and detailed results for different methods as follows.

###### Full Input Template

[VIDEO] “““Please act as a video quality evaluation expert and rate the given video and text prompt on a scale of 1-5 across the following three dimensions:

**Evaluation Dimensions:**

- 1. **Text Alignment**: Measures the consistency between the video content and the text description.

- - 1: Completely Irrelevant - Content is unrelated or severely contradicts the description.
- - 2: Mostly Mismatched - Only a few minor elements are relevant; the core concept is missing or incorrect.
- - 3: Partially Matched - The core idea is present but with significant deviations or missing key elements.
- - 4: Largely Consistent - Faithfully represents the description with only minor omissions or discrepancies.
- - 5: Perfectly Aligned - Comprehensive and accurate representation of the entire text description.

- 2. **Dynamics**: Evaluates the dynamism and fluidity of the entire scene, including camera movement, object motion, and scene transitions.

- - 1: Static / Disjointed - Little to no dynamic elements; or motion is severely broken and incoherent.
- - 2: Mostly Static - Limited, simple motion; dynamics feel stiff, mechanical, or poorly executed.
- - 3: Moderately Dynamic - Basic movement is present but lacks fluidity and natural flow; may appear robotic.
- - 4: Largely Dynamic - Generally fluid and engaging motion with a good sense of flow; minor imperfections may exist.
- - 5: Highly Dynamic - Exceptionally smooth, natural, and purposeful motion that enhances the visual narrative.

- 3. **Visual Quality**: Assesses the technical execution, including clarity, color grading, composition, and the absence of artifacts.

- - 1: Very Poor - Severely blurry, heavy visual artifacts (e.g., distortion, tearing), and/or extreme color issues (e.g., over-saturation, color banding).
- - 2: Poor - Consistently blurry, noticeable noise, unnatural color palette, or frequent minor artifacts.
- - 3: Fair - Passable clarity and color, but with visible technical flaws; composition may be unremarkable.
- - 4: Good - Clear and mostly sharp, with natural and balanced colors; good composition and only minor, infrequent issues.
- - 5: Excellent - Technically superior: sharp, well-composed, with vivid yet natural colors, and free from visible artifacts or distortions.

**Scoring Requirements:**

- - Please output strictly in the following format, only numbers and brief reasons: Text Alignment: [1-5] Reason: [brief explanation] Dynamics: [1-5] Reason: [brief explanation] Visual Quality: [1-5] Reason: [brief explanation]

Now please evaluate the following content: Text Prompt: ”” Video Content: Please carefully watch the provided video ”””

###### CausVid Full Results

Average Scores by Dimension: Text Alignment: 3.32 Dynamics: 3.16 Visual Quality: 4.66

Detailed Scores by Dimension: Text Alignment: [5, 1, 2, 4, 5, 5, 5, 5, 5, 1, 3, 4, 1, 1, 4, 2, 4, 4, 2, 2, 3, 3, 2, 1, 5, 3, 2, 5, 3, 4, 5, 4, 2, 3, 2, 1, 3, 3, 2, 3, 5, 3, 1, 4, 5, 2, 2, 3, 2, 4, 5,

- 5, 3, 5, 3, 5, 3, 5, 5, 5, 4, 2, 5, 5, 5, 5, 2, 5, 5, 5, 3, 2, 2, 5, 1, 3, 2, 4, 2, 2, 3, 2, 3, 3, 2, 1, 3, 2, 1, 2, 5, 3, 4, 4, 5, 3, 3, 5, 5, 1, 2, 4, 5, 5, 4, 3, 2, 5, 2, 5, 5, 3, 5, 4, 2, 5, 4, 2, 5, 5, 2, 1, 2, 5, 1, 3, 1, 5]

- Dynamics: [4, 3, 3, 5, 4, 3, 4, 4, 3, 1, 4, 4, 2, 2, 4, 2, 4, 3, 4, 3, 4, 4, 3, 4, 3, 2, 3, 4, 2, 3, 4, 3, 3, 4, 2, 3, 4, 2, 2, 3, 4, 3, 3, 4, 3, 2, 3, 2, 4, 3, 3, 4, 4,

- 3, 4, 4, 3, 3, 4, 3, 4, 4, 4, 1, 3, 4, 1, 5, 3, 4, 4, 2, 3, 4, 3, 2, 3, 2, 4, 4, 5, 2, 3, 2, 2, 1, 4, 3, 4, 3, 3, 3, 2, 3, 4, 3, 2, 4, 4, 1, 1, 3, 4, 4, 4, 2, 4, 3, 3, 3, 3,

- 2, 4, 5, 3, 4, 4, 3, 3, 3, 3, 4, 2, 4, 2, 2, 3, 3] Visual Quality: [5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 4, 5, 4, 5, 5, 4, 3, 5, 5, 4, 5, 5, 4, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,

- 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 4, 4, 5, 4, 4, 4, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 4, 5, 5, 5, 4, 4, 4, 4, 5,

- 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 4, 5]

###### SkyReels Full Results

Average Scores by Dimension:

- Text Alignment: 2.70 Dynamics: 3.05

- Visual Quality: 3.30

Detailed Scores by Dimension:

- Text Alignment: [4, 2, 3, 2, 5, 5, 4, 2, 4, 1, 5, 3, 1, 2, 3, 1, 5, 2, 3, 1, 1, 1, 4, 1, 1, 2, 2, 1, 3, 4, 4, 1, 3, 3, 3, 1, 2, 5, 2, 2, 2, 2, 1, 4, 1, 1, 1, 2, 2, 2, 4,

3, 2, 5, 4, 5, 2, 5, 1, 5, 2, 2, 5, 2, 1, 5, 2, 4, 5, 3, 4, 2, 3, 3, 2, 4, 3, 1, 2, 2, 1, 3, 1, 3, 1, 5, 5, 1, 1, 3, 4, 4, 2, 1, 5, 4, 2, 3, 1, 4, 3, 1, 3, 2, 5, 3, 1, 1, 1, 5, 5, 1, 3, 3, 2, 3, 4, 1, 3, 5, 3, 2, 4, 1, 1, 5, 1, 5] Dynamics: [5, 3, 1, 2, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3, 4, 3, 5, 4, 4, 2, 3, 2, 4, 1, 3, 3, 4, 1, 2, 3, 3, 3, 2, 4, 4, 3, 3, 4, 3, 3, 2, 1, 3, 4, 4, 1, 2, 4, 3, 3, 5, 4, 2, 4, 3, 4, 3, 3, 2, 3, 3, 3, 4, 1, 2, 4, 3, 5, 3, 2, 4, 4, 4, 3, 3, 4, 3, 2, 2, 3, 1, 4, 3, 2, 2, 5, 4, 2, 1, 3, 3, 3, 2, 1, 3, 3, 3, 4, 3, 4, 3, 1, 4, 3, 5, 2, 1, 3, 1, 3, 4,

- 3, 4, 4, 2, 3, 4, 4, 2, 4, 4, 4, 5, 2, 4, 4, 3, 3]

- Visual Quality: [4, 4, 2, 1, 4, 5, 3, 4, 3, 4, 5, 4, 4, 3, 4, 3, 5, 5, 3, 1, 3, 1, 4, 1, 3, 5, 4, 1, 4, 5, 4, 3, 5, 2, 4, 4, 3, 5, 4, 2, 3, 2, 3, 3, 4, 1, 3, 4, 3, 4, 4, 3,

- 2, 4, 3, 4, 3, 3, 3, 3, 2, 4, 3, 1, 3, 5, 3, 4, 3, 1, 4, 4, 4, 2, 3, 3, 4, 1, 3, 1, 3, 3, 2, 4, 4, 5, 3, 1, 2, 4, 4, 5, 3, 1, 4, 4, 4, 4, 2, 4, 3, 2, 4, 4, 5, 2, 3, 3, 1, 4,

4, 4, 3, 2, 2, 4, 3, 4, 2, 4, 3, 5, 4, 4, 5, 5, 4, 5]

Self Forcing Full Results

Average Scores by Dimension: Text Alignment: 3.11 Dynamics: 3.44

- Visual Quality: 3.89 Detailed Scores by Dimension:

- Text Alignment: [4, 1, 2, 2, 5, 3, 5, 3, 5, 1, 4, 3, 1, 1, 5, 3, 2, 2, 3, 3, 1, 2, 2, 1, 5, 4, 2, 5, 4, 5, 3, 2, 3, 3, 2, 5, 1, 3, 3, 1, 4, 3, 1, 2, 1, 2, 1, 4, 4, 3, 1,

5, 5, 5, 1, 5, 4, 3, 3, 2, 4, 2, 5, 5, 2, 5, 3, 5, 5, 3, 4, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 5, 5, 4, 2, 1, 2, 5, 4, 2, 5, 5, 3, 2, 5, 1, 1, 3, 1, 5, 5, 3, 4, 2, 5, 2, 5, 4, 3, 3, 3, 3, 5, 2, 2, 5, 5, 3, 3, 4, 5, 3, 2, 1, 5]

- Dynamics: [4, 3, 3, 2, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 4, 3, 4, 4, 4, 4, 3, 4, 3, 1, 4, 3, 3, 4, 3, 4, 3, 4, 2, 4, 4, 5, 3, 3, 3, 3, 4, 2, 3, 2, 4, 4, 3, 4, 3, 3, 4, 4, 2,

3, 2, 4, 4, 3, 4, 3, 4, 4, 5, 3, 4, 4, 4, 5, 3, 3, 5, 3, 4, 4, 4, 4, 4, 2, 4, 4, 5, 3, 3, 2, 5, 5, 4, 4, 4, 4, 3, 3, 2, 4, 3, 3, 3, 4, 2, 2, 4, 1, 4, 4, 4, 3, 5, 3, 4, 3, 3, 4, 4, 4, 4, 3, 4, 3, 3, 3, 4, 4, 4, 4, 4, 3, 2, 3] Visual Quality: [5, 4, 4, 2, 4, 4, 5, 4, 4, 4, 5, 3, 2, 3, 5, 4, 5, 4, 3, 4, 4, 3, 4, 2, 5, 4, 4, 4, 5, 5, 4, 4, 3, 3, 3, 5, 4, 4, 3, 2, 5, 4, 4, 3, 5, 3, 4, 4, 4, 3, 4, 5, 4, 4, 3, 3, 5, 4, 4, 3, 4, 5, 5, 4, 3, 4, 3, 5, 4, 4, 5, 4, 3, 4, 5, 5, 4, 1, 4, 3, 4, 2, 3, 4, 5, 5, 5, 4, 4, 5, 4, 5, 4, 4, 3, 3, 4, 4, 3, 4, 3, 1, 4, 4, 5, 5, 3, 3, 4, 3, 3, 4, 4, 5, 4, 5, 3, 3, 5, 4, 3, 5, 5, 4, 5, 4, 4, 4]

LongLive Full Results

Average Scores by Dimension:

- Text Alignment: 3.98

- Dynamics: 3.81

Visual Quality: 4.79

Detailed Scores by Dimension: Text Alignment: [5, 2, 4, 5, 5, 5, 5, 5, 5, 1, 5, 3, 1, 2, 5, 4, 4, 4, 3, 5, 3, 3, 3, 4, 5, 5, 3, 5, 3, 5, 5, 4, 4, 3, 3, 2, 5, 5, 2, 3, 5, 3, 1, 5, 5, 5, 4, 5, 4, 1, 5, 5, 5, 5, 3, 5, 4, 5, 5, 5, 5, 2, 5, 5, 5, 3, 2, 5, 5, 5, 4, 1, 5, 5, 2, 5, 5, 5, 4, 4, 1, 4, 4, 2, 3, 5, 5, 3, 1, 5, 5, 3, 3, 5, 5, 4, 2, 3, 5, 3, 3, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 5, 4, 5, 4, 3, 5, 5, 5, 2, 4, 5, 1, 3, 1, 5]

- Dynamics: [5, 4, 3, 4, 4, 4, 3, 4, 3, 4, 4, 3, 4, 3, 4, 3, 5, 5, 4, 5, 3, 4, 4, 3, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 3, 4, 4, 3, 4, 4, 2, 3, 5, 4, 4, 5, 5, 4, 3, 4, 4, 1,

- 3, 4, 4, 4, 3, 4, 3, 5, 4, 4, 1, 4, 4, 4, 4, 3, 3, 4, 3, 5, 5, 4, 5, 4, 4, 5, 5, 4, 4, 3, 4, 2, 5, 4, 4, 4, 5, 3, 3, 2, 3, 3, 3, 2, 3, 4, 3, 4, 4, 4, 4, 5, 5, 5, 3, 4, 3, 4, 5, 5, 5, 5, 3, 5, 4, 4, 4, 5, 4, 5, 3, 4, 2, 4, 3]

- Visual Quality: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5,

- 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 4, 3, 5, 5, 5, 5, 5, 4, 4, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5]

###### Reward Forcing Full Results

Average Scores by Dimension:

- Text Alignment: 4.04

- Dynamics: 4.18

- Visual Quality: 4.82

Detailed Scores by Dimension:

- Text Alignment: [5, 2, 4, 2, 5, 5, 5, 5, 3, 1, 5, 5, 1, 1, 5, 5, 5, 4, 5, 5, 3, 5, 5, 4, 5, 5, 4, 5, 3, 5, 5, 4, 5, 4, 3, 2, 4, 4, 1, 3, 5, 3, 1, 5, 5, 5, 2, 5, 3, 2, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 5, 2, 5, 5, 5, 2, 2, 5, 5, 5, 4, 1, 5, 5, 3, 5, 5, 5, 3, 4, 2, 4, 3, 4, 4, 5, 4, 5, 1, 5, 5, 4, 3, 5, 5, 4, 3, 5, 5, 2, 5, 4, 5, 5, 5, 4, 4, 5, 3,

- 5, 4, 5, 5, 2, 4, 5, 5, 3, 4, 5, 4, 2, 5, 5, 2, 5, 2, 5] Dynamics: [5, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 4, 5, 3, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 3, 4, 4, 3, 4, 5, 5, 4, 4, 5, 4, 4, 4, 5, 4,

4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 4, 4, 4, 3, 5, 4, 4, 5, 4, 5, 5, 4, 4, 5, 5, 4, 5, 4, 4, 4, 4, 3, 5, 4, 5, 4, 5, 4, 3, 3, 4, 4, 3, 4, 4, 4, 3, 5, 4, 4, 5, 5, 5, 5, 4, 4, 4, 3, 5, 5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 5, 4, 4, 4, 4, 4] Visual Quality: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5,

- 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 3, 5, 5, 5, 5, 5, 4, 4, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5]

Detailed results on VBench [31]. We report quantitative evaluation on VBench [31] using the extended prompts in Tab. 1 in the main paper. Specifically, the Quality Score is a weighted average of the following dimensions: subject consistency, background consistency, temporal flickering, motion smoothness, aesthetic quality, imaging quality, and dynamic degree. The Semantic Score is a weighted average of the following dimensions: object class, multiple objects, human action, color, spatial relationship, scene, appearance style, temporal style, and overall consistency. Each dimension’s results are normalized using the following formula: normalized score = (score - min) / (max - min). The normalization range (minimum and maximum) for each dimension and the assigned weights used to compute the weighted average are provided in the Tab. 5.

Table 5. Normalization ranges and weighting coefficients of VBench score.

Subject Background Temporal Motion Dynamic Aesthetic Imaging Overall Consistency Consistency Flickering Smoothness Degree Quality Quality Consistency min 0.1462 0.2615 0.6293 0.706 0.0 0.0 0.0 0.0

- max 1.0 1.0 1.0 0.9975 1.0 1.0 1.0 0.364 weighting coefficients 1 1 1 1 0.5 1 1 1

Object Multiple Human

Color

Spatial

Scene

Temporal Appearance Class Objects Action Relationship Style Style

min 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0009

- max 1.0 1.0 1.0 1.0 1.0 0.8222 0.364 0.2855 weighting coefficients 1 1 1 1 1 1 1 1

In addition, we provide detailed evaluation in Tabs. 6 and 7. Our method achieves a total score of 84.13 on the short video generation task using the extended VBench prompts, consistently surpassing current state-of-the-art baselines and demonstrating its effectiveness.

Table 6. Quality evaluation on extended VBench.

Subject Background Temporal Motion Dynamic Aesthetic Imaging Quality Total Consistency Consistency Flickering Smoothness Degree Quality Quality Score Score

Model

CausVid [89] 96.33 95.84 99.44 97.98 61.11 64.52 67.96 83.93 82.88 Self Forcing [30] 95.09 96.10 99.01 98.24 66.38 65.79 69.71 84.59 83.80 LongLive [82] 96.98 96.92 99.35 98.79 40.83 67.03 69.18 83.68 83.22

Reward Forcing 95.43 96.59 98.97 98.32 68.05 65.66 69.38 84.84 84.13

###### Table 7. Semantic evaluation on extended VBench.

Object Multiple Human

Temporal Appearance Overall Semantic Total Class Objects Action Relationship Style Style Consistency Score Score

Spatial

Model

Color

Scene

CausVid [89] 92.78 88.32 96.20 86.67 74.05 51.35 23.95 20.19 25.95 78.69 82.88 Self Forcing [30] 93.16 87.19 96.40 86.83 81.77 56.13 24.45 20.34 26.85 80.64 83.80 LongLive [82] 96.28 86.49 95.80 90.79 80.56 58.79 24.16 20.42 26.61 81.37 83.22

Reward Forcing 94.81 86.79 96.80 89.42 82.47 57.19 24.33 20.38 26.88 81.32 84.13

#### S4: More Implementation details

Noise schedule and model parameterization. Building upon the Wan2.1 and Self Forcing, our approach utilizes the flow matching framework. We implement a time step shift defined as t′(k,t) = (kt/1000)/(1 + (k − 1)(t/1000)) · 1000 with a shift factor k set to 5. In the forward process, a sample is generated according to xt = t

′

′

1000x + 1−t

1000ϵ, where ϵ is drawn from a standard normal distribution N(0,I) and t ranges from 0 to 1000. The data prediction model is formulated as :

Gθ(x,t,c) = cskip · ϵ − cout · vθ(cin · xt,cnoise(t′),c). (14)

The preconditioning coefficients remain consistent with the base models’ settings: specifically, cskip, cin, cout are all 1, and cnoise(t) = t. For our few-step diffusion sampling, we adopt a uniform 4-step schedule with time steps t1,t2,t3,t4 =

1000,750,500,250 .

#### S5: Further Related Works

Video diffusion models. Video diffusion models [3, 23, 26, 27] have evolved from UNet [61] backbones to Diffusion Transformers (DiTs) [53]. Early approaches extended image diffusion models temporally [63] but lacked scalability. DiT’s Transformer blocks enhance spatio-temporal modeling, enabling models like Sora [4] and Hunyuan-Video [37] to generate realistic, coherent videos. Hunyuan-Video integrates causal 3D VAE [36] and language models for textual control. Open-Sora [40] advanced efficiency and realism, while Wan 2.1 [72] validated large-scale pre-training benefits and CogVideoX [28, 83] improved alignment via adaptive LayerNorm. For long video generation, Phenaki [71] uses discrete tokens, LVDM [23] employs hierarchical 3D latent generation, and NUWA-XL [85] uses coarse-to-fine processing. LaVie [74] integrates rotary encoding and temporal attention, SEINE [9] enables smooth transitions via stochastic masking, and LCT [21] extends to multi-shot generation. Diffusion forcing [6] combines diffusion quality with autoregressive efficiency. StreamingT2V [24] adds memory modules, History-guided video [64] uses historical context, FramePack [94] compresses frames, Lumos-1 [91] integrates LLM-like architecture, and LongVie [18] introduces multi-modal guidance and degradation-aware training. Testtime training methods [12] generate minute-long videos but incur high costs. Training-free methods—RIFLEx [96] adjusting positional embeddings, FreeNoise [58] combining noise rescheduling with windowed attention, and FreeLong [47] integrating multi-frequency information.

Reinforcement learning for video models. Video generative models [5, 8, 35, 59, 66, 68, 70, 77] using MLE or reconstruction loss misalign with human preferences. RL enables direct optimization of preference-aligned objectives [19, 42]. Direct Preference Optimization (DPO) [16] dominates post-training alignment, including VideoDPO [46] for temporal consistency, VisionReward [79] for multi-objective preferences, and variants with physics-based generation. Group Relative Policy Optimization (GRPO), extending PPO [29, 50, 95], improves generalization as shown in DanceGRPO [80]. Rewardbased approaches like InstructVideo [90] with pretrained reward feedback and VADER [56] with unified differentiable rewards bypass policy learning. Inference-time methods like InfLVG [14] incorporate GRPO for dynamic long-form optimization. Collectively, RL serves as both post-training alignment and structural component for preference-aware generation [10, 15, 17, 22, 75], bridging surrogate objectives and human-valued quality.

#### S6: Discussion and Future Work

Generalizability. Our method is designed to be general-purpose and plug-and-play, enabling seamless integration with various video generation architectures without requiring substantial modifications to existing pipelines. This flexibility represents a significant practical advantage, as it allows researchers and practitioners to adopt our approach with minimal overhead.

Misalignment between reward functions and evaluation criteria. The first factor contributing to inconsistent performance is the misalignment between our reward function’s optimization direction and VBench’s evaluation criteria. VBench employs comprehensive metrics including temporal consistency, motion smoothness, subject consistency, background quality, aesthetics, and semantic alignment. Our reward model may prioritize certain dimensions over others—for example, heavily weighting temporal coherence while underemphasizing aesthetic qualities. This asymmetric optimization creates scenarios where reward improvements don’t translate proportionally to VBench score gains.

Video reward models. Our experiments show that current reward models can effectively guide quality improvements, as reflected in our competitive performance across multiple benchmarks. However, video reward models still face challenges in capturing certain nuanced aspects of video quality, such as long-range temporal dependencies, subtle temporal artifacts like frame jitter, and complex semantic attributes. These models are typically trained on datasets with subjective annotations that may not fully represent all quality dimensions. As video reward models continue to advance, our framework will naturally benefit from these improvements, enabling further optimization.

Future research directions. Future research should develop more sophisticated reward models capturing video quality nuances. Promising directions include: multi-objective reward modeling with separate components for different quality dimensions; hierarchical models assessing quality at multiple temporal scales; human-in-the-loop feedback mechanisms grounding models in perceptual judgments; domain-adaptive models adjusting criteria by content type; and architectures encoding physical and semantic priors about real-world dynamics. Advancing reward modeling along these dimensions could help our method achieve its full potential and demonstrate substantial, consistent improvements across comprehensive evaluation frameworks.

#### S7: Border Social Impact

This work on efficient streaming video generation presents both significant opportunities and risks. On the positive side, the reduced computational requirements could democratize access to video synthesis technology, benefiting educational content creators, small organizations, and researchers with limited resources. The improved efficiency also reduces energy consumption, contributing to more sustainable AI development. However, we acknowledge serious concerns regarding potential misuse. The accessibility and speed of our method lowers barriers for creating deepfakes and misleading visual content that could spread misinformation or enable identity fraud. Additionally, our reward-based prioritization of dynamic content may inadvertently amplify biases present in vision-language models, potentially marginalizing underrepresented groups or activities. Questions of copyright infringement and consent regarding training data and generated likenesses also warrant careful consideration. To mitigate these risks, we strongly advocate for implementing digital watermarking and provenance tracking in any deployment of this technology. We encourage development of detection tools for synthetic content, clear content labeling practices, and robust usage policies prohibiting malicious applications such as non-consensual deepfakes. We support collaborative efforts among researchers, policymakers, and civil society to establish ethical guidelines, legal frameworks, and media literacy initiatives. Effective governance requires not only technological safeguards but also transparent data practices, diverse evaluation metrics to reduce bias, and ongoing dialogue about responsible use of generative video technologies.

