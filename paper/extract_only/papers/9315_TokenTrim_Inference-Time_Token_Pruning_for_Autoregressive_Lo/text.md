# arXiv:2602.00268v1[cs.CV]30Jan2026

## TOKENTRIM: INFERENCE-TIME TOKEN PRUNING FOR AUTOREGRESSIVE LONG VIDEO GENERATION

Ariel Shaulov School of Computer Science Tel Aviv University, Israel arielshaulov@mail.tau.ac.il

Eitan Shaar Independent Researcher shaarei@biu.ac.il

Amit Edenzon School of Mathematics Bar-Ilan University, Israel amit.edenzon@live.biu.ac.il

Lior Wolf School of Computer Science Tel Aviv University, Israel wolf@cs.tau.ac.il

### TokenTrim Project Page

+TokenTrimRollingForcing

[Figure 1]

[Figure 2]

(a) “Aerial tracking camera shot. A white semi-truck drives on a highway”

[Figure 3]

###### +TokenTrimSelfForcing

[Figure 4]

(b) “Two ibexes navigating a rocky hillside”

Figure 1: Text-to-video results before and after applying TokenTrim on Rolling Forcing [18] and Self Forcing [19].

### ABSTRACT

Auto-regressive video generation enables long video synthesis by iteratively conditioning each new batch of frames on previously generated content. However, recent work has shown that such pipelines suffer from severe temporal drift, where errors accumulate and amplify over long horizons. We hypothesize that this drift does not primarily stem from insufficient model capacity, but rather from inference-time error propagation. Specifically, we contend that drift arises from the uncontrolled reuse

of corrupted latent conditioning tokens during auto-regressive inference. To correct this accumulation of errors, we propose a simple, inference-time method that mitigates temporal drift by identifying and removing unstable latent tokens before they are reused for conditioning. For this purpose, we define unstable tokens as latent tokens whose representations deviate significantly from those of the previously generated batch, indicating potential corruption or semantic drift. By explicitly removing corrupted latent tokens from the auto-regressive context, rather than modifying entire spatial regions or model parameters, our method prevents unreliable latent information from influencing future generation steps. As a result, it significantly improves long-horizon temporal consistency without modifying the model architecture, training procedure, or leaving latent space.

### 1 Introduction

Video diffusion models have rapidly advanced, enabling text-conditioned generation of high-quality videos with realistic appearance and expressive motion [37, 11, 9, 38, 12, 30, 17]. Despite this progress, generating long videos that remain temporally coherent is still a major open challenge [32, 19, 18, 36]. The dominant recipe for long-horizon synthesis is chunk-wise autoregressive generation: the model produces a short clip, then extends the sequence by generating the next clip while conditioning on latent representations of previously generated frames [32, 35, 19, 18]. While effective for extending duration, this autoregressive loop often exhibits temporal drift (error accumulation): small artifacts and inconsistencies introduced early can compound across chunks, leading to identity changes, structural degradation, and loss of global coherence [19, 18, 36].

Recent work has proposed to mitigate drift by strengthening long-range conditioning and stabilizing the autoregressive mechanism, e.g., through temporal key-value (KV) caching [19, 35], anchor/sink tokens [18, 36], and cache refresh or management strategies during training [36, 35]. However, long rollouts remain brittle at inference time [18, 32]. A key reason is that the conditioning context itself degrades: once a region of the latent state becomes corrupted, it is repeatedly reused in subsequent steps and can dominate attention, effectively propagating errors forward [18]. This suggests that temporal drift is not only a modeling or data issue, but also an inference-time information propagation problem: the system lacks a mechanism to assess which cached latent tokens are trustworthy.

In this work, we introduce TokenTrim, an inference-time method for identifying unstable latent tokens before reuse and removing them from the conditioning context. TokenTrim operates entirely in latent space, requires no architectural changes or retraining, and adds only negligible overhead. At each autoregressive step, our method estimates per-token drift by comparing a compact summary of the previous chunk’s latent state to the current chunk at its first denoising iteration. Tokens with high drift are hard-pruned from the cached context, preventing corrupted regions from influencing future generations.

TokenTrim is compatible with autoregressive video diffusion frameworks that rely on self-attention with KV caching, including Self Forcing [19] and Rolling Forcing [18]. It is also compatible with inference-time regularization techniques such as FlowMo [20], which can be applied independently as an additional motion-consistency term. By pruning high-drift cached tokens and retaining only reliable context for reuse, TokenTrim suppresses error amplification across autoregressive steps and improves long-horizon temporal consistency. Our results show that controlling the conditioning context at inference time alone can substantially reduce temporal drift, highlighting an underexplored mechanism for long video generation [18, 32].

In summary, we propose TokenTrim, a fully inference-time drift detection and hard-pruning mechanism that operates in latent space. We further show that combining motion-stabilized initialization with selective latent pruning significantly improves long-video coherence (see Fig. 1), without retraining or modifying the underlying model.

### 2 Related Work

##### 2.1 Auto-regressive Long Video Generation

The transition from short-clip synthesis [9, 11, 12, 10, 30, 44, 45] to long-horizon video generation [17, 14, 31, 13, 16, 41, 7, 4, 46] has necessitated a shift from holistic spatiotemporal modeling to causal, auto-regressive architectures that factorize generation into sequential chunks. Early approaches like NUWA-XL [7] employed a hierarchical "coarse-tofine" strategy, generating global key frames before filling local gaps. However, as this precludes real-time streaming, contemporary state-of-the-art methods favor continuous auto-regressive modeling, although often struggling with the “train-test discrepancy” or “exposure bias”. Self Forcing [19] addresses this by training on self-generated roll-outs with stochastic gradient truncation, ensuring the model learns to recover from its own inference artifacts. To bypass the computational cost of bidirectional attention in streaming contexts, CausVid [6] distills a bidirectional teacher into

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

2

- Figure 2: TokenTrim overview at autoregressive step t. (a) Given the candidate batch Xt and the previous batch Xt−1, we encode each frame and form latent summaries Zt and Zt−1 by averaging latents over the F frames in each batch. We compute per-token drift di = ∥Zt(i) − Zt−1(i)∥2 and select the top-pN largest drifts to form the unstable set St, from which we compute the drift severity Dt. (b) We compare Dt to the adaptive threshold µt +λσt. If Dt ≤ µt +λσt, the KV cache (K,V ) is left unchanged and the batch is accepted. Otherwise, we mask the selected token positions in the temporal KV cache to obtain (K,˜ V˜) and regenerate the current batch conditioned on the pruned cache. Running statistics and the cache are updated using the accepted batch Xt⋆.

a block-causal student, enabling efficient frame-by-frame generation. Addressing the stability of infinite generation, Rolling Forcing [18] introduces "relaxed causality" via rolling-window joint denoising and an attention sink mechanism to anchor identity. Similarly, LongLive [36] incorporates a KV re-cache mechanism to handle interactive prompt changes without breaking temporal continuity. These training-based methods often require substantial computational resources to realign the model’s internal distribution with the auto-regressive task. However, these training-based methods often suffer from substantial computational costs to realign the model’s distribution. More critically, they remain inherently vulnerable to snowballing errors. Even with extensive tuning, minor artifacts inevitably persist in the context and compound into severe temporal drift. In contrast, our approach overcomes this by actively detecting and pruning unstable latent tokens, offering the distinct additional benefit of suppressing error propagation at the source to sustain long video consistency without the need for expensive retraining.

##### 2.2 Inference-time guidance

Inference-time guidance offers a training-free paradigm to enhance generation quality by modifying the sampling dynamics or context. To improve temporal consistency, FreeInit [3] and FreeLong [8] leverage spectral analysis, iteratively refining the initial noise distribution to align low-frequency components with the training manifold, thereby stabilizing global structure. For motion coherence, Shaulov et al. [20] introduces a latent-optimization approach, calculating the patch-wise temporal variance across generated frames and applying gradient updates to minimize incoherent motion trajectories. Other methods rely on explicit conditioning anchors. For instance ConsistI2V [25] modifies spatiotemporal attention to attend to the high-frequency details of the first frame, preventing identity degradation. More recently, focus has shifted to managing the Key-Value cache during inference. While TeaCache [34] and TaoCache [5] prune tokens primarily for acceleration. This demonstrates that selective context management can serve as a powerful form of guidance, a direction our work advances by pruning based on latent instability.

##### 2.3 Error-Accumulation In Long Video Generation

A pervasive failure mode in auto-regressive video generation is temporal drift [41], where minor errors in early frames accumulate linearly or exponentially, leading to "error-accumulation" and semantic collapse. Theoretical analysis attributes this to exposure bias, i.e., the distributional shift between the ground truth history seen during training and the imperfect self-generated history at inference [18, 19, 40]. Empirical studies typically quantify this drift via metrics

+TokenTrimRollingForcing

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

(a) “Pikachu hopping across rocks near a mountain stream.”

+TokenTrimRollingForcing

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

(b) “A woman dancing hip hop, street dancing in the studio.”

[Figure 23]

[Figure 24]

###### +TokenTrimSelfForcing

[Figure 25]

[Figure 26]

(c) “A young girl skips down a quiet suburban street lined with trees.”

[Figure 27]

[Figure 28]

###### +TokenTrimSelfForcing

[Figure 29]

[Figure 30]

(d) “A brown bear walks in a grassy ﬁeld.”

- Figure 3: Qualitative results. Text-to-video results before and after applying TokenTrim on Rolling Forcing [18] and Self Forcing [19]. TokenTrim mitigates degradation over time, e.g., color shifts (c - background and girl, d - background and bear), artifacts (b - light lens flare) and unnatural motion (a - Pikatchu). For additional qualitative results see App. D and App. E.

such as Fréchet Video Distance (FVD) [4, 1, 31] which measures The overall visual quality and temporal coherence of the generated videos or the ∆ Quality Drift introduced in Rolling Forcing [18]. To mitigate drift, architectures like StreamingT2V [42] employ specialized long-term memory modules (Appearance Preservation Module) to reinject features from an anchor frame. Bagger [43] proposes a self-supervised training scheme that aggregates backward trajectories to correct drift. However, these methods often rely on rigid anchoring or extensive retraining. Recent findings in FreeLong [8] suggest that drift manifests non-uniformly across frequency bands, with high-frequency details degrading faster than low-frequency structure. However, rather than implicitly balancing frequency domains, we propose to explicitly intercept error accumulation at the token level.

### 3 Preliminaries: Self-Attention in Autoregressive Text-to-Video Models

Modern text-to-video (T2V) generators [39, 19, 35, 36] often produce long videos by iteratively synthesizing the output in chunks (batches of frames), while conditioning each new chunk on the prompt and on representations of previously generated content [32, 35, 36]. Many such systems employ a diffusion backbone for within-chunk generation, often implemented with diffusion transformers (DiTs) [29, 31, 30], together with a recurrent/iterative mechanism across chunks to extend temporal horizon [32].

At the core of these models are latent spatiotemporal representations and self-attention. Each generated chunk is represented as a set of latent tokens (e.g., patchified latent features over space and time). When generating subsequent chunks, the model conditions on text features as well as latent tokens from earlier chunks, enabling long-range temporal dependencies [31, 30].

Self-Attention and Temporal Key-Value Caching Self-attention [23] allows each token in the current chunk to aggregate information from other tokens in its context, spanning spatial (within-frame) and temporal (across-frame) dimensions. In video DiTs, this mechanism is repeatedly applied inside the denoising network across diffusion steps [29, 31].

To support long-horizon context efficiently, many chunk-wise T2V architectures use a temporal key-value (KV) cache that stores key/value projections from previously generated chunks, so current queries can attend to both current tokens and cached tokens without recomputing past projections [35, 36]. Related inference-time caching methods for video diffusion further accelerate generation by selectively reusing computations across denoising steps [33, 34].

Formally, let Q be the queries for the current tokens, and let (Kcache,Vcache) denote cached keys/values from earlier chunks. The attention operation is

QK⊤

V, (1) with

√

Attention(Q,K,V) = softmax

d

K = [Kcurr;Kcache], V = [Vcurr;Vcache], (2) where [·;·] denotes concatenation along the token dimension and d is the attention head dimension.

Temporal Drift While temporal KV caching enables long-range dependencies, it can also amplify temporal drift: imperfections in earlier latent tokens may be repeatedly attended to and propagated as the cache grows. Over long rollouts, this can accumulate and manifest as identity changes, structural inconsistency, or degraded motion coherence [32, 36]. This mechanism motivates inference-time interventions that control how information is retrieved from the cache and how errors propagate in recurrent T2V generation.

### 4 Method

We propose an inference-time framework for stabilizing long-horizon auto-regressive video diffusion via latent-domain token pruning. Our method is designed to operate on auto-regressive video diffusion models that condition future generation on previously generated frames via causal self-attention and a temporal key-value (KV) cache, such as Self Forcing [19] and Rolling Forcing [18]. The core mechanism of the proposed method focuses on identifying and removing unstable latent conditioning tokens during auto-regressive inference. Algorithm 1 outlines a single TokenTrim step.

##### 4.1 Motion-Stabilized Initialization

The quality of the first batch of generated frames is critical in auto-regressive video generation, as errors introduced at this stage propagate and amplify over time. To reduce early-stage corruption, we generate the first batch of frames

[Figure 31]

###### TokenTrimFlowMo

[Figure 32]

“”

(a) “A balloon animal dog walks forward”

[Figure 33]

###### TokenTrimFlowMo

[Figure 34]

- (b) “A dragon lowers its head and spreads its wings”

TokenTrimFlowMo

- (c) “A paper cutout character walks sideways ﬂatly.”

[Figure 35]

[Figure 36]

[Figure 37]

###### TokenTrimFlowMo

[Figure 38]

(d) “smoke drifts sideways with trailing motion”

- Figure 4: Text-to-video results from FlowMo [20] and TokenTrim. For additional qualitative results see App. F.

using FlowMo [20], which introduces motion-aware variance guidance at inference time. It encourages temporally coherent motion and reduces initial artifacts by guiding the denoising trajectory toward stable motion patterns. In our framework, FlowMo is applied only to the first batch of frames, producing a stable latent anchor. All subsequent batches are generated using the base auto-regressive model.

##### 4.2 Latent Summary Construction

Let Xt−1 = {x(t−f)1}Ff=1 denote the previously generated batch of F frames at auto-regressive step t − 1, where x(t−f)1 ∈ RH×W×3 denotes a single video frame. Each frame is encoded into a set of spatial latent tokens

Z(t−f)1 = E(x(t−f)1) ∈ RN×D, (3) where E(·) denotes the encoder, N is the number of spatial tokens, and D is the latent dimension. We construct a latent summary frame by averaging across the temporal dimension (frame index):

1 F

Zt−1 =

F

Z(t−f)1 ∈ RN×D. (4)

f=1

After generating a candidate current batch Xt, we treat it in the same manner as Xt−1 and compute a corresponding latent summary Zt ∈ RN×D. Averaging across frames ensures that Zt−1 and Zt share identical spatial token structure, enabling direct token-wise comparison.

##### 4.3 Per-Token Latent Drift Estimation

We estimate instability in the auto-regressive context by computing a per-token drift score using latentspace subtraction. For each spatial token index i ∈ {1,...,N}, the drift score is defined as

Algorithm 1 A Single TokenTrim Step

Input: Previous batch frames Xt−1, candidate current batch frames Xt, encoder E, temporal KV cache (Kcache, Vcache), running drift statistics (µt, σt), pruning fraction p, sensitivity λ, warm-up length Twarm.

di = ∥Zt(i) − Zt−1(i)∥2 . (5) This measure captures deviations in semantic and structural features at the patch level and serves as an indicator of latent instability between consecutive autoregressive steps.

Output: Accepted current batch Xt⋆ and updated KV cache.

- 1: Zt−1 ← F1 Ff=1 E(x(t−f)1) ∈ RN×D

- 2: Zt ← F1 Ff=1 E(x(tf)) ∈ RN×D

- 3: di ← ∥Zt(i) − Zt−1(i)∥2 ∀i ∈ {1, . . . , N}
- 4: St ← TopIndices({di}Ni=1, ⌈pN⌉)
- 5: Dt ← |S1

t| i∈St di

- 6: if t ≤ Twarm then
- 7: Xt⋆ ← Xt
- 8: UpdateStatsAndCache(Xt⋆, Kcache, Vcache)
- 9: Return (Xt⋆, Kcache, Vcache)
- 10: end if
- 11: if Dt ≤ µt + λσt then
- 12: Xt⋆ ← Xt
- 13: UpdateStatsAndCache(Xt⋆, Kcache, Vcache)
- 14: Return (Xt⋆, Kcache, Vcache)
- 15: else
- 16: mt(i) ← ⊮[i ∈/ St]
- 17: K˜ ← Kcache[mt], V˜ ← Vcache[mt]
- 18: Xt⋆ ← GENERATEBATCH(K˜ , V˜ )
- 19: UpdateStatsAndCache(Xt⋆, K˜ , V˜ )
- 20: Return (Xt⋆, K˜ , V˜ )
- 21: end if

This comparison is performed between latent summaries that share the same tokenization grid and are computed at the same spatial resolution; moderate camera motion is therefore reflected as smooth, coherent changes across neighboring tokens rather than isolated high-magnitude drift, while corrupted or unstable regions induce localized spikes in di.

##### 4.4 Drift Severity and Trigger Criterion

Rather than pruning unconditionally, we first assess whether the current batch exhibits abnormal drift. Let p ∈ (0,1) denote the pruning fraction, and let N be the number of spatial latent tokens. For each autoregressive step t, we rank the per-token drift values {di}Ni=1 and define

St ⊆ {1,...,N}, |St| = ⌈p · N⌉, as the set of indices corresponding to the top p · N spatial tokens with the largest drift scores. We define a scalar drift severity score Dt ∈ R as the mean drift over these tokens:

1 |St| i∈S

di, di ∈ R≥0. (6)

Dt =

t

To obtain an adaptive threshold, we maintain running statistics over previously accepted batches. A batch Xτ is considered accepted if it is finalized (completed a tokentrim step, see Alg. 1) and appended to the auto-regressive context. Let

At = {τ < t | Xτ was accepted} denote the index set of such accepted auto-regressive steps prior to t, and let |At| be its cardinality. We define the running mean µt ∈ R and standard deviation σt ∈ R≥0 of the drift severity as

1 |At| τ∈A

1 |At| τ∈A

(Dτ − µt)2. (7)

µt =

Dτ, σt =

t

t

Rolling Forcing

Self Forcing

Text alignment

15.2% 74.9% 9.9%

15.5% 76.9% 7.6%

30.3% 59.5% 10.2%

26.2% 64.4% 9.4%

Motion

- 40.3% 47.4% 12.3%
- 41.7% 43.1% 15.2%

43.7% 45.3% 11.0%

Quality

43.3% 44.5% 13.2%

(No) Drift

0 20 40 60 80 100

0 20 40 60 80 100

Win Rate

Win Rate

Ours

Both

Baseline

| |
|---|

| |
|---|

- Figure 5: Human preference study conducted on Rolling Forcing [18] (left) and Self Forcing [19] (right) using VideoJAM-bench [28]. TokenTrim is consistently preferred in terms of drift reduction, motion quality, and overall visual quality, while preserving text–video alignment. Error bars indicate 95% confidence intervals computed via Dirichlet sampling with Laplace smoothing.

We trigger a pruning intervention when the current drift severity exceeds the adaptive threshold:

Dt > µt + λσt, (8) where λ > 0 is a sensitivity hyperparameter controlling the strictness of the trigger (we use λ = 2.0). During the first Twarm auto-regressive steps, the statistics µt and σt may be unreliable due to limited history. We therefore disable pruning during this warm-up phase and only accumulate drift statistics.

##### 4.5 Hard Pruning and Regeneration

When the drift criterion is exceeded, we regenerate the current batch with hard pruning applied to the auto-regressive context. Specifically, we remove from the temporal KV cache all token positions whose spatial indices belong to St. Let mt ∈ {0,1}N denote the spatial pruning mask:

- 0, i ∈ St,
- 1, otherwise.

(9)

mt(i) =

We apply this mask to the cached keys and values (conceptually along the token dimension):

K˜ = Kcache[mt], V˜ = Vcache[mt], (10)

where Kcache and Vcache denote the cached keys and values corresponding to previously generated frames. The pruned tokens (K˜ ,V˜ ) are used to condition a second generation attempt of the current batch. If the drift criterion is not exceeded, the generated batch is accepted without pruning, see Fig 2. Formally, the current batch Xt is accepted as the final output for step t and appended to the auto-regressive context without any pruning or regeneration. For stability and efficiency, we limit the procedure to at most R regeneration attempts per batch (we use R = 1). That is, when pruning is triggered, the batch is regenerated at most once using the pruned KV cache; if the regenerated batch still violates the drift criterion, it is accepted as-is to avoid unbounded regeneration loops.

##### 4.6 Integration with Self Forcing and Rolling Forcing

In Self Forcing [19], which employs a rolling KV cache during inference, latent drift estimation and hard pruning are applied before appending new KV entries to the cache. In Rolling Forcing [18], which separates a global anchor cache (derived from initial frames) from a recent temporal context cache, hard pruning is applied exclusively to the recent context tokens.

Overall, our method addresses temporal drift by operating entirely in the latent space; it requires no additional training or supervision, and it introduces almost no computational overhead.

### 5 Experiments

We present a comprehensive evaluation of TokenTrim in the setting of long-horizon auto-regressive text-to-video generation. Our experiments examine whether selectively pruning unstable latent tokens at inference time can mitigate error accumulation across generation steps, while preserving visual quality, motion realism, and semantic alignment. We evaluate TokenTrim both quantitatively, using VBench metrics, and qualitatively, via visual comparisons and human preference studies. All evaluations are conducted under the same experimental conditions and include comparisons against both baseline auto-regressive methods and inference-time methods.

Implementation details. We evaluate TokenTrim on Rolling Forcing[18] and Self Forcing[19], two auto-regressive inference strategies built on the Wan2.1-1.3B text-to-video model [21]. TokenTrim operates purely at inference time and introduces no changes to the underlying model weights or denoising dynamics.

TokenTrim monitors latent drift between consecutive generated batches and triggers hard pruning when abnormal drift is detected. Unless otherwise stated, we use a pruning ratio of p=0.1, a drift threshold parameter λ = 2.0, and a warm-up period of Twarm = 2 batches. These parameters are specific to TokenTrim and are independent of the underlying generation model or inference strategy. When pruning is triggered, the current batch is regenerated once using the pruned temporal KV cache.

In the FlowMo-adapted setup, FlowMo[20] is applied during the generation of all the batches using its default inferencetime settings. In contrast, when combined with TokenTrim, FlowMo is applied only to the first batch to produce a stable initialization, as described in Sec. 4.1, and is disabled for subsequent batches.

All experiments are conducted on a single NVIDIA H100 GPU. Rolling Forcing and Self Forcing generate 30-second videos at 16 FPS and a resolution of 832 × 480.

##### 5.1 Quantitative Results

We evaluate TokenTrim using the VBench benchmark [22], which provides a comprehensive suite of motion, quality, and semantic metrics for text-to-video generation. We report both per-dimension scores and aggregated metrics, including Semantic Score, Quality Score, and the overall Final Score. Comparisons are made against the baseline auto-regressive methods, Rolling Forcing [18] and Self Forcing[19] as well as FlowMo [20], using identical prompts and generation settings.

Automatic metrics. Tab. 1 reports aggregated VBench results comparing TokenTrim against the baseline auto-regressive methods (Rolling Forcing and Self Forcing) as well as the inference-time method FlowMo, under identical generation settings. We enclose the aggregated metrics, which constitute an average of all the benchmark dimensions, and measure the overall quality of the generations. A full breakdown of all metrics is provided in App A.

Table 1: VBench evaluation results. Comparison of baseline auto-regressive inference, FlowMo, and TokenTrim across aggregated VBench scores.

Model Semantic Quality Final

Rolling Forcing 68.52% 81.72% 75.12% + FlowMo 69.53% 82.09% 75.81%

- + TokenTrim 72.05% 87.30% 79.67% (+4.55%)

Self Forcing 68.98% 82.89% 75.93% + FlowMo 68.25% 83.85% 76.05%

- + TokenTrim 73.89% 89.79% 81.84% (+5.91%)

When applied on top of Rolling Forcing, TokenTrim yields substantial improvements across all aggregated VBench metrics. The Final Score increases from 75.12% to 79.67% (+4.55%), supported by gains in

Quality Score (+5.58%) and Semantic Score (+3.53%). These improvements indicate that selectively pruning unstable latent tokens significantly enhances both perceptual quality and semantic consistency over long auto-regressive rollouts.

A similar trend is observed under Self Forcing. TokenTrim improves the Final Score from 75.93% to 81.84% (+5.91%), accompanied by strong gains in Quality Score (+6.90%) and Semantic Score (+4.91%). Notably, the magnitude of these improvements is even larger than under Rolling Forcing, highlighting TokenTrim’s effectiveness across different auto-regressive inference strategies.

TokenTrim also consistently outperforms FlowMo at the aggregate level. Under Self Forcing, TokenTrim raises the Final Score to 81.84% (+5.91%), whereas FlowMo yields only a marginal increase to 76.05% (+0.12%). This gap is further reflected in the Semantic Score, which improves substantially with TokenTrim (+4.91%) but slightly degrades with FlowMo (-0.73%), as well as in the Quality Score, where TokenTrim achieves a +6.90% gain compared to FlowMo’s +0.96%.

Overall, these aggregated results demonstrate that TokenTrim provides a significantly stronger and more reliable improvement in semantic fidelity and visual quality than both baseline auto-regressive methods and inference-time motion guidance, leading to higher overall generation quality in long-horizon video synthesis.

Inference-Time Overhead. Averaged over 128 generated 30s videos, TokenTrim increases wall-clock runtime by ×1.08 relative to the Rolling Forcing baseline. In contrast, applying the FlowMo-adapted setup incurs a substantially larger cost, resulting in a ×2.18 slowdown over Rolling Forcing.

##### 5.2 Qualitative Results

We qualitatively compare TokenTrim against the baseline auto-regressive methods and FlowMo using long-horizon generations exceeding one minute. Representative examples are shown in Fig. 1 and Fig. 3.

Across all examples, TokenTrim maintains stable object identities, colors, and structure over time, whereas the baseline methods and FlowMo exhibit progressive degradation. Common failure modes include color shifts, structural distortions, background corruption, and identity drift. For example, in Fig. 3(a), the baseline Rolling Forcing model produces a Pikachu character with missing or duplicated limbs over time. In Fig. 1(a,b), baseline generations show noticeable color drift and texture degradation, while TokenTrim preserves consistent appearance. Similarly, artifacts such as lens flare accumulation and background warping are visible in Fig. 3(b–d) for the baseline and FlowMo, but are largely absent when using TokenTrim. Additional qualitative comparisons between TokenTrim and Rolling Forcing are provided in App. E, with corresponding results for Self Forcing reported in App. D.

Fig. 4 directly compares TokenTrim with FlowMo under identical settings. While FlowMo often produces plausible short-term motion, it exhibits gradual structural drift over long horizons. For example, in Fig. 4(a), the balloon animal dog generated with FlowMo progressively deforms, most noticeably in the snout and head shape, whereas TokenTrim preserves consistent geometry and proportions. In Fig. 4(b), FlowMo introduces subtle distortions in the dragon’s wings and head during the rollout, while TokenTrim maintains coherent anatomy and smooth articulation. Similar effects are observed in Fig. 4(c,d), where FlowMo suffers from background warping and implied motion drift, whereas TokenTrim maintains stable silhouettes and coherent global motion. Additional qualitative comparisons between TokenTrim and FlowMo are provided in App. F.

TokenTrim vs FlowMo

Text alignment

14.9% 74.5% 10.6%

27.8% 57.1% 15.1%

Motion

41.7% 45.2% 13.1%

Quality

49.7% 40.1% 10.2%

(No) Drift

0 20 40 60 80 100

Win Rate

Ours

Both

Baseline

| |
|---|

| |
|---|

These examples illustrate that TokenTrim stabilizes longhorizon generation by preventing corrupted latent tokens from repeatedly influencing future steps, resulting in videos that remain visually coherent throughout extended rollouts.

Figure 6: Human preference study on VideoJAM-Bench [28]. Win rates comparing TokenTrim against FlowMo [20] under Rolling Forcing. Error bars: 95% CIs via Dirichlet sampling with Laplace smoothing.

##### 5.3 User Study

We conduct a human preference study using prompts from the VideoJAM benchmark [28]. For each prompt, participants are shown paired videos generated under identical settings, differing only in the inference-time method (baseline, FlowMo, or TokenTrim). Video order is randomized to avoid positional bias. Each pair is evaluated by five independent annotators, resulting in 640 responses per baseline. Participants evaluate videos along four criteria: text–video alignment, aesthetic quality, motion coherence, and temporal drift (see App. B).

Figure 5 reports human preference results comparing TokenTrim against the baseline under both Rolling Forcing (left) and Self Forcing (right). Under Rolling Forcing, TokenTrim achieves a preference rate of 15.2% in text–video alignment, compared to 9.9% for the baseline, indicating that pruning unstable tokens does not harm semantic consistency. For motion coherence, TokenTrim is preferred in 30.3% of cases, nearly three times higher than the baseline (10.2%), demonstrating a clear improvement in temporally plausible motion. A similar trend is observed for aesthetic quality, where TokenTrim reaches 40.3% preference versus 12.3% for the baseline. The largest improvement is observed in temporal drift, with TokenTrim attaining 41.7% preference compared to 15.2% for the baseline, highlighting its effectiveness in maintaining stable visual characteristics over long rollouts. A consistent pattern emerges under Self Forcing. TokenTrim improves text–video alignment to 15.5%, compared to 7.6% for the baseline. For motion coherence, TokenTrim achieves a preference rate of 26.2%, substantially higher than the baseline (9.4%). In aesthetic quality,

TokenTrim is preferred in 43.7% of comparisons, compared to 11.0% for the baseline. Finally, for temporal drift, TokenTrim reaches 43.3% preference, significantly outperforming the baseline (13.2%), confirming that the benefits of TokenTrim generalize across auto-regressive inference strategies.

Fig. 6 reports a direct human preference comparison between TokenTrim and FlowMo. Across all evaluated criteria, TokenTrim consistently receives more favorable votes than FlowMo. For text–video alignment, TokenTrim is preferred in 14.9% of comparisons, compared to 10.6% for FlowMo, indicating that TokenTrim provides stronger semantic alignment between the generated videos and the input prompts. In terms of motion coherence, TokenTrim achieves a higher preference rate of 27.8%, surpassing FlowMo’s 15.1%, demonstrating that TokenTrim exceeds FlowMo’s motion quality despite not explicitly optimizing motion dynamics. TokenTrim also shows a substantial advantage in aesthetic quality, receiving 41.7% of preferences compared to 13.1% for FlowMo, reflecting consistently stronger visual appearance. The largest gap is observed in temporal drift, where TokenTrim attains a preference rate of 49.7%, dramatically outperforming FlowMo’s 10.2%. This result highlights TokenTrim’s superior ability to preserve stable structure, appearance, and identity over long generation horizons. Overall, the user study confirms that TokenTrim produces videos that are perceived as more stable, coherent, and visually consistent over long horizons.

- Table 2: TokenTrim Ablation on Rolling Forcing. Ablation study of TokenTrim (TT) under different pruning ratios and initialization settings. The full TokenTrim method uses 10% pruning with FlowMo initialization. Columns correspond to the aggregated Semantic Score (Semantic), Quality Score (Quality), and the overall Final Score (Final) as defined by VBench.

Model Semantic Quality Final Full TT Method 72.05% 87.30% 79.67%

TT 5% Pruning 70.78% 85.92% 78.35% (-1.32%) TT 20% Pruning 64.22% 72.29% 68.25% (-11.87%) TT w/o FlowMo Init 71.17% 83.49% 77.33% (-2.34%)

##### 5.4 Ablation Study

We conduct an ablation study to analyze the contribution of each component in TokenTrim and to evaluate the effect of different pruning strategies. Tab. 2 reports aggregated VBench results for a set of TokenTrim variants evaluated on top of Rolling Forcing. In particular, we analyze alternative pruning strategies, including a variant that disables the FlowMobased motion-stabilized initialization, and examine sensitivity to the pruning ratio by testing fixed pruning rates of 5% and 20%, instead of the 10% we use throughout all other experiments. Pruning 5% of the most unstable tokens results in a moderate degradation over the baseline TokenTrim method, achieving a final score of 78.35% (−1.32% compared to the baseline TokenTrim method), while aggressive pruning at 20% severely degrades performance, reducing the final score to 68.25% (−11.87%). These results highlight that excessive removal of contextual tokens disrupts semantic continuity and visual quality. In another ablation, the FlowMo is removed from the initialization of the first batch (FlowMo is not used in the subsequent batches). This leads to a final score of 77.33% which is −2.34% less than the full TokenTrim method, demonstrating that motion-stabilized initialization provides a complementary benefit by improving early coherence. Interestingly, the degradation that occures in this ablation, is greater than the benefit of FlowMo to the baseline Rolling Forcing method (see Tab. 1), demonstrating that TokenTrim leads to better utilization of FlowMo than the baseline Rolling Forcing. A full breakdown of all metrics is provided in App. C.

### 6 Limitations & Future Work

TokenTrim operates purely at inference time and leaves model parameters unchanged; therefore, its gains are ultimately bounded by the capabilities and biases of the underlying video diffusion backbone and its training data. When the base model persistently struggles to represent an object, preserve identity, or produce stable motion, TokenTrim can primarily attenuate error propagation rather than fully correct the generation. Moreover, our current implementation uses a fixed hard-pruning budget (e.g., pruning a constant fraction or top-k tokens per step). Although this design is lightweight and adds negligible overhead, a single global pruning setting may be suboptimal across prompts, content types, and rollout lengths. An important direction for future work is to make pruning adaptive, i.e., dynamically choosing the pruning rate and structure at each step using drift statistics or uncertainty estimates so that difficult sequences receive stronger suppression while stable sequences retain richer context, especially over long rollouts.

### 7 Conclusions

Can we sustain long-video generation by simply knowing what to forget? In this work, we address the pervasive issue of temporal drift not by adding more capacity or retraining, but by pruning tokens at inference time. Since the uncontrolled accumulation of corrupted tokens in the auto-regressive context is a primary driver of semantic collapse, TokenTrim computes a drift score for the model’s latent representation and distinguishes between stable context and hallucinated artifacts. We demonstrate that pruning these unstable tokens allows the model to maintain its connection to reliable anchors, effectively breaking the feedback loop of error accumulation. This approach transforms the KV-cache from a passive history log into an active self-correcting memory mechanism. Our results suggest that future progress in infinite video synthesis lies not only in how much context a model can remember, but also in its ability to selectively discard the noise that threatens to distort reality.

#### 8 Acknowledgments This work was supported by a grant from the Tel Aviv University Center for AI and Data Science (TAD). References

- [1] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. MagicVideo: Efficient Video Generation with Latent Diffusion Models. arXiv preprint arXiv:2211.11018, 2023.
- [2] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, Di Zhang, and Bin Cui. VideoTetris: Towards Compositional Text-to-Video Generation. arXiv preprint arXiv:2406.04277, 2024.
- [3] Tianxing Wu, Chenyang Si, Yuming Jiang, Ziqi Huang, and Ziwei Liu. FreeInit: Bridging Initialization Gap in Video Diffusion Models. arXiv preprint arXiv:2312.07537, 2024.
- [4] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating Minute-level Long Videos with Autoregressive Language Models. arXiv preprint arXiv:2410.02757, 2025.
- [5] Zhentao Fan, Zongzuo Wang, and Weiwei Zhang. TaoCache: Structure-Maintained Video Generation Acceleration. arXiv preprint arXiv:2508.08978, 2025.
- [6] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models. arXiv preprint arXiv:2412.07772, 2025.
- [7] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, Jianlong Fu, Gong Ming, Lijuan Wang, Zicheng Liu, Houqiang Li, and Nan Duan. NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation. arXiv preprint arXiv:2303.12346, 2023.
- [8] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. FreeLong: Training-Free Long Video Generation with SpectralBlend Temporal Attention. arXiv preprint arXiv:2407.19918, 2024.
- [9] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-A-Video: Text-to-Video Generation without Text-Video Data. arXiv preprint arXiv:2209.14792, 2022.
- [10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [11] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen Video: High Definition Video Generation with Diffusion Models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [12] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127, 2023.
- [13] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, Tanghui Jia, Junwu Zhang, Zhenyu Tang, Yatian Pang, Bin She, Cen Yan, Zhiheng Hu, Xiaoyi Dong, Lin Chen, Zhang Pan, Xing Zhou, Shaoling Dong, and Yonghong Tian. Open-Sora Plan: Open-Source Large Video Generation Model. arXiv preprint arXiv:2412.00131, 2024.

- [14] Kling Team. Kling: A Video Generation Model with Spatiotemporal Attention. 2024. Available at https: //klingai.com/.
- [15] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal Flow Matching for Efficient Video Generative Modeling. arXiv preprint arXiv:2410.05954, 2025.
- [16] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. LTX-Video: Realtime Video Latent Diffusion. arXiv preprint arXiv:2501.00103, 2024.
- [17] OpenAI. Video Generation Models as World Simulators. 2024. Available at https://openai.com/research/ video-generation-models-as-world-simulators.
- [18] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling Forcing: Autoregressive Long Video Diffusion in Real Time. arXiv preprint arXiv:2509.25161, 2025.
- [19] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [20] Ariel Shaulov, Itay Hazan, Lior Wolf, and Hila Chefer. FlowMo: Variance-Based Flow Guidance for Coherent Motion in Video Generation. arXiv preprint arXiv:2506.01144, 2025.
- [21] Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, and others. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314, 2025.
- [22] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, and others. VBench: Comprehensive Benchmark Suite for Video Generative Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818, 2024.
- [23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.
- [24] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. CogVideo: Large-Scale Pretraining for Text-to-Video Generation via Transformers. arXiv preprint arXiv:2205.15868, 2022.
- [25] Weiming Ren, Huan Yang, Ge Zhang, Cong Wei, Xinrun Du, Wenhao Huang, and Wenhu Chen. Consisti2V: Enhancing Visual Consistency for Image-to-Video Generation. arXiv preprint arXiv:2402.04324, 2024.
- [26] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. ID-Animator: Zero-Shot Identity-Preserving Human Video Generation. arXiv preprint arXiv:2404.15275, 2024.
- [27] Jianwen Jiang, Chao Liang, Jiaqi Yang, Gaojie Lin, Tianyun Zhong, and Yanbo Zheng. Loopy: Taming AudioDriven Portrait Avatar with Long-Term Motion Dependency. arXiv preprint arXiv:2409.02634, 2024.
- [28] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models. arXiv preprint arXiv:2502.02492, 2025.
- [29] William Peebles and Saining Xie. Scalable Diffusion Models with Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- [30] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A Space-Time Diffusion Model for Video Generation. In Proceedings of the European Conference on Computer Vision (ECCV), 2024.
- [31] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-Video Diffusion Models with an Expert Transformer. arXiv preprint arXiv:2408.06072, 2024.
- [32] Siyang Zhang and Ser-Nam Lim. Towards Chunk-Wise Generation for Long Videos. arXiv preprint arXiv:2411.18668, 2024.
- [33] Kumara Kahatapitiya, Haozhe Liu, Sen He, Ding Liu, Menglin Jia, Chenyang Zhang, Michael S. Ryoo, and Tian Xie. Adaptive Caching for Faster Video Generation with Diffusion Transformers. arXiv preprint arXiv:2411.02397, 2024.

- [34] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. TeaCache: Timestep Embedding Tells: It’s Time to Cache for Video Diffusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- [35] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing. arXiv preprint arXiv:2411.16375, 2024.
- [36] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, Song Han, and Yukang Chen. LongLive: Real-Time Interactive Long Video Generation. arXiv preprint arXiv:2509.22622, 2025.
- [37] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video Diffusion Models. arXiv preprint arXiv:2204.03458, 2022.
- [38] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. ModelScope Text-toVideo Technical Report. arXiv preprint arXiv:2308.06571, 2023.
- [39] Ruben Villegas, Jiahui Liu, Kimin Lee, Zhuowen Tu, Bo Dai, and James Hays. Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [40] Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin Huang, and Dahua Lin. End-to-End Training for Autoregressive Video Diffusion via Self-Resampling. arXiv preprint arXiv:2512.15702, 2025.
- [41] Marco Pasini, Javier Nistal, Stefan Lattner, and George Fazekas. Continuous Autoregressive Models with Noise Augmentation Avoid Error Accumulation. arXiv preprint arXiv:2411.18447, 2024.
- [42] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text. arXiv preprint arXiv:2403.14773, 2025.
- [43] Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models. arXiv preprint arXiv:2512.12080, 2025.
- [44] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models. arXiv preprint arXiv:2401.09047, 2024.
- [45] Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models. arXiv preprint arXiv:2305.10474, 2024.
- [46] Tim Brooks, Janne Hellsten, Miika Aittala, Ting-Chun Wang, Timo Aila, Jaakko Lehtinen, Ming-Yu Liu, Alexei A. Efros, and Tero Karras. Generating Long Videos of Dynamic Scenes. arXiv preprint arXiv:2206.03429, 2022.

##### Table 3: VBench evaluation across all dimensions. Comparison of Rolling Forcing and Self Forcing with TokenTrim and FlowMo. Dimension Rolling Forcing + TokenTrim + FlowMo Self Forcing + TokenTrim + FlowMo

Subject Consistency 91.92% 90.12% 91.94% 88.29% 85.54% 87.99% Background Consistency 92.87% 93.54% 91.27% 89.53% 89.72% 87.12% Temporal Flickering 96.77% 98.89% 95.21% 98.90% 98.93% 98.91% Motion Smoothness 97.21% 99.02% 98.03% 97.63% 98.64% 97.95% Dynamic Degree 59.57% 62.11% 56.07% 67.75% 68.97% 65.46%

- Aesthetic Quality 64.19% 63.95% 63.05% 61.06% 63.13% 62.15% Imaging Quality 71.70% 71.91% 68.62% 68.92% 69.17% 66.99% Object Class 88.97% 89.63% 89.33% 81.24% 80.30% 82.24% Multiple Objects 68.31% 65.29% 67.28% 63.98% 60.37% 61.03% Human Action 75.81% 80.00% 74.89% 83.72% 83.93% 82.09%

- Color 87.98% 85.53% 86.20% 81.21% 81.28% 79.83% Spatial Relationship 79.26% 75.09% 80.53% 74.76% 76.02% 76.59% Scene 29.22% 32.20% 30.85% 32.59% 30.94% 29.25% Appearance Style 21.72% 21.40% 19.56% 23.49% 25.56% 24.15% Temporal Style 23.26% 25.49% 25.12% 20.55% 20.39% 20.02% Overall Consistency 24.92% 28.37% 26.98% 25.12% 25.76% 25.10%

- Semantic Score 68.52% 72.05% 69.53% 68.98% 73.89% 68.25% Quality Score 81.72% 87.30% 82.09% 82.89% 89.79% 83.85% Final Score 75.12% 79.67% (+4.55%) 75.81% 75.93% 81.84% (+5.91%) 76.05%

### A VBench Metrics Breakdown

We evaluate the impact of TokenTrim on both Rolling Forcing [18] and Self Forcing [19] using the VBench benchmark, with full per-dimension results reported in Tab. 3.

The VBench metrics [22] consistently demonstrate that TokenTrim yields substantial improvements across a wide range of dimensions for both auto-regressive baselines. Notably, TokenTrim improves performance in nearly all motion- and stability-related metrics, indicating effective mitigation of error accumulation across auto-regressive steps rather than localized, within-chunk refinement.

For Rolling Forcing, TokenTrim provides clear gains in key motion coherence metrics, improving Temporal Flickering by +2.12% (from 96.77% to 98.89%), Motion Smoothness by +1.81%, and Dynamic Degree by +2.54%. These improvements indicate that TokenTrim enhances temporal stability while preserving motion complexity, rather than suppressing dynamics. TokenTrim further improves higher-level consistency metrics such as Human Action (+4.19%), Scene (+2.98%), and Overall Consistency (+3.45%), reflecting stronger long-horizon semantic coherence.

- A similar pattern is observed for Self Forcing. TokenTrim improves Temporal Flickering (+0.03%), Motion Smoothness (+1.01%), and Dynamic Degree (+1.22%), while also yielding consistent gains in Human Action, Appearance Style (+2.07%), and Overall Consistency (+0.64%). Importantly, these improvements arise despite Self Forcing already employing stronger temporal anchoring, highlighting that TokenTrim addresses a complementary failure mode—namely, the repeated reuse of corrupted latent tokens across auto-regressive steps.

In the aggregated metrics, TokenTrim delivers strong and consistent improvements for both baselines. For Rolling Forcing, TokenTrim increases the Semantic Score by +3.53% and the Quality Score by +5.58%, resulting in a Final Score improvement of +4.55%. For Self Forcing, the gains are even larger, with +4.91% in Semantic Score, +6.90% in Quality Score, and a Final Score increase of +5.91%. These consistent improvements across both architectures confirm that TokenTrim robustly mitigates long-horizon drift by controlling context reuse, leading to higher-quality and more stable video generation over extended rollouts.

We also compare TokenTrim against FlowMo using the VBench benchmark, with detailed per-dimension results reported in Tab. 3.

The VBench metrics [22] clearly demonstrate that TokenTrim provides stronger and more consistent improvements than FlowMo across nearly all dimensions, particularly those related to long-horizon stability. Across the full set of VBench dimensions, TokenTrim achieves the best score in the majority of categories, whereas FlowMo only attains the highest score in a small subset and often trails both TokenTrim and the baseline.

Focusing first on motion-related metrics, TokenTrim yields substantial gains in Temporal Flickering (+2.12% over baseline, from 96.77% to 98.89%) and Motion Smoothness (+1.81%), outperforming FlowMo in both dimensions (95.21% and 98.03%, respectively). Importantly, TokenTrim also improves the Dynamic Degree (+2.54%), while FlowMo significantly reduces it (-3.50%), indicating that FlowMo’s apparent motion coherence may partially arise

Table 4: VBench ablation across all dimensions.

Dimension Rolling Forcing + Full TT Method + TT 5% Pruning + TT 20% Pruning + TT w/o FlowMo init

Subject Consistency 91.92% 90.12% 90.59% 85.31% 90.10% Background Consistency 92.87% 93.54% 93.07% 78.02% 93.28% Temporal Flickering 96.77% 98.89% 97.01% 89.83% 97.25% Motion Smoothness 97.21% 99.02% 97.65% 85.52% 97.56% Dynamic Degree 59.57% 62.11% 61.48% 55.97% 60.11%

- Aesthetic Quality 64.19% 63.95% 64.32% 59.82% 62.54% Imaging Quality 71.70% 71.91% 72.81% 62.68% 70.62% Object Class 88.97% 89.63% 87.66% 72.11% 88.03% Multiple Objects 68.31% 65.29% 64.13% 62.83% 64.42% Human Action 75.81% 80.00% 77.29% 69.18% 78.95%

- Color 87.98% 85.53% 87.51% 64.32% 86.06% Spatial Relationship 79.26% 75.09% 76.42% 68.04% 77.36% Scene 29.22% 32.20% 30.63% 29.75% 28.74% Appearance Style 21.72% 21.40% 21.98% 11.86% 20.91% Temporal Style 23.26% 25.49% 22.78% 15.29% 25.41% Overall Consistency 24.92% 28.37% 25.65% 19.21% 26.31%

- Semantic Score 68.52% 72.05% 70.78% 59.22% 71.17% Quality Score 81.72% 87.30% 85.92% 67.29% 83.49% Final Score 75.12% 79.67% (+4.55%) 78.35% (+3.23%) 63.25% (-11.87%) 77.33% (+2.21%)

from suppressing motion complexity rather than preserving it. This contrast suggests that TokenTrim maintains realistic motion dynamics while reducing drift, rather than converging to overly static trajectories.

TokenTrim further improves higher-level semantic and structural consistency. It achieves the strongest gains in Human Action (+4.19%), Scene (+2.98%), and Overall Consistency (+3.45%), consistently outperforming FlowMo, which provides smaller or inconsistent improvements in these categories. Notably, FlowMo only surpasses TokenTrim in Subject Consistency and Spatial Relationship, but these gains do not translate into improved aggregate performance.

In the aggregated metrics, TokenTrim clearly dominates. It improves the Semantic Score by +3.53% and the Quality Score by +5.58%, resulting in a Final Score of 79.67% (+4.55% over baseline). In contrast, FlowMo yields only marginal improvements, achieving a Final Score of 75.81% (+0.69%). These results indicate that while FlowMo offers limited local benefits, it does not effectively mitigate the accumulation of errors across auto-regressive steps. TokenTrim’s explicit control over context reuse enables significantly better long-horizon coherence and overall video quality.

In addition, a closer inspection of FlowMo’s per-dimension performance reveals that its improvements are uneven and largely confined to a small subset of metrics. Under Rolling Forcing, FlowMo provides modest gains in Motion Smoothness (+0.82%, from 97.21% to 98.03%) and Spatial Relationship (+1.27%), but these come at the expense of reduced motion diversity, as reflected by a substantial drop in Dynamic Degree (-3.50%, from 59.57% to 56.07%). FlowMo also degrades several stability-sensitive metrics, including Temporal Flickering (-1.56%) and Overall Consistency (from 24.92% to 26.98%, still well below TokenTrim’s 28.37%). A similar pattern is observed under Self Forcing. While FlowMo yields a small improvement in Quality Score (+0.96%), it reduces the Semantic Score (-0.73%) and lowers the Dynamic Degree from 67.75% to 65.46%. Across both baselines, FlowMo attains the best score in only a few isolated dimensions (e.g., Subject Consistency and Spatial Relationship), but these gains do not translate into consistent improvements in long-horizon stability or aggregated performance.

### B User Study: Instructions Provided to Participants

As part of the evaluations we performed on our method, we conducted a user study, as described in 5.1. The study was designed to assess human preferences on videos generated with and without FlowMo, using the videoJAM benchmark [28], which focuses on motion coherence.

The study was conducted using Google Forms. For each prompt, participants were shown a pair of videos—one with FlowMo and one without—generated with the same random seed (1024). The order of the videos was randomized to avoid positional bias. Each pair was evaluated by five different participants, resulting in 640 responses per baseline.

Participants were asked to evaluate the videos based on three criteria: text alignment, aesthetic quality, motion coherence, and (No) Drift. The instructions provided to the annotators are reproduced below, together with a screenshot of the annotation interface (see Fig. 7).

Annotator Instructions. Participants were first asked to carefully read the given text prompt and then watch two generated videos. After viewing both videos, they were instructed to answer the following questions:

- • Text alignment: Which video better matches the given caption?
- • Quality: From an aesthetic perspective, which video looks better overall?
- • Motion: Which video exhibits more coherent and physically plausible motion?

– Do Note: It is OK if the quality is less impressive as long as the motion looks better

- • (No) Drift: Which video better maintains consistent visual characteristics throughout the entire sequence, including stable colors, characters or objects, shapes and identities, and an unchanged environment or background, without noticeable visual corruption or changes over time?

[Figure 39]

Figure 7: Screenshot of the Google Form used in the user study.

### C Ablation Study - VBench Metrics Breakdown

We present an ablation study in Tab. 4 to analyze the contribution of each component of TokenTrim and to examine its sensitivity to pruning strength and initialization quality. All variants are evaluated on Rolling Forcing to isolate the effect of inference-time context management.

First, we observe that the full TokenTrim configuration consistently achieves the strongest performance across most motion- and stability-related dimensions, confirming that both adaptive pruning and drift-based triggering are essential. Compared to the baseline, TokenTrim improves Temporal Flickering by +2.12%, Motion Smoothness by +1.81%, Dynamic Degree by +2.54%, and Overall Consistency by +3.45%, indicating effective mitigation of long-horizon error accumulation without suppressing motion.

Next, we study the effect of pruning strength. A conservative pruning rate of 5% already yields noticeable improvements over the baseline, increasing the Final Score by +3.23%. However, its gains are consistently smaller than those achieved by the full TokenTrim configuration, particularly in Human Action (+1.48% vs. +4.19%) and Overall Consistency (+0.73% vs. +3.45%). In contrast, aggressive pruning (20%) leads to severe degradation across nearly all dimensions, including large drops in Motion Smoothness (-11.69%), Dynamic Degree (-3.60%), and Quality Score (-14.43%), resulting in a substantial Final Score decrease of -11.87%. This clearly demonstrates that excessive token removal harms semantic and visual fidelity, underscoring the necessity of adaptive, drift-aware pruning rather than fixed-rate pruning.

Finally, removing FlowMo from the initialization stage (w/o FlowMo) degrades performance relative to the full TokenTrim pipeline, reducing the Final Score by 2.22%. While this variant still outperforms the baseline, it exhibits weaker gains in Dynamic Degree, Human Action, and Overall Consistency, highlighting the importance of a stable initial latent anchor for preventing early corruption that later propagates through the auto-regressive context.

Overall, this ablation study confirms that TokenTrim’s effectiveness arises from the combination of (i) drift-triggered pruning, (ii) moderate, adaptive pruning ratios, and (iii) stable initialization. Removing or weakening any of these components leads to measurable degradation, while overly aggressive pruning causes catastrophic loss of video quality.

[Figure 40]

[Figure 41]

SelfForcingSelfForcing

[Figure 42]

[Figure 43]

###### +TokenTrim

(a) “A translucent jellyfish drifting slowly through deep blue water.”

[Figure 44]

[Figure 45]

###### +TokenTrim

- (b) “A cloud of white smoke drifting horizontally across a deep navy background.”

[Figure 46]

[Figure 47]

+TokenTrimSelfForcing

- (c) “A swarm of identical black birds moving as a single shape across a gray sky.”

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

###### +TokenTrimSelfForcing

[Figure 54]

[Figure 55]

(d) “A static camera shot of a robotic spider walking straight toward the camera.”

Figure 8: Additional Qualitative Results comparison between TokenTrim and Self Forcing [19].

### D Additional Qualitative Experiments: TokenTrim vs Self Forcing

- Fig. 8 compares TokenTrim with the Self Forcing baseline under identical long-horizon generation settings. Although Self Forcing improves short-term temporal anchoring, it still exhibits gradual degradation when errors accumulate across extended rollouts. In Fig.8(a), depicting a translucent jellyfish drifting through deep blue water, Self Forcing introduces subtle shape distortions and loss of translucency over time, while TokenTrim preserves a stable body structure and consistent appearance. In Fig.8(b), where a cloud of white smoke drifts horizontally across a dark background, Self

Forcing suffers from spatial warping and uneven density, whereas TokenTrim maintains coherent global motion and uniform texture. Similar behavior is observed in Fig.8(c), where a swarm of birds generated with Self Forcing gradually loses collective structure, while TokenTrim preserves synchronized motion and stable silhouettes. Finally, in Fig.8(d), a robotic spider walking toward the camera exhibits accumulated structural drift under Self Forcing, whereas TokenTrim maintains consistent limb geometry and forward motion. Overall, these examples demonstrate that TokenTrim more effectively suppresses long-horizon error accumulation than Self Forcing, resulting in improved structural stability and temporal coherence across diverse motion patterns.

### E Additional Qualitative Experiments: TokenTrim vs Rolling Forcing

- Fig. 9 compares long-horizon generations produced by Rolling Forcing with and without TokenTrim under identical settings. While Rolling Forcing generates plausible content in early frames, it exhibits progressive temporal degradation as the rollout advances. In Fig.3(a), depicting a helicopter flying over a forest, the baseline gradually introduces background distortion and structural inconsistencies in the helicopter, whereas TokenTrim preserves stable geometry and consistent background appearance throughout the sequence. In Fig.3(b), where a humanoid emerges from smoke and reforms, Rolling Forcing suffers from identity drift and shape instability during the reformation process, while TokenTrim maintains coherent structure and consistent appearance across frames. Fig.3(c) shows a glowing wireframe animal filling itself with flesh while running: the baseline exhibits color bleeding and deformation over time, whereas TokenTrim preserves smooth transitions and stable anatomy. Finally, in Fig.3(d), Rolling Forcing introduces cloth distortion and background warping during the walking motion, while TokenTrim maintains consistent cloth dynamics and a stable scene layout. Overall, these examples demonstrate that TokenTrim effectively suppresses long-horizon drift by preventing corrupted latent tokens from propagating through the auto-regressive context, resulting in significantly improved temporal stability compared to Rolling Forcing alone.

F Additional Qualitative Experiments: TokenTrim vs FlowMo

- Fig.10 presents a qualitative comparison between TokenTrim and FlowMo under identical long-horizon generation settings. While FlowMo often produces plausible motion in early frames, it exhibits progressive structural and semantic drift as the rollout proceeds. In Fig.10(a), showing a humanoid constructed from stacked books walking forward, FlowMo gradually introduces misalignment and deformation in the book stack, leading to inconsistent body proportions, whereas TokenTrim preserves a stable silhouette and coherent limb structure throughout the sequence. In Fig.10(b), depicting a wooden mannequin walking under soft studio lighting, FlowMo suffers from subtle identity drift and joint instability over time, while TokenTrim maintains consistent articulation and appearance. Fig.10(c) shows a dragon towering over a burning battlefield: FlowMo introduces background warping and deformation in the wings and body across frames, whereas TokenTrim preserves coherent anatomy and stable environmental structure. Finally, in Fig. 10(d), FlowMo exhibits temporal inconsistency in the arrangement of background figures and lighting, while TokenTrim maintains a stable composition and consistent scene layout.

+TokenTrimRollingForcing

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

(a) “A helicopter flying over a forest.”

+TokenTrimRollingForcing

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

(b) “A humanoid appears out of smoke, then disperses and reforms.”

RollingForcingRollingForcing

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

###### +TokenTrim

(c) “A glowing wireframe animal fills itself with flesh as it runs.”

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

###### +TokenTrim

(d) “A fabric humanoid walks forward with subtle cloth sway.”

Figure 9: Additional Qualitative Results comparison between TokenTrim and Rolling Forcing [18].

[Figure 72]

[Figure 73]

###### TokenTrimFlowMo

[Figure 74]

[Figure 75]

(a) “A side-profile shot of a humanoid constructed from stacked books walking forward.”

[Figure 76]

[Figure 77]

FlowMoFlowMoFlowMo

[Figure 78]

[Figure 79]

###### TokenTrim

(b) “A wooden mannequin walks forward under soft studio lighting.”

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

###### TokenTrim

(c) “Antares in full dragon form towering over a burning battlefield.”

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

###### TokenTrim

(d) “Ashborn standing in a vast shadow realm, endless soldiers kneeling.”

Figure 10: Additional Qualitative Results comparison between TokenTrim and FlowMo [20].

