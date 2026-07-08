## ScalingAR: Scaling Confidence for Autoregressive Image Generation

# arXiv:2509.26376v3[cs.CV]24Jun2026

Harold Haodong Chen*12 Xianfeng Wu*3 Wen-Jie Shu4 Rongjin Guo5 Disen Lan6 Harry Yang2 Ying-Cong Chen†12

### Abstract

Test-time strategies have shown remarkable success in improving large language models, but their application to next-token prediction (NTP) autoregressive (AR) image generation remains largely underexplored. Existing test-time scaling (TTS) methods for visual autoregressive models (VAR) rely on frequent partial decoding and external reward models, which are inefficient and often ineffective for NTP-based image generation due to the inherent instability of intermediate decoding results. To address these limitations, we propose ScalingAR, a novel test-time scaling framework tailored for NTP-based AR image generation. ScalingAR introduces token entropy as a confidence signal and operates at two complementary levels: (i) Profile Level, integrates intrinsic uncertainty and conditional utilization into a unified confidence state, and (ii) Policy Level, leverages this state for adaptive trajectory pruning and dynamic guidance scheduling. Without requiring early decoding or auxiliary rewards, ScalingAR achieves significant improvements across diverse benchmarks. Experiments show that ScalingAR (I) improves base models by 12.5% on GenEval and 15.2% on TIIF-Bench, (II) reduces visual token consumption by 62.0% while outperforming baselines, and (III) enhances robustness, mitigating performance degradation by 26.0% in challenging scenarios. These results establish ScalingAR as a robust and efficient test-time scaling solution for autoregressive image generation. Our code: ScalingAR Repository.

### 1. Introduction

Large language models (LLMs) (Brown et al., 2020; Radford et al., 2019) have demonstrated the capabilities of next-

*Equal contribution 1HKUST(GZ) 2HKUST 3UNC-Chapel Hill 4ZODA 5CityUHK 6FDU. Correspondence to: Ying-Cong Chen <yingcongchen@usk.hk>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

token prediction (NTP) paradigm. This success has renewed interest in applying autoregressive (AR) architectures beyond text, motivating recent visual generative models that represent images in discrete token spaces (Sun et al., 2024; Tian et al., 2024; Li et al., 2024) as shown in Figure 2 (b). Compared to diffusion models, which operate over continuous noise trajectories, token-based AR models promise a more unified modality interface.

As the field evolves, the parameters and training data of foundation models (Wang et al., 2024; Yang et al., 2025) have increasingly grown to levels that are inaccessible for most university researchers. In this context, many studies have started to investigate post-training methods. Inspired by recent advancements such as GRPO (Shao et al., 2024), a surge of reinforcement learning research has emerged in both language and visual generation domains (Jiang et al., 2025; Cui et al., 2025). Meanwhile, another research avenue focusing on test-time scaling (TTS) has emerged (Lightman et al., 2023; Muennighoff et al., 2025; Zuo et al., 2025), which aims to explore whether a slight increase in computational expense during inference can achieve performance on par with training-time methods, which typically incur much larger costs.

While test-time scaling has been extensively researched in language models, analogous progress for autoregressive visual generation remains sparse. Images differ from text in three practical ways that complicate direct transfer: (i) holism: dropping the last 20% of a text sequence may still leave a syntactically valid answer, whereas truncating an image token stream yields an unusable artifact; (ii) objective ambiguity: many language scaling setups optimize toward a verifiable final answer (e.g., math reasoning), whereas image generation lacks a single ground-truth target; and (iii) early signal scarcity: partial image token decodes are visually unstable, making premature selection risky. Moreover, recent work TTS-VAR (Chen et al., 2025b) introduced TTS for the next-scale prediction (NSP) paradigm in visual autoregressive model (VAR) (Tian et al., 2024) by predicting images in a coarse-to-fine manner (Figure 2 (a)). This intermediate visibility enables reward models to score during scaling but comes with limitations that require predicting large residual token maps at each scale and frequent decoding makes the process inefficient and less suitable for the NTP paradigm.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

###### LlamaGen+ScalingAR(Ours)

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 1. (Top) ScalingAR significantly improves the quality of autoregressive image generation. Detailed prompts are provided in Appendix §D. (Bottom Left) The token confidence trajectory over the generation process. (Bottom Right) Performance comparison of ScalingAR on TIIF-Bench with classic test-time scaling strategies, i.e., Importance Sampling (IS) and Best-of-N (BoN).

Building on these insights, we introduce ScalingAR, a novel test-time scaling framework tailored to the NTP paradigm in autoregressive image generation. Unlike nextscale TTS-VAR, ScalingAR eliminates the need for frequent partial decoding and external reward models (as shown in Figure 2 (d)), relying solely on intrinsic signals derived from visual token entropy and conditional signals to profile confidence. Specifically, in response to limitations, ScalingAR prunes unreliable trajectories without interrupting generation (holism), constructs confidence by combining intrinsic uncertainty and conditional signals (objective ambiguity), and extracts stability directly from model probabilities rather than intermediate outputs (early signal scarcity). Technically, ScalingAR features a two-level design: ♣ Profile Level, which constructs a unified confidence state by integrating intrinsic generation stability with conditioning effectiveness; and ♠ Policy Level, which leverages this confidence state to prune failing trajectories and dynamically adjust conditioning strength through adaptive termination and guidance scheduling. Our contributions can be summarized as follows:

❶ We propose ScalingAR, a novel test-time scaling framework tailored to next-token prediction AR image generation, featuring a two-level design with Profile Level for dual-channel confidence profiling on-the-fly, and Policy Level for trajectory pruning and guidance scheduling.

❷ We for the first time investigate token entropy in visual

token generation. By relying solely on intrinsic signals from the model, ScalingAR eliminates the need for frequent early decoding and external reward models, enabling a more efficient and reliable scaling process.

❸ Extensive experiments on both general and compositional benchmarks demonstrate that ScalingAR is:

- (I) high-performing, achieving significant performance gains over base models (i.e., LlamaGen and AR-GRPO), by 12.5% on GenEval and 15.2% on TIIF-Bench;
- (II) token-efficient, outperforming classic baselines (i.e., Importance Sampling and Best-of-N) while reducing visual token consumption by 62.0%; and (III) robust in challenging scenarios, mitigating performance degradation by 26.0% compared to base models in highly complex generation settings.

### 2. Related Work

Autoregressive Image Generation. Autoregressive models have leveraged the scaling capabilities of language models (Yang et al., 2025; Brown et al., 2020; Radford et al., 2019) to generate images. These approaches employ discrete image tokenizers (Van Den Oord et al., 2017; Razavi et al., 2019) in conjunction with transformers, using a nexttoken prediction strategy. VQ-based methods (Lee et al., 2022; Razavi et al., 2019; Esser et al., 2021), e.g., VQ-VAE (Van Den Oord et al., 2017), convert image patches into index-based tokens, which are then predicted sequentially

[Figure 14]

- Figure 2. (a) Next-scale prediction paradigm generates multi-scale token maps coarse-to-fine. (b) Next-token prediction paradigm sequentially predicts next image tokens. (c) Illustration of Best-of-N sampling that generates multiple candidate and selects the best via voting or scoring. (d) Overview of our proposed ScalingAR, highlighting its ability to leverage token entropy to early-stop low-confidence samples and identify winning samples without the need for additional reward models.

by a decoder-only transformer. However, these VQ-based AR methods are limited by the lack of scaled-up transformers and the inherent quantization error in VQ-VAE. This has prevented them from achieving performance on par with diffusion models. Recent advancements (Wu et al., 2025a; Yu et al., 2022; Team, 2024) have scaled up AR models for visual generation. Additionally, some variants have been proposed, such as the next-scale prediction paradigm of VAR (Tian et al., 2024; Han et al., 2025), which predicts from coarse to fine token maps, and the parallel token prediction of masked AR (MAR) (Li et al., 2024; Wu et al., 2025b; Fan et al., 2025). Despite these developments, the mainstream approach remains the NTP paradigm, particularly as the field moves towards unified models (Xie et al., 2025; Wang et al., 2024; Ge et al., 2024) that can jointly handle textual and visual tokens. This alignment with language modeling allows for more versatile and scalable architectures.

Test-Time Scaling. Current LLMs have increasingly succeeded by allocating substantial reasoning at inference time, a paradigm known as test-time scaling (Snell et al., 2024; Welleck et al., 2024). This scaling can occur along two main axes: (1) Chain-of-Thought (CoT) (Wei et al., 2022) Depth: lengthening a single reasoning trajectory through more thinking steps, often relying on large-scale reinforcement learning with many samples (Yang et al., 2025; Jaech et al., 2024; Guo et al., 2025a) or simpler post-training strategies (Ye et al., 2025; Muennighoff et al., 2025); (2) Parallel Generation: scaling by increasing the number of trajectories and aggregating them, as seen in works like Self-Consistency (Wang et al., 2023) and Best-of-N (Lightman et al., 2023). Recent advances (Kang et al., 2025; Fu

et al., 2025) have also incorporated token entropy for confidence estimation, improving the quality of reasoning traces. However, exploring TTS for AR image generation has been limited. This is due to the holistic nature of image generation, where overall coherence is paramount (see Figure 2 (c)), unlike reasoning tasks with well-defined ground truths. Recent work (Guo et al., 2025b) applies Best-of-N to AR image generation by introducing external reward models but without new TTS strategy design itself. Moreover, similar to TTS-VAR (Chen et al., 2025b), it relies on frequent early decoding and rewards. To this end, we propose the first TTS strategy tailored for AR image generation. Notably, we pioneer the exploration of token entropy in image generation, enabling our method to leverage visual token confidence without the need for early decoding or additional rewards.

### 3. Preliminaries

Next-Token Prediction Autoregressive Modeling. NTP is a fundamental paradigm in autoregressive models, where the model generates sequences by predicting the next token based on previously generated tokens. The generation process can be mathematically described as follows:

T

p(xt|x1,x2,...,xt−1). (1)

p(x1,x2,...,xT) =

t=1

This formulation allows the model to leverage past information to inform future predictions, making it particularly effective for sequential data generation.

The training of autoregressive models typically involves maximizing the likelihood of the observed sequences, which

###### CFG=1 CFG=7.5 CFG=15 Ours

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

- Figure 3. (Left) Confidence distribution of ScalingAR on GenEval and TIIF-Bench. (Right) Illustration of the trade-off between visual quality and semantic alignment with fixed Classifier-Free Guidance (CFG) in AR image generation. 1st: A 35 mm photo of a cityscape resembling Moscow floating in the sky on flying islands. 2nd: The colorful hot air balloon floated near the dark grey storm clouds. can be expressed as:

L =

T

t=1

log p(xt|x<t). (2)

This objective encourages the model to learn the underlying distribution of the data, enabling it to generate coherent and contextually appropriate sequences.

Token Entropy in Language Modeling. Token entropy is a critical metric for evaluating the uncertainty associated with the predictions made by language models (Kang et al., 2025). It quantifies the amount of unpredictability in the model’s output distribution for a given token. The entropy H at a specific position i in the sequence can be defined as:

Hi = −

j

pi(j)log pi(j), (3)

where pi(j) denotes the predicted probability of the j-th token in the vocabulary at position i. Low entropy indicates high certainty in the prediction, while high entropy reflects greater uncertainty.

Furthermore, token confidence can be derived from the predicted distribution (Fu et al., 2025). The confidence Ci for a token at position i is defined as:

Ci = −

1 k

k

j=1

log pi(j), (4)

where k denotes the number of top tokens considered. High confidence values correlate with sharper distributions, indicating that the model is more certain about its predictions.

- 4. Methodology

putational efficiency while maintaining high-quality generation. The framework consists of two coupled levels: (i) Dual-Channel Confidence Profile that transforms raw logit statistics into a normalized signal of generation stability (Section §4.1); and (ii) Confidence-Guided Policies that leverage this state to autonomously prune failing trajectories and modulate guidance strength (Section §4.2).

#### 4.1. Dual-Channel Confidence Profile

Autoregressive image generators traditionally treat all partial trajectories as equally promising until completion, as illustrated in Figure 2 (c). However, empirical inspection reveals two dominant failure modes during inference that often foreshadow poor final results: ❶ local instability, characterized by high entropy pockets and wavering token choices, as shown in Figure 1 (Bottom Left) and Figure 3 (Left); and ❷ semantic drift, where the semantic influence of the prompt gradually fades, resulting in misaligned or aesthetically suboptimal outputs, as shown in Figure 3 (Right).

To address these challenges, we introduce the Dual-Channel Confidence Profile, consisting of two complementary channels: ➀ intrinsic channel: captures localized instability and spatial anomalies within the token grid. ➁ conditional channel: quantifies the marginal contribution of textual conditioning to ensure semantic alignment.

4.1.1. INTRINSIC CHANNEL

The first failure mode in visual AR models is local instability, where the model becomes uncertain about the geometric or textural structure of a specific region. This manifests as a high entropy pocket that diffuses into global incoherence. We capture this via two complementary metrics.

To address the challenges of test-time scaling for NTPbased image generation, we propose ScalingAR, a framework that operates on a calibrated confidence state derived solely from the model’s internal logits. ScalingAR is designed to operate effectively without relying on frequent partial decoding or external reward models, ensuring com-

Token-level Confidence. The raw entropy of the softmax distribution reflects the model’s aleatoric uncertainty at each step. However, relying solely on raw entropy is brittle due to varying vocabulary sizes and context-dependent ambi-

guity. Therefore, we formulate a robust token confidence score stokt . Let πt denote the softmax distribution over the vocabulary V at step t. We combine the normalized entropy Ht = − v∈V πt(v)log πt(v) with the top-1/top-2 margin mt = πt(v1)−πt(v2), creating a surrogate for decisiveness. Crucially, to filter transient noise inherent in stochastic sampling, we apply an exponential moving average (EMA) to track the confidence trend:

stokt = EMA 1 − Norm( Ht,mt) . (5)

A persistent decline in stokt serves as a leading indicator of concept collapse, where the model loses track of the visual structure.

Worst-block Stability. A global average of token confidence often masks catastrophic failures occurring in small, localized regions (e.g., a distorted face in a high-quality background). To strictly penalize such local anomalies, we partition the latent grid into non-overlapping blocks and monitor the subset Wt with the highest mean entropy, i.e., the worst q% of blocks. We derive a stability metric Bt via dynamic normalization:

1 |Wt| k∈W

Ek . (6)

Bt = 1 − Nrolling

t

By focusing on the lower bound of spatial stability, Bt ensures that a trajectory is penalized if any critical region fails, aligning with the weakest link principle of visual perception. The final intrinsic score It is a calibrated aggregation of the temporal (stokt ) and spatial (Bt) signals.

- 4.1.2. CONDITIONAL CHANNEL

A trajectory may be visually stable (i.e., low entropy) but semantically unrelated to the prompt, i.e., semantic drift. This occurs when the autoregressive prior dominates the conditional probability. To quantify the effective influence of the text prompt y, we analyze the information gain provided by the condition at step t. Specifically, we compute the KL divergence between the conditional distribution p(·|x<t,y) and the unconditional counterpart p(·|x<t,∅):

Kt = KL(pc,t ∥ pu,t). (7) Since the magnitude of Kt varies significantly across different prompts and generation stages, absolute thresholding is prone to error. Instead, we employ a dynamic z-score normalization to map Kt into a standardized drift score Dt ∈ [0,1]. This adaptive normalization renders the metric robust to prompt complexity, requiring no manual tuning per sample. Low values of Dt indicate that the model is ignoring the prompt, signaling a need for intervention.

- 4.1.3. UNIFIED CONFIDENCE STATE

We fuse the intrinsic and conditional signals into a scalar unified confidence score Ct. Beyond the instantaneous value,

the dynamics of Ct provide critical information. Failing trajectories often exhibit a “confidence basin”, i.e., a prolonged period of low scores. To distinguish between temporary dips and permanent collapse, we track the relative rebound Rt:

Ct − Cmin(t) |Cmin(t)| + ε

, (8)

Rt =

where Cmin(t) is the running minimum. This unified state St = {Ct,Rt,It, Dt} provides a holistic view of the generation health, serving as the basis for our control policies.

#### 4.2. Confidence-Guided Policies

With a calibrated state St, we transition from passive observation to active test-time control, enabling dynamic intervention in autoregressive generation. To achieve this, we introduce two lightweight yet effective policies: ➀ an adaptive termination that prunes unpromising trajectories to reclaim computation; and ➁ a guidance scheduler that dynamically modulates CFG scale to balance semantic alignment.

- 4.2.1. ADAPTIVE TERMINATION

A core challenge in test-time scaling is determining when to abandon a computation. Static thresholds are suboptimal: a hard prompt may result in naturally lower confidence scores across all valid trajectories, while an easy prompt yields higher scores. A fixed cutoff would thus aggressively prune valid samples in difficult cases. To address this, we propose a distribution-aware pruning mechanism.

Quantile-Based Thresholding. We define a dynamic threshold θ↓ based on the statistics of the current batch. Initialized as the p-quantile of the confidence scores Ct across active beams, θ↓ adapts to the difficulty of the prompt. As generation proceeds, θ↓ is updated via EMA, ensuring it tracks the non-stationary distribution of token confidence:

θ↓ ← (1 − λ)θ↓ + λ · Quantilep({Ct}current). (9) This ensures that we always prune the relatively worstperforming trajectories, regardless of the absolute confidence scale.

Recovery Safeguard. To prevent the premature termination of trajectories experiencing transient instability, we implement a recovery hysteresis. A candidate flagged for pruning (Ct < θ↓) is granted a probation window. It is only terminated if it fails to demonstrate a significant rebound (i.e., low Rt) within this window. This mechanism effectively filters out false positives, preserving diversity while eliminating catastrophic failures.

- 4.2.2. GUIDANCE SCHEDULER

Classifier-free guidance (CFG) creates a trade-off: high guidance improves alignment but reduces diversity and visual quality, while low guidance risks semantic drift. Stan-

Table 1. Evaluation on GenEval (Ghosh et al., 2023) and TIIF-Bench (Wei et al., 2025) benchmarks. “Diff.+AR” refers to the unified architecture, and “MAR” indicates the masked AR architecture (Li et al., 2024). We bold the best results.

GenEval TIIF-Bench

Method #Params Arch.

Two Obj.↑ Posit.↑ Color Attr.↑ Over.↑ Basic↑ Advanced↑ Designer↑ Over.↑ DALLE·3 (Betker et al., 2023) - Diff. - - - 0.67 78.40 68.45 62.69 72.94 Show-o (Xie et al., 2025) 1.3B Diff.+AR 0.80 0.31 0.50 0.68 71.30 59.89 68.66 59.24 LightGen (Wu et al., 2025b) 0.8B MAR 0.65 0.22 0.43 0.62 53.99 45.76 59.70 46.42 Infinity (Han et al., 2025) 2B VAR 0.85 0.49 0.57 0.73 71.63 57.81 61.19 59.66 Emu3 (Han et al., 2025) 8.5B AR 0.81 0.49 0.45 0.66 - - - Janus (Wu et al., 2025a) 1.5B AR 0.68 0.46 0.42 0.61 - - - AR-GRPO (Yuan et al., 2025) 0.8B AR 0.27 0.02 0.03 0.31 19.59 14.91 17.91 16.22 + IS 0.8B AR 0.47 0.08 0.07 0.44 26.00 19.03 17.62 19.84 + BoN 0.8B AR 0.46 0.08 0.06 0.44 25.67 19.91 20.69 21.08 + ScalingAR (Ours) 0.8B AR 0.54 0.24 0.15 0.49 29.71 26.43 25.90 26.35 LlamaGen (Sun et al., 2024) 0.8B AR 0.34 0.21 0.04 0.32 49.58 40.44 40.30 40.35 + IS 0.8B AR 0.21 0.11 0.02 0.14 54.81 40.34 39.93 42.44 + BoN 0.8B AR 0.27 0.11 0.02 0.15 54.79 40.78 37.69 42.02 + ScalingAR (Ours) 0.8B AR 0.40 0.28 0.12 0.36 57.36 44.13 42.54 46.47

dard methods use a fixed scale, which is suboptimal as the need for guidance varies throughout the generation process. We propose a closed-loop guidance scheduler that modulates the CFG scale st in response to the confidence profile. The scheduler operates on a compensatory principle: it strengthens guidance when the model shows signs of drift or instability, and relaxes it when generation is confident. Formally, the target scale stargett is adjusted as:

stargett ∝ (1 − Dt)

##### − Rt

##### + Var(I)

##### .

Diversity Reward

Drift Correction

Volatility Dampening

(10) This formulation naturally increases st when semantic utilization Dt drops or when intrinsic stability I becomes volatile. Conversely, when the trajectory exhibits a strong rebound Rt (i.e., indicating a return to a high-confidence manifold), the scheduler reduces st to encourage visual diversity and prevent over-saturation. This dynamic adaptation allows ScalingAR to navigate the Pareto frontier of quality and alignment more effectively than static baselines.

### 5. Experiments

In this section, we conduct extensive experiments to answer the following research questions:

- RQ1: Does ScalingAR enhance the quality of generated images?
- RQ2: Does ScalingAR outperform other TTS strategies for both effectiveness and efficiency?
- RQ3: How sensitive is ScalingAR to its key components?
- RQ4: Whether ScalingAR holds advantages over other TTS strategies in both scalability and robustness?

#### 5.1. Experimental Settings

Baselines. We apply ScalingAR to the advanced models: LlamaGen (512 × 512) (Sun et al., 2024) and AR-GRPO (256 × 256) (Yuan et al., 2025). Since no prior work has explored TTS for the NTP image generation, we focus our

comparisons on the following conventional baselines: Importance Sampling (IS) (Owen & Zhou, 2000) and Best-ofN (BoN) (Lightman et al., 2023). We also provide results on SimpleAR (Wang et al., 2025a) and Janus-Pro (Chen et al., 2025a) in Appendix §B.

Evaluations. To evaluate the effectiveness of ScalingAR, we adopt GenEval (Ghosh et al., 2023) and TIIF-Bench (Wei et al., 2025) as primary benchmarks for both general and compositional text-to-image generation capabilities. These benchmarks offer a comprehensive evaluation of the model’s ability to produce high-quality and semantically consistent images from text prompts.

#### 5.2. Performance & Efficiency Comparison

To answer RQ1 and RQ2, we comprehensively compare ScalingAR against two baselines on general and compositional benchmarks in Table 1, alongside qualitative results, user study, and token consumption comparisons shown in Figure 1, 4, and Figure 5. Key observations are summarized as follows: Obs.❶ ScalingAR excels in enhancing both general and compositional generation quality. As illustrated in Table 1, our ScalingAR consistently outperforms baseline methods (i.e., IS and BoN), which achieve minimal or even negative performance gains, across benchmarks targeting distinct aspects of text-to-image generation. Figure 1 (Top) and Figure 4 provide qualitative evidence of ScalingAR’s capabilities, showcasing visually superior results that excel in aesthetic quality and semantic alignment, e.g., numerical accuracy, color fidelity, and subject clarity. Furthermore, Figure 5 (Left) highlights ScalingAR’s effectiveness in aligning image generation with human preferences, as validated through user studies. Obs.❷ ScalingAR is a token-efficient test-time AR image generation enhancer. Figure 5 (Middle) demonstrates that ScalingAR consistently surpasses other TTS strategies across benchmarks, requiring fewer visual tokens. Unlike BoN, which relies on external reward models and ex-

###### LlamaGen + IS + ScalingAR (Ours) LlamaGen + BoN + ScalingAR (Ours)

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

In the vibrant garden, five butterflies danced gracefully among the blooming flowers A photo of a breathtaking view from a mountain summit

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

A photo of a fluffy bunny A photo of a tranquil lake reflecting autumn foliage

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

A photo of a lively squirrel The blue mug is on top of the green coaster

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

The soft, warm glow of the candlelight cast a romantic ambiance over the room The fluffy white cat slept on the warm fuzzy blanket

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

The prickly green cactus contrasted with the smooth white walls The juicy burger sat on the soft bun and the crispy lettuce

Figure 4. Qualitative results of ScalingAR. More results on more base models are provided in Appendix §B and §E.

cessive token consumption, ScalingAR leverages intrinsic confidence signals to reduce computational overhead while maintaining high-quality outputs.

Table 2. Ablation study of ScalingAR. Method Bas.↑ Adv.↑ Des.↑ Over.↑

ScalingAR 57.4 44.1 42.5 46.5 w/o Conditional Channel 54.1 43.1 42.2 45.1 w/o Worst-Block Stability 52.3 41.8 41.4 44.2 w/o Token-Level Confidence 49.6 40.4 40.3 40.4

#### 5.3. Ablation Analysis

- To answer RQ3, we perform step by step evaluations on TIIF-Bench to analyze the contributions of ScalingAR’s

confidence profiles, as detailed in Table 2. We give the following observations: Obs.❸ Effectiveness of Intrinsic Signal Profiling. Removing Token-Level Confidence or Worst-Block Stability both lead to a noticeable drop in performance, highlighting their critical role in capturing fine-grained entropy signals during visual token generation. This demonstrates the effectiveness of intrinsic signal profiling for maintaining local token stability and ensuring high-quality generation. Obs.❹ Importance of Condition State Balance. Table 2 also reveals that removing the Conditional Channel leads to significant degradation. Figure 3 (Right) further confirms its critical role in balancing interactions between text guidance and visual generation, ensuring coherent and stable outputs. Since the Policy Level builds

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Figure 5. (Left) User study across five dimensions: overall preference, aesthetic quality, realism fidelity, semantic alignment, attribute binding. (Middle) Visual token consumption of ScalingAR vs. baselines on TIIF-Bench. (Right) Scaling width and depth across sample number and token length.

LlamaGen + IS + BoN + ScalingAR

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

Wool expands like a cotton cloud, lifting off from the grassy field.

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

A commercial aircraft takes off from the ocean's surface

Figure 6. Robustness testing with impossible prompt. Detailed prompts are provided in Appendix §D.

upon the Profile Level, we primarily conduct ablation on the Profile Level here and provide Policy Level ablation in Appendix §D with more analysis like wall-clock consumption and FLOPs efficiency.

#### 5.4. Scalability & Robustness Analysis

- To answer RQ4, we compare ScalingAR with other TTS strategies (i.e., IS and BoN) in scaling width (i.e., sample number N) and depth (i.e., token length), as shown in Figure 5 (Right). To further assess the robustness of ScalingAR, we adopt the idea of “impossible prompting” (Bai et al., 2025) (e.g., “A young boy ... using chopsticks as a writing instrument, ... in a photo-realistic scene...”) to evaluate its performance even when none of the candidates are ideal, with the results presented in Figure 6. Our observations are summarized as follows: Obs.❺ ScalingAR unlocks scalable generalization across both width and depth. As shown in Figure 5 (Right), ScalingAR consistently outperforms IS and BoN across varying sample numbers and token lengths. This suggests that our scaling strategy enables performance to scale up effectively as scaling width and depth increase, making it a reliable solution for diverse autoregressive tasks. Obs.❻ ScalingAR empowers robust generation beyond standard scenarios. Figure 6 (Left)

demonstrates that under impossible prompts for unrealistic scenarios, ScalingAR exhibits clear robustness advantages over baselines. Furthermore, Figure 6 (Right) confirms that our method achieves more effective scaling when generating under challenging conditions, highlighting its adaptability and reliability in adverse scenarios.

### 6. Conclusion

In this work, we introduce ScalingAR, a novel test-time scaling framework tailored to next-token prediction autoregressive image generation. Unlike existing TTS strategies, ScalingAR proposes to explore visual token entropy for the first time as intrinsic signals, without relying on partial decoding or external rewards. By adopting a two-level design: Profile Level for calibrated confidence profiling and Policy Level for adaptive pruning and dynamic conditioning, ScalingAR achieves phase-aware control, enhancing generation quality with minimal additional token consumption. Comprehensive evaluations on both general and compositional capability benchmarks demonstrate that ScalingAR substantially improves the generation quality of existing AR models, along with generalizability and robustness, making it a strong baseline for AR image generation TTS.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically in efficient autoregressive image generation. A primary societal benefit of our approach is the promotion of sustainable AI practices; by employing distribution-aware pruning to eliminate unpromising generation trajectories, our method significantly reduces computational overhead and energy consumption compared to standard test-time scaling strategies.

On the ethical front, as a inference framework, ScalingAR inherits the safety profiles and potential biases of the underlying base models. While ScalingAR improves generation fidelity, it does not intrinsically filter harmful content. Therefore, we recommend that real-world deployment of this technology be accompanied by robust safety guardrails and content moderation mechanisms to mitigate risks associated with the generation of misleading or harmful imagery.

### References

Bai, Z., Ci, H., and Shou, M. Z. Impossible videos. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=MNSW6U5zUA.

Betker, J., Goh, G., Jing, L., Brooks, T., Wang, J., Li, L., Ouyang, L., Zhuang, J., Lee, J., Guo, Y., et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3): 8, 2023.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.

Cao, S., Valiant, G., and Liang, P. On the entropy calibration of language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=CGLoEvCllI.

Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., and Ruan, C. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025a.

Chen, Z., Chu, R., Chen, Y., Zhang, S., Wei, Y., Zhang, Y., and Liu, X. Tts-var: A test-time scaling framework for visual auto-regressive generation. arXiv preprint arXiv:2507.18537, 2025b.

Cui, G., Zhang, Y., Chen, J., Yuan, L., Wang, Z., Zuo, Y., Li, H., Fan, Y., Chen, H., Chen, W., et al. The entropy mech-

anism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617, 2025.

Esser, P., Rombach, R., and Ommer, B. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Fan, L., Li, T., Qin, S., Li, Y., Sun, C., Rubinstein, M., Sun, D., He, K., and Tian, Y. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.

net/forum?id=jQP5o1VAVc. Fu, Y., Wang, X., Tian, Y., and Zhao, J. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.

Ge, Y., Zhao, S., Zhu, J., Ge, Y., Yi, K., Song, L., Li, C., Ding, X., and Shan, Y. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Guo, Z., Zhang, R., Tong, C., Zhao, Z., Huang, R., Zhang, H., Zhang, M., Liu, J., Zhang, S., Gao, P., et al. Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926, 2025b.

Han, J., Liu, J., Jiang, Y., Yan, B., Zhang, Y., Yuan, Z., Peng, B., and Liu, X. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15733–15744, 2025.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Jiang, D., Guo, Z., Zhang, R., Zong, Z., Li, H., Zhuo, L., Yan, S., Heng, P.-A., and Li, H. T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot. arXiv preprint arXiv:2505.00703, 2025.

Kang, Z., Zhao, X., and Song, D. Scalable best-of-n selection for large language models via self-certainty. arXiv preprint arXiv:2502.18581, 2025.

Lee, D., Kim, C., Kim, S., Cho, M., and Han, W.-S. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11523–11532, 2022.

Li, T., Tian, Y., Li, H., Deng, M., and He, K. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424– 56445, 2024.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,

- B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Cand`es, E., and Hashimoto, T. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Owen, A. and Zhou, Y. Safe and effective importance sampling. Journal of the American Statistical Association, 95(449):135–143, 2000.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Razavi, A., Van den Oord, A., and Vinyals, O. Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems, 32, 2019.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., and Yuan, Z. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Team, C. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Tian, K., Jiang, Y., Yuan, Z., Peng, B., and Wang, L. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024.

Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Wang, J., Tian, Z., Wang, X., Zhang, X., Huang, W., Wu, Z., and Jiang, Y.-G. Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl. arXiv preprint arXiv:2504.11455, 2025a.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X.-H., Yang, J., Zhang, Z., Liu, Y., Yang, A., Zhao, A., Yue, Y., Song, S., Yu, B., Huang, G., and Lin, J. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b. URL https:

//openreview.net/forum?id=yfcpdY4gMP.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=1PL1NIMMrw.

Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wei, X., Zhang, J., Wang, Z., Wei, H., Guo, Z., and Zhang, L. Tiif-bench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.

Welleck, S., Bertsch, A., Finlayson, M., Schoelkopf, H., Xie, A., Neubig, G., Kulikov, I., and Harchaoui, Z. From decoding to meta-generation: Inference-time algorithms for large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https:// openreview.net/forum?id=eskQMcIbMS. Survey Certification.

Wu, C., Chen, X., Wu, Z., Ma, Y., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C., et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12966–12977, 2025a.

Wu, X., Bai, Y., Zheng, H., Chen, H. H., Liu, Y., Wang, Z., Ma, X., Shu, W.-J., Wu, X., Yang, H., et al. Lightgen: Efficient image generation through knowledge distillation and direct preference optimization. arXiv preprint arXiv:2503.08619, 2025b.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. In The Thirteenth International Conference on Learning Representations, 2025. URL https:// openreview.net/forum?id=o6Ynz6OIQ6.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pp. 15903–15935, 2023.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan,

- Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang,
- Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Ye, Y., Huang, Z., Xiao, Y., Chern, E., Xia, S., and Liu, P. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.

Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., Hutchinson, B., Han, W., Parekh, Z., Li, X., Zhang, H., Baldridge, J., and Wu, Y. Scaling autoregressive models for content-rich text-to-image generation. Transactions on Machine Learning Research, 2022. ISSN 28358856. URL https://openreview.net/forum? id=AFDcYJKhND. Featured Certification.

Yuan, S., Liu, Y., Yue, Y., Zhang, J., Zuo, W., Wang, Q., Zhang, F., and Zhou, G. Ar-grpo: Training autoregressive image generation models via reinforcement learning. arXiv preprint arXiv:2508.06924, 2025.

Zuo, Y., Zhang, K., Sheng, L., Qu, S., Cui, G., Zhu, X., Li, H., Zhang, Y., Long, X., Hua, E., et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

### A. Further Illustration of Entropy in AR Image Generation

- A key motivation behind our ScalingAR lies in the observation that high-entropy/low-confidence regions often exhibit greater uncertainty, which increases the likelihood of undesirable outcomes. While high entropy does not guarantee poor results, it correlates strongly with elevated error probabilities, making it a critical signal for stabilizing AR image generation.

Relevant Evidence. Similar observations have been validated across various domains: ❶ Entropy calibration in language models: Cao et al. (2025) demonstrated that high local token entropy correlates with higher error probabilities, highlighting its role as a risk indicator in generative tasks. ❷ Reinforcement learning mechanisms for reasoning: Works (Cui et al., 2025; Fu et al., 2025; Wang et al., 2025b) for LLM Reasoning treat high-entropy tokens as positions with dense information but unstable decisions or higher error risks. These findings underscore the necessity of carefully managing entropy during generation to balance exploration and stability.

Connection on ScalingAR. Our method, ScalingAR, can be interpreted as a stabilization mechanism that prunes trajectories with low confidence, effectively mitigating the risks associated with high-entropy regions. By focusing on confidence signals, ScalingAR ensures that the generation process avoids prolonged instability, leading to improved image quality. In Figure 1 (Bottom Left) and Figure 3 (Left), we compare the confidence distributions of ScalingAR and the base model. The results clearly show that higher token confidence correlates with better image quality, further validating our motivation to leverage confidence signals for trajectory pruning.

LlamaGen+ScalingAR(Ours)

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

7

6

5

4

3

2

Figure 7. Visualization of token entropy. (1st) A sunflower field stretching to the horizon under a bright blue sky. (2nd) A majestic lion resting on a rocky outcrop in the golden savanna light. (3rd) A detailed macro shot of a butterfly on a blooming flower.

Visualizing Token Entropy in Generated Images. To provide a more intuitive understanding, we visualize the entropy distributions of generated images in Figure 7. The figure highlights that regions with poor generation quality often correspond to higher entropy, reinforcing the notion that high-entropy tokens are more likely to contribute to undesirable outcomes. ScalingAR’s ability to suppress these regions through confidence-based pruning plays a pivotal role in achieving stable and high-quality image generation.

- B. Results of More Base Models

To further validate the generalizability of ScalingAR, we deployed our method on two additional AR models: SimpleAR1.5B (Wang et al., 2025a) and Janus-Pro-1B (Chen et al., 2025a). Importantly, the hyperparameter settings for ScalingAR were kept consistent with those used in the main experiments on LlamaGen and AR-GRPO, without any model-specific tuning. This ensures a fair evaluation of ScalingAR’s adaptability across different architectures and scales. Quantitative results in Table 3 and qualitative results in Figure 8 show significant performance improvements for both models, demonstrating ScalingAR’s effectiveness and broad applicability as a general-purpose stabilization framework.

Table 3. Evaluation of ScalingAR on more base models on GenEval (Ghosh et al., 2023).

Method Two Obj.↑ Pos.↑ Color Attri.↑ Overall↑

SimpleAR 0.90 0.28 0.45 0.63 + ScalingAR (Ours) 0.93 0.36 0.51 0.67 Janus-Pro 0.82 0.65 0.56 0.73 + ScalingAR (Ours) 0.87 0.69 0.61 0.77

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

###### SimpleAR+ScalingAR(Ours)

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

A blue sky with fluffy white clouds

A bird perched on a tree branch

A bowl of fresh strawberries

A child playing with puppies A kite flying high in the blue sky

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### Janus-Pro+ScalingAR(Ours)

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

A majestic bald eagle perched on a pine branch overlooking a mountain range

A close-up of a monarch butterfly resting on a purple flower in a sunlit meadow

A tropical hummingbird feeding from a bright red flower, wings a blur of motion

A colorful parrot perched on a branch in a rainforest

A vintage steamship sailing through foggy waters at sunrise

Figure 8. Qualitative results of ScalingAR on SimpleAR (Top) and Janus-Pro (Bottom).

### C. Implementation Details

#### C.1. Universal Configuration Strategy

Crucially, ScalingAR utilizes a single, universal configuration across all experiments. We do not perform model-specific hyperparameter tuning. The settings detailed below were applied identically to LlamaGen (0.8B), AR-GRPO (0.8B), SimpleAR (1.5B), and Janus-Pro (1B), demonstrating the structural robustness of the proposed method.

#### C.2. Hyperparameters

To facilitate reproducibility, we group the parameters by their corresponding stage in the ScalingAR pipeline: (i) constructing the confidence state, corresponding to Section §4.1, and (ii) executing the guidance policies, corresponding to Section §4.2. The sensitivity analysis of key hyperparameters is placed in Section §D.2.

###### C.3. Algorithm Workflow We conclude the overall algorithm workflow of ScalingAR in Algorithm 1.

Notation Parameter Definition Value Rationale / Function Dual-Channel Confidence Profile αH/αM Token-Level Weights 0.5/0.5 Robust composite: combines Entropy (uncertainty) with Margin (decisiveness). λtok, λI Smoothing Factors 0.2 Standard EMA for noise filtering in score tracking. wtok/wblk Intrinsic Weights 0.65/0.35 Balances token-wise entropy with spatial block stability. b Spatial Block Size 4 × 4 Grid resolution for detecting localized failure spots. q Worst-Case Quantile 0.1 Focuses metric on the most unstable regions (weakest link). wI/wD Channel Fusion 0.75/0.25 Balances intrinsic stability with conditional alignment. Confidence-Guided Policies

p Pruning Quantile 0.2 Adaptive threshold: targets relative bottom-20% of batch. λθ Threshold Update 0.2 Update rate for tracking the batch’s confidence distribution. W0 Warm-up Period 12.5% Initial generation phase before pruning activation.

∆rec Recovery Window 32 steps Hysteresis window to prevent premature termination. δrec Recovery Threshold 0.05 Minimum confidence rebound required to trigger recovery. α, β Guidance Gains 0.3, 0.4 Response gain for Drift Correction and Volatility Dampening. λ Diversity Gain 0.4 Response gain for Rebound-based guidance relaxation.

#### Algorithm 1 ScalingAR Workflow

- 1: Input: Prompt y, Model pθ, Config C (weights, smoothing, quantiles). Output: Image xˆ
- 2: Initialize candidates S0 ← {∅} and warm-up threshold θ↓.
- 3: for t ← 1 to T do
- 4: if St−1 = ∅ then
- 5: break
- 6: end if
- 7: // Stage I: Signal Extraction (Batch Update)
- 8: Compute logits ℓc, ℓu and raw signals (Itraw, Kt) for all s ∈ St−1
- 9: Update State St(s) ← {It, Dt, Ct, Rt} using EMA and Rolling Stats
- 10: // Stage II: Dynamic Guidance & Sampling
- 11: for all s ∈ St−1 do
- 12: Calculate adaptive scale st ← sbase + α(1 − Dt) + βVar(I) − γRt
- 13: Sample xt ∼ softmax(ℓu + st · (ℓc − ℓu)) and append to s
- 14: end for
- 15: // Stage III: Adaptive Termination
- 16: Update global threshold θ↓ ← EMA(θ↓, Quantilep({Ct(s)}))
- 17: Prune s if Ct(s) < θ↓ and Rt < δrec (not recovering)
- 18: St ← survivors after pruning
- 19: end for
- 20: return Decode(sˆ), where sˆ ← arg maxs∈ST CT(s)

### D. More Experimental Settings and Analysis

- D.1. More Details of Experimental Settings Captions of Figure 1. For qualitative results in Figure 1 (Top), we further detail the prompts here:

- • 1st: “A red rose in full bloom sits on the top, above a pink rosebud.”
- • 2nd: “A photo of a cute puppy playing in a sunny backyard.”
- • 3rd: “A young boy holding a mysterious key, embarking on an adventure through various landscapes to find hidden treasure.”
- • 4th: “A masked hero jumping from a rooftop, comic book style with bold outlines and dialogue bubbles.”
- • 5th: “A close-up of an anime woman’s face with a shocked expression, featuring dark hair, drawn in the anime style. The image showcases colorful animation stills, close-up intensity, soft lighting, a low-angle camera view, and high detail.”

Baseline Setup. We benchmark against Importance Sampling (IS) and Best-of-N (BoN) following the protocols established in TTS-VAR (Chen et al., 2025b). We utilize ImageReward (Xu et al., 2023) as the external preference model due to its

alignment with human aesthetics, and set the primary candidate sample size to N = 8. For both baselines, the CFG scales are maintained at the default values of the respective base models to ensure a controlled comparison.

Notably, while BoN is often theoretically regarded as a performance upper bound via exhaustive selection, our method consistently surpasses it. We attribute this to the fundamental difference between passive selection and active rectification. BoN restricts itself to selecting from static trajectories and is therefore bound by the quality of the external reward proxy. In contrast, our approach employs closed-loop control, dynamically modulating guidance pressure to actively stabilize generation. This allows the model to correct intrinsic drift and access superior regions of the generation manifold that are unreachable by static decoding strategies, regardless of the selection budget.

Robustness Testing. To evaluate the robustness of ScalingAR, we further employ prompts from IPV-TXT from Impossible Videos [ICML’25] (Bai et al., 2025). Specifically, we filtered prompts suitable for image generation from IPV-TXT, then employed Impossible Prompt Following (IPF) as the evaluation metric, which measures the alignment between generated images and the semantic intent of impossible prompts. Following (Bai et al., 2025), we employed GPT-4o to perform binary judgments on each image based on prompt adherence. For qualitative results in Figure 6 (Right):

- • 1st: “A sheep peacefully grazing in a realistic meadow suddenly defies gravity as its wool expands dramatically, causing its body to balloon up like a cotton cloud. The fluffy animal then lifts off from the grassy field and drifts upward into the blue sky, its transformed woolly coat acting like a natural balloon.”
- • 2nd: “A commercial aircraft inexplicably takes off from the ocean’s surface as if the water were a solid runway, defying physics in this photo-realistic scene. The calm, glassy sea appears to have transformed into a firm platform, allowing the plane to accelerate and lift off smoothly, with spray trailing behind its wheels like it would on a wet tarmac.”

User Study. We conducted a user study to evaluate human preferences using the mean opinion score (MOS) metric. We designed a user-friendly interface to facilitate the evaluation process and collected feedback from a total of 15 volunteer participants. The detailed instructions provided to the participants are as follows:

User Study: Autoregressive Image Generation

Thank you for participating in our user study! Please follow these steps to complete your evaluation:

- 1. Image Generation: Carefully read the target prompt provided, and then view the provided images.
- 2. Scoring Criteria: Assign a score to each generated image based on the following aspects (1 being the lowest, 5 being the highest):

- • Overall Quality: The overall perceived quality and appeal of the generated image.
- • Aesthetic Quality: The visual aesthetics, composition, and artistic merit of the image.
- • Realism Fidelity: How realistically and faithfully the image captures the intended scene or subject matter.
- • Semantic Alignment: How well the generated image aligns with and represents the meaning of the textual prompt.
- • Attribute Binding: The degree to which the image accurately depicts the specific attributes and details described in the text.

- 3. Submission: Click the “Submit Scores” button to submit your scores. Notations:

- 1. We observe that the edge browser is not fully compatible with our interface. Chrome is recommended.
- 2. Remember to click the “Submit Scores” button after your evaluation.
- 3. If you see that images and the score sliders are not aligned, shrinking your page usually works.
- 4. If the page is not responsive for a long time, please try to refresh it.
- 5. If you have any questions, please directly ping us. Thank you for your time and effort!

#### D.2. More Analysis

Analysis of Ablation on Policy Level. While ablation study (Table 2) in main text focuses on the Profile Level, we conducted additional ablation studies to evaluate the contributions of the Policy Level, which builds upon the Profile Level,

Table 4. Ablation of Policy Level.

as shown in Table 4. (i) The “Termination Only” setup improves performance across all metrics, highlighting its ability to prune lowconfidence trajectories and mitigate failure modes, ensuring stable generation. (ii) The “Scheduler Only” setup also yields notable gains, demonstrating its effectiveness in dynamically modulating conditioning strength to balance semantic alignment and diversity. (iii) Integrating both mechanisms achieves the best results, showing their complementary roles in improving generation quality and efficiency. These results validate the Policy Level as essential for enhancing autoregressive image generation.

Method Basic↑ Advanced↑ Design↑ Overall↑

LlamaGen 49.6 40.4 40.3 40.4 + Termination Only 54.1 43.1 42.2 45.1 + Scheduler Only 53.6 42.0 41.0 43.8 + ScalingAR (Ours) 57.4 44.1 42.5 46.5

Computational Efficiency Analysis. We profile the computational overhead of ScalingAR against the base model and the BoN on GenEval, using NVIDIA H200 GPUs. Detailed statistics are reported in Table 5.

Table 5. Computation consumption comparison on GenEval with NVIDIA 140G H200 GPU.

Method N Per-step WC (s) Overall WC (s) Matched Tokens/Img FLOPs (TFLOPs) Memory (GB) Performance

LlamaGen 1 0.024 24.93 1024 5.60 6.44 0.32 + BoN 8 0.025 218.44 8192 39.12 48.72 0.15 + ScalingAR (Ours) 8 0.029 69.56 2350 4.23 18.16 0.36

- • Wall Clock Latency: Standard test-time scaling (BoN, N = 8) incurs a prohibitive latency penalty, increasing overall wall clock (WC) time to 218.44s. In contrast, ScalingAR effectively decouples sample width from latency. By actively pruning trajectories, it reduces the overall WC time to 69.56s, achieving a 3.1× speedup over BoN while maintaining superior generation quality.
- • Resource Consumption: The efficiency gains are further evidenced in compute and memory usage. BoN requires generating 8,192 tokens per instance, resulting in massive computational demand (39.12 TFLOPs) and high peak memory usage (48.72 GB). Conversely, ScalingAR processes only 2,350 tokens on average, i.e., a 71% reduction in token consumption compared to BoN. This aggressive pruning translates directly to a dramatically lower compute footprint (4.23 TFLOPs) and reduced memory overhead (18.16 GB), making test-time scaling feasible for resourceconstrained environments.

Analysis of Global Confidence & Guidance Weights. Figure 9 presents a detailed analysis of the impact of weights of unified confidence and guidance scheduler on the performance of ScalingAR on the TIIF-Bench. ❶ Unified Confidence (Figure 9 (Top)): Varying the balance between the Intrinsic (wI) and Conditional (wD) channels shows that emphasizing the Intrinsic channel slightly (wI/wD = 0.75/0.25) achieves the best TIIF-Bench performance across all subsets. This highlights the importance of capturing local uncertainty and stability while maintaining semantic alignment. Omitting the Conditional Channel (1.00/0.00) degrades performance, confirming its complementary role. ❷ Guidance Scheduler (Figure 9 (Bottom)): Adjusting the weights α, β, and λ, which control conditional utilization, intrinsic volatility, and confidence rebound, respectively, reveals that moderate emphasis on intrinsic volatility and rebound (β,λ) improves performance. The weight α peaks at 0.3, suggesting overemphasis may reduce diversity. This confirms the need for balanced, dynamic guidance to optimize semantic fidelity and diversity.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Figure 9. Analysis of weights of Unified Confidence (Top) and Guidance Scheduler (Bottom).

[Figure 113]

[Figure 114]

Analysis of Adaptive Termination Gate. We further analyze the impact of the confidence threshold quantile p and the recovery threshold δrec on the performance and token efficiency of ScalingAR, as illustrated in Figure 10. ❶ Confidence Threshold (Figure 10 (Left)): The choice of confidence threshold critically balances pruning aggressiveness and generation quality. Setting p too low leads to insufficient pruning, resulting in higher token consumption with limited accuracy gains. Conversely, an overly high threshold causes

Figure 10. Analysis of thresholds of Confidence (Left) and Recovery (Right).

premature termination of promising trajectories, degrading accuracy despite lower token usage. Our experiments show that an intermediate threshold (e.g., p = 0.20) achieves the best trade-off, significantly improving accuracy while maintaining efficient token consumption compared to both baseline and extreme settings. ❷ Recovery Threshold (Figure 10 (Right)): The recovery mechanism safeguards against false positives by allowing trajectories to rebound from transient confidence dips. Disabling this mechanism leads to noticeable performance drops, highlighting its necessity. Furthermore, setting the recovery threshold δrec too low or too high adversely affects accuracy and efficiency: a low threshold permits premature recovery of poor trajectories, increasing token cost, while a high threshold delays recovery, risking early termination of viable samples. An optimal value (e.g., δrec = 0.05) balances these effects, maximizing accuracy with minimal token overhead.

Analysis of Local Confidence Weights. We further analyze the impact of the weighting strategies for Token-Level Confidence αH/αM and Worst-Block Stability wtok/wblk on the performance of ScalingAR, as illustrated in Figure 11. ❶ Token-Level Confidence Weights (Figure 11 (Left)): Adjusting the balance between entropybased uncertainty (αH) and margin-based confidence (αM) reveals that prioritizing entropy signals (αH/αM = 0.7/0.3) achieves the best overall performance across all metrics. This suggests that entropy provides a more robust signal for capturing localized instability during generation. Conversely, overemphasizing margin-based confidence (αH/αM = 0.3/0.7) leads to performance degradation, particularly in advanced and designer subsets, as it fails to fully capture nuanced instability patterns. A balanced setting (αH/αM = 0.5/0.5) offers a reasonable trade-off, though slightly underperforms the optimal configuration. ❷ Worst-Block Stability Weights (Figure 11 (Right)): Varying the balance between token-level confidence (wtok) and block-level stability (wblk) shows that an emphasis on token-level signals (wtok/wblk = 0.85/0.15) slightly reduces performance, particularly in the advanced and designer subsets, as it underweights spatial anomalies that propagate into global failures. On the other hand, overemphasizing block-level stability (wtok/wblk = 0.35/0.65) also degrades results, as it may overreact to localized noise. The optimal configuration (wtok/wblk = 0.65/0.35) balances token-level and block-level signals effectively, achieving the highest scores across most metrics.

[Figure 115]

[Figure 116]

[Figure 117]

Figure 11. Analysis of weights of Token-Level Confidence (Left) and WorstBlock Stability (Right).

- E. Exhibition Board We provide more comparison results here in Figure 12 on AR-GRPO and Figure 13 on LlamaGen.
- F. Limitation and Future Works

ScalingAR pioneers test-time scaling for autoregressive image generation but faces key challenges. AR image modeling involves complex dependencies, making confidence estimation difficult; our exploration of token entropy is a first step but may not fully capture uncertainty and semantic alignment. Additionally, the approach relies on model calibration and entropy signals, which can vary with training and architecture. Future work includes developing finer-grained confidence measures for more precise scaling, and integrating entropy-based signals into both training-time and test-time to create a more unified pipeline.

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

###### AR-GRPO+ScalingAR(Ours)

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

The colorful flowers bloomed next to the sleek black fence

The rectangular painting was hung above the beige couch

The black mug was on top of the white coaster

The sharp red chili pepper contrasted with the soft green lettuce

The soft white snow covered the rough brown ground

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

###### AR-GRPO+ScalingAR(Ours)

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

The prickly green cactus contrasted with the smooth blue sky

The fluffy pillow was on top of the hard couch

The warm yellow light shone down on the cool blue ocean waves

The colorful flowers contrasted with the dull grey wall

The soft white clouds floated above the deep blue ocean

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

###### AR-GRPO+ScalingAR(Ours)

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

The bright red rose contrasted with the dull grey pavement

The smooth black leather chair sat in front of the prickly green plant

The sleek, aerodynamic shape of the speedboat sliced through the water, a thrilling ride of power and speed

The sharp, angular lines of the modernist sculpture were a striking statement of contemporary art

The vibrant, swirling colors of the aurora borealis danced across the night sky, a natural light show of wonder and awe

Figure 12. More results demonstrations of ScalingAR on AR-GRPO (Yuan et al., 2025).

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

###### LlamaGen+ScalingAR(Ours)

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

The blue bowl was on top of the white placemat

The smooth black river flowed next to the tall green trees

The striped black and white cat lay next to the soft grey blanket

The soft pink petals of the cherry blossom contrasted with the rough brown bark

The striped rug was on top of the wooden floor

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

###### LlamaGen+ScalingAR(Ours)

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

The soft pink blanket draped over the hard wooden chair

The flickering candle flame danced on the smooth wax and the textured holder

The sharp black cat clawed at the soft red blanket

A photo of a wise old owl The crunchy brown leaves

covered the damp grey sidewalk

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

###### LlamaGen+ScalingAR(Ours)

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

The striped black and white zebra grazed near the tall green tree

The delicate butterfly fluttered near the fragrant flower and the rustling leaves

The sleek black cat sat on top of the fluffy white pillow

The graceful swan glided across the calm lake and the reedy marsh

A gentle giant tending to a lush garden, nurturing magical plants that bloom with vibrant colors and unique powers

Figure 13. More results demonstrations of ScalingAR on LlamaGen (Sun et al., 2024).

