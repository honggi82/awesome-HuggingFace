# arXiv:2604.26951v1[cs.CL]29Apr2026

## Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models

###### Gongbo Zhang1 Wen Wang2 Ye Tian1 Li Yuan1,∗ 1Peking University 2Zhejiang University ∗Corresponding author: yuanli-ece@pku.edu.cn

Code: GitHub Models: Hugging Face Data: Hugging Face

#### Abstract

Diffusion large language models (dLLMs) offer parallel decoding and bidirectional context, but state-of-the-art dLLMs require billions of parameters for competitive performance. While existing distillation methods for dLLMs reduce inference steps within a single architecture, none address cross-architecture knowledge transfer, in which the teacher and student differ in architecture, attention mechanism, and tokenizer. We present TIDE, the first framework for cross-architecture dLLM distillation, comprising three modular components: (1) TIDAL, which jointly modulates distillation strength across training progress and diffusion timestep to account for the teacher’s noise-dependent reliability; (2) COMPDEMO, which enriches the teacher’s context via complementary mask splitting to improve predictions under heavy masking; and (3) Reverse CALM, a cross-tokenizer objective that inverts chunk-level likelihood matching, yielding bounded gradients and dual-end noise filtering. Distilling 8B dense and 16B MoE teachers into a 0.6B student via two heterogeneous pipelines outperforms the baseline by an average of 1.53 points across eight benchmarks, yielding notable gains in code generation, where HumanEval scores reach 48.78 compared to 32.3 for the AR baseline.

#### 1 Introduction

Autoregressive language models dominate natural language generation (Vaswani et al., 2017; Yang et al., 2024; 2025), yet diffusion language models have gained traction as an alternative paradigm. Earlier works, such as D3PM (Austin et al., 2021a) and MDLM (Sahoo et al., 2024), explore the basic training architectures. Recent works, such as LLaDA (Nie et al., 2025) and Dream (Ye et al., 2025), scale model size to that of large language models. For example, recent work, LLaDA 2.0 (Bie et al., 2025), scales the model size to 100B, achieving a new state-of-the-art performance. Compared with autoregressive models, dLLMs iteratively denoise a fully masked sequence, enabling parallel decoding and bidirectional context. However, competitive dLLMs require 8B-16B or even more parameters and computation costs (Nie et al., 2025; Liu et al., 2025; Bie et al., 2025; Ye et al., 2025), posing a barrier to deployment (Figure 1).

One simple solution for developing stronger small models is knowledge distillation (Hinton et al., 2015), which compresses large models into smaller ones. For autoregressive LMs, methods such as MiniLLM (Gu et al., 2024), DistiLLM (Ko et al.), GKD (Agarwal et al., 2024), and TAID (Shing et al., 2025) are well established, with various designs, including new Kullback-Leibler divergence losses, leveraging the teacher model’s feedback, etc. For dLLMs, however, existing methods—CDLM (Kim et al., 2025), DDD (Hayakawa et al., 2024), LSD (Fu et al., 2025), and SDTT (Deschenaux & Gulcehre, 2024)—focus exclusively on step compression within a single architecture, leaving cross-architecture distillation unexplored. This setting introduces three fundamental challenges. First, due to the random sampling of timesteps during training, the teacher’s reliability fluctuates drastically across the diffusion process, leading to inconsistent temporal dynamics. Second, severe masking at high

(a) Prior: Step Distillation vs (b) Ours: Cross-Architecture Distillation

Distill

[Figure 1]

+

Input Output

16B

| | |
|---|---|
| | |

Cross-Tokenizer Rev.CALM

MoE Teacher

Shared Tokenizer

0.6B Student

+ +

8B

Input model

Output 8B → 8B model

Tokenize TIDAL CompDemo 16B / 8B → 0.6B

Dense Teacher

Same model, fewer steps Different architectures, smaller student

| |Code Generation (HumanEval)|
|---|---|
| |+16.5↑ 48.8|
| |AR Distilled<br><br>32.3<br><br>|

) Overall (8 Tasks) Deployment

+1.5↑ 22x Less Memory

1.4 GB vs 31.3 GB

34.2 32 32.7

5x Faster

30

41.0 vs 8.2 tokens/s

BD3LM Distilled

- Figure 1: Cross-architecture distillation for dLLMs. Compared to prior step distillation (a) that retains the original model size, the TIDE framework (b) distills heterogeneous 16B MoE and 8B dense teachers into a 0.6B student. The distilled model achieves a +16.5 gain on HumanEval over the AR baseline, 22× memory reduction, and 5× faster inference.

noise levels greatly reduces available context, making the raw output of the teacher too uninformative to transfer rich spatial representations. Third, distinct tokenizer vocabularies render standard token-level likelihood objectives mathematically inapplicable. In this work, we present TIDE (Figure 2), the first unified framework for cross-architecture dLLM distillation designed to comprehensively overcome the aforementioned temporal, spatial, and vocabulary barriers. Rather than treating these challenges in isolation, TIDE integrates three synergistic components that orchestrate an end-to-end learning pipeline:

- • Scheduling Level (TIDAL): To resolve temporal inconsistencies, TIDAL dynamically modulates the distillation strength along both the training progress and diffusion timestep axes. It acts as the pacemaker of the framework, ensuring the student selectively learns from the teacher only when the teacher’s timestep-dependent signals are highly reliable.
- • Contextual Level (COMPDEMO): Operating within this scheduled process, COMPDEMO overcomes the context scarcity caused by heavy masking at high noise levels. It enriches the teacher’s signals via complementary mask splitting, providing the student with demonstration-conditioned targets that enable robust spatial knowledge transfer.
- • Output Level (Reverse CALM): Finally, to map the enriched contextual knowledge into the student’s output space, Reverse CALM overcomes the cross-tokenizer barrier. By inverting chunk-level likelihood matching, it avoids the instability of direct token mapping, achieving bounded gradients and dual-end noise filtering.

Collectively, these three modules constitute an integrated framework: TIDAL controls when to learn across timesteps, COMPDEMO determines what contextual information to enrich, and Reverse CALM specifies how to project this knowledge across distinct vocabularies.

We validate TIDE across two heterogeneous pipelines: (A) cross-tokenizer distillation from a 16B MoE teacher (LLaDA2 (Bie et al., 2025)) and (B) shared-tokenizer distillation from an 8B dense teacher (WeDLM (Liu et al., 2025)), both into a 0.6B block diffusion student (BD3LM (Arriola et al., 2025)). The best configuration improves the non-distilled baseline by +1.53 on the eight-benchmark average (34.20 vs. 32.67), with distilled dLLMs excelling at code generation (HumanEval 48.78 vs. 32.3 for the same-size AR model). Ablations confirm that each pipeline favors a distinct strategy, validating the modular design.

Byte-Level Align

|𝑥|
|---|

x No sé ya ni la …

Axis 2: Training Process

| |Axis 1: Diffusion Timestep<br><br>|
|---|---|
| |𝜆𝑡|
| |𝝀𝒕 = 𝝀𝒕𝒓𝒂𝒊𝒏<br><br>|Interpolated Target 𝒓𝒕|
|---|
<br><br>Linear Schedule|

Noise

Teacher Tokens (𝑉𝑇)

I don ‘t wanna

|𝑥𝑡|
|---|

| |[M]|[M]| | |[M]|[M]|…|
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Student Tokens (𝑉𝑆)

I do n‘t wan na Aligned Chunks

|𝑀 =|
|---|

Randomly partition of( )

𝑀𝐴 𝑀𝐵

- Pass 1 (reveal 𝑀𝐴)

fa ni wt fa ni [M]

- Pass 2 (reveal 𝑀𝐵)

𝑙𝑝𝑡 𝑙𝑝𝑠

1 0 0 0 0 1 1 0 0 0 0 1

0

|1|0| |0| |0|
|---|---|---|---|---|---|
|0|1| |1| |0|
|0|0|1|0| |1|

𝑡(1) at 𝑀𝐵

𝜆𝑡r𝑎𝑖𝑛

- 0

- 1

fa ni wt fa [M] wt

Cosine Schedule

×

×

Logits

|×|
|---|

at 𝑀𝐴& 𝑀𝐵

𝐿𝑃𝑇

𝐿𝑃𝑆

Reverse CALM Objective 𝐁𝐂𝐄(𝒑𝒔𝒄 ∥ 𝒑𝒕𝒄)

|× (𝟏 − 𝒕)| |
|---|---|
| | |

𝑡(2) at 𝑀𝐴

| | | |
|---|---|---|
| |𝑝𝑡𝑐| |
| | | |

𝑝𝑠𝑐

|Interpolated Target 𝒓𝒕|
|---|

d

[Figure 2]

(1) TIDAL (2) CompDemo (3) Reverse CALM

Teacher dLLM 8B/16B (e.g., WeDLM or LLaDA2)

Student dLLM 0.6B (e.g., Qwen-BD3LM)

TIDE

Cross-Architecture, Cross-Tokenizer

- Figure 2: Overview of the TIDE framework, transferring knowledge from a large teacher to a 0.6B student via three modular components: (1) TIDAL for dual-axis interpolation,

(2) COMPDEMO for complementary teacher demonstration, and (3) Reverse CALM for crosstokenizer alignment.

Our primary contributions are:

- • We introduce TIDE, the pioneering cross-architecture knowledge distillation framework for dLLMs, specifically designed to overcome challenges in heterogeneous transfer such as varying timestep reliability, mismatched attention, and distinct tokenizers.
- • We propose three novel, modular components to enable this transfer: TIDAL for dualaxis, timestep-aware distillation scheduling, COMPDEMO to enrich teacher signals via complementary masking, and Reverse CALM for robust cross-tokenizer alignment.
- • Experiments across eight benchmarks and two pipelines show that cross-architecture dLLM distillation is effective. Ablations confirm that each component contributes and that each pipeline favors a distinct configuration.

#### 2 Method

TIDE (figure 2) addresses the problem of distilling a large teacher dLLM fT (parameterized by θT) into a smaller student dLLM fS (parameterized by θS), where the teacher and the student may differ in foundational model architecture, attention mechanism, and tokenizer vocabulary. Let x = (x1, . . . , xL) denote a clean token sequence and xt denote the noised version at a diffusion timestep t ∈ [ϵ,1), where the positions within the mask set M are replaced with a special [MASK] token. The student is trained to predict the clean tokens at the masked positions, guided by both the ground-truth labels and the teacher’s predicted categorical token distribution.

##### 2.1 Time-Iteration Dual-Axis Lambda Modulation

Knowledge distillation for dLLMs encounters a unique challenge absent in autoregressive (AR) distillation: the reliability of the teacher signal varies with the diffusion timestep. At low masking ratios (t ≈ 0), the teacher observes nearly the entire sequence and produces highly reliable predictions. Conversely, at high masking ratios (t ≈ 1), even a large teacher model primarily relies on guessing. Furthermore, the student model’s capacity to absorb knowledge from the teacher evolves throughout training. These two phenomena motivate

a dual-axis scheduling approach that jointly modulates distillation strength along both the diffusion timestep and the training progress.

- Axis 1: Diffusion timestep. To account for the timestep-dependent reliability of the teacher, we modulate the interpolation coefficient λt according to the current diffusion timestep t:

λt = λtrain × (1 − t), (1)

where λtrain denotes a base coefficient determined by the training progress (defined below). At high noise levels (t ≈ 1), λt ≈ 0, and the target is dominated by the student, thereby avoiding the distillation of unreliable signals from the teacher. At low noise levels (t ≈ 0), λt ≈ λtrain, and the student fully relies on the teacher. This axis is unique to dLLMs; in AR models, the teacher consistently maintains access to the full left context, which renders predictions uniformly reliable and eliminates the need for timestep-dependent modulation.

- Axis 2: Training progress. The base coefficient λtrain follows a cosine schedule over the normalized training progress p ∈ [0,1]:

- 1

- 2

(1 − cos(π · p)) , (2)

λtrain = λinit + (λmax − λinit) ×

where λinit and λmax denote hyperparameters, with default values typically set to 0.1 and 0.9. In the initial phases of training, λtrain ≈ λinit, which ensures that the target is dominated by the student model to prevent representation collapse. As training progresses to the later stages, λtrain approaches λmax, thereby shifting the learning objective toward comprehensive teacher supervision.

A similar interpolation-based distillation approach has been proposed in TAID (Shing et al., 2025) for AR models. However, TAID operates along a single axis (training progress only) and does not account for the teacher’s timestep-dependent reliability, which is specific to the dLLM setting. Our Time-Iteration Dual-Axis Lambda modulation (TIDAL) extends this principle by incorporating a novel diffusion-timestep axis.

Interpolated target and loss. Given the student logits s and the teacher logits t at the masked positions, the interpolated target is formulated as:

(1 − λt) · s + λt · t T

, (3)

rt = softmax

where T denotes the temperature parameter. To prevent gradient flow through the mixed target, the interpolated target rt is detached from the computation graph. Consequently, the TIDAL loss is defined as:

##### s

T × T2. (4) To maintain memory efficiency, this loss is computed exclusively at the masked positions. Optionally, a midrange timestep weighting w(t) = exp −(t−2σ0.52 )2 with σ = 0.15 is applied to emphasize the most informative timesteps.

LTIDAL = DKL rt softmax

##### 2.2 Complementary Demonstration

In standard dLLM distillation, the teacher model receives the identical masked input xt as the student model. At high masking ratios, the limited context causes the teacher to produce noisy predictions, thereby degrading the quality of the distillation signal. To address this limitation, we propose Complementary Demonstration-Conditioned Denoising (COMPDEMO), which leverages the mask structure to enrich the context provided to the teacher. This mechanism is specific to dLLMs and lacks an equivalent in AR distillation.

Motivation. In dLLMs, the teacher must denoise under heavy masking, producing noisy predictions at high masking ratios. Shenfeld et al. (2026) show that demonstration-conditioned teachers yield better training signals in the AR setting. Discrete diffusion provides a natural analog: revealing a subset of masked tokens shifts the teacher to a lower effective timestep, improving its predictions. We partition the masked positions into two complementary

subsets; each subset serves as additional context for one of two teacher forward passes, so every masked position benefits from enriched context.

Mask splitting. Given the set of masked positions M, we randomly partition this set into two complementary subsets, MA and MB, such that:

MA ∪ MB = M, MA ∩ MB = ∅, |MA|/|M| ≈ ρ, (5) where ρ = 0.5 represents the demonstration ratio. Two-pass teacher inference. We perform two forward passes through the frozen teacher model:

- Pass 1: t(1) = fT(reveal MA,mask MB) → logits at MB, (6)
- Pass 2: t(2) = fT(reveal MB,mask MA) → logits at MA.

In Pass 1, positions in MA retain clean tokens as demonstration context while MB remains masked; Pass 2 is symmetric. The merged logits are tfinal[MB] ← t(1)[MB] and tfinal[MA] ← t(2)[MA].

Cost analysis. COMPDEMO doubles the teacher’s forward passes, but since the teacher is frozen (no gradient computation), overall training time increases by approximately 50%.

##### 2.3 Distillation Objectives

TIDE supports two distillation pipelines, which depend on the tokenizer compatibility between the teacher and student models. For each pipeline, we design a tailored objective that accounts for the granularity of alignment.

Shared-tokenizer objective (WeDLM → BD3LM). When the teacher and student models share an identical tokenizer family (Liu et al., 2025), the distributions at the token level are directly comparable. We apply the TIDAL loss (section 2.1) by employing KL divergence at the token level, combined with the optional COMPDEMO (section 2.2):

LB = LCE + wtidal · LTIDAL. (7)

Cross-tokenizer objective (LLaDA2 → BD3LM). When the teacher and the student employ different tokenizers (VT ̸= VS),the token-level KL divergence remains undefined due to vocabulary misalignment. To address this limitation, we introduce Chunk-level Approximate Likelihood Matching (CALM), which adapts ALM (Minixhofer et al., 2025) for the dLLM setting.

Chunk alignment. Using tokenkit (Minixhofer et al., 2024; 2025), we align the two token sequences at the byte level to identify chunks—the minimal text spans that contain one or more complete tokens from each vocabulary. Let C denote the total number of aligned

chunks. We construct binary alignment matrices AS ∈ {0,1}LS×C and AT ∈ {0,1}LT×C, where [AS]i,c = 1 if and only if the student token i is assigned to chunk c. Given that the teacher and student models use distinct chat templates with incompatible special tokens, we restrict the alignment process to content tokens only. We exclude template-specific markup to prevent the formation of spurious cross-tokenizer chunks.

Chunk-level log-probabilities. For each token xi, the log-probability is computed as log P(xi) = logitsx

− logsumexp(logits) to avoid the materialization of the full [b, L,V] softmax matrix. The chunk-level log-probabilities are subsequently obtained through matrix multiplication:

i

LPS = lpS · AS ∈ Rb×C, LPT = lpT · AT ∈ Rb×C, (8) where lpS and lpT denote the per-token log-probability vectors. The chunk probabilities are then derived via temperature scaling: pcs = exp(LPcS/T) and pct = exp(LPcT/T). Forward CALM. A natural baseline involves the application of a forward (mode-covering) binary cross-entropy loss on the chunk probabilities at masked positions:

LFwd-CALM = − [pct log pcs + (1 − pct) log(1 − pcs)] . (9)

Note that CALM operates on scalar chunk probabilities pc ∈ [0,1], so BCE is the appropriate loss and the forward/reverse analysis differs from the KL divergence used in token-level distillation.

The forward CALM objective can be further integrated with the progressive curriculum of TIDAL by performing interpolation within the chunk probability space:

pmix = (1 − λt) · pcs + λt · pct, LCALM-TIDAL = − [pmix log pcs + (1 − pmix) log(1 − pcs)] .

(10)

Limitations of forward CALM. The forward BCE gradient contains a ratio pct/pcs. When pct → 1 but pcs → 0, which is common under imperfect cross-tokenizer alignment, this ratio diverges, causing gradient explosion. The 1/pcs term also amplifies noise from misaligned chunks indiscriminately.

Reverse CALM. To address these limitations, we propose Reverse CALM, which reverses the direction of the BCE loss:

LRev-CALM = − [pcs log pct + (1 − pcs) log(1 − pct)] . (11) Swapping pcs and pct makes the gradient coefficient log p

c t

1−pct , which depends only on the fixed teacher and is bounded. This also provides dual-end noise filtering: poorly aligned chunks (pct ≈ 0.5) zero the coefficient, while low pcs suppresses noise via small ∂pcs/∂θ. Reverse CALM is equivalent to minimizing the Bernoulli KL KLBern(pcs∥pct), a mode-seeking objective in scalar space (Appendix C). Since TIDAL targets the instability of forward objectives, it is counterproductive for reverse CALM and is not applied (Appendix C).

Training objectives. The cross-tokenizer pipeline combines cross-entropy with a distillation loss:

LA = LCE + wcalm · Ldist, where Ldist ∈ {LCALM-TIDAL, LRev-CALM}. (12) Both losses are computed at masked positions. When COMPDEMO is enabled, teacher logits are replaced by the merged two-pass logits (section 2.2).

#### 3 Experiments

##### 3.1 Experimental Setup

Models. The student model is Qwen3-0.6B-BD3LM, a 0.6B-parameter block diffusion language model initialized from Qwen3-0.6B-Base (Yang et al., 2025) and obtained from Zhou et al. (2026). Following the BD3LM (Arriola et al., 2025) framework, the model takes the concatenation of [xt,x0] with a specialized attention mask during training and uses block diffusion with bidirectional attention for inference. We distill from two heterogeneous teachers: (A) LLaDA2.0-mini (Bie et al., 2025), an MoE dLLM with an independent tokenizer derived from the Ling series (Team et al., 2025); and (B) WeDLM-8B-Instruct (Liu et al., 2025), an 8B dense causal dLLM initialized from Qwen3-8B-Base(Yang et al., 2025).

Training. All experiments use a learning rate of 5e-5, 10 training epochs, and bfloat16 precision. Following the training recipe of Zhou et al. (2026), we combine four SFT datasets: Tulu-3 SFT Mixture (Lambert et al., 2024), SmolTalk (Allal et al., 2025), and OpenCoder OPC-SFT Stage 1 and Stage 2 (Huang et al., 2025). The student model’s sequence length is set to 512 tokens, with a block size of 32. Complete hyperparameter settings are provided in Appendix B.

Evaluation. We evaluate across eight benchmarks spanning reasoning (GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), BBH (Suzgun et al., 2023)), knowledge (MMLUPro (Wang et al., 2024), MMLU (Hendrycks et al., 2021a)), commonsense (HellaSwag (Zellers et al., 2019)), and code generation (HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021b)). Inference and evaluation hyperparameters follow Zhou et al. (2026); task-specific configurations are detailed in Appendix B.

Baselines. We adopt the baselines and their reported results from Zhou et al. (2026): (1) the AR model Qwen3-0.6B-Base, which shares the same architecture and tokenizer as the student

Table 1: Main results across eight benchmarks. All distillation methods include a crossentropy loss term. Bold: best among dLLM models; underline: second best.

Qwen3-0.6B Shared-Tokenizer Cross-Tokenizer

Benchmark AR BD3LM KL TIDE-Cross TIDE-Shared CALM TIDE-Shared TIDE-Cross

GSM8K 59.60 45.56 43.97 45.03 48.98 48.60 49.89 52.24 MATH 32.40 13.08 9.40 9.76 11.16 13.14 12.98 13.20 BBH 41.50 26.32 25.79 26.00 26.79 24.21 26.85 27.37 MMLU-Pro 24.70 13.80 13.19 12.88 14.48 13.47 14.02 14.52 HellaSwag 47.40 39.28 39.78 39.50 40.50 40.42 39.57 39.88 MMLU 52.80 39.15 39.57 39.09 39.92 39.42 39.54 39.59 HumanEval 32.30 46.34 41.46 42.68 48.78 43.90 49.39 48.17 MBPP 36.60 37.80 31.20 31.40 37.80 34.80 38.40 38.60

Avg 40.91 32.67 30.55 30.79 33.55 32.25 33.83 34.20

prior to block diffusion conversion; and (2) the undistilled BD3LM (Arriola et al., 2025), derived from Qwen3-0.6B-Base by fine-tuning on the same dataset, which serves as the direct non-distilled reference.

##### 3.2 Main Results

- Table 1 presents the evaluation results of two distinct distillation pipelines. The sharedtokenizer pipeline distills WeDLM-8B-Instruct into the Qwen3-0.6B-BD3LM model using token-level KL divergence as the baseline objective. In contrast, the cross-tokenizer pipeline distills LLaDA2.0-mini into the identical student model using CALM as the baseline objective. Within the TIDE framework, each pipeline is evaluated under two complementary strategies. Specifically, TIDE-Shared applies TIDAL and COMPDEMO to enhance signal quality through progressive scheduling and enriched teacher logits. Meanwhile, TIDE-Cross adopts a modeseeking optimization direction via Reverse CALM (or Reverse KL in the shared-tokenizer setting). To assess generalizability, we further evaluate each strategy within the non-native pipeline.

Cross-Architecture Distillation Is Effective. Both TIDE pipelines consistently outperform the non-distilled BD3LM baseline (with an average score of 32.67). The cross-tokenizer pipeline, utilizing the native TIDE-Cross strategy, achieves the highest average score of 34.20, while the shared-tokenizer pipeline reaches 33.55 using TIDE-Shared. Furthermore, the baseline distillation objectives, even without the components of TIDE, demonstrate improvement over the non-distilled model. This result confirms that the transfer of knowledge across architectures is viable across distinct tokenizers and attention mechanisms.

Each Pipeline Favors Its Native Strategy. The experimental results validate the modular design of TIDE, as each pipeline benefits from a distinct optimal configuration. Within the cross-tokenizer pipeline, the native TIDE-Cross strategy outperforms the swapped TIDEShared strategy by an average margin of 0.37. This observation indicates that the bounded gradients and dual-end noise filtering in the reverse objective are well-suited to scenarios involving alignment noise across different tokenizers (section 2.3). Conversely, within the shared-tokenizer pipeline, the native TIDE-Shared strategy surpasses TIDE-Cross by an average margin of 2.76. This finding demonstrates that a progressive curriculum and enriched teacher signals are more effective when token-level alignment is exact.

Distilled dLLMs Excel at Code Generation. Across all configurations, the distilled models exhibit strong proficiency in programming tasks. On the HumanEval benchmark, TIDEShared within the shared-tokenizer pipeline achieves a score of 48.78, and TIDE-Cross within the cross-tokenizer pipeline achieves 48.17 (TIDE-Shared further reaches 49.39), all substantially exceeding the 32.30 score of an equivalent-sized autoregressive model. A similar pattern emerges on the MBPP benchmark, where the best distilled model reaches a score of 38.60, compared to 36.60 for the autoregressive baseline. This advantage suggests that the parallel generation process of diffusion decoding, which maintains global coherence

- Table 2: Component-level ablation on the shared-tokenizer pipeline (WeDLM → Qwen3BD3LM). Bold: best per row.

Benchmark Baseline w/o Tstep w/o COMPDEMO Full

(w/o Train) (TIDAL + COMPDEMO)

GSM8K 48.07 48.82 48.90 48.90 MATH 11.74 11.96 11.84 11.68 BBH 26.37 26.51 26.77 26.66 MMLU-Pro 14.12 14.42 13.76 13.76 HellaSwag 40.03 40.35 40.16 40.27 MMLU 39.81 39.84 39.58 39.92 HumanEval 45.73 43.90 44.51 46.95 MBPP 38.60 37.20 38.20 37.00

Avg 33.06 32.88 32.97 33.14

across the entire output, is particularly suitable for structured outputs such as code. In these contexts, the syntactic and semantic consistency across the entire program remains critical.

##### 3.3 Ablation Studies

To isolate the contribution of each component within the TIDE-Cross (TIDAL + COMPDEMO) strategy, we conduct ablations on the shared-tokenizer pipeline (WeDLM → Qwen3-BD3LM) by removing one component at a time from the full method. The three ablation conditions are: removing the timestep axis, replacing the dual-axis scheduling with a timestep-only schedule which serves as our baseline, and removing COMPDEMO. Detailed configuration settings and formal definitions for these ablations are provided in Appendix B.

The complete method achieves the highest average score of 33.14 (Table 2), which confirms that each component provides a positive contribution.

The Timestep Axis Is the Most Impactful Component. The removal of the timestep axis causes the largest average performance drop of 0.26, which is primarily driven by a decline of 3.05 on HumanEval. This result validates the central motivation of TIDAL: the reliability of the teacher varies according to the masking ratio, and the modulation of λt along the diffusion timestep is essential for stable distillation. The timestep axis enables the student to decrease reliance on the teacher at high masking ratios where predictions are noisy, and to increase this reliance at low masking ratios where the teacher is confident. This dynamic is absent in autoregressive distillation, where the teacher always observes the complete left context.

CompDemo Provides Consistent Gains. The removal of COMPDEMO reduces the average performance by 0.17, with the most notable drops occurring on HumanEval (2.44) and MMLU (0.34). The complementary mask splitting strategy enriches the guidance signal from the teacher by exposing the student to two complementary views per training sample, which proves particularly beneficial for tasks that require structured generation.

Proposed Framework Outperforms Baseline. Compared to the Baseline—which relies solely on timestep scheduling as proposed in prior works—our complete TIDE framework achieves a higher average score. The integration of the training-progress axis into our dual-axis TIDAL, combined with COMPDEMO, stabilizes the early phase of distillation and prevents a significant performance drop of 0.83 on reasoning tasks such as GSM8K. This demonstrates the effectiveness and necessity of our proposed holistic approach over standalone scheduling methods.

##### 3.4 Inference Efficiency

To evaluate the benefits of distillation for practical deployment, we benchmark inference efficiency under two settings on a single NVIDIA H100-80GB GPU in bfloat16 (Appendix B). The controlled setting (Table 3) generates exactly 256 tokens from a single fixed prompt for

- Table 3: Inference efficiency comparison (controlled setting). Peak memory, latency, and throughput are measured on a single H100-80GB GPU generating 256 tokens in bfloat16.

Model Params Peak Mem Latency Tokens/s

(B) (GB) (s) Student (BD3LM-0.6B)

Distilled 0.60 1.4 6.25 41.0 No Distill 0.60 1.4 6.08 42.1

AR Baseline Qwen3-0.6B-Base 0.60 1.2 4.99 51.3

Teachers WeDLM-8B-Instruct 8.19 15.5 6.79 37.7 LLaDA2.0-mini 16.26 31.3 32.55 7.8

- Table 4: Per-benchmark inference speed (tokens/s) measured during actual evaluation runs. BD3LM exhibits near-constant throughput due to its fixed diffusion schedule.

Teachers AR Student

Benchmark LLaDA2 WeDLM Qwen3 No Distill Distilled

GSM8K 8.3 38.6 51.4 41.9 40.7 MATH 10.3 38.4 51.1 41.8 40.6 BBH 9.8 38.5 51.1 42.0 40.9 MMLU-Pro 5.6 38.3 51.0 42.2 40.5 HellaSwag 6.8 37.2 51.5 42.4 40.6 MMLU 6.2 37.1 51.4 41.9 41.1 HumanEval 9.8 36.8 51.1 42.5 41.7 MBPP 9.1 37.0 51.8 42.3 41.6

Avg 8.2 37.7 51.3 42.1 40.9

every model, providing a fair comparison of peak memory, latency, and throughput. The evaluation setting (Table 4) reports throughput during actual benchmark runs, where input prompts, context lengths, and few-shot configurations vary across eight tasks.

Distillation Enables Practical Deployment. Under the controlled setting (Table 3), the distilled student model requires only 1.4 GB of peak memory, representing a 22× reduction compared to LLaDA2 (31.3 GB) and an 11× reduction compared to WeDLM (15.5 GB). The inference latency of 6.25 s for 256 tokens yields a 5.2× speedup over LLaDA2 (32.55 s). As illustrated in figure 1, cross-architecture distillation compresses the knowledge of large teacher dLLMs into a model suitable for deployment on commodity hardware.

Distillation Adds Minimal Overhead. Under the controlled setting, distillation introduces only a 2.6% reduction in throughput relative to the undistilled BD3LM (41.0 vs. 42.1 tokens/s), with a marginal latency increase (6.25 s vs. 6.08 s) and identical memory footprint. The evaluation setting (Table 4) confirms that this overhead is uniform across all eight benchmarks despite varying prompt lengths and generation requirements, indicating that the distillation procedure does not degrade inference efficiency under realistic conditions. Compared to the AR baseline of the same size (51.3 tokens/s), the BD3LM student achieves approximately 80% throughput due to the iterative diffusion process; however, the quality gains from distillation (Table 1) and the inherent advantages of block-parallel generation make this trade-off favorable for practical deployment.

#### 4 Conclusion

We present TIDE, the first cross-architecture distillation framework for heterogeneous dLLMs. Experiments across two pipelines and eight benchmarks show that (1) crossarchitecture distillation improves the baseline by +1.53 on average, (2) each pipeline favors

a distinct strategy (Reverse CALM for cross-tokenizer, TIDAL + COMPDEMO for sharedtokenizer), and (3) distilled dLLMs outperform the same-size AR model by +16.48 on HumanEval. Future work includes scaling student capacity and extending the framework to continuous-state diffusion LMs.

#### Ethics Statement

This work focuses on improving the efficiency of diffusion language models through knowledge distillation. Publicly available datasets and pre-trained models are utilized. The computational requirements are moderate. No direct negative societal impacts are foreseen beyond the general concerns associated with the deployment of language models.

#### LLM Usage

In this section, we clarify the role of large language models (LLMs) in preparing this work. The model was used exclusively for language polishing, such as refining grammar, style, and readability, without contributing to the research design, analysis, or conclusions.

#### Acknowledgments

We would like to sincerely thank Xiangtai Li and Anran Wang for their selfless guidance and invaluable support throughout this project.

#### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations, 2024.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Mart´ın Bl´azquez, Guilherme Penedo, Lewis Tunstall, Andr´es Marafioti, Hynek Kydl´ıˇcek, Agust´ın Piqueres Lajar´ın, Vaibhav Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021a.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021b.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via selfdistillation through time. arXiv preprint arXiv:2410.21035, 2024.

Feiyang Fu, Tongxian Guo, and Zhaoqiang Liu. Learnable sampler distillation for discrete diffusion models. arXiv preprint arXiv:2509.19962, 2025.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations,

2024. URL https://openreview.net/forum?id=5h0qf7IBZZ.

Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. arXiv preprint arXiv:2410.08709, 2024.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id= d7KBjmI3GmQ.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021b.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Siming Huang, Tianhao Cheng, Jason Klein Liu, Weidi Xu, Jiaran Hao, Liuyihan Song, Yang Xu, Jian Yang, Jiaheng Liu, Chenchen Zhang, et al. Opencoder: The open cookbook for top-tier code large language models. pp. 33167–33193, 2025.

Minseo Kim, Chenfeng Xu, Coleman Hooper, Harman Singh, Ben Athiwaratkun, Ce Zhang, Kurt Keutzer, and Amir Gholami. Cdlm: Consistency diffusion language models for faster sampling. arXiv preprint arXiv:2511.19269, 2025.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. Distillm: Towards streamlined distillation for large language models. In Forty-first International Conference on Machine Learning.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, and Jie Zhou. Wedlm: Reconciling diffusion language models with standard causal attention for fast inference. arXiv preprint arXiv:2512.22737, 2025.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

Benjamin Minixhofer, Edoardo M Ponti, and Ivan Vuli´c. Zero-shot tokenizer transfer. Advances in Neural Information Processing Systems, 37:46791–46818, 2024.

Benjamin Minixhofer, Ivan Vuli´c, and Edoardo Maria Ponti. Universal cross-tokenizer distillation via approximate likelihood matching. arXiv preprint arXiv:2503.20083, 2025.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

Khouloud Saadi and Di Wang. What should feature distillation transfer in llms? a tasktangent geometry view, 2026. URL https://arxiv.org/abs/2507.10155.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Idan Shenfeld, Mehul Damani, Jonas Hubotter,¨ and Pulkit Agrawal. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897, 2026.

Makoto Shing, Kou Misaki, Han Bao, Sho Yokoi, and Takuya Akiba. Taid: Temporally adaptive interpolated distillation for efficient knowledge transfer in language models. arXiv preprint arXiv:2501.16937, 2025.

Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. Challenging bigbench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, 2023.

Ling Team, Ang Li, Ben Liu, Binbin Hu, Bing Li, Bingwei Zeng, Borui Ye, Caizhi Tang, Changxin Tian, Chao Huang, Chao Zhang, Chen Qian, Chenchen Ju, Chenchen Li, Chengfu Tang, Chilin Fu, Chunshao Ren, Chunwei Wu, Cong Zhang, Cunyin Peng, Dafeng Xu, Daixin Wang, Dalong Zhang, Dingnan Jin, Dingyuan Zhu, Dongke Hu, Fangzheng Zhao, Feifan Wu, Feng Zhu, Gangshan Wang, Haitao Zhang, Hailin Zhao, Hanxiao Zhang, Hanzi Wang, Hao Qian, Haoyi Yu, Heng Zhang, Hongliang Zhang, Hongzhi Luan, Huirong Dong, Huizhong Li, Jia Li, Jia Liu, Jialong Zhu, Jian Sha, Jianping Wei, Jiaolong Yang, Jieyue Ma, Jiewei Wu, Jinjing Huang, Jingyun Tian, Jingyuan Zhang, Jinquan Sun, Juanhui Tu, Jun Liu, Jun Xu, Jun Zhou, Junjie Ou, Junpeng Fang, Kaihong Zhang, Kaiqin Hu, Ke Shi, Kun Tang, Kunlong Chen, Lanyin Mei, Lei Liang, Lei Xu, Libo Zhang, Lin Ju, Lin Yuan, Ling Zhong, Lintao Ma, Lu Liu, Lu Yu, Lun Cai, Meiqi Zhu, Mengying Li, Min Chen, Minghao Xue, Minghong Cai, Mingming Yin, Peijie Jiang, Peilong Zhao, Pingping Liu, Qian Zhao, Qing Cui, Qingxiang Huang, Qingyuan Yang, Quankun Yu, Shaowei Wei, Shijie Lian, Shoujian Zheng, Shun Song, Shungen Zhang, Shuo Zhang, Siyuan Li, Song Liu, Ting Guo, Tong Zhao, Wanli Gu, Weichang Wu, Weiguang Han, Wenjing Fang, Wubin Wang, Xiang Shu, Xiao Shi, Xiaoshun Lan, Xiaolu Zhang, Xiaqing Sun, Xin Zhao, Xingyu Lu, Xiong Xu, Xudong Wang, Xudong Wang, Xuemin Yang, Yajie Yang, Yang Xiang, Yanzhe Li, Yi Zhang, Yilong Wang, Yingxue Li, Yongzhen Guo, Yuzhuo Fu, Yuanyuan Wang, Yue Yang, Yue Yu, Yufeng Deng, Yun Zhang, Yunfei Yu, Yuqi Zhang, Yuxiao He, Zengke Gui, Zhaoxin Huan, Zhaoyang Wang, Zhibo Zhu, Zhihao Wang, Zhiqiang Zhang, Zhoufei Wang, Zihang Zeng, Ziqi Liu, Zitao Xuan, and Zuoli Tang. Every activation boosted: Scaling general reasoner to 1 trillion open language foundation, 2025. URL https://arxiv.org/abs/2510.22115.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 4791–4800, 2019.

Songming Zhang, Xue Zhang, Zengkui Sun, Yufeng Chen, and Jinan Xu. Dual-space knowledge distillation for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 18164–18181, 2024.

Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song. dllm: Simple diffusion language modeling, 2026. URL https://arxiv.org/abs/2602.22661.

#### A Related Work

Diffusion Language Models. D3PM (Austin et al., 2021a) pioneered discrete diffusion for text generation. MDLM (Sahoo et al., 2024) and SEDD (Lou et al., 2023) further established theoretical foundations through simplified masked diffusion and score estimation, respectively. Building on these foundations, practical dLLMs have emerged with diverse architectures: LLaDA (Nie et al., 2025) adopts full bidirectional attention, BD3LM (Arriola et al., 2025) introduces block diffusion with staircase attention, Dream (Ye et al., 2025) extends masked diffusion with rectified estimation, and WeDLM (Liu et al., 2025) proposes a causal diffusion architecture combining sliding-window and global attention. DiffuLLaMA (Gong et al., 2024) converts pre-trained AR models into diffusion LMs. This architectural heterogeneity—encoder, decoder-block, and causal variants—motivates the need for a cross-architecture distillation framework.

Knowledge Distillation of Large Language Models. Knowledge distillation (Hinton et al., 2015) transfers knowledge from a large teacher to a smaller student through soft targets. For AR models, representative methods include reverse KL minimization (Gu et al., 2024), skewed KL divergence (Ko et al.), on-policy distillation (Agarwal et al., 2024), dual-space transfer (Zhang et al., 2024), feature-level distillation (Saadi & Wang, 2026), and time-varying interpolation (Shing et al., 2025). These methods target the left-to-right AR paradigm. TIDE builds on this line of work, particularly TAID’s interpolation principle, and adapts it to dLLMs, where teacher reliability varies with the diffusion timestep and all masked tokens are predicted simultaneously.

Distillation for Diffusion Language Models. Existing dLLM distillation methodsCDLM (Kim et al., 2025), DDD (Hayakawa et al., 2024), LSD (Fu et al., 2025), and SDTT (Deschenaux & Gulcehre, 2024)—focus exclusively on step compression: the student and teacher share the same architecture and tokenizer, and the goal is to reduce the number of inference steps. TIDE addresses a fundamentally different problem: cross-architecture distillation, where the teacher and student differ in architecture, attention mechanism, and potentially tokenizer. For the cross-tokenizer case, we build on ALM (Minixhofer et al., 2025) and ZeTT (Minixhofer et al., 2024), adapting chunk-level approximate likelihood matching from the AR setting to dLLMs as CALM (section 2.3).

#### B Training, Inference, and Evaluation Details

This section provides the comprehensive configurations for training, inference, and evaluation used throughout the experiments.

##### Training Configurations.

- Table 5 summarizes the complete set of training hyperparameters for both pipelines.

Ablation Study Configurations. For the component-level ablation studies presented in Section 3.3, all configurations are trained for 3 epochs. The specific definitions for the three ablation conditions are as follows:

- • w/o Tstep (removing the timestep axis): λt = λtrain, i.e. lambda depends only on the training progress.
- • Baseline (timestep-only schedule): λt = const × (1 − t), replacing the dual-axis scheduling with a schedule proposed in previous works.
- • w/o COMPDEMO: Removes the complementary demonstration strategy, using only a single teacher forward pass with the dual-axis TIDAL.

Evaluation Protocol. All evaluations employ diffusion sampling with a block size of 32 and a classifier-free guidance scale of 0.0. The number of sampling steps varies depending on the specific task, ranging from 3 to 256. Table 6 details the precise configuration for each evaluation.

Inference Efficiency Protocol. For the inference efficiency measurements in Section 3.4, all evaluations are conducted on a single NVIDIA H100-80GB GPU with bfloat16 precision. Under the controlled setting, we generate 256 tokens and report the best performance

Table 5: Training hyperparameters.

Parameter Cross-Tok. Shared-Tok. Teacher LLaDA2.0-mini (16B MoE) WeDLM-8B-Instruct (8B) Student init SFT v0.1 SFT v0.1 Methods Reverse CALM TIDAL + COMPDEMO Learning rate 5e-5 5e-5 Epochs 10 10 Sequence length (student) 512 512 Teacher max length 1024 768 Block size 32 32 Precision bfloat16 bfloat16 Teacher precision bfloat16 bfloat16 Attention impl. flex attention flex attention Strategy DDP DDP Dataset Tulu-3 SFT + SmolTalk + OpenCoder-SFT (Python) TIDAL λinit → λmax - 0.1 → 0.9 TIDAL schedule - cosine TIDAL timestep weight - midrange COMPDEMO demo ratio - 0.5 Temperature T - 2.0

Table 6: Per-task evaluation configuration.

Task max new tokens steps few-shot batch size

MMLU 3 3 0 128 MMLU-Pro 256 256 0 64 HellaSwag 3 3 0 128 GSM8K 256 256 0 128 BBH 256 256 3 32 MATH 256 256 0 32 HumanEval 256 256 0 128 MBPP 256 256 0 32

observed across five independent runs. Under the evaluation setting, we run 50 randomly sampled examples from each benchmark and report the average.

#### C Gradient Analysis

This appendix provides detailed derivations of the gradients that support the theoretical justification for Reverse CALM presented in Section 2.3.

Forward CALM gradient. The gradient of the forward CALM loss with respect to the student parameters θ contains a ratio pct/pcs, where pct and pcs are the chunk probabilities of the teacher and the student, respectively:

pct pcs −

1 − pct 1 − pcs

. (13)

gfwd =

When pcs → 0 but pct > 0, gfwd → +∞. In the cross-tokenizer setting, imperfect chunk alignment frequently produces chunks where the student assigns a low initial probability,

triggering this gradient explosion. Reverse CALM gradient. The reverse CALM gradient takes the form:

pct 1 − pct

∂pcs

∂Lrev ∂θ

### = −∑

∂θ · log

. (14)

c

c t

The gradient coefficient log p

1−pct depends solely on the fixed teacher probabilities and is bounded: |grev| ≤ | log 1−ϵ

ϵ | for pct ∈ [ϵ,1 − ϵ]. This provides a stable, self-selecting training

signal where the student naturally concentrates updates on the high-probability modes of the teacher.

c t

c s

∂θ · log p

Dual-end noise filtering. The full reverse gradient ∂p

1−pct is filtered on both ends:

(1) poorly aligned chunks yield pct ≈ 0.5, zeroing the teacher-end coefficient; (2) chunks with low pcs contribute small ∂p

c s

∂θ , suppressing the student-end signal. Forward CALM has no such dual filtering—1/pcs amplifies noise instead. Bernoulli KL equivalence. Reverse CALM is equivalent to minimizing the Bernoulli KL divergence KLBern(pcs∥pct) = pcs log p

c s

c s

pct + (1 − pcs) log 1−p

1−pct , up to an additive constant independent of θ. This gives Reverse CALM a principled information-theoretic interpretation as mode-seeking distillation in Bernoulli (scalar) space.

TIDAL and reverse direction. TIDAL addresses the instability of forward CALM via a curriculum where λt transitions from emphasizing the student to the teacher. However, reverse CALM does not exhibit this instability. Applying TIDAL to reverse CALM is counterproductive: during the late stages of training, the (1 − λt) factor approaches 0.1, suppressing the gradient and destroying the self-selection mechanism of reverse CALM. This confirms that TIDAL is effective with forward-direction objectives but should not be applied to reverse-direction objectives.

#### D Limitations and Future Work

The empirical scope of this work is limited to a 0.6B-parameter student model using block diffusion with staircase attention. A primary avenue for future research is to scale the student model to 1.3B or 3B parameters to assess whether cross-architecture distillation efficiency improves as the capacity gap narrows. Furthermore, while the TIDE framework is theoretically architecture-agnostic, empirical validation on alternative structures, such as continuous-state diffusion language models or encoder-style dLLMs, remains necessary. Adapting the proposed loss formulations from categorical distributions to continuous densities is a critical next step to broaden the framework’s applicability.

Furthermore, the training pipeline currently operates within a 512-token context window, leaving the efficacy of cross-tokenizer alignment and COMPDEMO on extended sequences unexplored. Future investigations must examine how an increase in the number of alignment chunks alters the relative contributions of these components. Additionally, the present methodology processes the cross-tokenizer and shared-tokenizer pipelines independently; formulating a unified multi-teacher distillation objective could facilitate complementary knowledge transfer and yield a more robust representation for the student model.

Finally, computational efficiency and optimization dynamics present crucial areas for refinement. The COMPDEMO component necessitates two forward passes through the frozen teacher model per step, increasing training duration by approximately 50%. Concurrently, gradient-suppression mechanisms render the combination of Reverse CALM and TIDAL counterproductive (Appendix C). Developing alternative scheduling paradigms, such as restricting TIDAL’s modulation exclusively to the cross-entropy objective, is essential for reconciling these optimization conflicts and realizing the cumulative benefits of both strategies.

#### E Case Study

To comprehend the information conveyed by knowledge distillation beyond aggregate benchmark improvements, we conduct a diagnostic study that examines the transfer of dark knowledge and qualitative error patterns.

Dark Knowledge Transfer. Within the shared-tokenizer pipeline, we measure the KL divergence between the predictions of the student and those of the teacher at intermediate denoising timesteps. As illustrated in Figure 3, TIDE-Shared reduces the KL divergence relative to the WeDLM teacher by 46% on the GSM8K dataset (6.69 compared to 12.44).

Dark Knowledge Transfer (WeDLM Teacher, GSM8K)

12.44

KL(teacherstudent)

12.5

10.0

46%

6.69

7.5

5.0

2.5

0.0

Distilled (TIDE-Shared)

No Distill (Baseline)

Figure 3: The KL divergence relative to the WeDLM teacher on the GSM8K dataset. The distilled student achieves a KL divergence that is 46% lower than that of the non-distilled baseline (6.69 compared to 12.44).

This reduction confirms that the distilled student inherits the teacher’s prediction distribution. The cross-tokenizer KL comparison (from LLaDA2 to the student) is omitted due to vocabulary misalignment.

Qualitative Error Analysis. We select four instances wherein the non-distilled baseline fails, whereas exactly one distilled variant succeeds (Table 8). Table 7 illustrates two representative examples in detail. For instance, #39 of GSM8K, the baseline confuses time allocation with speed in a multi-step distance problem; only TIDE-Cross (distilled from LLaDA2) correctly computes the travel distances. For instance, #50 of HumanEval, the baseline calls encode shift again instead of reversing the shift; only TIDE-Shared (distilled from WeDLM) correctly subtracts 5. These examples demonstrate that each teacher imparts distinct knowledge. Specifically, LLaDA2 corrects multi-step arithmetic omissions and enhances type-level awareness in code generation, whereas WeDLM resolves proportional reasoning errors and facilitates algorithmic inversion.

##### Verbatim Model Outputs

The following presents the complete, unedited model outputs for all four cases in Table 8. These raw responses confirm that each model produces a distinct reasoning chain or code implementation.

##### GSM8K #39: Baseline (No Distill) — Incorrect.

Dana can skip at a rate of speed 3 miles per hour, so she can run at a rate of speed 3 × 2 = 6 miles per hour. If she spends one-third of the time running, she will run 6/3 = 2 miles per hour. If she spends two-thirds of the time walking, she will walk 6/3 = 2 miles per hour. The total distance Dana can travel in 6 hours is

- 2 × 2 + 4 × 2 = 12 + 4 = 16 miles. #### 16. The answer is: 16

GSM8K #39: TIDE-Cross (LLaDA2) — Correct.

Dana can skip at a rate of 3 miles per hour, so she can run at a rate of 3 × 2 = 6 miles per hour. If she spends one-third of the time running, she will run for 6/3 = 2 hours. If she spends two-thirds of the time walking, she will walk for 6/3 = 2 hours. In 2 hours, she will run for 2 × 6 = 12 miles. In 2 hours, she will walk for

- 2 × 3 = 6 miles. Therefore, Dana can travel a total of 12 + 6 = 18 miles in six hours. #### 18. The answer is: 18

- Table 7: A detailed textual representation of case studies. This table compares the baseline model’s reasoning chains and generated code against those of distilled variants.

GSM8K #39: Multi-Step Distance Calculation

Prompt: Dana can run at a rate of speed four times faster than she can walk, but she can skip at a rate of speed that is half as fast as she can run. If she can skip at 3 miles per hour, how many miles can she travel in six hours if she spends one-third of the time running and two-thirds of the time walking? (Ground Truth: 18)

Model Output Analysis Baseline (No Distill)

Dana can skip at a rate of speed 3 miles per hour, so she can run at a rate of speed 3 ∗ 2 = 6 miles per hour. If she spends one-third of the time running, she will run 6/3 = 2 miles per hour. If she spends two-thirds of the time walking, she will walk 6/3 = 2 miles per hour. The total distance Dana can travel in 6 hours is 2 ∗ 2 + 4 ∗ 2 = 12 + 4 = 16 miles. Answer: 16

Confuses time allocation (6/3 = 2 hours) with speed (“2 miles per hour”).

Dana can skip at a rate of 3 miles per hour, so she can run at a rate of 3 ∗ 2 = 6 miles per hour. If she spends one-third of the time running, she will run for 6/3 = 2 hours. If she spends two-thirds of the time walking, she will walk for 6/3 = 2 hours. In 2 hours, she will run for 2 ∗ 6 = 12 miles. In 2 hours, she will walk for 2 ∗ 3 = 6 miles. Therefore, Dana can travel a total of 12 + 6 = 18 miles in six hours. Answer: 18

Correctly computes time allocation and travel distance.

TIDE-Cross (LLaDA2)

Dana can skip at a rate of 3 miles per hour, so she can run at a rate of 3 ∗ 2 = 6 miles per hour. If she spends one-third of the time running, she will spend 6/3 = 2 hours running. If she spends two-thirds of the time walking, she will spend 6/3 = 2 hours walking. In total, she will spend 2 + 2 = 4 hours running and walking. If she can run at a rate of 6 miles per hour, she will travel 6 ∗ 6 = 36 miles in 6 hours of running. If she can skip at a rate of 3 miles per hour, she will travel 3 ∗ 6 = 18 miles in 6 hours skipping. Therefore, she will travel a total of 36 + 18 = 54 miles. Answer: 54

Correctly allocates time but uses the full 6 hours instead of the allocated 2 hours.

TIDE-Shared (WeDLM)

HumanEval #50: Algorithmic Inversion Prompt: The function encode shift shifts each character by +5 in the alphabet. Implement the function decode shift: reverse the encoding, and return the original string.

Model Output Analysis Baseline (No Distill)

return encode shift(s) Calls encode shift again (+5) instead of reversing (−5).

Same erroneous code as Baseline, but with a longer (incorrect) explanation appended.

TIDE-Cross (LLaDA2)

return encode shift(s)

+ 500-char explanation that incorrectly justifies this as valid decoding.

Correctly subtracts 5 to reverse the shift.

TIDE-Shared (WeDLM)

return "".join( [chr(((ord(ch) - 5

- ord("a")) % 26)

+ ord("a")) for ch in s])

- Table 8: A summary of the case studies. The non-distilled baseline fails on all four problems, whereas each distillation pipeline resolves distinct errors.

Case & Error Type Base TIDE-Cross TIDE-Shared Math Reasoning (GSM8K) #39: rate-distance confusion in multi-step problem

###### × ✓ ×

###### × × ✓

#77: proportional relationships reversed

Code Generation (HumanEval) #11: bitwise XOR on string characters

###### × ✓ ×

× × ✓

#50: decode calls encode (no reversal)

##### GSM8K #39: TIDE-Shared (WeDLM) — Incorrect.

Dana can skip at a rate of 3 miles per hour, so she can run at a rate of 3 × 2 = 6 miles per hour. If she spends one-third of the time running, she will spend 6/3 = 2 hours running. If she spends two-thirds of the time walking, she will spend 6/3 = 2 hours walking. In total, she will spend 2 + 2 = 4 hours running and walking. If she can run at a rate of 6 miles per hour, she will travel 6 × 6 = 36 miles in 6 hours of running. If she can skip at a rate of 3 miles per hour, she will travel 3 × 6 = 18 miles in 6 hours skipping. Therefore, she will travel a total of 36 + 18 = 54 miles in 6 hours. #### 54. The answer is: 54

##### GSM8K #77: Baseline (No Distill) — Incorrect.

Sarah does 400 pounds of laundry. Raymond does 4 × 400 = 1600 pounds of laundry. Raymond does 1600/2 = 800 pounds of laundry. Raymond does 800 − 400 = 400 pounds of laundry than David.

##### GSM8K #77: TIDE-Cross (LLaDA2) — Incorrect.

400/4 = 100 pounds of laundry. Raymond does 100/2 = 50 pounds of laundry. The difference is 100 − 50 = 50 pounds.

##### GSM8K #77: TIDE-Shared (WeDLM) — Correct.

Sarah does 400 pounds of laundry. Sarah does 4 times as much laundry as David, so David does 400/4 = 100 pounds of laundry. Raymond does half as much laundry as Sarah, so Raymond does 400/2 = 200 pounds of laundry. The difference in the amount of laundry Raymond and David do is 200 − 100 = 100 pounds.

##### HumanEval #11 (string xor): Baseline (No Distill) — Failed.

result = "" for a, b in zip(a, b):

result += str(a ˆ b) return result

##### HumanEval #11 (string xor): TIDE-Cross (LLaDA2) — Passed.

result = '' for a, b in zip(a, b):

if a != b:

result += '1' else:

result += '0' return result

##### HumanEval #11 (string xor): TIDE-Shared (WeDLM) — Failed.

result = "" for a, b in zip(a, b):

result += str(a ˆ b)) return result[:-1]

##### HumanEval #50 (decode shift): Baseline (No Distill) — Failed.

return encode_shift(s)

##### HumanEval #50 (decode shift): TIDE-Cross (LLaDA2) — Failed.

return encode_shift(s)

Note: The generated code is identical to the Baseline. TIDE-Cross additionally provides a lengthy explanation that incorrectly characterizes this as a valid decoding strategy, suggesting that the model lacks an understanding that decoding requires a reversal operation.

##### HumanEval #50 (decode shift): TIDE-Shared (WeDLM) — Passed.

return "".join( [chr(((ord(ch) - 5 - ord("a"))

% 26) + ord("a")) for ch in s])

