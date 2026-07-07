# arXiv:2604.07023v1[cs.CL]8Apr2026

## MARS: Enabling Autoregressive Models Multi-Token Generation

#### Ziqi Jin1 Lei Wang2 Ziwei Luo3 Aixin Sun1

1Nanyang Technological University 2Singapore Management University 3Uppsala University

### Abstract

Autoregressive (AR) language models generate text one token at a time, even when consecutive tokens are highly predictable given earlier context. We introduce MARS (Mask AutoRegreSsion), a lightweight fine-tuning method that teaches an instruction-tuned AR model to predict multiple tokens per forward pass. MARS adds no architectural modifications, no extra parameters, and produces a single model that can still be called exactly like the original AR model with no performance degradation. Unlike speculative decoding, which maintains a separate draft model alongside the target, or multi-head approaches such as Medusa, which attach additional prediction heads, MARS requires only continued training on existing instruction data. When generating one token per forward pass, MARS matches or exceeds the AR baseline on six standard benchmarks. When allowed to accept multiple tokens per step, it maintains baseline-level accuracy while achieving 1.5–1.7× throughput. We further develop a block-level KV caching strategy for batch inference, achieving up to 1.71× wall-clock speedup over AR with KV cache on Qwen2.5-7B. Finally, MARS supports real-time speed adjustment via confidence thresholding: under high request load, the serving system can increase throughput on the fly without swapping models or restarting, providing a practical latency–quality knob for deployment. 1

### 1 Introduction

Autoregressive (AR) language models spend the same compute on every token: one forward pass produces exactly one token, whether that token is the inevitable “the answer is” following a multiple-choice prompt or a genuinely uncertain next word in an open-ended generation. This uniform cost is wasteful.

Existing approaches to multi-token generation all modify the deployment stack. Speculative decoding [Chen et al., 2023, Leviathan et al., 2023] maintains a separate draft model alongside the target, doubling memory footprint and adding orchestration complexity. Multi-head approaches such as Medusa [Cai et al., 2024] and EAGLE [Li et al., 2024] attach additional prediction heads to the architecture, requiring extra parameters and head-specific training. Both families introduce components beyond the original model, complicating production serving pipelines.

In this work we take a different angle: rather than building auxiliary components around an AR model, we ask whether lightweight fine-tuning alone can give the model an additional capability to generate multiple tokens per forward pass, while preserving its original behavior as a standard AR model. The resulting model should be a strict superset: it can still be served exactly like the original, producing one token at a time with no quality loss, but it also optionally accepts multiple tokens per step when confident, accelerating generation with minimal cost.

1Our code is available at: https://github.com/Xalp/MARS

Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?

###### MARS-7B (confidence threshold τ=0.95), 66 forwards, 2.55 tokens/forward:

Let ’s break it down step by step: 1 . Seattle has 20 sheep. 2. Charleston has 4 times as many sheep as Seattle , so Charleston has 4 x 20 = 80 sheep. 3. Toulouse has twice as many sheep as Charleston, so T oulouse has 2 x 80 = 16 0 sheep. Now , let ’s add up

... Total number of sheep : 20 + 80 + 16 0 = 1 00 + 160 = 260 sheep

Shaded regions = multiple tokens accepted in one forward pass. Unshaded tokens are generated one at a time. Darker = more tokens per step. Formulaic phrases and calculations are chunked; novel reasoning proceeds token-by-token.

- Figure 1: Example MARS generation on GSM8K. The model adaptively generates 1–4 tokens per forward pass based on confidence: predictable continuations are batched together, while novel content proceeds token-by-token. This achieves 2.55× token per forward over standard AR decoding.

- A natural starting point is block masked diffusion, which trains the model to predict multiple tokens per step within a fixed-size block. However, existing AR-to-block-diffusion conversions result in substantial quality degradation, particularly on reasoning and coding tasks [Arriola et al., 2025, Zhou et al., 2026, Gong et al., 2025, Fu et al., 2025]. In Section 3.1, we identify four gaps between AR and block diffusion that may account for this degradation, but find that only one is inherent to multi-token prediction: the use of mask tokens as placeholders for unknown future tokens. The other three arise from design choices that unnecessarily depart from the original AR model, and can be fully closed.

From these motivations, we introduce MARS (Mask AutoRegreSsion). MARS maintains causal intra-block attention, uses right-shifted logits, and generates strictly left-to-right, leaving only the inherent masking gap. As shown in Figure 1, the resulting model adaptively generates multiple tokens per forward pass when confident, while falling back to single-token generation for novel content. When generating one token per step, MARS matches or exceeds the original AR baseline on six benchmarks, meaning there is no quality cost for gaining the option to accelerate. When multitoken generation is enabled, MARS achieves 1.5–1.7× throughput while maintaining baseline-level accuracy.

#### Contributions.

- • We analyze the four gaps between AR and block-masked prediction and show that three of them are eliminable design choices rather than inherent limitations. Closing these gaps is sufficient to recover baseline quality without architectural changes.
- • We propose MARS, a lightweight fine-tuning method that teaches an instruction-tuned AR model to optionally predict multiple tokens per forward pass, with no architectural changes, no additional parameters, and reusing the same SFT data. The resulting model can be used exactly like the original AR model with no quality loss, or switched to multi-token mode for faster generation on the fly.
- • We identify an auxiliary SFT loss on the clean input stream as the key ingredient for preserving performance at larger block sizes, where the fraction of AR-like training signal would otherwise decay.
- • We develop a block-level KV caching strategy for batch inference and demonstrate wallclock speedups of up to 1.71× over AR with KV cache on Qwen2.5-7B.

### 2 Background and Related Work

An autoregressive (AR) language model generates text by predicting one token at a time, each conditioned on all previous tokens via causal (left-to-right) attention. Generating T tokens requires T serial forward passes, so inference cost scales linearly with output length regardless of how predictable individual tokens are. If we want a single forward pass to predict multiple future tokens using the same backbone and language model head, a natural construction is block-masked prediction: replace a contiguous block of B future tokens with [MASK] placeholders and train the model to recover them,

conditioned on the clean tokens from all preceding blocks. To train all blocks in parallel within a single forward pass, we concatenate the clean and masked sequences: [x;x˜], with a structured attention mask that controls visibility (detailed in Section 3.2). However, directly applying this to a pretrained AR model introduces a mismatch: the model was trained under purely causal, fully visible context, but now must predict from partially masked input with potentially different attention patterns and generation order. This mismatch is the source of the quality degradation observed in prior work [Arriola et al., 2025], and motivates the analysis in Section 3.1.

Discrete diffusion and masked language models. Diffusion models for discrete data [Austin et al., 2021, Li et al., 2022] generate tokens through iterative denoising. Early non-autoregressive work in machine translation [Gu et al., 2018, Ghazvininejad et al., 2019] demonstrated significant speedups at the cost of quality. More recently, masked diffusion language models [Sahoo et al., 2024, Shi et al., 2024, Lou et al., 2024, Gat et al., 2024, Ye et al., 2025, Nie et al., 2025] apply iterative unmasking to text generation and have begun to close the quality gap with AR models. These models typically use bidirectional attention, a design choice inherited from diffusion models in vision. Several works have converted pretrained AR models into diffusion language models [Arriola et al., 2025, Zhou et al., 2026, Gong et al., 2025, Fu et al., 2025], demonstrating that the transition is feasible without training from scratch, though often with significant quality degradation due to the mismatch between the AR model’s learned causal attention pattern and the bidirectional attention adopted during conversion.

Causality and multi-token generation. A growing body of work shows that causal attention does not conflict with multi-token parallel generation. Block Diffusion [Arriola et al., 2025] uses causal attention across blocks while keeping bidirectional attention within each block. Building on this, several works further accelerate diffusion LM inference [Wu et al., 2025], including Fast-dLLM which introduces KV caching for bidirectional diffusion models, and self-speculative decoding [Gao et al., 2025] which uses the diffusion model itself as both drafter and verifier. CARD [Ruan et al., 2026] applies strictly causal attention to the entire sequence and finds that this actually improves over bidirectional alternatives. ARMD [Karami and Ghodsi, 2026] and A3 [Du et al., 2026] similarly adopt causal structure for masked diffusion and any-order generation, respectively. Eso-LM [Sahoo et al., 2026] further unifies AR and masked diffusion under causal attention with KV caching. Together, these results establish that bidirectional attention in diffusion language models was a design choice inherited from vision, not a requirement for parallel token generation.

This observation motivates a natural question: if causality and multi-token generation are compatible, how much modification does an AR model actually need to generate multiple tokens at once? Our gap analysis (Section 3.1) suggests the answer is surprisingly little. Of the four gaps between AR and block-masked prediction, three turn out to be eliminable design choices. The sole inherent gap is some positions must predict from incomplete context, which can be addressed with a simple fine-tuning objective.

Multi-token prediction and parallel decoding. Multi-token prediction (MTP) [Gloeckle et al., 2024] trains AR models with k−1 auxiliary heads to predict future tokens in parallel; DeepSeekV3 [Liu et al., 2024] deploys this at 671B scale. Medusa [Cai et al., 2024] and EAGLE [Li et al., 2024] add lightweight heads for speculative multi-token proposals. These methods require additional parameters and architectural modifications. MARS achieves multi-token prediction without extra heads, using the same language model head for all positions within a block. Speculative decoding [Leviathan et al., 2023, Chen et al., 2023] avoids modifying the target model but requires maintaining a separate draft model.

Jacobi decoding [Teng et al., 2024] and Lookahead decoding [Fu et al., 2024] achieve parallel generation from a single unmodified model by treating AR generation as a fixed-point iteration. CLLMs [Kou et al., 2024] improve upon this by training the model to converge faster under Jacobi iteration. These methods initialize future positions with random tokens and rely on convergence from incorrect prefixes. MARS takes a complementary approach: instead of random initialization, we use [MASK] tokens as explicit placeholders and train the model to predict from this incomplete context directly. This yields substantially higher acceptance rates (1.5× vs 1.07× tokens per forward for Jacobi; Appendix C).

- Table 1: Gaps between AR and block diffusion. MARS aligns with AR on gaps (2)–(4), leaving only the inherent masking gap.

AR MARS Block Diffusion

- (1) Token Masking No masking Masked within block Masked within block

- (2) Attention Pattern Causal Causal Bidirectional (intra-block)

- (3) Logits Alignment Right-shifted Right-shifted Varies

- (4) Generation Order Left-to-right Left-to-right Confidence-based

### 3 Method

The design of MARS is guided by one principle: the model must remain a fully functional AR model. Multi-token prediction is an added capability, not a replacement. We begin by analyzing where prior block-masked approaches break this property, then show how MARS closes every eliminable gap at training and inference.

#### 3.1 Where Does Block-Masked Prediction Fail?

Block-masked prediction fails when it departs from the autoregressive model along multiple axes. We identify four gaps between AR and block diffusion (Table 1). Of these, one is inherent to multi-token prediction and cannot be avoided; the other three are eliminable design choices that unnecessarily break AR compatibility.

Gap (1), token masking, is inherent: to predict multiple tokens in parallel, future positions must be replaced with [MASK] placeholders. This is the fundamental cost of multi-token prediction. The remaining three gaps are eliminable design choices in prior systems that unnecessarily break AR compatibility. Gap (2), attention pattern: some block diffusion approaches [Arriola et al., 2025] use bidirectional attention within blocks, but a bidirectional model is no longer a functional AR model; MARS keeps strictly causal attention everywhere. Gap (3), logits alignment: AR models predict xt+1 from position t (right-shifted logits); changing this convention breaks the output head’s AR function, so MARS preserves it. Gap (4), generation order: confidence-based diffusion methods unmask tokens out of order within a block, breaking left-to-right generation; MARS always accepts tokens strictly left-to-right. By closing gaps (2)–(4), the resulting model remains a fully functional AR model, with the only difference being that it sees [MASK] placeholders within the current block.

#### 3.2 Training

MARS starts from an AR SFT checkpoint trained on the target instruction-following data with standard next-token prediction. Starting from this checkpoint ensures the model has already absorbed the training data distribution, so that MARS training focuses solely on learning the masked prediction paradigm (see Section 4.1 for details).

The high-level idea is simple: we run two copies of the sequence through the model in parallel. The clean stream keeps the original tokens intact and trains the model with ordinary AR next-token prediction. The noisy stream replaces each block of B tokens with [MASK] placeholders and asks the model to predict them, using the clean prefix from earlier blocks as context. Both streams share the same forward pass through a structured attention mask that enforces the correct visibility for each position. This design closes gaps (2) and (3) from Section 3.1 by construction: attention is strictly causal everywhere, and logits are right-shifted to match the AR convention.

Given a response sequence x = (x1,...,xL), we divide it into blocks of size B and replace all tokens in each block with [MASK]:

,...) (1)

x˜ = ([MASK],...,[MASK]

##### ,[MASK],...,[MASK]

B

B

The model processes a concatenated input z = [x;x˜] of length 2L, where the first L positions are the clean stream and the last L are the noisy stream.

点击下⽅按钮下载 SVG ⽂件（可直接嵌⼊ LaTeX / Word / Overleaf）

⬇ SVG ⬇ PDF (print) 📋 复制 SVG

###### (b) Inference: sliding window

###### (a) Training Mask

Clean Preﬁx Appended Masks

A B C D E F G H 0 1 2 3 4 5 6 7

M 0

M 1

M 2

M 3

M 4

M 5

M 6

M 7

###### Step 1 The [M] [M] [M] [M]

- A 0 B

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

- B 1 C

- C 2 D

- D 3 E

- E 4 F

- F 5 G

- G 6 H

- H 7 I

forward

accept 3

The sky is blue and

###### Step 2 The sky is blue [M] [M] [M] [M]

forward

accept 2

The sky is blue , the wind is

- M 0 A

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

- M 1 B

- M 2 C

- M 3 D

- M 4 E

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

- M 5 F

- M 6 G

- M 7 H

###### Step 3 The sky is blue , the [M] [M] [M] [M]

forward

accept 4

The sky is blue , the air is fresh .

Preﬁx / Input Clean causal Noisy causal Target Accepted [MASK] Rejected

- Figure 2: MARS attention mask and inference for L=8, B=4. Left: training mask with [x | x˜] concatenation. The orange cells show that noisy positions attend to each other causally within each block, in contrast to Block Diffusion [Arriola et al., 2025] which uses bidirectional attention within blocks. Right: sliding-window inference. The dashed line marks the generation cursor; B [MASK] tokens are appended and filled via one forward pass. Accepted tokens (blue) slide into the prefix for the next step.

We define the attention mask M ∈ {0,−∞}2L×2L over z. Let β(t) = ⌈t/B⌉ denote the block index of position t within either stream. For query position i and key position j:

 

0 if i,j ≤ L, j ≤ i (clean causal) 0 if i,j > L, β(i−L) = β(j−L), j ≤ i (noisy intra-block causal) 0 if i > L, j ≤ L, β(j) < β(i−L) (noisy → clean cross-stream) −∞ otherwise

(2)

Mij =



The clean causal case gives the clean stream (positions 1,...,L) standard causal self-attention, identical to AR training. The noisy intra-block causal case allows each noisy position to attend causally within its own block (only [MASK] tokens, so only positional information flows). The cross-stream case allows each noisy block k to see clean tokens from blocks 1,...,k−1, providing the prefix context needed for prediction.

The training loss on the noisy stream is cross-entropy over masked positions:

log pθ(xt | x˜,x<β(t)) (3)

Lmask = −

t∈M

where M denotes the set of masked positions and x<β(t) denotes clean tokens from all blocks preceding block β(t).

#### 3.3 Preserving Autoregressive Competence

Closing gaps (2)–(4) ensures that MARS behaves like an AR model at inference. But training must also ensure the model remains an AR model in terms of capability. Block-masked training, by itself, gradually erodes the AR signal, and this erosion gets worse exactly when larger blocks would be most useful. The SFT loss is not a regularization trick; it is the mechanism that keeps the model’s AR competence intact while it learns block prediction.

Within a block of size B, position t (1-indexed within the block) is conditioned on t−1 masked tokens from the same block, plus the fully clean prefix from prior blocks. Only the first position (t=1) sees entirely clean context, making its prediction exactly equivalent to AR next-token prediction. Positions t=2,...,B see progressively more [MASK] tokens in place of real context, with position

- B seeing B−1 placeholders. As a simple proxy for how much AR-like signal the model receives, we count the fraction of positions with fully clean context:

##### 1 B

rAR(mask only) =

(4)

This is a coarse measure—positions near the start of a block still receive mostly clean context—but it captures the trend: as B grows, the training signal becomes increasingly unlike standard AR. For B=4 the ratio is 25%; for B=8, 12.5%; for B=16, just 6.25%. Empirically, this decay leads to significant degradation on reasoning and coding tasks at larger block sizes (Table 2).

The clean stream in z = [x;x˜] already uses standard causal attention on the original tokens, and its logits are computed during the forward pass at no extra cost. By adding an AR loss on these logits, we ensure the model never stops practicing next-token prediction, no matter how large the block size:

L = Lmask + LAR (5)

where LAR = − Lt=1 log pθ(xt | x<t) is the standard next-token prediction loss computed from clean stream logits. The two terms are weighted equally.

With this combined loss, the AR-equivalent signal fraction becomes:

L + L/B 2L

rAR(combined) =

1 + 1/B 2

=

(6)

where the numerator counts L pure AR terms from the clean stream plus L/B AR-equivalent terms from the masked stream (one per block). For B=4, this is 62.5%; for B=16, 53.1%. In all cases the ratio stays above 50%, effectively decoupling the AR signal from block size. The model simultaneously learns to predict from masked context and maintains its original autoregressive competence, rather than replacing one with the other.

By default MARS includes the SFT loss. To isolate its effect, we also evaluate a variant trained without it (denoted “MARS w/o SFT loss”), which uses Lmask alone.

#### 3.4 Inference: Left-to-Right Sliding Window

The inference procedure completes the AR-consistent design by closing gap (4): tokens are always accepted strictly left-to-right, matching AR generation order. Combined with causal attention and right-shifted logits from training, the result is that MARS at inference is indistinguishable from AR generation when only one token is accepted per step, and smoothly extends to multi-token generation when the model is confident.

At each step, B number of [MASK] tokens are appended after the current prefix and the model runs a single forward pass with pure causal attention to obtain logits for all B positions. Starting from the leftmost masked position, tokens are accepted consecutively while maxv p(xt=v) ≥ τ, with at least one token always accepted. The N accepted tokens join the prefix, and N new [MASK] tokens are appended to keep the window at size B. This repeats until generation is complete. The guarantee that at least one token is always accepted ensures that the model degrades gracefully to standard AR decoding when no prediction is confident enough.

The threshold τ directly controls throughput: τ → 1.0 accepts at most one token per step (recovering exact AR behavior), while lower τ accepts more tokens per step at some quality cost. Crucially, τ can be adjusted on-the-fly per request during serving, without retraining or loading a different model. Section 4.4 characterizes the full tradeoff curve.

### 4 Experiments

We organize experiments around three claims: MARS preserves the original AR model’s quality (Section 4.2), the SFT loss is necessary and sufficient for this preservation at scale (Section 4.3), multi-token generation provides a smooth and controllable speed–quality frontier (Section 4.4), and these gains translate to wall-clock speedup in batch inference (Section 4.5).

- Table 2: One-token mode (τ=1.0): MARS vs. AR SFT, compute-matched AR SFT (10 epochs), and Block Diffusion. All models generate one token per forward pass. Bold: best per column within each scale.

Model IFEval BBH MMLU-Pro GPQA GSM8K HumanEval Avg Qwen2.5-0.5B-Instruct

Base (no fine-tune) 27.8 6.1 2.4 13.6 27.4 26.2 17.2 AR SFT (5 ep) 48.4 26.3 11.9 17.9 32.0 35.4 28.7 AR SFT (10 ep) 47.8 26.3 9.3 14.1 28.3 32.3 26.4 Block Diffusion Arriola et al. [2025] (B=4) 47.1 7.5 2.0 17.9 30.6 31.7 22.8 MARS-0.5B w/o SFT loss (B=4) 48.5 27.4 12.3 19.0 29.5 33.5 28.4 MARS-0.5B (B=4) 51.3 26.6 12.4 19.4 32.8 40.2 30.4

Qwen2.5-7B-Instruct

Base (no fine-tune) 63.4 24.0 10.5 11.6 57.3 82.9 41.6 AR SFT 67.0 54.0 43.9 27.5 68.7 78.7 56.6 MARS-7B (B=4) 68.2 54.6 44.4 26.6 73.2 81.7 58.1

#### 4.1 Setup

We evaluate at two scales: Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct [Qwen et al., 2025], both trained on Dolci-Instruct-SFT [Olmo et al., 2025] (∼2M examples). We first train an AR SFT model for 5 epochs with standard next-token prediction, then continue with MARS training on the same data for another 5 epochs. At 0.5B we train with block sizes B ∈ {4,8,16}; at 7B we use B=4. Full hyperparameters are in Appendix A.

We compare against the AR SFT starting point and Block Diffusion [Arriola et al., 2025] (0.5B only). Evaluation uses six benchmarks: IFEval (0-shot) [Zhou et al., 2023], BBH (3-shot) [Suzgun et al., 2022], MMLU-Pro (0-shot) [Wang et al., 2024], GPQA (0-shot) [Rein et al., 2023], GSM8K (0-shot) [Cobbe et al., 2021], and HumanEval (0-shot) [Chen et al., 2021], all with greedy decoding and max 256 new tokens.

#### 4.2 MARS Preserves AR Quality in One-Token Mode

The first claim is that MARS is a strict superset of AR: when generating one token per step, it should match or exceed the original model. Table 2 confirms this at both scales. At 0.5B, MARS achieves 30.4 average versus AR SFT’s 28.7, with a 4.8-point gain on HumanEval (40.2 vs 35.4). At 7B, MARS achieves 58.1 versus 56.6, with notable gains on GSM8K (+4.5) and HumanEval (+3.0). The multi-token capability comes at no cost to one-token quality; the additional masked-prediction training appears to act as a form of data augmentation that slightly improves the AR mode.

To rule out that the gains simply arise from extra training compute, we include a compute-matched AR SFT baseline trained for 10 epochs (same total fine-tuning budget as MARS: 5 epochs AR + 5 epochs masked). Continuing AR SFT beyond 5 epochs actually hurts: the average drops from 28.7 to 26.4, with MMLU-Pro falling from 11.9 to 9.3 and GSM8K from 32.0 to 28.3. The additional AR epochs overfit rather than improve, confirming that MARS’s gains come from the masked prediction objective, not from additional training alone.

Block Diffusion [Arriola et al., 2025], which uses bidirectional intra-block attention, tells the opposite story: it collapses on BBH (7.5) and MMLU-Pro (2.0), scores comparable to the untuned base model. This validates the gap analysis from Section 3.1: not all block prediction formulations are compatible with AR pretraining. The ones that break causality, logits alignment, or generation order destroy the model’s reasoning ability. MARS avoids all three.

#### 4.3 Why Larger Blocks Work: Validating the Signal Decay Hypothesis

Section 3.3 predicted that without the SFT loss, the AR training signal decays as 1/B, causing larger blocks to degrade. Table 3 confirms this directly.

Without the SFT loss, increasing block size from 4 to 16 drops the average from 28.4 to 22.2, a loss of 6.2 points. GSM8K falls from 29.5 to 21.0; HumanEval from 33.5 to 20.7. The degradation is

- Table 3: Effect of SFT loss across block sizes (0.5B, τ=1.0). Without the SFT loss, larger blocks rapidly erode quality; with it, performance is stable across B.

Model IFEval BBH MMLU-Pro GPQA GSM8K HumanEval Avg AR SFT 48.4 26.3 11.9 17.9 32.0 35.4 28.7 MARS w/o SFT loss

B=4 48.5 27.4 12.3 19.0 29.5 33.5 28.4

- B=8 48.5 24.3 11.1 20.8 24.9 28.7 26.4 B=16 42.6 21.7 10.6 16.3 21.0 20.7 22.2 MARS with SFT loss B=4 51.3 26.6 12.4 19.4 32.8 40.2 30.4

- B=8 49.6 26.9 12.0 19.6 32.8 37.2 29.7 B=16 50.7 27.0 12.1 17.9 33.8 36.6 29.7

(a) GSM8K

(b) HumanEval

35

40

30

35

| |
|---|

| |
|---|

HumanEvalPass@1(%)

GSM8KAccuracy(%)

| |
|---|

30

| |
|---|

| |
|---|

25

| |
|---|

25

| |
|---|

| |
|---|

20

20

| |
|---|

| |
|---|

| |
|---|

15

15

| |
|---|

B=4 B=8 B=16

B=4 B=8 B=16

10

| |
|---|

10

5

AR SFT

AR SFT

1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75

1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4

Tokens per Forward Pass

Tokens per Forward Pass

Solid: MARS (with SFT loss) | Dashed: w/o SFT loss | Dotted: AR SFT baseline

- Figure 3: Speed–quality Pareto curves on GSM8K (left) and HumanEval (right). Solid lines: MARS (with SFT loss). Dashed lines: w/o SFT loss. Dotted: AR SFT baseline. With SFT loss, MARS dominates at every operating point on both tasks.

systematic and concentrated in reasoning and coding tasks, exactly the capabilities most sensitive to the quality of the AR signal.

With the SFT loss, the same block size increase causes a drop of only 0.7 points (30.4 to 29.7). GSM8K actually improves from 32.8 to 33.8, and HumanEval drops modestly from 40.2 to 36.6. The SFT loss stabilizes the AR signal ratio above 50% regardless of B (Eq. 6), and the empirical results match this prediction closely.

- Figure 3 shows the same pattern across the full threshold sweep: at every operating point (every tokens-per-forward rate), MARS with SFT loss achieves higher accuracy than without, and the gap widens at larger block sizes. The SFT loss does not just help at τ=1.0; it lifts the entire Pareto frontier.

#### 4.4 A Smooth Speed–Quality Frontier via Confidence Thresholding

Having established that MARS preserves AR quality, we now show that the same model provides a smooth, controllable tradeoff when multi-token generation is enabled. Table 4 reports accuracy at τ=0.95 alongside the change from one-token mode. Full threshold sweep results are in Table 7 (Appendix).

The accuracy cost of multi-token generation is small and predictable. At 0.5B with B=8, the average drops by just 1.1 points (29.7→28.6) while generating 1.49 tokens per forward pass. Most individual benchmarks lose less than 1 point. The largest drops are on IFEval (∼5pp): IFEval evaluates strict adherence to formatting instructions (e.g., “write exactly 3 paragraphs”), and multi-token acceptance tends to skip over format-critical tokens that the model would have produced more carefully in single-token mode. At 7B, the tradeoff is even more favorable: MARS-7B loses only 1.3 points on

- Table 4: Multi-token mode. Each cell shows accuracy with the absolute change from τ=1.0 to τ=0.95 in parentheses. Gray italic rows show tokens accepted per forward pass for each task. Full threshold sweep in Appendix Table 7.

Model IFEval BBH MMLU-Pro GPQA GSM8K HumanEval Avg Qwen2.5-0.5B-Instruct MARS B=4 45.5(−5.8) 26.6(0.0) 11.5(−0.9) 19.9(+0.5) 31.1(−1.7) 39.6(−0.6) 29.0(−1.4)

- tok/fwd 1.20 1.90 1.50 1.36 1.51 1.26 1.46 MARS B=8 44.0(−5.6) 28.3(+1.4) 11.9(−0.1) 19.0(−0.6) 31.7(−1.1) 36.6(−0.6) 28.6(−1.1) tok/fwd 1.23 1.96 1.49 1.46 1.54 1.27 1.49 MARS B=16 45.3(−5.4) 25.9(−1.1) 11.9(−0.2) 17.9(0.0) 31.2(−2.6) 36.6(0.0) 28.1(−1.6)
- tok/fwd 1.21 1.86 1.51 1.43 1.57 1.25 1.47 Qwen2.5-7B-Instruct

MARS B=4 63.0(−5.2) 54.3(−0.3) 44.2(−0.2) 27.7(+1.1) 71.0(−2.2) 80.5(−1.2) 56.8(−1.3) tok/fwd 1.15 2.60 SVG1.47 Copy1.47 1.79 1.60 1.68

###### Step 1

###### Step 2

###### Next block

Forward B=4 [M] tokens. S1 accepts 4, S2 accepts 2, S3 accepts 1. Min=1, cache +1.

S1 idles (block ﬁlled). S2 accepts 2 more (done). S3 accepts 1 more. Min cursor=2, cache +1.

S3 ﬁlls remaining. All samples done. Full block (4 tokens) cached. New [M] appended.

- S1 preﬁx The ans wer is done

- S2 preﬁx A cat [M] [M]

- S3 preﬁx Par [M] [M] [M]

- S1 preﬁx The ans wer is idle

- S2 preﬁx A cat has four done

- S3 preﬁx Par is [M] [M]

- S1 preﬁx The ans wer is [M]*4

- S2 preﬁx A cat has four [M]*4

- S3 preﬁx Par is the cap [M]*4

KV cached Newly cached this step Accepted (not yet cached) [MASK]

- Figure 4: Block-level KV cache for batch inference (Bcache=4, batch size 3). Each step forwards B [MASK] tokens against the cached prefix. The cache advances by the minimum number of tokens accepted across all samples: after Step 1 (S1 accepts 4, S2 accepts 2, S3 accepts 1), one token is cached (green). S1 idles while S2 and S3 continue. Once all samples fill the block, the entire block is cached (yellow) and new [MASK] tokens are appended for the next block.

average (58.1→56.8) while generating 1.68 tokens per forward, reaching 2.60 on BBH where the model produces high-confidence reasoning chains.

Crucially, the 7B model at τ=0.95 (56.8) still exceeds the AR SFT baseline (56.6). The frontier is smooth with no cliff where quality suddenly collapses, giving the serving system fine-grained control. This is the “opt-in” property promised in Section 1: the same checkpoint serves quality-sensitive requests at τ=1.0 and latency-sensitive requests at lower τ, with no model swap and no retraining.

#### 4.5 Wall-Clock Speedup with Block-Level KV Cache

Tokens per forward pass measures algorithmic speedup, but wall-clock throughput is what matters in production. Standard AR decoding benefits from KV cache: each step processes only one new token. MARS processes B masked tokens per step, so without caching, full-sequence recomputation makes it slower than AR at batch sizes above 1.

We implement a block-level KV cache strategy (Figure 4): (1) compute the prefix KV cache once per block via a full forward pass, (2) iterate within the block using the cached prefix, where each inner step only forwards B tokens, (3) once all samples in the batch have filled the block, extend the cache with the completed block and advance to the next. Faster samples idle at block boundaries until the slowest sample finishes. Table 5 compares three configurations on GSM8K (256 questions) with Qwen2.5-7B at τ=0.95.

Block-level KV caching is essential for MARS: without it, throughput actually decreases as batch size grows (127→112→98 tok/s) due to O(T2) full-sequence recomputation per step. With the cache, MARS outperforms AR at every batch size tested. At Batch size=4, the best configuration (Bcache=32) finishes in 161.2s versus AR’s 276.2s, a 1.71× wall-clock speedup. At Batch size=8, MARS achieves 1.60× (105.6s vs 169.1s). At Batch size=16, MARS achieves 1.34× (68.7s vs 91.8s). The speedup is largest at smaller batch sizes where AR’s per-token overhead is proportionally higher. The optimal cache granularity shifts with batch size (Bcache=32 at Batch size=4/8, Bcache=16

- Table 5: Batch inference on GSM8K (256 questions) with Qwen2.5-7B at τ=0.95. Bcache: KV cache synchronization granularity. For each batch size we report tokens/sec, total wall-clock time (seconds), and speedup relative to AR with KV cache (×). Bold: best MARS configuration per batch size.

Batch size=4 Batch size=8 Batch size=16

Method Bcache tok/s time × tok/s time × tok/s time × AR (KV cache) – 143.4 276.2 1.00 236.7 169.1 1.00 434.3 91.8 1.00 MARS (no cache) – 127.1 306.5 0.90 112.4 346.9 0.49 97.7 399.1 0.23 MARS + block cache 4 157.2 248.8 1.11 236.4 165.4 1.02 399.3 97.2 0.94 MARS + block cache 8 196.0 199.4 1.39 289.9 134.4 1.26 484.1 80.1 1.15 MARS + block cache 16 228.5 170.5 1.62 338.0 115.1 1.47 566.0 68.7 1.34 MARS + block cache 32 241.9 161.2 1.71 368.4 105.6 1.60 544.9 71.5 1.28 MARS + block cache 64 239.5 162.0 1.70 342.6 113.8 1.49 383.8 101.5 0.90 MARS + block cache 128 194.2 200.7 1.38 217.9 178.8 0.95 223.9 174.0 0.53 MARS + block cache 256 122.7 317.6 0.87 128.7 302.7 0.56 124.1 313.9 0.29

at Batch size=16), reflecting a tradeoff between amortizing prefix recomputation and synchronization overhead at block boundaries. Accuracy is preserved across all configurations (78–82% GSM8K, within 2–3pp of AR).

### 5 Conclusion

We presented MARS, a lightweight fine-tuning method that gives instruction-tuned AR models the ability to generate multiple tokens per forward pass, with no architectural changes, no additional parameters, and a single checkpoint. The resulting model is a strict superset of the original: in one-token mode it matches or exceeds AR SFT at both 0.5B (+1.7 avg) and 7B (+1.5 avg), and in multi-token mode it achieves 1.5–1.7× throughput with minimal accuracy cost. With block-level KV caching, these gains translate to up to 1.71× wall-clock speedup over AR with KV cache on Qwen2.5-7B.

The method rests on two insights. First, block-masked prediction fails when it unnecessarily departs from AR behavior; closing the three eliminable gaps (attention pattern, logits alignment, generation order) is sufficient to recover baseline quality. Second, the SFT loss on the clean stream preserves the model’s AR competence during masked-prediction training, preventing the AR signal from decaying as 1/B and stabilizing it above 50% regardless of block size.

Due to computation constraints, we evaluate only B=4 for the 7B model. Our 0.5B experiments show that different block sizes yield similar Pareto frontiers on the speed–quality tradeoff (Table 3), with B=4 marginally ahead (less than 0.5pp in average accuracy). We expect this pattern to hold at larger scales, but verifying this remains future work. Other promising directions include: (1) cursor-based cache management that eliminates block-boundary synchronization, (2) adaptive block size selection based on input complexity, and (3) integration with speculative decoding for further acceleration.

Limitations. MARS training concatenates a clean and noisy copy of each sequence, doubling the per-sample sequence length and therefore the training-time compute relative to standard SFT. This overhead is nonetheless lightweight compared to continual pretraining: training requires only

- 5 epochs of SFT data rather than large-scale pre-training corpora. The speed–quality tradeoff at aggressive thresholds (τ < 0.7) shows substantial quality loss, suggesting room for improvement in the acceptance strategy. The block-level KV cache requires batch synchronization at block boundaries, which limits throughput gains at large batch sizes.

### References

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In The Thirteenth International Conference on Learning Representations, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. In International Conference on Machine Learning, pages 5209–5235. PMLR, 2024.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Tianqi Du, Lizhe Fang, Weijie Yang, Chenheng Zhang, Zeming Wei, Yifei Wang, and Yisen Wang. Autoregressive models rival diffusion models at any-order generation. arXiv preprint arXiv:2601.13228, 2026.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding. In International Conference on Machine Learning, pages 14060–14079. PMLR, 2024.

Yonggan Fu, Lexington Whalen, Zhifan Ye, Xin Dong, Shizhe Diao, Jingyu Liu, Chengyue Wu, Hao Zhang, Enze Xie, Song Han, et al. Efficient-dlm: From autoregressive to diffusion language models, and beyond in speed. arXiv preprint arXiv:2512.14067, 2025.

Yifeng Gao, Ziang Ji, Yuxuan Wang, Biqing Qi, Hanlin Xu, and Linfeng Zhang. Self speculative decoding for diffusion large language models. arXiv preprint arXiv:2510.04147, 2025.

Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. Advances in Neural Information Processing Systems, 37: 133345–133385, 2024.

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 6112–6121, 2019.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In International Conference on Machine Learning, pages 15706–15734. PMLR, 2024.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. In The Thirteenth International Conference on Learning Representations, 2025.

Jiatao Gu, James Bradbury, Caiming Xiong, Victor OK Li, and Richard Socher. Non-autoregressive neural machine translation. In International Conference on Learning Representations, 2018.

Mahdi Karami and Ali Ghodsi. Auto-regressive masked diffusion models. arXiv preprint arXiv:2601.16971, 2026.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: Consistency large language models. In Forty-first International Conference on Machine Learning, 2024.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. Advances in neural information processing systems, 35: 4328–4343, 2022.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. In International Conference on Machine Learning, pages 28935–

28948. PMLR, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, pages 32819–32848. PMLR, 2024.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Team Olmo, :, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark,

##### 2023. URL https://arxiv.org/abs/2311.12022.

Junhao Ruan, Bei Li, Yongjing Yin, Pengcheng Huang, Xin Chen, Jingang Wang, Xunliang Cai, Tong Xiao, and JingBo Zhu. Causal autoregressive diffusion language model. arXiv preprint arXiv:2601.22031, 2026.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, Johnna Liu, Deepansha Singh, Zhoujun Cheng, Zhengzhong Liu, Eric Xing, John Thickstun, and Arash Vahdat. Esoteric language models: Bridging autoregressive and masked diffusion llms, 2026. URL https://arxiv.org/abs/2506. 01928.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37: 103131–103167, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging bigbench tasks and whether chain-of-thought can solve them, 2022. URL https://arxiv.org/ abs/2210.09261.

Yao Teng, Han Shi, Xian Liu, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Accelerating auto-regressive text-to-image generation with training-free speculative jacobi decoding. In The Thirteenth International Conference on Learning Representations, 2024.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024. URL https://arxiv.org/abs/2406.01574.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song. dllm: Simple diffusion language modeling, 2026. URL https://arxiv.org/abs/2602.22661.

### A Training Details

- Table 6: Training hyperparameters. Both stages (AR SFT → MARS) use identical settings per model size. Effective batch sizes are matched across scales.

#### Hyperparameter 0.5B 7B

Base model Qwen2.5-0.5B-Instruct Qwen2.5-7B-Instruct Training data Dolci-Instruct-SFT (∼2M examples) Optimizer AdamW Learning rate 5 × 10−6 LR schedule Cosine decay Warmup ratio 0.03 Epochs 5 (per stage) Max sequence length 512 Per-device batch size 48 24 (AR) / 12 (MARS) Gradient accumulation 1 2 (AR) / 4 (MARS) Effective batch size 384 384 Precision bfloat16 Hardware 8× NVIDIA H200 Block sizes tested 4, 8, 16 4 SFT loss Included (MARS) / Excluded (MARS w/o SFT loss)

Evaluation

Generation length 256 tokens Decoding Greedy (temperature = 0) Steps (MARS) 256 (one token per step at τ=1.0)

Training cost. MARS training concatenates a clean and noisy copy of each sequence, doubling the effective sequence length. This incurs additional training cost: for the 0.5B model, AR SFT takes 15 H200-hours while MARS takes 33 H200-hours (2.2×); for the 7B model, 100 vs 202 H200-hours (2.0×). Peak GPU memory usage is approximately 1.5× that of AR SFT. This overhead is modest compared to continual pretraining: both stages use only 5 epochs of SFT data.

### B Threshold Sweep Details

Table 7 reports the full speed–quality tradeoff for MARS and MARS w/o SFT loss across block sizes B ∈ {4,8,16} on GSM8K and HumanEval, sweeping the acceptance threshold τ from 1.0 (one token per step, equivalent to AR) down to 0.5. At τ=0.95, MARS with B=4 achieves 1.51 tokens per forward on GSM8K with only 1.7pp accuracy loss relative to τ=1.0. Lowering τ further increases throughput but degrades quality, particularly for larger block sizes. The SFT loss variant consistently dominates the w/o SFT loss variant across all operating points, confirming that the SFT loss stabilizes performance across the entire Pareto frontier, not just at τ=1.0.

- Table 7: Speed–quality tradeoff on GSM8K (0-shot) and HumanEval (0-shot). Tok/Fwd: average tokens accepted per forward pass. τ=1.0: one token per step (from Table 2). Lower τ accepts more tokens but may reduce accuracy.

GSM8K HumanEval B=4 B=8 B=16 B=4 B=8 B=16 τ Acc T/F Acc T/F Acc T/F P@1 T/F P@1 T/F P@1 T/F MARS

AR SFT 32.0 1.00 32.0 1.00 32.0 1.00 35.4 1.00 35.4 1.00 35.4 1.00 1.0 32.8 1.00 32.8 1.00 33.8 1.00 40.2 1.00 37.2 1.00 36.6 1.00 0.95 31.1 1.51 31.7 1.54 31.2 1.57 39.6 1.26 36.6 1.27 36.6 1.25 0.9 28.8 1.66 30.0 1.71 28.7 1.74 40.2 1.37 36.6 1.38 34.2 1.35

- 0.8 26.9 1.86 26.5 1.94 24.0 1.98 34.8 1.54 26.2 1.57 32.9 1.55

- 0.7 26.0 2.04 22.1 2.17 20.7 2.18 25.0 1.73 16.5 1.81 25.0 1.71

- 0.6 21.5 2.24 18.0 2.43 13.4 2.45 17.7 1.92 11.6 2.01 15.9 1.89

- 0.5 18.6 2.47 10.5 2.75 9.8 2.78 17.7 2.18 8.5 2.27 9.8 2.16 MARS w/o SFT loss

- 1.0 29.5 1.00 24.9 1.00 21.0 1.00 33.5 1.00 28.7 1.00 20.7 1.00 0.95 29.0 1.48 25.1 1.51 21.0 1.50 33.5 1.34 28.7 1.27 20.7 1.23

0.9 28.1 1.62 24.1 1.68 19.6 1.69 33.5 1.46 28.1 1.37 19.5 1.34 0.8 27.1 1.83 20.9 1.91 18.0 1.94 32.9 1.65 23.2 1.57 16.5 1.54

- 0.7 26.2 2.02 18.1 2.13 14.6 2.17 26.2 1.84 13.4 1.86 9.2 1.75 0.6 22.7 2.19 14.6 2.38 11.2 2.42 22.0 2.04 13.4 2.14 4.3 1.96

- 0.5 18.6 2.43 9.8 2.74 7.5 2.77 15.9 2.33 9.2 2.53 4.3 2.35

C Jacobi Decoding Baseline

Table 8: Jacobi decoding [Teng et al., 2024] on the AR SFT checkpoint (0.5B, training-free). Jacobi uses fixed-point iteration: all future positions are initialized with random tokens and iteratively updated via causal forward passes until convergence. Tok/fwd: average tokens accepted per forward pass.

Method IFEval BBH MMLU-Pro GPQA GSM8K HumanEval Avg Tok/fwd

AR SFT 48.4 26.3 11.9 17.9 32.0 35.4 28.7 1.00 Jacobi (on AR SFT) 41.0 19.2 11.7 17.6 36.5 42.1 28.0 1.07

MARS (τ=0.95, B=4) 45.5 26.6 11.5 19.9 31.1 39.6 29.0 1.46

Table 8 compares Jacobi decoding with MARS on the same AR SFT checkpoint. Jacobi achieves only

- 1.07× average speedup (tokens per forward), compared to 1.46× for MARS. The limited speedup is

expected: in Jacobi, each position sees random tokens as context for earlier positions in the generation region. Since the AR model was never trained to predict from incorrect prefixes, its predictions rarely converge in fewer iterations than sequential generation. MARS addresses this directly by training the model to predict from [MASK] placeholders, yielding substantially higher acceptance rates.

Jacobi does have one structural advantage: because it initializes all N output positions at once, the model knows the exact generation length from the start, preventing it from generating beyond the intended boundary. This likely explains why Jacobi scores higher than AR SFT on GSM8K (36.5 vs 32.0) and HumanEval (42.1 vs 35.4), where output length control matters for correctness. On format-sensitive and reasoning tasks where this advantage does not apply, Jacobi drops significantly: IFEval (−7.4) and BBH (−7.1).

### D Acceptance Metric Sensitivity

The sliding-window inference by default accepts tokens left-to-right while a confidence score exceeds a threshold τ. In the main experiments we use the probability of the top token, maxv p(v | ·), as the confidence score. Here we evaluate two alternatives on GSM8K with MARS (B=4):

- • Entropy: H(p) = − v pv log pv. Accept while H ≤ τ. Lower entropy indicates higher confidence.
- • Top-2 margin: ptop

. Accept while margin ≥ τ. A large gap between the best and second-best token indicates high confidence.

− ptop

1

2

34

32

| |
|---|

30

GSM8KAccuracy(%)

| |
|---|

28

| |
|---|

26

24

22

| |
|---|

Probability

20

Entropy

Top-2 Margin

| |
|---|

18

AR SFT

1.4 1.6 1.8 2.0 2.2 2.4 2.6

Tokens per Forward Pass

Figure 5: Speed–quality trade-off under three acceptance metrics (MARS B=4, GSM8K). All three metrics trace similar Pareto frontiers, indicating that the speed–quality trade-off is robust to the choice of acceptance criterion. Entropy and top-2 margin degrade slightly more gracefully than raw probability at comparable tokens per forward pass.

Figure 5 shows the speed–quality frontier for each metric. All three trace similar Pareto curves, confirming that the sliding-window acceptance mechanism is robust to the specific confidence measure. Entropy and top-2 margin show marginally smoother degradation at comparable speedups, but the differences are small. We use probability in the main paper for its simplicity.

