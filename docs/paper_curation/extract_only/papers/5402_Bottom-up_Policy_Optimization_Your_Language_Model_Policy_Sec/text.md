# arXiv:2512.19673v3[cs.LG]29May2026

## Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies

Yuqiao Tan1,2* Minzheng Wang1,2* Shizhu He1,2† Huanxuan Liao1,2 Chengfeng Zhao1,2 Qiunan Lu Tian Liang3 Jun Zhao1,2 Kang Liu1,2 1Institute of Automation, Chinese Academy of Sciences 2University of Chinese Academy of Sciences 3Tencent AI Lab tanyuqiao2025@ia.ac.cn wangminzheng2023@ia.ac.cn shizhu.he@nlpr.ia.ac.cn

### Abstract

###### (a) Transformer Workflow (b) Different Policy Paradigm

Language Model Policy

u ≡

LN

Existing reinforcement learning (RL) approaches treat large language models (LLMs) as a unified policy, overlooking their internal mechanisms. In this paper, we decompose the LLM-based policy into Internal Layer Policies and Internal Modular Policies via the Transformer’s residual stream. Our entropy analysis of internal policy reveals distinct patterns: (1) universally, internal policies evolve from high-entropy exploration in early layers to deterministic refinement in the top layers; and (2) Qwen exhibits an explicit progressive reasoning structure, contrasting with the abrupt convergence in Llama. Furthermore, we discover that optimizing internal layers induces feature refinement, forcing lower layers to capture high-level reasoning representations early. Motivated by these findings, we propose Bottom-up Policy Optimization (BuPO), a novel RL paradigm that reconstructs the LLM’s reasoning foundation from the bottom up by optimizing internal layers in early stages. Extensive experiments on complex reasoning benchmarks demonstrate the effectiveness of BuPO. Our code is available here.

Internal Layer Policy

u Layer ≡

Layer

###### Internal Modular Policy

u ATTN ≡

1

ATTN

1

1

1

u FFN ≡

0

FFN

###### (c) Progressive Reasoning Patterns in Qwen-Series

Figure 1: (a) The residual stream within a Transformer flows from previous layer into self-attention and feedforward network (FFN) sequentially. (b) Any hidden states combined with the unembedding matrix Eu can be transformed into probability distribution P over the vocabulary space, which can be considered a policy.

###### Exploration Expansion

Prediction Convergenc

e

[Figure 1]

[Figure 2]

[Figure 3]

FFN ≈0 FFN <0

FFN >0

Higher Layers

Lower Layers

in its internal residual streams. Mechanistic interpretability provides tools for exposing such internal behavior (Belrose et al., 2023; Gupta et al., 2025), creating an opportunity to connect internal dynamics with more principled policy optimization.

While recent works leverage attention mechanisms to improve RL algorithms (Li et al., 2025; Liu et al., 2025), they largely overlook the information latent in residual streams. Logit-lens (nostalgebraist, 2020) offers initial insights by using the unembedding matrix Eu to decode intermediate layer representations into token space, revealing rich information that evolves across layers and modules (Gupta et al., 2025; Lindsey et al., 2025). Moreover, prior studies have further elucidated the roles of self-attention and feed-forward networks (FFNs) in shaping hidden states (Dai et al., 2022; Yu and Ananiadou, 2024; Jin et al., 2025). Collectively, these internal mechanisms offer a new perspective for algorithmic optimization.

### 1 Introduction

Reinforcement learning (RL) has become central to improving the complex reasoning abilities of large language models (LLMs) (Ouyang et al., 2022; Jaech et al., 2024). The success of DeepSeekR1 (Guo et al., 2025) further highlights reinforcement learning with verifiable rewards (RLVR) as an effective post-training paradigm for improving LLM reasoning across domains (Yang et al., 2025; Team et al., 2025; Yu et al., 2025). However, existing RL methods typically treat an LLM as a unified policy, optimizing only the final output distribution while overlooking the evolving information flow

In this paper, we investigate the evolution of language model policies across layers and modules

*Equal contribution. †Corresponding author.

to facilitate optimization and unravel complex internal reasoning mechanisms. Our formulation is grounded in two key insights. First, the residual stream naturally supports decomposition (Zhang et al., 2025; Lindsey et al., 2025), allowing us to isolate the individual roles of each layer and module (Figure 1(a)). Second, we conclude that the policy is intrinsically equivalent to the token distribution derived from the combination of hidden states H with the unembedding matrix Eu. Based on these, we construct the Internal Layer Policy πLayerl , which captures cumulative reasoning up to layer l, and the Internal Modular Policy πATTNl and πFFNl , which isolates the specific contributions of attention and FFN modules (Figure 1(b)). This decomposition allows us to ask: How does internal reasoning evolve through the model?

Through systematic analysis of commonly used Qwen and Llama series (Meta AI, 2024; Yang et al., 2024, 2025) based on Internal Policy Entropy in a policy-centric view, we uncover both universal and critical architectural differences: (1) Consistent internal reasoning structure. All models exhibit a universal reasoning structure: early layers maintain high entropy for exploring the solution space, while top layers converge to near-zero for final prediction (Lindsey et al., 2025). (2) Distinct internal reasoning pattern. Despite the shared trend, the pace of convergence differs significantly. Llama exhibits a sudden convergence only within the last three layers. In contrast, Qwen models demonstrate a progressive contraction, gradually reducing uncertainty throughout layers. To quantify these dynamics, we introduce Entropy Change. This metric shows that Llama exhibits increased internal entropy and tends to converge only in the final layer, whereas Qwen progressively leverages FFNs to expand exploration in lower layers, integrate parametric knowledge in intermediate layers, and consolidate predictions in upper layers.

These findings have profound implications for RL optimization: Since internal reasoning emerges from lower to higher, we can consider optimization from a bottom-up perspective. We first validate this hypothesis with Internal Policy Alignment (IPA), revealing distinct training dynamics and a remarkable phenomenon of internal reasoning feature refinement. Specifically, the optimized lower layers capture high-level reasoning capabilities by early alignment, providing a robust foundation for subsequent internal reasoning. Motivated by these

insights, we propose Bottom-up Policy Optimization (BuPO), a novel RL paradigm that optimizes fine-grained internal layer policies during the early stages of training to effectively guide the language model policy. By doing so, BuPO reconstructs foundational reasoning abilities and achieves superior performance. Extensive experiments on complex reasoning benchmarks demonstrate the effectiveness of our approach and the unique training dynamics compared to conventional RL methods that optimize the policy as a whole.

In summary, this paper makes the following contributions: (1) We are the first to decompose an LLM policy into internal layer and modular policies, revealing their distinct roles in reasoning. (2) We reveal a universal exploration-to-convergence shift, distinguishing Qwen’s progressive reasoning from Llama’s abrupt convergence. Moreover, we uncover a feature refinement phenomenon where internal optimization drives lower layers to preemptively capture high-level features. (3) We propose BuPO to align internal policies from the bottom up, which reconstructs the foundational reasoning capabilities at lower layers, achieving superior performance on complex reasoning benchmarks.

### 2 Preliminary

In this section, we aim to introduce the basic definitions that help us to understand the decomposition of the language model policy.

#### 2.1 The Residual Stream in Transformer

Transformer-based language models (Vaswani et al., 2017) form the foundation of modern LLMs (Brown et al., 2020). A decoder-only Transformer consists of L stacked layers, each containing a multi-head self-attention (MHSA) module and a feed-forward network (FFN) module.

Following Zhang et al. (2025), we formalize the forward process from input to output. Given an input sequence x = [x1,x2,...,xT], the model produces a probability distribution P over the vocabulary V with N tokens. Let H(2l−2) ∈ RT×dmodel denote the hidden state input to the l-th layer, where T is the sequence length and dmodel is the hidden dimension. The initial state is H(0) projected by E, where E ∈ RN×dmodel is the embedding matrix.

Each layer forwards sequentially through attention and FFN with residual connections: H(2l−1) = H(2l−2) + Al and H(2l) = H(2l−1) + Fl, where Al = MHSA(LN(H(2l−2))) and Fl =

###### Llama-3.2-3B-Instruct

Llama-3.1-8B-Instruct

Qwen2.5-Math-7B

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

12

12.5

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |
|Layer I/O ATTN FFN<br><br>| |
|---|
<br><br>| |
|---|
| |

- 8

- 9

- 10

- 11

10.0

10

7.5

8

Entropy

Entropy

Entropy

5.0

6

2.5

Layer I/O ATTN FFN

Layer I/O

4

ATTN FFN

| |
|---|

| |
|---|

| |
|---|

0.0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28

Layer Index

Layer Index

Layer Index

###### Qwen3-4B

Qwen3-8B

Qwen3-14B

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

12.5

12.5

12.5

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| |Layer I/O| |
|| |
|---|
<br><br>ATTN FFN| | |

| | |
|---|---|
| | |
| | |
| | |
|Layer I/O ATTN<br><br>| |
|---|
| |

10.0

10.0

10.0

7.5

7.5

7.5

Entropy

Entropy

Entropy

5.0

5.0

5.0

2.5

2.5

2.5

Layer I/O ATTN FFN

| |
|---|

FFN

0.0

0.0

0.0

| |
|---|

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36

1 2 3 4 5 6 7 8 9 10111213141516171819202122232425262728293031323334353637383940

Layer Index

Layer Index

Layer Index

- Figure 2: Continuous entropy dynamics of internal policy. The residual stream flows from Hl−1 into Al, Fl, then to the next layer Hl.

FFN(LN(H(2l−1))) are the attention and FFN outputs, and LN(·) denotes layer normalization.

After L layers, the final hidden states are projected to vocabulary logits:

P = softmax(LN(H(2L))ETu), (1)

where Eu ∈ RN×dmodel is the unembedding matrix, and P ∈ RT×N denotes the output distribution.

- 2.2 Reinforcement Learning for Language Model Policy

Language model generation can be formulated as a token-level Markov Decision Process (MDP). At each step t, the state st = [q;o<t] consists of the input and generated tokens so far. The language model policy πθ(·|st) samples the next token ot from vocabulary V , transitioning to st+1 = [st;ot]. To optimize the policy, we maximize:

J (πθ) = Eq∼Q,o∼πθ(·|q) [R(q,o) − βDKL[πθ(·|q))||πref(·|q)],

(2)

where R(q,o) = |to=1| r(st,ot) is the return (Sutton et al., 1998) and πref is a reference policy. We adopt sparse rewards where rt = 0 for t < n and rn ∈ [0,1] indicates task success. Following Hu et al. (2025a), we assume β = 0.

Policy Optimization. We adopt GRPO (Shao

- et al., 2024), which samples a group of responses {o1,...,oG} per question and estimates advan-

tages as Aˆi,t = Ri−std(mean(R)R):

G

1 |oi| |oi|

1 G

JGRPO(πθ) = Eq∼Q,{o

i}Gi=1∼πθold(·|q)

i=1

min ri,tAˆi,t,clip(ri,t,1 − ϵ,1 + ϵ)Aˆi,t ,

t=1

(3) where ri,t = ππθ(oi,t|si,t)

θold(oi,t|si,t) is the importance ratio.

### 3 Language Model Policy Secretly Contains Internal Policies

In this section, we introduce our key insight: the language model policy secretly contains internal policies. We provide implementation details of this section in Appendix A.3.

#### 3.1 Definition of Internal Policy

Residual Stream. In the residual stream of Transformer, the input to any layer equals the sum of all preceding outputs plus the original embedding. The hidden states satisfy:

Hl = H(2l) = H(0) +

l

Ai +

i=1

l

Fj, (4)

j=1

where we denote Hl for simplicity as the output hidden states of layer l, H(2l). According to this, the output of final layer can be regarded as the combination of previous hidden states by HL = H0 + Li=1 Ai + jj=1 Fj.

Internal Policy. During RL, we sample the next token ot from the final layer’s probability distribution, i.e., πθ ≡ P = softmax(LN(HL)ETu). We

Qwen2.5-Math-7B

Qwen3-4B

Qwen3-8B

###### Llama-3.2-3B-Instruct

10

10

10

InternalATTNPolicyInternalFFNPolicy

0.2

MeanEntropyChange

5

5

5

0.1

0

0

0

0.0

5

5

5

0.1

10

10

10

0.2

1 5 10 15 20 25 28

1 5 10 15 20 25 30 36

1 5 10 15 20 25 30 36

1 5 10 15 20 25 28

4

4

4

0.2

2

2

2

MeanEntropyChange

0

0

0

0.1

2

2

2

0.0

4

4

4

6

6

6

0.1

8

8

8

10

10

10

0.2

1 5 10 15 20 25 28

1 5 10 15 20 25 30 36

1 5 10 15 20 25 30 36

1 5 10 15 20 25 28

Layer Index

Layer Index

Layer Index

Layer Index

- Figure 3: Entropy change dynamics of internal policy. The entropy change ∆Hl across layers represents the uncertainty of current policy’s hidden exploration space. ∆Hl > 0 indicates increased exploration, ∆Hl ≈ 0 signifies exploitation of existing knowledge, and ∆Hl < 0 suggests convergence within the reasoning process.

propose that each internal hidden states can be combined with Eu to produce a policy. Specifically, we focus on two granularities. Internal Layer Policy refers to using hidden states from each layer Hl to combine with Eu, and Internal Modular Policy integrates Eu with states from specific module:

πLayerl ≡ PlLayer = softmax(HlETu), (5)

PlATTN = softmax(AlETu),for ATTN PlFFN = softmax(FlETu). for FFN

πModulel ≡

(6)

Each component contributes to the final policy through the residual stream. For instance, HL = Hl + Sl+1, where Sl+1 = Li=l+1 Ai +

L j=l+1 Fj represents contributions from subsequent layers. Hence, understanding these internal components is essential for unraveling how internal reasoning emerges and evolves through the model.

#### 3.2 Internal Policy Entropy Dynamics

In contrast to prior logit-lens approaches (nostalgebraist, 2020) that decode internal states into discrete tokens, we adopt a policy-centric perspective where internal probability distributions are treated as policies. We employ Entropy as our primary probing metric, motivated by its strong correlation with policy behavior (Cui et al., 2025; Cheng et al., 2025). We define Internal Policy Entropy as:

|V |

HLayerl = −

PlLayer,j · log(PlLayer,j), (7)

j=1

where |V | denotes the vocabulary size and we can obtain HFFNl and HATTNl in the same way.

Continuous Entropy Dynamics. Figure 2 shows that internal policy entropy dynamics exhibit consistent patterns across models: early layers maintain high entropy for exploration of the search space, while top layers converge to near-zero entropy. This aligns with findings that lower layers capture semantic information while higher layers aggregate and refine these representations to drive final decision-making (Lindsey et al., 2025).

While the overall entropy pattern is consistent across models, the fine-grained transition dynamics vary. To isolate intrinsic patterns from normalization and residual effects (He et al., 2016; Zhang and Sennrich, 2019), we introduce Entropy Change, which measures the incremental information gain within a single internal policy and is defined as:

∆Hl = HOutputl − HInputl , (8)

where the entropy change is defined as the difference between the internal policy entropy at a module’s input and output. This metric reveals how the exploration space evolves as information propagates through the specific module.

Entropy Change Dynamics of Attention vs. FFN. Self-attention modules (Vaswani et al., 2017) are widely regarded as central to model reasoning, particularly for integrating task-relevant contextual information (Jin et al., 2025; Liu et al., 2025). The upper panel of Figure 3 reveals a clear and model-dependent pattern in the entropy change of self-attention. Specifically, Qwen3 models exhibit consistently positive entropy change across layers (∆HATTNl > 0), indicating sustained expansion of the exploration space during reasoning. In contrast,

Layer 6 Layer6 Layer 26 Layer26 Layer 35 Layer35 Language Model Policy

8000

0.8

10

7000

ResponseLength

0.6

8

6000

Entropy

Reward

5000

6

0.4

4000

4

3000

0.2

2

2000

0.0

0

1000

0 50 100 150 200 250 300

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Training Step

Training Step

Training Step

(a) Rewards

(b) Entropy

(c) Response Length

- Figure 4: Training dynamics of internal policy. Effects of varying the optimized policy on (a) reward, (b) entropy of language model policy πθ, (c) response length. The backbone model is Qwen3-4B.

Qwen2.5-Math-7B shows uniformly negative entropy change, suggesting contraction and earlier convergence driven by attention. Llama models display a weaker but still positive trend, reflecting more conservative exploration dynamics.

Moreover, the FFN modules are widely regarded as the key-value memories of parametric knowledge (Geva et al., 2021; Dai et al., 2022). The lower panel of Figure 3 reveals clear and systematic differences in FFN entropy dynamics across model families. For the Llama models (Meta AI,

- 2024), FFN entropy remains consistently positive but weak across almost all layers, with convergence occurring only at the final layer.

By contrast, the Qwen models (Yang et al., 2024,

- 2025) exhibit a pronounced hierarchical entropy structure in the FFN, following a clear three-stage progression. Taking Qwen3-4B as an example, the lower FFN layers (layers 1–6) show increased en-

tropy (∆HFFNl > 0), corresponding to expanded exploration at the onset of reasoning. This is followed by a broad middle region (layers 7–26) where ∆HFFNl ≈ 0, indicating stable information integration through retrieval and reuse of parametric knowledge (Dai et al., 2022). In the upper layers (layers 27–36) where ∆HFFNl < 0, reflects gradual convergence toward the final prediction.

### 4 Internal Policy Alignment

Building on the preceding analysis, we observe a gradual emergence of internal reasoning: layerinduced policy distributions exhibit distinct roles and entropy dynamics for different models. This motivates a bottom-up question: can an internal policy distribution be refined before optimizing the full policy? We study this through Internal Policy Alignment (IPA), an advantage-weighted clipped

surrogate for a selected internal layer policy:

G

1 G

JIPA(πθ,πLayerl ) = Eq∼Q,{o

i}Gi=1∼πθold(·|q)

i=1

|oi|

1 |oi|

min ρli,tAˆi,t,clip(ρli,t,1 − ϵ,1 + ϵ)Aˆi,t ,

t=1

(9) where ρli,t = π

l Layer(oi,t|q,oi,<t)

πLayer,oldl (oi,t|q,oi,<t). The sample process and Aˆi,t are the same as in Eq. 3. Thus, ρli,t is not an importance correction for the rollout distribution, but an internal policy update ratio that limits the change of the selected layer-induced distribution on reward-driven tokens. IPA transfers reward feedback to intermediate representations while retaining the stability of clipped policy updates. Implementation details are in Appendix A.4.

Different Training Dynamics of Internal Policy. As shown in Figure 4, distinct patterns emerge across optimization of different internal layer policies. For the penultimate layer policy πLayer35 , entropy shows minor fluctuations before aligning with πθ. However, it suffers from repetition causing excessively long responses. In contrast, the last integration region policy πLayer26 exhibits unstable and increased entropy. And the last exploration region policy πLayer6 maintains stable entropy growth, with response lengths converging closer to πθ.

Analysis of Internal Policy Optimization. We further investigate the mechanism behind internal policy alignment. We find that it induces clear feature refinement in internal states. Taking πLayer6 as an example, Figure 5(a) shows that the optimized H6 becomes increasingly similar to final layer representations, suggesting that early layers acquire higher-level reasoning features before full policy optimization. Meanwhile, the entropy change in

(b) Entropy Change of Internal Layer Policy Layer6

(d) KL between Layer6 and Later Layers

(a) Similarity between H6 and Topmost States

(c) PPL Trend

36

[Figure 4]

0.00

1.0

1.28

H=HLayer6 HLayer5

17.5

30

0.05

15.0

0.8

EntropyChange()H

1.26

CosineSimilarity

Perplexity(PPL)

KLDivergence

12.5

0.10

24

0.6

Layer7-36

Training Steps

1.24

10.0

Step 0

Step 10 Step 20 Step 30 Step 40 Step 50

0.15

7.5

0.4

18

1.22

5.0

0.20

Step 100 Step 150 Step 200 Step 250 Step 300

0.2

2.5

12

1.20

0.25

0.0

0.0

7

10 15 20 25 30 35

0 50 100 150 200 250 300

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Training Step

Training Step

Training Step

Target Layer Index

- Figure 5: Analysis of internal policy optimization. (a) Similarity between the hidden states of optimized layer 6 and the higher layers. (b) Entropy change ∆HLayer6 of the optimized internal layer policy πLayer6 . (c) The PPL trend of πθ. (d) KL divergence trends between the optimized πLayer6 and later layer policies. The backbone model is Qwen3-4B.

Figure 5(b) shows progressive convergence of the optimized internal policy. Figure 5(c,d) reveals the trade-off of internal policy alignment: while the KL trends show that πLayer6 remains well aligned with later-layer policies during the early stage, this alignment deteriorates with prolonged optimization and coincides with PPL degradation, suggesting that limited internal alignment steps are preferable.

### 5 Bottom-up Policy Optimization

All prior RL methods for LLMs optimize πθ in a holistic manner (Ouyang et al., 2022; Rafailov et al., 2023; Guo et al., 2025). In Section 3, we decompose the final output HL into an intermediate representation Hl and the subsequent residual contribution Sl+1. This decomposition suggests that aligning the internal policy πLayerl associated with Hl may facilitate alignment of the overall policy πθ. We empirically support this intuition in Section 4, where we optimize the internal policy alone and observe pronounced feature refinement in lower layers. These findings further motivate a bottom-up alignment strategy: by aligning internal policies first, we explore whether the overall policy can be guided to reason more effectively.

To this end, we propose Bottom-up Policy Optimization (BuPO), which sequentially optimizes the internal layer policies πLayerl followed by the language model policy πθ:

JBuPO = JIPA, scur ≤ sinter, JGRPO, scur > sinter,

(10)

where scur denotes the current training step, sinter the training steps of the internal layer policy.

Training Setup. Using Qwen (Yang et al., 2025) and Llama (Grattafiori et al., 2024; Wang

- et al., 2025b) backbones, we assign πLayer6 for both Qwen3-4B and Qwen3-8B, πLayer27 for Llama-OctoThinker-3B-Base, and πLayer31 for

Llama-OctoThinker-8B-Base. For BuPO, we use the entropy-indicated target layer as the internal policy to optimize, and discuss this choice through the layer-policy ablation in Section 5.3. Refer to

- Appendix A.5 for detailed setup. Evaluation Setup. We evaluate BuPO against

several RL baselines, including GRPO, PPO (Sutton et al., 1998), Reinforce++ (Hu, 2025) and RLOO (Ahmadian et al., 2024). The benchmarks cover: MATH (Lightman et al., 2023), AMC23 (MAA, 2023), AIME24, and AIME25 (MAA, 2024, 2025). Due to high output variance in reasoning tasks, we report Avg@K (Pass@1 averaged over K outputs). For AIME24/25, we set K = 32, and for others K = 16. Additionally, we evaluate an unbiased Pass@K metric for a comprehensive evaluation. Detailed evaluation setups are in Appendix A.5.

5.1 Main Results

As shown in Table 1, BuPO consistently improves over baselines across models and benchmarks. On Qwen3-4B, BuPO achieves gains of 4.69 and 2.30 points on AIME24 and AIME25, respectively, with similar improvements on Qwen3-8B (+4.58 and +0.76). The Llama series further validates this trend, with average improvements of 1.01 points on Llama-OctoThinker-3B-Base and 3.68 points on Llama-OctoThinker-8B-Base. We further evaluate BuPO’s generalization on out-of-domain benchmarks spanning general reasoning and coding in

- Appendix B.1. These results demonstrate that aligning internal layer policies provides a consistent and effective training signal for improving reasoning.

For a comprehensive evaluation, we report the averaged Pass@K results of BuPO and GRPO, with K ranging from 1 to 256. In Figure 6, BuPO consistently achieves a favorable trade-off across a wide range of K. On Qwen3-8B, BuPO attains the best performance for all K values, while on

Methods AMC (Avg@16) MATH500 (Avg@16) AIME24 (Avg@32) AIME25 (Avg@32) 11Average11 Qwen3-4B

Vanilla 67.66 80.29 23.20 18.60 47.44 PPO 77.03 83.64 32.60 27.60 55.22 Reinforce++ 63.44 80.63 17.40 18.65 45.03 RLOO 77.66 82.73 30.83 24.79 54.00 GRPO 76.88 82.41 32.19 28.85 55.08 BuPO 81.09+4.21 84.90+2.49 36.88+4.69 31.15+2.30 58.51+3.43

Qwen3-8B

Vanilla 67.34 80.46 26.98 19.17 48.49 PPO 87.03 86.20 37.81 22.60 58.41 Reinforce++ 82.66 86.05 41.77 31.15 60.41 RLOO 86.41 87.32 46.67 33.02 63.36 GRPO 85.94 88.05 49.48 33.54 64.23 BuPO 89.22+3.28 87.76 54.06+4.58 34.38+0.76 66.36+2.13

Llama-OctoThinker-3B-Base Vanilla 1.24 5.26 0.21 0.00 1.68 PPO 22.19 43.23 1.04 0.31 16.69 Reinforce++ 9.38 11.59 0.00 0.10 5.27 RLOO 27.03 41.93 2.19 0.21 17.84 GRPO 27.50 46.07 0.63 0.10 18.58 BuPO 27.50+0.00 49.79+3.72 0.63+0.00 0.42+0.32 19.59+1.01

Llama-OctoThinker-8B-Base Vanilla 4.53 9.84 0.52 0.10 3.75 PPO 31.72 56.97 1.56 1.04 22.82 Reinforce++ 34.69 59.55 7.72 3.75 26.43 RLOO 27.66 55.97 3.54 1.56 22.18 GRPO 34.84 56.89 2.50 2.19 24.11 BuPO 37.66+2.82 62.05+5.16 4.69+2.19 6.77+4.58 27.79+3.68

Table 1: Avg@K results on MATH500, AMC23, AIME24 and AIME25. Bold and underlined denote the best and second best.

Qwen3-4B

Qwen3-8B

Llama-OctoThinker-3B-Base

Llama-OctoThinker-8B-Base

85

90

55

60

80

50

85

45

75

50

80

40

70

35

75

40

65

30

70

60

25

30

GRPO

65

55

20

BuPO(Ours)

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Figure 6: Average Pass@K results on MATH500, AMC23, AIME24 and AIME25.

Qwen3-4B, the only exception occurs at K = 256. For all Llama models, BuPO achieves the best results across all K values, yielding gains of 7.48 points on Llama-OctoThinker-3B-Base and 7.93 points on Llama-OctoThinker-8B-Base in Pass@256. These results further indicate the IPA effectively enhances the reasoning capacity of πθ. 5.2 Analysis

Training Dynamics of BuPO. We further visualize the training dynamics of BuPO. As shown in Figure 7, by aligning the internal layer policy at an early stage, all models exhibit enhanced entropy exploration initially. For Qwen models, optimizing layer 6 maintains stable exploration, consistent with the later layer-policy ablation. For Llama, we observe increased entropy during the bottom

alignment stage, indicating that feature refinement in lower layers effectively provides a larger exploration space for πθ.

#### 5.3 Ablation Study

Effect of Bottom Optimization Steps. We further study how the number of bottom optimization steps sinter affects BuPO. As shown in Table 2, moderate internal policy alignment consistently improves over GRPO on Qwen3-4B. Specifically, setting sinter = 30 improves the average score from 55.08 to 58.51, while increasing it to 50 or 70 steps causes a drop. These results are consistent with Figure 5: limited bottom optimization promotes internal policy alignment, whereas excessive alignment destabilizes the language model policy and harms final performance.

Qwen3-8B

Qwen3-4B

0.40

BuPO ( Layer5 )

BuPO ( Layer6 ) BuPO ( Layer26 ) BuPO ( Layer35 ) GRPO

0.30

GRPO

0.35

0.30

0.25

0.25

0.20

Entropy

Entropy

0.20

0.15

0.15

0.10

0.10

0.05

0.05

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Training Step

Training Step

Llama-OctoThinker-8B-Base

Llama-OctoThinker-3B-Base

BuPO ( Layer31 )

BuPO ( Layer27 )

1.0

GRPO

1.75

GRPO

1.50

0.8

1.25

Entropy

Entropy

0.6

1.00

0.75

0.4

0.50

0.2

0.25

0.0

0 50 100 150 200 250 300

0 50 100 150 200 250 300

Training Step

Training Step

Figure 7: Entropy dynamics during training with GRPO and BuPO with different internal policy.

Effect of Optimized Internal Policy. We further study the effect of choosing different optimized internal policies πLayerl . As shown in Table 3, the best-

performing layer varies across backbones: πLayer6 achieves the best performance on both Qwen3-4B

and Qwen3-8B, while πLayer26 and πLayer30 perform best on OctoThinker-3B and OctoThinker-8B, respectively. Despite this variation, these bestperforming layers share a common pattern in Figure 3: each corresponds to the last layer with a positive FFN exploration signal ∆HFFNl > 0, i.e., the final layer before the FFN entropy dynamics shift from exploration to integration or convergence. For clarity, we refer to this entropy-indicated transition point as the Boundary Layer. Across all four backbones, optimizing the Boundary Layer achieves the best average performance, suggesting that the entropy-change pattern provides a useful practical signal for choosing the target internal policy. We analyze the computational cost of entropybased inspection in Appendix A.6, showing that its overhead is small compared with downstream policy training.

### 6 Related Work

Reinforcement Learning with Verifiable Rewards. Recently, Reinforcement Learning with Verifiable Rewards (RLVR) has gained traction for its ability to foster LLM reasoning using rule-based rewards (Guo et al., 2025; Yang et al., 2025; Wang et al., 2025a; Tan et al., 2025b; Wang et al., 2026b; Tan et al., 2026; Wang et al., 2026a; Liao et al.,

- 2026). In this work, we shift the focus to inter-

Methods AMC MATH500 AIME24 AIME25 Average Qwen3-4B

GRPO 76.88 82.41 32.19 28.85 55.08 BuPO (sinter = 30) 81.09 84.90 36.88 31.15 58.51 BuPO (sinter = 50) 62.66 79.00 14.17 12.60 42.11 BuPO (sinter = 70) 14.14 25.20 0.21 0.00 9.89

- Table 2: Ablation of sinter with fixed layer policy πLayer6 .

Layer Policy AMC MATH500 AIME24 AIME25 Average Qwen3-4B

Boundary Layer (πLayer6 ) 81.09 84.90 36.88 31.15 58.51 πLayer26 81.56 84.40 35.42 30.52 57.98

- πLayer35 82.66 85.14 35.42 30.12 58.34 Qwen3-8B Boundary Layer (πLayer6 ) 89.22 87.76 54.06 34.38 66.36 πLayer21 75.55 84.40 32.60 21.67 53.56

- πLayer35 83.75 86.51 42.81 29.58 60.66 OctoThinker-3B Boundary Layer (πLayer26 ) 27.50 49.79 0.63 0.42 19.59 πLayer16 16.17 11.40 0.00 0.00 6.89 πLayer6 7.80 9.60 0.21 0.10 4.43 OctoThinker-8B Boundary Layer (πLayer30 ) 37.66 62.05 4.69 6.77 27.79 πLayer16 3.91 12.20 0.00 0.00 4.03 πLayer6 7.58 9.40 0.00 0.10 4.27

- Table 3: Ablation of the optimized internal policy πLayerl . The Boundary Layer denotes the target layer selected for BuPO, corresponding to the last layer with positive FFN entropy change.

nal policies and propose bottom-up policy optimization, which directly optimizes internal layer policies in early training stages. This targeted optimization refines internal reasoning representations and ultimately leads to improved performance.

Interpretability of LLMs. Interpretability studies have opened LLM black boxes by analyzing how models reason, store knowledge, and route information internally (Tan et al., 2025a; Gupta et al., 2025; Hu et al., 2025b), especially through selfattention (Zhou et al., 2025; Jin et al., 2025) and FFN (Dai et al., 2022; Meng et al., 2022). In this work, we conduct a systematic analysis of hidden states from a policy-centric perspective, revealing structured internal reasoning patterns that motivate internal policy alignment which refines intermediate features before full policy optimization.

### 7 Conclusion

In this paper, we decompose a language model policy into internal layer and modular policies, revealing systematic reasoning patterns through entropy analysis. We observe a broad transition from high-entropy exploration in early layers to deterministic convergence in higher layers, with Qwen models showing a progressive reasoning structure contrasting with the abrupt convergence observed

in Llama. Motivated by these findings, we propose Bottom-up Policy Optimization (BuPO), an RL paradigm that aligns internal layer policies during early training. Extensive experiments on complex reasoning benchmarks demonstrate its effectiveness. Further analysis shows that internal policy alignment refines foundational reasoning features, thereby improving the reasoning capacity of the overall policy.

### Limitations

While BuPO demonstrates consistent improvements by leveraging internal policy optimization, several practical limitations offer avenues for future research. (1) Similar to existing RLVR methods, BuPO requires substantial computational resources for rollout generation, reward evaluation, and policy optimization. Our experiments were conducted on a single node with 8 NVIDIA A100 GPUs. Such training requirements may limit accessibility for researchers with constrained computational budgets, and improving the efficiency of RL training remains an important direction for the community. (2) BuPO is studied in reasoning-oriented settings, where models often generate long reasoning trajectories before producing final answers. This increases both training and evaluation costs, especially when evaluating reasoning performance with large-sample metrics such as Pass@K with 300 sampled responses. Overall, we hope BuPO inspires the design of more effective RL algorithms that make use of internal model structures, and encourages closer connections between reinforcement learning and interpretability for reasoning language models.

### Ethical considerations

Our approach does not introduce ethical concerns. The datasets we used are public, and there are no privacy issues.

### References

Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, and Hao Peng. 2025. The unreasonable effectiveness of entropy minimization in LLM reasoning. In Proceedings of NeurIPS.

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting REINFORCE-style optimization for learning from human feedback in LLMs. In Proceedings of ACL, pages 12248–12267.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. 2023. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Proceedings of NeurIPS, volume 33, pages 1877–1901.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. 2025. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, and 1 others. 2025. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of ACL, pages 8493–8502.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are keyvalue memories. In Proceedings of EMNLP, pages 5484–5495.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad AlDahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, and 1 others. 2025. Deepseekr1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Akshat Gupta, Jay Yeung, Gopala Anumanchipalli, and Anna Ivanova. 2025. How do llms use their depth? arXiv preprint arXiv:2510.18871.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of ACL, pages 3828–3850.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of CVPR, pages 770–778.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, and 1 others. 2025. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proceedings of NeurIPS Datasets and Benchmarks Track.

Jian Hu. 2025. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262.

Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, and Heung-Yeung Shum. 2025a. Openreasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290.

Yi Hu, Cai Zhou, and Muhan Zhang. 2025b. What affects the effective depth of large language models? In Mechanistic Interpretability Workshop at NeurIPS 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, and 1 others. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Mingyu Jin, Kai Mei, Wujiang Xu, Mingjie Sun, Ruixiang Tang, Mengnan Du, Zirui Liu, and Yongfeng Zhang. 2025. Massive values in self-attention modules are the key to contextual knowledge understanding. In Proceedings of ICML.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of SOSP, pages 611–626.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, and 1 others. 2022. Solving quantitative reasoning problems with language models. In Proceedings of NeurIPS, volume 35, pages 3843– 3857.

Yang Li, Zhichen Dong, Yuhan Sun, Weixun Wang, Shaopan Xiong, Yijia Luo, Jiashun Liu, Han Lu, Jiamang Wang, Wenbo Su, and 1 others. 2025. Attention illuminates llm reasoning: The preplan-andanchor rhythm enables fine-grained policy optimization. arXiv preprint arXiv:2510.13554.

Huanxuan Liao, Zhongtao Jiang, Yupu Hao, Yuqiao Tan, Shizhu He, Ben Wang, Jun Zhao, Kun Xu, and Kang Liu. 2026. Resadapt: Adaptive resolution for efficient multimodal reasoning. arXiv preprint arXiv:2603.28610.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In Proceedings of ICLR.

Jack Lindsey, Wes Gurnee, Emmanuel Ameisen, Brian Chen, Adam Pearce, Nicholas L. Turner, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, and 8 others. 2025. On the biology of a large language model. Transformer Circuits Thread.

Runze Liu, Jiakang Wang, Yuling Shi, Zhihui Xie, Chenxin An, Kaiyan Zhang, Jian Zhao, Xiaodong Gu, Lei Lin, Wenping Hu, and 1 others. 2025. Attention as a compass: Efficient exploration for processsupervised rl in reasoning models. arXiv preprint arXiv:2509.26628.

- MAA. 2023. American mathematics contest 12 (amc 12).
- MAA. 2024. American invitational mathematics examination (aime).
- MAA. 2025. American invitational mathematics examination (aime).

Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Proceedings of NeurIPS.

Meta AI. 2024. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. Accessed: 2025-11-03.

nostalgebraist. 2020. interpreting gpt: the logit lens. LessWrong.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Proceedings of NeurIPS, volume 35, pages 27730–27744.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Proceedings of NeurIPS, volume 36, pages 53728–53741.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of EuroSys, pages 1279–1297.

Richard S Sutton, Andrew G Barto, and 1 others. 1998. Reinforcement learning: An introduction, volume 1. MIT press Cambridge.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and 1 others. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051.

- Yuqiao Tan, Shizhu He, Kang Liu, and Jun Zhao. 2025a. Neural incompatibility: The unbridgeable gap of cross-scale parametric knowledge transfer in large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21586– 21601.
- Yuqiao Tan, Shizhu He, Kang Liu, and Jun Zhao. 2025b. The zero-step thinking: An empirical study of mode selection as harder early exit in reasoning models. arXiv preprint arXiv:2510.19176.

Yuqiao Tan, Minzheng Wang, Bo Liu, Zichen Liu, Tian Liang, Shizhu He, Jun Zhao, and Kang Liu. 2026. From p(y|x) to p(y): Investigating reinforcement learning in pre-train space. arXiv preprint arXiv:2604.14142.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, and 1 others. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of NeurIPS, page 6000–6010.

Minzheng Wang, Yongbin Li, Haobo Wang, Xinghua Zhang, Nan Xu, Bingli Wu, Fei Huang, Haiyang Yu, and Wenji Mao. 2025a. Adaptive thinking via mode policy optimization for social language agents. arXiv preprint arXiv:2505.02156.

Minzheng Wang, Run Luo, Yanbo Wang, Zichen Liu, Yuqiao Tan, Tao Tan, Xu Nan, Yinhe Zheng, and Wenji Mao. 2026a. Breaking the impasse: Dualscale evolutionary policy training for social language agents. arXiv preprint arXiv:2605.08721.

Yanbo Wang, Minzheng Wang, Jian Liang, Lu Wang, Yongcan Yu, and Ran He. 2026b. Mitigating the safety-utility trade-off in llm alignment via adaptive safe context learning. arXiv preprint arXiv:2602.13562.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, and 1 others. 2024. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290.

Zengzhi Wang, Fan Zhou, Xuefeng Li, and Pengfei Liu. 2025b. Octothinker: Mid-training incentivizes reinforcement learning scaling. arXiv preprint arXiv:2506.20512.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, and 1 others. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025. DAPO: An open-source LLM reinforcement learning system at scale. In Proceedings of NeurIPS.

Zeping Yu and Sophia Ananiadou. 2024. Neuron-level knowledge attribution in large language models. In Proceedings of EMNLP, pages 3267–3280.

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. In Proceedings of NeurIPS, volume 32.

Honglin Zhang, Qianyue Hao, Fengli Xu, and Yong Li. 2025. Reinforcement learning fine-tuning enhances activation intensity and diversity in the internal circuitry of llms. arXiv preprint arXiv:2509.21044.

Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, and Yongbin Li. 2025. On the role of attention heads in large language model safety. In Proceedings of ICLR.

### A Detailed Experiment Settings

#### A.1 The Models for Experiments

We summarize all models used in our analysis and experiments in Table 4. These models are categorized into three types: Mix, Base, and Instruct.

#### A.2 The Template for Experiments

We adopt the following template for all experiments involving Qwen models, building upon the QwenMath template used for Qwen2.5 (Yang et al., 2024) and the Qwen-Nothinking template for Qwen3.

Qwen-Math Template <|im_start|>system

Please reason step by step, and put your final answer within \boxed{}. <|im_end|> <|im_start|>user {problem} <|im_end|> <|im_start|>assistant

Figure 8: Prompt template for Qwen-math.

Qwen3-NoThinking Template <|im_start|>system Please reason step by step, and put your final answer within \boxed{}. <|im_end|> <|im_start|>user {problem} <|im_end|> <|im_start|>assistant <think> </think>

Figure 9: Prompt template for Qwen3 NoThinking mode.

For training the Llama-OctoThinker models, we adopt the original prompt in Wang et al. (2025b) to ensure performance.

#### A.3 Implementation of Internal Policy Entropy Analysis

In this section, we detail the implementation of internal policy entropy analysis. Our primary objective is to extract internal hidden states during the

OctoThinker Template

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: You must put your answer inside \boxed{} and Your final answer will be extracted automatically by the \boxed{} tag.

{problem} Assistant:

Figure 10: Prompt template for OctoThinker.

forward pass. In the main experiments, we evaluate model-generated responses on the MATH test set (Hendrycks et al., 2021). Entropy is computed at the token level for each layer and module, and then averaged over all generated tokens. We find that the intrinsic reasoning patterns remain stable across different tasks, e.g., commonsense question answering (Rein et al., 2024).

The computation of internal policy entropy is illustrated in the pseudo-code below. We abstract the entropy computation as a function H(·). Accordingly, the entropy change of the internal layer policy is defined as:

∆HLayerl = H(Hl) − H(Hl−1). (11)

For the two core Transformer submodules, the entropy changes are computed separately. Specifically, for the self-attention module, we define:

∆HATTNl = H(Al) − H(LN(H(2l−2))), (12)

and for the feed-forward network (FFN), we define:

∆HFFNl = H(Fl) − H(LN(H(2l−1))). (13)

These definitions allow us to quantify how each layer and submodule contributes to the evolution of the internal policy entropy.

Discussion: Comparison to the Logit Lens. Notably, our definition of internal policy differs from the logit-lens (nostalgebraist, 2020), particularly in how layer normalization (LN) is handled. To clarify this distinction, we provide a systematic comparison between the two formulations in Table 6.

###### Model Huggingface Type Layers

Qwen3-4B https://huggingface.co/Qwen/Qwen3-4B Mix 36 Qwen3-8B https://huggingface.co/Qwen/Qwen3-8B Mix 36 Qwen3-14B https://huggingface.co/Qwen/Qwen3-14B Mix 40 Qwen3-4B-Base https://huggingface.co/Qwen/Qwen3-4B-Base Base 36 Qwen2.5-Math-7B https://huggingface.co/Qwen/Qwen2.5-Math-7B Base 28 Qwen3-4B-Instruct-2507 https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 Instruct 36 Llama-3.2-3B-Instruct https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct Instruct 28 Llama-3.1-8B-Instruct https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct Instruct 32 Llama-OctoThinker-3B-Base https://huggingface.co/OctoThinker/OctoThinker-3B-Long-Base Base 28 Llama-OctoThinker-8B-Base https://huggingface.co/OctoThinker/OctoThinker-8B-Long-Base Base 32 DeepSeek-Math-7B-Base https://huggingface.co/deepseek-ai/deepseek-math-7b-base Base 30 DeepSeek-R1-Distill-Qwen-7B https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B Instruct 28

Table 4: Detailed information about the selected models is provided. "Mix" refers to models that support both thinking and non-thinking modes. "Base" denotes the pre-trained model only. "Instruct" indicates models that undergo further fine-tuning based on the Base model to enhance instruction-following capabilities.

#### Calculation of Internal Policy Entropy (PyTorch Implementation)

# Get layer hidden states by register hook hidden_state = get_from_hook()

# Compute logits in the same way as in the original forward pass logits = self.model.lm_head(hidden_state)

# Apply softmax for normalization probs = torch.softmax(logits, dim=-1)

# Apply log_softmax for speedy computation log_probs = torch.log_softmax(logits, dim=-1)

# Calculate internal layer policy entropies = -(probs * log_probs).sum(dim=-1)

Figure 11: The pseudocode of internal policy entropy calculation.

Our definition adopts a policy-centric perspective, treating the internal hidden states as explicitly policy distribution. In contrast, the logit-lens is primarily designed to project hidden states into the discrete vocabulary space in order to inspect the most likely output tokens at intermediate layers. We intentionally omit LN based on empirical considerations: in our analysis experiments, incorporating LN leads to less stable entropy dynamics and weaker interpretability than those shown in Figures 2 and 3.

Extensive analyses and experiments with BuPO further demonstrate that our formulation of internal policy serves as a robust interpretability tool for uncovering internal reasoning mechanisms in LLMs. As shown in Table 5, we also provide an ablation on layer normalization, finding that removing LN yields consistently better performance, which validates our design choice of omitting it in BuPO.

Methods AMC MATH AIME24 AIME25 Qwen3-4B

BuPO-wo-LN (Ours) 81.09 84.90 36.88 31.15 BuPO-w-LN 80.08 84.80 32.29 30.00

Qwen3-8B

BuPO-wo-LN (Ours) 89.22 87.76 54.06 34.38 BuPO-w-LN 86.41 87.40 47.81 34.21

Table 5: Ablation on layer normalization in BuPO.

A.4 Implementation of Internal Policy Alignment

In internal policy alignment, namely IPA, at each optimization step, we select a specific internal layer l and optimize its internal policy πLayerl , defined as πLayerl = softmax(HlETu), where Hl is the hidden states at layer l. The gradient flow for internal policy optimization is determined by the residual structure of the Transformer, where the hidden state Hl is a function of all parameters from layer 1 to l,

Logit Lens Ours Perspective Discrete Token Policy Distribution Definition LN(Hl)ETu softmax(HlETu) Trainable ✗ ✓

- Table 6: Comparison between the logit lens and our definition

but is independent of parameters in higher layers.

Formally, for any parameter θk in layer k, the gradient of the IPA loss with respect to θk can be expressed using the chain rule:

∂JIPA(πLayerl ) ∂θk

=

∂πLayerl ∂Hl ·

∂Hl ∂θk

∂JIPA ∂πLayerl ·

(14)

Due to the residual connections, ∂∂θHl

̸= 0 only when k ≤ l, and is zero otherwise. Thus, the gradients for different layers can be summarized as:

k

 

l Layer

∂πLayerl · ∂π

∂Hl · ∂∂θHl

∂JIPA(πLayerl ) ∂θk

∂JIPA

, if k ≤ l 0, if k > l (15)

=

k



This means that, during IPA optimization for layer l, only the parameters of layers 0 through l and unembedding matrix Eu are updated, while all higher layers (k > l) remain unaffected. This targeted gradient flow ensures that internal policy optimization provides direct supervision to the selected layer and all lower layers, strengthening foundational reasoning capabilities without interfering with higher-level representations.

Training Setup. Based on the findings in Section 3, we select Qwen3-4B (Yang et al., 2025) in non-thinking mode for investigation. For the training set, we randomly sample 5k entries from DeepMath-103k (He et al., 2025). We train the models using the verl framework (Sheng et al., 2025). The prompt batch size is 128, with 8 rollouts generated per prompt. The sampling temperature during training is set to 1.0, and the maximum context length is set to 9,216 tokens. We update the model with a mini-batch size of 32 for 300 steps and a learning rate of 1e-6. These stage-wise FFN entropy patterns suggest that transition points between exploration and later integration/convergence may provide informative candidates for internal policy optimization. For Qwen3-4B, the region

boundaries identified in Section 3.2 lie at layers 6 and 26. Additionally, we focus on aligning the internal policy of the penultimate layer, which is critical for the final prediction. Accordingly, we compare the internal policies πLayer6 , πLayer26 , and πLayer35 optimized via IPA, against the overall policy πθ optimized with GRPO (He et al., 2025).

#### A.5 Implementation of Main Experiments

Detailed Training Setup. Specifically, we focus on the Qwen3 series, using Qwen3-4B and Qwen3-8B (Yang et al., 2025). Meanwhile, we select Llama-OctoThinker-3B-Base and Llama-OctoThinker-8B-Base from the Llama series, as these models demonstrate improved training behavior after mid-training based on Llama-3.2-Base (Wang et al., 2025b). We set sinter = 30 for Qwen3-4B and sinter = 20 for the other LLMs. We implement GRPO and other baseline algorithms using the veRL framework (Sheng et al., 2025). Across all algorithms and model variants, we adopt a unified set of hyperparameters, as reported in Table 8, and do not employ entropy regularization or KL-based losses. For PPO, the critic network is trained separately with a learning rate of 1 × 10−5. All experimental results are averaged over three random seeds.

Evaluation Setup. We use vLLM (Kwon et al., 2023) with temperature 1.0 and top_p 1.0. This metric is defined as Pass@K := Ex∼D 1 − nK−c / K n , where c denotes the number of correct completions out of n generated responses. To reduce evaluation variance on those datasets, we set n = 300.

#### A.6 The Cost of Internal Policy Analysis

The internal policy analysis is a lightweight onetime diagnostic used to identify the layer for bottom-up policy optimization. All experiments are conducted on a single node with 8 NVIDIA A100 GPUs. As shown in Figure 11, for each sample, the analysis only requires a standard forward pass: we register hooks to cache intermediate hidden states, project them with the shared unembedding matrix, and compute token-level entropy. Therefore, it introduces no backward pass, policy update or reference-model pass.

Time Cost. Taking Qwen3-4B as an example, the full RL training takes 13.22 hours, i.e., about 47,592 seconds. In contrast, the internal policy entropy analysis takes only 209.25 seconds. This cor-

###### Metric RL Training Analysis Ratio

Total Time 13.22h (∼47,592s) 209.25s ∼1:227.4 Total FLOPs 12288S TFLOPs 17.6S TFLOPs ∼1:698.2

- Table 7: Cost comparison between RL training and internal policy analysis on Qwen3-4B.

Hyperparameter Value

Optimizer AdamW Policy learning rate 1e−6 Critic learning rate 1e−5 (for PPO) Training batch size 128 prompts Samples per prompt 8 Mini-batch size 32 prompts Policy updates per rollout 16 Max prompt length 1024 tokens Max response length 7168 tokens (Qwen) / 3072 (Llama) Rollout temperature 1.0 Clip range ϵ 0.2

Table 8: RL Hyperparameters

responds to about 0.44% of the RL training time, or an analysis-to-training ratio of approximately 1 : 227.4. Such a small one-time cost is sufficient to recover stable internal reasoning patterns for selecting the optimized internal layer.

FLOPs Cost. Let P denote the number of model parameters, B the number of analyzed samples, and S the average token count per sample. A typical RL step contains generation, policy forward, backward/update, and reference passes, costing roughly 10PS FLOPs per sample. For Qwen3-4B, with 300 steps and 1,024 samples per step, the total training cost is approximately 300 × 1024 × 10 ×

- 4B × S ≈ 12288S TFLOPs. By comparison, the analysis requires only a single forward pass, about 2PBS, plus entropy computation dominated by LM-head projections across internal layers. The total cost is approximately 17.6S TFLOPs, yielding an analysis-to-training FLOPs ratio of about 1 : 698.2.

Moreover, the analysis can be directly performed on the baseline model rollouts at step 0, requiring no additional rollout collection. These results show that internal policy analysis is substantially cheaper than downstream RL training while providing the necessary signal for practical layer selection.

### B Extended Experiment Results

#### B.1 Out-of-Domain Generalization

To evaluate the generalization ability of BuPO, we extend our evaluation to broader benchmarks spanning advanced math (OlympiadBench (He et al., 2024), Minerva (Lewkowycz et al., 2022)), general reasoning (MMLU-Pro (Wang et al., 2024), GPQA-D (Rein et al., 2024), ARC-C (Clark et al., 2018), BBH (Suzgun et al., 2023)), and coding (HumanEval (Chen et al., 2021)). As shown in Table 9, BuPO consistently outperforms both the vanilla baseline and GRPO across all benchmarks on both model scales. BuPO achieves average gains of +1.83 and +0.62 points over GRPO on Qwen3-4B and Qwen3-8B, respectively, with particularly notable improvements on GPQA-D (+4.10 for 4B, +1.50 for 8B) and HumanEval (+3.60 for 4B, +1.83 for 8B). These consistent gains across diverse tasks suggest that optimizing internal layer policies encourages the model to develop more transferable reasoning capabilities rather than overfitting to training task patterns.

#### B.2 Comparison with Conventional Layer-aware Optimization

To examine whether BuPO specifically benefits from internal policy optimization, rather than merely from emphasizing lower layers, we compare it with two conventional layer-aware GRPO variants. The first baseline freezes layers up to the selected layer l, while the second applies a smaller learning rate (0.1×) to layers above l. As shown in Table 11, BuPO consistently outperforms both alternatives on Qwen3-4B and OctoThinker-3B. These results show that simply freezing layers or reweighting learning rates is insufficient to match BuPO. In contrast, optimizing the internal policy through IPA provides a more effective training signal, confirming the effectiveness of BuPO.

#### B.3 Multi-seed Robustness

To address whether the reported improvements depend on a single random seed, we rerun both GRPO and BuPO with three random seeds under the same training setup. Table 10 reports the mean and standard deviation across seeds. BuPO outperforms GRPO on most benchmarks. Averaged over the four benchmarks, BuPO improves over GRPO by +3.77, +2.53, +1.95, and +3.62 points on Qwen3-4B, Qwen3-8B, OctoThinker-3B, and OctoThinker-8B, respectively.

Methods Olympiad Minerva MMLU-Pro GPQA-D ARC-C BBH HumanEval Avg Qwen3-4B

Vanilla 32.27 23.99 60.50 45.50 87.40 75.50 78.70 57.69 GRPO 36.15 28.77 61.30 43.90 87.10 75.00 79.30 58.79 BuPO (Ours) 37.52 29.89 61.70 48.00 87.50 76.80 82.90 60.62 ∆ vs. GRPO +1.37 +1.12 +0.40 +4.10 +0.40 +1.80 +3.60 +1.83

Qwen3-8B

Vanilla 32.51 22.45 63.40 51.00 90.80 79.60 83.54 60.47 GRPO 40.31 31.55 66.60 52.00 90.87 81.55 84.15 63.86 BuPO (Ours) 40.63 31.92 66.80 53.50 91.00 81.50 85.98 64.48 ∆ vs. GRPO +0.32 +0.37 +0.20 +1.50 +0.13 -0.05 +1.83 +0.62

- Table 9: Out-of-domain evaluation results across general reasoning and coding benchmarks. Bold denotes the best.

Methods AMC MATH AIME24 AIME25 Qwen3-4B

BuPO 80.78±0.41 85.19±0.26 36.77±0.11 30.99±0.63 GRPO 76.67±0.36 83.12±0.62 32.15±0.47 26.70±1.93

Qwen3-8B

BuPO 89.06±0.72 88.00±0.22 53.85±0.18 34.90±0.48 GRPO 86.31±0.39 88.02±0.06 48.19±1.36 33.19±0.30

OctoThinker-3B

BuPO 27.71±0.24 49.88±0.17 0.66±0.06 3.86±0.08 GRPO 27.29±0.18 46.42±0.31 0.45±0.16 0.17±0.06

OctoThinker-8B

BuPO 38.08±0.59 62.02±0.05 4.76±0.12 6.67±0.11 GRPO 35.16±0.41 57.12±0.21 2.64±0.24 2.15±0.06

- Table 10: Multi-seed results of BuPO and GRPO. We report mean ± standard deviation over three random seeds.

Methods AMC MATH AIME24 AIME25 Avg. Qwen3-4B

BuPO 81.09 84.90 36.88 31.15 58.51 Frozen layers 78.91 83.60 31.04 29.06 55.65 Layerwise LR 80.86 84.00 36.88 30.73 58.12

OctoThinker-3B

BuPO 27.50 49.79 0.63 0.42 19.59 Frozen layers 23.67 46.40 0.31 0.42 17.70 Layerwise LR 26.17 44.80 0.63 0.21 17.95

- Table 11: Comparison with conventional layer-aware GRPO alternatives.

Base, Instruct, and Mix versions, as well as models trained with supervised fine-tuning (SFT) and reinforcement learning (RL). In addition, we include the DeepSeek-Math model (He et al., 2025) to further enrich the comparative analysis.

Further Training Has Limited Impact on Internal Reasoning Patterns. We further analyze models that undergo additional training beyond standard pre-training. Specifically, we include DeepSeek-R1-Distill-Qwen-7B, which is further trained from Qwen2.5-Math-7B using distilled responses from Guo et al. (2025) with SFT, as well as Llama-OctoThinker-3B-Base and Llama-OctoThinker-8B-Base, which are obtained via continued pre-training (i.e., midtraining) based on Llama-3.2-3B-Base and Llama-3.2-8B-Base, respectively (Wang et al., 2025b). Moreover, Qwen3-4B and Qwen3-4B-Base also show consistent pattern after post-training with RL. Notably, these additional training procedures exhibit only marginal influence on the internal reasoning patterns.

In summary, reinforcement learning, supervised fine-tuning, and mid-training do not substantially alter the model’s internal reasoning mechanisms, suggesting that these intrinsic patterns are primarily determined by the model architecture and initial pre-training. Understanding how such patterns emerge remains an important direction for future work.

The standard deviations are generally small relative to the performance gaps, indicating that the improvements are stable across random seeds. These results show that the gains of BuPO are not an artifact of a single run and further support the reliability of the main results.

Same Series of Models Exhibit Consistent Structures. After analyzing in all analysis plots across models, we find that models from the same series show consistent structures. For instance, all Qwen3 series models show progressive internal reasoning patterns including Qwen3-4B, Qwen3-8B and Qwen3-14B, also with other base or instruct version. Also, Llama-3.1 and Llama-3.2 show intra-

- B.4 Internal Policy Entropy Dynamics for More Models

In this section, we present additional preliminary analyses of internal policy entropy dynamics across a broader set of models. Specifically, we examine different variants of the same backbone, including

Qwen2.5-Math-7B

1.00

| |[Figure 5]| | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.75

CosineSimilarity

0.50

0.25

0.00

0.25

0.50

0.75

1.00

1 6 11 16 21 26

Qwen3-4B

1.00

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

0.75

CosineSimilarity

0.50

0.25

0.00

0.25

0.50

0.75

1.00

1 6 11 16 21 26 31 36

Qwen3-4B-Base

1.00

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

0.75

CosineSimilarity

0.50

0.25

0.00

0.25

0.50

0.75

1.00

1 6 11 16 21 26 31 36

Qwen3-8B

1.00

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

0.75

CosineSimilarity

0.50

0.25

0.00

0.25

0.50

0.75

1.00

1 6 11 16 21 26 31 36

- Figure 12: Residual cosine similarity across different Qwen models.

difference and inter-consistency.

Entropy Dynamics of DeepSeek-Math. We further analyze the internal reasoning mechanisms of DeepSeek-Math-7B-Base to provide a more comprehensive comparison. As illustrated in Figure 13, DeepSeek-Math-7B-Base exhibits a markedly different entropy dynamics: the overall internal policy entropy decreases substantially and converges primarily in the middle layers. This phenomenon is primarily driven by the consistently negative entropy change in the FFN module, i.e., ∆HFFNl < 0, as illustrated in Figure 14, particularly in the middle layers.

Based on this observation, we infer that both Qwen and DeepSeek-Math demonstrate strong capability in knowledge absorption during posttraining, indicating that convergence behavior in internal reasoning plays a critical role in effective learning, in contrast to Llama. Moreover, the generation search space of DeepSeek-Math appears more constrained than that of Qwen, particularly the Qwen3 series, suggesting reduced exploration capacity. We hypothesize that such internal reasoning patterns significantly influence the effectiveness of further training, pointing to promising directions for architectural design and optimization of foundation models.

- B.5 How do Internal modules influence the residual stream?

To further understand how internal modules shape the residual stream in Qwen models with a progressive reasoning pattern, we analyze residual cosine similarity, which quantifies how each module writes to the residual pathway (Hu et al., 2025b).

For a given layer l, we compute cossim(Al,Hl−1) for self-attention and cossim(Fl,Hl−1 + Al) for the FFN. A cosine similarity near zero indicates writing new, orthogonal features; negative values indicate feature suppression; and positive values indicate amplification of existing features.

As shown in Figure 12, the Qwen models largely follow the entropy dynamics discussed earlier, while exhibiting clear inter-generation differences. For Qwen3, self-attention consistently amplifies the residual stream, in line with its positive entropy change and expanded exploration behavior. In contrast, Qwen2.5 shows noticeably weaker attention write-in strength, with reduced cosine similarity magnitudes, consistent with its negative entropy change in self-attention.

The FFN modulates the residual stream in a stage-dependent manner across all models: in lower layers, it injects largely orthogonal features to support exploration; in middle layers, it suppresses vague signals while integrating parametric knowledge in FFN, corresponding to the Integration stage; and in upper layers, it amplifies and integrates features to drive convergence. Across all models, the final layer exhibits a sharp directional shift, underscoring its critical role in final prediction (Gupta et al., 2025; Agarwal et al., 2025).

###### Qwen3-4B-Base

Qwen3-4B-Instruct-2507

DeepSeek-R1-Distill-Qwen-7B

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

12.5

12.5

12.5

| | |
|---|---|
| | |
| | |
| | |
|Layer I/O ATTN FFN<br><br>| |
|---|
<br><br>| |
|---|
| |

| | |
|---|---|
| | |
| | |
| | |
|Layer I/O ATTN FFN<br><br>| |
|---|
| |

10.0

10.0

10.0

7.5

7.5

7.5

Entropy

Entropy

Entropy

5.0

5.0

5.0

2.5

2.5

2.5

Layer I/O ATTN FFN

| |
|---|

0.0

0.0

0.0

| |
|---|

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28

Layer Index

Layer Index

Layer Index

###### Llama-OctoThinker-3B-Base

###### Llama-OctoThinker-8B-Base

DeepSeek-Math-7B-Base

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

Continuous Entropy Flow Through Layers

12

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |
|Layer I/O ATTN<br><br>| |
|---|
| |

10.0

- 8

- 9

- 10

- 11

10

7.5

Entropy

Entropy

Entropy

8

5.0

2.5

6

Layer I/O ATTN FFN

Layer I/O ATTN FFN

| |
|---|

| |
|---|

FFN

| |
|---|

| |
|---|

0.0

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32

1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30

Layer Index

Layer Index

Layer Index

- Figure 13: Continuous entropy dynamics of internal policy for additional models. The information flows from Hl−1 into Al, Fl, then to the next layer Hl.

Qwen3-4B-Base

Qwen3-4B-Instruct-2507

Qwen3-14B

DeepSeek-R1-Distill-Qwen-7B

10

10

10

10

InternalATTNPolicyInternalFFNPolicy

MeanEntropyChange

5

5

5

5

0

0

0

0

5

5

5

5

10

10

10

10

1 6 11 16 21 26 31 36

1 6 11 16 21 26 31 36

1 6 11 16 21 26 31 36 40

1 6 11 16 21 26 28

4

4

4

4

2

2

2

2

MeanEntropyChange

0

0

0

0

2

2

2

2

4

4

4

4

6

6

6

6

8

8

8

8

10

10

10

10

12

12

12

12

1 6 11 16 21 26 31 36

1 6 11 16 21 26 31 36

1 6 11 16 21 26 31 36 40

1 6 11 16 21 26 28

Layer Index

Layer Index

Layer Index

Layer Index

Llama-OctoThinker-3B-Base

###### Llama-OctoThinker-8B-Base

###### Llama-3.1-8B-Instruct

###### DeepSeek-Math-7B-Base

InternalATTNPolicyInternalFFNPolicy

0.4

0.4

0.4

- 0

- 1

- 2

- 3

- 4

MeanEntropyChange

0.2

0.2

0.2

0.0

0.0

0.0

0.2

0.2

0.2

1

0.4

0.4

0.4

1 6 11 16 21 26

1 6 11 16 21 26 31

1 6 11 16 21 26 31

1 6 11 16 21 26

0.4

- 0
- 1

- 0
- 1

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0

0.2

MeanEntropyChange

0.0

2

1

1

0.2

2

2

4

0.4

3

3

0.6

6

4

4

0.8

8

1.0

5

5

1 6 11 16 21 26

1 6 11 16 21 26 31

1 6 11 16 21 26 31

1 6 11 16 21 26

Layer Index

Layer Index

Layer Index

Layer Index

Figure 14: Entropy change dynamics of internal policy with more models.

#### B.6 Pass@K Performance across Datasets

We further provide a detailed version of Figure 6 across AMC23, MATH500, AIME24, and AIME25. Our proposed BuPO consistently outperforms the vanilla GRPO baseline.

### C Use of AI Assistants

AI assistants were used only for minor language polishing and grammar refinement. All research ideas, technical contributions, experiments, analyses, and final claims were developed and verified by the authors.

AIME25

###### AIME24

###### AMC23

###### MATH500

###### AVG

94

85

70

95

70

80

92

60

75

Qwen3-4B

90

90

60

70

50

88

50

85

65

40

86

40

60

80

GRPO

30

84

BuPO(Ours)

55

30

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

100

90

80

90

94

98

85

70

80

95

Qwen3-8B

92

80

60

92

70

75

50

90

90

60

70

88

40

88

65

50

85

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Llama-OctoThinker-3B-Base

90

25

30

80

50

25

80

70

20

20

60

15

70

40

15

50

10

60

10

30

40

5

5

50

30

20

0

0

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Llama-OctoThinker-8B-Base

90

90

40

60

30

80

80

30

70

50

20

60

20

40

70

50

10

10

30

40

60

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

1 4 16 64 256

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Number of Samples K

Figure 15: Pass@K results on MATH500, AMC23, AIME24 and AIME25. To reduce evaluation variance, we set n = 300.

