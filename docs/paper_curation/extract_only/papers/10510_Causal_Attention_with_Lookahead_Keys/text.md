# arXiv:2509.07301v2[cs.CL]29Sep2025

[Figure 1]

## Causal Attention with Lookahead Keys

Zhuoqing Song1,2,∗, Peng Sun1, Huizhuo Yuan1, Quanquan Gu1,† 1ByteDance Seed, 2Princeton University ∗Work done during internship at ByteDance Seed, †Corresponding author

##### Abstract

In standard causal attention, each token’s query, key, and value (QKV) are static and encode only preceding context. We introduce CAuSal aTtention with Lookahead kEys (CASTLE), an attention mechanism that continually updates each token’s keys as the context unfolds. We term these updated keys lookahead keys because they belong to earlier positions yet integrate information from tokens that appear later relative to those positions, while strictly preserving the autoregressive property. Although the mechanism appears sequential, we derive a mathematical equivalence that avoids explicitly materializing lookahead keys at each position and enables efficient parallel training. On language modeling benchmarks, CASTLE consistently outperforms standard causal attention across model scales, reducing validation perplexity and improving performance on a range of downstream tasks.

Date: September 24, 2025 Correspondence: Quanquan Gu at quanquan.gu@bytedance.com

##### 1 Introduction

Causal attention [27] is a cornerstone of autoregressive sequence modeling, allowing each token to condition on its past while preserving the autoregressive structure that underpins language generation. Building on this mechanism, large language models (LLMs) have transformed natural language processing by scaling up the model size and the number of tokens trained [3, 12, 20]. While this trend has delivered impressive capabilities, it increasingly runs up against a practical bottleneck, that is high-quality tokens. This reality makes it imperative to improve the attention layer itself and to develop model architectures that are more token-efficient, delivering stronger performance under fixed training-token budgets.

In standard causal attention, each token’s query, key, and value (QKV) are computed from that token’s representation and remain fixed; they cannot incorporate information from subsequent tokens. As a result, a token’s QKV can encode only its preceding context. Recent work shows that such causal mask, which blocks each token’s access to its future information, limits models’ ability to capture global context, and impairs natural language understanding [8, 15, 23, 28, 33]. Here are some vivid illustrations that give intuitions for the limitations of causal masking. Garden-path sentences [19] are structurally ambiguous, often inducing an incorrect initial parse. For example, “the old man the boat”. Because garden-path sentences’ correct interpretation typically depends on information that appears later in the sentence, the causal mask restricting tokens to past context can cause models to struggle in resolving such ambiguities effectively [1]. In many tasks, the question/focus appears at the end of the input. Without access to future context, earlier tokens cannot encode the relevant information needed to anticipate the question/focus. As a result, early token

representations may fail to capture important cues and global context dependencies [28].

To address the shortcomings of standard causal attention in pretraining, we propose a novel attention mechanism, CAuSal aTtention with Lookahead kEys (CASTLE). In this approach, when generating the (t + 1)-th token, we update keys of all preceding tokens so that keys of token s (1 ≤ s ≤ t) are able to additionally encode information from token s + 1 to t. These keys are called lookahead keys. This design preserves the autoregressive structure while allowing the keys to evolve as the context unfolds. In Figure 1, we give an illustration of receptive fields of the keys in CASTLE. Although the mechanism appears recurrent, we establish a mathematical equivalence that avoids explicitly materializing the lookahead keys and enables efficient parallel training. We evaluate our approach on language modeling across multiple model scales. Experimental results show that CASTLE consistently outperforms standard causal attention in terms of validation perplexity and downstream task performance. We also introduce a variant, CASTLE-SWL, which applies sliding windows to lookahead keys. CASTLE-SWL has the same complexities with CASTLE in training and inference and preserves the performance gains of CASTLE while further improves efficiency in practice.

Generating the 6th token

Generating

the 4th token

T1 T2 T3 T4 T5

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

T2 T3

- key1
- key2
- key3
- key4
- key5

| | | |
|---|---|---|
| | | |
| | | |

- key1

T1

- key2
- key3

Standard Causal Attention

T1 T2 T3 T4 T5

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- key1
- key2
- key3
- key4
- key5

T1 T2 T3

| | | |
|---|---|---|
| | | |
| | | |

- key1
- key2
- key3

Causal Attention with

Lookahead Keys

- Figure 1 Receptive fields of keys in standard causal attention and CASTLE. The top row shows standard causal attention when generating the 4th token (left) and the 6th token (right). The bottom row shows CASTLE under the same settings. Here, key1, key2, · · · denote the keys corresponding to tokens 1, 2, · · · , while T1, T2, · · · denote the tokens. In standard causal attention, keys are static: when generating the (t + 1)-th token, each key i with 1 ≤ i ≤ t can only access information from {T1, · · · , Ti}, and key i remains the same for all later steps. In contrast, CASTLE continuously updates keys at each prediction step, i.e., when generating the (t + 1)-th token, the receptive field of any key i with 1 ≤ i ≤ t, is expanded to contain information from {T1, · · · , Tt}.

The remainder of this paper is organized as follows. We discuss related work in Section 1.1. In Section 2.1, we elaborate on our motivations. In Section 2.2, we formally define CASTLE in its recurrent form. Section 2.3 proves an equivalent parallel formulation of CASTLE and develops efficient parallel training algorithms. Section 2.4 introduces efficient inference algorithms together with the counterpart of the KV cache in CASTLE. Finally, Section 3 presents empirical results demonstrating the effectiveness of CASTLE across diverse tasks and model scales.

###### 1.1 Related Work

Several studies have observed that the causal mask, by preventing tokens from accessing future information, can hinder a model’s ability to capture global context and degrade its natural language understanding [8, 15, 23, 28, 33].

Much effort has been made to overcome this shortcoming in sentence embedding. BeLLM (Backward Dependency Enhanced LLM) [15] modifies decoder layers to enable sentence embeddings to integrate both past and future information. This yields substantial improvements on semantic textual similarity and downstream tasks. Echo Embeddings [23] duplicate the input sequence and extract embeddings from the second occurrence, letting early tokens attend to later ones without modifying the architecture. Similarly, Re-Reading (RE2) [28] prompts models to process inputs twice, so the second pass captures global information obtained in the first. These methods improve embedding quality and reasoning, but their benefits in large-scale pretraining remain unclear.

Selective Attention [14] introduces a parameter-free modification where tokens can mask out irrelevant or outdated tokens from future attention. By subtracting accumulated selection scores from the attention logits, selective attention reduces reliance on unneeded context. As a result, it achieves substantial memory and compute savings without degrading perplexity. However, selective attention primarily emphasizes filtering unneeded past tokens to enhance efficiency. As discussed in the introduction, many scenarios, such as garden-path sentences or cases where the key information appears at the end of the input, require mechanisms that actively incorporate crucial future information. Whether selective attention can address this challenge remains uncertain.

PaTH Attention [30] is a novel positional encoding scheme which introduces data-dependent encodings based on cumulative products of Householder-like transformations. Each transformation is conditioned on the input, enabling the model to dynamically adapt positional information as the sequence progresses. PaTH is related to CASTLE in the sense that in inference formulation of PaTH, keys are also updated via a rank-1 update. However, both the update mechanism of keys and the parallel training formulation in PaTH differ substantially from those in CASTLE, and PaTH remains fundamentally a positional encoding, making it orthogonal to our approach.

Encoder-only Next Token Prediction (ENTP) [9] performs next-token prediction with encoder-only Transformers, where the keys are naturally re-computed at each position. It demonstrates stronger sample efficiency on small-scale language modeling and in-context learning benchmarks. However, the per-token compute in ENTP scales quadratically with sequence length and cubically over full sequences, which presents challenges for scaling.

- 2 Causal Attention with Lookahead Keys

Our motivations are discussed in Section 2.1. Formal mathematical definitions of CASTLE in recurrent form are provided in Section 2.2. Direct application of the recurrent form of CASTLE in Section 2.2 is impractical for large-scale pretraining. To address this, we present efficient pretraining algorithms in Section 2.3. In Section 2.4, we describe efficient inference algorithms along with the counterparts of the KV cache in CASTLE.

###### 2.1 Motivations

We first recall the standard causal attention. Given contextualized representations XL ∈ RL×d

model where L is the sequence length and dmodel is the hidden dimension. The standard causal attention first computes Q = XLW Q, K = XLW K, V = XLW V ∈ RL×d, where d is the head dimension. Then, the standard causal attention is computed as follows

CausalAttention(XL) = row_softmax

###### QK⊤

+ M C V ∈ RL×d, (1)

√

d

where M C ∈ RL×L is the causal mask which prevents each token from attending to its future tokens, i.e., M Cij = 0 if i ≥ j and M Cij = −∞ otherwise.

To explain our motivations, we begin with the recurrent form of standard causal attention. Consider the





x1 x2

case when generating the (t + 1)-th token. Given contextualized representations Xt =

model,

∈ Rt×d

 

 

. xt

where the s-th row xs is the representation of token s. Unless otherwise specified, all vectors in this paper are treated as row vectors rather than column vectors.

The query, key and value of token s are qs = xsW Q, ks = xsW K, vs = xsW V . We also denote Kt = XtW K and V t = XtW V . Then, the standard causal attention follows the recurrent form

causal-attention(Xt) = softmax

###### qtKt⊤

√

d

V t =

t

- s=1 exp qtks⊤/

√

d vs

- t s=1 exp qtks⊤/

√

d ∈ R1×d. (2)

Due to the autoregressive structure, each xs only encodes information from token 1 to s. Thus, when generating token t + 1 with t + 1 > s, each ks only contains information from token 1 to s without containing information from token s + 1 to t. This can impair models’ ability of natural language understanding, yielding high-quality text embedding and capturing global context as mentioned in the introduction.

This motivates us to propose a novel attention mechanism, causal attention with lookahead keys (CASTLE), i.e., when generating the (t + 1)-th token, we first update keys ks of preceding tokens s with s < t + 1 to additionally incorporate information from token s + 1 to t. We refer to these as lookahead keys because their representations renew with the growing context. In this way, lookahead keys may lead to more accurate attention scores while preserving the autoregressive property.

Before describing the details of this mechanism, we first answer the following questions.

- • Why do we use lookahead keys instead of lookahead queries? The answer parallels the reason why

key–value (KV) pairs are cached instead of queries (Q). Each qs is only used once, namely when generating token s+1. Because we are designing an autoregressive model, past queries cannot be modified after generation,

making it meaningless to update qs. In contrast, ks is multiplied by the queries qt of all subsequent tokens t ≥ s. Keeping ks updated therefore can benefit all later tokens by possibly producing more accurate attention scores.

We also remark that updating the values vs similarly to lookahead keys could be beneficial while its efficient algorithm remains future study.

- • How do we maintain the model autoregressive with lookahead keys? When generating token t + 1, we update keys of each preceding token s < t + 1 with information from token s + 1 to t. Thus, all keys only contain information from token 1 to t. Queries and values are naturally only containing information from tokens up to t. No future information from tokens k > t is used. Thus, the model maintains autoregressive property. Further details of our design are presented in Section 2.2.

###### 2.2 Mathematical Definition in Recurrent Form

We give mathematical definitions of CASTLE in this section. Let L denote sequence length and dmodel, d denote the hidden dimension and head dimension, respectively.

Throughout Section 2.2, we fix t ∈ {1,2,··· ,L} and consider the setting where t tokens have been generated and the model is generating the (t + 1)-th token. Denote the input contextualized representations Xt =



x1 x2

model, where xs is the representation of token s.

∈ Rt×d

 

 

. xt

Utilizing lookahead keys lies at the core of CASTLE. However, the way a model learns to encode information into keys of token s from past tokens (k ≤ s) may differ from how it encodes information from subsequent tokens (tokens s < k < t + 1) when generating the (t + 1)-th token. To address this, we adopt a hybrid design. Specifically, we keep half of the keys the same as in standard causal which we call causal keys, while allowing the remaining half to renew as the context progresses which we call lookahead keys.

For each preceding token s (1 ≤ s < t + 1), causal keys of token s is a projection of xs, while lookahead keys of token s contain information from representations {xs+1,··· ,xt}. The receptive fields of causal keys and lookahead keys are illustrated in Figure 2.

x1 x2 x3 x4 x5

T1 T2 T3 T4 T5

| | | | | |
|---|---|---|---|---|
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

- causal key 1
- causal key 2
- causal key 3
- causal key 4
- causal key 5

- causal key 1
- causal key 2
- causal key 3
- causal key 4
- causal key 5

x1 x2 x3 x4 x5

T1 T2 T3 T4 T5

| | | | | |
|---|---|---|---|---|
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

- lookahead key 1
- lookahead key 2
- lookahead key 3
- lookahead key 4
- lookahead key 5

- lookahead key 1
- lookahead key 2
- lookahead key 3
- lookahead key 4
- lookahead key 5

- Figure 2 Receptive fields of causal keys and lookahead keys with respect to contextualized representations and tokens (excluding the first layer) when generating the 6th token. Tokens are denoted by Ti and their contextualized representations by xi. Causal key i corresponds to the causal key of token i, while lookahead key i corresponds to the lookahead key of token i. When generating the (t + 1)-th token, for token s (s < t + 1), the causal key of token s is a projection of xs. Due to the softmax in attention, except in the first layer, causal keys of s attend over tokens {T1, · · · , Ts}. For token s < t, lookahead keys of s incorporate information from {xs+1, · · · , xt} and attend over all

existing tokens {T1, · · · , Tt}. Since the last row of M U is defined as [M Ut ]t,: = (−∞)1×t, lookahead keys of token t are all-zeros vectors and thus have empty receptive fields when generating the (t + 1)-th token.

We first project Xt into key and value matrices KUt , V Ut , KCt , V Ct ∈ Rt×d by

KUt = XtW UK ∈ Rt×d

###### , V Ut = XtW UV ∈ Rt×d

, and

model

model

KCt = XtW CK ∈ Rt×d

###### , V Ct = XtW CV ∈ Rt×d

model

model

as well as query matrix QUt = XtW UQ ∈ Rt×d and query vector qCt = xtW CQ ∈ R1×d. Here, W UQ, W UK, W UV , W CQ, W CK, W CV ∈ Rd

model×d are learnable matrices.

The matrices KUt , V Ut , QUt are used to generate the lookahead key U t. Then, the causal key KCt and the lookahead key U t are multiplied by the query vector qCt to get the attention scores. Then, V Ct are multiplied

by the attention weights to get the output. Before we elaborate on details in the definition of CASTLE, we first give formal definitions of causal keys and lookahead keys.

Causal Keys. The causal keys in CASTLE are defined similarly to the keys in standard causal attention. More specifically, causal keys are defined as





kC1 kC2 . kCt

= XtW CK ∈ Rt×d.

KCt =

 

 

The s-th row ks of KCt satisfying ks = xsW CK is the causal key of token s. Causal keys are static, i.e., the s-th rows of KCt and KCt′ are equal to each other whenever t,t′ ≥ s. Lookahead Keys. We utilize a structure similar to the attention mechanism to define lookahead keys. An illustration for lookahead keys can be found in Figure 3. More specifically, the lookahead keys are defined as





- ut1
- ut2

QUt KUt ⊤

U t =

+ M Ut V Ut ∈ Rt×d, (3)

√

= sigmoid

 

 

. utt

d

where M Ut ∈ Rt×t is a mask matrix with

0, if i < j −∞, otherwise

[M Ut ]ij =

(4)

The s-th row uts of U t is the lookahead key of token s. We remark that in uts, the superscript t indicates that uts is defined when t tokens have already been generated and we are about to generate the (t + 1)-th token, while the subscript s indicates uts is the s-th row of U t.

The definition of M Ut guarantees that the lookahead key of token s, uts, is exposed to information from {xs+1,··· ,xt}. Since uts keeps renewing as the context goes, it is natural that uts ̸= uts+1. We have the following remarks regarding the definition of lookahead keys in (3).

- • Why are we using sigmoid? The sigmoid function is used in (3) instead of softmax due to the consideration that when generating token t + 1, for token s with s < t + 1, synthesizing information contained in tokens s + 1 to t should not be compulsory. However, since the probabilities in softmax sum up to 1, which forces uts to incorporate information from tokens s + 1 to t and is not desired.
- • Lookahead keys Ut maintains autoregressive property. First, CASTLE is defined in a recurrent form

which is naturally autoregressive. Second, when generating the (t + 1)-th token, each uts is only exposed to information from representations of tokens s + 1 to t as in (3). No information from tokens which are not yet generated is exposed.

- • Lookahead keys Ut only occur in CASTLE’s recurrent form definition and inference, but cannot be

materialized in parallel training. Since uts and uts+1 may vary, this prevents us from materializing U t for each t. The computation cost in (3) is O(t2d). If we materialize all U t in parallel, the computational cost

is at least Lt=1 t2d = O(L3d) which makes training on large-scale datasets impractical. In Section 2.3, we will give an equivalent form which removes the need of materializing each U t and enables efficient

parallel training.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

| |[Figure 6]|
|---|---|
| |[Figure 7]|

[Figure 8]

[Figure 9]

[Figure 10]

| | | |
|---|---|---|
| | | |
| | | |

0 0 0

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

| | |
|---|---|
| | |
| | |

[Figure 24]

[Figure 25]

[Figure 26]

| | |
|---|---|
| | |
| | |

- Figure 3 Illustration for the definition of lookahead keys in (3) when generating the 4-th token. Let dmodel and d denote the hidden dimension and head dimension, respectively. In this figure, we set t = 3 and d = 2.

CASTLE in Recurrent Form. After defining causal keys and lookahead keys, we are ready to give the formula of CASTLE in recurrent form. To generate the (t + 1)-th token, we utilize both the causal keys KCt ∈ Rt×d and the lookahead keys U t.

More specifically, let the causal-key attention scores be

qCt KCt ⊤

d ∈ R1×t. (5) Let the lookahead-key attention scores be

sCt =

√

qCt U t⊤

d ∈ R1×t. (6) Then, we define attention weights by combining the above attention scores as follows

sUt =

√

pt = softmax sCt − SiLU sUt ∈ R1×t, (7) where SiLU(x) = x ⊙ sigmoid(x). Then, the output is calculated as

attention Xt = ptV Ct ∈ R1×d. (8)

We remark that SiLU is applied in (7) because our ablation study indicates that it plays a crucial role in ensuring training stability. We hypothesize that this benefit arises because many past tokens become ‘noise’ as the context grows, and the SiLU transformation effectively acts as a gate, regulating the degree to which past tokens should be down-weighted.

An illustration of CASTLE in its recurrent form can be found in Figure 4.

[Figure 27]

[Figure 28]

Lookahead keys

[Figure 29]

[Figure 30]

Lookahead-key attention scores

[Figure 31]

Attention weights

[Figure 32]

[Figure 33]

[Figure 34]

| | |
|---|---|

[Figure 35]

[Figure 36]

Causal-key attention scores

Causal keys

[Figure 37]

- Figure 4 Illustration of CASTLE in recurrent form when generating the 4th token. The causal and lookahead keys

are queried by qCt to generate their respective attention scores, which are combined and then goes through softmax to yield attention weights pt. These weights are then multiplied by the value matrix V Ct to compute the output ot = attention Xt ∈ R1×d

CASTLE-SWL in Recurrent Form. We propose a variant of CASTLE, termed CASTLE-SWL, where we apply Sliding Windows to Lookahead keys. More specifically, denote sliding window size by W. When generating the (t + 1)-th token, lookahead keys of token s (s < t + 1) have access to information from {xk : s + 1 ≤ k ≤ min{s + W,t}}. Formally, the definition of lookahead keys in CASTLE-SWL follows (3) with M Ut defined as

0, if i < j ≤ min{t,i + W} −∞, otherwise

[M Ut ]ij =

(9)

An illustration for M Ut applied to lookahead keys in CASTLE-SWL can be found in Figure 5.

Sliding window modifies only lookahead keys while causal keys remains unchanged. Apart from changing the definition of M Ut from (4) to (9), all other components of CASTLE remain the same, yielding the definition of CASTLE-SWL.

The motivation for this design is that the semantic contribution of tokens may decay as the context unfolds. Allowing lookahead keys to aggregate information from distant tokens may introduce noise rather than useful signal.

In addition, although CASTLE-SWL has the same complexities with CASTLE in both training and inference as analyzed in Section 2.3 and 2.4, it can further reduce FLOPs and improves efficiency in practice through the use of sliding windows.

[Figure 38]

[Figure 39]

|[Figure 40]|0|0|0|[Figure 41]|[Figure 42]|
|---|---|---|---|---|---|
|[Figure 43]|[Figure 44]|0|0|0|[Figure 45]|
|[Figure 46]|[Figure 47]|[Figure 48]|0|0|0|
|[Figure 49]|[Figure 50]|[Figure 51]|[Figure 52]|0|0|
|[Figure 53]|[Figure 54]|[Figure 55]|[Figure 56]|[Figure 57]|0|
|[Figure 58]|[Figure 59]|[Figure 60]|[Figure 61]|[Figure 62]|[Figure 63]|

- Figure 5 An illustration of M Ut in the lookahead keys of CASTLE-SWL, as defined in (9), when generating the (t+1)-th token. Each lookahead key of token s (s < t) has access to representations {xk : s + 1 ≤ k ≤ min {t, , s + W}}, i.e., up to W subsequent tokens. Also, no information from tokens which are not yet generated is leaked and thereby preserving the autoregressive property. In this figure, we set t = 6 and window size W = 3. For example, lookahead keys of token 2 can “see” {x3, x4, x5}, while lookahead keys of token 4 can “see” {x5, x6}. As analyzed in Section 2.3 and 2.4, CASTLE-SWL has the same complexities with CASTLE in both training and inference, however, it can further reduce FLOPs and improves efficiency in practice through the use of sliding windows.

###### 2.3 Efficient Parallel Training

In this section, we introduce our efficient parallel training algorithms. As discussed in Section 2.2, a straightforward materializing each U t in parallel incurs at least O(L3d) computational costs. Such complexity makes training on large-scale datasets infeasible. To address this, we first derive an equivalent parallel formulation of CASTLE that avoids materializing lookahead keys, and then exploit the special structure of a matrix, which is expressible as a low-rank matrix multiplied by a mask, to reduce CASTLE’s training complexity to O(L2d). CASTLE-SWL naturally shares the same training complexity.

CASTLE in Parallel Form. Let attention Xt ∈ R1×d denote the output when generating the (t + 1)-th token as in (8). Then, given the inputs XL ∈ RL×d

model, the concatenated outputs are denoted by





attention(X1) attention(X2)

Attention(XL) =

∈ RL×d

. (10)

 

 

model

. attention(XL)

The following theorem provides a unified parallel formulation for both CASTLE and CASTLE-SWL that is equivalent to their recurrent forms introduced in Section 2.2. This formulation then serves as the basis for designing an efficient algorithm. Its proof is in Appendix B.

- Theorem 1. Consider inputs XL ∈ RL×d

model, where L is the sequence length and dmodel is the hidden dimension. Let QU = XLW UQ, KU = XLW UK, V U = XLW UV , QC = XLW CQ, KC = XLW CK,

- V C = XLW CV . Define matrix SU ∈ RL×L as

⊤

###### QUKU⊤

QCV U⊤

d ⊙ M C sigmoid

. (11)

+ M U

SU =

√

√

d

Then, the outputs Attention(XL) as in (10) satisfies that

Attention(XL) = row_softmax

QCKC⊤

√

d

+ M C − SiLU SU V C. (12)

Here, M C, M C are the causal masks which prevent tokens from attending to their future tokens, i.e., M Cij = 0 if i ≥ j and M Cij = −∞ otherwise; M Cij = 1 if i ≥ j and M Cij = 0 otherwise. For CASTLE, M U = M UL ∈ RL×L with M UL defined in (4), while for CASTLE-SWL, M U = M UL ∈ RL×L with M UL

defined in (9).

Efficient Parallel Training. Theorem 1 establishes the equivalence between the recurrent and parallel formulations for CASTLE and CASTLE-SWL. However, computing Attention(XL) directly from Theorem 1 still requires Ω(L3) complexity, since (11) involves matrix multiplications between L-by-L matrices.

To reduce this cost, notice that in (11), the term QCV U⊤ ⊙ M C is a masked low-rank matrix because the matrix QCV U⊤ is of rank d which is typically much smaller than L. This structure enables a more efficient computation of SU, which we exploit to design a parallel training algorithm as stated in Theorem 2. The proof of Theorem 2 is given in Appendix D.

- Theorem 2. Given XL’s query, key and value matrices QU = XLW UQ, KU = XLW UK, V U = XLW UV ,

QC = XLW CQ, KC = XLW CK, V C = XLW CV , for both CASTLE and CASTLE-SWL, Algorithm 1 (forward pass) and Algorithm 3 (backward pass) enable efficient parallel training and can compute Attention(XL) and the gradients with computational complexity O(L2d) and space complexity O(Ld).

###### 2.4 Efficient Inference with UQ-KV Cache

In this section, we introduce the inference algorithm which has unified forms for CASTLE and CASTLE-SWL. We first introduce the decoding algorithm. The decoding algorithm consists of the following 2 steps: updating step and combining step. Fix t ∈ {1,2,...,L}, and consider generating the (t + 1)-th token.

Updating step. We generate lookahead keys U t in the updating step. First, we compute qUt = xtW UQ, kUt = xtW UK and vUt = xtW UV . Next, rather than computing U t directly from (3), which requires O(t2d) computation, we update U t recursively

U t =

U t−1kUt ⊤

U t−1 + sigmoid Q

d + [M Ut ]:,t vUt 01×d

√

, (13)

where [M Ut ]:,t is the t-th column of M Ut . The proof of (13) is given in Appendix E. Next, we discuss the caching strategy and FLOPs of the updating step.

- • Caching in updating step. First, it is obvious that we need to cache U t so that we can implement the

recursive formula (13). Second, we need to cache QUt because qUs is used in any update from U t−1 to

U t with s ≤ t − 1. As kUt and vUt are only used in the update from U t−1 to U t and never used again in update from U j−1 to U j with j ̸= t, we do not need to cache any other variables except U t and QUt for the updating step.

- • FLOPs in updating step. With cached U t−1 and QUt−1, the updating formula (13) needs only O(td)

FLOPs. And computing qUt , kUt and vUt needs O(ddmodel) FLOPs, yielding total FLOPs of O(ddmodel + td).

Combining step. In the combining step, we compute qCt = xtW CQ, kCt = xtW CK and vCt = xtW CV . Next, the attention outputs are then obtained by applying (5), (6), (7) and (8) with O(td) FLOPs. At this stage,

since we already cached U t in the updating step, only KCt and V Ct need to be stored.

UQ-KV cache. From the above analysis, the counterpart of the KV cache in CASTLE consists of U t, QUt , KCt , and V Ct , which we collectively refer to as the UQ-KV cache. All other variables, including qCt , kUt , and vUt , can be safely disposed of after use.

model. We first compute QUL = XLW UQ, KUL = XLW UK, V UL = XLW UV , QCL = XLW CQ, KCL = XLW CK, V CL = XLW CV . Then, we apply the forward pass of the efficient parallel training algorithm (Algorithm 1) to get Attention(XL). For

For the prefilling stage, let the prompt length be L and inputs XL ∈ RL×d

the UQ-KV cache, since we already have QUL, KCL and V CL, we only need to obtain U L. This can be done similarly to FlashAttention-2 [5] due to the similarity between (3) and standard causal attention. The complete prefilling procedure is given in Algorithm 4. The analysis above leads to the following theorem.

- Theorem 3. Given inputs XL ∈ RL×d

model, for both CASTLE and CASTLE-SWL, prefilling has computational complexity O(Lddmodel + L2d) and space complexity O(Ld), and during the decoding stage, when generating the t-th token, the computational complexity is O(td + ddmodel) and the UQ-KV cache requires O(td) memory.

###### 2.5 Multi-Head Causal Attention with Lookahead Keys

As in standard causal attention, we also utilize multi-head mechanism. Let n denote the number of heads. In each head i, when generating the t-th token, denote the output as in (8) by attentioni(Xt) ∈ R1×d. Then, the output of multi-head causal attention with lookahead keys (multi-head CASTLE) is calculated as

multi-head-attention(Xt) = Concat attention1(Xt),...,attentionn(Xt) W O ∈ R1×d, (14) where W O ∈ Rnd×d

model is a learnable matrix. The formula for forward pass in parallel training and more details of multi-head CASTLE are in Appendix C. Multi-head CASTLE-SWL shares the same formulation and parameter count as CASTLE, and is omitted here to avoid redundancy.

##### 3 Experiments

###### 3.1 Experimental Setup

We use the nanoGPT [13] code base. Our baseline follows the improved Transformer architecture with SwiGLU [22], Rotary Positional Embeddings [24], and RMSNorm [32] following LLaMA [26]. We train models on four scales from scratch: small (0.16B), medium (0.35B), large (0.75B) and XL (1.3B). The medium, large and XL baseline models mirror the configuration of GPT-3 [3]. For the small setting, we increase the number of heads and the hidden dimension relative to the original GPT-3 configuration to better align parameter counts between standard causal attention and CASTLE. CASTLE-SWL shares identical configurations with CASTLE of each model scale. To isolate the effect of the attention mechanism, we replace standard causal attention with CASTLE or CASTLE-SWL while keeping all other components unchanged for a fair comparison. We use the AdamW optimizer [16] and follow the training recipe of [6]. All models are trained on FineWeb-Edu dataset [17] for 50B tokens. Further details of experimental setup can be found in Appendix A.1.

###### 3.2 Training & Validation Loss

We report training and validation loss and perplexity in Table 1. Training loss and validation loss curves are shown in Figure 6, Figure 7, Figure 8 and Figure 9. CASTLE and CASTLE-SWL consistently outperform the baselines in both training and validation loss across all model scales.

Specifically, after training 50B tokens, CASTLE outperforms baselines across all model scales and achieves validation losses that are 0.0059, 0.0245, 0.0356, and 0.0348 lower than the baseline for the small, medium, large, and XL models, respectively.

We hypothesize that this improvement stems from lookahead keys in CASTLE. By incorporating lookahead keys, CASTLE facilitates better global context capture. However, smaller models may struggle to fully leverage this mechanism due to limited capacity to model complex global relationships. As a result, the improvement margin for the small model is less significant compared to larger models.

CASTLE-SWL matches CASTLE’s performance and its validation loss outperforms baselines by 0.0084, 0.0241, 0.0366, 0.0369, respectively. The performance gains are particularly significant in the medium, large, and XL models.

Furthermore, as shown in Table 4, both CASTLE and CASTLE-SWL have the same or fewer parameters than their baseline counterparts. This further underscores CASTLE and CASTLE-SWL’s effectiveness in improving model performance.

- Table 1 Training and validation loss and perplexity for models with CASTLE, CASTLE-SWL and standard causal attention. We use “S”, “M”, “L”, “XL” to denote model scales. Each model is trained on FineWeb-Edu for 50B tokens. The best and second best results (when showing clear improvements upon baselines) are highlighted in bold and underline, respectively.

Train Eval nparams Loss PPL Loss PPL

Baseline-S 160M 2.795 16.364 2.798 16.411 CASTLE-S 160M 2.789 16.259 2.792 16.315 CASTLE-SWL-S 160M 2.786 16.213 2.790 16.273

Baseline-M 353M 2.641 14.030 2.639 14.004 CASTLE-M 351M 2.616 13.684 2.615 13.665 CASTLE-SWL-M 351M 2.618 13.708 2.615 13.670 Baseline-L 756M 2.513 12.346 2.507 12.269 CASTLE-L 753M 2.476 11.897 2.472 11.840 CASTLE-SWL-L 753M 2.476 11.890 2.471 11.832

Baseline-XL 1.310B 2.430 11.360 2.426 11.309 CASTLE-XL 1.304B 2.401 11.031 2.391 10.922 CASTLE-SWL-XL 1.304B 2.401 11.036 2.389 10.900

###### Training Loss (XL Model)

###### Validation Loss (XL Model)

3.1

3.0

Baseline-XL (1.310B)

Baseline-XL (1.310B)

CASTLE-XL (1.304B)

CASTLE-XL (1.304B)

3.0

CASTLE-SWL-XL (1.304B)

CASTLE-SWL-XL (1.304B)

2.9

2.9

2.8

2.8

2.7

2.7

2.6

2.6

2.5

2.5

2.4

2.4

0 10 20 30 40 50 Training Tokens (B)

0 10 20 30 40 50 Training Tokens (B)

- Figure 6 Training and validation loss curves of XL models. Training loss curve is smoothened with a moving window of 2000 training steps. Validation loss is evaluated every 100 training steps on 40M tokens, and its curve is smoothened by a moving window of 20 evaluation intervals. Loss curves for the small, medium and large models can be found in
- Figure 7, Figure 8 and Figure 9 of Appendix A.3. After 50B training tokens, CASTLE-XL achieves a 0.0294 lower training loss and a 0.0348 lower validation loss compared to Baseline-XL, while CASTLE-SWL-XL achieves a 0.0290 lower training loss and a 0.0369 lower validation loss compared to Baseline-XL

###### 3.3 Downstream Evaluation

We evaluate CASTLE and CASTLE-SWL on a diverse set of downstream benchmarks, including ARC [29], BoolQ [4], HellaSwag [31], MMLU [11], OBQA [18], PIQA [2], Winograde [21] using lm-evaluation-harness [10]. We report normalized accuracy for ARC-Challenge, HellaSwag, OBQA, and PIQA, and standard accuracy for the other benchmarks. Results are reported in Table 2 (0-shot) and Table 3 (5-shot). Across all model scales and evaluation settings, both CASTLE and CASTLE-SWL consistently outperform the baselines in average downstream accuracy under both 0-shot and 5-shot settings. These findings accompany lower loss and perplexity in Section 3.2 demonstrate that CASTLE and CASTLE-SWL not only lower perplexity but

###### also translate these gains into stronger performance on diverse downstream tasks.

- Table 2 Evaluation results (0-shot) for downstream tasks of different model scales. Each model is pretrained on FineWeb-Edu for 50B tokens. All values denote accuracy in percentage (%). The higher accuracy values are shown in bold. Hella.=HellaSwag, Wino.=Winograde.

Model Name ARC-C ARC-E BoolQ Hella. MMLU OBQA PIQA Wino. Avg. Baseline-S 26.71 54.76 52.51 35.78 22.89 30.40 63.98 52.57 42.45 CASTLE-S 26.19 56.69 59.85 36.28 23.00 31.60 64.25 52.25 43.76 CASTLE-SWL-S 26.62 54.12 59.60 35.97 22.87 32.60 64.31 50.51 43.33 Baseline-M 28.58 60.90 53.61 43.01 23.21 33.40 67.95 50.91 45.20 CASTLE-M 30.20 61.36 58.01 43.24 25.34 34.60 67.95 52.64 46.67 CASTLE-SWL-M 29.52 61.41 57.03 43.76 23.29 33.00 68.12 53.83 46.24 Baseline-L 32.59 65.07 57.49 47.45 23.57 32.60 70.51 50.75 47.50 CASTLE-L 32.34 65.15 57.65 47.87 24.51 35.60 70.78 53.51 48.43 CASTLE-SWL-L 32.76 63.51 60.95 48.50 23.72 36.00 70.02 54.85 48.79 Baseline-XL 33.79 66.62 61.04 51.40 26.72 36.20 72.58 54.06 50.30 CASTLE-XL 35.32 67.51 62.81 52.15 23.74 37.00 70.67 56.59 50.72 CASTLE-SWL-XL 36.43 69.07 60.24 51.99 24.47 37.40 71.27 55.09 50.74

- Table 3 Evaluation results (5-shot) for downstream tasks of different model scales. The higher accuracy values are shown in bold. All values denote accuracy in percentage (%). Each model is pretrained on FineWeb-Edu for 50B tokens. Hella.=HellaSwag, Wino.=Winograde.

Model Name ARC-C ARC-E BoolQ Hella. MMLU OBQA PIQA Wino. Avg. Baseline-S 25.68 54.97 56.09 33.81 25.54 28.20 63.98 52.57 42.60 CASTLE-S 26.02 54.25 57.13 35.24 25.22 29.80 64.53 50.99 42.90 CASTLE-SWL-S 27.39 56.19 56.09 35.46 25.34 30.20 64.85 51.22 43.34 Baseline-M 31.06 62.46 48.47 42.83 25.22 33.00 68.39 51.78 45.40 CASTLE-M 32.17 64.06 54.62 43.47 25.22 33.80 69.48 52.49 46.91 CASTLE-SWL-M 32.85 63.85 54.19 44.18 26.43 33.80 69.04 53.43 47.22 Baseline-L 33.36 63.64 59.24 46.16 26.82 33.40 69.53 54.06 48.28 CASTLE-L 37.37 67.89 50.95 47.71 26.11 34.20 70.18 54.06 48.56 CASTLE-SWL-L 36.26 65.53 60.58 48.55 24.70 34.80 69.10 53.67 49.15 Baseline-XL 35.58 65.78 61.07 50.84 26.71 36.20 71.27 52.72 50.02 CASTLE-XL 39.08 70.24 62.60 51.63 24.16 37.40 71.00 58.33 51.80 CASTLE-SWL-XL 38.99 70.08 61.74 52.35 25.85 37.20 72.52 56.75 51.93

- 4 Conclusion

We introduced CAuSal aTtention with Lookahead kEys (CASTLE), a novel attention mechanism that continually updates keys as the context evolves. This design allows each key representation to incorporate more recent information at every prediction step while strictly preserving the autoregressive property. Although CASTLE is defined recurrently, we derived a mathematical equivalence that eliminates the need to explicitly materialize lookahead keys at each position, enabling efficient parallel training. Experimental results on language modeling demonstrate that CASTLE consistently outperforms standard causal attention, achieving lower perplexity and stronger downstream performance.

##### References

- [1] Samuel Joseph Amouyal, Aya Meltzer-Asscher, and Jonathan Berant. When the lm misunderstood the human chuckled: Analyzing garden path effects in humans and language models. arXiv preprint arXiv:2502.09307, 2025.

- [2] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [4] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

- [5] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024.

- [6] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In International Conference on Machine Learning, pages 10041–10071. PMLR, 2024.

- [7] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

- [8] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 320–335, 2022.

- [9] Ethan Ewer, Daewon Chae, Thomas Zeng, Jinkyu Kim, and Kangwook Lee. Entp: Encoder-only next token prediction. arXiv preprint arXiv:2410.01600, 2024.

- [10] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/ records/12608602.
- [11] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

- [12] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- [13] Andrej Karpathy. nanogpt: The simplest, fastest repository for training/finetuning medium-sized gpts. URL https://github. com/karpathy/nanoGPT, 2023.

- [14] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Selective attention improves transformer. arXiv preprint arXiv:2410.02703, 2024.

- [15] Xianming Li and Jing Li. Bellm: Backward dependency enhanced large language model for sentence embeddings. arXiv preprint arXiv:2311.05296, 2023.

- [16] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

- [17] Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface. co/datasets/HuggingFaceFW/fineweb-edu.

- [18] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.

- [19] Bradley Louis Pritchett. Garden path phenomena and the grammatical basis of language processing. Harvard University, 1987.

- [20] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- [21] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020.

- [22] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

- [23] Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. In The Thirteenth International Conference on Learning Representations.

- [24] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

- [25] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019.

- [26] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

- [27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [28] Xiaohan Xu, Chongyang Tao, Tao Shen, Can Xu, Hongbo Xu, Guodong Long, Jian-Guang Lou, and Shuai Ma. Re-reading improves reasoning in large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15549–15575, 2024.

- [29] Vikas Yadav, Steven Bethard, and Mihai Surdeanu. Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering. arXiv preprint arXiv:1911.07176, 2019.

- [30] Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren, Rameswar Panda, and Yoon Kim. Path attention: Position encoding via accumulating householder transformations. arXiv preprint arXiv:2505.16381, 2025.

- [31] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

- [32] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

- [33] Siyue Zhang, Yilun Zhao, Liyuan Geng, Arman Cohan, Anh Tuan Luu, and Chen Zhao. Diffusion vs. autoregressive language models: A text embedding perspective. arXiv preprint arXiv:2505.15045, 2025.

## Appendix

##### A Experimental Details and Additional Results

- A.1 Experimental Setup We give details of our experimental setup in this section.

###### A.1.1 Model Architecture

We use the improved Transformer architecture with SwiGLU [22], Rotary Positional Embeddings [24], and RMSNorm [32] following LLaMA [26]. More specifically, in each layer l, given the contextualized representations X(l) ∈ RL×d

model where L is the sequence length and dmodel is the hidden dimension, then, X(l+1) is obtained by

Y (l) = MultiHead-Attention(RMSNorm(X(l))) X(l+1) = SwiGLU(RMSNorm(Y (l))),

model×83dmodel, W 3 ∈ R83dmodel×dmodel are learnable parameters.

model×83dmodel, W 2 ∈ Rd

where the SwiGLU(X) = Swish(XW 1)⊙(XW 2) W 3. Here, W 1 ∈ Rd

The function MultiHead-Attention is instantiated with either the standard multi-head causal attention or the multi-head CASTLE, or the multi-head CASTLE-SWL.

###### A.1.2 Model Configuration and Training Recipe

We train models on four scales from scratch: small (0.16B), medium (0.35B), large (0.75B) and XL (1.3B), where the medium, large and XL baseline models follow the configurations of GPT-3 [3]. For the small setting, we increase the number of heads and the hidden dimension in the original GPT-3 configuration to better align parameter counts between standard causal attention and CASTLE. The configurations of the models can be found in Table 4. To ensure a fair comparison between CASTLE and standard causal attention, we align the number of model parameters by adjusting the number of attention heads and keeping hidden dimensions and head dimensions invariant. This avoids changes to the representational upper bound of the models’ hidden states and the behavior of RoPE, both of which depend on the hidden dimension. As shown in Appendix C, the number of parameters in CASTLE matches that of standard causal attention when the number of heads satisfies the relation nCASTLE = 47nstandard, where nCASTLE and nstandard denote the number of heads in CASTLE and standard causal attention, respectively. For the small model, the baseline uses 14 heads. By setting CASTLE to 8 heads, we align the parameter counts. For the other settings, the baseline models use 16 heads. As 16 ∗ 47 ≈ 9.14, we choose 9 heads for CASTLE, resulting in a slightly smaller number of parameters than the baseline.

The model configurations and training recipes of CASTLE-SWL are identical to those of CASTLE at each model scale.

The sliding window size for lookahead keys in CASTLE-SWL is set to 128 in CASTLE-SWL-S, and 512 in CASTLE-SWL-M, CASTLE-SWL-L, and CASTLE-SWL-XL. Appendix A.2.4 presents an ablation study on sliding window sizes, showing that the performance of CASTLE-SWL is generally insensitive to the choice of window size within the range [128,1024].

We adopt the training hyper-parameters of [6]. We use the AdamW optimizer [16] with β1 = 0.9, β2 = 0.95, weight decay rate coefficient 0.1, and gradient clipping coefficient 1.0. All models are trained with sequence length 2K and batch size 0.5M tokens. The small, medium, large and XL models use peak learning rates of 6.0 × 10−4, 3.0 × 10−4, 2.5 × 10−4 and 2.0 × 10−4, respectively. All models are trained with 2000 warmup steps, and then, the cosine scheduler decays the learning rate to 10% of the peak learning rate. All models are trained on the FineWeb-Edu dataset [17] for 50 billion tokens. The efficient parallel training algorithm of CASTLE and CASTLE-SWL (forward pass in Algorithm 1 and backward pass in Algorithm 3) is implemented in Triton [25].

Table 4 Configurations and learning hyper-parameters (batch size in tokens and learning rate) of the models which we trained. CASTLE-SWL has exactly the same configurations and training hyper-parameters with CASTLE on each model scale, and is omitted from this table for clarity.

Model Name nparams nlayers dmodel nheads d Batch Size Learning Rate Baseline-S 160M 12 896 (=14 * 64) 14 64

0.5M 6.0 × 10−4

CASTLE-S 160M 12 896 8 64 Baseline-M 353M 24 1024 (=16 * 64) 16 64

0.5M 3.0 × 10−4

CASTLE-M 351M 24 1024 9 64 Baseline-L 756M 24 1536 (=16 * 96) 16 96

0.5M 2.5 × 10−4

CASTLE-L 753M 24 1536 9 96 Baseline-XL 1.310B 24 2048 (=16 * 128) 16 128

0.5M 2.0 × 10−4 CASTLE-XL 1.304B 24 2048 9 128

###### A.2 Ablation Studies

We conduct ablation studies to better understand the contributions of different design components in CASTLE and CASTLE-SWL. These studies address three key questions:

- • Are causal keys necessary, or could lookahead keys alone suffice?
- • Do the observed improvements arise from the mechanism of lookahead keys, or simply from increasing the number of keys?
- • What is the role of the SiLU function in (7)?
- • How do sliding window sizes of lookahead keys in CASTLE-SWL affect the performance?

We systematically investigate each question in the following sections. All ablation experiments are trained on FineWeb-Edu for 25B tokens.

###### A.2.1 Ablations on Causal Keys

As described in Section 2.2, CASTLE adopts a hybrid design that partitions keys into two groups: causal keys and lookahead keys. If all lookahead keys are replaced with causal keys, CASTLE reduces to standard causal attention. Thus, the performance gains demonstrated in Section 3 can be attributed to the introduction of lookahead keys. A natural follow-up question is whether causal keys are necessary, or if lookahead keys alone suffice.

To investigate this, we construct a variant of CASTLE in which all causal keys are removed. To ensure a fair comparison, we adjust the configurations so that the total parameter count remains the same, as shown in Table 5.

Table 5 Configurations of CASTLE and its variant without causal keys used in ablation study on causal keys.

CASTLE TYPE nparams nlayers dmodel nheads d

CASTLE 120M 12 768 6 64 CASTLE w/o causal keys 120M 12 768 7 64

The above 2 models are both trained on FineWeb-Edu for 25B tokens for efficiency, using the same learning hyper-parameters with the small models in Section A.1.2. Their training and validation loss and perplexity are presented in Table 6.

As shown in Table 6, removing causal keys results in a clear degradation in performance. This demonstrates that causal keys are indispensable in CASTLE.

- Table 6 Training and validation loss and perplexity of CASTLE and its variant without causal keys. Each model is trained on FineWeb-Edu for 25B tokens.

Train Eval Loss PPL Loss PPL

CASTLE 2.913 18.417 2.920 18.541 CASTLE w/o causal keys 3.006 20.213 3.021 20.505

While these results establish the necessity of both causal and lookahead keys, our current formulation in (7) employs a one-to-one pairing of a causal key and a lookahead key. An alternative design could involve grouping multiple causal keys with a single lookahead key, or vice versa. Exploring the optimal ratio between causal keys and lookahead keys is left for future work.

- A.2.2 Ablations on the Number of Keys As discussed in Section C, when CASTLE uses half as many heads as standard causal attention, its parameter

count becomes 78 of the baseline. To maintain comparable parameter counts, we adjust the number of heads accordingly. However, unlike the baseline where each head corresponds to one key, each head in CASTLE

introduces two keys—one causal key and one lookahead key.

This design results in CASTLE models having 16, 18, 18, and 18 keys for the small, medium, large, and XL scales, respectively, compared to the corresponding baselines with 14, 16, 16, and 16 keys (Table 4). Thus, CASTLE naturally uses slightly more keys than its baseline counterparts. A natural question arises: are the observed improvements due to the introduction of lookahead keys, or simply from having more keys overall?

To disentangle this effect, we construct CASTLE variants with only half as many heads as their baselines, ensuring that the total number of keys (nCausalKeys + nLookaheadKeys) matches the baselines. This adjustment results in CASTLE having a notably smaller parameter count than the baselines.

We also did the same ablation studies for CASTLE-SWL. For efficiency, we train the medium and XL variants on FineWeb-Edu for 25B tokens, using the same hyperparameters as in Section A.1.2. Results are reported in Table 8 (CASTLE) and Table 9 (CASTLE-SWL).

As shown in Table 8, despite having clearly fewer parameters, both CASTLE-M-16 and CASTLE-XL-16 outperform their baselines: CASTLE-M-16 lags behind CASTLE-M by only 0.005 in validation loss, yet surpasses the baseline by 0.026; CASTLE-XL-16 trails CASTLE-XL by 0.008 in validation loss, while exceeding the baseline by 0.032. Similar performance of CASTLE-SWL can be observed in Table 9.

These findings confirm that CASTLE and CASTLE-SWL’s advantage stems from its mechanism of incorporating lookahead keys, rather than from increasing the number of keys from 16 to 18. This further consolidates the advantage of CASTLE and CASTLE-SWL.

- Table 7 Configurations of baseline models, CASTLE, and its variants used in the ablation study on the number of keys. Baseline-M, Baseline-XL, CASTLE-M, and CASTLE-XL follow the same configurations as in Table 4. CASTLE-M-16 and CASTLE-XL-16 are constructed by reducing the number of heads in CASTLE-M and CASTLE-XL, respectively,

so that the total number of keys (nLookaheadKeys + nCausalKeys) matches the number of keys of the corresponding baseline models. CASTLE-SWL-M, CASTLE-SWL-XL, CASTLE-SWL-M-16, CASTLE-SWL-XL-16 have identical configurations with CASTLE-M, CASTLE-XL, CASTLE-M-16, CASTLE-XL-16, respectively and are omitted from this table for clarity.

Model Name nparams nlayers dmodel nheads nLookaheadKeys + nCausalKeys d

Baseline-M 353M 24 1024 (=16 * 64) 16 16 64 CASTLE-M 351M 24 1024 9 18 64 CASTLE-M-16 340M 24 1024 8 16 64

Baseline-XL 1.310B 24 2048 (=16 * 128) 16 16 128 CASTLE-XL 1.304B 24 2048 9 18 128 CASTLE-XL-16 1.260B 24 2048 8 16 128

- Table 8 Training and validation loss and perplexity of baseline models, CASTLE and CASTLE variants with the same number of keys as the baselines, after training for 25B tokens on FineWeb-Edu. The lowest loss and perplexity are shown in bold, and the second-lowest values are underlined.

Train Eval

nparams Loss PPL Loss PPL Baseline-M 353M 2.740 15.483 2.742 15.523 CASTLE-M 351M 2.709 15.018 2.711 15.039 CASTLE-M-16 340M 2.714 15.093 2.716 15.126 Baseline-XL 1.310B 2.548 12.779 2.543 12.723 CASTLE-XL 1.304B 2.507 12.267 2.503 12.219 CASTLE-XL-16 1.260B 2.514 12.349 2.511 12.316

- Table 9 Training and validation loss and perplexity of baseline models, CASTLE-SWL and CASTLE-SWL variants with the same number of keys as the baselines, after training for 25B tokens on FineWeb-Edu. The lowest loss and perplexity are shown in bold, and the second-lowest values are underlined.

Train Eval

nparams Loss PPL Loss PPL Baseline-M 353M 2.740 15.483 2.742 15.523 CASTLE-SWL-M 351M 2.710 15.036 2.713 15.068 CASTLE-SWL-M-16 340M 2.716 15.117 2.718 15.150 Baseline-XL 1.310B 2.548 12.779 2.543 12.723 CASTLE-SWL-XL 1.304B 2.506 12.255 2.503 12.217 CASTLE-SWL-XL-16 1.260B 2.513 12.339 2.508 12.276

###### A.2.3 Ablations on SiLU function in (7)

We investigate the role of the SiLU activation introduced in (7). When training CASTLE-XL and CASTLESWL-XL without SiLU, we consistently observed the loss blows up and shows NaN then. In particular, CASTLE-SWL-XL blows up around step 24300, whereas CASTLE-XL blows up earlier, around step 1500. Lowering the learning rate mitigates this instability. For example, CASTLE-SWL-XL without SiLU can remain stable up to at least 25B tokens when trained with peak learning rates of 1 × 10−4 or 5 × 10−5. However, such reductions in learning rate lead to noticeably worse performance, as shown in Table 10.

- Table 10 Ablation study of the SiLU function in (7). Removing SiLU causes training instability. More specifically, training CASTLE-SWL-XL without SiLU in (7) under the same learning rate (2 × 10−4) as in Table 4 will have the loss curve blow up at around 24300 training step. Reducing the learning rate alleviates this instability issue but results in significant performance degradation.

Train Eval Learning Rate Loss PPL Loss PPL

CASTLE-SWL-XL 2 × 10−4 2.506 12.255 2.503 12.217 CASTLE-SWL-XL w/o SiLU 1 × 10−4 2.523 12.468 2.520 12.424 CASTLE-SWL-XL w/o SiLU 5 × 10−5 2.571 13.084 2.571 13.075

A.2.4 Ablations on Sliding Window Size in Lookahead Keys of CASTLE-SWL

We investigate the effect of sliding window size on CASTLE-SWL across different model scales. For efficiency, each model is trained on the FineWeb-Edu dataset for 25B tokens. We evaluate CASTLE-SWL with window sizes of 128, 256, 512, and 1024, and report both training and validation loss and perplexity. The results for small, medium, large, and XL models are shown in Table 11, Table 12, Table 13, and Table 14, respectively. Throughout, we use the number following “SWL” to denote the sliding window size, e.g., CASTLE-SWL128-S refers to CASTLE-SWL-S with a window size of 128.

As shown in Table 11, a window size of 128 achieves the best performance for the small model. For medium, large, and XL models (Tables 12, 13, and 14), the optimal performance is obtained with a window size of 512. Based on these findings, we adopt window sizes of 128, 512, 512, and 512 for small, medium, large, and XL models, respectively, in the experiments presented in Section 3.

Overall, CASTLE-SWL performance is not sensitive to the choice of sliding window size. For example, for the medium model, the best sliding window size (512) reduces validation loss by 0.0297 compared to the baseline, but the gap between the best (512) and worst window sizes (256) is only 0.0038. Similarly, for the XL model, the best sliding window (512) improves upon the baseline by 0.0406 while the difference between the best (512) and worst (128) sliding window sizes is just 0.0099. These results suggest that while CASTLE-SWL consistently improves over the baselines across all model scales and sliding window sizes tested, and its performance is relatively robust to the choices of window sizes.

- Table 11 Ablations on sliding window sizes for CASTLE-SWL-S. Training and validation loss and perplexity of baseline models, CASTLE-SWL-S with different sliding window sizes after training for 25B tokens on FineWeb-Edu are reported. The lowest loss and perplexity are shown in bold, and the second-lowest values are underlined.

Train Eval nparams Loss PPL Loss PPL

Baseline-S 160M 2.892 18.037 2.901 18.197 CASTLE-SWL128-S 160M 2.883 17.859 2.889 17.971 CASTLE-SWL256-S 160M 2.888 17.952 2.895 18.092 CASTLE-SWL512-S 160M 2.885 17.910 2.892 18.023 CASTLE-SWL1024-S 160M 2.885 17.901 2.892 18.031

- Table 12 Ablations on sliding window sizes for CASTLE-SWL-M. Training and validation loss and perplexity of baseline models, CASTLE-SWL-M with different sliding window sizes after training for 25B tokens on FineWeb-Edu are reported. The lowest loss and perplexity are shown in bold, and the second-lowest values are underlined.

Train Eval nparams Loss PPL Loss PPL

Baseline-M 353M 2.740 15.483 2.742 15.523 CASTLE-SWL128-M 351M 2.715 15.098 2.716 15.124 CASTLE-SWL256-M 351M 2.715 15.112 2.716 15.127 CASTLE-SWL512-M 351M 2.710 15.036 2.713 15.068 CASTLE-SWL1024-M 351M 2.713 15.071 2.715 15.103

- Table 13 Ablations on sliding window sizes for CASTLE-SWL-L. Training and validation loss and perplexity of baseline models, CASTLE-SWL-L with different sliding window sizes after training for 25B tokens on FineWeb-Edu are reported. The lowest loss and perplexity are shown in bold, and the second-lowest values are underlined.

Train Eval nparams Loss PPL Loss PPL

Baseline-L 756M 2.740 15.483 2.742 15.523 CASTLE-SWL128-L 753M 2.597 13.425 2.596 13.411 CASTLE-SWL256-L 753M 2.589 13.314 2.587 13.290 CASTLE-SWL512-L 753M 2.582 13.219 2.580 13.202 CASTLE-SWL1024-L 753M 2.582 13.229 2.581 13.209

- Table 14 Ablations on sliding window sizes for CASTLE-SWL-XL. Training and validation loss and perplexity of baseline models, CASTLE-SWL-XL with different sliding window sizes after training for 25B tokens on FineWeb-Edu are reported. The lowest loss and perplexity are shown in bold, and the second-lowest values are underlined.

Train Eval nparams Loss PPL Loss PPL

Baseline-XL 1.310B 2.548 12.779 2.543 12.723 CASTLE-SWL128-XL 1.304B 2.517 12.393 2.513 12.339 CASTLE-SWL256-XL 1.304B 2.514 12.353 2.510 12.300 CASTLE-SWL512-XL 1.304B 2.506 12.255 2.503 12.217 CASTLE-SWL1024-XL 1.304B 2.514 12.351 2.509 12.294

###### A.3 Additional Loss Curves

This section presents additional training and validation loss curves for the small, medium, and large models, while the loss curves for the XL models are already shown in Figure 6. Each figure compares the baseline with CASTLE and CASTLE-SWL.

###### Training Loss (Small Model)

###### Validation Loss (Small Model)

Baseline-S (160M)

Baseline-S (160M)

3.3

3.3

CASTLE-S (160M)

CASTLE-S (160M)

CASTLE-SWl-S (160M)

CASTLE-SWL-S (160M)

3.2

3.2

3.1

3.1

3.0

3.0

2.9

2.9

2.8

2.8

0 10 20 30 40 50 Training Tokens (B)

0 10 20 30 40 50 Training Tokens (B)

- Figure 7 Training and validation loss curves of small models. Training loss curve is smoothened with a moving window of 2000 training steps. Validation loss is evaluated every 100 training steps on 40M tokens, and its curve is smoothened by a moving window of 20 evaluation intervals. As seen in Table 1 and in comparison with Figure 8, Figure 9 and Figure 6, CASTLE yields only marginal improvements over the baseline on small models. A likely explanation is that this is because the benefit of lookahead keys may lie in helping models capture global dependencies, but small models are capacity-limited and can primarily extract local features, making global relations less useful at this scale.

0 10 20 30 40 50 Training Tokens (B)

2.6

2.7

2.8

2.9

- 3.0

- 3.1

- 3.2

- 3.3

Training Loss (Medium Model)

Baseline-M (353M)

CASTLE-M (351M)

CASTLE-SWL-M (351M)

0 10 20 30 40 50 Training Tokens (B)

2.6

2.7

2.8

2.9

3.0

3.1

3.2

3.3

Validation Loss (Medium Model)

Baseline-M (353M)

CASTLE-M (351M)

CASTLE-SWL-M (351M)

- Figure 8 Training and validation loss curves of medium models. Training loss curve is smoothened with a moving window of 2000 training steps. Validation loss is evaluated every 100 training steps on 40M tokens, and its curve is smoothened by a moving window of 20 evaluation intervals. After 50B training tokens, CASTLE-M achieves a 0.0294 lower training loss and a 0.0245 lower validation loss compared to Baseline-M, while CASTLE-SWL-M achieves a 0.0232 lower training loss and a 0.0241 lower validation loss compared to Baseline-M

###### Training Loss (Large Model)

###### Validation Loss (Large Model)

Baseline-L (756M)

Baseline-L (756M)

3.1

3.1

CASTLE-L (753M)

CASTLE-L (753M)

CASTLE-SWL-L (753M)

CASTLE-SWL-L (753M)

3.0

3.0

2.9

2.9

2.8

2.8

2.7

2.7

2.6

2.6

2.5

2.5

0 10 20 30 40 50 Training Tokens (B)

0 10 20 30 40 50 Training Tokens (B)

- Figure 9 Training and validation loss curves of large models. Training loss curve is smoothened with a moving window of 2000 training steps. Validation loss is evaluated every 100 training steps on 40M tokens, and its curve is smoothened by a moving window of 20 evaluation intervals. After 50B training tokens, CASTLE-L achieves a 0.0371 lower training loss and a 0.0356 lower validation loss compared to Baseline-L, while CASTLE-SWL-L achieves a 0.0376 lower training loss and a 0.0366 lower validation loss compared to Baseline-L

##### B Proof of Theorem 1





x1 x2

First, recall the notations in Section 2.3. More specifically, consider inputs XL =

model, where

∈ RL×d

 

 

. xL

xt is the representation of the t-th input token, L is the sequence length and dmodel is the hidden dimension. For each 1 ≤ t ≤ L, denote

qUt = xtW UQ, kUt = xtW UK, vUt = xtW UV , qCt = xtW CQ, kCt = xtW CK, vCt = xtW CV .













- kU1
- kU2

- vU1
- vU2

- qU1
- qU2

= XtW UV ,

= XtW UK, V Ut =

= XtW UQ, KUt =

QUt =

 

 

 

 

 

 

. vUt

. qUt

. kUt













kC1 kC2 . kCt

vC1 vC2

qC1 qC2

= XtW CV .

= XtW CQ, KCt =

= XtW CK, V Ct =

QCt =

 

 

 

 

 

 

. vCt

. qCt

And M Ct , M Ct and M Ut are t-by-t mask matrices. M Ct is the t-by-t causal mask which prevents tokens from attending to their future tokens, i.e., [M Ct ]ij = 0 if i ≥ j and [M Ct ]ij = −∞ otherwise; [ M Ct ]ij = 1 if i ≥ j and [ M Ct ]ij = 0 otherwise. For CASTLE, M Ut is defined in (4) and for CASTLE-SWL, M Ut is defined in (9). For the projection matrices of the entire sequence XL, we drop L as in Theorem 1 for simplicity, i,e,

QU = QUL = XLW UQ, KU = KUL = XLW UK, V U = V UL = XLW UV , QC = QCL = XLW CQ, KC = KCL = XLW CK, V C = V CL = XLW CV .

And M U = M UL, M C = M CL. Then, M Ut is a t-by-t submatrix of M U = M UL. Similarly, M Ct is also a submatrix of M C.

Consider when we are generating the (t + 1)-th token. As in (3)





- ut1
- ut2

QUt KUt ⊤

+ M Ut V Ut ∈ Rt×d

U t =

√

= sigmoid

 

 

. utt

d

Then, the lookahead-key attention scores as in (6) are

qCt U t⊤

sUt =

√

d

⊤

U t KUt ⊤

qCt V Ut ⊤ sigmoid Q

d + M Ut

√

√

=

.

d

(15)

We will need the following lemma to proceed.

Lemma 1. For any vector a ∈ R1×t, let a = (a,01×(L−t)) ∈ R1×L, where 01×(L−t) is the all-zeros vector of size (1,L − t). Then,

  = a sigmoid

 a sigmoid

⊤

⊤

###### QUKU⊤

QUt KUt ⊤

+ M Ut

+ M U

,01×(L−t)

√

√

.

d

d

Proof of Lemma 1. The proof is straightforward by the fact that

- 1. The upper triangular entries of the transposed matrix sigmoid Q

U√KU⊤

d + M U

⊤

are all 0 by the definition of M U.

- 2. The matrix sigmoid Q

U t KUt ⊤

U√KU⊤

d +M Ut equals an upper-left block of the matrix sigmoid Q

d +M U , i.e.,

√

sigmoid

QUt KUt ⊤

+ M Ut = sigmoid

√

d

###### QUKU⊤

+ M U

√

d

where for any matrix A, A1:t,1:t refers to its top-left t-by-t submatrix.

,

1:t,1:t

| |
|---|

Define the vector at ∈ R1×L as [at]i = qCt V Ut i if 1 ≤ i ≤ t and [at]i = 0 otherwise. Denote sUt = sUt ,01×(L−t) . Then, by combining (15) with Lemma 1, we have





⊤

U t KUt ⊤

qCt V Ut ⊤ sigmoid Q

d + M Ut

√

sUt = sUt ,01×(L−t) =

,01×(L−t)

√

 

 

d

⊤

U√KU⊤

at sigmoid Q

d + M U

√

.

=

d

Since V Ut equals the submatrix which consists of first t rows of V U, [at]i = qCt V U⊤

for 1 ≤ i ≤ t. Then, by stacking aj together, we have

i





- a1
- a2

= QCV U⊤ ⊙ M C,

 

 

. aL

where M C is defined in Theorem 1 with M Cij = 1 if i ≥ j and M Cij = 0 otherwise.

We concatenate sUt

in an L-by-L matrix. Then, 

1≤t≤L







- sU1
- sU2

a1 a2

⊤

###### QUKU⊤

1 √

+ M U

√

=

sigmoid

 

 

 

 

. aL

. sUL

d

d

QCV U⊤

###### QUKU⊤

d ⊙ M C sigmoid

+ M U

√

√

=

d

= SU,

⊤

(16)

where SU is given in (11). The causal-key attention scores sCt in (5) is

qCt KCt ⊤

sCt =

√

.

d

We also denote sCt = sCt ,(−∞)1×(L−t) , where (−∞)1×(L−t) is a (L − t)-dimensional vector with all entries equaling −∞. Then, by concatenating sCt

into SC ∈ RL×L, we have

1≤t≤L





sC1 sC2 . sCL

QCKC⊤

SC =

+ M C. (17)

√

=

 

 

d

Then, the outputs Attention(XL) satisfies









attention X1 attention X2

softmax sC1 − SiLU(sU1 ) V C1 softmax sC2 − SiLU(sU2 ) V C2 . softmax sCL − SiLU(sUL) V CL

Attention XL =

=

 

 

 

 

. attention XL





softmax sC1 − SiLU( sU1 ) softmax sC2 − SiLU( sU2 )

###### V C

=

 

 

. softmax sCL − SiLU( sUL)

= row_softmax SC − SiLU SU V C

QCKC⊤

SU √

+ M C − SiLU

###### V C,

= row_softmax

√

d

d

where the last inequality above is from (16) and (17). This completes the proof of Theorem 1.

##### C Further Details on Multi-Head CASTLE

Forward pass for multi-head CASTLE. Denote n = nhead. Given contextualized representations XL, for each head, we can get Attentioni(XL) as in (12). Then, the outputs of multi-head CASTLE can be obtained by

Multi-head-Attention(XL) = Concat Attention1(XL),...,Attentionn(XL) W O ∈ RL×d.

Parameter count of multi-head CASTLE. In each head, the learnable parameters are W UQ, W UK, W UV , W CQ, W CK, W CV ∈ Rd

model×d. These parameters sum up to 6nheaddmodeld.

The matrix W O has nheadddmodel parameters. Thus, the multi-head CASTLE has 7nheadddmodel learnable parameters.

Multi-head CASTLE-SWL has identical formula and parameter counts with multi-head CASTLE and is omitted for clarity.

Parameter count of standard causal attention. The standard causal attention as in (2) has learnable parameters W Q, W K, W V ∈ Rd

model×d for each head and W O ∈ Rnd×d

model. These parameters sum up to

- 4nheadddmodel.

##### D Efficient Parallel Training Algorithm and Proof of Theorem 2 D.1 Forward Pass

Algorithm 1: Efficient parallel forward pass Require: QU, KU, V U, QC, KC, V C ∈ RL×d

- 1 # Initialization
- 2 Initialize D = 0d×L, O = 0L×d, ℓ = 0L×1 and m = (−∞)L×1 in HBM
- 3 # Diagonal blocks
- 4 for j = 0,...,N − 1 (in parallel) do

- 5 Load QCT

j,:, KCT

j,:, QUT

j,:, KUT

j,:, V UT

j,: from HBM to on-chip SRAM

- 6 On chip, compute SUT

j,Tj as in (22)

- 7 On chip, compute AT

j,Tj as in (19)

- 8 Implement online softmax update for block (Tj,Tj) by calling Algorithm 2
- 9 end
- 10 # 1-st, ···, (N − 1)-th off-diagonal blocks
- 11 for k = 1,...,N − 1 (sequential) do

- 12 for j = 0,...,N − k − 1 (in parallel) do

- 13 Load D:,T

j

, QUT

j,:, QCT

j+k

, KUT

j+k−1,:, KUT

j+k,:, V UT

j+k−1,:, V UT

j+k,: from HBM to on-chip SRAM

- 14 On chip, update D:,T

j

as in (23)

- 15 On chip, compute SUT

j+k,Tj as in (24)

- 16 Write D:,T

j

to HBM

- 17 On chip, compute AT

j+k,Tj as in (19)

- 18 Implement online softmax update for block (Tj+k,Tj) by calling Algorithm 2
- 19 end
- 20 end
- 21 Compute O ← diag(ℓ)−1 O ∈ RL×d, m ← m + log(ℓ) ∈ RL×1
- 22 Save m ∈ RL×1, D ∈ Rd×L for backward pass
- 23 Return the output O ∈ RL×d

Theorem 1 has shown a matrix form of Attention(XL) as in (12). However, computing Attention(XL) directly from (12) still need O(L3) computational costs because to compute SU, we need matrix multiplication between L-by-L matrices in (11). In this section, we give an efficient algorithm that enables efficient parallel training and reduces computational complexity to O(L2d).

We first introduce the notations in this section. We divide the sequence {1,...,L} into blocks of size B. For simplicity, we assume that L is divisible by B. Let N = BL and Ti = {i ∗ B + 1,...,(i + 1) ∗ B} for

- 0 ≤ i ≤ N − 1. Then, {1,2,...,L} = ∪Ni=0−1Ti.

For any matrix M , we use M T

i,Tj to denote the submatrix whose row and column indexes are in Ti and Tj,

Algorithm 2: Online softmax update for block (Ti,Tj) Require: AT

i,Tj on chip, ℓ, m, O, V C in HBM

- 1 Load ℓT

i

and mT

i

from HBM to on-chip SRAM

- 2 On chip, compute mnewT

i

= max(mT

i

,row_max(AT

i,Tj))

- 3 On chip, compute PT

i,Tj = exp(AT

i,Tj − mnewT

i

)

- 4 On chip, compute ℓnewT

i

= em

Ti−mnewTi ℓT

i

+ row_sum( PT

i,Tj)

- 5 Write OT

i,: ← diag em

Ti−mnewTi OT

i,: + PT

i,TjV CT

j,: to HBM

- 6 Write ℓT

###### , mT

to HBM

i ← ℓnewT

i ← mnewT

i

i

### Iteration 0

### Iteration 1

(1st off-diagonal blocks)

(Diagonal blocks)

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

### Iteration 2

### Iteration 3

(2nd off-diagonal blocks)

#### (3rd off-diagonal blocks)

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

Figure 10 Parallel scheme of Algorithm 1 (forward pass). We begin by computing the diagonal blocks of the attention score matrix A (Iteration 0). In each subsequent iteration k, the k-th off-diagonal blocks are computed. Blocks with different colors represent different kernel instances. In each iteration, each kernel instance is responsible for computing a single block of A and applying online softmax (Algorithm 2) to it. Kernel instances within the same iteration are launched in parallel, while the iterations are executed sequentially. The parallel scheme of Algorithm 3 (backward pass) is the reverse of the forward pass’s parallel scheme.

respectively. M T

i,: refers to the submatrix whose row indexes are in Ti. Analogously, M :,T

j

submatrix whose column indexes are in Tj. Denote the attention score matrix A by

refers to the

Then,

QCKC⊤

+ M C − SiLU SU . (18)

√

A =

d

Attention(XL) = row_softmax(A)V C and for any 0 ≤ i,j ≤ N − 1, the block AT

i,Tj satisfies

⊤

QCT

i,:KCT

i,Tj . (19) As in FlashAttention-2 [5], we compute each block of A and apply online softmax (Algorithm 2) on it to obtain Attention(XL). The first term Q

j,:

+ M CT

i,Tj − SiLU SUT

√

###### AT

i,Tj =

d

C Ti,:KCTj,:⊤

i,Tj as given by (19) can be computed similarly to FlashAttention-2. We mainly focus on the second term −SiLU SUT

d in AT

√

i,Tj which can incur a total computational complexity of O(L3 + L2d) if we apply (11) directly. Notice that as in (11),

SU =

QCV U⊤

d ⊙ M C sigmoid

√

###### QUKU⊤

+ M U

√

d

⊤

,

d ⊙ M C is a low-rank matrix multiplied by a mask matrix because QCV U⊤ has rank at most d which is normally much smaller than sequence length L. This motivates our algorithm to compute SU with computational complexity of O(L2d) while still enabling parallel training.

C√V U⊤

where the term Q

First, as the upper triangular entries in SU are all 0, we only need to focus on lower triangular entries in SU. We divide the computation of SU into the following parts

###### • diagonal blocks: SUT

j,Tj with 0 ≤ j ≤ N − 1.

###### • k-th off-diagonal blocks: SUT

with 0 ≤ j ≤ N − 1 − k

j+k,Tj

By the definition of SU, we can write SUT

j+k,Tj in a blockwise way as follows

 

 sigmoid

⊤

⊤

⊤

j+k−1

QUT

j,:KUT

V UT

i,:

i,:

SUT

j+k,Tj = QCT

+ M UT

√

√

j+k,:

j,Ti

d

d

i=j

(20)

 

 sigmoid

 

 

⊤

⊤

⊤

QCT

j+k,:V UT

QUT

j,:KUT

d ⊙ M CT

j+k,:

j+k,:

+ M UT

√

√

.

+

j+k,Tj+k

j,Tj+k

d

However, computing each SUT

j+k,Tj this way will lead to O(L3) computational cost in total. We need more efficient way to reduce training costs. The key observation in the development of this efficient parallel training algorithm is that the matrix

QCV U⊤ ⊙ M C is a low-rank matrix multiplied by a mask because QCV U⊤ has rank equal to or less

than the head dimension d rather than the sequence length L. This enables us to compute SU defined in (11) with lower computational costs. More specifically, we will rely on the following auxiliary variable to reduce computational costs.

We first init an auxiliary variable D(0) = 0d×L. Then, when computing the k-th off-diagonal blocks, we will maintain the following relation that

 sigmoid

 

⊤

⊤

j+k−1

QUT

j,:KUT

⊤

i,:

. (21)

D(:,Tk)

V UT

+ M UT

√

=

i,:

j,Ti

j

d

i=j

We will compute the blocks of SU in the following order: diagonal blocks, 1st off-diagonal blocks, 2nd off-diagonal blocks, ···, (N − 1)-th off-diagonal blocks.

Diagonal blocks of SU. For any 0 ≤ j ≤ N − 1,

 

 sigmoid

 

 

⊤

⊤

⊤

QUT

QCT

j,:KUT

j,:V UT

d ⊙ M CT

j,:

j,:

. (22)

+ M UT

SUT

√

√

j,Tj =

j,Tj

j,Tj

d

1st off-diagonal blocks of SU. For any 0 ≤ j ≤ N − 2, update D(1):,T

as

j

 sigmoid

 

⊤

⊤

j,:KUT

QUT

⊤

j,:

D(1):,T

###### = V UT

+ M UT

√

j,:

j,Tj

j

d

This satisfies (21) with k = 1. Then, it follows from (20) that

j+1,:D(1):,T √ j

QCT

SUT

j+1,Tj =

d

 

 sigmoid

 

 

⊤

⊤

⊤

QUT

j,:KUT

QCT

j+1,:V UT

d ⊙ M CT

j+1,:

j+1,:

+ M UT

√

√

+

.

j+1,Tj+1

j,Tj+1

d

The k-th off-diagonal blocks of SU. If we have already computed the (k − 1)-th off-diagonal blocks and D(k−1) satisfies (21) with k − 1, the k-th diagonal blocks can be computed in the following way. First, we update D(k) as follows

 sigmoid

 

⊤

⊤

j,:KUT

QUT

⊤

j+k−1,:

= D(:,Tk−1)

. (23)

D(:,Tk)

###### + V UT

+ M UT

√

j+k−1,:

j,Tj+k−1

j

j

d

By induction hypothesis (21), D(:,Tk)

also satisfies (21) with k. Then, it follows by (20) that

j

j+k,:D(:,Tk) √ j

QCT

SUT

j+k,Tj =

d

 

 

 sigmoid

 

(24)

⊤

⊤

⊤

QCT

j+k,:V UT

QUT

j,:KUT

d ⊙ M CT

j+k,:

j+k,:

+ M UT

√

√

+

.

j+k,Tj+k

j,Tj+k

d

We can compute all k-th off-diagonal blocks in the above way. When computing each SUT

j+k,Tj, we only need to update D(k) and then compute SUT

j+k,Tj as (24). Both (23) and (24) take O(L2d) FLOPs. Next, we describe the design of a parallel algorithm. The parallelization scheme must satisfy the following requirements:

- • SUT

j+k,Tj must be computed after SUT

j+k−1,Tj because computing SUT

j+k,Tj in (24) requires D(k) which is derived by updating D(k−1) in (23). Therefore, AT

j+k,Tj should be computed after AT

j+k−1,Tj.

- • To ensure correctness, online softmax cannot be applied simultaneously to AT

i,Tk for j ≠ k. To meet these constraints, we launch kernel instances to compute attention outputs with respect to blocks in the following order: starting with the diagonal blocks, followed by the first off-diagonal blocks, then the second, and so on up to the (N − 1)-th off-diagonal blocks. This parallel execution strategy for Algorithm 1 is illustrated in Figure 10.

i,Tj and AT

###### D.2 Backward Pass

In this section, we introduce the backward pass algorithm for efficient parallel training. It is mainly derived by reversing the forward pass.

Recall that in the forward pass, we compute the blocks in the following order: diagonal blocks, 1st off-diagonal blocks, 2nd off-diagonal blocks, ···, (N − 1)-th off-diagonal blocks. Then, in the backward pass, we compute the derives in the inverse order as follows: (N − 1)-th off-diagonal blocks, (N − 2)-th off-diagonal blocks, ···, 1-st off-diagonal blocks, diagonal blocks.

Algorithm 3: Efficient parallel backward pass Require: dO ∈ RL×d

model

- 1 # Initialization
- 2 Initialize dD = 0d×L, dQU = 0L×d, dKU = 0L×d, dV U = 0L×d, dQC = 0L×d, dKC = 0L×d, dV C = 0L×d
- 3 Let m ∈ RL×1, D ∈ Rd×L be the tensors saved in forward pass (Algorithm 1)
- 4 # Preprocess
- 5 Compute ∆ ∈ RL×1 with ∆i = dO⊤i,:Oi,:
- 6 # (N − 1)-th, ···, 1st off-diagonal blocks
- 7 for k = N − 1,...,1 (sequential) do

- 8 for j = 0,...,N − k − 1 (in parallel) do

- 9 Compute SUT

j+k,Tj as in (24) and AT

j+k,Tj as in (19)

- 10 Compute PT

j+k,Tj and dPT

j+k,Tj as in (25) and (26)

- 11 Compute dSCT

j+k,Tj and dSUT

j+k,Tj as in (28) and (29)

- 12 Update dV CT

j,:, dQCT

j+k,:, dKCT

j,: as in (27) and (30)

- 13 Update dQCT

j+k,:, dD(:,Tk)

j

, dV UT

j+k,:, dQUT

j,:, KUT

j+k,: as in (31)

- 14 Update dV UT

j+k−1,:, dQUT

j,:, dKUT

j+k−1,: as in (32)

- 15 Compute D(:,Tk−1)

j

as in (33)

- 16 end
- 17 end
- 18 # Diagonal blocks
- 19 for j = 0,...,N − 1 (in parallel) do

- 20 Compute SUT

j,Tj as in (22) and AT

j,Tj as in (19)

- 21 Compute PT

j,Tj and dPT

j,Tj as in (34) and (35)

- 22 Compute dSCT

j,Tj and dSUT

j,Tj as in (37) and (38)

- 23 Update dV CT

j,:, dQCT

j,:, dV UT

j,:, dQUT

j,:, dKUT

j,: as in (36) and (39)

- 24 end
- 25 Return the gradients dQU, dKU, dV U, dQC, dKC, dV C ∈ RL×d

As in [7], we first preprocess ∆ ∈ RL×1 with

∆i = dO⊤i,:Oi,:.

Let D(N−1) be the D saved for backward pass in Algorithm 1. Also, let m be the vector m saved for backward pass in Algorithm 1. Set dD(N−1) = 0d×L.

Then, we iterate over k = N − 1,...,1. In each iteration, we compute the corresponding gradients and update D(k) to D(k−1) inversely as in the forward pass. Thus, for each k, before we launch kernels for the k-th off-diagonal blocks, we already have D(k).

k-th off-diagonal blocks. For any 0 ≤ j ≤ N − 1 − k, we first compute SUT

j+k,Tj as in (19). Recall that in the end of the forward pass (Algorithm 1), we have set m ← m + log(ℓ). Then, the attention weights can be computed by

j+k,Tj as in (24) and AT

. (25) Compute the gradient of attention weights

j+k,Tk − mT

###### PT

j+k,Tj = exp AT

j+k

j+k,:V ⊤T

j,:. (26) Then, update the gradient of V CT

dPT

j+k,Tj = dOT

j,: as follows dV CT

j,: + P⊤T

j,: ← dV CT

j+k,:. (27) Then, compute the gradients of attention weights

j+k,TjdOT

dSCT

(28) and

j+k,Tj ⊙ dPT

j+k,Tj − ∆T

j+k,Tj = PT

j+k

j+k,Tj = −∇SiLU SUT

dSUT

. (29)

j+k,Tj ⊙ PT

j+k,Tj ⊙ dPT

j+k,Tj − ∆T

j+k

Then, we update the gradients of QCT

j+k,:, KCT

j,: through the backward pass of (19) as follows

dSCT

j+k,TjKCT √ j,:

dQCT

j+k,: ←dQCT

j+k,: +

,

d

⊤QCT √ j+k,:

dSCT

j+k,Tj

dKCT

j,: ←dKCT

j,: +

.

d

(30)

j+k,:, D(:,Tk)

Then, we update the gradients of QCT

###### , V UT

j+k,:, QUT

j,:, KUT

j+k,: through the backward pass of (24) as follows

j

⊤

j+k,TjD(:,Tk)

dSUT

dQCT

j+k,: ← dQCT

j

√

j+k,: +

d

⊤dSUT √ j+k,Tj

QCT

j+k,:

dD(:,Tk)

← dD(:,Tk)

+

j

j

d

E⊤QCT

√ j+k,: d

dV UT

j+k,: ← dV UT

j+k,: +

###### FKUT

√j+k,: d

dQUT

j,: ← dQUT

j,: +

F⊤QUT

√ j,: d

dKUT

j+k,: ← dKUT

j+k,: +

,

###### EV UT

√j+k,: d

+

(31)

where auxiliary matrices

- E =

 dSUT

j+k,Tjsigmoid

QUT

j,:KUT

j+k,:

⊤

√

d

+ M UT

j,Tj+k

  ⊙ M CT

j+k,Tj+k,

- F =

  dSUT

j+k,Tj

⊙ ∇sigmoid

 

 

⊤  

⊤

QCT

j+k,:V UT

d ⊙ M CT

j+k,:

√

j+k,Tj+k

⊤

QUT

j,:KUT

j+k,:

+ M UT

√

j,Tj+k .

d

Then, we update the gradients of V UT

j+k−1,:, QUT

j,:, KUT

j+k−1,: from the backward pass of (23) as follows dV UT

j+k−1,: ← dV UT

j+k−1,:

⊤

QUT

j,:KUT

⊤

⊤

j+k−1,:

dD(:,Tk)

+ M UT

√

+ sigmoid

j,Tj+k−1

j

d

(32)

###### GKUT

√j+k−1,: d

dQUT

j,: ← dQUT

j,: +

###### G⊤QUT

√ j,: d

dKUT

j+k−1,: ← dKUT

j+k−1,: +

where the auxiliary matrix

G = dD(:,Tk)

j

⊤

⊤ ⊙ ∇sigmoid

V UT

j+k−1,:

⊤

QUT

j,:KUT

j+k−1,:

+ M UT

√

j,Tj+k−1 .

d

After updating gradients, we set dD(:,Tk−1)

and get D(:,Tk−1)

← dD(:,Tk)

back from D(:,Tk)

by reversing (23) as follows

j

j

j

j

 sigmoid

 

⊤

⊤

j,:KUT

QUT

⊤

j+k−1,:

D(:,Tk−1)

. (33)

= D(:,Tk)

###### − V UT

+ M UT

√

j+k−1,:

j,Tj+k−1

j

j

d

Diagonal blocks. For any 0 ≤ j ≤ N − 1, we first compute SUT

j,Tj as in (19). Then, compute the attention weights

j,Tj as in (22) and AT

. (34) Compute the gradient of attention weights

j,Tj − mT

###### PT

j,Tj = exp AT

j

j,:V ⊤T

j,:. (35) Then, update the gradient of V CT

dPT

j,Tj = dOT

j,: as follows dV CT

j,: + P⊤T

j,: ← dV CT

j,:. (36) Then, compute the gradients of attention weights

j,TjdOT

dSCT

(37) and

j,Tj ⊙ dPT

j,Tj − ∆T

j,Tj = PT

j

j,Tj = −∇SiLU SUT

dSUT

j,Tj ⊙ PT

j,Tj ⊙ dPT

j,Tj − ∆T

j

. (38)

Then, we update the gradients of QCT

j,:, V UT

j,:, QUT

where auxiliary matrices

dQCT

dV UT

dQUT

dKUT

j,:, KUT

j,: through the backward pass of (22) as follows

###### EV UT

√ j,: d

j,: ← dQCT

,

j,: +

E⊤QCT

√ j,: d

j,: ← dV UT

,

j,: +

###### FKUT

√ j,: d

j,: ← dQUT

,

j,: +

F⊤QUT

√ j,: d

j,: ← dKUT

,

j,: +

(39)

- E = dSUT

j,Tjsigmoid

QUT

j,:KUT

j,:

⊤

√

d

+ M UT

j,Tj ⊙ M CT

j,Tj,

- F = dSUT

j,Tj

⊤

⊤ QCT

j,:V UT

d ⊙ M CT

j,:

√

j,Tj ⊙ ∇sigmoid

⊤

j,:KUT

QUT

j,:

+ M UT

√

j,Tj .

d

Then, the pseudo-code of backward pass is illustrated in Algorithm 3. The backward pass’s parallel scheme is naturally the reverse of the forward pass’s parallel scheme. We remark that when computing the gradients with respect to the block (Tj+k,Tj), we update both V UT

j+k,: and V UT

j+k−1,:. Consequently, the block V UT

j+k−1,: receives contributions from two sources: (Tj+k,Tj) and (Tj+k−1,Tj−1). To prevent overlapping updates, we introduce two auxiliary variables for storing intermediate results in dV U, namely d V U and d V U. Specifically, in block (Tj+k,Tj), the update to dV UT

j+k,: is accumulated in d V U, while the update to dV UT

j+k−1,: is accumulated in d V U. After all gradients with respect to off-diagonal blocks have been computed, we obtain the true gradient of dV U by dV U ← d V U + d V U. The same procedure is applide to dKU.

##### E Efficient Inference with UQ-KV Cache

Proof of (13). By (3),

Thus, for 1 ≤ s < t,

t

uts =

sigmoid

j=1

qUs kUj ⊤

+ [M Ut ]sj vUj .

√

d

uts − uts−1 = sigmoid

qUs kUt ⊤

+ [M Ut ]st vUt .

√

d

Since [M Ut ]t,t = 0, the last row of U t is all-zeros. This yields (13).

| |
|---|

Algorithm 4: Prefilling Algorithm Require: QU, KU, V U, QC, KC, V C ∈ RL×d

- 1 # Get UQ-KV Cache
- 2 Initialize U = 0L×d
- 3 for k = 0,...,N − 1 (in parallel) do

- 4 Load QUT

k,: from HBM to on-chip SRAM

- 5 for j = k,...,N − 1 (sequential) do

- 6 Load KUT

j

, V UT

j,:, U T

- j,: from HBM to on-chip SRAM

7 On chip, compute AUT

- k,Tj = sigmoid Q

- 8 On chip, U T

k,: ← U T

k,: + AUT

k,TjV UT

j,:

- 9 Write U T

k,: to HBM

- 10 end
- 11 end
- 12 Save UQ-KV cache U , QU, KC, V C ∈ RL×d
- 13 Call Algorithm 1 to get O
- 14 Return O ∈ RL×d

U Tk,:KUTj,:⊤

d + M UT

√

k,Tj

Algorithm 5: Decoding Algorithm (generating the (t + 1)-th token) Require: representation xt ∈ R1×d

model, UQ-KV cache U t−1, QUt−1, KCt−1, V Ct−1 ∈ R(t−1)×d

- 1 Compute qUt = xtW UQ, kUt = xtW UK, vUt = xtW UV
- 2 Compute qCt = xtW CQ, kCt = xtW CK, vCt = xtW CV
- 3 Update U t as in (13)
- 4 Update QUt = [QUt−1;qUt ]
- 5 Update KCt = [KCt−1;kCt ]
- 6 Update V Ct = [V Ct−1;vCt ]
- 7 Compute sCt = q

C t KCt ⊤

√

d ∈ R1×t as in (5)

- 8 Compute sUt = q

C t U t⊤

√

d ∈ R1×t as in (6)

- 9 Compute pt = softmax sCt − SiLU sUt ∈ R1×t as in (7)
- 10 Compute ot = ptV Ct ∈ R1×d as in (8)
- 11 Return ot ∈ R1×d and UQ-KV cache U t, QUt , KCt , V Ct ∈ Rt×d

