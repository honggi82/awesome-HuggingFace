## arXiv:2501.03489v2[cs.LG]8Jan2025

### ENTROPY-GUIDED ATTENTION FOR PRIVATE LLMS

Nandan Kumar Jha New York University nj2049@nyu.edu

Brandon Reagen New York University bjr5@nyu.edu

###### ABSTRACT

The pervasiveness of proprietary language models has raised critical privacy concerns, necessitating advancements in private inference (PI), where computations are performed directly on encrypted data without revealing users’ sensitive information. While PI offers a promising solution, its practical deployment is hindered by substantial communication and latency overheads, primarily stemming from nonlinear operations. To address this, we introduce an information-theoretic framework to characterize the role of nonlinearities in decoder-only language models, laying a principled foundation for optimizing transformer-architectures tailored to the demands of PI.

By leveraging Shannon’s entropy as a quantitative measure, we uncover the previously unexplored dual significance of nonlinearities: beyond ensuring training stability, they are crucial for maintaining attention head diversity. Specifically, we find that their removal triggers two critical failure modes: entropy collapse in deeper layers that destabilizes training, and entropic overload in earlier layers that leads to under-utilization of Multi-Head Attention’s (MHA) representational capacity.

We propose an entropy-guided attention mechanism paired with a novel entropy regularization technique to mitigate entropic overload. Additionally, we explore PI-friendly alternatives to layer normalization for preventing entropy collapse and stabilizing the training of LLMs with reducednonlinearities. Our study bridges the gap between information theory and architectural design, establishing entropy dynamics as a principled guide for developing efficient PI architectures. The code and implementation are available at entropy-guided-llm.

###### 1 Introduction

The widespread deployment of proprietary large language models (LLMs) has raised critical privacy concerns for users’ sensitive information [1, 2, 3, 4]. Private Inference (PI) offers a promising solution, enabling computations directly on encrypted data without exposing its contents.

However, despite its potential, the practical deployment of PI systems remains a significant challenge due to substantial latency and communication overheads, particularly for transformer-based LLMs. Generating a single output token with a GPT-2 model (125M parameters) over 128 input tokens takes 8.2 minutes and requires 25.3 GB of communication (see Table 3). Scaling this to a context size of 512 results in 30.7 minutes and 145.2 GB of communication (Table 4).

These inefficiencies primarily arise from the computational overhead of nonlinear operations, which are critical for model stability and performance. Nonlinear computations in privacy-preserving settings require secure multi-party computation (MPC) protocols and cryptographic primitives such as secure comparisons, oblivious transfer, and polynomial evaluations (e.g., for GELU [5]). These protocols involve multiple interaction rounds between users and service providers, significantly increasing communication and computational costs.

For instance, a single GELU activation in a BERT-base model requires 3.9×106 point-wise operations, each involving multiple secure multiplications and communication rounds, typically adding 1-2 KB per operation [6]. Recent work [7] has shown that nonlinear operations, primarily GELU and LayerNorm [8], constitute the major bottleneck in PI, accounting for 49% of latency and 59% of communication costs.

Designing LLMs with reduced-nonlinearities is a promising direction for efficient PI architectures. However, the fundamental role of nonlinearities in preserving transformer expressiveness and regulating internal information flow remains poorly understood. For instance, Li et al. [9] offer a theoretical analysis of attention and feed-forward network

(FFN) nonlinearities in in-context learning tasks, which is limited to a simplified setting: a one-layer model with a single softmax-based self-attention head and a ReLU-based FFN. While Cheng et al. [10] extend this investigation by analyzing a broader range of nonlinear architectures, they remain focused on specific in-context learning tasks.

These findings, while valuable, do not adequately address the comprehensive role of nonlinearities in maintaining model stability, and fostering attention head diversity in practical multi-layer LLMs or their implications for PI. Recent studies have shown an increasing focus on understanding the failure modes in transformer models, such as training instability [11, 12, 13] and rank collapse [14, 15, 16]. However, they predominantly focus on standard transformer architecture, leaving a critical question unaddressed: How do the removal of non-linearities, impact training dynamics?

To bridge this gap, we propose an information-theoretic framework to systematically analyze the role of nonlinearities in transformer-based models. Using Shannon’s entropy as a quantitative lens, we uncover the dual significance of nonlinearities: (1) they ensure training stability by preventing entropy collapse in deeper layers, and (2) they preserve the representational diversity of MHA by mitigating entropic overload in earlier layers, fostering head-wise specialization.

Our Contributions: Building on these insights, our work makes the following contributions:

- 1. PI-friendly layer normalization alternatives: To address training instability in LLM with reduced-nonlinearities, without relying on LayerNorm, we study the static normalization techniques such as weight and spectral normalization techniques [17, 18]. These methods mitigate entropy collapse in deeper layers while avoiding the overheads associated with nonlinear operations in LayerNorm.
- 2. Entropy regularization techniques: We introduce an entropy-guided attention mechanism and propose a novel entropy regularization technique to prevent entropic overload in LLMs with reduced-nonlinearities. Our approach incorporates two key innovations: (a) Headwise learnable thresholds to dynamically adjust regularization strength for each attention head, tailoring the process to the specific characteristic of individual heads; and (2) Tolerance margins to prevent over-regularization, preserving attention head diversity while preventing excessive penalization.
- 3. Practical design for PI: We implement the entropy-guided framework and demonstrate their effectiveness across various context sizes (128, 256, 512) and model depths (12L and 18L) on a wide range of training tokens (1.2B to 4.8B) from the CodeParrot [19] and Languini dataset [20] on GPT-2 models.

By analyzing entropy dynamics across layers, we provide a principled understanding of how architectural simplifications, such as removing nonlinearities, affect training stability and the representational diversity of attention heads in MHA. Our study establishes entropy dynamics as a foundational framework for optimizing privacy-preserving LLM architectures.

###### 2 Preliminaries

Notations. We denote the number of layers as L, number of heads as H, model dimensionality as d, head dimension as dk (where dk = Hd ), and context length as T. Table 1 illustrates the abbreviations for architectural configurations with simplified nonlinearities in a transformer-based LLM.

An overview of transformer-based decoder-only architecture. A transformer-based LLM is constructed by sequentially stacking L transformer blocks, where each block is composed of two sub-blocks: an attention mechanism and a feed-forward network (FFN), both having their own residual connections and normalization layers, positioned in the Pre-LN order to improves training stability [21]. Formally, transformer blocks take an input sequence Xin ∈ RT×d, consisting of T tokens of dimension d, and transform it into Xout as follows:

Xout = Xˆ SA + FFNGELU(LayerNorm2(Xˆ SA)), where Xˆ SA = Xin + MHA(LayerNorm1(Xin)). (1)

The Multi-Head Attention (MHA) sub-block enables input contextualization by sharing information between individual tokens. MHA employs the self-attention mechanism to compute the similarity score of each token with respect to all other tokens in the sequence, and transform the input sequence X into Attn(X) as follows:

1 √dk

###### (XWQ)(XWK)⊤ + M XWV. (2)

Attn(X) = Softmax

Here, each token generates query(Q), key(K), and value(V) vectors through the linear transformations WQ, WK, and WV ∈ Rd×dh, respectively. Then, similarity scores are computed by taking the dot product of the Q and K vectors, scaled by the inverse square root of the K dimension, and passed through a softmax function to obtain the attention weights. These weights are then used to compute a weighted sum of the V vectors, producing the output for each token. For auto-regressive models (e.g., GPT), mask M ∈ RT×T, which has values in {0, −∞} with Mi,j = 0iffi ≥ j, is deployed to prevent the tokens from obtaining information from future tokens.

The MHA sub-block employs a self-attention mechanism across all the heads, each with its own sets of Q, K, and V. This allows the attention heads to focus on different parts of the input sequence, capturing various aspects of the input

data simultaneously. The outputs from all heads are concatenated and linearly transformed (WO ∈ Rd×d) to produce the final MHA output as follows:

MHA(X) = Concat Attn1(X), Attn2(X), Attn3(X), . . . ,AttnH(X) WO. (3)

Following the MHA sub-block, the FFN sub-block transforms each token independently. The FFN sub-blocks have a single hidden layer whose dimension is a multiple of d (e.g., 4d in GPT [22] models). The FFN sub-block first applies a

linear transformation to the input X using Wffnin ∈ Rd×4d, followed by a non-linear transformation using an activation function such as GELU. This is then followed by another linear transformation using Wffnout ∈ R4d×d, as follows:

FFN(X) = (GELU(XWffnin ))Wffnout (4)

Threat model for private inference. We consider the standard two-party (2PC) client-server setting used in PPML, which provides security against semi-honest (honest-but-curious) adversaries bounded by probabilistic polynomial time [23, 6, 24, 7]. Both parties follow protocol specifications but may attempt to gain additional information from their outputs about the other party’s input. In this 2PC setting, the server holds the propriety LLM (e.g., ChatGPT), and the client queries the model with a piece of text (prompt). The protocols ensure the server learns nothing about the client’s input or query output, and the client learns nothing beyond the the server’s model architecture.

Server

Positional encoding

Head

Proprietary model

Outputs (Next Words)

Private Inference (PI)

| | |
|---|---|
|Embedding| |

Tokenization

FFN

Input Output

PI Protocols

MHA

Input prompt

Client

Figure 1: An illustration of threat model and cryptographic protocols used for LLM private inference.

Table 1: Architectural configurations of nonlinearities in LLMs, illustrating the combinations of Softmax (SM), LayerNorm (LN), GELU (G), and ReLU (R) functions (see Eq. 1, 2, 3 and 4).

Abbreviation Architectural configuration

SM + LN + G Xout = FFNGELU(LayerNorm2(MHA(AttnSoftmax(LayerNorm1(Xin))))) SM + LN + R Xout = FFNReLU(LayerNorm2(MHA(AttnSoftmax(LayerNorm1(Xin))))) SM + LN Xout = FFNIdentity(LayerNorm2(MHA(AttnSoftmax(LayerNorm1(Xin))))) SM + G Xout = FFNGELU(MHA(AttnSoftmax(Xin))) SM + R Xout = FFNReLU(MHA(AttnSoftmax(Xin))) SM Xout = FFNIdentity(MHA(AttnSoftmax(Xin)))

###### 3 Information-Theoretic Analysis of Nonlinearity in LLMs

In this section, we systematically decouple nonlinearities from transformer-based decoder-only LLMs, investigating their impact on training dynamics and expressiveness of attention mechanism, through the lens of Shannon’s entropy.

Shannon’s entropy for quantifying attention score distribution Shannon’s entropy quantifies the uncertainty in a probability distribution, measuring the amount of information needed to describe the state of a stochastic system [25, 26]. For a probability distribution P(x), the entropy is defined as E(P) = − ∑i P(xi) log P(xi).

In a softmax-based attention mechanism, each softmax operation yields an entropy value representing the sharpness or spread of the attention scores for each query position [27, 28]. Higher entropy indicates a more uniform distribution of softmax scores, while lower entropy signifies a more focused distribution on certain features [29].

Let A(h,l) ∈ RT×T be the attention matrix of h-th head in l-th layer, and each element in the attention matrix, aij(l,h), are attention weights for the i-th query and j-th key, which are non-negative and sum to one for a query:

T

A(l,h) = aij(l,h)

, where aij(l,h) ≥ 0 and

aij(l,h) = 1 (5)

##### ∑

T×T

j=1

This square matrix is generated by applying the softmax operation over the key length for each query position as follows

1 √dk

exp (xi)

A(h,l)(X) = Softmax

(XWQ)(XWK)⊤ , where Softmax(Xi) =

(6)

∑Tj=1 exp xj

Following [13], we compute the mean of entropy values across all query positions to obtain a single entropy value for each head. The entropy E(l,h) for the h-th head in the l-th layer of an attention matrix is given by:

exp √ 1

###### (XiWQ)(XjWK)⊤

T

T

1 T

dk

aij(l,h) log aij(l,h) + ϵ , where aij(l,h) =

E(l,h) = −

##### ∑

##### ∑

(7)

∑kT=1 exp √ 1

(XiWQ)(XkWK)⊤

i=1

j=1

dk

where ϵ is a small constant added for numerical stability to prevent taking the log of zero.

Well-behaved entropy distribution for LLMs We begin by analyzing the headwise entropy distribution of baseline architecture with GELU (SM + LN + G) and ReLU (SM + LN + R) in their FFN. We find that the majority of heads

(≈90%) possess entropy values between max4 and 3max4 , where max is maximum observed entropy value among all heads (see Figure 2a). This concentration in the mid-entropy range, while avoiding extremes, demonstrates a well-behaved

distribution, providing a benchmark for assessing the impact of nonlinearities on model behavior.

Entropic overload in nonlinearity-reduced LLMs We observed that in certain nonlinearity configurations, a disproportionately large fraction of the attention heads exhibit higher entropy values (between 3max4 and max), and we term this phenomenon as entropic overload. We hypothesize that this deviation form well-behaved entropy distribution results in under-utilization of the network’s representational capacity, as too many heads engaged in exploration, hindering the model from effectively leveraging the diversity of attention heads.

4.0

EvalCross-entropyLoss

74%

74%

SM + LN + G SM + LN + R SM + LN SM + G SM + R SM

SM+LN+G SM+LN+R SM+LN

70%

Configurations PPL +∆(%)

3.5

| |
|---|

| |
|---|

PercentageofHeads

60%

| |
|---|

SM + LN + G 2.69 0.00 SM + LN + R 2.76 2.53 SM + LN 3.38 25.58 SM + G 3.20 18.92 SM + R 2.94 9.20 SM NaNs -

58%

3.0

54%

SM+G SM+R SM

| |
|---|

50%

48%

| |
|---|

2.5

45%

| |
|---|

| |
|---|

40%

37%

| |
|---|

2.0

33%

| |
|---|

30%

| |
|---|

| | |
|---|---|
| | |

1.5

23%

23%

| | |
|---|---|
| | |

20%

20%

| |
|---|

18%

| | |
|---|---|

| |
|---|

16%

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

14%

| | |
|---|---|

| |
|---|

| | |
|---|---|

| | |
|---|---|

12%

| | |
|---|---|

| | |
|---|---|

1.0

| |
|---|

| | |
|---|---|

| | |
|---|---|

10%

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

10%

| |
|---|

| | |
|---|---|

| | |
|---|---|

| | |
|---|---|

8% 6%

8%

| | |
|---|---|

| | |
|---|---|

| |
|---|

| | |
|---|---|

7% 1%

Table 2: Evaluation perplexity for GPT-2 (small) models with reduced-nonlinearities, corresponding to Figure 2b. ∆ is increase in eval PPL over baseline network.

0.0 0.2 0.4 0.6 0.8 1.0 Epoch

4%

3%

3%

0%

[0, Max4 ) [Max4 , Max2 ) [Max2 , 3Max4 ) [3Max4 , Max]

(a) Headwise entropy distribution

(b) Loss curve

- Figure 2: (a) The fraction of attention heads distributed across different entropy ranges, and (b) evaluation loss for GPT-2 (small) models with reduced-nonlinearities, when trained from scratch on CodeParrot dataset.

We visualize the entropy heatmaps for LLM architectures with reduced nonlinearity, trained from scratch (Figure 3). Our analysis reveals severe entropic overload in the early layers of two specific architectures: the LayerNorm-free model with GELU (Figure 3d) and the Softmax-only model without LayerNorm and FFN activations (Figure 3f).

Specifically, 58% of heads in the LayerNorm-free GELU model have entropy values in the range 3max4 and max, compared to only 23% in the LayerNorm-free ReLU model (Figure 2a). Additionally, very few heads in the latter

approach maximum entropy, unlike their GELU counterpart (see yellow regions in Figure 3e and Figure 3d), which results in 8.2% improvement in perplexity (see Table 2). On the other hand, the 45% of heads in Softmax-only model

have entropy values in the 3max4 to max range, with many approaching the maximum (Figure 3f).

Entropy collapse in nonlinearity-reduced LLMs The absence of LayerNorm and FFN nonlinearity in Softmax-only model leads to entropy collapse in the deeper layers—a phenomenon characterized by near-zero entropy values and recognized as a key indicator of training instability in transformer architectures [13, 30]. Quantitatively, 33% of attention

heads demonstrate entropy values within the range of 0 to max4 (Figure 2a), with a significant concentration approaching zero (see Figure 3f). This systematic entropy collapse directly contributes to training instability, highlighting the critical

role of nonlinear components in maintaining stable training dynamics.

###### 4 Entropy-Guided LLM Architecture for Efficient Private Inference

We begin by exploring PI-friendly techniques to prevent entropy collapse in the absence of LayerNorm and FFN activations. Subsequently, we introduce an entropy-guided attention mechanism paired with an entropy regularization technique to mitigate entropic overload in nonlinearity-reduced LLMs.

- 2.16 2.91 3.83 3.33 3.55 3.12 3.00 3.20 2.68 2.87 3.04 3.04
- 3.46 0.77 0.73 1.00 0.93 1.13 0.95 1.48 1.38 1.56 1.10 0.61

- 1.50 2.46 3.39 2.86 3.67 2.80 2.62 1.66 2.05 2.57 2.77 1.77 3.35 0.59 0.88 0.96 0.78 1.50 0.93 1.53 0.97 2.03 0.60 0.57

- 1.70 1.86 1.63 1.03 1.48 0.84 3.27 2.01 1.62 2.06 1.46 2.15
- 2.48 2.27 2.37 2.10 1.82 2.26 1.73 1.82 1.92 2.14 2.53 2.36

- 2.65 2.45 2.35 2.91 2.30 2.43 2.53 1.02 1.99 2.75 2.55 2.60 2.41 2.74 2.56 1.70 1.89 2.27 2.58 2.20 2.53 2.24 2.58 1.38 2.33 2.37 2.63 2.38 1.75 2.45 2.44 2.72 2.46 2.51 2.62 2.26 2.52 2.63 2.21 2.25 2.51 2.52 2.44 2.38 1.58 2.22 2.15 1.76 2.59 2.45 2.19 2.69 2.20 2.30 2.55 2.54 2.71 2.62 2.52 2.55 2.83 2.69 2.65 2.62 2.65 2.60 2.49 2.57 2.50 2.50 2.53 2.54 2.63 2.55 2.75 2.65 2.66 2.40 2.78 2.70 2.42 2.50 2.55 2.57 2.65 2.60 2.80 2.73 2.59 2.67 2.65 2.50 2.62 2.67 2.57 2.62

[Figure 1]

2.49 3.04 3.72 3.09 2.98 3.05 3.28 2.88 2.97 3.57 3.16 3.03 2.79 0.40 0.93 0.67 0.71 1.37 0.53 0.87 0.96 0.04 2.25 0.99

01234567891011 Layerindex

01234567891011 Layerindex

3.5

01234567891011

3.0

- 1.64 1.92 1.71 1.01 1.57 1.06 2.11 1.43 1.70 1.83 1.35 2.40 3.18 2.33 2.04 2.31 2.45 2.32 2.20 1.94 2.44 2.40 2.84 2.17
- 2.58 2.52 2.27 1.88 2.24 2.61 2.63 2.14 2.50 3.07 2.41 2.76 2.09 0.64 2.71 1.39 2.17 2.30 2.77 1.64 2.46 2.43 2.77 1.78 2.38 2.72 2.50 2.27 2.55 2.37 2.34 2.79 2.57 2.35 2.78 2.37 2.56 2.79 2.31 2.60 2.51 2.51 2.58 2.49 2.33 2.60 2.57 1.98 2.72 2.43 2.30 2.61 2.26 2.36 2.65 2.72 2.70 2.62 2.84 2.65 2.78 2.73 2.66 2.68 2.67 2.55 2.70 2.66 2.62 2.51 2.71 2.81 2.69 2.72 2.90 2.71 2.70 2.60 2.83 2.71 2.57 2.48 2.57 2.71 2.75 2.75 2.69 2.88 2.70 2.66 2.73 2.60 2.70 2.68 2.62 2.62

- 1.31 2.22 2.41 1.15 1.91 0.84 1.69 0.92 1.81 1.41 1.22 0.92

- 1.73 1.44 1.84 1.66 1.33 1.48 1.52 0.94 1.48 1.45 1.12 1.43

- 1.77 1.96 1.64 1.19 2.31 1.85 1.75 1.62 1.72 1.61 1.77 1.99

- 1.41 2.04 1.78 1.49 1.69 2.01 1.92 1.68 1.84 2.06 2.19 1.53

- 1.94 1.94 2.10 1.77 2.03 1.76 1.65 2.32 2.15 1.83 1.82 1.80
- 2.15 2.41 2.52 1.94 2.09 1.73 1.85 2.26 2.23 1.99 1.84 1.47

- 2.14 2.50 1.69 1.79 2.24 1.95 2.25 2.36 2.74 1.77 1.84 2.05

- 2.03 2.06 2.03 2.28 2.12 2.15 2.11 2.18 2.17 2.18 2.05 2.06

- 2.12 2.22 2.21 2.25 2.51 2.11 2.27 2.20 2.17 2.26 2.11 2.29

- 2.16 2.50 2.19 2.05 2.31 2.24 1.82 2.24 2.26 2.22 2.12 2.16

2.5

Layerindex

2.0

1.5

1.0

0.5

0 1 2 3 4 5 6 7 8 9 10 11 Head index

0 1 2 3 4 5 6 7 8 9 10 11 Head index

0 1 2 3 4 5 6 7 8 9 10 11

Head index

(a) SM + LN + G

(b) SM + LN + R

(c) SM + LN

- 2.72 3.45 3.20 2.85 3.55 3.53 3.37 2.31 3.73 2.58 3.32 3.36
- 3.61 2.56 3.40 3.33 3.19 3.42 2.96 3.23 3.21 3.34 3.37 3.29

- 2.70 2.30 2.65 3.77 3.87 2.78 2.66 3.01 3.24 1.94 2.67 2.60

- 2.04 2.91 2.01 2.22 2.31 3.40 3.26 2.12 2.44 2.11 3.04 3.05
- 3.20 2.40 2.89 1.74 2.75 2.84 3.50 3.29 3.03 2.84 2.91 2.53

- 3.58 2.14 3.53 2.84 2.96 3.21 2.01 2.74 3.42 2.65 3.17 2.51 2.04 2.68 3.25 3.15 3.66 2.46 2.73 3.23 2.91 3.81 3.14 3.12 2.38 2.17 2.21 2.46 3.53 2.75 3.12 2.26 2.64 2.90 2.92 2.64

[Figure 2]

3.57 3.63 3.67 3.55 3.65 3.65 3.59 3.41 3.75 3.47 3.54 3.56 3.84 3.85 3.85 3.83 3.84 3.83 3.85 3.85 3.86 3.86 3.83 3.87 3.82 3.77 3.86 3.74 3.75 3.85 3.85 3.83 3.86 3.86 3.83 3.83 3.69 3.76 3.75 3.83 3.78 3.83 3.83 3.84 3.75 3.83 3.73 3.82 3.69 3.81 3.73 3.81 3.80 3.75 3.85 3.66 3.70 3.48 3.83 3.78 2.73 2.22 2.59 2.86 3.41 3.06 3.27 1.19 3.02 2.40 1.31 3.15 2.19 1.77 1.64 2.55 2.38 1.90 1.84 2.52 2.07 2.31 1.57 2.80 1.88 2.12 2.68 2.90 1.09 2.87 1.61 1.33 2.07 2.09 1.73 1.32 0.99 0.36 0.54 0.50 0.78 0.33 0.22 0.36 0.36 0.49 0.44 0.88 0.15 0.25 0.27 0.25 0.17 0.17 0.18 0.15 0.17 0.19 0.12 0.14 0.06 0.15 0.07 0.06 0.03 0.06 0.05 0.04 0.08 0.05 0.05 0.05 0.04 0.04 0.05 0.04 0.05 0.04 0.05 0.05 0.05 0.04 0.05 0.04

01234567891011 Layerindex

01234567891011

01234567891011 Layerindex

3.5

- 3.24 3.57 3.27 3.66 3.39 3.06 2.74 3.46 3.48 3.57 2.67 2.67

- 2.71 3.53 3.38 3.33 3.68 3.08 3.22 3.36 3.38 3.29 3.55 3.60
- 3.40 3.62 3.69 3.34 3.55 3.52 3.59 3.63 3.49 2.99 3.69 3.45

- 3.25 3.68 3.69 3.22 3.57 3.59 3.60 3.67 3.53 3.33 3.40 3.44

3.0

2.5

Layerindex

2.0

- 1.40 3.48 3.66 3.69 3.80 3.66 3.20 2.87 3.77 3.70 3.18 3.48 3.19 2.98 3.17 2.92 3.24 3.47 1.95 2.34 2.94 2.56 2.94 3.17
- 2.92 2.73 1.45 2.05 1.28 1.37 1.60 2.06 1.69 1.82 1.36 1.36 1.40 1.76 2.08 1.39 1.31 2.90 0.95 2.30 2.81 2.30 2.33 1.02 1.74 2.83 1.98 2.15 0.16 2.41 1.94 2.16 1.78 2.14 1.70 2.53 1.93 1.74 1.59 2.23 1.70 1.60 1.78 1.31 2.80 1.82 2.13 1.84

- 1.78 1.71 2.27 1.70 2.93 2.90 1.38 2.21 2.90 1.89 1.33 1.83
- 2.14 1.72 1.97 2.15 1.61 1.59 2.75 1.44 1.92 2.27 1.88 2.01 2.18 1.80 1.45 0.34 0.99 0.78 1.45 2.00 1.72 1.50 1.65 2.21 2.43 2.25 2.23 2.01 2.06 1.88 2.20 2.73 2.02 2.16 1.81 1.98 2.43 1.21 2.35 2.12 2.14 0.91 2.00 1.56 1.92 1.72 0.75 2.10 2.66 1.93 2.19 2.40 2.39 2.56 2.21 2.34 2.55 2.41 2.25 1.82

1.5

1.0

0.5

0 1 2 3 4 5 6 7 8 9 10 11 Head index

0 1 2 3 4 5 6 7 8 9 10 11 Head index

0 1 2 3 4 5 6 7 8 9 10 11

Head index

(d) SM + G

(e) SM + R

(f) SM

- Figure 3: Headwise entropy distribution in LLM architectures with reduced nonlinearities compared to baseline models. Yellow regions indicate high-entropy concentrations, revealing severe entropic overload predominantly in early layers.

PI-friendly layer normalization alternatives To address training instability, prior work has predominantly relied on LayerNorm applied to various parts of the network, such as QK-LayerNorm [31, 11, 32] and FFN-LayerNorm [12]. Since LayerNorm requires expensive inverse-square-root operations during inference [7], we shift our focus from activation normalization to weight normalization techniques that avoid nonlinear computations at inference.

We discover that the weight normalization [17] and spectral normalization [18] serves as static alternatives to LayerNorm by normalizing weights instead of activations. These methods incur no additional cost at inference, and effectively prevent entropy collapse in the deeper layers of LLMs, in the absence of LayerNorm and FFN activations (see Figure 5). Notably, the effectiveness of weight and spectral normalization depends on targeting the appropriate linear layers, as applying them in attention sub-block diminishes overall performance compared to when applied in FFN (see Table 5).

Furthermore, we employ a simpler technique to scale the outputs of the FFN sub-block by having learnable scaling factors for the FFN output and their residual output as follows (see Eq. 1):

1 α

Xout = βXˆ SA +

(FFNSM(XSA)) where α, β ∈ RL (8) Architectural simplifications and entropy-guided attention

We simplified the LLM architecture by designing a Softmaxonly model that eliminates LayerNorm and FFN nonlinearity. Subsequently, we merge the two linear layers in the FFN—

###### FFNMHA

Wffnin ∈ Rd×4d and Wffnout ∈ R4d×d—into a single linear layer Wffn ∈ Rd×d (see Figure 4), as they perform equivalent linear transformations in the absence of intervening nonlinearity. However, training this simplified LLM presents challenges, particularly entropy collapse in deeper layers. To address this, we incorporate FFN scaling method that employ learnable scaling factors α and β in the FFN sub-block. This approach stabilizes training more effectively than weight or spectral normalization, achieving lower perplexity (Table 6). We denote this simplified model as SM+ScFuFFN (Figure 4).

###### FFNMHA

| | |
|---|---|
|GELU| |

###### FFNMHA

| | |
|---|---|
| | |

Architectural Simplifications

Entropy-Guided Attention

|LayerNorm|
|---|

Softmax

Mask

Mask

Softmax

Mask

###### Entropy of attention weights

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

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

| | |
|---|---|
|LayerNorm| |

To preserve attention head diversity in our simplified architecture, we develop an entropy-guided attention mechanism. Inspired by [33], which employed temperature as a Lagrangian multiplier to control stochastic system entropy, we augment

Embedding

| | |
|---|---|
|Inputs| |

Figure 4: Nonlinearity-reduced simplified architecture with entropy-guided attention mechanism.

SM+ScFuFFN with learnable temperatures for each softmax operation (t ∈ RH×T). This allows the model to dynamically adjust entropy patterns during training by adjusting the temperature. Specifically, higher temperature values (t > 1) diffuse attention scores and increase entropy, while lower values (t < 1) sharpen attention scores and reduce entropy (see Appendix A). We refer this simplified architecture with entropy-guided attention as SM(t)+ScFuFFN.

Design principles for entropy regularization schemes to prevent entropic overload. Prior entropy regularization approaches have primarily aimed at penalizing low-entropy predictions [34, 35], based on the principle of maximum entropy [36]. However, our goal is to regularize higher entropy values, which presents two-fold challenges: (1) Since each attention head captures different aspects of the input, the regularization strength needs to be adjusted for each head individually. (2) Some heads naturally exhibit higher entropy even in well-behaved entropy distributions, thus, penalizing all high-entropy values without distinction could be harmful, requiring a more flexible approach.

Followings are the key design principles for our entropy regularization scheme (see Algorithm 1):

- • Dynamic thresholds with head-specific adaptation: To adapt the regularization strength based on the characteristics of each attention head [37], we use headwise learnable threshold parameter reg_threshold_weights ∈ RH. Consequently, the threshold for each head is computed as a learnable fraction of the maximum value of entropy (reg_threshold_weights × Emax), providing the fine-grained control (see Algorithm 1, line #11).
- • Tolerance margin to prevent over-regularization: To prevent over-regularization, we allow small deviations from the respective thresholds. Thus, a penalty is imposed only if the deviation from the threshold exceeds the tolerance margin, which is set as a fraction of Emax using the hyper-parameter γ (see Algorithm1, line #3).

penalty(l,h) =

 



deviation(l,h)

2

if deviation(l,h) > γEmax 0 otherwise

The deviation from threshold is computed as deviation(l,h) = E(l,h)(t) − θ(l,h)Emax, where θ(l,h) is reg_threshold_weights. The hyper-parameter γ ensures that the model is not excessively penalized for minor deviations from the desired entropy threshold, which could impede its capacity to learn effectively. This careful calibration between stringent regularization and desired flexibility improves the model’s robustness while maintaining its adaptability to various input distributions.

- • Maximum entropy reference: We set Emax = log(T) as a reference point for computing thresholds and tolerance margins to ensure consistency across different layers and heads for regularization. Specifically, it provides a quantifiable reference for measuring deviations in entropy, making the regularization process more understandable.

Algorithm 1 Entropy Regularization Loss Computation Inputs: attentions: List of attention matrices, Θ(L, H)= reg_threshold_weights, T: Sequence length, λ: Regularization loss weightage, γ: Hyper-parameter for Tolerance margin Output: Ltotal: Total loss including entropy regularization

- 1: Lentropy ← 0
- 2: Emax ← log(T) ▷ Theoretical maximum value of entropy
- 3: Tolmargin ← γEmax ▷ Tolerance margin is set as a small fraction of Emax
- 4: for each layer l in layers do
- 5: Llayer ← 0
- 6: A(t) ← attentions[l] ▷ Attention matrix with learnable temperature for each query position
- 7: E(t) ← −T1 ∑iT=1 ∑Tj=1 Aij(t) log(Aij(t)) ▷ Compute entropy, averaged over query length

- 8: for each head h in heads do
- 9: E(l,h) ← Slice(E(t), h) ▷ Entropy for head h
- 10: θ(l,h) ← Slice(Θ(L, H), h) ▷ Learnable threshold weight head h
- 11: δ(l,h) ← E(l,h)(t) − θ(l,h)Emax ▷ Deviation from head-specific threshold
- 12: penalty(l,h) ← (δ(l,h))21(|δ(l,h)| > Tolmargin) ▷ Penalize iff deviation exceeds Tolerance
- 13: Llayer ← Llayer + penalty(l,h)
- 14: end for
- 15: Llayer ← num_headsLlayer ▷ Average over heads

- 16: Lentropy ← Lentropy + Llayer
- 17: end for
- 18: Lentropy ← len(attentions)Lentropy ▷ Average over layers

- 19: Ltotal ← LCE + λLentropy
- 20: return Ltotal

###### 5 Experimental Results

System setup We use a SecretFlow setup [6] with the client and server simulated on two physically separate machines, each equipped with an AMD EPYC 7502 server with specifications of 2.5 GHz, 32 cores, and 256 GB RAM. We measure the end-to-end PI latency, including input embeddings and final output (vocabulary projection) layers, in WAN setting (bandwidth:100Mbps, latency:80ms), simulated using Linux Traffic Control (tc) commands. The number of threads is set to 32. Following [38, 20, 39], all the models are trained on a single RTX 3090 GPU.

Models and datasets We train GPT-2 (12 and 18 layers) models on the CodeParrot [19] and Languini book [20] datasets, which are standard benchmarks for LLMs [38, 30]. The CodeParrot dataset, sourced from 20 million Python files on GitHub, contains 8 GB of files with 16.7 million examples, each with 128 tokens, totaling 2.1 billion training tokens. We use a tokenizer with a vocabulary of 50K and train with context lengths of 128 and 256. The Languini book dataset includes 84.5 GB of text from 158,577 books, totaling 23.9 billion tokens with a WikiText-trained vocabulary of 16,384, and train with context length of 512. Each book averages 559 KB of text or about 150K tokens, with a median size of 476 KB or 128K tokens.

Training Hyperparameters For pre-training on the CodeParrot dataset, we adopt the training settings from [38]. Similarly, for training on the Languini dataset, we follow the settings from [20]. These settings remain consistent across all architectural variations to accurately reflect the impact of the architectural changes. When applying entropy regularization on the CodeParrot dataset, we initialize the learnable temperature to 1e-2 and set λ to 1e-5. For the Languini dataset, the temperature is initialized to 1e-1, and λ is set to 5e-5.

Entropy regularization prevents entropic overload in Softmax-only models While both weight and spectral normalization, and scaling methods effectively prevent entropy collapse in the deeper layers and stabilize the training of Softmax-only models, they fail to address the issue of entropic overload, (see Figures 5c and 5d). In contrast, the entropy regularization scheme penalizes the model to avoid extreme entropy values during training, resulting in a more balanced distribution. As a result, it complements the training stabilizing methods by further mitigating entropic overload in the early layers (see Figure 5f), improving the utilization of attention heads and leading to improved performance, as demonstrated by lower perplexity (see Table 3).

4.0

4.0

4.0

LayerwiseMeanEntropy

LayerwiseMeanEntropy

LayerwiseMeanEntropy

3.5

3.5

3.5

- Layer0
- Layer1

3.0

3.0

3.0

2.5

2.5

2.0

2.5

1.5

L0 L1 L2

L3 L4 L5

L6 L7 L8

L9

2.0

- L10

- L11

2.0

1.0

1.5

0.5

1.5

1.0

0.0

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

(a) SM + LN + G

(b) SM

(c) SM + WeightNormalization(FFN)

4.0

4.0

LayerwiseMeanEntropy

LayerwiseMeanEntropy

LayerwiseMeanEntropy

3.5

3.5

3.5

3.0

- 2.5

- 3.0

3.0

2.5

2.5

2.0

2.0

1.5

2.0

1.5

1.0

1.5

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

(d) SM + SpectralNormalization(FFN)

(e) SM + Scaled(FFN)

(f) EntropyReg(SM(t)+ScFuFFN)

- Figure 5: Layerwise entropy patterns in GPT-2 models (L = 12, H = 12, d = 768) trained from scratch on CodeParrot dataset. Shown are (a) baseline model, (b) Softmax-only model without normalization, and variants with (c) weight normalization, (d) spectral normalization, and (e) scaled-FFN. While these normalization methods prevent entropy collapse, they fail to address entropic overload in early layers. Our final configuration (f) incorporates entropy regularization within scaled-FFN to effectively manage both issues.

Significance of learnable thresholds in entropy regularization Figure 6 depicts the learnable threshold parameters (reg_threshold_weights) applied in the entropy regularization scheme after the model has been fully trained from scratch. They exhibit significant variability, both across layers and within individual heads of each layers, which reflects

the model’s ability to dynamically adjust the regularization strength in response to the specific roles of different attention heads. Such flexibility is essential for tailoring the regularization process to the distinct requirements of each head.

×10 3

[Figure 3]

0.43 0.44 0.42 0.45 0.44 0.49 0.43 0.46 0.43 0.43 0.43 0.40 0.42 0.36 0.35 0.38 0.41 0.39 0.31 0.36 0.43 0.38 0.39 0.46 0.33 0.52 0.34 0.41 0.33 0.35 0.38 0.40 0.43 0.37 0.34 0.42 0.31 0.32 0.31 0.31 0.34 0.45 0.40 0.35 0.29 0.37 0.32 0.52

0.50

01234567891011

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

- 0
- 1
- 2
- 3
- 4

Layerwisevariance

0.45

Layerindex

- 0.30 0.29 0.30 0.28 0.27 0.32 0.30 0.32 0.34 0.33 0.34 0.42 0.33 0.28 0.36 0.35 0.32 0.26 0.37 0.39 0.30 0.34 0.47 0.36
- 0.31 0.33 0.26 0.30 0.20 0.28 0.32 0.30 0.31 0.31 0.26 0.34
- 0.32 0.30 0.28 0.34 0.39 0.33 0.23 0.35 0.32 0.29 0.34 0.35

0.40

L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 Layer index

0.35

×10 1

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

- 0
- 1
- 2
- 3
- 4

Layerwisemean

0.30

- 0.36 0.29 0.35 0.30 0.35 0.34 0.33 0.34 0.34 0.28 0.35 0.34 0.49 0.36 0.34 0.33 0.35 0.39 0.38 0.37 0.35 0.34 0.33 0.35
- 0.37 0.36 0.38 0.49 0.40 0.46 0.48 0.38 0.33 0.41 0.48 0.45

0.25

0.51 0.43 0.40 0.42 0.34 0.45 0.41 0.41 0.44 0.42 0.51 0.47

0.20

0 1 2 3 4 5 6 7 8 9 10 11

L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 Layer index

Head index

(a) Values of learned threshold weights

(b) Layerwise mean and variance of threshold weights

- Figure 6: Analysis of learned threshold weights (reg_threshold_weights, see Eq. 4) in entropy regularization for softmax-only GPT-2 model: (a) Attention heads adaptively learn non-uniform threshold weights across different heads, setting individualized thresholds for entropy regularization; (b) The non-uniform means and non-zero variances across layers highlight the necessity and effectiveness of headwise learnable thresholds in adapting regularization strength.

Mitigating over-regularization with an appropriate threshold margin

[0, Max4 ) [Max4 , Max2 ) [Max2 , 3Max4 ) [3Max4 , Max]

0%

10%

20%

30%

40%

50%

60%

70%

PercentageofHeads

0.7%2%3%

6%

Preventing overregularization

= 0

| |
|---|

= 0.05 = 0.10 = 0.15 = 0.20 = 0.25 = 0.30

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 7: Headwise entropy distribution in the SM(t) + ScFuFFN GPT-2 model (L=12, H=12, d=768) when entropy regularization is applied with varying threshold margin, controlled by γ.

- Figure 7 illustrates the effect of γ on the headwise entropy distribution. The hyperparameter γ employed to adjust the threshold

margin in entropy regularization, defined as Tolmargin = γEmax (Algorithm1, line #3), effectively preventing over-regularization by ensuring that a sufficient fraction of heads maintains entropy

values in the upper range 3Max4 to Max. As γ increases from 0 to 0.15, only a small proportion of attention heads (0.7%) are

situated in the highest entropy range. However, as γ is increased beyond 0.15, the fraction of heads in this upper range starts increasing, reaching 2.08%, 3.47%, and 6.25% at γ=0.20, 0.25, and 0.30, respectively. This fine-grained control on the population of attention heads in the higher entropy range highlights the ability of entropy regularization to prevent over-regularization and maintain the attention heads’ diversity. We find that γ=0.2 yields slightly better performance in terms of lower perplexity compared to higher γ values, and thus, we adopt this value in the final entropy regularization scheme.

To gain deeper insights, Figure 8 illustrates entropy dynamics with increasing γ during training. As γ increases, the proportion of attention heads exhibiting higher entropy values grows. This is reflected in the rising mean entropy of the early layers, which plays a crucial role in preventing over-regularization and preserving the diversity of attention heads.

Results on GPT-2 model Table 3 presents results for GPT-2 small models, offering a detailed breakdown of nonlinear operations and FLOPs. The architectural simplification through nonlinearity reduction (SM(t) + ScFuFFN) achieves a 3.94× reduction in communication overhead and a 1.72× speedup in end-to-end PI latency. Additionally, entropy regularization enhances the perplexity of the SM(t) + ScFuFFN model by 7.8%, validating the effectiveness of the entropy-guided attention mechanism.

Scalability across model depth, context length, and training data To demonstrate the robustness of our approach, we evaluate both architectural simplifications and entropy-guided solutions across different model configurations. Experiments with deeper models (Table 8) and increased context lengths (Tables 7) show consistent benefits in terms of nonlinearity reduction and entropy regularization effectiveness.

We further analyze the scalability of our approach across different training regimes using the Languini dataset. Table 4 presents latency and communication improvements for GPT-2 models trained on varying token counts (1.2B, 2.4B, and

- 4.8B), demonstrating the consistency of our architectural benefits across different training scales.

L0 L1 L2 L3

L4 L5 L6 L7

L8 L9

LayerwiseMeanEntropy

LayerwiseMeanEntropy

LayerwiseMeanEntropy

3.5

3.5

3.5

- L10

- L11

3.0

3.0

3.0

2.5

2.5

2.5

2.0

2.0

2.0

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

(a) Tolmargin = 0

(b) Tolmargin = 0.05Emax

(c) Tolmargin = 0.10Emax

LayerwiseMeanEntropy

LayerwiseMeanEntropy

LayerwiseMeanEntropy

3.5

3.5

3.5

3.0

3.0

3.0

2.5

2.5

2.5

2.0

2.0

2.0

1.5

1.5

1.5

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

0K 5K 10K 15K 20K 25K 30K Steps

(d) Tolmargin = 0.15Emax

(e) Tolmargin = 0.20Emax

(f) Tolmargin = 0.25Emax

- Figure 8: Layerwise entropy dynamics when entropy regularization is employed with increasing threshold margin, defined as Tolmargin = γEmax (see Algorithm1, line #3). At higher γ, the mean entropy of the early layers increases.

Table 3: Results on GPT-2 (L=12, H=12, d=768), trained from scratch on the CodeParrot dataset (2.1B tokens, T=128).

#FLOPs

Savings FFN Attn. Comm. Lat.

Lat. (min.)

Comm. (GB)

Network Arch. PPL #Nonlinear Ops

SM:144 × R128×128 LN:24 × R128×768 14.5B 7.7B 25.32 8.21 1× 1× G:12 × R128×3072

SM + LN + G 2.69

Baseline

SM:144 × R128×128 LN:24 × R128×768 14.5B 7.7B 9.44 6.06 2.68× 1.35× R:12 × R128×3072

SM + LN + R 2.76

SM + ScFuFFN 3.48 SM:144 × R128×128 1.8B 7.7B 6.43 4.76 3.94× 1.72× EReg(SM(t) + ScFuFFN) 3.21 SM:144 × R128×128 1.8B 7.7B 6.43 4.76 3.94× 1.72×

Table 4: Results on GPT-2 (L=12, H=12, d=768) model, trained from scratch on Languini [20] (T=512)

Eval PPL

#FLOPs

Lat. 1.2B 2.4B 4.8B FFN Attn. (min.)

Comm. (GB)

Network Arch.

#Nonlinear Ops

SM:144 × R512×512 LN:24 × R512×768 58.0B 36.2B 145.24 30.74 G:12 × R512×3072

SM + LN + G 25.71 23.32 21.29

Baseline

SM:144 × R512×512 LN:24 × R512×768 58.0B 36.2B 81.71 23.54 R:12 × R512×3072

SM + LN + R 26.06 23.55 21.58

SM + ScFuFFN 33.77 30.82 28.59 SM:144 × R512×512 7.3B 36.2B 69.68 19.44 EReg(SM(t) + ScFuFFN) 31.54 28.70 26.55 SM:144 × R512×512 7.3B 36.2B 69.68 19.44

###### 6 Conclusion

In this work, we address the fundamental challenges posed by nonlinear operations in private LLMs inference. By leveraging an information-theoretic framework, we uncover the dual role of nonlinearities in ensuring training stability and maintaining attention head diversity. Our study introduces novel entropy regularization techniques, and PI-friendly alternatives for layer normalization, demonstrating their effectiveness in mitigating entropy collapse and entropic overload. These contributions pave the way for PI-optimized architectures with reduced-nonlinearities, significantly reducing latency and communication overheads. By addressing the critical trade-offs between nonlinearity, computational overhead, and entropy dynamics, we provide a clear path toward scalable and practical PI systems.

Limitations This study mainly focuses on pre-training performance, with perplexity as the primary metric, and does not include experiments to evaluate other capabilities such as transfer learning or few-shot learning. Additionally, the efficacy of the proposed Softmax-only models has been validated on LLMs with lesser than 1B parameters. Future work will explore broader experimental evaluations, including their adaption for large-scale models.

###### Additional Notes

This workshop version focuses on the fundamental role of nonlinearities in maintaining model stability and fostering attention head diversity in LLMs, as well as their implications for private inference. These findings are part of a broader study presented in our comprehensive paper AERO: Softmax-Only LLMs for Efficient Private Inference. The code and implementation are available at entropy-guided-llm.

###### References

- [1] Robin Staab, Mark Vero, Mislav Balunovic, and Martin Vechev. Beyond memorization: Violating privacy via inference with large language models. In The Twelfth International Conference on Learning Representations (ICLR), 2024.
- [2] Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, and Yejin Choi. Can LLMs keep a secret? testing privacy implications of language models via contextual integrity theory. In The Twelfth International Conference on Learning Representations, 2024.
- [3] Aman Priyanshu, Supriti Vijay, Ayush Kumar, Rakshit Naidu, and Fatemehsadat Mireshghallah. Are chatbots ready for privacy-sensitive applications? an investigation into input regurgitation and prompt-induced sanitization. arXiv preprint arXiv:2305.15008, 2023.
- [4] Goode Lauren and Will Knight. Chatgpt can now talk to you—and look into your life. https://www.wired. com/story/chatgpt-can-now-talk-to-you-and-look-into-your-life/, 2023.
- [5] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint, 2016.
- [6] Wen-jie Lu, Zhicong Huang, Zhen Gu, Jingyu Li, Jian Liu, Kui Ren, Cheng Hong, Tao Wei, and WenGuang Chen. Bumblebee: Secure two-party inference framework for large transformers. In Annual Network and Distributed System Security Symposium (NDSS), 2025.
- [7] Xiaoyang Hou, Jian Liu, Jingyu Li, Yuhan Li, Wen-jie Lu, Cheng Hong, and Kui Ren. Ciphergpt: Secure two-party gpt inference. Cryptology ePrint Archive, 2023.
- [8] Jimmy Lei Ba. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
- [9] Hongkang Li, Meng Wang, Songtao Lu, Xiaodong Cui, and Pin-Yu Chen. How do nonlinear transformers learn and generalize in in-context learning? In Forty-first International Conference on Machine Learning (ICML), 2024.
- [10] Xiang Cheng, Yuxin Chen, and Suvrit Sra. Transformers implement functional gradient descent to learn non-linear functions in context. In Forty-first International Conference on Machine Learning (ICML), 2024.
- [11] Mitchell Wortsman, Peter J Liu, Lechao Xiao, Katie E Everett, Alexander A Alemi, Ben Adlam, John D Co-Reyes, Izzeddin Gur, Abhishek Kumar, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein, Kelvin Xu, Jaehoon Lee, Justin Gilmer, and Simon Kornblith. Small-scale proxies for large-scale transformer training instabilities. In The Twelfth International Conference on Learning Representations (ICLR), 2024.
- [12] Oleg Rybakov, Mike Chrzanowski, Peter Dykas, Jinze Xue, and Ben Lanir. Methods of improving llm training stability. arXiv preprint arXiv:2410.16682, 2024.
- [13] Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Joshua M Susskind. Stabilizing transformer training by preventing attention entropy collapse. In International Conference on Machine Learning (ICML), 2023.

- [14] Han Bao, Ryuichiro Hataya, and Ryo Karakida. Self-attention networks localize when QK-eigenspectrum concentrates. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.
- [15] Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [16] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In International Conference on Machine Learning (ICML), 2021.
- [17] Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in neural information processing systems, 2016.
- [18] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations (ICLR), 2018.
- [19] Hugging Face. Codeparrot. https://huggingface.co/learn/nlp-course/chapter7/6.
- [20] Aleksandar Stani´c, Dylan Ashley, Oleg Serikov, Louis Kirsch, Francesco Faccio, Jürgen Schmidhuber, Thomas Hofmann, and Imanol Schlag. The languini kitchen: Enabling language modelling research at different scales of compute. arXiv preprint arXiv:2309.11197, 2023.
- [21] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning (ICML), 2020.
- [22] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019.
- [23] Jiawen Zhang, Jian Liu, Xinpeng Yang, Yinghao Wang, Kejia Chen, Xiaoyang Hou, Kui Ren, and Xiaohu Yang. Secure transformer inference made non-interactive. In Annual Network and Distributed System Security Symposium (NDSS), 2025.
- [24] Qi Pang, Jinhao Zhu, Helen Möllering, Wenting Zheng, and Thomas Schneider. Bolt: Privacy-preserving, accurate and efficient inference for transformers. In IEEE Symposium on Security and Privacy (SP), 2024.
- [25] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 1948.
- [26] Edwin T Jaynes. Information theory and statistical mechanics. Physical review, 1957.
- [27] Hamidreza Ghader and Christof Monz. What does attention in neural machine translation pay attention to? In Proceedings of the The 8th International Joint Conference on Natural Language Processing, 2017.
- [28] Jesse Vig and Yonatan Belinkov. Analyzing the structure of attention in a transformer language model. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, 2019.
- [29] Yury Nahshan, Joseph Kampeas, and Emir Haleva. Linear log-normal attention with unbiased concentration. In The Twelfth International Conference on Learning Representations (ICLR), 2024.
- [30] Bobby He, Lorenzo Noci, Daniele Paliotta, Imanol Schlag, and Thomas Hofmann. Understanding and minimising outlier features in neural network training. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [31] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In International Conference on Machine Learning (ICML), 2023.
- [32] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. Olmoe: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024.
- [33] David Miller, Ajit V Rao, Kenneth Rose, and Allen Gersho. A global optimization technique for statistical classifier design. IEEE transactions on signal processing, 1996.
- [34] Amrith Setlur, Benjamin Eysenbach, Virginia Smith, and Sergey Levine. Maximizing entropy on adversarial examples can improve generalization. In ICLR 2022 Workshop on PAIR^2Struct: Privacy, Accountability, Interpretability, Robustness, Reasoning on Structured Data, 2022.
- [35] Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geoffrey Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.
- [36] Edwin T Jaynes. On the rationale of maximum-entropy methods. In Proceedings of the IEEE, 1982.

- [37] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019.
- [38] Bobby He and Thomas Hofmann. Simplifying transformer blocks. In The Twelfth International Conference on Learning Representations (ICLR), 2024.
- [39] Jonas Geiping and Tom Goldstein. Cramming: Training a language model on a single gpu in one day. In International Conference on Machine Learning (ICML), 2023.
- [40] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations (ICML), 2024.
- [41] Nicola Cancedda. Spectral filters, dark signals, and attention sinks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024.
- [42] Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. arXiv preprint arXiv:2410.10781, 2024.
- [43] Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, and Furu Wei. Differential transformer. In arXiv preprint arXiv:2410.05258, 2024.
- [44] Jerry Yao-Chieh Hu, Pei-Hsuan Chang, Haozheng Luo, Hong-Yu Chen, Weijian Li, Wei-Po Wang, and Han Liu. Outlier-efficient hopfield layers for large transformer-based models. In Forty-first International Conference on Machine Learning (ICML), 2024.
- [45] Qingyu Yin, Xuzheng He, Xiang Zhuang, Yu Zhao, Jianhua Yao, Xiaoyu Shen, and Qiang Zhang. Stablemask: Refining causal masking in decoder-only transformer. In Forty-first International Conference on Machine Learning (ICML), 2024.
- [46] Zhongzhi Yu, Zheng Wang, Yonggan Fu, Huihong Shi, Khalid Shaikh, and Yingyan Celine Lin. Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration. In Forty-first International Conference on Machine Learning (ICML), 2024.
- [47] Entropix Development Team. Entropix: Tool for entropy based sampling and parallel cot decoding. https: //github.com/xjdr-alt/entropix, 2024.
- [48] Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, and Dimitris Bertsimas. Universal neurons in GPT2 language models. In Transactions on Machine Learning Research (TMLR), 2024.
- [49] Alessandro Stolfo, Ben Peng Wu, Wes Gurnee, Yonatan Belinkov, Xingyi Song, Mrinmaya Sachan, and Neel Nanda. Confidence regulation neurons in language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.
- [50] Pierre-Carl Langlais. Entropy is all you need? the quest for best tokens and the new physics of llms. https: //indico.cern.ch/event/1474571/, 2024.
- [51] Elliot Glazer, Ege Erdil, Tamay Besiroglu, Diego Chicharro, Evan Chen, Alex Gunning, Caroline Falkman Olsson, Jean-Stanislas Denain, Anson Ho, Emily de Oliveira Santos, et al. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai. arXiv preprint arXiv:2411.04872, 2024.
- [52] Petar Veliˇckovi´c, Christos Perivolaropoulos, Federico Barbero, and Razvan Pascanu. softmax is not enough (for sharp out-of-distribution). arXiv preprint arXiv:2410.01104, 2024.

# Appendix

#### Table of Contents

- A Softmax Learnable Temperature for Entropy-Guided Attention 14
- B PyTorch Implementation of Entropy Regularization 15
- C Additional Results 15

- C.1 Performance Comparison of Weight and Spectral Normalization, and Learnable FFN Scaling . . . . . 15
- C.2 Scalability for model depth and context length . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- D Broader Impacts and Potential of Entropy-Guided LLM Solutions 17

- A Softmax Learnable Temperature for Entropy-Guided Attention With the learnable temperature parameters (t), the attention matrix can be expressed as follows:

###### A(l,h)(t) = aij(l,h)(t)

exp 1

###### (XiWQ)(XjWK)⊤

###### √

ti

dk

, where aij(l,h)(t) =

T×T

∑kT=1 exp 1

(XiWQ)(XkWK)⊤ .

###### √

ti

dk

Let zij = XiWQ XjWK ⊤ represents the logits (attention scores before applying softmax). Now, substituting aij(l,h)(t) into the entropy formula: E(l,h)(t) = −

  

   .

exp 1

exp 1

zij

zij

###### √

###### √

T

T

1 T

t

dk

t

dk

##### ∑

##### ∑

log

∑kT=1 exp 1

∑kT=1 exp 1

i=1

j=1

zik

zik

###### √

###### √

t

dk

t

dk

Simplifying the logarithmic term:

  

   =

exp 1

zij

###### √

t

dk

log

∑kT=1 exp 1

zik

###### √

t

dk

Thus, the entropy simplifies to:

1 t√dk

zij − log

T

##### ∑

exp

k=1

1 t√dk

zik .

(9)

E(l,h)(t) =

T

1 T

##### ∑

i=1

log

T

##### ∑

exp

k=1

1 t√dk

1 t√dk

zik −

T

aij(l,h)(t)zij .

##### ∑

j=1

Further, it can be simplified as a function of expected value of zij under the attention distribution:

T

T

1 T

1 t√dk

zik

E(l,h)(t) =

##### ∑

##### ∑

log

exp

ij (t)[ zij ] (10) In the above expression (Eq. 10), the first term (log ∑) represents the overall spread of the logits when scaled by t, and the second term 1t E[zij] represents the expected value of the scaled logits under the attention distribution. Temperature cases when:

t√dk −

Ej∼a(l,h)

i=1

k=1

- 1. t > 1: The scaling factor 1t reduces the influence of the logits zij, making the softmax distribution more uniform. Consequently, the entropy increases.

- 2. t < 1: The scaling factor 1t increases the influence of the logits zij, making the softmax distribution more peaked. Consequently, the entropy decreases.

- 3. t → ∞: The logits are scaled down to zero, and the softmax becomes a uniform distribution. The entropy reaches its maximum value of log T.
- 4. t → 0: The logits dominate the softmax, and it becomes a one-hot distribution. The entropy approaches zero.

###### B PyTorch Implementation of Entropy Regularization

The PyTorch implementation below computes the entropy regularization loss for attention weights in a transformer model. This regularization ensures a balanced attention score distribution, fostering head-specialization in MHA.

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
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22
- 23
- 24
- 25
- 26
- 27
- 28
- 29
- 30
- 31
- 32
- 33
- 34
- 35
- 36
- 37
- 38
- 39
- 40
- 41

PyTorch Implementation 1: Entropy Regularization Loss Calculation

|import torch def calculate_entropy_reg_loss(attentions, blocks, seq_len):<br><br>""" Calculate the entropy regularization loss.<br><br>Parameters: attentions (list): A list of attention matrices from different layers. blocks (list): A list of transformer blocks. seq_len (int): The length of the sequence (context length).<br><br>Returns: float: The entropy regularization loss. """ entropy_reg_loss = 0 max_entropy = torch.log(torch.tensor(seq_len)) # Theoretical maximum entropy fraction = 0.10 # Design hyper-parameter for tolerance margin tolerance_margin = fraction * max_entropy # Set tolerance margin as fraction of the maximum<br><br>entropy for layer_idx, (block, attn_mat) in enumerate(zip(blocks, attentions)): reg_threshold_weights = block.attn.reg_threshold_weights # Head-wise learnable parameters to set head-specific threshold ent_val = -torch.sum(attn_mat * torch.log(attn_mat + 1e-9), dim=-1) # Compute entropy<br><br>averaged over sequence length layer_entropy_reg_loss = 0 for head_idx in range(block.attn.num_heads):<br><br>head_entropy = ent_val[:, head_idx, :] # Get head-specific entropy threshold = reg_threshold_weights[head_idx] * max_entropy deviation = torch.abs(head_entropy - threshold) penalty = torch.square(torch.where(deviation > tolerance_margin, deviation, torch.<br><br>zeros_like(deviation))) layer_entropy_reg_loss += penalty.sum()<br><br>layer_entropy_reg_loss /= block.attn.num_heads entropy_reg_loss += layer_entropy_reg_loss<br><br>entropy_reg_loss /= len(attentions) return entropy_reg_loss<br><br># Calculate the total loss including entropy regularization lambda_reg = 1e-5 # Hyperparameter for entropy regularization weight entropy_regularization = calculate_entropy_reg_loss(attentions, blocks, seq_len) total_loss = ce_loss + lambda_reg * entropy_regularization<br><br>|
|---|

###### C Additional Results

###### C.1 Performance Comparison of Weight and Spectral Normalization, and Learnable FFN Scaling

Table 5 compares the performance of weight and spectral normalization applied in various linear layers within the attention and FFN sub-blocks in Softmax-only model. The results show that applying these techniques to the attention blocks yields diminishing returns compared to their application in the FFN.

When comparing performance, we find that weight and spectral normalization led to similar performance while the learnable scaling method outperformed them with a lower perplexity (Table 6).

- Table 5: Comparison of weight normalization [17] and spectral normalization [18] when employed in Softmax-only GPT-2 (L=12, H=12, d=768) models, and trained from scratch on CodeParrot dataset with 128 input context length. FFN weight normalization yield the similar results; whereas, weight normalization works better in other linear layers.

Linear layers Eval PPL(Weight Normalization) Eval PPL(Spectral Normalization)

QK 3.89 4.25 FFN 3.64 3.63 QK+FFN 3.88 4.23 QKV+FFN 3.93 4.26 QKVO+FFN 3.98 4.34

- Table 6: Perplexity comparison of weight normalization, spectral normalization, and learnable scaling employed in FFN of softmax-only GPT-2 model, when trained from scratch on CodeParrot dataset with 128 input context length.

Weight Normalization Spectral Normalization Scaled-FFN Eval PPL 3.640 3.624 3.478

C.2 Scalability for model depth and context length GPT-2 Model with 256 tokens as input context Table 7 provides the latency and communication savings achieved on the GPT-2 model with 256 context length, along with a detailed breakdown of the nonlinear operations and FLOPs.

- Table 7: Results on GPT-2 (L=12, H=12, d=768), trained from scratch on the CodeParrot dataset (2.1B tokens, T=256).

Network Arch. PPL #Nonlinear Ops

#FLOPs

Comm. (GB)

Lat. (min.)

Savings FFN Attn. Comm. Lat.

Baseline

SM + LN + G 2.35

SM:144 × R256×256 LN:24 × R256×768 29.0B 16.3B 58.51 16.57 1× 1× G:12 × R256×3072

SM + LN + R 2.41

SM:144 × R256×256 LN:24 × R256×768 29.0B 16.3B 26.73 12.59 2.19× 1.32× R:12 × R256×3072

SM + ScFuFFN 3.03 SM:144 × R256×256 3.6B 16.3B 20.72 10.45 2.82× 1.59× EReg(SM(t) + ScFuFFN) 2.92 SM:144 × R256×256 3.6B 16.3B 20.72 10.45 2.82× 1.59×

GPT-2 Model with 18 Layers Table 8 provides the latency and communication savings achieved on a 18-layer GPT-2 model, along with a detailed breakdown of the nonlinear operations and FLOPs.

- Table 8: Results on GPT-2 (L=18, H=12, d=768), trained from scratch on the CodeParrot dataset (2.1B tokens, T=128).

Savings FFN Attn. Comm. Lat.

#FLOPs

Lat. (min.)

Comm. (GB)

Network Arch. PPL #Nonlinear Ops

SM:216 × R128×128 LN:36 × R128×768 21.7B 11.6B 37.17 10.77 1× 1× G:18 × R128×3072

SM + LN + G 2.56

Baseline

SM:216 × R128×128 LN:36 × R128×768 21.7B 11.6B 13.34 8.04 2.79× 1.34× R:18 × R128×3072

SM + LN + R 2.63

SM + ScFuFFN 3.24 SM:216 × R128×128 2.7B 11.6B 8.83 6.07 4.21× 1.77× EReg(SM(t) + ScFuFFN) 3.13 SM:216 × R128×128 2.7B 11.6B 8.83 6.07 4.21× 1.77×

###### D Broader Impacts and Potential of Entropy-Guided LLM Solutions

Entropy-guided framework for tackling softmax-inherent challenges in attention mechanism The softmax function, fundamental to transformer-based attention mechanisms, inherently assigns non-zero probabilities to all tokens due to its normalized exponential structure. This characteristic leads to two primary issues inherent to softmax: disproportionate emphasis on specific tokens (known as attention sink) [40, 41, 42]; and non-zero scores for irrelevant tokens (known as attention noise). These challenges can result in undesirable effects such as hallucinations [43], outlier activations [44], and inefficient use of model capacity, such as rank collapse [14].

While prior research has proposed various strategies to mitigate these issues [45, 46, 14], we introduce a principled approach to control attention entropy distribution. By penalizing excessively high entropy values and incorporating learnable threshold parameters, each attention head adaptively determine its optimal degree of focus. This could prevent the over-diffusion of attention scores while preserving the mathematical properties of softmax.

Entropy-guided framework for uncertainty estimation and mathematical reasoning Recent progress in entropybased methodologies, such as the Entropix framework for entropy-guided sampling [47], and the discovery of entropy neurons that regulates uncertainty in next-token prediction [48, 49], highlights a major shift in the entropy-drive LLM solutions. These approaches are particularly relevant to improving token-level performance in mathematical reasoning tasks [50]. Moreover, the recent FrontierMath benchmarks [51], which have attracted considerable attention in the research community, further highlight the critical need to improve the reasoning capabilities of LLMs.

A key observation from this research is that the demands of mathematical operations vary in terms of token selection confidence. For instance, deterministic (low-entropy) token selection may be more appropriate for simple arithmetic, while exploratory (high-entropy) token selection may be advantageous in complex problem-solving scenarios.

While our current work focuses on entropy regularization of attention scores in MHA, this concept can be extended to guide token selection during inference. This is analogous to adaptive temperature strategies, where model creativity is modulated based on logit entropy [52]. Furthermore, controlled entropy pathways tailored to numerical computations, coupled with task-specific entropy thresholds, present a promising direction for future work.

To complement these strategies, our simplified architecture could incorporate reasoning tokens, inspired by pause tokens proposed for reasoning processes [50]. By leveraging entropy regularization to influence the model’s interaction with such tokens, it is possible to construct more structured and interpretable pathways for mathematical reasoning.

Parallels between entropy-guided attention and differential attention mechanism Our entropy regularization framework exhibits conceptual alignment with the recently introduced Differential Transformer architecture [43], despite notable differences in their methodologies.

Similar to how the differential attention mechanism suppresses attention noise via contrastive learning, entropy-guided attention can achieve comparable outcomes by penalizing excessive dispersal of attention across tokens. Both methods ultimately encourage sparse attention patterns: the Differential Transformer accomplishes this by leveraging differences between attention maps, whereas entropy regularization explicitly penalizes high-entropy attention distributions.

These parallels highlight that selective attention can be fostered through either architectural innovations or targeted regularization strategies. Together, they offer complementary approaches to achieving the shared objective of promoting more focused and efficient attention mechanisms.

