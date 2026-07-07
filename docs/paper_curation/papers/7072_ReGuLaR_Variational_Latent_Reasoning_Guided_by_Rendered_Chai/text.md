# arXiv:2601.23184v1[cs.CL]30Jan2026

February 2, 2026

## ReGuLaR: Variational Latent Reasoning Guided by Rendered Chain-of-Thought

###### Fanmeng Wang1,2 , Haotian Liu1 , Guojiang Zhao2 , Hongteng Xu∗ 1,3,4 , Zhifeng Gao∗ 2

1Gaoling School of Artificial Intelligence, Renmin University of China 2DP Technology 3Beijing Key Laboratory of Research on Large Models and Intelligent Governance 4Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE

###### Abstract

While Chain-of-Thought (CoT) significantly enhances the performance of Large Language Models (LLMs), explicit reasoning chains introduce substantial computational redundancy. Recent latent reasoning methods attempt to mitigate this by compressing reasoning processes into latent space, but often suffer from severe performance degradation due to the lack of appropriate compression guidance. In this study, we propose Rendered CoT-Guided variational Latent Reasoning (ReGuLaR), a simple yet novel latent learning paradigm resolving this issue. Fundamentally, we formulate latent reasoning within the Variational Auto-Encoding (VAE) framework, sampling the current latent reasoning state from the posterior distribution conditioned on previous ones. Specifically, when learning this variational latent reasoning model, we render explicit reasoning chains as images, from which we extract dense visual-semantic representations to regularize the posterior distribution, thereby achieving efficient compression with minimal information loss. Extensive experiments demonstrate that ReGuLaR significantly outperforms existing latent reasoning methods across both computational efficiency and reasoning effectiveness, and even surpasses CoT through multi-modal reasoning, providing a new and insightful solution to latent reasoning. Code: https://github.com/FanmengWang/ReGuLaR.

### 1 Introduction

Large Language Models (LLMs) have demonstrated exceptional performance in solving complex problems, a success largely attributed to the adoption of Chain-of-Thought (CoT) techniques (Wei et al., 2022; Jin

- et al., 2024). By eliciting LLMs to generate intermediate reasoning steps in natural language, CoT effectively decomposes complex problems, significantly bolstering accuracy on challenging queries (Fei et al., 2023; Wang et al., 2024). However, such reasoning processes suffer from inherent inefficiency because they rely on explicit token-by-token generation, and many tokens can be redundant for improving reasoning (Kang et al., 2025; Sui et al., 2025). This results in prohibitive computational overhead and increased inference latency, fundamentally limiting the scalability of LLM reasoning. In this context, recent studies have explored latent reasoning as a compelling alternative to explicit CoT (Zhu
- et al., 2025b). By operating directly on continuous representations, latent reasoning compresses reasoning processes into the high-dimensional latent space, thereby circumventing the overhead of decoding intermediate reasoning tokens (Zhu et al., 2025a). To instantiate this paradigm, several representative frameworks have been proposed, e.g., Coconut (Hao et al., 2025) and CoLaR (Tan et al., 2025). However, while alleviating computational burdens, existing latent reasoning methods often suffer from severe performance degradation, primarily because the compression of reasoning processes lacks appropriate guidance. Specifically, these methods typically rely on recursively or dynamically utilizing the hidden states of reasoning tokens to propagate logical dependencies. In the absence of discrete tokens to anchor the reasoning trajectory, this

∗Corresponding authors. Email: hongtengxu@ruc.edu.cn and gaozf@dp.tech

unconstrained recursive process becomes highly susceptible to error accumulation, leading to significant information loss and semantic drift.

In this study, we propose a novel and insightful paradigm to resolve the above challenge, learning a Rendered CoT-Guided variational Latent Reasoning (ReGuLaR) model. Fundamentally, we formulate latent reasoning as a probabilistic modeling task within the Variational Auto-Encoding (VAE) framework (Kingma and Welling, 2014). In this formulation, the latent reasoning process is achieved by sampling the current latent reasoning state from the posterior distribution conditioned on previous ones. Here, we optimize this model by maximizing its Evidence Lower Bound (ELBO) (Neal and Hinton, 1998), wherein the prior distribution of the latent reasoning state plays a critical role in regularizing the posterior distribution. As illustrated in Figure 1, we render the explicit reasoning chain as images and then leverage the visual encoder to extract visual representations with dense semantics. This rendering step is lossless, so we utilize these visual representations to regularize the posterior distribution of the latent reasoning state during training, thereby leading to our ReGuLaR with compressed but semantically meaningful latent reasoning states.

To the best of our knowledge, ReGuLaR is the first work that applies the VAE framework to understanding and modeling latent reasoning. With this framework, we demonstrate the importance of the latent reasoning state prior and propose a promising approach to designing a semantically meaningful, information-preserving prior for latent reasoning states. Extensive experiments demonstrate that ReGuLaR provides a new and insightful solution to latent reasoning. Specifically, it significantly outperforms existing latent reasoning methods, achieving state-of-the-art performance with minimal reasoning length. Furthermore, ReGuLaR natively supports multi-modality within its latent reasoning processes by rendering non-textual elements alongside text, enabling it to surpass explicit CoT in complicated reasoning scenarios.

Explicit Chain-of-Thought

#### r1 r2 rL

##### r3

Rendering Text

###### K Images Visual Representations

Encoding

[Figure 1]

[Figure 2]

[Figure 3]

…

zˆ1 zˆ2 … zˆK

Guiding Latent Reasoning

z1 z2 zK

Question Answer

Latent Reasoning Process

Figure 1 . Illustration of our modeling principle. Given an explicit reasoning chain of length L, we render it onto K images (K ≪ L) and extract corresponding visual representations to guide the latent reasoning process with K steps.

### 2 Related Work

###### 2.1 LLM Reasoning and Latent Reasoning

The reasoning capabilities of LLMs have been advanced by CoT techniques, which prompt the generation of intermediate reasoning steps in natural language (Wei et al., 2022; Shao et al., 2024; Jin et al., 2024). Building on this, subsequent research has explored various CoT constructions, including Tab-CoT (Ziqi and Lu, 2023), ToT (Yao et al., 2023), and GoT-Rationale (Besta et al., 2024). While these methods manifest reasoning in different explicit forms, verbose intermediate reasoning steps inevitably incur substantial computational cost and inference latency (Kang et al., 2025; Sui et al., 2025).

To mitigate this bottleneck, iCoT (Deng et al., 2024) internalizes intermediate reasoning steps by progressively removing them during training. Moreover, the latent reasoning paradigm has emerged, which transforms intermediate reasoning tokens into continuous representations and eliminates the overhead of languagedecoding steps by executing latent reasoning processes (Zhu et al., 2025b). In particular, Coconut (Hao et al., 2025) pioneers this direction by recursively utilizing the last-layer hidden states of LLMs as the continuous latent thought, functioning as the next input embedding to drive subsequent reasoning. Meanwhile, CODI (Shen et al., 2025) further employs self-distillation to align the hidden activations of latent thoughts with explicit CoT trajectories. Most recently, CoLaR (Tan et al., 2025) achieves state-of-the-art performance

by leveraging training mechanisms with variable compression factors to support flexible reasoning length. However, these methods suffer from severe performance degradation compared with explicit CoT, primarily due to the lack of appropriate compression guidance, thereby limiting their practical utility.

###### 2.2 Empowering LLMs via Visual-Text Compression

LLMs typically rely on discrete tokenization to process text inputs, a mechanism that inevitably fragments the global semantic topology while incurring substantial computational cost (Zhao et al., 2023). To overcome these inefficiencies, the paradigm of visual-text compression (Zhao et al., 2025b) has been explored. It renders textual content as images and embeds them via the visual encoder, thereby exploiting the high information density of the visual modality. In particular, VisInContext (Wang et al., 2024) introduces a visualized in-context text processing framework that leverages compact visual tokens to replace long textual contexts, effectively expanding context windows without additional computational burden. Subsequently, VIST (Xing et al., 2025) proposes a fast-path compression mechanism that leverages the lightweight visual encoder to process rendered images of distant contexts for rapid skimming, significantly improving efficiency. Recently, DeepSeek-OCR (Wei et al., 2025) further brings this paradigm to the forefront by validating its feasibility and scalability on massive textual data, enabling the mapping of extensive textual contexts into ultra-compact visual tokens with high compression ratios.

While these works primarily focus on compressing input contexts, they provide strong evidence for visual representations as high-density carriers of textual information. Therefore, such visual-text compression should also be useful in empowering latent reasoning, as our work verifies.

### 3 Proposed Method

###### 3.1 Problem Statement and Preliminaries

Formally, suppose that we have a reasoning dataset D = {(Q,R,A)}, in which each tuple contains an input question Q, an intermediate reasoning chain R, and the final answer A. Here, we represent each element in the tuple as a token sequence, i.e., Q = {qi}L

i=1, R = {ri}L

i=1, and A = {ai}L

i=1, where Lq, Lr, and La are the respective sequence lengths. Additionally, for an arbitrary token sequence T , we denote the corresponding subsequence before the i-th token as T<i.

q

r

a

Learning an LLM with explicit CoT. Under the Chain-of-Thought (CoT) paradigm, the LLM explicitly generates the reasoning chain token by token before producing the final answer, thereby bridging the logical gap between the question and the corresponding answer. Accordingly, given (Q,R,A), we can learn the LLM via the Maximum Likelihood Estimation (MLE) as follows:

Lr

La

, (1)

log pθ(ri|Q,R<i)

log pθ(ai|Q,R,A<i)

max

+

θ

i=1

i=1

Lreasoning

Lanswer

where θ denotes the model parameters, pθ represents the conditional probability of each token given its history. In practice, we implement Lreasoning and Lanswer as two Cross-Entropy (CE) losses (Mao et al., 2023).

- As illustrated in Figure 2a, for the LLM using explicit reasoning, all tokens are first mapped into continuous

embedding vectors (denoted as eQ1:L

), and subsequently transformed by the model layers into the last hidden states (denoted as hQ1:L

, eR1:L

, and eA1:L

r

a

q

). In this context, the LLM necessitates the explicit token-by-token generation of the intermediate reasoning chain during inference, resulting in substantial computational overhead and inference latency.

, hR1:L

, and hA1:L

r

a

q

Learning an LLM with latent reasoning. The inefficiency of CoT further motivates the latent reasoning paradigm illustrated in Figure 2b. Specifically, latent reasoning replaces discrete reasoning tokens R = {ri}L

i=1 with continuous latent rea-

r

soning states Z = {zk}Kk=1. In this context, the LLM employs an additional latent reasoning head

to derive the latent reasoning state from the current hidden state, which functions as the subsequent input embedding, thereby eliminating the overhead of decoding those intermediate reasoning tokens. Moreover, the sequence of latent reasoning states can be much shorter than the corresponding reasoning chain (K ≪ Lr), which helps improve inference efficiency significantly.

Learning such a latent reasoning model corresponds to the following optimization problem, i.e.,

###### Latent Reasoning

###### Chain-of-Thought

a1

###### a1

|Language Head|
|---|

|Language Head|
|---|

h1 h2 hK→1 hK

hQLq hR1 hR2

hRL

hRL

hQL

r→1

q

r

| | | | | | | |
|---|---|---|---|---|---|---|
| |Auto|regres|sive Decoder| | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |Auto|regres|sive Decoder| | | |
| | | | | | | |

z1 z2 z3 zK

eR1 eR2 eR3 eRL

eQL

eQL

r

q

q

| | | | | | |
|---|---|---|---|---|---|
| |E|mbedd|ing La|yer| |
| | | | | | |

| | | | | |
|---|---|---|---|---|
|Latent Reasoning Head| | | | |

qL

rL

r1 r2 r3 r

q

(a) Explicit reasoning

(b) Latent reasoning

Figure 2 . Comparison of CoT-based explicit reasoning and latent reasoning, where the autoregressive decoder corresponds to the underlying LLM.

La i=1

log pθ′(ai | Q,Z,A<i), (2)

maxθ′={θ,τ}

where the final answer is conditioned on the latent reasoning state sequence Z rather than the explicit reasoning chain R. The parameter θ′ = {θ,τ}, where τ denotes the parameters of the latent reasoning head.

Compared to (1), this optimization problem is inherently challenging due to the absence of ground-truth supervision for latent reasoning states. To mitigate this, Coconut (Hao et al., 2025) employs the multi-stage curriculum to progressively replace discrete reasoning steps with the last hidden state of the preceding context. However, since it relies on distilling knowledge from the original CoT, its performance is fundamentally bounded. Furthermore, CoLaR (Tan et al., 2025) directly constructs latent reasoning states by dynamically compressing the embeddings of original reasoning tokens. Nevertheless, its token grouping strategy introduces arbitrary inductive biases, and this simple aggregation inevitably leads to semantic information loss.

To overcome these problems and achieve effective latent reasoning, we need to regularize latent reasoning states, and further impose additional information during training to ensure semantically meaningful, which motivates the proposed ReGuLaR method.

###### 3.2 The Variational Latent Reasoning Framework

Suppose that we would like to learn an LLM with the latent reasoning mechanism, employing the latent reasoning state sequence Z of length K to replace the explicit reasoning chain R of length Lr. For such an LLM, we decompose its parameters into two parts, i.e., θ′ = {ψ,ϕ}, where ψ denotes the parameters of the language head that outputs discrete tokens and ϕ denotes the remaining parameters that comprise the latent reasoning head deriving Z. In this context, we build a variational latent reasoning process: the LLM samples each latent reasoning state from its posterior distribution given the question and the previous ones, i.e.,

zk ∼ pϕ(·|Q,Z<k), for k = 1,...,K, (3)

where Z<1 = ∅. In this study, we model pϕ(·|Q,Z<k) as a normal distribution N(µk,diag(σk2)). Applying the reparametrization trick (Kingma and Welling, 2014), we leverage the latent reasoning head of the LLM to

output µk and log σk based on Q and Z<k, and then sample zk = µk + σk ⊙ ϵ, where ϵ ∼ N(0,I) and ⊙ denotes the Hadamard product operation.

Desideratum. Ideally, Z should have the same information as the original reasoning chain R. More specifically, for k = 1,...,K, the latent reasoning state zk should correspond to the k-th segment of R,

denoted as Rk, where Rk ∩ Rk′ = ∅ for k ̸= k′ and R = ∪Kk=1Rk. Accordingly, we should be able to sample tokens in Rk through the distribution conditioned on zk, i.e.,

r ∼ pψ(·|zk) and r ∈ Rk, for k = 1,...,K. (4)

Here, pψ is modeled using Softmax Regression, and the language head of the LLM is used to output corresponding logit values of sampled tokens.

Motivated by the above desideratum, we introduce the following conditional independence assumption.

Assumption 1 (Conditional Independence). For k = 1,...,K, i) the token r ∈ Rk is independent with tokens in {Q,R} \ Rk conditioned on the latent reasoning state zk, and ii) the latent reasoning state zk is independent with tokens in {Q,R} \ Rk conditioned on the tokens in Rk.

Suppose that r is the i-th token in R, which corresponds to the j-th token in Rk. For the conditional probability used in explicit CoT, we can rewrite it based on the assumption:

p(r|Q,R<i) =

p(r|zk,Q,R<i)p(zk|Q,R<i)dzk

zk

=

pψ(r|zk)pγ(zk|Rk,<j)dzk.

zk

(5)

Here, pψ(r|zk) is the probability of the token conditioned on zk, which is parametrized by the language head of the LLM. pγ(zk|Rk,<j) is the distribution of zk conditioned on the partial information (i.e., before the j-th token) of the reasoning segment Rk, whose parameters are denoted as γ.

Furthermore, when considering the posterior distribution in (3), we can derive the Evidence Lower Bound (ELBO) for log p(r|Q,R<i) as follows:

pψ(r|zk)pγ(zk|Rk,<j) pϕ(zk|Q,Z<k)

log p(r|Q,R<i) ≥Ez

k∼pϕ(·|Q,Z<k) log

(6)

k∼pϕ(·|Q,Z<k)[log pψ(r|zk)] − KL[pϕ(·|Q,Z<k)

=Ez

###### ∥ p(·|Rk,<j)

].

Posterior of zk

Prior of zk

The ELBO of log p(r|Q,R<i) naturally leads to a variational auto-encoding framework of latent reasoning. For the LLM, its autoregressive module with the latent reasoning head, whose parameters are ϕ, works as the encoder embedding the question and the previous latent reasoning states to the current latent reasoning state. Its language head ψ works as the decoder generating tokens based on the latent reasoning states. In (6), the first term corresponds to the latent reasoning loss, measuring the likelihood of reasoning tokens given the sampled latent reasoning states. The second term is the KL divergence between the posterior and prior distributions, regularizing the posterior distribution.

Considering the ELBO in (6) with the loss in (2), we can learn a variational latent reasoning model by solving the following optimization problem:

La i=1

EZ∼p

ϕ(·|Q)[log pθ′(ai|Q,Z,A<i)]

max

θ′={ϕ,ψ},γ

LLatentanswer

###### + K

###### Ez

###### k∼pϕ(·|Q,Z<k)[log pψ(rj|zk)]

k=1 rj∈Rk

LLatentreasoning − K

KL[pϕ(·|Q,Z<k)∥pγ(·|Rk)]

.

k=1

Regularizer of posterior

(7)

Here, Z ∼ pϕ(·|Q) is implemented by sequential sampling shown in (3). pγ(·|Rk) denotes the distribution of zk conditioned on Rk, whose parameters are denoted as γ. Similar to (Dilokthanakul et al., 2016; Tomczak and Welling, 2018; Xu et al., 2020), we learn the prior distribution of the latent reasoning state pγ together with the latent reasoning model. Obviously, this learning problem is analogous to that of explicit CoT in (1), which maximizes the likelihoods of both reasoning and answer tokens. Furthermore, the posterior distribution pϕ is optimized under the guidance of pγ.

Remark. In (7), we replace the p(·|Rk,<j) in (6) with pγ(·|Rk). Such a modification leads to a “stable” prior distribution invariant with the selection of the reasoning token rj ∈ Rk, which is reasonable in practice. Firstly, as aforementioned, an ideal latent reasoning state zk should cover the information of Rk, so that modeling its distribution conditioned on Rk rather than Rk,<j can impose more information to the model, leading to better regularization. Secondly, if the distribution pγ changes with respect to the selection of the reasoning tokens, we would have to recompute the KL divergence in (7) for each rj ∈ Rk, which would cause significant training overhead. Therefore, applying pγ(·|Rk) is reasonable and efficient in practice.

###### 3.3 Implementing the Framework via ReGuLaR

As analyzed in Section 3.1, the crux of learning latent reasoning models lies in guiding the posterior of latent reasoning states with less information loss. Therefore, designing and learning the semantically meaningful prior distribution pγ is critical for our variational latent reasoning model.

Inspired by recent advances (Xing et al., 2025; Wei et al., 2025) establishing visual representations as compact carriers of textual information, we propose ReGuLaR, a rendered CoT-guided variational latent reasoning method, to implement the VAE framework in (7).

Textual CoT

Textual Answer

[Figure 4]

[Figure 5]

CE Loss per Token CE Loss per Token

|Language Head|
|---|

- As illustrated in Figure 3, ReGuLaR parametrizes the prior distribution of latent reasoning states based on the visual representation of rendered CoT. Formally, we can render K segments of the reasoning chain R into K images and extract their visual representation as follows: For k = 1,...,K,

| | | |……| |
|---|---|---|---|---|
|Latent Reasoning Head<br><br>[Figure 6]| | | | |

……

……

[Figure 7]

###### Autoregressive Decoder

|LoRA|
|---|

……

###### z1 z2 z3 zK

……

KL Divergence per State

……

eQ1 eQ2 eA1 eA2 eAL

eQL

a

q

……

zˆ1 zˆ2 zˆ3 zˆK

- 1) Rendering: Ik = f(Rk),
- 2) Embedding: vk = v(Ik),
- 3) Adaptation: zˆk = gγ(vk).

| | | | |
|---|---|---|---|
|Embedding Head| | | |

| | | | |
|---|---|---|---|
|Embedding Head| | | |

| | | | | |
|---|---|---|---|---|
|Adapter| | | | |

(8)

[Figure 8]

| | |
|---|---|
|Tokenizer| |

| | |
|---|---|
|Tokenizer| |

Visual Encoder

Here, f is the predefined rendering function, which maps an arbitrary token sequence to an image with a size H × W. v is the pretrained visual encoder, which transforms pixel-wise images into visual representations. We employ the trainable adapter gγ : Rd

[Figure 9]

[Figure 10]

[Figure 11]

Rendered CoT

Figure 3 . Illustration of the proposed ReGuLaR. Here, only the latent reasoning head, adapter, and LoRA module are trainable. The blue arrows “→” indicate deterministic outputs, while the red arrows “→” indicate probabilistic outputs achieved by sampling. The special token “###” triggers the transition from the reasoning process to answer generation during inference.

 →

v

h to map visual representations V = [v1,...,vK] to the proposed latent reasoning space. In this study, we directly adopt the optimal rendering configuration identified in Glyph (Cheng et al., 2025) for f, which maximizes the semantic density. We implement v as the pretrained visual encoder in DeepSeek-OCR (Wei et al., 2025) since it has been architecturally optimized for visual-text compression, enabling it to encode high-resolution and text-dense inputs into compact representations with minimal semantic loss. The adapter gγ is instantiated as a multi-layer

Rd

perception (MLP) with parameters γ. Notably, as f and v are frozen in our work, we can pre-compute these visual representations offline before training, significantly reducing computational overhead.

As a result, given the reasoning chain R, we first segment it into K parts1 and then model pγ(·|Rk) as a normal distribution N(zˆk,I) for k = 1,...,K, whose mean is determined by (8) and variance is fixed as an identity matrix.

Accordingly, for k = 1,...,K, the KL divergence in (7) becomes

KL[pϕ∥pγ] = ∥µk − zˆk∥22 + ∥σk∥22

2 − log |diag(σk)|. (9) Following the previous work (Tan et al., 2025), we approximate the KL divergence during training as follows: Eϵ∼N(0,I)[∥µk + σk ⊙ ϵ − zˆk∥22] − log |diag(σk)|. (10)

- 1

- 2

Notably, the LLM trained by ReGuLaR still follows the standard latent reasoning workflow initiated solely by the textual input question: the model utilizes the trained latent reasoning head to generate latent reasoning states until the special termination token is encountered, which acts as a signal triggering the language head to decode the final answer. Algorithms 1 and 2 in Appendix A.3 have presented training and inference schemes of ReGuLaR in detail.

###### 3.4 Advantages over Existing Methods

Unlike the token grouping strategy in CoLaR (Tan et al., 2025), ReGuLaR renders reasoning chains into images to preserve semantic integrity. The visual representations of these rendered images provide better regularization than the aggregation of grouped token embeddings, resulting in less information loss. As previously noted, the inference phase remains consistent with standard latent reasoning, accepting pure text inputs and imposing no extra computational cost.

Moreover, non-textual elements (e.g., charts, graphs, and diagrams) can also be rendered and encoded alongside text. Therefore, in addition to mitigating textual information loss, ReGuLaR natively supports the use of multi-modal information in its latent reasoning processes. In complex tasks involving multi-modal reasoning information, this advantage enables ReGuLaR not only to outperform existing latent reasoning methods but also to surpass explicit textual CoT.

### 4 Experiments

###### 4.1 Experimental Setup

Datasets. Following previous work (Tan et al., 2025), we primarily train and evaluate models on GSM8KAug (Deng et al., 2023), and additionally evaluate trained models using three out-of-domain math reasoning datasets: GSM-Hard (Gao et al., 2023), SVAMP (Patel et al., 2021), and MultiArith (Roy and Roth, 2015). Meanwhile, we also train and evaluate on GSM8K-Aug-NL, the variant of GSM8K-Aug that preserves natural language explanations within reasoning processes, to explore their extreme compression capabilities. Additionally, we conduct experiments on AQUA-RAT (Ling et al., 2017) and MATH (Hendrycks et al., 2021) to verify their performance in more challenging problems.

Baselines. We employ various latent reasoning methods as baselines, including iCoT (Deng et al., 2024), CODI (Shen et al., 2025), Coconut (Hao et al., 2025), and CoLaR (Tan et al., 2025). Specifically, all these baselines are implemented following their default configurations within the unified framework provided by the state-of-the-art method CoLaR, thereby ensuring fairness and consistency.

1We regarding each sentence in R as a segment in Table 1 and analyze the impact of compression rate (i.e., |R|/K) in Figure 4b.

- Table 1 . Performance comparison on four math reasoning datasets using LLaMA-3.2-1B-Instruct as the LLM backbone. We report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

Method

iCoT 19.8±0.23 0.00±0.00 3.87±0.16 0.00±0.00 36.4±0.51 0.00±0.00 38.2±0.66 0.00±0.00 24.6 0.00 CODI 13.3±0.62 6.00±0.00 2.97±0.24 6.00±0.00 21.7±0.73 6.00±0.00 19.2±0.83 6.00±0.00 14.3 6.00 Coconut 20.5±0.68 6.00±0.00 4.86±0.30 6.00±0.00 39.8±0.71 6.00±0.00 41.4±0.69 6.00±0.00 26.6 6.00 CoLaR* 26.6±0.18 5.63±0.01 6.23±0.14 7.01±0.05 47.1±0.30 2.96±0.02 87.0±0.21 3.23±0.01 41.7 4.70 ReGuLaR 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

*Results here utilize the default maximum compression rate of 5, while comparison across varying rates is provided in Figure 4b

LLaMA-1B

###### DS-1.5B

80

14

Coconut CoLaR

12.8

ReasoningLength(#L)

50

12

CoLaR

| |
|---|

Accuracy(Acc.)

Accuracy(Acc.)

60

50

ReGuLaR

10

ReGuLaR

8

40

40

6

40

4

3.1

20

30

2

30

0

0

1 2 3 4 5 Compression Rate

1 2 3 4 5 Compression Rate

GSM8KAug

GSMHard

SVAMP MultiArith

GSM8KAug

GSMHard

SVAMP MultiArith

Average

Average

(a) Generalizability Analysis

(b) Compression Analysis

- Figure 4 . (a) Generalizability analysis using DeepSeek-R1-Distill-Qwen-1.5B as the LLM backbone, where the left panel reports Accuracy and the right panel reports Reasoning Length. (b) Compression Analysis on the GSM8K-Aug dataset using LLaMA-3.2-1B-Instruct (left) and DeepSeek-R1-Distill-Qwen-1.5B (right) as the LLM backbone, where the compression rate represents the number of explicit reasoning tokens corresponding to a single latent reasoning state.

Evaluation. We adopt standard evaluation metrics widely used in this domain (Hao et al., 2025) to assess performance, including i) Accuracy (Acc.), which evaluates reasoning effectiveness by calculating the percentage of correctly predicted answers; and ii) Reasoning Length (# L), which evaluates reasoning efficiency by calculating the number of reasoning steps per question. Specifically, all these evaluations are repeated over five independent runs with different random seeds, thereby ensuring statistical reliability.

Implementation. Unless otherwise specified, we employ LLaMA-3.2-1B-Instruct (LLaMA-1B (Grattafiori

- et al., 2024)) as the LLM backbone. Specifically, following established baselines, we keep the LLM backbone frozen and exclusively optimize LoRA (Hu et al., 2022) modules configured with r = 128 and α = 32. More details about datasets, baselines, and implementation have been provided in Appendix A.

###### 4.2 Main Results

Performance Comparison. Table 1 presents the results of various methods on four math reasoning datasets, where ReGuLaR achieves state-of-the-art performance. Specifically, compared with the strongest baseline CoLaR, ReGuLaR consistently delivers substantial accuracy gains across all datasets while simultaneously reducing the average reasoning length by approximately 35% (from 4.70 to 3.03). The results suggest that ReGuLaR successfully compresses the reasoning chain into a more compact and informative latent reasoning state sequence, underscoring both its computational efficiency and reasoning effectiveness.

Generalizability Analysis. To demonstrate the generalizability of ReGuLaR to different LLM backbones, we replace the underlying model (i.e., LLaMA-3.2-1B-Instruct) with DeepSeek-R1-Distill-Qwen-1.5B. As shown in Figure 4a, ReGuLaR consistently maintains its superiority across all datasets, achieving the highest accuracy with the shortest reasoning length. Especially on the GSM-Hard dataset, while the strongest baseline CoLaR requires an average of 12.8 reasoning steps, ReGuLaR achieves higher accuracy with only 3.1 steps.

GSM8K-Aug

###### GSM-Hard

###### SVAMP

MultiArith

70

100

50

12

Accuracy(Acc.)

60

80

10

40

8

Coconut

50

60

30

CoLaR

6

ReGuLaR

40

20

40

1B 3B 8B

1B 3B 8B

1B 3B 8B

1B 3B 8B

- Figure 5 . Scalability analysis across varying model sizes, where we employ LLaMA-3.2 (1B, 3B) and LLaMA-3.1 (8B) Instruct variants as the LLM backbones. Comprehensive results, including reasoning length, are provided in Figure 6.

- Table 2 . Extreme compression performance of ReGuLaR, where CoLaR follows its default configuration as a reference.

LLaMA-1B LLaMA-3B LLaMA-8B DS-1.5B Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

Dataset Method

GSM8K- CoLaR 18.7±0.34 13.1±0.04 31.0±0.26 14.8±0.11 31.7±0.21 13.1±0.07 16.4±0.23 15.4±0.03 24.4 14.1 Aug-NL ReGuLaR 20.2±0.71 1.00±0.00 32.6±0.43 1.00±0.00 38.6±0.33 1.00±0.00 31.3±0.09 1.00±0.00 30.7 1.00

AQUA- CoLaR 24.2±0.37 19.4±0.31 31.7±1.27 28.5±0.66 35.8±0.90 20.5±1.66 33.2±1.04 26.7±0.54 31.2 23.8 RAT ReGuLaR 37.1±0.61 1.00±0.00 39.7±0.50 1.00±0.00 41.2±0.34 1.00±0.00 41.0±0.48 1.00±0.00 39.8 1.00

CoLaR 3.65±0.13 58.1±0.29 8.03±0.25 60.4±0.19 9.06±0.20 67.7±0.59 10.3±0.22 62.6±0.21 7.76 62.2

MATH

ReGuLaR 6.62±0.18 1.00±0.00 11.8±0.03 1.00±0.00 13.9±0.10 1.00±0.00 15.6±0.14 1.00±0.00 11.9 1.00

Compression Analysis. While the strongest baseline CoLaR employs token embedding compression and ReGuLaR leverages visual-text compression, both map a sequence of explicit reasoning tokens to a single latent reasoning state. Given this commonality, we assess their performance under identical compression rates (i.e., the number of tokens condensed into one latent reasoning state). As shown in Figure 4b, although accuracy naturally decreases with higher compression rates, ReGuLaR consistently outperforms CoLaR across all settings on both LLM backbones, verifying its advantage in preserving semantic information.

Scalability Analysis. To evaluate the scaling potential of ReGuLaR, we conduct experiments across varying model sizes within the LLaMA-3 family, ranging from 1B to 8B (i.e., instruct variants of LLaMA-3.2 1B/3B and LLaMA-3.1 8B). As shown in Figure 5, ReGuLaR demonstrates strong positive scaling behavior, consistently maintaining a significant performance margin over the top-performing baselines (CoLaR and Coconut) across all model scales and datasets, demonstrating its seamless scalability and potential for broader application in large-scale foundation models.

More experimental results, including ablation studies, have been provided in Appendix B.

###### 4.3 Extreme Compression with ReGuLaR

As expressed in (8), the reasoning chain R is decomposed into K segments, which are subsequently rendered into images to yield K corresponding latent reasoning states during training, thereby impacting information preservation and compression efficiency. In Table 1, we follow the natural linguistic partition, treating each sentence as a segment. In this section, we further investigate the limit of model performance by introducing an extreme compression setting. Specifically, we directly render the entire reasoning chain into a single image, compressing all reasoning information into one latent reasoning state (i.e., K = 1). We conduct this experiment on the GSM8K-Aug-NL dataset, which preserves natural language explanations within the reasoning process, inherently yielding relatively long reasoning chains. The AQUA-RAT and MATH datasets are also incorporated to verify the performance on more challenging problems.

- Table 2 summarizes the performance of ReGuLaR under the extreme compression setting. Specifically,

- Table 3 . Performance comparison on the molecular captioning task. We report the averaged number and 95% confidence interval (±) on BLEU, METEOR, ROUGE, and Reasoning Length (# L) via five independent evaluations. Note that "w/o 2D" denotes the variant of ReGuLaR trained with the original textual reasoning chains for ablation.

Method BLEU-2↑ BLEU-4↑ METEOR↑ ROUGE-1↑ ROUGE-2↑ ROUGE-L↑ # L

GPT-4o 0.1093 0.0313 0.1682 0.2491 0.0778 0.1875 / DeepSeek-R1 0.1011 0.0297 0.2178 0.2506 0.0709 0.1722 /

CoT 0.2828±0.002 0.1804±0.002 0.3778±0.001 0.4632±0.002 0.2675±0.003 0.3914±0.001 314.825±4.143 CoLaR 0.0995±0.002 0.0400±0.001 0.1774±0.002 0.2306±0.004 0.0911±0.002 0.1900±0.003 212.347±0.660 ReGuLaR 0.3673±0.002 0.2692±0.002 0.4593±0.002 0.5319±0.002 0.3502±0.002 0.4635±0.002 1.000±0.000

LLaMA-1B

w/o 2D 0.2872±0.002 0.1845±0.002 0.3777±0.001 0.4678±0.001 0.2716±0.001 0.3962±0.000 1.000±0.000

CoT 0.3167±0.003 0.2146±0.003 0.4117±0.003 0.4948±0.001 0.3011±0.003 0.4230±0.002 316.829±3.502 CoLaR 0.1734±0.002 0.0861±0.002 0.2487±0.003 0.3381±0.002 0.1576±0.001 0.2840±0.002 200.267±5.437 ReGuLaR 0.4329±0.002 0.3394±0.002 0.5158±0.002 0.5860±0.002 0.4164±0.003 0.5208±0.002 1.000±0.000

LLaMA-3B

w/o 2D 0.3182±0.003 0.2119±0.003 0.4172±0.002 0.4940±0.002 0.2961±0.003 0.4184±0.002 1.000±0.000

CoT 0.3410±0.001 0.2378±0.001 0.4377±0.001 0.5138±0.001 0.3212±0.001 0.4403±0.001 295.736±4.105 CoLaR 0.2031±0.003 0.1385±0.002 0.2803±0.003 0.3883±0.001 0.2340±0.002 0.3449±0.002 163.655±3.981 ReGuLaR 0.4610±0.001 0.3691±0.001 0.5479±0.002 0.6094±0.001 0.4428±0.001 0.5439±0.001 1.000±0.000

LLaMA-8B

w/o 2D 0.3403±0.002 0.2364±0.001 0.4394±0.002 0.5132±0.002 0.3187±0.002 0.4367±0.003 1.000±0.000

CoT 0.2682±0.002 0.1659±0.002 0.3637±0.002 0.4469±0.001 0.2523±0.001 0.3774±0.001 344.067±6.366 CoLaR 0.0968±0.002 0.0516±0.001 0.1644±0.002 0.2149±0.004 0.0916±0.001 0.1801±0.003 580.558±9.779 ReGuLaR 0.3536±0.002 0.2545±0.001 0.4397±0.001 0.5187±0.002 0.3347±0.001 0.4514±0.001 1.000±0.000

DS-1.5B

w/o 2D 0.2672±0.002 0.1640±0.002 0.3764±0.002 0.4645±0.002 0.2499±0.002 0.3476±0.003 1.000±0.000

despite being constrained to a single latent reasoning step, ReGuLaR still outperforms the strongest baseline CoLaR across all model scales and datasets. This advantage is particularly evident on the MATH dataset, where ReGuLaR improves average accuracy from 7.76% to 11.9% while reducing reasoning length from 62.2 to 1.00, thereby underscoring its superior compression capability in complex reasoning scenarios.

### 5 Latent Reasoning Beyond Textual Domain

As discussed in Section 3.4, non-textual elements can also be rendered alongside text, making ReGuLaR support multi-modality within latent reasoning while maintaining the standard textual I/O interface. In this section, we conduct experiments to investigate the efficacy of this extended capability.

Dataset. Unlike existing multi-modal datasets that rely on image-based inputs and textual reasoning chains, we require datasets that maintain purely textual I/O while integrating multi-modal reasoning chains as intermediate bridges. To this end, we adopt the molecule captioning benchmark from MolReasoner (Zhao

- et al., 2025a), which requires the LLM to generate natural language descriptions of the given molecules (represented by SELFIES strings). In particular, while the original dataset only provides textual reasoning chains, we utilize RDKit (Landrum et al., 2013) to generate the corresponding 2D molecular graphs and combine them with the original textual reasoning chains, thereby constructing multi-modal reasoning chains.

Baselines and Evaluation. Consistent with Section 4.3, we also render the entire multi-modal reasoning chain into a single image to train ReGuLaR. The baselines include CoT and CoLaR, both of which apply the original textual reasoning chains for training due to their inherent lack of multi-modal support. The model performance is evaluated by the widely used BLEU, METEOR, and ROUGE metrics, which quantify the n-gram overlap and semantic similarity between the generated and reference captions.

Results. As presented in Table 3, ReGuLaR achieves state-of-the-art performance across all metrics and backbones. Specifically, despite being constrained to a single latent reasoning step, ReGuLaR significantly outperforms not only the strongest latent reasoning baseline CoLaR, but also the explicit CoT method, both

of which apply hundreds of reasoning steps. Notably, although CoT and CoLaR are trained using the original textual reasoning chains due to their inherent lack of multi-modal support, the comparison between our method and these two methods is fair to some extent because the 2D graphs in the multi-modal reasoning chains only convert the data format, which do not provide any additional information. To ensure a strictly fair comparison, we construct an ablation setting (denoted as “w/o 2D”) where only textual reasoning chains are rendered. Even in this setting, ReGuLaR maintains performance comparable to CoT while drastically reducing the reasoning steps. These results further validate the extreme compression capability of ReGuLaR and highlight its unique advantage of unifying textual and non-textual elements for comprehensive reasoning.

### 6 Conclusion and Future Work

In this paper, we propose a new and insightful latent reasoning paradigm that models latent reasoning within the VAE framework and learns it guided by rendered CoT. Our method significantly outperforms existing latent reasoning methods across both computational efficiency and reasoning ability, and even surpasses explicit CoT through supporting multi-modal latent reasoning.

Future work. Currently, standard benchmarks like GSM8K and GSM8K-Aug may limit the assessment of advanced reasoning capabilities because they have limited data sizes and overly simple reasoning chains. We plan to address this gap by developing a large-scale and high-quality reasoning dataset to evaluate latent reasoning methods in more demanding settings. In addition, we will further explore latent reasoning and study whether and how it can outperform explicit CoT in theory.

### References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, pages 17682–17690, 2024.

Jiale Cheng, Yusen Liu, Xinyu Zhang, Yulin Fei, Wenyi Hong, Ruiliang Lyu, Weihan Wang, Zhe Su, Xiaotao Gu, Xiao Liu, et al. Glyph: Scaling context windows via visual-text compression. arXiv preprint arXiv:2510.17800, 2025.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Yuntian Deng, Kiran Prasad, Roland Fernandez, Paul Smolensky, Vishrav Chaudhary, and Stuart Shieber. Implicit chain of thought reasoning via knowledge distillation. arXiv preprint arXiv:2311.01460, 2023.

Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step. arXiv preprint arXiv:2405.14838, 2024.

Nat Dilokthanakul, Pedro AM Mediano, Marta Garnelo, Matthew CH Lee, Hugh Salimbeni, Kai Arulkumaran, and Murray Shanahan. Deep unsupervised clustering with gaussian mixture variational autoencoders. arXiv preprint arXiv:1611.02648, 2016.

Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, and Tat-Seng Chua. Reasoning implicit sentiment with chain-ofthought prompting. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1171–1182, 2023.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: program-aided language models. In Proceedings of the 40th International Conference on Machine Learning, pages 10764–10799, 2023.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason E Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. In Second Conference on Language Modeling, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Mingyu Jin, Qinkai Yu, Shu Dong, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. The impact of reasoning step length on large language models. In Findings of the 62nd Annual Meeting of the Association for Computational Linguistics, ACL 2024, pages 1830–1842, 2024.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chain-of-thought without compromising effectiveness. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 24312–24320, 2025.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.

Greg Landrum et al. Rdkit: A software suite for cheminformatics, computational chemistry, and predictive modeling. Greg Landrum, 8(31.10):5281, 2013.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Anqi Mao, Mehryar Mohri, and Yutao Zhong. Cross-entropy loss functions: Theoretical analysis and applications. In International Conference on Machine learning, pages 23803–23828. PMLR, 2023.

Radford M Neal and Geoffrey E Hinton. A view of the em algorithm that justifies incremental, sparse, and other variants. In Learning in graphical models, pages 355–368. Springer, 1998.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2080–2094, 2021.

Subhro Roy and Dan Roth. Solving general arithmetic word problems. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1743–1752, 2015.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

Wenhui Tan, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Ruihua Song, and Jian Luan. Think silently, think fast: Dynamic latent compression of LLM reasoning chains. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Jakub Tomczak and Max Welling. Vae with a vampprior. In International conference on artificial intelligence and statistics, pages 1214–1223. PMLR, 2018.

Alex Jinpeng Wang, Linjie Li, Yiqi Lin, Min Li, Lijuan Wang, and Mike Zheng Shou. Leveraging visual tokens for extended text contexts in multi-modal learning. Advances in Neural Information Processing Systems, pages 14325–14348, 2024.

Haoran Wei, Yaofeng Sun, and Yukun Li. Deepseek-ocr: Contexts optical compression. arXiv preprint arXiv:2510.18234, 2025.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Ling Xing, Alex Jinpeng Wang, Rui Yan, Xiangbo Shu, and Jinhui Tang. Vision-centric token compression in large language model. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Hongteng Xu, Dixin Luo, Ricardo Henao, Svati Shah, and Lawrence Carin. Learning autoencoders with relational regularization. In International Conference on Machine Learning, pages 10576–10586. PMLR, 2020.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36: 11809–11822, 2023.

Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, et al. Molreasoner: Toward effective and interpretable reasoning for molecular llms. arXiv preprint arXiv:2508.02066, 2025a.

Hongbo Zhao, Meng Wang, Fei Zhu, Wenzhuo Liu, Bolin Ni, Fanhu Zeng, Gaofeng Meng, and Zhaoxiang Zhang. Vtcbench: Can vision-language models understand long context with vision-text compression? arXiv preprint arXiv:2512.15649, 2025b.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition: A theoretical perspective on chain of continuous thought. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a.

Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, et al. A survey on latent reasoning. arXiv preprint arXiv:2507.06203, 2025b.

Jin Ziqi and Wei Lu. Tab-cot: Zero-shot tabular chain of thought. In Findings of the Association for Computational Linguistics: ACL 2023, pages 10259–10277, 2023.

### Appendix

### A More Experimental Details

###### A.1 Datasets

Following previous work (Tan et al., 2025), we primarily train and evaluate our method on the GSM8K-Aug dataset (Deng et al., 2023), and additionally evaluate trained models using three out-of-domain math reasoning datasets: GSM-Hard (Gao et al., 2023), SVAMP (Patel et al., 2021), and MultiArith (Roy and Roth, 2015). Meanwhile, we also train and evaluate on the GSM8K-Aug-NL dataset, a variant of the GSM8K-Aug dataset that preserves natural language explanations, to demonstrate the extreme compression capability. In addition, we extend our experiments to the AQUA-RAT dataset (Ling et al., 2017) and MATH dataset (Hendrycks

et al., 2021) to verify the performance on more challenging problems. Here, the detailed description of the above datasets is provided below:

- • GSM8K-Aug (Deng et al., 2023): This dataset is an augmented version of the GSM8K dataset (Cobbe et al., 2021), constructed by prompting GPT-4 (Achiam et al., 2023) to extend the original training set to 385k samples. In particular, it eliminates natural language descriptions within the reasoning process, formalizing reasoning steps as mathematical expressions only.
- • GSM8K-Aug-NL (Deng et al., 2023): Similar to the GSM8K-Aug dataset, this dataset is also constructed by prompting GPT-4 (Achiam et al., 2023) to extend the original training set of the GSM8K dataset to 385k samples. The distinguishing feature is that GSM8K-Aug-NL preserves natural language explanations within the reasoning process, formalizing reasoning steps as natural language sentences. Consequently, compared to GSM8K-Aug, this dataset exhibits longer reasoning chains, with a reasoning style that more closely resembles verbose CoTs.
- • GSM8K-Hard (Gao et al., 2023): This dataset is the harder version of the GSM8K dataset, constructed by replacing the numbers in the original test set with larger numbers that are less common. In particular, this dataset serves as the out-of-domain dataset in our experiments.
- • SVAMP (Patel et al., 2021): This dataset comprises 1,000 elementary-level math word problems, designed to evaluate the robustness of models in solving fundamental mathematical problems. In particular, this dataset serves as the out-of-domain dataset in our experiments.
- • MultiArith (Roy and Roth, 2015): This dataset comprises 600 multi-step arithmetic problems, designed to evaluate the capability of models in handling tasks that require multi-step reasoning. In particular, this dataset serves as the out-of-domain dataset in our experiments.
- • AQUA-RAT (Ling et al., 2017): This dataset comprises about 100,000 algebraic word problems, where each problem is equipped with a step-by-step natural language explanation that details the logical derivation leading to the answer.
- • MATH (Hendrycks et al., 2021): This dataset comprises 12,500 highly challenging mathematics competition problems, where each problem is equipped with a comprehensive step-by-step solution.

###### A.2 Baselines

We employ various respective latent-based methods as baselines, including iCoT (Deng et al., 2024), CODI (Shen et al., 2025), Coconut (Hao et al., 2025), and CoLaR (Tan et al., 2025).

Here, the detailed description of the above baselines is provided below:

- • iCoT (Deng et al., 2024): This baseline gradually removes the intermediate reasoning steps during finetuning, thereby internalizing the reasoning process while maintaining high performance.

Table 4 . Detailed rendering configuration used in our experiments.

Parameter Value Description Canvas & Layout

Page Size 595 × 842 Standard A4 dimension (points) DPI 72 Screen resolution density Margins (X, Y) 10, 10 Minimal padding to maximize content area Background Color #FFFFFF White background Auto Crop True Crop to content bounding box

Typography

Font Family Verdana Sans-serif font for optical clarity Font Size 9 Compact size for high density Line Height 10 Tight vertical spacing Font Color #000000 Black text for high contrast Alignment LEFT Standard left-aligned text

Spacing & Indentation

Indent (First/Left/Right) 0 No indentation Spacing (Before/After) 0 No paragraph spacing Border Width 0 Borderless rendering

- • CODI (Shen et al., 2025): This baseline directly employs self-distillation to align the hidden activations of latent thoughts with CoT trajectories, thereby transferring reasoning capabilities into latent space.
- • Coconut (Hao et al., 2025): This baseline recursively utilizes the last hidden state of LLMs as latent thought, thereby functioning as the next input embedding to drive subsequent reasoning.
- • CoLaR (Tan et al., 2025): This baseline dynamically compresses embeddings of reasoning tokens and autoregressively predicts compressed embeddings, thereby supporting flexible reasoning lengths.

###### A.3 Implementation

Rendering Configuration. As expressed in (8), the rendering function Φ is parameterized by a specific configuration vector θ∗ to map an arbitrary token sequence to an image. To ensure consistent visual encoding and maximize the semantic density, we directly adopt the optimal rendering configuration identified in Glyph (Cheng et al., 2025). Specifically, we utilize the Verdana font family and set a tight layout to compact the logical topology. The detailed rendering specifications are summarized in Table 4.

Visual Encoder. As expressed in (8), rendered images obtained from the preceding stage are mapped into continuous space, transforming pixel-based inputs into visual-semantic vectors. To encode high-resolution and text-dense inputs into compact representations with minimal semantic loss, we adopt the trained visual encoder from DeepSeek-OCR (Wei et al., 2025). Specifically, this visual encoder integrates a SAM-Base backbone (80M) for fine-grained local perception (via window attention) and a CLIP-Large backbone (300M) for high-level semantic extraction (via global attention), bridged by a 16× convolutional compressor. Besides, it has been trained via the generative objective on the massive corpus of diverse optical data (including multilingual documents and synthesized charts/formulas), making it align perfectly with our requirements. In our experiments, we keep this visual encoder frozen and utilize the standard Tiny mode (resolution 512 × 512). This configuration initially maps each rendered image into a sequence of 64 visual tokens, which are subsequently aggregated via mean pooling to derive a single and compact visual-semantic representation.

- Algorithm 1 Training Scheme of ReGuLaR

Require: Training dataset D = {(Q,R,A)}, Rendering function f, Visual Encoder v, Adapter gγ, LLM

with parameters θ′={ϕ, ψ},

- 1: Stage 1: Offline Pre-computation
- 2: for each sample (Q,R,A) ∈ D do
- 3: Divide reasoning chain R into K segments {R1,...,RK}.
- 4: for k = 1 to K do
- 5: Render segment to image: Ik ← f(Rk)
- 6: Extract visual representation: vk ← v(Ik)
- 7: end for
- 8: Store pre-computed representations V = [v1,...,vK].
- 9: end for
- 10: Stage 2: Training
- 11: while not converged do
- 12: Sample batch of (Q,R,A,V ) from D.
- 13: Initialize total loss Ltotal ← 0.
- 14: Initialize latent reasoning state history Z<1 ← ∅.
- 15: for k = 1 to K do
- 16: // 1. Construct Prior
- 17: Compute prior mean via adapter: zˆk ← gγ(vk)
- 18: // 2. Sample Posterior (corresponds to (3))
- 19: Predict posterior parameters: µk,log σk ← pϕ(Q,Z<k)
- 20: Sample latent reasoning state: ϵ ∼ N(0,I), zk ← µk + σk ⊙ ϵ
- 21: Update history: Z<k+1 ← Z<k ∪ {zk}
- 22: // 3. Compute Step-wise Losses (corresponds to the last two terms in (7))
- 23: Latent Reasoning Loss: Sample a token rj ∈ Rk and compute L(reasoningk) ← −log pψ(rj | zk)
- 24: Regularizer Loss (KL): Compute L(KLk) between N(µk,σk2) and N(zˆk,I) using (10)
- 25: Accumulate: Ltotal ← Ltotal + L(reasoningk) + L(KLk)
- 26: end for
- 27: // 4. Compute Answer Loss (corresponds to the first term in (7))
- 28: Compute Lanswer ← − L

a

i=1 log pθ′(ai | Q,Z,A<i)

- 29: Ltotal ← Ltotal + Lanswer
- 30: Update Parameters: θ′,γ ← Optimizer(∇Ltotal)
- 31: end while

Hyperparameters. For the LLM backbone, we primarily leverage LLaMA-3.2-1B-Instruct (Grattafiori et al., 2024) unless otherwise specified. Specifically, we keep the LLM backbone frozen and exclusively optimize LoRA (Hu et al., 2022) modules, which are configured with r = 128 and α = 32 following established baselines. In addition, both the adapter and the latent reasoning head are instantiated as MultiLayer Perceptrons (MLPs), where the adapter maps the visual encoder’s output dimension (dv=1280) to the LLM’s hidden dimension (dh=2048) and the latent reasoning head directly operates within the LLM’s hidden dimension (dh=2048). For training, we optimize the model using the AdamW optimizer (Loshchilov and Hutter, 2019) with a weight decay of 0.01 and a learning rate of 1e-4, employing a constant schedule with a 1,000-step warmup phase. Specifically, we utilize Distributed Data Parallel across eight NVIDIA A100 GPUs to ensure training stability and efficiency. For inference, we employ a stochastic generation strategy using nucleus sampling with top-p of 0.9 and temperature of 1.0 to extract answers. Specifically, we perform five independent runs using distinct random seeds (from 0 to 4) to ensure reproducibility and reliability.

Additionally, the training and inference schemes of ReGuLaR are presented in Algorithms 1 and 2.

- Algorithm 2 Inference Scheme of ReGuLaR Require: Question Q, Trained LLM with parameters θ′={ϕ, ψ}, Max reasoning steps Kmax.

- 1: Initialization: Latent reasoning state history Z<1 ← ∅, Latent reasoning step k ← 1.
- 2: // Phase 1: Latent Reasoning
- 3: while k ≤ Kmax do
- 4: Predict posterior parameters: µk,log σk ← pϕ(Q,Z<k)
- 5: Sample latent reasoning state: ϵ ∼ N(0,I),zk ← µk + σk ⊙ ϵ
- 6: Decode representative token: rˆ ∼ pψ(r | zk)
- 7: if rˆ == <EOS_Reasoning> then
- 8: Break
- 9: end if
- 10: Update history: Z<k+1 ← Z<k ∪ {zk}
- 11: k ← k + 1
- 12: end while
- 13: // Phase 2: Answer Generation
- 14: Initialize answer sequence A<1 ← ∅, Answer generation step i ← 1.
- 15: while answer not finished do
- 16: Sample token: ai ∼ pθ′(a | Q,Z,A<i)
- 17: if ai == <EOS> then
- 18: Break
- 19: end if
- 20: Update answer: A<i+1 ← A<i ∪ {ai}
- 21: i ← i + 1
- 22: end while
- 23: return Answer A

### B More Experimental Results.

###### B.1 Ablation Studies on Rendering Configuration

In our standard implementation, we adopt the optimal rendering configuration (i.e., those summarized in

- Table 4) identified in Glyph (Cheng et al., 2025) to ensure consistent visual encoding and maximize the semantic density. Here, to verify the robustness and generalizability of our method across varying rendering configurations, we conduct ablation studies on two pivotal rendering parameters:

Font Size. We compare the performance of our proposed ReGuLaR by varying the font size from 9pt to 20pt. As presented in Table 5, ReGuLaR demonstrates remarkable stability across different font sizes. Specifically, despite substantial variations in font size, the average accuracy fluctuates only marginally (ranging from

- 44.5% to 45.6%), while the reasoning length remains largely consistent. These results indicate that our method effectively captures semantic information regardless of the text’s scale, ensuring robust performance without requiring precise font size tuning.

Rendering Density (DPI). We investigate the sensitivity of our proposed ReGuLaR to information density by adjusting the DPI from 72 to 300. As presented in Table 6, the performance remains highly consistent across this wide range. Notably, ReGuLaR achieves comparable average accuracy at both the lowest density (45.6% at 72 DPI) and the highest density (45.2% at 300 DPI). This suggests that ReGuLaR is resilient to variations in image clarity and pixel density, capable of extracting reliable features under diverse DPI settings.

- Table 5 . Performance comparison of our proposed ReGuLaR across different font sizes, where we report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

Font Size

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

20pt 34.4±0.25 4.46±0.17 8.02±0.15 4.11±0.45 48.4±0.23 2.17±0.36 88.4±0.11 2.62±0.28 44.8 3.34 16pt 33.2±0.22 4.15±0.20 8.74±0.07 5.23±0.33 48.9±0.26 1.86±0.15 87.2±0.25 2.09±0.09 44.5 3.33 12pt 34.0±0.33 4.01±0.25 8.51±0.09 4.81±0.56 51.1±0.41 2.51±0.74 87.6±0.19 2.16±0.06 45.3 3.37

9pt 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

- Table 6 . Performance comparison of our proposed ReGuLaR across different rendering density (DPI) settings, where we report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

DPI

300 33.3±0.18 3.93±0.35 8.07±0.15 3.73±0.41 50.7±0.20 2.16±0.28 88.9±0.28 2.24±0.31 45.2 3.02 144 33.4±0.07 4.87±0.36 7.67±0.06 5.05±0.39 49.9±0.42 2.19±0.31 87.5±0.12 3.07±0.19 44.6 3.80

96 32.6±0.16 4.13±0.16 7.87±0.13 5.19±0.42 48.2±0.11 2.27±0.03 88.3±0.16 2.12±0.04 44.2 3.43 72 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

###### B.2 Ablation Studies on Visual Encoder Modes

In our standard implementation, we employ the trained visual encoder from DeepSeek-OCR (Wei et al., 2025) to encode each rendered image into a sequence of visual tokens, which are subsequently aggregated via mean pooling to derive a single and compact visual-semantic representation. In particular, this trained visual encoder supports four modes: Tiny, Small, Base, and Large, corresponding to resolutions of 512 × 512, 640 × 640, 1024 × 1024, and 1280 × 1280, resulting in 64, 100, 256, and 400 vision tokens. Depending on the selected mode, the input rendered images are processed via either adaptive resizing (for Tiny/Small) or padding (for Base/Large) to align with the specific resolution. Therefore, while the input rendered images remain identical in their original resolution, selecting different modes forces the visual encoder to process them at varying internal resolutions and token budgets. Here, we conduct ablation studies across these four modes to investigate the impact of visual encoding granularity on our method.

- As presented in Table 7, the results reveal a counter-intuitive yet profound insight into the visual-semantic compression mechanism: the Tiny mode, despite internally resizing the input rendered images to 512 × 512 and utilizing only 64 intermediate tokens, achieves performance comparable to that of the high-resolution modes. We attribute this remarkable robustness to the information aggregation nature of our method. Since the intermediate visual tokens are ultimately pooled into a single and compact visual-semantic representation, it relies more on high-level semantic abstraction rather than fine-grained pixel-level details. Therefore, we adopt the Tiny mode as the default configuration in our experiments, effectively reducing the visual processing overhead by approximately 6× compared with the Large mode while ensuring that the final visual representation remains semantically rich and accurate.

###### B.3 Ablation Studies on Learning Paradigms

In our standard implementation, we train the proposed ReGuLaR via the unified objective function defined in (7), which integrates three critical components to optimize the model jointly: the answer generation loss (LLatentanswer) directly ensures answer correctness, the latent reasoning loss (LLatentreasoning) preserves semantic integrity, and the KL divergence term (LLatentKL ) regularizes the posterior distribution. To investigate the distinct contribution of each component, we conduct ablation studies by selectively removing the latter two terms.

- Table 7 . Performance comparison of our proposed ReGuLaR across different visual encoder modes, where we report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

Mode

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

Large 33.0±0.36 3.77±0.24 7.35±0.17 4.28±0.29 48.5±0.41 2.42±0.25 88.4±0.38 3.08±0.12 44.3 3.39 Base 34.3±0.46 3.86±0.48 7.67±0.09 4.60±0.36 51.6±0.33 2.23±0.22 88.6±0.41 2.41±0.32 45.5 3.28 Small 34.0±0.29 3.64±0.34 7.53±0.12 4.07±0.45 52.1±0.31 2.71±0.29 89.5±0.23 2.23±0.28 45.8 3.16

Tiny 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

- Table 8 . Performance comparison of our proposed ReGuLaR across different learning paradigms, where we report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

LLatentKL LLatentreasoning

× × 5.69±0.13 4.73±0.03 1.46±0.11 5.11±0.05 35.7±1.22 2.94±0.03 8.33±0.11 2.40±0.23 12.8 3.81 × ✓ 6.52±0.26 4.57±0.16 1.68±0.16 5.54±0.12 34.4±0.81 1.86±0.05 9.93±0.39 1.93±0.19 13.1 3.47 ✓ × 27.3±0.38 3.87±0.10 6.03±0.22 4.71±0.34 47.7±0.27 1.84±0.06 86.4±0.57 1.96±0.14 41.9 3.10 ✓ ✓ 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

Notably, the answer generation loss (i.e., the first term in (7)) is retained across all variants, as it serves as the fundamental supervision signal for the reasoning task.

- As presented in Table 8, the absence of the KL divergence term (LLatentKL ) leads to catastrophic failure

(i.e., accuracy< 14%), regardless of the latent reasoning loss. In stark contrast, introducing LLatentKL alone significantly boosts performance to 41.9%. This result empirically corroborates our analysis in Section 3.2: without the constraint imposed by the prior distribution, neither the distant supervision from the final answer nor the semantic supervision from textual reconstruction is sufficient. In addition, combining all components achieves the peak performance of 45.6%, demonstrating that the semantic richness from text reconstruction and the structural guidance from distribution regularization are synergistic and mutually indispensable.

B.4 Ablation Studies on Modeling Strategies

In our standard implementation, we model the latent reasoning process as a probabilistic transition (i.e., expressed in (3)), leveraging the latent reasoning head to predict the distribution parameters µ and log σ of the next latent reasoning state. To verify the effectiveness of this probabilistic modeling strategy, we conduct ablation studies comparing it against the deterministic variant. Specifically, this deterministic variant directly leverages the latent reasoning head to predict the next latent reasoning state, which is functionally equivalent to a greedy strategy that always selects the most likely mean vector.

- As presented in Table 9, the results demonstrate the clear superiority of the probabilistic modeling strategy. Specifically, it outperforms the deterministic variant across all datasets, achieving an average accuracy of

- 45.6%. We attribute the performance drop in the deterministic variant to the “mean collapse" phenomenon. Since the reasoning process often allows for multiple valid subsequent steps, a deterministic predictor tends to output the average of all possible outcomes to minimize the reconstruction error. This results in blurred semantic representations that fail to capture the precise logic required for complex reasoning. In contrast, our probabilistic strategy models the underlying distribution, enabling the sampling of sharp and distinct latent reasoning states that preserve semantic integrity.

- Table 9 . Performance comparison of our proposed ReGuLaR across different modeling strategies, where we report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

Strategy

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

Deterministic 32.6±0.34 3.45±0.02 7.24±0.06 5.65±0.52 48.9±0.37 2.72±0.25 88.1±0.14 2.22±0.29 44.2 3.51 Probabilistic 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

- Table 10 . Performance comparison of our proposed ReGuLaR across different regularization strategies, where we report the averaged number and 95% confidence interval (±) on Accuracy (Acc. %) and Reasoning Length (# L).

GSM8K-Aug GSM-Hard SVAMP MultiArith Average Acc. # L Acc. # L Acc. # L Acc. # L Acc. # L

Strategy

Text-based 28.3±0.20 3.53±0.31 6.53±0.11 4.21±0.03 47.5±0.18 1.97±0.03 86.9±0.37 2.07±0.13 42.3 2.95 Vision-based 34.9±0.26 3.69±0.21 8.27±0.14 4.12±0.48 50.1±0.39 2.02±0.18 89.2±0.27 2.28±0.27 45.6 3.03

###### B.5 Ablation Studies on Regularization Strategies

In our standard implementation, we decompose the reasoning chain into K segments and subsequently render them into images to yield K visual representations, which serve as dense semantic anchors to regularize the posterior distribution of the latent reasoning state during training (i.e., illustrated in Figure 1). To verify the distinct contribution of this vision-based regularization, we conduct ablation studies comparing it against the text-based variant. Specifically, this text-based variant directly aggregates the embeddings of tokens within the same segment into one textual representation to regularize the posterior distribution of the latent reasoning state, while keeping all other settings invariant.

- As presented in Table 10, the results demonstrate a substantial performance advantage for the vision-based regularization strategy. Specifically, it significantly outperforms the text-based variant across all datasets, increasing the average accuracy from 42.3% to 45.6%. We attribute this superiority to the dense information compression capability of the visual modality. The text-based variant, which relies on pooling token embeddings, tends to dilute the structural and topological details of the reasoning chain (e.g., the spatial layout of arithmetic operations), resulting in a “blurred" semantic target. In contrast, the vision-based approach compels the model to align its latent reasoning state with the corresponding rendered image, which serves as a highly compact and structured semantic anchor. This cross-modal constraint forces the model to capture the holistic logic of the segment rather than just the average meaning of its tokens, thereby providing a more robust signal for regularization. Notably, as discussed in Section 3.4, these visual representations are strictly confined to the training phase, meaning that they incur no additional cost during inference.

###### B.6 Further Scalability Analysis of ReGuLaR

- Figure 6 further illustrates the performance of ReGuLaR across varying model sizes, where ReGuLaR demonstrates strong positive scaling behavior. Specifically, compared with the top-performing baselines (i.e., CoLaR and Coconut), ReGuLaR consistently maintains the highest accuracy with the shortest reasoning length across all model scales and datasets, highlighting its seamless scalability and potential for broader application in large-scale foundation models.

### C Examples of Rendered Reasoning Chains

In the molecular captioning task, we utilize RDKit to generate the corresponding 2D molecular graph for each textual reasoning chain, thereby constructing multi-modal reasoning chains. Subsequently, these

GSM8K-Aug

###### GSM-Hard

###### SVAMP

###### MultiArith

70

100

50

12

Accuracy(Acc.)

60

80

10

40

8

50

60

Coconut

30

CoLaR

6

ReGuLaR

40

20

40

1B 3B 8B

1B 3B 8B

1B 3B 8B

1B 3B 8B

- 4

- 5

- 6

- 1B 3B 8B

- 2

- 3

- 4

- 5

- 6

- 1B 3B 8B

- 2

- 3

- 4

- 5

- 6

- 4

- 5

- 6

- 7

ReasoningLength(#L)

1B 3B 8B

1B 3B 8B

Figure 6 . Scalability analysis across varying model sizes, where we employ LLaMA-3.2 (1B, 3B) and LLaMA-3.1 (8B) Instruct variants as the LLM backbones.

reasoning chains are rendered into images, from which visual representations are extracted to regularize the posterior distribution of the latent reasoning state during training. Here, Figure 7 presents two examples of rendered reasoning chains, each shown in two variants: rendered multi-model reasoning chains with explicit 2D molecular graphs and original textual reasoning chains without 2D molecular graphs. For rendered multi-model reasoning chains that include 2D molecular graphs (e.g., Figures 7a and 7c), we position the corresponding 2D molecular graph at the top of the rendered image and annotate it below with its SMILES string. The remainder of the rendering is identical to the counterpart without 2D graphs, consisting solely of textual reasoning steps.

[Figure 12]

[Figure 13]

CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCSC(=O)CC=CCC=O

Alright, let's break this down step by step to arrive at a chemically accurate and concise description of the molecule:

---

- ### Step 1: Decode the SELFIES String and Structural Features The SELFIES string provided encodes a molecule with multiple functional groups and structural features. Here’s what the molecule contains based on its encoding and supplementary structural details:

- - **Phosphate groups** (e.g., O=P(O)(O)OP(=O)(O)O), indicating it includes an advanced functionality such as a Coenzyme A (CoA) scaffold.
- - **Carbonyl fragments** (C=O), which are commonly found in acyl chains or intermediates in metabolic pathways.
- - **Aromatic ring systems** (e.g., c1ncc2nc[nH]c2n1), indicative of purine or pyrimidine bases such as adenine or nicotinamide components in biochemical frameworks.
- - **Sulfur-containing groups** (e.g., O=CS), characteristic of CoA thioesters.
- - **Double bond fragments** (C=C), which suggest the molecule may belong to a family of unsaturated fatty acyl chains.

- ### Step 2: Interpretation of Fragments Examining the fragments reveals several telling features:

- 1. **CoA Core**: The molecule exhibits structural components consistent with a fatty acyl-CoA species, including a nucleotide-based handle (e.g., adenine-like aromatic ring) and thioester functionalities.
- 2. **Adipoyl Chain Derivative**: Adipoyl refers to a C6 dicarboxylic acid structure, and the molecule appears related to metabolic intermediates derived from adipoyl-CoA processing.
- 3. **Monounsaturated Chain**: The presence of a single double bond (C=C) suggests a monounsaturation, which aligns with cis-3-enoyl groups often characterized by this feature.
- 4. **Semialdehyde Conjugation**: Fragments like NC=O further point to semialdehyde functionalities common in biochemical dehydrogenation steps.
- 5. **Charge State**: The molecule seems to exist in its acid form rather than the deprotonated conjugate base, as described by the conjugate acid notation.

---

- ### Step 3: Analyze Significant Features Integrating all structural information and fragments:

- - CoA molecules typically serve as acyl transfer facilitators in metabolic processes.
- - The cis-3 double bond and adipoyl origin indicate the molecule is likely an intermediate in fatty acid beta-oxidation or elongation processes.
- - Conjugate acids of semialdehydes are chemically stable intermediates in biosynthetic pathways.
- ---

- ### Step 4: Natural Language Summary C ombining everything into a cohesive description:

--<answer> The molecule is a cis-3-enoyl-CoA and a monounsaturated fatty acyl-CoA. It derives from an adipoyl-CoA. It is a conjugate acid of a cis-3,4-didehydroadipoyl-CoA semialdehyde(4-). </answer>

(a) Rendered reasoning chains with 2D (example 1) (b) Rendered reasoning chains without 2D (example 1)

[Figure 14]

[Figure 15]

CCC=CCC(O)C(O)C=CC=CC=CCC=CCCCCCC(=O)O

To interpret the molecular structure represented by the provided SELFIES and structural information, we will follow a systematic approach:

- 1. **Identification of Key Features**:

- - Start with the provided structural features which indicate the absence of any rings, including aromatic rings.
- - The molecular weight is noted as approximately 362.51 g/mol, which gives us a clue about the size of the molecule.

- 2. **Analyzing the Fragments**:

- - The fragments indicate several features:
- - Multiple carbon-carbon double bonds (as evidenced by the C=C fragments).
- - Hydroxyl groups (indicated by the presence of O, especially in the O=CO and branching labels).
- - A long carbon chain (as indicated by fragments like C=C and CC featuring a sequence of connected carbon atoms).

- 3. **Constructing the Molecular Backbone**:

- - By interpreting the fragments marked with branches, we can ascertain that the structure is composed of a long, linear or slightly branched chain of carbon atoms modified with double bonds and hydroxyl groups.
- - The identification of a conjugate acid suggests that the molecule may possess acidic properties, likely due to the carboxylic acid functional group.

- 4. **Linking to Biological Significance**:

- The mention of "specialised pro-resolving mediators" indicates it is biologically relevant. The inclusion of “xenobiotic metabolite” underscores the molecule may interact with biological systems, possibly as a signaling molecule in inflammatory pathways.

- 5. **Final Identification Based on Structure and Function**:

- The structure suggested corresponds to an unsaturated fatty acid with specific stereochemistry (noted by Z and E designations). The hydroxyl groups at specific positions confirmed it is a polyunsaturated fatty acid derivative.

- 6. **Molecule Naming**:

- - Given the fragments and biological context, I deduce the complete name. Dihydroxydocosapentaenoic acid (DiHDPE) fits as it represents the
- - It carries hydroxy groups at the C16 and C17 positions on the docosapentaenoic acid backbone, producing the final connectivity and stereochemistry mentioned. After running through these points, the concise description can be articulated as follows:

<answer> The molecule is a dihydroxydocosapentaenoic acid (DiHDPE) that is (7Z,10Z,13E,14E,19Z)-docosapentaenoic acid carrying two hydroxy substituents at the 16 and 17-S positions. It is an intermediate of specialised proresolving mediators. It has a role as a human xenobiotic metabolite and a specialised pro-resolving mediator. It is a conjugate acid of a 16,17(S)-dihydroxy-(7Z,10Z,13E,14E,19Z)-docosapentaenoate. </answer>

(c) Rendered reasoning chains with 2D (example 2) (d) Rendered reasoning chains without 2D (example 2) Figure 7 . Examples of rendered reasoning chains for the molecular captioning task.

