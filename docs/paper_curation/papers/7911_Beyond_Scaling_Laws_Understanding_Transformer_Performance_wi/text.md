# arXiv:2405.08707v2[cs.LG]28Nov2024

## Beyond Scaling Laws: Understanding Transformer Performance with Associative Memory

Xueyan Niu niuxueyan3@huawei.com Theory Laboratory Central Research Institute, 2012 Laboratories Huawei Technologies Co., Ltd.

Bo Bai baibo8@huawei.com Lei Deng deng.lei2@huawei.com Wei Han harvey.hanwei@huawei.com

### Abstract

Increasing the size of a Transformer does not always lead to enhanced performance. This phenomenon cannot be explained by the empirical scaling laws. Furthermore, the model’s enhanced performance is closely associated with its memorization of the training samples. We present a theoretical framework that sheds light on the memorization during pretraining of transformer-based language models. We model the behavior of Transformers with associative memories using Hopfield networks, such that each transformer block effectively conducts an approximate nearest-neighbor search. In particular, the energy function in modern continuous Hopfield networks serves as an explanation for the attention mechanism, which we approximate with a distance-based energy function. By observing that the softmax function corresponds to the gradient of the LogSumExp function in the energy, and employing the majorization-minimization technique, we construct a global energy function designed to capture the layered architecture. We demonstrate a dependency between the model size and the dataset size for the model to achieve optimal performance, and we show that the achievable cross-entropy loss is bounded from below.

### 1 Introduction

Transformer-based neural networks have exhibited powerful capabilities in accomplishing a myriad of tasks such as text generation, editing, and question-answering. These models are rooted in the Transformer architecture (Vaswani et al., 2017) which employs the selfattention mechanisms to capture the context in which words appear, resulting in superior ability to handle long-range dependencies and improved training efficiency. In many cases, models with more parameters result in better performance measured by perplexity (Kaplan et al., 2020), as well as in the accuracies of end tasks (Khandelwal et al., 2019; Rae et al., 2021; Chowdhery et al., 2023). As a result, larger and larger models are being developed in the industry. Recent models (Smith et al., 2022) can reach up to 530 billion parameters, trained on hundreds of billions of tokens with more than 10K GPUs.

Nevertheless, it is not always the case that bigger models result in better performance. For example, the 2B model MiniCPM (Hu et al., 2024c) exhibits comparable capabilities to larger language models, such as Llama2-7B (Touvron et al., 2023), Mistral-7B (Jiang et al., 2023), Gemma-7B (Banks and Warkentin, 2024), and Llama-13B (Touvron et al., 2023). Moreover, as computational resources for training larger models increase, the size of avail-

able high-quality data may not keep pace. It has been documented that the generalization abilities of a range of models increase with the number of parameters and decrease when the number of training samples increases (Belkin et al., 2019; Nakkiran et al., 2021; d’Ascoli et al., 2020), indicating that generalization occurs beyond the memorization of training samples in over-parameterized neural networks (Power et al., 2022). Therefore, it is crucial to understand the convergence dynamics of training loss during memorization, both in relation to the model size and the dataset at hand. There has been an increasing interest in the empirical scaling laws under constraints on the training dataset size (Muennighoff et al., 2024). Let N denote the number of the parameters of the model and D denote the size of the dataset. Extensive experiments have led to the conclusion of the following empirical scaling laws (Kaplan et al., 2020) in terms of the model performance measured by the test cross-entropy loss L on held-out data:

L(N,D) =

Nc N

αN

Dc D

αD +

αD

,

where Nc,Dc,αN,αD are constants. Unfortunately, this scaling law does not explain why in many cases smaller models perform better.

In this paper, we focus on the theoretical aspects of the dependencies between the achievable performance, indicated by the pre-training loss, for transformer-based models, and the model and data sizes during memorization. It has been observed that a family of large language models tends to rely on knowledge memorized during training (Hsia et al., 2024), and the larger the models, the more they tend to encode the training data and organize the memory according to the similarity of textual context (Carlini et al., 2022; Tirumala et al., 2022). Therefore, we model the behavior of the Transformer layers with associative memory, which associates an input with a stored pattern, and inference aims to retrieve the related memories. A model for associative memory, known as the Hopfield network, was originally developed to retrieve stored binary-valued patterns based on part of the content (Amari, 1972; Hopfield, 1982). Recently, the Modern Continuous Hopfield Network (MCHN) was proposed and has been shown to exhibit equivalence to the attention mechanism (Ramsauer et al., 2020). With MCHN, the network can store well-separated data points with a size exponential in the embedding dimension. However, the MCHN only explains an individual Transformer layer and relies heavily on regularization.

Transformer-based models consist of a stack of homogeneous layers. The attention and feed-forward layers contribute to the majority of the parameters in large models and are also the key components of the attention mechanism. Furthermore, the layered structure of the transformer networks induces a sequential optimization, reminiscent of the majorizationminimization (MM) technique (Ortega and Rheinboldt, 1970; Sun et al., 2016), which has been extensively utilized across domains such as signal processing and machine learning. Using the MM framework, we construct a global energy function tailored for the layered structure of the transformer network.

Our model provides a theoretical framework for analyzing the performance of transformer-

based language models as they memorize training samples. Large language models only manifest capabilities for certain downstream tasks once the training loss reaches a specific threshold (Du et al., 2024). In practice, the training of large language models is terminated

when the loss curves plateau. On the one hand, the validation loss offers valuable insights for budgetary considerations; it has been observed that even after training on up to 2T tokens, some models have yet to exhibit signs of saturation (Touvron et al., 2023). On the other hand, implementing early stopping can potentially compromise the generalization capabilities of the models (Murty et al., 2023). In Appendix F, we include a series of experiments utilizing GPT-2, vanilla Transformer, and OpenELM models on various data. The experimental outcomes provide evidence to support our theoretical results. We believe this work offers valuable theoretical perspectives on the pre-training of large language models.

Our Contribution: (1) We take a new perspective by studying Transformer behavior using associative memories with Hopfield networks. We reveal the underlying connection between the attention mechanism and nearest-neighbor search. (2) We approximate the continuous Hopfield network using a distance-based energy function, excluding additional regularization terms. By recognizing that the softmax function corresponds to the gradient of the LogSumExp function, we employ the majorization-minimization technique to construct a global energy function to accommodate the layered architecture of the Transformer. (3) Using our theoretical framework, we characterize the dependencies between pre-training loss, model size, and dataset during memorization for transformer-based language models.

### 2 Related work

Scaling laws. Empirical evidence suggests that the performance of models increases as both the size of the models and the volume of training data scale up (Kaplan et al., 2020; Khandelwal et al., 2019; Rae et al., 2021; Chowdhery et al., 2023). Intensive experiments on transformer-based large language models have also been conducted to explore neural scaling laws under various conditions, including constraints on computational budget (Hoffmann et al., 2022b), data (Muennighoff et al., 2024), and instances of over-training (Gadre et al., 2024). In these analyses, a decomposition of the expected risk is utilized, leading to the following fit:

A Nα

B Dβ

Lˆ(N,D) = E +

+

, (1)

where N and D denote the number of parameters of the model and the size of the training data respectively. For Chinchilla models, the fitted parameters are (Hoffmann et al., 2022a)

α = 0.34, β = 0.28, E = 1.61, A = 406.4, B = 410.7.

A line of research concerns the generalization of over-parameterized neural networks (Belkin et al., 2019; Nakkiran et al., 2021; Power et al., 2022). Recent experiments show that over-trained Transformers exhibit inverted U-shaped scaling behavior (Murty et al., 2023), which is not explained by the empirical scaling laws. Further discussions on the relationship between our method and the Chinchilla scaling laws are deferred to Section 6.

Energy-based models. Energy-based models (LeCun et al., 2006), motivated by statistical physics, have become a fundamental modeling tool in various fields of machine learning over the past few decades. The central idea is to model the neural network through a parameterized probability density function pθ(x) for x ∈ Rn and to express the distribution in terms of a learnable energy function Eθ(x) : Rn  → R whose parameters correspond to the

model’s parameters as pθ(x) = exp(−ZEθ(x))

. Here, Zθ = exp(−Eθ(x)) dx is the normalizing constant known as the partition function.

θ

Hopfield models. Classical Hopfield networks (Amari, 1972; Hopfield, 1982) were introduced as paradigmatic examples of associative memory. The network’s update dynamics define an energy function, whose fixed points correspond to the stored memories. An important indicator is the number of patterns that the model can memorize, known as the network’s storage capacity. Modifications to the energy function (Krotov and Hopfield, 2016; Demircigil et al., 2017) result in higher storage capacities (see Table 1 in Appendix B). The original model operates on binary variables, and continuous Hopfield Networks have been developed later (Hopfield, 1984). The modern continuous Hopfield network (MCHN) (Ramsauer et al., 2020) connects the continuous formulation with the attention mechanism by introducing a specific model with a softmax activation function. Given an input (e.g., a prompt), the Hopfield layer retrieves a memory by converging to a local minimum of the energy landscape, and the update rule has a nice correspondence to the query-key-value mechanism in attention. Krotov (2021) proposes a Hierarchical Associative Memory (HAM) model with a global energy function for layered networks, as opposed to energy functions for individual layers. Further discussions on the relationship between the energy function utilized in this paper and other energy functions found in the literature on Hopfield networks is detailed in Section 6.

### 3 System model

We consider tokenized training samples D = {s1,s2,...,sd}, where each element is a sequence of tokens whose length is bounded by a number Tmax ∈ N. Let D = {s˜1,s˜2,...,s˜d′} be the set of held-out validation samples. Details on the pre-processing of the dataset is described in Appendix E. The size D ∈ N of the dataset is proportional to the number of samples d ∈ N. Let demb ∈ N be the embedding dimension of the tokens, so each input sequence has n = Tmaxdemb dimensions. Let N ∈ N be the number of parameters in the attention layers and the feed-forward layers, which constitute most of the parameters in the Transformer model. Suppose there are l layers, then

Aldemb Tmax

N ≈ Ald2emb =

n, D ≈ Tmaxd. (2)

for some constant A (see Appendix E). We use a generic distance metric d(·,·) in Euclidean space, which in practice can often be specified as the Euclidean norm.

#### 3.1 Associative memories

We consider models trained with a causal language modeling objective. Given an input sequence of tokens s≤t = (s1,s2,...,st), the l-layer Transformer outputs a distribution over the next token st+1. The transformer models are trained to maximize the log-probability of the correct token st+1 given s≤t. During pre-training, input texts are segmented into sequences of length Tmax, and the model develops the ability to generate desired content, such as predicting the correct continuations. Thus, the sequences can be viewed as patterns in the setting of associative memories, where stored patterns (e.g., sequences) can be retrieved

using partial contents of the patterns. Since each prediction may have access to a varying number of preceding tokens, the sequences padded to a length of Tmax earlier may have less amount of information for the associative memory retrieval. As demonstrated in (Svete and Cotterell, 2024; Svete et al., 2024), a Transformer can be modeled as a representationbased n-gram model with relative small n. Consequently, only a small number of padded sequences are affected by this discrepancy due to non-uniform lengths, and it does not significantly influence the collective behavior as analyzed within the framework of statistical physics. We note that for non-causal objectives, including masked token modeling and sequence-to-sequence modeling, similar logic can be applied within the latent space.

It has been observed that the models tend to memorize patterns from the training data

- (Carlini et al., 2021; Biderman et al., 2024). Empirical studies on large language models have shown that the larger the models are, the more they tend to memorize training data
- (Carlini et al., 2022; Tirumala et al., 2022). This memorization allows the models to learn important patterns, such as world knowledge (Hsia et al., 2024), individual words (Chang and Bergen, 2022), and linguistic structure (Chang and Bergen, 2024). In light of these findings, we make the following assumption regarding memorization. As passing the text data through an embedding layer reduces their correlation, we posit that the Transformer blocks serve to store the resulting latent representations, which are extracted from the sequences once they are embedded.

- Assumption 1 During the pre-training process, the model memorizes the (latent) training samples D as patterns {ρ1,ρ2,...,ρd}, where ρi ∈ Rn for i = 1,2,...,d.

To be economical with notations, we use D to directly address the patterns D = {ρ1,ρ2,...,ρd}. By memorizing the samples, we mean that the patterns are stored within the model and can be retrieved when provided with an adequate prompt. Specifically, we follow the definitions in (Ramsauer et al., 2020) for pattern storage and retrieval.

Definition 1 For every pattern ρi, denote by Bi := {x ∈ Rn : d(x,ci) ≤ ri} an n-ball such that ρi ∈ Bi. The pattern ρi is said to be stored if there exists a single fixed point ρi∗ ∈ Bi to which all points x ∈ Bi converge, and Bi ∩ Bj = ∅ for i ̸= j. Such Bi is said to be associated to the pattern ρi, and we denote Bi ∼ ρi. The pattern ρi is said to be retrieved if the converged point is ϵ-close to the fixed point ρi∗.

- Remark 1 In the work of Saha et al. (2023), a collective attraction mechanism is introduced to address the challenge of partial cluster assignments using associative memory.

When the intersection of clusters Bi and Bj is non-empty (Bi ∩ Bj ̸= ∅), analogous relaxations to those presented in Equation (7) of (Saha et al., 2023) can be made.

Without loss of generality, we assume ci = ρi. We also assume that the small set of heldout test samples exhibits the same patterns as those in the training set. In practice, the validation samples are randomly selected from the same dataset as the training samples, preserving the distribution.

- Assumption 2 We posit that the latent representations derived from the validation set, after being processed through an embedding layer, are stored in a manner analogous to those extracted from the training set. Specifically, for every element in the set of latent patterns

D corresponding to the validation data, there exists an Bi for some i ∈ [d]. Consequently, we assume that D ⊂ D.

#### 3.2 Transformer blocks

Transformer-based models, originated by Vaswani et al. (2017), are often made of a stack of homogeneous layers. The multi-head attention and feed-forward (FF) layers account for most of the parameters in the model. Appendix E provides more details using GPT-2 as an example.

Attention mechanism. The attention mechanism arguably contributes most to the overall performance of the transformer models. The attention mechanism takes three matrices WK ∈ Rdemb×dk,WQ ∈ Rdemb×dk, and WV ∈ Rdemb×dv as weights that can be interpreted as keys, queries, and values. Setting dv = demb facilitates the inclusion of residual connections. In a single update, an attention matrix is obtained using the update rule Attention(Q,K,V ) = V · softmax(QKT/√dk).

Feed-forward layers. It has been shown that the FF layers operate essentially as keyvalue memories (Geva et al., 2020) such that FF(x) = f(x·KT)·V, where K,V are parameter matrices and f is a non-linear activation function such as ReLU. The FF layers can be merged into the attention without degrading the Transformer’s performance (Sukhbaatar

- et al., 2019). Thus, the attention layer and the FF layer can be conceptually integrated into a unified transformer layer.

As the model scales, the attention and FF layers, being stacked, constitute the majority of the model’s parameters. Also, since the fundamental operations of the Transformer are the attention and FF layers, we consider the number of parameters N in these layers, which is almost proportional to the square of the embedding dimension. The ratio depends on the number of layers and the hidden dimensions of the transformer blocks. In the current work, we do not consider other modifications such as lateral connections, skip-layer connections, mixture of experts, mixture of depths, routing, or other compressive modules such as (Xiong et al., 2023; Fei et al., 2024; Munkhdalai et al., 2024).

### 4 A global energy function

For the attention layer, we employ an energy function that does not rely on additional regularization terms based on a distance metric. We then adapt this function to the layered transformer blocks using the majorization-minimization technique. For reference, related energy functions for Hopfield networks are listed in Table 1 in Appendix B. In particular, let M := (ρ1,ρ2,...,ρd), the energy function for the modern continuous Hopfield network (Ramsauer et al., 2020) is

maxi ∥ρi∥2 2

- 1

- 2

EMCHNβ (x) = −LogSumExp(β,MTx) +

xTx + β−1 log d +

, where

d

LogSumExp(β,y) := β−1 log

exp(βyi) , x ∈ Rn, y = (y1,...,yd) ∈ Rd.

i=1

It can be readily observed that the negative LogSumExp function was adapted from (Demircigil et al., 2017). However, in the continuous domain, the negative LogSumExp function is not convex, making it a less suitable candidate for the energy function. The MCHN energy

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

(a) −LogSumExp (b) Regularizations

[Figure 6]

(c) MCHN energy (d) Proposed energy

Figure 1: Left: Energy landscapes for a set of 2-dimensional patterns D = {(−2,−0.5),(0.2,−0.3),(1.5,1.5)}. (a) The negative LogSumExp function with β = 1, as an extension of (Demircigil et al., 2017). (b) The regularization

i∥2

terms 21xTx + β−1 log d + maxi ∥ρ

2 in the MCHN energy. (c) The MCHN energy EMCHN1 (x). (d) The layer-wise energy (4) with squared Euclidean norm. Right: Energy landscapes for a set of 1-dimensional patterns D = {−2,0,1}. The orange curves correspond to the MCHN energy with β = 1,2.

then adds regularization terms to create a convex energy function. These regularization terms involve both the max norm of the input and the number of patterns.

#### 4.1 A layer-wise energy function

Instead of designing different regularization terms, we apply an energy function by considering an auxiliary function

d(x,ρi), (3)

g(x) := min

1≤i≤d

which corresponds to the nearest neighbor search over the the set of patterns D. So it holds that g(x) ≥ 0, with g(x) = 0 if and only if x ∈ D = {ρ1,...,ρd}. According to Assumption 1, the model has memorized the patterns through pre-training; thus, the inference corresponds to a search algorithm based on some distance d(·,·). We use the squared Euclidean 2-norm d(x,y) = ∥x − y∥2 in the sequel. We consider a function E(x) which also takes the form of LogSumExp:

d

exp(−d(x,ρi)) . (4)

E(x) = −log

i=1

It is worth noting that the softmax function is the gradient of the LogSumExp function. So the Transformer integrates the search over layers. By summing up the negative distance between x and each stored pattern, the function assigns smaller values to points closer to the patterns. The distance-based energy function (4) with an inverse temperature has been applied by Saha et al. (2023) within the context of clustering. This energy function is wellsuited to a more generalized framework, namely the universal Hopfield networks (Millidge et al., 2022). By replacing the dot product in the MCHN energy with the distance metric, E(x) achieves similar goal without additional regularization. As shown in Figures 1a and 1b, as an extension of (Demircigil et al., 2017), the negative LogSumExp is not convex in the real domain, so regularization terms are applied in MCHN. Figures 1d and 1c show that the landscape of the proposed energy resembles that of the MCHN energy. In (Ramsauer

- et al., 2020), it is shown that EMCHN induces stationary points near the stored patterns. Here, the energy function E(x) serves as a smooth surrogate of the desired function g(x) in

(3), therefore also demonstrates the retrieval ability.

- Proposition 1 Given D = {ρ1 ...,ρd}, the layer-wise energy E(x) satisfies g(x) − log d ≤ E(x) ≤ g(x).

The proof of Proposition 1 is due to Lemma 3 and is deferred to Appendix C. Furthermore, we show that E(x) is close to the MCHN energy, as delineated below.

- Proposition 2 Let β = 2 we have

|E(x) − (2EMCHNβ=2 (x) − log d)| ≤ max 1≤i≤d

∥ρi∥2 − min

1≤i≤d

∥ρi∥2.

The proof is given in Appendix C. Fig. 1 provides visualizations for the two propositions using low dimensional patterns. The following result is a direct consequence of the aforementioned inequalities.

- Proposition 3

∥ρi∥2 ≤ g(x) − 2EMCHNβ=2 (x) ≤ max 1≤i≤d

∥ρi∥2 − max 1≤i≤d

∥ρi∥2 − min

∥ρi∥2 + log d.

min

1≤i≤d

1≤i≤d

Since the energy function (4) and the MCHN energy both approximate the search for the nearest pattern (desired stationary point), according to Theorem 4 in (Ramsauer et al., 2020), in each transformer layer, the probability density of the transformer layer, corresponding to the retrieval, is p(x) = Z1 exp(−E(x)|Ω), where Z is the normalizing factor, Ω = di=1 Bi, and Bi is as defined in Definition 1. We assume that Bi is centered at the i-th pattern. We make the following assumption on the samples, such that the (latent) patterns are well-separated.

- Assumption 3 Passing the input through an embedding layer reduces the correlation between the original samples. Therefore the patterns in the latent space in D are wellseparated, i.e., Bi ∩ Bj = ∅,∀1 ≤ i < j ≤ d.

Under Assumption 3, the energy function, confined in Ω, can be replaced by the nearest neighbor search g(x). So the probability density is

p(x) =

1 Z

exp(−g(x)). (5)

#### 4.2 The layered structure

As discussed in the related works, most Hopfield models only handle a single hidden layer, whereas SoTA transformer-based models often consist of a stack of homogeneous blocks of attention and FF layers. To model the multi-layered structure of Transformers, we employ a technique known as majorization-minimization (MM) (Ortega and Rheinboldt, 1970; Sun et al., 2016), which aims to accelerate optimization using surrogate convex functions. We argue that the layered structure serves the same purpose when the patterns memorized by all layers encompass the set of learned representations.

We divide the set of samples into D = ∪li=1Di, where Di = {ρi1,ρi2,...,ρidi}. Then, the energy function for each layer can be written as

1 Zt

d(x,ρtj).

exp(−gt(x)), where gt(x) := min

Et(x) =

1≤j≤dt

Denote by x(0) the embedding vector input into the first transformer layer and x(t) ∈ Rn the output of the t-th layer for t = 1,2,...,l. Let Et(x) be the energy function associated with the Hopfield model of the t-th layer, then the sequential structure of the transformer network is achieved by forwarding the output x(t−1) to the t-th layer as input, i.e.,

Et(x), Xt = {x ∈ Rn : d(x,x(t−1)) ≤ δt}, t = 1,...,l (6)

x(t) = arg min

x∈Xt

where the retrieved fix point attractor in the t-th layer is δt-close to x(t−1) in d(·,·) for some δt > 0. Such sequential optimization step is equivalent to the MM technique where every minimization step locally approximates the objective function. In particular, (6) corresponds to the surrogate function Eq. (3) in (Sun et al., 2016). Therefore, we define a global energy function

Eglobal(x) := −LogSumExp((−E1(x),−E2(x),...,−El(x))). (7)

Eglobal(x) is continuous but not convex. As opposed to the HAM (Krotov, 2021), the global energy function is not a linear combination of the component energies. According to Lemma 3, we have

Ei(x) − log l ≤ Eglobal(x) < min

min

Ei(x). (8)

1≤i≤l

1≤i≤l

So Et(x)|x∈Xt ≥ Eglobal(x)|x∈Xt + ct as in Eq. (2) in (Sun et al., 2016). The probability density function corresponding to the layered transformer network can then be written as

pθ(x) =

1 Zθ

exp(−Eglobal(x)), x ∈ Ω (9)

where θ denotes the model’s parameters and Zθ is the normalizing constant.

### 5 Cross-entropy loss

We now proceed to analyze the cross-entropy loss, a metric that quantifies the divergence between predicted probabilities and actual labels, and is widely utilized for training Transformer models.

#### 5.1 A lower bound

The attention mechanism encompasses a softmax operation that generates a probability distribution p ∈ ∆n. In practice, the final softmax output is subsequently input into a taskspecific layer to facilitate downstream tasks, such as predictions and classifications. The attention softmax influences the model’s ability to understand and process the input data, which in turn affects the output probabilities that are used to calculate the cross-entropy loss. In essence, the attention softmax indirectly influences the cross-entropy loss by shaping the model’s predictions. Consequently, we evaluate the alignment between the final softmax output of the transformer blocks and the target distribution. We demonstrate that the crossentropy loss can be articulated through the logarithm of the partition function of the model’s distribution. This formulation reveals how the allocation of attention weights is contingent upon the learned patterns, thereby establishing a relationship between the characteristics of the training data and the model size, which is pivotal for attaining optimal performance.

Let us consider the cross-entropy loss on the validation set D. Generally, the crossentropy loss is the negative log-likelihood computed over a mini-batch. Since we are considering un-batched validation samples, the loss is normalized by the size d′. According to (8), there exist a layer t such that Eglobal(x) is close to Et(x), i.e.,

Eglobal(x) = Et(x) − log l + c(x), c(x) ∈ C∞(Rn) (10)

such that 0 ≤ c(x) < log l. To simplify, we further assume that c(x) = c ∈ [0,log l) is constant. Under Assumption 1, the target distribution, which encodes all the patterns in D, is given by

d

piδ(x − ρi), x ∈ Rn

pD(x) =

i=1

where δ(·) is the Dirac delta function such that δ(x) = 0,∀x ̸= 0 and pi = Pr(x = ρi) is the probability mass assigned to pattern ρi for i = 1,2,...,d. Suppose the data points

are homogeneous, i.e., pi = d1, then PD(x) = d1 di=1 δ(x − ρi), and the corresponding test samples D induces

1 d′

P D(x) =

d′

δ(x − ρσ(i)), σ(·) ∈ Sym([d]). (11)

i=1

- Proposition 4 Let L be the cross-entropy loss of the above model, then

1 Zt ≥ 1, where c ∈ [0,log l).

L ≈ log Zt +

The proof is deferred to Appendix D.1. Note that the empirically obtained loss function (1) for the Chinchilla model converges to Lˆ(N,D) = 1.61 as N → ∞ and D → ∞, which corroborates our theory that L(N,D) ≈ log Zt + Z1

≥ 1, with minimum obtained when Zt = 1.

t

- 5.2 Interdependency between model size and training data Next, we explore the optimal balance between model size and data during memorization.

Let Bt(x) denote the n-ball with radius t centered at x. Let An−1 = 2Γ(πn/n 2

2) represent the hyper-volume of the (n − 1)-dimensional unit sphere.

∞

r

tn−1e−t dt are the lower and upper incomplete gamma functions. Then

tn−1e−t dt, Γ(n,r) =

γ(n,r) =

r

0

exp(−∥x − ρi∥2) dx =

x∈Bi

=

=

exp(−∥x − ρi∥2) dx =

exp(−∥y∥2) dy

∥x−ρi∥<ri

∥y∥<ri

ri

ri

e−t2Hn−1(∂Bt)dt (12)

e−t2 dHn−1dt =

0

0 ∂Bt(0)

2πn2 Γ(n2)

ri

ri

γ(n,ri) Γ(n2)

tn−1e−t2 dt = 2πn2

e−t2An−1tn−1 dt =

.

0

0

We take a closer look at the layer partition function, which gives us Zt =

d(x,ρti)) dµ

exp(−gt(x)) dµ =

exp(−min

i

x∈Ω

x∈Ω

d

d

γ(n,ri) Γ(n2)

exp(−∥x − ρti∥2) dx (=12) 2

πn2

=

, (13)

i=1 x∈Bti

i=1

where ri is the radius of Bi. According to Lemma 5, we have e−riVn(ri) ≤ 2πn2

γ(n,ri) Γ(n2)

exp(−∥x − ρi∥2) dx ≤ Vn(ri),

=

x∈Bi

where Vn(r) = πn2 rn/Γ(1 + n2) is the hyper-volume of the n-dimensional ball of radius r. Note that the volume of the unit ball Vn(1) in higher dimensions decreases fast with respect to the increase in dimensionality. The stability of the probabilities is attributed to the application of normalization operators, including LayerNorm (Xiong et al., 2020) and RMSNorm (Zhang and Sennrich, 2019), which regulate the distribution of activations. The gamma function can be approximated using Stirling’s approximation (Appendix D.2) for large values of its argument, which gives us

πn/2 2π(n/2) n/e2

1 √nπ

Vn(1) ≈

n/2 =

2πe n

n 2

.

For Vn(r) to have a volume of O(1), the radius r must be approximately n/(2πe) asymptotically. Bringing r = n/(2πe) to (13), we get

d · Vn( 2nπe) exp( 2nπe) ≤ Zt ≤ d · Vn(

n 2πe

). (14)

According to (2), N ≈ AldT emb

##### n and D ≈ Tmaxd. Therefore, for Zt to reach Zt = 1, we need N = O(D2) for well-separated patterns D. The following proposition summarizes the result.

max

- Proposition 5 During memorization of well-separated patterns learned from the data, to minimize the cross-entropy loss, the optimal balance between model size N and data size D is N = O(D2).

In Table 2 in Appendix B, we compare the reported cross-entropy loss of various transformer-based models in the literature. Usually, a family of models ranging in a variety of sizes is reported, and we select the largest ones. We observe that similar cross-entropy loss is achieved across a wide range of architectural shapes (including depth, width, attention heads, FF dimensions, and context lengths). Nevertheless, the pre-training cross-entropy losses all satisfy L > 1.

- Remark 2 We remark that some models add auxiliary regularization terms such as the z-loss (Chowdhery et al., 2023; Yang et al., 2023) during their training. In these cases, the scaling laws should take into consideration the additional terms. Also, modifications to the transformer blocks, such as additional layer normalization may contribute to the lower bound of the cross-entropy.

#### 5.3 Summary of experimentation

In Appendix F, we perform a series of experiments to validate our theoretical assumptions and results. Below is a concise summary of these experiments.

Evaluation of the radius with GPT-2 In Appendix F.1, we evaluate the radius r in Zl of a 24-layer pre-trained GPT-2 medium model. Our aim is to validate the hypothesis regarding the radius of patterns in Section 5. We randomly sample 100K chunks, each containing 256 tokens, from the OpenWebText dataset and record the activation vectors from the l-th layer. The distance between each activation vector and its nearest neighbor is computed using the Euclidean norm.

• The experiment found that the distances between activation vectors in the latent space are approximately equal to the hypothesized magnitude of 2 n/2πe in (14).

Training vanilla Transformers In Appendix F.2, we train vanilla Transformers on the Question-Formation dataset, comprising 2M tokens with pairs of English sentences in declarative and question forms. We train two vanilla Transformers with 6 and 10 layers, respectively, each with an embedding dimension of 512, following the configurations from (Murty et al., 2023).

• The Question-Formation dataset not only offers a fixed amount of data for evaluating models of different sizes, but also has a limited vocabulary that ensures the well-separated condition in Assumption 3. The training losses stabilize at a value of approximately 1, aligning with the prediction in Proposition 4.

Training models with varying widths and dimensions In Appendix F.3, we train models of varying sizes based on the OpenELM design, adjusting the model dimensions and number of heads to achieve different model sizes while maintaining a fixed number of layers. The models have Transformer parameter counts of approximately 40M, 60M, and

80M. These models are pre-trained from scratch using GPT-2 encoding on the Standardized Project Gutenberg Corpus (SPGC) dataset, with training data sizes ranging from 167.06M to 333.17M tokens. The optimal combinations of model parameters and dataset size are determined by ensuring that training losses plateau and test losses are minimized.

• The ratio of model parameters to the square of the dataset size consistently approaches a constant value, supporting the theoretical results in Proposition 5.

### 6 Discussion

Relationship to the Chinchilla Scaling Laws. Our findings are contingent upon the transformer layers’ memorization of patterns, which are presumed to be well-separated points learned from the data. In our experiments, we utilize a reduced dataset to emulate the conditions of pattern separation and memorization. Moreover, we have trained the models on their respective datasets repeatedly, ensuring that the training losses have stabilized and the test losses have begun to ascend, indicative of a mild degree of over-parameterization. These conditions diverge from those of the Chinchilla experiment. In practical scenarios, commercial LLMs, akin to the Chinchilla model, are not subjected to such conditions. As we have noted in our paper, it has been observed that even after training on up to 2T tokens, some models have yet to exhibit signs of saturation. In practice, we have observed that the majority of transformer models at the commercial level tend to achieve a cross-entropy loss of approximately 2.2. The optimal balance between model and data sizes, however, is often determined by the collective expertise of practitioners. Additionally, the performance of these models can be compromised by both early and delayed stopping. Therefore, our current experimental setup represents an idealized condition that has not been encountered in commercial LLMs. Nevertheless, considering the substantial computational resources allocated to other scaling law investigations, we recognize that our numerical experiments represent a preliminary assessment that is contingent upon computational limitations, with a thorough analysis reserved for subsequent research endeavors.

Relationship to Prior Work on Hopfield Networks Recently, there has been a growing interest in physics-informed neural networks. The Energy Transformer (Hoover et al., 2024) designs a energy attention mechanism and the corresponding energy function, resulting in a unique model with a strong theoretical foundation that achieves SoTA results on graph anomaly detection and graph classification tasks. Wu et al. (2024) introduces a kernelized version of modern Hopfield networks, aiming to express energy in a feature space where patterns are well-separated, thus avoiding memory interference. Hu et al. (2024b) presents a theoretical framework for deriving and analyzing a family of modern Hopfield models, and Hu et al. (2024a) offers a compression method for Hopfield models, showing superior post-quantization performance compared to vanilla Transformers. While the existing literature, such as Hierarchical Associative Memory Krotov (2021), employs a system of differential equations to design a global energy function that can encompass feedback connections, our proposed model tackles the global energy specific to feedforward architectures. Predictive coding networks (Tang et al., 2023; Li et al., 2023) incorporate recurrent connections; however, their dynamics are focused on minimizing the total squared prediction errors and has less connection to the attention mechanism. By conceptualizing the

information retrieval properties of the Transformer as a sequence of associative memories, our model endeavors to establish relationships between model size and memorization (of training data) within the framework of statistical physics.

### 7 Conclusion

We model transformer-based networks with associative memory and study the cross-entropy loss with respect to model and data sizes. We employ a distance-based layer-wise energy function that corresponds to a nearest neighbor search across patterns memorized during training. We then construct a global energy function for the layered structure of the transformer models using the majorization-minimization technique. In practice, we have observed that the majority of transformer models at the commercial level tend to achieve a cross-entropy loss of approximately 2.2. The optimal balance between model and data sizes, however, is often determined by the collective expertise of practitioners. Additionally, the performance of these models can be compromised by both early and delayed stopping. We believe the current paper represents an important step towards understanding the pretraining behaviors of large transformer models. Empirical evidence supporting our study can be found in Appendix F. However, given the significant allocation of computational resources to other scaling law investigations, we acknowledge that our numerical experiments constitute a preliminary evaluation, constrained by computational restrictions. However, it is imperative to underscore the theoretical significance of these findings.

### Acknowledgments

The author thanks Dr. Yongqi Xu for stimulating discussions and practical assistance with the experiments.

### References

- S.-I. Amari. Learning patterns and pattern sequences by self-organizing nets of threshold elements. IEEE Transactions on Computers, 100(11):1197–1206, 1972.

- R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

T. W. J. Banks and T. Warkentin. Gemma: Introducing new state-of-the-art open models, 2024.

- M. Belkin, D. Hsu, S. Ma, and S. Mandal. Reconciling modern machine-learning practice and the classical bias–variance trade-off. Proceedings of the National Academy of Sciences, 116(32):15849–15854, 2019.

- S. Biderman, U. Prashanth, L. Sutawika, H. Schoelkopf, Q. Anthony, S. Purohit, and

- E. Raff. Emergent and predictable memorization in large language models. Advances in Neural Information Processing Systems, 36, 2024.

- N. Carlini, F. Tramer, E. Wallace, M. Jagielski, A. Herbert-Voss, K. Lee, A. Roberts, T. Brown, D. Song, U. Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650, 2021.

N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramer, and C. Zhang. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations, 2022.

- T. A. Chang and B. K. Bergen. Word acquisition in neural language models. Transactions of the Association for Computational Linguistics, 10:1–16, 2022.

- T. A. Chang and B. K. Bergen. Language model behavior: A comprehensive survey. Computational Linguistics, pages 1–58, 2024.

A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

S. d’Ascoli, L. Sagun, and G. Biroli. Triple descent and the two kinds of overfitting: Where & why do they appear? Advances in Neural Information Processing Systems, 33:3058– 3069, 2020.

M. Demircigil, J. Heusel, M. L¨we, S. Upgang, and F. Vermet. On a model of associative memory with huge storage capacity. Journal of Statistical Physics, 168:288–299, 2017.

Z. Du, A. Zeng, Y. Dong, and J. Tang. Understanding emergent abilities of language models from the loss perspective. arXiv preprint arXiv:2403.15796, 2024.

W. Fei, X. Niu, P. Zhou, L. Hou, B. Bai, L. Deng, and W. Han. Extending context window of large language models via semantic compression. In Findings of the Association for Computational Linguistics: ACL 2024, pages 5169–5181. Association for Computational Linguistics, 2024.

S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat,

- A. Fang, J. Li, S. Keh, et al. Language models scale reliably with over-training and on downstream tasks. arXiv preprint arXiv:2403.08540, 2024.

M. Gerlach and F. Font-Clos. A standardized project gutenberg corpus for statistical analysis of natural language and quantitative linguistics. Entropy, 22(1):126, 2020.

- M. Geva, R. Schuster, J. Berant, and O. Levy. Transformer feed-forward layers are key-value memories. arXiv preprint arXiv:2012.14913, 2020.

- A. Gokaslan and V. Cohen. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

W. Grathwohl, K.-C. Wang, J.-H. Jacobsen, D. Duvenaud, M. Norouzi, and K. Swersky. Your classifier is secretly an energy based model and you should treat it like one. In International Conference on Learning Representations, 2019.

J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022a.

J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. An empirical analysis of compute-optimal large language model training. Advances in Neural Information Processing Systems, 35:30016–30030, 2022b.

- B. Hoover, Y. Liang, B. Pham, R. Panda, H. Strobelt, D. H. Chau, M. Zaki, and D. Krotov. Energy transformer. Advances in Neural Information Processing Systems, 36, 2024.

J. J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8):2554–2558, 1982.

J. J. Hopfield. Neurons with graded response have collective computational properties like those of two-state neurons. Proceedings of the National Academy of Sciences, 81(10): 3088–3092, 1984.

J. Hsia, A. Shaikh, Z. Wang, and G. Neubig. Ragged: Towards informed design of retrieval augmented generation systems. arXiv preprint arXiv:2403.09040, 2024.

J. Y.-C. Hu, P.-H. Chang, R. Luo, H.-Y. Chen, W. Li, W.-P. Wang, and H. Liu. Outlier-efficient hopfield layers for large transformer-based models. arXiv preprint arXiv:2404.03828, 2024a.

J. Y.-C. Hu, D. Yang, D. Wu, C. Xu, B.-Y. Chen, and H. Liu. On sparse modern hopfield model. Advances in Neural Information Processing Systems, 36, 2024b.

- S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024c.

- A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, and M. Lewis. Generalization through memorization: Nearest neighbor language models. arXiv preprint arXiv:1911.00172, 2019.

D. Krotov. Hierarchical associative memory. arXiv preprint arXiv:2107.06446, 2021. D. Krotov and J. J. Hopfield. Dense associative memory for pattern recognition. Advances

in Neural Information Processing Systems, 29, 2016.

Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, and F. Huang. A tutorial on energy-based learning. Predicting Structured Data, 1(0), 2006.

- T. Li, M. Tang, and R. Bogacz. Modeling recognition memory with predictive coding and hopfield networks. In NeurIPS 2023 Workshop on Associative Memory & Hopfield Networks, 2023.

- R. T. McCoy, R. Frank, and T. Linzen. Does syntax need to grow on trees? sources of hierarchical inductive bias in sequence-to-sequence networks. Transactions of the Association for Computational Linguistics, 8:125–140, 2020.
- S. Mehta, M. H. Sekhavat, Q. Cao, M. Horton, Y. Jin, C. Sun, S. I. Mirzadeh, M. Najibi, D. Belenko, P. Zatloukal, et al. OpenELM: An efficient language model family with open training and inference framework. In Workshop on Efficient Systems for Foundation Models II@ ICML2024, 2024.

B. Millidge, T. Salvatori, Y. Song, T. Lukasiewicz, and R. Bogacz. Universal hopfield networks: A general framework for single-shot associative memory models. In International Conference on Machine Learning, pages 15561–15583. PMLR, 2022.

N. Muennighoff, A. Rush, B. Barak, T. Le Scao, N. Tazi, A. Piktus, S. Pyysalo, T. Wolf, and C. A. Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36, 2024.

- T. Munkhdalai, M. Faruqui, and S. Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention. arXiv preprint arXiv:2404.07143, 2024.

S. Murty, P. Sharma, J. Andreas, and C. D. Manning. Grokking of hierarchical structure in vanilla transformers. arXiv preprint arXiv:2305.18741, 2023.

P. Nakkiran, G. Kaplun, Y. Bansal, T. Yang, B. Barak, and I. Sutskever. Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124003, 2021.

J. Ortega and W. Rheinboldt. Iterative Solution of Nonlinear Equations in Several Variables, volume 30. SIAM, 1970.

- B. Peng, J. Quesnelle, D. Rolnick, A. Lotter, U. H. Adil, and E. La Rocca. A Preliminary report on DisTrO, 2024. URL https://github.com/NousResearch/DisTrO/tree/main. Available at https://github.com/NousResearch/DisTrO/tree/main, date: 2024-0910.

A. Power, Y. Burda, H. Edwards, I. Babuschkin, and V. Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.

- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 2019.

- J. W. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.

H. Ramsauer, B. Sch¨fl, J. Lehner, P. Seidl, M. Widrich, L. Gruber, M. Holzleitner, T. Adler, D. Kreil, M. K. Kopp, et al. Hopfield networks is all you need. In International Conference on Learning Representations, 2020.

- B. Saha, D. Krotov, M. J. Zaki, and P. Ram. End-to-end differentiable clustering with associative memories. In International Conference on Machine Learning, pages 29649–

29670. PMLR, 2023.

S. Smith, M. Patwary, B. Norick, P. LeGresley, S. Rajbhandari, J. Casper, Z. Liu, S. Prabhumoye, G. Zerveas, V. Korthikanti, et al. Using deepspeed and megatron to train megatron-turing NLG 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990, 2022.

S. Sukhbaatar, E. Grave, G. Lample, H. Jegou, and A. Joulin. Augmenting self-attention with persistent memory. arXiv preprint arXiv:1907.01470, 2019.

Y. Sun, P. Babu, and D. P. Palomar. Majorization-minimization algorithms in signal processing, communications, and machine learning. IEEE Transactions on Signal Processing, 65(3):794–816, 2016.

A. Svete and R. Cotterell. Transformers can represent n-gram language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics, pages 6841–6874, 2024.

A. Svete, N. Borenstein, M. Zhou, I. Augenstein, and R. Cotterell. Can transformers learn n-gram language models? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9851–9867, 2024.

- M. Tang, T. Salvatori, B. Millidge, Y. Song, T. Lukasiewicz, and R. Bogacz. Recurrent predictive coding models for associative memory employing covariance learning. PLoS Computational Biology, 19(4):e1010719, 2023.

- K. Tirumala, A. Markosyan, L. Zettlemoyer, and A. Aghajanyan. Memorization without overfitting: Analyzing the training dynamics of large language models. Advances in Neural Information Processing Systems, 35:38274–38290, 2022.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,  L. Kaiser, and

I. Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

D. Wu, J. Y.-C. Hu, T.-Y. Hsiao, and H. Liu. Uniform memory retrieval with larger capacity for modern hopfield models. arXiv preprint arXiv:2404.03827, 2024.

R. Xiong, Y. Yang, D. He, K. Zheng, S. Zheng, C. Xing, H. Zhang, Y. Lan, L. Wang, and T. Liu. On layer normalization in the transformer architecture. In International Conference on Machine Learning, pages 10524–10533. PMLR, 2020.

W. Xiong, J. Liu, I. Molybog, H. Zhang, P. Bhargava, R. Hou, L. Martin, R. Rungta, K. A. Sankararaman, B. Oguz, et al. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039, 2023.

- A. Yang, B. Xiao, B. Wang, B. Zhang, C. Bian, C. Yin, C. Lv, D. Pan, D. Wang, D. Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.
- B. Zhang and R. Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

- Appendix A. Summary of Notations ρi The i-th pattern A Constant depends on the number of layers and the hidden dimension of the network Bi Ball associated to the i-th pattern d′ Number of samples in the test samples d(·,·) A distance metric D Size of the training data, D ≈ Tmaxd d Number of samples in the training samples demb Embedding dimension

L Cross-entropy loss l Number of Transformer layers N Number of Transformer parameters n Dimension of input sequences, n = Tmaxdemb R Radius of the sphere S of patterns Tmax Max number of tokens in a sequence

- Appendix B. Deferred tables

Table 1: Table of selected related works for Hopfield network, enumerating their domain, energy function, and memory capacity. For all the works above, n represents the dimension of the input vector. W is the outer product of the patterns. M is the matrix of patterns. r is the order of polynomial F(·), d is the number of patterns, and c is a positive constant.

Reference Domain Energy Capacity Hopfield (1982) {−1,+1}n E(x) = −21xTWx − bTx O(n)

Krotov and Hopfield (2016) {−1,+1}n E(x) = − ni=1 F((ρi)Tx) Θ(nr) Demircigil et al. (2017) {−1,+1}n E(x) = −LogSumExp(MTx) Θ(2n2 ) Ramsauer et al. (2020) Rn E(x) = −LogSumExp(β,MTx)+ Θ(cn−41)

- 1

- 2xTx + β−1 log d + maxi ∥ρi∥2/2

Table 2: Transformer-based language models and their reported cross-entropy loss. Model Model Size Data Size L Reference Transformer 1.5B 22B 2.5 Kaplan et al. (2020) Chinchilla 70B 1.4T 2.2 Hoffmann et al. (2022a) PaLM 2 16B 100B 2.4 Anil et al. (2023) GPT-2 8.7B 178B 2.3 Muennighoff et al. (2024) MiniCPM 2.4B 140B 2.4 Hu et al. (2024c) Nanotron 1.2B 105B 2.4 Peng et al. (2024)

### Appendix C. Some properties of the energy functions

We introduce some useful properties of the LogSumExp function defined below. This is particularly useful because The softmax function, widely utilized in the Transformer models, is the gradient of the LogSumExp function. As shown in (Grathwohl et al., 2019), the LogSumExp corresponds to the energy function of the a classifier.

LogSumExp(x) := log

n

exi, x = (x1,...,xn) ∈ Rn.

i=1

- Lemma 1 LogSumExp(x) is convex. Proof

n

n

eyi)1−t

exi)t(

tLogSumExp(x) + (1 − t)LogSumExp(y) = log(

i=1

i=1

n

etxi+(1−t)yi = LogSumExp(tx + (1 − t)y) ∀t ∈ [0,1].

≥ log

i=1

- Lemma 2 Suppose x = (x1,...,xn) ∈ Rn, then we have

max

1≤i≤n

xi < LogSumExp(x) ≤ max

1≤i≤n

xi + log n.

Proof Taking log on each side of the inequality

exp( max

1≤i≤n

xi) <

n

i=1

exp(xi) ≤

n

i=1

exp( max

1≤i≤n

xi)

yields the results.

Consequently, we have the following smooth approximation for the min function.

- Lemma 3 Suppose x = (x1,...,xn) ∈ Rn, then we have

min

1≤i≤n

xi − log n ≤ −LogSumExp(−x) < min

1≤i≤n

xi.

- Lemma 4 For x = (x1,...,xn),y = (y1,...,yn) ∈ Rn, we have |LogSumExp(x) − LogSumExp(y)| ≤ ∥x − y∥∞,

where ∥x∥∞ := max1≤i≤n |xi|. Proof Let

f(t) := LogSumExp(tx + (1 − t)y), ∀t ∈ [0,1]. According to the mean value theorem, ∃s ∈ (0,1) such that

n i=1 exp(sxi + (1 − s)yi)(xi − yi)

LogSumExp(x) − LogSumExp(y) = f′(s) =

.

n i=1 exp(sxi + (1 − s)yi)

So

n i=1 exp(sxi + (1 − s)yi)∥x − y∥∞

|LogSumExp(x) − LogSumExp(y)| ≤

= ∥x − y∥∞.

n i=1 exp(sxi + (1 − s)yi)

- C.1 Proof of Proposition 2 Proof Let ξ = max1≤i≤d ∥ρi∥, then we have

d

2EMCHNβ=2 (x) = −log

exp(2(ρi)Tx) + log d + ∥x∥2 + ξ2

i=1

d

exp(2(ρi)Tx) − log(exp(−(∥x∥2 + ξ2))) + log d.

= −log

i=1

So

d

2EMCHNβ=2 (x) − log d = −log(

i=1

d

= −log(

i=1

exp(2(ρi)Tx − ξ2 − ∥x∥2))

exp(∥ρi∥2 − ξ2 − ∥ρi − x∥2)).

Therefore, due to Lemma 4, we have |E(x) − (2EMCHNβ=2 (x) − log d)| = |LogSumExp(∥ρi∥2 − ξ2 − ∥ρi − x∥2) − LogSumExp(−∥x − ρi∥2)|

|∥ρi∥2 − ξ2| = max 1≤i≤d

∥ρi∥2 − min

∥ρi∥2.

≤ max

1≤i≤d

1≤i≤d

### Appendix D. Deferred proofs from Section 5

- D.1 Proof of Proposition 4 Proof

1 d′

log(pθ(x)) = −Ex∼p

L(N,D) = H(p D,pθ) = −

[log pθ(x)]

D

x∈ D

d′

1 d′ x∈Ω

δ(x − ρσ(i))Eglobal(x) dµ

= log Zθ

P D(x) dµ +

x∈Ω

i=1

1 d′

= log Zθ +

Eglobal(x)

ρσ(i)

1 Zt − log l + c (≈b) log Zt +

1 Zt

(=a) log Zθ +

where (a) is because g(ρσ(i)) = 0, and (b) is due to (10), where we have

l ec x∈Ω

exp(−Eglobal(x))dx =

exp(−Et(x))dx, and

Zθ =

x∈Ω

log Zθ ≈ log l − c + log exp(−Et(x))dx = log l − c + log Zt.

(15)

#### D.2

- Lemma 5 The incomplete gamma function γ(n,r) satisfies

rn n ≤ γ(n,r) ≤

rn n

e−r

Proof For 0 ≤ x ≤ r, we have

xn−1e−r ≤ xn−1e−x ≤ xn−1. Integrating from 0 to r on each side yields the result.

Theorem D.1 (Stirling’s approximation) For any complex z, the Stirling’s approximation gives that

2π z

z e

1 z

z

Γ(z) =

1 + O(

) .

For large z,

√

Γ(z + 1) ≈

2πz

z e

z

.

### Appendix E. Transformer details: using GPT-2 as an example

The original GPT-2 model was trained on a 40GB large dataset called WebText that is made of data derived from outbound links from Reddit. The model is trained on the next sentence prediction (NSP) task in a self-supervised manner. A pre-trained tokenizer can be applied to convert the text into tokens using a fixed vocabulary. A max token length Tmax (e.g., Tmax = 1024) is set, so during training, if the number of tokens is greater than Tmax, the documents will be truncated. The model is trained causally, which means that the prediction for the next token only depends on the inputs from earlier tokens. The model was trained with a global batch size of 512, and the test perplexity still improves if given more training time.

GPT-2 uses a byte-level version of Byte Pair Encoding (BPE), and the vocabulary size is nvoc = 50,257. The hidden dimension for the medium size model is demb = 1024. So the input sequence is of Tmaxdemb dimension. These sequences are passed through the model. For the medium size model, these include 24 transformer encoder blocks with 1024 hidden units and 16 self-attention heads (i.e., l = 24,demb = 1024,nh = 16). The number of parameters used for word embedding is nvoc · demb. The number of parameters in the multi-head attention layer is l · nh · (3 · demb · demb/nh) = 3ld2emb = 75,497,472. The number of parameters in the dense weights and layer normalization is l(d2emb + 2demb) = ld2emb + 2ldemb = 25,214,976, and the number of parameters in the feed-forward weight matrices and bias is l(2demb · dFF + demb + dFF) = 6ld2emb + 4ldemb = 151,093,248, with dFF = 3072 = 3demb. As we can observe, the multi-head attention and feed-forward layers account for most of the parameters in the model, and N ≈ Ald2emb with some constant A ≈ 10 in this case.

The loss used for the GPT-2 model is the log-probability of a dataset divided by the number of canonical units (e.g., a character, a byte, a word), which is equivalent to the cross-entropy loss. The cross-entropy loss is commonly used to measure the divergence between the predicted probabilities and the true labels. For the NSP task, the model is trained to predict the next token in a sequence based on the context of the previous tokens.

So the cross-entropy is taken between the predicted probabilities Prθ(xi) of the token xi and the labels’ probabilities PrD(xi) for all tokens xi in the vocabulary, i.e.,

D

1 D

−

log(pθ(xi)) = −

x∼pD

i=1

pD(x)log(pθ(x)).

Another commonly used loss is the perplexity, which is equivalent to the exponentiated version of the cross-entropy.

### Appendix F. Empirical results

We explore the hypothesis regarding the radius r in Section 5 using a pre-trained GPT2 medium model. Additionally, we train vanilla Transformers and OpenELM models of different sizes to explore their cross-entropy losses.

#### F.1 Empirical evaluation of the radius

We evaluate the radius r in Zl of a pre-trained GPT-2 medium model. We use the 24layer pre-trained GPT-2 model (Radford et al., 2019)1. The medium size model has 355M parameters. The model is pre-trained with the next sentence prediction task on a large (40 GB) text corpus extracted from web pages. The hidden dimension is demb = 1024.

We test the model on the OpenWebText (Gokaslan and Cohen, 2019) dataset, a reproduction of the WebText dataset used for training the GPT-2 model. The dataset contains 9B tokens from 8,013,769 documents. We randomly sample 80K chunks of 256 tokens from the dataset. These cover approximately 1% of the documents and constitute approximately

- 0.2% of the tokens used for training. For each sample chunk, we record the activation vector of the last layer for prediction of the next token. As discussed above, each vector should be close to a stored pattern ρli. We calculate the distance between each activation vector and

its nearest neighbor in terms of the L2 norm and find the nearest neighbor distance for each vector, which results in 80K distance values. In Fig. 2 (top), we plot the histogram of the nearest neighbor distances for these output activations using 25%,50%,75%, and 100% of the output vectors. In all these cases, the mean and median equals approximately to 20, so that a typical Bi of pattern ρi has radius 10. This corroborates (14) in Section 5, according to which the radius is of order 1024/(2πe) = 7.74. The activation is only collected for 1% of the documents, so the estimated radius may be greater than the actual value.

Significance: The experiment’s objective is to determine the radius r in Zl of a pretrained GPT-2 model, thereby validating the hypothesis regarding the radius of patterns as stated in (14). We compute the Euclidean distances between random sequences processed by the GPT-2 medium model to estimate the magnitude of the radius r. The results indicate that the mean and median of the nearest neighbor distances decrease as the sample size increases, and are around 20, which suggests that a typical Bi associated with the pattern ρi has a radius of approximately 10. Considering the relatively small sample size, we anticipate that the actual radius may be smaller than 10. Nevertheless, the experiment offers valuable

- 1. available at https://github.com/openai/gpt-2

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

Figure 2: Top: Distribution of nearest neighbor distances for output activations utilizing 25%,50%,75%, and 100% of output data. The mean and median values of these distances consistently hover around 20, aligning with the magnitude 2 n/2πe as hypothesized. Bottom-left: Performance of vanilla Transformers with 6 layers (left) and 10 layers (middle), each trained on the 2M Question-Formation dataset. The models were configured according to the experimental setup detailed in (Murty et al., 2023). The training losses for both models converge to a value of approximately 1, a finding that is consistent with Proposition 4. Bottom-right: The pre-training loss (dots) and validation loss (squares) of an OpenELM model across five training runs. The minimal validation losses are displayed in dashed lines. Each run’s performance is marked by distinct colors, with the minimum validation loss value for each run indicated along the y-axis.

insights into the order of magnitude of the radius and confirms the validity of the hypothesis concerning the radius of patterns within the model’s latent space.

#### F.2 Training vanilla Transformers

We next train vanilla Transformers using a small amount of high-quality data. The QuestionFormation dataset, proposed by McCoy et al. (2020), consists of pairs of English sentences in declarative formation and their corresponding question formation. The dataset contains D = 2M tokens. The sentences are context-free with a vocabulary size of 68 words, and the task is to convert declarative sentences into questions.

We follow the settings in (Murty et al., 2023) to train two vanilla Transformers (demb = 512,Tmax = 5000) with l = 6 layers and l = 10 layers respectively. The training losses are shown in Fig. 2 (bottom-left), where the losses stabilize at a value of around L = 1 as predicted in Proposition 4.

Significance: The experiment investigates the pre-training loss by training two vanilla Transformers with varying depths. Utilizing the Question-Formation dataset, this experiment offers a fixed amount of data to evaluate the model’s memorization ability. The dataset’s simplicity, featuring a limited vocabulary and a context-free structure, facilitates the well-separated condition in Assumption 3. The stabilization of training losses around a value of 1 not only aligns with the theoretical predictions in Proposition 4 but also indicates that the models have reached a point of diminishing returns, where memorization is likely to predominate.

#### F.3 Training models with varying widths and dimensions

Next, we train models of different sizes following the OpenELM (Mehta et al., 2024) design2, varying the model configurations including the dimensions and the numbers of heads to achieve different model sizes while keeping the number of layers fixed. We choose different widths and dimensions such that the number of parameters of the transformer layers are about N =40M, 60M, and 80M respectively. The configurations and hyperparameters can be found in Appendix G. Specifically, the number of layers is fixed in our experiments, as empirical evidence has demonstrated that the depth is a determinant factor influencing the performance.

We utilize the Standardized Project Gutenberg Corpus (SPGC) dataset (Gerlach and Font-Clos, 2020), which contains a filtered timeseries of word-tokens without punctuation, derived from the Project Gutenberg digital library of public domain literary works. We choose this subset because it offers a collection of high-quality word sequences. We prepare the training data using different proportions of the first 180M words of the SPGC, and we use the last 5% tokens as the validation set. We pre-train the models from scratch on the tokenized training data with GPT-2 encoding, which encompasses eight distinct sizes ranging uniformly from D =167.06M to D =333.17M. Throughout the training, we employ random sampling to select chunks of the pre-determined context length. Given the typically extensive length of the e-books within this dataset, it is plausible that sequences drawn with the context length of 1024 tokens originate from the same book.

We report the number of transformer parameters N and the corresponding D∗ such that training loss plateaus and the test loss is minimized, defined by D∗ = min {D : MSE(Ltrain(N,D),L(minN,D)) < σ2}, where L(trainN,D) is the training loss of model of size N using training data of size D, L(minN,D) is the minimal validation loss throughout the training steps (as is depicted in the bottom-right of Fig. 2), and MSE is the mean squared error taken over the proceeding 1000 iterations of the step where the validation loss is minimized. In Table 3, we report the MSE for each configuration, with D∗ highlighted in each row when σ2 = 0.04. Upon analyzing the optimal combinations, we have computed the ratio of model parameters to the square of the dataset size, as demonstrated in the last column. The threshold σ were empirically selected based on our preliminary experiments, as provided in

- Appendix G for visual inspection. Our findings indicate that the ratio N/D∗2 consistently approaches a constant value, reinforcing our theoretical results in Section 5.

2. available at https://github.com/apple/corenet.

Table 3: Mean squared error over 1000 iterations between training loss and minimal validation loss for different model configurations and pre-training settings. The last column reports the ratio between N and D2 for D∗ with unit 10−10.

MSE D = 167.06M D = 190.38M D∗ = 214.18M D = 237.98M N/D∗2 (10−10) N = 39.95M 0.07 0.05 0.04 0.04 8.71

D = 214.18M D = 237.98M D∗ = 261.78M D = 285.57M N/D∗2 (10−10) N = 60.26M 0.05 0.04 0.02 0.02 8.79

D = 261.78M D = 285.57M D∗ = 309.37M D = 333.17M N/D∗2 (10−10) N = 80.20M 0.05 0.04 0.02 0.02 8.38

Significance: The experiment aims to support Proposition 5, which predicts a quadratic relationship between the number of Transformer parameters and the size of well-separated datasets. We train models of varying widths and dimensions following the OpenELM design from scratch and adjust the training data size between 167.06M and 333.17M. We identify the optimal combinations of model parameters and dataset size by ensuring that the training losses plateau and the test losses are minimized. The consistent approach of the ratio of model parameters to the square of the dataset size towards a constant value corroborates Proposition 5, implying that an optimal ratio exists, which can inform the choice of model size in relation to dataset size for optimal performance.

### Appendix G. Experimental Details

#### G.1 Configurations

We follow the OpenELM (Mehta et al., 2024) architecture and choose the following configurations such that the number of transformer parameters are about 40M, 60M, and 80M.

#### Parameter Model 1 Model 2 Model 3

Number of Transformer Parameters 39.95M 60.26M 80.20M Model Dimension 954 1280 1440 Number of Transformer Layers 8 8 8 Number of KV Heads 3, 3, 3, 3, 4, 4, 4, 5 3, 3, 3, 3, 4, 4, 4, 5 3, 3, 3, 4, 4, 4, 5, 5 Number of Query Heads 6, 6, 6, 6, 8, 8, 8, 10 6, 6, 6, 6, 8, 8, 8, 10 6, 6, 6, 8, 8, 8, 10, 10

Hyperparameters for pre-training are listed below.

Parameter Detail Tokens per iteration 491,520 Vocabulary Size 50,257 Activation Function swish Attention Dropout 0.1 Embedding Dropout 0.1 Head Dimension 64 Initializer Range 0.02 Max Context Length 1024 Normalization Layer rms norm Normalize QK Projections True QKV Multipliers 0.5, 1.0 Batch size 12 AdamW β1 = 0.9,β2 = 0.95,ϵ = 10−8

#### G.2 Robustness to Deduplication

[Figure 14]

- Figure 3: The cross-entropy loss for one model configuration during pre-training (depicted with dots) and validation (depicted with squares) across five separate training runs. The minimal attainable validation loss is represented by dashed lines. Each individual run’s performance is distinguished by a unique color, and the y-axis highlights the lowest validation loss for each respective run.

In order to further confirm the validity of our analyses, we conduct five independent training runs of Model 1 to assess the model’s robustness and the consistency of its performance across different training instances. Fig. 3 illustrates the CE loss over the course of training iterations for each run, along with the minimum validation loss achieved during each run. It can be observed that the training runs exhibit varying degrees of performance,

- as indicated by the different trajectories of the CE loss curves. The minimum validation loss

- remains consistent to the second decimal place across various training runs, and is achieved
- at approximately the 6000th training step.

- G.3 Additional Results Figured below are the training dynamics of the models in Table 3 for visual inspection.

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

[Figure 25]

[Figure 26]

- Figure 4: Cross-entropy losses of eight models employing the OpenELM architecture as presented in Table 3.

