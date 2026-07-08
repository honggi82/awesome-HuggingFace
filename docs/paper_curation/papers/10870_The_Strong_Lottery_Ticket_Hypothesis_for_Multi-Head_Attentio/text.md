## The Strong Lottery Ticket Hypothesis for Multi-Head Attention Mechanisms

### Hikari Otsuka1,†, Daiki Chijiwa2, Yasuyuki Okoshi1, Daichi Fujiki1, Susumu Takeuchi2, Masato Motomura1

1Institute of Science Tokyo 2NTT, Inc. †otsuka.hikari@artic.iir.isct.ac.jp

# arXiv:2511.04217v1[cs.LG]6Nov2025

###### Abstract

The strong lottery ticket hypothesis (SLTH) conjectures that high-performing subnetworks, called strong lottery tickets (SLTs), are hidden in randomly initialized neural networks. Although recent theoretical studies have established the SLTH across various neural architectures, the SLTH for transformer architectures still lacks theoretical understanding. In particular, the current theory of the SLTH does not yet account for the multi-head attention (MHA) mechanism, a core component of transformers. To address this gap, we introduce a theoretical analysis of the existence of SLTs within MHAs. We prove that, if a randomly initialized MHA of H heads and input dimension d has the hidden dimension O(d log(Hd3/2)) for the key and value, it contains an SLT that approximates an arbitrary MHA with the same input dimension with high probability. Furthermore, by leveraging this theory for MHAs, we extend the SLTH to transformers without normalization layers. We empirically validate our theoretical findings, demonstrating that the approximation error between the SLT within a source model (MHA and transformer) and an approximate target counterpart decreases exponentially by increasing the hidden dimension of the source model.

### 1 Introduction

The lottery ticket hypothesis (Frankle and Carbin 2019)overparameterized networks contain subnetworks that achieve comparable accuracy to fully trained networks even if trained in isolation—presented new possibilities for compact and high-performing models inherent in recent deep neural networks. Later, a stronger claim, which is formally defined as the strong lottery ticket hypothesis (SLTH), was proposed (Ramanujan et al. 2020; Malach et al. 2020): overparameterized networks contain subnetworks (called strong lottery tickets (SLTs)) that achieve comparable accuracy to the trained dense network even without any training. Whether such subnetworks exist is a fascinating question in itself, and studying them can bring us closer to understanding the principles behind overparameterized models.

The rigorous proof for the SLTH was firstly established in fully-connected networks. Early studies showed that a randomly-weighted fully-connected network of sufficient width (a source network) contains an SLT, which approximates an arbitrary fully-connected network with half the

Existing Studies: Two-Layers-For-One Approximation

|[Figure 1]<br><br>Activation<br><br>(Fully-Connected Network, CNN, Equivariant NN, etc.)<br><br>|
|---|

This Work (Attention Mechanism)

|Softmax Softmax<br><br>[Figure 2]<br><br>[Figure 3]<br><br>[Figure 4]<br><br>[Figure 5]<br><br>Query Key Value|
|---|

Target Network (arbitrary weights)

Source Network (random & pruned )

Figure 1: Comparison of the approximation techniques in conventional theories of the SLTH (top) and in our attentionspecific approach (bottom). This work demonstrates that an arbitrary attention mechanism can be approximated by pruning a randomly initialized one.

depth (a target network) (Malach et al. 2020; Orseau, Hutter, and Rivasplata 2020; Pensia et al. 2020). These theories are built on the foundational argument called a two-layers-forone approximation: a two-layer source network with random weight matrices contains an SLT that approximates a singlelayer target network with an arbitrary weight matrix (the top panel of Figure 1). Following this finding, subsequent studies have succeeded in proving the existence of SLTs in more complex networks, such as convolutional and equivariant networks (da Cunha, Natale, and Viennot 2022; Burkholz 2022a; Ferbach et al. 2023).

However, the theoretical foundation of the SLTH for transformers, which form the basis of modern language models, remains unexplored—due to a transformer-specific component, an attention mechanism. As shown in the bottom panel of Figure 1 (right side), one of the distinctive structures in transformers is the inner product between two vectors called query and key, obtained as linear projections of given inputs. This structure fundamentally differs from

the conventional components of non-transformer architectures for which the SLTH has been established (the top panel of Figure 1); thus, it remains a mystery whether transformers contain SLTs under existing theoretical insights. This gap motivates our key research question: does an attention mechanism—an essential component of transformerscontain an SLT?

In this work, we prove the existence of SLTs within attention mechanisms, extending the SLTH to transformers. More precisely, we prove a suitably pruned source attention mechanism with random weights can approximate any target attention mechanism with arbitrary weights:

Theorem 1 (informal). Given inputs of length T, a suitably pruned randomly-initialized attention mechanism of the input dimension d and hidden dimension n = O(dlog(d3/2/ϵ)) can approximate an arbitrary attention mechanism of the same input dimension with an approximation error ϵ, with high probability.

Our key idea is to reinterpret the inner product between the query and key vectors in the attention mechanism as a (linear) neural network weighted by the query and key projection matrices. Then, we can view the source and target inner products as neural networks with different numbers of layers: the source one has two layers with query and key projection matrices as its weights, while the target one has a single layer with a weight matrix obtained by merging these two projections. This reinterpretation makes it possible to apply a variant of the two-layers-for-one approximation, leading to the SLT existence within attention mechanisms (Theorem 3). Note that, as can be seen by comparing the top and bottom panels of Figure 1, our arguments do not require additional layers in the MHA for approximation, in contrast to the previous two-layers-for-one argument for fully-connected networks. By exploiting this theorem, we further establish the SLTH for transformers without normalization layers: a randomly-initialized transformer has an SLT that approximates an arbitrary transformer with similar structures (Theorem 6).

We also empirically validate our theory and confirm its implications. Specifically, we show that 1) the approximation error between the source and target attentions (or, more generally, source and target transformers) decays exponentially as the hidden dimension increases; and 2) this approximation error does not diverge even when the input length T increases. Also, based on our theoretical arguments, we derive a new, practical weight initialization scheme, leading to better SLTs in our experiments.

Our contributions are summarized as follows:

- • We provide the first theoretical proof that SLTs exist within attention mechanisms and transformers by reinterpreting the inner product in attention mechanisms.
- • We then empirically validate our theory under conditions that are close to our theoretical assumptions. More precisely, we carefully designed a synthetic experiment to observe how the hidden dimension or input length affects the approximation error of SLTs.
- • Furthermore, we demonstrate that our theory not only explains the empirical results, but also provides a new in-

sight into a weight initialization for finding better SLTs in practical settings.

Notation: In this paper, scalars, vectors, and matrices are denoted by lowercase, bold lowercase, and bold uppercase letters, respectively. We use the norm of matrices and vectors ∥ · ∥ as the spectral norm unless otherwise specified by subscripts. We denote the uniform distribution on [a,b] by U[a,b]. ”⊙” represents an element-wise multiplication (i.e., the Hadamard product). The superscript (i) denotes the layer

index, and we write {x(i)}Hi=1 to denote the set of elements x(i) indexed by i from 1 to H.

### 2 Preliminaries

This section reviews the prior theoretical studies on the strong lottery ticket hypothesis (SLTH) and the formulation of multi-head attention (MHA) mechanisms.

#### 2.1 Strong Lottery Ticket Hypothesis

The strong lottery ticket hypothesis (SLTH) conjectured that a randomly-initialized network inherently contains subnetworks (strong lottery tickets (SLTs)) that achieve high accuracy comparable to trained dense networks, without any weight updates (Ramanujan et al. 2020; Malach et al. 2020). The first theoretical result of the SLTH was given by Malach et al. (2020). They proved the existence of SLTs in a fullyconnected ReLU network. Subsequent studies relaxed the requirements for source networks to contain SLTs that approximate some target network (Orseau, Hutter, and Rivasplata 2020; Pensia et al. 2020; Burkholz 2022b). In particular, Pensia et al. (2020) introduced a subset-sum approximation technique (Lueker 1998) into the SLTH context and concluded that the logarithmic overparameterization of the source network to a given target is approximately optimal:

2×d1, W˜ 1 ∈ Rn×d

###### Lemma 2. Given x ∈ Rd

1, and W˜ 2 ∈ Rd

1, W ∈ Rd

2×n, we define the target and pruned source

fully-connected networks as FT(x) := Wx, FS(x) := (W˜ 2 ⊙ M2)ReLU((W˜ 1 ⊙ M1)x),

where M1 ∈ {0,1}n×d

2×n are binary pruning masks. Assume that ∥W∥ ≤ 1, ∥x∥ ≤ 1, and each entry of W˜ 1 and W˜ 2 is drawn i.i.d. from U[−1,1]. Also, for 0 < ϵ < 1, suppose that the hidden dimension n satisfies n ≥ d1C log (2d1d2/ϵ), where C > 0 is some universal constant. Then, with probability at least 1 − ϵ, there exists a choice of binary pruning masks M1 and M2 such that

1 and M2 ∈ {0,1}d

∥FT(x) − FS(x)∥ ≤ ϵ.

This approach, which approximates a single weight matrix by pruning two randomly initialized matrices (the top panel of Figure 1), is called the two-layers-for-one approximation, and is now the theoretical foundation of the SLTH for more complex architectures and problems (da Cunha, Natale, and Viennot 2022; Burkholz 2022a; Ferbach et al. 2023; Natale et al. 2024; Otsuka et al. 2025).

[Figure 6]

[Figure 7]

Softmax

Softmax

Figure 2: The structure of an MHA. By partitioning the output projection, the final result can be interpreted as the sum of outputs from all heads.

#### 2.2 Multi-head Attention Mechanisms

1 be a sequence of T input vector embeddings. For each embedding xi, we define a binary attention mask ai ∈ {0,1}⊤, where ai,j = 1 indicates that the i-th embedding attends to the j-th one. We assume that each embedding attends to at least one other (i.e., ∥ai∥1 ≥ 1). Given such inputs, a multi-head attention (MHA) mechanism (Vaswani et al. 2017) is defined as a function that computes their pair-wise relationships at each of the H attention heads (the left panel of Figure 2). For the i-th embedding xi, the MHA is of the following form:

Let X = [x1,...,xT]⊤ ∈ RT×d

Attn(xi;X,{W(Qj),W(Kj),W(Vj)}Hj=1,WO) := head(1)i ,...,head(iH) WO ∈ R1×d

,

2

q(ij)K(j)⊤ √dK

head(ij) :=σ

;ai V (j),

q(ij) := x⊤i W(Qj), K(j) := XW(Kj), V (j) := XW(Vj),

ai,j exp(xi,j)

σ(xi;ai)j :=

.

T k=1 ai,k exp(xi,k)

Here, we define W(Qj),W(Kj) ∈ Rd

1×dK,W(Vj) ∈ Rd

1×dV, and WO ∈ RHd

V×d2 as single-layer projections for the

query q(ij), key K(j), value V (j), and output of the MHA, respectively. The softmax function with the attention mask

is defined as σ(·). As shown in the right panel of Figure 2, by partitioning the output weight matrix WO into

WO := [W(1)O ⊤,...,W(OH)⊤]⊤, W(Oj) ∈ Rd

V×d2, the form of Attn(·) can be represented as follows:

Attn(xi;X,{W(Qj),W(Kj),W(Vj)}Hj=1,WO)

= Attn(xi;X,W(1:Q:OH))

H

head(ij)W(Oj).

=

j=1

We denote the set of all weights as

W(1:Q:OH) := {W(Qj),W(Kj),W(Vj),W(Oj)}Hj=1.

### 3 Strong Lottery Ticket Hypothesis for Transformers

This section analyzes the existence of SLTs within multihead attention (MHA) mechanisms and extends it to the transformer architecture without normalization layers. For a detailed proof, see Section A.

#### 3.1 Setups

We consider two MHAs: a target MHA AttnT(·) with arbitrary (tuned) weights, and a pruned source MHA AttnS(·) with randomly-initialized weights, denoted as follows:

AttnT(xi) = Attn(xi;X,W(1:Q:OH)), (1) AttnS(xi) = Attn(xi;X,(W˜ ⊙ M)(1:Q:OH)). (2)

Here, similarly to the weight set W(1:Q:OH), we define the set of pruned random weights as

(W˜ ⊙ M)(1:Q:OH) := {W˜ (Qj) ⊙ M(Qj),W˜ (Kj) ⊙ M(Kj),

W˜ (Vj) ⊙ M(Vj),W˜ (Oj) ⊙ M(Oj)}Hj=1, where W˜ (Qj),W˜ (Kj) ∈ Rd

1×nK, W˜ (Vj) ∈ Rd

1×nV, and W˜ (Oj) ∈ Rn

V×d2 are the randomly-weighted query, key, value, and output projections of the j-th head in AttnS(·), respectively. Also, M(Qj), M(Kj), M(Vj), and M(Oj) are their corresponding binary pruning masks. Note that the target and source MHAs have different key and value hidden dimensions: dK and dV for the target, and nK and nV for the source. We assume that α ≥ max(√d1,√d2) for the inputs, and ∥W(Qj)∥,∥W(Kj)∥,∥W(Vj)∥,∥W(Oj)∥ ≤ 1 for the j-th head of the target MHA. The source MHA is initialized such that each entry of W˜ Q and W˜ K is drawn i.i.d. from U[−n1K/4,n1K/4], and each entry of W˜ V and W˜ O is drawn i.i.d. from U[−1,1].

##### 3.2 The Existence of SLTs Within an MHA Now, we prove the following SLT existence theorem:

Theorem 3. Let AttnT(·) and AttnS(·) be as defined in Equations (1) and (2). Then, with probability at least 1 − ϵ, there exists a choice of binary pruning masks

M(Qj),M(Kj),M(Vj),M(Oj) that satisfy

∥AttnS(xi) − AttnT(xi)∥ ≤ ϵ, if the source hidden dimensions satisfy

max

i∈[T]

nK ≥ d1C log

nV ≥ d1C log

8Hα3d31/2 ϵ

2Hαd1√d2 ϵ

,

,

for some universal constant C > 0.

- Figure 3 shows an overview of our proof. To prove Theo-

rem 3, we begin by focusing on the part before the softmax, the target and source inner products for the j-th head:

1 √dK

1 √dK

(x⊤i W(Qj))(XW(Kj))⊤, (3) 1 √nK

q(j)K(j)⊤ =

(x⊤i (W˜ (Qj) ⊙ M(Qj)))(X(W˜ (Kj) ⊙ M(Kj)))⊤. (4)

Since the only difference lies in the projection matrices, we consider the problem of pruning the source projections

W˜ (Qj) and W˜ (Kj) to approximate the target projections W(Qj) and W(Kj). A naive idea might be to approximate each target projection independently. In this case, a single source random matrix must approximate each target matrix. However, pruning a single random matrix cannot generally approximate arbitrary ones; thus, this approach is infeasible. To overcome this limitation, we revisit the structure of the target inner product. By closely examining the formulation of the target inner product (Equation (3)), we observe that the query and (transposed) key projections appear adjacently and can be merged into a single joint projection (the right panel of Figure 3):

1 √dK

(x⊤i W(Qj))(XW(Kj))⊤ = x⊤i W(QKj) X⊤,

1 √dk

W(Qj)(W(Kj))⊤. (5)

W(QKj) :=

This reformulation enables us to reinterpret the original problem—not as approximating two target matrices—but as approximating a single merged projection matrix.

We now approximate this merged matrix W(QKj) by pruning the two source projections. On the source side (Equation (4)) as well, the query and key projections are adjacent. Thus, the source inner product can be viewed as a computation that first calculates the query and key projections (the left panel of Figure 3):

- 1

(x⊤i (W˜ (Qj) ⊙ M(Qj)))(X(W˜ (Kj) ⊙ M(Kj)))⊤

√nK

= x⊤i (W˜ ′Q(j) ⊙ M(Qj))(W˜ ′K(j) ⊙ M(Kj))⊤ X⊤, W˜ ′Q(j) :=

1 n1K/4

1 n1K/4

W˜ (Qj), W˜ ′K(j) :=

W˜ (Kj),

where each entry of W˜ ′Q(j) and W˜ ′K(j) is drawn i.i.d. from U[−1,1] as per our assumption. Therefore, the task reduces

to selecting masks M(Qj) and M(Kj) such that the source matrix product (W˜ ′Q(j) ⊙ M(Qj))(W˜ ′K(j) ⊙ M(Kj))⊤ closely approximates the target W(QKj) . This allows us to draw an analogy to the conventional theoretical results of the SLTH, particularly the two-layers-for-one approximation (Lemma 2). We therefore establish and apply a variant of Lemma 2, which guarantees the existence of binary pruning masks that achieve such an approximation (the bottom panel of Figure 3):

Softmax Softmax

Source MHA

###### Target MHA

Change calculation order

Merge weights

Softmax

Softmax

Approximate target weights by two-layer-for-one approximation

Figure 3: The diagram of our proof. By merging the target projections and changing the calculation order of the source MHA, we can apply a variant of the two-layers-for-one approximation technique and approximate the target MHA while keeping the original source and target structures.

###### Lemma 4. Let W ∈ Rd

2×d1 be a target matrix with ∥W∥ ≤ 1, and W˜ 1 ∈ Rn×d

1 and W˜ 2 ∈ Rd

2×n be source matrices whose entries are drawn i.i.d. from U[−1,1]. Suppose that n ≥ d1C log(d1d2/ϵ) for some universal constant C > 0. Then, with probability at least 1 − ϵ, there exists a choice of binary pruning masks M1 and M2 such that

ϵ d1d2

###### W − (W˜ 2 ⊙ M2)(W˜ 1 ⊙ M1)

≤

.

max

We now turn to the components after the softmax function: the value and output projections. Similar to the query and key case, the value and output projections appear adjacently and can also be merged into a single composite transformation. Thus, we aim to approximate the target merged

matrix W(VOj) := W(Vj)W(Oj). This approximation follows the same principle as before: we leverage the matrix product

on the source side (W˜ (Vj)⊙M(Vj))(W˜ (Oj)⊙M(Oj)) to approximate the merged matrix W(VOj) . Lemma 4 ensures that, with high probability, this approximation is successful via appropriately chosen binary pruning masks M(Vj) and M(Oj).

Assuming that all weights in the target MHA are approximated by the above procedure, we next analyze the error of the entire attention mechanism by investigating the behavior of the softmax. As a natural idea, one might consider exploiting the 1-Lipschitz continuity of the softmax (Gao and

Pavel 2017), which enables internal errors to propagate linearly to the output. However, since the MHA subsequently multiplies the softmax output and the input matrix X, applying Lipschitz continuity results in a loose upper bound of the error between MHAs: as ∥X∥ can grow with T in the worst case, the bound depends on the input length T.

In contrast to this general approach, we provide a more precise analysis. In our setting, thanks to the accurate weight approximation technique mentioned earlier, the internal error of softmax is guaranteed to be finite and small. Leveraging this property, we analyze the softmax output and X simultaneously to obtain a T-independent bound as follows:

###### Lemma 5. Let ϵ ∈ Rd

1 be an error vector with ∥ϵ∥max ≤ ϵmax for some 0 ≤ ϵmax ≤ 1/2. Then,

∥σ(xi;ai)X − σ(xi + ϵ;ai)X∥ ≤ 4 d1αϵmax.

max

i∈[T]

Since this lemma provides a bound independent of ai, our theory holds for models with arbitrary attention masks, including encoder (Devlin et al. 2019) and decoder models (Radford et al. 2019). By applying these above analyses to each attention head, we complete the proof of Theorem 3. For the full proof, see Section A.3. We also empirically validate two main theoretical findings in Section 4.2: the accurate approximation of the target MHA becomes feasible with larger source hidden dimensions, and the the approximation error remains independent of the input length T.

Proof Sketch of Theorem 3: First, for each attention head, we reformulate the problem by merging the four original target projection matrices into two merged matrices: one combining the query and key projections, and the other the value and output projections. Applying Lemma 4 to these merged matrices enables us to prune each source head to produce an inner product that closely approximates the target one. Next, using Lemma 5, we bound how errors in approximating the query and key matrices propagate through the softmax operation. Lemma 5 ensures that the approximation error of the softmax depends on the approximation accuracy of the query-key projections and does not scale with the input length T. Thus, provided the source hidden dimensions nK and nV are sufficiently large, there exists a choice of binary masks for the source MHA which approximate the target MHA within an error ϵ. Also, by suitably setting lower bounds on nK and nV, a union bound guarantees that the approximation succeeds across all heads with probability at least 1 − ϵ.

#### 3.3 The Existence of SLTs Within a Transformer

By leveraging our main theorem, we now extend the SLTH to transformers. We consider a transformer without the normalization layers for the original definition (Vaswani et al. 2017). The target transformer of B blocks are of the following form:

TfT(xi) := Blk(TB)(Blk(TB−1) ...Blk(1)T (xi)), Blk(Tb)(x(ib)) := F(Tb)(AttnT(x(ib))⊤ + x(ib))

+ AttnT(x(ib))⊤ + x(ib),

where Blk(Tb) is a b-th target block, and x(ib) ∈ Rd is the i-th input embedding of the b-th target block. We employ

single-layer projection F(Tb)(·) for the fully-connected network of each target block. Similarly, we define the pruned

source transformer as follows:

TfS(xi) := Blk(SB)(Blk(SB−1) ...Blk(1)S (xi)), Blk(Sb)(x′i(b)) := F(Sb)(AttnS(x′i(b))⊤ + x′i(b))

+ AttnS(x′i(b))⊤ + x′i(b),

where Blk(Sb)(·) is a b-th source block and x′i(b) is the i-th input embedding of the b-th source block. We set the hidden

dimension of F(Sb)(·) as n(FCb) and assume the hidden dimensions of the MHA in the b-th source block as a same value

n(MHAb) for simplicity. Then, we prove the following theorem: Theorem 6. Assume B ≥ 2. Then, with probability at least 1 − ϵ for 0 < ϵ < 1, there exists a choice of binary pruning masks that satisfies

∥TfS(xi) − TfT(xi)∥ ≤ ϵ,

if the hidden dimensions of b-th source MHA and fullyconnected network satisfy

n(MHAb) ≥ d1C log

n(FCb) ≥ d1C log

cf11(b,B)Hf

2(b,B)d1f3(b,B) ϵ

cg21(b,B)Hg

2(b,B)dg13(b,B) ϵ

,

,

for universal constants C > 0 and c1,c2 > 0 including α. Here, f1,f2,f3,g1,g2,g3 are quadratic forms of b and B.

Proof Sketch of Theorem 6: From the existing work (Lemma 2) and Theorem 3, we already know that an MHA and FFN contain SLTs with high probability if each module has a large hidden dimension; thus, by determining the lower bound of the hidden dimension of each module based on the error propagation from the input to output, we can prove that there exists an SLT, which approximates the output of an target transformer to an error of ϵ, within a randomly initialized transformer. By the union bound, the probability that all approximations hold simultaneously is at least 1 − ϵ.

For simplicity, this theorem uses target and source fullyconnected networks as a single-layer and two-layer ReLU networks FT and FS in Lemma 2. It can be generalized to an L-layer target fully-connected network by applying the multi-layer approximation by Pensia et al. (2020). We show that theorem and its proof in Section A.5.

### 4 Experimental Results

This section empirically validates our SLTH theorems.

#### 4.1 Experimental Settings

To empirically validate the approximation guarantees established by our SLTH theorems, we evaluate the approximation error on a synthetic dataset for angular velocity estimation. The input consists of a sequence of two-dimensional

1.0

Curve Fitting: = O(exp( n))

ApproximationError

Subset-Sum Approximation

0.8

0.6

0.4

0.2

0.0

0 20 40 60 80 100

Hidden Dimensions nK and nV

- Figure 4: The approximation error ϵ of SLTs within a

source MHA for the hidden dimensions nK = nV. This result shows that the error ϵ satisfies ϵ = O(exp(−n)), consistently with Theorem 3.

vectors arranged on the unit circle with a fixed angular velocity. A regression token is used to estimate this velocity, and the source model uses the same regression token as the target model to ensure input consistency. The source and target models are both implemented as either single-head attention mechanisms or single-head transformers as defined in Section 3.3. Both models are initialized according to our theoretical setup: the entries of the query and key projection

weights are drawn i.i.d. from U[−n1K/4,n1K/4], and those of the value and output projection weights from U[−1,1]. To

identify SLTs that approximate the target network, we implement the weight approximation technique described in Lemma 4, which is based on the subset-sum approximation of Pensia et al. (2020). The target MHA is approximated using 100 randomly initialized source MHAs, and we report the mean and standard deviation of the approximation error.

We also investigate whether our theoretical insights generalize to practical settings. In this setting, we search for SLTs by the edge-popup algorithm (Ramanujan et al. 2020), which finds accurate subnetworks by backpropagation, instead of learning weights. We train models from the GPT-2 family (mini1, small, and medium) (Radford et al. 2019) on the WikiText-103 dataset (Merity et al. 2017). The weights of these models are initialized based on the GPT-

- 2 initialization scheme. For each model, we repeat training three times with different random seeds and report the mean and standard deviation of the final performance. See Section B for further details on experimental settings.

#### 4.2 Verification of Main Theorems

We empirically verify our theoretical results by pruning a source network to approximate the target network.

Varying the Hidden Dimensions: We validate Theorem 3 by showing that increasing the hidden dimensions leads to an exponential decrease in approximation error. When we fit the empirical results to the function ϵ = γ exp(−δnK), we obtain ϵ = 0.8exp(−0.06nK), which closely matches

1A 4-layer GPT-2. For details, see the following repository: https://huggingface.co/erwanf/gpt2-mini (Wolf et al. 2020)

0.8

Hidden Dimension

ApproximationError

20 40

0.6

0.4

0.2

0.0

23 25 27 29 211 213

Input Lengths T

- Figure 5: The approximation error ϵ of SLTs within an MHA for the sequence length T. This result suggests that the error ϵ does not diverge as T increases, as implied by Theorem 3.

100 125 150 175 200 225

Hidden Dimensions nMHA and nFC

0.0

0.2

0.4

0.6

0.8

ApproximationError

# of Blocks

- 1

- 2

- 3

- 4

Fitting

- Figure 6: The approximation error ϵ of SLTs within a randomly initialized transformer for the source hidden dimen-

sions nMHA = nFC. This result suggests that error accumulates as the number of blocks increases, while each error

holds ϵ = O(exp(−nMHA)), consistently with Theorem 3.

the observations. This finding supports our theoretical claim: given a target MHA, each source hidden dimension requires O(log(1/ϵ)) for the existence of SLTs.

Varying the Sequence Length: Theorem 3 also implies that the existence of SLTs in MHAs is independent of the input length T. In other words, with sufficiently large hidden dimensions, the approximation error has an upper bound that does not depend on T. Figure 5 empirically supports this argument: even as T increases, the error remains bounded, and the bound decreases with larger hidden dimensions.

Varying the Number of Blocks: To validate Theorem 6, we analyze how the approximation error behaves across different numbers of transformer blocks. We set nMHA = nFFN and use an untrained target model to be close to our theoretical assumptions. As in the MHA experiment, we fit an exponential decay ϵ = γ exp(−δnK) to the error of each block, using the same decay rate δ obtained from the first block, as predicted by Theorem 6. Figure 6 shows that, consistent with our theoretical implication, the approximation error decreases rapidly with increasing hidden dimensions for all numbers of blocks. Despite fitting only the coefficient

Method:

Trained Model

SLT

SLT w/ Weight Scaling

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Model (Width):

4.2

- Mini (×0.5)

- Mini (×1.0)

Val.Loss

Small (×1.0)

3.8

- Medium (×1.0)

- Medium (×2.0)

| |
|---|

3.4

| |
|---|
| |

3.0

107 108

# of Param.

Figure 7: Loss comparison between SLTs with and without query and key weight scaling. By introducing a scale based on our theoretical assumptions, we can obtain better SLTs.

γ per block, the shared δ provides curves that closely match the empirical results, supporting our theoretical claim that only the scale factor varies across blocks.

#### 4.3 Behavior of SLTs in Practical Settings

In the theoretical analysis, we employ a non-conventional initialization strategy: the query and key projection weights

are initialized from U[−n1K/4,n1K/4], scaled by a factor of n1K/4 compared to the value and output weights, which are initialized from U[−1,1]. This weight scaling was introduced to facilitate the application of the weight approximation lemma in our analysis, and played an important role in establishing our theory. Its theoretical contribution motivates the following question: does this scaled initialization strategy also benefit SLTs in realistic scenarios? We empirically evaluate SLTs using the GPT-2 architectures and the WikiText-103 dataset. Figure 7 compares the validation loss of SLTs with and without scaling the query and key weights by n1K/4 ≃ 2.8, with respect to the number of nonzero parameters. We observe that SLTs with the weight scaling tend to exhibit lower loss, approaching the performance of trained models. Interestingly, this specific scaling factor n1K/4 is nearly optimal for finding better SLTs: as shown in Figure 8, increasing the scale from 1 gradually decreases the loss up to a certain point, but further increasing it beyond n1K/4 results in increased loss. In all models, the lowest loss is consistently achieved around this scaling factor n1K/4. These findings suggest that our initialization strategy actually helps to ensure the existence of better SLTs within the practical transformer models.

### 5 Related Work

Strong Lottery Tickets: Zhou et al. (2019) and Ramanujan et al. (2020) empirically found the subnetworks that achieve high accuracy without any weight training. The existence of such high-performing subnetworks has been called the strong lottery ticket hypothesis (SLTH), and its theoretical proof was firstly provided in fully-connected ReLU networks (Malach et al. 2020; Orseau, Hutter, and Rivasplata

Model:

Mini

Small

Medium

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Original Scale

nK1/4 2.8

4.2

4.0

Val.Loss

3.8

3.6

3.4

3.2

1 2 3 4 5

Scale of Weights WQ and WK

Figure 8: Loss comparison with respect to the weight scaling factor applied to query and key weights. Interestingly, in all models, the loss reaches its minimum near the weight scaling of our theoretical assumptions.

- 2020; Pensia et al. 2020; Burkholz 2022b). Based on these pioneering studies, the SLTH has been ex-

tended in three main directions. The first direction involves introducing additional flexibility for relaxing the overparameterization of the source network (Chijiwa et al. 2021; Xiong, Liao, and Kyrillidis 2023). The second direction, in contrast, imposes additional constraints on the source network (Gadhikar, Mukherjee, and Burkholz 2023; Otsuka et al. 2025; Natale et al. 2024). The third direction extends the SLTH to various architectures (Diffenderfer and Kailkhura 2021; Burkholz 2022b; Fischer and Burkholz

- 2021; da Cunha, Natale, and Viennot 2022; Da Cunha and d’Amore 2023; Burkholz 2022a; Ferbach et al. 2023). Our work contributes to this third direction by proving the SLTH for attention mechanisms and transformers.

Randomly Weighted Transformers: Several studies have empirically investigated the capabilities of randomly weighted transformers. Shen et al. (2021a) demonstrated that a transformer with a few randomly weighted layers achieves accuracy comparable to fully trained models on translation and language understanding tasks. Zhong and Andreas (2024) found that randomly weighted transformers can solve toy tasks with high accuracy as the hidden dimension increases. Some studies empirically showed the existence of SLTs within randomly weighted transformers (Shen et al. 2021b; Ito et al. 2025). Our analysis provides theoretical support for these empirical results about the SLT existence. Furthermore, it provides a theoretical explanation for the improved performance of randomly weighted transformers as the hidden dimension increases, particularly when the pruning is used for optimization.

### 6 Conclusion

This work investigated the existence of SLTs within a multihead attention (MHA) mechanism. We extended the existing theory of the SLTH to MHAs and proved that, if the source MHA has logarithmically large hidden dimensions, it contains an SLT that approximates an arbitrary MHA with high probability. Our proof revealed that, for the SLTH in

MHAs, additional layers are not required for approximation, in contrast to the existing theories that rely on approximating a single-layer structure by a two-layer one. Furthermore, by exploiting our findings, we established the theory of the SLTH for transformers without normalization layers. We empirically validated our theory and confirmed that the results are consistent with the theoretical implications. Interestingly, our theoretical implication, which provides an appropriate weight scale for initializing query and key projection weights, contributed to improving the performance of SLTs in practical settings. Our results not only extend SLTH to transformers, but also indicate a new research direction in the SLTH for practical transformer models. We hope these findings will lead to a fundamental understanding of overparameterized models.

### Acknowledgments

This work was supported in part by JSPS KAKENHI Grant Number JP23H05489, JP25K03092, JP25KJ1236, and JSTALCA-Next Japan Grant # JPMJAN24F3.

### References

- Burkholz, R. 2022a. Convolutional and residual networks provably contain lottery tickets. In International Conference on Machine Learning, 2414–2433. PMLR.
- Burkholz, R. 2022b. Most activation functions can win the lottery without excessive depth. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., Advances in Neural Information Processing Systems.

Chijiwa, D.; Yamaguchi, S.; Ida, Y.; Umakoshi, K.; and Inoue, T. 2021. Pruning randomly initialized neural networks with iterative randomization. Advances in neural information processing systems, 34: 4503–4513.

Da Cunha, A.; and d’Amore, F. 2023. Polynomially overparameterized convolutional neural networks contain structured strong winning lottery tickets. Advances in Neural Information Processing Systems, 36: 25929–25957.

da Cunha, A.; Natale, E.; and Viennot, L. 2022. Proving the strong lottery ticket hypothesis for convolutional neural networks. In ICLR 2022-10th International Conference on Learning Representations.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 4171–4186.

Diffenderfer, J.; and Kailkhura, B. 2021. Multi-prize lottery ticket hypothesis: Finding accurate binary neural networks by pruning a randomly weighted network. In International Conference on Learning Representations.

Ferbach, D.; Tsirigotis, C.; Gidel, G.; and Bose, J. 2023. A general framework for proving the equivariant strong lottery ticket hypothesis. In The Eleventh International Conference on Learning Representations.

Fischer, J.; and Burkholz, R. 2021. Towards strong pruning for lottery tickets with non-zero biases. arXiv preprint arXiv:2110.11150.

Frankle, J.; and Carbin, M. 2019. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations.

Gadhikar, A. H.; Mukherjee, S.; and Burkholz, R. 2023. Why random pruning is all we need to start sparse. In International Conference on Machine Learning, 10542–10570. PMLR.

Gao, B.; and Pavel, L. 2017. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805.

Glorot, X.; and Bengio, Y. 2010. Understanding the difficulty of training deep feedforward neural networks. In Teh, Y. W.; and Titterington, M., eds., Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, 249–256. Chia Laguna Resort, Sardinia, Italy: PMLR.

Gurobi Optimization, LLC. 2024. Gurobi optimizer reference manual.

Ito, H.; Yan, J.; Otsuka, H.; Kawamura, K.; Motomura, M.; Chu, T. V.; and Fujiki, D. 2025. Uncovering strong lottery tickets in graph transformers: A path to memory efficient and robust graph learning. Transactions on Machine Learning Research.

Loshchilov, I.; and Hutter, F. 2017. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations.

Loshchilov, I.; and Hutter, F. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Lueker, G. S. 1998. Exponentially small bounds on the expected optimum of the partition and subset sum problems. Random Structures & Algorithms, 12(1): 51–62.

Malach, E.; Yehudai, G.; Shalev-Schwartz, S.; and Shamir, O. 2020. Proving the lottery ticket hypothesis: Pruning is all you need. In International Conference on Machine Learning, 6682–6691. PMLR.

Merity, S.; Xiong, C.; Bradbury, J.; and Socher, R. 2017. Pointer sentinel mixture models. In International Conference on Learning Representations.

Natale, E.; Ferre’, D.; Giambartolomei, G.; Giroire, F.; and Mallmann-Trenn, F. 2024. On the sparsity of the strong lottery ticket hypothesis. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Orseau, L.; Hutter, M.; and Rivasplata, O. 2020. Logarithmic pruning is all you need. Advances in Neural Information Processing Systems, 33: 2925–2934.

Otsuka, H.; Chijiwa, D.; Garc´ıa-Arias, A.´ L.; Okoshi, Y.; Kawamura, K.; Chu, T. V.; Fujiki, D.; Takeuchi, S.; and Motomura, M. 2025. Partially frozen random networks contain compact strong lottery tickets. Transactions on Machine Learning Research.

Pensia, A.; Rajput, S.; Nagle, A.; Vishwakarma, H.; and Papailiopoulos, D. 2020. Optimal lottery tickets via subset sum: Logarithmic over-parameterization is sufficient. Advances in neural information processing systems, 33: 2599– 2610.

Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.

Ramanujan, V.; Wortsman, M.; Kembhavi, A.; Farhadi, A.; and Rastegari, M. 2020. What’s hidden in a randomly weighted neural network? In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11893–11902.

Shen, S.; Baevski, A.; Morcos, A.; Keutzer, K.; Auli, M.; and Kiela, D. 2021a. Reservoir transformers. In Zong, C.; Xia, F.; Li, W.; and Navigli, R., eds., Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 4294– 4309. Online: Association for Computational Linguistics.

Shen, S.; Yao, Z.; Kiela, D.; Keutzer, K.; and Mahoney, M. 2021b. What‘s hidden in a one-layer randomly weighted transformer? In Moens, M.-F.; Huang, X.; Specia, L.; and Yih, S. W.-t., eds., Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2914– 2921. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Virtanen, P.; Gommers, R.; Oliphant, T. E.; Haberland, M.; Reddy, T.; Cournapeau, D.; Burovski, E.; Peterson, P.; Weckesser, W.; Bright, J.; van der Walt, S. J.; Brett, M.; Wilson, J.; Millman, K. J.; Mayorov, N.; Nelson, A. R. J.; Jones, E.; Kern, R.; Larson, E.; Carey, C. J.; Polat, ˙I.; Feng, Y.; Moore, E. W.; VanderPlas, J.; Laxalde, D.; Perktold, J.; Cimrman, R.; Henriksen, I.; Quintero, E. A.; Harris, C. R.; Archibald, A. M.; Ribeiro, A. H.; Pedregosa, F.; van Mulbregt, P.; and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental algorithms for scientific computing in python. Nature Methods, 17: 261–272.

Wolf, T.; Debut, L.; Sanh, V.; Chaumond, J.; Delangue, C.; Moi, A.; Cistac, P.; Rault, T.; Louf, R.; Funtowicz, M.; Davison, J.; Shleifer, S.; von Platen, P.; Ma, C.; Jernite, Y.; Plu, J.; Xu, C.; Le Scao, T.; Gugger, S.; Drame, M.; Lhoest, Q.; and Rush, A. 2020. HuggingFace’s transformers: State-of-the-art natural language processing. In Liu, Q.; and Schlangen, D., eds., Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. Online: Association for Computational Linguistics.

Xiong, Z.; Liao, F.; and Kyrillidis, A. 2023. Strong lottery ticket hypothesis with ε-perturbation. In International Conference on Artificial Intelligence and Statistics, 6879–6902. PMLR.

Zhong, Z.; and Andreas, J. 2024. Algorithmic capabilities of random transformers. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Zhou, H.; Lan, J.; Liu, R.; and Yosinski, J. 2019. Deconstructing lottery tickets: Zeros, signs, and the supermask. Advances in neural information processing systems, 32.

### A Proofs of Main Theorems

This section presents the detailed proofs of the main theorems in the manuscript. We first introduce two lemmas: one for approximating a target weight matrix by pruning two random weight matrices, and another for bounding the effect of perturbations in the softmax function. These lemmas are then used to establish the SLTH for attention mechanisms. Then, leveraging the theory of the SLTH for attention mechanisms, we prove the existence of SLTs in transformer blocks and transformers without normalization layers.

#### A.1 Weight Approximation

Pensia et al. (2020) have shown that a two-layer fully-connected ReLU network can approximate arbitrary matrices with high probability. Our problem setting can be viewed as a simplified version of their construction, in which the ReLU nonlinearity is omitted. We follow their proof strategy and simplify it to the linear (non-activated) case.

2×d1 be a target matrix with entries in [−1,1]. Let W˜ 1 ∈ Rn×d

1 and W˜ 2 ∈ Rd

- Lemma 7. Let W ∈ Rd

2×n be source random matrices whose entries are drawn i.i.d. from U[−1,1]. For any 0 < ϵ < 1, suppose that n ≥ d1C log(d

1d2

ϵ ) for some universal constant C > 0. Then, with probability at least 1 − ϵ, there exists a choice of binary masks M1 ∈ {0,1}n×d

1

and M2 ∈ {0,1}d

2×n such that

ϵ d1d2

###### W − (W˜ 2 ⊙ M2)(W˜ 1 ⊙ M1)

≤

.

max

Proof. Firstly, we structurally prune the random weight matrix W˜ 1 by the pruning mask M1:

  

  , (6)

u1 0 ··· 0 0 u2 ··· 0

W˜ 1 ⊙ M1 =

... .

. .

0 0 ··· ud

1

′

. Next, we decompose W˜ 2 ⊙ M2 as follows:

where ui ∈ Rn





(v1,1 ⊙ m1,1)⊤ (v1,2 ⊙ m1,2)⊤ ··· (v1,d

)⊤ (v2,1 ⊙ m2,1)⊤ (v2,2 ⊙ m2,2)⊤ ··· (v2,d

1 ⊙ m1,d

1

)⊤

1 ⊙ m2,d

W˜ 2 ⊙ M2 =

1

, (7)

 

 

... . (vd

. .

2,1)⊤ (vd

2,2)⊤ ··· (vd

2,d1)⊤

2,1 ⊙ md

2,2 ⊙ md

2,d1 ⊙ md

′

′

where vi,j ∈ Rn

and mi,j ∈ {0,1}n

. These operations enable us to rewrite the product of Equations (6) and (7) as follows:





(v1,1 ⊙ m1,1)⊤ u1 (v1,2 ⊙ m1,2)⊤ u2 ··· (v1,d

)⊤ ud

1 ⊙ m1,d

1

1

(v2,1 ⊙ m2,1)⊤ u1 (v2,2 ⊙ m2,2)⊤ u2 ··· (v2,d

)⊤ ud

1 ⊙ m2,d

(W˜ 2 ⊙ M2)(W˜ 1 ⊙ M1) =

1

1

(8)

 

 

... . (vd

. .

2,1)⊤ u1 (vd

2,2)⊤ u2 ··· (vd

2,d1)⊤ ud

2,1 ⊙ md

2,2 ⊙ md

2,d1 ⊙ md

1

We focus on the (i,j)-th entry of Equation (8). This entry can be rewritten as a subset sum of element-wise products between the vectors vi,j and uj:

n′

(vi,j ⊙ mi,j)⊤ uj =

mi,j,kvi,j,kuj,k.

k=1

Here, each mi,j,k determines whether the corresponding product vi,j,kuj,k is included in the subset sum. We aim to approximate the (i,j)-th entry of the target weight matrix W with the subset sum n

′

k=1 mi,j,kvi,j,kuj,k by appropriately choosing the binary mask mi,j. Since each entry of vi,j,k and uj,k is drawn i.i.d. from U[−1,1], each product vi,j,kuj,k can be viewed as drawn from the distribution including some uniform distribution; thus, we can apply Corollary 3.3 of Lueker (1998), which states that if n′ ≥ C log d

1d2

ϵ , then with probability at least 1 − d ϵ

1d2, there exists a binary mask vector mi,j such that the subset sum

n′ k=1 mi,j,kvi,j,kuj,k approximates the (i,j)-th entry of W within an error of d ϵ

1d2 . By the union bound, the probability that all entries of the weight matrix W are simultaneously approximated is at least 1 − ϵ:

d1

d2

ϵ d1d2

= 1 − ϵ.

1 −

i=1

j=1

Therefore, if n = d1n′ ≥ d1C log d

1d2

ϵ , then with probability at least 1 − ϵ, the following inequality holds:

ϵ d1d2

###### W − (W˜ 2 ⊙ M2)(W˜ 1 ⊙ M1)

≤

.

max

| |
|---|

#### A.2 Spectral Norm of Softmax Difference

In addition to approximating target weights, we need to analyze the stability of the softmax output under small input perturbations, with respect to the spectral norm of the resulting attention-weighted output.

- Lemma 8. Given ϵ ∈ Rd

1 as a perturbation vector such that ∥ϵ∥max ≤ ϵmax for some ϵmax ≥ 0, we have

∥σ(xi;ai)X − σ(xi + ϵ;ai)X∥ ≤ d1α (exp(2ϵmax) − 1). Proof. Let pi := σ(xi;ai) and p′i := σ(xi + ϵ;ai). Then, for each coordinate j, we have

exp(ϵj) Z

p′i,j = pi,j

,

T

Z =

pi,k exp(ϵk).

k=1

By the assumption ∥ϵ∥max ≤ ϵmax, we have the following bound:

exp(ϵj)

Z ≤ exp(2ϵmax) − 1. (9) Now, we can bound the spectral norm for the i-th input embedding:

1 −

∥piX − p′iX∥ ≤ d1 · ∥piX − p′iX∥max

T

(pi,k − p′i,k)xk,j

≤ d1 · max j∈[d1]

k=1

T

|xk,j| · |pi,k − p′i,k|

≤ d1 · max j∈[d1]

k=1

T

|pi,k − p′i,k|

≤ d1 · α

k=1

T

exp(ϵk) Z

≤ d1 · α

pi,k 1 −

k=1

T

pi,k (Using Equation (9))

≤ d1 · α (exp(2ϵmax) − 1)

k=1

= d1 · α (exp(2ϵmax) − 1).

This upper bound is independent of i; thus, the upper bound of maxi∈[T] ∥pX − p′X∥ is same as the final upper bound.

| |
|---|

##### A.3 SLT Existence within Attention Mechanisms By leveraging these two lemmas, we prove the following theorem:

- Theorem 9. Let AttnS(·) and AttnT(·) be as defined in Equations (1) and (2). Assume α ≥ max(√d1,√d2) for the inputs. Then, with probability at least 1 − ϵ, there exists a choice of binary masks M(Qj),M(Kj),M(Vj),M(Oj) that satisfy

∥AttnS(xi) − AttnT(xi)∥ ≤ ϵ, if the source dimensions satisfy

max

i∈[T]

- n1 ≥ d1C log

8Hα3d31/2 ϵ

,

- n2 ≥ d1C log

2Hαd1√d2 ϵ

, for some universal constant C > 0.

Proof. We prove the theorem in three steps.

- Step 1: Weight Merging. We begin by merging the weight matrices of the target and source MHAs. The target MHA weights are merged as

W(QKj) :=

1 √dK

W(Qj)(W(Kj))⊤, (10)

W(VOj) := W(Vj)W(Oj). (11) This operation (Equations (10) and (11)) enables us to represent each head of AttnT(·) as

AttnT(xi;X,W(1:Q:OH)) =

H

j=1

σ x⊤i W(QKj) X⊤;ai XW(VOj) .

From the assumption on the target weights, we have the following norm bounds:

∥W(QKj) ∥ ≤ 1/ dK, (12) ∥W(VOj) ∥ ≤ 1. (13)

For the source MHA, we incorporate the scaling factor 1/√nK into the query and key weight matrices:

W˜ ′Q(j) :=

1 d1K/4

W˜ (Qj),

W˜ ′K(j) :=

1 d1K/4

W˜ (Kj).

Assuming that each entry of W˜ (Qj) and W˜ (Kj) is drawn i.i.d. from U[−d1K/4,d1K/4], each entry of the scaled matrices W˜ ′Q(j) and W˜ ′K(j) is drawn i.i.d. from U[−1,1].

- Step 2: Weight Approximation. From Lemma 7, for any 0 < ϵ < 1, if

nK ≥ d1C log

8Hα3d31/2 ϵ

,

then with probability at least 1 − 8Hαϵ3√d1 , there exists a choice of binary masks M(Qj) and M(Kj) such that

⊤

ϵ 8Hα3d31/2

W(QKj) − W ˜ ′Q(j) ⊙ M(Qj) W ˜ ′K(j) ⊙ M(Kj)

. (14)

≤

max

This inequality Equation (14) implies a bound on the softmax input:

⊤

x⊤i W(QKj) X⊤ − x⊤i W ˜ ′Q(j) ⊙ M(Qj) W ˜ ′K(j) ⊙ M(Kj)

###### X⊤

∞

⊤

x⊤i W(QKj) xk − x⊤i W ˜ ′Q(j) ⊙ M(Qj) W ˜ ′K(j) ⊙ M(Kj)

= max

###### xk

k∈[T]

⊤

≤ α2 W(QKj) − W ˜ ′Q(j) ⊙ M(Qj) W ˜ ′K(j) ⊙ M(Kj)

⊤

≤ α2d1 W(QKj) − W ˜ ′Q(j) ⊙ M(Qj) W ˜ ′K(j) ⊙ M(Kj)

max ≤ α2d1

ϵ 8Hα3d31/2

ϵ 8Hα√d1

=

. Let

p(ij) := σ x⊤i W(QKj) X⊤;ai ,

p′i(j) := σ x⊤i (W˜ ′Q(j) ⊙ M(Qj))(W˜ ′K(j) ⊙ M(Kj))⊤X⊤;ai . Applying Lemma 8, we obtain

ϵ

p(ij)X − p′i(j)X ≤ d1α exp

4Hα√d1 − 1 ≤

ϵ 2H

< 1) For the value and output weights, from Lemma 7, if

. (Using 0 < 4Hαϵ√d

1

2Hαd1√d2 ϵ2

nV ≥ d1C log

,

√d2ϵ 2Hα , there exists a choice of binary pruning masks M(Vj) and M(Oj) such that

then with probability at least 1 −

ϵ 2Hαd1√d2

###### W(VOj) − (W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))

. (15)

≤

max

- Step 3: Total Error Analysis. We now bound the difference between the outputs of the source and target MHAs:

∥AttnT(xi) − AttnS(xi)∥ =

H

j=1

p(ij)XW(VOj) − p′i(j)X(W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))

H

p(ij)XW(VOj) − p′i(j)X(W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj)) . We apply the triangle inequality:

≤

j=1

p(ij)XW(VOj) − p′i(j)X(W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))

≤ (p(ij) − p′i(j))XW(VOj) + p′i(j)X(W(VOj) − (W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))) . For the first term, by using ∥W(VOj) ∥ ≤ 1 (Equation (13)), we obtain

(p(ij) − p′i(j))XW(VOj) ≤ (p(ij) − p′i(j))X W(VOj) ≤ (p(ij) − p′i(j))X ≤

ϵ 2H

. (16) For the second term, we obtain the following result by using Equation (15):

p′i(j)X(W(VOj) − (W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))) ≤ d1 p′i(j)X

d1d2 W(VOj) − (W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))

∞

max ≤ d1 d2α W(VOj) − (W˜ (Vj) ⊙ M(Vj))(W˜ (Oj) ⊙ M(Oj))

max ≤ d1 d2α

ϵ 2Hαd1√d2

(Using Equation (15))

ϵ 2H

. (17) These results of Equations (16) and (17) do not depend on the input index i; thus, adding the two terms across H heads gives

=

H

ϵ 2H

ϵ 2H

∥AttnT(xi) − AttnS(xi)∥ ≤

max

+

= ϵ.

i∈[T]

j=1

Finally, using the union bound and the assumption α ≥ max(√d1,√d2), the probability that all approximations hold is

√d2ϵ 2Hα ≥ 1 − ϵ.

ϵ 8Hα3√d1 −

1 −

| |
|---|

#### A.4 SLT Existence Within Transformer Blocks

By combining the SLT existence theorem for attention mechanisms (Theorem 9) and for multi-layer fully-connected ReLU networks (FC) proven by Pensia et al. (2020) (Theorem 10), we prove the SLT existence theorem for transformer blocks.

- Theorem 10 (Theorem 1 in Pensia et al. (2020)). Let FT (xi) = WLReLU(WL−1 ...ReLU(W1xi))

be a target FC with L layers. Assume that each weight matrix Wl ∈ Rd

l+1×dl satisfies |Wl| ≤ 1 for all l = 1,...,L. Consider a pruned source FC with 2L layers defined as

FS (xi) = W ˜ 2L ⊙ M2L ReLU W ˜ 2L−1 ⊙ M2L−1 ...ReLU W ˜ 2L−1 ⊙ M2L−1 xi , where W˜ 2l−1 ∈ Rn

l×dl and W˜ 2l ∈ Rd

l+1×nl for l = 1,...,L, and each entry of W˜ l is drawn i.i.d. from U[−1,1]. Then, with probability at least 1 − ϵ for any 0 < ϵ < 1, there exists a choice of binary pruning masks M1,...,M2L that holds the following inequality:

∥FS(xi) − FT(xi)∥ ≤ exp

αϵ

2 − 1, if each source dimension nl satisfies

nl ≥ dlC log

4Ldldl+1 ϵ

, for some universal constant C > 0.

We now state the main result for transformer blocks, which follows from combining the two SLT existence theorems.

- Theorem 11. Let BlkT(xi) = FT(AttnT(xi)⊤ + xi) + AttnT(xi)⊤ + xi

be a target transformer block of an MHA AttnT(·) and FC FT(·) with L layers. For simplicity, we assume each layer of target FC dimensions is all d1. Let

BlkS(xi) = FS(AttnS(xi)⊤ + xi) + AttnS(xi)⊤ + xi

be a pruned random source transformer block of a pruned random MHA AttnS(·) and FC FS(·) with 2L layers. Assume that the input dimension of each even layer of the source FC is nFC, and the input dimension of each odd layer is d1. Furthermore, for simplicity, we assume key and value dimensions of AttnS(·) are the same dimension nMHA. Then, with probability at least 1 − ϵ for 0 < ϵ < 1, there exists a choice of binary masks that satisfies

∥BlkS(xi) − BlkT(xi)∥ ≤ ϵ, if the hidden dimensions of the source MHA and FC satisfy

max

i∈[T]

nMHA ≥ d1C log

3 2

32α3Hd

1

ϵ

,

nFC ≥ d1C log

5 2

24αLHd

1

ϵ

,

for some universal constant C > 0. Proof. Our proof strategy is first to apply the SLT existence theorem to the attention mechanism, and then apply the result for FCs. From Theorem 9, with probability at least 1 − 4ϵ, there exists a choice of binary masks so that AttnS(·) satisfies

ϵ 4

. (18) This inequality Equation (18) implies

∥AttnS(xi) − AttnT(xi)∥ ≤

ϵ 4

∥AttnS(xi) − AttnT(xi)∥ ≤

,

ϵ 4

ϵ 4

=⇒ ∥AttnS(xi)∥ ≤

+ ∥AttnT(xi)∥ ≤

+ αH d1. Therefore, the norm of the input vector of the source FC satisfies

∥AttnS(xi)⊤ + x∥ ≤ ∥AttnS(xi)∥ + α

ϵ 4

≤

+ α(H d1 + 1)

≤ 3αH d1. (19) Assume that this upper bound of Equation (19) holds. Now, applying Theorem 10 to the source FC, if

4Ld21 · 6αH√d1 ϵ

nFC ≥ d1C log

5 2

24αLHd

1

= d1C log

,

ϵ

, there exists a choice of binary pruning masks so that FS(·) satisfies ∥FS(AttnS(xi)⊤ + xi)−FT(AttnS(xi)⊤ + xi)∥

with probability at least 1 − 6αHϵ√d

1

3αH√d1ϵ 2 · 6αH√d1 − 1

≤ exp

ϵ 4 − 1.

= exp

Finally, we bound the total error between the source and target transformer blocks: max

∥FS(AttnS(xi)⊤ + xi) + AttnS(xi)⊤ + x − FT(AttnT(xi)⊤ + xi) − AttnT(xi)⊤ − x∥ ≤ max

∥BlkS(xi) − BlkT(xi)∥ = max i∈[T]

i∈[T]

∥FS(AttnS(xi)⊤ + xi) − FT(AttnS(xi)⊤ + xi)∥

i∈[T]

+ ∥FT(AttnS(xi)⊤ + xi) − FT(AttnT(xi)⊤ + xi)∥ + ∥AttnS(xi) − AttnT(xi)∥ ≤ exp

ϵ 4 − 1 + max

2∥AttnS(xi) − AttnT(xi)∥ ≤

i∈[T]

ϵ 2

ϵ 2

+

= ϵ.

From a union bound, the probability that this approximation holds is at least 1 − ϵ:

ϵ 6αH√d1 ≥ 1 − ϵ.

ϵ 4 −

1 −

| |
|---|

#### A.5 SLT Existence Within Transformers Without Normalization Layers

By exploiting the SLT existence theorem for transformer blocks (Theorem 11), we prove the SLT existence theorem for transformers without normalization layers.

We firstly prove the two lemmas used in the proof of the theorem.

- Lemma 12. Let X′ = [x′1,...,x′T]⊤ be a perturbed input matrix, which satisfies maxi∈[T] ∥xi − x′i∥ ≤ ϵmax Then, an arbitrary target MHA AttnT(·) holds the following inequality:

∥AttnT(xi)−AttnT(x′i)∥ ≤ H d1(α (exp(4αϵmax) − 1) + ϵmax). Proof. We begin by analyzing the upper bound of differences for different inputs:

∥x⊤i W(QKj) X⊤ − x′⊤i W(QKj) X′⊤∥max = max k∈[T]

|x⊤i W(QKj) xk − x′⊤i W(QKj) x′k| ≤ max

|x⊤i W(QKj) xk − x′⊤i W(QKj) xk| + |x′⊤i W(QKj) xk − x′⊤i W(QKj) x′k| ≤ max

k∈[T]

∥xi − x′i∥∥W(QKj) ∥∥xk∥ + ∥x′i∥∥W(QKj) ∥∥xk − x′k∥ ≤ αϵmax + αϵmax

k∈[T]

= 2αϵmax.

Applying Lemma 8, the following inequality holds:

∥σ(x⊤i W(QKj) X⊤;ai)XW(VOj) − σ(x′⊤i W(QKj) X′⊤;ai)X′W(VOj) ∥ ≤ ∥σ(x⊤i W(QKj) X⊤;ai)X − σ(x′⊤i W(QKj) X′⊤;ai)X∥ + ∥σ(x′⊤i W(QKj) X′⊤;ai)X − σ(x′⊤i W(QKj) X′⊤;ai)X′∥

= ∥σ(x⊤i W(QKj) X⊤;ai)X − σ(x′⊤i W(QKj) X′⊤;ai)X∥ + ∥σ(x′⊤i W(QKj) X′⊤;ai)(X − X′)∥ ≤ d1α(exp(4αϵmax) − 1) + d1ϵmax.

Then, we have the following bound:

H

H

σ(x⊤i W(QKj) X⊤;ai)XW(VOj) −

∥AttnT(xi) − AttnT(x′)∥ = ∥

j=1

j=1

≤ H d1(α (exp(4αϵmax) − 1) + ϵmax).

σ(x′⊤i W(QKj) X′⊤;ai)X′W(VOj) ∥

| |
|---|

- Lemma 13. An arbitrary target Attention block BlkT(·) holds the following inequality:

∥BlkT(xi)−BlkT(x′i)∥ ≤ H d1(α (exp(4αϵmax) − 1) + 2ϵmax). Proof. From Lemma 12, we have the upper bound as follows:

∥BlkT(xi) − BlkT(x′i)∥ = ∥FT(AttnT(xi)⊤ + xi) + AttnT(xi)⊤ + xi − FT(AttnT(x′i)⊤ + x′i) − AttnT(x′i)⊤ − x′i∥ ≤ ∥FT(AttnT(xi)⊤ + xi) − FT(AttnT(x′i)⊤ + x′i)∥ + ∥AttnT(xi) − AttnT(x′i)∥ + ∥xi − x′i∥ ≤ 2∥AttnT(xi) − AttnT(x′i)∥ + 2∥xi − x′i∥ ≤ 2H d1(α(exp(4αϵmax) − 1) + ϵmax) + 2ϵmax ≤ 2H d1(α(exp(4αϵmax) − 1) + 2ϵmax)

| |
|---|

Theorem 14. Assume B ≥ 2, and let

TfT(xi) := Blk(TB)(Blk(TB−1) ...Blk(1)T (xi)) be a target transformer with B blocks. Let

TfS(xi) := Blk(SB)(Blk(SB−1)...Blk(1)S (xi))

be a pruned random transformer with B layers. Then, with probability at least 1 − ϵ for 0 < ϵ < 1, there exists a choice of binary masks that satisfies

∥TfS(xi) − TfT(xi)∥ ≤ ϵ, if the hidden dimensions of the b-th source MHA and FC satisfy

max

i∈[T]

n(MHAb) ≥ d1C log

n(FCb) ≥ d1C log

2(b,B)d1f3(b,B) ϵ

c1f1(b,B)Hf

2(b,B)dg13(b,B) ϵ

cg21(b,B)LHg

for some universal constant C > 0 and constants c1,c2 > 0 including α. Here, f1,f2,f3 and g1,g2,g3 are quadratic functions of B,b.

Proof. We analyze the approximation errors in each block sequentially and identify the accumulated error in the last block.

Notation for the Proof: Let x(ib) be an input vector to the b-th target block:

xi if b = 1, Blk(Tb−1)(x(ib−1)) if 2 ≤ b ≤ B.

x(ib) =

Then, the final output of the target transformer is TfT(xi) = BlkT(x(iB)). The spectral norm of these input vectors is



∥xi∥ = α

if b = 1,

=: β1

∥Blk(Tb−1)(x(ib−1))∥ ≤ 2(H d1 + 1)∥x(ib−1)∥ ≤ α(2(H d1 + 1))b−1 ≤ α(4H d1)b−1



∥x(ib)∥ =

if 2 ≤ b ≤ B.



=: βb

Similarly, let x′i(b) be an input vector to the b-th source block:

xi if b = 1, Blk(Sb−1)(x′i(b−1)) if 2 ≤ b ≤ B.

x′i(b) =

Then, the final output of the source transformer is TfS(xi) = BlkS(x′i(B)). First Block Error: From Theorem 11, if

 ,

 

1 2B−1 Bj=2 16H√d1βj2 ϵ

3 2

32β13Hd

n(1)MHA ≥ d1C log

 

 ,

1 2B−1 Bj=2 16H√d1βj2 ϵ

5 2

24β1LHd

n(1)FC ≥ d1C log

then with probability at least 1 − 2B−1 B ϵ j=2 16H√d1βj2 , the following inequality holds independently of the input index i: ∥x(2)i − x′i(2)∥ = ∥Blk(1)S (xi) − Blk(1)T (xi)∥

ϵ 2B−1 B

. (20)

≤

j=2 16H√d1βj2

This inequality Equation (20) implies the upper bound of ∥x′i(2)∥:

ϵ 2B−1 B

∥x(2)i − x′i(2)∥ ≤

j=2 16H√d1βj2

ϵ 2B−1 B

=⇒ ∥x′i(2)∥ ≤

+ ∥x(2)i ∥ (From the triangle inequality.)

j=2 16H√d1βj2

ϵ 2B−1 B

≤

+ β2

j=2 16H√d1βj2

≤ 2β2. (21)

Second Block Error: We assume the approximation of the first block is successful (i.e., Equation (21) holds). Then, from Lemma 13, the following bound holds:

∥Blk(2)T (x(2)i ) − Blk(2)T (x′i(2))∥ ≤ H d1 β2 exp

4β2ϵ 2B−1 B

j=2 16H√d1βj2

2ϵ 2B−1 B

− 1 +

j=2 16H√d1βj2

1 4H√d1β2

ϵ 2B−1 B

1 8H√d1β2

ϵ 2B−1 B

− β2 +

= H d1 β2 exp

j=3 16H√d1βj2 ≤ H d1

j=3 16H√d1βj2

1 8H√d1β2

ϵ 2B−1 B

ϵ 2B−1 B

1 2H√d1

(exp(x) ≤ 2x + 1 if 0 ≤ x ≤ 1.)

+

j=3 16H√d1βj2

j=3 16H√d1βj2

- 1

- 2

1 8β2

ϵ 2B−1 B

ϵ 2B−1 B

=

+

j=3 16H√d1βj2 ≤

j=3 16H√d1βj2

ϵ 2B−1 B

.

j=3 16H√d1βj2

From Theorem 11, if

 

 

1 2B−1 Bj=3 16H√d1βj2 ϵ

3 2

32(2β2)3Hd

n(2)MHA ≥ d1C log

 

 ,

1 2B−1 Bj=3 16H√d1βj2 ϵ

5 2

24(2β2)LHd

n(2)FC ≥ d1C log

- then with probability at least 1 − 2B−1 B ϵ

j=3 16H√d1βj2 , the following inequality holds: ∥Blk(2)S (x′i(2)) − Blk(2)T (x′i(2))∥ ≤

ϵ 2B−1 B

.

j=3 16H√d1βj2

Therefore, we have

∥x(3)i − x′3(3)∥ = ∥Blk(2)T (x(2)i ) − Blk(2)S (x′i(2))∥

= ∥Blk(2)T (x(2)i ) − Blk(2)T (x′i(2)) + Blk(2)T (x′i(2)) − Blk(2)S (x′i(2))∥ ≤ ∥Blk(2)T (x(2)i ) − Blk(2)T (x′i(2))∥ + ∥Blk(2)T (x′i(2)) − Blk(2)S (x′i(2))∥ ≤

ϵ 2B−1 B

ϵ 2B−1 B

+

j=3 16H√d1βj2

j=3 16H√d1βj2

ϵ 2B−2 B

. (22)

=

j=3 16H√d1βj2

This inequality Equation (22) implies the upper bound of ∥x′i(3)∥:

ϵ 2B−2 B

∥x(3)i − x′i(3)∥ ≤

j=3 16H√d1βj2

ϵ 2B−2 B

=⇒ ∥x′i(3)∥ ≤

+ ∥x(3)i ∥ (From the triangle inequality.)

j=3 16H√d1βj2

ϵ 2B−2 B

≤

+ β3

j=3 16H√d1βj2

≤ 2β3. (23)

Third Block Error: We assume the approximation of second block is succeessful (i.e., Equation (23) holds). Then, from Lemma 13, the following bound holds:

∥Blk(3)T (x(3)i ) − Blk(3)T (x′i(3))∥ ≤ H d1 β3 exp 4β3

ϵ 2B−2 B

j=3 16H√d1βj2

− 1 + 2

= H d1 β3 exp

1 4H√d1β3

ϵ 2B−2 B

j=4 16H√d1βj2

ϵ 2B−2 B

j=3 16H√d1βj2

1 8H√d1β3

ϵ 2B−2 B

− 1 +

j=4 16H√d1βj2

1 8H√d1β3

1 2H√d1

ϵ 2B−2 B

ϵ 2B−2 B

(exp(x) ≤ 2x + 1 if 0 ≤ x ≤ 1.)

≤ H d1

+

j=4 16H√d1βj2

j=4 16H√d1βj2

1 8β3

- 1

- 2

ϵ 2B−2 B

ϵ 2B−2 B

+

=

j=4 16H√d1βj2 ≤

j=4 16H√d1βj2

ϵ 2B−2 B

.

j=4 16H√d1βj2

From Theorem 11, if

 

 

1 2B−2 Bj=4 16H√d1βj2 ϵ

3 2

32(2β3)3Hd

n(3)MHA ≥ d1C log

 

 ,

1 2B−2 Bj=4 16H√d1βj2 ϵ

5 2

24(2β3)LHd

n(3)FC ≥ d1C log

j=4 16H√d1βj2 , the following inequality holds: ∥Blk(3)S (x′i(3)) − Blk(3)T (x′i(3))∥ ≤

- then with probability at least 1 − 2B−2 B ϵ

ϵ 2B−2 B

.

j=4 16H√d1βj2

Therefore, we have

∥x(4)i − x′3(4)∥

- = ∥Blk(3)T (x(3)i ) − Blk(3)S (x′i(3))∥ (24)
- = ∥Blk(3)T (x(3)i ) − Blk(3)T (x′i(3)) + Blk(3)T (x′i(3)) − Blk(3)S (x′i(3))∥ ≤ ∥Blk(3)T (x(3)i ) − Blk(3)T (x′i(3))∥ + ∥Blk(3)T (x′i(3)) − Blk(3)S (x′i(3))∥

ϵ 2B−2 B

ϵ 2B−2 B

≤

+

j=4 16H√d1βj2

j=4 16H√d1βj2

ϵ 2B−3 B

. (25)

=

j=4 16H√d1βj2

This inequality Equation (25) implies the upper bound of ∥x′i(4)∥:

ϵ 2B−3 B

∥x(4)i − x′i(4)∥ ≤

j=4 16H√d1βj2

ϵ 2B−3 B

=⇒ ∥x′i(4)∥ ≤

+ ∥x(4)i ∥ (From the triangle inequality.)

j=4 16H√d1βj2

ϵ 2B−3 B

≤

+ β4

j=4 16H√d1βj2

≤ 2β4. (26)

(B − 1)-th Block Error: By repeating the same proof procedure as above in each block, we can propagate the error to (B − 1)-th block. We assume all first-to-(B − 2)-th block approximations are successful. Then, from Lemma 13, the following bound holds:

∥Blk(TB−1)(x(iB−1)) − Blk(TB−1)(x′i(B−1))∥ ≤ H d1 βB−1 exp 4βB−1

ϵ 22 Bj=B−1 16H√d1βj2

ϵ 22 Bj=B−1 16H√d1βj2

− 1 + 2

1 8H√d1βB−1

ϵ 22 · 16H√d1βB2 ≤ H d1

1 4H√d1βB−1

ϵ 22 · 16H√d1βB2 − 1 +

= H d1 βB−1 exp

1 2H√d1

ϵ 22 · 16H√d1βB2

1 8H√d1βB−1

ϵ 22 · 16H√d1βB2

(exp(x) ≤ 2x + 1 if 0 ≤ x ≤ 1.)

+

- 1

- 2

ϵ 22 · 16H√d1βB2 ≤

ϵ 22 · 16H√d1βB2

1 8βB−1

=

+

ϵ 22 · 16H√d1βB2

. From Theorem 11, if

1 22 · 16H√d1βB2 ϵ

3 2

32(2βB−1)3Hd

n(MHAB−1) ≥ d1C log

1 22 · 16H√d1βB2 ϵ

5 2

24(2βB−1)LHd

n(FCB−1) ≥ d1C log

,

then with probability at least 1 − 22·16Hϵ√d1βB2 , the following inequality holds independently of the input index i:

ϵ 22 · 16H√d1βj2

∥Blk(SB−1)(x′i(B−1)) − Blk(TB−1)(x′i(B−1))∥ ≤

.

Therefore, we have the following inequality:

∥x(iB) − x′3(B)∥ = ∥Blk(TB−1)(x(iB−1)) − Blk(SB−1)(x′i(B−1))∥

= ∥Blk(TB−1)(x(iB−1)) − Blk(TB−1)(x′i(B−1)) + Blk(TB−1)(x′i(B−1)) − Blk(SB−1)(x′i(B−1))∥ ≤ ∥Blk(TB−1)(x(iB−1)) − Blk(TB−1)(x′i(B−1))∥ + ∥Blk(TB−1)(x′i(B−1)) − Blk(SB−1)(x′i(B−1))∥ ≤

ϵ 22 · 16H√d1βB2

ϵ 22 · 16H√d1βB2

+

ϵ 2 · 16H√d1βB2

. (27)

=

This inequality Equation (27) implies the upper bound of ∥x′i(B)∥:

ϵ 2 · 16H√d1βB2

∥x(iB) − x′i(B)∥ ≤

ϵ 2 · 16H√d1βB2

=⇒ ∥x′i(B)∥ ≤

+ ∥x(iB)∥ (From the triangle inequality.) ≤

ϵ 2 · 16H√d1βB2

+ βB ≤ 2βB. (28)

Final Block Error: We assume all first-to (B − 1)-th block approximations are successful. Then, from Lemma 13, the following bound holds:

∥Blk(TB)(x(iB)) − Blk(TB)(x′i(B))∥ ≤ H d1 βB exp 4βB

ϵ 2 · 16H√d1βB2

ϵ 2 · 16H√d1βB2 − 1 + 2

1 8H√d1βB

1 16H√d1βB

ϵ − 1 +

= H d1 βB exp

ϵ

1 4H√d1

1 16H√d1βB

ϵ (exp(x) ≤ 2x + 1 if 0 ≤ x ≤ 1.)

≤ H d1

ϵ +

1 16βB

1 4

ϵ ≤

ϵ +

=

ϵ 2

. From Theorem 11, if

3 2

32(2βB)3Hd

1 · 2 ϵ

n(MHAB) ≥ d1C log

,

n(FCB) ≥ d1C log

5 2

1 · 2 ϵ

24(2βB)LHd

,

then with probability at least 1 − 2ϵ, the following inequality holds independently of the input index i:

ϵ 2

∥Blk(SB)(x′i(B)) − Blk(TB)(x′i(B))∥ ≤

. Therefore, we finally obtain the following inequality:

∥TfT(xi) − TfS(xi)∥ = ∥Blk(TB)(x(iB)) − Blk(SB)(x′i(B))∥

= ∥Blk(TB)(x(iB)) − Blk(TB)(x′i(B)) + Blk(TB)(x′i(B)) − Blk(SB)(x′i(B))∥ ≤ ∥Blk(TB)(x(iB)) − Blk(TB)(x′i(B))∥ + ∥Blk(TB)(x′i(B)) − Blk(SB)(x′i(B))∥ ≤

ϵ 2

ϵ 2

+

= ϵ.

Success Probability of the Approximation: By the union bound, the probability that ∥TfT(xi) − TfS(xi)∥ ≤ ϵ holds is at least 1 − ϵ:

ϵ 2B−1 B

ϵ 2B−1 B

ϵ 2B−2 B

ϵ 22 · 16H√d1βj2 −

ϵ 2

1 −

−

−

− ··· −

j=2 16H√d1βj2

j=3 16H√d1βj2

j=4 16H√d1βj2

2 + Bk=2 kj=2 32H√d1βj2 2 Bj=2 32H√d1βj2

= 1 −

ϵ

1 + Bk=2 kj=2 32H√d1βj2 2 Bj=2 32H√d1βj2

1 2 Bj=2 32H√d1βj2

= 1 −

ϵ −

ϵ

1 + Bk=2 kj=2 32H√d1βj2 2 Bj=2 32H√d1βj2

1 64

ϵ −

≥ 1 −

ϵ

1 + Bk=2(32Hα2√d1)k−1 kj=2(16H2d1)j−1 2(32Hα2√d1)B−1 k

1 64

= 1 −

ϵ −

ϵ

j=2(16H2d1)j−1

- 1 + Bk=2(32Hα2√d1(16H2d1)12k)k−1

- 2(32Hα2√d1)B−1(16H2d1)12B(B−1) ϵ ≥ 1 −

1 64

ϵ −

= 1 −

B k=1(32Hα2√d1(16H2d1)12B)k−1

1 64

ϵ −

2(32Hα2√d1)B−1(16H2d1)12B(B−1) ϵ ≥ 1 −

(32Hα2√d1(16H2d1)12B)B − 1 (32Hα2√d1(16H2d1)12B − 1) ·

1 64

1

ϵ −

2(32Hα2√d1)B−1(16H2d1)12B(B−1) ϵ ≥ 1 −

(32Hα2√d1(16H2d1)12B)B (32Hα2√d1(16H2d1)12B − 1) ·

1 2(32Hα2√d1)B−1(16H2d1)12B(B−1) ϵ

1 64

ϵ −

1 64

1 2 1 − 1

= 1 −

ϵ −

ϵ

32Hα2√d1(16H2d1) 12

B

1 64

1 2 1 − 321

≥ 1 −

ϵ −

ϵ

1 64

16 32

= 1 −

ϵ −

ϵ

1055 1984

= 1 −

ϵ ≥ 1 − ϵ.

| |
|---|

### B Experimental Details

This section describes the detailed experimental settings. All experiments can be verified with four NVIDIA H100 SXM5 94GB GPUs.

- B.1 Synthetic Data Experiment We construct a synthetic dataset for angular velocity estimation, where each input sequence consists of T two-dimensional vectors x1,...,xT such that xt = (cos(ωt + θ0),sin(ωt + θ0)) for some angular velocity ω ∈ [−π,π] and initial phase θ0 ∈ [0,π]. The task is to estimate ω given the full sequence. Each sequence includes a special regression token—similar to the CLS token in BERT (Devlin et al. 2019)—at the beginning, and the model is trained to predict angular velocity by the regression token initialized to zero. We generate 10,000 samples each for training, validation, and test sets, and input sequence lengths vary from 4 to 256 during training. We experiment with MHAs and transformers. In the MHA experiment, both the source and target MHAs are configured as single-head attention modules, with input and output dimensions of 2 and 1, respectively. The networks are trained using the AdamW optimizer (Loshchilov and Hutter 2019) with a batch size of 1024 and a learning rate of 0.1. Each target MHA is trained for 25 epochs with weight decay set to 0.01. In the transformer experiment, both the source and target models follow the construction described in Section 3.3. Each MHA has a single attention head, and both its input and output dimensions are set to 2. The same regression token is used for both the source and target models to ensure that the approximation quality reflects differences in the behavior of the models rather than token-level discrepancies. The query and key dimensions of the target models are set to 8. Target networks are initialized according to the assumptions of our

theoretical results. Specifically, entries of the query and key projection matrices are drawn i.i.d. from U[−n1K/4,n1K/4], and those of the value and output projection matrices from U[−1,1]. The weights in fully-connected networks are also initialized with

U[−1,1]. Source networks are initialized with Xavier uniform distribution (Glorot and Bengio 2010). To identify SLTs, we use the weight approximation method in Lemma 4, based on the subset-sum approximation technique of Pensia et al. (2020). For each target network, we generate 100 source networks with random initialization and solve the associated subset-sum problem using Gurobi’s mixed-integer programming solver (Gurobi Optimization, LLC 2024). In the experiments varying the hidden dimension, the input length is fixed at 4. We report the mean and standard deviation of the approximation error over these 100 candidates. We also fit exponential decay curves to the approximation error using SciPy (Virtanen et al. 2020).

- B.2 Language Modeling Experiment

We further evaluate our theoretical framework in a practical language modeling setting. Here, we search for SLTs using the edge-popup algorithm (Ramanujan et al. 2020), which searches for accurate subnetworks by assigning scores to each connection and retaining only the top-k% entries during training. We set this k as 30. We train models from the GPT-2 family (Radford et al. 2019; Wolf et al. 2020) on the WikiText-103 dataset (Merity et al. 2017), using a maximum sequence length of 1024. The weights of these models are initialized based on the GPT-2 initialization scheme: they are drawn i.i.d. from a normal distribution with mean 0 and standard deviation 0.02. For the output projection in MHAs and the second layer of the fully-connected ReLU networks, the standard deviation is further scaled by (2b)−1/2, where b is the number of transformer blocks. We train the models for 50 epochs, with 227 steps per epoch. The AdamW optimizer is used with an initial learning rate of 0.0001, which is decayed to 0.00001 via a cosine annealing scheduler (Loshchilov and Hutter 2017). A linear learning rate warm-up is applied during the first epoch. For each model size, we repeat training with three different random seeds and report the mean and standard deviation of the best performance.

