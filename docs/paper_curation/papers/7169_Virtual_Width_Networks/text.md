# arXiv:2511.11238v2[cs.LG]17Nov2025

[Figure 1]

## Virtual Width Networks

1ByteDance Seed See Contributions section for full author list.

### Abstract

We introduce Virtual Width Networks (VWN), a framework that delivers the benefits of wider representations without incurring the quadratic cost of increasing the hidden size. VWN decouples representational width from backbone width, expanding the embedding space while keeping backbone compute nearly constant. In our large-scale experiment, an 8× expansion accelerates optimization by over 2× for next-token and 3× for next-2-token prediction. The advantage amplifies over training as both the loss gap grows and convergence-speedup ratio increase, showing that VWN is not only token-efficient but also increasingly effective with scale. Moreover, we identify an approximately log-linear scaling relation between virtual width and loss reduction, offering an initial empirical basis and motivation for exploring virtual-width scaling as a new dimension of large-model efficiency.

Date: November 18, 2025 Correspondence: Defa Zhu at zhudefa@bytedance.com

Downstream Benchmarks

Training Loss on Next2 Token Prediction

Training Loss on Next Token Prediction

1.675

MoE-A3.3B

MoE-A3.3B

###### +2.16

64

MoE-A3.3B-VWNx8

1.80

MoE-A3.3B-VWNx8

1.650

62

1.625

1.75

Accuracy(%)

60

Next2Loss

Next2Loss

1.600

58

1.575

1.70

56

###### 2.1×

1.550

###### 1.6×

###### 2.8×

54

1.65

###### 3.5×

###### 2.0×

1.525

###### 2.5×

52

=0.025

=0.049

1.500

MoE-A3.3B

1.60

50

=0.027

=0.051

MoE-A3.3B-VWNx8

=0.032

1.475

=0.056

0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2

0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2

0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7 3.0

Tokens (Trillions)

Tokens (Trillions)

Tokens (Trillions)

Figure 1 Results from large-scale experiments on a 3.3B-activation MoE using Virtual Width Networks (VWN). We compare the baseline MoE-A3.3B against MoE-A3.3B-VWNx8, configured with a virtual width factor of r=8. Left and middle: training loss for next-token and next-two-token prediction versus seen tokens. VWN reaches the same loss as the baseline using 2.5× and 3.5× fewer tokens, respectively. Right: average accuracy on a collection of open-source benchmarks (see Table 3), where scores are aggregated using internally defined task weights. A difference of one point corresponds to a notable performance gap under this weighting scheme.

### 1 Introduction

According to scaling laws [15, 20], expanding either model parameters or the size of the training corpus yields more capable models. In particular, increasing the model width (hidden dimensions) enables the

Network Width

Network Width

| | |
|---|---|

| | | |
|---|---|---|

Virtual Width

|transformer layer|
|---|

x N

|width connections<br><br>| | |x| |
|---|---|---|---|
| | | | |
<br><br>N<br><br>depth connections<br><br>|transformer layer (1x width)|
|---|
<br><br>VWN layer|
|---|

Embedding

(a) Transformer

1.5x Network Width

| | | |
|---|---|---|

|transformer layer (1.5x width)|
|---|

x N

| | | |
|---|---|---|

| | |
|---|---|
|Embedding (1.5x width)| |

| | |
|---|---|
|Embedding (1.5x width)| |

(b) 1.5x Width Transformer

(c) 1.5x Virtual Width Network

- Figure 2 Standard Transformer vs. Virtual Width Network (VWN). (a) A standard Transformer uses the same width for embeddings and backbone. (b) Naive width scaling expands both proportionally, causing quadratic growth in parameters and compute. (c) VWN decouples embedding width from backbone width. With Generalized Hyper-Connections, over-width embeddings (e.g., 1.5×) are coupled to a standard-width backbone, increasing representational capacity with minimal compute overhead.

representation of richer, more complex functions by packing additional information into each vector, which in turn substantially boosts performance. However, naively increasing hidden dimensions leads to quadratic growth in parameters and compute, posing challenges in resource-constrained settings.

To address the challenge and enhance the scalability of modern Transformers, researchers have developed conditional computation strategies that expand model capacity without proportionally increasing computational costs. A prominent example is mixture-of-experts (MoE) architectures [9, 22, 27], which dynamically activate specialized subnetworks per input token. By selectively engaging only a fraction of the total parameters for each token during computation, MoE models significantly improve throughput and enable efficient scaling to very large model sizes, without proportionally increasing per-token computational cost.

However, conventional MoE architectures can be viewed as expanding only the inner dimension of the feed-forward networks (FFN), while the backbone hidden dimension remains fixed. Consequently, the model’s representational capacity is still bottlenecked by the hidden dimension, resulting in a persistent performance gap compared to models with truly wider hidden layers. While directly increasing the hidden dimension can close this gap, it incurs a quadratic increase in parameters and computation. This prompts us to ask: Can we harness the benefits of wider representations while avoiding the quadratic cost explosion of naive scaling?

In this work, we address this challenge by proposing Virtual Width Networks (VWN), a general framework that enables scaling token-embedding width while keeping the hidden dimensions of the Transformer backbone fixed. Our key insight is that wider representations can be achieved by expanding the embeddings rather than by widening the hidden layers, the latter being the main source of quadratic computational cost. In this view, models employing methods such as Hyper-Connections [36] or AltUp [2] can be regarded as simplified instances within the broader VWN family. By enhancing the design of VWN, we further improve its representational capacity and uncover a favorable scaling property of virtual width—specifically, a scaling relation between the

loss and the virtual width factor under a fixed backbone—which offers the community a new dimension for scaling large models.

Having established the conceptual motivation and benefits of VWN, we now describe its internal mechanism. The input to Virtual Width Networks (VWN) is a widened token embedding, which we refer to as the Over-Width Embedding. Within VWN, the intermediate representations are correspondingly referred to as Over-Width Hidden States. To process these states, we replace the standard residual connections with Generalized Hyper-Connections (GHC)—a more general formulation that unifies the ideas of Hyper-Connections (HC) [36] and Frac-Connections (FC) [37]. GHC introduces a flexible mechanism that, with lightweight computation, compresses the Over-Width Hidden States to the backbone width before feeding them into the attention or feed-forward modules, and then expands the module outputs back to the Over-Width width to update the Over-Width Hidden States for the next layer. Finally, a reduce operator, such as a linear projection, maps the last Over-Width Hidden States back to the original hidden width before the unembedding layer to produce the output logits.

To better exploit the widened representations, we pair VWN with multi-token prediction (MTP), optimizing both the standard next-token objective and an auxiliary n-gram loss. Intuitively, the denser MTP supervision exercises the expanded virtual space, while the additional representational degrees of freedom from VWN improve short-range compositional modeling, yielding a synergistic effect.

We evaluate VWN across multiple regimes using internal MoE models. We report training dynamics and token efficiency relative to matched non-VWN baselines, and assess downstream generalization. Headline results show that VWN, which expands the embedding width by 8×, achieves the baseline’s next-token loss with

- 2.5× fewer tokens and the next-2-token loss with 3.5× fewer tokens, with the efficiency advantage increasing as training progresses, as shown in Figure 1.

Contributions. Our main contributions are summarized as follows:

- • Virtual Width Networks (VWN). We introduce VWN, which decouples embedding width from backbone width and enables r× virtual widening with minimal additional compute through Generalized Hyper-Connections (GHC). Through systematic scaling experiments, we further uncover a log-linear scaling law between the virtual width factor r and loss, shedding light on how virtual widening influences model performance.
- • Generalized Hyper-Connections (GHC). We formalize GHC as a unifying formulation that subsumes prior variants (e.g., Hyper- and Frac-Connections) and provides flexible routing and mixing between virtual and backbone hidden states.
- • Synergy with Multi-Token Prediction (MTP). We demonstrate that VWN synergizes with MTP, yielding consistent improvements in downstream accuracy.

### 2 Related Works

Scaling Model Capacity. Transformer models have demonstrated strong performance gains through increased model width, depth, and data scale [15, 20]. However, increasing hidden dimensionality often leads to quadratic growth in parameters and compute, posing challenges in resource-constrained settings. Several approaches have been proposed to decouple model capacity from computation. For instance, mixture-of-experts (MoE) models [9, 22, 27] conditionally activate subnetworks to scale model size efficiently. Our method increases effective capacity while avoiding the quadratic computational cost typically associated with widening hidden dimensions. This is achieved by decoupling the embedding width from the backbone hidden size.

Hyper- and Frac-Connections. Hyper-Connections (HC)[36] and AltUp[2] enhance model expressiveness by expanding the hidden dimension through low-cost compositional links across layers. However, in large hidden spaces, HC often under-utilizes the expanded representations, since each extension is updated using only a few scalar weights, making it difficult to fully exploit the additional capacity. Frac-Connections (FC) [37] take the opposite approach: instead of enlarging the hidden size, they partition the existing hidden dimension into multiple smaller segments, thereby realizing HC-like connectivity without increasing model width. Our

|(a) Transformer<br><br>|: Layers : Matrix : Hidden Vectors : Virtual Vectors : Embedding/LM-head 𝑡 : tokens<br><br>|
|---|
<br><br>Split<br><br>+ +<br><br>Cat<br><br>+<br><br>𝐴 ×(   )<br><br>𝐵 × <br><br>ℎ<br><br>(b) Virtual Width Network (VWN)<br><br>Attention<br><br>+<br><br>FFN<br><br>+ +<br><br>Cat<br><br>+<br><br>𝐴 ×(   )<br><br>𝐵 × <br><br>Attention<br><br>ℎ<br><br>FFN<br><br>+<br><br>ℎ<br><br>ℎ<br><br>| | |
|---|---|
| | |
<br><br>(c) VWN with MTP<br><br>Next token prediction Next2 token prediction<br><br>𝑡 𝑡 𝑡 𝑡<br><br>𝑡 𝑡 𝑡 𝑡<br><br>|LM-head|
|---|
<br><br>Input Tokens<br><br>Target Tokens<br><br>|LM-head|
|---|
<br><br>𝑡 𝑡 𝑡 𝑡<br><br>𝑡 𝑡 𝑡 𝑡<br><br>ℎ<br><br>|𝐿 VWN Layers ( × width)<br><br>|
|---|
<br><br>|E-Norm<br><br>Cat<br><br>Cat<br><br>Block-level Linear<br><br>𝐴 ×  <br><br>𝐵 × <br><br>H-Norm<br><br>+ + +| |
|---|---|
| | |
<br><br>Multi-Token Prediction<br><br>|1 VWN Layer|
|---|
<br><br>Group Norm<br><br>Reduce (Linear)|
|---|

- Figure 3 Overview of Virtual Width Networks (VWN). (a) The standard Transformer maintains a consistent width across input embeddings, intermediate hidden vectors at each layer, and final layer outputs. (b) VWN scales the embedding dimension through over-width embeddings while maintaining the layer dimension using lightweight Generalized Hyper-Connections (GHC). These dimensions interact flexibly through small matrices Al and Bl (l stands for the layer number). (c) We enable multiple token supervision (multi-token prediction), allowing for richer token representations.

proposed Generalized Hyper-Connections (GHC) integrate the advantages of both—expanding the hidden dimension while further subdividing it into structured sub-states. This design offers fine-grained control over capacity usage and enables more efficient utilization of the expanded representational space. Moreover, it introduces additional flexibility: the expansion ratio of the hidden dimension need not be an integer multiple, and such fractional expansions have been empirically validated as effective (see Sec. 5.1).

Embedding Expansion. Recent studies have highlighted the importance of vocabulary scaling in large language models [29], showing that expanding the input vocabulary—particularly through hierarchical n-gram token embeddings—can systematically improve model expressiveness and training efficiency with negligible computational overhead [17]. The Over-Tokenized Transformer framework [17] introduced Over-Encoding (OE) to scale input representations using multi-gram tokenization and Over-Decoding (OD) to enhance output supervision via multi-token prediction objectives. Notably, Multi-Token Prediction (MTP) [10] is regarded as an effective instantiation of OD for practical training.

### 3 Method

#### 3.1 Rethinking the Model Width

In a standard Transformer model with L layers and model width D, the initial token representation h0 ∈ RD is obtained through embedding lookup. This representation is subsequently processed through transformer layers, each composed of an attention block and a FeedForward Network (FFN) block. Specifically, at the l-th layer, an intermediate hidden vector hl ∈ RD is computed from hl−1. The final layer outputs the token representation hL ∈ RD, which is then transformed via a linear head to project it into a |V|-dimensional

vocabulary space. The computational complexity for transfomer is O(D2), indicating that scaling the model width D results in a quadratic increase in computational cost.

However, the embedding lookup operation represent only a minor fraction of the overall computational cost. Leveraging this insight, we decouple the embedding dimension from the hidden layer dimension so that the embedding dimension to be significantly expanded while maintaining the original hidden dimension for intermediate layer computations. Consequently, this approach preserves nearly the original computational cost while significantly enhancing the representational capacity of token embeddings.

#### 3.2 Over-Width Embedding

To increase the embedding dimensions, we propose the Over-Width Embedding technique. Given a fixed hidden size D, we enlarge the embedding dimension of input to a wider dimension D′, resulting in richer token embeddings without a substantial increase in computational overhead.

Formally, let hl ∈ RD represent the hidden state at l-th layer. We partition this hidden vector evenly into m disjoint segments:

hl = hl⊺1 hl⊺2 ... hl⊺m ⊺ , where hlk ∈ RD/m,k = 1,2,...,m. (1)

′

Next, we define an expanded embedding vector e ∈ RD

, where D′ = mn D, with integers n > m:

n m

e = e⊺1 e⊺2 ... e⊺n ⊺ , where ek ∈ RD

′/n with D′ =

D. (2)

Finally, at the input layer, we set h′0 = e, thereby utilizing wider token embeddings. When the expansion ratio mn is large, a single linear projection can optionally be used to map the original 1× embedding to the wider dimension:

Ewide = Wexpand Ebase, (3)

which is like to applying a low-rank decomposition to a very wide embedding table. In addition, one can adopt input-augmentation strategies [17] that inject more information per input than a single isolated token embedding to further enrich the widened representation.

For unembedding, the model needs to map the last over-width hidden states back to the original hidden width D before the unembedding layer. We introduce a reduce operator Wreduce ∈ RD×D

′

that performs a linear projection from the over-width dimension D′ to the original width D:

hLreduce = Wreduce h′L. (4) To stabilize training, normalization is applied before the reduce operator, as shown in Figure 3 (c). When the expansion ratio r = mn is large, the over-width dimension D′ may become very large (e.g., an 8× expansion of a 4096-dimensional hidden size yields a 32K-dimensional representation). Instead of directly normalizing across all D′ dimensions, we adopt Group Normalization [31], where the group size equals the original hidden size D.

#### 3.3 Generalized Hyper-Connections

We propose Generalized Hyper-Connections (GHC), a novel method to effectively leverage wider token embeddings while maintaining the original hidden dimension during intermediate layer computations. Specifically, at each layer l, GHC introduces a light transformation matrix GHCl that encodes weighted relationships between segments of the original hidden representations and the expanded token embeddings. Formally, this matrix is

defined as follows:

GHCl =

0 Bl Al

=

###### 0 Bl

◦

Al Aˆ l





0 ··· 0 β1l,1 ··· β1l,n

... . .

... .

.

0 ··· 0 βm,l 1 ··· βm,nl α1l,1 ··· α1l,m α1l,m+1 ··· α1l,m+n

=

... . αn,l 1 ··· αn,ml αn,ml +1 ··· αn,ml +n

... . .

 

 

.

∈ R2m×(m+n). (5)

Consider the l-th network layer T l, it integrates self-attention layers or feed-forward networks within transformers. The output of the GHC, denoted as H′l = Reshape(h′l,(n,D′/n)), represents the Over-Width Hidden States, and can be formulated as:

H′l = GHCl(T l,H′l−1)

◦

= Bl⊺T l

Al⊺H′l−1 + Aˆ l⊺H′l−1. (6)

Dynamic Generalized Hyper-Connections (DGHC). To further enhance adaptability in the forward process, we introduce a dynamic extension of the GHC method, termed Dynamic GHC (DGHC), where the transformation matrices are adaptively conditioned on input representations H′:

GHC(H′) =

0m×m B(H′)

◦ A(H′) Aˆ(H′)

. (7)

In practice, we adopt the hybrid strategy from Zhu et al. [36, 37], which integrates both static and dynamic parameters, while making slight adjustments to better fit our VWN framework. The dynamic parameters are generated through a lightweight linear projection network. To ensure training stability, input features are initially normalized. Subsequently, a linear transformation coupled with a tanh activation function is applied. The output is then scaled by a small, learnable matrix and combined with the corresponding static matrix:

H′ = norm(H′), (8)

⊤

H′Wβ τ

B(H′) = Sβ ◦ tanh

+ B, (9)

A(H′) = Sα ◦ tanh

H′Wα τ

+ A. (10)

where τ = D/m, Sβ ∈ Rm×n and Sα ∈ Rn×(m+n) are learnable scaling matrices initialized to 1 (same shapes as B and A, respectively). Let db := D′/n = D/m denote the per-block width and view H′ as an n × db matrix. The projection weights Wβ ∈ Rd

b×(m+n) are learnable parameters that generate the dynamic coefficients. With these shapes, H′Wβ ∈ Rn×m and H′Wα ∈ Rn×(m+n); the transpose in Eq. (9) makes the former m × n to match B ∈ Rm×n, while Eq. (10) already aligns with A ∈ Rn×(m+n).

b×m and Wα ∈ Rd

Initialization and Implementation. The dynamic parameters Wβ and Wα in Eqs. (9) and (10) are initialized to 0, while the static matrices are initialized as follows. It is worth noting that we do not perform any dedicated tuning of the initialization, and thus there remains room for improving learning efficiency.

The static matrix B ∈ Rm×n is initialized with a cyclic pattern:

1, if i = j mod m, 0, otherwise,

for i ∈ {0,...,m − 1}, j ∈ {0,...,n − 1}. (11)

B[i,j] =

The static matrix A ∈ Rn×n is initialized as a block matrix:

 

Im×m Im×m 0m×r , if n = m,

where r = n − m. (12)

A =

Im×m Im×m 0m×r 0r×m 0r×m Ir×r

, if n > m,



The static components B and A do not utilize weight decay, whereas the dynamic component does. The implementation details can be found in Appendix B, while the algorithm is presented in Algorithm 1.

Algorithm 1 Virtual Width Networks (VWN) Forward Pass Require: Over-width token embedding e ∈ RD

′

Require: Fraction rate m, Expanded width n, Backbone dimension D Require: Network layers {T 1,...,T L} and routing matrices {Al,Bl}Ll=1 Require: Compression matrix R ∈ Rn×m Ensure: Final output y

- 1: Initialize:
- 2: H′0 ← Reshape(e,(n,D′/n))⊺ ∈ RD

′/n×n

- 3: for l = 1 to L do
- 4: Xl ←

◦ Al

⊺

H′l−1

- 5: zl ← T l(Reshape(Xl,(D,))) ▷ Input to FFN or Attention layer in Transformer block
- 6: Zl ← Reshape(zl,(m,D/m))⊺
- 7: H′l ← Bl⊺Zl + Aˆ l⊺H′l−1
- 8: end for
- 9: hL ← Linear(GroupNorm(H′L))
- 10: y ← Unembedding(Norm(hL))
- 11: return y

#### 3.4 Multi-token Prediction

As for the output layer, previous research [17] has demonstrated that Multi-Token Prediction (MTP) serves as an approximation of k-gram decoding. Building upon this insight, we leverage MTP to provide fine-grained supervised signals by introducing additional VWN layers atop the backbone model, thus constructing an enhanced prediction head. Specifically, following DeepSeek-AI et al. [7], we concatenate the embedding of the next token with the last-layer embedding of the preceding token, applying a linear projection to generate logits, as illustrated in the upper part of part of Figure 3 (c).

However, adopting a single dense linear that mixes hidden states and embeddings as in DeepSeek-AI et al. [7] (i.e., a 2D→D projection) becomes prohibitively expensive under VWN, where the width is expanded by a factor of r. A naive dense mixing would scale to 2rD→rD; for r=8, both parameters and FLOPs grow substantially and are difficult to afford. To address this, we perform mixing with a block-level linear. We partition the rD-dimensional vectors into n = r × m segments of size D/m, and apply the same small linear per segment with shape (2D/m)→(D/m). In other words, we fuse the hidden-state and embedding features locally within each segment, sharing the linear projector across all blocks. This preserves the benefits of wider VWN representations while keeping the mixing cost comparable to the r=1 case.

##### 3.5 Cost Analysis Computational Cost. The theoretical computational overhead of VWN is relatively low. We focus on the dominant computational costs. The normalization operation (e.g. RMSNorm) requires 4mn D FLOPs, per token. Calculating the dynamic parameter A and B requires 2(2mm+n)nD FLOPs per token. The width connection incurs a cost of 2(m+mn)nD FLOPs, and the depth connection requires 2nD FLOPs. With modest

settings of m = 2 and n = 3, the normalization, dynamic parameter calculation, and width connection steps amount to 42D FLOPs, while the depth connection requires 6D FLOPs. These computational costs are minor for GPU-based training/inference systems, especially considering the I/O overhead associated with activation memory access, which becomes a bottleneck for VWN. To minimize I/O, the normalization, dynamic parameter calculation, and width connection operations are fused into a single GPU kernel. Furthermore, the width connection can be fused with the subsequent layer normalization in the transformer layer. When m is small, VWN adds roughly mn − 1 times the cost of layer normalization and residual addition due to the over-width hidden states. This overhead is negligible in such settings, though for larger m its effect varies with the configuration.

Memory Cost. During model training, intermediate activations must be stored for backpropagation. VWN introduces additional memory overhead for saving the VWN input activations. However, this can be mitigated through inexpensive recomputation. In a typical training framework like Megatron-LM, each token in a vanilla transformer layer requires 34D bytes for activation storage, employing selective activation recomputation [21]. VWN primarily adds the cost of saving the inputs for the A and B, requiring 2 × 2 × (mn + 1)D bytes, given that each number is represented using 2 bytes (16-bit float) and there are two width and depth connections per transformer layer. While attention and FFN inputs are typically saved for weight gradient computation, they can be efficiently recomputed from the width connection. By saving the input to the A in the width connection and the input to the B in the depth connection, the subsequent width connection input can be recomputed at a low cost. Using a factor η to represent the ratio of width connection inputs that are saved, the extra activation memory consumption of VWN for a transformer layer is 4ηmn D bytes. With modest settings of m = 2, n = 3, and η = 0.5 (saving the width connection input for attention and recomputing it for FFN), the added memory consumption is 3D bytes, which is approximately 8.8% of the memory footprint of the vanilla transformer layer. During model inference, the additional memory overhead arises solely from the extra parameters, a negligible amount compared to other memory consumption.

### 4 A Connectivity Perspective

We reinterpret Virtual Width Networks (VWN) through the lens of connectivity as attention along the depth axis. Consider the stack of layers as a “depth sequence,” where each layer index is like a token position and hidden states act as a “vertical KV cache”. Under this view, common connectivity patterns map to attention-like windows over prior layers: (1) a plain feed-forward stack without residuals corresponds to a sliding window of size 1 (each layer processes only its current input and forgets the previous one); (2) residual connections [11] implement a window of size 2 (current input plus the immediately preceding one); and (3) dense connectivity [16, 23, 32] extends the window size to include all previous layers, allowing each layer to reuse all prior representations. VWN with Generalized Hyper-Connections (GHC) sits in between: it realizes a learned, fixed-cost, linear-attention-like mechanism over depth that scales the accessible depth context.

Formally, let the widened state at layer l be a slot matrix H′l ∈ R(D/m)×n with n slots of size D/m, and let r := n/m be the width expansion measured in D-units. The GHC recurrence with the backbone mapping made

◦

Al⊺H′l−1 + Aˆ l⊺H′l−1, where A ˆ l ⊺ transports/attenuates information stored in the slots (a learned carry/forget operator), and Bl ⊺ writes the current layer’s backbone summary into selected slots. Unrolling Eq. (6) explicitly yields

explicit is in Eq. (6): H′l = Bl⊺T l

l−1

H′l =

t=0

t−1

i=0

A ˆ l−i ⊺ Bl−t ⊺T l−t

l−t

◦

A

⊺,H′(l−t−1) +

l−1

i=0

A ˆ l−i ⊺ H′0 (13)

with the convention that an empty product equals the identity. Equation (13) shows that H′l linearly aggregates backbone-transformed features from earlier layers, propagated by the “carry operator” Aˆ and written via B at each step—capturing the spirit of linear attention over a compressed depth cache.

Choosing m. The memory budget for storing depth information—measured in D-units—is r=n/m. GHC allocates this budget between per-layer fidelity and the number of layers remembered:

- • With m=1, the model stores up to r layers at full D-dimensional fidelity (fewer layers, higher bandwidth per layer).
- • With m>1, the model stores up to n=rm layers, each compressed to D/m dimensions (more layers, lower bandwidth per layer).

Thus, m controls per-layer compression, n controls the nominal depth window, and r fixes the total memory budget. The learned, input-dependent routing then provides a soft extension beyond the nominal window via attenuation rather than hard truncation. Intuitively, a larger m expands the effective number of remembered layers at the cost of lower per-layer fidelity. For wider models, the increased representational capacity offers sufficient bandwidth to accommodate a larger m. Similarly, deeper networks benefit from larger m since enabling each layer to access longer-range, shallower-layer information can alleviate optimization difficulty and improve gradient flow.

Hard vs. soft depth windows.

- • Hard routing. If Aˆ l and Bl are near-permutation/binary gates, the update behaves like a fixed-size sliding window over depth. With m=1, there are r=n slots of dimension D, so the model can retain the last r layers in full fidelity. With m>1, there are n=rm slots of size D/m; each layer’s D-dimensional state is compressed to D/m and written to one slot, giving a hard window of size n in compressed form.
- • Soft routing. With real-valued, potentially input-dependent Aˆ l and Bl (Dynamic GHC), information is partially retained and mixed across steps. When the spectral radius of A ˆ l ⊺ is below 1, Eq. (13) implies exponentially decayed contributions from preceding layers. The effective depth receptive field can exceed the nominal hard window (> r for m=1 or > n for m>1), albeit with progressively attenuated and mixed information.

A concrete configuration. Consider (m,n)=(8,64), so r=8. The model maintains n=64 slots of width D/8. Under hard routing, the current layer can leverage the most recent 64 layers, each represented at 1/8 of the original dimensionality. Under soft routing, contributions from layers earlier than 64 may persist with decay, effectively enlarging the “depth receptive field”.

On the scope of the attention analogy. Our analogy to attention chiefly borrows the KV-cache perspective along depth. It does not imply that inter-layer connections are built via similarity scores or pairwise correlations as in standard self-attention. GHC uses learned (static or input-conditioned) routing matrices to carry, compress, and write information across layers at fixed cost, rather than computing dot-product scores or softmax over layer indices.

### 5 Experiments

#### 5.1 VWN 1.5×

To examine the effectiveness of VWN under fractional virtual widening, we use the 1.5× configuration as a representative case. We jointly evaluate VWN and Multi-Token Prediction (MTP) in large-scale language-model pre-training, and measure downstream performance on Collection A, defined as the average score across the benchmarks listed in Table 2. In the 1.5× setting, group normalization preceding the reduce operator (used to aggregate virtual partitions) is omitted.

For our primary evaluation, we conduct comprehensive experiments on internal Mixture-of-Experts (MoE [27]) models of multiple scales, including MoE 0.4B/4B and MoE 2.5B/30B, all trained on large-scale internal datasets. Each VWN variant adopts the (m,n) = (2,3) configuration to realize a 1.5× virtual widening relative to the backbone hidden size, thereby decoupling the expanded embedding space from the fixed-width backbone at nearly constant compute. This setup enables controlled assessment of VWN and MTP generality across diverse model sizes and realistic production conditions.

0.4B/4B Models. We study the effects of VWN and MTP on 0.4B/4B MoE models (Figure 4). On the training objective (left), VWN consistently lowers the next-token prediction (NTP) loss relative to the baseline,

training loss

Collection A

50

2.10

MoE-0.4B/4B

MoE-0.4B/4B-VWN

48

2.08

MoE-0.4B/4B-MTP

46

MoE-0.4B/4B-VWN-MTP

2.06

Accuracy(%)

44

2.04

Loss

42

2.02

2.00

40

MoE-0.4B/4B

1.98

38

MoE-0.4B/4B-VWN

1.96

MoE-0.4B/4B-MTP

x1.5 gap=0.016

36

MoE-0.4B/4B-VWN-MTP

1.94

34

120 180 240 300 360 420 480 540 600

120 180 240 300 360 420 480 540 600

Tokens (Billions)

Tokens (Billions)

- Figure 4 Performance of VWN and MTP on 0.4B/4B MoE models. Left: Training loss versus seen tokens (billions). VWN lowers the next-token prediction loss, whereas MTP slightly hurts the NTP loss; combining VWN and MTP (VWN+MTP) yields the lowest final loss among the augmented variants but still shows a small gap ( 0.016) relative to the baseline metric when MTP is included. Right: Average downstream accuracy (%) versus tokens. Both VWN and MTP improve downstream accuracy over the baseline, and their combination delivers the largest gains throughout training. Models: MoE-0.4B/4B (baseline), MoE-0.4B/4B-VWN, MoE-0.4B/4B-MTP, and MoE-0.4B/4B-VWN-MTP.

300 450 600 750 900 1050 1200

Tokens (Billions)

1.74

1.76

1.78

1.80

1.82

1.84

1.86

- 1.88

Loss

training_loss

MoE-2.5B/25B

MoE-2.5B/25B-VWN

MoE-2.5B/25B-VWN-MTP

gap=0.015

x1.4

300 450 600 750 900 1050 1200

Tokens (Billions)

50

52

54

56

58

60

62

Accuracy(%)

Collection A

MoE-2.5B/25B

MoE-2.5B/25B-VWN

MoE-2.5B/25B-VWN-MTP

Figure 5 Performance of VWN and MTP on 2.5B/25B MoE models. Left: Training loss versus seen tokens (billions). VWN reduces the next-token prediction loss relative to the baseline, and adding MTP on top of VWN does not hurt the loss at this scale, with VWN+MTP reaching the lowest final loss, with a gap of 0.015 versus the baseline at the end of training. Right: Average downstream accuracy (%) versus tokens. Both VWN and VWN+MTP outperform the baseline, and VWN+MTP delivers the highest accuracy throughout training. Models: MoE-2.5B/25B (baseline), MoE-2.5B/25B-VWN, and MoE-2.5B/25B-VWN-MTP.

whereas MTP slightly increases the NTP loss. The combination VWN+MTP attains the lowest loss among the augmented variants but still shows a gap of 0.016 versus the baseline metric when MTP is included. On downstream evaluation of Collection A, MTP alone is comparable with the baseline, while VWN+MTP delivers the highest gains in average accuracy throughout training.

- 2.5B/25B Models. Figure 5 presents results for 2.5B/25B MoE variants. On the training objective (left), VWN reduces the next-token loss relative to the baseline, and adding MTP on top of VWN does not degrade optimization at this scale—both VWN and VWN+MTP achieve similarly low final losses, each approximately 0.015 below the baseline. On downstream evaluation (right), both variants outperform the baseline, with VWN+MTP consistently yielding the best average accuracy across training.

#### 5.2 Large Virtual Width

We study virtual-width scaling on top of a stronger internal baseline. All models include a Multi-Token Prediction (MTP) head by default, jointly optimizing the standard next-token and MTP objectives. We first run ablations on a 0.8B-activation MoE (MoE-A0.8B) to disentangle the effects between increasing m

(finer hidden-partitioning at fixed r) and increasing r (greater virtual width at fixed m). We then scale to a

- 3.3B-activation MoE (MoE-A3.3B) and evaluate the configuration (m,n) = (8,64), corresponding to r = 8, which delivers an 8× virtual widening of the embedding space while preserving the backbone width. We report training dynamics and token efficiency relative to matched non-VWN baselines. Downstream performance is evaluated on Collection B, defined as the average score across the benchmarks in Table 3.

Training Loss (VWNx2)

Training Loss (VWNx4)

Training Loss (VWNx8)

MoE-A0.8B-VWNx2 (m=2) MoE-A0.8B-VWNx2 (m=4)

MoE-A0.8B-VWNx4 (m=16)

MoE-A0.8B-VWNx8 (m=4) MoE-A0.8B-VWNx8 (m=8)

1.70

1.71

MoE-A0.8B-VWNx4 (m=8)

1.69

1.70

1.69

1.68

TrainingLoss

TrainingLoss

TrainingLoss

1.69

1.68

1.67

1.68

1.67

1.66

1.67

1.66

1.65

1.66

1.65

1.64

1.65

1.64

1.63

300 315 330 345 360 375 390

300 315 330 345 360 375 390

300 315 330 345 360 375 390

Tokens (Billions)

Tokens (Billions)

Tokens (Billions)

- Figure 6 Ablation on the fraction rate m under different virtual-width factors r on MoE-A0.8B. Each panel plots next-token training loss versus seen tokens (billions) for VWN×2 (left), VWN×4 (middle), and VWN×8 (right). At r=2, increasing m from 2 to 4 produces a modest but visible improvement. When r=4 or r=8, varying m between tested values leads to only minor differences, suggesting that beyond m≈4 the effect of finer hidden partitioning largely saturates at this model scale.

- Figure 6 presents an ablation on the fraction rate m under different virtual-width factors r on MoE-A0.8B. Each plot shows next-token training loss versus seen tokens (billions). From left to right: r=2, 4, and 8. At r=2, increasing m from 2 to 4 slightly improves convergence, yielding a noticeable but modest gap. At r=4, the variants with m=8 and m=16 nearly overlap, indicating negligible sensitivity to fraction rate. At r=8, the m=4 and m=8 curves are similarly close, with marginal advantage for m=8. Overall, the effect of m diminishes once m>4, suggesting that, at this scale, partition granularity beyond 4 provides limited benefit. Consistent with the discussion in § 4, we hypothesize that, under a fixed r, larger models tend to require higher m to maintain sufficient virtual capacity, whereas smaller models saturate at relatively low m values.

- 5.2.1 Scaling Law of the Virtual Width Factor.

100 150 200 250 300 350 400 450 500

Tokens (Billions)

1.60

1.65

1.70

1.75

1.80

1.85

TrainingLoss

Training Loss on Next Token Prediction

MoE-A0.8B

MoE-A0.8B-VWNx2 MoE-A0.8B-VWNx4 MoE-A0.8B-VWNx8

=0.035

100 150 200 250 300 350 400 450 500

Tokens (Billions)

1.75

1.80

1.85

1.90

1.95

2.00

2.05

TrainingLoss

Training Loss on Next-2 Token Prediction

MoE-A0.8B

MoE-A0.8B-VWNx2 MoE-A0.8B-VWNx4 MoE-A0.8B-VWNx8

=0.058

200 240 280 320 360 400 440 480

Tokens (Billions)

35.0

37.5

40.0

42.5

45.0

47.5

50.0

52.5

Accuracy(%)

Collection B

MoE-A0.8B

MoE-A0.8B-VWNx2 MoE-A0.8B-VWNx4 MoE-A0.8B-VWNx8

=4.16

- Figure 7 Token efficiency of VWN on MoE-A0.8B with a fixed fraction rate m = 8. We vary the virtual width factor by setting r ∈ {2, 4, 8} and n = r · m = {16, 32, 64}. Left/middle: training loss for next-token and next-2-token prediction versus seen tokens. Right: average accuracy on Collection B 3 versus tokens. VWN consistently improves sample efficiency; at 500B tokens, VWN×8 yields ∆ = 0.035 (next-token loss), ∆ = 0.058 (next-2 loss), and a +4.16-point accuracy gain (Collection B, Table 3) over the non-VWN baseline, by leveraging over-width embeddings and GHC without increasing the backbone width.

We evaluate VWN on MoE-A0.8B with a fixed fraction rate m = 8, while varying the virtual-widening factor r ∈ {2,4,8} (n = r · m), to analyze how scaling r influences loss and accuracy (Figure. 7). Across the 500B-token training horizon, VWN yields consistent, monotonic gains with larger r. Table 1 summarizes

improvements over the non-VWN baseline: at 500B tokens, VWN×2, VWN×4, and VWN×8 reduce next-token loss by ∆ = 0.020, 0.028, and 0.035, next-2-token loss by 0.030, 0.045, and 0.058, and improve downstream accuracy by +3.2, +3.5, and +4.16 points, respectively. The ordering VWN×8 > VWN×4 > VWN×2 > baseline remains consistent throughout training, indicating that enlarging the over-width embedding at fixed m systematically enhances model capacity. Results of representative benchmark are illustrated in Figure 9. This collection comprises publicly available benchmarks combined using internal task weights, where a 1-point gain reflects a notable performance difference.

The observed loss reductions follow a log-linear relation with respect to the virtual-width factor r (Figure 8).

- A fitted coefficient of −0.0069 indicates that each doubling of virtual width corresponds to an approximate loss reduction of 0.0069. While the effect size is modest, it suggests a systematic efficiency gain attributable to virtual widening. We hypothesize that more expressive backbones and improved mechanisms that more effectively leverage the virtual-width hidden representations could further amplify the efficiency gains observed with VWN.

2 4 8

Virtual Width Factor (r)

1.6000

1.6025

1.6050

1.6075

1.6100

1.6125

1.6150

1.6175

Loss

r=2 loss=1.6145

r=4 loss=1.6072

r=8 loss=1.6008

Scaling Law: Loss vs Virtual Width Factor

Fit: y = -0.0069·log (x) + 1.6212

Observed

Figure 8 Scaling law analysis of the relationship between virtual width factor r and loss. The observed data (red points) are fitted with a log-linear function y = −0.0069 · log2(x) + 1.6212, with a coefficient of determination R2 = 0.9986.

Table 1 Scaling behavior of VWN on MoE-A0.8B at fixed m = 8. All numbers denote improvements relative to the non-VWN baseline after training on 500B tokens. Each ∆ value represents the reduction in loss compared to the baseline, and accuracy gains are measured on collection B.

Model

∆ NTP Loss

∆ Next-2 Loss

Acc. (+pts)

VWN×2 0.020 0.030 +3.20 VWN×4 0.028 0.045 +3.50 VWN×8 0.035 0.058 +4.16

5.2.2 VWN on Large Scale Model

As shown in Figure 1, we further evaluate Virtual Width Scaling on a 3.3B-activation MoE (MoE-A3.3B) using (m,n) = (8,64), where the hidden dimension is divided into m = 8 partitions, realizing an 8× virtual width expansion. To flexibly control the training length, the learning rate is kept constant throughout training.

VWN markedly accelerates optimization. On MoE-A3.3B, it reaches the baseline’s next-token loss with 2.5× fewer tokens and the next-2-token loss with 3.5× fewer tokens. Meanwhile, the next-token loss gap relative to the baseline increases from ∆ = 0.025 at early stages to about ∆ = 0.032 at 3 T tokens, and the next-2-token loss gap grows from ∆ = 0.049 to ∆ = 0.056. These trends indicate that VWN’s advantage amplifies as training proceeds—its relative efficiency not only appears early but also strengthens over time. The larger gain on the multi-token objective further highlights a strong synergy between virtual width and MTP supervision: the over-width embedding provides richer representational degrees of freedom for short-range compositional targets, while the Generalized Hyper-Connections (GHC) transmit gradients between the virtual-width space and the backbone without expanding intermediate-layer width. On downstream evaluation across Collection

- B, VWN achieves a peak average accuracy that is +2.16 points higher than the baseline, confirming that the performance gap persists and continues to widen with extended training.

### 6 Conclusion

We introduced Virtual Width Networks (VWN) as a practical mechanism to decouple representational width from the quadratic compute typically associated with widening. With a modest 1.5× expansion, we observe consistent improvements. When scaling to 8× virtual width, optimization accelerates markedly: next-token

prediction loss converges more than 2× faster and multi-token prediction loss more than 3× faster relative to the baseline width. Beyond these discrete points, the performance of VWN exhibits a clear scaling behavior. We observe an approximately log-linear relation between the virtual-width factor r and loss reduction, with each doubling of r corresponding to an average loss decrease of about 0.0069. Although the magnitude of the gain is modest, it suggests that virtual width can be treated as a new and predictable dimension for scaling model efficiency, complementing depth-, width-, and data-scaling laws in existing literature. VWN integrates cleanly with standard Transformer stacks and training recipes, providing a concrete reference point for studying capacity/compute trade-offs and for exploring how controlled width expansion can improve quality efficiently. That said, translating these algorithmic gains into production efficiency depends on systems realities. Despite the promising quality-per-compute trade-off, VWN faces practical constraints: as hidden width grows, communication and memory-access overheads become non-negligible, and contemporary hardware is not particularly friendly to very wide activations and cross-device routing. At present, engineering support for extremely wide configurations remains limited, which constrains deployability. In practice, virtual width expansions in the 1.5×–4× range are more feasible on today’s stacks, while larger expansions may require co-design of software, memory layouts, and interconnect strategies to fully realize their potential.

### 7 Contribution

Contributors

Baisheng Li Banggu Wu Bole Ma Bowen Xiao Chaoyi Zhang Cheng Li Chengyi Wang Chengyin Xu Chi Zhang∗ Chong Hu Daoguang Zan Defa Zhu Dongyu Xu Du Li Faming Wu Fan Xia Ge Zhang Guang Shi Haobin Chen Hongyu Zhu Hongzhi Huang Huan Zhou Huanzhang Dou Jianhui Duan Jianqiao Lu Jianyu Jiang Jiayi Xu∗ Jiecao Chen Jin Chen Jin Ma Jing Su Jingji Chen Jun Wang Jun Yuan Juncai Liu Jundong Zhou Kai Hua Kai Shen Kai Xiang Kaiyuan Chen Kang Liu Ke Shen Liang Xiang Lin Yan Lishu Luo Mengyao Zhang Ming Ding Mofan Zhang Nianning Liang Peng Li

Penghao Huang Pengpeng Mu Qi Huang∗ Qianli Ma∗ Qiyang Min Qiying Yu Renming Pang Ru Zhang Shen Yan Shen Yan Shixiong Zhao Shuaishuai Cao Shuang Wu Siyan Chen Siyu Li Siyuan Qiao∗ Tao Sun Tian Xin Tiantian Fan Ting Huang Ting-Han Fan Wei Jia Wenqiang Zhang Wenxuan Liu Xiangzhong Wu Xiaochen Zuo Xiaoying Jia Ximing Yang Xin Liu Xin Yu Xingyan Bin Xintong Hao Xiongcai Luo Xujing Li Xun Zhou Yanghua Peng Yangrui Chen Yi Lin Yichong Leng Yinghao Li Yingshuan Song Yiyuan Ma Yong Shan Yongan Xiang Yonghui Wu Yongtao Zhang Yongzhen Yao Yu Bao Yuehang Yang Yufeng Yuan∗ Yunshui Li

Yuqiao Xian Yutao Zeng∗ Yuxuan Wang Zehua Hong Zehua Wang Zengzhi Wang Zeyu Yang Zhengqiang Yin Zhenyi Lu∗

Zhexi Zhang Zhi Chen Zhi Zhang Zhiqi Lin Zihao Huang Zilin Xu Ziyun Wei Zuo Wang

Authors are listed in alphabetical order. An asterisk (*) denotes former members of the team.

### References

- [1] Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. CoRR, abs/2108.07732, 2021.

- [2] Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, and Xin Wang. Alternating updates for efficient transformers. Advances in Neural Information Processing Systems, 36:76718–76736, 2023.

- [3] Linzheng Chai, Shukai Liu, Jian Yang, Yuwei Yin, Ke Jin, Jiaheng Liu, Tao Sun, Ge Zhang, Changyu Ren, Hongcheng Guo, et al. Mceval: Massively multilingual code evaluation. arXiv preprint arXiv:2406.07436, 2024.

- [4] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2409.19606, 2021.

- [5] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

- [7] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui

- Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412.19437.
- [8] Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. CoRR, abs/1903.00161, 2019.

- [9] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 2022.

- [10] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In Forty-first International Conference on Machine Learning,

2024. URL https://openreview.net/forum?id=pEWAcejiU2.

- [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.

- [12] Yancheng He, Shilong Li, Jiaheng Liu, Yingshui Tan, Weixun Wang, Hui Huang, Xingyuan Bu, Hangyu Guo, Chengwei Hu, Boren Zheng, et al. Chinese simpleqa: A chinese factuality evaluation for large language models. arXiv preprint arXiv:2411.07140, 2024.

- [13] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

- [14] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

- [15] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

- [16] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017.

- [17] Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, and Xun Zhou. Over-tokenized transformer: Vocabulary is generally worth scaling. arXiv preprint arXiv:2501.16975, 2025.

- [18] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, 2023.

- [19] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, Vancouver, Canada, July 2017. Association for Computational Linguistics.

- [20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- [21] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. In Proceedings of the 6th MLSys Conference, 2023.

- [22] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.

- [23] Haoyan Ma, Xiang Li, Xia Yuan, and Chunxia Zhao. Denseformer: A dense transformer framework for person re-identification. IET Computer Vision, 17(5), 2023.

- [24] Kaijing Ma, Xinrun Du, Yunran Wang, Haoran Zhang, Zhoufutu Wen, Xingwei Qu, Jian Yang, Jiaheng Liu, Minghao Liu, Xiang Yue, et al. Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks. arXiv preprint arXiv:2410.06526, 2024.

- [25] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=Ti67584b98.

- [26] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9), 2021.

- [27] N Shazeer, A Mirhoseini, K Maziarz, A Davis, Q Le, G Hinton, and J Dean. The sparsely-gated mixture-of-experts layer. Outrageously large neural networks, 2017.

- [28] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

- [29] Chaofan Tao, Qian Liu, Longxu Dou, Niklas Muennighoff, Zhongwei Wan, Ping Luo, Min Lin, and Ngai Wong. Scaling laws with vocabulary: Larger models deserve larger vocabularies. Advances in Neural Information Processing Systems, 37:114147–114179, 2024.

- [30] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.

- [31] Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018.

- [32] Da Xiao, Qingye Meng, Shengping Li, and Xingyuan Yuan. Muddformer: Breaking residual bottlenecks in transformers via multiway dynamic dense connections. arXiv preprint arXiv:2502.12170, 2025.

- [33] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

- [34] Wei Zhao, Mingyue Shang, Yang Liu, Liang Wang, and Jingming Liu. Ape210k: A large-scale and template-rich dataset of math word problems, 2020.
- [35] Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models, 2023.
- [36] Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou. Hyper-connections. arXiv preprint arXiv:2409.19606, 2024.

- [37] Defa Zhu, Hongzhi Huang, Jundong Zhou, Zihao Huang, Yutao Zeng, Banggu Wu, Qiyang Min, and Xun Zhou. Frac-connections: Fractional extension of hyper-connections. arXiv preprint arXiv:2503.14125, 2025.

## Appendix

### A Detailed Downstream Results for MoE-A0.8B Models

- 55

Accuracy(%)

DROP

MoE-A0.8B

MoE-A0.8B-VWNx8

=8.92

100 150 200 250 300 350 400 450 500

Tokens (Billions)

15

20

25

30

35

40

Accuracy(%)

MATH

MoE-A0.8B

MoE-A0.8B-VWNx8

=4.20

100 150 200 250 300 350 400 450 500

Tokens (Billions)

25

30

35

40

45

50

Accuracy(%)

HumanEval

MoE-A0.8B

MoE-A0.8B-VWNx8

=2.44

100 150 200 250 300 350 400 450 500

Tokens (Billions)

- 56

50

45

40

35

30

100 150 200 250 300 350 400 450 500

Tokens (Billions)

MMLU

70

###### =3.95

68

66

Accuracy(%)

64

62

60

58

MoE-A0.8B

MoE-A0.8B-VWNx8

MMLU-Pro

40.0

37.5

###### =5.25

35.0

Accuracy(%)

32.5

30.0

27.5

25.0

MoE-A0.8B

MoE-A0.8B-VWNx8

22.5

100 150 200 250 300 350 400 450 500

Tokens (Billions)

TriviaQA

65

###### =7.45

60

Accuracy(%)

55

50

45

MoE-A0.8B

40

MoE-A0.8B-VWNx8

100 150 200 250 300 350 400 450 500

Tokens (Billions)

- Figure 9 Performance of VWN on MoE-A0.8B across downstream benchmarks. We compare the non-VWN baseline with VWN×8 (r = 8; n = r · m = 64). VWN×8 consistently outperforms the baseline throughout training; at 500B tokens it yields +8.92 (DROP), +2.44 (HumanEval), +4.20 (MATH), +3.95 (MMLU), +5.25 (MMLU-Pro), and

+7.45 (TriviaQA) accuracy points.

As shown in Figure 9, which plots token efficiency curves across benchmarks, VWN×8 delivers a uniform left-shift of the learning curves, indicating better sample efficiency on all tasks. Improvements are largest on knowledge- and reasoning-heavy benchmarks (DROP, MATH), suggesting that the expanded over-width embedding improves compositional reasoning and retrieval without increasing core compute. HumanEval exhibits smaller gains, consistent with its limited test size. The advantages persist late in training, with no regressions observed, indicating that VWN continues to be utilized rather than saturating early. Notably, VWN achieves particularly strong gains on tasks with relatively long context, such as DROP and TriviaQA, where modeling extended dependencies and multi-sentence evidence aggregation benefits most from the enlarged embedding space. Overall, VWN consistently transfers its token-level efficiency gains to diverse downstream domains, strengthening generalization without increasing backbone width.

### B Implementation of Generalized Hyper-Connections

###### Algorithm 2 Pseudocode of Generalized Hyper-Connections in a PyTorch-like style.

# h: hidden vector (BxLxD) class GHyperConnection(nn.Module):

def __init__(self, dim, m, n_in=3, n_out=2): super().__init__() self.m, self.n_in, self.n_out = m, n_in, n_out self.factor = 1.0 / math.sqrt(dim // self.m)

# Initialize static beta: cyclic pattern static_beta_tensor = torch.zeros(self.m, n_in) for j in range(n_in):

static_beta_tensor[j % self.m, j] = 1.0 self.static_beta = nn.Parameter(static_beta_tensor.T.contiguous()) # Initialize static alpha: block matrix init_alpha = torch.cat([torch.eye(self.m), torch.eye(self.m),

torch.zeros((self.m, self.n_in - self.m))], dim=1) if self.n_in > self.m:

part2 = torch.cat([torch.zeros((self.n_in - self.m, self.m * 2)), torch.eye(self.n_in - self.m)], dim=1)

init_alpha = torch.cat([init_alpha, part2], dim=0) self.static_alpha = nn.Parameter(init_alpha.contiguous()) # Dynamic parameters self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim // self.m, self.m + self.n_in))) self.dynamic_alpha_scale = nn.Parameter(torch.ones_like(self.static_alpha)) self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim // self.m, self.m))) self.dynamic_beta_scale = nn.Parameter(torch.ones_like(self.static_beta)) self.layer_norm = RMSNorm(hidden_size=dim // self.m)

def _base_width_connection(self, h, dynamic_fn, dynamic_scale, static_scale): h_shape = h.shape N, NMM = static_scale.shape M = (NMM - N) // 2 h_reshape = h.reshape((h_shape[:-1].numel(),) + (N, h_shape[-1] // N)) norm_h = self.layer_norm(h_reshape) alpha_beta = (safe_tanh(norm_h @ dynamic_fn.T.to(dtype=norm_h.dtype) * self.factor)

* dynamic_scale[None, ...] + static_scale[None, ...]) alpha, beta = torch.split(alpha_beta, (M + N, M), dim=-1) mix_h = (h_reshape.transpose(1, 2) @ alpha.to(dtype=h_reshape.dtype)).transpose(1, 2) return mix_h.reshape(h_shape[:-1] + mix_h.shape[1:]), beta

def width_connection(self, h): dynamic_fn = torch.concat([self.dynamic_alpha_fn.T, self.dynamic_beta_fn.T], dim=0) dynamic_scale = torch.concat([self.dynamic_alpha_scale, self.dynamic_beta_scale],

dim=-1).contiguous() static_scale = torch.concat([self.static_alpha, self.static_beta], dim=-1) return self._base_width_connection(h, dynamic_fn.to(dtype=h.dtype),

dynamic_scale.to(dtype=h.dtype), static_scale.to(dtype=h.dtype))

def depth_connection(self, mix_h, h_o, beta): h_o_shape = h_o.shape h_o = h_o.reshape(h_o_shape[:-1] + (self.m, h_o_shape[-1] // self.m)) h_i = beta.view(h_o.shape[:2] + beta.shape[1:]).to(dtype=h_o.dtype) @ h_o h = h_i + mix_h[..., self.m:, :] h_shape = h.shape return h.reshape(h_shape[:-2] + (h_shape[-2] * h_shape[-1],)).contiguous()

###### Algorithm 3 Pseudocode of transformer with Generalized Hyper-Connections in a PyTorch-like style.

# h: hidden vector (BxLxD) # atten_ghyper_connection, ffn_ghyper_connection: ghyper-connection modules # attn_norm, ffn_norm: normalization modules

# Attention Block mix_h, beta = atten_ghyper_connection.width_connection(h) mix_h_shape = mix_h.shape h = mix_h[...,:self.rate,:].reshape(mix_h_shape[:-2] + (mix_h_shape[-2] // 2 * mix_h_shape[-1], )) h = attn_norm(h) h = self_attention(h) h = atten_ghyper_connection.depth_connection(mix_h, dropout(h), beta)

# FFN Block mix_h, beta = ffn_ghyper_connection.width_connection(h) mix_h_shape = mix_h.shape h = mix_h[...,:self.rate,:].reshape(mix_h_shape[:-2] + (mix_h_shape[-2] // 2 * mix_h_shape[-1], )) h = ffn_norm(h) h = ffn(h) h = ffn_ghyper_connection.depth_connection(mix_h, dropout(h), beta)

### C Downstream Benchmarks

- Table 2 Downstream Benchmarks Collection A.

|Downstream Benchmarks|
|---|
|ARC_Challenage [5] BBH [28] DROP [8] WinoGrande [26] Hellaswag [33] MMLU [13] MMLU-Pro [30] C-Eval [18] TriviaQA [19] Ape210K [34] GSM8K [6] MATH [14] MBPP [1] HumanEval [4] AGIEval [35] GPQA [25]<br><br>|

- Table 3 Downstream Benchmarks Collection B.

|Downstream Benchmarks|
|---|
|MMLU [13] MMLU-Pro [30] C-Eval [18] AGIEval [35] BBH [28] DROP [8] KOR-Bench-Easy [24] MATH [14] MBPP+ [1] HumanEval [4] McEval [3] TriviaQA [19] Chinese SimpleQA [12]<br><br>|

