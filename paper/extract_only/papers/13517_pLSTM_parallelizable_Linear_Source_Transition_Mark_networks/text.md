arXiv:2506.11997v1[cs.LG]13Jun2025

[Figure 1]

pLSTM: parallelizable Linear Source Transition Mark networks

[Figure 2]

Korbinian Pöppel1,2 Richard Freinschlag1 Thomas Schmied1 Wei Lin1

Sepp Hochreiter1,2 1 ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning Johannes Kepler University Linz, Austria 2 NXAI GmbH poeppel@ml.jku.at

Abstract

Modern recurrent architectures, such as xLSTM and Mamba, have recently challenged the Transformer in language modeling. However, their structure constrains their applicability to sequences only or requires processing multi-dimensionaldata structures, such as images or molecular graphs, in a pre-deﬁned sequential order. In contrast, Multi-Dimensional RNNs (MDRNNs) are well suited for data with a higher level structure, like 2D grids, trees, and directed acyclic graphs (DAGs). In this work, we extend the notion of multi-dimensionality to linear RNNs. We introduce parallelizable Linear Source Transition Mark networks (pLSTMs) using Source, Transition, and Mark gates that act on the linegraph of a general DAG. This enables parallelization in analogy to parallel associative scans and the chunkwise-recurrent form of sequential linear RNNs, but for DAGs. For regular grids (1D and 2D), like images, this scheme can be efﬁciently implemented using einsum operations, concatenations, and padding in logarithmic time. pLSTMs tackle the vanishing/exploding activation/gradient problem for long distances in DAGs via two distinct modes: a directed propagation mode (P-mode) and a diffusive distribution mode (D-mode). To showcase the long-range capabilities of pLSTM, we introduce arrow-pointing extrapolation as a synthetic computer vision task that contains long-distance directional information. We demonstrate that pLSTMs generalize well to larger image sizes, whereas Transformers struggle to extrapolate. On established molecular graph and computer vision benchmarks, pLSTMs also show strong performance. Code and Datasets are available at: https://github.com/ml-jku/plstm_experiments.

- 1 Introduction

Linear RNNs such as DeltaNet [Schlag et al., 2021], Gated Linear Attention [Yang et al., 2023], Mamba [Gu and Dao, 2023, Dao and Gu, 2024], and xLSTM (mLSTM) [Beck et al., 2025b] have recently evolved as a powerful alternative to the Transformer [Vaswani et al., 2017]. In addition to sequence-parallelizable training, such modern recurrent architectures are more efﬁcient at inference time than Transformers. In contrast to classical RNNs, they come with associative memory expansion, enabling more performant sequence modeling. However, these modern recurrent architectures are inherently limited to sequences. While linear RNNs have shown good performance for multi-dimensional data such as images, they enforce a pre-deﬁned sequential traversal struc-

Preprint. Under review.

ture [Zhu et al., 2024, Alkin et al., 2025]. To illustrate, for images, this corresponds to processing patches in a scanline form. However, this is problematic as, within one layer, even (vertical) neighbors in a patch grid are only distantly related, in the (horizontal line-wise) processing. Even when switching between a line- and column-wise order across layers, diagonal relationships are not covered sufﬁciently. This mismatch of short-range spatial and long-distance relations in a certain path with precise positioning requirements leads to credit assignment problems [Minsky, 1961, Bengio and Frasconi, 1993, Schmidhuber, 2015] and suffers from vanishing activations/gradients [Hochreiter, 1991, Bengio et al., 1994, Hochreiter et al., 2001, Hochreiter and Schmidhuber, 1997, Pascanu et al., 2013].

## Non-Recurrent

Recurrent 1D Recurrent in 2D

ViM / ViL VMamba

2DMamba V2M

pLSTM (ours)

ViT

PlainMamba GroupMamba Mamba-ND

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

D-mode

SSM2D Mamba2D

VSSD

CNN

+

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

P-mode

xed @ 45°

learnable input dep.

Signal Strength

Figure 1: Illustration of the receptive ﬁelds induced by pLSTM and related architectures (for a single layer). CNNs are locally restricted while ViTs have a global receptive ﬁeld. Modern recurrent architectures, such as ViM, traverse the 2D grid sequentially. pLSTM effectively extends the receptive ﬁeld via its combination of D-mode and P-mode.

Multi-dimensional RNNs (MDRNNs) [Graves et al., 2007] have demonstrated how non-linear RNNs like LSTMs [Hochreiter, 1991, Hochreiter and Schmidhuber, 1997] can be extended to multidimensional data, such as images [Graves et al., 2007, van den Oord et al., 2016], trees [Tai et al., 2015], and DAGs [Peng et al., 2017]. In this work, we translate the ﬁndings of MDRNNs to linear RNNs and introduce parallelizable Linear Source Transition Mark Networks (pLSTM). To achieve this, we ﬁrst translate their input, forget, and output gates to Source, Transition, and Mark gates, respectively. This is to make the resulting structure applicable to general DAGs. Second, we derive a parallelization scheme enabled by the previous adaptation, leading to higher-order Source, Transition, and Mark terms - forming a parallel associative scan in multiple dimensions and DAGs. This also allows for a chunkwise-parallel formulation known from linear RNNs [Yang et al., 2023, Beck et al., 2025a], using the parallelization up to a certain level and the recurrent option between chunks/patches. Third, we derive two stabilization options, the P-mode for directional propagation and the D-mode for undirected global distribution. In Figure 1, we illustrate how the two modes affect the receptive ﬁeld of pLSTM compared to other linear RNNs. We use pLSTM with MLP layers in a residual pre-LayerNorm backbone, resembling a Transformer with replaced multi-head attention.

To showcase the long-range capabilities of pLSTM in multiple dimensions, we ﬁrst introduce the synthetic arrow-pointing extrapolation task (see Figure 2). As the directional information is not tied to only testing horizontal and vertical cases separately, a certain line-wise or column-wise ordering (even in alternation) cannot capture the directional information in the sequential RNN transitions, at least for a limited number of layers. Importantly, pLSTM generalizes well to increasing image resolutions compared to the Transformer (see Section 5.1). Moreover, we demonstrate the efﬁcacy of pLSTM on established computer-vision and graph benchmarks. Experiments on ImageNet-1k (see Section 5.2) and on molecular graphs (see Section 5.4) show promising results compared to baselines and good scaling behavior to larger model sizes.

In summary, we make the following contributions:

- • We translate the ﬁndings of classic MDRNNs to linear RNNs and introduce pLSTM, which comes with adapted gates and a scalable chunkwise-parallel formulation.
- • We formally derive the general stabilization of pLSTMs for long-range propagation on general DAGs (including images), establishing the P- and D-mode cases.
- • We introduce the synthetic Arrow Pointing Task to highlight the theoretical advantage of pLSTMs, in which pLSTM shows strong extrapolation abilities, and provide a highly scalable implementation of pLSTM.

- 2 Related work

Modern Recurrent Architectures This work presents a new form of linear RNNs, where DeltaNet [Schlag et al., 2021, Yang et al., 2024], LRU [Orvieto et al., 2023], GLA [Yang et al.,

- 2023], Mamba [Gu and Dao, 2023, Dao and Gu, 2024], and xLSTM [Beck et al., 2025b] (in the mLSTM form) have shown their effectiveness on sequence modeling in the language domain. Recently, this line of work has been complemented by TTT [Sun et al., 2024], Titans [Behrouz et al.,
- 2024], and DeltaProduct [Siems et al., 2025], which motivate this structure as a gradient-based optimization in context, in line with early work on Fast-Weight Programmers [Schlag et al., 2021]. Due to their ability for parallelization during training, modern recurrent architectures scale to large-scale datasets similar to Transformers. Moreover, they come with efﬁcient inference, which is attractive for real-world applications [Schmidinger et al., 2024, Schiff et al., 2024, Schmied et al., 2024].

Figure 2: Illustration of the Arrow pointing task. The model has to classify whether an arrow is pointing towards a circle (top left). Models with global receptive ﬁelds, such as Vision Transformers (ViTs), can solve this task by leveraging directional information encoded via positional embeddings (top right), but they often struggle to generalize to higher resolutions. In contrast, pLSTMs can effectively solve this task in both diffusive distribution (D-mode, bottom left) and directed propagation (P-mode, bottom right) modes, enabling long-range reasoning and better scalability.

Non-Linear Multi-Dimensional RNNs The ﬁrst foundational extension of non-linear RNNs / LSTMs to multiple dimensions was carried out by Graves et al. [2007]. Subsequently, Stacked LSTM was proposed [Graves et al., 2013], which stacks LSTM layers on top of each other. MDRNNs are hard to parallelize, while re-arranging the traditional cuboid order of computations in MDLSTMs in pyramidal fashion led to PyraMiD LSTM [Stollenga et al., 2015], which can be parallelized. Grid LSTM [Kalchbrenner et al., 2015] operates in multiple processing dimensions simultaneously, thus generalizing one-dimensional LSTM to more dimensions. TreeLSTM [Tai et al., 2015] and DAG-LSTM [Zhu et al., 2016, Peng et al., 2017, Chen et al., 2017] extend MDRNNs to tree structures and DAGs. PixelRNN [Van Den Oord et al., 2016] operates directly on pixel-leveland improvesMDRNNs for images. Due to their strictly recurrent nature, these architectures are not parallelizable and therefore unsuitable for modern hardware. Moreover, they lack the associative memory component (state expansion) of modern linear RNNs, making them less powerful.

Linear RNNs on Multi-Dimensional data VisionMamba (ViM) [Zhu et al., 2024] and Vision-LSTM (ViL) [Alkin et al.,

- 2025] applied linear RNNs to the vision domain to challenge the common Vision Transformer (ViT) with its quadratic scaling [Dosovitskiy et al., 2021]. These works rely on traversing the 2D plane in a predeﬁned order to accommodate their sequential nature and often employ ﬂipping the traversal order across layers. This was also investigated further in MambaND [Li et al., 2025]. Recent works extend the Mamba architecture to two dimensions. Most related to our work are 2DMamba [Zhang et al., 2024a] and V2M [Wang et al., 2024], which cover only the D-mode of pLSTM, at the loss of directional propagation. Mamba2D [Baty et al., 2024] and 2-D SSM [Baron et al., 2024] cover the

P-mode of pLSTM, but in contrast to pLSTM, they are limited to the diagonal line instead of inputdirected propagation along any line because of their normalization factor of 1/2 (see Appendix C.2).

Relation to GNNs and MPNNs - Since pLSTM propagates recurrent states along a (directed acyclic) graph, it can be seen as a form of Graph Neural Network (GNN) [Scarselli et al., 2009] that covers a whole connected component of a DAG per layer instead of just a one-hop neighborhood.

- 3 Background

- 3.1 Multidimensional RNNs

The most generalform of MDRNNs[Graves et al., 2007] was introducedas DAG LSTM by Zhu et al. [2016], as it can be applied to all directed acyclic graphs P(N,E). We denote n→(e),n←(e), incoming and outgoing node of an edge e ∈ E, and E→(n),E←(n) the sets of incoming and outgoing edges of a node n ∈ N. The core element of (Multi-Dimensional) LSTMs is the LSTM Cell cn computed as:

cn =

n′∈PG(n)

fnn′ ⊙ cn′ + in ⊙ zn (1)

hn = on ⊙ tanh(cn) (2)

with hidden state / output hn and PG(n) denoting the parents of node n in a DAG G. In standard LSTMs, the input in, forget fn, output gates on and the Cell update zn are dependent on external inputs xn and on the previous/parent hidden states (hn′)n′∈PG(n), which makes them non-linear. In contrast, for undirected graphs, DAG LSTMs can be applied by ordering the nodes ﬁrst, thereby ﬁxing edge directions and using a DAG LSTM bi-directionally. Note that there is typically more than one ordering option, where different orders can lead to different outcomes.

- 3.2 Linear RNNs

Linear RNNs resemble the structure of the original LSTM. For linearization, they remove the previous hidden state dependency of the gates and cell update. Additionally, they commonly include a state expansion dimension or query / key dimension. This enables a form of associative memory with inner-product retrieval in relation to quadratic energy-based models [Hopﬁeld, 1982, Schmidhuber, 1992]. To include all relevant dimensions, we switch to the Einstein sum convention notation (so dimensions are visible in the indices) and absorb the input gate in the key and the output gate in the query. In reverse, query and key can also be interpreted as extended input and output gates. Additional normalization, as in mLSTM [Beck et al., 2025b], can be interpreted as a parallel execution with similar inputs and reduced dimensionality.

[Figure 3]

Figure 3: Transition from input/forget/output gating in between nodes (and input/output) in (linear) RNNs towards Source/Transition/Mark gating between edges and input/output (bottom/top) in pLSTMs.

Ctkv = FtCt−1kv + KtkVtv (3) Htv = QtkCtkv (4)

For certain architectures, the scalar forget gate Ft is extended to the key dimension as Ftk in GLA [Yang et al., 2023] or a matrix in DeltaNet as Ftkk′Ct−1k′v with Ftkk′ = 1−βtKtkKtk′ [Schlag et al., 2021, Yang et al., 2024] or Gated DeltaNet [Yang et al., 2025]. DeltaProduct [Siems et al., 2025] uses a product of multiple of these Householder-matrices. The important part is the linearization, by removing the dependence on the previous hidden state. All of Ft, Ktk, Vtv and Qtk depend only on the input of the layer xtd for this time step. This structure enables a chunkwise-recurrent form for efﬁcient execution on modern hardware [Beck et al., 2025a, Yang et al., 2023, 2024].

- 4 pLSTM

Similar to how LSTMs are extended to general DAGs, this can be applied to linear RNNs. Here, we extend the setting of MDRNNs and DAG LSTMs to enable guided directional propagation. Instead of being associated with a node in the graph, the Cell state Ce is now associated with an edge. The inputs and outputs / hidden states are still associated with nodes. Using this re-formulation, we can deﬁne generalized linear DAG networks that resemble the LSTM structure (see Figures 3 and 4):

Tn→(e)

ee′ Ce′kv + Sen→(e)Kkn→(e)Vvn→(e) (5)

Cekv =

e′∈E→(n→(e))

Hvn = Qnk

MenCekv + DnQnkKknVkn (6)

e∈E→(n)

e →(e) at the incoming node in addition to a sum of the Cell states of incoming edges to the incoming node modiﬁed by the respective Transition Teen→′ (e). The hidden state / network output Hvn at node n is a sum of all incoming edge Cell states Cekv modiﬁed by Mark Men and projected by query Qnk. Note that all Source, Transition, and Mark gates are inﬂuenced by the input at node n, and optionally, the edge features of their associated local edges. Functionally, the Source replaces the input gate, the Transition the forget gate, and the Mark the output gate of a traditional LSTM [Hochreiter and Schmidhuber, 1997, Gers, 1999], while acting here on node-edge, edge-edge, and edge-node combinations. The direct Dn term represents a skip directly from input to output at a node and also integrates the Qnk,Kkn,Vkn terms. These equations deﬁne general parallelizable Linear Source Transition Mark (pLSTM) networks on DAGs. This S-T-M structure can be computed sequentially, going through the DAG in a topological node order.

The Cell state receives an external input via the Source Sn

- 4.1 Parallelization on DAGs

- 4.1.1 Naïve Parallelization over Paths Since all operationsare linear, the iterative deﬁnition of Equation 5 can also be resolved to a sum over

connectingall ancestornodesnodesnand′ andancombinationvia ﬁrst edgeofeSource,′ and lastpathedgeTransitione: and Mark over all paths P(e′,e)

Hvn =

n′<n

Kn

′

k V n

′

v

e′,e

Sn

′ e′

⎛ ⎜

⎝ P∈P(e′,e) np∈P

Tn

p

en→pen←p

⎞ ⎟ ⎠

Men

Gn′n

Qnk + KknVkn Dn Gnn

Qnk (7)

The middle STM part in Equation 7 can be precomputed into the resulting gating matrix Gn

′n, and the state expansion and projection via Qnk and Kn

′

k can be integrated externally, similar to classical (linear) self-attention [Vaswani et al., 2017, Katharopoulos et al., 2020], the Dn term represents the diagonal entries. Computing the gating matrix via all paths is infeasible due to their exponential number (see Appendix C.1).

- 4.1.2 Hierarchical Parallelization

Similar to an associative scan in 1D, we can apply a divide-and-conquer approach for hierarchical parallelization. Recursively, we merge Source-Transition, Transition-Transition, and TransitionMark combinations to higher-order Source, Transition, and Mark objects that cover more than a single node. In Appendix A.2, we show how this procedure works for DAGs. For regular grids such as sequences in 1D and images in 2D, we can derive a scheme of einsum, concatenation, and padding transformations, which are well-supported by modern hardware and parallel computation frameworks. In Appendix A.3, we show the parallelization in 1D. In Appendix A.4, we show the parallelization scheme for 2D grids. To better convey the core idea of pLSTM parallelization, we include a sketch for all three cases and pLSTM in general in Figure 4 of the Appendix.

[Figure 4]

- Figure 4: Illustration of pLSTM on general DAGs (left), and on 1D / 2D grids (right): In the

top-left part, a general DAG is visualized, with Sources Sen′, Transitions Te′e, and Marks Men populating node-outgoing-edge, incoming-edge-outgoing-edge, and incoming-edge-node pairs. Due to

the associative structure of these linear operators, they can be combined; examples are shown in the center left. The end-to-end (node-to-node) result is the Gating matrix. In the bottom left, the two stable modes are depicted: The P-mode, where the sum over absolute Transitions of an in-coming edge have to be limited by one, and the D-mode, where the line graph over edges is reduced to a multitree, requiring the remaining Transitions each to be limited in absolute to one. On the right, the application of this to regular 1D and 2D grids is shown. For 2D, the P-mode results in a directional propagation, whereas the D-mode results in an un-directed distribution (shown with decay here).

- 4.2 Long-Range Stability

For sequential data, it is known since the seminal LSTM work of Hochreiter and Schmidhuber [1997] that having forget gates larger or smaller than one in magnitude leads to exploding/vanishing activations and gradients. For general DAGs, this was not explored yet, with recent analyses on undirected GNNs [Arroyo et al., 2025]. In DAG LSTM, Tree LSTM, Grid LSTM, or MD LSTM, potentially exploding parts are limited only in the hidden state by non-linear, bounded activation functions (i.e., tanh) [Hochreiter and Schmidhuber, 1997]. The Cell states, however, can grow exponentially if the forget gates are limited to only one. See Appendix C.1 for details.

Using the more general setting of Transitions Ten′e from edge to edge as introduced in this work, we can derive two modes of stability. First, a propagation mode (P-mode) that covers all paths in

the DAG, but is effectively limited to a line as mode of propagation. Second, a distribution mode (D-mode) that limits the Transitions to a subset, such that their resulting line-graph is reduced from a DAG to a multi-tree Jung [1978].

The line graph of a DAG G = (N,E) is the DAG G′ = (E,E′) formed by the edges E of the original DAG and the line edges e′ ∈ E′ ⟺ e′ = (e1,e2),∃ n ∈ N ∶ e1 ∈ E→(n) ∩ e2 ∈ E←(n), which connecttwo originaledges if these are connected by a node in G [Whitney, 1932]. Given the previous deﬁnitions of Cell states Ce and Transitions Ten′e, these can be interpreted as states and edges on the nodes of the line graph G′. The Transitions Te′e can be viewed as entries of the adjacency matrix T of G′. This way, we can apply the theory of power iterations and matrix norms. The application of a pLSTM of Equation 5 is equivalent to the application of the power iteration and can be bounded by compatible matrix norms ∣ ⋅ ∣:

∣c∣ = ∣ 

∞

p=0

Tp s∣ = ∣

⎛ ⎜ ⎝

PG′

p=0

Tp⎞ ⎟ ⎠

s∣

∣T ∣≤1

≤ PG′∣s∣ (8)

As G is a DAG, also its line graph G′ is a DAG and this one’s adjacency matrix Te′e is nilpotent, for a certain power ∃PG′ ≤ ∣N∣ ∶ ∀p > PG′ ∶ Tp = 0. For simplicity, we use matrix-vector notation here for Cell states c, Transitions T, and Sources s - without key-value extension.

The ﬁrst stabilization option for the P-mode is to limit the norm of T as in Equation 8. A node-local option to achieve this is to limit the Transition entries Ten′e per node as:

e∈E←(n)

∣Ten′e∣ ≤! 1 (9)

This limits the respective column e′ in the line graph adjacency matrix T, and in turn limits its L1norm ∣∣T∣∣1. Since the L1-norm is sub-multiplicative [Horn and Johnson, 1985], this can be applied to the matrix powers in Equation 8 and keeps the L1-norms of the Cell states Ce limited. Keeping the norm ﬁxed at one exactly enables long-range propagation (e.g., by division of the column entries by the norm). This stabilizes the gradient propagation with an L∞ bound, as gradients are propagated backwards and the applied transposed adjacency matrix TT is row-(sub)-normalized and therefore L∞ bounded (the dual norm to the L1 norm).

For the second stabilization option, the D-mode, we do not limit a matrix norm of T, but instead reduce T or G′ from a DAG to a multitree [Jung, 1978], i.e., a directed graph, such that two nodes are only connected by a single path. The original graph G is not reduced to such a multitree by that. Since there is only one path in the line graph structure, there is only one path term in Equation 7, so there is no exponential explosion of path numbers, and limiting single T entries by one is sufﬁcient.

- 4.3 pLSTM on regular grids

Although pLSTMs can be used and parallelized on general DAGs, the additional structure greatly helps with the parallelization for regular grids. All operations can be decomposed into generalized matrix operations in the form of views, einsums, concatenations, and paddings. We show this for the 1D case of sequences in Appendix A.3.

- 4.3.1 pLSTM in 2D - images

On a 2D undirected grid, pLSTM can be applied in different ways, as there are more than two options for DAGs covering the undirected graph. There are, however, four distinct DAG covers, for which the local structure translates to a global structure: allowing only Transitions exclusively left or right and up or down, both in combination: , , , . We focus on the ﬁrst, the right-down combination, for the description, but all four combinations should be used to cover all directional interactions. As Transitions follow all edge combinations, there are four options again: →, ⤷, ⤵, ↓. The level zero Source, Transition, and Mark tensors are therefore: Sxy↦ ,Sxy↧ ,Txy→,Txy⤷,Txy⤵,Txy↓ ,Mxy↣,M xy. Additionally, we now use u for internal indices along the x-axis, and v for internal indices along the y-axis, and a,b,c for an index along the edge/boundary direction, while keeping x,y for outer indices. These are needed for the higher-level tensors, leading to a recursion shown in Appendix A.4. The Source, Transition and Mark tensors at level l have the following structure: Sxyuval,↦ ,Sxyuval,↧ ,Txyabl,→ ,Txyabl,⤷ ,Txyabl,⤵ ,Txyabl,↓ ,Mxyuvbl,↣ ,M xyuvbl, with u,v,a,b ∈ {0..2l − 1}. The general DAG is depicted in Figure 4 in the center right. As the in- and out-degree of this DAG is now larger than one, there are more stabilization options, namely the P-mode and D-mode (Section 4.2).

P-mode When allowing all Transition options →, ⤷, ⤵, ↓, we are in the P-mode. In general, the Transition matrix at position x,y looks like Txy =

Txy→ Txy⤷ Txy⤵ Txy↓

. The P-mode stabilization now lim-

α β

1 − α 1 − β . When setting β = α, we even arrive at a geometrical interpretation of this Transition: it directionally propagates the signal with maximal amplitude along the line deﬁned by ∆∆xy = α (see Appendix C.2 for the derivation). In Figure 4, bottom right, we depict the receptive ﬁeld of a single non-zero Source with constant Transitions of this structure across the grid. For a practical parameterization, we therefore ﬁx this structure with the ’angle’ of propagation α and an additional decay factor γ:

its this matrix to absolute column sums ≤ 1. This limits the structure at criticality to

[Figure 5]

α α (1 − α) (1 − α) (10)

T = γ

D-mode While the P-mode offers a directional propagation, the receptive ﬁeld of a node/patch/pixel is limited to roughly a line, it cannot reach all other nodes. The D-mode can provide

this by losing the notion of directionality. For the D-mode, either Txy⤷ or Txy⤵ have to be set to zero globally. In this way, two edges can only be connected via a single path. In Figure 4, the bottom right, the second option is shown, with an additional decay. At criticality, this node can still reach all other nodes to its bottom right with magnitude one, a long-range propagation in two dimensions.

Directional Initialization and Combination Using the concept of multiple heads, as found in Vaswani et al. [2017], we can initialize our network to cover multiple directions in P-mode at criticality on initialization. Similarly, we can initialize different decay scales for both P- and D-mode for different heads. By using the P- and D-mode in alternating layers, we leverage both their potential.

State-Tracking Expansion In Appendix A.5, we show how pLSTMs can be extended for state tracking via non-diagonal Transitions [Merrill et al., 2024, Siems et al., 2025, Grazzi et al., 2025].

- 5 Experiments

We ﬁrst showcase the theoretical advantages of pLSTM on the synthetic arrow-pointing extrapolation task (see Section 5.1). Then we highlight the beneﬁts of pLSTM for two-dimensional input data on ImageNet-1K [Deng et al., 2009, Russakovsky et al., 2015], demonstrating scalability to large-scale datasets (see Sections 5.2 and 5.3). Finally, we illustrate how pLSTM scales to more than two input dimensions on established graph benchmarks (see Section 5.4).

[Figure 6]

- Figure 5: Training curves for the Arrow Pointing Extrapolation task, averaged over 5 seeds with 90% CI. ViT and EfﬁcientNet can quickly match the training set (left), and EfﬁcientNet reaches the best validation performance of all models on the samples of the same resolution as seen during training. pLSTM performs best on the extrapolation set (right) by a signiﬁcant gap. While Mamba2D and 2DMamba should cover restricted modes of pLSTM, their learning shows high variance. ViL and Mamba2D do not extrapolate at all beyond random performance.

- 5.1 Arrow Pointing Extrapolation

In this task, an image containing an arrow and a circle must be classiﬁed as to whether the arrow is pointing to the circle. This is a long-range directional task as it involves the relative positioning of two objects, in conjunction with local information from one of them - the direction of the arrow. To test for arrow pointing extrapolation capabilities, we generate a balanced dataset of 100k arrow pointing images at resolution 192×192, with positions of the arrow and circle randomly sampled. For validation, we generate 5120 images in the same resolution and at resolution 384 × 384 - to test for extrapolation capabilities. For all models, including the ViT [Dosovitskiy et al., 2021] baseline, we use a bicubic interpolation of the positional embeddings (via jax.image.resize) for scaling to higher resolutions (see also [Dosovitskiy et al., 2021]). For details on the training, see

- Appendix D.2.

In Figure 5, we show the learning curves for pLSTM and all baselines. We ﬁnd that pLSTM performs well on both the training and validation tasks. Importantly, pLSTM exhibits the strongest extrapolation abilities. In contrast, ViL performs poorly because it traverses the image patches sequentially along the scanline form and misses out on important directional information. Despite its global receptive ﬁeld, ViT falls short on extrapolation to larger image resolutions, which we surmise is due the its positional encoding. Similarly, EfﬁcientNet exhibits strong performance on training/validation tasks, but fails to extrapolate. Finally, pLSTM considerably outperforms Mamba2D (which covers the P-mode) and 2DMamba (which covers the D-mode), despite their multidimensional nature. In Figure 6, we highlight the performance differences between D- and P-mode only models on the arrow pointing task.

- 5.2 ImageNet1k

To test for real-world image model capabilities, we train pLSTM using the schedule of DeiTIII [Touvron et al., 2022], comparing our architecture to other vision models. pLSTM performs similarly to other popular approaches. With the integration of local features, such as additional convolutions, as used in Vision LSTM [Alkin et al., 2025], we see room to narrow the gap to stateof-the-art (SOTA) performance. For a detailed discussion on the results and relation to previous work, see Appendix D.3.

[Figure 7]

- Figure 6: Training curves for the Arrow Pointing Extrapolation task, averaged over 5 seeds with 90% CI on different model ablations. P-mode by itself performs worse, D-mode by itself is not as general in interpolation, but performs better on extrapolation compared to other models. pLSTM does not rely on the positional embedding for strong performance.

Table 1: ImageNet-1K pre-training accuracy. All models use a patch size of 16×16 with 224×224 resolution at maximum. Models with “+” in their “Epochs” column pre-train on lower resolution, followed by ﬁne-tuning on 224 × 224 resolution for some epochs. Values of reference models are taken from Alkin et al. [2025] and the original work for 2DMamba, Mamba2D and EfﬁcientNet [Tan and Le, 2019, Zhang et al., 2024b, Baty et al., 2024]. Note that EfﬁcientNet, as a CNNbased baseline, still outperforms the more recent baselines, but was also trained on larger resolutions for the scaled-up versions. Due to the chunkwise-recurrent option, pLSTM FLOPs could be optimized further (see Appendix A.7).

Model Epochs #Params FLOPS IN-1K EfﬁcientNet-B0 [Tan and Le, 2019] ? 5M 0.39G 77.1 DeiT-T [Touvron et al., 2021] 300 6M 1.3G 72.2 DeiT-III-T (reimpl.) [Touvron et al., 2022] 800+20 6M 1.1G 75.4 VRWKV-T [Duan et al., 2024] 300 6M 1.2G 75.1 Vim-T [Zhu et al., 2024] 300 7M 1.5G 76.1 ViL-T [Alkin et al., 2025] 800+20 6M 1.3G 78.3 pLSTM-Vis-T 800+20 6M 1.4G 75.2 EfﬁcientNet-B4 [Tan and Le, 2019] ? 19M 1.8G 82.9 DeiT-S [Touvron et al., 2021] 300 22M 4.6G 79.8 DeiT-III-S (reimpl.) [Touvron et al., 2022] 400+20 22M 4.6G 80.3 ConvNeXt-S (iso.) [Liu et al., 2022] 300 22M 4.3G 79.7 VRWKV-S [Duan et al., 2024] 300 24M 4.6G 80.1 Vim-S [Zhu et al., 2024] 300 26M 5.3G 80.5 Mamba2D-T [Baty et al., 2024] 300 27M - 82.4 2DMamba-T [Zhang et al., 2024a] ? 30M 4.9G 82.8 ViL-S [Alkin et al., 2025] 400+20 23M 4.7G 81.5 pLSTM-Vis-S 400+20 23M 4.9G 80.7 EfﬁcientNet-B6 [Tan and Le, 2019] ? 43M 19G 84.0 DeiT-B [Touvron et al., 2021] 300 86M 17.6G 81.8 DeiT-III-B (reimpl.) [Touvron et al., 2022] 400+20 87M 16.8G 83.5 ConvNeXt-B (iso.) [Liu et al., 2022] 300 87M 16.9G 82.0 VRWKV-B [Duan et al., 2024] 300 94M 18.2G 82.0 2DMamba-S [Zhang et al., 2024a] ? 50M 8.8G 83.8 ViL-B [Alkin et al., 2025] 400+5 89M 17.9G 82.4 pLSTM-Vis-B 400+20 89M 18.2 G 82.5

[Figure 8]

[Figure 9]

[Figure 10]

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

[Figure 24]

[Figure 25]

- 5.3 ImageNet-1k Ablation

We ablate our pLSTM and compare against a ViT trained using the ViT-T model scale and a simpler single pre-training schedule close to DeiT [Touvron et al., 2021] at 224 × 224 resolution (see

- Appendix D.3.1) for details. At this scale, we ﬁnd that pLSTM outperforms ViT by a signiﬁcant margin.

Table 2: Ablation of pLSTM variants and ViT (DeiT) re-training on ImageNet-1k. Model ImageNet-1k top-1 pLSTM 75.51 pLSTM / (no posemb.) 75.22 pLSTM / (P-mode) 74.86 pLSTM / (D-mode) 75.13 pLSTM / (STM bias only) 75.10 ViT 73.49

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

- 5.4 Graphs

Small molecules and proteins are typically depicted as an undirected graph. However, both exist as

- 3D structures in the real world, and GNNs trained on such datasets might beneﬁt from directional informationpropagation, as there is an underlyingspatial relation. We test this hypothesison popular small molecules and bioinformatics datasets from the TUDataset benchmark [Morris et al., 2020]. The results in Table 3 show that pLSTM can compete with popular GNN architectures on those datasets. Similar to the computer vision experiments (Section 5.2), pLSTM alternates between the P-mode and the D-mode. Since there is no external notion of direction, we use node and edge features to compute the Transition for the P-mode.

Table 3: Test accuracy on small molecule and bioinformatics datasets as provided by TUDataset.

[Figure 33]

model MUTAG NCI1 PROTEINS PTC_FM AVG

[Figure 34]

[Figure 35]

[Figure 36]

GAT [Velickoviˇ c´ et al., 2018] 0.7822 ± 0.09 0.7968 ± 0.03 0.7215 ± 0.03 0.6105 ± 0.05 0.7277 ± 0.06 GCN [Kipf and Welling, 2017] 0.7234 ± 0.08 0.7852 ± 0.02 0.7395 ± 0.03 0.6162 ± 0.04 0.7161 ± 0.05 GIN [Xu et al., 2018] 0.8251 ± 0.10 0.8175 ± 0.02 0.7350 ± 0.04 0.6097 ± 0.10 0.7468 ± 0.08 LSTM GNN [Liang et al., 2016] 0.7450 ± 0.11 0.7951 ± 0.02 0.7503 ± 0.04 0.6076 ± 0.04 0.7245 ± 0.06 MPNN [Gilmer et al., 2017] 0.7450 ± 0.09 0.8012 ± 0.02 0.7350 ± 0.04 0.5786 ± 0.07 0.7149 ± 0.06 PLSTM 0.8512 ± 0.06 0.7324 ± 0.03 0.7502 ± 0.05 0.6133 ± 0.08 0.7368 ± 0.06

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

- 6 Conclusion

In this work, we introduce pLSTM, which unites the beneﬁts of MDRNNs [Graves et al., 2007] and the recently introduced xLSTM [Beck et al., 2025b]. pLSTM overcomes the limitations of modern recurrent architectures when applied to multi-dimensional data, such as images and graphs. To achieve this, we modify the gating structure of xLSTM and introduce Source, Transition, and Mark gates. Then, we derive a parallelization scheme that enables processing data in multiple dimensions concurrently. pLSTM comes with a P-mode and a D-mode, which together enable a large and auto-tunable effective receptive ﬁeld. We demonstrate the theoretical advantages of pLSTM on the arrow-pointing task and highlight its ability to generalize to varying grid resolutions. Finally, we show the efﬁcacy of pLSTM both on classical computer vision and graph benchmarks.

Limitations & Future Work pLSTM shows promising results across different domains, however, there is still room for improvement compared to highly optimized domain-speciﬁc models. With the incorporation of domain-speciﬁc inductive biases, we are positive that the results can be further improved. While pLSTMs should theoretically enable long-range propagation of activations and gradients in multiple dimensions for recurrent, subquadratic models, the arrow pointing extrapolation task only tests this in a restricted way. Moreover, while pLSTM models generalize better than ViTs, which use position embeddings to encode spatial relations, there is still a gap to perfect extrapolation. This leaves room for improvements, both on the data side of harder multi-dimensional long-range benchmarks and the architectural side to better generalize to that data. Given the ﬂexibility of pLSTM to handle data with rich multi-dimensional structure, we anticipate that pLSTMs can be successfully applied across a broader spectrum of challenging domains, including biology, chemistry, medical imaging, and other scientiﬁc domains.

Acknowledgments and Disclosure of Funding

We thank Benedikt Alkin, Pieter-Jan Hoedt, Phillip Lippe, Maximilian Beck, Kajetan Schweighofer, Erich Kobler, and Günter Klambauer for helpful discussions and feedback.

The ELLIS Unit Linz, the LIT AI Lab, the Institute for Machine Learning, are supported by the Federal State Upper Austria. We thank the projects FWF AIRI FG 9-N (10.55776/FG9), AI4GreenHeatingGrids (FFG- 899943), Stars4Waters (HORIZON-CL6-2021-CLIMATE-01-01), FWF Bilateral Artiﬁcial Intelligence (10.55776/COE12). We thank NXAI GmbH, Audi AG, Silicon Austria Labs (SAL), Merck Healthcare KGaA, GLS (Univ. Waterloo), TÜV Holding GmbH, Software Competence Center Hagenberg GmbH, dSPACE GmbH, TRUMPF SE + Co. KG.

We acknowledge EuroHPC Joint Undertaking for awarding us access to Leonardo at CINECA, Italy and MareNostrum5 as BSC, Spain.

References

- B. Alkin, M. Beck, K. Pöppel, S. Hochreiter, and J. Brandstetter. Vision-LSTM: xLSTM as Generic Vision Backbone. In The Thirteenth InternationalConference on Learning Representations, 2025. URL https://openreview.net/forum?id=SiH7DwNKZZ.

A. Arroyo, A. Gravina, B. Gutteridge, F. Barbero, C. Gallicchio, X. Dong, M. Bronstein, and P. Vandergheynst. On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning, Feb. 2025. URL http://arxiv.org/abs/2502.10818. arXiv:2502.10818 [cs].

- E. Baron, I. Zimerman, and L. Wolf. A 2-Dimensional State Space Layer for Spatial Inductive Bias. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=BGkqypmGvm.

- E. Baty, A. H. Díaz, C. Bridges, R. Davidson, S. Eckersley, and S. Hadﬁeld. Mamba2D: A Natively Multi-Dimensional State-Space Model for Vision Tasks, Dec. 2024. URL http://arxiv.org/abs/2412.16146. arXiv:2412.16146 [cs].

M. Beck, K. Pöppel, P. Lippe, and S. Hochreiter. Tiled ﬂash linear attention: More efﬁcient linear rnn and xlstm kernels. arXiv preprint arXiv:2503.14376, 2025a.

M. Beck, K. Pöppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. xLSTM: Extended long short-term memory. Advances in Neural Information Processing Systems, 37:107547–107603, 2025b.

A. Behrouz, P. Zhong, and V. Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.

Y. Bengio and P. Frasconi. Credit assignment through time: Alternatives to backpropagation. Advances in neural information processing systems, 6, 1993.

Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difﬁcult. IEEE transactions on neural networks, 5(2):157–166, 1994.

- X. Chen, Z. Shi, X. Qiu, and X. Huang. Dag-based long short-term memory for neural word segmentation. arXiv preprint arXiv:1707.00248, 2017.

T. Dao and A. Gu. Transformers are ssms: Generalized models and efﬁcient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024.

J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani,

- M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Y. Duan, W. Wang, Z. Chen, X. Zhu, L. Lu, T. Lu, Y. Qiao, H. Li, J. Dai, and W. Wang. Visionrwkv: Efﬁcient and scalable visual perception with rwkv-like architectures. arXiv preprint arXiv:2403.02308, 2024.

F. Gers. Learning to forget: continual prediction with LSTM. In 9th International Conference on Artiﬁcial Neural Networks: ICANN ’99, volume 1999, pages 850–855, Edinburgh, UK, 1999. IEE. ISBN 978-0-85296-721-8. doi: 10.1049/cp:19991218. URL https://digital-library.theiet.org/content/conferences/10.1049/cp_19991218.

J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural Message Passing for Quantum Chemistry. In D. Precup and Y. W. Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 1263–1272, 2017.

Graves, A.-R. Mohamed, and G. E. Hinton. Speech recognition with deep recurrent neural networks. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 6645–6649. IEEE, 2013. doi: 10.1109/icassp.2013.6638947.

A. Graves, S. Fernández, and J. Schmidhuber. Multi-dimensional recurrent neural networks. In International Conference on Artiﬁcial Neural Networks (ICANN), volume 4668 of Lecture Notes in Computer Science, pages 6645–6649. Springer, Berlin, Heidelberg, 2007.

- R. Grazzi, J. Siems, J. K. H. Franke, A. Zela, F. Hutter, and M. Pontil. Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=UvTo3tVBk2.

A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

- S. Hochreiter. Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, Institut für Informatik, Lehrstuhl Prof. Brauer, Technische Universität München, 1991.

S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 9(8):1735–1780, 1997.

- S. Hochreiter, Y. Bengio, P. Frasconi, J. Schmidhuber, et al. Gradient ﬂow in recurrent nets: the difﬁculty of learning long-term dependencies, 2001.

J. J. Hopﬁeld. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8):2554– 2558, Apr. 1982. ISSN 0027-8424, 1091-6490. doi: 10.1073/pnas.79.8.2554. URL https://pnas.org/doi/full/10.1073/pnas.79.8.2554.

- R. A. Horn and C. R. Johnson. Matrix Analysis. Cambridge University Press, 1 edition, Dec. 1985. ISBN 978-0-521-38632-6 978-0-52130586-0 978-0-511-81081-7. doi: 10.1017/CBO9780511810817. URL https://www.cambridge.org/core/product/identifier/9780511810817/type/book.

- H. A. Jung. On a class of posets and the corresponding comparability graphs. Journal of Combinatorial Theory, Series B, 24(2):125–133, 1978. ISSN 0095-8956. doi: https://doi.org/10.1016/0095-8956(78)90013-8. URL https://www.sciencedirect.com/science/article/pii/0095895678900138.

- N. Kalchbrenner, I. Danihelka, and A. Graves. Grid long short-term memory. arXiv, 1507.01526, 2015.

A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are RNNs: fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.

- T. N. Kipf and M. Welling. Semi-Supervised Classiﬁcation with Graph Convolutional Networks. In International Conference on Learning Representations, 2017.

- S. Li, H. Singh, and A. Grover. Mamba-ND: Selective State Space Modeling for Multi-dimensional Data. In A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, editors, Computer Vision – ECCV 2024, pages 75–92, Cham, 2025. Springer Nature Switzerland. ISBN 978-3-031-73414-4.

- X. Liang, X. Shen, J. Feng, L. Lin, and S. Yan. Semantic object parsing with graph LSTM. CoRR, abs/1603.07063, 2016. URL http://arxiv.org/abs/1603.07063.

Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

- W. Merrill, J. Petty, and A. Sabharwal. The illusion of state in state-space models. arXiv preprint arXiv:2404.08819, 2024.

- M. Minsky. Steps toward artiﬁcial intelligence. Proceedings of the IRE, 49(1):8–30, 1961. doi: 10.1109/JRPROC.1961.287775.

C. Morris, N. M. Kriege, F. Bause, K. Kersting, P. Mutzel, and M. Neumann. TUDataset: A collection of benchmark datasets for learning with graphs. In ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020), 2020.

A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De. Resurrecting recurrent neural networks for long sequences. In International Conference on Machine Learning, pages 26670–26698. PMLR, 2023.

J. Park. deit3-jax, 2024. URL https://github.com/affjljoo3581/deit3-jax. Publication Title: GitHub repository.

- R. Pascanu, T. Mikolov, and Y. Bengio. On the difﬁculty of training recurrent neural networks. In International conference on machine learning, pages 1310–1318. Pmlr, 2013.

- N. Peng, H. Poon, C. Quirk, K. Toutanova, and W.-t. Yih. Cross-sentence n-ary relation extraction with graph lstms. Transactions of the Association for Computational Linguistics, 5:101–115, 2017.
- O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla,

- M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252, 2015. doi: 10.1007/ s11263-015-0816-y.

F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini. The Graph Neural Network Model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009. doi: 10.1109/TNN.2008. 2005605.

Y. Schiff, C.-H. Kao, A. Gokaslan, T. Dao, A. Gu, and V. Kuleshov. Caduceus: Bi-directional equivariant long-range dna sequence modeling. arXiv preprint arXiv:2403.03234, 2024.

- I. Schlag, K. Irie, and J. Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021.
- J. Schmidhuber. Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Networks. Neural Computation, 4:131–139, 1992. URL https://api.semanticscholar.org/CorpusID:16683347.

- J. Schmidhuber. Deep learning in neural networks: An overview. Neural networks, 61:85–117, 2015.

- N. Schmidinger, L. Schneckenreiter, P. Seidl, J. Schimunek, P.-J. Hoedt, J. Brandstetter, A. Mayr,

- S. Luukkonen, S. Hochreiter, and G. Klambauer. Bio-xlstm: Generative modeling, representation and in-context learning of biological and chemical sequences. arXiv preprint arXiv:2411.04165, 2024.

- T. Schmied, T. Adler, V. Patil, M. Beck, K. Pöppel, J. Brandstetter, G. Klambauer, R. Pascanu, and S. Hochreiter. A large recurrent action model: xlstm enables fast inference for robotics tasks. arXiv preprint arXiv:2410.22391, 2024.

- J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, and R. Grazzi. Deltaproduct: Improving state-tracking in linear rnns via householder products. arXiv preprint arXiv:2502.10297, 2025.

M. Stollenga, W. Byeon, M. Liwicki, and J. Schmidhuber. Parallel multi-dimensional lstm, with application to fast biomedical volumetric image segmentation. In Proceedings of the 28th International Conference on Neural Information Processing Systems (NIPS), pages 2998–3006, 2015.

Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, S. Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024.

- K. S. Tai, R. Socher, and C. D. Manning. Improved semantic representations from tree-structured long short-term memory networks. arXiv preprint arXiv:1503.00075, 2015.

M. Tan and Q. Le. Efﬁcientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.

H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou. Training data-efﬁcient image transformers& distillation through attention. In Internationalconference on machine learning, pages 10347–10357. PMLR, 2021.

H. Touvron, M. Cord, and H. Jégou. DeiT III: Revenge of the ViT. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIV, pages 516–533, Berlin, Heidelberg, 2022. Springer-Verlag. ISBN 978-3-031-20052-6. doi: 10.1007/ 978-3-031-20053-3_30. URL https://doi.org/10.1007/978-3-031-20053-3_30. eventplace: Tel Aviv, Israel.

A. Van Den Oord, N. Kalchbrenner, and K. Kavukcuoglu. Pixel recurrent neural networks. In International conference on machine learning, pages 1747–1756. PMLR, 2016.

A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu. Pixel Recurrent Neural Networks. In M. F. Balcan and K. Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1747–1756, New York, New York, USA, June 2016. PMLR. URL https://proceedings.mlr.press/v48/oord16.html.

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- P. Velickoviˇ c,´ G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio. Graph Attention Networks. In International Conference on Learning Representations, 2018.

C. Wang, W. Zheng, Y. Huang, J. Zhou, and J. Lu. V2M: Visual 2-Dimensional Mamba for Image Representation Learning. arXiv preprint arXiv:2410.10382, 2024.

H. Whitney. Congruent Graphs and the Connectivity of Graphs. American Journal of Mathematics, 54(1):150, Jan. 1932. ISSN 00029327. doi: 10.2307/2371086. URL https://www.jstor.org/stable/2371086?origin=crossref.

- K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How powerfulare graph neural networks? arXiv preprint arXiv:1810.00826, 2018.

- K. Xu, W. Hu, J. Leskovec, and S. Jegelka. How Powerful are Graph Neural Networks? In International Conference on Learning Representations, 2019.

- S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformerswith hardwareefﬁcient training. arXiv preprint arXiv:2312.06635, 2023.

S. Yang, B. Wang, Y. Zhang, Y. Shen, and Y. Kim. Parallelizing linear transformers with the delta rule over sequence length. arXiv preprint arXiv:2406.06484, 2024.

- S. Yang, J. Kautz, and A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=r8H7xhYPwz.

J. Zhang, A. T. Nguyen, X. Han, V. Q.-H. Trinh, H. Qin, D. Samaras, and M. S. Hosseini. 2dmamba: Efﬁcient state space model for image representation with applications on giga-pixel whole slide image classiﬁcation. arXiv preprint arXiv:2412.00678, 2024a.

J. Zhang, A. T. Nguyen, X. Han, V. Q.-H. Trinh, H. Qin, D. Samaras, and M. S. Hosseini. 2DMamba: Efﬁcient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classiﬁcation, Dec. 2024b. URL http://arxiv.org/abs/2412.00678. arXiv:2412.00678 [cs].

- L. Zhu, B. Liao, Q. Zhang, X. Wang, and W. Liu. Vision mamba: Efﬁcient visual representation learning with bidirectional state space model (preprint). 2024.

- X. Zhu, P. Sobhani, and H. Guo. Dag-structured long short-term memory for semantic compositionality. In Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: Human language technologies, pages 917–926, 2016.

- A Method Details

- A.1 Loosely self-similar graphs

Graphs modeling hierarchical order or spatial extensions, as in meshes or grids do typically have low node degree and can be decomposed recursively into subgraphs of similar structure. We use the term loosely self-similar graphs for these. In particular, a graph G ∶= G(L) should be decomposable into M subgraphs {Ga(L−1) = (Na(L−1),Ea(L−1))}a∈{1..M}, and boundary edges Bab(L−1) ⊂ Na(L−1) × Nb(L−1) such that G = G(L) = ⋃a Na(L−1),⋃a,b Bab(L−1) ∪ ⋃a Ea(L−1) . This decomposition should be applicable recursively down to single nodes, such that ∣Nα(0)∣ = 1 ∀ α with α = (aL−1...a0) the multi-index of successive decompositions. Ideally, all decompositions should be balanced, ∣Nα(l)∣ − ∣Nβ(l)∣ ∈ {−1,0,1} ∀ l,α,β, for optimal parallelization - which should serve as the deﬁnition of loosely self-similar. Choosing arbitrary subsets of nodes for subgraphs, one can always arrive at a decomposition of any graph, but the edges across subgraphs might not be balanced then.

- A.2 Hierarchical Parallelization

Given a decomposition of a (loosely self-similar) DAG, we can apply a divide-and-conquer strategy to the expensive Transition sum/product combination. We want to calculate now the Gn

′n matrix entry for two nodes n′,n ∈ N;n′ ≤ n, such that, at level l + 1, both are in the subgraph Gαl+1 ∶ n′,n ∈ Nαl+1 but going one level lower, they are in distinct subgraphs n′ ∈ Nαl∶a, n ∈ Nαl∶b a ≠ b. With incoming and outgoing boundary edges e′ ∈ B→l α∶a, e ∈ B←l α∶a, we deﬁne the generalized Source, Transition, and Mark tensors at level l:

′,l,α∶a

′ e′

Ten

Sn

Sn

→ en←P (11)

e ∶=

nP

P∈P(e′,e) nP ∈P

e′∈E←(n′)

Tel,α′e ∶k ∶=

Ten

→ en←P (12)

nP

P∈P(e′,e) nP ∈P

⎛ ⎜

⎞ ⎟ ⎠

Men,l,α′ ∶b ∶=

Ten

Men (13)

⎝ P∈P(e′,e) nP∈P

nP

→ en←P

e∈E→(n)

Given these, the gating matrix entry can be calculated as:

⎛ ⎜

⎞ ⎟ ⎠

′,l,α∶b e′

′n =

′,l,α∶a e

Tl,α∶c el,α∶c →ml c

Mn

Gn

Sn

⎝ Pl∈Pl(e′,e) c∈Pl

el,α∶c ←ml c

e∈B←l α∶a e′∈B→l α∶j

Here, Pl is the set of "meta-paths" that connect two edges e′, e via subgraphs at level l, in these meta-paths the subgraphs act as "nodes" and the boundary edges B←l α∶a, B→l α∶a as "edges". Note that, by Einstein sum convention we implicitly sum over multiple in-coming and out-going edges e→l,αm∶cl

′n and in the following. See Figure 4 for an illustration of this decomposition. In reverse, this means that higher-level Source, Transition, and Mark can be constructed recursively bottom-up from one level lower (a chosen such that n′ ∈ Nαl∶a or n ∈ Nαl∶a):

and e←l,αm∶cl

for generalized Transitions Tl,α∶c in Gn

c

c

′,l,α∶a e′

′,l+1,α

Tel,αl,α∶∶bb

Sn

Sn

(14)

e =

el,α∶b ←ml b

Pl∈Pl(e′,e) b∈Pl

→ml b

e′∈B←l α∶a

Tel′+e1,α =

Tel,αl,α∶∶bb

(15)

el,α∶b ←ml b

Pl∈Pl(e′,e) b∈Pl

→ml b

⎛ ⎜

⎞ ⎟ ⎠

Men,l′ +1,α =

Tel,αl,α∶∶bb

Men,l,α∶a (16)

⎝ Pl∈Pl(e′,e) b∈Pl

el,α∶b ←ml j

→ml b

e∈B→l α∶a

Note that using this hierarchy only up to a certain level, one can optimize the parallelization to the used hardware. Whereas ﬁrst, all the Source, Transition, Mark, and Gating matrix objects are constructed in parallel up to a certain level, the recurrent mode of Equation 5 is applied with generalized Source, Transition and Mark matrices of level l, in the topological ordering of the subgraphs at this level. This resembles the structure of the chunk-wise parallel formulation for linear RNNs.

- A.3 Parallelization of pLSTM in 1D - sequences

For a sequence as a DAG, pLSTMs reduce to linear RNNs. Here, all nodes and subgraphs have a single incoming and outgoing (boundary) edge. For parallelization, the decomposition is done hierarchically in powers-of-two splits. The S, T, M and G tensors are constructed as follows:

Sanl = S(l2−a1)(n⌋

0)T(l2−a1+1),S(l2−a1+1)(n⌉

0) 0 (17) Tal = T(l2−a1)T(l2−a1+1) (18)

Manl = M(l2−a1+1)(n⌋

0),T(l2−a1)M(l2−a1+1)(n⌉

0) 0 (19) Glan′n = Gl(2a)(n′⌋0)(n⌋1),0 0 , (20)

S(l2−a1)(n′⌋0)M(l2−a1+1)(n⌉

1),Gl(2a+1)(n′⌉0)(n⌉1) 0 1

with a the external index (Source index ∈ {0..S/2l − 1} of level l), n the internal index (node index ∈ {0..2l −1} within generalized Source Sal). Here, []i denotes a concatenation along the axis i, and ⌋i and ⌉i denote these indices are used as axis i for this concatenation - one taking the lower half, one the upper half, e.g. for vectors a ∈ RN; b,c ∈ RN/2:

an = b(n⌋

0),c(n⌉

0) 0 ∶=

bn n < N/2 cn−N/2 n ≥ N/2

(21)

Note that we use a causal setting here. For a non-causal structure, one would apply this bi– bidirectionally, effectively ﬁlling the top-right entries of the gating matrix with a different Source/Mark combination for the opposite order. Equivalently, both directions can be added in their outputs while computing the parallel parts twice.

- A.4 Parallelization of pLSTM in 2D - images

Sxyuwal+1,↦ = S(l,2↦x)(2y)(u⌋

0)(w⌋1)bT(l,2→x+1)(2y)b(a⌋

2),S(l,2↦x+1)(2y)(u⌉

0)(w⌋1)(a⌋2) 0, 0(u⌋

0)(w⌉1)(a⌋2),0(u⌋

0)(w⌉1)(a⌋2) 0 1

,

(22) S(l,2↦x)(2y)(u⌋

0)(w⌋1)bT(l,2⤵x+1)(2y)bcT(l,2⤷x+1)(2y+1)c(a⌉

2) + S(l,2↧x)(2y)(u⌋

0)(w⌋1)bT(l,2⤷x)(2y+1)bcT(l,2→x+1)(2y+1)c(a⌉

2),

S(l,2↧x+1)(2y)(u⌉

0)(w⌋1)bT(l,2⤷x+1)(2y+1)bc 0, S(l,2↦x)(2y+1)(u⌋

0)(w⌉1)bT(l,2→x+1)(2y+1)b(a⌉

2),S(l,2↦x+1)(2y+1)(u⌉

0)(w⌉1)(a⌉2) 0 1

2

(23)

Sxyuwal+1,↧ = S(l,2↧x)(2y)(u⌋

0)(w⌋1)bT(l,2↓x)(2y+1)b(a⌋

0)(w⌋0)(a⌋2) 0, S(l,2↧x)(2y+1)(u⌋

2),0(u⌉

0)(w⌉1)(a⌋2),0(u⌉

0)(w⌉0)(a⌋2) 0 1

,

(24) S(l,2↦x)(2y)(u⌋

0)(w⌋1)bT(l,2⤵x+1)(2y)bcT(l2↓x+1)(2y+1)c(a⌉

2) + S(l,2↧x)(2y)(u⌋

0)(w⌋1)bT(l,2⤷x)(2y+1)bcT(⤵2x+1)(2y+1)c(a⌉

2),

S(l,2↧x+1)(2y)(u⌉

0)(v⌋1)bT(l,2↓x+1)(2y+1)b(a⌉

0) 0, S(l,2↦x)(2y+1)(u⌋

0)(w⌉1)bT(l,2⤵x+1)(2y+1)b(a⌉

2),S(l,2↧x+1)(2y+1)(u⌉

0)(v⌉1)(a⌉2) 0 1

2

Txyabl,→ = T(l,2→x)(2y)(a⌋

0)cT(l,2→x+1)(2y)c(b⌋

0)(b⌋1) 0, (25) T(l,2↓x)(2y)(a⌋

1),0(a⌉

0c)T(l,2⤷x)(2y+1)cdT(l,2→x+1)(2y+1)d(b⌉

1) + T(l,2→x)(2y)(a⌋

0)cT(l,2⤵x+1)(2y)cdT(l,2⤷x+1)(2y+1)d(b⌉

1), T(l,2→x)(2y+1)(a⌉

0)cT(l,2→x+1)(2y+1)c(b⌉

1),0(a⌉

0)(b⌋1) 0 1

Txyabl,↓ = T(l,2↓x)(2y)(a⌋

0)cT(l,2↓x)(2y+1)c(b⌋

0)(b⌋1) 0, (26) T(l,2↓x)(2y)(a⌋

1),0(a⌉

0c)T(l,2⤷x)(2y+1)cdT(l,2⤵x+1)(2y+1)d(b⌉

1) + T(l,2→x)(2y)(a⌋

0)cT(l,2⤵x+1)(2y)cdT(l,2↓x+1)(2y+1)d(b⌉

1), T(l,2↓x+1)(2y)(a⌉

0)cT(l,2↓x+1)(2y+1)c(b⌉

1),0(a⌉

0)(b⌋1) 0 1

Txyabl,⤷ = T(l,2⤷x)(2y)(a⌋

0)cT(l,2→x+1)(2y)c(b⌋

1),T(l,2⤷x+1)(2y)(a⌉

0)(b⌋1) 0, (27) T(l,2⤷x)(2y)(a⌋

0)cT(l,2⤵x+1)(2y)cdT(l,2⤷x+1)(2y+1)d(b⌉

1) + T(l,2↓x)(2y)(a⌋

0)cT(l,2⤷x)(2y+1)cdT(l,2→x+1)(2y+1)d(b⌉

1), T(l,2↓x+1)(2y)(a⌉

0)cT(l,2⤷x+1)(2y+1)c(b⌉

1) 0 1

Txyabl,⤵ = T(l,2⤵x)(2y)(a⌋

0)cT(l,2↓x)(2y+1)c(b⌋

1),T(l,2⤵x)(2y+1)(a⌉

0)(b⌋1) 0, (28) T(l,2⤵x)(2y)(a⌋

0)cT(l,2⤷x)(2y+1)cdT(l,2⤵x+1)(2y+1)d(b⌉

1) + T(l,2→x)(2y)(a⌋)

0cT(l,2⤵x+1)(2y)cdT(l,2↓x+1)(2y+1)d(b⌉

1), T(l,2→x)(2y+1)(a⌉

0)cT(l,2⤵x+1)(2y+1)c(b⌉

1) 0 1

Mxyuval,↣ = M(l,2↣x)(2y)(u⌋

0)(v⌋1)(a⌋2),T(l,2→x)(2y)(a⌋

2)bM(l,2↣x+1)(2y)(u⌉

0)(v⌋1)b 0, T(l,2⤵x)(2y)(a⌋

2)bM (l,2x)(2y+1)(u⌋

0)(v⌉1)b,

(29) T(l,2⤵x)(2y)(a⌋

2)bT(l,2⤷x)(2y+1)bcM(l,2↣x+1)(2y+1)(u⌉

0)(v⌉1)c + T(l,2→x)(2y)(a⌋

2)bT(l,2⤵x+1)(2y)bcM (l,2x+1)(2y+1)(u⌉

0)(v⌉1)c 0 1

,

0)(v⌋1)(a⌉2) 0, M(l,2↣x)(2y+1)(u⌋

0)(v⌉1)(a⌉2),T(l,2→x)(2y+1)(a⌉

2)bM(l,2↣x+1)(2y+1)(u⌉

0(u⌋

0)(v⌋1)(a⌉2),0(u⌉

0)(v⌉1)b 0 1

2

2)bM (l,2x)(2y+1)(u⌋

0)(v⌋1)b 0, T(l,2↓x)(2y)(a⌋

2)bM(l,2↣x+1)(2y)(u⌉

0)(v⌋1)(a⌋2),T(l,2⤷x)(2y)(a⌋

- M xyuval, = M (l,2x)(2y)(u⌋

0)(v⌉1)b,

(30) T(l,2⤷x)(2y)(a⌋

2)bT(l,2⤵x+1)(2y)bcM (l,2x+1)(2y+1)(u⌉

0)(v⌉1)c + T(l,2↓x)(2y)(a⌋

2)bT(l,2⤷x)(2y+1)bcM(l,2↣x+1)(2y+1)(u⌉

0)(v⌉1)c 0 1

,

0)(v⌋1),M (l,2x+1)(2y)(u⌉

0)(v⌉1,T(l,2↓x+1)(2y)(a⌉

2)bM (l,2x+1)(2y+1)(u⌉

0(u⌋

0)(v⌋1) 0, 0(u⌋

0)(v⌉1)b 0 1

2

- A.5 State-Tracking Extension

Recent research has shown that having scalar, positive Transitions Ten

→e← does not allow for state tracking capabilities in these linear RNNs [Merrill et al., 2024, Grazzi et al., 2025]. We can generalize the Transition scalars to Transition matrices here as well: Tenn

→en← → Tenn

→en←ij i,j ∈ {1..JT}, JT being the Transition state tracking dimension. Note that in this way, the Source and Mark matrix have to include this extended dimension as well. We can deﬁne generalizations with additional state tracking dimensions JS,JM:

Sn

′,l,α

e → Sn

′,l,α

eki k ∈ {1..JS},i ∈ 1..JT (31) Tel,α′e → Tel,α′eij i,j ∈ {1..JT} (32)

Men,l′ +1,α → Men,l,α′jm j ∈ {1..JT},m ∈ {1..JM} (33) Gn

′n → Gn

′n

km k ∈ {1...JS},m ∈ {1..JM} (34)

This extension includes speciﬁc parameterizations such as deﬁning the Transition matrix as Tij = 1 + βkikj known from DeltaNet [Schlag et al., 2021]. For these, the chunk-wise parallelization of Yang et al. [2024] can be applied along multiple dimensions.

- A.6 Stability in the State Tracking Extension

The stabilization of Section A.5 can be applied in the state tracking extensions as well. In the extended case, the absolute values of Transitions are replaced by the spectral norm of the Transition matrix (Te′eij)ij deﬁned by its largest singular value. The stability can be ensured by limiting the sum of the largest singular values (instead of scalar entries) for the P-mode or zero matrix entries, resulting in a line graph tree with maximally unit norm matrices for tree Transitions in the D-mode. There are several options to limit or set the Transition matrices by/to one in magnitude. A straightforward option is to parametrize not the Transition entries Tij directly, but to parametrize its singular value decomposition, with orthogonal (or unitary) matrices T = UΣV T. The singular values Σ can be limited in magnitude by a tanh (for multiplicative decay) or ﬁxed to ±1 (long-range propagation). In the case of multiple directions (see 2D grid case), in the P-mode, they are multiplied by an additional softmax-limited propagation factor ("angle") - limiting the sum over the direction pairs in Equation 9. The orthogonal matrices U, V can be parametrized by the product of Householder matrices (generated from vectors), or the exponential of the Lie-group / generating group of special orthogonal matrices: the skew-symmetric matrices (directly parameterized). With these parametrizations, in turn depending on the network inputs (at nodes), state-tracking capabilities can be achieved while maintaining long-range stability [Merrill et al., 2024, Grazzi et al., 2025].

- A.7 Chunkwise-Parallel Form

Given the hierarchical structure of the parallelization shown in Sections A.2, A.3, and A.4, one can stop at a certain level of it. At this level, the resulting objects will again form a higher-level DAG or grid, which can now be processed sequentially, in the topological ordering of this DAG. This way, the quadratic complexity introduced by the parallelization is only introduced up to the optimal level of the hardware’s parallelization capabilities. While typically the FLOP-optimal point is at lower parallelization, Beck et al. [2025a] showed that a certain degree of parallelization is hardware optimal.

Multi-Directional Form For the 2D grid case of images, we use pLSTM in all four direction combinations of the 2D DAG at once in parallel, at each node going down/right, down/left, up/right, up-left. By pre-computing all of them in full parallelization, we can add up their gating matrices, resulting in just a single pass of the gated linear attention computation (see Equation 7). This fully parallelized form was used for the vision experiments.

- B Full Model Architecture

B.1 pLSTM for Vision

For the pLSTM-Vis, we use the ViT [Dosovitskiy et al., 2021] with pre-LayerNorm (actually RMSNorm) structure as backbone architecture, replacing the Multi-Head Attention with a pLSTM layer. Note that in addition to query-, key-, value-, and out-projections, we have linear layers for Source, Transition, and Mark gates in the four direction combinations. In addition, we apply a multi-head RMSNorm. Note that the Source, Mark, and Direct gates are sigmoid-activated, and the Transition is tanh-activated. For details, we refer to the attached source code.

- C Theoretical Analysis

- C.1 Exponential growth of path counts in a 2D grid

Assume a 2D grid structure of a DAG, then the number of paths between two nodes can be calculated in the following way. In total, the number of Transitions to the right and Transitions to the bottom are ﬁxed by the position offsets ∆x and ∆y between the nodes. In total, the number of paths can be counted by the number of orderings of a string consisting of ∆x times → and ∆y times ←. This results in the combinatorial factor, which is exponential in the path length ∆x + ∆y =∶ ∆:

[Figure 51]

∆x + ∆y ∆x+∆y

∆x + ∆y ∆x

∆x + ∆y ! ∆x!∆y!

∆x + ∆y 2π∆x∆y

Stirling≈

#Paths =

=

[Figure 52]

[Figure 53]

[Figure 54]

∆∆x x∆∆y y ∆x∶==β∆

[Figure 55]

β−β(1 − β)−(1−β) ≥1

1 2π∆β(1 − β)

∆

(35)

[Figure 56]

Given Transitions / forget gates potentially all at 1, this exponential number of paths leads to an exponentially growing magnitude of the Cell state C, as of Equation 7. For a non-linear MD-RNN, like DAG-LSTM, this path count applies as well, assuming that all forget gates are ﬁxed to one. Also in this case, the cell states accumulate all their ancestors’ values via all paths.

C.2 Long-Range Decay for P-mode in 2D

As the P-mode offers a directional propagation, here, we want to calculate the decay of one Source signal along the leading direction - given all Transitions are equal within the grid and at criticality:

α α

(1 − α) (1 − α) . This results in the following overall Transition from Source Sxy at x, y to Mark M(x+∆

- T =

x)(y+∆y) at x + ∆x, y + ∆y - with path length ∆ = ∆x + ∆y ≫ 1, direction β ≔ ∆

∆ :

x

[Figure 57]

∆ β∆ αβ∆(1 − α)(1−β)∆

α∆x(1 − α)∆y =

∆x + ∆y ∆x

Tr∆∗xTd∆∗y =

Txyfull(x+∆

x)(y+∆y) =

Paths

(36) Notice the binomial structure of the equation that ﬁxes the total sum ∑∆

x∈{0..∆} Txyfull(x+∆

x)(y+∆y)

to 1, so the total activation is conserved along the diagonal. Differentiating this equation by β, we can get the direction of largest propagation:

x)(y+∆y) = −∂β (log Γ(∆ + 1) + log Γ(β∆ + 1) + log Γ((1 − β)∆ + 1)) + ∆log(α) − ∆log(1 − α)

0 =! ∂β log Txyfull(x+∆

(37) Stirling≈ −∂β (β∆log(β) + (1 − β)∆log(1 − β)) + ∆log(α) − ∆log(1 − α)

α 1 − β β(1 − α)

= ∆log

[Figure 58]

This implies β = α for the direction of largest propagation. Now, inserting this direction into the Transition product:

2π∆ ∆e ∆ αα∆(1 − α)(1−α)∆ 2πα∆ αe∆ α∆ 2π(1 − α)∆ (1−eα)∆ (1−α)∆

[Figure 59]

∆ α∆ αα∆(1 − α)(1−α)∆ Stirling≈

x)(y+∆y)∣β=α =

[Figure 60]

Txyfull(x+∆

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

1 2πα(1 − α)∆

(38)

=

[Figure 67]

ofSo,theeven inpath length.the critical case of the P-mode, the signal is decaying, but only as a power law O(∆−1/2)

- D Experimental Results

Due to the complex structure of pLSTM in 2D and its parallelization, we use a jax 1-based implementation to enable efﬁcient compilation of the computational graph. In particular, we re-use parts of Park [2024]. Early experiments on using torch.compile on the torch 2 implementation showed a slowdown rather than a speed-up of the model computations, which is why we use jax. Our source code, released with this work, has a detached conﬁguration that works for both jax and torch and should enable a fast switch of frameworks for future changes. We use jax version 0.4.32 and CUDA 12.2.

- D.1 Initialization

While for the ablations and arrow pointing extrapolation, we use zeros initialization for the weight terms of Source, Mark, Transition, Direct as well as Orientation (P mode angle) layers, on the DeiT-III style training on ImageNet1k, using a non-zero small normal init leads to signiﬁcantly better results reported here. For ViT, we observe that LayerScale initialized with 1e-4 is important for ImageNet1k training, whereas on Arrow Pointing Extrapolation, it is important to initialize the LayerScale at 1 to reach the observed performance.

Table 4: General pLSTM initialization settings Parameter Value

[Figure 68]

[Figure 69]

Source Bias Init -4 Mark Bias Init -4 Direct Bias Init -6 Transition Bias Init 1 Transition Scaling Factor 5 Orientation Bias Init Headwise Range in [-2, 2] Multi-Head RMSNorm ǫ 1e-5 Pooling Corner Patches Mode P+D (alternating)

- D.2 Arrow Pointing

Here, we train for 50 epochs with batch size 128, using learning rates [ 1e-4, 3e-4, 1e-3 ] and report the mean validation curves over ﬁve seeds at the best learning rate. For ViL, we include 1e-5 as the learning rate, as it fails to improve for higher learning rates. The validation datasets (standard and extrapolation) are generated from the same validation seed for all runs. We use a linear-warmup + cosine-decay schedule with one warmup epoch starting from zero and ending at 0.001 × peak_lr. The models take about one hour of training on a single NVIDIA H100-64GB.

[Figure 70]

- 1https://jax.dev
- 2https://pytorch.org

- D.2.1 Test results

Table 5: Test Results on Arrow Pointing Extrapolation (5 seeds, 90% conﬁdence interval)

[Figure 71]

Best LR Test Acc. Test Acc. (Ext.) Model

[Figure 72]

pLSTM 0.0001 0.972 ± 0.003 0.778 ± 0.031 pLSTM / (no posemb.) 0.0001 0.975 ± 0.003 0.769 ± 0.018 pLSTM / (P-mode) 0.0003 0.978 ± 0.005 0.746 ± 0.027 pLSTM / (D-mode) 0.0001 0.957 ± 0.002 0.828 ± 0.028 pLSTM / (STM bias-only) 0.0001 0.975 ± 0.003 0.784 ± 0.020 ViT 0.0003 0.915 ± 0.019 0.707 ± 0.014

[Figure 73]

- D.3 ImageNet-1k

For training on ImageNet-1k, we match the original training schedule of DeiT-III [Touvron et al., 2022]. EfﬁcientNets [Tan and Le, 2019] as a convolution-based baseline architecture are still the SOTA at these scales, but it is important to note that larger models were also trained at larger resolutions, whereas training resolution was not scaled with the models for all other models. EfﬁcientNet, Mamba2D, and 2DMamba are non-isotropic, in that their embedding dimensions are scaled up with depth. ViT, ViL, and pLSTM are non-hierarchical, as is the reported isotropic ConvNeXt. For ConvNeXt, isotropic models showed lower performance compared to non-isotropic ones. For pLSTM, a transition to non-isotropic model could therefore lead to performance gains as well.

All of the models are pre-trained for up to 24 hours on 4 nodes, with 4 NVIDIA H100-64GB GPUs each. ViT models are faster (about 30-50%), as our models do not yet utilize speciﬁc kernels. For counting FLOPs, we use fvcore3 on the PyTorch implementation. For arrow pointing extrapolation training, the Mamba2D and 2DMamba variants were about twice as slow as pLSTM despite utilizing custom kernels.

Table 6: ImageNet1k - DeiT-III style training parameters Parameter Value ( → Fine-Tuning) Image Resolution 192 × 192 → 224 × 224 Training Epochs 800 (T), 400 (S), 400 (B) → 20 Hidden Dimension 192 (T), 384 (S), 768 (B) Num Heads 3 (T), 6 (S), 12 (B) DropPath Rate 0. (T), 0.05 (S), 0.1 (B) LayerScale Warmup Epochs 5 Peak Learning Rate 4e-3 (T), 4e-3 (S), 3e-3 (B) → 1e-5 Weight Decay 0.2 Gradient Clip Norm 1.0 Optimizer Lamb → AdamW Loss Type Binary Cross Entropy → Cross Entropy MixUp 0.8 CutMix 1.0 Label Smoothing 0.0 → 0.1 Global Batch Size 2048 → 512 ColorJitter 0.3 → 0.0 AutoAugment "3a" → "rand-m9-mstd0.5-inc1" RandomErasing 0. AugmentRepeats 3 → 1 TestCropRatio 1.0 RandomCrop rrc

[Figure 74]

[Figure 75]

[Figure 76]

3https://github.com/facebookresearch/fvcore

- D.3.1 Ablation Settings

For the ablation studies, we use a simpliﬁed training setting, without an additional ﬁne-tuning stage, resembling a DeiT-T training over 400 epochs [Touvron et al., 2021]. In Table 7, we provide a summary of the used training hyperparameters.

Table 7: ImageNet1k - DeiT style ablation training parameters Parameter Value Image Resolution 224 × 224 Training Epochs 400 Hidden Dimension 192 Num Heads 3 DropPath Rate 0. LayerScale Warmup Epochs 5 Peak Learning Rate 1e-3 Weight Decay 0.05 Gradient Clip Norm 1.0 Optimizer AdamW Loss Type Cross Entropy MixUp 0.8 CutMix 1.0 Label Smoothing 0.0 → 0.1 Global Batch Size 2048 ColorJitter 0.0 AutoAugment "rand-m9-mstd0.5-inc1" RandomErasing 0.25 AugmentRepeats 3 → 1 TestCropRatio 1.0 RandomCrop rrc

[Figure 77]

[Figure 78]

- D.4 Graph Benchmarks

For the graph version of pLSTM, we use torch, as the dynamic computation graph support is better for this framework. For simplicity, we also do not implement any parallelization of the graph computation, but use the recurrent form only.

Datasets. Our experiments are conducted with 10-fold cross-validation on popular TUDatasets [Morris et al., 2020]. For each fold, we use 1/10 of the respective training data as test set, 1/10 for validation and 8/10 for training. For every fold, we pick the test accuracy of the epoch with the best validation accuracy and report the average over all folds. For pLSTM, we encode the number of neighbors for each node with a standard positional encoding [Vaswani et al., 2017]. Otherwise, no data transformation, augmentation, or normalization is used.

Training procedure. We train all models for 100 epochs with AdamW, a learning rate of 0.001, batch size of 64, and a cosine decay learning rate schedule with 5 warmup epochs.

Models. The compared models are GAT [Velickoviˇ c´ et al., 2018], GCN [Kipf and Welling, 2017], GIN [Xu et al., 2019], LSTM GNN [Liang et al., 2016], MPNN [Gilmer et al., 2017]. All GNNs consist of an encoder, decoder, and 4 or 8 message passing layers. The best model conﬁguration is selected based on the validation accuracy. pLSTM also consists of the same encoder and decoder, but has 2 layers that operate in D-mode and 2 layers that operate in P-mode. To achieve an approximately similar parameter count, we ﬁx the hidden dimension of pLSTM to 96, while the other GNNs have a hidden dimension of 128. All trained models have a parameter count of < 300,000.

