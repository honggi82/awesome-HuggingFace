## Atlas: Multi-Scale Attention Improves Long Context Image Modeling

# arXiv:2503.12355v1[cs.CV]16Mar2025

Kumar Krishna Agrawal*1† Long Lian*1 Longchao Liu1 Natalia Harguindeguy12 Boyi Li1 Alexander Bick3 Maggie Chung2 Trevor Darrell1 Adam Yala12

[Figure 1]

### Abstract

- (a) Training efficiency on HR-IN100
- (b)

[Figure 2]

[Figure 3]

Efficiently modeling massive images is a longstanding challenge in machine learning. To this end, we introduce Multi-Scale Attention (MSA). MSA relies on two key ideas, (i) multi-scale representations (ii) bi-directional cross-scale communication. MSA creates O(logN) scales to represent the image across progressively coarser features and leverages cross-attention to propagate information across scales. We then introduce Atlas, a novel neural network architecture based on MSA. We demonstrate that Atlas significantly improves the compute-performance tradeoff of long-context image modeling in a high-resolution variant of ImageNet 100. At 1024px resolution, Atlas-B achieves 91.04% accuracy, comparable to ConvNext-B (91.92%) while being 4.3x faster. Atlas is 2.95x faster and 7.38% better than FasterViT, 2.25x faster and 4.96% better than LongViT. In comparisons against MambaVision-S, we find Atlas-S achieves 5%, 16% and 32% higher accuracy at 1024px, 2048px and 4096px respectively, while obtaining similar runtimes. Code for reproducing our experiments and pretrained models is available at https://github.com/yalalab/atlas.

|[Figure 4]<br><br>VisionTransformer<br><br>ConvNext<br><br>FasterViT<br><br>Atlas (ours)<br><br>MambaVision<br><br>Swin LongViT|
|---|

| | |
|---|---|
| | |

2500

Traintime(in8xH100hours)

2000

7.6x

1500

1000

OOM

1.8x

500

0

1024 2048 3072 4096

Input resolution (in px)

###### Atlas vs MambaVision gap on HR-IN100

40

32.84%

###### Δtop-1accuracy(in%)

30

16.5%

20

10

3.62%

1024 4096

2048

Input resolution (in px)

Figure 1. (a) Training efficiency comparison of different vision architectures on HR-IN100 across increasing input resolutions (1024-4096px). (b) Atlas exhibits similar runtime scaling as MambaVision while obtaining significantly better accuracy.

### 1. Introduction

Long-context image modeling remains a fundamental challenge in computer vision with broad applications to biomedicine (Xu et al., 2024), satellite imagery (Rad, 2024), and vision-language modeling (Gemini-Team et al., 2023; Wang et al., 2024; Qwen-Team, 2025; Chen et al., 2024). At the core of this challenge is a compute expressivity trade-off; we aim to develop models that efficiently scale to massive input sequences while capturing arbitrary pair-wise depen-

dencies between input tokens. As shown in Figure 1(a), self-attention, as used in Vision Transformers, is highly expressive, but its computational cost scales poorly (i.e., quadratically) with sequence length. It remains infeasible to train end-to-end Vision Transformers on massive imaging modalities such as mammograms or whole-slide pathology images. At another end of the spectrum, state space models (SSMs) and recurrent architectures are highly efficient, achieving linear computational complexity; however, SSM-based models perform poorly in long-context imaging modeling (Figure 1b).

*Equal contribution, †Project lead 1University of California, Berkeley 2University of California San Francisco 3Vanderbilt University . Correspondence to: Kumar Krishna Agrawal <kagrawal@berkeley.edu>, Adam Yala <yala@berkeley.edu>.

Long-context image modeling requires novel neural primitives and new benchmarks to guide their development. Recent work in efficient architecture design, such as FasterViT (Hatamizadeh et al., 2023) and MambaVision (Hatamizadeh & Kautz, 2024), has primarily focused on improving the throughput vs accuracy trade-offs in the context of standard resolution ImageNet experiments (224 × 224). While valuable, this setting yields little insight into how methods scale to larger input resolutions. To this end, we propose a new high-resolution benchmark based on ImageNet-100 (HR-IN 100). We evaluate the speed vs accuracy trade-off of different neural networks at progressively larger resolutions, ranging from 1024 × 1024 to 4096 × 4096 images. As input resolution increases, long-range communication across distant parts of the image becomes more essential for image classification, and asymptotic computational complexity begins to dominate model runtime.

In designing a novel neural primitive, we aim to enable arbitrary cross-token interaction with minimal intermediate steps (i.e., communication complexity) while minimizing computational complexity as a function of input sequence length. To this end, we propose Multiscale Attention (MSA), a novel primitive for high-resolution image modeling. MSA is built on two key ideas: multiscale representations and cross-scale communication. In each MSA block, we leverage a simple S-token max-pooling kernel to summarize small spatial regions (e.g., 4x4 input region), into progressively coarser summary representations across O(logS N) spatial scales, where N is the total sequence length. We then leverage a windowed cross-attention mechanism to enable information-sharing between tokens at different scales. At each scale, tokens attend to nearby tokens of the same scale and tokens from all coarser scales. This “top-down” communication enables MSA to integrate information across the entire sequence. Each scale’s tokens also cross-attend to its “parent” finer-grain scale tokens, allowing each coarse token to refine its representation through “bottom-up” communication. Altogether, this bi-directional communication pattern enables information mixing between all input tokens through O(log N) intermediate tokens (i.e. coarse scale representations) and within O(N log N) runtime. In controlled block-level experiments (see Table 3), we find that MSA outperforms alternative neural network primitives in long-context modeling, including LongNet’s dilated attention (Ding et al., 2023), MambaVision Mixer (Hatamizadeh & Kautz, 2024), and FasterViT’s Hierarchical Attention (Hatamizadeh et al., 2023).

We propose Atlas, a novel architecture designed around the unique advantages of MSA. Given a sequence length N, which defines log N scales within MSA, Atlas leverages log N macro-stages to progressively down-sample the input until MSA recovers only a single scale. We leverage the rich scale-2 representations of our MSA block as a

down-sampling mechanism, enabling both faster and more performant down-sampling than traditional approaches. We demonstrate that Atlas significantly improves the Pareto frontier in long-context modeling. In 1024 × 1024 experiments, as shown in Table 1, Atlas obtains comparable runtime to MambaVision (23.1hr vs 22.6hr) on the same hardware, while obtaining 6.1% higher accuracy (91.04 vs 84.86). Compared to FasterViT and LongViT, Atlas is 2.95× and 2.25× faster, obtaining 7.38% (91.04 vs 83.66) and 4.96% (91.04 vs 86.08) higher accuracy, respectively. Moreover, the performance advantage of Atlas is especially pronounced as we scale input resolution to 4096px, achieving a 34% accuracy improvement over MambaVision at similar runtime.

We summarize our contributions as follows:

- • We propose a High-Res ImageNet-100 (HR-IN 100), an efficient benchmark with input resolutions ranging from 1024 × 1024 to 4096 × 4096 for evaluating the frontier of long-context image modeling.
- • We introduce Multi-Scale Attention (MSA), a novel neural network primitive that maintains representations across O(log N) spatial scales and enables bidirectional information mixing across all scales within O(N log N) runtime. Building on MSA, we introduce Atlas, a novel neural network architecture.
- • With extensive experiments on High-Res ImageNet100, we demonstrate that Atlas improves the Pareto frontier in long-context image modeling. Atlas outperforms representative efficient architectures in longcontext image modeling, including FasterViT, MambaVision, and LongViT.

### 2. Related Work

Vision Transformers (ViTs). ViTs (Dosovitskiy, 2020) directly apply Transformers (Vaswani, 2017) architecture to image patches, demonstrating the effectiveness of selfattention in visual data processing. Building on this, DeiT (Touvron et al., 2021) improves training data efficiency. However, the self-attention primitive in ViT, which scales quadratically with input sequence length, limits its application toward high-resolution imaging. Our study focuses on developing efficient alternatives to standard self-attention to enable expressive and computationally efficient longcontext image modeling.

Efficient Long Sequence Modeling in Language. To address the challenges of long-sequence modeling, LongNet (Ding et al., 2023) introduces a dilated attention mechanism, allowing transformers to process sequences with up to one million tokens. LongNet was later adapted into a vision model as LongViT (Wang et al., 2023) to process whole-slide pathology images. State Space Models (SSMs), such as Mamba (Gu & Dao, 2023), provide a linear-time

[Figure 5]

patchify initialize multiscale

| |
|---|

scale1

MultiScale Attention

MultiScale Attention

scale2

pool

readout

DS DS DS

Dog

LayerNorm

LayerNorm

conv

conv

(256 x 768)

× d1 = 2

× d2 = 10

(4096 x 768)

[scale2]

[scale1, scale2]

64 x 64 x 768

1024 x 1024 x 3

- Figure 2. The Atlas architecture consists of a convolutional stem for initial feature extraction, followed by a series of Multi-Scale Attention (MSA) blocks that progressively downsample the feature maps while preserving global context. This hierarchical design facilitates the effective processing of high-resolution images with efficient communication between features.

alternative to full attention for efficient long sequence modeling. RetNet (Sun et al., 2023) combines the strengths of recurrence and attention, enabling linear-time sequence modeling. Longformer (Beltagy et al., 2020) integrated local and global attention for effective long-document processing. Our work is most similar to LongNet, which also achieves a communication complexity of O(log N), where N is the length of the input sequence.

a new form of attention and Pyramid Vision Transformer (PVT) (Wang et al., 2021) explores hierarchical attention for efficient modeling. Unlike these works, which focus on improving compute-accuracy trade-offs in modest resolution regimes (i.e. 224 x 224 pixels), our work focuses on modeling high-resolution images. In this context, we find that representative efficient architectures, including MambaVision and FasterViT, fail to effectively process high-resolution images.

Instead of using dilated attention, we propose multiscale attention (MSA), which captures distant dependencies by attending to a subset of the input through intermediate “coarser scale” tokens. Unlike dilated attention, MSA effectively leverages locality in the input, resulting in significantly improved long-context vision modeling.

Multi-resolution representations in Neural Networks. Dense cross-scale communication has been explored in the context of CNNs, as in DenseNets (Huang et al., 2017) and Feature Pyramid Networks (FPNs) (Lin et al., 2017). In these works, feature maps across multiple resolutions are integrated using fixed operations, including concatenation or summation. In contrast, we propose to fuse representations across scales in our MSA block through cross-attention. This data-dependent multi-scale integration strategy allows our model to learn complex interactions between features at different resolutions and optimize the fusion process jointly with feature extraction.

Efficient Visual Modeling. Vim and VMamba (Zhu et al., 2024; Liu et al., 2024) adapted State-space models (SSMs) to vision-specific tasks and demonstrated the effectiveness of SSMs for visual representation learning. MambaVision (Hatamizadeh & Kautz, 2024) proposed a hybrid SSM and self-attention-based architecture, and demonstrated improved performance over other SSM-based architectures. Swin (Liu et al., 2021) proposes leveraging window-shifting for cross-window communication and a hierarchical design to aggregate context. CSwin (Dong et al., 2022) proposes cross-shaped window attention to capture global and local dependencies. CrossViT (Chen et al., 2021a) uses a dualbranch architecture to process image patches of varying sizes. EdgeViT (Pan et al., 2022) and EfficientFormer (Li

### 3. Method

We propose Multi-Scale attention (MSA), a novel neural primitive for long-context image modeling. MSA builds representations across multiple spatial scales and leverages dense cross-attention operations to share information across scales. Building on this primitive, we build Atlas, a hierarchical macro-architecture that uses the intermediate scales in MSA as a novel down-sampling mechanism.

- et al., 2022) designed lightweight transformers that are specially optimized for edge devices. VisFormer (Chen et al., 2021b) combined convolutions and transformers for vision tasks. Twins (Chu et al., 2021) improved the spatial attention mechanisms for improved performance. The long-short transformer (Zhu et al., 2021) introduced hybrid attention for efficient modeling in vision and language. FasterViT (Hatamizadeh et al., 2023) introduced hierarchical attention for fast visual information processing, and demonstrated improved performance over Swin, Twins, CrossViT, and EfficientFormer. Focal Transformer (Yang et al., 2021) explores

#### 3.1. Preliminaries

Windowed self-attention (WA) adapts the standard MultiHead self-attention (MHSA) to operate efficiently on local regions of an input feature map. To lay the groundwork for multi-scale attention (MSA), we first describe the WA operation and analyze its computational benefits and limitations.

#### Windowed Self-attention. Consider a feature map X ∈

scale-1 scale-2 scale-3

scale-1 scale-2 scale-3

| |
|---|

| |
|---|

[Figure 6]

| |
|---|

| |
|---|

pool

pool

|| |
|---|
| |
|---|---|
| | |

|| |
|---|
| |
|---|---|
| | |

K V

K V

| |
|---|

K V

| |
|---|

K V

| |
|---|

| |
|---|

Q

Q

Q

Q

Cross-Attention

Cross-Attention Cross-Attention Cross-Attention

Self-Attention

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|| |
|---|
<br><br>| |
|---|
| |
|---|---|
| | |

|| |
|---|
<br><br>| |
|---|
| |
|---|---|
| | |

top-down multi-scale communication bottom-up multi-scale communication

- Figure 3. Illustration of top-down and bottom-up hierarchical communication in Multi-Scale Attention (MSA). The top-down Global Context Aggregation enables coarse-to-fine feature propagation. The bottom-up fine-to-coarse pathway propagates high resolution features into coarser scale representations.

RH×H×C, where H is the spatial and C is the channel dimension1. The WA mechanism operates in two key steps:

- 1. Window Partitioning: Divide the feature map into non-overlapping windows of size k × k, with the number of windows per dimension: H′ = H/k, total number of windows M = H′ × H′ = (H/k)2, and each window Wij (i,j ∈ {1,...,H′}) containing k2 tokens. Further, each window Wij is reshaped into a sequence, where

Wij ∈ Rk×k×C is viewed as Wij ∈ Rk

2×C after reshape.

- 2. Local Self-Attention: Apply standard Multi-Head SelfAttention (MHSA) within each window:

QKT √dk

Attention(Q,K,V ) = softmax

V

where Q,K,V = Linear Projections(Wij)

The computational complexity of WA within a single window is O(k2 · k2) = O(k4) due to the attention operation within k2 tokens. Since there are M = (H/k) × (H/k) = H2/k2 windows, the total complexity of WA across the entire feature map becomes O(M · k4) = O(H

2

k2 · k4) = O(H2k2) = O(Nk2), where N = H2 is the total number of tokens in the feature map. This is a significant reduction from the O(N2) complexity of global attention, especially when k ≪

√

N.

While computationally efficient, WA suffers from two key limitations: 1) Limited Receptive Field: Each window processes information independently, preventing direct communication between different image regions, and 2) Boundary Effects: Objects or features spanning multiple windows can-

1For simplicity, we focus on square 2D feature maps, but the concept is generalizable to 1D sequences or 3D volumes.

not be directly modeled within a single attention operation. For example, an object split across two windows must be processed independently in each window, relationships between parts can only be learned indirectly when all features merged at the final readout.

#### 3.2. Multi-Scale Attention

MSA’s core design centers on two key components: 1) a hierarchical representation that creates intermediate feature scales using fixed-size summarization kernels to preserve information density and 2) bi-directional communication that enables effective information exchange across multiple windows and scales, through dense cross-attention.

3.2.1. HIERARCHICAL REPRESENTATION

Multi-Scale Attention (MSA) builds hierarchical representations through iterative summarization with a fixed-size kernel of S-tokens. Starting with the input feature map F(1) at scale-1, we create coarser representations through a summarization operation S:

F(l) = S(F(l−1),S), for l = 2,...,L (1)

- scle1
- scle2
- scle3

window

pool

Figure 4. Multi-Scale features with iterative summarization.

Algorithm 2 Atlas Architecture Pseudocode

Algorithm 1 Multi-Scale Attention (MSA) Block

Input: Img : (B, Hin, Win, Cin), k × k ← window size P ← Patch Size, S ← Downsampling Rate D ← {d1, d2, ..., dL} ▷ Atlas Configuration

Input: X = [X(1), ..., X(L)], where X(l) : (B, Nl, Cin)

k × k ← Window Size, S ← Downsampling Rate Output: X = [X(1), ..., X(L)], where X(l) : (B, Nl, Cin)

Output: predictions : (B, Dout) ▷ downstream predictions

▷ Iterative summarization

- 1: X(1) ← ConvPatchify(Img, P) ▷ scale 1 feature map

▷ Initialize Multi-Scale features

- 2: for l = 2, ..., L do
- 3: X(l) ← Summarize(X(l−1), S) ▷ Strided MaxPool
- 4: end for

▷ Progressive Downsampling

- 5: for s = 1, 2, ..., L do ▷ Iterate through stages
- 6: for blk = 1, 2, ..., ds do
- 7: [X(s), ..., X(L)]← MSABlock([X(s), ..., X(L)], k, S)
- 8: ▷ Apply MSA Block
- 9: end for
- 10: end for
- 11: predictions ← readout(X(L))
- 12: return predictions

- 1: for l = 2, ..., L do ▷ Iterate from fine to coarse
- 2: X(l) += Summarize(X(l−1), S) ▷ Equation (1)
- 3: end for

▷ Top-Down Communication: Global Context Aggregation

- 4: for l = L, L − 1, ..., 1 do ▷ Iterate from coarse to fine
- 5: X(l) ← CrossAttention(X(l), [X(l), X(l+1), ..., X(L)])

▷ as in Equation (2)

- 6: end for

▷ Bottom-Up Communication: Fine-to-Coarse Refinement

- 7: for l = 2, 3, ..., L do ▷ Iterate from fine to coarse
- 8: X(l) ← CrossAttention(X(l), X(l−1))

▷ as in Equation (3)

- 9: end for
- 10: return X = [X(1), ..., X(L)]

where S is implemented as strided max-pooling with a fixed stride s (i.e. downsampling rate S = s × s tokens). This process continues until the feature map size at scale L is no larger than the window size k × k. With input sequence length N and downsampling rate S, the number of scales L grows logarithmically as O(logS N), where S = s2.

At each scale l, we operate on windows, by partitioning the feature map F(l) into non-overlapping regions of size k × k

(i.e. K = k2 tokens), yielding windows {Wij(l)}, for l = 1,..,L scales. As shown in Figure 4, this scheme creates a

directed acyclic graph (DAG) between windows at different spatial scales. With every summarization operation, we merge “parent” windows into new coarser “child” windows.

- 3.2.2. CROSS-SCALE COMMUNICATION: ATTENTION-BASED FUSION

The expressive power of MSA comes from its ability to efficiently propagate information across scales through two complementary mechanisms:

#### I. Top-Down Communication

In our top-down communication scheme, we propagate information from coarse “child” windows to their “parent” windows through a dense set of cross-attention operations.

Let W(l) be a window at scale l, and {Wl+1,...,WL} denote the corresponding coarse ”child” windows as illustrated in Figure 4. The cross-attention operation using standard Multi-Head Attention (MHA), as visualized in Figure 3, is

then given by:

W(l) = MHA(Ql,[Kl:L],[Vl:L]) (2)

where Ql,Kl,Vl are query/key/value projections of W(l), and Kl+1:L,Vl+1:L are concatenated key/value projections from coarser scales. This operation enables MSA to model relationships between tokens within the window, while also allowing each window to read from long-context information from all coarser scale ”child” windows. This dense cross-attention design allows each scale to directly observe global context through the coarsest scale ”child” window. At the coarsest scale l = L, this operation recovers standard self-attention.

#### II. Bottom-Up Communication

The bottom-up communication in MSA complements the top-down aggregation by refining coarser-scale ”child” representations with detailed information from finer-grain ”parent” tokens. This is a localized operation, in the sense that the fine grain refinement for each token is guided only by its direct parent window.

Specifically, let Ql be the query projection of Wl, and let Kl−1 and Vl−1 be the key and value projections from the parent window W(l−1). The updated window representation W(l) after bottom-up communication is obtained through cross-attention as:

W(l) = MHA(Ql,Kl−1,Vl−1) (3)

This targeted cross-attention allows for the recovery and integration of crucial local information potentially lost in the initial summarization.

The pseudocode for the full MSA block is shown in Algorithm 1.

#### Asymptotic Complexity. With a feature map X ∈ RN×C

and window of K tokens (typically K = k × k), downsampling rate S, MSA creates L = logS N feature scales. The most expensive operation is the dense top-down crossattention. In particular, for scale-1, each token cross attends to LK tokens (one window per scale), which scales to NLK complexity across a N-length sequence. The runtime for all subsequent scales 2,..,L is upper-bounded by NLK, giving an effective runtime complexity of O(NLK).

Plugging in L = logS N, we recover O(NK logS N) as the net runtime complexity of Atlas. Note that K and S are typically small constants depending on the hardware; in our experiments we find K = 256 (i.e. 16 × 16 window) and S = 16 (i.e. 4 × 4) to be most performant on an 8×H100 node. Our dense cross-scale communication strategy guarantees that each token must propagate information across at most O(logS N) intermediate tokens to interact with any another token in the sequence, where standard self-attention would obtain O(1) communication complexity but O(N2) runtime complexity.

#### 3.3. Atlas

The MSA block can be used as drop-in replacement for the standard MHA block in existing architectures like ViT (Dosovitskiy, 2020) or Swin Transformer (Liu et al., 2021). To fully leverage the benefits of MSA, we co-design the network structure for Atlas to optimize performance and efficiency. Our full architecture is illustrated in Figure 2, with the pseudocode in Algorithm 2.

Atlas is a multi-stage architecture, with a convolutional stem (Hatamizadeh et al., 2023; Hatamizadeh & Kautz, 2024; Xiao et al., 2021), followed by multiple stages of MSA blocks. We leverage the same convolutional stem as FasterViT (Hatamizadeh et al., 2023) to obtain localized patch-level representations. In particular, the stem utilizes two stages of residual convolutional blocks, yielding in feature map of RH/16×W/16×C. Given this feature map, fixed window size K and downsampling rate S, MSA builds a multi-scale layout with L = logS N scales, as outlined in Section 3.2.1.

As part of the co-design of Atlas, we fix the number of stages of MSA blocks to be identical to the number of scales, i.e. L = logS N. The key insight behind Atlas is to progressively reduce the number of tokens at each scale, focusing computational resources on high-level features. Given the multiscale structure of the MSA block, we propose a progressive scale-dropping strategy in Atlas. In other words, for a multi-scale input X = [X(1),X(2),...,X(L)], stage l of the Atlas only processes [X(l),X(l+1),...,X(L)] actively.

As a concrete instance, for MSA with L scales, let us define an Atlas config D = {d1,d2,...,dL} with L corresponding stages. Here, dl is the number of blocks at stage l. For example, for a 4-scale MSA block, an Atlas config would have 4 stages, e.g. D = {2,2,2,6}. This config indicates that the first scale is the finest resolution for the first two blocks, after which it becomes inactive and is dropped. Subsequently, the second scale becomes the finest active resolution for the next two blocks, with X(4) being the only active features for the last block.

This strategy is quite flexible, in that for a single scale MSA block, and K = N, it recovers the standard ViT with MHSA block. For the readout, there are multiple strategies to aggregate the final representations across scales. We find that simply using the last scale as the final representation works well in practice.

### 4. Experiments

#### 4.1. Image Classification

Setup. We propose using a novel high-resolution benchmark based on Imagenet-100 (Tian et al., 2020), High-Resolution ImageNet-100. The dataset extends the original Imagenet1k dataset with ∼126K unique training samples, 5000 validation samples, and 100 classes with high-resolution images (up to 4096px), where the images are upsampled to the desired resolution. We first focus on a system’s level comparison against representative architectures, including ViT, Swin, FasterViT, MambaVision, ConvNext, and LongViT. Together, these architectures encompass sparse attention, SSMs, convolutional, and dilated attention approaches. For each baseline , we utilize the provided code as is, without modifications to gradient accumulation, employing a linearly decaying learning rate proportional to the batch size, following (Goyal, 2017). This ensures consistency with prior work and facilitates a fair comparison.

Comparing Architectures: We benchmark all architectures on the same hardware, 1 server with 8×H100 Nvidia GPUs using 1024px input resolution (equivalent to 4K tokens with patch-size=16). To understand the runtime-performance tradeoff of Atlas design against existing architectures, we train Base-scale models (i.e. 12 head, 768 embed-dim following prior work (Dosovitskiy, 2020)) for 320 epochs.

Long-Context Image Modeling: To understand the efficacy of Atlas in long-context image modeling tasks, we seek to scale the evaluation to higher resolutions. Due to extreme cost of running our baselines for full convergence runs (320 epochs) at Base models, we focus our scaling experiments on only our two fastest models, namely Atlas and MambaVision in Small regime. As shown in Figure 1, all other architectures are significantly slower at higher resolutions. Prior work in architecture design for vision models (Xiao

et al., 2021) demonstrates meaningful comparisons with shorter training schedules. We adopt a similar approach and train Atlas-S and MambaVision-S models for 100 epochs for 1024px, 2048px and 4096px, scaling upto 64K tokens.

#### 4.2. Ablations

Attention Mechanism. To understand the efficacy of different token-mixing (e.g. attention or SSM-based) blocks in long-context image modeling, we conduct controlled experiments, using the same optimizer, learning rate schedules. We consider 384 × 384 inputs, with 4 × 4 patches, giving a sequence length N = 9216. We use 4-block architectures, with Base-equivalent blocks (i.e. 12 head, 768 embed-dim following prior work (Dosovitskiy, 2020). We compare our MSA block with Hierarchical Attention block (Hatamizadeh

- et al., 2023), MambaVision Mixer (Hatamizadeh & Kautz,
- 2024), dilated attention with the LongViT block (Ding et al.,

2023) and standard ViT, Window-ViT blocks.

Communication Mechanism. Our proposed Multi-Scale Attention (MSA) block relies on bi-directional communication to effectively model long-context. To understand the contribution of each mechanism, we conduct controlled ablations with 256 × 256 inputs, using 4 × 4 patches, giving a sequence length N = 4096, K=256 (i.e. 16 × 16 windows), S=16 (i.e. 4 × 4 strided max-pool). In this setting we have features at two scales, providing a sandbox to test the contribution of different communication mechanisms. We use a Small-scale 4-block architecture (i.e. with 6 heads, 384 dim following (Dosovitskiy, 2020)). The predictions from both scales are merged via average pool, before readout. In this setting, we compare the following variants of the block

- • no-multiscale : equivalent to vanilla single-scale WA
- • no communication: equivalent to WA at both scales.
- • top-down only: propagates from coarse to fine only
- • bottom-up only: propagates from fine to coarse only
- • top-down + bottom-up: both mechanisms as in MSA.

Composition Strategies. To identify the best MSA block composition strategy, we compare three different strategies of incorporating MSA

- • stack: vanilla stacking of blocks as in (Dosovitskiy, 2020), with averaging tokens across scales for readout.
- • convolutional downsampling : similar to prior work as in (Liu et al., 2021; Hatamizadeh et al., 2023) we use separate downsampling layer to reduce spatial resolution by 2 × 2 per stage. For this variant, we use a uniform 4-stage architecture, i.e. {3,3,3,3}
- • Atlas: a {d1=2, d2=10} config outlined in Section 3.3

We run each ablations with 512 × 512 inputs, using 8 × 8 patches, giving a sequence length N = 4096, with 12 Smallscale MSA blocks.

### 5. Results

#### 5.1. Image Classification

Comparing Architectures at 1024px resolution: The experimental results in Table 1 demonstrate that Atlas-B/16 is competitive with/outperforms existing vision backbones in accuracy, while being computationally efficient. In particular, Atlas achieves 91.04% accuracy while delivering substantial speed advantages: 4.3× faster than ConvNext-B (91.92%), 1.15× faster than ViT (90.66%), and 1.6× faster than Swin (90.89%) with competitive accuracy. Compared to other sparse-transformer backbones, Atlas is 2.95x faster and 7.3% better than FasterViT, 2.25x faster and 4.96% better than LongViT. Notably, while the runtimes are comparable, Atlas is 6.05% better than MambaVision. Additional experimental results from our 50-epoch runs are available in the supplementary material (Table 6).

Table 1. Comparison of vision backbones on 1024x1024 image resolution on the HR-IN100 benchmark. Each model is evaluated on runtime (in hours), relative speed compared to Atlas, and Top-1 accuracy (in %). All models are base scale and were trained for 320 epochs until convergence on single 8 × H100 GPU node.

Architecture Runtime Relative Top-1 Acc.

(hr) ↓ speedup ↓ (%) ↑ Transformer

ViT-B 26.77 1.15x 90.66 Swin-B 37.25 1.6x 90.89 FasterViT-4 68.31 2.9× 83.66 LongViT-B 52.23 2.2× 86.08

Convolutional ConvNext-B 100.11 4.3× 91.92 Mamba MambaVision-B 22.69 0.98× 84.86 Multi-Scale Atlas-B 23.12 1.00× 91.04

Long-Context Image Modeling: The results in Table 2 demonstrate the superior scaling capabilities of Atlas over MambaVision on high-resolution images. While both architectures show comparable runtime efficiency on a single 8×H100 node (MambaVision requiring 4.56, 14.73, and 55.5 hours for 1024px, 2048px, and 4096px respectively), Atlas-S/16 outperforms MambaVision-S/16 by 3.62% at 1024px resolution (81.82% vs. 78.82%), with this gap widening to 16.50% at 2048px and 32.84% at 4096px. These results highlight Atlas’s capability to effectively capture long-range dependencies at extreme context lengths up to 64K tokens where state-space based models struggle.

#### 5.2. Ablations

Attention Mechanism. To understand the efficacy of the MSA block, we run controlled ablations against existing primitives for long-context modelling. The results in Table 3 highlight the effectiveness of MSA for classification. While faster in runtime, the window-attention blocks of WViT and

- Table 2. Comparison of Mamba-based (MambaVision-S/16) and Multi-Scale Attention (Atlas-S/16) models across three image resolutions: 1024px, 2048px, and 4096px. The table presents both computational efficiency (runtime in hours on single 8xH100 node) and performance (Top-1 accuracy in %) metrics. All models were trained for 100 epochs per resolution. Atlas-S/16 demonstrates superior accuracy across all resolutions, with particularly significant advantages at higher resolutions (2048px and 4096px), while maintaining comparable computational demands. The substantial increase in runtime as resolution scales highlights the computational challenges inherent in high-resolution image processing.

Model Runtime (hr) ↓ Top-1 Accuracy (%) ↑

1024px 2048px 4096px 1024px 2048px 4096px

Mamba-Based MambaVision-S/16 4.56 14.73 55.5 78.2 51.42 23.36 Multi-Scale Attention Atlas-S/16 3.64 14.23 54.72 81.82 67.92 55.84

Swin perform ∼29% worse than MSA. MambaVisionMixer (Hatamizadeh & Kautz, 2024) performs ∼12% worse than MSA while requiring 0.88x the runtime. MSA outperforms the standard attention-block of ViT and the Hierarchichal Attention block from (Hatamizadeh et al., 2023), both in runtime and accuracy. MSA is 1.76× faster and ∼10% better than ViT block, while being 1.15× faster and 27% better than Hierarchical Attention.

- Table 3. Comparing different attention mechanisms at a block-level in controlled setting (100epoch runs).

Table 4. Ablations on the communication strategies.

###### Communication Strategy Top-1 (%) ↑

single-scale only (WindowViT) 59.39 multi-scale only 65.14 multi-scale + bottom-up 69.92 multi-scale + top-down 69.04 multi-scale + bottom-up + top-down (MSA) 72.02

pose MSA blocks into an efficient macro-architecture. As shown in Table 5, stacking MSA blocks without progressive downsampling resulted in an accuracy of 69.88% at a runtime of 75 minutes. Convolutional downsampling between MSA blocks accelerated training, with a runtime of 40 minutes; however, this led to a significant performance drop, with accuracy decreasing to 56.14%. The Atlas-specific D2D10 configuration, which progressively processes lowerresolution scales, emerged as the most effective strategy, achieving the highest accuracy of 70.09% at a runtime of 38 minutes. Our novel composition strategy is both led to faster runtimes than traditional convolutional downsampling while yielding comparable performance to no downsampling.

Block Runtime Relative Top-1 acc.

(min) ↓ speedup ↓ (in %)↑

Window ViT 55 0.60× 41.65 ShiftedWindow ViT (Swin) 68 0.75× 41.48 ViT 160 1.76× 60.57

Hierarchical Attn. (FasterViT) 105 1.15× 43.19 Dilated Attn. (LongViT) 218 2.39× 49.88 MambaVisionMixer 80 0.88× 58.79 Multi-Scale Attn. (Atlas) 91 1.00× 70.81

Finally, the MSA block is 2.39x faster and 20.9% better than Dilated Attention block from LongViT (Ding et al., 2023). Our results suggest that the MSA block can be used as dropin replacement to existing primitives, offering significant improvements for long-context modeling.

Table 5. Comparison of different composition strategies.

###### Composition Runtime (m) ↓ Top-1 (%) ↑

Stack 75m 69.88 Downsample (Conv) 40m 56.14 Atlas (D2D10) (ours) 38m 70.09

Communication Mechanism. The MSA block develops a bi-directional communication to efficiently model longcontext modeling. The results in Table 4 demonstrate that MSA significantly improves on vanilla Window-Self Attention (WA), improving classification accuracy by ∼12.5% (72.02 vs 59.39). Furthermore, we show that relying only on multi-scale features with WA is suboptimal, resulting in a

### 6. Conclusion

We propose Multiscale Attention (MSA), a novel primitive for long-context image modeling. In a controlled blocklevel experiment, we demonstrated that MSA significantly outperformed alternative cross-token communication strategies, including FasterVIT’s Hierarchical Attention block, and MambaVision Mixer. MSA achieves this performance through two key insights: multi-scale representations and bidirectional cross-scale communication. Building on rich multi-scale representations introduced by MSA, we propose

- 6.9% performance drop. While the top-down and bottom-up communication mechanisms, independently boost the accuracy by ∼3.5% each, they are complementary to each other. Using the bi-directional communication strategy (i.e. MSA) improves ∼3% over relying on only one of the mechanisms. Composition Strategies. Next, we studied how to best com-

Atlas, a novel neural network architecture for long context modeling. In system-level experiments, we find that Atlas significantly improves accuracy-runtime trade-offs in efficient long-context modeling, achieving massive gains over FasterViT, MambaVision, ConvNext, Swin and LongViT. Overall, these results demonstrate that multi-scale attention significantly improves long-context image modeling.

### 7. Acknowledgements

We thank the UCSF Facility of Advanced Computing team, including Hunter McCallum, Sandeep Giri, Rhett Hillary, Marissa Jules, Sean Locke, and John Gallias, for their work in supporting our computational environment. This was supported by a grant from EvansMDS, a funding initiative of the Edward P. Evans Foundation. Research reported in this publication was also supported by the National Cancer Institute of the National Institutes of Health under Award Number R37CA289821. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

### References

Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Chen, C.-F. R., Fan, Q., and Panda, R. Crossvit: Crossattention multi-scale vision transformer for image classification. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 357–366, 2021a.

Chen, Z., Xie, L., Niu, J., Liu, X., Wei, L., and Tian, Q. Visformer: The vision-friendly transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 589–598, 2021b.

Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

Chu, X., Tian, Z., Wang, Y., Zhang, B., Ren, H., Wei, X., Xia, H., and Shen, C. Twins: Revisiting the design of spatial attention in vision transformers. Advances in neural information processing systems, 34:9355–9366, 2021.

Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N., and Wei, F. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv preprint arXiv:2307.02486, 2023.

Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., and Guo, B. Cswin transformer: A general

vision transformer backbone with cross-shaped windows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12124–12134, 2022.

Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Gemini-Team, Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Goyal, P. Accurate, large minibatch sg d: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Hatamizadeh, A. and Kautz, J. Mambavision: A hybrid mamba-transformer vision backbone. arXiv preprint arXiv:2407.08083, 2024.

Hatamizadeh, A., Heinrich, G., Yin, H., Tao, A., Alvarez, J. M., Kautz, J., and Molchanov, P. Fastervit: Fast vision transformers with hierarchical attention. arXiv preprint arXiv:2306.06189, 2023.

Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700–4708, 2017.

Li, Y., Yuan, G., Wen, Y., Hu, J., Evangelidis, G., Tulyakov, S., Wang, Y., and Ren, J. Efficientformer: Vision transformers at mobilenet speed. Advances in Neural Information Processing Systems, 35:12934–12949, 2022.

Lin, T.-Y., Doll´ar, P., Girshick, R., He, K., Hariharan, B., and Belongie, S. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2117–2125, 2017.

- Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., and Liu, Y. Vmamba: Visual state space model, 2024. URL https://arxiv.org/ abs/2401.10166.
- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021.

Pan, J., Bulat, A., Tan, F., Zhu, X., Dudziak, L., Li, H., Tzimiropoulos, G., and Martinez, B. Edgevits: Competing light-weight cnns on mobile devices with vision transformers. In European Conference on Computer Vision, pp. 294–311. Springer, 2022.

Qwen-Team. Qwen2.5-vl, January 2025. URL https: //qwenlm.github.io/blog/qwen2.5-vl/.

Rad, R. Vision transformer for multispectral satellite imagery: Advancing landcover classification. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 8176–8183, January 2024.

Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.

Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16, pp. 776–794. Springer, 2020.

Xu, H., Xu, Q., Cong, F., Kang, J., Han, C., Liu, Z., Madabhushi, A., and Lu, C. Vision transformers for computational histopathology. IEEE Reviews in Biomedical Engineering, 17:63–79, 2024.

Yang, J., Li, C., Zhang, P., Dai, X., Xiao, B., Yuan, L., and Gao, J. Focal self-attention for local-global interactions in vision transformers. arXiv preprint arXiv:2107.00641, 2021.

Zhu, C., Ping, W., Xiao, C., Shoeybi, M., Goldstein, T., Anandkumar, A., and Catanzaro, B. Long-short transformer: Efficient transformers for language and vision. Advances in neural information processing systems, 34: 17723–17736, 2021.

Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., and Wang, X. Vision mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint arXiv:2401.09417, 2024.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and J´egou, H. Training data-efficient image transformers & distillation through attention. In International conference on machine learning, pp. 10347–10357. PMLR, 2021.

Vaswani, A. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., and Lin, J. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Wang, W., Xie, E., Li, X., Fan, D.-P., Song, K., Liang, D., Lu, T., Luo, P., and Shao, L. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 568–578, 2021.

Wang, W., Ma, S., Xu, H., Usuyama, N., Ding, J., Poon, H., and Wei, F. When an image is worth 1,024 x 1,024 words: A case study in computational pathology. arXiv preprint arXiv:2312.03558, 2023.

Xiao, T., Singh, M., Mintun, E., Darrell, T., Doll´ar, P., and Girshick, R. Early convolutions help transformers see better. Advances in neural information processing systems, 34:30392–30400, 2021.

### A. Implementation Details

For the baselines that we compared with in Table 1, we utilize the provided code as is, without modifications to gradient accumulation, employing a linearly decaying learning rate proportional to the batch size, following (Goyal, 2017). This ensures consistency with prior work and facilitates a fair comparison of performance. For the various model hyperparamters, we use the configs as provided by authors for Imagenet-1K where available.

in long sequences where cross-scale attention operations are frequent.

### B. Additional Experiments

To validate our findings, we conducted 50-epoch training runs following prior work showing that shorter training schedules still provide reliable signals about architectural performance (Xiao et al., 2021). These abbreviated runs maintain the same relative performance trends across architectures while requiring significantly less computational resources. As shown in Table 1, Atlas-B/16 maintains its superior accuracy-runtime trade-off across resolutions, achieving high accuracy while maintaining reasonable training times even at 2048px resolution, where several competing architectures exceed our 24-hour runtime limit.

### C. Additional Optimizations: QKV Caching for Multi-Scale Attention

A naive implementation of Multi-Scale Attention (MSA) would require recomputing Query, Key, and Value (QKV) projections for each window involved in cross-attention operations across different scales. Consider a window W(l) at scale l performing cross-attention with windows at coarser scales {W(l+1),...,W(L)} in the top-down pathway. In a naive implementation, the QKV for each window W(l) would be recalculated for every cross-attention instance, even if the underlying feature representation of W(l) remains unchanged. This repeated computation becomes increasingly inefficient as the number of scales and windows grows.

To overcome this challenge, we introduce a QKV cache mechanism within MSA. During both the top-down and bottom-up pathways, we maintain a cache at each scale l to

store the QKV projections for all windows {Wij(l)}. When a window at scale l needs to perform cross-attention, it first

queries this cache. If a valid QKV set for the current version of W(l) is available, it is directly retrieved from the cache. The cache is updated only when the feature representation of a window at a given scale is modified. This occurs after self-attention at the coarsest scale L, and after each dense cross-attention operation in the top-down and parent cross-attention in the bottom-up pathways. By reusing QKV projections, our cache signficantly accelerates MSA

Table 6. Comparison of vision models across different image resolutions. Each model has two rows: one for runtime (in minutes) and one for Top-1 accuracy (in %). We trained all models for 50 epochs for each resolution. We limited each experiment to a maximum runtime of 24hrs on an 8 × H100 GPU node and report “–” for experiments that could not be complete within our runtime limit.

###### Model Runtime (min) ↓ Top-1 Accuracy (%) ↑

256px 512px 1024px 2048px 256px 512px 1024px 2048px Transformer-Based

ViT-B/16 18 51 247 3480 63.68 72.60 69.42 – WViT-B/16 18 44 137 638 64.21 68.95 63.61 53.93

Convolutional ConvNext-B/16 66 237 955 3825 78.84 75.94 67.50 – Sparse-Transformer

FasterViT-4 49 168 675 2400 77.64 74.40 53.62 – LongViT-B/16 39 116 442 2000 55.20 51.88 45.32 –

Mamba-Based MambaVision-B/16 21 56 197 750 73.10 69.94 51.68 24.64 Multi-Scale Attention Atlas-B/16 25 54 198 786 80.05 83.75 82.73 74.74

