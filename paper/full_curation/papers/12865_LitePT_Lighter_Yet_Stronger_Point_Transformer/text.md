# arXiv:2512.13689v2[cs.CV]30Mar2026

## LitePT: Lighter Yet Stronger Point Transformer

Yuanwen Yue1,2 Damien Robert3 Jianyuan Wang2 Sunghwan Hong1 Jan Dirk Wegner3 Christian Rupprecht2 Konrad Schindler1

1ETH Zurich 2University of Oxford 3University of Zurich

[Figure 1]

[Figure 2]

Figure 1. LitePT is a lightweight, high-performance 3D point cloud architecture. Left: LitePT-S has 3.6× fewer parameters, 2× faster runtime and 2× lower memory footprint than the state-of-the-art Point Transformer V3, and is even more memory-efficient than classical convolutional backbones. Moreover, it remains fast and memory-efficient even when scaled up to 86M parameters (LitePT-L). Right: Already the smallest variant, LitePT-S, matches or outperforms state-of-the-art point cloud backbones across a range of benchmarks.

### Abstract

Modern neural architectures for 3D point cloud processing contain both convolutional layers and attention blocks, but the best way to assemble them remains unclear. We analyse the role of different computational blocks in 3D point cloud networks and find an intuitive behaviour: convolution is adequate to extract low-level geometry at highresolution in early layers, where attention is expensive without bringing any benefits; attention captures high-level semantics and context in low-resolution, deep layers more efficiently, where convolution inflates the parameter count. Guided by this design principle, we propose a new, improved 3D point cloud backbone that employs convolutions in early stages and switches to attention for deeper layers. To avoid the loss of spatial layout information when discarding redundant convolution layers, we introduce a novel, parameter-free 3D positional encoding, PointROPE. The resulting LitePT model has 3.6× fewer parameters, runs

2× faster, and uses 2× less memory than the state-of-the-art Point Transformer V3, but nonetheless matches or outperforms it on a range of tasks and datasets. Code and models are available at: https://github.com/prs-eth/LitePT.

### 1. Introduction

Visual understanding of 3D point clouds is central to a wide range of applications, including robotics [5, 90, 92, 102], autonomous driving [22, 72], localisation [48], mapping [55, 79, 81], and environmental monitoring [35, 68]. A variety of deep learning architectures and neural processing layers for unstructured point clouds have been proposed, yet the field still lacks a detailed understanding of their relative strengths and weaknesses, and principled guidelines on how to most efficiently combine them into versatile, highperformance architectures.

Lately, Transformer-based models have dominated 3D benchmarks. In particular, their most recent incarnation

Point Transformer V3 (PTv3) [88] has been shown to outperform earlier sparse convolutional [12, 23] and attentionbased models [26, 87, 103], and is considered the state of the art. Importantly, PTv3 is in fact not a pure Transformer architecture: 67% of its parameters are allocated to (residual) sparse convolution layers. These are interleaved with the Transformer-style attention+MLP blocks and, among others, serve as a form of positional encoding. That design, with both convolution and attention operations at all hierarchy levels (resp., depths) of a U-net-like encoder-decoder scheme [61], is common in modern 3D point cloud architectures, which naturally leads to the question: what are the respective roles of convolution and attention?

Here, we analyse the contribution and interplay of these layers in more detail. We find a clear division of labour along the feature hierarchy. Early, high-resolution stages are dominated by the encoding of local geometry. Convolution or attention perform similarly well for that purpose, as the locality of convolutions is the right inductive bias. However, attention is substantially more expensive for early layers with high spatial resolutions (i.e., a large number of tokens). Later, at lower-resolution stages, semantics and global context emerge. To capture the associated longrange interactions, the highly expressive attention mechanism is more suitable and also more parameter-efficient. As mentioned, in PTv3 and related architectures, the SparseConv [23] layer was primarily included to encode positional information. It turns out that, for that particular purpose, convolution is a possible solution, but not a necessity. We find that a ROPE-inspired [71] query-key positional encoding, which we call PointROPE, fulfills the role more effectively, while being more efficient and introducing no learnable parameters. Overall, our analysis points to a clear design principle: apply convolution when the focus is on local geometry, and attention when reasoning about semantics and global layout.

Building on these insights, we design LitePT, a hybrid network architecture for 3D point cloud analysis that leverages the computational tools in the most efficient manner; i.e., sparse convolutions in the early stages and PointROPEenhanced attention in the later stages. By tailoring the information processing to the level of abstraction, LitePT requires 3.6× fewer parameters than PTv3. Our architecture cuts memory consumption by 60.3% during training and by 51.2% during inference, and reduces latency by 34.5% during training and by 58.8% during inference. Remarkably, LitePT also improves performance compared to PTv3 across a range of benchmarks on 3D semantic segmentation, 3D instance segmentation, and 3D object detection.

### 2. Related Work

In line with the purpose of LitePT, we review deep learningbased point cloud representations, with a specific focus on

Transformer architectures and hybrid approaches.

Deep Point Cloud Understanding. To take advantage of mature image-based networks, early approaches used to project 3D point clouds into 2D image planes and then leverage standard 2D CNNs to extract features [4, 9, 38, 42, 70, 83]. These projection-based methods tend to work well only when several implicit assumptions are met, e.g., relatively uniform point density, sufficient coverage, opaque surfaces, etc. Voxel-based methods transform irregular point clouds to regular voxel grids and then apply 3D convolution operations [27, 34, 45, 50, 69]. However, voxel representations are both computationally expensive and memoryintensive, motivating follow-up works to develop efficient sparse convolution frameworks [10, 12, 23, 54, 74]. Instead of projecting or quantising irregular point clouds into regular grids in 2D or 3D, point-based methods design operators that work directly on raw point coordinates, better preserving geometric information. Point operators have progressed from early MLP-based designs [18, 49, 56–58, 100] to point convolutions [1, 24, 33, 43, 75, 85, 93], graph-based networks [41, 82], and, more recently, attention-based mechanisms [7, 26, 59, 60, 77, 87, 88, 103]. Among modern point cloud backbones, Transformer-based architectures represent the state of the art.

Point Cloud Transformers. Transformer-based architectures employ the attention mechanism as their core feature extractor. To mitigate the quadratic complexity of global self-attention, most approaches adopt some form of windowed attention, restricted to a local spatial neighbourhood. Point cloud Transformers mainly differ in how these localised attention patches are constructed to best balance performance and efficiency. Common strategies include k-nearest neighbour search [87, 98, 103], window or voxel partitioning [21, 46, 53, 73, 80, 96, 97, 101], superpoints [59, 60], and 1D sorting with space-filling curves [8, 88]. Such local attention mechanisms are often integrated with shifted patch grouping [97] and hierarchical architectures in the spirit of U-Net [61], so as to aggregate global context. Existing works typically apply attention at all stages of the hierarchical network. We argue that attention in shallow stages, where the number of tokens is large and local patterns dominate, is computationally inefficient and unnecessary, as seen in Secs. 3.1 and 4.1.

Positional Encoding in Point Cloud Transformers. Attention does not take into account spatial layout; therefore, positional encoding plays an important role in Transformers. PTv1 [103] and PTv2 [87] employ relative positional encoding (RPE), where an MLP encodes relative positions between points. Stratified Transformer [39] and Swin3D [97] use contextual relative positional encoding (cRPE), which maintains three learnable look-up tables for the (x,y,z) axes that are computationally rather inefficient. OctFormer [80] and PTv3 [88] employ conditional posi-

tional encoding (CPE) [13], which is implemented via a convolutional layer preceding each attention module. CPE improves efficiency, but introduces a substantial number of learnable parameters. Here, we adapt rotary positional embedding (RoPE) [71] to point cloud learning, a parameterfree module that offers both efficiency and strong empirical performance.

Hybrid Models. Convolution is by design capable of capturing local features, whereas Transformers excel at modelling long-range dependencies. In the vision domain, since the introduction of the Vision Transformer [19], numerous studies have explored the integration of convolutional operators with attention for efficient image analysis [15, 25, 51, 78, 84, 94]. Similarly, in the 3D point cloud field, several works have investigated hybrid architectures that combine the strengths of convolution and attention. DyCo3D [30] augments Sparse U-Net with a bottleneck Transformer to capture long-range context. Stratified Transformer [39] reports that a KPConv [75] block provides substantially stronger local features than attention. Superpoint Transformer [59] leverages a lightweight PointNet [56] to encode geometrically-homogeneous superpoints. PointConvFormer [86] and KPConvX [76] augment convolution kernels with attention to improve feature modelling. ConDaFormer [20] adds two sparse convolution blocks before and after each attention module to better capture local structure. We note that PTv3 [88] is also arguably a hybrid model, as it utilizes sparse convolutions as positional encoding, which account for the majority of its trainable parameters. Another common design paradigm applies a sparse U-Net for feature extraction followed by a taskspecific Transformer decoder [63, 95]. In contrast to prior works, which typically employ a uniform hybrid block repeated across the hierarchy, we rethink hybrid design from a multi-scale perspective and decouple convolution and attention, allowing for the selective use of each at different hierarchy levels to exploit their complementary advantages.

### 3. Methodology

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

#### + + +

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

[Figure 27]

[Figure 28]

Linear Convolution

Normalization

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

+ Sum

Activation Self-Attention

[Figure 35]

[Figure 36]

- Figure 2. PTv3 block. The block is composed of a convolutional conditional positional encoding module followed by an attention module.

To motivate our network design, we begin with an empirical study that investigates the respective roles of convo-

lution and attention in PTv3 [88]. We then introduce the components of LitePT: computational blocks that are reduced to the essentials and tailored to different processing stages (Sec. 3.2); and an alternative, learning-free positional encoding for the simplified blocks (Sec. 3.3). Finally, we describe the overall architecture in Sec. 3.4.

##### 3.1. Revisiting PTv3: Convolution vs. Attention

Preliminaries. PTv3 [88] represents the current state-ofthe-art architecture for point cloud understanding. Similar to earlier point cloud backbones [12, 59, 87, 103], it adopts a U-Net architecture [61] composed of multiple encoder and decoder stages with skip connections. Between consecutive encoding (or decoding) stages, pooling (or unpooling) operations are applied to downsample (or upsample) the point cloud and its associated features. Each encoder and decoder stage consists of several blocks. Fig. 2 depicts a single block as used in PTv3, consisting of a convolutional positional encoding module and an attention module . Inspired by [13], PTv3 adopts conditional positional encoding, implemented by prepending a sparse convolution layer, a linear projection, and a LayerNorm, with a skip connection, before each attention module. The attention module follows a standard pre-norm structure [91], where self-attention is applied between local groups of points obtained via serialisation sorting, followed by a multilayer perceptron (MLP).

Conditional positional encoding, and in particular its sparse convolution layer, has proved to be an important part of the overall architecture, but its precise role remains somewhat unclear. Does it indeed just serve to encode the spatial layout of the tokens that flow through the attention layer, or does it actually act as a local feature extractor in the spirit of classical convolutional networks? In the following, we analyse the parameter efficiency and the computational cost of different components along the U-Net hierarchy, revealing striking differences between the stages.

ScanNet [14] nuScenes [6] Model #Params mIoU mIoU PTv3 [88] 46.1M 77.5 80.4

- ⃝1 PTv3 w/o Transformer 32.4M 73.4 76.1
- ⃝2 PTv3 w/o SPConv 15.4M 70.7 74.9

Table 1. Revisiting PTv3. We evaluate two PTv3 variants: in ⃝1 , the attention and MLP modules are removed, and in ⃝2 , only the sparse convolution layers are removed.

Number of parameters. An often overlooked, yet important fact is that 67% of the total parameter budget in PTv3 is spent on the sparse convolution layers of the positional encoding, while the Transformer part (i.e., attention and MLP) only accounts for 30% of the learnable parameters. Furthermore, the parameter count of the sparse convolution layers increases substantially with depth and is largest near the bottleneck, due to the high feature dimension of the late

3/28/26, 2:29 AM Treemap

###### Upload CSV and visualize

Choose File No file chosen Choose File No file chosen Show Parameter Show Latency Hide Legend Download SVG Download PDF Rectangle Height: 100 Figure Width: 1500 Corner Radius: 8

Convolution Attention Pooling Unpooling Serialisation PointROPE

E0 E1 E2 E3 E4 D3 D2 D1 D0

PTv3

3/28/26, 2:31 AM Treemap

###### Upload CSV and visualize

LitePT-S

Choose File No file chosen Choose File No file chosen Show Parameter Show Latency Hide Legend Download SVG Download PDF Rectangle Height: 100 Figure Width: 1500 Corner Radius: 8

(a) Breakdown of trainable parameters

E0 E1 E2 E3 E4 D3 D2 D1 D0

PTv3

LitePT-S

(b) Breakdown of latencies

- Figure 3. Parameter count and latency. E0-E4 denote encoder stages from shallow to deep, and D3-D0 denote decoder stages from deep to shallow. The length of each bar reflects the relative parameter count or latency of the corresponding module. Top: In PTv3, the positional encoding implemented via a convolution block accounts for the majority of its parameters, particularly in the later stages. In contrast, our PointROPE is parameter-free. Bottom: The PTv3 latency map reveals the significant cost of early-stage attention. LitePT restricts attention to late stages, where it is most effective and less costly.

encoder and early decoder stages. See Fig. 3a.

Latency. Fig. 3b graphically depicts the computational latency of attention and convolution across different network stages. Attention, with its quadratic computational complexity, accounts for the majority of the computational cost. Importantly, that cost decreases as one progresses towards deeper stages near the bottleneck, because hierarchical downsampling quadratically reduces the number of point tokens.

Convolution vs. attention. So far, we have clarified that convolution accounts for the majority of trainable parameters, whereas attention dominates the computational cost, and that both vary strongly along the U-Net hierarchy. To separate the contributions of the two modules, we design two reduced variants of the PTv3 block. In the first one, we remove the attention modules. Using exclusively this variant degenerates to a classical sparse U-Net struc-

PTv3

w/oSPConv PTv3PTv3

w/oTransformer

Stage 0 Stage 1 Stage 2 Stage 3 Stage 4

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

[Figure 51]

- Figure 4. Representations learnt by the hierarchical U-Net encoder. The hierarchical U-Net encoder exhibits an operatoragnostic feature hierarchy: shallow stages consistently encode local geometric structure, while semantics emerge in deeper stages.

ture [12, 23]. In the second variant, we remove only the sparse convolution layer to obtain a “pure” Transformer. Table 1 contrasts the semantic segmentation performance of the two variants for ScanNet [14] and nuScenes [6]. It turns out that removing convolutions causes a larger performance drop than removing the attention modules, suggesting that the “positional encoding” actually does much of the heavy lifting. We visualise the learnt embeddings at each encoding stage for the three variants using PCA (Fig. 4) and find that a distinct division of labour emerges along the hierarchy, regardless of whether convolution, attention, or both are used. Early stages primarily encode local geometry, later stages capture high-level semantics.

127.0.0.1:5500/treemap_v7_yuanwen.html 1/1

127.0.0.1:5500/treemap_v7_yuanwen.html 1/1

Discussion. The above analysis leads us to the following hypotheses:

- 1. It may not be necessary to use both convolution and attention at every stage. In the early stages, which prioritise local feature extraction, convolution is adequate. In deep stages, where the focus is on long-range context and semantic concepts, attention is key.
- 2. It would be a sweet spot in terms of efficiency if one could indeed avoid attention at early stages, where it is most expensive, and convolution at late stages, where it inflates the parameter count.
- 3. Pure attention blocks will require an alternative positional encoding—but storing spatial layout is apparently not the main function of the convolution, so a more parameter-efficient replacement should be possible.

##### 3.2. Tailored Blocks for Different Network Stages

Driven by the insights from the study described above, we propose a simple yet effective design that retains only the essential operations in each stage. Convolutions are allo-

E0 E2 E3 E4 D2 D1 D0

E1 D3

| |×2<br><br>[Figure 52]<br><br>[Figure 53]| |×2<br><br>[Figure 54]<br><br>[Figure 55]<br><br>|[Figure 56]<br><br>|×2<br><br>[Figure 57]<br><br>|[Figure 58]<br><br>[Figure 59]<br><br>|×6<br><br>[Figure 60]<br><br>[Figure 61]|[Figure 62]<br><br>[Figure 63]<br><br>|×2<br><br>[Figure 64]<br><br>[Figure 65]|[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]|
|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 70]|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|[Figure 75]|[Figure 76]<br><br>[Figure 77]|[Figure 78]<br><br>[Figure 79]|[Figure 80]<br><br>[Figure 81]|[Figure 82]<br><br>[Figure 83]|[Figure 84]<br><br>[Figure 85]<br><br>[Figure 86]<br><br>[Figure 87]|

[Figure 88]

[Figure 89]

[Figure 90]

Convolution block Attention block Pooling block Unpooling block

Stem block

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

PointROPE

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Serialisation

[Figure 101]

- Figure 5. LitePT-S architecture. Our model comprises five stages, employing convolution blocks in the early stages and PointROPE augmented attention blocks in the later ones. LitePT-S uses a lightweight decoder. Alternatively, adding convolution or attention blocks symmetrically in the decoder produces LitePT-S*.

##### 3.3. Point Rotary Positional Embedding

cated to earlier stages with high spatial resolution and low channel depth, and attention is reserved for deep stages with only few, but high-dimensional tokens.

Discarding the expensive convolution layer at deep hierarchy levels has an undesired side effect: one loses the positional encoding. Hence, a more parameter-efficient replacement is needed.

Formally, let the hierarchical encoder consist of L stages, where the i-th stage transforms the feature representation fi−1 into fi via a function Bi(·):

Rotary Positional Embedding (RoPE) [71] has proven to be remarkably effective in natural language processing. In RoPE, relative positional awareness is introduced into the attention mechanism through rotations of the feature space. Originally, the method is designed for 1D sequence data. It does not have a direct generalisation to irregular point clouds in 3D point space.

fi = Bi(fi−1), i = 1,...,L (1)

Depending on the stage index, each block Bi is instantiated as either pure convolution or pure attention:

ConvBlocki, if i ≤ Lc AttnBlocki, if i > Lc

(2)

Bi =

We adapt RoPE to 3D in a straightforward manner to obtain Point Rotary Positional Embedding (PointROPE). Given a point feature vector fi ∈ Rd at position pi = (xi,yi,zi), we divide the embedding dimension d into three equal subspaces corresponding to the x, y, and z axes:

Early stages (i≤Lc) operate on point sets with high spatial resolution and density, where local geometric reasoning is critical. Employing convolution layers in these stages efficiently aggregates information over local receptive fields, with minimal parameter overhead. As one progresses to deeper stages (i > Lc), the number of point tokens is greatly reduced and semantic abstraction becomes more important, hence one switches to attention-based blocks. Optionally, one can also include a “hand-over” stage i with both ConvBlocki and AttnBlocki. See ablation studies in Sec. 4.1. More gradual transitions between the two mechanisms are, in principle, possible, but unnecessarily complicate the design.

fi = [fix;fiy;fiz], fix,fiy,fiz ∈ Rd/3 . (3)

We then independently apply the standard 1D RoPE embedding to each subspace, using the respective point coordinate, and concatenate the axis-wise embeddings to form the final point representation:

  =

 

  . (4)

 

f˜ix f˜iy f˜iz

RoPE1D(fix,xi) RoPE1D(fiy,yi) RoPE1D(fiz,zi)

f˜i =

Our LitePT follows a different philosophy than PTv3 and other hybrid point cloud Transformers: [20, 76, 86] all uniformly repeat the same computational block at all stages; as a consequence, that unit must include both attention and convolution. In contrast, we prefer to simplify individual blocks as much as possible, which then requires different forms of simplification depending on the network stage. Empirically, we find that strategically distributing custom blocks along the hierarchy yields higher performance with significantly lower memory footprint and computational cost.

For each point with coordinates (xi,yi,zi), we directly use its grid coordinates as input, which are already correctly scaled during the pooling operation.

The embedding scheme preserves the directional separability of 3D points while jointly encoding the feature’s positional phase rotation, effectively capturing relative geometry. Compared to the learned convolutional positional encoding of PTv3 [88], PointROPE is parameter-free, lightweight, and, by construction, rotation-friendly. As part

of our open source code, we provide an optimised CUDA implementation.

##### 3.4. Architecture

Our model follows the conventional U-Net [61] structure, with five stages. We build three variants of the encoder, with varying number C of channels in each stage and B blocks per stage. Note that C must be divisible by 6 in stages that include PointROPE.

LitePT-S: C = (36,72,144,252,504), B = (2,2,2,6,2) LitePT-B: C=(54,108,216,432,576), B=(3,3,3,12,3) LitePT-L: C=(72,144,288,576,864), B=(3,3,3,12,3)

We use LitePT-S as the main variant for the experiments, since it already delivers excellent performance across all benchmarks. Model scaling is examined in Tab. 5. Per default, we set Lc = 3, meaning that stages 1, 2, 3 use ConvBlocki, while stages 4, 5 use AttnBlocki. Each ConvBlocki consists of a sparse convolution layer, a linear layer and LayerNorm, and has a residual connection. Each AttnBlocki consists of a PointROPE embedding followed by attention, where the latter is computed locally within groups of points, found with the same serialisation sorting as in PTv3 [88]. For semantic segmentation, we simplify the decoder to only the linear projection layer and LayerNorm in each stage. For instance segmentation, we apply the stage-specific design also in the decoder and symmetrically assign ConvBlocki and AttnBlocki, in reverse order of the encoder.

### 4. Experiments

Training Inference Method #Params Latency Memory Latency Memory

MinkUNet [12] 39.2M 60ms 1.9G 21ms 2.4G PTv2 [87] 12.8M 188ms 22.8G 151ms 22.9G PTv3 [88] 46.1M 110ms 5.8G 51ms 4.1G

LitePT-S (Ours) 12.7M 72ms 2.3G 21ms 2.0G LitePT-S* (Ours) 16.0M 81ms 3.3G 26ms 2.0G LitePT-B (Ours) 45.1M 93ms 5.5G 33ms 2.4G LitePT-L (Ours) 85.9M 97ms 8.4G 41ms 2.6G

- Table 2. Efficiency comparison. Results are reported as average over the full ScanNet dataset using a single RTX 4090 GPU. Automatic Mixed Precision (AMP) is enabled for all models during training and disabled during inference. * denotes our variant with a heavier decoder that includes attention or convolutional blocks.

We begin with a series of ablation studies to analyse different configurations of our hybrid design, the model’s scaling behaviour, and PointROPE (Sec. 4.1). We then present comparisons with state-of-the-art methods for 3D semantic segmentation (Sec. 4.2), 3D instance segmentation (Sec. 4.3) and 3D object detection (Sec. 4.4).

Figure 6. Performance-efficiency trade off. Left: Progressively dropping attention in more of the early stages. Right: Progressively dropping convolution in more of the late stages.

##### 4.1. Ablation Studies and Analysis

Are both convolution and attention needed at every stage? To verify our first hypothesis from Sec. 3.1, we design two sets of experiments on nuScenes. We begin with a baseline model that incorporates both convolution and PointROPE attention at all stages. In Experiment 1, we progressively remove attention, first from stage 0, then from stages 0 and 1, etc. In Experiment 2, we progressively remove convolution, first from stage 4, then from stages 4 and 3, etc. We then plot the mIoU of those configurations against latency (resp. parameter count).

As shown in Fig. 6 (left), removing attention in early stages boosts efficiency with almost no drop in mIoU, whereas removing attention in later stages harms performance. On the other hand, Fig. 6 (right) shows that removing convolution in later stages greatly reduces the parameter count with a negligible change in mIoU, whereas removing convolution in early stages only marginally improves efficiency but adversely affects performance. The analysis confirms that one needs not include both convolution and attention at every stage. Their contribution and their cost highly depend on the hierarchy level.

Where is the sweet spot in terms of efficiency and performance? To determine the optimal transition point Lc between convolution and attention, we conduct an ablation study on nuScenes as shown in Tab. 3. Optionally, we include a “hand-over” stage, denoted by “X”, that includes both convolution and attention. Setting Lc = 3, i.e., convolution in the first three stages and attention in the last two, achieves the best trade-off between parameter count, latency, and mIoU. We adopt Lc = 3 as our default setting for all experiments.

Decoder design. The mixed design with blocks tailored to the layer depth is always used in the U-Net encoder. On the contrary, we propose two design variants for the UNet decoder. In LitePT-S*, the same mixed design is used in the decoder, in reverse order. In LitePT-S, we further strip down the architecture and keep only a linear projection layer per stage (as needed to integrate skip connections), making the method even more efficient. We find empirically that the optimal choice is task-dependent, as shown in

Lc Setting #Params Latency mIoU

- 0 A-A-A-A-A 11.8M 35.1ms 82.1
- 1 C-A-A-A-A 11.9M 30.4ms 81.7
- 2 C-C-A-A-A 12.0M 25.8ms 82.0

- 3 C-C-C-A-A 12.7M 21.5ms 82.2

- 4 C-C-C-C-A 18.8M 16.2ms 80.9
- 5 C-C-C-C-C 26.9M 13.5ms 75.4

C-X-A-A-A 12.2M 30.9ms 81.9 C-C-X-A-A 13.2M 26.7ms 82.3 C-C-C-X-A 23.4M 24.9ms 82.4

- Table 3. Effect of Lc and “hand-over” stage. C: convolutional block; A: attention block; X: both convolution and attention are used at that stage. We compare model variants and report latency, memory usage, and validation mIoU on the nuScenes dataset. The grey-shaded row is our recommended setting.

Tab. 4. For semantic segmentation, the simple decoder is the best choice. For instance segmentation, the variant with convolution and attention blocks has a noticeable edge. We point out that even the slightly heavier LitePT-S* is still a lot more efficient than other Point Transformers (see Tab. 2), and leave the choice of decoder to the user.

Semantic Segmentation (mIoU) Instance Segmentation (mAP50) Decoder ScanNet [14] Structured3D [104] nuScenes [6] Waymo [72] ScanNet [14]

LitePT-S 76.5 83.7 82.2 73.1 62.2 LitePT-S* 76.8 83.0 81.8 72.7 64.9

- Table 4. Decoder design. We compare two decoder variants: in LitePT-S*, we apply our stage-tailored design symmetrically to the decoder stages, while in LitePT-S, we retain only linear projection layers in all decoder stages.

Model scaling. Due to the parameter-free PointROPE encoding, our model has substantially fewer trainable weights. This offers the possibility to repurpose the saved capacity and scale up LitePT. We assess scaling behaviour on Structured3D, the largest dataset in our evaluation suite. As shown in Tab. 5, the model scales favourably: increasing the model size from LitePT-S to LitePT-L continuously improves performance, with only a modest increase in testtime latency and memory usage. Notably, even LitePT-L, with a parameter count twice that of PTv3, still runs faster than PTv3 and has a lower memory footprint.

Method #Params Latency Memory mIoU PTv3 [88] 46.1M 57ms 5.83G 82.4 LitePT-S (Ours) 12.7M 23ms 2.56G 83.6 LitePT-B (Ours) 45.1M 36ms 2.60G 85.1 LitePT-L (Ours) 85.9M 44ms 3.58G 85.4

- Table 5. Model scaling on Structured3D dataset. Our model scales efficiently, achieving consistent performance gains from small to large variants with modest increases in latency and memory. Even when scaled to twice the parameters of PTv3, LitePT-L remains more efficient.

PointROPE. In Tab. 6 we ablate the effectiveness of the proposed PointROPE, on nuScenes. Removing PointROPE leads to a significant performance drop of 2.6 percentage

points in mIoU. We additionally ablate the base frequency d, which controls how fast each embedding dimension “rotates” as the position increases (uniformly for the three axes). PointROPE is fairly robust to the choice of frequency. Setting b = 100 yields the best score; we fix that value for all datasets to avoid excessive hyperparameter tuning.

w/o PointROPE b = 10 b = 100 b = 1000 b = 10000 mIoU 79.6 81.7 82.2 81.8 81.3

- Table 6. PointROPE. Dedicated positional encoding is neededdropping PointROPE leads to a significant performance drop. PointROPE works similarly well with a wide range of base frequencies, the grey-shaded column is our recommended setting. 4.2. Semantic Segmentation

nuScenes [6] Waymo [72] Method #Param mIoU mAcc mIoU mAcc MinkUNet [12] 39.2M 73.3 - 65.9 76.6 SPVNAS [74] - 77.4 - - Cylinder3D [105] - 76.1 - - AF2S3Net [11] - 62.2 - - SphereFormer [40] - 78.4 - 69.9 PTv2 [87] 12.8M 80.2 - 70.6 80.2 PTv3 [88] 46.1M 80.4 87.2 71.3 80.5

LitePT-S (Ours) 12.7M 82.2 88.1 73.1 83.8

- Table 7. Outdoor semantic segmentation on nuScenes and Waymo validation set. Scores of prior work courtesy of [88, 89].

Limited Scenes (Pct.) Limited Annotations (Pts.)

Method #Params Full 1% 5% 10% 20% 20 50 100 200 MinkUNet [12] 39.2M 72.2 26.0 47.8 56.7 62.9 41.9 53.9 62.2 65.5 PTv2 [87] 12.8M 75.4 24.8 48.1 59.8 66.3 58.4 66.1 70.3 71.2 PTv3 [88] 46.1M 77.5 25.8 48.9 61.0 67.0 60.1 67.9 71.4 72.7 LitePT-S (Ours) 12.7M 76.5 27.3 50.6 63.1 67.3 62.5 68.4 70.9 72.8 LitePT-S* (Ours) 16.0M 76.8 27.2 51.6 63.0 67.1 63.2 69.5 72.0 74.2

- Table 8. Indoor semantic segmentation on ScanNet validation set. In mean IoU. Scores of prior work courtesy of [88].

Val Test Method #Params mIoU mAcc mIoU mAcc

MinkUNet [12] 39.2M 76.4 84.3 77.4 85.5 PTv2 [87] 12.8M 79.0 86.8 78.5 86.6 PTv3 [88] 46.1M 82.4 90.3 82.1 90.3

LitePT-S (Ours) 12.7M 83.6 90.7 82.4 90.3

- Table 9. Indoor semantic segmentation on Structured3D.

Setup. We perform semantic segmentation for four different datasets. nuScenes [6] and Waymo [72] are two outdoor datasets of first-person driving scenes, captured with vehicle-mounted LiDAR. ScanNet [14] and Structured3D [104] show indoor settings. The former was captured using an RGB-D camera. It is relatively small by today’s standards, comprising 1,201 training scenes. Structured3D is a synthetic dataset and the largest public collection of 3D scenes with semantic annotations, and contains 18,348 training scenes. We follow PTv3 and use test time augmentation (TTA). Results without TTA can be found in the appendix.

Results. Tab. 7 reports semantic segmentation results on the nuScenes and Waymo validation sets. LitePT achieves marked improvements over competing architectures, in both cases +1.8 mIoU. We note that automotive LiDAR has different, more challenging properties compared with indoor datasets: the model must learn to handle massive differences in point density due to the large range, and highly anisotropic point distributions due to the scan line pattern and frequent specular reflections and ray drops.

Table 8 shows IoU scores for the ScanNet validation set. Following the literature [32], we also report results with limited training, obtained either by restricting the number of available training scenes or by reducing the number of annotated points per scene. The performance of LitePT is comparable to PTv3, which has ≈4× more parameters—in data-constrained settings, even slightly better—and clearly superior to PTv2, which has a similar parameter count. On the more than 10× larger Structured3D dataset, LitePT consistently outperforms all competing methods, including the much larger state-of-the-art PTv3.

##### 4.3. Instance Segmentation

ScanNet [14] ScanNet200 [62]

PointGroup [37] #Params mAP25 mAP50 mAP mAP25 mAP50 mAP MinkUNet [12] 39.2M 72.8 56.9 36.0 32.2 24.5 15.8 PTv2 [87] 12.8M 76.3 60.0 38.3 39.6 31.9 21.4 PTv3 [88] 46.2M 77.5 61.7 40.9 40.1 33.2 23.1 LitePT-S* (Ours) 16.0M 78.5 64.9 41.7 40.3 33.1 22.2

Table 10. Indoor instance segmentation on ScanNet and ScanNet200 validation set. Scores of prior work courtesy of [88].

Setup. We evaluate our method for instance segmentation on ScanNet [14] and ScanNet200 [62]. Following the protocol of prior work, we employ PointGroup [37] as instance segmentation head on top of the decoder to achieve a fair comparison.

Results. Tab. 10 summarise the results. On ScanNet, LitePT again outperforms all prior backbones and sets a new state of the art, with 64.9 mAP50, a +3.2 percentage point improvement over PTv3. On ScanNet200, which includes a long tail of rare categories, the results are comparable to PTv3 and significantly better than all previous methods. For example, our method achieves 1.2% higher mAP50 than PTv2, which has a similar parameter count, but 11× larger memory footprint and 6× longer runtime.

##### 4.4. Object Detection

Setup. We evaluate 3D object detection on Waymo. For a fair comparison with prior work [46, 88], we employ the same 3D object detection framework, CenterPointPillar [99]. Consistent with [21, 46, 88], we avoid spatial downsampling, thus turning LitePT into a single-stage network with 8 blocks, to allow detection of small objects. Ob-

Vehicle L2 Pedestrian L2 Cyclist L2 Mean L2 Method mAP APH mAP APH mAP APH mAPH

PointPillars [42] 63.6 63.1 62.8 50.3 61.9 59.9 57.8 CenterPoint [99] 66.7 66.2 68.3 62.6 68.7 67.6 65.5 SST [21] 64.8 64.4 71.7 63.0 68.0 66.9 64.8 SST-Center [21] 66.6 66.2 72.4 65.0 68.9 67.6 66.3 VoxSet [28] 66.0 65.6 72.5 65.4 69.0 67.7 66.2 PillarNet [65] 70.4 69.9 71.6 64.9 67.8 66.7 67.2 FlatFormer [46] 69.0 68.6 71.5 65.3 68.6 67.5 67.2 PTv3 [88] 71.2 70.8 76.3 70.4 71.5 70.4 70.5

LitePT (Ours) 71.6 71.2 76.1 70.1 71.8 70.7 70.7

Table 11. Outdoor object detection on Waymo with single frames input. Scores of prior work courtesy of [88].

jects are divided into two difficulty levels, and we report level-2 metrics.

Results. Tab. 11 reports scores based on single-scan LiDAR inputs. Also in this application, LitePT reaches the highest score overall and on two out of three object categories, and comfortably matches the performance of the closest competitor, PTv3.

### 5. Conclusion and Discussion

We have introduced LitePT, a lighter yet stronger point Transformer for various point cloud processing tasks. Our starting point was the question, which distinct roles and impacts different operators have along the processing hierarchy. Experiments confirm that (sparse) convolutions are adequate, and more efficient, at early hierarchy levels, whereas attention comes into its own at higher levels, where semantic abstraction and global context over a comparatively small token set are key. In itself, these observations are not unexpected, but surprisingly, they have not been leveraged in contemporary point cloud architectures. LitePT embodies the simple principle “convolutions for low-level geometry, attention for high-level relations” and strategically places only the required operations at each hierarchy level, avoiding wasted computations. To achieve this, we equip our method with parameter-free PointROPE positional encoding to compensate for the loss of spatial layout information that occurs when discarding convolutional layers. We hope that LitePT will be useful as a generic high-performance backbone for 3D point cloud processing, and that our analysis can serve as practical guidance for architecture design beyond our current version.

In our architecture, attention is applied only in the later stages, where the reduced token count is small. It would therefore be affordable to compute self-attention globally across all tokens, rather than locally. In future work, it may be interesting to eliminate the local grouping operation, which could on the one hand strengthen long-range context modelling, and on the other hand further reduce the computation time at inference.

Acknowledgments. Part of the compute is supported by the Swiss AI Initiative under project a144 and a154 on Alps. We thank Xiaoyang Wu, Liyan Chen and Liyuan Zhu for their help with the comparison to PTv3. The project is supported by the Circular Bio-based Europe Joint Undertaking and its members under Grant Agreement No 101157488. Embed2Scale is co-funded by the EU Horizon Europe program under Grant Agreement No 101131841. Additional funding for this project has been provided by the Swiss State Secretariat for Education, Research and Innovation (SERI) and UK Research and Innovation (UKRI).

### Appendix

In this Appendix, we provide detailed architecture of LitePT (Sec. A), detailed experimental settings (Sec. B), additional experiments (Sec. C), and visualization of LitePT’s predictions for 3D semantic segmentation, 3D instance segmentation, and 3D object detection (Sec. D).

### A. Detailed Architecture

Our full architecture is shown in Fig. 7. It follows U-Netstyle [61] encoder-decoder design with skip connections, and is organized into five stages. Adjacent encoder (or decoder) stages are connected via pooling (or unpooling) blocks. We apply our stage-tailored design on the encoder: the first three stages use convolution blocks, while the final two use attention blocks. For LitePT-S/B/L, each stage in the decoder contains only an unpooling block. For LitePTS*, we mirror the stage-tailored design in the decoder as well. Detailed architecture specifications can be found in Tab. 12. Below, we describe each block type in detail.

Attention block. Each attention block consists of a PointROPE attention module and a feed-forward network (FFN) module. Following the pre-norm [91] convention, a LayerNorm [2] is placed before both the attention and FFN modules. The FFN uses a hidden dimension four times larger than the channel dimension of its stage. We observe that adding an extra LayerNorm before the attention block further stabilizes the training. In the PointROPE attention module (Fig. 8), input point features are projected to query (Q), key (K), and value (V) representations. PointROPE is computed from point coordinates P and applied to Q and K, leaving V unchanged. The resulting “rotated” Q′ and K′ are fed into a standard scaled dot-product multi-head attention together with V, followed by a linear projection to produce the final output embeddings. Our PointROPE implementation is compatible with FlashAttention [16, 17, 64], which we use in our model. We apply PointROPE to locallyaggregated groups of 1024 points, formed using the same serialization sorting strategy as [88].

Convolution block. The convolution block includes of a

single sparse convolution layer [12, 23] with a kernel size of 3 × 3 × 3, followed by a linear projection layer and a LayerNorm [2] layer. A residual connection [29] links the block’s input and output.

Pooling and unpooling blocks. We adopt the grid pooling and unpooling operation from [87]. During pooling, points are divided into non-overlapping partitions. Point features are first projected by a linear layer, then points within the same partition are max-pooled, followed by a GELU [31] activation and a BatchNorm layer [36]. The pooling stride is set to 2 at each stage, reducing the spatial resolution by a factor of 2 each time. During unpooling, point features from the current decoder stage and the corresponding encoder stage are each passed through their own linear layer, GELU activation, and BatchNorm. The resulting features are then merged through a skip connection via summation.

### B. Detailed Experimental Settings

For indoor datasets, we use RGB and surface normals as input features. For outdoor datasets, where RGB and normal information are unavailable, we use xyz coordinates and intensity (plus elongation for object detection). Following common practice [12, 87, 88], we first downsample the point cloud on a grid. For 3D segmentation tasks, we set the grid size to 0.02m for indoor scenes and 0.05m for outdoor scenes. For 3D object detection, we adopt grid sizes of 0.32m in the xy plane and 6m along the z axis, consistent with [46, 88]. Detailed training configurations for semantic segmentation, instance segmentation and object detection are provided in Tab. 13, Tab. 14, and Tab. 15, respectively.

### C. Additional Experiments C.1. Further Ablation on PointROPE

Spherical vs. Cartesian coordinates. In PointROPE, we divide each point’s feature embedding into three equal subspaces and then apply the standard 1D ROPE [71] embedding to each subspace using the respective Cartesian coordinates. Here, we investigate an alternative design that uses spherical coordinates. Specifically, we transform each point (xi,yi,zi) into spherical coordinates (ri, θi, ϕi), using the mean of all points as the origin. We then apply 1D ROPE using ri, θi and ϕi separately and concatenate the resulting embeddings. The motivation is that spherical coordinates decouple radial distance and angular structure, which could potentially make positional relationships easier to learn. However, as shown in Tab. 16, we empirically find that PointROPE in spherical coordinates is effective but offers no improvement over Cartesian coordinates, while adding additional computational overhead. Therefore, we retain our simpler per-axis Cartesian design.

Subdivision of the input space. For each attention head (with head dimension 18), we split the embedding evenly

E0 E2 E3 E4 D2 D1 D0

E1 D3

|[Figure 102]|×2<br><br>[Figure 103]<br><br>|[Figure 104]<br><br>|×2<br><br>[Figure 105]<br><br>|[Figure 106]<br><br>|×2<br><br>[Figure 107]<br><br>|[Figure 108]<br><br>[Figure 109]<br><br>|×6<br><br>[Figure 110]<br><br>|[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]|×2<br><br>[Figure 114]<br><br>|[Figure 115]<br><br>[Figure 116]<br><br>[Figure 117]<br><br>[Figure 118]|
|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|[Figure 123]|[Figure 124]|[Figure 125]<br><br>[Figure 126]|[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]|[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]| |[Figure 134]<br><br>[Figure 135]<br><br>[Figure 136]<br><br>[Figure 137]|

[Figure 138]

Stem block

[Figure 139]

[Figure 140]

Convolution block Attention block Pooling block Unpooling block

[Figure 141]

[Figure 142]

LitePT-S

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

E0 E2 E3 E4

E1 D3 D2 D1 D0

[Figure 148]

Serialisation

| |×2<br><br>[Figure 149]<br><br>[Figure 150]| |×2<br><br>[Figure 151]<br><br>[Figure 152]<br><br>| |×2<br><br>[Figure 153]<br><br>[Figure 154]<br><br>|[Figure 155]<br><br>[Figure 156]<br><br>|×6<br><br>[Figure 157]<br><br>|[Figure 158]<br><br>[Figure 159]<br><br>|×2<br><br>[Figure 160]|[Figure 161]<br><br>[Figure 162]|×2<br><br>[Figure 163]<br><br>| |[Figure 164]<br><br>×2<br><br>[Figure 165]| |[Figure 166]<br><br>×2<br><br>[Figure 167]| |[Figure 168]<br><br>×2<br><br>[Figure 169]| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 170]|[Figure 171]|[Figure 172]|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]<br><br>[Figure 177]|[Figure 178]<br><br>[Figure 179]<br><br>[Figure 180]|[Figure 181]<br><br>[Figure 182]|[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]|[Figure 186]<br><br>[Figure 187]<br><br>[Figure 188]| |[Figure 189]|[Figure 190]|[Figure 191]|[Figure 192]|[Figure 193]|[Figure 194]| |

[Figure 195]

[Figure 196]

Linear Activation Normalization

[Figure 197]

[Figure 198]

[Figure 199]

LitePT-S*

[Figure 200]

[Figure 201]

[Figure 202]

Self-Attention Convolution

[Figure 203]

E0

E1

E2

E3

E4

D3

D2

D1

D0

[Figure 204]

[Figure 205]

×2

×2

×2

×6

×2

×2

×2

×2

×2

|[Figure 206]<br><br>|[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>|[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]| |[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]|[Figure 217]<br><br>|[Figure 218]<br><br>|[Figure 219]<br><br>[Figure 220]<br><br>|[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]| |[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]| |[Figure 228]<br><br>[Figure 229]<br><br>[Figure 230]| |[Figure 231]<br><br>|[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>|[Figure 235]<br><br>[Figure 236]| |[Figure 237]|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 238]<br><br>[Figure 239]|[Figure 240]<br><br>[Figure 241]|[Figure 242]<br><br>|[Figure 243]<br><br>[Figure 244]|[Figure 245]<br><br>|[Figure 246]<br><br>[Figure 247]|[Figure 248]<br><br>|[Figure 249]<br><br>[Figure 250]|[Figure 251]<br><br>|[Figure 252]<br><br>[Figure 253]|[Figure 254]|[Figure 255]<br><br>[Figure 256]|[Figure 257]|[Figure 258]<br><br>[Figure 259]|[Figure 260]|[Figure 261]<br><br>[Figure 262]|[Figure 263]|[Figure 264]<br><br>[Figure 265]| |

[Figure 266]

M Max-Pooling

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

+ Sum

Point Transformer V3

[Figure 272]

[Figure 273]

PointROPE

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

###### +

###### + + M

###### +

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Stem block Unpooling block

Convolution block

Attention block

Pooling block

- Figure 7. Detailed architectures. We illustrate the full pipelines of LitePT-S, LitePT-S*, Point Transformer V3 [88], and the building blocks of each architecture.

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

- Figure 8. PointROPE attention. We apply PointROPE to query and key before standard scaled dot-product attention.

Other positional encodings. We additionally compare PointROPE with absolute positional encoding (APE) using sinusoidal functions, and with relative positional encoding (RPE) on nuScenes, in Tab. 17. PointROPE achieves the clearly highest mIoU while introducing the smallest increase in latency.

##### C.2. Cross-dataset Transfer

To evaluate cross-dataset transfer performance, we first pretrain PTv3 and LitePT-S for semantic segmentation on Structured3D, the largest dataset in our benchmark suite. We then freeze the pretrained encoders and test on ScanNet with a linear probe on the multi-scale features. As shown in Tab. 18, LitePT outperforms PTv3 by 7.9pp in mIoU, indicating better generalisation. We attribute the cross-dateset generalization to PointROPE, which preserves relative spatial relationships across scenes with differing layouts and sensor characteristics. This property enables the model to learn more transferable representations, reducing overfitting to dataset-specific structures and improving robustness under domain shift.

across three axes (xi,yi,zi). Here we explore the impact of different subdivisions on each axis. In addition to equal split (6:6:6), we try emphasizing the z axis (4:4:10) and emphasizing the xy axes (8:8:2). As shown in Tab. 16, uneven splits lead to suboptimal performance compared with equal weighting. This suggests that positional information along all three axes is similarly important, and manual reweighting is unnecessary.

| |LitePT-S<br><br>|LitePT-S*|LitePT-B<br><br>|LitePT-L|
|---|---|---|---|---|
|stem<br><br>|C=36,K=5×5×5|C=36,K=5×5×5<br><br>|C=36,K=5×5×5<br><br>|C=36,K=5×5×5|
|E0| <br><br>C=36 K=3×3×3<br><br> ×2<br><br>| <br><br>C=36 K=3×3×3<br><br> ×2| <br><br>C=54 K=3×3×3<br><br> ×3| <br><br>C=72 K=3×3×3<br><br> ×3|
|E1|pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |
| |<br><br>C=72 K=3×3×3 ×2|<br><br>C=72 K=3×3×3 ×2<br><br>|<br><br>C=108 K=3×3×3 ×3|<br><br>C=144 K=3×3×3 ×3|
|E2|pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |
| |<br><br>C=144 K=3×3×3 ×2|<br><br>C=144 K=3×3×3 ×2<br><br>|<br><br>C=216 K=3×3×3 ×3<br><br>|<br><br>C=288 K=3×3×3 ×3|
|E3<br><br>|pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |
| | <br><br>C=252,H=14 b=100,F=4 N=1024<br><br> ×6<br><br>| <br><br>C=252,H=14 b=100,F=4 N=1024<br><br> ×6| <br><br>C=432,H=24 b=100,F=4 N=1024<br><br> ×12<br><br>| <br><br>C=576,H=32 b=100,F=4 N=1024<br><br> ×12|
|E4|pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |pool stride 2<br><br> |
| | <br><br>C=504,H=28 b=100,F=4 N=1024<br><br> ×2<br><br>| <br><br>C=504,H=28 b=100,F=4 N=1024<br><br> ×2| <br><br>C=576,H=32 b=100,F=4 N=1024<br><br> ×3<br><br>| <br><br>C=864,H=48 b=100,F=4 N=1024<br><br> ×3|
|D3|unpool C=252<br><br>|unpool C=252<br><br> |unpool C=432<br><br>|unpool C=576|
| | | <br><br>C=252,H=14 b=100,F=4 N=1024<br><br> ×2| | |
|D2|unpool C=144<br><br>|unpool C=144  |unpool C=216|unpool C=288<br><br>|
| | |<br><br>C=144 K=3×3×3 ×2| | |
|D1|unpool C=72<br><br>|unpool C=72  <br><br>|unpool C=108<br><br>|unpool C=144|
| | |<br><br>C=72 K=3×3×3 ×2| | |
|D0|unpool C=72<br><br>|unpool C=72  <br><br>|unpool C=72|unpool C=72|
| | |<br><br>C=72 K=3×3×3 ×2| | |
|#Params<br><br>|12.7M<br><br>|16.0M|45.1M|85.9M|

###### Table 12. Detailed architecture specifications. C: channel dimension, K: kernel size in the convolution block, H: number of head, b: base frequency of PointROPE, F: MLP ratio in the FFN module, N: number of points in local group.

nuScenes [6] Waymo [72] ScanNet [14] Structured3D [104]

Input feature XYZ+Intensity RGB+Normal Grid size 0.05m 0.02m Head (framework) Linear segmentor Linear segmentor Loss CrossEntropy+Lovasz [3] CrossEntropy+Lovasz [3] Optimizer AdamW [47] AdamW [47] Weight decay 0.005 0.05 Scheduler OneCycleLR [67] OneCycleLR [67] Learning rate 0.002 0.006 0.012 Block lr rate 0.0002 0.0006 0.0012 Batch size 12 12 48 Epochs 50 1200 (800) 200 Num GPUs 4 4 16

Random shift, random dropout random rotate, random scale random flip, random jitter elastic distortion, color auto contrast color jitter, sphere crop color normalization

Random rotate, random scale random flip, random jitter

Data augmentation

###### Table 13. Detailed training settings for semantic segmentation.

ScanNet [14] ScanNet200 [62]

Input feature RGB+Normal Grid size 0.02m Head (framework) PointGroup [37] Loss check PointGroup [37] Optimizer AdamW [47] Weight decay 0.05 Scheduler OneCycleLR [67] Learning rate 0.006 Block lr rate 0.0006 Batch size 12 Epochs 800 Num GPUs 4

Random shift, random dropout random rotate, random scale random flip, random jitter elastic distortion, color auto contrast color jitter sphere crop, color normalization

Data augmentation

###### Table 14. Detailed training settings for instance segmentation.

Object Detection Waymo [72]

Input feature XYZ+Intensity+Elongation Grid size (0.32m, 0.32m, 6.0m) Head (framework) CenterPoint-Pillar [42] Loss check CenterPoint-Pillar [42] Optimizer Adam [52] Weight decay 0.01 Scheduler OneCycleLR [67] Learning rate 0.006 Block lr rate 0.006 Batch size 64 Epochs 40 Num GPUs 16

Random flip, random rotate random scale

Data augmentation

Table 15. Detailed training settings for object detection.

mIoU mAcc

w/o PointROPE 79.6 86.5 Spherical 80.7 87.1 Cartesian 82.2 88.1 x:y:z=6:6:6 82.2 88.1 x:y:z=4:4:10 80.3 86.8 x:y:z=8:8:2 80.3 86.7

- Table 16. Additional ablation on PointROPE on nuScenes.

w/o PosEnc APE (sinusoidal) RPE PointROPE

mIoU 79.6 80.2 80.6 82.2 Latency 20.7ms 23.3ms 27.6ms 21.5ms

- Table 17. Comparison with other positional encoding schemes on nuScenes.

Method #Param mIoU mAcc

PTv3 [88] 46.1M 57.1 69.5 LitePT-S 12.7M 65.0 77.2

- Table 18. Cross-dataset transfer. Models are pretrained on Structured3D and evaluated on ScanNet validation set using linear probing.

##### C.3. Comparison with PointMamba

We additionally compare LitePT with the Mamba-based point cloud backbone PointMamba (PM) [44]. For small, object-centric datasets, PM is indeed a viable alternative, see shape classification results for ModelNet40 in Tab. 19 ⃝1 . Both models are trained from scratch and evaluated without voting.

To adopt PM to scene-level semantic segmentation, we

fix the ##groupinput tokenspoints ratio of the best-reported model and progressively increase the number of input points (Fig. 9).

We observe that PM’s reliance on farthest point sampling (FPS), k-NN, and PointNet-style feature propagation leads to rapid growth in compute and memory, limiting its scalability to larger scenes. In Tab. 19 ⃝2 , we train PM on ScanNet with 640 tokens per scene, the maximum feasible on an RTX 4090, resulting in a 16pp mIoU drop compared to LitePT.

⃝1 ModelNet40 ⃝2 ScanNet #Params Latency Memory OA (%) mIoU

PointMamba 12.3M 10.5ms 0.20G 92.4 60.5 LitePT-S (Ours) 12.7M 11.7ms 0.13G 92.5 76.5

Table 19. Comparison with PointMamba.

##### C.4. Additional Results for LitePT-L

In Tab. 20, we show LitePT-L results on key benchmarks. We note that, in accordance with established scaling laws, model scaling should be considered in conjunction with data volume. Upgrading from LitePT-S to LitePT-

Figure 9. LitePT-S vs. PointMamba efficiency on ScanNet.

L consistently improves performance. As expected, the gain strongly depends on the dataset size (as denoted by #Points). For instance, scaling up to LitePT-L yields a +1.8pp gain in mIoU for Structured3D (9.5 billion points) whereas the improvement is only +0.4pp for nuScenes (1 billion points), as there is not enough data to fully exploit the higher capacity.

Semantic Segmentation (mIoU) Instance Segmentation (mAP) Structured3D Waymo nuScenes ScanNet

#Points (Billion) 9.55 4.15 0.98 0.17 PTv3 82.4 71.3 80.4 40.9 LitePT-S 83.6 73.1 82.2 41.7 LitePT-L 85.4 (+1.8) 74.0 (+0.9) 82.6 (+0.4) 42.3 (+0.6)

Table 20. More results for LitePT-L.

##### C.5. Benchmarking Outdoor Efficiency

We evaluate the efficiency of semantic segmentation on Waymo, a large-scale outdoor LiDAR dataset, in Tab. 21. Similar to our observation on indoor datasets, LitePT-S remains more efficient than PTv3, with > 2× faster runtime at inference.

Training Inference Method #Params Latency Memory Latency Memory mIoU PTv3 46.1M 153ms 14.1G 100ms 7.1G 71.3 LitePT-S (Ours) 12.7M 86ms 9.4G 47ms 5.4G 73.1

Table 21. Outdoor Efficiency comparison on Waymo. Results are reported as average over the full Waymo dataset using a single RTX 4090 GPU. Automatic Mixed Precision (AMP) is enabled for all models during training and disabled during inference.

##### C.6. Chunking and Test-Time Augmentation

In the main paper, we report semantic segmentation results following the same evaluation protocol as prior works [87, 88] to ensure a fair comparison. The testing pipeline applies chunking and test-time augmentations (TTA). Specifically, each augmented sample is partitioned into overlapping chunks, ensuring that every point is assigned to at least one chunk during grid sampling. The model is then run on each chunk individually, and the final label of each point is aggregated by voting across the predictions from all chunks it appears in. Although this multi-run and TTA protocol is common practice and is known to boost performance [66], it

Method #Param mIoU mAcc

PTv3 [88] 46.1M 80.4 87.2 LitePT-S 12.7M 82.2 88.1

PTv3 [88] (w/o chunking and TTA) 46.1M 78.3 86.0 LitePT-S (w/o chunking and TTA) 12.7M 80.4 86.9

Table 22. Semantic segmentation on nuScenes without chunking and TTA.

obscures the intrinsic merits of the underlying backbone. To communicate performance in a simpler single-pass setting useful for downstream users, we additionally report results for PTv3 and LitePT-S without TTA or chunking in Tab. 22. Overall, removing chunking and TTA reduces performance by roughly 2% mIoU for both methods.

### D. Visualization

We visualize sample predictions of LitePT on three tasks: 3D semantic segmentation (Figs. 10 to 13), 3D instance segmentation (Fig. 14), and 3D object detection (Fig. 15).

### References

- [1] Matan Atzmon, Haggai Maron, and Yaron Lipman. Point Convolutional Neural Networks by Extension Operators. ACM Transactions on Graphics (TOG), 2018. 2
- [2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer Normalization. arXiv preprint arXiv:1607.06450,

2016. 9

- [3] Maxim Berman, Amal Rannen Triki, and Matthew B Blaschko. The Lov´asz-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 12
- [4] Alexandre Boulch, Joris Guerry, Bertrand Le Saux, and Nicolas Audebert. SnapNet: 3D Point Cloud Semantic Labeling with 2D Deep Segmentation Networks. Computers & Graphics, 2018. 2
- [5] Finn Lukas Busch, Timon Homberger, Jes´us OrtegaPeimbert, Quantao Yang, and Olov Andersson. One Map to Find them All: Real-time Open-vocabulary Mapping for Zero-shot Multi-object Navigation. In International Conference on Robotics and Automation (ICRA), 2025. 1
- [6] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A Multimodal Dataset for Autonomous Driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3, 4, 7, 12
- [7] Liyan Chen, Gregory P Meyer, Zaiwei Zhang, Eric M Wolff, and Paul Vernaza. Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [8] Wanli Chen, Xinge Zhu, Guojin Chen, and Bei Yu. Efficient

- Point Cloud Analysis Using Hilbert Curve. In European Conference on Computer Vision (ECCV), 2022. 2
- [9] Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, and Tian Xia. Multi-View 3D Object Detection Network for Autonomous Driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2
- [10] Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, and Jiaya Jia. LargeKernel3D: Scaling Up Kernels in 3D Sparse CNNs. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [11] Ran Cheng, Ryan Razani, Ehsan Taghavi, Enxu Li, and Bingbing Liu. (AF)2-S3Net: Attentive Feature Fusion with Adaptive Feature Selection for Sparse Semantic Segmentation Network. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 7
- [12] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2, 3, 4, 6, 7, 8, 9
- [13] Xiangxiang Chu, Zhi Tian, Bo Zhang, Xinlong Wang, and Chunhua Shen. Conditional Positional Encodings for Vision Transformers. International Conference on Learning Representations (ICLR), 2023. 3
- [14] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 3, 4, 7, 8, 12
- [15] Zihang Dai, Hanxiao Liu, Quoc V Le, and Mingxing Tan. CoAtNet: Marrying Convolution and Attention for All Data Sizes. Advances in Neural Information Processing Systems (NeurIPS), 2021. 3
- [16] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. International Conference on Learning Representations (ICLR), 2024. 9
- [17] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Advances in Neural Information Processing Systems (NeurIPS), 2022. 9
- [18] Hao Deng, Kunlei Jing, Shengmei Cheng, Cheng Liu, Jiawei Ru, Jiang Bo, and Lin Wang. LinNet: Linear Network for Efficient Point Cloud Representation Learning. Advances in Neural Information Processing Systems (NeurIPS), 2024. 2
- [19] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations (ICLR), 2021. 3
- [20] Lunhao Duan, Shanshan Zhao, Nan Xue, Mingming Gong, Gui-Song Xia, and Dacheng Tao. ConDaFormer: Disassembled Transformer with Local Structure Enhancement for 3D Point Cloud Understanding. Advances in Neural Information Processing Systems (NeurIPS), 2023. 3, 5
- [21] Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, and Zhaoxiang

- Zhang. Embracing Single Stride 3D Object Detector with Sparse Transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 8
- [22] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2012. 1
- [23] Benjamin Graham, Martin Engelcke, and Laurens Van Der Maaten. 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2018. 2, 4, 9

- [24] Fabian Groh, Patrick Wieschollek, and Hendrik PA Lensch. Flex-Convolution. In Asian Conference on Computer Vision (ACCV), 2018. 2
- [25] Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, and Chang Xu. CMT: Convolutional Neural Networks Meet Vision Transformers . In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [26] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R Martin, and Shi-Min Hu. PCT: Point Cloud Transformer. Computational Visual Media, 2021. 2
- [27] Lei Han, Tian Zheng, Lan Xu, and Lu Fang. OccuSeg: Occupancy-aware 3D Instance Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
- [28] Chenhang He, Ruihuang Li, Shuai Li, and Lei Zhang. Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection From Point Clouds. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 8
- [29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 9
- [30] Tong He, Chunhua Shen, and Anton Van Den Hengel. DyCo3D: Robust Instance Segmentation of 3D Point Clouds Through Dynamic Convolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 3
- [31] Dan Hendrycks and Kevin Gimpel. Gaussian Error Linear Units (GELUs). arXiv preprint arXiv:1606.08415, 2016. 9
- [32] Ji Hou, Benjamin Graham, Matthias Nießner, and Saining Xie. Exploring Data-Efficient 3D Scene Understanding with Contrastive Scene Contexts. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2021. 8

- [33] Binh-Son Hua, Minh-Khoi Tran, and Sai-Kit Yeung. Pointwise Convolutional Neural Networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [34] Jing Huang and Suya You. Point Cloud Labeling Using 3D Convolutional Neural Network. In International Conference on Pattern Recognition (ICPR), 2016. 2
- [35] Jakob Iglhaut, Carlos Cabo, Stefano Puliti, Livia Piermattei, James O’Connor, and Jacqueline Rosette. Structure from Motion Photogrammetry in Forestry: A Review. Current Forestry Reports, 2019. 1

- [36] Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In International Conference on Machine Learning (ICML), 2015. 9
- [37] Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, ChiWing Fu, and Jiaya Jia. PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2020. 8, 12

- [38] Evangelos Kalogerakis, Melinos Averkiou, Subhransu Maji, and Siddhartha Chaudhuri. 3D Shape Segmentation with Projective Convolutional Networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2
- [39] Xin Lai, Jianhui Liu, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, and Jiaya Jia. Stratified Transformer for 3D Point Cloud Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3
- [40] Xin Lai, Yukang Chen, Fanbin Lu, Jianhui Liu, and Jiaya Jia. Spherical Transformer for LiDAR-Based 3D Recognition. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 7
- [41] Loic Landrieu and Martin Simonovsky. Large-Scale Point Cloud Semantic Segmentation with Superpoint Graphs. In IEEE/CVF Conference on Computer Vision and Pattern

- Recognition (CVPR), 2018. 2

[42] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. PointPillars: Fast Encoders for Object Detection from Point Clouds. In IEEE/CVF Conference on Computer Vision and Pattern

- Recognition (CVPR), 2019. 2, 8, 12

- [43] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. PointCNN: Convolution On XTransformed Points. Advances in Neural Information Processing Systems (NeurIPS), 2018. 2
- [44] Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, and Xiang Bai. PointMamba: A Simple State Space Model for Point Cloud Analysis. Advances in Neural Information Processing Systems (NeurIPS), 2024. 12
- [45] Zhijian Liu, Haotian Tang, Yujun Lin, and Song Han. PointVoxel CNN for Efficient 3D Deep Learning. Advances in Neural Information Processing Systems (NeurIPS), 2019. 2
- [46] Zhijian Liu, Xinyu Yang, Haotian Tang, Shang Yang, and Song Han. FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2023. 2, 8, 9

- [47] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. International Conference on Learning Representations (ICLR), 2019. 12
- [48] Kan Luo, Hongshan Yu, Xieyuanli Chen, Zhengeng Yang, Jingwen Wang, Panfei Cheng, and Ajmal Mian. 3D Point Cloud-based Place Recognition: A Survey. Artificial Intelligence Review, 2024. 1
- [49] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking Network Design and Local Geometry in Point

- Cloud: A Simple Residual MLP Framework. International Conference on Learning Representations (ICLR), 2022. 2
- [50] Daniel Maturana and Sebastian Scherer. VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2015. 2
- [51] Sachin Mehta and Mohammad Rastegari. MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transformer. International Conference on Learning Representations (ICLR), 2022. 3
- [52] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. International Conference on Learning Representations (ICLR), 2015. 12
- [53] Chunghyun Park, Yoonwoo Jeong, Minsu Cho, and Jaesik Park. Fast Point Transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [54] Bohao Peng, Xiaoyang Wu, Li Jiang, Yukang Chen, Hengshuang Zhao, Zhuotao Tian, and Jiaya Jia. OA-CNNs: Omni-Adaptive Sparse CNNs for 3D Semantic Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2
- [55] Patrick Pfaff, Rudolph Triebel, Cyrill Stachniss, Pierre Lamon, Wolfram Burgard, and Roland Siegwart. Towards Mapping of Cities. In International Conference on Robotics and Automation (ICRA), 2007. 1
- [56] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2, 3
- [57] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. Advances in Neural Information Processing Systems (NeurIPS), 2017.
- [58] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies. Advances in Neural Information Processing Systems (NeurIPS), 2022. 2
- [59] Damien Robert, Hugo Raguet, and Loic Landrieu. Efficient 3D Semantic Segmentation with Superpoint Transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 3
- [60] Damien Robert, Hugo Raguet, and Loic Landrieu. Scalable 3D Panoptic Segmentation As Superpoint Graph Clustering. In International Conference on 3D Vision (3DV), 2024. 2
- [61] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional Networks for Biomedical Image Segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention ( MICCAI), 2015. 2, 3, 6, 9
- [62] David Rozenberszki, Or Litany, and Angela Dai. Language-Grounded Indoor 3D Semantic Segmentation in the Wild. In European Conference on Computer Vision (ECCV), 2022. 8, 12
- [63] Jonas Schult, Francis Engelmann, Alexander Hermans, Or Litany, Siyu Tang, and Bastian Leibe. Mask3D:

- Mask Transformer for 3D Semantic Instance Segmentation. In International Conference on Robotics and Automation (ICRA), 2023. 3
- [64] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Lowprecision. Advances in Neural Information Processing Systems (NeurIPS), 2024. 9
- [65] Guangsheng Shi, Ruifeng Li, and Chao Ma. PillarNet: Real-Time and High-Performance Pillar-Based 3D Object Detection. In European Conference on Computer Vision (ECCV), 2022. 8
- [66] Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. International Conference on Learning Representations (ICLR), 2015. 13
- [67] Leslie N Smith and Nicholay Topin. Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, 2019. 12
- [68] Hongli Song, Weiliang Wen, Sheng Wu, and Xinyu Guo. Comprehensive Review on 3D Point Cloud Segmentation in Plants. Artificial Intelligence in Agriculture, 2025. 1
- [69] Shuran Song, Fisher Yu, Andy Zeng, Angel X Chang, Manolis Savva, and Thomas Funkhouser. Semantic Scene Completion from a Single Depth Image. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 2
- [70] Hang Su, Subhransu Maji, Evangelos Kalogerakis, and Erik Learned-Miller. Multi-View Convolutional Neural Networks for 3D Shape Recognition. In International Conference on Computer Vision (ICCV), 2015. 2
- [71] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 2024. 2, 3, 5, 9
- [72] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 1, 7, 12
- [73] Pei Sun, Mingxing Tan, Weiyue Wang, Chenxi Liu, Fei Xia, Zhaoqi Leng, and Dragomir Anguelov. SWFormer: Sparse Window Transformer for 3D Object Detection in Point Clouds. In European Conference on Computer Vision (ECCV), 2022. 2
- [74] Haotian Tang, Zhijian Liu, Shengyu Zhao, Yujun Lin, Ji Lin, Hanrui Wang, and Song Han. Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution. In European Conference on Computer Vision (ECCV), 2020. 2, 7
- [75] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J Guibas. KPConv: Flexible and Deformable Convolution for Point Clouds. In International Conference on Computer Vision (ICCV), 2019. 2, 3

- [76] Hugues Thomas, Yao-Hung Hubert Tsai, Timothy D Barfoot, and Jian Zhang. KPConvX: Modernizing Kernel Point Convolution with Kernel Attention. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

2024. 3, 5

- [77] Tuan Anh Tran, Duy Minh Ho Nguyen, Hoai-Chau Tran, Michael Barz, Khoa D. Doan, Roger Wattenhofer, Vien Anh Ngo, Mathias Niepert, Daniel Sonntag, and Paul Swoboda. How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need? Advances in Neural Information Processing Systems (NeurIPS), 2025. 2
- [78] Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan Bovik, and Yinxiao Li. MaxViT: Multi-axis Vision Transformer. In European Conference on Computer Vision (ECCV), 2022. 3
- [79] Nina Varney, Vijayan K Asari, and Quinn Graehling. DALES: A Large-scale Aerial LiDAR Data Set for Semantic Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020. 1
- [80] Peng-Shuai Wang. OctFormer: Octree-based Transformers for 3D Point Clouds. ACM Transactions on Graphics (TOG), 2023. 2
- [81] Ruisheng Wang, Jiju Peethambaran, and Dong Chen. Lidar Point Clouds to 3-D Urban Models: A Review. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2018. 1
- [82] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic Graph CNN for Learning on Point Clouds. ACM Transactions on Graphics (TOG), 2019. 2
- [83] Bichen Wu, Alvin Wan, Xiangyu Yue, and Kurt Keutzer. SqueezeSeg: Convolutional Neural Nets with Recurrent CRF for Real-Time Road-Object Segmentation from 3D Lidar Point Cloud. In International Conference on Robotics and Automation (ICRA), 2018. 2
- [84] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. CvT: Introducing Convolutions to Vision Transformers. In International Conference on Computer Vision (ICCV), 2021. 3
- [85] Wenxuan Wu, Zhongang Qi, and Li Fuxin. PointConv: Deep Convolutional Networks on 3D Point Clouds. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2
- [86] Wenxuan Wu, Li Fuxin, and Qi Shan. PointConvFormer: Revenge of the Point-based Convolution. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3, 5
- [87] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point Transformer V2: Grouped Vector Attention and Partition-Based Pooling. Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 3, 6, 7, 8, 9, 13
- [88] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point Transformer V3: Simpler, Faster, Stronger. In IEEE/CVF Conference on Computer Vision

- and Pattern Recognition (CVPR), 2024. 2, 3, 5, 6, 7, 8, 9, 10, 12, 13
- [89] Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, and Julian Straub. Sonata: SelfSupervised Learning of Reliable Point Representations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 7
- [90] Kai M Wurm, Armin Hornung, Maren Bennewitz, Cyrill Stachniss, and Wolfram Burgard. OctoMap: A Probabilistic, Flexible, and Compact 3D Map Representation for Robotic Systems. In International Conference on Robotics and Automation (ICRA), 2010. 1
- [91] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On Layer Normalization in the Transformer Architecture. In International Conference on Machine Learning (ICML), 2020. 3, 9
- [92] Shengdong Xu, Dominik Honegger, Marc Pollefeys, and Lionel Heng. Real-Time 3D Navigation for Autonomous Vision-Guided MAVs. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2015. 1
- [93] Yifan Xu, Tianqi Fan, Mingye Xu, Long Zeng, and Yu Qiao. SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filter. In European Conference on Computer Vision (ECCV), 2018. 2
- [94] Yufei Xu, Qiming Zhang, Jing Zhang, and Dacheng Tao. ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias. Advances in Neural Information Processing Systems (NeurIPS), 2021. 3
- [95] Honghui Yang, Wenxiao Wang, Minghao Chen, Binbin Lin, Tong He, Hua Chen, Xiaofei He, and Wanli Ouyang. PVTSSD: Single-Stage 3D Object Detector With Point-Voxel Transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3
- [96] Yu-Qi Yang, Yu-Xiao Guo, and Yang Liu. Swin3D++: Effective Multi-Source Pretraining for 3D Indoor Scene Understanding. Computational Visual Media, 2025. 2
- [97] Yu-Qi Yang, Yu-Xiao Guo, Jian-Yu Xiong, Yang Liu, Hao Pan, Peng-Shuai Wang, Xin Tong, and Baining Guo. Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding. Computational Visual Media,

2025. 2

- [98] Zetong Yang, Li Jiang, Yanan Sun, Bernt Schiele, and Jiaya Jia. A Unified Query-Based Paradigm for Point Cloud Understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [99] Tianwei Yin, Xingyi Zhou, and Philipp Krahenbuhl. Center-Based 3D Object Detection and Tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 8
- [100] Ziyin Zeng, Mingyue Dong, Jian Zhou, Huan Qiu, Zhen Dong, Man Luo, and Bijun Li. DeepLA-Net: Very Deep Local Aggregation Networks for Point Cloud Analysis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2
- [101] Cheng Zhang, Haocheng Wan, Xinyi Shen, and Zizhao Wu. PatchFormer: An Efficient Point Transformer with Patch

- Attention. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [102] Ji Zhang and Sanjiv Singh. LOAM: Lidar odometry and mapping in real-time. In Robotics: Science and Systems,

2014. 1

- [103] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point Transformer. In International Conference on Computer Vision (ICCV), 2021. 2, 3
- [104] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A Large Photo-Realistic Dataset for Structured 3D Modeling. In European Conference on Computer Vision (ECCV), 2020. 7, 12
- [105] Xinge Zhu, Hui Zhou, Tai Wang, Fangzhou Hong, Yuexin Ma, Wei Li, Hongsheng Li, and Dahua Lin. Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 7

- ● barrier ● bicycle ● bus ● car ● construction vehicle ● motorcycle ● pedestrian ● traffic cone ● trailer
- ● truck ● driveable surface ● other flat surface ● sidewalk ● terrain ● manmade ● vegetation ● unlabelled

1ccdbec944bd4994...049d115cb992491b...5f8393250fae4960...6bfd64d077884228...8f78c446a68d4854...2f678cb1e67d42ae...

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

(a) Input (b) Prediction (c) Ground Truth

- Figure 10. nuScenes semantic segmentation. We present various scenes of the nuScenes validation set: the input point cloud colored by

● car ● truck ● bus ● other vehicle ● motorcyclist ● bicyclist ● pedestrian ● sign ● traffic light ● traffic pole ● construction cone ● bicycle ● motorcycle ● building ● vegetation ● tree trunk ● curb ● road ● lane marker ● other ground ● walkable ● sidewalk ● unlabelled

3077229433993844...8956556778987472...1825211188287550...9041488218266405...110376513715...1833392207058224...

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

(a) Input (b) Prediction (c) Ground Truth

- Figure 11. Waymo semantic segmentation. We present various scenes of the Waymo validation set: the input point cloud colored by

● wall ● floor ● cabinet ● bed ● chair ● sofa ● table ● door ● window ● bookshelf ● picture ● counter ● desk ● curtain ● refrigerator ● shower ● toilet ● sink ● bathtub ● other furniture ● unlabelled

scene003000scene065100scene037802scene040602scene064501scene016900

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

(a) Input (b) Prediction (c) Ground Truth

- Figure 12. ScanNet semantic segmentation. We present various scenes of the ScanNet validation set: the input point cloud, the semantic segmentation from LitePT-S, and the corresponding ground truth.

● wall ● floor ● cabinet ● bed ● chair ● sofa ● table ● door ● window ● picture ● desk ● shelves ● curtain ● dresser ● pillow ● mirror ● ceiling ● refrigerator ● television ● nightstand ● sink ● lamp ● other structure ● other furniture ● other properties

scene03022room8765scene03034room401scene03223room4894scene03113room560scene03195room1764scene03237room2846

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

(a) Input (b) Prediction (c) Ground Truth

- Figure 13. Structured3D semantic segmentation. We present various scenes of the Structured3D validation set: the input point cloud, the semantic segmentation from LitePT-S, and the corresponding ground truth.

scene059102scene062100scene065102scene064501scene001101scene016400

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

(a) Input (b) Prediction (c) Ground Truth

- Figure 14. ScanNet instance segmentation. We present various scenes of the ScanNet validation set: the input point cloud, the instance segmentation from LitePT-S*, and the corresponding ground truth. Colors for each instance are randomly assigned.

● vehicle ● pedestrian ● cyclist

1335699760417784...6621886863973648...8956556778987472...1333688303428388...1430000760420586...3077939657605416...

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

(a) Input (b) Prediction (c) Ground Truth

- Figure 15. Waymo object detection. We present various scenes of the Waymo validation set: the input point cloud, the object detections from LitePT, and the corresponding ground truth.

