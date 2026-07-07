# arXiv:2508.09062v1[cs.GR]12Aug2025

## VertexRegen: Mesh Generation with Continuous Level of Detail

Xiang Zhang1* Yawar Siddiqui2 Armen Avetisyan2 Chris Xie2 Jakob Engel2 Henry Howard-Jenkins2

1UC San Diego 2Meta Reality Labs Research

#### (Ours)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

Steps Generation Process

5 20 40 80 120 160 200 250 300 350 375

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

Previous work

Figure 1. Mesh generation process of VertexRegen (coarse-to-fine process) vs. previous work (partial-to-complete process). Prior work generates meshes face by face, with the step count corresponding to the face count in the figure. In contrast, VertexRegen produces meshes with a continuously increasing level of detail, where each step yields a valid mesh Mk.

### Abstract

We introduce VertexRegen, a novel mesh generation framework that enables generation at a continuous level of detail. Existing autoregressive methods generate meshes in a partial-to-complete manner and thus intermediate steps of generation represent incomplete structures. VertexRegen takes inspiration from progressive meshes and reformulates the process as the reversal of edge collapse, i.e. vertex split, learned through a generative model. Experimental results demonstrate that VertexRegen produces meshes of comparable quality to state-of-the-art methods while uniquely offering anytime generation with the flexibility to halt at any step to yield valid meshes with varying levels of detail.

### 1. Introduction

Meshes are essential for 3D asset representation and are widely used in industries such as film, design, and gaming due to their compatibility with most 3D software and

*Work conducted while the author was an intern at Meta. Project page: https://vertexregen.github.io

hardware. However, manually creating high-quality meshes is costly and time-consuming, prompting extensive research into automated 3D content creation. Largely, these approaches have used alternative representations such as neural fields [11, 27, 31, 38], voxels [34, 51] or point clouds [54, 59] which are later post-processed to meshes instead of direct mesh modeling. Unfortunately, these post-processed meshes often exhibit poor topology, overtessellation, and artifacts, lacking the quality of artistcrafted meshes.

Recently, there has been a surge of approaches that directly generate meshes, using autoregressive models to represent meshes as sequences of triangles [2–4, 15, 37, 43]. These methods capture the high-fidelity and aesthetic qualities of artist-created works without the need for postgeneration conversion. Although significant progress has been made in improving tokenization schemes [2, 4, 37] and network architectures [15], these approaches do not fundamentally alter a key characteristic of generation: namely, to produce a valid mesh, the full sequence must be generated to completion. Consequently, these methods offer no control over the level of detail during generation; early stopping results in a mesh with missing faces. Simple extensions,

such as face-count conditioning tokens in EdgeRunner [43], have been proposed to introduce some coarse control over the detail by pre-specifying a target face count. However, each generation still must be completed in its entirety to yield a complete mesh, and thus a single generation still offers only a single level of detail.

To allow for a continuous level of detail generation, we take inspiration from Hoppe’s progressive mesh formulation [17]. Progressive meshes use two reversible operations to transition between levels of detail. The edge collapse operation simplifies a mesh into a coarser one by reducing one edge at a time. The vertex split operation reverses this simplification to add more detail by using information stored during edge collapse. By starting with a coarse mesh and keeping edge collapse records, this approach creates an efficient, lossless representation of the original mesh, allowing for continuous resolution adjustments by applying any number of vertex split operations.

In this paper, we build on this progressive mesh representation by learning the vertex split, i.e. reversing the edge collapse operation, as a generative problem. This allows the resulting mesh generation to inherit the properties of progressive meshes and provides a solution for anytime mesh generation, where the process can be stopped early to yield a coarser mesh rather than an incomplete one. By properly serializing the vertex split sequence, the entire generation process can be modeled using a Transformer trained with a next-token prediction objective, a widely adopted paradigm in mesh generation. We evaluate VertexRegen on the task of unconditional mesh generation, demonstrating superior results both qualitatively and quantitatively. We further provide examples of shape-conditioned generation. These experimental results illustrate the ability of our method to generate compelling meshes at a continuous resolution.

Our contributions can be summarized as follows:

- • Inspired by Hoppe’s progressive meshes [17], we reframe mesh generation as the reversal of edge collapse operations, i.e. generating vertex splits.
- • We formulate a token-efficient parameterization of a progressive mesh, through a half-edge data structure.
- • We propose VertexRegen for continuous level of detail mesh generation. VertexRegen creates meshes in a coarse-to-fine fashion, rather than partial-to-complete, uniquely providing a solution for anytime generation.

### 2. Related Work

3D Mesh Generation. Recent advancements in 3D shape generation have explored various representations, including point-clouds [30, 44, 52, 59], signed distance functions (SDFs) [7, 19, 22, 23, 36, 53, 57], neural radiance fields (NeRFs) [16, 18, 46, 55], and Gaussian splatting [14, 42]. These implicit representations require iso-surface extraction techniques [24, 35, 47] to output meshes, often resulting

in over-tessellated and excessively smooth outputs, which pose challenges for downstream applications such as geometric processing and manipulation.

In contrast, direct mesh generation inherently produces structured, well-defined geometry without the need for postprocessing or surface extraction, making it an increasingly prominent approach in recent years. Early methods tackle this task by generating meshes from surface patches [13], deforming ellipsoids [45], predicting mesh graphs [8], or employing binary space partitioning [6]. More recent techniques leverage generative models, particularly diffusion models and sequence-based approaches. PolyDiff [1] applies discrete diffusion, while PolyGen [29] autoregressively predicts vertices and faces using two separate networks. The sequence modeling paradigm has been further refined by representing the entire mesh as a single sequence. MeshGPT [37] introduces a tokenization scheme based on a vector quantized variational auto-encoder (VQVAE), while MeshXL [2] directly models discretized triangle soup sequences without compression. MeshAnything [4] extends MeshGPT by incorporating a point-cloud encoder for shape-conditioned generation. PivotMesh [49] introduces a hierarchical approach, generating pivot vertices before producing the full mesh. Further research has focused on optimizing tokenization and sequence modeling. Adjacency-aware compression techniques [3, 43, 50] improve tokenization efficiency and enable higher face counts within fixed context windows, while Meshtron [15] leverages Hourglass Transformers and sliding window attention to scale MeshXL sequences more effectively.

Despite these advancements, most existing approaches follow a partial-to-complete paradigm, where mesh regions are constructed sequentially at a uniform level of detail. Our work instead adopts a coarse-to-fine approach, continuously increasing the level of detail as new tokens are generated, achieving better control over geometric complexity.

Level of Detail Representations. Level of Detail [26] (LOD) is a widely used technique in computer graphics to optimize rendering performance by reducing the complexity of 3D models based on their size, distance from the camera, or importance in a scene. Different LOD strategies depend on the underlying representation of 3D shapes. For meshes, progressive meshes [17] generate a mesh sequence starting from a coarse base model, gradually refining it through a series of transformations that incrementally add detail. Traditional mesh simplification methods [12, 20] can be adopted to construct such sequences by iteratively reducing polygon count while preserving geometric fidelity. Progressive simplicial complexes [32] extend progressive meshes to handle arbitrary meshes, including non-manifold and non-orientable surfaces.

Recent studies have incorporated neural networks for

process can be repeated until there are no candidate edge collapses, e.g. without flipping face normals.

Edge collapse

𝑣

A vertex split, vsplit(vs,vl,vr,vt), defines the inverse of edge collapse. It restores vt and the two faces, {vt,vs,vl} and {vs,vt,vr}, which vanished during edge collapse, as well as vs to its original position. In case (vs,vt) is a boundary edge, either vl or vr will not exist, and there is only one face restored during the vertex split.

𝑣

𝑣

𝑣

𝑣

𝑣

𝑣

Vertex split

- Figure 2. Illustration of edge collapse and its inverse operation, i.e. vertex split. During edge collapse, vertex vt is collapsed into vs, resulting in two degenerate triangles (shaded in yellow), which forms two new edges in the result mesh.

The combination of edge collapse and its inverse vertex splits enables the progressive mesh representation, PM(M) = (M0,{vsplit0,··· ,vsplitn−1}) to express an arbitrary triangle mesh, M, as a combination of a coarse mesh, M0, obtained through edge collapse, and the sequence of n vertex split records required to reverse them:

LOD representations. Neural progressive meshes [5] propose a learned approach supporting LOD with a subdivision-based encoder-decoder. Similar representations have been explored for reconstruction approaches with signed distance functions (SDFs) [39–41] and neural fields [28]. However, these approaches are not generative; they typically encode and decode existing meshes [5] or optimize the LOD representation per shape [39]. In contrast, our approach is purely generative. VertexRegen learns to create meshes from scratch, progressively adding detail through autoregressive sequence generation. This allows us to generate new, high-fidelity meshes without relying on pre-existing structures, distinguishing our method from both traditional and neural LOD techniques.

M0 −−−−→vsplit0 M1 −−−−→vsplit1 ··· −−−−−−→vsplitn−1 M

##### 3.2. VertexRegen

Traditional progressive meshes require starting an initial detailed mesh to be simplified into a coarse mesh, M0, and recording each of these collapse steps as vertex split records to form the representation. We observe that the formation of the PM representation serves effectively as forward (edge collapse) and reverse (vertex split) processes that enable transition between detailed and coarse meshes.

VertexRegen frames the creation of a detailed mesh as the generation of a PM representation. Analogous to denoising for diffusion models, we train a generative model to reverse edge collapse. Concretely, VertexRegen first generates a coarse mesh, M0, from scratch, before increasing the level of detail through the generation of vertex split records.

### 3. Method

##### 3.1. Progressive Meshes Overview

A progressive mesh [17] (PM) proposes an efficient and lossless continuous-resolution representation for arbitrary triangle meshes. The representation is built off two observations: (i) that a single mesh transformation, edge collapse, is sufficient for effective simplification of meshes; (ii) that edge collapse transformations are invertible via a vertex split operation. In the following, we provide details of the edge collapse operation and its inversion with vertex splits, each illustrated in Fig. 2.

3.2.1. Progressive Mesh Parameterization We frame generation as an autoregressive sequential modeling task, with the full mesh sequence taking the form:

M : [ <bos>, [M 0 sequence], <sep>, #M0 [vsplit 0], ..., [vsplit n-1], <eos> ] #vsplits

The edge collapse operation, ecol(vs,vt), unifies two adjacent vertices vs and vt into a single vertex vs. The operation results in the vanishing of two faces, {vt,vs,vl} and {vs,vt,vr}, as well as the vertex vt. In the general case, a new position for vs is also specified; however, for the case of half-edge collapse, the original position for vs is kept.

In the following, we will define the formation of the subsequences [M 0 sequence] and [vsplit].

M0: Coarse Mesh Tokenization. For the initial coarse mesh, we follow the tokenization scheme defined in MeshXL [2]. In this formulation, embeddings are learned for discretized coordinates in an N3 grid. A vertex is represented via the sequential look-up of x-value, y-value, and z-value. A face is then constructed as the concatenation of its 3 vertices, totaling 9 tokens. The full mesh sequence is then defined as the concatenation of its constituent faces.

An initial mesh, M, can be simplified into a more coarse mesh, Mn, by n successive edge collapse operations:

M0 ←−−−ecol0 M1 ←−−−ecol1 ··· ←−−−−−ecoln−1 M

The order in which the edge collapse operations are performed is determined such that each successive edge collapse results in the minimum increase in Quadratic Error Metrics (QEM) [12] with respect to the original mesh. This

v : [ <x>, <y>, <z> ], #vertex F : [ [v 1], [v 2], [v 3] ], #face M : [ [F 1], [F 2], ..., [F N] ] #mesh

94% Original mesh

𝑣 𝑣

Face traversal direction

|𝑁 (𝑣 ): Neighbors of 𝑣 in ℳ 𝑁 (𝑣 ): Neighbors of 𝑣 in ℳ|
|---|

Coarsest mesh 0

ℋ

90%

ℋ

ℋ

ℋ

Percentage

20%

ℋ

ℋ

𝑣 𝑣 𝑣

Collect into 𝑁 (𝑣 )

Stop condition

16%

12%

ℋ ℋ ℋ

| |
|---|

8%

4%

Half-edge traversal order

𝑣 𝑣

0%

0 500 1000 1500 2000 2500 3000

- Figure 4. Illustration of half-edge data structure and the traversal process to determine the neighbors of vs and vt in Mk+1. Starting from half-edge Hls1 , we traverse the faces in clockwise direction until Hsr2 , where we collect the vertices v1 and v2 into Nk+1(vs).

1

2

3

4

5

7 6 8 BOS 1 2 3 2 4 3 3 4 5 SEP

3 5 1 6 6 N 1 7 6 5 N 8 EOS

ℳ Sequence

vsplit vsplit vsplit

|Vertex New vertex 𝑣 N <nil> token|
|---|

- Figure 5. Illustration of VertexRegen tokenization. The sequence begins with base mesh M0, followed by vertex split subsequences. A special <nil> token indicates either vl or vr does not exist.

Face count

- Figure 3. Face count distribution of the coarsest mesh M0 and original mesh M. The coarsest meshes (M0) contain significantly fewer faces than original (M), with an average of 18 and 457 faces, respectively.

Following [37], we sort vertices in z-y-x order (lower to higher). Within each face, vertices are cylindrically permuted to have the lowest-indexed vertex be first.

Although MeshXL [2] leverages this tokenization scheme to produce full detailed meshes, in VertexRegen only the coarsest level of the mesh is parameterized in this way. These coarse meshes consist of substantially fewer faces than the full detailed mesh, as demonstrated in Fig. 3, and the resulting coarse mesh only accounts for 5.68% of the total sequence length on average.

Fig. 4, we begin the traversal from Hls1 and proceed to the twin of the next half-edge, H12s. This operation is repeated until we reach H·rs. Throughout this process, we traverse the faces above (vl,vs) and (vs,vr) clockwise, obtaining:

Vertex Split Generation. Consider the k-th vertex split operation vsplitk, which converts a lower-detailed mesh Mk into a higher-detailed mesh Mk+1. We denote the neighbor vertices of vertex v in Mk as Nk(v). After determining the target vertex vs to split (in Mk) and the new vertex vt (in Mk+1), we need to obtain Nk+1(vs) and Nk+1(vt) from the vertex split record to connect vs and vt with correct neighbor vertices in Mk+1, which are prohibitive to generate as the number of neighbor vertices |Nk+1(vs)| and |Nk+1(vt)| is often large.

{vk |H·ks, k ̸= r} = Nk+1(vs) − {vl,vr,vt} (3) where H·ks is the half-edge encountered during traversal.

In conjunction with Eqs. (1) and (2), Nk+1(vt) can also be determined. When (vs,vt) is a boundary in Mk+1, we follow either Hls· clockwise or Hsr· counterclockwise until we reach a half-edge of the boundary.

After identifying the neighbors of vs and vt in the mesh Mk+1, we add the new vertex vt to the mesh Mk and reconnect the vertices in Nk+1(vt) to vt. Finally, we restore two faces, {vs,vl,vt} and {vr,vs,vt}, which are associated with the half-edges Hslt and Hrst , respectively. Using the half-edge data structure, the orientation of the newly created faces remains consistent with the rest of the mesh. When (vs,vt) lies on the boundary, only one face is restored.

However, as mesh Mk is generated from Mk+1 during edge collapse by merging vt into vs, we have

Nk+1(vs) ∩ Nk+1(vt) = {vl,vr} (1) Nk+1(vs) ∪ Nk+1(vt) − {vs,vt} = Nk(vs) (2)

which indicates we will only need to record vertex vl and vr. They split the vertex ring surrounding vs in Mk, where two halves of the vertices on the ring (top and bottom vertices in Fig. 2) belong to vt and vs respectively in Mk+1. However, an ambiguity arises in determining which half of the ring corresponds to vt’s neighbors in Mk+1. In the following, we demonstrate how the half-edge data structure [48] can be utilized to resolve this ambiguity.

Hence, with the above half-edge formulation, each vertex split operation can be uniquely defined by the selection of the target vertex vs and two neighbors vl and vr, as well as the position to place the new vertex vt.

Vertex Split Tokenization. Although vs, vl and vr only require references to existing vertices in the mesh, in practice we implement this reference as a raw prediction of each vertex to avoid a vocabulary size proportional to the sequence length. Thus, in the majority of cases, each vertex

Denote Hijk as a half-edge pointing from vi to vj, with vk being the third vertex in the associated face. As shown in

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

- Figure 6. Generation process for VertexRegen. M0 represents the predicted initial coarsest mesh, followed by subsequent meshes generated through the predicted vertex split sequence.

split is represented by a subsequence of 12 tokens (or 10 tokens when (vs,vt) is a boundary edge):

maintain a state machine and perform vertex split on the fly as new tokens are generated. After the coarsest mesh M0 is generated, we initialize the state. During each generation step, we enforce vs to be a vertex in the current mesh Mk, (vs,vl), (vs,vr) are valid edges (if vl and vr are not <nil>), and only one of vl and vr can be <nil>. Lastly, we decode vt, and perform vertex split with generated vs,vl,vr,vt information.

vsplit : [ <s x>, <s y>, <s z>, #vs <l x>, <l y>, <l z> | <nil>, #vl <r x>, <r y>, <r z> | <nil>, #vr <t x>, <t y>, <t z> ] #vt

where the special token <nil> signifies vl or vr does not exist, and only one of vl and vr is allowed to be <nil> for a valid vertex split subsequence. In Fig. 5 we illustrate the entire tokenization process.

##### 3.3. Conditional Generation

We consider generation conditioned on shapes. We adopt a pre-trained point-cloud encoder [58]. A LLaVa-style [21] projector is leveraged to project the condition features to the token embedding space. The projected feature tokens are prepended to the mesh sequence as the prefix. We supervise training with the next-token prediction objective while masking the loss for the prefix tokens.

The above progressive mesh serialization can then be trained with the standard next-token prediction target.

Vertex Split Decoding. The vertex split sequence must be consistent with the actual geometry to be valid. We

COV MMD 1-NNA JSD (%, ↑) (×103, ↓) (%) (↓)

Method Tokenization

MeshXL [2] Flattened Coords. 51.76 8.30 50.84 3.81 MeshAnything V2 [3] AMT 50.33 8.50 52.25 4.84 EdgeRunner [43] EdgeBreaker [33] 51.39 7.81 49.44 3.22 VertexRegen (Ours) Progressive 51.03 8.29 50.22 2.89

Table 1. Quantitative comparisons with state-of-the-art methods for unconditional mesh generation. Best results are bolded, second best are underlined. VertexRegen can generate meshes with comparable quality while enjoying the benefits of continuous level of detail.

MeshXL EdgeRunner MeshAnythingv2 VertexRegen (Ours)

140

22

50

90

120

20

45

###### 3MMD(×10,)

100

18

80

###### COV(%,)

40

###### 1-NNA(%)

JSD()

80

16

35

70

14

60

30

12

40

25

60

10

20

20

50

8

0

15

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

0 200 400 600 800

Face count constraint

Face count constraint

Face count constraint

Face count constraint

- Figure 7. Unconditional generation under face count constraints. VertexRegen achieves significantly better COV, MMD, and 1-NNA in early stages of generation.

### 4. Experiments

##### 4.1. Datasets

We pre-train our model and the baselines presented using two primary sources of meshes: Objaverse-XL1 [9] and an additional set of licensed artist-created meshes. Initially, we select meshes containing fewer than 8,000 faces without applying decimation. We use the CGAL [10] library to implement edge collapse and vertex split operations, where we modify the vertex placement to either vs or vt. We filter out non-manifold meshes and those that cannot be processed, which results in a final dataset of approximately 1.5M meshes with an average of 457 faces. For unconditional generation evaluation, we construct a highquality subset by further filtering with the alignment split of Objaverse-XL and with fewer than 800 faces. This yields approximately 18k meshes. We adopt a similar filtering process for shape-conditioned experiments.

We apply robust data augmentations to input meshes, including random shift within range [−0.1,0.1], random scaling between [0.9,1.1], random rotation by 0°, 90°, 180°, or 270°. During pre-training, we handle sequences exceeding the context window by discarding vertex split subsequences, while for other baselines, we truncate the tokenized sequence directly. The set for quantitative evaluation is chosen to avoid truncation for any method evaluated.

1We neither used assets from Sketchfab nor obtained any from the Polycam website.

##### 4.2. Implementation Details

VertexRegen and all other baselines are built upon the pretrained OPT-350M [56], with newly initialized token embeddings and position embeddings. We train our model on 64 H100 GPUs for approximately four days, with an effective batch size of 256. We use the AdamW [25] optimizer with a weight decay of 0.1, and betas (0.9,0.95). A cosine scheduler is employed, starting with an initial learning rate of 1 × 104 and gradually decreasing to 5 × 106. We clip the gradient norm to 1.0. During inference, we adopt the top-p sampling strategy with p = 0.95 by default.

##### 4.3. Results 4.3.1. Unconditional Generation

We follow evaluation protocols from prior works [1, 2, 37], employing point-cloud-based metrics to assess unconditional generation. Specifically, we sample the same number of meshes as the evaluation dataset and randomly sample 2,048 points per mesh. Coverage (COV) measures the diversity of the generated samples, where higher values indicate greater diversity. Minimum Matching Distance (MMD) computes the average distance from each reference sample to its nearest neighbor in the generated set, serving as a measure of generation quality, with lower values being preferable. 1-Nearest Neighbor Accuracy (1-NNA) evaluates both diversity and quality, where an optimal value is achieved at 50%. Additionally, we compute JensenShannon Divergence (JSD) to directly quantify the sim-

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

MeshXL

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

MeshAnything v2

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

EdgeRunner

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

VertexRegen (Ours)

- Figure 8. Qualitative comparison with state-of-the-art methods. VertexRegen is able to generate meshes with comparable quality to other baselines.

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 125]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

ℳ ℳ ℳ ℳ ℳ ℳ ℳ ℳ

[Figure 140]

[Figure 141]

[Figure 142]

ℳ

[Figure 143]

ℳ

[Figure 144]

ℳ

- Figure 9. Qualitative results on shape-conditioned generation on meshes held out from training. The first column shows the point cloud used as the condition, followed by the generation sequence progressing from the coarsest mesh M0 to the final output in the last column.

ilarity between the generated and reference distributions. We present results in Tab. 1. Our method can generate meshes of comparable quality to other state-of-the-art methods, with the added advantage of a continuous level of detail. We also show qualitative examples and their generation process in Fig. 6, and more generation results comparing with state-of-the-art methods in Fig. 8.

Generation with a face count constraint. We compare our method with the baselines under varying face count constraints. Our approach allows for generation to be paused at any point, accommodating the varying constraints. In contrast, to adapt to these constraints, baseline methods require either: (i) directly truncating the generation process, yielding incomplete intermediate meshes, or (ii) fine-tuning with an additional face count condition token.

For the direct truncation version, we evaluate unconditional generation metrics w.r.t. different face count constraints and present the results in Fig. 7. Additionally, we adopt a similar approach as proposed in EdgeRunner, where we split [1,800] range into 4 buckets and prepend the conditioning token to the mesh sequence. We compare VertexRegen and MeshXL fine-tuned with such conditioning scheme at face constraint 400, as shown in Tab. 2.

Due to its ability to generate meshes with a continuous level of detail from coarse to fine, VertexRegen effectively captures the overall structure even with very limited face counts, yielding significantly better COV, MMD, and 1NNA early in the process. As the face limit increases, all methods demonstrate improved results. Notably, for baselines such as EdgeRunner, while they enhance tokenization efficiency over MeshXL, they do not fundamentally address continuous level-of-detail generation and thus follow a similar trend to other baselines in the plot.

###### 4.3.2. Conditional Generation

We show qualitative examples of shape-conditioned generation in Fig. 9. We condition on 4,096 sampled points with normals from dense meshes. VertexRegen can generate a coarse M0, starting as simply as a tetrahedron, and progressively refine it by generating a sequence of vertex splits.

##### 4.4. Ablation Studies

Tokenization efficiency. We compute the tokenized sequence length of our proposed tokenization scheme (Sec. 3.2.1) on our dataset and report the average compression ratio relative to MeshXL (9 tokens per face) in Tab. 3. For VertexRegen, there are two primary sequence types: (i) a MeshXL-style sequence for the initial coarsest mesh M0 and (ii) sequences encoding vertex splits. The latter requires 12 tokens for two non-boundary faces or 10 tokens per boundary face. As a result, the highest compression is achieved when M0 is minimized and all vertex splits occur

COV MMD 1-NNA JSD (%, ↑) (×103, ↓) (%) (↓) |F| |V |

@400 Faces

VertexRegen 50.92 8.31 51.03 2.88 264 147 MeshXL (w/ FCC) 41.20 10.03 59.06 5.19 308 168

- Table 2. Comparison between VertexRegen and MeshXL with face count condition (FCC) when face count constraint is 400.

MeshXL

MeshAnything

EdgeRunner

VertexRegen

v2 w/ HE w/o HE Compression 1.0 0.46 0.47 0.73 0.89

- Table 3. Compression ratio of different tokenization schemes, VertexRegen with and without leveraging a half-edge (HE) structure.

Guided COV MMD 1-NNA JSD Decoding (%, ↑) (×103, ↓) (%) (↓) |F| |V | w/o 51.12 8.31 50.75 3.37 211 120 w/ 51.03 8.29 50.22 2.89 320 176

Table 4. Effects on geometry-guided decoding.

on non-boundary faces. On average, VertexRegen achieves a compression ratio of 0.73, approaching the theoretical limit of 0.67. The discrepancy arises from the overhead introduced by M0 (5.68% of all total tokenized length) and boundary vertex splits (3.64% of all vertex splits).

Without the half-edge data structure, identifying which half of the ring surrounding vs is associated with vs or vt in Mk+1 requires recording an additional vertex. This increases the token count to 15 per two non-boundary faces (or 13 per boundary face) and results in an average increase of 22% in tokenized sequence length.

Guided decoding. In Tab. 4, we ablate the effects of geometry-constrained decoding. As one step of the vertex split operation is dependent on all the preceding operations, predicting an invalid vertex split may break the chain and end the generation prematurely. With guided decoding, the model is able to generate longer sequences with more faces.

### 5. Conclusion

In this work, we introduced VertexRegen, a novel mesh generation framework that enables continuous levels of detail through a generative process based on vertex splits. Unlike conventional auto-regressive approaches that synthesize meshes in a partial-to-complete manner, VertexRegen reinterprets mesh generation as the reversal of edge collapse, providing an effectively anytime solution to mesh generation. Our experimental results demonstrate that VertexRegen achieves competitive performance compared to state-of-the-art methods while offering the unique advantage of halting generation at any stage to obtain meshes at different levels of detail.

### References

- [1] Antonio Alliegro, Yawar Siddiqui, Tatiana Tommasi, and Matthias Nießner. Polydiff: Generating 3d polygonal meshes with diffusion models. arXiv preprint arXiv:2312.11417,

2023. 2, 6

- [2] Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Billzb Wang, Jingyi Yu, Gang Yu, et al. Meshxl: Neural coordinate field for generative 3d foundation models. Advances in Neural Information Processing Systems, 37:97141–97166, 2024. 1, 2, 3, 4, 6
- [3] Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. arXiv preprint arXiv:2408.02555, 2024. 2, 6
- [4] Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Zhongang Cai, Lei Yang, Gang Yu, Guosheng Lin, and Chi Zhang. Meshanything: Artist-created mesh generation with autoregressive transformers. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2
- [5] Yun-Chun Chen, Vladimir Kim, Noam Aigerman, and Alec Jacobson. Neural progressive meshes. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–9, 2023. 3
- [6] Zhiqin Chen, Andrea Tagliasacchi, and Hao Zhang. Bsp-net: Generating compact meshes via binary space partitioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 2
- [7] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G. Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4456–4465, 2023. 2
- [8] Angela Dai and Matthias Niessner. Scan2mesh: From unstructured range scans to 3d meshes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 2
- [9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023. 6
- [10] Andreas Fabri and Sylvain Pion. Cgal: the computational geometry algorithms library. In Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, page 538–539, New York, NY, USA, 2009. Association for Computing Machinery. 6
- [11] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. In Advances In Neural Information Processing Systems, 2022. 1
- [12] Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, page 209–216, USA, 1997. ACM Press/Addison-Wesley Publishing Co. 2, 3

- [13] Thibault Groueix, Matthew Fisher, Vladimir G. Kim, Bryan C. Russell, and Mathieu Aubry. A papier-mˆach´e approach to learning 3d surface generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2
- [14] Antoine Gu´edon and Vincent Lepetit. Sugar: Surfacealigned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5354–5363, 2024. 2
- [15] Zekun Hao, David W Romero, Tsung-Yi Lin, and Ming-Yu Liu. Meshtron: High-fidelity, artist-like 3d mesh generation at scale. arXiv preprint arXiv:2412.09548, 2024. 1, 2
- [16] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large reconstruction model for single image to 3d. In The Twelfth International Conference on Learning Representations, 2024. 2
- [17] Hugues Hoppe. Progressive meshes. In Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques, page 99–108, New York, NY, USA, 1996. Association for Computing Machinery. 2, 3
- [18] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 2
- [19] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusionsdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12642–12651, 2023. 2
- [20] Peter Lindstrom and Greg Turk. Fast and memory efficient polygonal simplification. In Proceedings Visualization’98 (Cat. No. 98CB36276), pages 279–286. IEEE, 1998. 2
- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, pages 34892–34916. Curran Associates, Inc., 2023. 5
- [22] Minghua Liu, Chong Zeng, Xinyue Wei, Ruoxi Shi, Linghao Chen, Chao Xu, Mengqi Zhang, Zhaoning Wang, Xiaoshuai Zhang, Isabella Liu, Hongzhi Wu, and Hao Su. Meshformer: High-quality mesh generation with 3d-guided reconstruction model. In Advances in Neural Information Processing Systems, pages 59314–59341. Curran Associates, Inc., 2024. 2
- [23] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9970–9980,

2024. 2

- [24] William E. Lorensen and Harvey E. Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Proceedings of the 14th Annual Conference on Computer Graphics and Interactive Techniques, page 163–169, New York, NY, USA, 1987. Association for Computing Machinery. 2
- [25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6

- [26] David Luebke, Martin Reddy, Jonathan D. Cohen, Amitabh Varshney, Benjamin Watson, and Robert Huebner. Level of Detail for 3D Graphics. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 2002. 2
- [27] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 1
- [28] Felix Mujkanovic, Ntumba Elie Nsampi, Christian Theobalt, Hans-Peter Seidel, and Thomas Leimk¨uhler. Neural gaussian scale-space fields. ACM Transactions on Graphics (TOG), 43(4):1–15, 2024. 3
- [29] Charlie Nash, Yaroslav Ganin, SM Ali Eslami, and Peter Battaglia. Polygen: An autoregressive generative model of 3d meshes. In International conference on machine learning, pages 7220–7229. PMLR, 2020. 2
- [30] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 2
- [31] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 1
- [32] Jovan Popovi´c and Hugues Hoppe. Progressive simplicial complexes. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, page 217–224, USA, 1997. ACM Press/Addison-Wesley Publishing Co. 2
- [33] J. Rossignac. Edgebreaker: connectivity compression for triangle meshes. IEEE Transactions on Visualization and Computer Graphics, 5(1):47–61, 1999. 6
- [34] Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi Liao, and Andreas Geiger. Voxgraf: Fast 3d-aware image synthesis with sparse voxel grids. Advances in Neural Information Processing Systems, 35:33999–34011, 2022. 1
- [35] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems, 34:6087–6101,

2021. 2

- [36] Jaehyeok Shim, Changwoo Kang, and Kyungdon Joo. Diffusion-based signed distance fields for 3d shape generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20887– 20897, 2023. 2
- [37] Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19615–19625, 2024. 1, 2, 4, 6
- [38] Yawar Siddiqui, Tom Monnier, Filippos Kokkinos, Mahendra Kariya, Yanir Kleiman, Emilien Garreau, Oran Gafni, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and

- David Novotny. Meta 3d assetgen: Text-to-mesh generation with high-quality geometry, texture, and pbr materials. In Advances in Neural Information Processing Systems, pages 9532–9564. Curran Associates, Inc., 2024. 1
- [39] Towaki Takikawa, Joey Litalien, Kangxue Yin, Karsten Kreis, Charles Loop, Derek Nowrouzezahrai, Alec Jacobson, Morgan McGuire, and Sanja Fidler. Neural geometric level of detail: Real-time rendering with implicit 3d shapes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11358–11367,

2021. 3

- [40] Danhang Tang, Mingsong Dou, Peter Lincoln, Philip Davidson, Kaiwen Guo, Jonathan Taylor, Sean Fanello, Cem Keskin, Adarsh Kowdle, Sofien Bouaziz, et al. Real-time compression and streaming of 4d performances. ACM Transactions on Graphics (TOG), 37(6):1–11, 2018.
- [41] Danhang Tang, Saurabh Singh, Philip A. Chou, Christian Hane, Mingsong Dou, Sean Fanello, Jonathan Taylor, Philip Davidson, Onur G. Guleryuz, Yinda Zhang, Shahram Izadi, Andrea Tagliasacchi, Sofien Bouaziz, and Cem Keskin. Deep implicit volume compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3
- [42] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. In The Twelfth International Conference on Learning Representations, 2024. 2
- [43] Jiaxiang Tang, Zhaoshuo Li, Zekun Hao, Xian Liu, Gang Zeng, Ming-Yu Liu, and Qinsheng Zhang. Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. In The Thirteenth International Conference on Learning Representations, 2025. 1, 2, 6
- [44] Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffusion models for 3d shape generation. Advances in Neural Information Processing Systems, 35:10021–10039, 2022. 2
- [45] Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh models from single rgb images. In Proceedings of the European conference on computer vision (ECCV), pages 52–67,

2018. 2

- [46] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for highquality mesh. arXiv preprint arXiv:2404.12385, 2024. 2
- [47] Xinyue Wei, Fanbo Xiang, Sai Bi, Anpei Chen, Kalyan Sunkavalli, Zexiang Xu, and Hao Su. Neumanifold: Neural watertight manifold reconstruction with efficient and highquality rendering support. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pages 731–741, 2025. 2
- [48] Kevin J Weiler. Topological structures for geometric modeling (Boundary representation, manifold, radial edge structure). Rensselaer Polytechnic Institute, 1986. 4
- [49] Haohan Weng, Yikai Wang, Tong Zhang, CL Chen, and Jun Zhu. Pivotmesh: Generic 3d mesh generation via pivot vertices guidance. arXiv preprint arXiv:2405.16890, 2024. 2

- [50] Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, Tong Zhang, Shenghua Gao, and C.L. Philip Chen. Scaling mesh generation via compressive tokenization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11093– 11103, 2025. 2
- [51] Jiajun Wu, Chengkai Zhang, Tianfan Xue, Bill Freeman, and Josh Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2016. 1
- [52] Haiyang Xu, Yu Lei, Zeyuan Chen, Xiang Zhang, Yue Zhao, Yilin Wang, and Zhuowen Tu. Bayesian diffusion models for 3d shape reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10628–10638, 2024. 2
- [53] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

2024. 2

- [54] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3d shape generation. In Advances in Neural Information Processing Systems, 2022. 1
- [55] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG), 42(4):1–16, 2023. 2
- [56] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068,

2022. 6

- [57] Qingcheng Zhao, Xiang Zhang, Haiyang Xu, Zeyuan Chen, Jianwen Xie, Yuan Gao, and Zhuowen Tu. Depr: Depth guided single-view scene reconstruction with instance-level diffusion. arXiv preprint arXiv:2507.22825, 2025. 2
- [58] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, BIN FU, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. In Advances in Neural Information Processing Systems, pages 73969–

73982. Curran Associates, Inc., 2023. 5

- [59] Linqi Zhou, Yilun Du, and Jiajun Wu. 3d shape generation and completion through point-voxel diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 5826–5835, 2021. 1, 2

