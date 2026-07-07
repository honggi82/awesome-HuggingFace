# arXiv:2604.09132v2[cs.CV]2May2026

## Strips as Tokens: Artist Mesh Generation with Native UV Segmentation

RUI XU∗, The University of Hong Kong, Deemos Technology Co., Ltd. , China DAFEI QIN∗, The University of Hong Kong, Deemos Technology Co., Ltd., China KAICHUN QIAO, ShanghaiTech University, Deemos Technology Co., Ltd., China QIUJIE DONG, Shandong University, China HUAIJIN PI, The University of Hong Kong, China QIXUAN ZHANG‡, ShanghaiTech University, Deemos Technology Co., Ltd., China LONGWEN ZHANG, ShanghaiTech University, Deemos Technology Co., Ltd., China LAN XU†, ShanghaiTech University, China JINGYI YU, ShanghaiTech University, China WENPING WANG, Texas A&M University, USA TAKU KOMURA†, The University of Hong Kong, China

[Figure 1]

Fig. 1. Strips as Tokens (SATO) enables unified, high-quality artist mesh generation with native UV segmentation. Our strip-based tokenizer supports both triangle (left) and quad (right) meshes without retraining and automatically segments UV charts (side) during autoregressive generation.

∗Equal contribution. ‡ Project lead. † Corresponding authors.

Authors’ Contact Information: Rui Xu, The University of Hong Kong, Deemos Technology Co., Ltd. , China, ruixu1999@connect.hku.hk; Dafei Qin, The University of Hong Kong, Deemos Technology Co., Ltd., China, qindafei@connect.hku.hk; Kaichun Qiao, ShanghaiTech University, Deemos Technology Co., Ltd., China, qiaokch2022@ shanghaitech.edu.cn; Qiujie Dong, Shandong University, China, qiujie.jay.dong@ gmail.com; Huaijin Pi, The University of Hong Kong, China, huaijinpi@connect.hku. hk; Qixuan Zhang, ShanghaiTech University, Deemos Technology Co., Ltd., China, zhangqx1@shanghaitech.edu.cn; Longwen Zhang, ShanghaiTech University, Deemos Technology Co., Ltd., China, zhanglw2@shanghaitech.edu.cn; Lan Xu, ShanghaiTech University, China, xulan1@shanghaitech.edu.cn; Jingyi Yu, ShanghaiTech University, China, yujingyi@shanghaitech.edu.cn; Wenping Wang, Texas A&M University, USA, wenping@tamu.edu; Taku Komura, The University of Hong Kong, China, taku@cs.hku.hk.

This work is licensed under a Creative Commons Attribution-NonCommercialNoDerivatives 4.0 International License.

Recent advancements in autoregressive transformers have demonstrated remarkable potential for generating artist-quality meshes. However, the token ordering strategies employed by existing methods typically fail to meet professional artist standards, where coordinate-based sorting yields inefficiently long sequences, and patch-based heuristics disrupt the continuous edge flow and structural regularity essential for high-quality modeling. To address these limitations, we propose Strips as Tokens (SATO), a novel framework with a token ordering strategy inspired by triangle strips. By constructing the sequence as a connected chain of faces that explicitly encodes UV boundaries, our method naturally preserves the organized edge flow and semantic layout characteristic of artist-created meshes. A key advantage

© 2026 Copyright held by the owner/author(s). ACM 1557-7368/2026/7-ART75 https://doi.org/10.1145/3811286

of this formulation is its unified representation, enabling the same token sequence to be decoded into either a triangle or quadrilateral mesh. This flexibility facilitates joint training on both data types: large-scale triangle data provides fundamental structural priors, while high-quality quad data enhances the geometric regularity of the outputs. Extensive experiments demonstrate that SATO consistently outperforms prior methods in terms of geometric quality, structural coherence, and UV segmentation.

CCS Concepts: • Computing methodologies → Mesh models. Additional Key Words and Phrases: artist mesh generation, autoregressive, triangle strips, UV segmentation ACM Reference Format:

Rui Xu, Dafei Qin, Kaichun Qiao, Qiujie Dong, Huaijin Pi, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu, Wenping Wang, and Taku Komura. 2026. Strips as Tokens: Artist Mesh Generation with Native UV Segmentation. ACM Trans. Graph. 45, 4, Article 75 (July 2026), 18 pages. https://doi.org/10. 1145/3811286

- 1 Introduction

Artist-created meshes remain the dominant representation for 3D assets: they facilitate direct surface editing, provide precise control over connectivity and edge flow, and form the backbone of downstream stages such as deformation, simulation, and texture mapping. In contrast to meshes produced by generic remeshing algorithms, artist meshes usually adhere to consistent structural conventions. For instance, they often favor right-angled triangles over equilateral ones; triangles tend to align anisotropically with principal and secondary curvature directions; and sampling density increases in high-curvature regions while remaining sparse on flatter areas. These conventions profoundly impact rigging and animation quality, texturing workflows, and the long-term maintainability of production assets.

Generating high-quality 3D meshes that meet professional production standards is particularly challenging: a mesh must not only capture accurate high-fidelity geometry but also possess regular topology (i.e., clean edge flow) and semantic layouts (i.e., UV mapping) to be compatible with animation and rendering pipelines. Recently, autoregressive modeling has emerged as a promising alternative, treating mesh generation as a sequence prediction task [Chen et al. 2024a,c; Hao et al. 2024; Siddiqui et al. 2024]. By learning a distribution over discrete tokens, these methods attempt to capture geometric patterns directly. Early approaches typically rely on coordinate-based ordering [Chen et al. 2024a; Hao et al. 2024]. These methods directly tokenize the mesh by converting vertex coordinates into sorted triplets, each defining a tuple of quantized 3D coordinates. However, this fine-grained representation results in excessively long sequences. To address this, more recent methods [Weng et al. 2025; Xu et al. 2025; Zhao et al. 2025] employ patch-based tokenization that relies on Delaunay-style heuristics to organize the token order, thereby significantly shortening the sequence length. However, this approach inherently sacrifices the continuous surface curvature direction and coherent edge flow central to artist-created meshes, as Delaunay-style triangulation prioritizes mathematical compactness (e.g., maximizing minimum angles) over structural regularity. Fig. 2 highlights the distinct contrast between artist meshes, Delaunay-style meshes [Xu et al. 2024], and meshes obtained by Marching Cubes [Lorensen and Cline 1998].

[Figure 2]

### Artist Quad Artist Tri Delaunay Marching Cubes

Fig. 2. Artist meshes differ markedly from geometry-processed ones. Here we show quadrilateral and triangular meshes constructed by artists, as well as meshes created using geometric processing methods (such as Delaunay-style remeshing [Xu et al. 2024] and Marching Cubes [Lorensen and Cline 1998]).

Our key insight stems from the triangle strip, a classic concept representing a sequence of triangles that share vertices, thereby offering a memory-efficient mesh storage format. Each newly appended vertex deterministically forms a new triangle with the two preceding vertices, yielding a compact encoding that inherently couples connectivity with local surface continuity and directly exposes the flow-like structure of the original mesh. These properties create a strong structural alignment with artist-created topology, motivating the incorporation of the strip formulation into our tokenization strategy.

[Figure 3]

In this paper, we propose a novel framework, named Strips as Tokens (SATO), for generating artist-quality 3D meshes. Our core innovation lies in a strip-based tokenization strategy that organizes vertex ordering according to the topological definition of strips. Specifically, we construct the sequence as a connected chain of faces where each consecutive pair shares a common edge, a property that inherently aligns with the organized edge flow of artist meshes. Crucially, a key advantage of this formulation is that the unified vertex ordering enables a dual interpretation. Leveraging the topological fact that a quadrilateral naturally decomposes into two adjacent triangles, our framework allows the same token sequence to be decoded as either a triangle or a quadrilateral mesh. This flexibility facilitates the synergistic use of both data types: extensive triangle data provides fundamental structural priors, while high-quality quad data further enhances the geometric regularity of triangle outputs. Furthermore, we support native UV segmentation by extending the token vocabulary with specialized segmentation tokens. This mechanism encodes UV island boundaries directly into the token sequence without sacrificing compression efficiency, enabling the model to explicitly predict semantic partitioning.

We evaluate SATO across diverse datasets and tasks, observing consistent improvements over prior methods in geometric fidelity, structural coherence, and UV-aware generation. These results highlight the critical role of representation design in autoregressive

mesh generation, suggesting that artist-aligned tokenization is a key ingredient for making such models both learnable and practical. In summary, we make the following contributions:

- • Strip tokenization. We propose an artist-aligned strip-based serialization that preserves edge-flow coherence, achieves high compression efficiency, and makes the sequence structure easier for the model to learn.
- • Unified tri/quad decoding. A single token sequence supports both triangle and quad decoding, enabling triangle and quad data to synergistically reinforce each other through fine-tuning and bidirectional prior transfer.
- • Native UV segmentation. We explicitly encode UV island boundaries with dedicated tokens, making SATO the first autoregressive framework to simultaneously generate mesh geometry and UV chart partitions.

- 2 Related Work

- 2.1 3D Generation

A growing body of work synthesizes 3D assets utilizing implicit or hybrid representations, including signed distance fields, occupancy fields, and multi-view neural pipelines. Representative systems such as Wonder3D [Long et al. 2024], TRELLIS [Xiang et al. 2025b], TRELLIS.2 [Xiang et al. 2025a], CLAY [Zhang et al. 2024b], and Hunyuan3D-2.5 [Lai et al. 2025] achieve impressive end-to-end generation of textured geometry, often conditioned on text or images. More recently, several methods have shifted towards integrating stronger structural priors aimed at editability and decomposition. CraftsMan3D [Li et al. 2024] moves toward mesh-native outputs via 3D diffusion augmented by an (interactive) geometry refiner. OmniPart [Yang et al. 2025] and Ultra3D [Chen et al. 2025a] emphasize part-aware synthesis through semantic decoupling and part attention. BANG [Zhang et al. 2025] explores generative “exploded” dynamics for controllable asset division, and CAST [Yao et al. 2025] targets component-aligned reconstruction for multi-object scenes from a single image. Despite these advances, the final geometry often necessitates conversion to explicit meshes via iso-surfacing (commonly Marching Cubes [Lorensen and Cline 1998]) or related extraction procedures, which typically results in dense triangle meshes with connectivity that is poorly aligned with authoring conventions. Consequently, despite high visual fidelity, substantial post-processing is still required to obtain compact, editable, production-friendly meshes. Reliably producing truly artist-friendly meshes—clean topology with oriented regularities—therefore remains an open challenge, motivating our focus on artist mesh generation.

- 2.2 Mesh Generation

- 2.2.1 Triangle Mesh. Autoregressive mesh generation has emerged as a compelling paradigm for producing compact, artist-like triangle meshes by predicting discrete symbols in a causal order. MeshGPT [Siddiqui et al. 2024] is an early representative that learns a discrete vocabulary and generates meshes as sequences, demonstrating that transformer-style decoding can yield sharp yet compact triangulations. Subsequent work has expanded fidelity and

scale by refining tokenization and decoding strategies. MeshAnything [Chen et al. 2024b] and MeshAnythingV2 [Chen et al. 2024c] propose adjacency-aware tokenizations to shorten sequences and improve controllability, while MeshXL [Chen et al. 2024a] explores coordinate-field-stylerepresentations for large-scale sequential modeling. EdgeRunner [Tang et al. 2024] further improves token efficiency with classical-mesh-inspired serialization, and introduces an autoregressive auto-encoder that maps variable-length meshes into compact latent codes. Concurrently, network-oriented efforts address the computational bottlenecks of long-context decoding. Meshtron [Hao et al. 2024] scales triangle mesh generation to substantially higher face counts via an hourglass design with slidingwindow inference, and iFlame [Wang et al. 2025c] interleaves full and linear attention to reduce cost while preserving quality.

Beyond architectural innovations, recent approaches have significantly improved performance through sequence compression, distributedprocessing,andoptimizedserialization strategies. BPT [Weng

et al. 2025] reduces context length via blocked and patchified representations, enabling higher-resolution geometry under limited sequence budgets, and DeepMesh [Zhao et al. 2025] extends such compressed representations with preference optimization to better match human judgments. TreeMeshGPT [Lionar et al. 2025] proposes a dynamic tree-based sequencing scheme that adapts nexttoken prediction to mesh growth. Nautilus [Wang et al. 2025b] studies locality-aware encoding and decoding to better preserve local manifold structure under compression. FastMesh [Kim et al. 2025] further decouples geometry and connectivity by generating vertices autoregressively and then predicting adjacency in parallel with a bidirectional transformer, enabling substantially faster artistic mesh synthesis. MeshRipple [Lin et al. 2025] expands meshes from a dynamically maintained frontier using topology-aligned BFS tokenization and a global memory mechanism, improving structural completeness by retaining long-range topological context. MeshRFT [Liu et al. 2025b] targets post-training quality via face-level masked preference optimization with topology-aware scoring, enabling localized corrections while maintaining global coherence. MeshMosaic [Xu et al. 2025] increases the number of triangle faces by adopting a part-based, local-to-global processing strategy with explicit interaction awareness across parts. MeshSilksong [Song et al. 2025] adopts a weaving-style serialization that visits each vertex only once, substantially shortening sequences while promoting manifold, watertight meshes with consistent normals.

Collectively, these works improve scalability and structural fidelity through advances in sequencing, decoding, and post-training objectives. Despite these advances, most tokenizations remain fundamentally triangle-centric, in that they rely on individual triangles, or their immediate adjacency, as the primary unit of generation. Higher-order organization, continuous surface runs, stable edge flow, and coherent region growth must then emerge implicitly from many local triangle-level decisions. This misalignment hinders the faithful capture of mid-level regularities that artists intentionally embed in production meshes. SATO bridges this gap by elevating triangle strips to the token level, providing a compact primitive that couples connectivity with local continuity and encourages coherent surface progression. It is worth noting that triangle strip decomposition has a long history in classical computer graphics, where

various heuristic and graph-based stripification algorithms have been developed for efficient rendering of both triangle [Porcu and Scateni 2003; Vaněček and Kolingerová 2007; Xiang et al. 1999] and quadrilateral [Vanecek et al. 2005] meshes.

- 2.2.2 Quad Mesh. Compared to triangle meshes, quad-dominant meshes are often favored in production due to their regular edge flow and favorable deformation behavior. However, generating quads directly presents significant challenges, as it necessitates maintaining higher-order consistency beyond local triangulation decisions. A common strategy is therefore to first synthesize a triangle mesh and then promote quad-compatibility through scoring or postprocessing. Mesh-RFT [Liu et al. 2025b] encourages quad-friendly topology via preference optimization with topology-aware rewards computed after tri-to-quad merging. Conversely, QuadGPT [Liu et al. 2025a] targets quad dominance more directly by natively modeling mixed triangle and quad faces within a unified sequence, subsequently refining topology through topology-aware post-training.

Beyond quad-dominant meshes, pure-quad meshes impose even stricter regularity requirements, particularly for edge-aligned flow and globally consistent structure. NeurCross [Dong et al. 2025b] introduces a proxy surface to implicitly align quad edge directions with principal curvature directions, but it remains computationally expensive. CrossGen [Dong et al. 2025a] improves efficiency and generalization by training a VAE to enable fast synthesis of highquality pure-quad meshes. Nevertheless, these pipelines still depend heavily on a well-structured cross field as an explicit guiding signal, which makes fully end-to-end generation of production-quality pure-quad meshes from raw inputs difficult. In contrast, SATO circumvents this dependency by directly modeling strip-consistent edge flow, enabling the one-step generation of high-quality quad meshes.

- 2.3 UV Segmentation

Production-ready meshes must support not only geometry and connectivity, but also efficient texturing workflows. Many systems in the broader 3D generation pipeline output textured assets, such as Wonder3D [Long et al. 2024], CLAY [Zhang et al. 2024b], and Hunyuan3D-2.5 [Lai et al. 2025]; however, UV unwrapping and seam placement are typically relegated to downstream stages and handled by separate parameterization and atlasing tools, rather than being integrated as constraints maintained during mesh synthesis. Similarly, most autoregressive mesh generators prioritize geometry and topology, and either omit UVs entirely or re-segment charts after generation. This decoupling disrupts artist-style seam structure and adds nontrivial post-processing overhead. In contrast, SATO incorporates UV segmentation as an intrinsic part of the generative representation, by organizing strip sequences within UV regions and inserting explicit region delimiters, so that UV boundaries can be preserved and recovered during generation.

Recent learning-based methods improve UV unwrapping by explicitly learning seam placement. SeamGPT [Li et al. 2025] and ArtUV [Chen et al. 2025b] follow a production-inspired pipeline, where a GPT-based seam predictor proposes semantically meaningful cuts and a learned module refines an initialized UV map.

However, these approaches remain multi-stage and initializationdependent, and often lack explicit optimization for global packing efficiency. Nuvo [Srinivasan et al. 2025] models UVs as a neural field and optimizes them over visible surface points, which reduces fragmentation on challenging geometry. FAM [Zhang et al. 2024a] similarly learns global free-boundary parameterization directly on surface points in an unsupervised manner, reducing reliance on high-quality meshes. PartUV [Wang et al. 2025a] leverages semantic part decomposition to reduce chart fragmentation under a distortion budget, coupling charting with parameterization and packing. While robust on generated meshes, it depends on the quality of part segmentation and introduces additional stages, which compromises the end-to-end nature of the pipeline and can become unstable when parts are ambiguous.

3 Preliminaries

- 3.1 Triangle Strips

A triangle strip [Isenburg 2001] is a compact encoding of a connected sequence of triangles where adjacent triangles share an edge. Instead of storing triangles independently (as a triangle list), a strip represents triangles by an ordered vertex sequence S = (𝑣1,𝑣2, . . .,𝑣𝑚), which implicitly defines 𝑚 − 2 triangles:

𝑓𝑖 = (𝑣𝑖, 𝑣𝑖+1, 𝑣𝑖+2), 𝑖 = 1, . . .,𝑚 − 2. (1)

Consecutive triangles 𝑓𝑖 and 𝑓𝑖+1 share the edge (𝑣𝑖+1,𝑣𝑖+2), so each new triangle introduces only one new vertex index, yielding a highly efficient representation for rendering and storage.

[Figure 4]

Because the vertex order alternates along the strip, the face orientation flips between neighboring triangles. To maintain a consistent orientation (e.g., counterclockwise), one com-

monly reorders every other triangle as 𝑓𝑖′ = (𝑣𝑖,𝑣𝑖+2,𝑣𝑖+1) for even 𝑖 (or equivalently toggles a parity flag during decoding). In prac-

tice, a mesh can be decomposed into a set of strips. From artists’ perspective, this representation aligns well with how surfaces are commonly laid out and created: modelers often build geometry by extending a boundary and adding faces incrementally, forming long, coherent strips of triangles with stable local connectivity. Such striplike structure preserves a clear sequential order and strong local adjacency, which makes it easier to capture (and later regenerate) the mesh flow and regularity of artist meshes compared to treating faces as an unordered triangle list.

- 3.2 Autoregressive Mesh Generation Framework

FollowingMeshGPT[Siddiquietal.2024]andfollow-upworks[Chen

et al. 2024c; Weng et al. 2025], mesh generation is formulated as a conditional sequence modeling task. Given a 3D mesh, the process begins with a tokenizer that serializes the complex geometric and topological data into a discrete 1D sequence of tokens, denoted as T = (𝑡1,𝑡2, . . .,𝑡𝐿), where 𝐿 indicates the sequence length. This tokenization step bridges the gap between irregular 3D structures and standard sequence models.

[Figure 5]

- Fig. 3. The Pipeline of SATO. SATO uses a strip-based tokenizer to encode/decode both triangle and quad meshes as a unified discrete sequence. Conditioned on an input point cloud, a learnable point-cloud encoder cross-attends to the core Hourglass Transformer, which autoregressively generates token sequences that are decoded into triangle or quad meshes with native UV segmentation.

To generate meshes, a Transformer-based decoder (GPT) learns to predict the sequence autoregressively. Given a condition c (e.g., a point cloud), the model is trained to maximize the likelihood of the next token 𝑡𝑖 based on the preceding context 𝑡<𝑖. The training objective is to minimize the standard cross-entropy loss over the dataset:

Finally, we discuss our three-stage training strategy, covering largescale pretraining on triangles to fine-tuning on high-quality quad meshes (Sec. 4.4).

- 4.1 Hierarchical Geometry Quantization

We represent an artist mesh M as a tuple (V, F), where V is a set of 𝑁 vertices and F is a set of 𝑀 faces. Each vertex 𝑣 ∈ V is defined by its 3D coordinates. In professional modeling workflows, these vertices are organized into polygons that follow specific structural rules, predominantly triangles and quadrilaterals. Accordingly, each face 𝑓 ∈ F is defined as an ordered sequence of vertex indices, where the face degree |𝑓 | ∈ {3, 4} denotes a triangle or a quadrilateral, respectively.

To bridge the gap between continuous geometric space and discrete tokens, we quantize the vertex coordinates onto a 5123 voxel gridfollowingthethree-levelhierarchical strategy in DeepMesh [Zhao

et al. 2025]. Specifically, the mesh is normalized into a unit cube and each vertex is decomposed into a hierarchical tuple (𝑐1,𝑐2,𝑐3) corresponding to 43, 83, and 163 resolution levels (left of Fig. 4). Here 𝑐1 ∈ C1𝑔𝑒𝑜 identifies the coarsest grid cell, while {𝑐2,𝑐3} specify the local relative position of the vertex within its respective parent cell from the previous level. Together, this strategy provides the full 5123

precision, with C1𝑔𝑒𝑜 serving as the coarsest coordinate codebook. At this stage, the mesh geometry is fully discretized into a set of

hierarchical tuples, but they remain unordered and detached from the topological faces F.

- 4.2 Strip-based Serialization

∑︁𝐿

log𝑝(𝑡𝑖 | 𝑡<𝑖, c;𝜃) (2)

L = −

𝑖=1

where 𝜃 represents the learnable parameters of the Transformer.

- 4 Method

Given a set of 3D points as conditioning input, our goal is to generate an artist-style mesh with organized UV segmentation. To achieve this, we propose Strips as Tokens (SATO), a generative framework based on a unified strip-based representation. Our core contribution includes a serialization scheme that embeds macro-structural semantic cues like UV island boundaries into the token stream, and a stride-aware decoding protocol that allows the same model to generate both triangle and quadrilateral meshes. The overview of the proposed framework is illustrated in Fig. 3.

We first describe our hierarchical geometry quantization process, which maps 3D coordinates into a compact discrete vocabulary (Sec. 4.1). Next, we introduce our strip-based serialization, where meshes are converted into long, contiguous vertex streams with embedded UV transition markers (Sec. 4.2). Then, Sec. 4.3 details our multi-topology interpretation protocol, which allows the recovered sequence to be adaptively decoded as either triangle or quad meshes.

With the geometry discretized into hierarchical tokens, the remaining challenge is to establish a deterministic ordering that linearizes

Algorithm 1 Unified Strip Extraction (SATO) Require: Mesh faces F, Stride parameter 𝛿 (𝛿 = 1 for Triangle,

𝛿 = 2 for Quad).

Ensure: A set of extracted strips S = {S1, . . ., S𝑘}. Initialization.

- 1: Build Edge-to-Face adjacency map E2F.
- 2: Initialize visited[𝑓 ] ← false for all 𝑓 ∈ F.
- 3: Initialize strip list S ← [ ]. Extraction Loop.
- 4: while ∃𝑓 ∈ F s.t. visited[𝑓 ] = false do
- 5: Start new strip. S𝑐𝑢𝑟𝑟 ← [ ].
- 6: Pick lowest unvisited face 𝑓𝑠𝑒𝑒𝑑.
- 7: v ← GetVertices(𝑓𝑠𝑒𝑒𝑑).
- 8: if 𝛿 = 2 then
- 9: Swap the last two vertices of v.
- 10: end if
- 11: Append v to S𝑐𝑢𝑟𝑟.
- 12: Mark visited[𝑓𝑠𝑒𝑒𝑑] ← true.
- 13: Define boundary edge 𝑒𝑓 𝑟𝑜𝑛𝑡 ← (v[−2], v[−1]). // Zipper-like growth
- 14: while true do
- 15: 𝑓𝑛𝑒𝑥𝑡 ← NextFace(E2F,𝑒𝑓 𝑟𝑜𝑛𝑡, visited).
- 16: if 𝑓𝑛𝑒𝑥𝑡 = ∅ then
- 17: break ⊲ Hit boundary or visited face.
- 18: end if
- 19: v𝑛𝑒𝑤 ← GetNewVertices(𝑓𝑛𝑒𝑥𝑡,𝑒𝑓 𝑟𝑜𝑛𝑡). ⊲ Returns 1 v if 𝛿 = 1, pair of swapped v if 𝛿 = 2.
- 20: Append v𝑛𝑒𝑤 to S𝑐𝑢𝑟𝑟.
- 21: Mark visited[𝑓𝑛𝑒𝑥𝑡] ← true.
- 22: Update 𝑒𝑓 𝑟𝑜𝑛𝑡 based on v𝑛𝑒𝑤.
- 23: end while
- 24: Append S𝑐𝑢𝑟𝑟 to S.
- 25: end while
- 26: return S.

the mesh topology. Inspired by the concept of triangle strips [Isenburg 2001], we propose to serialize the mesh into a sequence of vertices guided by the structural “flow” of adjacent faces. A strip is defined as a connected sequence of faces where each consecutive pair shares a common edge, a property that aligns perfectly with the organized edge-flow of artist meshes. By traversing the mesh through these shared-edge boundaries, we convert the graph-like connectivity of F into a coherent vertex stream T.

4.2.1 Strip Extraction. We construct strips via a systematic “zipperlike” growth procedure that extracts topological paths from the input faces F. As detailed in Alg. 1, we first build an edge-toface adjacency map and initialize all faces as unvisited. To extract a strip, we pick the first unvisited face (e.g., the faces were sorted by the lowest coordinate) as a seed and append its vertices to the output sequence. The three vertices of the seed face are sorted by their coordinates, and the edge formed by the last two vertices in this order is designated as the initial boundary edge, which deterministically dictates the growth direction of the strip.

Starting from the seed face, the strip grows by repeatedly traversing across the current boundary edge to its adjacent unvisited face. This traversal is governed by a topology-specific stride 𝛿: in triangle mode (𝛿 = 1), each step crosses a boundary edge to add a single new vertex, whereas in quad mode (𝛿 = 2), each step crosses the edge to introduce a pair of new vertices. This unified traversal ensures that both mesh types follow an identical “grow-by-appending” logic, where each step expands the sequence to induce a new face. To maintain this structural alignment, we enforce a consistent vertex ordering within each quadrilateral by swapping the last two indices of each face. As illustrated in the inset figure, this swap ensures that quad strips follow the same forward-moving order as triangle strips. The quad token sequence in inset figure (a) is identical to the triangle token sequence in inset figure (b). By aligning the winding order in this manner, we achieve a structural consistency where both mesh types can be generated via a unified autoregressive flow.

[Figure 6]

The growth of a strip terminates when the current boundary edge either lies on the mesh boundary or connects only to faces that have already been visited. Once a strip reaches such a dead end, we select the next available unvisited face (following the same coordinate-based priority) as a new seed to initiate a subsequent strip. This process repeats iteratively until the entire face set F is covered, effectively decomposing the mesh into a collection of disjoint strips {S1, S2, . . ., S𝑘}. Crucially, this decomposition establishes a deterministic global vertex ordering. We deliberately chose this greedy, lowest-coordinate-first strategy because it yields a fixed, spatially coherent traversal pattern that the network can learn easily. A globally optimized strip decomposition might reduce the total number of strips, but it could introduce erratic seed face locations and inconsistent traversal patterns, which hinders optimization in practice. By concatenating these strips and mapping each vertex to its corresponding hierarchical code, we could transform the complex mesh graph into a token sequence.

- 4.2.2 Strip Transition. To serialize these disjoint strips into a unified token stream, we require a mechanism to explicitly delineate topological boundaries. Following the codebook expansion strategy introduced in DeepMesh [Zhao et al. 2025], we distinguish strip boundaries by expanding the vocabulary of the coarsest codebook

level C1𝑔𝑒𝑜. Specifically, we augment the coarsest codebook level C1𝑔𝑒𝑜 with a separate parallel set of tokens, denoted as C1𝑡. Each token in this auxiliary set corresponds to the same spatial grid position as its standard counterpart but serves a distinct semantic role. During sequence construction, the first vertex of every new strip is encoded using these specialized C1𝑡 tokens. In this way, we effectively embed the “start-of-strip” signal directly into the geometric sequence. This avoids the need for inserting separate delimiter tokens, ensuring that the explicit boundary definition does not increase the overall sequence length.

- 4.2.3 UV Segmentation. Beyond strip connectivity, our tokenizer natively supports UV segmentation to preserve the macro-structural

[Figure 7]

- Fig. 4. Mesh tokenization with prefix sharing. We use a three-level

hierarchical coordinate (𝑐1,𝑐2,𝑐3) with prefix sharing to compress token sequences following DeepMesh [Zhao et al. 2025], repeated prefixes between consecutive vertices are omitted. To mark a strip transition, we introduce a special top-level token vocabulary 𝑐1∗ (red), which is distinct from 𝑐1 but serves the same role. Whenever 𝑐1∗ appears, the prefix sharing state is reset, starting a new prefix context.

[Figure 8]

UV Map Texture UV Segment Artist Mesh

- Fig. 5. Artist-created meshes with UV chart partitions. We split artist meshes into UV parts and let SATO traverse all triangles within one part before a UV segmentation transition to the next part, enabling native UV segmentation during generation.

organization of artist meshes. As illustrated in Fig. 5, we partition the mesh faces into disjoint groups based on their UV islands and impose a deterministic traversal order across these islands (bottom to up). Within each island, the strip-based encoding proceeds as usual, with the constraint that the next seed face must be selected from the current island until all its constituent faces are exhausted. To distinguish these semantic boundaries, we further expand the coarsest codebook C1𝑔𝑒𝑜 with an additional set C1𝑢𝑣, which denote the completion of a UV island and a transition to the next UV segmentation. Notably, the C1𝑢𝑣 strictly subsumes the function of C1𝑡: it signals both the termination of a strip and a higher-level switch between distinct UV charts. By injecting these artist-preferred semantic cues into the sequence, we enable the model to learn not only the surface geometry but also the high-level layout intent inherent in professional mesh modeling. Note that our model learns only the UV chart partitioning (i.e., which faces belong to which island), not the UV coordinates themselves; a standard unwrapping algorithm in Blender [Blender 2025] is applied afterward to compute the actual 2D parameterization from the predicted segmentation.

Consequently, the final sequence T remains compact, with each vertex 𝑣𝑖 encoded by its hierarchical tokens (𝑐𝑖,1,𝑐𝑖,2,𝑐𝑖,3). While higher-level tokens remain standard, the first-level token 𝑐𝑖,1 is drawn from an augmented vocabulary C𝑖∗ that integrates spatial,

[Figure 9]

- (b)
- (c)

(a)

Fig. 6. Unified representation of triangle and quad using strips. Triangle strips may locally “turn” under edge flips (a). In contrast, quad strips avoid this ambiguity (b), as each step admits only a single forward direction. Moreover, sequences tokenized from a quad mesh can be decoded into triangles while still preserving high quality (c). Note that the quad token sequence of (b) is totally the same as the triangle token sequence of (c).

structural, and semantic information: 𝑐𝑖,1 ∈ C1∗ = C1𝑔𝑒𝑜

(3)

#### ∪ C1𝑡

#### ∪ C1𝑢𝑣

Strip Transition

UV Segmentation

Standard

Under this scheme, a typical vertex stream T appears as:

, . . ., (𝑐𝑡𝑗,1,𝑐𝑗,2,𝑐𝑗,3)

, . . ., (𝑐𝑘,𝑢𝑣1,𝑐𝑘,2,𝑐𝑘,3)

T = . . ., (𝑐𝑖,1,𝑐𝑖,2,𝑐𝑖,3)

.

Standard

New Strip

New UV Island

(4) This unified format ensures that the serialization is natively aware of the mesh’s macro-structural organization.

Finally, inspired by DeepMesh [Zhao et al. 2025], we employ a prefix sharing strategy to minimize sequence length by exploiting the inherent spatial continuity within each strip. Consecutive vertices often share identical coarse locations 𝑐1 or parent cells 𝑐2. In such cases, we omit the redundant prefixes. For instance, if a vertex 𝑣𝑖+1 and its preceding vertex 𝑣𝑖 share the same 𝑐1 and 𝑐2 codes, the original sequence [(𝑐𝑖,1,𝑐𝑖,2,𝑐𝑖,3), (𝑐𝑖+1,1,𝑐𝑖+1,2,𝑐𝑖+1,3)] is compressed into [𝑐𝑖,1,𝑐𝑖,2,𝑐𝑖,3,𝑐𝑖+1,3]. In this case, the representation of 𝑣𝑖+1 is reduced from a three-token tuple to a single token. Crucially, structural tokens from C1𝑡 and C1𝑢𝑣 serve as absolute synchronization points. They are never compressed and implicitly force a reset of the sharing context, ensuring that topological transitions remain explicit and unambiguous to the model. Fig. 4 shows a clear toy example to help understand how the token sequence is obtained. Empirically, on our test set the token distribution across levels is 𝑐1: 20.7%, 𝑐2: 35.0%, 𝑐3: 44.3%, confirming that prefix sharing effectively compresses the majority of vertices to one or two tokens.

4.2.4 Properties of the Representation. The proposed strip-based serialization offers three fundamental advantages for mesh generative modeling.

First, by capturing the long-range structural “flow” typical of artist meshes, our representation provides a stronger inductive bias for learning regular topology and consistent connectivity compared to randomized or patch-based orderings.

Second, our unified stride-based formulation enables topological synergy between disparate mesh types; by linearizing triangles and quadrilaterals into the same vertex stream, we allow the model to share geometric priors across different domains. Furthermore, training on quad meshes can improve the quality of triangle-strip

- Table 1. Comparison of vocabulary size and average compression rate. The compression rate is computed as the token sequence length divided by (face count × 9).

Metrics BPT DeepMesh SATO

Vocab Size ↓ 40960 4736 4800 Comp Rate ↓ 0.228 0.330 0.283

[Figure 10]

### BPT DeepMesh Ours

- Fig. 7. Different face ordering defined by other methods. BPT [Weng et al. 2025] and DeepMesh [Zhao et al. 2025] traverse local fan-/disk-shaped neighborhoods, i.e., triangles rotate around a vertex, which triggers patch transitions more frequently. In contrast, our strip-based ordering can, in principle, extend arbitrarily long.

sequences. As shown in Fig. 6 (a), triangle strips on certain artist meshes may exhibit occasional “turns.” While such turns will not affect the quality of the generation model, they introduce additional ordering variability, forcing the model to learn stronger traversal priors. Quad strips largely avoid this issue (Fig. 6 (b)): large-angle turns are rare within the quadrilateral zone, so the traversal naturally progresses forward and rarely produces large-angle bends.

Third, our approach significantly optimizes encoding efficiency relative to patch-based methods like DeepMesh [Zhao et al. 2025]. As illustrated in Fig. 7, patch-based methods partition the mesh into numerous small fragments (typically 5–7 faces each), each necessitating a transition token that resets the prefix sharing context. In contrast, our decomposition into long, contiguous strips drastically reduces the frequency of these resets, allowing spatial continuity to persist over larger spans and effectively amortizing the transition overhead to produce a more concise serialization. We report the average compression ratio achieved by different tokenizers on 100 randomly sampled meshes from Objaverse [Deitke et al.

- 2023] in Table 1. Despite using a slightly larger vocabulary than DeepMesh [Zhao et al. 2025] (the additional tokens are used to support UV segmentation), our tokenizer attains a noticeably higher compression ratio, indicating a more efficient sequence representation under the same discrete budget.

- 4.3 Topology-Specific Decoding The conversion from the token sequence T back to a mesh M is

governed by a deterministic decoding protocol. For each vertex 𝑣𝑖, the decoder first restores the full hierarchical coordinates: if the input is a compressed residual𝑐𝑖,3, it is prepended with the (𝑐𝑖,1,𝑐𝑖,2) prefix cached from the preceding vertex following DeepMesh [Zhao et al. 2025]. The global mesh structure is managed by structural markers embedded within the C1∗ vocabulary. Specifically, C1𝑡 signals the termination of the current strip and the initiation of a new one, while C1𝑢𝑣 indicates a transition between disjoint UV islands. Upon detecting either marker, the decoder immediately resets both the coordinate cache and the topological frontier, ensuring that the subsequent vertices are interpreted as a fresh seed face for the next segment.

The primary distinction of our protocol is its support for multitopology recovery via an adjustable vertex stride 𝛿 ∈ {1, 2}. Leveraging the consistent vertex ordering enforced during the encoding stage, the decoder can interpret a single geometric stream through disparate topological rules. In triangle-mesh mode (𝛿 = 1), each successive vertex 𝑣𝑖+2 completes a face 𝑓𝑖 = (𝑣𝑖,𝑣𝑖+1,𝑣𝑖+2). In quadrilateral-mesh mode (𝛿 = 2), the decoder processes vertices in pairs; for any two newly generated vertices (𝑣2𝑖+2,𝑣2𝑖+3), it assembles a quad face𝑞𝑖 = (𝑣2𝑖,𝑣2𝑖+1,𝑣2𝑖+3,𝑣2𝑖+2). For example, a six-vertex sequence (𝑣0, . . .,𝑣5) is interpreted as four triangles under 𝛿 = 1, or as two quadrilaterals 𝑞0 = (𝑣0,𝑣1,𝑣3,𝑣2) and 𝑞1 = (𝑣2,𝑣3,𝑣5,𝑣4) under 𝛿 = 2. This unified interpretive framework enables the same autoregressive model to learn shared geometric priors across heterogeneous datasets by simply toggling the decoding stride. Notably, switching between triangle and quad output requires no special tokens or architectural changes; the user simply sets 𝛿 at inference time. In quad mode, if a strip contains an odd number of vertices after the seed face, the final unpaired vertex is decoded as a triangle. The detokenizer also strictly discards structurally invalid markers (e.g., consecutive C1𝑡 or C2 tokens without intervening geometry tokens); in practice, this failure mode has never been observed with our trained model. Additionally, vertices from different strips that share the same quantized coordinates within a UV region are welded during decoding to ensure a connected mesh.

4.4 Training with SATO

The training pipeline of SATO is organized into three stages: (i) largescale triangle-mesh pretraining, (ii) UV-segmentation post-training, and (iii) quad-mesh fine-tuning.

- 4.4.1 Model Architecture and Optimization. SATO uses a 0.5B parameter autoregressive hourglass transformer backbone, which has been shown to be well-suited for mesh generation [Hao et al. 2024]. Specifically, the transformer consists of 21 layers with 8 attention heads and 1024-dimensional embeddings. For point cloud conditioning, instead of using a pretrained and frozen point cloud VAE encoder as in prior work [Zhao et al. 2025], we adopt the same VAE architecture as Hunyuan3D [Lei et al. 2025] but train it from scratch after reducing the layers and token length to better align with inputs; the resulting encoder has roughly 0.27B parameters. Concretely, we reduce the decoder from 16 layers to 12 layers and the condition token count from 4096 to 1024 to better match our point cloud inputs. We optimize the model using the standard cross-entropy loss. Due to the high resolution of our tokenization, mesh sequences often exceed the attention window of the Transformer. To address this, we adopt the truncated-window training strategy [Hao et al. 2024; Zhao et al. 2025] with 9K window size, where the model is trained on overlapping segments of the full sequence. Specifically, during each training iteration, we randomly select a contiguous subsequence of 9K tokens from the full mesh token stream as the training input. This allows SATO to maintain local geometric coherence while scaling to complex meshes with large token counts.
- 4.4.2 Data Processing. Removing noisy or low-quality samples from millions of training shapes is essential to mesh generation training. We apply the following filtering pipeline to construct our

[Figure 11]

- Fig. 8. The gallery of SATO illustrates our model’s outputs across three tasks. From bottom to top, it shows triangular mesh generation, shape generation with UV segmentation, and quadrilateral mesh generation. SATO supports all three tasks within a single framework and achieves compelling results on each of them.

dataset. For all meshes, we first discard non-manifold models and merge duplicate vertices. We then keep shapes whose face count lies in [500, 16000] and whose vertex-to-face ratio does not exceed 1.0; models violating the latter criterion are often highly fragmented and close to a triangle soup. All data are randomly rotated along the Z-axis at four angles [0, 90, 180, 270] before tokenization. For UV-related training, we additionally validate UV segmentation and keep only models whose number of UV islands lies in [10, 300] to avoid excessively fragmented UV layouts.

between mesh sequences and the conditioning input, and (b) discover higher-level semantic structure induced by UV islands (including predicting segment boundaries and handling inter-island transitions). These objectives interact and often lead to slow convergence or degenerate solutions during our test (Sec. 5.4.2). To mitigate this, we perform a second-stage post-training where we initialize from the pretrained triangle model and then introduce the UV segmentation tokenization described in Sec. 4.2.3. In this stage, the model mainly adapts to the newly injected segmentation tokens (e.g., C1𝑢𝑣) and the corresponding inter-island transition rules, while retaining the learned geometric and conditioning alignment from Stage I. This strategy can significantly accelerate the convergence of the UV segmentation module and improve its performance.

- 4.4.3 Training. Then we train our SATO network with three stages.

- Stage I: Triangle Mesh Pretraining. We first train the backbone Transformer together with the base SATO tokenizer without UV segmentation on a large corpus of triangle mesh datasets. This stage establishes strong geometric priors, including local strip continuation patterns and the alignment between mesh tokens and the conditioning point clouds. Empirically, such priors are crucial for stable autoregressive training.
- Stage II: UV Segmentation Post-Training. Directly training a UV segmentation model from scratch is challenging. In early training, the model must simultaneously (a) learn the basic correspondence

Stage III: Quad Mesh Fine-tuning. High-quality quad meshes are substantially less abundant than triangle meshes, making it impractical to train an autoregressive quad generator from scratch at scale. Thanks to the compatibility of our strip-based representation, we fine-tune the model initialized from Stage I/II using the quad-strip decoding rule in Sec. 4.2.1. This transfers the majority of the learned priors from the triangle domain and only requires a relatively small quad-mesh dataset to adapt the model to quad-specific connectivity

[Figure 12]

- Fig. 9. Overview of our test dataset. 250 models selected from the ShapeNet [Chang et al. 2015], Thingi10K [Zhou and Jacobson 2016] and Objaverse [Deitke et al. 2023] dataset.

and strip statistics, while also allowing quad fine-tuning to modestly feed back and improve triangle generation quality (Sec. 5.4.3).

- 5 Experimental Results

SATO supports three tasks within a single framework: triangular mesh generation, UV segmentation generation, and quadrilateral mesh generation. Fig. 8 presents a gallery of our representative outputs (bottom to top): generated triangular meshes, generated UV segmentation (shown with color encoding), and generated quadrilateral meshes, highlighting the strong generative capability of our model.

Implementation Details. Our curated artist mesh dataset is aggregated from Objaverse [Deitke et al. 2023], ShapeNet [Chang et al. 2015], Thingi10K [Zhou and Jacobson 2016] and licensed datasets from Shutterstock [Shutterstock 2025]. The model in Fig. 1 using asset by SDragonXF on Sketchfab [SDragonXF 2020]. And the shapes in Fig. 2, 3, 5, 8, 12, 15 and 22 are from Shutterstock [Shutterstock 2025]. After the preprocessing in Sec. 4.4.2, we obtain about 1.47𝑀 triangle meshes, among which 1.11𝑀 include high-quality UV chart partitions. We additionally collect 120𝐾 UV-annotated quad meshes for fine-tuning. For each mesh, we randomly sample 81920 points as the point cloud condition. We train our model in three stages. First, we pre-train the triangle-mesh generator on 64 NVIDIA A800 GPUs for approximately 200K steps (∼7 days). Then post-train the model on UV-segmented data using 256 A800 GPUs for approximately 80K steps (∼3 days) to enable native UV-aware generation. Finally, we fine-tune the model on a high-quality quad dataset using 64 A800 GPUs for approximately 25K steps (∼1 day). For both pre-training and post-training, we use a cosine learning-rate schedule decaying from 10−4 to 10−5; for quad mesh fine-tuning, we fix the learning rate to 10−5. During training, we randomly sample a contiguous subsequence of length 9K tokens from each full token stream as the training input. At inference time, we enable KV-cache throughout autoregressive decoding and use temperature sampling with 𝑇 = 0.5.

- 5.1 Triangle Mesh Generation

Approaches. We include 4 state-of-the-art (SOTA) methods for comparison: MeshAnythingV2 [Chen et al. 2024c], BPT [Weng et al. 2025], TreeMeshGPT [Lionar et al. 2025], and DeepMesh [Zhao et al.

[Figure 13]

### GTBPTOursMeshAthV2TreeMeshGPTDeepMesh

Fig. 10. Qualitative comparison with baseline methods across different shapes. Our approach consistently produces high-quality artist meshes with stable structure and clean surface.

2025]. It is worth noting that several strong methods have appeared recently; however, most do not release inference code or pre-trained weights. Given the substantial cost of training mesh generation models, we restrict our comparisons to the four baselines with publicly available weights. We also exclude closed-source commercial systems (e.g., Tripo 3D [Tripo3D 2025]), which typically employ substantially larger models and do not provide reproducible code. For Hunyuan3D [Lei et al. 2025], which is a commercial closedsource system built upon scaled-up BPT [Weng et al. 2025], we only compare against its open-source 0.5B variant, whose backbone size matches ours.

Indicators. We evaluate triangle-mesh generation using four complementary metrics: Normal Consistency (NC), Chamfer Distance (CD), Hausdorff Distance (HD), and F-score (F1). Specifically, NC measures normal consistency and reflects surface orientation and local geometric fidelity; CD quantifies the average bidirectional point-to-point deviation between reconstructed and reference surfaces; HD captures the worst-case geometric error and is sensitive

###### Table 2. Quantitative comparison on ShapeNet [Chang et al. 2015], Thingi10K [Zhou and Jacobson 2016], and Objaverse [Deitke et al. 2023] datasets. The best scores are emphasized in bold with underlining, while the second best scores are highlighted only in bold.

ShapeNet Thingi10K Objaverse

Method NC ↑ CD ↓ HD ↓ F1 ↑ NC ↑ CD ↓ HD ↓ F1 ↑ NC ↑ CD ↓ HD ↓ F1 ↑ MeshAnythingV2 [Chen et al. 2024c] 0.911 0.009 0.078 0.361 0.841 0.022 0.168 0.162 0.858 0.016 0.117 0.208 TreeMeshGPT [Lionar et al. 2025] 0.840 0.034 0.161 0.439 0.791 0.058 0.228 0.236 0.783 0.057 0.238 0.188 BPT [Weng et al. 2025] 0.962 0.003 0.017 0.605 0.874 0.028 0.141 0.248 0.841 0.030 0.137 0.265 DeepMesh [Zhao et al. 2025] 0.967 0.004 0.037 0.532 0.853 0.026 0.167 0.157 0.859 0.020 0.120 0.240 SATO 0.975 0.002 0.032 0.807 0.916 0.009 0.154 0.460 0.909 0.009 0.117 0.503

###### Table 3. User study with SOTA methods on triangle mesh generation. Each score is the mean ranking-based score over all participants (range [0, 3]; 1st=3, 2nd=2, 3rd=1, others=0).

MeshAthV2 TreeMeshGPT BPT DeepMesh Ours Scores 0.18 0.57 1.4 1.17 2.61

Variance 0.27 0.67 0.95 0.93 0.49

to outliers and fine-scale artifacts; and F1 summarizes precision recall trade-offs under a distance threshold, indicating overall surface coverage and completeness. For all metrics, we uniformly sample 100K points from both the predicted mesh and the ground-truth mesh. CD and HD are computed from the bidirectional nearestneighbor distances between these two point sets, taking the mean and maximum respectively. F1 is computed as the harmonic mean of precision and recall at a distance threshold of 0.003. Together, these metrics jointly characterize both average and worst-case geometric accuracy, as well as perceptual surface quality.

We randomly selected 50 shapes from ShapeNet [Chang et al.

- 2015] and 100 shapes each from Thingi10K [Zhou and Jacobson
- 2016] and Objaverse [Deitke et al. 2023] to form our quantitative test sets, which are strictly excluded from our training data. This 250-shape test set is used consistently across all quantitative evaluations, ablation studies, and user studies throughout the paper. Fig. 9 shows an overview of these 250 models in our test dataset. Since autoregressive mesh generation can produce occasional nonmanifold elements, we apply PyMeshLab [Muntoni and Cignoni 2021] as a lightweight post-processing step to all methods for fair evaluation, following MeshMosaic [Xu et al. 2025]. We evaluated our method against four baselines, MeshAnythingV2 [Chen et al.

- 2024c], BPT [Weng et al. 2025], TreeMeshGPT [Lionar et al. 2025], and DeepMesh [Zhao et al. 2025], and report the NC, CD, HD, and F1 metrics in Table 2. Our method consistently outperforms the baselines across multiple metrics on all three datasets, highlighting its superior representational capacity and stronger alignment with the input shape. We further provide qualitative visualizations with baseline methods in Fig. 10. Overall, our method produces more complete shapes, higher mesh quality, and more artist-like topology.

User Study. However, it is often difficult to quantitatively assess whether a generated mesh truly matches the characteristics of an artist-created mesh, as opposed to one produced by generic geometric processing. We therefore conduct a user study to evaluate how artist-like our generated meshes appear. We recruited 25 professionals from the 3D industry as volunteers to conduct subjective

- Table 4. User study with PartUV [Wang et al. 2025a]. The scores are calculated based on the rankings and range from [0, 3].

PartUV w/ Our Mesh PartUV w/ GT Mesh Ours Scores 2.04 1.36 2.6

Variance 0.49 0.36 0.38

- Table 5. UV distortion comparison. Our segmentation produces consistently lower UV distortion than PartField [Liu et al. 2025] across all four standard metrics with Blender’s [Blender 2025] angle-based unwrapping.

Method L2 Stretch (↓ to 1) Area Dist. (↓) Angle Dist. (↓ to 1) Sym. Dirichlet (↓ to 4) PartField 0.921 0.849 1.256 8.283

SATO 0.979 0.562 1.128 5.156 Ground Truth 1.010 0.146 1.138 4.102

evaluations. Each participant evaluated 30 shape groups (10 for triangle mesh, 10 for quad mesh, and 10 for UV segmentation). For each group, participants were presented with rendered images from four viewpoints together with the ground-truth shape and the input point cloud. The four criteria (regularity, artist-likeness, geometric fidelity, and shape consistency) served as holistic guidelines; participants gave a single overall top-3 ranking rather than separate per-criterion scores. Rankings were converted to scores as 1st=3, 2nd=2, 3rd=1, and others=0. Table 3 summarizes the comparative ratings from the user study. Overall, our method receives higher rankings from participants, indicating improved mesh quality and closer stylistic alignment with artist-created meshes.

5.2 UV Segmentation

Simultaneously generating UV segmentation during autoregressive mesh synthesis remains largely unexplored. To the best of our knowledge, SATO is the first method to explicitly support this task, which makes direct comparisons challenging. The closest recent open-source baseline is PartUV [Wang et al. 2025a], which performs UV segmentation on an input mesh with PartField [Liu et al. 2025] segmentation but does not generate meshes and instead operates on the provided geometry.

Fig. 11 reports a qualitative comparison with PartUV [Wang et al. 2025a], where (a, b) show the triangular mesh and UV segmentation result produced by our method, and (c) visualizes the UV layout obtained by unwrapping our predicted segmentation in Blender [Blender 2025]. In contrast, Fig. 11 (d,e) present PartUV’s Blender unwrappings when applied to (d) our generated triangular mesh and (e) a high-quality ground-truth triangular mesh, respectively. Our method yields consistently cleaner and higherquality UV layouts, whereas PartUV [Wang et al. 2025a] produces less regular unwrappings regardless of whether it is applied to our

[Figure 14]

### (a) Generated Mesh (b) Generated UV Seg (c) Ours UV (d) PartUV from Our Mesh (e) PartUV from GT Mesh

- Fig. 11. Qualitative comparison with PartUV [Wang et al. 2025a]. Our method generates an artist mesh (a) together with explicit UV segmentation (b). By applying angle-based UV unwrapping from Blender [Blender 2025], we further obtain a high-quality 2D UV layout (c). In contrast, PartUV relies on a PartField [Liu et al. 2025] pre-segmentation pipeline; regardless of whether it is applied to our generated mesh (d) or the ground-truth (GT) mesh (e), its resulting UV charts are consistently less clean and less well-structured than ours.

[Figure 15]

- Fig. 12. Gallery of UV unwrapping results using our generated UV segmentation. The shapes are taken from Fig. 8, and both UV unwrapping and visualization are obtained through the unwrapping algorithm in Blender [Blender 2025].

generated mesh or the ground-truth mesh. Furthermore, when PartUV [Wang et al. 2025a] is applied to our generated high-quality triangular meshes, it produces better UV unwrapping results than when applied to the original GT meshes. This also highlights the practicality and downstream usability of our generated triangular meshes. We further present a gallery of our UV unwrapping results in Fig. 12, where the model is derived from Fig. 8 and unwrapped using Blender [Blender 2025] following PartUV [Wang et al. 2025a].

We also conduct a user study for the UV unwrapping evaluation. Participants were asked to rate the UV layouts produced by our method and by PartUV [Wang et al. 2025a]. Table 4 reports the quantitative results, showing that artists consistently prefer the cleaner, more organized UV segmentation achieved by our approach.

UV Distortion. To quantitatively evaluate the UV quality resulting from our segmentation, we compute standard parameterization

[Figure 16]

### Ours MeshMosaic

- Fig. 13. Comparison with MeshMosaic [Xu et al. 2025]. Our method yields cleaner, more regular segmentation and mitigates the issue of overly long seams.

distortion metrics on the 10 generated meshes from Fig. 12. For each mesh, we apply Blender’s angle-based unwrapping [Blender 2025], then compute the Jacobian of the 3D-to-UV mapping per triangle, obtain its singular values 𝜎1 ≥ 𝜎2 > 0, and report four metrics: L2 Stretch √︃(𝜎12 + 𝜎22)/2, Area Distortion | log(𝜎1 · 𝜎2)|, Angle Distortion 𝜎1/𝜎2, and Symmetric Dirichlet energy 𝜎12 + 𝜎22 + 1/𝜎12 +1/𝜎22 [Smith and Schaefer 2015]. All values are area-weighted medians after per-island normalization. As the baseline, we compare against PartField [Liu et al. 2025] segmentation unwrapped with the same algorithm. Table 5 shows that our segmentation consistently yields lower distortion across all four metrics, indicating that our predicted chart boundaries better align with geometric features, producing more regular islands that are easier to unwrap with low distortion. Even compared with ground truth UVs created by artists, our method still shows advantages in certain aspects, such as angle distortion.

Another autoregressive mesh generation method related to segmentation is a very recent work, MeshMosaic [Xu et al. 2025]. It leverages PartField [Liu et al. 2025] predicted segmentations to decompose a shape into multiple parts and then autoregressively generates the mesh part by part in a fixed sequence. Fig. 13 compares MeshMosaic [Xu et al. 2025] with our method. While both approaches can follow the input shape, MeshMosaic’s reliance on precomputed part boundaries often yields unnatural transitions across parts, including visible seams, and can introduce asymmetry artifacts. In contrast, our method jointly generates the full mesh and its segmentation in a unified pass, which naturally enforces global consistency and avoids inter-part seams and symmetry issues.

More recently, MeshSilksong [Song et al. 2025] can predict connected components during autoregressive mesh generation. Although these labels are not UV segmentations, we include MeshSilksong [Song et al. 2025] as an additional point of comparison.

- Fig. 14 visualizes results from our method and MeshSilksong, where MeshSilksong uses different colors to denote different connected

[Figure 17]

### Ours Seg Ours Mesh MeshSilksong

- Fig. 14. Comparison with MeshSilksong [Song et al. 2025]. Our method produces higher-quality meshes and cleaner, more coherent segmentation.

[Figure 18]

- Fig. 15. Texture painting with our UV unwrapping. The high-quality UV unwrapping produced by our method makes it easy for artists to paint texture maps.

components, whereas our colors indicate UV charts. MeshSilksong almost failed at the segmentation task, only separating the rabbit’s eyes, while the rest of the whole was treated as a single connected component as output. Overall, our method produces meshes that are more complete and higher quality and yields more coherent, meaningful segmentations, highlighting the advantage of our joint generation approach.

Finally, we demonstrate the practical utility of the UV unwrapping produced by our method in Fig. 15. Artists can readily apply textures to the resulting UV layout: each component of the input shape is cleanly and consistently separated into well-defined islands, enabling targeted texture painting without inadvertently affecting other parts.

5.3 Quad Mesh Generation

Finally, we evaluate the quadrilateral meshes generated by our method. As described in Sec. 4.3, SATO supports quad-mesh generation with a simple switch of the detokenizer, without altering the model architecture. Since our quad detokenizer simply merges adjacent triangle pairs from the same token sequence, the geometric

[Figure 19]

### BPT DeepMesh Ours Ours UV

- Fig. 16. Qualitative comparison with BPT [Weng et al. 2025] and DeepMesh [Zhao et al. 2025] on diverse shapes. Compared with prior trianglemesh generation models, our method more consistently generates high-quality quadrilateral meshes, is more stable, and additionally predicts native UV segmentation.

[Figure 20]

Input Quadriflow Quadwild Ours

IM NeurCross CrossGen Ours

- Fig. 17. Qualitative comparison with quad remeshing and reconstruction methods. Due to reliance on quadrilateral parameterization, these methods typically struggle to produce highly simplified quad meshes. In contrast, our method can generate meshes with diverse densities and additionally supports native UV segmentation.

publicly provide code or pretrained weights, and is trained on an undisclosed proprietary dataset of 1.3M quad meshes, making direct comparison infeasible. Following QuadGPT, we do not enforce strict quad coplanarity, as artist-created quad meshes in practice also exhibit slight non-planarity. We therefore follow QuadGPT’s evaluation protocol and compare against representative trianglemesh autoregressive methods. Fig. 16 presents a visual comparison between the quadrilateral meshes produced by our method and those obtained by BPT [Weng et al. 2025] and DeepMesh [Zhao et al. 2025]. Our approach not only yields high-quality, high-fidelity quad meshes, but also simultaneously produces clean, artist-aligned UV segmentations.

To further demonstrate both the capability and practical value of our quad-mesh generator, we additionally compare against several established quadrilateral reconstruction and remeshing methods. Fig. 17 contrasts our results with five alternatives: IM [Jakob et al. 2015], QuadriFlow [Huang et al. 2018], QuadWild [Pietroni et al. 2021], NeurCross [Dong et al. 2025b], and CrossGen [Dong et al. 2025a]. IM,QuadriFlow,andQuadWildare classical parameterization-

based quad remeshing approaches, whereas NeurCross and CrossGen represent more recent cross-field generation methods. All baselines were run using their default parameter settings. To produce outputs of comparable resolution across methods while preserving sufficient geometric detail, we set the corresponding resolutioncontrol parameters for each baseline, namely, 3000 points for IM, 6000 faces for QuadriFlow, and a scaleFact parameter of 1.0 for

fidelity of quad outputs is nearly identical to that of the corresponding triangle outputs.

For assessing quad quality, the recently released QuadGPT [Liu et al. 2025a] is a strong reference baseline; however, it does not

###### Table 6. User study with remeshing and reconstruction methods on quad mesh generation. The scores are calculated based on the rankings and range from [0, 3].

IM QuadriFlow Quadwild NeurCross CrossGen Ours Scores 0.12 0.28 1.08 1.48 1.24 1.8

Variance 0.20 0.44 1.23 1.32 1.28 1.28

###### Table 7. Geometric metrics on the quad mesh from Fig. 17. Remeshing methods achieve near-identical fidelity to the input since they operate directly on the ground-truth geometry, whereas our method generates the mesh from a point cloud.

Method NC ↑ CD ↓ HD ↓ F1 ↑ IM 0.917 0.007 0.052 0.304

QuadriFlow 0.924 0.005 0.080 0.451 QuadWild 0.970 0.001 0.020 0.848 NeurCross 0.968 0.002 0.020 0.846 CrossGen 0.969 0.001 0.020 0.849

Ours 0.971 0.001 0.020 0.857

###### Table 8. Quantitative comparison with the DeepMesh [Zhao et al. 2025] tokenizer. Under the overfitting setup in Fig. 18, we report compression and training metrics for both tokenizers. Our method achieves better training speed and a higher compression ratio.

Methods Token Length ↓ Transitions ↓ Time (s) ↓ Training Speed (Steps/s) ↑ Infer (Tokens/s) ↑ DeepMesh 24674 1654 2.061 0.442 ∼55

Ours 20830 981 0.319 0.488 ∼58

QuadWild, NeurCross, and CrossGen. Because these methods generate isotropic quadrilateral meshes, reducing the resolution leads to a simpler output but inevitably sacrifices geometric detail. As shown in Fig. 17, these remeshing baselines often struggle to simultaneously achieve high quad utilization, low face count, and consistent alignment with salient feature lines. In contrast, our method produces compact, well-structured quad layouts that better match artist expectations. This comparison not only highlights the quality of our outputs, but also underscores the importance of autoregressive artist mesh generation as a complementary direction to conventional remeshing pipelines. We additionally report geometric metrics for the shape in Fig. 17 in Table 7. Remeshing methods achieve near-identical fidelity since they operate directly on the ground-truth geometry, whereas our method generates the mesh from a point cloud; despite this, our output achieves competitive or superior scores.

Furthermore, we also conduct a user study for this task, asking participants to evaluate the quad meshes produced by our method and by the five quadrilateral reconstruction/remeshing baselines described above. Table 6 summarizes the results, showing a clear preference for our outputs. This further validates the practicality of our artist quadrilateral mesh generation and highlights its value for real-world content creation workflows.

- 5.4 Ablation Studies

We conduct a series of ablation studies to verify our proposed ideas and to compare them in detail with other methods.

- 5.4.1 Tokenizer. First, we validate the advantage of SATO via a more fine-grained comparison with DeepMesh’s tokenizer [Zhao et al. 2025]. Specifically, we construct an overfitting experiment on

[Figure 21]

OursDeepMesh

10K Step 20K Step

Fig. 18. Ablation with the DeepMesh [Zhao et al. 2025] tokenizer. We constructed an overfitting ablation to compare our tokenizer with DeepMesh. Our tokenizer converges faster and is easier for the network to learn, even when augmented with UV segmentation.

the teapot model in Fig. 18, training with either SATO or DeepMesh tokenization. To prevent the network from trivially memorizing a fixed token sequence, we apply random rotations to the input shape during training.

We use the same 0.5B Hourglass Transformer architecture and identical training hyperparameters for both settings: training on 8×A800 GPUs for 20,000 steps, with the tokenizer as the only difference. Fig. 18 visualizes the generation results at 10,000 and 20,000 steps. Our method learns UV segmentation cues and reaches a nearperfect reconstruction noticeably faster. Even at early stages, the predicted segmentation is already clean and well-structured. Moreover, thanks to our strip-based serialization, the intermediate geometry appears cleaner and more refined, indicating easier optimization and faster convergence.

We also record a set of training statistics for this overfitting experiment, summarized in Table 8. When encoding the teapot model, the two tokenizers produce sequences of 24K and 20K tokens, respectively, meaning SATO requires only about 85% of the token length of DeepMesh. This reduction is largely attributable to fewer patch/strip transitions (0.9K vs. 1.6K). Moreover, thanks to our efficient next-face query structure in the tokenization code, SATO substantially reduces encoding time relative to the baseline, which in turn translates into a clear advantage in overall training throughput.

To further isolate the benefit of our tokenizer at scale, we conduct a controlled large-scale comparison. Since DeepMesh [Zhao et al. 2025] only releases inference code and pretrained weights but not its training pipeline, we re-implement its tokenizer within our training framework so that both settings share the identical model architecture, training data, optimizer, and hyperparameters. Both models are trained on 64×A800 GPUs for 200K steps, with the tokenizer as the sole variable. Table 9 reports the results on our 250-shape test set, confirming that the strip-based tokenizer yields consistent improvements across all metrics under strictly matched training conditions.

- Table 9. Large-scale tokenizer ablation. Both models use the same architecture, data, and training budget (64×A800, 200K steps); only the tokenizer differs.

Tokenizer NC ↑ CD ↓ HD ↓ F1 ↑ DeepMesh 0.908 0.022 0.108 0.455

##### Ours 0.925 0.014 0.103 0.560

[Figure 22]

MeshAthV2 TreeMeshGPT

Ours (a) Ours (b) Ours (c)

BPT DeepMesh

Ours (a) Ours (b) Ours (c)

Fig. 19. Ablation on UV training strategies with a manually generated unseen test shape. All methods use the same test shape, which is manually created with Rodin [Zhang et al. 2024b] from an input text prompt cute bunny astronaut toy, converted into an SDF, and then meshed with marching cubes. This construction guarantees that the shape lies strictly outside the training set. We compare SOTA methods and three training strategies: (a) using a pretrained point cloud VAE with UV segmentation data, (b) training from scratch on UV data, and (c) first training from scratch without UV data followed by post training on UV data. The first two strategies that directly use UV data cannot reliably align with the input shape and usually collapse to an almost random shape.

- 5.4.2 Pretraining for UV. As discussed in Sec. 4.4, we first pre-train our network on triangle mesh data without UV segmentation, and then post-train it on data with UV segmentation. We find that imposing UV supervision from the beginning often prevents the model from learning fine grained geometric details and makes it more difficult to align the generated mesh with the input.

Fig. 19 compares different training strategies on a manually generated test shape that is guaranteed to be unseen during training. Specifically, the input shape is manually created with Rodin [Zhang et al. 2024b] from an input prompt cute bunny astronaut toy, converted into an SDF, and then meshed with marching cubes, which ensures that it lies strictly outside the training set. All three settings use the same conditioning point-cloud input and are trained for three days on 256×A800 GPUs. When trained entirely from scratch (From scratch in Fig. 19), the model collapses to producing essentially random shapes, with only a coarse alignment in orientation. Even replacing our point-cloud encoder with a pretrained, frozen Hunyuan 3D [Lei et al. 2025] VAE encoder does not substantially alleviate this issue (Pretrained in Fig. 19). In contrast, our two-stage strategy that involves pre-training without UV followed by posttraining with UV achieves accurate alignment with the input while producing clean, well-structured UV segmentations.

[Figure 23]

### w/o fine-tune w/ fine-tune

- Fig. 20. Ablation on quad-mesh fine-tuning. After quad-mesh finetuning, the meshes in the black boxed region become markedly higher quality and more artist-aligned, with cleaner structure and easier downstream editing.

[Figure 24]

- Fig. 21. Generation from image and text prompts. By leveraging CLAY [Zhang et al. 2024b] for 3D generation, SATO can produce highquality artist meshes with native UV segmentation from either an input image or a text prompt.

- 5.4.3 Quad Mesh Fine-tuning. Also discussed in Sec. 4.4, incorporating high-quality quadrilateral mesh data can further improve our triangle-mesh generator. In practice, fine-tuning on quad meshes encourages neater mesh routing and increases the prevalence of well-shaped triangles (often closer to right-angled triangles), bringing the output closer to artist-created meshes. Fig. 20 compares results before and after fine-tuning with quadrilateral data. After fine-tuning, the generated meshes exhibit cleaner, more quad-like routing and a reduced tendency to produce dense regions of elongated, skinny triangles.
- 5.5 More Discussions

Generation with Image and Text Prompt. Techniques for generating 3D assets from diverse inputs have advanced rapidly in recent

[Figure 25]

- Fig. 22. Diversity results. Conditioned on the same input, our model generates diverse meshes and segmentation outcomes, demonstrating strong generative diversity.

[Figure 26]

- Fig. 23. Failure cases. Left: Degenerate triangle faces arising from quadrilateral strip decoding under irregular strip configurations. Right: Suboptimal quad layouts limited by the scale and consistency of available high-quality quad-mesh training data.

years [Lai et al. 2025; Zhang et al. 2024b]. Many SDF-based pipelines can produce highly detailed geometry, but they typically yield ultradense triangle meshes via Marching Cubes [Lorensen and Cline 1998], and still require substantial post-processing or downstream meshing to obtain lightweight, production-ready, artist-style meshes. Fig. 21 demonstrates how our method can be used as a remeshing stage for such generated shapes. Specifically, we use CLAY [Zhang et al. 2024b] as an upstream generator; given an image or a text prompt, it predicts a 3D SDF and extracts a high-face-count mesh using Marching Cubes. Starting from this mesh, our method further performs UV segmentation and produces high-quality, lightweight triangular or quadrilateral meshes that are directly usable in practice.

Diversity. As with other generative models, we showcase the diversity of our outputs in Fig. 22, which is an important property for all generative systems. Our model produces not only diverse mesh geometries, but also diverse UV segmentation. Despite this diversity, the generated UV charts remain clean, well-structured, and often symmetric, closely reflecting common artist modeling and layout conventions.

- 6 Limitations and Future Work

As the first framework to jointly model mesh generation and UV segmentation while supporting both triangle and quadrilateral outputs, our approach introduces several natural trade-offs.

First, our quad output is decoded from quadrilateral strips, which yields predominantly quad-dominant meshes in practice. However,

in a small number of cases, e.g., when a strip has an odd length or contains repeated vertices, local faces may degenerate into triangles (see Fig. 23, left). Importantly, these cases are structurally welldefined and can be further reduced with improved dataset quality or lightweight post-processing, which we leave to one of our future works.

Second, the attainable quad quality is currently bounded by the scale and consistency of available high-quality quad-mesh datasets. While our method already produces visually compelling quad layouts, dedicated quad-only approaches such as QuadGPT [Liu et al. 2025a] benefit from being optimized specifically for this setting. We view this as a complementary strength: our goal is a unified model that transfers strong priors from large-scale triangle data and extends them to quad meshes with minimal specialization (see Fig. 23, right).

Finally, we occasionally observe less regular edge routing on near-spherical shapes (like the bottom left shape in Fig. 13). This appears tied to data bias: many triangle datasets represent spheres using near-equilateral tessellations, whereas high-quality spherical exemplars are comparatively scarce in existing quad corpora. Consequently, quad fine-tuning provides a consistent but incremental improvement rather than fully resolving this corner case. We expect this gap to narrow as richer quad datasets become available and as we incorporate stronger shape-adaptive routing priors.

7 Conclusion

We present Strips as Tokens (SATO), an autoregressive framework for generating high-quality artist meshes with native UV segmentation. Our strip-based tokenization follows the edge flow of artist meshes and encodes UV island boundaries directly in the sequence, encouraging clean topology and well-structured UV chart partitions. Building on the same sequence format, SATO admits a unified triangle/quad interpretation, enabling mixed-data training that transfers and strengthens priors across formats. Extensive experiments show that SATO produces diverse, high-fidelity meshes with stronger topological quality than competitive baselines, highlighting its practical potential for downstream content creation pipelines.

References

Blender. 2025. Blender. https://www.blender.org/. Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang,

Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. 2015. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012 (2015).

Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Billzb Wang, Jingyi Yu, Gang Yu, et al. 2024a. Meshxl: Neural coordinate field for generative 3d foundation models. Advances in Neural Information Processing Systems 37 (2024), 97141–97166.

Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al. 2024b. Meshanything: Artist-created mesh generation with autoregressive transformers. arXiv preprint arXiv:2406.10163 (2024).

Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. 2025a. Ultra3D: Efficient and High-Fidelity 3D Generation with Part Attention. arXiv:2507.17745 [cs.CV]

Yuguang Chen, Xinhai Liu, Yang Li, Victor Cheung, Zhuo Chen, Dongyu Zhang, and Chunchao Guo. 2025b. ArtUV: Artist-style UV Unwrapping. arXiv preprint arXiv:2509.20710 (2025).

Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. 2024c. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. arXiv preprint arXiv:2408.02555 (2024).

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani,

Ludwig Schmidt, and Ali Farhadi. 2023. Objaverse-XL: A Universe of 10M+ 3D Objects. arXiv preprint arXiv:2307.05663 (2023).

Qiujie Dong, Jiepeng Wang, Rui Xu, Cheng Lin, Yuan Liu, Shiqing Xin, Zichun Zhong, Xin Li, Changhe Tu, Taku Komura, Leif Kobbelt, Scott Schaefer, and Wenping Wang. 2025a. CrossGen: Learning and Generating Cross Fields for Quad Meshing. ACM Trans. Graph. 44, 6 (2025). doi:10.1145/3763299

Qiujie Dong, Huibiao Wen, Rui Xu, Shuangmin Chen, Jiaran Zhou, Shiqing Xin, Changhe Tu, Taku Komura, and Wenping Wang. 2025b. NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation. ACM Trans. Graph. 44, 4 (2025). doi:10.1145/3731159

Zekun Hao, David W Romero, Tsung-Yi Lin, and Ming-Yu Liu. 2024. Meshtron: Highfidelity, artist-like 3d mesh generation at scale. arXiv preprint arXiv:2412.09548

(2024). Jingwei Huang, Yichao Zhou, Matthias Niessner, et al. 2018. QuadriFlow: A Scalable and Robust Method for Quadrangulation. Computer Graphics Forum (2018). Martin Isenburg. 2001. Triangle strip compression. In Computer Graphics Forum, Vol. 20. Wiley Online Library, 91–101. Wenzel Jakob, Marco Tarini, Daniele Panozzo, and Olga Sorkine-Hornung. 2015. Instant field-aligned meshes. ACM Transactions on Graphics 34, 6 (2015), 189. Jeonghwan Kim, Yushi Lan, Armando Fortes, Yongwei Chen, and Xingang Pan.

2025. FastMesh: Efficient Artistic Mesh Generation via Component Decoupling. arXiv:2508.19188

Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, et al. 2025. Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details. arXiv preprint arXiv:2506.16504 (2025).

Biwen Lei, Yang Li, Xinhai Liu, Shuhui Yang, Lixin Xu, Jingwei Huang, Ruining Tang, Haohan Weng, Jian Liu, Jing Xu, et al. 2025. Hunyuan3d studio: End-to-end ai pipeline for game-ready 3d asset generation. arXiv preprint arXiv:2509.12815 (2025).

Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. 2024. Craftsman3d: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979 (2024).

Yang Li, Victor Cheung, Xinhai Liu, Yuguang Chen, Zhongjin Luo, Biwen Lei, Haohan Weng, Zibo Zhao, Jingwei Huang, Zhuo Chen, et al. 2025. Auto-Regressive Surface Cutting. arXiv preprint arXiv:2506.18017 (2025).

Junkai Lin, Hang Long, Huipeng Guo, Jielei Zhang, JiaYi Yang, Tianle Guo, Yang Yang, Jianwen Li, Wenxiao Zhang, Matthias Nießner, et al. 2025. MeshRipple: Structured Autoregressive Generation of Artist-Meshes. arXiv preprint arXiv:2512.07514 (2025).

Stefan Lionar, Jiabin Liang, and Gim Hee Lee. 2025. Treemeshgpt: Artistic mesh generation with autoregressive tree sequencing. In Proceedings of the Computer Vision and Pattern Recognition Conference. 26608–26617.

Jian Liu, Chunshi Wang, Song Guo, Haohan Weng, Zhen Zhou, Zhiqi Li, Jiaao Yu, Yiling Zhu, Jing Xu, Biwen Lei, Zhuo Chen, and Chunchao Guo. 2025a. QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models. arXiv e-prints, Article arXiv:2509.21420 (Sept. 2025), arXiv:2509.21420 pages. arXiv:2509.21420 [cs.CV]

Jian Liu, Jing Xu, Song Guo, Jing Li, Jingfeng Guo, Jiaao Yu, Haohan Weng, Biwen Lei, Xianghui Yang, Zhuo Chen, Fangqi Zhu, Tao Han, and Chunchao Guo. 2025b. Mesh-RFT: Enhancing Mesh Generation via Fine-grained Reinforcement FineTuning. arXiv e-prints, Article arXiv:2505.16761 (May 2025), arXiv:2505.16761 pages. arXiv:2505.16761 [cs.CV]

Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2025. Partfield: Learning 3d feature fields for part segmentation and beyond. arXiv preprint arXiv:2504.11451 (2025).

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2024. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9970–9980.

William E Lorensen and Harvey E Cline. 1998. Marching cubes: A high resolution 3D surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field. 347–353.

Alessandro Muntoni and Paolo Cignoni. 2021. PyMeshLab. doi:10.5281/zenodo.4438750 Nico Pietroni, Stefano Nuvoli, Thomas Alderighi, Paolo Cignoni, and Marco Tarini.

2021. Reliable feature-line driven quad-remeshing. ACM Trans. Graph. 40, 4 (2021), Article 155. doi:10.1145/3450626.3459941

Massimiliano B Porcu and Riccardo Scateni. 2003. An Iterative Stripification Algorithm

Based on Dual Graph Operations. In Eurographics (Short Presentations). SDragonXF. 2020. dragon head3. Sketchfab. Licensed under CC BY NC ND 4.0. Shutterstock. 2025. Shutterstock. https://www.shutterstock.com/. Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. 2024. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 19615–19625.

Jason Smith and Scott Schaefer. 2015. Bijective parameterization with free boundaries. ACM Trans. Graph. 34, 4 (2015), 70–1.

Gaochao Song, Zibo Zhao, Haohan Weng, Jingbo Zeng, Rongfei Jia, and Shenghua Gao. 2025. Mesh Silksong: Auto-Regressive Mesh Generation as Weaving Silk. arXiv

preprint arXiv:2507.02477 (2025).

Pratul P. Srinivasan, Stephan J. Garbin, Dor Verbin, Jonathan T. Barron, and Ben Mildenhall. 2025. Nuvo: Neural UV Mapping for Unruly 3D Representations. In Computer Vision – ECCV 2024, Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol (Eds.). Springer Nature Switzerland, Cham, 18–34.

Jiaxiang Tang, Zhaoshuo Li, Zekun Hao, Xian Liu, Gang Zeng, Ming-Yu Liu, and Qinsheng Zhang. 2024. Edgerunner: Auto-regressive auto-encoder for artistic mesh generation. arXiv preprint arXiv:2409.18114 (2024).

Tripo3D. 2025. Tripo 3D. https://studio.tripo3d.ai/. Petr Vaněček and Ivana Kolingerová. 2007. Comparison of triangle strips algorithms.

Computers & Graphics 31, 1 (2007), 100–118. Petr Vanecek, Radek Svitak, Ivana Kolingerova, and Vaclav Skala. 2005. Quadrilateral meshes stripification. In Proceedings of ALGORITMY. 300–308.

Hanxiao Wang, Biao Zhang, Weize Quan, Dong-Ming Yan, and Peter Wonka. 2025c. iflame: Interleaving full and linear attention for efficient mesh generation. arXiv preprint arXiv:2503.16653 (2025).

Yuxuan Wang, Xuanyu Yi, Haohan Weng, Qingshan Xu, Xiaokang Wei, Xianghui Yang, Chunchao Guo, Long Chen, and Hanwang Zhang. 2025b. Nautilus: Locality-aware autoencoder for scalable mesh generation. arXiv preprint arXiv:2501.14317 (2025).

Zhaoning Wang, Xinyue Wei, Ruoxi Shi, Xiaoshuai Zhang, Hao Su, and Minghua Liu. 2025a. PartUV: Part-Based UV Unwrapping of 3D Meshes. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25). Association for Computing Machinery, New York, NY, USA, Article 15, 12 pages.

Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, et al. 2025. Scaling mesh generation via compressive tokenization. In Proceedings of the Computer Vision and Pattern Recognition Conference. 11093–11103.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. 2025a. Native and Compact Structured Latents for 3D Generation. arXiv preprint arXiv:2512.14692 (2025).

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025b. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference. 21469–21480.

Xinyu Xiang, Martin Held, and Joseph SB Mitchell. 1999. Fast and effective stripification of polygonal surface models. In Proceedings of the 1999 symposium on Interactive 3D graphics. 71–78.

Rui Xu, Longdu Liu, Ningna Wang, Shuangmin Chen, Shiqing Xin, Xiaohu Guo, Zichun Zhong, Taku Komura, Wenping Wang, and Changhe Tu. 2024. CWF: consolidating weak features in high-quality mesh simplification. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–14.

Rui Xu, Tianyang Xue, Qiujie Dong, Le Wan, Zhe Zhu, Peng Li, Zhiyang Dou, Cheng Lin, Shiqing Xin, Yuan Liu, et al. 2025. MeshMosaic: Scaling Artist Mesh Generation via Local-to-Global Assembly. arXiv preprint arXiv:2509.19995 (2025).

Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, and Xihui Liu. 2025. OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25). Association for Computing Machinery, New York, NY, USA, Article 59, 12 pages.

Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Lan Xu, Wei Yang, Jiayuan Gu, and Jingyi Yu. 2025. CAST: Component-Aligned 3D Scene Reconstruction from an RGB Image. ACM Trans. Graph. 44, 4, Article 86 (July 2025), 19 pages.

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024b. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG) 43, 4 (2024), 1–20.

Longwen Zhang, Qixuan Zhang, Haoran Jiang, Yinuo Bai, Wei Yang, Lan Xu, and Jingyi Yu. 2025. BANG: Dividing 3D Assets via Generative Exploded Dynamics. ACM Trans. Graph. 44, 4, Article 62 (July 2025), 21 pages.

Qijian Zhang, Junhui Hou, Wenping Wang, and Ying He. 2024a. Flatten Anything: Unsupervised Neural Surface Parameterization. In Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates, Inc., 2830–2850.

Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. 2025. Deepmesh: Auto-regressive artist-mesh creation with reinforcement learning. arXiv preprint arXiv:2503.15265 (2025).

Qingnan Zhou and Alec Jacobson. 2016. Thingi10k: A dataset of 10,000 3d-printing models. arXiv preprint arXiv:1605.04797 (2016).

