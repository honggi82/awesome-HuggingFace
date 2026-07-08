# arXiv:2511.16659v2[cs.CV]17Feb2026

### PartUV: Part-Based UV Unwrapping of 3D Meshes

ZHAONING WANG, Hillbot Inc., USA XINYUE WEI, University of California San Diego, USA and Hillbot Inc., USA RUOXI SHI, University of California San Diego, USA and Hillbot Inc., USA XIAOSHUAI ZHANG, Hillbot Inc., USA HAO SU, University of California San Diego, USA and Hillbot Inc., USA MINGHUA LIU, Hillbot Inc., USA

[Figure 1]

Fig. 1. We propose PartUV, a novel part-based UV unwrapping method for 3D meshes. Unlike traditional approaches that rely solely on local geometric priors and often produce over-fragmented charts, PartUV combines learned part priors with geometric cues to generate a small number of part-aligned charts. We evaluate our method on four diverse datasets—PartObjaverseTiny (man-made) [Yang et al. 2024], Trellis (AI-generated) [Xiang et al. 2024], ABC (CAD) [Koch et al. 2019], and Common Shapes [Jacobson and contributors 2023]—and show that it produces significantly less fragmented UV mappings while maintaining low distortion on par with baseline methods. Leveraging part-aware charts also enables applications such as generating one atlas per part.

Authors’ Contact Information: Zhaoning Wang, Hillbot Inc., USA, zhaoning.eric.wang@ gmail.com; Xinyue Wei, University of California San Diego, USA and Hillbot Inc., USA, xiwei@ucsd.edu; Ruoxi Shi, University of California San Diego, USA and Hillbot Inc., USA, r8shi@ucsd.edu; Xiaoshuai Zhang, Hillbot Inc., USA, x@hillbot.ai; Hao Su, University of California San Diego, USA and Hillbot Inc., USA, haosu@ucsd.edu; Minghua Liu, Hillbot Inc., USA, m@hillbot.ai.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SA Conference Papers ’25, Hong Kong, Hong Kong

UV unwrapping flattens 3D surfaces to 2D with minimal distortion, often requiring the complex surface to be decomposed into multiple charts. Although extensively studied, existing UV unwrapping methods frequently struggle with AI-generated meshes, which are typically noisy, bumpy, and poorly conditioned. These methods often produce highly fragmented charts and suboptimal boundaries, introducing artifacts and hindering downstream tasks. We introduce PartUV, a part-based UV unwrapping pipeline that generates significantly fewer, part-aligned charts while maintaining low distortion. Built on top of a recent learning-based part decomposition method PartField, PartUV combines high-level semantic part decomposition with novel geometric heuristics in a top-down recursive framework. It ensures

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2137-3/2025/12 https://doi.org/10.1145/3757377.3763843

each chart’s distortion remains below a user-specified threshold while minimizing the total number of charts. The pipeline integrates and extends parameterization and packing algorithms, incorporates dedicated handling of non-manifold and degenerate meshes, and is extensively parallelized for efficiency. Evaluated across four diverse datasets—including man-made, CAD, AI-generated, and Common Shapes—PartUV outperforms existing tools and recent neural methods in chart count and seam length, achieves comparable distortion, exhibits high success rates on challenging meshes, and enables new applications like part-specific multi-tiles packing. Our project page is at https://www.zhaoningwang.com/PartUV.

CCS Concepts: • Shape modeling → Shape analysis; • Mesh models → Mesh geometry processing; Parametrization; Geometric algorithms. ACM Reference Format: ZhaoningWang, Xinyue Wei, Ruoxi Shi, Xiaoshuai Zhang, Hao Su, and Minghua Liu. 2025. PartUV: Part-Based UV Unwrapping of 3D Meshes. In SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25), December 15–18, 2025, Hong Kong, Hong Kong. ACM, New York, NY, USA, 14 pages. https: //doi.org/10.1145/3757377.3763843

- 1 Introduction

UV unwrapping projects the 3D surface of a mesh onto a 2D plane, assigning every 3D surface point a corresponding 2D UV coordinate. This step is fundamental in 3D-content-creation pipelines because it enables detailed surface information-such as material properties (e.g., base-color, roughness, normal maps), along with auxiliary maps like ambient occlusion and displacement—to be efficiently stored and manipulated in 2D space.

A principal component of UV unwrapping is surface parameterization, which flattens the 3D surface while trying to preserve geometric properties such as angles and areas. For meshes with complex geometry, flattening the entire surface into a single 2D chart introduces large distortion. Consequently, chart segmentation (or seam cutting) is typically used to divide the mesh into multiple charts along strategically placed seams, allowing each chart to be flattened with reduced distortion. Finally, UV packing arranges the resulting charts within the unit square (the UV atlas) to maximize texture-space use.

Although UV unwrapping has been studied extensively, existing methods are typically tuned for well-behaved meshes, such as those created by professional 3D artists. They often fail on more complex, AI-generated meshes. Such meshes, typically extracted from neural-field isosurfaces (e.g., via marching cubes [Lorensen and Cline 1998]), tend to have bumpy surfaces, many small triangles, and poor geometric quality (e.g., disconnected components or holes). On such data, existing methods may time-out or return extremely fragmented atlases in which a single chart holds only one or a handful of triangles. This extreme fragmentation hampers texture painting and editing, introduces texture-bleed and baking or rendering artifacts at chart boundaries, and burdens downstream applications with an unwieldy number of charts.

Some recent approaches mitigate fragmentation [Srinivasan et al. 2024; Zhang et al. 2024; Zhao et al. 2025] by representing the UV mapping using a neural field and optimizing such a field for each 3D shape from scratch. While these methods can effectively control the number of charts generated, they typically run for more than thirty minutes and still exhibit noticeable distortion. Other methods [Li

et al. 2018; Poranne et al. 2017] jointly optimize seam length and distortion but are likewise computationally expensive. Moreover, some existing approaches like [Lévy et al. 2002; Sorkine et al. 2002; Zhou et al. 2004] segment charts or identify seams using heuristics based on local geometric properties, rather than leveraging the concept of geometric or semantic parts. This can lead to unintuitive or suboptimal chart boundaries that split semantically coherent regions across multiple charts, further complicating downstream tasks such as texture authoring, part-based editing, and semanticaware rendering.

In this paper, we introduce PartUV, a part-based UV unwrapping pipeline for 3D meshes that generates UV mappings with a small number of part-aligned charts while maintaining low distortion, as well as robust and efficient processing—typically completed within a few to several tens of seconds. PartUV builds on a recent learning-based method, PartField [Liu et al. 2025], which produces a hierarchical part tree for the input mesh. PartUV also proposes several novel geometric heuristics that further decompose simple local parts into charts that can be flattened with minimal distortion. Combining high-level semantic decomposition from PartField with fine-grained geometric cuts, PartUV employs a top-down recursive search that minimizes chart count while keeping each chart’s distortion below a user-specified threshold. PartUV uses established surface parameterization algorithms (e.g., ABF++ [Sheffer et al. 2005]) for chart flattening and proven packing algorithms for optimal atlas layout. To ensure high speed and robustness, the pipeline incorporates extensive parallelization and acceleration strategies, as well as dedicated handling of non-manifold and degenerate cases.

By explicitly incorporating semantic priors, our approach yields several key benefits. First, semantics improve decomposition by reducing excessive reliance on local geometry, which often causes over-segmentation and long runtimes. Second, semantic cues preserve the coherence of object parts, preventing chart boundaries from cutting through meaningful regions (e.g., across the flat surface of a TV screen, as in Figure 7, or a human face), thereby facilitating editing and rendering tasks. Third, semantic grouping naturally supports better chart packing strategies, allowing related charts to be organized together—either within a single atlas (top right of Figure 1) or across multiple atlases (bottom right of Figure 1). This enhances organizational clarity and simplifies the process of locating and editing related charts. Finally, seams guided by semantic boundaries tend to fall in perceptually unobtrusive locations, improving the visual quality of textured models.

We evaluate PartUV on four datasets spanning man-made, AIgenerated, CAD models, and common 3D shapes (e.g., Stanford Bunny, XYZ Dragon), and compare it against widely used UV unwrapping tools such as xatlas [Young 2019], Blender [Community [n.d.]], and Open3D [Zhou et al. 2018a], as well as the recent neural-based methods [Srinivasan et al. 2024]. As shown in Figure 1, our method decomposes input meshes into significantly fewer charts—also resulting in shorter seam lengths—while maintaining low angular and area distortion comparable to baseline methods. The incorporation of explicit part priors not only helps the segmented charts better align with part boundaries but also enables new applications, such as packing semantic-aligned parts into separate atlases. Moreover, PartUV maintains a high success rate, handles a wide

range of meshes (including large, complex, and non-manifold ones), and processes each mesh quickly—typically within a few to several tens of seconds.

- 2 Related Work

- 2.1 Mesh Parameterization

Disregarding mesh decomposition, mesh parameterization—the process of mapping a 3D mesh to a 2D domain while minimizing various distortion metrics (e.g., isometric, conformal, equiareal)—is a fundamental operation in UV unwrapping [Rabinovich et al. 2017; Sawhney and Crane 2017; Schüller et al. 2013]. Many methods directly optimize UV coordinates based on mesh connectivity by solving linear or nonlinear systems under fixed or free boundary conditions. These include barycentric embeddings like Tutte’s embedding [Tutte 1963], Laplacianeigen-decomposition techniques [Belkin and Niyogi 2003; Mullen et al. 2008; Taubin 1995], and distortion-minimization methods [Ben-Chen et al. 2008; Desbrun et al. 2002; Hormann and Greiner 2000; Ray and Lévy 2003; Sander et al. 2001; Su et al. 2016; Yan et al. 2005; Yueh et al. 2019], such as Least Squares Conformal Maps (LSCM) [Lévy et al. 2002]. Some adopt local/global strategies that combine per-triangle transformations (e.g., rotations) with global stitching [Alexa et al. 2023; Liu et al. 2008; Sorkine and Alexa 2007]. In contrast, Angle-Based Flattening (ABF) methods [Sheffer and De Sturler 2000; Sheffer and de Sturler 2001; Sheffer et al. 2005; Zayer et al. 2007] optimize triangle angles before converting them into UVs, typically offering better angle preservation than LSCM but at higher computational cost due to nonlinear solvers. In this paper, we adopt ABF for base unfolding, integrating it with various mesh decomposition strategies.

- 2.2 Mesh Segmentation for UV Unwrapping

Mesh segmentation is a key step in UV unwrapping, aiming to divide a complex 3D mesh into simpler patches, or charts, each of which can be flattened with minimal distortion. Existing approaches can be broadly categorized into four main strategies. Top-down methods recursively subdivide the mesh, often using spectral analysis [Liu and Zhang 2007a,b; Pothen et al. 1990] or cuts along feature lines [Zhang et al. 2005] and high-curvature regions [Lavoué et al. 2005]. Bottomup aggregation [Bhargava et al. 2025; Julius et al. 2005; Kalvin and Taylor 1996; Lévy et al. 2002; Pulla et al. 2001; Sorkine et al. 2002; Takahashi et al. 2011; Yamauchi et al. 2005; Zhou et al. 2004] grows charts from small seeds (e.g., triangles), merging them based on planarity or distortion bounds to maximize chart size under quality constraints. Cut optimization [Carr et al. 2006; Gu et al. 2002; Li et al. 2018; Pietroni et al. 2009; Poranne et al. 2017; Sander et al. 2003; Takahashi et al. 2011] focuses on identifying optimal seam networks that minimize seam length, reduce flattening distortion, or better suit specific parameterization goals. Clustering-based methods [Katz and Tal 2003; Roy 2023] group mesh elements based on geometric similarity to form coherent segments. Commonly used tools such as xatlas [Young 2019], Blender [Community [n.d.]], and Open3D [Zhou et al. 2018a] primarily adopt bottom-up aggregation strategies that rely solely on local geometric properties. In contrast,

PartUV integrates high-level semantic part priors from a learningbased module with novel local geometric decomposition strategies via a top-down recursive tree search.

- 2.3 Learning-Based Parameterization and Segmentation

Representing UV mappings with neural networks has attracted attention due to their differentiability and composability. Neural Surface Maps [Morreale et al. 2021] pioneered this direction by learning mappings across collections of surfaces. Several approaches jointly optimize UV parameterization and 3D reconstruction from multi-view images using cycle-consistency and distortion-based losses[Dasetal.2022;Xiangetal.2021]. Others,suchasNUVO[Srinivasan et al. 2024], solve for parameterizations directly from sampled 3D points. More recent methods [Zhang et al. 2024; Zhao et al. 2025] design neural networks to emulate physical operations, including face cutting, UV deformation, and unwrapping. However, all the aforementioned techniques typically rely on per-shape optimization, making them computationally expensive and often requiring tens of minutes per shape. A few feedforward models address specific tasks more efficiently, such as intra-category texture transfer [Chen et al. 2022] and low-distortion patch selection [Liu et al. 2023].

To enhance UV chart decomposition, we explore the integration of semantic part priors. Traditional learning-based part segmentation methods [Jiang et al. 2020; Qian et al. 2022; Vu et al. 2022] are restricted to closed-set categories due to the limited scale of part-annotated datasets [Mo et al. 2019; Yi et al. 2016]. Recent approaches [Guo et al. 2024; He et al. 2024; Xu et al. 2023, 2024; Yang et al. 2023; Yin et al. 2024; Zhou et al. 2018a] lifting priors from 2D vision to 3D models [Kirillov et al. 2023; Li et al. 2022] for openworld 3D part segmentation but rely on per-shape optimization, leading to slow runtimes and noise sensitivity. In contrast, a recent method, PartField [Liu et al. 2025], introduces a feedforward model that learns part-aware feature fields for fast, hierarchical 3D part decomposition. While semantic priors intuitively benefit UV unwrapping, naïve integration can yield suboptimal results. PartUV addresses this with a novel top-down strategy that interleaves semantic segmentation and geometric flattening.

3 Method

- 3.1 Overview

For a 3D mesh M = (𝑉, 𝐹), PartUV decomposes the mesh faces into a small collection of disjoint charts:

𝐾

𝐶𝑘, with 𝐶𝑖 ∩𝐶𝑗 = ∅ (𝑖 ≠ 𝑗), (1)

𝐹 =

𝑘=1

where each chart 𝐶𝑘 is a connected subset of faces. We utilize an Angle-Based Flattening algorithm(ABF++), to flatten each chart onto a 2D plane, yielding mappings:

𝜙𝑘 : 𝐶𝑘 −→ R2, (2) so that each vertex 𝑣𝑖 ∈ 𝑉 receives a corresponding 2D UV coordinate u𝑖 = 𝜙𝑘(𝑣𝑖) ∈ R2, determined by the chart it belongs.

The primary challenge lies in decomposing the mesh into charts {𝐶𝑘}, a task essentially equivalent to identifying optimal seams for mesh segmentation. Many state-of-the-art methods, such as xatlas and Blender’s Smart UV, rely on geometric heuristics—such as

[Figure 2]

- Fig. 2. Pipeline of PartUV. Given a mesh M, we first leverage the learning-based method PartField to predict a part-aware feature field. By applying a clustering algorithm to this field, we obtain a hierarchical part tree T for the input mesh M. We then recursively traverse the part tree T starting from the root node. For each visited node P, we apply two novel geometry-based strategies to segment the corresponding part mesh into multiple sets of charts 𝒞. Each chart in the set is then flattened using the ABF algorithm [Sheffer et al. 2005], and its distortion is evaluated. If the distortion exceeds a user-specified threshold 𝜏, we recursively traverse the left and right children of the part tree T. Otherwise, we adopt the segmented charts and their corresponding UV mappings for the part mesh.

bottom-up greedy chart expansion based on face normals or distortion metrics—to generate these charts. However, these strategies typically produce overly fragmented charts and unintuitive or suboptimal boundaries, often splitting semantically coherent regions across multiple charts.

triplane representation. Points whose features are similar—as measured by cosine similarity—are therefore more likely to belong to the same part.

3.3 Top-Down Recursive Tree Search

In contrast, PartUV adopts a coarse-to-fine, two-stage strategy for mesh decomposition. At a high level, it employs a recent learningbased method called PartField (introduced in Section 3.2) to partition mesh faces into semantically coherent parts: 𝐹 = 𝑚 𝑀=1 𝑃𝑚, where each part𝑃𝑘 exhibits relatively simple geometry—for instance, cylindrical limbs, spherical toes. Subsequently, we introduce two geometry-based heuristics (detailed in Section 3.4) to further segment each part into a small set of charts: 𝑃𝑚 = 𝑛 𝑁=𝑚1 𝐶𝑚,𝑛, ensuring each resulting chart 𝐶𝑚,𝑛 can be flattened onto a 2D plane with minimal distortion.

After obtaining the hierarchical part-based feature field using PartField, we first compute a representative part feature for each triangular face by uniformly sampling 𝑠 3D points within the face and averaging their corresponding point features. Next, we construct a hierarchical part tree T using agglomerative clustering [Johnson 1967] on these face features. In this hierarchical tree structure, the leaf nodes represent individual triangular faces, and the root node encompasses the entire mesh.

A straightforward approach to mesh decomposition might involve using PartField to directly generate a fixed number of parts and attempting to flatten each to 2D individually. Alternatively, one could adaptively traverse T from the root downward, checking whether each node’s corresponding geometry can be flattened into a 2D chart with minimal distortion. However, we observe that relying solely on parts generated by PartField often leads to suboptimal results. This limitation arises because PartField primarily focuses on semantic or coarse geometric partitioning and is less effective for finer-scale, UV-related decompositions—such as accurately segmenting cylindrical or spherical regions into charts that minimize distortion. To address this challenge, PartUV leverages PartField primarily for high-level semantic decomposition into geometrically simpler subparts and employs two geometry-based heuristics (elaborated in Section 3.4) to further divide these semantically meaningful parts into smaller charts amenable to low-distortion flattening. The search algorithm interleaves these two strategies.

To accomplish this, PartUV employs a top-down recursive decomposition search (detailed in Section 3.3) that minimizes the total number of generated charts while ensuring the distortion for each chart remains below a user-specified threshold 𝜏. Since exhaustive decomposition searches can incur substantial computational overhead, we introduce acceleration and parallelization techniques for efficiency. To ensure a complete and robust pipeline, Section 3.5 also describes additional preprocessing and postprocessing procedures, including handling non-manifold and multi-component meshes, as well as performing UV packing.

- 3.2 Preliminary: PartField

PartField [Liu et al. 2025] trains a feed-forward neural network that takes a 3D mesh as input and predicts a continuous, part-based feature field encoded as a triplane. By leveraging extensive contrastive learning on part-labeled 3D data and large-scale unlabeled 3D data with 2D pseudo part labels, PartField learns general hierarchical concepts of semantic and geometric parts. For any 3D point, we can obtain its high-dimensional part feature by interpolating the

Formally, PartUV employs a top-down recursive decomposition search—detailed in Algorithm 1—to optimally balance the chart count against distortion constraints. Given a subtree node P (initially the root node of the tree T), a distortion threshold 𝜏, and a

Algorithm 1 PartTreeSearch Require: P (a PartField subtree node), distortion threshold𝜏, chart

budget 𝐵

Ensure: A UVChartSet 𝒞 = {𝐶1,𝐶2, · · · ,𝐶𝑘} for P’s mesh, whose charts 𝐶𝑖 have distortion ≤ 𝜏 and whose total chart count 𝑘 is ≤ 𝐵; return ⊥ if no such set exists.

- 1: procedure PartTreeSearch(P,𝜏,𝐵)
- 2: if 𝐵 < 1 then ⊲ budget exhausted
- 3: return ⊥
- 4: end if
- 5: H1 ← GenCandidatesH1(P.𝑚𝑒𝑠ℎ, 10) ⊲ Heuristic1
- 6: for all 𝒞 ∈ H1 do ⊲ 𝒞 is one of candidate 𝑈𝑉𝐶ℎ𝑎𝑟𝑡𝑆𝑒𝑡
- 7: ParameterizeABF(𝒞) ⊲ flatten and compute distortion
- 8: end for
- 9: if min𝒞∈H1 𝒞.dist > 𝜏 then ⊲ no admissible H1 candidate
- 10: 𝐿 ← PartTreeSearch(P.𝑙𝑒𝑓 𝑡_𝑐ℎ𝑖𝑙𝑑,𝜏, ∞)
- 11: 𝑅 ← PartTreeSearch(P.𝑟𝑖𝑔ℎ𝑡_𝑐ℎ𝑖𝑙𝑑,𝜏, ∞)
- 12: return 𝐿 ⊕ 𝑅 ⊲ merge chart sets from the subtrees
- 13: else
- 14: H2 ← {GenCandidateH2(P.𝑚𝑒𝑠ℎ,𝜏)} ⊲ Heuristic2
- 15: ParameterizeABF(H2[0])
- 16: S ← 𝒞 ∈ H1 ∪ H2 | 𝒞.dist ≤ 𝜏 ∧ NoOverlap(𝒞)
- 17: 𝒞best ← argmin𝒞∈S 𝒞.count
- 18: ⊲ Recurse to determine whether a better solution exists
- 19: 𝐵′ ← min(𝐵, 𝒞best.count − 1)
- 20: 𝐿 ← PartTreeSearch(P.𝑙𝑒𝑓 𝑡,𝜏,𝐵′ − 1)
- 21: 𝑅 ← PartTreeSearch(P.𝑟𝑖𝑔ℎ𝑡,𝜏,𝐵′ − 𝐿.count)
- 22: 𝒞comb ← 𝐿 ⊕ 𝑅
- 23: 𝑖𝑠_𝑣𝑎𝑙𝑖𝑑 ← 𝒞comb.dist ≤ 𝜏 ∧ NoOverlap(𝒞comb)
- 24: if 𝒞comb.count < 𝒞best.count ∧ 𝑖𝑠_𝑣𝑎𝑙𝑖𝑑 then
- 25: 𝒞best ← 𝒞comb
- 26: end if
- 27: return 𝒞best
- 28: end if
- 29: end procedure

chart budget 𝐵 (initially set to ∞), the algorithm searches for a valid chart decomposition of P’s mesh that satisfies three key conditions: each chart has a distortion of at most 𝜏, no charts overlap in 2D, and the total number of charts does not exceed 𝐵. If no feasible decomposition exists, the algorithm returns failure (⊥).

Specifically, the algorithm begins by generating candidate chart decompositions for node P’s mesh using a primary geometric heuristic (Heuristic1). Each candidate chart set 𝒞 = {𝐶𝑖}, consisting of up to 𝑡 charts, is flattened using Angle-Based Flattening (ABF), and its distortion is evaluated. If none of the candidate decompositions with up to 𝑡 charts from Heuristic1 satisfy the distortion constraint (i.e., 𝛿min > 𝜏), the algorithm utilizes the PartField tree to divide the mesh and recursively searches both the left and right child subtrees without budget constraints, then merges their respective optimal chart sets.

Conversely, if an admissible candidate decomposition is found through Heuristic1, we further refine it using a secondary heuristic (Heuristic2) designed to potentially yield fewer charts. Among these

candidates, we select the best solution 𝒞best with the minimal chart count that satisfies the distortion and overlap constraints.

Before accepting this solution, the algorithm performs a final check by recursively exploring the left and right child nodes of the PartField subtree, using a reduced chart budget (𝐵′), which is derived from the best candidate’s chart count minus one. If the combined solution from this subtree search yields a valid and superior decomposition (i.e., fewer charts than 𝒞best), it replaces the previously identified best solution. The chart budget 𝐵 prevents the search from going too deep. As the search progresses, 𝐵 tightens, and recursion stops when candidates exceed it. In practice, the search rarely goes very deep.

By strategically interleaving hierarchical semantic guidance from PartField with fine-grained geometric heuristics and systematic recursive exploration, our proposed decomposition search achieves semantically coherent, distortion-bounded, and notably compact mesh parameterizations—characterized by a small number of charts.

3.4 Geometry-Based Part Decomposition

In this section, we introduce two heuristics, Normal and Merge, to further decompose a part 𝑃𝑚 generated by PartField and exhibiting simple geometry. The goal is to divide 𝑃𝑚 into multiple charts, denoted as 𝑃𝑚 = 𝑛 𝑁=𝑚1 𝐶𝑚,𝑛, such that each chart can be flattened to 2D with low distortion.

The first heuristic, referred to as Normal (line 5 in Algorithm 1), is based on face normals. We apply an agglomerative clustering algorithm [Johnson 1967] to the face normals of 𝑃𝑚’s mesh, partitioning its triangle faces into charts, where each chart is composed of connected faces with similar normals. This clustering is performed once for 𝑃𝑚, producing 𝑡 candidate decompositions with 1 to 𝑡 charts (we use 𝑡 = 10 in our experiments). For each candidate decomposition, we apply the Angle-Based Flattening (ABF) algorithm to flatten the charts and evaluate distortion. Since ABF aims to preserve angles, we quantify distortion using an area stretch metric, defined as:

##### ∑︁

1 stretch(𝑓 )

1 |𝐶|

max stretch(𝑓 ),

distortion(𝒞) = max

, (3)

𝐶∈𝒞

𝑓 ∈𝐶

where 𝐶 denotes a single chart composed of multiple connected faces, and 𝒞 denotes the set of all charts in the decomposition. The stretch of a triangle face 𝑓 is computed as:

- 𝑓 ′∈𝐶 area2𝐷(𝑓 ′)

- 𝑓 ′∈𝐶 area3𝐷(𝑓 ′)

- area2𝐷(𝑓 )

- area3𝐷(𝑓 )

stretch(𝑓 ) =

. (4)

Note that both PartField and the Normal heuristic employ the agglomerative clustering algorithm to group faces into parts or charts. The key difference lies in the features used: PartField utilizes learned high-level part features, while heuristic Normal relies on low-level geometric face normals. These two strategies are thus consistent in spirit and complementary in practice.

The Normal heuristic is simple, fast, and often yields satisfactory results. However, we also propose a second, more computationally expensive heuristic, called Merge (line 14 in Algorithm 1), which may produce decompositions with fewer charts. Given a part 𝑃𝑚, the Merge heuristic begins by computing its oriented bounding box (OBB). It then assigns each triangle face a label from 1 to 6, corresponding to the OBB face normal with which the triangle’s

normal is most closely aligned. Using both face connectivity and these labels, we segment the faces into multiple connected components. These components are then sorted by size (small to large), and we iteratively attempt to merge each component with its adjacent neighbors, starting from the one with the longest shared edges. For each merge attempt, we temporarily merge two components and apply ABF to flatten the combined chart. The merge is accepted if the resulting chart satisfies the distortion threshold and is free of overlaps. This merging process continues until no further valid merges can be made, after which the final chart set is returned. Since the Merge heuristic often begins with many small components and performs multiple ABF calls during its iterative merging process, it is significantly more expensive than the Normal heuristic. However, it may yield better decompositions with fewer charts. Therefore, we only invoke the Merge heuristic when the Normal heuristic returns a valid (admissible) decomposition.

- 3.5 Runtime Optimization and Pre- and Post-processing

Runtime Optimization To ensure efficiency despite the computational cost of recursive decomposition and repeated ABF invocations, we adopt two key strategies during the decomposition process. First, we parallelize all recursive calls to the left and right subtrees during the top-down search, allowing the algorithm to exploit multi-core processing and significantly accelerate the overall decomposition. Second, to avoid the high cost of repeatedly invoking Angle-Based Flattening (ABF) on dense meshes, we employ a GPU-accelerated mesh simplification algorithm [Oh et al. 2025] to generate low-resolution approximations of candidate charts. During simplification, the chart boundary is kept fixed to preserve the geometric structure relevant to UV mapping. These simplified charts are used to estimate distortion metrics quickly (i.e., surrogate distortion) during intermediate evaluations. Once the final chart set is determined, ABF is applied to the original high-resolution mesh to produce accurate UV coordinates and distortion measurements.

Non-Manifold and Multi-Component Meshes Since the ABF algorithm assumes each input chart is a manifold and connected surface, additional processing is required when handling non-manifold or multi-component meshes. For non-manifold meshes, we detect all non-manifold edges—i.e., edges shared by more than two faces—and resolve them by duplicating the shared vertices and splitting each such edge into 𝑁 − 1 distinct edges, where 𝑁 is the number of incident faces, thereby converting the structure into a manifold form suitable for flattening. For meshes containing multiple connected components,weinitiallyproceedwith the proposed PartField-guided hierarchical decomposition. If a part mesh consists of disconnected components, we skip heuristic-based decomposition at that level and instead recursively explore the left and right subtrees of the PartField hierarchy. However, if PartField fails to further segment the multiple components after reaching a predefined recursion depth, we fall back to applying the geometric heuristics and ABF flattening to each connected component at that level individually, in order to avoid excessively fragmented decomposition.

UV Packing While our primary focus is on decomposing 3D meshes into charts and generating corresponding low-distortion 2D parameterizations, our method is fully compatible with a variety of existing

UV packing algorithms. A distinguishing feature of our approach is that chart decompositions are grouped based on semantically meaningful parts, enabling more structured and application-aware packing strategies. For example, charts belonging to the same part can be grouped together during packing. Alternatively, part groups can be packed into an arbitrary number of UV atlas squares (e.g., multiple [0, 1]2 spaces) with a semantically balanced distribution across atlases. This semantic hierarchy not only improves organizational clarity but also benefits downstream applications such as texture painting or editing, where charts belonging to the same part remain spatially close and are easier to manipulate collectively.

4 Experiments 4.1 Implementation Details and Evaluation Setup

Implementation Details. We train PartField [Liu et al. 2025] on Objaverse [Deitke et al. 2022] using 8 NVIDIA H100 GPUs for 15 days. During inference, we sample 10 points per face and average their features for face clustering and part tree construction. The core pipeline of PartUV is implemented in C++17, and we use UVPackmaster [Łukasz Czyż 2025] for final group-based UV packing. A distortion threshold of 1.25 is used in all experiments. For ABF++, we run 5 iterations per call, and follow Blender to set the gradient early-stop condition. Mesh simplification is controlled by a curvature-related threshold of 1e-4 and a maximum iteration count of 1,000. All parameters are fixed across experiments. All methods are evaluated on a cluster node with a 96-core Intel® Xeon® Platinum 8468 CPU and an NVIDIA H100 GPU.

Evaluation Datasets. To comprehensively evaluate the approaches acrossdiversemeshsources,qualities, and styles, we use four datasets: (a) Common Shapes [Jacobson 2013], a GitHub repository of 24 widely used models in graphics (e.g., Bimba, Igea, Stanford Bunny) with provided processed .obj files; (b) PartObjaverseTiny [Yang et al. 2024], a 200-shape subset of Objaverse [Deitke et al. 2022] featuring high-quality, man-made meshes with multiple components and smooth surfaces; (c) ABC [Koch et al. 2019], a CAD dataset of mechanical models combining sharp and smooth features—we use the first 100 meshes from its initial batch; and (d) Trellis [Xiang et al. 2024], which includes 114 AI-generated meshes from a recent 3D diffusion-based generative model. These meshes are typically noisy and geometrically low-quality, posing greater challenges than human-made counterparts.

Evaluation Metrics. We evaluate the quality of the generated UV maps from four perspectives: (1) Number of Charts: For each shape, we count the number of charts and report both the average and median values across the dataset. (2) Seam Length: We compute the seam length by summing the lengths of all chart boundary edges, with UV coordinates normalized to a [0, 1] grid. The median value is reported across the dataset. (3) Angular (Conformal) Distortion: We compute the cosine between the tangent and bitangent vectors of each face. The distortion for a shape is defined as one minus the average cosine across all faces [Srinivasan et al. 2024]. We report the average distortion across the dataset. (4) Area (Equiareal) Distortion: We compute both area distortion and overall area distortion. Area distortion is defined as in Equation 3, based on the chart with the highest distortion in each shape. Overall area distortion is computed

[Figure 3]

- Fig. 3. Quantitative comparison between Blender [Community [n.d.]], xatlas [Young 2019], and PartUV. Unlike Blender and xatlas, which rely solely on local geometric properties for UV unwrapping, PartUV integrates high-level semantic priors with low-level geometric heuristics, enabling part-based chart decomposition. As a result, it produces significantly fewer charts, with boundaries that better align with semantic parts.

by aggregating all triangles across all charts, calculating individual distortions, and averaging them. For both metrics, we first compute per-shape values and then average them across the dataset. All triangle-level distortion values are clipped to a maximum of 10. Notably, overall area distortion may be smoothed by the number of faces, while area distortion more effectively highlights problematic regions in the UV maps. We did not report stretch 𝐿2 and 𝐿∞ [Sander et al. 2001] since they are per-triangle metrics that can diverge to infinity when triangles flip or have near-zero area—issues that are common in baselines such as xatlas, making direct comparison less meaningful.

4.2 Comparison with Baselines

Baselines.We comparePartUV with commonly used tools—Blender’s Smart UV Project [Community [n.d.]], xatlas [Young 2019], and Open3D—aswellasoptimization-based methods, including Nuvo [Srinivasan et al. 2024] and OptCuts [Li et al. 2018].

Blender, xatlas, and Open3D decompose meshes into charts using bottom-up strategies guided by local geometric priors. Blender clusters triangles based on mesh normals and flattens each chart using simple planar projection. xatlas employs a greedy algorithm that balances geometric deviation, UV distortion, and seam cost, followed by Least Squares Conformal Maps (LSCM) [Lévy et al.

[Figure 4]

- Fig. 4. Open3D not only suffers from a limited success rate but also produces results with large distortion, which may reduce their practical utility.

[Figure 5]

- Fig. 5. Nuvo and OptCuts incur significantly longer optimisation times for each shape. Nuvo’s results also exhibit large distortion.

2002] for flattening. Open3D builds on Microsoft’s UVAtlas [Corporation 2023; Sander et al. 2002; Zhou et al. 2004], which uses region growing guided by the isometric stretch metric and flattens charts with LSCM. Among optimization-based methods, Nuvo [Srinivasan et al. 2024] learns a continuous UV mapping via a neural field by minimizing distortion losses with various regularizations. OptCuts jointly optimizes surface cuts and distortion under a global bijectivity constraint. Both methods require significantly longer runtimes per mesh.

Results. As shown in Table 1, Blender and xatlas often produce over-fragmented charts, whereas PartUV generates UV maps with significantly fewer charts. For instance, on the Common Shapes dataset, PartUV uses only 1/31 as many charts as Blender. Consequently, it also results in shorter seam lengths. PartUV maintains low levels of both angular and area distortion, while xatlas may

- Table 1. Quantitative comparison between Blender [Community [n.d.]], xatlas [Young 2019], and PartUV.

success average median median overall dataset method rate # of # of seam angular area area time

(%) charts ↓ charts ↓ length ↓ distort. ↑ distort. ↓ distort. ↓ (s)

Common Shapes (24 shapes)

Blender 100.0 1360.3 332.5 44.7 0.906 1.172 1.102 0.3

xatlas 100.0 974.8 301.0 42.9 0.987 1.885 1.504 77.9 ours 100.0 48.6 16.5 16.6 0.990 1.283 1.128 39.4

Trellis (114 shapes)

Blender 100.0 3352.9 1957.0 94.5 0.921 1.252 1.107 1.1

xatlas 100.0 1541.6 895.0 91.2 0.984 2.357 1.093 13.1 ours 100.0 568.7 233.5 58.8 0.981 1.373 1.145 26.0

ABC (100 shapes)

Blender 100.0 305.3 78.0 25.0 0.992 1.122 1.067 0.7

xatlas 100.0 249.6 56.0 26.5 1.000 1.192 1.030 31.0

ours 100.0 85.4 15.5 17.3 0.998 1.191 1.073 31.9 Part Objaverse Tiny (200 shapes)

Blender 100.0 1509.2 647.5 70.2 0.925 1.325 1.115 0.2 xatlas 100.0 1142.1 491.5 67.0 0.982 1.728 1.286 4.4

ours 100.0 544.8 177.0 41.8 0.983 1.305 1.112 10.1

- Table 2. Quantitative comparison between Open3d [Zhou et al. 2018b] and PartUV. Note that Open3D has a limited success rate, and the reported numbers are averaged over the easier cases it successfully completes. Despite this, Open3D still exhibits large distortion.

success average overall average median median dataset method rate angular area area # of # of seam time

(%) distort. ↑ distort. ↓ distort. ↓ charts ↓ charts ↓ length ↓ (s)

Common Shapes

Open3d 79.2 0.852 1.509 1.191 19.8 12.0 11.8 19.8 ours 100.0 0.987 1.281 1.128 24.4 12.0 15.1 52.3

Trellis

Open3d 39.5 0.859 1.931 1.264 79.8 40.0 25.9 24.1 ours 100.0 0.984 1.308 1.144 97.9 51.0 27.4 23.7

ABC

Open3d 83.0 0.878 1.459 1.162 15.0 8.0 9.3 17.6

ours 100.0 0.994 1.171 1.062 35.0 15.0 18.6 39.6 PartObjaverse Tiny

Open3d 52.5 0.799 2.772 1.295 161.5 80.0 25.6 10.0 ours 100.0 0.957 1.254 1.117 227.1 91.0 30.6 14.0

- Table 3. Quantitative comparison between Nuvo [Srinivasan et al. 2024] and PartUV.

success average overall average dataset method rate angular area area # of time

(%) distortion ↑ distortion ↓ distortion ↓ charts ↓ (s)

nuvo 100.0 0.802 2.722 1.940 17.0 2908.8

Common Shapes

###### ours 100.0 0.987 1.281 1.128 24.4 52.3

exhibit large area distortion on certain challenging shapes. Despite utilizing a more exhaustive decomposition search and a computationally expensive ABF flattening algorithm to achieve higher quality, PartUV maintains a runtime comparable to xatlas, typically completing in tens of seconds. See Figure 3 for a qualitative comparison, where PartUV leverages semantic part information to produce chart boundaries that align more closely with object semantics.

We report the comparison results with Open3D separately in Table 2 due to its failure to complete some shapes within a reasonable time. For example, on the challenging Trellis dataset, it achieves a success rate of only 39.5%. In Table 2, we report the average performance only over the easy cases that Open3D successfully processes. For these cases, Open3D achieves a similar number of charts and seam lengths compared to our method. However, this comes at the cost of significant distortion. For instance, while Blender, xatlas, and PartUV all achieve angular distortion scores mostly above 0.95 across all datasets, Open3D consistently falls below 0.9 and even 0.8 in some cases. A similar phenomenon is observed for area distortion. Please refer to Figure 4 and Figure 6 for qualitative examples, where Open3D produces large distortions, rendering the UV mappings less suitable for practical applications.

[Figure 6]

Fig. 6. Texture-map comparison among Blender, xatlas, and our PartUV.

Compared to Nuvo and OptCuts, we observe that although both methods effectively reduce the number of charts, they typically require significantly longer optimization times—often exceeding 30 minutes or even several hours. Moreover, Nuvo leads to substantially higher distortion, as shown in Table 3 and Figure 5. While OptCuts achieves low distortion, its success rate is limited: it produces outputs for only 9 out of 24 shapes in the Common Shapes dataset. Additionally, neither method incorporates the concept of semantic parts during optimization.

See supplementary material for additional qualitative examples UV efficiency comparisons, and analysis of Open3D results.

- 4.3 Applications and Analysis

In this section, we demonstrate the benefits of using our PartUV:

Texture Editing and Replacement. Since our UV mappings are less fragmented, texture maps can be more effectively edited or modified in 2D space. In Figure 7, we show that our UV maps enable clean placement of conference logos, whereas the UV maps generated by Blender or xatlas fail to do so due to their overly fragmented layouts. Figure 8 further showcases an application where the texture is replaced with various tiling patterns. Because xatlas and Blender often produce numerous small charts containing only a few triangles, noticeable artifacts appear on the mesh surface. In contrast, our method preserves significantly better visual quality.

[Figure 7]

- Fig. 7. Unlike our baselines, which generate over-fragmented UV maps that hinder 2D texture editing, PartUV produces significantly fewer charts with part-aligned boundaries, enabling more effective 2D operations.

[Figure 8]

- Fig. 8. Our UV map enables easy texture replacement, whereas xatlas / Blender maps cause severe artefacts due to over-segmentation.

Texture Compression. UV maps always require padding between charts. When the UV layout is over-segmented, more padding is needed, which increases the risk of color bleeding. In Figure 10, we demonstrate that reducing the UV map resolution from 1024 to 128—a common setting in mobile games—results in noticeable color bleeding for textures generated by xatlas and Blender. In contrast, PartUV is free from such issues.

Multi-Atlas Wrapping. As shown in Figure 1 and Figure 9, PartUV supports part-based UV packing. Given the desired number of atlases, it can automatically extract semantic-meaningful parts and pack across separate atlases, facilitating downstream applications such as texture editing.

Adaptive Threshold Adjustment. PartUV allows users to specify a distortion threshold 𝜏, enabling adaptive control over the trade-off between the number of charts and the distortion in the generated UV maps, as shown in Figure 11.

4.4 Ablation Studies

PartField. We integrate semantic part priors with geometric heuristics through an adaptive recursive tree search. A naive way to combine them is to first use PartField to decompose the shape into a fixed set of 20 parts, and then further decompose each part using the two heuristics. As shown in Table 4(a), this naive combination results in high distortion, as some parts may remain too complex to

[Figure 9]

Fig. 9. Our part-based UV unwrapping can pack all charts into a single atlas or multiple atlases by semantic part, aiding tasks such as 2-D editing.

[Figure 10]

Fig. 10. xatlas and Blender generate over-fragmented UV maps that may introduce color bleeding, especially with low-resolution texture maps (e.g., in mobile games). In contrast, our results are free from such issues.

#### Table 4. Ablation study conducted on the Trellis dataset.

average median area angular seam

id # charts ↓ # charts ↓ distort.↓ distort.↑ length ↓ time

version

- a fixed 20 parts 397.02 223.00 2.18 0.9687 66.42 207.75

- b replace PF feat. with face normal 574.43 259.50 1.28 0.9708 64.46 83.67

- c no merge 778.64 299.50 1.42 0.9815 63.25 24.59

- d no recursion 633.29 255.50 1.35 0.9822 61.02 33.82

- e no distortion surrogate 580.52 248.00 1.25 0.9822 60.98 32.88

- f full 568.75 233.50 1.37 0.9811 59.85 26.03

be flattened with low distortion. We also experimented with replacing the PartField features with face normals and then applying the original recursive tree search. As shown in Table 4(b), this variant doubles the runtime, increases the number of charts, and produces chart boundaries that no longer align with semantic parts.

MergeHeuristic.We proposeaninnovativegeometry-based heuris-

tic, Merge, for chart decomposition. When this heuristic is removed and only the Normal heuristic is used (Table 4(c)), we observe an increase in the number of charts.

Recursion. As shown in Alg. 1, Line 18, we do not immediately return the first solution found but continue the search to find potentially better solutions. When this strategy is removed (Table 4(d)), we observe a significant increase in the number of charts.

Distortion Surrogate. To accelerate the search, we first simplify the mesh during intermediate iterations and then apply the ABF algorithm to the simplified mesh to compute an approximate distortion, which helps guide the search more efficiently. When this strategy is removed (Table 4(e)), we observe an increase in runtime.

We show more ablations and visualizations, including comparisons of different flattening algorithms, in the Appendix.

- 4.5 95th Percentile Distortion Metrics

To evaluateworst-casebehavior,we additionally report 95th-percentile distortion metrics and compare them with baseline methods on four datasets. As shown in Table 5, we consider two metrics: distortion 95th-shape, defined as the 95th-percentile area distortion across all shapes in a dataset, and distortion 95th-chart, which first computes the 95th-percentile area distortion across all charts within a shape and then averages the results over all shapes in the dataset. As the

[Figure 11]

Fig. 11. PartUV lets users set a distortion threshold, flexibly controlling both distortion and chart count.

Table 5. 95th-percentile area distortion metrics across four datasets.

Dataset Metric Blender xatlas ours

distortion 95th-shape ↓ 1.175 1.726 1.273 distortion 95th-chart ↓ 1.093 1.041 1.133

ABC

distortion 95th-shape ↓ 1.885 1.504 1.404 distortion 95th-chart ↓ 1.139 1.131 1.169

Common Shapes

distortion 95th-shape ↓ 1.728 1.286 1.271 distortion 95th-chart ↓ 1.132 1.079 1.116

PartObjaverseTiny

distortion 95th-shape ↓ 1.319 4.701 1.442 distortion 95th-chart ↓ 1.120 1.099 1.220

Trellis

table shows, our method consistently yields low 95th-percentile distortion values, with a maximum of 1.442, whereas baseline methods often produce much higher distortions (e.g., 4.701, 1.885, 1.728). These results demonstrate that our approach achieves more robust and stable performance under challenging cases.

- 5 Discussion on Failure Cases

Our method struggles with poor input mesh topology. For example, it cannot handle 3D meshes containing self-intersections. In such cases, the algorithm may recurse deeply in an attempt to resolve the intersections, which can lead to fragmented charts. Moreover, when input meshes are extremely fragmented—for instance, those with over 1000 components—the unwrapping results also become highly fragmented. In these cases, remeshing may be required before UV unwrapping. However, our method performs well on general meshes with few components and does not require meshes to be watertight or manifold.

- 6 Conclusion In this paper, we propose PartUV, a novel framework for UV unwrapping that strategically integrates semantic part priors from learning-based methods with two novel geometric heuristics. PartUV outperforms existing approaches by generating significantly fewer charts, low distortion, and chart boundaries that align with semantic parts. We demonstrate the advantages of this pipeline through several applications.

## Appendix

[Figure 12]

Fig. 12. A demonstration of effects from Open3D’s higher distortion results, creating curved (from angular distortion) and uneven (from area distortion) stripes.

Table 6. Efficiency (↑) comparison across methods

ours Blender Open3D xatlas

Part-Objaverse-Tiny 0.606 0.514 0.332 0.666 Trellis 0.571 0.430 0.242 0.619 ABC 0.568 0.554 0.490 0.646 Common Shapes 0.575 0.460 0.461 0.561

A1 Additional Results and Comparisons

- A1.1 More Qualitative Results

In the supplementary material, we provide an HTML file containing more qualitative and quantitative comparisons between our method and our baselines. The examples are gathered across the four datasets we used in the main paper. The metrics are computed and reported for individual shapes.

- A1.2 Efficiency

As we create the UV map with regard to semantic features and groups, it may raise doubts regarding the final efficiency of the resulting UV maps. In Table 6, we show the comparison of the average efficiency of valid results between our method and the baselines. In our experiments, we always group the charts from the same parts when packing, and we use UVPackMaster [Łukasz Czyż 2025] to get the final packed UV map. We set the "heuristic search" time to 3 seconds for UVPackMaster, while most meshes finish within microseconds. We define efficiency as the total area of valid 2D faces within the normalized 0–1 UV space. It can be seen that our method does not hurt the overall efficiency, and it remains competitive compared to our baseline methods.

- A1.3 Analysis and Visualization of Open3D Results

In our experiments, particularly on the more challenging meshes, Open3D frequently crashed or exceeded the allotted time budget.

[Figure 13]

Fig. 13. When using the face normals as agglomerative features for topdown tree construction, the pipeline generates messier results with no semantic alignment.

We designate a shape as timed-out when its runtime surpasses 30 seconds plus three times the longest runtime among other methods on the current shape.

Even for the shapes where Open3D successfully produces a UV map, the results still exhibit notable shortcomings. At first glance, Open3D seemingly generates results with chart numbers similar to our method, albeit with slightly higher distortion. However, such differences in distortion can significantly affect downstream tasks. In

- Fig. 12, we show one example from the ABC dataset, which Open3D creates a layout with a similar chart number with ours. However, with 0.2 difference in area distortion and 0.1 difference in angular distortion, the deformation of the texture is clearly noticeable. In particular, when using a striped texture, angular distortion leads to curved lines (bottom half of highlighted region), while area distortion causes uneven width (top half of highlighted region)—both prominently visible in the red-marked rectangle of the figure, not to mention the seams that disregard underlying geometric features. In contrast, our results exhibit minimal distortion, maintain the regularity of the shape, and feature seams that largely respect the underlying geometric structure.

A2 Additional Ablation Results A2.1 Replacing PartField feature with Face Normals

In our main pipeline, we create a top-down tree of faces with agglomerative clustering of PartField-predicted features. A simpler alternative would be replacing such features with face normals. In

- Fig. 13, we provide some visual examples of meshes using normal as Agglomerative features versus using the PartField-predicted ones. It can be seen that using normal is substantially prone to producing curly shapes, like the cherry on the left. Moreover, though being able to predict a similar number of charts under the same heuristic settings, using normal as the agglomerative features leads to way less organized and neat UV layouts, as exemplified on the right of the figure.

[Figure 14]

Fig. 14. Visual comparison of with and without the merge heuristics.

#### Table 7. Comparison between using ABF++ (Ours) and LSCM (OursLSCM) in our pipeline.

average median area seam time

# charts ↓ # charts ↓ distort. ↓ length ↓ (s)

Ours-LSCM 1154.12 363.00 1.27 65.48 58.61 Ours 538.81 221.50 1.30 57.92 41.88

- A2.2 Without Merge Heuristic

In the main manuscript, we propose two heuristics used to unwrap a part into charts, Normal and Merge. Despite the higher cost of computation, Merge often produces more visually appealing UV maps with fewer charts. As illustrated in Fig. 14, the Merge heuristic can usually unwrap the part in an "unfolding" manner, creating less number of charts with a neater layout, comparing under the same distortion threshold requirements with using only the Normal Heuristics.

- A2.3 Without Normal Heuristic

With two heuristics included in the pipeline, one can also opt to use only the Merge throughout the pipeline. We show several example results in Fig. 15. Due to its high computational cost, using only Merge, especially at the beginning, incurs substantial runtime overhead. To be precise, the Merge heuristic would compute an oriented bounding box (OBB) and use projection to get initial charts, and it could get hundreds of charts from a bumpy input. Trying to merge all of them would introduce significant overhead for such bumpy/uneven meshes. For example, on the bimba mesh on the right side of the figure, the runtime increases from 147 seconds to over 892 seconds. Moreover, due to its bumpy geometry, the merging process often results in overlapping, leading to an increase in the number of charts. In essence, Merge could yield better results, but only on simpler shapes. Therefore, it is most effective as a complement to the Normal heuristic and should be invoked only when Normal produces a sufficiently good result.

[Figure 15]

Fig. 15. Using only Merge heuristic from the beginning would incur more runtime, and more fragmented charts on bumpy meshes.

A2.4 Using LSCM instead of ABF++

In our main pipeline, we adopt ABF++ [Sheffer et al. 2005] as the primary flattening algorithm due to its fast performance and lowdistortion mappings. Most of our baselines utilize LSCM [Lévy et al. 2002], which generally offers faster runtimes thanks to its linear energy formulation. However, despite its speed, LSCM generally produces inferior mapping results compared to ABF++. Such suboptimal outputs can trigger additional recursions in our pipeline, resulting in increased runtimes and more fragmented UV charts. As shown in Table 7, using LSCM ultimately leads to both higher runtime and a larger number of charts on the Trellis dataset.

References

Marc Alexa, Daniel Cohen-Or, and David Levin. 2023. As-rigid-as-possible shape interpolation. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 165– 172.

Mikhail Belkin and Partha Niyogi. 2003. Laplacian eigenmaps for dimensionality reduction and data representation. Neural computation 15, 6 (2003), 1373–1396. Mirela Ben-Chen, Craig Gotsman, and Guy Bunin. 2008. Conformal flattening by curvature prescription and metric scaling. In Computer Graphics Forum, Vol. 27. Wiley Online Library, 449–458.

Manas Bhargava, Camille Schreck, Marco Freire, Pierre-Alexandre Hugron, Sylvain Lefebvre, Silvia Sellán, and Bernd Bickel. 2025. Mesh Simplification for Unfolding. In Computer Graphics Forum, Vol. 44. Wiley Online Library, e15269.

Nathan A Carr, Jared Hoberock, Keenan Crane, and John C Hart. 2006. Rectangular multi-chart geometry images. In Symposium on geometry processing. 181–190.

Zhiqin Chen, Kangxue Yin, and Sanja Fidler. 2022. Auv-net: Learning aligned uv maps for texture transfer and synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1465–1474.

Blender Online Community. [n.d.]. Blender - a 3D modelling and rendering package. http://www.blender.org. Microsoft Corporation. 2023. UVAtlas: Chart-based UV Atlas Generation Library. https://github.com/microsoft/UVAtlas. Accessed: 2025-05-21.

Sagnik Das, Ke Ma, Zhixin Shu, and Dimitris Samaras. 2022. Learning an Isometric Surface Parameterization for Texture Unwrapping. European Conference on Computer Vision (2022).

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2022. Objaverse: A Universe of Annotated 3D Objects. arXiv preprint arXiv:2212.08051 (2022).

Mathieu Desbrun, Mark Meyer, and Pierre Alliez. 2002. Intrinsic parameterizations of surface meshes. In Computer graphics forum, Vol. 21. Wiley Online Library, 209–218.

Xianfeng Gu, Steven J Gortler, and Hugues Hoppe. 2002. Geometry images. In Proceedings of the 29th annual conference on Computer graphics and interactive techniques. 355–361.

Haoyu Guo, He Zhu, Sida Peng, Yuang Wang, Yujun Shen, Ruizhen Hu, and Xiaowei Zhou. 2024. Sam-guided graph cut for 3d instance segmentation. In European Conference on Computer Vision. Springer, 234–251.

Qingdong He, Jinlong Peng, Zhengkai Jiang, Xiaobin Hu, Jiangning Zhang, Qiang Nie, Yabiao Wang, and Chengjie Wang. 2024. PointSeg: A Training-Free Paradigm for 3D Scene Segmentation via Foundation Models. arXiv preprint arXiv:2403.06403 (2024).

Kai Hormann and Günther Greiner. 2000. MIPS: An efficient global parametrization method. Curve and Surface Design: Saint-Malo 1999 (2000), 153–162. Alec Jacobson. 2013. common-3d-test-models. https://github.com/alecjacobson/ common-3d-test-models. Accessed: 2025-05-21.

Alec Jacobson and contributors. 2023. common-3d-test-models: Repository of common 3D test meshes. https://github.com/alecjacobson/common-3d-test-models. Git commit 8a4f864, accessed 2025-05-15.

Li Jiang, Hengshuang Zhao, Shaoshuai Shi, Shu Liu, Chi-Wing Fu, and Jiaya Jia. 2020. Pointgroup: Dual-set point grouping for 3d instance segmentation. In Proceedings of the IEEE/CVF conference on computer vision and Pattern recognition. 4867–4876. Stephen C Johnson. 1967. Hierarchical clustering schemes. Psychometrika 32, 3 (1967),

241–254. Dan Julius, Vladislav Kraevoy, and Alla Sheffer. 2005. D-charts: Quasi-developable mesh segmentation. In Computer Graphics Forum, Vol. 24. Citeseer, 581–590. Alan D Kalvin and Russell H Taylor. 1996. Superfaces: Polygonal mesh simplification with bounded error. IEEE Computer Graphics and Applications 16, 3 (1996), 64–77. Sagi Katz and Ayellet Tal. 2003. Hierarchical mesh decomposition using fuzzy clustering and cuts. ACM transactions on graphics (TOG) 22, 3 (2003), 954–961.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision. 4015–4026.

Sebastian Koch, Albert Matveev, Zhongshi Jiang, Francis Williams, Alexey Artemov, Evgeny Burnaev, Marc Alexa, Denis Zorin, and Daniele Panozzo. 2019. Abc: A big cad model dataset for geometric deep learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 9601–9611.

Guillaume Lavoué, Florent Dupont, and Atilla Baskurt. 2005. A new CAD mesh segmentation method, based on curvature tensor analysis. Computer-Aided Design 37, 10 (2005), 975–987.

Bruno Lévy, Sylvain Petitjean, Nicolas Ray, and Jérome Maillot. 2002. Least squares conformal maps for automatic texture atlas generation. ACM Trans. Graph. 21, 3 (July 2002), 362–371. https://doi.org/10.1145/566654.566590

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. 2022. Grounded language-image pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10965–10975.

Minchen Li, Danny M Kaufman, Vladimir G Kim, Justin Solomon, and Alla Sheffer. 2018. Optcuts: Joint optimization of surface cuts and parameterization. ACM transactions on graphics (TOG) 37, 6 (2018), 1–13.

Ligang Liu, Lei Zhang, Yin Xu, Craig Gotsman, and Steven J Gortler. 2008. A local/global approach to mesh parameterization. In Computer graphics forum, Vol. 27. Wiley Online Library, 1495–1504.

Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. 2025. PARTFIELD: Learning 3D Feature Fields for Part Segmentation and Beyond. arXiv preprint arXiv:2504.11451 (2025).

Richard Liu, Noam Aigerman, Vladimir G Kim, and Rana Hanocka. 2023. Da wand: Distortion-aware selection using neural mesh parameterization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16739–16749.

- Rong Liu and Hao Zhang. 2007a. Mesh segmentation via spectral embedding and contour analysis. In Computer Graphics Forum, Vol. 26. Wiley Online Library, 385– 394.
- Rong Liu and Hao Zhang. 2007b. Mesh Segmentation via Spectral Embedding and Contour Analysis. Computer Graphics Forum (Proc. Eurographics 2007) 26, 3 (2007), 385–394. https://doi.org/10.1111/j.1467-8659.2007.01061.x

William E Lorensen and Harvey E Cline. 1998. Marching cubes: A high resolution 3D surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field. 347–353.

Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. 2019. Partnet: A large-scale benchmark for fine-grained and hierarchical part-level 3d object understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 909–918.

Luca Morreale, Noam Aigerman, Vladimir G. Kim, and Niloy J. Mitra. 2021. Neural Surface Maps. In Conference on Computer Vision and Pattern Recognition.

Patrick Mullen, Yiying Tong, Pierre Alliez, and Mathieu Desbrun. 2008. Spectral conformal parameterization. In Computer Graphics Forum, Vol. 27. Wiley Online Library, 1487–1494.

Seonghun Oh, Xiaodi Yuan, Xinyue Wei, Ruoxi Shi, Fanbo Xiang, Minghua Liu, and Hao Su. 2025. PaMO: Parallel Mesh Optimization for Intersection-Free Low-Poly Modeling on the GPU. arXiv preprint arXiv:2509.05595 (2025).

Nico Pietroni, Marco Tarini, and Paolo Cignoni. 2009. Almost isometric mesh parameterization through abstract domains. IEEE Transactions on Visualization and Computer Graphics 16, 4 (2009), 621–635.

Roi Poranne, Marco Tarini, Sandro Huber, Daniele Panozzo, and Olga Sorkine-Hornung.

2017. AutoCuts: Simultaneous Distortion and Cut Optimization for UV Mapping. ACM Transactions on Graphics (2017).

Alex Pothen, Horst D Simon, and Kang-Pu Liou. 1990. Partitioning sparse matrices with eigenvectors of graphs. SIAM journal on matrix analysis and applications 11, 3

(1990), 430–452.

Sandeep Pulla, Anshuman Razdan, and Gerald Farin. 2001. Improved curvature estimation for watershed segmentation of 3-dimensional meshes. IEEE Transactions on Visualization and Computer Graphics 5, 4 (2001), 308–321.

Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. 2022. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. Advances in neural information processing systems 35 (2022), 23192–23204.

Michael Rabinovich, Roi Poranne, Daniele Panozzo, and Olga Sorkine-Hornung. 2017. Scalable locally injective mappings. ACM Transactions on Graphics (TOG) 36, 4

(2017), 1.

Nicolas Ray and Bruno Lévy. 2003. Hierarchical least squares conformal map. 11th Pacific Conference onComputer Graphics and Applications, 2003. Proceedings. (2003), 263–270. https://api.semanticscholar.org/CorpusID:1121017

Bruno Roy. 2023. Neural ShDF: Reviving an Efficient and Consistent Mesh Segmentation Method. arXiv preprint arXiv:2306.11737 (2023). Pedro V Sander, Steven Gortler, John Snyder, and Hugues Hoppe. 2002. Signalspecialized parameterization. (2002).

Pedro V Sander, John Snyder, Steven J Gortler, and Hugues Hoppe. 2001. Texture mapping progressive meshes. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques. 409–416.

Pedro V. Sander, Zoë J. Wood, Steven J. Gortler, John M. Snyder, and Hugues Hoppe.

2003. Multi-Chart Geometry Images. In Eurographics Symposium on Geometry Processing. https://api.semanticscholar.org/CorpusID:4893489

Rohan Sawhney and Keenan Crane. 2017. Boundary first flattening. ACM Transactions on Graphics (ToG) 37, 1 (2017), 1–14.

Christian Schüller, Ladislav Kavan, Daniele Panozzo, and Olga Sorkine-Hornung. 2013. Locally injective mappings. In Computer Graphics Forum, Vol. 32. Wiley Online Library, 125–135.

Alla Sheffer and Eric De Sturler. 2000. Surface Parameterization for Meshing by Triangulation Flattenin. Alla Sheffer and Eric de Sturler. 2001. Parameterization of faceted surfaces for meshing using angle-based flattening. Engineering with computers 17 (2001), 326–337.

Alla Sheffer, Bruno Lévy, Maxim Mogilnitsky, and Alexander Bogomyakov. 2005. ABF++: fast and robust angle based flattening. ACM Transactions on Graphics (TOG) 24, 2

(2005), 311–330. Olga Sorkine and Marc Alexa. 2007. As-rigid-as-possible surface modeling. In Symposium on Geometry processing, Vol. 4. Citeseer, 109–116.

Olga Sorkine, Daniel Cohen-Or, Rony Goldenthal, and Dani Lischinski. 2002. Boundeddistortion piecewise mesh parameterization. In IEEE Visualization, 2002. VIS 2002. IEEE, 355–362.

Pratul P Srinivasan, Stephan J Garbin, Dor Verbin, Jonathan T Barron, and Ben Mildenhall. 2024. Nuvo: Neural uv mapping for unruly 3d representations. In European Conference on Computer Vision. Springer, 18–34.

Kehua Su, Li Cui, Kun Qian, Na Lei, Junwei Zhang, Min Zhang, and Xianfeng David Gu.

2016. Area-preserving mesh parameterization for poly-annulus surfaces based on optimal mass transportation. Computer Aided Geometric Design 46 (2016), 76–91.

Shigeo Takahashi, Hsiang-Yun Wu, Seow Hui Saw, Chun-Cheng Lin, and Hsu-Chun Yen. 2011. Optimized topological surgery for unfolding 3d meshes. In Computer graphics forum, Vol. 30. Wiley Online Library, 2077–2086.

Gabriel Taubin. 1995. A signal processing approach to fair surface design. In Proceedings of the 22nd annual conference on Computer graphics and interactive techniques. 351– 358.

William Thomas Tutte. 1963. How to draw a graph. Proceedings of the London Mathematical Society 3, 1 (1963), 743–767.

Thang Vu, Kookhoi Kim, Tung M Luu, Thanh Nguyen, and Chang D Yoo. 2022. Softgroup for 3d instance segmentation on point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2708–2717.

Fanbo Xiang, Zexiang Xu, Milos Hasan, Yannick Hold-Geoffroy, Kalyan Sunkavalli, and Hao Su. 2021. Neutex: Neural texture mapping for volumetric neural rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7119–7128.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2024. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506 (2024).

Mutian Xu, Xingyilang Yin, Lingteng Qiu, Yang Liu, Xin Tong, and Xiaoguang Han.

2023. Sampro3d: Locating sam prompts in 3d for zero-shot scene segmentation. arXiv preprint arXiv:2311.17707 (2023).

Xiuwei Xu, Huangxing Chen, Linqing Zhao, Ziwei Wang, Jie Zhou, and Jiwen Lu.

2024. Embodiedsam: Online segment any 3d thing in real time. arXiv preprint arXiv:2408.11811 (2024).

Hitoshi Yamauchi, Stefan Gumhold, Rhaleb Zayer, and Hans-Peter Seidel. 2005. Mesh segmentation driven by Gaussian curvature. The Visual Computer 21 (2005), 659– 668.

Jingqi Yan, Xin Yang, Pengfei Shi, and David Zhang. 2005. Mesh parameterization by minimizing the synthesized distortion metric with the coefficient-optimizing algorithm. IEEE transactions on visualization and computer graphics 12, 1 (2005), 83–92.

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y Lam, Yan-Pei Cao, and Xihui Liu. 2024. Sampart3d: Segment any part in 3d objects. arXiv preprint arXiv:2411.07184 (2024).

Yunhan Yang, Xiaoyang Wu, Tong He, Hengshuang Zhao, and Xihui Liu. 2023. Sam3d: Segment anything in 3d scenes. arXiv preprint arXiv:2306.03908 (2023).

Li Yi, Vladimir G Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. 2016. A scalable active framework for region annotation in 3d shape collections. ACM Transactions on Graphics (ToG) 35, 6 (2016), 1–12.

Yingda Yin, Yuzheng Liu, Yang Xiao, Daniel Cohen-Or, Jingwei Huang, and Baoquan Chen. 2024. Sai3d: Segment any instance in 3d scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3292–3302.

Jonathan Young. 2019. xatlas. https://github.com/jpcy/xatlas. Accessed: 2025-09-26. Mei-Heng Yueh, Wen-Wei Lin, Chin-Tien Wu, and Shing-Tung Yau. 2019. A novel

stretch energy minimization algorithm for equiareal parameterizations. Journal of Scientific Computing 78 (2019), 1353–1386.

Rhaleb Zayer, Bruno Lévy, and Hans-Peter Seidel. 2007. Linear angle based parameterization. In Fifth Eurographics Symposium on Geometry Processing-SGP 2007. Eurographics Association, 135–141.

Eugene Zhang, Konstantin Mischaikow, and Greg Turk. 2005. Feature-based surface parameterization and texture mapping. ACM Transactions on Graphics (TOG) 24, 1

(2005), 1–27. Qijian Zhang, Junhui Hou, Wenping Wang, and Ying He. 2024. Flatten Anything: Unsupervised Neural Surface Parameterization. In Proc. NeurIPS.

Yuming Zhao, Qijian Zhang, Junhui Hou, Jiazhi Xia, Wenping Wang, and Ying He. 2025. FlexPara: Flexible Neural Surface Parameterization. arXiv preprint arXiv:2504.19210

(2025).

Kun Zhou, John Synder, Baining Guo, and Heung-Yeung Shum. 2004. Iso-charts: stretchdriven mesh parameterization using spectral analysis. In Proceedings of the 2004 Eurographics/ACM SIGGRAPH symposium on Geometry processing. 45–54.

Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. 2018a. Open3D: A modern library for 3D data processing. arXiv preprint arXiv:1801.09847 (2018).

Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. 2018b. Open3D: A Modern Library for 3D Data Processing. arXiv:1801.09847 (2018). 3 Coords Computing Łukasz Czyż. 2025. UVPackmaster 3: GPU Accelerated UV Packing Engine. https://uvpackmaster.com/. Accessed: 2025-05-21.

