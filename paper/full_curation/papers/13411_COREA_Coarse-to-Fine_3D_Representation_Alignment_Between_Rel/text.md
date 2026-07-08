# arXiv:2512.07107v3[cs.CV]17Mar2026

## COREA: Coupled Relightable 3D Gaussians and SDFs for Efficient Normal Alignment

##### Jaeyoon Lee⋆, Hojoon Jung⋆, Sungtae Hwang, Jihyong Oh†, and Jongwon Choi†

Chung-Ang University, Seoul, Korea {leejaeyoon, hjjung, sthwang}@vilab.cau.ac.kr, {jihyongoh, choijw}@cau.ac.kr ⋆Equal contribution. †Corresponding authors.

https://cau-vilab.github.io/COREA/

###### (a)

- (b) Inverse PBR Capable

- (c)

[Figure 1]

[Figure 2]

Inverse PBR & Relighting

Surface Reconstruction

30

[Figure 3]

[Figure 4]

[Figure 5]

PSNR(PBR)↑

G-Shader (CVPR'24)

[Figure 6]

25

[Figure 7]

[Figure 8]

GS-IR (CVPR'24) R3DG (ECCV'24) COREA (Ours)

[Figure 9]

20

Multi-view Images

15

[Figure 10]

dummy

20 24 28 32

[Figure 11]

[Figure 12]

PSNR↑ (SH-based NVS)

COREA

Surface Reconstructable

SH-based NVS

GOF (SIGGRAPH'24)

CD↓

[Figure 13]

[Figure 14]

GSDF (NeurIPS'24)

[Figure 15]

[Figure 16]

[Figure 17]

Neural SDF

GS-Pull (NeurIPS'24)

###### COREA (Ours)

Relightable 3DGS

0.7

dummy

27 29 31 33

PSNR↑ (SH-based NVS)

Fig. 1: Overview of COREA. (a) COREA is the first unified three-tasks framework that couples an SDF and relightable 3D Gaussians on a shared underlying surface for SH-based NVS, surface reconstruction, and inverse PBR. (b) Quantitative comparison among inverse-PBR methods shows that COREA achieves strong performance for both SH-based NVS and inverse PBR. (c) Among surfaces reconstruction methods, COREA delivers competitive geometric accuracy while maintaining high rendering fidelity.

Abstract. We present COREA, the first unified three-tasks framework that couples an SDF and relightable 3D Gaussians (3DGS) to jointly support SH-based novel-view synthesis (NVS), surface reconstruction, and inverse physically-based rendering (inverse PBR). While recent relightable 3DGS methods have progressed, inverse PBR remains bottlenecked by normal estimation, as the discrete nature of 3DGS often yields oversmoothed and unstable normals. To address this limitation, COREA couples the complementary geometric properties of an SDF and relightable 3DGS on a shared underlying surface, where geometry-constrained relightable 3DGS provides reliable depth signals to anchor SDF geometry and the continuous SDF normal field provides spatially consistent supervision for Gaussian normal learning. We couple these signals through depth-guided alignment and normal supervision with normal-aware densification, and introduce Dual-Density Control to regulate densification by balancing photometric and geometric gradients for stable, memory-efficient training. Experiments on standard benchmarks show that COREA is the only framework that supports all three tasks, achieving competitive performance overall, with particularly superior results in inverse PBR.

Keywords: 3D Gaussian Splatting · Signed Distance Field · Inverse Physically-Based Rendering

### 1 Introduction

Recent advancements in 3D Gaussian splatting (3DGS) [20] enable high-quality, real-time rendering with explicit 3D primitives, which are increasingly integrated with meshes for editing and animation, e.g., via mesh-bound Gaussians in tools such as Blender [10,14,15,28,34,36]. For practical deployment under diverse lighting, 3DGS-based frameworks should jointly support SH-based novel-view synthesis (NVS), surface reconstruction, and inverse physically-based rendering (inverse PBR), so that shading variations induced by manipulation-driven geometry changes are faithfully reproduced.

In practice, inverse PBR with 3DGS remains bottlenecked by accurate normal estimation, since surface normals directly govern BRDF-lighting decomposition and relighting quality [44]. Relightable 3DGS pipelines typically rely on rendered normal maps, often obtained via alpha blending [9,12,18,25,33,44,55], but the discrete splat representation oversmooths the rendered normals and suppresses fine-scale surface variations (Fig. 2 (A)). To also support surface reconstruction, many pipelines couple Gaussians with an implicit field, such as neural Signed Distance Fields (SDF), as explicit-implicit hybrid methods [7,27,45,46,51,55]. However, most of these hybrids treat the two branches as separate components, without complementary geometric interaction during joint optimization. Moreover, many hybrid pipelines still supervise the implicit branch using 3DGS-rendered alpha-blended normals, which blurs the implicit branch’s continuous normal field and weakens fine-scale geometric supervision back to the 3DGS branch. In contrast, pixel-wise depth gradients from the 3DGS depth map preserve sharper local structure, providing a stronger signal for aligning SDF normals (Fig. 2 (B)). As a result, existing relightable and hybrid pipelines remain incomplete, and none supports all three tasks within a single framework (Tab. 1).

To address these issues, we propose COREA, the first unified three-tasks framework that jointly learns an SDF and relightable 3D Gaussians, uniquely supporting (i) SH-based NVS, (ii) surface reconstruction, and (iii) inverse PBR. COREA couples an SDF and relightable 3DGS to use complementary geometric properties for stable normal learning on a shared underlying surface. The continuous SDF normal field stabilizes Gaussian normal formation, while geometry-constrained relightable 3DGS depth signals refine the SDF geometry. For SDF learning, geometry-constrained relightable 3DGS provides reliable depth signals for SDF ray sampling, and COREA further refines the SDF by aligning SDF normals with pixel-wise depth gradients from the 3DGS depth map rather than alpha-blended normals. For Gaussian learning, COREA projects Gaussians onto the SDF surface, aligns Gaussian normals with the SDF normal field, and drives normal-aware densification based on normal inconsistency to capture finescale detail. To keep normal-aware densification stable and efficient, we introduce Dual-Density Control (DDC) that balances photometric and geometric gradients,

|[Figure 18]|
|---|

|[Figure 19]<br><br>| |
|---|
|
|---|

|[Figure 20]<br><br>| |
|---|
|
|---|

- Table 1: Task Support Comparison. Comparison of supported tasks: SH-based NVS (SH), Surface Reconstruction (SR), and Inverse PBR (PBR).

(A) Alpha-blended Normal

(B) Pixel-wise Depth Gradient

Category Method SH SR PBR Vanilla 3DGS (SIGGRAPH’23) ✓ ✗ ✗

RGB Image

GaussianShader (CVPR’24) ✓ ✗ ✓ GS-IR (CVPR’24) ✓ ✗ ✓ R3DG (ECCV’24) ✓ ✗ ✓ SVG-IR (CVPR’25) ✓ ✗ ✓

Fig. 2: Comparison of normal supervision signals at an intermediate training stage (35k iterations). (A) Alphablended normals are blurrier and noisier, providing unstable supervision for SDF normal learning. (B) Pixel-wise depth gradients preserve sharper structure, yielding more stable SDF alignment.

Relightable

GOF (SIGGRAPH’24) ✓ ✓ ✗ GSDF (NeurIPS’24) ✓ ✓ ✗ GS-Pull (NeurIPS’24) ✓ ✓ ✗ GS-ROR2 (TOG’25) ✗ ✓ ✓ COREA (Ours) ✓ ✓ ✓

Hybrid

preventing excessive splitting while preserving geometric fidelity and stable rendering. With geometrically aligned Gaussians and stabilized normals, we perform inverse PBR [19] to robustly disentangle BRDF parameters and lighting, enabling faithful relighting under novel lighting.

As shown in Fig. 1 (a), COREA is the only unified three-tasks framework that simultaneously supports SH-based NVS, surface reconstruction, and inverse PBR (Tab. 1) by coupling relightable 3DGS and SDF to use complementary geometric properties on a shared underlying surface. It achieves superior performance in SH-based NVS and inverse PBR (Fig. 1 (b)), while also delivering high-quality reconstructed surfaces (Fig. 1 (c)).

Our key contributions are as follows:

- – We present COREA, the first unified three-tasks framework that jointly learns an SDF and relightable 3DGS, and supports SH-based NVS, surface reconstruction, and inverse PBR.
- – We couple an SDF and relightable 3DGS on a shared underlying surface: geometry-constrained relightable 3DGS provides reliable supervision for SDF geometry learning, and the continuous SDF normal field stabilizes Gaussian normal formation.
- – We propose a dual-density control mechanism that regularizes densification using both photometric and geometric gradients, preventing excessive Gaussian splitting while maintaining geometric fidelity and stable rendering.
- – Experiments show that COREA achieves superior performance in SH-based NVS and inverse PBR, while delivering high-quality reconstructed surfaces on diverse benchmarks.

### 2 Related Work

#### 2.1 Inverse Physically-based Rendering

Inverse PBR aims to recover accurate geometry, material properties, and illumination by modeling light-surface interactions from multi-view observations

and disentangling reflectance from lighting. NeRF [29]-based inverse rendering approaches [4,40,42,43,48,52] jointly estimate geometry, reflectance, and lighting, but typically require expensive volumetric rendering and remain computationally demanding. Several reflectance-aware works [35,53] further improve appearance decomposition by modeling specular reflections and indirect illumination.

Recent studies extend 3DGS to inverse rendering by building on its efficient explicit representation. Methods such as GIR [32], GS-IR [25], and GaussianShader [18] incorporate inverse rendering formulations into Gaussian representations by factorizing geometry, material, and lighting, enforcing normal regularization, and learning shading functions for reflective scenes. R3DG [9] introduces point-based ray tracing into 3DGS to enable finer BRDF and lighting decomposition. IRGS [12] incorporates the full rendering equation into 2D Gaussian Splatting through differentiable Gaussian ray tracing. Ref-Gaussian [50] captures fine inter-reflection effects using a physically based deferred rendering pipeline, while RNG [8] replaces the analytical rendering equation with neural modules that learn Gaussian-light interactions. GeoSplatting [44] and SVG-IR [33] further incorporate geometry-aware optimization and structured parameter disentanglement to model geometry, material, and illumination within Gaussian representations.

However, Gaussian normals rendered via alpha blending are often oversmoothed due to the discrete splat representation, making BRDF-lighting decomposition unreliable for inverse PBR. To address this limitation, we propose a unified framework that couples an SDF and relightable 3DGS, enabling complementary geometric supervision during optimization. This coupling provides stable surface normals and supports faithful BRDF and lighting decomposition under complex illumination.

#### 2.2 Hybrid 3DGS Methods for Surface Reconstruction

Recent explicit-implicit hybrid studies integrate 3DGS with neural implicit representations such as SDF to improve geometric stability and surface reconstruction [7,27,41,45,51]. Most hybrid approaches primarily combine the two representations by assigning separate roles: Gaussians provide strong photometric rendering with efficient explicit primitives, while implicit fields supply a continuous geometric representation for surface reconstruction. GSDF [45] jointly optimizes Gaussian primitives and an SDF representation to stabilize geometry. Several works enforce geometric alignment between Gaussian primitives and implicit fields. Methods such as G2SDF [22], DiGS [2], GS-SDF [26], and MonoGSDF [23] align Gaussian centers with the zero-level set of implicit fields or estimate SDF fields from Gaussian distributions. GS-Pull [51] further refines Gaussian geometry by pulling Gaussian centers toward the SDF zero-level set using SDF gradient guidance. Other approaches improve geometry directly within Gaussian representations. Gaussian Opacity Fields (GOF) [46] introduce opacitydriven geometric modeling that stabilizes geometry extraction. GS-ROR2 [55] incorporates SDF guidance into a relightable 3DGS framework by pruning Gaussians that deviate from the SDF zero-level set. However, this role-separated coupling is not designed to induce complementary geometric interaction during

Depth Alignment

Normal Refinement

Initial

Output

Relighting Stage 2. Inverse PBR (Sec. 3.3)

Stage 1. Geometric Coupling (Sec. 3.1)

[Figure 21]

Relighting

BRDF Parameters

- Step 1. Depth-guided SDF Alignment (DSA)
- Step 2. Normal-guided Gaussian Alignment (NGA)

SDF

|[Figure 22]|
|---|

| |
|---|

|𝐿𝐿𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑<br><br>|
|---|

|𝐿𝐿𝑆𝑆𝑆𝑆𝑆𝑆𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛<br><br>(Eq. (1))|
|---|

𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝐺𝐺𝑁𝑁

[Figure 23]

| | |
|---|---|
| | |

[Figure 24]

𝐴𝐴𝑁𝑁𝐴𝐴𝐷𝐷𝑛𝑛𝑁𝑁 𝑅𝑅𝑁𝑁𝐺𝐺𝑆𝑆𝐷𝐺𝐺𝐷𝐷𝐺𝐺𝐺𝐺

[Figure 25]

[Figure 26]

|[Figure 27]|
|---|

𝑆𝑆𝐺𝐺𝑁𝑁𝐷𝐷𝑆𝑆𝐷𝐷 𝐿𝐿𝐺𝐺𝑆𝑆𝐷𝐷𝐷 𝐼𝐼𝐺𝐺𝑛𝑛𝐺𝐺𝑁𝑁𝐷𝐷𝑆𝑆𝐷𝐷 𝐿𝐿𝐺𝐺𝑆𝑆𝐷𝐷𝐷

[Figure 28]

[Figure 29]

|𝐿𝐿𝐿𝐿3𝑆𝑆𝐷𝐷𝑆𝑆𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑛𝑆𝑆 𝑆𝑆<br><br>(Eq. (2))<br><br>|
|---|

|𝐿𝐿𝐿𝐿𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑𝑑<br><br>|
|---|

| |
|---|

DualDensity Control (Sec. 3.2) (Fig. 4)

[Figure 30]

|[Figure 31]|
|---|

𝑅𝑅𝐺𝐺𝑅𝑅

[Figure 32]

3DGS

3𝑆𝑆 𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺 𝑆𝑆𝑆𝑆𝑆𝑆 𝑆𝑆𝐷𝐷𝑆𝑆𝐷𝐷𝐷 𝑆𝑆𝐺𝐺𝑁𝑁𝑆𝑆𝐺𝐺𝑆𝑆𝐷𝐷 𝑈𝑈𝐺𝐺𝑛𝑛𝐷𝐷𝑁𝑁𝑁𝑁𝑅𝑅𝐺𝐺𝐺𝐺𝑆𝑆 𝑆𝑆𝐺𝐺𝑁𝑁𝑆𝑆𝐺𝐺𝑆𝑆𝐷𝐷 𝑆𝑆𝑆𝑆𝑆𝑆 𝑅𝑅𝐺𝐺𝑅𝑅 𝑆𝑆𝐺𝐺𝑁𝑁𝑆𝑆𝑁𝑁𝐺𝐺𝐺𝐺𝑆𝑆 𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺 𝑆𝑆𝐷𝐷𝑆𝑆𝐷𝐷𝐷 𝐺𝐺𝑁𝑁𝐺𝐺𝑛𝑛𝐺𝐺𝐷𝐷𝐺𝐺𝐷𝐷

𝑆𝑆𝑆𝑆𝑆𝑆 𝐴𝐴𝑁𝑁𝐺𝐺𝑆𝑆𝐺𝐺𝑁𝑁𝐷𝐷𝐺𝐺𝐷𝐷 𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺 𝐴𝐴𝑁𝑁𝐺𝐺𝑆𝑆𝐺𝐺𝑁𝑁𝐷𝐷𝐺𝐺𝐷𝐷 𝑆𝑆𝑆𝑆𝑆𝑆 𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝐺𝐺𝑁𝑁 𝐼𝐼𝐷𝐷𝐷𝐷𝑁𝑁𝐺𝐺𝐷𝐷𝐺𝐺𝑁𝑁𝐺𝐺

𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺𝐺 𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝑁𝐺𝐺𝑁𝑁

- Fig. 3: Overview of the COREA framework. COREA jointly trains an SDF and relightable 3D Gaussians via geometric coupling on a shared underlying surface, combining reliable depth signals from geometry-constrained relightable 3DGS with the continuous SDF normal field. The coupling uses two complementary modules: (i) DSA aligns the SDF to 3DGS via depth-guided ray sampling and matches SDF normals to pixel-wise depth gradients from the 3DGS depth map (Eq. (1)). (ii) NGA guides Gaussians toward the SDF depth surface via depth consistency and aligns Gaussian normals with the continuous SDF normal field, stabilizing normal learning for discrete Gaussians (Eq. (2)). DDC regulates normal-aware densification using photometric and geometric gradients to suppress redundant splitting. In the second stage, Inverse PBR optimizes BRDF and lighting on the learned geometry, enabling accurate BRDF-lighting decomposition and faithful relighting under novel illumination.

joint training. Consequently, the interaction remains weak and indirect, often relying on auxiliary constraints or pruning operations, which limits geometric cooptimization between the two branches. To move beyond simple combination, we couple an SDF with geometry-constrained relightable 3DGS and jointly optimize them through mutual geometric supervision. By combining complementary geometric properties of the SDF normal field and geometry-constrained relightable 3DGS, our strategy enables consistent alignment and stable normals, improving fine-scale detail.

### 3 Proposed Method: COREA

As illustrated in Fig. 3, COREA couples an SDF and relightable 3DGS on a shared underlying surface through the continuous SDF normal field and geometryconstrained depth signals from 3D Gaussians. The continuous SDF normal field stabilizes Gaussian normal formation and supports normal-aware densification to capture fine-scale geometry. Conversely, geometry-constrained relightable 3DGS provides reliable depth signals for SDF alignment, while pixel-wise depth

#### Algorithm 1 Training pipeline of COREA

Input: Multi-view images {Im}Mm=1, cameras {Cm}Mm=1, SfM point cloud P Output: SH-based NVS INVS, Surface M, inverse PBR IPBR & relighting Irelight

- 1: procedure TrainCOREA({Im,Cm}Mm=1,P)
- 2: G ← InitGaussians(P), F ← InitSDF()
- 3: Warm-up: Optimize G w.r.t. Limage for 15k
- 4: Stage 1 (Geometric coupling (Sec. 3.1, Fig. 3)):
- 5: for t = 1 to 30k do
- 6: if t ≤ 10k then
- 7: SDSA = {Ldepth}, SNGA = {Ldepth} ▷ Coarse depth alignment
- 8: else
- 9: SDSA = {Ldepth,LSDFnormal}, SNGA = {Ldepth,L3DGSnormal} ▷ Normal refinement
- 10: end if
- 11: Step 1: DSA(G,F; SDSA) (Suppl. Alg. 2) ▷ freeze G, update F
- 12: Step 2: NGA(G,F; SNGA) (Suppl. Alg. 3) ▷ freeze F, update G
- 13: if t > 10k and tmod 100 = 0 then
- 14: DDC(G) (Sec. 3.2, Fig. 4)
- 15: end if
- 16: end for
- 17: G⋆ ← G, F⋆ ← F
- 18: Outputs: INVS ← RenderSH(G⋆,Cnovel), M ← MarchingCubes(F⋆)
- 19: Stage 2 (Inverse PBR (Sec. 3.3, Fig. 3)): ▷ freeze {µ, Σ,n} of G⋆
- 20: for s = 1 to 10k do
- 21: Update BRDF and lighting by minimizing LPBR (Suppl. B.1, Suppl. B.2)
- 22: end for
- 23: Outputs: IPBR ← RenderPBR(G⋆,C), Irelight ← Relight(G⋆,C, lnewenv )
- 24: end procedure

gradients from the 3DGS depth map refine SDF geometry by guiding SDF normal alignment. A Dual-Density Control mechanism regularizes densification by balancing photometric and geometric gradients, stabilizing geometry learning and suppressing redundant splitting. COREA couples stable geometry learning from coarse depth to fine surface detail on a shared underlying surface and jointly supports SH-based NVS, surface reconstruction, and inverse PBR.

COREA operates in two stages (Alg. 1.) The first stage focuses on geometric coupling between an SDF and relightable 3DGS (Sec. 3.1). This stage alternates between two complementary steps: (i) Depth-guided SDF Alignment (DSA) performs depth-guided ray sampling using the relightable 3DGS depth map and refines the SDF normal field by matching SDF normals to pixel-wise depth gradients from the 3DGS depth map. (ii) Normal-guided Gaussian Alignment (NGA) projects Gaussians onto the SDF surface and then refines Gaussian normals with the SDF normal field to capture fine-scale geometry and drive normal-aware densification. Dual Density Control regularizes normal-aware densification with both photometric and geometric gradients to prevent excessive splitting. After the geometry converges, the second stage performs inverse PBR (Sec. 3.3), optimizing BRDF parameters and lighting on the learned geometry to enable accurate and spatially consistent relighting. In our implementation, we use R3DG [9] for the

- 3D Gaussian representation and NeuS [37] for the SDF representation. Further algorithmic details of DSA and NGA are provided in the supplementary material.

#### 3.1 Geometric Coupling Between SDF and Relightable 3DGS

To jointly refine geometry across an SDF and relightable 3DGS, we enforce cross-representation geometric consistency by coupling the two representations

through the continuous SDF normal field and 3DGS depth signals. This strategy consists of two complementary steps: DSA and NGA.

Depth-guided SDF Alignment (DSA). To achieve accurate and stable geometry learning, we design DSA to align an SDF with relightable 3DGS. Relightable 3DGS renders depth under a depth-distribution uncertainty constraint [9], producing reliable depth signals to anchor the coarse SDF geometry, while pixel-wise depth gradients capture fine local surface variations. Since the SDF gradient inherently represents surface normals, we refine fine-scale geometry by aligning SDF normals with depth-gradient directions derived from the 3DGS depth map.

We begin with coarse alignment by rendering a depth map d3DGS from relightable 3DGS to guide ray sampling of an SDF. Following the ray-based sampling strategy of GSDF [45], we adopt an adaptive sampling interval centered on d3DGS, allowing the SDF to focus on learning geometry near the 3DGSrendered depth. To encourage geometric consistency, we minimize an L1 loss (Ldepth) between the SDF-rendered and 3DGS-rendered depth maps.

For fine-grained alignment, we compute the 2D pixel-wise gradients of d3DGS (Fig. 2 (B)), instead of alpha-blended normals that become oversmoothed when compositing discrete Gaussian primitives (Fig. 2 (A)). We define nDG3DGS = ∇d3DGS/∥∇d3DGS∥. This choice provides a stronger signal for aligning SDF normals, as validated in our ablation study (Tab. 3). Since the SDF implicitly encodes surface normals via its gradient, we minimize a cosine-similarity loss between the normalized SDF gradient nSDF and nDG3DGS:

N

1 N

LSDFnormal =

i=1

1 − n(SDFi) ,nDG3DGS(i) , (1)

where N is the number of valid pixels. The loss LSDFnormal encourages the SDF gradient field to follow the depth-gradient directions of the 3DGS depth map under a local planarity assumption.

This depth-guided SDF learning yields a continuous normal field that later serves as a stable geometric prior for Gaussian normal learning in NGA, strengthening cross-representation geometric coupling beyond using alpha-blended normal signals alone. Consequently, the SDF reconstructs a surface tightly aligned with the 3DGS depth surface, providing a reliable foundation for subsequent relighting and geometry extraction.

Normal-guided Gaussian Alignment (NGA). Accurate BRDF decomposition and relighting rely on reliable fine-scale geometry and surface normals [44]. Since alpha-blended normals from discrete Gaussians are often oversmoothed and unstable in fine regions (Fig. 2 (A)), we introduce NGA to refine relightable 3DGS on the shared underlying surface by leveraging the continuous SDF normal field as a stable reference for normal learning.

We first align relightable 3DGS to the SDF by minimizing the depth consistency loss Ldepth between their rendered depth maps. Fine-scale accuracy is then

Sec. 3.2 Dual-Density Control (DDC)

𝑊𝑊𝐸𝐸𝐸𝐸𝐸𝐸ℎ𝐸𝐸𝐸𝐸𝑡𝑡 𝑆𝑆𝑆𝑆𝑆𝑆

| |
|---|

|[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]|
|---|

| |
|---|

𝑁𝑁𝐸𝐸𝐸𝐸-𝐸𝐸𝑠𝑠𝑠𝑠𝐸𝐸𝐸𝐸𝐸𝐸𝑠𝑠𝑠𝑠𝑠𝑠𝐸𝐸

𝑆𝑆𝑠𝑠𝑠𝑠𝐸𝐸𝐸𝐸𝐸𝐸𝑠𝑠𝑠𝑠𝑠𝑠𝐸𝐸

𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸𝐸 :

𝜆𝜆𝑚𝑚𝑚𝑚𝑚𝑚(𝑆𝑆𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡(𝑚𝑚) ) < 0 𝜆𝜆𝑚𝑚𝑚𝑚𝑚𝑚(𝑆𝑆𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡𝑡(𝑚𝑚) ) > 0

Photometric Gradient

Geometric Gradient

Total Splitting Matrix (Eq. (3))

Densification

- Fig. 4: Dual-Density Control (DDC). Gaussians whose accumulated gradients exceed the densification threshold are marked in red (candidates), while others remain blue. For candidates, DDC combines splitting matrices from photometric and geometric gradients into Stotal (Eq. (3)). A candidate is split into green Gaussians only if λmin(Stotal) < 0, indicating a descent direction that reduces the overall loss.

improved by minimizing a pixel-wise cosine-similarity loss between the 3DGSrendered alpha-blended normal map nAB3DGS and the SDF normal map nSDF, where nAB3DGS serves as a differentiable coupling interface to update Gaussians rather than a stable supervision signal:

N

1 N

L3DGSnormal =

i=1

1 − nAB3DGS(i),n(SDFi) . (2)

This cross-representation alignment distills SDF continuity into the discrete Gaussians, enabling more stable and spatially consistent normal learning.

To further enhance geometric detail, we use the gradient magnitude of L3DGSnormal as an additional criterion for adaptive Gaussian densification. While the original 3DGS performs densification based only on photometric gradients to improve rendering quality, we additionally densify Gaussians that exhibit large gradients of L3DGSnormal. This criterion encourages refinement in regions with large geometric discrepancies, improving both rendering fidelity and fine geometric detail.

#### 3.2 Dual-Density Control (DDC)

The original adaptive density control of 3DGS splits a Gaussian when the accumulated gradient magnitude exceeds a predefined threshold. In NGA, we densify Gaussians according to the pixel-wise normal alignment loss L3DGSnormal, which enhances fine-scale geometry. Here, L3DGSnormal is driven by the continuous SDF normal field, producing an additional geometric gradient that complements the photometric gradients used in standard 3DGS densification. However, since each Gaussian projects onto multiple pixels, the gradients accumulate excessively, often resulting in unnecessary and redundant splits.

To mitigate such redundant splitting, we propose a Dual-Density Control (DDC) strategy, illustrated in Fig. 4. Inspired by SteepGS [38], DDC jointly considers photometric gradients and geometric gradients, induced by L3DGSimage and the SDF-driven normal loss L3DGSnormal, respectively. For each Gaussian i, we compute splitting matrices Simage(i) and Snormal(i) from the two loss terms, respectively. These matrices approximate the local curvature of the loss landscape with respect to the

Gaussian center µ(i), providing both a criterion for whether splitting will reduce the loss and a direction in which to place the split Gaussians. To balance the influence of both loss terms, we then form a weighted sum of the two matrices:

Stotal(i) = (1 − α) · Simage(i) + α · Snormal(i) , (3)

where α ∈ [0,1] controls the contribution of the normal loss to the splitting decision. A Gaussian is split only if the minimum eigenvalue of Stotal(i) is negative, i.e., λmin(Stotal(i) ) < 0, indicating a descending curvature direction that benefits overall loss reduction.

Through DDC, we regulate densification by balancing photometric and geometric gradients, stabilizing convergence between the two representations. This curvature-guided splitting effectively constrains excessive Gaussian growth, maintaining geometric fidelity and stable rendering performance. Additional ablations and analysis of DDC are provided in the Supplementary Material.

- 3.3 Inverse Physically-Based Rendering (Inverse PBR)

COREA provides relightable 3D Gaussians with reliable surface normals via cross-representation geometric coupling, which stabilizes BRDF and lighting decomposition in inverse rendering. With the learned geometry and normals fixed, we optimize BRDF parameters and lighting in a physically consistent manner.

We supervise per-Gaussian BRDF parameters and environment lighting with an inverse rendering objective, following the formulation of R3DG [9]. To model visibility and shading, we employ the point-based BVH ray tracing scheme from R3DG [9], which enables transmittance-aware visibility estimation for Gaussian primitives. We render images by evaluating BRDF responses under the estimated visibility and incident illumination, and supervise them using an L1 loss against the ground-truth image. Reliable Gaussian normals obtained from the geometric coupling stage improve light-surface interaction modeling, leading to physically faithful rendering and improved relighting quality (Tab. 3, Fig. 7)

After optimizing BRDF and lighting, we relight by replacing the global environment map with a novel lighting condition while omitting precomputed indirect illumination, which enables consistent re-evaluation under the new illumination (details in the supplementary material). This stage completes COREA’s unified pipeline, enabling physically-based relighting with high fidelity and consistency.

- 4 Experiments

- 4.1 Experimental Setup

We evaluate our method on two datasets and three tasks to enable unified assessment of SH-based NVS, surface reconstruction, and inverse PBR. All experiments are conducted on a single NVIDIA A6000 GPU (48 GB) under the same hardware setting for fair comparison.

- Table 2: Quantitative Comparison. We evaluate SH-based NVS, surface reconstruction, and inverse PBR on the DTU and Tanks&Temples datasets. All methods are evaluated under the same single-GPU setting for a fair comparison. N/S denotes methods that do not support the corresponding task, and OOM indicates failures under the same GPU memory budget. COREA achieves strong performance in SH-based NVS and inverse PBR with competitive surface reconstruction (CD), while uniquely supporting all three tasks within a unified framework. The best, second-best, and third-best results are highlighted in red, yellow, and purple, respectively.

Dataset DTU (SH) Tanks&Temples (SH) DTU (PBR) Tanks&Temples (PBR) Category Method CD↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ Vanilla 3DGS (SIGGRAPH’23) N/S 27.30 0.867 0.184 27.46 0.910 0.115 N/S N/S N/S N/S N/S N/S

Relitable GaussianShader (CVPR’24) N/S 22.14 0.826 0.201 19.96 0.851 0.139 19.49 0.719 0.393 17.82 0.829 0.169 GS-IR (CVPR’24) N/S 28.29 0.874 0.173 28.93 0.929 0.089 24.56 0.752 0.206 22.96 0.652 0.186 R3DG (ECCV’24) N/S 31.82 0.944 0.105 29.92 0.941 0.079 26.44 0.914 0.138 26.31 0.912 0.103 SVG-IR (CVPR’25) N/S 28.02 0.923 0.055 27.68 0.934 0.045 OOM OOM OOM OOM OOM OOM

Hybrid GOF (SIGGRAPH’24) 0.802 28.70 0.909 0.124 OOM OOM OOM N/S N/S N/S N/S N/S N/S GSDF (NeurIPS’24) 0.859 27.76 0.874 0.176 OOM OOM OOM N/S N/S N/S N/S N/S N/S GS-Pull (NeurIPS’24) 0.916 27.36 0.872 0.208 26.29 0.893 0.142 N/S N/S N/S N/S N/S N/S

GS-ROR (TOG’25) 2.552 N/S N/S N/S N/S N/S N/S 20.38 0.850 0.206 22.44 0.887 0.145 COREA (Ours) 0.824 32.27 0.955 0.094 29.25 0.937 0.089 29.72 0.942 0.104 27.57 0.927 0.102

Datasets. The DTU dataset [17] provides 15 real-world object scans under controlled lighting and geometry conditions. For Tanks&Temples [21], we follow the R3DG [9] setup and use the four object-centric scenes defined in their work, where evaluation focuses on foreground objects. This setup enables consistent evaluation of geometry and appearance under real-world conditions.

Evaluation Metrics. For SH-based NVS, we compare all methods using PSNR, SSIM [39], and LPIPS [49]. For surface reconstruction, we evaluate methods that output meshes and measure geometric accuracy using Chamfer Distance (CD). Inverse PBR evaluation is conducted only for models capable of BRDF-based rendering, using the same metrics as SH-based NVS.

#### 4.2 Evaluation Results

We render qualitative results on a white background for consistent visualization across tasks, emphasizing overall rendering quality and clear separation between objects and the background. Black speckles or dark edge artifacts in some methods indicate opacity noise often caused by excessive Gaussian density. Methods that do not support specific tasks are marked as N/S, and those that exceed the available GPU memory during training under the same single-GPU setting are marked as OOM. COREA unifies an SDF and relightable 3DGS through geometric coupling on a shared underlying surface, and is the only framework that supports SH-based NVS, surface reconstruction, and inverse PBR within a single pipeline. COREA achieves competitive SH-based NVS and surface reconstruction performance, while delivering particularly strong inverse PBR results. Additional per-scene results are provided in the Supplementary Material and demo video.

SH-based Novel-View Synthesis Comparison. We evaluate SH-based NVS performance on the DTU [17] and Tanks&Temples [21] datasets. As summarized in Tab. 2 (SH), Our method achieves strong performance on DTU across PSNR,

GaussianShader

GS-IR

R3DG

3DGS

GS-Pull

GS-ROR

GOF

COREA

GSDF

SVG-IR

(CVPR’24)

(CVPR’24)

(ECCV’24)

(SIGGRAPH’23)

(NeurIPS’24)

(TOG’25)

(SIGGRAPH’24)

(Ours)

(CVPR’25)

(NeurIPS’24)

|[Figure 37]<br><br>N/S|
|---|

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

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 56]<br><br>N/S|
|---|

||[Figure 57]<br><br>N/S|
|---|
|
|---|

||[Figure 58]<br><br>N/S|
|---|
|
|---|

|[Figure 59]<br><br>N/S|
|---|

||[Figure 60]<br><br>N/S|
|---|
|
|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 71]<br><br>N/S|
|---|

||[Figure 72]<br><br>OOM|
|---|
|
|---|

||[Figure 73]<br><br>N/S|
|---|
|
|---|

|[Figure 74]<br><br>N/S|
|---|

||[Figure 75]<br><br>N/S|
|---|
|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

|[Figure 86]<br><br>OOM|
|---|

|[Figure 87]<br><br>OOM|
|---|

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

N/S

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 103]<br><br>N/S|
|---|

|[Figure 104]<br><br>N/S|
|---|

|[Figure 105]<br><br>N/S|
|---|

|[Figure 106]<br><br>N/S|
|---|

|[Figure 107]<br><br>N/S|
|---|

|[Figure 108]<br><br>OOM|
|---|

|[Figure 109]<br><br>OOM|
|---|

[Figure 110]

[Figure 111]

[Figure 112]

| |
|---|

| |
|---|

| |
|---|

[Figure 113]

[Figure 114]

[Figure 115]

|[Figure 116]<br><br>N/S|
|---|

|[Figure 117]<br><br>OOM|
|---|

|[Figure 118]<br><br>N/S|
|---|

|[Figure 119]<br><br>N/S|
|---|

|[Figure 120]<br><br>N/S|
|---|

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

| |
|---|

| |
|---|

| |
|---|

: SH-based NVS

: Surface Reconstruction

: Inverse PBR

- Fig. 5: Qualitative Comparison. We compare COREA with recent Gaussian-based methods on three tasks: SH-based NVS (blue), surface reconstruction (orange), and inverse PBR (green). All results are rendered on a white background for consistent visual comparison. Artifacts such as black background patches or dark speckles observed in some baselines stem from excessive Gaussian opacity; by contrast, COREA produces clean, artifact-free renderings on white backgrounds. Our method is the only framework that supports all three tasks under the same GPU setting, whereas others show N/S (Not Supported) or OOM (Out of Memory). Overall, COREA yields sharper novel views, more faithful BRDF and lighting decomposition, and finer geometric details through complementary cross-representation geometric coupling. Additional qualitative results are provided in the Supplementary Material and demo video.

SSIM, and LPIPS, demonstrating robust rendering on real-world captures. On Tanks&Temples, the performance is slightly below the top baseline, likely due to the larger scene scale and sparser camera coverage, which makes joint optimization and cross-representation alignment more challenging. Nevertheless, our framework achieves strong inverse PBR performance (Sec. 4.2) and competitive surface reconstruction quality (Sec. 4.2) on this dataset, while being the only method that supports all three tasks within a unified framework.

As shown in Fig. 5 (blue boxes), our method produces sharper and more coherent novel views with fewer background artifacts than other approaches. These results reflect stable geometric coupling and precise cross-representation alignment between the SDF and relightable 3DGS, which are essential for stable relighting under environment-map replacement.

Surface Reconstruction Comparison. We further evaluate surface reconstruction to assess the complementary geometric interaction enabled by coupling an SDF and geometry-constrained relightable 3DGS, and compare COREA with explicit-implicit hybrid baselines [45,46,51,55] on DTU using both quantitative and qualitative results. As summarized in Tab. 2 (CD), COREA demonstrates competitive CD performance compared to existing explicit-implicit hybrid baselines, indicating reliable surface reconstruction. Qualitative results in Fig. 5

[Figure 131]

[Figure 132]

GS-ROR (TOG’25)

COREA (Ours)

COREA (Ours)

GaussianShader (CVPR’24)

GS-IR (CVPR’24)

R3DG (ECCV’24)

GaussianShader (CVPR’24)

GS-IR (CVPR’24)

R3DG (ECCV’24)

Lighting Conditions

GS-ROR (TOG’25)

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

|[Figure 155]|
|---|

|[Figure 156]|
|---|

|[Figure 157]|
|---|

|[Figure 158]|
|---|

|[Figure 159]|
|---|

|[Figure 160]|
|---|

|[Figure 161]|
|---|

|[Figure 162]|
|---|

[Figure 163]

[Figure 164]

[Figure 165]

|[Figure 166]<br><br>[Figure 167]|
|---|

|[Figure 168]|
|---|

|[Figure 169]|
|---|

|[Figure 170]|
|---|

|[Figure 171]|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

|[Figure 174]<br><br>[Figure 175]|
|---|

[Figure 176]

[Figure 177]

|[Figure 178]<br><br>[Figure 179]|
|---|

|[Figure 180]<br><br>[Figure 181]|
|---|

|[Figure 182]|
|---|

|[Figure 183]|
|---|

|[Figure 184]|
|---|

|[Figure 185]|
|---|

|[Figure 186]|
|---|

|[Figure 187]|
|---|

[Figure 188]

[Figure 189]

[Figure 190]

- Fig. 6: Qualitative Results of Inverse PBR and Relighting. We compare COREA with recent relightable Gaussian-based methods under varying illumination conditions. The first row shows PBR renderings under the original lighting setup, while the remaining rows illustrate relighting results under directional lights and various HDR environment maps. Compared to previous methods that exhibit black background artifacts or unstable reflectance under varying illumination, COREA maintains consistent shading, clean appearance, and accurate reflectance reconstruction across all lighting conditions, demonstrating robust BRDF and geometry alignment.

(orange boxes) further demonstrate that COREA reconstructs sharper and more detailed surfaces, preserving fine-scale geometric structures that are often oversmoothed by baselines.

This improvement stems from our geometric coupling between an SDF and geometry-constrained relightable 3DGS on a shared underlying surface, where pixel-wise depth gradients from the 3DGS depth map provide a strong signal for refining the SDF geometry. The resulting surfaces also provide reliable normals for inverse PBR optimization.

Inverse Physically-Based Rendering Comparison. We evaluate our framework for inverse PBR against prior relightable Gaussian approaches [9,18,25]. As shown in Tab. 2 (PBR), our method achieves the highest PBR quality among relightable baselines, indicating that reliable geometry and normals are critical for accurate shading and reflectance recovery.

For qualitative evaluation, Fig. 5 (green boxes) presents PBR renderings under the original illumination. Our method reproduces fine-scale surface details more faithfully, as seen in the normal patches at the bottom right, where surface orientations are accurately captured and reflected in the rendered appearance.

Extended results are provided in Fig. 6. The first row shows the same PBR reconstruction, and the subsequent rows show relighting under novel illumination with changed light directions and environment maps. COREA maintains consistent shading and reflectance across lighting changes, producing clean foreground silhouettes with reduced opacity noise and fewer boundary artifacts. This robustness stems from our geometric coupling between an SDF and relightable 3DGS on a shared underlying surface, which stabilizes normal learning and supports reliable relighting under viewpoint changes.

|[Figure 191]<br><br>| |
|---|
<br><br>|
|---|

|[Figure 192]<br><br>| |
|---|
<br><br>|
|---|

|[Figure 193]|
|---|

|[Figure 194]|
|---|

- Table 3: Quantitative Comparison of Normal Supervision Strategies. We compare alpha-blended normal supervision and our pixel-wise depth gradient supervision for SDF alignment. Pixel-wise depth gradients provide more accurate geometric supervision, enabling more stable convergence of the SDF alignment and improving surface reconstruction (CD), SH-based NVS and inverse PBR performance.

|[Figure 195]<br><br>| |
|---|
<br><br>|
|---|

|[Figure 196]<br><br>| |
|---|
|
|---|

|[Figure 197]|
|---|

|[Figure 198]|
|---|

Alpha-blended Normal

Pixel-wise Depth Gradient (Ours)

Fig. 7: Qualitative Comparison of Normal Supervision Strategies. Upper row: inverse PBR renderings. Lower row: reconstructed surfaces. Pixel-wise depth gradient supervision yields more faithful relighting and more accurate surface reconstruction.

Mesh SH-based NVS Inverse PBR Method CD↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Alpha-blended normal 0.848 31.89 0.947 0.096 29.38 0.933 0.116 Pixel-wise depth gradient (Ours) 0.824 32.27 0.955 0.094 29.72 0.942 0.104

#### 4.3 Ablation Study

Depth-Gradient Supervision for Geometry Alignment. We conduct an ablation study to validate pixel-wise depth gradient supervision for geometry alignment in our coupled SDF and relightable 3DGS framework. Fig. 7 qualitatively compares alpha-blended normal supervision and depth-gradient supervision. The upper row shows inverse-PBR renderings with normal patches, while the lower row shows reconstructed surfaces with zoomed-in regions. With alphablended normal supervision, the normal patches exhibit weaker shading contrast and the reconstructed surfaces appear inflated with missing fine-scale structure. This degradation arises because the discrete Gaussians oversmooth rendered normals through alpha blending, weakening SDF supervision and allowing errors to propagate back to 3DGS through the coupled optimization. In contrast, pixelwise depth gradient supervision preserves sharper local structure and yields more refined surface geometry and normals.

Tab. 3 provides quantitative results across SH-based NVS, surface reconstruction, and inverse PBR. Compared to alpha-blended normal supervision, pixel-wise depth gradient supervision consistently improves rendering quality and geometric accuracy, achieving lower Chamfer Distance (CD) and higher PSNR and SSIM with lower LPIPS for SH-based NVS and inverse PBR. Overall, depth-gradient supervision provides a stronger geometric signal than alpha-blended normals, leading to more reliable geometric coupling and improved fidelity across tasks.

Geometric Coupling for Reliable Alignment. We analyze the role of each alignment module by disabling it while keeping the rest of the pipeline unchanged; quantitative results are summarized in Tab. 4.

Module ablations. Each ablation breaks one direction of our geometric coupling loop. In w/o DSA, the SDF no longer receives depth-based guidance from geometry-constrained relightable 3DGS (depth alignment and depth-gradient normal refinement), leading to a clear drop in surface reconstruction quality, particularly in CD. In w/o NGA, relightable 3DGS no longer benefits from the

- Table 4: Ablation of Alignment Modules and Update Strategies. w/o DSA removes depth-based guidance from geometry-constrained relightable 3DGS for SDF alignment and depth-gradient refinement, leading to less precise SDF geometry and a clear degradation in surface reconstruction (CD). w/o NGA removes SDF-guided depth alignment and continuity-based normal refinement for relightable 3DGS, yielding oversmoothed Gaussian normals and the largest drop in inverse PBR. With both modules enabled, Full-Alt. achieves the best overall performance by reducing interference through alternating updates, outperforming Full-Simul. on SH-based NVS, surface reconstruction, and inverse PBR.

Mesh NVS PBR Setting CD↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

w/o NGA 0.863 31.47 0.944 0.104 28.63 0.926 0.124 w/o DSA 0.932 31.50 0.945 0.097 28.23 0.908 0.132 Full-Simul. 0.852 31.62 0.946 0.105 29.07 0.931 0.123 Full-Alt. (COREA) 0.824 32.27 0.955 0.094 29.72 0.942 0.104

continuous SDF normal field, which prevents stable Gaussian normal formation and weakens normal-aware densification, causing the largest degradation in inverse PBR where sharp and consistent normals are essential.

Together, these results confirm that COREA derives its strength from coupling complementary geometric properties on a shared underlying surface: DSA anchors and refines the SDF using geometry-constrained depth signals from relightable 3DGS, while NGA stabilizes Gaussian normals using the continuous SDF normal field, yielding reliable geometry and improved inverse PBR quality.

Update strategy. We also compare two update strategies for the full pipeline (Alg. 1): a simultaneous scheme that updates DSA and NGA in the same step (Full-Simul.) and our alternative scheme that updates them in alternating steps (Full-Alt.). As shown in Tab. 4, the alternative strategy performs best overall, as alternating updates reduce interference and yield more stable optimization by updating one representation while keeping the other fixed. We adopt the alternative strategy by default.

- 5 Conclusion

We presented COREA, the first unified three-tasks framework that jointly learns an SDF and relightable 3D Gaussians by coupling their complementary geometric properties on a shared underlying surface. By anchoring SDF geometry with depth signals and pixel-wise depth gradients from geometry-constrained relightable 3DGS and stabilizing Gaussian normal learning using the continuous SDF normal field, COREA preserves fine-scale structure and provides reliable normals. We further introduce Dual-Density Control, which regularizes normalaware densification using both photometric and geometric gradients to prevent excessive Gaussian growth while maintaining stable rendering. These components provide a consistent geometric foundation for inverse PBR, enabling accurate BRDF and lighting decomposition and faithful relighting under novel illumination. Experiments show that COREA achieves strong results in SH-based NVS and inverse PBR while delivering competitive surface reconstruction, and remains the only framework that supports all three tasks.

### References

- 1. Adobe: Adobe after effects. https://www.adobe.com/products/aftereffects. html (2024)
- 2. Ben-Shabat, Y., Hewa Koneputugodage, C., Gould, S.: Digs: Divergence guided shape implicit neural representation for unoriented point clouds. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19323–19332 (2022)
- 3. Blender Foundation: Blender. https://www.blender.org/ (2024)
- 4. Boss, M., Braun, R., Jampani, V., Barron, J.T., Liu, C., Lensch, H.: Nerd: Neural reflectance decomposition from image collections. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12684–12694 (2021)
- 5. Bruneau, R., Brument, B., Quéau, Y., Mélou, J., Lauze, F.B., Durou, J.D., Calvet, L.: Multi-view surface reconstruction using normal and reflectance cues. International Journal of Computer Vision 134(2), 69 (2026)
- 6. Chen, D., Li, H., Ye, W., Wang, Y., Xie, W., Zhai, S., Wang, N., Liu, H., Bao, H., Zhang, G.: Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics 31(9), 6100–6111 (2024)
- 7. Chen, H., Li, C., Lee, G.H.: Neusg: Neural implicit surface reconstruction with 3d gaussian splatting guidance (2023)
- 8. Fan, J., Luan, F., Yang, J., Hašan, M., Wang, B.: Rng: Relightable neural gaussians

(2024)

- 9. Gao, J., Gu, C., Lin, Y., Li, Z., Zhu, H., Cao, X., Zhang, L., Yao, Y.: Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing. In: European Conference on Computer Vision. pp. 73–89. Springer (2024)
- 10. Gao, X., Li, X., Zhuang, Y., Zhang, Q., Hu, W., Zhang, C., Yao, Y., Shan, Y., Quan, L.: Mani-gs: Gaussian splatting manipulation with triangular mesh (2024)
- 11. Gropp, A., Yariv, L., Haim, N., Atzmon, M., Lipman, Y.: Implicit geometric regularization for learning shapes (2020)
- 12. Gu, C., Wei, X., Zeng, Z., Yao, Y., Zhang, L.: Irgs: Inter-reflective gaussian splatting with 2d gaussian ray tracing (2025), https://arxiv.org/abs/2412.15867
- 13. Guédon, A., Gomez, D., Maruani, N., Gong, B., Drettakis, G., Ovsjanikov, M.: Milo: Mesh-in-the-loop gaussian splatting for detailed and efficient surface reconstruction. ACM Transactions on Graphics (TOG) 44(6), 1–15 (2025)
- 14. Guédon, A., Lepetit, V.: Gaussian frosting: Editable complex radiance fields with real-time rendering. In: European Conference on Computer Vision. pp. 413–430

(2024)

- 15. Guédon, A., Lepetit, V.: Sugar: Surface-aligned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 5354–5363

(2024)

- 16. Huang, B., Yu, Z., Chen, A., Geiger, A., Gao, S.: 2d gaussian splatting for geometrically accurate radiance fields. In: ACM SIGGRAPH 2024 Conference Papers. pp. 1–11 (July 2024)
- 17. Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., Aanæs, H.: Large scale multi-view stereopsis evaluation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2014)
- 18. Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., Ma, Y.: Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces. In: Proceedings of

- the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5322–5332 (2024)
- 19. Kajiya, J.T.: The rendering equation. In: Proceedings of the 13th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH). pp. 143–150. ACM (August 1986)
- 20. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42(4), 139:1–139:14 (2023)
- 21. Knapitsch, A., Park, J., Zhou, Q.Y., Koltun, V.: Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (TOG) 36(4), 78:1–78:13 (2017)
- 22. Li, K., Niemeyer, M., Chen, Z., Navab, N., Tombari, F.: G2sdf: Surface reconstruction from explicit gaussians with implicit sdfs (2024), https://arxiv.org/abs/ 2411.16898
- 23. Li, K., Niemeyer, M., Chen, Z., Navab, N., Tombari, F.: Monogsdf: Exploring monocular geometric cues for gaussian splatting-guided implicit surface reconstruction

(2025), https://arxiv.org/abs/2411.16898

- 24. Li, Z., Müller, T., Evans, A., Taylor, R.H., Unberath, M., Liu, M.Y., Lin, C.H.: Neuralangelo: High-fidelity neural surface reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8456–8465

(2023)

- 25. Liang, Z., Zhang, Q., Feng, Y., Shan, Y., Jia, K.: Gs-ir: 3d gaussian splatting for inverse rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21644–21653 (2024)
- 26. Liu, J., Wan, Y., Wang, B., Zheng, C., Lin, J., Zhang, F.: Gs-sdf: Lidar-augmented gaussian splatting and neural sdf for geometrically consistent rendering and reconstruction. arXiv preprint arXiv:2108.10470 (2025)
- 27. Lyu, X., Sun, Y.T., Huang, Y.H., Wu, X., Yang, Z., Chen, Y., Pang, J., Qi, X.: 3dgsr: Implicit surface reconstruction with 3d gaussian splatting. ACM Transactions on Graphics (TOG) 43(6), 1–12 (2024)
- 28. Ma, S., Luo, Y., Yang, W., Yang, Y.: Mags: Reconstructing and simulating dynamic 3d objects with mesh-adsorbed gaussian splatting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 8745–8755 (2025)
- 29. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: European Conference on Computer Vision (2020)
- 30. PlayCanvas: Supersplat: Web-based 3d gaussian splat editor. https://github.com/ playcanvas/supersplat (2025)
- 31. Rosu, R.A., Behnke, S.: Permutosdf: Fast multi-view reconstruction with implicit surfaces using permutohedral lattices. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8466–8475 (2023)
- 32. Shi, Y., Wu, Y., Wu, C., Liu, X., Zhao, C., Feng, H., Zhang, J., Zhou, B., Ding, E., Wang, J.: Gir: 3d gaussian inverse rendering for relightable scene factorization

(2023)

- 33. Sun, H., Gao, Y., Xie, J., Yang, J., Wang, B.: Svg-ir: Spatially-varying gaussian splatting for inverse rendering (2025), https://arxiv.org/abs/2504.06815
- 34. Tobiasz, R., Wilczyński, G., Mazur, M., Tadeja, S., Spurek, P.: Meshsplats: mesh-based rendering with gaussian splatting initialization. arXiv preprint arXiv:2502.07754 (2025)
- 35. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J.T., Srinivasan, P.P.: Ref-nerf: Structured view-dependent appearance for neural radiance fields. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024)

- 36. Waczyńska, J., Borycki, P., Tadeja, S., Tabor, J., Spurek, P.: Games: Mesh-based adapting and modification of gaussian splatting. arXiv preprint arXiv:2402.01459

(2024)

- 37. Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., Wang, W.: Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction (2021)
- 38. Wang, P., Wang, Y., Wang, D., Mohan, S., Fan, Z., Wu, L., others, Ranjan, R.: Steepest descent density control for compact 3d gaussian splatting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 26663–26672 (2025)
- 39. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing 13(4), 600–612 (2004)
- 40. Wu, H., Hu, Z., Li, L., Zhang, Y., Fan, C., Yu, X.: Nefii: Inverse rendering for reflectance decomposition with near-field indirect illumination. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4295–4304 (2023)
- 41. Xiang, H., Li, X., Cheng, K., Lai, X., Zhang, W., Liao, Z., Zeng, L., Liu, X.: Gaussianroom: Improving 3d gaussian splatting with sdf guidance and monocular cues for indoor scene reconstruction (2025), https://arxiv.org/abs/2405.19671
- 42. Yang, J., Xiao, H., Teng, W., Cai, Y., Zhao, Y.: Light sampling field and brdf representation for physically-based neural rendering (2023)
- 43. Yao, Y., Zhang, J., Liu, J., Qu, Y., Fang, T., McKinnon, D., Tsin, Y., Quan, L.: Neilf: Neural incident light field for physically-based material estimation. In: European Conference on Computer Vision. pp. 700–716. Springer (2022)
- 44. Ye, K., Gao, C., Li, G., Chen, W., Chen, B.: Geosplatting: Towards geometry-guided gaussian splatting for physically-based inverse rendering (2024)
- 45. Yu, M., Lu, T., Xu, L., Jiang, L., Xiangli, Y., Dai, B.: Gsdf: 3dgs meets sdf for improved rendering and reconstruction (2024)
- 46. Yu, Z., Sattler, T., Geiger, A.: Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics (TOG) 43(6), 1–13 (2024)
- 47. Zhang, B., Fang, C., Shrestha, R., Liang, Y., Long, X., Tan, P.: Rade-gs: Rasterizing depth in gaussian splatting. arXiv preprint arXiv:2406.01467 (2024)
- 48. Zhang, J., Yao, Y., Li, S., Liu, J., Fang, T., McKinnon, D., Tsin, Y., Quan, L.: Neilf++: Inter-reflectable light fields for geometry and material estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3601–3610 (2023)
- 49. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable effectiveness of deep features as a perceptual metric (2018), https://arxiv.org/ abs/1801.03924
- 50. Zhang, R., Luo, T., Yang, W., Fei, B., Xu, J., Zhou, Q., Liu, K., He, Y.: Refgaussian: Disentangling reflections from 3d gaussian splatting for realistic rendering (2024), https://arxiv.org/abs/2406.05852
- 51. Zhang, W., Liu, Y.S., Han, Z.: Neural signed distance function inference through splatting 3d gaussians pulled on zero-level set (2024)
- 52. Zhang, X., Srinivasan, P.P., Deng, B., Debevec, P., Freeman, W.T., Barron, J.T.: Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. ACM Transactions on Graphics (ToG) 40(6), 1–18 (2021)
- 53. Zhang, Y., Sun, J., He, X., Fu, H., Jia, R., Zhou, X.: Modeling indirect illumination for inverse rendering. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18643–18652 (2022)

- 54. Zhang, Z., Hu, W., Lao, Y., He, T., Zhao, H.: Pixel-gs: Density control with pixelaware gradient for 3d gaussian splatting. In: European Conference on Computer Vision. pp. 326–342. Springer (2024)
- 55. Zhu, Z.L., Wang, B., Yang, J.: Gs-ror2: Bidirectional-guided 3dgs and sdf for reflective object relighting and reconstruction (2025), https://arxiv.org/abs/ 2406.18544

Supplementary Material

- A Demo Video

To present the proposed method more effectively, we provide a demo video. The video is organized into several segments, including an overall teaser, a brief visualization of our framework, and additional visual materials demonstrating the qualitative performance of COREA. It also presents comparisons across three key tasks: SH-based NVS, surface reconstruction, and inverse PBR. The final demo is edited using Adobe After Effects [1]. Source clips for SH-based NVS and inverse PBR are generated through the web-based renderer Supersplat [30], and clips for surface reconstruction are rendered using Blender [3].

- B Implementation details

- B.1 Inverse Rendering Background

Inverse rendering with 3DGS. Following the inverse rendering formulation of R3DG [9], each Gaussian primitive is augmented with BRDF parameters, including albedo b and roughness r, together with a learned indirect illumination term l and a shared global environment map lenv. The diffuse component fd is determined by b, while the specular component fs depends on the roughness r as well as the incoming and outgoing directions ωi and ωo, together defining the BRDF response of each Gaussian.

Incident radiance and physically-based color. Given a visibility term V (ωi) along direction ωi, the incident radiance is modeled as the sum of the visible environment lighting and a learned indirect illumination term:

L(ωi) = V (ωi)lenv(ωi) + l(ωi), (S1)

Ns

c′(ωo) =

[fd + fs(ωo,ωi)]L(ωi)(ωi · n)∆ωi, (S2)

i=1

where Li(ωi) denotes the incident radiance composed of the environment light lenv and the learned indirect term li, n is the Gaussian normal, c′(ωo) is the outgoing radiance toward ωo, Ns is the number of sampled incident directions, and ∆ωi is the solid-angle weight of the i-th sample. The physically-based color c′(ωo) is rendered by evaluating Eq. (S2) under the incident radiance defined in Eq. (S1). For relighting under a novel illumination, we replace the global environment map lenv with lnewenv and re-evaluate Eq. (S2) using the updated incident radiance, while omitting the learned indirect illumination term l(ωi) for consistent rendering under the new lighting.

This formulation enables spatially varying relighting while remaining compatible with the standard SH-based rendering pipeline of 3DGS. We estimate visibility V (ωi) following the BVH-based ray tracing scheme of R3DG [9], which models Gaussian transmittance along sampled incident rays.

#### B.2 Training Objectives

We adopt a two-stage training scheme: geometric coupling in the first stage, followed by inverse PBR in the second stage.

In the geometric coupling stage, an SDF and a relightable 3DGS are jointly optimized through depth and normal alignment with alternating updates for stable training. Accordingly, the total geometry loss is:

LGeometry = LSDF + L3DGS. (S3)

We first define a cross-representation depth consistency loss, which align the depth maps rendered from the SDF and the geometry-constrained relightable 3DGS using an L1 loss. This coarse alignment encourages the two representations to converge to a shared geometric structure:

N

1 N

Ldepth =

i=1

d(SDFi) − d(3DGSi) (S4)

where dSDF and d3DGS denote the rendered depth maps of the SDF and relightable 3DGS, respectively, and N is the number of valid pixels. The loss function for the SDF network is:

LSDF = λSDF1 LSDF1 + λdepthLdepth + λeikLeik+ λcurvLcurv + λnormalLSDFnormal (S5)

where LSDF1 is a reconstruction term used for coarse alignment, Leik enforces the Eikonal constraint [11], Lcurv encourages geometric smoothness [24,31], and LSDFnormal (Eq. (1)) aligns the SDF gradient with the pixel-wise depth gradient from 3DGS, as described in DSA (Sec. 3.1). The loss function for the 3DGS network is defined as:

L3DGS = L3DGSimage + λdepthLdepth + λOLO + λuLu + λnormalL3DGSnormal (S6)

Here, LO and Lu are the mask and depth-distribution constraints adopted from R3DG [9]. In particular, Lu regularizes the rendered depth distribution by minimizing its uncertainty, e.g., Lu = E[d2] − (E[d])2 with E[d] = i∈K widi and E[d2] = i∈K wid2i, where wi denotes the normalized rendering weight defined in R3DG [9]. These constraints suppress noise in the rendered depth map d3DGS, and consequently stabilize the pixel-wise depth gradient nDG3DGS = ∇d3DGS/∥∇d3DGS∥. As a result, d3DGS and nDG3DGS provide sharper and more reliable supervision signals than alpha-blended normals (Fig. 2), making them more stable supervision signals for DSA. We define the photometric objective used in densification as

L3DGSimage = λ3DGS1 L3DGS1 + λssimLssim, (S7) whose gradients correspond to the photometric gradients used in DDC. Similarly, L3DGSnormal (Eq. (2)) is the cosine loss used in NGA to align 3DGS normals with

#### Algorithm 2 Step 1: Depth-guided SDF Alignment (DSA)

- 1: procedure DSA(G, F; SDSA)
- 2: Freeze G
- 3: d3DGS ← RenderDepth(G)
- 4: (dSDF, nSDF) ← RenderSDF(F)
- 5: if Ldepth ∈ SDSA then
- 6: Compute Ldepth(dSDF, d3DGS) ▷ Suppl. Sec. B.2, Eq. (S4)
- 7: end if
- 8: if LSDFnormal ∈ SDSA then
- 9: n3DGS ← Normalize(∇d3DGS) ▷ Main. Fig. 2 (B)
- 10: Compute LSDFnormal(nSDF, n3DGS) ▷ Main. Sec. 3.1, Eq. (1)
- 11: end if
- 12: Update F ← arg minF LSDF ▷ Suppl. Sec. B.2, Eq. (S5)
- 13: end procedure

#### Algorithm 3 Step 2: Normal-guided Gaussian Alignment (NGA)

- 1: procedure NGA(G, F; SNGA)
- 2: Freeze F
- 3: d3DGS ← RenderDepth(G)
- 4: (dSDF, nSDF) ← RenderSDF(F)
- 5: if Ldepth ∈ SNGA then
- 6: Compute Ldepth(d3DGS, dSDF) ▷ Suppl. Sec. B.2, Eq. (S4)
- 7: end if
- 8: if L3DGSnormal ∈ SNGA then
- 9: nAB3DGS ← AlphaBlendedNormal(G) ▷ Main. Fig. 2 (A)
- 10: Compute L3DGSnormal(nAB3DGS, nSDF) ▷ Main. Sec. 3.1, Eq. (2)
- 11: end if
- 12: Update G ← arg minG L3DGS ▷ Suppl. Sec. B.2, Eq. (S6)
- 13: end procedure

SDF normals, and its gradients constitute the geometric gradients used in DDC for normal-aware densification. (Sec. 3.2)

After geometric coupling, we optimize BRDF parameters and lighting via inverse PBR. Following R3DG [9], we sample Ns = 64 incident directions for each Gaussian and optimize the rendered appearance through physically-based rendering supervision against the ground-truth image. The inverse PBR loss is defined as:

LPBR = λ1L1 + λssimLssim + λlightLlight + λBRDFLBRDF. (S8)

Here, Llightregularizes lighting estimation for stable decomposition of illumination, while LBRDF regularizes the BRDF parameters.

#### B.3 Training Schedule

Our training follows the two-stage pipeline summarized in Alg. 1. We first warm up 3DGS for 15k iterations, and then perform geometric coupling for 30k iterations, followed by inverse PBR for 10k iterations.

- Stage 1: Geometric coupling. Each iteration applies an alternating update scheme: we run DSA (Alg. 2) while freezing 3DGS and updating the SDF, and then run NGA (Alg. 3) while freezing the SDF and updating 3DGS (Tab. 4). For the first 10k iterations, we use depth consistency for coarse alignment. We then enable pixel-wise normal consistency for the remaining 20k iterations to refine fine-scale geometry. We activate DDC after 10k iterations and apply it every 100 iterations to regulate Gaussian splitting.
- Stage 2: Inverse PBR. We fix the learned geometry and normals of the Gaussians and optimize BRDF parameters and lighting for 10k iterations using LPBR (Eq. (S8)), following inverse rendering setup of R3DG.

#### B.4 Hyperparameter Settings

In our framework, depth and normal consistency terms are shared by both the SDF and relightable 3DGS branches. {λdepth,λnormal} are set to {0.01,0.001}, where the normal weight is kept relatively small since normal gradients tend to dominate optimization, which often leads to unstable training. For the SDF branch, we employ image reconstruction, Eikonal, and curvature regularization. {λSDF1 ,λeik,λcurv} are set to {1.0,0.1,0.05}, following the configuration of GSDF [45]. For the relightable 3DGS branch, we adopt the weights from the first stage setup of R3DG [9]. {λ3DGS1 ,λssim,λO,λu} are set to {0.8,0.2,0.01,0.01}. The DDC weight is set to α = 0.2, which balances photometric and geometric gradients during Gaussian densification, as described in Sec. C.1.

After geometry optimization, we fix the learned geometry and normals of the Gaussians and optimize BRDF parameters and lighting in the second stage. {λ1,λssim,λlight,λbrdf} are set to {0.8,0.2,0.0001,0.01}, following the secondstage configuration of R3DG.

#### B.5 Complexity and Runtime

All experiments were conducted on a single NVIDIA A6000 GPU. On the DTU dataset, COREA requires approximately 2.4 hours of training per scene. We report peak GPU memory usage, and COREA consumes roughly 24 GB on average during DTU training. The implementation is built on PyTorch 2.3 and CUDA 12.1 within our unified pipeline for joint SDF and relightable 3DGS optimization.

### C Ablation Studies

#### C.1 Trade-off Analysis of α in DDC

We analyze how the DDC weight α affects rendering performance and Gaussian splitting behavior. To this end, we vary α ∈ {0,0.2,0.5,0.7,1.0} and measure three quantities: PSNR for SH-based NVS and inverse PBR, the number of Gaussians normalized by the count at α = 0, and the cosine similarity between the principal

1.2

- 27
- 28
- 29
- 30
- 31
- 32
- 33

32.19 32.27 32.17 32.2 32.17

- 0.8
- 1

29.72 29.57 29.55 29.53

0.6

29.47

0.4

0.2

0

0 0.2 0.5 0.7 1

DDC Weight α

NVS (PSNR↑) PBR (PSNR↑) # Gaussian Cosine Similarity

- Fig. S1: Effect of the DDC weight α on rendering quality and splitting dynamics. The bar plots show the PSNR for SH-based NVS and inverse PBR, and the line plot shows the cosine similarity between the principal eigenvectors of Simage and Stotal (Eq. (3)). The relative number of Gaussians (normalized with respect to α=0) shows how the Gaussian count changes with α. As α increases, the cosine similarity drops sharply for α ≥ 0.5, indicating a shift from image-driven to normal-driven splitting. The inverse PBR PSNR peaks at α=0.2, showing the best balance between geometric fidelity and controlled Gaussian growth.

eigenvectors of the image-only splitting matrix Simage and the combined matrix Stotal (Eq. (3)). As shown in Fig. S1, the cosine similarity rapidly decreases once α ≥ 0.5, indicating that the splitting direction gradually shifts from an image-driven to a normal-driven direction. This shift is accompanied by a slight increase in the relative number of Gaussians, suggesting that the normal loss induces additional splitting for fine geometric refinement. While the SH-based NVS PSNR remains stable for all α, the inverse PBR performance shows a mild improvement, peaking at α = 0.2. For larger α, the rendering quality slightly decreases, implying that excessive normal-driven splitting destabilizes geometry learning.

These observations indicate that the DDC weight α controls the balance between photometric and geometric gradients, thereby regulating geometric fidelity and Gaussian growth. Based on this empirical analysis, we adopt α = 0.2 as the default setting, which provides the best balance between geometric fidelity, rendering quality, and controlled Gaussian growth across all experiments.

#### C.2 Dual-Density Control for Efficient Gaussian Densification

To further validate the effect of DDC on Gaussian growth, we conduct an ablation study on on the DTU dataset, as shown in Tab. S1, which reports per-scene Gaussian counts. With the Adaptive Density Control (ADC) used in vanilla 3DGS, the accumulated pixel-wise gradients from image and normal losses frequently trigger excessive Gaussian splitting, resulting in inefficient over-densification. In contrast, DDC reduces the Gaussian count by 17% on average while preserving rendering fidelity across all scenes.

- Table S1: Comparison between Adaptive and Dual-Density Control. All values indicate the number of Gaussians in millions (M). The original Adaptive Density Control (ADC) in 3DGS tends to over-split Gaussians, whereas our Dual-Density Control (DDC) effectively suppresses redundant splitting through a dual-loss scheme that jointly considers image and normal gradients. By regulating the generation of split Gaussians, DDC achieves geometry-aware densification and reduces the overall Gaussian count by an average of 17%, significantly lowering memory usage and training overhead.

#Gaussians↓ 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 Avg. ADC 0.93 1.21 1.78 1.14 0.19 0.21 0.34 0.13 0.51 0.32 0.39 0.16 0.37 0.27 0.24 0.54 DDC (Ours) 0.89 1.18 1.65 1.08 0.19 0.21 0.36 0.13 0.50 0.32 0.37 0.16 0.36 0.26 0.24 0.45 (17%↓)

||[Figure 199]|
|---|
<br><br>|[Figure 200]|
|---|
<br><br>|[Figure 201]|
|---|
<br><br>|[Figure 202]|
|---|
<br><br>|[Figure 203]|
|---|
<br><br>|[Figure 204]|
|---|
<br><br>|[Figure 205]|
|---|
<br><br>|[Figure 206]|
|---|
<br><br>|[Figure 207]|
|---|
<br><br>|[Figure 208]|
|---|
<br><br>|[Figure 209]|
|---|
<br><br>|[Figure 210]|
|---|
<br><br>|[Figure 211]|
|---|
<br><br>|[Figure 212]|
|---|
<br><br>|[Figure 213]|
|---|
<br><br>|[Figure 214]|
|---|
<br><br>|[Figure 215]|
|---|
<br><br>| |
|---|
<br><br>: SH-based NVS<br><br>| |
|---|
<br><br>: Inverse PBR<br><br>| |
|---|
<br><br>: Surface Reconstruction<br><br>|[Figure 216]|
|---|
|
|---|

- Fig. S2: Additional Qualitative Results. We show that COREA jointly supports SH-based NVS (blue), surface reconstruction (orange), and inverse PBR (green) within a unified three-tasks framework. Each row presents a different real-world scene, where COREA produces sharp novel views, accurate surface geometry, and stable relighting under white and HDR environment lighting, with clean object-background separation. These results highlight the effectiveness of the geometric coupling between an SDF and relightable 3DGS, which provides a shared geometric foundation across all three tasks.

By jointly considering photometric and geometric gradients, DDC triggers densification only when the combined gradients indicate meaningful refinement, effectively suppressing redundant growth. This produces geometry-aware densification that remains compact and geometrically consistent, significantly reducing memory usage and improving the computational efficiency of inverse PBR, since fewer Gaussians participate in ray sampling and shading evaluations. The resulting representation is lightweight yet geometrically accurate, supporting high-quality rendering and stable relighting.

### D More experiments

#### D.1 Per-scene Results

We present additional qualitative results in Fig. S2, which demonstrate the visual performance of our method across diverse scenes. Per-scene quantitative results

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

Ours (0.690)

MILo (0.650)

PGSR (0.487)

GT (CD)

- Fig. S3: Qualitative comparison of surface reconstruction under different CD. We compare COREA with PGSR and MILo, which achieve the best and second-best CD in Tab. S2, on an example DTU scene. Although COREA shows a higher CD (values in parentheses; lower is better), the zoomed-in region reveals that PGSR and MILo produce smoother surfaces with less distinct local patterns, whereas COREA preserves sharper surface structures. This suggests that CD does not fully reflect fine-scale surface details that affect local geometry and normal fidelity.

- Table S2: Additional comparison to mesh-oriented baselines on DTU. We report NVS image metrics (PSNR/SSIM/LPIPS) and surface reconstruction (SR) accuracy (CD) under a unified protocol for representative mesh-oriented 3DGS methods. Most baselines focus on surface reconstruction and do not support inverse PBR, while GSROR2 supports inverse PBR but does not provide SH-based NVS. For a consistent imagequality reference, we report image metrics from each method’s supported rendering mode (SH-based NVS when available, otherwise PBR rendering). COREA is the only unified framework in this table that simultaneously supports SH-based NVS, surface reconstruction, and inverse PBR.

Method SH SR PBR PSNR↑ SSIM↑ LPIPS↓ CD↓ 2DGS (SIGGRAPH’24) ✓ ✓ ✗ 34.65 0.939 0.165 0.801 GOF (TOG’24) ✓ ✓ ✗ 28.70 0.909 0.124 0.816 RaDe-GS (Arxiv’24) ✓ ✓ ✗ 28.99 0.917 0.158 0.677 PGSR (TVCG’24) ✓ ✓ ✗ 28.85 0.910 0.178 0.545 MILo (TOG’24) ✓ ✓ ✗ 29.45 0.919 0.159 0.673 GS-ROR2 (TOG’25) ✗ ✓ ✓ 19.76 0.812 0.219 1.562 COREA (Ours) ✓ ✓ ✓ 32.27 0.955 0.094 0.824

for SH-based NVS, surface reconstruction (SR), and inverse PBR are provided in Tabs. S3-S7.

#### D.2 Surface Reconstruction Comparison

To provide a broader empirical context, we additionally compare against representative mesh-oriented 3DGS methods [6,13,16,46,47,55] on DTU under a unified evaluation protocol. These methods primarily target surface reconstruction quality. Accordingly, we report image metrics and CD under each method’s supported rendering mode to provide a consistent comparison across methods with different rendering capabilities. Tab. S2 summarizes the results. COREA is the only unified framework in this comparison that simultaneously supports SH-based NVS, surface reconstruction, and inverse PBR. While COREA is not optimized solely for surface reconstruction, it maintains competitive CD on DTU, achieves strong NVS quality, and additionally supports inverse PBR.

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

SH

PBR

Normal

GT

- Fig. S4: Failure cases under low visibility and in thin-structure regions. When observations are sparse or the geometry contains thin structures, both Gaussian coverage and SDF sampling can become unreliable. This may lead to over-smoothed local geometry and unstable relighting results.

Fig. S3 highlights a qualitative discrepancy between CD and perceived surface detail. Although COREA has a higher CD than PGSR and MILo, the meshes reconstructed by these methods appear overly smooth in the magnified region, with local surface patterns becoming less distinct. In contrast, COREA better preserves fine local surface structures that influence normal fidelity, indicating that a lower CD does not necessarily imply higher fidelity in fine-scale geometry. [5]

### E Limitations

Our framework relies on cross-representation geometric consistency between an SDF and relightable 3DGS, which can become fragile in geometrically ambiguous regions. In sparse-view regions, depth and pixel-wise depth gradients can be weak or ambiguous due to limited visibility and depth discontinuities. In such cases, early alignment errors may trigger a cascading feedback loop: Gaussians may collapse or become overly sparse, SDF ray sampling becomes biased around inaccurate depth anchors, and the local geometry progressively blurs. This degradation is particularly visible under relighting, where small normal inaccuracies can amplify shading instability (Fig. S4).

A promising direction is to incorporate pixel-weighted Gaussian representations such as Pixel-GS [54], which may enable more reliable densification under sparse initial point clouds and improve Gaussian formation in low-visibility regions. This may further stabilize relighting in challenging cases.

Table S3: Quantitative results for SH-based NVS on the DTU dataset.

###### PSNR↑ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122

- 3DGS ✗ ✗ 23.63 21.45 23.29 23.66 28.36 28.35 27.64 30.30 24.43 27.55 28.82 28.73 27.18 32.28 32.06 GaussianShader ✗ ✓ 22.28 9.61 20.57 23.09 16.54 21.98 24.51 23.89 17.89 23.52 19.86 25.86 22.09 29.24 29.72 GS-IR ✗ ✓ 23.86 21.55 23.18 24.66 28.74 28.32 28.09 30.43 24.45 27.65 29.45 29.65 27.33 32.86 33.07 R3DG ✗ ✓ 28.54 25.41 27.62 32.98 31.39 31.82 29.60 32.43 27.04 30.01 35.51 33.61 31.16 37.98 37.79 SVG-IR ✗ ✓ 23.80 21.94 23.92 25.62 28.50 27.14 29.78 30.94 25.65 27.74 30.21 30.98 28.32 32.44 33.33

GOF ✓ ✗ 24.18 22.50 24.25 24.12 30.53 28.35 26.40 33.57 25.79 30.14 28.84 28.67 25.52 31.14 30.98 GSDF ✓ ✗ 24.87 22.09 23.84 25.74 28.48 29.18 26.81 25.45 24.92 27.74 29.80 29.51 25.04 32.18 31.12 GS-pull ✓ ✗ 23.87 21.75 23.91 24.88 28.09 27.19 27.70 29.69 24.73 26.83 29.59 28.41 27.57 32.33 32.24 COREA (Ours) ✓ ✓ 28.47 25.35 26.95 32.89 31.49 31.87 29.68 32.43 27.38 30.11 35.46 33.48 31.34 38.17 38.13

###### SSIM↑ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122

3DGS ✗ ✗ 0.728 0.720 0.674 0.802 0.910 0.927 0.883 0.946 0.865 0.877 0.870 0.906 0.847 0.911 0.925 GaussianShader ✗ ✓ 0.711 0.592 0.642 0.814 0.797 0.915 0.801 0.869 0.760 0.816 0.789 0.881 0.800 0.905 0.892 GS-IR ✗ ✓ 0.730 0.725 0.665 0.809 0.929 0.929 0.886 0.947 0.865 0.874 0.866 0.911 0.847 0.912 0.928 R3DG ✗ ✓ 0.921 0.894 0.891 0.966 0.961 0.961 0.935 0.969 0.932 0.944 0.957 0.954 0.937 0.966 0.971 SVG-IR ✗ ✓ 0.869 0.859 0.836 0.901 0.951 0.954 0.926 0.960 0.918 0.926 0.942 0.959 0.928 0.957 0.963

GOF ✓ ✗ 0.749 0.741 0.700 0.825 0.941 0.926 0.893 0.965 0.878 0.900 0.880 0.925 0.846 0.913 0.931 GSDF ✓ ✗ 0.774 0.741 0.697 0.847 0.932 0.939 0.844 0.889 0.881 0.884 0.886 0.913 0.820 0.908 0.910 GS-pull ✓ ✗ 0.750 0.745 0.691 0.818 0.926 0.927 0.888 0.941 0.872 0.870 0.882 0.907 0.854 0.915 0.928 COREA(Ours) ✓ ✓ 0.922 0.894 0.930 0.976 0.967 0.968 0.951 0.970 0.957 0.955 0.964 0.961 0.937 0.967 0.973

###### LPIPS↓ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122

3DGS ✗ ✗ 0.289 0.224 0.325 0.174 0.129 0.122 0.222 0.113 0.183 0.201 0.195 0.177 0.219 0.167 0.125 GaussianShader ✗ ✓ 0.264 0.318 0.304 0.180 0.206 0.144 0.244 0.143 0.229 0.230 0.216 0.193 0.246 0.172 0.137 GS-IR ✗ ✓ 0.246 0.203 0.282 0.152 0.120 0.116 0.207 0.102 0.173 0.179 0.179 0.164 0.202 0.148 0.111 R3DG ✗ ✓ 0.095 0.105 0.157 0.056 0.073 0.080 0.146 0.075 0.113 0.109 0.100 0.121 0.105 0.091 0.063 SVG-IR ✗ ✓ 0.093 0.079 0.127 0.055 0.035 0.037 0.057 0.027 0.053 0.050 0.046 0.050 0.065 0.035 0.023

GOF ✓ ✗ 0.128 0.111 0.160 0.079 0.035 0.069 0.079 0.021 0.065 0.048 0.063 0.048 0.088 0.053 0.041

- GSDF ✓ ✗ 0.243 0.206 0.294 0.152 0.124 0.115 0.225 0.136 0.170 0.186 0.179 0.179 0.218 0.152 0.121 GS-pull ✓ ✗ 0.321 0.245 0.351 0.187 0.144 0.136 0.250 0.126 0.205 0.226 0.212 0.194 0.248 0.186 0.141 COREA(Ours) ✓ ✓ 0.088 0.099 0.134 0.050 0.072 0.077 0.142 0.065 0.100 0.099 0.097 0.113 0.105 0.085 0.056

Table S4: Quantitative results for surface reconstruction (CD) on the DTU dataset.

CD↓ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 GOF ✓ ✗ 0.45 0.80 0.56 0.44 1.50 1.18 0.77 1.16 1.42 0.57 0.58 1.08 0.54 0.49 0.42

- GSDF ✓ ✗ 1.03 0.90 0.47 0.44 1.32 0.84 0.76 1.59 1.33 0.78 0.60 1.30 0.39 0.52 0.56 GS-pull ✓ ✗ 0.62 0.71 1.45 0.63 0.90 1.30 0.88 6.26 1.45 2.11 0.80 0.99 0.55 0.66 12.83 GS-ROR ✓ ✓ 3.16 4.60 3.51 0.26 2.06 1.95 2.76 2.24 3.10 1.90 1.87 3.47 3.24 2.11 2.06 COREA(Ours) ✓ ✓ 0.77 0.87 0.58 0.38 1.41 0.94 0.70 1.49 1.36 0.77 0.69 1.02 0.36 0.53 0.49

###### Table S5: Quantitative results for Inverse PBR on the DTU dataset.

PSNR↑ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122 GaussianShader ✗ ✓ 19.56 23.21 17.83 15.76 23.18 16.75 16.04 24.00 20.43 23.73 17.35 17.59 16.76 18.36 18.05

- GS-IR ✗ ✓ 21.06 20.46 21.73 22.77 25.40 23.53 26.07 29.24 23.24 24.61 27.16 27.77 25.04 27.58 27.48 R3DG ✗ ✓ 24.32 22.36 24.52 27.14 27.26 27.38 26.03 27.13 23.77 25.93 29.45 25.61 25.26 30.56 31.53 SVG-IR ✗ ✓ OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM GS-ROR ✓ ✓ 15.74 15.68 16.34 18.75 20.99 15.84 22.01 26.47 20.54 22.47 22.26 20.71 16.69 25.07 26.25 COREA (Ours) ✓ ✓ 26.98 25.48 27.64 30.25 30.28 30.50 28.31 31.16 25.82 29.40 32.20 29.89 27.70 33.38 34.58

SSIM↑ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122

GaussianShader ✗ ✓ 0.827 0.843 0.733 0.563 0.897 0.729 0.574 0.823 0.719 0.831 0.614 0.527 0.652 0.598 0.599 GS-IR ✗ ✓ 0.747 0.781 0.734 0.662 0.830 0.715 0.762 0.952 0.892 0.602 0.799 0.826 0.621 0.581 0.539 R3DG ✗ ✓ 0.869 0.850 0.850 0.932 0.917 0.946 0.896 0.948 0.890 0.908 0.932 0.913 0.911 0.944 0.953 SVG-IR ✗ ✓ OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM GS-ROR ✓ ✓ 0.724 0.751 0.712 0.822 0.918 0.894 0.869 0.940 0.859 0.880 0.877 0.867 0.817 0.908 0.926 COREA (Ours) ✓ ✓ 0.917 0.909 0.902 0.960 0.955 0.959 0.928 0.963 0.934 0.938 0.947 0.941 0.934 0.959 0.969

LPIPS↓ SR PBR 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122

GaussianShader ✗ ✓ 0.213 0.225 0.313 0.519 0.208 0.458 0.519 0.396 0.399 0.345 0.455 0.497 0.436 0.493 0.495 GS-IR ✗ ✓ 0.198 0.176 0.265 0.217 0.154 0.205 0.258 0.086 0.145 0.231 0.208 0.200 0.270 0.204 0.240 R3DG ✗ ✓ 0.154 0.142 0.200 0.089 0.105 0.097 0.196 0.104 0.154 0.149 0.133 0.166 0.136 0.118 0.082 SVG-IR ✗ ✓ OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM GS-ROR ✓ ✓ 0.314 0.241 0.329 0.175 0.149 0.180 0.238 0.113 0.204 0.211 0.193 0.199 0.249 0.175 0.125 COREA (Ours) ✓ ✓ 0.105 0.106 0.160 0.064 0.086 0.085 0.150 0.076 0.122 0.115 0.107 0.132 0.131 0.094 0.062

Table S6: Quantitative results for SH-based NVS on the Tanks&Temples dataset.

Barn Caterpillar Family Truck Method SR PBR PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

3DGS ✗ ✗ 27.53 0.879 0.158 24.45 0.891 0.133 32.34 0.963 0.050 25.51 0.908 0.119 GaussianShader ✗ ✓ 21.97 0.878 0.137 21.50 0.867 0.119 20.47 0.904 0.088 15.89 0.754 0.213 GS-IR ✗ ✓ 28.28 0.900 0.141 26.75 0.933 0.077 32.65 0.965 0.041 26.02 0.918 0.096 R3DG ✗ ✓ 28.82 0.906 0.138 27.92 0.949 0.067 34.60 0.977 0.031 26.34 0.933 0.081 SVG-IR ✗ ✓ 27.49 0.906 0.074 26.73 0.944 0.037 31.04 0.971 0.021 25.49 0.918 0.050

GOF ✓ ✗ OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM GSDF ✓ ✗ OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM GS-pull ✓ ✗ 26.25 0.847 0.206 22.80 0.870 0.158 31.38 0.957 0.060 24.71 0.897 0.143 COREA(Ours) ✓ ✓ 27.71 0.894 0.145 27.75 0.943 0.083 34.78 0.978 0.035 26.76 0.934 0.096

Table S7: Quantitative results for Inverse PBR on the Tanks&Temples dataset.

Barn Caterpillar Family Truck Method SR PBR PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ GaussianShader ✗ ✓ 18.72 0.816 0.202 17.54 0.849 0.156 20.61 0.888 0.103 14.40 0.762 0.215

- GS-IR ✗ ✓ 22.27 0.611 0.249 21.45 0.390 0.191 28.05 0.924 0.079 20.08 0.682 0.225 R3DG ✗ ✓ 26.11 0.881 0.159 24.27 0.900 0.098 30.52 0.960 0.046 24.35 0.906 0.108 SVG-IR ✗ ✓ OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM GS-ROR ✓ ✓ 22.20 0.825 0.252 21.54 0.902 0.109 26.27 0.943 0.075 19.75 0.879 0.145 COREA(Ours) ✓ ✓ 26.98 0.895 0.138 25.14 0.908 0.103 31.99 0.967 0.045 25.42 0.920 0.107

