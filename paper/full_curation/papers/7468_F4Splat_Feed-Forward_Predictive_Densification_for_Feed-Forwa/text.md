# arXiv:2603.21304v2[cs.CV]25Mar2026

### F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting

Injae Kim1, Chaehyeon Kim2, Minseong Bae1, Minseok Joo2, and Hyunwoo J. Kim1

1 KAIST 2 Korea University https://mlvlab.github.io/F4Splat

Abstract. Feed-forward 3D Gaussian Splatting methods enable singlepass reconstruction and real-time rendering. However, they typically adopt rigid pixel-to-Gaussian or voxel-to-Gaussian pipelines that uniformly allocate Gaussians, leading to redundant Gaussians across views. Moreover, they lack an effective mechanism to control the total number of Gaussians while maintaining reconstruction fidelity. To address these limitations, we present F4Splat, which performs Feed-Forward predictive densification for Feed-Forward 3D Gaussian Splatting, introducing a densification-score-guided allocation strategy that adaptively distributes Gaussians according to spatial complexity and multi-view overlap. Our model predicts per-region densification scores to estimate the required Gaussian density and allows explicit control over the final Gaussian budget without retraining. This spatially adaptive allocation reduces redundancy in simple regions and minimizes duplicate Gaussians across overlapping views, producing compact yet high-quality 3D representations. Extensive experiments demonstrate that our model achieves superior novel-view synthesis performance compared to prior uncalibrated feed-forward methods, while using significantly fewer Gaussians.

Keywords: Feed-Forward 3D Gaussian Splatting · Compact 3DGS

#### 1 Introduction

In modern computer vision, 3D scene reconstruction using deep learning has become the de facto standard. In particular, 3D Gaussian Splatting (3DGS) [25] has emerged as a highly efficient alternative to existing methodologies [15,36,38,42]. 3DGS represents scenes using an explicit set of 3D Gaussian primitives, enabling high-fidelity 3D reconstruction and real-time novel-view rendering. It incorporates adaptive density control (ADC), which periodically adds or removes Gaussians during optimization. These iterative updates assign a different number of Gaussians in each region, and through this adaptive assignment, the final 3DGS representation achieves high reconstruction fidelity with a relatively small number of Gaussians. However, the conventional 3DGS framework still inherits key limitations shared by other optimization-based 3D reconstruction methods [15,37,38]. It requires costly per-scene iterative optimization and typically

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### 2 Kim et al.

63K 44K 24K

# of Gaussians

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

…

[Figure 11]

[Figure 12]

###### Ours

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Context Images

Densification Score-based allocation

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

###### AnySplat

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Context Images

Voxel-based uniform allocation

| |
|---|

| |
|---|

Rendered View

Predicted 3D Gaussians

(a) Densification-score-guided allocation of F4Splat

(b) Rendering quality vs. Number of Gaussians

- Fig. 1: Comparison of Gaussian allocation under different target Gaussian budgets. (a) Given the same target Gaussian budget, F4Splat allocates Gaussians non-uniformly using predicted densification scores, avoiding over-allocation in simple regions and concentrating primitives on fine-detail regions (e.g., flowers) to better preserve details even under a small budget, in contrast to voxel-based uniform allocation (AnySplat [23]). (b) Compared to pose-free and uncalibrated baselines, F4Splat achieves the best reconstruction fidelity (LPIPS/PSNR), while largely preserving the rendering quality as the number of Gaussians is reduced.

relies on densely captured input views with known camera parameters, which can be impractical in real-world scenarios. This has motivated feed-forward 3DGS methods [4,8,22,23,32,54,56], which are trained on large-scale datasets to build strong 3D priors. These frameworks can reconstruct a 3D scene from only a few input images through a single forward pass, preserving the real-time rendering capabilities of 3DGS while enabling generalization to unseen scenes. However, existing feed-forward 3DGS methods have a significant limitation in that they do not allocate Gaussians efficiently. This limitation stems from removing the iterative optimization process of conventional 3DGS, which also eliminates its periodic adaptive density control (ADC) that densifies Gaussians during training. Most works [4,8,22,32,54,56,62] adopt a pixel-to-Gaussian pipeline, which assigns Gaussians at the pixel level. This fixes the total number of Gaussians to the number of pixels in the entire image and prevents flexible adjustment of Gaussian positions, resulting in duplicated Gaussians across different views. AnySplat [23], which employs a voxel-to-Gaussian pipeline, can adjust the number of Gaussians by changing the voxel size, but this typically requires training a new model. Moreover, because it allocates Gaussians uniformly in space (i.e., assigning one Gaussian per voxel), it struggles to produce a high-quality and compact Gaussian representation under a limited Gaussian budget.

To address inefficient Gaussian allocation, we introduce F4Splat, a feedforward network that performs predictive densification for 3D Gaussian Splatting from a set of uncalibrated images. Our approach predicts densification decisions in a single forward pass, treating Gaussian densification as a learnable

prediction problem within a unified feed-forward pipeline. Specifically, the network estimates a densification score that indicates whether additional Gaussians should be allocated to each region. By estimating both spatial complexity and multi-view overlap, the predicted densification score avoids over-allocation in simple regions and prevents duplicate Gaussians in areas covered by overlapping input images. This feed-forward densification strategy enables spatially adaptive allocation and yields compact Gaussian representations while maintaining competitive reconstruction fidelity. As illustrated in Fig. 1, this allows F4Splat to concentrate Gaussians on fine-detail regions while avoiding unnecessary allocation in simple regions, achieving higher rendering quality under the same Gaussian budget. Through extensive experiments, F4Splat achieves on-par or superior novel-view synthesis quality while using significantly fewer Gaussians than prior uncalibrated feed-forward methods that rely solely on image inputs.

The contributions of our work can be summarized as:

- – Gaussian-count controllable feed-forward 3DGS. We propose F4Splat, a feed-forward framework that reconstructs 3D Gaussian Splatting representations from sparse, uncalibrated images while enabling explicit control over the final number of Gaussians through feed-forward predictive densification.
- – Densification-score-guided allocation for high fidelity under a limited budget. We introduce a densification score that predicts where additional Gaussians should be allocated, enabling spatially adaptive Gaussian allocation without iterative optimization and maintaining high reconstruction fidelity even under a limited Gaussian budget.
- – State-of-the-art performance in the uncalibrated setting. Extensive experiments show that F4Splat achieves on-par or superior novel-view synthesis quality while using significantly fewer Gaussians than prior uncalibrated feed-forward methods that rely solely on image inputs.

- 2 Related Work
- 3D Gaussian Splatting for Novel View Synthesis. NeRF [37] established a dominant paradigm for neural scene representation and sparked extensive subsequent research [1–3,5,15,35,38,43,58], driving rapid progress in neural scene reconstruction. However, its per-ray volumetric rendering incurs high compute cost, motivating more efficient alternatives. 3DGS [25] mitigates this inefficiency by representing a scene with a set of 3D Gaussian primitives and rendering them through differentiable rasterization, enabling real-time rendering and faster optimization. To achieve high fidelity with a compact representation, it further employs adaptive density control (ADC), which periodically adds Gaussians in under-represented regions and prunes primitives with negligible contribution during iterative optimization. This has inspired a line of work on more compact 3DGS representations, spanning both refinements of the ADC strategy [10,13,17,26,27,34,45,57,60,63] and various compaction and pruning methods [6,12,16,29,39,41,49,55].

Despite the advantages of 3DGS, it still has several practical limitations. It typically assumes dozens to hundreds of diverse input views for stable reconstruction, which can be impractical in real-world scenarios. This dense-view requirement has been partly addressed by recent studies on sparse-view 3DGS [18, 28,31,53,61,66], which aim to reconstruct 3D scenes from only a few input images. Another major limitation is that 3DGS still requires iterative per-scene optimization, which remains a significant burden for practical deployment. To reduce this time-consuming optimization process, a variety of recent approaches [7, 14,19,48,64] have sought to accelerate the original 3DGS optimization pipeline through more efficient rasterization, parallelization, and improved optimization strategies. Among these approaches, feed-forward 3DGS represents a particularly promising paradigm, as it amortizes iterative optimization into a single feed-forward pass, thereby enabling much faster 3D reconstruction.

Feed-Forward 3D Gaussian Splatting. Feed-forward 3DGS approaches [4, 8, 9, 20, 22–24, 32, 47, 52, 56, 62] have been proposed to alleviate the costly perscene optimization of standard 3DGS. These methods are trained on large-scale datasets to learn strong priors, allowing them to predict 3D Gaussian representations in a single feed-forward pass without iterative optimization. Consequently, they can reconstruct from sparse views while enabling real-time rendering and generalization to unseen scenes. Early generalizable feed-forward 3DGS methods [4,8,52] typically assume calibrated multi-view inputs with known camera poses. Recent works [9,47,56,62] relax this assumption by moving to pose-free settings. In addition, self-supervised pose-free approaches [20,22,24] further reduce reliance on pose annotations by learning from reconstruction consistency, with pose estimation integrated into the pipeline. More recently, uncalibrated formulations [23,32] are enabling reconstruction without camera calibration.

Despite these advances in efficiency and robustness, existing feed-forward 3DGS methods largely rely on a uniform output parameterization, in which a fixed number of Gaussians is allocated per pixel or spatial unit. As a result, the total Gaussian count becomes tightly coupled with the input resolution, rather than being adaptively allocated according to scene complexity. This leads to redundant primitives in simple regions, while failing to sufficiently model geometrically complex regions, resulting in a suboptimal and non-compact representation under a limited Gaussian budget. In conventional optimization-based 3DGS, this issue has been mitigated through adaptive density control (ADC), which dynamically allocates Gaussians based on scene structure. However, such mechanisms rely on iterative per-scene optimization and are therefore not directly applicable to feed-forward pipelines. The recent feed-forward approach, AnySplat [23], is able to control the Gaussian count via voxel granularity. However, the allocation remains spatially uniform, and adapting to different budgets typically requires retraining, limiting flexibility and compactness. In contrast, we introduce Gaussian-count controllable feed-forward 3DGS, which predicts a budget-aware densification score that enables non-uniform and spatially adaptive Gaussian allocation. This yields a more compact 3D representation under a controllable Gaussian budget.

#### 3 Method

We propose F4Splat, a feed-forward network that generates 3D Gaussian primitives [25] from an image collection via feed-forward predictive densification. Unlike prior feed-forward 3DGS methods that rely on uniform allocation, our method allows users to adjust the number of Gaussians on demand through spatially adaptive Gaussian allocation, making more effective use of the available Gaussian budget. In this section, we formulate the problem in Sec. 3.1. Next, Sec. 3.2 presents the overall framework, and Sec. 3.3 details the training pipeline.

##### 3.1 Problem Formulation

i=1 , where Ictxi ∈ R3×H×W, most prior feed-forward 3D Gaussian Splatting (3DGS) works [8,22,23,32,54,56,62] uniformly allocate one Gaussian per pixel, resulting in a fixed number of Gaussians, NctxHW, to represent the scene. In contrast, our goal is to develop a feed-forward network Fθ that enables control over the number of Gaussians. Specifically, Fθ takes as input not only the context images but also a user-specified target Gaussian budget N¯G, and predicts a set of 3D Gaussian primitives G = {gg}N

Given Nctx input context images {Ictxi }N

ctx

g=1 and camera parameters {Pˆctxi }N

G

i=1 :

ctx

g=1, {Pˆctxi }N

i=1 = Fθ {Ictxi }N

{gg}N

i=1 , N¯G . (1)

ctx

ctx

G

Each Gaussian primitive gg ∈ RdG is parameterized by center µg ∈ R3, opacity σg ∈ R, rotation in quaternion qg ∈ R4, scale sg ∈ R3, and spherical harmonics

(SH) hg ∈ Rν. Each camera parameter tuple is denoted as Pˆctxi = (Kˆ ctxi ,Tˆctxi ), where Kˆ ctxi ∈ R3×3 is the intrinsic matrix and Tˆctxi ∈ R4×4 is the camera-toworld pose. We use fˆictx to denote the focal length encoded in Kˆ ctxi . Throughout the paper, (ˆ·) denotes quantities predicted by our network.

It is not enough to merely control the number of Gaussians; it is also important to use them efficiently to generate a high-quality scene representation. To this end, Gaussians should be allocated non-uniformly across the scene according to local characteristics, assigning more capacity to geometrically or visually complex regions. In the case of AnySplat [23], the number of Gaussians can be adjusted by changing the voxel size. However, due to the inherent limitation of uniformly assigning a single Gaussian primitive to each voxel, it represents the scene less faithfully even under the same Gaussian count. Uniform-Gaussianallocation methods [4,8,22,23,32,54,56,62] ignore the fact that different regions require different Gaussian densities, yielding redundant Gaussians in simple regions while under-allocating complex ones. Consequently, the Gaussian budget is not spent where it is most needed for faithful scene representation. To address this, we introduce a spatially adaptive Gaussian allocation framework that allocates Gaussians more effectively within a given budget.

Training Pipeline.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

…

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

…

Gaussian Center Head

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

…

[Figure 62]

[Figure 63]

Geometry Backbone

[Figure 64]

[Figure 65]

Gaussian Param Maps

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

…

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Gaussian Param Head

[Figure 102]

[Figure 103]

…

[Figure 104]

[Figure 105]

[Figure 106]

Context Images

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Spatially Adaptive Gaussian Allocation

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Densification Score Maps

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

Inference Pipeline.

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

: context

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

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

…

camera : target

[Figure 166]

…

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Gaussian Center Head

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

…

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

camera

…

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Geometry Backbone

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Spatially Adaptive Density Allocation

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

…

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

: Gaussian allocation (eq. (3))

[Figure 273]

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

…

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

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

Spatially Adaptive Density Allocation

[Figure 337]

[Figure 338]

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

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Gaussian Param Head

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

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

…

[Figure 381]

[Figure 382]

[Figure 383]

Spatially Adaptive Gaussian Allocation

: random sampling

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

- Fig. 2: Overview of F4Splat. Given multi-view context images, the Geometry Backbone predicts camera parameters, multi-scale Gaussian parameter maps, and densification score maps. During training, camera and Gaussian parameters are jointly optimized with the camera loss Lcamera, rendering loss Lrender, and scene-scale regularization Lscene. The predicted densification score maps are trained by score loss Lscore derived from the backpropagation of rendering loss. The final representation Gτ, constructed using a randomly sampled threshold τ, is also supervised by the rendering loss Lrender. At inference time, the predicted Gaussian parameter maps and densification

scores are used to generate compact, high-fidelity 3D Gaussian representations GτN¯G tailored to user-specified Gaussian budgets N¯G. This adaptation is efficient and does not require retraining.

##### 3.2 Spatially Adaptive Gaussian Allocation

As illustrated in Fig. 2, our framework consists of three parts: a Geometry Backbone that encodes geometric information from a multi-view image set and predicts camera parameters; Gaussian Center Head and Gaussian Parameter Head, which predict multi-scale Gaussian parameter maps along with densification score maps; and Spatially Adaptive Gaussian Allocation that effectively allocates the available Gaussian budget.

Geometry Backbone. To encode geometric information from a given image set {Ictxi }N

i=1 , we adopt a geometric backbone following the structure of VGGT [50]. Each input image Ictxi is processed by a pretrained DINOv2 encoder [40] to extract patch tokens. The resulting image tokens tIi are concatenated with learnable camera tokens tPi and register tokens tri. The reference view has its own learnable camera and register tokens, while the remaining views share their corresponding tokens. The combined tokens {[tIi;tPi ;tri]}N

ctx

i=1 are then passed through alternating frame-wise and global self-attention layers. The encoded camera tokens {t˜iP}N

ctx

i=1 are passed through four additional self-attention layers, followed by a projection head to estimate camera parameters {Pˆctxi }N

ctx

i=1 .

ctx

# of Gaussians

26K 52K 78K 104K

Densification score map

Context Images

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

ACIDRealEstate10K

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

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

- Fig. 3: Spatially adaptive Gaussian allocation. Given two context images (left), the model predicts densification score maps that estimate where additional Gaussian density is required. The red color indicates where more Gaussians are placed based on different fixed thresholds τ. In the top example (RealEstate10K [65]), we can clearly observe that more Gaussians are allocated in complex regions. The example below (ACID [33]) demonstrates redundancy-aware allocation across overlapping views, avoiding unnecessary allocations.

Multi-Scale Prediction. To control the final number of Gaussians, multiscale Gaussian parameter maps {Gli}Ll=1 and densification score maps {Dˆ li}Ll=1−1 are predicted from the encoded image tokens t˜iI encoded by the geometry backbone, where Gli ∈ RdG×Hl×Wl and Dˆ li ∈ RH

l×Wl. We modify a DPT-based decoder [44] to introduce two parallel heads, a Gaussian Center Head and a Gaussian Parameter Head. Before the final two layers, the decoded feature maps are bilinearly interpolated to the target resolution (Hl,Wl) at each level, and then the level-specific layers are applied. Each level-specific module consists of only two layers, enabling efficient multi-scale map prediction. The Gaussian center head predicts the Gaussian centers, and the Gaussian parameter head predicts the remaining Gaussian primitives along with the densification score maps. In the Gaussian parameter head, an RGB shortcut [56] is utilized before the levelspecific layers. As the level l increases, the spatial resolution doubles at each step, (Hl+1,Wl+1) = (2Hl,2Wl). By exclusively selecting a scale level for each spatial region, we can control the final number of Gaussians NG. In the extreme case, selecting all regions from the coarsest level (l=1) yields NctxH1W1 Gaussians, while selecting all regions from the finest level (l =L) yields NctxHLWL Gaussians. Therefore, NG is bounded as:

NctxH1W1 ≤ NG ≤ NctxHLWL. (2)

Spatially Adaptive Gaussian Allocation. Meanwhile, to represent a scene faithfully under a limited Gaussian budget, more Gaussians should be allocated to geometrically or photometrically complex regions. Additionally, redundant allocations to the same spatial locations across overlapping views should be minimized. If we can estimate how densely Gaussians should be in a given local space, we can allocate Gaussians more efficiently across the scene. To this

l×Wl, which indicates how densely Gaussians should be placed in each spatial region. More details on the computation of the densification map are provided in the following Sec. 3.3.

end, we utilize a densification score map Dˆ li ∈ RH

Using the densification score maps, we determine the appropriate representation level for each region via a simple thresholding rule. Starting from the coarsest level (l = 1), if the densification score is higher than a given threshold τ, more Gaussians are allocated to that region from a higher-level Gaussian map. Ultimately, as illustrated in Fig. 2, Gaussians are selected such that allocations are

l×Wl, which indicate whether a particular location is allocated, can be computed as:

non-overlapping across levels. The binary allocation masks Mlτ,i ∈ {0,1}H

 

i<τ} if l = 1, 1{Dˆl

1{Dˆl

i<τ} ⊙ 1 − lk−=11 Up(Mlτ,i−k;2k) if 1 < l < L, 1 − lk−=11 Up(Mlτ,i−k;2k) if l = L,

Mlτ,i =

(3)



where 1{·} is an indicator function that outputs 1 when the condition is satisfied, 1 denotes a matrix of ones, ⊙ is the element-wise product, and Up(·;2k) denotes the nearest-neighbor upsampling with a scaling factor of 2k. Using these masks, the final 3D Gaussian representation Gτ = {gg}NG

g=1 is generated, where the total number of Gaussians NG

τ

= l,i ||Mlτ,i||1. Given a target Gaussian-

τ

count budget N¯G, we can compute the minimum threshold τN¯G that satisfies the target budget by a simple budget-matching algorithm, since the Gaussians are

exclusively selected across levels. The computed threshold τN¯G guarantees that the final number of Gaussians NG

satisfies the following conditions: 0 ≤ N¯G − NG

τN¯G

< 4L−1 − 1. (4) The budget-matching algorithm is provided in the supplementary materials.

τN¯G

##### 3.3 Training Strategy

Feed-Forward Predictive Densification. To allocate a limited number of Gaussians adaptively, the densification signal must satisfy two key properties. First, it should correlate with potential quality gain. It should allocate more Gaussians to under-represented regions and fewer to simple regions, so that representation quality tends to improve as the number of Gaussians increases. Second, it must be available at inference time without requiring iterative optimization. A desirable densification score must be computable using only the input images. To satisfy these conditions, we draw inspiration from the adaptive density control (ADC) strategy of standard 3DGS works [25, 57], which iteratively optimizes 3D Gaussian primitives.

[Figure 418]

threshold 1.5 2.5

[Figure 419]

Context Images

###### Densification score map

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

Simple SceneComplex Scene

[Figure 431]

# of Gaussians: 55K

# of Gaussians: 26K

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

Gaussian Param Head

[Figure 439]

[Figure 440]

[Figure 441]

###### … …

[Figure 442]

# of Gaussians: 53K

# of Gaussians: 22K

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

# of Gaussians: 31K

# of Gaussians: 10K

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

# of Gaussians: 17K

# of Gaussians: 6K

Fig. 4: Calculating LscoreG . We convert the view-space rendering

Fig. 5: Effect of the threshold τ across scene complexity. For a fixed τ, complex scenes yield higher densification scores and allocate more Gaussians, while simpler scenes allocate fewer.

gradient vg for each gg ∈ G into a densification score dg and learn the network to predict it.

These previous works [25,57] periodically densify 3D Gaussians during training. In AbsGS [57], for a set of predicted Gaussian primitives G = {gg}N

g=1, whether to densify the Gaussian gg is guided by the homodirectional target view-space positional gradient of the Gaussian, which can be obtained by backpropagating the rendering loss. The rendering loss Lrender is calculated by the weighted sum of MSE and LPIPS loss between the predicted target image ˆItgt and the ground-truth target image Itgt. With this loss, we can calculate the homodirectional view-space positional gradient vg:

G

 

 . (5)

∂Lrenderj ∂µ¯g,x

∂Lrenderj ∂µ¯g,y

m

m

vg =

,

j=1

j=1

Here (µ¯g,x,µ¯g,y) denotes the center of Gaussian gg after projection onto the

- 2D image plane of the target view, Lrenderj denotes the rendering loss computed by the j-th pixel of the image ˆItgt, and m is the total number of pixels that Gaussian gg participates in rendering of ˆItgt. A large value of vg indicates that the Gaussian gg significantly affects the rendering loss, implying that the corresponding region is underrepresented. AbsGS [57] empirically shows that assigning more Gaussians based on the norm of this gradient improves the fidelity of the representation. Therefore, this gradient-based value is a suitable signal for our spatially adaptive Gaussian allocation pipeline. However, feed-forward
- 3DGS predicts 3D Gaussian primitives in a single forward pass without iterative scene optimization, and ground-truth images are not available at inference time. As a result, the gradient vg cannot be utilized directly at inference time.

To overcome this limitation, we instead learn to predict dˆg from the gradient vg. Here, dˆg denotes the predicted densification score associated with the spatial region occupied by Gaussian gg. During the training process, using the 3D Gaussian representation G predicted by input context images {Ictxi }N

i=1 , we can compute the rendering loss Lrender(ˆItgt,Itgt) and the homodirectional view-space positional gradient vg for each Gaussian gg ∈ G, as explained above. Based on the accumulated gradient vg, we define the supervision signal dg for the densification as a log-scaled value of the ℓ2 norm of the gradient: dg = log 1 + 104 · ∥vg∥2 . Then, the predicted densification score dˆg for gg is trained to match dg via the following ℓ1 loss:

ctx

LscoreG = Eg

g∈G ∥dˆg − dg∥1 . (6)

The schematic illustration of this process is given in Fig. 4. We use a novel view as a target view during training, allowing the model to learn densification scores that generalize better across viewpoints. Since the densification score is learned from gradient signals, it serves as an absolute and comparable criterion across different scenes, rather than a purely relative ranking within a single scene. Consequently, the threshold τ serves as a control knob for reconstruction fidelity, determining the level of detail preserved in the final representation. As shown in Fig. 5, under a fixed τ, a complex scene generally produces higher densification scores, resulting in higher Gaussian allocation. In contrast, a simple scene yields lower scores and consequently fewer Gaussians.

Training with Novel Views. Training the model by supervising the final 3D Gaussian representation using the input context views, rather than novel views, can lead to trivial reconstruction and potential overfitting, as the model may simply memorize the observed views instead of learning a representation that generalizes across viewpoints. To address this issue, unlike the previous feedforward approach [23] that supervises reconstruction of input context views, we instead use a novel view as the target image during training. However, directly using the ground-truth camera parameters of the target view is problematic because the camera coordinate system predicted by the model may differ from the ground-truth coordinate frame. To resolve this mismatch, we align the groundtruth target camera into the predicted coordinate system before rendering the target view. Specifically, let Tctxn

###### and Tctxn

###### denote the ground-truth poses of the two context views closest to the target view, and Tˆctxn

1

2

###### and Tˆctxn

denote their corresponding predicted poses. We estimate a similarity transformation matrix A ∈ Sim(3) that maps poses from the ground-truth coordinate frame to the predicted one. The context view n1 closest to the target view is used as the anchor reference for rotation and translation alignment, while the scale is estimated from the relative translation between the two context views n1 and n2 closest to the target view. The ground-truth target pose Ttgt is then transformed using the matrix A to obtain the aligned target pose, Tˆtgt = ATtgt. For the intrinsic parameters f, we adjust the focal length of the target view using the ratio between the predicted and ground-truth focal lengths of the nearest context view, fˆtgt = fˆnctx

1

2

/fnctx

###### · ftgt. This simple yet carefully designed alignment procedure

1

1

ensures that the aligned target view parameters (Tˆtgt,Kˆ tgt) are geometrically consistent with the coordinate system predicted by the model, enabling stable and reliable novel view training.

Overall Training Pipeline. During training, we randomly sample a threshold τ and predict the corresponding Gaussian set Gτ, which is supervised using the rendering loss Lrender. In addition, we also apply the rendering loss to intermediate Gaussian representations Gl ∈ RdG×NctxHlWl, which are constructed using only the Gaussians predicted at each level l. The densification score loss Lscore is computed using the densification scores derived from the multi-level Gaussian representations {Gl}Ll=1−1. The loss for camera parameter estimation, Lcamera, is learned in the same way as VGGT [50]. We further introduce a scene-scale regularization loss Lscene, which normalizes the average distance of Gaussian centers from the origin to 1, i.e., Lscene = |G| 1 g

g∈G ∥µg∥2 − 1 . This regularization is applied independently to the representation of each level. Without this regularization, training can become unstable. Finally, the total training objective is defined as the weighted sum of Lrender, Lscore, Lcamera, and Lscene.

#### 4 Experiments

Datasets. We train F4Splat on the large-scale RealEstate10K (RE10K) [65] and ACID [33] datasets, following prior works for train/test splits. For two-view evaluation, we adopt the same test split as prior feed-forward methods [4,8,32, 54,56]. For multi-view evaluation, we use the scene categorization provided by NoPoSplat [56] and select input pairs with small overlap. Additional input views are sampled between the selected pair without duplication to match the target number of input views (8, 16, and 24 views).

Implementation Details. We use three levels of multi-resolution feature maps (L = 3) in all experiments and choose the finest level to match the input image resolution, (HL,WL)=(H,W). We initialize our model from pretrained VGGT weights. We use a learning rate of 2 × 10−4 for most modules, while freezing the patch embedding weights of the geometry backbone and training the remaining backbone parameters with a smaller learning rate of 2 × 10−5. For multi-view training, we train on RE10K. At each iteration, each GPU independently samples the number of input views from {2,3,4,6,12,24} and selects context images accordingly. We additionally sample the same number of target novel views for supervision. To keep the total number of images processed per iteration constant, we use a dynamic batch size that is inversely proportional to the number of context images. The multi-view model is trained for 15,000 iterations. For two-view training, separate models are trained on RE10K and ACID, following the training configuration of NoPoSplat [56]. Each model is trained for 18,750 iterations with a total batch size of 128. We set the weight of MSE and LPIPS loss as 1 and 0.05 for Lrender. For the final total loss, we fix the weight of Lrender, Lscore, Lcamera, and Lscene as 1.0, 10−4, 10.0, and 10−2 respectively, in all experiments. All experiments are conducted on eight NVIDIA H200 GPUs, and each training run takes approximately 15 hours.

- Table 1: Novel view synthesis performance on RE10K [65] under different numbers of input views. We report LPIPS/SSIM/PSNR for 8, 16, and 24 input views. Best and second-best results are highlighted in bold and underlined, respectively. τ+ and τ− denote the high- and low-threshold variants, respectively.

Method

8 views 16 views 24 views #GS↓ LPIPS↓ SSIM↑ PSNR↑ #GS↓ LPIPS↓ SSIM↑ PSNR↑ #GS↓ LPIPS↓ SSIM↑ PSNR↑

Pose-Free

NoPoSplat 524K 0.213 0.756 22.31 1049K 0.252 0.713 21.11 1573K 0.275 0.691 20.49 VicaSplat 524K 0.241 0.713 21.18 1049K 0.384 0.596 17.56 1573K 0.443 0.546 16.13 SPFSplat 524K 0.142 0.849 25.66 1049K 0.137 0.855 25.88 1573K 0.139 0.853 25.78

Uncalibrated

VicaSplat 524K 0.258 0.686 20.77 1049K 0.417 0.556 16.78 1573K 0.470 0.517 15.58 AnySplat 447K 0.167 0.819 24.07 820K 0.148 0.842 25.10 1142K 0.143 0.849 25.40 F4Splatτ+ 105K 0.142 0.847 25.26 210K 0.130 0.860 25.75 315K 0.128 0.862 25.85 F4Splatτ− 447K 0.131 0.859 25.64 820K 0.120 0.869 26.10 1142K 0.119 0.870 26.18

- Table 2: Generalization to unseen datasets. Models are trained on RE10K [65] and evaluated on the unseen ACID dataset [33]. We compare reconstruction quality under different numbers of input views (8, 16, and 24).

8 views 16 views 24 views #GS↓ LPIPS↓ SSIM↑ PSNR↑ #GS↓ LPIPS↓ SSIM↑ PSNR↑ #GS↓ LPIPS↓ SSIM↑ PSNR↑

Method

NoPoSplat 524K 0.268 0.672 22.84 1049K 0.294 0.644 22.14 1573K 0.307 0.630 21.78 VicaSplat 524K 0.271 0.656 22.57 1049K 0.327 0.609 21.20 1573K 0.356 0.581 20.37 SPFSplat 524K 0.215 0.736 24.89 1049K 0.214 0.743 25.07 1573K 0.218 0.744 25.00

Pose-Free

VicaSplat 524K 0.315 0.588 21.31 1049K 0.377 0.547 19.82 1573K 0.399 0.529 19.22 AnySplat 481K 0.248 0.696 23.30 906K 0.236 0.720 23.88 1289K 0.234 0.727 24.04 F4Splatτ+ 52K 0.239 0.713 24.28 105K 0.230 0.726 24.54 315K 0.216 0.741 24.72 F4Splatτ− 481K 0.204 0.744 24.83 906K 0.201 0.753 25.01 1289K 0.203 0.752 24.88

Uncalibrated

Baselines and Evaluation Metrics. For multi-view evaluation, we compare F4Splat with recent feed-forward 3DGS methods under two settings: uncalibrated methods that take only images as input without camera poses or intrinsics, and pose-free methods that do not require camera poses but assume known intrinsics. For the uncalibrated setting, we compare against VicaSplat [32] and AnySplat [23]. For fair comparison, we retrain AnySplat under the same training setup as ours, which requires approximately 27 hours on eight NVIDIA H200 GPUs. VicaSplat uses the officially released multi-view pretrained weights on the RE10K dataset. For the pose-free setting, we compare against NoPoSplat [56], VicaSplat [32], and SPFSplat [22], using their officially released pretrained models. For two-view evaluation, we compare with pixelSplat [4], MVSplat [8], NoPoSplat [56], VicaSplat [32], and SPFSplat [22]. We also compare baselines that are not a feed-forward 3D Gaussian splatting method, including pixelNeRF [59], DUSt3R [51], MASt3R [30], and CoPoNeRF [21]. We evaluate the performance of novel view synthesis using three standard metrics: LPIPS, SSIM, and PSNR. We also report the number of final Gaussian primitives to enable joint comparison of rendering quality and Gaussian-count efficiency.

##### 4.1 Experimental Results

As shown in Tab. 1 and Tab. 2, our method achieves competitive performance among uncalibrated baselines and remains competitive with pose-free and poserequired methods. When using a similar number of Gaussian primitives as the baselines, our approach achieves strong reconstruction performance. Notably, even when using only 10-28% of the Gaussian primitives, our method still main-

NoPoSplat

VicaSplat

VicaSplat SPFSplat AnySplat Ours GT

(Pose-Free)

(Uncalibrated)

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

16 views24 views8 views

- 13.07/524K 16.38/524K 16.34/524K 16.77/524K 13.91/445K 21.00/105K

15.90/1049K 13.75/1049K 13.43/1049K 13.63/1049K 15.49/830K 17.72/210K

- 14.77/1573K 12.98/1573K 12.58/1573K 22.93/1573K 19.45/1079K 25.42/315K

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

Fig. 6: Qualitative comparisons of novel view synthesis on RE10K [65] The bottom-right corner of each image shows the PSNR of the rendered view and the number of Gaussian primitives used for scene reconstruction. Our method achieves high-quality rendering even with substantially fewer primitives, while consistently outperforming competing approaches.

tains competitive results. These results suggest that our explicit density control enables a compact yet high-fidelity 3D Gaussian representation, leading to improved efficiency without sacrificing reconstruction quality.

Fig. 6 shows qualitative comparisons on RE10K under different input-view settings. Our method produces sharper structures and more faithful scene details compared to existing approaches. Notably, even with substantially fewer Gaussian primitives (24-29%), our approach maintains high rendering quality while reducing blurring artifacts, resulting in visually more accurate renderings.

Tab. 3 demonstrates the novel view synthesis performance on the ACID dataset under the challenging two-view setting. Our method achieves superior performance compared to the uncalibrated baseline. Moreover, it remains competitive with both pose-required and pose-free approaches, including feedforward 3DGS methods and other baselines, demonstrating robust performance even with only two sparse input views.

##### 4.2 Ablation Studies and Analysis

Densification-Score-Guided Allocation. We validate the effectiveness of our learned densification score for Gaussian allocation by comparing it with two alternative strategies, (a) and (b) in Tab. 4. For (a) random-based allocation, we replace the learned densification score with a uniform score, which leads to random Gaussian allocation. For (b) frequency-based allocation, we compute a heuristic frequency score by resizing the input image to each scale level and applying a Sobel filter to emphasize high-frequency (edge) regions. Both alternatives underperform, demonstrating that the learned densification score provides a more effective signal for allocating Gaussians than either random selection or simple frequency-based heuristics. Qualitatively, Fig. 3 further illustrates the behavior of the learned densification score. The predicted maps of the top example (a) assign higher scores to structurally and visually complex regions (e.g.,

###### Table 3: Novel view synthesis performance comparison under 2-view setting. Results on ACID [33] dataset.

Average #GS↓ LPIPS↓ SSIM↑ PSNR↑

Method

pixelNeRF - 0.533 0.561 20.323 pixelSplat 131K 0.195 0.779 25.819 MVSplat 131K 0.196 0.773 25.512

PoseRequired

DUSt3R - 0.447 0.411 16.286 MASt3R - 0.461 0.409 16.179 CoPoNeRF - 0.406 0.606 20.950 NoPoSplat 131K 0.189 0.781 25.961 VicaSplat 131K 0.201 0.757 25.439 SPFSplat 131K 0.176 0.807 26.796

Pose-Free

VicaSplat 131K 0.218 0.726 24.548 F4Splatτ+ 52K 0.188 0.784 26.028 F4Splatτ− 131K 0.176 0.794 26.282

Uncalibrated

Table 4: Ablation Studies. Experiments are conducted with 24 input views, where the Gaussian budget is fixed to 20% of the maximum number of Gaussians. (a)-(b) compare different Gaussian allocation strategies, (c) removes the level-wise Gaussian supervision during training, and (d) removes the scene-scale regularization term.

Variant LPIPS↓ SSIM↑ PSNR↑

- (a) Rand. based allocation 0.194 0.828 24.68
- (b) Freq. based allocation 0.160 0.841 25.36
- (c) w/o level-wise GS train 0.192 0.813 24.25
- (d) w/o scene scale reg. 0.712 0.006 4.82
- (e) Ours 0.143 0.854 25.47

object boundaries, textured areas, and fine details), indicating where increased Gaussian density is likely to be beneficial. Moreover, as in the example below, overlapping areas observed from multiple context images tend to have lower scores, reflecting redundancy-awareness across views. This encourages allocating Gaussians to regions that need more capacity while avoiding redundancy in well-covered areas, improving efficiency and fidelity under a fixed budget.

Level-Wise Gaussian Supervision. Next, we examine the impact of levelwise Gaussian supervision during training. If we supervise only the final Gaussian set Gτ, the sampled threshold τ causes some scale-level Gaussians to be excluded from training in each iteration, resulting in less stable optimization. As shown in Tab. 4, removing level-wise supervision clearly degrades performance (c), confirming the benefit of supervising multi-scale Gaussians during training. Scene-Scale Regularization. Finally, we ablate the scene-scale regularization loss in Tab. 4 (d). Without this term, training becomes highly unstable, and the model fails to learn a meaningful reconstruction, leading to training failure. In the uncalibrated setting, where the model must jointly predict camera parameters and scene geometry, this simple regularizer is crucial for stable optimization.

#### 5 Conclusion

We present F4Splat, a feed-forward 3DGS framework that reconstructs compact representations from sparse, uncalibrated inputs. By introducing a feedforward densification through a densification-score-guided allocation strategy, our method adaptively distributes Gaussians according to spatial complexity and multi-view overlap, enabling explicit control over the final Gaussian budget without retraining. By allocating more primitives to informative regions while avoiding redundancy in simple or overlapping areas, F4Splat produces compact Gaussian representations while achieving high reconstruction fidelity. Experiments demonstrate that our approach achieves competitive or superior novelview synthesis performance compared to prior feed-forward methods while requiring significantly fewer Gaussians, highlighting the effectiveness of spatially adaptive Gaussian allocation for efficient feed-forward 3DGS reconstruction.

## Supplementary Material

F4Splat: Feed-Forward Predictive Densification for Feed-Forward 3D Gaussian Splatting

- S1 Budget Matching Algorithm

To efficiently determine the threshold τN¯G that matches a target Gaussian budget N¯G, we precompute a threshold–budget lookup table, τ˜ and N˜ G, from the predicted multi-level densification score maps {{Dˆ li}N

i=1 }Ll=1−1 using Algorithm 1.

ctx

##### Algorithm 1 Precomputing the threshold–budget lookup table

Require: Multi-level score maps {{Dˆ li}Ni=1ctx}Ll=1−1 for all views and levels Ensure: A sorted threshold list τ˜ and the corresponding Gaussian-count delta list N˜ G

- 1: Initialize all ∆Nil to 3 for all values with the same resolution as Dˆ li.
- 2: for l = L − 1, L − 2, . . . , 2 do
- 3: {Ali}Ni=1ctx ← 1{Dˆl i≥Up(Dˆ li−1;2)}

Nctx i=1

- 4: {∆Nil−1}Ni=1ctx ← ∆Nil−1 + SumPool2×2 ∆Nil ⊙ Ali Ni=1ctx
- 5: {∆Nil}Ni=1ctx ← ∆Nil ⊙ (1 − Ali) Ni=1ctx
- 6: end for
- 7: τ = Concat Flatten(Dˆ li) i=1,...,Nctx; l=1,...,L−1
- 8: ∆NG = Concat Flatten(∆Nil) i=1,...,N

ctx; l=1,...,L−1

- 9: π = argsort(τ) ▷ descending sorting permutation
- 10: τ˜ = τ[π]
- 11: ∆NG = ∆NG[π]
- 12: N˜ G = NctxH1W1 · 1 + Cumsum (∆NG) ▷ sorted in ascending order, unlike τ˜
- 13: return τ˜, N˜ G

Let N˜ G = [N˜ G]1,...,[N˜ G]K denote the Gaussian-count lookup table sorted according to the threshold list τ˜. Under the assumption that all densification score values are unique, the difference between two adjacent entries is upperbounded by 4L−1 − 1:

[N˜ G]k+1 − [N˜ G]k ≤ 4L−1 − 1.

This follows because lowering the threshold activates exactly one new score value, which changes the allocation at only one spatial position. The largest possible increase occurs when the activated position corresponds to the coarsest valid level and represents all unresolved descendants across finer levels. Since each

Table S1: Camera pose estimation accuracy on RE10K and ACID. Following prior works [11,46], we quantify accuracy via the AUC of the cumulative rotation error curve, using rotation thresholds of 5◦, 10◦, and 20◦.

###### RE10K ACID

Method 5◦ ↑ 10◦ ↑ 20◦ ↑ 5◦ ↑ 10◦ ↑ 20◦ ↑ DUSt3R 0.301 0.495 0.657 0.166 0.304 0.437 MASt3R 0.372 0.561 0.709 0.234 0.396 0.541 VGGT 0.335 0.531 0.696 0.219 0.399 0.576 AnySplat (RE10K) 0.315 0.522 0.694 0.265 0.445 0.608 F4Splat (RE10K) 0.541 0.704 0.814 0.262 0.449 0.618

Table S2: Computational overhead of Spatially Adaptive Gaussian Allocation. We compare the uniform pixel-to-Gaussian allocation with our adaptive allocation. For adaptive allocation, the total number of Gaussians is reduced by 20% compared to the uniform allocation.

Allocation Peak allocated Inference strategy VRAM (GB) time (s)

Uniform 8.699 0.440 Adaptive 8.855 (+1.8%) 0.488 (+10.1%)

finest-level position contributes 3 Gaussians, the maximum increment becomes

L−2

4j = 4L−1 − 1.

3

j=0

Therefore, for any target budget N¯G, there always exists an index k⋆ such that [N˜ G]k⋆ ≤ N¯G < [N˜ G]k⋆ + (4L−1 − 1),

which implies that the threshold τN¯G = τ˜k⋆ satisfies Eq. (4) of the main paper. Ultimately, we can efficiently obtain the threshold τN¯G via Algorithm 2.

##### Algorithm 2 Finding the budget-matching threshold

Require: τ˜, N˜ G, N¯G Ensure: τN¯G satisfying the target budget N¯G

- 1: k⋆ ← max k | [N˜ G]k ≤ N¯G ▷ found by binary search in O(log |N˜ G|) time
- 2: τN¯G ← [τ˜]k⋆
- 3: return τN¯G

#### S2 Additional Results

- Video 1. To provide further intuition, we additionally provide videos on the project page that visualize the spatial distribution of allocated Gaussians under varying Gaussian budgets for the samples shown in Fig. 1 and Fig. 3 of the main paper. In Fig. 3 of the main paper, for clearer visualization, only the locations allocated at the finest level (l = 3) are highlighted in red. In the supplementary videos, by contrast, all allocated locations are shown in red. These videos illustrate that, as the Gaussian budget increases, Gaussians are adaptively

- allocated to fine-detail regions, and redundant allocations are also minimized in regions with overlap among the context images.
- Video 2. Video 2 on the project page provides comparisons with AnySplat [23] under different Gaussian budgets. Both models use weights trained on the multiview RE10K setting, and the videos are generated by taking two input views from the RE10K and ACID datasets. For AnySplat, we performed inference while progressively doubling the voxel size from its default value. We then used the number of Gaussians produced by AnySplat as the Gaussian budget for F4Splat. The first row shows the rendered RGB quality, and the second row presents the corresponding depth maps. In the third row, allocated Gaussian locations are highlighted in red. While AnySplat distributes Gaussians uniformly over the scene, F4Splat allocates them in a spatially adaptive manner. Ultimately, F4Splat maintains high fidelity even with fewer Gaussians. Fig. S1. We also provide additional qualitative comparisons for the multi-view experiments in Fig. S1.

#### S3 Additional Experiments

Relative Pose Estimation. Following prior work [11,46], we additionally evaluate the quality of the predicted camera poses using the relative rotation accuracy metric. Specifically, we compare DUSt3R [51], MASt3R [30], and VGGT [50] against AnySplat and F4Splat, which are trained with multi-view supervision on RE10K. We construct the evaluation sets on RE10K and ACID following the same protocol as NoPoSplat [56], but, unlike NoPoSplat, we directly use the camera parameters predicted by our model without any test-time optimization. As shown in Tab. S1, our model, trained on the RE10K dataset, achieves higher accuracy across all thresholds. Furthermore, despite not being trained on the ACID dataset [33], F4Splat generalizes effectively to this unseen dataset, achieving the best performance overall while outperforming all baselines at 10° and 20° and remaining competitive at 5°. These results indicate that, although our geometry backbone is initialized from VGGT, the proposed architecture and training scheme further refine and extend its geometric reasoning, leading to consistently stronger pose estimation performance.

Computational cost of Spatially Adaptive Gaussian Allocation. Tab. S2 reports the computational overhead of introducing Spatially Adaptive Gaussian Allocation. The peak allocated VRAM increases by only 1.8%, and the inference time increases by 10.1%. As discussed in the Multi-Scale Prediction paragraph of Sec. 3.2 in the main paper, the proposed design predicts multi-scale maps efficiently, making adaptive Gaussian allocation possible with minimal extra cost.

NoPoSplat

VicaSplat

VicaSplat SPFSplat AnySplat Ours GT

(Pose-Free)

(Uncalibrated)

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

23.29/524K

23.14/524K

23.40/524K

25.68/524K

20.56/432K

28.69/105K

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

8views16views24views

- 13.07/524K

15.35/524K

12.45/1049K

15.90/1049K

17.81/1049K

- 14.77/1573K

16.38/524K

16.34/524K

- 16.77/524K

14.77/524K

- 17.20/1049K

13.91/445K

- 21.00/105K

20.50/105K

- 22.83/210K

17.72/210K

- 24.64/210K

- 25.42/315K

- 23.27/315K

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

14.77/524K

15.15/524K

13.06/413K

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

10.13/1049K

9.39/1049K

- 16.95/837K

15.49/830K

19.05/1049K

19.45/1079K

- 17.22/1186K

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

- 13.75/1049K

- 14.22/1049K

13.43/1049K

13.63/1049K

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

13.49/1049K

19.99/1049K

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

12.98/1573K

- 12.58/1573K

9.68/1573K

- 13.90/1573K

22.93/1573K

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

12.74/1573K

10.04/1573K

17.38/1573K

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

20.00/1573K

14.26/1573K

25.25/1573K

25.31/1020K

28.53/315K

###### Fig. S1: Additional qualitative comparisons of novel view synthesis on RE10K [65] dataset.

#### S4 Additional Details

For all experiments, we freeze only the image tokenizer, DINOv2 [40], while fine-tuning all other components, including the frame-wise and global attention layers, camera and register tokens, and the point and camera prediction heads. We use a base learning rate of 2×10−4 for all experiments, and apply a learning rate scaled by 1/10 to the fine-tuned components inherited from the pretrained model. For evaluation on RE10K [65] and ACID [33], we follow NoPoSplat and apply test-time camera pose optimization. The initial target camera parameters are obtained using the target-aligned projection formulation described in Sec. 3.3 of the main paper. For the ablation studies, we adopt a 5× reduced-iteration training schedule, where all iteration-related hyperparameters are reduced by a factor of 5. For the AnySplat results in Fig. 1 (a), we control the number of Gaussians by adjusting the voxel size. As the voxel size increases, we proportionally enlarge the scale of the output Gaussians so that their spatial coverage increases accordingly.

#### References

- 1. Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P.: Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In: ICCV (2021)
- 2. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In: CVPR (2022)
- 3. Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P.: Zip-nerf: Anti-aliased grid-based neural radiance fields. In: ICCV (2023)
- 4. Charatan, D., Li, S.L., Tagliasacchi, A., Sitzmann, V.: pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In: CVPR (2024)
- 5. Chen, A., Xu, Z., Geiger, A., Yu, J., Su, H.: Tensorf: Tensorial radiance fields. In: ECCV (2022)
- 6. Chen, Y., Wu, Q., Lin, W., Harandi, M., Cai, J.: Hac: Hash-grid assisted context for 3d gaussian splatting compression. In: ECCV (2024)
- 7. Chen, Y., Jiang, J., Jiang, K., Tang, X., Li, Z., Liu, X., Nie, Y.: Dashgaussian: Optimizing 3d gaussian splatting in 200 seconds. In: CVPR (2025)
- 8. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cham, T.J., Cai, J.: Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In: ECCV (2024)
- 9. Chen, Z., Yang, J., Yang, H.: Pref3r: Pose-free feed-forward 3d gaussian splatting from variable-length image sequence. arXiv preprint arXiv:2411.16877 (2024)
- 10. Cheng, K., Long, X., Yang, K., Yao, Y., Yin, W., Ma, Y., Wang, W., Chen, X.: Gaussianpro: 3d gaussian splatting with progressive propagation. In: ICML (2024)
- 11. Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., Felsberg, M.: Roma: Robust dense feature matching. In: CVPR (2024)
- 12. Fan, Z., Wang, K., Wen, K., Zhu, Z., Xu, D., Wang, Z., et al.: Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. NeurIPS

(2024)

- 13. Fang, G., Wang, B.: Mini-splatting: Representing scenes with a constrained number of gaussians. In: ECCV (2024)
- 14. Feng, G., Chen, S., Fu, R., Liao, Z., Wang, Y., Liu, T., Hu, B., Xu, L., Pei, Z., Li, H., et al.: Flashgs: Efficient 3d gaussian splatting for large-scale and high-resolution rendering. In: CVPR (2025)
- 15. Fridovich-Keil, S., Yu, A., Tancik, M., Chen, Q., Recht, B., Kanazawa, A.: Plenoxels: Radiance fields without neural networks. In: CVPR (2022)
- 16. Girish, S., Gupta, K., Shrivastava, A.: Eagles: Efficient accelerated 3d gaussians with lightweight encodings. In: ECCV (2024)
- 17. Grubert, G., Barthel, F.T., Hilsmann, A., Eisert, P.: Improving adaptive density control for 3d gaussian splatting. In: VISIGRAPP (2025)
- 18. He, Z., Xiao, Z., Chan, K.C., Zuo, Y., Xiao, J., Lam, K.M.: See in detail: Enhancing sparse-view 3d gaussian splatting with local depth and semantic regularization. In: ICASSP (2025)
- 19. Höllein, L., Božič, A., Zollhöfer, M., Nießner, M.: 3dgs-lm: Faster gaussian-splatting optimization with levenberg-marquardt. In: ICCV (2025)
- 20. Hong, S., Jung, J., Shin, H., Han, J., Yang, J., Luo, C., Kim, S.: Pf3plat: Pose-free feed-forward 3d gaussian splatting. In: ICML (2025)
- 21. Hong, S., Jung, J., Shin, H., Yang, J., Kim, S., Luo, C.: Unifying correspondence, pose and nerf for pose-free novel view synthesis from stereo pairs. arXiv preprint arXiv:2312.07246 (2023)

- 22. Huang, R., Mikolajczyk, K.: No pose at all: Self-supervised pose-free 3d gaussian splatting from sparse views. In: ICCV (2025)
- 23. Jiang, L., Mao, Y., Xu, L., Lu, T., Ren, K., Jin, Y., Xu, X., Yu, M., Pang, J., Zhao, F., et al.: Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. ACM Trans. Grphics (2025)
- 24. Kang, G., Yoo, J., Park, J., Nam, S., Im, H., Shin, S., Kim, S., Park, E.: Selfsplat: Pose-free and 3d prior-free generalizable 3d gaussian splatting. In: CVPR (2025)
- 25. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G.: 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graphics (2023)
- 26. Kheradmand, S., Rebain, D., Sharma, G., Sun, W., Tseng, Y.C., Isack, H., Kar, A., Tagliasacchi, A., Yi, K.M.: 3d gaussian splatting as markov chain monte carlo. NeruIPS (2024)
- 27. Kim, S., Lee, K., Lee, Y.: Color-cued efficient densification method for 3d gaussian splatting. In: CVPR Workshops (2024)
- 28. Kong, H., Yang, X., Wang, X.: Generative sparse-view gaussian splatting. In: CVPR (2025)
- 29. Lee, J.C., Rho, D., Sun, X., Ko, J.H., Park, E.: Compact 3d gaussian representation for radiance field. In: CVPR (2024)
- 30. Leroy, V., Cabon, Y., Revaud, J.: Grounding image matching in 3d with mast3r. In: ECCV (2024)
- 31. Li, J., Zhang, J., Bai, X., Zheng, J., Ning, X., Zhou, J., Gu, L.: Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In: CVPR (2024)
- 32. Li, Z., Dong, C., Chen, Y., Huang, Z., Liu, P.: Vicasplat: A single run is all you need for 3d gaussian splatting and camera estimation from unposed video frames. arXiv preprint arXiv:2503.10286 (2025)
- 33. Liu, A., Tucker, R., Jampani, V., Makadia, A., Snavely, N., Kanazawa, A.: Infinite nature: Perpetual view generation of natural scenes from a single image. In: ICCV

(2021)

- 34. Mallick, S.S., Goel, R., Kerbl, B., Steinberger, M., Carrasco, F.V., De La Torre, F.: Taming 3dgs: High-quality radiance fields with limited resources. In: SIGGRAPH Asia (2024)
- 35. Martin-Brualla, R., Radwan, N., Sajjadi, M.S., Barron, J.T., Dosovitskiy, A., Duckworth, D.: Nerf in the wild: Neural radiance fields for unconstrained photo collections. In: CVPR (2021)
- 36. Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., Geiger, A.: Occupancy networks: Learning 3d reconstruction in function space. In: CVPR (2019)
- 37. Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R.: Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV

(2020)

- 38. Müller, T., Evans, A., Schied, C., Keller, A.: Instant neural graphics primitives with a multiresolution hash encoding. ACM Trans. Graphics (2022)
- 39. Niedermayr, S., Stumpfegger, J., Westermann, R.: Compressed 3d gaussian splatting for accelerated novel view synthesis. In: CVPR (2024)
- 40. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 41. Papantonakis, P., Kopanas, G., Kerbl, B., Lanvin, A., Drettakis, G.: Reducing the memory footprint of 3d gaussian splatting. Proc. ACM Comput. Graphics Comput. Syst. (2024)

- 42. Park, J.J., Florence, P., Straub, J., Newcombe, R., Lovegrove, S.: Deepsdf: Learning continuous signed distance functions for shape representation. In: CVPR (2019)
- 43. Park, K., Sinha, U., Barron, J.T., Bouaziz, S., Goldman, D.B., Seitz, S.M., MartinBrualla, R.: Nerfies: Deformable neural radiance fields. In: ICCV (2021)
- 44. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. In: ICCV (2021)
- 45. Rota Bulò, S., Porzi, L., Kontschieder, P.: Revising densification in gaussian splatting. In: ECCV (2024)
- 46. Sarlin, P.E., DeTone, D., Malisiewicz, T., Rabinovich, A.: Superglue: Learning feature matching with graph neural networks. In: CVPR (2020)
- 47. Smart, B., Zheng, C., Laina, I., Prisacariu, V.A.: Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912 (2024)
- 48. Wang, C., Ma, G., Xue, Y., Lao, Y.: Faster and better 3d splatting via group training. In: ICCV (2025)
- 49. Wang, H., Zhu, H., He, T., Feng, R., Deng, J., Bian, J., Chen, Z.: End-to-end rate-distortion optimized 3d gaussian representation. In: ECCV (2024)
- 50. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: CVPR (2025)
- 51. Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B., Revaud, J.: Dust3r: Geometric 3d vision made easy. In: CVPR (2024)
- 52. Wang, Y., Huang, T., Chen, H., Lee, G.H.: Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. NeurIPS (2024)
- 53. Xiong, H., Muttukuru, S., Upadhyay, R., Chari, P., Kadambi, A.: Sparsegs: Real-time 360° sparse view synthesis using gaussian splatting. arXiv preprint arXiv:2312.00206 (2023)
- 54. Xu, H., Peng, S., Wang, F., Blum, H., Barath, D., Geiger, A., Pollefeys, M.: Depthsplat: Connecting gaussian splatting and depth. In: CVPR (2025)
- 55. Yang, R., Zhu, Z., Jiang, Z., Ye, B., Chen, X., Zhang, Y., Chen, Y., Zhao, J., Zhao, H.: Spectrally pruned gaussian fields with neural compensation. arXiv preprint arXiv:2405.00676 (2024)
- 56. Ye, B., Liu, S., Xu, H., Li, X., Pollefeys, M., Yang, M.H., Peng, S.: No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. In: ICLR (2025)
- 57. Ye, Z., Li, W., Liu, S., Qiao, P., Dou, Y.: Absgs: Recovering fine details in 3d gaussian splatting. In: ACMMM (2024)
- 58. Yu, A., Li, R., Tancik, M., Li, H., Ng, R., Kanazawa, A.: Plenoctrees for real-time rendering of neural radiance fields. In: ICCV (2021)
- 59. Yu, A., Ye, V., Tancik, M., Kanazawa, A.: pixelnerf: Neural radiance fields from one or few images. In: CVPR (2021)
- 60. Zhang, J., Zhan, F., Xu, M., Lu, S., Xing, E.: Fregs: 3d gaussian splatting with progressive frequency regularization. In: CVPR (2024)
- 61. Zhang, J., Li, J., Yu, X., Huang, L., Gu, L., Zheng, J., Bai, X.: Cor-gs: sparse-view 3d gaussian splatting via co-regularization. In: ECCV (2024)
- 62. Zhang, S., Wang, J., Xu, Y., Xue, N., Rupprecht, C., Zhou, X., Shen, Y., Wetzstein, G.: Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In: CVPR (2025)
- 63. Zhang, Z., Hu, W., Lao, Y., He, T., Zhao, H.: Pixel-gs: Density control with pixelaware gradient for 3d gaussian splatting. In: ECCV (2024)
- 64. Zhao, H., Weng, H., Lu, D., Li, A., Li, J., Panda, A., Xie, S.: On scaling up 3d gaussian splatting training. In: ICLR (2025)

- 65. Zhou, T., Tucker, R., Flynn, J., Fyffe, G., Snavely, N.: Stereo magnification: Learning view synthesis using multiplane images. ACM Trans. Graphics (2018)
- 66. Zhu, Z., Fan, Z., Jiang, Y., Wang, Z.: Fsgs: Real-time few-shot view synthesis using gaussian splatting. In: ECCV (2024)

