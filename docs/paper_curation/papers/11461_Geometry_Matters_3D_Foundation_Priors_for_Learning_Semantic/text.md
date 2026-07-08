# arXiv:2605.30093v1[cs.CV]28May2026

## Geometry Matters: 3D Foundation Priors for Learning Semantic Correspondence

Artur Jesslen1 Olaf Dünkel2 Adam Kortylewski3 1University of Freiburg, Germany 2Max Planck Institute for Informatics, Saarland Informatics Campus, Germany 3CISPA Helmholtz Center for Information Security, Germany

##### Abstract

Foundation features from self-supervised vision models and text-to-image diffusion models have proven effective for semantic correspondence estimation. However, because these features are learned primarily from 2D image objectives, they lack explicit 3D awareness and often confuse symmetric object sides, repeated parts, and visually similar structures that are distinct in 3D. We introduce a 3D-aware post-training framework that goes beyond available 2D foundation features by incorporating priors from 3D foundation models. Given an image, our method uses SAM3D to estimate object geometry and pose, and refines the pose through render-and-compare optimization. Subsequently, we render PartField descriptors from the reconstructed geometry into the image plane based on the estimated object pose. The resulting geometry-aware feature maps complement DINO and Stable Diffusion features, while geodesic distances on the reconstructed shapes enable reliable filtering of candidate correspondences. We use the filtered matches as supervision to train a lightweight adapter on top of DINO and Stable Diffusion for semantic correspondence. In contrast to prior post-training approaches that require pose annotations and rely on coarse spherical geometry, our method automatically obtains instance-specific 3D structure and uses it to guide correspondence learning. Experiments show that our approach improves semantic correspondence over the prior methods while reducing manual geometric supervision. Code and model can be found at /GenIntel/3D-SC.

##### 1 Introduction

Semantic correspondence aims to establish matches between semantically equivalent object parts across different images and is a fundamental problem in visual recognition, with many applications like in vision [38] or robotics [46]. Unlike low-level image matching, semantic correspondence requires robustness to changes in appearance, viewpoint, articulation, intra-class shape variation, and background clutter. As a result, it remains challenging to match object parts that are visually different but semantically equivalent, or visually similar but semantically distinct.

Recent progress has been driven by foundation features, with self-supervised vision transformers (DINOv2) and text-to-image diffusion models (Stable Diffusion) producing representations that transfer surprisingly well to dense semantic matching [4, 2, 28, 30, 36]. Their fusion has become a strong zero-shot baseline on benchmarks such as SPair-71k, PF-PASCAL, and TSS [26, 11, 35, 43], with noisy DINOv2 features complemented by the smoother spatial cues of diffusion models. However, these features are learned from 2D objectives and lack explicit 3D awareness, leading to systematic failure modes [24, 8]. For symmetric objects, such as cars, buses, and animals, 2D features may confuse left and right object sides [44]. For objects with repeated parts, such as wheels, legs, windows, or chair legs, visually similar regions may collapse to nearly identical feature representations despite corresponding to different object parts (see the nearest-neighbor visualization

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

(a) SD+DINO. (b) SD+DINO

(c) SD+DINO+Partfield + Geodesic Filtering.

+ Geodesic Filtering.

- Figure 1: 3D foundation priors improve both candidate generation and filtering of semantic correspondences. Existing zero-shot pipelines based on SD+DINO (a) suffer from left–right and repeated-part confusion, producing many incorrect matches. Adding our geodesic filter (b) removes wrong matches but is bottlenecked by feature quality, often leaving few surviving correspondences. Adding PartField features (c) yields dense and accurate correspondences even with large pose changes.

in Figure 1a). More generally, 2D features cannot reliably distinguish structures that are visually similar yet geometrically distinct.

Several recent methods address these ambiguities by injecting a weak 3D prior to guide feature learning and correspondence filtering [24, 8]. While effective, both approaches require human pose annotations and approximate object geometry with a coarse spherical proxy, which cannot represent the geometric structure of an actual instance. Therefore, some finer distinctions between symmetric or articulated parts are not captured. The reliance on manual pose annotations also limits scalability, as extending to new object categories requires additional labeling effort.

In this paper, we propose a 3D-aware post-training framework that incorporates priors from 3D foundation models without requiring manual pose annotations. Given an image, we use SAM3D to estimate object geometry and pose [31], then refine the pose via a render-and-compare optimization that aligns rendered geometry with the observed object. These refined predictions allow us to run PartField [21] on the reconstructed shape and render geometry-aware descriptors back into the image plane, complementing DINOv2 and Stable Diffusion features in two ways. First, rendered PartField descriptors disambiguate symmetric structures and repeated parts (e.g., front vs. rear wheels) that 2D features alone cannot separate. Second, geodesic distances on the 3D reconstructed shape enable more reliable filtering of candidate correspondences than coarse canonical-sphere proxies, yielding higher-quality pseudo-labels for a lightweight adapter trained on top of DINOv2 and Stable Diffusion features. Experiments on standard benchmarks show consistent improvements over prior approaches with less manual supervision. In summary, we make the following contributions:

- (i) a 3D-aware post-training framework for semantic correspondence that incorporates priors from 3D foundation models without human pose annotations;
- (ii) a render-and-compare pose refinement that allows rendering PartField features into the image plane, yielding geometry-aware features complementing DINOv2 and Stable Diffusion features;
- (iii) a pseudo-label filtering scheme based on geodesic distances on the estimated 3D shapes, providing higher-quality supervision than coarse spherical geometry; and
- (iv) geometry-aware refined features that achieve state-of-the-art semantic correspondence over prior methods with reduced manual supervision.

##### 2 Related Work

Semantic correspondence with foundation features. Semantic correspondence aims to match semantically equivalent parts across object instances, which is substantially harder than low-level image

matching because appearance, shape, pose, articulation, and visibility all vary. Early approaches relied on hand-crafted descriptors and learned matching networks [22, 20, 11, 41], and because dense annotations are costly, later work explored weak supervision, cycle-consistency losses, and pseudolabel expansion from sparse labels [45, 16, 18, 14]. Recent progress has shifted to foundation features: self-supervised vision transformers such as DINO and DINOv2 encode transferable semantic concepts [4, 2, 28], while text-to-image diffusion features provide complementary spatial and semantic cues [30, 13, 34, 23, 19]. Their fusion has become a strong zero-shot baseline [43], and distillation or adapter-based refinement further improves them when supervision is available [44, 10, 40]. However, since these features are learned from images, they remain prone to geometry-sensitive failures such as left-right confusion, front-back ambiguity, and repeated parts [44, 24, 8, 25]. Our work follows the weakly supervised, foundation-feature direction, but uses reconstructed 3D geometry to generate and filter dense pseudo-labels rather than relying on manual keypoint annotations.

Geometric priors and 3D-aware features. A complementary line of work introduces geometric structure to disambiguate the failures of purely image-based correspondence. CAD-based cycle consistency and canonical surface mappings link image pixels to a shared object surface [45, 17, 27], while category-level templates, atlases, and learned 3D representations capture correspondences via a shared geometric frame [15, 32, 33, 37, 5]. These methods show the value of 3D structure but typically require mesh templates, precise pose, or category-level reconstruction pipelines. Closer to our setting, Spherical Maps inject a weak 3D prior by mapping image features to a category-conditioned sphere with viewpoint supervision [24], and DIY-SC produces pseudo-labels from DINOv2 and Stable Diffusion features, then filters them against a spherical 3D prototype before training a lightweight adapter [8]. In parallel, 3D foundation models make instance-level geometry practical from a single image: SAM3D reconstructs object-centric 3D shape [31], orientation models help resolve canonicalframe ambiguities [39], and 3D feature fields or functional-map methods provide geometry-aware descriptors on surfaces [21, 29, 7, 9, 46, 38]. In contrast to spherical-prior approaches, we combine instance-specific SAM3D meshes with PartField descriptors to both generate and filter pseudo-labels using faithful, per-instance 3D structure – removing the need for manual pose annotations and coarse geometric proxies.

##### 3 Method

We estimate semantic correspondences by combining 2D foundation features with 3D geometric priors obtained from reconstructed object meshes. Our pipeline has three stages: (i) we first reconstruct and canonicalize an object-centric 3D mesh for each instance; (ii) we then render 3D-aware PartField descriptors into the image plane and use them together with DINOv2 and Stable Diffusion features to propose semantic correspondences; (iii) finally, we reject geometrically inconsistent matches using geodesic consistency on the reconstructed meshes and train a lightweight correspondence adapter on the retained pseudo-labels.

###### 3.1 Canonicalized 3D Object Reconstruction

Our correspondence pipeline relies on a 3D mesh for each object instance, expressed in a canonical frame that is consistent across instances of the same category. We obtain such meshes from a single image without manual pose annotation by combining recent foundation models for segmentation and single-image 3D reconstruction with two refinement stages. While these foundation models provide a strong geometric prior, their outputs exhibit two systematic issues: the predicted scale and translation can be inaccurate, causing the rendered mesh to misalign with the image, and the canonical orientation is ambiguous up to discrete yaw rotations across instances. We address the first issue with a render-and-compare optimization that aligns the rendered silhouette to the observed mask, and the second with a yaw canonicalization step based on multi-view orientation estimation. The full process is illustrated in Figure 2.

2D Mask and 3D Mesh Initialization. We extract a 2D instance mask M ∈ {0,1}H×W with SAM3 [3], using the image together with the dataset-provided category label. Given this mask, SAM3D [31] reconstructs an object-centric mesh from the masked image in a feed-forward manner and additionally predicts the camera parameters used for rendering. In the following, we show how we refine and canonicalize this initial reconstruction.

2026_05_04_pipeline_3D-SC 07.05.2026, 12:01

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Initial mask projection

| | |
|---|---|
| | |

[Figure 8]

[Figure 9]

[Figure 10]

| | |
|---|---|
| | |

DT & Soft IoU Optimization

| | |
|---|---|
| | |

[Figure 11]

SAM + SAM3D OrientAnythingV2

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Optimized mask projection

Majority voting

[Figure 20]

2D Mask + 3D Mesh Pose Reﬁnement Yaw Canonicalization

Canonicalized Meshes

- Figure 2: Canonicalized 3D object reconstruction pipeline. Given an image, we obtain an instance mask and a mesh from foundation models. We then refine the mesh pose via a two-phase render-andcompare optimization based on a distance-transform (DT) and a soft-IoU phase. Finally, we resolve the residual four-fold yaw ambiguity by rendering the mesh at eight known orientations and applying OrientAnything V2 with majority voting to select the canonical yaw correction ∆ψ∗.

Render-and-Compare Pose Refinement. To correct the residual scale and translation error in the SAM3D reconstruction, we apply a render-and-compare optimization on top of the predicted camera. Concretely, we optimize a scale factor s = eℓ ∈ R>0 (parameterized in log-space to remain strictly positive) and a translation t ∈ R3 applied to the mesh, by minimizing the discrepancy between the rendered soft silhouette Mˆ (s,t) ∈ [0,1]H×W and the observed mask M. Since the soft IoU between Mˆ and M has no gradient when the two are disjoint, we proceed in two sequential phases: a distance-transform (DT) phase that provides a global gradient signal regardless of initial alignment, followed by a soft-IoU phase that sharpens the fit.

Distance-transform attraction. We first dilate M by r to obtain M˜ , providing tolerance for coarse mesh boundaries, and compute two squared distance fields normalized by the image diagonal d:

∥p − p′∥22, Din(p) = d1 min

∥p − p′∥22. (1)

Dout(p) = d1 min

p′: M˜ (p′)=1

p′: M˜ (p′)=0

Dout is zero inside M˜ and grows with distance to the mask; Din is zero outside M˜ and grows with depth into its interior. The DT loss combines these into a mask-alignment objective:

1 HW p

M ˆ p Dout(p) + Din(p) 1 − λMˆ p . (2)

LDT =

The first term pulls rendered mass that falls outside the mask back toward it, weighted by how far outside it is. The second term simultaneously penalizes uncovered mask interior and, through the coefficient λ > 1, rewards rendered coverage of the interior. Without this reward, the optimization tends to under-cover the mask under partial occlusion — the rendered silhouette settles on a small fully-contained region rather than extending to the occluded extent of the object.

Soft-IoU refinement. Once the rendered and observed masks overlap, the soft IoU has a usable gradient and we switch to a differentiable soft-IoU loss:

###### Mˆ p Mp

LIoU = 1 − p

. (3)

M ˆ p + Mp − Mˆ p Mp

p

This phase tightens the alignment that the previous phase has approximately established.

Yaw Canonicalization. Even after pose refinement, SAM3D meshes do not necessarily share a consistent canonical orientation across instances of the same category. We find that roughly 6% of meshes are misaligned by a multiple of 90◦ around the vertical axis — a four-fold yaw ambiguity that is most common for symmetric or elongated objects such as buses, boats, and trains. To resolve this without manual annotation, we use OrientAnything V2 [39] as an external orientation estimator. For each mesh, we render eight views at known yaw angles ψknown ∈ {0◦,45◦,...,315◦} and we estimate the apparent orientation ψest of each rendering. If the mesh is correctly canonicalized, ψest should match ψknown up to estimator noise; otherwise, the two differ by a multiple of 90◦. For each rendered view we therefore pick the discrete correction that best closes this gap,

∆ψ∗ = arg min

ψest + ∆ψ − ψknown , (4)

∆ψ ∈ {0◦,90◦,180◦,270◦}

and aggregate the eight candidates into a single one by majority vote, which makes the procedure robust to occasional orientation estimation errors. Each mesh is then rotated by the selected ∆ψ∗, yielding a set of consistently canonicalized meshes that serve as the geometric backbone for what follows.

4

https://app.diagrams.net/?src=about Page 1 of 1

2026_05_04_pipeline_3D-SC 07.05.2026, 11:02

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

| |[Figure 26]<br><br>[Figure 27]| |
|---|---|---|
| | | |

[Figure 28]

DINO

[Figure 29]

[Figure 30]

[Figure 31]

NN + c.c.

[Figure 32]

SD

[Figure 33]

[Figure 34]

[Figure 35]

PF

Sec. 3.1

Raster.

SD+DINO+PartField Features

Geodesic Distance

Geodesic Filtering

Input pairs

PartField

Candidates

Pseudo-label

- Figure 3: Pseudo-label correspondences pipeline. Given two images, we fuse DINO, SD, and PartField features (rasterized from the meshes of Section 3.1) and propose candidate matches via nearest-neighbor (NN) search with relaxed cyclic consistency (c.c). Each candidate is then geometrically verified by lifting the matched pixels onto the reconstructed meshes and computing the

geodesic error dsgeo⇄t; candidates exceeding threshold τgeo are rejected. The retained pseudo-labels P are used to train a lightweight correspondence adapter on top of frozen DINO+SD features.

###### 3.2 Pseudo-Label Semantic Correspondences

Given a pair of images of the same object category, we generate correspondence pseudo-labels in two stages. First, we fuse 2D foundation features (DINO+SD) with 3D-aware PartField features rasterized from the canonicalized meshes, and apply relaxed cyclic consistency to discard obvious mismatches. Second, each surviving candidate is verified geometrically: matched points are lifted onto their respective meshes and rejected if their geodesic distance exceeds a threshold. The two stages are complementary — cyclic consistency is a cheap image-space filter, while geodesic verification is a geometry-grounded confidence measure that exploits the 3D shapes from Section 3.1.

Notation. We use the superscripts □s and □t to denote quantities associated with the source and target, respectively. We denote p as a point in image space while v denotes a point in the 3D space.

PartField Features. PartField [21] (PF) predicts a continuous per-vertex feature field encoding geometric and part-level structure directly from the 3D shape S. These descriptors naturally distinguish parts that are visually similar but geometrically distinct (e.g., front vs. rear wheels, left vs. right legs), exactly the cases where 2D foundation features tend to collapse. To use PartField in image space, we rasterize the per-vertex descriptors into the input image using the SAM3D camera together with the refined pose from Section 3.1. Vertices outside the camera frustum or outside the foreground mask are discarded, and foreground pixels with no projected descriptor are filled by nearest-neighbor propagation. The result is an image-space PartField map aligned with the RGB image, which can be combined with 2D image features for semantic correspondence estimation. PCA visualizations and rasterization details are deferred to Supp. Section B.1.

Candidate Generation. Given the fused image-space DINO+SD+PF features, we propose candidate matches via nearest-neighbor search, retaining only those that pass a cyclic consistency check.

Feature fusion. Following Zhang et al. [43], we fuse our three feature sources by independently L2normalizing each and concatenating them with category-agnostic weights. We denote the normalized

feature vectors as Fs = F

∥Fs∥2 . The fused representation is then defined as Ffused = √α FSD, β FDINO, √γ FPF , with γ = 1 − α − β. (5)

s

We use the weights α = 1/2, β = 1/3, and γ = 1/6, which we found offers a good balance between the three features in practice; a weight sweep is provided in Supp. Section B.2. Candidate matches are then proposed by nearest-neighbor search in the fused space.

Relaxed cyclic consistency. While the 3D-aware PartField features significantly enhance the matching quality, some candidates remain wrongly matched. To filter these mismatches, we apply a relaxed cyclic consistency check inspired by Aberman et al. [1]. As observed in Dünkel et al. [8], strict cyclic consistency rejects a large fraction of correct matches due to sub-pixel noise; we therefore relax the criterion to require only that the backward match lies within a small spatial tolerance of the source. A candidate (ps,pˆt) with pˆt = NNs→t

fused(ps) is retained if NNt→s

fused(ˆpt) − ps 2 < τcc · max(h,w), (6)

where NNfused denotes nearest-neighbor search in the fused feature space, h and w are the object’s bounding dimensions, and τcc is a tolerance ratio.

5

https://app.diagrams.net/?src=about Page 1 of 1

Candidate Verification via Geodesic Filtering. Our fused descriptor uses a fixed mixing strategy, and these fused features inevitably produce some wrong matches since objects greatly vary across instances. Cyclic consistency removes some of them but operates purely in feature space, ignoring 3D geometry. We therefore add a geodesic consistency stage: matched locations lifted onto canonically posed meshes must land in nearby surface regions.

Lifting matches to 3D. Given a candidate match (ps,pt), we cast a ray from each camera through the corresponding pixel and intersect it with the respective mesh, obtaining the unprojected points vs and vt together with their containing triangles and barycentric coordinates. Because geodesic distances are computed between mesh vertices, we snap each unprojected point to the dominant vertex of its triangle (the vertex with the largest barycentric weight), giving v¯s and v¯t.

Cross-mesh correspondence via PartField. The previous step places each candidate match onto the source and target meshes individually. However, the meshes share only a canonical orientation but not vertex correspondence. To compare the lifted source and target points, we therefore estimate a

- 3D correspondence between the meshes themselves. Hence, we use PartField nearest-neighbor as the cross-mesh correspondence: we interpolate the PartField descriptor at vs on the source mesh and

search for its nearest neighbor among the PartField descriptors on the target mesh, vˆt = NNs→t

PF (vs), yielding a target vertex vˆt that represents the cross-mesh counterpart of vs. A candidate is then geometrically consistent if this PartField-predicted target vˆt is geodesically close to the target obtained from the image-space match, v¯t.

Bicyclic geodesic error. We measure the disagreement between the two target predictions as a bicyclic geodesic distance, combining a forward and a backward geodesic error on the source and target meshes. The forward error measures, on the target mesh, the geodesic distance between the cross-mesh prediction vˆt and the target v¯t obtained from the image-space match:

dsgeo→t = dM

v ˆt, v¯t . (7) A symmetric computation in the reverse direction yields a backward error dtgeo→s = dM

t

(vˆs, v¯s), where vˆs = NNt→s

s

PF (vt). We average the two and normalize by the mesh bounding-box diagonals so that the score is comparable across instances and categories of varying scale:

dsgeo→t diag(Mt)

dtgeo→s diag(Ms)

- 1

- 2

dsgeo⇄t =

. (8)

+

Intuitively, dsgeo⇄t is small when the image-space candidate and the PartField cross-mesh correspondence agree on the same surface location, and large when they disagree.

Rejection of wrong pseudo-labels. We use the bicyclic geodesic error as a per-candidate quality score and threshold it to reject inconsistent pseudo-labels. A candidate (ps,pt) is retained if and only if its error falls below a threshold τgeo:

P = (ps,pt) dsgeo⇄t ≤ τgeo . (9)

Because dsgeo⇄t is normalized by the mesh bounding-box diagonals, a single value of τgeo applies across object instances and categories of varying scale. Crucially, we do not require correspondences to cover every object part: obtaining fewer but geometrically reliable pseudo-labels is preferable to dense but noisy supervision, since the adapter only benefits from matches it can trust.

###### Supervised Training with Pseudo-Labels.

We use the pseudo-labels P to train a lightweight adapter fp(·) on top of frozen DINOv2 and Stable Diffusion features, following Dünkel et al. [8]. The adapter has been shown to outperform zero-shot feature concatenation [43, 44] and weighted feature combinations with weak geometric regularization [24], while keeping the underlying foundation features unchanged. We denote the adapted features by Fs and Ft for the source and target images, respectively. We supervise fp(·) with two complementary losses. A sparse contrastive loss [23] acts on the labeled pseudo-correspondences, maximizing similarity between matched points and minimizing it against non-matching points:

Lsparse = CL Fs(Ps), Ft(Pt) . (10)

- Table 1: Evaluation on standard benchmarks. Per-image PCK (%, ↑) at multiple thresholds on SPair-71k (test and Geo-Aware subset), AP-10K and SpairU. ‘–’ indicates missing numbers. Best per method type is shown in bold. Full Table including Supervised methods can be found in Table C1.

SPair-71k SPair-Geo-Aware AP-10K (0.10) SpairU Method 0.01 0.05 0.10 0.01 0.05 0.10 I.S. C.S. C.F. 0.01 0.05 0.10 Unsupervised

DINOv2+NN Zhang et al. [43] 6.3 38.4 53.9 3.4 28.2 42.0 60.9 57.3 47.4 – – 54.9 DIFT Tang et al. [34] 7.2 39.7 52.9 3.4 28.2 42.5 50.3 46.0 35.0 – – 47.4

Weakly supervised with human annotations

Spherical Map. Mariotti et al. [24] 8.4 48.2 64.4 – – – 65.4 63.1 51.0 – – 61.0 DIY-SC Dünkel et al. [8] 10.1 53.8 71.6 7.7 47.7 67.5 70.6 69.8 57.8 5.4 44.0 67.9

Weakly supervised without human annotations

SD+DINOv2 Zhang et al. [43] 7.9 44.7 59.9 5.3 34.5 49.3 62.9 59.3 48.3 – – 59.4 DIY-SC+OriAny Dünkel et al. [8] 9.5 51.2 69.6 6.9 45.7 65.8 69.3 66.8 54.0 5.2 43.1 66.3 3D-SC (Ours) 10.2 54.8 73.0 7.8 50.1 70.8 69.6 68.5 56.9 5.6 43.5 67.3

A dense regression loss [44] additionally propagates gradients to image regions without explicit labels by predicting the target location with a window soft-argmax over the feature similarity map and penalizing its deviation from the labeled target:

p ˆt − (pt + ϵ) 2, pˆt = WINDOWSOFTARGMAX Fs(ps)⊤Ft , (11)

Ldense =

(ps,pt)∈P

where ϵ is small Gaussian noise that regularizes the predicted location at sub-pixel scale. The adapter is trained with the sum L = Lsparse + Ldense.

##### 4 Experiments

In this section, we evaluate on four semantic correspondence benchmarks against unsupervised and weakly supervised baselines, and ablate the key components of our pipeline.

###### 4.1 Implementation Details

We report the values for the method parameters introduced in Section 3. For the distance-transform objective in Equation (2), we set the interior-coverage reward to λ = 4. During pose refinement, we optimize log-scale and translation with Adam using separate learning rates lrscale = 0.05 and lrtrans = 0.02. Similarly to Zhang et al. [43], we extract SD and DINO features from images resized to 9602 (SD, DINOv3) and 8402 (DINOv2). PartField descriptors are rasterized at the shared correspondence-map resolution of 602. For the relaxed cyclic consistency, we set the tolerance to τcc = 0.05 of the object’s bounding box, with a lower bound of one feature-map patch. For geometric verification, we use τgeo = 0.05. Following prior work [44, 8], we use a four-layer, 5M-parameter adapter. We train it with AdamW with lr = 5·10−3, weight decay of 10−3, and a one-cycle schedule for 200k iterations. Each image pair has ∼1600 pseudo-annotations; we sample 50 per iteration to prevent denser pairs from dominating training. More details in Supp. Section C.1.

Benchmarks and Metrics. We evaluate on four standard semantic correspondence benchmarks. SPair-71k [26] contains 71k image pairs across 18 categories, with up to 20 keypoints per image and up to 900 images per category. Following Zhang et al. [44], we additionally report results on SPair-Geo-Aware, a subset of SPair-71k that emphasizes challenging correspondences involving symmetric or repeated parts and therefore better tests whether a method correctly captures object orientation and geometry. SPair-U [25] extends SPair-71k with ∼ 4 additional unseen keypoints per category, providing an interesting evaluation of keypoint-level generalization. AP-10K [42] is an animal pose dataset with 17 keypoints shared across 54 species spanning intra-species, cross-species, and cross-family matching. Following prior work [43], we use the Percentage of Correct Keypoints (PCK@α) as metric, for which a prediction is considered correct if it lies within a distance of α · max(h,w) from the ground-truth keypoint with h,w the object’s bounding-box dimensions. We only report the most common metric: per-image PCK averaged over the test set.

Baselines. We compare our performances with recent works which we categorize into 4 different categories: Unsupervised, Weakly Supervised with Human Annotations required, and Weakly Supervised without Human Annotations (3D-SC’s category). Our focus will remain on the Unsupervised

[Figure 36]

- Figure 4: Qualitative pseudo-annotations. We visualize pseudo-ground-truth annotations from 3D-SC and DIY-SC. 3D-SC produces denser and more geometrically consistent pseudo-annotations.

and Weakly supervised approaches. DIFT [23], and SD + DINOv2 [43] extract features from foundation models and perform nearest-neighbor matching in feature space. Spherical mapper [24] and DIY-SC [8] both leverage pose annotations as weak supervision during training.

###### 4.2 Experimental Results

Evaluation on SPair-71k. As shown in Table 1, 3D-SC establishes the strongest results among weakly supervised methods on SPair-71k, reaching 73.0 PCK@0.1. In particular, it improves over the strongest baseline in the same supervision regime, DIY-SC+OriAny, by 3.4 points. Per-category results in Table C2 show that the gains are concentrated in rigid categories with strong geometric symmetry, such as bus (+10.8), tv monitor (+9.8), car (+6.9), and motorcycle (+5.1), while nonrigid categories such as animals show no gain or can slightly regress. The gains are even more pronounced on SPair-Geo-Aware, where our method reaches 70.8 PCK@0.1, clearly surpassing all existing weakly supervised approaches. This behavior is consistent with our central hypothesis: because our pseudo-labels are grounded in reconstructed 3D geometry, they are especially effective on correspondences that require disambiguating symmetric or repeated parts, and viewpoint changes.

Evaluation on SPairU. On SPairU, 3D-SC obtains 67.3 PCK@0.1. This is the best result among methods without human annotations and is only 0.6 points below DIY-SC, which leverages human annotations. The smaller margin compared with SPair-Geo-Aware subset is expected: SPairU mainly probes generalization to previously unseen keypoints which are usually located at the middle of the limbs/parts. Our PartField features, trained on part contrastive learning, are not explicitly designed to differentiate keypoints within the same part. Hence we do not expect a large gain from PartField features on this benchmark, explaining the modest improvement over DIY-SC+OriAny (1 point). Nevertheless, the result shows that the representation learned from our pseudo-labels transfers also to these keypoint definitions.

Evaluation on AP-10K. Our method also transfers well to the more articulated and shape-diverse setting of AP-10K. 3D-SC achieves 69.6/68.5/56.9 PCK@0.1 on the intra-species, cross-species, and cross-family splits, outperforming the strongest baseline without human annotations on all three splits. These improvements are particularly meaningful on the harder cross-species and cross-family evaluations, where appearance cues alone are often insufficient. Although PartField descriptors can be less reliable for animals in unusual poses during the pseudo-annotation procedure, the overall results show that our 3D-aware pseudo-label generation and filtering pipeline remains effective well beyond the rigid object categories of SPair-71k.

Qualitative results. As shown in Figure 4 (additional visualizations in Figure C1), 3D-SC produces well-distributed pseudo-annotations that cover the commonly visible parts of each object. The matches are geometrically consistent and free from left-right ambiguities, a direct consequence of anchoring correspondence in instance-specific 3D geometry.

###### 4.3 Ablations

PartField Features. Table 2 reports the effect of adding PartField to the feature fusion. Compared to SD+DINO alone, SD+DINO+PartField simultaneously lowers the False Positive Rate (FPR) of unfiltered candidates and increases the average number of candidates retained per pair. These two effects together indicate that integrating PartField not only suppresses incorrect matches but also surfaces additional correct ones that SD+DINO misses, consistent with its ability to distinguish geometrically distinct but visually similar regions such as front and rear wheels or left and right parts. The downstream impact is confirmed in Table 3: adding PartField improves PCK@0.1 on SPair-71k by 0.6 points over the SD+DINO baseline.

Filtering. We validate the geodesic filtering stage on the SPair-71k validation set. For each annotated keypoint we compute its nearest neighbor in the fused feature space and check whether the prediction

- Table 2: Filtering evaluation on validation set. FPR refers to False Positive Rate, i.e., unfiltered wrong prediction. #Candidates refers to the average total number of filtered correspondences per pair.

Filter FPR #Candidates Features: SD+DINO

Spherical mapper 10.95 1856 Triplane 13.15 1948 PF Feature similarity 2.81 1608 Geodesic 1.82 1543

Features: SD+DINO+PartField

Spherical mapper 10.75 2001 Triplane 13.07 2090 PF Feature similarity 2.47 1694 Geodesic 1.78 1634

Table 3: Ablations on SPair-71k. All introduced components bring a significant improvement. The baseline is evaluated using the SD+DINO zero-shot approach with window soft argmax. ‘c.c.’ = cyclic consistency.

pseudo PF c.c. filter. sampl. DINO PCK@.1

|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓|✓ ✓ ✓ ✓ ✓<br><br>|✓ ✓ ✓<br><br>✓ ✓ ✓ ✓|✓ ✓<br><br>✓ ✓ ✓<br><br>|✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓<br><br>✓<br><br>|v2 v2 v2 v2 v3 v2 v2 v2 v3 v3<br><br>|64.9 67.0 67.6 71.6 72.4 66.9 68.8 72.1 72.4 73.0<br><br>|
|---|---|---|---|---|---|---|
|DIY-SC DIY-SC+OriAny<br><br>| | | | |v3 v3|72.1 70.4<br><br>|

is correct under PCK@0.1. A wrong prediction that survives filtering counts as a false positive; the FPR is therefore the fraction of unfiltered predictions that are incorrect. As shown in Table 2, our bicyclic geodesic filter achieves the lowest FPR of 1.78% among the filtering strategies we compared. The benefit also carries over to the trained adapter: Table 3 shows a gain of 3.3 PCK@0.1 points when geodesic filtering is applied versus using all cyclic-consistency candidates without further rejection. Finally, capping the number of pseudo-labels sampled per pair during training improves PCK@0.1 by 0.6 points; without this cap, pairs with denser pseudo-label sets dominate the gradient and reduce the effective diversity of training signal.

Backbone. Replacing DINOv2 with DINOv3 as the vision backbone yields an improvement of 0.9 PCK@0.1. To disentangle this gain from other design choices, we applied the same substitution to both DIY-SC variants and observe a similarly sized increase, confirming that a slight improvement (0.5-0.9) is attributable to the stronger backbone in general. Importantly, our method outperforms both DIY-SC variants in either backbone setting.

##### 5 Limitations and Future Work

Our pipeline depends on SAM3D’s pose and shape estimates; errors propagate through the 2D–3D reprojection and can degrade geodesic consistency, although our filtering removes most resulting false positives. PartField’s part-level contrastive training provides coarse regional cues rather than precise within-part localization, which motivates its relatively low fusion weight; this limitation is reflected in our SPairU results, where keypoints often lie in the middle of parts and PartField contributes less signal. A stronger 3D feature, ideally one tailored to deformable categories such as animals, would likely warrant a higher weight. Finally, our cross-mesh correspondence uses nearest-neighbor matching in PartField space; replacing it with denser registration via optimal transport or functional maps [29] is a natural next step, trading additional compute for finer alignment.

##### 6 Conclusion

We presented a 3D-aware post-training framework for semantic correspondence that leverages priors from 3D foundation models without requiring human pose annotations. By combining SAM3D-based geometry and pose estimation with a render-and-compare refinement step, we obtain instance-specific 3D structure that drives both feature construction and pseudo-label filtering: PartField descriptors rendered into the image plane provide geometry-aware cues that complement DINO and Stable Diffusion features, while geodesic distances on the reconstructed shapes enable principled filtering of inconsistent matches. The filtered correspondences supervise a lightweight adapter that yields consistent improvements over prior methods on standard benchmarks. Our results suggest that instance-specific 3D structure can be a more reliable geometric prior than the coarse spherical proxies used by prior post-training approaches, and that it can be obtained automatically from off-the-shelf 3D foundation models. We see this as an early step toward a new class of self-supervised pipelines where 3D foundation models act as geometric teachers for 2D tasks, a direction that becomes more powerful as 3D reconstruction quality continues to improve.

##### Acknowledgments and Disclosure of Funding

AK acknowledges support via his Emmy Noether Research Group funded by the German Research Foundation (DFG) under grant number 468670075. This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant number 539134284, through EFRE (FEIH_2698644) and the state of Baden-Württemberg.

[Figure 37]

[Figure 38]

##### References

- [1] K. Aberman, J. Liao, M. Shi, D. Lischinski, B. Chen, and D. Cohen-Or. Neural best-buddies: Sparse cross-domain correspondence. ACM Transactions on Graphics (TOG), 2018.
- [2] S. Amir, Y. Gandelsman, S. Bagon, and T. Dekel. Deep ViT features as dense visual descriptors. In European Conference on Computer Vision Workshop (ECCVW), 2022.
- [3] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun,

- R. Rädle, T. Afouras, E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. Hazra, S. Ding,
- S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Dollár, N. Ravi, K. Saenko, P. Zhang, and C. Feichtenhofer. SAM 3: Segment anything with concepts, 2025. URL https://arxiv.org/abs/2511.16719.

- [4] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9650–9660, 2021.
- [5] Y. Chi, L. Sommer, O. Dünkel, D. Muhle, D. Cremers, C. Theobalt, and A. Kortylewski. C3po: Canonicalization of 3d pose from partial views with generalizable correspondence features. In International Conference on 3D Vision (3DV), 2026.
- [6] C. Cuttano, G. Trivigno, C. Masone, and S. Roth. MARCO: Navigating the unseen space of semantic correspondence. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2026.
- [7] N. Donati, A. Sharma, and M. Ovsjanikov. Deep geometric functional maps: Robust feature learning for shape correspondence. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 8592–8601, 2020.
- [8] O. Dünkel, T. Wimmer, C. Theobalt, C. Rupprecht, and A. Kortylewski. Do it yourself: Learning semantic correspondence from pseudo-labels. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.
- [9] N. S. Dutt, S. Muralikrishnan, and N. J. Mitra. Diffusion 3d features (diff3f): Decorating untextured shapes with distilled semantic features. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4494–4504, 2024.
- [10] F. Fundel, J. Schusterbauer, V. T. Hu, and B. Ommer. Distillation of diffusion features for semantic correspondence. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), 2025.
- [11] B. Ham, M. Cho, C. Schmid, and J. Ponce. Proposal flow: Semantic correspondences from object proposals. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 40

(7):1711–1725, 2017.

- [12] R. Hartwig, D. Muhle, R. Marin, and D. Cremers. Geco: Geometrically consistent embedding with lightspeed inference. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9309–9319, 2025.

- [13] E. Hedlin, G. Sharma, S. Mahajan, H. Isack, A. Kar, A. Tagliasacchi, and K. M. Yi. Unsupervised semantic correspondence using stable diffusion. In Conference on Neural Information Processing Systems (NeurIPS), 2023.
- [14] Y. Huang, Y. Sun, C. Lai, Q. Xu, X. Wang, X. Shen, and W. Ge. Weakly supervised learning of semantic correspondence through cascaded online correspondence refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
- [15] A. Jesslen, G. Zhang, A. Wang, W. Ma, A. Yuille, and A. Kortylewski. Novum: Neural object volumes for robust object classification. In European Conference on Computer Vision (ECCV), 2024.
- [16] J. Kim, K. Ryoo, J. Seo, G. Lee, D. Kim, H. Cho, and S. Kim. Semi-supervised learning of semantic correspondence with pseudo-labels. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [17] N. Kulkarni, S. Tulsiani, and A. Gupta. Canonical surface mapping via geometric cycle consistency. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 2202–2211, 2019. doi: 10.1109/ICCV.2019.00229.
- [18] X. Li, D.-P. Fan, F. Yang, A. Luo, H. Cheng, and Z. Liu. Probabilistic model distillation for semantic correspondence. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
- [19] X. Li, J. Lu, K. Han, and V. A. Prisacariu. SD4Match: Learning to prompt Stable Diffusion model for semantic matching. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 27558–27568, 2024.
- [20] C. Liu, J. Yuen, and A. Torralba. Sift flow: Dense correspondence across scenes and its applications. IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 33(5): 978–994, 2011.
- [21] M. Liu, M. A. Uy, D. Xiang, H. Su, S. Fidler, N. Sharp, and J. Gao. PartField: Learning 3D feature fields for part segmentation and beyond. arXiv preprint arXiv:2504.11451, 2025.
- [22] D. G. Lowe. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision (IJCV), 60(2):91–110, 2004.
- [23] G. Luo, L. Dunlap, D. H. Park, A. Holynski, and T. Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. In Conference on Neural Information Processing Systems (NeurIPS), 2023.
- [24] O. Mariotti, O. Mac Aodha, and H. Bilen. Improving semantic correspondence with viewpointguided spherical maps. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [25] O. Mariotti, Z. Du, Y. Bhalgat, O. Mac Aodha, and H. Bilen. Jamais Vu: Exposing the generalization gap in supervised semantic correspondence. In Conference on Neural Information Processing Systems (NeurIPS), volume 38, 2025.
- [26] J. Min, J. Lee, J. Ponce, and M. Cho. Spair-71k: A large-scale benchmark for semantic correspondence, 2019. URL https://arxiv.org/abs/1908.10543.
- [27] N. Neverova, D. Novotny, V. Khalidov, M. Szafraniec, P. Labatut, and A. Vedaldi. Continuous surface embeddings for deformable shape correspondence. Conference on Neural Information Processing Systems (NeurIPS), 2020.
- [28] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. DINOv2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [29] M. Ovsjanikov, M. Ben-Chen, J. Solomon, A. Butscher, and L. Guibas. Functional maps: a flexible representation of maps between shapes. ACM Transactions on Graphics (TOG), 31(4): 1–11, 2012.

- [30] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022.
- [31] SAM 3D Team, X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, A. Lin, J. Liu, Z. Ma, A. Sagar, B. Song, X. Wang, J. Yang, B. Zhang, P. Dollár, G. Gkioxari, M. Feiszli, and J. Malik. SAM 3D: 3Dfy anything in images, 2025. URL https://arxiv.org/abs/2511.16624.
- [32] A. Shtedritski, C. Rupprecht, and A. Vedaldi. Shic: Shape-image correspondences with no keypoint supervision. In European Conference on Computer Vision (ECCV), 2024.
- [33] L. Sommer, O. Dünkel, C. Theobalt, and A. Kortylewski. Common3d: Self-supervised learning of 3d morphable models for common objects in neural feature space. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 6468–6479, June 2025.
- [34] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan. Emergent correspondence from image diffusion. In Conference on Neural Information Processing Systems (NeurIPS), 2023.
- [35] T. Taniai, S. N. Sinha, and Y. Sato. Joint recovery of dense correspondence and cosegmentation in two images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 4246–4255, 2016.
- [36] N. Tumanyan, M. Geyer, S. Bagon, and T. Dekel. Plug-and-play diffusion features for textdriven image-to-image translation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1921–1930, 2023.
- [37] K. Wandel and H. Wang. Semalign3d: Semantic correspondence between rgb-images through aligning 3d object-class representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1138–1147, 2025. doi: 10.1109/CVPR52734. 2025.00114.
- [38] P. Wang, T. Ikeda, R. Lee, and K. Nishiwaki. Gs-pose: Category-level object pose estimation via geometric and semantic correspondence. In European Conference on Computer Vision (ECCV), pages 108–126. Springer, 2024.
- [39] Z. Wang, Z. Zhang, J. Xu, J. Wang, T. Pang, C. Du, H. Zhao, and Z. Zhao. Orient anything v2: Unifying orientation and rotation understanding. In Conference on Neural Information Processing Systems (NeurIPS), 2025.
- [40] F. Xue, S. Elflein, L. Leal-Taixe, and Q. Zhou. MATCHA: Towards matching anything. arXiv preprint arXiv:2501.14945, 2025.
- [41] L. Yi, V. G. Kim, D. Ceylan, W. Shen, M. Yan, H. Su, C. Lu, Q. Huang, A. Sheffer, and L. Guibas. A scalable active framework for region annotation in 3d shape collections. In ACM Trans. Graphics (Proc. SIGGRAPH Asia), 2016.
- [42] H. Yu, Y. Xu, J. Zhang, W. Zhao, Z. Guan, and D. Tao. AP-10k: A benchmark for animal pose estimation in the wild. In Conference on Neural Information Processing Systems (NeurIPS), 2021.
- [43] J. Zhang, C. Herrmann, J. Hur, L. F. Polanía, V. Jampani, D. Sun, and M.-H. Yang. A tale of two features: Stable diffusion complements DINO for zero-shot semantic correspondence. In Conference on Neural Information Processing Systems (NeurIPS), 2023.
- [44] J. Zhang, C. Herrmann, J. Hur, E. Chen, V. Jampani, D. Sun, and M.-H. Yang. Telling left from right: Identifying geometry-aware semantic correspondence. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3076–3085, 2024.
- [45] T. Zhou, P. Krahenbuhl, M. Aubry, Q. Huang, and A. A. Efros. Learning dense correspondence via 3d-guided cycle consistency. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

###### [46] J. Zhu, Y. Ju, J. Zhang, M. Wang, Z. Yuan, K. Hu, and H. Xu. Densematcher: Learning 3d semantic correspondence for category-level manipulation from a single demo. International Conference on Learning Representations (ICLR), 2025.

### Geometry Matters: 3D Foundation Priors for Learning Semantic Correspondence

#### Supplementary Material

This supplement is organized as follows. Section A provides implementation details for the 3D reconstruction and pose canonicalization pipeline. Section B gives additional details on feature fusion, pseudo-label generation, and geodesic filtering. Section C reports per-category results and additional qualitative visualizations. Section D discusses reproducibility and the use of LLM assistance in writing this paper.

- (A) Pseudo-groundtruth via foundation models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- (B) Correspondence pseudo-annotations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- (C) Additional results and visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- (D) Reproducibility and LLM assistance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

##### A Pseudo-groundtruth via foundation models

This section provides additional details on the 3D reconstruction and canonicalization pipeline summarized in Section 3.1. The full pipeline is illustrated in Figure 2. Starting from a single image, we (i) extract a 2D instance mask and reconstruct an object-centric mesh using foundation models, (ii) refine the mesh pose via a two-phase render-and-compare optimization, and (iii) resolve any residual yaw ambiguity by comparing rendered views against estimated orientations.

- 2D Mask and 3D Mesh Initialization. To extract instance masks with SAM3 [3], we prompt the model with the category label provided by SPair-71k. These prompts improve mask quality and reduce failure cases, but are not strictly required: masks can be obtained without them, at the cost of additional noise in the downstream pipeline. We validate this choice empirically: a simple baseline using DINOv2 CLS token embeddings with kNN classification achieves ∼ 99% accuracy on the category classification task. Failure cases occur primarily when multiple objects occupy a single image (affecting <1% of instances), representing a negligible impact on overall performance. We consider this a reasonable choice since neither the bounding box nor the category label constitutes additional human annotation beyond what the dataset already provides, and both can be obtained automatically with off-the-shelf object detectors if needed.

Render-and-compare pose refinement. The interior-coverage reward (λ > 1 in Equation (2)) is critical in cases of strong occlusion. Without it, we found that the optimization sometimes escapes the distance-transform penalty by pushing the rendered silhouette entirely outside the image — avoiding the loss rather than solving it (e.g., the partially occluded chair in Figure 2). The interior-coverage term counteracts this by rewarding rendered mass that lands inside the observed mask, preventing the degenerate solution. We additionally apply a strong penalty whenever more than 25% of the rendered silhouette falls outside the image boundary.

We set λ = 4 and dilate the observed mask by r = 4 pixels before computing the distance-transform fields (see Equation (1)), providing tolerance for coarse mesh boundaries. We optimize log-scale and translation jointly with Adam, using separate learning rates lrscale = 0.05 and lrtrans = 0.02. The higher learning rate for scale biases the optimization toward correcting the dominant error (scale mismatch) rather than compensating with depth drift, which is harmless for 2D reprojection quality but can destabilize the geometry. We run the distance-transform phase for 100 gradient steps, then switch to soft-IoU refinement for a further 50 steps to tighten the final alignment.

Yaw Canonicalization statistics. Following the canonicalization verification procedure described in Section 3.1, we applied discrete orientation corrections to a small subset of the refined meshes. Excluding meshes marked as wrong, 79 out of 1,319 instances required a non-zero rotation, corresponding to 5.99% of the dataset. These corrections indicate cases where the estimated orientation differed from the target canonical pose by one of the allowed discrete rotations.

All corrections were rotations around the y-axis: 34 meshes required a 270◦ rotation, 24 required a 90◦ rotation, and 15 required a 180◦ rotation. The corrections were distributed across both splits, with 59 rotated meshes in the training set and 20 in the validation set. The most frequently affected classes were bus, boat, train, and cow, with 23, 15, 8, and 7 corrected meshes, respectively.

##### B Correspondence pseudo-annotations

This section provides additional details on the correspondence pseudo-annotation pipeline described in Section 3.2. We first give further analysis of the PartField features used in our feature fusion, including PCA visualizations and rasterization details (Section B.1). We then detail the feature fusion weight search and justify the square-root weighting scheme (Section B.2).

###### B.1 PartField Features

We visualize PartField features [21] using PCA projections (Figure B1) and query-based similarity heatmaps (Figure B2). These visualizations show that PartField features are spatially coherent within semantic parts while remaining discriminative across repeated or symmetric structures.

Rasterization details. To rasterize PartField vertex features into the image plane, we first render the reconstructed 3D mesh into the image using the estimated pose, obtaining a mapping from each pixel to its corresponding 3D point on the mesh. We then assign to each pixel the PartField feature of its corresponding 3D point, effectively projecting the 3D-aware features into the 2D image space. This process allows us to leverage the geometric context captured by PartField features while maintaining alignment with the original image, enabling more accurate correspondence estimation.

[Figure 39]

[Figure 40]

(a) PCA projection across two car instances. (b) PCA projection across two chair instances.

- Figure B1: PCA visualizations of PartField features. We project PartField features to RGB using PCA and visualize them on pairs of object instances. Consistent colors within individual parts indicate that the features are spatially coherent, while similar colors across instances suggest that corresponding geometric parts, such as chair legs or car body regions, are mapped to nearby feature representations.

[Figure 41]

(a) Highest similarity is correctly localized to the queried right-rear wheel.

[Figure 42]

(b) Highest similarity is correctly localized to the queried right-front chair leg.

- Figure B2: PartField features reduce repeated-part and symmetry ambiguities. For each example, the left mesh shows the query point in red, and the right mesh shows the cosine-similarity heatmap induced by the queried PartField feature. In the car example, similarity concentrates on the queried wheel rather than activating all repeated wheels. In the chair example, the response remains localized to the corresponding leg, avoiding front/back and left/right confusion. This suggests that PartField similarities are anchored in geometric context rather than only in semantic part identity.

###### B.2 Feature fusion

We select the fusion weights α, β, and γ from Equation (5) via a grid search on the SPair-71k validation set, sweeping α and β in increments of 1/6 with γ = 1 − α − β, and measuring PCK@0.1

of unfiltered predicted correspondences. As shown in Figure B3, several weight combinations reach similar peak performance, confirming that the method is not too sensitive to the exact weighting and all features provide some contributions. Among these, we select α = 1/2, β = 1/3, and γ = 1/6: configurations that upweight PartField (γ) tend to yield better downstream performance after adapter training, consistent with PartField resolving the geometrically ambiguous correspondences that provide the most informative supervision signal.

[Figure 43]

- Figure B3: Feature fusion weight search. PCK@0.10 of pseudo-correspondences before filtering on the SPair-71k validation set, as a function of the SD weight α and DINOv2 weight β (the PartField weight γ = 1 − α − β is determined by the other two). Multiple combinations achieve similar peak performance; we use α = 1/2, β = 1/3, γ = 1/6 as our default.

Square-root fusion weights. Let FSD, FDINO, and FPF denote the independently L2-normalized feature vectors of any two candidate points. The dot product between their fused features is

Ffused⊤ Ffused′ = √α FSD⊤

√α FSD′ + β FDINO⊤ β FDINO′ + √γ FPF⊤

√γ FPF′ (B.1) = α FSD⊤ FSD′ + β FDINO⊤ FDINO′ + γ FPF⊤ FPF′ . (B.2)

Since each feature source is L2-normalized independently, each dot product is exactly the cosine similarity within that feature space. Therefore, the cosine similarity in the concatenated fused space is equivalent to a weighted average of the cosine similarities computed independently for each feature source. The square roots appear because the weights are applied to both vectors before taking the dot product, e.g. √α√α = α.

Table C1: Evaluation on standard benchmarks. Per-image PCK (%, ↑) at multiple thresholds on SPair-71k (test set and Geo-Aware subset), AP-10K and SpairU. † Results obtained from the official checkpoint. ‘–’ indicates missing numbers. Best per method type is shown in bold.

SPair-71k SPair-Geo-Aware AP-10K (0.10) SpairU Method 0.01 0.05 0.10 0.01 0.05 0.10 I.S. C.S. C.F. 0.01 0.05 0.10 Supervised

DHF Luo et al. [23] 8.7 50.2 64.9 8.0 45.8 62.7 62.7 60.0 47.8 – – – SD+DINOv2 Zhang et al. [43] 9.6 57.7 74.6 9.9 57.0 77.0 77.0 74.0 65.8 – – – GECO Hartwig et al. [12] 14.2 59.6 73.6 – – – 82.5 81.2 76.6 – – 55.2 Jamais Vu Mariotti et al. [25] 20.5 71.9 82.5 – – – – – – – – 62.4 Geo-SC Zhang et al. [44] 21.7 72.8 83.2 – – – 87.7 85.9 78.5 – – 56.9 SemAlign3D Wandel and Wang [37] 15.8 77.5 88.9 – – – – – – – – – MARCO Cuttano et al. [6] 27.0 77.6 87.2 22.8† 76.8† 87.5† 89.1 88.3 83.4 5.0† 42.7† 67.5

Unsupervised

DINOv2+NN Zhang et al. [43] 6.3 38.4 53.9 3.4 28.2 42.0 60.9 57.3 47.4 – – 54.9 DIFT Tang et al. [34] 7.2 39.7 52.9 3.4 28.2 42.5 50.3 46.0 35.0 – – 47.4

Weakly Supervised with human annotations

Spherical Map. Mariotti et al. [24] 8.4 48.2 64.4 – – – 65.4 63.1 51.0 – – 61.0 DIY-SC Dünkel et al. [8] 10.1 53.8 71.6 7.7 47.7 67.5 70.6 69.8 57.8 5.4 44.0 67.9

Weakly Supervised without human annotations

SD+DINOv2 Zhang et al. [43] 7.9 44.7 59.9 5.3 34.5 49.3 62.9 59.3 48.3 – – 59.4 DIY-SC+OriAny Dünkel et al. [8] 9.5 51.2 69.6 6.9 45.7 65.8 69.3 66.8 54.0 5.2 43.1 66.3 3D-SC (Ours) 10.2 54.8 73.0 7.8 50.1 70.8 69.6 68.5 56.9 5.6 43.5 67.3

##### C Additional results and visualizations

###### C.1 Additional implementation details

Compute. Unless stated otherwise, all reported runtimes are measured on a single NVIDIA L40 GPU with 40GB of memory; our pipeline is also compatible with smaller memory budgets. The canonicalized 3D object reconstruction takes 12.42s per object on average. Computing the pseudolabels for the full SPair-71k training set (∼53k pairs) takes roughly 18h end-to-end, including SD, DINO, and PartField feature extraction, rasterization of PartField descriptors, cyclic consistency, and geodesic filtering for each image pair. Note that this pipeline could benefit from further optimization and parallelization to reduce runtime with minimal work. Training the adapter for 200k iterations takes about 4h on a single GPU.

###### C.2 Additional results

We provide additional results complementing those in the main paper. In particular, we extend the benchmark tables to include Supervised methods in Table C1, which were omitted from the main paper for space, and report per-category PCK results on SPair-71k in Table C2. We exclude the weakly supervised without human annotations variant of Telling Left from Right (Geo-SC) [44] from the main paper, as it reports PCK normalized per keypoint rather than per image, making direct comparison unreliable. For completeness, per-category results for Geo-SC are included in Table C2.

Per-category results are reported in Table C2 (per-keypoint PCK@0.1). The pattern of gains is consistent with our central hypothesis: the largest improvements over DIY-SC+OriAny (and even DIY-SC which was trained with human supervision) occur in rigid, man-made categories with strong geometric symmetry — bus (+10.8), tv/monitor (+9.8), bottle (+8.8), car (+6.9), train (+6.2), motorcycle (+5.1), and chair (+4.0). These are precisely the categories where 2D features tend to confuse symmetric sides or visually similar parts, and where PartField descriptors provide the strongest disambiguating signal. By contrast, non-rigid animal categories such as sheep (−2.7), cat (−1.5), and cow (−1.7) show slight regressions, which is expected: PartField is trained with a part-level contrastive objective on rigid objects and generalizes less reliably to deformable shapes. Potted plant similarly shows a marginal decrease (−0.6), likely because SAM3D reconstructs the pot and plant as a single merged shape, whereas evaluation keypoints typically land on the pot alone.

Table C2: Per-category PCK@0.1 scores (per-keypoint) on SPair-71k. Gains are largest for rigid, man-made categories with strong geometric symmetry (bus, tv/monitor, car, motorcycle), where PartField features resolve left–right and repeated-part ambiguities. Non-rigid categories such as animals show little to no improvement.

[Figure 44]

[Figure 45]

[Figure 46]

avg Supervised

MARCO 93.7 79.8 96.9 74.7 75.4 95.2 91.9 94.8 87.5 96.5 91.2 90.3 87.6 63.x 29.x 63.x 51.x 29.x 57.x Unsupervised

DINOv2+NN 72.7 62.0 85.2 41.3 40.4 52.3 51.5 71.1 36.2 67.1 64.6 67.6 61.0 68.2 30.7 62.0 54.3 24.2 55.6 DIFT 63.5 54.5 80.8 34.5 46.2 52.7 48.3 77.7 39.0 76.0 54.9 61.3 53.3 46.0 57.8 57.1 71.1 63.4 57.7

Weakly Supervised with human annotations

Spherical Mapper 75.3 63.8 87.7 48.2 50.9 74.9 71.1 81.7 47.3 81.6 66.9 73.1 65.4 61.8 55.5 70.2 75.0 58.5 67.8 DIY-SC 77.2 69.1 90.8 54.2 57.9 83.7 77.5 86.5 53.1 86.7 73.1 78.5 72.5 74.0 73.5 76.0 77.2 69.5 74.4

Weakly Supervised without human annotations

SD+DINOv2 73.0 64.1 86.4 40.7 52.9 55.0 53.8 78.6 45.5 77.3 64.7 69.7 63.3 69.2 58.4 67.6 66.2 53.5 64.0 Geo-SC 78.0 66.4 90.2 44.5 60.1 66.6 60.8 82.7 53.2 82.3 69.5 75.1 66.1 71.7 58.9 71.6 83.8 55.5 69.6 DIY-SC+OriAny 76.1 65.9 90.4 52.2 57.3 75.7 75.3 85.0 52.8 86.3 71.4 78.3 69.9 73.5 69.2 75.0 76.7 69.6 72.9 3D-SC (Ours) 77.6 70.3 90.4 54.8 66.1 86.5 82.2 83.5 56.8 84.6 72.6 77.8 75.0 72.5 68.6 72.3 82.9 79.4 76.3

###### C.3 Comparison of pseudo-annotations

Figure C1 extends the qualitative comparison from the main paper with additional examples. Across object categories, 3D-SC consistently produces denser pseudo-annotations that cover a larger fraction of the object surface, while remaining geometrically consistent and free from left-right ambiguities. By contrast, DIY-SC pseudo-labels are sparser and more prone to symmetric confusions, as its spherical geometric prior cannot resolve instance-level structure. These qualitative differences are directly reflected in the quantitative gains on SPair-Geo-Aware, which specifically targets symmetric and repeated-part correspondences.

[Figure 47]

[Figure 48]

(a) 3D-SC. (b) DIY-SC.

Figure C1: Qualitative pseudo-annotations. We visualize pseudo-ground-truth annotations from

- 3D-SC and DIY-SC. 3D-SC produces denser and more geometrically consistent pseudo-annotations.

##### D Reproducibility and LLM assistance

To ensure full reproducibility of our work, we will release all code and data used in this paper. The complete processing pipeline, including scripts for dataset preparation will be made publicly available on /GenIntel/3D-SC. Our training and inference code for the proposed model is provided in the same repository, together with configuration files and instructions for reproducing all experiments reported in the paper.

We used large language models (LLMs) in a limited capacity to assist with the writing of this paper and the design of parts of the code. Specifically, LLMs were employed only to (i) improve sentence clarity and conciseness, (ii) condense overly lengthy paragraphs, and (iii) provide coding assistance for implementation design. All technical contributions — including the method design, experimental setup, results, analyses, and final implementation decisions — are entirely our own work.

