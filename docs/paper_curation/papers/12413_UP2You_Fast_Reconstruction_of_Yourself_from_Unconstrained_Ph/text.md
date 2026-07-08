# arXiv:2509.24817v3[cs.CV]22Mar2026

## UP2YOU: FAST RECONSTRUCTION OF YOURSELF FROM UNCONSTRAINED PHOTO COLLECTIONS

Zeyu Cai1,2 Ziyang Li2 Xiaoben Li1 Boqian Li1 Zeyu Wang3 Zhenyu Zhang2† Yuliang Xiu1† 1Westlake University 2Nanjing University 3The Hong Kong University of Science and Technology (Guangzhou) †Shared Corresponding Author Project Page: https://zcai0612.github.io/UP2You

[Figure 1]

Input

Orthogonal View

High-Quality

UP2You Rectify

UP2You Reconstruct

Unconstrained Photos

Generation

Textured Mesh

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

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Figure 1: Overview of UP2You. Our method reconstructs high-quality, textured 3D clothed portraits from unconstrained photo collections. It robustly handles highly diverse and unstructured inputs by rectifying them into orthogonal multi-view images and corresponding normal maps, making them compatible with traditional reconstruction algorithms.

ABSTRACT

We present UP2You, the first tuning-free solution for reconstructing high-fidelity 3D clothed portraits from extremely unconstrained in-the-wild 2D photos. Unlike previous approaches that require “clean” inputs (e.g., full-body images with minimal occlusions, or well-calibrated cross-view captures), UP2You directly processes raw, unstructured photographs, which may vary significantly in pose, viewpoint, cropping, and occlusion. Instead of compressing data into tokens for slow online text-to-3D optimization, we introduce a data rectifier paradigm that efficiently converts unconstrained inputs into clean, orthogonal multi-view images in a single forward pass within seconds, simplifying the 3D reconstruction. Central to UP2You is a pose-correlated feature aggregation module (PCFA), that selectively fuses information from multiple reference images w.r.t. target poses, enabling better identity preservation and nearly constant memory footprint, with more observations. We also introduce a perceiver-based multi-reference shape predictor, removing the need for pre-captured body templates. Extensive experiments on 4D-Dress, PuzzleIOI, and in-the-wild captures demonstrate that UP2You consistently surpasses previous methods in both geometric accuracy (Chamfer-15%↓, P2S-18%↓ on PuzzleIOI) and texture fidelity (PSNR-21%↑, LPIPS-46%↓ on 4D-Dress). UP2You is efficient (1.5 minutes per person), and versatile (supports arbitrary pose control, and training-free multi-garment 3D virtual try-on), making it practical for real-world scenarios where humans are casually captured. Both models and code will be released to facilitate future research on this underexplored task.

1 INTRODUCTION

Reconstructing 3D clothed humans from unconstrained photo collections, like the personal albums (Fig. 2-Left), is a challenging and largely unexplored research frontier. Unlike prior tasks such as single-image 3D reconstruction [23, 48, 65, 92, 93], monocular video-based reconstruction [18, 28, 35], or multi-view 3D reconstruction [55, 64, 101], this problem is distinguished by the highly

unstructured nature of the input: appearance information is present but scattered across photos where subjects are often partially captured or occluded, and camera as well as body poses are rarely synchronized. As a result, establishing accurate 2D-to-3D correspondences is extremely difficult, even with the help of most advanced off-the-shelf human-centric estimators (i.e., camera, body pose, landmarks, geometric cues, etc). In contrast, traditional 3D reconstruction algorithms typically assume “clean captures” (i.e., full-body capture with simple poses, synchronized cameras, etc), where well-aligned 2D-to-3D correspondences can be readily established using the estimators above.

Two potential strategies to address above challenges: 1) Data Compressor: Crop and group photos into local and global patches (e.g., head, full-body) [102], or segment input photos into multiple assets (e.g., garments, hair, face, accessories) [94], then compress these patches or assets into learnable tokens, and finally assemble them as text prompt to generate 3D humans via text-to-3D techniques [62]; 2) Data Rectifier: Convert the incoming “dirty or incomplete captures” into clean and complete ones, e.g., orthogonal orbit views with canonical poses, which are easier to reconstruct with traditional 3D reconstruction algorithms. Essentially, the data compressor operates mainly at the representation level, without substantially improving the generative model’s ability to ensure 3D consistency and identity preservation — a limitation noted in PuzzleAvatar [94] as “unpredictable hallucination.” The data rectifier, however, refines not only the input data but also the generative model’s prior, via continued training on synthetic multi-view renderings of high-fidelity 3D clothed humans, enabling more consistent 3D reconstruction in terms of both identity and viewpoint, from unconstrained photographs. UP2You falls in the second category, as shown in Fig. 2.

PuzzleAvatar [94] is the representative of the first strategy, it first “decompose” the unconstrained photos into multiple asset soups, all of which are linked with unique learned tokens via DreamBooth [70], then it “compose” these assets into a 3D full-body representation via score-distillation sampling (SDS) [62], where the 3D reconstruction task is reformulated as a text-to-3D task, bypassing explicit canonicalization. However, this process takes hours since both DreamBooth fine-tuning and SDS-based optimization are time-consuming and unstable, see Fig. 2. Additionally, ground-truth SMPL-X meshes are needed for initialization, as predicting shape parameters from unconstrained photo collections is non-trivial. Regarding the second strategy — converting inputs into orthogonal orbit views —– some attempts [23, 48, 60] have been made. However, these methods are restricted to single-image inputs and cannot fully leverage the multiple unconstrained photos. Essentially, these methods act more as “data inpainters” [84] — synthesizing unseen views from seen capture — rather than as “data rectifiers” that unify the messy observations into structured output. Designed mainly for constrained inputs (i.e., a single image with full-body coverage), these methods cannot handle unconstrained photos or scale up the reconstruction accuracy with the number of inputs.

To the best of our knowledge, UP2You is the first work to unlock the “data rectifier” strategy on unconstrained photo collections, directly transforming raw unconstrained photo collections into orthogonal views while faithfully preserving subject identity. This is not a trivial extension of prior arts, as it 1) requires effectively aggregating information from multiple unconstrained inputs, which may vary significantly in terms of body poses, camera viewpoints, croppings, and occlusions; 2) must be efficient enough to process varying numbers of input photos (ranging from one to dozens) without incurring significant computational overhead; and 3) needs to overcome the dependency on ground-truth body shapes, which are often unavailable in real-world scenarios.

Specifically, UP2You aggregates ReferenceNet features [27], extracted from unconstrained photos according to body poses, via the proposed Pose-Correlated Feature Aggregation (PCFA) module. This module implicitly learns correlation weights between unconstrained reference images and target pose conditions (i.e., SMPL-X normal maps). Guided by these correlation maps, PCFA uses an optimized topk strategy to selectively aggregate the most informative image features for generating each orthogonal view. As a result, the memory footprint remains nearly constant regardless of the number of input photos, enabling effective and efficient information fusion.

To get rid of the dependence on ground-truth body shapes, we design a shape predictor based on perceiver structure [34, 46] to regress SMPL-X shape parameters directly from unconstrained photo collections. Lastly, with another MV-Adapter [32] to generate multi-view normal maps, followed by mesh carving and texture baking [48], UP2You reconstructs high-quality textured meshes from unconstrained photos in 1.5 minutes. We evaluate our generation results on PuzzleIOI, 4D-Dress, and self-collected in-the-wild datasets. Our method surpasses other state-of-the-art approaches in both geometric accuracy (Chamfer-15%↓, P2S-18%↓ on PuzzleIOI) and texture fidelity (PSNR-21%↑,

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Personal Tokens

SDS Optimization 3 hours

DreamBooth

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

1 hour

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

DB Weights

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

[Figure 56]

Data Rectifier 30 seconds

Reconstruction 1 minute

…

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Personal Unconstrained Photo Collections

Orthogonal View Images & Normals

Figure 2: Paradigm differences between previous works and UP2You. Top: Previous works like PuzzleAvatar [94] and AvatarBooth [102] compress unconstrained photos into implicit personal tokens and DreamBooth weights [70] through fine-tuning, then generate 3D humans via SDS optimization [70]. Bottom: UP2You directly rectifies unconstrained photo collections into orthogonal view images and normals, then reconstructs textured human meshes, achieving superior quality while reducing processing time from 4 hours to 1.5 minutes.

LPIPS-46%↓ on 4D-Dress), while also demonstrating flexibility and superior generalization for single-image reconstruction, and enabling 3D virtual try-on application, all without extra training.

Our main contributions w.r.t. the prior arts are as follows:

- • Efficient. As Fig. 2 shows, unlike previous DreamBooth + SDS paradigm (>4 hours), UP2You acts as a “data rectifier” instead, to directly generate “clean” multi-views from “dirty” unconstrained inputs in one forward pass (<15 secs). It can process one, several, or dozens of photos with a nearly constant memory footprint. The full pipeline, including multi-view normal generation plus mesh carving and texture baking, completes in 1.5 minute.
- • Effective. Thanks to the PCFA module, which selectively aggregates the most informative regions from the reference images for synthesizing target views, UP2You significantly outperforms prior SOTAs (PuzzleAvatar, AvatarBooth, PSHuman) in both geometry accuracy and texture fidelity, and delivers consistent shape and identity regardless of input forms or pose conditions. Notably, the reconstruction quality even scales up with more unconstrained inputs, echoing the principle of The More You See in 2D, the More You Perceive in 3D [21].
- • Versatile. PuzzleAvatar requires an A-posed body template with ground-truth shape for 3D initialization, while UP2You is flexible to random pose control, directly regresses body shapes from unconstrained photos, and inherently supports multi-garment 3D virtual try-on, for free.

- 2 RELATED WORK

- 2.1 3D CLOTHED HUMAN RECONSTRUCTION

The field of 3D clothed human reconstruction has been extensively studied over the past few decades. Early methods primarily focused on reconstructing human geometry and texture from dense multiview image captures [35, 52, 61]. Subsequent research has broadened the scope to include full-shot monocular video inputs [18, 19, 28, 90], enabling more flexible and accessible data acquisition. Recent advances in generative models, particularly diffusion models [25, 42, 69, 81], and the emergence of SDS-based 3D human generators [30, 44, 50, 53, 86, 86], have further propelled the field. An increasing number of video-based human reconstruction approaches now leverage learned generative priors to address common challenges in real-world video captures, such as occlusions [19, 59], view inconsistencies [36], and poor texture details [83].

Such generative priors, learned from large-scale datasets, play a more crucial role for the inherently ill-posed problem of 3D human reconstruction, especially when the input data is sparse or incomplete. The most sparse input format is a single image [23, 31, 33, 48, 65, 67, 71, 72, 92, 93, 107]. In essence, it can be regarded as a “conditional generation” problem [31], since large portions of the geometry — such as the unseen backside and occluded regions — must be plausibly inferred or synthesized from the visible pixels. Building on this “reconstruction as conditional generation” paradigm, numerous works have further advanced the field [3, 15, 48, 105]. Apart from multi-view posed captures, full-shot monocular video, and single image, numerous works have sought to expand the range of input modalities, for example, by incorporating dual front-back captures [38, 55] or multi-view unposed full-body images [29, 66, 97, 101, 108] to improve reconstruction fidelity and completeness.

Despite these advances, existing methods still fall short of handling truly “unconstrained” photos

— those with partial views, occlusions, extreme camera viewpoints, dynamic body poses, and inconsistent aspect ratios. Accurately estimating body shape [12, 43, 45, 89, 98, 104] from such unconstrained photo collections is nearly impossible. Moreover, given multiple “dirty” reference images, image-based HMR methods often fail to deliver consistent results. This inconsistency manifests as significant variations in the predicted body shapes for the same subject — some reconstructions may appear unnaturally thin, others excessively fat, and some may completely fail, especially in cases of partial or occluded inputs. As shown in Tab. 2 and Fig. 7, it becomes challenging to determine which, if any, of the predicted shapes truly represent the subject.

In contrast, UP2You addresses these data format constraints by functioning as a comprehensive “data rectifier,” directly transforming unstructured or “dirty” inputs into orthogonal “clean” views, with consistent 3D and identity, that can be seamlessly utilized for robust 3D reconstruction.

- 2.2 UNCONSTRAINED PHOTOS TO 3D

Most real-world data is inherently unstructured, presenting significant challenges for 3D reconstruction tasks that require reliable spatial correspondences. The earliest work in “Unconstrained Photos to 3D” can be traced back to Photo Tourism [80], which reconstructs 3D scenes from large collections of Internet photos. Recent advances in neural rendering and generative models have further advanced this field, enabling more robust and realistic 3D reconstructions from unstructured image collections [10, 49, 87]. However, these methods primarily focus on rigid objects or scenes and cannot be directly applied to 3D clothed human reconstruction, which involves highly articulated and non-rigid structures. A critical open question is how to effectively extract and aggregate identity features from unconstrained photos — not only for general objects [41, 103, 109], but especially for dynamic humans — and reproduce them in a 3D-consistent manner. Several works on subject-driven image generation [2, 4, 13, 14, 16, 40, 70, 73, 85], as well as ID-consistent 2D human portrait generation [9, 63, 75, 84, 95], are discussed in the Sup.Mat. (Sec. B). However, these methods are primarily designed for 2D image generation and lack the mechanisms to ensure cross-view consistency or the precise latent feature aggregation necessary for high-fidelity 3D reconstruction.

The most relevant works addressing this challenge are PuzzleAvatar [94] and AvatarBooth [102]. Both first employ few-shot personalization [4, 70], as Total Selfie [9] and RealFill [84], to distill identity information from unconstrained photos into a customized diffusion model, as unique tokens. Subsequently, guided by these unique tokens, they utilize Score Distillation Sampling (SDS) [7, 37, 62, 100] to optimize a neural-based 3D representation [56, 76]. In short, the entire pipeline of these methods can be summarized as “unconstrained photos → personalized diffusion models with learned specialized tokens → SDS-based Text-to-3D”. However, fine-tuning diffusion models and optimization-based SDS methods are extremely time-consuming. Moreover, these fine-tuning approaches act as a form of lossy compression: the strong priors of diffusion models often override subject-specific features, leading to a loss of identity and fine-grained details, or even introducing unpredictable hallucinations. In contrast, UP2You is a tuning-free method that faithfully reconstructs

- 3D humans from unconstrained photos in just 1.5 minutes, while well preserving human identities.

- 3 METHOD

Our objective is to reconstruct a high-quality textured mesh from unconstrained photos with unknown camera parameters and human poses. To this end, we first generate orthogonal full-body images from the unconstrained inputs, conditioned on SMPL-X normal maps that contain both camera and pose information (Sec. 3.1). Next, we utilize these orthogonal multi-view RGB images to generate corresponding multi-view normal maps, which serve as geometric cues for detailed mesh reconstruction (Sec. 3.2). To handle in-the-wild images without SMPL-X annotations, we further introduce a body shape estimator capable of inferring human body shape by integrating information from a handful of unconstrained photos (Sec. 3.3).

3.1 ORTHOGONAL MULTI-VIEW IMAGES GENERATION

To tackle orthogonal multi-view image generation from unconstrained photo collections, we adopt MV-Adapter [32] as our backbone (introduced in Sec. C). MV-Adapter integrates ReferenceNet

|<br><br>|
|---|

|<br><br>|
|---|

|<br><br>|
|---|

|<br><br>|
|---|

|<br><br><br><br><br><br><br><br><br><br><br><br><br><br>|
|---|

|<br><br><br><br><br><br><br><br><br><br><br><br><br><br>|
|---|

|<br><br><br><br><br><br><br><br><br><br><br><br><br><br>|
|---|

Shape Prediction (Sec 3.3)

ReferenceNet

|<br><br>|
|---|

|<br><br>|
|---|

|<br><br>|
|---|

|<br><br>|
|---|

|<br><br><br><br><br><br><br><br><br><br><br><br><br><br>|
|---|

|<br><br><br><br><br><br><br><br><br><br><br><br><br><br>|
|---|

|<br><br><br><br><br><br><br><br><br><br><br><br><br><br>|
|---|

|PCFA|
|---|

P

Correlation Maps C (Sec 3.1)

Unconstrained Photos I

Feature Selection (Sec 3.1)

Pose Guider

Sec 3.2

|MVDiffusion Model|
|---|

Rendering

Noise

Target Poses P

Generated Images V

Generated Normals N

Normal Maps Generation (Sec 3.2)

##### Published as a conference paper at ICLR 2026

[Figure 170]

[Figure 171]

|[Figure 172]<br><br>[Figure 173]|
|---|

|[Figure 174]<br><br>[Figure 175]|
|---|

|[Figure 176]<br><br>[Figure 177]|
|---|

|[Figure 178]<br><br>[Figure 179]|
|---|

|[Figure 180]<br><br>[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]<br><br>[Figure 184]<br><br>[Figure 185]<br><br>[Figure 186]<br><br>[Figure 187]|
|---|

|[Figure 188]<br><br>[Figure 189]<br><br>[Figure 190]<br><br>[Figure 191]<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]|
|---|

|[Figure 196]<br><br>[Figure 197]<br><br>[Figure 198]<br><br>[Figure 199]<br><br>[Figure 200]<br><br>[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]|
|---|

Shape Prediction (Sec 3.3)

ReferenceNet

[Figure 204]

|[Figure 205]<br><br>[Figure 206]|
|---|

|[Figure 207]<br><br>[Figure 208]|
|---|

|[Figure 209]<br><br>[Figure 210]|
|---|

|[Figure 211]<br><br>[Figure 212]|
|---|

|[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]<br><br>[Figure 219]<br><br>[Figure 220]|
|---|

|[Figure 221]<br><br>[Figure 222]<br><br>[Figure 223]<br><br>[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]|
|---|

|[Figure 229]<br><br>[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]<br><br>[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]<br><br>[Figure 236]|
|---|

Correlation Maps C (Sec 3.1)

[Figure 237]

Unconstrained Photos I

[Figure 238]

[Figure 239]

[Figure 240]

Feature Selection (Sec 3.1)

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

[Figure 255]

Pose Guider

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

Sec 3.2

[Figure 262]

|MVDiffusion Model|
|---|

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Rendering

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Noise

Target Poses P

Generated Images V

Normal Maps Generation (Sec 3.2)

Generated Normals N

Figure 3: Pipeline of UP2You. Given unconstrained input photos I, we first predict the SMPL-X shape parameters (Sec. 3.3) and initialize the SMPL-X mesh with predefined pose and expression parameters. We then generate orthogonal view images V based on I and SMPL-X normal rendering P with the proposed PCFA method—predict correlation maps C and select most informative features (Sec. 3.1). Finally, we produce multi-view normal maps N from P and V, and reconstruct the final textured mesh (Sec. 3.2).

R [27] as the reference image encoder and incorporates raymaps into the diffusion UNet as view conditions, enabling the synthesis of six orthogonal views. For our task, we use orthogonal SMPL-X normal maps as view conditions. Unlike the original MV-Adapter, which handles only single-image inputs, our approach extends it to process multiple unconstrained photos.

As shown in Fig. 3, given N unconstrained reference images I = {I1,...,IN} of a person in the same outfit, our goal is to synthesize M orthogonal target ,...,VM}, each conditioned on a corresponding normal map P = {P1,...,PM}. To extract th e features for each target view, we introduce d Feature Aggregation (PCFA) module, which correlation maps C = {Ci1,...,CiN}Mi=1 between reference and target views (see Fig. 4). Based on C, PCFA select features for each target viewpoint for the generation of orthogonal views V.

|[Figure 278]<br><br>[Figure 279]<br><br>[Figure 280]<br><br>[Figure 281]<br><br>[Figure 282]<br><br>[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]<br><br>[Figure 286]<br><br>Correlation Map<br><br>[Figure 287]|
|---|

Query Proj.

[Figure 288]

|[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]<br><br>[Figure 292]<br><br>[Figure 293]<br><br>[Figure 294]<br><br>[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]<br><br>Correlation Map<br><br>[Figure 298]<br><br>views V = {V1 SMPL-X the most informative the Pose-Correlated predicts|
|---|

[Figure 299]

Transformer Blocks

[Figure 300]

[Figure 301]

&

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

Query Proj.

[Figure 306]

Transformer Blocks

Pose Encoder

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Key Proj.

[Figure 311]

Correlation Map Prediction. Using all reference features for ortho-view generation is computationally intensive, as memory usage grows with the number of unconstrained references. However, many reference pixels are irrelevant for a given target view (e.g., back-view references for front-view synthesis). Therefore, we adaptively determine each reference’s contribution based on the target pose to reduce computational cost.

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

|[Figure 316]<br><br>[Figure 317]|
|---|

|[Figure 318]<br><br>[Figure 319]|
|---|

|[Figure 320]<br><br>[Figure 321]|
|---|

|[Figure 322]<br><br>[Figure 323]|
|---|

[Figure 324]

Key Proj.

Pose Encoder

[Figure 325]

|[Figure 326]<br><br>[Figure 327]|
|---|

|[Figure 328]<br><br>[Figure 329]|
|---|

|[Figure 330]<br><br>[Figure 331]|
|---|

|[Figure 332]<br><br>[Figure 333]|
|---|

DINO Features Xref

Target Pose

[Figure 334]

[Figure 335]

[Figure 336]

|[Figure 337]<br><br>[Figure 338]|
|---|

|[Figure 339]<br><br>[Figure 340]|
|---|

|[Figure 341]<br><br>[Figure 342]|
|---|

|[Figure 343]<br><br>[Figure 344]|
|---|

Figure 4: Pose-Dependent Correlation Map. Correlation is colored as Higher → Lower.

|[Figure 345]<br><br>[Figure 346]|
|---|

|[Figure 347]<br><br>[Figure 348]|
|---|

|[Figure 349]<br><br>[Figure 350]|
|---|

|[Figure 351]<br><br>[Figure 352]|
|---|

To achieve this, we disentangle human-specific identity features from viewpoint correlation information in the unconstrained reference inputs. Drawing inspiration from [26, 39], we predict correlation maps for reference images conditioned on target poses, as illustrated in Fig. 4. For each target pose Pi,i ∈ {1,2,...,M}, we estimate a correlation map that indicates the pixel-wise relevance of each reference image for generating the corresponding view. Specifically, we employ a pose image encoder Epose and a DINOv2 [57] model Eref to extract features from the target pose image and all reference images: Xposei = Epose(Pi) and Xref = Eref(I), where Xref represents the concatenation of all DINOv2 outputs {Xrefj }Nj=1. Subsequently, we feed both Xposei and Xref into a transformer block T that comprises layers of self-attention and cross-attention, where Xposei functions as the query, key, and value in self-attention operations, and as the query in cross-attention operations, while Xref serves as both key and value in cross-attention operations. Through T , an output feature Oi = T (Xposei ,Xref) that integrates reference information relevant to the target pose is produced. We derive the image correlation map Ci by computing the attention map between Oi and Xref:

DINO Features Xref

Target Pose

Ai = WqOi × WkXref⊤

, (1)

√

d

#### Ci = [Ci1,Ci2,...,CiN]

(2)

= ReLU(AvgPool(mean(Ai))),

Here, Wq and Wk are learnable projection matrices applied to Oi and Xref, respectively. The resulting attention map Ai ∈ Rl×Nhw captures the relevance between the target pose and reference features. To obtain the final reference correlation scores, we compute the mean along the first dimension of Ai using mean(·) : Rl×Nhw → RNhw. In this context, l is the token number of Oi, h and w denote the height and width of Xref, and d is the feature dimension of WqOi and WkXref. We further apply AvgPool to smooth the predicted correlation map and ReLU to suppress negative values.

The correlation maps of PCFA are based on fine-grained semantic correlation between target bodies and DINO features of references. Unlike previous methods [26, 39] that depend on landmark similarity, our correlation map encodes richer outfit details, enabling more accurate reconstruction.

Feature Selection. The predicted correlation maps enable PCFA to selectively aggregate the most informative reference features for each target view. Specifically, we utilize ReferenceNet R as the reference image encoder to extract multi-scale reference features F = {F1,F2,...,FL}, where L is the number of layers. For each target pose Pi and the reference feature Fk ∈ RNS

k×c at layer k,

we first interpolate the corresponding correlation map Ci ∈ RNhw to get Cˆi = Interpk(Ci) that aligns with the spatial dimensions of Fk. Here Sk denotes the spatial size of Fk, and Interpk(·) : RNhw → RNS

k denotes the interpolation operator.

We then select the most relevant reference features Fˆik for view Pi based on Cˆi. Specifically, we employ the topk selection strategy to obtain the selected indices of Fk:

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

ReferenceNet

#### [k1i,k2i,...,kγi Sk] = sort(topk(Cˆi)[: γSk]), (3)

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

###### V

[Figure 365]

[Figure 366]

where [k1i,k2i,...,kγi Sk] are the indices of the selected features, topk(·) returns the

indices, and γ controls the proportion of features retained. To preserve spatial order, we apply sort(·). Using these indices, we extract the selected reference features Fˆik ∈ RγSk×c:

[Figure 367]

|topMVDiffusionγSk Model|
|---|

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

Noise

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

Generated Normals N

Pose Guider

[Figure 383]

P

[Figure 384]

Fˆik = Fk[k1i,k2i,...,kγi Sk] · Cˆi[k1i,k2i,...,kγi Sk]. (4)

Given the aggregated reference features Fˆ = {Fˆ1k,Fˆ2k,...,FˆMk }Lk=1, we synthesize the orthogonal multi-view images as V = Drgb(Fˆ,Prgb(P)), where Drgb is our multi-view image generation model and Prgb(·) is the pose guider that encodes the pose condition into Drgb.

- 3.2 NORMAL MAP GENERATION AND MESH RECONSTRUCTION

For multi-view reconstruction (MVS) [51, 54, 91], we generate multi-view clothed normal maps N from the generated images V, conditioned on target poses P, and reconstruct the mesh using both V and N.

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

V

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

Normal Map Generation. To ensure multi-view consistency and provide strong geometric cues for normal map generation, we follow [93] and incorporate SMPL-X normal renderings as additional conditions. As Fig. 5 shows, we also adopt MV-Adapter as the backbone of clothed normal generator Dnormal. We utilize the generated orthogonal RGB views V as reference inputs, and employ the pose guider Pnormal(·) to incorporate multi-view pose conditions. The multi-view clothed normal maps are then generated via N = Dnormal(V,Pnormal(P)).

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

Noise

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

Generated Normals N

[Figure 416]

P

[Figure 417]

Figure 5: Normal Map Generation Pipeline. The main input difference with Fig. 3 is the generated multi-view orthogonal images V, instead of unconstrained inputs I.

Mesh Carving and Texture Baking. Starting from the initial SMPLX mesh, we refine mesh details using the generated N and project per-vertex colors from V, following PSHuman [48]. To better preserve hand geometry, we replace the hand region with that from the initial mesh as in ECON [93], and then perform texture baking using the generated multi-view RGB images.

| |PuzzleIOI<br><br>| |4D-Dress<br><br>| |in-the-wild|
|---|---|---|---|---|---|
| |PSNR↑ SSIM↑ LPIPS↓<br><br>|Chamfer↓ P2S↓ Normal↓<br><br>|PSNR↑ SSIM↑ LPIPS↓<br><br>|Chamfer↓ P2S↓ Normal↓<br><br>|CLIP-I↑ DINO↑<br><br>|
|AvatarBooth PuzzleAvatar Ours (Image) Ours (Mesh)<br><br>|16.879 0.860 0.1544 21.664 0.916 0.0639<br><br>23.896 0.926 0.0545<br><br>24.539 0.940 0.0474<br><br><br>|6.635 6.697 0.0274 3.204 3.165 0.0150 - - 2.724 2.605 0.0115<br><br>|18.186 0.850 0.1718 21.376 0.887 0.1081 25.848 0.920 0.0576 25.540 0.918 0.0654<br><br>|6.846 6.978 0.0311 1.956 2.045 0.0170 - - 1.140 1.119 0.0122<br><br>|0.878 0.619 0.907 0.742 0.972 0.932 0.971 0.916<br><br>|

Table 1: Quantitative Comparison with Baselines. UP2You achieves the best texture fidelity, geometry accuracy, and perception similarity.

- 3.3 MULTI-REFERENCE SHAPE PREDICTOR

The initial SMPL-X mesh is critical to the entire UP2You pipeline, as it provides the pose condition P for multi-view generation and serves as the basis for mesh reconstruction. SMPL-X mesh T ∈ R10754×3 are defined as T(β,θ,ψ), where β,θ,ψ are shape, pose, and expression parameters respectively. While the target pose and expression of the SMPL-X template can be predefined (e.g., T-pose or A-pose with neutral expression), the body shape parameters must be estimated from unconstrained input images. Existing shape predictors [23, 48, 65] are typically designed for single-image scenarios and struggle to effectively leverage multiple unconstrained references.

Perceiver Blocks

Shape Head

. . .

[Figure 418]

DINO Features Xref

|[Figure 419]<br><br>[Figure 420]|
|---|

|[Figure 421]<br><br>[Figure 422]|
|---|

|[Figure 423]<br><br>[Figure 424]|
|---|

|[Figure 425]<br><br>[Figure 426]|
|---|

|[Figure 427]<br><br>[Figure 428]|
|---|

|[Figure 429]<br><br>[Figure 430]|
|---|

|[Figure 431]<br><br>[Figure 432]|
|---|

|[Figure 433]<br><br>[Figure 434]|
|---|

Learnable Shape Tokens

[Figure 435]

. . .

Shape Parameters

[Figure 436]

[Figure 437]

Figure 6: Multi-reference Shape Predictor.

To address this limitation, we introduce a multireference shape predictor, S, as illustrated in Fig. 6. The prediction process is formulated as βpred = S(τ,Xref), where βpred denotes the predicted shape parameters, τ

are learnable query tokens, and Xref are DINOv2 features extracted from the reference images. Our shape predictor S employs a perceiver-style architecture [34, 46] that can use query tokens to effectively aggregate multiview information. The prediction head is a lightweight transformer, similar to the camera head design in [87].

Overall, through the shape predictor, multi-view image & normal generator, and mesh carving & texture baking steps, UP2You generates textured 3D humans from unconstrained photo inputs. See the detailed flowchat in Sup.Mat.’s Sec. D.5.

- 4 EXPERIMENTS

- 4.1 SETTINGS

Dataset. We train our multi-view image generation, normal map generation, and shape prediction models on the THuman2.1 [99], Human4DiT [74], 2K2K [20], and CustomHumans [24] datasets. For evaluation, we use the PuzzleIOI [94] and 4D-Dress [88] datasets as test sets. To further validate our approach, we collect an in-the-wild (in-the-wild) dataset comprising 12 distinct identities. Details on dataset selection and processing procedures are provided in Sec. D.2.

Baselines. We comprehensively compare UP2You with 1) album-to-human reconstruction methods, including PuzzleAvatar [94] and AvatarBooth [102]. Since single-view reconstruction is a special case of the unconstrained setting, we also include the leading 2) single-view method, PSHuman [48], in our comparisons. To ensure fair evaluation and isolate the impact of pose estimation errors, we provide ground truth SMPL-X parameters for all baseline methods. 3) For shape prediction, we present the first approach to estimate SMPL-X shape parameters from multiple unconstrained inputs. We compare our shape predictor with two single-input methods: Semantify [17], which is specifically designed for shape prediction, and PromptHMR [89], a state-of-the-art human mesh recovery method. Unless stated otherwise, results on PuzzleIOI and 4D-Dress use 12 reference images. 4) For in-the-wild, we use all available references (8–12) for each identity. Additional model and training details are in Sup.Mat.’s Secs. D.1 and D.3.

Metrics. For PuzzleIOI and 4D-Dress (with textured 3D GT), we report geometric metrics (Chamfer, P2S, Normal map L2) and image quality metrics (PSNR, SSIM, LPIPS). For in-the-wild, we use perceptual similarity (CLIP-I, DINO) between generated and frontal reference. Shape prediction is assessed by vertex-to-vertex (V2V) distance on all datasets. More details in Sup.Mat.’s Sec. D.4.

- 4.2 COMPARISONS

Quantitative Results. The quantitative results in Tab. 1 show that UP2You consistently surpasses all baselines across both 2D and 3D evaluation metrics on the PuzzleIOI and 4D-Dress datasets.

Inputs Ours PuzzleAvatar GT

AvatarBooth

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

…

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

…

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

…

[Figure 471]

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

…

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

…

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

…

Figure 8: Qualitative Comparisons on PuzzleIOI and 4D-Dress. See more 360-degree results in Sup.Mat.’s video.

Importantly, UP2You also achieves strong perceptual quality scores on the in-the-wild dataset, demonstrating its robustness and effectiveness in real-world unconstrained scenarios.

For single-view reconstruction, Tab. 3 shows that UP2You outperforms PSHuman on all 2D and 3D metrics. This is expected, as single front-view input is a special case of the unconstrained multi-view scenario for which UP2You is designed. Training on the more challenging unconstrained task enables our model to generalize well and excel in the simpler constrained setting.

[Figure 504]

[Figure 505]

Low High

Ours

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

Semantify

As shown in Tab. 2 and Fig. 7, our shape predictor outperforms single-view methods [17, 89], achieving more accurate and consistent results. Single-input baselines show high variance and instability, especially with partial input or failed detections. Leveraging multiple inputs, our method delivers more robust shape prediction, with performance further improving as more unconstrained references are used. Furthermore, Table 2 also shows that the perceiver transformer architecture is better than simple MLPs for the shape predictor.

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

PromptHMR

Failed

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

Inputs

Figure 7: Shape Prediction Error Map.

|Num of References|Semantify|PromptHMR<br><br>|MLP<br><br>|Ours|
|---|---|---|---|---|
| |Mean↓ Var↓<br><br>|Mean↓ Var↓<br><br>| | |
|3 6 9 12|11.087 4.234 11.066 5.706<br><br>10.978 6.424<br><br>11.097 6.597<br>|9.212 10.370 9.661 17.465 9.403 18.218 9.287 19.418<br><br>|8.819 8.046 8.275 8.336<br><br>|7.967 7.427 7.403 7.399|

| |PSNR↑ SSIM↑ LPIPS↓<br><br>|Chamfer↓ P2S↓ Normal↓<br><br>|
|---|---|---|
|PSHuman Ours Mesh|24.134 0.905 0.0895 26.651 0.935 0.0527<br><br>|2.759 2.926 0.0189 0.927 0.949 0.0096|

Table 3: Comparison of Single-Image based Reconstruction.

Table 2: V2V(↓) Comparions of Shape Prediction Reuslts.

Qualitative Results. The qualitative comparisons in Fig. 8 and Fig. 9 show that UP2You achieves high-fidelity, reference-faithful 3D reconstructions with strong realism and detail preservation. In contrast, baselines like AvatarBooth and PuzzleAvatar often fail to capture fine facial details and produce blurrier, less realistic results with poor subject-specific consistency. Figure 10 shows single-

Inputs Ours AvatarBooth PuzzleAvatar Reference

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

Input

[Figure 531]

…

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

PSHuman

…

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

Ours

…

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

Figure 9: Qualitative Comparisons on in-the-wild Data.

Figure 10: UP2You vs. PSHuman.

[Figure 575]

…

view 3D human reconstruction comparisons. Our method generalizes well to single-view inputs, producing visually comparable results to PSHuman, but with more accurate limb reconstruction due to consistent multi-view guidance. More visual comparisons and results are in Secs. E.1 and G.

| |Feature Aggregation| |Image Encoder|PuzzleIOI<br><br>|4D-Dress|
|---|---|---|---|---|---|
| |Mean Concat Corr.|sum topk<br><br>|CLIP DINOv2 Ref Net|PSNR↑ SSIM↑ LPIPS↓<br><br>|PSNR↑ SSIM↑ LPIPS↓<br><br>|
|Ours<br><br>A.<br><br>B.<br><br>C.<br><br>D.<br><br>E.<br><br><br>|✗ ✗ ✓<br><br>✓ ✗ ✗<br><br>✗ ✓ ✗<br>✗ ✗ ✓<br>✗ ✗ ✓<br>✗ ✗ ✓<br>|✗ ✓<br><br>✗ ✗<br><br>✗ ✗<br><br>✓ ✗<br><br>✗ ✓<br><br>✗ ✓<br><br><br>|✗ ✗ ✓<br>✗ ✗ ✓<br>✗ ✗ ✓<br>✗ ✗ ✓<br><br>✓ ✗ ✗<br><br>✗ ✓ ✗<br>|23.896 0.926 0.0545 17.412 0.864 0.1227 20.545 0.893 0.0949 20.167 0.889 0.1002 20.152 0.891 0.0976 19.744 0.886 0.1415<br><br>|25.848 0.920 0.0576 19.614 0.876 0.1098 23.366 0.901 0.0791 23.412 0.904 0.0794 23.405 0.903 0.0801 23.393 0.904 0.0813|

Table 4: Ablation Studies of our orthogonal view image generation model.

- 4.3 ABLATION STUDIES

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

Multi-View Image Generation. In Tab. 4, we analyze our multiview image generation model on the PuzzleIOI and 4D-Dress datasets. For feature aggregation, we compare simple averaging (A), concatenation (B), and our proposed PCFA, which achieves the best results. We also test a weighted sum strategy (C) after correlation map prediction. For reference feature extraction, we evaluate CLIP (D), DINOv2 (E), and ReferenceNet. Quantitative and visual results (Sec. F.1) show our design outperforms all alternatives.

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

Figure 11: Predicted Correlation Maps. See dynamic illustration of correlation maps in Sup.Mat.’s video.

Correlation Maps. Our correlation map prediction module identifies and prioritizes key regions in reference images based on the target pose. As shown in Figure 11, visualizations for front- and back-view targets confirm that our maps effectively select the most relevant areas for view generation. This targeted focus improves generation quality and reduces GPU memory usage by retaining only the most informative features. More visual results are shown in Sec. E.2.

Number of References. In Tab. 5, quality improves as more unconstrained references are used. PCFA module efficiently selects informative features, keeping GPU memory usage low, unlike direct concatenation, which increases memory linearly.

| |Ours|Concat|
|---|---|---|
| |PSNR↑ SSIM↑ LPIPS↓ GPU↓<br><br>|PSNR↑ SSIM↑ LPIPS↓ GPU↓<br><br>|
|3 refs 6 refs 9 refs 12 refs<br><br>|24.159 0.912 0.0680 18.65<br><br>25.041 0.917 0.0623 19.40 25.646 0.918 0.0592 20.16 25.848 0.920 0.0576 20.88<br><br><br>|22.759 0.897 0.0894 18.02<br>23.267 0.901 0.0807 24.33 23.362 0.901 0.0796 30.89 23.366 0.901 0.0791 37.96<br>|

Table 5: Multi-View Generation with Different Number of References.

Robustness to Inputs & Conditions. The generated human identity remains consistent across different target poses and reference combinations, with detailed discussions and results presented in Sup.Mat.’s Sec. F.2 and Sec. F.3. Notably, UP2You can effectively handle subjects with loose clothing and complex target poses, as demonstrated in Fig. 24 of Sec. F.2.

Image Encoder of PCFA. Given that DINOv2 has been demonstrated to effectively capture 2D-to-3D correspondences [58], we adopt it as the image encoder for our PCFA module. To further validate this design choice, we conduct additional experiments on the 4D-Dress dataset using alternative image encoders, including CLIP [68] and DINOv1 [8]. As presented in Tab. 6, DINOv2 consistently outperforms both alternatives on multi-view image generation quality.

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

|[Figure 614]|
|---|

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

Inputs Results

Inputs Results

Figure 12: Generation Results on Highly Occluded Inputs.

Figure 13: Generation Results on Missing Part.

| |PSNR↑ SSIM↑ LPIPS↓<br><br>|
|---|---|
|DINOv2 CLIP DINOv1<br><br>|25.848 0.920 0.0576<br><br>23.876 0.904 0.0767<br>24.170 0.907 0.0745<br>|

| |PSNR↑ SSIM↑ LPIPS↓ GPU↓<br><br>|
|---|---|
|γ = 1.0<br><br>γ = 2.0<br>γ = 3.0<br>|24.978 0.912 0.0665 18.64<br><br>25.848 0.920 0.0576 20.88<br><br><br>25.837 0.920 0.0569 23.12|

Table 7: Comparison of Different Number of Selected Features.

Table 6: Comparison of Different Image Encoders for PCFA.

Number of Selected Features. We set the default value of γ to 2.0 to control the number of reference features selected in the topk selection. To determine the optimal configuration, we evaluate different values of γ as shown in Table 7. The results demonstrate that γ = 2.0 achieves the best trade-off between generation quality and GPU memory efficiency.

Unconstrained Inputs vs. Single Front View. Compared to single full-body front-view inputs, unconstrained photos are easier to collect and capture richer information about side and back views. Using the comprehensive information from unconstrained photos leads to better reconstruction results. Table 8 compares UP2You against standard single front-view based methods on 4D-Dress dataset, including ICON [92], ECON [93], PIFuHD [72], PSHuman [48], and Human3Diff [96]. Our method achieves the best performance in both rendering quality and 3D accuracy, particularly for back-view rendering results, demonstrating the value of unconstrained inputs.

| |PSNR↑ SSIM↑ LPIPS↓ Front Back Front Back Front Back<br><br>|Chamfer↓ P2S↓ Normal↓<br><br>|
|---|---|---|
|Ours* PSHuman Human3Diff ICON ECON PIFuHD<br><br>|25.257 25.488 0.906 0.909 0.0724 0.0733 25.384 23.382 0.898 0.885 0.0934 0.1121 23.335 20720 0.883 0.872 0.1118 0.1248<br><br>- - - - - -<br>- - - - - -<br>- - - - - -<br>|1.140 1.119 0.0122<br><br>2.756 2.926 0.0189<br><br>4.275 4.322 0.0227 4.352 4.331 0.0188<br><br>3.780 3.642 0.0178<br><br><br>2.776 2.603 0.0154|

Table 8: Unconstrained Photos vs. Single Front View. * indicates our method uses unconstrained photos input, while other methods use single full-body front view input.

Generated Results in Extreme Situation. UP2You is robust to input variations and can effectively extract information from highly occluded photos. Figure 12 presents an example where one input image captures only the foot region, while other images lack this body part, demonstrating the capability of our method to handle inputs with high occlusion ratios. Figure 13 further examines scenarios where body parts are not fully visible across all images (e.g. the foot region). Due to diffusion hallucination, the generated results exhibit a somewhat reasonable structure; however, the texture is blended from other visible parts (more cases shown in Sup.Mat.’s Fig. 29). Therefore, inputs with complete body part coverage are more suitable for UP2You to achieve optimal results.

Animation. Since we adopt the A-Pose as the default target pose, the reconstructed mesh is naturally suited for animation. The textured mesh generated by UP2You can be easily animated using thirdparty tools such as Mixamo [6]. Moreover, the aligned SMPL-X parameters provided by UP2You enable animation based on skin weight transfer [1]. Finally, as UP2You can transform unconstrained inputs into different target pose configurations, animated rendering results can also be directly performed by itself, as demonstrated in Sup.Mat.’s Fig. 21.

- 5 CONCLUSION

UP2You acts as a “data rectifier,” converting unconstrained photos into orthogonal views suitable for MVS. It is efficient (1.5 minutes per person on one GPU), achieves SOTA quality, and well preserves identity and clothing style across diverse input forms and pose conditions. It also enables free 3D virtual try-on (Fig. 14, more in Sup.Mat.’s Fig. 30). Limitations and future work are discussed in Sup.Mat.’s Sec. H.

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

…

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

…

Figure 14: 3D Virtual Try-On.

ACKNOWLEDGEMENTS

We thank Siyuan Yu for the help in Houdini Simulation, Shunsuke Saito, Dianbing Xi, Yifei Zeng for the fruitful discussions, and the members of Endless AI Lab for their help on data capture and discussions. This work is funded by the Research Center for Industries of the Future (RCIF) at Westlake University, the Westlake Education Foundation.

REFERENCES

- [1] Rinat Abdrashitov, Kim Raichstat, Jared Monsen, and David Hill. Robust skin weights transfer via weight inpainting. In SIGGRAPH Asia 2023 Technical Communications, 2023. 10, 26
- [2] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time representation for text-to-image personalization. Transactions on Graphics (TOG), 2023. 4, 19
- [3] Badour AlBahar, Shunsuke Saito, Hung-Yu Tseng, Changil Kim, Johannes Kopf, and Jia-Bin Huang. Single-image 3d human digitization with shape-guided diffusion. In International Conference on Computer Graphics and Interactive Techniques in Asia (SIGGRAPH Asia),

2023. 3

- [4] Omri Avrahami, Kfir Aberman, Ohad Fried, Daniel Cohen-Or, and Dani Lischinski. Breaka-scene: Extracting multiple concepts from a single image. In International Conference on Computer Graphics and Interactive Techniques in Asia (SIGGRAPH Asia), 2023. 4, 19
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 19
- [6] Sue Blackman. Rigging with mixamo. In Unity for Absolute Beginners. Springer, 2014. 10, 26
- [7] Zeyu Cai, Duotun Wang, Yixun Liang, Zhijing Shao, Ying-Cong Chen, Xiaohang Zhan, and Zeyu Wang. Dreammapping: High-fidelity text-to-3d generation via variational distribution mapping. In Pacific Conference on Computer Graphics and Applications (PG), 2024. 4
- [8] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In International Conference on Computer Vision (ICCV), 2021. 9
- [9] Bowei Chen, Brian Curless, Ira Kemelmacher-Shlizerman, and Steven M Seitz. Total selfie: generating full-body selfies. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 4, 19
- [10] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. Mast3r-sfm: a fully-integrated solution for unconstrained structure-frommotion. In International Conference on 3D Vision (3DV), 2025. 4
- [11] Facebook. DINOv2-Large. https://huggingface.co/facebook/ dinov2-large, 2023. 20
- [12] Yao Feng, Vasileios Choutas, Timo Bolkart, Dimitrios Tzionas, and Michael J Black. Collaborative regression of expressive bodies using moderation. In International Conference on 3D Vision (3DV), 2021. 4
- [13] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In European Conference on Computer Vision (ECCV), 2024. 4, 19
- [14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022. 4, 19
- [15] Xiangjun Gao, Xiaoyu Li, Chaopeng Zhang, Qi Zhang, Yanpei Cao, Ying Shan, and Long Quan. Contex-human: Free-view rendering of human from a single image with textureconsistent synthesis. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3

- [16] Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, and Tali Dekel. Tokenverse: Versatile multi-concept personalization in token modulation space. Transactions on Graphics (TOG), 2025. 4, 19
- [17] Omer Gralnik, Guy Gafni, and Ariel Shamir. Semantify: Simplifying the control of 3d morphable models using clip. In International Conference on Computer Vision (ICCV), 2023. 7, 8
- [18] Chen Guo, Tianjian Jiang, Xu Chen, Jie Song, and Otmar Hilliges. Vid2avatar: 3d avatar reconstruction from videos in the wild via self-supervised scene decomposition. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 3
- [19] Chen Guo, Junxuan Li, Yash Kant, Yaser Sheikh, Shunsuke Saito, and Chen Cao. Vid2avatarpro: Authentic avatar from videos in the wild via universal prior. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [20] Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang, Young-Jae Park, and Hae-Gon Jeon. High-fidelity 3d human digitization from single 2k resolution images. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 7, 20
- [21] Xinyang Han, Zelin Gao, Angjoo Kanazawa, Shubham Goel, and Yossi Gandelsman. The more you see in 2d the more you perceive in 3d. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2016. 20
- [23] Xu He, Zhiyong Wu, Xiaoyu Li, Di Kang, Chaopeng Zhang, Jiangnan Ye, Liyang Chen, Xiangjun Gao, Han Zhang, and Haolin Zhuang. Magicman: Generative novel view synthesis of humans with 3d-aware diffusion and iterative refinement. In AAAI Conference on Artificial Intelligence, 2025. 1, 2, 3, 7
- [24] Hsuan-I Ho, Lixin Xue, Jie Song, and Otmar Hilliges. Learning locally editable virtual humans. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 7, 20
- [25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Conference on Neural Information Processing Systems (NeurIPS), 2020. 3, 19
- [26] Fa-Ting Hong, Zhan Xu, Haiyang Liu, Qinjie Lin, Luchuan Song, Zhixin Shu, Yang Zhou, Duygu Ceylan, and Dan Xu. Free-viewpoint human animation with pose-correlated reference selection. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 5, 6
- [27] Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 2, 5, 19
- [28] Liangxiao Hu, Hongwen Zhang, Yuxiang Zhang, Boyao Zhou, Boning Liu, Shengping Zhang, and Liqiang Nie. Gaussianavatar: Towards realistic human avatar modeling from a single video via animatable 3d gaussians. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3
- [29] Han Huang, Liliang Chen, and Xihao Wang. Unconfuse: avatar reconstruction from unconstrained images. In European Conference on Computer Vision (ECCV), 2022. 3
- [30] Xin Huang, Ruizhi Shao, Qi Zhang, Hongwen Zhang, Ying Feng, Yebin Liu, and Qing Wang. Humannorm: Learning normal diffusion model for high-quality and realistic 3d human generation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [31] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided reconstruction of lifelike clothed humans. In International Conference on 3D Vision (3DV), 2024. 3

- [32] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. In International Conference on Computer Vision (ICCV), 2025. 2, 4, 19, 20
- [33] Zeng Huang, Yuanlu Xu, Christoph Lassner, Hao Li, and Tony Tung. Arch: Animatable reconstruction of clothed humans. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020. 3
- [34] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning (ICML), 2021. 2, 7
- [35] Tianjian Jiang, Xu Chen, Jie Song, and Otmar Hilliges. Instantavatar: Learning avatars from monocular video in 60 seconds. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 3
- [36] Yudong Jin, Sida Peng, Xuan Wang, Tao Xie, Zhen Xu, Yifan Yang, Yujun Shen, Hujun Bao, and Xiaowei Zhou. Diffuman4d: 4d consistent human view synthesis from sparse-view videos with spatio-temporal diffusion models. arXiv preprint arXiv:2507.13344, 2025. 3
- [37] Oren Katzir, Or Patashnik, Daniel Cohen-Or, and Dani Lischinski. Noise-free score distillation. In International Conference on Learning Representations (ICLR), 2024. 4
- [38] Byungjun Kim, Patrick Kwon, Kwangho Lee, Myunggi Lee, Sookwan Han, Daesik Kim, and Hanbyul Joo. Chupa: Carving 3d clothed humans from skinned shape priors using 2d diffusion probabilistic models. In International Conference on Computer Vision (ICCV), 2023. 3
- [39] Xianghao Kong, Qiaosong Qi, Yuanbin Wang, Anyi Rao, Biaolong Chen, Aixi Zhang, Si Liu, and Hao Jiang. Profashion: Prototype-guided fashion video generation with multiple reference images. arXiv preprint arXiv:2505.06537, 2025. 5, 6
- [40] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multiconcept customization of text-to-image diffusion. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 4, 19
- [41] Nupur Kumari, Xi Yin, Jun-Yan Zhu, Ishan Misra, and Samaneh Azadi. Generating multiimage synthetic data for text-to-image customization. arXiv preprint arXiv:2502.01720, 2025. 4, 19
- [42] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. 3, 19
- [43] Boqian Li, Haiwen Feng, Zeyu Cai, Michael J Black, and Yuliang Xiu. Etch: Generalizing body fitting to clothed humans via equivariant tightness. In International Conference on Computer Vision (ICCV), 2025. 4
- [44] Boqian Li, Xuan Li, Ying Jiang, Tianyi Xie, Feng Gao, Huamin Wang, Yin Yang, and Chenfanfu Jiang. Garmentdreamer: 3dgs guided garment synthesis with diverse geometry and texture details. In International Conference on 3D Vision (3DV), 2025. 3
- [45] Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, and Cewu Lu. Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021. 4
- [46] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. In International Conference on Machine Learning (ICML), 2023. 2, 7
- [47] Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wei Xue, Wenhan Luo, et al. Era3d: High-resolution multiview diffusion using efficient row-wise attention. Conference on Neural Information Processing Systems (NeurIPS), 2024. 19

- [48] Peng Li, Wangguandong Zheng, Yuan Liu, Tao Yu, Yangguang Li, Xingqun Qi, Xiaowei Chi, Siyu Xia, Yan-Pei Cao, Wei Xue, et al. Pshuman: Photorealistic single-image 3d human reconstruction using cross-scale multiview diffusion and explicit remeshing. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 1, 2, 3, 6, 7, 10, 19
- [49] Yihui Li, Chengxin Lv, Hongyu Yang, and Di Huang. Micro-macro wavelet-based gaussian splatting for 3d reconstruction from unconstrained images. In AAAI Conference on Artificial Intelligence, 2025. 4
- [50] Tingting Liao, Hongwei Yi, Yuliang Xiu, Jiaxiang Tang, Yangyi Huang, Justus Thies, and Michael J Black. Tada! text to animatable digital avatars. In International Conference on 3D Vision (3DV), 2024. 3
- [51] Tingting Liao, Yujian Zheng, Yuliang Xiu, Adilbek Karmanov, Liwen Hu, Leyang Jin, and Hao Li. SOAP: Style-omniscient animatable portraits. In International Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 2025. 6
- [52] Lixiang Lin, Songyou Peng, Qijun Gan, and Jianke Zhu. Fasthuman: Reconstructing highquality clothed human in minutes. In International Conference on 3D Vision (3DV), 2024. 3
- [53] Xian Liu, Xiaohang Zhan, Jiaxiang Tang, Ying Shan, Gang Zeng, Dahua Lin, Xihui Liu, and Ziwei Liu. Humangaussian: Text-driven 3d human generation with gaussian splatting. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [54] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 6
- [55] Jia Lu, Taoran Yi, Jiemin Fang, Chen Yang, Chuiyun Wu, Wei Shen, Wenyu Liu, Qi Tian, and Xinggang Wang. Snap-snap: Taking two images to reconstruct 3d human gaussians in milliseconds. arXiv preprint arXiv:2508.14892, 2025. 1, 3
- [56] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision (ECCV), 2020. 4
- [57] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal (TMLR), 2024. 5
- [58] Evin Pınar Ornek,¨ Yann Labb´e, Bugra Tekin, Lingni Ma, Cem Keskin, Christian Forster, and Tomas Hodan. Foundpose: Unseen object pose estimation with foundation features. In European Conference on Computer Vision (ECCV), 2024. 9
- [59] Zhuoyang Pan, Angjoo Kanazawa, and Hang Gao. SOAR: Self-occluded avatar recovery from a single video in the wild. arXiv preprint arXiv:2410.23800, 2024. 3
- [60] Hao-Yang Peng, Jia-Peng Zhang, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. Charactergen: Efficient 3d character generation from single images with multi-view pose canonicalization. Transactions on Graphics (TOG), 2024. 2
- [61] Sida Peng, Chen Geng, Yuanqing Zhang, Yinghao Xu, Qianqian Wang, Qing Shuai, Xiaowei Zhou, and Hujun Bao. Implicit neural representations with structured latent codes for human body modeling. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2023. 3
- [62] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations (ICLR), 2023. 2, 4

- [63] Guocheng Qian, Kuan-Chieh Wang, Or Patashnik, Negin Heravi, Daniil Ostashev, Sergey Tulyakov, Daniel Cohen-Or, and Kfir Aberman. Omni-id: Holistic identity representation designed for generative tasks. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 4, 19
- [64] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias Nießner. Gaussianavatars: Photorealistic head avatars with rigged 3d gaussians. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 1
- [65] Lingteng Qiu, Xiaodong Gu, Peihao Li, Qi Zuo, Weichao Shen, Junfei Zhang, Kejie Qiu, Weihao Yuan, Guanying Chen, Zilong Dong, et al. Lhm: Large animatable human reconstruction model from a single image in seconds. In International Conference on Computer Vision (ICCV), 2025. 1, 3, 7
- [66] Lingteng Qiu, Peihao Li, Qi Zuo, Xiaodong Gu, Yuan Dong, Weihao Yuan, Siyu Zhu, Xiaoguang Han, Guanying Chen, and Zilong Dong. Pf-lhm: 3d animatable avatar reconstruction from pose-free articulated human images. arXiv preprint arXiv:2506.13766, 2025. 3
- [67] Lingteng Qiu, Shenhao Zhu, Qi Zuo, Xiaodong Gu, Yuan Dong, Junfei Zhang, Chao Xu, Zhe Li, Weihao Yuan, Liefeng Bo, et al. Anigs: Animatable gaussian avatar from a single image with inconsistent gaussian reconstruction. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [68] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. 9
- [69] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022. 3, 19
- [70] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 3, 4, 19
- [71] Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, and Hao Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. In International Conference on Computer Vision (ICCV), 2019. 3
- [72] Shunsuke Saito, Tomas Simon, Jason Saragih, and Hanbyul Joo. Pifuhd: Multi-level pixelaligned implicit function for high-resolution 3d human digitization. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2020. 3, 10
- [73] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision (ECCV), 2024. 4, 19
- [74] Ruizhi Shao, Youxin Pang, Zerong Zheng, Jingxiang Sun, and Yebin Liu. 360-degree human video generation with 4d diffusion transformer. Transactions on Graphics (TOG), 2024. 7, 20
- [75] Fei Shen and Jinhui Tang. Imagpose: A unified conditional framework for pose-guided person generation. Conference on Neural Information Processing Systems (NeurIPS), 2024. 4, 19
- [76] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Conference on Neural Information Processing Systems (NeurIPS), 2021. 4
- [77] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110, 2023. 19

- [78] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multiview diffusion for 3d generation. In International Conference on Learning Representations (ICLR), 2024. 19
- [79] Yukai Shi, Jianan Wang, Boshi Tang, Xianbiao Qi, Tianyu Yang, Yukun Huang, Shilong Liu, Lei Zhang, Heung-Yeung Shum, et al. Toss: High-quality text-guided novel view synthesis from a single image. In International Conference on Learning Representations (ICLR), 2024. 19
- [80] Noah Snavely, Steven M Seitz, and Richard Szeliski. Photo tourism: Exploring photo collections in 3d. In International Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 2006. 4
- [81] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations (ICLR), 2021. 3, 19
- [82] Stabilityai. Stable-Diffusion-2-1-Base. https://huggingface.co/stabilityai/ stable-diffusion-2-1-base, 2023. 20
- [83] Jiapeng Tang, Davide Davoli, Tobias Kirschstein, Liam Schoneveld, and Matthias Niessner. Gaf: Gaussian avatar reconstruction from monocular videos via multi-view diffusion. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 3
- [84] Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. Realfill: Referencedriven generation for authentic image completion. Transactions on Graphics (TOG), 2024. 2, 4, 19
- [85] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. p+: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522, 2023. 4, 19
- [86] Duotun Wang, Hengyu Meng, Zeyu Cai, Zhijing Shao, Qianxi Liu, Lin Wang, Mingming Fan, Xiaohang Zhan, and Zeyu Wang. Headevolver: Text to head avatars via expressive and attribute-preserving mesh deformation. In International Conference on 3D Vision (3DV), 2025. 3
- [87] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 4, 7
- [88] Wenbo Wang, Hsuan-I Ho, Chen Guo, Boxiang Rong, Artur Grigorev, Jie Song, Juan Jose Zarate, and Otmar Hilliges. 4d-dress: A 4d dataset of real-world human clothing with semantic annotations. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 7, 20
- [89] Yufu Wang, Yu Sun, Priyanka Patel, Kostas Daniilidis, Michael J Black, and Muhammed Kocabas. Prompthmr: Promptable human mesh recovery. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 4, 7, 8
- [90] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira KemelmacherShlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022. 3
- [91] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. Conference on Neural Information Processing Systems (NeurIPS), 2024. 6
- [92] Yuliang Xiu, Jinlong Yang, Dimitrios Tzionas, and Michael J Black. Icon: Implicit clothed humans obtained from normals. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2022. 1, 3, 10
- [93] Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, and Michael J Black. Econ: Explicit clothed humans optimized via normal integration. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023. 1, 3, 6, 10

- [94] Yuliang Xiu, Yufei Ye, Zhen Liu, Dimitrios Tzionas, and Michael J Black. Puzzleavatar: Assembling 3d avatars from personal albums. Transactions on Graphics (TOG), 2024. 2, 3, 4, 7, 20
- [95] Yifang Xu, Benxiang Zhai, Yunzhuo Sun, Ming Li, Yang Li, and Sidan Du. Hifi-portrait: zero-shot identity-preserved portrait generation with high-fidelity multi-face fusion. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2025. 4, 19
- [96] Yuxuan Xue, Xianghui Xie, Riccardo Marin, and Gerard Pons-Moll. Human-3diffusion: Realistic avatar creation via explicit 3d consistent diffusion models. Conference on Neural Information Processing Systems (NeurIPS), 2024. 10
- [97] Xihe Yang, Xingyu Chen, Daiheng Gao, Shaohui Wang, Xiaoguang Han, and Baoyuan Wang. Have-fun: Human avatar reconstruction from few-shot unconstrained images. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [98] Wanqi Yin, Zhongang Cai, Ruisi Wang, Ailing Zeng, Chen Wei, Qingping Sun, Haiyi Mei, Yanjun Wang, Hui En Pang, Mingyuan Zhang, et al. Smplest-x: Ultimate scaling for expressive human pose and shape estimation. arXiv preprint arXiv:2501.09782, 2025. 4
- [99] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2021. 7, 20
- [100] Xin Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Song-Hai Zhang, and Xiaojuan Qi. Textto-3d with classifier score distillation. In International Conference on Learning Representations (ICLR), 2024. 4
- [101] Zhiyuan Yu, Zhe Li, Hujun Bao, Can Yang, and Xiaowei Zhou. Humanram: Feed-forward human reconstruction and animation model using transformers. In International Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 2025. 1, 3
- [102] Yifei Zeng, Yuanxun Lu, Xinya Ji, Yao Yao, Hao Zhu, and Xun Cao. Avatarbooth: Highquality and customizable 3d human avatar generation. arXiv preprint arXiv:2306.09864, 2023. 2, 3, 4, 7
- [103] Yu Zeng, Vishal M Patel, Haochen Wang, Xun Huang, Ting-Chun Wang, Ming-Yu Liu, and Yogesh Balaji. Jedi: Joint-image diffusion models for finetuning-free personalized text-toimage generation. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR),

2024. 4, 19

- [104] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. Pymaf: 3d human pose and shape regression with pyramidal mesh alignment feedback loop. In International Conference on Computer Vision (ICCV), 2021. 4
- [105] Jingbo Zhang, Xiaoyu Li, Qi Zhang, Yanpei Cao, Ying Shan, and Jing Liao. Humanref: Single image to 3d human generation via reference-guided diffusion. In Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2024. 3
- [106] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 2024. 21
- [107] Zerong Zheng, Tao Yu, Yebin Liu, and Qionghai Dai. Pamir: Parametric model-conditioned implicit representation for image-based human reconstruction. Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021. 3
- [108] Xiangyu Zhu, Tingting Liao, Xiaomei Zhang, Jiangjing Lyu, Zhiwen Chen, Yunfeng Wang, Kan Guo, Qiong Cao, Stan Z Li, and Zhen Lei. MVP-Human Dataset for 3-D clothed human avatar reconstruction from multiple frames. IEEE Transactions on Biometrics, Behavior, and Identity Science, 2023. 3
- [109] Zhuofan Zong, Dongzhi Jiang, Bingqi Ma, Guanglu Song, Hao Shao, Dazhong Shen, Yu Liu, and Hongsheng Li. Easyref: Omni-generalized group image reference for diffusion models via multimodal llm. In International Conference on Machine Learning (ICML), 2024. 4, 19

### Appendix

### Table of Contents

- A Use of Large Language Models 19
- B Related Work 19 B.1 Subject-driven and ID-consistent Image Generation . . . . . . . . . . . . . . . 19
- C Priliminary 19
- D Implementation Details 20

- D.1 Model Structure . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.2 Dataset . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.3 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.4 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- D.5 Inference Process . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E Additional Visual Comparisons 22

- E.1 Qualitative Comparisons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.2 Correlation Maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- E.3 Animation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- F Additional Ablation Studies 27

- F.1 Visual Results of Different Orthogonal Images Generation Designs . . . . . . . 27
- F.2 Robustness of Target Pose Condition. . . . . . . . . . . . . . . . . . . . . . . 28
- F.3 Analysis of Shape Predictor. . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- F.4 Visual Results with Different Number of Inputs . . . . . . . . . . . . . . . . . 31

- G More Generation Results of UP2You 32
- H Limitations and Future Works 36

- A USE OF LARGE LANGUAGE MODELS

We used a large language model to assist with copy editing—grammar checking, wording suggestions, and minor style and clarity improvements—after the scientific content, methodology, analyses, and conclusions had been written by the authors.

- B RELATED WORK B.1 SUBJECT-DRIVEN AND ID-CONSISTENT IMAGE GENERATION

With the advent of powerful generative models [25, 42, 69, 81], subject-driven image generation has made remarkable progress in recent years. Various approaches have been proposed to generate images of specific subjects, such as optimizing specialized tokens to encode subject concepts [2, 14, 85], learning personalized modulation vectors for each concept [16], or fine-tuning pre-trained diffusion models [4, 13, 40, 70, 73] using a handful of reference images. Additionally, methods like JeDi [103] and SynCD [41] utilize global self-attention mechanisms to effectively fuse information from multiple images of a target subject, while EasyRef [109] leverages Vision-Language Models (VLMs) [5].

For human-centric generation, several methods have been developed to handle identity preservation. For instance, Omni-ID [63], IMAGPose [75], and HiFi-Portrait [95] utilize specialized image encoders to process multiple reference images for ID-preserving image synthesis. However, extending these techniques to the full body is non-trivial, as the human body’s highly articulated structure and nonrigid deformations introduce significant challenges for feature fusion. To tackle this, approaches like Total Selfie [9], and RealFill [84] employ few-shot personalization via fine-tuning [70] to capture consistent identities, including both facial features and overall appearance. Nevertheless, these methods are tailored for 2D image generation and lack the mechanisms needed to ensure cross-view consistency or the precise latent feature aggregation required for high-fidelity 3D reconstruction.

- C PRILIMINARY

We review the fundamentals of multi-view diffusion models [47, 77–79], with a particular focus on MV-Adapter [32], which serves as the foundation for the multi-view generation of UP2You.

Multi-View Diffusion Models. Multi-view diffusion models extend single-view generation by introducing multi-view attention mechanisms, enabling the synthesis of images that are consistent across different viewpoints. Several works [78, 79] generalize the self-attention mechanism of standard diffusion models to operate over all pixels from multiple views. Specifically, given fin as the input to the attention block, multi-view self-attention concatenates features from M views, allowing the model to capture global dependencies. However, this approach incurs significant computational overhead due to the need to process all pixels across all views. To mitigate this, row-wise selfattention [47, 48] leverages geometric correspondences between orthogonal views. For example, Era3D [47] restricts attention to the current view and corresponding rows from other views, which is well-suited for orthogonal multi-view generation and substantially reduces computational cost.

Building on row-wise self-attention, MV-Adapter [32] introduces an image-to-multiview (I2MV) generator with a parallel attention architecture. The original self-attention block is modified as:

fself = SelfAttn(fin) + MVAttn(fin) + RefAttn(fin,F) + fin, (5)

Here, MVAttn represents the row-wise self-attention mechanism, while RefAttn is a crossattention module that integrates the reference image feature F into fin. The feature F is extracted from the input image I using the reference network R [27]: F = R(I). The I2MV generation process in MV-Adapter is formulated as V = D(F,P(P)), where V = {V1,V2,...,VM} denotes the set of generated multi-view images, D represents the multi-view diffusion model, P = {P1,P2,...,PM} specifies the target viewpoint conditions, and P is the condition encoder that fuses viewpoint conditions into D. In MV-Adapter, only MVAttn, RefAttn, and P are trained for I2MV generation. Each P is encoded as a camera ray representation, referred to as a “raymap”. Typically, M = 6 orthogonal views are generated, corresponding to the target view angles {0°,45°,90°,135°,180°,270°}.

Given the efficient plug-and-play adapter training mechanism of MV-Adapter, combined with the robust feature extraction capabilities of ReferenceNet for processing unconstrained photographs, we adopt MV-Adapter as our multi-view diffusion model architecture. Furthermore, considering our focus on human-centric tasks, we utilize SMPL-X normal rendering as the viewpoint condition P.

- D IMPLEMENTATION DETAILS

- D.1 MODEL STRUCTURE

We adopt the framework architecture of MV-Adapter [32] with the stable-diffusion-2-1-base version [82] as the foundation for both multi-view image and normal generation. The number of selected reference features γ is set to 2.0 during both training and inference phases. We employ the DINOv2-Large [11] variant of the DINOv2 encoder Eref. For the pose image encoder Eref, we implement a lightweight ResNet [22] architecture. The learnable shape tokens τ ∈ R10×1024 are configured to align with the dimensions of Eref, and the perceiver blocks in S comprise 6 layers of cross-attention.

- D.2 DATASET

We train our multi-view image generation, normal map generation, and shape prediction models using the THuman2.1 [99], Human4DiT [74], 2K2K [20], and CustomHumans [24] datasets. Since our task requires handling scenarios where individuals with the same identity appear in different poses, we manually filter the data and group samples by identity. The final training dataset comprises 6,921 scans spanning 2,091 distinct identities. For each scan, we render 6 orthogonal views ({0°,45°,90°,135°,180°,270°}) of both images and normal maps, along with the corresponding SMPL-X normal rendering. Additionally, we render 8 views of each scan using randomly selected perspective cameras to provide “unconstrained photos”. During orthogonal image generation training, for each case, we randomly select 3 to 8 reference images from other cases sharing the same identity.

For evaluation, we select 40 identities from PuzzleIOI [94] and additionally choose “A-pose” configurations from all 68 identities in 4D-Dress [88], while utilizing the remaining poses as reference views. To ensure that SMPL-X camera normal rendering accurately represents viewpoint information, we rotate all scans so that the front view corresponds to zero azimuth. Beyond synthetic data, we also collect an in-the-wild dataset comprising 12 identities for further evaluation, ensuring robust evaluation in diverse scenarios.

- D.3 TRAINING DETAILS

We train the image and normal generation models end-to-end using denoising losses Lrgbd and Lnormald , respectively. During training, Lrgbd jointly optimizes the components Epose, T , Wq, Wk, AvgPool, Prgb, and Drgb. In normal maps generation training, Lnormald optimizes Pnormal and Dnormal. For shape prediction, we employ the loss function Lv = |T(βpred) − T(βgt)| to compute the vertex-wise distance between SMPL-X meshes generated from the predicted shape parameters βpred and the ground-truth shape parameters βgt.

The complete training process for the image and normal generation models requires approximately 3 and 2 days, respectively, on 8 NVIDIA 5880 GPUs. We employ a batch size of 1 per GPU under bfloat16 mixed precision and train for 50,000 iterations. All pose, input, and output image resolutions are consistently set to 768 × 768. The reference images for both image and normal generation are also configured at 768 × 768 resolution, while the target orthogonal view angles follow the same configuration as MV-Adapter. The shape prediction model undergoes training for 100,000 iterations on 8 NVIDIA 5880 GPUs with a batch size of 8 per GPU, requiring approximately 10 hours. We apply a constant learning rate of 5 × 10−5 with warm-up for training all models.

- D.4 EVALUATION METRICS

We employ three complementary metrics to assess geometric accuracy: (1) Chamfer distance (bidirectional point-to-surface distance in cm), which measures overall geometric similarity; (2) P2S

distance (unidirectional point-to-surface distance in cm), which captures reconstruction completeness; and (3) L2 error for Normal maps rendered from four canonical views ({0°,90°,180°,270°}), which evaluates fine-grained surface detail preservation.

We render multi-view color images from the same four canonical viewpoints and evaluate appearance fidelity using three established image quality metrics: PSNR (Peak Signal-to-Noise Ratio) for pixel-level accuracy, SSIM (Structural Similarity) for structural consistency, and LPIPS (Learned Perceptual Image Patch Similarity) for perceptual similarity.

For the in-the-wild dataset, which lacks 3D ground truth, we assess reconstruction quality using perceptual similarity metrics CLIP-I and DINO computed between the generated front view and the captured reference front view image with A-pose.

We further evaluate shape prediction accuracy by computing vertex-to-vertex (V2V) distances between predicted and ground truth SMPL-X meshes under canonical T-pose (zero pose and expression).

|Mesh Reconstruction (Sec 3.2)| |
|---|---|
| | |

Target Textured Mesh

Unconstrained Photos I

|MV Images Generation (Sec 3.1)| |
|---|---|
| | |

|Normal Maps Generation (Sec 3.2)|
|---|

|Shape Prediction (Sec 3.3)|Rendering|
|---|---|
| | |

Generated Normals N

Target Pose

Generated Images V

P

Figure 15: Inference Process of UP2You. Given only unconstrained photos I as inputs, UP2You can generate a high-quality textured mesh.

- D.5 INFERENCE PROCESS

The inference process of UP2You for unconstrained photo inputs I is illustrated in Fig. 15, which mainly consists of four steps as follows:

- (1) Use S to estimate SMPL-X shape parameters βpred from I, and initialize the SMPL-X mesh with βpred and a predefined pose (e.g., A-pose with zero expression) to obtain the pose condition P.
- (2) Generate multi-view images V using Drgb, conditioned on I and P.
- (3) Generate multi-view normal maps N using Dnormal, conditioned on V and P.
- (4) Reconstruct the textured mesh using the initialized SMPL-X mesh, V, and N.

For data pre- and post-processing, we employ [106] to remove backgrounds from input unconstrained photos. Additionally, the reference masks are resized and adapted to the correlation maps C to enhance the model’s focus on foreground regions.

Inference Time. The complete pipeline requires approximately 1.5 minutes to generate a textured mesh from a single unconstrained input. Specifically, the shape prediction step takes about 1 second, multi-view image generation requires approximately 15 seconds, normal map generation takes about 15 seconds, and mesh reconstruction, along with other processing steps (e.g., foreground segmentation, data postprocessing, and file saving), takes nearly 1 minute.

- E ADDITIONAL VISUAL COMPARISONS

- E.1 QUALITATIVE COMPARISONS

We present additional qualitative comparison results in Figs. 16 to 19, including mesh reconstruction, front-view 3D human reconstruction, and shape prediction comparisons. Please zoom in for details.

Inputs Ours AvatarBooth PuzzleAvatar GT

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

…

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

…

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

…

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

…

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

…

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

…

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

…

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

…

###### Figure 16: More Qualitative Comparisons on 4D-Dress and PuzzleIOI datasets.

Inputs Ours AvatarBooth PuzzleAvatar Reference

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

…

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

…

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

…

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

…

###### Figure 17: More Qualitative Comparisons on in-the-wild dataset.

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

Input

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

PSHuman

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

Ours

###### Figure 18: More Qualitative Comparisons of Single Image 3D Human Reconstruction with PSHuman.

[Figure 786]

[Figure 787]

Low High

Ours

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

Semantify

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

PromptHMR

Failed

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

Inputs

###### Figure 19: Error Maps of Shape Prediction.

- E.2 CORRELATION MAPS

Pose-dependent correlation maps generation is an important module of UP2You, as the first part of the proposed PCFA, it predicts the most relevant regions of input unconstrained photos for the conditioned pose. With the latter feature selection strategy, PCFA can focus on informative features for viewpoint generation. In Fig. 20, we provide more results of the generated correlation maps.

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

[Figure 884]

[Figure 885]

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

Figure 20: Visualize Results of Correlation Maps. Given the input reference images and target pose for multi-view image generation, the predicted correlation maps can effectively identify and discriminate correlated regions within the reference inputs. For example, when generating images in the front-view, reference regions that correspond to front-facing views exhibit higher correlation values, demonstrating the model’s ability to selectively attend to relevant spatial information.

- E.3 ANIMATION RESULTS

We present an animation sample generated by UP2You using the same reference with different target poses, as shown in Fig. 21. Notably, UP2You maintains identity consistency well across different target poses. However, since this approach just reconstructs a textured mesh independently for each frame, temporal consistency of the rendered images and mesh topology is not guaranteed. For production-quality animated sequences, we recommend using professional animation methods and tools [1, 6] for textured mesh animation.

Pose A Pose B Pose C Pose D

[Figure 900]

[Figure 901]

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

[Figure 918]

[Figure 919]

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

[Figure 947]

[Figure 948]

[Figure 949]

###### Figure 21: Animation Results of Textured Mesh Generated by UP2You.

- F ADDITIONAL ABLATION STUDIES F.1 VISUAL RESULTS OF DIFFERENT ORTHOGONAL IMAGES GENERATION DESIGNS

Here, we present the generated visual results in Fig. 22 for different design choices in the multi-view image generation model. As indicated earlier, approach A directly concatenates all reference features for viewpoint generation, which may provide irrelevant features during generation and lead to poor results. Approach B averages all reference features as global guidance. This method is time-efficient but loses important color features and generates suboptimal results. Approach C uses a weighted sum strategy to aggregate reference features after computing the correlation map, which loses details in some regions since regions with high correlation values may overlap. Approaches D and E utilize CLIP and DINOv2 features, respectively, rather than ReferenceNet as in our method. CLIP features have low resolution and are difficult to preserve details such as facial and clothing textures, while DINOv2 is texture-insensitive and thus difficult to restore reference textures accurately.

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

[Figure 955]

[Figure 956]

[Figure 957]

Ours

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

[Figure 965]

- A.

- B.

- C.

- D.

- E.

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

[Figure 981]

[Figure 982]

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

GT

Figure 22: Visual Comparisons of Different Multi-View Image Generation Designs.

##### F.2 ROBUSTNESS OF TARGET POSE CONDITION.

While previous experiments highlight the strong generation ability of UP2You, most target poses are in the “A-pose” configuration. Since 4D-Dress provides ground-truth multi-view images of persons with different poses, we further test robustness by randomly selecting three diverse target poses per identity from the 4D-Dress dataset and evaluating our multi-view image generation performance. As shown in Tab. 9, UP2You maintains high-quality results across varied target poses using the same unconstrained photo inputs. Figure 23 further demonstrates the visual results, where identity is consistently preserved across different poses. In addition, Figure 24 shows the generation results on subjects with loose clothing and complex target poses, further validating the generation capability and robustness of UP2You.

| |PSNR↑ SSIM↑ LIPIPS↓<br><br>|
|---|---|
|Pose A<br>Pose B<br>Pose C<br>|24.983 0.911 0.0664 24.400 0.902 0.0744 24.519 0.904 0.0715<br><br>|

Table 9: ID Consistency. UP2You achieves highquality multi-view image generation results in 4D-Dress dataset in three different pose condition.

[Figure 1006]

[Figure 1007]

[Figure 1008]

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

[Figure 1017]

- Pose A

- Pose B

- Pose C

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

[Figure 1033]

[Figure 1034]

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

[Figure 1051]

[Figure 1052]

[Figure 1053]

- Pose A

- Pose B

- Pose C

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

[Figure 1059]

[Figure 1060]

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

[Figure 1077]

Figure 23: Robustness of target pose conditions. Our method can generate high-quality multi-view images under different pose conditions with the same reference inputs, demonstrating that identity information is effectively disentangled from pose conditions in our approach.

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

[Figure 1085]

[Figure 1086]

[Figure 1087]

[Figure 1088]

[Figure 1089]

Unconstrained Inputs

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

Target Pose

[Figure 1096]

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

Generated Mesh

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

Unconstrained Inputs

[Figure 1114]

[Figure 1115]

[Figure 1116]

[Figure 1117]

[Figure 1118]

[Figure 1119]

Target Pose

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

Generated Mesh

###### Figure 24: Generation Results with Loose Clothing and Complex Target Pose.

##### F.3 ANALYSIS OF SHAPE PREDICTOR.

To evaluate whether our shape predictor can regress consistent shape parameters, we assess our shape prediction model using different groups of unconstrained reference inputs from the same identity. As shown in Tab. 10, our method achieves stable shape predictions across all input groups. Since the aggregated pixel-level features from reference inputs may contain information about personal shape characteristics, the multi-view image generation model in UP2You exhibits some degree of robustness to shape variations. However, in extreme cases, more accurate shape predictions can significantly enhance the quality of the final 3D human generation. We evaluate the impact of our shape predictor on the overall inference pipeline of UP2You and find that incorporating the proposed shape predictor leads to measurable improvements in generation quality on the in-the-wild dataset. As demonstrated in Fig. 25, our shape predictor enables more identity-consistent results for individuals with extreme body shapes, while Tab. 11 provides quantitative evidence that the proposed shape predictor improves performance on the in-the-wild dataset.

| |Ref Group A Ref Group B Ref Group C|
|---|---|
|V2V↓ (mm)|7.485 7.503 7.443|

Table 10: Shape prediction consistency on the 4D-Dress dataset. We input three different groups of 12 reference images of the same person into our shape predictor. The vertex-to-vertex (V2V) error of the predicted results shows stable values with low variance, demonstrating that our shape predictor is robust to unconstrained reference inputs.

[Figure 1126]

w/ Shape Predictor w/o Shape Predictor Referernce

[Figure 1127]

[Figure 1128]

| |w/ Shape Predictor<br><br>|w/o Shape Predictor|
|---|---|---|
| |Ours Image Ours Mesh<br><br>|Ours Image Ours Mesh|
|CLIP-I↑ DINO↑<br><br>|0.972 0.971 0.932 0.916|0.969 0.969 0.927 0.911<br><br>|

Table 11: Effects of Shape Predictor on the in-the-wild Dataset. Generation results with the aid of shape predictor have better performance.

- Figure 25: Shape Predictor Helps to Generate More Identity-Consistent Results for People in Extreme Shape.

- F.4 VISUAL RESULTS WITH DIFFERENT NUMBER OF INPUTS

In UP2You, as more unconstrained photos are provided as input, additional details can be extracted and refined in orthogonal views, thereby improving the reliability of the generated results. We demonstrate this principle through an illustrative example in Fig. 26.

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

3 Refs 6 Refs 9 Refs

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

12 Refs GT

[Figure 1147]

[Figure 1148]

[Figure 1149]

[Figure 1150]

- Figure 26: Generated Multi-View Image Results with Different Number of References. With more references input, more results are noticed and generated by our model, like facial details and clothing patterns.

- G MORE GENERATION RESULTS OF UP2YOU

Figures 27 and 28 present comprehensive generation results of UP2You on two representative cases, including the reference images, generated multi-view images and normal maps, as well as the rendered images and normal maps after mesh reconstruction. Figure 29 demonstrates that UP2You is robust to diverse inputs, performing well even in extreme scenarios such as inputs missing the face, lower body, or upper body. Additionally, Figure 30 provides further examples of 3D virtual try-on applications.

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

[Figure 1159]

[Figure 1160]

[Figure 1161]

Inputs

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

Ours Image Generate

[Figure 1168]

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

Ours Normal Generate

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

[Figure 1179]

Ours Image Rendering

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

Ours Normal Rendering

###### Figure 27: Generated Results of UP2You.

[Figure 1186]

[Figure 1187]

[Figure 1188]

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

Inputs

[Figure 1198]

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

Ours Image Generate

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

[Figure 1209]

Ours Normal Generate

[Figure 1210]

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

Ours Image Rendering

[Figure 1216]

[Figure 1217]

[Figure 1218]

[Figure 1219]

[Figure 1220]

[Figure 1221]

Ours Normal Rendering

###### Figure 28: Generated Results of UP2You.

Inputs Results

[Figure 1222]

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

[Figure 1235]

[Figure 1236]

[Figure 1237]

No Face

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

[Figure 1243]

[Figure 1244]

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

No Lower Body

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

[Figure 1277]

[Figure 1278]

[Figure 1279]

[Figure 1280]

[Figure 1281]

No Upper Body

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

Figure 29: More Generation Cases with Invisible Parts. UP2You generates reasonable results with different kinds of invisible scenarios.

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

…

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

…

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

…

[Figure 1310]

[Figure 1311]

[Figure 1312]

[Figure 1313]

…

Figure 30: More examples of 3D Virtual Try-On.

- H LIMITATIONS AND FUTURE WORKS

While our method shows promising results in generating high-quality 3D human avatars from unconstrained photos, there are still some limitations that we plan to address in future work:

- • Dependence on 3D Data for Training: Our method relies on a dataset of 3D human models for training the diffusion model. Acquiring high-quality 3D data can be challenging and may limit the diversity of the generated avatars. In future work, we aim to explore semi-supervised or unsupervised approaches that can leverage large-scale 2D image or video datasets to reduce this dependence on 3D data.
- • Texture Misalignment: Our method generates 6 orthogonal views for mesh reconstruction and texturing, which is insufficient for high-quality texture baking. Texture misalignment issues may arise in some cases (Fig. 31). In future work, we plan to adopt video generation models as the base framework for dense view synthesis to address this limitation.
- • Multiple Inference Stages: When processing in-the-wild photos, our mesh reconstruction pipeline involves four sequential stages: shape prediction, multi-view image generation, multi-view normal map generation, and mesh reconstruction. This multi-stage inference approach slows down the generation process and may introduce cumulative errors. We plan to develop a feed-forward model that directly predicts the final results.

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

Figure 31: Failure Cases of UP2You. Since only 6 orthogonal views {0°, 45°, 90°, 135°, 180°, 270°} are generated, the backside texture of generated humans is lacking in guidance, making the problem of texture misalignment.

